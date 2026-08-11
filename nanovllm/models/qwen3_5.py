"""
Qwen3.5 Hybrid model for nano-vllm.

This implements the Qwen3.5 architecture which uses a hybrid attention stack:
  - 75% Gated DeltaNet (linear attention) layers
  - 25% Gated Full Attention layers
arranged in a repeating 3:1 pattern.

Key differences from Qwen3:
  - Full attention layers have an output gate: output *= sigmoid(gate)
  - Partial RoPE: only partial_rotary_factor (0.25) of head_dim is rotated
  - Linear attention layers use GatedDeltaNet with independent head config
  - Mixed KV cache (full attn) + recurrent state (linear attn) architecture
"""

import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.gated_deltanet import GatedDeltaNet
from nanovllm.utils.context import get_context


def split_q_and_gate(
    q_with_gate: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack Qwen3.5's per-head interleaved query and output gate.

    The checkpoint stores ``[q_head_0, gate_head_0, q_head_1, gate_head_1,
    ...]``.  Splitting the flattened projection in half would instead mix the
    gate of one head into the query of another head.
    """
    expected_width = num_heads * head_dim * 2
    if q_with_gate.size(-1) != expected_width:
        raise ValueError(
            f"expected q_proj width {expected_width}, got {q_with_gate.size(-1)}"
        )
    per_head = q_with_gate.view(-1, num_heads, 2 * head_dim)
    query, gate = per_head.chunk(2, dim=-1)
    return query.flatten(1), gate.flatten(1)


class Qwen3_5RMSNorm(nn.Module):
    """Qwen3.5 RMSNorm uses (1 + weight), unlike the Qwen3 variant."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        input_dtype = x.dtype
        if residual is not None:
            x = x.float() + residual.float()
            residual = x.to(dtype=residual.dtype)
        x = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = (x * (1.0 + self.weight.float())).to(input_dtype)
        return (x, residual) if residual is not None else x


class Qwen3_5Attention(nn.Module):
    """
    Gated Full Attention for Qwen3.5. Used in 25% of layers.
    Differences from Qwen3Attention:
      - q_proj outputs 2x head_dim (half for query, half for output gate)
      - Output is gated: attn_output *= sigmoid(gate)
      - Partial RoPE (only rotary_dim = head_dim * partial_rotary_factor)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        partial_rotary_factor: float = 1.0,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # In Qwen3.5, q_proj outputs 2x for gate; we use a separate gate projection
        # Qwen3.5 stores query and output gate together in q_proj.  Unlike
        # Qwen3, its q_proj therefore has 2 * num_heads * head_dim rows.
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            2 * self.total_num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = ColumnParallelLinear(hidden_size, self.total_num_kv_heads * self.head_dim, bias=False)
        self.v_proj = ColumnParallelLinear(hidden_size, self.total_num_kv_heads * self.head_dim, bias=False)

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        # QK Norm (Qwen3.5 always uses qk_norm without bias)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=rms_norm_eps)

        # Partial RoPE
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position,
            base=rope_theta,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q_with_gate = self.q_proj(hidden_states)
        # The Q and output-gate vectors are interleaved *within each head* in
        # the Qwen3.5 checkpoint, rather than laid out as one contiguous Q
        # half followed by one contiguous gate half.
        q, gate = split_q_and_gate(q_with_gate, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # QK Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Partial RoPE
        q, k = self.rotary_emb(positions, q, k)

        # Flash attention
        o = self.attn(q, k, v)

        # Output gate: sigmoid(gate) * attn_output
        o_flat = o.flatten(1, -1)
        o_flat = o_flat * torch.sigmoid(gate)

        output = self.o_proj(o_flat)
        return output


class Qwen3_5MLP(nn.Module):
    """MLP layer for Qwen3.5 (same as Qwen3MLP)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3_5FullAttentionDecoderLayer(nn.Module):
    """Decoder layer with Gated Full Attention (25% of layers)."""

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        rope_params = getattr(config, "rope_parameters", {})
        rope_theta = rope_params.get("rope_theta", getattr(config, "rope_theta", 1000000))
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)

        self.self_attn = Qwen3_5Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
        )
        self.mlp = Qwen3_5MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5LinearAttentionDecoderLayer(nn.Module):
    """Decoder layer with Gated DeltaNet linear attention (75% of layers)."""

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.linear_attn = GatedDeltaNet(
            hidden_size=config.hidden_size,
            num_key_heads=getattr(config, 'linear_num_key_heads', 16),
            num_value_heads=getattr(config, 'linear_num_value_heads', 32),
            key_head_dim=getattr(config, 'linear_key_head_dim', 128),
            value_head_dim=getattr(config, 'linear_value_head_dim', 128),
            conv_kernel_size=getattr(config, 'linear_conv_kernel_dim', 4),
            rms_norm_eps=config.rms_norm_eps,
        )
        self.mlp = Qwen3_5MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # DeltaNet uses context to determine prefill vs decode
        context = get_context()
        hidden_states = self.linear_attn(hidden_states, is_prefill=context.is_prefill)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5Model(nn.Module):
    """
    Qwen3.5 hybrid model with mixed full_attention and linear_attention layers.
    """

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        # Build layers based on layer_types
        layer_types = getattr(config, "layer_types", None)
        num_layers = config.num_hidden_layers
        layers = []
        for i in range(num_layers):
            if layer_types is not None and layer_types[i] == "linear_attention":
                layers.append(Qwen3_5LinearAttentionDecoderLayer(config))
            else:
                layers.append(Qwen3_5FullAttentionDecoderLayer(config))
        self.layers = nn.ModuleList(layers)
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3_5ForCausalLM(nn.Module):
    """Top-level Qwen3.5 causal LM with hybrid attention."""

    packed_modules_mapping = {
        # MLP projections
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
