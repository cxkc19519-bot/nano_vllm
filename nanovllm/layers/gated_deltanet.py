"""
Gated DeltaNet linear attention layer for Qwen3.5 Hybrid models.

This implements the Gated DeltaNet mechanism that replaces standard self-attention
in 75% of the layers. It maintains a fixed-size recurrent state instead of a growing
KV cache, providing O(1) memory and O(L) compute per token.

Two execution paths are provided:
  - Chunked (prefill): Processes sequences in chunks for parallelism.
  - Recurrent (decode): Token-by-token state update for autoregressive generation.
"""

import torch
from torch import nn
import torch.nn.functional as F


class RMSNormGated(nn.Module):
    """RMSNorm with an output gating mechanism: norm(x) * weight * silu(gate)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.float())
        return hidden_states.to(input_dtype)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize along a given dimension."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
) -> torch.Tensor:
    """Pure-PyTorch causal 1D convolution. Input shape: (B, D, L)."""
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    return out.to(hidden_states.dtype)


def causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
) -> torch.Tensor:
    """Update conv state and compute output for decode (single-token). Input: (B, D, 1)."""
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    return out.to(hidden_states.dtype)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Chunked gated delta rule for prefill.
    All inputs: (B, num_heads, L, dim) except g/beta: (B, num_heads, L).
    Returns: (B, L, num_heads, v_dim), optional final_state.
    """
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # Chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # Process each chunk
    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Recurrent gated delta rule for decode (token-by-token).
    All inputs: (B, num_heads, L, dim) except g/beta: (B, num_heads, L).
    Returns: (B, L, num_heads, v_dim), optional final_state.
    """
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet linear attention layer for Qwen3.5 Hybrid.

    Maintains a fixed-size recurrent state (num_heads, key_dim, value_dim)
    instead of a growing KV cache, providing O(1) memory per layer.

    Attributes:
        recurrent_state: The persistent hidden state updated during inference.
        conv_state: The causal conv1d state for decode.
    """

    def __init__(
        self,
        hidden_size: int,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.key_dim = num_key_heads * key_head_dim
        self.value_dim = num_value_heads * value_head_dim
        self.conv_kernel_size = conv_kernel_size

        # Input projections
        # qkv: key + key (for query reuse in some variants) + value
        # In Qwen3.5: q shares key heads config, so in_proj_qkv projects to key_dim*2 + value_dim
        self.in_proj_qkv = nn.Linear(hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)       # output gate
        self.in_proj_b = nn.Linear(hidden_size, self.num_value_heads, bias=False)  # beta (learning rate)
        self.in_proj_a = nn.Linear(hidden_size, self.num_value_heads, bias=False)  # decay gate

        # Causal 1D convolution on QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=conv_kernel_size,
            groups=self.conv_dim,
            padding=conv_kernel_size - 1,
        )

        # Time step discretization
        self.dt_bias = nn.Parameter(torch.ones(self.num_value_heads))
        A = torch.empty(self.num_value_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # Gated RMSNorm
        self.norm = RMSNormGated(self.value_head_dim, eps=rms_norm_eps)

        # Output projection
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # State is pooled by request slot.  These buffers are deliberately not
        # checkpoint state: they are runtime cache, just like Attention.k_cache.
        self.register_buffer("recurrent_state_pool", torch.empty(0), persistent=False)
        self.register_buffer("conv_state_pool", torch.empty(0), persistent=False)

    def allocate_state_pool(self, num_slots: int):
        """Allocate one recurrent and convolution state per active request."""
        if self.num_value_heads % self.num_key_heads != 0:
            raise ValueError("linear_num_value_heads must be divisible by linear_num_key_heads")
        dtype = self.in_proj_qkv.weight.dtype
        device = self.in_proj_qkv.weight.device
        self.recurrent_state_pool = torch.zeros(
            num_slots, self.num_value_heads, self.key_head_dim, self.value_head_dim,
            dtype=dtype, device=device,
        )
        self.conv_state_pool = torch.zeros(
            num_slots, self.conv_dim, self.conv_kernel_size - 1,
            dtype=dtype, device=device,
        )

    def reset_state_slots(self, slots: torch.Tensor):
        if slots is None or slots.numel() == 0 or self.recurrent_state_pool.numel() == 0:
            return
        slots = torch.unique(slots.to(device=self.recurrent_state_pool.device, dtype=torch.long))
        self.recurrent_state_pool.index_fill_(0, slots, 0)
        self.conv_state_pool.index_fill_(0, slots, 0)

    def clear_state_slot(self, slot: int):
        if self.recurrent_state_pool.numel():
            self.reset_state_slots(torch.tensor([slot], device=self.recurrent_state_pool.device))

    def _conv_with_state(self, qkv: torch.Tensor, conv_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal convolution for a complete sequence, including its cached tail."""
        qkv_channels = qkv.transpose(0, 1).unsqueeze(0)
        if conv_state.shape[-1]:
            conv_input = torch.cat([conv_state, qkv_channels], dim=-1)
            output = F.conv1d(conv_input.to(self.conv1d.weight.dtype), self.conv1d.weight,
                              bias=self.conv1d.bias, padding=0, groups=self.conv_dim)
            next_state = conv_input[:, :, -(self.conv_kernel_size - 1):].to(conv_state.dtype)
        else:
            output = qkv_channels
            next_state = conv_state
        return F.silu(output.squeeze(0).transpose(0, 1)).to(qkv.dtype), next_state

    def _run_sequence(self, hidden_states: torch.Tensor, slot: torch.Tensor, use_chunked_rule: bool, reset_state: bool = False) -> torch.Tensor:
        """Run one logical sequence and scatter its final state back to its slot."""
        slot = slot.reshape(1).to(dtype=torch.long, device=hidden_states.device)
        recurrent_state = None if reset_state else self.recurrent_state_pool.index_select(0, slot)
        conv_state = self.conv_state_pool.index_select(0, slot)

        qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        beta = torch.sigmoid(self.in_proj_b(hidden_states))
        A = self.in_proj_a(hidden_states)
        qkv, next_conv_state = self._conv_with_state(qkv, conv_state)

        # The checkpoint packs this projection in Q, K, V order, which is also
        # the input order expected by the gated delta rule.
        query, key, value = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        key = key.view(-1, self.num_key_heads, self.key_head_dim)
        query = query.view(-1, self.num_key_heads, self.key_head_dim)
        value = value.view(-1, self.num_value_heads, self.value_head_dim)
        if self.num_key_heads != self.num_value_heads:
            repeats = self.num_value_heads // self.num_key_heads
            key = key.repeat_interleave(repeats, dim=1)
            query = query.repeat_interleave(repeats, dim=1)

        g = -self.A_log.float().exp() * F.softplus(A.float() + self.dt_bias)
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)

        delta_rule = torch_chunk_gated_delta_rule if use_chunked_rule else torch_recurrent_gated_delta_rule
        output, next_recurrent_state = delta_rule(
            query, key, value, g, beta, initial_state=recurrent_state, output_final_state=True,
        )
        self.recurrent_state_pool.index_copy_(0, slot, next_recurrent_state.to(self.recurrent_state_pool))
        self.conv_state_pool.index_copy_(0, slot, next_conv_state.to(self.conv_state_pool))

        output = output.squeeze(0)
        output = self.norm(output, z.view(-1, self.num_value_heads, self.value_head_dim))
        return self.out_proj(output.flatten(-2, -1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (total_tokens, hidden_size) for prefill or (batch, hidden_size) for decode.
            is_prefill: Whether in prefill or decode mode.
        """
        from nanovllm.utils.context import get_context

        context = get_context()
        if context.state_reset_slots is not None:
            self.reset_state_slots(context.state_reset_slots)
        slots = context.state_slot_mapping
        if slots is None or self.recurrent_state_pool.numel() == 0:
            raise RuntimeError("GatedDeltaNet requires request state slots allocated by ModelRunner")

        if is_prefill:
            if context.seqs is None or slots.numel() == 0:
                raise RuntimeError("prefill requires sequence boundaries and state slots")
            outputs = []
            start = 0
            reset_state = context.state_reset_slots is not None
            for index in range(slots.numel()):
                end = start + context.seqs[index].num_scheduled_tokens
                outputs.append(self._run_sequence(hidden_states[start:end], slots[index:index + 1], True, reset_state))
                start = end
            return torch.cat(outputs, dim=0)

        # Decode has one input token per request.  Keeping the request loop
        # makes state ownership explicit and remains CUDA-graph capturable for
        # each fixed batch bucket.
        return torch.cat([
            self._run_sequence(hidden_states[index:index + 1], slots[index:index + 1], False)
            for index in range(slots.numel())
        ], dim=0)
