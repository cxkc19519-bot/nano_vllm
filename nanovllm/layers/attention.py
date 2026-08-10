import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

    def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, causal: bool = False) -> torch.Tensor:
        # SDPA expects (batch, heads, sequence, dim).  GQA covers Qwen's
        # grouped-query attention when there are fewer KV heads than Q heads.
        return F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            dropout_p=0.0,
            is_causal=causal,
            scale=scale,
            enable_gqa=q.size(1) != k.size(1),
        ).squeeze(0).transpose(0, 1)

    def flash_attn_varlen_func(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, cu_seqlens_k, softmax_scale, causal, block_table=None):
        outputs = []
        for index in range(cu_seqlens_q.numel() - 1):
            q_start, q_end = int(cu_seqlens_q[index]), int(cu_seqlens_q[index + 1])
            k_start, k_end = int(cu_seqlens_k[index]), int(cu_seqlens_k[index + 1])
            if block_table is not None:
                blocks = block_table[index]
                valid_blocks = blocks[blocks >= 0]
                cache_k = k[valid_blocks].reshape(-1, k.size(-2), k.size(-1))[:k_end - k_start]
                cache_v = v[valid_blocks].reshape(-1, v.size(-2), v.size(-1))[:k_end - k_start]
                outputs.append(_sdpa(q[q_start:q_end], cache_k, cache_v, softmax_scale, causal))
            else:
                outputs.append(_sdpa(q[q_start:q_end], k[k_start:k_end], v[k_start:k_end], softmax_scale, causal))
        return torch.cat(outputs, dim=0)

    def flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale, causal):
        outputs = []
        for index in range(q.size(0)):
            cache_len = int(cache_seqlens[index])
            blocks = block_table[index]
            valid_blocks = blocks[blocks >= 0]
            k = k_cache[valid_blocks].reshape(-1, k_cache.size(-2), k_cache.size(-1))[:cache_len]
            v = v_cache[valid_blocks].reshape(-1, v_cache.size(-2), v_cache.size(-1))[:cache_len]
            outputs.append(_sdpa(q[index, 0:1], k, v, softmax_scale, False))
        return torch.cat(outputs, dim=0)
from nanovllm.utils.context import get_context
from collections import deque


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        # Kept on the attention module rather than Sequence so state remains
        # local to each model-runner process and is never serialized to TP peers.
        self._recent_queries: dict[int, deque] = {}

    def get_recent_queries(self, seq_id: int, limit: int) -> list[torch.Tensor]:
        history = self._recent_queries.get(seq_id)
        return [] if history is None else list(history)[-limit:]

    def clear_recent_queries(self, seq_id: int):
        self._recent_queries.pop(seq_id, None)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            if context.seqs is not None:
                for i, seq in enumerate(context.seqs):
                    history = self._recent_queries.setdefault(seq.seq_id, deque(maxlen=128))
                    history.append(q[i].detach().clone())
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
