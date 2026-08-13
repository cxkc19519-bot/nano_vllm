import torch
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.fused_kv_compaction import compact_kv_cache

class KVCacheCompressorWorker:
    def __init__(self, config: Config, attn_modules: list, kv_cache: torch.Tensor | None = None):
        self.config = config
        self.attn_modules = attn_modules
        self.kv_cache = kv_cache
        self.block_size = config.kvcache_block_size

    def _slots_for_sequence(self, seq: Sequence) -> torch.Tensor:
        """Return physical cache slots in physical-KV order."""
        num_tokens = seq.num_physical_kv_tokens
        if num_tokens == 0:
            return torch.empty(0, dtype=torch.long, device="cuda")
        slots = []
        full_blocks, remainder = divmod(num_tokens, self.block_size)
        for i in range(full_blocks):
            slots.extend(range(seq.block_table[i] * self.block_size, (seq.block_table[i] + 1) * self.block_size))
        if remainder:
            slots.extend(range(seq.block_table[full_blocks] * self.block_size, seq.block_table[full_blocks] * self.block_size + remainder))
        return torch.tensor(slots, dtype=torch.long, device="cuda")

    @torch.inference_mode()
    def compute_keep_indices(self, seq: Sequence) -> list[int]:
        num_tokens = seq.num_physical_kv_tokens
        M = self.config.compress_recent_queries
        sink_len = self.config.compress_sink_tokens
        window_len = self.config.compress_recent_window
        top_k = self.config.compress_top_k

        sink_len = min(sink_len, num_tokens)
        window_len = min(window_len, num_tokens - sink_len)
        middle_start = sink_len
        middle_end = num_tokens - window_len

        if middle_end <= middle_start:
            return list(range(num_tokens))

        global_scores = torch.zeros(middle_end - middle_start, device="cuda", dtype=torch.float32)

        slots_tensor = self._slots_for_sequence(seq)

        valid_layers_scored = 0
        for module in self.attn_modules:
            q_list = module.get_recent_queries(seq.seq_id, M)
            if not q_list:
                continue
            q_recent = torch.stack(q_list, dim=0)

            k_cache_flat = module.k_cache.view(-1, module.num_kv_heads, module.head_dim)
            k_seq = k_cache_flat[slots_tensor]
            k_mid = k_seq[middle_start:middle_end]

            num_heads = q_recent.shape[1]
            num_kv_heads = k_mid.shape[1]
            groups = num_heads // num_kv_heads
            k_mid_expand = k_mid.unsqueeze(2).expand(-1, -1, groups, -1).reshape(-1, num_heads, module.head_dim)

            # Sum across the recent M queries and all heads.  This produces a
            # shared physical-KV importance score across Full Attention layers.
            score = torch.einsum('mhd,lhd->l', q_recent.float(), k_mid_expand.float())
            global_scores += score
            valid_layers_scored += 1

        if valid_layers_scored == 0:
            return list(range(num_tokens))

        actual_top_k = min(top_k, middle_end - middle_start)
        _, top_k_indices = torch.topk(global_scores, actual_top_k)
        top_k_indices = top_k_indices + middle_start

        top_k_indices_sorted, _ = torch.sort(top_k_indices)

        sink_indices = torch.arange(0, sink_len, device="cuda")
        window_indices = torch.arange(middle_end, num_tokens, device="cuda")
        keep_indices = torch.cat([sink_indices, top_k_indices_sorted, window_indices])

        return keep_indices.cpu().tolist()

    @torch.inference_mode()
    def compact_kvcache_memory(self, seq: Sequence, keep_indices: list[int], new_block_table: list[int]):
        if self.kv_cache is None:
            raise RuntimeError("contiguous model KV cache is required for compaction")
        device = self.kv_cache.device
        old_blocks = torch.tensor(seq.block_table, dtype=torch.int32, device=device)
        new_blocks = torch.tensor(new_block_table, dtype=torch.int32, device=device)
        keep = torch.tensor(keep_indices, dtype=torch.int32, device=device)
        compact_kv_cache(
            self.kv_cache[0],
            self.kv_cache[1],
            old_blocks,
            new_blocks,
            keep,
            self.block_size,
        )
