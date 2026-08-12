import pickle
import os
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.attention import FLASH_ATTN_AVAILABLE
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.engine.compressor import KVCacheCompressorWorker

MODEL_REGISTRY = {
    "qwen3": Qwen3ForCausalLM,
    "qwen3_5": Qwen3_5ForCausalLM,
}


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | tuple[Event, Event] | list[tuple[Event, Event]]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        # The PyTorch SDPA compatibility path uses dynamic Python-side cache
        # gathering, which is intentionally eager-only.  CUDA Graph remains
        # available on the FlashAttention path.
        self.enforce_eager = config.enforce_eager or not FLASH_ATTN_AVAILABLE
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        if self.world_size > 1:
            if rank == 0:
                self.command_events = [channel[0] for channel in event]
                self.completed_events = [channel[1] for channel in event]
            else:
                self.command_event, self.completed_event = event

        backend = "nccl" if os.name != "nt" else "gloo"
        dist.init_process_group(backend, "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        model_cls = MODEL_REGISTRY.get(config.model_type, Qwen3ForCausalLM)
        self.model = model_cls(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.linear_attn_modules = [m for m in self.model.modules() if hasattr(m, "allocate_state_pool")]
        self.attn_modules = [m for m in self.model.modules() if hasattr(m, "k_cache")]
        self.max_graph_bs = min(config.max_num_seqs, 512)
        if self.linear_attn_modules:
            # Extra slots are reserved for unused CUDA-graph lanes.  They are
            # never assigned to a user request by Scheduler.
            for module in self.linear_attn_modules:
                module.allocate_state_pool(config.max_num_seqs + self.max_graph_bs)
        self.warmup_model()
        self.allocate_kv_cache()
        self.compressor = KVCacheCompressorWorker(config, self.attn_modules)
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            try:
                self._invoke(method_name, *args)
            finally:
                # This is intentionally a multiprocessing Event rather than
                # an NCCL barrier.  Long requests can have rank-local work
                # with different durations; a barrier made the fast rank hit
                # NCCL's watchdog even though the slow rank was still making
                # progress.
                self.completed_event.set()
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.command_event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.command_event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        for event in self.completed_events:
            event.clear()
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.command_events:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        result = self._invoke(method_name, *args)
        if self.world_size > 1 and self.rank == 0:
            for event in self.completed_events:
                event.wait()
        return result

    def _invoke(self, method_name, *args):
        method = getattr(self, method_name, None)
        if method is None:
            raise AttributeError(f"unknown model-runner method: {method_name}")
        return method(*args)

    def compute_keep_indices(self, seq: Sequence):
        return self.compressor.compute_keep_indices(seq)

    def compact_kvcache(self, seq: Sequence, keep_indices: list[int], new_block_table: list[int]):
        self.compressor.compact_kvcache_memory(seq, keep_indices, new_block_table)

    def release_sequence_resources(self, seq_ids: list[int], state_slots: list[int]):
        for module in self.attn_modules:
            for seq_id in seq_ids:
                module.clear_recent_queries(seq_id)
        for module in self.linear_attn_modules:
            for slot in state_slots:
                module.clear_state_slot(slot)

    def set_fused_rmsnorm_enabled(self, enabled: bool):
        """Apply the benchmark A/B switch consistently on every TP rank."""
        if enabled:
            os.environ.pop("NANOVLLM_DISABLE_FUSED_RMSNORM", None)
        else:
            os.environ["NANOVLLM_DISABLE_FUSED_RMSNORM"] = "1"

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for index, seq in enumerate(seqs):
            seq.num_scheduled_tokens = seq_len
            if self.linear_attn_modules:
                seq.state_slot = index
        self.run(seqs, True)
        for module in self.linear_attn_modules:
            module.recurrent_state_pool.zero_()
            module.conv_state_pool.zero_()
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # For hybrid models, only full_attention layers need KV cache
        num_kv_layers = config.num_full_attention_layers
        block_bytes = 2 * num_kv_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        max_num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        # A caller may deliberately reserve only a small, fixed KV cache for
        # a memory-constrained or tensor-parallel benchmark.  Preserve that
        # explicit cap instead of silently replacing it with the auto-sized
        # value derived from the full device budget.
        if config.num_kvcache_blocks > 0:
            assert config.num_kvcache_blocks <= max_num_kvcache_blocks, (
                f"requested {config.num_kvcache_blocks} KV blocks, but only "
                f"{max_num_kvcache_blocks} fit in the configured GPU budget"
            )
        else:
            config.num_kvcache_blocks = max_num_kvcache_blocks
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, num_kv_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        state_slots = []
        reset_state_slots = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            kv_start = seq.num_physical_kv_tokens
            seqlen_k = kv_start + seqlen_q
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if self.linear_attn_modules:
                assert seq.state_slot is not None
                state_slots.append(seq.state_slot)
                if seq.state_needs_reset:
                    reset_state_slots.append(seq.state_slot)
                    self._clear_recent_queries(seq.seq_id)
                    seq.state_needs_reset = False
            if not seq.block_table:    # warmup
                continue
            start_block = kv_start // self.block_size
            end_block = (kv_start + seqlen_q + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += kv_start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + kv_start + seqlen_q - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        state_slots = self._cuda_int_tensor(state_slots)
        reset_state_slots = self._cuda_int_tensor(reset_state_slots)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, seqs=seqs,
                    state_slot_mapping=state_slots, state_reset_slots=reset_state_slots)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        state_slots = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(seq.num_cached_tokens)
            context_lens.append(seq.num_physical_kv_tokens + 1)
            physical_offset = seq.num_physical_kv_tokens
            slot_mapping.append(seq.block_table[physical_offset // self.block_size] * self.block_size + physical_offset % self.block_size)
            if self.linear_attn_modules:
                assert seq.state_slot is not None
                state_slots.append(seq.state_slot)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, seqs=seqs,
                    state_slot_mapping=self._cuda_int_tensor(state_slots))
        return input_ids, positions

    @staticmethod
    def _cuda_int_tensor(values: list[int]) -> torch.Tensor | None:
        if not values:
            return None
        return torch.tensor(values, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

    def _clear_recent_queries(self, seq_id: int):
        for module in self.attn_modules:
            module.clear_recent_queries(seq_id)

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, seqs: list[Sequence]):
        # Query vectors are needed for the importance scorer.  CUDA graph
        # replay cannot append them to Python-side histories, so use eager for
        # the final M decode steps before a scheduled compression pass.
        collect_compression_queries = (
            bool(self.attn_modules)
            and any(
                seq.num_physical_kv_tokens >= max(0, self.config.compress_threshold - self.config.compress_recent_queries)
                for seq in seqs
            )
        )
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512 or collect_compression_queries:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            if context.state_slot_mapping is not None:
                graph_vars["state_slot_mapping"][:bs] = context.state_slot_mapping
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill, seqs)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = self.max_graph_bs
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # Tail lanes always point at dedicated scratch state slots.  A replay
        # for a smaller live batch can therefore safely execute a larger graph.
        state_slot_mapping = torch.arange(
            self.config.max_num_seqs, self.config.max_num_seqs + max_bs, dtype=torch.int64,
        )
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs],
                        state_slot_mapping=state_slot_mapping[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            state_slot_mapping=state_slot_mapping,
            outputs=outputs,
        )
