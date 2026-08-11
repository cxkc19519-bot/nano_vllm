import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        # Each tensor-parallel worker receives a command event and owns a
        # separate completion event.  The completion acknowledgement keeps
        # the shared-memory command buffer from being overwritten before the
        # worker has finished consuming the previous request.
        self.rpc_channels = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            command_event = ctx.Event()
            completed_event = ctx.Event()
            channel = (command_event, completed_event)
            process = ctx.Process(target=ModelRunner, args=(config, i, channel))
            process.start()
            self.ps.append(process)
            self.rpc_channels.append(channel)
        self.model_runner = ModelRunner(config, 0, self.rpc_channels)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # Public, process-local measurements consumed by the reproducible
        # benchmark.  Compression is asynchronous on CUDA, so `step` records
        # it with explicit synchronization around the actual operation.
        self.compression_stats = {
            "count": 0,
            "total_seconds": 0.0,
            "last_seconds": 0.0,
            "tokens_reclaimed": 0,
            "skipped_no_free_blocks": 0,
        }
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, "model_runner"):
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # Trigger KV cache compression if needed
        for seq in self.scheduler.running:
            if seq.num_physical_kv_tokens >= self.config.compress_threshold:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started_at = perf_counter()
                keep_indices = self.model_runner.call("compute_keep_indices", seq)
                if len(keep_indices) < seq.num_physical_kv_tokens:
                    new_num_tokens = len(keep_indices)
                    num_new_blocks = (new_num_tokens + self.scheduler.block_size - 1) // self.scheduler.block_size
                    # Copy-on-write compaction needs the destination blocks
                    # while the old table is still live.  Defer safely when
                    # the cache is too full instead of corrupting a sequence.
                    if len(self.scheduler.block_manager.free_block_ids) < num_new_blocks:
                        self.compression_stats["skipped_no_free_blocks"] += 1
                        continue
                    new_block_table = [self.scheduler.block_manager._allocate_block() for _ in range(num_new_blocks)]

                    self.model_runner.call("compact_kvcache", seq, keep_indices, new_block_table)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    for block_id in seq.block_table:
                        block = self.scheduler.block_manager.blocks[block_id]
                        block.ref_count -= 1
                        if block.ref_count == 0:
                            self.scheduler.block_manager._deallocate_block(block_id)

                    seq.block_table = new_block_table
                    # Preserve the complete logical request history.  The
                    # compressed cache is only a physical representation used
                    # by Full Attention; DeltaNet state still represents the
                    # full logical prefix.
                    previous_logical_indices = seq.kv_logical_indices
                    seq.kv_logical_indices = [previous_logical_indices[i] for i in keep_indices]
                    seq.num_physical_kv_tokens = new_num_tokens
                    seq.kv_is_compressed = True
                    for block_id in seq.block_table:
                        self.scheduler.block_manager.blocks[block_id].hash = -1
                    elapsed = perf_counter() - started_at
                    self.compression_stats["count"] += 1
                    self.compression_stats["total_seconds"] += elapsed
                    self.compression_stats["last_seconds"] = elapsed
                    self.compression_stats["tokens_reclaimed"] += len(previous_logical_indices) - new_num_tokens

        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        released = [seq for seq in seqs if hasattr(seq, "released_state_slot")]
        if released:
            self.model_runner.call(
                "release_sequence_resources",
                [seq.seq_id for seq in released],
                [seq.released_state_slot for seq in released],
            )
            for seq in released:
                del seq.released_state_slot
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
