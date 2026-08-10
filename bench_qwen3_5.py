"""Reproducible GPU benchmark for nano-vLLM's Qwen3.5 hybrid path.

Example (Windows / Anaconda)::

    $env:TRITON_CACHE_DIR = "$PWD\\.triton-cache"
    D:\\anaconda\\envs\\nano-vllm\\python.exe bench_qwen3_5.py `
      --model D:\\LLM\\models\\Qwen3.5-0.8B --prompt-tokens 512 `
      --output-tokens 64 --compress-threshold 512 --recent-queries 4

The script intentionally uses deterministic token IDs rather than natural
language prompts.  This makes prompt length, cache allocation and compression
behaviour reproducible across tokenizer versions.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

# On locked-down Windows profiles Triton cannot create its usual cache under
# C:\\Users.  Keep generated kernels inside the repository unless overridden.
os.environ.setdefault("TRITON_CACHE_DIR", str(Path.cwd() / ".triton-cache"))

from nanovllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Qwen3.5 model directory")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up requests before measurement")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-seqs", type=int, default=4)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument("--compress-threshold", type=int, default=512)
    parser.add_argument("--sink-tokens", type=int, default=64)
    parser.add_argument("--recent-window", type=int, default=128)
    parser.add_argument("--recent-queries", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def make_prompts(batch_size: int, prompt_tokens: int, vocab_size: int) -> list[list[int]]:
    # Avoid special IDs at the low end of the vocabulary; each request is
    # different, while the construction itself remains deterministic.
    usable_vocab = max(vocab_size - 256, 1)
    return [
        [256 + ((request_id * prompt_tokens + token_id) % usable_vocab) for token_id in range(prompt_tokens)]
        for request_id in range(batch_size)
    ]


def kv_bytes_per_block(engine: LLM) -> int:
    cfg = engine.config.hf_config
    num_layers = engine.config.num_full_attention_layers
    kv_heads = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads"))
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    dtype_bytes = next(engine.model_runner.model.parameters()).element_size()
    # K and V each own a page in every Full Attention layer.
    return num_layers * 2 * engine.config.kvcache_block_size * kv_heads * head_dim * dtype_bytes


def run_once(engine: LLM, prompts: list[list[int]], max_tokens: int, measure: bool) -> dict:
    sampling = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_tokens)
    for prompt in prompts:
        engine.add_request(prompt, sampling)

    started_at = perf_counter()
    ttft_seconds: float | None = None
    decode_step_seconds: list[float] = []
    decode_tokens = 0
    peak_blocks = 0
    peak_physical_kv_tokens = 0

    while not engine.is_finished():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_started_at = perf_counter()
        _, scheduled_tokens = engine.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = perf_counter() - step_started_at

        block_manager = engine.scheduler.block_manager
        peak_blocks = max(peak_blocks, len(block_manager.used_block_ids))
        active = list(engine.scheduler.running) + list(engine.scheduler.waiting)
        peak_physical_kv_tokens = max(
            peak_physical_kv_tokens,
            sum(seq.num_physical_kv_tokens for seq in active),
        )
        if scheduled_tokens > 0 and ttft_seconds is None:
            # The first prefill call also samples the request's first token.
            ttft_seconds = perf_counter() - started_at
        elif scheduled_tokens < 0:
            decode_step_seconds.append(elapsed)
            decode_tokens += -scheduled_tokens

    return {
        "ttft_seconds": ttft_seconds,
        "decode_step_seconds": decode_step_seconds,
        "decode_tokens": decode_tokens,
        "peak_blocks": peak_blocks,
        "peak_physical_kv_tokens": peak_physical_kv_tokens,
    } if measure else {}


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.prompt_tokens < 1 or args.output_tokens < 2:
        raise ValueError("batch-size and prompt-tokens must be positive; output-tokens must be at least 2")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    engine = LLM(
        args.model,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=args.max_seqs,
        num_kvcache_blocks=args.num_kvcache_blocks,
        compress_threshold=args.compress_threshold,
        compress_sink_tokens=args.sink_tokens,
        compress_recent_window=args.recent_window,
        compress_recent_queries=args.recent_queries,
        compress_top_k=args.top_k,
    )
    try:
        if args.batch_size > args.max_seqs:
            raise ValueError("batch-size cannot exceed max-seqs")
        prompts = make_prompts(args.batch_size, args.prompt_tokens, engine.tokenizer.vocab_size)
        for _ in range(args.warmup):
            run_once(engine, prompts, 2, measure=False)

        # Exclude warm-up compression events from the final report.
        for key in engine.compression_stats:
            engine.compression_stats[key] = 0 if key != "last_seconds" else 0.0
        measurement = run_once(engine, prompts, args.output_tokens, measure=True)
        decode_duration = sum(measurement["decode_step_seconds"])
        bytes_per_block = kv_bytes_per_block(engine)
        report = {
            "benchmark": "nano-vllm-qwen3.5-hybrid",
            "execution_mode": "eager",
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "seed": args.seed,
            "workload": {
                "batch_size": args.batch_size,
                "prompt_tokens_per_request": args.prompt_tokens,
                "output_tokens_per_request": args.output_tokens,
            },
            "metrics": {
                "ttft_ms": round(1000 * measurement["ttft_seconds"], 3),
                # TPOT is the mean wall time between streamed decode tokens for
                # one batch.  With batch_size=1 it is the conventional TPOT.
                "tpot_ms_per_decode_step": round(1000 * mean(measurement["decode_step_seconds"]), 3),
                "decode_throughput_tokens_per_second": round(measurement["decode_tokens"] / decode_duration, 3),
                "decode_tokens_measured": measurement["decode_tokens"],
            },
            "kv_cache": {
                "block_size_tokens": engine.config.kvcache_block_size,
                "total_blocks": engine.config.num_kvcache_blocks,
                "peak_used_blocks": measurement["peak_blocks"],
                "peak_used_block_ratio": round(measurement["peak_blocks"] / engine.config.num_kvcache_blocks, 6),
                "bytes_per_block": bytes_per_block,
                "peak_used_mib": round(measurement["peak_blocks"] * bytes_per_block / 1024**2, 3),
                "peak_physical_kv_tokens": measurement["peak_physical_kv_tokens"],
            },
            "compression": dict(engine.compression_stats),
            "compression_config": {
                "threshold": args.compress_threshold,
                "sink_tokens": args.sink_tokens,
                "recent_window": args.recent_window,
                "recent_queries": args.recent_queries,
                "top_k": args.top_k,
            },
        }
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
