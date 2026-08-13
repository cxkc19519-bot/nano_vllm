"""Microbenchmark PyTorch gather/scatter against fused KV compaction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nanovllm.layers.fused_kv_compaction import _torch_compact_kv_cache, compact_kv_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults match one TP rank of Qwen3.5-9B with TP=2.
    parser.add_argument("--layers", type=int, default=9)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--tokens", default="256,512,1024,2048,4096,8192")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/kernels/fused_kv_compaction_4090.json"),
    )
    return parser.parse_args()


def elapsed_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def temporary_peak_bytes(fn) -> int:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - baseline


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    token_counts = [int(item) for item in args.tokens.split(",") if item.strip()]
    torch.manual_seed(20260813)
    results = []
    for token_count in token_counts:
        kept_tokens = max(args.block_size, token_count // 2)
        old_block_count = (token_count + args.block_size - 1) // args.block_size
        new_block_count = (kept_tokens + args.block_size - 1) // args.block_size
        total_blocks = old_block_count + new_block_count
        shape = (
            args.layers,
            total_blocks,
            args.block_size,
            args.num_kv_heads,
            args.head_dim,
        )
        k_cache = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        v_cache = torch.randn_like(k_cache)
        old_blocks = torch.arange(old_block_count, device="cuda", dtype=torch.int32)
        new_blocks = torch.arange(
            old_block_count, total_blocks, device="cuda", dtype=torch.int32
        )
        keep = torch.linspace(0, token_count - 1, kept_tokens, device="cuda").to(torch.int32)
        reference = lambda: _torch_compact_kv_cache(
            k_cache, v_cache, old_blocks, new_blocks, keep, args.block_size
        )
        fused = lambda: compact_kv_cache(
            k_cache, v_cache, old_blocks, new_blocks, keep, args.block_size
        )

        reference()
        expected_k = k_cache[:, old_block_count:].clone()
        expected_v = v_cache[:, old_block_count:].clone()
        fused()
        torch.cuda.synchronize()
        bit_exact = torch.equal(k_cache[:, old_block_count:], expected_k) and torch.equal(
            v_cache[:, old_block_count:], expected_v
        )
        reference_ms = elapsed_ms(reference, args.warmup, args.iterations)
        fused_ms = elapsed_ms(fused, args.warmup, args.iterations)
        reference_temp = temporary_peak_bytes(reference)
        fused_temp = temporary_peak_bytes(fused)
        copied_bytes = kept_tokens * args.layers * args.num_kv_heads * args.head_dim * 2 * 2
        results.append(
            {
                "source_tokens": token_count,
                "kept_tokens": kept_tokens,
                "copied_mib": round(copied_bytes / 1024**2, 3),
                "pytorch_ms": round(reference_ms, 6),
                "fused_ms": round(fused_ms, 6),
                "speedup": round(reference_ms / fused_ms, 3),
                "pytorch_temporary_mib": round(reference_temp / 1024**2, 3),
                "fused_temporary_mib": round(fused_temp / 1024**2, 3),
                "bit_exact": bit_exact,
            }
        )

    report = {
        "benchmark": "fused-kv-cache-compaction",
        "device": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "layers": args.layers,
        "num_kv_heads_per_rank": args.num_kv_heads,
        "head_dim": args.head_dim,
        "block_size": args.block_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
