"""Microbenchmark for Qwen3.5 fused Add + RMSNorm on CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nanovllm.layers.fused_add_rmsnorm import _torch_qwen3_5_rmsnorm, qwen3_5_add_rmsnorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--rows", default="1,4,16,128,512,2048,8192")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/kernels/fused_add_rmsnorm_4090.json"),
    )
    return parser.parse_args()


def elapsed_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    for _ in range(iterations):
        fn()
    finished.record()
    finished.synchronize()
    return started.elapsed_time(finished) / iterations


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    rows = [int(value) for value in args.rows.split(",") if value.strip()]
    torch.manual_seed(20260813)
    results = []
    for row_count in rows:
        x = torch.randn(row_count, args.hidden_size, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        weight = torch.randn(args.hidden_size, device="cuda", dtype=torch.bfloat16)
        reference = lambda: _torch_qwen3_5_rmsnorm(x, weight, 1e-6, residual)
        fused = lambda: qwen3_5_add_rmsnorm(x, weight, 1e-6, residual)
        expected_output, expected_residual = reference()
        actual_output, actual_residual = fused()
        max_abs_error = max(
            (actual_output.float() - expected_output.float()).abs().max().item(),
            (actual_residual.float() - expected_residual.float()).abs().max().item(),
        )
        reference_ms = elapsed_ms(reference, args.warmup, args.iterations)
        fused_ms = elapsed_ms(fused, args.warmup, args.iterations)
        results.append(
            {
                "rows": row_count,
                "hidden_size": args.hidden_size,
                "reference_ms": round(reference_ms, 6),
                "fused_ms": round(fused_ms, 6),
                "speedup": round(reference_ms / fused_ms, 3),
                "max_abs_error": max_abs_error,
            }
        )
    report = {
        "benchmark": "qwen3.5-fused-add-rmsnorm",
        "device": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
