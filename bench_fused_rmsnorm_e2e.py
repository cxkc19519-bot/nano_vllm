"""Paired end-to-end A/B benchmark for Qwen3.5 fused Add + RMSNorm.

The same engine instance executes the reference and fused variants in an
alternating order.  This avoids model-load differences and reduces bias from
GPU temperature or background load.  Every measured pair uses the same input
and RNG seed, and generated token IDs are checked for exact equality.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

import torch

os.environ.setdefault("TRITON_CACHE_DIR", str(Path.cwd() / ".triton-cache"))

from nanovllm import LLM, SamplingParams


def parse_scenarios(value: str) -> list[dict[str, int]]:
    scenarios = []
    for item in value.split(","):
        fields = item.strip().split(":")
        if len(fields) != 3:
            raise argparse.ArgumentTypeError(
                "scenarios must use prompt_tokens:batch_size:output_tokens"
            )
        prompt_tokens, batch_size, output_tokens = map(int, fields)
        if min(prompt_tokens, batch_size) < 1 or output_tokens < 2:
            raise argparse.ArgumentTypeError("scenario values must be positive; output must be >= 2")
        scenarios.append(
            {
                "prompt_tokens": prompt_tokens,
                "batch_size": batch_size,
                "output_tokens": output_tokens,
            }
        )
    if not scenarios:
        raise argparse.ArgumentTypeError("at least one scenario is required")
    return scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--scenarios",
        type=parse_scenarios,
        default=parse_scenarios("512:1:64,2048:4:64,8192:8:64"),
        help="comma-separated prompt_tokens:batch_size:output_tokens",
    )
    parser.add_argument("--warmup-pairs", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=9216)
    parser.add_argument("--max-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-seqs", type=int, default=8)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/kernels/fused_add_rmsnorm_e2e_4090.json"),
    )
    return parser.parse_args()


def make_prompts(batch_size: int, prompt_tokens: int, vocab_size: int) -> list[list[int]]:
    usable_vocab = max(vocab_size - 256, 1)
    return [
        [256 + ((request_id * prompt_tokens + token_id) % usable_vocab) for token_id in range(prompt_tokens)]
        for request_id in range(batch_size)
    ]


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(mean(values), 4),
        "p50": round(percentile(values, 50), 4),
        "p95": round(percentile(values, 95), 4),
        "stddev": round(pstdev(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def set_fused_enabled(engine: LLM, enabled: bool) -> None:
    # The RPC call is required for tensor parallelism: changing os.environ in
    # rank 0 alone does not update already-spawned worker processes.
    engine.model_runner.call("set_fused_rmsnorm_enabled", enabled)


def run_once(
    engine: LLM,
    prompts: list[list[int]],
    output_tokens: int,
    seed: int,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sampling = SamplingParams(temperature=1e-5, ignore_eos=True, max_tokens=output_tokens)
    tracked = []
    for prompt in prompts:
        engine.add_request(prompt, sampling)
        tracked.append(engine.scheduler.waiting[-1])

    torch.cuda.synchronize()
    started = perf_counter()
    first_token_seconds: dict[int, float] = {}
    decode_step_seconds = []
    while not engine.is_finished():
        torch.cuda.synchronize()
        step_started = perf_counter()
        _, scheduled_tokens = engine.step()
        torch.cuda.synchronize()
        now = perf_counter()
        if scheduled_tokens < 0:
            decode_step_seconds.append(now - step_started)
        for seq in tracked:
            if seq.seq_id not in first_token_seconds and seq.num_completion_tokens:
                first_token_seconds[seq.seq_id] = now - started

    total_seconds = perf_counter() - started
    outputs = [seq.completion_token_ids for seq in tracked]
    decode_tokens_after_first = len(prompts) * (output_tokens - 1)
    decode_seconds = sum(decode_step_seconds)
    return {
        "ttft_ms": 1000 * mean(first_token_seconds.values()),
        "tpot_ms": 1000 * mean(decode_step_seconds),
        "decode_throughput_tokens_per_second": decode_tokens_after_first / decode_seconds,
        "request_throughput_per_second": len(prompts) / total_seconds,
        "total_latency_ms": 1000 * total_seconds,
        "outputs": outputs,
    }


def main() -> None:
    args = parse_args()
    if args.warmup_pairs < 1 or args.repetitions < 2:
        raise ValueError("warmup-pairs must be >= 1 and repetitions must be >= 2")
    largest_batch = max(item["batch_size"] for item in args.scenarios)
    largest_sequence = max(item["prompt_tokens"] + item["output_tokens"] for item in args.scenarios)
    if largest_batch > args.max_seqs:
        raise ValueError("a scenario batch size exceeds --max-seqs")
    if largest_sequence > args.max_model_len:
        raise ValueError("a scenario prompt + output length exceeds --max-model-len")

    engine = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=args.max_seqs,
        num_kvcache_blocks=args.num_kvcache_blocks,
        # Keep compression out of this operator benchmark.
        compress_threshold=args.max_model_len + 1,
    )
    try:
        scenario_reports = []
        all_pairs_equal = True
        for scenario_id, scenario in enumerate(args.scenarios):
            prompts = make_prompts(
                scenario["batch_size"], scenario["prompt_tokens"], engine.tokenizer.vocab_size
            )
            for warmup_id in range(args.warmup_pairs):
                for enabled in (False, True):
                    set_fused_enabled(engine, enabled)
                    run_once(engine, prompts, 2, args.seed + scenario_id * 1000 + warmup_id)

            samples = {"reference": [], "fused": []}
            matching_pairs = 0
            for repetition in range(args.repetitions):
                # Alternate the order to balance thermal and temporal drift.
                order = (False, True) if repetition % 2 == 0 else (True, False)
                paired = {}
                paired_seed = args.seed + scenario_id * 1000 + 100 + repetition
                for enabled in order:
                    set_fused_enabled(engine, enabled)
                    result = run_once(engine, prompts, scenario["output_tokens"], paired_seed)
                    variant = "fused" if enabled else "reference"
                    paired[variant] = result
                    samples[variant].append(result)
                equal = paired["reference"]["outputs"] == paired["fused"]["outputs"]
                matching_pairs += int(equal)
                all_pairs_equal &= equal

            metrics = {}
            metric_names = (
                "ttft_ms",
                "tpot_ms",
                "decode_throughput_tokens_per_second",
                "request_throughput_per_second",
                "total_latency_ms",
            )
            for variant in ("reference", "fused"):
                metrics[variant] = {
                    name: summarize([sample[name] for sample in samples[variant]])
                    for name in metric_names
                }
            deltas = {}
            for name in metric_names:
                reference_mean = metrics["reference"][name]["mean"]
                fused_mean = metrics["fused"][name]["mean"]
                deltas[f"{name}_percent"] = round(
                    100 * (fused_mean - reference_mean) / reference_mean, 3
                )
            scenario_reports.append(
                {
                    "workload": scenario,
                    "measured_pairs": args.repetitions,
                    "measured_requests_per_variant": args.repetitions * scenario["batch_size"],
                    "exact_output_matching_pairs": matching_pairs,
                    "metrics": metrics,
                    "fused_vs_reference_delta": deltas,
                }
            )

        report = {
            "benchmark": "qwen3.5-fused-add-rmsnorm-end-to-end-paired",
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "tensor_parallel_size": args.tensor_parallel_size,
            "seed": args.seed,
            "warmup_pairs_per_scenario": args.warmup_pairs,
            "repetitions_per_variant_per_scenario": args.repetitions,
            "execution_mode": "eager",
            "sampling": {"temperature": 1e-5, "ignore_eos": True},
            "all_output_pairs_exactly_equal": all_pairs_equal,
            "scenarios": scenario_reports,
        }
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    finally:
        set_fused_enabled(engine, True)
        engine.exit()


if __name__ == "__main__":
    main()
