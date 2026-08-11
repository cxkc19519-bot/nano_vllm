"""Needle-in-a-Haystack quality evaluation for Qwen3.5 Hybrid KV compression.

Example (two RTX 4090):

    CUDA_VISIBLE_DEVICES=0,1 /path/to/python eval_needle.py \
      --model /path/to/Qwen3.5-9B --tensor-parallel-size 2

The runner places a unique retrieval key at several relative depths in a
token-count-controlled context.  It evaluates the same cases with KV
compression disabled and enabled, then writes a JSON report under
``benchmarks/quality/``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


FILLER = (
    "This background paragraph is unrelated to the requested retrieval key. "
    "It repeats neutral facts about an imaginary archive and should be ignored. "
)


def comma_separated_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("必须提供一个或多个正整数")
    return values


def comma_separated_depths(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 or item > 1 for item in values):
        raise argparse.ArgumentTypeError("深度必须位于 0 到 1 之间")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="本地 Qwen3.5 模型目录")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--contexts", type=comma_separated_ints, default=[2048, 4096, 8192])
    parser.add_argument("--depths", type=comma_separated_depths, default=[0.1, 0.5, 0.9])
    # Retrieval keys can be split into more than 16 tokenizer pieces (for
    # example around punctuation and digits).  Reserve enough budget to avoid
    # marking an otherwise correct answer as a failed retrieval merely because
    # it was truncated by the evaluator.
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--num-kvcache-blocks", type=int, default=128)
    parser.add_argument("--compress-threshold", type=int, default=1024)
    parser.add_argument("--sink-tokens", type=int, default=64)
    parser.add_argument("--recent-window", type=int, default=512)
    parser.add_argument("--recent-queries", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--modes",
        default="baseline,compressed",
        help="以逗号分隔的模式：baseline、compressed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/quality/needle_qwen3_5.json"),
    )
    return parser.parse_args()


def repeat_to_length(token_ids: list[int], length: int) -> list[int]:
    if length <= 0:
        return []
    if not token_ids:
        raise ValueError("填充文本未能编码为 token")
    repeats, remainder = divmod(length, len(token_ids))
    return token_ids * repeats + token_ids[:remainder]


def build_case_tokens(tokenizer, context_tokens: int, depth: float, key: str) -> list[int]:
    """Create a token-length-controlled prompt with the needle at ``depth``."""
    prefix = "Read the following archive carefully. Find the retrieval key.\n\n"
    needle = f"\n\nIMPORTANT RETRIEVAL KEY: {key}\n\n"
    question = "\n\nQuestion: What is the retrieval key? Answer with the key only.\nAnswer:"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
    available = context_tokens - len(prefix_ids) - len(needle_ids) - len(question_ids)
    if available < 0:
        raise ValueError("context_tokens 小于指令、needle 和问题本身的长度")
    left_length = int(available * depth)
    right_length = available - left_length
    return (
        prefix_ids
        + repeat_to_length(filler_ids, left_length)
        + needle_ids
        + repeat_to_length(filler_ids, right_length)
        + question_ids
    )


def wrap_as_chat_prompt(tokenizer, prompt_ids: list[int]) -> list[int]:
    """Apply Qwen's chat template after the token-controlled context is built."""
    if not getattr(tokenizer, "chat_template", None):
        return prompt_ids
    prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    messages = [{"role": "user", "content": prompt}]
    try:
        # Qwen reasoning templates otherwise begin generation with <think>,
        # which consumes a short retrieval-answer budget before the key.
        wrapped = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Keep compatibility with tokenizers that predate this Qwen option.
        wrapped = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    # Different Transformers versions return either a flat list, a batched
    # tensor/list, or a BatchEncoding-like mapping.  nano-vLLM accepts only a
    # flat list[int], so normalize the official tokenizer output explicitly.
    if hasattr(wrapped, "get") and "input_ids" in wrapped:
        wrapped = wrapped["input_ids"]
    if hasattr(wrapped, "tolist"):
        wrapped = wrapped.tolist()
    while len(wrapped) == 1 and isinstance(wrapped[0], (list, tuple)):
        wrapped = wrapped[0]
    if not all(isinstance(token_id, int) for token_id in wrapped):
        first_type = type(wrapped[0]).__name__ if wrapped else "empty"
        raise TypeError(
            "chat template 未返回一维 token ID 列表："
            f"container={type(wrapped).__name__}, first={first_type}"
        )
    return wrapped


def snapshot_compression(stats: dict) -> dict:
    return {
        "count": int(stats["count"]),
        "total_seconds": float(stats["total_seconds"]),
        "tokens_reclaimed": int(stats["tokens_reclaimed"]),
        "skipped_no_free_blocks": int(stats["skipped_no_free_blocks"]),
    }


def diff_compression(before: dict, after: dict) -> dict:
    return {key: after[key] - before[key] for key in before}


def run_mode(args: argparse.Namespace, mode: str) -> list[dict]:
    if mode not in {"baseline", "compressed"}:
        raise ValueError(f"未知模式：{mode}")
    threshold = args.max_model_len + args.max_output_tokens + 1 if mode == "baseline" else args.compress_threshold
    engine = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        num_kvcache_blocks=args.num_kvcache_blocks,
        compress_threshold=threshold,
        compress_sink_tokens=args.sink_tokens,
        compress_recent_window=args.recent_window,
        compress_recent_queries=args.recent_queries,
        compress_top_k=args.top_k,
    )
    sampling = SamplingParams(temperature=0.01, max_tokens=args.max_output_tokens)
    results: list[dict] = []
    try:
        for context_tokens in args.contexts:
            if context_tokens + args.max_output_tokens > args.max_model_len:
                raise ValueError("context 长度加输出长度不能超过 max_model_len")
            for depth in args.depths:
                key = f"needle-{context_tokens}-{int(depth * 100):02d}-{args.seed}"
                prompt_ids = build_case_tokens(engine.tokenizer, context_tokens, depth, key)
                prompt_ids = wrap_as_chat_prompt(engine.tokenizer, prompt_ids)
                before = snapshot_compression(engine.compression_stats)
                started_at = perf_counter()
                output = engine.generate([prompt_ids], sampling, use_tqdm=False)[0]["text"].strip()
                elapsed = perf_counter() - started_at
                after = snapshot_compression(engine.compression_stats)
                results.append(
                    {
                        "mode": mode,
                        "target_context_tokens": context_tokens,
                        "actual_prompt_tokens": len(prompt_ids),
                        "needle_depth": depth,
                        "expected_key": key,
                        "prediction": output,
                        "correct": key.lower() in output.lower(),
                        "elapsed_seconds": round(elapsed, 4),
                        "compression": diff_compression(before, after),
                    }
                )
    finally:
        engine.exit()
    return results


def summarize(results: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for mode in sorted({item["mode"] for item in results}):
        mode_results = [item for item in results if item["mode"] == mode]
        correct = sum(item["correct"] for item in mode_results)
        summary[mode] = {
            "cases": len(mode_results),
            "correct": correct,
            "accuracy": round(correct / len(mode_results), 4) if mode_results else None,
        }
    return summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    results = [item for mode in modes for item in run_mode(args, mode)]
    report = {
        "benchmark": "needle-in-a-haystack-qwen3.5-hybrid",
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "contexts": args.contexts,
        "depths": args.depths,
        "compression_config": {
            "threshold": args.compress_threshold,
            "sink_tokens": args.sink_tokens,
            "recent_window": args.recent_window,
            "recent_queries": args.recent_queries,
            "top_k": args.top_k,
        },
        "summary": summarize(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"报告已写入：{args.output}")


if __name__ == "__main__":
    main()
