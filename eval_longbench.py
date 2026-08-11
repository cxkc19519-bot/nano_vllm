"""LongBench-E subset evaluator for nano-vLLM Qwen3.5 Hybrid inference.

This runner uses the official LongBench v1 dataset IDs, prompt formats, middle
truncation policy, and task-specific scoring rules for a reproducible subset.
It intentionally labels its result as a *subset*: a complete LongBench score
requires all 21 tasks and 4,750 samples, which is not practical as a routine
local regression test.

Example (two RTX 4090):

    CUDA_VISIBLE_DEVICES=0,1 /path/to/python eval_longbench.py \
      --model /path/to/Qwen3.5-9B --tensor-parallel-size 2 --max-samples 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


# Exact prompt formats for the selected LongBench tasks, copied from the
# official THUDM/LongBench config/dataset2prompt.json file.
PROMPTS = {
    "qasper": (
        "You are given a scientific article and a question. Answer the question as "
        "concisely as you can, using a single phrase or sentence if possible. If the "
        "question cannot be answered based on the information in the article, write "
        "\"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", "
        "or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n "
        "Answer the question based on the above article as concisely as you can, using "
        "a single phrase or sentence if possible. If the question cannot be answered "
        "based on the information in the article, write \"unanswerable\". If the "
        "question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
        "Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and "
        "do not output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and "
        "do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine "
        "which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n"
        "{input}\n\nPlease enter the number of the paragraph that the abstract is from. The "
        "answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "
    ),
}

MAX_GENERATION = {"qasper": 128, "hotpotqa": 32, "passage_retrieval_en": 32}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="本地 Qwen3.5 模型目录")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--tasks",
        default="qasper,hotpotqa,passage_retrieval_en",
        help="官方 LongBench-E 子集任务，逗号分隔",
    )
    parser.add_argument("--max-samples", type=int, default=3, help="每个任务均匀抽取的样本数")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="已解压的官方 LongBench data.zip 所在目录；目录内应包含 data/",
    )
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--num-kvcache-blocks", type=int, default=128)
    parser.add_argument("--compress-threshold", type=int, default=2048)
    parser.add_argument("--sink-tokens", type=int, default=64)
    parser.add_argument("--recent-window", type=int, default=512)
    parser.add_argument("--recent-queries", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--modes", default="baseline,compressed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/quality/longbench_e_subset_qwen3_5.json"),
    )
    return parser.parse_args()


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(char for char in text if char not in string.punctuation)
    return " ".join(text.split())


def f1_score(prediction: str, answer: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    if not prediction_tokens or not answer_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(answer_tokens)
    matched = sum(common.values())
    if not matched:
        return 0.0
    precision = matched / len(prediction_tokens)
    recall = matched / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def retrieval_score(prediction: str, answer: str) -> float:
    expected = re.findall(r"Paragraph (\d+)", answer)
    predicted = re.findall(r"\d+", prediction)
    if not expected or not predicted:
        return 0.0
    return sum(number == expected[0] for number in predicted) / len(predicted)


def score(task: str, prediction: str, answers: list[str]) -> float:
    if task == "passage_retrieval_en":
        return max(retrieval_score(prediction, answer) for answer in answers)
    return max(f1_score(prediction, answer) for answer in answers)


def choose_indices(length: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("max-samples 必须大于等于 1")
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})


def middle_truncate(tokenizer, prompt: str, max_tokens: int) -> tuple[str, int, bool]:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    original_length = len(token_ids)
    if original_length <= max_tokens:
        return prompt, original_length, False
    left = max_tokens // 2
    right = max_tokens - left
    truncated = tokenizer.decode(token_ids[:left] + token_ids[-right:], skip_special_tokens=True)
    return truncated, original_length, True


def build_prompt(tokenizer, task: str, item: dict, max_tokens: int) -> tuple[str, int, bool]:
    prompt = PROMPTS[task].format(**item)
    prompt, source_tokens, truncated = middle_truncate(tokenizer, prompt, max_tokens)
    # LongBench's source prompts are model-agnostic.  Qwen3.5 is chat-tuned,
    # so every selected task (including passage retrieval) must enter through
    # its chat template.  Otherwise it starts a reasoning block by default
    # and a short answer budget is exhausted before it emits the answer.
    if getattr(tokenizer, "chat_template", None):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt, source_tokens, truncated


def compression_snapshot(stats: dict) -> dict:
    return {
        "count": int(stats["count"]),
        "total_seconds": float(stats["total_seconds"]),
        "tokens_reclaimed": int(stats["tokens_reclaimed"]),
    }


def load_task(task: str, data_root: Path | None):
    if data_root is not None:
        path = data_root / "data" / f"{task}_e.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"找不到官方 LongBench-E 数据文件：{path}")
        with path.open(encoding="utf-8") as data_file:
            return [json.loads(line) for line in data_file]
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("缺少评测依赖，请执行：pip install -r requirements-eval.txt") from error
    # LongBench v1 stores its data in a dataset loading script, so keep the
    # datasets dependency below v3 and explicitly allow the official script.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return load_dataset("THUDM/LongBench", f"{task}_e", split="test", trust_remote_code=True)


def run_mode(args: argparse.Namespace, tasks: list[str], mode: str) -> list[dict]:
    if mode not in {"baseline", "compressed"}:
        raise ValueError(f"未知模式：{mode}")
    max_model_len = args.max_input_tokens + max(MAX_GENERATION[task] for task in tasks) + 128
    threshold = max_model_len + 1 if mode == "baseline" else args.compress_threshold
    engine = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=max_model_len,
        max_num_batched_tokens=args.max_input_tokens,
        max_num_seqs=1,
        num_kvcache_blocks=args.num_kvcache_blocks,
        compress_threshold=threshold,
        compress_sink_tokens=args.sink_tokens,
        compress_recent_window=args.recent_window,
        compress_recent_queries=args.recent_queries,
        compress_top_k=args.top_k,
    )
    results: list[dict] = []
    try:
        for task in tasks:
            dataset = load_task(task, args.data_root)
            for index in choose_indices(len(dataset), args.max_samples):
                item = dict(dataset[index])
                prompt, source_tokens, truncated = build_prompt(
                    engine.tokenizer, task, item, args.max_input_tokens
                )
                prompt_tokens = engine.tokenizer.encode(prompt, add_special_tokens=False)
                before = compression_snapshot(engine.compression_stats)
                started_at = perf_counter()
                prediction = engine.generate(
                    [prompt_tokens],
                    SamplingParams(temperature=0.01, max_tokens=MAX_GENERATION[task]),
                    use_tqdm=False,
                )[0]["text"].strip()
                elapsed = perf_counter() - started_at
                after = compression_snapshot(engine.compression_stats)
                results.append(
                    {
                        "mode": mode,
                        "task": task,
                        "longbench_e_index": index,
                        "sample_id": item.get("_id"),
                        "source_prompt_tokens": source_tokens,
                        "actual_prompt_tokens": len(prompt_tokens),
                        "middle_truncated": truncated,
                        "prediction": prediction,
                        "answers": item["answers"],
                        "score": score(task, prediction, item["answers"]),
                        "elapsed_seconds": round(elapsed, 4),
                        "compression": {key: after[key] - before[key] for key in before},
                    }
                )
    finally:
        engine.exit()
    return results


def summarize(results: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for mode in sorted({item["mode"] for item in results}):
        summary[mode] = {}
        for task in sorted({item["task"] for item in results if item["mode"] == mode}):
            values = [item["score"] for item in results if item["mode"] == mode and item["task"] == task]
            summary[mode][task] = {
                "samples": len(values),
                "score_percent": round(100 * sum(values) / len(values), 2),
            }
    return summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown_tasks = set(tasks) - set(PROMPTS)
    if unknown_tasks:
        raise ValueError(f"暂不支持的任务：{', '.join(sorted(unknown_tasks))}")
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    results = [item for mode in modes for item in run_mode(args, tasks, mode)]
    report = {
        "benchmark": "LongBench-E official-subset",
        "source": "THUDM/LongBench LongBench-E",
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "tasks": tasks,
        "max_samples_per_task": args.max_samples,
        "data_root": str(args.data_root) if args.data_root else None,
        "max_input_tokens": args.max_input_tokens,
        "summary": summarize(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"报告已写入：{args.output}")


if __name__ == "__main__":
    main()
