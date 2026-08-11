"""One-command Qwen3.5 benchmark runner.

Edit only ``CONFIG`` below, then run this file with the Python interpreter of
the ``nano-vllm`` environment.  On the configured server:

    /home/user/jhk/anaconda/envs/nano-vllm/bin/python app.py

The report is always saved under ``benchmarks/``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# ============================== Change these values ==============================
CONFIG = {
    # Local model folder.  Change this when switching to 2B, 4B, or another model.
    "model": "/home/user/jhk/models/Qwen3.5-9B",
    # Visible GPU indices.  Use "0" for one card or "0,1" for the two-card 9B run.
    "gpus": "0,1",
    "tensor_parallel_size": 2,
    "prompt_tokens": 512,
    "output_tokens": 16,
    "warmup": 1,
    "max_model_len": 1024,
    "max_batched_tokens": 1024,
    "max_seqs": 2,
    # Use a finite number to leave GPU memory headroom; -1 enables auto sizing.
    "num_kvcache_blocks": 64,
    "compress_threshold": 256,
    "sink_tokens": 32,
    "recent_window": 64,
    "recent_queries": 4,
    "top_k": 64,
    # This is only a file name.  Results are kept together in benchmarks/.
    "report_name": "manual_qwen3_5_9b_tp2_4090.json",
}
# ==============================================================================


def main() -> None:
    root = Path(__file__).resolve().parent
    model_path = Path(str(CONFIG["model"])).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model folder does not exist: {model_path}")

    gpu_ids = [gpu.strip() for gpu in str(CONFIG["gpus"]).split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("CONFIG['gpus'] cannot be empty")
    if int(CONFIG["tensor_parallel_size"]) != len(gpu_ids):
        raise ValueError(
            "tensor_parallel_size must equal the number of GPU IDs in CONFIG['gpus']"
        )

    report_name = Path(str(CONFIG["report_name"]))
    if report_name.name != str(report_name) or report_name.suffix != ".json":
        raise ValueError("report_name must be a JSON file name without a directory")
    output_path = Path("benchmarks") / report_name

    command = [
        sys.executable,
        "bench_qwen3_5.py",
        "--model", str(model_path),
        "--tensor-parallel-size", str(CONFIG["tensor_parallel_size"]),
        "--prompt-tokens", str(CONFIG["prompt_tokens"]),
        "--output-tokens", str(CONFIG["output_tokens"]),
        "--warmup", str(CONFIG["warmup"]),
        "--max-model-len", str(CONFIG["max_model_len"]),
        "--max-batched-tokens", str(CONFIG["max_batched_tokens"]),
        "--max-seqs", str(CONFIG["max_seqs"]),
        "--num-kvcache-blocks", str(CONFIG["num_kvcache_blocks"]),
        "--compress-threshold", str(CONFIG["compress_threshold"]),
        "--sink-tokens", str(CONFIG["sink_tokens"]),
        "--recent-window", str(CONFIG["recent_window"]),
        "--recent-queries", str(CONFIG["recent_queries"]),
        "--top-k", str(CONFIG["top_k"]),
        "--output", str(output_path),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    print(f"Model: {model_path}")
    print(f"GPUs: {environment['CUDA_VISIBLE_DEVICES']} (TP={CONFIG['tensor_parallel_size']})")
    print(f"Report: {root / output_path}")
    print("Starting benchmark...\n")
    subprocess.run(command, cwd=root, env=environment, check=True)


if __name__ == "__main__":
    main()
