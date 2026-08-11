"""Qwen3.5 一键 Benchmark 运行器。

只需要修改下方的 ``CONFIG``，然后使用 ``nano-vllm`` 环境的 Python 运行本
文件。在当前服务器上可执行：

    /home/user/jhk/anaconda/envs/nano-vllm/bin/python app.py

报告会统一保存到 ``benchmarks/`` 目录。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# ============================= 只修改这一段配置 =============================
CONFIG = {
    # 模型所在的本地目录；切换 2B、4B 或其他模型时改这里。
    "model": "/home/user/jhk/models/Qwen3.5-9B",
    # 使用哪些 GPU：单卡写 "0"，双卡写 "0,1"。
    "gpus": "0,1",
    # Tensor Parallel（张量并行）卡数，必须等于上面 GPU 编号的数量。
    "tensor_parallel_size": 2,
    # 每条请求的输入 token 数；数值越大，Prefill 阶段越长。
    "prompt_tokens": 512,
    # 每条请求生成的 token 数；至少为 2，才能计算 TPOT。
    "output_tokens": 16,
    # 正式计时前的预热次数，不计入最终结果。
    "warmup": 1,
    # 引擎允许的最大上下文长度。
    "max_model_len": 1024,
    # 单次调度最多处理的 token 数。
    "max_batched_tokens": 1024,
    # 最大并发请求数；当前 Benchmark 默认只测 1 条请求。
    "max_seqs": 2,
    # KV Cache 的 Block 数。设为固定值可保留显存余量；设为 -1 表示自动估算。
    "num_kvcache_blocks": 64,
    # 物理 KV 长度达到该阈值时，开始尝试 KV 压缩。
    "compress_threshold": 256,
    # 压缩时固定保留开头的 Attention Sink token 数。
    "sink_tokens": 32,
    # 压缩时固定保留末尾滑动窗口的 token 数。
    "recent_window": 64,
    # 用最近多少个 Query 的注意力分数评估中间历史 KV 的重要性。
    "recent_queries": 4,
    # 从中间历史区域额外保留的 Top-K 重要 KV 数。
    "top_k": 64,
    # 只填写文件名；结果会自动保存到 benchmarks/，不要填写目录。
    "report_name": "manual_qwen3_5_9b_tp2_4090.json",
}
# ==============================================================================


def main() -> None:
    # 项目根目录，即 app.py 所在的位置。
    root = Path(__file__).resolve().parent
    model_path = Path(str(CONFIG["model"])).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"找不到模型目录：{model_path}")

    # 将 "0,1" 拆分成 ["0", "1"]，随后传给 CUDA_VISIBLE_DEVICES。
    gpu_ids = [gpu.strip() for gpu in str(CONFIG["gpus"]).split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("CONFIG['gpus'] 不能为空")
    if int(CONFIG["tensor_parallel_size"]) != len(gpu_ids):
        raise ValueError(
            "tensor_parallel_size 必须与 CONFIG['gpus'] 中 GPU 编号的数量一致"
        )

    # 禁止把报告写到 benchmarks 外面，保证所有结果集中管理。
    report_name = Path(str(CONFIG["report_name"]))
    if report_name.name != str(report_name) or report_name.suffix != ".json":
        raise ValueError("report_name 必须是不带目录的 .json 文件名")
    output_path = Path("benchmarks") / report_name

    # app.py 只是把配置转换成 bench_qwen3_5.py 所需的命令行参数。
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
    # 复制当前环境，再指定本次进程可见的 GPU；不会影响其他终端或任务。
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    print(f"模型：{model_path}")
    print(f"GPU：{environment['CUDA_VISIBLE_DEVICES']}（TP={CONFIG['tensor_parallel_size']}）")
    print(f"报告：{root / output_path}")
    print("开始运行 Benchmark...\n")
    subprocess.run(command, cwd=root, env=environment, check=True)


if __name__ == "__main__":
    main()
