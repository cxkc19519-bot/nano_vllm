# Nano-vLLM（中文文档）

一个从零实现的轻量级 vLLM 风格离线推理引擎。本分支在基础 Qwen3 支持之上，新增了 Qwen3.5 Hybrid 架构的运行时适配与可复现实验工具。

[English](README.md) | [中文](README.zh-CN.md)

## 已实现能力

- Qwen3.5 的混合层调度：区分 Gated DeltaNet（线性注意力）与 Full Attention 执行路径。
- 为每个请求分配、复位和回收 DeltaNet 的 recurrent state、卷积 state slot；抢占后会重建状态，避免与其他请求串扰。
- Full Attention 支持逻辑 token 长度与物理 KV 长度分离：KV 压缩不会删除完整的请求 token 历史。
- KV Cache 压缩：超过阈值时保留 Attention Sink、最近滑动窗口，并从中间历史中依据最近 Query 的注意力分数选取 Top-K KV。
- Chunked Prefill、Decode、抢占重算的状态流转测试。
- `bench_qwen3_5.py`：可重复测量 TTFT、TPOT、Decode Throughput、KV Block 占用和压缩耗时。

## 环境安装

推荐 Linux + NVIDIA GPU + CUDA 环境。以 Conda 环境为例：

```bash
conda activate nano-vllm
pip install -e .
```

在 Linux 环境中，项目依赖原生 `flash-attn` 以获得高性能 Attention 与 CUDA Graph 路径。Windows 上若没有可用的 FlashAttention，运行时会回退到 PyTorch SDPA eager 模式；功能可以验证，但性能数据不应与 FlashAttention 路径直接比较。

## 下载模型

例如下载本仓库双 RTX 4090 测试使用的官方 Qwen3.5-9B：

```bash
hf download Qwen/Qwen3.5-9B \
  --local-dir /path/to/Qwen3.5-9B
```

## 基本推理

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen3.5-9B",
    enforce_eager=True,
    tensor_parallel_size=2,
    max_model_len=2048,
    max_num_seqs=4,
)
outputs = llm.generate(
    ["用一句话介绍北京"],
    SamplingParams(temperature=0.6, max_tokens=64),
)
print(outputs[0]["text"])
llm.exit()
```

## 可复现 Benchmark

以下命令会使用固定随机种子和确定性 token ID 构造请求，输出 JSON 指标报告。报告统一存放在 [`benchmarks/`](benchmarks/)；默认输出路径为 `benchmarks/benchmark_qwen3_5.json`：

### 一键运行

服务器上不需要逐项输入参数。打开 [app.py](app.py)，只修改顶部 `CONFIG` 中的模型路径、GPU 编号、TP 数、输入/输出长度、KV 参数和报告名；然后在项目根目录执行一条命令即可。脚本会使用同一个 Python 解释器运行 Benchmark，并将结果固定写入 `benchmarks/`：

```bash
/home/user/jhk/anaconda/envs/nano-vllm/bin/python app.py
```

### 长上下文质量评测

`eval_needle.py` 会在不同上下文长度和 Needle 插入深度下，对比关闭/开启 KV 压缩的精确检索率。`eval_longbench.py` 使用官方 [LongBench-E](https://github.com/THUDM/LongBench) 数据格式、提示词和 F1/检索评分规则评测代表性子集；它会明确标记为“子集”，不能当作完整 21 任务 LongBench 总分。

```bash
CUDA_VISIBLE_DEVICES=0,1 /home/user/jhk/anaconda/envs/nano-vllm/bin/python eval_needle.py \
  --model /home/user/jhk/models/Qwen3.5-9B --tensor-parallel-size 2

pip install -r requirements-eval.txt
CUDA_VISIBLE_DEVICES=0,1 /home/user/jhk/anaconda/envs/nano-vllm/bin/python eval_longbench.py \
  --model /home/user/jhk/models/Qwen3.5-9B --tensor-parallel-size 2 \
  --data-root /home/user/jhk/datasets/LongBench
```

两类报告都会写入 `benchmarks/quality/`。解释压缩造成的质量变化前，必须先跑无压缩基线；如果基线连简单检索都失败，应先修复数值/生成正确性，不能把分数变化归因于 KV 压缩。

```bash
CUDA_VISIBLE_DEVICES=0,1 python bench_qwen3_5.py \
  --model /path/to/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --prompt-tokens 512 \
  --output-tokens 16 \
  --warmup 1 \
  --max-model-len 1024 \
  --max-batched-tokens 1024 \
  --max-seqs 2 \
  --num-kvcache-blocks 64 \
  --compress-threshold 256 \
  --sink-tokens 32 \
  --recent-window 64 \
  --recent-queries 4 \
  --top-k 64 \
  --output benchmarks/benchmark_qwen3_5_9b_tp2_4090.json
```

报告包含：

- `ttft_ms`：从请求提交到产生第一个 token 的时间。
- `tpot_ms_per_decode_step`：Decode 阶段相邻输出 token 的平均间隔（单请求时即常规 TPOT）。
- `decode_throughput_tokens_per_second`：Decode 阶段总吞吐。
- `kv_cache.peak_used_blocks` / `peak_used_mib`：KV Block 的峰值占用。
- `compression`：压缩次数、总耗时、回收 token 数，以及因没有足够临时 Block 而安全跳过的次数。

### RTX 4090 / Qwen3.5 大模型实测

本仓库仅保留 RTX 4090 的测试记录。其中 `Qwen/Qwen3.5-9B` 使用两张 RTX 4090、TP=2；此前的 2B 和 4B 使用单张 RTX 4090。三组测试均采用 512 输入 token / 16 输出 token 的 eager Benchmark。

| 模型 | 执行方式 | TTFT | TPOT | Decode Throughput | KV Block 峰值 | KV 峰值显存 | 压缩结果 |
|------|----------|------|------|-------------------|---------------|-------------|----------|
| Qwen3.5-2B | 1 x RTX 4090，TP=1 | 192.970 ms | 42.101 ms | 23.753 token/s | 3 | 9.0 MiB | 1 次，20.833 ms，回收 353 token |
| Qwen3.5-4B | 1 x RTX 4090，TP=1 | 237.910 ms | 51.058 ms | 19.586 token/s | 3 | 24.0 MiB | 1 次，19.465 ms，回收 353 token |
| Qwen3.5-9B | 2 x RTX 4090，TP=2 | 361.281 ms | 77.929 ms | 12.832 token/s | 3 / 64 | 24.0 MiB | 1 次，28.069 ms，回收 353 token |

环境为 PyTorch 2.6.0+cu124、FlashAttention 2.7.4、Transformers 5.15.0。原始报告见 [2B 报告](benchmarks/benchmark_qwen3_5_2b_4090.json)、[4B 报告](benchmarks/benchmark_qwen3_5_4b_4090.json) 和 [9B 双卡报告](benchmarks/benchmark_qwen3_5_9b_tp2_4090.json)。共享 GPU 同时有其他任务运行，因此该结果用于功能和可复现性记录，不代表隔离环境下的峰值性能。

### 并发长上下文 KV 压缩对照

以下对照在两张 RTX 4090（TP=2）上运行 Qwen3.5-9B：4 个并发请求，每条 2048 输入 token / 128 输出 token，两组均固定为 128 个 KV Block。

| 版本 | TTFT | TPOT | Decode Throughput | KV Block 峰值 | 物理 KV token 峰值 | 压缩结果 |
|------|------|------|-------------------|---------------|-------------------|----------|
| 无压缩基线 | 2276.400 ms | 134.457 ms | 29.749 token/s | 36 | 8696 | 0 次 |
| 开启 KV 压缩 | 2234.307 ms | 132.768 ms | 30.128 token/s | 36 | 8196 | 4 次，总计 49.939 ms，回收 4868 token |

压缩后物理 KV token 峰值降低 5.75%。由于指标会记录压缩前的首次完整 Prefill，KV Block 峰值仍为 36；因此这组对照不能表述为“KV Block 峰值降低”。原始数据见 [无压缩基线](benchmarks/qwen3_5_9b_tp2_4090_b4_p2048_o128_baseline.json) 和 [压缩版](benchmarks/qwen3_5_9b_tp2_4090_b4_p2048_o128_compressed.json)。服务器为共享环境，该结果用于可复现性记录，不代表隔离环境下的峰值性能。

## 当前边界

项目重点是帮助理解推理引擎的核心机制，不等同于生产级 vLLM。建议在部署前对目标模型完成数值对齐、长上下文、并发请求、抢占与压缩场景验证。
