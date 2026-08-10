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

例如下载官方 Qwen3.5-0.8B：

```bash
huggingface-cli download Qwen/Qwen3.5-0.8B \
  --local-dir /path/to/Qwen3.5-0.8B
```

## 基本推理

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen3.5-0.8B",
    enforce_eager=True,
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

以下命令会使用固定随机种子和确定性 token ID 构造请求，输出 JSON 指标报告：

```bash
python bench_qwen3_5.py \
  --model /path/to/Qwen3.5-0.8B \
  --prompt-tokens 512 \
  --output-tokens 64 \
  --compress-threshold 512 \
  --sink-tokens 64 \
  --recent-window 128 \
  --recent-queries 4 \
  --top-k 128 \
  --output benchmark_qwen3_5.json
```

报告包含：

- `ttft_ms`：从请求提交到产生第一个 token 的时间。
- `tpot_ms_per_decode_step`：Decode 阶段相邻输出 token 的平均间隔（单请求时即常规 TPOT）。
- `decode_throughput_tokens_per_second`：Decode 阶段总吞吐。
- `kv_cache.peak_used_blocks` / `peak_used_mib`：KV Block 的峰值占用。
- `compression`：压缩次数、总耗时、回收 token 数，以及因没有足够临时 Block 而安全跳过的次数。

仓库内的 [benchmark_qwen3_5_3060.json](benchmark_qwen3_5_3060.json) 是 RTX 3060 12GB 上的一次参考运行结果；实际性能会随 GPU、CUDA、FlashAttention、batch size 与模型版本变化。

## 当前边界

项目重点是帮助理解推理引擎的核心机制，不等同于生产级 vLLM。建议在部署前对目标模型完成数值对齐、长上下文、并发请求、抢占与压缩场景验证。
