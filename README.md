<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

[English](README.md) | [中文](README.zh-CN.md)

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

Use `bench_qwen3_5.py` to run a reproducible Qwen3.5 Hybrid benchmark. It measures TTFT, TPOT, Decode Throughput, KV Cache / Block usage, and KV compression time. JSON reports are stored in [`benchmarks/`](benchmarks/); its default output path is `benchmarks/benchmark_qwen3_5.json`.

For the configured 4090 server, edit the `CONFIG` block at the top of [`app.py`](app.py), then run the following single command. It launches the benchmark with that configuration and writes the report to `benchmarks/`.

```bash
/home/user/jhk/anaconda/envs/nano-vllm/bin/python app.py
```

### RTX 4090 / Qwen3.5 Larger-Model Validation

This repository contains only RTX 4090 validation records. The `Qwen/Qwen3.5-9B` run uses tensor parallelism across two RTX 4090 GPUs; the earlier 2B and 4B runs use one RTX 4090.

- Hardware: NVIDIA GeForce RTX 4090; 9B uses 2 GPUs (TP=2)
- Software: PyTorch 2.6.0+cu124, FlashAttention 2.7.4, Transformers 5.15.0
- Input / output: 512 prompt tokens / 16 generated tokens
- KV compression: threshold 256, sink 32, recent window 64, recent queries 4, Top-K 64

| Model | Execution | TTFT (ms) | TPOT (ms) | Decode Throughput (tokens/s) | Peak KV Blocks | Peak KV Memory | Compression |
|-------|-----------|-----------|-----------|------------------------------|----------------|----------------|-------------|
| Qwen3.5-2B | 1 x RTX 4090, TP=1 | 192.970 | 42.101 | 23.753 | 3 | 9.0 MiB | 1 run, 20.833 ms, 353 tokens reclaimed |
| Qwen3.5-4B | 1 x RTX 4090, TP=1 | 237.910 | 51.058 | 19.586 | 3 | 24.0 MiB | 1 run, 19.465 ms, 353 tokens reclaimed |
| Qwen3.5-9B | 2 x RTX 4090, TP=2 | 361.281 | 77.929 | 12.832 | 3 / 64 | 24.0 MiB | 1 run, 28.069 ms, 353 tokens reclaimed |

Reproduce the result with:

```bash
CUDA_VISIBLE_DEVICES=0,1 python bench_qwen3_5.py \
  --model /path/to/Qwen3.5-9B --tensor-parallel-size 2 \
  --prompt-tokens 512 --output-tokens 16 --warmup 1 \
  --max-model-len 1024 --max-batched-tokens 1024 --max-seqs 2 \
  --num-kvcache-blocks 64 \
  --compress-threshold 256 --sink-tokens 32 --recent-window 64 \
  --recent-queries 4 --top-k 64 --output benchmarks/benchmark_qwen3_5_9b_tp2_4090.json
```

Source reports: [`benchmark_qwen3_5_2b_4090.json`](benchmarks/benchmark_qwen3_5_2b_4090.json), [`benchmark_qwen3_5_4b_4090.json`](benchmarks/benchmark_qwen3_5_4b_4090.json), and [`benchmark_qwen3_5_9b_tp2_4090.json`](benchmarks/benchmark_qwen3_5_9b_tp2_4090.json). These were eager-mode functional benchmarks; treat them as reproducibility records, not isolated peak-performance claims.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
