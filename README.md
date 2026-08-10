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

Use `bench_qwen3_5.py` to run a reproducible Qwen3.5 Hybrid benchmark. It measures TTFT, TPOT, Decode Throughput, KV Cache / Block usage, and KV compression time.

**Reference Configuration (local machine):**
- Hardware: NVIDIA GeForce RTX 3060 (12GB)
- Model: Qwen/Qwen3.5-0.8B
- Input / output: 512 prompt tokens / 16 generated tokens
- Execution mode: eager PyTorch SDPA fallback on Windows (no native FlashAttention)
- KV compression: threshold 256, sink 32, recent window 64, recent queries 4, Top-K 64

**Measured Result:**
| TTFT (ms) | TPOT (ms) | Decode Throughput (tokens/s) | Peak KV Blocks | Peak KV Memory | Compression |
|-----------|-----------|------------------------------|----------------|----------------|-------------|
| 183.910   | 44.564    | 22.440                       | 3              | 9.0 MiB        | 1 run, 32.712 ms, 353 tokens reclaimed |

Reproduce the result with:

```bash
python bench_qwen3_5.py --model /path/to/Qwen3.5-0.8B \
  --prompt-tokens 512 --output-tokens 16 --warmup 1 \
  --max-model-len 1024 --max-batched-tokens 1024 --max-seqs 2 \
  --compress-threshold 256 --sink-tokens 32 --recent-window 64 \
  --recent-queries 4 --top-k 64 --output benchmark_qwen3_5.json
```

The full reference report is available in [`benchmark_qwen3_5_3060.json`](benchmark_qwen3_5_3060.json). Performance varies by GPU, driver, CUDA, FlashAttention availability and workload.

### RTX 4090: Larger-Model Validation

This branch also includes a real server-side validation with the larger `Qwen/Qwen3.5-2B` model (2.5x the parameter count of the local 0.8B reference):

- Hardware: NVIDIA GeForce RTX 4090 (GPU 1 on a shared server)
- Software: PyTorch 2.6.0+cu124, FlashAttention 2.7.4, Transformers 5.15.0
- Input / output: 512 prompt tokens / 16 generated tokens
- KV compression: threshold 256, sink 32, recent window 64, recent queries 4, Top-K 64

| TTFT (ms) | TPOT (ms) | Decode Throughput (tokens/s) | Peak KV Blocks | Peak KV Memory | Compression |
|-----------|-----------|------------------------------|----------------|----------------|-------------|
| 192.970   | 42.101    | 23.753                       | 3              | 9.0 MiB        | 1 run, 20.833 ms, 353 tokens reclaimed |

The source report is [`benchmark_qwen3_5_2b_4090.json`](benchmark_qwen3_5_2b_4090.json). This was an eager-mode functional benchmark run while the shared GPUs had other workloads; treat it as a reproducibility record, not an isolated peak-performance claim.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
