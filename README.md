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

### Long-Context Quality Evaluation

`eval_needle.py` compares exact retrieval accuracy with KV compression disabled and enabled across configurable context lengths and insertion depths. `eval_longbench.py` evaluates an explicitly labeled [LongBench-E](https://github.com/THUDM/LongBench) subset using its official prompt formats and F1/retrieval scoring; it is not a full 21-task LongBench score.

```bash
CUDA_VISIBLE_DEVICES=0,1 /home/user/jhk/anaconda/envs/nano-vllm/bin/python eval_needle.py \
  --model /home/user/jhk/models/Qwen3.5-9B --tensor-parallel-size 2

pip install -r requirements-eval.txt
CUDA_VISIBLE_DEVICES=0,1 /home/user/jhk/anaconda/envs/nano-vllm/bin/python eval_longbench.py \
  --model /home/user/jhk/models/Qwen3.5-9B --tensor-parallel-size 2 \
  --data-root /home/user/jhk/datasets/LongBench
```

Both scripts write JSON reports under `benchmarks/quality/`. Run a no-compression baseline before interpreting a quality delta. If the baseline fails simple retrieval, fix numerical/generation correctness before attributing a score change to KV compression.

#### 2 x RTX 4090 Long-Context Results

Qwen3.5-9B was evaluated with TP=2 and an 8,192-token Chunked Prefill budget. The Needle run covers four context lengths and five insertion depths, for 20 unique cases and 40 baseline/compressed generations. Every case remained correct after compression.

| Context | Baseline | Compressed | Physical KV tokens reclaimed | Mean compression time |
|---------|----------|------------|------------------------------|-----------------------|
| 4k | 5 / 5 | 5 / 5 | 59.49% | 9.48 ms |
| 8k | 5 / 5 | 5 / 5 | 79.71% | 5.93 ms |
| 16k | 5 / 5 | 5 / 5 | 89.84% | 16.30 ms |
| 32k | 5 / 5 | 5 / 5 | 94.92% | 27.20 ms |

The LongBench-E representative subset contains 15 unique samples from Qasper, HotpotQA, and Passage Retrieval, each evaluated with compression disabled and enabled (30 generations total).

| Task | Baseline score | Compressed score | Delta | Physical KV tokens reclaimed | Mean compression time |
|------|----------------|------------------|-------|------------------------------|-----------------------|
| Qasper | 56.19 | 55.50 | -0.69 pt | 78.53% | 16.06 ms |
| HotpotQA | 42.67 | 42.67 | 0.00 pt | 84.36% | 9.07 ms |
| Passage Retrieval | 100.00 | 100.00 | 0.00 pt | 80.50% | 6.72 ms |
| Overall | 66.29 | 66.06 | -0.23 pt | 81.46% | 10.62 ms |

These percentages describe reclaimed physical KV tokens after compaction, not peak allocated GPU memory. The LongBench result is a reproducible three-task subset rather than the full 21-task benchmark. Source reports: [Needle 4k–32k](benchmarks/quality/needle_9b_tp2_scale_4k_32k_chunked_prefill_b160.json) and [LongBench-E subset](benchmarks/quality/longbench_e_9b_tp2_scale_3tasks_5samples_chunked.json).

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

### Triton Fused Add + RMSNorm

The Qwen3.5 path includes a custom Triton kernel that fuses residual addition, FP32 RMS reduction, normalization, and Qwen3.5's `(1 + weight)` scaling. The same implementation also specializes the no-residual RMSNorm path used by Q/K normalization, while unsupported devices or layouts fall back to the aligned PyTorch formula.

The following operator-level results were measured on one RTX 4090 with BF16 activations, hidden size 4096, 30 warm-up iterations, and 200 timed iterations:

| Rows | PyTorch reference | Triton fused | Speedup |
|-----:|------------------:|-------------:|--------:|
| 1 | 0.0970 ms | 0.0437 ms | 2.22x |
| 128 | 0.0986 ms | 0.0440 ms | 2.24x |
| 2,048 | 0.3195 ms | 0.0466 ms | 6.86x |
| 8,192 | 2.3197 ms | 0.2924 ms | 7.93x |

The maximum BF16 absolute error across the benchmark was 0.03125. A Qwen3.5-9B / TP=2 Needle A/B test produced the exact same generated text with the fused kernel disabled and enabled. These speedups are kernel-level measurements and must not be interpreted as end-to-end model speedups. Reproduce them with `bench_fused_add_rmsnorm.py`; see the [raw report](benchmarks/kernels/fused_add_rmsnorm_4090.json).

To replace the earlier single-request observation, a paired end-to-end A/B benchmark alternates both variants inside the same Qwen3.5-9B engine on 2 x RTX 4090 (TP=2, eager mode). Each of three workloads uses three warm-up pairs and 20 measured pairs. This executes 260 requests per variant, or 520 requests in total. The table reports changes in the 20-run means; negative latency and positive throughput are improvements.

| Prompt / batch / output | TTFT | TPOT | Decode Throughput | Total latency |
|-------------------------|-----:|-----:|------------------:|--------------:|
| 512 / 1 / 64 | -0.75% | -1.98% | +2.02% | -1.90% |
| 2,048 / 4 / 64 | -6.32% | +0.07% | -0.04% | -1.26% |
| 8,192 / 8 / 64 | -7.88% | -0.16% | +0.15% | -3.97% |

The larger sample shows that the end-to-end benefit is concentrated in long-prefill TTFT, while decode metrics are nearly neutral at larger batches. All 20 single-request pairs produced identical token IDs. In sampled batched generation, small BF16 reduction-order differences can be amplified autoregressively, so this report must not be described as universal token-exact equivalence; Needle and LongBench remain the semantic-quality checks. Reproduce the paired benchmark with `bench_fused_rmsnorm_e2e.py`; the [raw report](benchmarks/kernels/fused_add_rmsnorm_e2e_9b_tp2_4090.json) contains mean, P50, P95, standard deviation, minimum, and maximum.

### Triton Fused KV Cache Compaction

The KV-compression copy path now uses a custom Triton kernel. It derives source and destination physical slots directly from the old Block Table, keep indices, and new Block Table, then copies K and V across every Full Attention layer in one launch. This replaces per-layer PyTorch gather/scatter operations and removes temporary tensors proportional to the retained KV size. Unsupported environments retain a semantically equivalent PyTorch fallback.

The microbenchmark below uses the per-rank KV geometry of Qwen3.5-9B with TP=2: nine Full Attention layers, two KV heads per rank, head dimension 256, and BF16. It uses 30 warm-ups and 100 timed iterations; every result is bit-exact.

| Source → retained tokens | PyTorch | Triton fused | Speedup | PyTorch temporary | Fused temporary |
|--------------------------|--------:|-------------:|--------:|------------------:|----------------:|
| 512 → 256 | 0.1819 ms | 0.0412 ms | 4.42x | 4.51 MiB | 0 MiB |
| 2,048 → 1,024 | 0.1829 ms | 0.0412 ms | 4.44x | 18.02 MiB | 0 MiB |
| 4,096 → 2,048 | 0.1948 ms | 0.0412 ms | 4.73x | 36.05 MiB | 0 MiB |
| 8,192 → 4,096 | 0.3891 ms | 0.1679 ms | 2.32x | 72.09 MiB | 0 MiB |

In a real Qwen3.5-9B run on 2 x RTX 4090 (TP=2) with four concurrent 2,048-token prompts and 128 output tokens, four KV-copy operations fell from 31.01 ms to 7.53 ms (4.12x, 75.71% lower). The complete compression path, including importance scoring, block allocation, and copying, fell from 74.04 ms to 49.61 ms (32.99% lower); both variants reclaimed 6,916 physical KV tokens. See the [operator report](benchmarks/kernels/fused_kv_compaction_9b_tp2_rank_4090.json) and the paired [PyTorch](benchmarks/kernels/kv_compaction_e2e_9b_tp2_pytorch_4090.json) / [fused](benchmarks/kernels/kv_compaction_e2e_9b_tp2_fused_4090.json) end-to-end reports.

### Concurrent Long-Context Compression Comparison

The following paired run uses Qwen3.5-9B on 2 x RTX 4090 (TP=2), 4 concurrent requests, 2,048 prompt tokens and 128 output tokens per request. It uses the same 128-Block KV-cache budget for both variants.

| Variant | TTFT (ms) | TPOT (ms) | Decode Throughput (tokens/s) | Peak KV Blocks | Peak Physical KV Tokens | Compression |
|---------|-----------|-----------|------------------------------|----------------|-------------------------|-------------|
| Baseline (disabled) | 2276.400 | 134.457 | 29.749 | 36 | 8696 | 0 runs |
| KV compression | 2234.307 | 132.768 | 30.128 | 36 | 8196 | 4 runs, 49.939 ms total, 4868 tokens reclaimed |

The physical-KV peak is 5.75% lower with compression. Peak block usage remains 36 because the metric includes the initial full prefill before compaction; this paired test must not be described as a peak-Block reduction. See the [baseline report](benchmarks/qwen3_5_9b_tp2_4090_b4_p2048_o128_baseline.json) and [compressed report](benchmarks/qwen3_5_9b_tp2_4090_b4_p2048_o128_compressed.json). Results come from a shared server and are reproducibility data rather than an isolated performance claim.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
