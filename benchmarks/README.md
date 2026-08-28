# Benchmark Reports / 评测报告索引

All retained benchmark results are maintained in the unified
[nano_vllm repository](https://github.com/cxkc19519-bot/nano_vllm).
This directory contains performance, kernel and quality reports. Historical
reports keep their original fields and values; missing metadata is not invented.

所有保留的结果集中放在本目录：3060 历史记录、4090 单卡/双卡性能记录、
Triton 算子与端到端对照、Needle 和 LongBench-E 质量评测。已有报告路径保持不变，
避免破坏引用；不会重新加入此前因错误而删除的质量报告。

## Report Inventory / 完整报告目录

| Hardware / scope | Report | Interpretation |
|------------------|--------|----------------|
| RTX 3060 12GB，早期单请求 | [3060 历史报告](benchmark_qwen3_5_3060.json) | 原始值恢复；早于数值对齐与融合算子，详见下方来源说明 |
| RTX 4090，Qwen3.5-2B | [2B 报告](benchmark_qwen3_5_2b_4090.json) | 早期 eager 功能/耗时记录 |
| RTX 4090，Qwen3.5-4B | [4B 报告](benchmark_qwen3_5_4b_4090.json) | 早期 eager 功能/耗时记录 |
| 2×RTX 4090，Qwen3.5-9B，TP=2 | [9B 报告](benchmark_qwen3_5_9b_tp2_4090.json) | 双卡 eager 功能/耗时记录 |
| 2×RTX 4090，9B，4 并发，2048 输入 / 128 输出 | [无压缩](qwen3_5_9b_tp2_4090_b4_p2048_o128_baseline.json)、[开启压缩](qwen3_5_9b_tp2_4090_b4_p2048_o128_compressed.json) | 相同 128 Block 预算下的性能与 KV 占用对照 |
| RTX 4090，BF16，Hidden Size 4096 | [Fused Add + RMSNorm 微基准](kernels/fused_add_rmsnorm_4090.json) | 算子级测量，不等同于模型端到端加速 |
| 2×RTX 4090，9B，3 组 workload | [Fused RMSNorm 配对端到端报告](kernels/fused_add_rmsnorm_e2e_9b_tp2_4090.json) | 每组 20 对正式测量，共 520 个请求 |
| RTX 4090，9B TP=2 单 Rank KV 形状 | [Fused KV Compaction 微基准](kernels/fused_kv_compaction_9b_tp2_rank_4090.json) | 4 种形状，搬运结果 bit-exact |
| 2×RTX 4090，9B，4 并发 | [PyTorch 搬运](kernels/kv_compaction_e2e_9b_tp2_pytorch_4090.json)、[Triton 搬运](kernels/kv_compaction_e2e_9b_tp2_fused_4090.json) | 区分打分、KV 搬运与完整压缩耗时 |
| 2×RTX 4090，9B，8192-token Chunked Prefill | [Needle 4K–32K](quality/needle_9b_tp2_scale_4k_32k_chunked_prefill_b160.json) | 20 个用例、40 次基线/压缩生成 |
| 2×RTX 4090，9B，Chunked Prefill | [LongBench-E 子集](quality/longbench_e_9b_tp2_scale_3tasks_5samples_chunked.json) | 3 个任务、15 个样本、30 次对照生成；非完整 LongBench |

## RTX 3060 Historical Report

The restored [JSON](benchmark_qwen3_5_3060.json) comes from
[`benchmark_qwen3_5_3060.json` at commit 6a7671c](https://github.com/cxkc19519-bot/nano_vllm/blob/6a7671cfa61ad4933322f0cb05192de90d6e3a39/benchmark_qwen3_5_3060.json).
Its values are preserved unchanged. The
[README at the same commit](https://github.com/cxkc19519-bot/nano_vllm/blob/6a7671cfa61ad4933322f0cb05192de90d6e3a39/README.md)
labels the model Qwen3.5-0.8B and the backend Windows / eager PyTorch SDPA.
The JSON does not include a model ID, model revision, operating system or a full
dependency lockfile. Its `torch` version string is retained as recorded, not
independently verified. This is therefore a historical measurement, not a
fully pinned current-version reproduction.

来源提交是统一仓库现有历史的一部分，不依赖已删除仓库继续存在。该记录只包含
1 个正式请求（512 输入、16 输出，15 个 Decode token）；TTFT 为 183.910 ms，
TPOT 为 44.564 ms，Decode Throughput 为 22.440 token/s，压缩 1 次耗时
32.712 ms，回收 353 个物理 KV token。

### Rerun the recorded workload / 复跑原工作负载

下面是在当前代码上复跑同一组输入和压缩参数的命令，并非承诺复现历史耗时。
模型路径需要改为实际路径；输出使用新文件名，避免覆盖历史数据。

```bash
python bench_qwen3_5.py --model /path/to/Qwen3.5-0.8B \
  --tensor-parallel-size 1 --batch-size 1 --seed 20260810 \
  --prompt-tokens 512 --output-tokens 16 --warmup 1 \
  --max-model-len 1024 --max-batched-tokens 1024 --max-seqs 2 \
  --compress-threshold 256 --sink-tokens 32 --recent-window 64 \
  --recent-queries 4 --top-k 64 \
  --output benchmarks/benchmark_qwen3_5_3060_rerun.json
```

The command uses Bash line continuations. In Windows PowerShell, run it on one
line or use PowerShell's backtick continuation syntax; activate the `nano-vllm`
environment first.

## Metric Boundaries / 指标口径

- 3060 和 4090 的模型、运行后端、代码版本、环境和负载并不统一，不能作为 GPU 性能横向对比或优化 A/B 结论。
- 早期 3060 报告早于 Qwen3.5 数值对齐修复；该记录不能用来证明当前生成质量、融合算子收益或 CUDA Graph 正确性。
- `peak_used_mib` 是已用 KV Block 折算容量，不是 CUDA 分配器记录的整个 GPU 显存峰值，也不是预分配 KV 池的大小。
- 压缩后回收物理 KV token 的比例不等同于峰值显存下降比例；原报告中的每种指标保持其原口径。
- 算子级加速与端到端加速分开报告，LongBench-E 结果明确限定为三任务子集。
- 本次仓库合并仅归档历史报告和更新文档，没有重新执行 GPU 性能或质量 Benchmark。

## RTX 4090 Report Details

The performance reports emitted by `bench_qwen3_5.py` record workload,
latency/throughput, KV usage and compression statistics; newer reports also
include tensor-parallel configuration.

The `qwen3_5_9b_tp2_4090_b4_p2048_o128_baseline.json` and
`qwen3_5_9b_tp2_4090_b4_p2048_o128_compressed.json` files are a paired,
four-request long-context comparison with KV compression disabled and enabled.

The `quality/` directory contains the larger Qwen3.5-9B / 2 x RTX 4090
long-context quality comparisons.  The Needle report covers 4k through 32k
contexts at five insertion depths (40 baseline/compressed generations).  The
LongBench-E report covers three tasks and 15 unique samples (30 generations).

The `kernels/` directory contains reproducible operator microbenchmarks.  The
Fused Add + RMSNorm report compares the aligned PyTorch formula with the custom
Triton kernel on an RTX 4090; its speedups are operator-level rather than
end-to-end model claims.  It also contains a paired Qwen3.5-9B / TP=2
end-to-end report with three workloads, 20 measured pairs per workload, and
520 generated requests in total.

The fused KV-compaction reports compare the original PyTorch gather/scatter
path with a direct Triton K/V copy.  The operator report uses Qwen3.5-9B TP=2
per-rank cache geometry; the two end-to-end reports split importance-selection
time from physical KV-copy time for a four-request workload.
