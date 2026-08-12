# Benchmark Reports

This directory contains reproducible JSON reports emitted by `bench_qwen3_5.py`.
Each report records the workload, tensor-parallel configuration, latency and
throughput metrics, KV-cache usage, and KV-compression statistics.

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
