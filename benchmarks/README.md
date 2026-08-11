# Benchmark Reports

This directory contains reproducible JSON reports emitted by `bench_qwen3_5.py`.
Each report records the workload, tensor-parallel configuration, latency and
throughput metrics, KV-cache usage, and KV-compression statistics.

The `qwen3_5_9b_tp2_4090_b4_p2048_o128_baseline.json` and
`qwen3_5_9b_tp2_4090_b4_p2048_o128_compressed.json` files are a paired,
four-request long-context comparison with KV compression disabled and enabled.
