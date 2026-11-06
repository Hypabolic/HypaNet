# Parallel MPS readout benchmark (2026-11-06)

**Question** (Matthew): how many Qwen readout sessions can run in parallel on this Mac, to speed up the val readouts? GPU only ("we already know the cpu is slow").

**Machine:** Apple M4 Pro (8 performance and 4 efficiency cores), 24 GB unified memory, macOS 27. Qwen3.5-0.8B in fp32 on MPS with the torch Gated DeltaNet fallback, using `envs/qwen` (torch 2.14.1, transformers 5.22.1). Nothing else was using the GPU.

**Method:** `bench.py` runs N unchanged `hypaqwen.run_readout --split val --sessions SHARD --device mps` processes at once. They read the first 48 sessions of `datasets/v0_4/VAL_QUICK.txt` (85,731 readout tokens, 941 windows), dealt round-robin to the workers. Base Qwen, no edit, label-free. Each run's numbers:

- aggregate tok/s, which is total tokens divided by the slowest worker's readout loop seconds, so model loading is left out;
- wall time;
- the lowest system free-memory percentage that `memory_pressure` reported (sampled every 3 s).

`compare.py` checks every parallel run's logits against the single-process run. The raw results are in `results.jsonl`. The run outputs were written to the session scratchpad and not kept; a re-run of `bench.py` writes them to `runs/` (gitignored). compare.py output at the time: `mps2`, `mps3`, `mps4` and `mps4t2`: max |dlogit| 0.0, row_pos equal True, argmax changes 0 of 941. An independent check re-ran it on the scratchpad outputs and got the same.

## Results

| Workers (threads) | Aggregate tok/s | Speed-up | Per worker tok/s | Readout s | Wall s | Min free memory |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (default) | 456.5 | 1.00× | 456.5 | 187.8 | 192.7 | 72% |
| 2 (default) | 732.7 | 1.60× | 370, 382 | 117.0 | 120.5 | 54% |
| **3 (default)** | **841.3** | **1.84×** | 298, 302, 314 | 101.9 | 108.6 | **39%** |
| 4 (default) | 915.0 | 2.00× | 243, 240, 226, 229 | 93.7 | 105.4 | 14% |
| 4 (2 CPU threads each) | 915.0 | 2.00× | 242, 239, 226, 241 | 93.7 | 102.4 | 11% |

**Correctness:** the 2-, 3- and 4-worker runs give the single-process logits exactly. The max |logit difference| is 0.0, the row positions are equal, and 0 of 941 argmaxes changed. Each session is read out on its own, so how sessions are split across processes makes no difference.

## Conclusions

- **The GPU is the limit.** Each added worker gains less (+60%, +15%, +9%). Capping each process at 2 CPU threads changes nothing, so CPU contention is not the bottleneck.
- **Memory caps the count at 4.** Each fp32 process takes about 4 GB. At 4 workers free memory fell to 11–14%, and a fifth would swap.
- **Chosen: 3 workers** (Matthew, "Yes"). They give about 92% of the 4-worker throughput and keep 39% of memory free. An earlier single readout died unexplained while the machine was loaded (see `experiments/v0_4/STATUS.md`), so this margin matters.
- **Implemented as deviation D2:** `scripts/v04_parallel_readout.py` (registry revision 3, PREREGISTRATION §13 log). In the first real run, base Qwen on full val, the workers held about 280–285 tok/s each, so about 850 tok/s in total.

| Job | One process | 3 workers |
| --- | --- | --- |
| Quick look (1,000 sessions, about 1.94M tokens) | about 75 min | about 41 min |
| Full val (4,004 sessions, 7.87M tokens) | about 5 h (424–440 tok/s on real runs) | about 2.5–3 h |

## Reproduce

```sh
cd envs/qwen
uv run python ../../experiments/v0_4/benchmarks/parallel_readout/bench.py mps1 mps:0
uv run python ../../experiments/v0_4/benchmarks/parallel_readout/bench.py mps3 mps:0 mps:0 mps:0
.venv/bin/python ../../experiments/v0_4/benchmarks/parallel_readout/compare.py mps3
```

`mps:0` means MPS with the readout's default thread count. `mps:2` sets `--threads 2`. Run nothing else on the GPU while benchmarking.
