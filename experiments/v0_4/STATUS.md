# v0.4 status

_Last update: 2026-11-05 20:30. Plan: `agent_doom_loop_detector_plan_v0_4.md` (draft). Nothing has been scored. No Qwen output has been compared with a label._

## M0 setup (in progress)

**Machine.** The work now runs on an Apple M4 Pro (24 GB, 12 cores), not the Intel Mac the plan's feasibility tests used. Two of the plan's compute findings do not hold here:

- the Mac GPU works (MPS);
- pip ships current torch for this machine, so no conda environment is needed.

**Data.** All datasets were rebuilt from the scripts and match the committed manifests: every canonical v0.2 file is byte-identical, and every v0.3 raw file matches its sha256. The source-hash fields in the manifests now differ because the code grew after v0.2. The committed manifests are kept unchanged.

**Qwen environment.** `envs/qwen` is a separate uv project with Python 3.12, torch 2.14.1 and transformers 5.22.1. The model is `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17` (text model, 752M parameters with tied embeddings; vision encoder unused). The frozen v0.3 environment at the repo root is unchanged.

**Task export.** `scripts/v04_export_tasks.py` runs in the frozen environment and writes `datasets/v0_4/{dev,val}.tasks.jsonl.gz` and `{split}.labels.npz` (gitignored), plus `EXPORT_MANIFEST.json`.

- The tasks file holds rendered blocks and window positions only. It has no labels, families or gate results.
- val has 4,004 sessions and 87,176 windows (75,078 scored). dev has 12,000 sessions and 259,290 windows (223,930 scored). These are the v0.3 counts.
- `tests/test_v04_export.py` checks the rows against `v03.scoring.load_val` and that test, OOD and counterfactual are refused.

**Verbaliser readout.** `envs/qwen/hypaqwen/readout.py` makes one pass per session. At each window end it snapshots the cache (full-attention KV plus the DeltaNet conv and recurrent states), runs the suffix, reads the four label-word logits, then restores the cache. `transformers`' `Cache.crop` cannot do this, because it does not roll back the DeltaNet recurrent state.

`envs/qwen/tests/test_readout.py` checks the readout against a separate prompt per window on 3 dev sessions (first, middle and last windows):

- logits agree within 2e-3 and every argmax is the same;
- a label never changes when later blocks are removed;
- passes on MPS and on CPU.

**Interface (DRAFT, not frozen).** `envs/qwen/hypaqwen/interface.py`:

- Qwen chat format, thinking off;
- the four class definitions from spec v0.3 §2;
- suffix `State:`;
- label words ` productive`, ` recovering`, ` stalled`, ` looping`.

` oscillating` is two tokens in Qwen's vocabulary, so the plan's example word list cannot be used as written. ` looping` is the single-token stand-in. G1 freezes the final text under a written choice rule on dev.

**Token audit** (`artifacts/reports/v04_token_audit.json`, draft interface):

| Split | Tokens per session, mean (p95, max) | vs v0.3 tokeniser | Readout tokens, total |
| --- | --- | --- | --- |
| val | 1,726 (3,478, 6,849) | 1.97× | 7.87M |
| dev | 1,721 (3,505, 6,522) | 1.98× | 23.5M |

The counts include the 167-token prefix. Without it, val averages 1,559 tokens, as the plan measured. The 11 suffix tokens per window add 14%.

**Throughput on this Mac** (fp32, draft interface, 30 dev sessions):

| Backend | Readout tok/s | One full val pass |
| --- | --- | --- |
| MPS | 445 | about 5 h |
| CPU (12 threads) | see below | — |

The plan assumed 215 tok/s and about 20 h on CPU. MPS is compute-bound here: batching 8 sequences costs about 8×. Each forward call also has a fixed 40–70 ms overhead from the pure-PyTorch DeltaNet fallback, and an 11-token suffix costs about as much as a 50-token block. A smaller DeltaNet chunk size gives only about 15%, so stock transformers is kept. Dev-scale work (probe features on 23.5M tokens, LoRA) still belongs on a CUDA GPU with the fused kernels.

## Next

1. Matthew's decisions (plan, "Decisions needed"). The fresh sealed test set blocks the rest of M0.
2. Fable kernel: the plan's tools, a token-budget meter, and v0.3's guards against autograd and optimisers.
3. The prompt choice rule and candidate prompts, judged on dev only. Then the G1 pre-registration.
4. Scoring bridge: Qwen logits to `metrics.score_report` in the frozen environment.

## Known pre-existing test failures (not v0.4)

Two older tests fail on this machine because pytest's temp directory is under `/private/var`, outside the repo:

- `tests/test_m4_eval.py::test_sealed_script_refuses_second_run` (path not under the repo root);
- `tests/test_v03_runner.py::test_side_table_is_unreachable_at_run_time[absolute]` (the guard's message includes the temp path).

The first fails with the v0.4 changes stashed too; the second is in v0.3 runner code that v0.4 does not touch. Every other test passes.
