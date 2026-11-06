# Status (updated as results land)

_Last update: 2026-11-05 18:30. The one-time v0.3 sealed evaluation is complete (`artifacts/reports/v03_sealed.json`)._

## v0.2: small MLP on engineered features (complete)

**Sealed test and OOD** results were computed once (`artifacts/reports/m4_sealed_v02.json`). Values are macro-F1 with 95% session-bootstrap CIs. For Phase C the figure is the mean ± sd over 3 seeds.

| Column | Labels seen by the author | Test macro-F1 | Test false alarm | OOD macro-F1 | OOD false alarm |
|---|---|---|---|---|---|
| A-strict = A-selfcheck (hand-written, 0 labels) | 0 | 0.667 [0.659, 0.675] | 2.5% | **0.810** [0.803, 0.817] | 1.3% |
| **B (hand-written, one revision)** | 200 windows | **0.936** [0.932, 0.940] | 3.1% | **0.870** [0.867, 0.873] | 8.5% |
| C-32 (trained) | 563 windows | 0.618 ± 0.130 | 8.2% | 0.517 ± 0.182 | 30.6% |
| C-128 (trained) | 2,207 | 0.932 ± 0.006 | 4.9% | 0.644 ± 0.026 | 39.1% |
| C-512 (trained) | 9,216 | 0.966 ± 0.003 | 3.1% | 0.727 ± 0.026 | 28.6% |
| C-2048 (trained) | 38,120 | 0.985 ± 0.000 | 1.6% | 0.768 ± 0.034 | 19.9% |
| C-all (trained) | 223,930 | 0.992 ± 0.001 | 0.8% | 0.843 ± 0.023 | 11.6% |
| Rules engine (Codex) | 0 | 0.417 | 1.1% | 0.415 | 2.4% |
| Constant prior | 0 | 0.136 | 0 | 0.181 | 0 |

**What the table shows:**
- **Compilation is exact.** H2 fidelity is 1.000 in every hand-written phase. That holds on test, OOD, val and on 100,000 random valid windows, so the network always computes exactly Fable's stated policy.
- **In-distribution, trained beats hand-written.** Fable's one-revision network (B, 200 labelled windows) matches a trained network given about 2,200 windows (C-128). Trained networks with more data reach 0.99.
- **Out of distribution, hand-written beats trained.** B scores 0.870 on the separately written OOD generator, above every trained network including C-all (0.843). Even zero-label Phase A (0.810) beats C-512. Trained networks also raise far more false loop alarms on OOD (12–39%, against 1–9%).
- **Phase A's weakness was policy design, not compilation.** STALLED recall was low and PRODUCTIVE was often confused with RECOVERY. That is H1, not H2.

Costs: Phase A about $59, Phase B $40, all `claude-fable-5-1`.

## v0.3 sealed results (one attempt, 2026-11-05 16:05–18:21)

`scripts/v03_sealed_eval.py --sealed --confirm-one-time`. The pre-registration is `experiments/v0_3/sealed/README.md`, committed at 2fce67f before the run. The run log is `experiments/v0_3/sealed/run.log`.

**Checks before scoring.** All passed before the attempt record:
- the val gate reproduced the official A/B val reports exactly;
- 0 policy failures on val and counterfactual;
- test and OOD matched their manifests;
- the window set is row-identical to v0.2's, so the v0.2 columns are cited from `m4_sealed_v02.json`.

**Results.** Values are macro-F1. False alarms are the false-loop alarm rate. Phase C is mean ± sd over 3 seeds where there are 3.

| Column | Labels | Test F1 | Test false alarms | OOD F1 | OOD false alarms |
|---|---|---|---|---|---|
| **v0.3 A** (hand-written transformer, raw logs) | 0 | 0.744 [0.734, 0.754] | 1.9% | **0.815** [0.806, 0.823] | 4.0% |
| **v0.3 B** (hand-written, 1 revision) | 200 windows | **0.957** [0.953, 0.961] | 2.5% | **0.908** [0.905, 0.912] | 6.8% |
| v0.3 C-32 (trained transformer) | 32 sessions | 0.347 ± 0.034 | 16–35% | 0.241 ± 0.055 | 45–62% |
| v0.3 C-128 | 128 sessions | 0.450 ± 0.035 | 10–16% | 0.288 ± 0.035 | 35–44% |
| v0.3 C-512 | 512 sessions | 0.707 ± 0.008 | 5–13% | 0.305 ± 0.034 | 49–80% |
| v0.3 C-2048 (1 seed) | 2,048 sessions | 0.755 | 4.4% | 0.400 | 50.4% |
| v0.3 C-all (1 seed) | 11,864 sessions | 0.923 | 6.1% | 0.488 | 34.5% |
| v0.2 A (hand-written MLP, engineered features) | 0 | 0.667 | 2.5% | 0.810 | 1.3% |
| v0.2 B | 200 windows | 0.936 | 3.1% | 0.870 | 8.5% |
| v0.2 C-all (trained MLP, 3 seeds) | 223,930 windows | 0.992 | 0.8% | 0.843 | 11.6% |
| Rules engine (Codex) | 0 | 0.417 | 1.1% | 0.415 | 2.4% |
| Constant prior | 0 | 0.136 | 0 | 0.181 | 0 |

**Compilation (H2) and counterfactual consistency:**
- **H2 fidelity:** A is 1.000 on test and OOD. B is 0.9995 on test and 1.000 on OOD.
- **Counterfactual consistency** (strict endpoint rate): v0.3 A 0.750, v0.3 B 0.862, C-all 0.496, C-2048 0.047. The v0.2 figures are A 0.715 and B 0.961.

**What it shows:**
- **On raw logs, the hand-written transformer beats the trained one with about 60× fewer labelled sessions.** v0.3 B, from 200 labelled windows, scores 0.957 on test. C-all, trained on all 11,864 dev sessions, scores 0.923.
- **Out of distribution, the gap is very large.** v0.3 B scores 0.908 on OOD, the best OOD result of any model in the project; the previous best was v0.2 B at 0.870. Zero-label v0.3 A (0.815) beats every trained transformer.
- **Trained transformers collapse on the separately written OOD generator.** They score 0.24–0.49, with 35–80% false loop alarms.
- **Raw logs versus engineered features.** In distribution, the trained v0.2 MLP on engineered features is still the best model (0.992). The hand-written v0.3 network does not need the engineered features to beat v0.2's hand-written columns.

**Caveats:**
- **The trained transformers are capped by CPU budget.** E8 and the Phase C README allow 3–4 epochs for the large budgets, and C-2048 was still improving at its cap. The trained baselines are therefore under-trained against what a GPU budget would allow.
- **All data is synthetic** (rendered generator logs). OOD is a separately written generator, not real transcripts.
- **2048 and all are single seeds** (E8).

## v0.3: hand-programmed transformer on raw logs

- **Spec:** `agent_doom_loop_detector_spec_v0_3.md` (r2). §0.1 lists every decision taken under delegated authority (E1–E11) for you to review.
- **Model:** a 4-layer causal transformer, d=128, 8 heads, with an event-distance attention bias and no LayerNorm. It has **2,128,932 parameters**, all starting at zero.
- **Input:** tokenised log text, one block per tool call. Each block holds a header (tool, arguments, status, test counts, tokens), an error line (class, location, noise) and opaque per-file diff ids. There are no engineered features.
- **Labels, sessions and scored windows** are exactly those of v0.2, so the two versions are directly comparable.
- **Data integrity:**
  - The raw capture draws no RNG. Regenerated v0.2 data is byte-identical.
  - The generators' code text revealed the latent kind of each edit. Diffs are therefore shown as opaque ids.
  - Two smaller leaks were found and removed: the `bytes=` size and the number of hunks per file.
  - The dev leakage audit is now at chance.
- **Feasibility spike:** skipped to save time (E10). It was replaced by a paper analysis. Read any Phase A shortfall with that in mind.
- **Isolation:**
  - The author has a fresh Fable session in a new sealed directory.
  - Its `./py` runs under a guard that blocks reads outside the directory, RNG, autograd and optim.
  - Builds run from a single-file private copy.
  - The kit tests are recorded in `KIT_TESTS.md`.
- **Phase A:** launched 03:25. Cost cap US$500.
  - **Interrupted 04:42** by the account's shared 5-hour usage limit (HTTP 429). It had made 14 tool calls, all reads of the packet plus a 153-byte directory-listing script in scratch, so nothing was authored. It had used 256K thinking tokens and cost $14.89.
  - **Audit note:** the audit's single `[EXECUTED]` finding is `./py -c "import os; print(sorted(os.listdir('.'))); ..."`. The semicolons sit inside a quoted Python string that lists the author's own directory. The compound-command heuristic matched it, but it is not a redirection and reads nothing outside the sandbox. Reviewed and judged benign.
  - **Continuation 1** was launched at 09:17 with `scripts/v03_launch.sh continue 1`, using the fixed `prompt_continue.md` (sha256 matches HANDOVER).
  - **Submitted final at 10:51.** Continuation 1 made 64 tool calls. Phase A cost $42.73 in total: the logged `total_cost_usd` is cumulative across resumed launches, so it already includes the interrupted launch's $14.89, and continuation 1 itself cost $27.84. (Corrected 6 Nov 2026; this line said $57.62, double-counting the $14.89.) The audit found 0 violations and 0 denials.
  - **Archived at 10:52** in `weight_program/v0_3/phase-a/`:
    - artifact sha256 `ba6f4743e6c4c32a0a71124357511146889a94204d8263d547edaa85b251dfb6`;
    - the rebuild is byte-identical and the packet is registered;
    - 121,126 of 2,128,932 parameters are non-zero.
  - **Build attempts:** `attempts/0001.json` was an operator error. `--author-dir` was given as the author root instead of `output/v03-phase-a`, so the runner reported the required files missing. The author's files were not involved, and no error text was disclosed to the author. `0002.json` is the build that was archived.
  - **Validation (harness only, never shown to the author)** is in `artifacts/reports/v03_val_v03-phase-a.json`. The primary population, `excluding_disclosed`, has 3,804 sessions and 69,974 scored windows.
    - Macro-F1 is **0.750** [0.740, 0.760] and the false-alarm rate 2.0%. H2 fidelity is **1.000**, with 0 policy failures. For all of val the macro-F1 is 0.748.
    - For comparison, v0.2 Phase A on val (engineered features) scored 0.668 and v0.2 Phase B 0.936.
    - Per-class F1:

      | Class | F1 | Recall | Precision |
      |---|---|---|---|
      | OSCILLATING | 0.942 | 0.93 | 0.96 |
      | RECOVERY | 0.771 | 0.68 | 0.90 |
      | PRODUCTIVE | 0.751 | 0.96 | 0.62 |
      | STALLED | 0.535 | **0.37** | 0.97 |

    - The main weakness, as in v0.2, is STALLED recall: stalls are mostly called PRODUCTIVE.
- **Phase B:** the disclosure (200 windows, sha256 `27c8deb0…`) was handed over and Fable resumed at 10:54 (`scripts/v03_launch.sh phase-b`). Cost cap US$250.
  - **Submitted final at 11:53.** The audit is clean. Phase B cost $21.89 (the cumulative `total_cost_usd` went from $42.73 to $64.62), so v0.3 hand-written work cost $64.62 in total. (Corrected 6 Nov 2026; this line said the session cost $64.62 and the total $122.24, summing cumulative figures.)
  - **Fable's self-report:** Phase A's policy got 141/200 on the disclosure, and the revised policy gets 199/200 (V053 left deliberately). The changes are listed in `REVISION.md`:
    - repetition is judged over the whole session;
    - failed validations are treated as episodes, with retries and resolutions;
    - repeated identical validations count as stall evidence;
    - progress means a new high in passed tests.
  - **Archived at 11:54** in `weight_program/v0_3/phase-b/`:
    - sha256 `e2c9aeff4bb18244395ab17a9f9ac59dd7b10404944098d414d94c5e965492a0`;
    - built on the first attempt, with a byte-identical rebuild;
    - 156,501 parameters are non-zero.
  - **Validation (harness only)** is in `artifacts/reports/v03_val_v03-phase-b.json`. On `excluding_disclosed` the network scores macro-F1 **0.954** [0.950, 0.958], with a 2.4% false-alarm rate. The policy scores 0.954. For all of val it is 0.955.
    - This is above v0.2 Phase B on val (0.936), on raw logs.
    - Per-class F1:

      | Class | F1 | Recall |
      |---|---|---|
      | STALLED | 0.966 | 0.95 (up from 0.37) |
      | OSCILLATING | 0.964 | — |
      | PRODUCTIVE | 0.955 | — |
      | RECOVERY | 0.931 | — |

  - **H2 fidelity is 0.9997, not 1.000.** On all of val, 25 of 84,679 gate-passing windows disagree, all in the `abab_oscillation` family (25 of its 9,706). This is a small compilation gap in the Phase B circuit, which Fable's 4,484-prefix check did not reach.

## Waiting on

1. v0.3 Phase A. Then the harness build, validation scoring (harness only), the Phase B disclosure and Phase B.
2. v0.2 M4: Phase C MLPs and the sealed evaluation.
3. **v0.3 Phase C, the trained transformers, is complete** (11 runs, 14:06). The table gives each run's best early-stopping macro-F1. That is measured on the 500-session ES subset of val, which is not the population used for the A/B val numbers. Full scoring happens in the sealed evaluation.

   | Budget | Seeds | ES macro-F1 |
   |---|---|---|
   | 32 | 3 | 0.344 / 0.292 / 0.372 |
   | 128 | 3 | 0.404 / 0.450 / 0.468 |
   | 512 | 3 | 0.732 / 0.729 / 0.724 |
   | 2048 | 1 | 0.798 (still improving at the 4-epoch cap) |
   | all (11,864 sessions) | 1 | 0.934 (3-epoch cap) |
