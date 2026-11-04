# Status (updated as results land)

_Last update: 2026-11-05 10:55. v0.3 Phase A is submitted and archived; the Phase B disclosure is being built._

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
  - **Submitted final at 10:51.** Continuation 1 made 64 tool calls and cost $42.73, so Phase A cost $57.62 in total. The audit found 0 violations and 0 denials.
  - **Archived at 10:52** in `weight_program/v0_3/phase-a/`:
    - artifact sha256 `ba6f4743e6c4c32a0a71124357511146889a94204d8263d547edaa85b251dfb6`;
    - the rebuild is byte-identical and the packet is registered;
    - 121,126 of 2,128,932 parameters are non-zero.
  - **Build attempts:** `attempts/0001.json` was an operator error. `--author-dir` was given as the author root instead of `output/v03-phase-a`, so the runner reported the required files missing. The author's files were not involved, and no error text was disclosed to the author. `0002.json` is the build that was archived.

## Waiting on

1. v0.3 Phase A. Then the harness build, validation scoring (harness only), the Phase B disclosure and Phase B.
2. v0.2 M4: Phase C MLPs and the sealed evaluation.
3. v0.3 Phase C, the trained transformers. 9 of 11 runs are done (32, 128 and 512 × 3 seeds). The runner was killed at 09:01 and restarted at 09:17, and 2048 seed 0 is running again from scratch. Still to run: 2048 seed 0 and all seed 0.
