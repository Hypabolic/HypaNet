# Status (updated as results land)

_Last update: 2026-11-05 03:30. The v0.3 Phase A Fable session is running._

## v0.2: small MLP on engineered features (finished apart from M4)

All scores are on the **validation** split. Macro-F1 is given with its 95% session-bootstrap CI.

| Phase | Labelled examples seen | Macro-F1 | False-loop alarm | Network = policy |
|---|---|---|---|---|
| A-strict / A-selfcheck (identical weights) | 0 | 0.668 [0.660, 0.676] | 2.6% | 100% (84,679 windows) |
| **B (one revision after 200 disclosed windows)** | 200 | **0.936 [0.932, 0.940]** | 3.1% | 100% |

- **Compilation is exact in every phase.** The hand-written network always agrees with Fable's own policy.
- **The Phase A shortfall came from the policy, not the compilation.** STALLED recall was 0.45 and PRODUCTIVE was often confused with RECOVERY. Phase B fixed most of this.
- **Phase B cost.** 31.6 min, US$40, `claude-fable-5-1` only. The audit flagged 3 items, all false positives: a `~` inside an f-string was read as a home-directory path (`experiments/m3/logs/AUDIT_NOTE.md`).
- **Still to do (M4, running now).** Trained same-shape baselines, the rules engine and the priors, then one sealed test and OOD evaluation of every v0.2 column.

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

## Waiting on

1. v0.3 Phase A. Then the harness build, validation scoring (harness only), the Phase B disclosure and Phase B.
2. v0.2 M4: Phase C MLPs and the sealed evaluation.
3. v0.3 Phase C, the trained transformers. These are CPU-heavy, so they run after Fable finishes.
