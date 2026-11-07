# v0.5 status (6 Nov 2026)

v0.5 applies the premise of Experiment 5 of "Sparks of Weight-Space Programming" (Takehi, Teoh et al.,
MIT CSAIL, 2 Nov 2026) to the doom-loop task: Fable writes the weights of the v0.3 transformer with **no
tools**, by reasoning alone, in the `edits` language. Then, as in v0.3 Phase B, a new Fable session improves
that network with tools and the same 200 labelled val windows.

## Results (validation, excluding_disclosed: 3,804 sessions, 69,974 scored windows)

| Network | How it was made | Labels | Macro-F1 [95% CI] | False alarms | Cost |
|---|---|---|---|---|---|
| **v0.5 r1** | Fable, no tools, 4 replies (1 prompt + 3 "Continue.") | 0 | **0.320** [0.313, 0.328] | 95.8% | US$16.70 |
| **v0.5 Phase B** | new Fable session, r1 + tools + 200 labelled windows, one revision | 200 | **0.957** [0.954, 0.960] | 3.2% | US$40.75 |
| v0.3 A | Fable, with code | 0 | 0.750 [0.740, 0.760] | 2.0% | US$42.73 |
| v0.3 B | Fable, same session as A, 200 labelled windows | 200 | 0.954 | 2.4% | US$21.89 |
| v0.3 C-all | trained, same architecture | 223,930 | 0.926 (early-stopped on val, optimistic) | 5.8% | |

Source reports: `artifacts/reports/v05_val_r1.json` (official) and
`experiments/v0_5/phase_b/val_report/v05_val_v05b.json` (scored with the same scorer in its `--artifact`
mode, so marked `"official": false`; the scorer's official mode only takes `experiments/v0_5/runs/RUN`).
v0.3 rows: `experiments/v0_3/STATUS.md`.

**Paired on val** (excluding_disclosed, shared session bootstrap; `experiments/v0_5/sealed_dryrun/`, which
also reproduced every v0.5 and v0.3 val report exactly):

| Pair | Macro-F1 difference | False-alarm difference |
|---|---|---|
| v0.5 B − v0.3 B | +0.0028 [−0.0011, +0.0068] (level) | +0.0077 [+0.0062, +0.0092] |
| v0.5 B − v0.3 C-all | +0.0313 [+0.0264, +0.0366] | −0.0256 [−0.0302, −0.0213] |
| v0.5 B − v0.3 A | +0.2073 [+0.1976, +0.2171] | +0.0120 |
| v0.5 r1 − v0.3 A | −0.4292 [−0.4420, −0.4168] | +0.9376 |

## Sealed test and OOD (one-time run, 6 Nov; `artifacts/reports/v05_sealed.json`)

Pre-registered in `experiments/v0_5/sealed/README.md` (committed `abe8b83`, before the run). All sessions,
95% session-bootstrap CIs; every v0.3 column re-scored in the same run.

| Network | Sealed test macro-F1 | FA | Sealed OOD macro-F1 | FA |
|---|---|---|---|---|
| v0.5 r1 (no code, 0 labels) | 0.326 [0.319, 0.333] | 95.7% | 0.292 [0.283, 0.300] | 81.5% |
| **v0.5 B** (no code + 200 labels) | **0.959** [0.956, 0.961] | 3.2% | **0.765** [0.762, 0.769] | **31.4%** |
| v0.3 A (code, 0 labels) | 0.744 [0.734, 0.754] | 1.9% | 0.815 [0.806, 0.823] | 4.0% |
| v0.3 B (code + 200 labels) | 0.957 [0.953, 0.961] | 2.5% | 0.908 [0.905, 0.912] | 6.8% |
| v0.3 C-all (trained, 223,930 labels) | 0.923 [0.918, 0.929] | 6.1% | 0.488 [0.478, 0.497] | 34.5% |

Paired (pre-registered primaries first):

| Pair | Test ΔF1 | Test ΔFA | OOD ΔF1 | OOD ΔFA |
|---|---|---|---|---|
| v0.5 B − v0.3 B | +0.0015 [−0.0020, +0.0052] | +0.0078 [+0.0064, +0.0093] | **−0.1430** [−0.1473, −0.1386] | **+0.2462** [+0.2392, +0.2533] |
| v0.5 B − v0.3 C-all | +0.0351 [+0.0300, +0.0405] | −0.0282 [−0.0329, −0.0237] | +0.2774 [+0.2680, +0.2871] | −0.0317 [−0.0419, −0.0215] |
| v0.5 B − v0.3 A | +0.2145 [+0.2048, +0.2239] | +0.0136 | −0.0495 [−0.0586, −0.0399] | +0.2739 |
| v0.5 r1 − v0.3 A | −0.4183 [−0.4303, −0.4066] | +0.9378 | −0.5230 [−0.5345, −0.5110] | +0.7748 |

- **In distribution, the no-code lineage ends level with v0.3 B** (test +0.0015, CI spans 0; slightly more
  false alarms).
- **Out of distribution it does not hold up like v0.3 B:** −0.143 macro-F1 and false alarms 31.4% against
  6.8%. It is worse than v0.3 B on all 10 OOD families (false alarms 12–48% per family against 0–15%), and still
  better than the trained C-all on all 10 by macro-F1. Its OOD false-alarm rate is about the trained network's
  (31.4% against 34.5%). **Why** (`ood_analysis/FINDINGS.md`, 7 Nov, on a fresh OOD
  set from the same generator, no sealed data): almost all of the loss is PRODUCTIVE called STALLED. Phase B's
  behavioural "ineffective" rules (a failure on an already-seen error line, an edit after a repeated test
  result, a repeated target keyed on its first word) agree with the oracle 62–84% in distribution but 28–44%
  OOD. Unlike v0.3 A/B, there is no progress veto (a new high in a passed count), so 92% of the false STALLED
  windows have real progress that the network never looks for.
- **Reproduction:** every v0.3 column reproduced `v03_sealed.json` on val, counterfactual and test. On OOD, one
  trained model (C-128 seed0) differs in one window of 74,000 (macro-F1 0.2742913 against 0.2742901, an arm64 vs
  x86_64 float difference); it is flagged in the report and affects no v0.5 number or primary pair.
- Run log: `experiments/v0_5/sealed/sealed_run.log`.

- **r1:** OSCILLATING F1 0.83 (precision 0.98, recall 0.73) from a hand-reasoned hunk-reversion circuit and a
  residue code (token id mod 4, 5, 7; mod 3, 13) written as strided fills; STALLED predicted on most windows.
  Turn 1 was ~236k thinking tokens before any text; one message hit the 128k output cap (`replies/INDEX.json`: 001.p1 cut off).
- **Phase B:** found one wrong bias in r1 (layer 0 MLP units 32–63, −2 instead of −3) that made the
  target code fire everywhere, then rebuilt the decision logic around the oracle's rules (1,262 statements, 2,813
  entries changed). Artifact `223d0743…`, base r1 `46d4b2a2…`. Its own account: `weight_program/v0_5/phase-b/`
  (`REVISION.md`, `CIRCUITS.md`, `ITERATIONS.md`, `author_output/scratch/`).

## Deviations and incidents

- Experiment 5 deviations (all in `PROMPT_NOTES.md`): multi-reply with only "Continue." (Matthew: no
  single-response or output limit); every edits block applied in order; one run; Claude Code adds its identity
  system block, environment/date/org attachments, an in-turn "Output token limit hit…" recovery message
  (r1 turn 1) and a trailing newline to the prompt on resumed requests.
- Phase B stopped once on the subscription session limit ("You've hit your session limit"), after 43 tool
  calls and US$22.69; resumed with the fixed technical continuation (`v05b.c1`) on the console login.
- Phase B audit: Claude Code executed two read-only `ls -la` commands on the author dir despite the `./py`-only
  rule (auto-approved read-only commands inside the working directory); nothing outside the author dir was
  listed or read. Reads of Claude Code's own tool-result file under `~/.claude/` were refused.
- The Phase B reviewer found that v0.3's `./py` guard could be bypassed through macOS firmlink paths
  (`/System/Volumes/Data/...`); fixed for v0.5 B. Check the v0.3 transcripts before claiming sealed isolation
  for v0.3.

## Phase C (7 Nov): another Fable round on B, with 200 windows from the second generator

Matthew's design (`experiments/v0_5/phase_c/heldout/README.md`, pre-registered before launch, `d23acd3`): a NEW
Fable session (effort max, no fallback, US$250 cap) revises v0.5 B. It is handed the original 200 val windows (with
B's classes) and 200 labelled windows from the independent OOD generator (`synth_ood`, seed 20261109). These come
from 6 of its 10 families, re-timed so the clock does not give the generator away. It is given nothing from the
OOD diagnosis. It is scored once on fresh held-out sets built and hashed before launch: second generator, seed
20261110, 2,000 sessions, all 10 families; standard generator, seed 20261111, 4,000 sessions. The sealed sets were
not reused.

- **Run:** US$56.58, 114 turns, 66 min. 430 statements on top of B (999 entries changed), artefact `a7f2890c…`
  (`weight_program/v0_5/phase-c/`). Self-check on the disclosures: V 0.985, W 0.975 (B: 0.995, 0.795). Without
  our diagnosis it found two of its three mechanisms: failure identity now uses line + passed count, and plan
  repeats use `verb:argument`. It added no progress check as such.
- **Source audit before scoring** (`phase_c/SOURCE_AUDIT.md`): no source switch. Clock, gaps and token counts
  change 0 decisions, and vocabulary renames are at noise level. Three rules are tuned to the second generator
  and applied to every log: a fresh `read_file` blocks STALLED (the largest effect, knowingly traded against V134
  and V146); an append rule whose `h0` trigger only that generator emits; `verb:argument` plan keys. Compliant:
  no search or fitting. About 31 hand-written rule variants were chosen by their disclosed scores, as Phase B did.

**Held-out results** (one-time, `artifacts/reports/v05c_heldout.json`; macro-F1, FA; paired 95% CIs):

| Population | Phase C | v0.5 B | v0.3 B | v0.3 C-all |
|---|---|---|---|---|
| Held-out OOD, 4 unseen families | **0.750**, 3.0% | 0.679, 19.4% | 0.718, 4.3% | 0.349, 12.5% |
| Held-out OOD, 6 disclosed families | 0.957, 5.0% | 0.752, 39.5% | 0.883, 8.7% | 0.516, 50.3% |
| Held-out OOD, all | 0.955, 4.2% | 0.766, 31.2% | 0.906, 6.9% | 0.494, 34.7% |
| Held-out in-distribution | 0.949, 2.1% | 0.960, 3.2% | 0.954, 2.4% | 0.923, 6.4% |
| Val (excluding disclosed) | 0.947, 2.1% | 0.957, 3.2% | 0.954, 2.4% | 0.926, 5.8% |

| Pair | Unseen ΔF1 (primary) | Unseen ΔFA | OOD-all ΔF1 | In-dist ΔF1 | Val ΔF1 |
|---|---|---|---|---|---|
| C − B | **+0.071** [+0.064, +0.079] | −0.164 | +0.189 | −0.011 [−0.013, −0.009] | −0.010 |
| C − v0.3 B | **+0.031** [+0.024, +0.039] | −0.012 | +0.048 | −0.006 [−0.010, −0.001] | −0.007 |

- Both primaries are positive with CIs above 0. On the 4 families it never saw, Phase C beats both B and
  v0.3 B, with the fewest false alarms of any column. Pooled unseen macro-F1 is low for every network because
  STALLED windows there come almost only from lockfile_churn. Per family, C ≥ v0.3 B on lockfile_churn (0.667 vs
  0.592) and serializer_duel (0.950 vs 0.933), and is level on index_watermark and lease_reconnect (0.997/0.996 vs 1.000).
- In distribution, it pays a small, significant cost: −0.011 vs B and −0.006 vs v0.3 B, with fewer false alarms.
  The audit traces part of this to the fresh-read rule and the rest to RECOVERY→PRODUCTIVE drift.
- Caveat: the unseen families come from the same second generator as the disclosure, and the disclosed-family
  gains rest partly on rules tuned to it (SOURCE_AUDIT.md). The held-out OOD sets were built with the same
  re-timing as the disclosure.
- Checks: held-out hashes verified before scoring; r1 and B reproduce their val reports, v0.3 its val blocks;
  held-out sets disjoint from the disclosure and every earlier seed. Log: `phase_c/heldout/heldout_run.log`.
