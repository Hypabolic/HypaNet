# v0.5 status (6 Nov 2026)

v0.5 applies the premise of Experiment 5 of "Sparks of Weight-Space Programming" (Takehi, Teoh et al.,
MIT CSAIL, 2 Nov 2026) to the doom-loop task: Fable writes the weights of the v0.3 transformer with **no
tools**, by reasoning alone, in the `edits` language. Then, as in v0.3 Phase B, a new Fable session improves
that network with tools and the same 200 labelled val windows.

## Results (validation, excluding_disclosed: 3,804 sessions, 69,974 scored windows)

| Network | How it was made | Labels | Macro-F1 [95% CI] | False alarms | Cost |
|---|---|---|---|---|---|
| **v0.5 r1** | Fable, no tools, 4 replies (1 prompt + 3 "Continue.") | 0 | **0.321** [0.313, 0.328] | 95.8% | US$16.70 |
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
  Turn 1 was ~233k thinking tokens (two 128k-capped messages) before any text.
- **Phase B:** found one wrong bias in r1 (layer 0 MLP units 32–63, −2 instead of −3) that made the
  target code fire everywhere, then rebuilt layers 1–3 around the oracle's rules (1,262 statements, 2,813
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
