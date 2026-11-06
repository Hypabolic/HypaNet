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
v0.3 rows: `experiments/v0_3/STATUS.md`. v0.5 B against v0.3 B has not been tested as a paired difference.

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
