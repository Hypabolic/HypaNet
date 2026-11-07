# Why v0.5 B fails off distribution (7 Nov 2026)

Post-hoc diagnosis of the sealed result (OOD macro-F1 0.765, false alarms 31.4%, against v0.3 B 0.908 / 6.8%).
**No sealed data was opened.** `analyse.py` generates a fresh OOD set from the same independent generator
(`synth_ood`, master seed 20261107, 100 sessions per family; the sealed set used 20261104) and a fresh
in-distribution set (standard generator, 40 per family), runs v0.5 B and v0.3 B, and reads v0.5 B's residual
stream at every EOE using the dimension map in `weight_program/v0_5/phase-b/CIRCUITS.md`. Window rows go to
`rows.jsonl.gz` / `rows_indist.jsonl.gz` (not committed; rerun the script, about 80 s and 20 s).

The fresh set reproduces the sealed failure: v0.5 B calls 32.8% of PRODUCTIVE windows STALLED (sealed 32.6%),
v0.3 B 4.6% (sealed 4.6%).

## 1. It is one error: PRODUCTIVE called STALLED

Sealed OOD confusion, v0.5 B: 14,347 of 44,002 PRODUCTIVE windows are called STALLED (v0.3 B: 2,041); the
other cells are close to v0.3 B's. It happens in all 10 families (12–48% false alarms per family).

## 2. The oracle never comes close to STALLED on those windows

On the fresh set's 7,175 false STALLED windows:

- 92% have real progress in the window (Δp > ε_p), which by itself rules STALLED out in the oracle.
- In only 8% (574) does the oracle count the 5 stall-ineffective events and an ineffective final event. In 61%
  it counts **none**. The network counts 5–8 on every one of them.
- v0.3 B calls 89% of them PRODUCTIVE.

So this is not a threshold or a near-miss. v0.5 B's per-event "ineffective" signal (INEFF, d75) fires on
57% of OOD events, while the oracle marks 29%. In distribution the figures are 47% against 38%.

## 3. Which rules over-fire

Each INEFF event is attributed to the Phase B rule that fired it and compared with the oracle's latent
effect for that event:

| Rule (REVISION.md §3) | Agrees with the oracle in distribution | OOD | What those OOD events actually are |
|---|---|---|---|
| **EDR**: a file change within 3 events after a repeated test result (rule 7, "scenario 6") | 84% | **28%** | 69% PROGRESS |
| **g_SAME**: a failure whose error line was seen in the last 15 events (rules 2, 7, "every scenario-5 window") | 62% | **44%** (21% below the 4th repeat) | mostly NONE |
| **g_REP**: a repeated read/search/plan/build target (rule 7) | 80% | **36%** | mostly NONE |
| REV: a reverting hunk (kept from r1) | 99.5% | 96% | |
| NOOP, POLLOUT | 100% | (rare) | |

(EDR and g_REP are counted when they fire alone; reversions excluded.)

The rules that hold up are **structural**: a hunk returns a file to an earlier state, an edit changes
nothing. The rules that break are **behavioural surface patterns** that Phase B induced from 200 windows of
the standard generator. REVISION.md names them by generator scenario ("every scenario-5 window", "the
scenario-6 windows").

## 4. Three concrete mechanisms

**(a) No progress veto.** The oracle's STALLED requires Δp ≤ ε_p and no new facts. The prompt says progress
"shows only through later validation". v0.3 B implements this with a `prog` flag (a validation reaches a new
high in its passed count), which blocks STALLED. v0.3 A already had it. r1's written replies never mention
progress, and Phase B decided not to add it (REVISION.md §5: "No progress estimate (a decreasing failure
count) was added"). In distribution the omission cost almost nothing, because windows with 5+ redundant-looking
events rarely had progress. OOD they often do.

**(b) Failure identity = first failing line.** Typical OOD false alarm (`shard_migration`, Δp +0.29, oracle
I = 0):

```
write_file contract_5   -> ok                      INEFF (EDR)     oracle PROGRESS
run_tests -> FAILED 2/14  at contract_0.clj:354    INEFF (g_SAME)  oracle NONE
write_file contract_10  -> ok                      INEFF (EDR)     oracle PROGRESS
run_tests -> FAILED 3/14  at contract_0.clj:354    INEFF (g_SAME)  oracle NONE
write_file contract_13  -> ok                      INEFF (EDR)     oracle PROGRESS
run_tests -> FAILED 4/14  at contract_0.clj:354    INEFF (g_SAME)  oracle NONE
write_file contract_11  -> ok                      INEFF (EDR)     oracle PROGRESS
run_tests -> FAILED 5/14  at contract_0.clj:354    INEFF (g_SAME)  oracle NONE
```

The passed count rises every run, but the first failing location stays the same. v0.5 B treats an unchanged
error line as the same failure (rule 2), so every run is a repeat. Because each run is a repeat, every edit
after one is "a change after a repeated result" (EDR). That makes 8 of 8 events ineffective, so STALLED.
v0.3 B sees 2 → 3 → 4 → 5 as new highs and calls it PRODUCTIVE.

**(c) Target keys see only the first word.** v0.5 B keys a plan action on its first hashed word. OOD plans are
`isolate:<path>`, `trace:<path>`, `record:<path>`, `handoff:42:0`, so different actions collide. Of the plan events flagged
as repeats, 88% are different actions OOD; in distribution 97% are exact repeats (single-word actions such as
`summarise_findings`).

## 5. Why v0.3 B generalised and v0.5 B did not

Both revisions had the same 200 windows. v0.3 B's notion of "ineffective" is closer to the oracle's
definition. A re-validation is stagnant only if nothing changed since the last one and the count is not
better. A retry needs no workspace change in between. Above all, any measured improvement vetoes STALLED.
v0.5 B fitted the standard generator's observable patterns closely: 0.995 on the 200 windows, level with
v0.3 B on val and sealed test. It did so without the veto that stops those patterns from firing when real
work is happening. Without code, r1 never wrote a progress circuit, and Phase B, starting from r1's design,
fixed what the disclosed windows showed rather than what the spec defined.

A caveat: this explains v0.5 B's gap to v0.3 B. It does not show that a progress veto would close it. No
patched network was built or scored, and none should be scored on the sealed OOD set.
