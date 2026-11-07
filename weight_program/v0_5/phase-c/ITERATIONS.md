# ITERATIONS.md — every self-check iteration of the v05 Phase C revision

All runs use `./py` from the working directory; scripts live in `output/v05-phase-c/scratch/`. "V" means the
200 windows of `disclosure/DISCLOSURE.json` (V001–V200), "W" the 200 of `disclosure/DISCLOSURE_W.json`
(W001–W200), loaded with `v05_load.windows()`. Macro-F1 figures are over those windows only; they are a
self-check, not an evaluation. No weight was ever chosen by a loop over candidates: the only loop-driven step is
`scratch/rules.py`, a Python re-statement of candidate *rules* (no weights) used to read off which rule
definitions agree with the oracle on both reports before compiling the chosen definitions by hand.

## Iteration 0 — baseline of Phase B's network

- Ran: `scratch/baseline.py` (Phase B's network, class at the last EOE of every window, confusion matrices).
- Inputs: V001–V200, W001–W200.
- Saw: the loader reproduces the disclosed `network_class` on all 400 windows (0 mismatches). V: macro-F1 0.995
  (V051 only). W: macro-F1 0.795, false-alarm 0.34; confusion (rows oracle, cols network) PRODUCTIVE 24/1/23/2,
  RECOVERY 0/41/7/2, STALLED 4/0/45/1, OSCILLATING 0/0/0/50. The 23 PRODUCTIVE→STALLED windows all have logits
  [0.5, −2, 4, 0] (sc and INEFF_t on).
- Changed: nothing yet.

## Iteration 1 — reading the failures (`scratch/dump.py`)

- Ran: `scratch/dump.py - W008 W024 W025 W033 W035` (blocks with Phase B's per-event features).
- Saw: in W every failing `run_tests` has the same error line, so SAMEg fires on each failure and EDR on every
  write after it; W008's counts rise 1/13 → 4/13 while every event is INEFF. W025's retries (3/17 ×4 after a
  write and a read) reach kpen at the second retry because the line tally counts the earlier 2/17 and 3/17 runs.
- Changed: nothing; wrote the survey.

## Iteration 2 — survey of all W windows (`scratch/survey.py`)

- Ran: `scratch/survey.py W` → `survey_W_{PRODUCTIVE,RECOVERY,STALLED,OSCILLATING}.txt` (one line per event
  with repeat / count / reversion / no-op / retry flags and Phase B's class per event); read all four files.
- Saw (the facts REVISION.md §2 rests on): constant error lines; counts as the progress signal; `verb:arg` plans
  and `handoff:N:k` runs (W055, W057, W063, W077, W131, W163, W196, W199); `h0 >` append loops (W010, W014, W028,
  W065, W099, W117 …); fresh reads only in PRODUCTIVE/RECOVERY/OSCILLATING windows; http_call hunks (W132,
  W179, W186); retries after a progress failure (W025, W131, W196 RECOVERY; W116, W174 STALLED); re-runs after
  fresh reads not retries (W053); window-relative reversions (W005, W080, W168).
- Changed: nothing; quantified next.

## Iteration 3 — descriptive statistics (`scratch/stats.py`)

- Ran: `scratch/stats.py` on V and W.
- Saw: `h0 >` mutations inside windows: V none; W 129 in STALLED, 28 in PRODUCTIVE, 3 in RECOVERY. Fresh reads
  inside windows: V 99 PRODUCTIVE / 55 RECOVERY / 7 OSCILLATING / 1 STALLED; W 69 / 13 / 15 / 0. http_call events
  with hunks: V 0, W 28. Plan shapes: V all one word; W 187 one word, 329 `verb:word`, 79 `verb:digit:digit`.
  Distinct failing lines per session: W PRODUCTIVE sessions mostly 1 (39 of 50), V STALLED sessions mostly 1 (45
  of 50) but V PRODUCTIVE sessions usually several.
- Changed: the rule design (REVISION.md §3).

## Iteration 4 — rule evaluator, first batch (`scratch/rules.py`)

- Ran: `scratch/rules.py phaseB ident_lc ident_lc_min min_only` (after fixing a stale-variable bug in the
  evaluator that had broken its resolution logic).
- Inputs: V and W.
- Saw: the `phaseB` configuration reproduces Phase B's network on W exactly (0.7948, the same 40 windows) and
  nearly on V (4 wrong instead of 1: V022, V051, V053, V059 are evaluator approximations). Identifying a failure
  by line AND count (`ident_lc`) gives V 0.985 / W 0.899 (20 wrong); approximating the exact pair by separate
  line and count tallies (`min_only`, what the network can compute) gives the same numbers.
- Changed: adopted the separate-tally approximation.

## Iteration 5 — rule evaluator, second batch

- Ran: `scratch/rules.py min_plan2 min_planboth min_planboth_gate min_pb_gate_h0 min_pb_gate_h0_fresh
  all_window all_split all_chain all_split_freshbreak`, then `C0 C_kr4 C_kr5 C_kr6 C_excl C_excl_s7 C_excl_s71
  C_excl_exact C_excl_s4`; `scratch/rules_debug.py min_planboth W063 W024 W058 W083 W025 W131`.
- Saw: keying plans on the second word alone breaks W033 (the path was just read) and loses W131/W196 through
  the tally; keying on both words fixes W063 and keeps V134. The mutation gate fixes W132-type counting. The
  append rule (with "path mutated before") fixes W028, W065, W099, W117 and keeps W061; a fresh-read STALLED
  suppressor fixes W042, W047, W058, W067, W081, W083 at the cost of V134, V146. Shortening the count tally's range
  (4, 5, 6 events) to drop a repair's first failure wrecks V's interleaved stalls (V006, V007, V009, V048, V064,
  V066 … become RECOVERY); excluding "progress failures" from the tally exactly would need a key suppression the
  network cannot compute in time. Reversion schedules: excluding the window's first event fixes W080 and W168;
  an exact window-relative count would also need per-distance lookbacks and breaks W179 when http hunks are
  dropped. Chain-length retry counting loses V009, V059, W061, W071, W073.
- Changed: chose: separate tallies, plans keyed on both words, mutation gate, append rule with the path check,
  fresh read_file suppressor, 7-event reversion sum.

## Iteration 6 — network-faithful rule set (`scratch/rules.py N4 N5 … FINAL FINAL_readonly`)

- Ran: the configurations that mirror what the weights will compute (count matches within 31 events of any
  status, a FAILED-count trigger for EDR and the `ok`-after-failure resolution, kpenC at 4 or at 5, append rule
  with or without the path check, fresh read with or without list_dir, previous-event read_file as a chain
  breaker).
- Saw: kpenC at 5 (one more count match than line match) recovers W025, W131, W196 with no V change (V 0.975
  / W 0.970 versus 0.955 at 4); the path check keeps W061; including list_dir in the fresh-read rule costs V007,
  V163, V172 (V 0.965); the read_file chain breaker fixes W053 with no V change. `FINAL_readonly`: V 0.980 (V051,
  V053, V134, V146), W 0.975 (W002, W005, W024, W079, W132).
- Changed: fixed the definitions of REVISION.md §3.

## Iteration 7 — free dims, units and heads (`scratch/freecheck.py`)

- Ran: `scratch/freecheck.py` on Phase B's state dict.
- Saw: `h0` is word id 5861; embedding columns 30, 78, 87–95, 112, 127 are zero; d30 and d112 are never written;
  d127 is written only by layer-1 MLP units 13/14 (KV); layer-0 heads 5–7, layer-0 MLP units 242+, layer-1 MLP
  91+, layer-2 MLP 13+, layer-3 MLP 8+ and layer-3 heads 3–7 are all zero; layer-2 heads 4–7 are Phase B's
  single-flag recent heads; the offset features d19 = o, d20 = o² are present for every offset.
- Changed: the dimension and unit allocation of CIRCUITS.md §1.

## Iteration 8 — first compiled revision (`scratch/gen_revision.py`, 421 statements)

- Ran: `./py -m v05_load check` (421 applied, 0 skipped), `scratch/evalrev.py REVISION.edits`.
- Saw: V 0.9445, W 0.8843: 21 windows (V004, V021, V031, V046, V078, V109, W004, W006, W022, W033 …) with
  RECOVERY logit 3.0 = a false resolution, and V179 / W179 with STALLED logits 10–16 (INEFF above 1). Cause: the
  merged recent head wrote values 3× so that a single flag reads ≥ 1 when attention splits; with r = 3 the
  resolution units `clip(2·ok + cls + r − 3)` fired without ok or the class, and EDR fired at non-mutations.
- Changed: recent values 1× (each present flag reads 1/m, m ≤ 3), RES output weights ±3 so RES ≥ 1 when it fires.

## Iteration 9 — second compiled revision (426 statements)

- Ran: check, `evalrev`; `scratch/dump.py REVISION.edits W061 W024`; `scratch/heads.py 12 W061`;
  `scratch/attn_debug.py REVISION.edits W061 13 10 1 6 7` and `… W061 13 -1 2 7 5`.
- Saw: V 0.985 (V051, V134, V146), W 0.970 (W002, W005, W024, W061, W079, W132). W061 differs from the evaluator:
  g_REPMUT = 1 at events 13 and 17 although the paths were never mutated before; the REPMUT flag was 1 on the
  *tool* token, where the class flag is 2 (+20 against the −19 bias). The same leak existed in REPg_plan.
- Changed: −20·[offset 7] in the REPMUT and REPg_plan units (E5, E6).

## Iteration 10 — third compiled revision (430 statements; the submitted one)

- Ran: check (430 applied, 0 skipped, the harness would accept it), `evalrev`; `scratch/dump.py REVISION.edits
  W025 W131 W047 V062 V154 V134`.
- Saw: V 0.985 (V051 OSCILLATING by progress, V134 and V146 PRODUCTIVE through the fresh-read rule), W 0.975
  (W002, W005, W024, W079, W132 as predicted). W025 events 10–12 are RECOVERY with INPOL = 1 (kpenL on, kpenC
  off); W131 events 15–17 likewise; W047's fresh read at 34 suppresses STALLED at 35; V062 is STALLED at 12
  (sc and INEFF), V154 RECOVERY at 10–13 (INPOL = 5, act_b = 1).
- Changed: nothing in the weights.

## Iteration 11 — synthetic sessions (`scratch/synth.py` → `scratch/synth_out.txt`)

- Ran: `scratch/synth.py REVISION.edits` on fourteen hand-built `RawOp` sessions rendered with the packet
  renderer: (1) a W-style repair (counts 1..7, one line); (2) a W-style append stall; (3) a progress failure, a
  fresh read, eight identical re-runs; (4) a progress failure, five `handoff:N:k` plans, four re-runs; (5)
  read/isolate/write of eight fresh files; (6) a V-style A/B/A/B oscillation with tests between and three 7-file
  reverts; (7) a V-style stall with re-reads between the failures, then eight fresh reads; (8) http retries on
  `localhost:8080`, a resolution, a search whose query contains `error`; (9) a V-style repair, six polls in
  policy, ready; (10) eleven pending polls; (11) V-style scenario 6 (edits, same failure); (12) repeated
  single-word plans; (13) writes of fresh files after a repeated failure; (14) edit, ok, revert, edit again.
- Saw: (1) PRODUCTIVE throughout; (2) STALLED once five ineffective events have accumulated (event 8; the
  first append of the loop is not ineffective, every later append and every unchanged failure is);
  (3) RECOVERY at the three retries after the first re-run, then not, STALLED once the fresh read leaves the
  window; (4) handoffs are not ineffective, RECOVERY at the three retries; (5) PRODUCTIVE; (6) OSCILLATING from
  the second reversion at t ≥ 7 (at t = 4 one reversion already fires because the window head sees only 5
  events; windows before block 7 are not scored); (7) RECOVERY for retries 1–3, STALLED later, PRODUCTIVE from
  the first fresh read (the W behaviour); (8) RECOVERY for the retries and four events after the resolution, the
  `error` query has no effect; (9) PRODUCTIVE through the repair, RECOVERY through the polls and after ready;
  (10) RECOVERY up to the sixth pending, STALLED once five are out of policy; (11) STALLED from the fifth
  ineffective event; (12) STALLED once five repeated plans accumulate; (13) EDR only for the writes within 3
  events of the repeated failure; (14) one reversion and its re-application give OSCILLATING, as the rule says.
- Changed: nothing.

## Iteration 12 — non-zero inventory (`scratch/nz_summary.py` → `scratch/nz_summary.txt`)

- Ran: `scratch/nz_summary.py` on the revised network.
- Saw: every non-zero block matches CIRCUITS.md §11 (tok_emb columns 0–18, 30, 32–63, 127; layer-0 MLP units
  0–79, 82–306; layer-1 MLP units 0–12, 15–136 with 13–14 zero; layer-2 heads 0–7 and MLP units 0–17; layer-3 heads
  0–3, MLP units 0–9; head rows 1–3 and the bias).
- Changed: nothing. `./py -m v05_load check` run once more as the last step: 430 statements applied, 0 skipped,
  the harness would accept the file.
