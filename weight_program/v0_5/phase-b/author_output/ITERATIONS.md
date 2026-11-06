# ITERATIONS.md — every self-check iteration of the v05 Phase B revision

All runs use `./py` from the working directory. Scripts live in `output/v05-phase-b/scratch/`. "The 200
windows" always means the disclosed report `disclosure/DISCLOSURE.json` (V001–V200), loaded with
`v05_load.windows()`. Macro-F1 figures are over those 200 windows only; they are a self-check, not an
evaluation.

## Iteration 0 — baseline of r1's network

- Ran: `scratch/baseline.py` (r1's network, class at the last EOE of each window, confusion matrix).
- Inputs: V001–V200.
- Saw: the loader reproduces the disclosed `network_class` on all 200 windows (0 mismatches), so the local
  runs agree with the harness. Confusion (rows oracle, cols network): PRODUCTIVE 3/0/47/0, RECOVERY 4/0/46/0,
  STALLED 14/0/36/0, OSCILLATING 0/0/12/38. Macro-F1 0.331, false-alarm rate 0.93. STALLED logits of
  ~60–90 everywhere; RECOVERY never predicted.
- Changed: nothing yet; went to inspect activations.

## Iteration 1 — locating the defect (`scratch/inspect.py` on V001)

- Ran: `scratch/inspect.py V001 - 2` (per-event features at every EOE and per-token values of the last two
  events, r1's dimension names).
- Saw: `MATCHtok` (d118) ≈ 6.33 on every hashed or digit token, `REPg` ≈ 8.7 and `CNTg` ≈ 5.3 on every token,
  `INEFF` ≈ 10 at every EOE. Cause: r1's layer-0 MLP "target code" units 32–63 use bias −2 with three gates
  plus the code bit, so at a read path (offset 10) or search word (offset 14) all 16 code dimensions fire
  (sum 19 instead of a 3-hot code); every later code then "matches" these keys.
- Changed: wrote `scratch/rev_a.edits` with the single statement `layers.0.mlp.fc1.bias[32:64] = -3`.

## Iteration 2 — the one-line fix (`scratch/rev_a.edits`)

- Ran: `scratch/evalrev.py scratch/rev_a.edits v`.
- Inputs: V001–V200.
- Saw: macro-F1 0.714 (PRODUCTIVE 35/14/0/1, RECOVERY 6/42/2/0, STALLED 28/4/18/0, OSCILLATING 0/0/0/50),
  false-alarm rate 0.03. Remaining errors: STALLED→PRODUCTIVE (28), PRODUCTIVE→RECOVERY (14),
  RECOVERY→PRODUCTIVE (6), STALLED→RECOVERY (4).
- Changed: nothing in the weights; started reading the windows to understand the oracle (iterations 3–5).

## Iteration 3 — reading the windows (`scratch/dump.py`, `scratch/survey.py`)

- Ran: `scratch/dump.py scratch/rev_a.edits` (blocks of every window with r1's event features) and
  `scratch/survey.py` (one compact line per event with observable flags: repeated target, repeated count,
  reverting hunk, no-op mutation, retry counter).
- Inputs: V001–V200 (read the STALLED, RECOVERY and PRODUCTIVE dumps in full).
- Saw (the facts the revision is built on; window ids in REVISION.md):
  - redundancy is judged against the whole session, not the window (V009, V119, V172 end on a re-read or
    re-search of something seen 10–18 events earlier);
  - repeated identical failures with ≤3 retries are RECOVERY (V154, V157, V174, V176), ≥4 retries are
    STALLED (V006, V037, V062, V066); edits between the failures do not reset the count (V037, V062);
  - the failure identity is the error location (the line number is a stable hash of it): V062 has
    `FAILED 4/11` then `FAILED 0/1` ×4 at the same `:670` and is STALLED; V003/V029 have the same test file
    failing with different tests and are PRODUCTIVE;
  - a retry is a failure repeated with no mutation in between (V029: `FAILED 4/8` then a focused
    `FAILED 0/1` two events later is not a retry; V088: `FAILED 0/3` twice in a row is);
  - "edits that change files but do not improve validation" (scenario 6) are STALLED with the edits
    themselves counted as ineffective (V027, V040, V060, V063, V079, V091, V099, V104, V107, V116, V118, V139,
    V158); the observable signature is the unchanged pass count around the edits;
  - a resolution (ok/ready after a failure of the same kind) within the last 4 events is RECOVERY
    (V088, V115, V124, V129, V131, V137, V160, V175, V182, V186, V189, V192, V193, V199, V200), and a
    resolution 4+ events back is not (V004, V031, V046, V077, V078, V097, V047, V085);
  - a first failure at the end of a window is not RECOVERY (V005, V019, V046, V085, V097, V105, V108);
  - FAILED → edit → ok without an identical retry is not a resolution (V008, V016, V052);
  - pending polls are in policy up to the sixth (V030 ends on the sixth pending and is RECOVERY), beyond
    that they are ineffective (V044, V163);
  - repeated plan actions count as redundant (V134).
- Changed: the design (iterations 6+).

## Iteration 4 — feature tables (`scratch/features.py`, `scratch/features2.py`)

- Ran: both scripts over V001–V200, printing per-window counts of candidate observable flags and a check of
  the candidate decision rule (reversions ≥ 2 → OSCILLATING; else an active in-policy retry → RECOVERY;
  else ≥5 ineffective events with an ineffective final event → STALLED; else a resolution within 4 events →
  RECOVERY; else PRODUCTIVE).
- Saw: with the error-location *file* as the retry key the rule misses 12 windows (V003, V015, V021, V029
  ... PRODUCTIVE windows look like active retries); with the *line number* plus the pass count and the
  "no mutation in between" condition, the rule misses only V051, V053 (reversions with real progress), V071
  (final `lint` repeat), V142 (a single retry the oracle does not treat as in policy) and V078/V022
  (resolution-distance details that the network's own definition handles).
- Changed: fixed the per-event definitions used in the network (REVISION.md §3).

## Iteration 5 — first full revision (`scratch/gen_revision.py`, 1030 statements)

- Ran: `./py -m v05_load check` (1030 applied, 0 skipped), `scratch/evalrev.py REVISION.edits v`.
- Inputs: V001–V200.
- Saw: macro-F1 0.945 (PRODUCTIVE 41/8/0/1, RECOVERY 0/50/0/0, STALLED 0/2/48/0, OSCILLATING 0/0/0/50).
  Wrong: V003, V004, V021, V031, V046, V059, V078, V109 (PRODUCTIVE→RECOVERY, RECOVERY logit 3 from the
  resolution path), V051 (→OSCILLATING), V119, V140 (STALLED→RECOVERY).
- Inspected with `scratch/inspect2.py V003`, `V119`, `V140` and `scratch/attn_debug.py V119 1 2 19 LINE`:
  - the resolution units multiplied the "recent flag" value (an average over flagged tokens) by 6, so with
    the value 1 the unit fired without the ok/class conditions (V003 event 19: a read became a resolution);
  - the repeat key looked only 15 events back; V119's final `.editorconfig` was read 18 events earlier and
    V140's final read 16+ events earlier;
  - the retry count was 0.807 instead of 0.800 for k = 4 because a second gather head (keyed on hunk/target
    flags) also wrote the KV dimension and leaked a 1/n share when it had nothing to attend to; the
    attention of the counting head itself was exact (4 keys at 0.1974, BOS 0.2105).
- Changed: one gather head per token type; four single-flag "recent" heads (binary outputs); a second code
  (B) for target keys with the repeat range 31; the status broadcast moved to layer 0 to free a head;
  doubled queries of the previous-token heads so the copied flags are exactly 1.

## Iteration 6 — second full revision (1258 statements)

- Ran: check (1258 applied), `evalrev`.
- Saw: macro-F1 0.980. Wrong: V051 (OSCILLATING), V119, V128, V142 (STALLED→PRODUCTIVE, STALLED logit −2).
- Inspected V119: the final read is `.editorconfig`, tokenised as `.` + `editorconfig`; the `.` token had a
  code A but no code B, so the two-code repeat test could not fire. V128 ends on `list_dir path=.`.
- Changed: code B for token id 141 (`.`): `tok_emb.weight[141, 48] = 1`, `tok_emb.weight[141, 62] = 1`.

## Iteration 7 — third full revision (1260 statements)

- Ran: check, `evalrev`.
- Saw: macro-F1 0.995; only V051 wrong (PRODUCTIVE with three reversions and an improving test count: the
  oracle's progress rule, which the network cannot see).
- Changed: nothing on the windows; ran synthetic checks (iteration 8).

## Iteration 8 — synthetic sessions (`scratch/synth.py`, `scratch/debug_resw.py`)

- Ran: `scratch/synth.py REVISION.edits` on five hand-built `RawOp` sessions rendered with the packet
  renderer: (1) A/B/A/B edits of one file with tests between, then three 7-file reverts (a `@ TRUNC` line);
  (2) the same failure 20 times with re-reads between, then 8 fresh reads; (3) three http errors on
  `localhost:8080`, an ok call, a read/edit/test, and a search whose query contains the word `error`;
  (4) a productive repair (2/9 → 5/9 → 7/9 → 9/9) then six pending polls and a ready; (5) nine pending
  polls of one resource.
- Saw: (1) OSCILLATING from the second reversion, including on the multi-file reverts; (2) RECOVERY for
  retries 1–3, STALLED from the fourth retry, PRODUCTIVE after the fresh reads; (4) PRODUCTIVE through the
  repair, RECOVERY during the polls and after the ready; but (3) the third http error was PRODUCTIVE: the
  header `https://localhost:8080` contains a digit after `:` preceded by a word, so each event carried two
  line keys and the retry count doubled; and (5) `resw` was 0.5 with no resolution anywhere.
- `scratch/debug_resw.py`: the resolution unit fired at the `poll_status` tool token, where the in-place
  class broadcast makes the class flag 2 (`0 + 2 + 1 − 2 = 1`), and the window head averaged those tokens in.
- Changed: the line key is gated on the status broadcast (only body tokens after `->` of a failing event
  see it) instead of on "previous-but-one token is a word"; the ok/ready flag has weight 2 in the
  resolution units; the unused "prev2 hashed" copy was removed.

## Iteration 9 — final revision (1262 statements)

- Ran: check (1262 applied, 0 skipped, the harness would accept it), `evalrev`, `synth.py`.
- Saw: V001–V200: macro-F1 0.995, accuracy 0.995, false-alarm rate 0.01 (V051 only). Synthetic sessions:
  all five behave as intended (the http retries are RECOVERY up to the third error, the resolution is
  RECOVERY for the next four events, the search query containing `error` has no effect; the polls beyond the
  sixth are ineffective but, with fewer than five ineffective events, PRODUCTIVE).
- `scratch/nz_summary.py`: listed every non-zero block of the revised network for CIRCUITS.md.
- Changed: nothing further. This is the submitted revision.
