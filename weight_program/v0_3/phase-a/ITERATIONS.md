# ITERATIONS.md: Phase A self-check log

All inputs are logs I wrote myself as `RawOp` sequences rendered with `packet/renderer.py`
(`scratch/common.py` holds the op factories; the sessions are in `scratch/check1.py`,
`scratch/check2.py`, `scratch/check3.py`). No other data was used; every change was a direct,
reasoned edit. Agreement figures are argmax(network) vs `reference_policy` at every `<EOE>` of
every prefix of each session.

## Iteration 0: design checks before any forward pass

- **Inputs:** none (pure arithmetic, `scratch/kernel_check.py`).
- **What:** chose the within-block offset code used for exact relative-offset lookups. Candidate
  frequency sets were evaluated for the worst off-target kernel value over `|d| <= 191`;
  `w_i = pi / 1.5^i` with 7 frequencies gives K(0) = 7 against a worst miss of 5.27 (at d = 162).
- **Diff:** the first draft of `build_weights.py` assumed a worst miss of 4.4 and `A_OFF = 20`;
  re-centred the sink to (7 + 5.3)/2 * A and raised `A_OFF` to 30 so both margins are >= 26.
- **Also:** read the author guard in `tools/` to pin the exact static and runtime rules.

## Iteration 1: first full comparison

- **Inputs:** `scratch/check1.py`, 14 sessions, 122 windows: incremental repair; productive
  exploration; transient http failure then success; bounded poll backoff; 9 identical failed test
  runs; failed runs interleaved with re-reads; distinct edits with unchanged validation; A/B/A/B
  edits; edit/revert cycles; benign improving runs; 9 pending polls then ready; 10 http errors;
  8 identical reads; plan / write / five no-op edits / ask.
- **Result:** 0 / 122 disagreements on the first build. Feature traces (SI, INPOL, RESOLV,
  REVBLK, counts, anchors, PV, decision bits) matched the designed values exactly: e.g. the
  anchor count `OUTALL = C/(1+C)`, STALLED from the 6th identical run, OSCILLATING from the third
  A/B edit, RECOVERY for the 4 blocks after a resolved http retry.
- **Diff:** none to the weights.

## Iteration 2: RECOVERY after an exhausted retry budget (policy refinement + circuit change)

- **Reason:** the traces of the long-poll and many-http-errors sessions showed RECOVERY persisting
  for three blocks after the retries had gone out of policy, because the rule was "any in-policy
  failure or resolution among the last four blocks". The oracle's rule 4 needs the episode to be
  in policy *at the endpoint*, so an out-of-policy retry should end the recovery evidence.
- **Diff (policy):** `_relational` also returns `kill` (= retry beyond policy); the window rule
  scans `t, t-1, t-2, t-3` and takes the first block that is in-policy / resolution / kill, and
  RECOVERY holds iff that block is not the kill. Constants renamed to `RETRY_LIMIT = 4`,
  `POLL_LIMIT = 5`.
- **Diff (weights):** new residual dim KILL (110) written by the two out-of-policy units of MLP 2;
  head H3.1 changed from an anchor average over distances 0..3 to a nearest-of-type head
  (gate = EOE and INPOL + RESOLV + KILL, bias `-20c`, sink `-80`, value `INPOL + RESOLV - KILL`);
  decision unit A4 = `clip(40 * NEAR4 - 20)`; dims ANY4 / ANCH4 retired.
- **Inputs:** `check1.py` again (0 / 122) and the new `scratch/check2.py`, 8 stress sessions,
  89 windows: a 40-event mixed session crossing several 128-row query blocks; 7-file edits with
  `@ TRUNC` lines and multi-path reverts; dropped observations and `[workspace untracked]`;
  3-digit passed counts; read errors with retries and a late success; create / delete cycles
  (`h0` must never match); poll error then pending then ready; no-op edits mixed with repeated
  searches.
- **Result:** 0 / 89 disagreements. `scratch/build_check.py`: build 0.67 s CPU, all tensors
  finite and float32 in template order, two in-process builds byte-identical.

## Iteration 3: long and extreme inputs, static self-audit

- **Inputs:** `scratch/check3.py`, 5 sessions, 170 windows: a 139-block session (all family
  sessions back to back, 4,037 tokens); maximal blocks (30-test `tests=` list truncated in the
  header, 7 changed files with the error line, 6 body lines); events with a missing timestamp
  (`?` shifts the header by six offsets, so tool and slot reads are garbage for both network and
  policy, but they must agree); reserved words (`passed`, `error`, `ok`, `FAILED`, `pending`,
  `ready`) inside search queries and action names; 12-path reverts truncated with `TRUNC`.
- **Result:** 0 / 170 disagreements; `reference_policy` takes 10.6 ms CPU per call on the
  139-block prefix (it only reads the last 15 blocks).
- **Diff:** `scratch/static_audit.py` scans both deliverables for the literal-size, dunder and
  name rules. Findings and fixes: the policy's module docstring was 3,524 characters, moved to
  `CIRCUITS.md` section 1 and replaced by a short summary (trimmed twice, to 458 characters);
  `re.compile` replaced by `re.match` on a pattern constant (avoids the name `compile`); two
  comments in `build_weights.py` mentioning the `@` marker reworded. No behavioural change; all
  three suites re-run on the final files (0 / 122, 0 / 89, 0 / 170), audit clean (longest string
  literals 482 and 458 characters, largest sequence literal 34 elements, only the `"\n@"` surface
  string contains an `@` in the policy).

## Final state

- Network vs `reference_policy`: 0 disagreements on 381 windows over 27 hand-written sessions.
- Build: deterministic, 0.67 s CPU, 121,422 non-zero parameters.
