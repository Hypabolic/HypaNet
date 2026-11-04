# Independent rules baseline

This is a pre-registered, hand-written policy using only the 120 features in
`feature_index.json`, with eight oldest-first blocks of 15 (§4). It reads no
files at inference time, uses no fitted parameters, and returns the canonical
indices PRODUCTIVE=0, RECOVERY=1, STALLED=2, OSCILLATING=3. The caller supplies
valid encoded windows after the instrumentation gate (§5.2). Lengths other than
120 raise ValueError; semantic ambiguity never causes abstention.

## Observable proxies

- Progress: sum comparable validation deltas over all eight events; progress
  is strictly greater than 0.02. This borrows epsilon_p from §6.2, but validation
  ratios are only a proxy for objective completion. Negative deltas cancel
  positive ones. No tolerance or rounding is added. Workspace changes, SUCCESS,
  and token cost alone never establish progress (§4).
- Reversions: count MUTATE events with state_revisited=1. Requiring mutation
  excludes observed external resets; it cannot prove an earlier agent mutation
  was undone (§6.1).
- Knowledge: count non-ERROR INSPECT events with repeated_intent=0. Any such
  event vetoes STALLED, approximating its F=0 requirement (§6.3). Two or more
  are compatible with F_prod=2, but both one and zero ultimately default to
  PRODUCTIVE when no other rule applies. Novelty cannot establish new facts.
- Stall-ineffective: a mutation reversion, or repeated_intent/repeated_failure
  without a strictly positive comparable validation delta. Count each event
  once. Changed files do not exempt repeated operations. Novel failed edits
  without repeat/reversion evidence are not counted (§4, §6.1).

## Episode approximation

Scan oldest to newest, tracking one active failure category and a retry count.
An ERROR starts an episode at count zero if none is active or its category
differs. Another ERROR in the active category increments the count only when
repeated_intent=1. A non-error repeated operation in that category increments
the count, except that repeated SUCCESS resolves the episode. Any strictly
positive comparable validation delta also resolves an active episode. Record
the most recent resolution index. A later ERROR starts a new episode. Events
in other categories otherwise leave the active episode unchanged.

The endpoint episode is in policy at counts <=3, or <=5 for EXTERNAL, borrowing
R_max and P_max (§6.2). The initial failure counts as zero retries (§6.1).
Additionally, a trailing uninterrupted run of neutral EXTERNAL events is
treated as active polling if it contains 1 through 5 repeated_intent events.
The first novel poll is treated as the initial request. There is no backoff
feature, so backoff is not verified. Counts restart at the window boundary.

## First-match precedence

1. **OSCILLATING:** at least 2 mutation reversions and no proxy progress.
   N_rev=2 and the progress tolerance follow §6.2–6.3. This overrides recovery
   and novel investigation, as the oracle does.
2. **STALLED:** no proxy progress, zero novel non-error inspections, at least
   5 stall-ineffective events, final event stall-ineffective, and no endpoint
   in-policy episode. N_stall=5 and the endpoint guards follow §6.3. A recent
   resolution alone does not veto this rule, matching oracle precedence.
3. **RECOVERY:** endpoint episode is in policy, or latest resolution occurred
   at index 4, 5, 6, or 7. This is the last k_rec=4 events (§6.2–6.3).
   Recovery overrides even positive proxy progress.
4. **PRODUCTIVE:** otherwise. This covers measured improvement, novel
   investigation, and all forced decisions on ambiguous windows. It is the
   benign default (§2.2, §5.2); §6.3's semantic INSUFFICIENT_EVIDENCE cannot be
   returned under this commission's four-class interface.

## Expected errors and independence

The oracle uses hidden causal state (§6), so these are approximations, not
equivalent oracle rules. Repeated reads can teach new facts; novel reads can
be useless. Different novel edits can repeatedly fail without triggering the
stall proxy. Category-level episode tracking conflates unrelated intents and
failures, cannot represent concurrent episodes, and cannot distinguish
persistent failures from transient or diagnosed ones. A first visible failure
may already exceed its latent retry budget. Poll counts can mix resources.
SUCCESS can fail to resolve the underlying cause, while a novel successful
repair without validation may go unrecognized. Old unresolved errors may
overextend recovery. Successful polls without a visible failure are not
recognized as resolution.

Validation deltas from different suites can cancel or double-count and do not
measure hidden progress; improvement can coexist with a real loop. State
revisits lack causal provenance. History before the window is absent except
for comparable validation deltas. These limitations can produce false loop
alarms and missed loops, including feature collisions (§6.4). Token cost is
ignored because expenditure alone is not a label (§7.1). Thresholds come
solely from the spec and the explicit initial-request convention above.

Only the authoring packet was consulted. Tests are individual hand-constructed
branch and boundary checks, not a generated dataset or threshold tuning.
