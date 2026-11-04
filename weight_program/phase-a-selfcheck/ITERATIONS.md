# ITERATIONS.md — Phase A-selfcheck (spec v0.2, §9.3)

Base: the archived A-strict deliverables (archive sha256
`0b019e9d57e391ff4e2e7cc49bd554393d991066f5b02ec83ee8c95eba3397f7`).
No data, labels or evaluation output were used. No automated weight search was run.
All scripts live in `output/selfcheck_scripts/`; `build_weights.py` was never executed
for a forward pass.

## Iteration 0 — copy of the A-strict deliverables

- **Diff:** none. `copy_strict.py` copied `reference_policy.py`, `build_weights.py` and
  `CIRCUITS.md` byte-for-byte (sha256 prefixes `db8fb337…`, `00a9a543…`, `8f5119ad…`,
  verified equal to the A-strict files).
- **Reason:** required starting point for this phase.
- **Inputs used:** none.

## Iteration 1 — network vs policy self-check (`selfcheck.py`)

- **Diff to weights / policy:** none. The only edit in this phase is the title line of
  `CIRCUITS.md`, which now names the phase and states that the circuit is unchanged.
- **Reason:** every comparison agreed, and every hand-written window produced the class the
  policy is meant to produce, so there was nothing to fix. The design trade-offs were
  re-examined by reasoning (below) and kept.
- **Inputs used (all constructed in the script):**
  1. 35 hand-written windows, one or more per scenario family of spec §7.1 plus edge cases:
     F1 repair (improving; early with no delta), F2 exploration, F3 transient external failure
     (mid-retry; resolved at the endpoint; resolved at slot 4; resolved at slot 3), F4 diagnosed
     repair via build and build retries (3 in policy, 4 out of policy), F5 repeated failing tests
     (full window; interleaved re-reads; 3 repeats; 2 repeats), F6 repeated edits (same edit
     re-applied; novel variants with the same failure), F7 A/B/A/B oscillation (with and without
     net progress), F8 tool oscillation (full; starting late), a single reversion, F9 repeated
     runs with improvement, F10 polling (neutral polls in flight; polls then success; 5 and 6
     error polls; re-reads while waiting), F11 rising-cost re-reads, all-novel inspects, repeated
     neutral plans, stall evidence with a novel endpoint, stall evidence with a fresh external
     error, redundant reads with in-policy retries, a first-of-session test failure, and the
     all-zero input. Each was given the class my policy is intended to return.
  2. Four exhaustive enumerations of all 5^8 = 390,625 windows over per-slot vocabularies
     (slot 0 forced to repeated_intent = repeated_failure = 0):
     A stall/retry mix {novel read, repeated read, repeated external error, repeated comparable
     test failure, reverting edit}; B recovery/resolution {novel read, external error, repeated
     external success, comparable passing run +0.05, repeated neutral poll}; C
     progress/oscillation {edit, reverting edit, comparable failing run −0.25, repeated
     comparable failing run +0.25, repeated build failure}; D neutral/control/mutate {repeated
     plan, neutral search, repeated edit, repeated successful build, repeated failed edit}.
  3. A random valid-window sweep, n = 300,000, numpy seed 20261104: uniform category and
     outcome one-hots, repeated_intent Bernoulli(0.5), repeated_failure = ERROR·Bernoulli(0.6),
     workspace_changed Bernoulli(0.5), state_revisited = changed·Bernoulli(0.4),
     validation_comparable = VALIDATE·Bernoulli(0.7), validation_delta = comparable·U(−1, 1)
     with 20 % exact zeros, token cost U(0, 1); slot 0 repeats forced to 0; all values float32.
  4. A progress-ramp probe: net validation delta swept over [0.0170, 0.0180) in steps of 1e-5,
     placed on the endpoint run, in an oscillation context and in a stall context.
  5. Batch consistency: 5,000 random windows (seed 7) at batch size 1 versus batched.
- **Results:**

  | Input set | Windows | Network ≠ policy |
  |---|---:|---:|
  | Hand-written scenarios | 35 | 0 (and all 35 equal the intended class) |
  | Enumeration A | 390,625 | 0 |
  | Enumeration B | 390,625 | 0 |
  | Enumeration C | 390,625 | 0 |
  | Enumeration D | 390,625 | 0 |
  | Random valid sweep | 300,000 | 0 |
  | Ramp probe (2 contexts) | 200 | 0 |
  | Batch 1 vs batched | 5,000 | 0 |

  The policy is written as the exact arithmetic of the circuit, so agreement holds inside the
  1e-4-wide progress ramp as well; only float32-versus-float64 rounding at an exact ramp edge
  could separate them, and none of the probe points did.

- **Design points re-examined and kept (reasoning only, no inputs):**
  - Repeated identical failures add one point of stall evidence on top of the repeated intent.
    This is what lets "repeated edits that change files but do not improve validation" (§7.1
    family 6) reach the stall threshold when the edits themselves are novel variants; the cost is
    that a window ending exactly at the third identical failure of a retried operation is called
    STALLED. The spec defines re-running validation on an unchanged workspace as REDUNDANT, so
    such windows are expected to be grey-zone (unscored) rather than RECOVERY.
  - ERROR outcomes of comparable validation runs are treated as task-level failures, not
    episodes. The alternative (treating them as episodes) would turn most incremental-repair
    windows into RECOVERY; the other alternative (ignoring all VALIDATE errors) would turn
    in-policy build retries into STALLED false alarms.
  - Episode activity is read from the last 4 events and resolution from the last 4 events
    (`k_rec = 4`, endpoint included); an older unresolved error does not veto a stall.
  - The progress threshold stays at 0.0175, strictly between 1/60 and 1/50 so that the ramp
    contains no common pass-fraction step.
  - `relative_token_cost` and `workspace_changed` remain unused: cost is weak evidence by spec,
    and a workspace change is not progress.

## Final state

`reference_policy.py` and `build_weights.py` are byte-identical to the A-strict archive;
`CIRCUITS.md` differs only in its title line. `tools/shape_check.py` on this directory passes
with the same sparsity (233 non-zero slots of 3,256).
