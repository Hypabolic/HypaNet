# REVISION.md — Phase B, one diagnostic revision (spec v0.2, §9.4)

## What was disclosed

- File: `disclosure/DISCLOSURE.json`, sha256
  `484f885a1cc94113e19fa7016ad739630c5cc7b959314774bbe534610349c3b6`
  (verified by `scratch/copy_selfcheck.py`; also in `disclosure/DISCLOSURE.sha256`).
- Contents: 200 windows from the validation split, stratified 50 per oracle class, with
  opaque ids `D001`–`D200`. For each window exactly four things: the 8 × 15 feature
  matrix (with feature names), the oracle label, the archived A-selfcheck network's class
  (artefact sha256 `561366300f9d84f6d81adc88f17daf4eeafd69f9206efb30614a52caa6c615ad`),
  and the archived A-selfcheck `reference_policy`'s class.
- Nothing else was used: no other windows, data, labels or metrics. The base was the
  archived A-selfcheck deliverables, copied byte-for-byte by `scratch/copy_selfcheck.py`.

## Baseline on the disclosed set (archived A-selfcheck files, `scratch/analysis_base.txt`)

139/200 correct, macro-F1 0.695, false-loop-alarm rate 4.0 % (4 of 100 PRODUCTIVE +
RECOVERY windows predicted STALLED). Confusions: STALLED→PRODUCTIVE 27, RECOVERY→PRODUCTIVE 13,
PRODUCTIVE→RECOVERY 11, RECOVERY→STALLED 4, STALLED→RECOVERY 3, OSCILLATING→PRODUCTIVE 3.
The archived network and policy agreed on all 200 windows, so every error was a policy error.

## The revision (policy and weights together; `reference_policy.py`, `build_weights.py`, `CIRCUITS.md`)

Each change below was made by reading the disclosed windows against the oracle rules of spec
§6.3 and re-deriving the observable proxy; no weight was searched or fitted. The network was
rebuilt from the revised policy by literal assignment and then checked for agreement.

1. **Stall evidence redefined** (circuit A, layer-2 unit `STALL_A`).
   - Per-slot redundancy `r_s = relu(RI + RF + SR - NEU - 3*EXT)` over slots 1–7 replaces the
     old score; EXTERNAL events are excluded entirely (their repeats are retries, polls or
     resolutions). Threshold lowered from 5 to **2**; the "endpoint must be ineffective"
     condition is **removed**; retries beyond policy (`OOP`, ≥ 4 repeated failures) count as
     stall evidence on their own.
   - Reason: the oracle's REDUNDANT count includes hidden duplicates (re-reads with different
     line ranges, edits already reflected in the workspace, novel-looking closing events), so
     the within-window repeat flags are only a lower bound. STALLED windows with 2–3 visible
     repeats: D001, D060, D070, D111, D130, D155; edit/run chains ending in a novel edit or
     read: D035, D046, D050, D065, D075, D078, D084, D088, D127, D132, D140, D148, D151,
     D152, D154; repeated green re-runs with zero delta: D061, D109, D131; D108, D123. The
     EXTERNAL weight of −3 (not −1) comes from D021, a PRODUCTIVE window whose two external
     retries would otherwise score one point each (a false alarm in my first draft).
2. **No-change condition for STALLED now looks at both directions** (units `PROG`, `NZ`).
   A net regression (`S <= -0.1`) vetoes STALLED like an improvement does: D031 and D047 are
   PRODUCTIVE windows with repeated failures but changing test results.
3. **OSCILLATING no longer vetoed by net progress** (layer-2 unit `OSC = relu(VM1)`).
   D033, D139, D170 are oscillations whose window boundary leaves a positive net delta; the
   oracle's `Δp <= ε_p` is latent and an A/B/A/B cycle never makes progress.
4. **RECOVERY: an active episode needs an actual retry** (circuit C, unit `ACT`).
   - Old rule: any non-comparable ERROR in the last 4 events. New rule: a repeated identical
     failure in the last 4 events (`RT`), fresh-failure evidence (`NF > 0`: a suite's first
     failing result or a non-validation error, or a recent regression; zero for a chain that
     merely repeats a pre-existing comparable failure, or after recent improvement), and at
     most 3 repeated failures in the window (`OOP = 0`).
   - Reason: lone errors are the normal start of repair work, not episodes — D004, D013,
     D015, D022, D040, D042, D044, D063, D083, D096, D119 are PRODUCTIVE. Fresh chains with
     immediate retries are RECOVERY — D091, D185, D192, D195 (first run non-comparable), D194,
     D199 (regression then retries) — while chains repeating a pre-existing failure are
     STALLED — D050, D090, D092, D101, D148. The improvement term keeps D006 and D094 out of
     the retry branch; the retry bound keeps D055, D085, D103, D151, D152 out of RECOVERY.
5. **Resolution detector split and extended** (circuit B).
   - `resA`: successful repeat of an EXTERNAL call or non-comparable VALIDATE op after an
     earlier ERROR **or NEUTRAL** event: poll runs that come good — D028, D110, D115, D121,
     D161, D181, D200 — and transient failures retried once — D180, D182.
   - `resB`: comparable passing re-run after an earlier repeated failure — D124, D133, D196.
     Without a prior repeated failure a passing re-run is ordinary repair (D044, D119, D018).
   - MUTATE removed from the resolution set: re-applied edits are stall evidence (D030).
6. **Polling** (circuit E). In-flight polling is RECOVERY only while the run is within policy:
   ≤ 6 neutral events and the run did not already fill slot 0. Over-polling is STALLED —
   D077, D118, D097 — while D141, D145, D187, D191, D197 stay RECOVERY.
7. **Output precedence is now OSCILLATING > RECOVERY > STALLED > PRODUCTIVE.** RECOVERY
   signals veto STALLED (the oracle's "no episode in policy at the endpoint", and retries are
   IN_POLICY_RETRY rather than REDUNDANT): D107, D189 contain retry chains whose repeats would
   otherwise count as stall evidence.

Result on the disclosed set (`scratch/analysis_rev2.txt`): **195/200**, macro-F1 0.975,
false-loop-alarm rate 1.0 %; network = policy on all 200.

| Oracle \ predicted | PROD | REC | STALL | OSC |
|---|---:|---:|---:|---:|
| PRODUCTIVE | 47 | 3 | 0 | 0 |
| RECOVERY | 0 | 49 | 1 | 0 |
| STALLED | 1 | 0 | 49 | 0 |
| OSCILLATING | 0 | 0 | 0 | 50 |

Remaining errors, each accepted deliberately:
- D027 (STALLED → PRODUCTIVE): one re-read plus a pre-existing failure; a threshold of 1 on
  redundancy would alarm on any single repeat.
- D142 (RECOVERY → STALLED): a chain that repeats a pre-existing comparable failure, which
  the other five such windows label STALLED.
- D006 (PRODUCTIVE → RECOVERY): a repeated failure with partial improvement followed by a green
  re-run; excluding it would need a per-slot "repeat without improvement" unit that the layer
  budget does not allow, or would cost D018 and D117.
- D047, D093 (PRODUCTIVE → RECOVERY): fresh failure chains re-run after edits; the retry-versus-
  repair distinction needs ordering information the window sums do not carry.

## Left unchanged, and why

- `relative_token_cost` and `workspace_changed` remain unused: cost is weak evidence by spec,
  and the high-cost stall windows (D060, D155) are caught by redundancy.
- The resolution window stays at the last 4 events: D115, D177, D183, D196 (resolved at slot 4,
  RECOVERY) against D021, D057, D072, D117 (resolved at slot 3, PRODUCTIVE) confirm `k_rec = 4`
  means slots 4–7.
- OSCILLATING threshold `V >= 2` and the progress threshold 0.02: no disclosed window argued
  against them.
- The first-index tie rule and PRODUCTIVE as the default class.
- The compiled form: the policy remains an exact transcription of the circuit.

Generalisation risks I am aware of: the redundancy threshold of 2 will alarm on a productive
window that happens to repeat two intents (none of the 50 disclosed PRODUCTIVE windows repeats
any non-external intent); treating a net regression as a fresh failure makes a regression
followed by retries RECOVERY rather than STALLED.

## Verification (`scratch/fidelity.py`, `tools/shape_check.py`)

- Network vs policy: 0 disagreements on 5 exhaustive enumerations of 390,625 windows each
  (1,953,125), a seeded random valid-window sweep of 300,000 (numpy seed 20261105), 1,000
  analogue-gate probe points and a 5,000-window batch-size check.
- Shape check passes: 295 non-zero slots (fc1.weight 204, fc1.bias 15, fc2.weight 55,
  fc2.bias 6, fc3.weight 15, fc3.bias 0).
- The scratch folder holds the analysis scripts and their outputs (`analysis_base.txt`,
  `analysis_rev1.txt`, `analysis_rev2.txt`).
