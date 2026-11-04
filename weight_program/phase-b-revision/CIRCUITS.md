# CIRCUITS.md — Phase B revision (spec v0.2)

Network: `Linear(120, 24) -> ReLU -> Linear(24, 12) -> ReLU -> Linear(12, 4)`.
Input slot `s` (0 = oldest, 7 = endpoint) occupies flat indices `15*s .. 15*s+14`
with per-event offsets `INS 0, MUT 1, VAL 2, EXT 3, CTL 4, SUC 5, ERR 6, NEU 7,
RI 8 (repeated_intent), RF 9 (repeated_failure), WC 10, SR 11 (state_revisited),
VC 12 (validation_comparable), VD 13 (validation_delta), TC 14`.
`WC`, `TC`, `INS`, `MUT` and `CTL` are never read. `relu(v) = max(0, v)`;
`clip01(v) = relu(v) - relu(v - 1) = min(1, relu(v))`.

Every value below is exact for the binary features. The analogue quantities are the
net validation delta `S` (gates at +0.02 and −0.1, each a ramp of width 0.01 in `S`)
and the fresh-failure term `NF` (which scales a recent validation delta).

Non-zero slot counts: fc1.weight 204, fc1.bias 15, fc2.weight 55, fc2.bias 6,
fc3.weight 15, fc3.bias 0 (295 of 3,256 overall). All 24 layer-1 units are used;
layer-2 units 9–11 are all-zero and unused.

## Layer 1 (fc1): per-slot scores and window sums

### Circuit A — redundancy evidence (units 0–6)

| Unit | Computes | Meaning |
|---|---|---|
| 0–6 | `r_s = relu(RI_s + RF_s + SR_s - NEU_s - 3*EXT_s)` for slot `s = unit + 1` (slots 1–7; slot 0 never repeats) | Redundant work in slot `s`: a repeated intent, a repeated identical failure and a state reversion each add one point. EXTERNAL events never count (their repeats are retries, polls or resolutions, handled by circuits D–F); NEUTRAL-outcome repeats (polls) do not count. Values 0–3. |

### Circuit B — resolution of an episode (units 7–14)

| Unit | Computes | Meaning |
|---|---|---|
| 7–10 | `resA_s = relu(RI_s + SUC_s + EXT_s + VAL_s - VC_s - 3 + (1/8) * sum_{j<s} (ERR_j + NEU_j))` for `s = 4, 5, 6, 7` (unit 7 = slot 4 … unit 10 = slot 7), bias −3 | A successful repeat of an EXTERNAL call or of a non-comparable VALIDATE op in the last 4 events, after at least one earlier ERROR or NEUTRAL (poll) event in the window: a transient failure or a poll that came good. Value `n/8`, n = number of prior ERROR/NEUTRAL events. |
| 11–14 | `resB_s = relu(RI_s + SUC_s + VC_s - 3 + (1/8) * sum_{j<s} RF_j)` for `s = 4, 5, 6, 7`, bias −3 | A comparable passing re-run in the last 4 events after at least one repeated identical failure earlier in the window: a retried validation failure that was repaired. Value `n/8`, n = number of prior repeated failures. |

### Circuit C — recent retry and fresh-failure evidence (units 15–17)

| Unit | Computes | Meaning |
|---|---|---|
| 15 | `RT_a = relu(sum_{s=4..7} RF_s)` | Repeated identical failures in the last 4 events. |
| 16 | `RT_b = relu(sum_{s=4..7} RF_s - 1)`, bias −1 | Clip helper: layer 2 forms `RT = RT_a - RT_b = [a recent repeated failure exists]`. |
| 17 | `NF = relu(sum_s (ERR_s - VC_s) - 10 * sum_{s=4..7} VD_s)` | Fresh-failure evidence: non-comparable errors (a suite's first failing result, an external or tool error) minus comparable passing runs, plus 10× any recent regression, minus 10× any recent improvement. Positive when the failure chain is new; zero when the window only repeats a pre-existing failure (comparable error with delta 0) or when the recent runs improved. |

### Circuit D — reversions, progress and retry bound (units 18–21)

| Unit | Computes | Meaning |
|---|---|---|
| 18 | `VM1 = relu(sum_s SR_s - 1)`, weight 1 on every `SR`, bias −1 | ≥ 1 iff at least two state reversions (`N_rev = 2`). |
| 19 | `PROG = relu(100 * sum_s VD_s - 1)`, weight 100 on every `VD`, bias −1 | ≥ 1 iff the net validation delta `S >= 0.02`. Veto role only. |
| 20 | `NZ = relu(-20 * sum_s VD_s - 1)`, weight −20 on every `VD`, bias −1 | ≥ 1 iff `S <= -0.1` (a net regression: the tests are changing, so the window is not a stall). Veto role only. |
| 21 | `OOP = relu(sum_s RF_s - 3)`, weight 1 on every `RF`, bias −3 | ≥ 1 iff the window holds ≥ 4 repeated identical failures: retries beyond `R_max = 3`. |

### Circuit E — polling (units 22–23)

| Unit | Computes | Meaning |
|---|---|---|
| 22 | `POLL = relu(RI_7 + NEU_7 + EXT_7 - 2)`, bias −2 | The endpoint is a repeated NEUTRAL EXTERNAL operation (a poll in flight). |
| 23 | `OVER = relu(sum_s NEU_s + 3 * NEU_0 - 6)`, weight 1 on every `NEU` (4 on `NEU_0`), bias −6 | The poll run is beyond policy: 7 or more neutral events in the window, or the run already filled slot 0 (it started before the window) with 4 or more neutral events (`P_max = 5`). |

## Layer 2 (fc2): class conditions

`veto = 40 * (PROG + NZ + VM1)` (any progress, any net regression, or two reversions).

| Unit | Computes | Meaning |
|---|---|---|
| 0 | `OSC = relu(VM1)` | OSCILLATING condition: ≥ 2 reversions. |
| 1 | `SA_a = relu(R + 2*OOP - 1 - veto)`, `R = u0 + … + u6` (weights 1 on units 0–6, 2 on unit 21, −40 on units 18, 19, 20, bias −1) | Stall evidence: redundancy `R >= 2`, or retries beyond policy, with no change in validation results and fewer than two reversions. |
| 2 | `SA_b = relu(R + 2*OOP - 2 - veto)`, same weights, bias −2 | Clip helper: `STALL_A = SA_a - SA_b` is 0/1. |
| 3 | `SB_a = relu(2*POLL + OVER - 2 - veto)` (weights 2 on unit 22, 1 on unit 23, −40 on units 18, 19, 20, bias −2) | Over-polling stall: the endpoint poll belongs to a run beyond policy (or ≥ 3 units of OVER on their own). |
| 4 | `SB_b = relu(2*POLL + OVER - 3 - veto)`, same weights, bias −3 | Clip helper: `STALL_B = SB_a - SB_b` is 0/1. |
| 5 | `ACT = relu(2*(RT_a - RT_b) + NF/32 - 2 - 10*OOP)` (weights 2, −2 on units 15, 16; 1/32 on unit 17; −10 on unit 21; bias −2) | Active in-policy episode: a recent repeated failure, fresh-failure evidence, and at most 3 repeated failures. The 2/−2 gate can only open when `RT = 1`; `NF/32 <= 1.5` can never open it alone. Value `NF/32` when active. |
| 6 | `RES_a = relu(8 * (u7 + … + u14))` | Any resolution-like event (circuit B). |
| 7 | `RES_b = relu(8 * (u7 + … + u14) - 1)`, bias −1 | Clip helper: `RES = RES_a - RES_b` is 0/1 (each resolution unit fires with value ≥ 1/8). |
| 8 | `POLLf = relu(POLL - OVER)` (weights 1 on unit 22, −1 on unit 23) | In-flight polling that is still within policy. |

## Layer 3 (fc3): output logits and the policy branch each implements

All output biases are 0. With `OSC`, `STALL_A`, `STALL_B`, `ACT`, `RES`, `POLLf` as above:

| Logit | Weights | Reference-policy branch |
|---|---|---|
| 0 PRODUCTIVE | all zero | The default branch: no loop condition holds, every logit is 0 and the first-index tie rule selects PRODUCTIVE. |
| 1 RECOVERY | `2000*ACT + 20*RES + 20*POLLf - 10000*OSC` (unit 5: +2000, 6: +20, 7: −20, 8: +20, 0: −10000) | `logit_recovery`: RECOVERY iff an in-policy episode is active, or an episode resolved within the last 4 events, or polling is in flight within policy, and not OSCILLATING. |
| 2 STALLED | `20*(STALL_A + STALL_B) - 2000*ACT - 100*RES - 100*POLLf - 1000*OSC` (unit 1: +20, 2: −20, 3: +20, 4: −20, 5: −2000, 6: −100, 7: +100, 8: −100, 0: −1000) | `logit_stalled`: STALLED iff the stall evidence holds and no RECOVERY signal is present (spec 6.3 rule 3: no episode in policy at the endpoint) and not OSCILLATING. |
| 3 OSCILLATING | `20*OSC` (unit 0: +20) | `logit_oscillating`: OSCILLATING iff `V >= 2`. Highest precedence: it vetoes RECOVERY (−10000) and STALLED (−1000). |

Precedence check: if OSCILLATING holds, logit 3 ≥ 20 while logits 1 and 2 are negative.
If a RECOVERY signal holds (and OSC does not), logit 1 ≥ 20 (≥ 62.5 via `ACT`) while logit 2 ≤ 40 − 100 < 0.
Otherwise a stall condition gives logit 2 = 20 or 40 with the others ≤ 0.
The all-zero input activates no unit and predicts PRODUCTIVE.
