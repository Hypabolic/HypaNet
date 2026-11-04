# CIRCUITS.md — Phase A-strict (spec v0.2)

Network: `Linear(120, 24) -> ReLU -> Linear(24, 12) -> ReLU -> Linear(12, 4)`.
Input slot `s` (0 = oldest, 7 = endpoint) occupies flat indices `15*s .. 15*s+14`
with per-event offsets `INS 0, MUT 1, VAL 2, EXT 3, CTL 4, SUC 5, ERR 6, NEU 7,
RI 8 (repeated_intent), RF 9 (repeated_failure), WC 10, SR 11 (state_revisited),
VC 12 (validation_comparable), VD 13 (validation_delta), TC 14`.
`WC`, `TC`, `INS` and `CTL` are never read. `relu(v) = max(0, v)`;
`clip01(v) = relu(v) - relu(v - 1) = min(1, relu(v))`.

Every value below is exact for the binary features, so each clipped indicator
is exactly 0 or 1. The only analogue quantity is the net validation delta,
whose gate is a ramp of width 1e-4 (see unit 19/20).

Non-zero slot counts: fc1.weight 156, fc1.bias 11, fc2.weight 49, fc2.bias 4,
fc3.weight 13, fc3.bias 0 (233 of 3,256 overall). Hidden units 23 (layer 1)
and 8–11 (layer 2) are all-zero and unused.

## Layer 1 (fc1): per-slot scores and window sums

### Circuit A — stall-ineffective evidence (units 0–8)

| Unit | Computes | Meaning |
|---|---|---|
| 0–7 | `ine_s = relu(RI_s + SR_s + RF_s - NEU_s)`, unit `s` reads slot `s` | Stall-ineffective score of event `s`: 1 per repeated intent, per state reversion and per repeated identical failure; a NEUTRAL-outcome repeat (poll, plan) scores 0. Values 0–3. |
| 8 | `end2 = relu(RI_7 + SR_7 + RF_7 - NEU_7 - 1)`, bias −1 | Clip helper for the endpoint: layer 2 forms `END = ine_7 - end2 = min(1, ine_7)`, i.e. "the final event is stall-ineffective". |

### Circuit B — reversions (units 17, 18)

| Unit | Computes | Meaning |
|---|---|---|
| 17 | `Vm1 = relu(sum_s SR_s - 1)`, weight 1 on every `SR`, bias −1 | ≥ 1 iff at least two state reversions in the window. |
| 18 | `Vm2 = relu(sum_s SR_s - 2)`, bias −2 | Clip helper: layer 2 forms `V2 = Vm1 - Vm2 = [V >= 2]` (`N_rev = 2`). |

### Circuit C — net validation progress (units 19, 20)

| Unit | Computes | Meaning |
|---|---|---|
| 19 | `PROG_a = relu(1e4 * sum_s VD_s - 174)`, weight 10000 on every `VD`, bias −174 | Ramp that reaches 1 when the net validation delta over the window reaches 0.0175. |
| 20 | `PROG_b = relu(1e4 * sum_s VD_s - 175)`, bias −175 | Clip helper: `PROG = PROG_a - PROG_b = [S >= 0.0175]` (ramp over `S` in [0.0174, 0.0175]). "No progress" is the `Δp <= ε_p` proxy used by the OSCILLATING and STALLED rules. |

### Circuit D — active in-policy episode (units 9–12, 21)

| Unit | Computes | Meaning |
|---|---|---|
| 9–12 | `ee_s = relu(ERR_s - VC_s)` for `s = 4, 5, 6, 7` (unit 9 = slot 4 … unit 12 = slot 7) | Episode-like error in the last 4 events: an ERROR that is not a comparable test result (comparable failing test runs are task-level, not episodes). |
| 21 | `OOP = relu(sum_s RF_s - 2 * EXT_7 - 3)`, weight 1 on every `RF`, −2 on `EXT_7`, bias −3 | Retries beyond policy: ≥ 1 iff the window holds ≥ 4 repeated identical failures (`R_max = 3`), or ≥ 6 when the endpoint is EXTERNAL (`P_max = 5` polls). Used only as a veto. |

### Circuit E — recently resolved episode (units 13–16)

| Unit | Computes | Meaning |
|---|---|---|
| 13–16 | `rs_s = relu(RI_s + SUC_s + MUT_s + VAL_s + EXT_s - VC_s - 3 + (1/8) * sum_{j<s} (ERR_j - VC_j))` for `s = 4, 5, 6, 7` (unit 13 = slot 4 … unit 16 = slot 7), bias −3 | Resolution-like event in the last 4 events: a successful repeat of a MUTATE / VALIDATE / EXTERNAL operation that is not a comparable test result, preceded in the window by at least one episode-like error. Fires with value `n/8` (n = number of prior episode-like errors net of comparable successes), else 0. |

### Circuit F — endpoint poll (unit 22)

| Unit | Computes | Meaning |
|---|---|---|
| 22 | `POLL = relu(RI_7 + NEU_7 + EXT_7 - 2)`, bias −2 | The endpoint is a repeated NEUTRAL EXTERNAL operation: in-flight polling of a dependency. |

## Layer 2 (fc2): class conditions

| Unit | Computes | Meaning |
|---|---|---|
| 0 | `OSC = relu((u17 - u18) - (u19 - u20)) = relu(V2 - PROG)` | OSCILLATING condition: ≥ 2 reversions and no net progress. |
| 1 | `STALL_a = relu(pre)`, `pre = sum_{s} ine_s + 30*(ine_7 - end2) - 34 - 30*PROG - 30*V2` (weights: 1 on units 0–6, 31 on unit 7, −30 on unit 8, −30/+30 on units 19/20, −30/+30 on units 17/18, bias −34) | ≥ 1 iff stall evidence `C >= 5` (`N_stall = 5`) **and** the endpoint is stall-ineffective, vetoed (−30) by progress and by `V >= 2`. |
| 2 | `STALL_b = relu(pre - 1)`, same weights, bias −35 | Clip helper: `STALL' = STALL_a - STALL_b` is the 0/1 STALLED condition before the episode veto. |
| 3 | `ACT_a = relu(E - 10*OOP)`, `E = u9 + u10 + u11 + u12` | Recent episode-like error with retries still within policy. |
| 4 | `ACT_b = relu(E - 10*OOP - 1)`, bias −1 | Clip helper: `ACT = ACT_a - ACT_b = [E >= 1 and OOP = 0]` — an in-policy episode is active at the endpoint. |
| 5 | `RES_a = relu(8 * (u13 + u14 + u15 + u16))` | Any resolution-like event in the last 4 events. |
| 6 | `RES_b = relu(8 * (…) - 1)`, bias −1 | Clip helper: `RES = RES_a - RES_b` is the 0/1 "episode resolved within `k_rec = 4`" indicator. |
| 7 | `POLL = relu(u22)` | Pass-through of the endpoint-poll detector. |

## Layer 3 (fc3): output logits and the policy branch each implements

All output biases are 0. With `OSC`, `STALL'`, `ACT`, `RES`, `POLL` as above:

| Logit | Weights | Reference-policy branch |
|---|---|---|
| 0 PRODUCTIVE | all zero | The default branch (`return 0`): no loop condition holds, so every logit is 0 and the first-index tie rule selects PRODUCTIVE. |
| 1 RECOVERY | `20*RES + 20*POLL + 200*ACT - 100*STALL' - 1000*OSC` (unit 5: +20, 6: −20, 7: +20, 3: +200, 4: −200, 1: −100, 2: +100, 0: −1000) | `logit_recovery` in `reference_policy`: RECOVERY iff (`ACT` or `RES` or `POLL`) and not OSCILLATING and not STALLED. The +200 on `ACT` outweighs the −100 `STALL'` veto, so an active in-policy episode overrides the stall count (spec 6.3 rule 3: "no episode is in policy at the endpoint"). |
| 2 STALLED | `20*STALL' - 60*ACT` (unit 1: +20, 2: −20, 3: −60, 4: +60) | `logit_stalled`: STALLED iff `C >= 5`, endpoint stall-ineffective, no progress, `V < 2` and no active in-policy episode. |
| 3 OSCILLATING | `20*OSC` (unit 0: +20) | `logit_oscillating`: OSCILLATING iff `V >= 2` and no net validation progress. Highest precedence: it vetoes RECOVERY (−1000) and `STALL'` (−30·V2 in layer 2). |

Precedence check: if OSCILLATING holds, logit 3 = 20 while logits 1 and 2 are ≤ 0.
If STALLED holds (and OSC does not), logit 2 = 20, logit 1 ≤ 40 − 100 < 0.
If an in-policy episode is active together with the stall condition, logit 1 ≥ 100
and logit 2 = −40. Otherwise any RECOVERY signal gives logit 1 ≥ 20 with the others ≤ 0.
The all-zero input activates no unit and predicts PRODUCTIVE.
