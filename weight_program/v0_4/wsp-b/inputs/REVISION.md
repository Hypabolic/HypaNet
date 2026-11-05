# WSP-B revision record

## What was disclosed

`disclosure/DISCLOSURE.json`, sha256 `d8c75fbcd909b78213aba4110a7d42d6039ffa04d9ca72f1538ff0fb86097cef` (636,553 bytes):
200 validation windows `V001`…`V200`, 50 per oracle class, each with the rendered prefix (blocks 0..t, the
`### window start` marker removed before use), its oracle label and the class the archived WSP-A edit gives it.
Nothing else was used. Base edit: WSP-A hash `74576466ac6d22c6cc6d7545e4162a0ae2daf38a6049624e232caf635910079a`.
Final edit: hash `0ea2eb774a950389f6cd60da4e39b413cd300c47af353a59b2f035363d6bf142`.

## What the windows showed about the WSP-A edit

`scratch/inspect_b01.py`, `scratch/disc_features.py` (surface features of the last 8 blocks, parsed from the text),
`scratch/ana_disc.py` (logits from fwd calls 1-13, model=both, WSP-A hash 74576466…):

- WSP-A on the 200 windows: accuracy 0.315; recall PROD 0.06, RECO 0.18, STAL 0.46, OSCI 0.56. The original
  model: 0.240 with PROD recall 0.00: its ` productive` logit is 4.2 below the best other logit on average, on every
  class, because almost every real context contains failures (sessions are long and busy: median 20 blocks, up to 54).
- The WSP-A ` productive` direction (progress + resolution − repeated failures) *lowered* the productive logit on real
  productive windows by 3.2 (median): their median deficit was 9.7 logits (47 of 50 wrong, e.g. V001, V011, V017,
  V045, V083, V102, V103, V106).
- Real class signatures (means over the last 8 blocks): PRODUCTIVE has 3.7 distinct new reads/searches, 0.4
  repeated blocks, 0.1 repeated failing blocks, 88% contain some failure; RECOVERY has 2.4 new facts, 1.6 repeated
  failing blocks (http/poll retries), 1.2 pending polls, a resolution inside 64% of windows; STALLED has 0.04 new
  facts, 4.3 repeated blocks, no reversals, 36% are "fresh-id edits with unchanged failing tests" (V027, V037, V062,
  V063, V071, V079, V091 …) and 16% have no errors (V040, V111, V158, V159); OSCILLATING has hunk ids returning to
  earlier states in 100% of windows and no errors in 48% (V002, V010, V018, V069, V096, V135, V141 …).
- The WSP-A neurons' direct-path contributions on the real windows (edited − original logit): the STALLED neuron
  adds +4.3 (median) on STALLED windows but also +0.9 (median, +4.4 at the 90th percentile) on RECOVERY windows
  (V054, V074, V131, V152, V154, V174, V183, V188, V190, V198 → STALLED); the OSCILLATING neuron adds +5.2 on
  OSCILLATING windows but needs "trouble", so error-free oscillation went to PRODUCTIVE (V002, V010, V018, V056,
  V070, V092, V135, V136, V141, V165); the RECOVERY neuron lowered ` recovering` on recovery windows (median −0.3).
- Simulation on the same logits (`scratch/sim_disc.py`): replacing the productive row by a constant +4.2 and keeping
  only the STALLED/OSCILLATING neurons gives 0.385 (PROD recall 0.44); this fixed the direction of the revision.

Probes on my own texts under realistic context (`scratch/mk3.py`, `mk5.py`, `scen_hb5.py`; identical-history
matched pairs, calls 19-23; realistic labelled set R, calls 17-18): "reads present" (NF2/NF3), "identical failing
test / http error / pending poll repeated" (REPF/REPH/REPP) and "pending polls present" (PEND) are consistently
encoded (leave-one-out agreement 1.00); distinct-vs-repeated reads and same-file A/B edits are weak (0.18 of
scale); resolution is weak but consistent. Earlier pair sets with mismatched histories (calls 14-16) were discarded.

## Changes from the WSP-A edit (each with its reason and the windows behind it)

1. **Label rows, ` productive` row (M3).** Replaced the WSP-A delta `1.6 (z_PROG + z_RES − z_REP)` by
   `+4.2 + 1.5 z_NFR − 1.0 z_REPX` (x24 space, carried on set R's mean direction, divided by `1 + norm.weight`).
   Reason: the productive logit must overcome the measured 4.2 deficit; real productive windows are marked by
   reads/searches and by the absence of identical repeated failures, not by resolution. Windows: V001, V011, V012,
   V017, V022, V045, V049, V083, V102, V103, V106, V110 (productive → recovering under WSP-A); V001, V011, V012,
   V017, V022, V049, V083, V102, V110 are named productive by the revision.
2. **Label rows, ` recovering` row (M3, same tensor).** New linear term
   `+2 z_PEND + 2 z_REPH + 2 z_REPP − 1.5 z_REPF + 0.5 z_NFR + 0.5 z_RESH`. Reason: real recovery windows are
   bounded http/poll retries and polls (V026, V030, V039, V068, V101, V149, V163, V184, V187, V196) or short test
   retries with reads; a gated neuron for this (tried offline, `scratch/derive_b2.py`-`derive_b4.py`) could not be
   separated from the STALLED gate, so a linear term is used. Weights doubled relative to the set-R rule because
   pending polls are nearly exclusive to RECOVERY on the real windows.
3. **Layer 21 MLP (unit 1007) reverted to the original.** The WSP-A RECOVERY neuron lowered ` recovering` on recovery
   windows (median −0.3) and on others, with no benefit on the real data (simulation (a) vs (b) in
   `scratch/sim_disc.py`: 0.325 vs 0.385).
4. **Layer 22 MLP (unit 1613, STALLED, M2).** Gate: WSP-A `z_REP + 2 z_CHAIN` plus `+ z_REPF − z_PEND − 0.5 z_REPH`;
   up: WSP-A `−z_RES − z_PROG + z_CHAIN − 2 z_EDITS − 2 z_EDITREPfree` plus `− z_NFR − z_PEND − 0.5 z_REPH − 0.5 z_REPP`;
   threshold 3.478 and offset 1.144 by the median rules on set R; wire scale 0.382 by the set-R rule, then 0.1055 after
   the real-window rescale (change 6). Reason: on the real windows the neuron fired on recovery retry chains (V054,
   V074, V131, V152, V188, V190, V198: http/poll repeats with reads); pending polls, repeated http failures and reads
   mark retries, not stalls.
5. **Layer 23 MLP (unit 1046, OSCILLATING, M1).** Gate unchanged in form (`z_EDITS + z_EDITREPfree`, threshold 1.642
   by the median rule on set R); up: `1.5 e + z_EDITCOUNT − z_NFR − 2.433` (the WSP-A "trouble" term T is dropped);
   wire scale 0.814 (set-R rule). Reason: half of the real oscillation windows have no failures (V002, V010, V018,
   V056, V070, V092, V135, V136, V141, V165) and their edits are not accompanied by reads.
6. **STALLED wire rescaled by 0.2764** (`scratch/rescale_stal.py`): the revision as run gave the stalled logit a
   median gain of +18.1 on the disclosed STALLED windows (designed: +5), which made STALLED win on 21 oscillating
   (V043, V076, V081, V087, V093, V126, V130, V132, V133, V138, V144, V169, V170 …) and 17 recovery windows. The
   same calibration rule used in the derivation (median target gain = 5), applied to the real-window gains, gives
   the factor. Offline on the same logits (`scratch/sim_rescale.py`) the effect is 0.425 → 0.440 (recall
   0.22 / 0.40 / 0.68 / 0.46). The OSCILLATING wire was not rescaled: its real-window gains are bimodal (median +0.5,
   90th percentile +53), so the median rule would be meaningless there.

Deliberately unchanged: the units (1046, 1613), the readout wires (orthogonal components of the label rows), the
null-space projection of the STALLED wire, the WSP-A feature directions and their standardisation (their behaviour
on the real windows is known), the slope 3 of the gates, and the ` stalled` / ` looping` label rows.

## Forward passes (ledger call ids, with the edit hash each ran under)

| calls | edit hash | texts | purpose |
| --- | --- | --- | --- |
| 1-13 | 74576466… (WSP-A), model=both | disclosed windows | the permitted pre-revision run; logits and WSP-A contributions |
| 14-16 | 74576466…, model=original | pair set H (mismatched histories; discarded) | first direction probes |
| 17-18 | 74576466…, model=original | set R (64 realistic labelled sessions) | class statistics, standardisation, baseline |
| 19-23 | 74576466…, model=original | identical-history pairs (80 sessions, 10 properties) | new feature directions |
| 24-25 | ac483550…, model=edited | set R | revision before the STALLED rescale: acc 0.42 vs 0.19 original; leakage check corr ≥ 0.98 |
| 26-38 | ac483550…, model=edited | disclosed windows | the permitted revision run: acc 0.425 (recall 0.20 / 0.36 / 0.70 / 0.44), 45 WSP-A errors fixed, 23 broken |
| 39-40 | 0ea2eb77…, model=edited | set R | final state: acc 0.44 vs 0.19; leakage check corr ≥ 0.99; layer-21 unit silent |

Edit states run on the disclosed windows: 74576466… and ac483550… (two). The final state 0ea2eb77… differs from
ac483550… only by the STALLED wire scale; its effect on the disclosed windows is the offline rescale above.

## Iteration log

- `938f4bc9…`→ (start) `74576466…`: WSP-A edit, confirmed as the kernel state; calls 1-13 on the disclosed windows.
- Offline designs b1-b4 (`scratch/derive_b.py`, `derive_b2.py`-`derive_b4.py`, params files, reports): gated RECOVERY
  neuron variants; rejected by reasoning (no gate separation from STALLED, noise amplification), never applied.
- `ac483550ce6d5b858e0150059d1631e4418e1c2856f36affbd6ddd9d7fcabde7`: design b5/b6 (`scratch/derive_b5.py`,
  `params_b6.json`) applied through `build_edit.build`; tested on set R (calls 24-25) and on the disclosed windows
  (calls 26-38).
- `0ea2eb774a950389f6cd60da4e39b413cd300c47af353a59b2f035363d6bf142`: STALLED wire × 0.2764 (`scratch/rescale_stal.py`),
  applied through `build_edit.build`; `scratch/verify_build_b.py` confirms build == kernel, twice; tested on set R
  (calls 39-40). Final.

Meter: 40 requests, 1,439,522 tokens of 2,000,000.
