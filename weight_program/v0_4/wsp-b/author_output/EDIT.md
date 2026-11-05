# WSP-B edit: the WSP-A readout neurons recalibrated for real logs, productive as the default

Final edit hash: `0ea2eb774a950389f6cd60da4e39b413cd300c47af353a59b2f035363d6bf142` (kernel state; `build_edit.py` with
`edit_constants.safetensors` reproduces it tensor for tensor, checked twice by `scratch/verify_build_b.py`).
This is one revision of the WSP-A edit (`74576466…`); `REVISION.md` lists every change with its reasons and the
disclosed windows behind it. This file describes the whole final edit.

## Summary

The edit keeps the WSP-A architecture: hand-built SwiGLU neurons in otherwise silent MLP units, each reading
standardised feature directions of the readout residual and writing along a readout wire that moves one label
logit only, plus a readout change in the label rows. What changed is what the readout treats as the default and
which features gate the overrides, both driven by what the 200 disclosed windows showed (`REVISION.md`):

- **PRODUCTIVE is the default.** On real logs the original model never names `productive` (its logit sits 4.2
  below the others on every class, because nearly every real context contains failures). The ` productive` row
  now carries a constant +4.2 (the measured deficit) plus "reads/searches present" and minus "identical failing
  block repeated"; the WSP-A progress/resolution direction, which lowered `productive` on real productive windows,
  is gone.
- **STALLED and OSCILLATING are presence-gated overrides.** The two WSP-A neurons keep their feature combinations
  (whose behaviour on the disclosed windows is known), with new terms: reads present, pending polls and repeated
  http/poll failures count against STALLED (they mark retries); the OSCILLATING up-term no longer needs failures
  (half of the real oscillation windows have passing tests) and is suppressed by reads.
- **RECOVERY is a linear term in the label rows** (pending polls, repeated http/poll failures, resolution, reads;
  minus repeated failing tests). The WSP-A RECOVERY neuron is removed and layer 21 returns to the original.

All directions are means of matched-pair differences of the unit-RMS residual at the exact input of the host MLP
(`x_L = hidden[L+1] − down_proj_L @ act_L`) or at the final residual (label rows). The WSP-A features are kept with
their WSP-A standardisation; the new features (from identical-history pairs under long busy sessions) are
standardised on a realistic labelled set (set R). Constants ride on set R's mean readout direction (projection
stable to ±0.2 on 31.5). Rows are divided elementwise by the host layer's `(1 + post_attention_layernorm)`.

## What changed, tensor by tensor (relative to the original weights)

| tensor | change |
| --- | --- |
| `model.language_model.embed_tokens.weight` | row 25399 (` productive`) += Δ_p; row 39923 (` recovering`) += Δ_r (2048 entries, rank 2; rows 65552 and 60477 unchanged). Δ_p = 4.2 + 1.5 z_NFR − 1.0 z_REPX, Δ_r = 2 z_PEND + 2 z_REPH + 2 z_REPP − 1.5 z_REPF + 0.5 z_NFR + 0.5 z_RESH, both carried in the final-residual space and divided by `(1 + norm.weight)`. |
| `model.language_model.layers.23.mlp.gate_proj.weight` | row 1046 replaced (OSCILLATING gate). |
| `model.language_model.layers.23.mlp.up_proj.weight` | row 1046 replaced (OSCILLATING up). |
| `model.language_model.layers.23.mlp.down_proj.weight` | column 1046 replaced (OSCILLATING wire, scale 0.814). |
| `model.language_model.layers.22.mlp.gate_proj.weight` | row 1613 replaced (STALLED gate). |
| `model.language_model.layers.22.mlp.up_proj.weight` | row 1613 replaced (STALLED up). |
| `model.language_model.layers.22.mlp.down_proj.weight` | column 1613 replaced (STALLED wire in the null space of layer 23's value projection, scale 0.1055). |

Layer 21 (edited in WSP-A) is back at the original weights. Every edited layer tensor changes exactly one row or
one column; each ΔW has rank 1 (rank 2 for the embedding).

## Features (z-scores of unit-RMS residual projections)

| name | direction (mean of matched-pair differences) | derived from | standardised on |
| --- | --- | --- | --- |
| NFR | reads/searches/list_dir present in the window vs plans/polls/http in their place (NF2 + NF3) | WSP-B calls 19-23 | set R (calls 17-18) |
| REPF / REPH / REPP | identical failing test / http error / pending poll repeated vs varied blocks; REPX = their sum | calls 19-23 | set R |
| PEND | pending polls present vs plans/http in their place | calls 19-23 | set R |
| RESH | http retries then ok vs retries continuing (weak, consistent) | calls 19-23 | set R |
| REP, CHAIN, RES, PROG | identical failing blocks repeated; long poll chain; success after errors; improving validation | WSP-A calls 5-11 | WSP-A sets B, D |
| EDITS, EDITREPfree, EDITCOUNT | edit blocks among repeated failures; edit repetition without errors; several edits vs one | WSP-A calls 9-11, 24-25 | WSP-A sets B, D |

## Mechanism M1: OSCILLATING neuron (layer 23, unit 1046)

- gate `e = z_EDITS + z_EDITREPfree` (edit-type repetition); activation `silu(3 (e − 1.642)) · (1.5 e + z_EDITCOUNT − z_NFR − 2.433)`;
- writes `0.814 · act · u_looping` (` looping` readout component orthogonal to the other rows; gain 2.78 logit per unit act before the final RMS);
- threshold 1.642 = midpoint of the OSCILLATING and other-class medians of `e` on set R; offset −2.433 puts the median OSCILLATING up at +1; scale 0.814 gives a median gain of +5 logits on set R's oscillating windows;
- effect: fires on repeated edits of the same file (A/B edits, edit/revert, write/revert cycles) with or without validation failures; suppressed when reads/searches are present. On the disclosed windows (as run) it adds +12.8 logits on average to `looping` on OSCILLATING windows and +2.4 on PRODUCTIVE ones; its response is bimodal (half of the oscillation windows get little), which is why it was not rescaled.
- limitation: cannot see hunk-id reversal, so family-6 stalls (fresh-id edits with unchanged failures) look like oscillation, and productive incremental repairs with several same-file edits can fire it.

## Mechanism M2: STALLED neuron (layer 22, unit 1613)

- gate `r = z_REP + 2 z_CHAIN + z_REPF − z_PEND − 0.5 z_REPH` (identical failing tests and long chains; pending polls and repeated http errors mark retries instead); activation `silu(3 (r − 3.478)) · (−z_RES − z_PROG + z_CHAIN − 2 z_EDITS − 2 z_EDITREPfree − z_NFR − z_PEND − 0.5 z_REPH − 0.5 z_REPP + 1.144)`;
- writes `0.1055 · act · u_stalled_null` (` stalled` readout component orthogonal to the other rows, projected into the null space of `v_proj_23 ⊙ (1 + input_layernorm_23)` so that firing at log positions cannot reach the readout through layer 23's values; gains on the four rows [−0.16, −0.21, +1.85, −0.14] per unit of the wire);
- threshold and offset by the same rules as M1 on set R; the scale was first set by the set-R rule (0.382) and then divided by 3.62 so that the median gain on the disclosed STALLED windows is +5 logits (it was +18, which made STALLED win on 21 oscillating and 17 recovery windows; `scratch/rescale_stal.py`);
- effect: fires on windows dominated by identical failing blocks without reads, edits, polls or http retries; on the disclosed windows (before the final rescale) it added +16.4 logits on STALLED windows, +2.6 on RECOVERY, +1.0 on PRODUCTIVE and +5.6 on OSCILLATING ones.

## Mechanism M3: label rows (PRODUCTIVE default and linear RECOVERY term)

- ` productive` row: +4.2 (mean productive deficit on the disclosed windows) + 1.5 z_NFR − 1.0 z_REPX: windows with reads and without identical repeated failures are named productive unless an override fires;
- ` recovering` row: + 2 z_PEND + 2 z_REPH + 2 z_REPP − 1.5 z_REPF + 0.5 z_NFR + 0.5 z_RESH: pending polls and repeated http/poll failures (bounded retries), resolution and reads raise `recovering`; repeated failing tests lower it. Doubled weights relative to the set-R rule because on the disclosed windows pending polls (1.16 per RECOVERY window vs at most 0.2 for the other classes) and http retries are nearly exclusive to RECOVERY.

## Results on texts (my own and disclosed)

| set | windows | original | WSP-A edit | revision |
| --- | --- | --- | --- | --- |
| disclosed validation windows (calls 1-13 vs 26-38; the revision as run, before the final STALLED rescale) | 200 | 0.240 | 0.315 | 0.425 (recall PROD 0.20 / RECO 0.36 / STAL 0.70 / OSCI 0.44) |
| same windows, final STALLED scale (offline from the same logits, `scratch/sim_rescale.py`) | 200 | | | 0.440 (0.22 / 0.40 / 0.68 / 0.46) |
| set R, my realistic sessions, revision as run on the disclosed windows (calls 17-18 vs 24-25) | 64 | 0.19 | — | 0.42 (0.25 / 0.12 / 0.69 / 0.62) |
| set R, final state after the STALLED rescale (calls 39-40) | 64 | 0.19 | — | 0.44 (0.31 / 0.31 / 0.38 / 0.75) |

On set R the observed logit changes match the direct-path prediction from the two activations and the label-row
deltas (correlation ≥ 0.98), and the layer-21 unit is silent, as in the original.

## Mechanisms

| id | tensors | target class | claim |
| --- | --- | --- | --- |
| M1 | model.language_model.layers.23.mlp.gate_proj.weight, model.language_model.layers.23.mlp.up_proj.weight, model.language_model.layers.23.mlp.down_proj.weight | OSCILLATING | Unit 1046 fires on repeated edit-type blocks (A/B edits, edit/revert, write/revert cycles) with or without validation failures and without reads, and adds its activation to the ` looping` logit only, so edit cycles are named looping instead of productive or recovering. |
| M2 | model.language_model.layers.22.mlp.gate_proj.weight, model.language_model.layers.22.mlp.up_proj.weight, model.language_model.layers.22.mlp.down_proj.weight | STALLED | Unit 1613 fires when identical failing tests or long failure chains dominate the window without reads, edits, pending polls or http retries, and adds its activation to the ` stalled` logit only, so repeated identical failures are named stalled instead of productive or recovering. |
| M3 | label rows | PRODUCTIVE | The ` productive` row gains the measured productive deficit plus reads-present minus repeated-failure directions (and the ` recovering` row a linear retry/poll/resolution term), so windows with investigation and no repeated identical failures are named productive instead of recovering. |
