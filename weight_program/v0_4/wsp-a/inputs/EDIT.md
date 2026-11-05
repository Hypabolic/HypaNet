# WSP-A edit: three hand-built readout neurons and a productive readout direction

Final edit hash: `74576466ac6d22c6cc6d7545e4162a0ae2daf38a6049624e232caf635910079a` (kernel state; `build_edit.py`
with `edit_constants.safetensors` reproduces it tensor for tensor, checked by `scratch/verify_build.py`).

## Summary

The original model reads the log through a one-dimensional "trouble" axis: windows with an error line get
`recovering` (or, nearly tied, `stalled`/`looping`), clean windows get `productive`. Matched-pair probes of my own
texts (fwd calls 5-11, 24-25) showed what the readout position does and does not encode linearly:

- encoded, consistently: errors present and recent (ERR), a success right after errors (RES), improving validation
  followed by a pass (PROG), identical failing blocks repeating (REP, saturating after two repeats), edit blocks
  present among repeated failures (EDITS), edit repetition without errors (EDITREPfree), long versus short
  poll/retry chains (CHAIN), several edits versus one edit among repeated failures (EDITCOUNT);
- not encoded: hunk-id reversal (alternating ids minus fresh ids, tool matched: d' ≈ −0.3), hunk lines versus
  no-op edits, improving pass counts without a final pass, counts of any kind (the DeltaNet delta rule
  overwrites repeated keys; attention averages).

So the edit builds the class decision from these presence/recency features with explicit conjunctions, one
SwiGLU neuron per hard class, each in its own layer so that it can be reverted alone, plus a linear PRODUCTIVE
direction in the label rows. Each neuron is written into a unit that is silent at the readout position in the
original model (max |activation| < 0.01 over 2212 windows of sets B and D; fwd calls 2-4, 9-11).

All feature directions are **means of matched-pair differences** of the unit-RMS residual at the exact input of the
host MLP (`x_L = hidden[L+1] − down_proj_L @ act_L`), unit-normalised, then standardised (`z_F = (F·x̂ − μ_F)/σ_F`)
with μ, σ over my confidently labelled windows of sets B and D. Constants are carried on the mean readout
direction m̂ of the host space, whose projection is 31.5 ± 0.2 on every window, so a constant k is implemented as
`(k / 31.5) m̂`. Rows are divided elementwise by the host layer's `(1 + post_attention_layernorm)` so that the neuron
sees `F·x̂` exactly. Scripts: `scratch/derive_v3.py` with `scratch/params_v3b.json` (the final constants),
`scratch/apply_build.py` (applies `build_edit.build` to the kernel).

## What changed, tensor by tensor

| tensor | change |
| --- | --- |
| `model.language_model.embed_tokens.weight` | row 25399 (` productive`) += Δ (1024 entries, rank 1; the other three label rows are unchanged). Δ = 1.6 (z_PROG + z_RES − z_REP) carried in the final-residual space, divided by `(1 + norm.weight)`. New row norm 1.04 (was 0.64), cos with the original row 0.73. |
| `model.language_model.layers.23.mlp.gate_proj.weight` | row 1046 replaced (OSCILLATING gate). |
| `model.language_model.layers.23.mlp.up_proj.weight` | row 1046 replaced (OSCILLATING up). |
| `model.language_model.layers.23.mlp.down_proj.weight` | column 1046 replaced (OSCILLATING wire, scale 0.14). |
| `model.language_model.layers.22.mlp.gate_proj.weight` | row 1613 replaced (STALLED gate). |
| `model.language_model.layers.22.mlp.up_proj.weight` | row 1613 replaced (STALLED up). |
| `model.language_model.layers.22.mlp.down_proj.weight` | column 1613 replaced (STALLED wire, scale 0.045, in the null space of layer 23's value projection). |
| `model.language_model.layers.21.mlp.gate_proj.weight` | row 1007 replaced (RECOVERY gate). |
| `model.language_model.layers.21.mlp.up_proj.weight` | row 1007 replaced (RECOVERY up). |
| `model.language_model.layers.21.mlp.down_proj.weight` | column 1007 replaced (RECOVERY wire, scale 0.4). |

Every edited tensor changes exactly 1024 entries (one row or one column); each ΔW has rank 1.

## Mechanism M1: OSCILLATING neuron (layer 23, unit 1046)

- reads (at the readout position, input of the layer-23 MLP): `e = z_EDITS + z_EDITREPfree` (edit-type
  repetition), `T = −z_PROG − z_RES + z_REP + z_CHAIN` (trouble: identical failures, long chains, no progress,
  no resolution), `z_EDITCOUNT` (several edits rather than one);
- activation: `silu(3 (e − 0.5)) · (T + 1.5 e + 1.0 z_EDITCOUNT − 2.5)`;
- writes `0.14 · act · u_looping`, where `u_looping` is the component of the ` looping` readout direction
  (label row ⊙ (1 + final norm)) orthogonal to the other three label rows, so only the `looping` logit moves
  (gain 0.388 logit per unit activation, before the final RMS normalisation);
- effect: on A/B edit cycles with failing validation between, and on edit/revert and write/revert cycles, the
  activation is 5 to 30 (+5 to +30 logits on `looping`); on stalled windows without edits the gate is closed
  (e ≈ −0.3) and the activation is ≈ 0; on single-edit retries and diagnosed repairs the up-term is ≈ 0 or
  negative. Layer 23's MLP output at other positions is never read by anything, so this neuron cannot leak.
- known limitation: incremental repair (edit, failing tests with improving counts, edit, ...) is not separable
  from A/B oscillation with these features (set G probe, calls 24-25); error-free A/B edits with passing builds are
  missed (no reversal feature exists).

## Mechanism M2: STALLED neuron (layer 22, unit 1613)

- reads (input of the layer-22 MLP): `r = z_REP + 2 z_CHAIN` (repeated identical failures and long chains; the
  chain feature separates RECOVERY retry windows from STALLED ones best), `q = −z_RES − z_PROG + z_CHAIN −
  2 z_EDITS − 2 z_EDITREPfree` (no resolution, no progress, long chain, and no edit-type repetition);
- activation: `silu(3 (r − 2.5)) · q`; the threshold 2.5 is where ~90% of STALLED windows pass and ~10% of
  RECOVERY windows (10th/90th percentiles of r on sets B and D);
- writes `0.045 · act · u_stalled_null`, where `u_stalled_null` is the ` stalled` readout component orthogonal
  to the other rows, projected into the null space of `v_proj_23 ⊙ (1 + input_layernorm_23)` (so that if the unit
  fires at log positions, layer 23's attention cannot carry it to the readout through values; gain on the four
  rows [−0.24, −0.16, +1.85, −0.13] per unit of the wire);
- effect: on windows of ≥5 identical failing blocks (tests, builds, lint, http) the activation is 20 to 50
  (+4 to +10 logits on `stalled`); on A/B oscillation with failures the −2 z_EDITS − 2 z_EDITREPfree term keeps
  q low so M1 wins; on short retry chains (≤3) and resolutions the gate is closed.
- known limitation: long poll chains (pending ×10) trigger r only weakly; repeated reads/searches are not
  detected (REP responds to failing blocks only).

## Mechanism M3: RECOVERY neuron (layer 21, unit 1007)

- reads (input of the layer-21 MLP; directions derived in the hidden[22] space, which differs from that input
  by the layer-21 MLP output only): `z_ERR` (errors recently), `z_RES` (a success after errors);
- activation: `silu(3 z_ERR) · (z_RES + 0.9)`;
- writes `0.4 · act · u_recovering` (the ` recovering` readout component orthogonal to the other rows; gain
  1.07 logit per unit activation);
- effect: lowers `recovering` by 2 to 5 logits on stalled windows (errors present, no resolution), so that M1/M2
  do not have to overcome the original model's `recovering` default; raises it by 1 to 3 logits in the few
  blocks after a resolution. It leaves clean windows alone (gate closed).

## Mechanism M4: PRODUCTIVE readout direction (label rows)

- the ` productive` row gains `1.6 (z_PROG + z_RES − z_REP)` in the final-residual space (progress with a
  passing validation, a success after errors, and no identical failing blocks), with the standardisation
  constant carried on m̂;
- effect: +4 logits (median) on productive windows, −6 on stalled ones, −3 on oscillating ones; it lets
  incremental repair, post-resolution exploration and repeated distinct reads be named `productive` despite the
  error lines in context, which the original readout turned into `recovering`.

## Provenance of every constant (ledger call ids)

| constant | derived from |
| --- | --- |
| ERR, PROG, REP, RES directions (per host space) | set C matched pairs, calls 5, 6, 7, 8 (original model) |
| EDITS, EDITREPfree, CHAIN directions | set D matched groups, calls 9, 10, 11 |
| EDITCOUNT direction | set G matched pairs, calls 24, 25 |
| standardisation μ, σ of every feature; mean directions m̂ | confidently labelled windows of sets B (calls 2, 3, 4) and D (calls 9, 10, 11) |
| choice of silent units 1046 / 1613 / 1007 | mlp_act arrays of calls 2-4 and 9-11 |
| readout wires u_c, null-space projection | weights only: label rows, `norm.weight`, `layers.23.self_attn.v_proj.weight`, `layers.23.input_layernorm.weight` |
| gate thresholds, slopes, offsets, scales | percentiles on sets B and D (calls 2-4, 9-11) and reasoning on sets E and F (calls 12-23, 26-28); values in `scratch/params_v3b.json` |

`edit_constants.safetensors` holds the ten finished vectors (`label_row_deltas [4,1024]`, and for each of layers 21,
22, 23 a `gate_row`, an `up_row` and a `down_col` of 1024 entries); `build_edit.py` only writes them into place.

## What my own forward passes showed

Accuracy over my own confidently labelled windows (my reading of the oracle rules), original → edited:

| set | windows | original | edited | recall PROD / RECO / STAL / OSCI (edited) |
| --- | --- | --- | --- | --- |
| E (60 fresh sessions, calls 18-20, edit v2c) | 727 | 0.40 | 0.62 | 0.57 / 0.68 / 0.70 / 0.62 |
| F (54 fresh sessions, calls 26-28, final edit) | 849 | 0.38 | 0.56 | 0.41 / 0.70 / 0.77 / 0.65 |

Observed minus original logits match the direct-path prediction from the three activations and the label-row
delta (correlation ≥ 0.98, residual std ≤ 0.55 logits, sets E and F), so the mechanisms act only through the
readout position. The original model names `recovering` on 55-70% of all windows; the edit names each class
with precision 0.48-0.78.

## Mechanisms

| id | tensors | target class | claim |
| --- | --- | --- | --- |
| M1 | model.language_model.layers.23.mlp.gate_proj.weight, model.language_model.layers.23.mlp.up_proj.weight, model.language_model.layers.23.mlp.down_proj.weight | OSCILLATING | Unit 1046 fires on edit-type repetition (edit/revert blocks recurring) combined with repeated failures and no progress, and adds its activation to the ` looping` logit only, so A/B edit cycles and edit/revert cycles are named looping instead of recovering. |
| M2 | model.language_model.layers.22.mlp.gate_proj.weight, model.language_model.layers.22.mlp.up_proj.weight, model.language_model.layers.22.mlp.down_proj.weight | STALLED | Unit 1613 fires when identical failing blocks or a long retry chain dominate the window without a resolution, progress or edit repetition, and adds its activation to the ` stalled` logit only, so repeated identical failures are named stalled instead of recovering or looping. |
| M3 | model.language_model.layers.21.mlp.gate_proj.weight, model.language_model.layers.21.mlp.up_proj.weight, model.language_model.layers.21.mlp.down_proj.weight | RECOVERY | Unit 1007 reads errors-present times recent-success and moves only the ` recovering` logit, lowering it on stalled windows and raising it in the blocks after a resolution, so recovering is named for resolved or in-policy retries rather than as the default for every error. |
| M4 | label rows | PRODUCTIVE | The ` productive` row gains the progress-plus-resolution-minus-repeated-failure direction of the final residual, so windows with progress or new investigation are named productive even when error lines are in context. |
