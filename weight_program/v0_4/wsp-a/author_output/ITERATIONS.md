# WSP-A iterations

All texts were written by me (hand-designed operation records rendered with `packet/renderer.py`; generators in
`scratch/scen_*.py`, library `scratch/mk1.py`, `scratch/mk2.py`). Every window of my sessions was labelled by my own
reading of the v0.2 oracle rules (`Session.labels()` in `scratch/mk1.py`); windows the rules leave in the grey zone
were marked AMBIG and never used. "acc" below is accuracy over my confidently-labelled windows and is only a
relative yardstick between edit states on the same texts.

## 0. Exploration of the original model (no edit; hash 938f4bc9…)

- **fwd call 1** (set A, 21 sessions, `scratch/scen_a1.py`): the original model names `recovering` for almost every
  window that contains an error line, `productive` for clean windows, and keeps `recovering`/`stalled`/`looping`
  within ~0.5 logit of each other on error-containing windows. acc 0.52.
- **fwd calls 2-4** (set B, 90 sessions, 45 recipes × 2 name tables, `scratch/scen_b.py`): baseline acc 0.40;
  recall PROD 0.33, RECO 0.71, STAL 0.55, OSCI 0.16 (set B). Mean-difference class directions at the readout
  generalise at only ~58-65% across session halves; the most discriminative MLP neurons track session length, so
  position in session was confounded with class in that set.
- **fwd calls 5-8** (set C, 144 matched pairs, `scratch/scen_c.py`): matched pairs isolating one property each. The
  mean pair differences are highly consistent (sign agreement: ERR 1.00, PROG 1.00, RES 1.00, REP 0.91, REV 0.94;
  leave-one-pair-out agreement the same). Finding: the "reversal" direction is essentially the repetition
  direction; alternating hunk ids minus fresh hunk ids, with the tool matched, has no separating power
  (d' ≈ −0.3 for OSCI vs STAL). The repetition direction saturates after two identical failing blocks and does not
  respond to repeated reads; resolution shows mostly in the 1-3 blocks after the success. The model encodes
  presence and recency, never counts.
- **fwd calls 9-11** (set D, 126 sessions in matched groups, `scratch/scen_d.py`): proxies for OSCILLATING:
  EDITS (edit blocks present among repeated failures; leave-one-out agreement 1.00), EDITREPfree (alternating
  edits vs repeated reads, error-free; 1.00), CHAIN (11 pending polls vs 4; 0.90). Hunk lines present vs no-op
  edits is barely encoded (LOO 0.15). Also found neurons that never fire at the readout position (max |act| <
  0.01 over 2212 windows): layer 23 #1046, layer 22 #1613, layer 21 #1007.

## 1. Edit v1c (hash 91863b69…)

Design (`scratch/derive_v1.py`, `derive_v1b.py`, `derive_final.py` + `params_v1c.json`): two hand-built SwiGLU
neurons plus a PRODUCTIVE readout direction in the label rows (see EDIT.md for the exact form).
- OSC neuron (layer 23, #1046): gate = edit-type repetition (z_EDITS + z_EDITREPfree), up = no progress
  (−z_PROG − z_RES) + 1.5, output along the looping wire.
- STAL neuron (layer 22, #1613): gate = z_REP + 2 z_CHAIN, up = −z_RES − z_PROG − z_EDITS + z_CHAIN, output along
  the stalled wire projected into the null space of layer 23's value projection.
- PROD row: + 1.6 (z_PROG + z_RES − z_REP).
Predicted (direct path) acc on B+D: 0.59 (from 0.35).
- **fwd calls 12-14** (set E, 60 fresh sessions, new vocabularies, `scratch/scen_e.py`, model=both): acc 0.49
  vs 0.40 original; recall PROD 0.56, RECO 0.42, STAL 0.86, OSCI 0.34. Observed logit deltas match the
  direct-path prediction (corr ≥ 0.995, residual std ≤ 0.34): the neurons act only through the readout
  position. Problems: STALLED fired on short retry chains (≤3) and post-stall tails; the OSC neuron fired on
  incremental and diagnosed repairs; STALLED beat OSCILLATING when failing validation sat between edits.

## 2. Edit v2b (hash 551152d6…)

Revision by reasoning on the set-E feature table (`scratch/ana_e2.py`; `scratch/derive_v2.py` + `params_v2b.json`):
- OSC up = T + 1.5 e − 1.5 with T = −z_PROG − z_RES + z_REP + z_CHAIN (trouble) and e the gate score, so that
  edit repetition fires only together with trouble; scale lowered to 0.14.
- STAL gate threshold raised to 2.5 (where ~90% of STALLED windows pass and ~10% of RECOVERY ones), up term
  penalises edit features twice (−2 z_EDITS − 2 z_EDITREPfree) so that oscillation wins on edit-heavy windows.
- New RECO neuron (layer 21, #1007): gate = z_ERR, up = z_RES + 0.9; lowers `recovering` on stalled windows and
  raises it right after a resolution.
Predicted acc on B+D: 0.65.
- **fwd calls 15-17** (set E again, returning layer-21 activations): acc 0.59; recall 0.56/0.59/0.70/0.60;
  leakage check residual std ≤ 0.55 for all three neurons.

## 3. Edit v2c (hash 3ef7a5ee…)

`params_v2c.json`: STAL scale 0.07 → 0.045 (the SwiGLU product is unbounded; moderate activations on retry
chains were enough to flip RECOVERY), OSC up = T + 2 e − 2.5 (repairs come out near zero), RECO scale 0.4.
- **fwd calls 18-20** (set E, third and last edit state on these texts): acc 0.62; recall 0.57/0.68/0.70/0.62;
  precision 0.77/0.63/0.37/0.60.
- **fwd calls 21-23** (set F, 54 fresh sessions, new vocabularies and recipes, `scratch/scen_f.py`): acc 0.57 vs
  0.38 original; recall PROD 0.41, RECO 0.64, STAL 0.75, OSCI 0.74. Dominant error: incremental repair (edit,
  failing tests with improving counts, edit, …) is named `looping`.
- **fwd calls 24-25** (set G, `scratch/scen_g.py`, original model): matched pairs for "pass counts improving vs
  constant" (IMPROV) and "four edits vs one edit among repeated failures" (EDITCOUNT). IMPROV is weak and
  inconsistent (LOO 0.33 ± 0.3): the model does not linearly encode improving counts at the readout, so family 1
  vs family 7 remains an accepted limitation. EDITCOUNT is consistent (LOO 0.77, agreement 1.00) and clearly
  negative on RECOVERY windows.

## 4. Edit v3b (final; hash 74576466ac6d22c6cc6d7545e4162a0ae2daf38a6049624e232caf635910079a)

`scratch/derive_v3.py` + `params_v3b.json`: OSC up = T + 1.5 e + 1.0 z_EDITCOUNT − 2.5. In the neuron's own units
incremental-repair windows have trouble scores of about 2 to 2.5 and A/B oscillation 3.6 to 5.3, so the offset −2.5
zeroes most repairs while the edit-repetition and edit-count terms keep revert cycles positive. Everything else as
v2c. Predicted acc on B+D: 0.66.
- **fwd calls 26-28** (set F, second edit state on these texts): acc 0.56 vs 0.38 original; recall PROD 0.41,
  RECO 0.70, STAL 0.77, OSCI 0.65; precision 0.74 / 0.53 / 0.48 / 0.48. Leakage check: residual std ≤ 0.50 for all
  classes (corr ≥ 0.98). Compared with v2c on the same texts (0.57): oscillation recall drops (the error-free
  A→B→C→A cycle with passing builds is lost), recovery recall rises, and the unavoidable incremental-repair errors
  move from `looping` to `recovering`; v3b names `looping` on other-class windows less often than v2c. The two are
  within noise on accuracy; I keep v3b as the more conservative oscillation mechanism.

## Final state

- Kernel edit hash `74576466ac6d22c6cc6d7545e4162a0ae2daf38a6049624e232caf635910079a`; `scratch/verify_build.py`
  confirms that `build_edit.build(orig, consts)` reproduces every edited tensor exactly and that two builds are
  identical.
- Forward-pass meter: 28 requests, 1,113,610 tokens of 2,000,000. Edit states run: 938f4bc9 (original, calls 1-11,
  24-25), 91863b69 (v1c, calls 12-14), 551152d6 (v2b, calls 15-17), 3ef7a5ee (v2c, calls 18-23), 74576466 (v3b,
  calls 26-28). Set E texts were run under three edit states (v1c, v2b, v2c), set F under two (v2c, v3b); no text set
  was run under more than three edit states.
