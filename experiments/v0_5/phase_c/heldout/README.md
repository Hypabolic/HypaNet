# v0.5 Phase C held-out evaluation: pre-registration (7 Nov 2026)

Committed before the Phase C author session is launched, and before
`scripts/v05c_heldout_eval.py --heldout --confirm-one-time` can run (the script refuses unless this file,
`HELDOUT_MANIFEST.json`, `../DISCLOSURE_OOD_PRIVATE.json` and the scoring code are committed and unchanged, and
the scoring code is byte for byte what the manifest pinned before launch). Matthew decided the
Phase C design on 7 Nov 2026: a new Fable session revises v0.5 Phase B's network with v0.3's 200 val windows
(again, re-labelled with B's classes) and 200 new labelled windows from the OOD generator, restricted to 6 of
its 10 families; it is evaluated on fresh held-out sets, not on the sealed sets.

The sealed v0.2/v0.3 test and OOD sets are **not** opened, hashed or reused by Phase C.

## 1. Seeds (fixed; `agent_loop_lab.v05.phase_c_data`)

| Role | Master seed | Generator |
|---|---|---|
| OOD disclosure (handed over) | 20261109 | `synth_ood`, 6 seen families, 200 sessions each |
| Held-out OOD (never handed over) | 20261110 | `synth_ood`, all 10 families, 200 sessions each (2,000) |
| Held-out in-distribution (never handed over) | 20261111 | standard generator, 4,000 sessions over the 12 `FAMILY_NAMES` |

Refused everywhere (`check_seed`): 20261104 (the sealed sets' master seed), 20261107 and 20261108 (used by the
post-hoc OOD diagnosis), and any role's seed used for another role. OOD sessions use `ood_seed(master_seed,
family, k)` and ids `o-{seed:016x}` (as `datasets/ood.py`); standard sessions use `session_seed(master_seed,
family, k)`, `family_counts(4000, FAMILY_NAMES)` and ids `s-{seed:016x}` (as `datasets/build.py`).

**Disjointness** (asserted when the held-out sets are built and again when they are scored): no held-out session
id is in the id space of seeds 20261104, 20261107, 20261108 or 20261109 (OOD and standard formulas, over the OOD
and standard family names, k < 2,000, computed from the formulas; no data file is opened), nor among the
disclosed windows' sessions (`DISCLOSURE_OOD_PRIVATE.json`); the two held-out sets are disjoint.

**Re-timing of OOD sessions** (Matthew, 7 Nov 2026, before launch and before any Phase C data was handed over or used for scoring; an
un-re-timed OOD disclosure had been built, and B's confusion on it computed, then discarded and rebuilt).
`synth_ood`'s engine starts every session's clock at 2026-03-14 00:00:00 UTC, so every rendered block header of
its sessions would read `00:0x:..` (the renderer prints the time of day and the network tokenises it), while the
standard generator starts each session at a random day and a uniform random second of the day: the source
generator would be detectable from the clock alone. Every `synth_ood` session Phase C generates (the OOD
disclosure's candidates, seed 20261109, and the held-out OOD set, seed 20261110) is therefore re-timed by one
per-session offset of whole seconds,
`offset = int.from_bytes(sha256(f"v05c-retime:{session_id}").digest()[:8], "big") % 86400` (in [0, 86399]),
added to every timestamp of the session, in its canonical events and its raw ops alike, before anything is
rendered, tokenised or written (`phase_c_data.retime_offset_s`, `retime_session`, applied in `generate_ood`).
Event order, gaps, every other field, the oracle record and so the window rows and labels are unchanged; the
raw-op/event alignment check runs before and after the shift. The renderer's per-operation noise tokens (`req=...`, and the
`(pid N)` / `(0x...)` suffix of error lines) do change, because `v03.renderer.render_rng` seeds them from the
operation's canonical JSON, which includes its timestamp; they carry no label information. `synth_ood` itself is not modified (its source
hash is recorded). The in-distribution held-out set, val and the original disclosure are not re-timed. The rule
is recorded in `HELDOUT_MANIFEST.json` (`sets.heldout_ood.retime`; the evaluation refuses a manifest without
it) and in `../DISCLOSURE_OOD_PRIVATE.json` (`retime`, and each window's `retime_offset_s`); the handed-over
`DISCLOSURE_W.json` does not mention it.

## 2. Family rule (fixed before any Phase C data was generated)

Sort the 10 `synth_ood` family names by `sha256(f"20261109:v05c-family:{name}").hexdigest()` and take the first 6
as **seen** (the OOD disclosure draws only from them); the other 4 are **unseen**. Result:

- **seen:** fixture_archaeology, annotation_spiral, merge_ping_pong, shard_migration, telemetry_gaps,
  pinned_validation
- **unseen:** lease_reconnect, index_watermark, lockfile_churn, serializer_duel

The author is told only that the new windows come from "a second, independently written log generator"
restricted to some of its scenario families; neither which nor how many families are held out, nor that its
sessions were re-timed.

## 3. The OOD disclosure (`scripts/v05c_build_disclosure.py`)

Candidates: 200 sessions in each seen family (seed 20261109), re-timed (§1), rendered and tokenised by the v0.3
renderer and tokeniser, windows and runtime gate as v0.3. Selection: v0.3's rule by import (`v03.scoring.disclosure_selection`:
gate passed, scored oracle label, 50 per class in class order, ascending fixed hash key of (session, endpoint), at
most one window per session); no model output is used. Ids W001..W200 in key order; each window has `prefix`
(with the `### window start` marker, `render_disclosed_prefix`), `oracle_label` and `network_class` (v0.5 B's
argmax). The handed-over file is `disclosure/DISCLOSURE_W.json` (format `v05-phase-c-disclosure-w/1`, Phase B's
header keys only); neither its name nor its header says "OOD". No family name, session id, endpoint or seed is in
the handed-over file (checked by the builder and by `kit_c check-disclosure`); the
private record `experiments/v0_5/phase_c/DISCLOSURE_OOD_PRIVATE.json` keeps them. The original disclosure
(V001..V200) is handed over again with `network_class` = B's class; its prefixes and labels are checked equal to
v0.3's and to Phase B's.

## 4. Held-out sets (`scripts/v05c_build_heldout.py`, run before launch)

Data in `datasets/v0_5c/{heldout_ood,heldout_indist}/` (gitignored); `HELDOUT_MANIFEST.json` (committed)
records each file's sha256, the sha256 of the window rows (`[[session_id, family, endpoint, oracle label,
gate_passed], ...]`), the sha256 of the token encoding, the counts, seeds, the re-timing rule (§1), family rule
and split, and the source sha256 of both generators, the renderer, the tokeniser, the raw-op module and the
window/gate module, and the sha256 of the scoring code (`phase_c_data.SCORING_CODE_FILES`: `scripts/v05c_heldout_eval.py`,
`v05/phase_c_eval.py`, `v05/phase_c_data.py` and the shared scoring, metric, model, renderer and tokeniser
modules they import), which implements §5-§7. The scoring code is therefore final when the official manifest is
built; a later change needs a rebuilt manifest (deterministic data, same hashes) committed before launch. The
launcher (`scripts/v05c_launch.sh`, real mode) refuses unless this file, exactly `HELDOUT_MANIFEST.json`
(official, passing the checks below), `../DISCLOSURE_OOD_PRIVATE.json` and the scoring code are committed
unchanged (`python -m agent_loop_lab.v05.phase_c_eval prelaunch-check`). The evaluation refuses a manifest whose
seeds, sizes, family split, re-timing rule, source hashes or scoring-code hashes differ, verifies every file hash before the
attempt record and, after it, that the decoded rows and encoding hash to the manifest's values.

Plus **val**: the existing split, `excluding_disclosed` (every val session except v0.3's 200 disclosed ones;
primary, as in v0.3 and v0.5) and `all_val` (transparency). Val is reference only: those 200 sessions were shown
to Phase B and are shown again to Phase C, and v0.3 C-all early-stopped on val.

## 5. Columns

| Column | Artefact |
|---|---|
| **v0.5 Phase C** | `weight_program/v0_5/phase-c/artifact.dlw` (checked: `artifact.sha256`, `apply.json`, `BUILD_RECORD.json`, metadata phase `v05-phase-c`, base = B's artefact `223d0743…` in metadata and build record) |
| v0.5 B | `weight_program/v0_5/phase-b/artifact.dlw` (`223d0743…`, base r1) |
| v0.5 r1 | `experiments/v0_5/runs/r1/artifact.dlw` (`46d4b2a2…`) |
| v0.3 A, v0.3 B | `weight_program/v0_3/phase-{a,b}/artifact.dlw` (networks only; `v03.scoring.load_phase`) |
| v0.3 C-all seed0 | `experiments/phase-c-trained/v0_3/all/seed0/model.pt` |
| prior always-PRODUCTIVE | reference only, not compared |

Predictions: one causal forward pass per session, argmax at each window's last EOE (the v0.3 path,
`v05.scoring.network_predictions`). Metrics as v0.3 (`eval.m4.score_columns`): macro-F1, false-alarm rate,
accuracy, per-class precision/recall/F1, confusion, gate report, overall and per family; 95% session-bootstrap
intervals, 10,000 resamples, seed 20261104 (per-family seeds derived as v0.3), all columns on the same draws.

## 6. Pre-registered comparisons (paired, shared session bootstrap)

Macro-F1 difference (`v04.scoring.paired_macro_f1`) and false-alarm-rate difference (same computation) with
95% percentile intervals over the shared draws of the population.

**Primary:** on the held-out OOD set, **UNSEEN families**:

1. v0.5 Phase C − v0.5 B
2. v0.5 Phase C − v0.3 B

**Then, in this order:** the same two pairs on held-out OOD SEEN families; held-out OOD ALL families; held-out
in-distribution; val (`excluding_disclosed`).

Also reported in every population, not primary: Phase C − r1, Phase C − v0.3 A, Phase C − v0.3 C-all seed0, and
v0.5 B − v0.3 B (the sealed-run comparison repeated on fresh data). Per-family results are descriptive. A
difference is called a difference only when its 95% interval excludes 0; no other comparison is a primary result,
and no threshold or set is chosen after seeing held-out numbers.

## 7. Order and refusals (`scripts/v05c_heldout_eval.py --heldout --confirm-one-time`)

Refuses if `artifacts/reports/v05c_heldout.json` or the attempt record `artifacts/reports/v05c_heldout.started.json`
exists (no `--force`), or if this file, `HELDOUT_MANIFEST.json`, `DISCLOSURE_OOD_PRIVATE.json` or any scoring
code file is missing, untracked or differs from HEAD. Then: all columns checked (Phase C's base must be B); manifest checks and held-out
file hashes (nothing decoded); val with its checks (fatal: val rows = v0.3's sealed record, the v0.3 disclosure
rule, v0.5 B's disclosure record, r1 and B reproduce their val reports exactly; recorded: the v0.3 columns' val
blocks against `v03_sealed.json`, with a caveat on any pair using a column that does not reproduce); the attempt
record; then each held-out set is decoded, checked against the manifest and the disclosure, predicted and scored
in memory; the report is written atomically and metrics are printed only after that. A failure after the attempt
record leaves it as a spent attempt (renamed `v05c_heldout.attempt<N>.json` by the adjudicator if a rerun is
authorised; earlier attempts are listed in the next report).

The dry run (`--dry-run-val`) scores val only, never reads held-out data or the attempt record, writes a report
marked `"official": false` outside `artifacts/reports/`, and may put a stand-in artefact in the Phase C column
(`--phase-c-artifact`; the report is then marked `"phase_c_stand_in": true` with a STAND-IN warning).

## 8. What the author never sees

The held-out sets, their manifest, this file, the private disclosure records, any val, test, OOD or held-out
metric, the sealed results, `STATUS.md`, `artifacts/reports/` and `experiments/v0_5/ood_analysis/`.
