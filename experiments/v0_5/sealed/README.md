# v0.5 sealed evaluation: pre-registration (6 Nov 2026)

Committed before `scripts/v05_sealed_eval.py --sealed --confirm-one-time` runs. Matthew approved scoring the
v0.5 networks on v0.3's sealed sets on 6 Nov 2026 ("do both", after the val paired test was proposed with it).

## What is scored, once

- **Sets:** v0.3's sealed test (3,996 sessions) and sealed OOD (2,000 sessions, Codex's independent generator),
  plus val and the unsealed counterfactual set, by v0.3's own loaders (`agent_loop_lab.v03.sealed.build_split`),
  windows, gate, populations and metrics. The sealed files are verified against the hash records committed
  before any v0.3 authoring; test and OOD rows must hash to the `rows_sha256` recorded in
  `artifacts/reports/v03_sealed.json` (fatal: otherwise the attempt is spent and no report is written).
- **New columns:** v0.5 r1 (`experiments/v0_5/runs/r1/artifact.dlw`, sha256 `46d4b2a2…`) and v0.5 Phase B
  (`weight_program/v0_5/phase-b/artifact.dlw`, sha256 `223d0743…`, base checked to be r1).
- **Reference columns, re-scored in the same run:** every v0.3 column (A and B networks and reference
  policies, the 11 Phase C models, both priors). They must reproduce `v03_sealed.json`; a mismatch is recorded
  loudly in the report (WARNING note, `v03_sealed_reproduced: false`, a caveat on every pair that uses the
  column), not hidden.

## Primary comparisons (paired, shared session bootstrap, seed 20261104, 10,000 resamples)

Reported for macro-F1 and false-alarm rate on test and OOD (and val, counterfactual):

1. v0.5 B − v0.3 B (the headline: written without code + 200 labels, against written with code + 200 labels)
2. v0.5 B − v0.3 C-all seed0 (against the trained network)
3. v0.5 B − v0.3 A, v0.5 B − v0.5 r1, v0.5 r1 − v0.3 A

No other comparison is a primary result. Per-family breakdowns are reported as v0.3 reports them.

## Order and refusals (as v0.3, `experiments/v0_3/sealed/README.md`)

All input hashes (v0.5 artefacts, v0.3 columns, sealed hash records, unsealed data, code, this file); val and
counterfactual scoring with every check (v0.5 val reports and v0.3 official val reports reproduced exactly,
Phase C early-stopping check, v0.2 identity, scoring path, disclosure); isolated latency; then the attempt
record `artifacts/reports/v05_sealed.started.json`; then the sealed files are hashed, decoded, identity-checked
and scored in memory; the report `artifacts/reports/v05_sealed.json` is written atomically and metrics are
printed only after that. The script refuses if the report or the attempt record exists. There is no `--force`.

## Differences from v0.3's sealed protocol

- Extra columns (the two v0.5 networks) and the pairs above.
- v0.3 columns are re-scored and their reproduction of `v03_sealed.json` is recorded, not fatal (the v0.3 run
  was on x86_64; this machine is arm64). The full val dry run reproduced every v0.3 val and counterfactual
  row exactly here (`experiments/v0_5/sealed_dryrun/v05_sealed_dryrun_val.json`).
- New fatal check: test and OOD rows must hash to v0.3's recorded `rows_sha256`.
- The v0.5 networks were never shown val metrics, test or OOD. Their only labelled input was v0.3's 200
  disclosed val windows (Phase B).

## Dry run (6 Nov, before this commit)

Full val, 650 s. Paired on val (excluding_disclosed): v0.5 B − v0.3 B macro-F1 +0.0028 [−0.0011, +0.0068],
false alarms +0.0077 [+0.0062, +0.0092]; v0.5 B − v0.3 C-all +0.0313 [+0.0264, +0.0366].
