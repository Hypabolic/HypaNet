# M3 Phase B runbook

Operational steps for Phase B. The binding rules are the pre-registration in `README.md` (frozen) and spec v0.2 §9.4; if this file and those differ, they win. Run everything from the repo root. Nothing here touches the sealed test or OOD sets.

## Files

| File | Role |
|---|---|
| `scripts/m3_build_disclosure.py` | Builds `experiments/m3/disclosure/DISCLOSURE.json` + `DISCLOSURE.sha256` (and the repo-only `DISCLOSURE_PRIVATE.json`) by the §2 sampling rule |
| `experiments/m3/settings_phase_b.json` | Phase B tool permissions: Bash only `./py …`; Write/Edit only in `output/phase-b-revision/`; reads only `packet/`, `disclosure/`, `tools/`, `output/` |
| `experiments/m3/prompt_phase_b.md` | Phase B instruction; `{A_SELFCHECK_SHA256}` and `{DISCLOSURE_SHA256}` are substituted at launch |
| `scripts/m3_prepare_author_dir.sh` | Verifies and copies the disclosure into the author dir (read-only), creates `output/phase-b-revision/`, writes `experiments/m3/HANDOVER.json` |
| `scripts/m2_launch.sh phase-b-revision` | Resumes the weight-author session (`experiments/m2/SESSION_ID`); logs go to `experiments/m3/logs/` |
| `scripts/m2_audit_session.py … phase-b-revision` | Run automatically after the launch; writes are violations in `output/phase-a-strict/`, `output/phase-a-selfcheck/`, `output/selfcheck_scripts/`, `disclosure/`, `tools/`, `packet/` |

## Steps

1. **Build the disclosure.**
   `uv run python scripts/m3_build_disclosure.py`
   Check that it holds 200 windows, 50 per class, ids `D001`…`D200`, and per window only the features, oracle label, A-selfcheck network class and A-selfcheck `reference_policy` class.

2. **Commit the disclosure** (`experiments/m3/disclosure/`, including `DISCLOSURE_PRIVATE.json`) **before** handing it over.

3. **Hand it over.**
   `scripts/m3_prepare_author_dir.sh` (default author dir `../HypaNet-m2-author`)
   This refuses if the disclosure, the A-selfcheck artefact or the author-dir packet fails its hash check, if `disclosure/` already exists in the author dir with different content, or if `output/phase-b-revision/` is not empty.

4. **Commit `experiments/m3/HANDOVER.json`** (it records the disclosure sha, the A-selfcheck artefact sha, the packet sha, and the prompt and settings hashes).

5. **Launch Phase B.**
   `scripts/m2_launch.sh phase-b-revision`
   The launch refuses unless `weight_program/phase-a-selfcheck/artifact.sha256` matches the archived artefact, the author-dir disclosure copy matches `experiments/m3/disclosure/DISCLOSURE.sha256`, and `HANDOVER.json` records both hashes.
   - If a response ends with an API or output-limit error before anything was written: `scripts/m2_launch.sh continue phase-b-revision K` (fixed, feedback-free prompt).

6. **Check the audit.** Read `experiments/m3/logs/phase-b-revision.stream.audit.json`:
   - `models`: which model actually wrote the revision. `--fallback-model` can switch mid-session; record it in the archive commit.
   - `executed_violations` must be 0. Refused attempts are listed but were blocked.

7. **Build and archive the one revision.**
   `uv run python scripts/run_build.py phase-b-revision --author-dir ../HypaNet-m2-author/output/phase-b-revision --packet-dir ../HypaNet-m2-author/packet --out-dir weight_program/phase-b-revision`
   - On a `BuildError` (no behavioural content): `scripts/m2_launch.sh resubmit phase-b-revision N ERROR_FILE`, with N = 1..3, error text only. Then audit and build again.
   - The runner archives only the four required files. Copy the author's analysis scripts (`output/phase-b-revision/scratch/` and anything else in that directory) to `experiments/m3/revision_scripts/` so the whole directory is archived, as was done for `experiments/m2/selfcheck_scripts/`.

8. **Commit the archive** before any scoring: `weight_program/phase-b-revision/`, `experiments/m3/logs/`, `experiments/m3/revision_scripts/` and any `experiments/m3/resubmissions/`. The commit message records the model(s) from the audit and the number of resubmissions.

9. **Score on validation only.**
   `uv run python scripts/score_validation.py phase-b-revision` → `artifacts/reports/m3_val_phase-b-revision.json`. Never pass any result back to the weight author.
