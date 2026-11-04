# M3: Phase B pre-registration (spec v0.2 §9.4)

Committed **2026-11-04, before any network, reference-policy or oracle comparison has been run on any benchmark split.** Nothing below may be changed after validation scoring starts. A change requires a recorded spec revision.

## 1. Validation scoring (harness only)

- **Split.** The harness scores only the **validation** split. That covers A-strict, A-selfcheck and `reference_policy.py` alone. The scoring script hard-codes `{"val"}` and has no override. The sealed test and OOD sets are scored once, in M4, by a separate script.
- **Metrics.** The §10.1 metrics, each with 95% session-bootstrap confidence intervals (10,000 resamples, fixed seed):
  - macro-F1;
  - per-class precision and recall;
  - false-loop alarm rate;
  - confusion matrix;
  - per-family breakdown;
  - H2 fidelity (network vs `reference_policy`);
  - prediction distribution on oracle-ambiguous windows.
- **Code sources.** `reference_policy.py` and `artifact.dlw` are loaded from `weight_program/<phase>/`, the archived copies, never from the author directory.
- **Where results go.** Reports are written to `artifacts/reports/m3_val_<phase>.json`. **No aggregate validation metric is ever given to the weight author.** The only feedback Phase B allows is the disclosed report in §2.

## 2. Disclosed diagnostic report: sampling rule

- **Population.** Validation windows that pass the runtime gate **and** carry a scored oracle label (PRODUCTIVE, RECOVERY, STALLED or OSCILLATING). A window is identified by `(session_id, endpoint sequence_index)`.
- **Key.** Each window has the key `sha256(f"20261104:phase-b-disclosure:{session_id}:{endpoint}")`, compared as a hex string.
- **Selection.** The selection never depends on any model's output.
  1. For each oracle class, sort candidates by key, ascending.
  2. Walk the sorted list and take a window unless that class has already taken a window from the same session.
  3. Stop at **50**.
  4. This gives 4 × 50 = **200** windows. Windows from one session may appear under different classes.
- **Disclosure order.** All 200 windows are sorted by key. They get opaque ids `D001`…`D200` in that order.
- **Disclosed per window.** Exactly the four items §9.4 lists:
  1. the 8 × 15 feature matrix, with feature names;
  2. the oracle label;
  3. the A-selfcheck network's predicted class;
  4. the A-selfcheck `reference_policy` class.

  The report includes nothing else: no session id, family, endpoint, logits or aggregate statistics.
- **Records.** The disclosed file and its sha256 are committed as `experiments/m3/disclosure/`. A private mapping (`DISCLOSURE_PRIVATE.json`, repo only) records each `Dnnn`'s session id, endpoint and family. This lets M4 account for the disclosed windows.

## 3. Reading of "one revision"

- **Base and session.** The base is the A-selfcheck artefact (`561366300f…`). Phase B resumes the same weight-author session.
- **Allowed inputs.** The author may run forward passes and `reference_policy` on the 200 disclosed windows and on inputs it builds itself. The §9.3 permissions carry over.
- **Forbidden.** Automated or numerical weight search, fitting or distillation; any other data; any aggregate metric.
- **What counts as the revision.** The single revision is the **one submitted artefact** in `output/phase-b-revision/`, together with `reference_policy.py`, `CIRCUITS.md` and `REVISION.md`. `REVISION.md` logs every edit with its reason and the disclosed windows that motivated it. The author may iterate locally before submitting, but the harness builds and archives one submission.
- **Build rejections.** A rejected submission with a `BuildError` (no behavioural content) may be resubmitted under the existing M2 rule: error text only, at most 3 times.
- **Reporting.** Phase B is reported as **200 labelled examples, one revision**.

## 4. Isolation

- **Settings.** Phase B uses its own settings file:
  - write is denied on `output/phase-a-strict/`, `output/phase-a-selfcheck/`, `packet/` and `disclosure/`;
  - Bash is limited to `./py …`;
  - there is no repo access, web, MCP or subagents.
- **Audit.** The transcript is audited as in M2. The archived-deliverable write check is extended to Phase B.
