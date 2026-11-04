# v0.3 Phase B pre-registration and validation scoring (spec v0.3 §9.3, §10, E7, E11)

Written **2026-11-05**, while the v0.3 Phase A author session is running and before any author-written v0.3 network or `reference_policy` has been run on any benchmark split. The only val runs so far are harness dry runs of an all-zero network and a constant policy, which test the tooling. The rules below are fixed from now on. Changing one needs a recorded spec revision.

## 1. Validation scoring (harness only)

- **Split.** The harness scores the **validation** split only. `scripts/v03_score_validation.py` and `src/agent_loop_lab/v03/scoring.py` hard-code `val`, and no option, argument or environment variable changes this. The sealed test and OOD sets are scored once, at the times E7 sets, by a separate script.
- **Phases.** `v03-phase-a` and `v03-phase-b` are scored from the archived directories `weight_program/v0_3/phase-a/` and `weight_program/v0_3/phase-b/`. These are the paths the frozen `scripts/v03_launch.sh` and `scripts/v03_run_build.py` use. The script never reads the author directory. Before scoring it checks:
  - `artifact.dlw` must match `artifact.sha256`;
  - the artefact metadata must name the phase, its `packet_sha256` must equal the registered `experiments/registry_v0_3.yaml` value, and its `reference_policy_sha256` must match the archived `reference_policy.py`;
  - `reference_policy.py` must pass the v0.3 static rules (`source_rules.check_reference_policy`).
- **Inputs.** For each val session, the RawOps (`datasets/v0_3/val.raw.jsonl.gz`, same line order as the v0.2 files) are rendered with the frozen renderer and tokenised with the frozen tokeniser. Labels come from the v0.2 oracle file (`datasets/v0_2/standard/val.oracle.jsonl.gz`). The scored mask comes from the v0.2 runtime gate (`agent_loop_lab.windows.gate`), run over the v0.2 canonical events. A window is named by `(session_id, endpoint)`, where `endpoint` is the `sequence_index` of its last event. It reads block `t` = the event at list position `t`.
- **Network (H3).** There is one causal forward pass per session (`infer.eoe_logits`, float32, CPU, `torch.no_grad()`). The class of window `t` is the argmax at `EOE_t`. Ties go to the first index, so all-zero logits give PRODUCTIVE.
- **Reference policy (H1).** The archived `reference_policy.py` is called once per **gate-passing** window. Scored windows are needed for H1 and all gate-passing windows for H2. Its input is the surface-token prefix `surface[:eoe_index[t] + 1]` and `event_ord[:eoe_index[t] + 1]`: `BOS`, blocks `0..t`, and `EOE_t` included, the same form as the build runner's smoke call.
  - **Isolation.** Calls go only through `runner.evaluate_policy`. Each chunk of sessions runs in a fresh interpreter under the BUILD guard (no I/O, no RNG). A private copy of `reference_policy.py` is the only author file present. At most 2 chunks run at a time. The scoring process never imports the policy.
  - **Guard violations.** If a policy call breaks the guard (I/O, RNG), the frozen runner raises a `BuildError`. That aborts scoring for the phase; it is not a per-window failure.
  - **Failures.** Each call gets 0.5 s of CPU. A timeout, an exception or a value that is not an `int` in 0..3 is a **policy failure**.
- **Policy failures.** These counts are reported by window population: scored, ambiguous, instrumentation. In the metrics:
  - **H2:** a failure is a disagreement.
  - **H1:** a failure is a wrong prediction. For counting, it is recorded as the lowest class index that is neither the network's class nor (for a scored window) the oracle class. With zero failures the metrics are unchanged.
- **Metrics.** These are the v0.2 §10.1 metrics, computed with `eval/metrics.py` (`score_report`):
  - macro-F1;
  - per-class precision, recall and F1;
  - accuracy;
  - false-loop alarm rate;
  - confusion matrix;
  - prediction distribution on gate-passing ambiguous and instrumentation windows;
  - H2 fidelity (network vs `reference_policy`) on all gate-passing windows and on scored windows;
  - the per-family breakdown.

  They are reported for the network (H3) and for the policy (H1). Each has a 95% CI from a session bootstrap: 10,000 resamples, seed `20261104` (`metrics.BOOTSTRAP_SEED`), per-family seeds from `metrics.family_seed`. Also reported: CPU time of the forward passes per session, and the non-zero parameter count.
- **Two populations** (spec §10):
  1. **`excluding_disclosed`**: all val sessions except the 200 Phase B disclosed sessions (§2). This is the **primary** v0.3 validation result, for both phases.
  2. **`all_val`**: every val session, reported for transparency.

  The disclosed set comes from the rule in §2 alone. It uses no model output, so it is the same before and after Phase B exists.
- **Where results go.** Reports go to `artifacts/reports/v03_val_<phase>.json`. The script refuses to overwrite a report unless given `--force`. **No aggregate validation metric is ever given to the weight author.** E11: the first archived Phase A artefact is the v0.3 Phase A result. Any later author launch is a new registered version, v0.3.k.

## 2. Disclosed diagnostic report: sampling rule (spec §9.3)

- **Population.** Validation windows that pass the v0.2 runtime gate **and** carry a scored oracle label (PRODUCTIVE, RECOVERY, STALLED or OSCILLATING). A window is `(session_id, endpoint)`, with `endpoint` its last event's `sequence_index`.
- **Key.** Each window's key is `sha256(f"20261105:v03-phase-b-disclosure:{session_id}:{endpoint}")`, a lowercase hex string compared as a string.
- **Selection.** The selection never depends on any model output.
  1. Take the classes in canonical order: PRODUCTIVE, RECOVERY, STALLED, OSCILLATING.
  2. For each class, sort its candidate windows by key, ascending.
  3. Walk the sorted list. Take a window unless its session is already used, by **any** class, including earlier classes. Once a window is taken, its session is used.
  4. Stop the class at **50** windows.
  5. This gives 4 × 50 = **200** windows from **200 distinct sessions**. A class that cannot reach 50 is an error, and nothing is written.
- **Disclosure order and ids.** The 200 windows are sorted by key, ascending, and numbered with opaque ids `V001`…`V200` in that order.
- **Disclosed per window.** Exactly these four items:
  1. **`prefix`**: the rendered prefix of blocks `0..t` as one text. The blocks are rendered by the frozen renderer, are in order and are joined with `"\n"`. A single line reading exactly `### window start` is inserted immediately before block `t−7`, the window's first block. Block `t` is the window's last block, and the text ends with it.
     - **The marker is not log text.** Remove that one line before rendering or tokenising.
     - **Recovering the blocks.** After removing the marker, every line that does not start with a space starts a new block: each header begins with `HH:MM:SS.mmm`, or `?` when the timestamp was dropped. Body lines start with two spaces.
     - **Round trip.** The builder checks that this parse gives back exactly the rendered blocks, and that no log line equals the marker.
  2. **`oracle_label`**: the window's oracle label.
  3. **`network_class`**: the archived Phase A (`v03-phase-a`) network's argmax at `EOE_t`, from one causal forward pass over the whole session. That equals the pass over the prefix, because the model is causal.
  4. **`reference_policy_class`**: the class returned by the archived Phase A `reference_policy.py` on the surface-token prefix through `EOE_t`, under the §1 guard and CPU limit. It is `null` if the call failed (timeout, exception or invalid value).

  The disclosed file also has these, and nothing else:
  - a format tag;
  - the class names;
  - the marker string;
  - one-sentence descriptions of the fields, naming the Phase A artefact sha256 (the author already holds that).

  It has no session id, family, endpoint, logits, scores, per-class or aggregate statistics, or anything from the oracle beyond the label. Hunk ids and error line numbers inside the rendered text are per-session hashes, part of the rendered log. They are disclosed as rendered.
- **Records** (revised r1, §4).
  - `experiments/v0_3/disclosure/DISCLOSURE.json` and `DISCLOSURE.sha256` (`"<sha256>  DISCLOSURE.json"`) are the disclosed file and its hash. These are the paths the frozen `scripts/v03_launch.sh phase-b` checks.
  - `experiments/v0_3/DISCLOSURE_PRIVATE.json` is repo only and never handed over. For each `Vnnn` it records the session id, endpoint, family, oracle label and key, plus the hashes of the inputs. Scoring and the sealed evaluation use it to account for the disclosed sessions.
  - `scripts/v03_build_disclosure.py` refuses to overwrite these files unless given `--force`.
- **Handover.** `scripts/v03_prepare_phase_b.sh` checks the archived Phase A artefact against its `artifact.sha256`, `DISCLOSURE.json` against `DISCLOSURE.sha256` and the author-dir packet against the registry. It then copies `DISCLOSURE.json` and `DISCLOSURE.sha256` (only these two files) into `AUTHOR_DIR/disclosure/`, read-only. An existing copy must be byte-identical. The script makes `output/v03-phase-a/`, `tools/` and `packet/` read-only, creates an empty `output/v03-phase-b/` and writes the hashes handed over to `experiments/v0_3/phase_b/HANDOVER.json`. It launches nothing.
- **Source of the predictions.** Only the archived `weight_program/v0_3/phase-a/` (`v03-phase-a`). The disclosure is built once, after the Phase A archive and before Phase B starts.

## 3. Reporting

- Phase A is reported as **0 labelled examples**.
- Phase B is reported as **200 labelled examples, 1 revision**.
- Validation metrics of both phases are reported on `excluding_disclosed` first, with `all_val` alongside.

## 4. Revisions

- **r1 (2026-11-05): disclosure location only.** The first text of §2 put the disclosed file at `experiments/v0_3/phase_b/disclosure/` and the private record at `experiments/v0_3/phase_b/DISCLOSURE_PRIVATE.json`. The frozen `scripts/v03_launch.sh` (`experiments/registry_v0_3.yaml`) checks the registered disclosure at `experiments/v0_3/disclosure/DISCLOSURE.{json,sha256}`, and it refuses to start Phase B when that file is missing. §2 Records now names the launcher's path, and the private record moves next to it, to `experiments/v0_3/DISCLOSURE_PRIVATE.json`. The frozen launcher also needs a copy of the disclosure in `AUTHOR_DIR/disclosure/`, so §2 adds the handover step (`scripts/v03_prepare_phase_b.sh`). The sampling rule, the disclosed contents, the source of the predictions and §1 and §3 are unchanged. No disclosure had been built and Phase A had not been archived when this revision was made.
