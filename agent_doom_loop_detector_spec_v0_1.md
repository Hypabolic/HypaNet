# Agent Doom Loop Detector — Direct Weight Programming Experiment

**Specification:** v0.1  
**Status:** ready for implementation  
**Purpose:** Evaluate whether a reasoning LLM can construct the parameters of a useful agent-trace classifier directly, without optimisation or supervised training.

## 1. Objective and hypothesis

Build a small, inspectable PyTorch network that classifies recent AI coding-agent behaviour into four observed patterns:

1. **PRODUCTIVE** — evidence of forward progress or materially different investigation.
2. **RECOVERY** — bounded retry or recovery after a transient or diagnosed failure.
3. **STALLED** — repeated ineffective operations without observed progress.
4. **OSCILLATING** — recurring reversal between previously observed execution/workspace states.

**Primary hypothesis:** An LLM can directly construct sparse weights for a fixed small MLP which correctly identifies these patterns on held-out trace windows, without updating weights from examples by an optimizer.

The classifier evaluates *observed eight-operation windows*, not whether the agent will eventually fail. The runtime must support **INSUFFICIENT_EVIDENCE** as an external, non-neural result when a window is too short, lacks necessary instrumentation or is semantically ambiguous.

This is a research prototype, not a production safety or remediation controller. Do not automatically interrupt agent execution in v0.1.

## 2. Architectural boundary

```text
Synthetic session generator ────────┐
                                    v
Recorded agent spans -> canonical events -> 8-event window -> feature encoder
                                                           |
                                                           v
                                            PyTorch MLP (written weights)
                                                           |
                                                           v
                                          logits + attribution/debug view
                                                           |
                                                           v
                                      benchmark against sealed ground truth
```

Keep model construction, trace generation, feature encoding, ground-truth adjudication and evaluation in separate packages. The source event format should be compatible with future adapters from an agent trajectory IR or OpenTelemetry, but no collector, database, dashboard or live agent integration is required in v0.1.

### Target model (immutable within an experimental run)

- **Input:** `float32[batch, 104]` = 8 ordered operations × 13 features.
- **Network:** `Linear(104, 24) -> ReLU -> Linear(24, 12) -> ReLU -> Linear(12, 4)`.
- **Output:** Four raw logits in canonical class order `[PRODUCTIVE, RECOVERY, STALLED, OSCILLATING]`.
- **Parameters:** 2,872 trainable-shaped scalar slots (weights and biases). Count: `(104*24+24)+(24*12+12)+(12*4+4)`.
- **Initial state:** Every tensor set to zero, then explicitly populated by a generated weight-construction program. CPU/float32 throughout.
- **No softmax layer** in the model. Use argmax for prediction; softmax may be calculated only for display, never interpreted as calibrated confidence in v0.1.
- **No learned embeddings, BatchNorm, attention, recurrence or feature scaling learned from data.**

Architecture, ordered features, normalisation, class ordering and evidence-gating logic must be frozen before requesting the first set of weights.

## 3. Canonical event schema

For each completed agent operation store, at minimum:

- `session_id`, `agent_id`, `task_id`, `event_id`, `sequence_index`, `timestamp`.
- `operation_category`: `INSPECT`, `MUTATE`, `VALIDATE`, `EXTERNAL`, `CONTROL`.
- `outcome`: `SUCCESS`, `ERROR`, `NEUTRAL`.
- `intent_fingerprint`: deterministic canonical hash of operation family, relevant target and semantically relevant arguments. Not a raw text embedding or an LLM classification.
- `workspace_state_before`, `workspace_state_after`: digests from stable snapshots; nullable when not observed.
- `validation_passed`, `validation_total`, `validation_suite_id`: optional comparable verification data.
- `input_tokens`, `output_tokens`: tokens attributable to the agent step, if available.
- `failure_fingerprint`: optional normalized error identifier; used for fixtures, diagnostics and later extensions, but **not** a v0.1 network feature.
- `provenance`: simulator or telemetry adapter, plus instrumentation quality flags.

One event equals one *completed logical operation*, not every low-level span. Retry attempts which must be classified independently must be represented as separate operation attempts in the canonical stream. Preserve original trace/span IDs for future diagnostics.

Do not put task labels, simulator scenario IDs, hidden objective progress, or future outcomes on this public inference object.

## 4. Deterministic feature encoding: exactly 13 per operation

Flatten events oldest to newest. The feature indices shown are **zero-based within each 13-feature event block**.

| Index | Feature | Values and meaning |
|---:|---|---|
| 0–4 | `category_one_hot` | Exactly one of `[INSPECT, MUTATE, VALIDATE, EXTERNAL, CONTROL]` |
| 5–7 | `outcome_one_hot` | Exactly one of `[SUCCESS, ERROR, NEUTRAL]` |
| 8 | `repeated_intent` | 1 iff this event's intent fingerprint appeared in an earlier event within this eight-event window; else 0 |
| 9 | `workspace_changed` | 1 iff a known before/after state digest differs; else 0 |
| 10 | `state_revisited` | 1 iff a changed-to state digest matches a state previously encountered in this window (including its initial state); else 0 |
| 11 | `validation_delta` | For comparable validation results, clipped change in pass fraction since the preceding result from the same suite: `clip(new_pass_fraction - old_pass_fraction, -1, 1)`; otherwise 0 |
| 12 | `relative_token_cost` | `min(1, log1p(step_tokens)/log1p(100000))`; 0 for a verified zero-token step |

`step_tokens = input_tokens + output_tokens`, counted once for that operation. Do not derive it from future spans or the full-session total.

**Missing data:** Preserve unknown versus false in the canonical event. The fixed 13-feature encoding has no missingness masks; therefore reject windows with missing required state or token observations where those features would be material. In strictly synthetic v0.1, populate all required fields. For production-oriented expansion, add explicit masks and version the model; never silently map unknown observations to zero.

**Important semantic distinction:** `workspace_changed` is not evidence of useful progress. `validation_delta` is computed only from *observable, comparable* test results, not the simulator's hidden truth about whether the agent is closer to the goal. Even a green tool result need not imply progress.

## 5. Windowing and evidence policy

- Non-overlapping session/task/agent identity is mandatory; do not mix unrelated tasks or concurrent agents into one sequence.
- Create one window for each event after eight completed operations; sliding stride 1 for online-style classification.
- For benchmark splits, group by **whole simulated session**, never random windows: adjacent windows from the same session must not appear in different partitions.
- Only observed events up to and including the window endpoint may contribute to features.
- Insufficient evidence gate: return `INSUFFICIENT_EVIDENCE` before inference when fewer than eight events exist, required observations are missing, or neither repeated operations nor state changes nor comparable validation evidence offers support for a behaviour classification. Report evidence-gate coverage separately.
- The four classes describe a current pattern, not a guaranteed eventual outcome.

## 6. Ground-truth taxonomy and adjudication

Use a simulator containing *latent task state* for generating plausible histories, but calculate inference features from separately emitted observable events.

- **PRODUCTIVE:** objective/validation state improves over the window, or novel investigative operations materially narrow an unresolved problem; do not require every action to succeed.
- **RECOVERY:** a bounded transient failure or deliberate repair/retry sequence that either resolves or is still inside an explicitly modelled reasonable retry policy; no unbounded repetition.
- **STALLED:** substantially repeated operations/errors with no observed improvement, no material novel investigation and no state-reversion cycle.
- **OSCILLATING:** returns to prior workspace or execution state and repeats earlier actions without net progress; precedence over STALLED when both occur.

Define precedence `OSCILLATING > STALLED > RECOVERY > PRODUCTIVE` only for examples with adequate evidence; otherwise classify as `INSUFFICIENT_EVIDENCE` in the oracle and exclude from four-class accuracy, while counting rejection rate.

Avoid circular labelling such as `STALLED` iff `repeated_intent == 1`. Scenario labels must depend on the simulator's causal state and outcome, while observable features are generated independently. Run a **feature-collision audit**: if two identical 104-feature vectors receive different non-ambiguous labels, record the representational limitation and mark them ambiguous; never expect the network to resolve information absent from its inputs.

## 7. Synthetic generator

Build a stateful, seedable simulator, not a collection of fixed event strings. A session should model task state, repository state, tests, transient tool failures, retries, and token expenditure. It produces both `CanonicalEvent[]` and a separate, inaccessible-to-model `OracleRecord`.

Minimum scenario families:

1. Incremental productive repair: failing tests, targeted edits, fewer failures, eventually passes.
2. Productive exploration: several inspections and diagnostics before a meaningful change.
3. Transient external failure: repeated timeout then success, followed by productive action.
4. Bounded backoff/retry: recoverable failure with one or two controlled retries.
5. Repeated failed test invocation with unchanged artifact and unchanged failure fingerprint.
6. Repeated edits that change files but produce no improvement in validation.
7. A/B/A/B workspace oscillation with alternation between competing edits.
8. Tool oscillation: two operations repeatedly undo each other's effects.
9. Benign lookalike: repeated tests with genuine incremental improvements.
10. Benign lookalike: repeated reads or polls while an external dependency legitimately recovers.
11. Context expenditure increases while useful work does not (cost should be weak supporting evidence, not a label shortcut).
12. Ambiguous or under-instrumented sessions for coverage/abstention tests.

For each family, vary tool names, arguments, error fingerprints, cost distributions, success/error ratios, durations, task length and where the eight-event window lands. Ensure that identical superficial tool sequences can yield opposite correct labels when actual observable progress differs.

**Recommended dataset:** 20,000 generated sessions (not 20,000 nearly identical overlapping windows): 60% development, 20% validation, 20% sealed test, grouped by session and generator seed. Stratify class balance and scenario families. Keep a **separate holdout generator** with new scenario implementations and different distributions, not merely different random seeds. Produce counterfactual pairs that differ in one causal observation. Publish the seeds and a versioned dataset manifest.

Dataset generation should create the benchmark; it must not be called from the weight-construction process.

## 8. Direct-weight construction protocol

### Phase A — one-shot / zero-example

Provide the weight-authoring LLM with only:

- This specification (domain semantics, feature order, normalization and output classes).
- The immutable PyTorch architecture and tensor shapes.
- An empty parameter template and the admissible method of writing weights.
- A requirement to describe the intended purpose of each constructed neuron/circuit.

Ask for a deterministic `build_weights.py` that:

1. Starts from zeros.
2. Sets `Linear.weight` and `Linear.bias` entries by **literal assignment or closed-form analytic rules**.
3. Produces a complete `state_dict` and model artifact.
4. Explains each material weight block and emits sparsity statistics.

Allowed: arithmetic reasoning, explicit weight constants, structured matrix assignment, simple deterministic loops which populate known neuron circuits, verification of shapes and finite numbers.

Not allowed for Phase A: access to labelled examples, fitting, gradients, backpropagation, `torch.optim`, numerical search over candidate weights, evolutionary optimisation, distillation, hidden train/eval calls, or reading sealed fixtures.

Archive the **first valid weight artifact before running any behavioural test**. Compilation and shape checks do not count as behavioural feedback. If the model fails an example after this point, it is no longer a zero-feedback result.

### Phase B — diagnostic revision

Take the frozen Phase A artifact and permit **one LLM-authored direct edit** based on a fixed report from a small, disclosed diagnostic set. Log exactly which diagnostics and labels the LLM has seen. This is *feedback-assisted weight programming*, not zero-shot, and must be reported separately. Do not expose sealed test outputs until the final evaluation.

### Phase C — conventional training baseline

Train the **identical architecture and feature encoding** with supervised cross-entropy/Adam, using predefined dataset budgets (e.g. 512, 5,062 and 8,192 labelled windows), with three seeds per budget. This is an explicitly separate baseline; gradient use here is expected. Also evaluate a pre-registered deterministic rules engine and a constant-class prior baseline.

Do not modify the architecture after seeing test results. Any architecture change becomes a new registered experiment version.

## 9. Evaluation

Primary metrics:

- Macro F1 and per-class precision/recall on held-out sessions.
- **False-alarm rate on PRODUCTIVE and RECOVERY**, particularly false STALLED/OSCILLATING flags.
- Confusion matrix, with classes and sample counts.
- Sparsity: non-zero weights divided by 2,872 (using an explicit numeric tolerance, e.g. `abs(w) > 1e-9`).
- Prediction latency and active neuron/weight visualisation.
- Coverage and accuracy of the external insufficient-evidence gate.
- Counterfactual consistency and performance on wholly unseen scenario implementations.

Reports must show Phase A, Phase B, trained baselines and the rule baseline side-by-side, with **number of labelled examples observed by the weight-authoring LLM** and number of weight revisions.

Suggested research targets, not acceptance guarantees: Phase A macro-F1 >= 0.80 on the standard sealed set, false-loop alarms <= 5% of benign windows, and nontrivial generalisation on unseen scenario implementations. Do not infer that beating a weak trained baseline proves direct programming is generally superior to training.

**Fairness safeguards:** no duplicated sessions across splits; no oracle features; no scenario ID in features; inspect class priors; test with held-out scenario families; report 95% bootstrap confidence intervals over sessions; preserve all failed attempts and intermediate artifacts.

## 10. Verification and quality requirements

- Unit tests: feature order, one-hot validity, correct numeric ranges, deterministic normalisation and serialization, causal/non-future windowing, stable scenario generation.
- Architecture test: exact tensor shapes, exactly 2,872 slots, all-zero initial parameters, finite weights, forward output shape `[batch, 4]`.
- Anti-training test: direct-weight build path never invokes autograd, backward, optimiser steps or a dataset-dependent weight update.
- Data leakage tests: oracle metadata absent from inference data; no sessions shared across partitions; no duplicate windows crossing splits; feature-collision audit.
- Counterfactual examples: successful retry vs doomed retry, improving test failures vs identical repeating failures, productive edits vs A/B/A/B state reversal.
- Repeatability: same generator seed + same state_dict yield byte-identical feature tensors and identical CPU predictions.

## 11. Diagnostic interface

Provide a CLI-first inspector:

- Given a canonical eight-event trace, print the 8 × 13 feature matrix with meanings.
- Show the four raw logits and predicted class.
- Show the non-zero parameters of each layer and a layer-by-layer activation vector.
- For each input feature, optionally run a perturbation analysis, **without changing model weights**, to measure prediction sensitivity.
- Export confusion matrices, hard examples, counterfactual pairs and sparsity summaries.

Avoid a dashboard in v0.1. The interesting artefact is the network and its behaviour, not a production observability UI.

## 12. Suggested repository layout

```text
agent-loop-lab/
  SPEC.md
  pyproject.toml
  src/agent_loop_lab/
    schema.py            # CanonicalEvent + OracleRecord contracts
    encode.py            # frozen 13-feature encoder
    windows.py           # 8-event windowing and evidence policy
    model.py             # immutable MLP topology
    synth/               # stateful generators and independent oracle
    weight_program/      # generated build_weights.py and weights
    baselines/           # heuristics and supervised training
    eval/                # split, metrics, hardcases, OOD evaluation
    inspect/             # neuron, trace and matrix introspection
  tests/
  datasets/              # versioned manifests, not leaked into weight build
  experiments/
    phase-a-one-shot/
    phase-b-revision/
    phase-c-trained/
  artifacts/
```

Optional future integration: an adapter consumes Trajectory-normalized agent events or OpenTelemetry GenAI tool-execution spans. The core classifier remains ignorant of the telemetry provider. OpenTelemetry GenAI conventions cover tool executions and agent/workflow spans, but `workspace_changed`, state revisits and validation deltas will require application-specific enrichment.

## 13. Recommended milestones

**M0 — Contracts and harness.** Immutable schema, encoder, MLP with all-zero weights, tests, experiment manifests.

**M1 — Synthetic oracle.** Stateful session generator, twelve scenario families, sealed split and counterfactual tests. No LLM weight programming yet.

**M2 — Phase A.** Give the frozen spec to the weight-authoring model and archive its first valid weights *before any behavioural test*. Measure the one-shot result.

**M3 — Phase B.** Allow one documented direct weight revision using a disclosed diagnostic set. Compare with Phase A.

**M4 — Baselines and report.** Supervised same-topology baseline, deterministic baseline, OOD set, confidence intervals, feature-collision analysis and full reproducibility report.

**M5 — Optional real traces.** Manually adjudicate de-identified agent sessions and study domain shift. No autonomous agent interruption.

## 14. Definition of done

The experiment is complete when an independent person can reproduce the one-shot weight artifact, verify that no optimiser or labelled examples were used to construct it, inspect its active neural circuits, and compare it with both supervised and deterministic baselines on truly held-out trace families.

A poor Phase A result is still useful if the analysis shows *why* hand-programming fails: limited input representation, insufficient architecture capacity, interference between circuits, ambiguous observability or failure to extrapolate to unseen trace structures.
