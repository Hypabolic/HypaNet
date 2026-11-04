# Agent Doom Loop Detector — Direct Weight Programming Experiment

**Specification:** v0.2 (supersedes v0.1)
**Status:** design approved 2026-11-04. Encoder, architecture, fingerprint rules and oracle parameters are frozen. Implementation details are confirmed at M0 sign-off.
**Purpose:** Measure whether a reasoning LLM can (a) design a sound classification policy for agent-trace behaviour and (b) compile that policy directly into the parameters of a small fixed MLP, with no optimiser and no labelled examples.

---

## 1. Objective and hypotheses

Build a small, inspectable PyTorch network that classifies the most recent eight operations of an AI coding agent into one of four observed patterns:

1. **PRODUCTIVE**: evidence of forward progress or materially new investigation.
2. **RECOVERY**: a bounded retry or repair episode after a transient or diagnosed failure, still inside policy or recently resolved.
3. **STALLED**: repeated ineffective operations without observed progress and without a state-reversion cycle.
4. **OSCILLATING**: recurring reversal between previously held workspace or execution states without net progress.

The classifier describes the **current observed pattern**. It does not predict eventual task failure. `INSUFFICIENT_EVIDENCE` is a non-neural result returned by the runtime gate (§5.2) and by the oracle for ambiguous windows (§6.3).

### 1.1 What this experiment tests

The features in §4 are already partial detectors (`repeated_intent`, `state_revisited`, `validation_delta`). Any network built on them is effectively a compiled rule policy. So the experiment does **not** test whether neural networks beat rules. It tests three separable claims:

| ID | Claim | Measured as | Uses labels? |
|---|---|---|---|
| **H1 Design** | An LLM can write, without labelled examples, a classification policy over the 120 features that matches the latent-state oracle. | `reference_policy.py` vs oracle: macro-F1, false-alarm rate | Yes (evaluation only) |
| **H2 Compilation** | The same LLM can compile its own policy into the fixed 120→24→12→4 ReLU network. | Agreement of network argmax with `reference_policy.py` on benchmark windows and on a random valid-window sweep | **No** |
| **H3 End-to-end** | The directly written network is a useful classifier. | Network vs oracle: macro-F1, false-alarm rate, OOD generalisation | Yes (evaluation only) |

A poor H3 result can then be attributed to policy design (H1 low), compilation (H2 low), representation limits (high collision rate, low Bayes ceiling) or distribution shift (OOD gap).

### 1.2 Non-goals for v0.2

- No automatic interruption or remediation of agents.
- No live collector, database, dashboard or production integration.
- No claim that direct programming is generally better than training.

---

## 2. Architecture

### 2.1 Boundary

```text
Synthetic session generator (latent state) ──► OracleRecord (sealed)
        │
        ▼
CanonicalEvent[] ──► session prefix ──► 8-event window ──► runtime gate ──► encoder (120 floats)
                                                                               │
                                                     ┌─────────────────────────┤
                                                     ▼                         ▼
                                    MLP (directly written weights)   reference_policy.py
                                                     │                         │
                                                     ▼                         ▼
                                             4 logits + debug view       class index
                                                     │                         │
                                                     └──────► evaluation ◄─────┘
                                                                 ▲
                                                          OracleRecord labels
```

Model construction, generation, encoding, adjudication and evaluation live in separate packages (§13). The canonical event format should stay compatible with future adapters from an agent-trajectory IR or OpenTelemetry GenAI spans. No adapter is built in v0.2.

### 2.2 Target model (immutable within an experimental run)

- **Input:** `float32[batch, 120]` = 8 ordered events × 15 features, oldest first.
- **Network:** `Linear(120, 24) → ReLU → Linear(24, 12) → ReLU → Linear(12, 4)`.
- **Output:** four raw logits in canonical order `[PRODUCTIVE, RECOVERY, STALLED, OSCILLATING]`.
- **Parameter slots:** `(120·24 + 24) + (24·12 + 12) + (12·4 + 4) = 2,904 + 300 + 52 = 3,256`.
- **Initial state:** every tensor zero, then populated only by the weight-construction program. CPU, float32, `torch.no_grad()`.
- **Prediction:** `torch.argmax(logits, dim=-1)`. Ties resolve to the first maximal index, so an all-zero output predicts PRODUCTIVE. That makes the benign class the default.
- **No softmax** in the model. Softmax may be computed for display only and must not be read as calibrated confidence.
- **Excluded:** learned embeddings, normalisation layers, attention, recurrence, data-fitted feature scaling.

Architecture, feature order, encoding rules, fingerprint rules, class order, gate logic and oracle parameters are frozen at M0 sign-off, before any weights are requested. Changing any of them creates a new registered experiment version.

---

## 3. Canonical event schema

### 3.1 Fields

For each completed logical operation:

| Field | Type | Notes |
|---|---|---|
| `session_id`, `agent_id`, `task_id` | str | A sequence never mixes tasks or concurrent agents. |
| `event_id`, `sequence_index`, `timestamp` | str, int, ISO-8601 | `sequence_index` is dense and 0-based within the session. |
| `operation_family` | str | For example `read_file`, `run_tests`. Not a model feature. Used for fingerprinting. |
| `operation_category` | enum | `INSPECT`, `MUTATE`, `VALIDATE`, `EXTERNAL`, `CONTROL` |
| `outcome` | enum | `SUCCESS`, `ERROR`, `NEUTRAL` |
| `intent_fingerprint` | hex str | §3.2 |
| `failure_fingerprint` | hex str \| null | Required when `outcome = ERROR`. A normalised error identifier (§3.2). |
| `workspace_state_before`, `workspace_state_after` | hex str \| null | Digests of tracked workspace content. Null means not observed, which is distinct from unchanged. |
| `validation_suite_id`, `validation_passed`, `validation_total` | str, int, int \| null | Present only for events that produced a comparable validation result. |
| `input_tokens`, `output_tokens` | int \| null | Tokens attributable to this operation only. |
| `provenance` | object | Source (`simulator:<version>` or adapter), instrumentation-quality flags, original trace/span IDs. |

One event is one completed logical operation, not every low-level span. Retry attempts are separate events. The public inference object must **not** carry task labels, scenario or family IDs, latent progress, retry-policy state or future outcomes.

### 3.2 Fingerprint canonicalisation (frozen with the encoder)

`intent_fingerprint = sha256(canonical_json({family, target, args}))`. Canonical JSON means sorted keys, UTF-8, no insignificant whitespace, and normalised repo-relative POSIX paths.

| Family | Category | `target` | `args` included | Excluded |
|---|---|---|---|---|
| `read_file` | INSPECT | path | line range | — |
| `search` | INSPECT | scope path | query, lower-cased, whitespace-collapsed | result limit |
| `list_dir` | INSPECT | path | — | — |
| `edit_file` | MUTATE | path | sha256 of normalised patch | — |
| `write_file` | MUTATE | path | sha256 of content | — |
| `revert` | MUTATE | sorted path set | target revision digest | — |
| `run_tests` | VALIDATE | suite id | sorted selected test ids | verbosity, parallelism, timeout |
| `build` / `lint` | VALIDATE | target | — | flags not affecting result |
| `http_call` | EXTERNAL | method + host + path template | sha256 of normalised body for non-idempotent methods | request id, timestamp, retry counter |
| `poll_status` | EXTERNAL | resource id | — | attempt number, backoff |
| `plan` / `ask` / `yield` | CONTROL | — | action type | free text |

Across all families, timestamps, attempt counters, timeouts, request IDs and free-text reasoning are always excluded. As a result, the same edit repeated gives the same intent, and two different edits to the same file give different intents.

`failure_fingerprint = sha256(canonical_json({family, error_class, normalised_location}))`. Messages are stripped of numbers, paths outside the repo, addresses and timestamps.

`workspace_state_*` = sha256 over the sorted list of `(path, sha256(content))` for tracked files. INSPECT events must report `before == after` when nothing changed. They must not report null.

---

## 4. Deterministic feature encoding: exactly 15 per event

The encoder takes the **session prefix** up to and including the window endpoint. Window events are flattened oldest to newest. Indices are zero-based within each 15-feature block. Every feature is computed from observed events at or before the endpoint.

| Idx | Feature | Definition | Scope |
|---:|---|---|---|
| 0–4 | `category_one_hot` | One of `[INSPECT, MUTATE, VALIDATE, EXTERNAL, CONTROL]` | event |
| 5–7 | `outcome_one_hot` | One of `[SUCCESS, ERROR, NEUTRAL]` | event |
| 8 | `repeated_intent` | 1 iff `intent_fingerprint` equals that of an earlier event **in this window** | window |
| 9 | `repeated_failure` | 1 iff `outcome = ERROR` and `failure_fingerprint` equals that of an earlier ERROR event **in this window** | window |
| 10 | `workspace_changed` | 1 iff `workspace_state_before ≠ workspace_state_after` | event |
| 11 | `state_revisited` | 1 iff `workspace_changed = 1` and `workspace_state_after` equals any before- or after-digest of an earlier event **in this window** (including the first event's before-state) | window |
| 12 | `validation_comparable` | 1 iff this event produced a validation result **and** an earlier result from the same `validation_suite_id` exists **anywhere earlier in the session** | session prefix |
| 13 | `validation_delta` | If `validation_comparable = 1`: `clip(passed/total − prev_passed/prev_total, −1, 1)` against the most recent earlier result from that suite. Otherwise 0. | session prefix |
| 14 | `relative_token_cost` | `min(1, log1p(input_tokens + output_tokens) / log1p(100000))`. Exactly 0 for a verified zero-token step. | event |

Invariants (unit-tested):

- One-hot groups sum to exactly 1.
- `repeated_failure ≤ outcome_ERROR`, `state_revisited ≤ workspace_changed`, `validation_comparable ≤ category_VALIDATE`, and `validation_delta ≠ 0 ⇒ validation_comparable = 1`.
- All values lie in [0, 1] except `validation_delta` ∈ [−1, 1].
- Index 0 of each window always has `repeated_intent = repeated_failure = 0`. This is a known positional asymmetry.

`workspace_changed` is not evidence of progress. `validation_delta` uses only observable, comparable results. A green tool result does not by itself imply progress.

**Missing data:** the encoder has no missingness masks. Missing values are never silently mapped to 0. Windows with missing required observations are rejected by the gate (§5.2). Masks are a future versioned extension for real telemetry.

---

## 5. Windowing and runtime gate

### 5.1 Windows

- A sequence is one `(session_id, agent_id, task_id)` stream ordered by `sequence_index`.
- For each endpoint `t ≥ 7` (0-based), the window is events `t−7 … t`. Stride is 1.
- Benchmark splits are grouped by **whole session**. Windows from one session never appear in two partitions.

### 5.2 Runtime evidence gate (non-neural, applied before inference)

The gate returns `INSUFFICIENT_EVIDENCE` if and only if:

1. fewer than 8 events exist, **or**
2. any window event is missing a required observation:
   - category, outcome or either token count;
   - either workspace digest;
   - `failure_fingerprint` when `outcome = ERROR`;
   - suite id, passed or total for an event flagged as having produced a validation result;
   - **or** the validation baseline needed for features 12–13 is flagged as unobserved, for example truncated history.

The gate makes **no semantic judgement**. Windows with no repeats, no state change and no tests are valid, and they are usually PRODUCTIVE exploration. Semantic ambiguity is decided only by the oracle (§6.3). Gate coverage and gate agreement with oracle instrumentation flags are reported separately (§10).

---

## 6. Ground truth: latent state and oracle

### 6.1 Latent state (simulator-internal, never encoded)

Per session the simulator maintains:

- `progress p_t ∈ [0, 1]`: hidden objective completion.
- `knowledge K_t`: a set of latent facts about the fault. An operation may add facts. A new fact is "material narrowing".
- `workspace W_t`: real simulated file contents. Digests are computed from these.
- `episodes`: failure episodes with latent cause (`transient`, `persistent`, `diagnosed_repair`), a retry counter and a resolution flag.
- Per-operation latent effect: one of `PROGRESS`, `NEW_FACT`, `RESOLVES_EPISODE`, `IN_POLICY_RETRY`, `IN_POLICY_POLL`, `REVERSION`, `REDUNDANT`, `NONE`.

`REDUNDANT` means the operation repeats work already reflected in latent state. It re-acquires a fact already in `K_t`, re-applies an edit already present in `W_t`, re-runs validation on an unchanged workspace, or is an out-of-policy retry or poll. Redundancy is defined against `K_t` and `W_t`, **not** against `intent_fingerprint`, so it stays independent of the encoded features. `NONE` means the operation is novel but yielded nothing, for example a first read of a file that turns out to be irrelevant.

An episode's **initial failing operation** has effect `NONE`, because the attempt itself was reasonable. Its in-policy retries are `IN_POLICY_RETRY`. Retries beyond `R_max` are `REDUNDANT`.

An operation is **stall-ineffective** iff its latent effect is `REDUNDANT` or `REVERSION`. An operation is **ineffective** iff it is stall-ineffective or has effect `NONE`.

A **reversion** is an agent mutation whose latent after-state equals a workspace state held earlier in the window **and** that undoes an earlier agent mutation. External resets don't count.

### 6.2 Pre-registered oracle parameters

| Symbol | Value | Meaning |
|---|---:|---|
| `ε_p` | 0.02 | Progress tolerance over a window |
| `F_prod` | 2 | Minimum new facts in the window for investigative PRODUCTIVE |
| `N_stall` | 5 | Minimum ineffective operations of 8 for STALLED |
| `N_grey` | 3 | Lower bound of the stall grey zone |
| `N_rev` | 2 | Minimum reversions in the window for OSCILLATING |
| `R_max` | 3 | Maximum retries of one intent within an episode while still in policy |
| `P_max` | 5 | Maximum polls of an external dependency (with backoff) while still in policy |
| `k_rec` | 4 | Recency horizon for RECOVERY |

These values live in `oracle_config.v0_2.yaml` and are frozen at M0.

### 6.3 Window labelling rules (evaluated in order; first match wins)

Let `Δp` be the progress change across the window, `F` the number of new facts, `I` the number of **stall-ineffective** operations (`REDUNDANT` + `REVERSION`) and `V` the number of reversions.

1. **INSUFFICIENT_EVIDENCE (instrumentation):** the generator deliberately dropped a required observation (scenario 12).
2. **OSCILLATING:** `V ≥ N_rev` and `Δp ≤ ε_p`.
3. **STALLED:** `Δp ≤ ε_p`, `F = 0`, `I ≥ N_stall`, the final event is stall-ineffective, **and** no episode is in policy at the endpoint.
4. **RECOVERY:** an episode is active at the endpoint with retries ≤ `R_max` (or polls ≤ `P_max`), **or** an episode resolved within the last `k_rec` events.
5. **PRODUCTIVE:** `Δp > ε_p` or `F ≥ F_prod`.
6. **INSUFFICIENT_EVIDENCE (ambiguous):** everything else. This includes the grey zone `N_grey ≤ I < N_stall` with no progress, windows with a single new fact and no progress, and aimless novel activity (mostly `NONE` effects). Novel but fruitless exploration is never labelled STALLED. Repetition of already-done work is what defines a stall.

This ordering gives `OSCILLATING > STALLED > RECOVERY > PRODUCTIVE`. The recency bound in rule 4 stops a long-resolved retry from overriding later productive work.

Labels depend on latent causal state. Observable features are emitted independently by the event layer. Labels are never defined in terms of encoded features. For example, there must be no rule of the form "STALLED iff `repeated_intent`".

### 6.4 Feature-collision audit

Group windows by exact float32 byte equality of the 120-vector. If one group contains different non-ambiguous labels, record it as a representational collision. Report:

- the collision rate per class and per family;
- the **Bayes ceiling**: accuracy and macro-F1 of predicting each group's majority label.

Collisions stay in the primary metric. The Bayes ceiling tells readers how much of any shortfall the inputs cannot resolve.

---

## 7. Synthetic generator

### 7.1 Scenario families (names only; this list is in the authoring packet)

1. Incremental productive repair: failing tests, targeted edits, fewer failures, eventually green.
2. Productive exploration: several inspections and diagnostics before a meaningful change.
3. Transient external failure: repeated timeout, then success, then productive work.
4. Bounded backoff or retry: a recoverable failure with one or two controlled retries.
5. Repeated failed test invocation with unchanged artefact and unchanged failure.
6. Repeated edits that change files but don't improve validation.
7. A/B/A/B workspace oscillation between competing edits.
8. Tool oscillation: two operations repeatedly undo each other's effects.
9. Benign lookalike: repeated test runs with genuine incremental improvement.
10. Benign lookalike: repeated reads or polls while an external dependency legitimately recovers.
11. Rising context or token expenditure without useful work. Cost is weak supporting evidence, not a label shortcut.
12. Under-instrumented sessions, for gate coverage.

---

## 8. Roles and independence

| Role | Does | Must not have seen |
|---|---|---|
| **Harness implementer** | M0/M1 code: schema, encoder, gate, model shell, standard generator, oracle, eval harness | — |
| **Weight author** | Phase A/B: `reference_policy.py`, `build_weights.py` | Generator or oracle code, any dataset, any eval output (until Phase B's disclosed set) |
| **OOD generator author** | OOD holdout generator from a written brief | Weight artefacts, standard generator code |
| **Rules-baseline author** | Independent pre-registered rules engine | Datasets and labels |
| **Adjudicator** (human) | Approves the frozen spec, verifies logs, runs final evaluation | — |

- The weight author runs in a **fresh context** (new session, no repository access beyond the authoring packet directory). A context that implemented or read the generator cannot act as the Phase A weight author.
- A different model family is recommended for the OOD generator author.
- **Authoring packet** (built by script, sha256 recorded): this spec minus §0, §2.3 and §7.2–§7.3; `model.py`; the zero-filled `state_dict` template with tensor names and shapes; the encoder's feature-index map; and the rules in §9.

---

## 9. Direct-weight construction protocol

### 9.1 Required deliverables from the weight author

1. `reference_policy.py`: a pure, deterministic function `reference_policy(x: tuple[float, ...]) -> int ∈ {0,1,2,3}`. `x` holds the 120 input values as Python floats, each exactly representable in float32. The function does no I/O, imports nothing beyond the standard library and accesses no data. It states the author's intended policy.
2. `build_weights.py`: defines `build()`, which starts from `zero_model()`, sets every `Linear.weight` and `Linear.bias` entry by literal assignment or closed-form rule, and returns the model. It runs no forward computation, matrix products or linear-algebra solvers, and writes no files. The harness runs `build()` under the anti-training guard and writes the `.dlw` artefact, its sha256 and a build record with sparsity statistics.
3. `CIRCUITS.md`: the purpose of every non-zero hidden unit, grouped by circuit, plus which reference-policy branch each output logit implements.

Allowed: arithmetic reasoning, explicit constants, structured matrix assignment, and deterministic loops that fill known circuits. The authoring packet's `AUTHOR_INSTRUCTIONS.md` gives the exact interfaces and the full list of checks the harness applies.

Forbidden in Phases A and B: labelled examples (except Phase B's disclosed set), fitting, autograd, `.backward()`, `torch.optim`, numerical or evolutionary search over weights, distillation, and importing `synth/`, `datasets/` or `eval/`.

### 9.2 Phase A-strict (zero execution)

The author may run only shape, dtype and finiteness checks plus `reference_policy.py` syntax checks. No forward passes. The harness runs `build()` and calls `reference_policy` once on the all-zero input as a type check, without recording the result. That is the only execution. The harness archives the first valid artefact and the adjudicator git-commits it with its sha256 **before anything else runs**.

### 9.3 Phase A-selfcheck (zero labels)

This starts from the archived A-strict artefact, in the same authoring context. The author may run forward passes and `reference_policy.py` **on inputs they construct themselves** (hand-written windows, or enumerations of their own policy's branch conditions). They may compare the two and revise weights by direct, reasoned edits. Not allowed: generator output, datasets, labels, or automated weight search. Log every iteration with a diff and the reason. Archive the final artefact.

Both A-strict and A-selfcheck are reported. Both count as **0 labelled examples**.

### 9.4 Phase B (one diagnostic revision)

The base is the A-selfcheck artefact. The author receives a fixed disclosed diagnostic report: 200 windows from the **validation** split (fixed seed, stratified by class), each with features, oracle label, network prediction and reference-policy prediction. They may make **one** revision to the weights and to `reference_policy.py`. Log exactly what was disclosed. This is feedback-assisted programming and is reported as 200 labelled examples with one revision.

### 9.5 Phase C and other baselines

- **Supervised, same topology and encoder:** cross-entropy with Adam (lr 1e-3, batch 256, up to 50 epochs, early stopping on validation macro-F1). Budgets are **32, 128, 512, 2,048 and all development sessions** (window counts reported). Three seeds per budget. Hyperparameters are fixed before training. An L1-regularised variant is optional, for a sparsity comparison.
- **Independent rules engine:** written by the rules-baseline author against this spec, before M2.
- **Author reference policy:** `reference_policy.py` is itself evaluated as a baseline (H1).
- **Constant prior:** always predicts the development-set majority class, plus always-PRODUCTIVE.

---

## 10. Evaluation

### 10.1 Metrics (standard test set, OOD set and each family)

- Macro-F1, and per-class precision and recall, on oracle-labelled (non-INSUFFICIENT) windows.
- **False-loop alarm rate:** the fraction of PRODUCTIVE + RECOVERY windows predicted STALLED or OSCILLATING.
- Confusion matrices with counts.
- **Compilation fidelity (H2):** argmax agreement between the network and `reference_policy.py` on (a) all benchmark windows and (b) 100,000 windows from a random valid-window sampler that respects §4's invariants but ignores the generator. No labels are needed.
- Sparsity: `count(|w| > 1e-9) / 3,256`, per layer and overall.
- Collision rate and Bayes ceiling (§6.4).
- Gate coverage, and gate agreement with oracle instrumentation flags. Report the class distribution of network predictions on oracle-ambiguous windows, but don't score it.
- Counterfactual consistency: the fraction of counterfactual pairs whose predicted classes differ in the direction of the oracle labels.
- CPU latency per batch of 1 and of 1,024.
- 95% bootstrap confidence intervals over **sessions** (10,000 resamples).

### 10.2 Reporting

One table with columns for A-strict, A-selfcheck, B, C (per budget, mean ± sd over seeds), independent rules, author reference policy and constant prior. Rows cover each metric above, **labelled examples seen by the weight author**, and the number of weight revisions or self-check iterations.

Suggested research targets (not acceptance gates): A-selfcheck macro-F1 ≥ 0.80 on the standard sealed set, false-loop alarms ≤ 5%, compilation fidelity ≥ 0.99, and a non-trivial OOD score. Beating a weak trained baseline does not show that direct programming is generally better.

Sealed-test and OOD outputs are computed once, after all phases are archived. Every failed attempt and intermediate artefact is kept.

---

## 11. Verification requirements

- **Encoder:** feature order, one-hot validity, invariants in §4, numeric ranges, byte-identical output for identical session prefixes, and no future leakage (appending events after the endpoint never changes that window's features).
- **Fingerprints:** golden tests per family in §3.2, including excluded-argument insensitivity.
- **Architecture:** exact shapes, exactly 3,256 slots, all-zero initial state, finite weights, output `[batch, 4]`, and the argmax tie rule.
- **Anti-training:** the weight-build path runs under a guard that fails if autograd records a graph, `.backward()` is called, any `torch.optim` object is created, or any module under `synth/`, `datasets/` or `eval/` is imported. Use an import hook plus `torch.is_grad_enabled()` assertions.
- **Leakage:** no oracle fields on `CanonicalEvent`, no session shared across splits, no identical window crossing splits (reported, not just checked), and the authoring packet contains no generator or dataset files (checked against the packet manifest).
- **Gate:** scenario-12 sessions are rejected, and all-novel INSPECT windows are **not** rejected.
- **Oracle:** unit tests per rule in §6.3, including the RECOVERY recency bound, grey-zone handling, and the requirement that eight novel, fruitless operations are never labelled STALLED.
- **Repeatability:** the same seed and the same `state_dict` give byte-identical feature tensors and identical CPU predictions.

---

## 12. Diagnostic interface (CLI)

- `inspect window <file>`: prints the 8 × 15 feature matrix with names, gate result, raw logits, predicted class and reference-policy class.
- `inspect model <artefact>`: lists non-zero parameters per layer with the `CIRCUITS.md` label for each hidden unit.
- `inspect activations <file>`: shows the layer-by-layer activation vectors, with active units named.
- `inspect perturb <file>`: flips or nudges each input feature and reports changes in prediction and logits. Weights are never modified.
- `export`: writes confusion matrices, hard examples, fidelity disagreements, counterfactual pairs, collision groups and sparsity summaries.

No dashboard in v0.2.

---

## 13. Repository layout

```text
agent_doom_loop_detector_spec_v0_2.md  # this document (frozen)
pyproject.toml
oracle_config.v0_2.yaml
src/agent_loop_lab/
  schema.py                   # CanonicalEvent, OracleRecord
  fingerprint.py              # §3.2 canonicalisation
  encode.py                   # frozen 15-feature encoder
  windows.py                  # windowing + runtime gate
  model.py                    # immutable MLP
  synth/                      # standard generator + oracle   (never in authoring packet)
  synth_ood/                  # OOD generator, separate author (never in authoring packet)
  baselines/                  # rules engine, constant prior, supervised training
  eval/                       # splits, metrics, fidelity, collisions, bootstrap
  inspect/                    # CLI
  guards.py                   # anti-training import/autograd guard
authoring_packet/             # generated, hashed; the only input to the weight author
weight_program/
  phase-a-strict/             # reference_policy.py, build_weights.py, CIRCUITS.md, artefact, sha256
  phase-a-selfcheck/          # + iteration log
  phase-b-revision/           # + disclosed diagnostic report
tests/
datasets/                     # manifests + split hashes; data files git-ignored
experiments/phase-c-trained/
artifacts/reports/
```

---

## 14. Milestones

- **M0, contracts and harness:** schema, fingerprints, encoder, gate, zero-weight model, guards, tests, `oracle_config` and the authoring-packet builder. **Freeze sign-off by the adjudicator.**
- **M1, synthetic oracle:** standard generator, oracle, 12 families, counterfactual pairs, splits, manifests, collision audit on the development split. Commission the OOD generator and the independent rules engine in parallel, from separate authors.
- **M2, Phase A:** a fresh weight-author context gets the hashed packet. Archive A-strict, then A-selfcheck.
- **M3, Phase B:** disclose the diagnostic report and accept one revision.
- **M4, baselines and report:** Phase C, rules, priors, sealed and OOD evaluation, bootstrap CIs, fidelity and collision analysis, reproducibility report.
- **M5 (optional), real traces:** manually adjudicated, de-identified agent sessions through an adapter. Study domain shift. No autonomous interruption.

## 15. Definition of done

An independent person can:

1. rebuild the authoring packet and confirm its hash and contents;
2. re-run `build_weights.py` and get a byte-identical artefact for each phase;
3. verify from logs and guards that no optimiser, search or unauthorised labelled data touched Phases A or B;
4. read `CIRCUITS.md` alongside `inspect model` and trace each active circuit;
5. reproduce the comparison table, including H1, H2 and H3 separately, against supervised and rules baselines on the sealed and OOD sets.

A poor result still counts as success if the analysis attributes it to policy design (H1), compilation (H2), representation limits (Bayes ceiling), network capacity or distribution shift (OOD gap).
