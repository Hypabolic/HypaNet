# M4: v0.2 baselines and sealed evaluation, pre-registration (spec v0.2 §9.5, §10)

Written **2026-11-05, before any Phase C training run and before any sealed test or OOD window has been scored.** Authority for sealed scoring now: spec v0.3 E7 ("v0.2 sealed test/OOD scoring may run once V0c is committed"); V0c is commit `e05f371`. Nothing below may change after Phase C training starts. A change needs a recorded spec revision, and the report states it.

## 1. Phase C: supervised MLP, same topology and encoder (§9.5)

All hyperparameters are fixed here, before training.

- **Model.** `agent_loop_lab.model.DoomLoopMLP` (120→24→12→4, ReLU), float32, CPU. **Initialisation:** standard PyTorch default `nn.Linear` init: `torch.manual_seed(seed)`, then `DoomLoopMLP()`. It is **not** `zero_model()`, because zero init would make every hidden unit identical.
- **Training population.** Development windows that pass the runtime gate **and** carry a scored oracle label (PRODUCTIVE, RECOVERY, STALLED, OSCILLATING), from the sessions in the budget subset. The target is the label's class index. The inputs are the frozen encoder's float32 120-vectors (`collisions.session_windows`).
- **Budgets.** 32, 128, 512, 2,048 and all (12,000) dev sessions.
- **Budget subsets.**
  - **Order.** The dev session ids from `datasets/v0_2/splits.json` (`splits.dev`) are sorted ascending by `sha256(f"20261105:m4-budget:{session_id}").hexdigest()`, compared as hex strings.
  - **Selection.** The subset for budget *k* is the first *k* sessions in that order. Subsets are nested, and every seed uses the same subsets.
  - **Session counting.** A session counts towards the budget even if it gives no training windows (for example `under_instrumented` sessions whose windows are all gate-rejected).
  - **Reporting.** Each run reports its window count and class counts.
- **Seeds.** 0, 1, 2 for each budget, so 15 runs.
- **Optimisation.**
  - **Loss and optimiser.** `torch.nn.functional.cross_entropy` (mean, unweighted). `torch.optim.Adam(lr=1e-3)` with PyTorch defaults: betas (0.9, 0.999), eps 1e-8, no weight decay.
  - **Batching.** Batch size 256. Each epoch shuffles with `torch.randperm` from `torch.Generator().manual_seed(seed)`, a generator created once per run. The last partial batch is kept.
  - **Runtime.** `torch.set_num_threads(2)` and `torch.use_deterministic_algorithms(True)`.
- **Early stopping.**
  - **When validation is scored.** After every epoch, macro-F1 is computed on all gate-passing, scored **validation** windows (every val session, including the Phase B disclosed sessions), with the `eval.metrics.metrics_from_counts` definition.
  - **Best state.** A new best needs a strictly higher macro-F1, so the earliest maximum is kept.
  - **Stopping rule.** Training stops after 5 consecutive epochs without a new best, or after 50 epochs. The best state is kept.
- **Outputs.** Each run writes `experiments/phase-c-trained/v0_2/<budget>/seed<seed>/` with these files:
  - `model.dlw`: deterministic `.dlw` state_dict with metadata;
  - `model.sha256`;
  - `training_log.json`: per-epoch loss and val macro-F1, epochs run, best epoch, best val macro-F1, training window and session counts, and the code's sha256.
- **Isolation.** The trainer is `src/agent_loop_lab/baselines/phase_c.py`, run by `scripts/m4_train_phase_c.py`. It is never imported by any weight-build path (runner, packet, guards, `weight_program/`).
- **L1-regularised variant.** Optional in §9.5. **Skipped.**

## 2. Other baselines

- **Independent rules engine.** `experiments/commissions/rules_baseline/rules.py`, function `rules_classify(x)`, with its sha256 checked against `DELIVERABLES.sha256` before use. It is applied to the 120-vector (as Python floats) of every gate-passing window, which is the input its `RULES.md` specifies. Gate-rejected windows get no prediction, as for every column.
- **Constant priors.**
  - **dev-majority:** the majority class over gate-passing, scored dev windows (all 12,000 sessions).
  - **always-PRODUCTIVE.**
  - **Expected overlap:** dev-majority is expected to be PRODUCTIVE, which would make the two columns identical. Both are still reported, and the report records the dev class counts.
- **Weight-program columns.**
  - **Phases.** A-strict, A-selfcheck and B (`weight_program/<phase>/`, `artifact.dlw` sha-checked against `artifact.sha256`).
  - **Columns per phase.** Each phase gives two columns: the **network** and the author's **reference policy** (H1).
  - **Static checks.** Reference policies pass `runner.check_reference_policy_source` first, as in M3.

## 3. Sealed evaluation (§10.1)

There is one entry point, `scripts/m4_sealed_eval_v02.py`. It computes everything **once**.

### 3.1 Splits

| Split | Role |
|---|---|
| test | Sealed standard test set |
| OOD | Sealed independently generated holdout (`datasets/v0_2/ood/`) |
| val | Reference only. Phase C used val for early stopping, so its val numbers are optimistic. Phase B's 200 disclosed windows come from val |
| counterfactual | Counterfactual consistency only. This set is not sealed |

### 3.2 Columns, per split

- A-strict network and reference policy;
- A-selfcheck network and reference policy;
- B network and reference policy;
- Phase C for each budget × seed, plus the per-budget **mean ± sd** (sample sd, ddof 1) over the 3 seeds;
- rules;
- prior dev-majority;
- prior always-PRODUCTIVE.

### 3.3 Metrics

- **Base definitions.** Populations and definitions are exactly those of `eval/metrics.py`, as used in M3.
- **Per column, overall and per family:**
  - macro-F1;
  - accuracy;
  - per-class precision, recall and F1;
  - false-loop alarm rate;
  - confusion matrix (counts);
  - prediction distribution on oracle-ambiguous and on gate-passing instrumentation windows (reported, not scored).
- **Confidence intervals.**
  - **Resampling.** 95% session-bootstrap CIs: 10,000 resamples, seed 20261104, chunk 250, and percentiles 2.5/97.5 (numpy linear).
  - **Shared draws.** Within a split, every column uses the same resampled sessions. The draws depend only on the number of sessions and the seed.
  - **Per family.** Per-family CIs use `metrics.family_seed`.
  - **Phase C.** For each budget, the CI of the 3-seed mean is computed on the shared draws.
- **H2 compilation fidelity, network vs own reference policy, for each weight-program phase:**
  - **(a) Benchmark windows.** All gate-passing windows of each split, and the scored subset, with CIs.
  - **(b) Random valid-window sampler.** 100,000 windows (§3.4), with agreement overall and broken down by reference-policy class.
- **Counterfactual consistency, every column (§3.5).**
- **Gate.** Coverage is the fraction of windows passing the gate. Agreement with the oracle's instrumentation flag is reported as a 2×2 table: gate-rejected vs label INSUFFICIENT_INSTRUMENTATION.
- **Collision rate and Bayes ceiling.** `eval.collisions.collision_audit` on test and on OOD. These are label-only and use no model.
- **Sparsity.** `model.sparsity` for every network column.
- **CPU latency.**
  - **Setup.** Every network column, with `torch.set_num_threads(2)`, under `torch.no_grad()`, timing forward plus argmax.
  - **Batch 1.** 50 warm-up calls, then the median of 1,000 calls.
  - **Batch 1,024.** The median of 200 calls.
  - **Inputs.** The inputs are sampler windows, so no sealed data is involved.
  - **Python policies.** The rules and reference policies are timed as the mean per-window time over 1,024 sampler windows.
- **Labelled examples seen.**
  - A-strict 0, A-selfcheck 0, rules 0;
  - B 200, with one revision;
  - Phase C: the number of training windows, as reported;
  - Priors: dev label counts.
  - Revisions or self-check iterations come from the archived records.

### 3.4 Random valid-window sampler (H2 part b)

- **Source.** `eval/sampler.py`. It ignores the generator.
- **Seed.** `numpy.random.default_rng(20261105)`, with 100,000 windows drawn in one call sequence.
- **Draw order.** Positions are drawn oldest to newest. Each position *i* (0–7) is drawn independently, except where the encoder-reachability constraints below apply.

| Feature | Rule |
|---|---|
| category | Uniform over the 5 categories (one-hot) |
| outcome | Uniform over the 3 outcomes (one-hot) |
| `repeated_intent` | Bernoulli(0.5) if *i* > 0 and some earlier position has the same category. Otherwise 0. The intent fingerprint hashes the operation family, which fixes the category |
| `repeated_failure` | Bernoulli(0.5) if the outcome is ERROR, *i* > 0, and some earlier position is an ERROR of the same category. Otherwise 0. The failure fingerprint hashes the family |
| `workspace_changed` | Bernoulli(0.5) |
| `state_revisited` | Bernoulli(0.5) if `workspace_changed` = 1 and *i* > 0. Otherwise 0. The encoder can only match the first event's before-state from a later event |
| `validation_comparable` | Bernoulli(0.5) if the category is VALIDATE. Otherwise 0 |
| `validation_delta` | Only when comparable: 0 with probability 0.25, otherwise float32(Uniform[−1, 1]). Otherwise 0 |
| `relative_token_cost` | 0 with probability 0.1, 1 with probability 0.05, otherwise float32(Uniform[0, 1]) |

- **Checks.** Every §4 invariant is asserted on every sampled window:
  - one-hot sums;
  - `repeated_failure ≤ ERROR`, `state_revisited ≤ workspace_changed`, `validation_comparable ≤ VALIDATE`, and `delta ≠ 0 ⇒ comparable`;
  - repeat flags at index 0 are 0;
  - all features in [0, 1] except delta, which is in [−1, 1].
- **Reference policy inputs.** Reference policies receive each window as a tuple of Python floats, each exactly representable in float32.

### 3.5 Counterfactual consistency

- **Pair structure.**
  - **Pairing.** Pairs are identified by `OracleRecord.latent_trace["counterfactual"]`: `pair_id` groups the two sessions, and `branch` is `a` (benign) or `b` (degenerate).
  - **Contrast endpoints.** The unit is a **contrast endpoint** from that record's `contrast_endpoints`. Both branches have such a window, `a`'s label is in the benign set and `b`'s label is the degenerate label (`synth.counterfactual.INTENDED`).
  - **Excluded pairs.** Pairs with no contrast endpoint are excluded and counted (123 of 900 in the manifest).
  - **Gate rejection.** Endpoints where either window is gate-rejected are excluded and counted.
- **Primary measure: directional.** An endpoint is consistent iff pred(a) ∈ {PRODUCTIVE, RECOVERY} and pred(b) ∈ {STALLED, OSCILLATING}.
- **Strict measure.** pred(a) = oracle label of `a` and pred(b) = oracle label of `b`.
- **Reported for each measure, overall and per kind:**
  - the pooled endpoint rate;
  - the fraction of pairs consistent at **every** contrast endpoint (pair-level, "fraction of counterfactual pairs" in §10.1);
  - the fraction of pairs consistent at **any** contrast endpoint.
- **CIs.** Bootstrap over pairs: 10,000 resamples, seed 20261104.

### 3.6 Once-only discipline

- **Refusal.**
  - **Files checked.** The script refuses to run if `artifacts/reports/m4_sealed_v02.json` or the start marker `artifacts/reports/m4_sealed_v02.started.json` exists, unless `--force` is passed.
  - **Marker timing.** The marker is written before any sealed file is opened.
  - **Forced reruns.** A forced rerun records the earlier marker(s) in the report.
- **Fail-fast checks before touching sealed data.**
  - all 15 Phase C artefacts present and sha-checked;
  - weight-program artefacts sha-checked;
  - `rules.py` sha-checked;
  - test and OOD data file sha256 checked against `MANIFEST.json` and `OOD_MANIFEST.json`.
- **Provenance recorded.**
  - run timestamp (UTC);
  - `git rev-parse HEAD`;
  - `git status --porcelain`;
  - torch and numpy versions;
  - thread count;
  - sha256 of every input and of the scoring code.
- **Dry run.** `--dry-run-val` runs the same pipeline on **val only**.
  - **What it skips.** It opens no test, OOD or counterfactual file. The sampler fidelity check uses only 2,000 windows.
  - **Output.** It writes the scratch report `artifacts/reports/m4_dryrun_val_v02.json`.
  - **What it leaves alone.** It never writes or checks the sealed marker or report.
  - **Purpose.** It is run before the sealed run, to test the pipeline end to end. Counterfactual scoring is covered by unit tests on hand-built rows.
- **Disclosure.** Sealed results are never shown to any weight author, v0.2 or v0.3 (v0.3 E7).
