# v0.3 sealed evaluation: pre-registration (spec v0.3 §10, E7, §14 V4)

Written **2026-11-05, before any v0.3 sealed run.** No v0.3 network, reference policy or Phase C model has been run on the test or OOD split. Nothing below may change after the first sealed attempt starts. A change needs a recorded revision in §9, and the report states it.

**Authority.** E7: "v0.3 sealed scoring runs once, after the v0.3 Phase B archive". The Phase B archive exists (`weight_program/v0_3/phase-b/`, sha256 `e2c9aeff…`). The adjudicator asked for the sealed test and OOD evaluation on 2026-11-05.

**Code.** `scripts/v03_sealed_eval.py` and `src/agent_loop_lab/v03/sealed.py`. They reuse the frozen v0.3 validation scorer (`src/agent_loop_lab/v03/scoring.py`), the frozen runner (`runner.evaluate_policy`), `eval/metrics.py` and the v0.2 multi-column scorer `eval/m4.py`. None of these is changed.

## 1. Splits

| Split | Role | Sealed |
|---|---|---|
| test | Sealed standard test set (3,996 sessions) | yes |
| ood | Sealed OOD holdout from the separate Codex generator (2,000 sessions) | yes |
| val | Reference only. It also checks the pipeline before any sealed file is opened (§5) | no |
| counterfactual | Counterfactual consistency only (as in v0.2 M4 §3.1) | no |

- **Data.** For each split, `datasets/v0_3/<split>.raw.jsonl.gz` is rendered with the frozen renderer and tokenised with the frozen tokeniser. Labels come from the v0.2 oracle files. The scored mask comes from the v0.2 runtime gate over the v0.2 canonical events. OOD events and labels are `datasets/v0_2/ood/ood.*.jsonl.gz`.
- **Windows.** A window is `(session_id, endpoint)`. The rules are the Phase B README §1 rules, unchanged: the class of window `t` is the argmax at `EOE_t`, and a window is scored when it passes the gate and has a scored oracle label.
- **Builder.** Test, OOD and counterfactual are built by `sealed.build_split`. It follows `scoring._build_val` step by step: raw/event alignment per event, render, tokenise, `validate_sequence`, `window_endpoints`, gate. It never writes a cache. It opens test or OOD only with the token of the attempt record that this run wrote (§6). Val uses `scoring.load_val()` itself.
- **Token bound.** The renderer caps every block at 191 tokens (§4.6, enforced by `HEADER_MAX_TOKENS`, `BODY_MAX_LINES` and `LINE_MAX_TOKENS`), so the tokeniser's offset check cannot fire on OOD text. If any tokeniser or renderer error happens anyway, the run stops. That is a crash under §6.

## 2. Populations

- **test, ood:** every session. The Phase B disclosed sessions are val sessions, so no exclusion applies.
- **val:** two populations, as in the Phase B README §1: `excluding_disclosed` (primary) and `all_val`. The disclosed set comes from `experiments/v0_3/DISCLOSURE_PRIVATE.json`. On full val the run checks that it equals the README §2 rule.
- **counterfactual:** the contrast endpoints of every pair (v0.2 M4 §3.5).

## 3. Columns

| Column | Source | Labelled examples |
|---|---|---|
| `v0.3 A network` (H3) | `weight_program/v0_3/phase-a/artifact.dlw` | 0 |
| `v0.3 A reference_policy` (H1) | `weight_program/v0_3/phase-a/reference_policy.py` | 0 |
| `v0.3 B network` (H3) | `weight_program/v0_3/phase-b/artifact.dlw` | 200, 1 revision |
| `v0.3 B reference_policy` (H1) | `weight_program/v0_3/phase-b/reference_policy.py` | 200, 1 revision |
| `v0.3 C-<budget> seed<k>` | `experiments/phase-c-trained/v0_3/<budget>/seed<k>/model.pt` | training positions in the log |
| `prior dev-majority` | majority class of gate-passing scored dev windows | dev label counts |
| `prior always-PRODUCTIVE` | constant | 0 |
| v0.2 columns (cited) | `artifacts/reports/m4_sealed_v02.json` | as reported there |

- **H2 fidelity:** `v0.3 A network` against `v0.3 A reference_policy`, and the same for B. It is measured on all gate-passing windows and on scored windows. There is no random-window sampler; spec r1 dropped the grammar-fuzzer sweep.
- **Phase C budgets:** 32, 128 and 512 have 3 seeds each. The report gives each seed, the mean and the sample sd (ddof 1), and a CI of the 3-seed mean on the shared draws. Budgets 2,048 and all have 1 seed. They are reported as that seed; their sd is null.
- **Constant priors.** The v0.3 sessions, labels and gate are those of v0.2, so the prior is the same. It is computed the v0.2 way: the majority over gate-passing scored dev windows, ties to the first class. The dev counts must equal the v0.2 report's `dev_prior`, or the run stops before any sealed file is opened.
- **v0.2 columns are cited, not re-scored.** These are the v0.2 networks and policies, the v0.2 Phase C MLPs, the rules engine and the priors. The report records the sha256 of `m4_sealed_v02.json` and copies its `summary` and `dev_prior` entries verbatim. This is valid only if the window set is identical (§4).

## 4. Identity with the v0.2 evaluation

v0.2 recorded no hash of its window list, so the check rebuilds it. For each of val, counterfactual, test and OOD:

1. **Rows.** v0.2's own loader (`eval.m4.load_split`) builds the v0.2 window rows. The run compares them row by row, in order, with the v0.3 rows: session id, family, endpoint, oracle label and gate result. The sha256 of this canonical list is recorded.
2. **Files.** The v0.2 data hashes recorded in `m4_sealed_v02.json` (`inputs.data_sha256`) must equal the current files' hashes and the v0.2 manifests.
3. **Counts.** The population block (sessions, windows, gate counts by label) and the gate block must equal those in `m4_sealed_v02.json` for that split. Val is compared on `all_val`. Counterfactual has only a gate block.
4. **Priors (confirmation).** After scoring, the v0.3 prior columns' blocks are compared with the v0.2 report's prior columns. They are recorded as `match` true or false. They should match exactly, because the draws depend only on the sessions and the seed.

Checks 1 to 3 run on val and counterfactual before the attempt record, and on test and OOD after it but before any inference. If one fails, the run stops and prints only the name of the failed check. No metric is computed.

## 5. Metrics

- **Definitions.** These are the v0.2 §10.1 metrics, from `eval/metrics.py`, in the multi-column form of `eval/m4.py` (`score_columns`) that the v0.2 sealed run used:
  - macro-F1, accuracy, per-class precision, recall and F1;
  - false-loop alarm rate and the confusion matrix;
  - prediction distributions on ambiguous and instrumentation windows;
  - H2 fidelity (all and scored);
  - per family.
- **CIs.** Session bootstrap: 10,000 resamples, seed `20261104` (`metrics.BOOTSTRAP_SEED`), per-family seeds from `metrics.family_seed`. Every column of one split and population uses the same draws.
- **Policy failures.** These follow the Phase B README §1: a timeout, an exception or an invalid value is an H2 disagreement. In H1 it is recorded as the lowest class that is neither the network's class nor the oracle class. Failure counts are reported by window population.
- **Policy run conditions.** As in the official val scorer (`scripts/v03_score_validation.py`, `score_phase`), the reference policies run only after network inference for the split has finished and the worker pool is idle. The two policies then run one after the other, each through `scoring.policy_predictions` (its 2 guarded children), with nothing else running. The runner's 0.5 s CPU limit per call is therefore applied under the same load on every split. Progress lines give each policy's call and failure counts.
- **Counterfactual consistency** for every v0.3 column, with the v0.2 M4 §3.5 definitions (`m4.counterfactual_consistency`): directional and strict, at endpoint and pair level, with pair-bootstrap CIs.
- **Gate.** `m4.gate_report` per split.
- **CPU latency per session.** This is the time of one `infer.eoe_logits` call per session (`time.process_time`), with 2 torch threads. It is reported twice:
  - **isolated (the reported figure):** 100 val sessions, evenly spaced in file order, one call each after one warm-up call. It runs in the main process after val and counterfactual, before the attempt record, while nothing else runs;
  - **under load:** every session of every split, measured in the worker processes while several workers run at once (no policy runs at the same time). It is inflated by contention and is recorded for completeness.
  
  Each gives the mean, p50, p95 and max per session, and CPU ms per 1,000 tokens.
- **Non-zero parameters.** `artifact.sparsity(model)["overall"]` for every network.
- **Reproduction check on val (before any sealed file is opened).**
  - The v0.3 A and B blocks must equal `artifacts/reports/v03_val_v03-phase-{a,b}.json` exactly, for both populations: network, reference policy, H2 fidelity, per family, population, policy failures and non-zero parameters.
  - The same blocks from `scoring.score` (the official val path) must equal those from `m4.score_columns`.
  - Each Phase C model's macro-F1 on its 500-session early-stopping subset of val must equal `best_es_macro_f1` in its training log.
  - If any of these fails, the run stops before the attempt record.

## 6. One-time rule and attempt record

- **Command.** `uv run python scripts/v03_sealed_eval.py --sealed --confirm-one-time`.
- **Refusal.** The script refuses to start if `artifacts/reports/v03_sealed.json` or the attempt record `artifacts/reports/v03_sealed.started.json` exists. There is no `--force`. `--limit-sessions` and `--report-dir` are refused in sealed mode.
- **Order.**
  1. Check every input hash, without opening a sealed file:
     - each archive against `artifact.sha256`, its metadata against the registered packet, and its policy hash and static rules (`scoring.load_phase`);
     - each Phase C `model.pt` against `model.sha256` and the training log (not a smoke run, with the budget and seed of its path);
     - the renderer and tokeniser sources against `MANIFEST_v0_3.json`;
     - the v0.2 manifests against `MANIFEST_v0_3.json`;
     - val, dev and counterfactual files against the manifests;
     - the sealed files' recorded hashes, compared across `MANIFEST_v0_3.json`, `MANIFEST.json`, `OOD_MANIFEST.json` and `m4_sealed_v02.json` (hash against hash; no sealed file is read).
  2. Score val and counterfactual. Run the §4 and §5 checks. Measure the isolated latency.
  3. Write the attempt record: start time, git state, input and code hashes, and a random token.
  4. Hash the test and OOD files against the manifests. Then decode them.
  5. Run the identity checks (§4), inference, policies and metrics, all in memory.
  6. Write the report atomically (temporary file, then rename). Then print the summary.
- **Progress lines** give only counts and timings. No metric is printed or written before step 6.
- **Crash semantics.** If the run stops after step 3, the attempt record stays as the record of a spent attempt. No sealed metric has been shown, so nothing has leaked. A second attempt needs the adjudicator. The adjudicator renames the record to `v03_sealed.attempt<N>.json` and states the reason. The script then embeds every earlier attempt record in the new record and in the report, which reports the run as attempt N+1. A stop before step 3 (a failed check on unsealed data) is not an attempt.
- **Running it.** The run takes about 2.5 to 3.5 hours (the policies take about 6 minutes per split on val). Start it detached from any agent session, for example `nohup caffeinate -i uv run python scripts/v03_sealed_eval.py --sealed --confirm-one-time > sealed.log 2>&1 &`. A kill after step 3 is a spent attempt. An external kill leaves the worker processes and the guarded policy children running; kill them by hand.
- **Never shown to a v0.3 author.** No sealed result, val result or derived statistic goes to any v0.3 weight author or author directory.

## 7. Dry run

`uv run python scripts/v03_sealed_eval.py --dry-run-val --report-dir DIR [--limit-sessions N]`

- It scores val and counterfactual with the same code and columns. It never opens test or OOD and never reads or writes the attempt record. `DIR` may not be `artifacts/reports/`. The report says `"official": false`.
- `--limit-sessions N` (dry run only) keeps the first N sessions of val and counterfactual, and only complete counterfactual pairs. With a limit, the comparison against the official val reports, the v0.2 count comparison and the Phase C early-stopping check are skipped and marked as skipped. The comparison of `scoring.score` against `m4.score_columns` and the row-level v0.2 identity check still run.

## 8. Reporting

- **Report.** `artifacts/reports/v03_sealed.json` holds:
  - a summary table per split: macro-F1 and false-alarm rate with CIs, recall per class and H2 fidelity, plus Phase C mean ± sd per budget;
  - the full blocks per split and population;
  - counterfactual consistency, gate reports, latency and non-zero counts;
  - the identity and reproduction checks;
  - the cited v0.2 summary;
  - every input and code hash.
- **Labelled examples.** Phase A: 0. Phase B: 200 windows and 1 revision. Phase C: the training positions of each run.
- **Optimism.** Val is reference only. Phase C early-stopped on 500 val sessions, and the 200 Phase B disclosed sessions are val sessions.

## 9. Revisions

None.
