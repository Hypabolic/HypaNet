# M1 contracts: synthetic oracle

This is the binding interface for M1 modules. The spec (`agent_doom_loop_detector_spec_v0_2.md`, revision r2, especially §0.2, §6 and §7) takes precedence over this document. Report any disagreement; don't resolve it silently. M0 code and frozen files (`experiments/registry.yaml`) must not change. If they need to, stop and report.

Already written (don't edit; report problems instead):
- `src/agent_loop_lab/synth/latent.py`: `LatentStep`, `latent_trace`, `steps_from_record`.
- `src/agent_loop_lab/synth/labeller.py`: `window_stats`, `label_from_stats`, `label_session`, `validate_steps`. This is §6.3 verbatim.

Package placement is fixed by the frozen guard prefixes. Generator code goes under `agent_loop_lab.synth`, split and dataset tooling under `agent_loop_lab.datasets`, and audits under `agent_loop_lab.eval`. No new top-level packages.

## 1. Latent effects: causal rules the engine must apply

The engine decides each operation's effect from **latent state**, never from encoded features or fingerprints. It applies these rules in this order:

1. **Episodes.**
   - A failure *episode* opens at the first **exact retry**: the same operation on the same relevant state, immediately after it failed or returned not-ready. It does not open on the initial failure. That keeps an ordinary failing test in an edit-then-test cycle from turning a window into RECOVERY.
   - The initial failing operation has effect `NONE`.
   - Kind `retry` (ERROR outcomes): attempts 1..`R_max` are `IN_POLICY_RETRY`, and later attempts are `REDUNDANT`.
   - Kind `poll` (NEUTRAL not-ready outcomes): attempts 1..`P_max` are `IN_POLICY_POLL`, and later attempts are `REDUNDANT`.
   - The operation that succeeds ends the episode: effect `RESOLVES_EPISODE`, `resolved_episode_here = True`.
   - An episode is abandoned when the agent does something else and never retries. It is then closed silently (`open_episode_kind = None`) at the first non-retry operation that changes relevant state, or after 2 unrelated operations.
   - At most one episode is open at a time.
   - Re-running validation on an unchanged workspace after a deterministic failure is still a retry, so attempts within `R_max` are in policy. Retries beyond `R_max` are `REDUNDANT`.
2. **Progress.** An operation that raises latent progress (`progress_after > previous`) is `PROGRESS`. Progress is credited to the mutation that fixes a fault, not to the test run that reveals the fix.
3. **Facts.** An operation that adds facts to `K` (clues about a fault, revealed by inspecting the right location, a diagnostic, or a test run that newly reveals a failure location) is `NEW_FACT`, with `new_facts` = the number added.
4. **Redundancy (§6.1 with the r2 reading in §0.2).** `REDUNDANT` covers:
   - re-reading content whose facts are all already in `K`, or re-reading content already read that yielded no facts;
   - an edit whose **behavioural** change is already present: it changes text but not behaviour, such as reformatting, renaming locals or re-adding an existing guard;
   - validation re-run when the **behavioural** workspace is unchanged since the last run of the same suite and selection, unless it is an in-policy retry under rule 1;
   - out-of-policy retries and polls;
   - re-issuing an identical plan or control step with no new information.
5. **Reversion.** An agent mutation whose latent after-state equals a latent workspace state that existed earlier in the session, and that undoes an earlier agent mutation, sets `undoes_prior_mutation = True`. Its effect is `REVERSION` if the matching state lies within its own trailing 8-event window. The labeller recomputes reversions per window.
6. Everything else is `NONE`: a novel but fruitless inspection, a novel behaviour-changing edit that doesn't help, or a first plan.

If an operation qualifies for several effects, precedence is RESOLVES_EPISODE > PROGRESS > NEW_FACT > REVERSION > IN_POLICY_RETRY/POLL > REDUNDANT > NONE. `new_facts` and `progress_after` are always recorded as they are, whatever the precedence.

`ws_before` / `ws_after` are latent workspace content ids: sha256 of the true file contents, which are unaffected by dropped observations. `CanonicalEvent.workspace_state_*` uses `fingerprint.workspace_digest` over the same contents, so the two agree whenever the observation isn't dropped.

## 2. World and engine (`synth/world.py`, `synth/engine.py`)

- **Repository:** `files: dict[path, str]`, real text, with digests via `fingerprint.workspace_digest`. Each file also carries a latent behavioural state: a set of semantic facts or patch ids, separate from cosmetic text. Cosmetic edits change text only.
- **Faults:** each task has 1–4 latent faults. Each has a location, the clue facts that reveal it, a correct fix (a behavioural patch), and the tests that depend on it.
- **Tests:** suites of test ids. A test passes iff every fault it depends on is fixed and no regression introduced by wrong edits applies. The observable `validation_passed/total` comes from this. Failing runs produce `failure_fingerprint` via `fingerprint.failure_fingerprint` (the first failing test's error class and location).
- **External services:** latent availability. Transient (recovers after k attempts or polls) or persistent (never recovers within the session), plus a not-ready/ready model for polling.
- **Progress:** `p` = fraction of faults fixed, adjusted down for active regressions, in [0, 1].
- **Every operation** goes through the engine. It mutates the world and returns `(CanonicalEvent, LatentStep)`.
  - Intents come from `fingerprint.intent_fingerprint`, using only §3.2 families.
  - Tokens are drawn from per-category lognormal distributions, scaled by a family or session cost profile.
  - Timestamps are monotone ISO-8601.
  - Provenance is `simulator:<generator_version>`.
  - Inspect operations report `before == after` digests.
- **Determinism:** all randomness comes from `random.Random(session_seed)` passed explicitly. No module-level RNG, no set-iteration-order dependence, no time.

## 3. Families (`synth/families/*.py`)

Each family module exposes `FAMILY = "<name>"` and `generate(rng: random.Random, session_id: str, cfg: OracleConfig) -> tuple[list[CanonicalEvent], OracleRecord]`. The record is built with `latent_trace(...)`, and `window_labels` comes from `label_session(...)`. Sessions run 12–60 events. Every family varies, per §7.2:
- tool families and targets;
- paths, suites, error classes and hosts;
- costs;
- success/error ratios;
- session length;
- where the core behaviour sits, via optional productive or neutral prefixes and suffixes.

| # | Family name | Core latent behaviour | Intended label mix (scored windows) | Rule that fires |
|---|---|---|---|---|
| 1 | `incremental_repair` | inspect clue (NEW_FACT), fix edit (PROGRESS), tests improve | PRODUCTIVE | 5 (Δp > ε) |
| 2 | `exploration` | inspections and diagnostics, ≥2 facts per 8 ops, then a meaningful fix | PRODUCTIVE | 5 (F ≥ 2) |
| 3 | `transient_external` | external call times out, in-policy retries, success, then productive work | RECOVERY, then PRODUCTIVE once the resolution is more than `k_rec` back | 4, then 5 |
| 4 | `bounded_retry` | recoverable failure, 1–2 retries, resolved (or a diagnosed repair then retry) | RECOVERY | 4 |
| 5 | `repeated_test_failure` | same tests re-run on an unchanged workspace, same failure, interleaved re-reads | RECOVERY (first ≤ `R_max` retries), then STALLED | 4, then 3 |
| 6 | `cosmetic_churn` | edits that change text but not behaviour, plus tests re-run on a behaviourally unchanged workspace (r2 reading) | STALLED | 3 |
| 7 | `abab_oscillation` | alternating competing edits A/B with tests; progress doesn't net up | OSCILLATING | 2 |
| 8 | `tool_oscillation` | two operations repeatedly undo each other (for example a formatter edit and a revert) | OSCILLATING | 2 |
| 9 | `improving_tests` | repeated edit-and-test of the same suite with genuine improvement each time | PRODUCTIVE | 5 |
| 10 | `polling_recovery` | polls while a dependency legitimately recovers (≤ `P_max`), then ready, then work | RECOVERY, then PRODUCTIVE. Doomed variants (more than `P_max` polls) are STALLED. | 4, then 5 / 3 |
| 11 | `rising_cost` | redundant operations with steadily rising token cost. The label comes from redundancy, not cost. Some sessions do the same with productive work. | STALLED (redundant variant), PRODUCTIVE (productive variant) | 3 / 5 |
| 12 | `under_instrumented` | otherwise normal sessions with dropped observations (tokens, digests, failure fingerprints, validation counts, or truncated history), `dropped = True` | INSUFFICIENT_INSTRUMENTATION for windows containing a dropped event, others as the base behaviour | 1 |

Every family will also produce some `INSUFFICIENT_AMBIGUOUS` windows (rule 6) at transitions. That is expected, not a defect. **Never adjust a family to hit a label target by consulting encoded features.** Labels follow from latent behaviour only.

**Lookalike requirement (§7.2):** identical category/outcome sequences must yield opposite labels across families when observable progress differs. The spec's pairs are examples, and family 5 has no edits by definition, so the required lookalike pairs are these three. Each must produce at least one shared 8-event category/outcome sequence:
- 6 vs 9: MUTATE then VALIDATE-ERROR alternation. 6 stalls, 9 improves.
- 5 vs 4: repeated VALIDATE- or EXTERNAL-ERROR retries. 5 is doomed beyond `R_max`, 4 resolves.
- 7 vs 1: edits with a single intentional revert in 1, against A/B/A/B in 7.

## 4. Counterfactual pairs (`synth/counterfactual.py`)

`generate_pair(kind: str, rng, pair_id: str, cfg) -> tuple[(events_a, record_a), (events_b, record_b)]`. The two sessions share an identical prefix, both events and latent state, and then branch on exactly one causal latent variable. `record.latent_trace["counterfactual"] = {"pair_id", "kind", "branch": "a"|"b", "branch_index", "variable"}`. Required kinds:
- `retry_success_vs_doomed`: a transient vs persistent external failure;
- `improving_vs_identical_failures`: a fix edit vs a cosmetic edit before re-running tests;
- `productive_vs_abab`: a new productive edit vs reverting to the earlier state.

## 5. Datasets (`agent_loop_lab/datasets/`, `scripts/build_datasets.py`)

- The master seed is recorded. Session seed = `int.from_bytes(sha256(f"{master_seed}:{family}:{k}").digest()[:8], "big")`. 20,000 sessions in the standard set, split evenly across the 12 families (remainder by family order). Split 60/20/20 into dev/val/test, by session and seed, within each family. Assignment is a deterministic shuffle keyed on the master seed.
- Outputs go under `datasets/v0_2/standard/` (gitignored data):
  - `{split}.events.jsonl.gz`: one session per line, `{"session_id", "events": [CanonicalEvent.to_dict()...]}`;
  - `{split}.oracle.jsonl.gz`: `OracleRecord.to_dict()` per line;
  - `counterfactual.events.jsonl.gz` and `counterfactual.oracle.jsonl.gz`: 300 pairs per kind.
- gzip uses `mtime=0` and a fixed compression level. Rebuilding gives byte-identical files.
- Committed: `datasets/v0_2/MANIFEST.json` (generator version, master seed, per-split session counts, class counts, family counts, per-file sha256) and `datasets/v0_2/splits.json` (session ids per split).
- Event files and oracle files are separate. The events file must contain no oracle fields.
- A `--sessions N` option allows small builds for tests.

## 6. Audits (`agent_loop_lab/eval/collisions.py`, `agent_loop_lab/eval/dataset_stats.py`)

- **Collision audit (§6.4):** group gate-passing, oracle-scored windows by the exact bytes of their float32 120-vectors. Report collision groups that contain more than one scored label, the collision rate per class and per family, and the Bayes ceiling (accuracy and macro-F1 of predicting each group's majority label; ties broken by canonical class order). Also count identical vectors that appear in more than one split.
- **Dataset stats:** class priors per split, per-family label mix, gate pass rate per family (families 1–11 must be 100%; family 12 windows that contain a dropped event must be rejected, so gate and oracle agree), session length distribution, and the window count.
- The reports go to `artifacts/reports/m1_dataset_stats.json` and `artifacts/reports/m1_collisions_dev.json`. Both are committed and deterministic.

## 7. Tests required

- **Labeller:** a unit test per §6.3 rule and per precedence pair, the RECOVERY recency bound (`k_rec`), the grey zone, eight novel fruitless operations never STALLED, window-relative reversion, and `validate_steps` errors.
- **Engine:** each effect rule in §1, including episode opening on the first exact retry only and the effect precedence.
- **Families:**
  - determinism (same seed gives byte-identical events and records);
  - labels recomputed from `latent_trace` equal the stored labels;
  - no oracle field on any event;
  - all events pass `CanonicalEvent` validation and `windows.validate_sequence`;
  - families 1–11 pass the gate on 100% of windows, and family 12 is rejected exactly on its dropped windows;
  - each family's intended scored label (table above) is the plurality label of its scored windows over 50 seeds.
- **Counterfactuals:** identical prefix, exactly one latent variable differs, and the oracle labels differ after the branch in the intended direction.
- **Datasets:** a small build is byte-identical across two runs, no session id is shared across splits, the manifest hashes match the files, and the split proportions per family are right.
- **Audits:** the collision audit on a constructed collision case, and the Bayes-ceiling arithmetic.
