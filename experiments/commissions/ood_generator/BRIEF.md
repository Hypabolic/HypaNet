# Commission: out-of-distribution (OOD) holdout generator (spec v0.2 §7.3, role "OOD generator author", §8)

You are the **OOD generator author** for a research experiment that classifies 8-event windows of
AI coding-agent behaviour as PRODUCTIVE / RECOVERY / STALLED / OSCILLATING. A separate team has
written a *standard* synthetic generator that you will never see. Your job is to write an
**independently implemented** generator whose sessions test whether classifiers generalise to
scenario implementations they were not designed around.

## Read first (the only material you may read)
- `agent_doom_loop_detector_spec_v0_2.md` — the full spec. Most relevant: §0.2, §3 (event schema,
  fingerprint rules), §4 (features), §5 (gate), §6 (latent state and oracle), §7 (generator, and §7.3 OOD).
- `docs/LATENT_EFFECT_RULES.md` — binding rules for assigning each operation's latent effect. Labels
  must mean the same thing in your generator as in the standard one, so follow these exactly.
- `src/agent_loop_lab/` — shared contracts you build on and **must not modify**:
  `schema.py` (CanonicalEvent, OracleRecord), `fingerprint.py` (intent/failure fingerprints, workspace
  digests), `oracle_config.py` + `oracle_config.v0_2.yaml` (frozen parameters),
  `synth/latent.py` (LatentStep — one per event), `synth/labeller.py` (window labels from latent steps),
  `encode.py` / `windows.py` (feature encoder and evidence gate, for your tests).

Do not read, search or open anything outside this directory. Do not use the network.

## Deliverable: package `src/agent_loop_lab/synth_ood/`
- Your own world model and engine (repository with real file text and a latent behavioural state,
  faults, tests, external services, token costs) — designed by you, not modelled on anything else.
- `FAMILIES: dict[str, Callable]` in `synth_ood/__init__.py`, each
  `generate(rng: random.Random, session_id: str, cfg: OracleConfig) -> tuple[list[CanonicalEvent], OracleRecord]`.
  - All randomness from the passed `rng`; deterministic (same seed -> byte-identical `to_json()` output).
  - `OracleRecord.generator_version = "ood-0.1"`, `scenario_family = "ood:<name>"`,
    `latent_trace = latent_trace(steps, initial_progress)`, `window_labels = label_session(steps, p0, cfg)`,
    `event_effects` = the steps' effects, `dropped_observations` = sequence indices with `dropped=True`.
  - Events must pass `CanonicalEvent` validation and `windows.validate_sequence`; use only the
    operation families in spec §3.2 and `fingerprint.py` to make fingerprints and digests.
  - Never put labels, family names, latent state or future information on events.
- Family coverage (spec §7.3):
  - New implementations whose sessions exercise all four classes: at least 6 families re-imagining
    the spec §7.1 scenario names with different mechanics.
  - **At least two families absent from the spec §7.1 list** (your choice; e.g. merge-conflict
    ping-pong, dependency install/uninstall thrash, flaky-test quarantine retries, pagination
    polling, config flag toggling, lockfile churn).
  - One under-instrumented family (dropped observations with `dropped=True`, spec §5.2 reasons).
  - Different tool vocabularies (paths, languages, suites, hosts, error classes) and different
    session-length (aim 20–90 events) and token-cost distributions from what you'd consider
    a "default" design.
- `tests_ood/` (pytest), run with `PYTHONPATH=src .venv/bin/python -m pytest tests_ood -q`:
  determinism; labels recomputed from `latent_trace` equal `window_labels`; every event validates;
  every non-instrumentation family passes `windows.gate` on 100% of windows and the instrumentation
  family is rejected exactly on windows containing a dropped event; each family's scored windows
  include the label(s) you intend (state them in the family docstring); `labeller.validate_steps` passes.
- `OOD_GENERATOR.md`: each family, its latent mechanics, intended label mix, and how it differs from
  the spec's §7.1 descriptions.

## Rules
- Do not modify anything under `src/agent_loop_lab/` except creating `synth_ood/`.
- Labels must follow from latent behaviour via `labeller.label_session`; never tune a family by looking
  at encoded features.
- No terminal file redirection tricks are needed; write files with your editing tools.
- When finished, print a short summary (families, intended labels, test results).
