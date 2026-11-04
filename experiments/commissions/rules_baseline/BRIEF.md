# Commission: independent rules baseline (spec §9.5, role "Rules-baseline author", §8)

You are the **rules-baseline author** for a research experiment. Your job is to write a
deterministic, hand-written rules engine that classifies an 8-event agent-trace window into
one of four classes, using only the 120 input features described in `SPEC_AUTHORING.md` §4
and `feature_index.json`. You do NOT write neural-network weights; that is a different role.

## Inputs (the only material you may read)
- `SPEC_AUTHORING.md` — the experiment spec (domain semantics, features, class definitions, oracle rules over latent state in §6).
- `feature_index.json` — flat index of each (position, feature) in the 120-vector, and class order.

Do not read, search or open anything outside this directory. Do not use the network.
There is no data, and no labelled examples exist for you; that is deliberate (the baseline is pre-registered).

## Deliverables (write them in this directory)
1. `rules.py` — defines `def rules_classify(x: tuple[float, ...]) -> int`.
   - `x` is a tuple of exactly 120 Python floats ordered as in `feature_index.json` (8 events oldest-first × 15 features).
   - Return 0..3 in the canonical order `[PRODUCTIVE, RECOVERY, STALLED, OSCILLATING]`.
   - Standard library only; pure and deterministic; no I/O; no global mutable state.
   - Windows have already passed the runtime evidence gate (§5.2), so every window you see is fully instrumented. You must always return one of the four classes.
2. `RULES.md` — each rule, its order/precedence, its thresholds, and why (cite spec sections). Explain how you approximate the latent-state oracle (§6) from observable features, and where you expect the rules to be wrong.
3. `test_rules.py` — `unittest` tests on hand-constructed windows of your own (no data exists), covering each rule and the precedence between them. Run them with `python3 -m unittest test_rules -v` (python 3.12+ is available as `python3`).

## Rules for you
- No fitting, search or tuning against data — you have none, and you must not generate a synthetic dataset to tune thresholds. Thresholds come from reasoning about the spec.
- Keep it simple and inspectable; this is a baseline that the experiment's other methods are compared against.
- When finished, print a short summary of the rules.
