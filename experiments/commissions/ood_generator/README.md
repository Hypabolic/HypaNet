# OOD generator commission (Codex, isolated)

Author: Codex (codex-cli 0.166.0, model gpt-6.1-sol), in an isolated directory outside the repo.
Inputs: `BRIEF.md`, the full spec, `LATENT_EFFECT_RULES.md` (M1_CONTRACTS §1 excerpt) and copies of the
shared contracts (schema, fingerprint, oracle_config, encode, windows, synth/latent, synth/labeller).
It never saw the standard generator (synth/world.py, engine.py, families/). Hashes of every input are
in `BRIEF.sha256`; logs show no access outside the commission directory.

History:
- v1 (low reasoning effort): valid but narrow (9-48 distinct sequences per 50 seeds; most families single-class).
- v2 (`REVISION_1.md`, high effort): structural variety, transitions, lookalikes, richer telemetry gaps.
- v3 (`REVISION_2.md`): `lockfile_churn` was a duplicate of `merge_ping_pong`; given distinct mechanics; uniqueness tests added.

The harness implementer measured outputs and wrote the revision requests but did not edit the code.
`deliverable/` is the final v3 output, hashed in `DELIVERABLE.sha256`.
