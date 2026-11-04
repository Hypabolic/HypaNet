# Revision request 1 (from the harness implementer, who has measured your output but not changed it)

Your generator is valid (tests pass, gate behaviour correct, all four classes present). As an
out-of-distribution benchmark it is currently too narrow. Measured over seeds 0-49 per family
(`generate(random.Random(seed), ...)`):

| family | distinct (operation_family, outcome) sequences / 50 | scored label mix |
|---|---|---|
| shard_migration | 13 | 100% PRODUCTIVE |
| fixture_archaeology | 27 | 100% PRODUCTIVE |
| lease_reconnect | 39 | PRODUCTIVE 34%, RECOVERY 66% |
| index_watermark | 48 | mostly RECOVERY |
| pinned_validation | 43 | mostly STALLED |
| annotation_spiral | 24 | 100% STALLED |
| serializer_duel | 30 | 100% OSCILLATING |
| merge_ping_pong | 18 | 100% OSCILLATING |
| lockfile_churn | 22 | 100% OSCILLATING |
| telemetry_gaps | 9 | mostly INSUFFICIENT_INSTRUMENTATION |

Please revise `synth_ood/` (same API, same rules as BRIEF.md) so that:

1. **Structural variety:** every family yields at least 40 distinct (operation_family, outcome) sequences
   over seeds 0-49, via varied session structure (number, order and length of phases; optional
   interleaved side work), not just vocabulary.
2. **Realistic transitions:** sessions mix behaviours the way real agent sessions do, e.g. productive
   work before a stall begins, a recovery followed by productive work, an oscillation that is eventually
   broken by a real fix. Every family except possibly one should have at least two different scored
   labels across its windows, and some INSUFFICIENT_AMBIGUOUS windows at transitions are expected.
   Labels must still come only from latent behaviour via `label_session`.
3. **Benign lookalikes:** at least two pairs of families (or variants) that share an identical 8-event
   (category, outcome) sequence but receive opposite scored labels because latent progress differs
   (e.g. a doomed vs a successful retry; churn vs genuine improvement).
4. **telemetry_gaps:** base behaviour drawn from several of your other families, with drops of every
   §5.2 kind, so its non-dropped windows cover several classes.
5. Add tests asserting 1-4 (keep runtime reasonable), keep all existing tests passing, and update
   `OOD_GENERATOR.md`.

Same constraints: read only this directory, do not modify the shared `src/agent_loop_lab/` files.
