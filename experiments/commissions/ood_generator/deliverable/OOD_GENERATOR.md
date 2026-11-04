# Independent OOD generator 0.1 — revision 2

`agent_loop_lab.synth_ood.FAMILIES` exposes ten callables accepting
`(rng: random.Random, session_id: str, cfg: OracleConfig)` and returning
`(list[CanonicalEvent], OracleRecord)`. The API and generator version remain
unchanged. Only this commission directory was consulted; no standard generator,
datasets, classifiers, weights or network services were used.

## World and causal model

`engine.py` maintains an in-memory repository with actual text contents. A
small contract language interprets `enabled=1` as a satisfied shard predicate
and `enabled=0` as a fault. Each shard also has a deployment `mode`. Comments
are behaviourally inert. Progress is the fraction of satisfied contracts;
validation counts come from the same predicates. This is a simulated language
interpreter, not execution of Elixir, Zig, Lua or Clojure. Those extensions and
suite names provide varied tool vocabulary.

Inspecting an unexplored contract reveals its predicate and fixture, adding
two material facts. The engine tracks knowledge, read history, behavioural
validation history, controls, workspace contents, one active failure episode,
and independent mirror/index readiness counters. Request bodies distinguish
service retries; backoff metadata does not. Service success reveals lease
knowledge without repairing repository faults.

Lockfile and telemetry sessions additionally enable a package resolver. Its
registry has three immutable cedar artifacts with ABIs 1, 2 and 3; the
application manifest requires ABI 3. A simulated HTTP GET fetches an artifact
into a cache and reveals its ABI, without changing the workspace. A simulated
POST installs a cached artifact into `vendor/cedar.artifact`; an unfetched
artifact cannot be installed. DELETE removes the installed artifact. These
requests run entirely in memory, with no network or package-manager execution.

Dependency builds read the lockfile, installed artifact and application
manifest. The single dependency assertion passes only when package identity,
locked/installed versions and the required ABI match. This assertion adds one
objective to the progress denominator alongside the shard predicates. Neither
fetching a compatible release nor rewriting its pin alone earns progress;
installing it with matching pins repairs the dependency fault. Other families
retain their original shard-only objective and engine initialization.

Effects follow the binding latent-effect rules. The first failure opens no
episode. An exact consecutive retry opens one; retries and polls consume the
config's budgets. Success resolves the episode. A relevant behavioural change
or two unrelated operations abandon it. Progress belongs to the mutation that
repairs a predicate. Comments remain redundant despite changing digests.
Restoring earlier contents records an undo; the shared labeller recomputes
reversions within each window. Progress, facts and undo flags remain recorded
even when another effect takes precedence.

Fingerprints and digests use the shared fingerprint module. Public events
contain no scenario tags or hidden effects. Callers should supply opaque
session IDs. Sealed records use `generator_version="ood-0.1"`,
`scenario_family="ood:<name>"`, `latent_trace` and `label_session`. Scripts
choose operations, never effects or labels, and do not consult encoded
features. `OracleRecord.seed` is a nonce drawn from the supplied RNG, not the
caller's original seed; replay requires that original seed or saved RNG state.

## Families and intended labels

The labels below are minimum coverage per session under frozen v0.2 parameters,
except telemetry's additional classes depend on its sampled workflows.
Optional phases can add other labels. Ambiguous windows are retained.

| Family | Intended scored labels | Mechanics and difference from §7.1 |
|---|---|---|
| `shard_migration` | PRODUCTIVE, RECOVERY | Eight independent shard repairs precede one to three retry/poll episodes. Optional commentary can stall; actual repairs resume afterward. Reimagines incremental repair and improving validation (§7.1.1, 9) as distributed migration interrupted by quorum acquisition. |
| `fixture_archaeology` | PRODUCTIVE, STALLED | Discover five to eight predicates/fixtures, optionally attempt a repair, then pursue inert documentation before repairing real faults. Reimagines exploration (§7.1.2) as an investigation with a dead end. |
| `lease_reconnect` | PRODUCTIVE, RECOVERY | Repairs surround two to four mirror outages. Optional investigation and a watermark wait add another external phase. Reimagines transient failures and bounded retries (§7.1.3, 4) as multi-host lease acquisition. |
| `index_watermark` | PRODUCTIVE, RECOVERY | Two to four neutral not-ready episodes have variable delays, with optional repairs between waits. Backoff does not distinguish intents. Reimagines polling (§7.1.10) as replicated-index maintenance. |
| `pinned_validation` | PRODUCTIVE, RECOVERY, STALLED | Useful repair precedes 14–24 unchanged failing validations. Early retries are in policy; later retries stall. Optional discovery and actual repairs break the loop. Reimagines §7.1.5 as a pinned replica assertion. |
| `annotation_spiral` | STALLED, PRODUCTIVE | Eight comment/test cycles change text without progress. Optional further notes precede actual predicate repairs; an optional lease outage and more repair extend the session. Reimagines ineffective edits and rising expenditure (§7.1.6, 11). |
| `serializer_duel` | OSCILLATING, PRODUCTIVE | Opposing emitters restore concrete wire conventions. Variable continuation intersperses inspections, plans and tests. Optional recovery precedes real repairs that end the oscillation. Reimagines workspace/tool oscillation (§7.1.7, 8). |
| `merge_ping_pong` | PRODUCTIVE, OSCILLATING | Repairs precede 10–16 competing ownership resolutions, with optional inspections/plans/tests. Optional recovery and actual repairs follow. **Absent from §7.1:** upstream/local merge ownership negotiation; no real git merge runs. |
| `lockfile_churn` | PRODUCTIVE, OSCILLATING | Repairs precede fetches of two ABI-incompatible releases. Six to ten cycles rewrite the lockfile, install the chosen cached artifact, and run a failing dependency build, optionally inspecting the lockfile. Restoring earlier pins and installed artifacts causes real reversions. Fetching and installing release 3 resolves the ABI fault, either in place or after uninstalling; renewed shard repair follows. **Absent from §7.1:** a resolver/installer loop whose validation depends on pins and installed ABI. |
| `telemetry_gaps` | PRODUCTIVE and at least one other scored class | Sample two workflows from validation stalls, serializer/merge oscillation, package resolution, lease retries and watermark polls, surrounded by real repairs. The package variant uses five to six install/build cycles and a compatible installation. A transport-loss burst follows the clean workflows. Reimagines under-instrumentation (§7.1.12) across several behaviours. |

## Mechanism independence

`merge_ping_pong` owns its ownership-negotiation workflow directly; the former
shared `_negotiation` controller has been removed. `lockfile_churn` uses a
separate resolver workflow with cached external artifacts, installed workspace
state and a dependency-specific build assertion. Merge ownership changes
cannot repair a shard predicate; installing a compatible pinned package can
repair the dependency objective. Lockfile histories always contain EXTERNAL
installation operations and `build` validations, while merge histories contain
no `build` operations and only optional external lease/poll recovery.

The other registered families were checked for shared core controllers. Their
existing migration, discovery, multi-host retry, interleaved watermark, pinned
validation, annotation and serializer workflows remain separate. Small repair,
inspection and recovery helpers are shared. Telemetry intentionally samples
several mechanisms, as required by Revision 1, and now samples the real
resolver workflow rather than a path-swapped ownership oscillation.

Across seeds 0–49, every pair of families has disjoint full-session
`(category, outcome)` sequences, including comparisons between different seed
indices. All 45 family pairs also differ in their aggregate operation-family
multisets across those seeds. This full-session requirement preserves the
intentional eight-event lookalike windows below.

## Structural variety and transitions

Each repair chooses a still-broken predicate and independently chooses between
inspection before writing, planning before writing, inspection after validation,
or recording results after validation. Investigation optionally interleaves
tests or plans. Recovery varies episode counts, protocols, readiness delays
and side work between resolved episodes. Oscillation varies lengths and the
operations between reversals. Optional phases and handoffs vary phase counts
and boundaries. Randomness changes the number and order of operations, beyond
vocabulary or token costs.

The resolver varies fetch ordering, install/build cycle counts, intervening
inspections, and direct replacement versus uninstall/reinstall resolution.
Its incompatible cycles restore earlier complete workspace states; its final
compatible installation increases latent progress and changes the dependency
build from ERROR to SUCCESS. Both behaviours occur within each lockfile session.

Sessions include productive work before stalls, recovery followed by renewed
productive work, and oscillation followed by mutations that actually increase
progress. Phase boundaries are allowed to yield INSUFFICIENT_AMBIGUOUS. They
are not forced to yield a fixed label schedule: some recovery-to-stall
boundaries move directly between scored labels.

The repository has 12–18 contracts, four language/suite vocabularies, varying
error classes and generated resource identities. Generation uses only the
passed RNG, with no I/O or wall-clock reads. Across seeds 0–49, observed
lengths are 28–79 events, within the commissioned 20–90 range. Lockfile
histories span 46–70 events and telemetry histories span 46–79 events.

Costs are zero with probability 0.13; otherwise they are lognormal with
parameters `(7.5, 1.65)`, capped at 400,000 tokens and randomly divided between
input/output. Costs are independent of labels. High spend is not a stall rule.

## Benign lookalikes and counterfactuals

Two family pairs deliberately share their first 17 operation/category/outcome
events: `shard_migration` versus `annotation_spiral`, and `shard_migration`
versus `serializer_duel`. At endpoint 16, their identical eight-event
category/outcome windows are PRODUCTIVE versus STALLED and PRODUCTIVE versus
OSCILLATING, respectively. The causal difference is repairing distinct faults,
adding inert comments, or restoring competing contents. Variable continuations
follow these probes, so the paired prefixes do not determine session labels.

`synth_ood.counterfactuals.counterfactual_pairs(rng, session_id, cfg)` retains
the three §7.2 paired experiments. Nine byte-identical prefix events precede
one intervention: service persistence, or effective edits versus annotations
or reversals. Endpoints contrast PRODUCTIVE with STALLED or OSCILLATING.
Both alternatives share prefix IDs: keep a pair together in dataset partitions
and treat branches as alternative sessions, never concatenated histories.
Continuation costs still consume the passed RNG. These pairs remain separate
from the ten registered families.

## Telemetry loss and restoration

Telemetry samples two workflows without replacement from six mechanisms
shared with other families. Clean windows collectively cover all four scored
classes over seeds 0–49; each session has at least two scored classes.

The loss burst omits input tokens, both workspace digests, a failure
fingerprint, then suite id, passed count and total count on separate validations.
The next validation has complete current fields but an unobserved baseline.
The engine derives this dependency from previous validation observations and
sets that step's `dropped=True` too. A complete result restores the following
result's baseline. Rejected full windows are exactly those containing an
affected step, including dependent baseline loss. Rejection ends eight events
after the final affected event. Later repairs provide clean post-gap windows.

All representable §5.2 missing-observation reasons are exercised:
MISSING_TOKENS, MISSING_WORKSPACE_DIGEST, MISSING_FAILURE_FINGERPRINT,
MISSING_VALIDATION_FIELDS and UNOBSERVED_VALIDATION_BASELINE. Short prefixes
exercise TOO_SHORT. Category and outcome remain present because the shared
CanonicalEvent schema requires them; missing category/outcome cannot be emitted
while satisfying the commission's schema-validation rule. Telemetry loss
changes observations and dropped flags, never true contents or causal effects.

## Verification

Run from this directory:

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests_ood -q
```

The original contract audit covers ten families over 24 seeds: byte-identical
event/record serialization, schema round trips, sequence and latent validation,
label recomputation, effect/dropped-index consistency, intended labels and each
window's gate/encoder path. Its gap test now requires all five observation-loss
reasons and restoration after the last overlapping loss. Causal engine and
counterfactual checks remain.

`tests_ood/test_revision.py` reuses a 500-session corpus for seeds 0–49. It
requires at least 40 distinct operation-family/outcome sequences per family,
varied operation order after collapsing repeated runs, at least two scored
labels per session, ordered transitions and actual progress after problematic
phases, ambiguous boundary windows, both lookalike pairs, telemetry class and
reason coverage, and exact agreement between gate rejection and dropped steps.
Focused baseline-loss tests verify restoration and unchanged causal state.

Revision 2 adds all-pairs full-session uniqueness and aggregate operation
multiset checks on the same corpus. Causal resolver tests require fetching
before installation, prove that pin changes alone do not fix builds, exercise
all three release ABIs, change the manifest to demonstrate text-derived build
results, and check pin mismatches plus uninstall/reinstall effects. The corpus
audit requires reverted installed states, constant progress during incompatible
cycles, eventual compatible installation progress and a successful build, and
both resolution routes across seeds.

The revision audit measured **50 distinct sequences per family** and **two or
more scored labels in all 500 sessions**. Telemetry's clean windows cover
PRODUCTIVE, RECOVERY, STALLED and OSCILLATING. The tests assert these acceptance
properties rather than selecting seeds or tuning against encoded features.
The complete suite passes **315 tests**.
