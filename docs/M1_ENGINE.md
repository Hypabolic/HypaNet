# M1 engine: family author guide

This guide covers `src/agent_loop_lab/synth/world.py` and `engine.py`, which implement M1_CONTRACTS §1–§2. It is for anyone writing a family module under `synth/families/`. If this guide disagrees with the spec (r2) or M1_CONTRACTS, those win. Report the disagreement.

A family is a short script of operations against a seeded `Session`. Every operation mutates the latent world. From that latent state, and never from encoded features, the engine decides the effect, then emits one `(CanonicalEvent, LatentStep)` pair. `finish()` labels the session with `labeller.label_session`. Never pick operations by looking at fingerprints, digests or encoded features. Branching on latent state is fine: `s.world`, `s.knows()`, `s.progress`, `s.episode`, `s.first_failure()`. So is branching on what the agent observes: `OpResult.outcome`, `failing_test`, `passed`.

## Module contract

```python
FAMILY = "my_family"            # must equal the module name and a FAMILY_NAMES entry

def generate(rng, session_id, cfg):
    s = Session(rng, session_id, cfg, family=FAMILY, n_faults=2)   # extra kwargs go to World.generate
    ...operations...
    return s.finish()           # (events, OracleRecord)
```

`families.generate_session(family, seed, session_id, cfg)` is the entry point. It builds `random.Random(seed)`, calls `generate`, and stamps `record.seed`. `generate` can't know the seed, so a direct call leaves it at 0. All randomness must come from `rng` (also available as `s.rng`). Don't use module-level RNG, the clock, `hash()`, or set iteration order.

## The world (`World.generate(rng, ...)` / `World.from_spec(...)`)

| Piece | Model |
|---|---|
| Files | Each file is a header plus ordered named **blocks**: functions in source files; keys, sections or log entries in data files. Each block has a `behaviour` id (`orig`, `fix`, or a wrong, neutral or competing variant) and a cosmetic style (rename/comment counter, `formatted` flag). The rendered text is real. The latent ws id is `workspace_digest(world.texts())`. The behaviour key ignores cosmetic style. |
| Languages | python, typescript, go, rust, java, ruby. Each has its own paths, suite ids, test-id format, error classes, config files, and lint/build targets. |
| Faults | 1–4 per world (`n_faults=`). Each `Fault` has `id, path, block, symbol, error_class, suite, tests (>=2), facts`. It is fixed iff its block's behaviour is `fix`. |
| Facts | `F:loc` comes from the fault's first failing test in a run, or from `search(scope ⊇ path, query ⊇ symbol)`. `F:code` comes from a read whose range overlaps the block. `F:c1..` are cause clues (`extra_clues=(lo, hi)`): a config or log read, a keyword search, `lint(target)` or `build(target)` failing, or an observability `http GET`. `s.unknown_clues(f)` lists the clues still unknown, and `s.inspect_clue(c)` runs the right operation for one. |
| Tests | A test passes iff each of its faults is fixed (or a competing variant happens to cover it) and no regressing variant sits on a block it covers. Suite order is sorted test id. The "first failing test" gives `failure_fingerprint("run_tests", error_class, test_id)` and the revealed fact. |
| Progress | `p = clip((fixed − 0.5·active_regressions) / n_faults, 0, 1)`. Because of the clip, a regression only lowers `p` once something is fixed. |
| Services | `world.add_service(host, mode, fail_attempts=k, error_class=, requires=)`, with `mode` one of `ok`, `transient` (the first k calls fail), `persistent`, or `needs_fix` (fails until fault `requires` is fixed, i.e. a diagnosed-repair cause). An unknown host is auto-created as `ok`. |
| Resources | `world.add_resource(id, ready_after=k)`: not ready for k polls, then ready. `ready_after=None` means it is never ready. |
| Vocabulary | `world.vocab` holds per-world picks of hosts, path templates, poll resources, decoy queries, plan actions and service names. It also has `world.lint_targets`, `world.build_targets`, `world.suites` and `world.decoy_paths()`. |

`World.from_spec(language=, files={path: [blocks]}, faults=[{id, path, block, suite, tests, error_class, clues}], healthy_tests={suite: [(test_id, ["path:block"])]}, services=[...], resources=[...])` builds a world by hand. `tests/test_engine.py::small_world` is an example.

## Operations and their effects

Every operation takes `cost=` (a one-off token multiplier) and `drop=` (see Instrumentation). Each returns an `OpResult`: `.event`, `.step`, `.effect`, `.outcome`, `.ok/.failed/.not_ready`, `.new_facts`, plus `.failing_test/.error_class/.passed/.total` where they apply.

| Call | Family / category | Outcome | Effect when not an episode attempt |
|---|---|---|---|
| `read(path, lines=None, block=, pad=)` | read_file / INSPECT | SUCCESS, or ERROR if the path is missing | NEW_FACT if it reveals a fact. REDUNDANT if no new fact and either the same content was read before or all of the content's facts are already in K. Otherwise NONE. |
| `search(scope, query, limit=)` | search / INSPECT | SUCCESS | Same rule as `read`. Content = (scope, normalised query, scope text). |
| `list_dir(path)` | list_dir / INSPECT | SUCCESS, or ERROR if missing | NONE the first time, REDUNDANT if the listing was seen before |
| `novel_decoy(kinds=)` | read, list or search | SUCCESS | Always NONE: an unread decoy, an unlisted directory, or a clue-free search |
| `edit(path, change, block=)` | edit_file / MUTATE | SUCCESS, or ERROR if missing | See the change table |
| `write(path, change)` / `write(new_path)` | write_file / MUTATE | SUCCESS | Same as `edit`. Creating a file is a novel behaviour (NONE). |
| `revert(paths, to="last"/"initial")` | revert / MUTATE | SUCCESS | Pops the per-file undo stack, or restores the session-start content |
| `run_tests(suite, tests=None)` | run_tests / VALIDATE | SUCCESS if all selected tests pass, else ERROR | NEW_FACT if the first failing test's fact is new. REDUNDANT if the behaviour key is unchanged since the last run of the same suite and selection. Otherwise NONE. |
| `lint(target)` / `build(target)` | lint / build / VALIDATE | ERROR while a matching clue's fault is unfixed | NEW_FACT reveals **all** matching facts. Redundancy is keyed on behaviour, so text-only changes don't count. |
| `http(method, host, path_template, body=)` | http_call / EXTERNAL | ERROR while the service is down | On success, NEW_FACT if an observability clue matches, and REDUNDANT if this identical call has already succeeded. The initial failure is NONE. |
| `poll(resource, backoff=)` | poll_status / EXTERNAL | NEUTRAL while pending, SUCCESS when ready | REDUNDANT if the status equals the last poll of that resource. The first poll is NONE. |
| `plan/ask/yield_(action_type)` | CONTROL | SUCCESS | REDUNDANT if the same action was issued with no new information since. Information = a new fact, a change in `p` or behaviour, a resolution, or a never-seen observation. |

| Change | Latent effect |
|---|---|
| `fix(f)` | The block becomes `fix`, so `p` rises (PROGRESS). On an already-fixed block it re-adds the guard: text only, so REDUNDANT. |
| `cosmetic(kind)` | `format` is file-wide and idempotent. A no-op format is REDUNDANT with `agent_mutation=False`. `rename_local`, `comment`, `whitespace` and `docstring` each add new text. All are REDUNDANT (r2 reading). |
| `competing(f, "A")` | A competing alternative on the fault's block. It never fixes the fault and causes no regression, so `p` stays flat. Alternative *i* passes the fault tests at positions `j` where `(j+i) % 2 == 0`, so A and B pass different tests. Re-applying the current label is REDUNDANT. |
| `wrong(regress=False)` | A novel behaviour that doesn't help (NONE). `regress=True` breaks the tests covering the block and lowers `p`. The default block is the first *unfixed* fault block in the file, else the first block. Pass `block=` to be explicit. |
| `neutral()` | A novel behaviour with no test effect (NONE) |
| `undo()` | Reverse patch of this file's most recent mutation. It usually gives REVERSION, or PROGRESS if it removes a regression. |

### Episodes (rule 1): read this before writing any retry, poll or test loop

- **Arming.** An ERROR, or a NEUTRAL poll, arms a candidate keyed on the operation (family, target, args) and its relevant state. Relevant state is the behaviour key for run_tests/lint/build, the text ws for inspections and mutations, and the service or resource for external calls.
- **Opening.** Only the **immediately next** operation can open the episode, and only if it has the same key and the same relevant state. That operation is attempt 1. The initial failure is *not* an attempt: rules 2–6 decide its effect, so it is NONE, or NEW_FACT if it revealed something.
- **While open**:
  - A same-key operation is the next attempt.
  - One unrelated operation in between is tolerated. The episode closes silently at the 2nd consecutive unrelated operation.
  - It also closes when a non-repair operation changes its relevant state, e.g. `neutral()` during a test episode. A text-only edit doesn't change behaviour, so it only counts as unrelated.
  - A **diagnosed repair** keeps it open: a mutation that fixes one of the episode's causes. The causes are the faults or regressions behind the failing tests, or a `needs_fix` service's fault. The next same-key operation then counts as an attempt even though the state changed.
- **Attempts.**
  - Attempts `1..R_max` (3) are IN_POLICY_RETRY, and attempts `1..P_max` (5) are IN_POLICY_POLL.
  - Later attempts are REDUNDANT, and the episode stays open with `attempts > limit`. That doesn't count as "in policy", so STALLED becomes possible.
  - The attempt that succeeds is RESOLVES_EPISODE. That includes attempt 1, e.g. fail, then a retry that succeeds.
- **Exclusivity.** At most one episode is open at a time. Opening a new one replaces the old.
- **Precedence.** RESOLVES_EPISODE > PROGRESS > NEW_FACT > REVERSION > IN_POLICY_* > REDUNDANT > NONE.

### Reversion (rule 5)

`undoes_prior_mutation` is set when a mutation's after-state equals any ws the session held earlier. The effect is REVERSION only if that state is in the trailing window, i.e. the ws before/after of the previous 7 steps, as `labeller.window_stats` computes it. The labeller recomputes reversions per window from these flags. A/B alternation (`competing` A, B, A, B), cosmetic-then-`undo()` cycles, and `revert` cycles all revisit states. The 1st and 2nd edits of an A/B run are novel (NONE). Edit 3 onwards is a reversion.

## Getting each label (labeller §6.3, per 8-event window)

| Label | What the window needs | Typical script |
|---|---|---|
| PRODUCTIVE | Δp > 0.02 (a `fix`) or ≥ 2 new facts | inspect clue, `fix`, `run_tests` (a failing run reveals the next location) |
| RECOVERY | An open episode within policy at the endpoint, or a resolution in the last `k_rec`=4 events | `http` fail, retry, retry, success. `poll` pending ×k, then ready. Failing test, retry, ≤1 inspection, `fix(cause)`, retry. |
| STALLED | Δp ≤ ε, **F = 0**, ≥ 5 REDUNDANT/REVERSION, the last event REDUNDANT, no in-policy episode at the endpoint | cosmetic edit / non-immediate re-run alternation; re-reads of known content; attempts beyond R_max |
| OSCILLATING | ≥ 2 window reversions and Δp ≤ ε | (`competing` A, test, `competing` B, test) × 2+; (`cosmetic`, `undo`) × 2+ |
| INSUFFICIENT_INSTRUMENTATION | Any dropped event in the window | `drop=...` or `finish(truncate_before=m)` |
| INSUFFICIENT_AMBIGUOUS | Everything else, e.g. transitions, aimless NONE runs, 3–4 stall-ineffective operations | expected, not a defect |

## Instrumentation drops (family 12)

- **Field drops.** Pass `drop=(...)` with names from `DROP_FIELDS`: `tokens`, `input_tokens`, `output_tokens`, `digests`, `digest_before`, `digest_after`, `failure` (ERROR events only), `validation_counts` or `validation_suite` (validation events only). The observed field becomes `None` and the step gets `dropped=True`.
- **Truncation.** `finish(truncate_before=m)` drops the first m events. The remaining events keep their `sequence_index` and get `history_complete=False`.
- **Propagated drops.** At `finish`, the engine also marks `dropped` any validation event whose baseline is unobserved in the §5.2 sense. That covers three cases:
  - the scan back hits an event with a dropped suite;
  - the same-suite baseline has dropped counts;
  - there is no same-suite baseline after truncation.

  `record.dropped_observations` lists deliberate and propagated drops. `latent_trace["drops"]` separates them. The gate then rejects exactly the windows labelled INSUFFICIENT_INSTRUMENTATION (`synth_helpers.assert_gate_matches_drops`).
- **Alignment.** `event_effects` is aligned with the returned events list. After truncation that is by list position, not `sequence_index`.

## Costs and time

Tokens are drawn per category from a lognormal around fixed medians. Each session draws a `CostProfile` with a scale, a sigma and per-category multipliers, or you can pass `cost_profile=`. ERROR events cost 1.3×, and reads scale with the number of lines. Set `s.cost_multiplier` to scale every later operation, e.g. raise it each step for rising cost (family 11). Cost never affects effects or labels. Timestamps are monotone ISO-8601 ms values with category-dependent durations drawn from `rng`.

## Pitfalls

1. **Accidental episodes.** Repeating a failing `run_tests`, `lint` or `build`, a failing `http`, or a pending `poll` as the very next operation opens an episode, so the windows turn RECOVERY. Productive families must put something between them, and for tests that means a behaviour change.
2. **Accidental redundancy.** All of these are REDUNDANT:
   - searching for a symbol whose location is already known, or re-reading a block whose code fact is known;
   - re-running a suite and selection with no behaviour change, even after a cosmetic edit;
   - an identical successful HTTP call;
   - the same plan twice with nothing new in between;
   - `fix()` on a fixed block;
   - every cosmetic edit.

   `novel_decoy` pools include test files, so a later `read` of a test file the decoy already read is REDUNDANT.
3. **STALLED needs F = 0.** The failing run that first reveals a location is NEW_FACT. Windows that contain it can't be STALLED, so a stall only shows once it has left the window.
4. **One fact per test run.** `run_tests` reveals only the first failing test's fact. `lint` and `build` reveal all their matches.
5. **Never touch `world.files` / block states outside an operation**, or `validate_steps` fails on the `ws_before == previous ws_after` chain. Changing a service or resource (e.g. `world.services[h].mode`) is a latent change. Only do it to set up the world, or straight after `fork()` for a counterfactual branch.
6. **Length.** `finish()` raises `SessionLengthError` outside 12–60 events, and it never trims. Budget with `s.n` and `s.room(k)`, and test 2,000+ seeds.
7. **The diagnosed-repair window is tight.** With fail, retry, read, read, the 2nd read closes the episode. Use fail, retry, ≤1 inspection, `fix(cause)`, retry.
8. **Competing variants never fix.** An oscillation family that wants the session to end green needs a separate `fix(f)` at the end.
9. **Identifiers carry no oracle information.** Standard session ids are `s-{seed:016x}`. Counterfactual sessions use `counterfactual.branch_session_id(pair_id, branch)` (`cf-` + 16 hex digits of a hash), never `<pair_id>-a` / `-b`. The builder also orders each pair's two lines by `counterfactual_branch_order(pair_seed)`, so neither ids nor file position reveal which branch is the degenerate one. The branch is recorded only in `latent_trace["counterfactual"]`. Derive ids by hashing, not from the session `rng`, or every later draw shifts.

## Lookalike strings (§7.2)

- **7 vs 1.** Family 1 emits `(MUTATE,SUCCESS),(VALIDATE,ERROR)` × 4, containing exactly one `undo()`. The sequence is fix(prev), test, wrong, test, undo, test, fix, test, with net Δp > 0, so it is PRODUCTIVE. Family 7 should emit the same category/outcome string as `competing` A, test, B, test, A, test, B, test with Δp = 0, so it is OSCILLATING.
- **6 vs 9.** Both alternate MUTATE-SUCCESS and VALIDATE-ERROR. In 6 the edits are `cosmetic` and the runs are REDUNDANT; in 9 they are `fix`es with improving counts.
- **5 vs 4.** Both are repeated VALIDATE-ERROR or EXTERNAL-ERROR retries. 4 resolves (transient or `needs_fix` plus repair). 5 goes past R_max (persistent, or unchanged behaviour).

## Worked example (shaped like family 3)

```python
from ..engine import Session, fix
from ..world import HTTP_ERRORS

FAMILY = "transient_external"

def generate(rng, session_id, cfg):
    s = Session(rng, session_id, cfg, family=FAMILY, n_faults=rng.choice((1, 2)))
    w = s.world
    host = rng.choice(w.vocab.hosts)
    tmpl = rng.choice(w.vocab.path_templates)
    k = rng.randint(1, cfg.r_max)                   # recovers within policy
    w.add_service(host, "transient", fail_attempts=k, error_class=rng.choice(HTTP_ERRORS))
    for _ in range(rng.randint(0, 3)):              # neutral prefix
        s.novel_decoy()
    r = s.http("GET", host, tmpl)                   # ERROR, NONE (the initial failure)
    while r.failed:                                 # immediate exact retries
        r = s.http("GET", host, tmpl)               # IN_POLICY_RETRY ... RESOLVES_EPISODE
    for f in w.faults.values():                     # productive work afterwards
        s.read(f.path, block=f.block)               # NEW_FACT
        s.edit(f.path, fix(f))                      # PROGRESS
        s.run_tests(f.suite)                        # NONE / NEW_FACT
    while s.n < 12:
        s.novel_decoy()
    return s.finish()
```

The windows go RECOVERY while the episode is open, then for 4 events after the resolution, then PRODUCTIVE. Don't call `s.http("GET", host, tmpl)` again after it succeeded: that call is REDUNDANT.

## Testing a family

`tests/synth_helpers.py` provides `check_family(name, intended)`, which runs every §7 check over 50 seeds: determinism, label recompute, `validate_steps`, event validation and `validate_sequence`, no oracle fields, digests matching latent state, gate 100% (or `gate="drops"` for family 12), and the plurality of pooled scored labels. It returns the label mix. The individual assertions are also exported. Use `H.sessions(name)` for family-specific checks, as in `tests/test_family_incremental_repair.py` and `tests/test_family_exploration.py`.

## Interpretations made by the engine (reported to the adjudicator)

1. **Initial failing operation.** §1 says it "has effect NONE". Here, the episode rule gives it nothing, and rules 2–6 decide, so a run that newly reveals a location is NEW_FACT.
2. **"Immediately after" is strict** for opening an episode. The one-unrelated-operation tolerance applies only once the episode is open. This is what separates family 6 (a cosmetic edit between failing runs, so no episode, so STALLED) from family 5.
3. **Diagnosed repair** keeps an episode open. This extends §1, whose literal text closes an episode on any change to relevant state. It is needed for spec §6.1's `diagnosed_repair` cause and for family 4.
4. **REDUNDANT inspections** include a first read of content whose facts are all already in K: it "re-acquires a known fact".
5. **External and control redundancy.**
   - An identical successful external call repeated is REDUNDANT.
   - A non-attempt poll that sees the same status as the last poll is REDUNDANT.
   - A control operation is REDUNDANT when the information epoch hasn't changed.
6. **lint/build** are VALIDATE without a validation result. Their redundancy is behavioural.
7. **Seed and alignment.** `seed` is stamped by `generate_session`. Under truncation, `event_effects` is aligned by list position. `dropped_observations` includes propagated baseline drops.
