"""Family 9, improving tests: a benign lookalike (spec §7.1 item 9).

Core latent behaviour: a tight edit-and-test loop on one suite. Every run of
the suite comes after a behaviour-changing edit, and every edit before a run
is a real fix of the fault whose location the previous run revealed, so the
pass count climbs on each run and latent progress rises with each edit
(PROGRESS). Each failing run reveals the next fault's location (NEW_FACT).
Intended label: PRODUCTIVE (rule 5, Δp > ε).

Variation: language, 2-4 faults (usually all in one suite, sometimes one
suite after another), tests per fault, edit vs whole-file write, a first
attempt that changes behaviour but doesn't fix (adding a debug assertion or
logging, or a plausible wrong patch) before the loop, optional reads of the
failing block or test, targeted runs of the just-fixed tests, cause-clue
inspections, neutral or investigative prefixes, and verification suffixes.

Lookalike (6 vs 9): the default loop is the MUTATE-SUCCESS / VALIDATE-ERROR
alternation that family 6 (cosmetic churn) emits. Here each edit fixes a fault
and each run newly reveals a location, so the 8-event windows are PRODUCTIVE
where family 6's are STALLED. With 4 faults the loop yields 8 consecutive
events in both orders: ``first attempt, test, fix, test, fix, test, fix, test``
(MUTATE first) and ``test, fix, test, fix, test, fix, test, fix`` (VALIDATE
first; the run after the last fix passes).
"""

from __future__ import annotations

import random

from ...oracle_config import OracleConfig
from ...schema import CanonicalEvent, OracleRecord
from ..engine import MAX_EVENTS, MIN_EVENTS, Session, fix, neutral, wrong
from ..world import Fault

FAMILY = "improving_tests"


def _fault_of(s: Session, test_id: str | None) -> Fault | None:
    if test_id is None:
        return None
    fids = s.world.tests[test_id].faults
    return s.world.faults[fids[0]] if fids else None


def _neutral_prefix(s: Session, rng: random.Random, k: int) -> None:
    for i in range(k):
        roll = rng.random()
        if i == 0 and roll < 0.35:
            s.plan(rng.choice(s.world.vocab.plan_actions))
        elif roll < 0.5:
            s.novel_decoy(kinds=("list",))
        else:
            s.novel_decoy(kinds=("read", "search"))


def generate(rng: random.Random, session_id: str, cfg: OracleConfig) -> tuple[list[CanonicalEvent], OracleRecord]:
    tight = rng.random() < 0.45  # strict edit/test alternation, no reads in the loop
    n_faults = rng.choice((3, 4, 4)) if tight else rng.choice((2, 3, 3, 4, 4))
    long_form = not tight and rng.random() < 0.4
    if long_form:
        n_faults = rng.choice((3, 4, 4))
    s = Session(rng, session_id, cfg, family=FAMILY, n_faults=n_faults,
                extra_clues=(2, 4) if long_form else (0, 1), single_suite=rng.random() < 0.8,
                tests_per_fault=rng.choice(((2, 3), (2, 4), (3, 5))))
    w = s.world
    first_attempt = rng.choices(("none", "probe", "wrong"), weights=(40, 35, 25))[0]
    read_rate = 0.0 if tight else (0.9 if long_form else rng.choice((0.3, 0.6, 0.9)))
    targeted_rate = 0.0 if tight else rng.choice((0.0, 0.25, 0.5))
    clue_rate = 0.0 if tight else (0.9 if long_form else 0.4)
    use_write = rng.random() < 0.2
    prefix = rng.choice(("neutral", "neutral", "investigate"))
    prefix_len = rng.randint(6, 16) if long_form else rng.randint(0, 9)
    s.meta.update(tight=tight, n_faults=n_faults, long_form=long_form, first_attempt=first_attempt,
                  read_rate=read_rate, targeted_rate=targeted_rate, prefix=prefix, prefix_len=prefix_len)

    def spare() -> int:
        # Events left after reserving fix + test per open fault and a short suffix.
        return MAX_EVENTS - s.n - 2 * len(w.unfixed_faults()) - 4

    # Prefix.
    suites = list(dict.fromkeys(f.suite for f in w.faults.values()))
    if prefix == "investigate":
        f0 = next(iter(w.faults.values()))
        s.search(f0.path.split("/", 1)[0], f0.symbol)
        s.read(f0.path, block=f0.block, pad=rng.randint(0, 6))
        _neutral_prefix(s, rng, max(0, prefix_len - 2))
    else:
        _neutral_prefix(s, rng, prefix_len)

    # Optional first attempt: a behaviour-changing edit that doesn't fix (NONE), right before the first run.
    if first_attempt != "none":
        f0 = next(iter(w.faults.values()))
        if first_attempt == "probe":
            s.edit(f0.path, neutral(), block=w.files[f0.path].blocks[0])  # add a debug assertion / logging
        else:
            s.edit(f0.path, wrong(regress=False), block=f0.block)  # a plausible first patch that doesn't help

    s.meta["core_start"] = s.n
    read_tests: list[str] = []
    r = s.run_tests(suites[0])
    while w.unfixed_faults():
        f = _fault_of(s, r.failing_test)
        if f is None or w.is_fixed(f.id):
            if r.failed:
                break  # failing for a reason outside the fault set: never re-run unchanged
            nxt = next(x.suite for x in w.unfixed_faults())
            r = s.run_tests(nxt)  # this suite is green: move to the next one
            continue
        if not s.knows(f.code_fact) and rng.random() < read_rate and spare() > 0:
            s.read(f.path, block=f.block, pad=rng.randint(0, 8))
        for c in s.unknown_clues(f):
            if c.fact.endswith((":loc", ":code")):
                continue
            if rng.random() < clue_rate and spare() > 0:
                s.inspect_clue(c)
        if not tight and rng.random() < 0.15 and spare() > 0:
            test_path = r.failing_test.split("::", 1)[0] if r.failing_test else None
            if test_path in w.files and test_path not in read_tests:
                read_tests.append(test_path)
                s.read(test_path)  # the failing test itself
        if use_write:
            s.write(f.path, fix(f))
        else:
            s.edit(f.path, fix(f))
        if rng.random() < targeted_rate and spare() > 0:
            s.run_tests(f.suite, list(f.tests))  # the just-fixed tests now pass
        r = s.run_tests(f.suite)
    s.meta["core_end"] = s.n

    # Suffix: confirm other suites, lint/build, wrap up (fuller in the long form).
    for suite in w.suites:
        if suite not in suites and rng.random() < (0.8 if long_form else 0.3) and s.room(3):
            s.run_tests(suite)
    checks = [("lint", t) for t in w.lint_targets] + [("build", t) for t in w.build_targets]
    rng.shuffle(checks)
    n_checks = rng.randint(1, 3) if long_form else (1 if rng.random() < 0.35 else 0)
    for kind, tgt in checks[:n_checks]:
        if not s.room(2):
            break
        if kind == "lint":
            s.lint(tgt)
        else:
            s.build(tgt)
    if long_form and rng.random() < 0.5 and s.room(3):
        s.novel_decoy(kinds=("read",))  # e.g. the changelog or contributing notes
    while s.n < MIN_EVENTS:
        s.novel_decoy()
    if rng.random() < 0.5 and s.room(1):
        s.yield_("done")
    return s.finish()
