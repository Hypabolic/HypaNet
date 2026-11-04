"""Family 1, incremental productive repair (spec §7.1 item 1).

Core latent behaviour: a failing run reveals a fault's location (NEW_FACT),
reading the block reveals its code (NEW_FACT), the fix edit raises progress
(PROGRESS), and the next run reveals the next fault, until the suites pass.
Intended label: PRODUCTIVE (rule 5, Δp > ε).

Variation: language, 1-4 faults, one or several suites, optional cause clues,
edit vs whole-file write, selected-test vs full-suite verification, an
optional search-first locate, cosmetic tidy-ups, neutral or productive
prefixes, short suffixes.

Lookalike (7 vs 1): with >= 3 faults a session may take a single detour,
``fix(prev) / test / wrong edit / test / undo / test / fix / test``, which is
an 8-event MUTATE-SUCCESS / VALIDATE-ERROR alternation with exactly one
reversion (the undo) and net progress, so it stays PRODUCTIVE.
"""

from __future__ import annotations

import random

from ...oracle_config import OracleConfig
from ...schema import CanonicalEvent, OracleRecord
from ..engine import MAX_EVENTS, MIN_EVENTS, Session, cosmetic, fix, undo, wrong
from ..world import Fault

FAMILY = "incremental_repair"


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
        elif roll < 0.55:
            s.novel_decoy(kinds=("list",))
        else:
            s.novel_decoy(kinds=("read", "search"))


def _verify(s: Session, rng: random.Random, f: Fault, simple: bool):
    """Re-run after a fix. Returns the last run's result."""
    if not simple and rng.random() < 0.3:
        s.run_tests(f.suite, list(f.tests))  # targeted run of the fixed fault's tests (passes)
    return s.run_tests(f.suite)


def generate(rng: random.Random, session_id: str, cfg: OracleConfig) -> tuple[list[CanonicalEvent], OracleRecord]:
    long_form = rng.random() < 0.3
    n_faults = rng.choice((3, 4)) if long_form else rng.choice((1, 2, 2, 3, 3, 4))
    s = Session(
        rng, session_id, cfg, family=FAMILY, n_faults=n_faults,
        extra_clues=(1, 3) if long_form else ((0, 2) if n_faults <= 2 else (0, 1)), single_suite=rng.random() < 0.6,
    )
    w = s.world
    thoroughness = rng.choice((2, 3)) if long_form else rng.choice((0, 1, 1, 2))
    read_tests: set[str] = set()

    def spare() -> int:
        # Events left after reserving read + fix + test per open fault and a short suffix.
        return MAX_EVENTS - s.n - 3 * len(w.unfixed_faults()) - 4
    est_core = 1 + n_faults * (4 + thoroughness // 2)
    target = rng.randint(MIN_EVENTS, 44)
    prefix_len = max(0, min(9, target - est_core - 2))
    style = rng.choice(("neutral", "neutral", "productive"))
    s.meta.update(long_form=long_form, n_faults=n_faults, prefix=style, prefix_len=prefix_len, thoroughness=thoroughness)

    if style == "productive":
        # The issue names the failing area: search the first fault's symbol, read related config.
        f0 = next(iter(w.faults.values()))
        s.search(f0.path.split("/", 1)[0], f0.symbol)
        for c in s.unknown_clues(f0)[:1]:
            if c.kind == "read" and c.block != f0.block:
                s.inspect_clue(c)
        _neutral_prefix(s, rng, max(0, prefix_len - 2))
    else:
        _neutral_prefix(s, rng, prefix_len)

    suites = list(dict.fromkeys(f.suite for f in w.faults.values()))
    r = s.run_tests(suites[0])
    detour_used = False
    prev_simple = False
    while w.unfixed_faults():
        f = _fault_of(s, r.failing_test)
        if f is None or w.is_fixed(f.id):
            if r.failed:
                break  # failing for a reason outside the fault set: never re-run unchanged (accidental episode)
            # This suite is green: run the next suite that still has failures.
            nxt = next(x.suite for x in w.unfixed_faults())
            r = s.run_tests(nxt)
            prev_simple = False
            continue
        remaining_after = len(w.unfixed_faults()) - 1
        if (not detour_used and prev_simple and remaining_after >= 1 and n_faults >= 3 and spare() >= 4
                and rng.random() < 0.5):
            # A quick wrong attempt, noticed and undone: one intentional revert.
            detour_used = True
            s.edit(f.path, wrong(regress=rng.random() < 0.5), block=f.block)
            s.run_tests(f.suite)
            s.edit(f.path, undo())
            s.run_tests(f.suite)
            s.edit(f.path, fix(f))
            r = s.run_tests(f.suite)
            s.meta["detour"] = f.id
            prev_simple = True
            continue

        if not s.knows(f.code_fact) and spare() > -2:
            s.read(f.path, block=f.block, pad=rng.randint(0, 8))
        for c in s.unknown_clues(f):
            if c.fact.endswith((":loc", ":code")):
                continue
            if rng.random() < (0.8 if long_form else 0.6) and spare() > 0:
                s.inspect_clue(c)
        for _ in range(rng.randint(0, thoroughness)):
            if spare() <= 0:
                break
            roll = rng.random()
            test_path = f.tests[0].split("::", 1)[0]
            if roll < 0.3 and test_path not in read_tests:
                read_tests.add(test_path)
                s.read(test_path)  # the failing test itself (novel, no clue)
            elif roll < 0.5:
                s.plan(rng.choice(w.vocab.plan_actions))
            else:
                s.novel_decoy()
        simple = rng.random() < 0.6
        if rng.random() < 0.2:
            s.write(f.path, fix(f))
        else:
            s.edit(f.path, fix(f))
        if not simple and rng.random() < 0.3 and spare() > 0:
            s.edit(f.path, cosmetic(rng.choice(("format", "rename_local", "comment"))))
        r = _verify(s, rng, f, simple or spare() <= 0)
        prev_simple = simple

    # Suffix: confirm other suites, lint/build, wrap up.
    for suite in w.suites:
        if suite not in suites and rng.random() < 0.3 and s.room(3):
            s.run_tests(suite)
    if rng.random() < 0.35 and s.room(2):
        if rng.random() < 0.5:
            s.lint(rng.choice(w.lint_targets))
        else:
            s.build(rng.choice(w.build_targets))
    while s.n < MIN_EVENTS:
        s.novel_decoy()
    if rng.random() < 0.5 and s.room(1):
        s.yield_("done")
    return s.finish()
