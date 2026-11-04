"""Family 7, A/B/A/B workspace oscillation (spec §7.1 item 7).

Core latent behaviour: on one open fault the agent alternates between two (or,
sometimes, three) competing alternatives of the faulty block, usually running
the tests after each edit. Competing alternatives never fix the fault and cause
no regression, so latent progress stays flat across the core. From the third
edit on, each edit returns the workspace to a state it held a few events
earlier (REVERSION), so 8-event windows over the core hold >= 2 reversions
with Δp = 0. Intended label: OSCILLATING (rule 2).

Variation: language, 1-4 faults, which fault oscillates, 2- or 3-way
alternation, edit vs whole-file write, validation style (full suite, the
fault's tests, a targeted run of the previously failing test, lint/build, or
none), occasional re-reads of the block during the core, cycle count,
neutral / diagnostic / productive prefixes (fixing other faults first), and a
suffix that breaks out (undo + real fix), gives up (yield/ask) or simply ends.

Lookalike (7 vs 1): the default core is ``competing A / test / competing B /
test`` x >= 2 with failing runs, i.e. the 8-event MUTATE-SUCCESS /
VALIDATE-ERROR alternation that family 1 emits with a single intentional
revert, but here with >= 2 reversions and no progress, so OSCILLATING.
"""

from __future__ import annotations

import random

from ...oracle_config import OracleConfig
from ...schema import CanonicalEvent, OracleRecord
from ..engine import MAX_EVENTS, MIN_EVENTS, Session, competing, fix
from ..world import Fault

FAMILY = "abab_oscillation"

_LABEL_SETS = (("A", "B"), ("A", "B", "C"))


def _neutral_prefix(s: Session, rng: random.Random, k: int) -> None:
    for i in range(k):
        roll = rng.random()
        if i == 0 and roll < 0.35:
            s.plan(rng.choice(s.world.vocab.plan_actions))
        elif roll < 0.5:
            s.novel_decoy(kinds=("list",))
        else:
            s.novel_decoy(kinds=("read", "search"))


def _repair(s: Session, rng: random.Random, f: Fault) -> None:
    """Productive prefix work on a fault other than the oscillating one."""
    if not s.knows(f.loc_fact):
        s.search(f.path.split("/", 1)[0], f.symbol)
    if not s.knows(f.code_fact):
        s.read(f.path, block=f.block, pad=rng.randint(0, 6))
    if rng.random() < 0.2:
        s.write(f.path, fix(f))
    else:
        s.edit(f.path, fix(f))
    if rng.random() < 0.6:
        s.run_tests(f.suite, list(f.tests))


def _validate(s: Session, rng: random.Random, f: Fault, style: str, last_failing: list[str]) -> None:
    """One validation step after a competing edit, per the session's style.

    ``last_failing`` holds the most recent failing test of the oscillating
    fault seen by the agent (observable), for targeted re-runs.
    """
    if style == "mixed":
        style = rng.choice(("suite", "suite", "selection", "lint"))
    if style == "none":
        return
    if style == "suite":
        r = s.run_tests(f.suite)
    elif style == "selection":
        r = s.run_tests(f.suite, list(f.tests))
    elif style == "targeted":
        # Run the test that failed last time (now passing), then the fault's tests (failing elsewhere).
        if last_failing and last_failing[0] in f.tests:
            s.run_tests(f.suite, [last_failing[0]])
        r = s.run_tests(f.suite, list(f.tests))
    elif style == "lint":
        s.lint(s.meta["lint_target"])
        return
    elif style == "build":
        s.build(s.meta["build_target"])
        return
    else:
        raise ValueError(style)
    if r.failing_test is not None:
        last_failing[:] = [r.failing_test]


def generate(rng: random.Random, session_id: str, cfg: OracleConfig) -> tuple[list[CanonicalEvent], OracleRecord]:
    n_faults = rng.choice((1, 1, 2, 2, 3, 4))
    s = Session(rng, session_id, cfg, family=FAMILY, n_faults=n_faults, extra_clues=(0, 2),
                single_suite=rng.random() < 0.5, tests_per_fault=rng.choice(((2, 3), (2, 4), (3, 4))))
    w = s.world
    faults = list(w.faults.values())
    target = faults[rng.randrange(len(faults))]
    others = [f for f in faults if f.id != target.id]

    labels = _LABEL_SETS[0] if rng.random() < 0.85 else _LABEL_SETS[1]
    style = rng.choices(("suite", "selection", "targeted", "mixed", "lint", "build", "none"),
                        weights=(40, 18, 10, 12, 6, 5, 9))[0]
    use_write = rng.random() < 0.2  # whole-file writes instead of patches
    reread = rng.choice((0.0, 0.0, 0.15, 0.3))
    prefix = rng.choice(("neutral", "neutral", "diagnostic", "productive")) if others else rng.choice(("neutral", "diagnostic"))
    suffix = rng.choices(("breakout", "giveup", "none"), weights=(35, 30, 35))[0]
    target_len = rng.randint(MIN_EVENTS, MAX_EVENTS - 4)
    s.meta.update(n_faults=n_faults, target=target.id, labels="".join(labels), style=style, use_write=use_write,
                  reread=reread, prefix=prefix, suffix=suffix, lint_target=rng.choice(w.lint_targets),
                  build_target=rng.choice(w.build_targets))

    # Prefix: put the oscillation somewhere other than the session start.
    if prefix == "productive":
        for f in others[: rng.randint(1, len(others))]:
            _repair(s, rng, f)
        _neutral_prefix(s, rng, rng.randint(0, 2))
    elif prefix == "diagnostic":
        _neutral_prefix(s, rng, rng.randint(0, 3))
    else:
        _neutral_prefix(s, rng, rng.randint(0, 8))
    if prefix != "neutral" or rng.random() < 0.6:
        # Locate and read the oscillating fault before trying alternatives.
        if rng.random() < 0.6:
            s.run_tests(target.suite, list(target.tests))
        else:
            s.search(".", target.symbol)
        s.read(target.path, block=target.block, pad=rng.randint(0, 6))

    # Core: alternate competing alternatives, validating after each.
    per_cycle = {"none": 1, "targeted": 3}.get(style, 2)
    reserve = {"breakout": 4, "giveup": 2, "none": 0}[suffix]
    min_edits = 2 * len(labels)  # enough for >= 2 reversions inside one window
    n_edits = max(min_edits, (target_len - s.n - reserve) // per_cycle)
    s.meta["core_start"] = s.n
    last_failing: list[str] = []
    for k in range(n_edits):
        change = competing(target, labels[k % len(labels)])
        if use_write:
            s.write(target.path, change)
        else:
            s.edit(target.path, change)
        _validate(s, rng, target, style, last_failing)
        if k + 1 >= min_edits and not s.room(reserve + per_cycle + 1):
            break
        if rng.random() < reread and s.room(reserve + 2 * per_cycle + 1):
            s.read(target.path, block=target.block, pad=rng.randint(0, 4))  # code fact known: REDUNDANT
    s.meta["core_end"] = s.n

    # Suffix.
    if suffix == "breakout" and s.room(2):
        # Step back to the original code (sometimes), then apply the real fix.
        same_file_touched = any(f.path == target.path and w.is_fixed(f.id) for f in others)
        if not same_file_touched and rng.random() < 0.4 and s.room(3):
            s.revert(target.path, to="initial")
        s.edit(target.path, fix(target))
        if s.room(1):
            s.run_tests(target.suite)
    elif suffix == "giveup" and s.room(1):
        if rng.random() < 0.5:
            s.ask(rng.choice(("clarify_requirement", "confirm_scope")))
        if s.room(1):
            s.yield_(rng.choice(("blocked", "partial_result", "handoff")))
    while s.n < MIN_EVENTS:
        s.novel_decoy()
    return s.finish()
