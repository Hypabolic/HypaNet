"""Family 6, cosmetic churn (spec §7.1 item 6, with the r2 reading in §0.2).

Core latent behaviour: edits that change file text but not behaviour (the
behaviour they aim at is already present in ``W_t``): reformatting, renaming
locals, comments, whitespace, docstrings, or re-adding a guard that is already
there (``fix`` on an already-fixed block). Between them the agent re-runs
validation on a workspace whose behaviour has not changed since the last run
of the same suite and selection. Under the r2 reading the edits and the
re-runs are all REDUNDANT, so once the facts revealed at the start have left
the window, the windows are STALLED (rule 3).

An edit always sits between two re-runs of the same validation, so no retry
episode ever opens (an episode opens only on the *immediately* following
identical operation, M1_ENGINE interpretation 2); nothing is ever undone, so
no workspace state is revisited (no reversions).

Shapes:
- ``failing`` (most sessions): a fault stays unfixed and the re-runs keep
  failing (MUTATE-SUCCESS / VALIDATE-ERROR alternation).
- ``green``: the faults were fixed first; the agent keeps polishing and
  re-running a green suite (MUTATE-SUCCESS / VALIDATE-SUCCESS).

Variation (§7.2): language and world; 1-3 faults; neutral or productive
prefix (fixing another fault first) of varying length; cosmetic kind mix;
edit vs whole-file write; the edited files (fault file, other sources, the
test file); full-suite vs selected re-runs, optional lint/build re-checks;
edits per re-run; occasional redundant re-reads and repeated plans; churn
length; optional suffix (the real fix at last, or a hand-off).

Lookalike (6 vs 9): the 8-event (MUTATE,SUCCESS),(VALIDATE,ERROR) x 4
alternation, here STALLED (in family 9 the edits are fixes and counts improve).
"""

from __future__ import annotations

import random

from ...oracle_config import OracleConfig
from ...schema import CanonicalEvent, OracleRecord
from ..engine import COSMETIC_KINDS, MAX_EVENTS, MIN_EVENTS, Change, Session, cosmetic, fix
from ..world import Fault

FAMILY = "cosmetic_churn"


def _fault_of(s: Session, test_id: str | None) -> Fault | None:
    if test_id is None:
        return None
    fids = s.world.tests[test_id].faults
    return s.world.faults[fids[0]] if fids else None


def _neutral(s: Session, rng: random.Random, k: int) -> None:
    for i in range(k):
        roll = rng.random()
        if i == 0 and roll < 0.3:
            s.plan(rng.choice(s.world.vocab.plan_actions))
        elif roll < 0.35:
            s.novel_decoy(kinds=("list",))
        else:
            s.novel_decoy(kinds=("read", "search"))


def _repair(s: Session, rng: random.Random, f: Fault) -> None:
    """Productive repair of one fault: read the block, fix it."""
    if not s.knows(f.code_fact):
        s.read(f.path, block=f.block, pad=rng.randint(0, 6))
    if rng.random() < 0.2:
        s.write(f.path, fix(f))
    else:
        s.edit(f.path, fix(f))


def generate(rng: random.Random, session_id: str, cfg: OracleConfig) -> tuple[list[CanonicalEvent], OracleRecord]:
    shape = "green" if rng.random() < 0.15 else "failing"
    n_faults = rng.choice((1, 2)) if shape == "green" else rng.choice((1, 2, 2, 3))
    s = Session(rng, session_id, cfg, family=FAMILY, n_faults=n_faults, extra_clues=(0, 1),
                single_suite=rng.random() < 0.6)
    w = s.world
    prefix = rng.choice(("neutral", "neutral", "productive")) if shape == "failing" else "productive"
    if prefix == "productive" and shape == "failing" and n_faults == 1:
        prefix = "neutral"
    prefix_len = rng.randint(0, 8)
    target_len = rng.randint(MIN_EVENTS + 6, MAX_EVENTS)
    kinds = rng.sample(COSMETIC_KINDS, rng.randint(2, len(COSMETIC_KINDS)))
    write_rate = rng.choice((0.0, 0.1, 0.3))
    double_edit = rng.choice((0.0, 0.15, 0.4))
    inspect_rate = rng.choice((0.0, 0.1, 0.25))
    suffix = rng.choice(("none", "none", "yield", "real_fix")) if shape == "failing" else rng.choice(("none", "yield"))
    s.meta.update(shape=shape, n_faults=n_faults, prefix=prefix, prefix_len=prefix_len, kinds=kinds,
                  write_rate=write_rate, double_edit=double_edit, inspect_rate=inspect_rate, suffix=suffix)

    _neutral(s, rng, prefix_len)
    suites = list(dict.fromkeys(f.suite for f in w.faults.values()))
    suite = suites[0]

    # -- reveal: the failure the agent is (not) addressing -------------------
    r = s.run_tests(suite)
    if prefix == "productive":
        # Fix whatever the runs reveal, one fault at a time; in the failing
        # shape, stop with one fault still open.
        keep_open = 1 if shape == "failing" else 0
        while len(w.unfixed_faults()) > keep_open:
            f = _fault_of(s, r.failing_test)
            if f is None or w.is_fixed(f.id):
                nxt = next(x.suite for x in w.unfixed_faults())
                if r.failed or nxt == suite:
                    break
                suite = nxt
                r = s.run_tests(suite)
                continue
            _repair(s, rng, f)
            r = s.run_tests(suite)
        if shape == "failing" and r.ok:
            suite = next(x.suite for x in w.unfixed_faults())
            r = s.run_tests(suite)
    stuck = _fault_of(s, r.failing_test) if r.failed else None
    if stuck is not None and not s.knows(stuck.code_fact) and rng.random() < 0.6:
        s.read(stuck.path, block=stuck.block, pad=rng.randint(0, 6))

    # Which validation the agent keeps re-running (it has already run each once).
    selection = None
    if r.failed and rng.random() < 0.3:
        selection = [r.failing_test] if rng.random() < 0.5 else list(stuck.tests)  # type: ignore[union-attr]
        s.run_tests(suite, selection)
    validations = [lambda: s.run_tests(suite, selection)]
    if rng.random() < 0.25:
        tgt = rng.choice(w.lint_targets) if rng.random() < 0.5 else None
        btgt = None if tgt else rng.choice(w.build_targets)
        if tgt:
            s.lint(tgt)
            validations.append(lambda: s.lint(tgt))
        else:
            s.build(btgt)  # type: ignore[arg-type]
            validations.append(lambda: s.build(btgt))  # type: ignore[arg-type]
    if not w.unfixed_faults():
        stuck = None

    # Files the agent polishes. Prefer the code it is looking at.
    sources = [p for p, fm in w.files.items() if fm.role == "source"]
    focus = stuck.path if stuck is not None else rng.choice([f.path for f in w.faults.values()])
    test_file = (r.failing_test or rng.choice(w.suites[suite])).split("::", 1)[0]
    others = [p for p in sources if p != focus]
    fixed = [f for f in w.faults.values() if w.is_fixed(f.id)]
    readbacks = [(focus, stuck.block)] if stuck is not None and s.knows(stuck.code_fact) else []
    readbacks += [(f.path, f.block) for f in fixed]
    plan_action = rng.choice(w.vocab.plan_actions)

    def churn_edit() -> None:
        roll = rng.random()
        if fixed and roll < 0.2:
            f = rng.choice(fixed)
            change: Change = fix(f)  # re-add the guard that is already there
            path, block = f.path, None
        else:
            if roll < 0.7 or not others:
                path = focus
            elif roll < 0.85 or test_file not in w.files:
                path = rng.choice(others)
            else:
                path = test_file
            change = cosmetic(rng.choice(kinds))
            blocks = w.files[path].blocks
            block = stuck.block if (stuck is not None and path == stuck.path and rng.random() < 0.6) else rng.choice(blocks)
        if rng.random() < write_rate:
            s.write(path, change, block=block)
        else:
            s.edit(path, change, block=block)

    # -- churn: text-only edits and re-runs on unchanged behaviour -----------
    suffix_room = {"none": 0, "yield": 1, "real_fix": 3}[suffix]
    planned = False
    # One iteration emits up to 4 events: edit, optional second edit, optional read/plan, validation.
    while s.room(4 + suffix_room) and s.n < target_len - suffix_room:
        churn_edit()
        if rng.random() < double_edit:
            churn_edit()
        if readbacks and rng.random() < inspect_rate:
            p, b = rng.choice(readbacks)
            s.read(p, block=b)
        elif rng.random() < inspect_rate / 2:
            s.plan(plan_action)  # first time NONE, later REDUNDANT while nothing new is learnt
            planned = True
        v = validations[0] if len(validations) == 1 or rng.random() < 0.75 else rng.choice(validations[1:])
        v()
    s.meta["planned"] = planned

    # -- suffix ----------------------------------------------------------------
    if suffix == "real_fix" and stuck is not None and s.room(3):
        _repair(s, rng, stuck)
        s.run_tests(suite)
    elif suffix == "yield" or suffix == "real_fix":
        s.yield_(rng.choice(("done", "handoff", "partial_result")))
    while s.n < MIN_EVENTS:
        s.novel_decoy()
    return s.finish()
