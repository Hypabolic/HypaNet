"""Family 8, tool oscillation (spec §7.1 item 8).

Core latent behaviour: two operations repeatedly undo each other's effect on
one file. A text-only "forward" operation (a formatter pass, a local rename, a
comment or docstring pass; each REDUNDANT under the r2 reading) is followed by
an operation that takes the file back (``revert``, an ``undo`` patch, or a
whole-file rewrite of the previous content). Re-applying the forward operation
recreates the earlier text exactly, so from the second step on every core
mutation returns the workspace to a state held a few events earlier
(REVERSION). Behaviour never changes, so latent progress is flat across the
core and 8-event windows hold >= 2 reversions. Intended label: OSCILLATING
(rule 2).

Variation: language, 1-4 faults, file (source, config or test file), pair
type (format/revert, format/undo, formatter write/undo, rename/undo,
comment/revert, docstring/rewrite, format/revert-to-initial), what runs
between the two halves (nothing, lint, build, the test suite, a re-read of
the file), cycle count, neutral or productive prefixes (fixing faults first,
so later test runs pass), and a suffix that breaks out with real work, gives
up, or simply ends.
"""

from __future__ import annotations

import random

from ...oracle_config import OracleConfig
from ...schema import CanonicalEvent, OracleRecord
from ..engine import MAX_EVENTS, MIN_EVENTS, Change, Session, cosmetic, fix, undo
from ..world import Fault

FAMILY = "tool_oscillation"

# (forward op, backward op). Forward: (family, cosmetic kind). Backward: how the file is taken back.
PAIRS: tuple[tuple[str, str, str], ...] = (
    ("edit", "format", "revert"),
    ("edit", "format", "undo"),
    ("write", "format", "undo"),
    ("edit", "format", "rewrite"),
    ("edit", "rename_local", "undo"),
    ("edit", "comment", "revert"),
    ("edit", "docstring", "rewrite"),
    ("edit", "format", "revert_initial"),
)
_PAIR_WEIGHTS = (24, 14, 10, 8, 14, 10, 8, 12)

_BETWEEN = ("none", "lint", "build", "tests", "read", "mixed")
_BETWEEN_WEIGHTS = (30, 18, 10, 22, 8, 12)


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
    """Productive prefix: locate, read and fix one fault, sometimes verify."""
    if not s.knows(f.loc_fact):
        if rng.random() < 0.5:
            s.run_tests(f.suite, list(f.tests))
        else:
            s.search(f.path.split("/", 1)[0], f.symbol)
    if not s.knows(f.code_fact):
        s.read(f.path, block=f.block, pad=rng.randint(0, 6))
    s.edit(f.path, fix(f))
    if rng.random() < 0.5:
        s.run_tests(f.suite)


def _pick_file(s: Session, rng: random.Random, pair: tuple[str, str, str]) -> tuple[str, str]:
    """Choose the oscillating file and block: never formatted already, and for
    ``revert_initial`` never touched before (so the revert can't undo a fix)."""
    w = s.world
    touched = {f.path for f in w.faults.values() if w.is_fixed(f.id)}
    kind = pair[1]
    roles = ("source", "source", "source", "config", "test") if kind in ("format", "rename_local") else ("source",)
    role = rng.choice(roles)
    cands = [p for p, fm in w.files.items()
             if fm.role == role and not any(st.formatted for st in fm.states.values())
             and (pair[2] != "revert_initial" or p not in touched)]
    if kind in ("rename_local", "docstring", "comment"):
        cands = [p for p in cands if w.files[p].lang is not None]
    if not cands:
        cands = [p for p, fm in w.files.items() if fm.lang is not None and fm.role == "source"
                 and not any(st.formatted for st in fm.states.values())
                 and (pair[2] != "revert_initial" or p not in touched)]
    # Prefer a file the agent has just been working in.
    hot = [p for p in cands if p in touched]
    path = rng.choice(hot) if hot and rng.random() < 0.5 else rng.choice(sorted(cands))
    blocks = w.files[path].blocks
    return path, blocks[rng.randrange(len(blocks))]


def _forward(s: Session, pair: tuple[str, str, str], path: str, block: str) -> None:
    family, kind, _ = pair
    ch: Change = cosmetic(kind)
    if family == "write":
        s.write(path, ch, block=block)
    else:
        s.edit(path, ch, block=block)


def _backward(s: Session, pair: tuple[str, str, str], path: str) -> None:
    back = pair[2]
    if back == "revert":
        s.revert(path)
    elif back == "revert_initial":
        s.revert(path, to="initial")
    elif back == "undo":
        s.edit(path, undo())
    elif back == "rewrite":
        s.write(path, undo())  # whole-file rewrite with the previous content
    else:
        raise ValueError(back)


def _between(s: Session, rng: random.Random, mode: str, path: str, suite: str, lint_t: str, build_t: str) -> None:
    if mode == "mixed":
        mode = rng.choice(("none", "lint", "tests", "read"))
    if mode == "none":
        return
    if mode == "lint":
        s.lint(lint_t)
    elif mode == "build":
        s.build(build_t)
    elif mode == "tests":
        s.run_tests(suite)
    elif mode == "read":
        s.read(path)
    else:
        raise ValueError(mode)


def generate(rng: random.Random, session_id: str, cfg: OracleConfig) -> tuple[list[CanonicalEvent], OracleRecord]:
    n_faults = rng.choice((1, 1, 2, 2, 3, 4))
    s = Session(rng, session_id, cfg, family=FAMILY, n_faults=n_faults, extra_clues=(0, 2),
                single_suite=rng.random() < 0.5)
    w = s.world
    faults = list(w.faults.values())
    pair = rng.choices(PAIRS, weights=_PAIR_WEIGHTS)[0]
    between = rng.choices(_BETWEEN, weights=_BETWEEN_WEIGHTS)[0]
    between_both = rng.random() < 0.6  # validate after both halves, or only after the forward op
    prefix = rng.choice(("neutral", "neutral", "productive", "productive"))
    suffix = rng.choices(("breakout", "giveup", "none"), weights=(30, 25, 45))[0]
    target_len = rng.randint(MIN_EVENTS, MAX_EVENTS - 4)

    if prefix == "productive":
        for f in faults[: rng.randint(1, len(faults))]:
            _repair(s, rng, f)
        _neutral_prefix(s, rng, rng.randint(0, 2))
    else:
        _neutral_prefix(s, rng, rng.randint(0, 8))

    path, block = _pick_file(s, rng, pair)
    suite = rng.choice(list(dict.fromkeys(f.suite for f in faults)))
    lint_t, build_t = rng.choice(w.lint_targets), rng.choice(w.build_targets)
    if rng.random() < 0.4:
        s.read(path)  # look at the file first (novel or fact-bearing)
    s.meta.update(n_faults=n_faults, pair="/".join(pair), between=between, between_both=between_both,
                  prefix=prefix, suffix=suffix, path=path)

    per_half = 1 if between == "none" else 2
    per_cycle = per_half * (2 if between_both else 1) + (0 if between_both else 1)
    reserve = {"breakout": 4, "giveup": 2, "none": 0}[suffix]
    min_cycles = 3
    n_cycles = max(min_cycles, (target_len - s.n - reserve) // per_cycle)
    s.meta["core_start"] = s.n
    for k in range(n_cycles):
        _forward(s, pair, path, block)
        _between(s, rng, between, path, suite, lint_t, build_t)
        _backward(s, pair, path)
        if between_both:
            _between(s, rng, between, path, suite, lint_t, build_t)
        if k + 1 >= min_cycles and not s.room(reserve + per_cycle):
            break
    s.meta["core_end"] = s.n

    # Suffix.
    open_faults = w.unfixed_faults()
    if suffix == "breakout" and open_faults and s.room(3):
        f = open_faults[0]
        if not s.knows(f.code_fact):
            s.read(f.path, block=f.block, pad=rng.randint(0, 4))
        s.edit(f.path, fix(f))
        if s.room(1):
            s.run_tests(f.suite)
    elif suffix in ("breakout", "giveup") and s.room(1):
        s.yield_(rng.choice(("blocked", "partial_result", "handoff", "done")))
    while s.n < MIN_EVENTS:
        s.novel_decoy()
    return s.finish()
