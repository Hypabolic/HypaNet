"""Family 5, repeated failed test invocation (spec §7.1 item 5).

Core latent behaviour: a test run fails deterministically and the agent keeps
re-running the same suite and selection on a behaviourally unchanged
workspace, seeing the same failure each time, with re-reads of content it has
already read in between. Nothing is ever edited (the contract notes family 5
has no edits by definition), so the failure cannot change.

Latent effects (decided by the engine, M1_CONTRACTS §1):
- the initial failing run is NONE, or NEW_FACT if it newly reveals the
  failing location;
- the immediate re-run opens a retry episode; attempts ``1..R_max`` are
  IN_POLICY_RETRY, so windows ending there are RECOVERY (rule 4);
- attempts beyond ``R_max`` are REDUNDANT with the episode still open but out
  of policy, and interleaved re-reads of known content are REDUNDANT, so once
  the facts have left the window the windows are STALLED (rule 3).

Two tail shapes: ``continuous`` keeps at most one interleaved operation
between re-runs (the episode stays open, attempts keep counting), and
``spaced`` breaks the episode with two interleaved operations and from then
on always keeps at least one operation between re-runs (no new episode can
open; every re-run is REDUNDANT because the behaviour is unchanged).

Variation (§7.2): language, paths, suites and error classes from the world;
1-3 faults; full-suite vs selected-test re-runs; neutral, investigative or
baseline-run prefixes of varying length (so the core sits at different
positions); interleave density and kinds (re-read of the fault block, of a
test or decoy file, re-listing, re-searching the symbol, a repeated plan, a
re-run of a passing suite); doomed-tail length; optional give-up suffix.

Lookalike (5 vs 4): runs of consecutive VALIDATE-ERROR re-runs, and
INSPECT-SUCCESS / VALIDATE-ERROR alternations, past ``R_max`` (STALLED).
"""

from __future__ import annotations

import random
from typing import Callable

from ...oracle_config import OracleConfig
from ...schema import CanonicalEvent, OracleRecord
from ..engine import MAX_EVENTS, MIN_EVENTS, OpResult, Session
from ..world import Fault

FAMILY = "repeated_test_failure"


def _fault_of(s: Session, test_id: str) -> Fault:
    fids = s.world.tests[test_id].faults
    return s.world.faults[fids[0]]


class _Repeatables:
    """Operations already performed once; repeating them re-acquires known
    content (REDUNDANT per the engine's latent rules)."""

    def __init__(self) -> None:
        self.ops: list[tuple[str, Callable[[], OpResult]]] = []
        self._names: set[str] = set()

    def add(self, name: str, op: Callable[[], OpResult]) -> None:
        if name not in self._names:
            self._names.add(name)
            self.ops.append((name, op))

    def pick(self, rng: random.Random) -> Callable[[], OpResult] | None:
        if not self.ops:
            return None
        return self.ops[rng.randrange(len(self.ops))][1]


def generate(rng: random.Random, session_id: str, cfg: OracleConfig) -> tuple[list[CanonicalEvent], OracleRecord]:
    n_faults = rng.choice((1, 1, 2, 2, 3))
    s = Session(rng, session_id, cfg, family=FAMILY, n_faults=n_faults, extra_clues=(0, 2),
                single_suite=rng.random() < 0.5)
    w = s.world
    rep = _Repeatables()

    # The suite the agent gets stuck on: one that holds a fault (it always fails, nothing is edited).
    stuck_suite = rng.choice(list(dict.fromkeys(f.suite for f in w.faults.values())))
    passing_suites = [x for x in w.suites if x not in {f.suite for f in w.faults.values()}]

    prefix_style = rng.choice(("neutral", "investigate", "baseline"))
    prefix_len = rng.randint(5, 16)
    tail_shape = rng.choice(("continuous", "continuous", "spaced"))
    interleave_in = rng.choice((0.0, 0.3, 0.6))
    interleave_tail = rng.choice((0.0, 0.25, 0.5, 0.8)) if tail_shape == "continuous" else 1.0
    target_len = rng.randint(max(MIN_EVENTS + 8, prefix_len + 16), MAX_EVENTS)
    suffix = rng.choice(("none", "none", "yield", "ask", "insight"))
    s.meta.update(n_faults=n_faults, prefix=prefix_style, prefix_len=prefix_len, tail=tail_shape,
                  interleave_in=interleave_in, interleave_tail=interleave_tail, suffix=suffix)

    def read_decoy() -> None:
        paths = [p for p in w.decoy_paths() if not p.startswith(("logs/",))] or w.decoy_paths()
        p = rng.choice(paths)
        s.read(p)
        rep.add(f"read:{p}", lambda p=p: s.read(p))

    def list_dir() -> None:
        d = rng.choice(w.dirs())
        s.list_dir(d)
        rep.add(f"ls:{d}", lambda d=d: s.list_dir(d))

    def plan() -> None:
        a = rng.choice(w.vocab.plan_actions)
        s.plan(a)
        rep.add(f"plan:{a}", lambda a=a: s.plan(a))

    def read_block(f: Fault) -> None:
        pad = rng.randint(0, 6)
        s.read(f.path, block=f.block, pad=pad)
        rep.add(f"block:{f.id}", lambda f=f, pad=pad: s.read(f.path, block=f.block, pad=pad))

    def search_symbol(f: Fault) -> None:
        scope = f.path.split("/", 1)[0] if rng.random() < 0.7 else "."
        s.search(scope, f.symbol)
        rep.add(f"search:{f.id}", lambda f=f, scope=scope: s.search(scope, f.symbol))

    def read_test_file(test_id: str) -> None:
        tpath = test_id.split("::", 1)[0]
        s.read(tpath)
        rep.add(f"read:{tpath}", lambda tpath=tpath: s.read(tpath))

    # -- prefix (no edits) ---------------------------------------------------
    selection: list[str] | None = None
    if prefix_style == "baseline":
        # A first full run shows the failure; the agent then narrows to the failing test(s).
        r = s.run_tests(stuck_suite)
        f = _fault_of(s, r.failing_test)
        read_test_file(r.failing_test)
        if rng.random() < 0.7:
            read_block(f)
        selection = [r.failing_test] if rng.random() < 0.5 else list(f.tests)
    elif prefix_style == "investigate":
        f = next(x for x in w.faults.values() if x.suite == stuck_suite)
        search_symbol(f)
        read_block(f)
        for c in s.unknown_clues(f)[: rng.randint(0, 2)]:
            if c.kind == "read":
                s.inspect_clue(c)
        if rng.random() < 0.4:
            selection = list(f.tests)  # the tests that exercise the block it just read
    while s.n < prefix_len:
        roll = rng.random()
        if roll < 0.15:
            plan()
        elif roll < 0.35:
            list_dir()
        elif roll < 0.6:
            read_decoy()
        else:
            s.novel_decoy()

    def rerun() -> OpResult:
        return s.run_tests(stuck_suite, selection)

    # -- core: fail, immediate retries within policy -------------------------
    r0 = rerun()
    stuck = _fault_of(s, r0.failing_test)
    s.meta.update(stuck_fault=stuck.id, selection="full" if selection is None else len(selection))
    rerun()  # attempt 1 (opens the retry episode)
    for _ in range(cfg.r_max - 1):
        if rng.random() < interleave_in:
            roll = rng.random()
            if not s.knows(stuck.code_fact) and roll < 0.5:
                read_block(stuck)
            elif roll < 0.75:
                read_test_file(r0.failing_test)
            else:
                op = rep.pick(rng)
                (op or (lambda: read_test_file(r0.failing_test)))()
        rerun()  # attempts 2..R_max (in policy)

    # Make sure there is known content to re-read in the tail.
    if not rep.ops:
        rep.add(f"read:{r0.failing_test}", lambda: s.read(r0.failing_test.split("::", 1)[0]))
    if s.knows(stuck.code_fact):
        rep.add(f"block:{stuck.id}", lambda: s.read(stuck.path, block=stuck.block))
    if s.knows(stuck.loc_fact) and rng.random() < 0.5:
        scope = stuck.path.split("/", 1)[0]
        rep.add(f"search:{stuck.id}", lambda: s.search(scope, stuck.symbol))
    if passing_suites and rng.random() < 0.4:
        ps = rng.choice(passing_suites)
        rep.add(f"green:{ps}", lambda: s.run_tests(ps))

    # -- doomed tail: re-runs past R_max on the unchanged workspace ----------
    suffix_room = 0 if suffix == "none" else (3 if suffix == "insight" else 1)
    spaced_from = rng.randint(1, 4) if tail_shape == "spaced" else None
    k = 0
    while s.n < target_len - suffix_room and s.room(1 + suffix_room):
        k += 1
        if spaced_from is not None and k >= spaced_from:
            n_between = 2 if k == spaced_from else rng.choice((1, 1, 2))
        else:
            n_between = 1 if rng.random() < interleave_tail else 0
        if not s.room(n_between + 1 + suffix_room):
            break
        for _ in range(n_between):
            rep.pick(rng)()
        rerun()

    # -- suffix ----------------------------------------------------------------
    if suffix == "yield":
        s.yield_(rng.choice(("blocked", "partial_result", "handoff")))
    elif suffix == "ask":
        s.ask(rng.choice(("clarify_requirement", "confirm_scope")))
    elif suffix == "insight":
        clues = [c for c in s.unknown_clues(stuck) if c.kind in ("read", "search")]
        if clues:
            s.inspect_clue(clues[0])
        else:
            s.novel_decoy()
        s.yield_(rng.choice(("blocked", "handoff")))
    while s.n < MIN_EVENTS:
        s.novel_decoy()
    return s.finish()
