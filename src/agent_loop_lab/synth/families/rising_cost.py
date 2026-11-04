"""Family 11, rising context or token expenditure (spec §7.1 item 11).

Token cost rises steadily through every session of this family, in both of
its variants, by the same per-session ramp procedure. Cost never decides an
effect or a label (the engine ignores it); the label comes from latent
redundancy or progress only, so cost alone cannot separate the variants.

- ``redundant`` (STALLED): after a short orientation that reads the relevant
  code, runs the tests once and settles on a plan, the agent keeps
  re-acquiring what it already has: re-reading files and blocks whose content
  or facts it already holds, repeating searches and directory listings,
  re-issuing the same plan step, repeating an identical successful HTTP call,
  re-running a suite on an unchanged behavioural workspace (never as the
  immediately next operation, so no retry episode opens), and occasionally a
  text-only edit. All of these are REDUNDANT (rule 3 once the orientation
  facts have left the window).
- ``productive`` (PRODUCTIVE): the same rising spend on real work: failing
  runs reveal locations, reads and cause clues add facts, fix edits raise
  progress (rule 5).

Ramp shapes (drawn independently of the variant): geometric, linear, or
stepwise growth of ``Session.cost_multiplier`` per operation, capped so the
engine's per-event token clip is rarely reached.

Variation (§7.2): language and world; 1-4 faults; orientation length; the
redundant-operation mix (inspection-heavy, control-heavy, validation-heavy);
productive investigation depth; ramp shape, start and slope; session length;
optional suffix.
"""

from __future__ import annotations

import random
from typing import Any, Callable

from ...oracle_config import OracleConfig
from ...schema import CanonicalEvent, OracleRecord
from ..engine import MAX_EVENTS, MIN_EVENTS, OpResult, Session, cosmetic, fix
from ..world import Fault

FAMILY = "rising_cost"
REDUNDANT_SHARE = 0.6


class _Ramp:
    """Per-session cost ramp: multiplier as a function of the event index."""

    def __init__(self, rng: random.Random) -> None:
        self.shape = rng.choice(("geometric", "linear", "steps"))
        self.start = rng.uniform(0.5, 1.2)
        self.cap = rng.uniform(6.0, 12.0)
        if self.shape == "geometric":
            self.rate = rng.uniform(1.025, 1.07)
        elif self.shape == "linear":
            self.rate = rng.uniform(0.04, 0.14)
        else:
            self.rate = rng.uniform(1.15, 1.4)
            self.step = rng.randint(3, 6)

    def __call__(self, k: int) -> float:
        if self.shape == "geometric":
            m = self.start * self.rate**k
        elif self.shape == "linear":
            m = self.start * (1.0 + self.rate * k)
        else:
            m = self.start * self.rate ** (k // self.step)
        return min(m, self.start * self.cap)

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {"shape": self.shape, "start": round(self.start, 4), "rate": round(self.rate, 4),
                             "cap": round(self.cap, 3)}
        if self.shape == "steps":
            d["step"] = self.step
        return d


class _RampSession(Session):
    """A Session whose token multiplier follows the ramp at every operation."""

    ramp: _Ramp

    def _commit(self, *args: Any, **kwargs: Any) -> OpResult:  # noqa: D401 - engine hook
        self.cost_multiplier = self.ramp(self.n)
        return super()._commit(*args, **kwargs)


def _fault_of(s: Session, test_id: str | None) -> Fault | None:
    if test_id is None:
        return None
    fids = s.world.tests[test_id].faults
    return s.world.faults[fids[0]] if fids else None


def _redundant(s: Session, rng: random.Random, target_len: int) -> None:
    w = s.world
    reps: list[tuple[str, Callable[[], OpResult]]] = []
    names: set[str] = set()

    def keep(name: str, kind: str, op: Callable[[], OpResult]) -> None:
        if name not in names:
            names.add(name)
            reps.append((kind, op))

    # Orientation: everything later re-acquired is first acquired here.
    faults = list(w.faults.values())
    if rng.random() < 0.6:
        s.list_dir(".")
        keep("ls:.", "inspect", lambda: s.list_dir("."))
    suite = faults[0].suite
    r = s.run_tests(suite)
    keep(f"test:{suite}", "validate", lambda: s.run_tests(suite))
    f = _fault_of(s, r.failing_test) or faults[0]
    pad = rng.randint(0, 6)
    s.read(f.path, block=f.block, pad=pad)
    keep(f"block:{f.id}", "inspect", lambda: s.read(f.path, block=f.block, pad=pad))
    if rng.random() < 0.6:
        s.read(f.path)  # the whole file (big context)
        keep(f"file:{f.path}", "inspect", lambda: s.read(f.path))
    tpath = (r.failing_test or f.tests[0]).split("::", 1)[0]
    s.read(tpath)
    keep(f"file:{tpath}", "inspect", lambda: s.read(tpath))
    for _ in range(rng.randint(0, 2)):
        decoys = [p for p in w.decoy_paths() if f"file:{p}" not in names]
        if decoys:
            p = rng.choice(decoys)
            s.read(p)
            keep(f"file:{p}", "inspect", lambda p=p: s.read(p))
    if rng.random() < 0.5:
        scope = f.path.split("/", 1)[0]
        s.search(scope, f.symbol)
        keep(f"search:{f.id}", "inspect", lambda: s.search(scope, f.symbol))
    if rng.random() < 0.5:
        host, tmpl = rng.choice(w.vocab.hosts), rng.choice(w.vocab.path_templates)
        s.http("GET", host, tmpl)  # a healthy service: succeeds
        keep(f"http:{host}{tmpl}", "external", lambda: s.http("GET", host, tmpl))
    actions = rng.sample(w.vocab.plan_actions, rng.randint(1, 2))
    for a in actions:
        s.plan(a)
        keep(f"plan:{a}", "control", lambda a=a: s.plan(a))
    last_kind = "control"
    if len(w.suites) > 1 and rng.random() < 0.4:
        other = rng.choice([x for x in w.suites if x != suite])
        s.run_tests(other)
        keep(f"test:{other}", "validate", lambda: s.run_tests(other))
        last_kind = "validate"

    # Churn: re-acquire known content with rising spend.
    weights = {"inspect": rng.uniform(1.0, 4.0), "control": rng.uniform(0.3, 2.0), "validate": rng.uniform(0.3, 1.5),
               "external": rng.uniform(0.2, 1.0)}
    cosmetic_rate = rng.choice((0.0, 0.0, 0.1))
    s.meta.update(mix={k: round(v, 3) for k, v in weights.items()}, cosmetic_rate=cosmetic_rate)
    while s.n < target_len and s.room(2):
        if rng.random() < cosmetic_rate:
            s.edit(f.path, cosmetic(rng.choice(("comment", "rename_local", "whitespace", "docstring"))))
            last_kind = "mutate"
            continue
        # Never re-run a (failing) validation as the immediately next operation.
        pool = [(k, op) for k, op in reps if not (k == "validate" and last_kind == "validate")]
        total = sum(weights[k] for k, _ in pool)
        x = rng.uniform(0.0, total)
        for k, op in pool:
            x -= weights[k]
            if x <= 0:
                break
        op()
        last_kind = k


def _productive(s: Session, rng: random.Random, target_len: int) -> None:
    w = s.world
    depth = rng.choice((0.7, 0.9, 1.0))
    decoy_rate = rng.choice((0.0, 0.15, 0.3))
    s.meta.update(depth=depth, decoy_rate=decoy_rate)
    if rng.random() < 0.4:
        s.plan(rng.choice(w.vocab.plan_actions))
    if rng.random() < 0.4:
        s.list_dir(".")
    suites = list(dict.fromkeys(f.suite for f in w.faults.values()))
    r = s.run_tests(suites[0])
    while w.unfixed_faults() and s.room(4):
        f = _fault_of(s, r.failing_test)
        if f is None or w.is_fixed(f.id):
            nxt = next(x.suite for x in w.unfixed_faults())
            r = s.run_tests(nxt)
            continue
        if not s.knows(f.code_fact):
            s.read(f.path, block=f.block, pad=rng.randint(0, 8))
        for c in s.unknown_clues(f):
            if c.fact.endswith((":loc", ":code")):
                continue
            if rng.random() < depth and s.room(4):
                s.inspect_clue(c)
                if rng.random() < decoy_rate and s.room(5):
                    s.novel_decoy()  # at most one fruitless look per fact
        if rng.random() < 0.2:
            s.write(f.path, fix(f))
        else:
            s.edit(f.path, fix(f))
        r = s.run_tests(f.suite)
    for suite in w.suites:
        if suite not in suites and s.n < target_len and s.room(2) and rng.random() < 0.5:
            s.run_tests(suite)
    if s.n < target_len and s.room(2) and rng.random() < 0.4:
        if rng.random() < 0.5:
            s.lint(rng.choice(w.lint_targets))
        else:
            s.build(rng.choice(w.build_targets))


def generate(rng: random.Random, session_id: str, cfg: OracleConfig) -> tuple[list[CanonicalEvent], OracleRecord]:
    ramp = _Ramp(rng)
    variant = "redundant" if rng.random() < REDUNDANT_SHARE else "productive"
    if variant == "redundant":
        n_faults = rng.choice((1, 2, 3))
        extra = (0, 2)
    else:
        n_faults = rng.choice((2, 3, 3, 4, 4))
        extra = (2, 4)
    s = _RampSession(rng, session_id, cfg, family=FAMILY, n_faults=n_faults, extra_clues=extra,
                     single_suite=rng.random() < 0.5)
    s.ramp = ramp
    # Shared length budget, so session length (and with it the ramp's reach) does not separate the variants.
    target_len = rng.randint(MIN_EVENTS + 4, 40)
    s.meta.update(variant=variant, n_faults=n_faults, ramp=ramp.describe(), target_len=target_len)
    if variant == "redundant":
        _redundant(s, rng, target_len)
    else:
        _productive(s, rng, target_len)
    while s.n < MIN_EVENTS:
        s.novel_decoy()
    if rng.random() < 0.3 and s.room(1):
        s.yield_(rng.choice(("done", "handoff", "partial_result")))
    return s.finish()
