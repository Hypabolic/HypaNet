"""Family 12, under-instrumented sessions (spec §7.1 item 12, M1_CONTRACTS §3).

Core latent behaviour: an otherwise normal session, drawn from one of several
base behaviours (incremental repair, exploration, transient external retry,
polling recovery, repeated test failure, cosmetic churn, A/B oscillation),
whose observation layer loses required fields on chosen events. Each gate
reason of spec §5.2 is exercised:

* ``tokens``      -> MISSING_TOKENS (input, output or both, possibly a burst)
* ``digests``     -> MISSING_WORKSPACE_DIGEST (before, after or both, possibly a burst)
* ``failure``     -> MISSING_FAILURE_FINGERPRINT (on an ERROR event)
* ``validation``  -> MISSING_VALIDATION_FIELDS (suite id and/or counts on a run_tests event;
  the engine also marks the next run whose baseline is thereby unobserved)
* ``truncation``  -> UNOBSERVED_VALIDATION_BASELINE (``finish(truncate_before=m)``: the first
  observed validation run after the cut has no observable baseline)

Dropped events carry ``LatentStep.dropped = True`` (set by the engine), so
labeller rule 1 fires exactly on the windows that contain one. Drops never
change latent state, effects or the random stream, so every window without a
dropped event keeps the label its base behaviour gives it, and passes the gate.
Intended label mix: INSUFFICIENT_INSTRUMENTATION on dropped windows; the clean
windows follow the base behaviours (PRODUCTIVE plurality by base weighting).

Construction: the base script runs twice from the same sub-seed. The first,
clean pass fixes the event stream; drop sites are then chosen among the events
eligible for each field (an ERROR event for ``failure``, a validation result
for ``validation``), and the second pass replays the identical script with the
drops injected at those indices. Site choice reads only the operation category,
outcome and validation flag of the clean events, never fingerprints, encoded
features or labels.

Why in-module base scripts rather than importing the other families'
``generate``: a family's ``generate`` builds and seals its own ``Session``, and
``drop=`` can only be applied when the engine commits an event, so drops could
only be added to a sealed record by re-implementing the engine's drop and
baseline-propagation logic outside it.
"""

from __future__ import annotations

import random
from typing import Callable, Mapping, Sequence

from ...oracle_config import OracleConfig
from ...schema import CanonicalEvent, OracleRecord, Outcome
from ..engine import MAX_EVENTS, MIN_EVENTS, Session, competing, cosmetic, fix
from ..world import HTTP_ERRORS, Fault

FAMILY = "under_instrumented"

# Base behaviours and their weights. PRODUCTIVE-leaning bases dominate so the
# benign class is the plurality of the clean (scored) windows.
BASES: tuple[tuple[str, float], ...] = (
    ("repair", 3.0),
    ("exploration", 2.0),
    ("transient_http", 2.0),
    ("polling", 1.5),
    ("repeated_test", 1.5),
    ("cosmetic_churn", 1.0),
    ("abab", 1.5),
)

# Gate-reason kinds (spec §5.2) and their field variants (engine DROP_FIELDS).
DROP_KINDS = ("tokens", "digests", "failure", "validation", "truncation")
_FIELD_VARIANTS: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "tokens": (("tokens",), ("input_tokens",), ("output_tokens",)),
    "digests": (("digests",), ("digest_before",), ("digest_after",)),
    "failure": (("failure",),),
    "validation": (("validation_counts",), ("validation_suite",), ("validation_counts", "validation_suite")),
}


class _InstrumentedSession(Session):
    """A Session that adds planned ``drop=`` fields at given event indices.

    Everything else (latent effects, random draws) is the base Session's; the
    engine itself nulls the observed fields and sets ``dropped``.
    """

    def __init__(self, *args, planned_drops: Mapping[int, tuple[str, ...]] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._planned_drops = dict(planned_drops or {})

    def _commit(self, family, target, args, **kw):  # type: ignore[override]
        extra = self._planned_drops.get(len(self._events), ())
        if extra:
            kw["drop"] = tuple(dict.fromkeys(tuple(kw.get("drop", ())) + tuple(extra)))
        return super()._commit(family, target, args, **kw)


# ---------------------------------------------------------------------------
# Base behaviours (each an ordinary session script; no drops here)
# ---------------------------------------------------------------------------


def _neutral_prefix(s: Session, rng: random.Random, k: int) -> None:
    for i in range(k):
        roll = rng.random()
        if i == 0 and roll < 0.3:
            s.plan(rng.choice(s.world.vocab.plan_actions))
        elif roll < 0.5:
            s.novel_decoy(kinds=("list",))
        else:
            s.novel_decoy(kinds=("read", "search"))


def _prefix_len(rng: random.Random, target: int, core: int) -> int:
    """Neutral prefix length: places the core behaviour early, mid or late in the session."""
    return max(0, min(10, target - core - rng.randint(0, 6)))


def _suffix(s: Session, rng: random.Random, min_len: int, target: int = 0) -> None:
    """Wrap-up: optional extra investigation towards ``target`` length, lint/build, padding, yield."""
    w = s.world
    if s.n < target - 3 and rng.random() < 0.7:
        # Follow-up reading: leftover cause clues (new facts), novel decoys and plans.
        leftovers = [c for f in w.faults.values() for c in s.unknown_clues(f)]
        while s.n < target - 2 and s.room(4):
            roll = rng.random()
            if leftovers and roll < 0.35:
                c = leftovers.pop(0)
                if not s.knows(c.fact):
                    s.inspect_clue(c)
            elif roll < 0.5:
                s.plan(rng.choice(w.vocab.plan_actions))
            else:
                s.novel_decoy()
    if rng.random() < 0.3 and s.room(4):
        if rng.random() < 0.5:
            s.lint(rng.choice(w.lint_targets))
        else:
            s.build(rng.choice(w.build_targets))
    while s.n < min_len:
        s.novel_decoy()
    if rng.random() < 0.4 and s.room(1):
        s.yield_(rng.choice(("done", "handoff", "partial_result")))


def _repair_fault(s: Session, rng: random.Random, f: Fault, *, clues: int = 1, thorough: int = 0) -> None:
    """Locate (targeted failing run or symbol search), read, inspect clues, fix, verify."""
    if not s.knows(f.loc_fact):
        if rng.random() < 0.6:
            s.run_tests(f.suite, list(f.tests))
        else:
            s.search(f.path.split("/", 1)[0], f.symbol)
    if not s.knows(f.code_fact):
        s.read(f.path, block=f.block, pad=rng.randint(0, 6))
    extras = [c for c in s.unknown_clues(f) if not c.fact.endswith((":loc", ":code"))]
    for c in extras[: rng.randint(0, clues)]:
        if s.room(10):
            s.inspect_clue(c)
    for _ in range(rng.randint(0, thorough)):
        if not s.room(10):
            break
        if rng.random() < 0.4:
            s.plan(rng.choice(s.world.vocab.plan_actions))
        else:
            s.novel_decoy()
    if rng.random() < 0.2:
        s.write(f.path, fix(f))
    else:
        s.edit(f.path, fix(f))
    if rng.random() < 0.25 and s.room(8):
        s.run_tests(f.suite, list(f.tests))  # targeted confirmation (passes)
    s.run_tests(f.suite)


def _repair_all(s: Session, rng: random.Random, faults: Sequence[Fault], thorough: int) -> None:
    for f in faults:
        if not s.room(12) or s.world.is_fixed(f.id):
            continue
        _repair_fault(s, rng, f, clues=rng.choice((0, 1, 2)), thorough=thorough)


def _faults_for(rng: random.Random, target: int, per_fault: int, reserve: int) -> int:
    return max(1, min(4, (target - reserve) // per_fault + rng.choice((-1, 0, 0, 1))))


def _base_repair(s: Session, rng: random.Random, target: int, min_len: int) -> None:
    faults = list(s.world.faults.values())
    thorough = rng.choice((0, 1, 2))
    _neutral_prefix(s, rng, _prefix_len(rng, target, len(faults) * (5 + thorough)))
    _repair_all(s, rng, faults, thorough)
    for suite in s.world.suites:
        if rng.random() < 0.25 and s.room(4):
            s.run_tests(suite)
    _suffix(s, rng, min_len, target)


def _base_exploration(s: Session, rng: random.Random, target: int, min_len: int) -> None:
    w = s.world
    if rng.random() < 0.5:
        s.plan(rng.choice(w.vocab.plan_actions))
    if rng.random() < 0.5:
        s.list_dir(".")
    faults = list(w.faults.values())
    decoy_rate = rng.uniform(0.1, 0.5)
    reserve = 2 * len(faults) + 6
    for f in faults:
        if s.n + reserve > MAX_EVENTS:
            break
        if not s.knows(f.loc_fact):
            s.search(f.path.split("/", 1)[0] if rng.random() < 0.7 else ".", f.symbol)
        if not s.knows(f.code_fact):
            s.read(f.path, block=f.block, pad=rng.randint(0, 10))
        for c in s.unknown_clues(f):
            if s.n + reserve + 2 > MAX_EVENTS:
                break
            if not s.knows(c.fact):
                s.inspect_clue(c)
            if rng.random() < decoy_rate:
                s.novel_decoy()
    for f in faults:
        (s.write if rng.random() < 0.2 else s.edit)(f.path, fix(f))
    for suite in dict.fromkeys(f.suite for f in faults):
        s.run_tests(suite)
    _suffix(s, rng, min_len, target)


def _base_transient_http(s: Session, rng: random.Random, target: int, min_len: int) -> None:
    w = s.world
    host = rng.choice(w.vocab.hosts)
    tmpl = rng.choice(w.vocab.path_templates)
    k = rng.randint(1, s.cfg.r_max)
    w.add_service(host, "transient", fail_attempts=k, error_class=rng.choice(HTTP_ERRORS))
    method = rng.choice(("GET", "GET", "POST"))
    body = '{"dry_run": false, "id": 1}' if method == "POST" else None
    faults = list(w.faults.values())
    _neutral_prefix(s, rng, _prefix_len(rng, target, k + 1 + 5 * len(faults)))
    r = s.http(method, host, tmpl, body=body)
    while r.failed:
        r = s.http(method, host, tmpl, body=body)
    _repair_all(s, rng, faults, rng.choice((0, 1)))
    _suffix(s, rng, min_len, target)


def _base_polling(s: Session, rng: random.Random, target: int, min_len: int) -> None:
    w = s.world
    res = rng.choice(w.vocab.resources)
    k = rng.randint(1, s.cfg.p_max)
    w.add_resource(res, ready_after=k)
    faults = list(w.faults.values())
    _neutral_prefix(s, rng, _prefix_len(rng, target, k + 1 + 5 * len(faults)))
    backoff = rng.choice((None, 1, 2))
    r = s.poll(res, backoff=backoff)
    i = 0
    while r.not_ready:
        i += 1
        r = s.poll(res, backoff=None if backoff is None else backoff * 2 ** min(i, 4))
    _repair_all(s, rng, faults, rng.choice((0, 1)))
    _suffix(s, rng, min_len, target)


def _productive_lead(s: Session, rng: random.Random, faults: Sequence[Fault]) -> None:
    """Optional productive prefix for the stall/oscillation bases: repair the other faults first."""
    if len(faults) > 1 and rng.random() < 0.5:
        _repair_all(s, rng, faults[1:], 0)


def _core_reps(rng: random.Random, s: Session, target: int, per_rep: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, (target - s.n - 2) // per_rep + rng.choice((-1, 0, 1))))


def _base_repeated_test(s: Session, rng: random.Random, target: int, min_len: int) -> None:
    """Same failing suite re-run on an unchanged workspace with re-reads: RECOVERY then STALLED."""
    faults = list(s.world.faults.values())
    f = faults[0]
    _productive_lead(s, rng, faults)
    _neutral_prefix(s, rng, rng.randint(0, 4))
    s.read(f.path, block=f.block)
    sel = list(f.tests) if rng.random() < 0.4 else None
    s.run_tests(f.suite, sel)
    s.run_tests(f.suite, sel)
    reread = rng.uniform(0.4, 0.9)
    for _ in range(_core_reps(rng, s, target, 2, 4, 14)):
        if not s.room(10):
            break
        if rng.random() < reread:
            s.read(f.path, block=f.block)  # re-read of known content (REDUNDANT, one unrelated op)
        s.run_tests(f.suite, sel)
    if rng.random() < 0.4 and s.room(8):  # the agent eventually changes approach
        s.edit(f.path, fix(f))
        s.run_tests(f.suite)
    _suffix(s, rng, min_len, target)


def _base_cosmetic_churn(s: Session, rng: random.Random, target: int, min_len: int) -> None:
    """Text-only edits with re-runs on a behaviourally unchanged workspace: STALLED (r2)."""
    faults = list(s.world.faults.values())
    f = faults[0]
    _productive_lead(s, rng, faults)
    _neutral_prefix(s, rng, rng.randint(0, 4))
    s.run_tests(f.suite)
    kinds = ("rename_local", "comment", "whitespace", "docstring")
    for _ in range(_core_reps(rng, s, target, 2, 4, 14)):
        if not s.room(10):
            break
        s.edit(f.path, cosmetic(rng.choice(kinds)))
        s.run_tests(f.suite)
    if rng.random() < 0.4 and s.room(8):
        if not s.knows(f.code_fact):
            s.read(f.path, block=f.block)
        s.edit(f.path, fix(f))
        s.run_tests(f.suite)
    _suffix(s, rng, min_len, target)


def _base_abab(s: Session, rng: random.Random, target: int, min_len: int) -> None:
    """Competing edits A/B/A/B with tests between them: OSCILLATING."""
    faults = list(s.world.faults.values())
    f = faults[0]
    _productive_lead(s, rng, faults)
    _neutral_prefix(s, rng, rng.randint(0, 4))
    if rng.random() < 0.6:
        s.read(f.path, block=f.block)
    sel = list(f.tests) if rng.random() < 0.6 else None
    for i in range(2 * _core_reps(rng, s, target, 4, 2, 7)):
        if not s.room(10):
            break
        s.edit(f.path, competing(f, "AB"[i % 2]))
        s.run_tests(f.suite, sel)
    if rng.random() < 0.4 and s.room(8):
        s.edit(f.path, fix(f))
        s.run_tests(f.suite)
    _suffix(s, rng, min_len, target)


_BASE_SCRIPTS: Mapping[str, Callable[[Session, random.Random, int, int], None]] = {
    "repair": _base_repair,
    "exploration": _base_exploration,
    "transient_http": _base_transient_http,
    "polling": _base_polling,
    "repeated_test": _base_repeated_test,
    "cosmetic_churn": _base_cosmetic_churn,
    "abab": _base_abab,
}

# World.generate kwargs per base, scaled by the session's target length.
_WORLD_KWARGS: Mapping[str, Callable[[random.Random, int], dict]] = {
    "repair": lambda r, t: dict(n_faults=_faults_for(r, t, 6, 4), extra_clues=(0, 2), single_suite=r.random() < 0.5),
    "exploration": lambda r, t: dict(n_faults=_faults_for(r, t, 7, 4), extra_clues=(2, 4)),
    "transient_http": lambda r, t: dict(n_faults=_faults_for(r, t, 6, 8)),
    "polling": lambda r, t: dict(n_faults=_faults_for(r, t, 6, 9)),
    "repeated_test": lambda r, t: dict(n_faults=r.choice((1, 2, 2, 3))),
    "cosmetic_churn": lambda r, t: dict(n_faults=r.choice((1, 2, 2, 3))),
    "abab": lambda r, t: dict(n_faults=r.choice((1, 2, 2, 3))),
}


def _run_base(base: str, script_seed: int, session_id: str, cfg: OracleConfig, target: int, min_len: int,
              planned: Mapping[int, tuple[str, ...]]) -> _InstrumentedSession:
    rng = random.Random(script_seed)
    world_kwargs = _WORLD_KWARGS[base](rng, target)
    s = _InstrumentedSession(rng, session_id, cfg, family=FAMILY, planned_drops=planned, **world_kwargs)
    _BASE_SCRIPTS[base](s, rng, target, min_len)
    return s


# ---------------------------------------------------------------------------
# Drop planning
# ---------------------------------------------------------------------------


def _eligible(kind: str, events: Sequence[CanonicalEvent]) -> list[int]:
    if kind in ("tokens", "digests"):
        return list(range(len(events)))
    if kind == "failure":
        return [i for i, e in enumerate(events) if e.outcome is Outcome.ERROR]
    if kind == "validation":
        return [i for i, e in enumerate(events) if e.validation_produced]
    raise ValueError(kind)


def _truncation_point(rng: random.Random, events: Sequence[CanonicalEvent]) -> int:
    """A cut ``m`` (>= 1, leaving >= MIN_EVENTS) with a validation run at or after it; 0 if none.

    Half the time the cut falls just after an earlier validation run, so the
    next run's baseline exists but is lost; otherwise anywhere before a run.
    """
    n = len(events)
    vals = [i for i, e in enumerate(events) if e.validation_produced]
    cands = [m for m in range(1, n - MIN_EVENTS + 1) if any(v >= m for v in vals)]
    if not cands:
        return 0
    cut_after = [v + 1 for v in vals if v + 1 in cands]
    if cut_after and rng.random() < 0.5:
        return rng.choice(cut_after)
    return rng.choice(cands)


def _plan_drops(rng: random.Random, events: Sequence[CanonicalEvent], kinds: Sequence[str]
                ) -> tuple[dict[int, tuple[str, ...]], int, list[dict]]:
    """Choose drop sites for each requested kind on the clean event stream.

    Returns ``(planned {index: fields}, truncate_before, sites)``. A kind with
    no eligible event is skipped; at least one site is always placed (a token
    drop is always possible).
    """
    n = len(events)
    planned: dict[int, tuple[str, ...]] = {}
    sites: list[dict] = []
    m = 0
    anchor: int | None = None

    def add(i: int, fields: tuple[str, ...]) -> None:
        planned[i] = tuple(dict.fromkeys(planned.get(i, ()) + fields))

    # Truncation first, so field sites are placed on events that remain observed.
    for kind in sorted(kinds, key=lambda k: k != "truncation"):
        if kind == "truncation":
            if m == 0:
                m = _truncation_point(rng, events)
                if m:
                    sites.append({"kind": "truncation", "truncate_before": m})
            continue
        cand = [i for i in _eligible(kind, events) if i >= m]
        if not cand:
            continue
        if anchor is not None and rng.random() < 0.6:  # clustered outage near the first site
            near = [i for i in cand if abs(i - anchor) <= 4]
            cand = near or cand
        i = rng.choice(cand)
        fields = rng.choice(_FIELD_VARIANTS[kind])
        span = 1
        if kind in ("tokens", "digests") and rng.random() < 0.35:
            span = rng.randint(2, 4)  # a short telemetry gap
        idxs = [j for j in range(i, min(n, i + span))]
        for j in idxs:
            add(j, fields)
        sites.append({"kind": kind, "fields": list(fields), "index": i, "span": len(idxs)})
        if anchor is None:
            anchor = i
    if not planned and not m:
        i = rng.randrange(n)
        add(i, ("tokens",))
        sites.append({"kind": "tokens", "fields": ["tokens"], "index": i, "span": 1})
    # Never drop on an event that truncation removes (indices refer to the full stream).
    planned = {i: f for i, f in planned.items() if i >= m}
    return planned, m, sites


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _draw_kinds(rng: random.Random) -> list[str]:
    kinds = [DROP_KINDS[rng.randrange(len(DROP_KINDS))]]
    roll = rng.random()
    extra = 0 if roll < 0.55 else (1 if roll < 0.9 else 2)
    for _ in range(extra):
        kinds.append(rng.choice(DROP_KINDS))
    return kinds


def generate_with_base(
    rng: random.Random, session_id: str, cfg: OracleConfig, *, with_base: bool = True
) -> tuple[tuple[list[CanonicalEvent], OracleRecord], tuple[list[CanonicalEvent], OracleRecord] | None]:
    """``(instrumented, clean)``: the family session and, if ``with_base``, the
    same script finished with no drops and no truncation (for tests/audits)."""
    names = [b for b, _ in BASES]
    base = rng.choices(names, weights=[w for _, w in BASES])[0]
    script_seed = rng.getrandbits(64)
    kinds = _draw_kinds(rng)
    target = rng.randint(MIN_EVENTS, 58)  # desired length; the base scales its core to it
    # Truncation needs room for the cut: the base pads to a longer minimum.
    min_len = rng.randint(MIN_EVENTS + 3, MIN_EVENTS + 9) if "truncation" in kinds else MIN_EVENTS

    clean = _run_base(base, script_seed, session_id, cfg, target, min_len, {})
    planned, m, sites = _plan_drops(rng, clean.events, kinds)
    base_pair = None
    if with_base:
        clean.meta.update(base=base, clean_reference=True)
        base_pair = clean.finish()

    s = _run_base(base, script_seed, session_id, cfg, target, min_len, planned)
    s.meta.update(base=base, target=target, drop_kinds=list(kinds), drop_sites=sites)
    return s.finish(truncate_before=m), base_pair


def generate(rng: random.Random, session_id: str, cfg: OracleConfig) -> tuple[list[CanonicalEvent], OracleRecord]:
    return generate_with_base(rng, session_id, cfg, with_base=False)[0]
