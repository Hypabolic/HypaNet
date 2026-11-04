"""Family 3, transient external failure (spec §7.1 item 3).

Core latent behaviour: an external call (``http_call``) fails because the
service is transiently down, the agent retries it immediately (in-policy
retries, at most ``R_max`` failing attempts), the call succeeds
(RESOLVES_EPISODE), and productive repair work follows straight away.
Intended label mix: RECOVERY while the episode is open and for ``k_rec``
events after the resolution (rule 4), then PRODUCTIVE once the resolution is
more than ``k_rec`` back and the window holds a fix (rule 5).

Variation (§7.2): language, 1-4 faults, 1-3 outages per session on distinct
hosts, methods (GET/POST/PUT/PATCH/DELETE, fixed request bodies), path
templates, transient error classes, 1..R_max+1 failing calls per outage, an
optional single unrelated operation between two retries, outages on an
observability endpoint whose eventual success yields clue facts, neutral or
productive prefixes, edit vs write fixes, targeted vs full-suite verification,
optional lint/build/yield suffixes, and per-session cost profiles.

This module also holds :class:`Actor`, small scripted-agent helpers shared by
families 3, 4 (``bounded_retry``) and 10 (``polling_recovery``). They only
choose operations from latent state (``s.knows``, faults, services); the
engine decides every effect.
"""

from __future__ import annotations

import json
import random
from typing import Sequence

from ...oracle_config import OracleConfig
from ...schema import CanonicalEvent, OracleRecord
from ..engine import MAX_EVENTS, MIN_EVENTS, CostProfile, OpResult, Session, cosmetic, fix
from ..world import HOSTS, OBS_ENDPOINTS, Fault

FAMILY = "transient_external"

OBS_HOSTS: tuple[str, ...] = tuple(h for h, _ in OBS_ENDPOINTS)
TRANSIENT_ERRORS: tuple[str, ...] = (
    "TimeoutError", "TimeoutError", "HTTP504", "HTTP503", "HTTP502", "ConnectionResetError",
    "ConnectionRefusedError", "DNSResolutionError", "SSLError",
)
METHODS: tuple[str, ...] = ("GET", "GET", "GET", "POST", "POST", "PUT", "PATCH", "DELETE")


def request_body(rng: random.Random, method: str) -> str | None:
    """A fixed JSON body for POST/PATCH (identical on every retry), else None."""
    if method not in ("POST", "PATCH"):
        return None
    keys = rng.sample(("ref", "name", "status", "limit", "query", "target", "labels", "dry_run"), rng.randint(1, 3))
    body = {}
    for k in keys:
        roll = rng.random()
        body[k] = rng.randint(1, 500) if roll < 0.3 else (rng.random() < 0.5 if roll < 0.45 else f"{k}-{rng.getrandbits(16):04x}")
    return json.dumps(body, sort_keys=True)


def pick_hosts(s: Session, rng: random.Random, k: int) -> list[str]:
    """``k`` distinct external hosts with no clue or service attached yet."""
    w = s.world
    taken = set(OBS_HOSTS) | set(w.services) | {c.host for c in w.clues if c.host}
    pool = [h for h in w.vocab.hosts if h not in taken]
    extra = [h for h in HOSTS if h not in taken and h not in pool]
    rng.shuffle(extra)
    return (pool + extra)[:k]


class Actor:
    """Scripted-agent helpers over a :class:`Session` (shared by families 3, 4, 10).

    ``reserved_hosts``: hosts whose services a family set up for its core
    behaviour; clue inspections never call them, so the core episode's
    first call is the planned one. ``reserve``: events every helper leaves
    free under ``MAX_EVENTS`` (set it to what a later core behaviour needs).
    """

    def __init__(self, s: Session, rng: random.Random, reserved_hosts: Sequence[str] = ()) -> None:
        self.s = s
        self.rng = rng
        self.reserved = {h.lower() for h in reserved_hosts}
        self.reserve = 0
        self._plans: list[str] = []

    def room(self, k: int = 1) -> bool:
        """True if ``k`` more events fit while keeping ``reserve`` events free."""
        return self.s.room(k + self.reserve)

    # -- neutral padding -------------------------------------------------

    def plan(self) -> OpResult | None:
        """A plan step with an action not issued before (novel control, NONE)."""
        left = [a for a in self.s.world.vocab.plan_actions if a not in self._plans]
        if not left:
            return None
        a = self.rng.choice(left)
        self._plans.append(a)
        return self.s.plan(a)

    def decoy(self, kinds: Sequence[str] = ("read", "list", "search")) -> OpResult:
        return self.s.novel_decoy(kinds=kinds)

    def neutral(self, k: int, *, plan_p: float = 0.25) -> None:
        """``k`` novel, fruitless operations (NONE): decoy reads, listings, searches, first plans."""
        for _ in range(k):
            if not self.room(1):
                return
            if self.rng.random() < plan_p and self.plan() is not None:
                continue
            roll = self.rng.random()
            self.decoy(("list",) if roll < 0.25 else ("read", "search") if roll < 0.8 else ("read", "list", "search"))

    # -- productive work ---------------------------------------------------

    def locate(self, f: Fault) -> None:
        """Reveal a fault's location: a targeted failing run or a symbol search (NEW_FACT)."""
        if self.s.knows(f.loc_fact):
            return
        if self.rng.random() < 0.5:
            self.s.run_tests(f.suite, list(f.tests))
        else:
            scope = f.path.split("/", 1)[0] if self.rng.random() < 0.7 else "."
            self.s.search(scope, f.symbol)

    def clues(self, f: Fault, p: float) -> None:
        """Inspect some of the fault's unknown cause clues (NEW_FACT each)."""
        for c in self.s.unknown_clues(f):
            if c.fact.endswith((":loc", ":code")) or self.s.knows(c.fact):
                continue
            if c.kind == "http" and (c.host or "").lower() in self.reserved:
                continue
            if self.rng.random() < p and self.room(4):
                self.s.inspect_clue(c)

    def fault_work(self, f: Fault, *, clue_p: float = 0.4, verify: str | None = None,
                   locate_first: bool = True) -> OpResult | None:
        """Productive repair of one fault: locate, read, maybe inspect clues, fix, verify.

        ``verify``: ``"targeted"`` (the fault's own tests; passes), ``"suite"``
        (the whole suite; may fail on the next fault), ``"none"``, or None to
        draw one. Returns the verification result, if any.
        """
        s, rng = self.s, self.rng
        if s.world.is_fixed(f.id):
            return None
        if locate_first:
            self.locate(f)
        if not s.knows(f.code_fact):
            s.read(f.path, block=f.block, pad=rng.randint(0, 6))
        self.clues(f, clue_p)
        if rng.random() < 0.2:
            s.write(f.path, fix(f))
        else:
            s.edit(f.path, fix(f))
        if rng.random() < 0.08 and self.room(3):
            s.edit(f.path, cosmetic(rng.choice(("format", "comment", "rename_local"))))
        if verify is None:
            verify = rng.choices(("targeted", "suite", "none"), (0.35, 0.5, 0.15))[0]
        if not self.room(2):
            verify = "none"
        if verify == "targeted":
            return s.run_tests(f.suite, list(f.tests))
        if verify == "suite":
            return s.run_tests(f.suite)
        return None

    def wrap_up(self, *, final_suites: bool = True) -> None:
        """Optional suffix: confirm suites not yet green, a lint/build, yield."""
        s, rng, w = self.s, self.rng, self.s.world
        if final_suites and rng.random() < 0.5:
            for suite in w.suites:
                green = all(w.is_fixed(f.id) for f in w.faults.values() if f.suite == suite)
                if green and self.room(3) and rng.random() < 0.5:
                    s.run_tests(suite)  # passes, so it never arms a retry
        if rng.random() < 0.25 and self.room(2):
            if rng.random() < 0.5:
                s.lint(rng.choice(w.lint_targets))
            else:
                s.build(rng.choice(w.build_targets))
        while s.n < MIN_EVENTS:
            self.decoy()
        if rng.random() < 0.45 and self.room(1):
            s.yield_(rng.choice(("done", "done", "handoff", "partial_result")))


def outage(a: Actor, method: str, host: str, tmpl: str, body: str | None, *, interleave_p: float) -> OpResult:
    """Call ``host`` until it answers: the initial failure, then immediate retries.

    At most one unrelated operation is placed between two retries (the
    episode tolerates one). Returns the resolving call.
    """
    s, rng = a.s, a.rng
    retry_cost = rng.choice((None, None, 0.7, 0.9, 1.2))
    r = s.http(method, host, tmpl, body=body)
    attempts = 0
    while r.failed:
        if attempts >= 1 and rng.random() < interleave_p:
            if rng.random() < 0.5 and a.plan() is not None:
                pass
            else:
                a.decoy(("read", "search"))
        r = s.http(method, host, tmpl, body=body, cost=retry_cost)
        attempts += 1
        if attempts > 2 * s.cfg.r_max:  # never happens with fail_attempts <= R_max + 1
            raise RuntimeError("transient outage did not recover")
    return r


def generate(rng: random.Random, session_id: str, cfg: OracleConfig) -> tuple[list[CanonicalEvent], OracleRecord]:
    long_form = rng.random() < 0.25
    n_outages = rng.choices((1, 2, 3), (0.45, 0.4, 0.15))[0]
    n_faults = min(4, n_outages + (rng.choice((1, 2, 3)) if long_form else rng.choice((0, 0, 1, 1, 2))))
    s = Session(
        rng, session_id, cfg, family=FAMILY, n_faults=n_faults,
        extra_clues=(1, 3) if long_form else rng.choice(((0, 1), (0, 2), (1, 2))),
        single_suite=rng.random() < 0.5, cost_profile=CostProfile.draw(rng, scale=rng.lognormvariate(0.0, 0.5)),
    )
    w = s.world
    faults = list(w.faults.values())
    rng.shuffle(faults)

    # One outage may hit an observability endpoint that carries a clue: its success yields facts.
    obs_clues = [c for f in faults for c in w.clues_for(f.id) if c.kind == "http"]
    hosts = pick_hosts(s, rng, n_outages)
    plan_outages: list[tuple[str, str, str, str | None, Fault | None]] = []
    obs_used = False
    for i in range(n_outages):
        if obs_clues and not obs_used and rng.random() < 0.6:
            c = obs_clues[0]
            obs_used = True
            plan_outages.append(("GET", c.host or "", c.path_template or "", None, w.faults[c.fault]))
            continue
        m = rng.choice(METHODS)
        plan_outages.append((m, hosts[i], rng.choice(w.vocab.path_templates), request_body(rng, m), None))
    reserved = [o[1] for o in plan_outages]
    for _, host, _, _, _ in plan_outages:
        w.add_service(host, "transient", fail_attempts=rng.randint(1, cfg.r_max + 1), error_class=rng.choice(TRANSIENT_ERRORS))
    a = Actor(s, rng, reserved)
    interleave_p = rng.choice((0.0, 0.0, 0.25, 0.5))

    # Fault schedule: which faults are worked on after which outage (at least one after each).
    pending = [f for f in faults if not any(o[4] is f for o in plan_outages)]
    after: list[list[Fault]] = []
    for _, _, _, _, obs_f in plan_outages:
        seg = [obs_f] if obs_f is not None else ([pending.pop(0)] if pending else [])
        after.append(seg)
    prefix_fault = pending.pop(0) if pending and rng.random() < 0.4 else None
    while pending:
        after[rng.randrange(len(after))].append(pending.pop(0))

    style = "productive" if prefix_fault is not None else "neutral"
    s.meta.update(variant="long" if long_form else "short", n_outages=n_outages, n_faults=n_faults, prefix=style,
                  interleave_p=interleave_p,
                  outages=[{"method": m, "host": h, "fail_attempts": w.services[h.lower()].fail_attempts, "obs": f is not None}
                           for m, h, _, _, f in plan_outages])

    # Prefix: >= 3 events so the first resolution's RECOVERY windows exist.
    if prefix_fault is not None:
        a.neutral(rng.randint(0, 2))
        a.fault_work(prefix_fault, clue_p=0.8 if long_form else 0.4, verify=rng.choice(("targeted", "suite", "none")))
    a.neutral(max(0, rng.randint(3, 10 if long_form else 7) - s.n))

    # Events an outage can take: its failing calls, the resolving call, and interleaved ops.
    need = [w.services[h.lower()].fail_attempts + 1 + (cfg.r_max if interleave_p > 0 else 0)
            for _, h, _, _, _ in plan_outages]
    for i, (method, host, tmpl, body, obs_f) in enumerate(plan_outages):
        if i > 0 and rng.random() < 0.5:
            a.neutral(rng.randint(1, 4 if long_form else 2))
        a.reserve = 0
        outage(a, method, host, tmpl, body, interleave_p=interleave_p)
        # Work up to the next outage keeps room for it, so every planned outage happens.
        a.reserve = need[i + 1] if i + 1 < len(need) else 0
        # Productive work starts straight after the resolution.
        for f in after[i]:
            if not a.room(8):
                break
            a.fault_work(f, clue_p=0.8 if long_form else rng.choice((0.2, 0.5, 0.8)))
            if long_form and rng.random() < 0.3:
                a.neutral(1, plan_p=0.4)

    a.wrap_up()
    return s.finish()
