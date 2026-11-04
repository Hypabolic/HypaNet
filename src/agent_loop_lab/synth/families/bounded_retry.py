"""Family 4, bounded backoff or retry (spec §7.1 item 4).

Core latent behaviour: a recoverable failure followed by one or two controlled
retries, then resolution. Each session has 1-3 such episodes, each drawn from:

- ``transient_http``: an external call fails transiently (rate limit, 5xx,
  reset) for 1-2 calls; 1-2 immediate retries, the last one succeeds.
- ``diag_test``: a targeted test run fails, the agent re-runs it once or twice
  (in-policy retries on an unchanged workspace), makes at most one inspection,
  repairs the diagnosed cause (``fix``), and the retry passes: a diagnosed
  repair, then retry.
- ``diag_http``: an external call fails because of a code fault (``needs_fix``
  service, auth-class error); retry, at most one inspection, repair, retry ok.
- ``diag_tool``: a lint or build target fails on a fault; re-run, inspect,
  repair, re-run ok.

Every episode resolves within ``R_max`` attempts. Intended label: RECOVERY
(rule 4) while each episode is open and for ``k_rec`` events after it
resolves; the work between episodes gives some PRODUCTIVE windows.

Lookalike (5 vs 4): contiguous VALIDATE-ERROR and EXTERNAL-ERROR retry runs
that resolve here, where family 5 goes beyond ``R_max`` without resolving.

Variation (§7.2): language, episode kinds and order, 1-4 faults, hosts,
methods and bodies, error classes, 1-2 failing retries, optional single
interleaved operation, targeted vs full-suite runs, edit vs write repairs,
neutral or productive prefixes and gaps, suffixes, cost profiles.
"""

from __future__ import annotations

import random

from ...oracle_config import OracleConfig
from ...schema import CanonicalEvent, OracleRecord
from ..engine import CostProfile, OpResult, Session, fix
from ..world import AUTH_ERRORS, Fault
from .transient_external import METHODS, Actor, pick_hosts, request_body

FAMILY = "bounded_retry"

RETRYABLE_ERRORS: tuple[str, ...] = (
    "HTTP429", "HTTP429", "HTTP503", "HTTP502", "HTTP500", "ConnectionResetError", "TimeoutError", "HTTP504",
)
KINDS: tuple[str, ...] = ("transient_http", "diag_test", "diag_http", "diag_tool")


def _tool_target(s: Session, f: Fault) -> tuple[str, str] | None:
    """A lint/build target that fails only because of ``f`` (so repairing ``f`` makes it pass)."""
    w = s.world
    for c in w.clues_for(f.id):
        if c.kind not in ("lint", "build") or c.target is None:
            continue
        others = [x for x in w.tool_clues(c.kind, c.target) if x.fault != f.id and not w.is_fixed(x.fault)]
        if not others:
            return c.kind, c.target
    return None


def _repair(a: Actor, f: Fault, *, inspect: bool) -> None:
    """At most one inspection (the faulty block, if its code is unknown), then the fix."""
    s, rng = a.s, a.rng
    if inspect and not s.knows(f.code_fact):
        s.read(f.path, block=f.block, pad=rng.randint(0, 5))
    if rng.random() < 0.2:
        s.write(f.path, fix(f))
    else:
        s.edit(f.path, fix(f))


def _retries(a: Actor, op, n_failing: int, *, interleave_p: float) -> OpResult:
    """``n_failing`` immediate re-runs of ``op``; one unrelated op may sit between two of them."""
    r = None
    for k in range(n_failing):
        if k >= 1 and a.rng.random() < interleave_p:
            if a.plan() is None:
                a.decoy(("read", "search"))
        r = op()
    return r


def transient_http(a: Actor, host: str, *, interleave_p: float) -> OpResult:
    s, rng, w = a.s, a.rng, a.s.world
    m = rng.choice(METHODS)
    tmpl = rng.choice(w.vocab.path_templates)
    body = request_body(rng, m)
    cost = rng.choice((None, 0.6, 0.8))
    r = s.http(m, host, tmpl, body=body)
    k = w.services[host.lower()].fail_attempts
    r = _retries(a, lambda: s.http(m, host, tmpl, body=body, cost=cost), k, interleave_p=interleave_p)
    return r


def diag_test(a: Actor, f: Fault, *, interleave_p: float) -> OpResult:
    s, rng, w = a.s, a.rng, a.s.world
    solo = [x for x in w.unfixed_faults() if x.suite == f.suite] == [f] and not w.regressions()
    prev_run = s.n > 0 and s.events[-1].operation_family == "run_tests"
    sel = None if (solo and not prev_run and rng.random() < 0.35) else list(f.tests)
    s.run_tests(f.suite, sel)  # fails (NEW_FACT if it reveals the location)
    _retries(a, lambda: s.run_tests(f.suite, sel), rng.choice((1, 1, 2)), interleave_p=interleave_p)
    _repair(a, f, inspect=rng.random() < 0.8)
    return s.run_tests(f.suite, sel)  # passes: RESOLVES_EPISODE


def diag_http(a: Actor, f: Fault, host: str, *, interleave_p: float) -> OpResult:
    s, rng, w = a.s, a.rng, a.s.world
    m = rng.choice(("GET", "GET", "POST", "PUT"))
    tmpl = rng.choice(w.vocab.path_templates)
    body = request_body(rng, m)
    s.http(m, host, tmpl, body=body)
    _retries(a, lambda: s.http(m, host, tmpl, body=body), rng.choice((1, 1, 2)), interleave_p=interleave_p)
    _repair(a, f, inspect=rng.random() < 0.7)
    return s.http(m, host, tmpl, body=body)


def diag_tool(a: Actor, f: Fault, kind: str, target: str, *, interleave_p: float) -> OpResult:
    s, rng = a.s, a.rng
    op = s.lint if kind == "lint" else s.build
    op(target)
    _retries(a, lambda: op(target), rng.choice((1, 1, 2)), interleave_p=interleave_p)
    _repair(a, f, inspect=rng.random() < 0.8)
    return op(target)


def triage(a: Actor, f: Fault) -> None:
    """Run the failing suite, then single failing tests one by one (all distinct
    selections, so no retry episode opens); the diagnosed-repair episode on ``f``
    follows immediately."""
    s, rng, w = a.s, a.rng, a.s.world
    s.run_tests(f.suite)
    singles = [t for x in w.unfixed_faults() if x.suite == f.suite for t in x.tests]
    rng.shuffle(singles)
    for t in singles[: rng.randint(1, 5)]:
        if s.room(10):
            s.run_tests(f.suite, [t])


def generate(rng: random.Random, session_id: str, cfg: OracleConfig) -> tuple[list[CanonicalEvent], OracleRecord]:
    long_form = rng.random() < 0.25
    n_eps = rng.choices((2, 3, 4), (0.3, 0.4, 0.3))[0] if long_form else rng.choices((1, 2, 3), (0.3, 0.45, 0.25))[0]
    kinds = [rng.choices(KINDS, (0.35, 0.35, 0.15, 0.15))[0] for _ in range(n_eps)]
    n_diag = sum(k != "transient_http" for k in kinds)
    n_faults = 4 if long_form else max(1, min(4, n_diag + rng.choice((0, 0, 1, 1, 2))))
    s = Session(
        rng, session_id, cfg, family=FAMILY, n_faults=n_faults,
        extra_clues=(1, 3) if long_form else rng.choice(((0, 1), (0, 2), (1, 2))),
        clue_kinds=("config", "log", "keyword", "lint", "build", "lint", "build", "http"),
        single_suite=rng.random() < 0.4, cost_profile=CostProfile.draw(rng, scale=rng.lognormvariate(0.0, 0.5)),
    )
    w = s.world
    faults = list(w.faults.values())
    rng.shuffle(faults)

    # Assign faults to diagnosed-repair episodes; a diag_tool needs a target only its fault breaks.
    hosts = pick_hosts(s, rng, n_eps)
    plan: list[dict] = []
    free = list(faults)
    for i, kind in enumerate(kinds):
        if kind == "diag_tool":
            cand = next((f for f in free if _tool_target(s, f) is not None), None)
            kind = "diag_tool" if cand is not None else "diag_test"
        if kind == "transient_http" or not free:
            kind = "transient_http"
            w.add_service(hosts[i], "transient", fail_attempts=rng.choice((1, 1, 2)), error_class=rng.choice(RETRYABLE_ERRORS))
            plan.append({"kind": kind, "host": hosts[i]})
            continue
        f = cand if kind == "diag_tool" else free[0]
        free.remove(f)
        ep = {"kind": kind, "fault": f}
        if kind == "diag_http":
            w.add_service(hosts[i], "needs_fix", error_class=rng.choice(AUTH_ERRORS), requires=f.id)
            ep["host"] = hosts[i]
        plan.append(ep)
    a = Actor(s, rng, [ep["host"] for ep in plan if "host" in ep])
    interleave_p = rng.choice((0.0, 0.0, 0.3))
    s.meta.update(variant="long" if long_form else "short", n_faults=n_faults, interleave_p=interleave_p,
                  episodes=[{k: (v.id if isinstance(v, Fault) else v) for k, v in ep.items()} for ep in plan])
    clue_p = 0.8 if long_form else 0.4
    do_triage = plan[0]["kind"] == "diag_test" and rng.random() < 0.4
    s.meta["triage"] = do_triage

    # Prefix (>= 3 events): neutral orientation, sometimes a productive repair of a free fault,
    # sometimes a triage of the failing tests right before the first (diagnosed-repair) episode.
    if free and rng.random() < (0.6 if long_form else 0.35):
        a.neutral(rng.randint(0, 2))
        a.fault_work(free.pop(0), clue_p=clue_p, verify=rng.choice(("targeted", "suite", "none")))
    a.neutral(max(0, rng.randint(1 if do_triage else 3, 10 if long_form else 7) - s.n))
    if do_triage:
        triage(a, plan[0]["fault"])

    for i, ep in enumerate(plan):
        if i > 0:
            # Gap between episodes: productive work on a free fault, or a little neutral activity.
            if free and rng.random() < (0.8 if long_form else 0.5) and s.room(12):
                a.fault_work(free.pop(0), clue_p=clue_p, verify=rng.choice(("targeted", "suite", "none")))
            elif rng.random() < 0.7:
                a.neutral(rng.randint(1, 3))
        if not s.room(7):
            break
        # Each episode's first operation has a key no earlier operation used (reserved hosts,
        # a targeted selection, a full suite only after a non-test op), so no accidental pairing.
        kind = ep["kind"]
        if kind == "transient_http":
            transient_http(a, ep["host"], interleave_p=interleave_p)
        elif kind == "diag_test":
            diag_test(a, ep["fault"], interleave_p=interleave_p)
        elif kind == "diag_http":
            diag_http(a, ep["fault"], ep["host"], interleave_p=interleave_p)
        else:
            tt = _tool_target(s, ep["fault"])
            if tt is None:
                diag_test(a, ep["fault"], interleave_p=interleave_p)
            else:
                diag_tool(a, ep["fault"], tt[0], tt[1], interleave_p=interleave_p)

    # Suffix: finish any free faults, then wrap up.
    for f in free:
        if s.room(10) and rng.random() < (0.9 if long_form else 0.6):
            a.fault_work(f, clue_p=clue_p)
    a.wrap_up()
    return s.finish()
