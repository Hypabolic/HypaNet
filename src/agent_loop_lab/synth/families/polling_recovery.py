"""Family 10, polling while a dependency recovers (spec §7.1 item 10).

Core latent behaviour: the agent waits on an external resource (CI pipeline,
rollout, migration, ...) with ``poll_status``: the first poll sees it pending,
immediate re-polls with backoff are in-policy polls, and the resource becomes
ready within ``P_max`` polls (RESOLVES_EPISODE); productive work follows.
Variants:

- ``legit``: 1-3 polling episodes, each ready within policy. Intended label
  mix: RECOVERY (rule 4) while waiting and for ``k_rec`` events after ready,
  then PRODUCTIVE (rule 5).
- ``doomed``: the dependency never becomes ready; the agent keeps polling
  well past ``P_max`` (out-of-policy polls are REDUNDANT), sometimes
  re-reading a status file it already read between polls, then gives up or
  switches to other work. STALLED (rule 3) once five or more out-of-policy
  polls fill a window.
- ``late``: the dependency becomes ready, but only after more than ``P_max``
  polls (out-of-policy waiting, then a resolution).

Variation (§7.2): language, resources, backoff schedules, 1..P_max+1 pending
polls, an optional single unrelated operation (a novel decoy, a first plan,
or a re-read of a status file) between polls, a triggering external call
before the wait, neutral or productive prefixes, the work after the wait,
give-up actions, cost profiles.
"""

from __future__ import annotations

import random

from ...oracle_config import OracleConfig
from ...schema import CanonicalEvent, OracleRecord
from ..engine import CostProfile, OpResult, Session
from ..world import Fault
from .transient_external import METHODS, Actor, pick_hosts, request_body

FAMILY = "polling_recovery"


def _backoffs(rng: random.Random) -> list[int]:
    style = rng.choice(("exp", "linear", "fixed", "none"))
    if style == "exp":
        base = rng.choice((1, 2))
        return [min(120, base * 2 ** i) for i in range(40)]
    if style == "linear":
        step = rng.choice((2, 5, 10))
        return [step * (i + 1) for i in range(40)]
    if style == "fixed":
        return [rng.choice((5, 10, 15, 30))] * 40
    return [0] * 40


class _Waiter:
    """Polls one resource; may place one unrelated op between two polls."""

    def __init__(self, a: Actor, resource: str, *, interleave_p: float, status_path: str | None) -> None:
        self.a = a
        self.resource = resource
        self.interleave_p = interleave_p
        self.status_path = status_path  # a file already read once: re-reading it is REDUNDANT
        self.backoff = _backoffs(a.rng)
        self.cost = a.rng.choice((None, 0.5, 0.7))
        self.polls = 0

    def poll(self) -> OpResult:
        s, rng = self.a.s, self.a.rng
        b = self.backoff[min(self.polls, len(self.backoff) - 1)]
        self.polls += 1
        return s.poll(self.resource, backoff=b if b else None, cost=self.cost)

    def interleave(self) -> None:
        a = self.a
        if self.status_path is not None and a.rng.random() < 0.6:
            a.s.read(self.status_path)
        elif a.rng.random() < 0.3 and a.plan() is not None:
            pass
        else:
            a.decoy(("read", "search", "list"))

    def wait(self, max_polls: int) -> OpResult:
        """Poll until ready or ``max_polls`` polls were made."""
        r = self.poll()
        while r.not_ready and self.polls < max_polls and self.a.s.room(2):
            if self.polls >= 2 and self.a.rng.random() < self.interleave_p and self.a.s.room(3):
                self.interleave()
            r = self.poll()
        return r


def _status_file(a: Actor) -> str | None:
    """Read a decoy file now (novel, NONE) so later re-reads of it are REDUNDANT."""
    paths = [p for p in a.s.world.decoy_paths() if p.endswith((".log", ".md", ".yml", ".yaml", ".sh", "Makefile"))]
    if not paths:
        return None
    p = a.rng.choice(sorted(paths))
    a.s.read(p)
    return p


def _work_after(a: Actor, faults: list[Fault], *, clue_p: float) -> None:
    """Productive work right after the dependency is ready."""
    for f in faults:
        if not a.s.room(8):
            return
        a.fault_work(f, clue_p=clue_p)


def generate(rng: random.Random, session_id: str, cfg: OracleConfig) -> tuple[list[CanonicalEvent], OracleRecord]:
    variant = rng.choices(("legit", "doomed", "late"), (0.62, 0.28, 0.10))[0]
    n_waits = rng.choices((1, 2, 3), (0.45, 0.4, 0.15))[0] if variant == "legit" else rng.choice((1, 1, 2))
    n_faults = min(4, n_waits + rng.choice((0, 0, 1, 1, 2)))
    s = Session(
        rng, session_id, cfg, family=FAMILY, n_faults=n_faults, extra_clues=rng.choice(((0, 1), (0, 2), (1, 2))),
        single_suite=rng.random() < 0.5, cost_profile=CostProfile.draw(rng, scale=rng.lognormvariate(0.0, 0.5)),
    )
    w = s.world
    resources = list(w.vocab.resources)
    rng.shuffle(resources)
    resources = resources[:n_waits]
    # The last wait carries the variant's behaviour; earlier ones are legitimate.
    readies: list[int | None] = [rng.randint(1, cfg.p_max + 1) for _ in resources]
    if variant == "doomed":
        readies[-1] = None
    elif variant == "late":
        readies[-1] = cfg.p_max + rng.randint(2, 6)
    for res, k in zip(resources, readies):
        w.add_resource(res, ready_after=k)
    trigger_hosts = pick_hosts(s, rng, n_waits)
    a = Actor(s, rng, trigger_hosts)
    interleave_p = rng.choice((0.0, 0.0, 0.25, 0.5)) if variant == "legit" else rng.choice((0.0, 0.3, 0.6))

    faults = list(w.faults.values())
    rng.shuffle(faults)
    after: list[list[Fault]] = [[] for _ in resources]
    for i in range(len(resources)):
        if faults and (variant == "legit" or i < len(resources) - 1):
            after[i].append(faults.pop(0))
    prefix_fault = faults.pop(0) if faults and rng.random() < 0.45 else None
    for f in faults:
        legit_slots = [i for i in range(len(resources)) if readies[i] is not None]
        if legit_slots:
            after[rng.choice(legit_slots)].append(f)
    s.meta.update(variant=variant, n_waits=n_waits, n_faults=n_faults, interleave_p=interleave_p,
                  waits=[{"resource": r_, "ready_after": k} for r_, k in zip(resources, readies)],
                  prefix="productive" if prefix_fault is not None else "neutral")

    # Prefix (>= 3 events before the first poll).
    if prefix_fault is not None:
        a.neutral(rng.randint(0, 2))
        a.fault_work(prefix_fault, verify=rng.choice(("targeted", "suite", "none")))
    a.neutral(max(0, rng.randint(3, 7) - s.n))

    for i, (res, k) in enumerate(zip(resources, readies)):
        if i > 0 and rng.random() < 0.5:
            a.neutral(rng.randint(1, 2))
        status = _status_file(a) if (k is None or k > cfg.p_max + 1) and rng.random() < 0.6 else None
        if rng.random() < 0.4:
            # Kick off the job being waited on (an ordinary successful external call, NONE).
            m = rng.choice(("POST", "POST", "PUT") + METHODS[:1])
            s.http(m, trigger_hosts[i], rng.choice(w.vocab.path_templates), body=request_body(rng, m))
        waiter = _Waiter(a, res, interleave_p=interleave_p, status_path=status)
        if k is None:
            # Doomed: keep polling well past P_max, then give up or move on.
            r = waiter.wait(cfg.p_max + 1 + rng.randint(6, 16))
            roll = rng.random()
            if roll < 0.4:
                s.yield_(rng.choice(("blocked", "handoff")))
            elif roll < 0.6:
                s.ask(rng.choice(("request_credentials", "confirm_scope", "clarify_requirement")))
            elif faults_left := [f for f in w.unfixed_faults()]:
                _work_after(a, faults_left[:1], clue_p=0.4)
        else:
            r = waiter.wait(k + 1)
            if r.ok:
                _work_after(a, after[i], clue_p=rng.choice((0.2, 0.5, 0.8)))
        if not s.room(10):
            break

    a.wrap_up()
    return s.finish()
