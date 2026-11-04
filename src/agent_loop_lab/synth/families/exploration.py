"""Family 2, productive exploration (spec §7.1 item 2).

Core latent behaviour: a run of inspections and diagnostics (reads, searches,
directory listings, lint/build, observability HTTP GETs, targeted failing test
runs) that keeps adding facts to ``K``, at least 2 per 8 operations, with at
most one fruitless inspection per fact, followed by the meaningful fix edits.
Intended label: PRODUCTIVE (rule 5, F >= F_prod while investigating, then
Δp > ε while fixing).

Variation: language, 1-4 faults with 2-5 cause clues each, depth-first or
interleaved investigation order, decoy ratio, diagnostic style (search vs a
targeted failing run to locate), edit vs write for fixes, prefix and suffix.
"""

from __future__ import annotations

import random

from ...oracle_config import OracleConfig
from ...schema import CanonicalEvent, OracleRecord
from ..engine import MAX_EVENTS, MIN_EVENTS, Session, fix
from ..world import Clue, Fault

FAMILY = "exploration"


def _locate(s: Session, rng: random.Random, f: Fault, diag_runs: float) -> None:
    """Reveal the fault's location: a targeted failing run (a diagnostic) or a symbol search."""
    if rng.random() < diag_runs:
        s.run_tests(f.suite, list(f.tests))
    else:
        scope = f.path.split("/", 1)[0] if rng.random() < 0.7 else "."
        s.search(scope, f.symbol)


def generate(rng: random.Random, session_id: str, cfg: OracleConfig) -> tuple[list[CanonicalEvent], OracleRecord]:
    n_faults = rng.choice((1, 2, 2, 3, 3, 4))
    s = Session(rng, session_id, cfg, family=FAMILY, n_faults=n_faults,
                extra_clues=(3, 5) if n_faults == 1 else (2, 4))
    w = s.world
    decoy_rate = rng.uniform(0.1, 0.6)
    diag_runs = rng.choice((0.0, 0.3, 0.6))
    order = rng.choice(("depth", "interleaved"))
    s.meta.update(n_faults=n_faults, decoy_rate=round(decoy_rate, 3), diag_runs=diag_runs, order=order)

    # Prefix: orientation (0-3 ops, never more than one fruitless op before the first fact).
    if rng.random() < 0.5:
        s.plan(rng.choice(w.vocab.plan_actions))
    if rng.random() < 0.6:
        s.list_dir(".")
    for _ in range(rng.randint(0, 1)):
        s.novel_decoy(kinds=("read", "search"))

    # Investigation agenda: every fact of every fault, in depth-first or interleaved order.
    faults = list(w.faults.values())
    agenda: list[tuple[Fault, str]] = []
    for f in faults:
        items = [(f, f.loc_fact), (f, f.code_fact)]
        extras = [(f, fact) for fact in f.facts[2:]]
        rng.shuffle(extras)
        agenda.extend(items + extras)
    if order == "interleaved":
        rng.shuffle(agenda)

    reserve = 2 * n_faults + 3  # fixes + verification + suffix
    for f, fact in agenda:
        if s.knows(fact):
            continue  # revealed on the way (e.g. a lint run that reported two findings)
        if s.n + 2 + reserve > MAX_EVENTS:
            break
        if fact == f.loc_fact:
            _locate(s, rng, f, diag_runs)
        elif fact == f.code_fact:
            s.read(f.path, block=f.block, pad=rng.randint(0, 10))
        else:
            clue: Clue = next(c for c in w.clues_for(f.id) if c.fact == fact)
            s.inspect_clue(clue)
        if rng.random() < decoy_rate and s.n + 1 + reserve <= MAX_EVENTS:
            s.novel_decoy()

    # The meaningful change, then verification.
    if rng.random() < 0.4:
        s.plan(rng.choice(("summarise_findings", "list_hypotheses", "verify_plan")))
    for f in faults:
        if rng.random() < 0.2:
            s.write(f.path, fix(f))
        else:
            s.edit(f.path, fix(f))
    for suite in dict.fromkeys(f.suite for f in faults):
        s.run_tests(suite)

    # Suffix.
    while s.n < MIN_EVENTS:
        s.novel_decoy()
    if rng.random() < 0.4 and s.room(1):
        s.yield_(rng.choice(("done", "handoff")))
    return s.finish()
