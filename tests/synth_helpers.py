"""Shared assertions for generator family tests (M1_CONTRACTS §7).

Typical family test module::

    from tests import synth_helpers as H

    FAMILY = "incremental_repair"

    def test_contract():
        H.check_family(FAMILY, intended="PRODUCTIVE")

or call the individual helpers. Seeds come from :func:`seeds` so every family
test uses the same derivation as the dataset builder (sha256 of a label).
"""

from __future__ import annotations

import collections
import hashlib
import json
from typing import Iterable, Sequence

from agent_loop_lab.oracle_config import OracleConfig, load_oracle_config
from agent_loop_lab.schema import FORBIDDEN_INFERENCE_FIELDS, CanonicalEvent, OracleLabel, OracleRecord
from agent_loop_lab.synth.engine import MAX_EVENTS, MIN_EVENTS, PROVENANCE_SOURCE
from agent_loop_lab.synth.families import generate_session
from agent_loop_lab.synth.labeller import label_session, validate_steps
from agent_loop_lab.synth.latent import steps_from_record
from agent_loop_lab.windows import iter_windows, validate_sequence

N_SEEDS = 50


def cfg() -> OracleConfig:
    return load_oracle_config()


def seeds(family: str, n: int = N_SEEDS, salt: str = "tests") -> list[int]:
    return [int.from_bytes(hashlib.sha256(f"{salt}:{family}:{k}".encode()).digest()[:8], "big") for k in range(n)]


def generate(family: str, seed: int, session_id: str | None = None) -> tuple[list[CanonicalEvent], OracleRecord]:
    return generate_session(family, seed, session_id or f"{family}-{seed:016x}", cfg())


def sessions(family: str, n: int = N_SEEDS) -> list[tuple[list[CanonicalEvent], OracleRecord]]:
    return [generate(family, sd) for sd in seeds(family, n)]


# -- individual assertions ---------------------------------------------------


def _record_json(record: OracleRecord) -> str:
    return json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))


def assert_deterministic(family: str, seed_list: Sequence[int]) -> None:
    """Same seed -> byte-identical events and records (two fresh generations)."""
    for sd in seed_list:
        ev1, rec1 = generate(family, sd)
        ev2, rec2 = generate(family, sd)
        assert [e.to_json() for e in ev1] == [e.to_json() for e in ev2], f"{family} seed {sd}: events differ"
        assert _record_json(rec1) == _record_json(rec2), f"{family} seed {sd}: records differ"


def assert_record_consistent(events: Sequence[CanonicalEvent], record: OracleRecord, family: str, seed: int | None = None) -> None:
    """Labels recomputed from latent_trace equal the stored labels; steps validate; effects align."""
    c = cfg()
    steps, p0 = steps_from_record(record)
    validate_steps(steps, c)
    assert record.scenario_family == family
    if seed is not None:
        assert record.seed == seed
    assert len(steps) == len(events) == len(record.event_effects)
    assert [s.sequence_index for s in steps] == [e.sequence_index for e in events]
    assert tuple(s.effect for s in steps) == record.event_effects
    assert label_session(steps, p0, c) == dict(record.window_labels)
    assert sorted(record.dropped_observations) == [s.sequence_index for s in steps if s.dropped]
    json.dumps(record.to_dict())  # serialisable
    assert OracleRecord.from_dict(json.loads(_record_json(record))).to_dict() == record.to_dict()


def assert_events_valid(events: Sequence[CanonicalEvent]) -> None:
    """Every event round-trips CanonicalEvent validation; the list is one valid stream."""
    assert MIN_EVENTS <= len(events) <= MAX_EVENTS, f"length {len(events)}"
    for e in events:
        assert CanonicalEvent.from_json(e.to_json()) == e
        assert e.provenance.source == PROVENANCE_SOURCE
    validate_sequence(events)
    ts = [e.timestamp for e in events]
    assert ts == sorted(ts) and len(set(ts)) == len(ts), "timestamps must be strictly increasing"
    for e in events:
        if e.operation_category.value == "INSPECT" and e.workspace_state_before is not None and e.workspace_state_after is not None:
            assert e.workspace_state_before == e.workspace_state_after, "INSPECT must report before == after"


def assert_no_oracle_fields(events: Iterable[CanonicalEvent]) -> None:
    for e in events:
        d = e.to_dict()
        leaked = set(d) & FORBIDDEN_INFERENCE_FIELDS
        assert not leaked, f"oracle fields on event: {sorted(leaked)}"
        assert not (set(d["provenance"]) & FORBIDDEN_INFERENCE_FIELDS)


def assert_digests_match_latent(events: Sequence[CanonicalEvent], record: OracleRecord) -> None:
    """Observed digests equal the latent workspace ids wherever they are not dropped."""
    steps, _ = steps_from_record(record)
    for e, s in zip(events, steps):
        if e.workspace_state_before is not None:
            assert e.workspace_state_before == s.ws_before
        if e.workspace_state_after is not None:
            assert e.workspace_state_after == s.ws_after


def gate_agreement(events: Sequence[CanonicalEvent], record: OracleRecord) -> tuple[int, int, list[int]]:
    """``(windows, gate_passed, disagreements)``; a disagreement is a window
    where gate rejection != oracle INSUFFICIENT_INSTRUMENTATION."""
    labels = record.window_labels
    n = passed = 0
    bad: list[int] = []
    for w in iter_windows(events):
        n += 1
        passed += w.gate.passed
        instr = labels[w.endpoint_index] is OracleLabel.INSUFFICIENT_INSTRUMENTATION
        if w.gate.passed == instr:
            bad.append(w.endpoint_index)
    return n, passed, bad


def assert_gate_all_pass(events: Sequence[CanonicalEvent], record: OracleRecord) -> None:
    n, passed, bad = gate_agreement(events, record)
    assert passed == n, f"gate rejected {n - passed}/{n} windows"
    assert not bad


def assert_gate_matches_drops(events: Sequence[CanonicalEvent], record: OracleRecord) -> None:
    """Family 12: the gate rejects exactly the windows containing a dropped event."""
    _, _, bad = gate_agreement(events, record)
    assert not bad, f"gate/oracle disagree at endpoints {bad}"


def label_mix(pairs: Iterable[tuple[list[CanonicalEvent], OracleRecord]]) -> collections.Counter:
    """Pooled label counts over all windows of the given sessions (all labels)."""
    mix: collections.Counter = collections.Counter()
    for _, rec in pairs:
        mix.update(v.value for v in rec.window_labels.values())
    return mix


def scored_mix(mix: collections.Counter) -> collections.Counter:
    return collections.Counter({k: v for k, v in mix.items() if OracleLabel(k).is_scored})


def assert_plurality(mix: collections.Counter, intended: str) -> None:
    scored = scored_mix(mix)
    assert scored, "no scored windows"
    top = max(scored.values())
    assert scored[intended] == top and list(scored.values()).count(top) == 1, f"plurality is not {intended}: {dict(scored)}"


# -- all-in-one ----------------------------------------------------------------


def check_family(family: str, intended: str, *, n: int = N_SEEDS, gate: str = "all_pass") -> collections.Counter:
    """Run every §7 family check over ``n`` seeds; returns the pooled label mix.

    ``gate``: ``"all_pass"`` (families 1-11) or ``"drops"`` (family 12).
    """
    sds = seeds(family, n)
    assert_deterministic(family, sds[:3])
    pairs = []
    for sd in sds:
        events, record = generate(family, sd)
        assert_events_valid(events)
        assert_no_oracle_fields(events)
        assert_record_consistent(events, record, family, seed=sd)
        assert_digests_match_latent(events, record)
        if gate == "all_pass":
            assert_gate_all_pass(events, record)
        else:
            assert_gate_matches_drops(events, record)
        pairs.append((events, record))
    mix = label_mix(pairs)
    assert_plurality(mix, intended)
    return mix
