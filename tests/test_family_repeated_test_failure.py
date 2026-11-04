"""Family 5, repeated_test_failure (M1_CONTRACTS §3, §7)."""

from __future__ import annotations

import collections

from agent_loop_lab.schema import LatentEffect, OracleLabel
from agent_loop_lab.synth.latent import steps_from_record
from tests import synth_helpers as H

FAMILY = "repeated_test_failure"
VALIDATE_ERRORS = [("VALIDATE", "ERROR")] * 8
REREAD_ALTERNATION = [("INSPECT", "SUCCESS"), ("VALIDATE", "ERROR")] * 4


def test_family_contract_and_plurality():
    mix = H.check_family(FAMILY, intended="STALLED")
    assert mix["OSCILLATING"] == 0
    assert mix["RECOVERY"] > 0


def test_no_edits_and_the_failure_never_changes():
    for events, record in H.sessions(FAMILY):
        steps, p0 = steps_from_record(record)
        assert all(e.operation_category.value != "MUTATE" for e in events)
        assert all(s.progress_after == p0 for s in steps)
        assert len({s.ws_after for s in steps}) == 1  # workspace never changes
        failures = {e.failure_fingerprint for e in events if e.operation_family == "run_tests" and e.outcome.value == "ERROR"}
        assert len(failures) == 1, record.session_id
        assert LatentEffect.RESOLVES_EPISODE not in record.event_effects


def test_recovery_within_policy_then_stalled_beyond_it():
    c = H.cfg()
    stalled_sessions = 0
    for events, record in H.sessions(FAMILY):
        steps, _ = steps_from_record(record)
        labels = record.window_labels
        order = [labels[e.sequence_index] for e in events[7:]]
        assert OracleLabel.RECOVERY in order, record.session_id
        retries = [s for s in steps if s.effect is LatentEffect.IN_POLICY_RETRY]
        assert [s.open_episode_attempts for s in retries] == list(range(1, c.r_max + 1))
        for e, s in zip(events, steps):
            if labels.get(e.sequence_index) is OracleLabel.RECOVERY:
                assert s.open_episode_kind == "retry" and s.open_episode_attempts <= c.r_max
        if OracleLabel.STALLED in order:
            stalled_sessions += 1
            assert order.index(OracleLabel.RECOVERY) < order.index(OracleLabel.STALLED)
            # The first stalled window comes after the last in-policy retry.
            last_retry = max(i for i, s in enumerate(steps) if s.effect is LatentEffect.IN_POLICY_RETRY)
            first_stall = next(i for i, e in enumerate(events) if labels.get(e.sequence_index) is OracleLabel.STALLED)
            assert first_stall > last_retry
    assert stalled_sessions >= 45


def test_lookalike_repeated_validation_errors_past_r_max_are_stalled():
    """5 vs 4: repeated VALIDATE-ERROR retries; here they never resolve."""
    found = collections.Counter()
    for events, record in H.sessions(FAMILY):
        seq = [(e.operation_category.value, e.outcome.value) for e in events]
        for t in range(7, len(events)):
            window = seq[t - 7 : t + 1]
            label = record.window_labels[events[t].sequence_index]
            for name, pattern in (("errors", VALIDATE_ERRORS), ("reread", REREAD_ALTERNATION)):
                if window == pattern:
                    found[(name, label.value)] += 1
    assert found[("errors", "STALLED")] >= 1 and found[("reread", "STALLED")] >= 1, dict(found)
    # Eight consecutive failing runs always include attempts beyond R_max: never RECOVERY or PRODUCTIVE.
    assert {lab for name, lab in found if name == "errors"} <= {"STALLED", "INSUFFICIENT_AMBIGUOUS"}, dict(found)


def test_sessions_vary():
    langs, lengths, families, shapes = set(), set(), collections.Counter(), collections.Counter()
    for events, record in H.sessions(FAMILY):
        langs.add(record.latent_trace["world"]["language"])
        lengths.add(len(events))
        families.update(e.operation_family for e in events)
        meta = record.latent_trace["meta"]
        shapes.update([meta["prefix"], meta["tail"], f"sel:{meta['selection']}"])
    assert len(langs) >= 4 and max(lengths) - min(lengths) >= 20
    for key in ("neutral", "investigate", "baseline", "continuous", "spaced", "sel:full"):
        assert shapes[key] > 0, key
    for fam in ("read_file", "search", "list_dir", "run_tests", "plan"):
        assert families[fam] > 0, fam
