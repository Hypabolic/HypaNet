"""Family 3, transient_external (M1_CONTRACTS §3, §7)."""

from __future__ import annotations

import collections

from agent_loop_lab.schema import LatentEffect, OracleLabel
from agent_loop_lab.synth.latent import steps_from_record
from tests import synth_helpers as H

FAMILY = "transient_external"
E = LatentEffect


def test_family_contract_and_plurality():
    mix = H.check_family(FAMILY, intended="RECOVERY")
    assert mix["STALLED"] == mix["OSCILLATING"] == 0
    assert mix["PRODUCTIVE"] > 0


def test_outages_are_in_policy_http_retries_that_resolve():
    c = H.cfg()
    for events, record in H.sessions(FAMILY):
        steps, _ = steps_from_record(record)
        n_outages = record.latent_trace["meta"]["n_outages"]
        res = [i for i, s in enumerate(steps) if s.resolved_episode_here]
        assert len(res) == n_outages, record.session_id
        assert all(events[i].operation_family == "http_call" and events[i].outcome.value == "SUCCESS" for i in res)
        for s in steps:
            assert s.open_episode_kind in (None, "retry")
            assert s.open_episode_attempts <= c.r_max  # every failing retry is in policy
        assert steps[-1].open_episode_kind is None
        # The initial failure of each outage is not an attempt (NONE), and no attempt is REDUNDANT.
        assert not any(s.effect is E.REDUNDANT and events[i].operation_family == "http_call" for i, s in enumerate(steps))


def test_recovery_then_productive_after_resolution():
    c = H.cfg()
    shows_productive_after = 0
    sessions = H.sessions(FAMILY)
    for events, record in sessions:
        steps, _ = steps_from_record(record)
        labels = [record.window_labels.get(e.sequence_index) for e in events]
        res = [i for i, s in enumerate(steps) if s.resolved_episode_here]
        assert any(lab is OracleLabel.RECOVERY for lab in labels), record.session_id
        for r in res:
            for t in range(max(r, 7), min(r + c.k_rec, len(events))):
                assert labels[t] is OracleLabel.RECOVERY, (record.session_id, r, t)
        last = res[-1]
        if any(labels[t] is OracleLabel.PRODUCTIVE for t in range(last + c.k_rec, len(events))):
            shows_productive_after += 1
    assert shows_productive_after >= 0.8 * len(sessions)


def test_variation():
    langs, lengths, methods, errors, fails, first = set(), set(), set(), set(), set(), set()
    families = collections.Counter()
    obs = obs_facts = interleaved = 0
    for events, record in H.sessions(FAMILY):
        steps, _ = steps_from_record(record)
        meta = record.latent_trace["meta"]
        langs.add(record.latent_trace["world"]["language"])
        lengths.add(len(events))
        families.update(e.operation_family for e in events)
        for o in meta["outages"]:
            methods.add(o["method"])
            fails.add(o["fail_attempts"])
            obs += o["obs"]
        errors.update(e.failure_fingerprint for e in events if e.operation_family == "http_call" and e.failure_fingerprint)
        # Where each outage starts: its initial failure (not an attempt of an open episode).
        first.update(i for i, e in enumerate(events) if e.operation_family == "http_call" and e.outcome.value == "ERROR"
                     and steps[i].effect is E.NONE)
        # An unrelated operation between two retries of one open episode.
        interleaved += any(
            s.open_episode_kind == "retry" and events[i].operation_family != "http_call" for i, s in enumerate(steps)
        )
        # An observability endpoint's outage resolves with the clue facts it serves.
        obs_facts += any(s.resolved_episode_here and s.new_facts > 0 for s in steps)
    assert len(langs) >= 4 and max(lengths) - min(lengths) >= 25
    assert len(methods) >= 4 and len(errors) >= 10 and fails == set(range(1, H.cfg().r_max + 2))
    assert max(first) - min(first) >= 20, "outages must sit at varied positions"
    assert obs >= 1 and obs_facts >= 1 and interleaved >= 1
    for fam in ("read_file", "search", "edit_file", "run_tests", "http_call"):
        assert families[fam] > 0, fam
