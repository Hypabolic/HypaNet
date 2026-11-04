"""Family 10, polling_recovery (M1_CONTRACTS §3, §7)."""

from __future__ import annotations

import collections

from agent_loop_lab.schema import LatentEffect, OracleLabel
from agent_loop_lab.synth.latent import steps_from_record
from tests import synth_helpers as H

FAMILY = "polling_recovery"
E = LatentEffect


def _labels(events, record):
    return [record.window_labels.get(e.sequence_index) for e in events]


def test_family_contract_and_plurality():
    mix = H.check_family(FAMILY, intended="RECOVERY")
    assert mix["OSCILLATING"] == 0
    assert mix["STALLED"] > 0 and mix["PRODUCTIVE"] > 0


def test_legitimate_waits_resolve_within_policy_then_work():
    c = H.cfg()
    legit = productive_after = 0
    for events, record in H.sessions(FAMILY):
        meta = record.latent_trace["meta"]
        if meta["variant"] != "legit":
            continue
        legit += 1
        steps, _ = steps_from_record(record)
        labels = _labels(events, record)
        res = [i for i, s in enumerate(steps) if s.resolved_episode_here]
        assert all(events[i].operation_family == "poll_status" for i in res)
        assert len(res) == sum(1 for wt in meta["waits"] if wt["ready_after"] > 0)
        assert OracleLabel.STALLED not in labels
        for i, s in enumerate(steps):
            assert s.open_episode_kind in (None, "poll")
            assert s.open_episode_attempts <= c.p_max
            assert not (events[i].operation_family == "poll_status" and s.effect is E.REDUNDANT)
        for r in res:
            for t in range(max(r, 7), min(r + c.k_rec, len(events))):
                assert labels[t] is OracleLabel.RECOVERY
        if any(lab is OracleLabel.PRODUCTIVE for lab in labels[res[-1] + c.k_rec :]):
            productive_after += 1
    assert legit >= 20 and productive_after >= 0.75 * legit


def test_doomed_waits_go_past_p_max_and_stall():
    c = H.cfg()
    doomed = late = 0
    for events, record in H.sessions(FAMILY):
        meta = record.latent_trace["meta"]
        steps, _ = steps_from_record(record)
        labels = _labels(events, record)
        last_wait = meta["waits"][-1]
        polls = [i for i, e in enumerate(events) if e.operation_family == "poll_status"]
        if meta["variant"] == "doomed":
            doomed += 1
            assert last_wait["ready_after"] is None
            assert max(s.open_episode_attempts for s in steps) > c.p_max + 4
            assert OracleLabel.STALLED in labels, record.session_id
            # The doomed resource never resolves: every resolution belongs to an earlier, legitimate wait.
            assert sum(s.resolved_episode_here for s in steps) == len(meta["waits"]) - 1
            # RECOVERY while in policy, never once the polls are out of policy.
            for t in range(7, len(events)):
                s = steps[t]
                if s.open_episode_kind == "poll" and s.open_episode_attempts > c.p_max and not any(
                    x.resolved_episode_here for x in steps[t - c.k_rec + 1 : t + 1]
                ):
                    assert labels[t] is not OracleLabel.RECOVERY
            assert events[polls[-1]].outcome.value == "NEUTRAL"
        elif meta["variant"] == "late":
            late += 1
            assert max(s.open_episode_attempts for s in steps) > c.p_max
            assert steps[polls[-1]].resolved_episode_here
    assert doomed >= 5 and late >= 1


def test_variation():
    langs, lengths, resources, first, ready = set(), set(), set(), set(), set()
    families = collections.Counter()
    interleaved = triggered = 0
    for events, record in H.sessions(FAMILY):
        steps, _ = steps_from_record(record)
        meta = record.latent_trace["meta"]
        langs.add(record.latent_trace["world"]["language"])
        lengths.add(len(events))
        families.update(e.operation_family for e in events)
        resources.update(wt["resource"] for wt in meta["waits"])
        ready.update(wt["ready_after"] for wt in meta["waits"] if wt["ready_after"] is not None)
        first.add(next(i for i, e in enumerate(events) if e.operation_family == "poll_status"))
        interleaved += any(s.open_episode_kind == "poll" and events[i].operation_family != "poll_status"
                           for i, s in enumerate(steps))
        triggered += any(e.operation_family == "http_call" for e in events)
    c = H.cfg()
    assert len(langs) >= 4 and max(lengths) - min(lengths) >= 30 and len(resources) >= 30
    assert set(range(1, c.p_max + 2)) <= ready
    assert max(first) - min(first) >= 6 and interleaved >= 5 and triggered >= 5
    for fam in ("read_file", "search", "edit_file", "run_tests", "poll_status"):
        assert families[fam] > 0, fam
