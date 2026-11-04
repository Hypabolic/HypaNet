"""Family 4, bounded_retry (M1_CONTRACTS §3, §7)."""

from __future__ import annotations

import collections
import importlib.util

import pytest

from agent_loop_lab.schema import LatentEffect, OracleLabel
from agent_loop_lab.synth.latent import steps_from_record
from tests import synth_helpers as H

FAMILY = "bounded_retry"
E = LatentEffect


def _cat_out(events, t):
    return tuple((e.operation_category.value, e.outcome.value) for e in events[t - 7 : t + 1])


def test_family_contract_and_plurality():
    mix = H.check_family(FAMILY, intended="RECOVERY")
    assert mix["STALLED"] == mix["OSCILLATING"] == 0


def test_every_episode_is_bounded_and_resolves():
    c = H.cfg()
    for events, record in H.sessions(FAMILY):
        steps, _ = steps_from_record(record)
        assert any(s.resolved_episode_here for s in steps), record.session_id
        assert steps[-1].open_episode_kind is None
        for i, s in enumerate(steps):
            assert s.open_episode_kind in (None, "retry")
            assert s.open_episode_attempts <= 2, "at most two failing retries before the resolution"
            assert s.effect is not E.REDUNDANT or s.open_episode_kind is None
            if s.resolved_episode_here:
                prev_attempts = steps[i - 1].open_episode_attempts if steps[i - 1].open_episode_kind else 0
                assert prev_attempts + 1 <= c.r_max
        # Every opened episode resolves: an episode opens where attempts go 0 -> 1, or at a resolving
        # attempt 1; count resolutions against openings.
        opened = sum(1 for i, s in enumerate(steps) if s.open_episode_attempts == 1 and steps[i - 1].open_episode_attempts != 1)
        resolved_at_1 = sum(1 for i, s in enumerate(steps) if s.resolved_episode_here and not steps[i - 1].open_episode_kind)
        assert opened + resolved_at_1 == sum(s.resolved_episode_here for s in steps), record.session_id


def test_variants_present_including_diagnosed_repair():
    resolved_by = collections.Counter()
    repairs = collections.Counter()
    kinds = collections.Counter()
    for events, record in H.sessions(FAMILY):
        steps, _ = steps_from_record(record)
        kinds.update(ep["kind"] for ep in record.latent_trace["meta"]["episodes"])
        resolved_by.update(events[i].operation_family for i, s in enumerate(steps) if s.resolved_episode_here)
        # Diagnosed repair: a fix (PROGRESS) while the retry episode stays open, then the resolving retry.
        for i, s in enumerate(steps):
            if s.effect is E.PROGRESS and s.open_episode_kind == "retry":
                nxt = next(j for j in range(i + 1, len(steps)) if steps[j].open_episode_kind != "retry" or j == len(steps) - 1)
                assert steps[nxt].resolved_episode_here, record.session_id
                repairs[events[nxt].operation_family] += 1
    assert set(kinds) == {"transient_http", "diag_test", "diag_http", "diag_tool"}
    assert resolved_by["http_call"] > 0 and resolved_by["run_tests"] > 0 and resolved_by["lint"] + resolved_by["build"] > 0
    assert repairs["run_tests"] > 0 and repairs["http_call"] > 0


def test_contiguous_validate_and_external_error_retries_resolve():
    """Family 4's side of the 5-vs-4 lookalike: runs of VALIDATE-ERROR / EXTERNAL-ERROR that are RECOVERY."""
    found = collections.Counter()
    for events, record in H.sessions(FAMILY):
        for t in range(7, len(events)):
            tail = _cat_out(events, t)[-3:]
            if record.window_labels[events[t].sequence_index] is not OracleLabel.RECOVERY:
                continue
            for cat in ("VALIDATE", "EXTERNAL"):
                if all(x == (cat, "ERROR") for x in tail):
                    found[cat] += 1
            if all(x == ("VALIDATE", "ERROR") for x in _cat_out(events, t)):
                found["VALIDATE_x8"] += 1
    assert found["VALIDATE"] > 0 and found["EXTERNAL"] > 0 and found["VALIDATE_x8"] > 0


@pytest.mark.skipif(
    importlib.util.find_spec("agent_loop_lab.synth.families.repeated_test_failure") is None,
    reason="family 5 not written yet",
)
def test_lookalike_with_repeated_test_failure():
    """5 vs 4: a shared 8-event category/outcome string is STALLED in 5 and RECOVERY in 4."""
    rec4 = {_cat_out(ev, t) for ev, r in H.sessions(FAMILY) for t in range(7, len(ev))
            if r.window_labels[ev[t].sequence_index] is OracleLabel.RECOVERY}
    st5 = {_cat_out(ev, t) for ev, r in H.sessions("repeated_test_failure") for t in range(7, len(ev))
           if r.window_labels[ev[t].sequence_index] is OracleLabel.STALLED}
    assert rec4 & st5


def test_variation():
    langs, lengths, errors, first = set(), set(), set(), set()
    families = collections.Counter()
    for events, record in H.sessions(FAMILY):
        langs.add(record.latent_trace["world"]["language"])
        lengths.add(len(events))
        families.update(e.operation_family for e in events)
        errors.update(e.failure_fingerprint for e in events if e.failure_fingerprint)
        first.add(next(i for i, e in enumerate(events) if e.outcome.value == "ERROR"))
    assert len(langs) >= 4 and max(lengths) - min(lengths) >= 20 and len(errors) >= 30
    assert max(first) - min(first) >= 5
    for fam in ("read_file", "search", "edit_file", "write_file", "run_tests", "http_call", "lint", "build"):
        assert families[fam] > 0, fam
