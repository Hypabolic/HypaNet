"""Family 2, exploration (M1_CONTRACTS §3, §7)."""

from __future__ import annotations

import collections

from agent_loop_lab.schema import LatentEffect
from agent_loop_lab.synth.latent import steps_from_record
from tests import synth_helpers as H

FAMILY = "exploration"


def test_family_contract_and_plurality():
    mix = H.check_family(FAMILY, intended="PRODUCTIVE")
    assert mix["STALLED"] == mix["OSCILLATING"] == mix["RECOVERY"] == 0


def test_investigation_yields_facts_densely_then_a_fix():
    for events, record in H.sessions(FAMILY):
        steps, _ = steps_from_record(record)
        assert all(s.effect is not LatentEffect.REDUNDANT for s in steps)
        first_fix = next(i for i, s in enumerate(steps) if s.effect is LatentEffect.PROGRESS)
        assert sum(s.new_facts for s in steps[:first_fix]) >= 4
        # Any 8 consecutive investigation ops (before the fixes) add >= 2 facts.
        for t in range(7, first_fix):
            assert sum(s.new_facts for s in steps[t - 7 : t + 1]) >= 2, (record.session_id, t)
        assert steps[-1].progress_after == 1.0 or len(events) >= 50


def test_tool_variety():
    families = collections.Counter()
    for events, _ in H.sessions(FAMILY):
        families.update(e.operation_family for e in events)
    for fam in ("read_file", "search", "list_dir", "lint", "build", "http_call", "run_tests", "edit_file"):
        assert families[fam] > 0, fam
