"""Family 1, incremental_repair (M1_CONTRACTS §3, §7)."""

from __future__ import annotations

import collections

from agent_loop_lab.schema import OracleLabel
from agent_loop_lab.synth.latent import steps_from_record
from tests import synth_helpers as H

FAMILY = "incremental_repair"
MV_ALTERNATION = [("MUTATE", "SUCCESS"), ("VALIDATE", "ERROR")] * 4


def test_family_contract_and_plurality():
    mix = H.check_family(FAMILY, intended="PRODUCTIVE")
    assert mix["STALLED"] == mix["OSCILLATING"] == 0


def test_lookalike_edit_test_alternation_with_single_revert_is_productive():
    """7 vs 1: an 8-event MUTATE-SUCCESS / VALIDATE-ERROR alternation containing one undo."""
    found = 0
    for events, record in H.sessions(FAMILY):
        steps, _ = steps_from_record(record)
        seq = [(e.operation_category.value, e.outcome.value) for e in events]
        for t in range(7, len(events)):
            if seq[t - 7 : t + 1] != MV_ALTERNATION:
                continue
            window = steps[t - 7 : t + 1]
            if sum(s.undoes_prior_mutation for s in window) == 1:
                assert record.window_labels[events[t].sequence_index] is OracleLabel.PRODUCTIVE
                found += 1
    assert found >= 1


def test_no_accidental_episodes_and_sessions_vary():
    langs, lengths, families = set(), set(), collections.Counter()
    for events, record in H.sessions(FAMILY):
        steps, _ = steps_from_record(record)
        assert all(s.open_episode_kind is None for s in steps)
        assert record.latent_trace["steps"][-1]["progress_after"] == 1.0 or len(events) >= 40
        langs.add(record.latent_trace["world"]["language"])
        lengths.add(len(events))
        families.update(e.operation_family for e in events)
    assert len(langs) >= 4 and max(lengths) - min(lengths) >= 15
    for fam in ("read_file", "search", "edit_file", "write_file", "run_tests"):
        assert families[fam] > 0, fam
