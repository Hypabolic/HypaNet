"""Family 9, improving_tests (M1_CONTRACTS §3, §7)."""

from __future__ import annotations

import collections

from agent_loop_lab.schema import EVENTS_PER_WINDOW, LatentEffect, OracleLabel
from agent_loop_lab.synth.latent import steps_from_record
from tests import synth_helpers as H

FAMILY = "improving_tests"
MV_ALTERNATION = [("MUTATE", "SUCCESS"), ("VALIDATE", "ERROR")] * 4
VM_ALTERNATION = [("VALIDATE", "ERROR"), ("MUTATE", "SUCCESS")] * 4


def test_family_contract_and_plurality():
    mix = H.check_family(FAMILY, intended="PRODUCTIVE")
    assert mix["STALLED"] == mix["OSCILLATING"] == mix["RECOVERY"] == 0


def test_each_suite_run_improves_and_core_is_never_stall_ineffective():
    for events, record in H.sessions(FAMILY):
        steps, _ = steps_from_record(record)
        meta = record.latent_trace["meta"]
        lo, hi = meta["core_start"], meta["core_end"]
        assert all(s.open_episode_kind is None for s in steps)
        # Edits and test runs in the loop are never stall-ineffective (a re-read of a file can be).
        for e, s in zip(events[lo:hi], steps[lo:hi]):
            if e.operation_category.value != "INSPECT":
                assert s.effect not in (LatentEffect.REDUNDANT, LatentEffect.REVERSION), (record.session_id, s)
        assert steps[hi - 1].progress_after == 1.0
        # Full-suite runs of the same suite have strictly increasing pass counts.
        suite_sizes = record.latent_trace["world"]["suites"]
        last_passed: dict[str, int] = {}
        for e, s in zip(events[lo:hi], steps[lo:hi]):
            if s.agent_mutation:
                assert s.effect is LatentEffect.PROGRESS  # every loop edit is a fix (any first attempt precedes core_start)
            if e.operation_family != "run_tests" or e.validation_total != suite_sizes[e.validation_suite_id]:
                continue  # not a full-suite run (a targeted selection)
            suite = e.validation_suite_id
            if suite in last_passed:
                assert e.validation_passed > last_passed[suite], (record.session_id, e.sequence_index)
            last_passed[suite] = e.validation_passed


def test_lookalike_mutate_validate_error_alternation_is_productive():
    """6 vs 9: the MUTATE-SUCCESS / VALIDATE-ERROR alternation, here with genuine fixes."""
    found = collections.Counter()
    for events, record in H.sessions(FAMILY):
        seq = [(e.operation_category.value, e.outcome.value) for e in events]
        for t in range(EVENTS_PER_WINDOW - 1, len(events)):
            w = seq[t - 7 : t + 1]
            for name, pat in (("MV", MV_ALTERNATION), ("VM", VM_ALTERNATION)):
                if w == pat:
                    assert record.window_labels[events[t].sequence_index] is OracleLabel.PRODUCTIVE
                    found[name] += 1
    assert found["MV"] >= 1 and found["VM"] >= 1, dict(found)


def test_sessions_vary():
    langs, lengths, fams, attempts, starts = set(), set(), collections.Counter(), set(), set()
    for events, record in H.sessions(FAMILY):
        langs.add(record.latent_trace["world"]["language"])
        lengths.add(len(events))
        fams.update(e.operation_family for e in events)
        meta = record.latent_trace["meta"]
        attempts.add(meta["first_attempt"])
        starts.add(meta["core_start"])
    assert len(langs) >= 4 and max(lengths) - min(lengths) >= 20
    assert attempts == {"none", "probe", "wrong"} and len(starts) >= 6
    for fam in ("read_file", "search", "edit_file", "write_file", "run_tests"):
        assert fams[fam] > 0, fam
