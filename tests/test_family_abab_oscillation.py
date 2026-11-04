"""Family 7, abab_oscillation (M1_CONTRACTS §3, §7)."""

from __future__ import annotations

import collections

from agent_loop_lab.schema import EVENTS_PER_WINDOW, LatentEffect, OracleLabel
from agent_loop_lab.synth.labeller import window_stats
from agent_loop_lab.synth.latent import steps_from_record
from tests import synth_helpers as H

FAMILY = "abab_oscillation"
MV_ALTERNATION = [("MUTATE", "SUCCESS"), ("VALIDATE", "ERROR")] * 4


def test_family_contract_and_plurality():
    mix = H.check_family(FAMILY, intended="OSCILLATING")
    scored = H.scored_mix(mix)
    assert scored["OSCILLATING"] > sum(scored.values()) / 2


def test_core_windows_have_flat_progress_and_oscillate():
    """Windows inside the A/B core: Δp <= eps, and every one with >= 2 reversions is OSCILLATING."""
    c = H.cfg()
    n_core_windows = n_osc = 0
    for events, record in H.sessions(FAMILY):
        steps, p0 = steps_from_record(record)
        meta = record.latent_trace["meta"]
        lo, hi = meta["core_start"], meta["core_end"]
        assert hi - lo >= 4
        for t in range(max(lo + EVENTS_PER_WINDOW - 1, EVENTS_PER_WINDOW - 1), hi):
            st = window_stats(steps, p0, t, c)
            assert st.delta_p <= c.eps_p, (record.session_id, t)
            n_core_windows += 1
            if st.reversions >= c.n_rev:
                assert record.window_labels[events[t].sequence_index] is OracleLabel.OSCILLATING
                n_osc += 1
        # Competing alternatives never move progress and never open an episode.
        assert all(s.open_episode_kind is None for s in steps)
        assert all(s.effect is not LatentEffect.PROGRESS for s in steps[lo:hi])
    assert n_core_windows > 0 and n_osc / n_core_windows > 0.6


def test_lookalike_alternation_with_reversions_is_oscillating():
    """7 vs 1: the 8-event MUTATE-SUCCESS / VALIDATE-ERROR alternation, here with A/B reversions."""
    c = H.cfg()
    found = 0
    for events, record in H.sessions(FAMILY):
        steps, p0 = steps_from_record(record)
        meta = record.latent_trace["meta"]
        seq = [(e.operation_category.value, e.outcome.value) for e in events]
        for t in range(max(meta["core_start"] + EVENTS_PER_WINDOW - 1, EVENTS_PER_WINDOW - 1), meta["core_end"]):
            if seq[t - 7 : t + 1] != MV_ALTERNATION:
                continue
            st = window_stats(steps, p0, t, c)
            if st.reversions >= 2:
                assert st.delta_p <= c.eps_p
                assert record.window_labels[events[t].sequence_index] is OracleLabel.OSCILLATING
                found += 1
    assert found >= 5


def test_sessions_vary():
    langs, lengths, fams, styles, starts = set(), set(), collections.Counter(), set(), set()
    for events, record in H.sessions(FAMILY):
        langs.add(record.latent_trace["world"]["language"])
        lengths.add(len(events))
        fams.update(e.operation_family for e in events)
        meta = record.latent_trace["meta"]
        styles.add(meta["style"])
        starts.add(meta["core_start"])
    assert len(langs) >= 4 and max(lengths) - min(lengths) >= 25
    assert len(styles) >= 4 and len(starts) >= 6
    for fam in ("edit_file", "write_file", "run_tests", "read_file"):
        assert fams[fam] > 0, fam
