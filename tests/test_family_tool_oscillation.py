"""Family 8, tool_oscillation (M1_CONTRACTS §3, §7)."""

from __future__ import annotations

import collections

from agent_loop_lab.schema import EVENTS_PER_WINDOW, LatentEffect, OracleLabel
from agent_loop_lab.synth.labeller import window_stats
from agent_loop_lab.synth.latent import steps_from_record
from tests import synth_helpers as H

FAMILY = "tool_oscillation"


def test_family_contract_and_plurality():
    mix = H.check_family(FAMILY, intended="OSCILLATING")
    scored = H.scored_mix(mix)
    assert scored["OSCILLATING"] > sum(scored.values()) / 2


def test_core_mutations_undo_each_other_without_progress():
    """Core mutations are text-only (REDUNDANT) or reversions; core windows have Δp <= eps,
    and every core window with >= 2 window reversions is OSCILLATING."""
    c = H.cfg()
    n_core_windows = n_osc = 0
    for events, record in H.sessions(FAMILY):
        steps, p0 = steps_from_record(record)
        meta = record.latent_trace["meta"]
        lo, hi = meta["core_start"], meta["core_end"]
        core = steps[lo:hi]
        muts = [s for s in core if s.agent_mutation]
        assert len(muts) >= 6
        assert all(s.effect in (LatentEffect.REDUNDANT, LatentEffect.REVERSION) for s in muts)
        assert sum(s.effect is LatentEffect.REVERSION for s in muts) >= len(muts) - 1
        assert all(s.open_episode_kind is None for s in steps)
        assert len({s.progress_after for s in core}) == 1
        for t in range(max(lo + EVENTS_PER_WINDOW - 1, EVENTS_PER_WINDOW - 1), hi):
            st = window_stats(steps, p0, t, c)
            assert st.delta_p <= c.eps_p
            n_core_windows += 1
            if st.reversions >= c.n_rev:
                assert record.window_labels[events[t].sequence_index] is OracleLabel.OSCILLATING
                n_osc += 1
    assert n_core_windows > 0 and n_osc / n_core_windows > 0.8


def test_sessions_vary():
    langs, lengths, fams, pairs, between, starts = set(), set(), collections.Counter(), set(), set(), set()
    for events, record in H.sessions(FAMILY):
        langs.add(record.latent_trace["world"]["language"])
        lengths.add(len(events))
        fams.update(e.operation_family for e in events)
        meta = record.latent_trace["meta"]
        pairs.add(meta["pair"])
        between.add(meta["between"])
        starts.add(meta["core_start"])
    assert len(langs) >= 4 and max(lengths) - min(lengths) >= 25
    assert len(pairs) >= 5 and len(between) >= 4 and len(starts) >= 6
    for fam in ("edit_file", "write_file", "revert", "lint", "run_tests"):
        assert fams[fam] > 0, fam
