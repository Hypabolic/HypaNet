"""Family 6, cosmetic_churn (M1_CONTRACTS §3, §7; spec §0.2 r2 reading)."""

from __future__ import annotations

import collections

from agent_loop_lab.schema import LatentEffect, OracleLabel
from agent_loop_lab.synth.latent import steps_from_record
from tests import synth_helpers as H

FAMILY = "cosmetic_churn"
MV_ALTERNATION = [("MUTATE", "SUCCESS"), ("VALIDATE", "ERROR")] * 4
MV_GREEN = [("MUTATE", "SUCCESS"), ("VALIDATE", "SUCCESS")] * 4


def test_family_contract_and_plurality():
    mix = H.check_family(FAMILY, intended="STALLED")
    assert mix["RECOVERY"] == mix["OSCILLATING"] == 0


def test_edits_change_text_not_behaviour():
    for events, record in H.sessions(FAMILY):
        steps, _ = steps_from_record(record)
        assert all(s.open_episode_kind is None for s in steps)
        assert not any(s.undoes_prior_mutation for s in steps)
        churn = [(e, s) for e, s in zip(events, steps)
                 if e.operation_category.value == "MUTATE" and s.effect is LatentEffect.REDUNDANT]
        assert len(churn) >= 4, record.session_id
        # Text-only edits leave progress where it was.
        for i, (e, s) in enumerate(zip(events, steps)):
            if e.operation_category.value == "MUTATE" and s.effect is LatentEffect.REDUNDANT and i > 0:
                assert s.progress_after == steps[i - 1].progress_after
        # Re-runs after text-only edits are REDUNDANT validation on an unchanged behavioural workspace.
        reruns = [s for e, s in zip(events, steps)
                  if e.operation_family == "run_tests" and s.effect is LatentEffect.REDUNDANT]
        assert reruns, record.session_id


def test_lookalike_edit_test_alternation_is_stalled():
    """6 vs 9: the MUTATE-SUCCESS / VALIDATE-ERROR alternation, with text-only edits."""
    found = collections.Counter()
    for events, record in H.sessions(FAMILY):
        seq = [(e.operation_category.value, e.outcome.value) for e in events]
        for t in range(7, len(events)):
            window = seq[t - 7 : t + 1]
            label = record.window_labels[events[t].sequence_index]
            if window == MV_ALTERNATION:
                found[("failing", label.value)] += 1
            elif window == MV_GREEN:
                found[("green", label.value)] += 1
    assert found[("failing", "STALLED")] >= 10, dict(found)
    assert found[("green", "STALLED")] >= 1, dict(found)
    assert found[("failing", "PRODUCTIVE")] <= found[("failing", "STALLED")] // 4


def test_sessions_vary():
    langs, lengths, families, kinds, shapes = set(), set(), collections.Counter(), collections.Counter(), collections.Counter()
    for events, record in H.sessions(FAMILY):
        langs.add(record.latent_trace["world"]["language"])
        lengths.add(len(events))
        families.update(e.operation_family for e in events)
        kinds.update(op.rsplit(" ", 1)[-1] for op in record.latent_trace["ops"] if op.startswith(("edit_file", "write_file")))
        meta = record.latent_trace["meta"]
        shapes.update([meta["shape"], meta["prefix"]])
    assert len(langs) >= 4 and max(lengths) - min(lengths) >= 20
    for key in ("failing", "green", "neutral", "productive"):
        assert shapes[key] > 0, key
    for kind in ("cosmetic:format", "cosmetic:rename_local", "cosmetic:comment", "cosmetic:whitespace", "cosmetic:docstring"):
        assert kinds[kind] > 0, kind
    assert sum(v for k, v in kinds.items() if k.startswith("fix:")) > 0  # re-adding an existing guard
    for fam in ("edit_file", "write_file", "run_tests", "read_file"):
        assert families[fam] > 0, fam
    assert families["lint"] + families["build"] > 0


def test_session_length_bound_on_canonical_build_seeds():
    """Regression: these canonical-build seeds produced 61 events before the churn loop reserved room
    for its worst-case 4 events per iteration."""
    from agent_loop_lab.synth.families import generate_session

    for seed in (11059343153159220918, 3536032727643485715):
        events, _ = generate_session("cosmetic_churn", seed, "regression")
        assert 12 <= len(events) <= 60
