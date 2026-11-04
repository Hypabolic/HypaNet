"""Acceptance coverage across independent seeds, plus causal engine checks."""

import json
import random

import numpy as np
import pytest

from agent_loop_lab.encode import encode_window
from agent_loop_lab.fingerprint import FAMILY_CATEGORY, workspace_digest
from agent_loop_lab.oracle_config import load_oracle_config
from agent_loop_lab.schema import CanonicalEvent, LatentEffect as E, OracleLabel as L, OracleRecord
from agent_loop_lab.synth.labeller import label_session, validate_steps
from agent_loop_lab.synth.latent import steps_from_record
from agent_loop_lab.synth_ood import FAMILIES
from agent_loop_lab.synth_ood.engine import Engine
from agent_loop_lab.synth_ood.counterfactuals import counterfactual_pairs
from agent_loop_lab.windows import GateReason, gate, iter_windows, validate_sequence


CFG = load_oracle_config("oracle_config.v0_2.yaml")
INTENDED = {
    "shard_migration": {L.PRODUCTIVE, L.RECOVERY},
    "fixture_archaeology": {L.PRODUCTIVE, L.STALLED},
    "lease_reconnect": {L.PRODUCTIVE, L.RECOVERY},
    "index_watermark": {L.PRODUCTIVE, L.RECOVERY},
    "pinned_validation": {L.PRODUCTIVE, L.RECOVERY, L.STALLED},
    "annotation_spiral": {L.STALLED, L.PRODUCTIVE},
    "serializer_duel": {L.OSCILLATING, L.PRODUCTIVE},
    "merge_ping_pong": {L.OSCILLATING, L.PRODUCTIVE},
    "lockfile_churn": {L.OSCILLATING, L.PRODUCTIVE},
    "telemetry_gaps": {L.PRODUCTIVE},
}


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("seed", range(24))
def test_session_contract(name, seed):
    generate = FAMILIES[name]
    events, oracle = generate(random.Random(seed), "session-17", CFG)
    other_events, other_oracle = generate(random.Random(seed), "session-17", CFG)
    assert [e.to_json() for e in events] == [e.to_json() for e in other_events]
    assert json.dumps(oracle.to_dict(), sort_keys=True) == json.dumps(other_oracle.to_dict(), sort_keys=True)
    assert OracleRecord.from_dict(oracle.to_dict()) == oracle
    assert 20 <= len(events) <= 90
    assert oracle.generator_version == "ood-0.1"
    assert oracle.scenario_family == f"ood:{name}"
    steps, p0 = steps_from_record(oracle)
    validate_steps(steps, CFG)
    validate_sequence(events)
    assert len(steps) == len(events)
    assert label_session(steps, p0, CFG) == oracle.window_labels
    assert tuple(s.effect for s in steps) == oracle.event_effects
    assert tuple(s.sequence_index for s in steps if s.dropped) == oracle.dropped_observations
    assert INTENDED[name] <= set(oracle.window_labels.values())
    assert all(label.value in generate.__doc__ for label in INTENDED[name])
    for event, step in zip(events, steps):
        assert CanonicalEvent.from_json(event.to_json()) == event
        assert event.operation_category == FAMILY_CATEGORY[event.operation_family]
        assert event.sequence_index == step.sequence_index
        assert event.session_id == oracle.session_id
        if not step.dropped:
            assert event.workspace_state_before == step.ws_before
            assert event.workspace_state_after == step.ws_after
        if step.agent_mutation:
            assert step.ws_before != step.ws_after
        assert event.provenance.source == "simulator:ood-0.1"
        assert not event.provenance.instrumentation_flags
        assert name not in event.to_json()
    for result in iter_windows(events):
        t = result.endpoint_index
        dropped = any(s.dropped for s in steps[t-7:t+1])
        assert result.gate.passed == (not dropped)
        assert (oracle.window_labels[t] == L.INSUFFICIENT_INSTRUMENTATION) == dropped
        if result.gate.passed:
            assert result.features.shape == (120,)
            assert result.features.dtype == np.float32
            assert np.isfinite(result.features).all()
        else:
            assert name == "telemetry_gaps"
            assert result.features is None


@pytest.mark.parametrize("name", FAMILIES)
def test_rng_controls_variation(name):
    sessions = [FAMILIES[name](random.Random(i), "s", CFG)[0] for i in range(8)]
    assert len({len(events) for events in sessions}) > 1 or name == "telemetry_gaps"
    assert len({events[0].to_json() for events in sessions}) > 1


def engine(seed=1):
    return Engine(random.Random(seed), "causal", CFG)


def test_real_text_drives_progress_validation_and_digest():
    e = engine()
    path = e.repo.paths[0]
    e.test()
    old = e.repo.digest()
    e.write(path, e.repo.files[path].replace("enabled=0", "enabled=1"))
    e.test()
    assert e.events[0].validation_passed == 0
    assert e.events[-1].validation_passed == 1
    assert e.steps[1].effect == E.PROGRESS
    assert e.steps[2].effect == E.NONE
    assert e.steps[1].progress_after == 1 / len(e.repo.paths)
    assert old != e.repo.digest() == workspace_digest(e.repo.files)
    assert e.events[-1].workspace_state_after == e.repo.digest()


def test_facts_and_cosmetic_writes_are_not_progress():
    e = engine()
    path = e.repo.paths[0]
    e.read(path)
    e.read(path)
    e.write(path, e.repo.files[path] + "# formatting only\n")
    assert [s.effect for s in e.steps] == [E.NEW_FACT, E.REDUNDANT, E.REDUNDANT]
    assert [s.new_facts for s in e.steps] == [2, 0, 0]
    assert all(s.progress_after == 0 for s in e.steps)
    assert e.steps[-1].agent_mutation


def test_retry_budget_and_episode_resolution():
    e = engine()
    e.service("peer.cedar.invalid", "retry", 5)
    target = {"method": "GET", "host": "peer.cedar.invalid", "path_template": "/lease"}
    for _ in range(6):
        e.perform("http_call", target)
    assert [s.effect for s in e.steps] == [E.NONE] + [E.IN_POLICY_RETRY]*3 + [E.REDUNDANT, E.RESOLVES_EPISODE]
    assert e.steps[0].open_episode_kind is None
    assert e.steps[1].open_episode_attempts == 1
    assert e.steps[-1].open_episode_kind is None
    assert e.steps[-1].resolved_episode_here
    # New service knowledge is recorded even though resolution has precedence.
    assert e.steps[-1].new_facts == 1


def test_poll_budget_ignores_backoff_metadata():
    e = engine()
    e.service("index:42", "poll", 7)
    for i in range(8):
        e.perform("poll_status", "index:42", {"backoff": 2**i})
    assert [s.effect for s in e.steps] == [E.NONE] + [E.IN_POLICY_POLL]*5 + [E.REDUNDANT, E.RESOLVES_EPISODE]
    assert len({ev.intent_fingerprint for ev in e.events}) == 1


def test_episode_abandonment_and_cosmetic_validation():
    e = engine()
    e.test()
    e.test()
    assert e.steps[-1].open_episode_kind == "retry"
    path = e.repo.paths[0]
    e.write(path, e.repo.files[path].replace("enabled=0", "enabled=1"))
    assert e.steps[-1].open_episode_kind is None
    assert e.steps[-1].effect == E.PROGRESS
    e.test()
    e.test()
    e.perform("plan", args={"action_type": "consider"})
    assert e.steps[-1].open_episode_kind == "retry"
    e.perform("plan", args={"action_type": "consider-more"})
    assert e.steps[-1].open_episode_kind is None
    e.write(path, e.repo.files[path] + "# pretty\n")
    e.test()
    assert e.steps[-1].effect == E.REDUNDANT


def test_reversion_and_restored_progress_are_derived():
    e = engine()
    path = e.repo.paths[0]
    original = e.repo.files[path]
    e.write(path, original.replace("enabled=0", "enabled=1"))
    e.write(path, original)
    e.write(path, original.replace("enabled=0", "enabled=1"))
    assert [s.effect for s in e.steps] == [E.PROGRESS, E.REVERSION, E.PROGRESS]
    assert e.steps[-1].undoes_prior_mutation  # progress outranks reversion
    validate_steps(e.steps, CFG)


def test_dropped_reason_coverage_and_recovery_after_gap():
    events, oracle = FAMILIES["telemetry_gaps"](random.Random(9), "s", CFG)
    reasons = {reason for t in range(7, len(events)) for _, reason in gate(events[:t+1]).reasons}
    assert reasons == set(GateReason) - {GateReason.TOO_SHORT}
    for i in oracle.dropped_observations:
        assert not gate(events[:i+1]).passed
    # A burst may contain several overlapping losses, including a result whose
    # own fields are complete but whose baseline was lost. Rejection ends when
    # the final affected event leaves the window.
    last = max(oracle.dropped_observations)
    assert not gate(events[:last+8]).passed
    assert gate(events[:last+9]).passed


def test_encoding_is_prefix_causal():
    events, _ = FAMILIES["shard_migration"](random.Random(2), "s", CFG)
    first = encode_window(events[:8]).tobytes()
    list(iter_windows(events))
    assert encode_window(events[:8]).tobytes() == first


@pytest.mark.parametrize("seed", range(8))
def test_identical_category_outcome_sequences_can_have_opposite_labels(seed):
    productive, good_oracle = FAMILIES["shard_migration"](random.Random(seed), "same", CFG)
    stalled, bad_oracle = FAMILIES["annotation_spiral"](random.Random(seed), "same", CFG)
    # Both start with a failure and eight write/failing-test cycles. Only the
    # causal contents distinguish real repairs from inert annotation writes.
    assert [(e.operation_category, e.outcome) for e in productive[:17]] == [
        (e.operation_category, e.outcome) for e in stalled[:17]
    ]
    assert good_oracle.window_labels[16] == L.PRODUCTIVE
    assert bad_oracle.window_labels[16] == L.STALLED


@pytest.mark.parametrize("seed", range(8))
def test_causal_counterfactual_pairs(seed):
    pairs = counterfactual_pairs(random.Random(seed), "paired", CFG)
    repeated = counterfactual_pairs(random.Random(seed), "paired", CFG)
    expected = {"service_persistence": L.STALLED, "edit_annotation": L.STALLED,
                "edit_reversal": L.OSCILLATING}
    for name, (good, bad) in pairs.items():
        assert [e.to_json() for e in good[0][:9]] == [e.to_json() for e in bad[0][:9]]
        assert len(good[0]) == len(bad[0])
        assert good[1].window_labels[len(good[0])-1] == L.PRODUCTIVE
        assert bad[1].window_labels[len(bad[0])-1] == expected[name]
        for branch, again in zip((good, bad), repeated[name]):
            events, oracle = branch
            assert [e.to_json() for e in events] == [e.to_json() for e in again[0]]
            assert oracle.to_dict() == again[1].to_dict()
            validate_sequence(events)
            steps, p0 = steps_from_record(oracle)
            validate_steps(steps, CFG)
            assert label_session(steps, p0, CFG) == oracle.window_labels
            assert all(w.gate.passed for w in iter_windows(events))
