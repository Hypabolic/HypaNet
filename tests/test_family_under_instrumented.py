"""Family 12, under_instrumented (M1_CONTRACTS §3, §7; spec §5.2, §6.3 rule 1)."""

from __future__ import annotations

import collections
import random

from agent_loop_lab.schema import OracleLabel
from agent_loop_lab.synth.families import under_instrumented as UI
from agent_loop_lab.synth.latent import steps_from_record
from agent_loop_lab.windows import GateReason, iter_windows
from tests import synth_helpers as H

FAMILY = "under_instrumented"
INSTR = OracleLabel.INSUFFICIENT_INSTRUMENTATION
EVENT_FIELDS = {
    "tokens": ("input_tokens", "output_tokens"),
    "input_tokens": ("input_tokens",),
    "output_tokens": ("output_tokens",),
    "digests": ("workspace_state_before", "workspace_state_after"),
    "digest_before": ("workspace_state_before",),
    "digest_after": ("workspace_state_after",),
    "failure": ("failure_fingerprint",),
    "validation_counts": ("validation_passed", "validation_total"),
    "validation_suite": ("validation_suite_id",),
}


def _with_base(seed: int):
    return UI.generate_with_base(random.Random(seed), f"{FAMILY}-{seed:016x}", H.cfg())


def test_family_contract_and_plurality():
    # Gate rejects exactly the windows with a dropped event; the clean (scored)
    # windows follow the base behaviours, PRODUCTIVE-weighted.
    mix = H.check_family(FAMILY, intended="PRODUCTIVE", gate="drops")
    assert mix[INSTR.value] > 0
    assert sum(1 for k in H.scored_mix(mix) if mix[k] > 0) == 4  # several base behaviours are wrapped


def test_rule_1_fires_exactly_on_windows_containing_a_dropped_event():
    for events, record in H.sessions(FAMILY):
        steps, _ = steps_from_record(record)
        assert any(s.dropped for s in steps), "every session drops something"
        for t in range(7, len(steps)):
            has_drop = any(s.dropped for s in steps[t - 7 : t + 1])
            assert (record.window_labels[steps[t].sequence_index] is INSTR) == has_drop


def test_deliberate_drops_null_the_named_fields():
    for events, record in H.sessions(FAMILY):
        by_idx = {e.sequence_index: e for e in events}
        drops = record.latent_trace["drops"]
        for idx, fields in drops["fields"]:
            e = by_idx[idx]
            for f in fields:
                for attr in EVENT_FIELDS[f]:
                    assert getattr(e, attr) is None, (idx, f, attr)
        assert set(drops["deliberate"]) | set(drops["propagated"]) == set(record.dropped_observations)


def test_clean_windows_keep_base_labels_effects_and_pass_the_gate():
    """Drops change observations only: latent effects match the clean pass, and
    every window without a dropped event has the base label and passes the gate."""
    clean_windows = 0
    for seed in H.seeds(FAMILY):
        (events, record), (bev, brec) = _with_base(seed)
        assert brec.dropped_observations == () and all(e.provenance.history_complete for e in bev)
        H.assert_gate_all_pass(bev, brec)
        offset = events[0].sequence_index  # > 0 under truncation
        assert record.event_effects == brec.event_effects[offset:]
        steps, _ = steps_from_record(record)
        gate = {w.endpoint_index: w.gate.passed for w in iter_windows(events)}
        for t in range(7, len(steps)):
            if any(s.dropped for s in steps[t - 7 : t + 1]):
                continue
            idx = steps[t].sequence_index
            assert record.window_labels[idx] == brec.window_labels[idx], (seed, idx)
            assert gate[idx]
            clean_windows += 1
    assert clean_windows > 0


def test_every_gate_reason_kind_is_exercised():
    reasons: collections.Counter = collections.Counter()
    truncated = 0
    for events, record in H.sessions(FAMILY):
        for w in iter_windows(events):
            reasons.update(r for _, r in w.gate.reasons)
        m = record.latent_trace["drops"]["truncated_before"]
        if m:
            truncated += 1
            assert events[0].sequence_index == m and not events[0].provenance.history_complete
            # The first observed validation run has no observable baseline: dropped
            # (propagated, or already deliberately dropped for another field).
            first_val = next(e.sequence_index for e in events if e.validation_produced)
            assert first_val in record.dropped_observations
    expected = set(GateReason) - {GateReason.TOO_SHORT}
    assert expected <= set(reasons), f"missing gate reasons: {sorted(r.value for r in expected - set(reasons))}"
    assert truncated > 0


def test_sessions_vary():
    bases, kinds, lengths, families, langs = (collections.Counter() for _ in range(5))
    first_drop_pos = set()
    for events, record in H.sessions(FAMILY):
        meta = record.latent_trace["meta"]
        bases[meta["base"]] += 1
        kinds.update(meta["drop_kinds"])
        lengths[len(events)] += 1
        families.update(e.operation_family for e in events)
        langs[record.latent_trace["world"]["language"]] += 1
        first = min(record.dropped_observations) - events[0].sequence_index
        first_drop_pos.add(min(3, first * 4 // len(events)))
    assert len(bases) >= 6 and set(kinds) == set(UI.DROP_KINDS)
    assert max(lengths) - min(lengths) >= 25 and max(lengths) >= 40
    assert len(langs) >= 4 and len(first_drop_pos) >= 3  # drops early, mid and late
    for fam in ("read_file", "search", "edit_file", "run_tests", "http_call", "poll_status"):
        assert families[fam] > 0, fam


def test_generate_and_generate_with_base_agree():
    seed = H.seeds(FAMILY, 1)[0]
    events, record = H.generate(FAMILY, seed)
    (ev2, rec2), _ = _with_base(seed)
    assert [e.to_json() for e in events] == [e.to_json() for e in ev2]
    assert record.window_labels == rec2.window_labels
