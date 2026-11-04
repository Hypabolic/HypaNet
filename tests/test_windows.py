"""Tests for windowing, sequence validation and the runtime gate (spec v0.2 §5, §11; M0 contracts §B)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from agent_loop_lab.encode import EncodingError, encode_window
from agent_loop_lab.schema import CanonicalEvent, Outcome, Provenance
from agent_loop_lab.windows import (
    INSUFFICIENT_EVIDENCE,
    GateDecision,
    GateReason,
    SequenceError,
    WindowResult,
    encode_session,
    gate,
    iter_windows,
    validate_sequence,
    window_endpoints,
)
from tests.factories import h, make_event, novel_inspections, session

R = GateReason
TRUNCATED = Provenance(source="test:truncated", history_complete=False)


def val(suite: str, passed, total, **kw) -> dict:
    return {"category": "VALIDATE", "outcome": "SUCCESS", "validation_produced": True,
            "validation_suite_id": suite, "validation_passed": passed, "validation_total": total, **kw}


# ---------------------------------------------------------------------------
# GateReason / GateDecision
# ---------------------------------------------------------------------------


def test_gate_reason_members():
    assert [r.value for r in GateReason] == [
        "TOO_SHORT", "MISSING_TOKENS", "MISSING_WORKSPACE_DIGEST", "MISSING_FAILURE_FINGERPRINT",
        "MISSING_VALIDATION_FIELDS", "UNOBSERVED_VALIDATION_BASELINE",
    ]
    assert INSUFFICIENT_EVIDENCE == "INSUFFICIENT_EVIDENCE"


def test_gate_decision_result_and_consistency():
    assert GateDecision(True, ()).result == "PASS"
    assert GateDecision(False, ((0, R.MISSING_TOKENS),)).result == INSUFFICIENT_EVIDENCE
    with pytest.raises(ValueError):
        GateDecision(True, ((0, R.MISSING_TOKENS),))
    with pytest.raises(ValueError):
        GateDecision(False, ())


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", range(0, 8))
def test_too_short(n):
    d = gate(novel_inspections(n))
    assert not d.passed
    assert d.reasons == ((-1, R.TOO_SHORT),)
    assert d.result == INSUFFICIENT_EVIDENCE


def test_too_short_even_if_events_are_also_missing_observations():
    ev = session([{"tokens": (None, None)} for _ in range(3)])
    assert gate(ev).reasons == ((-1, R.TOO_SHORT),)


def test_eight_novel_inspections_pass():
    # spec §11: all-novel INSPECT windows are NOT rejected
    d = gate(novel_inspections(8))
    assert d.passed and d.reasons == () and d.result == "PASS"
    assert gate(novel_inspections(20)).passed


def test_gate_makes_no_semantic_judgement():
    # repeated identical failing operations, fully instrumented, still pass the gate
    ev = session([{"intent": "x", "outcome": "ERROR", "failure": "F", "category": "VALIDATE"} for _ in range(8)])
    assert gate(ev).passed
    mixed = session([{"category": c, "outcome": "NEUTRAL"} for c in
                     ["CONTROL", "EXTERNAL", "INSPECT", "MUTATE", "VALIDATE", "CONTROL", "EXTERNAL", "INSPECT"]])
    assert gate(mixed).passed


GATE_CASES = {
    "input_tokens": ({"tokens": (None, 5)}, R.MISSING_TOKENS),
    "output_tokens": ({"tokens": (5, None)}, R.MISSING_TOKENS),
    "both_tokens": ({"tokens": (None, None)}, R.MISSING_TOKENS),
    "both_digests": ({"before": None}, R.MISSING_WORKSPACE_DIGEST),
    "after_digest": ({"workspace_state_after": None}, R.MISSING_WORKSPACE_DIGEST),
    "before_digest": ({"workspace_state_before": None}, R.MISSING_WORKSPACE_DIGEST),
    "failure_fingerprint": ({"outcome": "ERROR", "failure_fingerprint": None}, R.MISSING_FAILURE_FINGERPRINT),
    "validation_suite": (val(None, 1, 2), R.MISSING_VALIDATION_FIELDS),
    "validation_passed": (val("u", None, 2), R.MISSING_VALIDATION_FIELDS),
    "validation_total": (val("u", 1, None), R.MISSING_VALIDATION_FIELDS),
}


@pytest.mark.parametrize("case", sorted(GATE_CASES))
@pytest.mark.parametrize("pos", [0, 4, 7])
def test_each_gate_reason(case, pos):
    spec, reason = GATE_CASES[case]
    specs = [{} for _ in range(8)]
    specs[pos] = spec
    ev = session(specs)
    d = gate(ev)
    assert not d.passed
    assert d.reasons == ((pos, reason),)
    # the encoder must refuse what the gate rejects (never zero-fill)
    with pytest.raises(EncodingError):
        encode_window(ev)


@pytest.mark.parametrize("case", sorted(GATE_CASES))
def test_missing_observation_before_window_is_not_gated(case):
    spec, _ = GATE_CASES[case]
    ev = session([spec] + [{} for _ in range(8)])
    assert gate(ev).passed


def test_positions_are_window_relative_and_reasons_sorted():
    specs = [{} for _ in range(12)]
    specs[11] = {"outcome": "ERROR", "failure_fingerprint": None}
    specs[6] = {"tokens": (None, 1), "before": None}
    specs[4] = {"tokens": (None, 1)}  # window is 4..11 -> position 0
    d = gate(session(specs))
    assert d.reasons == (
        (0, R.MISSING_TOKENS),
        (2, R.MISSING_TOKENS),
        (2, R.MISSING_WORKSPACE_DIGEST),
        (7, R.MISSING_FAILURE_FINGERPRINT),
    )
    assert list(d.reasons) == sorted(d.reasons, key=lambda r: (r[0], r[1].value))


def test_unobserved_baseline_before_window():
    ev = session([val("unit", None, 10)] + [{} for _ in range(7)] + [val("unit", 5, 10)])
    d = gate(ev)
    assert d.reasons == ((7, R.UNOBSERVED_VALIDATION_BASELINE),)
    with pytest.raises(EncodingError):
        encode_window(ev)


def test_unobserved_baseline_inside_window_flags_both():
    ev = session([{}, {}, val("unit", None, 10), {}, {}, val("unit", 5, 10), {}, {}])
    assert gate(ev).reasons == ((2, R.MISSING_VALIDATION_FIELDS), (5, R.UNOBSERVED_VALIDATION_BASELINE))


def test_unobserved_older_baseline_shadowed_by_observed_newer_one():
    # window is events 2..9; both earlier results are before it, the most recent (index 1) is observed
    ev = session([val("unit", None, 10), val("unit", 4, 10)] + [{} for _ in range(7)] + [val("unit", 5, 10)])
    assert gate(ev).passed
    assert encode_window(ev).shape == (120,)


def _unknown_suite(passed, total) -> dict:
    return {"category": "VALIDATE", "outcome": "SUCCESS", "validation_produced": True,
            "validation_suite_id": None, "validation_passed": passed, "validation_total": total}


def test_unknown_suite_result_before_window_shadows_older_baseline():
    # e0 = unit 5/10; e1 = result with unobserved suite id (might be unit); window is e2..e9.
    # The most recent 'unit' baseline is unknown, so the window must not compare against e0.
    ev = session([val("unit", 5, 10), _unknown_suite(10, 10)] + [{} for _ in range(7)] + [val("unit", 10, 10)])
    validate_sequence(ev)
    d = gate(ev)
    assert d.reasons == ((7, R.UNOBSERVED_VALIDATION_BASELINE),)
    with pytest.raises(EncodingError):
        encode_window(ev)


def test_unknown_suite_result_before_window_without_other_baseline():
    # e0 = unobserved-suite result; window e1..e8. Must not become "no comparable result" (0/0).
    ev = session([_unknown_suite(3, 10)] + [{} for _ in range(7)] + [val("unit", 10, 10)])
    validate_sequence(ev)
    assert gate(ev).reasons == ((7, R.UNOBSERVED_VALIDATION_BASELINE),)
    with pytest.raises(EncodingError):
        encode_window(ev)


def test_unknown_suite_result_inside_window_flags_both():
    ev = session([val("unit", 5, 10), {}, {}, _unknown_suite(4, 10), {}, {}, val("unit", 6, 10), {}, {}])
    # window is e1..e8 -> unknown suite at position 2, dependent result at position 5
    assert gate(ev).reasons == ((2, R.MISSING_VALIDATION_FIELDS), (5, R.UNOBSERVED_VALIDATION_BASELINE))


def test_unknown_suite_result_older_than_same_suite_baseline_is_irrelevant():
    # the observed 'unit' result at e1 is more recent than the unknown-suite one, so it is the baseline
    ev = session([_unknown_suite(1, 10), val("unit", 4, 10)] + [{} for _ in range(7)] + [val("unit", 5, 10)])
    assert gate(ev).passed
    assert encode_window(ev).shape == (120,)


def test_unknown_suite_result_does_not_affect_non_validation_window():
    ev = session([_unknown_suite(1, 10)] + [{} for _ in range(8)])
    assert gate(ev).passed


def test_iter_windows_rejects_every_window_depending_on_unknown_suite_result():
    # scenario-12 path: once the dropped-suite event leaves the window, later windows that
    # depend on it as a baseline must still be rejected, not silently pass.
    ev = session([val("unit", 5, 10), _unknown_suite(10, 10)] + [{} for _ in range(7)]
                 + [val("unit", 10, 10), {}, {}])
    results = {w.endpoint_index: w for w in iter_windows(ev)}
    assert not results[8].gate.passed  # unknown-suite event inside the window
    for t in (9, 10, 11):
        # the 'unit' result at e9 sits at window position 9 - (t - 7)
        assert results[t].gate.reasons == ((16 - t, R.UNOBSERVED_VALIDATION_BASELINE),)
        assert results[t].features is None
    endpoints, _ = encode_session(ev)
    assert endpoints == []


def test_unobserved_baseline_of_other_suite_is_irrelevant():
    ev = session([val("other", None, 10)] + [{} for _ in range(7)] + [val("unit", 5, 10)])
    assert gate(ev).passed


def test_missing_suite_id_does_not_also_report_baseline():
    ev = [make_event(i, provenance=TRUNCATED) for i in range(30, 37)]
    ev.append(make_event(37, "VALIDATE", provenance=TRUNCATED, validation_produced=True,
                         validation_suite_id=None, validation_passed=1, validation_total=2))
    assert gate(ev).reasons == ((7, R.MISSING_VALIDATION_FIELDS),)


def test_truncated_history_without_baseline_is_unobserved():
    ev = [make_event(i, provenance=TRUNCATED) for i in range(10, 17)]
    ev.append(make_event(17, "VALIDATE", validation=("unit", 5, 10), provenance=TRUNCATED))
    validate_sequence(ev)
    assert gate(ev).reasons == ((7, R.UNOBSERVED_VALIDATION_BASELINE),)
    with pytest.raises(EncodingError):
        encode_window(ev)


def test_truncated_history_with_baseline_in_prefix_passes():
    ev = [make_event(10, "VALIDATE", validation=("unit", 2, 10), provenance=TRUNCATED)]
    ev += [make_event(i, provenance=TRUNCATED) for i in range(11, 18)]
    ev.append(make_event(18, "VALIDATE", validation=("unit", 5, 10), provenance=TRUNCATED))
    assert gate(ev).passed


def test_truncated_history_without_validation_passes():
    ev = [make_event(i, provenance=TRUNCATED) for i in range(10, 18)]
    assert gate(ev).passed


def test_complete_history_with_no_baseline_passes():
    ev = session([{} for _ in range(7)] + [val("unit", 5, 10)])
    d = gate(ev)
    assert d.passed
    vec = encode_window(ev)
    assert vec[7 * 15 + 12] == 0.0 and vec[7 * 15 + 13] == 0.0


def test_gate_rejection_implies_encoder_rejection_combined():
    specs = [{} for _ in range(8)]
    specs[1] = {"tokens": (None, 3)}
    specs[3] = {"outcome": "ERROR", "failure_fingerprint": None, "before": None}
    ev = session(specs)
    assert gate(ev).reasons == ((1, R.MISSING_TOKENS), (3, R.MISSING_FAILURE_FINGERPRINT),
                                (3, R.MISSING_WORKSPACE_DIGEST))
    with pytest.raises(EncodingError):
        encode_window(ev)


# ---------------------------------------------------------------------------
# validate_sequence
# ---------------------------------------------------------------------------


def test_validate_sequence_ok():
    validate_sequence([])
    validate_sequence(novel_inspections(1))
    validate_sequence(novel_inspections(20))
    validate_sequence([make_event(i, provenance=TRUNCATED) for i in range(5, 15)])


@pytest.mark.parametrize("field", ["session_id", "agent_id", "task_id"])
def test_validate_sequence_mixed_identity(field):
    ev = novel_inspections(10)
    ev[6] = dataclasses.replace(ev[6], **{field: "intruder"})
    with pytest.raises(SequenceError):
        validate_sequence(ev)


def test_validate_sequence_duplicate_event_id():
    ev = novel_inspections(10)
    ev[6] = dataclasses.replace(ev[6], event_id=ev[2].event_id)
    with pytest.raises(SequenceError):
        validate_sequence(ev)


@pytest.mark.parametrize("indices", [
    [0, 1, 2, 4, 5],      # gap
    [0, 1, 2, 2, 3],      # repeat
    [0, 2, 1, 3, 4],      # out of order
    [0, 1, 2, 3, 3 + 2],  # jump at end
])
def test_validate_sequence_bad_indices(indices):
    ev = [make_event(i) for i in indices]
    # event_id derives from index; make ids unique so only the ordering rule can fire
    ev = [dataclasses.replace(e, event_id=f"uniq-{k}") for k, e in enumerate(ev)]
    with pytest.raises(SequenceError):
        validate_sequence(ev)


def test_validate_sequence_complete_history_must_start_at_zero():
    ev = [make_event(i) for i in range(3, 12)]  # default provenance: history_complete=True
    with pytest.raises(SequenceError):
        validate_sequence(ev)


def test_validate_sequence_rejects_non_events():
    with pytest.raises(SequenceError):
        validate_sequence(novel_inspections(3) + [{"not": "an event"}])  # type: ignore[list-item]


def test_sequence_error_is_value_error():
    assert issubclass(SequenceError, ValueError)


# ---------------------------------------------------------------------------
# window_endpoints / iter_windows / encode_session
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,expected", [(0, []), (1, []), (7, []), (8, [7]), (12, [7, 8, 9, 10, 11])])
def test_window_endpoints(n, expected):
    assert window_endpoints(novel_inspections(n)) == expected


def test_iter_windows_basic():
    ev = novel_inspections(12)
    results = list(iter_windows(ev))
    assert [r.endpoint_index for r in results] == [7, 8, 9, 10, 11]
    for r in results:
        assert isinstance(r, WindowResult)
        assert r.gate.passed
        assert r.features.shape == (120,) and r.features.dtype == np.float32
        assert r.features.tobytes() == encode_window(ev[: r.endpoint_index + 1]).tobytes()


def test_iter_windows_short_session_yields_nothing():
    assert list(iter_windows(novel_inspections(7))) == []
    assert list(iter_windows([])) == []


def test_iter_windows_validates_sequence():
    ev = novel_inspections(10)
    ev[5] = dataclasses.replace(ev[5], session_id="other")
    with pytest.raises(SequenceError):
        list(iter_windows(ev))
    with pytest.raises(SequenceError):
        encode_session(ev)


def test_iter_windows_failing_windows_have_no_features():
    specs = [{} for _ in range(12)]
    specs[9] = {"tokens": (None, 1)}  # in windows ending at 9, 10, 11 (and not 7, 8)
    results = list(iter_windows(session(specs)))
    assert [r.gate.passed for r in results] == [True, True, False, False, False]
    assert [r.features is None for r in results] == [False, False, True, True, True]
    assert results[2].gate.reasons == ((7, R.MISSING_TOKENS),)
    assert results[3].gate.reasons == ((6, R.MISSING_TOKENS),)
    assert results[4].gate.reasons == ((5, R.MISSING_TOKENS),)


def test_encode_session_shapes_and_endpoints():
    specs = [{} for _ in range(12)]
    specs[9] = {"tokens": (None, 1)}
    endpoints, feats = encode_session(session(specs))
    assert endpoints == [7, 8]
    assert feats.shape == (2, 120) and feats.dtype == np.float32

    endpoints, feats = encode_session(novel_inspections(15))
    assert endpoints == list(range(7, 15))
    assert feats.shape == (8, 120) and feats.dtype == np.float32
    assert feats.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("n", [0, 3, 7])
def test_encode_session_empty_shape(n):
    endpoints, feats = encode_session(novel_inspections(n))
    assert endpoints == []
    assert feats.shape == (0, 120) and feats.dtype == np.float32


def test_encode_session_all_windows_fail_shape():
    ev = session([{"tokens": (None, None)} for _ in range(10)])
    endpoints, feats = encode_session(ev)
    assert endpoints == []
    assert feats.shape == (0, 120) and feats.dtype == np.float32


def test_truncated_session_endpoint_index_is_sequence_index():
    ev = [make_event(i, provenance=TRUNCATED) for i in range(40, 52)]
    assert window_endpoints(ev) == [7, 8, 9, 10, 11]  # list positions
    assert [r.endpoint_index for r in iter_windows(ev)] == [47, 48, 49, 50, 51]  # sequence indices
    endpoints, feats = encode_session(ev)
    assert endpoints == [47, 48, 49, 50, 51] and feats.shape == (5, 120)


# ---------------------------------------------------------------------------
# Causality and determinism through the session path
# ---------------------------------------------------------------------------


def _rich_session(n: int = 12) -> list[CanonicalEvent]:
    specs = []
    for i in range(n):
        k = i % 4
        if k == 0:
            specs.append({"category": "MUTATE", "before": "A", "after": "B", "intent": "edit-b"})
        elif k == 1:
            specs.append({"category": "VALIDATE", "before": "B", "outcome": "ERROR", "failure": "F",
                          "validation": ("unit", i % 7, 10), "intent": "test"})
        elif k == 2:
            specs.append({"category": "MUTATE", "before": "B", "after": "A", "intent": "edit-a"})
        else:
            specs.append({"category": "EXTERNAL", "before": "A", "intent": f"call-{i}", "tokens": (i * 1000, 7)})
    return session(specs)


def test_causality_append_and_mutate_later_events():
    ev = _rich_session(12)
    endpoints, base = encode_session(ev)
    assert endpoints == [7, 8, 9, 10, 11]
    rows = {t: base[i].tobytes() for i, t in enumerate(endpoints)}

    # append later events
    longer = _rich_session(16)
    assert longer[:12] == ev
    e2, f2 = encode_session(longer)
    for i, t in enumerate(e2):
        if t in rows:
            assert f2[i].tobytes() == rows[t]

    # mutate the last event: windows ending before it must not change
    mutated = list(ev)
    mutated[11] = dataclasses.replace(
        ev[11], outcome=Outcome.SUCCESS, intent_fingerprint=h("edit-b"), workspace_state_before=h("A"),
        workspace_state_after=h("B"), input_tokens=0, output_tokens=0,
    )
    e3, f3 = encode_session(mutated)
    assert e3 == endpoints
    for i, t in enumerate(e3):
        if t < 11:
            assert f3[i].tobytes() == rows[t]
    assert f3[e3.index(11)].tobytes() != rows[11]  # the mutation is visible at its own window

    # a later event that would itself fail the gate does not affect earlier windows either
    broken = ev + [make_event(12, tokens=(None, None))]
    e4, f4 = encode_session(broken)
    assert e4 == endpoints
    assert f4.tobytes() == base.tobytes()


def test_causality_later_validation_result_does_not_change_earlier_baselines():
    ev = _rich_session(12)
    _, base = encode_session(ev)
    later = ev + [make_event(12, "VALIDATE", validation=("unit", 10, 10), intent="test", before="A")]
    _, f = encode_session(later)
    assert f[: len(base)].tobytes() == base.tobytes()


def test_session_encoding_is_byte_identical():
    ev = _rich_session(20)
    e1, f1 = encode_session(ev)
    e2, f2 = encode_session(list(ev))
    rebuilt = [CanonicalEvent.from_json(e.to_json()) for e in ev]
    e3, f3 = encode_session(rebuilt)
    assert e1 == e2 == e3
    assert f1.tobytes() == f2.tobytes() == f3.tobytes()


_NULLABLE = {
    "tokens_in": {"input_tokens": None},
    "tokens_out": {"output_tokens": None},
    "before": {"workspace_state_before": None},
    "after": {"workspace_state_after": None},
}


@pytest.mark.parametrize("seed", range(4))
def test_randomized_gate_encoder_agreement(seed):
    """Randomly null observations in valid sessions: gate passes iff the encoder does not raise."""
    import random

    from tests.test_encode import random_session

    rng = random.Random(1000 + seed)
    rejected = passed = 0
    for _ in range(60):
        ev = random_session(rng)
        for _ in range(rng.randint(0, 3)):
            k = rng.randrange(len(ev))
            e = ev[k]
            choices = list(_NULLABLE)
            if e.outcome is Outcome.ERROR:
                choices.append("failure")
            if e.validation_produced:
                choices += ["suite", "passed", "total"]
            pick = rng.choice(choices)
            change = {"failure": {"failure_fingerprint": None}, "suite": {"validation_suite_id": None},
                      "passed": {"validation_passed": None}, "total": {"validation_total": None}}.get(
                pick, _NULLABLE.get(pick))
            ev[k] = dataclasses.replace(e, **change)
        if rng.random() < 0.3:  # also exercise truncated histories
            cut = rng.randint(1, max(1, len(ev) - 8))
            ev = [dataclasses.replace(e, provenance=TRUNCATED) for e in ev[cut:]]
        results = list(iter_windows(ev))
        for t, r in zip(window_endpoints(ev), results):
            prefix = ev[: t + 1]
            try:
                encode_window(prefix)
                raised = False
            except EncodingError:
                raised = True
            assert r.gate.passed == (not raised), (seed, t, r.gate.reasons)
            assert (r.features is None) == raised
            rejected += raised
            passed += not raised
    assert rejected > 20 and passed > 20
