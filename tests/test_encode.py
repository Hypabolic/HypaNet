"""Tests for the frozen 15-feature encoder (spec v0.2 §4, §11; M0 contracts §B)."""

from __future__ import annotations

import dataclasses
import math
import random

import numpy as np
import pytest

from agent_loop_lab.encode import (
    TOKEN_LOG_SCALE,
    EncodingError,
    describe_matrix,
    encode_event_matrix,
    encode_window,
    relative_token_cost,
    validation_baseline,
)
from agent_loop_lab.schema import (
    CATEGORY_ORDER,
    EVENTS_PER_WINDOW,
    FEATURE_INDEX,
    FEATURE_NAMES,
    FEATURES_PER_EVENT,
    INPUT_DIM,
    OUTCOME_ORDER,
    CanonicalEvent,
    Outcome,
    Provenance,
    flat_index,
)
from tests.factories import h, make_event, novel_inspections, session

C = FEATURE_INDEX  # column lookup by name
TRUNCATED = Provenance(source="test:truncated", history_complete=False)


def m(events) -> np.ndarray:
    return encode_event_matrix(events)


def val(suite: str, passed: int, total: int, **kw) -> dict:
    return {"category": "VALIDATE", "outcome": "SUCCESS", "validation": (suite, passed, total), **kw}


def pad_front(specs: list[dict], n: int = 8) -> list[CanonicalEvent]:
    """Prepend novel, fully instrumented INSPECT events so the session has >= n events."""
    fill = max(0, n - len(specs))
    return session([{} for _ in range(fill)] + list(specs))


# ---------------------------------------------------------------------------
# Feature order and layout
# ---------------------------------------------------------------------------

SPEC_FEATURE_ORDER = (
    "cat_INSPECT",
    "cat_MUTATE",
    "cat_VALIDATE",
    "cat_EXTERNAL",
    "cat_CONTROL",
    "out_SUCCESS",
    "out_ERROR",
    "out_NEUTRAL",
    "repeated_intent",
    "repeated_failure",
    "workspace_changed",
    "state_revisited",
    "validation_comparable",
    "validation_delta",
    "relative_token_cost",
)


def test_feature_order_matches_spec_table():
    assert FEATURE_NAMES == SPEC_FEATURE_ORDER
    assert FEATURES_PER_EVENT == 15 and EVENTS_PER_WINDOW == 8 and INPUT_DIM == 120
    assert TOKEN_LOG_SCALE == math.log1p(100000)


def test_shapes_dtype_and_flattening():
    ev = novel_inspections(8)
    mat = m(ev)
    vec = encode_window(ev)
    assert mat.shape == (8, 15) and mat.dtype == np.float32
    assert vec.shape == (120,) and vec.dtype == np.float32
    assert vec.flags["C_CONTIGUOUS"]
    assert vec.tobytes() == mat.reshape(-1).tobytes()
    # flat layout = position * 15 + feature index, oldest first
    for pos in range(8):
        for name in FEATURE_NAMES:
            assert vec[flat_index(pos, name)] == mat[pos, C[name]]


@pytest.mark.parametrize("cat", [c.value for c in CATEGORY_ORDER])
@pytest.mark.parametrize("out", [o.value for o in OUTCOME_ORDER])
def test_one_hot_columns(cat, out):
    ev = pad_front([{"category": cat, "outcome": out}])
    row = m(ev)[7]
    cats = row[0:5]
    outs = row[5:8]
    assert cats.sum() == 1.0 and outs.sum() == 1.0
    assert row[C[f"cat_{cat}"]] == 1.0
    assert row[C[f"out_{out}"]] == 1.0
    assert set(np.unique(cats)) <= {0.0, 1.0} and set(np.unique(outs)) <= {0.0, 1.0}


def test_novel_inspections_vector():
    mat = m(novel_inspections(8))
    expected_cost = np.float32(math.log1p(150) / math.log1p(100000))
    for row in mat:
        assert row[C["cat_INSPECT"]] == 1.0 and row[C["out_SUCCESS"]] == 1.0
        for name in ("repeated_intent", "repeated_failure", "workspace_changed", "state_revisited",
                     "validation_comparable", "validation_delta"):
            assert row[C[name]] == 0.0
        assert row[C["relative_token_cost"]] == expected_cost


# ---------------------------------------------------------------------------
# repeated_intent / repeated_failure
# ---------------------------------------------------------------------------


def test_repeated_intent_basic():
    ev = session([{"intent": "A"}, {"intent": "B"}, {"intent": "A"}, {"intent": "C"},
                  {"intent": "B"}, {}, {}, {"intent": "C"}])
    col = m(ev)[:, C["repeated_intent"]]
    assert col.tolist() == [0, 0, 1, 0, 1, 0, 0, 1]


def test_repeated_intent_ignores_category_and_outcome():
    ev = pad_front([{"intent": "X", "category": "VALIDATE", "outcome": "ERROR", "failure": "f1"},
                    {"intent": "X", "category": "MUTATE", "outcome": "SUCCESS", "before": "S0", "after": "S1"}])
    assert m(ev)[7, C["repeated_intent"]] == 1.0


def test_position_zero_asymmetry():
    # every event identical intent and identical failure: position 0 still has 0
    ev = session([{"intent": "same", "outcome": "ERROR", "failure": "F"} for _ in range(8)])
    mat = m(ev)
    assert mat[0, C["repeated_intent"]] == 0.0 and mat[0, C["repeated_failure"]] == 0.0
    assert mat[1:, C["repeated_intent"]].tolist() == [1.0] * 7
    assert mat[1:, C["repeated_failure"]].tolist() == [1.0] * 7


def test_position_zero_asymmetry_even_with_repeat_before_window():
    ev = session([{"intent": "same", "outcome": "ERROR", "failure": "F"} for _ in range(12)])
    mat = m(ev)
    assert mat[0, C["repeated_intent"]] == 0.0 and mat[0, C["repeated_failure"]] == 0.0


def test_repeated_intent_window_locality():
    # event 0 and event 8 share an intent; the window for endpoint 8 is events 1..8.
    specs = [{"intent": "X"}] + [{} for _ in range(7)] + [{"intent": "X"}]
    ev = session(specs)
    assert m(ev)[7, C["repeated_intent"]] == 0.0
    # contrast: inside the window it does count
    assert m(ev[:8] + [dataclasses.replace(ev[8], intent_fingerprint=ev[1].intent_fingerprint)])[7, C["repeated_intent"]] == 1.0


def test_repeated_failure_window_locality():
    specs = [{"outcome": "ERROR", "failure": "F"}] + [{} for _ in range(7)] + [{"outcome": "ERROR", "failure": "F"}]
    ev = session(specs)
    assert m(ev)[7, C["repeated_failure"]] == 0.0
    specs2 = [{}] + [{"outcome": "ERROR", "failure": "F"}] + [{} for _ in range(6)] + [{"outcome": "ERROR", "failure": "F"}]
    assert m(session(specs2))[7, C["repeated_failure"]] == 1.0


def test_repeated_failure_requires_matching_error_fingerprint():
    ev = pad_front([
        {"intent": "A", "outcome": "ERROR", "failure": "F1"},
        {"intent": "A", "outcome": "ERROR", "failure": "F2"},  # same intent, different failure
        {"intent": "B", "outcome": "ERROR", "failure": "F1"},  # different intent, same failure
        {"intent": "A", "outcome": "SUCCESS"},                 # not an ERROR
    ])
    mat = m(ev)
    assert mat[4, C["repeated_intent"]] == 0.0 and mat[4, C["repeated_failure"]] == 0.0
    assert mat[5, C["repeated_intent"]] == 1.0 and mat[5, C["repeated_failure"]] == 0.0
    assert mat[6, C["repeated_intent"]] == 0.0 and mat[6, C["repeated_failure"]] == 1.0
    assert mat[7, C["repeated_intent"]] == 1.0 and mat[7, C["repeated_failure"]] == 0.0


# ---------------------------------------------------------------------------
# workspace_changed / state_revisited
# ---------------------------------------------------------------------------


def test_workspace_changed():
    ev = pad_front([{"category": "MUTATE", "before": "S0", "after": "S1"}, {"before": "S1"}])
    mat = m(ev)
    assert mat[6, C["workspace_changed"]] == 1.0
    assert mat[7, C["workspace_changed"]] == 0.0


def test_state_revisited_abab_oscillation():
    mut = lambda b, a: {"category": "MUTATE", "before": b, "after": a}  # noqa: E731
    ev = session([mut("A", "B"), mut("B", "A"), mut("A", "B"), mut("B", "A"),
                  {"before": "A"}, {"before": "A"}, {"before": "A"}, {"before": "A"}])
    mat = m(ev)
    assert mat[:, C["workspace_changed"]].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert mat[:, C["state_revisited"]].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]


def test_state_revisited_first_event_before_state():
    # S0 appears only as window[0].before; returning to it counts.
    mut = lambda b, a: {"category": "MUTATE", "before": b, "after": a}  # noqa: E731
    ev = session([mut("S0", "S1"), mut("S1", "S2"), mut("S2", "S0")] + [{"before": "S0"} for _ in range(5)])
    mat = m(ev)
    assert mat[:3, C["state_revisited"]].tolist() == [0, 0, 1]


def test_state_revisited_new_state_is_zero_and_position0_is_zero():
    mut = lambda b, a: {"category": "MUTATE", "before": b, "after": a}  # noqa: E731
    ev = session([mut("S0", "S1"), mut("S1", "S2"), mut("S2", "S3")] + [{"before": "S3"} for _ in range(5)])
    mat = m(ev)
    assert mat[:, C["state_revisited"]].tolist() == [0] * 8
    assert mat[:3, C["workspace_changed"]].tolist() == [1, 1, 1]


def test_state_revisited_window_locality():
    # X is held only before the window (event 0's before-state); returning to it is not a revisit.
    mut = lambda b, a: {"category": "MUTATE", "before": b, "after": a}  # noqa: E731
    specs = [mut("X", "Y")] + [{"before": "Y"} for _ in range(7)] + [mut("Y", "X")]
    ev = session(specs)
    mat = m(ev)
    assert mat[7, C["workspace_changed"]] == 1.0
    assert mat[7, C["state_revisited"]] == 0.0


def test_state_revisited_unchanged_event_is_zero():
    # after equals an earlier digest but nothing changed -> state_revisited must be 0
    ev = pad_front([{"category": "MUTATE", "before": "S0", "after": "S1"}, {"before": "S0"}])
    assert m(ev)[7, C["state_revisited"]] == 0.0


# ---------------------------------------------------------------------------
# validation_comparable / validation_delta
# ---------------------------------------------------------------------------


def test_validation_no_prior_result_not_comparable():
    ev = pad_front([val("unit", 5, 10)])
    row = m(ev)[7]
    assert row[C["cat_VALIDATE"]] == 1.0
    assert row[C["validation_comparable"]] == 0.0 and row[C["validation_delta"]] == 0.0


def test_validation_delta_within_window():
    ev = pad_front([val("unit", 5, 10), val("unit", 7, 10)])
    row = m(ev)[7]
    assert row[C["validation_comparable"]] == 1.0
    assert row[C["validation_delta"]] == np.float32(7 / 10 - 5 / 10)


def test_validation_baseline_from_before_window():
    specs = [val("unit", 2, 10)] + [{} for _ in range(7)] + [val("unit", 6, 10)]
    ev = session(specs)
    row = m(ev)[7]
    assert row[C["validation_comparable"]] == 1.0
    assert row[C["validation_delta"]] == np.float32(6 / 10 - 2 / 10)


def test_validation_baseline_far_before_window():
    specs = [val("unit", 9, 10)] + [{} for _ in range(30)] + [val("unit", 3, 10)]
    row = m(session(specs))[7]
    assert row[C["validation_comparable"]] == 1.0
    assert row[C["validation_delta"]] == np.float32(3 / 10 - 9 / 10)


def test_different_suites_do_not_compare():
    ev = pad_front([val("unit", 5, 10), val("integration", 7, 10)])
    row = m(ev)[7]
    assert row[C["validation_comparable"]] == 0.0 and row[C["validation_delta"]] == 0.0


def test_most_recent_same_suite_baseline_is_used():
    ev = pad_front([val("unit", 1, 10), val("other", 9, 10), val("unit", 5, 10), val("other", 2, 10),
                    {"category": "VALIDATE"},  # VALIDATE without a result: not a baseline
                    val("unit", 9, 10)])
    row = m(ev)[7]
    assert row[C["validation_comparable"]] == 1.0
    assert row[C["validation_delta"]] == np.float32(9 / 10 - 5 / 10)


def test_comparable_with_zero_delta():
    ev = pad_front([val("unit", 5, 10), val("unit", 5, 10)])
    row = m(ev)[7]
    assert row[C["validation_comparable"]] == 1.0 and row[C["validation_delta"]] == 0.0


def test_delta_uses_ratios_with_different_totals():
    ev = pad_front([val("unit", 1, 3), val("unit", 2, 7)])
    assert m(ev)[7, C["validation_delta"]] == np.float32(2 / 7 - 1 / 3)


@pytest.mark.parametrize("prev,cur,expected", [((0, 10), (10, 10), 1.0), ((10, 10), (0, 10), -1.0),
                                               ((0, 1), (1, 1), 1.0), ((1, 1), (0, 5), -1.0)])
def test_delta_clip_boundaries(prev, cur, expected):
    ev = pad_front([val("unit", *prev), val("unit", *cur)])
    d = m(ev)[7, C["validation_delta"]]
    assert d == np.float32(expected)
    assert -1.0 <= d <= 1.0


def test_unobserved_baseline_raises():
    specs = [{"category": "VALIDATE", "validation_produced": True, "validation_suite_id": "unit",
              "validation_passed": None, "validation_total": 10}] + [{} for _ in range(7)] + [val("unit", 5, 10)]
    with pytest.raises(EncodingError):
        m(session(specs))


UNKNOWN_SUITE = {"category": "VALIDATE", "validation_produced": True, "validation_suite_id": None,
                 "validation_passed": 10, "validation_total": 10}


def test_unknown_suite_result_before_window_with_older_baseline_raises():
    # e1's suite is unobserved and it is more recent than e0: the 'unit' baseline is unknown,
    # so delta must not be computed against e0.
    specs = [val("unit", 5, 10), UNKNOWN_SUITE] + [{} for _ in range(7)] + [val("unit", 10, 10)]
    with pytest.raises(EncodingError):
        m(session(specs))


def test_unknown_suite_result_before_window_without_baseline_raises():
    # must not be silently encoded as "no comparable result" (comparable = delta = 0)
    specs = [UNKNOWN_SUITE] + [{} for _ in range(7)] + [val("unit", 10, 10)]
    with pytest.raises(EncodingError):
        m(session(specs))


def test_unknown_suite_result_older_than_baseline_is_ignored():
    specs = [UNKNOWN_SUITE, val("unit", 4, 10)] + [{} for _ in range(7)] + [val("unit", 5, 10)]
    row = m(session(specs))[7]
    assert row[C["validation_comparable"]] == 1.0
    assert row[C["validation_delta"]] == np.float32(5 / 10 - 4 / 10)


def test_validation_baseline_helper():
    ev = session([val("unit", 5, 10), UNKNOWN_SUITE, val("unit", 7, 10), val("other", 1, 2)])
    assert validation_baseline(ev, 0) == (None, True)
    assert validation_baseline(ev, 2) == (None, False)
    assert validation_baseline(ev, 3) == (None, False)
    ok = session([val("unit", 5, 10), val("other", 1, 2), val("unit", 7, 10)])
    assert validation_baseline(ok, 2) == (ok[0], True)
    assert validation_baseline(ok, 1) == (None, True)
    with pytest.raises(ValueError):
        validation_baseline(session([{}]), 0)


def test_truncated_history_without_baseline_raises():
    ev = [make_event(i, provenance=TRUNCATED) for i in range(10, 17)]
    ev.append(make_event(17, "VALIDATE", validation=("unit", 5, 10), provenance=TRUNCATED))
    with pytest.raises(EncodingError):
        m(ev)


def test_truncated_history_with_observed_baseline_encodes():
    # baseline (index 10) sits before the window 11..18, so no window event lacks a baseline
    ev = [make_event(10, "VALIDATE", validation=("unit", 3, 10), provenance=TRUNCATED)]
    ev += [make_event(i, provenance=TRUNCATED) for i in range(11, 18)]
    ev.append(make_event(18, "VALIDATE", validation=("unit", 5, 10), provenance=TRUNCATED))
    row = m(ev)[7]
    assert row[C["validation_comparable"]] == 1.0
    assert row[C["validation_delta"]] == np.float32(5 / 10 - 3 / 10)


def test_truncated_history_non_validation_window_encodes():
    ev = [make_event(i, provenance=TRUNCATED) for i in range(10, 18)]
    assert m(ev).shape == (8, 15)


# ---------------------------------------------------------------------------
# relative_token_cost
# ---------------------------------------------------------------------------


def _cost(tokens) -> np.float32:
    return m(pad_front([{"tokens": tokens}]))[7, C["relative_token_cost"]]


def test_token_cost_zero_is_exactly_zero():
    c = _cost((0, 0))
    assert c == 0.0 and c.tobytes() == np.float32(0.0).tobytes()
    assert relative_token_cost(0, 0) == 0.0


@pytest.mark.parametrize("tokens", [(100000, 0), (0, 100000), (60000, 40000)])
def test_token_cost_saturation_point_is_one(tokens):
    assert _cost(tokens) == 1.0


@pytest.mark.parametrize("tokens", [(100001, 0), (10**9, 10**9), (99999, 2)])
def test_token_cost_clipped_to_one(tokens):
    assert _cost(tokens) == 1.0


def test_token_cost_value_and_sum_before_log():
    assert _cost((100, 50)) == np.float32(math.log1p(150) / math.log1p(100000))
    assert _cost((150, 0)) == _cost((0, 150)) == _cost((75, 75))


def test_token_cost_monotone():
    totals = [0, 1, 2, 5, 10, 100, 1000, 9999, 50000, 99999, 100000, 100001, 10**6]
    values = [relative_token_cost(t, 0) for t in totals]
    assert all(a <= b for a, b in zip(values, values[1:]))
    assert all(a < b for a, b in zip(values[:10], values[1:11]))  # strictly below saturation
    encoded = [float(_cost((t, 0))) for t in totals]
    assert all(a <= b for a, b in zip(encoded, encoded[1:]))
    assert all(0.0 <= v <= 1.0 for v in encoded)


# ---------------------------------------------------------------------------
# Missing observations: raise, never zero-fill
# ---------------------------------------------------------------------------

MISSING_CASES = {
    "input_tokens": {"tokens": (None, 5)},
    "output_tokens": {"tokens": (5, None)},
    "both_tokens": {"tokens": (None, None)},
    "before_and_after_digest": {"before": None},
    "after_digest": {"workspace_state_after": None},
    "before_digest": {"workspace_state_before": None, "after": "S0"},
    "failure_fingerprint": {"outcome": "ERROR", "failure_fingerprint": None},
    "validation_suite": {"category": "VALIDATE", "validation_produced": True, "validation_suite_id": None,
                         "validation_passed": 1, "validation_total": 2},
    "validation_passed": {"category": "VALIDATE", "validation_produced": True, "validation_suite_id": "u",
                          "validation_passed": None, "validation_total": 2},
    "validation_total": {"category": "VALIDATE", "validation_produced": True, "validation_suite_id": "u",
                         "validation_passed": 1, "validation_total": None},
}


@pytest.mark.parametrize("case", sorted(MISSING_CASES))
@pytest.mark.parametrize("pos", [0, 3, 7])
def test_missing_observation_raises(case, pos):
    specs = [{} for _ in range(8)]
    specs[pos] = MISSING_CASES[case]
    ev = session(specs)
    with pytest.raises(EncodingError):
        encode_event_matrix(ev)
    with pytest.raises(EncodingError):
        encode_window(ev)


@pytest.mark.parametrize("n", range(0, 8))
def test_short_prefix_raises(n):
    with pytest.raises(EncodingError):
        encode_window(novel_inspections(n))


@pytest.mark.parametrize("case", sorted(MISSING_CASES))
def test_missing_observation_before_window_is_ignored(case):
    specs = [MISSING_CASES[case]] + [{} for _ in range(8)]
    ev = session(specs)
    assert encode_window(ev).shape == (120,)


def test_encoding_error_is_value_error():
    assert issubclass(EncodingError, ValueError)


# ---------------------------------------------------------------------------
# Determinism and causality
# ---------------------------------------------------------------------------


def _mixed_session() -> list[CanonicalEvent]:
    return session([
        {"intent": "r1"},
        val("unit", 3, 10, intent="t"),
        {"category": "MUTATE", "before": "S0", "after": "S1", "intent": "e1", "tokens": (900, 120)},
        val("unit", 5, 10, intent="t", before="S1", outcome="ERROR", failure="assert"),
        {"category": "MUTATE", "before": "S1", "after": "S0", "intent": "rv", "tokens": (0, 0)},
        val("unit", 3, 10, intent="t", before="S0", outcome="ERROR", failure="assert"),
        {"category": "EXTERNAL", "outcome": "ERROR", "failure": "timeout", "before": "S0", "intent": "poll"},
        {"category": "EXTERNAL", "outcome": "ERROR", "failure": "timeout", "before": "S0", "intent": "poll"},
        {"category": "CONTROL", "outcome": "NEUTRAL", "before": "S0", "intent": "plan"},
        {"category": "MUTATE", "before": "S0", "after": "S1", "intent": "e1", "tokens": (90000, 90000)},
    ])


def test_byte_identical_determinism():
    ev = _mixed_session()
    a = encode_window(ev)
    b = encode_window(list(ev))
    rebuilt = [CanonicalEvent.from_json(e.to_json()) for e in ev]
    c = encode_window(rebuilt)
    assert a.tobytes() == b.tobytes() == c.tobytes()
    assert encode_event_matrix(tuple(ev)).tobytes() == a.reshape(8, 15).tobytes()


def test_mixed_session_expected_values():
    mat = m(_mixed_session())  # window = events 2..9
    assert mat[:, C["repeated_intent"]].tolist() == [0, 0, 0, 1, 0, 1, 0, 1]
    assert mat[:, C["repeated_failure"]].tolist() == [0, 0, 0, 1, 0, 1, 0, 0]
    assert mat[:, C["workspace_changed"]].tolist() == [1, 0, 1, 0, 0, 0, 0, 1]
    assert mat[:, C["state_revisited"]].tolist() == [0, 0, 1, 0, 0, 0, 0, 1]
    assert mat[:, C["validation_comparable"]].tolist() == [0, 1, 0, 1, 0, 0, 0, 0]
    assert mat[1, C["validation_delta"]] == np.float32(5 / 10 - 3 / 10)  # baseline before window
    assert mat[3, C["validation_delta"]] == np.float32(3 / 10 - 5 / 10)
    assert mat[2, C["relative_token_cost"]] == 0.0
    assert mat[7, C["relative_token_cost"]] == 1.0


def test_causality_later_events_do_not_change_earlier_windows():
    ev = _mixed_session()
    t = 8
    base = encode_window(ev[: t + 1]).tobytes()
    extended = ev + [make_event(10, "VALIDATE", validation=("unit", 10, 10), intent="t")]
    assert encode_window(extended[: t + 1]).tobytes() == base
    mutated = list(ev)
    mutated[9] = dataclasses.replace(ev[9], outcome=Outcome.SUCCESS, intent_fingerprint=h("other"),
                                     workspace_state_after=h("S9"), input_tokens=0, output_tokens=0)
    assert encode_window(mutated[: t + 1]).tobytes() == base


def test_describe_matrix():
    mat = m(_mixed_session())
    text = describe_matrix(mat)
    for name in FEATURE_NAMES:
        assert name in text
    assert len(text.splitlines()) == 2 + 8
    assert describe_matrix(mat.reshape(-1)) == text
    with pytest.raises(ValueError):
        describe_matrix(np.zeros((7, 15), dtype=np.float32))


# ---------------------------------------------------------------------------
# Invariants (hand-built and randomized property test)
# ---------------------------------------------------------------------------


def assert_invariants(mat: np.ndarray) -> None:
    assert mat.shape == (8, 15) and mat.dtype == np.float32
    assert np.all(np.isfinite(mat))
    cats, outs = mat[:, 0:5], mat[:, 5:8]
    assert np.all(cats.sum(axis=1) == 1.0) and np.all(outs.sum(axis=1) == 1.0)
    binary = [i for i, n in enumerate(FEATURE_NAMES) if n not in ("validation_delta", "relative_token_cost")]
    assert np.all(np.isin(mat[:, binary], (0.0, 1.0)))
    assert np.all(mat[:, C["repeated_failure"]] <= mat[:, C["out_ERROR"]])
    assert np.all(mat[:, C["state_revisited"]] <= mat[:, C["workspace_changed"]])
    assert np.all(mat[:, C["validation_comparable"]] <= mat[:, C["cat_VALIDATE"]])
    delta = mat[:, C["validation_delta"]]
    assert np.all((delta == 0.0) | (mat[:, C["validation_comparable"]] == 1.0))
    assert np.all((delta >= -1.0) & (delta <= 1.0))
    non_delta = [i for i in range(15) if i != C["validation_delta"]]
    assert np.all((mat[:, non_delta] >= 0.0) & (mat[:, non_delta] <= 1.0))
    assert mat[0, C["repeated_intent"]] == 0.0 and mat[0, C["repeated_failure"]] == 0.0
    assert mat[0, C["state_revisited"]] == 0.0


def reference_encode(prefix: list[CanonicalEvent]) -> np.ndarray:
    """Independent, literal re-derivation of spec §4 used as a differential oracle."""
    w = prefix[-8:]
    start = len(prefix) - 8
    cats = ["INSPECT", "MUTATE", "VALIDATE", "EXTERNAL", "CONTROL"]
    outs = ["SUCCESS", "ERROR", "NEUTRAL"]
    rows = []
    for i, e in enumerate(w):
        cat = [1.0 if e.operation_category.value == c else 0.0 for c in cats]
        out = [1.0 if e.outcome.value == o else 0.0 for o in outs]
        ri = 1.0 if any(w[j].intent_fingerprint == e.intent_fingerprint for j in range(i)) else 0.0
        rf = 1.0 if (e.outcome.value == "ERROR" and any(
            w[j].outcome.value == "ERROR" and w[j].failure_fingerprint == e.failure_fingerprint for j in range(i)
        )) else 0.0
        wc = 1.0 if e.workspace_state_before != e.workspace_state_after else 0.0
        digests = [w[0].workspace_state_before]
        for j in range(i):
            digests += [w[j].workspace_state_before, w[j].workspace_state_after]
        sr = 1.0 if (wc == 1.0 and e.workspace_state_after in digests) else 0.0
        comp, delta = 0.0, 0.0
        if e.validation_produced:
            same = [p for p in prefix[: start + i] if p.validation_produced
                    and p.validation_suite_id == e.validation_suite_id]
            if same:
                prev = same[-1]
                comp = 1.0
                delta = float(np.clip(e.validation_passed / e.validation_total
                                      - prev.validation_passed / prev.validation_total, -1.0, 1.0))
        cost = min(1.0, math.log1p(e.input_tokens + e.output_tokens) / math.log1p(100000))
        rows.append(cat + out + [ri, rf, wc, sr, comp, delta, cost])
    return np.array(rows, dtype=np.float64).astype(np.float32)


def test_hand_built_invariants():
    ev = _mixed_session()
    for t in range(7, len(ev)):
        mat = m(ev[: t + 1])
        assert_invariants(mat)
        assert mat.tobytes() == reference_encode(ev[: t + 1]).tobytes()


def random_session(rng: random.Random) -> list[CanonicalEvent]:
    n = rng.randint(8, 30)
    intents = [f"I{k}" for k in range(rng.randint(1, 5))]
    failures = [f"F{k}" for k in range(rng.randint(1, 3))]
    states = [f"S{k}" for k in range(rng.randint(1, 4))]
    suites = [f"suite{k}" for k in range(rng.randint(1, 2))]
    state = rng.choice(states)
    specs = []
    for _ in range(n):
        cat = rng.choice([c.value for c in CATEGORY_ORDER])
        out = rng.choice([o.value for o in OUTCOME_ORDER])
        before = state if rng.random() < 0.85 else rng.choice(states)
        p_change = 0.7 if cat == "MUTATE" else 0.05
        after = rng.choice(states) if rng.random() < p_change else before
        spec: dict = {"category": cat, "outcome": out, "intent": rng.choice(intents), "before": before, "after": after}
        if out == "ERROR":
            spec["failure"] = rng.choice(failures)
        if cat == "VALIDATE" and rng.random() < 0.7:
            total = rng.randint(1, 10)
            spec["validation"] = (rng.choice(suites), rng.randint(0, total), total)

        def tok() -> int:
            r = rng.random()
            return 0 if r < 0.15 else rng.randint(0, 300) if r < 0.6 else rng.randint(0, 250000)

        spec["tokens"] = (tok(), tok())
        specs.append(spec)
        state = after
    return session(specs)


@pytest.mark.parametrize("seed", range(8))
def test_randomized_invariants_property(seed):
    from agent_loop_lab.windows import encode_session, gate, validate_sequence

    rng = random.Random(seed)
    windows_checked = 0
    for _ in range(60):
        ev = random_session(rng)
        validate_sequence(ev)
        endpoints, feats = encode_session(ev)
        assert endpoints == list(range(7, len(ev)))
        assert feats.shape == (len(ev) - 7, 120) and feats.dtype == np.float32
        for row_i, t in enumerate(endpoints):
            prefix = ev[: t + 1]
            assert gate(prefix).passed
            mat = encode_event_matrix(prefix)
            assert_invariants(mat)
            assert mat.tobytes() == reference_encode(prefix).tobytes()
            assert feats[row_i].tobytes() == mat.reshape(-1).tobytes()
            windows_checked += 1
    assert windows_checked > 300


def test_bare_window_slice_is_treated_as_truncated():
    # A slice that does not start at sequence_index 0 cannot see baselines before it, so a
    # validation event without an in-slice baseline must raise rather than silently get comparable=0.
    specs = [val("unit", 2, 10)] + [{} for _ in range(7)] + [val("unit", 6, 10)]
    ev = session(specs)
    with pytest.raises(EncodingError):
        encode_window(ev[1:])
    assert encode_window(ev)[flat_index(7, "validation_comparable")] == 1.0


def test_encode_does_not_import_windows():
    import ast
    import inspect as _inspect

    import agent_loop_lab.encode as enc

    tree = ast.parse(_inspect.getsource(enc))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    assert not any("windows" in n for n in names)
