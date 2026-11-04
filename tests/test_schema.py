"""Canonical event / oracle schema tests (spec v0.2 §3, §4, §6; §11 Leakage)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from agent_loop_lab.schema import (
    CATEGORY_ORDER,
    CLASS_NAMES,
    EVENTS_PER_WINDOW,
    FEATURE_DEFINITIONS,
    FEATURE_INDEX,
    FEATURE_NAMES,
    FEATURES_PER_EVENT,
    FORBIDDEN_INFERENCE_FIELDS,
    INPUT_DIM,
    OUTCOME_ORDER,
    CanonicalEvent,
    ClassLabel,
    LatentEffect,
    OperationCategory,
    OracleLabel,
    OracleRecord,
    Outcome,
    Provenance,
    SchemaError,
    flat_index,
)
from tests.factories import h, make_event

# -- leakage --------------------------------------------------------------------


@pytest.mark.parametrize("extra", ["label", "scenario_family", "event_effects", "progress", "latent_effect", "foo"])
def test_from_dict_rejects_unknown_fields(extra):
    d = make_event(0).to_dict()
    d[extra] = "x"
    with pytest.raises(SchemaError, match="unknown CanonicalEvent fields"):
        CanonicalEvent.from_dict(d)


def test_provenance_from_dict_rejects_unknown_fields():
    d = Provenance(source="test").to_dict()
    d["scenario"] = "5"
    with pytest.raises(SchemaError, match="unknown provenance fields"):
        Provenance.from_dict(d)
    ev = make_event(0).to_dict()
    ev["provenance"]["oracle"] = {}
    with pytest.raises(SchemaError):
        CanonicalEvent.from_dict(ev)


def test_no_inference_field_is_forbidden():
    for cls in (CanonicalEvent, Provenance):
        names = {f.name for f in dataclasses.fields(cls)}
        assert not names & FORBIDDEN_INFERENCE_FIELDS, (cls.__name__, names & FORBIDDEN_INFERENCE_FIELDS)
    assert not set(make_event(0).to_dict()) & FORBIDDEN_INFERENCE_FIELDS
    assert not set(make_event(0).to_dict()["provenance"]) & FORBIDDEN_INFERENCE_FIELDS


def test_oracle_only_fields_are_in_forbidden_set():
    oracle_fields = {"window_labels", "scenario_family", "event_effects"}
    assert oracle_fields <= {f.name for f in dataclasses.fields(OracleRecord)}
    assert oracle_fields <= FORBIDDEN_INFERENCE_FIELDS


# -- validation invariants -----------------------------------------------------------


def test_factory_event_is_valid_and_frozen():
    e = make_event(0)
    assert e.operation_category is OperationCategory.INSPECT
    assert e.outcome is Outcome.SUCCESS
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.outcome = Outcome.ERROR  # type: ignore[misc]


@pytest.mark.parametrize("name", ["session_id", "agent_id", "task_id", "event_id", "timestamp", "operation_family"])
def test_required_strings_non_empty(name):
    with pytest.raises(SchemaError):
        make_event(0, **{name: ""})
    with pytest.raises(SchemaError):
        make_event(0, **{name: 3})


@pytest.mark.parametrize("bad", [-1, True, 1.0, "0"])
def test_sequence_index_must_be_nonneg_int(bad):
    with pytest.raises(SchemaError):
        make_event(0, sequence_index=bad)


@pytest.mark.parametrize("field", ["intent_fingerprint", "workspace_state_before", "workspace_state_after"])
@pytest.mark.parametrize("bad", ["abc", h("x")[:-1], "z" * 64, 123])
def test_digest_fields_must_be_64_hex(field, bad):
    with pytest.raises(SchemaError):
        make_event(0, **{field: bad})


def test_intent_fingerprint_required():
    with pytest.raises(SchemaError):
        make_event(0, intent_fingerprint=None)


def test_workspace_digests_may_be_unobserved():
    e = make_event(0, before=None, after=None)
    assert e.workspace_state_before is None and e.workspace_state_after is None


def test_factory_after_none_means_unobserved_not_unchanged():
    # Explicit after=None is an unobserved after-digest; omitting after means "same as before".
    assert make_event(3, after=None).workspace_state_after is None
    assert make_event(3, after=None).workspace_state_before == h("S0")
    assert make_event(3).workspace_state_after == h("S0")
    assert make_event(3, before="S1").workspace_state_after == h("S1")
    assert make_event(3, before=None).workspace_state_after is None


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("bad", [-1, True, 1.5, "10"])
def test_token_counts_nonneg_int_or_none(field, bad):
    with pytest.raises(SchemaError):
        make_event(0, **{field: bad})
    assert getattr(make_event(0, **{field: None}), field) is None
    assert getattr(make_event(0, **{field: 0}), field) == 0


def test_step_tokens():
    assert make_event(0, tokens=(10, 5)).step_tokens == 15
    assert make_event(0, tokens=(0, 0)).step_tokens == 0
    assert make_event(0, tokens=(None, 5)).step_tokens is None
    assert make_event(0, tokens=(5, None)).step_tokens is None


def test_failure_fingerprint_only_with_error():
    assert make_event(0, "INSPECT", "ERROR", failure="boom").failure_fingerprint == h("boom")
    for outcome in ("SUCCESS", "NEUTRAL"):
        with pytest.raises(SchemaError, match="failure_fingerprint only allowed"):
            make_event(0, "INSPECT", outcome, failure_fingerprint=h("boom"))


def test_error_without_failure_fingerprint_is_allowed_by_schema():
    # Missing fingerprint on ERROR is an instrumentation gap rejected by the gate (§5.2), not a schema error.
    e = make_event(0, "INSPECT", "ERROR", failure_fingerprint=None)
    assert e.failure_fingerprint is None


def test_validation_result_requires_validate_category():
    with pytest.raises(SchemaError, match="VALIDATE"):
        make_event(0, "INSPECT", validation=("unit", 1, 2))
    e = make_event(0, "VALIDATE", validation=("unit", 1, 2))
    assert (e.validation_produced, e.validation_suite_id, e.validation_passed, e.validation_total) == (True, "unit", 1, 2)


@pytest.mark.parametrize(
    "kw",
    [{"validation_suite_id": "unit"}, {"validation_passed": 1}, {"validation_total": 2}],
)
def test_validation_fields_must_be_none_without_result(kw):
    with pytest.raises(SchemaError, match="must be None"):
        make_event(0, "VALIDATE", **kw)


def test_validation_counts():
    with pytest.raises(SchemaError, match="validation_total must be >= 1"):
        make_event(0, "VALIDATE", validation=("unit", 0, 0))
    with pytest.raises(SchemaError, match="validation_passed > validation_total"):
        make_event(0, "VALIDATE", validation=("unit", 3, 2))
    with pytest.raises(SchemaError):
        make_event(0, "VALIDATE", validation=("unit", -1, 2))
    assert make_event(0, "VALIDATE", validation=("unit", 0, 1)).validation_passed == 0
    assert make_event(0, "VALIDATE", validation=("unit", 2, 2)).validation_passed == 2


def test_unobserved_validation_counts_are_allowed_by_schema():
    # None means unobserved; the gate (§5.2) rejects it, the schema keeps it.
    e = make_event(0, "VALIDATE", validation=("unit", None, None))
    assert e.validation_produced and e.validation_passed is None and e.validation_total is None
    e = make_event(0, "VALIDATE", validation=(None, 1, 2))
    assert e.validation_suite_id is None


def test_validation_produced_must_be_bool():
    with pytest.raises(SchemaError):
        make_event(0, "VALIDATE", validation_produced=1)


def test_provenance_required_and_typed():
    with pytest.raises(SchemaError):
        make_event(0, provenance={"source": "x"})
    with pytest.raises(SchemaError):
        Provenance(source="")
    with pytest.raises(SchemaError):
        Provenance(source="x", history_complete="yes")  # type: ignore[arg-type]
    p = Provenance(source="x", instrumentation_flags=["a", "b"], span_ids=["s1"])
    assert p.instrumentation_flags == ("a", "b") and p.span_ids == ("s1",)
    assert Provenance(source="x").history_complete is True


# -- enum coercion -------------------------------------------------------------


def _with(**changes):
    """Build via the real constructor (from_dict), bypassing factory coercion."""
    d = make_event(0).to_dict()
    d.update(changes)
    return CanonicalEvent.from_dict(d)


@pytest.mark.parametrize("cat", [c.value for c in OperationCategory])
def test_category_string_coerced_to_enum(cat):
    e = _with(operation_category=cat)
    assert isinstance(e.operation_category, OperationCategory)
    assert e.operation_category is OperationCategory(cat)


@pytest.mark.parametrize("out", [o.value for o in Outcome])
def test_outcome_string_coerced_to_enum(out):
    e = _with(outcome=out)
    assert isinstance(e.outcome, Outcome)
    assert e.outcome is Outcome(out)


def test_enum_instances_accepted_unchanged():
    e = _with(operation_category=OperationCategory.MUTATE, outcome=Outcome.NEUTRAL)
    assert e.operation_category is OperationCategory.MUTATE and e.outcome is Outcome.NEUTRAL


@pytest.mark.parametrize(
    "field,bad",
    [("operation_category", "inspect"), ("operation_category", "READ"), ("outcome", "FAIL"), ("outcome", 1), ("outcome", None)],
)
def test_invalid_enum_values_rejected(field, bad):
    with pytest.raises(SchemaError):
        _with(**{field: bad})


def test_category_and_outcome_orders():
    assert [c.value for c in CATEGORY_ORDER] == ["INSPECT", "MUTATE", "VALIDATE", "EXTERNAL", "CONTROL"]
    assert [o.value for o in OUTCOME_ORDER] == ["SUCCESS", "ERROR", "NEUTRAL"]
    assert [f"cat_{c.value}" for c in CATEGORY_ORDER] == list(FEATURE_NAMES[0:5])
    assert [f"out_{o.value}" for o in OUTCOME_ORDER] == list(FEATURE_NAMES[5:8])


# -- JSON round trip -------------------------------------------------------------


def _rich_event() -> CanonicalEvent:
    return make_event(
        3,
        "VALIDATE",
        "ERROR",
        failure="AssertionError@tests/test_x.py::test_a",
        before="S1",
        after="S2",
        validation=("unit-suite", 7, 10),
        tokens=(1234, 0),
        family="run_tests",
        session_id="sess-ü-1",
        provenance=Provenance(
            source="simulator:0.2.0",
            history_complete=False,
            instrumentation_flags=("tokens_estimated",),
            trace_id="t-1",
            span_ids=("a", "b"),
        ),
    )


@pytest.mark.parametrize("event", [make_event(0), _rich_event(), make_event(1, before=None, tokens=(None, None))])
def test_json_round_trip_is_byte_stable(event):
    s = event.to_json()
    back = CanonicalEvent.from_json(s)
    assert back == event
    assert back.to_json() == s
    assert CanonicalEvent.from_dict(event.to_dict()) == event
    # Canonical form: sorted keys, no insignificant whitespace, non-ASCII kept.
    assert s == json.dumps(json.loads(s), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    # Key order of the input doesn't matter.
    shuffled = dict(reversed(list(json.loads(s).items())))
    assert CanonicalEvent.from_dict(shuffled).to_json() == s


def test_json_keeps_non_ascii_and_enum_values():
    s = _rich_event().to_json()
    assert "sess-ü-1" in s
    d = json.loads(s)
    assert d["operation_category"] == "VALIDATE" and d["outcome"] == "ERROR"
    assert d["provenance"]["span_ids"] == ["a", "b"]


# -- OracleRecord / labels ---------------------------------------------------------


def _record() -> OracleRecord:
    return OracleRecord(
        session_id="sess-1",
        generator_version="synth:0.2.0",
        seed=42,
        scenario_family="5",
        event_effects=(LatentEffect.NONE, LatentEffect.REDUNDANT, LatentEffect.IN_POLICY_RETRY, LatentEffect.PROGRESS),
        window_labels={11: OracleLabel.STALLED, 7: OracleLabel.PRODUCTIVE, 9: OracleLabel.INSUFFICIENT_AMBIGUOUS},
        dropped_observations=(2,),
        latent_trace={"p": [0.0, 0.1]},
    )


def test_oracle_record_round_trip():
    r = _record()
    d = r.to_dict()
    assert list(d["window_labels"]) == ["7", "9", "11"]  # sorted numerically, str keys for JSON
    s = json.dumps(d, sort_keys=True)
    back = OracleRecord.from_dict(json.loads(s))
    assert back == r
    assert back.window_labels == {7: OracleLabel.PRODUCTIVE, 9: OracleLabel.INSUFFICIENT_AMBIGUOUS, 11: OracleLabel.STALLED}
    assert all(isinstance(e, LatentEffect) for e in back.event_effects)
    assert json.dumps(back.to_dict(), sort_keys=True) == s


def test_oracle_record_from_dict_defaults():
    r = OracleRecord.from_dict(
        {"session_id": "s", "generator_version": "g", "seed": "3", "scenario_family": "1", "event_effects": ["NONE"]}
    )
    assert r.seed == 3 and r.window_labels == {} and r.dropped_observations == () and r.latent_trace == {}


def test_oracle_label_is_scored_and_to_class():
    scored = {OracleLabel.PRODUCTIVE, OracleLabel.RECOVERY, OracleLabel.STALLED, OracleLabel.OSCILLATING}
    for lab in OracleLabel:
        assert lab.is_scored is (lab in scored)
    for lab in scored:
        c = lab.to_class()
        assert isinstance(c, ClassLabel) and c.name == lab.name
    for lab in (OracleLabel.INSUFFICIENT_INSTRUMENTATION, OracleLabel.INSUFFICIENT_AMBIGUOUS):
        with pytest.raises(ValueError):
            lab.to_class()


def test_class_label_canonical_order():
    assert [(c.name, int(c)) for c in ClassLabel] == [("PRODUCTIVE", 0), ("RECOVERY", 1), ("STALLED", 2), ("OSCILLATING", 3)]
    assert CLASS_NAMES == ("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING")


def test_latent_effects_match_spec():
    assert {e.value for e in LatentEffect} == {
        "PROGRESS", "NEW_FACT", "RESOLVES_EPISODE", "IN_POLICY_RETRY", "IN_POLICY_POLL", "REVERSION", "REDUNDANT", "NONE",
    }


# -- feature layout --------------------------------------------------------------

SPEC_FEATURE_ORDER = (
    "cat_INSPECT", "cat_MUTATE", "cat_VALIDATE", "cat_EXTERNAL", "cat_CONTROL",
    "out_SUCCESS", "out_ERROR", "out_NEUTRAL",
    "repeated_intent", "repeated_failure", "workspace_changed", "state_revisited",
    "validation_comparable", "validation_delta", "relative_token_cost",
)  # fmt: skip


def test_feature_names_length_and_order():
    assert FEATURES_PER_EVENT == 15 and EVENTS_PER_WINDOW == 8 and INPUT_DIM == 120
    assert FEATURE_NAMES == SPEC_FEATURE_ORDER
    assert len(set(FEATURE_NAMES)) == 15
    assert FEATURE_INDEX == {n: i for i, n in enumerate(SPEC_FEATURE_ORDER)}
    assert list(FEATURE_DEFINITIONS) == list(FEATURE_NAMES)
    assert all(isinstance(v, str) and v for v in FEATURE_DEFINITIONS.values())


def test_flat_index():
    assert flat_index(0, 0) == 0
    assert flat_index(0, "cat_INSPECT") == 0
    assert flat_index(7, "relative_token_cost") == 119
    assert flat_index(3, "repeated_intent") == 3 * 15 + 8
    assert flat_index(1, "validation_delta") == 28
    seen = [flat_index(p, f) for p in range(8) for f in FEATURE_NAMES]
    assert seen == list(range(120))
    for p in range(8):
        for i, name in enumerate(FEATURE_NAMES):
            assert flat_index(p, name) == flat_index(p, i)


@pytest.mark.parametrize("pos,feat", [(8, 0), (-1, 0), (0, 15), (0, -1)])
def test_flat_index_out_of_range(pos, feat):
    with pytest.raises(IndexError):
        flat_index(pos, feat)


def test_flat_index_unknown_name():
    with pytest.raises(KeyError):
        flat_index(0, "progress")


# -- digests must be canonical lowercase hex (compared by string equality downstream) ---

_NON_CANONICAL_DIGESTS = [
    h("x").upper(),
    "0x" + h("x")[:62],
    h("x")[:30] + "_" + h("x")[31:],
    " " + h("x")[1:],
    h("x")[:63] + "\n",
    h("x")[:63] + "g",
]
_NON_CANONICAL_IDS = ["upper", "0x", "underscore", "space", "newline", "non-hex"]


@pytest.mark.parametrize("variant", _NON_CANONICAL_DIGESTS, ids=_NON_CANONICAL_IDS)
@pytest.mark.parametrize(
    "field", ["intent_fingerprint", "failure_fingerprint", "workspace_state_before", "workspace_state_after"]
)
def test_digest_must_be_canonical_lowercase_hex(field, variant):
    with pytest.raises(SchemaError):
        make_event(0, "INSPECT", "ERROR", **{field: variant})


def test_canonical_digest_accepted_in_every_digest_field():
    d = h("x")
    e = make_event(0, "INSPECT", "ERROR", intent_fingerprint=d, failure_fingerprint=d,
                   workspace_state_before=d, workspace_state_after=d)
    assert e.intent_fingerprint == e.failure_fingerprint == e.workspace_state_before == e.workspace_state_after == d


# -- timestamp (spec §3.1: ISO-8601) ------------------------------------------------


@pytest.mark.parametrize(
    "ts",
    [
        "2026-01-01T00:00:00Z",
        "2026-11-04T15:44:07.123456Z",
        "2026-11-04T15:44:07+00:00",
        "2026-11-04T15:44:07.5-07:30",
        "2026-11-04T15:44:07",
        "2026-11-04 15:44:07",
        "2026-11-04T15:44",
        "2026-11-04",
        "20261104T154407Z",
    ],
)
def test_timestamp_accepts_iso8601(ts):
    e = make_event(0, timestamp=ts)
    assert e.timestamp == ts  # validated, never normalised
    assert CanonicalEvent.from_json(e.to_json()).timestamp == ts


@pytest.mark.parametrize(
    "ts",
    [
        "t",
        "Z",
        "yesterday",
        "2026-13-01T00:00:00Z",
        "2026-02-30T00:00:00Z",
        "2026-01-01T24:00:00Z",
        "2026-01-01T00:60:00Z",
        "2026-01-01T00:00:00ZZ",
        "2026-01-01T00:00:00z",
        "2026-01-01T00:00:00+25:00",
        " 2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z ",
        "1735689600",
    ],
)
def test_timestamp_rejects_non_iso8601(ts):
    with pytest.raises(SchemaError, match="ISO-8601"):
        make_event(0, timestamp=ts)


def test_timestamp_validated_on_from_dict():
    d = make_event(0).to_dict()
    d["timestamp"] = "not a time"
    with pytest.raises(SchemaError, match="ISO-8601"):
        CanonicalEvent.from_dict(d)


def test_timestamp_type_checked_before_format():
    with pytest.raises(SchemaError, match="non-empty str"):
        make_event(0, timestamp=20260101)
    with pytest.raises(SchemaError, match="non-empty str"):
        make_event(0, timestamp="")


def test_factory_timestamps_are_valid():
    for i in (0, 59, 60, 3599):
        assert make_event(i).timestamp.endswith("Z")
