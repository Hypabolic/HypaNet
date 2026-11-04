"""Collision audit and Bayes ceiling (M1_CONTRACTS §6, §7 "Audits")."""

from __future__ import annotations

import json
import random
from fractions import Fraction

import pytest

from agent_loop_lab.eval import audit_dataset, collision_report, to_json
from agent_loop_lab.eval.collisions import (
    WindowRow,
    bayes_ceiling,
    collision_audit,
    cross_split_overlap,
    majority_label,
    session_windows,
    split_order,
)
from agent_loop_lab.schema import ClassLabel, LatentEffect, OracleLabel, OracleRecord
from tests.factories import session

P, R, S, O = "PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"
AMB = "INSUFFICIENT_AMBIGUOUS"
INSTR = "INSUFFICIENT_INSTRUMENTATION"

# Last-event token counts select the window's input vector: same variant -> byte-identical vector.
VARIANT_TOKENS = {"A": (100, 50), "B": (200, 50), "C": (300, 50), "D": (400, 50)}


def make_session(sid: str, variant: str, label: str, family: str = "fam_x", *, drop_tokens: bool = False):
    """One 8-event session (a single window) whose vector is fixed by ``variant``."""
    specs = [{"intent": f"read-{i}"} for i in range(8)]
    specs[-1]["tokens"] = (None, None) if drop_tokens else VARIANT_TOKENS[variant]
    events = session(specs, session_id=sid)
    record = OracleRecord(
        session_id=sid,
        generator_version="test",
        seed=0,
        scenario_family=family,
        event_effects=(LatentEffect.NONE,) * len(events),
        window_labels={7: OracleLabel(label)},
    )
    return events, record


def constructed_splits():
    dev = [
        make_session("s1", "A", S, "fam_x"),
        make_session("s2", "A", S, "fam_y"),
        make_session("s3", "A", P, "fam_y"),
        make_session("s4", "B", P, "fam_x"),
        make_session("s5", "C", R, "fam_x"),
        make_session("s6", "C", O, "fam_y"),
        make_session("s7", "A", AMB, "fam_x"),  # gate passes, unscored: excluded from groups
        make_session("s8", "A", INSTR, "fam_x", drop_tokens=True),  # gate rejects: excluded
    ]
    val = [make_session("v1", "A", P), make_session("v2", "D", R)]
    test = [make_session("t1", "A", S), make_session("t2", "B", AMB)]
    return {"dev": dev, "val": val, "test": test}


# -- Bayes ceiling arithmetic --------------------------------------------------


def test_majority_tie_breaks_by_canonical_order():
    assert majority_label({ClassLabel.STALLED: 2, ClassLabel.OSCILLATING: 2}) is ClassLabel.STALLED
    assert majority_label({ClassLabel.OSCILLATING: 1, ClassLabel.PRODUCTIVE: 1, ClassLabel.RECOVERY: 1}) is ClassLabel.PRODUCTIVE
    assert majority_label({ClassLabel.OSCILLATING: 3, ClassLabel.PRODUCTIVE: 1}) is ClassLabel.OSCILLATING
    with pytest.raises(ValueError):
        majority_label({})


def test_bayes_ceiling_hand_computed():
    # G1 = [P,P,P,R] -> P; G2 = [S,S,O,O] -> tie -> S; G3 = [O] -> O.
    # correct = 3 + 2 + 1 = 6 of 9.
    # P: tp 3, predicted 4 (all of G1), support 3 -> precision 3/4, recall 1 -> F1 6/7
    # R: tp 0, predicted 0, support 1 -> precision 0, recall 0, F1 0
    # S: tp 2, predicted 4 (all of G2), support 2 -> precision 1/2, recall 1 -> F1 2/3
    # O: tp 1, predicted 1 (G3), support 3 -> precision 1, recall 1/3 -> F1 1/2
    # macro-F1 = (6/7 + 0 + 2/3 + 1/2) / 4 = 85/168
    out = bayes_ceiling([[P, P, P, R], [S, S, O, O], [O]])
    assert out["windows"] == 9 and out["groups"] == 3 and out["correct"] == 6
    assert out["accuracy"] == pytest.approx(float(Fraction(2, 3)))
    assert out["macro_f1"] == pytest.approx(float(Fraction(85, 168)))
    assert out["macro_f1_classes"] == [P, R, S, O]
    pc = out["per_class"]
    assert (pc[P]["predicted"], pc[P]["correct"]) == (4, 3)
    assert (pc[P]["precision"], pc[P]["recall"]) == (0.75, 1.0) and pc[P]["f1"] == pytest.approx(6 / 7)
    assert (pc[R]["predicted"], pc[R]["precision"], pc[R]["recall"], pc[R]["f1"]) == (0, 0.0, 0.0, 0.0)
    assert pc[S]["predicted"] == 4 and pc[S]["f1"] == pytest.approx(2 / 3)
    assert pc[O]["recall"] == pytest.approx(1 / 3) and pc[O]["f1"] == pytest.approx(0.5)


def test_bayes_ceiling_accepts_counts_and_skips_unsupported_classes():
    by_lists = bayes_ceiling([[P, P, S], [S]])
    by_counts = bayes_ceiling([{ClassLabel.PRODUCTIVE: 2, ClassLabel.STALLED: 1}, {"STALLED": 1, "RECOVERY": 0}])
    assert by_lists == by_counts
    # Only P and S have support: macro over those two.
    # P: tp 2, pred 3, support 2 -> precision 2/3, recall 1 -> F1 4/5
    # S: tp 1, pred 1, support 2 -> precision 1, recall 1/2 -> F1 2/3
    assert by_lists["macro_f1_classes"] == [P, S]
    assert by_lists["macro_f1"] == pytest.approx((4 / 5 + 2 / 3) / 2)
    assert by_lists["per_class"][R]["f1"] is None and by_lists["per_class"][R]["precision"] is None


def test_bayes_ceiling_perfect_without_collisions():
    out = bayes_ceiling([[P], [R, R], [S], [O, O, O]])
    assert out["accuracy"] == 1.0 and out["macro_f1"] == 1.0


def test_bayes_ceiling_rejects_unscored_and_empty():
    with pytest.raises(ValueError):
        bayes_ceiling([[AMB]])
    with pytest.raises(ValueError):
        bayes_ceiling([[]])
    empty = bayes_ceiling([])
    assert empty["accuracy"] is None and empty["macro_f1"] is None


# -- window rows -----------------------------------------------------------------


def test_session_windows_rows():
    ev, rec = make_session("s1", "A", S)
    (row,) = session_windows(ev, rec, "dev")
    assert isinstance(row, WindowRow)
    assert (row.session_id, row.family, row.split, row.endpoint, row.label) == ("s1", "fam_x", "dev", 7, OracleLabel.STALLED)
    assert row.gate_passed and row.in_population and len(row.vector) == 120 * 4
    ev2, rec2 = make_session("s2", "A", P)
    assert session_windows(ev2, rec2)[0].vector == row.vector  # identical streams -> identical bytes
    ev3, rec3 = make_session("s3", "A", INSTR, drop_tokens=True)
    (r3,) = session_windows(ev3, rec3)
    assert not r3.gate_passed and r3.vector is None and not r3.in_population


def test_session_windows_rejects_mismatched_record_or_missing_label():
    ev, _ = make_session("s1", "A", S)
    _, other = make_session("s2", "A", S)
    with pytest.raises(ValueError):
        session_windows(ev, other)
    _, rec = make_session("s1", "A", S)
    bare = OracleRecord(session_id="s1", generator_version="t", seed=0, scenario_family="f", event_effects=rec.event_effects)
    with pytest.raises(ValueError):
        session_windows(ev, bare)


def test_split_order():
    assert split_order({"test", "zeta", "dev", "alpha", "val"}) == ["dev", "val", "test", "alpha", "zeta"]


# -- constructed collision case ---------------------------------------------------


def _rows(sessions, split=None):
    return [r for ev, rec in sessions for r in session_windows(ev, rec, split)]


def test_collision_audit_constructed_case():
    rep = collision_audit(_rows(constructed_splits()["dev"], "dev"))
    pop = rep["population"]
    assert pop["windows_total"] == 8 and pop["gate_rejected"] == 1 and pop["scored_gate_passed"] == 6
    assert pop["gate_passed_unscored"] == {INSTR: 0, AMB: 1}
    assert rep["groups"] == 3 and rep["singleton_groups"] == 1
    assert rep["group_size_histogram"] == {"1": 1, "2": 1, "3": 1}
    assert rep["collision_groups_count"] == 2 and rep["windows_in_collision_groups"] == 5
    assert rep["collision_rate"] == pytest.approx(5 / 6)

    g_a, g_c = rep["collision_groups"]  # sorted by size desc
    assert g_a["windows"] == 3 and g_a["label_counts"] == {P: 1, S: 2} and g_a["majority"] == S
    assert g_a["family_counts"] == {"fam_x": 1, "fam_y": 2}
    assert [e["session_id"] for e in g_a["examples"]] == ["s1", "s2", "s3"]
    assert g_c["windows"] == 2 and g_c["label_counts"] == {R: 1, O: 1} and g_c["majority"] == R  # tie -> canonical order
    assert len(g_a["vector_sha256"]) == 64

    pc = rep["per_class"]
    assert {k: (v["windows"], v["in_collision_groups"]) for k, v in pc.items()} == {P: (2, 1), R: (1, 1), S: (2, 2), O: (1, 1)}
    assert pc[P]["collision_rate"] == 0.5 and pc[S]["collision_rate"] == 1.0
    pf = rep["per_family"]
    assert list(pf) == ["fam_x", "fam_y"]
    assert (pf["fam_x"]["windows"], pf["fam_x"]["in_collision_groups"]) == (3, 2)
    assert pf["fam_x"]["collision_rate"] == pytest.approx(2 / 3)
    assert (pf["fam_y"]["windows"], pf["fam_y"]["in_collision_groups"], pf["fam_y"]["collision_rate"]) == (3, 3, 1.0)

    # Groups A=[S,S,P] -> S, B=[P] -> P, C=[R,O] -> R. Correct 2 + 1 + 1 = 4 of 6.
    # P: tp 1, pred 1, sup 2 -> F1 2/3; R: tp 1, pred 2, sup 1 -> F1 2/3;
    # S: tp 2, pred 3, sup 2 -> F1 4/5; O: pred 0, sup 1 -> F1 0. Macro = 8/15.
    bc = rep["bayes_ceiling"]
    assert bc["accuracy"] == pytest.approx(4 / 6)
    assert bc["macro_f1"] == pytest.approx(8 / 15)
    assert bc == bayes_ceiling([[S, S, P], [P], [R, O]])


def test_collision_audit_examples_capped_and_minimal():
    sessions = [make_session(f"s{k:02d}", "A", S if k % 2 else P) for k in range(30)]
    random.Random(3).shuffle(sessions)
    rep = collision_audit(_rows(sessions), max_examples=3)
    (g,) = rep["collision_groups"]
    assert [e["session_id"] for e in g["examples"]] == ["s00", "s01", "s02"]
    assert collision_audit(_rows(sessions), max_examples=0)["collision_groups"][0]["examples"] == []


def test_cross_split_overlap_constructed_case():
    splits = constructed_splits()
    rows = [r for name, ss in splits.items() for r in _rows(ss, name)]
    rep = cross_split_overlap(rows)
    assert rep["splits"] == ["dev", "val", "test"]

    gp = rep["gate_passed"]
    assert gp["vectors_in_multiple_splits"] == 2  # A (all three splits) and B (dev, test via ambiguous t2)
    assert gp["vectors_in_multiple_splits_with_conflicting_scored_labels"] == 1  # A
    assert gp["per_split"]["dev"] == {
        "windows": 7,
        "distinct_vectors": 3,
        "windows_with_vector_in_another_split": 5,
        "distinct_vectors_in_another_split": 2,
        "shared_window_rate": pytest.approx(5 / 7),
    }
    assert gp["per_split"]["val"]["windows_with_vector_in_another_split"] == 1
    assert gp["per_split"]["test"]["distinct_vectors_in_another_split"] == 2
    assert gp["pairs"]["dev|val"] == {"shared_distinct_vectors": 1, "windows_in_dev": 4, "windows_in_val": 1}
    assert gp["pairs"]["dev|test"] == {"shared_distinct_vectors": 2, "windows_in_dev": 5, "windows_in_test": 2}
    assert gp["pairs"]["val|test"] == {"shared_distinct_vectors": 1, "windows_in_val": 1, "windows_in_test": 1}

    sc = rep["scored"]
    assert sc["vectors_in_multiple_splits"] == 1 and sc["vectors_in_multiple_splits_with_conflicting_scored_labels"] == 1
    assert sc["per_split"]["dev"]["windows"] == 6 and sc["per_split"]["dev"]["windows_with_vector_in_another_split"] == 3
    assert sc["per_split"]["test"]["windows"] == 1
    assert sc["pairs"]["dev|test"]["shared_distinct_vectors"] == 1


def test_cross_split_requires_split_and_lists_empty_splits():
    ev, rec = make_session("s1", "A", S)
    with pytest.raises(ValueError):
        cross_split_overlap(session_windows(ev, rec))
    rep = cross_split_overlap(session_windows(ev, rec, "dev"), splits=["dev", "val", "test"])
    assert rep["splits"] == ["dev", "val", "test"]
    assert rep["scored"]["per_split"]["val"]["shared_window_rate"] is None


def test_collision_report_deterministic_order_independent_and_json():
    splits = constructed_splits()
    rep = collision_report(splits)
    assert rep["audit_split"] == "dev"
    assert rep["collisions"] == collision_audit(_rows(splits["dev"], "dev"))
    text = to_json(rep)
    assert json.loads(text) == json.loads(json.dumps(rep))
    shuffled = {}
    for name in ("test", "dev", "val"):
        ss = list(splits[name])
        random.Random(name).shuffle(ss)
        shuffled[name] = ss[::-1]
    assert to_json(collision_report(shuffled)) == text
    assert to_json(audit_dataset(splits)[1]) == text
    with pytest.raises(KeyError):
        collision_report({"val": splits["val"]})
