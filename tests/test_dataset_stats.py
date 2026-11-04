"""Dataset statistics (M1_CONTRACTS §6, §7 "Audits")."""

from __future__ import annotations

import json
import random

import pytest

from agent_loop_lab.eval import audit_dataset, to_json
from agent_loop_lab.eval.dataset_stats import UNDER_INSTRUMENTED, DatasetStatsAccumulator, dataset_stats, length_summary
from agent_loop_lab.schema import CLASS_NAMES, LatentEffect, OracleLabel, OracleRecord
from tests.factories import session

P, R, S, O = "PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"
AMB = "INSUFFICIENT_AMBIGUOUS"
INSTR = "INSUFFICIENT_INSTRUMENTATION"


def make_session(sid: str, family: str, labels: list[str], *, drop_at: int | None = None):
    """A session with ``len(labels) + 7`` events; ``drop_at`` drops that event's tokens."""
    n = len(labels) + 7
    specs = [{"intent": f"{sid}-read-{i}", "tokens": (100 + i, 50)} for i in range(n)]
    if drop_at is not None:
        specs[drop_at]["tokens"] = (None, 50)
    events = session(specs, session_id=sid)
    record = OracleRecord(
        session_id=sid,
        generator_version="test",
        seed=0,
        scenario_family=family,
        event_effects=(LatentEffect.NONE,) * n,
        window_labels={7 + k: OracleLabel(lab) for k, lab in enumerate(labels)},
        dropped_observations=(drop_at,) if drop_at is not None else (),
    )
    return events, record


def constructed_splits(*, family12_disagrees: bool = False):
    # Session 'u1': 10 events, token drop at event 8 -> windows ending at 8, 9 are gate-rejected.
    u1_labels = [P, INSTR, INSTR if not family12_disagrees else P]
    dev = [
        make_session("a1", "alpha", [P, P, R]),  # 10 events
        make_session("a2", "alpha", [S, S]),  # 9 events
        make_session("b1", "beta", [O, AMB, O, O]),  # 11 events
        make_session("u1", UNDER_INSTRUMENTED, u1_labels, drop_at=8),  # 10 events
    ]
    val = [make_session("a3", "alpha", [R])]  # 8 events
    test = [make_session("b2", "beta", [P, P, P, P, S])]  # 12 events
    return {"dev": dev, "val": val, "test": test}


def test_length_summary():
    assert length_summary({}) == {"sessions": 0, "min": None, "max": None, "mean": None, "median": None, "histogram": {}}
    odd = length_summary({12: 1, 20: 1, 30: 1})
    assert (odd["min"], odd["max"], odd["mean"], odd["median"]) == (12, 30, pytest.approx(62 / 3), 20)
    even = length_summary({10: 2, 15: 1, 40: 1})  # 10, 10, 15, 40
    assert even["median"] == 12.5 and even["mean"] == pytest.approx(75 / 4)
    assert length_summary({10: 3, 40: 1})["median"] == 10
    assert list(even["histogram"]) == ["10", "15", "40"]


def test_dataset_stats_constructed():
    rep = dataset_stats(constructed_splits())
    assert rep["splits"] == ["dev", "val", "test"]
    assert rep["families"] == ["alpha", "beta", UNDER_INSTRUMENTED]

    sess = rep["sessions"]
    assert sess["total"] == 6
    assert sess["per_split"] == {"dev": 4, "val": 1, "test": 1}
    assert sess["per_family"] == {"alpha": 3, "beta": 2, UNDER_INSTRUMENTED: 1}
    assert sess["per_split_family"]["dev"] == {"alpha": 2, "beta": 1, UNDER_INSTRUMENTED: 1}
    assert sess["per_split_family"]["val"] == {"alpha": 1}

    wc = rep["window_counts"]
    assert wc["total"] == {"windows": 18, "gate_passed": 16, "scored": 15, "scored_gate_passed": 15}
    assert wc["per_split"]["dev"] == {"windows": 12, "gate_passed": 10, "scored": 9, "scored_gate_passed": 9}
    assert wc["per_family"][UNDER_INSTRUMENTED] == {"windows": 3, "gate_passed": 1, "scored": 1, "scored_gate_passed": 1}

    dev = rep["class_priors"]["per_split"]["dev"]
    assert dev["label_counts"] == {P: 3, R: 1, S: 2, O: 3, INSTR: 2, AMB: 1}
    assert dev["scored_windows"] == 9
    assert list(dev["scored_priors"]) == list(CLASS_NAMES)
    assert dev["scored_priors"] == {P: pytest.approx(3 / 9), R: pytest.approx(1 / 9), S: pytest.approx(2 / 9), O: pytest.approx(3 / 9)}
    assert rep["class_priors"]["per_split"]["val"]["scored_priors"] == {P: 0.0, R: 1.0, S: 0.0, O: 0.0}
    assert rep["class_priors"]["total"]["scored_windows"] == 15

    mix = rep["family_label_mix"]
    assert mix["alpha"]["label_counts"][P] == 2 and mix["alpha"]["label_counts"][R] == 2 and mix["alpha"]["label_counts"][S] == 2
    assert mix["beta"]["label_counts"] == {P: 4, R: 0, S: 1, O: 3, INSTR: 0, AMB: 1}
    assert mix["beta"]["scored_priors"][P] == pytest.approx(0.5)

    gate = rep["gate"]["per_family"]
    assert gate["alpha"]["gate_pass_rate"] == 1.0 and gate["beta"]["gate_pass_rate"] == 1.0
    u = gate[UNDER_INSTRUMENTED]
    assert (u["windows"], u["gate_passed"], u["gate_oracle_disagreements"]) == (3, 1, 0)
    assert u["gate_pass_rate"] == pytest.approx(1 / 3) and u["gate_oracle_agreement_rate"] == 1.0
    assert rep["under_instrumented_gate_oracle"] == u

    sl = rep["session_length"]
    assert sl["total"]["histogram"] == {"8": 1, "9": 1, "10": 2, "11": 1, "12": 1}
    assert (sl["total"]["min"], sl["total"]["max"], sl["total"]["median"]) == (8, 12, 10)
    assert sl["per_split"]["dev"]["mean"] == pytest.approx(40 / 4)
    assert sl["per_family"]["beta"]["median"] == 11.5

    assert rep["checks"] == {
        "gate_all_pass_families_1_11": True,
        "families_below_full_gate_pass": [],
        "under_instrumented_present": True,
        "under_instrumented_gate_oracle_agree": True,
        "gate_oracle_agree_all_families": True,
    }


def test_gate_oracle_disagreement_is_reported_not_raised():
    rep = dataset_stats(constructed_splits(family12_disagrees=True))
    u = rep["gate"]["per_family"][UNDER_INSTRUMENTED]
    assert u["gate_oracle_disagreements"] == 1
    assert u["disagreement_examples"] == [{"session_id": "u1", "endpoint": 9}]
    assert rep["checks"]["under_instrumented_gate_oracle_agree"] is False
    assert rep["checks"]["gate_oracle_agree_all_families"] is False
    assert rep["checks"]["gate_all_pass_families_1_11"] is True


def test_gate_failure_in_a_normal_family_flags_the_check():
    splits = constructed_splits()
    splits["val"].append(make_session("a9", "alpha", [INSTR], drop_at=3))
    rep = dataset_stats(splits)
    assert rep["checks"]["gate_all_pass_families_1_11"] is False
    assert rep["checks"]["families_below_full_gate_pass"] == ["alpha"]
    assert rep["checks"]["gate_oracle_agree_all_families"] is True  # rejected window is labelled INSTR


def test_duplicate_session_rejected():
    acc = DatasetStatsAccumulator()
    ev, rec = make_session("a1", "alpha", [P])
    acc.add_session("dev", ev, rec)
    with pytest.raises(ValueError):
        acc.add_session("val", ev, rec)


def test_deterministic_order_independent_and_json():
    splits = constructed_splits()
    rep = dataset_stats(splits)
    text = to_json(rep)
    assert json.loads(text) == json.loads(json.dumps(rep))
    shuffled = {}
    for name in ("val", "test", "dev"):
        ss = list(splits[name])
        random.Random(name).shuffle(ss)
        shuffled[name] = ss[::-1]
    assert to_json(dataset_stats(shuffled)) == text
    stats, _ = audit_dataset(splits)
    assert to_json(stats) == text


def test_real_families_smoke():
    """A tiny real build: every family appears, gate checks hold, report serialises."""
    from agent_loop_lab.oracle_config import load_oracle_config
    from agent_loop_lab.synth.families import FAMILY_NAMES, generate_session

    cfg = load_oracle_config()
    splits: dict[str, list] = {"dev": [], "val": []}
    for i, fam in enumerate(FAMILY_NAMES):
        for k in range(2):
            seed = 1000 * i + k
            splits["dev" if k == 0 else "val"].append(generate_session(fam, seed, f"{fam}-{k}", cfg))
    stats, coll = audit_dataset(splits)
    assert stats["families"] == sorted(FAMILY_NAMES)
    assert stats["sessions"]["total"] == 2 * len(FAMILY_NAMES)
    assert stats["checks"]["gate_all_pass_families_1_11"], stats["checks"]
    assert stats["checks"]["gate_oracle_agree_all_families"], stats["checks"]
    assert coll["collisions"]["population"]["scored_gate_passed"] == stats["window_counts"]["per_split"]["dev"]["scored_gate_passed"]
    json.loads(to_json(stats))
    json.loads(to_json(coll))
