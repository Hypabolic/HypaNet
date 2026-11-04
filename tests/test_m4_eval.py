"""M4 multi-column scoring and counterfactual consistency on hand-built window rows (no dataset read)."""

from __future__ import annotations

import random

import numpy as np

from agent_loop_lab.eval import m4
from agent_loop_lab.eval import metrics
from agent_loop_lab.eval.collisions import WindowRow
from agent_loop_lab.schema import CLASS_NAMES, OracleLabel

SCORED = [OracleLabel(n) for n in CLASS_NAMES]
VEC = np.zeros(120, np.float32).tobytes()


def _rows(n_sessions=25, seed=3, split="val"):
    rng = random.Random(seed)
    rows = []
    for s in range(n_sessions):
        for e in range(7, 7 + rng.randint(1, 6)):
            u = rng.random()
            if u < 0.08:
                lab, gate = OracleLabel.INSUFFICIENT_INSTRUMENTATION, rng.random() < 0.2
            elif u < 0.18:
                lab, gate = OracleLabel.INSUFFICIENT_AMBIGUOUS, True
            else:
                lab, gate = rng.choice(SCORED), rng.random() > 0.05
            rows.append(WindowRow(f"s{s:03d}", f"fam_{s % 3}", split, e, lab, gate, VEC if gate else None))
    return rows


def _preds(rows, seed):
    rng = random.Random(seed)
    return np.array([rng.randrange(4) if r.gate_passed else -1 for r in rows])


def test_score_columns_matches_m3_score_report():
    rows = _rows()
    net, ref = _preds(rows, 1), _preds(rows, 2)
    m3 = metrics.score_report(rows, net, ref, n_resamples=300)
    rep = m4.score_columns(rows, {"net": net, "ref": ref, "other": _preds(rows, 3)}, {"P": ("net", "ref")},
                           {"G": ["net", "ref", "other"]}, n_resamples=300)
    assert rep["overall"]["columns"]["net"] == m3["overall"]["network"]
    assert rep["overall"]["columns"]["ref"] == m3["overall"]["reference_policy"]
    assert rep["overall"]["h2_fidelity"]["P"] == m3["overall"]["h2_fidelity"]
    for fam in m3["per_family"]:
        assert rep["per_family"][fam]["columns"]["net"] == m3["per_family"][fam]["network"]
    assert rep["population"] == m3["population"]
    g = rep["overall"]["groups"]["G"]["macro_f1"]
    vals = [rep["overall"]["columns"][c]["macro_f1"]["value"] for c in ("net", "ref", "other")]
    assert np.isclose(g["mean"], np.mean(vals)) and np.isclose(g["sd"], np.std(vals, ddof=1))
    lo, hi = g["seed_mean_ci95"]
    assert lo <= hi


def test_gate_report():
    rows = _rows()
    g = m4.gate_report(rows)
    assert g["windows"] == len(rows)
    assert g["gate_passed"] == sum(r.gate_passed for r in rows)
    assert (g["rejected_and_instrumentation"] + g["rejected_not_instrumentation"] + g["passed_and_instrumentation"]
            + g["passed_not_instrumentation"]) == len(rows)


def _cf_fixture():
    """Two pairs: retry kind (contrast 9, 10) and abab (contrast 8), one gate-rejected endpoint."""
    P, R, S, O = (OracleLabel.PRODUCTIVE, OracleLabel.RECOVERY, OracleLabel.STALLED, OracleLabel.OSCILLATING)
    rows = [
        WindowRow("x-a", "retry_success_vs_doomed", "counterfactual", 9, R, True, VEC),
        WindowRow("x-a", "retry_success_vs_doomed", "counterfactual", 10, P, True, VEC),
        WindowRow("x-b", "retry_success_vs_doomed", "counterfactual", 9, S, True, VEC),
        WindowRow("x-b", "retry_success_vs_doomed", "counterfactual", 10, S, True, VEC),
        WindowRow("y-a", "productive_vs_abab", "counterfactual", 8, P, True, VEC),
        WindowRow("y-b", "productive_vs_abab", "counterfactual", 8, O, True, VEC),
        WindowRow("y-a", "productive_vs_abab", "counterfactual", 11, P, False, None),
        WindowRow("y-b", "productive_vs_abab", "counterfactual", 11, O, True, VEC),
        WindowRow("z-a", "productive_vs_abab", "counterfactual", 8, P, True, VEC),
        WindowRow("z-b", "productive_vs_abab", "counterfactual", 8, P, True, VEC),
    ]
    meta = {
        "x-a": {"pair_id": "X", "kind": "retry_success_vs_doomed", "branch": "a", "contrast_endpoints": [9, 10]},
        "x-b": {"pair_id": "X", "kind": "retry_success_vs_doomed", "branch": "b", "contrast_endpoints": [9, 10]},
        "y-a": {"pair_id": "Y", "kind": "productive_vs_abab", "branch": "a", "contrast_endpoints": [8, 11]},
        "y-b": {"pair_id": "Y", "kind": "productive_vs_abab", "branch": "b", "contrast_endpoints": [8, 11]},
        "z-a": {"pair_id": "Z", "kind": "productive_vs_abab", "branch": "a", "contrast_endpoints": []},
        "z-b": {"pair_id": "Z", "kind": "productive_vs_abab", "branch": "b", "contrast_endpoints": []},
    }
    for v in meta.values():
        v["labels"] = {}
    return rows, meta


def test_counterfactual_consistency():
    rows, meta = _cf_fixture()
    # x: endpoint 9 -> (RECOVERY, STALLED) strict+directional; 10 -> (PRODUCTIVE, OSCILLATING) directional only.
    # y: endpoint 8 -> (STALLED, OSCILLATING): neither; endpoint 11 gate-excluded.
    pred = np.array([1, 0, 2, 3, 2, 3, -1, 3, 0, 0])
    rep = m4.counterfactual_consistency(rows, meta, {"m": pred}, n_resamples=200)
    assert rep["pairs_total"] == 3
    assert rep["pairs_without_contrast_endpoints"] == 1
    assert rep["contrast_endpoints_gate_excluded"] == 1
    assert rep["pairs_scored"] == 2
    o = rep["columns"]["m"]["overall"]
    assert o["endpoints"] == 3
    assert o["directional"]["endpoint_rate"] == 2 / 3
    assert o["directional"]["pairs_all"] == 0.5 and o["directional"]["pairs_any"] == 0.5
    assert o["strict"]["endpoint_rate"] == 1 / 3
    assert o["strict"]["pairs_all"] == 0.0 and o["strict"]["pairs_any"] == 0.5
    assert rep["columns"]["m"]["retry_success_vs_doomed"]["directional"]["pairs_all"] == 1.0
    assert rep["columns"]["m"]["productive_vs_abab"]["directional"]["endpoint_rate"] == 0.0


def test_counterfactual_rejects_non_contrast_labels():
    import pytest

    rows, meta = _cf_fixture()
    rows[2] = WindowRow("x-b", "retry_success_vs_doomed", "counterfactual", 9, OracleLabel.PRODUCTIVE, True, VEC)
    with pytest.raises(ValueError):
        m4.counterfactual_consistency(rows, meta, {"m": np.zeros(len(rows), int)}, n_resamples=10)


def test_load_split_refuses_unknown():
    import pytest

    with pytest.raises(ValueError):
        m4.load_split("/nonexistent", "sealed")


def _load_sealed_script():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "m4_sealed_eval_v02.py"
    spec = importlib.util.spec_from_file_location("_test_m4_sealed", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


def test_sealed_script_refuses_second_run(tmp_path, monkeypatch):
    mod = _load_sealed_script()
    for existing in ("report", "marker"):
        report, marker = tmp_path / f"{existing}-r.json", tmp_path / f"{existing}-m.json"
        (report if existing == "report" else marker).write_text("{}")
        monkeypatch.setattr(mod, "SEALED_REPORT", report)
        monkeypatch.setattr(mod, "SEALED_MARKER", marker)
        monkeypatch.setattr(mod, "load_columns", lambda: (_ for _ in ()).throw(AssertionError("must not load")))
        assert mod.main([]) == 2
