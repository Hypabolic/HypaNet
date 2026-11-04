"""M3 validation scoring (src/agent_loop_lab/eval/metrics.py, scripts/score_validation.py) and the
Phase B disclosure sampler (scripts/m3_build_disclosure.py).

Synthetic window rows only: no dataset split is read and no archived artefact is loaded.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from agent_loop_lab.eval import metrics
from agent_loop_lab.eval.collisions import WindowRow
from agent_loop_lab.model import zero_model
from agent_loop_lab.schema import CLASS_NAMES, FEATURE_NAMES, OracleLabel

REPO = Path(__file__).resolve().parents[1]
SCORED = [OracleLabel(n) for n in CLASS_NAMES]


def _load_script(name: str):
    path = REPO / "scripts" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


@pytest.fixture(scope="module")
def sv():
    return _load_script("score_validation")


@pytest.fixture(scope="module")
def disclosure():
    return _load_script("m3_build_disclosure")


def _vec(rng: random.Random) -> bytes:
    return np.array([rng.choice((0.0, 1.0, 0.25)) for _ in range(120)], dtype=np.float32).tobytes()


def synthetic_rows(n_sessions: int = 30, seed: int = 3, split: str | None = "val") -> list[WindowRow]:
    rng = random.Random(seed)
    rows = []
    for s in range(n_sessions):
        fam = f"fam_{s % 3}"
        for e in range(7, 7 + rng.randint(1, 6)):
            u = rng.random()
            if u < 0.08:
                lab, gate = OracleLabel.INSUFFICIENT_INSTRUMENTATION, rng.random() < 0.2
            elif u < 0.18:
                lab, gate = OracleLabel.INSUFFICIENT_AMBIGUOUS, True
            else:
                lab, gate = rng.choice(SCORED), rng.random() > 0.05
            rows.append(WindowRow(f"s{s:03d}", fam, split, e, lab, gate, _vec(rng) if gate else None))
    return rows


def synthetic_predictions(rows, seed: int = 5):
    rng = random.Random(seed)
    net = np.array([rng.randrange(4) if r.gate_passed else -1 for r in rows])
    ref = np.array([(p if rng.random() < 0.8 else rng.randrange(4)) if p >= 0 else -1 for p in net])
    return net, ref


# ---------------------------------------------------------------------------
# Split guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["test", "dev", "ood", "counterfactual", "VAL", "", None])
def test_check_split_refuses_everything_but_val(split):
    with pytest.raises(ValueError):
        metrics.check_split(split)
    assert metrics.check_split("val") == "val"
    assert metrics.ALLOWED_SPLITS == frozenset({"val"})


@pytest.mark.parametrize("split", ["test", "dev"])
def test_load_rows_refuses_before_reading(monkeypatch, tmp_path, split):
    import agent_loop_lab.datasets.build as build

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("iter_sessions called for a refused split")

    monkeypatch.setattr(build, "iter_sessions", boom)
    with pytest.raises(ValueError, match="may not be scored"):
        metrics.load_rows(tmp_path / "does-not-exist", split)


def test_score_report_refuses_rows_of_other_splits():
    rows = synthetic_rows(4, split="test")
    net, ref = synthetic_predictions(rows)
    with pytest.raises(ValueError, match="may not be scored"):
        metrics.score_report(rows, net, ref, n_resamples=5)


def test_score_validation_has_no_split_override(sv):
    assert sv.SPLIT == "val"
    with pytest.raises(SystemExit):
        sv.main(["--split", "test", "phase-a-selfcheck"])
    with pytest.raises(ValueError, match="not a phase name"):
        sv.phase_dir("../datasets")


def test_disclosure_selection_refuses_other_splits(disclosure):
    with pytest.raises(ValueError, match="may not be scored"):
        disclosure.select(synthetic_rows(4, split="test"), per_class=1)


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------


def _counts_from_cm(cm) -> np.ndarray:
    c = np.zeros(metrics.MODEL_COLUMNS)
    c[:16] = np.asarray(cm, dtype=float).reshape(16)
    return c


def test_metrics_on_a_hand_confusion_matrix():
    # rows = oracle, cols = predicted (P, R, S, O)
    cm = [
        [8, 1, 1, 0],
        [2, 6, 0, 2],
        [0, 0, 5, 0],
        [0, 0, 0, 0],  # OSCILLATING: no support, predicted twice
    ]
    m = {k: float(v) for k, v in metrics.metrics_from_counts(_counts_from_cm(cm)).items()}
    assert m["precision.PRODUCTIVE"] == pytest.approx(8 / 10)
    assert m["recall.PRODUCTIVE"] == pytest.approx(8 / 10)
    assert m["precision.STALLED"] == pytest.approx(5 / 6)
    assert m["precision.OSCILLATING"] == 0.0  # predicted, never correct
    assert np.isnan(m["recall.OSCILLATING"]) and np.isnan(m["f1.OSCILLATING"])
    f1 = {
        "P": 2 * 0.8 * 0.8 / 1.6,
        "R": 2 * (6 / 7) * 0.6 / ((6 / 7) + 0.6),
        "S": 2 * (5 / 6) * 1.0 / ((5 / 6) + 1.0),
    }
    assert m["macro_f1"] == pytest.approx(sum(f1.values()) / 3)  # classes with support only
    assert m["accuracy"] == pytest.approx(19 / 25)
    # false alarm: oracle P/R predicted S/O = 1 (P->S) + 2 (R->O) over 20
    assert m["false_alarm_rate"] == pytest.approx(3 / 20)
    assert m["confusion.RECOVERY.OSCILLATING"] == 2


def test_class_with_support_never_predicted_has_zero_precision_and_f1():
    cm = [[5, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    m = {k: float(v) for k, v in metrics.metrics_from_counts(_counts_from_cm(cm)).items()}
    assert m["precision.RECOVERY"] == 0.0 and m["f1.RECOVERY"] == 0.0
    assert np.isnan(m["precision.STALLED"])
    assert m["macro_f1"] == pytest.approx((2 * (5 / 8) * 1 / (5 / 8 + 1) + 0.0) / 2)
    assert m["false_alarm_rate"] == 0.0


# ---------------------------------------------------------------------------
# Bootstrap: count path == naive per-resample recomputation from rows
# ---------------------------------------------------------------------------


def _naive(rows, net, ref):
    """Plain-Python metrics from a list of rows (with multiplicity)."""
    pairs = [(int(r.label.to_class()), int(p)) for r, p in zip(rows, net) if r.gate_passed and r.label.is_scored]
    f1s = []
    for k in range(4):
        sup = sum(t == k for t, _ in pairs)
        if not sup:
            continue
        pred = sum(p == k for _, p in pairs)
        tp = sum(t == k and p == k for t, p in pairs)
        prec = tp / pred if pred else 0.0
        rec = tp / sup
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    safe = [(t, p) for t, p in pairs if t in (0, 1)]
    fa = sum(p in (2, 3) for _, p in safe) / len(safe) if safe else float("nan")
    gp = [(a, b) for r, a, b in zip(rows, net, ref) if r.gate_passed]
    fid = sum(a == b for a, b in gp) / len(gp) if gp else float("nan")
    amb = [p for r, p in zip(rows, net) if r.gate_passed and r.label is OracleLabel.INSUFFICIENT_AMBIGUOUS]
    amb_s = sum(p == 2 for p in amb) / len(amb) if amb else float("nan")
    return {
        "network:macro_f1": sum(f1s) / len(f1s) if f1s else float("nan"),
        "network:false_alarm_rate": fa,
        "h2_fidelity:all.agreement": fid,
        "network:ambiguous.fraction.STALLED": amb_s,
    }


def test_bootstrap_matches_naive_loop():
    rows = synthetic_rows(25)
    net, ref = synthetic_predictions(rows)
    ids, _fams, sidx = metrics.session_index(rows)
    counts = metrics.session_counts(rows, net, ref, len(ids), sidx)
    B, seed = 120, 11
    dist = metrics.bootstrap_distribution(counts, B, seed)
    by_session = {s: [i for i in range(len(rows)) if sidx[i] == s] for s in range(len(ids))}
    naive = {k: [] for k in _naive(rows, net, ref)}
    for chunk in metrics.bootstrap_indices(len(ids), B, seed):
        for draw in chunk:
            pick = [i for s in draw for i in by_session[int(s)]]
            vals = _naive([rows[i] for i in pick], net[pick], ref[pick])
            for k, v in vals.items():
                naive[k].append(v)
    for k, v in naive.items():
        np.testing.assert_allclose(dist[k], np.array(v), rtol=1e-12, atol=0, equal_nan=True, err_msg=k)


def test_bootstrap_indices_are_seeded_and_complete():
    a = np.concatenate(list(metrics.bootstrap_indices(7, 600, 1)))
    b = np.concatenate(list(metrics.bootstrap_indices(7, 600, 1)))
    c = np.concatenate(list(metrics.bootstrap_indices(7, 600, 2)))
    assert a.shape == (600, 7) and (a == b).all() and not (a == c).all()
    assert a.min() >= 0 and a.max() < 7


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_score_report_populations_and_determinism():
    rows = synthetic_rows(30)
    net, ref = synthetic_predictions(rows)
    rep = metrics.score_report(rows, net, ref, n_resamples=200)
    again = metrics.score_report(rows, net, ref, n_resamples=200)
    assert json.dumps(rep, sort_keys=True) == json.dumps(again, sort_keys=True)

    pop = rep["population"]
    gate_passed = [r for r in rows if r.gate_passed]
    scored = [r for r in gate_passed if r.label.is_scored]
    assert pop["windows_total"] == len(rows)
    assert pop["gate_passed"] == len(gate_passed)
    assert pop["scored_windows"] == len(scored) == rep["overall"]["network"]["scored_windows"]
    cm = np.array(rep["overall"]["network"]["confusion_matrix"]["counts"])
    assert cm.sum() == len(scored)
    for r_i, t in enumerate(CLASS_NAMES):  # rows = oracle label
        assert cm[r_i].sum() == sum(r.label.value == t for r in scored)
    amb = rep["overall"]["network"]["ambiguous_predictions"]
    assert amb["windows"] == sum(r.label is OracleLabel.INSUFFICIENT_AMBIGUOUS for r in gate_passed)
    fid = rep["overall"]["h2_fidelity"]["all"]
    assert fid["windows"] == len(gate_passed)
    assert fid["agree"] == sum(int(a == b) for r, a, b in zip(rows, net, ref) if r.gate_passed)
    # point values equal the naive computation
    naive = _naive(rows, net, ref)
    assert rep["overall"]["network"]["macro_f1"]["value"] == pytest.approx(naive["network:macro_f1"], rel=1e-12)
    assert rep["overall"]["network"]["false_alarm_rate"]["value"] == pytest.approx(naive["network:false_alarm_rate"], rel=1e-12)
    lo, hi = rep["overall"]["network"]["macro_f1"]["ci95"]
    assert lo <= rep["overall"]["network"]["macro_f1"]["value"] <= hi
    # per-family blocks cover every family and partition the scored windows
    assert sorted(rep["per_family"]) == ["fam_0", "fam_1", "fam_2"]
    assert sum(b["network"]["scored_windows"] for b in rep["per_family"].values()) == len(scored)
    assert rep["bootstrap"]["resamples"] == 200 and rep["bootstrap"]["seed"] == metrics.BOOTSTRAP_SEED


def test_gate_rejected_predictions_are_ignored():
    rows = synthetic_rows(10)
    net, ref = synthetic_predictions(rows)
    net2, ref2 = net.copy(), ref.copy()
    for i, r in enumerate(rows):
        if not r.gate_passed:
            net2[i] = ref2[i] = 3
    a = metrics.score_report(rows, net, ref, n_resamples=20)
    b = metrics.score_report(rows, net2, ref2, n_resamples=20)
    assert a == b


def test_missing_prediction_for_gate_passed_window_is_refused():
    rows = synthetic_rows(5)
    net, ref = synthetic_predictions(rows)
    i = next(i for i, r in enumerate(rows) if r.gate_passed)
    net[i] = -1
    with pytest.raises(ValueError, match="network"):
        metrics.score_report(rows, net, ref, n_resamples=5)


# ---------------------------------------------------------------------------
# Disclosure sampler (README §2)
# ---------------------------------------------------------------------------


def test_disclosure_selection_rule(disclosure):
    rows = synthetic_rows(60, seed=9)
    k = 4
    chosen = disclosure.select(rows, per_class=k)
    keys = [key for key, _ in chosen]
    assert keys == sorted(keys) and len(chosen) == 4 * k
    for key, r in chosen:
        assert r.gate_passed and r.label.is_scored
        assert key == disclosure.window_key(r.session_id, r.endpoint)
    for c in CLASS_NAMES:
        picked = [r for _, r in chosen if r.label.value == c]
        assert len(picked) == k
        assert len({r.session_id for r in picked}) == k  # one window per session per class
        # independent walk: ascending key, skip sessions already taken
        cands = sorted(
            (disclosure.window_key(r.session_id, r.endpoint), r)
            for r in rows
            if r.gate_passed and r.label.value == c
        )
        want, seen = [], set()
        for key, r in cands:
            if r.session_id not in seen:
                want.append((r.session_id, r.endpoint))
                seen.add(r.session_id)
            if len(want) == k:
                break
        assert sorted(want) == sorted((r.session_id, r.endpoint) for r in picked)


def test_disclosure_selection_ignores_predictions_and_refuses_short_classes(disclosure):
    rows = synthetic_rows(6, seed=1)
    with pytest.raises(ValueError, match="distinct sessions"):
        disclosure.select(rows, per_class=50)


def test_disclosure_key_string(disclosure):
    import hashlib

    assert disclosure.window_key("s-abc", 12) == hashlib.sha256(b"20261104:phase-b-disclosure:s-abc:12").hexdigest()


def test_disclosure_public_fields(disclosure):
    rows = synthetic_rows(80, seed=4)
    public, private = disclosure.build(rows, zero_model(), lambda x: 3, {"artifact_sha256": "a" * 64}, {}, per_class=3)
    assert set(public) == {"format", "feature_names", "features_layout", "class_names", "fields", "windows"}
    assert public["feature_names"] == list(FEATURE_NAMES)
    ids = [w["id"] for w in public["windows"]]
    assert ids == [f"D{i:03d}" for i in range(1, 13)]
    by_key = {(r.session_id, r.endpoint): r for r in rows}
    for w, p in zip(public["windows"], private["windows"]):
        assert set(w) == {"id", "features", "oracle_label", "network_class", "reference_policy_class"}
        assert w["id"] == p["id"] and w["oracle_label"] == p["oracle_label"]
        assert w["network_class"] == "PRODUCTIVE" and w["reference_policy_class"] == "OSCILLATING"
        r = by_key[(p["session_id"], p["endpoint"])]
        x = np.asarray(w["features"], dtype=np.float32)
        assert x.shape == (8, 15) and x.tobytes() == r.vector
    keys = [p["key"] for p in private["windows"]]
    assert keys == sorted(keys)
    text = json.dumps(public)
    for p in private["windows"]:
        assert p["session_id"] not in text
