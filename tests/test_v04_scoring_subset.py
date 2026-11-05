"""v0.4 scoring bridge on synthetic data only: full val path, quick-look subsets, coverage refusals,
priors, v0.3 columns and paired macro-F1 differences over shared bootstrap draws. No real labels are read."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from agent_loop_lab.eval import m4
from agent_loop_lab.eval import metrics as M
from agent_loop_lab.v04 import scoring, v03_val

REPO = Path(__file__).resolve().parents[1]
FAMS = ("fam_a", "fam_b", "fam_c")
NR = 40  # bootstrap resamples: enough to exercise the draws, cheap


def synth_lab(ids: list[str], seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    row_session, row_endpoint, row_label, row_gate, row_pos = [], [], [], [], []
    for s in range(len(ids)):
        for w in range(int(rng.integers(2, 6))):
            row_session.append(s)
            row_endpoint.append(w)
            g = bool(rng.random() > 0.1)
            row_gate.append(g)
            row_label.append(int(rng.integers(0, 4)) if g else 4)  # gate-rejected: INSUFFICIENT_INSTRUMENTATION
            row_pos.append(w + 1)
    return {"session_ids": np.array(ids), "families": np.array([FAMS[i % 3] for i in range(len(ids))]),
            "row_session": np.array(row_session, dtype=np.int64), "row_endpoint": np.array(row_endpoint),
            "row_label": np.array(row_label), "row_gate": np.array(row_gate), "row_pos": np.array(row_pos, dtype=np.int32)}


def write_column(path: Path, lab: dict, *, seed: int = 1, subset: dict | None = None, **over) -> Path:
    n = len(lab["row_pos"])
    rng = np.random.default_rng(seed)
    meta = {"split": "val", "limit": None, "subset": subset}
    arr = {"label_logits": rng.normal(size=(n, 4)).astype(np.float32), "logsumexp": np.full(n, 5.0, np.float32),
           "session_ids": lab["session_ids"], "row_pos": lab["row_pos"], "meta": json.dumps(meta)}
    arr.update(over)
    np.savez(path, **arr)
    return path


def write_subset(path: Path, ids: list[str]) -> dict:
    raw = "".join(s + "\n" for s in ids).encode()
    path.write_bytes(raw)
    return {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "sessions": len(ids)}


@pytest.fixture
def full():
    ids = [f"s-{i:04d}" for i in range(48)]
    lab = synth_lab(ids)
    rows = scoring.rows_from_labels(lab)
    return ids, lab, rows, frozenset(ids[:3])  # three "disclosed" sessions


def _build(rows, lab, disclosed, columns, **kw):
    return scoring.build_report("test", columns, rows, lab, disclosed, majority_index=0, n_resamples=NR, **kw)


# ---------------------------------------------------------------------------
# Full val: unchanged
# ---------------------------------------------------------------------------


def test_full_val_path_unchanged(full, tmp_path):
    ids, lab, rows, disclosed = full
    p = write_column(tmp_path / "q0.npz", lab)
    rep = _build(rows, lab, disclosed, {"Qwen-0": p})
    assert rep["report"] == "v04_val" and rep["subset"] is None and "quick_look" not in rep
    assert rep["disclosed_sessions"]["in_scored_sessions"] == 3
    pred, _, _ = scoring.load_column(p, lab)
    excl = np.array([r.session_id not in disclosed for r in rows])
    idx = np.flatnonzero(excl)
    direct = m4.score_columns([rows[i] for i in idx], {"Qwen-0": pred[idx]}, fidelity_pairs={}, n_resamples=NR)
    assert rep["excluding_disclosed"]["overall"]["columns"]["Qwen-0"] == direct["overall"]["columns"]["Qwen-0"]
    for f, blk in direct["per_family"].items():
        assert rep["excluding_disclosed"]["per_family"][f]["columns"]["Qwen-0"] == blk["columns"]["Qwen-0"]
    assert set(rep["excluding_disclosed"]["overall"]["columns"]) == {"Qwen-0", *scoring.PRIOR_COLUMNS}
    assert rep["visibility"].startswith("harness only")


def test_full_column_without_subset_key_still_loads(full, tmp_path):
    _, lab, _, _ = full
    n = len(lab["row_pos"])
    p = write_column(tmp_path / "old.npz", lab, meta=json.dumps({"split": "val", "limit": None}))
    pred, _, _ = scoring.load_column(p, lab)
    assert pred.shape == (n,)
    assert scoring.coverage({"Qwen-0": scoring.column_meta(p)}) is None


def test_limit_and_misalignment_refused(full, tmp_path):
    _, lab, _, _ = full
    for over in ({"meta": json.dumps({"split": "val", "limit": 10})}, {"session_ids": lab["session_ids"][::-1]},
                 {"row_pos": np.roll(lab["row_pos"], 1)}):
        with pytest.raises(scoring.ScoringError):
            scoring.load_column(write_column(tmp_path / "bad.npz", lab, **over), lab)


# ---------------------------------------------------------------------------
# Subsets
# ---------------------------------------------------------------------------


def test_subset_accepted_and_restricted(full, tmp_path):
    ids, lab, rows, disclosed = full
    chosen = [ids[i] for i in (40, 5, 17, 9, 33, 21, 12, 44, 28, 7)]  # file order is not export order
    rec = write_subset(tmp_path / "SUB.txt", chosen)
    _, sublab, mask = scoring.restrict(rows, lab, chosen)
    assert list(sublab["session_ids"]) == [s for s in ids if s in set(chosen)]
    a = write_column(tmp_path / "q0.npz", sublab, subset=rec, seed=1)
    b = write_column(tmp_path / "wa.npz", sublab, subset=rec, seed=2)
    rep = _build(rows, lab, disclosed, {"Qwen-0": a, "WSP-A": b}, subset_file=tmp_path / "SUB.txt")
    assert rep["report"] == "v04_val_subset" and rep["subset"] == rec
    assert rep["coverage"] == "subset SUB.txt (10 sessions)" and "quick_look" in rep
    assert rep["excluding_disclosed"]["overall"]["sessions"] == 10
    assert rep["disclosed_sessions"] == {**rep["disclosed_sessions"], "count": 3, "in_scored_sessions": 0}
    assert int(mask.sum()) == len(sublab["row_pos"])
    # the restricted rows are the full rows of those sessions, in order
    sub_rows = scoring.rows_from_labels(sublab)
    assert sub_rows == [r for r, m in zip(rows, mask) if m]


def test_val_quick_is_the_default_subset(tmp_path):
    quick, rec = scoring.read_subset(scoring.VAL_QUICK)
    assert rec["sha256"] == scoring.VAL_QUICK_SHA256 and len(quick) == 1000
    ids = sorted(quick + [f"s-extra{i:03d}" for i in range(30)])  # session ids only, no labels
    lab = synth_lab(ids, seed=3)
    rows = scoring.rows_from_labels(lab)
    _, sublab, _ = scoring.restrict(rows, lab, quick)
    a = write_column(tmp_path / "q0.npz", sublab, subset={"file": "VAL_QUICK.txt", "sha256": rec["sha256"],
                                                          "sessions": 1000})
    rep = _build(rows, lab, frozenset({"s-extra000"}), {"Qwen-0": a})
    assert rep["report"] == "v04_val_quick" and rep["subset"] == {**rec}
    assert rep["excluding_disclosed"]["overall"]["sessions"] == 1000


def test_wrong_subset_sha_refused(full, tmp_path):
    ids, lab, rows, disclosed = full
    chosen = ids[10:20]
    rec = write_subset(tmp_path / "SUB.txt", chosen)
    _, sublab, _ = scoring.restrict(rows, lab, chosen)
    bad = write_column(tmp_path / "q0.npz", sublab, subset={**rec, "sha256": "0" * 64})
    with pytest.raises(scoring.ScoringError, match="sha256"):
        _build(rows, lab, disclosed, {"Qwen-0": bad}, subset_file=tmp_path / "SUB.txt")
    # the default file (VAL_QUICK.txt) is not this subset either
    good = write_column(tmp_path / "q0b.npz", sublab, subset=rec)
    with pytest.raises(scoring.ScoringError, match="sha256"):
        _build(rows, lab, disclosed, {"Qwen-0": good})


def test_subset_column_with_other_sessions_refused(full, tmp_path):
    ids, lab, rows, disclosed = full
    chosen = ids[10:20]
    rec = write_subset(tmp_path / "SUB.txt", chosen)
    _, other, _ = scoring.restrict(rows, lab, ids[10:19])  # one session short
    p = write_column(tmp_path / "q0.npz", other, subset=rec)
    with pytest.raises(scoring.ScoringError, match="session ids"):
        _build(rows, lab, disclosed, {"Qwen-0": p}, subset_file=tmp_path / "SUB.txt")


def test_mixed_coverage_refused(full, tmp_path):
    ids, lab, rows, disclosed = full
    rec1 = write_subset(tmp_path / "S1.txt", ids[0:10])
    rec2 = write_subset(tmp_path / "S2.txt", ids[10:20])
    _, l1, _ = scoring.restrict(rows, lab, ids[0:10])
    _, l2, _ = scoring.restrict(rows, lab, ids[10:20])
    fullc = write_column(tmp_path / "full.npz", lab)
    s1 = write_column(tmp_path / "s1.npz", l1, subset=rec1)
    s2 = write_column(tmp_path / "s2.npz", l2, subset=rec2)
    for cols in ({"Qwen-0": fullc, "WSP-A": s1}, {"Qwen-0": s1, "WSP-A": s2}):
        with pytest.raises(scoring.ScoringError, match="different sessions"):
            _build(rows, lab, disclosed, cols, subset_file=tmp_path / "S1.txt")
    with pytest.raises(scoring.ScoringError, match="full val"):
        _build(rows, lab, disclosed, {"Qwen-0": fullc}, subset_file=tmp_path / "S1.txt")


# ---------------------------------------------------------------------------
# Paired differences
# ---------------------------------------------------------------------------


def test_paired_diff_uses_the_shared_draws(full, tmp_path):
    ids, lab, rows, disclosed = full
    a = write_column(tmp_path / "q0.npz", lab, seed=1)
    b = write_column(tmp_path / "wb.npz", lab, seed=2)
    rep = _build(rows, lab, disclosed, {"Qwen-0": a, "WSP-B": b})
    assert set(rep["paired_comparisons"]) == {"WSP-B - Qwen-0", "Qwen-0 - prior dev-majority"}
    for pop in ("excluding_disclosed", "all_val"):
        cols = rep[pop]["overall"]["columns"]
        d = rep[pop]["paired_macro_f1"]["WSP-B - Qwen-0"]
        assert d["value"] == pytest.approx(cols["WSP-B"]["macro_f1"]["value"] - cols["Qwen-0"]["macro_f1"]["value"])
    # the per-column draws behind the differences are score_columns' own draws: same CIs
    preds = {"Qwen-0": scoring.load_column(a, lab)[0], "WSP-B": scoring.load_column(b, lab)[0]}
    point, dist = scoring._macro_f1_draws(rows, preds, ["Qwen-0", "WSP-B"], n_resamples=NR, seed=M.BOOTSTRAP_SEED)
    sc = m4.score_columns(rows, preds, fidelity_pairs={}, n_resamples=NR)
    for j, c in enumerate(("Qwen-0", "WSP-B")):
        assert point[j] == sc["overall"]["columns"][c]["macro_f1"]["value"]
        assert M._ci(dist[:, j])[0] == sc["overall"]["columns"][c]["macro_f1"]["ci95"]
    d = scoring.paired_macro_f1(rows, preds, {"x": ("WSP-B", "Qwen-0")}, n_resamples=NR)["x"]
    assert d["ci95"] == M._ci(dist[:, 1] - dist[:, 0])[0]


def test_paired_diff_of_a_column_with_itself_is_zero(full, tmp_path):
    _, lab, rows, _ = full
    pred = scoring.load_column(write_column(tmp_path / "q0.npz", lab), lab)[0]
    d = scoring.paired_macro_f1(rows, {"a": pred, "b": pred.copy()}, {"a-b": ("a", "b")}, n_resamples=NR)["a-b"]
    assert d["value"] == 0.0 and d["ci95"] == [0.0, 0.0] and d["ci95_above_0"] is False


def test_explicit_diff_and_unknown_column(full, tmp_path):
    _, lab, rows, disclosed = full
    a = write_column(tmp_path / "q0.npz", lab, seed=1)
    b = write_column(tmp_path / "wa.npz", lab, seed=2)
    rep = _build(rows, lab, disclosed, {"Qwen-0": a, "WSP-A": b},
                 pairs={"WSP-A - prior always-PRODUCTIVE": ("WSP-A", "prior always-PRODUCTIVE")})
    assert list(rep["excluding_disclosed"]["paired_macro_f1"]) == ["WSP-A - prior always-PRODUCTIVE"]
    with pytest.raises(scoring.ScoringError, match="unknown"):
        _build(rows, lab, disclosed, {"Qwen-0": a}, pairs={"x": ("WSP-A", "Qwen-0")})


def test_default_pairs():
    assert scoring.default_pairs(["WSP-A"]) == {}
    assert scoring.default_pairs(["Qwen-0", "WSP-A", "WSP-B"]) == {
        "WSP-A - Qwen-0": ("WSP-A", "Qwen-0"), "WSP-B - Qwen-0": ("WSP-B", "Qwen-0"),
        "Qwen-0 - prior dev-majority": ("Qwen-0", "prior dev-majority")}


def test_priors_follow_the_gate(full):
    _, _, rows, _ = full
    pr = scoring.prior_columns(rows, 2)
    gate = np.array([r.gate_passed for r in rows])
    assert set(pr) == set(scoring.PRIOR_COLUMNS)
    assert (pr["prior dev-majority"][gate] == 2).all() and (pr["prior dev-majority"][~gate] == -1).all()
    assert (pr["prior always-PRODUCTIVE"][gate] == 0).all()
    assert scoring.recorded_dev_prior()["majority_index"] == 0  # the v0.3 sealed record: PRODUCTIVE


# ---------------------------------------------------------------------------
# v0.3 columns
# ---------------------------------------------------------------------------


def _v03_file(tmp_path, rows, *, match=True, rows_sha=scoring.VAL_ROWS_SHA256):
    rng = np.random.default_rng(7)
    preds = {"v0.3 B network": rng.integers(0, 4, len(rows)), "v0.3 C-32 seed0": rng.integers(0, 4, len(rows)),
             "v0.3 C-32 seed1": rng.integers(0, 4, len(rows))}
    meta = {"format": v03_val.FORMAT, "val_rows_sha256": rows_sha,
            "groups": {"v0.3 C-32": ["v0.3 C-32 seed0", "v0.3 C-32 seed1"]}, "reproduction": {"match": match}}
    p = tmp_path / "v03.val.npz"
    v03_val.save(p, list(preds), preds, meta)
    return p, preds


def test_v03_columns_enter_the_shared_draws(full, tmp_path):
    ids, lab, rows, disclosed = full
    p, v3 = _v03_file(tmp_path, rows)
    q = write_column(tmp_path / "wb.npz", lab, seed=2)
    rep = _build(rows, lab, disclosed, {"WSP-B": q}, v03_columns=p,
                 pairs={"WSP-B - v0.3 B network": ("WSP-B", "v0.3 B network")})
    cols = rep["excluding_disclosed"]["overall"]["columns"]
    assert {"WSP-B", "v0.3 B network", "v0.3 C-32 seed0", *scoring.PRIOR_COLUMNS} <= set(cols)
    assert "v0.3 C-32" in rep["excluding_disclosed"]["overall"]["groups"]
    d = rep["excluding_disclosed"]["paired_macro_f1"]["WSP-B - v0.3 B network"]
    assert d["value"] == pytest.approx(cols["WSP-B"]["macro_f1"]["value"] - cols["v0.3 B network"]["macro_f1"]["value"])
    # on a subset the v0.3 columns are restricted to the subset's rows
    rec = write_subset(tmp_path / "SUB.txt", ids[20:30])
    _, sublab, mask = scoring.restrict(rows, lab, ids[20:30])
    qs = write_column(tmp_path / "wbs.npz", sublab, subset=rec)
    reps = _build(rows, lab, disclosed, {"WSP-B": qs}, v03_columns=p, subset_file=tmp_path / "SUB.txt")
    sub_rows = [r for r, m in zip(rows, mask) if m]
    direct = m4.score_columns(sub_rows, {"v0.3 B network": v3["v0.3 B network"][mask]}, fidelity_pairs={},
                              n_resamples=NR)
    assert reps["all_val"]["overall"]["columns"]["v0.3 B network"] == direct["overall"]["columns"]["v0.3 B network"]


@pytest.mark.parametrize("bad", ["rows", "match"])
def test_v03_file_refused(full, tmp_path, bad):
    _, lab, rows, disclosed = full
    p, _ = _v03_file(tmp_path, rows, **({"rows_sha": "f" * 64} if bad == "rows" else {"match": False}))
    q = write_column(tmp_path / "wb.npz", lab)
    with pytest.raises(scoring.ScoringError):
        _build(rows, lab, disclosed, {"WSP-B": q}, v03_columns=p)


def test_reserved_names_refused(full, tmp_path):
    _, lab, rows, disclosed = full
    q = write_column(tmp_path / "wb.npz", lab)
    with pytest.raises(scoring.ScoringError, match="reserved"):
        _build(rows, lab, disclosed, {"prior dev-majority": q})


def test_v03_reproduction_check(full):
    _, _, rows, disclosed = full
    rng = np.random.default_rng(5)
    preds = {"v0.3 A network": rng.integers(0, 4, len(rows)), "v0.3 C-32 seed0": rng.integers(0, 4, len(rows)),
             **scoring.prior_columns(rows, 0)}
    groups = {"v0.3 C-32": ["v0.3 C-32 seed0"]}
    excl = np.flatnonzero([r.session_id not in disclosed for r in rows])
    recorded = {
        "excluding_disclosed": m4.score_columns([rows[i] for i in excl], {k: v[excl] for k, v in preds.items()}, {},
                                                groups, n_resamples=NR),
        "all_val": m4.score_columns(rows, preds, {}, groups, n_resamples=NR),
    }
    recorded = json.loads(json.dumps(recorded))  # as read back from the report
    out = v03_val.reproduction_check(rows, preds, groups, disclosed, recorded, n_resamples=NR)
    assert out["match"] and out["column_blocks_compared"] == 2 * 4 * (1 + len(FAMS))
    bad = dict(preds)
    bad["v0.3 A network"] = (preds["v0.3 A network"] + 1) % 4
    with pytest.raises(v03_val.ReproductionError, match="v0.3 A network"):
        v03_val.reproduction_check(rows, bad, groups, disclosed, recorded, n_resamples=NR)


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------


def _script():
    spec = importlib.util.spec_from_file_location("v04_score_validation", REPO / "scripts" / "v04_score_validation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_names_and_diffs():
    s = _script()
    assert s.report_name("wsp_a", None) == "v04_val_wsp_a.json"
    assert s.report_name("wsp_a", {"val_quick": True}) == "v04_val_quick_wsp_a.json"
    assert s.report_name("wsp_a", {"val_quick": False}) == "v04_val_subset_wsp_a.json"
    assert s._diffs([]) is None
    assert s._diffs(["WSP-A,Qwen-0"]) == {"WSP-A - Qwen-0": ("WSP-A", "Qwen-0")}
    with pytest.raises(SystemExit):
        s._diffs(["WSP-A"])


def test_script_refuses_mixed_coverage_before_scoring(full, tmp_path):
    ids, lab, rows, _ = full
    rec = write_subset(tmp_path / "S1.txt", ids[0:10])
    _, l1, _ = scoring.restrict(rows, lab, ids[0:10])
    a = write_column(tmp_path / "full.npz", lab)
    b = write_column(tmp_path / "s1.npz", l1, subset=rec)
    with pytest.raises(SystemExit, match="different sessions"):
        _script().main(["--arm", "x", "--column", f"Qwen-0={a}", "--column", f"WSP-A={b}",
                        "--report-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="VAL_QUICK"):
        _script().main(["--arm", "x", "--column", f"Qwen-0={b}", "--subset", str(tmp_path / "S1.txt")])
