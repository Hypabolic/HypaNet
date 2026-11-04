"""v0.3 dev rendering audits (spec §7.3): dev-only guard and the audit arithmetic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agent_loop_lab.v03 import audits as A

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("split", ["val", "test", "counterfactual", "ood"])
def test_refuses_non_dev(split, tmp_path):
    with pytest.raises(ValueError, match="dev"):
        A.run_dev_audits(tmp_path, tmp_path, split=split)


def test_stats_refuse_sealed_splits(tmp_path):
    from agent_loop_lab.v03.token_stats import iter_encoded

    for split in ("test", "counterfactual", "ood"):
        with pytest.raises(ValueError):
            next(iter_encoded(tmp_path, split))


def test_hunk_ids_and_evidence():
    blocks = [
        "x edit_file path=a -> ok in=1 out=1\n  @ a haaaaaaaaaa > hbbbbbbbbbb",
        "x read_file path=a -> ok in=1 out=1",
        "x edit_file path=a -> ok in=1 out=1\n  @ a hbbbbbbbbbb > hcccccccccc\n  @ TRUNC 3",
        "x write_file path=a bytes=1 -> ok in=1 out=1\n  @ a hcccccccccc > haaaaaaaaaa",
        "x revert paths=a -> ok in=1 out=1\n  @ a haaaaaaaaaa > h0",
    ]
    ids = [A.hunk_ids(b) for b in blocks]
    assert ids[0] == [("a", "haaaaaaaaaa", "hbbbbbbbbbb")] and ids[1] == [] and len(ids[2]) == 1
    assert A._evidence(3, 0, ids)  # returns to the content first replaced at event 0
    assert not A._evidence(3, 1, ids)  # ...but not if event 0 is outside the window
    assert not A._evidence(4, 0, ids)  # an h0 (now empty) side is never evidence
    assert not A._evidence(1, 0, ids)


def test_macro_f1_and_prior():
    y = np.array([1, 1, 1, 0])
    assert A.macro_f1(y, np.array([1, 1, 1, 1])) == pytest.approx((6 / 7 + 0) / 2)
    assert A.macro_f1(y, y.copy()) == 1.0


def test_logistic_cv_finds_signal_and_not_noise():
    n = 2000
    idx = np.arange(n)
    folds = idx % 5
    x_sig = (idx % 3 == 0).astype(float)[:, None]
    y = x_sig[:, 0].copy()
    good = A.logistic_cv(x_sig, y, folds)
    assert good["macro_f1_threshold_0_5"] == 1.0 and good["auc"] == 1.0
    x_noise = ((idx * 7919) % 11 / 10.0)[:, None]
    bad = A.logistic_cv(x_noise, (idx % 2).astype(float), folds)
    assert bad["auc"] < 0.6 and bad["macro_f1_constant_prior"] <= 0.5


def test_leakage_markers_are_the_requested_set():
    assert set(A.LEAKAGE_MARKERS) == {"cleanup pass", "revised", "-ok", "enabled=", "investigation", "explanation",
                                      "+1)"}


def test_small_dev_run():
    if not (REPO / "datasets" / "v0_3" / "dev.raw.jsonl.gz").exists():
        pytest.skip("v0.3 raw datasets not built")
    rep = A.run_dev_audits(REPO / "datasets" / "v0_3", REPO / "datasets" / "v0_2", limit=60)
    assert rep["split"] == "dev" and rep["sessions"] == 60
    assert rep["leakage_scan"]["clean"] is True
    assert rep["marker_audit"]["n"] > 0
    assert set(rep["length"]["tokens_per_block_by_family"]) <= {
        "read_file", "search", "list_dir", "edit_file", "write_file", "revert", "run_tests", "build", "lint",
        "http_call", "poll_status", "plan", "ask", "yield"}
