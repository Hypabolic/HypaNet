"""v0.4 scoring bridge: v0.3 val rows, strict column alignment, both populations, H2 not applicable."""

from __future__ import annotations

import json

import numpy as np
import pytest

from agent_loop_lab.v04 import export, scoring

pytestmark = pytest.mark.skipif(not export.labels_path("val").is_file(), reason="needs scripts/v04_export_tasks.py")


@pytest.fixture(scope="module")
def val():
    return scoring.load_rows()


def _column(tmp_path, lab, **over):
    n = len(lab["row_pos"])
    rng = np.random.default_rng(0)
    arr = {"label_logits": rng.normal(size=(n, 4)).astype(np.float32), "logsumexp": np.full(n, 5.0, np.float32),
           "session_ids": lab["session_ids"], "row_pos": lab["row_pos"],
           "meta": json.dumps({"split": "val", "limit": None})}
    arr.update(over)
    p = tmp_path / "col.npz"
    np.savez(p, **arr)
    return p


def test_rows_are_v03_rows(val):
    rows, lab = val
    assert len(rows) == 87176 and len(lab["session_ids"]) == 4004


def test_aligned_column_loads(val, tmp_path):
    _, lab = val
    pred, meta, _ = scoring.load_column(_column(tmp_path, lab), lab)
    assert pred.shape == (87176,) and pred.min() >= 0 and pred.max() <= 3


@pytest.mark.parametrize("bad", ["order", "row_pos", "limit", "nan"])
def test_misaligned_column_refused(val, tmp_path, bad):
    _, lab = val
    n = len(lab["row_pos"])
    over = {
        "order": {"session_ids": lab["session_ids"][::-1]},
        "row_pos": {"row_pos": np.roll(lab["row_pos"], 1)},
        "limit": {"meta": json.dumps({"split": "val", "limit": 10})},
        "nan": {"label_logits": np.full((n, 4), np.nan, np.float32)},
    }[bad]
    with pytest.raises(scoring.ScoringError):
        scoring.load_column(_column(tmp_path, lab, **over), lab)


def test_score_both_populations(val):
    rows, _ = val
    body = scoring.score(rows, {"constant": np.zeros(len(rows), dtype=np.int64)}, n_resamples=20)
    assert body["disclosed_sessions"]["count"] == 200
    assert set(body) >= {"excluding_disclosed", "all_val"}
