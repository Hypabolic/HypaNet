"""v0.4 task export: same windows as the v0.3 val scorer, rendered blocks unchanged, sealed splits refused."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from agent_loop_lab.v03 import scoring
from agent_loop_lab.v04 import export

N = 20


@pytest.mark.parametrize("split", ["test", "ood", "counterfactual", "train"])
def test_sealed_splits_refused(split, tmp_path):
    with pytest.raises(ValueError, match="may read only"):
        export.export_split(split, out_dir=tmp_path)


def test_rows_match_v03_val(tmp_path):
    entry = export.export_split("val", out_dir=tmp_path, limit=N)
    val = scoring.load_val(cache=False, limit=N)
    with np.load(export.labels_path("val", tmp_path)) as z:
        lab = {k: z[k] for k in z.files}
    assert entry["sessions"] == N == len(val.session_ids)
    assert list(lab["session_ids"]) == val.session_ids
    assert list(lab["families"]) == val.families
    np.testing.assert_array_equal(lab["row_session"], val.row_session)
    np.testing.assert_array_equal(lab["row_pos"], val.row_pos)
    assert [int(e) for e in lab["row_endpoint"]] == [r.endpoint for r in val.rows]
    assert [bool(g) for g in lab["row_gate"]] == [r.gate_passed for r in val.rows]
    assert [scoring._LABELS[i] for i in lab["row_label"]] == [r.label for r in val.rows]

    with gzip.open(export.tasks_path("val", tmp_path), "rt") as f:
        tasks = [json.loads(line) for line in f]
    for s, t in enumerate(tasks):
        assert set(t) == {"session_id", "blocks", "window_pos"}  # no labels, families or gate
        assert t["window_pos"] == val.row_pos[val.rows_of(s)].tolist()
        # One rendered block per event, as in the v0.3 encoding.
        assert len(t["blocks"]) == len(val.sessions[s].eoe_index)


def test_rebuild_is_byte_identical(tmp_path):
    a = export.export_split("val", out_dir=tmp_path / "a", limit=5)
    b = export.export_split("val", out_dir=tmp_path / "b", limit=5)
    assert a["files"]["val.tasks.jsonl.gz"] == b["files"]["val.tasks.jsonl.gz"]
    assert a["labels_content_sha256"] == b["labels_content_sha256"]
