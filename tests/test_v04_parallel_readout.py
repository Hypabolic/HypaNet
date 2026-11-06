"""scripts/v04_parallel_readout.py (deviation D2): sharding, seeding and the merge, with fake shards (no model)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v04_parallel_readout", REPO / "scripts" / "v04_parallel_readout.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

WINDOWS = {f"s{i}": list(range(7, 7 + 1 + i % 3)) for i in range(10)}  # export order s0..s9, 1-3 windows each
ORDER = list(WINDOWS)
META = {"split": "val", "limit": None, "device": "mps", "dtype": "fp32", "torch": "t", "transformers": "x",
        "platform": "p", "python": "3", "attn_implementation": "sdpa", "model": {"id": "q"}, "interface_sha256": "i",
        "interface": {"spec_sha256": "s"}, "edit": None, "subset": {"file": "shard.txt"}, "tokens": 10, "seconds": 5.0,
        "sessions": 0}


def _logit(s, p):
    return [float(int(s[1:]) * 100 + p + c / 10) for c in range(4)]


def _rows(ids, **meta):
    pos = [p for s in ids for p in WINDOWS[s]]
    return {"label_logits": np.array([_logit(s, p) for s in ids for p in WINDOWS[s]], np.float32).reshape(-1, 4),
            "logsumexp": np.array([float(int(s[1:])) for s in ids for _ in WINDOWS[s]], np.float32),
            "session_ids": list(ids), "row_pos": np.array(pos, np.int32), "meta": {**META, "sessions": len(ids), **meta}}


def _merge(shards, wanted=ORDER, seed=None, subset=None):
    return P.merge(ORDER, WINDOWS, wanted, shards, seed, subset, {"workers": len(shards)}, 50.0)


def test_plan_is_round_robin_in_export_order_and_skips_the_seed():
    assert P.plan(ORDER, None, [], 3) == [["s0", "s3", "s6", "s9"], ["s1", "s4", "s7"], ["s2", "s5", "s8"]]
    assert P.plan(ORDER, ["s9", "s2", "s4"], ["s2"], 2) == [["s4"], ["s9"]]
    with pytest.raises(SystemExit, match="not in val"):
        P.plan(ORDER, ["s1", "x"], [], 2)
    with pytest.raises(SystemExit, match="outside"):
        P.plan(ORDER, ["s1"], ["s2"], 2)


def test_merge_puts_rows_back_in_export_order():
    shards = [(f"shard{k}", _rows(sh)) for k, sh in enumerate(P.plan(ORDER, None, [], 3))]
    arr, meta = _merge(shards)
    ref = _rows(ORDER)
    for k in ("label_logits", "logsumexp", "row_pos"):
        assert np.array_equal(arr[k], ref[k])
    assert list(arr["session_ids"]) == ORDER and arr["row_pos"].dtype == np.int32
    assert meta["subset"] is None and meta["sessions"] == 10 and meta["tokens"] == 30 and meta["limit"] is None
    assert meta["edit"] is None and meta["model"] == {"id": "q"} and meta["parallel"] == {"workers": 3}


def test_merge_with_a_seed_partial_and_a_subset():
    wanted = ["s1", "s2", "s5", "s8"]
    seed = ("p.partial.npz", {**_rows(["s1", "s2"]), "meta": {"run_args": {}, "tokens": 7, "seconds": 3.0}})
    shards = [(f"shard{k}", _rows(sh)) for k, sh in enumerate(P.plan(ORDER, wanted, ["s1", "s2"], 2))]
    sub = {"file": "SUB.txt", "sha256": "h", "sessions": 4}
    arr, meta = _merge(shards, wanted, seed, sub)
    assert list(arr["session_ids"]) == wanted and np.array_equal(arr["label_logits"], _rows(wanted)["label_logits"])
    assert meta["subset"] == sub and meta["resumed_after_sessions"] == 2 and meta["tokens"] == 27


def test_merge_refuses_inconsistent_shards():
    a, b = P.plan(ORDER, None, [], 2)
    with pytest.raises(P.MergeError, match="meta differs"):
        _merge([("shard0", _rows(a)), ("shard1", _rows(b, edit={"edit_state_sha256": "e"}))])
    with pytest.raises(P.MergeError, match="missing"):
        _merge([("shard0", _rows(a))])
    with pytest.raises(P.MergeError, match="two shards"):
        _merge([("shard0", _rows(a)), ("shard1", _rows(b + ["s0"]))])
    bad = _rows(b)
    bad["row_pos"] = bad["row_pos"] + 1
    with pytest.raises(P.MergeError, match="window_pos"):
        _merge([("shard0", _rows(a)), ("shard1", bad)])
    short = _rows(b)
    short["logsumexp"] = short["logsumexp"][:-1]
    with pytest.raises(P.MergeError, match="left over"):
        _merge([("shard0", _rows(a)), ("shard1", short)])
    with pytest.raises(P.MergeError, match="complete val"):
        _merge([("shard0", _rows(a, limit=5)), ("shard1", _rows(b, limit=5))])


def test_merged_file_is_what_scoring_load_column_reads(tmp_path):
    from agent_loop_lab.v04 import scoring as S

    shards = [(f"shard{k}", _rows(sh)) for k, sh in enumerate(P.plan(ORDER, None, [], 3))]
    arr, meta = _merge(shards)
    P.save(tmp_path / "col.npz", arr, meta)
    lab = {"session_ids": np.array(ORDER), "row_pos": np.array([p for s in ORDER for p in WINDOWS[s]])}
    pred, got, _ = S.load_column(tmp_path / "col.npz", lab)
    assert got == meta and pred.tolist() == [3] * len(lab["row_pos"])


def test_a_kept_shard_must_match_the_requested_arguments_and_its_shard_list():
    meta = {**META, "edit": {"build_record_sha256": "e"}, "subset": {"sha256": "s"}}
    P.check_shard("shard0", meta, device="mps", dtype="fp32", edit_sha="e", shard_sha="s")
    for kw, why in (({"edit_sha": "f"}, "edit BUILD_RECORD"), ({"edit_sha": None}, "edit BUILD_RECORD"),
                    ({"shard_sha": "t"}, "not the plan's shard"), ({"device": "cpu"}, "not cpu/fp32")):
        args = {"device": "mps", "dtype": "fp32", "edit_sha": "e", "shard_sha": "s", **kw}
        with pytest.raises(P.MergeError, match=why):
            P.check_shard("shard0", meta, **args)
    P.check_shard("shard0", {**META, "subset": {"sha256": "s"}}, device="mps", dtype="fp32", edit_sha=None,
                  shard_sha="s")


def test_merge_without_worker_shards_is_a_refusal_not_a_crash():
    with pytest.raises(P.MergeError, match="no worker shards"):
        _merge([])
