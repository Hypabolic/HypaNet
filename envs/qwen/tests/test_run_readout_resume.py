"""``hypaqwen.run_readout --sessions FILE`` and checkpoint/resume, with a fake model, tasks and verbaliser (CPU)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hypaqwen import run_readout as R
from hypaqwen.data import Task

TASKS = [Task(f"s-{i:02d}", [f"b{i}"] * 3, [0, 2]) for i in range(12)]


class _Crash(Exception):
    pass


def _fake(monkeypatch, crash_after=None):
    calls = []

    class Verb:
        def __init__(self, model, iface):
            pass

        def session(self, blocks, window_pos):
            if crash_after is not None and len(calls) == crash_after:
                raise _Crash
            i = int(blocks[0][1:])
            calls.append(i)
            return SimpleNamespace(label_logits=torch.full((2, 4), float(i)), logsumexp=torch.zeros(2),
                                   pos=[0, 2], tokens=10)

    monkeypatch.setattr(R.qmodel, "load", lambda device, dtype: (None, None))
    monkeypatch.setattr(R, "build_interface", lambda tok: SimpleNamespace(block_ids=lambda tok, b: b))
    monkeypatch.setattr(R, "Verbaliser", Verb)
    monkeypatch.setattr(R, "iter_tasks", lambda split: iter(TASKS))
    monkeypatch.setattr(R, "CHECKPOINT_EVERY", 2)
    return calls


def _subset(tmp_path, ids):
    p = tmp_path / "SUB.txt"
    p.write_text("".join(s + "\n" for s in ids))
    return p


def test_subset_keeps_export_order_and_is_named(tmp_path, monkeypatch):
    calls = _fake(monkeypatch)
    sub = _subset(tmp_path, ["s-07", "s-01", "s-04"])
    out = tmp_path / "o.npz"
    assert R.main(["--split", "val", "--sessions", str(sub), "--out", str(out)]) == 0
    z = np.load(out)
    assert calls == [1, 4, 7] and list(z["session_ids"]) == ["s-01", "s-04", "s-07"]
    meta = json.loads(str(z["meta"]))
    assert meta["subset"]["file"] == "SUB.txt" and meta["subset"]["sessions"] == 3
    assert not R.partial_path(out).exists()


def test_subset_with_unknown_id_is_refused(tmp_path, monkeypatch):
    _fake(monkeypatch)
    with pytest.raises(SystemExit, match="not in val"):
        R.main(["--split", "val", "--sessions", str(_subset(tmp_path, ["s-01", "s-99"])), "--out", str(tmp_path / "o.npz")])


def test_resume_after_crash_equals_one_run(tmp_path, monkeypatch):
    ref = tmp_path / "ref.npz"
    _fake(monkeypatch)
    R.main(["--split", "val", "--out", str(ref)])
    out = tmp_path / "o.npz"
    _fake(monkeypatch, crash_after=5)
    with pytest.raises(_Crash):
        R.main(["--split", "val", "--out", str(out)])
    assert R.partial_path(out).exists() and not out.exists()
    calls = _fake(monkeypatch)
    R.main(["--split", "val", "--out", str(out)])
    assert calls == list(range(4, 12))  # the checkpoint held 4 sessions
    a, b = np.load(ref), np.load(out)
    for k in ("label_logits", "logsumexp", "session_ids", "row_pos"):
        assert np.array_equal(a[k], b[k])
    assert json.loads(str(b["meta"]))["resumed_after_sessions"] == 4
    assert not R.partial_path(out).exists()


def test_partial_with_other_arguments_is_refused(tmp_path, monkeypatch):
    out = tmp_path / "o.npz"
    _fake(monkeypatch, crash_after=3)
    with pytest.raises(_Crash):
        R.main(["--split", "val", "--out", str(out)])
    _fake(monkeypatch)
    with pytest.raises(SystemExit, match="other arguments"):
        R.main(["--split", "val", "--sessions", str(_subset(tmp_path, ["s-01"])), "--out", str(out)])
