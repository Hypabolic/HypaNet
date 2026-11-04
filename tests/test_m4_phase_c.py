"""M4 Phase C: budget-subset rule and a tiny trainer smoke (experiments/m4/README.md §1)."""

from __future__ import annotations

import hashlib
import subprocess
import sys

import numpy as np
import torch

from agent_loop_lab.baselines import phase_c
from agent_loop_lab.model import DoomLoopMLP


def _ids(n: int) -> list[str]:
    return [f"s-{i:016x}" for i in range(n)]


def test_budget_key_formula():
    sid = "s-0123456789abcdef"
    assert phase_c.budget_key(sid) == hashlib.sha256(f"20261105:m4-budget:{sid}".encode()).hexdigest()


def test_budget_order_is_ascending_hex_and_input_order_free():
    ids = _ids(300)
    order = phase_c.budget_order(ids)
    keys = [hashlib.sha256(f"20261105:m4-budget:{s}".encode()).hexdigest() for s in order]
    assert keys == sorted(keys)
    assert phase_c.budget_order(list(reversed(ids))) == order
    assert sorted(order) == sorted(ids)


def test_budget_subsets_nested_and_sized():
    ids = _ids(600)
    prev: list[str] = []
    for b in (32, 128, 512):
        sub = phase_c.budget_subset(ids, b)
        assert len(sub) == b and len(set(sub)) == b
        assert sub[: len(prev)] == prev
        prev = sub
    assert phase_c.budget_subset(ids, "all") == phase_c.budget_order(ids)


def test_budget_rejects_bad_values():
    import pytest

    for bad in (0, -1, 601, "some", True):
        with pytest.raises(ValueError):
            phase_c.budget_subset(_ids(600), bad)
    with pytest.raises(ValueError):
        phase_c.budget_order(["a", "a"])


def _toy(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 120)).astype(np.float32)
    y = (x[:, 0] > 0.5).astype(np.int64) + 2 * (x[:, 1] > 0.5).astype(np.int64)
    return x, y


def test_trainer_smoke_learns_and_is_deterministic():
    torch.set_num_threads(1)
    x, y = _toy(600, 0)
    xv, yv = _toy(300, 1)
    r1 = phase_c.train_run(x, y, xv, yv, seed=0, budget="toy", max_epochs=6, patience=3)
    r2 = phase_c.train_run(x, y, xv, yv, seed=0, budget="toy", max_epochs=6, patience=3)
    assert r1.train_windows == 600
    assert 1 <= r1.best_epoch <= r1.epochs_run <= 6
    assert len(r1.history) == r1.epochs_run
    assert r1.best_val_macro_f1 == max(h["val_macro_f1"] for h in r1.history)
    # best = earliest maximum
    assert r1.history[r1.best_epoch - 1]["val_macro_f1"] == r1.best_val_macro_f1
    assert all(h["val_macro_f1"] < r1.best_val_macro_f1 for h in r1.history[: r1.best_epoch - 1])
    for k, v in r1.model.state_dict().items():
        assert torch.equal(v, r2.model.state_dict()[k])
    # The returned model reproduces the recorded best val macro-F1.
    with torch.no_grad():
        pred = torch.argmax(r1.model(torch.from_numpy(xv)), -1).numpy()
    assert phase_c.macro_f1(yv, pred) == r1.best_val_macro_f1
    # Not zero-initialised: a different seed gives different weights.
    r3 = phase_c.train_run(x, y, xv, yv, seed=1, max_epochs=1, patience=1)
    assert not torch.equal(r1.model.fc1.weight, r3.model.fc1.weight)


def test_default_init_is_not_zero():
    torch.manual_seed(0)
    m = DoomLoopMLP()
    assert m.fc1.weight.abs().sum() > 0


def test_patience_stops_early():
    x, y = _toy(64, 2)
    xv = np.zeros((10, 120), np.float32)  # constant input: val macro-F1 can never strictly improve twice
    yv = np.zeros(10, np.int64)
    r = phase_c.train_run(x, y, xv, yv, seed=0, max_epochs=50, patience=2)
    assert r.epochs_run < 50 and r.stopped.startswith("patience")


def test_weight_build_path_does_not_import_baselines():
    code = (
        "import sys; import agent_loop_lab.runner, agent_loop_lab.packet, agent_loop_lab.guards; "
        "print(any(m.startswith('agent_loop_lab.baselines') for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"
