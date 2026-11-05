"""Probe recipe (PREREGISTRATION §4): macro-F1, folds, tie rule, weights, L-BFGS fit, features, CLI end to end.

CPU only: a tiny random Qwen3.5 for the feature hook, synthetic features and labels for the fit.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from conftest import TINY_IFACE, CharTok, make_tiny, synthetic_sessions
from hypaqwen import budgets as B
from hypaqwen import probe as P
from hypaqwen.columns import macro_f1, session_rows


def test_macro_f1_by_hand():
    # class 0: P 1, R 1/2, F1 2/3; class 1: P 2/3, R 1, F1 4/5; class 2: P 1, R 1/2, F1 2/3; class 3: no support.
    assert macro_f1([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 3, 2]) == pytest.approx((2 / 3 + 4 / 5 + 2 / 3) / 3)
    # class 1 has support but is never predicted: precision 0, F1 0 (metrics_from_counts).
    assert macro_f1([0, 1], [0, 0]) == pytest.approx((2 / 3 + 0) / 2)
    assert macro_f1([3, 3], [3, 3]) == 1.0
    assert np.isnan(macro_f1([], []))


def test_cross_entropy_rows_matches_torch():
    """The elementwise form (deterministic on CUDA) equals ``F.cross_entropy``, value and gradient."""
    from hypaqwen.columns import cross_entropy_rows

    g = torch.Generator().manual_seed(0)
    x = (torch.randn(50, 4, generator=g, dtype=torch.float64) * 5).requires_grad_()
    y = torch.randint(0, 4, (50,), generator=g)
    a = cross_entropy_rows(x, y)
    b = torch.nn.functional.cross_entropy(x, y, reduction="none")
    assert torch.allclose(a, b, atol=1e-12)
    ga, = torch.autograd.grad(a.sum(), x)
    gb, = torch.autograd.grad(b.sum(), x)
    assert torch.allclose(ga, gb, atol=1e-12)


def test_fold_rule_is_rank_of_key_mod_5():
    ids = [f"s{i}" for i in range(23)]
    f = P.folds(ids)
    keyed = sorted(ids, key=lambda s: hashlib.sha256(f"20261105:v04-probe-fold:{s}".encode()).hexdigest())
    assert f == {s: r % 5 for r, s in enumerate(keyed)}
    assert P.folds(list(reversed(ids))) == f
    assert sorted(np.bincount(list(f.values()))) == [4, 4, 5, 5, 5]
    with pytest.raises(ValueError):
        P.folds(["a", "a"])


def test_balanced_weights():
    w = P.class_weights(np.array([0, 0, 0, 1]), 2)
    assert w == pytest.approx([4 / (2 * 3), 4 / (2 * 1)])
    y = np.array([0, 1, 1, 2, 2, 2])
    assert P.class_weights(y, 3)[y].sum() == pytest.approx(len(y))  # every class weighs n / k in total


def test_standardise_population_std_and_floor():
    X = torch.tensor([[1.0, 5.0], [3.0, 5.0]])
    mean, std = P.standardise(X)
    assert mean.tolist() == [2.0, 5.0] and std.tolist() == [1.0, P.STD_FLOOR]
    assert mean.dtype == torch.float64


class _Stub:
    def __init__(self, pred):
        self.pred, self.info = pred, {}

    def predict(self, Z):
        return np.full(Z.shape[0], self.pred, dtype=np.int64)


def _toy_set(n_sessions=20):
    rng = np.random.default_rng(0)
    sessions = [f"s{i}" for i in range(n_sessions)]
    groups = [s for s in sessions for _ in range(3)]
    return P.TrainSet(rng.normal(size=(len(groups), 4)).astype(np.float32), rng.integers(0, 4, len(groups)),
                      groups, sessions)


def test_ties_go_to_the_smaller_C():
    ts = _toy_set()
    c, table = P.cross_validate(ts, fit=lambda X, y, C: (torch.zeros(4), torch.ones(4), _Stub(0)))
    assert c == min(P.C_GRID) and len({r["oof_macro_f1"] for r in table}) == 1
    # strictly better only at C = 1: chosen; C = 10 ties with 1 and loses.
    best = int(np.bincount(ts.y).argmax())

    def fit(X, y, C):
        return torch.zeros(4), torch.ones(4), _Stub(best if C >= 1.0 else (best + 1) % 4)

    c, table = P.cross_validate(ts, fit=fit)
    assert c == 1.0
    assert [r["C"] for r in table] == sorted(P.C_GRID)


def test_cv_standardises_on_training_folds_only():
    ts = _toy_set()
    seen = []

    def fit(X, y, C):
        seen.append(X.shape[0])
        return torch.zeros(4), torch.ones(4), _Stub(0)

    P.cross_validate(ts, grid=[1.0], fit=fit)
    fold_of = P.folds(ts.sessions)
    sizes = np.bincount([fold_of[g] for g in ts.groups], minlength=5)
    assert seen == [len(ts.y) - s for s in sizes]


def _blobs(n=400, d=5, classes=(0, 1, 2, 3), sep=6.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(np.array(classes), size=n)
    centres = rng.normal(size=(4, d)) * sep
    return (centres[y] + rng.normal(size=(n, d))).astype(np.float64), y


def test_lbfgs_converges_on_a_separable_set():
    X, y = _blobs()
    mean, std = P.standardise(torch.as_tensor(X))
    Z = (torch.as_tensor(X) - mean) / std
    lr = P.fit_logreg(Z, y, 1.0)
    assert lr.info["converged"] and lr.info["grad_max"] <= P.GTOL and 0 < lr.info["iterations"] < P.MAX_ITER
    assert (lr.predict(Z) == y).all()
    # stationarity of the §4 objective: the intercept gradient is the weighted residual sum
    p = torch.softmax(Z @ lr.W.T + lr.b, dim=1)
    onehot = torch.nn.functional.one_hot(torch.as_tensor(y), 4).to(torch.float64)
    sw = torch.as_tensor(P.class_weights(y, 4))[torch.as_tensor(y)]
    assert ((sw[:, None] * (p - onehot)).sum(0)).abs().max() < 1e-5
    # the penalty binds: a smaller C gives smaller weights
    assert P.fit_logreg(Z, y, 1e-3).W.norm() < lr.W.norm()


def test_absent_class_gets_no_row_and_is_never_predicted():
    X, y = _blobs(classes=(0, 2, 3))
    lr = P.fit_logreg(torch.as_tensor(X), y, 1.0)
    assert lr.classes.tolist() == [0, 2, 3] and lr.W.shape[0] == 3
    logits = lr.logits4(torch.as_tensor(X))
    assert (logits[:, 1] == P.ABSENT_LOGIT).all() and torch.isfinite(logits).all()
    assert 1 not in set(lr.predict(torch.as_tensor(X)).tolist())


def test_single_class_budget():
    lr = P.fit_logreg(torch.zeros(5, 3, dtype=torch.float64), np.full(5, 2), 1.0)
    assert lr.predict(torch.zeros(5, 3, dtype=torch.float64)).tolist() == [2] * 5


def test_feature_hook_reads_the_final_norm_input(tiny_config):
    """hidden[n_layers] at each window equals the pre-norm residual of the separate prompt; logits unchanged."""
    model = make_tiny(tiny_config)
    tok, iface = CharTok(), TINY_IFACE
    reader = P.FeatureReader(model, iface)
    s = synthetic_sessions(1, seed=5)[0]
    bids = iface.block_ids(tok, s["blocks"])
    seen = {}
    with torch.no_grad():
        r, h = reader.features(bids, s["window_pos"])
        assert h.shape == (len(s["window_pos"]), tiny_config.hidden_size)
        handle = model.model.norm.register_forward_pre_hook(
            lambda m, a: seen.__setitem__("x", a[0][0, -1].clone()) if a[0].ndim == 3 else None)
        try:
            for i, t in enumerate(s["window_pos"]):
                out = model.model(input_ids=torch.tensor([iface.window_prompt(bids, t)]))
                assert torch.allclose(h[i], seen["x"], atol=1e-4, rtol=1e-4)
                # the final norm of the feature is the readout state, and the feature is not that state
                assert torch.allclose(model.model.norm(seen["x"]), out.last_hidden_state[0, -1], atol=1e-5)
                assert not torch.allclose(h[i], out.last_hidden_state[0, -1], atol=1e-3)
        finally:
            handle.remove()
        for i, t in enumerate(s["window_pos"]):
            assert torch.allclose(r.label_logits[i], reader.reference_window(bids, t).label_logits[0], atol=1e-4)
    assert not model.model.norm._forward_pre_hooks  # the reader's hook is removed after each session


def _features_file(path, sessions, labels, split, d=6, seed=0):
    """A features file whose features carry the label (plus noise), in export row order."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(6, d)) * 2.0
    lab = labels["row_label"].astype(np.int64)
    X = (centres[lab] + rng.normal(size=(len(lab), d))).astype(np.float32)
    meta = {"format": P.FEATURES_FORMAT, "split": split, "limit": None, "hidden": {"layer": 24}, "device": "cpu",
            "dtype": "fp32"}
    np.savez(path, features=X, label_logits=np.zeros((len(lab), 4), np.float32), logsumexp=np.zeros(len(lab)),
             session_ids=labels["session_ids"], row_session=labels["row_session"], row_pos=labels["row_pos"],
             meta=json.dumps(meta))
    return path


def test_train_sets_take_gate_passing_scored_windows(world, tmp_path):
    feats = P.load_features(_features_file(tmp_path / "dev.npz", world["dev"], world["dev_labels"], "dev"))
    b = B.load(world["budgets"])
    ts = P.dev_train_set(feats, world["dev_labels"], b.sessions("32"))
    rows = session_rows(world["dev_labels"])
    want = sum(int(rows[s].train.sum()) for s in b.sessions("32"))
    assert len(ts.y) == want and ts.y.max() < 4 and ts.sessions == list(b.sessions("32"))
    vf = P.load_features(_features_file(tmp_path / "val.npz", world["val"], world["val_labels"], "val"))
    ws = P.window_train_set(vf, world["val_labels"], b.disclosed_200w)
    assert len(ws.y) == 200 and sorted(np.bincount(ws.y).tolist()) == [50, 50, 50, 50]


def test_cli_fit_and_predict(world, tmp_path):
    dev = _features_file(tmp_path / "dev.npz", world["dev"], world["dev_labels"], "dev")
    val = _features_file(tmp_path / "val.npz", world["val"], world["val_labels"], "val", seed=0)
    common = ["--budgets", str(world["budgets"]), "--data-dir", str(world["dir"]), "--device", "cpu"]
    assert P.main(["fit", "--budget", "32", "--features", str(dev), "--out", str(tmp_path / "p32"), *common]) == 0
    log = json.loads((tmp_path / "p32" / "probe_log.json").read_text())
    assert log["chosen_C"] in P.C_GRID and len(log["cv"]) == len(P.C_GRID) and log["refit"]["converged"]
    assert P.main(["fit", "--budget", "200w", "--val-features", str(val), "--out", str(tmp_path / "p200"),
                   *common]) == 0
    log200 = json.loads((tmp_path / "p200" / "probe_log.json").read_text())
    assert log200["train_windows"] == 200 and log200["features_split"] == "val"
    with pytest.raises(ValueError):  # dev budgets refuse val features
        P.main(["fit", "--budget", "32", "--features", str(val), "--out", str(tmp_path / "bad"), *common])
    col = tmp_path / "col.npz"
    assert P.main(["predict", "--model", str(tmp_path / "p32"), "--features", str(val), "--out", str(col)]) == 0
    with np.load(col) as z:
        arr = {k: z[k] for k in z.files}
    meta = json.loads(str(arr["meta"]))
    assert meta["readout"] == "probe" and meta["split"] == "val" and meta["limit"] is None
    assert list(arr["session_ids"]) == list(world["val_labels"]["session_ids"])
    assert np.array_equal(arr["row_pos"], world["val_labels"]["row_pos"])
    lg = arr["label_logits"]
    assert lg.shape == (len(arr["row_pos"]), 4) and np.isfinite(lg).all()
    assert np.allclose(arr["logsumexp"], np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) + lg.max(1))
    # the features carry the label: the probe is far above chance on the scored val windows
    lab = world["val_labels"]
    m = lab["row_gate"] & (lab["row_label"] < 4)
    assert macro_f1(lab["row_label"][m], lg.argmax(1)[m]) > 0.8


def test_cli_features_on_a_tiny_model(world, tmp_path, monkeypatch, tiny_factory):
    """``features`` end to end with the tiny model in place of the pinned checkpoint: export row order, meta."""
    from hypaqwen import interface as iface_mod
    from hypaqwen import model as qmodel

    monkeypatch.setattr(qmodel, "load", tiny_factory)
    monkeypatch.setattr(iface_mod, "build_interface", lambda tok: TINY_IFACE)
    out = tmp_path / "f.npz"
    assert P.main(["features", "--split", "dev", "--data-dir", str(world["dir"]), "--limit", "4", "--device", "cpu",
                   "--out", str(out)]) == 0
    f = P.load_features(out)
    lab = world["dev_labels"]
    n = int((lab["row_session"] < 4).sum())
    assert f.session_ids == [str(s) for s in lab["session_ids"][:4]] and np.array_equal(f.row_pos, lab["row_pos"][:n])
    assert f.features.shape == (n, 64) and f.meta["split"] == "dev" and f.meta["hidden"]["layer"] == 4
    sub = tmp_path / "f32.npz"
    assert P.main(["features", "--split", "dev", "--budget", "32", "--budgets", str(world["budgets"]),
                   "--data-dir", str(world["dir"]), "--limit", "2", "--device", "cpu", "--out", str(sub)]) == 0
    assert set(P.load_features(sub).session_ids) <= set(B.load(world["budgets"]).sessions("32"))
    with pytest.raises(ValueError):  # a sealed split never comes from the export reader
        P.main(["features", "--split", "test", "--data-dir", str(world["dir"]), "--device", "cpu",
                "--out", str(tmp_path / "t.npz")])
