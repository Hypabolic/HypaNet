"""LoRA recipe (PREREGISTRATION §5): targets and count, caps, schedule, ES rule, branched loss, train/merge/readout.

CPU, fp32: the parameter count on the meta device with the pinned configuration; everything that
runs a model uses a tiny random Qwen3.5 built from that configuration, synthetic sessions and
synthetic targets. ``test_branched_equals_separate_prompts_real_model`` is the slow variant on the
pinned checkpoint and 4 dev sessions (no labels): ``HYPAQWEN_SLOW=1``, device ``HYPAQWEN_TEST_DEVICE``.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import TINY_IFACE, CharTok, make_tiny, synthetic_sessions
from hypaqwen import budgets as B
from hypaqwen import lora as L
from hypaqwen.columns import write_column
from hypaqwen.readout import Verbaliser

REL_TOL = 1e-4  # §5: branched vs separate prompts, fp32


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


# ---------------------------------------------------------------- protocol numbers


def test_target_modules_and_trainable_count(real_config):
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(real_config)
    assert sum(p.numel() for p in model.parameters()) == L.TEXT_PARAMETERS
    names = L.target_modules(model.config)
    assert len(names) == 6 * 4 + 18 * 3 + 24 * 3
    attn = sorted({int(n.split(".")[2]) for n in names if ".self_attn." in n})
    assert tuple(attn) == L.FULL_ATTENTION_LAYERS
    assert {n.rsplit(".", 1)[1] for n in names} == {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv",
                                                    "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj"}
    assert not any(x in n for n in names for x in ("in_proj_a", "in_proj_b", "conv1d", "embed", "lm_head", "norm"))
    pm = L.attach_lora(model, 0)
    assert L.trainable_parameters(pm) == L.TRAINABLE_PARAMETERS == 10_223_616
    assert round(100 * L.TRAINABLE_PARAMETERS / L.TEXT_PARAMETERS, 2) == 1.36
    trainable = [n for n, p in pm.named_parameters() if p.requires_grad]
    assert all(".lora_A." in n or ".lora_B." in n for n in trainable) and len(trainable) == 2 * len(names)
    from peft.tuners.lora import LoraLayer

    lora = {n: m for n, m in pm.get_base_model().named_modules() if isinstance(m, LoraLayer)}
    assert sorted(lora) == sorted(names)
    m = lora[names[0]]
    assert m.r["default"] == 16 and m.scaling["default"] == 2.0 and m.lora_dropout["default"].p == 0.05


def test_epoch_caps_and_planned_steps():
    caps = {32: 60, 128: 48, 512: 12, 2048: 3, 12000: 3, 200: 31}
    assert {b: L.epoch_cap(b) for b in caps} == caps
    # §5's planned steps with full pools (all-dev pool 11,864 sessions in v0.3)
    assert [L.planned_steps(L.epoch_cap(b), p) for b, p in
            ((32, 32), (128, 128), (512, 512), (2048, 2048), (12000, 11864), (200, 200))] == [240, 768, 768, 768,
                                                                                              4449, 775]
    assert L.planned_steps(3, 9) == 6  # last partial batch kept


def test_schedule_shape():
    for S in (240, 768, 775, 4449):
        w = L.warmup_steps(S)
        assert w == math.ceil(0.05 * S)
        lr = [L.lr_at(s, S) for s in range(1, S + 1)]
        assert lr[0] == pytest.approx(L.LR / w) and lr[w - 1] == pytest.approx(L.LR)
        assert all(a < b for a, b in zip(lr[:w - 1], lr[1:w]))  # linear warm-up
        assert all(a >= b for a, b in zip(lr[w - 1:], lr[w:]))  # then non-increasing
        assert lr[-1] == pytest.approx(0.0, abs=1e-15)
        mid = w + (S - w) // 2
        if (S - w) % 2 == 0:
            assert L.lr_at(mid, S) == pytest.approx(L.LR / 2)
    with pytest.raises(ValueError):
        L.lr_at(0, 10)


def test_es_evaluation_epochs():
    assert L.es_eval_epochs(60) == list(range(6, 61, 6))
    assert L.es_eval_epochs(48) == [5, 10, 15, 20, 25, 30, 35, 40, 45, 48]
    assert L.es_eval_epochs(12) == [2, 4, 6, 8, 10, 12]
    assert L.es_eval_epochs(3) == [1, 2, 3]
    assert all(len(L.es_eval_epochs(c)) <= 11 and L.es_eval_epochs(c)[-1] == c for c in range(1, 61))


# ---------------------------------------------------------------- branched loss


def _session_blocks(n, seed=0):
    tok = CharTok()
    return [(s["session_id"], TINY_IFACE.block_ids(tok, s["blocks"]), s["window_pos"])
            for s in synthetic_sessions(n, seed=seed)]


def _lora_tiny(cfg, seed=0, b_std=0.05):
    pm = L.attach_lora(make_tiny(cfg), seed, check_count=False)
    with torch.no_grad():
        g = torch.Generator().manual_seed(123)
        for n, p in pm.named_parameters():
            if "lora_B" in n:  # B = 0 at init would make the adapters invisible to the check
                p.copy_(torch.randn(p.shape, generator=g) * b_std)
    return pm.eval()


def test_branched_equals_separate_prompts(tiny_config):
    """§5: the branched logits (and loss, and gradients) equal separate prompts on 4 sessions, fp32."""
    pm = _lora_tiny(tiny_config)
    hf = pm.get_base_model()
    params = [p for p in pm.parameters() if p.requires_grad]
    for sid, bids, w in _session_blocks(4):
        targets = torch.tensor([(i * 7) % 4 for i in range(len(w))])
        la = L.branched_label_logits(hf, TINY_IFACE, bids, w)
        lb = L.separate_label_logits(hf, TINY_IFACE, bids, w)
        assert la.shape == (len(w), 4) and _rel(la, lb) <= REL_TOL, sid
        ca = torch.nn.functional.cross_entropy(la, targets)
        cb = torch.nn.functional.cross_entropy(lb, targets)
        assert abs(float(ca - cb)) <= REL_TOL * abs(float(cb))
        ga = torch.cat([g.reshape(-1) for g in torch.autograd.grad(ca, params)])
        gb = torch.cat([g.reshape(-1) for g in torch.autograd.grad(cb, params)])
        assert _rel(ga, gb) <= 1e-3, sid
        # the readout path (in-place cache, no grad) gives the same logits
        v = Verbaliser(hf, TINY_IFACE).session(bids, w).label_logits
        assert _rel(v, lb.detach()) <= REL_TOL


def test_branch_never_changes_the_main_stream(tiny_config):
    model = make_tiny(tiny_config)
    _, bids, w = _session_blocks(1)[0]
    full = L.branched_label_logits(model, TINY_IFACE, bids, w)
    t = w[0]
    cut = L.branched_label_logits(model, TINY_IFACE, bids[: t + 1], [t])
    assert torch.equal(full[0], cut[0])  # the label after block t never sees block t + 1
    sparse = L.branched_label_logits(model, TINY_IFACE, bids, w[1::2])  # different chunking of the main stream
    assert _rel(sparse, full[1::2].detach()) <= REL_TOL
    with pytest.raises(ValueError):
        L.branched_label_logits(model, TINY_IFACE, bids, [w[1], w[0]])


def test_merge_matches_peft(tiny_config):
    pm = _lora_tiny(tiny_config)
    adapter = L.adapter_state(pm)
    plain = make_tiny(tiny_config)
    L.merge_adapter(plain, adapter)
    merged = pm.merge_and_unload()
    assert L.weights_sha256(plain) == L.weights_sha256(merged)
    assert L.weights_sha256(plain) != L.weights_sha256(make_tiny(tiny_config))
    bad = dict(adapter)
    bad.pop(next(iter(bad)))
    with pytest.raises(ValueError):
        L.merge_adapter(make_tiny(tiny_config), bad)


# ---------------------------------------------------------------- training


def _items(n, seed=0, windows_per=None):
    out = []
    for sid, bids, w in _session_blocks(n, seed):
        w = w if windows_per is None else w[:windows_per]
        out.append(L.Item.build(sid, bids, w, [(len(sid) + p) % 4 for p in w]))
    return out


def _inputs(pool_n=12, es_n=3, es=True):
    return L.TrainInputs(pool=_items(pool_n, 0), es=_items(es_n, 9) if es else None, budget_size=pool_n,
                         provenance={"synthetic": True})


def _run(tmp, loader, cap=1, seed=0, es=True):
    return L.run_training(_inputs(es=es), budget="32" if es else "200w", seed=seed, device="cpu", out=tmp,
                          loader=loader, iface_of=lambda tok: TINY_IFACE, cap=cap, check_count=False)


def test_item_keeps_blocks_up_to_the_last_window():
    it = _items(1)[0]
    _, bids, w = _session_blocks(1)[0]
    assert it.block_ids() == [list(b) for b in bids[: w[-1] + 1]]
    with pytest.raises(ValueError):
        L.Item.build("x", bids, [8, 7], [0, 1])
    with pytest.raises(ValueError):
        L.Item.build("x", bids, [7], [4])  # INSUFFICIENT is never a target


def test_smoke_train_merge_readout(tmp_path, tiny_factory):
    log = _run(tmp_path / "run", tiny_factory)
    assert log["status"] == "ok" and log["steps"] == 2 and log["planned_steps"] == 2 and log["epochs_run"] == 1
    assert log["es_evaluations"] and log["kept_epoch"] == 1 and np.isfinite(log["epochs"][0]["train_loss"])
    assert log["pool_sessions"] == 12 and log["train_windows"] == sum(len(i.windows) for i in _items(12))
    for k in ("protocol", "environment", "adapter_tensors_sha256", "merged"):
        assert k in log
    adapter = L._read_adapter(tmp_path / "run", log)
    assert any(float(v.abs().max()) > 0 for k, v in adapter.items() if k.endswith("lora_B"))  # B moved off zero
    tok, merged, rlog, sha, match = L.load_merged(tmp_path / "run", "cpu", tiny_factory)
    assert match and sha == log["merged"]["sha256"] != L.weights_sha256(tiny_factory()[1])
    tasks = [type("T", (), {"session_id": s["session_id"], "blocks": s["blocks"], "window_pos": s["window_pos"]})
             for s in synthetic_sessions(3, seed=4)]
    with torch.no_grad():
        arr = L.readout_tasks(merged, tok, TINY_IFACE, tasks)
        ref = Verbaliser(merged, TINY_IFACE).session(TINY_IFACE.block_ids(tok, tasks[0].blocks), tasks[0].window_pos)
    assert torch.equal(torch.as_tensor(arr["label_logits"][: len(ref.pos)]), ref.label_logits)
    out = tmp_path / "col.npz"
    write_column(out, label_logits=arr["label_logits"], logsumexp=arr["logsumexp"], session_ids=arr["session_ids"],
                 row_pos=arr["row_pos"], meta={"split": "val", "limit": None, "readout": "verbaliser"})
    with np.load(out) as z:
        assert z["label_logits"].shape == (sum(len(t.window_pos) for t in tasks), 4)
        assert list(z["session_ids"]) == [t.session_id for t in tasks]
    with pytest.raises(FileExistsError):
        _run(tmp_path / "run", tiny_factory)


def test_two_smoke_runs_are_identical(tmp_path, tiny_factory):
    a = _run(tmp_path / "a", tiny_factory, cap=2)
    b = _run(tmp_path / "b", tiny_factory, cap=2)
    assert a["adapter_tensors_sha256"] == b["adapter_tensors_sha256"]
    assert a["merged"]["sha256"] == b["merged"]["sha256"]
    assert [e["train_loss"] for e in a["epochs"]] == [e["train_loss"] for e in b["epochs"]]
    assert [e["es_macro_f1"] for e in a["es_evaluations"]] == [e["es_macro_f1"] for e in b["es_evaluations"]]
    c = _run(tmp_path / "c", tiny_factory, cap=2, seed=1)
    assert c["adapter_tensors_sha256"] != a["adapter_tensors_sha256"]


def test_200w_has_no_early_stopping(tmp_path, tiny_factory):
    log = _run(tmp_path / "w", tiny_factory, cap=2, es=False)
    assert log["es_evaluations"] == [] and log["kept_epoch"] == 2 and log["best_es_macro_f1"] is None
    assert log["stopped"].startswith("no early stopping")


def test_patience_and_keep_best(tiny_config):
    pm = L.attach_lora(make_tiny(tiny_config), 0, check_count=False)
    scores = iter([0.1, 0.3, 0.3, 0.2, 0.25, 0.9, 0.9])
    kept = {}

    def evaluate(model):
        s = next(scores)
        kept[len(kept) + 1] = L.adapter_state(model)
        return s

    r = L.train_run(pm, _items(4, windows_per=1), TINY_IFACE, seed=0, cap=10, device="cpu", evaluate=evaluate)
    assert [e["epoch"] for e in r.es_evaluations] == [1, 2, 3, 4, 5]  # ceil(10/10) = every epoch
    assert r.kept_epoch == 2 and r.best_es_macro_f1 == 0.3 and r.epochs_run == 5 and r.stopped.startswith("patience")
    assert all(torch.equal(r.adapter[k], kept[2][k]) for k in r.adapter)  # a tie (epoch 3) is not a new best
    assert r.planned_steps == 10 and r.steps == 5  # the schedule ran over the planned steps, ES truncated it


def test_divergence_is_reported(tmp_path, tiny_config):
    pm = L.attach_lora(make_tiny(tiny_config), 0, check_count=False)
    with torch.no_grad():
        next(p for n, p in pm.named_parameters() if "lora_A" in n).fill_(float("nan"))
    with pytest.raises(L.Divergence):
        L.train_run(pm, _items(2, windows_per=1), TINY_IFACE, seed=0, cap=1, device="cpu", evaluate=None)


def test_build_inputs_from_the_export(world):
    """The CLI's data path on synthetic exports: budget order, training windows only, ES set, 200w windows."""
    b = B.load(world["budgets"])
    tok = CharTok()
    inp = L.build_inputs(tok, TINY_IFACE, "128", b, world["dir"])
    from hypaqwen.columns import session_rows

    rows = session_rows(world["dev_labels"])
    want = [s for s in b.sessions("128") if rows[s].train.any()]
    assert [it.session_id for it in inp.pool] == want and inp.budget_size == 128
    it = inp.pool[0]
    r = rows[it.session_id]
    assert it.windows == r.row_pos[r.train].tolist() and it.targets.tolist() == r.label[r.train].tolist()
    vrows = session_rows(world["val_labels"])
    assert [e.session_id for e in inp.es] == [s for s in b.es_val if vrows[s].train.any()]
    w = L.build_inputs(tok, TINY_IFACE, "200w", b, world["dir"])
    assert w.es is None and w.budget_size == 200 and len(w.pool) == 200
    assert [(i.session_id, i.windows[0]) for i in w.pool] == [(x.session_id, x.row_pos) for x in b.disclosed_200w]
    assert L.epoch_cap(w.budget_size) == 31


def test_session_budgets_verify_the_disclosure(world, tmp_path):
    """A BUDGETS.json whose disclosed set (and the ES set built from it) is consistent but not the v0.3 rule is refused."""
    import copy
    import shutil

    doc = copy.deepcopy(world["doc"])
    disclosed = {w["session_id"] for w in doc["disclosed_200w"]}
    spare = next(s for s in sorted((s["session_id"] for s in world["val"]), key=B.es_key) if s not in disclosed)
    vrows = {str(x): i for i, x in enumerate(world["val_labels"]["session_ids"])}
    lab = world["val_labels"]
    pos = int(lab["row_pos"][np.flatnonzero(lab["row_session"] == vrows[spare])[0]])
    doc["disclosed_200w"][0] = {"session_id": spare, "row_pos": pos}  # a wrong but well-formed window
    dset = {w["session_id"] for w in doc["disclosed_200w"]}
    doc["es_val"] = [s for s in sorted((s["session_id"] for s in world["val"]), key=B.es_key)
                     if s not in dset][: B.ES_SESSIONS]
    for k in ("es_val", "disclosed_200w"):
        doc["sha256"][k] = B.canonical_sha256(doc[k])
    d = tmp_path / "data"
    shutil.copytree(world["dir"], d)
    (d / "BUDGETS.json").write_text(json.dumps(doc))
    b = B.load(d / "BUDGETS.json")
    B.check_val_ids(b, [s["session_id"] for s in world["val"]])  # internally consistent: the ES check alone passes
    for budget in ("32", "200w"):
        with pytest.raises(B.BudgetError, match="disclosure rule"):
            L.build_inputs(CharTok(), TINY_IFACE, budget, b, d)


def test_cli_readout_on_a_tiny_run(world, tmp_path, monkeypatch, tiny_factory):
    run = tmp_path / "run"
    _run(run, tiny_factory)
    monkeypatch.setattr(L, "_pinned_loader", tiny_factory)
    monkeypatch.setattr(L, "_frozen_iface", lambda tok: TINY_IFACE)
    col = tmp_path / "col.npz"
    assert L.main(["readout", "--run", str(run), "--split", "val", "--data-dir", str(world["dir"]), "--limit", "3",
                   "--device", "cpu", "--out", str(col)]) == 0
    lab = world["val_labels"]
    with np.load(col) as z:
        meta = json.loads(str(z["meta"]))
        assert list(z["session_ids"]) == [str(s) for s in lab["session_ids"][:3]]
        assert np.array_equal(z["row_pos"], lab["row_pos"][: int((lab["row_session"] < 3).sum())])
    assert meta["readout"] == "verbaliser" and meta["arm"] == "lora-32" and meta["split"] == "val"
    assert meta["lora_run"]["merged_sha256_matches_training"] and meta["limit"] == 3


def test_train_cli_refuses_unregistered_seed():
    with pytest.raises(SystemExit):
        L.main(["train", "--budget", "2048", "--seed", "1", "--out", "/nonexistent", "--device", "cpu"])


# ---------------------------------------------------------------- slow: the pinned checkpoint


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("HYPAQWEN_SLOW"), reason="set HYPAQWEN_SLOW=1 (pinned checkpoint, dev export)")
def test_branched_equals_separate_prompts_real_model():
    """§5 on the real model: 4 dev sessions (task file only, no labels), fp32, every window (or the first
    HYPAQWEN_SLOW_WINDOWS), relative difference <= 1e-4, with random non-zero adapters attached."""
    from hypaqwen import model as qmodel
    from hypaqwen.data import iter_tasks
    from hypaqwen.interface import build_interface

    device = os.environ.get("HYPAQWEN_TEST_DEVICE", "cpu")
    data_dir = Path(os.environ.get("HYPAQWEN_DATA_DIR", str(L.V04_DATA)))
    if not (data_dir / "dev.tasks.jsonl.gz").exists():
        pytest.skip(f"no dev export in {data_dir}")
    n_win = int(os.environ.get("HYPAQWEN_SLOW_WINDOWS", "0")) or None
    tok, base = qmodel.load("cpu", "fp32")
    iface = build_interface(tok)
    pm = L.attach_lora(base, 0)
    with torch.no_grad():
        g = torch.Generator().manual_seed(123)
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.01)
    hf = pm.to(device).eval().get_base_model()
    worst = 0.0
    with torch.no_grad():
        for task in itertools.islice(iter_tasks("dev", data_dir), 4):
            bids = iface.block_ids(tok, task.blocks)
            w = task.window_pos[:n_win]
            a = L.branched_label_logits(hf, iface, bids, w).cpu()
            b = L.separate_label_logits(hf, iface, bids, w).cpu()
            worst = max(worst, _rel(a, b))
            assert _rel(a, b) <= REL_TOL, (task.session_id, _rel(a, b))
    print(json.dumps({"device": device, "worst_relative_difference": worst}))
