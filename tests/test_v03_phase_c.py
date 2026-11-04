"""v0.3 Phase C trainer (src/agent_loop_lab/baselines/phase_c_v03.py; experiments/v0_3/phase_c/README.md).

Synthetic sessions only, except one small dev integration test (``load_dev(limit=...)``).
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
import torch
from torch import nn

from agent_loop_lab.baselines import phase_c_v03 as pc
from agent_loop_lab.v03.model import BOS, EOE, O, V


@pytest.fixture(autouse=True)
def _threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def _session(n_events: int, ev_len: int, seed: int, sid: str = "s") -> pc.TrainSession:
    g = torch.Generator().manual_seed(seed)
    tok, eo, off = [BOS], [0], [0]
    eoe = []
    for e in range(1, n_events + 1):
        body = torch.randint(3, V, (ev_len - 1,), generator=g).tolist()
        tok += body + [EOE]
        eo += [e] * ev_len
        off += list(range(ev_len))
        eoe.append(len(tok) - 1)
    assert max(off) < O
    pos = torch.tensor(eoe[2:], dtype=torch.long)
    y = torch.randint(0, 4, (pos.numel(),), generator=g)
    return pc.TrainSession(sid, torch.tensor(tok), torch.tensor(eo), torch.tensor(off), pos, y)


def _batch_loss(model, batch) -> float:
    total = sum(ts.positions.numel() for ts in batch)
    with torch.no_grad():
        return float(sum(nn.functional.cross_entropy(model(ts.tokens, ts.event_ord, ts.offset)[ts.positions],
                                                     ts.targets, reduction="sum") for ts in batch) / total)


# -- schedule constants -------------------------------------------------------


def test_registered_schedule():
    assert pc.SCHEDULE == ((32, 0), (32, 1), (32, 2), (128, 0), (128, 1), (128, 2), (512, 0), (512, 1), (512, 2),
                           (2048, 0), ("all", 0))
    assert pc.EPOCH_CAPS == {32: 60, 128: 30, 512: 12, 2048: 4, "all": 3}
    assert (pc.LR, pc.BATCH_SESSIONS, pc.GRAD_CLIP, pc.PATIENCE_EVALS, pc.ES_SESSIONS) == (3e-4, 8, 1.0, 3, 500)


@pytest.mark.parametrize("cap, epochs", [
    (60, list(range(6, 61, 6))), (30, list(range(3, 31, 3))), (12, [2, 4, 6, 8, 10, 12]),
    (4, [1, 2, 3, 4]), (3, [1, 2, 3]), (2, [1, 2]), (7, [1, 2, 3, 4, 5, 6, 7]), (25, [3, 6, 9, 12, 15, 18, 21, 24, 25]),
])
def test_es_eval_epochs(cap, epochs):
    got = pc.es_eval_epochs(cap)
    assert got == epochs
    assert got[-1] == cap and len(got) <= pc.MAX_ES_EVALS + 1


def test_budget_subsets_nested_and_keyed():
    ids = [f"s-{i:016x}" for i in range(300)]
    order = pc.budget_order(ids)
    assert order == sorted(ids, key=lambda s: hashlib.sha256(f"20261105:v03-phase-c-budget:{s}".encode()).hexdigest())
    a, b = pc.budget_subset(ids, 32), pc.budget_subset(ids, 128)
    assert b[:32] == a and pc.budget_subset(ids, "all") == order
    with pytest.raises(ValueError):
        pc.budget_subset(ids, 301)
    with pytest.raises(ValueError):
        pc.budget_order(ids + ids[:1])


def test_es_ids_filter_then_take():
    ids = [f"s-{i:016x}" for i in range(60)]
    order = sorted(ids, key=pc.es_key)
    disclosed = set(order[:5]) | {order[20]}
    got = pc.es_session_ids(ids, disclosed, n=30)
    assert len(got) == 30 and not set(got) & disclosed
    assert got == [s for s in order if s not in disclosed][:30]
    with pytest.raises(ValueError):
        pc.es_session_ids(ids, disclosed, n=55)


def test_train_split_is_dev_only():
    for split in ("val", "test", "ood", "counterfactual"):
        with pytest.raises(ValueError, match="only"):
            pc._build_dev(None, split=split)


# -- initialisation -------------------------------------------------------------


def test_init_model_seeded_standard_init():
    a, b, c = pc.init_model(0), pc.init_model(0), pc.init_model(1)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert not torch.equal(sa["layers.0.attn.q.weight"], sc["layers.0.attn.q.weight"])
    for k, v in sa.items():
        if k.endswith("rel_bias"):
            assert torch.count_nonzero(v) == 0
        else:
            assert torch.count_nonzero(v) == v.numel(), k
    assert abs(float(sa["tok_emb.weight"].std()) - 0.02) < 1e-3
    bound = 1 / math.sqrt(128)
    w = sa["layers.0.attn.q.weight"]
    assert float(w.abs().max()) <= bound and float(w.abs().max()) > 0.9 * bound  # kaiming_uniform(a=sqrt(5))
    # The draw order is reset_parameters in modules() order after manual_seed.
    torch.manual_seed(0)
    from agent_loop_lab.v03.model import DoomLoopTransformer

    ref = DoomLoopTransformer()
    with torch.no_grad():
        for m in ref.modules():
            if isinstance(m, nn.Linear):
                nn.Linear.reset_parameters(m)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)
    assert all(torch.equal(ref.state_dict()[k], sa[k]) for k in sa)


# -- autograd and training --------------------------------------------------------


def test_autograd_reaches_attention_weights():
    model = pc.init_model(0)
    ts = _session(6, 30, seed=1)  # 181 tokens: two query blocks, so cross-block keys are used
    logits = model(ts.tokens, ts.event_ord, ts.offset)
    nn.functional.cross_entropy(logits[ts.positions], ts.targets).backward()
    for i in range(4):
        for name in ("q", "k", "v", "o"):
            g = getattr(model.layers[i].attn, name).weight.grad
            assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0, (i, name)
        assert model.layers[i].attn.rel_bias.grad is not None
        assert float(model.layers[i].attn.rel_bias.grad.abs().sum()) > 0
    assert float(model.tok_emb.weight.grad.abs().sum()) > 0


def test_per_session_accumulation_matches_padded_batch():
    batch = [_session(3, 20, seed=2), _session(5, 12, seed=3), _session(4, 25, seed=4)]
    m1, m2 = pc.init_model(3), pc.init_model(3)
    opt = torch.optim.SGD(m1.parameters(), lr=0.0)
    loss_loop, total = pc.batch_step(m1, opt, batch, clip=1e9)
    n = max(ts.tokens.numel() for ts in batch)
    tok = torch.zeros(len(batch), n, dtype=torch.long)
    eo, off = torch.zeros_like(tok), torch.zeros_like(tok)
    mask = torch.zeros(len(batch), n, dtype=torch.bool)
    for i, ts in enumerate(batch):
        k = ts.tokens.numel()
        tok[i, :k], eo[i, :k], off[i, :k], mask[i, :k] = ts.tokens, ts.event_ord, ts.offset, True
    logits = m2.forward_batch(tok, eo, off, mask)
    sel = torch.cat([logits[i][ts.positions] for i, ts in enumerate(batch)])
    y = torch.cat([ts.targets for ts in batch])
    loss = nn.functional.cross_entropy(sel, y)  # pooled mean
    loss.backward()
    assert total == y.numel()
    assert abs(loss_loop - float(loss)) < 1e-5
    for (k, p1), p2 in zip(m1.named_parameters(), m2.parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-6, rtol=1e-4), k


def test_few_steps_reduce_loss():
    batch = [_session(4, 15, seed=10 + i, sid=f"s{i}") for i in range(4)]
    model = pc.init_model(0)
    opt = torch.optim.Adam(model.parameters(), lr=pc.LR)
    before = _batch_loss(model, batch)
    for _ in range(3):
        pc.batch_step(model, opt, batch)
    after = _batch_loss(model, batch)
    assert after < before - 1e-3, (before, after)


def test_train_run_deterministic_early_stop_and_keep_best():
    pool = [_session(3, 10, seed=20 + i, sid=f"s{i}") for i in range(10)]
    scores = iter([0.5, 0.7, 0.6, 0.65, 0.69, 0.99])

    def evaluate(model):
        return next(scores)

    res = pc.train_run(pool, evaluate, seed=0, cap=6, progress=None)
    assert res.stopped.startswith("patience") and res.epochs_run == 5 and res.best_epoch == 2
    assert [e["epoch"] for e in res.es_evaluations] == [1, 2, 3, 4, 5]
    assert res.steps == 5 * 2 and res.pool_sessions == 10
    assert res.train_positions == sum(ts.positions.numel() for ts in pool)
    # Same seed, constant evaluation: identical weights and losses (deterministic).
    r1 = pc.train_run(pool, lambda m: 0.1, seed=4, cap=2)
    r2 = pc.train_run(pool, lambda m: 0.1, seed=4, cap=2)
    assert r1.best_epoch == 1 and r1.stopped == "epoch_cap"
    assert all(torch.equal(r1.state_dict[k], r2.state_dict[k]) for k in r1.state_dict)
    assert [e["train_loss"] for e in r1.epochs] == [e["train_loss"] for e in r2.epochs]


def test_load_dev_small_integration():
    dev = pc.load_dev(limit=3)
    assert len(dev.session_ids) == 3 and dev.inputs["built_now"]
    for s in range(3):
        ts = dev.session(s)
        assert int(ts.tokens[0]) == BOS and ts.tokens.shape == ts.event_ord.shape == ts.offset.shape
        assert ts.positions.numel() == ts.targets.numel()
        if ts.positions.numel():
            assert bool((ts.tokens[ts.positions] == EOE).all())
            assert int(ts.targets.max()) < 4
    pool = pc.train_sessions(dev, dev.session_ids)
    assert all(ts.positions.numel() > 0 for ts in pool)
    np.testing.assert_array_equal(dev.lengths(), [dev.session(s).tokens.numel() for s in range(3)])
