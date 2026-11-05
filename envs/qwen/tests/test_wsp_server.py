"""WSP model server (docs/V04_WSP_KERNEL.md §2-§4): edit validation, receipts, hash, journal, meter, forward, socket.

Needs the pinned Qwen checkpoint. Device: HYPAQWEN_TEST_DEVICE (default mps when available). Model
runs use two short synthetic sessions; no dataset is read.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import socket
import struct
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from hypaqwen import model as qmodel
from hypaqwen.interface import build_interface
from hypaqwen.readout import Verbaliser
from hypaqwen.wsp import server as S
from hypaqwen.wsp.author import client as C
from hypaqwen.wsp.author import fwd
from hypaqwen.wsp.author import protocol as P

DEVICE = os.environ.get("HYPAQWEN_TEST_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
L = "model.language_model.layers."
DOWN5, DOWN20, NORM3 = L + "5.mlp.down_proj.weight", L + "20.mlp.down_proj.weight", L + "3.input_layernorm.weight"
S1 = {"id": "s1", "blocks": ["00:01 bash ls -> ok 12/40", "00:02 edit app.py -> ok 30/8", "00:03 bash pytest -> fail 80/20"],
      "windows": [1, 2]}
S2 = {"id": "s2", "blocks": ["00:01 bash make -> fail 10/5"], "windows": [0]}


@pytest.fixture(scope="module")
def loaded():
    tok, model = qmodel.load(DEVICE, "fp32")
    return tok, model


@pytest.fixture(scope="module")
def baseline(loaded):
    """S1 read by a plain Verbaliser before any edit."""
    tok, model = loaded
    iface = build_interface(tok)
    return Verbaliser(model, iface, hidden_layers=(12,)).session(iface.block_ids(tok, S1["blocks"]), S1["windows"])


@pytest.fixture
def make(loaded, tmp_path):
    made = []

    def _make(**kw):
        kw.setdefault("log_dir", tmp_path / "logs")
        s = S.ModelServer(loaded[1], loaded[0], phase="T", **kw)
        made.append(s)
        return s

    yield _make
    for s in made:
        s.release()


@pytest.fixture
def srv(make):
    return make()


def _rank1_delta(shape, rows, cols, scale=0.01):
    a, b = torch.zeros(shape[0]), torch.zeros(shape[1])
    a[list(rows)] = scale * torch.arange(1, len(rows) + 1, dtype=torch.float32)
    b[list(cols)] = torch.linspace(0.5, 1.5, len(cols)) * torch.tensor([(-1.0) ** i for i in range(len(cols))])
    return torch.outer(a, b)


def _fwd(srv, model, sessions=(S1,), **ret):
    return srv.forward({"model": model, "sessions": list(sessions), "return": ret})


# ---------------------------------------------------------------- scope and validation (§2)


def test_names_shapes_arch(srv):
    assert len(srv.names) == 320 and all(n.startswith("model.language_model.") for n in srv.names)
    assert srv.shapes()[S.EMBED] == [248320, 1024] and srv.shapes()[DOWN5] == [1024, 3584]
    a = srv.arch()
    assert a["num_hidden_layers"] == 24 and len(a["label_ids"]) == 4 and a["layer_types"][3] == "full_attention"
    assert srv.edit_hash() == S.EMPTY_EDIT_HASH and srv.budget() == {"tokens_used": 0, "tokens_left": 2_000_000,
                                                                    "requests": 0}


@pytest.mark.parametrize("name", ["model.language_model.norm.weight", "mtp.fc.weight", "mtp.layers.0.mlp.up_proj.weight",
                                  "model.visual.blocks.0.attn.qkv.weight", "lm_head.weight", "model.layers.5.mlp.down_proj.weight",
                                  "model.language_model.layers.99.mlp.up_proj.weight", 7])
def test_refuses_names_out_of_scope(srv, name):
    with pytest.raises(S.EditRefused):
        srv.save_edit(name, torch.zeros(1024))
    assert srv.edited() == [] and not srv.journal_dir.exists()


def test_refuses_bad_tensors(srv):
    W = srv.load(DOWN5)
    bad = [W[:, :-1].clone(), W.double(), W.numpy()]
    for v in (float("nan"), float("inf")):
        x = W.clone()
        x[3, 4] = v
        bad.append(x)
    for t in bad:
        with pytest.raises(S.EditRefused):
            srv.save_edit(DOWN5, t)
    assert srv.edit_hash() == S.EMPTY_EDIT_HASH


@pytest.mark.parametrize("c", [2.0, 0.5, 0.0, -1.0])
def test_refuses_scalar_multiples(srv, c):
    for name in (DOWN5, NORM3):
        with pytest.raises(S.EditRefused, match="scalar multiple"):
            srv.save_edit(name, c * srv.load_original(name))
    assert srv.edited() == []


def _near(W, c, eps=1e-3):
    new = c * W
    new.view(-1)[0] += eps
    return new


def _keep_first(W, c):
    new = c * W
    new[0] = W[0]
    return new


@pytest.mark.parametrize("make_new", [
    lambda W: _near(W, 3.0), lambda W: _near(W, 0.0), lambda W: _near(W, -1.0), lambda W: _near(W, 1.01),
    lambda W: _keep_first(W, 2.0), lambda W: torch.zeros_like(W), lambda W: 0.5 * W + 0.01 * W.std() * torch.randn(
        W.shape, generator=torch.Generator().manual_seed(0)),
])
def test_scalar_multiple_rule_refuses_near_multiples(make_new):
    W = 0.03 * torch.randn(64, 1024, generator=torch.Generator().manual_seed(1))
    assert S.scalar_multiple(make_new(W), W) is not None
    rows = W[:4].clone()  # the label-row block: also refuses a common rescaling of 2 or more of the 4 rows
    assert S.label_rows_rescaled(make_new(rows), rows) is not None


def test_scalar_multiple_rule_accepts_structural_edits():
    g = torch.Generator().manual_seed(2)
    W = torch.randn(1024, 2048, generator=g)
    zero_head = W.clone()
    zero_head[:, 256:512] = 0  # one of eight heads' columns: c = 7/8 but the change is not along W
    assert S.scalar_multiple(zero_head, W) is None
    assert S.scalar_multiple(W + torch.outer(torch.randn(1024, generator=g), torch.randn(2048, generator=g)), W) is None
    half = W.clone()
    half[:512] *= 3.0  # rescales half the rows: c = 2, but the rest is as large as the rescaling
    assert S.scalar_multiple(half, W) is None
    rows = 0.03 * torch.randn(4, 1024, generator=g)
    one_row = rows.clone()
    one_row[2] *= 1.5
    assert S.label_rows_rescaled(one_row, rows) is None
    assert S.label_rows_rescaled(rows + 0.01 * torch.randn(4, 1024, generator=g), rows) is None
    assert S.label_rows_rescaled(1.5 * rows + torch.randn(1024, generator=g), rows) is not None  # shared offset
    assert S.label_rows_rescaled(_near(rows, 3.0), rows) is not None


@pytest.mark.parametrize("make_new", [lambda W: _near(W, 3.0), lambda W: _near(W, 0.0), lambda W: _near(W, -1.0),
                                      lambda W: _keep_first(W, 2.0)])
def test_refuses_near_scalar_multiples(srv, make_new):
    for name in (DOWN5, NORM3):
        with pytest.raises(S.EditRefused, match="scalar multiple"):
            srv.save_edit(name, make_new(srv.load_original(name)))
    rows = srv.load_label_rows()
    for bad in (_near(rows, 3.0), 2.0 * rows + rows.mean(0, keepdim=True), 0.5 * rows + 0.1):
        with pytest.raises(S.EditRefused, match="scalar multiple"):
            srv.save_label_rows(bad)
    assert srv.edited() == []
    name, Z = L + "3.self_attn.o_proj.weight", srv.load_original(L + "3.self_attn.o_proj.weight")
    Z[:, 256:512] = 0
    assert srv.save_edit(name, Z).changed_cols == 256  # zeroing one head is not a rescaling


def test_embed_accepts_label_rows_only(srv):
    E = srv.load_original(S.EMBED)
    ids = srv.iface.label_ids
    other = next(i for i in range(10) if i not in ids)
    bad = E.clone()
    bad[other] += 0.01
    bad[ids[0]] += 0.01
    with pytest.raises(S.EditRefused, match="label rows"):
        srv.save_edit(S.EMBED, bad)
    ok = E.clone()
    ok[ids[1], :8] += 0.02
    r = srv.save_edit(S.EMBED, ok)
    assert r.label_rows and (r.changed_entries, r.changed_rows, r.changed_cols, r.delta_rank) == (8, 1, 8, 1)
    assert srv.edited() == [S.EMBED]
    assert torch.equal(srv.load_label_rows(), ok[list(ids)]) and torch.equal(srv.load_label_rows(True), E[list(ids)])
    assert torch.equal(srv.load_original(S.EMBED), E) and torch.equal(srv.load(S.EMBED), ok)
    with pytest.raises(S.EditRefused):
        srv.save_label_rows(torch.zeros(3, 1024))
    with pytest.raises(S.EditRefused, match="scalar multiple"):
        srv.save_label_rows(1.5 * E[list(ids)])


# ---------------------------------------------------------------- receipts, hash, journal (§3)


def test_receipt_of_rank1_edit(srv):
    W = srv.load_original(DOWN5)
    new = W + _rank1_delta(W.shape, rows=[2, 50, 900], cols=[0, 7, 100, 2000, 3583])
    r = srv.save_edit(DOWN5, new)
    assert (r.name, r.changed_entries, r.changed_rows, r.changed_cols, r.delta_rank) == (DOWN5, 15, 3, 5, 1)
    d = (new.double() - W.double())
    assert r.rel_frobenius == pytest.approx((d.norm() / W.double().norm()).item(), rel=1e-9)
    assert r.sha256 == hashlib.sha256(new.numpy().tobytes()).hexdigest()
    assert 0 < r.bf16_changed_entries <= 15 and not r.label_rows
    assert r.edit_hash == srv.edit_hash() != S.EMPTY_EDIT_HASH
    assert torch.equal(srv.load(DOWN5), new) and torch.equal(srv.load_original(DOWN5), W)
    two = W + _rank1_delta(W.shape, [1], [1]) + _rank1_delta(W.shape, [5], [9])
    assert srv.save_edit(DOWN5, two).delta_rank == 2


def test_edit_hash_is_canonical(srv):
    A = srv.load_original(DOWN5) + _rank1_delta((1024, 3584), [1], [2])
    B = srv.load_original(NORM3) + 0.01 * torch.eye(1024)[3]
    srv.save_edit(DOWN5, A)
    h_a = srv.edit_hash()
    srv.save_edit(NORM3, B)
    h_ab = srv.edit_hash()
    assert len({S.EMPTY_EDIT_HASH, h_a, h_ab}) == 3
    assert srv.drop_edit(DOWN5) and srv.drop_edit(NORM3) and not srv.drop_edit(NORM3)
    assert srv.edit_hash() == S.EMPTY_EDIT_HASH
    srv.save_edit(NORM3, B)
    srv.save_edit(DOWN5, A)
    assert srv.edit_hash() == h_ab  # order-free
    r = srv.save_edit(DOWN5, srv.load_original(DOWN5))  # equal to the original: the edit goes away
    assert r.changed_entries == 0 and srv.edited() == [NORM3] and srv.edit_hash() != h_a
    assert torch.equal(srv.load(DOWN5), srv.load_original(DOWN5))


def test_journal_replay_restores_state(make, tmp_path):
    s1 = make()
    A = s1.load_original(DOWN5) + _rank1_delta((1024, 3584), [4], [4, 5])
    s1.save_edit(DOWN5, A)
    s1.save_edit(NORM3, s1.load_original(NORM3) + 0.02 * torch.eye(1024)[7])
    rows = s1.load_label_rows()
    rows[3, 0] += 0.1
    s1.save_label_rows(rows)
    s1.drop_edit(NORM3)
    h, names = s1.edit_hash(), s1.edited()
    s1.release()  # the model holds the original weights again; the journal still has the edits
    assert torch.equal(s1.load(DOWN5), s1.load_original(DOWN5))

    s2 = make()
    assert (s2.edit_hash(), s2.edited()) == (h, names) == (h, sorted([DOWN5, S.EMBED]))
    assert torch.equal(s2.load(DOWN5), A) and torch.equal(s2.load_label_rows(), rows)
    assert torch.equal(s2.load(NORM3), s2.load_original(NORM3))
    s2.release()

    entries = [json.loads(x) for x in (s2.journal_dir / "journal.jsonl").read_text().splitlines()]
    assert [e["op"] for e in entries] == ["save", "save", "save", "drop"] and entries[-1]["edit_hash"] == h
    victim = s2.journal_dir / entries[0]["file"]
    shutil.copy(s2.journal_dir / entries[1]["file"], victim)
    with pytest.raises(ValueError, match="sha256"):
        make()


# ---------------------------------------------------------------- meter and ledger


def test_meter_budget_cap_and_failed_requests(make, loaded):
    probe = make()
    cost = probe.session_cost(probe.iface.block_ids(loaded[0], S2["blocks"]), S2["windows"])
    srv = make(budget=cost + cost // 2, request_cap=cost)
    r = _fwd(srv, "edited", [S2])
    assert r.result["tokens"] == cost and srv.budget() == {"tokens_used": cost, "tokens_left": cost // 2, "requests": 1}
    with pytest.raises(S.BudgetRefused, match="phase budget"):
        _fwd(srv, "edited", [S2])
    with pytest.raises(S.BudgetRefused, match="per-request cap"):
        _fwd(make(budget=10 * cost, request_cap=cost, log_dir=srv.ledger_path.parent / "other"), "both", [S2])
    with pytest.raises(S.RequestError):
        srv.forward({"model": "edited", "sessions": [{"id": "x", "blocks": ["a"], "windows": [3]}]})
    assert srv.budget()["tokens_used"] == cost and srv.budget()["requests"] == 1

    lines = [json.loads(x) for x in srv.ledger_path.read_text().splitlines()]
    assert [(e["call_id"], e["status"], e["tokens"], e["cumulative_tokens"]) for e in lines] == \
        [(1, "ok", cost, cost), (2, "refused", 0, cost), (3, "refused", 0, cost)]
    ok = lines[0]
    assert ok["edit_hash"] == S.EMPTY_EDIT_HASH and ok["model"] == "edited" and len(ok["session_sha256"]) == 1
    ids = probe.iface.block_ids(loaded[0], S2["blocks"])
    assert ok["session_sha256"] == [S.session_sha256(ids, S2["windows"])] and ok["windows"] == [S2["windows"]]
    assert ok["block_sha256"] == [[S.block_sha16(b) for b in ids]]
    again = make(budget=cost + cost // 2, request_cap=cost)  # a continuation re-reads the meter
    assert again.budget() == srv.budget() and again._calls == 3


def test_ledger_hashes_what_runs(srv, loaded):
    """The session hash is over the token ids and windows that run, so renaming or padding changes nothing."""
    srv.forward({"model": "edited", "sessions": [S1, {**S2, "id": "other"}]})
    srv.forward({"model": "edited", "sessions": [{**S1, "id": "renamed"}, {**S2, "id": "x"}]})
    with pytest.raises(S.RequestError, match="after the last window"):  # a dummy trailing block is refused
        srv.forward({"model": "edited", "sessions": [{**S1, "blocks": S1["blocks"] + ["x1"]}]})
    srv.forward({"model": "edited", "sessions": [{**S1, "windows": [2]}]})
    a, b, refused, c = [json.loads(x) for x in srv.ledger_path.read_text().splitlines()]
    assert a["session_sha256"] == b["session_sha256"] and refused["status"] == "refused"
    assert c["session_sha256"][0] != a["session_sha256"][0] and c["windows"] == [[2]]
    assert c["block_sha256"][0] == a["block_sha256"][0]  # same blocks, different windows


# ---------------------------------------------------------------- forward


def test_forward_both_with_noop_edit_gives_zero_deltas(srv):
    r = srv.save_edit(DOWN20, srv.load_original(DOWN20).clone())
    assert r.changed_entries == 0 and srv.edited() == []
    s = _fwd(srv, "both").result["sessions"][0]
    assert s["edited"]["label_logits"] == s["original"]["label_logits"]
    assert all(v == 0 for row in s["delta"]["label_logits"] for v in row) and set(s["delta"]["logsumexp"]) == {0.0}
    assert s["delta"]["label_argmax_flips"] == s["delta"]["argmax_flips"] == 0


def test_forward_edited_differs_and_original_matches_fresh_readout(srv, baseline, loaded):
    W = srv.load_original(DOWN20)
    g = torch.Generator().manual_seed(0)
    new = W + torch.outer(torch.randn(1024, generator=g), torch.randn(3584, generator=g)) * 0.05
    srv.save_edit(DOWN20, new)
    r = _fwd(srv, "both", hidden=[12, 24], mlp_act=[3], topk=5)
    s = r.result["sessions"][0]
    e, o = torch.tensor(s["edited"]["label_logits"]), torch.tensor(s["original"]["label_logits"])
    assert (e - o).abs().max() > 1e-2
    assert torch.allclose(o, baseline.label_logits, atol=1e-5, rtol=0)
    assert torch.allclose(r.arrays["original.s0.hidden.L12"], baseline.hidden[12], atol=1e-5, rtol=0)
    assert r.result["tokens"] == 2 * baseline.tokens

    # Activations: shapes, and the pre-norm final residual reproduces the readout.
    assert r.arrays["edited.s0.mlp_act.L3"].shape == (2, 3584) and r.arrays["edited.s0.hidden.L24"].shape == (2, 1024)
    model = loaded[1]
    with torch.no_grad():
        h = model.model.norm(r.arrays["edited.s0.hidden.L24"].to(DEVICE))
        rows = model.get_input_embeddings().weight[list(srv.iface.label_ids)]
        assert torch.allclose((h @ rows.T).cpu(), e, atol=1e-3)
    assert [t[0] for t in s["edited"]["topk"]] == s["edited"]["argmax"] and len(s["edited"]["topk"][0]) == 5

    # The pristine swap is undone: the edit is still live afterwards.
    assert torch.equal(srv.load(DOWN20), new)
    again = _fwd(srv, "edited").result["sessions"][0]["edited"]["label_logits"]
    assert torch.allclose(torch.tensor(again), e, atol=1e-5, rtol=0)


def test_label_row_edit_reaches_forward(srv):
    rows = srv.load_label_rows()
    rows[2] += 0.05 * rows[2].sign()
    srv.save_label_rows(rows)
    d = torch.tensor(_fwd(srv, "both", [S2]).result["sessions"][0]["delta"]["label_logits"])
    assert d[:, 2].abs().min() > 1e-3


DISCLOSED = {"V001": S.DisclosedWindow(tuple(S1["blocks"]), 1),
             "V002": S.DisclosedWindow(("00:01 search scope=src query=x -> ok in=10 out=3",
                                        "00:02 plan action=summarise -> ok in=900 out=60",
                                        "00:03 edit file=src/app.py -> fail in=40 out=8"), 2)}


@pytest.mark.parametrize("session", [
    {"id": "copy", "blocks": list(S1["blocks"][:2]), "windows": [1]},  # the window, verbatim
    {"id": "more", "blocks": list(S1["blocks"]), "windows": [1, 2]},  # its prefix is one of the windows
    {"id": "merged", "blocks": ["\n".join(DISCLOSED["V002"].blocks)], "windows": [0]},  # blocks joined into one
    {"id": "case", "blocks": [b.upper().replace(" ", "  ") for b in DISCLOSED["V002"].blocks], "windows": [0, 1, 2]},
    {"id": "inside", "blocks": ["00:00 bash make -> ok 1/1", *DISCLOSED["V002"].blocks, "00:09 bash ls -> ok 2/2"],
     "windows": [4]},
])
def test_forward_refuses_disclosed_text(make, session):
    srv = make(disclosed=DISCLOSED)
    with pytest.raises(S.RequestError, match="disclosed window"):
        srv.forward({"model": "edited", "sessions": [S2, session], "return": {"hidden": [12]}})
    e = json.loads(srv.ledger_path.read_text().splitlines()[-1])
    assert e["status"] == "refused" and e["tokens"] == 0 and srv.budget()["tokens_used"] == 0
    assert e["disclosed_overlap"][1]["window"] in DISCLOSED and e["disclosed_overlap"][1]["score"] >= 0.6
    assert len(e["session_sha256"]) == 2


def test_forward_ledgers_disclosed_overlap_of_unrelated_text(make):
    srv = make(disclosed=DISCLOSED)
    srv.forward({"model": "edited", "sessions": [S2]})
    e = json.loads(srv.ledger_path.read_text().splitlines()[-1])
    assert e["status"] == "ok" and e["disclosed_overlap"][0]["score"] < S.DISCLOSED_OVERLAP_REFUSE
    assert not e["disclosed_overlap"][0]["exact"]


def test_forward_disclosed_limits_distinct_edit_states(make):
    srv = make(disclosed=DISCLOSED, disclosed_max_edits=2)
    W = srv.load_original(DOWN20)
    edits = [W + _rank1_delta(W.shape, [i], [i]) for i in range(3)]
    srv.forward_disclosed(["V001"], "edited")  # the empty edit state is the original model: not counted
    srv.forward_disclosed(["V001"], "original")
    for e in edits[:2]:
        srv.save_edit(DOWN20, e)
        srv.forward_disclosed(["V001"], "both")
    srv.save_edit(DOWN20, edits[2])
    with pytest.raises(S.BudgetRefused, match="distinct edit states"):
        srv.forward_disclosed(["V001", "V002"], "edited")
    srv.forward_disclosed(["V001"], "original")  # the original is not limited
    srv.save_edit(DOWN20, edits[0])
    srv.forward_disclosed(["V002"], "edited")  # an edit state already counted
    lines = [json.loads(x) for x in srv.ledger_path.read_text().splitlines()]
    assert [x["status"] for x in lines] == ["ok", "ok", "ok", "ok", "refused", "ok", "ok"]
    assert lines[4]["tokens"] == 0 and lines[3]["disclosed_edit_hashes"] == 2
    srv.release()
    again = make(disclosed=DISCLOSED, disclosed_max_edits=2)  # a continuation re-reads the count
    again.save_edit(DOWN20, edits[2])
    with pytest.raises(S.BudgetRefused):
        again.forward_disclosed(["V001"], "edited")


def test_forward_disclosed_returns_label_logits_only(make):
    disclosed = {"V001": S.DisclosedWindow(tuple(S1["blocks"]), 1)}
    srv = make(disclosed=disclosed)
    r = srv.forward_disclosed(["V001"], "edited")
    assert r.arrays == {} and r.result["sessions"] == [
        {"id": "V001", "edited": {"label_logits": r.result["sessions"][0]["edited"]["label_logits"],
                                  "label_argmax": r.result["sessions"][0]["edited"]["label_argmax"]}}]
    assert len(r.result["sessions"][0]["edited"]["label_logits"]) == 4
    with pytest.raises(S.RequestError):
        srv.forward_disclosed(["V999"], "edited")
    with pytest.raises(S.RequestError, match="not enabled"):
        make(log_dir=srv.ledger_path.parent / "b").forward_disclosed(["V001"], "edited")
    assert [json.loads(x)["status"] for x in srv.ledger_path.read_text().splitlines()] == ["ok", "refused"]


# ---------------------------------------------------------------- weight-load quota and kernel ledger


def _wire(srv, op, conn=1, **args):
    header, out = srv.handle({"op": op, "args": args}, {}, "kernel", conn)
    return header, out


def test_load_quota_counts_distinct_layer_tensors(make, srv):
    nbytes = {n: int(np.prod(sh)) * 4 for n, sh in srv.shapes().items()}
    quota = nbytes[DOWN5] + nbytes[NORM3]
    s = make(load_quota_bytes=quota, log_dir=srv.ledger_path.parent / "q")
    for op, name in (("load", DOWN5), ("load", DOWN5), ("load_original", DOWN5), ("load", NORM3),
                     ("load", S.EMBED), ("load", S.FINAL_NORM), ("load_label_rows", None)):
        header, out = _wire(s, op, **({"name": name} if name else {}))
        assert header["ok"], header
    header, out = _wire(s, "load", conn=2, name=DOWN20)
    assert not header["ok"] and header["error"]["type"] == "QuotaRefused" and out == {}
    assert s.load_quota() == {"bytes_used": quota, "bytes_left": 0, "bytes_quota": quota, "tensors": 2}
    log = [json.loads(x) for x in s.kernel_ledger_path.read_text().splitlines()]
    assert [(e["op"], e.get("name"), e["status"], e["conn"]) for e in log][-2:] == \
        [("load_label_rows", None, "ok", 1), ("load", DOWN20, "refused", 2)]
    assert log[0]["bytes_out"] == nbytes[DOWN5] and log[0]["quota"] and not log[4]["quota"]
    s.release()
    again = make(load_quota_bytes=quota, log_dir=srv.ledger_path.parent / "q")  # a continuation keeps the quota
    assert again.load_quota()["bytes_used"] == quota and again._kernel_seq == len(log)
    assert _wire(again, "load", name=DOWN5)[0]["ok"] and not _wire(again, "load", name=DOWN20)[0]["ok"]


def test_default_load_quota_refuses_a_whole_model_load(srv):
    layer_bytes = sum(int(np.prod(sh)) * 4 for n, sh in srv.shapes().items() if n not in (S.EMBED, S.FINAL_NORM))
    assert srv.load_quota_bytes == int(S.LOAD_QUOTA_FRACTION * layer_bytes)
    refused = 0
    for name in srv.names:
        try:
            srv.client_load(name)
        except S.QuotaRefused:
            refused += 1
    assert refused > 0 and srv.load_quota()["bytes_used"] <= srv.load_quota_bytes
    assert srv.load_quota()["bytes_used"] < layer_bytes / 2 + max(int(np.prod(s)) * 4 for s in srv.shapes().values())


# ---------------------------------------------------------------- sockets: author client and ./fwd


def _hostile(path, payload: bytes) -> None:
    s = P.connect(path, timeout=10)
    try:
        with contextlib.suppress(OSError):  # the server may drop us before reading it all
            s.sendall(payload)
            s.shutdown(socket.SHUT_WR)
            while s.recv(1 << 16):
                pass
    finally:
        s.close()


def _frame(header) -> bytes:
    head = header if isinstance(header, bytes) else json.dumps(header).encode()
    return P.MAGIC + struct.pack("<Q", len(head)) + head


def test_socket_round_trip(make, tmp_path, monkeypatch):
    srv = make()
    sock_dir = Path(tempfile.mkdtemp(prefix="hnk"))  # short: sun_path is 104 bytes on macOS
    kpath, fpath = sock_dir / "kernel.sock", sock_dir / "fwd.sock"
    stop, ready = threading.Event(), threading.Event()
    t = threading.Thread(target=srv.serve, args=(kpath, fpath), kwargs={"stop": stop, "on_ready": ready.set},
                         daemon=True)
    t.start()
    try:
        assert ready.wait(10)
        # Hostile frames cost their own connection only; the server keeps serving.
        for bad in ({"op": "load", "tensors": [{"key": "t", "shape": [2 ** 62, 4]}]},
                    {"op": "load", "tensors": [{"key": "t", "shape": [2 ** 32] * 3}]},
                    {"op": "load", "tensors": [{"key": "t", "shape": [10 ** 30]}]},
                    {"op": "load", "tensors": [1]},
                    b"[" * 200_000 + b"]" * 200_000,
                    {"op": "save_edit", "args": {"name": DOWN5}, "tensors": [{"key": "tensor", "shape": [1 << 30]}]}):
            _hostile(kpath, _frame(bad))
        _hostile(fpath, _frame({"op": "forward", "args": {"request": {"model": "x"}}, "tensors": [None]}))
        assert t.is_alive()
        c = C.KernelClient(kpath)
        assert c.names() == srv.names and c.shapes()[DOWN5] == (1024, 3584)
        W = c.load_original(DOWN5)
        assert W.dtype == torch.float32 and torch.equal(W, srv.load_original(DOWN5))
        new = W + _rank1_delta(W.shape, [0, 1], [3])
        r = c.save_edit(DOWN5, new)
        assert isinstance(r, C.EditReceipt) and (r.changed_entries, r.delta_rank) == (2, 1)
        assert torch.equal(c.load(DOWN5), new) and c.edited() == [DOWN5] and c.edit_hash() == srv.edit_hash()
        with pytest.raises(C.EditRefused):
            c.save_edit(S.FINAL_NORM, c.load(S.FINAL_NORM) + 0.1)
        with pytest.raises(C.RequestError, match="not available"):
            c.call("forward", request={"model": "edited", "sessions": [S2]})
        assert c.budget()["requests"] == 0
        c.close()  # one connection at a time: ./fwd waits for this one

        author = tmp_path / "author"
        author.mkdir()
        req = author / "request.json"
        req.write_text(json.dumps({"model": "edited", "sessions": [S2], "return": {"hidden": [24], "mlp_act": [3]}}))
        monkeypatch.setenv(fwd.AUTHOR_ENV, str(author))
        monkeypatch.setenv(fwd.SOCKET_ENV, str(fpath))
        assert fwd.main([str(req)]) == 0
        out = author / "output" / "T" / "fwd" / "1"
        resp = json.loads((out / "response.json").read_text())
        assert resp["edit_hash"] == srv.edit_hash() and resp["budget"]["requests"] == 1
        assert resp["request_file"] == "request.json"
        assert json.loads(srv.ledger_path.read_text().splitlines()[-1])["request_path"] == "request.json"
        assert set(resp["arrays"]) == {"edited.s0.hidden.L24", "edited.s0.mlp_act.L3"}
        h = np.load(out / "edited.s0.hidden.L24.npy")
        assert h.shape == (1, 1024) and np.load(out / "edited.s0.mlp_act.L3.npy").shape == (1, 3584)

        # Module-level API and helpers on the same server.
        C.connect(kpath)
        try:
            assert C.NAMES == srv.names and C.LABEL_IDS == srv.iface.label_ids and len(C.CLASS_NAMES) == 4
            lens = C.label_lens(h)
            assert torch.allclose(lens, torch.tensor(resp["sessions"][0]["edited"]["label_logits"]), atol=1e-3)
            name, Z = C.zero_head(3, 1)
            assert name == L + "3.self_attn.o_proj.weight" and Z[:, 256:512].abs().sum() == 0
            assert torch.equal(Z[:, :256], C.load(name)[:, :256])
            U, Sv, Vh = C.svd(DOWN5, 3)
            assert U.shape == (1024, 3) and Vh.shape == (3, 3584) and Sv[0] >= Sv[1]
            idx, sc = C.top_writers(DOWN5, U[:, 0], k=4)
            assert idx.shape == (4,) and sc.abs().max() > 0
            assert C.drop_edit(DOWN5) and C.edit_hash() == S.EMPTY_EDIT_HASH
        finally:
            C._client.close()
            C._client = None
            C._cache.clear()
    finally:
        stop.set()
        t.join(10)
        shutil.rmtree(sock_dir, ignore_errors=True)
    assert not t.is_alive() and not kpath.exists()
    log = [json.loads(x) for x in srv.kernel_ledger_path.read_text().splitlines()]
    loads = [e for e in log if e["op"] == "load_original" and e.get("name") == DOWN5]
    assert loads and loads[0]["status"] == "ok" and loads[0]["bytes_out"] == 1024 * 3584 * 4
    assert isinstance(loads[0]["conn"], int) and any(e["op"] == "forward" and e["status"] == "refused" for e in log)
