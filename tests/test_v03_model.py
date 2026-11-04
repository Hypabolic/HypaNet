"""v0.3 transformer tests (spec v0.3 §2.2, §11 "Model")."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from agent_loop_lab.v03 import model as M
from agent_loop_lab.v03.artifact import (
    MAGIC,
    artifact_sha256,
    deserialize_state_dict,
    format_ranges,
    load_artifact,
    nonzero_blocks,
    save_artifact,
    serialize_state_dict,
    sparsity,
    state_dict_template,
    validate_state_dict,
    write_template,
)
from agent_loop_lab.v03.infer import eoe_logits, eoe_predictions, run_sessions
from agent_loop_lab.v03.model import (
    BLOCK,
    BOS,
    EOE,
    O,
    PARAM_COUNT,
    PARAM_SHAPES,
    R,
    V,
    DoomLoopTransformer,
    predict,
    zero_model,
)

MODEL_PY = Path(M.__file__)


# -- helpers (test only: random weights are allowed here, never in a build) ------


def random_model(seed: int = 0, scale: float = 0.08) -> DoomLoopTransformer:
    """Deterministic non-zero weights from a local generator."""
    g = torch.Generator().manual_seed(seed)
    m = zero_model()
    with torch.no_grad():
        for name, p in m.named_parameters():
            s = 1.0 if name.endswith("rel_bias") else (0.5 if "emb" in name else scale)
            p.copy_(torch.randn(p.shape, generator=g) * s)
    return m


def make_session(event_lengths: list[int], seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BOS followed by events of the given token counts (each ends with EOE)."""
    g = torch.Generator().manual_seed(seed)
    toks, eo, off = [BOS], [0], [0]
    for k, n in enumerate(event_lengths, start=1):
        assert 1 <= n <= O
        for j in range(n):
            toks.append(EOE if j == n - 1 else int(torch.randint(3, V, (1,), generator=g)))
            eo.append(k)
            off.append(j)
    return torch.tensor(toks), torch.tensor(eo), torch.tensor(off)


def lengths_to_cuts(cuts: list[int], tail: list[int]) -> list[int]:
    """Event lengths whose cumulative ends (after BOS) land exactly on ``cuts``."""
    out, pos = [], 1
    for c in cuts:
        out.append(c - pos)
        pos = c
    return out + tail


def naive_logits(model: DoomLoopTransformer, tokens, event_ord, offset) -> np.ndarray:
    """Spec §2.2 written out with explicit loops, in float64."""
    sd = {k: v.detach().double().numpy() for k, v in model.state_dict().items()}
    tok = tokens.tolist()
    eo = event_ord.tolist()
    off = offset.tolist()
    n = len(tok)
    h = np.stack([sd["tok_emb.weight"][tok[i]] + sd["off_emb.weight"][off[i]] for i in range(n)])
    for li in range(M.L):
        p = f"layers.{li}."
        q = h @ sd[p + "attn.q.weight"].T + sd[p + "attn.q.bias"]
        k = h @ sd[p + "attn.k.weight"].T + sd[p + "attn.k.bias"]
        v = h @ sd[p + "attn.v.weight"].T + sd[p + "attn.v.bias"]
        B = sd[p + "attn.rel_bias"]
        ctx = np.zeros((n, M.D))
        for hh in range(M.H):
            sl = slice(hh * M.DH, (hh + 1) * M.DH)
            for i in range(n):
                scores = []
                for j in range(i + 1):
                    c = R if eo[j] == 0 else min(eo[i] - eo[j], R - 1)
                    scores.append(float(np.dot(q[i, sl], k[j, sl])) / 4.0 + B[hh, c])
                s = np.array(scores)
                w = np.exp(s - s.max())
                w /= w.sum()
                ctx[i, sl] = w @ v[: i + 1, sl]
        h = h + ctx @ sd[p + "attn.o.weight"].T + sd[p + "attn.o.bias"]
        hid = np.maximum(h @ sd[p + "mlp.fc1.weight"].T + sd[p + "mlp.fc1.bias"], 0.0)
        h = h + hid @ sd[p + "mlp.fc2.weight"].T + sd[p + "mlp.fc2.bias"]
    return h @ sd["head.weight"].T + sd["head.bias"]


# -- parameters -------------------------------------------------------------------


def test_param_count_names_and_shapes():
    m = zero_model()
    sd = m.state_dict()
    assert PARAM_COUNT == 2_128_932
    assert sum(p.numel() for p in m.parameters()) == PARAM_COUNT
    assert list(sd.keys()) == list(PARAM_SHAPES.keys())
    for k, shape in PARAM_SHAPES.items():
        assert tuple(sd[k].shape) == shape, k
        assert sd[k].dtype == torch.float32
    # spec §2.2 breakdown
    assert PARAM_SHAPES["tok_emb.weight"] == (10240, 128)
    assert PARAM_SHAPES["off_emb.weight"] == (192, 128)
    per_layer = sum(math.prod(s) for k, s in PARAM_SHAPES.items() if k.startswith("layers.0."))
    assert per_layer == 198_280
    attn = sum(math.prod(s) for k, s in PARAM_SHAPES.items() if k.startswith("layers.0.attn.") and "rel_bias" not in k)
    assert attn == 66_048
    assert PARAM_SHAPES["layers.0.attn.rel_bias"] == (8, 65)
    assert sum(math.prod(PARAM_SHAPES[k]) for k in ("head.weight", "head.bias")) == 516
    assert len([k for k in PARAM_SHAPES if k.startswith("layers.")]) == 4 * 13


def test_module_structure_is_exact():
    m = zero_model()
    for li, layer in enumerate(m.layers):
        for name in ("q", "k", "v", "o"):
            lin = getattr(layer.attn, name)
            assert isinstance(lin, nn.Linear) and lin.in_features == 128 and lin.out_features == 128
            assert lin.bias is not None
        assert isinstance(layer.mlp.fc1, nn.Linear) and (layer.mlp.fc1.in_features, layer.mlp.fc1.out_features) == (128, 512)
        assert isinstance(layer.mlp.fc2, nn.Linear) and (layer.mlp.fc2.in_features, layer.mlp.fc2.out_features) == (512, 128)
    assert isinstance(m.head, nn.Linear) and (m.head.in_features, m.head.out_features) == (128, 4)
    banned = (nn.LayerNorm, nn.Dropout, nn.BatchNorm1d, nn.RNNBase, nn.GroupNorm)
    assert not any(isinstance(mod, banned) for mod in m.modules())


def test_zero_model_is_zero_and_draws_no_random_numbers():
    state = torch.get_rng_state()
    m = zero_model()
    DoomLoopTransformer()
    assert torch.equal(torch.get_rng_state(), state)
    for p in m.parameters():
        assert not p.requires_grad
        assert torch.count_nonzero(p) == 0
    assert sparsity(m)["overall"]["nonzero"] == 0


def test_zero_model_logits_zero_class_productive():
    t, e, o = make_session([5, 30, 1, 140, 7])
    with torch.no_grad():
        logits = zero_model()(t, e, o)
    assert logits.shape == (t.shape[0], 4)
    assert torch.count_nonzero(logits) == 0
    assert predict(logits).eq(0).all()


def test_predict_ties_go_to_first_index():
    cases = {
        (0.0, 0.0, 0.0, 0.0): 0,
        (1.0, 3.0, 3.0, 0.0): 1,
        (-1.0, -1.0, 2.0, 2.0): 2,
        (0.0, 5.0, 0.0, 5.0): 1,
        (7.0, 7.0, 7.0, 7.0): 0,
    }
    for row, want in cases.items():
        assert int(predict(torch.tensor(row))) == want
    # Many rows at once (vectorised CPU kernel path).
    g = torch.Generator().manual_seed(3)
    x = torch.randint(0, 3, (5000, 4), generator=g).float()
    got = predict(x).tolist()
    want = [row.index(max(row)) for row in x.tolist()]
    assert got == want


# -- standalone packet module --------------------------------------------------------


def test_model_py_is_standalone():
    tree = ast.parse(MODEL_PY.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import in model.py"
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
    assert roots <= {"__future__", "math", "collections", "torch"}, roots
    assert "functional" not in MODEL_PY.read_text(encoding="utf-8")


def test_model_py_loads_as_top_level_module(tmp_path):
    dst = tmp_path / "model.py"
    dst.write_bytes(MODEL_PY.read_bytes())
    spec = importlib.util.spec_from_file_location("model", dst)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.zero_model()
    assert sum(p.numel() for p in m.parameters()) == 2_128_932


# -- input checks ---------------------------------------------------------------------


def test_input_validation():
    m = zero_model()
    t, e, o = make_session([4, 6])
    with pytest.raises(ValueError):
        m(t[1:], e[1:], o[1:])  # no BOS
    bad = o.clone()
    bad[3] = O
    with pytest.raises(ValueError):
        m(t, e, bad)
    bad = e.clone()
    bad[-1] = 1
    with pytest.raises(ValueError):
        m(t, bad, o)  # decreasing event_ord
    bad = e.clone()
    bad[2] = 0
    with pytest.raises(ValueError):
        m(t, bad, o)  # second BOS-ordinal token
    bad = t.clone()
    bad[2] = V
    with pytest.raises(ValueError):
        m(bad, e, o)
    with pytest.raises(ValueError):
        m(t[None], e[None], o[None])  # forward takes one session


# -- causality (bitwise) -------------------------------------------------------------


CUTS = [8, BLOCK - 1, BLOCK, BLOCK + 1, 2 * BLOCK - 1, 2 * BLOCK, 2 * BLOCK + 1, 3 * BLOCK + 3, 4 * BLOCK + 5]


def test_appending_events_never_changes_earlier_logits_bitwise():
    model = random_model(1)
    lengths = lengths_to_cuts(CUTS, tail=[60, 150, 33, 1, 90])
    t, e, o = make_session(lengths, seed=4)
    with torch.no_grad():
        full = model(t, e, o)
        assert torch.isfinite(full).all()
        assert full.abs().max() > 0.1  # non-trivial weights
        for cut in CUTS:
            pre = model(t[:cut], e[:cut], o[:cut])
            assert torch.equal(pre, full[:cut]), cut
        # token-level prefixes inside an event, too
        for cut in (BLOCK + 40, 3 * BLOCK - 7, t.shape[0] - 1):
            assert torch.equal(model(t[:cut], e[:cut], o[:cut]), full[:cut]), cut


def test_causality_with_autograd_live():
    model = random_model(2)
    for p in model.parameters():
        p.requires_grad_(True)
    lengths = lengths_to_cuts([BLOCK + 1, 2 * BLOCK - 1], tail=[70, 20])
    t, e, o = make_session(lengths, seed=5)
    full = model(t, e, o)
    for cut in (BLOCK + 1, 2 * BLOCK - 1):
        assert torch.equal(model(t[:cut], e[:cut], o[:cut]), full[:cut])
    with torch.no_grad():
        assert torch.allclose(model(t, e, o), full.detach(), atol=1e-5, rtol=0)


# -- event-distance bias, BOS column, clamp ---------------------------------------------


def _bias_only_model(col_values: dict[int, float]) -> DoomLoopTransformer:
    m = zero_model()
    with torch.no_grad():
        for c, val in col_values.items():
            m.layers[0].attn.rel_bias[0, c] = val
    return m


def test_bos_column_used_for_key_zero():
    # Only the BOS column is non-zero: every query sends ~all mass to BOS.
    t, e, o = make_session([3, 5, 2])
    acts = _bias_only_model({R: 30.0}).activations(t, e, o)
    a = acts["attn"][0][0]  # head 0, [n, n]
    assert torch.allclose(a[:, 0], torch.ones(t.shape[0]), atol=1e-6)
    # Bucket 1 high instead: an event-1 query has no key at event distance 1,
    # because BOS is in its own column, so its attention stays uniform.
    acts = _bias_only_model({1: 30.0}).activations(t, e, o)
    a = acts["attn"][0][0]
    i = 3  # EOE of event 1 (positions 1..3)
    assert torch.allclose(a[i, : i + 1], torch.full((i + 1,), 1.0 / (i + 1)), atol=1e-6)
    # ...while an event-2 query concentrates on event 1's tokens.
    i = 8  # EOE of event 2 (positions 4..8)
    assert torch.allclose(a[i, 1:4].sum(), torch.tensor(1.0), atol=1e-6)
    # Same-event bucket 0 never covers BOS either.
    acts = _bias_only_model({0: 30.0}).activations(t, e, o)
    a = acts["attn"][0][0]
    assert float(a[3, 0]) < 1e-9 and torch.allclose(a[3, 1:4], torch.full((3,), 1 / 3), atol=1e-6)
    assert float(a[0, 0]) == 1.0  # BOS attends to itself


def test_event_distance_clamps_at_r_minus_one():
    n_events = 75
    t, e, o = make_session([2] * n_events)
    acts = _bias_only_model({R - 1: 30.0}).activations(t, e, o)
    a = acts["attn"][0][0]
    eo = e.tolist()
    i = len(eo) - 1  # EOE of the last event, event_ord 75
    far = [j for j in range(i + 1) if eo[j] != 0 and eo[i] - eo[j] >= R - 1]
    assert len(far) == 2 * (n_events - (R - 1))  # events 1..12, two tokens each
    assert torch.allclose(a[i, far], torch.full((len(far),), 1.0 / len(far)), atol=1e-6)
    near = [j for j in range(i + 1) if j not in far]
    assert float(a[i, near].sum()) < 1e-9
    # A query at event distance exactly R-2 from the farthest key sees no bucket R-1 key.
    i2 = eo.index(R - 1) + 1  # EOE of event R-1; farthest key event 1 at distance R-2
    assert torch.allclose(a[i2, : i2 + 1], torch.full((i2 + 1,), 1.0 / (i2 + 1)), atol=1e-6)


def test_bias_changes_logits_only_through_attention():
    # q = k = 0, so scores are pure bias; v carries the token embedding into the head.
    t, e, o = make_session([2] * 4)
    m = _bias_only_model({R: 20.0})
    with torch.no_grad():
        m.tok_emb.weight[BOS, 0] = 1.0
        m.layers[0].attn.v.weight[0, 0] = 1.0
        m.layers[0].attn.o.weight[1, 0] = 1.0
        m.head.weight[3, 1] = 5.0
        logits_bos = m(t, e, o)
        m.layers[0].attn.rel_bias.zero_()
        logits_flat = m(t, e, o)
    assert predict(logits_bos).eq(3).all()  # mass on BOS -> class 3 everywhere
    assert float(logits_flat[-1, 3]) < float(logits_bos[-1, 3]) / 2


# -- reference implementation ----------------------------------------------------------


@pytest.mark.parametrize("lengths", [[5, 9, 1, 12], [40, 70, 3, 30, 25]])
def test_forward_matches_naive_reference(lengths):
    model = random_model(7, scale=0.04)
    t, e, o = make_session(lengths, seed=8)
    with torch.no_grad():
        got = model(t, e, o).double().numpy()
    want = naive_logits(model, t, e, o)
    assert np.abs(want).max() > 0.1
    np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)


def test_activations_consistent_with_forward():
    model = random_model(9)
    t, e, o = make_session([10, 20, 15])
    with torch.no_grad():
        acts = model.activations(t, e, o, query_positions=[0, 30, t.shape[0] - 1])
        logits = model(t, e, o)
    assert torch.equal(acts["logits"], logits)
    assert len(acts["attn"]) == 4 and acts["attn"][0].shape == (8, 3, t.shape[0])
    assert len(acts["resid_attn"]) == 4 and acts["resid_mlp"][3].shape == (t.shape[0], 128)
    for a in acts["attn"]:
        assert torch.allclose(a.sum(-1), torch.ones(8, 3), atol=1e-5)
        assert float(a[:, 1, 31:].abs().sum()) == 0.0  # causal
    head = acts["resid_mlp"][3] @ model.head.weight.T + model.head.bias
    assert torch.allclose(head, logits, atol=1e-5)


# -- batching ----------------------------------------------------------------------------


def _pad_batch(sessions):
    n = max(s[0].shape[0] for s in sessions)
    B = len(sessions)
    tok = torch.full((B, n), 5, dtype=torch.long)  # junk in padded slots on purpose
    eo = torch.full((B, n), 999, dtype=torch.long)
    off = torch.full((B, n), 7, dtype=torch.long)
    mask = torch.zeros(B, n, dtype=torch.bool)
    for i, (t, e, o) in enumerate(sessions):
        k = t.shape[0]
        tok[i, :k], eo[i, :k], off[i, :k], mask[i, :k] = t, e, o, True
    return tok, eo, off, mask


def test_batched_forward_equals_per_session():
    model = random_model(11)
    sessions = [make_session(L, seed=i) for i, L in enumerate([[4, 30], [60, 70, 9], [100, 150, 40, 20], [1]])]
    tok, eo, off, mask = _pad_batch(sessions)
    with torch.no_grad():
        batched = model.forward_batch(tok, eo, off, mask)
        for i, (t, e, o) in enumerate(sessions):
            single = model(t, e, o)
            k = t.shape[0]
            assert torch.allclose(batched[i, :k], single, atol=1e-5, rtol=0), i
            assert torch.count_nonzero(batched[i, k:]) == 0


def test_forward_batch_is_autograd_clean():
    model = random_model(12)
    for p in model.parameters():
        p.requires_grad_(True)
    sessions = [make_session(L, seed=i) for i, L in enumerate([[20, 150], [3, 4]])]
    tok, eo, off, mask = _pad_batch(sessions)
    model.forward_batch(tok, eo, off, mask).sum().backward()
    assert model.layers[0].attn.rel_bias.grad is not None
    assert torch.isfinite(model.layers[0].attn.q.weight.grad).all()


def test_forward_batch_rejects_non_prefix_mask():
    t, e, o = make_session([3])
    mask = torch.tensor([[True, False, True, True, True]])
    with pytest.raises(ValueError):
        zero_model().forward_batch(t[None], e[None], o[None], mask[:, : t.shape[0]])


# -- inference helpers --------------------------------------------------------------------


class _Enc:
    def __init__(self, t, e, o):
        self.tokens, self.event_ord, self.offset = t.tolist(), e.tolist(), o.tolist()
        self.eoe_index = [i for i, x in enumerate(self.tokens) if x == EOE]
        self.surface = [str(x) for x in self.tokens]


def test_infer_reads_eoe_positions():
    model = random_model(13)
    encs = [_Enc(*make_session(L, seed=i)) for i, L in enumerate([[5, 9], [130, 2, 3], [7]])]
    single = run_sessions(model, encs)
    batched = run_sessions(model, encs, batch_size=2)
    for enc, s, b in zip(encs, single, batched):
        assert s.shape == (len(enc.eoe_index), 4)
        with torch.no_grad():
            full = model(torch.tensor(enc.tokens), torch.tensor(enc.event_ord), torch.tensor(enc.offset))
        assert torch.equal(s, full[enc.eoe_index])
        assert torch.allclose(s, b, atol=1e-5, rtol=0)
        assert eoe_predictions(model, enc) == predict(s).tolist()
    assert torch.equal(eoe_logits(model, encs[0]), single[0])


# -- artefacts -----------------------------------------------------------------------------


def test_artifact_roundtrip_and_byte_identity(tmp_path):
    model = random_model(14)
    h1 = save_artifact(model, tmp_path / "a.dlw", {"phase": "test"})
    h2 = save_artifact(random_model(14), tmp_path / "b.dlw", {"phase": "test"})
    assert h1 == h2 == artifact_sha256(tmp_path / "a.dlw")
    assert (tmp_path / "a.dlw").read_bytes()[:4] == MAGIC
    loaded, meta = load_artifact(tmp_path / "a.dlw")
    assert meta == {"phase": "test"}
    for k, v in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v)
    assert hashlib.sha256((tmp_path / "a.dlw").read_bytes()).hexdigest() == h1


def test_artifact_validation_rejects_bad_input():
    sd = zero_model().state_dict()
    data = serialize_state_dict(sd)
    with pytest.raises(ValueError):
        deserialize_state_dict(b"DLW1" + data[4:])  # v0.2 magic
    with pytest.raises(ValueError):
        deserialize_state_dict(data[:-4])
    with pytest.raises(ValueError):
        deserialize_state_dict(data + b"\0\0\0\0")
    bad = dict(sd)
    bad["head.bias"] = torch.zeros(5)
    with pytest.raises(ValueError):
        validate_state_dict(bad)
    bad = dict(sd)
    bad["head.bias"] = torch.zeros(4, dtype=torch.float64)
    with pytest.raises(ValueError):
        validate_state_dict(bad)
    bad = dict(sd)
    bad["head.bias"] = torch.tensor([0.0, float("nan"), 0.0, 0.0])
    with pytest.raises(ValueError):
        validate_state_dict(bad)
    bad = {k: v for k, v in sd.items() if k != "head.bias"}
    with pytest.raises(ValueError):
        validate_state_dict(bad)
    bad = dict(sd)
    bad["extra"] = torch.zeros(1)
    with pytest.raises(ValueError):
        validate_state_dict(bad)


def test_template(tmp_path):
    tmpl = state_dict_template()
    assert list(tmpl) == list(PARAM_SHAPES) and sum(math.prod(s) for s in tmpl.values()) == PARAM_COUNT
    write_template(tmp_path / "t.json")
    import json

    doc = json.loads((tmp_path / "t.json").read_text())
    assert doc["param_count"] == PARAM_COUNT and [p[0] for p in doc["params"]] == list(PARAM_SHAPES)


def test_nonzero_blocks():
    t = torch.zeros(8, 10)
    t[1:3, 2:5] = 1.0
    t[3, 2:5] = 2.0
    t[5, [0, 9]] = -1.0
    blocks = nonzero_blocks(t)
    assert blocks == [
        {"rows": (1, 3), "cols": [(2, 4)], "nonzero": 9},
        {"rows": (5, 5), "cols": [(0, 0), (9, 9)], "nonzero": 2},
    ]
    assert format_ranges(blocks[1]["cols"]) == "0,9"
    assert nonzero_blocks(torch.tensor([0.0, 1.0, 1.0, 0.0, 3.0])) == [{"rows": None, "cols": [(1, 2), (4, 4)], "nonzero": 3}]
    assert nonzero_blocks(torch.zeros(3, 3)) == []


def test_constants_match_tokenizer_and_encoded_session_runs():
    tok = pytest.importorskip("agent_loop_lab.v03.tokenizer")
    assert (tok.V, tok.O, tok.PAD, tok.BOS, tok.EOE) == (M.V, M.O, M.PAD, M.BOS, M.EOE)
    enc = tok.encode_session(["00:00:01.000 read_file path=a.py -> ok in=10 out=200", "00:00:02.500 run_tests suite=unit -> FAILED 3/18 passed in=? out=7"])
    logits = eoe_logits(random_model(15), enc)
    assert logits.shape == (2, 4) and torch.isfinite(logits).all()
