"""Architecture tests (spec v0.2 §2.2, §11 "Architecture", §15 byte-identical artefacts)."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import struct
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from agent_loop_lab.model import (
    HIDDEN1,
    HIDDEN2,
    OUTPUT_DIM,
    PARAM_COUNT,
    PARAM_SHAPES,
    SPARSITY_TOLERANCE,
    DoomLoopMLP,
    artifact_sha256,
    deserialize_state_dict,
    load_artifact,
    load_state_dict_into,
    predict,
    predict_classes,
    save_artifact,
    serialize_state_dict,
    sparsity,
    state_dict_template,
    validate_state_dict,
    zero_model,
)
from agent_loop_lab.schema import CLASS_NAMES, INPUT_DIM, ClassLabel

EXPECTED_SHAPES = OrderedDict(
    [
        ("fc1.weight", (24, 120)),
        ("fc1.bias", (24,)),
        ("fc2.weight", (12, 24)),
        ("fc2.bias", (12,)),
        ("fc3.weight", (4, 12)),
        ("fc3.bias", (4,)),
    ]
)


def _patterned_model() -> DoomLoopMLP:
    """Deterministic, non-trivial weights with mixed signs and some zeros."""
    m = zero_model()
    with torch.no_grad():
        for k, p in m.named_parameters():
            n = p.numel()
            vals = torch.arange(n, dtype=torch.float32).reshape(p.shape)
            p.copy_(torch.where(vals % 3 == 0, torch.zeros_like(vals), (vals - n / 2) / 64.0))
    return m


# -- shapes / slots / initial state ----------------------------------------------


def test_param_shapes_exact():
    assert PARAM_SHAPES == EXPECTED_SHAPES
    assert (HIDDEN1, HIDDEN2, OUTPUT_DIM, INPUT_DIM) == (24, 12, 4, 120)
    m = zero_model()
    assert OrderedDict((k, tuple(v.shape)) for k, v in m.state_dict().items()) == EXPECTED_SHAPES
    assert [k for k, _ in m.named_parameters()] == list(EXPECTED_SHAPES)


def test_exactly_3256_slots():
    assert PARAM_COUNT == 3256
    assert (120 * 24 + 24) + (24 * 12 + 12) + (12 * 4 + 4) == 2904 + 300 + 52 == 3256
    assert sum(p.numel() for p in zero_model().parameters()) == 3256
    assert sum(p.numel() for p in DoomLoopMLP().parameters()) == 3256


def test_architecture_layers_only_linear():
    m = zero_model()
    kinds = [type(mod) for name, mod in m.named_modules() if name]
    assert kinds == [nn.Linear, nn.Linear, nn.Linear]
    assert (m.fc1.in_features, m.fc1.out_features) == (120, 24)
    assert (m.fc2.in_features, m.fc2.out_features) == (24, 12)
    assert (m.fc3.in_features, m.fc3.out_features) == (12, 4)
    assert list(m.buffers()) == []


def test_zero_model_initial_state():
    m = zero_model()
    for k, p in m.named_parameters():
        assert p.dtype == torch.float32, k
        assert p.device.type == "cpu", k
        assert not p.requires_grad, k
        assert p.grad is None, k
        assert torch.isfinite(p).all(), k
        assert torch.count_nonzero(p) == 0, k
        assert not torch.signbit(p).any(), f"{k}: -0.0 present"
    assert zero_model(requires_grad=True).fc1.weight.requires_grad


def test_zero_models_identical():
    a = serialize_state_dict(zero_model().state_dict())
    b = serialize_state_dict(zero_model().state_dict())
    assert a == b


def test_state_dict_template():
    t = state_dict_template()
    assert list(t) == list(EXPECTED_SHAPES)
    assert all(t[k] == list(v) for k, v in EXPECTED_SHAPES.items())


# -- forward / prediction ----------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 3, 1024])
def test_forward_output_shape(batch):
    m = _patterned_model()
    out = m(torch.rand(batch, INPUT_DIM, generator=torch.Generator().manual_seed(0)))
    assert out.shape == (batch, 4) and out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_zero_model_outputs_zero_and_predicts_productive():
    m = zero_model()
    x = torch.rand(5, INPUT_DIM, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        out = m(x)
    assert torch.equal(out, torch.zeros(5, 4))
    assert predict(out).tolist() == [ClassLabel.PRODUCTIVE] * 5
    assert predict_classes(m, x.numpy()).tolist() == [0] * 5


@pytest.mark.parametrize(
    "logits,expected",
    [
        ([0.0, 0.0, 0.0, 0.0], 0),  # all-zero -> PRODUCTIVE
        ([-1.0, -1.0, -1.0, -1.0], 0),
        ([0.0, 3.0, 3.0, 1.0], 1),  # first max on ties
        ([0.0, 0.0, 5.0, 5.0], 2),
        ([2.0, -1.0, 2.0, 2.0], 0),
        ([0.0, 0.0, 0.0, 1e-6], 3),
        ([-5.0, -2.0, -3.0, -2.0], 1),
    ],
)
def test_argmax_tie_rule(logits, expected):
    assert predict(torch.tensor(logits)).item() == expected
    assert predict(torch.tensor([logits, logits])).tolist() == [expected, expected]


def test_argmax_tie_rule_exhaustive_small_grid():
    vals = (-1.0, 0.0, 1.0)
    grid = torch.tensor([[a, b, c, d] for a in vals for b in vals for c in vals for d in vals])
    got = predict(grid).tolist()
    want = [row.index(max(row)) for row in grid.tolist()]  # list.index -> first maximal
    assert got == want


def test_predict_classes_shapes_and_validation():
    m = zero_model()
    assert predict_classes(m, np.zeros(INPUT_DIM, dtype=np.float32)).shape == (1,)
    assert predict_classes(m, np.zeros((4, INPUT_DIM))).shape == (4,)
    assert predict_classes(m, torch.zeros(2, INPUT_DIM)).tolist() == [0, 0]
    with pytest.raises(ValueError):
        predict_classes(m, np.zeros((2, INPUT_DIM - 1)))


def test_no_softmax_raw_logits():
    m = zero_model()
    with torch.no_grad():
        m.fc3.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    out = m(torch.zeros(2, INPUT_DIM))
    assert torch.equal(out, torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 2))  # not normalised
    with torch.no_grad():
        m.fc3.bias.copy_(torch.tensor([-5.0, 10.0, 0.0, 3.0]))
        assert torch.equal(m(torch.zeros(1, INPUT_DIM))[0], torch.tensor([-5.0, 10.0, 0.0, 3.0]))
    src = inspect.getsource(DoomLoopMLP)
    assert "softmax" not in src.lower()
    assert not any(isinstance(mod, (nn.Softmax, nn.LogSoftmax)) for mod in m.modules())


def test_relu_between_layers():
    m = zero_model()
    with torch.no_grad():
        m.fc1.bias[0] = -1.0  # negative pre-activation must be clipped
        m.fc1.bias[1] = 2.0
        m.fc2.weight[0, 0] = 1.0
        m.fc2.weight[0, 1] = 1.0
        m.fc2.bias[1] = -3.0
        m.fc3.weight[0, 0] = 1.0
        m.fc3.weight[1, 1] = 1.0
        out = m(torch.zeros(1, INPUT_DIM))
    assert out[0].tolist() == [2.0, 0.0, 0.0, 0.0]


def test_activations_keys_and_shapes():
    m = _patterned_model()
    x = torch.rand(6, INPUT_DIM, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        acts = m.activations(x)
        out = m(x)
    assert list(acts) == ["h1_pre", "h1", "h2_pre", "h2", "logits"]
    shapes = {k: tuple(v.shape) for k, v in acts.items()}
    assert shapes == {"h1_pre": (6, 24), "h1": (6, 24), "h2_pre": (6, 12), "h2": (6, 12), "logits": (6, 4)}
    assert torch.equal(acts["h1"], torch.relu(acts["h1_pre"]))
    assert torch.equal(acts["h2"], torch.relu(acts["h2_pre"]))
    assert torch.equal(acts["logits"], out)


# -- validate_state_dict -------------------------------------------------------------


def test_validate_state_dict_accepts_model_state():
    validate_state_dict(zero_model().state_dict())
    validate_state_dict(_patterned_model().state_dict())


def _sd() -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((k, v.clone()) for k, v in zero_model().state_dict().items())


def test_validate_state_dict_missing_key():
    sd = _sd()
    del sd["fc2.bias"]
    with pytest.raises(ValueError, match="keys"):
        validate_state_dict(sd)


def test_validate_state_dict_extra_key():
    sd = _sd()
    sd["fc4.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="keys"):
        validate_state_dict(sd)


def test_validate_state_dict_wrong_order():
    sd = _sd()
    reordered = OrderedDict(reversed(list(sd.items())))
    with pytest.raises(ValueError, match="keys"):
        validate_state_dict(reordered)


def test_validate_state_dict_wrong_shape():
    sd = _sd()
    sd["fc1.weight"] = torch.zeros(120, 24)
    with pytest.raises(ValueError, match="shape"):
        validate_state_dict(sd)


def test_validate_state_dict_wrong_dtype():
    sd = _sd()
    sd["fc3.bias"] = torch.zeros(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="dtype"):
        validate_state_dict(sd)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_validate_state_dict_non_finite(bad):
    sd = _sd()
    sd["fc2.weight"][3, 4] = bad
    with pytest.raises(ValueError, match="non-finite"):
        validate_state_dict(sd)
    with pytest.raises(ValueError):
        serialize_state_dict(sd)


def test_load_state_dict_into_copies():
    src = _patterned_model().state_dict()
    m = load_state_dict_into(zero_model(), src)
    for k in PARAM_SHAPES:
        assert torch.equal(m.state_dict()[k], src[k])
    with torch.no_grad():
        src["fc1.bias"][0] += 100.0
    assert m.state_dict()["fc1.bias"][0] != src["fc1.bias"][0]


# -- .dlw serialisation ----------------------------------------------------------------


def test_dlw_round_trip_exact():
    m = _patterned_model()
    data = serialize_state_dict(m.state_dict())
    sd, meta = deserialize_state_dict(data)
    assert meta == {}
    assert list(sd) == list(PARAM_SHAPES)
    for k in PARAM_SHAPES:
        assert sd[k].dtype == torch.float32 and tuple(sd[k].shape) == PARAM_SHAPES[k]
        assert torch.equal(sd[k], m.state_dict()[k]), k
        assert sd[k].numpy().tobytes() == m.state_dict()[k].numpy().tobytes()  # incl. -0.0 / bit patterns


def test_dlw_layout():
    m = _patterned_model()
    data = serialize_state_dict(m.state_dict(), {"phase": "a-strict"})
    assert data[:4] == b"DLW1"
    (hlen,) = struct.unpack("<I", data[4:8])
    header = json.loads(data[8 : 8 + hlen])
    assert header["format"] == "DLW1"
    assert header["tensors"] == [[k, list(v)] for k, v in PARAM_SHAPES.items()]
    assert header["metadata"] == {"phase": "a-strict"}
    assert len(data) == 8 + hlen + 4 * PARAM_COUNT
    payload = np.frombuffer(data[8 + hlen :], dtype="<f4")
    flat = np.concatenate([m.state_dict()[k].numpy().reshape(-1) for k in PARAM_SHAPES])
    assert payload.tobytes() == flat.astype("<f4").tobytes()


def test_dlw_byte_identical_reserialisation(tmp_path):
    m = _patterned_model()
    meta = {"phase": "a-selfcheck", "iteration": 3, "notes": "ünïcode ok", "nested": {"b": [1, 2], "a": None}}
    p1, p2 = tmp_path / "a.dlw", tmp_path / "b.dlw"
    sha1 = save_artifact(m, p1, meta)
    sha2 = save_artifact(_patterned_model(), p2, meta)
    assert p1.read_bytes() == p2.read_bytes()
    assert sha1 == sha2 == artifact_sha256(p1) == hashlib.sha256(p1.read_bytes()).hexdigest()
    loaded, meta_back = load_artifact(p1)
    assert meta_back == meta
    assert serialize_state_dict(loaded.state_dict(), meta_back) == p1.read_bytes()
    assert isinstance(loaded, DoomLoopMLP)
    assert all(not p.requires_grad for p in loaded.parameters())
    x = torch.rand(4, INPUT_DIM, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        assert torch.equal(loaded(x), m(x))


def test_dlw_metadata_key_order_irrelevant():
    sd = _patterned_model().state_dict()
    a = serialize_state_dict(sd, {"x": 1, "y": 2})
    b = serialize_state_dict(sd, OrderedDict([("y", 2), ("x", 1)]))
    assert a == b
    assert serialize_state_dict(sd, {"x": 1}) != serialize_state_dict(sd, {"x": 2})
    assert serialize_state_dict(sd) == serialize_state_dict(sd, {})


def test_dlw_different_weights_different_bytes():
    sd = _sd()
    base = serialize_state_dict(sd)
    sd["fc3.bias"][2] = 1.0
    assert serialize_state_dict(sd) != base


def test_dlw_rejects_bad_magic():
    data = bytearray(serialize_state_dict(_sd()))
    data[0:4] = b"DLW2"
    with pytest.raises(ValueError, match="not a DLW1"):
        deserialize_state_dict(bytes(data))


def test_dlw_rejects_trailing_bytes():
    data = serialize_state_dict(_sd())
    with pytest.raises(ValueError, match="trailing"):
        deserialize_state_dict(data + b"\x00")
    with pytest.raises(ValueError, match="trailing"):
        deserialize_state_dict(data + b"\x00" * 4)


def test_dlw_rejects_truncated_payload():
    data = serialize_state_dict(_sd())
    with pytest.raises(ValueError):
        deserialize_state_dict(data[:-1])
    with pytest.raises(ValueError):
        deserialize_state_dict(data[:-4])


def test_dlw_rejects_nan_payload():
    data = serialize_state_dict(_sd())
    corrupt = data[:-4] + struct.pack("<f", math.nan)
    with pytest.raises(ValueError, match="non-finite"):
        deserialize_state_dict(corrupt)


def test_dlw_rejects_tampered_header():
    data = serialize_state_dict(_sd())
    (hlen,) = struct.unpack("<I", data[4:8])
    header = json.loads(data[8 : 8 + hlen])
    header["tensors"][0], header["tensors"][1] = header["tensors"][1], header["tensors"][0]
    hb = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError):
        deserialize_state_dict(data[:4] + struct.pack("<I", len(hb)) + hb + data[8 + hlen :])


@pytest.mark.parametrize("data", [b"", b"DLW", b"DLW1", b"DLW1\x00", b"DLW1\x00\x00", b"DLW1\x00\x00\x00"])
def test_dlw_short_input_raises_value_error(data):
    # Every malformed artefact must raise ValueError (never struct.error).
    with pytest.raises(ValueError):
        deserialize_state_dict(data)


def test_dlw_truncated_artefact_raises_value_error():
    data = serialize_state_dict(_sd())
    (hlen,) = struct.unpack("<I", data[4:8])
    for cut in (8, 8 + hlen // 2, 8 + hlen, 8 + hlen + 3, len(data) - 1):
        with pytest.raises(ValueError):
            deserialize_state_dict(data[:cut])
    # header length claims more bytes than exist
    with pytest.raises(ValueError, match="truncated"):
        deserialize_state_dict(data[:4] + struct.pack("<I", 10_000) + data[8:40])


# -- global RNG isolation (spec §9.5 three seeds, §11 repeatability) ----------------


def test_zero_model_leaves_global_rng_untouched():
    torch.manual_seed(12345)
    before = torch.get_rng_state()
    zero_model()
    zero_model(requires_grad=True)
    assert torch.equal(before, torch.get_rng_state())


def test_load_artifact_leaves_global_rng_untouched(tmp_path):
    p = tmp_path / "m.dlw"
    save_artifact(_patterned_model(), p)
    torch.manual_seed(54321)
    before = torch.get_rng_state()
    load_artifact(p)
    assert torch.equal(before, torch.get_rng_state())


def test_model_construction_does_not_collapse_seeds(tmp_path):
    p = tmp_path / "m.dlw"
    save_artifact(_patterned_model(), p)
    draws = []
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        zero_model(requires_grad=True)
        load_artifact(p)
        draws.append(torch.rand(4))
        torch.manual_seed(seed)
        assert torch.equal(draws[-1], torch.rand(4))  # same as if no model had been built
    assert not torch.equal(draws[0], draws[1]) and not torch.equal(draws[1], draws[2])


# -- sparsity -------------------------------------------------------------------------


def test_sparsity_zero_model():
    rep = sparsity(zero_model())
    assert set(rep) == set(PARAM_SHAPES) | {"overall"}
    assert rep["overall"] == {"nonzero": 0, "slots": 3256, "density": 0.0}
    for k, shape in PARAM_SHAPES.items():
        assert rep[k]["nonzero"] == 0 and rep[k]["slots"] == int(np.prod(shape))


def test_sparsity_counts_with_tolerance():
    assert SPARSITY_TOLERANCE == 1e-9
    m = zero_model()
    with torch.no_grad():
        m.fc1.weight[0, 0] = 1.0
        m.fc1.weight[5, 7] = -2.0
        m.fc1.weight[6, 7] = 1e-10  # below tolerance: not counted
        m.fc1.weight[7, 7] = -1e-10  # below tolerance: not counted
        m.fc2.bias[3] = 1e-8  # above tolerance: counted
        m.fc3.weight[1, 2] = -0.5
        m.fc3.bias[:] = 0.25
    rep = sparsity(m)
    assert rep["fc1.weight"]["nonzero"] == 2
    assert rep["fc1.bias"]["nonzero"] == 0
    assert rep["fc2.weight"]["nonzero"] == 0
    assert rep["fc2.bias"]["nonzero"] == 1
    assert rep["fc3.weight"]["nonzero"] == 1
    assert rep["fc3.bias"]["nonzero"] == 4
    assert rep["overall"]["nonzero"] == 8
    assert rep["overall"]["density"] == pytest.approx(8 / 3256)
    assert rep["fc3.bias"]["density"] == 1.0
    # A looser tolerance drops the 1e-8 entry.
    assert sparsity(m, tol=1e-6)["overall"]["nonzero"] == 7


def test_sparsity_dense_model():
    m = zero_model()
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(0.1)
    assert sparsity(m)["overall"] == {"nonzero": 3256, "slots": 3256, "density": 1.0}


def test_class_names_match_output_order():
    assert CLASS_NAMES == ("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING")
    assert OUTPUT_DIM == len(CLASS_NAMES)
    assert [ClassLabel(i).name for i in range(OUTPUT_DIM)] == list(CLASS_NAMES)


# -- two-process repeatability (spec §11 "Repeatability", §15) ------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

_CHILD = """
import json, sys
import numpy as np
import torch
from agent_loop_lab.model import load_artifact, predict
model, meta = load_artifact(sys.argv[1])
x = torch.from_numpy(np.frombuffer(open(sys.argv[2], "rb").read(), dtype="<f4").astype(np.float32).reshape(-1, 120))
with torch.no_grad():
    logits = model(x)
print(json.dumps({
    "logits": logits.numpy().astype("<f4").tobytes().hex(),
    "predictions": predict(logits).tolist(),
    "metadata": meta,
}))
"""


def _literal_model() -> DoomLoopMLP:
    """A small hand-written circuit with literal constants (no training, no search).

    fc1 units 0-3 detect inputs 0-3, unit 4 averages all inputs; fc2 passes
    units 0-3 through and mixes in unit 4; fc3 maps fc2 unit k to class k.
    """
    m = zero_model()
    with torch.no_grad():
        for k, w in enumerate((1.25, -0.75, 0.5, 2.0)):
            m.fc1.weight[k, k] = abs(w)
            m.fc1.weight[k, (k + 7) % INPUT_DIM] = w / 3.0
        m.fc1.weight[4, :] = 1.0 / 120.0
        m.fc1.bias[:5] = torch.tensor([-0.1, 0.05, -0.2, -0.3, 0.0])
        for k in range(4):
            m.fc2.weight[k, k] = 1.0 + 0.1 * k
            m.fc2.weight[k, 4] = (-1.0) ** k * 0.37
        m.fc2.bias[:4] = torch.tensor([0.01, -0.02, 0.03, -0.04])
        for k in range(4):
            m.fc3.weight[k, k] = 3.0
        m.fc3.bias[:] = torch.tensor([0.2, 0.0, -0.1, -0.3])
    return m


def _fixed_batch() -> np.ndarray:
    rows = [np.zeros(INPUT_DIM)]
    for k in range(4):
        e = np.zeros(INPUT_DIM)
        e[k] = 1.0
        rows.append(e)
    i = np.arange(INPUT_DIM)
    for r in range(11):
        rows.append(((i * 7 + r * 13) % 17) / 16.0)
    return np.stack(rows).astype(np.float32)


def test_two_process_repeatability(tmp_path):
    """Same artefact + same batch in two fresh interpreters -> byte-identical logits, identical predictions."""
    model = _literal_model()
    art = tmp_path / "literal.dlw"
    sha = save_artifact(model, art, {"phase": "test", "kind": "literal"})
    assert sparsity(model)["overall"]["nonzero"] > 0
    batch = _fixed_batch()
    batch_path = tmp_path / "batch.f32"
    batch_path.write_bytes(batch.astype("<f4").tobytes())

    with torch.no_grad():
        expected_logits = model(torch.from_numpy(batch))
    expected = {
        "logits": expected_logits.numpy().astype("<f4").tobytes().hex(),
        "predictions": predict(expected_logits).tolist(),
        "metadata": {"phase": "test", "kind": "literal"},
    }
    # Non-trivial: the circuit separates the four one-hot probes, and logits aren't all equal.
    assert expected["predictions"][:5] == [0, 0, 1, 2, 3]
    assert len(set(expected["predictions"])) == 4
    assert len(set(expected_logits.reshape(-1).tolist())) > 4

    src = str(REPO_ROOT / "src")
    outputs = []
    for hashseed in ("1", "2"):
        env = dict(os.environ)
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONHASHSEED"] = hashseed
        r = subprocess.run(
            [sys.executable, "-c", _CHILD, str(art), str(batch_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmp_path,
            timeout=300,
        )
        assert r.returncode == 0, r.stderr
        outputs.append(json.loads(r.stdout))

    first, second = outputs
    assert first["logits"] == second["logits"]  # byte-identical float32 logits
    assert first["predictions"] == second["predictions"]
    assert first == second == expected
    assert bytes.fromhex(first["logits"]) == expected_logits.numpy().astype("<f4").tobytes()
    assert artifact_sha256(art) == sha  # the children only read the artefact
