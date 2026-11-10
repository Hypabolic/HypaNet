"""slop-net.bin format (SPEC.md §7) and JS runtime parity with the numpy reference (§12.3)."""

from __future__ import annotations

import json
import struct
import subprocess

import numpy as np
import pytest

from conftest import SLOP, run_node
from lint import compile as net_compile
from ui import model as mdl
from ui.tests.test_model import random_tensors
from validate import build_floor_net

NET_RUN = SLOP / "lint" / "tests" / "net_run.js"
TOL = 1e-6


def _artefact(tmp_path, name, tensors, vocab=("save", "set", "project"), ph=(4, 0)):
    d = tmp_path / name
    mdl.save_artifact(d, tensors, list(vocab), ph)
    art = mdl.load_artifact(d)
    net = tmp_path / f"{name}.bin"
    net.write_bytes(net_compile.compile_artifact(art, (d / "vocab.txt").read_bytes()))
    return art, net


def _sparse_random(seed):
    t = random_tensors(seed, scale=0.5)
    rng = np.random.default_rng(seed + 100)
    for name in ("tok_emb.weight", "role_emb.weight", "layers.1.mlp.fc1.weight"):
        rows = rng.choice(t[name].shape[0], size=t[name].shape[0] // 2, replace=False)
        t[name][rows] = 0.0
    t["layers.3.attn.q.bias"][:] = 0.0
    return t


def test_compile_round_trip_and_layout(tmp_path):
    art, net = _artefact(tmp_path, "sparse", _sparse_random(11))
    data = net.read_bytes()
    assert data[:8] == b"SLOPNET1"
    (hlen,) = struct.unpack("<I", data[8:12])
    assert (12 + hlen) % 8 == 0
    header, tensors, vocab = net_compile.read_net(data)
    assert header["artifact_sha256"] == art.sha256 and header["vocab_sha256"] == art.vocab_sha256
    assert header["pointer_head"] == [4, 0] and header["param_count"] == mdl.PARAM_COUNT
    assert vocab == (tmp_path / "sparse" / "vocab.txt").read_bytes()
    for k in mdl.TENSOR_SHAPES:
        assert np.array_equal(tensors[k], art.tensors[k]), k
    tok = next(e for e in header["tensors"] if e["name"] == "tok_emb.weight")
    stored = sum(b - a for a, b in tok["rows"])
    assert stored == int(np.count_nonzero(np.any(art.tensors["tok_emb.weight"] != 0, axis=1))) < mdl.V
    qb = next(e for e in header["tensors"] if e["name"] == "layers.3.attn.q.bias")
    assert qb["rows"] == [] and qb["nbytes"] == 0
    assert net_compile.compile_artifact(art, vocab) == data  # deterministic


def test_dense_size_is_about_8_5_mb(tmp_path):
    _, net = _artefact(tmp_path, "dense", random_tensors(12))
    size = net.stat().st_size
    assert 4 * mdl.PARAM_COUNT < size < 4 * mdl.PARAM_COUNT + 400_000
    print(f"\ndense slop-net.bin: {size} bytes")


def test_js_refuses_a_tampered_vocabulary(tmp_path):
    _, net = _artefact(tmp_path, "t", mdl.zero_tensors())
    data = bytearray(net.read_bytes())
    data[-2] ^= 0x01
    bad = tmp_path / "bad.bin"
    bad.write_bytes(bytes(data))
    script = f"import {{loadNet}} from './lint/runtime.js'; import fs from 'node:fs'; loadNet(fs.readFileSync({json.dumps(str(bad))}));"
    res = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, cwd=SLOP)
    assert res.returncode != 0 and "vocab sha256" in res.stderr


def _parity(art: mdl.Artifact, net, screens) -> dict:
    out = run_node(NET_RUN, {"net": str(net), "screens": screens})
    assert out["sha256"] == art.sha256 and out["pointerHead"] == list(art.pointer_head)
    nodes = cls_diff = ptr_diff = bad_logits = flagged = pointed = 0
    worst = 0.0
    for s, js in zip(screens, out["screens"]):
        for a, b in zip(mdl.predict_screen(art, s), js):
            nodes += 1
            flagged += a.cls != "KEEP"
            pointed += a.pointer is not None
            cls_diff += a.cls != b["cls"]
            ptr_diff += a.pointer != b["pointer"]
            rel = max(abs(x - y) / max(1.0, abs(x)) for x, y in zip(a.logits, b["logits"]))
            bad_logits += rel > TOL
            worst = max(worst, rel)
    return {"nodes": nodes, "class_differs": cls_diff, "pointer_differs": ptr_diff, "logits_out_of_tolerance": bad_logits,
            "worst": worst, "flagged": flagged, "pointed": pointed, "ms": out["ms"]}


@pytest.mark.parametrize("name,make,ph", [
    ("random-1", lambda: random_tensors(21), (4, 0)),
    ("random-2", lambda: random_tensors(22, scale=0.3), (2, 5)),
    ("random-3-sparse", lambda: _sparse_random(23), (3, 7)),
    ("zero", mdl.zero_tensors, (1, 0)),
    ("floor-net", build_floor_net.build_tensors, build_floor_net.POINTER_HEAD),
])
def test_js_runtime_matches_numpy(tmp_path, test_screens, name, make, ph):
    vocab = build_floor_net.VOCAB if name == "floor-net" else ("save", "set", "project")
    art, net = _artefact(tmp_path, name, make(), vocab, ph)
    r = _parity(art, net, test_screens)
    print(f"\n{name}: {r['nodes']} nodes, {r['flagged']} non-KEEP, {r['pointed']} with a pointer; class differs "
          f"{r['class_differs']}, pointer differs {r['pointer_differs']}, logits out of tolerance {r['logits_out_of_tolerance']}, "
          f"worst relative logit difference {r['worst']:.2e}; JS {r['ms']:.0f} ms")
    assert r["class_differs"] == r["pointer_differs"] == r["logits_out_of_tolerance"] == 0
    if name.startswith("random"):
        assert r["flagged"] > 50 and r["pointed"] > 20  # the pointer path is exercised
