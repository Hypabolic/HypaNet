"""slop-net.bin format (SPEC.md §7), both id layouts and both file formats, and JS runtime parity with the numpy
reference (§12.3)."""

from __future__ import annotations

import json
import struct
import subprocess

import numpy as np
import pytest

from conftest import SLOP, run_node
from lint import compile as net_compile
from ui import model as mdl
from ui import tokenizer as tk
from ui.tests.test_model import random_tensors
from validate import build_floor_net

NET_RUN = SLOP / "lint" / "tests" / "net_run.js"
TOL = 1e-6


R1 = SLOP / "runs" / "r1" / "work"
DEV = SLOP / "datasets" / "dev" / "screens.jsonl"


def _artefact(tmp_path, name, tensors, vocab=("save", "set", "project"), ph=(4, 0), v1=False):
    d = tmp_path / name
    mdl.save_artifact(d, tensors, list(vocab), ph)
    art = mdl.load_artifact(d)
    net = tmp_path / f"{name}.bin"
    compiler = _compile_v1 if v1 else net_compile.compile_artifact
    net.write_bytes(compiler(art, (d / "vocab.txt").read_bytes()))
    return art, net


def _compile_v1(art: mdl.Artifact, vocab_bytes: bytes) -> bytes:
    """r1's compiler (slop.net/1), frozen here so the readers' support for old files stays tested."""
    assert art.ids == 1
    entries, chunks, offset = [], [], 0
    for name, shape in mdl.TENSOR_SHAPES.items():
        t = art.tensors[name]
        rows2d = t.reshape(1, -1) if t.ndim == 1 else t
        runs: list[list[int]] = []
        for r in np.nonzero(np.any(rows2d != 0, axis=1))[0].tolist():
            if runs and runs[-1][1] == r:
                runs[-1][1] = r + 1
            else:
                runs.append([r, r + 1])
        data = b"".join(np.ascontiguousarray(rows2d[a:b], dtype="<f4").tobytes() for a, b in runs)
        entries.append({"name": name, "shape": list(shape), "rows": runs, "offset": offset, "nbytes": len(data)})
        chunks.append(data)
        offset += len(data)
    header = {"format": "slop.net/1", "artifact_sha256": art.sha256, "vocab_sha256": art.vocab_sha256,
              "pointer_head": list(art.pointer_head), "classes": list(mdl.CLASSES), "param_count": mdl.PARAM_COUNT,
              "tensors": entries, "vocab": {"offset": offset, "nbytes": len(vocab_bytes)}}
    hbytes = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    hbytes += b" " * ((-(8 + 4 + len(hbytes))) % 8)
    return b"SLOPNET1" + struct.pack("<I", len(hbytes)) + hbytes + b"".join(chunks) + vocab_bytes


def _sparse_cells(seed, ids, n_rows, per_row, vocab_len=0, density=0.05):
    """Mostly-zero tensors shaped like an authored network: ``n_rows`` tok_emb rows with ``per_row`` cells each."""
    rng = np.random.default_rng(seed)
    t = mdl.zero_tensors(ids)
    lay = tk.layout(ids)
    rows = np.concatenate([np.arange(512, 512 + vocab_len),
                           rng.choice(np.arange(lay.hash_base, lay.vocab_size), n_rows - vocab_len, replace=False)])
    for r in rows:
        cols = rng.choice(128, per_row, replace=False)
        t["tok_emb.weight"][r, cols] = rng.choice([-1.0, 0.5, 1.0, 2.0], per_row)
    for name, x in t.items():
        if name != "tok_emb.weight":
            m = rng.random(x.shape) < density
            x[m] = (rng.standard_normal(int(m.sum())) * 0.3).astype(np.float32)
    return t


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
    assert header["format"] == "slop.net/2" and header["ids"] == 1
    assert header["artifact_sha256"] == art.sha256 and header["vocab_sha256"] == art.vocab_sha256
    assert header["pointer_head"] == [4, 0] and header["param_count"] == mdl.PARAM_COUNT
    assert vocab == (tmp_path / "sparse" / "vocab.txt").read_bytes()
    for k in mdl.TENSOR_SHAPES:
        assert np.array_equal(tensors[k], art.tensors[k]), k
    tok = next(e for e in header["tensors"] if e["name"] == "tok_emb.weight")
    assert tok["enc"] == "rows"  # dense random rows: cells would cost more
    stored = sum(b - a for a, b in tok["rows"])
    assert stored == int(np.count_nonzero(np.any(art.tensors["tok_emb.weight"] != 0, axis=1))) < mdl.V
    qb = next(e for e in header["tensors"] if e["name"] == "layers.3.attn.q.bias")
    assert qb["enc"] == "rows" and qb["rows"] == [] and qb["nbytes"] == 0
    assert net_compile.compile_artifact(art, vocab) == data  # deterministic


def test_dense_size_is_about_8_5_mb(tmp_path):
    _, net = _artefact(tmp_path, "dense", random_tensors(12))
    size = net.stat().st_size
    assert 4 * mdl.PARAM_COUNT < size < 4 * mdl.PARAM_COUNT + 400_000
    print(f"\ndense slop-net.bin: {size} bytes")


@pytest.mark.parametrize("ids", [1, 2])
def test_cells_encoding_round_trips_bit_for_bit(tmp_path, ids):
    t = _sparse_cells(31, ids, 900, 7, vocab_len=3)
    t["tok_emb.weight"][600, 5] = -0.0  # a lone negative zero is a stored cell
    t["head.bias"][2] = -0.0
    art, net = _artefact(tmp_path, f"cells{ids}", t)
    assert art.ids == ids
    header, tensors, _ = net_compile.read_net(net.read_bytes())
    assert header["ids"] == ids and header["param_count"] == mdl.param_count(ids)
    encs = {e["name"]: e["enc"] for e in header["tensors"]}
    assert encs["tok_emb.weight"] == "cells" and encs["layers.0.attn.q.weight"] == "cells"
    tok = next(e for e in header["tensors"] if e["name"] == "tok_emb.weight")
    assert tok["nrows"] == 901 and tok["nbytes"] == 901 * (4 + 16) + 4 * (900 * 7 + 1)
    for k in mdl.tensor_shapes(ids):
        assert np.array_equal(tensors[k].view(np.uint32), art.tensors[k].view(np.uint32)), k
    assert net_compile.compile_artifact(art, (tmp_path / f"cells{ids}" / "vocab.txt").read_bytes()) == net.read_bytes()


def test_old_slop_net_1_files_still_load(tmp_path, test_screens):
    t = _sparse_random(41)
    art, net1 = _artefact(tmp_path, "old", t, ph=(2, 6), v1=True)
    net2 = tmp_path / "new.bin"
    net2.write_bytes(net_compile.compile_artifact(art, (tmp_path / "old" / "vocab.txt").read_bytes()))
    h1, t1, v1 = net_compile.read_net(net1.read_bytes())
    h2, t2, v2 = net_compile.read_net(net2.read_bytes())
    assert h1["format"] == "slop.net/1" and "ids" not in h1 and h2["format"] == "slop.net/2"
    assert v1 == v2 and all(np.array_equal(t1[k], t2[k]) for k in mdl.TENSOR_SHAPES)
    screens = test_screens[:12]
    js1 = run_node(NET_RUN, {"net": str(net1), "screens": screens})
    js2 = run_node(NET_RUN, {"net": str(net2), "screens": screens})
    assert js1["sha256"] == js2["sha256"] == art.sha256 and js1["screens"] == js2["screens"]


def test_layout_mismatch_in_the_header_is_refused(tmp_path):
    _, net = _artefact(tmp_path, "mm", mdl.zero_tensors())
    data = net.read_bytes()
    assert data.count(b'"ids":1,') == 1
    bad = tmp_path / "mm-bad.bin"
    bad.write_bytes(data.replace(b'"ids":1,', b'"ids":2,'))
    with pytest.raises(ValueError, match="layout 2"):
        net_compile.read_net(bad.read_bytes())
    script = f"import {{loadNet}} from './lint/runtime.js'; import fs from 'node:fs'; loadNet(fs.readFileSync({json.dumps(str(bad))}));"
    res = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, cwd=SLOP)
    assert res.returncode != 0 and "id layout 2" in res.stderr


def test_sizes(tmp_path):
    """r1-like sparsity (about 19 cells per stored tok_emb row) and fully dense stored rows, in both layouts."""
    sizes = {}
    for ids, n_rows in ((1, 7211), (2, 12000)):
        for per_row in (19, 128):
            vocab = [f"w{i}" for i in range(1077 if ids == 1 else 6000)]  # r1 had 1,077 stems
            t = _sparse_cells(50 + ids, ids, n_rows, per_row, len(vocab), density=0.005)  # r1: 0.6% outside tok_emb
            _, net = _artefact(tmp_path, f"s{ids}-{per_row}", t, vocab)
            sizes[(ids, per_row)] = net.stat().st_size
            tok = next(e for e in net_compile.read_net(net.read_bytes())[0]["tensors"] if e["name"] == "tok_emb.weight")
            if per_row == 19:
                assert tok["enc"] == "cells" and tok["nbytes"] == n_rows * (4 + 16) + n_rows * per_row * 4
            else:
                assert tok["enc"] == "rows" and tok["nbytes"] == n_rows * 128 * 4
    print("\nslop-net.bin sizes: " + ", ".join(f"layout {i}, {r} rows x {c} cells: {sizes[(i, c)]:,} bytes"
                                              for (i, r) in ((1, 7211), (2, 12000)) for c in (19, 128)))
    assert sizes[(2, 19)] < 1_300_000  # 12,000 rows: index 48 kB + masks 192 kB + values 912 kB + the rest
    assert sizes[(2, 128)] < 12000 * 128 * 4 + 300_000  # dense rows fall back to row runs


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
    ("ids2-random", lambda: random_tensors(24, scale=0.5, ids=2), (4, 2)),
    ("ids2-zero", lambda: mdl.zero_tensors(2), (1, 0)),
])
def test_js_runtime_matches_numpy(tmp_path, test_screens, name, make, ph):
    vocab = build_floor_net.VOCAB if name == "floor-net" else ("save", "set", "project")
    if name.startswith("ids2"):  # more stems than layout 1 allows, the test screens' words among them
        vocab = ("save", "set", "project", "notif", "account") + tuple(f"w{i}" for i in range(4000))
    art, net = _artefact(tmp_path, name, make(), vocab, ph)
    r = _parity(art, net, test_screens)
    print(f"\n{name}: {r['nodes']} nodes, {r['flagged']} non-KEEP, {r['pointed']} with a pointer; class differs "
          f"{r['class_differs']}, pointer differs {r['pointer_differs']}, logits out of tolerance {r['logits_out_of_tolerance']}, "
          f"worst relative logit difference {r['worst']:.2e}; JS {r['ms']:.0f} ms")
    assert r["class_differs"] == r["pointer_differs"] == r["logits_out_of_tolerance"] == 0
    if "random" in name:
        assert r["flagged"] > 50 and r["pointed"] > 20  # the pointer path is exercised


def test_js_runtime_matches_numpy_on_sparse_layout_2(tmp_path, test_screens):
    """An authored-looking layout-2 net (cells encoding throughout) whose rows include the test screens' stems."""
    stems = sorted({t.stem for s in test_screens for n in s["nodes"] for t in tk.tokenize_text(n["text"]).tokens if t.stem})
    vocab = stems[:200] + [f"w{i}" for i in range(5000)]
    t = _sparse_cells(61, 2, 6000, 12, vocab_len=len(vocab))
    rng = np.random.default_rng(62)
    for st in stems[200:]:
        t["tok_emb.weight"][tk.hash_bucket(st, 2), rng.choice(128, 6, replace=False)] = 1.0
    for name in t:
        if name.startswith("layers.") and name.endswith(".weight"):
            t[name] *= 3.0
    t["head.weight"][1:] = (rng.standard_normal((3, 128)) * 0.4).astype(np.float32)
    art, net = _artefact(tmp_path, "ids2-sparse", t, vocab, (2, 4))
    header, _, _ = net_compile.read_net(net.read_bytes())
    assert header["ids"] == 2 and {e["enc"] for e in header["tensors"]} == {"rows", "cells"}
    r = _parity(art, net, test_screens)
    print(f"\nids2-sparse: {r['nodes']} nodes, {r['flagged']} non-KEEP, {r['pointed']} pointed, worst {r['worst']:.2e}")
    assert r["class_differs"] == r["pointer_differs"] == r["logits_out_of_tolerance"] == 0
    assert r["flagged"] > 20


@pytest.mark.skipif(not (R1 / "artifact.npz").is_file() or not DEV.is_file(), reason="needs runs/r1 and the dev set")
def test_r1_artefact_parity_on_dev_screens(tmp_path):
    """The r1 network (id layout 1) through slop.net/2 and the JS runtime matches numpy on 200 dev screens."""
    art = mdl.load_artifact(R1)
    assert art.ids == 1 and "ids" not in art.meta
    net = tmp_path / "r1.bin"
    net.write_bytes(net_compile.compile_artifact(art, (R1 / "vocab.txt").read_bytes()))
    with DEV.open(encoding="utf-8") as f:
        lines = f.readlines()
    screens = [json.loads(lines[i]) for i in range(0, len(lines), max(1, len(lines) // 200))][:200]
    r = _parity(art, net, screens)
    print(f"\nr1 on {len(screens)} dev screens: {r['nodes']} nodes, {r['flagged']} non-KEEP; slop-net.bin "
          f"{net.stat().st_size:,} bytes; worst {r['worst']:.2e}")
    assert r["class_differs"] == r["pointer_differs"] == r["logits_out_of_tolerance"] == 0 and r["flagged"] > 50
