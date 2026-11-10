"""numpy reference model (SPEC.md §7): tensors, artefact files, forward pass and pointer rule."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from ui import model as mdl
from ui import tokenizer as tk


def random_tensors(seed: int, scale: float = 1.0, ids: int = 1) -> "dict[str, np.ndarray]":
    """Seeded random weights for tests (harness code, not an authored network), for id layout ``ids``."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, shape in mdl.tensor_shapes(ids).items():
        if name.endswith("rel_bias"):
            std = 1.0
        elif name.endswith("emb.weight"):
            std = 0.5
        elif name.endswith("bias"):
            std = 0.1
        else:
            std = 1.5 / np.sqrt(shape[1])
        out[name] = (rng.standard_normal(shape) * std * scale).astype(np.float32)
    return out


def test_shapes_and_parameter_count():
    assert mdl.PARAM_COUNT == 2_128_900
    share = np.prod(mdl.TENSOR_SHAPES["tok_emb.weight"]) / mdl.PARAM_COUNT
    assert round(share * 100, 1) == 61.6
    assert mdl.TENSOR_SHAPES["layers.0.attn.rel_bias"] == (8, 128)
    assert mdl.TENSOR_SHAPES["role_emb.weight"] == (48, 128) and mdl.TENSOR_SHAPES["flag_emb.weight"] == (32, 128)


def test_save_load_and_sha256(tmp_path):
    t = random_tensors(1)
    sha = mdl.save_artifact(tmp_path / "a", t, ["save", "set"], (2, 5))
    sha2 = mdl.save_artifact(tmp_path / "b", t, ["save", "set"], (2, 5))
    assert sha == sha2
    for f in mdl.ARTIFACT_FILES:
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()
    art = mdl.load_artifact(tmp_path / "a")
    import hashlib

    parts = "".join(hashlib.sha256((tmp_path / "a" / f).read_bytes()).hexdigest() for f in ("artifact.npz", "vocab.txt", "meta.json"))
    assert art.sha256 == sha == hashlib.sha256(parts.encode()).hexdigest()
    assert art.pointer_head == (2, 5) and art.vocab.stems == ["save", "set"]
    for k in mdl.TENSOR_SHAPES:
        assert np.array_equal(art.tensors[k], t[k])
    with np.load(tmp_path / "a" / "artifact.npz") as z:  # an ordinary npz to numpy
        assert sorted(z.files) == sorted(mdl.TENSOR_SHAPES)


def test_layout_2_shapes():
    s1, s2 = mdl.tensor_shapes(1), mdl.tensor_shapes(2)
    assert s1 == mdl.TENSOR_SHAPES and list(s1) == list(s2)
    assert s2["tok_emb.weight"] == (25088, 128) and s1["tok_emb.weight"] == (10240, 128)
    assert all(s1[k] == s2[k] for k in s1 if k != "tok_emb.weight")
    assert mdl.param_count(2) == 4_029_444 == mdl.PARAM_COUNT + (25088 - 10240) * 128
    assert mdl.zero_tensors(2)["tok_emb.weight"].shape == (25088, 128)


def test_layout_2_save_load_and_meta(tmp_path):
    t = random_tensors(3, ids=2)
    vocab = [f"w{i}" for i in range(5000)]  # more than layout 1's 3,584
    sha = mdl.save_artifact(tmp_path / "a", t, vocab, (3, 1))  # layout inferred from tok_emb
    meta = json.loads((tmp_path / "a" / "meta.json").read_text())
    assert meta == {"format": "slop.artifact/1", "pointer_head": [3, 1], "ids": 2}
    art = mdl.load_artifact(tmp_path / "a")
    assert art.sha256 == sha and art.ids == 2 and art.vocab.ids_version == 2
    assert art.vocab.word_id("w4999") == 512 + 4999 and art.vocab.word_id("zzz") == tk.hash_bucket("zzz", 2)
    assert all(np.array_equal(art.tensors[k], t[k]) for k in mdl.tensor_shapes(2))
    assert mdl.save_artifact(tmp_path / "b", t, vocab, (3, 1), ids=2) == sha
    assert mdl.from_tensors(t, vocab, (3, 1)).ids == 2
    # layout 1 writes no "ids" key (r1's meta.json, byte for byte), whether inferred or explicit
    mdl.save_artifact(tmp_path / "c", mdl.zero_tensors(), ["save"], (4, 0), ids=1)
    assert (tmp_path / "c" / "meta.json").read_text() == '{"format": "slop.artifact/1", "pointer_head": [4, 0]}\n'
    assert mdl.load_artifact(tmp_path / "c").ids == 1
    with pytest.raises(mdl.ArtifactError, match="tok_emb"):
        mdl.save_artifact(tmp_path / "d", t, vocab, (3, 1), ids=1)
    with pytest.raises(tk.TokenizerError, match="max"):
        mdl.save_artifact(tmp_path / "e", mdl.zero_tensors(), vocab, (3, 1))


def test_layout_mismatch_is_refused(tmp_path):
    t2 = mdl.zero_tensors(2)
    with pytest.raises(mdl.ArtifactError, match="tok_emb"):  # layout-2 tensors, meta without "ids"
        mdl.load_artifact(_write(tmp_path, t2))
    with pytest.raises(mdl.ArtifactError, match="tok_emb"):  # layout-1 tensors, meta naming layout 2
        mdl.load_artifact(_write(tmp_path, mdl.zero_tensors(), meta={"format": "slop.artifact/1", "pointer_head": [4, 0], "ids": 2}))
    for bad in (3, 0, "2", True, None):
        with pytest.raises(mdl.ArtifactError, match="ids"):
            mdl.load_artifact(_write(tmp_path, t2, meta={"format": "slop.artifact/1", "pointer_head": [4, 0], "ids": bad}))
    meta1 = {"format": "slop.artifact/1", "pointer_head": [4, 0], "ids": 1}  # explicit 1 is allowed
    assert mdl.load_artifact(_write(tmp_path, mdl.zero_tensors(), meta=meta1)).ids == 1
    big = "".join(f"w{i}\n" for i in range(4000))
    with pytest.raises(mdl.ArtifactError, match="vocab"):
        mdl.load_artifact(_write(tmp_path, mdl.zero_tensors(), vocab=big))
    assert mdl.load_artifact(_write(tmp_path, t2, meta={"format": "slop.artifact/1", "pointer_head": [4, 0], "ids": 2},
                                    vocab=big)).vocab.word_id("w3999") == 512 + 3999


def test_layout_2_forward_uses_the_moved_rows(test_screens):
    """A layout-2 net whose tok_emb is a layout-1 net's rows moved to their layout-2 ids predicts exactly the same."""
    from collections import Counter

    stems = sorted({t.stem for s in test_screens for n in s["nodes"] for t in tk.tokenize_text(n["text"]).tokens if t.stem})
    b1, b2 = Counter(tk.hash_bucket(st, 1) for st in stems), Counter(tk.hash_bucket(st, 2) for st in stems)
    # stems sharing a bucket in either layout go in the vocabulary, so every hashed stem has a row of its own
    vocab = ["save", "set"] + [st for st in stems if b1[tk.hash_bucket(st, 1)] > 1 or b2[tk.hash_bucket(st, 2)] > 1]
    vocab = list(dict.fromkeys(vocab))
    hashed = [st for st in stems if st not in vocab]
    assert len(hashed) > 300
    t1 = random_tensors(5, scale=0.5)
    e2 = np.zeros((25088, 128), dtype=np.float32)
    e2[: 512 + len(vocab)] = t1["tok_emb.weight"][: 512 + len(vocab)]
    for st in hashed:
        e2[tk.hash_bucket(st, 2)] = t1["tok_emb.weight"][tk.hash_bucket(st, 1)]
    a1 = mdl.from_tensors(t1, vocab, (2, 3))
    a2 = mdl.from_tensors({**t1, "tok_emb.weight": e2}, vocab, (2, 3))
    assert (a1.ids, a2.ids) == (1, 2)
    flagged = 0
    for s in test_screens:
        p1, p2 = mdl.predict_screen(a1, s), mdl.predict_screen(a2, s)
        assert [(p.cls, p.pointer, p.logits) for p in p1] == [(p.cls, p.pointer, p.logits) for p in p2], s["id"]
        flagged += sum(p.cls != "KEEP" for p in p1)
    assert flagged > 20


def _write(tmp_path, tensors, meta=None, vocab="save\n", compressed=False):
    d = tmp_path / "art"
    d.mkdir(exist_ok=True)
    (np.savez_compressed if compressed else np.savez)(d / "artifact.npz", **tensors)
    (d / "vocab.txt").write_text(vocab)
    (d / "meta.json").write_text(json.dumps(meta or {"format": "slop.artifact/1", "pointer_head": [4, 0]}))
    return d


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "dtype", "nan", "inf"])
def test_bad_tensors_are_refused(tmp_path, change):
    t = dict(mdl.zero_tensors())
    if change == "missing":
        del t["head.bias"]
    elif change == "extra":
        t["head.extra"] = np.zeros(4, dtype=np.float32)
    elif change == "shape":
        t["head.bias"] = np.zeros(5, dtype=np.float32)
    elif change == "dtype":
        t["head.bias"] = np.zeros(4, dtype=np.float64)
    elif change == "nan":
        t["head.bias"] = np.array([0, np.nan, 0, 0], dtype=np.float32)
    else:
        t["layers.2.mlp.fc1.weight"] = t["layers.2.mlp.fc1.weight"].copy()
        t["layers.2.mlp.fc1.weight"][3, 3] = np.inf
    with pytest.raises(mdl.ArtifactError):
        mdl.load_artifact(_write(tmp_path, t))


def test_bad_files_are_refused(tmp_path):
    t = mdl.zero_tensors()
    with pytest.raises(mdl.ArtifactError, match="compressed"):
        mdl.load_artifact(_write(tmp_path, t, compressed=True))
    for meta in ({"format": "x", "pointer_head": [4, 0]}, {"format": "slop.artifact/1", "pointer_head": [0, 0]},
                 {"format": "slop.artifact/1", "pointer_head": [4, 8]}, {"format": "slop.artifact/1"}):
        with pytest.raises(mdl.ArtifactError):
            mdl.load_artifact(_write(tmp_path, t, meta=meta))
    with pytest.raises(mdl.ArtifactError):
        mdl.load_artifact(_write(tmp_path, t, vocab="save\nsave\n"))
    assert mdl.load_artifact(_write(tmp_path, t)).pointer_head == (4, 0)


def test_zero_network_keeps_everything(test_screens):
    art = mdl.from_tensors(mdl.zero_tensors(), [], (4, 0))
    for s in test_screens[:5]:
        for p in mdl.predict_screen(art, s):
            assert p.cls == "KEEP" and p.pointer is None and p.logits == [0.0] * 4


def test_bias_columns():
    c = mdl.bias_columns(np.array([0, 1, 1, 2, 70]))
    assert c[:, 0].tolist() == [127] * 5
    assert c[1, 1:].tolist() == [63, 63, 64, 126]
    assert c[3, 1:].tolist() == [62, 62, 63, 126]
    assert c[4, 1:].tolist() == [0, 0, 0, 63]
    assert c[0, 1:].tolist() == [64, 64, 65, 126]


def test_pointer_rule():
    node = [0, 1, 1, 2, 2, 3]
    tgt, bos, _ = mdl.pointer_from_attn(np.array([0.1, 0.2, 0.0, 0.3, 0.1, 0.3]), node, 1)
    assert tgt == 2 and bos == pytest.approx(0.1)  # node 2 sums 0.4 > node 3's 0.3; own node ignored
    tgt, _, _ = mdl.pointer_from_attn(np.array([0.2, 0.0, 0.0, 0.2, 0.2, 0.4]), node, 1)
    assert tgt == 2  # tie between nodes 2 and 3 goes to the earlier
    tgt, _, _ = mdl.pointer_from_attn(np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0]), node, 1)
    assert tgt is None  # BOS weight at least 0.5
    assert mdl.argmax_class(np.array([1.0, 1.0, 0.0, 1.0])) == 0


def _reference_forward(art: mdl.Artifact, w: tk.Window) -> np.ndarray:
    """A slow, loop-by-loop restatement of SPEC.md §7 to check the vectorised forward pass."""
    p = art.f64
    n = len(w.tokens)
    h = np.zeros((n, mdl.D))
    for t in range(n):
        h[t] = p["tok_emb.weight"][w.tokens[t]]
        if t > 0:
            h[t] = h[t] + p["off_emb.weight"][w.offsets[t]] + p["role_emb.weight"][w.roles[t]] + p["cluster_emb.weight"][w.clusters[t] % 64]
            for f in range(32):
                if (w.flags[t] >> f) & 1:
                    h[t] = h[t] + p["flag_emb.weight"][f]
    for li in range(mdl.L):
        pre = f"layers.{li}."
        q = h @ p[pre + "attn.q.weight"].T + p[pre + "attn.q.bias"]
        k = h @ p[pre + "attn.k.weight"].T + p[pre + "attn.k.bias"]
        v = h @ p[pre + "attn.v.weight"].T + p[pre + "attn.v.bias"]
        ctx = np.zeros((n, mdl.D))
        for hd in range(mdl.H):
            sl = slice(16 * hd, 16 * hd + 16)
            for i in range(n):
                s = np.zeros(n)
                for j in range(n):
                    c = 127 if j == 0 else max(-63, min(63, w.node[j] - w.node[i])) + 63
                    s[j] = q[i, sl] @ k[j, sl] / 4 + p[pre + "attn.rel_bias"][hd, c]
                a = np.exp(s - s.max())
                a /= a.sum()
                ctx[i, sl] = a @ v[:, sl]
        h = h + ctx @ p[pre + "attn.o.weight"].T + p[pre + "attn.o.bias"]
        h = h + np.maximum(h @ p[pre + "mlp.fc1.weight"].T + p[pre + "mlp.fc1.bias"], 0) @ p[pre + "mlp.fc2.weight"].T + p[pre + "mlp.fc2.bias"]
    return h[w.eon] @ p["head.weight"].T + p["head.bias"]


def test_forward_matches_a_loop_reference(test_screens):
    art = mdl.from_tensors(random_tensors(3), ["save"], (3, 2))
    screen = next(s for s in test_screens if s["source"]["path"].endswith("signup_form.tsx"))
    w = tk.encode_screen(screen, art.vocab).windows[0]
    got = mdl.forward(art, w).logits
    ref = _reference_forward(art, w)
    assert np.abs(got - ref).max() <= 1e-9 * max(1.0, np.abs(ref).max()), np.abs(got - ref).max()


def test_bos_takes_only_its_token_embedding():
    t = random_tensors(5)
    art = mdl.from_tensors(t, [], (1, 0))
    w = tk.encode_screen({"schema": "slop.screen/1", "nodes": [{"i": 0, "k": 9, "role": "h1", "text": "Hi", "flags": ["pill"]}]}).windows[0]
    h = mdl.embed(art.f64, w)
    assert np.array_equal(h[0], t["tok_emb.weight"][tk.BOS].astype(np.float64))
    assert not np.array_equal(h[1], t["tok_emb.weight"][w.tokens[1]].astype(np.float64))


def test_npz_is_uncompressed_and_ordered(tmp_path):
    data = mdl.npz_bytes(mdl.zero_tensors())
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert [i.filename for i in zf.infolist()] == [k + ".npy" for k in mdl.TENSOR_SHAPES]
        assert all(i.compress_type == zipfile.ZIP_STORED for i in zf.infolist())
