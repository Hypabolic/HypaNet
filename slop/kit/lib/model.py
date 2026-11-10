"""Slop transformer, numpy reference (SPEC.md §7). Mirrored by ``lint/runtime.js``.

The v0.3 transformer (``src/agent_loop_lab/v03/model.py``) with three changes: attention is bidirectional, the bias
uses the signed node distance, and role, cluster and flag embeddings are added to every token of a node::

    h0[t] = E[tok_t] + P[off_t] + Rr[role_t] + Cc[cluster_t mod 64] + sum_{f in flags_t} F[f]   (BOS: E[BOS] only)
    for each of 4 layers:
        q, k, v = h W_q^T + b_q, h W_k^T + b_k, h W_v^T + b_v                (8 heads x 16)
        score[h,i,j] = q_{h,i} . k_{h,j} / 4 + B[h, c(i,j)]                  (all positions, no causal mask)
        c(i,j) = 127 if key j is BOS, else clamp(node_j - node_i, -63, 63) + 63
        h = h + concat_h(softmax_j(score) v_h) W_o^T + b_o
        h = h + W_2 ReLU(W_1 h + b_1) + b_2
    logits = W_out h + b_out at each node's EON, classes (KEEP, ECHO, NARRATE, DECOR)

No normalisation, no dropout. Weights are stored as float32 and every computation runs in float64. Argmax ties go to
the lower class index, so all-zero logits mean KEEP.

Pointer: ``meta.json`` names ``pointer_head: [layer, head]`` (layer 1-4, head 0-7). For a node predicted ECHO or
NARRATE, that head's attention weights from the node's EON are summed over each other node's tokens; the target is
the node with the largest sum (ties to the earlier node), or none if the weight on BOS is at least 0.5.

Artefact directory: ``artifact.npz`` (uncompressed; exactly the tensors of :func:`tensor_shapes`, float32, finite),
``vocab.txt``, ``meta.json`` (``{"format": "slop.artifact/1", "pointer_head": [L, H]}``, plus ``"ids": 2`` for the
second id layout; absent means 1). Its sha256 is the sha256 of the three files' lower-case hex sha256 digests
concatenated in that order (192 ASCII characters). Tensor names follow the v0.3 ``state_dict`` names; layers are
numbered 0-3 in names and 1-4 in ``pointer_head``.

Id layouts (``tokenizer.LAYOUTS``): only ``tok_emb.weight`` depends on it, [10240, 128] in layout 1 (r1: 3,584 stems,
6,144 hash buckets; 2,128,900 parameters) and [25088, 128] in layout 2 (r2: 16,384 stems, 8,192 hash buckets;
4,029,444 parameters). :data:`TENSOR_SHAPES`, :data:`PARAM_COUNT` and :data:`V` are layout 1.

Standalone: numpy, the standard library and ``tokenizer.py``.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from . import tokenizer as tk
except ImportError:  # top-level module, as in the author kit's lib/
    import tokenizer as tk

D = 128  # residual width
L = 4  # layers
H = 8  # heads
DH = 16  # head width
M = 512  # MLP width
O = 32  # offsets inside a node (0-31)
N_ROLES = 48
N_CLUSTERS = 64
N_FLAGS = 32
NB = 128  # bias columns: 0-126 signed node distance -63..63, 127 the BOS key
BOS_COL = 127
MAX_DIST = 63
N_CLASSES = 4
QK_SCALE = 0.25  # 1 / sqrt(DH), exact in binary floating point
V = tk.V
CLASSES = tk.CLASSES
ARTIFACT_FORMAT = "slop.artifact/1"
ARTIFACT_FILES = ("artifact.npz", "vocab.txt", "meta.json")
assert H * DH == D


def tensor_shapes(ids: int = 1) -> "OrderedDict[str, tuple[int, ...]]":
    """Tensor names and shapes in storage order for id layout ``ids``."""
    s: "OrderedDict[str, tuple[int, ...]]" = OrderedDict()
    s["tok_emb.weight"] = (tk.layout(ids).vocab_size, D)
    s["off_emb.weight"] = (O, D)
    s["role_emb.weight"] = (N_ROLES, D)
    s["cluster_emb.weight"] = (N_CLUSTERS, D)
    s["flag_emb.weight"] = (N_FLAGS, D)
    for i in range(L):
        p = f"layers.{i}."
        s[p + "attn.rel_bias"] = (H, NB)
        for name in ("q", "k", "v", "o"):
            s[p + f"attn.{name}.weight"] = (D, D)
            s[p + f"attn.{name}.bias"] = (D,)
        s[p + "mlp.fc1.weight"] = (M, D)
        s[p + "mlp.fc1.bias"] = (M,)
        s[p + "mlp.fc2.weight"] = (D, M)
        s[p + "mlp.fc2.bias"] = (D,)
    s["head.weight"] = (N_CLASSES, D)
    s["head.bias"] = (N_CLASSES,)
    return s


def param_count(ids: int = 1) -> int:
    return sum(math.prod(v) for v in tensor_shapes(ids).values())


TENSOR_SHAPES = tensor_shapes(1)
PARAM_COUNT = param_count(1)
assert PARAM_COUNT == 2_128_900, PARAM_COUNT
assert param_count(2) == 4_029_444, param_count(2)


class ArtifactError(ValueError):
    pass


def zero_tensors(ids: int = 1) -> "OrderedDict[str, np.ndarray]":
    """Every tensor at 0.0 (SPEC.md §7: the starting state), for id layout ``ids``."""
    return OrderedDict((k, np.zeros(s, dtype=np.float32)) for k, s in tensor_shapes(ids).items())


def _check_ids(ids: object) -> int:
    try:
        return tk.layout(ids).version
    except tk.TokenizerError as exc:
        raise ArtifactError(str(exc)) from None


def infer_ids(tensors: "dict[str, np.ndarray]") -> int:
    """The id layout whose ``tok_emb.weight`` row count matches (1 when it matches none; validation then says why)."""
    t = tensors.get("tok_emb.weight")
    rows = t.shape[0] if isinstance(t, np.ndarray) and t.ndim == 2 else None
    for k, lay in tk.LAYOUTS.items():
        if rows == lay.vocab_size:
            return k
    return 1


def validate_tensors(tensors: "dict[str, np.ndarray]", ids: int = 1) -> None:
    """Exact names, shapes (for id layout ``ids``), float32 and finiteness."""
    shapes = tensor_shapes(_check_ids(ids))
    names = set(tensors)
    missing = [k for k in shapes if k not in names]
    extra = sorted(names - set(shapes))
    if missing or extra:
        raise ArtifactError(f"tensor names differ from the template (missing {missing}, extra {extra})")
    for k, shape in shapes.items():
        t = tensors[k]
        if not isinstance(t, np.ndarray):
            raise ArtifactError(f"{k}: not an array")
        if tuple(t.shape) != shape:
            extra_note = f" (id layout {ids})" if k == "tok_emb.weight" else ""
            raise ArtifactError(f"{k}: shape {tuple(t.shape)} != {shape}{extra_note}")
        if t.dtype != np.float32:
            raise ArtifactError(f"{k}: dtype {t.dtype} != float32")
        if not bool(np.isfinite(t).all()):
            raise ArtifactError(f"{k}: non-finite values")


def validate_meta(meta: object) -> tuple[int, int]:
    """``(layer, head)`` of the pointer head; layer is 1-based."""
    if not isinstance(meta, dict) or meta.get("format") != ARTIFACT_FORMAT:
        raise ArtifactError(f"meta.json: format must be {ARTIFACT_FORMAT!r}")
    ph = meta.get("pointer_head")
    if (
        not isinstance(ph, list) or len(ph) != 2 or not all(isinstance(x, int) and not isinstance(x, bool) for x in ph)
        or not 1 <= ph[0] <= L or not 0 <= ph[1] < H
    ):
        raise ArtifactError("meta.json: pointer_head must be [layer 1-4, head 0-7]")
    if "ids" in meta:
        try:
            tk.layout(meta["ids"])
        except tk.TokenizerError as exc:
            raise ArtifactError(f'meta.json: "ids": {exc}') from None
    return ph[0], ph[1]


def make_meta(pointer_head: tuple[int, int], ids: int = 1) -> dict:
    """``meta.json`` content; ``"ids"`` is written only for layout 2, so a layout-1 artefact is unchanged from r1."""
    meta: dict = {"format": ARTIFACT_FORMAT, "pointer_head": [int(pointer_head[0]), int(pointer_head[1])]}
    if _check_ids(ids) != 1:
        meta["ids"] = ids
    validate_meta(meta)
    return meta


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def combine_sha256(npz_sha: str, vocab_sha: str, meta_sha: str) -> str:
    return sha256_hex((npz_sha + vocab_sha + meta_sha).encode("ascii"))


def artifact_sha256(directory: str | Path) -> str:
    d = Path(directory)
    return combine_sha256(*(sha256_hex((d / f).read_bytes()) for f in ARTIFACT_FILES))


def npz_bytes(tensors: "dict[str, np.ndarray]", ids: int | None = None) -> bytes:
    """Deterministic uncompressed npz (fixed member order and timestamps), readable by ``np.load``."""
    ids = infer_ids(tensors) if ids is None else ids
    validate_tensors(tensors, ids)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for k in tensor_shapes(ids):
            arr = io.BytesIO()
            np.lib.format.write_array(arr, np.ascontiguousarray(tensors[k], dtype=np.float32), allow_pickle=False)
            info = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, arr.getvalue())
    return buf.getvalue()


def save_artifact(directory: str | Path, tensors: "dict[str, np.ndarray]", vocab_stems: list[str],
                  pointer_head: tuple[int, int], ids: int | None = None) -> str:
    """Write the three artefact files (byte-for-byte reproducible) and return the artefact sha256.

    ``ids`` is the id layout; None takes it from ``tok_emb.weight``'s row count (10,240 -> 1, 25,088 -> 2)."""
    d = Path(directory)
    ids = infer_ids(tensors) if ids is None else _check_ids(ids)
    meta = make_meta(pointer_head, ids)
    validate_tensors(tensors, ids)
    tk.Vocab(vocab_stems, ids)  # validates
    d.mkdir(parents=True, exist_ok=True)
    (d / "artifact.npz").write_bytes(npz_bytes(tensors, ids))
    (d / "vocab.txt").write_bytes("".join(s + "\n" for s in vocab_stems).encode("utf-8"))
    (d / "meta.json").write_bytes((json.dumps(meta) + "\n").encode("utf-8"))
    return artifact_sha256(d)


@dataclass
class Artifact:
    tensors: "OrderedDict[str, np.ndarray]"  # float32, as stored
    vocab: tk.Vocab
    meta: dict
    pointer_head: tuple[int, int]
    sha256: str | None = None
    vocab_sha256: str | None = None
    _f64: "dict[str, np.ndarray] | None" = None
    ids: int = 1  # id layout (meta.json "ids", absent = 1)

    @property
    def f64(self) -> "dict[str, np.ndarray]":
        if self._f64 is None:
            self._f64 = {k: v.astype(np.float64) for k, v in self.tensors.items()}
        return self._f64


def from_tensors(tensors: "dict[str, np.ndarray]", vocab_stems: list[str], pointer_head: tuple[int, int],
                 ids: int | None = None) -> Artifact:
    """An in-memory artefact (no files, no sha256); ``ids`` as in :func:`save_artifact`."""
    ids = infer_ids(tensors) if ids is None else _check_ids(ids)
    validate_tensors(tensors, ids)
    meta = make_meta(pointer_head, ids)
    ph = validate_meta(meta)
    return Artifact(OrderedDict((k, tensors[k]) for k in tensor_shapes(ids)), tk.Vocab(vocab_stems, ids), meta, ph,
                    ids=ids)


def load_artifact(directory: str | Path) -> Artifact:
    d = Path(directory)
    for f in ARTIFACT_FILES:
        if not (d / f).is_file():
            raise ArtifactError(f"{d / f}: missing")
    raw = {f: (d / f).read_bytes() for f in ARTIFACT_FILES}
    with zipfile.ZipFile(io.BytesIO(raw["artifact.npz"])) as zf:
        for info in zf.infolist():
            if info.compress_type != zipfile.ZIP_STORED:
                raise ArtifactError(f"artifact.npz: {info.filename} is compressed (must be uncompressed)")
    with np.load(io.BytesIO(raw["artifact.npz"]), allow_pickle=False) as z:
        tensors = {k: z[k] for k in z.files}
    try:
        meta = json.loads(raw["meta.json"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"meta.json is not JSON: {exc}") from exc
    ph = validate_meta(meta)
    ids = tk.meta_ids(meta)
    validate_tensors(tensors, ids)
    try:
        vocab = tk.Vocab.from_text(raw["vocab.txt"].decode("utf-8"), ids)
    except (UnicodeDecodeError, tk.TokenizerError) as exc:
        raise ArtifactError(f"vocab.txt: {exc}") from exc
    shas = [sha256_hex(raw[f]) for f in ARTIFACT_FILES]
    return Artifact(
        OrderedDict((k, tensors[k]) for k in tensor_shapes(ids)), vocab, meta, ph, combine_sha256(*shas), shas[1],
        ids=ids,
    )


# -- forward pass ---------------------------------------------------------------------------------------------------


def bias_columns(node: np.ndarray) -> np.ndarray:
    """``c(i, j)`` for every query i and key j of one window."""
    c = np.clip(node[None, :] - node[:, None], -MAX_DIST, MAX_DIST) + MAX_DIST
    c[:, node == 0] = BOS_COL
    return c


def embed(p: "dict[str, np.ndarray]", w: tk.Window) -> np.ndarray:
    tok = np.asarray(w.tokens, dtype=np.int64)
    n = tok.shape[0]
    if n == 0 or tok[0] != tk.BOS or w.node[0] != 0:
        raise ValueError("a window starts with BOS at node 0")
    h = p["tok_emb.weight"][tok].copy()
    rest = slice(1, n)
    h[rest] += p["off_emb.weight"][np.asarray(w.offsets[1:], dtype=np.int64)]
    h[rest] += p["role_emb.weight"][np.asarray(w.roles[1:], dtype=np.int64)]
    h[rest] += p["cluster_emb.weight"][np.asarray(w.clusters[1:], dtype=np.int64) % N_CLUSTERS]
    flags = np.asarray(w.flags, dtype=np.int64)
    for f in range(N_FLAGS):
        sel = np.nonzero((flags >> f) & 1)[0]
        sel = sel[sel > 0]
        if sel.size:
            h[sel] += p["flag_emb.weight"][f]
    return h


@dataclass
class ForwardResult:
    logits: np.ndarray  # [n_nodes, 4] float64, at each node's EON
    pointer_attn: np.ndarray  # [n_nodes, n] pointer head's attention rows at each EON
    collect: "dict | None" = None


def forward(art: Artifact, w: tk.Window, collect: bool = False) -> ForwardResult:
    """One window. ``collect`` keeps the embedding, every residual and every attention map (inspector use)."""
    p = art.f64
    node = np.asarray(w.node, dtype=np.int64)
    n = node.shape[0]
    cols = bias_columns(node)
    h = embed(p, w)
    out: "dict | None" = {"embed": h.copy(), "attn": [], "resid_attn": [], "resid_mlp": []} if collect else None
    ptr_layer, ptr_head = art.pointer_head
    ptr_attn = None
    for li in range(L):
        pre = f"layers.{li}."
        q = (h @ p[pre + "attn.q.weight"].T + p[pre + "attn.q.bias"]).reshape(n, H, DH).transpose(1, 0, 2)
        k = (h @ p[pre + "attn.k.weight"].T + p[pre + "attn.k.bias"]).reshape(n, H, DH).transpose(1, 0, 2)
        v = (h @ p[pre + "attn.v.weight"].T + p[pre + "attn.v.bias"]).reshape(n, H, DH).transpose(1, 0, 2)
        bias = p[pre + "attn.rel_bias"][:, cols]  # [H, n, n]
        score = (q * QK_SCALE) @ k.transpose(0, 2, 1) + bias
        score -= score.max(axis=-1, keepdims=True)
        a = np.exp(score)
        a /= a.sum(axis=-1, keepdims=True)
        ctx = (a @ v).transpose(1, 0, 2).reshape(n, D)
        h = h + (ctx @ p[pre + "attn.o.weight"].T + p[pre + "attn.o.bias"])
        if li == ptr_layer - 1:
            ptr_attn = a[ptr_head][np.asarray(w.eon, dtype=np.int64)]
        if out is not None:
            out["attn"].append(a)
            out["resid_attn"].append(h.copy())
        hid = np.maximum(h @ p[pre + "mlp.fc1.weight"].T + p[pre + "mlp.fc1.bias"], 0.0)
        h = h + (hid @ p[pre + "mlp.fc2.weight"].T + p[pre + "mlp.fc2.bias"])
        if out is not None:
            out["resid_mlp"].append(h.copy())
    eon = np.asarray(w.eon, dtype=np.int64)
    logits = h[eon] @ p["head.weight"].T + p["head.bias"]
    assert ptr_attn is not None
    return ForwardResult(logits, ptr_attn, out)


def argmax_class(logits: np.ndarray) -> int:
    """argmax, ties to the lower index (``np.argmax`` returns the first maximum)."""
    return int(np.argmax(logits))


def pointer_from_attn(row: np.ndarray, node: list[int], self_node: int) -> tuple[int | None, float, np.ndarray]:
    """(target window node or None, BOS weight, per-node sums) for one EON attention row."""
    node_arr = np.asarray(node, dtype=np.int64)
    n_nodes = int(node_arr.max())
    sums = np.zeros(n_nodes + 1, dtype=np.float64)
    np.add.at(sums, node_arr, row)
    bos = float(row[0])
    best = None
    best_val = -1.0
    for m in range(1, n_nodes + 1):
        if m == self_node:
            continue
        if sums[m] > best_val:
            best, best_val = m, sums[m]
    if bos >= 0.5:
        best = None
    return best, bos, sums


@dataclass
class NodePrediction:
    index: int  # screen node list index
    cls: str
    logits: list[float]
    pointer: int | None  # screen node list index of the target (ECHO and NARRATE only)
    bos_weight: float
    window: int


def predict_encoded(art: Artifact, enc: tk.EncodedScreen) -> list[NodePrediction]:
    preds: list[NodePrediction | None] = [None] * len(enc.nodes)
    for wi, w in enumerate(enc.windows):
        res = forward(art, w)
        for m, x in enumerate(w.nodes):
            lg = res.logits[m]
            cls = CLASSES[argmax_class(lg)]
            tgt, bos, _ = pointer_from_attn(res.pointer_attn[m], w.node, m + 1)
            ptr = w.nodes[tgt - 1] if (tgt is not None and cls in ("ECHO", "NARRATE")) else None
            preds[x] = NodePrediction(x, cls, [float(v) for v in lg], ptr, bos, wi)
    return [p for p in preds if p is not None]


def predict_screen(art: Artifact, screen: dict) -> list[NodePrediction]:
    return predict_encoded(art, tk.encode_screen(screen, art.vocab))
