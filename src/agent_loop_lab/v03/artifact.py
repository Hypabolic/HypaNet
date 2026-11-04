"""v0.3 weight artefacts (harness side; not shipped as packet code).

State-dict template and validation for :class:`DoomLoopTransformer`, plus a
small deterministic binary format, ``.dlw`` version ``DLW3``, mirroring the
v0.2 ``DLW1`` format (spec v0.2 §15): identical tensors and metadata give
identical bytes, so a rebuilt artefact can be compared by sha256.

Layout: ``b"DLW3"`` | u32 little-endian header length | canonical-JSON header
| little-endian float32 payload in :data:`PARAM_SHAPES` order. The magic
differs from v0.2's ``DLW1`` so neither loader accepts the other's files.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .model import PARAM_COUNT, PARAM_SHAPES, DoomLoopTransformer, zero_model

MAGIC = b"DLW3"
FORMAT = "DLW3"
SPARSITY_TOLERANCE = 1e-9

__all__ = [
    "MAGIC",
    "SPARSITY_TOLERANCE",
    "state_dict_template",
    "write_template",
    "validate_state_dict",
    "load_state_dict_into",
    "serialize_state_dict",
    "deserialize_state_dict",
    "save_artifact",
    "load_artifact",
    "artifact_sha256",
    "sparsity",
    "index_ranges",
    "nonzero_blocks",
]


def state_dict_template() -> "OrderedDict[str, list[int]]":
    """Parameter name -> shape, in canonical order."""
    return OrderedDict((k, list(v)) for k, v in PARAM_SHAPES.items())


def write_template(path: str | Path) -> None:
    """Write the zero state_dict template (names, shapes, dtype) as canonical JSON."""
    doc = {
        "dtype": "float32",
        "initial_value": 0.0,
        "param_count": PARAM_COUNT,
        "params": [[k, list(v)] for k, v in PARAM_SHAPES.items()],
    }
    Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def validate_state_dict(sd: Mapping[str, torch.Tensor]) -> None:
    """Exact key list (and order), shapes, float32 dtype and finiteness."""
    keys = list(sd.keys())
    if keys != list(PARAM_SHAPES.keys()):
        missing = [k for k in PARAM_SHAPES if k not in sd]
        extra = [k for k in keys if k not in PARAM_SHAPES]
        raise ValueError(f"state_dict keys differ from the template (missing {missing}, extra {extra}, or order)")
    for k, shape in PARAM_SHAPES.items():
        t = sd[k]
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{k}: not a tensor")
        if tuple(t.shape) != shape:
            raise ValueError(f"{k}: shape {tuple(t.shape)} != {shape}")
        if t.dtype != torch.float32:
            raise ValueError(f"{k}: dtype {t.dtype} != float32")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{k}: non-finite values")


def load_state_dict_into(model: DoomLoopTransformer, sd: Mapping[str, torch.Tensor]) -> DoomLoopTransformer:
    validate_state_dict(sd)
    with torch.no_grad():
        model.load_state_dict(OrderedDict((k, sd[k].detach().clone()) for k in PARAM_SHAPES))
    return model


def serialize_state_dict(sd: Mapping[str, torch.Tensor], metadata: Mapping[str, object] | None = None) -> bytes:
    validate_state_dict(sd)
    header = {
        "format": FORMAT,
        "tensors": [[k, list(PARAM_SHAPES[k])] for k in PARAM_SHAPES],
        "metadata": dict(metadata or {}),
    }
    hbytes = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload = b"".join(
        sd[k].detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")
        for k in PARAM_SHAPES
    )
    return MAGIC + struct.pack("<I", len(hbytes)) + hbytes + payload


def deserialize_state_dict(data: bytes) -> tuple["OrderedDict[str, torch.Tensor]", dict]:
    """Inverse of :func:`serialize_state_dict`. Every malformed input raises ``ValueError``."""
    if data[:4] != MAGIC:
        raise ValueError("not a DLW3 (v0.3) artefact")
    if len(data) < 8:
        raise ValueError("truncated DLW3 artefact (no header length)")
    (hlen,) = struct.unpack("<I", data[4:8])
    if 8 + hlen > len(data):
        raise ValueError("truncated DLW3 artefact (header extends past end)")
    try:
        header = json.loads(data[8 : 8 + hlen].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"DLW3 header is not JSON: {exc}") from exc
    if not isinstance(header, dict) or header.get("format") != FORMAT:
        raise ValueError("DLW3 header has the wrong format tag")
    expected = [[k, list(v)] for k, v in PARAM_SHAPES.items()]
    if header.get("tensors") != expected:
        raise ValueError("DLW3 header tensor list differs from the v0.3 template")
    offset = 8 + hlen
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, shape in expected:
        n = math.prod(shape)
        if offset + 4 * n > len(data):
            raise ValueError("truncated DLW3 artefact (payload)")
        arr = np.frombuffer(data, dtype="<f4", count=n, offset=offset).reshape(shape).astype(np.float32)
        out[name] = torch.from_numpy(arr.copy())
        offset += 4 * n
    if offset != len(data):
        raise ValueError("trailing bytes in DLW3 artefact")
    validate_state_dict(out)
    meta = header.get("metadata", {})
    return out, meta if isinstance(meta, dict) else {}


def save_artifact(model: DoomLoopTransformer, path: str | Path, metadata: Mapping[str, object] | None = None) -> str:
    """Write a ``.dlw`` (DLW3) artefact and return its sha256 hex digest."""
    data = serialize_state_dict(model.state_dict(), metadata)
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def load_artifact(path: str | Path) -> tuple[DoomLoopTransformer, dict]:
    sd, meta = deserialize_state_dict(Path(path).read_bytes())
    return load_state_dict_into(zero_model(), sd), meta


def artifact_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- statistics -------------------------------------------------------------


def sparsity(model: DoomLoopTransformer, tol: float = SPARSITY_TOLERANCE) -> dict[str, dict[str, float | int]]:
    """Non-zero counts (``|w| > tol``) per tensor and overall, over PARAM_COUNT slots."""
    sd = model.state_dict()
    report: dict[str, dict[str, float | int]] = {}
    total = 0
    for k in PARAM_SHAPES:
        t = sd[k]
        nz = int((t.abs() > tol).sum())
        total += nz
        report[k] = {"nonzero": nz, "slots": t.numel(), "density": nz / t.numel()}
    report["overall"] = {"nonzero": total, "slots": PARAM_COUNT, "density": total / PARAM_COUNT}
    return report


def index_ranges(indices: "list[int]") -> "list[tuple[int, int]]":
    """Sorted indices -> inclusive ``(start, end)`` runs."""
    runs: list[tuple[int, int]] = []
    for i in indices:
        if runs and i == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], i)
        else:
            runs.append((i, i))
    return runs


def format_ranges(runs: "list[tuple[int, int]]") -> str:
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs) or "-"


def nonzero_blocks(t: torch.Tensor, tol: float = SPARSITY_TOLERANCE) -> "list[dict[str, object]]":
    """Describe where a tensor is non-zero.

    1-D: one block with the non-zero index runs. 2-D: consecutive rows with an
    identical non-zero column set are merged into one block ``{rows, cols,
    nonzero}``; all-zero rows are omitted.
    """
    nz = t.detach().abs() > tol
    if nz.ndim == 1:
        idx = nz.nonzero().flatten().tolist()
        return [{"rows": None, "cols": index_ranges(idx), "nonzero": len(idx)}] if idx else []
    if nz.ndim != 2:
        raise ValueError("nonzero_blocks expects a 1-D or 2-D tensor")
    blocks: list[dict[str, object]] = []
    rows = nz.any(dim=1).nonzero().flatten().tolist()
    prev_row = None
    prev_cols = None
    for r in rows:
        cols = tuple(nz[r].nonzero().flatten().tolist())
        if blocks and prev_row == r - 1 and cols == prev_cols:
            blk = blocks[-1]
            blk["rows"] = (blk["rows"][0], r)  # type: ignore[index]
            blk["nonzero"] = int(blk["nonzero"]) + len(cols)  # type: ignore[arg-type]
        else:
            blocks.append({"rows": (r, r), "cols": index_ranges(list(cols)), "nonzero": len(cols)})
        prev_row, prev_cols = r, cols
    return blocks
