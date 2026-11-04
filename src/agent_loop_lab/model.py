"""Immutable target MLP (spec v0.2 §2.2).

``Linear(120, 24) -> ReLU -> Linear(24, 12) -> ReLU -> Linear(12, 4)``,
float32, CPU. Raw logits in canonical order
``[PRODUCTIVE, RECOVERY, STALLED, OSCILLATING]``. No softmax in the model.

Artefacts use a small deterministic binary format (``.dlw``) instead of
``torch.save`` so that rebuilding identical weights yields byte-identical files
(spec §15).
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from .schema import CLASS_NAMES, INPUT_DIM

HIDDEN1 = 24
HIDDEN2 = 12
OUTPUT_DIM = 4

PARAM_SHAPES: "OrderedDict[str, tuple[int, ...]]" = OrderedDict(
    [
        ("fc1.weight", (HIDDEN1, INPUT_DIM)),
        ("fc1.bias", (HIDDEN1,)),
        ("fc2.weight", (HIDDEN2, HIDDEN1)),
        ("fc2.bias", (HIDDEN2,)),
        ("fc3.weight", (OUTPUT_DIM, HIDDEN2)),
        ("fc3.bias", (OUTPUT_DIM,)),
    ]
)
PARAM_COUNT = sum(int(np.prod(s)) for s in PARAM_SHAPES.values())
assert PARAM_COUNT == 3256, PARAM_COUNT
assert len(CLASS_NAMES) == OUTPUT_DIM

SPARSITY_TOLERANCE = 1e-9

_MAGIC = b"DLW1"


class DoomLoopMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc3(h2)

    def activations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Layer-by-layer activations for the inspector (no weight changes)."""
        h1_pre = self.fc1(x)
        h1 = torch.relu(h1_pre)
        h2_pre = self.fc2(h1)
        h2 = torch.relu(h2_pre)
        logits = self.fc3(h2)
        return {"h1_pre": h1_pre, "h1": h1, "h2_pre": h2_pre, "h2": h2, "logits": logits}


def zero_model(requires_grad: bool = False) -> DoomLoopMLP:
    """Model with every parameter exactly 0.0 (spec §2.2 initial state).

    The direct-weight path uses ``requires_grad=False``. Phase C training may
    pass ``True``; it is the only caller allowed to.
    """
    # nn.Linear's default init draws from the process-global torch RNG. Every
    # parameter is overwritten with zeros, so isolate construction from the
    # global stream: building or loading a model must not change the RNG state
    # seen by callers (spec §9.5 three seeds, §11 repeatability).
    with torch.random.fork_rng(devices=[]):
        model = DoomLoopMLP()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
            p.requires_grad_(requires_grad)
    return model.to(dtype=torch.float32, device="cpu")


def predict(logits: torch.Tensor) -> torch.Tensor:
    """argmax over the class dimension; ties resolve to the first maximal index.

    An all-zero logit vector therefore predicts PRODUCTIVE (index 0).
    """
    return torch.argmax(logits, dim=-1)


def predict_classes(model: DoomLoopMLP, x: np.ndarray | torch.Tensor) -> np.ndarray:
    t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if t.shape[-1] != INPUT_DIM:
        raise ValueError(f"expected last dim {INPUT_DIM}, got {tuple(t.shape)}")
    with torch.no_grad():
        return predict(model(t)).cpu().numpy()


def state_dict_template() -> "OrderedDict[str, list[int]]":
    return OrderedDict((k, list(v)) for k, v in PARAM_SHAPES.items())


def validate_state_dict(sd: Mapping[str, torch.Tensor]) -> None:
    """Exact key set, shapes, float32 dtype and finiteness."""
    if list(sd.keys()) != list(PARAM_SHAPES.keys()):
        raise ValueError(f"state_dict keys {list(sd.keys())} != {list(PARAM_SHAPES.keys())}")
    for k, shape in PARAM_SHAPES.items():
        t = sd[k]
        if tuple(t.shape) != shape:
            raise ValueError(f"{k}: shape {tuple(t.shape)} != {shape}")
        if t.dtype != torch.float32:
            raise ValueError(f"{k}: dtype {t.dtype} != float32")
        if not torch.isfinite(t).all():
            raise ValueError(f"{k}: non-finite values")


def load_state_dict_into(model: DoomLoopMLP, sd: Mapping[str, torch.Tensor]) -> DoomLoopMLP:
    validate_state_dict(sd)
    with torch.no_grad():
        model.load_state_dict(OrderedDict((k, sd[k].clone()) for k in PARAM_SHAPES))
    return model


# -- deterministic serialisation -------------------------------------------


def serialize_state_dict(sd: Mapping[str, torch.Tensor], metadata: Mapping[str, object] | None = None) -> bytes:
    """``DLW1`` | u32 header_len | canonical-JSON header | little-endian float32 payload.

    The header records tensor names/shapes in canonical order and optional
    JSON-serialisable metadata. Identical tensors + metadata => identical bytes.
    """
    validate_state_dict(sd)
    header = {
        "format": "DLW1",
        "tensors": [[k, list(PARAM_SHAPES[k])] for k in PARAM_SHAPES],
        "metadata": dict(metadata or {}),
    }
    hbytes = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload = b"".join(
        sd[k].detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes(order="C") for k in PARAM_SHAPES
    )
    return _MAGIC + struct.pack("<I", len(hbytes)) + hbytes + payload


def deserialize_state_dict(data: bytes) -> tuple["OrderedDict[str, torch.Tensor]", dict]:
    """Inverse of :func:`serialize_state_dict`. Every malformed input raises ``ValueError``."""
    if data[:4] != _MAGIC:
        raise ValueError("not a DLW1 artefact")
    if len(data) < 8:
        raise ValueError("truncated DLW1 artefact (no header length)")
    (hlen,) = struct.unpack("<I", data[4:8])
    if 8 + hlen > len(data):
        raise ValueError("truncated DLW1 artefact (header extends past end)")
    header = json.loads(data[8 : 8 + hlen].decode("utf-8"))
    offset = 8 + hlen
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, shape in header["tensors"]:
        n = int(np.prod(shape))
        arr = np.frombuffer(data, dtype="<f4", count=n, offset=offset).reshape(shape).astype(np.float32)
        out[name] = torch.from_numpy(arr.copy())
        offset += 4 * n
    if offset != len(data):
        raise ValueError("trailing bytes in DLW1 artefact")
    validate_state_dict(out)
    return out, header.get("metadata", {})


def save_artifact(model: DoomLoopMLP, path: str | Path, metadata: Mapping[str, object] | None = None) -> str:
    """Write a ``.dlw`` artefact and return its sha256 hex digest."""
    data = serialize_state_dict(model.state_dict(), metadata)
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def load_artifact(path: str | Path) -> tuple[DoomLoopMLP, dict]:
    sd, meta = deserialize_state_dict(Path(path).read_bytes())
    model = load_state_dict_into(zero_model(), sd)
    return model, meta


def artifact_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- statistics -------------------------------------------------------------


def sparsity(model: DoomLoopMLP, tol: float = SPARSITY_TOLERANCE) -> dict[str, dict[str, float | int]]:
    """Non-zero counts (``|w| > tol``) per tensor and overall, over PARAM_COUNT slots."""
    sd = model.state_dict()
    report: dict[str, dict[str, float | int]] = {}
    total_nz = 0
    for k in PARAM_SHAPES:
        t = sd[k]
        nz = int((t.abs() > tol).sum().item())
        total_nz += nz
        report[k] = {"nonzero": nz, "slots": t.numel(), "density": nz / t.numel()}
    report["overall"] = {"nonzero": total_nz, "slots": PARAM_COUNT, "density": total_nz / PARAM_COUNT}
    return report
