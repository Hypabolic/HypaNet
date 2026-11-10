"""Compile an artefact directory into ``slop-net.bin`` for the JS runtime (SPEC.md §7).

Layout::

    b"SLOPNET1" | u32 little-endian header length | JSON header (UTF-8, padded with spaces so the data section
    starts at a multiple of 8) | float32 little-endian tensor data | vocab.txt bytes

The header (sorted keys, no extra white space before the padding) holds ``format`` (``slop.net/1``),
``artifact_sha256``, ``vocab_sha256``, ``pointer_head``, ``classes``, ``param_count``, ``tensors`` and ``vocab``.
Each tensor entry gives ``name``, ``shape``, ``rows`` (half-open ``[start, end)`` runs of the rows that are stored;
rows outside them are all zero; a 1-D tensor counts as one row), ``offset`` and ``nbytes`` (relative to the start of
the data section). ``vocab`` gives the offset and length of the vocabulary bytes, also relative to the data section.
Tensors appear in ``ui/model.py`` ``TENSOR_SHAPES`` order. The same artefact always compiles to the same bytes.

    uv run python lint/compile.py ART_DIR [-o slop-net.bin]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui import model as mdl  # noqa: E402

MAGIC = b"SLOPNET1"
FORMAT = "slop.net/1"


def _row_runs(t: np.ndarray) -> list[list[int]]:
    rows = t.reshape(1, -1) if t.ndim == 1 else t
    nz = np.nonzero(np.any(rows != 0, axis=1))[0].tolist()
    runs: list[list[int]] = []
    for r in nz:
        if runs and runs[-1][1] == r:
            runs[-1][1] = r + 1
        else:
            runs.append([r, r + 1])
    return runs


def compile_artifact(art: mdl.Artifact, vocab_bytes: bytes) -> bytes:
    if art.sha256 is None or art.vocab_sha256 is None:
        raise ValueError("compile a loaded artefact (it needs its sha256)")
    entries = []
    chunks: list[bytes] = []
    offset = 0
    for name, shape in mdl.TENSOR_SHAPES.items():
        t = art.tensors[name]
        rows2d = t.reshape(1, -1) if t.ndim == 1 else t
        runs = _row_runs(t)
        data = b"".join(np.ascontiguousarray(rows2d[a:b], dtype="<f4").tobytes() for a, b in runs)
        entries.append({"name": name, "shape": list(shape), "rows": runs, "offset": offset, "nbytes": len(data)})
        chunks.append(data)
        offset += len(data)
    header = {
        "format": FORMAT,
        "artifact_sha256": art.sha256,
        "vocab_sha256": art.vocab_sha256,
        "pointer_head": list(art.pointer_head),
        "classes": list(mdl.CLASSES),
        "param_count": mdl.PARAM_COUNT,
        "tensors": entries,
        "vocab": {"offset": offset, "nbytes": len(vocab_bytes)},
    }
    hbytes = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    pad = (-(len(MAGIC) + 4 + len(hbytes))) % 8
    hbytes += b" " * pad
    return MAGIC + struct.pack("<I", len(hbytes)) + hbytes + b"".join(chunks) + vocab_bytes


def read_net(data: bytes) -> tuple[dict, "dict[str, np.ndarray]", bytes]:
    """Inverse of :func:`compile_artifact`: (header, float32 tensors, vocab bytes)."""
    if data[: len(MAGIC)] != MAGIC:
        raise ValueError("not a SLOPNET1 file")
    (hlen,) = struct.unpack("<I", data[8:12])
    header = json.loads(data[12 : 12 + hlen].decode("utf-8"))
    if header.get("format") != FORMAT:
        raise ValueError("wrong slop-net format tag")
    base = 12 + hlen
    tensors = {}
    for e in header["tensors"]:
        shape = tuple(e["shape"])
        rows = (1, shape[0]) if len(shape) == 1 else shape
        full = np.zeros(rows, dtype=np.float32)
        pos = base + e["offset"]
        for a, b in e["rows"]:
            n = (b - a) * rows[1]
            full[a:b] = np.frombuffer(data, dtype="<f4", count=n, offset=pos).reshape(b - a, rows[1])
            pos += 4 * n
        if pos != base + e["offset"] + e["nbytes"]:
            raise ValueError(f"{e['name']}: byte count mismatch")
        tensors[e["name"]] = full.reshape(shape)
    v = header["vocab"]
    vocab = data[base + v["offset"] : base + v["offset"] + v["nbytes"]]
    if base + v["offset"] + v["nbytes"] != len(data):
        raise ValueError("trailing bytes after the vocabulary")
    return header, tensors, vocab


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compile an artefact directory into slop-net.bin")
    ap.add_argument("artifact", help="directory holding artifact.npz, vocab.txt and meta.json")
    ap.add_argument("-o", "--out", default="slop-net.bin")
    args = ap.parse_args(argv)
    art = mdl.load_artifact(args.artifact)
    vocab_bytes = (Path(args.artifact) / "vocab.txt").read_bytes()
    data = compile_artifact(art, vocab_bytes)
    Path(args.out).write_bytes(data)
    stored = sum(int(np.count_nonzero(np.any((t.reshape(1, -1) if t.ndim == 1 else t) != 0, axis=1))) for t in art.tensors.values())
    print(f"{args.out}: {len(data)} bytes, artefact {art.sha256[:16]}, {stored} non-zero rows stored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
