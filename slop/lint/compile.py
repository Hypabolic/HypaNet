"""Compile an artefact directory into ``slop-net.bin`` for the JS runtime (SPEC.md §7).

Layout::

    b"SLOPNET1" | u32 little-endian header length | JSON header (UTF-8, padded with spaces so the data section
    starts at a multiple of 8) | tensor data | vocab.txt bytes

The header (sorted keys, no extra white space before the padding) holds ``format``, ``ids`` (the artefact's id
layout, 1 or 2), ``artifact_sha256``, ``vocab_sha256``, ``pointer_head``, ``classes``, ``param_count``, ``tensors``
and ``vocab``. Tensors appear in ``ui/model.py`` ``tensor_shapes(ids)`` order; each entry gives ``name``, ``shape``,
``offset`` and ``nbytes`` (relative to the start of the data section) and how its values are stored (a 1-D tensor
counts as one row; everything not stored is zero):

* ``"enc": "rows"`` with ``rows``: half-open ``[start, end)`` runs of the stored rows, whose float32 little-endian
  values follow in order.
* ``"enc": "cells"`` with ``nrows``: the stored rows' indices (``nrows`` u32 little-endian, ascending), then one
  column mask per stored row (``ceil(cols / 8)`` bytes; bit ``c & 7`` of byte ``c >> 3`` is column c), then the
  float32 little-endian values of the set bits, row by row and column by column.

A row or cell is stored when its float32 bit pattern is not all zero (so a ``-0.0`` survives). The compiler picks,
per tensor, ``cells`` when its data is strictly smaller than ``rows``'s, else ``rows``. ``vocab`` gives the offset
and length of the vocabulary bytes, also relative to the data section. The same artefact always compiles to the
same bytes.

Format ``slop.net/2`` is written; ``slop.net/1`` (r1's compiler: no ``ids``, so layout 1, no ``enc``, every tensor
in ``rows`` form with rows chosen by ``!= 0``) is still read here and by ``lint/runtime.js``.

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
FORMAT = "slop.net/2"
FORMAT_V1 = "slop.net/1"
FORMATS = (FORMAT_V1, FORMAT)


def _as_rows(t: np.ndarray) -> np.ndarray:
    return t.reshape(1, -1) if t.ndim == 1 else t


def _row_runs(mask: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    for r in np.nonzero(mask)[0].tolist():
        if runs and runs[-1][1] == r:
            runs[-1][1] = r + 1
        else:
            runs.append([r, r + 1])
    return runs


def encode_tensor(name: str, t: np.ndarray) -> tuple[dict, bytes]:
    """(header entry without ``offset``, data bytes) for one tensor, in the smaller of the two encodings."""
    rows = np.ascontiguousarray(_as_rows(t), dtype="<f4")
    cols = rows.shape[1]
    live = rows.view("<u4") != 0  # bit pattern, so -0.0 is kept
    row_live = live.any(axis=1)
    stored = np.nonzero(row_live)[0]
    n_cells = int(live.sum())
    mask_bytes = (cols + 7) // 8
    rows_size = 4 * cols * len(stored)
    cells_size = 4 * len(stored) + mask_bytes * len(stored) + 4 * n_cells
    if cells_size < rows_size:
        sub = live[stored]
        masks = np.packbits(sub, axis=1, bitorder="little")
        data = (stored.astype("<u4").tobytes() + masks.tobytes() + rows[stored][sub].astype("<f4").tobytes())
        entry = {"name": name, "shape": list(t.shape), "enc": "cells", "nrows": int(len(stored)), "nbytes": len(data)}
    else:
        runs = _row_runs(row_live)
        data = b"".join(rows[a:b].tobytes() for a, b in runs)
        entry = {"name": name, "shape": list(t.shape), "enc": "rows", "rows": runs, "nbytes": len(data)}
    return entry, data


def compile_artifact(art: mdl.Artifact, vocab_bytes: bytes) -> bytes:
    if art.sha256 is None or art.vocab_sha256 is None:
        raise ValueError("compile a loaded artefact (it needs its sha256)")
    entries = []
    chunks: list[bytes] = []
    offset = 0
    for name in mdl.tensor_shapes(art.ids):
        entry, data = encode_tensor(name, art.tensors[name])
        entry["offset"] = offset
        entries.append(entry)
        chunks.append(data)
        offset += len(data)
    header = {
        "format": FORMAT,
        "ids": art.ids,
        "artifact_sha256": art.sha256,
        "vocab_sha256": art.vocab_sha256,
        "pointer_head": list(art.pointer_head),
        "classes": list(mdl.CLASSES),
        "param_count": mdl.param_count(art.ids),
        "tensors": entries,
        "vocab": {"offset": offset, "nbytes": len(vocab_bytes)},
    }
    hbytes = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    pad = (-(len(MAGIC) + 4 + len(hbytes))) % 8
    hbytes += b" " * pad
    return MAGIC + struct.pack("<I", len(hbytes)) + hbytes + b"".join(chunks) + vocab_bytes


def read_net(data: bytes) -> tuple[dict, "dict[str, np.ndarray]", bytes]:
    """Inverse of :func:`compile_artifact` (either format): (header, float32 tensors, vocab bytes)."""
    if data[: len(MAGIC)] != MAGIC:
        raise ValueError("not a SLOPNET1 file")
    (hlen,) = struct.unpack("<I", data[8:12])
    header = json.loads(data[12 : 12 + hlen].decode("utf-8"))
    if header.get("format") not in FORMATS:
        raise ValueError("wrong slop-net format tag")
    ids = header.get("ids", 1) if header["format"] == FORMAT else 1
    try:
        shapes = mdl.tensor_shapes(ids)
    except mdl.tk.TokenizerError as exc:
        raise ValueError(f"header ids: {exc}") from None
    if [e["name"] for e in header["tensors"]] != list(shapes):
        raise ValueError("tensor list differs from the template")
    base = 12 + hlen
    tensors = {}
    for e in header["tensors"]:
        shape = tuple(e["shape"])
        if shape != shapes[e["name"]]:
            raise ValueError(f"{e['name']}: shape {shape} != {shapes[e['name']]} (id layout {ids})")
        rows = (1, shape[0]) if len(shape) == 1 else shape
        full = np.zeros(rows, dtype=np.float32)
        pos = base + e["offset"]
        enc = e.get("enc", "rows")
        if enc == "rows":
            for a, b in e["rows"]:
                n = (b - a) * rows[1]
                full[a:b] = np.frombuffer(data, dtype="<f4", count=n, offset=pos).reshape(b - a, rows[1])
                pos += 4 * n
        elif enc == "cells" and header["format"] == FORMAT:
            k = e["nrows"]
            idx = np.frombuffer(data, dtype="<u4", count=k, offset=pos).astype(np.int64)
            pos += 4 * k
            mb = (rows[1] + 7) // 8
            masks = np.frombuffer(data, dtype=np.uint8, count=k * mb, offset=pos).reshape(k, mb)
            pos += k * mb
            live = np.unpackbits(masks, axis=1, count=rows[1], bitorder="little").astype(bool)
            if k and (np.any(np.diff(idx) <= 0) or idx[-1] >= rows[0]):
                raise ValueError(f"{e['name']}: row index not ascending or out of range")
            n = int(live.sum())
            sub = np.zeros((k, rows[1]), dtype=np.float32)
            sub[live] = np.frombuffer(data, dtype="<f4", count=n, offset=pos)
            pos += 4 * n
            full[idx] = sub
        else:
            raise ValueError(f"{e['name']}: unknown encoding {enc!r}")
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
    tok = art.tensors["tok_emb.weight"]
    print(f"{args.out}: {len(data)} bytes ({FORMAT}, id layout {art.ids}), artefact {art.sha256[:16]}, "
          f"tok_emb {int(np.any(tok != 0, axis=1).sum())} non-zero rows, {int(np.count_nonzero(tok))} non-zero cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
