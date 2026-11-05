"""v0.4 task export (plan v0.4, Task interface). Harness only; runs in the frozen v0.3 environment.

Writes, for one split, the two files the Qwen environment reads:

* ``datasets/v0_4/{split}.tasks.jsonl.gz``: one line per session, in v0.2 line order,
  ``{"session_id", "blocks": [rendered block text...], "window_pos": [...]}``. Blocks are the
  frozen v0.3 renderer's output, unchanged. ``window_pos`` lists the block positions that end a
  window (``windows.window_endpoints``). No labels, families, gate results or oracle fields: this
  file is safe to hand to a component that must not see labels.
* ``datasets/v0_4/{split}.labels.npz``: the window rows, grouped by session and in the same order
  as ``window_pos``: ``row_session``, ``row_pos``, ``row_endpoint``, ``row_label`` (index into
  ``OracleLabel``; scored labels have the class index), ``row_gate``, plus ``session_ids`` and
  ``families``.

The rows are built exactly as ``v03.scoring._build_val`` and ``baselines.phase_c_v03._build_dev``
build them, so a Qwen column is scored on the same windows as every v0.3 column.

**Splits.** Only ``dev`` and ``val`` (:data:`ALLOWED_SPLITS`). The check runs before any dataset
module is imported or any file is opened. v0.4 exports only dev and val: test, counterfactual and
OOD are not exported in v0.4 and stay sealed for a later version (PREREGISTRATION §3.4).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import CLASS_NAMES, OracleLabel
from ..v03 import renderer

__all__ = ["ALLOWED_SPLITS", "EXPORT_FORMAT", "V04_DATA", "check_split", "export_split", "tasks_path", "labels_path"]

ALLOWED_SPLITS = frozenset({"dev", "val"})
EXPORT_FORMAT = "v04-task-export/1"

REPO = Path(__file__).resolve().parents[3]
V03_DATA = REPO / "datasets" / "v0_3"
V02_DATA = REPO / "datasets" / "v0_2"
V04_DATA = REPO / "datasets" / "v0_4"
_LABELS = tuple(OracleLabel)
assert tuple(l.name for l in _LABELS[: len(CLASS_NAMES)]) == CLASS_NAMES  # label index == class index when scored


def check_split(split: str) -> None:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"v0.4 export may read only {sorted(ALLOWED_SPLITS)}, not {split!r}")


def tasks_path(split: str, out_dir: Path = V04_DATA) -> Path:
    return out_dir / f"{split}.tasks.jsonl.gz"


def labels_path(split: str, out_dir: Path = V04_DATA) -> Path:
    return out_dir / f"{split}.labels.npz"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _input_hashes(split: str) -> dict[str, str]:
    return {
        "renderer_sha256": _sha256_file(Path(renderer.__file__)),
        f"datasets/v0_3/{split}.raw.jsonl.gz": _sha256_file(V03_DATA / f"{split}.raw.jsonl.gz"),
        f"datasets/v0_2/standard/{split}.events.jsonl.gz": _sha256_file(V02_DATA / "standard" / f"{split}.events.jsonl.gz"),
        f"datasets/v0_2/standard/{split}.oracle.jsonl.gz": _sha256_file(V02_DATA / "standard" / f"{split}.oracle.jsonl.gz"),
        "datasets/v0_3/MANIFEST_v0_3.json": _sha256_file(V03_DATA / "MANIFEST_v0_3.json"),
    }


def _check_manifest(split: str, hashes: dict[str, str]) -> None:
    man = json.loads((V03_DATA / "MANIFEST_v0_3.json").read_text())
    if man.get("renderer_source_sha256") != hashes["renderer_sha256"]:
        raise ValueError(f"renderer {hashes['renderer_sha256']} differs from MANIFEST_v0_3.json")
    raw = f"{split}.raw.jsonl.gz"
    want = man.get("files", {}).get(raw, {}).get("sha256")
    if want != hashes[f"datasets/v0_3/{raw}"]:
        raise ValueError(f"datasets/v0_3/{raw} sha256 differs from MANIFEST_v0_3.json ({want})")


def export_split(split: str, *, out_dir: Path = V04_DATA, limit: int | None = None) -> dict[str, Any]:
    """Write the split's tasks and labels files; return their manifest entry.

    ``limit`` (tests only) keeps the first ``limit`` sessions.
    """
    check_split(split)
    hashes = _input_hashes(split)
    _check_manifest(split, hashes)
    from ..datasets.build import iter_sessions  # imported only after the split check
    from ..v03.build_raw import alignment_errors, iter_raw
    from ..windows import gate, validate_sequence, window_endpoints

    out_dir.mkdir(parents=True, exist_ok=True)
    tpath, lpath = tasks_path(split, out_dir), labels_path(split, out_dir)
    ids: list[str] = []
    fams: list[str] = []
    r_sess, r_pos, r_end, r_lab, r_gate = [], [], [], [], []
    n_blocks = 0
    tmp = tpath.with_suffix(".tmp")
    # mtime=0 and no file name in the header, as the v0.3 raw files, so a rebuild is byte-identical.
    with open(tmp, "wb") as fb, gzip.GzipFile(filename="", mode="wb", fileobj=fb, compresslevel=6, mtime=0) as gz:
        pairs = zip(iter_raw(V03_DATA, split), iter_sessions(V02_DATA, split), strict=True)
        for n, ((sid, raw), (events, record)) in enumerate(pairs):
            if limit is not None and n >= limit:
                break
            if sid != record.session_id or len(raw) != len(events):
                raise ValueError(f"{split} line {n + 1}: raw/events/oracle misaligned for {sid!r}")
            for k, (r, e) in enumerate(zip(raw, events)):
                errs = alignment_errors(r, e)
                if errs:
                    raise ValueError(f"{sid} event {k}: raw op misaligned: " + "; ".join(errs))
            blocks = renderer.render_session([r.to_dict() for r in raw], sid)
            validate_sequence(events)
            s = len(ids)
            ids.append(sid)
            fams.append(record.scenario_family)
            pos_list = window_endpoints(events)
            for pos in pos_list:
                endpoint = events[pos].sequence_index
                try:
                    label = OracleLabel(record.window_labels[endpoint])
                except KeyError:
                    raise ValueError(f"{sid}: no oracle label for window endpoint {endpoint}") from None
                r_sess.append(s), r_pos.append(pos), r_end.append(endpoint)
                r_lab.append(_LABELS.index(label)), r_gate.append(gate(events[: pos + 1]).passed)
            line = {"session_id": sid, "blocks": blocks, "window_pos": pos_list}
            gz.write((json.dumps(line, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode())
            n_blocks += len(blocks)
    tmp.replace(tpath)
    arrays = {
        "session_ids": np.array(ids),
        "families": np.array(fams),
        "row_session": np.array(r_sess, dtype=np.int32),
        "row_pos": np.array(r_pos, dtype=np.int32),
        "row_endpoint": np.array(r_end, dtype=np.int32),
        "row_label": np.array(r_lab, dtype=np.int8),
        "row_gate": np.array(r_gate, dtype=bool),
    }
    ltmp = lpath.with_name(lpath.name + ".tmp")
    with open(ltmp, "wb") as f:
        np.savez(f, **arrays)
    ltmp.replace(lpath)
    scored = arrays["row_gate"] & (arrays["row_label"] < len(CLASS_NAMES))
    return {
        "format": EXPORT_FORMAT,
        "split": split,
        "limit": limit,
        "sessions": len(ids),
        "blocks": n_blocks,
        "windows": len(r_sess),
        "scored_windows": int(scored.sum()),
        "inputs": hashes,
        "files": {
            tpath.name: _sha256_file(tpath),
            # npz holds zip timestamps, so its hash is recorded but not expected to repeat across builds.
            lpath.name: _sha256_file(lpath),
        },
        "labels_content_sha256": {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                                  for k, v in arrays.items() if k.startswith("row_")},
    }
