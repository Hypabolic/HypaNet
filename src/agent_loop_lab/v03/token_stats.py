"""Token statistics and renderer/tokeniser hashes for ``MANIFEST_v0_3.json`` (spec v0.3 §7.2).

Harness only. Runs after :func:`agent_loop_lab.v03.build_raw.build_raw`, which is renderer-independent:
it reads the built raw files, renders and encodes **dev and val only**, and amends the manifest in place
with ``renderer_source_sha256``, ``tokeniser_source_sha256`` and ``token_statistics``. Test,
counterfactual and OOD are never rendered here; for them the manifest records only that the file exists
and its sha256. Nothing here reads labels or oracle records.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..datasets.build import _write_json, file_sha256
from . import renderer, tokenizer
from .build_raw import MANIFEST_V03, RAW_SPLITS, iter_raw, raw_path

__all__ = ["STATS_SPLITS", "iter_encoded", "split_token_statistics", "source_sha256", "amend_manifest"]

STATS_SPLITS: tuple[str, ...] = ("dev", "val")


def source_sha256(module: Any) -> str:
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def iter_encoded(out_dir: str | Path, split: str) -> Iterator[tuple[str, list, list[str], tokenizer.Encoded]]:
    """``(session_id, raw ops, rendered blocks, encoding)`` per session of a dev or val raw file."""
    if split not in STATS_SPLITS:
        raise ValueError(f"rendering is allowed on {STATS_SPLITS} only, not {split!r}")
    for sid, raw in iter_raw(out_dir, split):
        blocks = renderer.render_session([r.to_dict() for r in raw], sid)
        yield sid, raw, blocks, tokenizer.encode_session(blocks)


def _dist(values: list[int]) -> dict[str, float | int]:
    a = np.asarray(values, dtype=np.int64)
    return {"mean": round(float(a.mean()), 2), "p50": round(float(np.percentile(a, 50)), 2),
            "p99": round(float(np.percentile(a, 99)), 2), "max": int(a.max())}


def split_token_statistics(out_dir: str | Path, split: str) -> dict[str, Any]:
    """Tokens per session (BOS and every EOE included) and per event block (EOE included), plus the
    largest header, body-line count and body-line length seen (spec §4.6 caps: 64, 6, 21)."""
    per_session: list[int] = []
    per_block: list[int] = []
    max_header = max_lines = max_line = 0
    trunc_blocks = 0
    for _sid, _raw, blocks, enc in iter_encoded(out_dir, split):
        per_session.append(len(enc.tokens))
        prev = 0
        for e in enc.eoe_index:
            per_block.append(e - prev)
            prev = e
        for b in blocks:
            lines = b.split("\n")
            max_header = max(max_header, len(tokenizer.tokenize_block(lines[0])))
            max_lines = max(max_lines, len(lines) - 1)
            for ln in lines[1:]:
                max_line = max(max_line, len(tokenizer.tokenize_block("\n" + ln)))
            trunc_blocks += renderer.TRUNC in b
    return {
        "sessions": len(per_session),
        "event_blocks": len(per_block),
        "tokens_per_session": _dist(per_session),
        "tokens_per_block": _dist(per_block),
        "max_header_tokens": max_header,
        "max_body_lines": max_lines,
        "max_body_line_tokens_incl_marker": max_line,
        "blocks_with_TRUNC": trunc_blocks,
    }


def amend_manifest(out_dir: str | Path, *, progress=None) -> dict[str, Any]:
    """Add renderer/tokeniser source hashes and dev/val token statistics to ``MANIFEST_v0_3.json``."""
    say = progress or (lambda _m: None)
    out = Path(out_dir)
    mpath = out / MANIFEST_V03
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    for split in RAW_SPLITS:  # the files must be the ones the manifest describes
        p = raw_path(out, split)
        if file_sha256(p) != manifest["files"][p.name]["sha256"]:
            raise ValueError(f"{p.name} does not match its manifest hash; rebuild first")
    stats: dict[str, Any] = {}
    for split in STATS_SPLITS:
        stats[split] = split_token_statistics(out, split)
        say(f"token statistics {split}: {stats[split]['tokens_per_session']} per session, "
            f"{stats[split]['tokens_per_block']} per block")
    for split in RAW_SPLITS:
        if split not in STATS_SPLITS:
            p = raw_path(out, split)
            stats[split] = {"statistics": "not computed (sealed or evaluation-only split)",
                            "file": p.name, "exists": p.exists(), "sha256": file_sha256(p)}
    manifest["renderer_source_sha256"] = source_sha256(renderer)
    manifest["tokeniser_source_sha256"] = source_sha256(tokenizer)
    manifest["token_statistics"] = {
        "definition": ("tokens_per_session = len(encode_session(render_session(raw)).tokens), BOS and every EOE "
                       "included; tokens_per_block includes the block's EOE; percentiles are numpy linear"),
        "computed_on": list(STATS_SPLITS),
        "splits": stats,
    }
    _write_json(mpath, manifest)
    return manifest
