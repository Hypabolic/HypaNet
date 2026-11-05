"""Token audit of the v0.4 interface on dev/val (plan v0.4, M0; Risks: opaque ids tokenise badly).

    uv run python -m hypaqwen.token_audit [--splits val,dev] [--out ../../artifacts/reports/v04_token_audit.json]

Counts Qwen tokens per session (prefix + every block, without suffixes), per block, and the
tokens a one-pass verbaliser readout runs (session + one suffix per window). Tokeniser only: no
model, no labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from . import MODEL_ID, MODEL_REVISION, REPO
from . import interface as iface_mod
from .data import iter_tasks
from .interface import build_interface


def _dist(a) -> dict:
    a = np.asarray(a, dtype=np.float64)
    return {"mean": round(float(a.mean()), 2), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": round(float(np.percentile(a, 99)), 2), "max": int(a.max())}


def audit(split: str, tok, iface) -> dict:
    sess, blocks, readout, windows = [], [], [], 0
    for task in iter_tasks(split):
        bids = iface.block_ids(tok, task.blocks)
        n = len(iface.prefix_ids) + sum(len(b) for b in bids)
        sess.append(n)
        blocks.extend(len(b) for b in bids)
        readout.append(n + len(task.window_pos) * len(iface.suffix_ids))
        windows += len(task.window_pos)
    return {
        "sessions": len(sess),
        "windows": windows,
        "tokens_per_session": _dist(sess),
        "tokens_per_block": _dist(blocks),
        "total_session_tokens": int(sum(sess)),
        "total_readout_tokens": int(sum(readout)),
        "readout_over_session": round(sum(readout) / sum(sess), 3),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="val,dev")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "reports" / "v04_token_audit.json")
    a = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    iface = build_interface(tok)
    v03 = json.loads((REPO / "datasets" / "v0_3" / "MANIFEST_v0_3.json").read_text())["token_statistics"]["splits"]
    rep = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "interface": {"status": "DRAFT (not frozen; G1 freezes it)",
                      "source_sha256": hashlib.sha256(Path(iface_mod.__file__).read_bytes()).hexdigest(),
                      "prefix_tokens": len(iface.prefix_ids), "suffix_tokens": len(iface.suffix_ids),
                      "label_words": list(iface_mod.LABEL_WORDS), "label_token_ids": list(iface.label_ids)},
        "definition": "tokens_per_session = prefix + every block piece (block text + newline), no suffix; "
                      "readout tokens add one suffix per window; percentiles are numpy linear",
        "splits": {},
    }
    for split in a.splits.split(","):
        r = audit(split, tok, iface)
        v = v03.get(split, {}).get("tokens_per_session", {})
        if v:
            r["v03_tokens_per_session_mean"] = v["mean"]
            r["qwen_over_v03_mean"] = round(r["tokens_per_session"]["mean"] / v["mean"], 3)
        rep["splits"][split] = r
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
