"""Run the verbaliser readout over a split and save per-window label-word logits.

    uv run python -m hypaqwen.run_readout --split dev --limit 20 --device mps --out OUT.npz

Rows are in the export's row order (sessions in file order, ``window_pos`` in order), so they line
up with ``{split}.labels.npz``. Writes no scores; scoring runs in the frozen v0.3 environment.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import MODEL_ID, MODEL_REVISION
from . import interface as iface_mod
from . import model as qmodel
from .data import iter_tasks
from .interface import build_interface
from .readout import Verbaliser


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp32", choices=sorted(qmodel.DTYPES))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    tok, model = qmodel.load(a.device, a.dtype)
    iface = build_interface(tok)
    verb = Verbaliser(model, iface)
    logits, lse, sess_ids, n_tok = [], [], [], 0
    t0 = time.monotonic()
    tasks = iter_tasks(a.split)
    if a.limit is not None:
        tasks = itertools.islice(tasks, a.limit)
    for n, task in enumerate(tasks):
        r = verb.session(iface.block_ids(tok, task.blocks), task.window_pos)
        logits.append(r.label_logits.numpy())
        lse.append(r.logsumexp.numpy())
        sess_ids.append(task.session_id)
        n_tok += r.tokens
        if (n + 1) % 10 == 0:
            el = time.monotonic() - t0
            print(f"{n + 1} sessions, {n_tok} tokens, {n_tok / el:.0f} tok/s", flush=True)
    el = time.monotonic() - t0
    meta = {
        "split": a.split, "limit": a.limit, "device": a.device, "dtype": a.dtype,
        "torch": torch.__version__, "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "interface_sha256": hashlib.sha256(Path(iface_mod.__file__).read_bytes()).hexdigest(),
        "sessions": len(sess_ids), "tokens": n_tok, "seconds": round(el, 1), "tokens_per_second": round(n_tok / el, 1),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, label_logits=np.concatenate(logits), logsumexp=np.concatenate(lse),
             session_ids=np.array(sess_ids), meta=json.dumps(meta))
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
