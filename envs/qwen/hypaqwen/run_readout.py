"""Run the verbaliser readout over a split and save per-window label-word logits.

    uv run python -m hypaqwen.run_readout --split dev --limit 20 --device mps --out OUT.npz
    uv run python -m hypaqwen.run_readout --split val --device mps --dtype fp32 --edit ARCHIVE --out OUT.npz

Rows are in the export's row order (sessions in file order, ``window_pos`` in order), so they line
up with ``{split}.labels.npz``. Writes no scores; scoring runs in the frozen v0.3 environment.

``--edit ARCHIVE`` reads out a WSP build archive's edit (``scripts/v04_run_build.py``; e.g. the WSP-A
column the WSP-B disclosure needs, PREREGISTRATION §7.1): :func:`hypaqwen.wsp.runner.load_edit` loads
its tensors, checked against ``BUILD_RECORD.json`` (journal-checked builds under the real guard only),
and :func:`hypaqwen.wsp.runner.apply_edit` copies them into the fp32 model before the first session.
``meta["edit"]`` then names the archive: its ``edit_state_sha256``, the sha256 of its
``BUILD_RECORD.json`` and its phase (``agent_loop_lab.v04.disclosure.check_column`` requires both
hashes); without ``--edit`` it is null.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers

from . import MODEL_ID, MODEL_REVISION
from . import interface as iface_mod
from . import model as qmodel
from .data import iter_tasks
from .interface import build_interface
from .readout import Verbaliser


def apply_archive_edit(model, archive: Path, *, dtype: str, load=None, apply=None) -> dict:
    """Apply a build archive's edit to ``model`` (module docstring); returns ``meta["edit"]``.

    ``load`` and ``apply`` default to :func:`hypaqwen.wsp.runner.load_edit` and ``apply_edit`` (tests pass
    fakes: the real ``load_edit`` needs the pinned checkpoint)."""
    if dtype != "fp32":
        raise SystemExit(f"--edit needs --dtype fp32 (edits are fp32), not {dtype}")
    if load is None or apply is None:
        from .wsp import runner

        load, apply = load or runner.load_edit, apply or runner.apply_edit
    archive = Path(archive)
    record_bytes = (archive / "BUILD_RECORD.json").read_bytes()
    record = json.loads(record_bytes)
    apply(model, load(archive))
    return {"archive": archive.name, "phase": record.get("phase"), "edit_state_sha256": record["edit_state_sha256"],
            "build_record_sha256": hashlib.sha256(record_bytes).hexdigest()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp32", choices=sorted(qmodel.DTYPES))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--edit", type=Path, default=None, help="a WSP build archive whose edit is read out (fp32)")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.edit is not None and a.dtype != "fp32":
        ap.error("--edit needs --dtype fp32 (edits are fp32)")
    tok, model = qmodel.load(a.device, a.dtype)
    edit_meta = None if a.edit is None else apply_archive_edit(model, a.edit, dtype=a.dtype)
    iface = build_interface(tok)
    verb = Verbaliser(model, iface)
    logits, lse, sess_ids, row_pos, n_tok = [], [], [], [], 0
    t0 = time.monotonic()
    tasks = iter_tasks(a.split)
    if a.limit is not None:
        tasks = itertools.islice(tasks, a.limit)
    for n, task in enumerate(tasks):
        r = verb.session(iface.block_ids(tok, task.blocks), task.window_pos)
        logits.append(r.label_logits.numpy())
        lse.append(r.logsumexp.numpy())
        sess_ids.append(task.session_id)
        row_pos.extend(r.pos)
        n_tok += r.tokens
        if (n + 1) % 10 == 0:
            el = time.monotonic() - t0
            print(f"{n + 1} sessions, {n_tok} tokens, {n_tok / el:.0f} tok/s", flush=True)
    el = time.monotonic() - t0
    meta = {
        "split": a.split, "limit": a.limit, "device": a.device, "dtype": a.dtype,
        "torch": torch.__version__, "transformers": transformers.__version__, "platform": platform.platform(),
        "python": platform.python_version(), "attn_implementation": qmodel.ATTN_IMPLEMENTATION,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "file_sha256": qmodel.FILE_SHA256},
        "interface_sha256": hashlib.sha256(Path(iface_mod.__file__).read_bytes()).hexdigest(),
        "interface": {"name": iface_mod.FROZEN.name, "spec_sha256": iface_mod.spec_sha256()},
        "edit": edit_meta,
        "sessions": len(sess_ids), "tokens": n_tok, "seconds": round(el, 1), "tokens_per_second": round(n_tok / el, 1),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, label_logits=np.concatenate(logits), logsumexp=np.concatenate(lse),
             session_ids=np.array(sess_ids), row_pos=np.array(row_pos, dtype=np.int32), meta=json.dumps(meta))
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
