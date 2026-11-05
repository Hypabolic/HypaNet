"""The v0.3 columns on val for the v0.4 report (PREREGISTRATION §1): recomputed, checked exactly, saved.

Runs in the frozen v0.3 environment on this Mac's CPU, with v0.3's own code path (the path of the
unofficial v0.3 val dry run, without its counterfactual pass):

* ``v03.sealed.build_split("val")`` (val only: no sealed split and no counterfactual file is opened),
  whose rows must hash to the v0.3 val ``rows_sha256``, so they are the v0.4 bridge's rows in order;
* ``v03.sealed.load_columns()`` (the archived v0.3 A and B networks and every Phase C run, each
  sha-checked) and ``v03.sealed.network_predictions`` (the networks only; the v0.3 reference
  policies are not columns of the v0.4 report and are cited from the v0.3 report);
* the two constant priors as the v0.4 bridge builds them (``scoring.prior_columns``).

Every column's val block (overall and per family, both populations) and every Phase C group block
must equal ``artifacts/reports/v03_sealed.json`` exactly, or nothing is written (no tolerance: plan
revision 10). The file holds the network columns' class per val row (format :data:`FORMAT`), read
by ``scoring.load_v03_columns``. It holds no label and no score.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..eval import m4
from ..eval.collisions import WindowRow
from ..v03 import sealed as S
from . import scoring

__all__ = ["FORMAT", "DEFAULT_OUT", "ReproductionError", "reproduction_check", "save", "export"]

FORMAT = "v04-v03-val-columns/1"
DEFAULT_OUT = S.REPO / "experiments" / "v0_4" / "columns" / "v03.val.npz"
POPULATIONS = ("excluding_disclosed", "all_val")


class ReproductionError(RuntimeError):
    pass


def reproduction_check(rows: Sequence[WindowRow], preds: Mapping[str, np.ndarray], groups: Mapping[str, Sequence[str]],
                       disclosed: frozenset[str], recorded: Mapping[str, Any], *,
                       n_resamples: int = scoring.M.BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Score ``preds`` on both val populations and compare every column and group block with ``recorded``
    (``v03_sealed.json`` ``splits.val.populations``) exactly; raise :class:`ReproductionError` on any difference."""
    excl = np.array([r.session_id not in disclosed for r in rows], dtype=bool)
    compared = 0
    for pop, mask in (("excluding_disclosed", excl), ("all_val", np.ones(len(rows), dtype=bool))):
        idx = np.flatnonzero(mask)
        sub = [rows[i] for i in idx.tolist()]
        mine = m4.score_columns(sub, {c: np.asarray(p)[idx] for c, p in preds.items()}, {}, groups,
                                n_resamples=n_resamples)
        rec = recorded[pop]
        if set(mine["per_family"]) != set(rec["per_family"]):
            raise ReproductionError(f"val {pop}: families differ from the v0.3 report")
        blocks = [("overall", mine["overall"], rec["overall"])]
        blocks += [(f"per_family/{f}", mine["per_family"][f], rec["per_family"][f]) for f in sorted(rec["per_family"])]
        for where, a, b in blocks:
            for name in preds:
                if name not in b["columns"]:
                    raise ReproductionError(f"val {pop} {where}: {name} is not in the v0.3 report")
                d = S.first_difference(S.canon(a["columns"][name]), S.canon(b["columns"][name]))
                if d is not None:
                    raise ReproductionError(f"val {pop} {where}: {name} differs from the v0.3 report at {d}")
                compared += 1
            for g in groups:
                d = S.first_difference(S.canon(a["groups"][g]), S.canon(b["groups"][g]))
                if d is not None:
                    raise ReproductionError(f"val {pop} {where}: group {g} differs from the v0.3 report at {d}")
    return {"match": True, "populations": list(POPULATIONS), "column_blocks_compared": compared,
            "columns": list(preds), "groups": sorted(groups)}


def save(out: Path, names: Sequence[str], preds: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, pred=np.stack([np.asarray(preds[n], dtype=np.int64) for n in names]), names=np.array(list(names)),
                 meta=json.dumps(dict(meta), sort_keys=True))
    os.replace(tmp, out)


def export(out: Path = DEFAULT_OUT, *, workers: int = S.DEFAULT_WORKERS,
           progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Recompute, check and write the v0.3 val columns (module docstring); returns the file's meta."""
    import torch

    say = progress or (lambda m: None)
    torch.set_num_threads(S.TORCH_THREADS)
    sd = S.build_split("val")
    rows = sd.data.rows
    got = S.rows_sha256(rows)
    if got != scoring.VAL_ROWS_SHA256:
        raise ReproductionError(f"val rows sha256 {got} != {scoring.VAL_ROWS_SHA256}")
    disclosed = scoring.checked_disclosed(rows)
    cols = S.load_columns()
    say(f"val: {len(sd.data.sessions)} sessions, {len(rows)} windows; {len(cols.nets)} v0.3 networks")
    with S.make_pool(workers) as pool:
        nets = S.network_predictions(cols.nets, sd.data, pool, workers=workers, label="val", progress=say)
    gate = np.array([r.gate_passed for r in rows], dtype=bool)
    preds: dict[str, np.ndarray] = {}
    for spec in cols.nets:
        net = nets[spec.name][0]
        if np.any(net[gate] < 0):
            raise ReproductionError(f"val: a gate-passing window has no {spec.name} prediction")
        preds[spec.name] = net
    names = list(preds)
    prior = scoring.recorded_dev_prior()
    checked = {**preds, **scoring.prior_columns(rows, int(prior["majority_index"]))}
    rec = json.loads(Path(S.SEALED_REPORT).read_text())
    say("val: comparing every block with artifacts/reports/v03_sealed.json")
    repro = reproduction_check(rows, checked, cols.groups, disclosed, rec["splits"]["val"]["populations"])
    repro["report_sha256"] = S.sha256_file(Path(S.SEALED_REPORT))
    meta = {
        "format": FORMAT,
        "split": "val",
        "val_rows_sha256": got,
        "groups": cols.groups,
        "reproduction": repro,
        "inputs": cols.inputs,
        "columns_info": {n: cols.info[n] for n in names},
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "torch": torch.__version__, "torch_threads": S.TORCH_THREADS, "workers": workers},
        "note": "v0.3 network classes per val row; no labels, no scores (PREREGISTRATION §1)",
    }
    save(out, names, preds, meta)
    return meta
