"""v0.5 validation scoring of a no-tools weight-programming run: the network only. Harness only.

``scripts/v05_score_validation.py`` drives this module. A v0.5 run (``experiments/v0_5/runs/RUN/``)
leaves one ``artifact.dlw`` (DLW3, the v0.3 transformer, written from the author's ``edits``
blocks) and ``artifact.sha256``. There is no reference policy in this arm.

Reuse, not re-implementation (every number is computed by v0.3 / v0.4 code):

* val is loaded with ``v03.scoring.load_val`` (the official cached encoding), and its window rows
  must hash to ``v04.scoring.VAL_ROWS_SHA256`` (the v0.3 val rows the sealed run recorded), so the
  windows, gate and labels are exactly v0.3's;
* the 200 disclosed sessions come from the v0.3 rule and are checked against
  ``experiments/v0_3/DISCLOSURE_PRIVATE.json`` (``v04.scoring.checked_disclosed``);
* network classes are ``model.predict(infer.eoe_logits(model, enc))`` per session (the v0.3 path),
  read at each window's last event, in a ``v03.sealed.make_pool`` worker pool (spawn, 2 torch
  threads per worker) or in process;
* metrics come from ``v03.scoring.score`` (``eval.metrics.score_report`` on both populations, the
  v0.3 bootstrap: 10,000 session resamples, seed 20261104, per-family seeds), with the archived
  v0.3 Phase A network in the slot v0.3 gave the reference policy. The network block does not
  depend on that slot; the slot's block is the v0.3 A network on the same draws, and the H2
  fidelity block becomes the argmax agreement of the two networks (every gate-passing window,
  and the scored ones);
* paired macro-F1 differences over the shared draws come from ``v04.scoring.paired_macro_f1``.

**Reproduction check.** The v0.3 A column re-scored here must equal the network blocks of
``artifacts/reports/v03_val_v03-phase-a.json`` (overall and per family, both populations). The
population and bootstrap blocks and the disclosed-session hash must also be equal, or scoring
refuses (the windows would differ from v0.3's). A difference in the v0.3 A blocks alone (a CPU
float difference in that network's argmax) is recorded, not fatal: the v0.5 network's numbers do not
depend on it, but the paired comparison with v0.3 A is then flagged.

**Logit diagnostics** (label-free, every event of every val session): events whose four logits at
``EOE`` are not all finite, events whose top logit is shared by two or more classes (argmax then
takes the first, so an all-zero output reads PRODUCTIVE), events whose four logits are all equal,
and the largest finite ``|logit|``.

Val numbers are never shown to a weight author (a v0.5 run gets no feedback of any kind).
"""

from __future__ import annotations

import json
import platform
import re
import time
from concurrent.futures import as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ..eval import metrics as M
from ..schema import CLASS_NAMES
from ..v03 import artifact as v03_artifact
from ..v03 import infer
from ..v03 import scoring as v03_scoring
from ..v03 import sealed as v03_sealed
from ..v03.model import predict
from ..v04 import scoring as v04_scoring

__all__ = [
    "SPLIT", "RUNS_ROOT", "REPORT_DIR", "NET_COLUMN", "V03_A_COLUMN", "PRIOR_COLUMN", "ScoringError",
    "check_run_name", "run_dir", "read_sha256_file", "load_run_artifact", "check_apply_report", "v03_a_spec",
    "logit_diagnostics",
    "NetResult", "network_predictions", "compare_v03_a", "score_network", "report",
]

SPLIT = v03_scoring.SPLIT  # "val", hard-coded in v0.3; there is no override
REPO = v03_scoring.REPO
RUNS_ROOT = REPO / "experiments" / "v0_5" / "runs"
REPORT_DIR = REPO / "artifacts" / "reports"
REPORT_NAME = "v05_val_{run}.json"
V03_A_DIR = v03_scoring.WEIGHT_ROOT / v03_scoring.PHASE_DIRS["v03-phase-a"]
V03_A_REPORT = REPORT_DIR / "v03_val_v03-phase-a.json"
TORCH_THREADS = v03_scoring.TORCH_THREADS
DEFAULT_WORKERS = v03_sealed.DEFAULT_WORKERS
CHUNK_SESSIONS = v03_sealed.CHUNK_SESSIONS
POPULATIONS = ("excluding_disclosed", "all_val")

NET_COLUMN = "v0.5 network"
V03_A_COLUMN = "v0.3 A network"
PRIOR_COLUMN = v04_scoring.PRIOR_COLUMNS[1]  # "prior always-PRODUCTIVE" (= the all-zero network; the dev majority is PRODUCTIVE too)
A_BLOCK = "v03_phase_a_network"
AGREE_BLOCK = "agreement_with_v03_phase_a_network"
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

DEFINITIONS = {
    "network": "the v0.5 network's classes, scored exactly as v0.3 scored a network (eval.metrics, v0.3 windows, gate and populations)",
    A_BLOCK: ("the archived v0.3 Phase A network (weight_program/v0_3/phase-a/artifact.dlw) on the same windows and "
              "bootstrap draws; it fills the slot v0.3 gave the reference policy, which does not change the network block"),
    AGREE_BLOCK: ("argmax agreement between the v0.5 network and the v0.3 A network; 'all' = every gate-passing "
                  "validation window (any oracle label), 'scored' = the scored population; no labels are used"),
    "paired_macro_f1": ("macro-F1(minuend) - macro-F1(subtrahend) over the shared session-bootstrap draws (overall, not "
                        "per family), v04.scoring.paired_macro_f1"),
    PRIOR_COLUMN: "PRODUCTIVE on every gate-passing window (what an all-zero network predicts: argmax ties go to index 0)",
}
POPULATION_TEXT = {
    "excluding_disclosed": ("every validation session except the 200 v0.3 Phase B disclosed sessions (v0.3 rule, checked "
                            "against experiments/v0_3/DISCLOSURE_PRIVATE.json); primary, as in v0.3"),
    "all_val": "every validation session (reported for transparency)",
}
VISIBILITY = "harness only: never shown to a weight author (a v0.5 run gets no feedback)"


class ScoringError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Runs and artefacts
# ---------------------------------------------------------------------------


def check_run_name(run: str) -> str:
    if not isinstance(run, str) or not RUN_NAME.fullmatch(run):
        raise ScoringError(f"not a run name: {run!r} (letters, digits, '.', '_', '-'; e.g. r1)")
    return run


def run_dir(run: str, runs_root: Path = RUNS_ROOT) -> Path:
    """``runs_root/RUN`` holding ``artifact.dlw`` and ``artifact.sha256``."""
    d = Path(runs_root) / check_run_name(run)
    for f in ("artifact.dlw", "artifact.sha256"):
        if not (d / f).is_file():
            raise FileNotFoundError(f"{d / f} missing (only a finished run's artefact is scored)")
    return d


def read_sha256_file(path: Path, name: str) -> str:
    """The digest a ``sha256sum``-style file records for ``name`` (a bare digest is accepted too)."""
    lines = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    for parts in lines:
        if (len(parts) == 1 and len(lines) == 1) or (len(parts) >= 2 and parts[1].lstrip("*") == name):
            if not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
                raise ScoringError(f"{path}: not a sha256 digest: {parts[0]!r}")
            return parts[0]
    raise ScoringError(f"{path}: no digest for {name}")


def load_run_artifact(path: Path, sha_file: Path | None, *, run: str | None = None
                      ) -> tuple[v03_sealed.NetSpec, dict[str, Any]]:
    """Check ``path`` against ``sha_file`` (when given), load and validate it as a DLW3 v0.3 network.

    Returns the pool spec and a record (sha256, metadata, non-zero parameters). With ``run`` (official
    scoring) the artefact's metadata must name that run, as the v0.5 edits reader writes it."""
    path = Path(path)
    actual = v03_sealed.sha256_file(path)
    checked = None
    if sha_file is not None:
        recorded = read_sha256_file(sha_file, path.name)
        if actual != recorded:
            raise ScoringError(f"{path}: sha256 {actual} != recorded {recorded} ({sha_file})")
        checked = str(sha_file)
    model, meta = v03_artifact.load_artifact(path)  # validates names, shapes, dtype, finiteness
    if run is not None and meta.get("run") != run:
        raise ScoringError(f"{path}: artefact metadata run {meta.get('run')!r} != {run!r} "
                           "(an official artefact is written by scripts/v05_apply_reply.py for this run)")
    sp = v03_artifact.sparsity(model)
    info = {
        "path": str(path),
        "artifact_sha256": actual,
        "sha256_checked_against": checked,
        "artifact_metadata": meta,
        "nonzero_parameters": sp["overall"],
        "nonzero_per_tensor": {k: v["nonzero"] for k, v in sp.items() if k != "overall"},
    }
    return v03_sealed.NetSpec(NET_COLUMN, "artifact", str(path), actual), info


def check_apply_report(run_dir: Path, artifact_sha256: str) -> str:
    """An official artefact must be the one ``scripts/v05_apply_reply.py`` wrote: ``RUN/apply.json`` records it."""
    path = Path(run_dir) / "apply.json"
    if not path.is_file():
        raise ScoringError(f"{path} missing: an official artefact is written by scripts/v05_apply_reply.py")
    try:
        recorded = json.loads(path.read_text())["artifact"]["sha256"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ScoringError(f"{path}: no artifact.sha256 ({exc})") from exc
    if recorded != artifact_sha256:
        raise ScoringError(f"{path}: artifact sha256 {recorded} != the artefact's {artifact_sha256}")
    return recorded


def v03_a_spec(weight_dir: Path = V03_A_DIR, report: Path = V03_A_REPORT) -> tuple[v03_sealed.NetSpec, dict[str, Any]]:
    """The archived v0.3 A network, checked against its ``artifact.sha256`` and its official val report."""
    dlw = Path(weight_dir) / "artifact.dlw"
    actual = v03_sealed.sha256_file(dlw)
    recorded = read_sha256_file(Path(weight_dir) / "artifact.sha256", dlw.name)
    if actual != recorded:
        raise ScoringError(f"{dlw}: sha256 {actual} != recorded {recorded}")
    in_report = json.loads(Path(report).read_text())["inputs"]["artifact_sha256"]
    if actual != in_report:
        raise ScoringError(f"{dlw}: sha256 {actual} != {report} inputs.artifact_sha256 {in_report}")
    return v03_sealed.NetSpec(V03_A_COLUMN, "artifact", str(dlw), actual), {"path": str(dlw), "artifact_sha256": actual}


# ---------------------------------------------------------------------------
# Predictions (the v0.3 path, plus label-free logit diagnostics)
# ---------------------------------------------------------------------------


def logit_diagnostics(logits: torch.Tensor) -> tuple[int, int, int, int, float]:
    """``(events, non-finite, tied top, all equal, max finite |logit|)`` of ``[events, 4]`` EOE logits."""
    fin = torch.isfinite(logits).all(dim=-1)
    lf = logits[fin]
    n_fin = int(fin.sum())
    if n_fin:
        top = lf.max(dim=-1).values[:, None]
        tied = int(((lf == top).sum(dim=-1) >= 2).sum())
        equal = int((lf == top).all(dim=-1).sum())
        mx = float(lf.abs().max())
    else:
        tied = equal = 0
        mx = 0.0
    return int(logits.shape[0]), int(logits.shape[0]) - n_fin, tied, equal, mx


_W_MODELS: dict[tuple[str, str], Any] = {}


def _net_job(spec: v03_sealed.NetSpec, key: int, encs: list[Any]) -> tuple[str, int, list[tuple]]:
    model = _W_MODELS.get((spec.name, spec.sha256))
    if model is None:
        model = _W_MODELS[(spec.name, spec.sha256)] = v03_sealed.load_net(spec)
    out = []
    for enc in encs:
        t0, w0 = time.process_time(), time.perf_counter()
        logits = infer.eoe_logits(model, enc)
        pred = predict(logits).numpy()
        cpu, wall = time.process_time() - t0, time.perf_counter() - w0
        out.append((pred.astype(np.int8), cpu, wall, len(enc.tokens), logit_diagnostics(logits)))
    return spec.name, key, out


@dataclass
class NetResult:
    pred: np.ndarray  # class per row (-1 for rows of sessions not run)
    latency: dict[str, Any]
    logits: dict[str, Any]  # label-free diagnostics


def network_predictions(specs: Sequence[v03_sealed.NetSpec], data: v03_scoring.ValData, pool: Any = None, *,
                        workers: int = 0, chunk_sessions: int = CHUNK_SESSIONS,
                        progress: Callable[[str], None] | None = None) -> dict[str, NetResult]:
    """Every spec's class per row, latency and logit diagnostics: through ``pool`` (``v03.sealed.make_pool``,
    jobs as ``v03.sealed.network_predictions`` schedules them), or in this process when ``pool`` is None."""
    n_s = len(data.sessions)
    chunks = [list(range(a, min(a + chunk_sessions, n_s))) for a in range(0, n_s, chunk_sessions)]
    payload = [[v03_sealed._Enc(data.sessions[s].tokens, data.sessions[s].event_ord, data.sessions[s].offset,
                                data.sessions[s].eoe_index) for s in ch] for ch in chunks]
    cost = [sum(len(e.tokens) ** 1.5 for e in p) for p in payload]
    jobs = sorted(((k, spec) for k in range(len(chunks)) for spec in specs), key=lambda kj: -cost[kj[0]])
    per: dict[str, list[Any]] = {spec.name: [None] * len(chunks) for spec in specs}
    t0 = time.time()
    step = max(1, len(jobs) // 10)

    def note(done: int) -> None:
        if progress is not None and (done % step == 0 or done == len(jobs)):
            progress(f"networks: {done}/{len(jobs)} jobs ({time.time() - t0:.0f}s)")

    if pool is None:
        torch.set_num_threads(TORCH_THREADS)
        for done, (k, spec) in enumerate(jobs, 1):
            name, _, out = _net_job(spec, k, payload[k])
            per[name][k] = out
            note(done)
    else:
        futs = [pool.submit(_net_job, spec, k, payload[k]) for k, spec in jobs]
        try:
            for done, f in enumerate(as_completed(futs), 1):
                name, k, out = f.result()
                per[name][k] = out
                note(done)
        except BaseException:
            for f in futs:
                f.cancel()
            raise
    res: dict[str, NetResult] = {}
    for spec in specs:
        net = np.full(len(data.rows), -1, dtype=np.int64)
        cpu, wall, ntok = [], [], []
        ev = nonfin = tied = equal = 0
        mx = 0.0
        for k, ch in enumerate(chunks):
            for s, (pred, c, w, nt, diag) in zip(ch, per[spec.name][k]):
                idx = data.rows_of(s)
                net[idx] = pred.astype(np.int64)[data.row_pos[idx]]
                cpu.append(c), wall.append(w), ntok.append(nt)
                ev, nonfin, tied, equal = ev + diag[0], nonfin + diag[1], tied + diag[2], equal + diag[3]
                mx = max(mx, diag[4])
        lat = v03_sealed.latency_summary(cpu, wall, ntok, workers)
        diag_rec = {"events": ev, "nonfinite_events": nonfin, "tied_top_events": tied, "all_equal_events": equal,
                    "max_abs_finite_logit": mx,
                    "definition": ("EOE logits of every event of every val session (no labels): non-finite = any of the "
                                   "four logits is NaN/inf; tied top = two or more classes share the top logit (argmax "
                                   "takes the first); all equal = all four logits equal (e.g. an all-zero output)")}
        res[spec.name] = NetResult(net, lat, diag_rec)
    return res


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def compare_v03_a(body: Mapping[str, Any], recorded: Mapping[str, Any]) -> dict[str, Any]:
    """Compare a :func:`v03_scoring.score` body whose reference-policy slot holds the v0.3 A network with the
    official v0.3 A val report: population, bootstrap and disclosed-session blocks strictly (raises), the
    v0.3 A network blocks (overall and per family, both populations) recorded."""
    canon, diff = v03_sealed.canon, v03_sealed.first_difference
    if body["disclosed_sessions"]["sha256_of_sorted_ids"] != recorded["disclosed_sessions"]["sha256_of_sorted_ids"]:
        raise ScoringError("disclosed sessions differ from the v0.3 A val report")
    first: str | None = None
    compared = 0
    for pop in POPULATIONS:
        mine, rec = body[pop], recorded[pop]
        for key in ("population", "bootstrap"):
            d = diff(canon(mine[key]), canon(rec[key]))
            if d is not None:
                raise ScoringError(f"val {pop} {key} differs from the v0.3 A val report at {d}")
        if set(mine["per_family"]) != set(rec["per_family"]):
            raise ScoringError(f"val {pop}: families differ from the v0.3 A val report")
        blocks = [("overall", mine["overall"], rec["overall"])]
        blocks += [(f"per_family/{f}", mine["per_family"][f], rec["per_family"][f]) for f in sorted(rec["per_family"])]
        for where, a, b in blocks:
            d = diff(canon(a["reference_policy"]), canon(b["network"]))
            compared += 1
            if d is not None and first is None:
                first = f"val {pop} {where}{d}"
    return {"reproduced": first is None, "first_difference": first, "blocks_compared": compared}


def _stat(x: Mapping[str, Any]) -> dict[str, Any]:
    return {"value": x["value"], "ci95": x["ci95"]}


def _summary(rep: Mapping[str, Any], with_a: bool) -> dict[str, Any]:
    o = rep["overall"]
    out: dict[str, Any] = {
        "sessions": o["sessions"],
        "scored_windows": rep["population"]["scored_windows"],
        "network": {k: _stat(o["network"][k]) for k in ("macro_f1", "false_alarm_rate", "accuracy")},
        "network_f1_per_class": {c: _stat(o["network"]["per_class"][c]["f1"]) for c in CLASS_NAMES},
        "paired_macro_f1": {k: {"value": v["value"], "ci95": v["ci95"]} for k, v in rep["paired_macro_f1"].items()},
    }
    if with_a:
        out[A_BLOCK] = {k: _stat(o[A_BLOCK][k]) for k in ("macro_f1", "false_alarm_rate", "accuracy")}
        out[AGREE_BLOCK] = {p: _stat(o[AGREE_BLOCK][p]["agreement"]) for p in ("all", "scored")}
    return out


def score_network(data: v03_scoring.ValData, net: np.ndarray, a_net: np.ndarray | None = None, *,
                  disclosed: frozenset[str] | None = None, recorded_a: Mapping[str, Any] | None = None,
                  n_resamples: int = M.BOOTSTRAP_RESAMPLES, seed: int = M.BOOTSTRAP_SEED) -> dict[str, Any]:
    """Both populations' v0.3 metrics of ``net`` (and of ``a_net``, the v0.3 A network, with the two
    networks' agreement), paired macro-F1 differences and a summary. ``recorded_a`` (the v0.3 A val
    report) runs :func:`compare_v03_a`."""
    gate = np.array([r.gate_passed for r in data.rows], dtype=bool)
    for name, p in ((NET_COLUMN, net), (V03_A_COLUMN, a_net)):
        if p is not None and np.any(np.asarray(p)[gate] < 0):
            raise ScoringError(f"a gate-passing window has no {name} prediction")
    with_a = a_net is not None
    rows = data.rows
    if disclosed is None:
        disclosed = v03_scoring.disclosed_sessions(rows)
    body = v03_scoring.score(data, net, a_net if with_a else net, n_resamples=n_resamples, seed=seed,
                             disclosed=disclosed)
    body["disclosed_sessions"]["source"] = "computed from the v0.3 Phase B rule (no model output)"
    repro = compare_v03_a(body, recorded_a) if (with_a and recorded_a is not None) else None
    excl = np.array([r.session_id not in disclosed for r in rows], dtype=bool)
    prior = np.where(gate, 0, -1).astype(np.int64)
    body.pop("summary")
    body["populations"] = dict(POPULATION_TEXT)
    body["summary"] = {}
    for pop, mask in (("excluding_disclosed", excl), ("all_val", np.ones(len(rows), dtype=bool))):
        rep = body[pop]
        rep.pop("policy_failures")
        for blk in (rep["overall"], *rep["per_family"].values()):
            a_blk, fid = blk.pop("reference_policy"), blk.pop("h2_fidelity")
            if with_a:
                blk[A_BLOCK], blk[AGREE_BLOCK] = a_blk, fid
        rep["definitions"] = {k: v for k, v in rep["definitions"].items() if k != "h2_fidelity"} | DEFINITIONS
        idx = np.flatnonzero(mask)
        sub = [rows[i] for i in idx.tolist()]
        preds = {NET_COLUMN: np.asarray(net)[idx], PRIOR_COLUMN: prior[idx]}
        pairs = {v04_scoring.pair_name(NET_COLUMN, PRIOR_COLUMN): (NET_COLUMN, PRIOR_COLUMN)}
        if with_a:
            preds[V03_A_COLUMN] = np.asarray(a_net)[idx]
            pairs = {v04_scoring.pair_name(NET_COLUMN, V03_A_COLUMN): (NET_COLUMN, V03_A_COLUMN), **pairs}
        rep["paired_macro_f1"] = v04_scoring.paired_macro_f1(sub, preds, pairs, n_resamples=n_resamples, seed=seed)
        if repro is not None and not repro["reproduced"]:
            for k in rep["paired_macro_f1"]:
                if V03_A_COLUMN in k:
                    rep["paired_macro_f1"][k]["caveat"] = "the v0.3 A column did not reproduce its official val blocks"
        body["summary"][pop] = _summary(rep, with_a)
    body["v03_phase_a_reproduction"] = repro
    return body


def _code_hashes() -> dict[str, str]:
    files = [Path(__file__), Path(v03_scoring.__file__), Path(v03_sealed.__file__), Path(M.__file__),
             Path(v04_scoring.__file__), Path(infer.__file__), REPO / "scripts" / "v05_score_validation.py"]
    return {str(p.resolve().relative_to(REPO)): v03_sealed.sha256_file(p) for p in files if p.is_file()}


def _run_files(d: Path, max_bytes: int = 1 << 28) -> dict[str, Any]:
    """sha256 and size of each regular file directly in the run directory (the record the score belongs to)."""
    out: dict[str, Any] = {}
    for p in sorted(d.iterdir()):
        if p.is_file() and not p.is_symlink():
            size = p.stat().st_size
            out[p.name] = {"bytes": size, "sha256": v03_sealed.sha256_file(p) if size <= max_bytes else None}
    return out


def report(run: str, *, artifact: Path, sha_file: Path | None, official: bool, include_v03_a: bool = True,
           workers: int = DEFAULT_WORKERS, progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Load val, predict, score; the full ``v05_val`` report (the caller writes it)."""
    check_run_name(run)
    if official and not include_v03_a:
        raise ScoringError("an official report always includes the v0.3 A comparison")
    t0 = time.time()
    spec, art = load_run_artifact(artifact, sha_file, run=run if official else None)
    apply_sha = check_apply_report(Path(artifact).parent, art["artifact_sha256"]) if official else None
    specs = [spec]
    a_info = None
    if include_v03_a:
        a_spec, a_info = v03_a_spec()
        specs.append(a_spec)
    data = v03_scoring.load_val()
    rows = data.rows
    got = v03_sealed.rows_sha256(rows)
    if got != v04_scoring.VAL_ROWS_SHA256:
        raise ScoringError(f"val rows sha256 {got} != v0.3 sealed record {v04_scoring.VAL_ROWS_SHA256}")
    disclosed = v04_scoring.checked_disclosed(rows)
    progress(f"{SPLIT}: {len(data.sessions)} sessions, {len(rows)} windows loaded in {time.time() - t0:.1f}s")
    t1 = time.time()
    if workers > 0:
        with v03_sealed.make_pool(workers) as pool:
            res = network_predictions(specs, data, pool, workers=workers, progress=lambda m: progress("  " + m))
    else:
        res = network_predictions(specs, data, None, workers=0, progress=lambda m: progress("  " + m))
    t2 = time.time()
    progress(f"  networks: {t2 - t1:.1f}s")
    recorded_a = json.loads(V03_A_REPORT.read_text()) if include_v03_a else None
    if recorded_a is not None and data.inputs["cache_key"] != recorded_a["inputs"]["dataset"]:
        raise ScoringError("the val encoding's cache key differs from the one the v0.3 A val report was scored on "
                           f"({V03_A_REPORT.name} inputs.dataset); the v0.5 network would see other token inputs")
    a_net = res[V03_A_COLUMN].pred if include_v03_a else None
    body = score_network(data, res[NET_COLUMN].pred, a_net, disclosed=disclosed, recorded_a=recorded_a)
    progress(f"  metrics + bootstrap: {time.time() - t2:.1f}s")
    rel = (lambda p: str(Path(p).resolve().relative_to(REPO))) if official else str
    inputs: dict[str, Any] = {
        "run_dir": rel(Path(artifact).parent) if official else None,
        "artifact": {**art, "path": rel(art["path"])},
        "dataset": data.inputs["cache_key"],
        "val_rows_sha256": got,
        "scoring_code_sha256": _code_hashes(),
    }
    if official:
        inputs["run_files"] = _run_files(Path(artifact).parent)
        inputs["apply_report_artifact_sha256"] = apply_sha
    if a_info is not None:
        inputs["v03_phase_a"] = {**a_info, "path": str(Path(a_info["path"]).resolve().relative_to(REPO)),
                                 "official_val_report": str(V03_A_REPORT.relative_to(REPO)),
                                 "official_val_report_sha256": v03_sealed.sha256_file(V03_A_REPORT)}
    return {
        "report": "v05_val",
        "spec_version": "0.5",
        "run": run,
        "split": SPLIT,
        "official": official,
        "visibility": VISIBILITY,
        "scored": "the network only (no reference policy in this arm)",
        "inputs": inputs,
        "nonzero_parameters": art["nonzero_parameters"],
        "network_latency": res[NET_COLUMN].latency,
        "network_logits": res[NET_COLUMN].logits,
        "v03_phase_a_logits": res[V03_A_COLUMN].logits if include_v03_a else None,
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "torch": torch.__version__, "torch_threads": TORCH_THREADS, "workers": workers},
        "runtime_s": {"load": t1 - t0, "networks": t2 - t1, "score": time.time() - t2},
        **body,
    }
