"""v0.3 one-time sealed evaluation (spec v0.3 §10, E7, §14 V4).

Harness only; never part of the packet or a weight-build path. The pre-registration is
``experiments/v0_3/sealed/README.md``; ``scripts/v03_sealed_eval.py`` drives this module.

Reuse, not re-implementation:

* val is loaded with ``scoring.load_val`` (the official cached encoding);
* policies run only through ``scoring.policy_predictions`` -> ``runner.evaluate_policy``;
* archived phases are loaded and checked with ``scoring.load_phase``;
* metrics come from ``eval.m4.score_columns`` (the v0.2 sealed multi-column form of
  ``eval.metrics``), and on val also from ``scoring.score`` for the reproduction check.

**Sealed access.** :func:`build_split` and :func:`verify_sealed_files` open ``test`` / ``ood`` only
when given the :class:`Attempt` returned by :func:`write_attempt_record`; the marker file on disk
must hold the attempt's random token. Nothing here writes a cache of any split.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import os
import secrets
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ..eval import m4, metrics
from ..eval.collisions import WindowRow
from ..schema import CLASS_NAMES, OracleLabel
from . import artifact as v03_artifact
from . import infer, renderer, scoring, tokenizer
from .model import predict, zero_model

REPO = scoring.REPO
V03_DATA = scoring.V03_DATA
V02_DATA = scoring.V02_DATA
REPORT_DIR = REPO / "artifacts" / "reports"
SEALED_REPORT = REPORT_DIR / "v03_sealed.json"
SEALED_MARKER = REPORT_DIR / "v03_sealed.started.json"
PREVIOUS_ATTEMPT_GLOB = "v03_sealed.attempt*.json"
V02_REPORT = REPORT_DIR / "m4_sealed_v02.json"
VAL_REPORTS = {p: REPORT_DIR / f"v03_val_{p}.json" for p in scoring.PHASE_DIRS}
PRIVATE_DISCLOSURE = REPO / "experiments" / "v0_3" / "DISCLOSURE_PRIVATE.json"
PHASE_C_ROOT = REPO / "experiments" / "phase-c-trained" / "v0_3"
PRE_REGISTRATION = "experiments/v0_3/sealed/README.md"

SEALED_SPLITS = ("test", "ood")
UNSEALED_SPLITS = ("val", "counterfactual")
TORCH_THREADS = scoring.TORCH_THREADS  # 2, the thread count of the official val reports and Phase C
DEFAULT_WORKERS = 8  # physical cores of the evaluation machine
CHUNK_SESSIONS = 50
PHASES = (("v0.3 A", "v03-phase-a"), ("v0.3 B", "v03-phase-b"))
LABELLED = {"v03-phase-a": {"labelled_examples": 0, "revisions": 0},
            "v03-phase-b": {"labelled_examples": 200, "revisions": 1}}
PRIOR_COLUMNS = ("prior dev-majority", "prior always-PRODUCTIVE")
_LABELS = tuple(OracleLabel)
_SEP = "\x00"


class SealedAccessError(RuntimeError):
    """A sealed split was requested without a valid attempt record."""


class CheckFailed(RuntimeError):
    """A pre-registered check failed; the message names the check only (never a metric)."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canon(obj: Any) -> Any:
    """JSON round trip (the form in which reports are compared)."""
    return json.loads(json.dumps(obj, sort_keys=True, allow_nan=False))


def first_difference(a: Any, b: Any, path: str = "") -> str | None:
    """Key path of the first difference between two JSON-like values, or None if equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}/{k} (missing on one side)"
            d = first_difference(a[k], b[k], f"{path}/{k}")
            if d is not None:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path} (length)"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d is not None:
                return d
        return None
    return None if (type(a) is type(b) and a == b) else (path or "/")


# ---------------------------------------------------------------------------
# Attempt record (README §6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Attempt:
    marker: Path
    token: str


def previous_attempts(report_dir: Path | None = None) -> list[dict[str, Any]]:
    out = []
    report_dir = SEALED_MARKER.parent if report_dir is None else report_dir
    for p in sorted(report_dir.glob(PREVIOUS_ATTEMPT_GLOB)):
        out.append({"file": p.name, "sha256": sha256_file(p), "content": json.loads(p.read_text())})
    return out


def refuse_reasons(report: Path | None = None, marker: Path | None = None) -> list[str]:
    """Why a sealed run may not start (empty: it may)."""
    report = SEALED_REPORT if report is None else report
    marker = SEALED_MARKER if marker is None else marker
    return [f"{p} exists; the v0.3 sealed evaluation runs once (README §6)" for p in (report, marker) if p.exists()]


def write_attempt_record(record: Mapping[str, Any], marker: Path | None = None) -> Attempt:
    """Create the attempt record exclusively (fails if it exists) and return the access token."""
    marker = SEALED_MARKER if marker is None else marker
    token = secrets.token_hex(16)
    doc = {**record, "attempt_token": token}
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "x", encoding="utf-8") as f:
        f.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return Attempt(marker, token)


def _check_attempt(split: str, attempt: Attempt | None) -> None:
    if split not in SEALED_SPLITS:
        return
    if attempt is None:
        raise SealedAccessError(f"{split} is sealed: it is opened only after the attempt record is written")
    try:
        doc = json.loads(Path(attempt.marker).read_text())
    except (OSError, ValueError) as exc:
        raise SealedAccessError(f"attempt record unreadable: {exc}") from None
    if doc.get("attempt_token") != attempt.token:
        raise SealedAccessError("attempt record token mismatch")


# ---------------------------------------------------------------------------
# Data files and manifests
# ---------------------------------------------------------------------------


def v02_files(split: str) -> dict[str, Path]:
    """v0.2 events / oracle files of ``split`` keyed by their manifest name (relative to datasets/v0_2)."""
    if split == "ood":
        rel = ("ood/ood.events.jsonl.gz", "ood/ood.oracle.jsonl.gz")
    elif split in ("dev", "val", "test", "counterfactual"):
        rel = (f"standard/{split}.events.jsonl.gz", f"standard/{split}.oracle.jsonl.gz")
    else:
        raise ValueError(f"unknown split {split!r}")
    return {r: V02_DATA / r for r in rel}


def raw_file(split: str) -> tuple[str, Path]:
    return f"{split}.raw.jsonl.gz", V03_DATA / f"{split}.raw.jsonl.gz"


def _manifests() -> dict[str, Any]:
    return {
        "v03": json.loads((V03_DATA / "MANIFEST_v0_3.json").read_text()),
        "v02": json.loads((V02_DATA / "MANIFEST.json").read_text()),
        "v02_ood": json.loads((V02_DATA / "OOD_MANIFEST.json").read_text()),
    }


def expected_hashes(split: str, man: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Manifest hash of every data file of ``split`` (keys: ``datasets/...`` paths)."""
    man = man or _manifests()
    out: dict[str, str] = {}
    rname, _ = raw_file(split)
    out[f"datasets/v0_3/{rname}"] = man["v03"]["files"][rname]["sha256"]
    for rel in v02_files(split):
        if rel.startswith("ood/"):
            out[f"datasets/v0_2/{rel}"] = man["v02_ood"]["files"][rel]
        else:
            out[f"datasets/v0_2/{rel}"] = man["v02"]["files"][rel]["sha256"]
    return out


def verify_unsealed_inputs(splits: Sequence[str] = ("dev", "val", "counterfactual")) -> dict[str, str]:
    """Hash every unsealed data file and the manifests; return the hashes (raises CheckFailed)."""
    for s in splits:
        if s in SEALED_SPLITS:
            raise SealedAccessError(f"{s} is sealed")
    man = _manifests()
    out: dict[str, str] = {}
    v03 = man["v03"]
    for name, path in (("datasets/v0_2/MANIFEST.json", V02_DATA / "MANIFEST.json"),
                       ("datasets/v0_2/OOD_MANIFEST.json", V02_DATA / "OOD_MANIFEST.json")):
        h = sha256_file(path)
        if v03["extends"].get(name) != h:
            raise CheckFailed(f"{name}: sha256 differs from MANIFEST_v0_3.json extends")
        out[name] = h
    out["datasets/v0_3/MANIFEST_v0_3.json"] = sha256_file(V03_DATA / "MANIFEST_v0_3.json")
    for key, mkey, mod in (("renderer", "renderer_source_sha256", renderer), ("tokenizer", "tokeniser_source_sha256", tokenizer)):
        h = sha256_file(Path(mod.__file__))
        if v03.get(mkey) != h:
            raise CheckFailed(f"{key} source sha256 differs from MANIFEST_v0_3.json {mkey}")
        out[f"src/agent_loop_lab/v03/{key}.py"] = h
    for s in splits:
        for name, want in expected_hashes(s, man).items():
            h = sha256_file(REPO / name)
            if h != want:
                raise CheckFailed(f"{name}: sha256 differs from its manifest")
            out[name] = h
    return out


def verify_sealed_hash_records(v02_report: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Cross-check the sealed files' recorded hashes (manifests, v0.3 acceptance, v0.2 report); reads no sealed file."""
    man = _manifests()
    acc = man["v03"]["acceptance"]["canonical_files"]
    out: dict[str, str] = {}
    for split in SEALED_SPLITS:
        exp = expected_hashes(split, man)
        rname, _ = raw_file(split)
        stat = man["v03"]["token_statistics"]["splits"].get(split, {})
        if stat.get("sha256") != exp[f"datasets/v0_3/{rname}"]:
            raise CheckFailed(f"{rname}: MANIFEST_v0_3.json files and token_statistics disagree")
        for name, h in exp.items():
            if name.startswith("datasets/v0_2/"):
                rel = name[len("datasets/v0_2/"):]
                if acc[rel]["v0_2_sha256"] != h or acc[rel]["regenerated_sha256"] != h:
                    raise CheckFailed(f"{rel}: v0.2 manifest and MANIFEST_v0_3 acceptance disagree")
                if v02_report is not None and v02_report["inputs"]["data_sha256"].get(rel) != h:
                    raise CheckFailed(f"{rel}: hash recorded in m4_sealed_v02.json differs from the manifest")
            out[name] = h
    return out


def verify_sealed_files(attempt: Attempt) -> dict[str, str]:
    """Hash the sealed files against the manifests (after the attempt record; opens sealed files)."""
    man = _manifests()
    out: dict[str, str] = {}
    for split in SEALED_SPLITS:
        _check_attempt(split, attempt)
        for name, want in expected_hashes(split, man).items():
            h = sha256_file(REPO / name)
            if h != want:
                raise CheckFailed(f"{name}: sha256 differs from its manifest")
            out[name] = h
    return out


# ---------------------------------------------------------------------------
# Split data (mirrors scoring._build_val, parameterised by split, never cached)
# ---------------------------------------------------------------------------


@dataclass
class SplitData:
    split: str
    data: scoring.ValData
    meta: dict[str, Any] = field(default_factory=dict)  # counterfactual metadata per session (m4.load_split form)
    inputs: dict[str, Any] = field(default_factory=dict)


def _v02_sessions(split: str):
    if split == "ood":
        from ..datasets.ood import iter_ood_sessions

        return iter_ood_sessions(V02_DATA)
    from ..datasets.build import iter_sessions

    return iter_sessions(V02_DATA, split)


def _cf_meta(record: Any) -> dict[str, Any] | None:
    cf = record.latent_trace.get("counterfactual") if isinstance(record.latent_trace, Mapping) else None
    if cf is None:
        return None
    return {**cf, "labels": {int(k): OracleLabel(v) for k, v in record.window_labels.items()}}


def build_split(split: str, *, limit: int | None = None, attempt: Attempt | None = None) -> SplitData:
    """Rendered and tokenised sessions and window rows of ``split`` (val: ``scoring.load_val``)."""
    if split == "val":
        if attempt is not None:
            raise ValueError("val needs no attempt record")
        data = scoring.load_val(limit=limit)
        return SplitData("val", data, {}, {"source": "scoring.load_val", **data.inputs})
    if split not in ("counterfactual", *SEALED_SPLITS):
        raise ValueError(f"unknown split {split!r}")
    return _encode_split(split, limit=limit, attempt=attempt)


def _encode_split(split: str, *, limit: int | None = None, attempt: Attempt | None = None) -> SplitData:
    """The generic builder (``scoring._build_val`` step by step, any split, no cache).

    ``build_split`` uses it for counterfactual, test and OOD; tests run it on val to compare it with
    ``scoring.load_val``. The sealed-access check is here too, so no caller can bypass it."""
    _check_attempt(split, attempt)
    if split in SEALED_SPLITS and limit is not None:
        raise SealedAccessError("--limit-sessions is not allowed on a sealed split")
    from ..windows import gate, validate_sequence, window_endpoints
    from .build_raw import alignment_errors, iter_raw

    ids: list[str] = []
    fams: list[str] = []
    sessions: list[scoring.EncodedSession] = []
    rows: list[WindowRow] = []
    r_sess: list[int] = []
    r_pos: list[int] = []
    meta: dict[str, Any] = {}
    pairs = zip(iter_raw(V03_DATA, split), _v02_sessions(split), strict=True)
    for n, ((sid, raw), (events, record)) in enumerate(pairs):
        if limit is not None and n >= limit:
            break
        if sid != record.session_id or len(raw) != len(events):
            raise ValueError(f"{split} line {n + 1}: raw/events/oracle misaligned")
        for k, (r, e) in enumerate(zip(raw, events)):
            errs = alignment_errors(r, e)
            if errs:
                raise ValueError(f"{split} line {n + 1} event {k}: raw op misaligned")
        blocks = renderer.render_session([r.to_dict() for r in raw], sid)
        enc = tokenizer.encode_session(blocks)
        if len(enc.eoe_index) != len(events):
            raise ValueError(f"{split} line {n + 1}: EOE count differs from the event count")
        s = len(ids)
        ids.append(sid)
        fams.append(record.scenario_family)
        sessions.append(scoring.EncodedSession(list(enc.tokens), list(enc.event_ord), list(enc.offset),
                                               list(enc.eoe_index), list(enc.surface)))
        validate_sequence(events)
        for pos in window_endpoints(events):
            endpoint = events[pos].sequence_index
            try:
                label = OracleLabel(record.window_labels[endpoint])
            except KeyError:
                raise ValueError(f"{split} line {n + 1}: no oracle label for a window endpoint") from None
            rows.append(WindowRow(session_id=sid, family=record.scenario_family, split=split, endpoint=int(endpoint),
                                  label=label, gate_passed=bool(gate(events[: pos + 1]).passed), vector=None))
            r_sess.append(s)
            r_pos.append(pos)
        cf = _cf_meta(record)
        if cf is not None:
            meta[sid] = cf
    data = scoring.ValData(ids, fams, sessions, rows, np.asarray(r_sess, dtype=np.int64),
                           np.asarray(r_pos, dtype=np.int64))
    if limit is not None and split == "counterfactual":
        meta = complete_pairs(meta)
    return SplitData(split, data, meta, {"source": "sealed.build_split", "limit": limit})


def complete_pairs(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only counterfactual sessions whose pair has both branches present."""
    branches: dict[str, set[str]] = {}
    for cf in meta.values():
        branches.setdefault(cf["pair_id"], set()).add(cf["branch"])
    return {sid: cf for sid, cf in meta.items() if {"a", "b"} <= branches[cf["pair_id"]]}


def row_tuples(rows: Sequence[WindowRow]) -> list[list[Any]]:
    return [[r.session_id, r.family, int(r.endpoint), r.label.value, bool(r.gate_passed)] for r in rows]


def rows_sha256(rows: Sequence[WindowRow]) -> str:
    return hashlib.sha256(json.dumps(row_tuples(rows), separators=(",", ":")).encode()).hexdigest()


def v02_window_rows(split: str, *, limit: int | None = None, attempt: Attempt | None = None) -> list[WindowRow]:
    """v0.2's own window rows (``m4.load_split``; first ``limit`` sessions via ``session_windows``)."""
    _check_attempt(split, attempt)
    if limit is None:
        rows, _ = m4.load_split(V02_DATA, split)
        return rows
    if split in SEALED_SPLITS:
        raise SealedAccessError("--limit-sessions is not allowed on a sealed split")
    from ..eval.collisions import session_windows

    rows = []
    for events, record in itertools.islice(_v02_sessions(split), limit):
        rows.extend(session_windows(events, record, split))
    return rows


def identity_check(sd: SplitData, v02_report: Mapping[str, Any] | None, *, limit: int | None = None,
                   attempt: Attempt | None = None, population_rows: Sequence[WindowRow] | None = None) -> dict[str, Any]:
    """README §4 checks 1-3. Raises CheckFailed naming the check; returns the record."""
    rows = sd.data.rows
    v2 = v02_window_rows(sd.split, limit=limit, attempt=attempt)
    if row_tuples(v2) != row_tuples(rows):
        raise CheckFailed(f"{sd.split}: v0.3 window rows differ from v0.2 m4.load_split rows")
    out: dict[str, Any] = {"windows": len(rows), "sessions": len(sd.data.session_ids),
                           "rows_sha256": rows_sha256(rows),
                           "rows_definition": "sha256 of compact JSON [[session_id, family, endpoint, oracle label, gate_passed], ...] in file order",
                           "matches_v02_load_split": True}
    if v02_report is None or limit is not None:
        out["v02_report_counts"] = "skipped (limited dry run)" if limit is not None else "skipped (no v0.2 report)"
        return out
    rep_split = v02_report["splits"].get(sd.split)
    if rep_split is None:
        raise CheckFailed(f"{sd.split}: m4_sealed_v02.json has no such split")
    files = {}
    for rel in v02_files(sd.split):
        if v02_report["inputs"]["data_sha256"].get(rel) != expected_hashes(sd.split)[f"datasets/v0_2/{rel}"]:
            raise CheckFailed(f"{sd.split}: m4_sealed_v02.json data hash differs for {rel}")
        files[rel] = True
    if canon(m4.gate_report(rows)) != canon(rep_split["gate"]):
        raise CheckFailed(f"{sd.split}: gate block differs from m4_sealed_v02.json")
    checked = ["rows", "data_sha256", "gate"]
    if "scores" in rep_split:
        prow = list(rows if population_rows is None else population_rows)
        ids, _, _ = metrics.session_index(prow)
        if canon(metrics._population(prow, len(ids))) != canon(rep_split["scores"]["population"]):
            raise CheckFailed(f"{sd.split}: population block differs from m4_sealed_v02.json")
        checked.append("population")
    out["v02_report_counts"] = {"matched": checked}
    return out


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NetSpec:
    name: str
    kind: str  # "artifact" (.dlw) or "state_dict" (torch.save of a state_dict)
    path: str
    sha256: str


def load_net(spec: NetSpec):
    data = Path(spec.path).read_bytes()
    if hashlib.sha256(data).hexdigest() != spec.sha256:
        raise CheckFailed(f"{spec.name}: model file sha256 changed")
    if spec.kind == "artifact":
        sd, _ = v03_artifact.deserialize_state_dict(data)
    elif spec.kind == "state_dict":
        sd = torch.load(io.BytesIO(data), weights_only=True)
    else:
        raise ValueError(spec.kind)
    model = v03_artifact.load_state_dict_into(zero_model(), sd)
    model.eval()
    return model


@dataclass
class Columns:
    nets: list[NetSpec]
    policies: dict[str, Path]  # phase column prefix -> reference_policy.py
    info: dict[str, dict[str, Any]]
    fid_pairs: dict[str, tuple[str, str]]
    groups: dict[str, list[str]]
    phase_c_logs: dict[str, dict[str, Any]]
    inputs: dict[str, Any]


def phase_c_dir(budget: int | str, seed: int, root: Path = PHASE_C_ROOT) -> Path:
    return root / str(budget) / f"seed{seed}"


def check_phase_c_run(budget: int | str, seed: int, root: Path = PHASE_C_ROOT) -> tuple[NetSpec, dict, dict]:
    """sha-check one Phase C run (model.pt vs model.sha256 vs training log); returns spec, log, info."""
    d = phase_c_dir(budget, seed, root)
    pt = d / "model.pt"
    for f in ("model.pt", "model.sha256", "training_log.json"):
        if not (d / f).is_file():
            raise CheckFailed(f"Phase C {budget}/seed{seed}: {f} missing")
    recorded = (d / "model.sha256").read_text().split()[0]
    actual = sha256_file(pt)
    if actual != recorded:
        raise CheckFailed(f"Phase C {budget}/seed{seed}: model.pt sha256 != model.sha256")
    log = json.loads((d / "training_log.json").read_text())
    if log.get("model_sha256") != actual:
        raise CheckFailed(f"Phase C {budget}/seed{seed}: training log model_sha256 differs")
    if log.get("smoke") is not False or log.get("phase") != "v03-phase-c":
        raise CheckFailed(f"Phase C {budget}/seed{seed}: not a registered (non-smoke) run")
    if str(log.get("budget")) != str(budget) or log.get("seed") != seed:
        raise CheckFailed(f"Phase C {budget}/seed{seed}: training log budget/seed differ from its path")
    name = f"v0.3 C-{budget} seed{seed}"
    info = {"kind": "network", "labelled_examples": log["train_positions"], "budget_sessions": log["budget_sessions"],
            "pool_sessions": log["pool_sessions"], "epochs_run": log["epochs_run"], "best_epoch": log["best_epoch"],
            "best_es_macro_f1_training_log": log["best_es_macro_f1"], "stopped": log["stopped"],
            "training_threads": (log.get("environment") or {}).get("torch_threads"),
            "artifact_sha256": actual, "training_log_sha256": sha256_file(d / "training_log.json")}
    return NetSpec(name, "state_dict", str(pt), actual), log, info


def load_columns(*, phase_c_root: Path = PHASE_C_ROOT, weight_dirs: Mapping[str, Path] | None = None) -> Columns:
    """Check and describe every v0.3 column (A/B archives via ``scoring.load_phase``; Phase C runs)."""
    from ..baselines import phase_c_v03 as pc

    nets: list[NetSpec] = []
    policies: dict[str, Path] = {}
    info: dict[str, dict[str, Any]] = {}
    fid: dict[str, tuple[str, str]] = {}
    inputs: dict[str, Any] = {"weight_program": {}, "phase_c": {}}
    for short, phase in PHASES:
        d = scoring.phase_dir(phase, None if weight_dirs is None else weight_dirs.get(phase))
        model, meta, pol, shas = scoring.load_phase(phase, d)
        nz = v03_artifact.sparsity(model)["overall"]
        base = {**LABELLED[phase], **shas, "archive_dir": str(d)}
        nets.append(NetSpec(f"{short} network", "artifact", str(d / "artifact.dlw"), shas["artifact_sha256"]))
        policies[short] = pol
        info[f"{short} network"] = {"kind": "network", **base, "nonzero_parameters": nz}
        info[f"{short} reference_policy"] = {"kind": "policy", **base}
        fid[short] = (f"{short} network", f"{short} reference_policy")
        inputs["weight_program"][phase] = {**shas, "packet_sha256": meta.get("packet_sha256")}
    groups: dict[str, list[str]] = {}
    logs: dict[str, dict[str, Any]] = {}
    for b in pc.BUDGETS:
        names = []
        for s in pc.SEEDS[b]:
            spec, log, inf = check_phase_c_run(b, s, phase_c_root)
            model = load_net(spec)
            inf["nonzero_parameters"] = v03_artifact.sparsity(model)["overall"]
            nets.append(spec)
            info[spec.name] = inf
            logs[spec.name] = log
            names.append(spec.name)
            inputs["phase_c"][spec.name] = {"artifact_sha256": inf["artifact_sha256"],
                                            "training_log_sha256": inf["training_log_sha256"]}
        groups[f"v0.3 C-{b}"] = names
    return Columns(nets, policies, info, fid, groups, logs, inputs)


def dev_prior() -> dict[str, Any]:
    """Majority class over gate-passing scored dev windows (v0.2 M4 rule), from the dev encoding cache."""
    from ..baselines import phase_c_v03 as pc

    dev = pc.load_dev()
    lab = dev.payload["row_label"].numpy().astype(np.int64)
    g = dev.payload["row_gate"].numpy().astype(bool)
    keep = g & (lab < len(CLASS_NAMES))
    counts = np.bincount(lab[keep], minlength=len(CLASS_NAMES))[: len(CLASS_NAMES)]
    maj = int(np.argmax(counts))
    return {"class_counts": {c: int(counts[i]) for i, c in enumerate(CLASS_NAMES)}, "majority": CLASS_NAMES[maj],
            "majority_index": maj, "scored_windows": int(counts.sum())}


# ---------------------------------------------------------------------------
# Parallel network inference (one causal forward per session, as scoring.network_predictions)
# ---------------------------------------------------------------------------


@dataclass
class _Enc:
    tokens: list[int]
    event_ord: list[int]
    offset: list[int]
    eoe_index: list[int]


_W_MODELS: dict[str, Any] = {}


def _worker_init(threads: int) -> None:
    torch.set_num_threads(threads)


def _net_job(spec: NetSpec, key: int, encs: list[_Enc]) -> tuple[str, int, list[tuple[np.ndarray, float, float, int]]]:
    model = _W_MODELS.get(spec.name)
    if model is None:
        model = _W_MODELS[spec.name] = load_net(spec)
    out = []
    for enc in encs:
        t0, w0 = time.process_time(), time.perf_counter()
        pred = predict(infer.eoe_logits(model, enc)).numpy()
        cpu, wall = time.process_time() - t0, time.perf_counter() - w0
        out.append((pred.astype(np.int8), cpu, wall, len(enc.tokens)))
    return spec.name, key, out


def latency_summary(cpu: Sequence[float], wall: Sequence[float], ntok: Sequence[int], workers: int) -> dict[str, Any]:
    c, w = np.asarray(cpu, dtype=np.float64), np.asarray(wall, dtype=np.float64)
    return {
        "torch_threads": TORCH_THREADS,
        "workers": workers,
        "sessions": len(c),
        "tokens": int(sum(ntok)),
        "cpu_s_total": float(c.sum()),
        "wall_s_total": float(w.sum()),
        "cpu_s_per_session": {"mean": float(c.mean()), "p50": float(np.percentile(c, 50)),
                              "p95": float(np.percentile(c, 95)), "max": float(c.max())} if len(c) else None,
        "cpu_ms_per_1k_tokens": float(1e6 * c.sum() / max(1, sum(ntok))),
        "definition": ("time.process_time (all threads of the worker process) and perf_counter around one "
                       "infer.eoe_logits call per session, in one of `workers` concurrent worker processes"),
    }


def network_predictions(specs: Sequence[NetSpec], data: scoring.ValData, pool: Any, *, workers: int,
                        chunk_sessions: int = CHUNK_SESSIONS, label: str = "",
                        progress: Callable[[str], None] | None = None) -> dict[str, tuple[np.ndarray, dict]]:
    """Network class per row (-1 for rows of no session) and latency, for every spec, via ``pool``."""
    n_s = len(data.sessions)
    chunks = [list(range(a, min(a + chunk_sessions, n_s))) for a in range(0, n_s, chunk_sessions)]
    payload = [[_Enc(data.sessions[s].tokens, data.sessions[s].event_ord, data.sessions[s].offset,
                     data.sessions[s].eoe_index) for s in ch] for ch in chunks]
    cost = [sum(len(e.tokens) ** 1.5 for e in p) for p in payload]
    jobs = sorted(((k, spec) for k in range(len(chunks)) for spec in specs), key=lambda kj: -cost[kj[0]])
    per: dict[str, list[Any]] = {spec.name: [None] * len(chunks) for spec in specs}
    t0 = time.time()
    futs = [pool.submit(_net_job, spec, k, payload[k]) for k, spec in jobs]
    done, step = 0, max(1, len(futs) // 10)
    try:
        for f in as_completed(futs):
            name, k, out = f.result()
            per[name][k] = out
            done += 1
            if progress is not None and (done % step == 0 or done == len(futs)):
                progress(f"{label} networks: {done}/{len(futs)} jobs ({time.time() - t0:.0f}s)")
    except BaseException:
        for f in futs:  # stop quickly: queued jobs are dropped, not run
            f.cancel()
        raise
    res: dict[str, tuple[np.ndarray, dict]] = {}
    for spec in specs:
        net = np.full(len(data.rows), -1, dtype=np.int64)
        cpu, wall, ntok = [], [], []
        for k, ch in enumerate(chunks):
            for s, (pred, c, w, nt) in zip(ch, per[spec.name][k]):
                idx = data.rows_of(s)
                net[idx] = pred.astype(np.int64)[data.row_pos[idx]]
                cpu.append(c), wall.append(w), ntok.append(nt)
        res[spec.name] = (net, latency_summary(cpu, wall, ntok, workers))
    return res


LATENCY_SESSIONS = 100


def latency_sample(data: scoring.ValData, n: int = LATENCY_SESSIONS) -> tuple[list[str], list[_Enc]]:
    """Ids and encodings of ``n`` val sessions, evenly spaced in file order (a fixed, label-free choice)."""
    k = len(data.sessions)
    idx = sorted({int(i) for i in np.linspace(0, k - 1, num=min(n, k))}) if k else []
    return [data.session_ids[s] for s in idx], [_Enc(data.sessions[s].tokens, data.sessions[s].event_ord,
                                                     data.sessions[s].offset, data.sessions[s].eoe_index) for s in idx]


def isolated_latency(specs: Sequence[NetSpec], encs: Sequence[_Enc]) -> dict[str, Any]:
    """CPU latency per session with no other work running: this process, 2 torch threads, one
    warm-up call, then one ``infer.eoe_logits`` call per sample session (README §5)."""
    torch.set_num_threads(TORCH_THREADS)
    out: dict[str, Any] = {}
    for spec in specs:
        model = load_net(spec)
        if encs:
            infer.eoe_logits(model, encs[0])
        cpu, wall, ntok = [], [], []
        for enc in encs:
            t0, w0 = time.process_time(), time.perf_counter()
            predict(infer.eoe_logits(model, enc)).numpy()
            cpu.append(time.process_time() - t0), wall.append(time.perf_counter() - w0), ntok.append(len(enc.tokens))
        lat = latency_summary(cpu, wall, ntok, 1)
        lat["definition"] = ("isolated: time.process_time and perf_counter around one infer.eoe_logits call per "
                             "session, main process only, after the worker pool and policies have finished")
        out[spec.name] = lat
    return out


def make_pool(workers: int) -> ProcessPoolExecutor:
    import multiprocessing as mp

    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                               initializer=_worker_init, initargs=(TORCH_THREADS,))


# ---------------------------------------------------------------------------
# Predictions for one split (networks in the pool, then each policy alone, as score_phase)
# ---------------------------------------------------------------------------


@dataclass
class SplitPredictions:
    preds: dict[str, np.ndarray]  # column -> class per row (policies filled per scoring.fill_policy_failures)
    raw_ref: dict[str, np.ndarray]  # phase prefix -> policy output with -2 failures
    latency: dict[str, dict]
    policy_wall_s: dict[str, float]


def predict_split(sd: SplitData, cols: Columns, prior: Mapping[str, Any], pool: Any, *, workers: int,
                  progress: Callable[[str], None] | None = None,
                  evaluate: Callable[..., list[int | None]] | None = None) -> SplitPredictions:
    data = sd.data
    say = progress or (lambda m: None)
    t0 = time.time()
    pol_out: dict[str, np.ndarray] = {}
    pol_wall: dict[str, float] = {}

    def run_policy(short: str) -> None:
        t = time.time()
        last = [0.0]

        def pp(msg: str) -> None:
            if time.time() - last[0] > 60:
                last[0] = time.time()
                say(f"{sd.split} {short} reference_policy: {msg}")

        pol_out[short] = scoring.policy_predictions(cols.policies[short], data, evaluate=evaluate, progress=pp)
        pol_wall[short] = time.time() - t
        say(f"{sd.split} {short} reference_policy: {int((pol_out[short] != -1).sum())} calls, "
            f"{int((pol_out[short] == -2).sum())} failures in {pol_wall[short]:.0f}s")

    # Same measurement conditions as scripts/v03_score_validation.py score_phase: the networks finish
    # first (every pool job has returned, so the workers are idle), then each policy runs alone, one
    # after the other, through scoring.policy_predictions. Nothing else competes for the CPU while the
    # runner's 0.5 s CPU limit per call applies.
    nets = network_predictions(cols.nets, data, pool, workers=workers, label=sd.split, progress=say)
    for short in cols.policies:
        run_policy(short)
    gate = np.array([r.gate_passed for r in data.rows], dtype=bool)
    preds: dict[str, np.ndarray] = {}
    latency: dict[str, dict] = {}
    for spec in cols.nets:
        net, lat = nets[spec.name]
        if np.any(net[gate] < 0):
            raise RuntimeError(f"{sd.split}: a gate-passing window has no {spec.name} prediction")
        preds[spec.name] = net
        latency[spec.name] = lat
    for short, ref in pol_out.items():
        if np.any(ref[gate] == -1):
            raise RuntimeError(f"{sd.split}: a gate-passing window has no {short} policy prediction")
        preds[f"{short} reference_policy"] = scoring.fill_policy_failures(data.rows, preds[f"{short} network"], ref)
    ordered = {}
    for name in column_order(cols):
        if name in preds:
            ordered[name] = preds[name]
    maj = int(prior["majority_index"])
    ordered["prior dev-majority"] = np.where(gate, maj, -1).astype(np.int64)
    ordered["prior always-PRODUCTIVE"] = np.where(gate, 0, -1).astype(np.int64)
    say(f"{sd.split}: predictions done ({time.time() - t0:.0f}s)")
    return SplitPredictions(ordered, pol_out, latency, pol_wall)


def column_order(cols: Columns) -> list[str]:
    out = []
    for short, _ in PHASES:
        out += [f"{short} network", f"{short} reference_policy"]
    for names in cols.groups.values():
        out += names
    return out + list(PRIOR_COLUMNS)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def population_masks(sd: SplitData, disclosed: frozenset[str]) -> dict[str, np.ndarray]:
    rows = sd.data.rows
    allm = np.ones(len(rows), dtype=bool)
    if sd.split == "val":
        return {"excluding_disclosed": np.array([r.session_id not in disclosed for r in rows], dtype=bool),
                "all_val": allm}
    return {"all": allm}


def score_split(sd: SplitData, sp: SplitPredictions, cols: Columns, disclosed: frozenset[str], *,
                n_resamples: int = metrics.BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    rows = sd.data.rows
    out: dict[str, Any] = {"gate": m4.gate_report(rows), "populations": {}}
    for pop, mask in population_masks(sd, disclosed).items():
        idx = np.flatnonzero(mask)
        sub = [rows[i] for i in idx.tolist()]
        sc = m4.score_columns(sub, {k: v[idx] for k, v in sp.preds.items()}, cols.fid_pairs, cols.groups,
                              n_resamples=n_resamples)
        sc["policy_failures"] = {short: scoring._failure_counts(rows, ref, mask) for short, ref in sp.raw_ref.items()}
        out["populations"][pop] = sc
    return out


def counterfactual_scores(sd: SplitData, sp: SplitPredictions, *, n_resamples: int = metrics.BOOTSTRAP_RESAMPLES
                          ) -> dict[str, Any]:
    return {"gate": m4.gate_report(sd.data.rows),
            "counterfactual_consistency": m4.counterfactual_consistency(sd.data.rows, sd.meta, sp.preds,
                                                                        n_resamples=n_resamples),
            "policy_failures": {short: scoring._failure_counts(sd.data.rows, ref, np.ones(len(sd.data.rows), bool))
                                for short, ref in sp.raw_ref.items()}}


# ---------------------------------------------------------------------------
# Val checks (README §5)
# ---------------------------------------------------------------------------


def private_disclosed(path: Path | None = None) -> frozenset[str]:
    rec = json.loads((PRIVATE_DISCLOSURE if path is None else path).read_text())
    return frozenset(w["session_id"] for w in rec["windows"])


def scoring_path_check(sd: SplitData, sp: SplitPredictions, scored: Mapping[str, Any], disclosed: frozenset[str], *,
                       n_resamples: int = metrics.BOOTSTRAP_RESAMPLES) -> tuple[dict[str, Any], dict[str, Any]]:
    """``scoring.score`` per phase (the official val path); its blocks must equal the m4 blocks.

    Returns ``(record, bodies)``; ``bodies[phase]`` is the ``scoring.score`` output."""
    out: dict[str, dict[str, Any]] = {}
    for short, phase in PHASES:
        body = scoring.score(sd.data, sp.preds[f"{short} network"], sp.raw_ref[short], n_resamples=n_resamples,
                             disclosed=disclosed)
        for pop in ("excluding_disclosed", "all_val"):
            rep, sc = body[pop], scored["populations"][pop]
            pairs = [("population", rep["population"], sc["population"]),
                     ("policy_failures", rep["policy_failures"], sc["policy_failures"][short])]
            for blk, scope in (("overall", None), *[("per_family", f) for f in sorted(rep["per_family"])]):
                r = rep[blk] if scope is None else rep[blk][scope]
                s = sc["overall"] if scope is None else sc["per_family"][scope]
                tag = blk if scope is None else f"{blk}/{scope}"
                pairs += [(f"{tag}/sessions", r["sessions"], s["sessions"]),
                          (f"{tag}/network", r["network"], s["columns"][f"{short} network"]),
                          (f"{tag}/reference_policy", r["reference_policy"], s["columns"][f"{short} reference_policy"]),
                          (f"{tag}/h2_fidelity", r["h2_fidelity"], s["h2_fidelity"][short])]
            for what, a, b in pairs:
                d = first_difference(canon(a), canon(b))
                if d is not None:
                    raise CheckFailed(f"val {phase} {pop}: scoring.score and m4.score_columns differ at {what}{d}")
        out[phase] = body
    return {p: {"match": True, "compared": "population, policy_failures, overall and per-family network / reference_policy / h2_fidelity"} for p in out}, out


def official_val_check(bodies: Mapping[str, Any], cols: Columns, disclosed: frozenset[str],
                       reports: Mapping[str, Path] | None = None) -> dict[str, Any]:
    """The A/B val blocks must equal the official v03_val reports exactly (full val only)."""
    reports = VAL_REPORTS if reports is None else reports
    out = {}
    for short, phase in PHASES:
        off = json.loads(Path(reports[phase]).read_text())
        body = canon(bodies[phase])
        checks = {"population/excluding_disclosed": (off["excluding_disclosed"], body["excluding_disclosed"]),
                  "population/all_val": (off["all_val"], body["all_val"]),
                  "summary": (off["summary"], body["summary"]),
                  "disclosed_sessions/count": (off["disclosed_sessions"]["count"], body["disclosed_sessions"]["count"]),
                  "disclosed_sessions/sha256": (off["disclosed_sessions"]["sha256_of_sorted_ids"],
                                                body["disclosed_sessions"]["sha256_of_sorted_ids"]),
                  "nonzero_parameters": (off["nonzero_parameters"], canon(cols.info[f"{short} network"]["nonzero_parameters"])),
                  "artifact_sha256": (off["inputs"]["artifact_sha256"], cols.info[f"{short} network"]["artifact_sha256"]),
                  "reference_policy_sha256": (off["inputs"]["reference_policy_sha256"],
                                              cols.info[f"{short} network"]["reference_policy_sha256"])}
        for what, (a, b) in checks.items():
            d = first_difference(canon(a), canon(b))
            if d is not None:
                raise CheckFailed(f"val {phase}: differs from {Path(reports[phase]).name} at {what}{d}")
        out[phase] = {"match": True, "report": Path(reports[phase]).name, "report_sha256": sha256_file(Path(reports[phase])),
                      "compared": sorted(checks)}
    return out


def phase_c_es_check(sd: SplitData, sp: SplitPredictions, cols: Columns, disclosed: frozenset[str]) -> dict[str, Any]:
    """Each Phase C model's macro-F1 on its ES subset must equal ``best_es_macro_f1`` (full val only)."""
    from ..baselines import phase_c as pc2
    from ..baselines import phase_c_v03 as pc

    data = sd.data
    ids = pc.es_session_ids(data.session_ids, disclosed)
    sorted_sha = pc.ids_sha256(sorted(ids))
    keep = set(ids)
    mask = np.array([r.session_id in keep and r.gate_passed and r.label.is_scored for r in data.rows], dtype=bool)
    y = np.array([int(r.label.to_class()) if m else 0 for r, m in zip(data.rows, mask)], dtype=np.int64)[mask]
    out = {}
    for names in cols.groups.values():
        for name in names:
            log = cols.phase_c_logs[name]
            if log["es_set"]["sha256_of_sorted_ids"] != sorted_sha:
                raise CheckFailed(f"{name}: ES set differs from the training log")
            got = pc2.macro_f1(y, sp.preds[name][mask])
            want = float(log["best_es_macro_f1"])
            if got != want:
                raise CheckFailed(f"{name}: ES macro-F1 recomputed on val differs from the training log")
            out[name] = {"match": True, "es_windows": int(mask.sum())}
    return out


def prior_confirmation(scored: Mapping[str, Any], v02_report: Mapping[str, Any], split: str, pop: str) -> dict[str, Any]:
    """README §4 check 4: v0.3 prior blocks vs the v0.2 report's (recorded, not a stop)."""
    rep = v02_report["splits"].get(split, {}).get("scores")
    if rep is None:
        return {"compared": False}
    mine = scored["populations"][pop]
    out = {}
    for name in PRIOR_COLUMNS:
        same = first_difference(canon(mine["overall"]["columns"][name]), canon(rep["overall"]["columns"][name])) is None
        fam_same = all(
            first_difference(canon(mine["per_family"][f]["columns"][name]), canon(rep["per_family"][f]["columns"][name])) is None
            for f in rep["per_family"]) and set(mine["per_family"]) == set(rep["per_family"])
        out[name] = {"overall_match": same, "per_family_match": fam_same}
    return out


# ---------------------------------------------------------------------------
# Report pieces
# ---------------------------------------------------------------------------


def summary_table(scored: Mapping[str, Any], cols: Columns) -> dict[str, Any]:
    t: dict[str, Any] = {}
    o = scored["overall"]
    for name, b in o["columns"].items():
        t[name] = {"macro_f1": [b["macro_f1"]["value"], b["macro_f1"]["ci95"]],
                   "false_alarm_rate": [b["false_alarm_rate"]["value"], b["false_alarm_rate"]["ci95"]],
                   "accuracy": [b["accuracy"]["value"], b["accuracy"]["ci95"]],
                   "recall": {k: b["per_class"][k]["recall"]["value"] for k in CLASS_NAMES}}
    for short, f in o["h2_fidelity"].items():
        t[f"{short} network"]["h2_fidelity_all"] = [f["all"]["agreement"]["value"], f["all"]["agreement"]["ci95"]]
        t[f"{short} network"]["h2_fidelity_scored"] = [f["scored"]["agreement"]["value"], f["scored"]["agreement"]["ci95"]]
    for g, gg in o["groups"].items():
        t[f"{g} mean±sd"] = {m: {"per_seed": gg[m]["per_seed"], "mean": gg[m]["mean"], "sd": gg[m]["sd"],
                                 "seed_mean_ci95": gg[m]["seed_mean_ci95"]}
                             for m in ("macro_f1", "false_alarm_rate", "accuracy")}
    return t


def cite_v02(v02_report: Mapping[str, Any], path: Path | None = None) -> dict[str, Any]:
    """The v0.2 columns, copied verbatim from m4_sealed_v02.json (README §3)."""
    path = V02_REPORT if path is None else path
    cf = v02_report["splits"].get("counterfactual", {}).get("counterfactual_consistency", {})
    return {
        "source": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
        "sha256": sha256_file(path),
        "note": "v0.2 columns (networks, policies, Phase C MLPs, rules, priors) are cited, not re-scored; valid because the window set is identical (identity_v02)",
        "summary": v02_report["summary"],
        "dev_prior": v02_report["dev_prior"],
        "columns": v02_report["columns"],
        "phase_c_groups": v02_report["phase_c_groups"],
        "counterfactual_consistency_overall": {k: v.get("overall") for k, v in cf.get("columns", {}).items()},
    }


def git_state(exclude: Sequence[str] = ()) -> dict[str, Any]:
    import subprocess

    def run(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "status_porcelain": [ln for ln in run("status", "--porcelain").splitlines()
                                     if not any(x in ln for x in exclude)]}
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__}


CODE_FILES = (
    "scripts/v03_sealed_eval.py",
    "src/agent_loop_lab/v03/sealed.py",
    "src/agent_loop_lab/v03/scoring.py",
    "src/agent_loop_lab/v03/runner.py",
    "src/agent_loop_lab/v03/infer.py",
    "src/agent_loop_lab/v03/model.py",
    "src/agent_loop_lab/v03/renderer.py",
    "src/agent_loop_lab/v03/tokenizer.py",
    "src/agent_loop_lab/eval/metrics.py",
    "src/agent_loop_lab/eval/m4.py",
    "src/agent_loop_lab/baselines/phase_c_v03.py",
    "experiments/v0_3/sealed/README.md",
)


def code_hashes() -> dict[str, str]:
    return {p: sha256_file(REPO / p) for p in CODE_FILES}


def write_json_atomic(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)
