"""v0.3 validation scoring and the Phase B disclosure rule (spec v0.3 §9.3, §10, E7, E11).

Harness only; never part of the packet or the weight-build path. The pre-registration is
``experiments/v0_3/phase_b/README.md``; this module implements it.

**Split.** Everything here works on the validation split, hard-coded as :data:`SPLIT`.
:func:`load_val` checks it with ``eval.metrics.check_split`` before any dataset module is
imported or any file is opened. No argument selects another split. The sealed test and OOD
sets are scored once, at the times E7 sets, by a separate script.

**Data.** ``datasets/v0_3/val.raw.jsonl.gz`` (RawOps, in v0.2 line order) is rendered with
the frozen renderer and tokenised with the frozen tokeniser. ``datasets/v0_2/standard/val.*``
gives the canonical events (for the v0.2 runtime gate ``windows.gate``) and the oracle labels.
A window is ``(session_id, endpoint)``, where ``endpoint`` is the ``sequence_index`` of its last
event; it reads block ``pos`` = the event at list position ``pos``. The rendered and tokenised
sessions and the window rows are cached in ``datasets/v0_3/val.encoded.pt`` (gitignored). The
cache key holds the renderer and tokeniser source hashes and the input file hashes.

**Predictions.**

* Network: one causal forward pass per session (``infer.eoe_logits``); the class of window
  ``pos`` is the argmax at ``EOE_pos``.
* Reference policy: called once per gate-passing window on ``surface[:eoe[pos] + 1]`` and
  ``event_ord[:eoe[pos] + 1]``, only through ``runner.evaluate_policy``. Each chunk of sessions
  runs in a fresh interpreter under the BUILD guard, with a 0.5 s CPU limit per call and at most
  :data:`POLICY_WORKERS` chunks at a time. A ``None`` (timeout, exception, invalid value) is a
  policy failure. In H2 it counts as a disagreement. In H1 it counts as a wrong prediction,
  recorded as the lowest class that is neither the network's class nor the oracle class
  (:func:`fill_policy_failures`).

**Disclosure rule** (:func:`disclosure_selection`): needs only session ids, endpoints, gate
results and oracle labels, never a model output.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

from ..eval import metrics
from ..eval.collisions import WindowRow
from ..schema import CLASS_NAMES, EVENTS_PER_WINDOW, ClassLabel, OracleLabel
from . import artifact as v03_artifact
from . import infer, renderer, tokenizer
from .model import predict

__all__ = [
    "SPLIT",
    "PHASE_DIRS",
    "WINDOW_MARKER",
    "DISCLOSURE_KEY_FORMAT",
    "PER_CLASS",
    "ValData",
    "load_val",
    "disclosure_key",
    "disclosure_selection",
    "disclosed_sessions",
    "phase_dir",
    "load_phase",
    "network_predictions",
    "policy_predictions",
    "fill_policy_failures",
    "score",
    "render_disclosed_prefix",
    "parse_disclosed_prefix",
]

SPLIT = "val"  # hard-coded (experiments/v0_3/phase_b/README.md §1); there is no override
assert frozenset({SPLIT}) == metrics.ALLOWED_SPLITS

REPO = Path(__file__).resolve().parents[3]
V03_DATA = REPO / "datasets" / "v0_3"
V02_DATA = REPO / "datasets" / "v0_2"
WEIGHT_ROOT = REPO / "weight_program" / "v0_3"
CACHE_PATH = V03_DATA / f"{SPLIT}.encoded.pt"
CACHE_FORMAT = "v03-val-encoded/1"

# CLI phase name -> archive directory under weight_program/v0_3/ (the paths of scripts/v03_launch.sh).
PHASE_DIRS = {"v03-phase-a": "phase-a", "v03-phase-b": "phase-b"}
ARCHIVE_FILES = ("artifact.dlw", "artifact.sha256", "reference_policy.py")

DISCLOSURE_KEY_FORMAT = "20261105:v03-phase-b-disclosure:{session_id}:{endpoint}"
PER_CLASS = 50
WINDOW_MARKER = "### window start"

POLICY_WORKERS = 2
POLICY_CHUNK_SESSIONS = 100
TORCH_THREADS = 2
_SEP = "\x00"
_LABELS = tuple(OracleLabel)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Validation data (rendered + tokenised sessions, gate, labels)
# ---------------------------------------------------------------------------


@dataclass
class EncodedSession:
    tokens: list[int]
    event_ord: list[int]
    offset: list[int]
    eoe_index: list[int]
    surface: list[str]


@dataclass
class ValData:
    session_ids: list[str]
    families: list[str]
    sessions: list[EncodedSession]
    rows: list[WindowRow]
    row_session: np.ndarray  # session index per row
    row_pos: np.ndarray  # block position (list index of the window's last event) per row
    inputs: dict[str, Any] = field(default_factory=dict)

    def rows_of(self, s: int) -> np.ndarray:
        """Row indices of session ``s`` (rows are grouped by session, in session order)."""
        a, b = np.searchsorted(self.row_session, [s, s + 1])
        return np.arange(a, b)


def _input_hashes() -> dict[str, Any]:
    return {
        "renderer_sha256": _sha256_file(Path(renderer.__file__)),
        "tokenizer_sha256": _sha256_file(Path(tokenizer.__file__)),
        f"datasets/v0_3/{SPLIT}.raw.jsonl.gz": _sha256_file(V03_DATA / f"{SPLIT}.raw.jsonl.gz"),
        f"datasets/v0_2/standard/{SPLIT}.events.jsonl.gz": _sha256_file(V02_DATA / "standard" / f"{SPLIT}.events.jsonl.gz"),
        f"datasets/v0_2/standard/{SPLIT}.oracle.jsonl.gz": _sha256_file(V02_DATA / "standard" / f"{SPLIT}.oracle.jsonl.gz"),
        "datasets/v0_3/MANIFEST_v0_3.json": _sha256_file(V03_DATA / "MANIFEST_v0_3.json"),
    }


def _check_manifest(hashes: dict[str, Any]) -> None:
    man = json.loads((V03_DATA / "MANIFEST_v0_3.json").read_text())
    for key, mkey in (("renderer_sha256", "renderer_source_sha256"), ("tokenizer_sha256", "tokeniser_source_sha256")):
        if man.get(mkey) != hashes[key]:
            raise ValueError(f"{key} {hashes[key]} differs from MANIFEST_v0_3.json {mkey} {man.get(mkey)}")


def _build_val(limit: int | None) -> dict[str, Any]:
    from ..datasets.build import iter_sessions  # imported only after the split check
    from ..windows import gate, validate_sequence, window_endpoints
    from .build_raw import alignment_errors, iter_raw

    ids: list[str] = []
    fams: list[str] = []
    tok, eo, off, eoe, lengths, n_events, surfaces = [], [], [], [], [], [], []
    r_sess, r_pos, r_end, r_lab, r_gate = [], [], [], [], []
    pairs = zip(iter_raw(V03_DATA, SPLIT), iter_sessions(V02_DATA, SPLIT), strict=True)
    for n, ((sid, raw), (events, record)) in enumerate(pairs):
        if limit is not None and n >= limit:
            break
        if sid != record.session_id or len(raw) != len(events):
            raise ValueError(f"{SPLIT} line {n + 1}: raw/events/oracle misaligned for {sid!r}")
        for k, (r, e) in enumerate(zip(raw, events)):
            errs = alignment_errors(r, e)
            if errs:
                raise ValueError(f"{sid} event {k}: raw op misaligned: " + "; ".join(errs))
        blocks = renderer.render_session([r.to_dict() for r in raw], sid)
        enc = tokenizer.encode_session(blocks)
        if any(_SEP in s for s in enc.surface):
            raise ValueError(f"{sid}: surface token holds the cache separator")
        if len(enc.eoe_index) != len(events):
            raise ValueError(f"{sid}: {len(enc.eoe_index)} EOE tokens for {len(events)} events")
        s = len(ids)
        ids.append(sid)
        fams.append(record.scenario_family)
        tok.extend(enc.tokens), eo.extend(enc.event_ord), off.extend(enc.offset), eoe.extend(enc.eoe_index)
        lengths.append(len(enc.tokens))
        n_events.append(len(enc.eoe_index))
        surfaces.append(_SEP.join(enc.surface))
        validate_sequence(events)
        for pos in window_endpoints(events):
            endpoint = events[pos].sequence_index
            try:
                label = OracleLabel(record.window_labels[endpoint])
            except KeyError:
                raise ValueError(f"{sid}: no oracle label for window endpoint {endpoint}") from None
            r_sess.append(s), r_pos.append(pos), r_end.append(endpoint)
            r_lab.append(_LABELS.index(label)), r_gate.append(gate(events[: pos + 1]).passed)
    if max(tok, default=0) >= 2**15 or max(eo, default=0) >= 2**15:
        raise ValueError("token id or event ordinal does not fit int16")
    return {
        "session_ids": ids,
        "families": fams,
        "tokens": torch.tensor(tok, dtype=torch.int16),
        "event_ord": torch.tensor(eo, dtype=torch.int16),
        "offset": torch.tensor(off, dtype=torch.int16),
        "eoe_index": torch.tensor(eoe, dtype=torch.int32),
        "lengths": torch.tensor(lengths, dtype=torch.int64),
        "n_events": torch.tensor(n_events, dtype=torch.int64),
        "surface": surfaces,
        "row_session": torch.tensor(r_sess, dtype=torch.int32),
        "row_pos": torch.tensor(r_pos, dtype=torch.int32),
        "row_endpoint": torch.tensor(r_end, dtype=torch.int32),
        "row_label": torch.tensor(r_lab, dtype=torch.int8),
        "row_gate": torch.tensor(r_gate, dtype=torch.bool),
    }


def _from_payload(p: dict[str, Any]) -> ValData:
    tokens, event_ord, offset, eoe = (p[k].tolist() for k in ("tokens", "event_ord", "offset", "eoe_index"))
    sessions: list[EncodedSession] = []
    a = e = 0
    for n_tok, n_ev, surf in zip(p["lengths"].tolist(), p["n_events"].tolist(), p["surface"], strict=True):
        sessions.append(EncodedSession(tokens[a : a + n_tok], event_ord[a : a + n_tok], offset[a : a + n_tok],
                                       eoe[e : e + n_ev], surf.split(_SEP)))
        a += n_tok
        e += n_ev
    ids, fams = list(p["session_ids"]), list(p["families"])
    r_sess = p["row_session"].numpy().astype(np.int64)
    r_pos = p["row_pos"].numpy().astype(np.int64)
    rows = [
        WindowRow(session_id=ids[s], family=fams[s], split=SPLIT, endpoint=int(end), label=_LABELS[lab],
                  gate_passed=bool(g), vector=None)
        for s, end, lab, g in zip(r_sess.tolist(), p["row_endpoint"].tolist(), p["row_label"].tolist(),
                                  p["row_gate"].tolist())
    ]
    if np.any(np.diff(r_sess) < 0):
        raise ValueError("window rows are not grouped by session")
    return ValData(ids, fams, sessions, rows, r_sess, r_pos)


def load_val(*, cache: bool = True, limit: int | None = None, cache_path: Path | None = None) -> ValData:
    """Rendered and tokenised validation sessions with their window rows (the only split allowed).

    ``limit`` (Python API only, for tests) keeps the first ``limit`` sessions and is never cached.
    """
    metrics.check_split(SPLIT)
    hashes = _input_hashes()
    _check_manifest(hashes)
    key = {"format": CACHE_FORMAT, "split": SPLIT, **hashes}
    path = cache_path or CACHE_PATH
    use_cache = cache and limit is None
    payload = None
    if use_cache and path.is_file():
        try:
            stored = torch.load(path, weights_only=True)
        except Exception:  # noqa: BLE001 - a broken cache is rebuilt
            stored = None
        if isinstance(stored, dict) and stored.get("key") == key:
            payload = stored["payload"]
    if payload is None:
        payload = _build_val(limit)
        if use_cache:
            tmp = path.with_suffix(".pt.tmp")
            torch.save({"key": key, "payload": payload}, tmp)
            tmp.replace(path)
    data = _from_payload(payload)
    data.inputs = {"cache_key": key, "limit": limit}
    return data


# ---------------------------------------------------------------------------
# Phase B disclosure rule (README §2)
# ---------------------------------------------------------------------------


def disclosure_key(session_id: str, endpoint: int) -> str:
    return hashlib.sha256(DISCLOSURE_KEY_FORMAT.format(session_id=session_id, endpoint=endpoint).encode()).hexdigest()


def disclosure_selection(rows: Sequence[WindowRow], per_class: int = PER_CLASS) -> list[tuple[str, int]]:
    """``(key, row index)`` of the disclosed windows, sorted by key (README §2).

    Classes in canonical order; within a class ascending key; a window is skipped if its session
    was already used by any class. Raises if a class cannot reach ``per_class``.
    """
    by_class: dict[ClassLabel, list[tuple[str, int]]] = {c: [] for c in ClassLabel}
    for i, r in enumerate(rows):
        if r.split is not None:
            metrics.check_split(r.split)
        if r.gate_passed and r.label.is_scored:
            by_class[r.label.to_class()].append((disclosure_key(r.session_id, r.endpoint), i))
    used: set[str] = set()
    chosen: list[tuple[str, int]] = []
    for c in ClassLabel:
        cands = sorted(by_class[c])
        if len({k for k, _ in cands}) != len(cands):
            raise ValueError(f"{c.name}: duplicate disclosure keys")
        taken = 0
        for k, i in cands:
            sid = rows[i].session_id
            if sid in used:
                continue
            chosen.append((k, i))
            used.add(sid)
            taken += 1
            if taken == per_class:
                break
        if taken < per_class:
            raise ValueError(f"{c.name}: only {taken} windows from unused sessions, need {per_class}")
    chosen.sort()
    return chosen


def disclosed_sessions(rows: Sequence[WindowRow], per_class: int = PER_CLASS) -> frozenset[str]:
    return frozenset(rows[i].session_id for _, i in disclosure_selection(rows, per_class))


def _session_set_sha256(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def render_disclosed_prefix(blocks: Sequence[str], pos: int) -> str:
    """Blocks ``0..pos`` joined with newlines, with :data:`WINDOW_MARKER` on its own line before block ``pos - 7``."""
    start = pos - (EVENTS_PER_WINDOW - 1)
    if start < 0 or pos >= len(blocks):
        raise ValueError(f"window end {pos} out of range for {len(blocks)} blocks")
    lines: list[str] = []
    for k, b in enumerate(blocks[: pos + 1]):
        if k == start:
            lines.append(WINDOW_MARKER)
        lines.extend(b.split("\n"))
    text = "\n".join(lines)
    if parse_disclosed_prefix(text) != (list(blocks[: pos + 1]), start):
        raise ValueError("disclosed prefix does not round-trip")
    return text


def parse_disclosed_prefix(text: str) -> tuple[list[str], int]:
    """Inverse of :func:`render_disclosed_prefix`: ``(blocks, index of the marked block)``."""
    blocks: list[list[str]] = []
    marked: int | None = None
    pending = False
    for line in text.split("\n"):
        if line == WINDOW_MARKER:
            if marked is not None or pending:
                raise ValueError("more than one window marker")
            pending = True
            continue
        if not line.startswith(" "):
            blocks.append([line])
            if pending:
                marked, pending = len(blocks) - 1, False
        else:
            if not blocks or pending:
                raise ValueError("body line outside a block")
            blocks[-1].append(line)
    if marked is None:
        raise ValueError("no window marker")
    return ["\n".join(b) for b in blocks], marked


# ---------------------------------------------------------------------------
# Archived phases
# ---------------------------------------------------------------------------


def phase_dir(phase: str, weight_dir: Path | None = None) -> Path:
    """Archive directory of ``phase`` (``weight_dir`` overrides it; the split never changes)."""
    if phase not in PHASE_DIRS:
        raise ValueError(f"not a v0.3 phase: {phase!r} (expected one of {sorted(PHASE_DIRS)})")
    d = Path(weight_dir) if weight_dir is not None else WEIGHT_ROOT / PHASE_DIRS[phase]
    for f in ARCHIVE_FILES:
        if not (d / f).is_file():
            raise FileNotFoundError(f"{d / f} missing (only archived phases are scored)")
    return d


def load_phase(phase: str, d: Path) -> tuple[Any, dict, Path, dict[str, str]]:
    """Archived network and the path of its reference policy.

    Checks: ``artifact.dlw`` against ``artifact.sha256``; the artefact metadata names ``phase``, the
    registered packet (``experiments/registry_v0_3.yaml``) and the archived policy's sha256; the
    policy passes ``source_rules.check_reference_policy``."""
    from .source_rules import check_reference_policy

    dlw = d / "artifact.dlw"
    recorded = (d / "artifact.sha256").read_text().split()[0]
    actual = v03_artifact.artifact_sha256(dlw)
    if actual != recorded:
        raise ValueError(f"{dlw}: sha256 {actual} != recorded {recorded}")
    model, meta = v03_artifact.load_artifact(dlw)
    model.eval()
    pol = d / "reference_policy.py"
    pol_sha = _sha256_file(pol)
    if meta.get("phase") != phase:
        raise ValueError(f"{dlw}: artefact metadata phase {meta.get('phase')!r} != {phase!r}")
    if meta.get("reference_policy_sha256") != pol_sha:
        raise ValueError(f"{pol}: sha256 {pol_sha} != artefact metadata reference_policy_sha256")
    import yaml

    registered = yaml.safe_load((REPO / "experiments" / "registry_v0_3.yaml").read_text()).get("packet_sha256")
    if meta.get("packet_sha256") != registered:
        raise ValueError(f"{dlw}: artefact packet_sha256 {meta.get('packet_sha256')!r} != registered {registered!r}")
    problems = check_reference_policy(pol)
    if problems:
        raise ValueError(f"{pol} fails the reference_policy static rules: {problems}")
    return model, meta, pol, {"artifact_sha256": actual, "reference_policy_sha256": pol_sha}


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def network_predictions(model: Any, data: ValData, sessions: Sequence[int] | None = None
                        ) -> tuple[np.ndarray, dict[str, Any]]:
    """Network class per row (-1 for rows of sessions not run) and CPU latency statistics."""
    torch.set_num_threads(TORCH_THREADS)
    net = np.full(len(data.rows), -1, dtype=np.int64)
    order = range(len(data.sessions)) if sessions is None else sessions
    cpu, wall, ntok = [], [], []
    for s in order:
        enc = data.sessions[s]
        t0, w0 = time.process_time(), time.perf_counter()
        pred = predict(infer.eoe_logits(model, enc)).numpy()
        cpu.append(time.process_time() - t0)
        wall.append(time.perf_counter() - w0)
        ntok.append(len(enc.tokens))
        idx = data.rows_of(s)
        net[idx] = pred[data.row_pos[idx]]
    c, w = np.asarray(cpu), np.asarray(wall)
    lat = {
        "torch_threads": TORCH_THREADS,
        "sessions": len(cpu),
        "tokens": int(sum(ntok)),
        "cpu_s_total": float(c.sum()),
        "wall_s_total": float(w.sum()),
        "cpu_s_per_session": {"mean": float(c.mean()), "p50": float(np.percentile(c, 50)),
                              "p95": float(np.percentile(c, 95)), "max": float(c.max())} if len(c) else None,
        "cpu_ms_per_1k_tokens": float(1e6 * c.sum() / max(1, sum(ntok))),
        "definition": "time.process_time (all threads of this process) and perf_counter around one infer.eoe_logits call per session",
    }
    return net, lat


def _policy_jobs(data: ValData, sessions: Sequence[int]) -> tuple[list[int], list[tuple[list[str], list[int]]]]:
    idx_out: list[int] = []
    inputs: list[tuple[list[str], list[int]]] = []
    for s in sessions:
        enc = data.sessions[s]
        for i in data.rows_of(s).tolist():
            if not data.rows[i].gate_passed:
                continue
            e = enc.eoe_index[int(data.row_pos[i])] + 1
            idx_out.append(i)
            inputs.append((enc.surface[:e], enc.event_ord[:e]))
    return idx_out, inputs


def policy_predictions(policy_file: Path, data: ValData, *, sessions: Sequence[int] | None = None,
                       rows: Sequence[int] | None = None, workers: int = POLICY_WORKERS,
                       chunk_sessions: int = POLICY_CHUNK_SESSIONS,
                       evaluate: Callable[..., list[int | None]] | None = None,
                       progress: Callable[[str], None] | None = None) -> np.ndarray:
    """Policy class per row: 0..3, -2 for a policy failure, -1 where not run (gate-rejected, or not selected).

    Every call goes through ``runner.evaluate_policy`` (fresh guarded interpreter per chunk).
    ``rows`` restricts the calls to those (gate-passing) rows.
    """
    if evaluate is None:
        from .runner import evaluate_policy as evaluate
    if workers < 1 or workers > POLICY_WORKERS:
        raise ValueError(f"workers must be 1..{POLICY_WORKERS}")
    order = list(range(len(data.sessions)) if sessions is None else sessions)
    keep = None if rows is None else set(int(i) for i in rows)
    chunks = [order[a : a + chunk_sessions] for a in range(0, len(order), chunk_sessions)]
    out = np.full(len(data.rows), -1, dtype=np.int64)

    def run(chunk: list[int]) -> tuple[list[int], list[int | None]]:
        idx, inputs = _policy_jobs(data, chunk)
        if keep is not None:
            pairs = [(i, x) for i, x in zip(idx, inputs) if i in keep]
            idx, inputs = [i for i, _ in pairs], [x for _, x in pairs]
        if not inputs:
            return idx, []
        return idx, evaluate(policy_file, inputs)

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, values in pool.map(run, chunks):
            if len(values) != len(idx):
                raise ValueError("policy evaluation returned the wrong number of values")
            for i, v in zip(idx, values):
                out[i] = v if (type(v) is int and 0 <= v <= 3) else -2
            done += 1
            if progress is not None:
                progress(f"policy chunk {done}/{len(chunks)} ({time.time() - t0:.0f}s)")
    return out


def fill_policy_failures(rows: Sequence[WindowRow], net: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Replace failures (-2) by the lowest class that is neither the network's class nor the oracle class."""
    ref = np.asarray(ref, dtype=np.int64).copy()
    for i in np.flatnonzero(ref == -2).tolist():
        avoid = {int(net[i])}
        if rows[i].label.is_scored:
            avoid.add(int(rows[i].label.to_class()))
        ref[i] = min(c for c in range(len(CLASS_NAMES)) if c not in avoid)
    return ref


def _failure_counts(rows: Sequence[WindowRow], ref: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    out = {"total": 0, "scored": 0, "ambiguous": 0, "instrumentation": 0, "calls": 0}
    for i in np.flatnonzero(mask).tolist():
        r = rows[i]
        if not r.gate_passed:
            continue
        out["calls"] += 1
        if ref[i] != -2:
            continue
        out["total"] += 1
        if r.label.is_scored:
            out["scored"] += 1
        elif r.label is OracleLabel.INSUFFICIENT_AMBIGUOUS:
            out["ambiguous"] += 1
        else:
            out["instrumentation"] += 1
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _summary(rep: dict) -> dict[str, Any]:
    o = rep["overall"]

    def f(stat: dict) -> dict:
        return {"value": stat["value"], "ci95": stat["ci95"]}

    return {
        m: {k: f(o[m][k]) for k in ("macro_f1", "false_alarm_rate", "accuracy")}
        for m in ("network", "reference_policy")
    } | {"h2_fidelity_all": f(o["h2_fidelity"]["all"]["agreement"]),
         "h2_fidelity_scored": f(o["h2_fidelity"]["scored"]["agreement"]),
         "sessions": o["sessions"], "scored_windows": rep["population"]["scored_windows"]}


POPULATIONS = {
    "excluding_disclosed": "every validation session except the 200 Phase B disclosed sessions (README §2); the primary v0.3 validation result (spec §10)",
    "all_val": "every validation session (reported for transparency)",
}


def score(data: ValData, net: np.ndarray, ref: np.ndarray, *, n_resamples: int = metrics.BOOTSTRAP_RESAMPLES,
          seed: int = metrics.BOOTSTRAP_SEED, disclosed: frozenset[str] | None = None) -> dict[str, Any]:
    """Both populations' ``metrics.score_report`` plus policy-failure counts; ``ref`` may hold -2 (failures)."""
    rows = data.rows
    if disclosed is None:
        disclosed = disclosed_sessions(rows)
    filled = fill_policy_failures(rows, net, ref)
    excl = np.array([r.session_id not in disclosed for r in rows], dtype=bool)
    out: dict[str, Any] = {
        "disclosed_sessions": {
            "count": len(disclosed),
            "sha256_of_sorted_ids": _session_set_sha256(disclosed),
            "source": "computed from the README §2 rule (no model output)",
        },
        "populations": dict(POPULATIONS),
        "summary": {},
    }
    for name, mask in (("excluding_disclosed", excl), ("all_val", np.ones(len(rows), dtype=bool))):
        idx = np.flatnonzero(mask)
        sub = [rows[i] for i in idx.tolist()]
        rep = metrics.score_report(sub, net[idx], filled[idx], n_resamples=n_resamples, seed=seed)
        rep["policy_failures"] = _failure_counts(rows, ref, mask)
        out[name] = rep
        out["summary"][name] = _summary(rep) | {"policy_failures": rep["policy_failures"]["total"]}
    return out
