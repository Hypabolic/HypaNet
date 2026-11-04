"""Build the v0.3 raw-operation datasets (spec v0.3 §7.1-7.2).

Harness only. This module imports the generators and the dataset builders, so
it must never be imported from the weight-build path (``v03/__init__`` does
not import it; :mod:`.rawop` is the only module the renderer may depend on).

Procedure:

1. Open a :func:`~agent_loop_lab.v03.rawop.capture_raw` sink.
2. Re-run :func:`agent_loop_lab.datasets.build.build` and
   :func:`agent_loop_lab.datasets.ood.build_ood` **unchanged**, with the v0.2
   master seed and sizes, into a temporary directory. Their observe-only
   ``on_session`` hook hands each written session to this module, which pops
   the session's raw ops from the sink, checks ``len(raw) == len(events)`` and
   ``raw[i].family == events[i].operation_family``, and streams one line
   ``{"session_id", "raw": [RawOp.to_dict()...]}`` to
   ``<out>/{split}.raw.jsonl.gz`` (dev, val, test, counterfactual) or
   ``<out>/ood.raw.jsonl.gz``. Line order is therefore the v0.2 file order.
3. **Acceptance check**: the sha256 of every regenerated canonical events and
   oracle file must equal the hash recorded in the committed
   ``datasets/v0_2/MANIFEST.json`` / ``OOD_MANIFEST.json``. Any mismatch raises
   :class:`RawBuildError`, and the raw files are deleted (no plausible partial
   output is left behind). The temporary v0.2 manifests themselves are *not*
   compared: their ``generator_source_sha256`` covers the engine sources this
   change edits.
4. Write ``<out>/MANIFEST_v0_3.json``. Token statistics and the renderer/tokeniser hashes are added
   afterwards by :func:`agent_loop_lab.v03.token_stats.amend_manifest` (the raw capture itself does
   not depend on the renderer).

Nothing here computes labels, label statistics, predictions or metrics. (The
unchanged v0.2 builders compute their own label counts into the temporary
manifests, which are discarded and never copied.)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from ..datasets.build import (
    COUNTERFACTUAL,
    DEFAULT_MASTER_SEED,
    GZIP_LEVEL,
    MANIFEST_FILE,
    PAIRS_PER_KIND,
    SPLITS,
    STANDARD_SESSIONS,
    BuildError,
    _canon_json,
    _GzLines,
    _write_json,
    build,
    file_sha256,
)
from ..datasets.ood import OOD_MANIFEST, OOD_SESSIONS_PER_FAMILY, build_ood
from ..schema import CanonicalEvent, OracleRecord
from .rawop import DROPPED_VOCAB, FIELDS, RawCapture, RawOp, capture_raw

__all__ = ["RAW_SPLITS", "MANIFEST_V03", "RawBuildError", "raw_path", "raw_line", "iter_raw", "build_raw",
           "alignment_errors"]

RAW_SPLITS: tuple[str, ...] = (*SPLITS, COUNTERFACTUAL, "ood")
MANIFEST_V03 = "MANIFEST_v0_3.json"
_SRC_ROOT = Path(__file__).resolve().parents[1]  # src/agent_loop_lab


class RawBuildError(BuildError):
    """Raw capture misaligned, or the acceptance check (byte identity with v0.2) failed."""


def raw_path(out_dir: str | Path, split: str) -> Path:
    if split not in RAW_SPLITS:
        raise ValueError(f"split must be one of {RAW_SPLITS}, got {split!r}")
    return Path(out_dir) / f"{split}.raw.jsonl.gz"


def raw_line(session_id: str, raw: list[RawOp]) -> str:
    return _canon_json({"session_id": session_id, "raw": [r.to_dict() for r in raw]})


def iter_raw(out_dir: str | Path, split: str):
    """Yield ``(session_id, [RawOp...])`` in file order (strict key check per line)."""
    import gzip

    with gzip.open(raw_path(out_dir, split), "rt", encoding="utf-8") as f:
        for n, line in enumerate(f):
            d = json.loads(line)
            if set(d) != {"session_id", "raw"}:
                raise RawBuildError(f"{split} line {n + 1}: keys {sorted(d)} != ['raw', 'session_id']")
            yield d["session_id"], [RawOp.from_dict(r) for r in d["raw"]]


def alignment_errors(r: RawOp, e: CanonicalEvent) -> list[str]:
    """Ways ``r`` disagrees with its canonical event under the r1 schema (spec v0.3 §3); empty if aligned.

    Checks: family, outcome, emitted tokens and timestamp; each :data:`DROPPED_VOCAB` entry against the
    field it withholds (a dropped field is null on the event, and the event's ``None`` is explained by a
    drop); ``validation`` present exactly when the event produced one, with the true values agreeing with
    every observed count; ``failure`` only on ERROR events; ``changed`` sorted by path, unique, each entry
    a real change, and non-empty exactly when the event's workspace digests (when both observed) differ.
    """
    out: list[str] = []
    d = set(r.dropped)
    if r.family != e.operation_family:
        out.append(f"family {r.family!r} != {e.operation_family!r}")
    if r.outcome != e.outcome.value:
        out.append(f"outcome {r.outcome!r} != {e.outcome.value!r}")
    if (r.input_tokens, r.output_tokens, r.timestamp) != (e.input_tokens, e.output_tokens, e.timestamp):
        out.append("tokens/timestamp differ from the event")
    for name, value in (("input_tokens", e.input_tokens), ("output_tokens", e.output_tokens),
                        ("digest_before", e.workspace_state_before), ("digest_after", e.workspace_state_after)):
        if (name in d) != (value is None):
            out.append(f"dropped {name!r}={name in d} but the event value is {'null' if value is None else 'set'}")
    if "failure" in d and (e.outcome.value != "ERROR" or e.failure_fingerprint is not None):
        out.append("failure dropped on a non-ERROR event or with a fingerprint present")
    if e.outcome.value == "ERROR" and e.failure_fingerprint is None and "failure" not in d:
        out.append("ERROR event without a failure fingerprint and no failure drop")
    if (r.validation is not None) != e.validation_produced:
        out.append("validation presence differs from validation_produced")
    elif r.validation is not None:
        for name, ev_value, true_value in (
                ("validation_suite", e.validation_suite_id, r.validation["suite"]),
                ("validation_passed", e.validation_passed, r.validation["passed"]),
                ("validation_total", e.validation_total, r.validation["total"])):
            if (name in d) != (ev_value is None):
                out.append(f"dropped {name!r}={name in d} but the event value is {ev_value!r}")
            if ev_value is not None and ev_value != true_value:
                out.append(f"{name}: event {ev_value!r} != raw {true_value!r}")
    elif d & {"validation_suite", "validation_passed", "validation_total"}:
        out.append("validation drop on an event without validation")
    changed = r.changed or []
    paths = [c["path"] for c in changed]
    if paths != sorted(set(paths)):
        out.append("changed paths not sorted/unique")
    if any(c["before"] == c["after"] for c in changed):
        out.append("changed entry with before == after")
    if e.workspace_state_before is not None and e.workspace_state_after is not None:
        if bool(changed) != (e.workspace_state_before != e.workspace_state_after):
            out.append(f"changed has {len(changed)} entries but the workspace digests "
                       f"{'differ' if e.workspace_state_before != e.workspace_state_after else 'agree'}")
    return out


def _check_aligned(session_id: str, events: list[CanonicalEvent], raw: list[RawOp]) -> None:
    if len(raw) != len(events):
        raise RawBuildError(f"session {session_id!r}: {len(raw)} raw ops for {len(events)} events")
    for i, (r, e) in enumerate(zip(raw, events)):
        errs = alignment_errors(r, e)
        if errs:
            raise RawBuildError(f"session {session_id!r} event {i}: " + "; ".join(errs))


def _sources_sha256(*subdirs: str) -> dict[str, str]:
    out = {}
    for sub in subdirs:
        h = hashlib.sha256()
        root = _SRC_ROOT / sub
        for p in sorted(root.rglob("*.py"), key=lambda q: q.relative_to(root).as_posix()):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(root).as_posix().encode("utf-8")
            data = p.read_bytes()
            h.update(len(rel).to_bytes(8, "big") + rel + len(data).to_bytes(8, "big") + data)
        out[sub] = h.hexdigest()
    return out


def _expected_hashes(v02_dir: Path) -> dict[str, str]:
    std = json.loads((v02_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    ood = json.loads((v02_dir / OOD_MANIFEST).read_text(encoding="utf-8"))
    out = {rel: meta["sha256"] for rel, meta in std["files"].items()}
    for rel, sha in ood["files"].items():
        out[rel] = sha if isinstance(sha, str) else sha["sha256"]
    return out


def build_raw(
    out_dir: str | Path,
    *,
    v02_dir: str | Path,
    tmp_dir: str | Path | None = None,
    keep_tmp: bool = False,  # keep the default temporary regeneration directory
    master_seed: int = DEFAULT_MASTER_SEED,
    sessions: int = STANDARD_SESSIONS,
    pairs_per_kind: int = PAIRS_PER_KIND,
    ood_sessions_per_family: int = OOD_SESSIONS_PER_FAMILY,
    verify: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Regenerate v0.2 with raw capture, verify byte identity, write raw files and the v0.3 manifest.

    ``verify=False`` (small test builds only) skips the comparison with the
    committed v0.2 hashes and writes no manifest; the script never sets it.
    """
    say = progress or (lambda _m: None)
    out, v02 = Path(out_dir), Path(v02_dir)
    if out.resolve() == v02.resolve():
        raise RawBuildError("refusing to write raw files into the v0.2 dataset directory")
    out.mkdir(parents=True, exist_ok=True)
    expected = _expected_hashes(v02) if verify else {}
    own_tmp = tmp_dir is None
    if own_tmp:
        tmp = Path(tempfile.mkdtemp(prefix="v03_regen_"))
    else:
        tmp = Path(tmp_dir)
        tmp.mkdir(parents=True, exist_ok=True)
    if tmp.resolve() == v02.resolve():
        raise RawBuildError("the regeneration directory must not be the v0.2 dataset directory")

    writers = {s: _GzLines(raw_path(out, s)) for s in RAW_SPLITS}
    n_sessions = {s: 0 for s in RAW_SPLITS}
    n_ops = {s: 0 for s in RAW_SPLITS}
    timings: dict[str, float] = {}
    leftovers: dict[str, int] = {}

    def _write(split: str, cap: RawCapture, events: list[CanonicalEvent], record: OracleRecord) -> None:
        sid = record.session_id
        raw = cap.pop(sid, events)
        _check_aligned(sid, events, raw)
        writers[split].write(raw_line(sid, raw))
        n_sessions[split] += 1
        n_ops[split] += len(raw)

    try:
        t0 = time.monotonic()
        with capture_raw() as cap:
            build(tmp, master_seed=master_seed, sessions=sessions, pairs_per_kind=pairs_per_kind,
                  reports_dir=None, progress=say,
                  on_session=lambda split, ev, rec: _write(split, cap, ev, rec))
            leftovers["standard"] = len(cap)
        timings["standard_and_counterfactual_s"] = round(time.monotonic() - t0, 1)
        say(f"standard + counterfactual regenerated in {timings['standard_and_counterfactual_s']}s")
        t1 = time.monotonic()
        with capture_raw() as cap:
            build_ood(tmp, master_seed=master_seed, sessions_per_family=ood_sessions_per_family,
                      on_session=lambda ev, rec: _write("ood", cap, ev, rec))
            leftovers["ood"] = len(cap)
        timings["ood_s"] = round(time.monotonic() - t1, 1)
        say(f"ood regenerated in {timings['ood_s']}s")
        for w in writers.values():
            w.close()

        acceptance: dict[str, Any] = {}
        if verify:
            regenerated = {rel: file_sha256(tmp / rel) for rel in expected}
            per_file = {rel: {"v0_2_sha256": expected[rel], "regenerated_sha256": regenerated[rel],
                              "byte_identical": expected[rel] == regenerated[rel]} for rel in sorted(expected)}
            bad = [rel for rel, v in per_file.items() if not v["byte_identical"]]
            for rel, v in per_file.items():
                say(f"acceptance {rel}: {'IDENTICAL' if v['byte_identical'] else 'MISMATCH'}")
            if bad:
                raise RawBuildError(f"ACCEPTANCE FAILED: regenerated files differ from v0.2: {bad}")
            acceptance = {"canonical_files": per_file, "all_byte_identical": True}
    except BaseException:
        for w in writers.values():
            w.close()
            w.path.unlink(missing_ok=True)
        (out / MANIFEST_V03).unlink(missing_ok=True)
        raise
    finally:
        # A caller-supplied tmp_dir is never deleted; the default temporary one is unless keep_tmp.
        if own_tmp and not keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    files = {w.path.name: {"sha256": file_sha256(w.path), "bytes": w.path.stat().st_size, "lines": w.lines}
             for _, w in sorted(writers.items(), key=lambda kv: kv[1].path.name)}
    manifest: dict[str, Any] = {
        "dataset": "v0_3-raw",
        "spec_version": "0.3 (DRAFT)",
        "extends": {
            f"datasets/v0_2/{MANIFEST_FILE}": file_sha256(v02 / MANIFEST_FILE),
            f"datasets/v0_2/{OOD_MANIFEST}": file_sha256(v02 / OOD_MANIFEST),
        } if verify else {},
        "master_seed": master_seed,
        "procedure": ("v0.2 build() and build_ood() re-run unchanged with raw capture; raw line i of each split "
                      "is the session on line i of the matching v0.2 events file"),
        "line_format": '{"session_id", "raw": [RawOp.to_dict()...]}; canonical JSON (sorted keys, no whitespace)',
        "gzip": {"mtime": 0, "compresslevel": GZIP_LEVEL, "header_filename": ""},
        "rawop_fields": list(FIELDS),
        "dropped_vocab": list(DROPPED_VOCAB),
        "source_sha256": _sources_sha256("synth", "synth_ood", "v03"),
        # Filled by agent_loop_lab.v03.token_stats.amend_manifest (renderer-dependent; dev/val only).
        "renderer_source_sha256": None,
        "tokeniser_source_sha256": None,
        "token_statistics": None,
        "sessions": n_sessions,
        "raw_ops": n_ops,
        "unclaimed_finished_sessions": leftovers,
        "acceptance": acceptance,
        "files": files,
    }
    if verify:
        _write_json(out / MANIFEST_V03, manifest)
    manifest["_timings"] = timings  # returned to the caller only, not written (keeps the manifest reproducible)
    return manifest
