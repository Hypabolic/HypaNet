"""v0.4 fresh sealed test set (plan v0.4, "Data, splits and sealing"). Harness only.

The v0.4 primary result is scored on a test set no one has seen any result on. It is the v0.2
generator, unchanged, with a new master seed. Each of the 12 families gets indices
``k = 0..PER_FAMILY-1``, all assigned to test, so there is no split shuffle and no dev, val or
counterfactual data for the new seed. With ``PER_FAMILY = 333`` that gives 3,996 sessions, the
size and family balance of the v0.2/v0.3 test split. Every session is captured with its raw
operations, so the frozen v0.3 renderer can render it.

**Sealing order** (``scripts/v04_seal_fresh_test.py``):

1. ``register`` writes :data:`REGISTRATION` (master seed, formulas, code hashes). It is committed
   before anything is generated, so the seed cannot be chosen after seeing data.
2. ``build`` refuses unless that registration is committed and the code is unchanged since. It
   generates the files, regenerates them under a different ``PYTHONHASHSEED`` and compares hashes,
   then writes :data:`MANIFEST`. The manifest is committed before any v0.4 run that compares a
   Qwen output with a label (:func:`require_sealed`).
3. The v0.4 sealed evaluation is the only reader of the files, once.

The manifest has counts, hashes and label-free identity anchors only: windows, gate-passing
windows, and ``rows_sha256`` (the v0.3 sealed-eval row hash, so the sealed run can check its rows).
It has no label or class statistics: the test-set class prior is sealed information.

This module never opens a v0.2/v0.3 test, OOD or counterfactual file. It reads only
``datasets/v0_2/splits.json`` (session ids, for the disjointness check) and the manifests.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

from ..datasets.build import (
    GZIP_LEVEL,
    SEED_FORMULA,
    BuildError,
    _check_events,
    _GzLines,
    data_path,
    events_line,
    file_sha256,
    oracle_line,
    session_id_for,
    session_seed,
)
from ..oracle_config import DEFAULT_CONFIG_PATH, load_oracle_config
from ..v03.build_raw import _check_aligned, _sources_sha256, raw_line, raw_path
from ..v03.rawop import capture_raw

__all__ = [
    "FRESH_MASTER_SEED", "PER_FAMILY", "SPLIT", "FRESH_DIR", "MANIFEST", "REGISTRATION", "FreshTestError",
    "fresh_plan", "existing_session_ids", "check_disjoint", "check_generator", "code_sha256",
    "registration_record", "build_fresh_test", "verify_files", "require_sealed",
]

# Not used anywhere else in the tree. 20261104 is the v0.2 master, bootstrap and disclosure seed;
# 20261105 prefixes the v0.3 disclosure, budget and early-stopping keys.
FRESH_MASTER_SEED = 20261106
PER_FAMILY = 333
SPLIT = "test"  # file names follow the sealed test.* convention

REPO = Path(__file__).resolve().parents[3]
FRESH_DIR = REPO / "datasets" / "v0_4" / "fresh_test"
MANIFEST = FRESH_DIR / "MANIFEST_fresh_test.json"
REGISTRATION = REPO / "experiments" / "v0_4" / "fresh_test" / "REGISTRATION.json"
V02_DATA = REPO / "datasets" / "v0_2"
V03_MANIFEST = REPO / "datasets" / "v0_3" / "MANIFEST_v0_3.json"
V02_REGISTRY = REPO / "experiments" / "registry.yaml"
GENERATOR_VERSION_EXPECTED = "0.2.0"
FORMAT = "v04-fresh-test/1"

# Code whose bytes decide what is generated or how it is checked; their hashes are registered.
CODE_FILES = (
    "src/agent_loop_lab/v04/fresh_test.py",
    "scripts/v04_seal_fresh_test.py",
    "src/agent_loop_lab/datasets/build.py",
    "src/agent_loop_lab/v03/build_raw.py",
    "src/agent_loop_lab/v03/rawop.py",
    "src/agent_loop_lab/schema.py",
    "src/agent_loop_lab/windows.py",
    "src/agent_loop_lab/oracle_config.py",
    "src/agent_loop_lab/fingerprint.py",
    "src/agent_loop_lab/encode.py",
    "src/agent_loop_lab/eval/collisions.py",
    "src/agent_loop_lab/v03/sealed.py",
    "oracle_config.v0_2.yaml",
)
# Files on the generation path that the v0.2 freeze registry pins; each must still match it.
V02_FROZEN_GENERATION_FILES = (
    "oracle_config.v0_2.yaml",
    "src/agent_loop_lab/oracle_config.py",
    "src/agent_loop_lab/fingerprint.py",
    "src/agent_loop_lab/encode.py",
    "src/agent_loop_lab/schema.py",
    "src/agent_loop_lab/windows.py",
)


class FreshTestError(RuntimeError):
    pass


class _HashedGzLines(_GzLines):
    """``_GzLines`` that also hashes the uncompressed stream, which does not depend on the zlib build."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.content = hashlib.sha256()

    def write(self, line: str) -> None:
        super().write(line)
        self.content.update(line.encode("utf-8") + b"\n")


def _canon(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def fresh_plan(master_seed: int = FRESH_MASTER_SEED, per_family: int = PER_FAMILY) -> list[tuple[str, int, int, str]]:
    """``(family, k, seed, session_id)`` in file order: FAMILY_NAMES order, then k."""
    from ..synth.families import FAMILY_NAMES

    out = []
    for family in FAMILY_NAMES:
        for k in range(per_family):
            seed = session_seed(master_seed, family, k)
            out.append((family, k, seed, session_id_for(seed)))
    return out


def existing_session_ids(v02_dir: Path = V02_DATA) -> set[str]:
    """Every standard-split session id (dev, val, test) from ``splits.json``; reads no data file.

    OOD ids start ``o-`` and counterfactual pair ids ``cf-``, so they cannot collide with ``s-`` ids."""
    splits = json.loads((v02_dir / "splits.json").read_text(encoding="utf-8"))["splits"]
    return {sid for ids in splits.values() for sid in ids}


def check_disjoint(plan: list[tuple[str, int, int, str]], existing: set[str]) -> None:
    ids = [p[3] for p in plan]
    if len(set(ids)) != len(ids):
        raise FreshTestError("fresh session ids are not unique")
    if not all(sid.startswith("s-") for sid in ids):
        raise FreshTestError("fresh session ids must use the s-{seed:016x} format")
    overlap = set(ids) & existing
    if overlap:
        raise FreshTestError(f"{len(overlap)} fresh session ids already exist in v0.2 splits")


def _registry_hash(rel: str) -> str | None:
    for line in V02_REGISTRY.read_text(encoding="utf-8").splitlines():
        key, _, val = line.strip().partition(": ")
        if key == rel:
            return val.strip()
    return None


def check_generator() -> dict[str, str]:
    """The generator and oracle config are exactly the ones v0.2/v0.3 data came from."""
    from ..synth.engine import GENERATOR_VERSION

    if GENERATOR_VERSION != GENERATOR_VERSION_EXPECTED:
        raise FreshTestError(f"generator version {GENERATOR_VERSION} != {GENERATOR_VERSION_EXPECTED}")
    synth = _sources_sha256("synth")["synth"]
    want = json.loads(V03_MANIFEST.read_text(encoding="utf-8"))["source_sha256"]["synth"]
    if synth != want:
        raise FreshTestError(f"synth sources {synth} differ from MANIFEST_v0_3.json ({want})")
    frozen = {}
    for rel in V02_FROZEN_GENERATION_FILES:
        got = file_sha256(REPO / rel)
        if got != _registry_hash(rel):
            raise FreshTestError(f"{rel} differs from experiments/registry.yaml")
        frozen[rel] = got
    return {"generator_version": GENERATOR_VERSION, "synth_sources_sha256": synth, "v02_frozen_files": frozen}


def code_sha256() -> dict[str, str]:
    return {rel: file_sha256(REPO / rel) for rel in CODE_FILES}


def registration_record(master_seed: int = FRESH_MASTER_SEED, per_family: int = PER_FAMILY) -> dict[str, Any]:
    from ..synth.families import FAMILY_NAMES

    plan = fresh_plan(master_seed, per_family)
    check_disjoint(plan, existing_session_ids())
    return {
        "format": FORMAT,
        "what": "v0.4 fresh sealed test set: registration (committed before any generation)",
        "master_seed": master_seed,
        "per_family": per_family,
        "families": list(FAMILY_NAMES),
        "sessions": len(plan),
        "seed_formula": SEED_FORMULA,
        "session_id_format": "s-{seed:016x}",
        "assignment": "every k in 0..per_family-1 of every family goes to test (no split shuffle)",
        "file_order": "FAMILY_NAMES order, then k",
        "session_ids_sha256": hashlib.sha256("\n".join(p[3] for p in plan).encode()).hexdigest(),
        "disjoint_from": {"datasets/v0_2/splits.json": file_sha256(V02_DATA / "splits.json"), "overlap": 0},
        "generator": check_generator(),
        "code_sha256": code_sha256(),
        "seed_choice": "20261106 is used nowhere else in the tree: 20261104 is the v0.2 master/bootstrap/disclosure "
                       "seed and 20261105 prefixes the v0.3 disclosure, budget and ES keys. Chosen before any "
                       "generation; never re-rolled.",
    }


def _identity_rows(rows: list) -> dict[str, Any]:
    from ..v03.sealed import rows_sha256

    return {"windows": len(rows), "gate_passed": sum(1 for r in rows if r.gate_passed), "rows_sha256": rows_sha256(rows)}


def build_fresh_test(out_dir: Path, *, master_seed: int = FRESH_MASTER_SEED, per_family: int = PER_FAMILY,
                     check_ids: bool = True, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Generate the fresh test files into ``out_dir``; return counts, file hashes and identity anchors.

    Writes ``standard/test.events.jsonl.gz``, ``standard/test.oracle.jsonl.gz`` and
    ``test.raw.jsonl.gz``. On any failure every output file is removed. Prints and returns no label
    or class statistics.
    """
    from ..eval.collisions import WindowRow
    from ..schema import OracleLabel
    from ..synth.families import generate_session
    from ..windows import gate, window_endpoints

    say = progress or (lambda _m: None)
    out = Path(out_dir)
    plan = fresh_plan(master_seed, per_family)
    if check_ids:
        check_disjoint(plan, existing_session_ids())
    cfg = load_oracle_config()
    (out / "standard").mkdir(parents=True, exist_ok=True)
    writers = {
        "events": _HashedGzLines(data_path(out, SPLIT, "events")),
        "oracle": _HashedGzLines(data_path(out, SPLIT, "oracle")),
        "raw": _HashedGzLines(raw_path(out, SPLIT)),
    }
    rows: list = []
    n_events = n_ops = 0
    families: dict[str, int] = {}
    try:
        with capture_raw() as cap:
            for family, k, seed, sid in plan:
                try:
                    events, record = generate_session(family, seed, sid, cfg)
                    _check_events(events, sid)
                except Exception as exc:
                    raise BuildError(f"family {family} k={k} seed={seed}: {type(exc).__name__}: {exc}") from exc
                if record.seed != seed or record.session_id != sid:
                    raise FreshTestError(f"{sid}: oracle record seed/id mismatch")
                raw = cap.pop(sid, events)
                _check_aligned(sid, events, raw)
                writers["events"].write(events_line(sid, events))
                writers["oracle"].write(oracle_line(record))
                writers["raw"].write(raw_line(sid, raw))
                n_events += len(events)
                n_ops += len(raw)
                families[family] = families.get(family, 0) + 1
                for pos in window_endpoints(events):
                    endpoint = events[pos].sequence_index
                    rows.append(WindowRow(session_id=sid, family=record.scenario_family, split=None, endpoint=endpoint,
                                          label=OracleLabel(record.window_labels[endpoint]),
                                          gate_passed=gate(events[: pos + 1]).passed, vector=None))
                if k == per_family - 1:
                    say(f"{family}: {per_family} sessions")
            unclaimed = len(cap)
        if unclaimed:
            raise FreshTestError(f"{unclaimed} finished sessions were not written")
        for w in writers.values():
            w.close()
    except BaseException:
        for w in writers.values():
            w.close()
            w.path.unlink(missing_ok=True)
        raise
    files = {w.path.relative_to(out).as_posix(): {"sha256": file_sha256(w.path), "bytes": w.path.stat().st_size,
                                                  "lines": w.lines, "content_sha256": w.content.hexdigest()}
             for w in writers.values()}
    return {
        "sessions": len(plan),
        "family_sessions": families,
        "events": n_events,
        "raw_ops": n_ops,
        "identity": _identity_rows(rows),
        "files": files,
    }


def verify_files(manifest: dict[str, Any], data_dir: Path = FRESH_DIR) -> None:
    """Re-hash the sealed files against the manifest (reads bytes only, decodes nothing)."""
    for rel, meta in manifest["files"].items():
        got = file_sha256(data_dir / rel)
        if got != meta["sha256"]:
            raise FreshTestError(f"{rel}: sha256 {got} != manifest {meta['sha256']}")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout


def committed_unchanged(path: Path) -> bool:
    """``path`` is tracked at HEAD and has no staged or unstaged change."""
    rel = path.relative_to(REPO).as_posix()
    if not _git("ls-files", "--", rel).strip():
        return False
    return not _git("status", "--porcelain", "--", rel).strip()


def require_sealed() -> dict[str, Any]:
    """Refuse unless the fresh-test manifest is committed unchanged at HEAD and the files still match
    it. Call it before any v0.4 step that compares a Qwen output with a label (probe, LoRA, scoring)."""
    if not MANIFEST.is_file() or not committed_unchanged(MANIFEST):
        raise FreshTestError(f"{MANIFEST.relative_to(REPO)} must be committed before any labelled v0.4 step")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if man["registration"]["sha256"] != file_sha256(REGISTRATION):
        raise FreshTestError("REGISTRATION.json differs from the one the manifest cites")
    verify_files(man)
    return man


def manifest_record(built: dict[str, Any], registration: dict[str, Any], rebuild: dict[str, Any]) -> dict[str, Any]:
    reg_rel = REGISTRATION.relative_to(REPO).as_posix()
    import platform
    import sys

    return {
        "format": FORMAT,
        "dataset": "v0_4-fresh-test",
        "registration": {"path": reg_rel, "sha256": file_sha256(REGISTRATION),
                         "commit": _git("log", "-1", "--format=%H", "--", reg_rel).strip()},
        "master_seed": registration["master_seed"],
        "per_family": registration["per_family"],
        "seed_formula": registration["seed_formula"],
        "session_id_format": registration["session_id_format"],
        "session_ids_sha256": registration["session_ids_sha256"],
        "generator": check_generator(),
        "code_sha256": code_sha256(),
        "extends": {rel: file_sha256(REPO / rel) for rel in (
            "datasets/v0_2/MANIFEST.json", "datasets/v0_2/OOD_MANIFEST.json", "datasets/v0_2/splits.json",
            "datasets/v0_3/MANIFEST_v0_3.json")},
        "gzip": {"mtime": 0, "compresslevel": GZIP_LEVEL, "header_filename": ""},
        "line_format": {
            "events": '{"session_id", "events": [CanonicalEvent.to_dict()...]}; canonical JSON',
            "oracle": "OracleRecord.to_dict(); canonical JSON; line i matches events line i",
            "raw": '{"session_id", "raw": [RawOp.to_dict()...]}; canonical JSON; line i matches events line i',
        },
        "sessions": built["sessions"],
        "family_sessions": built["family_sessions"],
        "events": built["events"],
        "raw_ops": built["raw_ops"],
        "identity": {**built["identity"], "rows_definition": "v03.sealed.rows_sha256 over [session_id, family, "
                     "endpoint, label, gate_passed] per window_endpoints row, file order"},
        "label_statistics": "none recorded (sealed)",
        "rebuild": rebuild,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "files": built["files"],
    }
