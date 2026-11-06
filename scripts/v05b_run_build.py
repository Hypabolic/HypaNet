"""Build and archive the v0.5 Phase B revision: r1's weights + the author's REVISION.edits.

Usage:
    uv run python scripts/v05b_run_build.py [--author-dir DIR] [--r1-dir DIR] [--out-dir DIR] [--force]
                                            [--registry PATH | --no-registry]

Defaults: AUTHOR_DIR ../HypaNet-v05b-author, R1_DIR experiments/v0_5/runs/r1, OUT_DIR weight_program/v0_5/phase-b.

Checks, in order (the first failure rejects the build):

1. r1's archived ``artifact.dlw`` matches its ``artifact.sha256``, and the author dir's ``r1/artifact.dlw`` is
   byte-identical to it; the author dir's packet manifest matches the registered v0.3 packet sha256
   (experiments/registry_v0_3.yaml, unless --no-registry);
2. ``output/v05-phase-b/`` holds the four deliverables ``REVISION.edits``, ``CIRCUITS.md``, ``REVISION.md`` and
   ``ITERATIONS.md`` (the three .md files non-empty), no symlinks, at most 256 MB in all; REVISION.edits is UTF-8
   and at most 64 MB;
3. REVISION.edits is applied in order on top of r1's weights by agent_loop_lab.v05.revision (the frozen v0.5
   reader's lexer, parser and apply step); every statement must read and apply and the file may hold no fence
   or END OF WEIGHTS line (``revision.build_problems``), otherwise the build is rejected;
4. the revised state dict is valid (names, shapes, float32, finite) and two applications give identical bytes.

On success OUT_DIR receives ``artifact.dlw`` (DLW3; metadata names the base artefact and the revision sha256),
``artifact.sha256``, ``apply.json`` (the applier's report, like r1's apply.json), ``BUILD_RECORD.json``, copies of
the four deliverables and ``author_output/`` (the whole ``output/v05-phase-b/``, ``scratch/`` included). Every
run, built or rejected, logs ``OUT_DIR/attempts/NNNN.json`` (with the sha256 of every file in the output dir);
a rejection also writes ``OUT_DIR/attempts/NNNN.error.txt``, the error text capped at 2000 characters, which is
exactly what ``scripts/v05b_launch.sh resubmit N ERROR_FILE`` may disclose. An existing artefact is never
replaced without --force (the first valid build is the Phase B result). The script does not git-commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_loop_lab.v03.artifact import deserialize_state_dict, serialize_state_dict, validate_state_dict  # noqa: E402
from agent_loop_lab.v05.edits import READER_VERSION, render_report  # noqa: E402
from agent_loop_lab.v05.revision import REVISION_FILE, REVISION_VERSION, apply_revision, build_problems  # noqa: E402

PHASE = "v05-phase-b"
OUTPUT_REL = f"output/{PHASE}"
REQUIRED = (REVISION_FILE, "CIRCUITS.md", "REVISION.md", "ITERATIONS.md")
ERROR_CAP = 2000
MAX_EDITS_BYTES = 64 << 20
MAX_OUTPUT_BYTES = 256 << 20
RECORD_FILE = "BUILD_RECORD.json"
DEFAULT_AUTHOR_DIR = REPO_ROOT.parent / "HypaNet-v05b-author"
DEFAULT_R1_DIR = REPO_ROOT / "experiments" / "v0_5" / "runs" / "r1"
DEFAULT_OUT_DIR = REPO_ROOT / "weight_program" / "v0_5" / "phase-b"
DEFAULT_REGISTRY = REPO_ROOT / "experiments" / "registry_v0_3.yaml"


class BuildError(RuntimeError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_sha_file(p: Path) -> str:
    if not p.is_file():
        raise BuildError(f"{p.name} missing")
    h = (p.read_text().split() or [""])[0]
    if not re.fullmatch(r"[0-9a-f]{64}", h):
        raise BuildError(f"{p.name} does not hold a sha256")
    return h


def cap_error(text: str) -> str:
    tail = " ... (truncated)"
    return text if len(text) <= ERROR_CAP else text[: ERROR_CAP - len(tail)] + tail


def output_manifest(out: Path) -> dict[str, str]:
    files = {}
    for p in sorted(out.rglob("*")):
        if "__pycache__" in p.relative_to(out).parts:
            continue
        if p.is_symlink():
            raise BuildError(f"{OUTPUT_REL}/{p.relative_to(out).as_posix()} is a symlink (not allowed)")
        if p.is_file():
            files[p.relative_to(out).as_posix()] = _sha(p.read_bytes())
    return files


def registered_packet_sha(registry: Path) -> str:
    m = re.search(r"^packet_sha256:\s*([0-9a-f]{64})\s*$", registry.read_text(), re.MULTILINE)
    if not m:
        raise BuildError(f"no packet_sha256 in {registry}")
    return m.group(1)


def build(author_dir: Path, r1_dir: Path, registry: Path | None) -> tuple[bytes, dict, dict]:
    """``(artefact bytes, apply report, record)``; raises BuildError with the author-facing text."""
    out = author_dir / OUTPUT_REL
    # 1. inputs
    base_bytes = (r1_dir / "artifact.dlw").read_bytes() if (r1_dir / "artifact.dlw").is_file() else None
    if base_bytes is None:
        raise BuildError("harness: r1 artefact missing")
    base_sha = _sha(base_bytes)
    if base_sha != _read_sha_file(r1_dir / "artifact.sha256"):
        raise BuildError("harness: r1 artefact does not match its sha256")
    handed = author_dir / "r1" / "artifact.dlw"
    if not handed.is_file() or handed.read_bytes() != base_bytes:
        raise BuildError("harness: r1/artifact.dlw in the author directory differs from the archived r1 artefact")
    packet_sha = _sha((author_dir / "packet" / "PACKET_MANIFEST.json").read_bytes())
    if registry is not None and packet_sha != registered_packet_sha(registry):
        raise BuildError("harness: the packet does not match the registered v0.3 packet")
    # 2. deliverables
    if not out.is_dir():
        raise BuildError(f"{OUTPUT_REL}/ is missing")
    manifest = output_manifest(out)
    missing = [f for f in REQUIRED if not (out / f).is_file()]
    if missing:
        raise BuildError(f"missing required file(s) in {OUTPUT_REL}/: {', '.join(missing)}")
    empty = [f for f in REQUIRED[1:] if not (out / f).read_text(encoding="utf-8", errors="replace").strip()]
    if empty:
        raise BuildError(f"empty required file(s) in {OUTPUT_REL}/: {', '.join(empty)}")
    total = sum((out / f).stat().st_size for f in manifest)
    if total > MAX_OUTPUT_BYTES:
        raise BuildError(f"{OUTPUT_REL}/ holds {total} bytes (at most {MAX_OUTPUT_BYTES})")
    raw = (out / REVISION_FILE).read_bytes()
    if len(raw) > MAX_EDITS_BYTES:
        raise BuildError(f"{REVISION_FILE} is {len(raw)} bytes (at most {MAX_EDITS_BYTES})")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BuildError(f"{REVISION_FILE} is not UTF-8 (byte {exc.start})") from None
    # 3. apply
    base_sd, _base_meta = deserialize_state_dict(base_bytes)
    sd, report = apply_revision(base_sd, text)
    problems = build_problems(report)
    if problems:
        er = report["edit_report"]
        head = (f"{REVISION_FILE} rejected: {len(report['structural_problems'])} structural line(s) and "
                f"{er['skipped']} statement(s) that could not be read or applied (every statement must apply; "
                f"check with ./py -m v05_load check):")
        raise BuildError(head + "\n" + "\n".join(problems))
    # 4. artefact
    validate_state_dict(sd)
    metadata = {
        "experiment": "v0.5",
        "phase": PHASE,
        "builder": "revision applier",
        "reader_version": READER_VERSION,
        "revision_version": REVISION_VERSION,
        "base_artifact_sha256": base_sha,
        "revision_sha256": _sha(raw),
        "packet_sha256": packet_sha,
    }
    data = serialize_state_dict(sd, metadata)
    sd2, _ = apply_revision(deserialize_state_dict(base_bytes)[0], text)
    if serialize_state_dict(sd2, metadata) != data:
        raise BuildError("harness: two applications of the revision differ")
    sha = _sha(data)
    apply_report = {"phase": PHASE, "base_artifact_sha256": base_sha, **report,
                    "artifact": {"file": "artifact.dlw", "sha256": sha, "metadata": metadata}}
    record = {
        "phase": PHASE,
        "artifact_sha256": sha,
        "base_artifact_sha256": base_sha,
        "revision_sha256": _sha(raw),
        "packet_sha256": packet_sha,
        "reader_version": READER_VERSION,
        "revision_version": REVISION_VERSION,
        "statements": report["edit_report"]["statements"],
        "entries_written": report["edit_report"]["entries_written"],
        "changed_entries_total": report["changed_entries_total"],
        "changed_entries": report["changed_entries"],
        "nonzero_before_total": report["nonzero_before_total"],
        "nonzero_after_total": report["nonzero_after_total"],
        "parameters": report["parameters"],
        "output_files": manifest,
        "checks": {"r1_artifact_sha256": True, "author_r1_identical": True,
                   "packet_registered": registry is not None, "required_files": list(REQUIRED),
                   "every_statement_applied": True, "rebuild_identical": True},
    }
    return data, apply_report, record


def _next_attempt(attempts: Path) -> int:
    nums = [int(m.group(1)) for p in attempts.glob("*.json") if (m := re.fullmatch(r"(\d{4})\.json", p.name))]
    return max(nums, default=0) + 1


def run(author_dir: Path, r1_dir: Path, out_dir: Path, *, registry: Path | None, force: bool = False) -> dict:
    author_dir, r1_dir, out_dir = author_dir.resolve(), r1_dir.resolve(), out_dir.resolve()
    if (out_dir / "artifact.dlw").exists() and not force:
        raise SystemExit(f"refusing: {out_dir}/artifact.dlw exists (the first valid build is the result; --force)")
    attempts = out_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    n = _next_attempt(attempts)
    att = {"attempt": n, "utc": _utc(), "phase": PHASE, "author_dir": str(author_dir), "r1_dir": str(r1_dir)}
    try:
        att["output_files"] = output_manifest(author_dir / OUTPUT_REL) if (author_dir / OUTPUT_REL).is_dir() else {}
    except BuildError:
        att["output_files"] = None
    try:
        data, apply_report, record = build(author_dir, r1_dir, registry)
    except BuildError as exc:
        err = cap_error(str(exc))
        att.update(status="rejected", error=err)
        (attempts / f"{n:04d}.json").write_text(json.dumps(att, indent=2, sort_keys=True) + "\n")
        (attempts / f"{n:04d}.error.txt").write_text(err + "\n")
        raise
    out = author_dir / OUTPUT_REL
    if (out_dir / "author_output").exists():
        shutil.rmtree(out_dir / "author_output")
    shutil.copytree(out, out_dir / "author_output", symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    if output_manifest(out_dir / "author_output") != record["output_files"]:
        raise BuildError("harness: the output directory changed while it was archived")
    for f in REQUIRED:
        shutil.copyfile(out / f, out_dir / f)
    (out_dir / "artifact.dlw").write_bytes(data)
    (out_dir / "artifact.sha256").write_text(f"{record['artifact_sha256']}  artifact.dlw\n")
    (out_dir / "apply.json").write_text(render_report(apply_report))
    record.update(utc=att["utc"], attempt=n, author_dir=str(author_dir),
                  r1_dir=os.path.relpath(r1_dir, REPO_ROOT) if str(r1_dir).startswith(str(REPO_ROOT) + "/") else str(r1_dir))
    (out_dir / RECORD_FILE).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    att.update(status="built", artifact_sha256=record["artifact_sha256"])
    (attempts / f"{n:04d}.json").write_text(json.dumps(att, indent=2, sort_keys=True) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--author-dir", type=Path, default=DEFAULT_AUTHOR_DIR)
    ap.add_argument("--r1-dir", type=Path, default=DEFAULT_R1_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--force", action="store_true")
    reg = ap.add_mutually_exclusive_group()
    reg.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    reg.add_argument("--no-registry", action="store_true")
    args = ap.parse_args(argv)
    try:
        record = run(args.author_dir, args.r1_dir, args.out_dir,
                     registry=None if args.no_registry else args.registry, force=args.force)
    except (BuildError, OSError, ValueError) as exc:
        print(f"ERROR: {cap_error(str(exc))}", file=sys.stderr)
        return 1
    print(f"{PHASE} archived in {args.out_dir}")
    print(f"artifact sha256: {record['artifact_sha256']}")
    print(f"base (r1):       {record['base_artifact_sha256']}")
    print(f"statements:      {record['statements']} (entries changed {record['changed_entries_total']})")
    print(f"non-zero params: {record['nonzero_before_total']} -> {record['nonzero_after_total']} / {record['parameters']}")
    print(f"record:          {args.out_dir / RECORD_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
