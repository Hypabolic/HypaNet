"""Read a v0.5 run's replies into the v0.3 network: RUN_DIR/artifact.dlw, artifact.sha256, apply.json.

Usage:
    uv run python scripts/v05_apply_reply.py RUN_DIR
    uv run python scripts/v05_apply_reply.py RUN_DIR --check

Reads ``RUN_DIR/replies/NNN.md`` (the author's replies as the launcher wrote them, numbered
contiguously from 0 or 1; other files there are ignored) in order, applies every ``edits`` block of
every reply to a zero v0.3 state dict (``agent_loop_lab.v05.edits``: rules, policies and the report
format are in its docstring), and writes the DLW3 artefact (deterministic metadata, no timestamps),
its sha256 line and the apply report. The network is never run. Refuses to overwrite any of the
three files. ``--check`` reads the replies again and compares with the existing files byte for byte;
it writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_loop_lab.v05.edits import (  # noqa: E402
    APPLY_FILE,
    ARTIFACT_FILE,
    ARTIFACT_SHA_FILE,
    RunDirError,
    build_run,
    render_report,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the existing files; write nothing")
    args = ap.parse_args(argv)
    run_dir: Path = args.run_dir
    outs = {name: run_dir / name for name in (ARTIFACT_FILE, ARTIFACT_SHA_FILE, APPLY_FILE)}
    existing = [name for name, p in outs.items() if p.exists()]
    if not args.check and existing:
        print(f"ERROR: {run_dir}: refusing to overwrite {', '.join(existing)}", file=sys.stderr)
        return 1
    try:
        data, sha, report = build_run(run_dir)
    except (RunDirError, OSError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    files = {
        ARTIFACT_FILE: data,
        ARTIFACT_SHA_FILE: f"{sha}  {ARTIFACT_FILE}\n".encode("ascii"),
        APPLY_FILE: render_report(report).encode("utf-8"),
    }
    if args.check:
        bad = [name for name, b in files.items() if not outs[name].exists() or outs[name].read_bytes() != b]
        for name in bad:
            print(f"CHECK: {outs[name]} differs from a fresh reading (or is missing)", file=sys.stderr)
        print(f"{run_dir}: {'ok' if not bad else 'FAILED'}", file=sys.stderr)
        return 1 if bad else 0
    tmps = []
    for name, b in files.items():
        tmp = outs[name].with_name(outs[name].name + ".tmp")
        tmp.write_bytes(b)
        tmps.append((tmp, outs[name]))
    for tmp, out in tmps:
        tmp.replace(out)
    er = report["edit_report"]
    print(f"{run_dir}: {report['n_replies']} replies, {er['blocks']} blocks, "
          f"{er['applied']}/{er['statements']} statements applied, {er['skipped']} skipped, "
          f"{len(er['cut_off_statements'])} cut off; end_of_weights={report['end_of_weights']}")
    print(f"non-zero params: {report['nonzero_total']} / {report['parameters']}")
    print(f"artifact sha256: {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
