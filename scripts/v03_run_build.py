"""Build and archive one v0.3 phase of the weight author's deliverables (spec v0.3 §9, §15).

Usage:
    uv run python scripts/v03_run_build.py PHASE --author-dir DIR --packet-dir DIR --out-dir DIR
                                         [--no-rebuild-check] [--force] [--registry PATH | --no-registry]

PHASE is v03-phase-a or v03-phase-b. See agent_loop_lab.v03.runner for every check. OUT_DIR (for
example weight_program/v0_3/phase-a/) receives artifact.dlw, artifact.sha256, BUILD_RECORD.json and
copies of the required author files; every run, built or rejected, logs OUT_DIR/attempts/NNNN.json.
By default the packet must match experiments/registry_v0_3.yaml when that file exists. On a
rejection the error text is printed to stderr (that text, and nothing else, is what a resubmission
discloses). The runner does not git-commit.
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

from agent_loop_lab.v03.runner import PHASES, RECORD_FILE, BuildError, build_phase  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=PHASES)
    ap.add_argument("--author-dir", type=Path, required=True)
    ap.add_argument("--packet-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--no-rebuild-check", action="store_true")
    ap.add_argument("--force", action="store_true")
    reg = ap.add_mutually_exclusive_group()
    reg.add_argument("--registry", type=Path)
    reg.add_argument("--no-registry", action="store_true")
    args = ap.parse_args(argv)
    try:
        record = build_phase(
            args.phase, args.author_dir, args.packet_dir, args.out_dir,
            rebuild_check=not args.no_rebuild_check, force=args.force,
            registry_path=None if args.no_registry else (args.registry or "auto"),
        )
    except (BuildError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{args.phase} archived in {args.out_dir}")
    print(f"artifact sha256: {record['artifact_sha256']}")
    print(f"packet sha256:   {record['packet_sha256']}")
    print(f"non-zero params: {record['sparsity']['overall']['nonzero']} / {record['sparsity']['overall']['slots']}")
    print(f"rebuild check:   {record['checks']['rebuild_check']}")
    print(f"registered:      {'yes' if record['checks']['packet_registered'] else 'no registry checked'}")
    print(f"record:          {args.out_dir / RECORD_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
