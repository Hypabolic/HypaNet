"""Build and archive one phase of the weight author's deliverables (spec v0.2 §9, §15).

Usage:
    uv run python scripts/run_build.py PHASE --author-dir DIR --packet-dir DIR --out-dir DIR
                                       [--no-rebuild-check] [--force] [--registry PATH | --no-registry]

PHASE is one of phase-a-strict, phase-a-selfcheck, phase-b-revision. The packet is
verified, the author's build_weights.py and reference_policy.py are checked and run
under the anti-training guard, the build is repeated in a fresh interpreter, and
OUT_DIR receives artifact.dlw, artifact.sha256, BUILD_RECORD.json and copies of the
author files. An existing OUT_DIR/artifact.dlw is never overwritten without --force.
Every run, built or rejected, also logs OUT_DIR/attempts/NNNN.json.

The packet's packet_sha256 must equal the freeze registry's: by default
experiments/registry.yaml in this repository when it exists, or the file given with
--registry (which must exist); --no-registry skips the check.
The runner does not git-commit; commit the archive afterwards.

Prints the artefact sha256 and exits non-zero on any rejection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover - running without the package installed
    sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_loop_lab.runner import PHASES, RECORD_FILE, BuildError, build_phase  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=PHASES)
    ap.add_argument("--author-dir", type=Path, required=True, help="the weight author's output directory for this phase")
    ap.add_argument("--packet-dir", type=Path, default=REPO_ROOT / "authoring_packet", help="the authoring packet")
    ap.add_argument("--out-dir", type=Path, required=True, help="archive directory (e.g. weight_program/<phase>/)")
    ap.add_argument("--no-rebuild-check", action="store_true", help="skip the fresh-interpreter rebuild comparison")
    ap.add_argument("--force", action="store_true", help="overwrite an existing archived artefact")
    reg = ap.add_mutually_exclusive_group()
    reg.add_argument("--registry", type=Path, help="freeze registry to check the packet against (must exist)")
    reg.add_argument("--no-registry", action="store_true", help="do not check the packet against a freeze registry")
    args = ap.parse_args(argv)

    try:
        record = build_phase(
            args.phase,
            args.author_dir,
            args.packet_dir,
            args.out_dir,
            rebuild_check=not args.no_rebuild_check,
            force=args.force,
            registry_path=None if args.no_registry else (args.registry or "auto"),
        )
    except (BuildError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{args.phase} archived in {args.out_dir}")
    print(f"artifact sha256: {record['artifact_sha256']}")
    print(f"packet sha256:   {record['packet_sha256']}")
    print(f"rebuild check:   {record['checks']['rebuild_check']}")
    print(f"registered:      {'yes' if record['checks']['packet_registered'] else 'no registry checked'}")
    print(f"record:          {args.out_dir / RECORD_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
