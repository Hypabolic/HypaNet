"""Build and verify the weight-author packet (spec v0.2 §8).

Usage:
    uv run python scripts/build_authoring_packet.py [--spec PATH] [--model PATH] [--out DIR] [--verify-only]

Defaults are relative to the repository root: the v0.2 spec, src/agent_loop_lab/model.py
and authoring_packet/. Prints the packet sha256 and exits non-zero on any problem.
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

from agent_loop_lab.packet import PacketError, build_packet, verify_packet  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, default=REPO_ROOT / "agent_doom_loop_detector_spec_v0_2.md")
    ap.add_argument("--model", type=Path, default=REPO_ROOT / "src" / "agent_loop_lab" / "model.py")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "authoring_packet")
    ap.add_argument("--verify-only", action="store_true", help="verify an existing packet without rebuilding")
    args = ap.parse_args(argv)

    if args.verify_only:
        problems = verify_packet(args.out)
        for p in problems:
            print(f"PROBLEM: {p}", file=sys.stderr)
        print("packet OK" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0

    try:
        manifest = build_packet(args.spec, args.model, args.out)
    except (PacketError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"packet written to {args.out}")
    for name, sha in manifest["files"].items():
        print(f"  {sha}  {name}")
    print(f"source spec sha256: {manifest['source_spec_sha256']}")
    print(f"packet sha256:      {manifest['packet_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
