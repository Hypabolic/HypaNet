"""Build and verify the v0.3 authoring packet (spec v0.3 §8).

Usage:
    uv run python scripts/v03_build_packet.py --out DIR [--source-dir DIR] [--spec PATH] [--v02-spec PATH]
    uv run python scripts/v03_build_packet.py --out DIR --verify-only

Defaults: the v0.3 and v0.2 specs at the repository root and src/agent_loop_lab/v03/ as the source of
model.py, tokenizer.py and renderer.py (rawop.py is not shipped: spec §8). Fails (exit 1) with a clear
message if any of those sources is missing. Prints the packet sha256 (sha256 of PACKET_MANIFEST.json).
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

from agent_loop_lab.v03.packet import PacketError, build_packet, packet_sha256, verify_packet  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-dir", type=Path, default=REPO_ROOT / "src" / "agent_loop_lab" / "v03")
    ap.add_argument("--spec", type=Path, default=REPO_ROOT / "agent_doom_loop_detector_spec_v0_3.md")
    ap.add_argument("--v02-spec", type=Path, default=REPO_ROOT / "agent_doom_loop_detector_spec_v0_2.md")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)
    if args.verify_only:
        problems = verify_packet(args.out)
        for p in problems:
            print(f"PROBLEM: {p}", file=sys.stderr)
        if problems:
            return 1
        print(f"packet OK, sha256 {packet_sha256(args.out)}")
        return 0
    try:
        manifest = build_packet(args.spec, args.v02_spec, args.source_dir, args.out)
    except (PacketError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"packet written to {args.out}")
    for name, sha in manifest["files"].items():
        print(f"  {sha}  {name}")
    print(f"packet sha256: {manifest['packet_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
