"""Build and verify the v0.4 WSP authoring packet (spec v0.4 §13; src/agent_loop_lab/v04/packet.py).

Usage:
    uv run python scripts/v04_build_packet.py --out DIR [--source-dir DIR] [--spec PATH] [--v02-spec PATH]
                                              [--qwen-snapshot DIR] [--interface PATH]
    uv run python scripts/v04_build_packet.py --out DIR --verify-only

Defaults: the v0.4 and v0.2 specs at the repository root, src/agent_loop_lab/v03/ as the source of
renderer.py and tokenizer.py, envs/qwen/hypaqwen/interface.py (loaded by file path; standard library
only) and the pinned Qwen snapshot in the local Hugging Face cache (HF_HUB_CACHE / HF_HOME, else
~/.cache/huggingface/hub), whose files are checked against envs/qwen/hypaqwen/model.py FILE_SHA256.
Nothing is downloaded. Fails (exit 1) with a clear message if a source is missing or a check fails.
Prints the packet sha256 (sha256 of PACKET_MANIFEST.json).
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

from agent_loop_lab.v04.packet import (  # noqa: E402
    DEFAULT_INTERFACE,
    PacketError,
    build_packet,
    packet_sha256,
    verify_packet,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-dir", type=Path, default=REPO_ROOT / "src" / "agent_loop_lab" / "v03")
    ap.add_argument("--spec", type=Path, default=REPO_ROOT / "agent_doom_loop_detector_spec_v0_4.md")
    ap.add_argument("--v02-spec", type=Path, default=REPO_ROOT / "agent_doom_loop_detector_spec_v0_2.md")
    ap.add_argument("--qwen-snapshot", type=Path, default=None, help="the pinned snapshot dir (default: HF cache)")
    ap.add_argument("--interface", type=Path, default=DEFAULT_INTERFACE)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)
    if args.verify_only:
        problems = verify_packet(args.out, interface_path=args.interface)
        for p in problems:
            print(f"PROBLEM: {p}", file=sys.stderr)
        if problems:
            return 1
        print("packet OK")
        print(f"packet sha256: {packet_sha256(args.out)}")  # the last line, as for a build
        return 0
    try:
        manifest = build_packet(args.spec, args.v02_spec, args.source_dir, args.out,
                                qwen_snapshot=args.qwen_snapshot, interface_path=args.interface)
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
