"""Compute, write or check the v0.3 freeze registry (experiments/registry_v0_3.yaml; spec v0.3 §14 V0c).

Usage:
    uv run python scripts/v03_freeze.py [--root DIR]            # print the current freeze record
    uv run python scripts/v03_freeze.py [--root DIR] --write [--force]
    uv run python scripts/v03_freeze.py [--root DIR] --check

The record holds the sha256 of every frozen file (agent_loop_lab.v03.registry.FROZEN_FILES) and the
packet sha256 (sha256 of PACKET_MANIFEST.json of a packet rebuilt in a temporary directory).
--write refuses to replace a differing registry unless --force is given; --check exits non-zero on
any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_loop_lab.v03.registry import RegistryError, check_registry, compute_freeze, write_registry  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.check:
        problems = check_registry(args.root)
        for p in problems:
            print(f"MISMATCH: {p}", file=sys.stderr)
        print("registry OK" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    try:
        if args.write:
            path = write_registry(args.root, overwrite=args.force)
            rec = compute_freeze(args.root)
            print(f"registry written to {path}")
            print(f"packet sha256: {rec['packet_sha256']}")
            print(f"freeze sha256: {rec['freeze_sha256']}")
        else:
            print(json.dumps(compute_freeze(args.root), indent=2, sort_keys=True))
    except RegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
