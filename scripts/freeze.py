"""Compute, write or check the M0 freeze registry (experiments/registry.yaml).

Usage:
    uv run python scripts/freeze.py [--root DIR]            # print the current freeze record
    uv run python scripts/freeze.py [--root DIR] --write [--force]
    uv run python scripts/freeze.py [--root DIR] --check

--root defaults to the repository root. The record holds the sha256 of every
frozen file plus the hash of the authoring packet, which is rebuilt into a
temporary directory each time (nothing is written under --root except the
registry itself on --write). --write refuses to replace a registry whose
content differs unless --force is given. --check exits non-zero on any
mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover - running without the package installed
    sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_loop_lab.registry import RegistryError, check_registry, compute_freeze, write_registry  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write experiments/registry.yaml")
    mode.add_argument("--check", action="store_true", help="check frozen files and the rebuilt packet against the registry")
    ap.add_argument("--force", action="store_true", help="with --write, replace a differing registry")
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
            freeze = compute_freeze(args.root)
            print(f"registry written to {path}")
            print(f"packet sha256: {freeze['packet_sha256']}")
            print(f"freeze sha256: {freeze['freeze_sha256']}")
        else:
            print(json.dumps(compute_freeze(args.root), indent=2, sort_keys=True))
    except (RegistryError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
