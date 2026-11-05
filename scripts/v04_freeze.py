"""Compute, write or check the v0.4 freeze registry (experiments/registry_v0_4.yaml; plan v0.4 G1).

Usage:
    uv run python scripts/v04_freeze.py [--root DIR]                  # print the current freeze record
    uv run python scripts/v04_freeze.py [--root DIR] --list           # print the frozen file list only
    uv run python scripts/v04_freeze.py [--root DIR] --write [--force]
    uv run python scripts/v04_freeze.py [--root DIR] --check

The record (``agent_loop_lab.v04.registry``) holds the sha256 of every frozen file (the required
files plus every match of the globs listed there), the frozen interface's ``spec_sha256``, the model
id, revision and ``FILE_SHA256``, and the packet sha256 (``scripts/v04_build_packet.py`` rebuilt in a
temporary directory). --write refuses when the registry exists unless --force is given, and refuses
when a required file or the packet builder is missing or a glob matches nothing; --check exits
non-zero and names every mismatch.
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

from agent_loop_lab.v04.registry import (  # noqa: E402
    RegistryError,
    check_registry,
    compute_freeze,
    frozen_files,
    problems_text,
    write_registry,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.check:
        problems = check_registry(args.root)
        if problems:
            print(problems_text(problems), file=sys.stderr)
        print("registry OK" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    if args.list:
        files, problems = frozen_files(args.root)
        print("\n".join(files))
        for p in problems:
            print(f"PROBLEM: {p}", file=sys.stderr)
        return 1 if problems else 0
    try:
        if args.write:
            path, rec = write_registry(args.root, overwrite=args.force)
            print(f"registry written to {path}")
            print(f"files: {len(rec['files'])}")
            print(f"interface spec sha256: {rec['interface_spec_sha256']}")
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
