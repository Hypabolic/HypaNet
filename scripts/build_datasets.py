"""Build the v0.2 standard and counterfactual datasets (M1_CONTRACTS §5).

Usage:
    uv run python scripts/build_datasets.py [--out DIR] [--sessions N] [--pairs P]
                                            [--master-seed S] [--reports DIR | --no-reports]

Defaults: --out datasets/v0_2, --sessions 20000, --pairs 300 (per kind),
--master-seed 20261104, --reports artifacts/reports. Writes the split and
counterfactual files under OUT/standard/, OUT/MANIFEST.json and OUT/splits.json,
then reads the splits back and writes m1_dataset_stats.json and
m1_collisions_dev.json to the reports directory. Rebuilding with the same
arguments and code gives byte-identical files.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover - running without the package installed
    sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_loop_lab.datasets.build import (  # noqa: E402
    DEFAULT_MASTER_SEED,
    PAIRS_PER_KIND,
    STANDARD_SESSIONS,
    BuildError,
    build,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "datasets" / "v0_2")
    ap.add_argument("--sessions", type=int, default=STANDARD_SESSIONS)
    ap.add_argument("--pairs", type=int, default=PAIRS_PER_KIND, help="counterfactual pairs per kind")
    ap.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--reports", type=Path, default=REPO_ROOT / "artifacts" / "reports")
    grp.add_argument("--no-reports", action="store_true")
    args = ap.parse_args(argv)
    if args.sessions < 0 or args.pairs < 0:
        ap.error("--sessions and --pairs must be >= 0")

    t0 = time.monotonic()

    def say(msg: str) -> None:
        print(f"[{time.monotonic() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)

    try:
        res = build(
            args.out,
            master_seed=args.master_seed,
            sessions=args.sessions,
            pairs_per_kind=args.pairs,
            reports_dir=None if args.no_reports else args.reports,
            progress=say,
        )
    except BuildError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    std = res.manifest["standard"]["splits"]
    for split, c in std.items():
        say(f"{split}: {c['sessions']} sessions, {c['windows']} windows")
    say(f"done -> {res.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
