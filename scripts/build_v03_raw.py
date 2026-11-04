"""Build the v0.3 raw-operation datasets (spec v0.3 §7.1-7.2).

Usage:
    uv run python scripts/build_v03_raw.py [--out datasets/v0_3] [--v02 datasets/v0_2] [--tmp DIR] [--keep-tmp]
    uv run python scripts/build_v03_raw.py --stats-only     # amend an existing build's manifest only

Re-runs the v0.2 standard, counterfactual and OOD builds unchanged (same
seeds and sizes) with raw capture into a temporary directory, checks every
regenerated canonical events/oracle file is byte-identical to the hashes in
the committed v0.2 MANIFEST.json and OOD_MANIFEST.json (fails with exit 1
otherwise), and writes OUT/{dev,val,test,counterfactual,ood}.raw.jsonl.gz plus
OUT/MANIFEST_v0_3.json. It then renders and tokenises dev and val (only) and
adds token statistics and the renderer/tokeniser source hashes to the manifest.
Never writes into the v0.2 directory. Computes no labels, label statistics,
predictions or metrics, and never renders test, counterfactual or OOD.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_loop_lab.datasets.build import BuildError  # noqa: E402
from agent_loop_lab.v03.build_raw import build_raw  # noqa: E402
from agent_loop_lab.v03.token_stats import amend_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "datasets" / "v0_3")
    ap.add_argument("--v02", type=Path, default=REPO_ROOT / "datasets" / "v0_2")
    ap.add_argument("--tmp", type=Path, default=None, help="regeneration directory (default: a fresh temp dir)")
    ap.add_argument("--keep-tmp", action="store_true", help="keep the default temp regeneration directory")
    ap.add_argument("--stats-only", action="store_true",
                    help="skip the rebuild; only add token statistics and source hashes to OUT's manifest")
    args = ap.parse_args(argv)
    t0 = time.monotonic()

    def say(msg: str) -> None:
        print(f"[{time.monotonic() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)

    if args.stats_only:
        m = amend_manifest(args.out, progress=say)
        print(json.dumps(m["token_statistics"], indent=1))
        return 0
    try:
        m = build_raw(args.out, v02_dir=args.v02, tmp_dir=args.tmp, keep_tmp=args.keep_tmp, progress=say)
    except BuildError as exc:
        print(f"v0.3 raw build FAILED: {exc}", file=sys.stderr)
        return 1
    if any(m["unclaimed_finished_sessions"].values()):
        say(f"WARNING: sessions finished but not written: {m['unclaimed_finished_sessions']}")
    say(f"timings: {m['_timings']}")
    for name, meta in m["files"].items():
        say(f"{name}: {meta['lines']} sessions, {meta['bytes']:,} bytes, sha256 {meta['sha256'][:16]}")
    say(f"acceptance: all canonical files byte-identical = {m['acceptance']['all_byte_identical']}")
    print(json.dumps({k: v for k, v in m.items() if k in ("sessions", "raw_ops", "_timings")}, indent=1))
    t1 = time.monotonic()
    m = amend_manifest(args.out, progress=say)
    say(f"token statistics (dev, val) added in {time.monotonic() - t1:.1f}s")
    print(json.dumps(m["token_statistics"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
