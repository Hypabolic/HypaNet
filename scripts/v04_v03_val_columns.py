"""Recompute the v0.3 columns on val in the frozen v0.3 environment (``agent_loop_lab.v04.v03_val``).

Usage:
    uv run python scripts/v04_v03_val_columns.py [--out experiments/v0_4/columns/v03.val.npz] [--workers W] [--force]

Val only: no sealed split and no counterfactual file is opened. Every v0.3 column must reproduce
its val block in artifacts/reports/v03_sealed.json exactly, or nothing is written. Pass the file to
scripts/v04_score_validation.py --v03. Harness only.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_loop_lab.v04 import v03_val  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=v03_val.DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=v03_val.S.DEFAULT_WORKERS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.out.exists() and not a.force:
        raise SystemExit(f"{a.out} exists; pass --force to replace it")
    t0 = time.time()
    meta = v03_val.export(a.out, workers=a.workers,
                          progress=lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", file=sys.stderr, flush=True))
    print(f"wrote {a.out}: {len(meta['columns_info'])} v0.3 networks, "
          f"{meta['reproduction']['column_blocks_compared']} blocks equal to v03_sealed.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
