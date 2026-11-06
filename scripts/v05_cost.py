"""Author API cost over the v0.5 turn transcripts: US$500 cap summed over r1, r2 and r3.

Usage:
    uv run python scripts/v05_cost.py [--logs DIR] [--check [--accept-unresolved]]
    uv run python scripts/v05_cost.py --remaining          # remaining cap in US$, two decimals, rounded down

Reads every ``*.stream.jsonl`` in DIR (default experiments/v0_5/logs/) and charges each launch (one turn, or a
failed attempt of one) by session-id delta of the cumulative ``total_cost_usd`` (agent_loop_lab.v05.cost, which
reuses v0.4's accounting, including the charging of killed launches). Prints JSON. With --check: exit 3 once
the cap is reached, exit 4 while a launch is charged only a lower bound (unless --accept-unresolved).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_loop_lab.v05.cost import check, summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, default=REPO_ROOT / "experiments" / "v0_5" / "logs")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--remaining", action="store_true")
    ap.add_argument("--accept-unresolved", action="store_true",
                    help="charge a killed launch without a launch record its lower bound (operator decision)")
    args = ap.parse_args(argv)
    s = summary(args.logs)
    if args.check or args.remaining:
        code, msg = check(s, accept_unresolved=args.accept_unresolved)
        print(msg, file=sys.stderr)
        if code:
            return code
        if args.remaining:
            print(f"{math.floor(s['cap']['remaining_usd'] * 100) / 100:.2f}")
            return 0
    print(json.dumps(s, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
