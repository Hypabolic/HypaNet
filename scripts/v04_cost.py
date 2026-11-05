"""Weight-author API cost over the v0.4 launch transcripts, by session-id delta (plan revision 11).

Usage:
    uv run python scripts/v04_cost.py [--logs DIR] [--phase wsp-a|wsp-b|wsp-f] [--check [--accept-unresolved]]
    uv run python scripts/v04_cost.py --phase wsp-a --remaining        # the number for --max-budget-usd

Reads every ``*.stream.jsonl`` in DIR (default experiments/v0_4/logs/). ``total_cost_usd`` is cumulative
per session id across resumed launches, so each launch is charged its last total minus the previous
launch's total for the same session id (agent_loop_lab.v04.cost); v0.3's script summed the totals and
double-counted. A launch killed before its result event is never covered by a later resume: it is charged
the operator's figure in DIR/COST_OVERRIDES.json, else the --max-budget-usd ceiling from DIR/NAME.launch.json
(written by the launcher before claude starts), else only a lower bound. Prints JSON. With --check and
--phase: exit 3 once that phase's cap is reached (WSP-A US$500, WSP-B US$250, WSP-F US$250), exit 4 while a
launch of the phase is charged only a lower bound, unless --accept-unresolved (which charges that lower
bound, never US$0). --remaining prints only the remaining cap in US$ (two decimals, rounded down), or exits
like --check.
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

from agent_loop_lab.v04.cost import CAPS_USD, check, summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, default=REPO_ROOT / "experiments" / "v0_4" / "logs")
    ap.add_argument("--phase", choices=sorted(CAPS_USD))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--remaining", action="store_true")
    ap.add_argument("--accept-unresolved", action="store_true",
                    help="charge a killed launch without a launch record its lower bound and allow the launch (operator decision)")
    args = ap.parse_args(argv)
    s = summary(args.logs)
    if (args.check or args.remaining) and not args.phase:
        print("--check and --remaining need --phase", file=sys.stderr)
        return 2
    if args.check or args.remaining:
        code, msg = check(s, args.phase, accept_unresolved=args.accept_unresolved)
        print(msg, file=sys.stderr)
        if code:
            return code
        if args.remaining:
            print(f"{math.floor(s['phases'][args.phase]['remaining_usd'] * 100) / 100:.2f}")
            return 0
    print(json.dumps(s, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
