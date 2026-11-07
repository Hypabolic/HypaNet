"""Author API cost of v0.5 Phase C: a cap of US$250 summed over every Phase C launch (as v0.5 Phase B).

Usage:
    uv run python scripts/v05c_cost.py [--logs DIR] [--check [--accept-unresolved]]
    uv run python scripts/v05c_cost.py --remaining          # remaining cap in US$, two decimals, rounded down

Reads every ``*.stream.jsonl`` in DIR (default experiments/v0_5/phase_c/logs/). Launch names are ``v05c``
(the new session), ``v05c.c<K>`` (technical continuations) and ``v05c.r<N>`` (build-rejection resubmissions);
the continuations and resubmissions resume the same session, so each launch is charged by session-id delta of
the cumulative ``total_cost_usd`` (agent_loop_lab.v04.cost.account, the accounting v0.4 and v0.5 r1 use; v0.3's
plain per-launch sum double-counts resumed sessions). A launch killed before its result event is charged an
operator figure from DIR/COST_OVERRIDES.json, else its ``--max-budget-usd`` ceiling from DIR/<name>.launch.json
(scripts/v05c_launch.sh writes it before claude starts), else a lower bound that needs --accept-unresolved.
Prints JSON. With --check: exit 3 once the cap is reached, exit 4 while a launch is charged only a lower bound.
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

from agent_loop_lab.v04.cost import EXIT_CAP, EXIT_UNRESOLVED, OVERRIDES, account, SUFFIX  # noqa: E402

PHASE = "v05c"
CAP_USD = 250.0


def summary(logs: Path, cap_usd: float = CAP_USD) -> dict:
    files = sorted(Path(logs).glob("*" + SUFFIX)) if Path(logs).is_dir() else []
    s = account(files, (PHASE,))
    rec = s["phases"][PHASE]
    rec["cap_usd"] = cap_usd
    rec["remaining_usd"] = round(max(0.0, cap_usd - rec["total_usd"]), 6)
    rec["reached"] = rec["total_usd"] >= cap_usd
    rec["overshoot_usd"] = round(max(0.0, rec["total_usd"] - cap_usd), 6)
    stray = [n for n, r in s["launches"].items() if r["phase"] is None]
    s["not_phase_c"] = stray  # transcripts in DIR whose name is not v05c[.*] (not charged here)
    return s


def check(s: dict, *, accept_unresolved: bool = False) -> tuple[int, str]:
    rec = s["phases"][PHASE]
    if rec["reached"]:
        return EXIT_CAP, f"v0.5 Phase C cost cap reached: US${rec['total_usd']} >= US${rec['cap_usd']}"
    if rec["lower_bound_only"] and not accept_unresolved:
        return EXIT_UNRESOLVED, (f"launch(es) {rec['lower_bound_only']} ended without a result event and have no launch "
                                 f"record; add a figure to {OVERRIDES} or accept the lower bound explicitly")
    note = f" (unresolved, charged as recorded: {rec['unresolved']})" if rec["unresolved"] else ""
    return 0, f"v0.5 Phase C: US${rec['total_usd']} spent, US${rec['remaining_usd']} of US${rec['cap_usd']} left{note}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, default=REPO_ROOT / "experiments" / "v0_5" / "phase_c" / "logs")
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
            print(f"{math.floor(s['phases'][PHASE]['remaining_usd'] * 100) / 100:.2f}")
            return 0
    print(json.dumps(s, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
