"""Sum the weight author's API cost over v0.3 launch transcripts (spec v0.3 E5, §9.2-9.3).

Usage:
    uv run python scripts/v03_cost.py [--logs DIR] [--phase v03-phase-a|v03-phase-b] [--check]

Sums ``total_cost_usd`` over every ``result`` event of every ``*.stream.jsonl`` in DIR (default
experiments/v0_3/logs/), per launch and per phase (a log belongs to the phase its file name starts
with). Prints JSON. With --check and --phase, exits 3 if that phase's sum has reached its cap
(Phase A US$500, Phase B US$250): no new launch is allowed once the cap is reached; an overshoot
within the last launch is reported in the JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CAPS_USD = {"v03-phase-a": 500.0, "v03-phase-b": 250.0}


def launch_cost(log: Path) -> float:
    total = 0.0
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict) and ev.get("type") == "result" and isinstance(ev.get("total_cost_usd"), (int, float)):
            total += float(ev["total_cost_usd"])
    return total


def phase_of(name: str) -> str | None:
    return next((p for p in CAPS_USD if name == p or name.startswith(p + ".")), None)


def summary(logs: Path) -> dict:
    launches = {}
    phases = {p: 0.0 for p in CAPS_USD}
    for log in sorted(logs.glob("*.stream.jsonl")):
        name = log.name[: -len(".stream.jsonl")]
        cost = launch_cost(log)
        launches[name] = round(cost, 6)
        p = phase_of(name)
        if p is not None:
            phases[p] += cost
    return {
        "launches": launches,
        "phases": {
            p: {"total_usd": round(v, 6), "cap_usd": CAPS_USD[p], "reached": v >= CAPS_USD[p],
                "overshoot_usd": round(max(0.0, v - CAPS_USD[p]), 6)}
            for p, v in phases.items()
        },
        "total_usd": round(sum(launch_cost(l) for l in logs.glob("*.stream.jsonl")), 6),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, default=REPO_ROOT / "experiments" / "v0_3" / "logs")
    ap.add_argument("--phase", choices=sorted(CAPS_USD))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    s = summary(args.logs) if args.logs.is_dir() else {"launches": {}, "phases": {p: {"total_usd": 0.0, "cap_usd": c, "reached": False, "overshoot_usd": 0.0} for p, c in CAPS_USD.items()}, "total_usd": 0.0}
    print(json.dumps(s, indent=2, sort_keys=True))
    if args.check:
        if not args.phase:
            print("--check needs --phase", file=sys.stderr)
            return 2
        if s["phases"][args.phase]["reached"]:
            print(f"cost cap reached for {args.phase}: {s['phases'][args.phase]['total_usd']} >= {CAPS_USD[args.phase]}", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
