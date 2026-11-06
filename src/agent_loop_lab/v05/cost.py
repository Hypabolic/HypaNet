"""Author API cost for v0.5: one cap of US$500 summed over every launch of every run (r1, r2, r3).

Each turn is one ``claude -p`` launch with its own transcript ``logs/<run>.t<NNN>.stream.jsonl`` (failed
attempts: ``<run>.t<NNN>.fail<K>.stream.jsonl``). ``total_cost_usd`` in a ``result`` event is cumulative per
session id across resumed launches, so the launches are charged by session-id delta, exactly as v0.4 does
(:func:`agent_loop_lab.v04.cost.account`): a launch killed before its result event is charged an operator figure
from ``logs/COST_OVERRIDES.json``, else its ``--max-budget-usd`` ceiling from ``logs/<name>.launch.json`` (the
runner writes it before claude starts), else a lower bound that needs the operator's acceptance. The plain sum
of ``total_cost_usd`` (v0.3's method, which double-counts resumed sessions) is reported alongside as
``naive_sum_usd`` and is never used for the cap.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_loop_lab.v04.cost import SUFFIX, account
from agent_loop_lab.v05.session import RUNS

CAP_USD = 500.0
EXIT_CAP, EXIT_UNRESOLVED = 3, 4


def _naive(log: Path) -> float:
    total = 0.0
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict) and ev.get("type") == "result" and isinstance(ev.get("total_cost_usd"), (int, float)):
            total += float(ev["total_cost_usd"])
    return total


def summary(logs_dir: Path, cap_usd: float = CAP_USD) -> dict:
    logs = sorted(Path(logs_dir).glob("*" + SUFFIX)) if Path(logs_dir).is_dir() else []
    s = account(logs, RUNS)
    total = round(sum(r["total_usd"] for r in s["phases"].values()), 6)
    unresolved = [n for r in s["phases"].values() for n in r["unresolved"]]
    lower = [n for r in s["phases"].values() for n in r["lower_bound_only"]]
    s["runs"] = s.pop("phases")
    s["cap"] = {"cap_usd": cap_usd, "total_usd": total, "remaining_usd": round(max(0.0, cap_usd - total), 6),
                "reached": total >= cap_usd, "overshoot_usd": round(max(0.0, total - cap_usd), 6),
                "unresolved": unresolved, "lower_bound_only": lower}
    s["naive_sum_usd"] = round(sum(_naive(p) for p in logs), 6)
    return s


def check(s: dict, *, accept_unresolved: bool = False) -> tuple[int, str]:
    """``(exit code, message)``: 0 allows another launch, EXIT_CAP once the cap is reached, EXIT_UNRESOLVED while
    a launch is charged only a lower bound (unless ``accept_unresolved``)."""
    c = s["cap"]
    if c["reached"]:
        return EXIT_CAP, f"v0.5 cost cap reached: US${c['total_usd']} >= US${c['cap_usd']}"
    if c["lower_bound_only"] and not accept_unresolved:
        return EXIT_UNRESOLVED, (f"launch(es) {c['lower_bound_only']} ended without a result event and have no launch "
                                 "record; add a figure to COST_OVERRIDES.json or accept the lower bound explicitly")
    return 0, f"v0.5: US${c['total_usd']} spent, US${c['remaining_usd']} of US${c['cap_usd']} left"
