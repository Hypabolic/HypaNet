"""Weight-author API cost for v0.4, by session-id delta (plan "Revisions after M0" 11; docs/V04_WSP_KERNEL.md §7).

Claude Code's ``result`` event carries ``total_cost_usd``, which is **cumulative per session id across
resumed launches** (checked on v0.3's transcripts and on a throwaway resumed ``claude -p`` session).
v0.3's ``scripts/v03_cost.py`` summed it per launch and so double-counted every resumed session. Here:

* a launch is one ``<name>.stream.jsonl`` transcript; its total for a session id is the last
  ``total_cost_usd`` of that session's ``result`` events in the file;
* the launches of one session id are ordered by that total (it never decreases), and each launch is
  charged its total minus the previous launch's total for the same session id (the first launch is
  charged its total);
* a launch belongs to the phase its file name starts with (``wsp-a``, ``wsp-a.c1``, ``wsp-a.r2``,
  ``wsp-b`` ...), so a WSP-B launch that resumes the WSP-A session is charged only its own spend;
* a launch with no ``result`` event (killed, machine asleep) is **unresolved**: its spend is unknown
  until a later launch of the same session reports a larger total, which then carries it. It is
  listed, and :func:`check` refuses while one is unresolved unless the operator accepts it.

On v0.3's own logs this gives Phase A US$42.73 (14.89 + 27.85) and Phase B US$21.89, the corrected
figures of plan revision 11 (v0.3's script reported US$122.24 in total).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

CAPS_USD = {"wsp-a": 500.0, "wsp-b": 250.0, "wsp-f": 250.0}
SUFFIX = ".stream.jsonl"
EXIT_CAP, EXIT_UNRESOLVED = 3, 4


def phase_of(name: str, phases: Iterable[str] = CAPS_USD) -> str | None:
    """The phase a launch name (``wsp-a.c2``) belongs to, or None."""
    return next((p for p in phases if name == p or name.startswith(p + ".")), None)


def launch_totals(log: Path) -> tuple[dict[str, float], list[str], int]:
    """``({session id: last total_cost_usd}, models, result events)`` of one transcript."""
    totals: dict[str, float] = {}
    models: set[str] = set()
    n = 0
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict) or ev.get("type") != "result":
            continue
        n += 1
        sid, total = ev.get("session_id"), ev.get("total_cost_usd")
        if isinstance(sid, str) and isinstance(total, (int, float)):
            totals[sid] = float(total)
        models.update(k for k in (ev.get("modelUsage") or {}) if isinstance(k, str))
    return totals, sorted(models), n


def account(logs: Iterable[Path], phases: Iterable[str] = CAPS_USD) -> dict:
    """Per-launch delta costs and per-phase totals (module docstring)."""
    phases = tuple(phases)
    launches: dict[str, dict] = {}
    by_session: dict[str, list[tuple[float, str]]] = {}
    for log in sorted(Path(p) for p in logs):
        name = log.name[: -len(SUFFIX)] if log.name.endswith(SUFFIX) else log.stem
        totals, models, n = launch_totals(log)
        launches[name] = {"log": log.name, "phase": phase_of(name, phases), "models": models, "result_events": n,
                          "session_totals_usd": {s: round(t, 6) for s, t in totals.items()}, "cost_usd": 0.0,
                          "resolved": bool(totals)}
        for sid, total in totals.items():
            by_session.setdefault(sid, []).append((total, name))
    for sid, runs in by_session.items():
        prev = 0.0
        for total, name in sorted(runs):
            launches[name]["cost_usd"] += total - prev
            prev = total
    per_phase = {p: {"total_usd": 0.0, "cap_usd": CAPS_USD.get(p), "launches": [], "unresolved": []} for p in phases}
    for name, rec in launches.items():
        rec["cost_usd"] = round(rec["cost_usd"], 6)
        p = rec["phase"]
        if p is None:
            continue
        per_phase[p]["launches"].append(name)
        per_phase[p]["total_usd"] += rec["cost_usd"]
        if not rec["resolved"]:
            per_phase[p]["unresolved"].append(name)
    for p, rec in per_phase.items():
        cap = rec["cap_usd"]
        rec["total_usd"] = round(rec["total_usd"], 6)
        if cap is not None:
            rec["remaining_usd"] = round(max(0.0, cap - rec["total_usd"]), 6)
            rec["reached"] = rec["total_usd"] >= cap
            rec["overshoot_usd"] = round(max(0.0, rec["total_usd"] - cap), 6)
    return {"launches": launches, "phases": per_phase, "sessions": sorted(by_session),
            "total_usd": round(sum(r["cost_usd"] for r in launches.values()), 6),
            "method": "per-session-id delta of the cumulative total_cost_usd"}


def summary(logs_dir: Path, phases: Iterable[str] = CAPS_USD) -> dict:
    return account(sorted(Path(logs_dir).glob("*" + SUFFIX)) if Path(logs_dir).is_dir() else [], phases)


def check(s: dict, phase: str, *, accept_unresolved: bool = False) -> tuple[int, str]:
    """``(exit code, message)``: 0 to allow a launch of ``phase``, EXIT_CAP at the cap, EXIT_UNRESOLVED while a
    launch's cost is unknown (unless ``accept_unresolved``)."""
    rec = s["phases"][phase]
    if rec.get("reached"):
        return EXIT_CAP, f"cost cap reached for {phase}: US${rec['total_usd']} >= US${rec['cap_usd']}"
    if rec["unresolved"] and not accept_unresolved:
        return EXIT_UNRESOLVED, (f"{phase}: launch(es) {rec['unresolved']} have no result event, so their cost is "
                                 "unknown until the session is resumed; accept explicitly to launch anyway")
    return 0, f"{phase}: US${rec['total_usd']} spent, US${rec['remaining_usd']} of US${rec['cap_usd']} left"
