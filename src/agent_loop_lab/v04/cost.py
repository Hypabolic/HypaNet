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
  ``wsp-b`` ...), so a WSP-B launch that resumes the WSP-A session is charged only its own spend.

**Killed launches.** A launch with no ``result`` event (SIGKILL, a crash, the machine asleep), or with
assistant turns after its last one, is **unresolved**, and its spend is never recovered later: the
cumulative total a resumed launch reports restarts from the cost saved at the last normal exit, so it
covers only that launch's own turns (checked with throwaway sessions: a resume after a killed
seven-turn launch reported only its own single turn). Per-message ``usage`` in the transcript does not
recover it either: ``output_tokens`` there is the stream's opening count (v0.3 Phase A: 113 of 258,051),
so the output and thinking tokens, most of the cost, are missing. An unresolved launch is charged, in
order of preference:

1. an operator figure from ``LOGDIR/COST_OVERRIDES.json`` (``{"<name>": {"usd": x, "reason": "..."}}``,
   for example from the console's usage page);
2. its ``--max-budget-usd`` ceiling, from ``LOGDIR/<name>.launch.json``, which the launcher writes
   **before** claude starts (``max(ceiling, the delta its result events give)``; the CLI checks the budget
   between turns, so the last turn can overshoot it a little);
3. otherwise a lower bound only: its result delta plus the priced input side of its assistant messages
   (:data:`PRICES_PER_MTOK`, which reproduces ``modelUsage.costUSD`` exactly on v0.3's and the kit's
   transcripts). :func:`check` refuses while any launch is charged this way unless the operator accepts
   it (``--accept-unresolved``): the figure is then charged, never US$0.

Unresolved launches stay listed in every summary, however they are charged.

On v0.3's own logs this gives Phase A US$42.73 (14.89 + 27.85) and Phase B US$21.89, the corrected
figures of plan revision 11 (v0.3's script reported US$122.24 in total).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

CAPS_USD = {"wsp-a": 500.0, "wsp-b": 250.0, "wsp-f": 250.0}
SUFFIX = ".stream.jsonl"
LAUNCH_RECORD = ".launch.json"
OVERRIDES = "COST_OVERRIDES.json"
EXIT_CAP, EXIT_UNRESOLVED = 3, 4
# US$ per million tokens: input, output, cache write 5 min, cache write 1 h, cache read (first-party list
# prices; claude-api skill, cached 2026-10-06; checked against modelUsage.costUSD on v0.3's Fable logs and the
# kit's Haiku logs). An unknown model is priced as the most expensive one.
PRICES_PER_MTOK = {
    "claude-fable-5-1": (10.0, 50.0, 12.5, 20.0, 0.25),
    "claude-opus-5-5": (4.0, 20.0, 5.0, 8.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 2.5, 4.0, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.20, 0.01),
}
_MAX_PRICE = tuple(max(p[i] for p in PRICES_PER_MTOK.values()) for i in range(5))


def phase_of(name: str, phases: Iterable[str] = CAPS_USD) -> str | None:
    """The phase a launch name (``wsp-a.c2``) belongs to, or None."""
    return next((p for p in phases if name == p or name.startswith(p + ".")), None)


def _events(log: Path) -> list[dict]:
    out = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            out.append(ev)
    return out


def _message_input_usd(usage: dict, model: str | None) -> float:
    """The input side of one assistant message (uncached, cache writes, cache reads); output is left out."""
    p = PRICES_PER_MTOK.get(model or "", _MAX_PRICE)
    n = lambda k: int(usage.get(k) or 0)  # noqa: E731
    cc = usage.get("cache_creation") if isinstance(usage.get("cache_creation"), dict) else None
    w5 = int(cc.get("ephemeral_5m_input_tokens") or 0) if cc else 0
    w1 = int(cc.get("ephemeral_1h_input_tokens") or 0) if cc else n("cache_creation_input_tokens")
    return (n("input_tokens") * p[0] + w5 * p[2] + w1 * p[3] + n("cache_read_input_tokens") * p[4]) / 1e6


def launch_totals(log: Path) -> tuple[dict[str, float], list[str], int]:
    """``({session id: last total_cost_usd}, models, result events)`` of one transcript."""
    rec = scan_launch(log)
    return rec["totals"], rec["models"], rec["result_events"]


def scan_launch(log: Path) -> dict:
    """One transcript: its session totals, models, result events, and whether assistant turns follow the last
    result event (``tail_messages``) with the lower-bound price of those turns (``tail_input_usd``)."""
    totals: dict[str, float] = {}
    models: set[str] = set()
    n = 0
    tail: dict[str, tuple[str | None, dict]] = {}  # message id -> (model, usage) since the last result event
    for ev in _events(log):
        if ev.get("type") == "result":
            n += 1
            sid, total = ev.get("session_id"), ev.get("total_cost_usd")
            if isinstance(sid, str) and isinstance(total, (int, float)):
                totals[sid] = float(total)
            models.update(k for k in (ev.get("modelUsage") or {}) if isinstance(k, str))
            tail = {}
        elif ev.get("type") == "assistant" and isinstance(ev.get("message"), dict):
            m = ev["message"]
            if isinstance(m.get("usage"), dict):
                tail[str(m.get("id") or len(tail))] = (m.get("model"), m["usage"])  # one message spans several events
            if isinstance(m.get("model"), str):
                models.add(m["model"])
    return {"totals": totals, "models": sorted(models), "result_events": n, "tail_messages": len(tail),
            "tail_input_usd": sum(_message_input_usd(u, mod) for mod, u in tail.values())}


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def account(logs: Iterable[Path], phases: Iterable[str] = CAPS_USD) -> dict:
    """Per-launch delta costs and per-phase totals, unresolved launches charged as the module docstring says."""
    phases = tuple(phases)
    launches: dict[str, dict] = {}
    by_session: dict[str, list[tuple[float, str]]] = {}
    overrides: dict[str, Any] = {}
    for log in sorted(Path(p) for p in logs):
        name = log.name[: -len(SUFFIX)] if log.name.endswith(SUFFIX) else log.stem
        scan = scan_launch(log)
        ov = _json(log.parent / OVERRIDES)
        overrides.update(ov if isinstance(ov, dict) else {})
        rec = _json(log.parent / (name + LAUNCH_RECORD))
        ceiling = rec.get("max_budget_usd") if isinstance(rec, dict) else None
        launches[name] = {"log": log.name, "phase": phase_of(name, phases), "models": scan["models"],
                          "result_events": scan["result_events"],
                          "session_totals_usd": {s: round(t, 6) for s, t in scan["totals"].items()}, "cost_usd": 0.0,
                          "resolved": bool(scan["totals"]) and not scan["tail_messages"],
                          "turns_after_last_result": scan["tail_messages"],
                          "max_budget_usd": float(ceiling) if isinstance(ceiling, (int, float)) else None,
                          "charged_as": "result delta", "_tail_usd": scan["tail_input_usd"]}
        for sid, total in scan["totals"].items():
            by_session.setdefault(sid, []).append((total, name))
    for sid, runs in by_session.items():
        prev = 0.0
        for total, name in sorted(runs):
            launches[name]["cost_usd"] += total - prev
            prev = total
    for name, rec in launches.items():
        tail = rec.pop("_tail_usd")
        if rec["resolved"]:
            continue
        ov = overrides.get(name)
        if isinstance(ov, dict) and isinstance(ov.get("usd"), (int, float)):
            rec["cost_usd"], rec["charged_as"] = float(ov["usd"]), f"operator override ({ov.get('reason', 'no reason given')})"
        elif rec["max_budget_usd"] is not None:
            rec["cost_usd"], rec["charged_as"] = max(rec["cost_usd"], rec["max_budget_usd"]), "max_budget_usd ceiling"
        else:
            rec["cost_usd"] += tail
            rec["charged_as"] = "lower bound (input side only; needs the operator's acceptance)"
    per_phase = {p: {"total_usd": 0.0, "cap_usd": CAPS_USD.get(p), "launches": [], "unresolved": [],
                     "lower_bound_only": []} for p in phases}
    for name, rec in launches.items():
        rec["cost_usd"] = round(rec["cost_usd"], 6)
        p = rec["phase"]
        if p is None:
            continue
        per_phase[p]["launches"].append(name)
        per_phase[p]["total_usd"] += rec["cost_usd"]
        if not rec["resolved"]:
            per_phase[p]["unresolved"].append(name)
            if rec["charged_as"].startswith("lower bound"):
                per_phase[p]["lower_bound_only"].append(name)
    for p, rec in per_phase.items():
        cap = rec["cap_usd"]
        rec["total_usd"] = round(rec["total_usd"], 6)
        if cap is not None:
            rec["remaining_usd"] = round(max(0.0, cap - rec["total_usd"]), 6)
            rec["reached"] = rec["total_usd"] >= cap
            rec["overshoot_usd"] = round(max(0.0, rec["total_usd"] - cap), 6)
    return {"launches": launches, "phases": per_phase, "sessions": sorted(by_session),
            "total_usd": round(sum(r["cost_usd"] for r in launches.values()), 6),
            "method": "per-session-id delta of the cumulative total_cost_usd; unresolved launches at an operator "
                      "figure, their --max-budget-usd ceiling, or a lower bound"}


def summary(logs_dir: Path, phases: Iterable[str] = CAPS_USD) -> dict:
    return account(sorted(Path(logs_dir).glob("*" + SUFFIX)) if Path(logs_dir).is_dir() else [], phases)


def check(s: dict, phase: str, *, accept_unresolved: bool = False) -> tuple[int, str]:
    """``(exit code, message)``: 0 to allow a launch of ``phase``, EXIT_CAP at the cap, EXIT_UNRESOLVED while a
    launch is charged only a lower bound (no launch record, no override), unless ``accept_unresolved``."""
    rec = s["phases"][phase]
    if rec.get("reached"):
        return EXIT_CAP, f"cost cap reached for {phase}: US${rec['total_usd']} >= US${rec['cap_usd']}"
    if rec["lower_bound_only"] and not accept_unresolved:
        return EXIT_UNRESOLVED, (f"{phase}: launch(es) {rec['lower_bound_only']} ended without a result event and have "
                                 "no launch record, so only a lower bound of their cost is known (a resumed launch "
                                 f"never reports it); add a figure to {OVERRIDES} or accept the lower bound explicitly")
    note = f" (unresolved, charged as recorded: {rec['unresolved']})" if rec["unresolved"] else ""
    return 0, f"{phase}: US${rec['total_usd']} spent, US${rec['remaining_usd']} of US${rec['cap_usd']} left{note}"
