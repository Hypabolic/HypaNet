"""Author API cost of the HypaNet Clouds run: a cap of US$500 over every launch of run r1 (SPEC section 7).

    uv run python scripts/cost.py [--logs DIR]                # JSON summary
    uv run python scripts/cost.py [--logs DIR] --check        # exit 3 at the cap, 4 while a launch is unresolved
    uv run python scripts/cost.py [--logs DIR] --remaining    # remaining US$, two decimals, rounded down
    uv run python scripts/cost.py [--logs DIR] --session-total SID   # the session's cumulative total_cost_usd so far
    uv run python scripts/cost.py [--logs DIR] --budget-stops        # launches that ended at --max-budget-usd

Reads every ``*.stream.jsonl`` in DIR (default clouds/runs/r1/logs/). Launch names are ``r1`` (the session) and
``r1.c<K>`` (technical continuations, which resume the same session). The accounting is v0.4/v0.5's
(src/agent_loop_lab/v04/cost.py, re-implemented here so clouds/ stands alone):

* Claude Code's ``result`` event carries ``total_cost_usd``, cumulative per session id across resumed launches;
  each launch is charged its last total minus the previous launch's total for the same session id;
* a launch with no ``result`` event, or with assistant turns after its last one (killed, crashed), is unresolved and
  is charged an operator figure from DIR/COST_OVERRIDES.json (``{"<name>": {"usd": x, "reason": "..."}}``), else
  its ``--max-budget-usd`` ceiling from DIR/<name>.launch.json (launch.sh writes it before claude starts), else only
  a lower bound (its input-side token price), which --check refuses unless --accept-unresolved is given.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

CLOUDS = Path(__file__).resolve().parents[1]
DEFAULT_LOGS = CLOUDS / "runs" / "r1" / "logs"
PHASE = "r1"
CAP_USD = 500.0
SUFFIX = ".stream.jsonl"
EXIT_CAP, EXIT_UNRESOLVED = 3, 4
# US$ per million tokens: input, output, cache write 5 min, cache write 1 h, cache read (as src/agent_loop_lab/v04)
PRICES = {"claude-fable-5-1": (10.0, 50.0, 12.5, 20.0, 0.25)}
MAX_PRICE = PRICES["claude-fable-5-1"]


def events(log: Path):
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            yield ev


def input_usd(usage: dict, model) -> float:
    p = PRICES.get(model or "", MAX_PRICE)
    n = lambda k: int(usage.get(k) or 0)  # noqa: E731
    cc = usage.get("cache_creation") if isinstance(usage.get("cache_creation"), dict) else None
    w5 = int(cc.get("ephemeral_5m_input_tokens") or 0) if cc else 0
    w1 = int(cc.get("ephemeral_1h_input_tokens") or 0) if cc else n("cache_creation_input_tokens")
    return (n("input_tokens") * p[0] + w5 * p[2] + w1 * p[3] + n("cache_read_input_tokens") * p[4]) / 1e6


def scan(log: Path) -> dict:
    totals, models, n, tail, turns = {}, set(), 0, {}, 0
    for ev in events(log):
        if ev.get("type") == "result":
            n += 1
            sid, total = ev.get("session_id"), ev.get("total_cost_usd")
            if isinstance(sid, str) and isinstance(total, (int, float)):
                totals[sid] = float(total)
            if isinstance(ev.get("num_turns"), int):
                turns = max(turns, ev["num_turns"])
            models.update(k for k in (ev.get("modelUsage") or {}) if isinstance(k, str))
            tail = {}
        elif ev.get("type") == "assistant" and isinstance(ev.get("message"), dict):
            m = ev["message"]
            if isinstance(m.get("usage"), dict):
                tail[str(m.get("id") or len(tail))] = (m.get("model"), m["usage"])
            if isinstance(m.get("model"), str):
                models.add(m["model"])
    return {"totals": totals, "models": sorted(models), "result_events": n, "num_turns": turns,
            "tail_messages": len(tail), "tail_input_usd": sum(input_usd(u, mod) for mod, u in tail.values())}


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def summary(logs: Path, cap: float = CAP_USD) -> dict:
    files = sorted(logs.glob("*" + SUFFIX)) if logs.is_dir() else []
    ov = _json(logs / "COST_OVERRIDES.json")
    overrides = ov if isinstance(ov, dict) else {}
    launches, by_session = {}, {}
    for log in files:
        name = log.name[: -len(SUFFIX)]
        s = scan(log)
        rec = _json(logs / f"{name}.launch.json")
        ceiling = rec.get("max_budget_usd") if isinstance(rec, dict) else None
        launches[name] = {"log": log.name, "charged": name == PHASE or name.startswith(PHASE + "."),
                          "models": s["models"], "result_events": s["result_events"], "num_turns": s["num_turns"],
                          "session_totals_usd": {k: round(v, 6) for k, v in s["totals"].items()}, "cost_usd": 0.0,
                          "resolved": bool(s["totals"]) and not s["tail_messages"],
                          "turns_after_last_result": s["tail_messages"],
                          "max_budget_usd": float(ceiling) if isinstance(ceiling, (int, float)) else None,
                          "charged_as": "result delta", "_tail": s["tail_input_usd"]}
        for sid, total in s["totals"].items():
            by_session.setdefault(sid, []).append((total, name))
    for runs in by_session.values():
        prev = 0.0
        for total, name in sorted(runs):
            launches[name]["cost_usd"] += total - prev
            prev = total
    for name, r in launches.items():
        tail = r.pop("_tail")
        if r["resolved"]:
            continue
        o = overrides.get(name)
        if isinstance(o, dict) and isinstance(o.get("usd"), (int, float)):
            r["cost_usd"], r["charged_as"] = float(o["usd"]), f"operator override ({o.get('reason', 'no reason')})"
        elif r["max_budget_usd"] is not None:
            r["cost_usd"], r["charged_as"] = max(r["cost_usd"], r["max_budget_usd"]), "max_budget_usd ceiling"
        else:
            r["cost_usd"] += tail
            r["charged_as"] = "lower bound (input side only; needs the operator's acceptance)"
    total = 0.0
    unresolved, lower = [], []
    for name, r in launches.items():
        r["cost_usd"] = round(r["cost_usd"], 6)
        if not r["charged"]:
            continue
        total += r["cost_usd"]
        if not r["resolved"]:
            unresolved.append(name)
            if r["charged_as"].startswith("lower bound"):
                lower.append(name)
    total = round(total, 6)
    return {"phase": PHASE, "cap_usd": cap, "total_usd": total, "remaining_usd": round(max(0.0, cap - total), 6),
            "reached": total >= cap, "unresolved": unresolved, "lower_bound_only": lower, "launches": launches,
            "sessions": sorted(by_session),
            "method": "per-session-id delta of the cumulative total_cost_usd; unresolved launches at an operator "
                      "figure, their --max-budget-usd ceiling, or a lower bound"}


def session_total(logs: Path, sid: str) -> float:
    """The session's cumulative total_cost_usd so far: the largest last total over its launches (it only grows)."""
    best = 0.0
    for log in sorted(logs.glob("*" + SUFFIX)) if logs.is_dir() else []:
        t = scan(log)["totals"].get(sid)
        if t is not None:
            best = max(best, t)
    return best


def budget_stops(logs: Path) -> list[str]:
    """Launches whose last result event says Claude Code stopped at --max-budget-usd."""
    out = []
    for log in sorted(logs.glob("*" + SUFFIX)) if logs.is_dir() else []:
        last = None
        for ev in events(log):
            if ev.get("type") == "result":
                last = ev
        if last is not None and last.get("subtype") == "error_max_budget_usd":
            out.append(log.name[: -len(SUFFIX)])
    return out


def check(s: dict, accept_unresolved: bool = False) -> tuple[int, str]:
    if s["reached"]:
        return EXIT_CAP, f"HypaNet Clouds cost cap reached: US${s['total_usd']} >= US${s['cap_usd']}"
    if s["lower_bound_only"] and not accept_unresolved:
        return EXIT_UNRESOLVED, (f"launch(es) {s['lower_bound_only']} ended without a result event and have no launch "
                                 "record; add a figure to COST_OVERRIDES.json or pass --accept-unresolved")
    return 0, f"HypaNet Clouds r1: US${s['total_usd']} spent, US${s['remaining_usd']} of US${s['cap_usd']} left"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--remaining", action="store_true")
    g.add_argument("--session-total", metavar="SID",
                   help="print the last cumulative total_cost_usd of session SID over every launch (0.000000 if none)")
    g.add_argument("--budget-stops", action="store_true",
                   help="print the launches whose last result has subtype error_max_budget_usd (exit 0 either way)")
    ap.add_argument("--accept-unresolved", action="store_true")
    a = ap.parse_args(argv)
    if a.session_total is not None:
        print(f"{session_total(a.logs, a.session_total):.6f}")
        return 0
    if a.budget_stops:
        for name in budget_stops(a.logs):
            print(name)
        return 0
    s = summary(a.logs)
    if a.check or a.remaining:
        code, msg = check(s, a.accept_unresolved)
        print(msg, file=sys.stderr)
        if code:
            return code
        if a.remaining:
            print(f"{math.floor(s['remaining_usd'] * 100) / 100:.2f}")
        return 0
    print(json.dumps(s, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
