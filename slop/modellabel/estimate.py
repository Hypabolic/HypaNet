"""Cost estimate for the model labellers, from prompts rendered without calling anything.

Every number here is an estimate to be replaced by the 3-screen pilot's measured usage. Assumptions (stated in the
output as well):

- Tokens = characters / CHARS_PER_TOKEN (3.0: code, JSON and markup tokenise denser than prose).
- Output: OUT_PER_NODE tokens per labelled node (60) plus thinking at effort high of THINK_BASE + THINK_PER_NODE ×
  nodes (1,500 + 120 per node), billed as output.
- Opus 5.5 prices per million tokens: US$4 input, US$20 output, US$0.20 cache read, cache write 1.25 × input (5-minute
  entries). Claude Code has been seen writing 1-hour entries (2 × input; `ephemeral_1h_input_tokens` in the Clouds r1
  log), so a second column prices writes at 2 ×.
- The prefix (preamble + guide, the Opus system prompt) is written to the cache by the first call of a run and read
  by every later one. Claude Code also marks the last message for caching, so each screen's body is billed as a cache
  write, not plain input. RETRY_SHARE (5%) of calls are repeated once.
- CLI overhead per call (the Claude Code request around the prompt, with no tools): 500 tokens, as plain input.
- sealed_real: 250 screens with 9,474 / 422 nodes each (the sealed repos' totals); body size per screen is the
  measured sets' mean scaled by the ratio of nodes per screen. No sealed file is opened.
- Adjudication: 10% of labelable nodes disputed, independently per node, so a screen of n nodes goes to a second pass
  with probability 1 − 0.9ⁿ; each such screen costs one call per model with the screen body, 25 tokens per agreed
  node and 140 per disputed node (both labels), and 60 output tokens plus thinking per disputed node.
- Codex runs on the user's ChatGPT plan: token volume only, with CODEX_OVERHEAD (6,000) tokens of Codex's own
  instructions per call and the same reply assumptions.
- Thinking is the least certain figure, so SCENARIOS prices the totals again with more thinking per call and more
  disputes (up to 8,000 + 600 per node, 25% of nodes disputed). Until the pilot has measured it, the cap is a range.
- The pilot row prices the screens `run.py label --pilot` labels (stratified by size, run.pilot_screens).
"""

from __future__ import annotations

import json
from pathlib import Path

from modellabel import common as c
from modellabel import prompt as pr

CHARS_PER_TOKEN = 3.0
OUT_PER_NODE = 60
THINK_BASE = 1500
THINK_PER_NODE = 120
RETRY_SHARE = 0.05
CLI_OVERHEAD = 500
CODEX_OVERHEAD = 6000
PRICE = {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write_5m": 5.0, "cache_write_1h": 8.0}
SEALED_SCREENS = 250
SEALED_TOTAL_SCREENS = 422
SEALED_TOTAL_NODES = 9474
DISPUTE_SHARE = 0.10
ADJ_AGREED_TOKENS = 25
ADJ_DISPUTED_TOKENS = 140
PILOT_SCREENS = 3
# (name, thinking base, thinking per node, share of nodes disputed)
SCENARIOS = [("assumed", THINK_BASE, THINK_PER_NODE, DISPUTE_SHARE), ("more thinking", 4000, 250, 0.10),
             ("more thinking, more disputes", 4000, 400, 0.20), ("high", 8000, 600, 0.25)]


def tok(chars: float) -> float:
    return chars / CHARS_PER_TOKEN


def measure(ls: c.LabelSet, guide_text: str) -> dict:
    rows = []
    prefix = None
    for scr in ls.screens:
        p = pr.render(scr, guide_text)
        prefix = p.prefix
        rows.append({"screen": scr.id, "body_chars": len(p.body), "nodes": len(scr.labelable())})
    adj_prefix = pr.prefix(guide_text, adjudicate=True)
    return {"set": ls.name, "screens": rows, "prefix_chars": len(prefix or pr.prefix(guide_text)),
            "adj_prefix_chars": len(adj_prefix)}


def label_cost(calls: list, prefix_tokens: float, think_base=THINK_BASE, think_per_node=THINK_PER_NODE) -> dict:
    """calls: [(body_tokens, nodes)] for one run in which the prefix is written once."""
    n = len(calls)
    k = 1 + RETRY_SHARE
    body = sum(b for b, _ in calls) * k
    out = sum(OUT_PER_NODE * x for _, x in calls) * k
    think = sum(think_base + think_per_node * x for _, x in calls) * k
    over = CLI_OVERHEAD * n * k
    reads = prefix_tokens * max(0, n * k - 1)
    writes_prefix = prefix_tokens if n else 0
    usd = {}
    for col, wp in (("usd_5m", PRICE["cache_write_5m"]), ("usd_1h", PRICE["cache_write_1h"])):
        usd[col] = ((writes_prefix + body) * wp + reads * PRICE["cache_read"] + over * PRICE["input"]
                    + (out + think) * PRICE["output"]) / 1e6
    codex_in = (prefix_tokens + CODEX_OVERHEAD) * n * k + body
    return {"calls": round(n * k, 1), "nodes": sum(x for _, x in calls),
            "input_tokens": writes_prefix + reads + body + over, "cached_prefix_tokens": reads,
            "output_tokens": out + think, "thinking_tokens": think, **usd,
            "codex_input_tokens": codex_in, "codex_cached_tokens": (prefix_tokens + CODEX_OVERHEAD) * max(0, n * k - 1),
            "codex_output_tokens": out + think}


def adj_calls(screens: list, share=DISPUTE_SHARE) -> list:
    """Expected second-pass calls for [(body_tokens, nodes)]: each screen weighted by P(at least one dispute)."""
    out = []
    for body, n in screens:
        p = 1 - (1 - share) ** n
        d = max(1.0, share * n / p) if p > 0 else 0  # expected disputed nodes given at least one
        out.append((p, body + ADJ_AGREED_TOKENS * (n - d) + ADJ_DISPUTED_TOKENS * d, d))
    return out


def adj_cost(screens: list, prefix_tokens: float, think_base=THINK_BASE, think_per_node=THINK_PER_NODE,
             share=DISPUTE_SHARE) -> dict:
    """Both models make one call per disputed screen; Opus is priced, Codex counted."""
    rows = adj_calls(screens, share)
    calls = sum(p for p, _, _ in rows)
    k = 1 + RETRY_SHARE
    body = sum(p * b for p, b, _ in rows) * k
    out = sum(p * (OUT_PER_NODE * d) for p, _, d in rows) * k
    think = sum(p * (think_base + think_per_node * d) for p, _, d in rows) * k
    over = CLI_OVERHEAD * calls * k
    reads = prefix_tokens * max(0, calls * k - 1)
    usd = {}
    for col, wp in (("usd_5m", PRICE["cache_write_5m"]), ("usd_1h", PRICE["cache_write_1h"])):
        usd[col] = ((prefix_tokens + body) * wp + reads * PRICE["cache_read"] + over * PRICE["input"]
                    + (out + think) * PRICE["output"]) / 1e6
    return {"calls": round(calls * k, 1), "nodes": round(sum(p * d for p, _, d in rows), 1),
            "input_tokens": prefix_tokens + reads + body + over, "cached_prefix_tokens": reads,
            "output_tokens": out + think, "thinking_tokens": think, **usd,
            "codex_input_tokens": (prefix_tokens + CODEX_OVERHEAD) * calls * k + body,
            "codex_cached_tokens": (prefix_tokens + CODEX_OVERHEAD) * max(0, calls * k - 1),
            "codex_output_tokens": out + think}


def estimate(sets: list, datasets_dir: Path = c.DATASETS_DIR, guide: Path = c.GUIDE, pilot=None) -> dict:
    """pilot(ls) -> the pilot's screens (run.pilot_screens); without it the first PILOT_SCREENS in set order."""
    guide_text = Path(guide).read_text(encoding="utf-8")
    measured, sets_loaded = {}, {}
    for name in ("reference_real", "calibration"):
        sets_loaded[name] = c.load_set(name, datasets_dir)
        measured[name] = measure(sets_loaded[name], guide_text)
    any_m = next(iter(measured.values()))
    prefix_t = tok(any_m["prefix_chars"])
    adj_prefix_t = tok(any_m["adj_prefix_chars"])
    per_set = {}
    for name, m in measured.items():
        per_set[name] = [(tok(r["body_chars"]), r["nodes"]) for r in m["screens"]]
    # sealed_real, scaled
    allrows = [r for m in measured.values() for r in m["screens"]]
    mean_body = sum(r["body_chars"] for r in allrows) / len(allrows)
    mean_nodes = sum(r["nodes"] for r in allrows) / len(allrows)
    sealed_nodes = SEALED_TOTAL_NODES / SEALED_TOTAL_SCREENS
    sealed_body = mean_body * sealed_nodes / mean_nodes
    per_set["sealed_real"] = [(tok(sealed_body), sealed_nodes)] * SEALED_SCREENS
    rows = []
    want = ["reference_real", "calibration", "sealed_real"] if sets == ["all"] else sets
    ref = sets_loaded["reference_real"]
    pilot_ids = [s.id for s in (pilot(ref) if pilot else ref.screens[:PILOT_SCREENS])]
    by_id = {r["screen"]: (tok(r["body_chars"]), r["nodes"]) for r in measured["reference_real"]["screens"]}
    pilot_rows = [by_id[x] for x in pilot_ids]
    rows.append({"row": f"pilot ({len(pilot_rows)} reference_real screens of {', '.join(str(n) for _, n in pilot_rows)}"
                        " labelable nodes, both labellers)", "screens": len(pilot_rows),
                 **label_cost(pilot_rows, prefix_t)})
    for name in want:
        rows.append({"row": f"{name}: labelling", "screens": len(per_set[name]), **label_cost(per_set[name], prefix_t)})
        rows.append({"row": f"{name}: adjudication (10% of nodes disputed)", "screens": len(per_set[name]),
                     **adj_cost(per_set[name], adj_prefix_t)})
    total = {k: sum(r[k] for r in rows[1:]) for k in ("usd_5m", "usd_1h", "input_tokens", "output_tokens",
                                                      "codex_input_tokens", "codex_output_tokens", "calls")}
    total_with_pilot = {k: total[k] + rows[0][k] for k in ("usd_5m", "usd_1h")}
    scen = []
    for sname, tb, tn, share in SCENARIOS:
        part = {}
        for name in want:
            lc = label_cost(per_set[name], prefix_t, tb, tn)
            ac = adj_cost(per_set[name], adj_prefix_t, tb, tn, share)
            part[name] = {k: lc[k] + ac[k] for k in ("usd_5m", "usd_1h")}
        pc = label_cost(pilot_rows, prefix_t, tb, tn)
        scen.append({"scenario": sname, "thinking": f"{tb:,} + {tn} per node", "dispute_share": share,
                     "pilot": {k: pc[k] for k in ("usd_5m", "usd_1h")}, "sets": part,
                     "total": {k: sum(v[k] for v in part.values()) for k in ("usd_5m", "usd_1h")}})
    return {
        "assumptions": {"chars_per_token": CHARS_PER_TOKEN, "output_tokens_per_node": OUT_PER_NODE,
                        "thinking_tokens": f"{THINK_BASE} + {THINK_PER_NODE} per node (effort high)",
                        "retry_share": RETRY_SHARE, "cli_overhead_tokens": CLI_OVERHEAD,
                        "codex_overhead_tokens": CODEX_OVERHEAD, "prices_usd_per_mtok": PRICE,
                        "sealed": f"{SEALED_SCREENS} screens × {sealed_nodes:.2f} nodes, body {tok(sealed_body):,.0f}"
                                  " tokens each (scaled)", "dispute_share": DISPUTE_SHARE},
        "measured": {name: {"screens": len(m["screens"]), "nodes": sum(r["nodes"] for r in m["screens"]),
                            "prefix_chars": m["prefix_chars"], "prefix_tokens": round(tok(m["prefix_chars"])),
                            "body_chars": sum(r["body_chars"] for r in m["screens"]),
                            "body_tokens_mean": round(tok(sum(r["body_chars"] for r in m["screens"])
                                                          / len(m["screens"]))),
                            "body_tokens_max": round(tok(max(r["body_chars"] for r in m["screens"])))}
                     for name, m in measured.items()},
        "rows": rows, "total": total, "total_with_pilot": total_with_pilot, "pilot_screens": pilot_ids,
        "scenarios": scen,
        "note": "Estimate only (characters / 3.0 = tokens; thinking assumed); calibrate with the 3-screen pilot.",
    }


def markdown(est: dict) -> str:
    L = ["| Item | Screens | Calls (Opus) | Opus input tok | Opus output tok (incl. thinking) | Opus US$ (5-min "
         "cache writes) | Opus US$ (1-hour writes) | Codex input tok | Codex output tok |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in est["rows"]:
        L.append(f"| {r['row']} | {r['screens']} | {r['calls']:.0f} | {r['input_tokens'] / 1e6:.2f}M | "
                 f"{r['output_tokens'] / 1e6:.2f}M | {r['usd_5m']:.2f} | {r['usd_1h']:.2f} | "
                 f"{r['codex_input_tokens'] / 1e6:.2f}M | {r['codex_output_tokens'] / 1e6:.2f}M |")
    t = est["total"]
    L.append(f"| **Total (without pilot)** | | {t['calls']:.0f} | {t['input_tokens'] / 1e6:.2f}M | "
             f"{t['output_tokens'] / 1e6:.2f}M | **{t['usd_5m']:.2f}** | {t['usd_1h']:.2f} | "
             f"{t['codex_input_tokens'] / 1e6:.2f}M | {t['codex_output_tokens'] / 1e6:.2f}M |")
    m = est["measured"]
    L += ["", "Measured from rendered prompts: " + "; ".join(
        f"{k}: {v['screens']} screens, {v['nodes']} labelable nodes, prefix {v['prefix_tokens']:,} tokens, body mean "
        f"{v['body_tokens_mean']:,} / max {v['body_tokens_max']:,} tokens" for k, v in m.items()) + ".",
          "", est["note"], "Assumptions: " + json.dumps(est["assumptions"])]
    if est.get("scenarios"):
        names = list(est["scenarios"][0]["sets"])
        L += ["", "Scenarios (Opus US$, labelling and adjudication, 5-min / 1-hour cache writes):", "",
              "| Scenario | Thinking per call | Disputed | Pilot | " + " | ".join(names) + " | Total |",
              "|---|---|---|---|" + "---|" * len(names) + "---|"]
        for sc in est["scenarios"]:
            cell = lambda d: f"{d['usd_5m']:.2f} / {d['usd_1h']:.2f}"  # noqa: E731
            L.append(f"| {sc['scenario']} | {sc['thinking']} | {sc['dispute_share']:.0%} | {cell(sc['pilot'])} | "
                     + " | ".join(cell(sc["sets"][n]) for n in names) + f" | {cell(sc['total'])} |")
    if est.get("pilot_screens"):
        L += ["", "Pilot screens (run.py label --pilot): " + ", ".join(est["pilot_screens"]) + "."]
    return "\n".join(L) + "\n"
