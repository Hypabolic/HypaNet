"""F22 (post Figure 25): "Three rounds, three sessions", the no-tools lineage.

Run from the repo root:  uv run python scripts/blog/figures/f22.py

Reads only committed run records and reports: the no-tools draft's run.json and apply.json and its prompt; the two
revision rounds' BUILD_RECORD.json, LAUNCHES.jsonl, HANDOVER.json and the final result event of each launch's
stream log (cost and turns, as Claude Code reported them); the val, sealed and held-out reports for the result
chips; and experiments/v0_4/STATUS.md for the code-written transformer's corrected v0.3 authoring cost.
Opens no dataset; runs no evaluator or scorer.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F22.json"
GENERATED_BY = "scripts/blog/figures/f22.py"

V05 = ROOT / "experiments" / "v0_5"
R1_RUN = V05 / "runs" / "r1" / "run.json"
R1_APPLY = V05 / "runs" / "r1" / "apply.json"
PROMPT = V05 / "PROMPT.txt"
B_BUILD = ROOT / "weight_program" / "v0_5" / "phase-b" / "BUILD_RECORD.json"
B_LAUNCHES = V05 / "phase_b" / "logs" / "LAUNCHES.jsonl"
B_HANDOVER = V05 / "phase_b" / "HANDOVER.json"
C_BUILD = ROOT / "weight_program" / "v0_5" / "phase-c" / "BUILD_RECORD.json"
C_LAUNCHES = V05 / "phase_c" / "logs" / "LAUNCHES.jsonl"
C_HANDOVER = V05 / "phase_c" / "HANDOVER.json"
VAL_R1 = ROOT / "artifacts" / "reports" / "v05_val_r1.json"
SEALED = ROOT / "artifacts" / "reports" / "v05_sealed.json"
HELDOUT = ROOT / "artifacts" / "reports" / "v05c_heldout.json"
V04_STATUS = ROOT / "experiments" / "v0_4" / "STATUS.md"
SOURCES = [R1_RUN, R1_APPLY, PROMPT, B_BUILD, B_LAUNCHES, B_HANDOVER, C_BUILD, C_LAUNCHES, C_HANDOVER,
           VAL_R1, SEALED, HELDOUT, V04_STATUS]
DP = 5


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def r(x: float) -> float:
    return round(float(x), DP)


def utc(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def minutes(start: str, end: str) -> float:
    return round((utc(end) - utc(start)).total_seconds() / 60, 1)


def launches(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def stream_result(name: str, logs: Path) -> dict:
    """The final `result` event of a launch's stream log: Claude Code's cumulative cost and turns for the session."""
    last = None
    with open(logs / f"{name}.stream.jsonl") as fh:
        for line in fh:
            if '"type":"result"' in line.replace(" ", ""):
                last = json.loads(line)
    if last is None:
        raise SystemExit(f"no result event in {name}.stream.jsonl")
    return {"total_cost_usd": last["total_cost_usd"], "num_turns": last["num_turns"]}


def chip(report_cols: dict, col: str, split: str, population: str) -> dict:
    c = report_cols[col]
    return {
        "split": split,
        "population": population,
        "macro_f1": {"v": r(c["macro_f1"]["value"]), "ci": [r(x) for x in c["macro_f1"]["ci95"]]},
        "false_alarm": r(c["false_alarm_rate"]["value"]),
    }


def v03_costs() -> dict:
    text = V04_STATUS.read_text()
    m = re.search(r"Phase A US\$([\d.]+), Phase B US\$([\d.]+)", text)
    if not m:
        raise SystemExit("v0.3 cost line not found in experiments/v0_4/STATUS.md")
    return {"A": float(m.group(1)), "B": float(m.group(2))}


def build() -> dict:
    run = json.loads(R1_RUN.read_text())
    apply = json.loads(R1_APPLY.read_text())
    b_build = json.loads(B_BUILD.read_text())
    c_build = json.loads(C_BUILD.read_text())
    b_l = launches(B_LAUNCHES)
    c_l = launches(C_LAUNCHES)
    b_res = stream_result(b_l[-1]["name"], B_LAUNCHES.parent)
    c_res = stream_result(c_l[-1]["name"], C_LAUNCHES.parent)
    c_hand = json.loads(C_HANDOVER.read_text())["handed_over"]
    b_hand = json.loads(B_HANDOVER.read_text())["handed_over"]

    val = json.loads(VAL_R1.read_text())
    sealed = json.loads(SEALED.read_text())
    held = json.loads(HELDOUT.read_text())
    s_val = sealed["splits"]["val"]["populations"]["excluding_disclosed"]["overall"]["columns"]
    s_test = sealed["splits"]["test"]["populations"]["all"]["overall"]["columns"]
    s_ood = sealed["splits"]["ood"]["populations"]["all"]["overall"]["columns"]
    h_ood = held["sets"]["heldout_ood"]["populations"]
    h_in = held["sets"]["heldout_indist"]["populations"]["all"]["overall"]["columns"]

    r1_val = val["summary"]["excluding_disclosed"]["network"]
    # The val report's own figure must match the sealed report's re-score of the same network.
    if abs(r1_val["macro_f1"]["value"] - s_val["v0.5 r1 network"]["macro_f1"]["value"]) > 1e-9:
        raise SystemExit("r1 full-val macro-F1 differs between v05_val_r1.json and v05_sealed.json")

    prompt_chars = len(PROMPT.read_text(encoding="utf-8"))
    continues = [t for t in run["turns"] if t["message"] == run["continue_message"]]

    rounds = [
        {
            "id": "r1",
            "name": "No-tools draft",
            "tag": "v0.5 r1",
            "experiment": 4,
            "received": f"one {prompt_chars:,}-character prompt (spec, rules, grammar sheet, renderer, tokeniser, model "
                        f"source, the edits language), then “Continue.” ×{len(continues)}",
            "not_allowed": "tools, code, a calculator, files, data, labels or feedback",
            "labels": 0,
            "tools": "none",
            "cost_usd": r(run["cost"]["total_usd"]),
            "minutes": [minutes(run["utc_created"] if "utc_created" in run else run["events"][0]["utc"],
                                run["utc_stopped"] if "utc_stopped" in run else run["events"][-1]["utc"])],
            "turns": len(run["turns"]),
            "statements": apply["edit_report"]["statements"],
            "weights_note": {"kind": "non-zero weights", "n": apply["nonzero_total"]},
            "results": [
                chip(s_val, "v0.5 r1 network", "full val", "excluding the 200 disclosed sessions"),
                chip(s_test, "v0.5 r1 network", "sealed test", "all"),
                chip(s_ood, "v0.5 r1 network", "sealed OOD", "all"),
            ],
        },
        {
            "id": "B",
            "name": "No-tools revision",
            "tag": "v0.5 B",
            "experiment": 5,
            "received": "round 1's replies and weights, the same packet, and 200 labelled val windows with round 1's "
                        "answers",
            "not_allowed": "automated search or fitting",
            "labels": 200,
            "tools": "run the network and read activations at any layer",
            "cost_usd": r(b_res["total_cost_usd"]),
            "minutes": [minutes(x["utc_start"], x["utc_end"]) for x in b_l],
            "launch_note": "two launches: it hit the plan's usage limit once and was resumed with a fixed "
                           "“continue, same rules” message",
            # The resumed launch's result event counts only its own turns, so no session total is given.
            "turns": None,
            "statements": b_build["statements"],
            "weights_note": {"kind": "weight entries changed", "n": b_build["changed_entries_total"]},
            "results": [
                chip(s_val, "v0.5 B network", "full val", "excluding the 200 disclosed sessions"),
                chip(s_test, "v0.5 B network", "sealed test", "all"),
                chip(s_ood, "v0.5 B network", "sealed OOD", "all"),
            ],
        },
        {
            "id": "C",
            "name": "Third round",
            "tag": "v0.5 Phase C",
            "experiment": 7,
            "received": "round 2's weights and notes, round 1's replies, the same 200 val windows and 200 new labelled "
                        "windows from Codex's generator (6 of its 10 families, re-timed)",
            "not_allowed": "automated search or fitting; nothing from the diagnosis (Experiment 6)",
            "labels": 400,
            "tools": "the same as round 2",
            "cost_usd": r(c_res["total_cost_usd"]),
            "minutes": [minutes(x["utc_start"], x["utc_end"]) for x in c_l],
            "turns": c_res["num_turns"],
            "statements": c_build["statements"],
            "weights_note": {"kind": "weight entries changed", "n": c_build["changed_entries_total"]},
            "results": [
                chip(h_ood["unseen"]["overall"]["columns"], "v0.5 Phase C network", "held-out OOD",
                     "4 unseen families"),
                chip(h_ood["all"]["overall"]["columns"], "v0.5 Phase C network", "held-out OOD", "all 10 families"),
                chip(h_in, "v0.5 Phase C network", "held-out in-distribution", "all"),
            ],
        },
    ]

    # What each new session was handed, checked against the handover records.
    if "r1" not in b_hand or "disclosure" not in b_hand:
        raise SystemExit("phase B handover does not list round 1 and the disclosure")
    if not {"r1", "phase_b", "disclosure"} <= set(c_hand) or "DISCLOSURE_W.json" not in c_hand["disclosure"]:
        raise SystemExit("phase C handover does not list round 1, round 2 and both disclosures")
    if c_build["base_artifact_sha256"] != b_build["artifact_sha256"] or b_build["base_artifact_sha256"] != apply["artifact"]["sha256"]:
        raise SystemExit("lineage artefact hashes do not chain")

    costs = v03_costs()
    s_v03 = {
        "A": [chip(s_test, "v0.3 A network", "sealed test", "all"), chip(s_ood, "v0.3 A network", "sealed OOD", "all")],
        "B": [chip(s_test, "v0.3 B network", "sealed test", "all"), chip(s_ood, "v0.3 B network", "sealed OOD", "all")],
    }
    total = sum(x["cost_usd"] for x in rounds)
    return {
        "_provenance": {
            "source": [rel(p) for p in SOURCES],
            "sha256": {rel(p): sha256(p) for p in SOURCES},
            "source_commits": {"r1": "5b214c6", "phase_b": "b1f9e3b", "phase_c": "6f6980e", "v05_sealed": "cba5a80",
                               "v05c_heldout": "cefd087"},
            "git_commit": git_head(),
            "generated_by": GENERATED_BY,
            "notes": [
                "cost = model usage as Claude Code reported it (API-equivalent): r1 run.json cost.total_usd; B and C "
                "the final result event's total_cost_usd (cumulative across B's resumed launch)",
                "total_usd is the exact sum of the three costs (US$114.023), so US$114.02; the three rounded costs "
                "add to US$114.03",
                "minutes from LAUNCHES.jsonl utc_start/utc_end (B: two launches), r1 from run.json created/stopped",
                "full val = val excluding the 200 disclosed sessions (v05_sealed.json re-score, equal to "
                "v05_val_r1.json for r1); sealed test and sealed OOD scored once; held-out sets scored once",
                "v0.3 lane: A written with code, B revised in the same session; corrected costs from "
                "experiments/v0_4/STATUS.md (Phase A US$42.73, Phase B US$21.89)",
            ],
        },
        "figure": "F22",
        "title": "Three rounds, three sessions",
        "rounds": rounds,
        "total_usd": r(total),
        "v03_lane": {
            "A": {"name": "Zero-label transformer", "tag": "v0.3 A", "made": "written with code, no labels",
                  "cost_usd": costs["A"], "results": s_v03["A"]},
            "B": {"name": "Revised transformer", "tag": "v0.3 B",
                  "made": "revised in the same session after 200 labelled windows", "cost_usd": costs["B"],
                  "results": s_v03["B"]},
        },
        "diagnosis": {"experiment": 6, "shown_to_fable": False,
                      "what": "fresh-batch analysis of why the no-tools revision failed off distribution"},
    }


def main() -> int:
    data = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
