"""F28 and F29 (the TL;DR pair): every version's score, and its score against the time and model cost it took.

Run from the repo root:  uv run python scripts/blog/figures/f28.py

One data file for two figures: F28 (bars, one row per version) and F29 (score against hours and against model
cost) both read blog_figures/F28.json, so the two can never disagree.

Reads only committed reports, the Claude Code stream logs of each authoring session (the last `result` event of a
launch gives its cumulative total_cost_usd and its own duration_ms) and the trained transformers' training logs
(wall_seconds.train_run). Never opens a sealed, held-out or fresh dataset and runs no evaluator.

Times are wall clock. Fable's are its authoring sessions (every launch, including the v0.3 first launch that hit the
plan's usage limit before writing anything, because its cost is in the v0.3 total too). Training's is the training
run itself on two CPU threads, mean over seeds. None include writing the spec, building the harness, generating
labels, or the Qwen readouts Fable ran outside its sessions.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F28.json"
GENERATED_BY = "scripts/blog/figures/f28.py"
DP = 5

V03 = ROOT / "artifacts/reports/v03_sealed.json"
V05 = ROOT / "artifacts/reports/v05_sealed.json"
WSPB_QUICK = ROOT / "artifacts/reports/v04_val_quick_wspb.json"
V03_LOGS = ROOT / "experiments/v0_3/logs"
V04_LOGS = ROOT / "experiments/v0_4/logs"
V05_LOGS = ROOT / "experiments/v0_5/logs"
V05B_LOGS = ROOT / "experiments/v0_5/phase_b/logs"
TRAINED = ROOT / "experiments/phase-c-trained/v0_3"
C_BUDGETS = ("32", "128", "512", "2048", "all")
RESUMED = {"v03B", "wspb"}
# Training's cost on the same axis as Fable's: renting the smallest Azure H100 VM for the run's measured wall time.
# That VM would run the same job no slower than the Intel Mac did, so this is a ceiling; on the GPU it would likely
# take minutes. Price from the Azure Retail Prices API (https://prices.azure.com/api/retail/prices), retrieved
# 2026-11-08; the meter has been effective since 2024-03-01.
AZURE_H100 = {
    "vm": "Standard_NC40ads_H100_v5 (1x H100 NVL)",
    "region": "East US",
    "os": "Linux",
    "price_type": "pay-as-you-go (Consumption)",
    "usd_per_hour": 6.98,
    "source": "https://prices.azure.com/api/retail/prices?$filter=armSkuName eq 'Standard_NC40ads_H100_v5' and armRegionName eq 'eastus'",
    "retrieved": "2026-11-08",
    "effective_from": "2024-03-01",
}

# Each written or edited version: the launches of its own session, in order. Cost is cumulative within a session.
SESSIONS = {
    "v03A": [V03_LOGS / "v03-phase-a.stream.jsonl", V03_LOGS / "v03-phase-a.c1.stream.jsonl"],
    "v03B": [V03_LOGS / "v03-phase-b.stream.jsonl"],
    "r1": [V05_LOGS / f"r1.t00{i}.stream.jsonl" for i in (1, 2, 3, 4)],
    "v05B": [V05B_LOGS / "v05b.stream.jsonl", V05B_LOGS / "v05b.c1.stream.jsonl"],
    "wspa": [V04_LOGS / "wsp-a.stream.jsonl"],
    "wspb": [V04_LOGS / "wsp-b.stream.jsonl"],
}


def r(x):
    return None if x is None else round(float(x), DP)


def load(p: Path):
    return json.loads(p.read_text())


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def last_result(log: Path) -> dict:
    last = None
    for line in log.read_text().splitlines():
        if '"type":"result"' in line.replace(" ", ""):
            last = json.loads(line)
    if last is None:
        raise SystemExit(f"F28: no result event in {rel(log)}")
    return last


def session(key: str) -> dict:
    """Minutes summed over a session's launches; cost is the last launch's cumulative total."""
    results = [last_result(p) for p in SESSIONS[key]]
    return {"minutes": round(sum(x["duration_ms"] for x in results) / 60000, 1),
            "cost_usd": r(results[-1]["total_cost_usd"]),
            "launches": [{"log": rel(p), "minutes": round(x["duration_ms"] / 60000, 1)} for p, x in zip(SESSIONS[key], results)]}


def ci_pair(pair) -> dict:
    return {"v": r(pair[0]), "ci": [r(pair[1][0]), r(pair[1][1])]}


def build() -> dict:
    v03, v05, wb = load(V03), load(V05), load(WSPB_QUICK)
    s03 = v03["summary"]
    s05 = v05["summary"]

    def sealed03(key):
        return {sp: ci_pair(s03[sp]["all"][key]["macro_f1"]) for sp in ("test", "ood")}

    def sealed05(key):
        return {sp: ci_pair(s05[sp]["all"]["columns"][key]["macro_f1"]) for sp in ("test", "ood")}

    quick = wb["excluding_disclosed"]
    qcols = quick["overall"]["columns"]

    def quick_look(key):
        return {"val_quick": {"v": r(qcols[key]["macro_f1"]["value"]), "ci": [r(x) for x in qcols[key]["macro_f1"]["ci95"]]}}

    sess = {k: session(k) for k in SESSIONS}
    trained_sources: list[Path] = []
    trained = []
    for b in C_BUDGETS:
        g = f"v0.3 C-{b}"
        keys = v03["phase_c_groups"][g]
        logs = [TRAINED / b / f"seed{k.rsplit('seed', 1)[1]}" / "training_log.json" for k in keys]
        trained_sources += logs
        hours = [load(p)["wall_seconds"]["train_run"] / 3600 for p in logs]
        trained.append({
            "id": f"C-{b}", "key": g, "n_seeds": len(keys),
            "labels": v03["columns"][keys[0]]["labelled_examples"],
            "test": r(s03["test"]["all"][f"{g} mean±sd"]["macro_f1"]["mean"]),
            "ood": r(s03["ood"]["all"][f"{g} mean±sd"]["macro_f1"]["mean"]),
            "train_hours": r(sum(hours) / len(hours)),
            "train_hours_per_seed": [r(h) for h in hours],
            "est_cost_usd": r(sum(hours) / len(hours) * AZURE_H100["usd_per_hour"]),
            "threads": v03["columns"][keys[0]]["training_threads"],
        })

    def step(key, label, exp, labels, scores, session_key, after=None):
        s = sess[session_key]
        prev = next((x for x in versions if x["id"] == after), None) if after else None
        # v0.3 B resumed the Phase A session and WSP-B resumed WSP-A's, so their logs' cumulative cost already
        # includes the draft; v0.5 B was a new session.
        resumed = session_key in RESUMED
        total = s["cost_usd"] if resumed or not prev else s["cost_usd"] + prev["cost_usd"]
        return {
            "id": key, "label": label, "experiment": exp, "labels": labels, **scores,
            "session_minutes": s["minutes"],
            "session_cost_usd": r(s["cost_usd"] - prev["cost_usd"]) if resumed and prev else s["cost_usd"],
            # Totals for the lineage up to and including this version (a revision builds on its draft).
            "hours": r((s["minutes"] + (prev["hours"] * 60 if prev else 0)) / 60),
            "cost_usd": r(total),
            "launches": s["launches"],
        }

    versions: list[dict] = []
    versions.append({**step("written-A", "Written, no labels", 2, 0, sealed03("v0.3 A network"), "v03A"), "lineage": "written"})
    versions.append({**step("written-B", "Written, revised with 200 labels", 3, 200, sealed03("v0.3 B network"), "v03B",
                            after="written-A"), "lineage": "written"})
    versions.append({**step("notools-r1", "No-tools draft", 4, 0, sealed05("v0.5 r1 network"), "r1"), "lineage": "notools"})
    versions.append({**step("notools-B", "Draft repaired with tools and 200 labels", 5, 200, sealed05("v0.5 B network"), "v05B",
                            after="notools-r1"), "lineage": "notools"})
    versions.append({"id": "qwen-base", "label": "Base Qwen, unedited", "experiment": None, "labels": 0, "lineage": "qwen",
                     **quick_look("Qwen-0"), "hours": 0.0, "cost_usd": 0.0, "session_minutes": 0.0, "session_cost_usd": 0.0,
                     "launches": []})
    versions.append({**step("qwen-A", "Qwen, edited with no labels", 9, 0, quick_look("WSP-A"), "wspa"), "lineage": "qwen"})
    versions.append({**step("qwen-B", "Qwen, edit revised with 200 labels", 10, 200, quick_look("WSP-B"), "wspb",
                            after="qwen-A"), "lineage": "qwen"})

    # Sanity: the totals the post quotes.
    by = {v["id"]: v for v in versions}
    expect = {"written-B": 64.62, "notools-B": 57.44, "qwen-B": 62.39}
    for k, usd in expect.items():
        if abs(by[k]["cost_usd"] - usd) > 0.01:
            raise SystemExit(f"F28: {k} total US${by[k]['cost_usd']} != post's US${usd}")

    sources = [V03, V05, WSPB_QUICK, *[p for ps in SESSIONS.values() for p in ps], *trained_sources]
    return {
        "_provenance": {
            "source": [rel(p) for p in sources],
            "sha256": {rel(p): sha256(p) for p in sources},
            "git_commit": git_head(),
            "generated_by": GENERATED_BY,
        },
        "figure": "F28",
        "title": "Every version, scored; and against time and cost",
        "metrics": "macro-F1; sealed test and sealed OOD (each scored once, 95% session-bootstrap CI) for the "
                   "transformers; val quick look (1,000 val sessions excluding the 200 disclosed) for Qwen",
        "notes": [
            "hours and cost_usd are lineage totals: a revision's include its draft's session",
            "Fable minutes: sum of duration_ms over a session's launches (Claude Code result events); "
            "v0.3 A includes a first launch cut off by the plan's usage limit before it wrote anything",
            "cost: Fable's is model usage (API-equivalent) as Claude Code reported it; training's (est_cost_usd) is a "
            "ceiling: train_hours x the Azure H100 VM's hourly price (azure_h100), since that VM would run the job no slower",
            "trained hours: wall_seconds.train_run, mean over seeds, two CPU threads on an Intel Mac; "
            "excludes generating the labels (free, from the simulator)",
            "Qwen sessions ran on an M4 Pro; the val readouts (about 5 h per full val pass) are not included",
            "base Qwen has no session: it is the starting point of the Qwen lineage",
        ],
        "quick_look_sessions": quick["overall"]["sessions"],
        "azure_h100": AZURE_H100,
        "versions": versions,
        "trained": trained,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {rel(OUT)}")


if __name__ == "__main__":
    main()
