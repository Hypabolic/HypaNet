"""F0 (post Figure 1): teaser panel, one card per route (Write, Write without tools, Train, Modify).

Run from the repo root:  uv run python scripts/blog/figures/f0.py

Reads only committed reports, session cost records, the Phase B builder and circuit notes, the WSP-B build
record, the Phase C training log and the v0.5 r1 replies. Never opens a sealed, held-out or fresh dataset and
runs no evaluator. Writes blog_figures/F0.json with a "_provenance" object.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F0.json"
GENERATED_BY = "scripts/blog/figures/f0.py"
DP = 5

V03 = ROOT / "artifacts/reports/v03_sealed.json"
V05 = ROOT / "artifacts/reports/v05_sealed.json"
WSPB_QUICK = ROOT / "artifacts/reports/v04_val_quick_wspb.json"
Q0_FULL = ROOT / "artifacts/reports/v04_val_qwen0.json"
WSPB_RECORD = ROOT / "weight_program/v0_4/wsp-b/BUILD_RECORD.json"
B_BUILDER = ROOT / "weight_program/v0_3/phase-b/build_weights.py"
B_CIRCUITS = ROOT / "weight_program/v0_3/phase-b/CIRCUITS.md"
V03_B_LOG = ROOT / "experiments/v0_3/logs/v03-phase-b.stream.jsonl"
WSPA_COST = ROOT / "experiments/v0_4/logs/wsp-a.cost.json"
WSPB_COST = ROOT / "experiments/v0_4/logs/wsp-b.cost.json"
R1_RUN = ROOT / "experiments/v0_5/runs/r1/run.json"
V05B_LOG = ROOT / "experiments/v0_5/phase_b/logs/v05b.c1.stream.jsonl"
CALL_LOG = ROOT / "experiments/phase-c-trained/v0_3/all/seed0/training_log.json"
R1_REPLY = ROOT / "experiments/v0_5/runs/r1/replies/001.md"

SOURCES = [V03, V05, WSPB_QUICK, Q0_FULL, WSPB_RECORD, B_BUILDER, B_CIRCUITS, V03_B_LOG, WSPA_COST, WSPB_COST,
           R1_RUN, V05B_LOG, CALL_LOG, R1_REPLY]

# Example hashed-word id for the residue-code mini: zork/zeta/widget.qx, the prompt's grammar-sheet id
# (PROMPT.txt line 657; not benchmark data). F23 uses the same id.
EXAMPLE_ID = 6414
EXAMPLE_WORD = "zork/zeta/widget.qx"


def r(x):
    return None if x is None else round(float(x), DP)


def load(p: Path):
    return json.loads(p.read_text())


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def last_result_cost(log: Path) -> float:
    """Cumulative total_cost_usd of the last result event in a Claude Code stream log."""
    cost = None
    for line in log.read_text().splitlines():
        if '"type":"result"' not in line:
            continue
        cost = json.loads(line)["total_cost_usd"]
    if cost is None:
        raise SystemExit(f"F0: no result event in {rel(log)}")
    return cost


def v03_col(rep: dict, split: str, key: str) -> dict:
    c = rep["summary"][split]["all"][key]
    return {"macro_f1": r(c["macro_f1"][0]), "macro_f1_ci": [r(x) for x in c["macro_f1"][1]],
            "false_alarm": r(c["false_alarm_rate"][0])}


def v05_col(rep: dict, split: str, key: str) -> dict:
    c = rep["summary"][split]["all"]["columns"][key]
    return {"macro_f1": r(c["macro_f1"][0]), "macro_f1_ci": [r(x) for x in c["macro_f1"][1]],
            "false_alarm": r(c["false_alarm_rate"][0])}


def head_lines() -> list[dict]:
    """The output-layer assignments of the Phase B builder (the `# === head ===` block)."""
    text = B_BUILDER.read_text().splitlines()
    start = next(i for i, ln in enumerate(text) if "= head =" in ln)
    out = []
    for i in range(start + 1, len(text)):
        m = re.match(r"^\s+(W\[\d, [A-Z0-9]+\]|b\[\d\]) = (-?\d+\.\d+)$", text[i])
        if m:
            out.append({"lhs": m.group(1), "rhs": m.group(2), "line": i + 1})
        elif text[i].strip().startswith("return"):
            break
    if len(out) != 10:
        raise SystemExit(f"F0: expected 10 head assignments, found {len(out)}")
    return out


def head_logits() -> list[dict]:
    """Section 6 of CIRCUITS.md: each logit's active value (PRODUCTIVE is the constant 0)."""
    rows = []
    for ln in B_CIRCUITS.read_text().splitlines():
        m = re.match(r"^\| ([A-Z]+) \((\d)\) \| `?([^`|]+?)`? \| (\d+)? ?\|", ln)
        if m:
            rows.append({"class": m.group(1), "index": int(m.group(2)), "expression": m.group(3).strip(),
                         "active": int(m.group(4)) if m.group(4) else 0})
    if [x["class"] for x in rows] != ["OSCILLATING", "STALLED", "RECOVERY", "PRODUCTIVE"]:
        raise SystemExit(f"F0: unexpected head table {rows}")
    return rows


def residue_code() -> dict:
    """Code A of the v0.5 r1 draft: one-hot residues mod 4, 5, 7 in d32-47, set by strided fills in reply 001."""
    lit = []
    groups = []
    for ln in R1_REPLY.read_text().splitlines():
        m = re.match(r"^tok_emb\.weight\[(\d+):10240:(\d+), (\d+)\] = 1$", ln.strip())
        if not m:
            continue
        start, step, dim = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not 32 <= dim <= 47:
            continue
        if start <= EXAMPLE_ID < 10240 and (EXAMPLE_ID - start) % step == 0:
            lit.append({"dim": dim, "statement": ln.strip()})
    for mod, lo in ((4, 32), (5, 36), (7, 41)):
        groups.append({"mod": mod, "first_dim": lo, "last_dim": lo + mod - 1})
    if len(lit) != 3:
        raise SystemExit(f"F0: expected 3 lit code-A dims for id {EXAMPLE_ID}, found {lit}")
    return {"id": EXAMPLE_ID, "word": EXAMPLE_WORD, "dims": [32, 47], "groups": groups, "lit": lit,
            "source": rel(R1_REPLY)}


def build() -> dict:
    v03, v05 = load(V03), load(V05)
    wb, q0 = load(WSPB_QUICK), load(Q0_FULL)
    rec = load(WSPB_RECORD)
    call_log = load(CALL_LOG)

    b_test, b_ood = v03_col(v03, "test", "v0.3 B network"), v03_col(v03, "ood", "v0.3 B network")
    a_test = v03_col(v03, "test", "v0.3 A network")
    c_test, c_ood = v03_col(v03, "test", "v0.3 C-all seed0"), v03_col(v03, "ood", "v0.3 C-all seed0")
    r1_test = v05_col(v05, "test", "v0.5 r1 network")
    v5b_test, v5b_ood = v05_col(v05, "test", "v0.5 B network"), v05_col(v05, "ood", "v0.5 B network")
    cols03 = v03["columns"]

    wsp = wb["excluding_disclosed"]
    wsp_b = wsp["overall"]["columns"]["WSP-B"]
    pair = wsp["paired_macro_f1"]["WSP-B - Qwen-0"]
    q0_full = q0["excluding_disclosed"]["overall"]

    tensors = []
    for name, t in rec["tensors"].items():
        m = re.search(r"layers\.(\d+)\.mlp\.(\w+)_proj", name)
        tensors.append({
            "name": name,
            "where": f"layer {m.group(1)} MLP {m.group(2)}" if m else "embedding (label rows)",
            "layer": int(m.group(1)) if m else None,
            "changed_entries": t["changed_entries"],
            "delta_rank": t["delta_rank"],
        })

    wspa_cost = load(WSPA_COST)["phases"]["wsp-a"]["total_usd"]
    wspb_cost = load(WSPB_COST)["phases"]["wsp-b"]["total_usd"]
    r1_cost = load(R1_RUN)["cost"]["total_usd"]

    cards = [
        {
            "id": "write", "group": "hand", "label": "Write",
            "network": "v0.3 B (revised transformer)",
            "parameters": cols03["v0.3 B network"]["nonzero_parameters"]["slots"],
            "nonzero": cols03["v0.3 B network"]["nonzero_parameters"]["nonzero"],
            "labels": cols03["v0.3 B network"]["labelled_examples"],
            "revisions": cols03["v0.3 B network"]["revisions"],
            "sealed_test": b_test, "sealed_ood": b_ood,
            "zero_label_sealed_test": a_test,
            "cost_usd": r(last_result_cost(V03_B_LOG)),
            "cost_note": "model usage (API-equivalent), Phase A and B in one resumed session",
            "head_lines": head_lines(), "head_logits": head_logits(),
            "head_source": f"{rel(B_BUILDER)} @ {git('log', '-1', '--format=%h', '--', rel(B_BUILDER))}",
        },
        {
            "id": "train", "group": "trained", "label": "Train",
            "network": "v0.3 C-all seed0 (trained transformer)",
            "parameters": cols03["v0.3 C-all seed0"]["nonzero_parameters"]["slots"],
            "labels": cols03["v0.3 C-all seed0"]["labelled_examples"],
            "sealed_test": c_test, "sealed_ood": c_ood,
            "train_wall_hours": r(call_log["wall_seconds"]["train_run"] / 3600),
            "train_threads": cols03["v0.3 C-all seed0"]["training_threads"],
            "train_platform": call_log["environment"]["platform"],
        },
        {
            "id": "modify", "group": "qwen", "label": "Modify",
            "network": "v0.4 WSP-B (Qwen3.5-0.8B, hand-edited, revised)",
            "labels": 200,
            "val_quick": {"macro_f1": r(wsp_b["macro_f1"]["value"]), "macro_f1_ci": [r(x) for x in wsp_b["macro_f1"]["ci95"]],
                          "false_alarm": r(wsp_b["false_alarm_rate"]["value"]), "sessions": wsp["overall"]["sessions"]},
            "paired_vs_base": {"v": r(pair["value"]), "ci": [r(x) for x in pair["ci95"]], "population": "val quick look"},
            "base_quick": {"macro_f1": r(wsp["overall"]["columns"]["Qwen-0"]["macro_f1"]["value"]),
                           "macro_f1_ci": [r(x) for x in wsp["overall"]["columns"]["Qwen-0"]["macro_f1"]["ci95"]],
                           "sessions": wsp["overall"]["sessions"], "population": "val quick look (same sessions as WSP-B)"},
            "base_full_val": {"macro_f1": r(q0_full["columns"]["Qwen-0"]["macro_f1"]["value"]), "sessions": q0_full["sessions"]},
            "changed_entries": rec["overall"]["changed_entries"],
            "text_model_params": rec["overall"]["text_model_params"],
            "changed_fraction": rec["overall"]["changed_fraction"],
            "tensors_edited": rec["overall"]["tensors_edited"],
            "tensors": tensors,
            "n_layers": 24,
            "cost_usd": r(wspa_cost + wspb_cost), "cost_parts_usd": [r(wspa_cost), r(wspb_cost)],
            "cost_note": "model usage (API-equivalent), WSP-A plus WSP-B",
        },
        {
            "id": "notools", "group": "hand", "label": "Write without tools",
            "network": "v0.5 r1 (no tools), then v0.5 B (a session with tools revises it)",
            "labels_first": v05["columns"]["v0.5 r1 network"]["labelled_examples"],
            "labels_revision": v05["columns"]["v0.5 B network"]["labelled_examples"],
            "r1_sealed_test": r1_test,
            "b_sealed_test": v5b_test, "b_sealed_ood": v5b_ood,
            "cost_usd": [r(r1_cost), r(last_result_cost(V05B_LOG))],
            "cost_note": "model usage (API-equivalent): r1, then Phase B",
            "residue_code": residue_code(),
        },
    ]

    return {
        "_provenance": {
            "source": [rel(p) for p in SOURCES],
            "sha256": {rel(p): sha256(p) for p in SOURCES},
            "source_commits": {rel(p): git("log", "-1", "--format=%h", "--", rel(p)) for p in SOURCES},
            "git_commit": git("rev-parse", "HEAD"),
            "generated_by": GENERATED_BY,
        },
        "figure": "F0",
        "splits": {
            "sealed_test": {"sessions": v03["splits"]["test"]["populations"]["all"]["overall"]["sessions"],
                            "note": "standard simulator, hashed before any network existed, scored once per version"},
            "sealed_ood": {"sessions": v03["splits"]["ood"]["populations"]["all"]["overall"]["sessions"],
                           "note": "Codex's independent generator, hashed before any network existed, scored once per version"},
            "val_quick": {"sessions": wsp["overall"]["sessions"],
                          "note": "quick look: 1,000 val sessions (VAL_QUICK.txt), excluding the 200 disclosed sessions"},
            "full_val": {"sessions": q0_full["sessions"], "note": "val, excluding the 200 disclosed sessions"},
        },
        "cards": cards,
        "footer": "Synthetic agent logs. Write, Write without tools and Train: sealed sets, opened once per version. "
                  "Modify: validation only; no OOD measured.",
    }


def main() -> None:
    data = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {rel(OUT)}")


if __name__ == "__main__":
    main()
