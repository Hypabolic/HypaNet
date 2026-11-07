"""F18 (post Figure 12): "Where the trained network's mistakes go", 4x4 confusion matrices.

Run from the repo root:  uv run python scripts/blog/figures/f18.py

Copies the confusion-matrix counts of the revised written transformer (v0.3 B) and the largest trained
transformer (v0.3 C-all seed0) on the two sealed splits from the committed report
artifacts/reports/v03_sealed.json (splits.{test,ood}.populations.all.overall.columns[*].confusion_matrix).
Rows are the oracle label, columns the predicted class. No evaluator, scorer or dataset is opened.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F18.json"
GENERATED_BY = "scripts/blog/figures/f18.py"
V03 = ROOT / "artifacts" / "reports" / "v03_sealed.json"
CLASSES = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
NETS = [
    {"id": "B", "key": "v0.3 B network", "name": "Revised transformer", "method": "written by Fable, 200 labelled windows"},
    {"id": "C-all", "key": "v0.3 C-all seed0", "name": "Trained transformer", "method": "trained, 223,930 labelled windows, one seed"},
]
SPLITS = [("test", "Sealed test"), ("ood", "Sealed OOD")]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def build() -> dict:
    rep = json.loads(V03.read_text())
    splits = {}
    for sp, name in SPLITS:
        pop = rep["splits"][sp]["populations"]["all"]
        cols = pop["overall"]["columns"]
        nets = []
        for n in NETS:
            cm = cols[n["key"]]["confusion_matrix"]
            assert cm["classes"] == CLASSES and cm["orientation"].startswith("rows = oracle label")
            nets.append({**n, "counts": cm["counts"], "macro_f1": round(cols[n["key"]]["macro_f1"]["value"], 5)})
        scored = pop["population"]["scored_windows"]
        # Every scored window sits in exactly one cell, and both networks score the same windows.
        assert all(sum(map(sum, x["counts"])) == scored for x in nets)
        assert nets[0]["counts"] and [sum(r) for r in nets[0]["counts"]] == [sum(r) for r in nets[1]["counts"]]
        splits[sp] = {"name": name, "sessions": pop["population"]["sessions"], "scored_windows": scored, "nets": nets}

    # The caption's numbers, checked against the report.
    ood = {x["id"]: x["counts"] for x in splits["ood"]["nets"]}
    test = {x["id"]: x["counts"] for x in splits["test"]["nets"]}
    assert ood["C-all"][0][2] == 13828 and sum(ood["C-all"][0]) == 44002
    assert ood["C-all"][3][0] == 4235 and sum(ood["C-all"][3]) == 12632
    assert ood["B"][0][2] == 2041
    assert round(test["C-all"][3][3] / sum(test["C-all"][3]), 3) == 0.984
    assert round(ood["C-all"][3][3] / sum(ood["C-all"][3]), 3) == 0.351
    assert splits["test"]["scored_windows"] == 75022 and splits["ood"]["scored_windows"] == 77538

    return {
        "_provenance": {
            "source": ["artifacts/reports/v03_sealed.json"],
            "sha256": {"artifacts/reports/v03_sealed.json": hashlib.sha256(V03.read_bytes()).hexdigest()},
            "source_commit": _git("log", "-1", "--format=%H", "--", "artifacts/reports/v03_sealed.json"),
            "git_commit": _git("rev-parse", "HEAD"),
            "generated_by": GENERATED_BY,
            "path": "splits.{test,ood}.populations.all.overall.columns[*].confusion_matrix.counts",
        },
        "figure": "F18",
        "title": "Where the trained network's mistakes go",
        "classes": CLASSES,
        "orientation": "rows = oracle label, columns = predicted class",
        "splits": splits,
        "highlights": [
            {"split": "ood", "net": "C-all", "row": 0, "col": 2},
            {"split": "ood", "net": "C-all", "row": 3, "col": 0},
            {"split": "ood", "net": "B", "row": 0, "col": 2},
        ],
        "notes": [
            "sealed splits, scored once (v03_sealed.json, official=true)",
            "C-all is a single training seed (223,930 labelled windows)",
            "shading is row-normalised: each cell's share of its true class",
        ],
    }


def main() -> None:
    obj = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{OUT.relative_to(ROOT)} {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    sys.exit(main())
