"""F15 (post Figure 26): which approach, when. A 2x2 of routes with one tested result per quadrant.

Run from the repo root:  uv run python scripts/blog/figures/f15.py

Reads only committed sealed and validation reports. Writes blog_figures/F15.json with a "_provenance" object.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F15.json"
GENERATED_BY = "scripts/blog/figures/f15.py"
DP = 5

V03 = ROOT / "artifacts/reports/v03_sealed.json"
V02 = ROOT / "artifacts/reports/m4_sealed_v02.json"
WSPB_QUICK = ROOT / "artifacts/reports/v04_val_quick_wspb.json"
SOURCES = [V03, V02, WSPB_QUICK]


def r(x):
    return None if x is None else round(float(x), DP)


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def build() -> dict:
    v03 = json.loads(V03.read_text())
    v02 = json.loads(V02.read_text())
    wb = json.loads(WSPB_QUICK.read_text())

    b = v03["summary"]["ood"]["all"]["v0.3 B network"]["macro_f1"]
    c = v02["summary"]["test"]["C-all mean±sd"]["macro_f1"]
    q = wb["excluding_disclosed"]["overall"]["columns"]["WSP-B"]["macro_f1"]
    q_sessions = wb["excluding_disclosed"]["overall"]["sessions"]

    # Axes: x = do you own the representation? (yes left, no right); y = can you write the rule down? (yes top).
    quadrants = [
        {"id": "write", "rule": True, "own": True, "group": "hand", "route": "Write it",
         "network": "revised transformer (v0.3 B), written by Fable from the spec + 200 labelled windows",
         "metric": {"v": r(b[0]), "ci": [r(x) for x in b[1]], "split": "sealed OOD",
                    "sessions": v03["splits"]["ood"]["populations"]["all"]["overall"]["sessions"],
                    "source": rel(V03), "key": "summary.ood.all['v0.3 B network'].macro_f1"}},
        {"id": "train", "rule": False, "own": True, "group": "trained", "route": "Train it",
         "network": f"warm-up MLP (v0.2 C-all), trained on {v02['columns']['C-all seed0']['labelled_examples']:,} labelled windows, three seeds",
         "metric": {"v": r(c["mean"]), "ci": [r(x) for x in c["seed_mean_ci95"]], "split": "sealed test, three-seed mean",
                    "sessions": v03["splits"]["test"]["populations"]["all"]["overall"]["sessions"], "source": rel(V02), "key": "summary.test['C-all mean±sd'].macro_f1"}},
        {"id": "route", "rule": True, "own": False, "group": "qwen", "route": "Route existing features",
         "network": "Qwen3.5-0.8B hand-edited by Fable (v0.4 WSP-B), 200 labelled windows",
         "metric": {"v": r(q["value"]), "ci": [r(x) for x in q["ci95"]], "split": "val, quick look",
                    "sessions": q_sessions, "source": rel(WSPB_QUICK),
                    "key": "excluding_disclosed.overall.columns['WSP-B'].macro_f1"}},
        {"id": "finetune", "rule": False, "own": False, "group": "prior", "route": "Fine-tune or LoRA",
         "network": "training a small add-on to a frozen pretrained model", "metric": None, "untested": True},
    ]
    return {
        "_provenance": {
            "source": [rel(p) for p in SOURCES],
            "sha256": {rel(p): sha256(p) for p in SOURCES},
            "source_commits": {rel(p): git("log", "-1", "--format=%h", "--", rel(p)) for p in SOURCES},
            "git_commit": git("rev-parse", "HEAD"),
            "generated_by": GENERATED_BY,
        },
        "figure": "F15",
        "axes": {"x": "Do you own the representation?", "y": "Can you write the rule down?"},
        "quadrants": quadrants,
        "note": "Numbers are on different populations and are not comparable across quadrants.",
    }


def main() -> None:
    data = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {rel(OUT)}")


if __name__ == "__main__":
    main()
