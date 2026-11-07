"""F25 (post Figure 21): "Level in distribution, not off it", per-family sealed OOD results for three networks.

Run from the repo root:  uv run python scripts/blog/figures/f25.py

Reads only the committed sealed report artifacts/reports/v05_sealed.json (the one-time pre-registered run; the
scored numbers are already in it) and checks that the protocol README it cites exists. Exports macro-F1 and the
false-alarm rate, each with its 95% session-bootstrap CI, for the no-tools revision (v0.5 B), the revised
transformer (v0.3 B) and the trained transformer (v0.3 C-all) in each of the 10 sealed OOD families, plus the
overall sealed test and OOD rows for the table note. No dataset, evaluator or scorer is opened.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F25.json"
GENERATED_BY = "scripts/blog/figures/f25.py"

REPORT = ROOT / "artifacts" / "reports" / "v05_sealed.json"
README = ROOT / "experiments" / "v0_5" / "sealed" / "README.md"
SOURCES = [REPORT, README]

MODELS = [
    {"key": "v05b", "column": "v0.5 B network", "name": "No-tools revision", "tag": "v0.5 B"},
    {"key": "v03b", "column": "v0.3 B network", "name": "Revised transformer", "tag": "v0.3 B"},
    {"key": "call", "column": "v0.3 C-all seed0", "name": "Trained transformer", "tag": "v0.3 C-all"},
]
DP = 5


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def r(x: float) -> float:
    return round(float(x), DP)


def metrics(col: dict) -> dict:
    return {
        "macro_f1": {"v": r(col["macro_f1"]["value"]), "ci": [r(x) for x in col["macro_f1"]["ci95"]]},
        "false_alarm": {
            "v": r(col["false_alarm_rate"]["value"]),
            "ci": [r(x) for x in col["false_alarm_rate"]["ci95"]],
            "windows": col["false_alarm_rate"]["windows"],
            "denominator": col["false_alarm_rate"]["denominator"],
        },
        "scored_windows": col["scored_windows"],
    }


def build() -> dict:
    assert README.exists()
    rep = json.loads(REPORT.read_text())
    ood = rep["splits"]["ood"]["populations"]["all"]
    test = rep["splits"]["test"]["populations"]["all"]

    families = []
    for fam_key, fam in ood["per_family"].items():
        cols = fam["columns"]
        families.append({
            "family": fam_key.split(":", 1)[1],
            "sessions": fam["sessions"] if isinstance(fam["sessions"], int) else len(fam["sessions"]),
            "models": {m["key"]: metrics(cols[m["column"]]) for m in MODELS},
        })

    overall = {
        split: {m["key"]: metrics(pop["overall"]["columns"][m["column"]]) for m in MODELS}
        for split, pop in (("sealed test", test), ("sealed OOD", ood))
    }
    sessions = {"sealed test": test["population"]["sessions"], "sealed OOD": ood["population"]["sessions"]}

    obj = {
        "_provenance": {
            "source": [rel(p) for p in SOURCES],
            "sha256": {rel(p): sha256(p) for p in SOURCES},
            "source_commits": {rel(p): git("log", "-1", "--format=%h", "--", rel(p)) for p in SOURCES},
            "git_commit": git("rev-parse", "HEAD"),
            "generated_by": GENERATED_BY,
            "fields": "splits.ood.populations.all.per_family; splits.{test,ood}.populations.all.overall",
            "bootstrap": "95% session-bootstrap CIs, 10,000 resamples (as reported)",
        },
        "split": "sealed OOD",
        "models": [{k: m[k] for k in ("key", "name", "tag")} for m in MODELS],
        "families": families,
        "overall": overall,
        "sessions": sessions,
    }

    # The figure's claims, checked against the built data.
    assert len(families) == 10
    f1 = lambda f, k: f["models"][k]["macro_f1"]["v"]  # noqa: E731
    fa = lambda f, k: f["models"][k]["false_alarm"]["v"]  # noqa: E731
    assert all(f1(f, "v05b") < f1(f, "v03b") for f in families), "v0.5 B behind v0.3 B in every family"
    assert all(f1(f, "v05b") > f1(f, "call") for f in families), "v0.5 B ahead of C-all in every family"
    fewer = sorted(f["family"] for f in families if fa(f, "call") < fa(f, "v05b"))
    assert fewer == ["index_watermark", "lockfile_churn", "shard_migration", "telemetry_gaps"], fewer
    assert (round(min(fa(f, "v05b") for f in families) * 100), round(max(fa(f, "v05b") for f in families) * 100)) == (12, 48)
    assert (round(min(fa(f, "v03b") for f in families) * 100), round(max(fa(f, "v03b") for f in families) * 100)) == (0, 15)
    o = overall["sealed OOD"]
    assert (round(o["v05b"]["macro_f1"]["v"], 3), round(o["v03b"]["macro_f1"]["v"], 3), round(o["call"]["macro_f1"]["v"], 3)) == (0.765, 0.908, 0.488)
    assert sessions == {"sealed test": 3996, "sealed OOD": 2000}
    return obj


def main() -> None:
    obj = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{OUT.relative_to(ROOT)} {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    sys.exit(main())
