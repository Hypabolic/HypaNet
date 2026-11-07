"""F21 (post Figure 24): "What the third round bought", per-family held-out OOD results.

Run from the repo root:  uv run python scripts/blog/figures/f21.py

Reads only the committed one-time held-out report artifacts/reports/v05c_heldout.json (@ cefd087):
sets.heldout_ood.populations.{all,seen,unseen} (per-family and pooled macro-F1 and false-alarm rate, with
session-bootstrap 95% CIs) and the family split it records. No held-out dataset, evaluator or scorer is opened.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F21.json"
GENERATED_BY = "scripts/blog/figures/f21.py"

REPORT = ROOT / "artifacts" / "reports" / "v05c_heldout.json"
README = ROOT / "experiments" / "v0_5" / "phase_c" / "heldout" / "README.md"
SOURCES = [REPORT, README]
SOURCE_COMMIT = "cefd087"
CLASSES = ("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING")
DP = 5

# Report column -> figure id, name in the post, and how it was made.
NETWORKS = [
    ("v0.5 Phase C network", "C", "Third round", "v0.5 Phase C", "hand",
     "written by Fable; saw 200 labelled windows from this generator (6 of its 10 families)"),
    ("v0.5 B network", "B5", "No-tools revision", "v0.5 B", "hand", "written by Fable; saw none from this generator"),
    ("v0.3 B network", "B3", "Revised transformer", "v0.3 B", "hand",
     "written by Fable with code; saw none from this generator"),
    ("v0.3 C-all seed0", "CALL", "Trained transformer", "v0.3 C-all", "trained",
     "trained on 223,930 labels from the original simulator; saw none from this generator"),
]


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def r(x: float) -> float:
    return round(float(x), DP)


def metric(col: dict, key: str) -> dict:
    m = col[key]
    return {"v": r(m["value"]), "ci": [r(m["ci95"][0]), r(m["ci95"][1])]}


def scores(columns: dict) -> dict:
    out = {}
    for col, nid, *_ in NETWORKS:
        c = columns[col]
        out[nid] = {"macro_f1": metric(c, "macro_f1"), "false_alarm": metric(c, "false_alarm_rate")}
    return out


def build() -> dict:
    rep = json.loads(REPORT.read_text())
    ood = rep["sets"]["heldout_ood"]["populations"]
    split = rep["family_split"]
    seen, unseen = set(split["seen"]), set(split["unseen"])

    families = []
    for key, fam in ood["all"]["per_family"].items():
        name = key.split(":", 1)[1]
        c = fam["columns"]["v0.5 Phase C network"]
        counts = c["confusion_matrix"]["counts"]
        families.append({
            "family": name,
            "seen": name in seen,
            "sessions": fam["sessions"],
            "scored_windows": c["scored_windows"],
            "windows_by_label": {cls: sum(counts[i]) for i, cls in enumerate(CLASSES)},
            "healthy_windows": c["false_alarm_rate"]["denominator"],
            "networks": scores(fam["columns"]),
            # STALLED F1 per network (None where the family has no STALLED windows).
            "stalled_f1": {
                nid: (None if (v := fam["columns"][col]["per_class"]["STALLED"]["f1"].get("value")) is None else r(v))
                for col, nid, *_ in NETWORKS
            },
        })
    if {f["family"] for f in families} != seen | unseen or len(unseen) != 4:
        raise SystemExit("family split does not match the per-family results")
    # Unseen first, then seen; alphabetical within each (the report's order).
    families.sort(key=lambda f: (f["seen"], f["family"]))

    pooled = {}
    for pop in ("unseen", "seen", "all"):
        p = ood[pop]
        pooled[pop] = {
            "sessions": p["population"]["sessions"],
            "scored_windows": p["population"]["scored_windows"],
            "stalled_windows": p["population"]["gate_passed_by_oracle_label"]["STALLED"],
            "networks": scores(p["overall"]["columns"]),
        }

    # Claims the caption makes, checked here so the figure cannot drift from them.
    for f in families:
        n = f["networks"]
        if not (n["C"]["macro_f1"]["v"] > n["B5"]["macro_f1"]["v"] and n["C"]["false_alarm"]["v"] < n["B5"]["false_alarm"]["v"]):
            raise SystemExit(f"{f['family']}: third round is not ahead of v0.5 B on both metrics")
        if not f["seen"]:
            d = n["C"]["macro_f1"]["v"] - n["B3"]["macro_f1"]["v"]
            if d < 0 and d < -0.004:
                raise SystemExit(f"{f['family']}: third round more than 0.004 behind v0.3 B")
    stalled_unseen = {f["family"]: f["windows_by_label"]["STALLED"] for f in families if not f["seen"]}
    if pooled["unseen"]["stalled_windows"] != sum(stalled_unseen.values()):
        raise SystemExit("unseen STALLED windows do not add up")

    return {
        "_provenance": {
            "source": [rel(p) for p in SOURCES],
            "sha256": {rel(p): sha256(p) for p in SOURCES},
            "source_commit": SOURCE_COMMIT,
            "git_commit": git_head(),
            "generated_by": GENERATED_BY,
            "notes": [
                "split: held-out OOD (Codex's independently written generator, all 10 families, 200 sessions each, "
                "re-timed), scored once; per family from sets.heldout_ood.populations.all.per_family",
                "seen = the 6 families the third round's 200 OOD labelled windows were drawn from; unseen = the other 4 "
                "(family_split in the report, fixed by hash before any Phase C data was generated)",
                "macro-F1 over the four classes; false-alarm rate = PRODUCTIVE or RECOVERY windows predicted STALLED "
                "or OSCILLATING, over PRODUCTIVE + RECOVERY windows; 95% session-bootstrap CIs, 10,000 resamples",
                "only the third round saw labelled windows from this generator; the others saw none",
            ],
        },
        "figure": "F21",
        "title": "What the third round bought",
        "split": "held-out OOD",
        "networks": [
            {"id": nid, "column": col, "name": name, "tag": tag, "group": group, "made": made}
            for col, nid, name, tag, group, made in NETWORKS
        ],
        "families": families,
        "pooled": pooled,
        "stalled_unseen_by_family": stalled_unseen,
    }


def main() -> int:
    data = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
