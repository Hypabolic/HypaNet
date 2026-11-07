"""F8 (post Figure 8): "Where the errors live", the 200 disclosed windows before and after the revision.

Run from the repo root:  uv run python scripts/blog/figures/f8.py

Sources (all committed; no evaluator, scorer or sealed dataset is opened):
- experiments/v0_3/disclosure/DISCLOSURE.json: the 200 disclosed val windows with the oracle label, the archived
  zero-label (Phase A) network's class and the Phase A reference policy's class. Only those three fields and the
  window id are read; the rendered prefixes are not exported.
- weight_program/v0_3/phase-b/REVISION.md: the revised (Phase B) network scores 199 / 200 at the endpoints, the one
  miss is V053 (PRODUCTIVE, predicted OSCILLATING), and it agrees with the revised policy on 4,484 / 4,484 prefixes.
  The Phase B class of every other window is therefore its oracle label; this file is the only source for it.
- artifacts/reports/v03_sealed.json: per-class recall of the A and B networks on full val excluding the disclosed
  sessions (val/excluding_disclosed), sealed test and sealed OOD.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F8.json"
GENERATED_BY = "scripts/blog/figures/f8.py"
DISCLOSURE = ROOT / "experiments" / "v0_3" / "disclosure" / "DISCLOSURE.json"
REVISION = ROOT / "weight_program" / "v0_3" / "phase-b" / "REVISION.md"
V03 = ROOT / "artifacts" / "reports" / "v03_sealed.json"
CLASSES = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
RECALL_SPLITS = [
    ("val", "excluding_disclosed", "Full val", "full val, excluding the 200 disclosed sessions"),
    ("test", "all", "Sealed test", "sealed test"),
    ("ood", "all", "Sealed OOD", "sealed OOD"),
]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def _phase_b_from_revision(text: str) -> dict:
    """The facts REVISION.md states about the revised network on the disclosure."""
    m = re.search(r"at the 200 endpoints it scores (\d+) / 200 against the oracle", text)
    assert m, "REVISION.md: endpoint score not found"
    right = int(m.group(1))
    m = re.search(r"Agreement of the rebuilt network with the revised policy: ([\d,]+) / ([\d,]+) prefixes", text)
    assert m, "REVISION.md: prefix agreement not found"
    agree, prefixes = int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
    misses = re.findall(r"\*\*(V\d{3})\*\* \((\w+), predicted (\w+)\)", text)
    assert len(misses) == 200 - right, misses
    return {"right": right, "agree": agree, "prefixes": prefixes, "misses": {w: (t, p) for w, t, p in misses}}


def build() -> dict:
    disc = json.loads(DISCLOSURE.read_text())
    assert disc["class_names"] == CLASSES
    revision = _phase_b_from_revision(REVISION.read_text())

    windows = []
    for w in disc["windows"]:
        true, a_net, a_pol = w["oracle_label"], w["network_class"], w["reference_policy_class"]
        miss = revision["misses"].get(w["id"])
        if miss:
            assert miss[0] == true
        b = miss[1] if miss else true
        windows.append({"id": w["id"], "true": true, "a": a_net, "a_policy": a_pol, "b": b})

    a_right = sum(w["a"] == w["true"] for w in windows)
    a_disagree = sum(w["a"] != w["a_policy"] for w in windows)
    b_right = sum(w["b"] == w["true"] for w in windows)
    per_class = Counter(w["true"] for w in windows)
    assert len(windows) == 200 and all(per_class[c] == 50 for c in CLASSES)
    assert a_right == 141 and a_disagree == 0 and b_right == revision["right"] == 199
    assert revision["agree"] == revision["prefixes"] == 4484

    def confusion(key: str) -> list[list[int]]:
        return [[sum(1 for w in windows if w["true"] == t and w[key] == p) for p in CLASSES] for t in CLASSES]

    rep = json.loads(V03.read_text())
    recall = []
    for sp, pop, name, long in RECALL_SPLITS:
        P = rep["splits"][sp]["populations"][pop]
        cols = P["overall"]["columns"]
        row = {"split": sp, "population": pop, "name": name, "long": long,
               "sessions": P["population"]["sessions"], "scored_windows": P["population"]["scored_windows"], "classes": {}}
        for c in CLASSES:
            row["classes"][c] = {
                net: {"v": round(cols[key]["per_class"][c]["recall"]["value"], 5),
                      "ci": [round(x, 5) for x in cols[key]["per_class"][c]["recall"]["ci95"]],
                      "support": cols[key]["per_class"][c]["support"]}
                for net, key in (("A", "v0.3 A network"), ("B", "v0.3 B network"))
            }
        recall.append(row)

    st = {r["split"]: (round(r["classes"]["STALLED"]["A"]["v"], 3), round(r["classes"]["STALLED"]["B"]["v"], 3)) for r in recall}
    assert st == {"val": (0.369, 0.953), "test": (0.356, 0.956), "ood": (0.364, 0.951)}, st

    sources = [DISCLOSURE, REVISION, V03]
    return {
        "_provenance": {
            "source": [_rel(p) for p in sources],
            "sha256": {_rel(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "source_commit": {_rel(p): _git("log", "-1", "--format=%H", "--", _rel(p)) for p in sources},
            "git_commit": _git("rev-parse", "HEAD"),
            "generated_by": GENERATED_BY,
            "derivation": "window b class = oracle label except the misses REVISION.md names (V053 PRODUCTIVE -> OSCILLATING); "
                          "REVISION.md states 199/200 at the endpoints and 4,484/4,484 prefix agreement with the revised policy",
        },
        "figure": "F8",
        "title": "Where the errors live",
        "classes": CLASSES,
        "disclosed": {
            "windows": 200,
            "per_class": 50,
            "split": "200 disclosed val windows (in-sample: Fable saw them for the revision)",
            "a": {"name": "Zero-label transformer", "right": a_right, "network_ne_policy": a_disagree,
                  "agreement_unit": "windows", "agreement_of": 200, "confusion": confusion("a")},
            "b": {"name": "Revised transformer", "right": b_right, "network_ne_policy": revision["prefixes"] - revision["agree"],
                  "agreement_unit": "prefixes", "agreement_of": revision["prefixes"], "confusion": confusion("b")},
        },
        "windows": windows,
        "recall": recall,
    }


def main() -> None:
    obj = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{OUT.relative_to(ROOT)} {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    sys.exit(main())
