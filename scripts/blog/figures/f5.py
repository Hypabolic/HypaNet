"""F5 (post Figure 7): "Counting with a softmax", the 1/(1+C) counter of the v0.3 transformer.

Run from the repo root:  uv run python scripts/blog/figures/f5.py

Analytic. The figure's numbers are the closed form of head H2.0 in the zero-label v0.3 network
(Phase A): the start token (BOS) and every one of C matching earlier blocks get the same attention
score, so each gets weight 1/(1+C). The constants (the shared score, the K = 4 threshold written as
3.5 * ANCH2, the 40x sharpness of the MLP unit that reads it, the 7-block reach of the head) are read
from the committed CIRCUITS.md and build_weights.py and checked against each other. No model is run,
and no dataset, evaluator or scorer is opened.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F5.json"
GENERATED_BY = "scripts/blog/figures/f5.py"

A_DIR = ROOT / "weight_program" / "v0_3" / "phase-a"
B_DIR = ROOT / "weight_program" / "v0_3" / "phase-b"
CIRCUITS_A = A_DIR / "CIRCUITS.md"
BUILD_A = A_DIR / "build_weights.py"
CIRCUITS_B = B_DIR / "CIRCUITS.md"
SOURCES = [CIRCUITS_A, BUILD_A, CIRCUITS_B]
SOURCE_COMMITS = {"phase_a": "eb38b33", "phase_b": "a64a4f3"}
DP = 5


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def r(x: float) -> float:
    return round(float(x), DP)


def need(pattern: str, text: str, what: str) -> re.Match:
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"F5: could not find {what}")
    return m


def clip01(v: float) -> float:
    return min(1.0, max(0.0, v))


def build() -> dict:
    circ_a = CIRCUITS_A.read_text()
    build_a = BUILD_A.read_text()
    circ_b = CIRCUITS_B.read_text()

    # Head H2.0 (Phase A): BOS anchor and exact matches share one score; non-matches score lower.
    h20 = need(r"\| H2\.0 \|[^\n]*BOS (\d+) = exact match[^\n]*non-match <= (\d+)", circ_a, "H2.0 row").groups()
    shared_score, nonmatch_max = int(h20[0]), int(h20[1])
    reach = need(r"\| H2\.0 \|[^\n]*`B\[1\.\.(\d+)\] = 0`", circ_a, "H2.0 window").group(1)
    reach = int(reach)

    # Retry-beyond-policy unit (MLP 2, units 7-8): gated clip 40*(OUTFAIL - 3.5*ANCH2).
    row = need(r"\| 7-8 \| retry beyond policy \(non-poll, `C_fail >= (\d+)`\) \| gated clip `(\d+)\*\(OUTFAIL - ([\d.]+)\*ANCH2\)",
               circ_a, "retry-beyond-policy row").groups()
    k_doc, kstep_doc, half_doc = int(row[0]), float(row[1]), float(row[2])
    kstep = float(need(r"(?m)^KSTEP = ([\d.]+)", build_a, "KSTEP").group(1))
    code = need(r"clip_unit\(2, \{OUTFAIL: KSTEP, ANCH2: -([\d.]+) \* KSTEP, BSFAIL: KGATE, BTPOLL: -KGATE\}", build_a,
                "retry clip_unit").group(1)
    if not (kstep == kstep_doc and float(code) == half_doc and half_doc == k_doc - 0.5):
        raise SystemExit("F5: CIRCUITS.md and build_weights.py disagree on the K = 4 test")

    # Phase B moves the counter to H1.4 (layer 1); record its shared score and reach for the table note.
    h14 = need(r"\| H1\.4 \|[^\n]*`B\[1\.\.(\d+)\] = 0`, BOS ([\d,]+) \(anchor = exact match\)", circ_b, "H1.4 row").groups()
    b_reach, b_score = int(h14[0]), int(h14[1].replace(",", ""))

    rows = []
    for c in range(0, reach + 1):
        anchor = 1.0 / (1 + c)
        outfail = c / (1 + c)  # every matching block failed: C_fail = C_all = C
        threshold = (k_doc - 0.5) * anchor
        test = outfail - threshold
        rows.append({
            "c": c,
            "anchor": r(anchor),
            "matches_total": r(outfail),
            "each_match": r(anchor) if c else 0.0,
            "threshold": r(threshold),
            "test": r(test),
            "unit_pre": r(kstep * test),
            "unit_out": r(clip01(kstep * test)),
            "fires": test >= 0,
        })

    return {
        "_provenance": {
            "source": [rel(p) for p in SOURCES],
            "sha256": {rel(p): sha256(p) for p in SOURCES},
            "source_commits": SOURCE_COMMITS,
            "git_commit": git_head(),
            "generated_by": GENERATED_BY,
            "kind": "analytic",
        },
        "figure": "F5",
        "title": "Counting with a softmax",
        "assumption": "every matching block failed, so C_fail = C_all = C",
        "head": {
            "phase_a": {"name": "H2.0", "layer": 2, "shared_score": shared_score, "nonmatch_max": nonmatch_max,
                        "reach_blocks": reach},
            "phase_b": {"name": "H1.4", "layer": 1, "shared_score": b_score, "reach_blocks": b_reach},
        },
        "rule": {"name": "retry beyond policy (non-poll)", "k": k_doc, "coef": half_doc, "kstep": kstep,
                 "units": "MLP 2 units 7-8", "expr": f"{int(kstep)}*(OUTFAIL - {half_doc}*ANCH2)"},
        "rows": rows,
    }


if __name__ == "__main__":
    data = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {rel(OUT)}: C = 0..{data['rows'][-1]['c']}, K = {data['rule']['k']}")
