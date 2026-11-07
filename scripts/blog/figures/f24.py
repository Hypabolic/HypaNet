"""F24 (post Figure 20): "One bias, every path a repeat", r1's key-gate bias before and after the one-line fix.

Run from the repo root:  uv run python scripts/blog/figures/f24.py

Reads only committed text: r1's reply 003.md (the gate formula and the two `bias = -2` statements), the prompt
(HASH_BASE, and the grammar-sheet id of zork/zeta/widget.qx used for the example key), r1's reply 001.md (the
code-A fills, to light the right cells), and Phase B's ITERATIONS.md and REVISION.md (the fix statement, the
activations Fable read on disclosed window V001, the two confusion matrices on the 200 disclosed windows, and the
141-of-200 count). The key values are computed from the gate formula; the confusion matrices are parsed and their
macro-F1 and false-alarm rate recomputed and checked against the stated values. Everything is an in-sample
self-check on the 200 disclosed windows, not an evaluation. No dataset, evaluator or scorer is opened.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F24.json"
GENERATED_BY = "scripts/blog/figures/f24.py"

REPLY_1 = ROOT / "experiments" / "v0_5" / "runs" / "r1" / "replies" / "001.md"
REPLY_3 = ROOT / "experiments" / "v0_5" / "runs" / "r1" / "replies" / "003.md"
PROMPT = ROOT / "experiments" / "v0_5" / "PROMPT.txt"
ITERATIONS = ROOT / "weight_program" / "v0_5" / "phase-b" / "ITERATIONS.md"
REVISION = ROOT / "weight_program" / "v0_5" / "phase-b" / "REVISION.md"
SOURCES = [REPLY_3, REPLY_1, PROMPT, ITERATIONS, REVISION]

CLASSES = ("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING")
EXAMPLE_PATH = "zork/zeta/widget.qx"


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def code_a_cells(word_id: int, base: int) -> list[int]:
    """Indices 0..15 of the code-A cells (d32-47) lit for a word id, from r1's strided fills."""
    fills = []
    for line in REPLY_1.read_text().splitlines():
        m = re.match(r"^tok_emb\.weight\[(\d+):10240:(\d+), (\d+)\] = 1\s*$", line)
        if m and int(m.group(1)) >= base and 32 <= int(m.group(3)) < 48:
            fills.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    assert len(fills) == 16
    return sorted(dim - 32 for start, step, dim in fills if (word_id - start) % step == 0)


def parse_matrix(text: str) -> list[list[int]]:
    rows = []
    for cls in CLASSES:
        m = re.search(rf"{cls} (\d+)/(\d+)/(\d+)/(\d+)", text)
        rows.append([int(x) for x in m.groups()])
    return rows


def macro_f1(cm: list[list[int]]) -> float:
    f1s = []
    for k in range(4):
        tp = cm[k][k]
        pred = sum(cm[r][k] for r in range(4))
        sup = sum(cm[k])
        p = tp / pred if pred else 0.0
        r = tp / sup if sup else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(f1s) / 4


def false_alarm(cm: list[list[int]]) -> float:
    """Share of PRODUCTIVE or RECOVERY windows called STALLED or OSCILLATING."""
    healthy = cm[0] + cm[1]
    return (cm[0][2] + cm[0][3] + cm[1][2] + cm[1][3]) / (sum(cm[0]) + sum(cm[1])) if healthy else 0.0


def section(text: str, start: str, end: str) -> str:
    return text[text.index(start):text.index(end)]


def build() -> dict:
    r3 = REPLY_3.read_text()
    prompt = PROMPT.read_text()
    iters = ITERATIONS.read_text()
    revision = REVISION.read_text()

    base = int(re.search(r"^HASH_BASE = (\d+)", prompt, re.M).group(1))
    word_id = int(re.search(rf"^\s*10\s+(\d+)\s+{re.escape(EXAMPLE_PATH)}\s*$", prompt, re.M).group(1))

    # r1's gate: "# units 32-47: T_d (read/list path at offset 10) = relu(codeA_d + hashed + is_o10 + bc_rl - 2) -> d96+d"
    gate_line = next(line for line in r3.splitlines() if line.startswith("# units 32-47"))
    formula = re.search(r"relu\((.*)\)", gate_line).group(1)
    terms = [t.strip() for t in re.split(r"\s[+-]\s", formula)]
    assert terms == ["codeA_d", "hashed", "is_o10", "bc_rl", "2"], terms
    r1_bias_statements = re.findall(r"^layers\.0\.mlp\.fc1\.bias\[(?:32:48|48:64)\] = -2$", r3, re.M)
    assert r1_bias_statements == ["layers.0.mlp.fc1.bias[32:48] = -2", "layers.0.mlp.fc1.bias[48:64] = -2"]
    fix = "layers.0.mlp.fc1.bias[32:64] = -3"
    assert f"`{fix}`" in iters and f"`{fix}`" in revision

    lit = code_a_cells(word_id, base)
    gates = 3  # at a read path (offset 10, read/list event, hashed word) all three gates are 1

    def key(bias: int) -> list[int]:
        return [max(0, (1 if d in lit else 0) + gates + bias) for d in range(16)]

    before, after = key(-2), key(-3)

    it0 = section(iters, "## Iteration 0", "## Iteration 1")
    it1 = section(iters, "## Iteration 1", "## Iteration 2")
    it2 = section(iters, "## Iteration 2", "## Iteration 3")
    cm0, cm2 = parse_matrix(it0), parse_matrix(it2)
    stated = {
        "f1_0": float(re.search(r"Macro-F1 ([0-9]+\.[0-9]+)", it0).group(1)),
        "fa_0": float(re.search(r"false-alarm rate ([0-9]+\.[0-9]+)", it0).group(1)),
        "f1_2": float(re.search(r"macro-F1 ([0-9]+\.[0-9]+)", it2).group(1)),
        "fa_2": float(re.search(r"false-alarm rate ([0-9]+\.[0-9]+)", it2).group(1)),
    }
    v001 = {
        "MATCHtok": float(re.search(r"`MATCHtok` \(d118\) ≈ ([0-9.]+)", it1).group(1)),
        "REPg": float(re.search(r"`REPg` ≈ ([0-9.]+)", it1).group(1)),
        "INEFF": float(re.search(r"`INEFF` ≈ ([0-9.]+)", it1).group(1)),
    }
    m = re.search(r"STALLED logits of (\d+)–(\d+) on (\d+) of (\d+) windows", revision)
    logits = {"lo": int(m.group(1)), "hi": int(m.group(2)), "windows": int(m.group(3)), "of": int(m.group(4))}

    obj = {
        "_provenance": {
            "source": [rel(p) for p in SOURCES],
            "sha256": {rel(p): sha256(p) for p in SOURCES},
            "source_commits": {rel(p): git("log", "-1", "--format=%h", "--", rel(p)) for p in SOURCES},
            "git_commit": git("rev-parse", "HEAD"),
            "generated_by": GENERATED_BY,
            "population": "the 200 disclosed validation windows V001-V200 (50 per class): an in-sample self-check, not an evaluation",
        },
        "example": {"path": EXAMPLE_PATH, "id": word_id, "lit_cells": lit},
        "gate": {
            "units": "layer 0 MLP units 32-63 (key at a read/list path, units 32-47, and at a search word, 48-63)",
            "r1_comment": gate_line.lstrip("# ").strip(),
            "inputs": ["code bit", "hashed word", "offset 10", "read/list event"],
            "gates_at_read_path": gates,
        },
        "before": {
            "label": "Round 1 (r1)",
            "bias": -2,
            "statements": r1_bias_statements,
            "key": before,
            "key_sum": sum(before),
            "cells_at_2": sum(1 for v in before if v == 2),
            "cells_at_1": sum(1 for v in before if v == 1),
            "match_by_construction": round(sum(before) / 3, 2),
            "v001": v001,
            "stalled_logits": logits,
            "confusion": cm0,
            "macro_f1": round(macro_f1(cm0), 3),
            "false_alarm": round(false_alarm(cm0), 2),
        },
        "after": {
            "label": "Round 2, first change",
            "bias": -3,
            "statements": [fix],
            "key": after,
            "key_sum": sum(after),
            "match_by_construction": round(sum(after) / 3, 2),
            "confusion": cm2,
            "macro_f1": round(macro_f1(cm2), 3),
            "false_alarm": round(false_alarm(cm2), 2),
        },
        "classes": list(CLASSES),
        "orientation": "rows = labeller (oracle) class, columns = network class",
    }

    # The figure's claims, checked against the built data.
    assert word_id == 6414 and lit == [2, 5, 14]
    assert sum(before) == 19 and obj["before"]["cells_at_2"] == 3 and obj["before"]["cells_at_1"] == 13
    assert sum(after) == 3 and sorted(set(after)) == [0, 1]
    assert abs(sum(before) / 3 - v001["MATCHtok"]) < 0.01  # the measured 6.33 is the broken key's sum / 3
    assert v001 == {"MATCHtok": 6.33, "REPg": 8.7, "INEFF": 10.0}
    assert logits == {"lo": 60, "hi": 90, "windows": 141, "of": 200}
    assert cm0 == [[3, 0, 47, 0], [4, 0, 46, 0], [14, 0, 36, 0], [0, 0, 12, 38]]
    assert cm2 == [[35, 14, 0, 1], [6, 42, 2, 0], [28, 4, 18, 0], [0, 0, 0, 50]]
    assert all(sum(r) == 50 for r in cm0 + cm2)
    assert round(macro_f1(cm0), 3) == stated["f1_0"] == 0.331 and round(macro_f1(cm2), 3) == stated["f1_2"] == 0.714
    assert round(false_alarm(cm0), 2) == stated["fa_0"] and round(false_alarm(cm2), 2) == stated["fa_2"]
    return obj


def main() -> None:
    obj = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{OUT.relative_to(ROOT)} {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    sys.exit(main())
