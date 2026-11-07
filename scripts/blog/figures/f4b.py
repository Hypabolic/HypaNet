"""F4b (post Figure 6): "Anatomy of a block", three grammar-sheet blocks as token strips with layer-0 marks.

Run from the repo root:  uv run python scripts/blog/figures/f4b.py

Reads the invented example blocks (edit_file, revert, and the failed run_tests) and their token tables from
the v0.3 grammar sheet that was handed to Fable, and the head definitions from the zero-label network's
CIRCUITS.md (v0.3 Phase A, sections 1 and 5). Which token each head marks is worked out here from the
positional rules CIRCUITS.md states (tool at offset 7; old id 2 and new id 4 tokens after each NL@; last
status word; passed count 3 tokens before the last `passed`; an h_new equal to an earlier h_old). No model
is run, and no dataset, evaluator or scorer is opened. No benchmark data is involved.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F4b.json"
GENERATED_BY = "scripts/blog/figures/f4b.py"

GRAMMAR = ROOT / "experiments" / "v0_3" / "packet_handed_over" / "GRAMMAR_SHEET.md"
CIRCUITS = ROOT / "weight_program" / "v0_3" / "phase-a" / "CIRCUITS.md"
SOURCES = [GRAMMAR, CIRCUITS]
SOURCE_COMMITS = {"grammar_sheet": "e842bc3", "circuits_phase_a": "eb38b33"}

# Grammar-sheet sections, in the order the figure shows them.
BLOCKS = [
    ("edit_file", "## edit_file"),
    ("revert", "## revert (undoes"),
    ("run_tests", "## run_tests (failed"),
]
STATUS_WORDS = {"ok", "ready", "error", "pending", "FAILED"}

# Plain-language roles; the formal rows are copied from CIRCUITS.md below.
HEADS = {
    "H0.2": {"role": "tool", "text": "copies the tool name at offset 7 to every token of its block"},
    "H0.0": {"role": "old id", "text": "marks the old content id, the token 2 after @ (HOLD)"},
    "H0.1": {"role": "new id", "text": "marks the new content id, the token 4 after @ (HNEW)"},
    "H0.3": {"role": "status", "text": "reads the block's last status word"},
    "H0.4": {"role": "count", "text": "finds the passed count, 3 tokens before the last `passed`"},
    "H1.0": {"role": "undo", "text": "layer 1: a new id equal to an earlier old id is an undo (REVMATCH)"},
}


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def parse_block(sheet: str, heading: str) -> dict:
    i = sheet.index(heading)
    title = sheet[i:].split("\n", 1)[0].lstrip("# ").strip()
    m = re.search(r"```text\n(.*?)\n```\s*```tokens\n(.*?)\n```", sheet[i:], re.S)
    if not m:
        raise SystemExit(f"F4b: block '{heading}' not found")
    tokens = []
    for line in m.group(2).split("\n"):
        off, tid, surface = line.split(None, 2)
        tokens.append({"o": int(off), "id": int(tid), "s": surface})
    if [t["o"] for t in tokens] != list(range(len(tokens))):
        raise SystemExit(f"F4b: offsets of '{heading}' are not 0..n-1")
    return {"title": title, "text": m.group(1).split("\n"), "tokens": tokens}


def mark(block: dict) -> None:
    toks = block["tokens"]
    marks: dict[int, list[str]] = {}
    marks.setdefault(7, []).append("H0.2")  # the tool name sits at offset 7 (the timestamp fills 0-6)
    for t in toks:
        if t["s"] == "NL@":
            marks.setdefault(t["o"] + 2, []).append("H0.0")
            marks.setdefault(t["o"] + 4, []).append("H0.1")
    status = [t["o"] for t in toks if t["s"] in STATUS_WORDS]
    if status:
        marks.setdefault(status[-1], []).append("H0.3")
    passed = [t["o"] for t in toks if t["s"] == "passed"]
    if passed:
        marks.setdefault(passed[-1] - 3, []).append("H0.4")
    for t in toks:
        t["m"] = marks.get(t["o"], [])
    block["anchors"] = [t["o"] for t in toks if t["s"] == "NL@"]


def build() -> dict:
    sheet = GRAMMAR.read_text()
    circuits = CIRCUITS.read_text()
    blocks = []
    for tool, heading in BLOCKS:
        b = parse_block(sheet, heading)
        if b["tokens"][7]["s"] != tool:
            raise SystemExit(f"F4b: offset 7 of '{heading}' is not {tool}")
        mark(b)
        b["tool"] = tool
        blocks.append(b)

    # Layer 1: the revert's h_new equals the edit's h_old (same token id) -> REVMATCH.
    edit, revert = blocks[0], blocks[1]
    old_ids = {t["id"]: t["o"] for t in edit["tokens"] if "H0.0" in t["m"]}
    links = []
    for t in revert["tokens"]:
        if "H0.1" in t["m"] and t["id"] in old_ids:
            links.append({"from_block": 1, "from_offset": t["o"], "to_block": 0, "to_offset": old_ids[t["id"]],
                          "id": t["id"], "surface": t["s"]})
    if len(links) != 1:
        raise SystemExit(f"F4b: expected one revert match, found {len(links)}")

    rows = {}
    for h in ("H0.0", "H0.1", "H0.2", "H0.3", "H0.4", "H1.0"):
        m = re.search(rf"^\| {re.escape(h)} \|[^\n]*", circuits, re.M)
        if not m:
            raise SystemExit(f"F4b: {h} missing from CIRCUITS.md")
        rows[h] = m.group(0)
    tool_rule = re.search(r"\| `tool` \| ([^|]+) \|", circuits).group(1).strip()
    hold_rule = re.search(r"\| `h_old`, `h_new` \| ([^|]+) \|", circuits).group(1).strip()

    return {
        "_provenance": {
            "source": [rel(p) for p in SOURCES],
            "sha256": {rel(p): sha256(p) for p in SOURCES},
            "source_commits": SOURCE_COMMITS,
            "git_commit": git_head(),
            "generated_by": GENERATED_BY,
            "note": "invented grammar-sheet blocks, not benchmark data; marks follow the positional rules of CIRCUITS.md",
        },
        "figure": "F4b",
        "title": "Anatomy of a block",
        "network": "v0.3 Phase A (zero-label)",
        "blocks": blocks,
        "links": links,
        "heads": {h: {**HEADS[h], "circuits_row": rows[h]} for h in HEADS},
        "rules": {"tool": tool_rule, "h_old_h_new": hold_rule},
    }


if __name__ == "__main__":
    data = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {rel(OUT)}: {[len(b['tokens']) for b in data['blocks']]} tokens, link {data['links'][0]}")
