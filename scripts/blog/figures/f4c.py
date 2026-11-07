"""F4c (post Figure 5): "Closing the leaks", the v0.3 raw-text label leaks and the dev marker audit.

Run from the repo root:  uv run python scripts/blog/figures/f4c.py

Numbers come from the committed dev audit report (artifacts/reports/v03_dev_audits.json: marker_audit and
leakage_scan) and the leak list from spec v0.3 (E9 and the r2 data-audit row). The example block is invented:
the "after" lines are the v0.3 grammar (header from GRAMMAR_SHEET.md's edit_file example, re-labelled
write_file so the bytes= leak has a place), and the "before" diff is what a plain unified diff of one invented
code block would have shown, built with the generator's own code templates (synth/world.py), so the leak
markers have exactly the generator's format. No dataset, evaluator or scorer is opened here.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from agent_loop_lab.synth.world import _CODE_SYNTAX  # noqa: E402  (code templates only; no RNG, no data)

OUT = ROOT / "blog_figures" / "F4c.json"
GENERATED_BY = "scripts/blog/figures/f4c.py"

AUDITS = ROOT / "artifacts" / "reports" / "v03_dev_audits.json"
SPEC = ROOT / "agent_doom_loop_detector_spec_v0_3.md"
GRAMMAR = ROOT / "experiments" / "v0_3" / "packet_handed_over" / "GRAMMAR_SHEET.md"
WORLD = ROOT / "src" / "agent_loop_lab" / "synth" / "world.py"
SOURCES = [AUDITS, SPEC, GRAMMAR, WORLD]
SOURCE_COMMITS = {"spec": "e05f371", "audits": "d08f05f", "grammar_sheet": "e842bc3"}
DP = 4


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def need(pattern: str, text: str, what: str) -> str:
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"F4c: could not find {what}")
    return m.group(0)


def grammar_edit_block() -> list[str]:
    """The rendered text of GRAMMAR_SHEET.md's edit_file example."""
    text = GRAMMAR.read_text()
    m = re.search(r"## edit_file[^\n]*\n\n```text\n(.*?)\n```", text, re.S)
    if not m:
        raise SystemExit("F4c: edit_file example missing from the grammar sheet")
    return m.group(1).split("\n")


def code_block(behaviour: str) -> list[str]:
    """One python block as the generator renders it (World._render_code_block), with invented names."""
    syn = _CODE_SYNTAX["python"]
    p, local, helper, k, sp = "item", "acc", "clamp", 418, " "
    expr = f"{helper}({local},{sp}{k}{sp}+{sp}1)" if behaviour == "fix" else f"{helper}({local},{sp}{k})"
    out = [syn["open"].format(name="widget_total", p=p), syn["decl"].format(local=local, p=p, sp=sp)]
    if behaviour == "fix":
        out.extend(g.format(p=p, local=local) for g in syn["guard"])
    out.append(syn["stmt"].format(local=local, expr=expr, sp=sp))
    out.append(syn["ret"].format(local=local))
    return out


def build() -> dict:
    audits = json.loads(AUDITS.read_text())
    if audits["split"] != "dev":
        raise SystemExit("F4c: the audit report must be the dev split")
    ma = audits["marker_audit"]
    scan = audits["leakage_scan"]
    spec = SPEC.read_text()

    # The three leaks, as the spec words them.
    e9 = need(r"\| E9 \| Diff rendering \|[^\n]*", spec, "E9 row")
    r2 = need(r"\| r2 \(data audit, dev\): `bytes=`[^\n]*", spec, "r2 data-audit row")
    for marker in ("`+1` and a guard line for fixes", "`1000+n` constants for wrong variants",
                   "`cleanup pass N` for cosmetic edits", "`-ok` for data fixes"):
        if marker not in e9:
            raise SystemExit(f"F4c: E9 no longer names {marker}")

    # "Before": a unified diff (n=0) of one invented block, original -> fix, in the generator's format.
    orig, fix = code_block("orig"), code_block("fix")
    diff = list(difflib.unified_diff(orig, fix, lineterm="", n=0))[2:]
    hunks = sum(1 for ln in diff if ln.startswith("@@"))
    body = [ln for ln in diff if not ln.startswith("@@")]

    after = grammar_edit_block()
    header = after[0].replace(" edit_file ", " write_file ")
    hunk_line = after[1]
    path = re.search(r"path=(\S+)", header).group(1)
    before_header = header.replace(f"path={path} ", f"path={path} bytes=1284 ")

    tools = ma["counts_by_tool_and_hunk_lines"]
    revert_events, revert_pos = tools["revert"]["1"]
    hunk_line_values = sorted({k for t in ("edit_file", "write_file", "revert") for k in tools[t]})

    def r(x: float) -> float:
        return round(float(x), DP)

    return {
        "_provenance": {
            "source": [rel(p) for p in SOURCES],
            "sha256": {rel(p): sha256(p) for p in SOURCES},
            "source_commits": SOURCE_COMMITS,
            "git_commit": git_head(),
            "generated_by": GENERATED_BY,
        },
        "figure": "F4c",
        "title": "Closing the leaks",
        "example": {
            "note": "invented block; 'after' follows the v0.3 grammar (GRAMMAR_SHEET.md edit_file example, tool "
                    "re-labelled write_file), 'before' is a plain unified diff built with the generator's python "
                    "code templates",
            "before": {
                "header": before_header,
                "bytes_field": "bytes=1284",
                "hunk_line": f"  @ {path} ({hunks} hunk{'s' if hunks != 1 else ''})",
                "diff": body,
            },
            "after": {"header": header, "hunk_line": hunk_line},
            "other_markers": [
                {"kind": "wrong variant", "text": "    acc = clamp(acc, 1007)", "mark": "1007"},
                {"kind": "cosmetic edit", "text": "    # cleanup pass 2", "mark": "cleanup pass 2"},
                {"kind": "data fix", "text": 'value = "v412-ok"', "mark": "-ok"},
            ],
        },
        "leaks": [
            {"n": 1, "what": "Code text in the diff", "detail": "a fix adds `+ 1` and a guard line, a wrong variant "
             "a `1000+n` constant, a cosmetic edit `cleanup pass N`, a data fix `-ok`",
             "fix": "one pair of opaque content ids per changed file", "where": "spec v0.3 E9"},
            {"n": 2, "what": "`bytes=` on write_file", "detail": "the byte size separated progress from "
             "non-progress edits", "fix": "removed", "where": "spec v0.3 r2 data audit (dev)"},
            {"n": 3, "what": "Hunk count per file", "detail": "the number of hunks separated progress from "
             "non-progress edits", "fix": "removed: one line per changed file", "where": "spec v0.3 r2 data audit (dev)"},
        ],
        "spec_rows": {"e9": e9, "r2": r2},
        "probe": {
            "split": "dev",
            "n": ma["n"],
            "positive_rate": r(ma["positive_rate"]),
            "label": ma["label"],
            "features": ma["features"],
            "folds": ma["folds"],
            "macro_f1": r(ma["macro_f1_threshold_0_5"]),
            "macro_f1_prior": r(ma["macro_f1_constant_prior"]),
            "macro_f1_base_rate_threshold": r(ma["macro_f1_threshold_base_rate"]),
            "auc": r(ma["auc"]),
            "auc_chance": 0.5,
            "auc_without_tool": r(ma["ablation_without_tool"]["auc"]),
            "macro_f1_without_tool": r(ma["ablation_without_tool"]["macro_f1_threshold_0_5"]),
            "macro_f1_without_tool_base_rate_threshold": r(ma["ablation_without_tool"]["macro_f1_threshold_base_rate"]),
            "auc_tool_only": r(ma["ablation_tool_only"]["auc"]),
            "hunk_lines_values": hunk_line_values,
            "by_tool": {t: {"events": tools[t]["1"][0], "progress": tools[t]["1"][1]} for t in ("edit_file", "write_file", "revert")},
            "revert_events": revert_events,
            "revert_progress": revert_pos,
        },
        "leakage_scan": {"split": "dev", "sessions": audits["sessions"], "events": audits["events"],
                         "markers": scan["markers"], "marker_hits": sum(scan["marker_counts"].values()),
                         "file_text_lines_found": scan["file_text_lines_found_in_blocks"], "clean": scan["clean"]},
    }


if __name__ == "__main__":
    data = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    p = data["probe"]
    print(f"wrote {rel(OUT)}: probe macro-F1 {p['macro_f1']} vs prior {p['macro_f1_prior']}, AUC {p['auc']} (n={p['n']})")
