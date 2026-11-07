"""F1 (post Figure 2): same rhythm, opposite verdicts. Three dev sessions as event timelines.

Run from the repo root:  uv run python scripts/blog/figures/f1.py

Reads three sessions from the DEV split only: the raw ops (datasets/v0_3/dev.raw.jsonl.gz) rendered with the
frozen v0.3 renderer, and the oracle records (datasets/v0_2/standard/dev.oracle.jsonl.gz) for each event's hidden
effect and each window's label. Both files are gitignored; they rebuild byte-identically from the manifests, and
this script refuses to run unless their sha256 equals the hash the committed manifest pins. Opens no other split,
runs no model, evaluator or scorer, and regenerates nothing (no seed is used). Writes blog_figures/F1.json.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from agent_loop_lab.v03.renderer import render_session

OUT = ROOT / "blog_figures" / "F1.json"
GENERATED_BY = "scripts/blog/figures/f1.py"

DEV_RAW = ROOT / "datasets/v0_3/dev.raw.jsonl.gz"
DEV_ORACLE = ROOT / "datasets/v0_2/standard/dev.oracle.jsonl.gz"
MANIFEST_V03 = ROOT / "datasets/v0_3/MANIFEST_v0_3.json"
MANIFEST_V02 = ROOT / "datasets/v0_2/MANIFEST.json"
RENDERER = ROOT / "src/agent_loop_lab/v03/renderer.py"

# The three sessions the post names (row order in the figure).
SESSIONS = [
    ("s-8c4e2bce5b5ce097", "loop", "Flips webhook.go between two versions"),
    ("s-35f046b90b6b6352", "healthy", "Fixes something on each pass"),
    ("s-b85d987d455fc994", "retry", "Retries the same failing suite"),
]
DEFAULT_ENDPOINT = 13  # window ending at index 13: rows 1 and 2 match operation for operation
WINDOW = 8

FAMILY_GLYPH = {
    "edit_file": "edit", "write_file": "edit", "revert": "edit",
    "run_tests": "test", "build": "test", "lint": "test",
    "read_file": "read", "search": "search", "list_dir": "list", "plan": "plan",
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def check_hashes() -> dict:
    m3 = json.loads(MANIFEST_V03.read_text())
    canonical = m3["acceptance"]["canonical_files"]
    pinned_raw = m3["files"]["dev.raw.jsonl.gz"]["sha256"]
    pinned_oracle = canonical["standard/dev.oracle.jsonl.gz"]["v0_2_sha256"]
    got_raw, got_oracle = sha256(DEV_RAW), sha256(DEV_ORACLE)
    if got_raw != pinned_raw or got_oracle != pinned_oracle:
        raise SystemExit("F1: dev data does not match the committed manifest hashes; rebuild it first")
    # The v0.2 manifest pins the oracle file too; both must agree.
    if pinned_oracle not in MANIFEST_V02.read_text():
        raise SystemExit("F1: v0.2 manifest does not pin the dev oracle hash")
    return {rel(DEV_RAW): got_raw, rel(DEV_ORACLE): got_oracle}


def read_sessions() -> tuple[dict, dict]:
    ids = {s[0] for s in SESSIONS}
    raw, oracle = {}, {}
    with gzip.open(DEV_RAW, "rt") as f:
        for line in f:
            tail = line[-40:]
            if any(i in tail for i in ids):
                d = json.loads(line)
                if d["session_id"] in ids:
                    raw[d["session_id"]] = d["raw"]
    with gzip.open(DEV_ORACLE, "rt") as f:
        for line in f:
            if any(i in line for i in ids):
                d = json.loads(line)
                if d["session_id"] in ids:
                    oracle[d["session_id"]] = d
    if set(raw) != ids or set(oracle) != ids:
        raise SystemExit("F1: a session is missing from the dev files")
    return raw, oracle


def op_entry(i: int, op: dict, block: str, effect: str) -> dict:
    header, *body = block.split("\n")
    status = re.search(r" -> (\S+)", header)
    hunks = [re.search(r"(h[0-9a-f]{10}) > (h[0-9a-f]{10})", b) for b in body]
    hunk = next((h for h in hunks if h), None)
    val = op.get("validation")
    return {
        "i": i,
        "family": op["family"],
        "glyph": FAMILY_GLYPH.get(op["family"], "other"),
        "target": op["target"] if isinstance(op["target"], str) else None,
        "status": status.group(1) if status else None,
        "failed": op["outcome"] == "ERROR" or (status is not None and status.group(1) in ("FAILED", "error")),
        "passed": val["passed"] if val else None,
        "total": val["total"] if val else None,
        "hash_before": hunk.group(1) if hunk else None,
        "hash_after": hunk.group(2) if hunk else None,
        "effect": effect,
        "text": block,
    }


def label_runs(labels: dict[str, str]) -> list[dict]:
    """Consecutive window endpoints with the same label, as [start, end] runs."""
    items = sorted((int(k), v) for k, v in labels.items())
    runs: list[dict] = []
    for t, lab in items:
        if runs and runs[-1]["label"] == lab and runs[-1]["end"] == t - 1:
            runs[-1]["end"] = t
        else:
            runs.append({"label": lab, "start": t, "end": t})
    return runs


def build() -> dict:
    hashes = check_hashes()
    raw, oracle = read_sessions()
    rows = []
    for sid, role, title in SESSIONS:
        ops = raw[sid]
        o = oracle[sid]
        blocks = render_session(ops, sid)
        effects = o["event_effects"]
        if not (len(ops) == len(blocks) == len(effects)):
            raise SystemExit(f"F1: length mismatch in {sid}")
        labels = {str(k): v for k, v in o["window_labels"].items()}
        rows.append({
            "session_id": sid,
            "role": role,
            "title": title,
            "family": o["scenario_family"],
            "ops": [op_entry(i, op, b, e) for i, (op, b, e) in enumerate(zip(ops, blocks, effects))],
            "window_labels": labels,
            "label_runs": label_runs(labels),
        })

    # Check the claim the figure pins: rows 1 and 2 match glyph for glyph and outcome for outcome in the
    # default window, while their labels differ.
    lo, hi = DEFAULT_ENDPOINT - WINDOW + 1, DEFAULT_ENDPOINT
    sig = [[(op["glyph"], op["failed"]) for op in row["ops"][lo:hi + 1]] for row in rows[:2]]
    if sig[0] != sig[1]:
        raise SystemExit("F1: rows 1 and 2 do not match in the default window")
    lab = [row["window_labels"].get(str(DEFAULT_ENDPOINT)) for row in rows]

    return {
        "_provenance": {
            "source": [rel(MANIFEST_V02), rel(MANIFEST_V03), rel(RENDERER), rel(DEV_RAW), rel(DEV_ORACLE)],
            "sha256": {rel(MANIFEST_V02): sha256(MANIFEST_V02), rel(MANIFEST_V03): sha256(MANIFEST_V03),
                       rel(RENDERER): sha256(RENDERER), **hashes},
            "source_commits": {rel(p): git("log", "-1", "--format=%h", "--", rel(p))
                               for p in (MANIFEST_V02, MANIFEST_V03, RENDERER)},
            "note": "dev data files are gitignored; they rebuild byte-identically and are pinned by the manifest hashes",
            "git_commit": git("rev-parse", "HEAD"),
            "generated_by": GENERATED_BY,
        },
        "figure": "F1",
        "split": "dev (standard simulator); synthetic",
        "window": WINDOW,
        "default_endpoint": DEFAULT_ENDPOINT,
        "default_labels": lab,
        "rows": rows,
    }


def main() -> None:
    data = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {rel(OUT)}")


if __name__ == "__main__":
    main()
