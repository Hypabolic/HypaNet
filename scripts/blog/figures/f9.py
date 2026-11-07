"""F9 (post Figure 13): "Turning a language model into a classifier", the frozen prompt and six example windows.

Run from the repo root:  uv run python scripts/blog/figures/f9.py

Reads the committed val quick-look readout columns (four label-word logits per window for base Qwen, WSP-A and
WSP-B), the frozen interface (envs/qwen/hypaqwen/interface.py) and the prompt-choice report (prefix and suffix token
counts), plus the local val task export (block text) and val labels (scoring mask and the true class shown beside
each example). Val only: no sealed, OOD, held-out or fresh_test data is opened, and no evaluator or scorer runs.

Selection rule (fixed before looking at any example, stated in the JSON):
  key(window) = sha256("hypanet-blog-f9:{session_id}:{window_end_block}")
  candidates  = the scored windows of the 1,000 val quick-look sessions (the benchmark's scoring mask)
  slots 1-4   = for each class in PRODUCTIVE, RECOVERY, STALLED, OSCILLATING order, the smallest-key candidate
                that base Qwen answers with that class
  slots 5-6   = the two smallest-key remaining candidates where WSP-B's answer differs from WSP-A's
The rule reads predictions and hashes only; the true class is looked up after selection, for display.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F9.json"
GENERATED_BY = "scripts/blog/figures/f9.py"

COLS = ROOT / "experiments" / "v0_4" / "columns"
NPZ = {"qwen0": COLS / "qwen0.val_quick.npz", "wspa": COLS / "wspa.val_quick.npz", "wspb": COLS / "wspb.val_quick.npz"}
MODEL_NAMES = {"qwen0": "Base Qwen", "wspa": "WSP-A", "wspb": "WSP-B"}
INTERFACE = ROOT / "envs" / "qwen" / "hypaqwen" / "interface.py"
PROMPT_CHOICE = ROOT / "artifacts" / "reports" / "v04_prompt_choice.json"
QUICK = ROOT / "datasets" / "v0_4" / "VAL_QUICK.txt"
TASKS = ROOT / "datasets" / "v0_4" / "val.tasks.jsonl.gz"  # local, gitignored
LABELS = ROOT / "datasets" / "v0_4" / "val.labels.npz"  # local, gitignored
COMMITS = {"interface": "a616b93", "columns": "48acfc6"}

CLASSES = ("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING")
LABEL_WORDS = (" productive", " recovering", " stalled", " looping")
KEY_FMT = "hypanet-blog-f9:{session_id}:{pos}"
SHOW_BLOCKS = 8
DP = 4
RULE = (
    "key = sha256('hypanet-blog-f9:{session_id}:{window_end_block}'); candidates = scored windows of the 1,000 val "
    "quick-look sessions; slots 1-4: for each class (PRODUCTIVE, RECOVERY, STALLED, OSCILLATING) the smallest-key "
    "candidate that base Qwen answers with that class; slots 5-6: the two smallest-key remaining candidates where "
    "WSP-B's answer differs from WSP-A's. The rule reads predictions and hashes only; the true class is looked up "
    "after selection, for display."
)


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def interface_text() -> dict:
    """The frozen prompt's words, read from interface.py without importing the Qwen environment."""
    ns: dict = {}
    src = INTERFACE.read_text()
    exec(compile(src.replace("from __future__ import annotations\n", ""), str(INTERFACE), "exec"), ns)  # plain dataclasses only
    frozen = ns["FROZEN"]
    if tuple(frozen.label_words) != LABEL_WORDS or tuple(ns["CLASS_NAMES"]) != CLASSES:
        raise SystemExit("F9: interface label words or class order changed")
    return {
        "name": frozen.name,
        "instruction": frozen.instruction(),
        "prefix": frozen.prefix(),
        "suffix": frozen.suffix(),
        "answer_cue": frozen.answer_cue,
        "block_template": ns["BLOCK_TEMPLATE"],
    }


def build() -> dict:
    cols = {k: np.load(p) for k, p in NPZ.items()}
    sids = [str(s) for s in cols["qwen0"]["session_ids"]]
    pos = cols["qwen0"]["row_pos"]
    for k, z in cols.items():
        if [str(s) for s in z["session_ids"]] != sids or not np.array_equal(z["row_pos"], pos):
            raise SystemExit(f"F9: {k} rows do not line up with qwen0")
    quick = [s.strip() for s in QUICK.read_text().split() if s.strip()]
    if sorted(quick) != sorted(sids):
        raise SystemExit("F9: npz sessions are not the VAL_QUICK.txt sessions")
    logits = {k: z["label_logits"].astype(np.float64) for k, z in cols.items()}
    lse = {k: z["logsumexp"].astype(np.float64) for k, z in cols.items()}
    pred = {k: v.argmax(1) for k, v in logits.items()}

    # Row -> session: rows are in session order, one per window_pos of that session's task.
    want = set(sids)
    tasks = {}
    with gzip.open(TASKS, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["session_id"] in want:
                tasks[d["session_id"]] = d
    row_sid, row_pos = [], []
    for s in sids:
        for p in tasks[s]["window_pos"]:
            row_sid.append(s)
            row_pos.append(p)
    if len(row_sid) != len(pos) or not np.array_equal(np.array(row_pos), pos):
        raise SystemExit("F9: task window_pos does not match the readout rows")

    with np.load(LABELS) as z:
        lab = {k: z[k] for k in z.files}
    lab_sids = [str(s) for s in lab["session_ids"]]
    label_of = {}
    for si, p, l, g in zip(lab["row_session"].tolist(), lab["row_pos"].tolist(), lab["row_label"].tolist(),
                           lab["row_gate"].tolist()):
        s = lab_sids[si]
        if s in want:
            label_of[(s, p)] = (int(l), bool(g))
    scored = np.array([label_of[(s, int(p))][1] and label_of[(s, int(p))][0] < len(CLASSES)
                       for s, p in zip(row_sid, row_pos)])

    keys = [hashlib.sha256(KEY_FMT.format(session_id=s, pos=int(p)).encode()).hexdigest()
            for s, p in zip(row_sid, row_pos)]
    order = sorted(np.flatnonzero(scored).tolist(), key=lambda i: keys[i])

    picks: list[tuple[int, str]] = []
    for c, name in enumerate(CLASSES):
        i = next(i for i in order if pred["qwen0"][i] == c)
        picks.append((i, f"Base Qwen answers {LABEL_WORDS[c].strip()}"))
    taken = {i for i, _ in picks}
    flips = [i for i in order if i not in taken and pred["wspa"][i] != pred["wspb"][i]][:2]
    picks += [(i, "WSP-B's answer differs from WSP-A's") for i in flips]

    windows = []
    for slot, (i, why) in enumerate(picks, start=1):
        s, p = row_sid[i], int(row_pos[i])
        blocks = tasks[s]["blocks"]
        true = label_of[(s, p)][0]
        first = max(0, p - SHOW_BLOCKS + 1)
        windows.append({
            "slot": slot,
            "why": why,
            "session_id": s,
            "end_block": p,
            "blocks_in_window": p + 1,
            "session_blocks": len(blocks),
            "key_sha256": keys[i],
            "true_class": CLASSES[true],
            "shown_from_block": first,
            "last_blocks": blocks[first:p + 1],
            "models": {
                k: {
                    "logits": [round(float(v), DP) for v in logits[k][i]],
                    "logsumexp_all": round(float(lse[k][i]), DP),
                    "answer": CLASSES[int(pred[k][i])],
                }
                for k in NPZ
            },
        })

    pc = json.loads(PROMPT_CHOICE.read_text())
    chosen = next(c for c in pc["candidates"] if c["name"] == "w1-state-full")
    iface = interface_text()
    if iface["name"] != chosen["name"]:
        raise SystemExit("F9: frozen interface is not w1-state-full")
    meta = json.loads(str(cols["qwen0"]["meta"]))

    committed = [INTERFACE, PROMPT_CHOICE, QUICK, *NPZ.values()]
    local = [TASKS, LABELS]
    return {
        "_provenance": {
            "source": [rel(INTERFACE), rel(COLS) + "/"],
            "files": [rel(p) for p in committed],
            "sha256": {rel(p): sha256(p) for p in committed + local},
            "local_sources": [rel(p) for p in local],
            "source_commits": COMMITS,
            "git_commit": git_head(),
            "generated_by": GENERATED_BY,
            "selection_rule": RULE,
            "notes": [
                "split: val, quick look (the 1,000 sessions in VAL_QUICK.txt, fixed by hash in advance)",
                "logits: the next-token logits of the four label words after the suffix, as read out on the Mac's "
                "GPU (MPS, fp32); answer = argmax of the four",
                "true_class: the oracle label of the window, looked up after selection (display only)",
                "last_blocks: the last 8 blocks of the window (the model reads every block from the session start)",
            ],
        },
        "figure": "F9",
        "split": "val, quick look",
        "sessions": meta["sessions"],
        "rows": int(len(row_sid)),
        "scored_windows": int(scored.sum()),
        "classes": list(CLASSES),
        "label_words": list(LABEL_WORDS),
        "models": [{"id": k, "name": MODEL_NAMES[k]} for k in NPZ],
        "interface": {
            **iface,
            "prefix_tokens": chosen["prefix_tokens"],
            "suffix_tokens": chosen["suffix_tokens"],
            "vocab_size": 248320,
            "model": pc["model"]["id"],
        },
        "windows": windows,
    }


def main() -> None:
    data = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {rel(OUT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
