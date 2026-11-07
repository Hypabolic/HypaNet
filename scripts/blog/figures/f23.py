"""F23 (post Figure 19): "How a network with no hash function spots a repeat", r1's residue codes.

Run from the repo root:  uv run python scripts/blog/figures/f23.py

Reads only committed text: r1's replies 001.md (the 48 strided token-embedding fills that write the residue
codes) and 003.md (the layer-1 head that matches a path's code A against earlier read-path keys: query weight,
start-token sink), the v0.5 prompt (QK_SCALE, HASH_BASE / HASH_BUCKETS and the two grammar-sheet ids, which are
example tokens in the prompt, not benchmark data) and the Phase B REVISION.md (the ~1/140 chance-match note).
Every number in the figure is either parsed from those files or computed from them here. No dataset, evaluator or
scorer is opened.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from math import prod
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F23.json"
GENERATED_BY = "scripts/blog/figures/f23.py"

REPLY_1 = ROOT / "experiments" / "v0_5" / "runs" / "r1" / "replies" / "001.md"
REPLY_3 = ROOT / "experiments" / "v0_5" / "runs" / "r1" / "replies" / "003.md"
PROMPT = ROOT / "experiments" / "v0_5" / "PROMPT.txt"
REVISION = ROOT / "weight_program" / "v0_5" / "phase-b" / "REVISION.md"
SOURCES = [REPLY_1, REPLY_3, PROMPT, REVISION]

# The two example word ids, read from the prompt's grammar sheet (token 10 of the read_file and write_file lines).
EXAMPLE_PATHS = ("zork/zeta/widget.qx", "zork/zeta/gizmo_plinth.qx")

FILL = re.compile(r"^tok_emb\.weight\[(\d+):(\d+):(\d+), (\d+)\] = 1\s*$")


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def strided_fills() -> list[dict]:
    """Every `tok_emb.weight[a:b:m, d] = 1` statement in r1's first reply, in order."""
    out = []
    for line in REPLY_1.read_text().splitlines():
        m = FILL.match(line)
        if m:
            start, stop, step, dim = map(int, m.groups())
            out.append({"start": start, "stop": stop, "step": step, "dim": dim, "text": line.strip()})
    return out


def prompt_constants() -> dict:
    text = PROMPT.read_text()
    base = int(re.search(r"^HASH_BASE = (\d+)", text, re.M).group(1))
    buckets = int(re.search(r"^HASH_BUCKETS = (\d+)", text, re.M).group(1))
    qk = float(re.search(r"^QK_SCALE = ([0-9.]+)", text, re.M).group(1))
    ids = {}
    for path in EXAMPLE_PATHS:
        m = re.search(rf"^\s*10\s+(\d+)\s+{re.escape(path)}\s*$", text, re.M)
        assert m, f"grammar-sheet id for {path} not found"
        ids[path] = int(m.group(1))
    return {"hash_base": base, "hash_buckets": buckets, "qk_scale": qk, "ids": ids}


def match_head() -> dict:
    """Layer 1 head 2 of r1: query weight on code A, key dims, start-token (BOS) sink."""
    text = REPLY_3.read_text()
    block = text[text.index("# head 2: target/count match on code A"):text.index("# head 3:")]
    q = {int(m.group(2)): float(m.group(3)) for m in re.finditer(r"layers\.1\.attn\.q\.weight\[(\d+), (\d+)\] = ([0-9.]+)", block)}
    sink = float(re.search(r"layers\.1\.attn\.rel_bias\[2, 64\] = (-?[0-9.]+)", block).group(1))
    assert sorted(q) == list(range(32, 48)) and len(set(q.values())) == 1, "head 2 query is not one weight on d32-47"
    return {"query_weight": q[32], "sink": sink, "header": block.splitlines()[0].lstrip("# ").strip()}


def build() -> dict:
    fills = strided_fills()
    consts = prompt_constants()
    base, buckets = consts["hash_base"], consts["hash_buckets"]
    stop = base + buckets
    word_fills = [f for f in fills if f["start"] >= base and f["stop"] == stop]
    digit_fills = [f for f in fills if f["start"] < base]

    # Code A = mod 4, 5, 7 on d32-47; code B = mod 3, 13 on d48-63. Read the moduli and dims from the fills.
    def segments(dims: range) -> list[dict]:
        segs: dict[int, list[dict]] = {}
        for f in word_fills:
            if f["dim"] in dims:
                segs.setdefault(f["step"], []).append(f)
        out = []
        for step, fs in segs.items():
            fs = sorted(fs, key=lambda f: f["dim"])
            # each fill covers ids whose offset from HASH_BASE has residue (start - base) mod step
            assert [f["start"] - base for f in fs] == list(range(step)), f"mod {step} fills are not residues 0..{step - 1}"
            out.append({"mod": step, "dims": [f["dim"] for f in fs], "first_dim": fs[0]["dim"], "last_dim": fs[-1]["dim"]})
        return sorted(out, key=lambda s: s["first_dim"])

    code_a = segments(range(32, 48))
    code_b = segments(range(48, 64))
    assert [s["mod"] for s in code_a] == [4, 5, 7] and [s["mod"] for s in code_b] == [3, 13]
    assert sum(len(s["dims"]) for s in code_a) == 16 and sum(len(s["dims"]) for s in code_b) == 16

    def code_of(word_id: int) -> dict:
        offset = word_id - base
        cells = []
        for code, segs in (("A", code_a), ("B", code_b)):
            for s in segs:
                r = offset % s["mod"]
                dim = s["dims"][r]
                stmt = next(f for f in word_fills if f["dim"] == dim)
                assert (word_id - stmt["start"]) % stmt["step"] == 0 and stmt["start"] <= word_id < stmt["stop"]
                cells.append({"code": code, "mod": s["mod"], "residue": r, "dim": dim, "statement": stmt["text"]})
        return {"offset": offset, "cells": cells}

    ids = []
    for path, wid in consts["ids"].items():
        ids.append({"path": path, "id": wid, **code_of(wid)})

    head = match_head()
    per_cell = head["query_weight"] * consts["qk_scale"]  # one matching cell of a one-hot key adds this to the score

    def dims_a(entry: dict) -> set[int]:
        return {c["dim"] for c in entry["cells"] if c["code"] == "A"}

    matches = []
    for q in ids:
        for k in ids:
            shared = sorted(dims_a(q) & dims_a(k))
            score = per_cell * len(shared)
            matches.append({"query": q["id"], "key": k["id"], "shared_dims": shared, "cells": len(shared),
                            "score": score, "beats_sink": score > head["sink"]})

    n_a = prod(s["mod"] for s in code_a)
    n_b = prod(s["mod"] for s in code_b)
    per_code = [0] * n_a
    for off in range(buckets):
        per_code[off % n_a] += 1  # CRT: offset mod 140 identifies the code-A value
    revision_note = "a false repeat with code A alone is ~1/140 per key"
    assert revision_note in REVISION.read_text()

    obj = {
        "_provenance": {
            "source": [rel(p) for p in SOURCES],
            "sha256": {rel(p): sha256(p) for p in SOURCES},
            "source_commits": {rel(p): git("log", "-1", "--format=%h", "--", rel(p)) for p in SOURCES},
            "git_commit": git("rev-parse", "HEAD"),
            "generated_by": GENERATED_BY,
            "note": "Word ids are example tokens from the prompt's grammar sheet, not benchmark data.",
        },
        "hash_base": base,
        "hash_buckets": buckets,
        "fills": {"total": len(fills), "word_ids": len(word_fills), "digit_groups": len(digit_fills),
                  "example": word_fills[2]["text"]},
        "word_fills": [{"dim": f["dim"], "mod": f["step"], "residue": f["start"] - base, "text": f["text"]} for f in word_fills],
        "code_a": code_a,
        "code_b": code_b,
        "ids": ids,
        "match": {
            "head": "layer 1, head 2",
            "head_comment": head["header"],
            "query_weight": head["query_weight"],
            "qk_scale": consts["qk_scale"],
            "per_cell": per_cell,
            "sink": head["sink"],
            "pairs": matches,
        },
        "collisions": {
            "code_a_values": n_a,
            "code_b_values": n_b,
            "combined_values": n_a * n_b,
            "ids_per_code_a": [min(per_code), max(per_code)],
            "ids_per_code_a_mean": round(buckets / n_a, 2),
            "revision_note": revision_note,
        },
    }

    # The figure's claims, checked against the built data.
    assert obj["fills"]["total"] == 48 and obj["fills"]["word_ids"] == 32 and obj["fills"]["digit_groups"] == 16
    assert obj["fills"]["example"] == "tok_emb.weight[2050:10240:4, 34] = 1"
    assert [i["id"] for i in ids] == [6414, 7212]
    assert [c["dim"] for c in ids[0]["cells"]] == [34, 37, 46, 49, 62]
    assert [c["dim"] for c in ids[1]["cells"]] == [32, 40, 46, 49, 54]
    assert [c["residue"] for c in ids[0]["cells"]] == [2, 1, 5, 1, 11]
    assert [c["residue"] for c in ids[1]["cells"]] == [0, 4, 5, 1, 3]
    assert (n_a, n_a * n_b) == (140, 5460) and obj["collisions"]["ids_per_code_a"] == [58, 59]
    same = [m for m in matches if m["query"] == m["key"]]
    diff = [m for m in matches if m["query"] != m["key"]]
    assert all(m["cells"] == 3 and m["score"] == 24 and m["beats_sink"] for m in same)
    assert all(m["cells"] == 1 and m["shared_dims"] == [46] and m["score"] == 8 and not m["beats_sink"] for m in diff)
    assert head["sink"] == 20
    return obj


def main() -> None:
    obj = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{OUT.relative_to(ROOT)} {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    sys.exit(main())
