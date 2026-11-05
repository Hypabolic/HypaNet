"""Write the v0.4 val quick-look subset: 1,000 val sessions chosen by hash key, before any edit exists.

    python3 scripts/v04_val_quick_subset.py   # writes datasets/v0_4/VAL_QUICK.txt

The sessions are the 1,000 val sessions with the smallest ``sha256(f"20261106:v04-val-quick:{session_id}")``
after the 200 WSP-B disclosed sessions (``datasets/v0_4/BUDGETS.json`` ``disclosed_200w``) are removed.
One id per line, in key order. Reads only the val task export's session ids; no labels and no model output.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "datasets" / "v0_4"
OUT = DATA / "VAL_QUICK.txt"
KEY = "20261106:v04-val-quick:{}"
N = 1000


def key(session_id: str) -> str:
    return hashlib.sha256(KEY.format(session_id).encode()).hexdigest()


def select(val_ids: list[str], disclosed: set[str], n: int = N) -> list[str]:
    pool = [s for s in val_ids if s not in disclosed]
    return sorted(pool, key=key)[:n]


def main() -> int:
    with gzip.open(DATA / "val.tasks.jsonl.gz", "rt") as f:
        val_ids = [json.loads(line)["session_id"] for line in f]
    disclosed = {w["session_id"] for w in json.loads((DATA / "BUDGETS.json").read_text())["disclosed_200w"]}
    ids = select(val_ids, disclosed)
    OUT.write_text("".join(s + "\n" for s in ids))
    print(f"{OUT.relative_to(REPO)}: {len(ids)} of {len(val_ids)} val sessions "
          f"({len(disclosed)} disclosed removed), sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
