"""Write the 200 WSP-B disclosed sessions as a readout subset (deviation D1, PREREGISTRATION §13 log).

    python3 scripts/v04_disclosed_sessions.py   # writes datasets/v0_4/DISCLOSED_SESSIONS.txt

The session ids of ``datasets/v0_4/BUDGETS.json`` ``disclosed_200w``, sorted, one per line. The WSP-A column the
disclosure builder reads may be a readout of just these sessions (``run_readout --split val --sessions FILE``)
instead of full val: each session is read out on its own, so the WSP-A class at a disclosed window is the same
either way. Reads no labels and no model output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "datasets" / "v0_4"
OUT = DATA / "DISCLOSED_SESSIONS.txt"


def select(budgets: dict) -> list[str]:
    ids = sorted({w["session_id"] for w in budgets["disclosed_200w"]})
    if len(ids) != len(budgets["disclosed_200w"]):
        raise SystemExit("two disclosed windows share a session")
    return ids


def main() -> int:
    ids = select(json.loads((DATA / "BUDGETS.json").read_text()))
    OUT.write_text("".join(s + "\n" for s in ids))
    print(f"{OUT.relative_to(REPO)}: {len(ids)} sessions, sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
