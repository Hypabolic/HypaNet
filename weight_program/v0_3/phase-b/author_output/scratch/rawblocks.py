"""Print raw rendered blocks: rawblocks.py V029 6 7 8  (window id, block indices)."""
import sys

from disc import load

wins = {w["id"]: w for w in load()}
w = wins[sys.argv[1]]
for i in sys.argv[2:]:
    b = w["blocks"][int(i)]
    print(f"--- {w['id']} block {i} (oracle {w['label']})")
    print(b)
