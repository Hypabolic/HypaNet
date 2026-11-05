"""WSP-B start: kernel state, budget, disclosure structure and the WSP-A edit's confusion on the 200 windows."""
import json
import hashlib
import numpy as np
from wsp import edited, edit_hash, budget, load_quota

print("edit hash:", edit_hash())
print("edited:", [n.split("language_model.")[1] for n in edited()])
print("budget:", budget())
print("quota:", load_quota())
raw = open("disclosure/DISCLOSURE.json", "rb").read()
print("disclosure sha256:", hashlib.sha256(raw).hexdigest(), "bytes:", len(raw))
D = json.loads(raw)
print("type:", type(D).__name__, "len:", len(D))
items = D if isinstance(D, list) else D.get("windows", D)
if isinstance(items, dict):
    keys = list(items)
    print("keys sample:", keys[:3])
    first = items[keys[0]]
else:
    first = items[0]
print("first item keys:", list(first.keys()) if isinstance(first, dict) else type(first))
CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
conf = np.zeros((4, 4), int)
rows = []
it = items.items() if isinstance(items, dict) else [(w.get("id", i), w) for i, w in enumerate(items)]
for wid, w in it:
    y, p = w["oracle_label"], w["network_class"]
    conf[CL.index(y), CL.index(p)] += 1
    text = w["prefix"]
    lines = text.split("\n")
    nblocks = sum(1 for ln in lines if ln and not ln.startswith(" ") and not ln.startswith("###"))
    rows.append((wid, y, p, nblocks, len(text)))
print("WSP-A edit confusion on disclosed windows (rows=oracle, cols=network):\n", conf)
print("acc:", np.trace(conf) / conf.sum(), "recall:", np.round(np.diag(conf) / conf.sum(1), 2), "precision:", np.round(np.diag(conf) / np.maximum(conf.sum(0), 1), 2))
nb = np.array([r[3] for r in rows]); ch = np.array([r[4] for r in rows])
print("blocks per window: min/median/max", nb.min(), int(np.median(nb)), nb.max(), " chars: min/median/max", ch.min(), int(np.median(ch)), ch.max())
with open("output/wsp-b/scratch/disc_index.txt", "w") as f:
    for r in rows:
        f.write(f"{r[0]} oracle={r[1]:11s} net={r[2]:11s} blocks={r[3]:3d} chars={r[4]}\n")
# show one window of each class, the first disclosed example
shown = set()
for wid, w in it:
    if w["oracle_label"] in shown:
        continue
    shown.add(w["oracle_label"])
    print(f"=============== {wid} oracle={w['oracle_label']} net={w['network_class']}")
    print(w["prefix"][-2500:])
