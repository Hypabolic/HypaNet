"""Write disclosed-window request files (model=both) in chunks under 50,000 tokens each (counting both models)."""
import json
from wsp import tok

D = json.load(open("disclosure/DISCLOSURE.json"))
items = D["windows"] if "windows" in D else D
if isinstance(items, dict):
    items = list(items.values())
ids, costs = [], []
for w in items:
    text = "\n".join(ln for ln in w["prefix"].split("\n") if ln != "### window start")
    n = 167 + len(tok.encode(text + "\n", add_special_tokens=False).ids) + 11
    ids.append(w["id"]); costs.append(2 * n)
chunks, cur, tot = [], [], 0
for wid, c in zip(ids, costs):
    if tot + c > 46000:
        chunks.append(cur); cur, tot = [], 0
    cur.append(wid); tot += c
if cur:
    chunks.append(cur)
for i, ch in enumerate(chunks, 1):
    json.dump({"disclosed_windows": ch, "model": "both"}, open(f"output/wsp-b/scratch/disc_{i:02d}.json", "w"))
    print(f"disc_{i:02d}.json: {len(ch)} windows, est tokens {sum(costs[ids.index(w)] for w in ch)}")
print("total est tokens:", sum(costs))
