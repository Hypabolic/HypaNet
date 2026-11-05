"""Write requests for scenario set D (matched groups), original model, all windows, hidden 20-24, mlp_act 21-23."""
import json
from mk1 import write_request
from wsp import tok
import scen_d

groups = scen_d.build_all()
sessions = [g[2] for g in groups]
info = {g[2].sid: {"group": g[0], "tag": g[1]} for g in groups}
ret = {"hidden": [20, 21, 22, 23, 24], "mlp_act": [21, 22, 23], "topk": 3}
chunks, cur, cur_tok = [], [], 0
for s in sessions:
    t = 167 + sum(len(tok.encode(b + "\n", add_special_tokens=False).ids) for b in s.blocks()) + 11 * len(s.blocks())
    if cur_tok + t > 45000:
        chunks.append(cur); cur, cur_tok = [], 0
    cur.append(s); cur_tok += t
if cur:
    chunks.append(cur)
counts = {}
for i, part in enumerate(chunks, 1):
    req, meta = write_request(f"output/wsp-a/scratch/req_d{i:02d}.json", part, model="original", windows_mode="all",
                              ret=ret, meta_path=f"output/wsp-a/scratch/meta_d{i:02d}.json")
    for m in meta:
        m.update(info[m["id"]])
    with open(f"output/wsp-a/scratch/meta_d{i:02d}.json", "w") as f:
        json.dump(meta, f, indent=1)
    total = sum(167 + sum(len(tok.encode(b + "\n", add_special_tokens=False).ids) for b in s["blocks"]) + 11 * len(s["windows"]) for s in req["sessions"])
    print(f"req_d{i:02d}: sessions={len(req['sessions'])} est tokens={total}")
    for m in meta:
        for l in m["labels"]:
            counts[l] = counts.get(l, 0) + 1
print("sessions:", len(sessions), "window label counts:", counts)
tags = {}
for g in groups:
    tags[g[1]] = tags.get(g[1], 0) + 1
print("sessions per tag:", tags)
for s in sessions[:3]:
    labels, _ = s.labels()
    print("----", s.sid, info[s.sid], " ".join(l[:4] for l in labels))
    print("\n".join(s.blocks()))
