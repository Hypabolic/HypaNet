"""Write requests for scenario set C (matched pairs), original model, all windows, hidden states."""
import json
from mk1 import write_request
from wsp import tok
import scen_c

pairs = scen_c.build_all()
sessions = [p[3] for p in pairs]
info = {p[3].sid: {"pair": p[0], "feat": p[1], "sign": p[2]} for p in pairs}
ret = {"hidden": [16, 20, 22, 23, 24], "mlp_act": [22, 23], "topk": 3}
# split into chunks under ~45k tokens, keeping pairs together
chunks, cur, cur_tok = [], [], 0
for i in range(0, len(sessions), 2):
    pr = sessions[i:i + 2]
    t = sum(167 + sum(len(tok.encode(b + "\n", add_special_tokens=False).ids) for b in s.blocks()) + 11 * len(s.blocks()) for s in pr)
    if cur_tok + t > 45000:
        chunks.append(cur); cur, cur_tok = [], 0
    cur += pr; cur_tok += t
if cur:
    chunks.append(cur)
counts = {}
for i, part in enumerate(chunks, 1):
    req, meta = write_request(f"output/wsp-a/scratch/req_c{i:02d}.json", part, model="original", windows_mode="all",
                              ret=ret, meta_path=f"output/wsp-a/scratch/meta_c{i:02d}.json")
    for m in meta:
        m.update(info[m["id"]])
    with open(f"output/wsp-a/scratch/meta_c{i:02d}.json", "w") as f:
        json.dump(meta, f, indent=1)
    total = sum(167 + sum(len(tok.encode(b + "\n", add_special_tokens=False).ids) for b in s["blocks"]) + 11 * len(s["windows"]) for s in req["sessions"])
    print(f"req_c{i:02d}: sessions={len(req['sessions'])} est tokens={total}")
    for m in meta:
        for l in m["labels"]:
            counts[l] = counts.get(l, 0) + 1
print("pairs:", len(pairs) // 2, "window label counts:", counts)
feats = {}
for p in pairs:
    feats[p[1]] = feats.get(p[1], 0) + 1
print("sessions per feature:", feats)
for s in sessions[:2] + sessions[40:42]:
    labels, _ = s.labels()
    print("----", s.sid, info[s.sid], " ".join(l[:4] for l in labels))
    print("\n".join(s.blocks()))
