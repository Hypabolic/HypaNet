"""Write request files for set G (original model, all windows, hidden 22-24, mlp_act 22-23)."""
import json
from mk1 import write_request
from wsp import tok
import scen_g

pairs = scen_g.build_all()
sessions = [p[3] for p in pairs]
info = {p[3].sid: {"pair": p[0], "feat": p[1], "sign": p[2]} for p in pairs}
ret = {"hidden": [22, 23, 24], "mlp_act": [22, 23], "topk": 3}
chunks, cur, cur_tok = [], [], 0
for i in range(0, len(sessions), 2):
    pr = sessions[i:i + 2]
    t = sum(167 + sum(len(tok.encode(b + "\n", add_special_tokens=False).ids) for b in s.blocks()) + 11 * len(s.blocks()) for s in pr)
    if cur_tok + t > 45000:
        chunks.append(cur); cur, cur_tok = [], 0
    cur += pr; cur_tok += t
if cur:
    chunks.append(cur)
for i, part in enumerate(chunks, 1):
    req, meta = write_request(f"output/wsp-a/scratch/req_g{i:02d}.json", part, model="original", windows_mode="all",
                              ret=ret, meta_path=f"output/wsp-a/scratch/meta_g{i:02d}.json")
    for m in meta:
        m.update(info[m["id"]])
    with open(f"output/wsp-a/scratch/meta_g{i:02d}.json", "w") as f:
        json.dump(meta, f, indent=1)
    total = sum(167 + sum(len(tok.encode(b + "\n", add_special_tokens=False).ids) for b in s["blocks"]) + 11 * len(s["windows"]) for s in req["sessions"])
    print(f"req_g{i:02d}: sessions={len(req['sessions'])} est tokens={total}")
print("pairs:", len(pairs) // 2, {f: sum(1 for p in pairs if p[1] == f) // 2 for f in ("IMPROV", "EDITCOUNT")})
for s in sessions[:2]:
    print("----", s.sid, info[s.sid]); print("\n".join(s.blocks()))
