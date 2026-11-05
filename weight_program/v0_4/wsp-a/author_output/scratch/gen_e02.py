"""Write request files E2-01..03: the same set-E sessions and windows as req_e01..03 (both models), returning
mlp_act at layers 21-23 so that all three neurons can be checked."""
import json
from mk1 import write_request
from wsp import tok
import scen_e

sessions = scen_e.all_sessions()
ret = {"hidden": [24], "mlp_act": [21, 22, 23], "topk": 3}
chunks, cur, cur_tok = [], [], 0
for s in sessions:
    t = 2 * (167 + sum(len(tok.encode(b + "\n", add_special_tokens=False).ids) for b in s.blocks()) + 11 * len(s.blocks()))
    if cur_tok + t > 48000:
        chunks.append(cur); cur, cur_tok = [], 0
    cur.append(s); cur_tok += t
if cur:
    chunks.append(cur)
for i, part in enumerate(chunks, 1):
    req, meta = write_request(f"output/wsp-a/scratch/req_e2_{i:02d}.json", part, model="both", windows_mode="all",
                              ret=ret, meta_path=f"output/wsp-a/scratch/meta_e2_{i:02d}.json")
    print(f"req_e2_{i:02d}: sessions={len(req['sessions'])}")
