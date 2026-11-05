"""Write request E01 (both models, all windows) for the fresh evaluation set E, plus activations of the edited model."""
import json
from mk1 import write_request
from wsp import tok
import scen_e

sessions = scen_e.all_sessions()
ret = {"hidden": [24], "mlp_act": [22, 23], "topk": 3}
chunks, cur, cur_tok = [], [], 0
for s in sessions:
    t = 2 * (167 + sum(len(tok.encode(b + "\n", add_special_tokens=False).ids) for b in s.blocks()) + 11 * len(s.blocks()))
    if cur_tok + t > 48000:
        chunks.append(cur); cur, cur_tok = [], 0
    cur.append(s); cur_tok += t
if cur:
    chunks.append(cur)
counts = {}
for i, part in enumerate(chunks, 1):
    req, meta = write_request(f"output/wsp-a/scratch/req_e{i:02d}.json", part, model="both", windows_mode="all",
                              ret=ret, meta_path=f"output/wsp-a/scratch/meta_e{i:02d}.json")
    total = 2 * sum(167 + sum(len(tok.encode(b + "\n", add_special_tokens=False).ids) for b in s["blocks"]) + 11 * len(s["windows"]) for s in req["sessions"])
    print(f"req_e{i:02d}: sessions={len(req['sessions'])} est tokens={total}")
    for m in meta:
        for l in m["labels"]:
            counts[l] = counts.get(l, 0) + 1
print("sessions:", len(sessions), "window label counts:", counts)
for s in sessions[:2]:
    labels, _ = s.labels()
    print("----", s.sid, " ".join(l[:4] for l in labels))
    print("\n".join(s.blocks()))
