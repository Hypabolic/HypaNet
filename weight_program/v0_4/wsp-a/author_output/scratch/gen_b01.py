"""Write requests B01..B03: scenario set B split in three, every window, original model, activations."""
import json
from mk1 import write_request
from wsp import tok
import scen_b

sessions = scen_b.all_sessions()
ret = {"hidden": [12, 16, 20, 22, 23, 24], "mlp_act": [20, 21, 22, 23], "topk": 5}
parts = [sessions[0::3], sessions[1::3], sessions[2::3]]
counts = {}
for i, part in enumerate(parts, 1):
    req, meta = write_request(f"output/wsp-a/scratch/req_b0{i}.json", part, model="original", windows_mode="all",
                              ret=ret, meta_path=f"output/wsp-a/scratch/meta_b0{i}.json")
    total = 0
    for s in req["sessions"]:
        total += 167 + sum(len(tok.encode(b + "\n", add_special_tokens=False).ids) for b in s["blocks"]) + 11 * len(s["windows"])
    print(f"req_b0{i}: sessions={len(req['sessions'])} est tokens={total}")
    for m in meta:
        for l in m["labels"]:
            counts[l] = counts.get(l, 0) + 1
print("window label counts:", counts)
for s in sessions[:90:30]:
    labels, _ = s.labels()
    print("----", s.sid, " ".join(l[:4] for l in labels))
    print("\n".join(s.blocks()))
