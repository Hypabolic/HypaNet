"""Write request A02: all scenario-A1 sessions, every window, original model, hidden states at several layers."""
import json
from mk1 import write_request
from wsp import tok
import scen_a1

sessions = scen_a1.all_sessions()
req, meta = write_request("output/wsp-a/scratch/req_a02.json", sessions, model="original", windows_mode="all",
                          ret={"hidden": [4, 8, 12, 16, 20, 22, 23, 24], "mlp_act": [20, 21, 22, 23], "topk": 10},
                          meta_path="output/wsp-a/scratch/meta_a02.json")
total = 0
for s in req["sessions"]:
    n = 167 + sum(len(tok.encode(b + "\n", add_special_tokens=False).ids) for b in s["blocks"]) + 11 * len(s["windows"])
    total += n
print("sessions:", len(req["sessions"]), "est tokens:", total)
for m in meta:
    print(m["id"], m["n"], " ".join(l[:4] for l in m["labels"]))
# show two rendered sessions for a sanity check
for s in req["sessions"][:1] + req["sessions"][17:18]:
    print("----", s["id"])
    print("\n".join(s["blocks"]))
