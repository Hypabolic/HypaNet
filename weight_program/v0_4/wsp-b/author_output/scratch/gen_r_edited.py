"""Write set-R request files for the edited model (same sessions and windows as req_r01/r02), with activations."""
import json

for name in ("r01", "r02"):
    req = json.load(open(f"output/wsp-b/scratch/req_{name}.json"))
    req["model"] = "edited"
    req["return"] = {"hidden": [24], "mlp_act": [21, 22, 23], "topk": 3}
    json.dump(req, open(f"output/wsp-b/scratch/req_{name}_ed.json", "w"))
    print("wrote", f"req_{name}_ed.json", len(req["sessions"]), "sessions")
