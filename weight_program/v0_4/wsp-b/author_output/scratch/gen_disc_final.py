"""Write the final disclosed-window request files (model=edited), same chunks as disc_01..13."""
import json

for i in range(1, 14):
    req = json.load(open(f"output/wsp-b/scratch/disc_{i:02d}.json"))
    json.dump({"disclosed_windows": req["disclosed_windows"], "model": "edited"}, open(f"output/wsp-b/scratch/disc_f_{i:02d}.json", "w"))
    print(f"disc_f_{i:02d}.json: {len(req['disclosed_windows'])} windows")
