"""Final calibration step: scale the STALLED neuron's output column so that its median logit gain on the disclosed
STALLED windows equals the designed TARGET_GAIN of 5 (factor = 5 / median(revision - original stalled logit on
STALLED windows, calls 1-13 vs 26-38)). Overwrites output/wsp-b/edit_constants.safetensors (down_col_L22 only)."""
import json
import numpy as np
import torch
from safetensors.torch import load_file, save_file

CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
feat = {r["id"]: r for r in json.load(open("output/wsp-b/scratch/disc_feat.json"))}
L = json.load(open("output/wsp-b/scratch/disc_logits.json"))
LF = {}
for call in range(26, 39):
    for e in json.load(open(f"output/wsp-b/fwd/{call}/response.json"))["sessions"]:
        ed = e["edited"]; LF[e["id"]] = np.array(ed["label_logits"][0] if isinstance(ed["label_logits"][0], list) else ed["label_logits"])
gains = [LF[w][2] - L[w]["lo"][2] for w in feat if feat[w]["oracle"] == "STALLED"]
factor = 5.0 / float(np.median(gains))
print(f"median stalled gain on STALLED windows = {np.median(gains):.3f}; factor = {factor:.4f}")
consts = load_file("output/wsp-b/edit_constants.safetensors")
old = consts["down_col_L22"]
consts["down_col_L22"] = (old.double() * factor).float()
print(f"down_col_L22 norm {float(old.norm()):.4f} -> {float(consts['down_col_L22'].norm()):.4f}")
save_file(consts, "output/wsp-b/edit_constants.safetensors")
json.dump({"factor": factor, "median_gain": float(np.median(gains))}, open("output/wsp-b/scratch/rescale_stal.json", "w"), indent=1)
print("saved")
