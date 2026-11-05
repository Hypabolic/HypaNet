"""Set G analysis: IMPROV and EDITCOUNT directions at x22/x23/x24 (exact MLP-input spaces), LOO consistency, and
their projections (standardised on sets B+D) by class and by recipe on sets E and F (original-model arrays).
usage: ./py ana_g.py G_CALL:META ... -- E/F CALL:META ..."""
import json
import sys
import numpy as np
import torch
from wsp import load_original

CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
P = "model.language_model."
down22 = load_original(P + "layers.22.mlp.down_proj.weight").double()
down23 = load_original(P + "layers.23.mlp.down_proj.weight").double()
args = sys.argv[1:]
sep = args.index("--")
g_calls, ef_calls = [a.split(":") for a in args[:sep]], [a.split(":") for a in args[sep + 1:]]


def unit_rms(x):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def spaces(call, si, resp, kind="original", with22=True):
    a = lambda k: torch.tensor(np.load(f"output/wsp-a/fwd/{call}/{resp['arrays'][f'{kind}.s{si}.{k}']['file']}")).double()
    h24, m23 = a("hidden.L24"), a("mlp_act.L23")
    out = {"x23": unit_rms(h24 - m23 @ down23.T), "x24": unit_rms(h24)}
    if with22:
        out["x22"] = unit_rms(a("hidden.L23") - a("mlp_act.L22") @ down22.T)
    return out


G = {}
for call, metaf in g_calls:
    resp = json.load(open(f"output/wsp-a/fwd/{call}/response.json"))
    meta = {m["id"]: m for m in json.load(open(metaf))}
    for si, s in enumerate(resp["sessions"]):
        m = meta[s["id"]]
        sp = spaces(call, si, resp)
        G[s["id"]] = {"pair": m["pair"], "feat": m["feat"], "sign": m["sign"], **{k: v[-1] for k, v in sp.items()}}
DIRS = {}
for space in ("x22", "x23", "x24"):
    for feat in ("IMPROV", "EDITCOUNT"):
        pairs = sorted({e["pair"] for e in G.values() if e["feat"] == feat})
        D = torch.stack([G[p + "p"][space] - G[p + "n"][space] for p in pairs])
        d = D.mean(0); dn = d / d.norm()
        proj = D @ dn
        loo = np.array([float(D[i] @ ((D.sum(0) - D[i]) / (len(D) - 1)) / ((D.sum(0) - D[i]) / (len(D) - 1)).norm()) for i in range(len(D))])
        print(f"{space} {feat:9s}: n={len(D)} |mean|={d.norm():.2f} proj {proj.mean():.2f}+-{proj.std():.2f} agree={(proj > 0).float().mean():.2f} LOO {loo.mean():.2f}+-{loo.std():.2f} agree={(loo > 0).mean():.2f}")
        DIRS[(space, feat)] = dn
# cos with the existing directions
old = np.load("output/wsp-a/scratch/v1b_dirs.npz")
for space in ("x22", "x23", "x24"):
    for feat in ("IMPROV", "EDITCOUNT"):
        cs = {f: round(float(DIRS[(space, feat)] @ torch.tensor(old[f"{space}_{f}"]).double()), 2) for f in ("ERR", "PROG", "REP", "RES", "EDITS", "EDITREPfree", "CHAIN")}
        print(f"  cos({space} {feat}, old):", cs)

# projections on set B (npz) and set D (calls 9-11) confident windows at x23, standardised; and E/F by recipe
b = np.load("output/wsp-a/scratch/setB_orig.npz", allow_pickle=True)
XB = {"x23": unit_rms(torch.tensor(b["h24"]).double() - torch.tensor(b["m23"]).double() @ down23.T), "x24": unit_rms(torch.tensor(b["h24"]).double()),
      "x22": unit_rms(torch.tensor(b["h23"]).double() - torch.tensor(b["m22"]).double() @ down22.T)}
YB = b["Y"]
XD, YD = {"x22": [], "x23": [], "x24": []}, []
for call in ("9", "10", "11"):
    resp = json.load(open(f"output/wsp-a/fwd/{call}/response.json"))
    i = {"9": 1, "10": 2, "11": 3}[call]
    meta = {m["id"]: m for m in json.load(open(f"output/wsp-a/scratch/meta_d{i:02d}.json"))}
    for si, s in enumerate(resp["sessions"]):
        sp = spaces(call, si, resp)
        for wi, t in enumerate(s["windows"]):
            lab = meta[s["id"]]["labels"][t]
            YD.append(CL.index(lab) if lab != "AMBIG" else -1)
            for k in XD:
                XD[k].append(sp[k][wi])
YD = np.array(YD)
X = {k: torch.cat([XB[k][YB >= 0], torch.stack(XD[k])[YD >= 0]]) for k in XD}
Y = np.concatenate([YB[YB >= 0], YD[YD >= 0]])
stats = {}
for space in ("x22", "x23", "x24"):
    for feat in ("IMPROV", "EDITCOUNT"):
        s = (X[space] @ DIRS[(space, feat)]).numpy()
        stats[(space, feat)] = (s.mean(), s.std())
        parts = [f"{CL[c][:4]} {s[Y == c].mean():5.2f}±{s[Y == c].std():4.2f} [d'={(s[Y == c].mean() - s[Y != c].mean()) / np.sqrt(0.5 * (s[Y == c].var() + s[Y != c].var())):5.2f}]" for c in range(4)]
        print(f"{space} {feat:9s} B+D: " + " | ".join(parts))
# E/F by recipe (x23), standardised by B+D stats
rows = {}
for call, metaf in ef_calls:
    resp = json.load(open(f"output/wsp-a/fwd/{call}/response.json"))
    meta = {m["id"]: m for m in json.load(open(metaf))}
    for si, s in enumerate(resp["sessions"]):
        sp = spaces(call, si, resp, with22=False)
        for wi, t in enumerate(s["windows"]):
            lab = meta[s["id"]]["labels"][t]
            key = (s["id"].split("_")[0], lab)
            rows.setdefault(key, []).append([(float(sp["x23"][wi] @ DIRS[("x23", f)]) - stats[("x23", f)][0]) / stats[("x23", f)][1] for f in ("IMPROV", "EDITCOUNT")])
print("recipe label   n  zIMPROV zEDITCOUNT (x23, standardised on B+D)")
for key in sorted(rows, key=lambda k: (k[0][0], int(k[0][1:]), k[1])):
    v = np.array(rows[key])
    if key[1] == "AMBIG":
        continue
    print(f"{key[0]:6s} {key[1][:5]:5s} {len(v):3d}  {v[:, 0].mean():6.2f}  {v[:, 1].mean():6.2f}")
np.savez("output/wsp-a/scratch/dirs_G.npz", **{f"{sp}_{f}": DIRS[(sp, f)].float().numpy() for (sp, f) in DIRS})
