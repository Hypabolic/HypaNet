"""Set E feature table: x23-space feature projections (original model) and neuron activations (edited model),
grouped by (recipe, label). Writes the per-window table to output/wsp-a/scratch/ana_e_table.txt."""
import json
import sys
import numpy as np
import torch
from wsp import load_original

CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
P = "model.language_model."
N23, N22 = 1046, 1613
dirs = np.load("output/wsp-a/scratch/v1b_dirs.npz")
FE = ["ERR", "PROG", "REP", "RES", "EDITS", "EDITREPfree", "CHAIN"]
D23 = {f: torch.tensor(dirs[f"x23_{f}"]).double() for f in FE}
D24 = {f: torch.tensor(dirs[f"x24_{f}"]).double() for f in FE}
down23 = load_original(P + "layers.23.mlp.down_proj.weight").double()


def unit_rms(x):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


rows = []
for call, metaf in [a.split(":") for a in sys.argv[1:]]:
    resp = json.load(open(f"output/wsp-a/fwd/{call}/response.json"))
    req = {s["id"]: s for s in json.load(open(f"output/wsp-a/fwd/{call}/request.json"))["sessions"]}
    meta = {m["id"]: m for m in json.load(open(metaf))}
    for si, s in enumerate(resp["sessions"]):
        m = meta[s["id"]]
        a = lambda k: np.load(f"output/wsp-a/fwd/{call}/{resp['arrays'][k]['file']}")
        h24o = torch.tensor(a(f"original.s{si}.hidden.L24")).double()
        m23o = torch.tensor(a(f"original.s{si}.mlp_act.L23")).double()
        x23 = unit_rms(h24o - m23o @ down23.T)
        x24 = unit_rms(h24o)
        m22e, m23e = a(f"edited.s{si}.mlp_act.L22"), a(f"edited.s{si}.mlp_act.L23")
        blocks = req[s["id"]]["blocks"]
        for wi, t in enumerate(s["windows"]):
            f23 = {f: float(x23[wi] @ D23[f]) for f in FE}
            f24 = {f: float(x24[wi] @ D24[f]) for f in FE}
            rows.append({"sid": s["id"], "recipe": s["id"].split("_")[0], "t": t, "lab": m["labels"][t], "tool": blocks[t].split()[1],
                         "pe": int(np.argmax(s["edited"]["label_logits"][wi])), "po": int(np.argmax(s["original"]["label_logits"][wi])),
                         "act_o": float(m23e[wi, N23]), "act_s": float(m22e[wi, N22]), "f23": f23, "f24": f24,
                         "le": np.array(s["edited"]["label_logits"][wi])})
# pooled stats on E for display standardisation
allf = {f: np.array([r["f23"][f] for r in rows]) for f in FE}
mu = {f: allf[f].mean() for f in FE}; sd = {f: allf[f].std() for f in FE}
print("x23 pooled mu/sd on E:", {f: (round(mu[f], 2), round(sd[f], 2)) for f in FE})
groups = {}
for r in rows:
    groups.setdefault((r["recipe"], r["lab"]), []).append(r)
hdr = "recipe   label  n  acc_e acc_o | " + " ".join(f"{f[:6]:>6s}" for f in FE) + " | act_o act_s"
print(hdr)
order = sorted(groups, key=lambda k: (int(k[0][1:]), k[1]))
for key in order:
    g = groups[key]
    z = [np.mean([(r["f23"][f] - mu[f]) / sd[f] for r in g]) for f in FE]
    acc_e = np.mean([CL.index(r["lab"]) == r["pe"] for r in g]) if key[1] != "AMBIG" else float("nan")
    acc_o = np.mean([CL.index(r["lab"]) == r["po"] for r in g]) if key[1] != "AMBIG" else float("nan")
    print(f"{key[0]:8s} {key[1][:5]:5s} {len(g):2d}  {acc_e:5.2f} {acc_o:5.2f} | " + " ".join(f"{v:6.2f}" for v in z) +
          f" | {np.mean([r['act_o'] for r in g]):5.1f} {np.mean([r['act_s'] for r in g]):5.1f}")
with open("output/wsp-a/scratch/ana_e_table.txt", "w") as fh:
    for r in rows:
        z = " ".join(f"{(r['f23'][f] - mu[f]) / sd[f]:6.2f}" for f in FE)
        fh.write(f"{r['sid']:6s} t{r['t']:2d} {r['lab'][:4]} {r['tool']:11s} orig={CL[r['po']][:4]} edit={CL[r['pe']][:4]} | {z} | act_o={r['act_o']:6.2f} act_s={r['act_s']:6.2f} le={np.round(r['le'], 1)}\n")
print("table written")
