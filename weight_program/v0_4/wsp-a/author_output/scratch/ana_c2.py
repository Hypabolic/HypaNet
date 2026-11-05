"""Residual structure at the readout (massive dims, mean direction), cross-layer stability of feature directions,
and the OSCI-vs-STAL contrast (set B) vs an edit-type-repetition contrast (set C)."""
import json
import sys
import numpy as np
import torch
from wsp import load, label_rows

CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
fd = np.load("output/wsp-a/scratch/feat_dirs.npz", allow_pickle=True)
b = np.load("output/wsp-a/scratch/setB_orig.npz", allow_pickle=True)
Y = b["Y"]; mask = Y >= 0


def unit_rms(x):
    x = torch.tensor(np.asarray(x)).double()
    return x / x.pow(2).mean(-1, keepdim=True).sqrt()


# 1. residual structure at L24 and L23
for L in (24, 23, 22, 20):
    X = unit_rms(b[f"h{L}"])
    m = X.mean(0)
    print(f"L{L}: |x|={X.norm(dim=1).mean():.2f} |mean|={m.norm():.2f} cos(x, mean) mean/min={((X @ m) / (X.norm(dim=1) * m.norm())).mean():.3f}/{((X @ m) / (X.norm(dim=1) * m.norm())).min():.3f}")
    top = m.abs().topk(8)
    print("   largest |mean| dims:", [(int(i), round(float(m[i]), 2)) for i in top.indices])
    sd = X.std(0)
    print("   their std across windows:", [round(float(sd[i]), 3) for i in top.indices], " median std:", round(float(sd.median()), 3))
    raw = torch.tensor(b[f"h{L}"]).double()
    print(f"   raw residual norm mean/std: {raw.norm(dim=1).mean():.2f}/{raw.norm(dim=1).std():.2f}; raw largest dims:",
          [(int(i), round(float(raw.mean(0)[i]), 2)) for i in raw.mean(0).abs().topk(5).indices])

# 2. cross-layer stability of feature directions
feats = list(fd["feats"])
for f in feats:
    vs = torch.tensor(np.stack([fd[f"{f}_L{L}"] for L in (16, 20, 22, 23, 24)]))
    print(f"{f}: cos across layers 16/20/22/23/24:\n", np.round((vs @ vs.T).numpy(), 2))

# 3. class contrasts on set B at L24 vs feature directions
X = unit_rms(b["h24"][mask]); y = Y[mask]
mu = torch.stack([X[y == c].mean(0) for c in range(4)])
g = X.mean(0)
F = torch.tensor(np.stack([fd[f"{f}_L24"] for f in feats])).double()
Fn = F / F.norm(dim=1, keepdim=True)
osc_stal = mu[3] - mu[2]
reco_stal = mu[1] - mu[2]
prod_rest = mu[0] - X[y != 0].mean(0)
print("|OSCI-STAL|=", round(float(osc_stal.norm()), 3), " cos with feats:", dict(zip(feats, np.round((Fn @ (osc_stal / osc_stal.norm())).numpy(), 2))))
print("|RECO-STAL|=", round(float(reco_stal.norm()), 3), " cos with feats:", dict(zip(feats, np.round((Fn @ (reco_stal / reco_stal.norm())).numpy(), 2))))
print("|PROD-rest|=", round(float(prod_rest.norm()), 3), " cos with feats:", dict(zip(feats, np.round((Fn @ (prod_rest / prod_rest.norm())).numpy(), 2))))
# how much of each class contrast lies in the span of the 5 feature directions
Q, _ = torch.linalg.qr(F.T)  # orthonormal basis of the span
for name, v in (("OSCI-STAL", osc_stal), ("RECO-STAL", reco_stal), ("PROD-rest", prod_rest), ("OSCI-rest", mu[3] - X[y != 3].mean(0)), ("STAL-rest", mu[2] - X[y != 2].mean(0)), ("RECO-rest", mu[1] - X[y != 1].mean(0))):
    inspan = Q @ (Q.T @ v)
    print(f"  {name}: fraction of norm in feature span = {float(inspan.norm() / v.norm()):.2f}")

# 4. edit-type repetition contrast from set C: rev_ab positives (edits alternating) vs rep_* positives (non-edit repeats)
sess = {}
for call, metaf in [a.split(":") for a in sys.argv[1:]]:
    resp = json.load(open(f"output/wsp-a/fwd/{call}/response.json"))
    meta = {m["id"]: m for m in json.load(open(metaf))}
    for si, s in enumerate(resp["sessions"]):
        m = meta[s["id"]]
        sess[s["id"]] = {"feat": m["feat"], "sign": m["sign"], "blocks": s["blocks"],
                         "h24": np.load(f"output/wsp-a/fwd/{call}/{resp['arrays'][f'original.s{si}.hidden.L24']['file']}")}
edit_rep, other_rep, revert_rep = [], [], []
for sid, e in sess.items():
    if e["sign"] != 1:
        continue
    last = e["blocks"][-1]
    x = unit_rms(e["h24"])[-1]
    if e["feat"] == "REV" and "revert" in last:
        revert_rep.append(x)
    elif e["feat"] == "REV" and ("edit_file" in last or "write_file" in last):
        edit_rep.append(x)
    elif e["feat"] == "REP" and not ("edit_file" in last):
        other_rep.append(x)
edit_rep, other_rep, revert_rep = torch.stack(edit_rep), torch.stack(other_rep), torch.stack(revert_rep)
print("n edit-rep, other-rep, revert-rep:", len(edit_rep), len(other_rep), len(revert_rep))
d_er = edit_rep.mean(0) - other_rep.mean(0)
d_rv = revert_rep.mean(0) - other_rep.mean(0)
print("|edit-rep minus other-rep|=", round(float(d_er.norm()), 3), " cos with OSCI-STAL:", round(float(d_er @ osc_stal / d_er.norm() / osc_stal.norm()), 3))
print("|revert-rep minus other-rep|=", round(float(d_rv.norm()), 3), " cos with OSCI-STAL:", round(float(d_rv @ osc_stal / d_rv.norm() / osc_stal.norm()), 3))
print("cos(edit-rep dir, revert-rep dir):", round(float(d_er @ d_rv / d_er.norm() / d_rv.norm()), 3))
# d' of these directions for OSCI vs STAL on set B
for name, v in (("edit-rep", d_er), ("revert-rep", d_rv), ("OSCI-STAL", osc_stal)):
    vn = v / v.norm()
    s = (X @ vn).numpy()
    a, r = s[y == 3], s[y == 2]
    print(f"  {name}: set B OSCI {a.mean():.2f}+-{a.std():.2f} vs STAL {r.mean():.2f}+-{r.std():.2f}  d'={(a.mean() - r.mean()) / np.sqrt(0.5 * (a.var() + r.var())):.2f}",
          " | RECO", round(float(s[y == 1].mean()), 2), "PROD", round(float(s[y == 0].mean()), 2))
np.savez("output/wsp-a/scratch/contrasts_L24.npz", osc_stal=osc_stal.float().numpy(), edit_rep=d_er.float().numpy(), revert_rep=d_rv.float().numpy(),
         mean24=g.float().numpy(), mu=mu.float().numpy())
