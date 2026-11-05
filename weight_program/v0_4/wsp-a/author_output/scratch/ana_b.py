"""Pool fwd calls over a scenario set; baseline confusion; separability per layer; discriminative MLP neurons.
usage: ./py ana_b.py OUT.npz CALL:META [CALL:META ...]"""
import json
import sys
import numpy as np
import torch
from wsp import load, label_rows

out_path = sys.argv[1]
pairs = [a.split(":") for a in sys.argv[2:]]
CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
H_LAYERS, M_LAYERS = [12, 16, 20, 22, 23, 24], [20, 21, 22, 23]
data = {f"h{L}": [] for L in H_LAYERS}
data.update({f"m{L}": [] for L in M_LAYERS})
Y, LOG, SID, WIN, NBLK = [], [], [], [], []
conf = np.zeros((4, 4), int)
for call, metaf in pairs:
    resp = json.load(open(f"output/wsp-a/fwd/{call}/response.json"))
    meta = {m["id"]: m for m in json.load(open(metaf))}
    kind = "original" if resp["model"] == "original" else "edited"
    for si, s in enumerate(resp["sessions"]):
        m = meta[s["id"]]
        r = s[kind]
        arrs = {}
        for L in H_LAYERS:
            arrs[f"h{L}"] = np.load(f"output/wsp-a/fwd/{call}/{resp['arrays'][f'{kind}.s{si}.hidden.L{L}']['file']}")
        for L in M_LAYERS:
            arrs[f"m{L}"] = np.load(f"output/wsp-a/fwd/{call}/{resp['arrays'][f'{kind}.s{si}.mlp_act.L{L}']['file']}")
        for wi, w in enumerate(s["windows"]):
            lab = m["labels"][w]
            lg = r["label_logits"][wi]
            y = CL.index(lab) if lab != "AMBIG" else -1
            if y >= 0:
                conf[y, int(np.argmax(lg))] += 1
            Y.append(y); LOG.append(lg); SID.append(s["id"]); WIN.append(w); NBLK.append(m["n"])
            for k in arrs:
                data[k].append(arrs[k][wi])
Y = np.array(Y); LOG = np.array(LOG)
np.savez(out_path, Y=Y, LOG=LOG, SID=np.array(SID), WIN=np.array(WIN), NBLK=np.array(NBLK), **{k: np.stack(v) for k, v in data.items()})
print("confusion (rows=my label, cols=pred):\n", conf, "\nacc:", round(np.trace(conf) / conf.sum(), 3), "n:", conf.sum())

def dprime(a, b):
    return (a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-9)

mask = Y >= 0
for L in H_LAYERS:
    X = torch.tensor(data[f"h{L}"])[mask].double()
    y = torch.tensor(Y[mask])
    Xn = X / X.pow(2).mean(1, keepdim=True).sqrt()
    mu = torch.stack([Xn[y == c].mean(0) for c in range(4)])
    print(f"--- hidden L{L}: pairwise d' along diff-of-means (resubstitution)")
    for a in range(4):
        row = []
        for b in range(4):
            if a == b:
                row.append("  -  "); continue
            d = mu[a] - mu[b]; d = d / d.norm()
            pa, pb = (Xn[y == a] @ d).numpy(), (Xn[y == b] @ d).numpy()
            row.append(f"{dprime(pa, pb):5.2f}")
        print("  ", CL[a][:4], " ".join(row))
    # one-vs-rest
    row = []
    for c in range(4):
        d = mu[c] - Xn[y != c].mean(0); d = d / d.norm()
        row.append(f"{CL[c][:4]}:{dprime((Xn[y == c] @ d).numpy(), (Xn[y != c] @ d).numpy()):4.2f}")
    print("   one-vs-rest d':", " ".join(row))

for L in M_LAYERS:
    A = torch.tensor(data[f"m{L}"])[mask].double()
    y = torch.tensor(Y[mask])
    print(f"--- mlp_act L{L}: act rms={A.pow(2).mean().sqrt():.3f}")
    for c in range(4):
        a_c, a_r = A[y == c], A[y != c]
        dn = (a_c.mean(0) - a_r.mean(0)) / torch.sqrt(0.5 * (a_c.var(0) + a_r.var(0)) + 1e-6)
        top = dn.abs().topk(8)
        feat = (a_c.mean(0) - a_r.mean(0))
        f_c, f_r = (a_c @ feat).numpy(), (a_r @ feat).numpy()
        # sparse feature: top-32 neurons by |d'|, signed
        idx = dn.abs().topk(32).indices
        sp = torch.zeros_like(feat); sp[idx] = feat[idx]
        s_c, s_r = (a_c @ sp).numpy(), (a_r @ sp).numpy()
        print(f"  {CL[c][:4]} vs rest: top neurons {[(int(i), round(float(dn[i]), 2)) for i in top.indices]}")
        print(f"        dense diff-of-means feature d'={dprime(f_c, f_r):.2f}; top-32 sparse feature d'={dprime(s_c, s_r):.2f}")
