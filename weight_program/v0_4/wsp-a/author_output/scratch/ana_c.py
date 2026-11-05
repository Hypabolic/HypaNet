"""Matched-pair feature directions from set C (calls 5-8), their consistency, and class projections on set B.
usage: ./py ana_c.py OUT.npz SETB.npz CALL:META ..."""
import json
import sys
import numpy as np
import torch

out_path, setb_path = sys.argv[1], sys.argv[2]
pairs = [a.split(":") for a in sys.argv[3:]]
CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
LAYERS = [16, 20, 22, 23, 24]
sess = {}  # sid -> dict(h{L}: [n_win, d], labels, pair, feat, sign, logits)
for call, metaf in pairs:
    resp = json.load(open(f"output/wsp-a/fwd/{call}/response.json"))
    meta = {m["id"]: m for m in json.load(open(metaf))}
    for si, s in enumerate(resp["sessions"]):
        m = meta[s["id"]]
        e = {"labels": m["labels"], "pair": m["pair"], "feat": m["feat"], "sign": m["sign"],
             "logits": np.array(s["original"]["label_logits"])}
        for L in LAYERS:
            e[f"h{L}"] = np.load(f"output/wsp-a/fwd/{call}/{resp['arrays'][f'original.s{si}.hidden.L{L}']['file']}")
        sess[s["id"]] = e


def unit_rms(x):
    x = torch.tensor(x).double()
    return x / x.pow(2).mean(-1, keepdim=True).sqrt()


feats = sorted({e["feat"] for e in sess.values()})
dirs = {}
print("pairs per feature:", {f: sum(1 for e in sess.values() if e["feat"] == f) // 2 for f in feats})
for L in LAYERS:
    print(f"=== layer {L}")
    for f in feats:
        deltas, deltas_prev = [], []
        for pid in sorted({e["pair"] for e in sess.values() if e["feat"] == f}):
            p, n = sess[pid + "p"], sess[pid + "n"]
            xp, xn = unit_rms(p[f"h{L}"]), unit_rms(n[f"h{L}"])
            deltas.append(xp[-1] - xn[-1])
            if len(xp) > 1 and len(xn) > 1:
                deltas_prev.append(xp[-2] - xn[-2])
        D = torch.stack(deltas)
        d = D.mean(0)
        dn = d / d.norm()
        proj = D @ dn
        Dp = torch.stack(deltas_prev)
        projp = Dp @ dn
        # leave-one-out consistency: direction from all other pairs, projection of the held-out pair
        loo = []
        for i in range(len(D)):
            di = (D.sum(0) - D[i]) / (len(D) - 1)
            loo.append(float(D[i] @ (di / di.norm())))
        loo = np.array(loo)
        print(f"  {f:4s}: |mean delta|={d.norm():.3f}  proj mean/std={proj.mean():.3f}/{proj.std():.3f} sign-agree={(proj > 0).float().mean():.2f}"
              f"  LOO mean/std={loo.mean():.3f}/{loo.std():.3f} agree={(loo > 0).mean():.2f}  prev-window agree={(projp > 0).float().mean():.2f}")
        dirs[f"{f}_L{L}"] = dn.float().numpy()
    M = torch.tensor(np.stack([dirs[f"{f}_L{L}"] for f in feats]))
    print("  cos between feature directions:\n", np.round((M @ M.T).numpy(), 2))

np.savez(out_path, **dirs, feats=np.array(feats))

# class projections on set B (confident windows), per layer
b = np.load(setb_path, allow_pickle=True)
Y = b["Y"]
mask = Y >= 0
for L in LAYERS:
    X = unit_rms(b[f"h{L}"][mask])
    y = Y[mask]
    print(f"=== set B projections, layer {L} (mean +- std by class; d' one-vs-rest in brackets)")
    for f in feats:
        v = torch.tensor(dirs[f"{f}_L{L}"]).double()
        s = (X @ v).numpy()
        parts = []
        for c in range(4):
            a, r = s[y == c], s[y != c]
            dp = (a.mean() - r.mean()) / np.sqrt(0.5 * (a.var() + r.var()))
            parts.append(f"{CL[c][:4]} {a.mean():6.2f}+-{a.std():4.2f} [{dp:5.2f}]")
        print(f"  {f:4s}: " + " | ".join(parts))
