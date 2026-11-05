"""Split-half check of direction quality: derive on sessions ending in _a, evaluate on _b (and vice versa).
Also time courses of a few top neurons. usage: ./py ana_b2.py setB_orig.npz"""
import sys
import numpy as np
import torch

d = np.load(sys.argv[1], allow_pickle=True)
Y, SID, WIN = d["Y"], d["SID"], d["WIN"]
CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
mask = Y >= 0
is_a = np.array([s.endswith("_a") for s in SID])


def dprime(a, b):
    return (a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-9)


def eval_dirs(key, normalise, k_sparse=None):
    X = torch.tensor(d[key]).double()
    if normalise:
        X = X / X.pow(2).mean(1, keepdim=True).sqrt()
    res = []
    for tr, te in ((is_a, ~is_a), (~is_a, is_a)):
        trm, tem = mask & tr, mask & te
        Xtr, ytr, Xte, yte = X[trm], Y[trm], X[tem], Y[tem]
        mu = torch.stack([Xtr[ytr == c].mean(0) for c in range(4)])
        g = Xtr.mean(0)
        D = mu - g
        if k_sparse:
            sd = torch.sqrt(torch.stack([Xtr[ytr == c].var(0) for c in range(4)]).mean(0) + 1e-6)
            for c in range(4):
                dn = (mu[c] - Xtr[ytr != c].mean(0)) / sd
                idx = dn.abs().topk(k_sparse).indices
                sp = torch.zeros_like(D[c]); sp[idx] = (mu[c] - Xtr[ytr != c].mean(0))[idx]
                D[c] = sp
        # one-vs-rest d' on test
        row = []
        for c in range(4):
            v = D[c] / D[c].norm()
            row.append(dprime((Xte[yte == c] @ v).numpy(), (Xte[yte != c] @ v).numpy()))
        # nearest-direction classifier with per-class threshold = midpoint of train class means along own dir
        S = Xte @ (D / D.norm(dim=1, keepdim=True)).T
        Str = Xtr @ (D / D.norm(dim=1, keepdim=True)).T
        # centre each score by the train mean over all classes (so biases are comparable)
        S = S - Str.mean(0)
        pred = S.argmax(1).numpy()
        cm = np.zeros((4, 4), int)
        for y, p in zip(yte, pred):
            cm[y, p] += 1
        res.append((row, cm))
    return res


for key, norm in [("h24", True), ("h23", True), ("h22", True), ("h20", True), ("h16", True), ("h12", True)]:
    r = eval_dirs(key, norm)
    print(f"--- {key} dense diff-of-means, split-half one-vs-rest d' (a->b, b->a):",
          [round(x, 2) for x in r[0][0]], [round(x, 2) for x in r[1][0]])
    cm = r[0][1] + r[1][1]
    print("    nearest-centred-direction confusion (both folds):\n", cm, " acc:", round(np.trace(cm) / cm.sum(), 3))
for key in ["m20", "m21", "m22", "m23"]:
    for k in (32, 128, None):
        r = eval_dirs(key, False, k)
        cm = r[0][1] + r[1][1]
        print(f"--- {key} sparse={k}: split-half one-vs-rest d'", [round(x, 2) for x in r[0][0]], [round(x, 2) for x in r[1][0]],
              " acc:", round(np.trace(cm) / cm.sum(), 3))
    print("    confusion (k=None):\n", cm)

# time courses of top neurons (from the full-data ranking in ana_b) in a few sessions
A = d["m23"]
for sid in ["S1_a", "S6_a", "R1_a", "P7_a", "O1_a", "O2_a", "P1_a"]:
    sel = np.where(SID == sid)[0]
    labs = [CL[Y[i]][:2] if Y[i] >= 0 else "--" for i in sel]
    print(sid, "labels:", " ".join(labs))
    for n in (2405, 2993, 1318, 1010, 2195, 3440, 857):
        print(f"   n{n}: " + " ".join(f"{A[i, n]:5.2f}" for i in sel))
