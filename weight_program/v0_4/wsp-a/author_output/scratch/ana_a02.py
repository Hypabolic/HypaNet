"""Analyse fwd call 1 (req_a02): per-window logits vs my labels, confusion, and class geometry at L24."""
import json
import sys
import numpy as np
import torch
from wsp import tok, label_lens, label_rows, load

CALL = sys.argv[1] if len(sys.argv) > 1 else "1"
META = sys.argv[2] if len(sys.argv) > 2 else "output/wsp-a/scratch/meta_a02.json"
resp = json.load(open(f"output/wsp-a/fwd/{CALL}/response.json"))
meta = {m["id"]: m for m in json.load(open(META))}
kind = "original" if resp["model"] == "original" else "edited"
CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
conf = np.zeros((4, 4), int)
rows = []
H = {}
for si, s in enumerate(resp["sessions"]):
    m = meta[s["id"]]
    r = s[kind]
    print(f"== {s['id']}")
    for wi, w in enumerate(s["windows"]):
        lab = m["labels"][w]
        lg = r["label_logits"][wi]
        pred = int(np.argmax(lg))
        top = r["argmax"][wi]
        flag = ""
        if lab != "AMBIG":
            conf[CL.index(lab), pred] += 1
            flag = "ok" if CL.index(lab) == pred else "XX"
        print(f"  w{w:2d} {lab[:4]} pred={CL[pred][:4]} {flag:2s} logits={np.round(lg, 2)} top={repr(tok.decode([top]))}")
        rows.append((s["id"], w, lab, pred, lg))
print("confusion (rows=my label, cols=pred):")
print(conf)
print("acc:", np.trace(conf) / conf.sum())

# geometry at L24 (and other layers) over confident windows
for L in [24, 20, 16, 12]:
    X, Y = [], []
    for si, s in enumerate(resp["sessions"]):
        m = meta[s["id"]]
        key = f"{kind}.s{si}.hidden.L{L}"
        a = np.load(f"output/wsp-a/fwd/{CALL}/{resp['arrays'][key]['file']}")
        for wi, w in enumerate(s["windows"]):
            lab = m["labels"][w]
            if lab == "AMBIG":
                continue
            X.append(a[wi]); Y.append(CL.index(lab))
    X = torch.tensor(np.stack(X)); Y = torch.tensor(Y)
    print(f"--- layer {L}: n={len(Y)} per class={[(Y == c).sum().item() for c in range(4)]}, resid norm mean={X.norm(dim=1).mean():.1f}")
    # normalise like the final norm would (unit RMS) so that scale differences do not dominate
    Xn = X / X.pow(2).mean(1, keepdim=True).sqrt()
    mu = torch.stack([Xn[Y == c].mean(0) for c in range(4)])
    g = Xn.mean(0)
    D = mu - g
    # cosine between class directions
    Dn = D / D.norm(dim=1, keepdim=True)
    print(" class-direction norms:", np.round(D.norm(dim=1).numpy(), 3))
    print(" cos between class directions:\n", np.round((Dn @ Dn.T).numpy(), 2))
    # nearest-class-mean accuracy with those directions (resubstitution; a sanity check of separability only)
    sc = Xn @ D.T
    pred = sc.argmax(1)
    cm = np.zeros((4, 4), int)
    for y, p in zip(Y.tolist(), pred.tolist()):
        cm[y, p] += 1
    print(" nearest-direction confusion:\n", cm)
    if L == 24:
        # how do the original label rows align with the class directions (in normed space incl. (1+w))
        w = load("model.language_model.norm.weight")
        E = label_rows(original=True)
        Eg = E * (1 + w)  # effective readout direction on unit-RMS residual
        print(" cos(label row (1+w), class dir):\n", np.round(((Eg / Eg.norm(dim=1, keepdim=True)) @ Dn.T).numpy(), 3))
        # per-class projection stats of windows on each class direction
        for c in range(4):
            v = sc[:, c]
            print(f" dir {CL[c][:4]}: mean proj by true class", [round(v[Y == k].mean().item(), 2) for k in range(4)],
                  "std", [round(v[Y == k].std().item(), 2) if (Y == k).sum() > 1 else 0 for k in range(4)])
