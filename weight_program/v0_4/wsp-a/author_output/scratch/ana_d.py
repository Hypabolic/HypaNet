"""Set D analysis: matched-group directions at layers 21-24, class projections (sets B and D), and dead-neuron
candidates at the readout in layers 21-23. Saves output/wsp-a/scratch/dirs_D.npz."""
import json
import sys
import numpy as np
import torch

CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
LAYERS = [20, 21, 22, 23, 24]
MLAYERS = [21, 22, 23]
sess = {}
for call in ("9", "10", "11"):
    resp = json.load(open(f"output/wsp-a/fwd/{call}/response.json"))
    i = {"9": 1, "10": 2, "11": 3}[call]
    meta = {m["id"]: m for m in json.load(open(f"output/wsp-a/scratch/meta_d{i:02d}.json"))}
    for si, s in enumerate(resp["sessions"]):
        m = meta[s["id"]]
        e = {"group": m["group"], "tag": m["tag"], "labels": m["labels"], "logits": np.array(s["original"]["label_logits"])}
        for L in LAYERS:
            e[f"h{L}"] = np.load(f"output/wsp-a/fwd/{call}/{resp['arrays'][f'original.s{si}.hidden.L{L}']['file']}")
        for L in MLAYERS:
            e[f"m{L}"] = np.load(f"output/wsp-a/fwd/{call}/{resp['arrays'][f'original.s{si}.mlp_act.L{L}']['file']}")
        sess[s["id"]] = e


def unit_rms(x):
    x = torch.tensor(np.asarray(x)).double()
    return x / x.pow(2).mean(-1, keepdim=True).sqrt()


def group_kind(gid):
    # identify the d1a/d1b/... kind by the tags present in the group
    tags = sorted(e["tag"] for e in sess.values() if e["group"] == gid)
    return {("osc", "stal5", "stal6"): "d1a", ("osc", "prod", "stal6"): "d1b", ("osc", "prod"): "d1c", ("osc", "stal"): "d1d",
            ("prod", "stal", "stal2"): "d2", ("reco", "stal"): "d3or4", ("reco",): "d5"}[tuple(tags)]


groups = {}
for sid, e in sess.items():
    groups.setdefault(e["group"], {})[e["tag"]] = sid
kinds = {g: group_kind(g) for g in groups}
# d3 vs d4: http vs poll, distinguish by session length (d3 stal has 10 blocks, d4 stal 12)
for g in groups:
    if kinds[g] == "d3or4":
        kinds[g] = "d3" if len(sess[groups[g]["stal"]]["labels"]) == 10 else "d4"
print("group kinds:", {k: sum(1 for g in kinds.values() if g == k) for k in set(kinds.values())})

CONTRASTS = {
    "EDITS": ("d1a", "osc", "stal5"), "HUNKa": ("d1a", "osc", "stal6"), "HUNKb": ("d1b", "osc", "stal6"),
    "OSCPRODb": ("d1b", "osc", "prod"), "REVCYC": ("d1c", "osc", "prod"), "EDITREPfree": ("d1d", "osc", "stal"),
    "REPREAD": ("d2", "stal", "prod"), "REPREAD2": ("d2", "stal2", "prod"), "CHAINhttp": ("d3", "stal", "reco"), "CHAINpoll": ("d4", "stal", "reco"),
}
dirs = {}
for L in LAYERS:
    print(f"=== layer {L}")
    for name, (kind, pos, neg) in CONTRASTS.items():
        deltas = []
        for g, k in kinds.items():
            if k != kind:
                continue
            xp = unit_rms(sess[groups[g][pos]][f"h{L}"])[-1]
            xn = unit_rms(sess[groups[g][neg]][f"h{L}"])[-1]
            deltas.append(xp - xn)
        D = torch.stack(deltas)
        d = D.mean(0); dn = d / d.norm()
        proj = D @ dn
        loo = []
        for i in range(len(D)):
            di = (D.sum(0) - D[i]) / (len(D) - 1)
            loo.append(float(D[i] @ (di / di.norm())))
        loo = np.array(loo)
        print(f"  {name:12s} n={len(D):2d} |mean|={d.norm():.2f} proj {proj.mean():.2f}+-{proj.std():.2f} agree={(proj > 0).float().mean():.2f} LOO {loo.mean():.2f}+-{loo.std():.2f} agree={(loo > 0).mean():.2f}")
        dirs[f"{name}_L{L}"] = dn.float().numpy()
    names = list(CONTRASTS)
    M = torch.tensor(np.stack([dirs[f"{n}_L{L}"] for n in names]))
    if L == 24:
        print("  cos between contrasts (L24):")
        print("   " + " ".join(f"{n[:6]:>6s}" for n in names))
        for i, n in enumerate(names):
            print(f"   {n[:6]:>6s} " + " ".join(f"{float(M[i] @ M[j]):6.2f}" for j in range(len(names))))

# class projections on set B and set D confident windows (L24 and L23)
fd = np.load("output/wsp-a/scratch/feat_dirs.npz", allow_pickle=True)
b = np.load("output/wsp-a/scratch/setB_orig.npz", allow_pickle=True)
for L in (24, 23):
    print(f"=== class projections at L{L}: set B | set D  (mean+-std; one-vs-rest d' in brackets)")
    XB = unit_rms(b[f"h{L}"][b["Y"] >= 0]); yB = b["Y"][b["Y"] >= 0]
    XD, yD = [], []
    for sid, e in sess.items():
        h = unit_rms(e[f"h{L}"])
        for t, lab in enumerate(e["labels"]):
            if lab != "AMBIG":
                XD.append(h[t]); yD.append(CL.index(lab))
    XD = torch.stack(XD); yD = np.array(yD)
    print("  set D class counts:", [int((yD == c).sum()) for c in range(4)])
    allnames = [f"{n}_L{L}" for n in CONTRASTS] + [f"{f}_L{L}" for f in ("ERR", "PROG", "REP", "RES", "REV")]
    for key in allnames:
        v = torch.tensor(dirs[key] if key in dirs else fd[key]).double()
        parts = []
        for X, y in ((XB, yB), (XD, yD)):
            s = (X @ v).numpy()
            sub = []
            for c in range(4):
                a, r = s[y == c], s[y != c]
                sub.append(f"{a.mean():5.2f}±{a.std():3.1f}[{(a.mean() - r.mean()) / np.sqrt(0.5 * (a.var() + r.var())):5.2f}]")
            parts.append(" ".join(sub))
        print(f"  {key[:-4]:12s} B: {parts[0]} | D: {parts[1]}")

# readout residual stats at layers 20-24 (mean direction, dim-0 projection) from set D
stats = {}
for L in LAYERS:
    X = torch.cat([unit_rms(e[f"h{L}"]) for e in sess.values()])
    m = X.mean(0)
    stats[f"mean_L{L}"] = m.float().numpy()
    e0 = X[:, 0]
    print(f"L{L}: |mean|={m.norm():.2f}; dim0 proj {e0.mean():.2f}+-{e0.std():.2f} (min {e0.min():.2f} max {e0.max():.2f}); dim16 {X[:, 16].mean():.2f}+-{X[:, 16].std():.2f}")

# dead neurons at the readout: smallest max|act| over all windows of sets B and D
for L in MLAYERS:
    A = torch.cat([torch.tensor(e[f"m{L}"]) for e in sess.values()])
    if f"m{L}" in b:
        A = torch.cat([A, torch.tensor(b[f"m{L}"])])
    mx = A.abs().max(0).values
    idx = mx.argsort()[:12]
    print(f"L{L} mlp_act: n_windows={len(A)}; quietest neurons (max|act|):", [(int(i), round(float(mx[i]), 4)) for i in idx])
    stats[f"quiet_L{L}"] = idx.numpy()
np.savez("output/wsp-a/scratch/dirs_D.npz", **dirs, **stats)
