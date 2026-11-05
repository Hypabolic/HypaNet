"""Identical-history pairs (calls 19-23): directions per feature in x21/x22/x23/x24/h22, LOO consistency (last and
previous window), cosines, and class projections on set R (calls 17-18). Saves dirs_k.npz and stats_rk.json."""
import json
import numpy as np
import torch
from wsp import load_original

CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
P = "model.language_model."
down = {L: load_original(f"{P}layers.{L}.mlp.down_proj.weight").double() for L in (21, 22, 23)}


def unit_rms(x):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def spaces(d, si, resp):
    a = lambda k: torch.tensor(np.load(f"{d}/{resp['arrays'][f'original.s{si}.{k}']['file']}")).double()
    h = {L: a(f"hidden.L{L}") for L in (21, 22, 23, 24)}
    m = {L: a(f"mlp_act.L{L}") for L in (21, 22, 23)}
    return {"x21": unit_rms(h[22] - m[21] @ down[21].T), "x22": unit_rms(h[23] - m[22] @ down[22].T), "x23": unit_rms(h[24] - m[23] @ down[23].T), "x24": unit_rms(h[24]), "h22": unit_rms(h[22])}


K, Rw = {}, []
for call, name in ((19, "k01"), (20, "k02"), (21, "k03"), (22, "k04"), (23, "k05"), (17, "r01"), (18, "r02")):
    d = f"output/wsp-b/fwd/{call}"
    resp = json.load(open(f"{d}/response.json"))
    meta = {m["id"]: m for m in json.load(open(f"output/wsp-b/scratch/meta_{name}.json"))}
    for si, s in enumerate(resp["sessions"]):
        sp = spaces(d, si, resp); m = meta[s["id"]]
        for wi in range(2):
            e = {k: v[wi] for k, v in sp.items()}
            e["logits"] = np.array(s["original"]["label_logits"][wi]); e["last"] = wi == 1; e["id"] = s["id"]
            if name.startswith("k"):
                e.update({"pair": m["pair"], "feat": m["feat"], "sign": m["sign"]}); K.setdefault(s["id"], {})[wi] = e
            else:
                e["label"] = m["label"]; Rw.append(e)
SP = ("x21", "x22", "x23", "x24", "h22")
FEATS = ("NF1", "NF2", "NF3", "RDS", "REPF", "REPH", "REPP", "PEND", "OSCOK", "RESH")
DIRS = {}
print("=== identical-history pairs: directions (x23 shown; LOO over 8 pairs; 'prev' = previous window)")
for feat in FEATS:
    pairs = sorted({e[1]["pair"] for e in K.values() if e[1]["feat"] == feat})
    for sp in SP:
        D = torch.stack([K[p + "p"][1][sp] - K[p + "n"][1][sp] for p in pairs])
        Dp = torch.stack([K[p + "p"][0][sp] - K[p + "n"][0][sp] for p in pairs])
        d = D.mean(0); dn = d / d.norm()
        loo = np.array([float(D[i] @ ((D.sum(0) - D[i]) / 7) / ((D.sum(0) - D[i]) / 7).norm()) for i in range(len(D))])
        DIRS[(sp, feat)] = dn
        if sp == "x23":
            print(f"  {feat:6s} |mean|={d.norm():.2f} proj {float((D @ dn).mean()):.2f}±{float((D @ dn).std()):.2f} LOO {loo.mean():.2f}±{loo.std():.2f} agree={(loo > 0).mean():.2f} | prev-window proj {float((Dp @ dn).mean()):.2f} agree={(Dp @ dn > 0).float().mean():.2f}")
# combined 'reads present' direction from NF2 and NF3 (both: reads vs plans/polls/http in their place)
for sp in SP:
    v = DIRS[(sp, "NF2")] + DIRS[(sp, "NF3")]; DIRS[(sp, "NFR")] = v / v.norm()
    v = DIRS[(sp, "REPF")] + DIRS[(sp, "REPH")] + DIRS[(sp, "REPP")]; DIRS[(sp, "REPX")] = v / v.norm()
print("=== cos at x23")
FE = list(FEATS) + ["NFR", "REPX"]
M = torch.stack([DIRS[("x23", f)] for f in FE]); G = (M @ M.T).numpy()
print("        " + " ".join(f"{f[:5]:>6s}" for f in FE))
for i, f in enumerate(FE):
    print(f"{f[:7]:8s}" + " ".join(f"{G[i, j]:6.2f}" for j in range(len(FE))))
# class projections on R
Y = np.array([CL.index(e["label"]) for e in Rw])
stats = {}
for sp in ("x23", "x22", "x21", "x24"):
    X = torch.stack([e[sp] for e in Rw])
    print(f"=== set R class projections at {sp} (mean±std [d' one-vs-rest])")
    for f in FE:
        s = (X @ DIRS[(sp, f)]).numpy(); stats[(sp, f)] = (float(s.mean()), float(s.std()))
        parts = [f"{CL[c][:4]} {s[Y == c].mean():5.2f}±{s[Y == c].std():4.2f}[{(s[Y == c].mean() - s[Y != c].mean()) / np.sqrt(0.5 * (s[Y == c].var() + s[Y != c].var())):5.2f}]" for c in range(4)]
        print(f"  {f:6s} " + " | ".join(parts))
np.savez("output/wsp-b/scratch/dirs_k.npz", **{f"{sp}_{f}": DIRS[(sp, f)].float().numpy() for (sp, f) in DIRS})
json.dump({f"{k[0]}_{k[1]}": v for k, v in stats.items()}, open("output/wsp-b/scratch/stats_rk.json", "w"), indent=1)
# per-R-window table of key projections for later inspection
with open("output/wsp-b/scratch/r_table.txt", "w") as fh:
    for e in Rw:
        fh.write(f"{e['id']:12s} {'last' if e['last'] else 'prev'} {e['label'][:4]} orig={CL[int(np.argmax(e['logits']))][:4]} " +
                 " ".join(f"{f}={float(e['x23'] @ DIRS[('x23', f)]):5.2f}" for f in ("NFR", "NF1", "RDS", "REPX", "PEND", "OSCOK", "RESH")) + "\n")
print("saved")
