"""Set H directions (NF1, NF2, REPF, OSCOK, RESH) and the WSP-A directions recomputed, all in the exact MLP-input
spaces x21, x22, x23 and in x24; LOO consistency; class projections on set R (realistic). Saves dirs_b.npz."""
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


# ---- WSP-B sets H and R
H, Rw = {}, []
for call, name in ((14, "h01"), (15, "h02"), (16, "h03"), (17, "r01"), (18, "r02")):
    d = f"output/wsp-b/fwd/{call}"
    resp = json.load(open(f"{d}/response.json"))
    meta = {m["id"]: m for m in json.load(open(f"output/wsp-b/scratch/meta_{name}.json"))}
    for si, s in enumerate(resp["sessions"]):
        sp = spaces(d, si, resp)
        m = meta[s["id"]]
        for wi in range(2):
            e = {k: v[wi] for k, v in sp.items()}
            e["logits"] = np.array(s["original"]["label_logits"][wi]); e["last"] = wi == 1; e["id"] = s["id"]
            if name.startswith("h"):
                e.update({"pair": m["pair"], "feat": m["feat"], "sign": m["sign"]})
                H.setdefault(s["id"], {})[wi] = e
            else:
                e["label"] = m["label"]; Rw.append(e)
SPACES = ("x21", "x22", "x23", "x24")
DIRS = {}
print("=== set H directions (last window; LOO over pairs)")
for feat in ("NF1", "NF2", "REPF", "OSCOK", "RESH"):
    pairs = sorted({e[1]["pair"] for e in H.values() if e[1]["feat"] == feat})
    for sp in SPACES:
        D = torch.stack([H[p + "p"][1][sp] - H[p + "n"][1][sp] for p in pairs])
        d = D.mean(0); dn = d / d.norm()
        loo = np.array([float(D[i] @ ((D.sum(0) - D[i]) / (len(D) - 1)) / ((D.sum(0) - D[i]) / (len(D) - 1)).norm()) for i in range(len(D))])
        Dp = torch.stack([H[p + "p"][0][sp] - H[p + "n"][0][sp] for p in pairs])
        if sp == "x23":
            print(f"  {feat:6s} n={len(D)} |mean|={d.norm():.2f} proj {float((D @ dn).mean()):.2f}±{float((D @ dn).std()):.2f} LOO {loo.mean():.2f}±{loo.std():.2f} agree={(loo > 0).mean():.2f} prev-window agree={(Dp @ dn > 0).float().mean():.2f}")
        DIRS[(sp, feat)] = dn
# NF = mean of NF1 and NF2 directions (both are 'new facts present'); keep separately too
for sp in SPACES:
    v = DIRS[(sp, "NF1")] + DIRS[(sp, "NF2")]; DIRS[(sp, "NF")] = v / v.norm()

# ---- WSP-A directions recomputed from the archived data (sets C, D, G), in the same spaces
W = []


def add_a(call, metaf, setname):
    d = f"output/wsp-a/fwd/{call}"
    resp = json.load(open(f"{d}/response.json"))
    meta = {m["id"]: m for m in json.load(open(metaf))}
    for si, s in enumerate(resp["sessions"]):
        m = meta[s["id"]]
        a = lambda k: torch.tensor(np.load(f"{d}/{resp['arrays'][f'original.s{si}.{k}']['file']}")).double()
        h23, h24 = a("hidden.L23"), a("hidden.L24"); m22, m23 = a("mlp_act.L22"), a("mlp_act.L23")
        sp = {"x22": unit_rms(h23 - m22 @ down[22].T), "x23": unit_rms(h24 - m23 @ down[23].T), "x24": unit_rms(h24)}
        try:
            sp["h22"] = unit_rms(a("hidden.L22"))
        except KeyError:
            pass
        n = len(m["labels"])
        for wi, t in enumerate(s["windows"]):
            if t == n - 1:
                W.append({"sid": s["id"], "set": setname, "feat": m.get("feat"), "sign": m.get("sign"), "pair": m.get("pair"), "group": m.get("group"), "tag": m.get("tag"), **{k: v[wi] for k, v in sp.items()}})


for c in (5, 6, 7, 8):
    add_a(c, f"output/wsp-a/scratch/meta_c{c - 4:02d}.json", "C")
for c in (9, 10, 11):
    add_a(c, f"output/wsp-a/scratch/meta_d{c - 8:02d}.json", "D")
for c in (24, 25):
    add_a(c, f"output/wsp-a/scratch/meta_g{c - 23:02d}.json", "G")
last = {w["sid"]: w for w in W}
for sp in ("x22", "x23", "x24", "h22"):
    for feat, setname in (("ERR", "C"), ("PROG", "C"), ("REP", "C"), ("RES", "C"), ("EDITCOUNT", "G")):
        pairs = sorted({w["pair"] for w in W if w["set"] == setname and w["feat"] == feat and sp in w})
        d = torch.stack([last[p + "p"][sp] - last[p + "n"][sp] for p in pairs]).mean(0); DIRS[(sp, feat)] = d / d.norm()
    groups = {}
    for w in W:
        if w["set"] == "D" and sp in w:
            groups.setdefault(w["group"], {})[w["tag"]] = w
    for feat, pos, neg, kind in (("EDITS", "osc", "stal5", {"osc", "stal5", "stal6"}), ("EDITREPfree", "osc", "stal", {"osc", "stal"}), ("CHAIN", "stal", "reco", {"stal", "reco"})):
        d = torch.stack([t[pos][sp] - t[neg][sp] for t in groups.values() if set(t) == kind]).mean(0); DIRS[(sp, feat)] = d / d.norm()
# x21 versions of the old directions: not available (no x21 in WSP-A data); use h22-derived for layer 21 as before.
FE = ["NF", "NF1", "NF2", "REPF", "OSCOK", "RESH", "ERR", "PROG", "REP", "RES", "EDITS", "EDITREPfree", "CHAIN", "EDITCOUNT"]
print("=== cos between directions at x23 (new vs old)")
M = torch.stack([DIRS[("x23", f)] for f in FE])
G = (M @ M.T).numpy()
print("        " + " ".join(f"{f[:5]:>6s}" for f in FE))
for i, f in enumerate(FE):
    print(f"{f[:7]:8s}" + " ".join(f"{G[i, j]:6.2f}" for j in range(len(FE))))
# ---- class projections on set R (both windows), standardised over R
Y = np.array([CL.index(e["label"]) for e in Rw])
print("=== set R: original-model confusion (last windows)")
cm = np.zeros((4, 4), int)
for e in Rw:
    if e["last"]:
        cm[CL.index(e["label"]), int(np.argmax(e["logits"]))] += 1
print(cm, " acc", np.trace(cm) / cm.sum())
print("mean original logits by class (last windows):")
for c in range(4):
    print(f"  {CL[c][:4]} {np.round(np.mean([e['logits'] for e in Rw if e['last'] and e['label'] == CL[c]], 0), 2)}")
stats = {}
for sp in ("x23", "x22", "x24"):
    print(f"=== set R class projections at {sp} (mean±std; d' one-vs-rest)")
    X = torch.stack([e[sp] for e in Rw])
    for f in FE:
        if (sp, f) not in DIRS:
            continue
        s = (X @ DIRS[(sp, f)]).numpy()
        stats[(sp, f)] = (float(s.mean()), float(s.std()))
        parts = [f"{CL[c][:4]} {s[Y == c].mean():5.2f}±{s[Y == c].std():4.2f}[{(s[Y == c].mean() - s[Y != c].mean()) / np.sqrt(0.5 * (s[Y == c].var() + s[Y != c].var())):5.2f}]" for c in range(4)]
        print(f"  {f:12s} " + " | ".join(parts))
np.savez("output/wsp-b/scratch/dirs_b.npz", **{f"{sp}_{f}": DIRS[(sp, f)].float().numpy() for (sp, f) in DIRS})
json.dump({f"{k[0]}_{k[1]}": v for k, v in stats.items()}, open("output/wsp-b/scratch/stats_r.json", "w"), indent=1)
print("saved dirs_b.npz, stats_r.json")
