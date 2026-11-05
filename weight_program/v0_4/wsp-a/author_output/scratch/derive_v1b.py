"""Edit v1b constants (revision of derive_v1 by reasoning on per-class percentiles):
  - constants carried on the mean unit-RMS direction m_hat of each space (more stable than dim 0);
  - STAL gate weights the chain-length feature twice (it separates RECOVERY from STALLED best), steeper slope,
    threshold at the point where most STALLED windows pass but few RECOVERY/OSCILLATING ones do;
  - STAL up adds the chain feature; OSC neuron gets a larger up offset and scale.
Writes output/wsp-a/edit_constants.safetensors and output/wsp-a/scratch/v1b_params.json.
"""
import json
import numpy as np
import torch
from safetensors.torch import save_file
from wsp import load, load_original, label_rows

torch.set_num_threads(1)
CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
P = "model.language_model."
EPS = 1e-6


def unit_rms(x):
    x = torch.as_tensor(np.asarray(x)).double()
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


down22 = load_original(P + "layers.22.mlp.down_proj.weight").double()
down23 = load_original(P + "layers.23.mlp.down_proj.weight").double()
W = []


def add_call(call, metaf, setname):
    resp = json.load(open(f"output/wsp-a/fwd/{call}/response.json"))
    meta = {m["id"]: m for m in json.load(open(metaf))}
    for si, s in enumerate(resp["sessions"]):
        m = meta[s["id"]]
        a = lambda k: np.load(f"output/wsp-a/fwd/{call}/{resp['arrays'][f'original.s{si}.{k}']['file']}")
        h23, h24 = torch.tensor(a("hidden.L23")).double(), torch.tensor(a("hidden.L24")).double()
        m22, m23 = torch.tensor(a("mlp_act.L22")).double(), torch.tensor(a("mlp_act.L23")).double()
        x23 = h24 - m23 @ down23.T
        x22 = h23 - m22 @ down22.T
        for wi, t in enumerate(s["windows"]):
            lab = m["labels"][t]
            W.append({"x22": x22[wi], "x23": x23[wi], "x24": h24[wi], "y": CL.index(lab) if lab != "AMBIG" else -1,
                      "set": setname, "sid": s["id"], "t": t, "last": t == len(m["labels"]) - 1,
                      "feat": m.get("feat"), "sign": m.get("sign"), "pair": m.get("pair"), "group": m.get("group"), "tag": m.get("tag"),
                      "logits": np.array(s["original"]["label_logits"][wi])})


for c in (2, 3, 4):
    add_call(c, f"output/wsp-a/scratch/meta_b{c - 1:02d}.json", "B")
for c in (5, 6, 7, 8):
    add_call(c, f"output/wsp-a/scratch/meta_c{c - 4:02d}.json", "C")
for c in (9, 10, 11):
    add_call(c, f"output/wsp-a/scratch/meta_d{c - 8:02d}.json", "D")
X = {k: torch.stack([w[k] for w in W]) for k in ("x22", "x23", "x24")}
Xn = {k: unit_rms(v) for k, v in X.items()}
Y = np.array([w["y"] for w in W])
SET = np.array([w["set"] for w in W])
idx_last = {w["sid"]: i for i, w in enumerate(W) if w["last"]}
stat_mask = (Y >= 0) & ((SET == "B") | (SET == "D"))


def pair_dir(space, feat):
    pairs = sorted({w["pair"] for w in W if w["set"] == "C" and w["feat"] == feat})
    d = torch.stack([Xn[space][idx_last[p + "p"]] - Xn[space][idx_last[p + "n"]] for p in pairs]).mean(0)
    return d / d.norm()


def group_dir(space, pos, neg, kind):
    groups = {}
    for w in W:
        if w["set"] == "D" and w["last"]:
            groups.setdefault(w["group"], {})[w["tag"]] = idx_last[w["sid"]]
    d = torch.stack([Xn[space][t[pos]] - Xn[space][t[neg]] for t in groups.values() if set(t) == kind]).mean(0)
    return d / d.norm()


DIRS = {}
for sp in ("x22", "x23", "x24"):
    for f in ("ERR", "PROG", "REP", "RES"):
        DIRS[(sp, f)] = pair_dir(sp, f)
    DIRS[(sp, "EDITS")] = group_dir(sp, "osc", "stal5", {"osc", "stal5", "stal6"})
    DIRS[(sp, "EDITREPfree")] = group_dir(sp, "osc", "stal", {"osc", "stal"})
    DIRS[(sp, "CHAIN")] = group_dir(sp, "stal", "reco", {"stal", "reco"})

# constant carrier: mean direction of each space; check stability of its projection
MH, CM = {}, {}
for sp in ("x22", "x23", "x24"):
    m = Xn[sp].mean(0); MH[sp] = m / m.norm()
    pr = Xn[sp] @ MH[sp]
    CM[sp] = float(pr.mean())
    print(f"{sp}: m_hat projection mean {pr.mean():.3f} std {pr.std():.3f} min {pr.min():.3f} max {pr.max():.3f}")


def zvec(sp, f):
    s = Xn[sp][stat_mask] @ DIRS[(sp, f)]
    m0, s0 = float(s.mean()), float(s.std())
    return DIRS[(sp, f)] / s0, -m0 / s0


def combo(sp, terms):
    v = torch.zeros(1024, dtype=torch.float64); k = 0.0
    for f, c in terms:
        vf, kf = zvec(sp, f); v = v + c * vf; k = k + c * kf
    return v, k


def with_const(sp, v, k):
    return v + (k / CM[sp]) * MH[sp]


def score(sp, v, k):
    return (Xn[sp] @ v + k).numpy()


def report(name, s, pct=False):
    parts = []
    for c in range(4):
        a = s[stat_mask & (Y == c)]
        if pct:
            q = np.percentile(a, [10, 50, 90])
            parts.append(f"{CL[c][:4]} {q[0]:5.2f}/{q[1]:5.2f}/{q[2]:5.2f}")
        else:
            parts.append(f"{CL[c][:4]} {a.mean():6.2f}±{a.std():4.2f}")
    print(f"  {name:16s}: " + " | ".join(parts))


def silu(x):
    return x / (1.0 + np.exp(-x))


w_fin = load(P + "norm.weight").double()
E = label_rows(original=True).double()
rms24 = torch.sqrt(X["x24"].pow(2).mean(-1) + EPS).numpy()


def wire(c, Rows):
    others = torch.stack([Rows[j] for j in range(4) if j != c])
    Q, _ = torch.linalg.qr(others.T)
    r = Rows[c] - Q @ (Q.T @ Rows[c])
    return r / r.norm()


# ---------------- PROD label row
v_prod, k_prod = combo("x24", [("PROG", 1.0), ("RES", 1.0), ("REP", -1.0)])
s_prod = score("x24", v_prod, k_prod)
print("PROD score percentiles (10/50/90):"); report("prod score", s_prod, pct=True)
BETA_PROD, BIAS_PROD = 1.6, 0.0
v_prod_full = with_const("x24", BETA_PROD * v_prod, BETA_PROD * k_prod + BIAS_PROD)
dE_prod = v_prod_full / (1 + w_fin)
E_new = E.clone(); E_new[0] = E[0] + dE_prod
R_new = E_new * (1 + w_fin)
dl = (Xn["x24"] @ v_prod_full).numpy()
report("PROD dlogit", dl); report("PROD dlogit pct", dl, pct=True)
u_osc = wire(3, R_new); g_osc = float(R_new[3] @ u_osc)
u_stal = wire(2, R_new)

# ---------------- OSC neuron (layer 23, neuron 1046)
w_post23 = load_original(P + "layers.23.post_attention_layernorm.weight").double()
v_g, k_g = combo("x23", [("EDITS", 1.0), ("EDITREPfree", 1.0)])
s_g = score("x23", v_g, k_g)
v_u, k_u = combo("x23", [("PROG", -1.0), ("RES", -1.0)])
s_u = score("x23", v_u, k_u)
print("OSC neuron:"); report("gate e pct", s_g, pct=True); report("up t pct", s_u, pct=True)
A_OSC, TH_OSC, UP_OFF_OSC, S_OSC = 3.0, 1.0, 1.5, 0.22
act = silu(A_OSC * (s_g - TH_OSC)) * (s_u + UP_OFF_OSC)
dlog = act * S_OSC * g_osc / rms24
report("act", act); report("pred dlogit OSC", dlog); report("pred dlogit pct", dlog, pct=True)
gate_row23 = with_const("x23", A_OSC * v_g, -A_OSC * (TH_OSC - k_g)) / (1 + w_post23)
up_row23 = with_const("x23", v_u, k_u + UP_OFF_OSC) / (1 + w_post23)
down_col23 = S_OSC * u_osc
z23 = Xn["x23"] * (1 + w_post23)
act_chk = silu((z23 @ gate_row23).numpy()) * (z23 @ up_row23).numpy()
print("  act check max abs diff:", float(np.abs(act_chk - act).max()), " mean abs:", float(np.abs(act_chk - act).mean()))

# ---------------- STAL neuron (layer 22, neuron 1613)
w_post22 = load_original(P + "layers.22.post_attention_layernorm.weight").double()
v_g2, k_g2 = combo("x22", [("REP", 1.0), ("CHAIN", 2.0)])
s_g2 = score("x22", v_g2, k_g2)
v_u2, k_u2 = combo("x22", [("RES", -1.0), ("PROG", -1.0), ("EDITS", -1.0), ("CHAIN", 1.0)])
s_u2 = score("x22", v_u2, k_u2)
print("STAL neuron:"); report("gate r pct", s_g2, pct=True); report("up q pct", s_u2, pct=True)
A_STAL, TH_STAL, UP_OFF_STAL, S_STAL = 3.0, 2.0, 0.0, 0.14
act2 = silu(A_STAL * (s_g2 - TH_STAL)) * (s_u2 + UP_OFF_STAL)
V23 = load_original(P + "layers.23.self_attn.v_proj.weight").double()
w_in23 = load_original(P + "layers.23.input_layernorm.weight").double()
Vt = V23 * (1 + w_in23)[None, :]
_, _, Vh = torch.linalg.svd(Vt, full_matrices=True)
null_basis = Vh[512:]
u_stal_null = null_basis.T @ (null_basis @ u_stal)
print(f"  STAL wire: |proj onto null(V23)|={u_stal_null.norm():.3f}")
u_stal_null = u_stal_null / u_stal_null.norm()
g_stal_null = float(R_new[2] @ u_stal_null)
print(f"  gains of null-space wire on the four rows: {np.round([float(R_new[c] @ u_stal_null) for c in range(4)], 3)}")
dlog2 = act2 * S_STAL * g_stal_null / rms24
report("act", act2); report("pred dlogit STAL", dlog2); report("pred dlogit pct", dlog2, pct=True)
gate_row22 = with_const("x22", A_STAL * v_g2, -A_STAL * (TH_STAL - k_g2)) / (1 + w_post22)
up_row22 = with_const("x22", v_u2, k_u2 + UP_OFF_STAL) / (1 + w_post22)
down_col22 = S_STAL * u_stal_null
z22 = Xn["x22"] * (1 + w_post22)
act_chk2 = silu((z22 @ gate_row22).numpy()) * (z22 @ up_row22).numpy()
print("  act check max abs diff:", float(np.abs(act_chk2 - act2).max()), " mean abs:", float(np.abs(act_chk2 - act2).mean()))

# ---------------- predicted decisions (direct path)
logits = np.stack([w["logits"] for w in W])
pred = logits.copy(); pred[:, 0] += dl; pred[:, 3] += dlog; pred[:, 2] += dlog2
for name, lg in (("original", logits), ("predicted v1b", pred)):
    cm = np.zeros((4, 4), int)
    for i in np.where(stat_mask)[0]:
        cm[Y[i], int(np.argmax(lg[i]))] += 1
    print(f"{name} confusion on B+D confident windows:\n{cm}\n  acc={np.trace(cm) / cm.sum():.3f}  per-class recall={np.round(np.diag(cm) / cm.sum(1), 2)}")
    # also set B only and set D only
    for sn in ("B", "D"):
        cm = np.zeros((4, 4), int)
        for i in np.where(stat_mask & (SET == sn))[0]:
            cm[Y[i], int(np.argmax(lg[i]))] += 1
        print(f"   set {sn}: acc={np.trace(cm) / cm.sum():.3f} recall={np.round(np.diag(cm) / cm.sum(1), 2)}")

consts = {"prod_row_delta": dE_prod.float(),
          "gate_row_L23": gate_row23.float(), "up_row_L23": up_row23.float(), "down_col_L23": down_col23.float(),
          "gate_row_L22": gate_row22.float(), "up_row_L22": up_row22.float(), "down_col_L22": down_col22.float()}
save_file(consts, "output/wsp-a/edit_constants.safetensors")
params = {"neuron_L23": 1046, "neuron_L22": 1613, "BETA_PROD": BETA_PROD, "BIAS_PROD": BIAS_PROD,
          "A_OSC": A_OSC, "TH_OSC": TH_OSC, "UP_OFF_OSC": UP_OFF_OSC, "S_OSC": S_OSC,
          "A_STAL": A_STAL, "TH_STAL": TH_STAL, "UP_OFF_STAL": UP_OFF_STAL, "S_STAL": S_STAL,
          "CM": CM, "gains": {"stal_null": g_stal_null, "osc": g_osc}, "rms24": float(rms24.mean())}
json.dump(params, open("output/wsp-a/scratch/v1b_params.json", "w"), indent=1)
np.savez("output/wsp-a/scratch/v1b_dirs.npz", **{f"{sp}_{f}": DIRS[(sp, f)].float().numpy() for (sp, f) in DIRS}, **{f"mhat_{sp}": MH[sp].float().numpy() for sp in MH})
print("saved constants")
