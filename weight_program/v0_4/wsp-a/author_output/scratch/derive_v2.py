"""Edit v2 derivation (usage: ./py derive_v2.py PARAMS.json). Revision of derive_final.py by reasoning on set E
(calls 12-14): see ITERATIONS.md.

Spaces: x22 = hidden23 - down22 @ act22 (layer-22 MLP input), x23 = hidden24 - down23 @ act23 (layer-23 MLP input),
h22 = hidden22 (stands in for the layer-21 MLP input, which differs from it by the layer-21 MLP output only),
x24 = hidden24 (label rows). Directions: means of matched-pair differences (set C: ERR, PROG, REP, RES; set D: EDITS,
EDITREPfree, CHAIN). Standardisation over confident windows of sets B and D. Constants ride on the mean direction.

Mechanisms:
  OSC neuron  (layer 23, n=1046): e = z_EDITS + z_EDITREPfree; T = -z_PROG - z_RES + z_REP + z_CHAIN;
              act = silu(A_OSC (e - TH_OSC)) * (T + W_E_OSC * e + OFF_OSC); down = S_OSC * looping wire.
  STAL neuron (layer 22, n=1613): r = z_REP + 2 z_CHAIN; q = -z_RES - z_PROG + z_CHAIN - W_E_STAL (z_EDITS + z_EDITREPfree);
              act = silu(A_STAL (r - TH_STAL)) * (q + OFF_STAL); down = S_STAL * stalled wire in null(V23).
  RECO neuron (layer 21, n=1007): act = silu(A_RECO (z_ERR - TH_RECO)) * (z_RES + OFF_RECO); down = S_RECO * recovering wire.
  PROD rows   (label rows): E_prod += BETA_PROD (z_PROG + z_RES - z_REP) / (1+w_final); per-class biases BIAS_ALL.
"""
import json
import sys
import numpy as np
import torch
from safetensors.torch import save_file
from wsp import load, load_original, label_rows

torch.set_num_threads(1)
PARAMS = json.load(open(sys.argv[1]))
CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
P = "model.language_model."
EPS = 1e-6
SPACES = ("x22", "x23", "x24", "h22")


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
        h22, h23, h24 = (torch.tensor(a(f"hidden.L{L}")).double() for L in (22, 23, 24))
        m22, m23 = torch.tensor(a("mlp_act.L22")).double(), torch.tensor(a("mlp_act.L23")).double()
        x23 = h24 - m23 @ down23.T
        x22 = h23 - m22 @ down22.T
        for wi, t in enumerate(s["windows"]):
            lab = m["labels"][t]
            W.append({"x22": x22[wi], "x23": x23[wi], "x24": h24[wi], "h22": h22[wi], "y": CL.index(lab) if lab != "AMBIG" else -1,
                      "set": setname, "sid": s["id"], "t": t, "last": t == len(m["labels"]) - 1,
                      "feat": m.get("feat"), "sign": m.get("sign"), "pair": m.get("pair"), "group": m.get("group"), "tag": m.get("tag"),
                      "logits": np.array(s["original"]["label_logits"][wi])})


for c in (2, 3, 4):
    add_call(c, f"output/wsp-a/scratch/meta_b{c - 1:02d}.json", "B")
for c in (5, 6, 7, 8):
    add_call(c, f"output/wsp-a/scratch/meta_c{c - 4:02d}.json", "C")
for c in (9, 10, 11):
    add_call(c, f"output/wsp-a/scratch/meta_d{c - 8:02d}.json", "D")
X = {k: torch.stack([w[k] for w in W]) for k in SPACES}
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
for sp in SPACES:
    for f in ("ERR", "PROG", "REP", "RES"):
        DIRS[(sp, f)] = pair_dir(sp, f)
    DIRS[(sp, "EDITS")] = group_dir(sp, "osc", "stal5", {"osc", "stal5", "stal6"})
    DIRS[(sp, "EDITREPfree")] = group_dir(sp, "osc", "stal", {"osc", "stal"})
    DIRS[(sp, "CHAIN")] = group_dir(sp, "stal", "reco", {"stal", "reco"})
MH, CM, ZS = {}, {}, {}
for sp in SPACES:
    m = Xn[sp].mean(0); MH[sp] = m / m.norm(); CM[sp] = float((Xn[sp] @ MH[sp]).mean())
    for f in ("ERR", "PROG", "REP", "RES", "EDITS", "EDITREPfree", "CHAIN"):
        s = Xn[sp][stat_mask] @ DIRS[(sp, f)]
        ZS[(sp, f)] = (float(s.mean()), float(s.std()))


def zvec(sp, f):
    m0, s0 = ZS[(sp, f)]
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


lines = []


def report(name, s):
    parts = []
    for c in range(4):
        a = s[stat_mask & (Y == c)]
        q = np.percentile(a, [10, 50, 90])
        parts.append(f"{CL[c][:4]} {a.mean():6.2f} ({q[0]:5.2f}/{q[1]:5.2f}/{q[2]:5.2f})")
    line = f"  {name:16s}: " + " | ".join(parts)
    print(line); lines.append(line)


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


# ---------------- label rows
v_prod, k_prod = combo("x24", [("PROG", 1.0), ("RES", 1.0), ("REP", -1.0)])
BETA, BIAS_ALL = PARAMS["BETA_PROD"], PARAMS.get("BIAS_ALL", [0.0, 0.0, 0.0, 0.0])
dE = torch.zeros(4, 1024, dtype=torch.float64)
dE[0] = with_const("x24", BETA * v_prod, BETA * k_prod) / (1 + w_fin)
for c in range(4):
    if BIAS_ALL[c] != 0.0:
        dE[c] = dE[c] + with_const("x24", torch.zeros(1024, dtype=torch.float64), BIAS_ALL[c]) / (1 + w_fin)
E_new = E + dE
R_new = E_new * (1 + w_fin)
dl = (Xn["x24"] @ (R_new - E * (1 + w_fin)).T).numpy()
print("label rows:"); report("PROD dlogit", dl[:, 0])
u_osc = wire(3, R_new); g_osc = float(R_new[3] @ u_osc)
u_stal = wire(2, R_new)
u_reco = wire(1, R_new); g_reco = float(R_new[1] @ u_reco)

# ---------------- OSC neuron (layer 23)
w_post23 = load_original(P + "layers.23.post_attention_layernorm.weight").double()
v_e, k_e = combo("x23", [("EDITS", 1.0), ("EDITREPfree", 1.0)])
v_T, k_T = combo("x23", [("PROG", -1.0), ("RES", -1.0), ("REP", 1.0), ("CHAIN", 1.0)])
s_e, s_T = score("x23", v_e, k_e), score("x23", v_T, k_T)
A, TH, WE, OFF, S = PARAMS["A_OSC"], PARAMS["TH_OSC"], PARAMS["W_E_OSC"], PARAMS["OFF_OSC"], PARAMS["S_OSC"]
up_v, up_k = v_T + WE * v_e, k_T + WE * k_e + OFF
s_up = score("x23", up_v, up_k)
act = silu(A * (s_e - TH)) * s_up
dlog = act * S * g_osc / rms24
print("OSC neuron:"); report("gate e", s_e); report("trouble T", s_T); report("up", s_up); report("act", act); report("pred dlogit OSC", dlog)
gate_row23 = with_const("x23", A * v_e, -A * (TH - k_e)) / (1 + w_post23)
up_row23 = with_const("x23", up_v, up_k) / (1 + w_post23)
down_col23 = S * u_osc

# ---------------- STAL neuron (layer 22)
w_post22 = load_original(P + "layers.22.post_attention_layernorm.weight").double()
v_r, k_r = combo("x22", [("REP", 1.0), ("CHAIN", 2.0)])
WES = PARAMS["W_E_STAL"]
v_q, k_q = combo("x22", [("RES", -1.0), ("PROG", -1.0), ("CHAIN", 1.0), ("EDITS", -WES), ("EDITREPfree", -WES)])
s_r, s_q = score("x22", v_r, k_r), score("x22", v_q, k_q)
A2, TH2, OFF2, S2 = PARAMS["A_STAL"], PARAMS["TH_STAL"], PARAMS["OFF_STAL"], PARAMS["S_STAL"]
act2 = silu(A2 * (s_r - TH2)) * (s_q + OFF2)
V23 = load_original(P + "layers.23.self_attn.v_proj.weight").double()
w_in23 = load_original(P + "layers.23.input_layernorm.weight").double()
Vt = V23 * (1 + w_in23)[None, :]
_, _, Vh = torch.linalg.svd(Vt, full_matrices=True)
null_basis = Vh[512:]
u_stal_null = null_basis.T @ (null_basis @ u_stal)
u_stal_null = u_stal_null / u_stal_null.norm()
g_stal_null = float(R_new[2] @ u_stal_null)
dlog2 = act2 * S2 * g_stal_null / rms24
print("STAL neuron:"); report("gate r", s_r); report("up q", s_q); report("act", act2); report("pred dlogit STAL", dlog2)
gate_row22 = with_const("x22", A2 * v_r, -A2 * (TH2 - k_r)) / (1 + w_post22)
up_row22 = with_const("x22", v_q, k_q + OFF2) / (1 + w_post22)
down_col22 = S2 * u_stal_null

# ---------------- RECO neuron (layer 21), derived in h22 space
w_post21 = load_original(P + "layers.21.post_attention_layernorm.weight").double()
v_err, k_err = combo("h22", [("ERR", 1.0)])
v_res, k_res = combo("h22", [("RES", 1.0)])
s_err, s_res = score("h22", v_err, k_err), score("h22", v_res, k_res)
A3, TH3, OFF3, S3 = PARAMS["A_RECO"], PARAMS["TH_RECO"], PARAMS["OFF_RECO"], PARAMS["S_RECO"]
act3 = silu(A3 * (s_err - TH3)) * (s_res + OFF3)
dlog3 = act3 * S3 * g_reco / rms24
print("RECO neuron:"); report("gate zERR", s_err); report("up zRES", s_res); report("act", act3); report("pred dlogit RECO", dlog3)
gate_row21 = with_const("h22", A3 * v_err, -A3 * (TH3 - k_err)) / (1 + w_post21)
up_row21 = with_const("h22", v_res, k_res + OFF3) / (1 + w_post21)
down_col21 = S3 * u_reco
print("  wire gains: osc %.3f stal(null) %.3f reco %.3f; stal wire on rows %s" % (g_osc, g_stal_null, g_reco, np.round([float(R_new[c] @ u_stal_null) for c in range(4)], 3)))

# ---------------- predicted decisions (direct path)
logits = np.stack([w["logits"] for w in W])
pred = logits + dl; pred[:, 3] += dlog; pred[:, 2] += dlog2; pred[:, 1] += dlog3
for name, lg in (("original", logits), ("predicted", pred)):
    for sn in ("B", "D", "BD"):
        msk = stat_mask if sn == "BD" else (stat_mask & (SET == sn))
        cm = np.zeros((4, 4), int)
        for i in np.where(msk)[0]:
            cm[Y[i], int(np.argmax(lg[i]))] += 1
        line = f"{name} set {sn}: acc={np.trace(cm) / cm.sum():.3f} recall={np.round(np.diag(cm) / cm.sum(1), 2)} precision={np.round(np.diag(cm) / np.maximum(cm.sum(0), 1), 2)}"
        print(line); lines.append(line)
        if sn == "BD":
            print(cm); lines.append(str(cm))

consts = {"label_row_deltas": dE.float(),
          "gate_row_L23": gate_row23.float(), "up_row_L23": up_row23.float(), "down_col_L23": down_col23.float(),
          "gate_row_L22": gate_row22.float(), "up_row_L22": up_row22.float(), "down_col_L22": down_col22.float(),
          "gate_row_L21": gate_row21.float(), "up_row_L21": up_row21.float(), "down_col_L21": down_col21.float()}
save_file(consts, "output/wsp-a/edit_constants.safetensors")
with open(sys.argv[1].replace(".json", "_report.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
json.dump({"ZS": {f"{k[0]}_{k[1]}": v for k, v in ZS.items()}, "CM": CM}, open(sys.argv[1].replace(".json", "_stats.json"), "w"), indent=1)
print("saved constants for", sys.argv[1])
