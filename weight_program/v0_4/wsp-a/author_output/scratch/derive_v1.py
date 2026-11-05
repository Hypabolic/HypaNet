"""Derive edit v1 constants from fwd calls 2-11 (sets B, C, D; original model).

Mechanisms:
  OSC neuron  (layer 23 MLP, neuron 1046): gate = edit-type repetition score, up = no-progress score, down = looping wire.
  STAL neuron (layer 22 MLP, neuron 1613): gate = repeated-failure/long-chain score, up = no-resolution/no-progress/non-edit
              score, down = stalled wire projected into the null space of layer-23 value projection.
  PROD rows   (label rows): productive row += beta * progress direction / (1+w_final); bias via constant dim.
All directions are means of matched-pair differences (set C: ERR, PROG, REP, RES; set D: EDITS, EDITREPfree, CHAINpoll)
recomputed on the exact MLP-input residual of the host layer; standardisation stats over confident windows of sets B, D.
Writes output/wsp-a/edit_constants.safetensors and output/wsp-a/scratch/v1_params.json.
"""
import json
import numpy as np
import torch
from safetensors.torch import save_file
from wsp import load, load_original, label_rows, ARCH

torch.set_num_threads(1)
CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
P = "model.language_model."
EPS = 1e-6


def unit_rms(x):
    x = torch.as_tensor(np.asarray(x)).double()
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


# ------------------------------------------------------------------ load all windows of sets B, C, D
down22 = load_original(P + "layers.22.mlp.down_proj.weight").double()
down23 = load_original(P + "layers.23.mlp.down_proj.weight").double()
W = []  # dicts per window: x22 (mlp-22 input), x23 (mlp-23 input), x24, y, set, sid, t, feat/sign/group/tag


def add_call(call, metaf, setname):
    resp = json.load(open(f"output/wsp-a/fwd/{call}/response.json"))
    meta = {m["id"]: m for m in json.load(open(metaf))}
    for si, s in enumerate(resp["sessions"]):
        m = meta[s["id"]]
        a = lambda k: np.load(f"output/wsp-a/fwd/{call}/{resp['arrays'][f'original.s{si}.{k}']['file']}")
        h23, h24 = torch.tensor(a("hidden.L23")).double(), torch.tensor(a("hidden.L24")).double()
        m22, m23 = torch.tensor(a("mlp_act.L22")).double(), torch.tensor(a("mlp_act.L23")).double()
        x23 = h24 - m23 @ down23.T          # exact input residual of layer-23 MLP (post-attention)
        x22 = h23 - m22 @ down22.T          # exact input residual of layer-22 MLP
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
print("windows:", len(W), "confident:", sum(1 for w in W if w["y"] >= 0))

X = {k: torch.stack([w[k] for w in W]) for k in ("x22", "x23", "x24")}
Xn = {k: unit_rms(v) for k, v in X.items()}
Y = np.array([w["y"] for w in W])
SET = np.array([w["set"] for w in W])
LAST = np.array([w["last"] for w in W])
idx_by_sid_last = {w["sid"]: i for i, w in enumerate(W) if w["last"]}


# ------------------------------------------------------------------ directions at each space
def pair_dir(space, feat):
    deltas = []
    pairs = sorted({w["pair"] for w in W if w["set"] == "C" and w["feat"] == feat})
    for pid in pairs:
        deltas.append(Xn[space][idx_by_sid_last[pid + "p"]] - Xn[space][idx_by_sid_last[pid + "n"]])
    d = torch.stack(deltas).mean(0)
    return d / d.norm()


def group_dir(space, pos, neg, kind_filter):
    deltas = []
    groups = {}
    for w in W:
        if w["set"] == "D" and w["last"]:
            groups.setdefault(w["group"], {})[w["tag"]] = idx_by_sid_last[w["sid"]]
    for g, tags in groups.items():
        if kind_filter(tags) and pos in tags and neg in tags:
            deltas.append(Xn[space][tags[pos]] - Xn[space][tags[neg]])
    d = torch.stack(deltas).mean(0)
    return d / d.norm(), len(deltas)


DIRS = {}
for space in ("x22", "x23", "x24"):
    for f in ("ERR", "PROG", "REP", "RES"):
        DIRS[(space, f)] = pair_dir(space, f)
    DIRS[(space, "EDITS")], n1 = group_dir(space, "osc", "stal5", lambda t: set(t) == {"osc", "stal5", "stal6"})
    DIRS[(space, "EDITREPfree")], n2 = group_dir(space, "osc", "stal", lambda t: set(t) == {"osc", "stal"})
    DIRS[(space, "CHAINpoll")], n3 = group_dir(space, "stal", "reco", lambda t: set(t) == {"stal", "reco"} and len(t) == 2)
print("group counts EDITS/EDITREPfree/CHAIN:", n1, n2, n3)

# standardisation stats over confident windows of sets B and D (not C, whose windows are mostly productive)
stat_mask = (Y >= 0) & ((SET == "B") | (SET == "D"))


def zstats(space, f):
    s = Xn[space][stat_mask] @ DIRS[(space, f)]
    return float(s.mean()), float(s.std())


def class_means(space, f):
    s = (Xn[space] @ DIRS[(space, f)]).numpy()
    return [float(s[stat_mask & (Y == c)].mean()) for c in range(4)], [float(s[stat_mask & (Y == c)].std()) for c in range(4)]


for space in ("x22", "x23", "x24"):
    print(f"--- {space}: feature class means (PROD RECO STAL OSCI) and pooled mu/sigma")
    for f in ("ERR", "PROG", "REP", "RES", "EDITS", "EDITREPfree", "CHAINpoll"):
        mu, sd = class_means(space, f)
        m0, s0 = zstats(space, f)
        print(f"  {f:12s} means={np.round(mu, 2)} stds={np.round(sd, 2)}  pooled {m0:.2f}/{s0:.2f}")

# constant dims for thresholds: dim 0 at x22/x23 (massive, stable), dim 16 at x24
for space, dim in (("x22", 0), ("x23", 0), ("x24", 16)):
    v = Xn[space][:, dim]
    print(f"{space} dim{dim}: mean {v.mean():.3f} std {v.std():.3f} min {v.min():.3f} max {v.max():.3f}")
C0 = {"x22": float(Xn["x22"][:, 0].mean()), "x23": float(Xn["x23"][:, 0].mean()), "x24": float(Xn["x24"][:, 16].mean())}
CDIM = {"x22": 0, "x23": 0, "x24": 16}


def zvec(space, f):
    """vector v and constant k such that v.x + k*(const dim) = (F.x - mu)/sigma  (standardised feature)."""
    m0, s0 = zstats(space, f)
    return DIRS[(space, f)] / s0, -m0 / s0


def combo(space, terms):
    """standardised linear combination sum_i c_i z_i  -> (direction vector, constant offset)."""
    v = torch.zeros(1024, dtype=torch.float64)
    k = 0.0
    for f, c in terms:
        vf, kf = zvec(space, f)
        v = v + c * vf
        k = k + c * kf
    return v, k


def with_const(space, v, k):
    """embed the constant k through the constant dim: v' such that v'.x ~= v.x + k."""
    v = v.clone()
    v[CDIM[space]] += k / C0[space]
    return v


def score(space, v, k):
    return (Xn[space] @ v + k).numpy()


def report(name, s):
    parts = [f"{CL[c][:4]} {s[stat_mask & (Y == c)].mean():6.2f}±{s[stat_mask & (Y == c)].std():4.2f}" for c in range(4)]
    print(f"  {name:18s}: " + " | ".join(parts))


def silu(x):
    return x / (1.0 + np.exp(-x))


# ------------------------------------------------------------------ readout wiring
w_fin = load(P + "norm.weight").double()
E = label_rows(original=True).double()
R = E * (1 + w_fin)  # effective readout directions in unit-RMS space of x24
rms24 = torch.sqrt(X["x24"].pow(2).mean(-1) + EPS)
print("rms(x24) mean/std:", float(rms24.mean()), float(rms24.std()))


def wire(c, Rows):
    """component of row c orthogonal to the other three rows (Gram-Schmidt), unit length; and its gain."""
    others = torch.stack([Rows[j] for j in range(4) if j != c])
    Q, _ = torch.linalg.qr(others.T)
    r = Rows[c] - Q @ (Q.T @ Rows[c])
    u = r / r.norm()
    gain = float(Rows[c] @ u)  # logit_c change per unit of u added to x24 (before /rms)
    return u, gain


# ------------------------------------------------------------------ PROD label-row mechanism (defines final rows)
# productive score: progress + resolution - repeated failures (standardised), beta scaled; bias via dim 16
v_prod, k_prod = combo("x24", [("PROG", 1.0), ("RES", 1.0), ("REP", -1.0)])
s_prod = score("x24", v_prod, k_prod)
print("PROD score (standardised sum):")
report("prod score", s_prod)
BETA_PROD = 1.6   # logits per unit of standardised score: typical productive window (+2.5) gains ~+4
BIAS_PROD = 0.0
v_prod_full = with_const("x24", BETA_PROD * v_prod, BETA_PROD * k_prod + BIAS_PROD)
dE_prod = v_prod_full / (1 + w_fin)
E_new = E.clone()
E_new[0] = E[0] + dE_prod
R_new = E_new * (1 + w_fin)
print("PROD row: |orig|=%.3f |new|=%.3f |delta|=%.3f cos(new,orig)=%.3f" % (E[0].norm(), E_new[0].norm(), dE_prod.norm(), float(E[0] @ E_new[0] / E[0].norm() / E_new[0].norm())))
pred_dlogit_prod = (torch.tensor(s_prod) * BETA_PROD + BIAS_PROD) / rms24 * 1.0  # approx: (v.x_hat)/rms? no: logit = R.x_hat exactly
# exact: logit_c = R_c . x24_hat  (x24_hat unit-RMS)  since final norm = x/rms*(1+w) and E.(..)= (E*(1+w)).x_hat
dl = (Xn["x24"] @ v_prod_full).numpy()
report("PROD dlogit", dl)

# wires for STAL and OSC from the final rows
u_stal, g_stal = wire(2, R_new)
u_osc, g_osc = wire(3, R_new)
print(f"wire gains (logit per unit added to x24_hat): STAL {g_stal:.3f} OSC {g_osc:.3f}")
# note: adding delta to the raw residual x24 changes x24_hat by delta/rms24 (rms24 ~ 0.39)

# ------------------------------------------------------------------ OSC neuron (layer 23)
w_post23 = load_original(P + "layers.23.post_attention_layernorm.weight").double()
v_g, k_g = combo("x23", [("EDITS", 1.0), ("EDITREPfree", 1.0)])
s_g = score("x23", v_g, k_g)
v_u, k_u = combo("x23", [("PROG", -1.0), ("RES", -1.0)])
s_u = score("x23", v_u, k_u)
print("OSC neuron components:")
report("gate score e", s_g)
report("up score t", s_u)
A_OSC, TH_OSC, UP_OFF_OSC = 2.0, 0.9, 1.0   # gate = silu(A*(e - TH)); up = t + UP_OFF
pre = A_OSC * (s_g - TH_OSC)
act = silu(pre) * (s_u + UP_OFF_OSC)
report("act", act)
S_OSC = 0.12  # down-column scale (units of x24 per unit act)
dlog = act * S_OSC * g_osc / rms24.numpy()
report("pred dlogit OSC", dlog)
gate_row23 = with_const("x23", A_OSC * v_g, -A_OSC * (TH_OSC - k_g)) / (1 + w_post23)
up_row23 = with_const("x23", v_u, k_u + UP_OFF_OSC) / (1 + w_post23)
down_col23 = S_OSC * u_osc
# sanity: recompute act from rows applied to the actual normalised MLP input z = x_hat*(1+w_post)
z23 = Xn["x23"] * (1 + w_post23)
act_chk = silu((z23 @ gate_row23).numpy()) * (z23 @ up_row23).numpy()
print("  act check max abs diff:", float(np.abs(act_chk - act).max()))

# ------------------------------------------------------------------ STAL neuron (layer 22)
w_post22 = load_original(P + "layers.22.post_attention_layernorm.weight").double()
v_g2, k_g2 = combo("x22", [("REP", 1.0), ("CHAINpoll", 1.0)])
s_g2 = score("x22", v_g2, k_g2)
v_u2, k_u2 = combo("x22", [("RES", -1.0), ("PROG", -1.0), ("EDITS", -1.0)])
s_u2 = score("x22", v_u2, k_u2)
print("STAL neuron components:")
report("gate score r", s_g2)
report("up score q", s_u2)
A_STAL, TH_STAL, UP_OFF_STAL = 2.0, 0.6, 0.5
pre2 = A_STAL * (s_g2 - TH_STAL)
act2 = silu(pre2) * (s_u2 + UP_OFF_STAL)
report("act", act2)
# wire into null space of layer-23 value projection (so firing at log positions cannot reach the readout via values)
V23 = load_original(P + "layers.23.self_attn.v_proj.weight").double()
w_in23 = load_original(P + "layers.23.input_layernorm.weight").double()
Vt = V23 * (1 + w_in23)[None, :]
Uv, Sv, Vh = torch.linalg.svd(Vt, full_matrices=True)
null_basis = Vh[512:]  # rows: orthonormal basis of null(Vt)
u_stal_null = null_basis.T @ (null_basis @ u_stal)
print(f"  STAL wire: |proj onto null(V23)|={u_stal_null.norm():.3f}; residual |Vt u|={float((Vt @ u_stal_null).norm()):.2e}")
u_stal_null = u_stal_null / u_stal_null.norm()
g_stal_null = float(R_new[2] @ u_stal_null)
leak = [float(R_new[c] @ u_stal_null) for c in range(4)]
print(f"  gains of null-space wire on the four rows: {np.round(leak, 3)}")
S_STAL = 0.12
dlog2 = act2 * S_STAL * g_stal_null / rms24.numpy()
report("pred dlogit STAL", dlog2)
gate_row22 = with_const("x22", A_STAL * v_g2, -A_STAL * (TH_STAL - k_g2)) / (1 + w_post22)
up_row22 = with_const("x22", v_u2, k_u2 + UP_OFF_STAL) / (1 + w_post22)
down_col22 = S_STAL * u_stal_null
z22 = Xn["x22"] * (1 + w_post22)
act_chk2 = silu((z22 @ gate_row22).numpy()) * (z22 @ up_row22).numpy()
print("  act check max abs diff:", float(np.abs(act_chk2 - act2).max()))

# ------------------------------------------------------------------ predicted decisions (direct path only)
logits = np.stack([w["logits"] for w in W])
pred_logits = logits.copy()
pred_logits[:, 0] += dl
pred_logits[:, 3] += dlog
pred_logits[:, 2] += dlog2
for name, lg in (("original", logits), ("predicted v1", pred_logits)):
    cm = np.zeros((4, 4), int)
    for i in np.where(stat_mask)[0]:
        cm[Y[i], int(np.argmax(lg[i]))] += 1
    print(f"{name} confusion on B+D confident windows:\n{cm}\n  acc={np.trace(cm) / cm.sum():.3f}")

# ------------------------------------------------------------------ save constants
consts = {
    "prod_row_delta": dE_prod.float(),
    "gate_row_L23": gate_row23.float(), "up_row_L23": up_row23.float(), "down_col_L23": down_col23.float(),
    "gate_row_L22": gate_row22.float(), "up_row_L22": up_row22.float(), "down_col_L22": down_col22.float(),
}
save_file(consts, "output/wsp-a/edit_constants.safetensors")
params = {"neuron_L23": 1046, "neuron_L22": 1613, "BETA_PROD": BETA_PROD, "BIAS_PROD": BIAS_PROD,
          "A_OSC": A_OSC, "TH_OSC": TH_OSC, "UP_OFF_OSC": UP_OFF_OSC, "S_OSC": S_OSC,
          "A_STAL": A_STAL, "TH_STAL": TH_STAL, "UP_OFF_STAL": UP_OFF_STAL, "S_STAL": S_STAL,
          "C0": C0, "gains": {"stal_null": g_stal_null, "osc": g_osc}, "rms24": float(rms24.mean())}
json.dump(params, open("output/wsp-a/scratch/v1_params.json", "w"), indent=1)
print("saved constants:", {k: tuple(v.shape) for k, v in consts.items()})
