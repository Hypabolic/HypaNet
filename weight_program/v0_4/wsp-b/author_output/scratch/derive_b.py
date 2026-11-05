"""WSP-B revision derivation (usage: ./py derive_b.py PARAMS.json).

Spaces (unit-RMS residual at the exact MLP input): x21 = hidden22 - down21 @ act21, x22 = hidden23 - down22 @ act22,
x23 = hidden24 - down23 @ act23; x24 = hidden24 (label rows). Directions:
  new (identical-history pairs, calls 19-23, dirs_k.npz): NFR (reads present), NF1 (distinct vs repeated reads), REPF
  (identical failing test repeated), REPH (identical http error repeated), REPP (identical poll pending repeated),
  REPX (REPF+REPH+REPP), PEND (pending polls present), OSCOK (same-file A/B edits vs fresh edits), RESH (retries then ok);
  kept from WSP-A (dirs_b.npz, from calls 5-11 of WSP-A): EDITS, EDITREPfree (edit repetition).
Standardisation (mu, sigma) over the 128 windows of set R (calls 17-18, realistic sessions, balanced classes).
Constants ride on the mean direction m_hat of each space (computed on set R).

Mechanisms (all read z-scores; silu gates; outputs along readout wires orthogonal to the other label rows):
  PROD rows   : E_prod += BIAS_PROD + A_NFR z_NFR - A_REPX z_REPX (carried in x24 space, divided by (1+w_final)).
  OSC  (L23)  : gate = silu(A_OSC (e - TH_OSC)), e = z_EDITS + z_EDITREPfree;  up = e + W_OK z_OSCOK - W_NF z_NFR + OFF_OSC.
  STAL (L22)  : gate = silu(A_STAL (r - TH_STAL)), r = z_REPF + W_RX z_REPX;  up = -W_NFS z_NFR - z_EDITS - z_PEND + OFF_STAL;
                wire projected into null(v_proj_23 (1+ln_23)).
  RECO (L21)  : gate = silu(A_RECO (g - TH_RECO)), g = z_PEND + z_REPH + z_REPP + W_RS z_RESH;  up = z_NFR + W_RS2 z_RESH + OFF_RECO.
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
down = {L: load_original(f"{P}layers.{L}.mlp.down_proj.weight").double() for L in (21, 22, 23)}


def unit_rms(x):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


# ---------------- set R windows (realistic, labelled by construction)
Rw = []
for call, name in ((17, "r01"), (18, "r02")):
    d = f"output/wsp-b/fwd/{call}"
    resp = json.load(open(f"{d}/response.json"))
    meta = {m["id"]: m for m in json.load(open(f"output/wsp-b/scratch/meta_{name}.json"))}
    for si, s in enumerate(resp["sessions"]):
        a = lambda k: torch.tensor(np.load(f"{d}/{resp['arrays'][f'original.s{si}.{k}']['file']}")).double()
        h = {L: a(f"hidden.L{L}") for L in (21, 22, 23, 24)}
        m = {L: a(f"mlp_act.L{L}") for L in (21, 22, 23)}
        sp = {"x21": unit_rms(h[22] - m[21] @ down[21].T), "x22": unit_rms(h[23] - m[22] @ down[22].T), "x23": unit_rms(h[24] - m[23] @ down[23].T), "x24": unit_rms(h[24])}
        for wi in range(2):
            Rw.append({**{k: v[wi] for k, v in sp.items()}, "raw24": h[24][wi], "y": CL.index(meta[s["id"]]["label"]), "last": wi == 1, "id": s["id"],
                       "logits": np.array(s["original"]["label_logits"][wi])})
X = {sp: torch.stack([w[sp] for w in Rw]) for sp in ("x21", "x22", "x23", "x24")}
Y = np.array([w["y"] for w in Rw]); LAST = np.array([w["last"] for w in Rw])
rms24 = torch.sqrt(torch.stack([w["raw24"] for w in Rw]).pow(2).mean(-1) + EPS).numpy()
dk = np.load("output/wsp-b/scratch/dirs_k.npz"); db = np.load("output/wsp-b/scratch/dirs_b.npz")


def D(sp, f):
    key = f"{sp}_{f}"
    if key in dk.files:
        return torch.tensor(dk[key]).double()
    if key in db.files:
        return torch.tensor(db[key]).double()
    raise KeyError(key)


MH, CM, ZS = {}, {}, {}
for sp in X:
    m = X[sp].mean(0); MH[sp] = m / m.norm(); CM[sp] = float((X[sp] @ MH[sp]).mean())


def zvec(sp, f):
    if (sp, f) not in ZS:
        s = X[sp] @ D(sp, f); ZS[(sp, f)] = (float(s.mean()), float(s.std()))
    m0, s0 = ZS[(sp, f)]
    return D(sp, f) / s0, -m0 / s0


def combo(sp, terms):
    v = torch.zeros(1024, dtype=torch.float64); k = 0.0
    for f, c in terms:
        if c == 0.0:
            continue
        vf, kf = zvec(sp, f); v = v + c * vf; k = k + c * kf
    return v, k


def with_const(sp, v, k):
    return v + (k / CM[sp]) * MH[sp]


def score(sp, v, k):
    return (X[sp] @ v + k).numpy()


lines = []


def report(name, s):
    parts = []
    for c in range(4):
        a = s[Y == c]; q = np.percentile(a, [10, 50, 90])
        parts.append(f"{CL[c][:4]} {a.mean():6.2f} ({q[0]:5.2f}/{q[1]:5.2f}/{q[2]:5.2f})")
    line = f"  {name:16s}: " + " | ".join(parts); print(line); lines.append(line)


def silu(x):
    return x / (1.0 + np.exp(-x))


w_fin = load(P + "norm.weight").double()
E = label_rows(original=True).double()


def wire(c, Rows):
    others = torch.stack([Rows[j] for j in range(4) if j != c])
    Q, _ = torch.linalg.qr(others.T)
    r = Rows[c] - Q @ (Q.T @ Rows[c])
    return r / r.norm()


# ---------------- label rows (PROD default)
v_p, k_p = combo("x24", [("NFR", PARAMS["A_NFR"]), ("REPX", -PARAMS["A_REPX"])])
dE = torch.zeros(4, 1024, dtype=torch.float64)
dE[0] = with_const("x24", v_p, k_p + PARAMS["BIAS_PROD"]) / (1 + w_fin)
for c in range(1, 4):
    b = PARAMS.get("BIAS_ALL", [0, 0, 0, 0])[c]
    if b:
        dE[c] = with_const("x24", torch.zeros(1024, dtype=torch.float64), b) / (1 + w_fin)
E_new = E + dE
R_new = E_new * (1 + w_fin)
dl = (X["x24"] @ (R_new - E * (1 + w_fin)).T).numpy()
print("label rows:"); report("PROD dlogit", dl[:, 0])
u_osc = wire(3, R_new); g_osc = float(R_new[3] @ u_osc)
u_stal = wire(2, R_new); u_reco = wire(1, R_new); g_reco = float(R_new[1] @ u_reco)

# ---------------- OSC neuron (layer 23, unit 1046)
w_post23 = load_original(P + "layers.23.post_attention_layernorm.weight").double()
v_e, k_e = combo("x23", [("EDITS", 1.0), ("EDITREPfree", 1.0)])
v_u, k_u = combo("x23", [("EDITS", 1.0), ("EDITREPfree", 1.0), ("OSCOK", PARAMS["W_OK"]), ("NFR", -PARAMS["W_NF"])])
s_e, s_u = score("x23", v_e, k_e), score("x23", v_u, k_u) + PARAMS["OFF_OSC"]
act = silu(PARAMS["A_OSC"] * (s_e - PARAMS["TH_OSC"])) * s_u
dlog = act * PARAMS["S_OSC"] * g_osc / rms24
print("OSC neuron:"); report("gate e", s_e); report("up", s_u); report("act", act); report("pred dlogit OSC", dlog)
gate_row23 = with_const("x23", PARAMS["A_OSC"] * v_e, -PARAMS["A_OSC"] * (PARAMS["TH_OSC"] - k_e)) / (1 + w_post23)
up_row23 = with_const("x23", v_u, k_u + PARAMS["OFF_OSC"]) / (1 + w_post23)
down_col23 = PARAMS["S_OSC"] * u_osc

# ---------------- STAL neuron (layer 22, unit 1613)
w_post22 = load_original(P + "layers.22.post_attention_layernorm.weight").double()
v_r, k_r = combo("x22", [("REPF", 1.0), ("REPX", PARAMS["W_RX"])])
v_q, k_q = combo("x22", [("NFR", -PARAMS["W_NFS"]), ("EDITS", -1.0), ("PEND", -1.0)])
s_r, s_q = score("x22", v_r, k_r), score("x22", v_q, k_q) + PARAMS["OFF_STAL"]
act2 = silu(PARAMS["A_STAL"] * (s_r - PARAMS["TH_STAL"])) * s_q
V23 = load_original(P + "layers.23.self_attn.v_proj.weight").double()
w_in23 = load_original(P + "layers.23.input_layernorm.weight").double()
_, _, Vh = torch.linalg.svd(V23 * (1 + w_in23)[None, :], full_matrices=True)
nb = Vh[512:]
u_stal_null = nb.T @ (nb @ u_stal); u_stal_null = u_stal_null / u_stal_null.norm()
g_stal = float(R_new[2] @ u_stal_null)
dlog2 = act2 * PARAMS["S_STAL"] * g_stal / rms24
print("STAL neuron:"); report("gate r", s_r); report("up q", s_q); report("act", act2); report("pred dlogit STAL", dlog2)
gate_row22 = with_const("x22", PARAMS["A_STAL"] * v_r, -PARAMS["A_STAL"] * (PARAMS["TH_STAL"] - k_r)) / (1 + w_post22)
up_row22 = with_const("x22", v_q, k_q + PARAMS["OFF_STAL"]) / (1 + w_post22)
down_col22 = PARAMS["S_STAL"] * u_stal_null

# ---------------- RECO neuron (layer 21, unit 1007), x21 space
w_post21 = load_original(P + "layers.21.post_attention_layernorm.weight").double()
v_g, k_g = combo("x21", [("PEND", 1.0), ("REPH", 1.0), ("REPP", 1.0), ("RESH", PARAMS["W_RS"])])
v_h, k_h = combo("x21", [("NFR", 1.0), ("RESH", PARAMS["W_RS2"])])
s_g, s_h = score("x21", v_g, k_g), score("x21", v_h, k_h) + PARAMS["OFF_RECO"]
act3 = silu(PARAMS["A_RECO"] * (s_g - PARAMS["TH_RECO"])) * s_h
dlog3 = act3 * PARAMS["S_RECO"] * g_reco / rms24
print("RECO neuron:"); report("gate g", s_g); report("up h", s_h); report("act", act3); report("pred dlogit RECO", dlog3)
gate_row21 = with_const("x21", PARAMS["A_RECO"] * v_g, -PARAMS["A_RECO"] * (PARAMS["TH_RECO"] - k_g)) / (1 + w_post21)
up_row21 = with_const("x21", v_h, k_h + PARAMS["OFF_RECO"]) / (1 + w_post21)
down_col21 = PARAMS["S_RECO"] * u_reco
print("  wire gains: osc %.3f stal(null) %.3f reco %.3f; stal wire on rows %s" % (g_osc, g_stal, g_reco, np.round([float(R_new[c] @ u_stal_null) for c in range(4)], 3)))

# ---------------- predicted decisions on set R (direct path)
logits = np.stack([w["logits"] for w in Rw])
pred = logits + dl; pred[:, 3] += dlog; pred[:, 2] += dlog2; pred[:, 1] += dlog3
for name, lg in (("original", logits), ("predicted", pred)):
    for sel, sn in ((LAST, "last windows"), (np.ones_like(LAST, bool), "all windows")):
        cm = np.zeros((4, 4), int)
        for i in np.where(sel)[0]:
            cm[Y[i], int(np.argmax(lg[i]))] += 1
        line = f"{name} ({sn}): acc={np.trace(cm) / cm.sum():.3f} recall={np.round(np.diag(cm) / cm.sum(1), 2)} precision={np.round(np.diag(cm) / np.maximum(cm.sum(0), 1), 2)}"
        print(line); lines.append(line)
        if sn == "last windows":
            print(cm); lines.append(str(cm))
consts = {"label_row_deltas": dE.float(),
          "gate_row_L23": gate_row23.float(), "up_row_L23": up_row23.float(), "down_col_L23": down_col23.float(),
          "gate_row_L22": gate_row22.float(), "up_row_L22": up_row22.float(), "down_col_L22": down_col22.float(),
          "gate_row_L21": gate_row21.float(), "up_row_L21": up_row21.float(), "down_col_L21": down_col21.float()}
save_file(consts, "output/wsp-b/edit_constants.safetensors")
with open(sys.argv[1].replace(".json", "_report.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
json.dump({"ZS": {f"{k[0]}_{k[1]}": v for k, v in ZS.items()}, "CM": CM}, open(sys.argv[1].replace(".json", "_stats.json"), "w"), indent=1)
print("saved output/wsp-b/edit_constants.safetensors for", sys.argv[1])
