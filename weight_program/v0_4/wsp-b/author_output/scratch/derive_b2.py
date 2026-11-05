"""WSP-B revision derivation, hybrid design (usage: ./py derive_b2.py PARAMS.json).

Old terms (WSP-A features ERR, PROG, REP, RES, CHAIN, EDITS, EDITREPfree, EDITCOUNT; directions recomputed in
dirs_b.npz from WSP-A calls 5-11, 24-25; standardisation from WSP-A sets B+D, params_v3b_stats.json) keep the
feature combinations whose behaviour on the disclosed windows is known. New terms (NFR, REPF, REPH, REPP, PEND,
RESH; identical-history pairs, calls 19-23, dirs_k.npz) are standardised on set R (calls 17-18, realistic).
Constants ride on set R's mean direction. Rules (no sweeps): gate threshold = midpoint of the target class's median
gate score and the other classes' pooled median (set R); up offset = 1 - median(up on target windows); scale so that
the target class's median predicted logit gain is TARGET_GAIN (set R). Weights are the only free choices.

  PROD rows : E_prod += BIAS_PROD + A_NFR z_NFR - A_REPX z_REPX   (x24; bias = mean productive deficit on the disclosed windows)
  OSC (L23) : gate e = z_EDITS + z_EDITREPfree (old);  up = W_E e + z_EDITCOUNT(old) - W_NF z_NFR + OFF
  STAL (L22): gate r = z_REP + 2 z_CHAIN (old) + W_RF z_REPF - W_PD z_PEND - W_RH z_REPH;
              up = -z_RES - z_PROG + z_CHAIN - 2 z_EDITS - 2 z_EDITREPfree (old) - W_NFS z_NFR - z_PEND + OFF; wire in null(V23)
  RECO (L21): gate g = z_PEND + z_REPH + z_REPP + W_RS z_RESH (new, x21);  up = z_NFR + W_RS2 z_RESH + OFF
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
OLD_ZS = json.load(open("output/wsp-a/scratch/params_v3b_stats.json"))["ZS"]
OLD = {"ERR", "PROG", "REP", "RES", "CHAIN", "EDITS", "EDITREPfree", "EDITCOUNT"}
MH, CM, ZS = {}, {}, {}
for sp in X:
    mm = X[sp].mean(0); MH[sp] = mm / mm.norm(); CM[sp] = float((X[sp] @ MH[sp]).mean())


def D(sp, f):
    key = f"{sp}_{f}"
    return torch.tensor(dk[key] if key in dk.files else db[key]).double()


def zvec(sp, f):
    if f in OLD:
        m0, s0 = OLD_ZS[f"{sp}_{f}"]
    else:
        if (sp, f) not in ZS:
            s = X[sp] @ D(sp, f); ZS[(sp, f)] = (float(s.mean()), float(s.std()))
        m0, s0 = ZS[(sp, f)]
    return D(sp, f) / s0, -m0 / s0


def combo(sp, terms):
    v = torch.zeros(1024, dtype=torch.float64); k = 0.0
    for f, c in terms:
        if c:
            vf, kf = zvec(sp, f); v = v + c * vf; k = k + c * kf
    return v, k


def with_const(sp, v, k):
    return v + (k / CM[sp]) * MH[sp]


def score(sp, v, k):
    return (X[sp] @ v + k).numpy()


lines = []


def say(line):
    print(line); lines.append(line)


def report(name, s):
    parts = []
    for c in range(4):
        a = s[Y == c]; q = np.percentile(a, [10, 50, 90])
        parts.append(f"{CL[c][:4]} {a.mean():6.2f} ({q[0]:5.2f}/{q[1]:5.2f}/{q[2]:5.2f})")
    say(f"  {name:16s}: " + " | ".join(parts))


def silu(x):
    return x / (1.0 + np.exp(-x))


def threshold(s, target):
    return 0.5 * (np.median(s[Y == target]) + np.median(s[Y != target]))


w_fin = load(P + "norm.weight").double()
E = label_rows(original=True).double()


def wire(c, Rows):
    others = torch.stack([Rows[j] for j in range(4) if j != c])
    Q, _ = torch.linalg.qr(others.T)
    r = Rows[c] - Q @ (Q.T @ Rows[c])
    return r / r.norm()


A_SLOPE, TARGET_GAIN = PARAMS["A_SLOPE"], PARAMS["TARGET_GAIN"]
# ---------------- label rows
v_p, k_p = combo("x24", [("NFR", PARAMS["A_NFR"]), ("REPX", -PARAMS["A_REPX"])])
dE = torch.zeros(4, 1024, dtype=torch.float64)
dE[0] = with_const("x24", v_p, k_p + PARAMS["BIAS_PROD"]) / (1 + w_fin)
E_new = E + dE; R_new = E_new * (1 + w_fin)
dl = (X["x24"] @ (R_new - E * (1 + w_fin)).T).numpy()
say("label rows:"); report("PROD dlogit", dl[:, 0])
u_osc = wire(3, R_new); g_osc = float(R_new[3] @ u_osc)
u_stal = wire(2, R_new); u_reco = wire(1, R_new); g_reco = float(R_new[1] @ u_reco)
V23 = load_original(P + "layers.23.self_attn.v_proj.weight").double()
w_in23 = load_original(P + "layers.23.input_layernorm.weight").double()
_, _, Vh = torch.linalg.svd(V23 * (1 + w_in23)[None, :], full_matrices=True)
nb = Vh[512:]; u_stal_null = nb.T @ (nb @ u_stal); u_stal_null = u_stal_null / u_stal_null.norm(); g_stal = float(R_new[2] @ u_stal_null)
w_post = {L: load_original(f"{P}layers.{L}.post_attention_layernorm.weight").double() for L in (21, 22, 23)}


def neuron(name, sp, target, gate_terms, up_terms, gain):
    v_g, k_g = combo(sp, gate_terms); v_u, k_u = combo(sp, up_terms)
    s_g, s_u = score(sp, v_g, k_g), score(sp, v_u, k_u)
    TH = threshold(s_g, target)
    OFF = 1.0 - float(np.median(s_u[Y == target]))
    act = silu(A_SLOPE * (s_g - TH)) * (s_u + OFF)
    med_act = float(np.median(act[Y == target]))
    S = TARGET_GAIN / (med_act * gain / float(np.median(rms24))) if med_act > 0 else 0.0
    dlog = act * S * gain / rms24
    say(f"{name}: TH={TH:.3f} OFF={OFF:.3f} S={S:.4f} (median target act {med_act:.2f}, gain {gain:.3f})")
    report("gate", s_g); report("up+OFF", s_u + OFF); report("act", act); report("pred dlogit", dlog)
    gate_row = with_const(sp, A_SLOPE * v_g, -A_SLOPE * (TH - k_g)) / (1 + w_post[int(sp[1:])])
    up_row = with_const(sp, v_u, k_u + OFF) / (1 + w_post[int(sp[1:])])
    return gate_row, up_row, S, dlog, {"TH": TH, "OFF": OFF, "S": S}


W = PARAMS
g23, u23, S23, dlog3o, p_osc = neuron("OSC (L23)", "x23", 3,
                                        [("EDITS", 1.0), ("EDITREPfree", 1.0)],
                                        [("EDITS", W["W_E"]), ("EDITREPfree", W["W_E"]), ("EDITCOUNT", 1.0), ("NFR", -W["W_NF"])], g_osc)
g22, u22, S22, dlog2s, p_stal = neuron("STAL (L22)", "x22", 2,
                                         [("REP", 1.0), ("CHAIN", 2.0), ("REPF", W["W_RF"]), ("PEND", -W["W_PD"]), ("REPH", -W["W_RH"])],
                                         [("RES", -1.0), ("PROG", -1.0), ("CHAIN", 1.0), ("EDITS", -2.0), ("EDITREPfree", -2.0), ("NFR", -W["W_NFS"]), ("PEND", -1.0)], g_stal)
g21, u21, S21, dlog1r, p_reco = neuron("RECO (L21)", "x21", 1,
                                         [("PEND", 1.0), ("REPH", 1.0), ("REPP", 1.0), ("RESH", W["W_RS"])],
                                         [("NFR", 1.0), ("RESH", W["W_RS2"])], g_reco)
down_col23, down_col22, down_col21 = S23 * u_osc, S22 * u_stal_null, S21 * u_reco
say("  wire gains: osc %.3f stal(null) %.3f reco %.3f; stal wire on rows %s" % (g_osc, g_stal, g_reco, np.round([float(R_new[c] @ u_stal_null) for c in range(4)], 3)))
logits = np.stack([w["logits"] for w in Rw])
pred = logits + dl; pred[:, 3] += dlog3o; pred[:, 2] += dlog2s; pred[:, 1] += dlog1r
for name, lg in (("original", logits), ("predicted", pred)):
    for sel, sn in ((LAST, "last windows"), (np.ones_like(LAST, bool), "all windows")):
        cm = np.zeros((4, 4), int)
        for i in np.where(sel)[0]:
            cm[Y[i], int(np.argmax(lg[i]))] += 1
        say(f"{name} ({sn}): acc={np.trace(cm) / cm.sum():.3f} recall={np.round(np.diag(cm) / cm.sum(1), 2)} precision={np.round(np.diag(cm) / np.maximum(cm.sum(0), 1), 2)}")
        if sn == "last windows":
            say(str(cm))
consts = {"label_row_deltas": dE.float(),
          "gate_row_L23": g23.float(), "up_row_L23": u23.float(), "down_col_L23": down_col23.float(),
          "gate_row_L22": g22.float(), "up_row_L22": u22.float(), "down_col_L22": down_col22.float(),
          "gate_row_L21": g21.float(), "up_row_L21": u21.float(), "down_col_L21": down_col21.float()}
save_file(consts, "output/wsp-b/edit_constants.safetensors")
with open(sys.argv[1].replace(".json", "_report.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
json.dump({"derived": {"osc": p_osc, "stal": p_stal, "reco": p_reco}, "ZS_new": {f"{k[0]}_{k[1]}": v for k, v in ZS.items()}, "CM": CM},
          open(sys.argv[1].replace(".json", "_derived.json"), "w"), indent=1)
print("saved output/wsp-b/edit_constants.safetensors for", sys.argv[1])
