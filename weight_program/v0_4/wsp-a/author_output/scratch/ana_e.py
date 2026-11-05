"""Set E (calls 12-14, model=both): confusion edited vs original; observed vs predicted logit deltas (leakage check);
neuron activations by session type. usage: ./py ana_e.py PARAMS.json CALL:META ..."""
import json
import sys
import numpy as np
import torch
from wsp import load_original, label_rows, load

CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
P = "model.language_model."
params = json.load(open(sys.argv[1]))
N23, N22 = 1046, 1613
w_fin = load(P + "norm.weight").double()
E0 = label_rows(original=True).double()
E1 = label_rows(original=False).double()
dR = (E1 - E0) * (1 + w_fin)
down23 = load_original(P + "layers.23.mlp.down_proj.weight").double()
d23 = load(P + "layers.23.mlp.down_proj.weight").double()[:, N23]
d22 = load(P + "layers.22.mlp.down_proj.weight").double()[:, N22]
R1 = E1 * (1 + w_fin)
g_osc = float(R1[3] @ (d23 / d23.norm())) * float(d23.norm())
g_stal = float(R1[2] @ (d22 / d22.norm())) * float(d22.norm())
print(f"per-unit-act logit gains (before /rms): OSC {g_osc:.3f} STAL {g_stal:.3f}")

conf = {"edited": np.zeros((4, 4), int), "original": np.zeros((4, 4), int)}
rows = []
for call, metaf in [a.split(":") for a in sys.argv[2:]]:
    resp = json.load(open(f"output/wsp-a/fwd/{call}/response.json"))
    meta = {m["id"]: m for m in json.load(open(metaf))}
    for si, s in enumerate(resp["sessions"]):
        m = meta[s["id"]]
        a = lambda k: np.load(f"output/wsp-a/fwd/{call}/{resp['arrays'][k]['file']}")
        h24e = torch.tensor(a(f"edited.s{si}.hidden.L24")).double()
        m22e, m23e = a(f"edited.s{si}.mlp_act.L22"), a(f"edited.s{si}.mlp_act.L23")
        rms = torch.sqrt(h24e.pow(2).mean(-1) + 1e-6).numpy()
        xhat = (h24e / torch.sqrt(h24e.pow(2).mean(-1, keepdim=True) + 1e-6))
        dl_rows = (xhat @ dR.T).numpy()
        for wi, t in enumerate(s["windows"]):
            lab = m["labels"][t]
            le, lo = np.array(s["edited"]["label_logits"][wi]), np.array(s["original"]["label_logits"][wi])
            pe, po = int(np.argmax(le)), int(np.argmax(lo))
            if lab != "AMBIG":
                conf["edited"][CL.index(lab), pe] += 1
                conf["original"][CL.index(lab), po] += 1
            act_o, act_s = float(m23e[wi, N23]), float(m22e[wi, N22])
            pred = dl_rows[wi].copy()
            pred[3] += act_o * g_osc / rms[wi]
            pred[2] += act_s * g_stal / rms[wi]
            rows.append({"sid": s["id"], "t": t, "lab": lab, "pe": pe, "po": po, "obs": le - lo, "pred": pred, "act_o": act_o, "act_s": act_s, "lo": lo, "le": le})
for k in conf:
    cm = conf[k]
    print(f"{k} confusion on set E confident windows:\n{cm}\n  acc={np.trace(cm) / cm.sum():.3f} recall={np.round(np.diag(cm) / cm.sum(1), 2)} precision={np.round(np.diag(cm) / np.maximum(cm.sum(0), 1), 2)}")

obs = np.stack([r["obs"] for r in rows]); pred = np.stack([r["pred"] for r in rows])
# observed deltas include a common shift (final-norm rescaling); compare after removing the per-window mean over classes
obs_c = obs - obs.mean(1, keepdims=True); pred_c = pred - pred.mean(1, keepdims=True)
for c in range(4):
    err = obs_c[:, c] - pred_c[:, c]
    print(f"{CL[c][:4]}: obs-centred delta mean {obs_c[:, c].mean():6.2f} std {obs_c[:, c].std():5.2f} | pred {pred_c[:, c].mean():6.2f} std {pred_c[:, c].std():5.2f} | residual mean {err.mean():6.2f} std {err.std():5.2f} corr(obs,pred)={np.corrcoef(obs_c[:, c], pred_c[:, c])[0, 1]:.3f}")
print("per-window common shift (obs mean over classes): mean %.2f std %.2f" % (obs.mean(1).mean(), obs.mean(1).std()))

# per-session table
cur = None
for r in rows:
    if r["sid"] != cur:
        cur = r["sid"]; print(f"== {cur}")
    flag = "" if r["lab"] == "AMBIG" else ("ok" if CL.index(r["lab"]) == r["pe"] else "XX")
    print(f"  t{r['t']:2d} {r['lab'][:4]} orig={CL[r['po']][:4]} edit={CL[r['pe']][:4]} {flag:2s} act_osc={r['act_o']:6.2f} act_stal={r['act_s']:6.2f} "
          f"obs_d={np.round(r['obs'], 1)} le={np.round(r['le'], 1)}")
