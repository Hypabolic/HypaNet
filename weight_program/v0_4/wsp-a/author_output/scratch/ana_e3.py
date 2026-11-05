"""Set E analysis for a 3-neuron edit (calls with model=both, mlp_act 21-23): confusion edited vs original,
observed vs predicted logit deltas (leakage check), per-(recipe,label) accuracy and mean activations.
usage: ./py ana_e3.py TABLE_OUT CALL:META ..."""
import json
import sys
import numpy as np
import torch
from wsp import label_rows, load

CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
P = "model.language_model."
NEURON = {23: 1046, 22: 1613, 21: 1007}
w_fin = load(P + "norm.weight").double()
E0, E1 = label_rows(original=True).double(), label_rows(original=False).double()
R1 = E1 * (1 + w_fin)
dR = (E1 - E0) * (1 + w_fin)
gain = {}
for L, n in NEURON.items():
    d = load(f"{P}layers.{L}.mlp.down_proj.weight").double()[:, n]
    gain[L] = (R1 @ d).numpy()  # logit change of each class per unit act (before /rms)
print("per-unit-act gains:", {L: np.round(g, 3) for L, g in gain.items()})
conf = {"edited": np.zeros((4, 4), int), "original": np.zeros((4, 4), int)}
rows = []
for call, metaf in [a.split(":") for a in sys.argv[2:]]:
    resp = json.load(open(f"output/wsp-a/fwd/{call}/response.json"))
    meta = {m["id"]: m for m in json.load(open(metaf))}
    for si, s in enumerate(resp["sessions"]):
        m = meta[s["id"]]
        a = lambda k: np.load(f"output/wsp-a/fwd/{call}/{resp['arrays'][k]['file']}")
        h24e = torch.tensor(a(f"edited.s{si}.hidden.L24")).double()
        acts = {L: a(f"edited.s{si}.mlp_act.L{L}")[:, n] for L, n in NEURON.items()}
        rms = torch.sqrt(h24e.pow(2).mean(-1) + 1e-6).numpy()
        xhat = h24e / torch.sqrt(h24e.pow(2).mean(-1, keepdim=True) + 1e-6)
        dl_rows = (xhat @ dR.T).numpy()
        for wi, t in enumerate(s["windows"]):
            lab = m["labels"][t]
            le, lo = np.array(s["edited"]["label_logits"][wi]), np.array(s["original"]["label_logits"][wi])
            pe, po = int(np.argmax(le)), int(np.argmax(lo))
            if lab != "AMBIG":
                conf["edited"][CL.index(lab), pe] += 1; conf["original"][CL.index(lab), po] += 1
            pred = dl_rows[wi].copy()
            for L in NEURON:
                pred += acts[L][wi] * gain[L] / rms[wi]
            rows.append({"sid": s["id"], "recipe": s["id"].split("_")[0], "t": t, "lab": lab, "pe": pe, "po": po, "obs": le - lo, "pred": pred,
                         "acts": {L: float(acts[L][wi]) for L in NEURON}, "le": le})
for k in conf:
    cm = conf[k]
    print(f"{k} confusion on set E confident windows:\n{cm}\n  acc={np.trace(cm) / cm.sum():.3f} recall={np.round(np.diag(cm) / cm.sum(1), 2)} precision={np.round(np.diag(cm) / np.maximum(cm.sum(0), 1), 2)}")
obs = np.stack([r["obs"] for r in rows]); pred = np.stack([r["pred"] for r in rows])
obs_c = obs - obs.mean(1, keepdims=True); pred_c = pred - pred.mean(1, keepdims=True)
for c in range(4):
    err = obs_c[:, c] - pred_c[:, c]
    print(f"{CL[c][:4]}: obs-centred mean {obs_c[:, c].mean():6.2f} std {obs_c[:, c].std():5.2f} | pred mean {pred_c[:, c].mean():6.2f} std {pred_c[:, c].std():5.2f} | residual mean {err.mean():6.2f} std {err.std():5.2f} corr={np.corrcoef(obs_c[:, c], pred_c[:, c])[0, 1]:.3f}")
groups = {}
for r in rows:
    groups.setdefault((r["recipe"], r["lab"]), []).append(r)
print("recipe   label  n  acc_e acc_o | act21 act22 act23 | pred-class counts (edited)")
for key in sorted(groups, key=lambda k: (int(k[0][1:]), k[1])):
    g = groups[key]
    acc_e = np.mean([CL.index(r["lab"]) == r["pe"] for r in g]) if key[1] != "AMBIG" else float("nan")
    acc_o = np.mean([CL.index(r["lab"]) == r["po"] for r in g]) if key[1] != "AMBIG" else float("nan")
    cnt = np.bincount([r["pe"] for r in g], minlength=4)
    print(f"{key[0]:8s} {key[1][:5]:5s} {len(g):2d}  {acc_e:5.2f} {acc_o:5.2f} | " + " ".join(f"{np.mean([r['acts'][L] for r in g]):5.1f}" for L in (21, 22, 23)) + f" | {cnt}")
with open(sys.argv[1], "w") as fh:
    for r in rows:
        fh.write(f"{r['sid']:6s} t{r['t']:2d} {r['lab'][:4]} orig={CL[r['po']][:4]} edit={CL[r['pe']][:4]} acts=" +
                 " ".join(f"{r['acts'][L]:6.2f}" for L in (21, 22, 23)) + f" obs={np.round(r['obs'], 1)} le={np.round(r['le'], 1)}\n")
print("table written to", sys.argv[1])
