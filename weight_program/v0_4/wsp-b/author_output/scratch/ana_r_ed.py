"""Set R under the revision (edited runs): confusion vs the original runs (calls 17-18), and the leakage check
(observed - original logits vs the direct-path prediction from the two neuron activations and the label-row deltas).
usage: ./py ana_r_ed.py CALL_R01 CALL_R02"""
import json
import sys
import numpy as np
import torch
from wsp import label_rows, load

CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
P = "model.language_model."
NEURON = {23: 1046, 22: 1613}
w_fin = load(P + "norm.weight").double()
E0, E1 = label_rows(original=True).double(), label_rows(original=False).double()
R1 = E1 * (1 + w_fin); dR = (E1 - E0) * (1 + w_fin)
gain = {L: (R1 @ load(f"{P}layers.{L}.mlp.down_proj.weight").double()[:, n]).numpy() for L, n in NEURON.items()}
orig = {}
for call, name in ((17, "r01"), (18, "r02")):
    resp = json.load(open(f"output/wsp-b/fwd/{call}/response.json"))
    for s in resp["sessions"]:
        orig[s["id"]] = np.array(s["original"]["label_logits"])
cm_o, cm_e = np.zeros((4, 4), int), np.zeros((4, 4), int)
rows = []
for call, name in zip(sys.argv[1:3], ("r01", "r02")):
    d = f"output/wsp-b/fwd/{call}"
    resp = json.load(open(f"{d}/response.json"))
    meta = {m["id"]: m for m in json.load(open(f"output/wsp-b/scratch/meta_{name}.json"))}
    for si, s in enumerate(resp["sessions"]):
        y = CL.index(meta[s["id"]]["label"])
        h24 = torch.tensor(np.load(f"{d}/{resp['arrays'][f'edited.s{si}.hidden.L24']['file']}")).double()
        acts = {L: np.load(f"{d}/{resp['arrays'][f'edited.s{si}.mlp_act.L{L}']['file']}")[:, n] for L, n in NEURON.items()}
        a21 = np.load(f"{d}/{resp['arrays'][f'edited.s{si}.mlp_act.L21']['file']}")[:, 1007]
        rms = torch.sqrt(h24.pow(2).mean(-1) + 1e-6).numpy()
        xhat = h24 / torch.sqrt(h24.pow(2).mean(-1, keepdim=True) + 1e-6)
        dl = (xhat @ dR.T).numpy()
        for wi in range(2):
            le, lo = np.array(s["edited"]["label_logits"][wi]), orig[s["id"]][wi]
            pred = dl[wi].copy()
            for L in NEURON:
                pred += acts[L][wi] * gain[L] / rms[wi]
            rows.append({"id": s["id"], "y": y, "last": wi == 1, "le": le, "lo": lo, "obs": le - lo, "pred": pred, "a23": acts[23][wi], "a22": acts[22][wi], "a21": a21[wi]})
            if wi == 1:
                cm_o[y, int(np.argmax(lo))] += 1; cm_e[y, int(np.argmax(le))] += 1
for name, cm in (("original", cm_o), ("revision", cm_e)):
    print(f"{name} (last windows): acc={np.trace(cm) / cm.sum():.3f} recall={np.round(np.diag(cm) / cm.sum(1), 2)} precision={np.round(np.diag(cm) / np.maximum(cm.sum(0), 1), 2)}\n{cm}")
obs = np.stack([r["obs"] for r in rows]); pred = np.stack([r["pred"] for r in rows])
obs_c = obs - obs.mean(1, keepdims=True); pred_c = pred - pred.mean(1, keepdims=True)
for c in range(4):
    err = obs_c[:, c] - pred_c[:, c]
    print(f"{CL[c][:4]}: obs-centred mean {obs_c[:, c].mean():6.2f} std {obs_c[:, c].std():5.2f} | pred mean {pred_c[:, c].mean():6.2f} std {pred_c[:, c].std():5.2f} | residual std {err.std():5.2f} corr={np.corrcoef(obs_c[:, c], pred_c[:, c])[0, 1]:.3f}")
print("layer-21 unit 1007 activation (should be the original unit, ~0): max |a21| =", max(abs(r["a21"]) for r in rows))
print("per class (last windows): mean act22 / act23 and mean deltas")
for c in range(4):
    sel = [r for r in rows if r["y"] == c and r["last"]]
    print(f"  {CL[c][:4]} act22={np.mean([r['a22'] for r in sel]):6.2f} act23={np.mean([r['a23'] for r in sel]):6.2f} delta={np.round(np.mean([r['obs'] for r in sel], 0), 2)}")
with open("output/wsp-b/scratch/r_ed_table.txt", "w") as fh:
    for r in rows:
        fh.write(f"{r['id']:12s} {'last' if r['last'] else 'prev'} {CL[r['y']][:4]} orig={CL[int(np.argmax(r['lo']))][:4]} rev={CL[int(np.argmax(r['le']))][:4]} le={np.round(r['le'], 1)} a22={r['a22']:6.2f} a23={r['a23']:6.2f}\n")
