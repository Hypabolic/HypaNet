"""Edit-type repetition contrast from set C (fixed grouping: 'revert-rep' = REV positives containing a revert
block; 'edit-rep' = other REV positives ending in an edit/write; 'other-rep' = REP positives not ending in an
edit), compared with the OSCI-vs-STAL contrast of set B."""
import json
import sys
import numpy as np
import torch

CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
b = np.load("output/wsp-a/scratch/setB_orig.npz", allow_pickle=True)
Y = b["Y"]; mask = Y >= 0


def unit_rms(x):
    x = torch.tensor(np.asarray(x)).double()
    return x / x.pow(2).mean(-1, keepdim=True).sqrt()


X = unit_rms(b["h24"][mask]); y = Y[mask]
mu = torch.stack([X[y == c].mean(0) for c in range(4)])
g = X.mean(0)
osc_stal = mu[3] - mu[2]
sess = {}
for call, metaf in [a.split(":") for a in sys.argv[1:]]:
    resp = json.load(open(f"output/wsp-a/fwd/{call}/response.json"))
    req = {s["id"]: s for s in json.load(open(f"output/wsp-a/fwd/{call}/request.json"))["sessions"]}
    meta = {m["id"]: m for m in json.load(open(metaf))}
    for si, s in enumerate(resp["sessions"]):
        m = meta[s["id"]]
        sess[s["id"]] = {"feat": m["feat"], "sign": m["sign"], "blocks": req[s["id"]]["blocks"], "labels": m["labels"],
                         "pred": int(np.argmax(s["original"]["label_logits"][-1])),
                         "h24": np.load(f"output/wsp-a/fwd/{call}/{resp['arrays'][f'original.s{si}.hidden.L24']['file']}")}
groups = {"edit-rep": [], "other-rep": [], "revert-rep": [], "fresh-edits": []}
preds = {k: np.zeros(4, int) for k in groups}
for sid, e in sess.items():
    last = e["blocks"][-1].split()[1]
    has_revert = any(bl.split()[1] == "revert" for bl in e["blocks"])
    x = unit_rms(e["h24"])[-1]
    key = None
    if e["sign"] == 1 and e["feat"] == "REV" and has_revert:
        key = "revert-rep"
    elif e["sign"] == 1 and e["feat"] == "REV" and last in ("edit_file", "write_file"):
        key = "edit-rep"
    elif e["sign"] == 1 and e["feat"] == "REP" and last != "edit_file":
        key = "other-rep"
    elif e["sign"] == -1 and e["feat"] == "REV" and last in ("edit_file", "write_file") and not has_revert:
        key = "fresh-edits"
    if key:
        groups[key].append(x); preds[key][e["pred"]] += 1
G = {k: torch.stack(v) for k, v in groups.items()}
print({k: len(v) for k, v in G.items()}, "baseline predictions on last window:", {k: v.tolist() for k, v in preds.items()})
d_er = G["edit-rep"].mean(0) - G["other-rep"].mean(0)
d_rv = G["revert-rep"].mean(0) - G["other-rep"].mean(0)
d_fe = G["fresh-edits"].mean(0) - G["other-rep"].mean(0)
d_ef = G["edit-rep"].mean(0) - G["fresh-edits"].mean(0)
cos = lambda a, c: round(float(a @ c / a.norm() / c.norm()), 3)
print("|edit-rep - other-rep|=", round(float(d_er.norm()), 3), " cos OSCI-STAL:", cos(d_er, osc_stal))
print("|revert-rep - other-rep|=", round(float(d_rv.norm()), 3), " cos OSCI-STAL:", cos(d_rv, osc_stal), " cos edit-rep:", cos(d_rv, d_er))
print("|fresh-edits - other-rep|=", round(float(d_fe.norm()), 3), " cos OSCI-STAL:", cos(d_fe, osc_stal), " cos edit-rep:", cos(d_fe, d_er))
print("|edit-rep - fresh-edits|=", round(float(d_ef.norm()), 3), " cos OSCI-STAL:", cos(d_ef, osc_stal))
for name, v in (("edit-rep", d_er), ("revert-rep", d_rv), ("fresh-edits", d_fe), ("edit-rep minus fresh", d_ef), ("OSCI-STAL", osc_stal)):
    vn = v / v.norm()
    s = (X @ vn).numpy()
    a, r = s[y == 3], s[y == 2]
    print(f"  {name:22s}: set B OSCI {a.mean():6.2f}+-{a.std():4.2f} vs STAL {r.mean():6.2f}+-{r.std():4.2f} d'={(a.mean() - r.mean()) / np.sqrt(0.5 * (a.var() + r.var())):5.2f}"
          f" | RECO {s[y == 1].mean():6.2f}+-{s[y == 1].std():4.2f} PROD {s[y == 0].mean():6.2f}+-{s[y == 0].std():4.2f}")
np.savez("output/wsp-a/scratch/contrasts_L24.npz", osc_stal=osc_stal.float().numpy(), edit_rep=d_er.float().numpy(), revert_rep=d_rv.float().numpy(),
         fresh_edits=d_fe.float().numpy(), edit_minus_fresh=d_ef.float().numpy(), mean24=g.float().numpy(), mu=mu.float().numpy())
