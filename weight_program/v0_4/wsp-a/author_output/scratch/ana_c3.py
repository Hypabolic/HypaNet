"""Edit-type repetition contrast from set C (rev positives whose last block is an edit/write or a revert, vs
rep positives whose last block is not an edit), compared with the OSCI-vs-STAL contrast of set B."""
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
                         "h24": np.load(f"output/wsp-a/fwd/{call}/{resp['arrays'][f'original.s{si}.hidden.L24']['file']}")}
edit_rep, other_rep, revert_rep, fresh_edits = [], [], [], []
for sid, e in sess.items():
    last = e["blocks"][-1]
    x = unit_rms(e["h24"])[-1]
    if e["sign"] == 1:
        if e["feat"] == "REV" and last.split()[1] == "revert":
            revert_rep.append(x)
        elif e["feat"] == "REV" and last.split()[1] in ("edit_file", "write_file"):
            edit_rep.append(x)
        elif e["feat"] == "REP" and last.split()[1] not in ("edit_file",):
            other_rep.append(x)
    elif e["feat"] == "REV" and last.split()[1] in ("edit_file", "write_file"):
        fresh_edits.append(x)
edit_rep, other_rep, revert_rep, fresh_edits = [torch.stack(v) for v in (edit_rep, other_rep, revert_rep, fresh_edits)]
print("n edit-rep, other-rep, revert-rep, fresh-edits:", len(edit_rep), len(other_rep), len(revert_rep), len(fresh_edits))
d_er = edit_rep.mean(0) - other_rep.mean(0)
d_rv = revert_rep.mean(0) - other_rep.mean(0)
d_fe = fresh_edits.mean(0) - other_rep.mean(0)
cos = lambda a, c: round(float(a @ c / a.norm() / c.norm()), 3)
print("|edit-rep - other-rep|=", round(float(d_er.norm()), 3), " cos with OSCI-STAL:", cos(d_er, osc_stal))
print("|revert-rep - other-rep|=", round(float(d_rv.norm()), 3), " cos with OSCI-STAL:", cos(d_rv, osc_stal))
print("|fresh-edits - other-rep|=", round(float(d_fe.norm()), 3), " cos with OSCI-STAL:", cos(d_fe, osc_stal), " cos with edit-rep dir:", cos(d_fe, d_er))
print("cos(edit-rep dir, revert-rep dir):", cos(d_er, d_rv))
for name, v in (("edit-rep", d_er), ("revert-rep", d_rv), ("fresh-edits", d_fe), ("OSCI-STAL", osc_stal)):
    vn = v / v.norm()
    s = (X @ vn).numpy()
    a, r = s[y == 3], s[y == 2]
    print(f"  {name}: set B OSCI {a.mean():.2f}+-{a.std():.2f} vs STAL {r.mean():.2f}+-{r.std():.2f}  d'={(a.mean() - r.mean()) / np.sqrt(0.5 * (a.var() + r.var())):.2f}",
          " | RECO", round(float(s[y == 1].mean()), 2), "+-", round(float(s[y == 1].std()), 2), "PROD", round(float(s[y == 0].mean()), 2), "+-", round(float(s[y == 0].std()), 2))
np.savez("output/wsp-a/scratch/contrasts_L24.npz", osc_stal=osc_stal.float().numpy(), edit_rep=d_er.float().numpy(), revert_rep=d_rv.float().numpy(),
         fresh_edits=d_fe.float().numpy(), mean24=g.float().numpy(), mu=mu.float().numpy())
# what does the baseline model predict on the edit-rep vs other-rep sessions' last windows?
for name, keys in (("edit-rep", ("REV", 1, ("edit_file", "write_file"))), ("other-rep", ("REP", 1, None))):
    cnt = np.zeros(4, int)
    for sid, e in sess.items():
        if e["feat"] == keys[0] and e["sign"] == keys[1] and (keys[2] is None or e["blocks"][-1].split()[1] in keys[2]):
            pass
    print(name, "(see earlier fwd summaries for predictions)")
