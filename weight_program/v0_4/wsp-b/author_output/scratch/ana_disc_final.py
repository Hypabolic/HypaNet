"""Confusion of the final revision on the disclosed windows (calls 26-38), compared with the WSP-A edit and the
original (calls 1-13). Also the mean logit change per class and the fate of the WSP-A errors."""
import json
import numpy as np

CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
feat = {r["id"]: r for r in json.load(open("output/wsp-b/scratch/disc_feat.json"))}
L = json.load(open("output/wsp-b/scratch/disc_logits.json"))
LF = {}
for call in range(26, 39):
    resp = json.load(open(f"output/wsp-b/fwd/{call}/response.json"))
    for e in resp["sessions"]:
        ed = e["edited"]
        LF[e["id"]] = np.array(ed["label_logits"][0] if isinstance(ed["label_logits"][0], list) else ed["label_logits"])
ids = sorted(feat, key=lambda s: int(s[1:]))
y = np.array([CL.index(feat[w]["oracle"]) for w in ids])
for name, lg in (("original", np.array([L[w]["lo"] for w in ids])), ("WSP-A edit", np.array([L[w]["le"] for w in ids])), ("WSP-B revision", np.array([LF[w] for w in ids]))):
    cm = np.zeros((4, 4), int)
    for i in range(len(ids)):
        cm[y[i], int(np.argmax(lg[i]))] += 1
    print(f"{name}: acc={np.trace(cm) / cm.sum():.3f} recall={np.round(np.diag(cm) / cm.sum(1), 2)} precision={np.round(np.diag(cm) / np.maximum(cm.sum(0), 1), 2)}\n{cm}")
lo = np.array([L[w]["lo"] for w in ids]); lf = np.array([LF[w] for w in ids])
print("mean logit change (revision - original) by oracle class:")
for c in range(4):
    print(f"  {CL[c][:4]} {np.round((lf - lo)[y == c].mean(0), 2)}")
# windows named by the revision as each class, by oracle class, with ids (for REVISION.md)
pa = np.array([int(np.argmax(L[w]["le"])) for w in ids]); pb = np.array([int(np.argmax(LF[w])) for w in ids])
print("fixed (WSP-A wrong -> revision right):", [ids[i] for i in range(len(ids)) if pa[i] != y[i] and pb[i] == y[i]])
print("broken (WSP-A right -> revision wrong):", [ids[i] for i in range(len(ids)) if pa[i] == y[i] and pb[i] != y[i]])
with open("output/wsp-b/scratch/disc_final_table.txt", "w") as fh:
    for i, w in enumerate(ids):
        fh.write(f"{w} {CL[y[i]][:4]} wspa={CL[pa[i]][:4]} rev={CL[pb[i]][:4]} lo={np.round(lo[i], 1)} lf={np.round(lf[i], 1)} | {feat[w]['tools']}\n")
