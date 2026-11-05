"""Offline what-if on the disclosed windows: rescale the STAL and OSC neuron contributions (revision - original logit
of that class, the only direct-path term on those logits) by the calibration rule 'median gain on the target class
= TARGET_GAIN (5 logits)'. Prints the factors and the resulting confusions."""
import json
import numpy as np

CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
feat = {r["id"]: r for r in json.load(open("output/wsp-b/scratch/disc_feat.json"))}
L = json.load(open("output/wsp-b/scratch/disc_logits.json"))
LF = {}
for call in range(26, 39):
    for e in json.load(open(f"output/wsp-b/fwd/{call}/response.json"))["sessions"]:
        ed = e["edited"]; LF[e["id"]] = np.array(ed["label_logits"][0] if isinstance(ed["label_logits"][0], list) else ed["label_logits"])
ids = sorted(feat, key=lambda s: int(s[1:]))
y = np.array([CL.index(feat[w]["oracle"]) for w in ids])
lo = np.array([L[w]["lo"] for w in ids]); lf = np.array([LF[w] for w in ids])
d = lf - lo
med_s = float(np.median(d[y == 2, 2])); med_o = float(np.median(d[y == 3, 3]))
f_s, f_o = 5.0 / med_s, 5.0 / med_o
print(f"median STAL gain on STALLED windows {med_s:.2f} -> factor {f_s:.3f}; median OSC gain on OSCILLATING windows {med_o:.2f} -> factor {f_o:.3f}")
print("percentiles of STAL gain by class:", {CL[c][:4]: np.round(np.percentile(d[y == c, 2], [10, 50, 90]), 1).tolist() for c in range(4)})
print("percentiles of OSC gain by class:", {CL[c][:4]: np.round(np.percentile(d[y == c, 3], [10, 50, 90]), 1).tolist() for c in range(4)})


def confusion(lg, title):
    cm = np.zeros((4, 4), int)
    for i in range(len(ids)):
        cm[y[i], int(np.argmax(lg[i]))] += 1
    print(f"{title}: acc={np.trace(cm) / cm.sum():.3f} recall={np.round(np.diag(cm) / cm.sum(1), 2)} precision={np.round(np.diag(cm) / np.maximum(cm.sum(0), 1), 2)}\n{cm}")


confusion(lf, "revision as run")
for fs, fo, title in ((f_s, 1.0, "STAL rescaled by rule"), (1.0, f_o, "OSC rescaled by rule"), (f_s, f_o, "both rescaled by rule")):
    sim = lf.copy()
    sim[:, 2] = lo[:, 2] + fs * d[:, 2]
    sim[:, 3] = lo[:, 3] + fo * d[:, 3]
    confusion(sim, title)
