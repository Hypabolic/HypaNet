"""What-if on the disclosed windows using the WSP-A direct-path deltas (edited - original logits):
(a) PROD row replaced by a constant bias equal to the mean original deficit of the productive logit;
(b) same, with the WSP-A RECO neuron's contribution removed (to be rebuilt);
(c) per-class percentiles of each WSP-A contribution on the real windows."""
import json
import numpy as np

CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
feat = {r["id"]: r for r in json.load(open("output/wsp-b/scratch/disc_feat.json"))}
L = json.load(open("output/wsp-b/scratch/disc_logits.json"))
ids = sorted(feat, key=lambda s: int(s[1:]))
lo = np.array([L[w]["lo"] for w in ids]); le = np.array([L[w]["le"] for w in ids]); y = np.array([CL.index(feat[w]["oracle"]) for w in ids])
delta = le - lo
print("per-class percentiles (10/50/90) of each WSP-A contribution (delta of that logit):")
for j, nm in enumerate(["PROD row", "RECO neuron", "STAL neuron", "OSC neuron"]):
    print(f"  {nm:12s} " + " | ".join(f"{CL[c][:4]} {np.percentile(delta[y == c, j], 10):6.2f}/{np.percentile(delta[y == c, j], 50):6.2f}/{np.percentile(delta[y == c, j], 90):6.2f}" for c in range(4)))
deficit = (np.max(np.delete(lo, 0, axis=1), axis=1) - lo[:, 0])
print(f"original productive deficit (max other - productive): mean {deficit.mean():.2f}, by class " + " ".join(f"{CL[c][:4]}={deficit[y == c].mean():.2f}" for c in range(4)))


def confusion(lg, title):
    cm = np.zeros((4, 4), int)
    for i in range(len(ids)):
        cm[y[i], int(np.argmax(lg[i]))] += 1
    print(f"{title}: acc={np.trace(cm) / cm.sum():.3f} recall={np.round(np.diag(cm) / cm.sum(1), 2)} precision={np.round(np.diag(cm) / np.maximum(cm.sum(0), 1), 2)}\n{cm}")


B = round(float(deficit.mean()), 1)
confusion(le, "WSP-A edit as is")
sim = lo.copy(); sim[:, 0] += B; sim[:, 1] += delta[:, 1]; sim[:, 2] += delta[:, 2]; sim[:, 3] += delta[:, 3]
confusion(sim, f"(a) PROD bias {B} + old RECO/STAL/OSC neurons")
sim2 = lo.copy(); sim2[:, 0] += B; sim2[:, 2] += delta[:, 2]; sim2[:, 3] += delta[:, 3]
confusion(sim2, f"(b) PROD bias {B} + old STAL/OSC neurons only")
# (c) margins the overrides need: on RECO windows under (b), how far is RECO below the winner?
m = sim2[:, 1] - np.max(np.delete(sim2, 1, axis=1), axis=1)
print("under (b): RECO margin on RECO windows: median %.2f (10th %.2f, 90th %.2f); on PROD windows median %.2f" % (np.median(m[y == 1]), np.percentile(m[y == 1], 10), np.percentile(m[y == 1], 90), np.median(m[y == 0])))
# relation of the old STAL contribution to surface features on real windows
for k in ("n_repeat_fail", "n_pending", "new_facts", "n_reads", "n_edits", "reversals"):
    x = np.array([feat[w][k] for w in ids], float)
    print(f"corr({k:14s}, STAL delta)={np.corrcoef(x, delta[:, 2])[0, 1]:5.2f}  corr(., OSC delta)={np.corrcoef(x, delta[:, 3])[0, 1]:5.2f}  corr(., RECO delta)={np.corrcoef(x, delta[:, 1])[0, 1]:5.2f}")
