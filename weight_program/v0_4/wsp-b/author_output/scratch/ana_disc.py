"""Disclosed windows under the WSP-A edit (calls 1-13, model=both): confusions, margins, and logits vs surface
features. Writes output/wsp-b/scratch/disc_logits.json."""
import json
import numpy as np

CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
feat = {r["id"]: r for r in json.load(open("output/wsp-b/scratch/disc_feat.json"))}
resp0 = json.load(open("output/wsp-b/fwd/1/response.json"))
print("response keys:", list(resp0.keys()))
L = {}
for call in range(1, 14):
    resp = json.load(open(f"output/wsp-b/fwd/{call}/response.json"))
    ws = resp.get("windows") or resp.get("sessions")
    if call == 1:
        print("entry keys:", list(ws[0].keys()) if isinstance(ws, list) else list(ws.values())[0].keys())
    entries = ws if isinstance(ws, list) else [dict(id=k, **v) for k, v in ws.items()]
    for e in entries:
        wid = e.get("id") or e.get("window_id")
        ed, og = e["edited"], e["original"]
        le = np.array(ed["label_logits"][0] if isinstance(ed["label_logits"][0], list) else ed["label_logits"])
        lo = np.array(og["label_logits"][0] if isinstance(og["label_logits"][0], list) else og["label_logits"])
        L[wid] = {"le": le.tolist(), "lo": lo.tolist()}
json.dump(L, open("output/wsp-b/scratch/disc_logits.json", "w"))
print("windows with logits:", len(L))
for kind in ("lo", "le"):
    cm = np.zeros((4, 4), int)
    for wid, r in feat.items():
        cm[CL.index(r["oracle"]), int(np.argmax(L[wid][kind]))] += 1
    print(f"{'original' if kind == 'lo' else 'WSP-A edit'} confusion:\n{cm}\n  acc={np.trace(cm) / cm.sum():.3f} recall={np.round(np.diag(cm) / cm.sum(1), 2)} precision={np.round(np.diag(cm) / np.maximum(cm.sum(0), 1), 2)}")
# consistency with the disclosed network_class
agree = np.mean([CL[int(np.argmax(L[w]["le"]))] == feat[w]["net"] for w in feat])
print("argmax(edited logits) agrees with disclosed network_class:", agree)
# per-class mean logits and deltas
print("mean logits by oracle class (original | edited | delta):")
for c in CL:
    lo = np.array([L[w]["lo"] for w in feat if feat[w]["oracle"] == c]); le = np.array([L[w]["le"] for w in feat if feat[w]["oracle"] == c])
    print(f"  {c[:4]}: orig {np.round(lo.mean(0), 2)} | edited {np.round(le.mean(0), 2)} | delta {np.round((le - lo).mean(0), 2)}")
# margins: for each window, the edited logit of the true class minus the best other
print("margin (true - best other) by oracle class, edited: mean / 10th / 50th / 90th pct; and the needed boost for errors")
for c in CL:
    m = []
    for w in feat:
        if feat[w]["oracle"] != c:
            continue
        le = np.array(L[w]["le"]); t = CL.index(c)
        m.append(le[t] - max(le[j] for j in range(4) if j != t))
    m = np.array(m)
    print(f"  {c[:4]}: {m.mean():6.2f} / {np.percentile(m, 10):6.2f} / {np.percentile(m, 50):6.2f} / {np.percentile(m, 90):6.2f}   errors: {int((m < 0).sum())}, median deficit of errors {np.median(m[m < 0]) if (m < 0).any() else 0:.2f}")
# logit deltas vs surface features: correlation of each class's edited logit (relative to window mean) with features
keys = ["new_facts", "n_repeat", "n_repeat_fail", "reversals", "n_edits", "max_edits_same_path", "n_err", "n_pending", "resolved_offset", "val_improve", "n_reads"]
print("corr(feature, edited logit - mean over classes):")
print(f"{'feature':20s} " + " ".join(f"{c[:5]:>6s}" for c in CL))
for k in keys:
    x = np.array([feat[w][k] if k != "resolved_offset" else (feat[w][k] >= 0) for w in feat], dtype=float)
    row = []
    for j in range(4):
        y = np.array([L[w]["le"][j] - np.mean(L[w]["le"]) for w in feat])
        row.append(np.corrcoef(x, y)[0, 1])
    print(f"{k:20s} " + " ".join(f"{v:6.2f}" for v in row))
# per-window table
with open("output/wsp-b/scratch/disc_table.txt", "w") as fh:
    for w in sorted(feat, key=lambda s: int(s[1:])):
        r = feat[w]
        fh.write(f"{w} {r['oracle'][:4]} orig={CL[int(np.argmax(L[w]['lo']))][:4]} edit={CL[int(np.argmax(L[w]['le']))][:4]} lo={np.round(L[w]['lo'], 1)} le={np.round(L[w]['le'], 1)} "
                 f"nf={r['new_facts']} rep={r['n_repeat']} repf={r['n_repeat_fail']} rev={r['reversals']} ed={r['n_edits']} sp={r['max_edits_same_path']} err={r['n_err']} pend={r['n_pending']} res={r['resolved_offset']} | {r['tools']}\n")
print("table written")
