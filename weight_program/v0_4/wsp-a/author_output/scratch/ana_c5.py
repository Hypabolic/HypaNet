"""Feature projections vs window index in set C positives: REP vs number of repeats; RES/ERR after resolution."""
import json
import sys
import numpy as np
import torch

fd = np.load("output/wsp-a/scratch/feat_dirs.npz", allow_pickle=True)
L = 23
D = {f: torch.tensor(fd[f"{f}_L{L}"]).double() for f in ("REP", "RES", "ERR", "PROG", "REV")}


def unit_rms(x):
    x = torch.tensor(np.asarray(x)).double()
    return x / x.pow(2).mean(-1, keepdim=True).sqrt()


rows = []
for call, metaf in [a.split(":") for a in sys.argv[1:]]:
    resp = json.load(open(f"output/wsp-a/fwd/{call}/response.json"))
    req = {s["id"]: s for s in json.load(open(f"output/wsp-a/fwd/{call}/request.json"))["sessions"]}
    meta = {m["id"]: m for m in json.load(open(metaf))}
    for si, s in enumerate(resp["sessions"]):
        m = meta[s["id"]]
        h = unit_rms(np.load(f"output/wsp-a/fwd/{call}/{resp['arrays'][f'original.s{si}.hidden.L{L}']['file']}"))
        blocks = req[s["id"]]["blocks"]
        rows.append((s["id"], m["feat"], m["sign"], blocks, h, m["labels"]))

print("=== REP positives: REP projection vs number of identical recent blocks (header without timestamp/tokens)")
by_k = {}
for sid, feat, sign, blocks, h, labels in rows:
    if feat != "REP" or sign != 1:
        continue
    tool = blocks[-1].split()[1]
    for t in range(len(blocks)):
        # count how many of the last 8 blocks equal the final block's header core (tool + fields + status)
        def core(b):
            parts = b.split("\n")[0].split()
            return " ".join(p for p in parts[1:] if not p.startswith(("in=", "out=", "req=")))
        c0 = core(blocks[t])
        k = sum(1 for b in blocks[max(0, t - 7):t + 1] if core(b) == c0)
        by_k.setdefault((tool, k), []).append(float(h[t] @ D["REP"]))
for key in sorted(by_k):
    v = np.array(by_k[key])
    print(f"  {key[0]:12s} repeats={key[1]:2d}: n={len(v):3d} REP proj mean={v.mean():6.2f} std={v.std():4.2f}")

print("=== REP negatives (distinct blocks) by window index")
by_t = {}
for sid, feat, sign, blocks, h, labels in rows:
    if feat != "REP" or sign != -1:
        continue
    for t in range(len(blocks)):
        by_t.setdefault(t, []).append(float(h[t] @ D["REP"]))
for t in sorted(by_t):
    v = np.array(by_t[t]); print(f"  t={t:2d}: n={len(v):3d} REP proj mean={v.mean():6.2f} std={v.std():4.2f}")

print("=== RES positives: RES / ERR / REP projection by offset from the resolving block (0 = the ok block)")
by_off = {}
for sid, feat, sign, blocks, h, labels in rows:
    if feat != "RES":
        continue
    # resolving block = last block with status ok/ready after an error/pending/FAILED block
    res_idx = None
    for t, b in enumerate(blocks):
        st = b.split("\n")[0].split(" -> ")[1].split()[0]
        if st in ("ok", "ready") and t > 0 and blocks[t - 1].split("\n")[0].split(" -> ")[1].split()[0] in ("error", "pending", "FAILED"):
            res_idx = t
    if res_idx is None:
        key0 = "n"
    else:
        key0 = "p"
    for t in range(len(blocks)):
        off = (t - res_idx) if res_idx is not None else (t - (len(blocks) - 1))
        by_off.setdefault((key0, off), []).append([float(h[t] @ D[f]) for f in ("RES", "ERR", "REP", "PROG")])
for key in sorted(by_off):
    v = np.array(by_off[key])
    print(f"  {'resolved' if key[0] == 'p' else 'unresolved':10s} offset={key[1]:3d}: n={len(v):3d} RES={v[:, 0].mean():6.2f} ERR={v[:, 1].mean():6.2f} REP={v[:, 2].mean():6.2f} PROG={v[:, 3].mean():6.2f}")
