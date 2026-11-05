"""Write request files for the identical-history pair set (scen_hb5), original model, last two windows, activations."""
import json
from wsp import tok
import scen_hb5

ret = {"hidden": [21, 22, 23, 24], "mlp_act": [21, 22, 23], "topk": 3}
pairs = scen_hb5.build_all()
S = [(r.sid, r.blocks(), {"pair": pid, "feat": feat, "sign": sign}) for pid, feat, sign, r in pairs]
chunks, cur, tot = [], [], 0
for i in range(0, len(S), 2):
    pr = S[i:i + 2]
    n = sum(167 + sum(len(tok.encode(b + "\n", add_special_tokens=False).ids) for b in blocks) + 22 for _, blocks, _ in pr)
    if tot + n > 46000:
        chunks.append(cur); cur, tot = [], 0
    cur += pr; tot += n
if cur:
    chunks.append(cur)
for i, ch in enumerate(chunks, 1):
    req = {"model": "original", "sessions": [{"id": sid, "blocks": blocks, "windows": [len(blocks) - 2, len(blocks) - 1]} for sid, blocks, m in ch], "return": ret}
    json.dump(req, open(f"output/wsp-b/scratch/req_k{i:02d}.json", "w"))
    json.dump([dict(id=sid, n=len(blocks), **m) for sid, blocks, m in ch], open(f"output/wsp-b/scratch/meta_k{i:02d}.json", "w"), indent=1)
    print(f"req_k{i:02d}: {len(ch)} sessions, est tokens {sum(167 + sum(len(tok.encode(b + chr(10), add_special_tokens=False).ids) for b in blocks) + 22 for _, blocks, _ in ch)}")
print("pairs:", len(pairs) // 2, {f: sum(1 for p in pairs if p[1] == f) // 2 for f in sorted({p[1] for p in pairs})})
ok = True
for i in range(0, len(S), 2):
    p0, n0 = S[i][1], S[i + 1][1]
    nh = min(len(p0), len(n0)) - 8
    ok = ok and all(a == b for a, b in zip(p0[:nh], n0[:nh]))
print("identical histories for all pairs:", ok)
print("----", S[0][0]); print("\n".join(S[0][1][-9:]))
print("----", S[1][0]); print("\n".join(S[1][1][-9:]))
