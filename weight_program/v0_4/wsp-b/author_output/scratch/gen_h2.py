"""Write request files for set H2 (original model, last two windows, hidden 21-24, mlp_act 21-23)."""
import json
from wsp import tok
import scen_h2

ret = {"hidden": [21, 22, 23, 24], "mlp_act": [21, 22, 23], "topk": 3}
pairs = scen_h2.build_all()
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
    json.dump(req, open(f"output/wsp-b/scratch/req_i{i:02d}.json", "w"))
    json.dump([dict(id=sid, n=len(blocks), **m) for sid, blocks, m in ch], open(f"output/wsp-b/scratch/meta_i{i:02d}.json", "w"), indent=1)
    print(f"req_i{i:02d}: {len(ch)} sessions")
print("pairs:", len(pairs) // 2)
for sid, blocks, m in S[:2]:
    print("----", sid, m); print("\n".join(blocks[-9:]))
