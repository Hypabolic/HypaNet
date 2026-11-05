"""Write request files for set H (pairs) and set R (labelled), original model, last two windows read out, with
hidden 21-24 and mlp_act 21-23. Also prints a couple of rendered sessions."""
import json
from wsp import tok
import scen_h
import scen_r

ret = {"hidden": [21, 22, 23, 24], "mlp_act": [21, 22, 23], "topk": 3}


def write(name, sessions, meta):
    chunks, cur, tot = [], [], 0
    for sid, blocks, m in sessions:
        n = 167 + sum(len(tok.encode(b + "\n", add_special_tokens=False).ids) for b in blocks) + 22
        if tot + n > 46000:
            chunks.append(cur); cur, tot = [], 0
        cur.append((sid, blocks, m)); tot += n
    if cur:
        chunks.append(cur)
    for i, ch in enumerate(chunks, 1):
        req = {"model": "original", "sessions": [{"id": sid, "blocks": blocks, "windows": [len(blocks) - 2, len(blocks) - 1]} for sid, blocks, m in ch], "return": ret}
        json.dump(req, open(f"output/wsp-b/scratch/req_{name}{i:02d}.json", "w"))
        json.dump([dict(id=sid, n=len(blocks), **m) for sid, blocks, m in ch], open(f"output/wsp-b/scratch/meta_{name}{i:02d}.json", "w"), indent=1)
        print(f"req_{name}{i:02d}: {len(ch)} sessions, est tokens {sum(167 + sum(len(tok.encode(b + chr(10), add_special_tokens=False).ids) for b in blocks) + 22 for _, blocks, _ in ch)}")


pairs = scen_h.build_all()
H = [(r.sid, r.blocks(), {"pair": pid, "feat": feat, "sign": sign}) for pid, feat, sign, r in pairs]
write("h", H, None)
Rs = scen_r.all_sessions()
RR = [(r.sid, r.blocks(), {"label": lab}) for lab, r in Rs]
write("r", RR, None)
print("H sessions:", len(H), "R sessions:", len(RR), {lab: sum(1 for l, _ in Rs if l == lab) for lab in set(l for l, _ in Rs)})
for sid, blocks, m in [RR[0], RR[40], RR[55]]:
    print("----", sid, m); print("\n".join(blocks[-9:]))
