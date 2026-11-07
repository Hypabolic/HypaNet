"""Check which dims / units / heads of Phase B's network are free before the revision writes into them."""
import torch

import tokenizer
import v05_load

sd = v05_load.b_state()


def nz(name, sl=None):
    t = sd[name]
    if sl is not None:
        t = t[sl]
    return int((t != 0).sum())


print("word_id h0 =", tokenizer.word_id("h0"), " read_file id =", tokenizer.RESERVED_IDS["read_file"])
for col in (30, 78, 87, 88, 89, 90, 91, 92, 93, 94, 95, 112, 127):
    print(f"tok_emb col {col} nonzero:", nz("tok_emb.weight", (slice(None), col)), " off_emb col:", nz("off_emb.weight", (slice(None), col)))
print("off_emb rows 1..191 col 19/20 sample:", sd["off_emb.weight"][12, 19].item(), sd["off_emb.weight"][12, 20].item())
print("tok_emb[7, :] nonzero dims:", torch.nonzero(sd["tok_emb.weight"][7]).flatten().tolist())
print("tok_emb[5861, :] nonzero dims:", torch.nonzero(sd["tok_emb.weight"][5861]).flatten().tolist())
# layer 0 attention head 2 value rows 40-47
print("L0 v rows 40:48 nonzero:", nz("layers.0.attn.v.weight", (slice(40, 48), slice(None))), " o cols 40:48:", nz("layers.0.attn.o.weight", (slice(None), slice(40, 48))))
# layer 0 heads 5-7
for h in (5, 6, 7):
    r = slice(16 * h, 16 * h + 16)
    print(f"L0 head {h}: q {nz('layers.0.attn.q.weight', (r, slice(None)))} qb {nz('layers.0.attn.q.bias', r)} k {nz('layers.0.attn.k.weight', (r, slice(None)))} v {nz('layers.0.attn.v.weight', (r, slice(None)))} o {nz('layers.0.attn.o.weight', (slice(None), r))} bias {sd['layers.0.attn.rel_bias'][h].tolist()[:3]}")
# MLP free units
print("L0 MLP units 242+ nonzero fc1 rows:", nz("layers.0.mlp.fc1.weight", (slice(242, 512), slice(None))), " fc2 cols:", nz("layers.0.mlp.fc2.weight", (slice(None), slice(242, 512))), " bias:", nz("layers.0.mlp.fc1.bias", slice(242, 512)))
print("L1 MLP units 91+ nonzero:", nz("layers.1.mlp.fc1.weight", (slice(91, 512), slice(None))), nz("layers.1.mlp.fc2.weight", (slice(None), slice(91, 512))), nz("layers.1.mlp.fc1.bias", slice(91, 512)))
print("L2 MLP units 13+ nonzero:", nz("layers.2.mlp.fc1.weight", (slice(13, 512), slice(None))), nz("layers.2.mlp.fc2.weight", (slice(None), slice(13, 512))), nz("layers.2.mlp.fc1.bias", slice(13, 512)))
print("L3 MLP units 8+ nonzero:", nz("layers.3.mlp.fc1.weight", (slice(8, 512), slice(None))), nz("layers.3.mlp.fc2.weight", (slice(None), slice(8, 512))), nz("layers.3.mlp.fc1.bias", slice(8, 512)))
# layer 3 heads 3-7
for h in range(3, 8):
    r = slice(16 * h, 16 * h + 16)
    print(f"L3 head {h}: q {nz('layers.3.attn.q.weight', (r, slice(None)))} qb {nz('layers.3.attn.q.bias', r)} k {nz('layers.3.attn.k.weight', (r, slice(None)))} kb {nz('layers.3.attn.k.bias', r)} v {nz('layers.3.attn.v.weight', (r, slice(None)))} o {nz('layers.3.attn.o.weight', (slice(None), r))} relbias nz {nz('layers.3.attn.rel_bias', h)}")
# layer 2 heads 4-7 current content
for h in range(4, 8):
    r = slice(16 * h, 16 * h + 16)
    print(f"L2 head {h}: qb {torch.nonzero(sd['layers.2.attn.q.bias'][r]).flatten().tolist()} k {torch.nonzero(sd['layers.2.attn.k.weight'][r]).tolist()} v {torch.nonzero(sd['layers.2.attn.v.weight'][r]).tolist()} o {torch.nonzero(sd['layers.2.attn.o.weight'][:, r]).tolist()}")
# all-zero rows in residual: which dims does nothing write? (approx: columns of o/fc2 and tok/off emb)
written = set()
for name in ("tok_emb.weight", "off_emb.weight"):
    written |= set(torch.nonzero(sd[name].abs().sum(0)).flatten().tolist())
for l in range(4):
    written |= set(torch.nonzero(sd[f"layers.{l}.attn.o.weight"].abs().sum(1)).flatten().tolist())
    written |= set(torch.nonzero(sd[f"layers.{l}.mlp.fc2.weight"].abs().sum(1)).flatten().tolist())
print("dims never written:", sorted(set(range(128)) - written))
# bias entries non-zero
for l in range(4):
    for n in ("attn.q.bias", "attn.k.bias", "attn.v.bias", "attn.o.bias", "mlp.fc2.bias"):
        c = nz(f"layers.{l}.{n}")
        if c:
            print(f"layers.{l}.{n} nonzero: {c}", torch.nonzero(sd[f'layers.{l}.{n}']).flatten().tolist())
print("head.weight nonzero:", torch.nonzero(sd["head.weight"]).tolist(), "bias", sd["head.bias"].tolist())
print("L2 MLP unit 12 fc1 nonzero:", torch.nonzero(sd["layers.2.mlp.fc1.weight"][12]).flatten().tolist(), sd["layers.2.mlp.fc1.bias"][12].item())
print("L1 MLP unit 0 fc1 nonzero:", torch.nonzero(sd["layers.1.mlp.fc1.weight"][0]).flatten().tolist(), sd["layers.1.mlp.fc1.bias"][0].item())
print("L1 MLP units 1,2 fc1 nonzero:", torch.nonzero(sd["layers.1.mlp.fc1.weight"][1]).flatten().tolist())
print("L1 MLP units 18,19 fc1 nonzero:", torch.nonzero(sd["layers.1.mlp.fc1.weight"][18]).flatten().tolist())
print("L1 MLP units 13,14 fc1/fc2:", torch.nonzero(sd["layers.1.mlp.fc1.weight"][13]).flatten().tolist(), torch.nonzero(sd["layers.1.mlp.fc2.weight"][:, 13]).flatten().tolist())
print("L2 head0 v rows 0-5 nonzero:", torch.nonzero(sd["layers.2.attn.v.weight"][0:6]).tolist())
print("L2 head1 v rows 16-21 nonzero:", torch.nonzero(sd["layers.2.attn.v.weight"][16:22]).tolist())
print("L3 head0 v rows 0-3 nonzero:", torch.nonzero(sd["layers.3.attn.v.weight"][0:4]).tolist(), " o cols 0-3:", torch.nonzero(sd["layers.3.attn.o.weight"][:, 0:4]).tolist())
print("L3 rel_bias row 0:", sd["layers.3.attn.rel_bias"][0].tolist()[:9], sd["layers.3.attn.rel_bias"][0, 64].item())
