"""Attention debug: for a window, event index and token offset, print the layer-L attention of head H (top keys)
and selected residual dims at that token. Usage: ./py attn_debug.py EDITS ID EVENT OFFSET LAYER HEAD [HEAD...]"""
import sys

import torch

import v05_load


def main():
    edits, wid, ev, off, layer = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    heads = [int(h) for h in sys.argv[6:]]
    m = v05_load.load_b() if edits == "-" else v05_load.apply_edits_file(edits)[0]
    w = [x for x in v05_load.windows("all") if x["id"] == wid][0]
    enc = v05_load.encode(w["blocks"])
    if off < 0:
        pos = enc.eoe_index[ev]
    else:
        pos = [i for i, (e, o) in enumerate(zip(enc.event_ord, enc.offset)) if e == ev + 1 and o == off][0]
    act = v05_load.activations(m, w["blocks"], query_positions=[pos])
    print(f"{wid} event {ev} offset {off} token {enc.surface[pos]!r} pos {pos}")
    r_in = act["embed"] if layer == 0 else act["resid_mlp"][layer - 1]
    r_attn = act["resid_attn"][layer]
    r_mlp = act["resid_mlp"][layer]
    ev_pos = [i for i, e in enumerate(enc.event_ord) if e == ev + 1]
    print(" tokens of the event with nonzero input d87-95 / d112 / d90:")
    for j in ev_pos:
        vals = [round(float(r_in[j, d]), 2) for d in (87, 88, 89, 90, 91, 92, 93, 94, 95, 112)]
        if any(abs(v) > 1e-6 for v in vals):
            print(f"   pos {j} off {enc.offset[j]} {enc.surface[j]!r}: {vals}")
    print(" input dims 32:48 ", [round(float(x), 2) for x in r_in[pos, 32:48]])
    print(" input dims 48:64 ", [round(float(x), 2) for x in r_in[pos, 48:64]])
    print(" after attn d26,d9:", round(float(r_attn[pos, 26]), 3), round(float(r_attn[pos, 9]), 3),
          " d8,d18,d1,d22:", [round(float(r_in[pos, d]), 2) for d in (8, 18, 1, 22)])
    print(" after mlp d89,d91,d122,d87,d88,d90:", [round(float(r_mlp[pos, d]), 3) for d in (89, 91, 122, 87, 88, 90)])
    for h in heads:
        a = act["attn"][layer][h, 0]
        top = torch.topk(a, 8)
        print(f" head {h} top keys:")
        for wgt, j in zip(top.values.tolist(), top.indices.tolist()):
            if wgt < 1e-4:
                continue
            print(f"   w={wgt:.4f} pos {j} ev {enc.event_ord[j] - 1} off {enc.offset[j]} {enc.surface[j]!r} block3A sum {float(r_in[j, 32:48].sum()):.2f} B sum {float(r_in[j, 48:64].sum()):.2f}")


if __name__ == "__main__":
    main()
