"""Print the attention weights of one head at one query token of a disclosed window (revised network).
Usage: ./py attn_debug.py Vnnn LAYER HEAD EVENT SURFACE_SUBSTRING [edits]"""
import sys

import torch

import v05_load


def main():
    wid, layer, head, ev, sub = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    path = sys.argv[6] if len(sys.argv) > 6 else None
    m = v05_load.apply_edits_file(path)[0]
    w = [x for x in v05_load.windows() if x["id"] == wid][0]
    enc = v05_load.encode(w["blocks"])
    # query position: the first token of event ev whose surface contains sub
    qpos = None
    for i, (s, e) in enumerate(zip(enc.surface, enc.event_ord)):
        if e != ev + 1:
            continue
        if sub == "LINE":
            if s.isdigit() and enc.surface[i - 1] == ":" and enc.offset[i] > 7 and not enc.surface[i - 2].isdigit():
                qpos = i
                break
        elif sub in s:
            qpos = i
            break
    if qpos is None:
        print("no such token")
        return
    act = v05_load.activations(m, w["blocks"], query_positions=[qpos])
    a = act["attn"][layer][head, 0]
    print(f"query token {qpos} '{enc.surface[qpos]}' event {ev} offset {enc.offset[qpos]}")
    top = torch.argsort(a, descending=True)[:12]
    for j in top:
        j = int(j)
        print(f"  w={float(a[j]):.4f} pos {j} ev {enc.event_ord[j] - 1} off {enc.offset[j]} '{enc.surface[j]}'")
    # residual values at the query token after the layer's attention
    r = act["resid_attn"][layer][qpos]
    print("resid_attn dims 113,116-120,48,26:", [round(float(r[d]), 3) for d in (113, 116, 117, 118, 119, 120, 48, 26)])


if __name__ == "__main__":
    main()
