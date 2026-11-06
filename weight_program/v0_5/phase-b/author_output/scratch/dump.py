"""Dump disclosed windows compactly to scratch/dump_<label>.txt: for each window, the blocks from start-3 to t,
with the window marker, plus per-event r1(+edits) features. Usage: ./py dump.py [edits|-] [LABEL ...]"""
import os
import sys

import torch

import model
import v05_load

CN = model.CLASS_NAMES
OUT = os.path.join(v05_load.OUTPUT, "scratch")
FEATS = {"err": 114, "pend": 115, "REV": 122, "REP": 123, "CNT": 124, "FNC": 40, "INEFF": 125, "fail": 126,
         "NOOP": 127, "lines": 24, "hunk": 31}


def short(block):
    lines = block.split("\n")
    h = lines[0]
    # drop the timestamp and the in=/out= tail for compactness
    parts = h.split(" ", 1)
    h = parts[1] if len(parts) > 1 else h
    body = [ln.strip() for ln in lines[1:]]
    return h, body


def main():
    edits = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    labels = sys.argv[2:] or list(CN)
    m = v05_load.apply_edits_file(edits)[0] if edits else v05_load.load_r1()
    ws = v05_load.windows()
    for lab in labels:
        path = os.path.join(OUT, f"dump_{lab}.txt")
        with open(path, "w") as f:
            for w in ws:
                if w["oracle_label"] != lab:
                    continue
                blocks = w["blocks"]
                act = v05_load.activations(m, blocks, query_positions=[])
                eoe = act["eoe_index"]
                l2 = act["resid_mlp"][2]
                lg = act["logits"]
                c = int(torch.argmax(lg[eoe[-1]]))
                f.write(f"=== {w['id']} oracle={lab} net={CN[c]} t={w['t']} start={w['start']} "
                        f"logits={[round(float(x), 1) for x in lg[eoe[-1]]]}\n")
                lo = max(0, w["start"] - 3)
                for i in range(lo, len(blocks)):
                    h, body = short(blocks[i])
                    mark = "*" if i >= w["start"] else " "
                    p = eoe[i]
                    fv = " ".join(f"{k}={float(l2[p, d]):.1f}" for k, d in FEATS.items() if abs(float(l2[p, d])) > 0.05)
                    pred = CN[int(torch.argmax(lg[p]))][:4]
                    f.write(f"{mark}[{i:2d}] {h}   || {pred} {fv}\n")
                    for b in body:
                        f.write(f"        {b}\n")
                f.write("\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
