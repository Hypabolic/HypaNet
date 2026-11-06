"""Inspect the revised network on one disclosed window: per-event features at each EOE (revised dim map).
Usage: ./py inspect2.py Vnnn [edits-file]"""
import sys

import torch

import model
import v05_load

CN = model.CLASS_NAMES
L2 = {"gSAME": 59, "gLINE2": 60, "gSAMEF": 61, "KV": 62, "gCNTOK": 63, "gCNT2": 64, "gHasC": 65, "gREV": 66,
      "gREP": 67, "gPOLL2": 68, "prevmut": 69, "rHTTP": 70, "rPOLL": 71, "rSAMEF": 72, "rCNTOK": 73, "kpen": 74}
L2M = {"INEFF": 75, "RES": 76, "INPOL": 77}
L3 = {"A1": 79, "A2": 80, "act": 81, "resw": 82, "osc": 83, "sc": 84, "actb": 85, "reswb": 86}
L1 = {"err": 114, "pend": 115, "FAIL": 28, "ok": 29, "NOOP": 49, "out": 53, "muth": 58}


def main():
    wid = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else None
    m = v05_load.apply_edits_file(path)[0]
    w = [x for x in v05_load.windows() if x["id"] == wid][0]
    blocks = w["blocks"]
    print(wid, "oracle", w["oracle_label"], "t", w["t"], "start", w["start"])
    act = v05_load.activations(m, blocks, query_positions=[])
    eoe = act["eoe_index"]
    r1 = act["resid_mlp"][1]
    r2a = act["resid_attn"][2]
    r2 = act["resid_mlp"][2]
    r3 = act["resid_mlp"][3]
    lg = act["logits"]
    names1 = list(L1)
    names2 = list(L2)
    names2m = list(L2M)
    names3 = list(L3)
    print("ev  hdr | " + " ".join(f"{n:>5}" for n in names1) + " | " + " ".join(f"{n:>6}" for n in names2) + " | " +
          " ".join(f"{n:>5}" for n in names2m) + " | " + " ".join(f"{n:>5}" for n in names3) + " | logits")
    for i, p in enumerate(eoe):
        hdr = blocks[i].split("\n")[0].split(" ", 1)[1][:34]
        v1 = " ".join(f"{float(r1[p, d]):5.2f}" for d in L1.values())
        v2 = " ".join(f"{float(r2a[p, d]):6.3f}" for d in L2.values())
        v2m = " ".join(f"{float(r2[p, d]):5.2f}" for d in L2M.values())
        v3 = " ".join(f"{float(r3[p, d]):5.2f}" for d in L3.values())
        mark = "*" if i >= w["start"] else " "
        lgs = [round(float(x), 1) for x in lg[p]]
        print(f"{mark}{i:2d} {hdr:34s} | {v1} | {v2} | {v2m} | {v3} | {lgs} {CN[int(torch.argmax(lg[p]))][:4]}")


if __name__ == "__main__":
    main()
