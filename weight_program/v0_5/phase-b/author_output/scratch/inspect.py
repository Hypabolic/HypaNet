"""Inspect r1's (or a revised) network on one disclosed window: per-event features at each EOE, and per-token
values of key residual dims for the last few events. Usage: ./py inspect.py V001 [edits-file] [n_last_events]"""
import os
import sys

import torch

import model
import v05_load

CN = model.CLASS_NAMES

# residual dims of interest at the EOE (after each layer), by r1's allocation
EOE_DIMS = {
    "err114": 114, "pend115": 115, "REVev122": 122, "REPev123": 123, "CNTev124": 124, "FNCev40": 40,
    "INEFF125": 125, "fail126": 126, "NOOP127": 127, "lines24": 24, "hunk31": 31,
    "A1_32": 32, "A2_33": 33, "A3_34": 34, "A4_35": 35, "osc36": 36, "sc37": 37, "rf38": 38, "sh39": 39,
}
TOK_DIMS = {
    "hash1": 1, "dig2": 2, "gt3": 3, "NL@4": 4, "p>25": 25, "pF26": 26, "pp@27": 27, "bcRL28": 28, "bcS29": 29,
    "bcM30": 30, "hunk31": 31, "st112": 112, "FNC113": 113, "REVA116": 116, "REVB117": 117, "MATCH118": 118,
    "REVg119": 119, "REPg120": 120, "CNTg121": 121, "NOOP127": 127, "err114": 114, "pend115": 115, "lines24": 24,
}


def main():
    wid = sys.argv[1]
    edits = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    nlast = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    if edits:
        m, _ = v05_load.apply_edits_file(edits)
    else:
        m = v05_load.load_r1()
    w = [x for x in v05_load.windows() if x["id"] == wid][0]
    blocks = w["blocks"]
    print(wid, "oracle", w["oracle_label"], "disclosed net", w["network_class"], "t", w["t"], "start", w["start"])
    for i, b in enumerate(blocks):
        mark = "*" if i >= w["start"] else " "
        print(f"{mark}[{i:2d}] " + b.replace("\n", "\n      "))
    act = v05_load.activations(m, blocks, query_positions=[])
    eoe = act["eoe_index"]
    surf = act["surface"]
    final = act["resid_mlp"][3]
    l2 = act["resid_mlp"][2]
    print("\nper-event features at EOE (after layer 2 MLP for event feats; after layer 3 for A/ramps):")
    names = list(EOE_DIMS)
    print("ev   " + " ".join(f"{n:>9}" for n in names) + "   logits")
    for i, p in enumerate(eoe):
        vals = []
        for n in names:
            d = EOE_DIMS[n]
            src = final if n.startswith(("A", "osc", "sc", "rf", "sh")) else l2
            vals.append(float(src[p, d]))
        lg = [round(float(x), 2) for x in act["logits"][p]]
        print(f"{i:3d}  " + " ".join(f"{v:9.3f}" for v in vals) + "   " + str(lg) + " " + CN[int(torch.argmax(act['logits'][p]))][:5])
    # per-token dump of the last nlast events, after layer 1 MLP (resid_mlp[1]) and after layer 0 MLP
    l0 = act["resid_mlp"][0]
    l1 = act["resid_mlp"][1]
    n = len(surf)
    start_tok = eoe[-nlast - 1] + 1 if len(eoe) > nlast else 1
    tn = list(TOK_DIMS)
    print("\nper-token (after layer 1 MLP) for the last", nlast, "events:")
    print("pos  surface                        " + " ".join(f"{x:>8}" for x in tn))
    for p in range(start_tok, n):
        vals = [float(l1[p, TOK_DIMS[x]]) for x in tn]
        print(f"{p:4d} {surf[p][:30]:30s} " + " ".join(f"{v:8.2f}" for v in vals))


if __name__ == "__main__":
    main()
