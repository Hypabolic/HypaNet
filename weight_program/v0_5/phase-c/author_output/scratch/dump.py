"""Dump windows with per-event network features (Phase B dim map) for reading.
Usage: ./py dump.py [edits|-] [ids... | all | wrong | LABEL] [which=V|W|all]
Writes to stdout. Per event: index, header (shortened), body lines, then feature values at the EOE:
  L1-MLP: NOOP d49, out d53, muth d58, FAIL d28, ok d29, err d114, pend d115
  L2-attn: gSAME d59 gLINE2 d60 gSAMEF d61 KV d62 gCNTOK d63 gCNT2 d64 gHasC d65 gREV d66 gREP d67 gPOLL2 d68 prevmut d69
           rHTTP d70 rPOLL d71 rSAMEF d72 rCNTOK d73 kpen d74
  L2-MLP: INEFF d75 RES d76 INPOL d77
  L3: A1 d79 A2 d80 act d81 resw d82 osc d83 sc d84 actb d85 reswb d86
"""
import sys

import torch

import model
import v05_load

CN = model.CLASS_NAMES
L2 = {"gSAME": 59, "gLINE2": 60, "gCNT15": 61, "gkpenC": 62, "gCNTOK": 63, "gCNT2": 64, "gHasC": 65, "gREV": 66,
      "gREP": 67, "gPOLL2": 68, "prevX": 69, "rHTTP": 70, "rPOLL": 71, "rFAILREP": 72, "rCNTOK": 73, "gkpenL": 74,
      "gH0": 78, "gPLANREP": 92, "gREPMUT": 93}
L2M = {"INEFF": 75, "RES": 76, "INPOL": 77, "FRESH": 94}
L3 = {"A1": 79, "A2": 80, "act": 81, "resw": 82, "A3": 95, "osc": 83, "sc": 84, "actb": 85, "reswb": 86, "freshb": 112}
L1 = {"NOOP": 49, "out": 53, "muth": 58, "FAIL": 28, "ok": 29, "err": 114, "pend": 115, "rf": 127}


def dump_window(m, w, before=3):
    blocks = w["blocks"]
    act = v05_load.activations(m, blocks, query_positions=[])
    eoe = act["eoe_index"]
    r1 = act["resid_mlp"][1]
    r2a = act["resid_attn"][2]
    r2 = act["resid_mlp"][2]
    r3 = act["resid_mlp"][3]
    lg = act["logits"]
    c = int(torch.argmax(lg[eoe[-1]]))
    print(f"=== {w['id']} oracle={w['oracle_label']} phaseB={w['network_class']} now={CN[c]} t={w['t']} start={w['start']} "
          f"logits={[round(float(x), 1) for x in lg[eoe[-1]]]}")
    for i in range(max(0, w["start"] - before), len(blocks)):
        p = eoe[i]
        mark = "*" if i >= w["start"] else " "
        lines = blocks[i].split("\n")
        hdr = lines[0].split(" ", 1)[1]
        print(f"{mark}{i:3d} {hdr}")
        for ln in lines[1:]:
            print(f"        {ln.strip()}")
        f1 = " ".join(f"{k}={float(r1[p, d]):.2g}" for k, d in L1.items() if abs(float(r1[p, d])) > 1e-6)
        f2 = " ".join(f"{k}={float(r2a[p, d]):.3g}" for k, d in L2.items() if abs(float(r2a[p, d])) > 1e-6)
        f2m = " ".join(f"{k}={float(r2[p, d]):.3g}" for k, d in L2M.items() if abs(float(r2[p, d])) > 1e-6)
        f3 = " ".join(f"{k}={float(r3[p, d]):.3g}" for k, d in L3.items() if abs(float(r3[p, d])) > 1e-6)
        print(f"        >> {CN[int(torch.argmax(lg[p]))][:4]} {[round(float(x), 1) for x in lg[p]]} | {f1} | {f2} | {f2m} | {f3}")


def main():
    edits = sys.argv[1] if len(sys.argv) > 1 else "-"
    sel = sys.argv[2:] or ["wrong"]
    which = "all"
    sel2 = []
    for s in sel:
        if s.startswith("which="):
            which = s[6:]
        else:
            sel2.append(s)
    sel = sel2
    m = v05_load.load_b() if edits == "-" else v05_load.apply_edits_file(edits)[0]
    ws = v05_load.windows(which)
    for w in ws:
        want = False
        if "all" in sel:
            want = True
        elif w["id"] in sel:
            want = True
        elif w["oracle_label"] in sel:
            want = True
        elif "wrong" in sel:
            c = v05_load.classify(m, w["blocks"])
            want = CN[c] != w["oracle_label"]
        if want:
            dump_window(m, w)


if __name__ == "__main__":
    main()
