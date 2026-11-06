"""Baseline: r1's network on the 200 disclosed windows. Confusion matrix, per-class F1, agreement with the
disclosed network_class field, and a per-window listing with the EOE logits of the last block."""
import json
import os
import sys

import torch

import model
import v05_load

CN = model.CLASS_NAMES
OUT = os.path.join(v05_load.OUTPUT, "scratch")


def main():
    m = v05_load.load_r1()
    ws = v05_load.windows()
    conf = [[0] * 4 for _ in range(4)]
    rows = []
    mismatch = 0
    for w in ws:
        lg = v05_load.eoe_logits(m, w["blocks"])
        c = int(model.predict(lg[-1:])[0])
        o = CN.index(w["oracle_label"])
        conf[o][c] += 1
        if CN[c] != w["network_class"]:
            mismatch += 1
        rows.append((w["id"], w["oracle_label"], w["network_class"], CN[c], [round(float(x), 2) for x in lg[-1]],
                     w["t"], w["start"]))
    print("disclosed network_class mismatches:", mismatch)
    print("confusion (rows oracle, cols network):")
    print("        " + " ".join(f"{n[:5]:>6}" for n in CN))
    for i in range(4):
        print(f"{CN[i][:7]:>7} " + " ".join(f"{conf[i][j]:6d}" for j in range(4)))
    f1s = []
    for c in range(4):
        tp = conf[c][c]
        fp = sum(conf[o][c] for o in range(4)) - tp
        fn = sum(conf[c][p] for p in range(4)) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1)
        print(f"{CN[c]:>12}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print("macro-F1:", round(sum(f1s) / 4, 4))
    acc = sum(conf[i][i] for i in range(4)) / 200
    print("accuracy:", acc)
    fa = sum(conf[o][c] for o in (0, 1) for c in (2, 3)) / sum(sum(conf[o]) for o in (0, 1))
    print("false-alarm rate:", round(fa, 4))
    with open(os.path.join(OUT, "baseline_rows.json"), "w") as f:
        json.dump(rows, f)
    print()
    for r in rows:
        print(r[0], r[1][:5], "net", r[3][:5], "disc", r[2][:5], "logits", r[4], "t", r[5], "start", r[6])


if __name__ == "__main__":
    main()
