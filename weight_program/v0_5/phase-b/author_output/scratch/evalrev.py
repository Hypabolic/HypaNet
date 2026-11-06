"""Evaluate an edits file (applied on top of r1) on the 200 disclosed windows.
Usage: ./py evalrev.py [edits-file | -] [verbose]
Prints the confusion matrix, per-class F1, macro-F1 and (verbose) the misclassified windows with logits."""
import sys

import torch

import model
import v05_load

CN = model.CLASS_NAMES


def evaluate(m, verbose=False, only=None):
    ws = v05_load.windows()
    conf = [[0] * 4 for _ in range(4)]
    wrong = []
    for w in ws:
        if only and w["id"] not in only:
            continue
        lg = v05_load.eoe_logits(m, w["blocks"])
        c = int(model.predict(lg[-1:])[0])
        o = CN.index(w["oracle_label"])
        conf[o][c] += 1
        if c != o:
            wrong.append((w["id"], w["oracle_label"], CN[c], [round(float(x), 2) for x in lg[-1]]))
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
    n = sum(sum(r) for r in conf)
    print("macro-F1:", round(sum(f1s) / 4, 4), " accuracy:", round(sum(conf[i][i] for i in range(4)) / n, 4))
    pr = sum(sum(conf[o]) for o in (0, 1))
    if pr:
        fa = sum(conf[o][c] for o in (0, 1) for c in (2, 3)) / pr
        print("false-alarm rate:", round(fa, 4))
    if verbose:
        for r in wrong:
            print("  WRONG", r[0], "oracle", r[1][:5], "net", r[2][:5], r[3])
    return conf, wrong


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "-"
    verbose = len(sys.argv) > 2
    if path == "-":
        m = v05_load.load_r1()
    else:
        m, report = v05_load.apply_edits_file(path)
        probs = v05_load.build_problems(report)
        er = report["edit_report"]
        print(f"{path}: statements {er['statements']} applied {er['applied']} skipped {er['skipped']}")
        for p in probs:
            print("  PROBLEM:", p)
    evaluate(m, verbose)


if __name__ == "__main__":
    main()
