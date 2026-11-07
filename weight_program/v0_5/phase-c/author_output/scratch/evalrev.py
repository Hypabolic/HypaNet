"""Evaluate an edits file (applied on top of Phase B's weights) on the V and W windows.
Usage: ./py evalrev.py [edits-file | -] [which=V|W|all]
Prints the confusion matrix, per-class F1, macro-F1, false-alarm rate and the misclassified windows with logits."""
import sys

import model
import v05_load

CN = model.CLASS_NAMES


def evaluate(m, which):
    ws = v05_load.windows(which)
    conf = [[0] * 4 for _ in range(4)]
    wrong = []
    for w in ws:
        lg = v05_load.eoe_logits(m, w["blocks"])
        c = int(model.predict(lg[-1:])[0])
        o = CN.index(w["oracle_label"])
        conf[o][c] += 1
        if c != o:
            wrong.append((w["id"], w["oracle_label"][:4], CN[c][:4], [round(float(x), 1) for x in lg[-1]]))
    f1s = []
    for c in range(4):
        tp = conf[c][c]
        fp = sum(conf[o][c] for o in range(4)) - tp
        fn = sum(conf[c][p] for p in range(4)) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    n = sum(sum(r) for r in conf)
    pr = sum(sum(conf[o]) for o in (0, 1))
    fa = sum(conf[o][c] for o in (0, 1) for c in (2, 3)) / pr if pr else 0.0
    print(f"=== {which}: macro-F1 {sum(f1s) / 4:.4f}  accuracy {sum(conf[i][i] for i in range(4)) / n:.4f}  false-alarm {fa:.4f}  "
          f"per-class F1 {[round(x, 3) for x in f1s]}")
    print("    confusion (rows oracle, cols network):", conf)
    for r in wrong:
        print("    WRONG", r[0], r[1], "->", r[2], r[3])
    return conf, wrong


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "-"
    which = "all"
    for a in sys.argv[2:]:
        if a.startswith("which="):
            which = a[6:]
    if path == "-":
        m = v05_load.load_b()
    else:
        m, report = v05_load.apply_edits_file(path)
        probs = v05_load.build_problems(report)
        er = report["edit_report"]
        print(f"{path}: statements {er['statements']} applied {er['applied']} skipped {er['skipped']}")
        for p in probs:
            print("  PROBLEM:", p)
    for wh in (("V", "W") if which == "all" else (which,)):
        evaluate(m, wh)


if __name__ == "__main__":
    main()
