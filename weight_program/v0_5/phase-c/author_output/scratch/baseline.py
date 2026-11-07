"""Baseline: Phase B's network on the V and W windows. Confusion matrices, macro-F1, false-alarm rate,
agreement with the disclosed network_class, and the list of wrong windows with logits."""
import sys
import v05_load
import model

CN = model.CLASS_NAMES


def f1s(conf):
    out = []
    for c in range(4):
        tp = conf[c][c]
        fp = sum(conf[r][c] for r in range(4)) - tp
        fn = sum(conf[c][k] for k in range(4)) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * p * r / (p + r) if p + r else 0.0)
    return out


def run(m, which, show_wrong=True):
    ws = v05_load.windows(which)
    conf = [[0] * 4 for _ in range(4)]
    mism = 0
    wrong = []
    fa_num = fa_den = 0
    for w in ws:
        lg = v05_load.eoe_logits(m, w["blocks"])[-1]
        c = int(model.predict(lg[None])[0])
        o = CN.index(w["oracle_label"])
        conf[o][c] += 1
        if CN[c] != w["network_class"]:
            mism += 1
        if o in (0, 1):
            fa_den += 1
            if c in (2, 3):
                fa_num += 1
        if c != o:
            wrong.append((w["id"], w["oracle_label"], CN[c], [round(float(x), 2) for x in lg]))
    f = f1s(conf)
    print(f"=== {which}: {len(ws)} windows ===")
    print("confusion (rows oracle, cols network) order", CN)
    for o in range(4):
        print(f"  {CN[o]:12s}", conf[o])
    print("per-class F1", [round(x, 3) for x in f], "macro-F1", round(sum(f) / 4, 4),
          "accuracy", round(sum(conf[i][i] for i in range(4)) / len(ws), 4),
          "false-alarm", round(fa_num / fa_den, 4) if fa_den else None)
    print("mismatches with disclosed network_class:", mism)
    if show_wrong:
        for t in wrong:
            print("  WRONG", t)
    return wrong


if __name__ == "__main__":
    m = v05_load.load_b()
    run(m, "V")
    run(m, "W")
