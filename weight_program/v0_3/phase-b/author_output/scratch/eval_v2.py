"""Evaluate scratch/policy_v2 on the 200 disclosed windows; dump mismatches by category.

usage: eval_v2.py                      summary
       eval_v2.py cat ORACLE PRED [n]  dump mismatching windows
       eval_v2.py ids V001 ...
"""
import sys
from collections import Counter

import policy_v2 as pv
from analyze import short_block
from disc import CLASSES, load


def blocks_of(enc):
    blocks = []
    cur, cur_ord = None, None
    for s, e in zip(enc.surface, enc.event_ord):
        if e == 0:
            continue
        if e != cur_ord:
            cur = []
            blocks.append(cur)
            cur_ord = e
        cur.append(s)
    return blocks


def dump(w):
    blocks = blocks_of(w["enc"])
    feats, rel = pv.analyse(blocks)
    t = len(blocks) - 1
    pred = pv.decide(feats, rel, t)
    print(f"#### {w['id']} oracle={w['label']} v2={CLASSES[pred]} phaseA={w['net']} blocks={t + 1} start={w['start']}")
    lo = max(0, w["start"] - 7)
    for b in range(lo, t + 1):
        r = rel[b]
        fl = [k.upper() for k in ("rev", "si", "inpol", "resolv", "kill", "improved", "stagnant", "futile", "redinsp", "redok", "noop") if r[k]]
        mark = ">" if b >= w["start"] else " "
        print(f" {mark}{b:3d} c{r['c_fail']} [{','.join(fl):30s}] {short_block(w['blocks'][b])[:110]}")


if __name__ == "__main__":
    wins = load()
    mode = sys.argv[1] if len(sys.argv) > 1 else "summary"
    preds = {}
    for w in wins:
        preds[w["id"]] = pv.reference_policy(w["enc"].surface, w["enc"].event_ord)
    if mode == "summary":
        c = Counter((w["label"], CLASSES[preds[w["id"]]]) for w in wins)
        for k in sorted(c):
            print(k, c[k])
        correct = sum(v for (a, b), v in c.items() if a == b)
        print("correct", correct, "/", len(wins))
        f1s = []
        for cl in CLASSES:
            tp = c[(cl, cl)]
            fn = sum(v for (a, b), v in c.items() if a == cl and b != cl)
            fp = sum(v for (a, b), v in c.items() if b == cl and a != cl)
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            f1s.append(f1)
            print(f"{cl:12s} tp={tp:2d} fn={fn:2d} fp={fp:2d} P={p:.2f} R={r:.2f} F1={f1:.2f}")
        print(f"macro-F1 {sum(f1s) / 4:.3f}")
        print("mismatches:", " ".join(w["id"] + ":" + w["label"][:4] + ">" + CLASSES[preds[w["id"]]][:4]
                                     for w in wins if CLASSES[preds[w["id"]]] != w["label"]))
    elif mode == "cat":
        oracle, pred_name = sys.argv[2], sys.argv[3]
        limit = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
        n = 0
        for w in wins:
            if w["label"] == oracle and CLASSES[preds[w["id"]]] == pred_name:
                dump(w)
                n += 1
                if n >= limit:
                    break
    elif mode == "ids":
        want = set(sys.argv[2:])
        for w in wins:
            if w["id"] in want:
                dump(w)
