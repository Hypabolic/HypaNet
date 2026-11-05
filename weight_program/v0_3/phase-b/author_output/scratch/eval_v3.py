"""Evaluate a candidate policy module on the 200 disclosed windows.

usage: eval_v3.py [-m module] summary
       eval_v3.py [-m module] cat ORACLE PRED [n]
       eval_v3.py [-m module] ids V001 ...        (window + 7 look-back blocks)
       eval_v3.py [-m module] full V001 ...       (every block of the prefix)
"""
import importlib
import sys
from collections import Counter

from analyze import short_block
from disc import CLASSES, load

FLAGS = ("rev", "si", "inpol", "resolv", "kill", "improved", "stagnant", "futile", "redinsp", "noop", "retry")


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


def dump(pv, w, full=False):
    blocks = blocks_of(w["enc"])
    feats, rel = pv.analyse(blocks)
    t = len(blocks) - 1
    pred = pv.decide(feats, rel, t)
    print(f"#### {w['id']} oracle={w['label']} cand={CLASSES[pred]} phaseA={w['net']} blocks={t + 1} start={w['start']}")
    lo = 0 if full else max(0, w["start"] - 7)
    for b in range(lo, t + 1):
        r = rel[b]
        fl = [k.upper() for k in FLAGS if r.get(k)]
        mark = ">" if b >= w["start"] else " "
        print(f" {mark}{b:3d} r{r.get('r', 0)} [{','.join(fl):28s}] {short_block(w['blocks'][b])[:112]}")


if __name__ == "__main__":
    args = sys.argv[1:]
    modname = "policy_v3"
    if args and args[0] == "-m":
        modname = args[1]
        args = args[2:]
    pv = importlib.import_module(modname)
    wins = load()
    mode = args[0] if args else "summary"
    preds = {w["id"]: pv.reference_policy(w["enc"].surface, w["enc"].event_ord) for w in wins}
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
        oracle, pred_name = args[1], args[2]
        limit = int(args[3]) if len(args) > 3 else 1000
        n = 0
        for w in wins:
            if w["label"] == oracle and CLASSES[preds[w["id"]]] == pred_name:
                dump(pv, w)
                n += 1
                if n >= limit:
                    break
    elif mode in ("ids", "full"):
        want = set(args[1:])
        for w in wins:
            if w["id"] in want:
                dump(pv, w, full=(mode == "full"))
