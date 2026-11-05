"""Dump disclosed windows by (oracle, predicted) category with the policy's per-block flags.

usage: analyze.py summary
       analyze.py cat ORACLE PRED [limit]      e.g. cat STALLED PRODUCTIVE 10
       analyze.py ids V001 V002 ...
"""
import sys
from collections import Counter

import reference_policy as rp
from disc import CLASSES, load


def short_block(b):
    lines = b.split("\n")
    h = lines[0]
    # drop the timestamp and token counts for compactness
    h = h[13:]
    i = h.find(" in=")
    if i > 0:
        h = h[:i]
    body = []
    for ln in lines[1:]:
        ln = ln.strip()
        if ln.startswith("@"):
            parts = ln.split()
            if len(parts) >= 4:
                body.append(f"@{parts[1].split('/')[-1]} {parts[2][:6]}>{parts[4][:6]}")
            else:
                body.append(ln)
        elif ln.startswith("!"):
            body.append(ln[:40])
        else:
            body.append(ln)
    return h[:95] + ("  |" + "; ".join(body) if body else "")


def flags(w):
    """Policy internals for every block (as the policy sees the full prefix)."""
    enc = w["enc"]
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
    feats = [rp._block_features(tb) for tb in blocks]
    rel = [rp._relational(feats, b) for b in range(len(blocks))]
    return feats, rel


def dump(w):
    feats, rel = flags(w)
    t = len(w["blocks"]) - 1
    pred = rp.reference_policy(w["enc"].surface, w["enc"].event_ord)
    print(f"#### {w['id']} oracle={w['label']} phaseA_net={w['net']} pol_now={CLASSES[pred]} blocks={t + 1} start={w['start']}")
    lo = max(0, w["start"] - 7)
    for b in range(lo, t + 1):
        f, r = feats[b], rel[b]
        fl = []
        if r["rev"]:
            fl.append("REV")
        if r["si"]:
            fl.append("SI")
        if r["inpol"]:
            fl.append("INPOL")
        if r["resolv"]:
            fl.append("RESOLV")
        if r["kill"]:
            fl.append("KILL")
        mark = ">" if b >= w["start"] else " "
        print(f" {mark}{b:3d} [{','.join(fl):18s}] {short_block(w['blocks'][b])}")


if __name__ == "__main__":
    wins = load()
    mode = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if mode == "summary":
        c = Counter()
        for w in wins:
            pred = rp.reference_policy(w["enc"].surface, w["enc"].event_ord)
            c[(w["label"], CLASSES[pred])] += 1
        correct = sum(v for (a, b), v in c.items() if a == b)
        for k in sorted(c):
            print(k, c[k])
        print("correct", correct, "/", len(wins))
        # per-class recall / precision and macro-F1
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
    elif mode == "cat":
        oracle, pred_name = sys.argv[2], sys.argv[3]
        limit = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
        n = 0
        for w in wins:
            pred = rp.reference_policy(w["enc"].surface, w["enc"].event_ord)
            if w["label"] == oracle and CLASSES[pred] == pred_name:
                dump(w)
                n += 1
                if n >= limit:
                    break
    elif mode == "ids":
        want = set(sys.argv[2:])
        for w in wins:
            if w["id"] in want:
                dump(w)
