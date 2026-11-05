"""Evaluate reasoned variants of policy_v5 on the disclosure (each a specific hypothesis):

A: progress excludes resolutions (V125); one retry is in policy (V177, V180, V185)
B: A + a retry links only to a same-key failure within the last 7 blocks (V142)
C: A + run_tests intent = first test when tests= is given, suite otherwise (V029); the kill count
   stays suite-level
D: B + C
"""
import sys
from collections import Counter

import policy_v5 as p5
from disc import CLASSES, load
from eval_v3 import blocks_of

VARIANT = {"horizon": False, "tests_aware": False}


def _intent_tests_aware(tb, tool):
    n = len(tb)
    if tool == "run_tests" and n > 11 and tb[11] == "tests":
        return p5._intent(tb, 13)
    return p5._intent(tb, 14 if tool == "search" else 10)


def analyse(blocks):
    feats = [p5._block_features(tb) for tb in blocks]
    if VARIANT["tests_aware"]:
        for tb, f in zip(blocks, feats):
            if f["tool"] == "run_tests":
                it = _intent_tests_aware(tb, f["tool"])
                f["key2"] = (f["cls"], it) if it is not None else None
            else:
                f["key2"] = f["key"]
    else:
        for f in feats:
            f["key2"] = f["key"]
    rel = []
    for b, f in enumerate(feats):
        lo = max(0, b - (p5.WINDOW - 1))
        rev = any(h in feats[e]["holds"] for e in range(lo, b) for h in f["hnews"])
        same = [i for i in range(b) if f["key"] is not None and feats[i]["key"] == f["key"]]
        same2 = [i for i in range(b) if f["key2"] is not None and feats[i]["key2"] == f["key2"]]
        prev_i = same2[-1] if same2 else None
        prev = feats[prev_i] if prev_i is not None else None
        exists = prev is not None
        maxpv = max([feats[i]["pv"] for i in same2], default=0)
        improved = f["val"] and exists and f["pv"] > maxpv
        ident = exists and prev["s_ok"] == f["s_ok"] and prev["s_fail"] == f["s_fail"] and prev["pv"] == f["pv"]
        nearest = 0
        for i in range(b - 1, -1, -1):
            if feats[i]["val"] or feats[i]["has_hunk"]:
                nearest = int(feats[i]["val"]) - int(feats[i]["has_hunk"])
                break
        near_enough = exists and (not VARIANT["horizon"] or b - prev_i <= p5.WINDOW - 1)
        retry = f["s_fail"] and near_enough and prev["s_fail"] and (nearest == 1 if f["val"] else True)
        c_fail = sum(1 for i in same if feats[i]["s_fail"])
        kill = retry and c_fail >= (p5.KILL_COUNT_POLL if f["poll"] else p5.KILL_COUNT)
        inpol = retry and not kill
        resolv = f["s_ok"] and exists and prev["s_fail"] and (rel[prev_i]["retry"] if f["val"] else True)
        improved = improved and not resolv
        stagnant = f["val"] and ident
        nv_stag = False
        for i in range(b - 1, -1, -1):
            if feats[i]["val"]:
                nv_stag = rel[i]["stagnant"]
                break
        futile = f["mut"] and f["has_hunk"] and not f["s_fail"] and nv_stag
        noop = f["mut"] and not f["has_hunk"] and not f["s_fail"]
        redinsp = f["insp"] and f["s_ok"] and exists
        si_revert = f["revert"] and not rev
        si = rev or si_revert or noop or futile or redinsp or (stagnant and not inpol) or (kill and not f["val"])
        rel.append({"rev": rev, "si": si, "inpol": inpol, "resolv": resolv, "kill": kill, "improved": improved,
                    "stagnant": stagnant, "futile": futile, "redinsp": redinsp, "noop": noop, "retry": retry,
                    "r": int(retry), "fail": f["s_fail"], "ident": ident})
    return feats, rel


def run(name, horizon, tests_aware, wins):
    VARIANT["horizon"], VARIANT["tests_aware"] = horizon, tests_aware
    c = Counter()
    mism = []
    for w in wins:
        blocks = blocks_of(w["enc"])
        feats, rel = analyse(blocks)
        pred = CLASSES[p5.decide(feats, rel, len(blocks) - 1)]
        c[(w["label"], pred)] += 1
        if pred != w["label"]:
            mism.append(f"{w['id']}:{w['label'][:4]}>{pred[:4]}")
    correct = sum(v for (a, b), v in c.items() if a == b)
    print(f"variant {name}: correct {correct}/200  mismatches: {' '.join(mism)}")


if __name__ == "__main__":
    wins = load()
    run("A", False, False, wins)
    run("B", True, False, wins)
    run("C", False, True, wins)
    run("D", True, True, wins)
