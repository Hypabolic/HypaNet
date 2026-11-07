"""Per-event feature dump of the rule evaluator for given windows. Usage: ./py rules_debug.py CONFIG ID [ID ...]"""
import sys

import v05_load
from rules import CONFIGS, decide, features, CN


def main():
    cfg = CONFIGS[sys.argv[1]]
    ids = sys.argv[2:]
    ws = [w for w in v05_load.windows("all") if w["id"] in ids]
    for w in ws:
        F = features(w["blocks"], cfg)
        c = decide(F, w["t"], cfg)
        print(f"=== {w['id']} oracle={w['oracle_label']} rule={CN[c]} t={w['t']} start={w['start']}")
        for i in range(max(0, w["start"] - 3), w["t"] + 1):
            f = F[i]
            hdr = w["blocks"][i].split("\n")[0].split(" ", 1)[1][:70]
            parts = " ".join(k for k, v in f["parts"].items() if v)
            flags = []
            if f["INPOL"]:
                flags.append("INPOL")
            if f["RES"]:
                flags.append("RES")
            if f["FRESHREAD"]:
                flags.append("FRESH")
            if f["outcome"]:
                flags.append("out")
            mark = "*" if i >= w["start"] else " "
            print(f"{mark}{i:3d} {hdr:70s} k={f['k']} kl={f['kline15']} kc={f['kcnt15']} revd={f['revd']} | {parts} | {' '.join(flags)}")


if __name__ == "__main__":
    main()
