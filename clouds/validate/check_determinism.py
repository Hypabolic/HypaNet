"""SPEC 6.4: the same inputs give the same PNG (and the same linear .npy).

    uv run python validate/check_determinism.py

Renders each case twice in separate render.py processes (fresh browser each time), plus once with a different
--budget-ms (which changes the pass / tile schedule but must not change the image), and compares bytes.
"""

from __future__ import annotations

import hashlib
import json

from common import OUT, VALIDATE, render

CASES = [
    ("net", str(VALIDATE / "test_networks" / "exact_single_net.json"), "backlit_golden"),
    ("net", str(VALIDATE / "test_networks" / "act_sin.json"), "side_sunset"),
    ("exact", None, "front_noon"),
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d = OUT / "determinism"
    d.mkdir(parents=True, exist_ok=True)
    res, ok_all = [], True
    for mode, net, pid in CASES:
        hashes = []
        for run, extra in (("a", []), ("b", []), ("c", ["--budget-ms", "12"])):
            out = d / f"{pid}_{mode}_{run}.png"
            args = ([net] if net else ["--mode", "exact-single"]) + ["--pair", pid, "--frames", "16", "--out", str(out),
                                                                       *extra]
            render(args)
            hashes.append((sha(out), sha(out.with_suffix(".npy"))))
        same = all(h == hashes[0] for h in hashes)
        ok_all &= same
        res.append({"mode": mode, "net": net, "pair": pid, "png_sha256": [h[0] for h in hashes],
                    "npy_sha256": [h[1] for h in hashes], "identical": same})
        print(f"[{'PASS' if same else 'FAIL'}] {pid} {mode}{' ' + net.split('/')[-1] if net else ''}: "
              f"3 renders (2 identical runs + budget 12 ms) -> png {hashes[0][0][:12]}..., "
              f"{'all identical' if same else 'DIFFER'}")
    (d / "summary.json").write_text(json.dumps(res, indent=1))
    print("deterministic" if ok_all else "NOT deterministic")


if __name__ == "__main__":
    main()
