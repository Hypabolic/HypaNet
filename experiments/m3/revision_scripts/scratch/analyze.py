"""Phase B analysis: evaluate the disclosed 200 windows with the current Phase B build and policy.

Usage: ./py output/phase-b-revision/scratch/analyze.py [phase_dir] [--all]
Prints confusion matrices (oracle vs network, oracle vs policy), checks the current build
against the disclosed archived predictions, and lists every misclassified window with the
policy's internal signals. With --all, lists all 200 windows.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent.parent.parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
PHASE = Path(args[0]).resolve() if args else ROOT / "output" / "phase-b-revision"
SHOW_ALL = "--all" in sys.argv
TAG = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--tag=")), "current")


class _Tee:
    """Mirror stdout into scratch/analysis_<tag>.txt so the full report can be read back."""

    def __init__(self, path: Path):
        self._out = sys.stdout
        self._file = open(path, "w")

    def write(self, s):
        self._out.write(s)
        self._file.write(s)

    def flush(self):
        self._out.flush()
        self._file.flush()


sys.stdout = _Tee(ROOT / "output" / "phase-b-revision" / "scratch" / f"analysis_{TAG}.txt")
CLASS = ("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING")
SHORT = ("PROD", "REC", "STALL", "OSC")


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, PHASE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


bw = load("build_weights")
rp = load("reference_policy")
NET = bw.build().eval()

disc = json.loads((ROOT / "disclosure" / "DISCLOSURE.json").read_text())
wins = disc["windows"]
ids = [w["id"] for w in wins]
X = np.array([[v for row in w["features"] for v in row] for w in wins], dtype=np.float32)
oracle = np.array([CLASS.index(w["oracle_label"]) for w in wins])
disc_net = np.array([CLASS.index(w["network_class"]) for w in wins])
disc_pol = np.array([CLASS.index(w["reference_policy_class"]) for w in wins])

with torch.no_grad():
    net_pred = torch.argmax(NET(torch.from_numpy(X)), dim=-1).numpy()
pol_pred = np.array([rp.reference_policy(tuple(float(v) for v in row)) for row in X])

print(f"phase dir: {PHASE}")
print(f"current build == disclosed network class on all 200: {bool((net_pred == disc_net).all())}")
print(f"current policy == disclosed policy class on all 200: {bool((pol_pred == disc_pol).all())}")
print(f"network == policy on all 200: {bool((net_pred == pol_pred).all())}")


def confusion(name, pred):
    print(f"\n{name}: rows = oracle, cols = predicted ({', '.join(SHORT)})")
    m = np.zeros((4, 4), dtype=int)
    for o, p in zip(oracle, pred):
        m[o, p] += 1
    for i in range(4):
        print(f"  {SHORT[i]:6s} " + " ".join(f"{m[i, j]:4d}" for j in range(4)) + f"   recall {m[i, i] / m[i].sum():.2f}")
    acc = np.trace(m) / m.sum()
    f1s = []
    for c in range(4):
        tp = m[c, c]
        fp = m[:, c].sum() - tp
        fn = m[c].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    false_alarm = (m[0, 2] + m[0, 3] + m[1, 2] + m[1, 3]) / (m[0].sum() + m[1].sum())
    print(f"  accuracy {acc:.3f}  macro-F1 {np.mean(f1s):.3f}  per-class F1 {[round(f, 2) for f in f1s]}  false-loop-alarm {false_alarm:.3f}")
    return m


confusion("ORACLE vs NETWORK (current build)", net_pred)
confusion("ORACLE vs POLICY (current policy)", pol_pred)

# ----------------------------------------------------------- internals ----
CAT = ("INS", "MUT", "VAL", "EXT", "CTL")
OUTC = ("SUC", "ERR", "NEU")


def relu(v):
    return v if v > 0 else 0.0


def clip01(v):
    return relu(v) - relu(v - 1.0)


def internals(x):
    """Phase B revision signals (mirrors reference_policy.py)."""
    f = lambda s, o: float(x[15 * s + o])
    R = sum(relu(f(s, 8) + f(s, 9) + f(s, 11) - f(s, 7) - 3.0 * f(s, 3)) for s in range(1, 8))
    resA = 0.0
    resB = 0.0
    for s in (4, 5, 6, 7):
        pa = sum(f(j, 6) + f(j, 7) for j in range(s))
        resA += relu(f(s, 8) + f(s, 5) + f(s, 3) + f(s, 2) - f(s, 12) - 3.0 + pa / 8.0)
        pb = sum(f(j, 9) for j in range(s))
        resB += relu(f(s, 8) + f(s, 5) + f(s, 12) - 3.0 + pb / 8.0)
    RT = clip01(sum(f(s, 9) for s in (4, 5, 6, 7)))
    NF = relu(sum(f(s, 6) - f(s, 12) for s in range(8)) - 10.0 * sum(f(s, 13) for s in (4, 5, 6, 7)))
    V = sum(f(s, 11) for s in range(8))
    S = sum(f(s, 13) for s in range(8))
    VM1 = relu(V - 1.0)
    PROG = relu(100.0 * S - 1.0)
    NZ = relu(-20.0 * S - 1.0)
    RFs = sum(f(s, 9) for s in range(8))
    OOP = relu(RFs - 3.0)
    POLLend = relu(f(7, 8) + f(7, 7) + f(7, 3) - 2.0)
    OVER = relu(sum(f(s, 7) for s in range(8)) + 3.0 * f(0, 7) - 6.0)
    veto = 40.0 * (PROG + NZ + VM1)
    SA = clip01(R + 2.0 * OOP - 1.0 - veto)
    SB = clip01(2.0 * POLLend + OVER - 2.0 - veto)
    ACT = relu(2.0 * RT + NF / 32.0 - 2.0 - 10.0 * OOP)
    RES = clip01(8.0 * (resA + resB))
    POLL = relu(POLLend - OVER)
    return dict(R=R, V=V, S=round(S, 4), RT=RT, NF=round(NF, 2), RF=RFs, OOP=OOP, SA=SA, SB=SB,
                ACT=round(ACT, 3), RES=RES, POLL=POLL, OSC=VM1)


def describe(x):
    parts = []
    for s in range(8):
        e = x[15 * s : 15 * s + 15]
        cat = CAT[int(np.argmax(e[0:5]))]
        out = OUTC[int(np.argmax(e[5:8]))]
        flags = "".join(n for n, i in (("R", 8), ("F", 9), ("W", 10), ("S", 11), ("C", 12)) if e[i] == 1.0)
        vd = f"{e[13]:+.2f}" if e[13] != 0.0 else ""
        parts.append(f"{cat}/{out}{'/' + flags if flags else ''}{vd}~{e[14]:.1f}")
    return "  ".join(parts)


print("\n== windows ==  (per slot: CAT/OUT/flags[R=rep_intent F=rep_failure W=ws_changed S=state_revisited C=comparable]delta~tokencost)")
by_pair = Counter()
for i in range(len(wins)):
    wrong = pol_pred[i] != oracle[i]
    by_pair[(SHORT[oracle[i]], SHORT[pol_pred[i]])] += 1
    if SHOW_ALL or wrong:
        mark = "XX" if wrong else "ok"
        print(f"{ids[i]} {mark} oracle={SHORT[oracle[i]]:5s} net={SHORT[net_pred[i]]:5s} policy={SHORT[pol_pred[i]]:5s}")
        print("     " + describe(X[i]))
        print("     " + " ".join(f"{k}={v}" for k, v in internals(X[i]).items()))

print("\n== oracle -> policy pair counts ==")
for (o, p), n in sorted(by_pair.items()):
    print(f"  {o:5s} -> {p:5s}: {n}")
