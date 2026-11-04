"""Phase B fidelity check: network argmax vs reference_policy on self-constructed inputs.

Same input families as the A-selfcheck harness, pointed at output/phase-b-revision:
  1. exhaustive enumerations of 8-slot windows over small per-slot event vocabularies;
  2. a seeded random valid-window sweep respecting the spec 4 invariants;
  3. probes of the analogue gates (net validation delta near +0.02 and -0.1, fresh-failure term);
  4. batch-size consistency.
Also prints the policy's class on a few hand-written family windows.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent.parent.parent
PHASE = ROOT / "output" / "phase-b-revision"
CLASS = ("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING")


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, PHASE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


bw = load("build_weights")
rp = load("reference_policy")
NET = bw.build().eval()


def net_predict(X: np.ndarray) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 65536):
            logits = NET(torch.from_numpy(np.ascontiguousarray(X[i : i + 65536])))
            out.append(torch.argmax(logits, dim=-1).numpy())
    return np.concatenate(out)


def policy_predict(X: np.ndarray) -> np.ndarray:
    return np.array([rp.reference_policy(tuple(float(v) for v in row)) for row in X], dtype=np.int64)


CAT = {"INS": 0, "MUT": 1, "VAL": 2, "EXT": 3, "CTL": 4}
OUT = {"SUC": 5, "ERR": 6, "NEU": 7}


def ev(cat, out, ri=0, rf=0, wc=0, sr=0, vc=0, vd=0.0, tc=0.5):
    f = [0.0] * 15
    f[CAT[cat]] = 1.0
    f[OUT[out]] = 1.0
    f[8], f[9], f[10], f[11], f[12], f[13], f[14] = ri, rf, wc, sr, vc, vd, tc
    assert rf <= (1 if out == "ERR" else 0)
    assert sr <= wc
    assert vc <= (1 if cat == "VAL" else 0)
    assert vd == 0.0 or vc == 1
    return [float(v) for v in f]


def window(events) -> list[float]:
    first = list(events[0])
    first[8] = 0.0
    first[9] = 0.0
    rows = [first] + [list(e) for e in events[1:]]
    return [v for e in rows for v in e]


def describe(x) -> str:
    parts = []
    for s in range(8):
        e = x[15 * s : 15 * s + 15]
        cat = [k for k, i in CAT.items() if e[i] == 1.0]
        out = [k for k, i in OUT.items() if e[i] == 1.0]
        flags = "".join(n for n, i in (("R", 8), ("F", 9), ("W", 10), ("S", 11), ("C", 12)) if e[i] == 1.0)
        vd = f"{e[13]:+.3f}" if e[13] != 0.0 else ""
        parts.append(f"{cat[0] if cat else '?'}/{out[0] if out else '?'}{'/' + flags if flags else ''}{vd}")
    return " ".join(parts)


read = ev("INS", "SUC")
rep_read = ev("INS", "SUC", ri=1)
edit = ev("MUT", "SUC", wc=1)
edit_rev = ev("MUT", "SUC", ri=1, wc=1, sr=1)
ext_err = ev("EXT", "ERR")
ext_err_rep = ev("EXT", "ERR", ri=1, rf=1)
ext_ok_rep = ev("EXT", "SUC", ri=1)
poll_neu = ev("EXT", "NEU")
poll_neu_rep = ev("EXT", "NEU", ri=1)
plan = ev("CTL", "SUC")
plan_rep = ev("CTL", "SUC", ri=1)
run_new_fail = ev("VAL", "ERR")
run_stale = ev("VAL", "ERR", vc=1)
run_stale_rep = ev("VAL", "ERR", ri=1, rf=1, vc=1)
run_pass_rep0 = ev("VAL", "SUC", ri=1, vc=1)


def run_fail(ri, rf, vd):
    return ev("VAL", "ERR", ri=ri, rf=rf, vc=1, vd=vd)


def run_pass(ri, vd):
    return ev("VAL", "SUC", ri=ri, vc=1, vd=vd)


VOCABS = {
    "A stall/retry mix": [read, rep_read, ext_err_rep, run_stale_rep, edit_rev],
    "B recovery/resolution": [read, ext_err, ext_ok_rep, run_pass(1, 0.05), poll_neu_rep],
    "C progress/oscillation": [edit, edit_rev, run_fail(1, 0, -0.25), run_fail(1, 1, 0.25), ev("VAL", "ERR", ri=1, rf=1)],
    "D neutral/control/validation": [plan_rep, poll_neu, run_new_fail, run_pass_rep0, ev("MUT", "ERR", ri=1, rf=1)],
    "E fresh-vs-stale chains": [read, run_new_fail, run_stale, run_stale_rep, run_pass(1, 1.0)],
}


def run_enumerations() -> int:
    print("== 1. exhaustive enumerations over per-slot vocabularies (5^8 windows each) ==")
    total_bad = 0
    for vname, vocab in VOCABS.items():
        t0 = time.time()
        wins = [window([vocab[i] for i in combo]) for combo in itertools.product(range(5), repeat=8)]
        X = np.array(wins, dtype=np.float32)
        pn = net_predict(X)
        pp = policy_predict(X)
        bad = np.nonzero(pn != pp)[0]
        total_bad += len(bad)
        counts = np.bincount(pp, minlength=4)
        print(f"  {vname}: {len(X)} windows, disagreements {len(bad)}, policy class counts {counts.tolist()}  ({time.time() - t0:.0f}s)")
        for i in bad[:5]:
            print(f"    net={CLASS[pn[i]]} policy={CLASS[pp[i]]} :: {describe(X[i])}")
    return total_bad


def random_valid_windows(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 120), dtype=np.float32)
    for s in range(8):
        b = 15 * s
        cat = rng.integers(0, 5, n)
        out = rng.integers(0, 3, n)
        X[np.arange(n), b + cat] = 1.0
        X[np.arange(n), b + 5 + out] = 1.0
        err = (out == 1).astype(np.float32)
        val = (cat == 2).astype(np.float32)
        ri = (rng.random(n) < 0.5).astype(np.float32)
        rf = err * (rng.random(n) < 0.6)
        wc = (rng.random(n) < 0.5).astype(np.float32)
        sr = wc * (rng.random(n) < 0.4)
        vc = val * (rng.random(n) < 0.7)
        vd = vc * rng.uniform(-1.0, 1.0, n) * (rng.random(n) < 0.8)
        tc = rng.random(n)
        if s == 0:
            ri[:] = 0.0
            rf[:] = 0.0
        X[:, b + 8] = ri
        X[:, b + 9] = rf
        X[:, b + 10] = wc
        X[:, b + 11] = sr
        X[:, b + 12] = vc
        X[:, b + 13] = vd.astype(np.float32)
        X[:, b + 14] = tc.astype(np.float32)
    return X


def run_random(n: int, seed: int) -> int:
    print(f"== 2. random valid-window sweep: n={n}, seed={seed} ==")
    t0 = time.time()
    X = random_valid_windows(n, seed)
    pn = net_predict(X)
    pp = policy_predict(X)
    bad = np.nonzero(pn != pp)[0]
    counts = np.bincount(pp, minlength=4)
    print(f"  disagreements {len(bad)} / {n}, policy class counts {counts.tolist()}  ({time.time() - t0:.0f}s)")
    for i in bad[:10]:
        print(f"    net={CLASS[pn[i]]} policy={CLASS[pp[i]]} :: {describe(X[i])}")
    return len(bad)


def run_probes() -> int:
    print("== 3. analogue-gate probes ==")
    bad_total = 0
    stall_ctx = [run_stale] + [run_stale_rep] * 7
    retry_ctx = [read, read, read, read, run_stale, run_stale_rep, run_stale_rep, run_stale_rep]
    for name, ctx, slot, lo, hi in (
        ("progress gate (S near +0.02), stall context", stall_ctx, 7, 0.010, 0.030),
        ("regression gate (S near -0.1), stall context", stall_ctx, 7, -0.120, -0.080),
        ("fresh-failure term (recent VD near 0), retry context", retry_ctx, 4, -0.020, 0.020),
    ):
        mism = []
        grid = np.arange(lo, hi, 0.0001)
        for sval in grid:
            x = window(ctx)
            x[15 * slot + 13] = float(np.float32(sval))
            X = np.array([x], dtype=np.float32)
            a = int(net_predict(X)[0])
            b = int(policy_predict(X)[0])
            if a != b:
                mism.append((float(np.float32(sval)), CLASS[a], CLASS[b]))
        bad_total += len(mism)
        print(f"  {name}: {len(grid)} points, mismatches {len(mism)}" + (f" in [{mism[0][0]:.5f}, {mism[-1][0]:.5f}]" if mism else ""))
    return bad_total


def run_batch_consistency(n: int, seed: int) -> int:
    print("== 4. batch-size consistency (batch 1 vs batched) ==")
    X = random_valid_windows(n, seed)
    batched = net_predict(X)
    with torch.no_grad():
        single = np.array([int(torch.argmax(NET(torch.from_numpy(X[i : i + 1])), dim=-1)) for i in range(n)])
    bad = int((batched != single).sum())
    print(f"  {n} windows, differences {bad}")
    return bad


def run_hand_written() -> None:
    print("== 5. hand-written family windows (policy class; sanity only) ==")
    cases = [
        ("exploration, all novel", [read] * 7 + [edit]),
        ("two re-reads among novel work", [read, rep_read, read, read, rep_read, read, edit, read]),
        ("stale failure chain with edits, ends with edit", [edit, run_stale, edit, run_stale_rep, edit, run_stale_rep, edit, edit]),
        ("fresh failure, 3 immediate retries", [read, read, read, read, run_new_fail, run_stale_rep, run_stale_rep, run_stale_rep]),
        ("fresh failure, 4 retries (out of policy)", [read, read, read, run_new_fail, run_stale_rep, run_stale_rep, run_stale_rep, run_stale_rep]),
        ("transient external, resolved at slot 5", [read, read, read, ext_err, ext_err_rep, ext_ok_rep, read, edit]),
        ("polling in flight, started in window", [read, read, ev("EXT", "SUC"), poll_neu, poll_neu_rep, poll_neu_rep, poll_neu_rep, poll_neu_rep]),
        ("polling that filled the window", [poll_neu] + [poll_neu_rep] * 7),
        ("A/B/A/B oscillation with net positive delta", [edit, run_fail(0, 0, 0.3), edit, run_fail(1, 0, -0.1), edit_rev, run_fail(1, 1, 0.3), edit_rev, run_fail(1, 0, -0.1)]),
        ("repair loop with improvement", [run_new_fail, edit, run_fail(1, 1, 0.2), edit, run_fail(1, 0, 0.3), edit, run_pass(1, 0.5), read]),
        ("all zero input", None),
    ]
    rows = [[0.0] * 120 if e is None else window(e) for _, e in cases]
    X = np.array(rows, dtype=np.float32)
    pn = net_predict(X)
    pp = policy_predict(X)
    for (name, _), a, b in zip(cases, pn, pp):
        print(f"  net={CLASS[a]:11s} policy={CLASS[b]:11s} | {name}")


if __name__ == "__main__":
    b1 = run_enumerations()
    b2 = run_random(300000, 20261105)
    b3 = run_probes()
    b4 = run_batch_consistency(5000, 7)
    run_hand_written()
    print(f"== total net/policy disagreements: enumerations {b1}, random {b2}, probes {b3}, batch {b4} ==")
