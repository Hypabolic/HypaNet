"""Phase A-selfcheck harness: network argmax vs reference_policy on self-constructed inputs.

Inputs used (all constructed here, no data, no labels):
  1. hand-written windows, one or more per scenario family in spec 7.1, with the
     class my own policy is meant to return;
  2. exhaustive enumerations of 8-slot windows over small per-slot event vocabularies
     (branch-condition enumerations of my policy);
  3. a seeded random valid-window sweep that respects the spec 4 invariants;
  4. a probe of the net-validation-delta ramp around the 0.0175 threshold;
  5. a batch-size consistency check (batch 1 vs batched forward).
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
PHASE = ROOT / "output" / "phase-a-selfcheck"
sys.dont_write_bytecode = True

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
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def policy_predict(X: np.ndarray) -> np.ndarray:
    return np.array([rp.reference_policy(tuple(float(v) for v in row)) for row in X], dtype=np.int64)


# ---------------------------------------------------------------- events ----
CAT = {"INS": 0, "MUT": 1, "VAL": 2, "EXT": 3, "CTL": 4}
OUT = {"SUC": 5, "ERR": 6, "NEU": 7}


def ev(cat, out, ri=0, rf=0, wc=0, sr=0, vc=0, vd=0.0, tc=0.5):
    f = [0.0] * 15
    f[CAT[cat]] = 1.0
    f[OUT[out]] = 1.0
    f[8], f[9], f[10], f[11], f[12], f[13], f[14] = ri, rf, wc, sr, vc, vd, tc
    assert rf <= (1 if out == "ERR" else 0), "repeated_failure <= out_ERROR"
    assert sr <= wc, "state_revisited <= workspace_changed"
    assert vc <= (1 if cat == "VAL" else 0), "validation_comparable <= cat_VALIDATE"
    assert vd == 0.0 or vc == 1, "validation_delta != 0 => validation_comparable"
    return [float(v) for v in f]


def window(events) -> list[float]:
    assert len(events) == 8
    first = list(events[0])
    first[8] = 0.0  # slot 0 never has repeated_intent / repeated_failure
    first[9] = 0.0
    rows = [first] + [list(e) for e in events[1:]]
    return [v for e in rows for v in e]


def describe(x) -> str:
    parts = []
    for s in range(8):
        e = x[15 * s : 15 * s + 15]
        cat = [k for k, i in CAT.items() if e[i] == 1.0]
        out = [k for k, i in OUT.items() if e[i] == 1.0]
        flags = "".join(
            n for n, i in (("R", 8), ("F", 9), ("W", 10), ("S", 11), ("C", 12)) if e[i] == 1.0
        )
        vd = f"{e[13]:+.3f}" if e[13] != 0.0 else ""
        parts.append(f"{s}:{cat[0] if cat else '?'}/{out[0] if out else '?'}{'/' + flags if flags else ''}{vd}")
    return " ".join(parts)


# ----------------------------------------------------------- vocabulary ----
read = ev("INS", "SUC")
rep_read = ev("INS", "SUC", ri=1)
search_neu = ev("INS", "NEU")
edit = ev("MUT", "SUC", wc=1)
edit_rep = ev("MUT", "SUC", ri=1, wc=1)
edit_rev = ev("MUT", "SUC", ri=1, wc=1, sr=1)
revert = ev("MUT", "SUC", wc=1, sr=1)
run_first_fail = ev("VAL", "ERR")
build_fail = ev("VAL", "ERR")
build_fail_rep = ev("VAL", "ERR", ri=1, rf=1)
build_ok_rep = ev("VAL", "SUC", ri=1)
ext_err = ev("EXT", "ERR")
ext_err_rep = ev("EXT", "ERR", ri=1, rf=1)
ext_ok = ev("EXT", "SUC")
ext_ok_rep = ev("EXT", "SUC", ri=1)
poll_neu = ev("EXT", "NEU")
poll_neu_rep = ev("EXT", "NEU", ri=1)
plan = ev("CTL", "NEU")
plan_rep = ev("CTL", "NEU", ri=1)


def run_fail(ri, rf, vd):
    return ev("VAL", "ERR", ri=ri, rf=rf, vc=1, vd=vd)


def run_pass(ri, vd):
    return ev("VAL", "SUC", ri=ri, vc=1, vd=vd)


# ---------------------------------------------------------- scenarios -----
SCENARIOS = [
    ("F1 repair improving, green at end",
     [read, run_first_fail, edit, run_fail(1, 1, 0.2), edit, run_fail(1, 0, 0.3), edit, run_pass(1, 0.5)], 0),
    ("F1 repair early, no delta yet",
     [read, read, edit, run_first_fail, edit, run_fail(1, 1, 0.0), edit, run_fail(1, 1, 0.0)], 0),
    ("F2 exploration, all novel then edit",
     [read, read, search_neu, read, read, read, read, edit], 0),
    ("F3 transient ext, mid retry (2 retries)",
     [read, read, read, edit, read, ext_err, ext_err_rep, ext_err_rep], 1),
    ("F3 transient ext, resolved at endpoint",
     [read, read, read, edit, ext_err, ext_err_rep, ext_err_rep, ext_ok_rep], 1),
    ("F3 resolved at slot 4 (within k_rec)",
     [ext_err, ext_err_rep, ext_err_rep, read, ext_ok_rep, read, edit, read], 1),
    ("F3 resolved at slot 3 (outside k_rec)",
     [read, ext_err, ext_err_rep, ext_ok_rep, read, edit, read, edit], 0),
    ("F4 diagnosed repair via build",
     [read, edit, build_fail, read, read, edit, build_ok_rep, read], 1),
    ("F4 build retry in policy (3 retries)",
     [read, edit, read, read, build_fail, build_fail_rep, build_fail_rep, build_fail_rep], 1),
    ("F4 build retry out of policy (4 retries)",
     [read, edit, read, build_fail, build_fail_rep, build_fail_rep, build_fail_rep, build_fail_rep], 2),
    ("F5 repeated failing tests, full window",
     [run_fail(0, 0, 0.0)] + [run_fail(1, 1, 0.0)] * 7, 2),
    ("F5 with interleaved re-reads",
     [run_fail(0, 0, 0.0), read, run_fail(1, 1, 0.0), rep_read, run_fail(1, 1, 0.0), rep_read, run_fail(1, 1, 0.0), rep_read], 2),
    ("F5 early (3 repeats)",
     [read, read, edit, read, run_fail(0, 0, 0.0), run_fail(1, 1, 0.0), run_fail(1, 1, 0.0), run_fail(1, 1, 0.0)], 2),
    ("F5 two repeats only",
     [read, read, edit, read, read, run_fail(0, 0, 0.0), run_fail(1, 1, 0.0), run_fail(1, 1, 0.0)], 0),
    ("F6 same edit re-applied, same failure",
     [edit, run_fail(0, 0, 0.0), edit_rep, run_fail(1, 1, 0.0), edit_rep, run_fail(1, 1, 0.0), edit_rep, run_fail(1, 1, 0.0)], 2),
    ("F6 variant edits, same failure",
     [edit, run_fail(0, 0, 0.0), edit, run_fail(1, 1, 0.0), edit, run_fail(1, 1, 0.0), edit, run_fail(1, 1, 0.0)], 2),
    ("F7 A/B/A/B workspace oscillation",
     [edit, run_fail(0, 0, 0.2), edit, run_fail(1, 0, -0.2), edit_rev, run_fail(1, 1, 0.2), edit_rev, run_fail(1, 0, -0.2)], 3),
    ("F7 oscillation but net progress",
     [edit, run_fail(0, 0, 0.3), edit, run_fail(1, 0, -0.1), edit_rev, run_fail(1, 1, 0.3), edit_rev, run_fail(1, 0, -0.1)], 0),
    ("F8 tool oscillation X/Y undo",
     [edit, revert, edit_rev, edit_rev, edit_rev, edit_rev, edit_rev, edit_rev], 3),
    ("F8 oscillation starting late (V=2)",
     [read, read, read, read, edit, edit, edit_rev, edit_rev], 3),
    ("single reversion at end",
     [read, read, read, read, read, edit, edit, revert], 0),
    ("F9 lookalike: repeated runs improving",
     [run_first_fail, edit, run_fail(1, 1, 0.1), edit, run_fail(1, 1, 0.1), edit, run_fail(1, 0, 0.2), run_pass(1, 0.1)], 0),
    ("F10 neutral polls in flight (endpoint poll)",
     [read, edit, ext_err, poll_neu, poll_neu_rep, poll_neu_rep, poll_neu_rep, poll_neu_rep], 1),
    ("F10 polls then success",
     [edit, ext_err, poll_neu, poll_neu_rep, poll_neu_rep, poll_neu_rep, ext_ok_rep, read], 1),
    ("F10 error polls x5 after initial (in policy)",
     [read, edit, ext_err, ext_err_rep, ext_err_rep, ext_err_rep, ext_err_rep, ext_err_rep], 1),
    ("F10 error polls x6 after initial (out of policy)",
     [read, ext_err, ext_err_rep, ext_err_rep, ext_err_rep, ext_err_rep, ext_err_rep, ext_err_rep], 2),
    ("F10 re-reads while waiting (known limitation)",
     [ext_err, read, rep_read, rep_read, rep_read, rep_read, rep_read, rep_read], 2),
    ("F11 rising-cost re-reads",
     [read, ev("INS", "SUC", ri=1, tc=0.6), ev("INS", "SUC", ri=1, tc=0.7), ev("INS", "SUC", ri=1, tc=0.8),
      ev("INS", "SUC", ri=1, tc=0.9), ev("INS", "SUC", ri=1, tc=0.95), ev("INS", "SUC", ri=1, tc=1.0), ev("INS", "SUC", ri=1, tc=1.0)], 2),
    ("all novel inspects", [read] * 8, 0),
    ("repeated plans (neutral)", [plan] + [plan_rep] * 7, 0),
    ("stall but endpoint novel", [run_fail(0, 0, 0.0)] + [run_fail(1, 1, 0.0)] * 6 + [read], 0),
    ("stall evidence + fresh ext error at endpoint", [read] + [rep_read] * 6 + [ext_err], 1),
    ("redundant reads + in-policy ext retries at end", [read, rep_read, rep_read, rep_read, rep_read, ext_err, ext_err_rep, ext_err_rep], 1),
    ("first session test failure in recent slots", [read, read, read, read, run_first_fail, edit, read, edit], 1),
    ("all zero input", None, 0),
]


def run_scenarios() -> int:
    print("== 1. hand-written scenarios ==")
    rows = []
    for name, events, intended in SCENARIOS:
        x = [0.0] * 120 if events is None else window(events)
        rows.append((name, x, intended))
    X = np.array([r[1] for r in rows], dtype=np.float32)
    pn = net_predict(X)
    pp = policy_predict(X)
    bad = 0
    for (name, x, intended), a, b in zip(rows, pn, pp):
        flag = "" if (a == b == intended) else "  <-- " + ("net!=policy" if a != b else "policy!=intended")
        bad += a != b
        print(f"  net={CLASS[a]:11s} policy={CLASS[b]:11s} intended={CLASS[intended]:11s} | {name}{flag}")
        if flag:
            print("      " + describe(x))
    print(f"  scenarios: {len(rows)}, net/policy disagreements: {bad}")
    return bad


# -------------------------------------------------------- enumerations -----
VOCABS = {
    "A stall/retry mix": [read, rep_read, ext_err_rep, run_fail(1, 1, 0.0), edit_rev],
    "B recovery/resolution": [read, ext_err, ext_ok_rep, run_pass(1, 0.05), poll_neu_rep],
    "C progress/oscillation": [edit, edit_rev, run_fail(1, 0, -0.25), run_fail(1, 1, 0.25), build_fail_rep],
    "D neutral/control/mutate": [plan_rep, search_neu, edit_rep, build_ok_rep, ev("MUT", "ERR", ri=1, rf=1, wc=0)],
}


def run_enumerations() -> int:
    print("== 2. exhaustive enumerations over per-slot vocabularies (5^8 windows each) ==")
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


# ------------------------------------------------------- random sweep ------
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
    print(f"== 3. random valid-window sweep: n={n}, seed={seed} ==")
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


# --------------------------------------------------------- ramp probe ------
def run_ramp_probe() -> None:
    print("== 4. progress-ramp probe (net validation delta near 0.0175) ==")
    base = [edit, run_fail(0, 0, 0.0), edit, run_fail(1, 0, 0.0), edit_rev, run_fail(1, 1, 0.0), edit_rev, run_fail(1, 0, 0.0)]
    for ctx_name, ctx in (("oscillation context", base), ("stall context", [run_fail(0, 0, 0.0)] + [run_fail(1, 1, 0.0)] * 7)):
        mism = []
        grid = np.arange(0.0170, 0.0180, 0.00001)
        for sval in grid:
            x = window(ctx)
            x[15 * 7 + 13] = float(np.float32(sval))  # put the whole delta on the endpoint run
            X = np.array([x], dtype=np.float32)
            a = int(net_predict(X)[0])
            b = int(policy_predict(X)[0])
            if a != b:
                mism.append((float(np.float32(sval)), CLASS[a], CLASS[b]))
        print(f"  {ctx_name}: {len(grid)} points, mismatches {len(mism)}" + (f" in [{mism[0][0]:.6f}, {mism[-1][0]:.6f}]" if mism else ""))
        for m in mism[:4]:
            print(f"    S={m[0]:.6f} net={m[1]} policy={m[2]}")


# ----------------------------------------------------- batch consistency ---
def run_batch_consistency(n: int, seed: int) -> int:
    print("== 5. batch-size consistency (batch 1 vs batched) ==")
    X = random_valid_windows(n, seed)
    batched = net_predict(X)
    with torch.no_grad():
        single = np.array([int(torch.argmax(NET(torch.from_numpy(X[i : i + 1])), dim=-1)) for i in range(n)])
    bad = int((batched != single).sum())
    print(f"  {n} windows, differences {bad}")
    return bad


if __name__ == "__main__":
    b1 = run_scenarios()
    b2 = run_enumerations()
    b3 = run_random(300000, 20261104)
    run_ramp_probe()
    b4 = run_batch_consistency(5000, 7)
    print(f"== total net/policy disagreements: scenarios {b1}, enumerations {b2}, random {b3}, batch {b4} ==")
