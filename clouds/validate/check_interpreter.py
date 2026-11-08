"""SPEC 6.3: the MLP interpreter and probes.

    uv run python validate/make_test_net.py && uv run python validate/check_interpreter.py [--skip-render]

A. test_networks/exact_single_net.json (probes + activations only) rendered with render.py vs --mode exact-single,
   on the same 6 pairs as compare_single.py (validate/out/exact/ must exist: run compare_single.py --render first).
   The background T * sky(v) is identical in both (same march), so the difference is the a_sun part alone.
B. Activation identities: for every activation, test_networks/act_<name>.json is run through renderer/inspect.py
   (the author's tool, --json) on a few pixels; every recorded unit must equal act(W h + b) computed in float64
   numpy from the recorded probe values, with act written from the SPEC definitions here (independently of the
   renderer's own tests).
C. Scalar probes from the same records: height, view_up, cos_view_sun recomputed from the sample position and the
   ray direction; the raw world density probe vs a numpy trilinear lookup (cell-centred) of density.u8.
"""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np

from common import (CLOUDS, OUT, SCENE_OUT, VALIDATE, block, diverging, exposure, labelled, lum, render, row, shots,
                    stack, tonemap)

PAIRS = ["front_noon", "side_noon", "side_sunset", "backlit_golden", "below_noon", "distant_sunset"]
SPEC_ACTS = {  # SPEC section 4, written out independently
    "linear": lambda x: x,
    "relu": lambda x: np.maximum(x, 0.0),
    "exp": lambda x: np.exp(np.clip(x, -80, 80)),
    "log": lambda x: np.log(np.maximum(x, 1e-8)),
    "tanh": np.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "softplus": lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0),
    "abs": np.abs,
    "sin": np.sin,
    "square": lambda x: x * x,
    "sqrt": lambda x: np.sqrt(np.maximum(x, 0.0)),
    "recip": lambda x: 1.0 / np.maximum(x, 1e-6),
}
INSPECT_PIXELS = [("front_noon", 330, 170), ("backlit_golden", 300, 250), ("below_noon", 300, 150),
                  ("side_sunset", 300, 180)]
results = {"pass": [], "fail": []}


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    results["pass" if ok else "fail"].append(name)


def part_a(skip_render, stem="exact_single_net"):
    outdir = OUT / "net"
    net = VALIDATE / "test_networks" / f"{stem}.json"
    if not skip_render:
        render([str(net), "--frames", "64", "--pair", *PAIRS, "--out", str(outdir) + "/"])
    tiles, summary = [], []
    for pid in PAIRS:
        rn = np.load(outdir / f"{pid}_{stem}.npy").astype(np.float64)
        rx = np.load(OUT / "exact" / f"{pid}_exact_single.npy").astype(np.float64)
        rs = np.load(OUT / "exact_nobg" / f"{pid}_exact_single.npy").astype(np.float64)
        bg = rx - rs
        ns = rn - bg  # network's sun part
        cloud = lum(rs) > 0
        Yn, Ys = lum(ns), lum(rs)
        ratio = Yn[cloud].sum() / Ys[cloud].sum()
        l1 = np.abs(Yn - Ys)[cloud].sum() / Ys[cloud].sum()
        k = 4
        bn, bs = block(Yn, k), block(Ys, k)
        bm = block(cloud.astype(float), k) > 0.999
        den = np.maximum(bs[bm], 0.02 * bs[bm].mean())
        mrel4 = float(np.mean(np.abs(bn - bs)[bm] / den))
        tot_l1 = np.abs(lum(rn) - lum(rx))[cloud].sum() / lum(rx)[cloud].sum()
        o = {"pair": pid, "sun_ratio_of_sums": float(ratio), "sun_l1w": float(l1), "sun_meanrel_k4": mrel4,
             "total_l1w_cloud": float(tot_l1)}
        summary.append(o)
        check(f"A {stem} {pid}", l1 < 0.03, f"net/exact sun ratio {ratio:.4f}, sun L1w {l1:.4f}, sun mean rel (4x4) "
                                     f"{mrel4:.4f}, total-image L1w {tot_l1:.4f}")
        ex = exposure(pid)
        p99 = np.percentile(Ys[cloud], 99)
        rel = np.repeat(np.repeat((bn - bs) / np.maximum(bs, 1e-3 * bs.max()), k, 0), k, 1)
        tiles.append(row([labelled(tonemap(rn, ex), f"{pid}: {stem}.json (network)"),
                          labelled(tonemap(rx, ex), "--mode exact-single"),
                          labelled(tonemap(ns, 1 / p99), "network sun part (exposure 1/p99)"),
                          labelled(diverging(rel, cloud, 0.10), "sun part (net - exact)/exact, 4x4, +-10%")]))
    (OUT / "interpreter").mkdir(parents=True, exist_ok=True)
    stack(tiles).save(OUT / "interpreter" / f"{stem}_compare.png")
    return summary


def inspect_json(net, pid, px, py, path):
    cmd = ["uv", "run", "python", "renderer/inspect.py", str(net), "--pair", pid, "--pixel", str(px), str(py),
           "--samples", "all", "--json", str(path)]
    r = subprocess.run(cmd, cwd=CLOUDS, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)
    return json.loads(path.read_text())


def trilinear(d, p, bmin, vox):
    """Cell-centred trilinear (GL texture convention, clamp to edge) of d[z, y, x] at world points p (N, 3)."""
    c = (p - bmin) / vox - 0.5
    nz, ny, nx = d.shape
    c = np.clip(c, 0, [nx - 1, ny - 1, nz - 1])
    i0 = np.floor(c).astype(int)
    i0 = np.minimum(i0, [nx - 2, ny - 2, nz - 2])
    f = c - i0
    out = np.zeros(len(p))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out += w * d[i0[:, 2] + dz, i0[:, 1] + dy, i0[:, 0] + dx]
    return out


def part_bc():
    S = shots()
    meta = json.loads((SCENE_OUT / "density.json").read_text())
    nx, ny, nz = meta["dims"]
    dens = np.fromfile(SCENE_OUT / "density.u8", np.uint8).reshape(nz, ny, nx).astype(np.float64) / 255
    bmin = np.array(meta["box_min"]); bmax = np.array(meta["box_max"]); vox = np.array(meta["voxel_size"])
    tmp = OUT / "interpreter" / "inspect"
    tmp.mkdir(parents=True, exist_ok=True)
    probe_err = {"height": 0.0, "view_up": 0.0, "cos_view_sun": 0.0, "density": 0.0}
    nsamp = 0
    act_summary = {}
    comb_worst = 0.0
    for name in SPEC_ACTS:
        net_path = VALIDATE / "test_networks" / f"act_{name}.json"
        net = json.loads(net_path.read_text())
        W1 = np.array(net["layers"][0]["W"]); b1 = np.array(net["layers"][0]["b"])
        worst, worst_abs, n, skipped, bad_any = 0.0, 0.0, 0, 0, False
        for pid, px, py in INSPECT_PIXELS:
            js = inspect_json(net_path, pid, px, py, tmp / f"{name}_{pid}.json")
            names = js["names"]
            rec = np.array([[s[k] for k in names] for s in js["samples"]], np.float64)
            if not len(rec):
                continue
            pv = rec[:, [i for i, k in enumerate(names) if k.split(":")[0] in ("p0", "p1", "p2", "p3", "p4")]]
            units = rec[:, [i for i, k in enumerate(names) if k.startswith("L0.")]]
            outs = rec[:, [i for i, k in enumerate(names) if k.startswith("L1.")]]
            Srec = rec[:, [names.index("S.r"), names.index("S.g"), names.index("S.b")]]
            assert pv.shape[1] == 5 and units.shape[1] == 9 and outs.shape[1] == 9, names
            # identity output layer and the S combination (SPEC: S = a_sun E_sun + a_sky L_sky + a_gnd L_gnd)
            S_want = (outs[:, 0:3] * np.array(js["E_sun"]) + outs[:, 3:6] * np.array(js["L_sky"])
                      + outs[:, 6:9] * np.array(js["L_gnd"]))
            fin = np.isfinite(S_want).all(1) & (np.abs(S_want).max(1) < 1e30)
            s_err = np.abs(Srec - S_want)[fin] / np.maximum(np.abs(S_want[fin]).max(1, keepdims=True), 1e-6)
            out_err = np.abs(outs - units)[np.isfinite(units)] / np.maximum(np.abs(units[np.isfinite(units)]), 1e-6)
            comb_worst = max(comb_worst, float(s_err.max()) if s_err.size else 0.0, float(out_err.max()))
            pre = pv @ W1.T + b1
            want = SPEC_ACTS[name](pre)
            got = units
            ok = np.isfinite(want)
            if name == "sin":
                ok &= np.abs(pre) < 50
            # float32: relative 2e-5 of the value, plus 2e-6 absolute, plus the derivative times float32 input error
            # float32 evaluation: the pre-activation itself carries ~1e-7 relative error, which exp / recip / log
            # amplify; 1e-4 relative (+1e-6 absolute) is far below anything visible and far above float32 noise
            tol = 1e-4 * np.abs(want) + 1e-6
            err = np.abs(got - want)
            rel = err / np.maximum(np.abs(want), 1e-6)
            bad = (err > tol) & ok
            bad_any = bad_any or bool(bad.any())
            worst = max(worst, float(rel[ok].max()))
            worst_abs = max(worst_abs, float(err[ok].max()))
            n += int(ok.sum())
            skipped += int((~ok).sum())
            if bad.any():
                i, j = np.argwhere(bad)[0]
                print(f"    {name} {pid}: sample {i} unit {j}: pre {pre[i, j]:.6g} want {want[i, j]:.6g} got {got[i, j]:.6g}")
            # C: scalar probes (once, from the linear net's records)
            if name == "linear":
                sh = S["shots"][pid.split("_", 1)[0]]
                pr = S["presets"][pid.split("_", 1)[1]]
                sm = js["summary"]
                v = np.array([sm["dir.x"], sm["dir.y"], sm["dir.z"]])
                x = rec[:, [names.index("x"), names.index("y"), names.index("z")]]
                hgt = (x[:, 1] - bmin[1]) / (bmax[1] - bmin[1])
                probe_err["height"] = max(probe_err["height"], float(np.abs(pv[:, 0] - hgt).max()))
                probe_err["view_up"] = max(probe_err["view_up"], float(np.abs(pv[:, 1] - v[1]).max()))
                probe_err["cos_view_sun"] = max(probe_err["cos_view_sun"],
                                                float(np.abs(pv[:, 2] - v @ np.array(pr["toward_sun"])).max()))
                probe_err["density"] = max(probe_err["density"], float(np.abs(pv[:, 3] - trilinear(dens, x, bmin, vox)).max()))
                nsamp += len(rec)
                # also: the camera direction recorded by the shader vs the camera position -> sample
                dirs = (x - np.array(sh["position"])) / np.linalg.norm(x - np.array(sh["position"]), axis=1)[:, None]
                probe_err.setdefault("ray_dir", 0.0)
                probe_err["ray_dir"] = max(probe_err["ray_dir"], float(np.abs(dirs - v).max()))
        act_summary[name] = {"samples_x_units": n, "skipped": skipped, "max_rel": worst, "max_abs": worst_abs}
        check(f"B act {name}", n > 0 and not bad_any,
              f"{n} unit values, max rel err {worst:.2e}, max abs err {worst_abs:.2e}"
              + (f", {skipped} skipped" if skipped else ""))
    check("B output layer and S combination", comb_worst < 1e-4,
          f"max rel err {comb_worst:.1e} (identity layer; S = a_sun E_sun + a_sky L_sky + a_gnd L_gnd)")
    check("C scalar probes", probe_err["height"] < 1e-4 and probe_err["view_up"] < 1e-5
          and probe_err["cos_view_sun"] < 1e-5 and probe_err["ray_dir"] < 1e-4,
          f"{nsamp} samples: max |err| height {probe_err['height']:.1e}, view_up {probe_err['view_up']:.1e}, "
          f"cos_view_sun {probe_err['cos_view_sun']:.1e}, ray dir vs (x - cam) {probe_err['ray_dir']:.1e}")
    check("C raw density probe (world frame)", probe_err["density"] < 4e-3,
          f"max |GPU - numpy trilinear| {probe_err['density']:.2e} (8-bit texture filtering weights)")
    return act_summary, probe_err


def part_d():
    """D. Density probes: frames (light / view / world, 40 m away, raw) and blur levels (world, at the sample) from
    inspect records of test_networks/probe_net.json, against numpy."""
    import make_test_net as mtn
    S = shots()
    meta = json.loads((SCENE_OUT / "density.json").read_text())
    nx, ny, nz = meta["dims"]
    dens = np.fromfile(SCENE_OUT / "density.u8", np.uint8).reshape(nz, ny, nx).astype(np.float64) / 255
    bmin = np.array(meta["box_min"]); bmax = np.array(meta["box_max"]); vox = np.array(meta["voxel_size"])

    def field(p):
        inside = np.all((p >= bmin) & (p <= bmax), axis=-1)
        return np.where(inside, trilinear(dens, p, bmin, vox), 0.0)

    net_path = VALIDATE / "test_networks" / "probe_net.json"
    rng = np.random.default_rng(1)
    q = 12  # stratified q^3 points per box average
    g = (np.arange(q) + 0.5) / q - 0.5
    G3 = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    blur_err = {b: [] for b in mtn.BLURS}
    frame_err = {f"{f}{d}": [] for f, d in mtn.FRAME_PROBES}
    tmp = OUT / "interpreter" / "inspect"
    for pid, px, py in INSPECT_PIXELS:
        js = inspect_json(net_path, pid, px, py, tmp / f"probe_{pid}.json")
        names = js["names"]
        rec = np.array([[smp[k] for k in names] for smp in js["samples"]], np.float64)
        if not len(rec):
            continue
        pcols = [i for i, k in enumerate(names) if k.split(":")[0].startswith("p") and k.split(":")[0][1:].isdigit()]
        pv = rec[:, pcols]
        x = rec[:, [names.index("x"), names.index("y"), names.index("z")]]
        sm = js["summary"]
        v = np.array([sm["dir.x"], sm["dir.y"], sm["dir.z"]])
        shot, preset = pid.split("_", 1)
        s_ = np.array(S["presets"][preset]["toward_sun"], float)
        u = np.cross(s_, [0, 1, 0]); u /= np.linalg.norm(u); w = np.cross(u, s_)
        # renderer/NOTES.md 2: e2 = normalize(cross(v, camera up)), e3 = cross(e2, v), with the orthonormal camera up
        Sh = S["shots"][shot]
        fwd = np.array(Sh["look_at"], float) - np.array(Sh["position"], float); fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array(Sh["up"], float)); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        e2 = np.cross(v, up); e2 /= np.linalg.norm(e2); e3 = np.cross(e2, v)
        basis = {"light": (s_, u, w), "view": (v, e2, e3), "world": (np.eye(3)[0], np.eye(3)[1], np.eye(3)[2])}
        for j, b in enumerate(mtn.BLURS):
            if b == 0:
                want = field(x)
            else:
                pts = x[:, None, :] + (G3 + (rng.random(G3.shape) - 0.5) / q)[None] * b
                want = field(pts.reshape(-1, 3)).reshape(len(x), -1).mean(1)
            blur_err[b] += np.abs(pv[:, j] - want).tolist()
        for k, (f, d) in enumerate(mtn.FRAME_PROBES):
            dd = np.array(d, float); dd /= np.linalg.norm(dd)
            B = basis[f]
            dw = dd[0] * B[0] + dd[1] * B[1] + dd[2] * B[2]
            want = field(x + mtn.FRAME_DIST * dw[None])
            frame_err[f"{f}{d}"] += np.abs(pv[:, len(mtn.BLURS) + k] - want).tolist()
    fe = {k: float(np.max(e)) for k, e in frame_err.items()}
    check("D density probe frames (light/view/world, 40 m, raw)", max(fe.values()) < 4e-3,
          "max |GPU - numpy| " + ", ".join(f"{k} {e:.1e}" for k, e in fe.items()))
    be = {b: (float(np.mean(e)), float(np.max(e))) for b, e in blur_err.items()}
    # SPEC: blur is "about" a box-filter width; the renderer uses a box pyramid on voxel cells (raw trilinear up to
    # one voxel, 6.98 m) interpolated between power-of-two levels. Reference here: the mean of the trilinear field
    # over a cube of side `blur`. Pass: mean |err| < 0.015 and max < 0.07 at every level.
    check("D blurred density vs exact numpy cube average",
          max(m for m, _ in be.values()) < 0.015 and max(x for _, x in be.values()) < 0.07,
          "blur m: mean/max |err| " + ", ".join(f"{b:g}: {m:.3f}/{x:.3f}" for b, (m, x) in be.items()))
    return {"frames_max_abs": fe, "blur_mean_max_abs": {str(b): v for b, v in be.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--only", choices=["a", "bc", "d"])
    a = ap.parse_args()
    out = {}
    if a.only in (None, "a"):
        out["exact_single_net"] = part_a(a.skip_render)
        out["exact_single_net_noblur"] = part_a(a.skip_render, "exact_single_net_noblur")
    if a.only == "d":
        out["density_probes"] = part_d()
    elif a.only != "a":
        out["activations"], out["probes"] = part_bc()
        out["density_probes"] = part_d()
    (OUT / "interpreter").mkdir(parents=True, exist_ok=True)
    (OUT / "interpreter" / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"\n{len(results['pass'])} passed, {len(results['fail'])} failed: {results['fail']}")


if __name__ == "__main__":
    main()
