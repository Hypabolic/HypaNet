"""Direct numpy evaluation of the SPEC exact sun single scattering for a pixel block (diagnostics for 6.1).

    uv run python validate/numpy_single.py backlit_golden --row 224 --col 432 [--size 8] [--sub 2] [--dt 1.0]

Prints the block mean of the sun part for: the continuous integral (fine steps, no density threshold) and the same
with the renderer's d > 0.002 sample threshold, next to the renderer's and Mitsuba's block means.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy import ndimage

from common import CLOUDS, OUT, REF_OUT, lum, shots

sys.path.insert(0, str(CLOUDS / "reference"))
import render_ref as rr  # noqa: E402


def phase(mu):
    hg = lambda g: (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * mu) ** 1.5)  # noqa: E731
    return 0.8 * hg(0.9) + 0.2 * hg(-0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair")
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--col", type=int, required=True)
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--sub", type=int, default=2)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--ds", type=float, default=2.0)
    a = ap.parse_args()
    meta, sh = rr.load_meta()
    d = rr.load_density(meta).astype(np.float64)
    bmin = np.array(meta["box_min"]); bmax = np.array(meta["box_max"]); vox = np.array(meta["voxel_size"])
    sig = meta["sigma_t_max"]; om = meta["albedo"]
    shot, preset = a.pair.split("_", 1)
    S = sh["shots"][shot]; P = sh["presets"][preset]
    s = np.array(P["toward_sun"]); E = np.array(P["E_sun"])
    W, H = S["width"], S["height"]

    def dens(p):
        c = (p - bmin) / vox - 0.5
        inside = np.all((p >= bmin) & (p <= bmax), axis=-1)
        v = ndimage.map_coordinates(d, [c[:, 2], c[:, 1], c[:, 0]], order=1, mode="nearest")
        return np.where(inside, v, 0.0)

    def box_t(o, v):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (bmin - o) / v; t2 = (bmax - o) / v
        return max(np.nanmax(np.minimum(t1, t2)), 0.0), np.nanmin(np.maximum(t1, t2))

    n = a.sub
    V = rr.camera_rays(S, W * n, H * n)
    o = np.array(S["position"], float)
    res = {0.0: [], 0.002: []}
    for j in range(a.row * n, (a.row + a.size) * n):
        for i in range(a.col * n, (a.col + a.size) * n):
            v = V[j, i]
            tn, tf = box_t(o, v)
            if tf <= tn:
                for k in res: res[k].append(0.0)
                continue
            t = np.arange(tn + a.dt / 2, tf, a.dt)
            p = o + t[:, None] * v
            dd = dens(p)
            st = sig * dd
            for thr in res:
                use = dd > thr
                stu = np.where(use, st, 0.0)
                Tc = np.exp(-np.concatenate([[0], np.cumsum(stu * a.dt)[:-1]]))
                keep = use & (Tc > 1e-6)
                if not keep.any():
                    res[thr].append(0.0); continue
                pk = p[keep]
                tau = np.zeros(len(pk))
                for k in range(int(np.linalg.norm(bmax - bmin) / a.ds) + 1):
                    q = pk + (k + 0.5) * a.ds * s
                    dq = dens(q)
                    tau += sig * dq * a.ds
                w = Tc[keep] * om * np.exp(-tau) * (1 - np.exp(-stu[keep] * a.dt))
                res[thr].append(float(lum((w.sum() * E * phase(v @ s))[None])[0]))
    r = np.load(OUT / "exact_nobg" / f"{a.pair}_exact_single.npy")[a.row:a.row + a.size, a.col:a.col + a.size]
    m = np.load(REF_OUT / "single" / f"{a.pair}_sun.npy")[a.row:a.row + a.size, a.col:a.col + a.size]
    print(f"{a.pair} block ({a.row},{a.col}) {a.size}x{a.size}, {n}x{n} sub-pixels, dt {a.dt} m, shadow ds {a.ds} m")
    print(f"  numpy continuous (no threshold): {np.mean(res[0.0]):.5f}")
    print(f"  numpy with d > 0.002 threshold : {np.mean(res[0.002]):.5f}")
    print(f"  renderer exact-single          : {lum(r).mean():.5f}")
    print(f"  Mitsuba single (sun part)      : {lum(m).mean():.5f}")


if __name__ == "__main__":
    main()
