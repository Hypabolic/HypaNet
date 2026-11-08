"""Independent check of the Mitsuba single-sun mode against a direct numpy evaluation of the SPEC formula

    L = sum T * omega * E_sun * T_sun * p(dot(v, s)) * (1 - exp(-sigma_t dt))   (+ T * sky(v) background)

on a sparse grid of pixels (each pixel = mean of 2 x 2 sub-pixel rays, close to Mitsuba's box filter). Fine steps
(1.5 m along the view ray, 3 m toward the sun), cell-centred trilinear density with zero outside the box.

    uv run python reference/tests/test_single_numpy.py front_noon [--dir reference/out/single] [--every 16]

Needs <pair>_sun.npy / <pair>_bg.npy from `render_ref.py --mode single`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference"))
import render_ref as rr  # noqa: E402


def phase(mu):
    hg = lambda g: (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * mu) ** 1.5)  # noqa: E731
    return 0.8 * hg(0.9) + 0.2 * hg(-0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--dir", default=str(ROOT / "reference" / "out" / "single"))
    ap.add_argument("--every", type=int, default=16)
    args = ap.parse_args()
    meta, shots = rr.load_meta()
    d = rr.load_density(meta)
    bmin = np.array(meta["box_min"]); bmax = np.array(meta["box_max"]); vox = np.array(meta["voxel_size"])
    sig = meta["sigma_t_max"]; om = meta["albedo"]

    def dens(p):
        c = (p - bmin) / vox - 0.5
        inside = np.all((p >= bmin) & (p <= bmax), axis=-1)
        v = ndimage.map_coordinates(d, [c[:, 2], c[:, 1], c[:, 0]], order=1, mode="nearest")
        return np.where(inside, v, 0.0)

    def box_t(o, v):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (bmin - o) / v; t2 = (bmax - o) / v
        tn = np.nanmax(np.minimum(t1, t2), -1); tf = np.nanmin(np.maximum(t1, t2), -1)
        return max(tn, 0.0), tf

    for pid in args.pairs:
        shot, preset = pid.split("_", 1)
        S = shots["shots"][shot]; P = shots["presets"][preset]
        s = np.array(P["toward_sun"]); E = np.array(P["E_sun"])
        sun = np.load(Path(args.dir) / f"{pid}_sun.npy"); bg = np.load(Path(args.dir) / f"{pid}_bg.npy")
        H, W, _ = sun.shape
        sub = []
        for oy in (0.25, 0.75):
            for ox in (0.25, 0.75):
                SS = dict(S)
                v = rr.camera_rays(SS, W * 2, H * 2)[int(oy * 2)::2, int(ox * 2)::2]
                sub.append(v)
        o = np.array(S["position"], float)
        rows = range(args.every // 2, H, args.every); cols = range(args.every // 2, W, args.every)
        ref_sun, ref_T, mit_sun, mit_bg = [], [], [], []
        for j in rows:
            for i in cols:
                acc = np.zeros(3); accT = 0.0
                for v in (x[j, i] for x in sub):
                    tn, tf = box_t(o, v)
                    if tf <= tn:
                        accT += 1.0; continue
                    dt = 1.5
                    t = np.arange(tn + dt / 2, tf, dt)
                    if len(t) == 0:
                        accT += 1.0; continue
                    p = o + t[:, None] * v
                    dd = dens(p)
                    st = sig * dd
                    Tcam = np.exp(-np.concatenate([[0], np.cumsum(st * dt)[:-1]]))
                    keep = (dd > 0) & (Tcam > 1e-5)
                    L = np.zeros(3)
                    if keep.any():
                        pk = p[keep]
                        # shadow ray to the sun, up to the box exit
                        tau = np.zeros(len(pk))
                        ds = 3.0
                        for k in range(int(np.linalg.norm(bmax - bmin) / ds) + 1):
                            q = pk + (k + 0.5) * ds * s
                            tau += sig * dens(q) * ds
                        w = Tcam[keep] * om * np.exp(-tau) * (1 - np.exp(-st[keep] * dt))
                        L = w.sum() * E * phase(v @ s)
                    acc += L; accT += Tcam[-1] * np.exp(-st[-1] * dt)
                ref_sun.append(acc / 4); ref_T.append(accT / 4)
                mit_sun.append(sun[j, i]); mit_bg.append(bg[j, i])
        ref_sun = np.array(ref_sun); mit_sun = np.array(mit_sun)
        lum = ref_sun @ rr.LUM
        m = lum > np.percentile(lum[lum > 0], 10) if (lum > 0).any() else lum > 0
        rel = (mit_sun @ rr.LUM)[m] / lum[m] - 1
        print(f"{pid}: {m.sum()} cloud pixels; Mitsuba/numpy sun single-scattering: ratio of sums "
              f"{(mit_sun[m].sum(0) / ref_sun[m].sum(0)).round(4)}, per-pixel rel err median {np.median(rel):+.4f}, "
              f"mean |rel| {np.abs(rel).mean():.4f}")
        # background: Mitsuba bg / sky(v) is the camera-ray transmittance
        sky = rr.load_sky(P)
        vv = rr.camera_rays(S, W, H)
        skyv = np.array([rr.sky_lookup(sky, vv[j, i], "pixel") for j in rows for i in cols])
        Tm = (np.array(mit_bg) @ rr.LUM) / (skyv @ rr.LUM)
        ref_T = np.array(ref_T)
        print(f"   background: transmittance Mitsuba vs numpy: mean {Tm.mean():.4f} vs {ref_T.mean():.4f}; "
              f"max |diff| {np.abs(Tm - ref_T).max():.4f}; clear-sky pixels max |T-1| "
              f"{np.abs(Tm[ref_T > 0.9999] - 1).max() if (ref_T > 0.9999).any() else float('nan'):.4f}")


if __name__ == "__main__":
    main()
