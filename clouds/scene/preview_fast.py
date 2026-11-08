"""Fast, noise-free quick-look of the cloud shape (numpy/scipy raymarch). NOT physically exact, NOT the reference.

Sun light: per-voxel optical depth toward the sun (half-resolution grid), shaded with a Wrenninge-style
multiple-scattering octave sum (a^i * exp(-b^i tau) * p(c^i mu)); ambient: L_sky scaled by a crude "openness"
(exp of the optical depth straight up, blurred). Good enough to judge the silhouette, billows, crevice shadows,
the flat base and the wispy edges without path-tracing noise.

    uv run python scene/preview_fast.py side_noon front_noon [--scale 1.0] [--zoom 1.6] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tonemap import save_png  # noqa: E402

OUT = HERE / "out"


def hg(mu, g):
    return (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * mu) ** 1.5)


def phase(mu, k=1.0):
    return 0.8 * hg(mu, 0.9 * k) + 0.2 * hg(mu, -0.3 * k)


def load():
    meta = json.loads((OUT / "density.json").read_text())
    nx, ny, nz = meta["dims"]
    d = np.frombuffer((OUT / "density.u8").read_bytes(), np.uint8).reshape(nz, ny, nx).astype(np.float32) / 255.0
    return meta, d


def tau_toward(d, meta, s, factor=2, step=8.0):
    """Optical depth from each (downsampled) voxel centre toward direction s; returns (grid, voxel size)."""
    bmin = np.array(meta["box_min"]); bmax = np.array(meta["box_max"])
    dd = d.reshape(d.shape[0] // factor, factor, d.shape[1] // factor, factor, d.shape[2] // factor, factor).mean(axis=(1, 3, 5))
    nz, ny, nx = dd.shape
    h = (bmax - bmin) / np.array([nx, ny, nz])
    zi, yi, xi = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    p = np.stack([bmin[0] + (xi + 0.5) * h[0], bmin[1] + (yi + 0.5) * h[1], bmin[2] + (zi + 0.5) * h[2]], -1).reshape(-1, 3)
    tau = np.zeros(len(p))
    L = np.linalg.norm(bmax - bmin)
    for t in np.arange(0.5 * step, L, step):
        q = p + t * s
        c = (q - bmin) / h - 0.5
        v = ndimage.map_coordinates(dd, [c[:, 2], c[:, 1], c[:, 0]], order=1, mode="constant", cval=0.0)
        tau += v * step
    return (tau * meta["sigma_t_max"]).reshape(nz, ny, nx).astype(np.float32), h


def render(pid, scale=1.0, zoom=1.0, step=4.0):
    meta, d = load()
    shots = json.loads((OUT / "shots.json").read_text())
    shot, preset = pid.split("_", 1)
    S = shots["shots"][shot]; P = shots["presets"][preset]
    s = np.array(P["toward_sun"]); E = np.array(P["E_sun"]); Lsky = np.array(P["L_sky"]); Lg = np.array(P["L_gnd"])
    bmin = np.array(meta["box_min"]); bmax = np.array(meta["box_max"])
    sig = meta["sigma_t_max"]
    t0 = time.time()
    tsun, hs = tau_toward(d, meta, s)
    tup, _ = tau_toward(d, meta, np.array([0.0, 1.0, 0.0]))
    tup = ndimage.uniform_filter(tup, 5)
    tdn, _ = tau_toward(d, meta, np.array([0.0, -1.0, 0.0]))
    tdn = ndimage.uniform_filter(tdn, 5)
    W, H = int(S["width"] * scale), int(S["height"] * scale)
    pos = np.array(S["position"]); tgt = np.array(S["look_at"])
    fwd = (tgt - pos) / np.linalg.norm(tgt - pos)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    th = np.tan(np.radians(S["fov_y_deg"] / 2)) / zoom
    jj, ii = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    px = (2 * (ii + 0.5) / W - 1) * th * W / H
    py = (1 - 2 * (jj + 0.5) / H) * th
    v = fwd + px[..., None] * right + py[..., None] * up
    v = (v / np.linalg.norm(v, axis=-1, keepdims=True)).reshape(-1, 3)
    # sky background
    side = json.loads((OUT / P["sky_json"]).read_text())
    sky = np.frombuffer((OUT / P["sky_map"]).read_bytes(), "<f4").reshape(side["height"], side["width"], 3)
    theta = np.arccos(np.clip(v[:, 1], -1, 1)); phi = np.mod(np.arctan2(v[:, 0], -v[:, 2]), 2 * np.pi)
    bg = sky[np.clip((theta / np.pi * side["height"]).astype(int), 0, side["height"] - 1),
             np.clip((phi / (2 * np.pi) * side["width"]).astype(int), 0, side["width"] - 1)]
    # box intersection
    inv = 1.0 / np.where(np.abs(v) < 1e-9, 1e-9, v)
    t1 = (bmin - pos) * inv; t2 = (bmax - pos) * inv
    tn = np.maximum(np.max(np.minimum(t1, t2), axis=1), 0.0); tf = np.min(np.maximum(t1, t2), axis=1)
    hit = np.where(tf > tn)[0]
    Lacc = np.zeros((len(v), 3)); T = np.ones(len(v))
    nzd, nyd, nxd = d.shape
    hd = (bmax - bmin) / np.array([nxd, nyd, nzd])
    mu = v @ s
    # multiple-scattering octaves (Wrenninge 2015): a = 0.5^i, b = 0.5^i, c = 0.5^i
    oct_ = [(0.5 ** i, 0.5 ** i, 0.5 ** i) for i in range(6)]
    nsteps = int(np.ceil((tf[hit] - tn[hit]).max() / step)) if len(hit) else 0
    act = hit.copy()
    for k in range(nsteps):
        t = tn[act] + (k + 0.5) * step
        keep = (t < tf[act]) & (T[act] > 0.003)
        act = act[keep]; t = t[keep]
        if len(act) == 0:
            break
        q = pos + t[:, None] * v[act]
        c = (q - bmin) / hd - 0.5
        dv = ndimage.map_coordinates(d, [c[:, 2], c[:, 1], c[:, 0]], order=1, mode="constant")
        m = dv > 0.002
        if not m.any():
            continue
        a2 = act[m]; dv = dv[m]; q = q[m]
        cs = (q - bmin) / hs - 0.5
        coords = [cs[:, 2], cs[:, 1], cs[:, 0]]
        ts = ndimage.map_coordinates(tsun, coords, order=1, mode="nearest")
        tu = ndimage.map_coordinates(tup, coords, order=1, mode="nearest")
        tdw = ndimage.map_coordinates(tdn, coords, order=1, mode="nearest")
        sun = sum(a * np.exp(-b * ts) * phase(mu[a2], cc) for a, b, cc in oct_)
        amb = Lsky * (0.25 + 0.75 * np.exp(-0.25 * tu))[:, None] + Lg * (0.25 + 0.75 * np.exp(-0.25 * tdw))[:, None]
        Ssc = E * sun[:, None] + amb * 0.9
        st = sig * dv
        a_ = 1 - np.exp(-st * step)
        Lacc[a2] += (T[a2] * 0.999 * a_)[:, None] * Ssc
        T[a2] *= np.exp(-st * step)
    img = (Lacc + T[:, None] * bg).reshape(H, W, 3)
    print(f"{pid}: {time.time() - t0:.1f}s")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--exposure", type=float, default=None)
    ap.add_argument("--out", default=str(OUT / "previews"))
    args = ap.parse_args()
    shots = json.loads((OUT / "shots.json").read_text())
    exp = {p["id"]: p["exposure"] for g in shots["pairs"].values() for p in g}
    exp.update(shots.get("exposure_table", {}))
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    for pid in args.ids:
        img = render(pid, args.scale, args.zoom)
        ex = args.exposure if args.exposure is not None else exp.get(pid, 0.25)
        save_png(outdir / f"{pid}_fast.png", img, ex)


if __name__ == "__main__":
    main()
