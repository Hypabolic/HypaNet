"""Suggest a per-(shot, preset) exposure from quick low-res Mitsuba renders (harness helper, run by hand).

    uv run python scene/choose_exposure.py [--scale 0.25] [--spp 512]

For every shot x preset it renders the full scene (sun + sky, all bounces), finds the cloud pixels (optical depth
along the pixel ray > 1, from the density grid) and picks the exposure that puts the cloud's 90th-percentile
luminance at ACES input 1.0 (display value ~0.80 after the tone map: bright but below the shoulder), then rounds
it down to a 1/3-stop step. Paste the printed table into EXPOSURE in build_scene.py.
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
import build_scene  # noqa: E402
from preview_mitsuba import load_scene_dict  # noqa: E402

OUT = HERE / "out"
TARGET = 1.0  # exposure * p90(cloud luminance)
LUM = np.array([0.2126, 0.7152, 0.0722])


def cloud_mask(S, W, H, meta, d):
    """Optical depth through the density grid along each pixel's centre ray."""
    bmin = np.array(meta["box_min"]); bmax = np.array(meta["box_max"])
    nz, ny, nx = d.shape
    hd = (bmax - bmin) / np.array([nx, ny, nz])
    pos = np.array(S["position"]); tgt = np.array(S["look_at"])
    fwd = (tgt - pos) / np.linalg.norm(tgt - pos)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    th = np.tan(np.radians(S["fov_y_deg"] / 2))
    jj, ii = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    px = (2 * (ii + 0.5) / W - 1) * th * W / H
    py = (1 - 2 * (jj + 0.5) / H) * th
    v = fwd + px[..., None] * right + py[..., None] * up
    v = (v / np.linalg.norm(v, axis=-1, keepdims=True)).reshape(-1, 3)
    inv = 1.0 / np.where(np.abs(v) < 1e-9, 1e-9, v)
    t1 = (bmin - pos) * inv; t2 = (bmax - pos) * inv
    tn = np.maximum(np.max(np.minimum(t1, t2), axis=1), 0.0); tf = np.min(np.maximum(t1, t2), axis=1)
    tau = np.zeros(len(v))
    step = 6.0
    hit = tf > tn
    nsteps = int(np.ceil((tf[hit] - tn[hit]).max() / step)) if hit.any() else 0
    for k in range(nsteps):
        t = tn + (k + 0.5) * step
        ok = hit & (t < tf)
        q = pos + t[ok, None] * v[ok]
        c = (q - bmin) / hd - 0.5
        tau[ok] += ndimage.map_coordinates(d, [c[:, 2], c[:, 1], c[:, 0]], order=1, mode="constant") * step
    return (tau * meta["sigma_t_max"]).reshape(H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--spp", type=int, default=512)
    args = ap.parse_args()
    import mitsuba as mi

    mi.set_variant("metal_ad_rgb")
    meta = json.loads((OUT / "density.json").read_text())
    nx, ny, nz = meta["dims"]
    d = np.frombuffer((OUT / "density.u8").read_bytes(), np.uint8).reshape(nz, ny, nx).astype(np.float32) / 255.0
    shots = json.loads((OUT / "shots.json").read_text())
    table = {}
    for shot, S in shots["shots"].items():
        W, H = int(S["width"] * args.scale), int(S["height"] * args.scale)
        tau = cloud_mask(S, W, H, meta, d)
        m = tau > 1.0
        for preset in shots["presets"]:
            t = time.time()
            img = np.array(mi.render(mi.load_dict(load_scene_dict(mi, shot, preset, args.scale, args.spp)), spp=args.spp))[..., :3]
            lum = img @ LUM
            p50, p90, p99 = np.percentile(lum[m], [50, 90, 99])
            raw = TARGET / p90
            ex = float(2.0 ** (np.floor(3 * np.log2(raw)) / 3))
            table[(shot, preset)] = round(ex, 4)
            print(f"{shot:8s} {preset:9s} cloud px {m.mean():.2f}  lum p50 {p50:7.3f} p90 {p90:7.3f} p99 {p99:7.3f}  "
                  f"-> exposure {ex:.4f}  ({time.time() - t:.1f}s)", flush=True)
    print("\nEXPOSURE = {")
    for (shot, preset), ex in table.items():
        print(f"    ({shot!r}, {preset!r}): {ex},")
    print("}")


if __name__ == "__main__":
    main()
