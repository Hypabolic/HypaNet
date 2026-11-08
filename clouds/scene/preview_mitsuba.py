"""Quick-look Mitsuba render of the scene (quality check for scene/out; NOT the reference renderer).

    uv run python scene/preview_mitsuba.py front_noon side_sunset --scale 0.5 --spp 64 [--out DIR]

Pair ids are <shot>_<preset> (any combination, not only the listed pairs). Writes <id>.png (tone-mapped with the
pair's exposure from shots.json, or --exposure) and <id>.npy (linear RGB) to --out (default scene/out/previews).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tonemap import save_png  # noqa: E402

import build_scene  # noqa: E402

OUT = HERE / "out"


def load_scene_dict(mi, shot, preset, scale=1.0, spp=64, max_depth=-1, rr_depth=64, sun=True, sky=True, zoom=1.0):
    meta = json.loads((OUT / "density.json").read_text())
    shots = json.loads((OUT / "shots.json").read_text())
    nx, ny, nz = meta["dims"]
    d = np.frombuffer((OUT / "density.u8").read_bytes(), np.uint8).reshape(nz, ny, nx).astype(np.float32) / 255.0
    bmin = np.array(meta["box_min"]); bmax = np.array(meta["box_max"])
    sz = bmax - bmin
    P = shots["presets"][preset]
    S = shots["shots"][shot]
    T = mi.ScalarTransform4f
    grid = mi.VolumeGrid(d[..., None])
    medium = {
        "type": "heterogeneous",
        "albedo": meta["albedo"],
        "scale": meta["sigma_t_max"],
        "sigma_t": {"type": "gridvolume", "grid": grid, "filter_type": "trilinear",
                    "to_world": T().translate(bmin.tolist()).scale(sz.tolist())},
        "phase": {"type": "blendphase", "weight": 0.2,
                  "phase1": {"type": "hg", "g": 0.9}, "phase2": {"type": "hg", "g": -0.3}},
    }
    scene = {
        "type": "scene",
        "integrator": {"type": "volpath", "max_depth": max_depth, "rr_depth": rr_depth},
        "cloud": {"type": "cube", "bsdf": {"type": "null"}, "interior": medium,
                  "to_world": T().translate(((bmin + bmax) / 2).tolist()).scale((sz / 2).tolist())},
        "sensor": {
            "type": "perspective", "far_clip": 1.0e6, "fov": float(np.degrees(2 * np.arctan(np.tan(np.radians(S["fov_y_deg"] / 2)) / zoom))), "fov_axis": "y",
            "to_world": T().look_at(origin=S["position"], target=S["look_at"], up=S["up"]),
            "sampler": {"type": "independent", "sample_count": spp},
            "film": {"type": "hdrfilm", "width": int(S["width"] * scale), "height": int(S["height"] * scale),
                     "rfilter": {"type": "box"}, "pixel_format": "rgb"},
        },
    }
    if sun:
        scene["sun"] = {"type": "directional", "direction": (-np.array(P["toward_sun"])).tolist(),
                        "irradiance": {"type": "rgb", "value": P["E_sun"]}}
    if sky:
        side = json.loads((OUT / P["sky_json"]).read_text())
        img = np.frombuffer((OUT / P["sky_map"]).read_bytes(), "<f4").reshape(side["height"], side["width"], 3)
        scene["sky"] = {"type": "envmap", "bitmap": mi.Bitmap(img.astype(np.float32).copy()), "scale": 1.0}
    return scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--exposure", type=float, default=None)
    ap.add_argument("--out", default=str(OUT / "previews"))
    ap.add_argument("--variant", default="metal_ad_rgb")
    ap.add_argument("--zoom", type=float, default=1.0, help="inspection only: narrow the field of view by this factor")
    args = ap.parse_args()
    import mitsuba as mi

    mi.set_variant(args.variant)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    shots = json.loads((OUT / "shots.json").read_text())
    for pid in args.ids:
        shot, preset = pid.split("_", 1)
        t = time.time()
        scene = mi.load_dict(load_scene_dict(mi, shot, preset, args.scale, args.spp, zoom=args.zoom))
        img = np.array(mi.render(scene, spp=args.spp))[..., :3]
        ex = args.exposure if args.exposure is not None else build_scene.EXPOSURE[(shot, preset)]
        np.save(outdir / f"{pid}.npy", img.astype(np.float32))
        save_png(outdir / f"{pid}.png", img, ex)
        lum = img @ np.array([0.2126, 0.7152, 0.0722])
        print(f"{pid}: {time.time() - t:.1f}s  p50 lum {np.percentile(lum, 50):.3f}  p99 {np.percentile(lum, 99):.3f}  "
              f"max {lum.max():.3f}  exposure {ex}")


if __name__ == "__main__":
    main()
