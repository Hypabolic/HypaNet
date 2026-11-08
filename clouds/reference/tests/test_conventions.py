"""Convention checks for the Mitsuba reference renderer (run before trusting reference/out).

    uv run python reference/tests/test_conventions.py [--variant metal_ad_rgb]

A. Phase direction, directional-sun irradiance and the max_depth semantics: a thin homogeneous slab (1 m thick,
   tau = 0.01) lit by the sun, seen from below at several view directions v. Single scattering has a closed form:
       L = omega sigma_t E p(dot(v, s)) * int_0^h exp(-sigma_t (h - y)/s_y) exp(-sigma_t y / v_y) dy / v_y
   with p = 0.8 HG(0.9) + 0.2 HG(-0.3) in the SPEC (propagation) convention. Mitsuba volpath max_depth 2 must
   match (forward lobe toward the sun when looking at the sun), max_depth 1 must be 0.
B. Envmap orientation + camera: an analytic direction-coded map f(d) = (1.5 + d.x, 1.5 + d.y, 1.5 + d.z) seen
   through each shot's camera must give f(v) at every pixel (detects any mirroring / axis swap / fov error).
C. Scale: the real sky maps rendered as background must match a bilinear lookup of the raw map (SPEC section 6.2
   pre-check; reports both the pixel-centred and Mitsuba's vertex-centred lookup).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference"))
import render_ref as rr  # noqa: E402


def hg(mu, g):
    return (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * mu) ** 1.5)


def phase(mu):
    return 0.8 * hg(mu, 0.9) + 0.2 * hg(mu, -0.3)


def slab_test(mi):
    T = mi.ScalarTransform4f
    elev = np.radians(60.0)
    s = np.array([np.cos(elev), np.sin(elev), 0.0])  # toward sun, azimuth +x
    E = np.array([3.0, 2.0, 1.0])
    sig, h, om = 1e-4, 100.0, 0.999
    res = []
    for ang_deg, az_deg in [(60, 0), (75, 0), (90, 0), (60, 90), (40, 180), (30, 180), (10, 180), (85, 45)]:
        el, az = np.radians(ang_deg), np.radians(az_deg)
        v = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
        mu = float(v @ s)
        y = np.linspace(0, h, 20001)
        integrand = np.exp(-sig * (h - y) / s[1]) * np.exp(-sig * y / v[1])
        integral = np.trapezoid(integrand, y) / v[1]
        L_exact = om * sig * E * phase(mu) * integral
        out = {}
        for md in (1, 2):
            scene = {
                "type": "scene",
                "integrator": {"type": "volpath", "max_depth": md},
                "slab": {"type": "cube", "bsdf": {"type": "null"},
                         "interior": {"type": "homogeneous", "albedo": om, "sigma_t": sig,
                                      "phase": {"type": "blendphase", "weight": 0.2,
                                                "phase1": {"type": "hg", "g": 0.9},
                                                "phase2": {"type": "hg", "g": -0.3}}},
                         "to_world": T().translate([0, h / 2, 0]).scale([2e4, h / 2, 2e4])},
                "sun": {"type": "directional", "direction": (-s).tolist(),
                        "irradiance": {"type": "rgb", "value": E.tolist()}},
                "sensor": {"type": "perspective", "fov": 0.5, "fov_axis": "y", "far_clip": 1e7,
                           "to_world": T().look_at(origin=[0, -10, 0], target=(np.array([0, -10, 0]) + v).tolist(),
                                                   up=[0, 0, 1] if abs(v[1]) > 0.99 else [0, 1, 0]),
                           "sampler": {"type": "independent", "sample_count": 256},
                           "film": {"type": "hdrfilm", "width": 64, "height": 64, "rfilter": {"type": "box"},
                                    "pixel_format": "rgb"}},
            }
            sc = mi.load_dict(scene)
            # many passes of modest spp: Metal float atomics make thousands of samples per pixel in one launch slow
            img = np.mean([np.array(mi.render(sc, seed=k, spp=256))[..., :3] for k in range(16)], 0)
            out[md] = img.reshape(-1, 3).mean(0)
        rel = out[2] / L_exact - 1
        # what the result would be with the opposite (mirrored) lobe convention
        rel_mirror = om * sig * E * phase(-mu) * integral / L_exact - 1
        res.append((ang_deg, az_deg, mu, L_exact, out[2], rel, out[1], rel_mirror))
        print(f"  view el {ang_deg:3d} az {az_deg:3d}  mu={mu:+.3f}  exact {L_exact.round(6)}  mitsuba {out[2].round(6)}"
              f"  rel {np.round(rel, 4)}  (depth1 {out[1].round(8)}; mirrored-convention would be {rel_mirror[0]:+.2f})")
    worst = max(np.abs(r[5]).max() for r in res)
    d1 = max(np.abs(r[6]).max() for r in res)
    print(f"A. slab single scattering: worst |rel err| {worst:.4f} (want < 0.01); max depth-1 radiance {d1:.2e} (want 0)")
    return worst < 0.01 and d1 == 0


def coded_map(H=512, W=1024):
    col = np.arange(W) + 0.5
    row = np.arange(H) + 0.5
    phi = 2 * np.pi * col / W
    th = np.pi * row / H
    TH, PH = np.meshgrid(th, phi, indexing="ij")
    d = np.stack([np.sin(TH) * np.sin(PH), np.cos(TH), -np.sin(TH) * np.cos(PH)], -1)
    return (1.5 + d).astype(np.float32)


def env_tests(mi):
    shots = json.loads((rr.SCENE_OUT / "shots.json").read_text())
    ok = True
    # B: orientation with the coded map
    m = coded_map()
    for shot in shots["shots"]:
        S = shots["shots"][shot]
        scene = rr.build_scene_dict(mi, shot, None, spp=64, scale=0.25, medium=False, sun=False,
                                    sky_bitmap=m, max_depth=1)
        img = np.array(mi.render(mi.load_dict(scene), spp=64))[..., :3]
        v = rr.camera_rays(S, img.shape[1], img.shape[0])
        err = np.abs(img - (1.5 + v)).max()
        print(f"B. coded envmap, shot {shot:8s}: max |mitsuba - f(v)| = {err:.4f} (want < 0.01)")
        ok &= err < 0.01
    # C: real skies, scale and sub-pixel registration
    for preset in shots["presets"]:
        for shot in ("front", "backlit", "distant"):
            S = shots["shots"][shot]
            scene = rr.build_scene_dict(mi, shot, preset, spp=256, scale=0.25, medium=False, sun=False, max_depth=1)
            img = np.array(mi.render(mi.load_dict(scene), spp=256))[..., :3]
            sky = rr.load_sky(shots["presets"][preset])
            v = rr.camera_rays(S, img.shape[1], img.shape[0])
            a = rr.sky_lookup(sky, v, "pixel")
            b = rr.sky_lookup(sky, v, "vertex")
            mask = v[..., 1] > 0.02  # away from the horizon step (sub-pixel footprint averaging there)
            ea = np.abs(img / a - 1)[mask].mean()
            eb = np.abs(img / b - 1)[mask].mean()
            print(f"C. sky {preset:9s} {shot:8s}: mean |rel err| vs pixel-centred lookup {ea:.4f}, vs vertex-centred "
                  f"{eb:.4f}; ratio of means {img[mask].mean(0) / a[mask].mean(0)}")
            ok &= ea < 0.01
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="metal_ad_rgb")
    ap.add_argument("--skip", default="")
    args = ap.parse_args()
    import mitsuba as mi
    mi.set_variant(args.variant)
    ok = True
    if "A" not in args.skip:
        ok &= slab_test(mi)
    if "B" not in args.skip:
        ok &= env_tests(mi)
    print("ALL OK" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
