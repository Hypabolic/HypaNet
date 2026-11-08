"""How much does multiple scattering (and skylight) add? Full-mode Mitsuba reference vs exact sun single scattering.

    uv run python validate/multiple_scattering.py [--split-spp 1024] [pairs ...]

For each pair: the renderer's exact single scattering (validate/out/exact/, sun single scattering + T * sky) next
to the full-mode reference (reference/out/<pair>.npy, all orders, sun + sky) and the ratio full / single on the
cloud. With --split-spp > 0 it also renders, at half resolution, the full reference split into a sun-only part and
a sky-only part (Mitsuba, all orders), so the cloud's light can be divided into
  sun single scattering | sun multiple scattering | skylight (all orders, incl. ground).
Writes validate/out/multiple/<pair>_compare.png and summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from common import CLOUDS, OUT, REF_OUT, block, exposure, labelled, lum, row, stack, tonemap

sys.path.insert(0, str(CLOUDS / "reference"))
PAIRS = ["front_noon", "backlit_golden", "side_sunset"]


def ratio_img(r, mask):
    """log2 ratio -> colour: 0 (x1) black, 1 (x2) purple, 2 (x4) red, 3 (x8) orange, >= 4 (x16+) yellow-white."""
    t = np.clip(np.log2(np.maximum(r, 1e-6)) / 5.0, 0, 1)
    rgb = np.stack([np.clip(2.0 * t, 0, 1), np.clip(2.0 * t - 0.8, 0, 1), np.clip(np.sin(np.pi * t) * 0.9 + (t > 0.85) * (t - 0.85) * 5, 0, 1)], -1)
    rgb = np.where(mask[..., None], rgb, 0.12)
    return (rgb * 255 + 0.5).astype(np.uint8)


def split_render(pid, spp):
    import mitsuba as mi
    mi.set_variant("metal_ad_rgb")
    import render_ref as rr
    shot, preset = pid.split("_", 1)
    out = {}
    for part, sun, sky in (("sun", True, False), ("sky", False, True)):
        sd = rr.build_scene_dict(mi, shot, preset, 64, scale=0.5, sun=sun, sky=sky, max_depth=-1)
        scn = mi.load_dict(sd)
        n = max(1, spp // 64)
        acc = None
        for k in range(n):
            img = np.array(mi.render(scn, seed=1000 + k), dtype=np.float64)
            acc = img if acc is None else acc + img
        out[part] = acc / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*", default=PAIRS)
    ap.add_argument("--split-spp", type=int, default=1024)
    a = ap.parse_args()
    od = OUT / "multiple"
    od.mkdir(parents=True, exist_ok=True)
    res, tiles = [], []
    for pid in a.pairs:
        single = np.load(OUT / "exact" / f"{pid}_exact_single.npy").astype(np.float64)
        sun1 = np.load(OUT / "exact_nobg" / f"{pid}_exact_single.npy").astype(np.float64)
        full = np.load(REF_OUT / f"{pid}.npy").astype(np.float64)
        bg = single - sun1
        cloud = lum(sun1) > 0
        ins_full = full - bg  # light scattered toward the camera by the cloud (full) = full - T * sky
        o = {"pair": pid}
        o["cloud_px"] = int(cloud.sum())
        o["single_fraction_of_inscatter_lum"] = float(lum(sun1)[cloud].sum() / lum(ins_full)[cloud].sum())
        o["full_over_single_inscatter_rgb"] = [float(x) for x in ins_full[cloud].sum(0) / sun1[cloud].sum(0)]
        k = 4
        rb = block(lum(ins_full), k) / np.maximum(block(lum(sun1), k), 1e-9)
        mb = block(cloud.astype(float), k) > 0.999
        o["full_over_single_4x4_percentiles_10_50_90"] = [float(x) for x in np.percentile(rb[mb], [10, 50, 90])]
        o["total_image_full_over_single_cloud"] = float(lum(full)[cloud].sum() / lum(single)[cloud].sum())
        if a.split_spp > 0:
            sp = split_render(pid, a.split_spp)
            np.save(od / f"{pid}_split_sun.npy", sp["sun"].astype(np.float32))
            np.save(od / f"{pid}_split_sky.npy", sp["sky"].astype(np.float32))
            # half-resolution comparison: the sky part includes T * sky (the background), remove it
            bg_h = block(bg, 2); sun1_h = block(sun1, 2)
            cl_h = block(cloud.astype(float), 2) > 0.999
            sun_all = sp["sun"]; sky_in = sp["sky"] - bg_h
            tot = lum(sun_all)[cl_h].sum() + lum(sky_in)[cl_h].sum()
            o["split_spp"] = a.split_spp
            o["share_sun_single"] = float(lum(sun1_h)[cl_h].sum() / tot)
            o["share_sun_multiple"] = float((lum(sun_all)[cl_h].sum() - lum(sun1_h)[cl_h].sum()) / tot)
            o["share_sky_all_orders"] = float(lum(sky_in)[cl_h].sum() / tot)
            o["split_vs_full_check"] = float(tot / lum(block(ins_full, 2))[cl_h].sum())
        res.append(o)
        ex = exposure(pid)
        rmap = np.repeat(np.repeat(rb, k, 0), k, 1)
        foot = (f"cloud in-scattering: full / sun-single = {o['full_over_single_inscatter_rgb'][1]:.2f} (G); "
                f"4x4 p10/p50/p90 {', '.join(f'{x:.1f}' for x in o['full_over_single_4x4_percentiles_10_50_90'])}")
        if "share_sun_single" in o:
            foot += (f" | shares: sun single {o['share_sun_single']:.0%}, sun multiple {o['share_sun_multiple']:.0%}, "
                     f"sky {o['share_sky_all_orders']:.0%}")
        tiles.append(row([labelled(tonemap(single, ex), f"{pid}: exact sun single scattering + T*sky"),
                          labelled(tonemap(full, ex), "Mitsuba full (all orders, sun + sky)"),
                          labelled(ratio_img(rmap, cloud), "full / single in-scattering, 4x4 (black 1x .. yellow 32x)")],
                         footer=foot))
        print(json.dumps(o))
    stack(tiles).save(od / "multiple_vs_single.png")
    (od / "summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
