"""SPEC 6.2: empty-density render vs Mitsuba (sky background only), for every shot x preset (24 combinations).

    uv run python validate/check_background.py [--spp 256]

Renderer: a copy of scene/out with an all-zero density.u8 (validate/out/empty_scene/), rendered twice:
  --mode exact-single, and net mode with a zero network (validate/test_networks/zero_net.json), so the path the
  author uses is covered too.
Mitsuba: the same camera with the envmap only (no medium, no sun), box filter, `--spp` samples per pixel.
Prints per-combination mean / max per-pixel relative error (luminance) and per-channel ratio of means, writes
validate/out/background/summary.json and a contact sheet of side-by-sides and difference maps.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys

import numpy as np

from common import CLOUDS, exposure, OUT, SCENE_OUT, VALIDATE, diverging, labelled, lum, render, row, shots, stack, tonemap

sys.path.insert(0, str(CLOUDS / "reference"))


def make_empty_scene():
    dst = OUT / "empty_scene"
    dst.mkdir(parents=True, exist_ok=True)
    for f in SCENE_OUT.iterdir():
        if f.is_file() and f.name != "density.u8":
            if not (dst / f.name).exists() or (dst / f.name).stat().st_mtime < f.stat().st_mtime:
                shutil.copy2(f, dst / f.name)
    meta = json.loads((SCENE_OUT / "density.json").read_text())
    nx, ny, nz = meta["dims"]
    np.zeros(nx * ny * nz, np.uint8).tofile(dst / "density.u8")
    return dst


def zero_net():
    p = VALIDATE / "test_networks" / "zero_net.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"probes": [{"type": "const", "value": 1.0}],
                             "layers": [{"W": [[0.0]] * 9, "b": [0.0] * 9, "act": "linear"}]}))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--skip-render", action="store_true")
    a = ap.parse_args()
    S = shots()
    combos = [f"{sh}_{pr}" for sh in S["shots"] for pr in S["presets"]]
    bgdir = OUT / "background"
    bgdir.mkdir(parents=True, exist_ok=True)
    if not a.skip_render:
        scene = make_empty_scene()
        net = zero_net()
        render(["--mode", "exact-single", "--frames", str(a.frames), "--scene-dir", str(scene), "--pair", *combos,
                "--out", str(bgdir / "renderer_exact") + "/"])
        render([str(net), "--frames", str(a.frames), "--scene-dir", str(scene), "--pair", *combos,
                "--out", str(bgdir / "renderer_net") + "/"])
        import mitsuba as mi
        mi.set_variant("metal_ad_rgb")
        import render_ref as rr
        (bgdir / "mitsuba").mkdir(exist_ok=True)
        for pid in combos:
            shot, preset = pid.split("_", 1)
            sd = rr.build_scene_dict(mi, shot, preset, a.spp, medium=False, sun=False, sky=True, max_depth=1)
            img = np.zeros((S["shots"][shot]["height"], S["shots"][shot]["width"], 3))
            n = max(1, a.spp // 64)
            sd["sensor"]["sampler"]["sample_count"] = a.spp // n
            scn = mi.load_dict(sd)
            for k in range(n):
                img += np.array(mi.render(scn, seed=k)) / n
            np.save(bgdir / "mitsuba" / f"{pid}.npy", img.astype(np.float32))
    res, tiles = [], []
    for pid in combos:
        m = np.load(bgdir / "mitsuba" / f"{pid}.npy").astype(np.float64)
        o = {"pair": pid}
        for tag in ("exact", "net"):
            r = np.load(bgdir / f"renderer_{tag}" / f"{pid}_{'exact_single' if tag == 'exact' else 'zero_net'}.npy")
            r = r.astype(np.float64)
            rel = (lum(r) - lum(m)) / np.maximum(lum(m), 1e-12)
            # the horizon row (sky/ground step) is box-filtered by both, but with different sub-pixel sampling:
            # report it separately
            Ym = lum(m)
            step = np.zeros_like(Ym, bool)
            g = np.abs(np.diff(Ym, axis=0)) / np.maximum(Ym[:-1], 1e-12) > 0.05
            step[:-1] |= g; step[1:] |= g
            o[f"{tag}_meanrel"] = float(np.abs(rel).mean())
            o[f"{tag}_meanrel_no_horizon"] = float(np.abs(rel[~step]).mean())
            o[f"{tag}_maxrel_no_horizon"] = float(np.abs(rel[~step]).max())
            o[f"{tag}_horizon_px"] = int(step.sum())
            o[f"{tag}_ratio_rgb"] = [float(x) for x in r.reshape(-1, 3).mean(0) / m.reshape(-1, 3).mean(0)]
            if tag == "net":
                ex = exposure(pid)
                if pid in ("distant_sunset", "below_noon", "backlit_golden", "front_noon", "side_sunset", "quarter_afternoon"):
                    tiles.append(row([labelled(tonemap(r, ex), f"{pid} renderer (zero net, empty density)"),
                                      labelled(tonemap(m, ex), f"Mitsuba envmap, {a.spp} spp"),
                                      labelled(diverging(rel, np.ones_like(step), 0.01), "rel. diff, +-1%")]))
        res.append(o)
        print(f"{pid:18s} exact: mean {o['exact_meanrel']:.5f} (excl. horizon {o['exact_meanrel_no_horizon']:.5f}, "
              f"max {o['exact_maxrel_no_horizon']:.4f})  net: mean {o['net_meanrel']:.5f} max "
              f"{o['net_maxrel_no_horizon']:.4f}  ratio rgb {np.round(o['net_ratio_rgb'], 4).tolist()}")
    (bgdir / "summary.json").write_text(json.dumps(res, indent=1))
    stack(tiles).save(bgdir / "background_compare.png")
    w = max(max(o["exact_meanrel"], o["net_meanrel"]) for o in res)
    print(f"worst mean |rel| over all 24: {w:.5f}")


if __name__ == "__main__":
    main()
