"""Headless renderer CLI (SPEC section 4): Playwright + Chromium (Metal ANGLE) running the three.js ray marcher.

    uv run python renderer/render.py NET.json --pair front_noon [--frames 16] [--out out.png] [--compare [REF.png]]
    uv run python renderer/render.py --mode exact-single --pair side_noon        # HARNESS ONLY
    uv run python renderer/render.py NET.json --set reference --out renderer/out/r1/

For each pair writes <out>.png (tone-mapped by the shader's tone map), <out>.npy (linear float32 RGB, H x W x 3,
row 0 at the top) and <out>.json (timings, settings, tone-map self-check). --compare makes <out>_compare.png
(render | reference) with labels and, if the reference .npy exists, reports the linear relative error.
"""

from __future__ import annotations

import os
import sys

# This directory holds inspect.py, which would shadow the standard library's `inspect`: move it to the end.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE] + [_HERE]

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from harness_client import RENDERER_DIR, Harness, HarnessError, read_net, resolve_pair  # noqa: E402

CLOUDS_DIR = RENDERER_DIR.parent
REFERENCE_DIR = CLOUDS_DIR / "reference" / "out"


def tonemap_np(lin, exposure):
    sys.path.insert(0, str(CLOUDS_DIR / "scene"))
    try:
        from tonemap import tonemap  # scene/tonemap.py (the SPEC tone map, numpy)
    finally:
        sys.path.pop(0)
    return tonemap(lin, exposure)


def font(size=14):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def labelled(img: Image.Image, text: str) -> Image.Image:
    bar = 22
    out = Image.new("RGB", (img.width, img.height + bar), (20, 20, 20))
    out.paste(img, (0, bar))
    ImageDraw.Draw(out).text((6, 3), text, fill=(235, 235, 235), font=font(14))
    return out


def side_by_side(a: Image.Image, la: str, b: Image.Image, lb: str, footer: str | None = None) -> Image.Image:
    A, B = labelled(a, la), labelled(b, lb)
    if B.size != A.size:
        B = B.resize(A.size, Image.LANCZOS)
    foot = 22 if footer else 0
    out = Image.new("RGB", (A.width * 2 + 4, A.height + foot), (20, 20, 20))
    out.paste(A, (0, 0))
    out.paste(B, (A.width + 4, 0))
    if footer:
        ImageDraw.Draw(out).text((6, A.height + 3), footer, fill=(235, 235, 235), font=font(13))
    return out


def rel_error(lin: np.ndarray, ref: np.ndarray) -> dict:
    """Linear relative error: mean |a - b| / mean b (per image, luminance-weighted channels), plus RGB means."""
    if lin.shape != ref.shape:
        return {"error": f"shape mismatch {lin.shape} vs {ref.shape}"}
    diff = np.abs(lin.astype(np.float64) - ref.astype(np.float64))
    return {"mean_abs_over_mean_ref": float(diff.mean() / max(ref.mean(), 1e-12)),
            "mean_render": [float(x) for x in lin.reshape(-1, 3).mean(0)],
            "mean_ref": [float(x) for x in ref.reshape(-1, 3).mean(0)],
            "ratio_of_means": [float(a / max(b, 1e-12)) for a, b in
                               zip(lin.reshape(-1, 3).mean(0), ref.reshape(-1, 3).mean(0))]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("net", nargs="?", help="network.json (omit with --mode exact-single)")
    ap.add_argument("--mode", choices=["net", "exact-single"], default="net",
                    help="exact-single = HARNESS-ONLY exact sun single scattering (no network)")
    ap.add_argument("--pair", nargs="+", default=[], help="pair id(s) like front_noon (any shot x preset)")
    ap.add_argument("--shot", nargs="+", default=[], help="shot name(s) (with --preset) or pair id(s)")
    ap.add_argument("--preset", help="sun preset when --shot is a bare shot name")
    ap.add_argument("--set", choices=["reference", "heldback", "all"], help="render a whole pair set")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--res", help="WxH override (vertical fov kept)")
    ap.add_argument("--exposure", type=float, help="override the pair's exposure")
    ap.add_argument("--budget-ms", type=float, default=40.0,
                    help="target time of one draw; network evaluations per pixel per pass adapt to it (macOS drops "
                         "draws whose GPU threads run ~100+ ms)")
    ap.add_argument("--out", help="output PNG path (one pair) or directory (several); default renderer/out/")
    ap.add_argument("--compare", nargs="?", const="auto", help="side-by-side with REF.png (default: reference/out/<pair>.png)")
    ap.add_argument("--scene-dir", help="scene output directory (default ../scene/out)")
    ap.add_argument("--no-background", action="store_true",
                    help="HARNESS ONLY (with --mode exact-single): drop the background term T * sky(v)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    if a.mode == "net" and not a.net:
        ap.error("give NET.json, or --mode exact-single")
    if a.mode == "exact-single" and a.net:
        ap.error("--mode exact-single takes no network")
    if a.no_background and a.mode != "exact-single":
        ap.error("--no-background is harness-only and needs --mode exact-single")
    if a.frames < 1:
        ap.error("--frames must be >= 1")
    res = [int(x) for x in a.res.lower().split("x")] if a.res else [None, None]
    net_text = read_net(a.net) if a.net else None
    tag = Path(a.net).stem if a.net else "exact_single"

    todo = None
    budget = a.budget_ms
    for attempt in range(3):
        with Harness(scene_dir=a.scene_dir, harness_mode=(a.mode == "exact-single"), verbose=a.verbose) as h:
            shots = h.shots
            if todo is None:
                todo = []
                for p in a.pair:
                    todo.append(resolve_pair(shots, p))
                for s in a.shot:
                    todo.append(resolve_pair(shots, s, a.preset))
                if a.set:
                    for g in (["reference", "heldback"] if a.set == "all" else [a.set]):
                        for p in shots["pairs"][g]:
                            todo.append(resolve_pair(shots, p["id"]))
                if not todo:
                    ap.error("nothing to render: give --pair, --shot or --set")
                multi = len(todo) > 1
                as_dir = multi or not a.out or a.out.endswith("/")
                out_base = Path(a.out) if a.out else RENDERER_DIR / "out"
                if as_dir:
                    out_base.mkdir(parents=True, exist_ok=True)
            net_info = None
            if net_text is not None:
                try:
                    net_info = h.set_network(net_text)
                except HarnessError as e:
                    print(f"network error: {e}", file=sys.stderr)
                    return 2
            print(f"GPU: {h.info['gl']['renderer']} | startup {h.info['startupS']:.1f}s (pyramid {h.info['pyramidMs']:.0f} ms)")
            if net_info and attempt == 0:
                print(f"network: {net_info['probes']} probes, layers {net_info['layers']}, {net_info['params']} params")
            while todo:
                shot, preset, exposure, pid = todo[0]
                if a.exposure is not None:
                    exposure = a.exposure
                png = out_base / f"{pid}_{tag}.png" if as_dir else Path(a.out)
                png.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.perf_counter()
                try:
                    lin, ldr, stats = h.render(shot=shot, preset=preset, width=res[0], height=res[1],
                                               mode="exact" if a.mode == "exact-single" else "net",
                                               frames=a.frames, budgetMs=budget, exposure=exposure,
                                               noBackground=a.no_background)
                except HarnessError as e:
                    if "GPU_DROPPED" in str(e) and attempt < 2:
                        budget /= 3
                        print(f"{pid}: {e}\n    restarting the browser and retrying with --budget-ms {budget:.0f}",
                              file=sys.stderr)
                        break
                    print(f"{pid}: render failed: {e}", file=sys.stderr)
                    return 3
                wall = time.perf_counter() - t0
                finish(a, h, png, pid, shot, preset, exposure, tag, lin, ldr, stats, wall, budget)
                todo.pop(0)
        if not todo:
            return 0
    return 3


def finish(a, h, png, pid, shot, preset, exposure, tag, lin, ldr, stats, wall, budget):
    Image.fromarray(ldr, "RGB").save(png)
    np.save(png.with_suffix(".npy"), lin.astype(np.float32))
    nonfinite = int((~np.isfinite(lin)).any(-1).sum())
    tm_diff = np.abs(tonemap_np(lin, exposure).astype(int) - ldr.astype(int))
    meta = {"pair": pid, "shot": shot, "preset": preset, "exposure": exposure, "mode": a.mode,
            "no_background": a.no_background, "network": str(a.net) if a.net else None, "frames": a.frames,
            "width": lin.shape[1], "height": lin.shape[0], "render_s": stats["ms"] / 1000, "wall_s": wall,
            "compile_ms": stats["compileMs"], "draws": stats["draws"], "passes": stats["passes"],
            "tiles": stats["tiles"], "tile_splits": stats["splits"], "max_draw_ms": stats["maxDrawMs"],
            "budget_ms": budget, "evals_per_frame": stats["evalsPerFrame"], "final_eval_cap": stats["finalCap"],
            "nonfinite_pixels": nonfinite,
            "tonemap_check": {"max_abs_diff_8bit": int(tm_diff.max()), "pixels_differing": int((tm_diff > 0).any(-1).sum())},
            "gpu": h.info["gl"]["renderer"]}
    line = (f"{pid}: {lin.shape[1]}x{lin.shape[0]}, {a.frames} frames, {stats['ms'] / 1000:.2f} s render "
            f"({wall:.2f} s wall; {stats['tiles']} tiles, {stats['draws']} draws, max draw {stats['maxDrawMs']:.0f} ms, "
            f"eval cap {stats['finalCap']}; {stats['evalsPerFrame'] / 1e6:.2f} M evals/frame) -> {png}")
    if nonfinite:
        line += f"\n    WARNING: {nonfinite} non-finite pixels (NaN/inf; tone-mapped as black)"
    if tm_diff.max() > 1:
        line += f"\n    WARNING: GLSL vs numpy tone map differ by up to {tm_diff.max()}"
    if a.compare:
        ref_png = Path(a.compare) if a.compare != "auto" else REFERENCE_DIR / f"{pid}.png"
        if ref_png.is_file():
            footer = None
            ref_npy = ref_png.with_suffix(".npy")
            if ref_npy.is_file():
                err = rel_error(lin, np.load(ref_npy))
                meta["compare"] = {"reference": str(ref_png), **err}
                if "error" not in err:
                    footer = (f"linear: mean|render-ref|/mean(ref) = {err['mean_abs_over_mean_ref']:.3f}; "
                              f"mean render/ref (R,G,B) = " + ", ".join(f"{r:.3f}" for r in err["ratio_of_means"]))
                    line += f"\n    vs reference: {footer}"
            cmp = side_by_side(Image.fromarray(ldr, "RGB"), f"{tag}  ({pid}, {a.frames} frames)",
                               Image.open(ref_png).convert("RGB"), f"reference ({ref_png.name})", footer)
            cpath = png.with_name(png.stem + "_compare.png")
            cmp.save(cpath)
            line += f"\n    compare -> {cpath}"
        else:
            line += f"\n    (no reference image at {ref_png})"
    png.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(line, flush=True)


if __name__ == "__main__":
    sys.exit(main())
