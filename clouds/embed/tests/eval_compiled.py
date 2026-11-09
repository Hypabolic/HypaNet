"""Verify and tune the compiled network (embed/DESIGN.md "Compile", "Bake") in the research harness.

    uv run python embed/tests/eval_compiled.py exact                       # --no-bake == interpreter (+ test nets)
    uv run python embed/tests/eval_compiled.py tune [--configs a,b,...]    # bake variants, REFERENCE split only
    uv run python embed/tests/eval_compiled.py heldback --config NAME      # once, at the end: final config + interpreter
    uv run python embed/tests/eval_compiled.py timing --configs a,b,...    # interleaved frame_ms (3 rounds, 2 pairs)

Every render is 640x400 (the pair's camera), 16 frames, through render.py's path (window.harness.render) with the
OPT-IN compiled hook (mode "compiled"). Per pair and config:

- vs interpreter: lum_cloud = sum |Y_c - Y_i| / sum Y_i over cloud pixels (the reference's mask, as score.py),
  lum_full = mean |Y_c - Y_i| / mean Y_i, max_rel / mean_rel = max / mean |c - i| over all channels / mean(i);
- vs the Mitsuba references: scripts/score.py's metrics (full, cloud, block8, ratio_rgb), and d_cloud = cloud minus
  the interpreter's cloud on the same pair;
- render_s (harness tile scheduler, 16 frames, per-pass readbacks: overhead-dominated for cheap shaders), frame_ms
  (16 frames rendered as one full-image tile, 40 ms draw budget; closer to GPU cost), bake_ms (all textures, first
  bake; sun textures rebake on every sun change), fetches per sample (manifest).

Writes embed/tests/out/<command>.json and prints a table. Interpreter renders are cached in embed/tests/out/interp/
(gitignored .npy). Timings are indicative: other processes may share the GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

CLOUDS = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CLOUDS / "renderer"), str(CLOUDS / "scripts"), str(CLOUDS / "embed" / "compile")]
from compile_net import compile_net  # noqa: E402
from harness_client import Harness, resolve_pair  # noqa: E402
from score import score  # noqa: E402

NET = CLOUDS / "runs" / "r1" / "author" / "work" / "network.json"
SCENE = CLOUDS / "scene" / "out"
REF = CLOUDS / "reference" / "out"
OUT = CLOUDS / "embed" / "tests" / "out"
TEST_NETS = CLOUDS / "renderer" / "tests" / "nets"
LUM = np.array([0.2126, 0.7152, 0.0722])
FRAMES = 16

# name -> compile_net options (tuned on the reference split only)
CONFIGS = {
    "nobake": dict(no_bake=True),
    "b128": dict(res_spec="128x80x128"),
    "b256": dict(res_spec="256x160x256"),
    "b192": dict(res_spec="192x120x192"),
    "b128_live01": dict(res_spec="128x80x128", live="p0,p1"),
    "b256_live01": dict(res_spec="256x160x256", live="p0,p1"),
    "b128_live0": dict(res_spec="128x80x128", live="p0"),
    "b128_live012": dict(res_spec="128x80x128", live="p0,p1,p2"),
    "s256_t128_live01": dict(res_spec="sun=256x160x256,static=128x80x128", live="p0,p1"),
    "s128_t256": dict(res_spec="sun=128x80x128,static=256x160x256"),
    "s256_t128": dict(res_spec="sun=256x160x256,static=128x80x128"),
    "b128_live01_c64": dict(res_spec="128x80x128,coarse=64x40x64", coarse_blur=150, live="p0,p1"),
    "b128_c64": dict(res_spec="128x80x128,coarse=64x40x64", coarse_blur=150),
    "b96_live01": dict(res_spec="96x60x96", live="p0,p1"),
    "b128_f32": dict(res_spec="128x80x128", fmt="RGBA32F"),
    "b256_f32": dict(res_spec="256x160x256", fmt="RGBA32F"),
    "s192_t128": dict(res_spec="sun=192x120x192,static=128x80x128"),
    "s192_t128_live01": dict(res_spec="sun=192x120x192,static=128x80x128", live="p0,p1"),
    "s256_t96": dict(res_spec="sun=256x160x256,static=96x60x96"),
    "s256_t96_live01": dict(res_spec="sun=256x160x256,static=96x60x96", live="p0,p1"),
    "b128_live0123": dict(res_spec="128x80x128", live="p0,p1,p2,p3"),
    "s256c96_t128": dict(res_spec="sun=256x160x256,static=128x80x128,coarse=96x60x96", coarse_blur=150),
    "s128_t96_live0123": dict(res_spec="sun=128x80x128,static=96x60x96", live="p0,p1,p2,p3"),
    "s128_t64_live0123": dict(res_spec="sun=128x80x128,static=64x40x64", live="p0,p1,p2,p3"),
    "s128_t96_live01234": dict(res_spec="sun=128x80x128,static=96x60x96", live="p0,p1,p2,p3,p4"),
    "b96_live0123": dict(res_spec="96x60x96", live="p0,p1,p2,p3"),
    "b128_live01234": dict(res_spec="128x80x128", live="p0,p1,p2,p3,p4"),
    "s128c64_t96_live0123": dict(res_spec="sun=128x80x128,static=96x60x96,coarse=64x40x64", coarse_blur=150,
                                 live="p0,p1,p2,p3"),
    "s192c96_t128_live01": dict(res_spec="sun=192x120x192,static=128x80x128,coarse=96x60x96", coarse_blur=150,
                                live="p0,p1"),
}


def pairs(split):
    shots = json.loads((SCENE / "shots.json").read_text())
    return [p["id"] for p in shots["pairs"][split]]


def ref_mask(pid):
    return np.load(REF / f"{pid}_se.npy").max(-1) > 1e-3


def vs_interp(c, i, mask):
    yc, yi = c @ LUM, i @ LUM
    d = np.abs(c.astype(np.float64) - i.astype(np.float64))
    return {"lum_cloud": float(np.abs(yc - yi)[mask].sum() / yi[mask].sum()),
            "lum_full": float(np.abs(yc - yi).mean() / yi.mean()),
            "max_rel": float(d.max() / i.mean()), "mean_rel": float(d.mean() / i.mean()),
            "max_rel_pixel": float((d / (np.abs(i.astype(np.float64)) + 1e-3 * i.mean())).max())}


def vs_ref(lin, pid):
    return score(lin, np.load(REF / f"{pid}.npy"), np.load(REF / f"{pid}_se.npy"))


def render(h, pid, mode, **kw):
    shot, preset, ex, _ = resolve_pair(h.shots, pid)
    lin, _, st = h.render(shot=shot, preset=preset, mode=mode, frames=kw.pop("frames", FRAMES), budgetMs=40,
                          exposure=ex, **kw)
    return lin, st


def frame_ms(h, pid, mode):
    """16 frames as one full-image tile (after a warm-up) -> ms per frame."""
    render(h, pid, mode, frames=2, tile=[640, 400])
    _, st = render(h, pid, mode, tile=[640, 400])
    return st["ms"] / FRAMES


def interp_renders(h, split, need_timing=True):
    d = OUT / "interp"
    d.mkdir(parents=True, exist_ok=True)
    h.set_network(NET.read_text())
    res = {}
    for pid in pairs(split):
        f = d / f"{pid}.npy"
        meta = d / f"{pid}.json"
        if f.is_file() and meta.is_file():
            lin, st = np.load(f), json.loads(meta.read_text())
        else:
            lin, st = render(h, pid, "net")
            st = {"render_s": st["ms"] / 1000, "max_draw_ms": st["maxDrawMs"]}
            np.save(f, lin)
            meta.write_text(json.dumps(st))
        res[pid] = (lin, st)
    return res


def run_config(h, name, opts, split, interp, gen_root, timing_pair="backlit_golden"):
    gen = gen_root / name
    man = compile_net(NET, SCENE, gen, **opts)
    h.set_compiled(f"/compiled/{name}/")
    rows = {}
    first_bake = None
    for pid in pairs(split):
        lin, st = render(h, pid, "compiled")
        if first_bake is None:
            first_bake = st.get("bakeMs", 0)
        il, ist = interp[pid]
        mask = ref_mask(pid)
        sc, si = vs_ref(lin, pid), vs_ref(il, pid)
        rows[pid] = {"vs_interp": vs_interp(lin, il, mask), "score": sc, "interp_score": si,
                     "d_cloud": sc["cloud"] - si["cloud"], "d_block8": sc["block8"] - si["block8"],
                     "render_s": st["ms"] / 1000, "bake_ms": st.get("bakeMs", 0), "interp_render_s": ist["render_s"],
                     "nonfinite": int((~np.isfinite(lin)).sum())}
    fm = frame_ms(h, timing_pair, "compiled") if timing_pair in rows else None
    agg = summarize(rows)
    agg.update({"fetches_per_sample": man["stats"]["fetches_per_sample_after"],
                "bake_textures": [f"{t['name']} {t['class']} {'x'.join(map(str, t['dims']))}" for t in man["textures"]],
                "bake_texels": man["stats"]["bake_texels"], "bake_ms_first": first_bake,
                "bake_mb": sum(np.prod(t["dims"]) * (16 if t["format"] == "RGBA32F" else 8) for t in man["textures"]) / 2 ** 20,
                # sun textures rebake whenever the preset changes between consecutive pairs (shader already compiled)
                "rebake_sun_ms_median": float(np.median([r["bake_ms"] for r in list(rows.values())[1:] if r["bake_ms"] > 0]
                                                        or [0])),
                f"frame_ms_{timing_pair}": fm, "live_probes": man["live_probes"]})
    return {"options": {k: v for k, v in opts.items()}, "summary": agg, "pairs": rows}


def summarize(rows):
    v = list(rows.values())
    m = lambda f: float(np.mean([f(r) for r in v]))  # noqa: E731
    return {"n": len(v), "lum_cloud_mean": m(lambda r: r["vs_interp"]["lum_cloud"]),
            "lum_cloud_max": float(max(r["vs_interp"]["lum_cloud"] for r in v)),
            "max_rel_max": float(max(r["vs_interp"]["max_rel"] for r in v)),
            "mean_rel_mean": m(lambda r: r["vs_interp"]["mean_rel"]),
            "cloud_mean": m(lambda r: r["score"]["cloud"]), "interp_cloud_mean": m(lambda r: r["interp_score"]["cloud"]),
            "block8_mean": m(lambda r: r["score"]["block8"]), "interp_block8_mean": m(lambda r: r["interp_score"]["block8"]),
            "full_mean": m(lambda r: r["score"]["full"]), "interp_full_mean": m(lambda r: r["interp_score"]["full"]),
            "d_cloud_mean": m(lambda r: r["d_cloud"]), "d_cloud_absmax": float(max(abs(r["d_cloud"]) for r in v)),
            "d_block8_absmax": float(max(abs(r["d_block8"]) for r in v)),
            "render_s_total": float(sum(r["render_s"] for r in v)),
            "interp_render_s_total": float(sum(r["interp_render_s"] for r in v))}


def print_table(results):
    print(f"\n{'config':20s} {'fetch':>5s} {'lumC':>7s} {'lumCmax':>7s} {'maxrel':>7s} {'cloud':>6s} {'icloud':>6s} "
          f"{'dC|max':>7s} {'blk8':>6s} {'iblk8':>6s} {'rend_s':>6s} {'frame':>6s} {'rebake':>6s} {'MB':>5s}")
    for name, r in results.items():
        s = r["summary"]
        fm = next((v for k, v in s.items() if k.startswith("frame_ms")), None)
        print(f"{name:20s} {s['fetches_per_sample']:5d} {s['lum_cloud_mean']:7.4f} {s['lum_cloud_max']:7.4f} "
              f"{s['max_rel_max']:7.4f} {s['cloud_mean']:6.4f} {s['interp_cloud_mean']:6.4f} {s['d_cloud_absmax']:7.4f} "
              f"{s['block8_mean']:6.4f} {s['interp_block8_mean']:6.4f} {s['render_s_total']:6.2f} "
              f"{(fm or 0):6.1f} {s['rebake_sun_ms_median']:6.1f} {s['bake_mb']:5.1f}")


def cmd_exact(h, gen_root):
    interp = interp_renders(h, "reference")
    res = {"r1_reference": run_config(h, "nobake", CONFIGS["nobake"], "reference", interp, gen_root)}
    res["r1_interp_frame_ms_backlit_golden"] = frame_ms(h, "backlit_golden", "net")
    # generality: the harness test nets, interpreter vs --no-bake (and the baked variant runs and is finite)
    tn = {}
    for f in sorted(TEST_NETS.glob("*.json")):
        h.set_network(f.read_text())
        for variant, opts in (("nobake", dict(no_bake=True)), ("bake", dict(res_spec="128x80x128"))):
            name = f"test_{f.stem}_{variant}"
            man = compile_net(f, SCENE, gen_root / name, **opts)
            h.set_compiled(f"/compiled/{name}/")
            for pid in ("front_noon", "backlit_golden", "below_golden"):
                kw = dict(width=320, height=200, frames=4)
                il, _ = render(h, pid, "net", **kw)
                cl, _ = render(h, pid, "compiled", **kw)
                d = np.abs(cl.astype(np.float64) - il.astype(np.float64))
                tn[f"{name}/{pid}"] = {"max_rel": float(d.max() / il.mean()), "mean_rel": float(d.mean() / il.mean()),
                                        "fetches": man["stats"]["fetches_per_sample_after"],
                                        "nonfinite": int((~np.isfinite(cl)).sum())}
    res["test_nets"] = tn
    h.set_network(NET.read_text())
    print_table({"nobake": res["r1_reference"]})
    for pid, r in res["r1_reference"]["pairs"].items():
        print(f"  {pid:20s} max|c-i|/mean(i) {r['vs_interp']['max_rel']:.2e}  mean {r['vs_interp']['mean_rel']:.2e}  "
              f"max per-pixel |c-i|/(|i| + 1e-3 mean) {r['vs_interp']['max_rel_pixel']:.2e}")
    print(f"  interpreter frame_ms backlit_golden {res['r1_interp_frame_ms_backlit_golden']:.1f}")
    for k, r in tn.items():
        print(f"  {k:45s} max {r['max_rel']:.2e} mean {r['mean_rel']:.2e} fetches {r['fetches']} nonfinite {r['nonfinite']}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["exact", "tune", "heldback", "timing"])
    ap.add_argument("--configs", help="comma list of config names (tune); default all")
    ap.add_argument("--config", help="final config name (heldback)")
    ap.add_argument("--tag", help="output file tag (default: the command)")
    a = ap.parse_args(argv)
    gen_root = OUT / "gen"
    gen_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Harness(extra_mounts={"/compiled/": gen_root}) as h:
        print(f"GPU: {h.info['gl']['renderer']}")
        h.set_network(NET.read_text())
        if a.cmd == "exact":
            res = cmd_exact(h, gen_root)
        elif a.cmd == "timing":
            names = a.configs.split(",")
            for n in names:
                compile_net(NET, SCENE, gen_root / n, **CONFIGS[n])
            res = {n: {} for n in names + ["interpreter"]}
            for rnd in range(3):
                for n in names + ["interpreter"]:
                    if n != "interpreter":
                        h.set_compiled(f"/compiled/{n}/")
                    for pid in ("backlit_golden", "front_noon"):
                        res[n].setdefault(pid, []).append(frame_ms(h, pid, "net" if n == "interpreter" else "compiled"))
            for n, r in res.items():
                print(f"{n:22s} " + "  ".join(f"{pid} {np.median(v):6.2f} ms/frame (min {min(v):.2f})" for pid, v in r.items()))
        elif a.cmd == "tune":
            interp = interp_renders(h, "reference")
            names = a.configs.split(",") if a.configs else [n for n in CONFIGS if n != "nobake"]
            res = {}
            for n in names:
                print(f"config {n} ...", flush=True)
                res[n] = run_config(h, n, CONFIGS[n], "reference", interp, gen_root)
                print_table({n: res[n]})
            print_table(res)
        else:
            if not a.config:
                ap.error("heldback needs --config")
            interp = interp_renders(h, "heldback")
            res = {a.config: run_config(h, a.config, CONFIGS[a.config], "heldback", interp, gen_root,
                                        timing_pair="backlit_noon")}
            print_table(res)
            for pid, r in res[a.config]["pairs"].items():
                print(f"  HELD-BACK {pid:20s} cloud {r['score']['cloud']:.4f} (interp {r['interp_score']['cloud']:.4f}) "
                      f"block8 {r['score']['block8']:.4f} (interp {r['interp_score']['block8']:.4f}) "
                      f"lum vs interp {r['vs_interp']['lum_cloud']:.4f}")
    res["_meta"] = {"split": {"exact": "reference", "tune": "reference", "heldback": "heldback", "timing": "reference"}[a.cmd],
                    "frames": FRAMES, "wall_s": time.time() - t0, "gpu": h.info["gl"]["renderer"]}
    out = OUT / f"{a.tag or a.cmd}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
