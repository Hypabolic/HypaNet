"""Watch the neurons along one pixel's ray (implementation of renderer/inspect.py).

    uv run python renderer/inspect.py NET.json --pair front_noon --pixel 320 200 [--samples 8|all] [--frame 0]
                                      [--res WxH] [--json OUT.json] [--harness-exact]

The numbers come from the shader itself: a debug variant of march.frag.glsl (DEBUG_RAY) re-marches the chosen
pixel's ray (same sub-pixel and start jitter as frame --frame of a render) and writes, for every accepted
sample (d > 0.002), the march state, every probe value, every unit's activation and S into a float texture.
The pixel radiance is rebuilt from those records exactly as the march does it:
    L = sum_k T_k * albedo * S_k * (1 - exp(-sigma_k dt)) + T_final * sky(v)
which equals `render.py --frames 1` at that pixel (frame 0) up to float rounding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from harness_client import Harness, HarnessError, read_net, resolve_pair  # noqa: E402

SUMMARY = ["n_samples", "t0", "t1", "dt", "T_final", "sky.r", "sky.g", "sky.b", "dir.x", "dir.y", "dir.z"]


def run_inspect(h: Harness, shot, preset, px, py, frame=0, width=None, height=None, exact=False):
    res = h.inspect(shot=shot, preset=preset, px=px, py=py, frame=frame, width=width, height=height, exact=exact)
    tab = res["table"]
    names = res["names"]
    summ = dict(zip(SUMMARY, tab[0, : len(SUMMARY)].tolist()))
    n = int(round(summ["n_samples"]))
    rec = tab[1 : 1 + n, : len(names)].astype(np.float64)
    albedo = res["albedo"]
    iS = names.index("S.r")
    S = rec[:, iS : iS + 3]
    w = rec[:, names.index("T")] * rec[:, names.index("alpha")]  # T_k (1 - exp(-sigma dt))
    contrib = albedo * w[:, None] * S
    sky = np.array([summ["sky.r"], summ["sky.g"], summ["sky.b"]])
    L_in = contrib.sum(0) if n else np.zeros(3)
    L_bg = summ["T_final"] * sky
    return {"names": names, "summary": summ, "records": rec, "contrib": contrib, "weight": w,
            "L": L_in + L_bg, "L_inscatter": L_in, "L_background": L_bg, "env": res}


def pick_samples(weight: np.ndarray, k: int) -> list[int]:
    n = len(weight)
    if n == 0:
        return []
    if k >= n:
        return list(range(n))
    tot = weight.sum()
    if tot <= 0:
        return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))
    cdf = np.cumsum(weight) / tot
    qs = (np.arange(k) + 0.5) / k
    idx = sorted(set(np.searchsorted(cdf, qs).clip(0, n - 1).tolist()))
    return idx


def fmt(x: float) -> str:
    if x == 0:
        return "0"
    a = abs(x)
    if 1e-3 <= a < 1e5:
        return f"{x:.5g}"
    return f"{x:.4e}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("net")
    ap.add_argument("--pair", help="pair id like front_noon")
    ap.add_argument("--shot", help="pair id, or a shot name with --preset")
    ap.add_argument("--preset")
    ap.add_argument("--pixel", nargs=2, type=int, required=True, metavar=("X", "Y"), help="column, row (0,0 = top left)")
    ap.add_argument("--samples", default="8", help="number of samples to print (spread over the ray's contribution), or 'all'")
    ap.add_argument("--frame", type=int, default=0, help="frame index whose jitter to replay (render.py frame 0..N-1)")
    ap.add_argument("--res", help="WxH override (must match the render you are inspecting)")
    ap.add_argument("--json", help="write every sample's full record to this JSON file")
    ap.add_argument("--harness-exact", action="store_true", help="HARNESS ONLY: also show exact T_sun and phase")
    ap.add_argument("--scene-dir")
    a = ap.parse_args(argv)
    ident = a.pair or a.shot
    if not ident:
        ap.error("give --pair (or --shot)")
    res = [int(x) for x in a.res.lower().split("x")] if a.res else [None, None]

    with Harness(scene_dir=a.scene_dir, harness_mode=a.harness_exact) as h:
        shot, preset, exposure, pid = resolve_pair(h.shots, ident, a.preset)
        W = res[0] or h.shots["shots"][shot]["width"]
        H = res[1] or h.shots["shots"][shot]["height"]
        if not (0 <= a.pixel[0] < W and 0 <= a.pixel[1] < H):
            ap.error(f"--pixel must be inside {W}x{H}")
        try:
            info = h.set_network(read_net(a.net))
            r = run_inspect(h, shot, preset, a.pixel[0], a.pixel[1], a.frame, res[0], res[1], a.harness_exact)
        except HarnessError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scene"))
    from tonemap import tonemap

    names, rec, s = r["names"], r["records"], r["summary"]
    n = len(rec)
    env = r["env"]
    hdr = len(["t", "x", "y", "z", "d", "T", "dt", "alpha"])
    nprobe = info["probes"]
    layer_sizes = info["layers"]
    print(f"{pid}  pixel ({a.pixel[0]}, {a.pixel[1]}) of {W}x{H}, frame {a.frame} jitter;  network: {nprobe} probes, "
          f"layers {layer_sizes}, {info['params']} params")
    print(f"ray dir v = ({s['dir.x']:.5f}, {s['dir.y']:.5f}, {s['dir.z']:.5f}); box t = {s['t0']:.1f} .. {s['t1']:.1f} m; "
          f"step {s['dt']:.3f} m; {n} samples with d > 0.002; T_final = {s['T_final']:.5g}")
    print(f"E_sun = {np.round(env['E_sun'], 4).tolist()}  L_sky = {np.round(env['L_sky'], 4).tolist()}  "
          f"L_gnd = {np.round(env['L_gnd'], 4).tolist()}  albedo {env['albedo']}")
    L = r["L"]
    print(f"pixel L = ({fmt(L[0])}, {fmt(L[1])}, {fmt(L[2])})  = in-scatter ({fmt(r['L_inscatter'][0])}, "
          f"{fmt(r['L_inscatter'][1])}, {fmt(r['L_inscatter'][2])}) + T_final * sky ({fmt(r['L_background'][0])}, "
          f"{fmt(r['L_background'][1])}, {fmt(r['L_background'][2])});  8-bit at exposure {exposure}: "
          f"{tonemap(L[None, :], exposure)[0].tolist()}")
    k = n if a.samples == "all" else int(a.samples)
    idx = pick_samples(r["weight"], k)
    if n:
        print(f"showing {len(idx)} of {n} samples" + ("" if a.samples == "all" else
              " (spread evenly over the ray's cumulative contribution weight T*(1-exp(-sigma dt)); --samples all for every one)"))
    for i in idx:
        v = rec[i]
        print(f"\n#{i}: t={v[0]:.1f} m  x=({v[1]:.1f}, {v[2]:.1f}, {v[3]:.1f})  d={v[4]:.4f}  T={v[5]:.5g}  "
              f"1-exp(-sigma dt)={v[7]:.5g}  weight T*(1-e)={r['weight'][i]:.5g}")
        print("  probes:")
        for j in range(nprobe):
            print(f"    {names[hdr + j]} = {fmt(v[hdr + j])}")
        base = hdr + nprobe
        for li, nout in enumerate(layer_sizes):
            vals = v[base : base + nout]
            acts = [names[base + j].split(":", 1)[1] for j in range(nout)]
            print(f"  layer {li} ({nout} units):")
            for j0 in range(0, nout, 6):
                print("    " + "  ".join(f"u{j}[{acts[j]}]={fmt(vals[j])}" for j in range(j0, min(nout, j0 + 6))))
            base += nout
        out9 = rec[i, base - 9 : base]
        S = rec[i, base : base + 3]
        print(f"  a_sun = ({fmt(out9[0])}, {fmt(out9[1])}, {fmt(out9[2])})  a_sky = ({fmt(out9[3])}, {fmt(out9[4])}, "
              f"{fmt(out9[5])})  a_gnd = ({fmt(out9[6])}, {fmt(out9[7])}, {fmt(out9[8])})")
        print(f"  S = ({fmt(S[0])}, {fmt(S[1])}, {fmt(S[2])})  contribution to L = ({fmt(r['contrib'][i][0])}, "
              f"{fmt(r['contrib'][i][1])}, {fmt(r['contrib'][i][2])})")
        if a.harness_exact:
            ts, ph = rec[i, base + 3], rec[i, base + 4]
            print(f"  [harness] exact T_sun = {fmt(ts)}, phase = {fmt(ph)}, exact a_sun = {fmt(ts * ph)}")

    if a.json:
        out = {"pair": pid, "pixel": a.pixel, "frame": a.frame, "width": W, "height": H, "summary": s,
               "L": L.tolist(), "L_inscatter": r["L_inscatter"].tolist(), "L_background": r["L_background"].tolist(),
               "E_sun": env["E_sun"], "L_sky": env["L_sky"], "L_gnd": env["L_gnd"], "albedo": env["albedo"],
               "names": names, "samples": [dict(zip(names, map(float, row))) for row in rec],
               "contribution": r["contrib"].tolist()}
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nfull records -> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
