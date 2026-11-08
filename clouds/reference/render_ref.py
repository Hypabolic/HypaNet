"""HypaNet Clouds reference renderer (SPEC section 5): Mitsuba 3.9.1 volumetric path tracing of the scene in scene/out.

    uv run python reference/render_ref.py front_noon side_sunset ...       # full mode, default spp
    uv run python reference/render_ref.py --set reference                    # the 12 reference pairs
    uv run python reference/render_ref.py --set heldback                     # the 6 held-back pairs
    uv run python reference/render_ref.py --mode single front_noon ...       # sun single scattering only (validation)
    options: --spp N (total), --pass-spp N (per GPU launch), --scale S (resolution factor, tests only),
             --variant metal_ad_rgb|llvm_ad_rgb, --rr-depth N, --out DIR

Scene (identical to the three.js renderer, see reference/NOTES.md for the verification of each convention):
- density.u8 as a `gridvolume` (trilinear), to_world = translate(box_min) * scale(box_max - box_min), inside a cube
  with a null BSDF; heterogeneous medium, sigma_t = sigma_t_max * d, albedo 0.999,
  phase = blendphase(weight 0.2, hg(0.9), hg(-0.3)) = 0.8 HG(0.9) + 0.2 HG(-0.3).
- sun: `directional`, direction = -toward_sun, irradiance = E_sun (radiometric irradiance on a plane normal to the
  sun, verified with a closed-form slab test).
- sky: `envmap` from sky_<preset>.f32 with identity to_world and scale 1.
- camera: `perspective`, fov = fov_y_deg with fov_axis "y", look_at(position, look_at, up), far_clip 1e6,
  box pixel filter, 640 x 400.

Modes:
- full:   volpath, max_depth -1 (unlimited), Russian roulette from --rr-depth on. All light, all orders.
- single: L = sum T * omega * E_sun * T_sun * p * (1 - exp(-sigma_t dt)) + T * sky(v), the renderer's exact-single
          mode, as the sum of two Mitsuba renders of the same scene:
            (a) sun only (no envmap), volpath max_depth 2: camera ray -> one real scattering vertex -> next-event
                estimation to the (delta) sun with ratio-tracked T_sun. Depth 2 = one scattering event; the sun is a
                delta light, so nothing else contributes.
            (b) envmap only, volpath max_depth 1: the camera ray reaches the sky only if it is not scattered, so the
                expectation is T(camera ray) * sky(v); scattered paths die at the depth limit (no sky in-scattering).
          Both parts are also saved (<pair>_sun.npy, <pair>_bg.npy) so the validation can compare them separately.

Outputs (in --out, default reference/out, single mode in reference/out/single): <pair>.npy (linear float32 RGB,
H x W x 3, row 0 = top), <pair>.png (scene/tonemap.py with the pair's exposure), <pair>.json (spp, passes, seconds,
noise estimate, settings).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCENE_OUT = ROOT / "scene" / "out"
sys.path.insert(0, str(ROOT / "scene"))
from tonemap import save_png, tonemap_float  # noqa: E402

DEFAULT_RR_DEPTH = 256
LUM = np.array([0.2126, 0.7152, 0.0722])


# ----------------------------------------------------------------------------------------------- scene helpers
def load_meta():
    return json.loads((SCENE_OUT / "density.json").read_text()), json.loads((SCENE_OUT / "shots.json").read_text())


def load_density(meta):
    nx, ny, nz = meta["dims"]
    raw = np.fromfile(SCENE_OUT / "density.u8", np.uint8)
    assert raw.size == nx * ny * nz
    return raw.reshape(nz, ny, nx).astype(np.float32) / 255.0  # [z, y, x]: Mitsuba VolumeGrid order (x fastest)


def load_sky(preset_entry):
    side = json.loads((SCENE_OUT / preset_entry["sky_json"]).read_text())
    return np.fromfile(SCENE_OUT / preset_entry["sky_map"], "<f4").reshape(side["height"], side["width"], 3)


def camera_rays(S, W, H):
    """Unit view directions (H, W, 3) through pixel centres, shots.json camera model (row 0 = top)."""
    pos = np.array(S["position"], float); tgt = np.array(S["look_at"], float); up = np.array(S["up"], float)
    f = tgt - pos; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    t = np.tan(np.radians(S["fov_y_deg"]) / 2)
    i = (np.arange(W) + 0.5) / W * 2 - 1
    j = 1 - (np.arange(H) + 0.5) / H * 2
    I, J = np.meshgrid(i, j)
    d = f + (I * t * W / H)[..., None] * r + (J * t)[..., None] * u
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def sky_lookup(sky, d, convention="pixel"):
    """Bilinear lookup of the equirect map. 'pixel': scene/out convention (texel centres at (i+0.5)/N);
    'vertex': Mitsuba's internal convention (column j at u = j/W with periodic wrap, row i at v = i/(H-1))."""
    H, W, _ = sky.shape
    th = np.arccos(np.clip(d[..., 1], -1, 1))
    ph = np.mod(np.arctan2(d[..., 0], -d[..., 2]), 2 * np.pi)
    u = ph / (2 * np.pi); v = th / np.pi
    if convention == "pixel":
        x = u * W - 0.5; y = np.clip(v * H - 0.5, 0, H - 1)
    else:
        x = u * W; y = v * (H - 1)
    x0 = np.floor(x).astype(int); y0 = np.clip(np.floor(y).astype(int), 0, H - 2)
    fx = (x - x0)[..., None]; fy = np.clip(y - y0, 0, 1)[..., None]
    x0w = np.mod(x0, W); x1w = np.mod(x0 + 1, W)
    a = sky[y0, x0w] * (1 - fx) + sky[y0, x1w] * fx
    b = sky[y0 + 1, x0w] * (1 - fx) + sky[y0 + 1, x1w] * fx
    return a * (1 - fy) + b * fy


def crop_grid(d, bmin, bmax, vox, margin=2):
    """Crop the [z, y, x] grid to its occupied voxels plus `margin` zero voxels per side, returning the new box.
    Cell-centred trilinear sampling with clamp-to-edge then gives exactly the same field inside the new box, and the
    original field is 0 outside it."""
    occ = [np.nonzero(d.max(axis=tuple(a for a in range(3) if a != ax)))[0] for ax in range(3)]  # z, y, x
    lo = [max(int(o.min()) - margin, 0) for o in occ]
    hi = [min(int(o.max()) + 1 + margin, d.shape[ax]) for ax, o in enumerate(occ)]
    dc = np.ascontiguousarray(d[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    lo_xyz = np.array(lo[::-1], float); hi_xyz = np.array(hi[::-1], float)
    return dc, bmin + lo_xyz * vox, bmin + hi_xyz * vox


def pair_exposure(shots, pid):
    for group in ("reference", "heldback"):
        for p in shots["pairs"][group]:
            if p["id"] == pid:
                return p["exposure"]
    shot, preset = pid.split("_", 1)
    return shots["exposure_table"][shot][preset] if isinstance(shots["exposure_table"].get(shot), dict) else \
        shots["exposure_table"][pid]


def build_scene_dict(mi, shot, preset, spp, scale=1.0, medium=True, sun=True, sky=True, sky_bitmap=None,
                     max_depth=-1, rr_depth=DEFAULT_RR_DEPTH, density_scale=None, crop=True, accel=False):
    meta, shots = load_meta()
    S = shots["shots"][shot]
    T = mi.ScalarTransform4f
    scene = {
        "type": "scene",
        "integrator": {"type": "volpath", "max_depth": int(max_depth), "rr_depth": int(rr_depth)},
        "sensor": {
            "type": "perspective", "fov": float(S["fov_y_deg"]), "fov_axis": "y",
            "near_clip": 0.1, "far_clip": 1.0e6,
            "to_world": T().look_at(origin=S["position"], target=S["look_at"], up=S["up"]),
            "sampler": {"type": "independent", "sample_count": int(spp)},
            "film": {"type": "hdrfilm", "width": int(round(S["width"] * scale)),
                     "height": int(round(S["height"] * scale)), "rfilter": {"type": "box"}, "pixel_format": "rgb"},
        },
    }
    if medium:
        bmin = np.array(meta["box_min"], float); bmax = np.array(meta["box_max"], float)
        d = load_density(meta)
        if crop:  # exact: same field, smaller box (fewer null collisions); keeps 2 zero voxels on every face
            d, bmin, bmax = crop_grid(d, bmin, bmax, np.array(meta["voxel_size"], float))
        sz = bmax - bmin
        grid = mi.VolumeGrid(d[..., None])
        scene["cloud"] = {
            "type": "cube", "bsdf": {"type": "null"},
            "to_world": T().translate(((bmin + bmax) / 2).tolist()).scale((sz / 2).tolist()),
            "interior": {
                "type": "heterogeneous",
                "albedo": float(meta["albedo"]),
                "scale": float(meta["sigma_t_max"] if density_scale is None else density_scale),
                "sigma_t": {"type": "gridvolume", "grid": grid, "filter_type": "trilinear", "accel": bool(accel),
                            "to_world": T().translate(bmin.tolist()).scale(sz.tolist())},
                "phase": {"type": "blendphase", "weight": 0.2,  # (1 - w) * phase1 + w * phase2
                          "phase1": {"type": "hg", "g": 0.9}, "phase2": {"type": "hg", "g": -0.3}},
            },
        }
    P = shots["presets"][preset] if preset else None
    if sun:
        scene["sun"] = {"type": "directional", "direction": (-np.array(P["toward_sun"], float)).tolist(),
                        "irradiance": {"type": "rgb", "value": [float(x) for x in P["E_sun"]]}}
    if sky:
        img = sky_bitmap if sky_bitmap is not None else load_sky(P)
        scene["sky"] = {"type": "envmap", "bitmap": mi.Bitmap(np.ascontiguousarray(img, dtype=np.float32)),
                        "scale": 1.0}
    return scene


# ----------------------------------------------------------------------------------------------- progressive render
class Accum:
    """Running per-pixel sums over independent passes (pass k uses seed seed0 + k). Saved to an .npz so a render can
    be resumed / topped up with more passes later; the result is identical to rendering all passes in one go."""

    def __init__(self, path, settings):
        self.path, self.settings = Path(path), settings
        self.s1 = self.s2 = None; self.n = 0; self.seconds = 0.0
        if self.path.exists():
            z = np.load(self.path, allow_pickle=False)
            if json.loads(str(z["settings"])) == settings:
                self.s1, self.s2, self.n, self.seconds = z["s1"], z["s2"], int(z["n"]), float(z["seconds"])

    def add(self, img):
        if self.s1 is None:
            self.s1 = np.zeros_like(img); self.s2 = np.zeros_like(img)
        self.s1 += img; self.s2 += img * img; self.n += 1

    def save(self):
        np.savez(self.path, s1=self.s1, s2=self.s2, n=self.n, seconds=self.seconds,
                 settings=json.dumps(self.settings))

    def result(self):
        mean = self.s1 / self.n
        var = np.maximum(self.s2 / self.n - mean * mean, 0) / max(self.n - 1, 1)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def render_progressive(mi, scene_dict, spp, pass_spp, acc_path, settings, seed0=0, log=None):
    """Render (or resume) spp samples as independent passes of pass_spp (distinct seeds).
    Returns mean image, per-pixel standard error of the mean, total spp, total seconds."""
    acc = Accum(acc_path, dict(settings, pass_spp=pass_spp, seed0=seed0))
    n = max(1, int(np.ceil(spp / pass_spp)))
    if acc.n < n:
        scene = mi.load_dict(scene_dict)
        t0 = time.time(); t_save = t0; k0 = acc.n
        for k in range(acc.n, n):
            img = np.array(mi.render(scene, seed=seed0 + k, spp=pass_spp), dtype=np.float64)[..., :3]
            acc.add(img)
            now = time.time()
            if now - t_save > 60 or k == n - 1:
                acc.seconds += now - t_save; t_save = now; acc.save()
            if log and (k == k0 or (k + 1) % max(1, n // 8) == 0 or k == n - 1):
                el = now - t0
                log(f"    pass {k + 1}/{n}  {el:.0f}s  (eta {el / (k + 1 - k0) * (n - k - 1):.0f}s)")
    mean, se = acc.result()
    return mean, se, acc.n * pass_spp, acc.seconds


def noise_stats(mean, se, exposure):
    """Noise of the tone-mapped image in 8-bit levels, d = |tm(mean + se) - tm(mean)| (max over channels; se = 1-sigma
    standard error of the mean, estimated from the spread of the passes), over the cloud pixels, i.e. those with a
    relative standard error above 1e-3 (sky pixels vary only by the sub-pixel position of the box filter, ~1e-5)."""
    a = tonemap_float(mean, exposure) * 255
    b = tonemap_float(mean + se, exposure) * 255
    d = np.abs(b - a).max(-1)
    rel = (se @ LUM) / np.maximum(mean @ LUM, 1e-12)
    m = rel > 1e-3
    if not m.any():
        m = np.ones_like(m)
    return {"noise_pixels_fraction": float(m.mean()),
            "tm8_noise_median": float(np.median(d[m])), "tm8_noise_p90": float(np.percentile(d[m], 90)),
            "tm8_noise_p99": float(np.percentile(d[m], 99)), "tm8_noise_max": float(d.max()),
            "rel_se_median": float(np.median(rel[m])), "rel_se_p90": float(np.percentile(rel[m], 90))}


def render_pair(mi, pid, mode, spp, pass_spp, scale, rr_depth, outdir, variant, log=print):
    meta, shots = load_meta()
    shot, preset = pid.split("_", 1)
    ex = pair_exposure(shots, pid)
    info = {"pair": pid, "shot": shot, "preset": preset, "mode": mode, "exposure": ex, "variant": variant,
            "mitsuba": mi.__version__, "integrator": "volpath", "scale": scale,
            "density_sha256": meta["sha256"], "pixel_filter": "box", "grid": "cropped to occupied voxels + 2, "
            "software trilinear (accel=False)"}
    base = {"pair": pid, "scale": scale, "variant": variant, "density_sha256": meta["sha256"]}
    accdir = outdir / "accum"; accdir.mkdir(exist_ok=True)
    if mode == "full":
        sd = build_scene_dict(mi, shot, preset, pass_spp, scale, max_depth=-1, rr_depth=rr_depth)
        img, se, n, dt = render_progressive(mi, sd, spp, pass_spp, accdir / f"{pid}.npz",
                                            dict(base, mode="full", rr_depth=rr_depth), log=log)
        info.update({"max_depth": -1, "rr_depth": rr_depth, "spp": n, "seconds": round(dt, 1),
                     "ms_per_spp": round(1000 * dt / n, 2)})
    else:
        sd_a = build_scene_dict(mi, shot, preset, pass_spp, scale, sky=False, max_depth=2, rr_depth=1 << 30)
        sun_img, se_a, n_a, dt_a = render_progressive(mi, sd_a, spp, pass_spp, accdir / f"{pid}_sun.npz",
                                                      dict(base, mode="single_sun"), seed0=0, log=log)
        bg_spp = max(pass_spp, spp // 4)
        sd_b = build_scene_dict(mi, shot, preset, pass_spp, scale, sun=False, max_depth=1, rr_depth=1 << 30)
        bg_img, se_b, n_b, dt_b = render_progressive(mi, sd_b, bg_spp, pass_spp, accdir / f"{pid}_bg.npz",
                                                     dict(base, mode="single_bg"), seed0=1 << 20, log=log)
        img = sun_img + bg_img
        se = np.sqrt(se_a ** 2 + se_b ** 2)
        n, dt = n_a, dt_a + dt_b
        np.save(outdir / f"{pid}_sun.npy", sun_img)
        np.save(outdir / f"{pid}_bg.npy", bg_img)
        info.update({"parts": {"sun_single": {"file": f"{pid}_sun.npy", "max_depth": 2, "envmap": False, "spp": n_a,
                                              "seconds": round(dt_a, 1), **noise_stats(sun_img, se_a, ex)},
                               "background": {"file": f"{pid}_bg.npy", "max_depth": 1, "sun": False, "spp": n_b,
                                              "seconds": round(dt_b, 1)}},
                     "spp": n_a, "seconds": round(dt, 1)})
    info.update({"width": img.shape[1], "height": img.shape[0], "pass_spp": pass_spp,
                 "lum_p50": float(np.percentile(img @ LUM, 50)), "lum_p99": float(np.percentile(img @ LUM, 99)),
                 "lum_max": float((img @ LUM).max())})
    info.update(noise_stats(img, se, ex))
    np.save(outdir / f"{pid}.npy", img.astype(np.float32))
    np.save(outdir / f"{pid}_se.npy", se.astype(np.float32))
    save_png(outdir / f"{pid}.png", img, ex)
    (outdir / f"{pid}.json").write_text(json.dumps(info, indent=2) + "\n")
    log(f"  {pid} [{mode}] spp {info['spp']} {info['seconds']}s  cloud noise (8-bit levels, 1 sigma) median "
        f"{info['tm8_noise_median']:.2f} p90 {info['tm8_noise_p90']:.2f} p99 {info['tm8_noise_p99']:.2f}; "
        f"rel se median {info['rel_se_median']:.3f}")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--set", choices=["reference", "heldback", "all"])
    ap.add_argument("--mode", choices=["full", "single"], default="full")
    ap.add_argument("--spp", type=int, default=None)
    ap.add_argument("--pass-spp", type=int, default=64)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--rr-depth", type=int, default=DEFAULT_RR_DEPTH)
    ap.add_argument("--variant", default="metal_ad_rgb")
    ap.add_argument("--out", default=None)
    ap.add_argument("--restat", action="store_true", help="only recompute the noise statistics in the JSON files")
    args = ap.parse_args()

    import mitsuba as mi
    try:
        mi.set_variant(args.variant)
    except Exception as e:  # noqa: BLE001
        print(f"variant {args.variant} failed ({e}); falling back to llvm_ad_rgb")
        args.variant = "llvm_ad_rgb"; mi.set_variant(args.variant)

    _, shots = load_meta()
    ids = list(args.ids)
    if args.set:
        groups = ["reference", "heldback"] if args.set == "all" else [args.set]
        ids += [p["id"] for g in groups for p in shots["pairs"][g]]
    spp = args.spp or (SPP_FULL if args.mode == "full" else SPP_SINGLE)
    outdir = Path(args.out) if args.out else (HERE / "out" if args.mode == "full" else HERE / "out" / "single")
    outdir.mkdir(parents=True, exist_ok=True)
    if args.restat:
        for pid in ids:
            if not (outdir / f"{pid}.json").exists():
                continue
            info = json.loads((outdir / f"{pid}.json").read_text())
            info.update(noise_stats(np.load(outdir / f"{pid}.npy"), np.load(outdir / f"{pid}_se.npy").astype(np.float32),
                                    info["exposure"]))
            (outdir / f"{pid}.json").write_text(json.dumps(info, indent=2) + "\n")
            print(pid, {k: round(v, 3) for k, v in info.items() if k.startswith(("tm8", "rel_se"))})
        return
    t = time.time()
    for pid in ids:
        render_pair(mi, pid, args.mode, spp, args.pass_spp, args.scale, args.rr_depth, outdir, args.variant,
                    log=lambda s: print(s, flush=True))
    print(f"done {len(ids)} pairs in {time.time() - t:.0f}s", flush=True)


SPP_FULL = 16384
SPP_SINGLE = 4096

if __name__ == "__main__":
    main()
