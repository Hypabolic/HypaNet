"""Sun-motion videos of Fable's r1 network (the network is unchanged; only the light and camera move).

    uv run python video/make_video.py arc    [--frames 240] [--res 960x600] [--spp 24] [--stage all|scene|render|post]
    uv run python video/make_video.py sunset [--frames 300] ...

arc     the sun arcs across the southern sky from left (east) to right (west), passing behind the cloud
sunset  the sun sinks straight down past the horizon; the cloud stays lit after the ground is in shadow

For every frame this writes a scene directory the harness renderer reads (video/out/<name>/scene/): a shots.json with
one camera and one preset per frame, and a sky map per frame. The light comes from the same physical atmosphere as the
references (scene/sky.py: Rayleigh + Mie + ozone single scattering on a spherical planet), with three additions that
the fixed reference presets did not need:

- E_sun by altitude: sunlight reaching height y is 20 exp(-tau(y, sun elevation)) from the atmosphere's transmittance
  table, averaged over the sun's disk (0.533 degrees). The table is 0 for rays that hit the planet, so when the sun is
  below the horizon the ground and the cloud base fall into Earth's shadow before the tops do, and the light that still
  reaches the tops has crossed the lower atmosphere at grazing incidence (deep red). No refraction.
- a per-frame sky map seen from the camera's altitude (the references used one map per preset seen from 2 km), and
  L_sky (cloud sky light) from a coarse map at the cloud's altitude, 2 km;
- the sun disk (limb-darkened, its radiance per view elevation from the same table), dimmed by the cloud's
  transmittance along the view ray.

Post: a small lens glare (three Gaussians holding 1.2 % of the energy), auto-exposure that follows 3/4 of the change
in scene brightness (smoothed over time, so dusk still darkens), the SPEC tone map, then ffmpeg (MP4 and GIF).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CLOUDS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLOUDS / "scene"))
import sky  # noqa: E402
from tonemap import tonemap  # noqa: E402

NET = CLOUDS / "runs" / "r1" / "author" / "work" / "network.json"
OUT = CLOUDS / "video" / "out"
SUN_R = math.radians(0.5332 / 2)  # angular radius of the sun
SUN_OMEGA = 2 * math.pi * (1 - math.cos(SUN_R))
LUT_Y0, LUT_DY, LUT_N = 0.0, 150.0, 32  # E_sun by altitude, 0 .. 4650 m
CLOUD_Y = 2000.0


def look(pos, yaw_deg, pitch_deg, fov_y, res):
    """Camera at pos looking toward compass azimuth yaw (clockwise from north = -z), pitched up by pitch."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    d = [math.cos(p) * math.sin(y), math.sin(p), -math.cos(p) * math.cos(y)]
    return {"position": list(pos), "look_at": [pos[i] + 1000 * d[i] for i in range(3)], "up": [0.0, 1.0, 0.0],
            "fov_y_deg": fov_y, "width": res[0], "height": res[1]}


def arc_path(n):
    """South-facing camera 3.6 km north of the cloud; the sun goes from azimuth 140 (left) to 220 (right), rising
    from 5 to 26 degrees and back, so it passes behind the cloud's body (18-32 degrees up from the camera)."""
    t = np.linspace(-1.0, 1.0, n)
    return [(26.0 - 21.0 * x * x, 180.0 + 40.0 * x) for x in t]


def sunset_path(n):
    """The sun sinks from 30 to -5 degrees at azimuth 270 (west). Frames are packed more densely near the horizon
    (a speed ramp: the sun slows to about 1/4 speed between +3 and -4 degrees, where the colour changes)."""
    el = np.linspace(30.0, -5.0, 4000)
    dens = 1.0 + 3.0 * np.exp(-((el + 0.5) / 3.5) ** 2)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.abs(np.diff(el)))])
    target = np.linspace(0.0, cum[-1], n)
    return [(float(np.interp(c, cum, el)), 270.0) for c in target]


VIDEOS = {
    # camera 300 m up, 3.6 km north of the cloud, facing south and pitched up 18 degrees: horizon near the bottom
    "arc": {"camera": lambda res: look((0.0, 300.0, -3600.0), 180.0, 18.0, 50.0, res), "path": arc_path,
            "frames": 240, "fps": 30},
    # camera 20 m up, 4.8 km south-east of the cloud, facing azimuth 298: the sun sets 28 degrees left of centre,
    # the cloud stands 24 degrees right of centre (52 degrees from the sun, lit from the side and behind)
    "sunset": {"camera": lambda res: look((2955.0, 20.0, 3782.0), 298.0, 12.0, 54.0, res), "path": sunset_path,
               "frames": 300, "fps": 30},
}

_TABLE = None


def _init():
    global _TABLE
    _TABLE = sky.TransmittanceTable()


def E_at(y, elev_deg):
    """Sun irradiance (RGB) at altitude y for a point of the sun at elevation elev_deg (0 when Earth blocks it)."""
    tau = _TABLE.lookup(np.array([y], dtype=float), np.array([math.sin(math.radians(elev_deg))]))[0]
    return sky.E_TOA * np.exp(-np.minimum(tau, 700.0))


def E_disk(y, elev_deg, n=9):
    """E_sun at altitude y averaged over the sun's disk (chord-weighted rows across its vertical diameter)."""
    off = np.linspace(-1, 1, n + 2)[1:-1]
    w = np.sqrt(1 - off * off)
    rad = math.degrees(SUN_R)
    return sum(wi * E_at(y, elev_deg + oi * rad) for oi, wi in zip(off, w)) / w.sum()


def frame_light(args):
    """One preset: E_sun LUT, sun disk, sky map at the camera's altitude, L_sky at 2 km, L_gnd."""
    k, elev, az, cam_y, scene_dir = args
    s = sky.sun_dir(elev, az)
    lut = [E_disk(LUT_Y0 + i * LUT_DY, elev) for i in range(LUT_N)]
    E_ground = lut[0]
    sky.H_OBS = cam_y
    img, _, L_gnd, _ = sky.build_sky_map(s, E_ground, _TABLE)
    img.astype(np.float32).tofile(scene_dir / f"sky_f{k:03d}.f32")
    # sky light on the cloud: mean radiance of the upper hemisphere seen from 2 km (a coarse 128 x 64 map)
    sky.H_OBS = CLOUD_Y
    d, theta = sky.equirect_dirs(128, 64)
    up = theta < math.pi / 2
    rad = sky.sky_radiance(d[up].reshape(-1, 3), s, _TABLE).reshape(up.sum(), 128, 3)
    w = np.sin(theta[up])[:, None, None]
    L_sky = (rad * w).sum(axis=(0, 1)) / (w.sum() * 128)
    lo, hi = math.radians(elev) - 1.2 * SUN_R, math.radians(elev) + 1.2 * SUN_R
    disk = [list(E_at(cam_y, math.degrees(lo + (hi - lo) * i / 15)) / SUN_OMEGA) for i in range(16)]
    return f"f{k:03d}", {
        "elevation_deg": elev, "azimuth_deg": az, "toward_sun": [float(x) for x in s],
        "E_sun": [float(x) for x in lut[int(CLOUD_Y / LUT_DY)]], "L_sky": [float(x) for x in L_sky],
        "L_gnd": [float(x) for x in L_gnd], "sky_map": f"sky_f{k:03d}.f32",
        "E_sun_lut": {"y0": LUT_Y0, "dy": LUT_DY, "values": [[float(x) for x in v] for v in lut]},
        "sun_disk": {"cos_radius": math.cos(SUN_R), "el_lo": lo, "el_hi": hi,
                     "radiance": [[float(x) for x in v] for v in disk]},
    }


def build_scene(name, cfg, n, res):
    scene = OUT / name / "scene"
    if scene.exists():
        shutil.rmtree(scene)
    scene.mkdir(parents=True)
    src = CLOUDS / "scene" / "out"
    for f in ("density.json", "density.u8"):
        shutil.copy(src / f, scene / f)
    base = json.loads((src / "shots.json").read_text())
    cam = cfg["camera"](res)
    path = cfg["path"](n)
    with Pool(8, initializer=_init) as pool:
        presets = dict(pool.map(frame_light, [(k, e, a, cam["position"][1], scene) for k, (e, a) in enumerate(path)]))
    shots = {"version": 1, "units": base["units"], "resolution": list(res), "camera_model": base["camera_model"],
             "medium": base["medium"], "sun_model": base["sun_model"], "shots": {"cam": cam}, "presets": presets,
             "pairs": {}, "exposure_table": {f"cam_{p}": 1.0 for p in presets},
             "note": f"video '{name}' (clouds/video/make_video.py); per-frame presets with E_sun_lut and sun_disk"}
    (scene / "shots.json").write_text(json.dumps(shots))
    print(f"scene: {n} frames -> {scene}")


def render(name, n, res, spp):
    scene = OUT / name / "scene"
    out = OUT / name / "linear"
    out.mkdir(parents=True, exist_ok=True)
    todo = [f"cam_f{k:03d}" for k in range(n) if not (out / f"cam_f{k:03d}_network.npy").exists()]
    for i in range(0, len(todo), 40):  # a fresh harness every 40 frames
        cmd = [sys.executable, str(CLOUDS / "renderer" / "render.py"), str(NET), "--scene-dir", str(scene),
               "--pair", *todo[i:i + 40], "--frames", str(spp), "--res", f"{res[0]}x{res[1]}", "--out", str(out) + "/"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        print(f"rendered {min(i + 40, len(todo))}/{len(todo)}", flush=True)


def gauss_fft(shape, sigma):
    h, w = shape
    fy = np.fft.fftfreq(2 * h)[:, None]
    fx = np.fft.rfftfreq(2 * w)[None, :]
    return np.exp(-2 * (math.pi * sigma) ** 2 * (fx * fx + fy * fy))


def post(name, n, fps):
    lin_dir, png_dir = OUT / name / "linear", OUT / name / "frames"
    png_dir.mkdir(parents=True, exist_ok=True)
    frames = [np.load(lin_dir / f"cam_f{k:03d}_network.npy").astype(np.float64) for k in range(n)]
    h, w, _ = frames[0].shape
    glare = [(0.008, 1.5), (0.003, 10.0), (0.001, 60.0)]  # (energy fraction, sigma in pixels)
    kern = [(a, gauss_fft((h, w), s)) for a, s in glare]
    keep = 1.0 - sum(a for a, _ in glare)
    # auto-exposure from the log-mean luminance below the 99.5th percentile (the sun disk excluded), following 3/4
    # of the change, smoothed over +-0.4 s; calibrated so a golden-hour frame lands near the references' exposure
    lum = np.array([0.2126, 0.7152, 0.0722])
    keys = []
    for f in frames:
        y = f @ lum
        y = y[y < np.percentile(y, 99.5)]
        keys.append(math.exp(np.log(y + 1e-6).mean()))
    log_e = np.log(0.5) - 0.75 * (np.log(np.array(keys)) - math.log(0.12))
    sig = 0.4 * fps
    ker = np.exp(-0.5 * (np.arange(-3 * sig, 3 * sig + 1) / sig) ** 2)
    pad = np.pad(log_e, len(ker) // 2, mode="edge")
    log_e = np.convolve(pad, ker / ker.sum(), mode="valid")
    from PIL import Image
    for k, f in enumerate(frames):
        F = np.zeros_like(f)
        for c in range(3):
            spec = np.fft.rfft2(f[..., c], s=(2 * h, 2 * w))
            F[..., c] = keep * f[..., c] + np.fft.irfft2(spec * sum(a * g for a, g in kern), s=(2 * h, 2 * w))[:h, :w]
        Image.fromarray(tonemap(F, math.exp(log_e[k])), "RGB").save(png_dir / f"{k:03d}.png")
    mp4, gif = OUT / name / f"{name}.mp4", OUT / name / f"{name}.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(png_dir / "%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", str(mp4)], check=True)
    vf = ("fps=15,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];"
          "[b][p]paletteuse=dither=sierra2_4a")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf", vf, "-loop", "0", str(gif)], check=True)
    print(f"post: exposure {math.exp(log_e.min()):.3g}..{math.exp(log_e.max()):.3g}; {mp4}; {gif}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=sorted(VIDEOS))
    ap.add_argument("--frames", type=int)
    ap.add_argument("--res", default="960x600")
    ap.add_argument("--spp", type=int, default=24, help="progressive frames per video frame")
    ap.add_argument("--stage", choices=["all", "scene", "render", "post"], default="all")
    a = ap.parse_args(argv)
    cfg = VIDEOS[a.name]
    n = a.frames or cfg["frames"]
    res = [int(x) for x in a.res.split("x")]
    if a.stage in ("all", "scene"):
        build_scene(a.name, cfg, n, res)
    if a.stage in ("all", "render"):
        render(a.name, n, res, a.spp)
    if a.stage in ("all", "post"):
        post(a.name, n, cfg["fps"])


if __name__ == "__main__":
    main()
