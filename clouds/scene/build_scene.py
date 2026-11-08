"""Build the HypaNet Clouds scene (SPEC section 3) into scene/out/.

    uv run python scene/build_scene.py            # everything (about 1 minute)
    uv run python scene/build_scene.py --no-sky   # density + shots only (reuses existing sky stats if present)

Outputs (all deterministic):
    density.u8, density.json          cloud density grid (uint8, x fastest, then y, then z) + metadata
    density_preview.png               quick orthographic preview (not a render)
    sky_<preset>.f32, sky_<preset>.json, sky_<preset>.png   equirect sky maps (float32 RGB) + sidecar + preview
    shots.json                        sun presets (E_sun, L_sky, L_gnd), camera shots, reference / held-back pairs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import density as dn  # noqa: E402
import sky  # noqa: E402
from tonemap import tonemap  # noqa: E402

OUT = HERE / "out"

SIGMA_T_MAX = 0.04
ALBEDO = 0.999
WIDTH, HEIGHT = 640, 400

SUN_AZIMUTH = 250.0  # degrees clockwise from north (north = -z, east = +x); same for every preset
PRESETS = {  # name: elevation (deg)
    "noon": 55.0,
    "afternoon": 25.0,
    "golden": 8.0,
    "sunset": 2.0,
}

CLOUD_CENTRE = np.array([0.0, 1900.0, 0.0])


def _horiz(az_deg):
    a = np.radians(az_deg)
    return np.array([np.sin(a), 0.0, -np.cos(a)])


def make_shots():
    """Camera shots. Azimuths are relative to the sun azimuth so the lighting geometry holds for every preset."""
    shots = {}

    def add(name, pos, target, fov, desc):
        shots[name] = {
            "position": [round(float(v), 3) for v in pos],
            "look_at": [round(float(v), 3) for v in target],
            "up": [0.0, 1.0, 0.0],
            "fov_y_deg": fov,
            "width": WIDTH,
            "height": HEIGHT,
            "description": desc,
        }

    c = CLOUD_CENTRE
    # 1. side view, sun behind the camera
    add("front", c * [1, 0, 1] + 3300.0 * _horiz(SUN_AZIMUTH) + [0, 1150.0, 0], c + [0, 30.0, 0], 30.0,
        "side view, camera on the sun side (sun behind the camera), from below cloud-base height looking slightly up")
    # 2. side view, sun at 90 degrees (terminator across the cloud)
    add("side", c * [1, 0, 1] + 3300.0 * _horiz(SUN_AZIMUTH + 90.0) + [0, 1150.0, 0], c + [0, 30.0, 0], 30.0,
        "side view, sun from the left at 90 degrees azimuth (terminator)")
    # 3. backlit, looking toward the sun (silver lining)
    add("backlit", c * [1, 0, 1] + 3300.0 * _horiz(SUN_AZIMUTH + 180.0) + [0, 1200.0, 0], c + [0, 50.0, 0], 32.0,
        "backlit: camera on the far side looking toward the sun, slightly upward (silver lining)")
    # 4. below the base
    add("below", c * [1, 0, 1] + 950.0 * _horiz(SUN_AZIMUTH + 120.0) + [0, 600.0, 0], np.array([0.0, 1700.0, 0.0]), 65.0,
        "from below the base near its edge, looking up into the flat base and the towers")
    # 5. distant, whole cloud low in the sky, horizon at the bottom
    pos = c * [1, 0, 1] + 14000.0 * _horiz(SUN_AZIMUTH + 60.0) + [0, 400.0, 0]
    look_dir = -_horiz(SUN_AZIMUTH + 60.0) * np.cos(np.radians(5.0)) + np.array([0, np.sin(np.radians(5.0)), 0])
    add("distant", pos, pos + 1000.0 * look_dir, 12.0,
        "distant view from 14 km near the ground, cloud low in the sky, horizon just above the bottom edge; sun behind-left")
    # 6. three-quarter backlit close-up
    add("quarter", c * [1, 0, 1] + 2800.0 * _horiz(SUN_AZIMUTH + 135.0) + [0, 2000.0, 0], c, 36.0,
        "three-quarter backlit close-up from cloud-middle altitude (sun ahead-right of the camera)")
    return shots


# exposure per (shot, preset) pair: scene/choose_exposure.py (cloud p90 luminance -> ACES input 1.0, rounded down to
# a 1/3 stop) on 160x100, 512 spp Mitsuba renders of the full scene; see NOTES.md
EXPOSURE = {
    ("front", "noon"): 0.315, ("front", "afternoon"): 0.315, ("front", "golden"): 0.5, ("front", "sunset"): 2.0,
    ("side", "noon"): 0.315, ("side", "afternoon"): 0.3969, ("side", "golden"): 0.7937, ("side", "sunset"): 2.5198,
    ("backlit", "noon"): 0.3969, ("backlit", "afternoon"): 0.1984, ("backlit", "golden"): 0.315, ("backlit", "sunset"): 1.5874,
    ("below", "noon"): 0.25, ("below", "afternoon"): 0.3969, ("below", "golden"): 1.0, ("below", "sunset"): 4.0,
    ("distant", "noon"): 0.315, ("distant", "afternoon"): 0.3969, ("distant", "golden"): 0.63, ("distant", "sunset"): 2.5198,
    ("quarter", "noon"): 0.3969, ("quarter", "afternoon"): 0.3969, ("quarter", "golden"): 0.63, ("quarter", "sunset"): 2.5198,
}

REFERENCE_PAIRS = [
    ("front", "noon"), ("front", "golden"), ("front", "sunset"),
    ("side", "noon"), ("side", "afternoon"), ("side", "sunset"),
    ("backlit", "afternoon"), ("backlit", "golden"), ("backlit", "sunset"),
    ("below", "noon"), ("below", "golden"),
    ("distant", "sunset"),
]
HELDBACK_PAIRS = [
    ("front", "afternoon"), ("side", "golden"), ("backlit", "noon"),
    ("below", "sunset"), ("distant", "golden"), ("quarter", "afternoon"),
]


def build_density_files():
    t = time.time()
    d = dn.build_density(verbose=True)
    q = dn.quantise(d)  # (nz, ny, nx) -> C order = x fastest, then y, then z
    raw = q.tobytes(order="C")
    (OUT / "density.u8").write_bytes(raw)
    occ = q > 0
    ys = np.where(occ.any(axis=(0, 2)))[0]
    xs = np.where(occ.any(axis=(0, 1)))[0]
    zs = np.where(occ.any(axis=(1, 2)))[0]
    h = (dn.BOX_MAX - dn.BOX_MIN) / np.array(dn.DIMS)
    meta = {
        "file": "density.u8",
        "dims": list(dn.DIMS),
        "dims_order": "[nx, ny, nz]",
        "layout": "uint8, x fastest, then y, then z: index = x + nx * (y + ny * z); value d = byte / 255",
        "box_min": dn.BOX_MIN.tolist(),
        "box_max": dn.BOX_MAX.tolist(),
        "voxel_size": [round(float(v), 6) for v in h],
        "cell_centred": "voxel (i, j, k) is centred at box_min + (i + 0.5, j + 0.5, k + 0.5) * voxel_size "
                        "(GL texture convention: texcoord = (p - box_min) / (box_max - box_min))",
        "sigma_t_max": SIGMA_T_MAX,
        "albedo": ALBEDO,
        "phase": "0.8 * HG(mu; 0.9) + 0.2 * HG(mu; -0.3), mu = cos(light travel dir, scattered dir)",
        "seed": dn.SEED,
        "base_y": dn.BASE_Y,
        "occupied_bounds_m": {
            "x": [float(dn.BOX_MIN[0] + xs[0] * h[0]), float(dn.BOX_MIN[0] + (xs[-1] + 1) * h[0])],
            "y": [float(dn.BOX_MIN[1] + ys[0] * h[1]), float(dn.BOX_MIN[1] + (ys[-1] + 1) * h[1])],
            "z": [float(dn.BOX_MIN[2] + zs[0] * h[2]), float(dn.BOX_MIN[2] + (zs[-1] + 1) * h[2])],
        },
        "occupied_fraction": float(np.mean(q > 0)),
        "max_byte": int(q.max()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    (OUT / "density.json").write_text(json.dumps(meta, indent=2) + "\n")
    _density_preview(q.astype(np.float32) / 255.0, h)
    print(f"density: {time.time() - t:.1f}s, occupied {meta['occupied_bounds_m']}")
    return meta


def _density_preview(d, h):
    """Orthographic optical-depth silhouettes (front along z, side along x, top along y). Not a render."""
    from PIL import Image

    sig = d * SIGMA_T_MAX
    a = 1.0 - np.exp(-sig.sum(axis=0) * h[2])  # (ny, nx), view along z
    b = 1.0 - np.exp(-sig.sum(axis=2) * h[0])  # (nz, ny), view along x
    c = 1.0 - np.exp(-sig.sum(axis=1) * h[1])  # (nz, nx), view down y
    a = a[::-1]
    b = b.T[::-1]
    pad = np.zeros((a.shape[0], 8))
    row1 = np.concatenate([a, pad, b], axis=1)
    row2 = np.concatenate([c, np.zeros((c.shape[0], 8 + b.shape[1]))], axis=1)
    img = np.concatenate([row1, np.zeros((8, row1.shape[1])), row2], axis=0)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(OUT / "density_preview.png")


def build_presets(do_sky=True):
    presets = {}
    table = sky.TransmittanceTable() if do_sky else None
    for name, elev in PRESETS.items():
        t = time.time()
        s = sky.sun_dir(elev, SUN_AZIMUTH)
        E = sky.E_sun(elev)
        jpath = OUT / f"sky_{name}.json"
        if do_sky:
            img, L_sky, L_gnd, E_sky = sky.build_sky_map(s, E, table)
            raw = img.astype("<f4").tobytes(order="C")
            (OUT / f"sky_{name}.f32").write_bytes(raw)
            side = {
                "file": f"sky_{name}.f32",
                "preset": name,
                "width": sky.SKY_W,
                "height": sky.SKY_H,
                "channels": 3,
                "dtype": "float32 little-endian, RGB interleaved, row-major, row 0 = top (zenith)",
                "units": "radiance in the scene's linear RGB units (top-of-atmosphere solar irradiance = 20)",
                "mapping": "u = (col + 0.5) / W, v = (row + 0.5) / H, phi = 2 pi u, theta = pi v; "
                           "direction = (sin theta sin phi, cos theta, -sin theta cos phi) (y up). "
                           "Inverse: theta = acos(d.y), phi = atan2(d.x, -d.z) mod 2 pi. Same as Mitsuba 3 envmap "
                           "with identity to_world (lookup with wi = -d).",
                "lower_hemisphere": "rows 256..511 (theta > 90 deg) hold the constant ground radiance L_gnd",
                "toward_sun": s.tolist(),
                "E_sun": E.tolist(),
                "L_sky": L_sky.tolist(),
                "L_gnd": L_gnd.tolist(),
                "E_sky": E_sky.tolist(),
                "observer_altitude_m": sky.H_OBS,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            jpath.write_text(json.dumps(side, indent=2) + "\n")
            _sky_preview(name, img)
        else:
            side = json.loads(jpath.read_text())
            L_sky, L_gnd, E_sky = (np.array(side[k]) for k in ("L_sky", "L_gnd", "E_sky"))
        presets[name] = {
            "elevation_deg": elev,
            "azimuth_deg": SUN_AZIMUTH,
            "toward_sun": [round(float(v), 9) for v in s],
            "air_mass": float(sky.kasten_young(elev)),
            "T_atm": [float(v) for v in sky.T_atm(elev)],
            "E_sun": [float(v) for v in E],
            "L_sky": [float(v) for v in L_sky],
            "L_gnd": [float(v) for v in L_gnd],
            "E_sky": [float(v) for v in E_sky],
            "sky_map": f"sky_{name}.f32",
            "sky_json": f"sky_{name}.json",
        }
        print(f"preset {name}: E_sun {np.round(E, 3)} L_sky {np.round(L_sky, 4)} L_gnd {np.round(L_gnd, 4)} ({time.time() - t:.1f}s)")
    return presets


def _sky_preview(name, img):
    from PIL import Image

    ex = {"noon": 0.25, "afternoon": 0.25, "golden": 0.35, "sunset": 1.0}[name]
    small = img.reshape(256, 2, 512, 2, 3).mean(axis=(1, 3))
    Image.fromarray(tonemap(small, ex)).save(OUT / f"sky_{name}.png")


def build_shots(presets):
    shots = make_shots()

    def pair(shot, preset):
        return {"id": f"{shot}_{preset}", "shot": shot, "preset": preset, "exposure": EXPOSURE[(shot, preset)]}

    doc = {
        "version": 1,
        "units": "metres; y is up; world frame: north = -z, east = +x",
        "resolution": [WIDTH, HEIGHT],
        "camera_model": "pinhole; vertical field of view fov_y_deg; look_at with up = +y; aspect = width / height",
        "medium": {"sigma_t_max": SIGMA_T_MAX, "albedo": ALBEDO,
                   "phase": "0.8 * HG(mu; 0.9) + 0.2 * HG(mu; -0.3)"},
        "sun_model": {
            "E_sun": "20 * T_atm(h), T_atm(h) = exp(-(beta_R * 8000 + beta_M_ext * 1200 + tau_O3) * m(h))",
            "air_mass": "Kasten-Young: m(h) = 1 / (sin h + 0.50572 (h_deg + 6.07995)^-1.6364)",
            "beta_R": sky.BETA_R.tolist(),
            "beta_M_ext": sky.BETA_M_EXT,
            "tau_O3": sky.TAU_O3.tolist(),
            "wavelengths_nm": [680, 550, 440],
            "toward_sun": "(cos e sin a, sin e, -cos e cos a), e = elevation, a = azimuth clockwise from north (-z)",
        },
        "presets": presets,
        "shots": shots,
        "pairs": {
            "reference": [pair(*p) for p in REFERENCE_PAIRS],
            "heldback": [pair(*p) for p in HELDBACK_PAIRS],
        },
        "exposure_table": {f"{s}_{p}": e for (s, p), e in EXPOSURE.items()},
        "exposure_note": "exposure is per (shot, preset) pair (exposure_table covers every combination; the pairs "
                         "repeat it); chosen by scene/choose_exposure.py so the cloud's 90th-percentile luminance "
                         "maps to ACES input ~1.0; tone map in scene/tonemap.py",
    }
    (OUT / "shots.json").write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sky", action="store_true")
    ap.add_argument("--no-density", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.no_density:
        build_density_files()
    presets = build_presets(do_sky=not args.no_sky)
    build_shots(presets)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
