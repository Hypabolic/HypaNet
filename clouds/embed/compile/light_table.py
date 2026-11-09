"""Per-sun-elevation lighting table for the embed page (DESIGN.md "Lighting").

    uv run python embed/compile/light_table.py [--out embed/web/data/light_table.json] [--workers 8]

All lighting the network sees depends only on the sun's elevation (azimuth only rotates the sky), so it is
tabulated once here from the same atmosphere as the references and the videos (scene/sky.py) and interpolated
linearly in elevation by the page (web/sky.js: lightAt; L_sky and L_gnd geometrically below -6 degrees, where the
twilight sky falls ~5x per 1-degree step). Per elevation:

- E_sun_lut: sun irradiance at 32 altitudes y = 0, 150, ..., 4650 m, averaged over the sun's disk exactly as
  video/make_video.py: E_disk (that function is called directly); 0 in Earth's shadow;
- L_sky: mean upper-hemisphere radiance seen from 2 km on a 128 x 64 map, as make_video.py: frame_light;
- L_gnd: albedo / pi * (E_sun(0 m) * max(s.y, 0) + E_sky) at 0 m, as sky.build_sky_map. E_sky (the cosine-weighted
  upper-hemisphere integral) comes from a 512 x 256 map instead of 1024 x 512: within 1.3e-4 of the full map
  (checked at -4, -0.5, 0.3, 3, 20 and 60 degrees) at a quarter of the cost.

Grid: 1 degree steps from -10 to 90, refined to 0.1 degree in [-6, 8] where the colour changes fastest.
Floats are rounded to 5 significant digits; values below 1e-12 are written as 0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CLOUDS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CLOUDS / "scene"))
sys.path.insert(0, str(CLOUDS / "video"))
import sky  # noqa: E402
import make_video as mv  # noqa: E402

OUT = CLOUDS / "embed" / "web" / "data" / "light_table.json"
AZ = 0.0  # any azimuth: the hemisphere means do not depend on it (up to the map's discretisation, ~1e-4)
ESKY_W, ESKY_H = 512, 256

FORMAT = ("light_table v1 (clouds/embed/compile/light_table.py). Per sun elevation elevations[i] (degrees, "
          "ascending): E_sun_lut[i] = 32 RGB triples flattened to 96 numbers [r0, g0, b0, r1, ...], sun irradiance "
          "at altitude y = lut_y0 + k * lut_dy (m), disk-averaged, 0 in Earth's shadow; L_sky[i] = RGB mean "
          "upper-hemisphere sky radiance seen from l_sky_altitude (m); L_gnd[i] = RGB ground radiance at 0 m, "
          "albedo/pi * (E_sun(0) max(s.y, 0) + E_sky). Interpolate linearly in elevation, clamp at the ends; "
          "exception: L_sky and L_gnd geometrically (linear in log) on the 1-degree segments below -6 degrees, where "
          "they fall ~5x per degree (linear: up to 28 % error mid-segment; geometric: 1.2 %). "
          "RGB = 680, 550, 440 nm, same units as the references (E_TOA = 20).")


def elevation_grid():
    coarse_lo = np.arange(-10.0, -6.0, 1.0)
    fine = np.round(np.arange(-60, 81) * 0.1, 1)
    coarse_hi = np.arange(9.0, 91.0, 1.0)
    return [float(x) for x in np.concatenate([coarse_lo, fine, coarse_hi])]


def _init():
    mv._init()  # the transmittance table, shared with E_disk / E_at


def L_sky_at(s, y):
    """make_video.frame_light: mean radiance of the upper hemisphere of a 128 x 64 map seen from altitude y."""
    sky.H_OBS = y
    d, theta = sky.equirect_dirs(128, 64)
    up = theta < math.pi / 2
    rad = sky.sky_radiance(d[up].reshape(-1, 3), s, mv._TABLE).reshape(up.sum(), 128, 3)
    w = np.sin(theta[up])[:, None, None]
    return (rad * w).sum(axis=(0, 1)) / (w.sum() * 128)


def L_gnd_at(s, E_ground, w=ESKY_W, h=ESKY_H):
    """sky.build_sky_map's L_gnd at 0 m (E_sky from a w x h map; same quadrature)."""
    sky.H_OBS = 0.0
    d, theta = sky.equirect_dirs(w, h)
    up = theta < math.pi / 2
    rad = sky.sky_radiance(d[up].reshape(-1, 3), s, mv._TABLE).reshape(up.sum(), w, 3)
    th = theta[up]
    dw = (np.sin(th) * np.cos(th))[:, None, None] * (math.pi / h) * (2 * math.pi / w)
    E_sky = (rad * dw).sum(axis=(0, 1))
    return sky.GROUND_ALBEDO / math.pi * (E_ground * max(s[1], 0.0) + E_sky)


def entry(elev):
    s = sky.sun_dir(elev, AZ)
    lut = [mv.E_disk(mv.LUT_Y0 + i * mv.LUT_DY, elev) for i in range(mv.LUT_N)]
    return {"E_sun_lut": np.array(lut), "L_sky": L_sky_at(s, mv.CLOUD_Y), "L_gnd": L_gnd_at(s, lut[0])}


def rnd(x):
    x = float(x)
    return 0.0 if abs(x) < 1e-12 else float(f"{x:.5g}")


def build(workers=8):
    els = elevation_grid()
    with Pool(workers, initializer=_init) as pool:
        rows = pool.map(entry, els, chunksize=4)
    return {
        "format": FORMAT,
        "version": 1,
        "lut_y0": mv.LUT_Y0, "lut_dy": mv.LUT_DY, "lut_n": mv.LUT_N, "l_sky_altitude": mv.CLOUD_Y,
        "elevations": els,
        "E_sun_lut": [[rnd(v) for v in r["E_sun_lut"].ravel()] for r in rows],
        "L_sky": [[rnd(v) for v in r["L_sky"]] for r in rows],
        "L_gnd": [[rnd(v) for v in r["L_gnd"]] for r in rows],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    table = build(a.workers)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, separators=(",", ":")))
    print(f"light table: {len(table['elevations'])} elevations, {out.stat().st_size / 1024:.0f} KiB -> {out} "
          f"({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
