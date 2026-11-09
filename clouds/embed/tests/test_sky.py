"""GPU sky (embed/web/sky.js + shaders/sky_*.glsl) and light table vs the Python model (scene/sky.py).

    uv run python embed/tests/test_sky.py [--workers 8] [--reps 10]

Headless Chromium (Metal ANGLE, renderer/harness_client.py GPU_ARGS) runs embed/tests/sky_test.html. Checks:
1. transmittance LUT: every node vs sky.TransmittanceTable.tau (ground flags must agree);
2. skyTau at grazing angles: around the horizon dip at camera-like altitudes vs TransmittanceTable.lookup;
3. sunDiskRadiance at points across the disk vs E_TOA exp(-tau) / Omega * limb;
4. sky maps at sun elevations [-4, -1, 0, 1, 3, 8, 20, 55] from 300 m and 1150 m vs sky.build_sky_map (H_OBS set),
   upper hemisphere, per-channel relative error where the radiance > 1e-3 * the map's max (target ~2 %);
5. light_table.json through the page's lightAt at off-grid elevations vs light_table.entry computed directly;
6. indicative pass timings (other GPU users may be active).
Exit status 1 when a check fails its threshold.
"""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CLOUDS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CLOUDS / "scene"))
sys.path.insert(0, str(CLOUDS / "renderer"))
sys.path.insert(0, str(CLOUDS / "embed" / "compile"))
import sky  # noqa: E402
import light_table as lt  # noqa: E402
from harness_client import GPU_ARGS, _Handler, _Server  # noqa: E402

ELEVS = [-4, -1, 0, 1, 3, 8, 20, 55]
CAM_HS = [300.0, 1150.0]
AZ = 200.0  # sky maps: any azimuth (the map is a function of the true sun vector)
OFF_GRID = [-9.5, -8.5, -6.5, -5.55, -3.05, -2.37, -1.85, -0.55, -0.25, 0.05, 1.23, 2.47, 7.95, 10.5, 45.3, 89.5]


def serve():
    import threading
    handler = type("H", (_Handler,), {"mounts": {"/embed/": (CLOUDS / "embed").resolve()}})
    handler.extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                              ".glsl": "text/plain", ".json": "application/json"}
    srv = _Server(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def f32(s):
    return np.frombuffer(base64.b64decode(s), dtype="<f4")


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


_TABLE = None


def _init():
    global _TABLE
    lt._init()
    _TABLE = lt.mv._TABLE


def py_sky(args):
    elev, cam_h = args
    s = sky.sun_dir(elev, AZ)
    sky.H_OBS = cam_h
    img, _, L_gnd, _ = sky.build_sky_map(s, lt.mv.E_disk(0.0, elev), _TABLE)
    return img, L_gnd


def py_light(elev):
    return lt.entry(elev)


def stats(rel):
    return {"median": float(np.median(rel)), "p95": float(np.percentile(rel, 95)), "max": float(rel.max()),
            "n": int(rel.size)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args(argv)
    from playwright.sync_api import sync_playwright

    pool = Pool(a.workers, initializer=_init)
    jobs = [(e, h) for h in CAM_HS for e in ELEVS]
    py_maps = pool.map_async(py_sky, jobs)
    py_lights = pool.map_async(py_light, OFF_GRID)
    _init()
    table = _TABLE
    report, ok = {}, True

    srv = serve()
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True, args=GPU_ARGS)
    try:
        page = browser.new_page(viewport={"width": 64, "height": 64})
        console = []
        page.on("console", lambda m: console.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
        page.goto(f"http://127.0.0.1:{srv.server_address[1]}/embed/tests/sky_test.html")
        page.wait_for_function("window.skyTestLoaded === true", timeout=30000)
        page.set_default_timeout(0)
        info = page.evaluate("() => window.skyTest.init()")
        report["gpu"] = info
        print(f"GPU: {info['renderer']}")

        # 1. LUT nodes
        lut = f32(page.evaluate("() => window.skyTest.readLut()")).reshape(info["nh"], info["nmu"], 4)[..., :3]
        flag_py, flag_gpu = table.tau >= 1e4, lut >= 1e4
        mism = int((flag_py != flag_gpu).any(axis=-1).sum())
        # relative error where tau > 1e-6 (rays leaving the top of the atmosphere have tau = 0 exactly) and the
        # transmittance can still show (tau < 50); absolute transmittance error over every unflagged node
        m = ~flag_py & (table.tau < 50.0) & (table.tau > 1e-6)
        rel = np.abs(lut[m] / table.tau[m] - 1.0)
        dT = np.abs(np.exp(-lut[~flag_py]) - np.exp(-table.tau[~flag_py]))
        report["lut"] = {"flag_mismatch_nodes": mism, "tau_rel": stats(rel), "transmittance_abs_max": float(dT.max())}
        print(f"LUT: ground-flag mismatches {mism}; tau rel err median {np.median(rel):.2e} p95 "
              f"{np.percentile(rel, 95):.2e} max {rel.max():.2e}; |dT| max {dT.max():.2e}")
        ok &= mism == 0 and rel.max() < 1e-3 and dT.max() < 1e-4

        # 2. grazing lookups
        hs = np.array([0.0, 20.0, 300.0, 1150.0, 2000.0, 4650.0, 10000.0])
        rows = []
        for h in hs:
            dip = -math.sqrt(max(0.0, 1.0 - (sky.R_EARTH / (sky.R_EARTH + h)) ** 2))
            mus = np.concatenate([dip + np.linspace(-0.003, 0.006, 91), np.linspace(-0.02, 0.05, 141)])
            rows += [(h, mu) for mu in mus]
        q = np.array(rows)
        inp = np.zeros((len(q), 4)); inp[:, :2] = q
        gpu = f32(page.evaluate("(a) => window.skyTest.probe(a)", {"mode": 0, "inputs": b64(inp)})).reshape(-1, 4)[:, :3]
        ref = table.lookup(q[:, 0], q[:, 1])
        Tg, Tr = np.exp(-np.minimum(gpu, 700.0)), np.exp(-np.minimum(ref, 700.0))
        vis = Tr > 1e-4
        relT = np.abs(Tg[vis] / Tr[vis] - 1.0)
        report["grazing"] = {"n": len(q), "transmittance_abs_max": float(np.abs(Tg - Tr).max()),
                             "transmittance_rel_where_T>1e-4": stats(relT)}
        print(f"grazing skyTau ({len(q)} lookups, h 0..10 km, mu around the horizon dip): |dT| max "
              f"{np.abs(Tg - Tr).max():.2e}; rel dT (T > 1e-4) median {np.median(relT):.2e} max {relT.max():.2e}")
        ok &= np.abs(Tg - Tr).max() < 1e-3 and relT.max() < 1e-2

        # 3. sun disk
        rad_s = math.radians(0.5332 / 2)
        dres = []
        for el in [0.2, 1.0, 5.0, 30.0]:
            for cam_h in CAM_HS:
                s = sky.sun_dir(el, 120.0)
                up = np.array([0.0, 1.0, 0.0])
                e1 = np.cross(s, up); e1 /= np.linalg.norm(e1); e2 = np.cross(e1, s)
                pts = []
                for rr in [0.0, 0.3, 0.6, 0.9, 0.99, 1.02]:
                    for ang in [0.0, 1.6, 3.3]:
                        a_ = rr * rad_s
                        d = math.cos(a_) * s + math.sin(a_) * (math.cos(ang) * e1 + math.sin(ang) * e2)
                        pts.append((d, rr))
                inp = np.array([[*p[0], cam_h] for p in pts])
                gpu = f32(page.evaluate("(a) => window.skyTest.probe(a)",
                                        {"mode": 1, "inputs": b64(inp), "sun": list(map(float, s))})).reshape(-1, 4)[:, :3]
                for (d, rr), g in zip(pts, gpu):
                    tau = table.lookup(np.array([cam_h]), np.array([d[1]]))[0]
                    limb = (1 - 0.6 * (1 - math.sqrt(max(0.0, 1 - rr * rr)))) / 0.8 if rr < 1 else 0.0
                    want = sky.E_TOA * np.exp(-np.minimum(tau, 700.0)) / lt.mv.SUN_OMEGA * limb if d[1] > 0 else 0 * tau
                    if rr in (0.99, 1.02):  # edge: compare membership only
                        dres.append(float((g.max() > 0) != (want.max() > 0)))
                    else:
                        dres.append(float(np.abs(g / np.maximum(want, 1e-30) - 1).max()))
        report["sun_disk_rel_max"] = max(dres)
        print(f"sun disk: max rel err {max(dres):.2e} (radius 0, .3, .6, .9; edge membership at .99 / 1.02)")
        ok &= max(dres) < 1e-3

        # 4. sky maps
        maps = py_maps.get()
        report["sky"] = []
        for (elev, cam_h), (img, L_gnd) in zip(jobs, maps):
            res = page.evaluate("(a) => window.skyTest.sky(a)",
                                {"s": list(map(float, sky.sun_dir(elev, AZ))), "camH": cam_h, "Lgnd": list(map(float, L_gnd))})
            g = f32(res["data"]).reshape(res["h"], res["w"], 4)[..., :3]
            hh = res["h"] // 2
            pu, gu = img[:hh].astype(np.float64), g[:hh].astype(np.float64)
            m = pu > 1e-3 * pu.max()
            rel = np.abs(gu[m] / pu[m] - 1.0)
            low = float(np.abs(g[hh:] - img[hh:]).max())
            st = stats(rel)
            report["sky"].append({"elev": elev, "cam_h": cam_h, **st, "lower_abs_max": low, "py_max": float(pu.max())})
            print(f"sky el {elev:5.1f} cam {cam_h:6.0f} m: rel err median {st['median']:.2e} p95 {st['p95']:.2e} "
                  f"max {st['max']:.2e} ({m.mean() * 100:.0f} % of upper texels > 1e-3 max); lower |d| {low:.1e}")
            ok &= st["p95"] < 0.02

        # 5. light table interpolation
        lights = page.evaluate("(e) => window.skyTest.lightAt(e)", OFF_GRID)
        report["light_table"] = []
        for elev, js, py in zip(OFF_GRID, lights, py_lights.get()):
            row = {"elev": elev}
            for k in ("E_sun_lut", "L_sky", "L_gnd"):
                a_, b_ = np.array(js[k], dtype=float).reshape(-1), np.asarray(py[k], dtype=float).reshape(-1)
                floor = max(1e-3 * b_.max(), 1e-9)
                mk = b_ > floor
                row[k] = float(np.abs(a_[mk] / b_[mk] - 1).max()) if mk.any() else 0.0
                # error relative to the entry's largest value (E_sun_lut: over all altitudes and channels), and for
                # E_sun_lut also to E_TOA: at the Earth's-shadow terminator E(y) rises from 0 over ~0.5 degrees, so
                # 0.1-degree linear steps are off by up to ~0.1 % of E_TOA there while the entry's own max is ~0
                row[k + "_of_max"] = float(np.abs(a_ - b_).max() / max(b_.max(), 1e-9))
            row["E_sun_lut_of_E_TOA"] = float(np.abs(np.array(js["E_sun_lut"]).ravel() - np.ravel(py["E_sun_lut"])).max()
                                              / sky.E_TOA)
            report["light_table"].append(row)
            print(f"light table el {elev:6.2f}: max rel err (where > 1e-3 max) / max abs err as a fraction of max: "
                  + ", ".join(f"{k} {row[k]:.1e} / {row[k + '_of_max']:.1e}" for k in ("E_sun_lut", "L_sky", "L_gnd"))
                  + f"; E_sun_lut abs err / E_TOA {row['E_sun_lut_of_E_TOA']:.1e}")
            ok &= row["E_sun_lut_of_E_TOA"] < 0.005 and row["L_sky_of_max"] < 0.02 and row["L_gnd_of_max"] < 0.02

        # 6. timings
        tm = page.evaluate("(a) => window.skyTest.timing(a)", {"reps": a.reps, "sizes": [[1024, 512], [512, 256]]})
        report["timing_ms"] = tm
        print("timing (indicative, median of %d, includes a 1-px readback): " % a.reps
              + ", ".join(f"{k} {v:.2f}" for k, v in tm.items()))
        errs = [c for c in console if "error" in c.lower()]
        if errs:
            print("\n".join(errs))
    finally:
        browser.close()
        pw.stop()
        srv.shutdown()
        pool.close()
    out = CLOUDS / "embed" / "tests" / "out"
    out.mkdir(exist_ok=True)
    (out / "test_sky.json").write_text(json.dumps(report, indent=1))
    print("ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
