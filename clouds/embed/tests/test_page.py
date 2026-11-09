"""The embeddable page (embed/web/) in headless Chromium on the GPU (renderer/harness_client.py GPU_ARGS).

    uv run python embed/tests/test_page.py [all|load|shots|drag|timing|score] [--out embed/tests/out/page]

load    loads index.html: no console errors, a first frame, load timings, download size (every request is local)
shots   screenshots of every view at several sun positions (interactive path: auto-exposure, glare, accumulation)
drag    simulated pointer drags on the canvas and the compass move the sun and relight (pixels change, caption
        follows); arrow keys; the play button; the page goes idle once a still image has converged
timing  window.cloudsHarness at 1280x800 and 1920x1080 internal pixels (one sample per pixel, every frame waited
        for), sun-bake ms (full and drag grids), static-bake ms, load-to-first-frame ms
score   front_noon and backlit_golden at 640x400, 16 frames, the reference presets' lighting (constant E_sun, L_sky,
        L_gnd, sky map from 2 km, no sun disk) through the page's harness hook, scored with scripts/score.py's score()
        against clouds/reference/out (and compared pixel by pixel with the research harness's compiled render and
        the interpreter render cached by embed/tests/eval_compiled.py, when present)

Writes embed/tests/out/page/<cmd>.json and PNGs. Exit status 1 when a check fails.
"""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

CLOUDS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CLOUDS / "renderer"))
sys.path.insert(0, str(CLOUDS / "scripts"))
from harness_client import GPU_ARGS, _Handler, _Server  # noqa: E402
from score import score  # noqa: E402

WEB = CLOUDS / "embed" / "web"
REF = CLOUDS / "reference" / "out"
SCORE_PAIRS = ["front_noon", "backlit_golden"]   # reference split only
FAILS: list[str] = []


def check(ok, msg):
    print(("ok    " if ok else "FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


def serve():
    handler = type("H", (_Handler,), {"mounts": {"/web/": WEB.resolve(), "/tests/": (CLOUDS / "embed" / "tests").resolve()}})
    handler.extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                              ".glsl": "text/plain", ".json": "application/json", ".gz": "application/gzip"}
    srv = _Server(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class Page:
    def __init__(self, pw, port, query="", viewport=(1280, 800), dpr=1.0):
        self.browser = pw.chromium.launch(headless=True, args=GPU_ARGS)
        self.ctx = self.browser.new_context(viewport={"width": viewport[0], "height": viewport[1]}, device_scale_factor=dpr)
        self.page = self.ctx.new_page()
        self.console, self.requests = [], []
        self.page.on("console", lambda m: self.console.append(f"[{m.type}] {m.text}"))
        self.page.on("pageerror", lambda e: self.console.append(f"[pageerror] {e}"))
        self.page.on("requestfinished", self._req)
        self.t0 = time.perf_counter()
        self.page.goto(f"http://127.0.0.1:{port}/web/index.html{query}")
        self.page.wait_for_function("window.cloudsHarness && (window.cloudsHarness.error || window.cloudsHarness.state().ready)",
                                    timeout=60000)
        err = self.page.evaluate("window.cloudsHarness.error || null")
        if err:
            raise RuntimeError(f"page failed to load: {err}\n" + "\n".join(self.console))
        self.page.set_default_timeout(120000)

    def _req(self, req):
        try:
            resp = req.response()
            size = len(resp.body()) if resp else 0
        except Exception:
            size = 0
        self.requests.append((req.url, size))

    def ev(self, fn, arg=None):
        return self.page.evaluate(fn, arg)

    def wait_idle(self, timeout=30.0):
        t = time.time()
        while time.time() - t < timeout:
            s = self.ev("window.cloudsHarness.state()")
            if s["idle"]:
                return True
            time.sleep(0.1)
        return False

    def close(self):
        self.browser.close()


def errors(p):
    return [c for c in p.console if c.startswith("[error]") or c.startswith("[pageerror]") or "WebGL" in c or "GL_INVALID" in c]


def render(p, **kw):
    js = """async (a) => { const r = await window.cloudsHarness.render(a);
      const u8 = new Uint8Array(r.linear.buffer); let s = '';
      for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
      return { width: r.width, height: r.height, timing: r.timing, linear: btoa(s) }; }"""
    r = p.ev(js, kw)
    lin = np.frombuffer(base64.b64decode(r["linear"]), dtype="<f4").reshape(r["height"], r["width"], 3).copy()
    return lin, r["timing"]


def png(p, path):
    p.page.screenshot(path=str(path))


def img_stats(path):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float64)
    return a


# ---------------------------------------------------------------------------------------------------------------------
def cmd_load(pw, port, out):
    p = Page(pw, port, "?harness=1")
    load_s = time.perf_counter() - p.t0
    ok_frame = p.wait_idle(40)
    tm = p.ev("window.cloudsHarness.timing()")
    png(p, out / "load.png")
    a = img_stats(out / "load.png")
    total = sum(s for _, s in p.requests)
    local = all(u.startswith(f"http://127.0.0.1:{port}/") for u, _ in p.requests)
    check(not errors(p), f"load: no console errors ({len(errors(p))}: {errors(p)[:3]})")
    check(local, f"load: all {len(p.requests)} requests local")
    check(ok_frame, "load: the page converges and goes idle")
    check(a.std() > 5 and 10 < a.mean() < 245, f"load: screenshot is not blank (mean {a.mean():.1f}, std {a.std():.1f})")
    res = {"wall_to_ready_s": load_s, "timing_load": tm["load"], "bakes": tm["bakes"], "gl": tm["gl"],
           "requests": sorted(p.requests, key=lambda r: -r[1]), "download_bytes": total, "console": p.console,
           "state": tm["state"]}
    p.close()
    return res


SHOTS = {
    "front": [(24, 250), (55, 250), (8, 250), (2, 250), (-2, 250), (35, 160)],
    "backlit": [(8, 250), (25, 250), (3, 250), (40, 300), (0.5, 250)],
    "arc": [(22, 200), (15, 172), (5, 140), (26, 180), (40, 90)],
    "sunset": [(6, 270), (1.5, 270), (-1, 270), (-4, 270), (20, 300)],
}


def cmd_shots(pw, port, out, views=None):
    p = Page(pw, port, "?harness=1", viewport=(960, 600))
    res = {}
    for view, suns in SHOTS.items():
        if views and view not in views:
            continue
        p.ev("(v) => window.cloudsHarness.controller.setView(v)", view)
        for el, az in suns:
            p.ev("([e, a]) => window.cloudsHarness.controller.setSun(e, a)", [el, az])
            time.sleep(0.2)
            idle = p.wait_idle(40)
            time.sleep(1.0)
            f = out / f"shot_{view}_{el:g}_{az:g}.png"
            png(p, f)
            a = img_stats(f)
            res[f.name] = {"mean": a.mean(), "std": a.std(), "idle": idle,
                           "frac_white": float((a.min(-1) >= 254).mean()), "frac_black": float((a.max(-1) <= 1).mean())}
            check(idle and a.std() > 3, f"shots: {f.name} (mean {a.mean():.0f}, std {a.std():.0f}, white {res[f.name]['frac_white']:.3f})")
    check(not errors(p), f"shots: no console errors {errors(p)[:3]}")
    p.close()
    return res


def cmd_drag(pw, port, out):
    p = Page(pw, port, "?harness=1&view=front&el=20&az=250", viewport=(960, 600))
    p.wait_idle(40)
    png(p, out / "drag_0.png")
    s0 = p.ev("window.cloudsHarness.state()")
    box = p.page.locator("#clouds").bounding_box()
    m = p.page.mouse
    # drag across the upper part of the frame: the sun follows the pointer
    m.move(box["x"] + 200, box["y"] + 150)
    m.down()
    frames = []
    for k in range(30):
        m.move(box["x"] + 200 + 18 * k, box["y"] + 150 + 4 * k)
        time.sleep(1 / 60)
        if k in (10, 29):
            frames.append(p.ev("window.cloudsHarness.state()"))
    png(p, out / "drag_moving.png")
    m.up()
    s1 = p.ev("window.cloudsHarness.state()")
    p.wait_idle(40)
    time.sleep(0.5)
    png(p, out / "drag_1.png")
    a0, a1 = img_stats(out / "drag_0.png"), img_stats(out / "drag_1.png")
    check(abs(s1["el"] - s0["el"]) + abs(s1["az"] - s0["az"]) > 5, f"drag: canvas drag moved the sun {s0['el']:.1f},{s0['az']:.1f} -> {s1['el']:.1f},{s1['az']:.1f}")
    check(np.abs(a1 - a0).mean() > 2, f"drag: the image relit (mean |diff| {np.abs(a1 - a0).mean():.1f})")
    # the sun goes to the view ray under the pointer: project the new sun back to the screen
    pos = p.ev("""() => { const u = window.cloudsHarness.engine.u, s = u.uSun.value, f = u.uCamFwd.value;
      const z = s.dot(f); return [s.dot(u.uCamRight.value) / z / u.uTan.value.x, s.dot(u.uCamUp.value) / z / u.uTan.value.y]; }""")
    want = [(2 * (200 + 18 * 29) / box["width"] - 1), 1 - 2 * (150 + 4 * 29) / box["height"]]
    check(abs(pos[0] - want[0]) < 0.01 and abs(pos[1] - want[1]) < 0.01, f"drag: sun under the pointer (ndc {pos} vs {want})")
    caption = p.page.locator(".caption").first
    cap = p.ev("() => document.querySelector('#clouds div').shadowRoot.querySelector('.caption').textContent")
    check(f"{round(s1['az'] % 360)}" in cap, f"drag: caption follows ({cap!r})")
    # compass: drag to straight overhead-ish and to the north rim (behind the camera)
    cb = p.ev("""() => { const r = document.querySelector('#clouds div').shadowRoot.querySelector('.compass').getBoundingClientRect();
      return { x: r.left, y: r.top, w: r.width, h: r.height }; }""")
    cx, cy = cb["x"] + cb["w"] / 2, cb["y"] + cb["h"] / 2
    m.move(cx, cy - cb["h"] * 0.2)
    m.down()
    m.move(cx + cb["w"] * 0.05, cy - cb["h"] * 0.38, steps=6)
    m.up()
    s2 = p.ev("window.cloudsHarness.state()")
    check(s2["el"] < 15 and (s2["az"] < 30 or s2["az"] > 330), f"drag: compass sets azimuth/elevation (el {s2['el']:.1f}, az {s2['az']:.1f})")
    p.wait_idle(40)
    png(p, out / "drag_compass.png")
    # keys
    p.page.locator("#clouds").click(position={"x": 10, "y": 10})   # focus (also a click: the sun moves there)
    s3 = p.ev("window.cloudsHarness.state()")
    p.ev("() => document.querySelector('#clouds div').shadowRoot.querySelector('.root').focus()")
    for _ in range(3):
        p.page.keyboard.press("ArrowUp")
    p.page.keyboard.press("ArrowRight")
    s4 = p.ev("window.cloudsHarness.state()")
    check(abs(s4["el"] - min(89, s3["el"] + 3)) < 1e-6 and abs(((s4["az"] - s3["az"]) % 360) - 2) < 1e-6,
          f"keys: arrows move the sun ({s3['el']:.1f},{s3['az']:.1f} -> {s4['el']:.1f},{s4['az']:.1f})")
    # play
    p.ev("() => document.querySelector('#clouds div').shadowRoot.querySelector('.play').click()")
    time.sleep(1.5)
    s5 = p.ev("window.cloudsHarness.state()")
    time.sleep(1.0)
    s6 = p.ev("window.cloudsHarness.state()")
    png(p, out / "drag_play.png")
    check(s5["playing"] and (s5["el"], s5["az"]) != (s6["el"], s6["az"]), f"play: the sun sweeps ({s5['el']:.2f},{s5['az']:.2f} -> {s6['el']:.2f},{s6['az']:.2f})")
    p.ev("() => document.querySelector('#clouds div').shadowRoot.querySelector('.play').click()")
    idle = p.wait_idle(40)
    check(idle, "idle: the loop stops after the still image converges")
    # touch: a tap on the canvas puts the sun there
    tm = p.ev("window.cloudsHarness.timing()")
    check(not errors(p), f"drag: no console errors {errors(p)[:3]}")
    res = {"s0": s0, "during": frames, "s1": s1, "compass": s2, "keys": [s3, s4], "play": [s5, s6], "timing_state": tm["state"],
           "frames_tail": tm["frames"][-40:]}
    p.close()
    # touch device
    p = Page(pw, port, "?harness=1&view=backlit", viewport=(390, 700))
    p.wait_idle(40)
    s0 = p.ev("window.cloudsHarness.state()")
    p.page.dispatch_event("canvas", "pointerdown", {"pointerType": "touch", "clientX": 120, "clientY": 200, "pointerId": 7,
                                                     "isPrimary": True, "button": 0})
    p.page.dispatch_event("canvas", "pointerup", {"pointerType": "touch", "clientX": 120, "clientY": 200, "pointerId": 7})
    s1 = p.ev("window.cloudsHarness.state()")
    check((s0["el"], s0["az"]) != (s1["el"], s1["az"]), f"touch: a tap moves the sun ({s0['el']:.1f},{s0['az']:.1f} -> {s1['el']:.1f},{s1['az']:.1f})")
    p.wait_idle(40)
    png(p, out / "touch_portrait.png")
    check(not errors(p), f"touch: no console errors {errors(p)[:3]}")
    res["touch"] = [s0, s1]
    p.close()
    return res


def cmd_timing(pw, port, out):
    p = Page(pw, port, "?harness=1", viewport=(640, 400))
    res = {"load": p.ev("window.cloudsHarness.timing()")["load"]}
    res["bake"] = p.ev("window.cloudsHarness.bakeTiming(10)")
    res["static_bake"] = p.ev("window.cloudsHarness.timing()")["bakes"]["static"]
    for view, el, az in [("front", 24, 250), ("backlit", 8, 250), ("arc", 22, 200), ("sunset", 6, 270)]:
        for w, h in [(1280, 800), (1920, 1080)]:
            render(p, shot=view, elevation=el, azimuth=az, res=[w, h], frames=2)   # warm
            _, t = render(p, shot=view, elevation=el, azimuth=az, res=[w, h], frames=12)
            fm = sorted(t["frameMs"])
            res[f"{view}_{w}x{h}"] = {"median_ms": fm[len(fm) // 2], "min_ms": fm[0], "max_ms": fm[-1]}
            print(f"{view} {w}x{h}: median {fm[len(fm) // 2]:.1f} ms (min {fm[0]:.1f}, max {fm[-1]:.1f})")
    p.close()
    # interactive loop frame pacing while dragging at 1280x800 (device px)
    for vw, vh in [(1280, 800), (1920, 1080)]:
        p = Page(pw, port, "?harness=1&view=front", viewport=(vw, vh))
        p.wait_idle(40)
        m = p.page.mouse
        m.move(300, 200)
        m.down()
        t0 = time.perf_counter()
        k = 0
        while time.perf_counter() - t0 < 4.0:
            m.move(300 + (k % 120) * 5, 200 + (k % 40) * 2)
            k += 1
            time.sleep(1 / 120)
        m.up()
        tm = p.ev("window.cloudsHarness.timing()")
        cpu = [f for f in tm["frames"] if f["kind"] == "cpu" and f.get("moving") and f["dtMs"] > 0]
        dts = sorted(f["dtMs"] for f in cpu[5:])
        march = [f for f in tm["frames"] if f["kind"] == "march"]
        res[f"drag_{vw}x{vh}"] = {"frames": len(dts), "median_dt_ms": dts[len(dts) // 2] if dts else None,
                                  "p90_dt_ms": dts[int(len(dts) * 0.9)] if dts else None, "state": tm["state"],
                                  "gpu_march_ms": [f["gpuMs"] for f in march][-10:], "timer_query": tm["gl"]["timerQuery"]}
        print(f"drag {vw}x{vh}: {res[f'drag_{vw}x{vh}']}")
        p.close()
    return res


def cmd_score(pw, port, out):
    shots = json.loads((CLOUDS / "scene" / "out" / "shots.json").read_text())
    p = Page(pw, port, "?harness=1", viewport=(640, 400))
    res = {}
    for pid in SCORE_PAIRS:
        shot, preset = pid.rsplit("_", 1)
        cam, pr = shots["shots"][shot], shots["presets"][preset]
        light = {"E_sun": pr["E_sun"], "L_sky": pr["L_sky"], "L_gnd": pr["L_gnd"], "sky_h": 2000.0}
        lin, t = render(p, camera=cam, elevation=pr["elevation_deg"], azimuth=pr["azimuth_deg"], res=[640, 400],
                        frames=16, light=light, sunDisk=False)
        np.save(out / f"{pid}_page.npy", lin)
        ref, se = np.load(REF / f"{pid}.npy"), np.load(REF / f"{pid}_se.npy")
        r = {"page": score(lin, ref, se), "page_ms_per_frame": t["medianFrameMs"], "finite": bool(np.isfinite(lin).all())}
        for tag, f in [("research_compiled", out / f"{pid}_research_compiled.npy"),
                       ("interpreter", CLOUDS / "embed" / "tests" / "out" / "interp" / f"{pid}.npy")]:
            if f.is_file():
                o = np.load(f)
                d = np.abs(lin.astype(np.float64) - o)
                r[tag] = {"score": score(o, ref, se), "max_abs_diff_rel_mean": float(d.max() / o.mean()),
                          "mean_abs_diff_rel_mean": float(d.mean() / o.mean())}
        res[pid] = r
        print(pid, json.dumps(r))
        check(r["finite"], f"score: {pid} finite")
    check(not errors(p), f"score: no console errors {errors(p)[:3]}")
    p.close()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="all", choices=["all", "load", "shots", "drag", "timing", "score"])
    ap.add_argument("--out", type=Path, default=CLOUDS / "embed" / "tests" / "out" / "page")
    ap.add_argument("--views", default="")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright
    srv = serve()
    port = srv.server_address[1]
    cmds = ["load", "shots", "drag", "timing", "score"] if a.cmd == "all" else [a.cmd]
    with sync_playwright() as pw:
        for c in cmds:
            t = time.time()
            kw = {"views": [v for v in a.views.split(",") if v]} if c == "shots" else {}
            res = globals()[f"cmd_{c}"](pw, port, a.out, **kw)
            res["_wall_s"] = time.time() - t
            (a.out / f"{c}.json").write_text(json.dumps(res, indent=1, default=float))
    srv.shutdown()
    print("ALL PASS" if not FAILS else f"FAIL ({len(FAILS)}): " + "; ".join(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
