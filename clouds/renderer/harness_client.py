"""Shared plumbing for render.py and inspect.py: a local static server and a headless Chromium (Playwright)
running renderer/index.html, driven through window.harness (harness.js).

The server has two mounts: /renderer/ -> this directory, /scene/ -> the scene output directory
(default ../scene/out). Chromium is launched with the Metal ANGLE flags so WebGL runs on the GPU.
"""

from __future__ import annotations

import base64
import functools
import http.server
import json
import os
import socketserver
import sys
import threading
import time
from pathlib import Path

import numpy as np

RENDERER_DIR = Path(__file__).resolve().parent
DEFAULT_SCENE_DIR = (RENDERER_DIR / ".." / "scene" / "out").resolve()
GPU_ARGS = ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"]
# Harness-only features (exact single-scattering mode, background toggle, inspect's exact extras).
# The author-kit copy must set this to False (or drop those code paths); see NOTES.md.
HARNESS_FEATURES = True


class _Handler(http.server.SimpleHTTPRequestHandler):
    mounts: dict[str, Path] = {}

    def translate_path(self, path):
        path = path.split("?", 1)[0].split("#", 1)[0]
        for prefix, root in self.mounts.items():
            if path.startswith(prefix):
                rel = path[len(prefix):]
                p = (root / rel).resolve()
                if root not in p.parents and p != root:
                    return str(root / "__forbidden__")
                return str(p)
        return str(RENDERER_DIR / "__not_found__")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # quiet
        pass


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_server(scene_dir: Path, port: int = 0):
    handler = type("H", (_Handler,), {"mounts": {"/renderer/": RENDERER_DIR, "/scene/": Path(scene_dir).resolve()}})
    handler.extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                              ".glsl": "text/plain", ".json": "application/json", ".f32": "application/octet-stream",
                              ".u8": "application/octet-stream"}
    srv = _Server(("127.0.0.1", port), handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv


class HarnessError(RuntimeError):
    pass


class Harness:
    """with Harness(harness_mode=True) as h: h.render(...)"""

    def __init__(self, scene_dir: Path | None = None, harness_mode: bool = False, verbose: bool = False):
        if harness_mode and not HARNESS_FEATURES:
            raise SystemExit("harness-only features are disabled in this copy")
        self.scene_dir = Path(scene_dir or DEFAULT_SCENE_DIR)
        self.harness_mode = harness_mode
        self.verbose = verbose
        self.console: list[str] = []

    def __enter__(self):
        from playwright.sync_api import sync_playwright

        t0 = time.perf_counter()
        self.srv = start_server(self.scene_dir)
        port = self.srv.server_address[1]
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=True, args=GPU_ARGS)
        self.page = self.browser.new_page(viewport={"width": 64, "height": 64})
        self.page.on("console", self._on_console)
        self.page.on("pageerror", lambda e: self.console.append(f"pageerror: {e}"))
        url = f"http://127.0.0.1:{port}/renderer/index.html?scene=/scene/" + ("&harness=1" if self.harness_mode else "") + ("&trace=1" if self.verbose else "")
        self.page.goto(url)
        self.page.wait_for_function("window.harnessLoaded === true", timeout=30000)
        self.page.set_default_timeout(0)
        self.info = self._call("() => window.harness.init()")
        self.info["startupS"] = time.perf_counter() - t0
        gpu = self.info["gl"]["renderer"]
        if "swiftshader" in gpu.lower():
            print(f"WARNING: WebGL is running on SwiftShader ({gpu}), not the GPU; renders will be ~20x slower",
                  file=sys.stderr)
        return self

    def _on_console(self, msg):
        text = f"[{msg.type}] {msg.text}"
        self.console.append(text)
        if self.verbose or msg.type == "error":
            print("browser:", text, file=sys.stderr)

    def _call(self, fn, arg=None):
        try:
            return self.page.evaluate(fn, arg)
        except Exception as e:  # playwright.Error carries the JS message
            msg = str(e)
            raise HarnessError(msg.split("\n    at ")[0]) from None

    def __exit__(self, *exc):
        try:
            self.browser.close()
            self._pw.stop()
        finally:
            self.srv.shutdown()

    @property
    def shots(self):
        return self.info["shots"]

    def set_network(self, text: str):
        return self._call("(t) => window.harness.setNetwork(t)", text)

    def render(self, **kw):
        res = self._call("(a) => window.harness.render(a)", kw)
        w, h = res["width"], res["height"]
        lin = np.frombuffer(base64.b64decode(res["linear"]), dtype="<f4").reshape(h, w, 4)[::-1, :, :3].copy()
        ldr = np.frombuffer(base64.b64decode(res["ldr"]), dtype=np.uint8).reshape(h, w, 4)[::-1, :, :3].copy()
        return lin, ldr, res["stats"]

    def inspect(self, **kw):
        res = self._call("(a) => window.harness.inspect(a)", kw)
        data = np.frombuffer(base64.b64decode(res.pop("data")), dtype="<f4").reshape(res["rows"], res["cols"] * 4)
        res["table"] = data
        return res


def resolve_pair(shots: dict, ident: str, preset: str | None = None):
    """'front_noon' or shot 'front' + preset -> (shot, preset, exposure, id)."""
    if preset is None:
        if "_" not in ident:
            raise SystemExit(f"--shot {ident!r}: give a pair id like front_noon, or add --preset")
        shot, preset = ident.rsplit("_", 1)
    else:
        shot = ident
    if shot not in shots["shots"]:
        raise SystemExit(f"unknown shot {shot!r}; known: {', '.join(shots['shots'])}")
    if preset not in shots["presets"]:
        raise SystemExit(f"unknown preset {preset!r}; known: {', '.join(shots['presets'])}")
    pid = f"{shot}_{preset}"
    exposure = shots.get("exposure_table", {}).get(pid)
    for grp in shots["pairs"].values():
        for p in grp:
            if p["id"] == pid:
                exposure = p["exposure"]
    if exposure is None:
        exposure = 1.0
    return shot, preset, exposure, pid


def read_net(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"network file not found: {path}")
    return p.read_text()
