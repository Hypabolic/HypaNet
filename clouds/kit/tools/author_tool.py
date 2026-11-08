"""The author's ./render and ./inspect (SPEC section 7), run on the harness side.

The author directory holds two thin shell wrappers that call this file with the harness's uv environment:

    author_tool.py AUTHOR_DIR EXTRA_ROOT render  [author arguments ...]
    author_tool.py AUTHOR_DIR EXTRA_ROOT inspect [author arguments ...]

AUTHOR_DIR and EXTRA_ROOT are fixed by the wrapper (EXTRA_ROOT is "-" in a real hand-over; a test hand-over made
with scripts/prepare_author_dir.sh --test-root DIR may read networks from and write output to DIR as well). The
author's own arguments come after the subcommand and cannot change them.

What the author can reach:
- the 12 reference pairs by id (--shot front_noon), or --set reference;
- a free mode: one of the five reference camera shots (--shot front) or any camera (--camera X,Y,Z --look-at X,Y,Z
  [--fov DEG]) with a sun preset or any sun (--sun noon | --sun ELEV,AZ);
- nothing else: no held-back pair ids, no held-back camera, no harness-only modes. The browser itself is served a
  sanitised scene directory (kit/build/scene/, built by kit/kit_build.py) whose shots.json holds only the reference
  shots, presets and pairs.

The rendering and inspection are renderer/render.py and renderer/inspect_ray.py, unchanged; this file only
validates the author's arguments, points them at the sanitised scene and the author's references/, and adds the
free camera and free sun through a small patch of the page's window.harness.setup (camera from the shot table,
sun from CloudRenderer.setSun, the viewer's interpolation of the sky between presets).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

CLOUDS = Path(__file__).resolve().parents[2]
RENDERER = CLOUDS / "renderer"
SCENE = CLOUDS / "kit" / "build" / "scene"
# renderer/ goes at the END of sys.path: it holds inspect.py, which must not shadow the standard library's inspect
sys.path.append(str(RENDERER))

import harness_client  # noqa: E402

harness_client.HARNESS_FEATURES = False  # no exact single-scattering mode, no background toggle, no exact extras
import inspect_ray  # noqa: E402
import render as render_mod  # noqa: E402

SHOT_NAMES = ("front", "side", "backlit", "below", "distant")
PRESETS = ("noon", "afternoon", "golden", "sunset")
# free-mode default exposure per preset (the reference pairs' exposures span about 0.2 to 2.5)
DEFAULT_EXPOSURE = {"noon": 0.315, "afternoon": 0.3969, "golden": 0.5, "sunset": 2.0}
PRESET_ELEV = {"noon": 55.0, "afternoon": 25.0, "golden": 8.0, "sunset": 2.0}
MAX_RES = (1280, 800)
MAX_FRAMES = 256
MAX_NET_BYTES = 8_000_000

PATCH_JS = """(cam) => {
  const r = window.harnessState.r, h = window.harness;
  if (cam) r.shots.shots.custom = cam;
  const orig = h.setup;
  h.setup = async function (a) {
    const m = /^sun:(-?[0-9.]+),(-?[0-9.]+)$/.exec(String(a.preset));
    if (!m) return orig.call(this, a);
    const c = r.shots.shots[a.shot];
    if (!c) throw new Error(`unknown shot ${a.shot}`);
    r.setResolution(a.width || c.width, a.height || c.height);
    r.setCamera(c);
    await r.setSun(parseFloat(m[1]), parseFloat(m[2]));
  };
  return true;
}"""


class ToolError(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"error: {msg}")


def _under(path: str, root: str) -> bool:
    root = root.rstrip("/")
    return path == root or path.startswith(root + "/")


class Roots:
    def __init__(self, author_dir: str, extra: str):
        self.author = os.path.realpath(author_dir)
        self.work = os.path.join(self.author, "work")
        self.extra = None if extra in ("", "-") else os.path.realpath(extra)
        self.refs = os.path.join(self.author, "references")

    def readable(self, p: str) -> bool:
        return _under(p, self.author) or (self.extra is not None and _under(p, self.extra))

    def writable(self, p: str) -> bool:
        return _under(p, self.work) or (self.extra is not None and _under(p, self.extra))

    def where(self, write: bool) -> str:
        base = "work/ in your directory" if write else "your directory"
        return base + (f" (or the test root {self.extra})" if self.extra else "")


def load_scene() -> dict:
    return json.loads((SCENE / "shots.json").read_text())


def reference_pairs(scene: dict) -> dict[str, dict]:
    return {p["id"]: p for p in scene["pairs"]["reference"]}


def parse_vec(text: str, what: str) -> list[float]:
    try:
        v = [float(x) for x in text.split(",")]
    except ValueError:
        raise ToolError(f"{what} must be three numbers separated by commas, like 0,1900,0") from None
    if len(v) != 3 or not all(math.isfinite(x) for x in v):
        raise ToolError(f"{what} must be three finite numbers separated by commas")
    return v


def parse_sun(text: str) -> tuple[str, str, float]:
    """--sun noon | ELEV,AZ -> (preset key for the page, label, elevation for the default exposure)."""
    if text in PRESETS:
        return text, text, PRESET_ELEV[text]
    m = re.fullmatch(r"\s*(-?[0-9]+(?:\.[0-9]*)?)\s*,\s*(-?[0-9]+(?:\.[0-9]*)?)\s*", text)
    if not m:
        raise ToolError(f"--sun must be a preset ({', '.join(PRESETS)}) or ELEVATION,AZIMUTH in degrees, like 25,250")
    e, a = float(m.group(1)), float(m.group(2)) % 360.0
    if not 1.0 <= e <= 89.0:
        raise ToolError("--sun elevation must be between 1 and 89 degrees")
    return f"sun:{e:g},{a:g}", f"e{e:g}_a{a:g}", e


def default_exposure(elev: float) -> float:
    best = min(PRESETS, key=lambda p: abs(PRESET_ELEV[p] - elev))
    return DEFAULT_EXPOSURE[best]


def parse_res(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    m = re.fullmatch(r"([0-9]+)[xX]([0-9]+)", text.strip())
    if not m:
        raise ToolError("--res must look like 320x200")
    w, h = int(m.group(1)), int(m.group(2))
    if not (16 <= w <= MAX_RES[0] and 16 <= h <= MAX_RES[1]):
        raise ToolError(f"--res must be between 16x16 and {MAX_RES[0]}x{MAX_RES[1]}")
    return w, h


def selection(a, scene: dict) -> tuple[list[tuple[str, tuple]], dict | None]:
    """The views to render: [(token, (shot, preset_key, exposure, pid))], plus the free camera (or None)."""
    refs = reference_pairs(scene)
    jobs: list[tuple[str, tuple]] = []
    cam = None
    if getattr(a, "set", None):
        if a.shot or a.sun or a.camera:
            raise ToolError("--set reference cannot be combined with --shot, --sun or --camera")
        for pid, p in refs.items():
            jobs.append((pid, (p["shot"], p["preset"], p["exposure"], pid)))
        return jobs, None
    if a.camera:
        if a.shot:
            raise ToolError("give either --shot or --camera, not both")
        if not a.look_at or not a.sun:
            raise ToolError("--camera needs --look-at X,Y,Z and --sun (a preset or ELEV,AZ)")
        pos, look = parse_vec(a.camera, "--camera"), parse_vec(a.look_at, "--look-at")
        f = [look[i] - pos[i] for i in range(3)]
        n = math.sqrt(sum(x * x for x in f))
        if n < 1.0:
            raise ToolError("--look-at must be at least 1 m from --camera")
        if abs(f[1] / n) > 0.999:
            raise ToolError("the camera cannot look straight up or down (its up vector is world +y)")
        fov = 30.0 if a.fov is None else a.fov
        if not 1.0 <= fov <= 120.0:
            raise ToolError("--fov must be between 1 and 120 degrees (vertical field of view)")
        cam = {"position": pos, "look_at": look, "up": [0.0, 1.0, 0.0], "fov_y_deg": fov, "width": 640, "height": 400}
        key, label, elev = parse_sun(a.sun)
        jobs.append(("custom_" + label, ("custom", key, default_exposure(elev), "custom_" + label)))
        return jobs, cam
    if a.look_at or a.fov is not None:
        raise ToolError("--look-at and --fov go with --camera")
    if not a.shot:
        raise ToolError("give --shot PAIR_ID (one of the 12 reference pairs), --shot SHOT --sun SUN, "
                        "--camera ... --look-at ... --sun SUN" + (", or --set reference" if hasattr(a, "set") else ""))
    if a.shot in refs and not a.sun:
        p = refs[a.shot]
        return [(a.shot, (p["shot"], p["preset"], p["exposure"], a.shot))], None
    if a.shot in SHOT_NAMES:
        if not a.sun:
            raise ToolError(f"--shot {a.shot} is a camera; add --sun (a preset or ELEV,AZ), or give a reference pair id")
        key, label, elev = parse_sun(a.sun)
        pid = f"{a.shot}_{label}"
        exposure = refs[pid]["exposure"] if pid in refs else default_exposure(elev)
        return [(pid, (a.shot, key, exposure, pid))], None
    if a.shot in refs:
        raise ToolError(f"--shot {a.shot} is a reference pair, which fixes the sun; to change the sun use "
                        f"--shot {refs[a.shot]['shot']} --sun ...")
    raise ToolError(f"unknown --shot {a.shot!r}. Reference pairs: {', '.join(refs)}. "
                    f"Camera shots (with --sun): {', '.join(SHOT_NAMES)}")


def make_harness(cam: dict | None):
    class KitHarness(harness_client.Harness):
        def __init__(self, scene_dir=None, harness_mode=False, verbose=False):
            if harness_mode:
                raise ToolError("that mode is not available")
            super().__init__(scene_dir=SCENE, harness_mode=False, verbose=False)

        def __enter__(self):
            super().__enter__()
            self._call(PATCH_JS, cam)
            if cam is not None:
                self.info["shots"]["shots"]["custom"] = cam
            return self

    return KitHarness


def install(jobs, cam, roots: Roots):
    table = dict(jobs)

    def resolve(shots, ident, preset=None):
        if ident not in table:
            raise ToolError(f"unknown view {ident!r}")
        return table[ident]

    H = make_harness(cam)
    for mod in (render_mod, inspect_ray):
        mod.Harness = H
        mod.resolve_pair = resolve
    render_mod.REFERENCE_DIR = Path(roots.refs)


def net_path(text: str, roots: Roots) -> str:
    p = os.path.realpath(text)
    if not roots.readable(p):
        raise ToolError(f"the network file must be inside {roots.where(False)}: {text}")
    if not os.path.isfile(p):
        raise ToolError(f"network file not found: {text}")
    if os.path.getsize(p) > MAX_NET_BYTES:
        raise ToolError(f"network file larger than {MAX_NET_BYTES} bytes")
    try:
        json.loads(Path(p).read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ToolError(f"{text} is not valid JSON: {e}") from None
    return p


def out_path(text: str, roots: Roots, *, is_dir: bool) -> str:
    p = os.path.realpath(text)
    if not roots.writable(p):
        raise ToolError(f"output must go inside {roots.where(True)}: {text}")
    if is_dir:
        return p.rstrip("/") + "/"
    return p


def common_args(ap: argparse.ArgumentParser):
    ap.add_argument("net", help="your network JSON file (inside your directory)")
    ap.add_argument("--shot", help="a reference pair id (front_noon ...) or, with --sun, a camera shot "
                                   "(front, side, backlit, below, distant)")
    ap.add_argument("--sun", help="free sun: a preset (noon, afternoon, golden, sunset) or ELEV,AZ in degrees "
                                  "(elevation 1..89, azimuth clockwise from north = -z; the presets use azimuth 250)")
    ap.add_argument("--camera", help="free camera position X,Y,Z in metres (needs --look-at and --sun)")
    ap.add_argument("--look-at", dest="look_at", help="free camera target X,Y,Z in metres")
    ap.add_argument("--fov", type=float, help="free camera vertical field of view in degrees (default 30)")
    ap.add_argument("--res", help="WxH, default 640x400 (vertical field of view kept); up to 1280x800")
    ap.add_argument("--exposure", type=float, help="override the exposure used for the PNG")


def cmd_render(argv, roots: Roots) -> int:
    ap = argparse.ArgumentParser(
        prog="./render",
        description="Render your network: tone-mapped PNG, linear .npy (H x W x 3 float32, row 0 at the top) and a "
                    ".json of settings and timings. Default output: work/renders/<view>_<network name>.png.")
    common_args(ap)
    ap.add_argument("--set", choices=["reference"], help="render all 12 reference pairs")
    ap.add_argument("--frames", type=int, default=16, help=f"frames to accumulate (1..{MAX_FRAMES}, default 16)")
    ap.add_argument("--out", help="output .png path (one view) or directory ending in / (default work/renders/)")
    ap.add_argument("--compare", action="store_true",
                    help="also write <out>_compare.png (render | reference) and print the linear error "
                         "(reference pairs only)")
    a = ap.parse_args(argv)
    if not 1 <= a.frames <= MAX_FRAMES:
        raise ToolError(f"--frames must be 1..{MAX_FRAMES}")
    if a.exposure is not None and not (math.isfinite(a.exposure) and 0 < a.exposure <= 1e4):
        raise ToolError("--exposure must be a positive number")
    res = parse_res(a.res)
    scene = load_scene()
    jobs, cam = selection(a, scene)
    net = net_path(a.net, roots)
    if a.out:
        as_dir = a.out.endswith("/") or len(jobs) > 1
        if not as_dir and not a.out.lower().endswith(".png"):
            raise ToolError("--out must end in .png (one view) or / (a directory)")
        out = out_path(a.out, roots, is_dir=as_dir)
    else:
        out = out_path(os.path.join(roots.work, "renders"), roots, is_dir=True)
    install(jobs, cam, roots)
    args = [net, "--pair", *[t for t, _ in jobs], "--frames", str(a.frames), "--out", out]
    if res:
        args += ["--res", f"{res[0]}x{res[1]}"]
    if a.exposure is not None:
        args += ["--exposure", repr(a.exposure)]
    if a.compare:
        args += ["--compare"]
    for t, (_, _, exposure, _) in jobs:
        if t.startswith("custom") or t not in reference_pairs(scene):
            print(f"{t}: free view (no reference image), exposure {a.exposure or exposure}")
    return render_mod.main(args)


def cmd_inspect(argv, roots: Roots) -> int:
    ap = argparse.ArgumentParser(
        prog="./inspect",
        description="Watch the neurons along one pixel's ray: for each sample with d > 0.002, the march state, every "
                    "probe value, every unit's activation, the 9 outputs, S and the sample's contribution to the "
                    "pixel. The numbers come from the shader itself (same jitter as frame --frame of a render).")
    common_args(ap)
    ap.add_argument("--pixel", nargs=2, type=int, required=True, metavar=("X", "Y"),
                    help="column, row (0,0 = top left) at the render's resolution")
    ap.add_argument("--samples", default="8", help="how many samples to print (spread over the ray's contribution), "
                                                   "or 'all'")
    ap.add_argument("--frame", type=int, default=0, help="which render frame's jitter to replay (default 0)")
    ap.add_argument("--json", help="also write every sample's full record to this JSON file (inside work/)")
    a = ap.parse_args(argv)
    if a.samples != "all" and not (a.samples.isdigit() and int(a.samples) >= 1):
        raise ToolError("--samples must be a positive integer or 'all'")
    if not 0 <= a.frame < 100000:
        raise ToolError("--frame must be 0..99999")
    res = parse_res(a.res)
    W, H = res or (640, 400)  # every view is 640 x 400 unless --res says otherwise
    if not (0 <= a.pixel[0] < W and 0 <= a.pixel[1] < H):
        ap.error(f"--pixel must be inside {W}x{H} (column 0..{W - 1}, row 0..{H - 1})")
    scene = load_scene()
    a.set = None
    jobs, cam = selection(a, scene)
    net = net_path(a.net, roots)
    install(jobs, cam, roots)
    token, (shot, key, exposure, pid) = jobs[0]
    if a.exposure is not None:
        inspect_ray.resolve_pair = lambda shots, ident, preset=None: (shot, key, a.exposure, pid)
    args = [net, "--pair", token, "--pixel", str(a.pixel[0]), str(a.pixel[1]), "--samples", a.samples,
            "--frame", str(a.frame)]
    if res:
        args += ["--res", f"{res[0]}x{res[1]}"]
    if a.json:
        if not a.json.lower().endswith(".json"):
            raise ToolError("--json must end in .json")
        args += ["--json", out_path(a.json, roots, is_dir=False)]
    return inspect_ray.main(args)


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[2] not in ("render", "inspect"):
        print("author_tool.py AUTHOR_DIR EXTRA_ROOT render|inspect [args]", file=sys.stderr)
        return 2
    roots = Roots(argv[0], argv[1])
    # argparse in renderer/render.py and renderer/inspect_ray.py names itself after basename(sys.argv[0]): "render" /
    # "inspect", not this harness file
    sys.argv[0] = "./" + argv[2]
    if not (SCENE / "shots.json").is_file():
        print("error: the kit's scene directory is missing (harness: run scripts/prepare_author_dir.sh)",
              file=sys.stderr)
        return 2
    try:
        if argv[2] == "render":
            return cmd_render(argv[3:], roots)
        return cmd_inspect(argv[3:], roots)
    except harness_client.HarnessError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
