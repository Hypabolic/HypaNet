"""Build and check the author hand-over (SPEC section 7). Called by scripts/prepare_author_dir.sh and scripts/launch.sh.

    uv run python kit/kit_build.py scene                      # kit/build/scene/: the sanitised scene the tools serve
    uv run python kit/kit_build.py sky-table                  # KIT.md's sky radiance table (zenith, horizon)
    uv run python kit/kit_build.py references AUTHOR_DIR      # AUTHOR_DIR/references/: 12 pairs + scene.json
    uv run python kit/kit_build.py record AUTHOR_DIR RECORD [--test-root DIR]   # HANDOVER.json (both copies)
    uv run python kit/kit_build.py verify AUTHOR_DIR RECORD [--pristine]        # exit 1 on any difference

The sanitised scene (kit/build/scene/) is what the author's ./render and ./inspect serve to the browser: the density
grid, the four sky maps and a shots.json with only the five reference camera shots, the four presets and the 12
reference pairs (no held-back pair, no held-back camera, no exposure of a held-back pair). The side and distant shot
descriptions are corrected there (the sun is on the screen's right in both; scene/out/shots.json says left).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CLOUDS = Path(__file__).resolve().parents[1]
SCENE_OUT = CLOUDS / "scene" / "out"
REF_OUT = CLOUDS / "reference" / "out"
BUILD_SCENE = CLOUDS / "kit" / "build" / "scene"
KIT = CLOUDS / "kit"
REFERENCE_SHOTS = ("front", "side", "backlit", "below", "distant")
DESCRIPTIONS = {
    "front": "camera on the sun side (sun behind the camera), 3.3 km out, below cloud-base height, looking slightly up",
    "side": "sun at 90 degrees to the view, on the screen's right (terminator), 3.3 km out, looking slightly up",
    "backlit": "camera on the far side looking toward the sun, slightly upward (silver lining)",
    "below": "from below the base near its edge, looking up into the flat base and the towers; sun ahead-right",
    "distant": "distant view from 14 km near the ground, cloud low in the sky, horizon near the bottom edge; "
               "sun behind the camera and to the right",
}
# files of the harness the author's tools run; recorded at hand-over and checked before every launch
HARNESS_GLOBS = ("kit/tools/*.py", "kit/kit_build.py", "kit/make_settings.py", "kit/prompt.md", "kit/prompt_continue.md",
                 "kit/KIT.md", "renderer/*.py", "renderer/*.js", "renderer/*.html", "renderer/shaders/*",
                 "renderer/vendor/*", "scene/tonemap.py", "kit/build/scene/*")
AUTHOR_SKIP_DIRS = (".venv", "work")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitised_shots() -> dict:
    full = json.loads((SCENE_OUT / "shots.json").read_text())
    ref = full["pairs"]["reference"]
    assert len(ref) == 12, len(ref)
    ids = {p["id"] for p in ref}
    held = {p["id"] for p in full["pairs"].get("heldback", [])}
    assert not ids & held
    shots = {}
    for s in REFERENCE_SHOTS:
        c = dict(full["shots"][s])
        c["description"] = DESCRIPTIONS[s]
        shots[s] = c
    out = {k: full[k] for k in ("version", "units", "resolution", "camera_model", "medium", "sun_model", "presets")}
    out["shots"] = shots
    out["pairs"] = {"reference": ref}
    out["exposure_table"] = {p["id"]: p["exposure"] for p in ref}
    out["exposure_note"] = "exposure is per (shot, preset) pair; tone map: ACES fit of exposure * L, then sRGB"
    return out


def cmd_scene(_a) -> int:
    if BUILD_SCENE.exists():
        shutil.rmtree(BUILD_SCENE)
    BUILD_SCENE.mkdir(parents=True)
    s = sanitised_shots()
    (BUILD_SCENE / "shots.json").write_text(json.dumps(s, indent=2) + "\n")
    names = ["density.u8", "density.json"]
    for p in s["presets"].values():
        names += [p["sky_map"], p["sky_json"]]
    for n in names:
        shutil.copyfile(SCENE_OUT / n, BUILD_SCENE / n)  # copies: the harness server refuses symlinks out of its root
    text = (BUILD_SCENE / "shots.json").read_text()
    for bad in ("heldback", "quarter", "front_afternoon", "side_golden", "backlit_noon", "below_sunset",
                "distant_golden"):
        assert bad not in text, bad
    print(f"sanitised scene -> {BUILD_SCENE} ({len(names) + 1} files)")
    return 0


def cmd_references(a) -> int:
    author = Path(a.author_dir)
    dst = author / "references"
    dst.mkdir(parents=True, exist_ok=False)
    s = sanitised_shots()
    dens = json.loads((SCENE_OUT / "density.json").read_text())
    for p in s["pairs"]["reference"]:
        pid = p["id"]
        meta = json.loads((REF_OUT / f"{pid}.json").read_text())
        assert meta.get("mode") == "full" and meta["pair"] == pid, pid
        shutil.copyfile(REF_OUT / f"{pid}.png", dst / f"{pid}.png")
        shutil.copyfile(REF_OUT / f"{pid}.npy", dst / f"{pid}.npy")
        pre = s["presets"][p["preset"]]
        cam = s["shots"][p["shot"]]
        info = {
            "id": pid, "shot": p["shot"], "preset": p["preset"], "exposure": p["exposure"],
            "camera": {k: cam[k] for k in ("position", "look_at", "up", "fov_y_deg", "width", "height", "description")},
            "sun": {k: pre[k] for k in ("elevation_deg", "azimuth_deg", "toward_sun", "air_mass", "T_atm", "E_sun",
                                        "L_sky", "L_gnd", "E_sky")},
            "files": {"png": f"{pid}.png", "linear": f"{pid}.npy (float32, 400 x 640 x 3 linear RGB radiance, row 0 "
                                                     "at the top)"},
            "reference_render": {
                "renderer": f"Mitsuba {meta['mitsuba']} volpath, {meta['variant']}: unlimited scattering depth "
                            f"(Russian roulette from depth {meta['rr_depth']}), sun + sky map as lights",
                "spp": meta["spp"], "pixel_filter": meta["pixel_filter"],
                "noise_after_tone_map_8bit_levels": {"median": round(meta["tm8_noise_median"], 2),
                                                     "p90": round(meta["tm8_noise_p90"], 2)},
                "relative_standard_error_median": round(meta["rel_se_median"], 4),
            },
        }
        (dst / f"{pid}.json").write_text(json.dumps(info, indent=2) + "\n")
    scene = {
        "units": s["units"], "camera_model": s["camera_model"], "resolution": s["resolution"],
        "medium": s["medium"], "sun_model": s["sun_model"],
        "density_grid": {k: dens[k] for k in ("dims", "dims_order", "box_min", "box_max", "voxel_size", "cell_centred",
                                              "sigma_t_max", "albedo", "phase")},
        "presets": {n: {k: v for k, v in p.items() if k not in ("sky_map", "sky_json")} for n, p in s["presets"].items()},
        "shots": s["shots"],
        "reference_pairs": s["pairs"]["reference"],
    }
    (dst / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    for name in s["presets"]:  # sky previews: physics inputs (the sky the cloud sits in), not reference pixels
        shutil.copyfile(SCENE_OUT / f"sky_{name}.png", dst / f"sky_{name}.png")
    print(f"references -> {dst} (12 pairs + scene.json + {len(s['presets'])} sky previews)")
    return 0


def sky_radiance(name: str, elev_deg: float, az_deg: float):
    """Bilinear sample of scene/out/sky_<name>.f32 in a direction (elevation, azimuth clockwise from north = -z)."""
    import numpy as np
    meta = json.loads((SCENE_OUT / f"sky_{name}.json").read_text())
    W, H = meta["width"], meta["height"]
    img = np.fromfile(SCENE_OUT / meta["file"], dtype="<f4").reshape(H, W, 3)
    theta, phi = np.radians(90.0 - elev_deg), np.radians(az_deg % 360.0)
    x, y = phi / (2 * np.pi) * W - 0.5, theta / np.pi * H - 0.5
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - x0, y - y0
    y0c, y1c = min(max(y0, 0), H - 1), min(max(y0 + 1, 0), H - 1)
    x0w, x1w = x0 % W, (x0 + 1) % W
    top = img[y0c, x0w] * (1 - fx) + img[y0c, x1w] * fx
    bot = img[y1c, x0w] * (1 - fx) + img[y1c, x1w] * fx
    return top * (1 - fy) + bot * fy


SKY_HORIZON_ELEV = 1.0


def cmd_sky_table(_a) -> int:
    """Print KIT.md's table of sky radiance at the zenith and just above the horizon (markdown)."""
    s = sanitised_shots()
    f = lambda v: ", ".join(f"{x:.4g}" for x in v)  # noqa: E731
    print(f"| preset | zenith | horizon toward the sun (az 250°) | horizon across (az 160° / 340°) | "
          f"horizon away from the sun (az 70°) |")
    print("|---|---|---|---|---|")
    for name, p in s["presets"].items():
        az = p["azimuth_deg"]
        h = SKY_HORIZON_ELEV
        across = (sky_radiance(name, h, az - 90) + sky_radiance(name, h, az + 90)) / 2
        print(f"| {name} | {f(sky_radiance(name, 90.0, 0.0))} | {f(sky_radiance(name, h, az))} | {f(across)} | "
              f"{f(sky_radiance(name, h, az + 180))} |")
    return 0


def author_files(author: Path) -> dict[str, str]:
    out = {}
    for root, dirs, files in os.walk(author):
        rel_root = os.path.relpath(root, author)
        if rel_root == ".":
            dirs[:] = [d for d in dirs if d not in AUTHOR_SKIP_DIRS]
        for f in files:
            rel = os.path.normpath(os.path.join(rel_root, f))
            if rel == "HANDOVER.json":
                continue
            out[rel] = sha256(author / rel)
    return dict(sorted(out.items()))


def harness_files() -> dict[str, str]:
    out = {}
    for g in HARNESS_GLOBS:
        for p in sorted(CLOUDS.glob(g)):
            if p.is_file():
                out[str(p.relative_to(CLOUDS))] = sha256(p)
    return dict(sorted(out.items()))


def venv_info(author: Path) -> dict:
    py = author / ".venv" / "bin" / "python"
    code = ("import sys, numpy, importlib.metadata as m; print(sys.version.split()[0]); print(numpy.__version__); "
            "print(','.join(sorted(d.metadata['Name'] for d in m.distributions())))")
    r = subprocess.run([str(py), "-I", "-B", "-c", code], capture_output=True, text=True, check=True)
    ver, npv, dists = r.stdout.strip().splitlines()
    return {"python": ver, "numpy": npv, "distributions": dists.split(",")}


def cmd_record(a) -> int:
    author = Path(a.author_dir).resolve()
    rec = {
        "kind": "hypanet-clouds-author",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test": bool(a.test_root),
        "test_root": a.test_root,
        "author_dir": str(author),
        "files": author_files(author),
        "venv": venv_info(author),
        "harness": harness_files(),
    }
    text = json.dumps(rec, indent=2) + "\n"
    (author / "HANDOVER.json").write_text(text)
    Path(a.record).parent.mkdir(parents=True, exist_ok=True)
    Path(a.record).write_text(text)
    print(f"hand-over record: {len(rec['files'])} author files, {len(rec['harness'])} harness files -> {a.record}")
    return 0


def problems(author: Path, record: dict, pristine: bool) -> list[str]:
    out = []
    if record.get("kind") != "hypanet-clouds-author":
        return ["the record is not a HypaNet Clouds hand-over record"]
    if os.path.realpath(author) != record["author_dir"]:
        out.append(f"author dir {author} is not the recorded {record['author_dir']}")
    hv = author / "HANDOVER.json"
    if not hv.is_file() or json.loads(hv.read_text()) != record:
        out.append("AUTHOR_DIR/HANDOVER.json differs from the record")
    now = author_files(author)
    for rel, h in record["files"].items():
        if rel not in now:
            out.append(f"missing: {rel}")
        elif now[rel] != h:
            out.append(f"changed: {rel}")
    for rel in now:
        if rel not in record["files"]:
            out.append(f"added outside work/: {rel}")
    hnow = harness_files()
    for rel, h in record["harness"].items():
        if hnow.get(rel) != h:
            out.append(f"harness file changed since the hand-over: {rel}")
    for rel in hnow:
        if rel not in record["harness"]:
            out.append(f"harness file added since the hand-over: {rel}")
    try:
        if venv_info(author) != record["venv"]:
            out.append("the private venv differs from the hand-over")
    except (subprocess.CalledProcessError, OSError, ValueError) as e:
        out.append(f"the private venv does not run: {e}")
    work = author / "work"
    if not work.is_dir():
        out.append("work/ missing")
    elif pristine and any(work.iterdir()):
        out.append("work/ is not empty (a fresh session needs a pristine hand-over)")
    for d in ("references", "tools"):
        for p in (author / d).rglob("*"):
            if os.access(p, os.W_OK):
                out.append(f"writable: {d}/{p.relative_to(author / d)}")
                break
    return out


def cmd_verify(a) -> int:
    author = Path(a.author_dir).resolve()
    record = json.loads(Path(a.record).read_text())
    probs = problems(author, record, a.pristine)
    for p in probs:
        print("PROBLEM:", p)
    if probs:
        return 1
    print(f"hand-over verified: {len(record['files'])} author files, {len(record['harness'])} harness files"
          + (" (TEST hand-over)" if record["test"] else ""))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scene")
    sub.add_parser("sky-table")
    r = sub.add_parser("references")
    r.add_argument("author_dir")
    r = sub.add_parser("record")
    r.add_argument("author_dir")
    r.add_argument("record")
    r.add_argument("--test-root")
    r = sub.add_parser("verify")
    r.add_argument("author_dir")
    r.add_argument("record")
    r.add_argument("--pristine", action="store_true")
    a = ap.parse_args(argv)
    return {"scene": cmd_scene, "sky-table": cmd_sky_table, "references": cmd_references, "record": cmd_record, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
