"""Diagnostic for 6.1: how much do the renderer's two SPEC cut-offs cost against the continuous integral?

    uv run python validate/diag_cutoffs.py [--t-stop 1e-6] [--d-min 0.002] [--tag term] [pairs ...]

SPEC 4.2 has the march skip samples with d <= 0.002 and stop when T < 0.002. Mitsuba has neither. This script copies
renderer/ to validate/out/diag/renderer (renderer/ itself is not touched), patches those two constants in the copy's
march shader, and renders --mode exact-single --no-background into validate/out/<tag>/. compare_single.py then
reports the patched images next to the SPEC ones.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess

from common import CLOUDS, OUT

PAIRS = ["front_noon", "side_noon", "side_sunset", "backlit_golden", "below_noon", "distant_sunset"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*", default=PAIRS)
    ap.add_argument("--t-stop", default="1e-6")
    ap.add_argument("--d-min", default="0.002")
    ap.add_argument("--tag", default="term")
    ap.add_argument("--frames", type=int, default=64)
    a = ap.parse_args()
    diag = OUT / "diag"
    if diag.exists():
        shutil.rmtree(diag)
    diag.mkdir(parents=True)
    shutil.copytree(CLOUDS / "renderer", diag / "renderer",
                    ignore=shutil.ignore_patterns("node_modules", "out", "tests", "__pycache__"))
    (diag / "scene").symlink_to(CLOUDS / "scene")  # the copy resolves ../scene/out and ../scene/tonemap.py
    sh = diag / "renderer" / "shaders" / "march.frag.glsl"
    src = sh.read_text()
    for old, new in [("    if (d > 0.002) {", f"    if (d > {a.d_min}) {{"),
                     ("      if (T < 0.002) { i++; return true; }", f"      if (T < {a.t_stop}) {{ i++; return true; }}")]:
        assert src.count(old) == 1, f"shader changed: cannot find {old!r}"
        src = src.replace(old, new)
    sh.write_text(src)
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "python", str(diag / "renderer" / "render.py"), "--mode", "exact-single", "--no-background",
           "--frames", str(a.frames), "--pair", *a.pairs, "--out", str(out) + "/"]
    r = subprocess.run(cmd, cwd=CLOUDS, capture_output=True, text=True)
    print(r.stdout, r.stderr)
    shutil.rmtree(diag)
    r.check_returncode()


if __name__ == "__main__":
    main()
