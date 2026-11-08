"""Shared helpers for the SPEC section 6 validation scripts (harness only; never shipped in the author kit)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VALIDATE = Path(__file__).resolve().parent
CLOUDS = VALIDATE.parent
SCENE_OUT = CLOUDS / "scene" / "out"
REF_OUT = CLOUDS / "reference" / "out"
OUT = VALIDATE / "out"
LUM = np.array([0.2126, 0.7152, 0.0722])

sys.path.insert(0, str(CLOUDS / "scene"))
from tonemap import tonemap  # noqa: E402  (the SPEC tone map, numpy twin of the shader's)


def shots():
    return json.loads((SCENE_OUT / "shots.json").read_text())


def exposure(pid):
    S = shots()
    for g in ("reference", "heldback"):
        for p in S["pairs"][g]:
            if p["id"] == pid:
                return p["exposure"]
    shot, preset = pid.split("_", 1)
    t = S["exposure_table"]
    return t[shot][preset] if isinstance(t.get(shot), dict) else t[pid]


def lum(img):
    return img.astype(np.float64) @ LUM


def render(args, check=True):
    """Run renderer/render.py with the given argument list (from clouds/)."""
    cmd = ["uv", "run", "python", "renderer/render.py", *args]
    r = subprocess.run(cmd, cwd=CLOUDS, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"render failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r


def block(img, k):
    """k x k box average (crops to a multiple of k)."""
    H, W = img.shape[:2]
    H2, W2 = H // k * k, W // k * k
    x = img[:H2, :W2]
    return x.reshape(H2 // k, k, W2 // k, k, *img.shape[2:]).mean(axis=(1, 3))


def font(size=14):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def labelled(arr, text):
    img = Image.fromarray(arr, "RGB")
    out = Image.new("RGB", (img.width, img.height + 22), (20, 20, 20))
    out.paste(img, (0, 22))
    ImageDraw.Draw(out).text((6, 3), text, fill=(235, 235, 235), font=font(14))
    return out


def row(images, footer=None, gap=4):
    h = max(i.height for i in images)
    w = sum(i.width for i in images) + gap * (len(images) - 1)
    foot = 22 if footer else 0
    out = Image.new("RGB", (w, h + foot), (20, 20, 20))
    x = 0
    for i in images:
        out.paste(i, (x, 0))
        x += i.width + gap
    if footer:
        ImageDraw.Draw(out).text((6, h + 3), footer, fill=(235, 235, 235), font=font(13))
    return out


def stack(images, gap=6):
    w = max(i.width for i in images)
    h = sum(i.height for i in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (w, h), (10, 10, 10))
    y = 0
    for i in images:
        out.paste(i, (0, y))
        y += i.height + gap
    return out


def diverging(rel, mask, scale):
    """Signed relative difference -> RGB (blue = renderer low, red = renderer high, white = 0, saturated at
    +-scale); pixels outside `mask` are dark grey."""
    t = np.clip(rel / scale, -1, 1)
    r = np.where(t > 0, 1.0, 1.0 + t)
    g = 1.0 - np.abs(t)
    b = np.where(t < 0, 1.0, 1.0 - t)
    rgb = np.stack([r, g, b], -1)
    rgb = np.where(mask[..., None], rgb, 0.18)
    return (rgb * 255 + 0.5).astype(np.uint8)
