"""Tone map for HypaNet Clouds (SPEC section 3), identical in numpy and GLSL.

    c = exposure * L                     (linear RGB radiance, per channel)
    f(x) = clamp((x (2.51 x + 0.03)) / (x (2.43 x + 0.59) + 0.14), 0, 1)   (ACES fit, Narkowicz)
    y = sRGB OETF(f(x))                  (x <= 0.0031308 ? 12.92 x : 1.055 x^(1/2.4) - 0.055)
    out = floor(255 * y + 0.5)           (8-bit, round half up; GLSL: floor(y * 255.0 + 0.5))

Negative or NaN radiance is treated as 0 (the ACES fit is clamped to [0, 1]).

GLSL equivalent:

    vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
    vec3 srgb(vec3 x) { return mix(12.92 * x, 1.055 * pow(x, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, x)); }
    vec3 c = aces(max(exposure * L, 0.0)); vec3 y = srgb(c); // 8-bit: floor(y * 255.0 + 0.5)
"""

from __future__ import annotations

import numpy as np


def aces(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1.0 / 2.4) - 0.055)


def tonemap_float(linear: np.ndarray, exposure: float) -> np.ndarray:
    """Linear RGB radiance (..., 3) -> display-referred sRGB in [0, 1] (float64, before quantisation)."""
    c = np.nan_to_num(np.asarray(linear, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    c = np.maximum(exposure * c, 0.0)
    return srgb_oetf(aces(c))


def tonemap(linear: np.ndarray, exposure: float) -> np.ndarray:
    """Linear RGB radiance (..., 3) -> uint8 sRGB (..., 3)."""
    y = tonemap_float(linear, exposure)
    return np.floor(y * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def save_png(path, linear: np.ndarray, exposure: float) -> None:
    from PIL import Image

    Image.fromarray(tonemap(linear, exposure)[..., :3], "RGB").save(path)


if __name__ == "__main__":
    xs = np.array([0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    for x in xs:
        print(f"x={x:6.2f}  aces={aces(x):.4f}  sRGB8={int(tonemap(np.array([x, x, x]), 1.0)[0])}")
