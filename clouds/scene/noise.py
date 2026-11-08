"""Deterministic, vectorised 3D noise for the cloud density field (numpy only).

All functions take broadcastable coordinate arrays (x, y, z) in lattice units and an integer `period`
(the lattice repeats every `period` cells, so the hash stays small), plus a seeded permutation table.
"""

from __future__ import annotations

import numpy as np

_GRADS = np.array(
    [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
     [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
     [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
     [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1]], dtype=np.float32)


class Noise:
    def __init__(self, seed: int, size: int = 512):
        rng = np.random.default_rng(seed)
        self.size = size
        p = rng.permutation(size).astype(np.int64)
        self.perm = np.concatenate([p, p])
        # per-lattice-point random jitter for Worley (3 floats), indexed by hash
        self.jitter = rng.random((size, 3)).astype(np.float32)

    def _hash(self, xi, yi, zi):
        P = self.perm
        n = self.size
        return P[(P[(P[xi % n] + yi) % n] + zi) % n]

    # ---------------- Perlin (improved) gradient noise, range about [-1, 1] ----------------
    def perlin(self, x, y, z, period: int | None = None):
        x = np.asarray(x, np.float32); y = np.asarray(y, np.float32); z = np.asarray(z, np.float32)
        xf0 = np.floor(x); yf0 = np.floor(y); zf0 = np.floor(z)
        fx = x - xf0; fy = y - yf0; fz = z - zf0
        xi = xf0.astype(np.int64); yi = yf0.astype(np.int64); zi = zf0.astype(np.int64)
        per = period if period is not None else self.size

        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)

        u, v, w = fade(fx), fade(fy), fade(fz)
        out = None
        for dx in (0, 1):
            wx = u if dx else 1 - u
            for dy in (0, 1):
                wy = v if dy else 1 - v
                for dz in (0, 1):
                    wz = w if dz else 1 - w
                    h = self._hash((xi + dx) % per, (yi + dy) % per, (zi + dz) % per) & 15
                    g = _GRADS[h]
                    dot = g[..., 0] * (fx - dx) + g[..., 1] * (fy - dy) + g[..., 2] * (fz - dz)
                    term = wx * wy * wz * dot
                    out = term if out is None else out + term
        return out.astype(np.float32)

    def perlin_fbm(self, x, y, z, octaves: int, period: int, lacunarity: float = 2.0, gain: float = 0.5):
        amp, freq, tot, norm = 1.0, 1.0, 0.0, 0.0
        for o in range(octaves):
            tot = tot + amp * self.perlin(x * freq, y * freq, z * freq, int(period * freq))
            norm += amp
            amp *= gain
            freq *= lacunarity
        return (tot / norm).astype(np.float32)

    # ---------------- Worley F1 cellular noise, returns distance in [0, ~1.0] ----------------
    def worley(self, x, y, z, period: int):
        x = np.asarray(x, np.float32); y = np.asarray(y, np.float32); z = np.asarray(z, np.float32)
        xf0 = np.floor(x); yf0 = np.floor(y); zf0 = np.floor(z)
        fx = x - xf0; fy = y - yf0; fz = z - zf0
        xi = xf0.astype(np.int64); yi = yf0.astype(np.int64); zi = zf0.astype(np.int64)
        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    h = self._hash((xi + dx) % period, (yi + dy) % period, (zi + dz) % period)
                    j = self.jitter[h]
                    ddx = dx + j[..., 0] - fx
                    ddy = dy + j[..., 1] - fy
                    ddz = dz + j[..., 2] - fz
                    d2 = ddx * ddx + ddy * ddy + ddz * ddz
                    best = d2 if best is None else np.minimum(best, d2)
        return np.sqrt(best).astype(np.float32)

    def worley_fbm(self, x, y, z, period: int, octaves: int = 3):
        """Inverted Worley FBM in [0, 1]: 1 at cell centres (billows), 0 at cell walls (cracks)."""
        weights = [0.625, 0.25, 0.125, 0.0625][:octaves]
        tot = 0.0
        for o, wgt in enumerate(weights):
            f = 2 ** o
            tot = tot + wgt * (1.0 - np.clip(self.worley(x * f, y * f, z * f, period * f), 0.0, 1.0))
        return (tot / sum(weights)).astype(np.float32)


def remap(v, lo, hi, nlo=0.0, nhi=1.0):
    return nlo + (v - lo) * (nhi - nlo) / (hi - lo)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)
