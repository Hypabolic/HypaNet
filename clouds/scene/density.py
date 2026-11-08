"""Procedural cumulus congestus density field (deterministic).

Recipe (SPEC section 3: displaced metaballs + Perlin-Worley FBM erosion + height profile):

1. Skeleton: a broad, low "base slab" of overlapping spheres sitting on the condensation level, and several rising
   towers (thermals) of different heights, each a chain of spheres that narrows upward, leans slightly downwind
   and ends in a rounded crown. The towers are far enough apart that they read as separate turrets with dips
   between them.
2. Billows: three generations of smaller spheres seeded on the current iso-surface, biased toward upward-facing
   surface and toward height, each only partly sunk into its parent so it bulges out (cauliflower). Billow size
   grows with height (young, vigorous tops have big round heads; the lower flanks are finer and more ragged).
3. Smooth union (log-sum-exp of signed distances, small radius so the creases between billows stay sharp),
   evaluated at domain-warped coordinates, plus a medium-scale FBM displacement. The base is cut flat at ~1500 m
   with a small ragged undulation.
4. Edge: the signed distance becomes a density ramp across a narrow band (crisp at the tops, wider low down),
   after an inward carve by Perlin-Worley noise: fine billowy cracks on the tops, deep wispy erosion on the lower
   flanks and around the base rim. Thin wisps (fringe) hang just outside the lower flanks.
5. Interior density follows a height profile (liquid water content grows with height above the base) with gentle
   low-frequency variation.
"""

from __future__ import annotations

import time

import numpy as np

from noise import Noise, smoothstep

DIMS = (256, 160, 256)  # (nx, ny, nz)
BOX_MIN = np.array([-900.0, 1450.0, -900.0])
BOX_MAX = np.array([900.0, 2550.0, 900.0])
BASE_Y = 1500.0
TOP_LIMIT = 2500.0  # no billow sphere may reach above this (box top 2550, margin for noise)
SEED = 20261108
K_SMOOTH = 6.0  # smooth-union radius (m); small = sharp creases between billows
LEAN = np.array([0.10, 0.0, 0.03])  # tower lean (downwind shift per metre of height): weak shear toward +x

# base slab: broad low spheres on the condensation level; (x, z, r). Centre height BASE_Y + 0.25 r.
SLAB = [
    (0.0, 0.0, 300.0), (-330.0, 120.0, 260.0), (320.0, 200.0, 250.0), (-180.0, -320.0, 240.0),
    (300.0, -300.0, 230.0), (-470.0, -170.0, 200.0), (110.0, 380.0, 220.0), (-200.0, 380.0, 190.0),
    (520.0, -40.0, 190.0), (-560.0, 160.0, 160.0),
]

# towers: (x, z, top_y, r_base, r_top) -- different heights, the main one near the box top
TOWERS = [
    (30.0, -20.0, 2470.0, 270.0, 165.0),     # main tower
    (-360.0, 140.0, 2230.0, 220.0, 140.0),   # second
    (330.0, 220.0, 2070.0, 200.0, 125.0),    # third
    (-170.0, -360.0, 1950.0, 180.0, 115.0),
    (330.0, -330.0, 1870.0, 165.0, 105.0),
    (90.0, 400.0, 1820.0, 160.0, 100.0),
    (-500.0, -170.0, 1760.0, 140.0, 90.0),
]

# billow generations: n, radius range (m), sink (fraction of r inside the parent), y_lo, y_full, up_bias
LEVELS = [
    (110, (75.0, 130.0), 0.40, BASE_Y + 120, BASE_Y + 500, 0.65),
    (420, (38.0, 68.0), 0.34, BASE_Y + 60, BASE_Y + 400, 0.6),
    (1500, (17.0, 30.0), 0.28, BASE_Y + 40, BASE_Y + 300, 0.5),
]
CARVE_TOP, CARVE_LOW = 7.0, 40.0    # erosion carve depth (m)
BAND_TOP, BAND_LOW = 7.0, 15.0      # edge ramp width (m)
PROF_BASE = 0.85                    # interior density at the base (1.0 high up)
FRINGE = 0.25                       # peak density of the wispy fringe outside the surface


def grid_coords():
    nx, ny, nz = DIMS
    h = (BOX_MAX - BOX_MIN) / np.array(DIMS)
    xs = (BOX_MIN[0] + (np.arange(nx) + 0.5) * h[0]).astype(np.float32)
    ys = (BOX_MIN[1] + (np.arange(ny) + 0.5) * h[1]).astype(np.float32)
    zs = (BOX_MIN[2] + (np.arange(nz) + 0.5) * h[2]).astype(np.float32)
    return xs, ys, zs, h


class SmoothUnion:
    """Accumulates sum_i exp(-d_i / k) over the grid, sphere by sphere, within each sphere's bounding sub-box."""

    def __init__(self, wx, wy, wz, xs, ys, zs, h, k):
        self.wx, self.wy, self.wz = wx, wy, wz  # warped coordinates, shape (nz, ny, nx)
        self.xs, self.ys, self.zs, self.h, self.k = xs, ys, zs, h, k
        self.S = np.full(wx.shape, 1e-30, np.float64)
        self.warp_margin = 0.0

    def add(self, cx, cy, cz, r):
        k = self.k
        m = r + 7.0 * k + self.warp_margin
        i0 = max(0, int((cx - m - self.xs[0]) / self.h[0])); i1 = min(len(self.xs), int((cx + m - self.xs[0]) / self.h[0]) + 2)
        j0 = max(0, int((cy - m - self.ys[0]) / self.h[1])); j1 = min(len(self.ys), int((cy + m - self.ys[0]) / self.h[1]) + 2)
        l0 = max(0, int((cz - m - self.zs[0]) / self.h[2])); l1 = min(len(self.zs), int((cz + m - self.zs[0]) / self.h[2]) + 2)
        if i0 >= i1 or j0 >= j1 or l0 >= l1:
            return
        sl = (slice(l0, l1), slice(j0, j1), slice(i0, i1))
        d = np.sqrt((self.wx[sl] - cx) ** 2 + (self.wy[sl] - cy) ** 2 + (self.wz[sl] - cz) ** 2) - r
        self.S[sl] += np.exp(-np.clip(d, -40.0 * k, 60.0 * k) / k)

    def sdf(self):
        return (-self.k * np.log(self.S)).astype(np.float32)


def skeleton(rng):
    """Level-0 spheres (x, y, z, r): the base slab and the tower chains."""
    out = []
    for x, z, r in SLAB:
        out.append((x, BASE_Y + 0.25 * r, z, r))
    for x, z, top, rb, rt in TOWERS:
        y = BASE_Y + 0.55 * rb
        while True:
            f = np.clip((y - BASE_Y) / max(top - BASE_Y - rt, 1.0), 0.0, 1.0)
            r = rb + (rt - rb) * f ** 0.8
            if y + r >= top:
                y = top - r
                f = 1.0
            off = LEAN * (y - BASE_Y) + np.array([rng.normal(0, 0.12 * r), 0.0, rng.normal(0, 0.12 * r)])
            out.append((x + off[0], y, z + off[2], r))
            if f >= 1.0:
                break
            y += 0.55 * r
    return out


def surface_samples(sdf, xs, ys, zs, h, n, rng, r_range, sink, y_lo, y_full, up_bias, exclude_below):
    """Pick n sphere centres on the current iso-surface, biased by height and by upward-facing normals."""
    gz, gy, gx = np.gradient(sdf, h[2], h[1], h[0])
    near = np.abs(sdf) < 0.6 * h.min()
    idx = np.argwhere(near)  # (z, y, x)
    if len(idx) == 0:
        return []
    py = ys[idx[:, 1]]
    nxv = gx[near]; nyv = gy[near]; nzv = gz[near]
    nn = np.sqrt(nxv ** 2 + nyv ** 2 + nzv ** 2) + 1e-6
    nxv, nyv, nzv = nxv / nn, nyv / nn, nzv / nn
    w = (0.08 + smoothstep(y_lo, y_full, py)) * (1.0 - up_bias + up_bias * np.clip(0.5 + 0.5 * nyv, 0, 1) ** 2)
    w = w * (nyv > -0.35) * (py > exclude_below)
    w = w / w.sum()
    pick = rng.choice(len(idx), size=n, replace=True, p=w)
    out = []
    for p in pick:
        z, y, x = idx[p]
        r = rng.uniform(*r_range)
        # larger billows higher up (vigorous rising tops), smaller near the base
        r *= 0.6 + 0.55 * smoothstep(BASE_Y, BASE_Y + 800, ys[y])
        nvec = np.array([nxv[p], nyv[p], nzv[p]])
        c = np.array([xs[x], ys[y], zs[z]]) - nvec * sink * r
        while c[1] + r > TOP_LIMIT and r > 0.5 * r_range[0]:  # keep the crown inside the box
            r *= 0.85
            c = np.array([xs[x], ys[y], zs[z]]) - nvec * sink * r
        if c[1] + r > TOP_LIMIT:
            continue
        out.append((float(c[0]), float(c[1]), float(c[2]), float(r)))
    return out


def build_density(verbose=True):
    t0 = time.time()
    noise = Noise(SEED)
    rng = np.random.default_rng(SEED + 1)
    xs, ys, zs, h = grid_coords()
    X = xs[None, None, :]; Y = ys[None, :, None]; Z = zs[:, None, None]
    shape = (len(zs), len(ys), len(xs))

    # --- domain warp (large-scale irregularity of the sphere skeleton), ~30 m amplitude, 500 m wavelength
    L = 500.0
    wamp = 30.0
    wx = X + wamp * noise.perlin_fbm(X / L + 11.3, Y / L + 3.1, Z / L + 7.7, 2, 16)
    wy = Y + 0.5 * wamp * noise.perlin_fbm(X / L + 2.9, Y / L + 19.4, Z / L + 5.2, 2, 16)
    wz = Z + wamp * noise.perlin_fbm(X / L + 13.6, Y / L + 8.8, Z / L + 23.1, 2, 16)
    wx = np.broadcast_to(wx, shape).astype(np.float32)
    wy = np.broadcast_to(wy, shape).astype(np.float32)
    wz = np.broadcast_to(wz, shape).astype(np.float32)
    if verbose:
        print(f"  warp {time.time() - t0:.1f}s")

    su = SmoothUnion(wx, wy, wz, xs, ys, zs, h, K_SMOOTH)
    su.warp_margin = 2.2 * wamp
    for s in skeleton(rng):
        su.add(*s)
    sdf = su.sdf()
    for n, rr, sink, ylo, yfull, ub in LEVELS:
        cut = np.maximum(sdf, (BASE_Y - wy))
        for c in surface_samples(cut, xs, ys, zs, h, n, rng, rr, sink, ylo, yfull, ub, BASE_Y + 25):
            su.add(*c)
        sdf = su.sdf()
        if verbose:
            print(f"  level n={n} {time.time() - t0:.1f}s")

    # --- displacement noise on the distance field (breaks sphere regularity at medium scales)
    sdf = sdf + 10.0 * noise.perlin_fbm(X / 200.0 + 31.7, Y / 200.0 + 4.4, Z / 200.0 + 17.9, 3, 32)

    # --- flat, slightly ragged base
    base = BASE_Y + 6.0 * noise.perlin_fbm(X / 120.0 + 5.5, 0.0 * Y + 2.2, Z / 120.0 + 9.1, 3, 64)
    sdf = np.maximum(sdf, (base - Y).astype(np.float32))
    if verbose:
        print(f"  sdf done {time.time() - t0:.1f}s")

    # --- Perlin-Worley erosion, applied as an inward carve of the distance field (metres):
    #     billowy cracks (inverted Worley FBM) high up, wispy Perlin-Worley lower down
    hn = smoothstep(BASE_Y + 60.0, BASE_Y + 650.0, Y)  # 0 low on the flanks, 1 high up
    Fw = 1.0 / 90.0
    wor = noise.worley_fbm(X * Fw + 0.37, Y * Fw + 0.11, Z * Fw + 0.73, 64, 3)
    per = 0.5 + 0.5 * noise.perlin_fbm(X / 70.0 + 41.0, Y / 70.0 + 7.0, Z / 70.0 + 3.0, 4, 64)
    wispy = np.clip((per - (1.0 - wor) * 0.6) / 0.4, 0.0, 1.0)  # Perlin-Worley in [0, 1]
    carve = hn * (1.0 - wor) + (1.0 - hn) * (1.0 - wispy)
    sdf_e = sdf + carve * (CARVE_TOP + (CARVE_LOW - CARVE_TOP) * (1.0 - hn))
    if verbose:
        print(f"  erosion done {time.time() - t0:.1f}s")

    # --- crisp edge ramp (narrow on the billowed top, wider low down)
    band = BAND_TOP + (BAND_LOW - BAND_TOP) * (1.0 - hn)
    dens = smoothstep(0.0, 1.0, -sdf_e / band)

    # --- interior profile: denser with height above the base, mild low-frequency variation
    prof = PROF_BASE + (1.0 - PROF_BASE) * smoothstep(BASE_Y, BASE_Y + 500.0, Y)
    var = 0.9 + 0.1 * noise.perlin_fbm(X / 180.0 + 71.0, Y / 180.0 + 13.0, Z / 180.0 + 29.0, 2, 32)
    dens = dens * prof * var

    # --- tenuous wisps just outside the surface on the lower flanks (ragged, eroded edges)
    outside = np.clip(sdf_e, 0.0, None)
    fringe = FRINGE * (1.0 - hn) ** 1.5 * np.clip(2.0 * wispy - 0.6, 0.0, 1.0) * np.clip(1.0 - outside / 45.0, 0.0, 1.0)
    fringe = fringe * (Y > base + 3.0)
    dens = np.maximum(dens, fringe)

    # keep two empty voxels at every box face
    dens[:2] = 0; dens[-2:] = 0; dens[:, :2] = 0; dens[:, -2:] = 0; dens[:, :, :2] = 0; dens[:, :, -2:] = 0
    dens = np.clip(dens, 0.0, 1.0).astype(np.float32)
    if verbose:
        print(f"  density done {time.time() - t0:.1f}s; occupied frac {np.mean(dens > 0.002):.3f}, mean d(occ) {dens[dens > 0.002].mean():.3f}")
    return dens  # (nz, ny, nx)


def quantise(dens):
    return np.floor(dens * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
