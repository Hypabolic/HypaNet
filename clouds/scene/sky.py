"""Sun and sky model for HypaNet Clouds (SPEC section 3).

Sun: E_sun = 20 * T_atm(h), T_atm(h) = exp(-(beta_R * 8000 + beta_M_ext * 1200 + tau_O3) * m(h)), Kasten-Young air mass.

Sky: single-scattering Rayleigh + Mie atmosphere with ozone absorption (Nishita 1993 style, spherical planet), seen
from the cloud layer altitude H_OBS, lit by a solar irradiance of 20 (RGB, flat) at the top of the atmosphere. No sun
disk. Lower hemisphere: Lambertian ground, albedo 0.1, constant radiance (albedo / pi) * (E_sun * sin h + E_sky).

RGB channels stand for 680, 550 and 440 nm.

Equirectangular convention (identical to Mitsuba 3's `envmap` with an identity to_world, y up):
    image W x H = 1024 x 512, row 0 is the top (zenith), pixel centres at
    u = (col + 0.5) / W,  v = (row + 0.5) / H,  phi = 2 pi u,  theta = pi v   (theta = polar angle from +y)
    direction d = (sin(theta) sin(phi), cos(theta), -sin(theta) cos(phi))
  inverse: theta = acos(d.y), phi = atan2(d.x, -d.z) wrapped to [0, 2 pi).
  So u = 0 / 1 looks toward -z, u = 0.25 toward +x, u = 0.5 toward +z, u = 0.75 toward -x.
"""

from __future__ import annotations

import numpy as np

# ---- constants -------------------------------------------------------------------------------------------------
E_TOA = 20.0
BETA_R = np.array([5.802e-6, 13.558e-6, 33.100e-6])  # Rayleigh scattering = extinction, m^-1 (sea level)
BETA_M_SCA = 3.996e-6  # Mie scattering, m^-1 (sea level), spectrally flat
BETA_M_EXT = BETA_M_SCA * 1.11  # Mie extinction
H_R = 8000.0
H_M = 1200.0
MIE_G = 0.8
# Ozone absorption coefficient at the layer peak (Hillaire 2020 / Bruneton 2017 values at 680/550/440 nm), m^-1,
# tent profile 0 at 10 km -> 1 at 25 km -> 0 at 40 km (column-equivalent thickness 15 km).
SIGMA_O3 = np.array([0.650e-6, 1.881e-6, 0.085e-6])
O3_CENTER, O3_HALF = 25000.0, 15000.0
TAU_O3 = SIGMA_O3 * O3_HALF  # zenith ozone optical depth from the ground = (0.00975, 0.028215, 0.001275)

R_EARTH = 6_360_000.0
R_TOP = 6_460_000.0  # 100 km atmosphere top
H_OBS = 2000.0  # observer altitude for the sky map (middle of the cloud layer, 1500-2500 m)
GROUND_ALBEDO = 0.1

SKY_W, SKY_H = 1024, 512


def sun_dir(elev_deg: float, azim_deg: float) -> np.ndarray:
    """Toward-sun unit vector. Azimuth clockwise from north, north = -z, east = +x."""
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    return np.array([np.cos(e) * np.sin(a), np.sin(e), -np.cos(e) * np.cos(a)])


def kasten_young(elev_deg: float) -> float:
    h = elev_deg
    return 1.0 / (np.sin(np.radians(h)) + 0.50572 * (h + 6.07995) ** (-1.6364))


def T_atm(elev_deg: float) -> np.ndarray:
    tau = BETA_R * H_R + BETA_M_EXT * H_M + TAU_O3
    return np.exp(-tau * kasten_young(elev_deg))


def E_sun(elev_deg: float) -> np.ndarray:
    return E_TOA * T_atm(elev_deg)


# ---- atmosphere densities --------------------------------------------------------------------------------------
def dens_R(h):
    return np.exp(-np.maximum(h, 0.0) / H_R)


def dens_M(h):
    return np.exp(-np.maximum(h, 0.0) / H_M)


def dens_O(h):
    return np.maximum(0.0, 1.0 - np.abs(h - O3_CENTER) / O3_HALF)


def extinction(h):
    """(..., 3) extinction coefficient at altitude h (m)."""
    h = np.asarray(h)[..., None]
    return BETA_R * dens_R(h) + BETA_M_EXT * dens_M(h) + SIGMA_O3 * dens_O(h)


def ray_to_top(r, mu):
    """Distance from radius r along direction with cos(zenith) mu to the top of the atmosphere."""
    disc = r * r * (mu * mu - 1.0) + R_TOP * R_TOP
    return -r * mu + np.sqrt(np.maximum(disc, 0.0))


def hits_ground(r, mu):
    return (mu < 0.0) & (r * r * (mu * mu - 1.0) + R_EARTH * R_EARTH >= 0.0)


# ---- transmittance table to the top of the atmosphere ------------------------------------------------------------
class TransmittanceTable:
    """Optical depth (RGB) from altitude h along cos-zenith mu to space; inf-like when the ray hits the planet."""

    def __init__(self, nh=192, nmu=2048, nsteps=400):
        self.nh, self.nmu = nh, nmu
        self.hmax = R_TOP - R_EARTH
        uh = np.linspace(0.0, 1.0, nh)
        self.h = self.hmax * uh * uh
        self.mu = np.linspace(-1.0, 1.0, nmu)
        tau = np.zeros((nh, nmu, 3))
        s = (np.arange(nsteps) + 0.5) / nsteps
        for i, hh in enumerate(self.h):
            r = R_EARTH + hh
            L = ray_to_top(r, self.mu)  # (nmu,)
            # quadratic spacing, denser near the start of the ray
            t = L[:, None] * s[None, :] ** 2
            dt = L[:, None] * 2.0 * s[None, :] / nsteps
            rr = np.sqrt(r * r + t * t + 2.0 * r * t * self.mu[:, None])
            ext = extinction(rr - R_EARTH)  # (nmu, nsteps, 3)
            tau[i] = np.sum(ext * dt[..., None], axis=1)
            tau[i][hits_ground(r, self.mu)] = 1e4
        self.tau = tau

    def lookup(self, h, mu):
        """Bilinear lookup; h, mu arrays of the same shape -> (..., 3)."""
        uh = np.sqrt(np.clip(h, 0.0, self.hmax) / self.hmax) * (self.nh - 1)
        um = (np.clip(mu, -1.0, 1.0) + 1.0) * 0.5 * (self.nmu - 1)
        i0 = np.clip(np.floor(uh).astype(np.int64), 0, self.nh - 2); fi = (uh - i0)[..., None]
        j0 = np.clip(np.floor(um).astype(np.int64), 0, self.nmu - 2); fj = (um - j0)[..., None]
        T = self.tau
        return ((1 - fi) * ((1 - fj) * T[i0, j0] + fj * T[i0, j0 + 1]) + fi * ((1 - fj) * T[i0 + 1, j0] + fj * T[i0 + 1, j0 + 1]))


# ---- phase functions (angle between toward-sun and toward-viewer-source directions) ------------------------------
def phase_rayleigh(nu):
    return 3.0 / (16.0 * np.pi) * (1.0 + nu * nu)


def phase_mie(nu, g=MIE_G):
    # Cornette-Shanks
    return 3.0 / (8.0 * np.pi) * (1 - g * g) * (1 + nu * nu) / ((2 + g * g) * (1 + g * g - 2 * g * nu) ** 1.5)


def equirect_dirs(w=SKY_W, h=SKY_H):
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    phi = 2 * np.pi * u[None, :]
    theta = np.pi * v[:, None]
    d = np.stack(np.broadcast_arrays(np.sin(theta) * np.sin(phi), np.cos(theta), -np.sin(theta) * np.cos(phi)), axis=-1)
    return d, theta[:, 0]


def sky_radiance(dirs, s, table, nsteps=160):
    """Single-scattering sky radiance for upward unit view dirs (N, 3), toward-sun s, observer at H_OBS."""
    r0 = R_EARTH + H_OBS
    mu = dirs[:, 1]
    L = ray_to_top(r0, mu)
    k = (np.arange(nsteps) + 0.5) / nsteps
    t = L[:, None] * k[None, :] ** 2
    dt = L[:, None] * 2.0 * k[None, :] / nsteps
    # positions relative to the planet centre (observer at (0, r0, 0))
    px = dirs[:, None, 0] * t
    py = r0 + dirs[:, None, 1] * t
    pz = dirs[:, None, 2] * t
    rr = np.sqrt(px * px + py * py + pz * pz)
    hh = rr - R_EARTH
    mu_s = (px * s[0] + py * s[1] + pz * s[2]) / rr
    ext = extinction(hh)  # (N, n, 3)
    tau_v = np.cumsum(ext * dt[..., None], axis=1) - 0.5 * ext * dt[..., None]  # observer -> sample (midpoint)
    tau_s = table.lookup(hh, mu_s)
    T = np.exp(-(tau_v + tau_s))
    nu = dirs @ s
    sca = (BETA_R * dens_R(hh)[..., None] * phase_rayleigh(nu)[:, None, None]
           + BETA_M_SCA * dens_M(hh)[..., None] * phase_mie(nu)[:, None, None])
    return E_TOA * np.sum(sca * T * dt[..., None], axis=1)


def build_sky_map(s, E_sun_rgb, table):
    """Returns (img (H, W, 3) float32, L_sky (3,), L_gnd (3,), E_sky (3,))."""
    d, theta = equirect_dirs()
    img = np.zeros((SKY_H, SKY_W, 3))
    up_rows = np.where(theta < np.pi / 2)[0]
    for r0 in range(0, len(up_rows), 16):
        rows = up_rows[r0:r0 + 16]
        dd = d[rows].reshape(-1, 3)
        img[rows] = sky_radiance(dd, s, table).reshape(len(rows), SKY_W, 3)
    # hemispherical integrals over the upper half
    dtheta = np.pi / SKY_H
    dphi = 2 * np.pi / SKY_W
    th = theta[up_rows]
    dw = (np.sin(th) * dtheta * dphi)[:, None, None]
    up = img[up_rows]
    solid = np.sum(np.sin(th) * dtheta * dphi) * SKY_W
    L_sky = np.sum(up * dw, axis=(0, 1)) / solid
    E_sky = np.sum(up * dw * np.cos(th)[:, None, None], axis=(0, 1))
    E_down = E_sun_rgb * max(s[1], 0.0) + E_sky
    L_gnd = GROUND_ALBEDO / np.pi * E_down
    img[theta >= np.pi / 2] = L_gnd
    return img.astype(np.float32), L_sky, L_gnd, E_sky
