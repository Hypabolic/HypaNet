"""Monte Carlo light transport in idealised shapes of the KIT medium.

Medium: sigma_t = 0.04 * d (1/m), albedo 0.999, phase 0.8 HG(0.9) + 0.2 HG(-0.3).
Shapes: a slab (density 1 between depth 0 and D) and a sphere (density 1 inside radius R).
Source: a collimated beam of irradiance 1 (per unit area normal to the beam).

Tally (collision estimator): at every collision at x with incoming travel direction w and
weight wt we add  wt * p(w . (-v))  to the bin of x for a list of view directions v
(v = direction from the camera toward the point, as in KIT section 4).  Dividing by
sigma_t * V_bin gives  S(x, v) = int p(w.(-v)) L(x, w) dw  per unit incident irradiance,
i.e. exactly a_sun as the renderer uses it.  Also tallies the fluence phi and the flux
vector J.  Scattering orders are tallied separately: 0 = unscattered (the direct beam),
1 = once scattered, 2 = 2..5 scatterings, 3 = 6..30, 4 = more than 30.

Usage:  ./python3 work/mc.py slab D_metres theta0_deg N
        ./python3 work/mc.py sphere R_metres N
Writes work/mc_out/<name>.npz
"""
import sys, time, os
import numpy as np

SIG, ALB = 0.04, 0.999
G1, G2, W1 = 0.9, -0.3, 0.8


def hg(mu, g):
    return (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * mu) ** 1.5)


def phase(mu):
    return W1 * hg(mu, G1) + (1 - W1) * hg(mu, G2)


def sample_mu(n, rng):
    u = rng.random(n)
    g = np.where(rng.random(n) < W1, G1, G2)
    s = (1 - g * g) / (1 - g + 2 * g * u)
    return np.clip((1 + g * g - s * s) / (2 * g), -1, 1)


def scatter(w, rng):
    n = len(w)
    mu = sample_mu(n, rng)
    phi = 2 * np.pi * rng.random(n)
    st = np.sqrt(np.maximum(0.0, 1 - mu * mu))
    a = np.where(np.abs(w[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    u = np.cross(a, w)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(w, u)
    return w * mu[:, None] + st[:, None] * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v)


def order_class(k):
    return np.where(k == 0, 0, np.where(k == 1, 1, np.where(k <= 5, 2, np.where(k <= 30, 3, 4))))


# ---------------------------------------------------------------- slab
def run_slab(D, theta0_deg, N, nb=50, seed=1):
    """Slab z in [0, D]; beam enters at z=0 travelling along (sin t0, 0, cos t0)."""
    rng = np.random.default_rng(seed)
    t0 = np.radians(theta0_deg)
    # view directions v: in the x-z plane, angle a from +z (a = 0: v = +z = along the beam -> backlit)
    angs = np.arange(0, 360, 15)
    V = np.stack([np.sin(np.radians(angs)), np.zeros_like(angs, dtype=float), np.cos(np.radians(angs))], 1)
    S = np.zeros((5, nb, len(V)))
    PHI = np.zeros((5, nb)); J = np.zeros((5, nb, 3))
    refl = 0.0; trans = 0.0
    pos = np.zeros((N, 3)); w = np.tile(np.array([np.sin(t0), 0, np.cos(t0)]), (N, 1))
    wt = np.full(N, np.cos(t0) / N); k = np.zeros(N, int)
    it = 0
    while len(wt) > 0 and it < 20000:
        it += 1
        ell = -np.log(rng.random(len(wt))) / SIG
        pos = pos + ell[:, None] * w
        z = pos[:, 2]
        out_r = z < 0; out_t = z > D
        refl += wt[out_r].sum(); trans += wt[out_t].sum()
        alive = ~(out_r | out_t)
        pos, w, wt, k = pos[alive], w[alive], wt[alive], k[alive]
        if len(wt) == 0:
            break
        b = np.minimum((pos[:, 2] / D * nb).astype(int), nb - 1)
        oc = order_class(k)
        mu_v = -(w @ V.T)                      # w . (-v)
        P = phase(mu_v) * wt[:, None]
        for c in range(5):
            m = oc == c
            if m.any():
                np.add.at(S[c], b[m], P[m])
                np.add.at(PHI[c], b[m], wt[m])
                np.add.at(J[c], b[m], wt[m, None] * w[m])
        w = scatter(w, rng); wt = wt * ALB; k = k + 1
        # Russian roulette on very low weight
        low = wt < 1e-4 / N
        if low.any():
            keep = rng.random(len(wt)) < 0.5
            wt = np.where(low, np.where(keep, wt * 2, 0.0), wt)
            m = wt > 0
            pos, w, wt, k = pos[m], w[m], wt[m], k[m]
    dz = D / nb
    norm = SIG * dz
    return dict(S=S / norm, PHI=PHI / norm, J=J / norm, z=(np.arange(nb) + 0.5) * dz, angs=angs, V=V,
                refl=refl, trans=trans, D=D, theta0=theta0_deg, iters=it)


# ---------------------------------------------------------------- sphere
def run_sphere(R, N, nr=10, nc=12, seed=1):
    """Sphere of radius R at the origin; beam travels along -z (sun toward +z).
    Bins: r/R in nr uniform bins, cos(theta) = z/r in nc uniform bins.
    Collisions are rotated about z so the position lies in the x-z half plane (x >= 0).
    View directions: in the x-z plane at angles a from +z (v = (sin a, 0, cos a)), every 15 deg,
    plus v = +y (index -1, tangential)."""
    rng = np.random.default_rng(seed)
    angs = np.arange(0, 360, 15)
    V = np.stack([np.sin(np.radians(angs)), np.zeros_like(angs, dtype=float), np.cos(np.radians(angs))], 1)
    V = np.vstack([V, [[0, 1.0, 0]]])
    S = np.zeros((5, nr, nc, len(V)))
    PHI = np.zeros((5, nr, nc)); J = np.zeros((5, nr, nc, 3))
    # start on the top hemisphere surface
    rho = R * np.sqrt(rng.random(N)); ph = 2 * np.pi * rng.random(N)
    x0, y0 = rho * np.cos(ph), rho * np.sin(ph)
    z0 = np.sqrt(np.maximum(0, R * R - rho * rho)) - 1e-6
    pos = np.stack([x0, y0, z0], 1); w = np.tile(np.array([0, 0, -1.0]), (N, 1))
    wt = np.full(N, np.pi * R * R / N); k = np.zeros(N, int)
    escaped_w = []; escaped_wt = []
    it = 0
    while len(wt) > 0 and it < 50000:
        it += 1
        ell = -np.log(rng.random(len(wt))) / SIG
        pos = pos + ell[:, None] * w
        r = np.linalg.norm(pos, axis=1)
        out = r > R
        if out.any():
            escaped_w.append(w[out]); escaped_wt.append(wt[out])
        alive = ~out
        pos, w, wt, k, r = pos[alive], w[alive], wt[alive], k[alive], r[alive]
        if len(wt) == 0:
            break
        ir = np.minimum((r / R * nr).astype(int), nr - 1)
        ct = pos[:, 2] / np.maximum(r, 1e-9)
        ic = np.minimum(((ct + 1) / 2 * nc).astype(int), nc - 1)
        rxy = np.hypot(pos[:, 0], pos[:, 1])
        cph = np.where(rxy > 1e-9, pos[:, 0] / np.maximum(rxy, 1e-9), 1.0)
        sph = np.where(rxy > 1e-9, pos[:, 1] / np.maximum(rxy, 1e-9), 0.0)
        wr = np.stack([w[:, 0] * cph + w[:, 1] * sph, -w[:, 0] * sph + w[:, 1] * cph, w[:, 2]], 1)
        oc = order_class(k)
        mu_v = -(wr @ V.T)
        P = phase(mu_v) * wt[:, None]
        flat = ir * nc + ic
        for c in range(5):
            m = oc == c
            if m.any():
                np.add.at(S[c].reshape(nr * nc, -1), flat[m], P[m])
                np.add.at(PHI[c].reshape(nr * nc), flat[m], wt[m])
                np.add.at(J[c].reshape(nr * nc, 3), flat[m], wt[m, None] * wr[m])
        w = scatter(w, rng); wt = wt * ALB; k = k + 1
        low = wt < 1e-4 * np.pi * R * R / N
        if low.any():
            keep = rng.random(len(wt)) < 0.5
            wt = np.where(low, np.where(keep, wt * 2, 0.0), wt)
            m = wt > 0
            pos, w, wt, k = pos[m], w[m], wt[m], k[m]
    re = np.linspace(0, R, nr + 1); ce = np.linspace(-1, 1, nc + 1)
    vol = (2 * np.pi / 3) * (re[1:] ** 3 - re[:-1] ** 3)[:, None] * np.diff(ce)[None, :]
    norm = SIG * vol
    ew = np.concatenate(escaped_w); ewt = np.concatenate(escaped_wt)
    # escaping power up (toward the sun, z>0) and down
    up = ewt[ew[:, 2] > 0].sum() / (np.pi * R * R); down = ewt[ew[:, 2] <= 0].sum() / (np.pi * R * R)
    return dict(S=S / norm[None, :, :, None], PHI=PHI / norm[None], J=J / norm[None, :, :, None],
                r=(re[1:] + re[:-1]) / 2, ct=(ce[1:] + ce[:-1]) / 2, angs=angs, V=V, R=R,
                up=up, down=down, iters=it)


if __name__ == "__main__":
    os.makedirs("work/mc_out", exist_ok=True)
    kind = sys.argv[1]
    t = time.time()
    if kind == "slab":
        D, th, N = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
        out = run_slab(D, th, N)
        name = f"slab_D{int(D)}_t{int(th)}"
        print(f"{name}: refl {out['refl']:.4f} trans {out['trans']:.4f} iters {out['iters']} ({time.time()-t:.1f}s)")
    else:
        R, N = float(sys.argv[2]), int(sys.argv[3])
        out = run_sphere(R, N)
        name = f"sphere_R{int(R)}"
        print(f"{name}: escaped up {out['up']:.4f} down {out['down']:.4f} iters {out['iters']} ({time.time()-t:.1f}s)")
    np.savez(f"work/mc_out/{name}.npz", **out)
    print("saved", name)
