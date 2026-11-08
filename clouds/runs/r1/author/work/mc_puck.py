"""Monte Carlo in a 'puck' (vertical cylinder) of uniform density d, lit by a collimated beam at elevation h
coming from the -x side.  Delta tracking.  Tallies fluence and phase-weighted S on an (x, y) grid for |z| < ZW.

./python3 work/mc_puck.py R H d elev_deg N
Writes work/mc_out/puck_R{R}_H{H}_e{elev}.npz and prints a summary.
"""
import sys, time, os
import numpy as np

SIGMAX, ALB = 0.04, 0.999
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


def run(R, H, dens, elev, N, seed=1):
    rng = np.random.default_rng(seed)
    sig = SIGMAX * dens
    h = np.radians(elev)
    s = np.array([-np.cos(h), np.sin(h), 0.0])      # toward the sun
    w0 = -s                                           # beam travel direction
    # start on a rectangle perpendicular to the beam, 2.2R x (H + 2R) wide, 1.5 R before the puck centre
    e2 = np.array([0.0, 0.0, 1.0]); e3 = np.cross(w0, e2)
    L2, L3 = 2.2 * R, (H + 2 * R * np.sin(h) + 0.1 * R)
    a = (rng.random(N) - 0.5) * L2; b = (rng.random(N) - 0.5) * L3
    pos = -w0 * (R + H) + a[:, None] * e2[None, :] + b[:, None] * e3[None, :]
    w = np.tile(w0, (N, 1))
    wt = np.full(N, L2 * L3 / N)                     # irradiance 1 on the start plane
    k = np.zeros(N, int)
    # tally grid
    xe = np.linspace(-R, R, 15); ye = np.linspace(-H / 2, H / 2, 9); ZW = 150.0
    V = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, -1.0, 0], [-1.0, 0, 0], [0, 0, 1.0]])
    S = np.zeros((5, 14, 8, len(V))); PHI = np.zeros((5, 14, 8))
    it = 0
    esc = 0.0
    while len(wt) > 0 and it < 100000:
        it += 1
        ell = -np.log(rng.random(len(wt))) / sig
        pos = pos + ell[:, None] * w
        r2 = pos[:, 0] ** 2 + pos[:, 2] ** 2
        inside = (r2 < R * R) & (np.abs(pos[:, 1]) < H / 2)
        # outside the puck: free flight continues only if the ray can still hit the puck; simple rule:
        # photons that are outside and moving away from the bounding sphere escape
        far = (np.linalg.norm(pos, axis=1) > (R + H)) & ((pos * w).sum(1) > 0)
        esc += wt[far].sum()
        keep = ~far
        pos, w, wt, k = pos[keep], w[keep], wt[keep], k[keep]
        inside = inside[keep]
        if len(wt) == 0:
            break
        # null collision outside the puck (density 0): just continue; real collision inside
        real = inside
        if real.any():
            p = pos[real]; ww = w[real]; wr = wt[real]; kr = k[real]
            ix = np.clip(((p[:, 0] + R) / (2 * R) * 14).astype(int), 0, 13)
            iy = np.clip(((p[:, 1] + H / 2) / H * 8).astype(int), 0, 7)
            inz = np.abs(p[:, 2]) < ZW
            oc = order_class(kr)
            mu_v = -(ww @ V.T)
            P = phase(mu_v) * wr[:, None]
            flat = ix * 8 + iy
            for c in range(5):
                m = (oc == c) & inz
                if m.any():
                    np.add.at(S[c].reshape(14 * 8, -1), flat[m], P[m])
                    np.add.at(PHI[c].reshape(14 * 8), flat[m], wr[m])
            w[real] = scatter(ww, rng); wt[real] = wr * ALB; k[real] = kr + 1
        low = wt < 1e-4 * L2 * L3 / N
        if low.any():
            keepr = rng.random(len(wt)) < 0.5
            wt = np.where(low, np.where(keepr, wt * 2, 0.0), wt)
            m = wt > 0
            pos, w, wt, k = pos[m], w[m], wt[m], k[m]
    vol = (2 * R / 14) * (H / 8) * (2 * ZW)
    norm = sig * vol
    return dict(S=S / norm, PHI=PHI / norm, xe=xe, ye=ye, R=R, H=H, dens=dens, elev=elev, iters=it, V=V)


if __name__ == "__main__":
    os.makedirs("work/mc_out", exist_ok=True)
    R, H, dens, elev, N = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
    t = time.time()
    out = run(R, H, dens, elev, N)
    name = f"puck_R{int(R)}_H{int(H)}_e{int(elev)}"
    np.savez(f"work/mc_out/{name}.npz", **out)
    print(name, "iters", out["iters"], f"{time.time() - t:.1f}s")
    PHI = out["PHI"].sum(0); S = out["S"]
    xc = 0.5 * (out["xe"][1:] + out["xe"][:-1]); yc = 0.5 * (out["ye"][1:] + out["ye"][:-1])
    np.set_printoptions(linewidth=220, precision=3, suppress=True)
    print("fluence phi(x, y) per unit beam irradiance; rows x from the lit edge (-R) to the far edge, cols y bottom..top")
    for i in range(14):
        print(f"x={xc[i]:7.0f} " + " ".join(f"{PHI[i, j]:7.3f}" for j in range(8)))
    print("\nS for v = +y (camera below looking up), all orders")
    for i in range(14):
        print(f"x={xc[i]:7.0f} " + " ".join(f"{S[:, i, j, 0].sum():7.4f}" for j in range(8)))
    print("\nS for v = +x (camera on the sun side looking along the beam)")
    for i in range(14):
        print(f"x={xc[i]:7.0f} " + " ".join(f"{S[:, i, j, 1].sum():7.4f}" for j in range(8)))
    print("\nS for v = +z (camera at the side, mu = 0)")
    for i in range(14):
        print(f"x={xc[i]:7.0f} " + " ".join(f"{S[:, i, j, 4].sum():7.4f}" for j in range(8)))
