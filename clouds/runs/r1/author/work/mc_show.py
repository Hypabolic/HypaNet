"""Print tables from the MC results.

./python3 work/mc_show.py slab NAME        S(depth, view angle) per order class, phi, J
./python3 work/mc_show.py sphere NAME      S(r, cos theta, view) etc.
"""
import sys
import numpy as np

np.set_printoptions(linewidth=200, precision=4, suppress=True)
kind, name = sys.argv[1], sys.argv[2]
d = np.load(f"work/mc_out/{name}.npz")
S, PHI, J = d["S"], d["PHI"], d["J"]
angs = d["angs"]
SIG = 0.04

if kind == "slab":
    z = d["z"]; tau = z * SIG
    print(f"slab D={d['D']} theta0={d['theta0']} refl {d['refl']:.4f} trans {d['trans']:.4f}")
    # view angles: a=0 -> v=+z (looking along the beam: backlit, mu=+1); a=180 -> v=-z (sun side, mu=-1)
    sel = [0, 2, 4, 6, 8, 10, 12]   # 0,30,60,90,120,150,180 deg
    print("view angle a (deg) from +z:", angs[sel], "   mu = cos(a)")
    print("\nS total (all orders) per unit irradiance, rows = depth tau, cols = view angle")
    St = S.sum(0)
    for i in range(len(z)):
        if i % 2 == 0 or len(z) < 20:
            print(f"tau={tau[i]:6.2f} " + " ".join(f"{St[i, j]:8.4f}" for j in sel) +
                  f"   phi={PHI[:, i].sum():.4f}  Jz={J[:, i, 2].sum():.4f}  phi_direct={PHI[0, i]:.4f}")
    print("\nS by order class at a few depths (cols = view angles)")
    for i in [0, 1, 2, 4, 8, 12, 16, 24, 32, 40, 48]:
        if i >= len(z):
            continue
        print(f"tau={tau[i]:6.2f}")
        for c, lab in enumerate(["order0", "order1", "2-5", "6-30", ">30"]):
            print(f"   {lab:7s} " + " ".join(f"{S[c, i, j]:8.4f}" for j in sel) + f"   phi={PHI[c, i]:.4f}")
    print("\nDiffuse (orders>=1) S vs angle, normalised by phi_diffuse/(4pi):")
    for i in [0, 1, 2, 4, 8, 16, 32, 48]:
        if i >= len(z):
            continue
        Sd = S[1:, i].sum(0); pd = PHI[1:, i].sum()
        print(f"tau={tau[i]:6.2f} " + " ".join(f"{Sd[j] / (pd / (4 * np.pi)):8.3f}" for j in range(0, 24, 2)) +
              f"   |J|/phi={np.linalg.norm(J[1:, i].sum(0)) / pd:.3f}")
else:
    r = d["r"]; ct = d["ct"]; R = float(d["R"])
    print(f"sphere R={R} tau_R={R * SIG:.1f} escaped up {d['up']:.4f} down {d['down']:.4f}")
    St = S.sum(0); Pt = PHI.sum(0)
    print("\nfluence phi (all orders) rows = r/R, cols = cos theta (-1 bottom/dark ... +1 top/lit)")
    print("        " + " ".join(f"{c:7.2f}" for c in ct))
    for i in range(len(r)):
        print(f"r={r[i] / R:5.2f} " + " ".join(f"{Pt[i, j]:7.3f}" for j in range(len(ct))))
    print("\nfluence direct only")
    for i in range(len(r)):
        print(f"r={r[i] / R:5.2f} " + " ".join(f"{PHI[0, i, j]:7.3f}" for j in range(len(ct))))
    print("\nfluence diffuse (orders>=1)")
    for i in range(len(r)):
        print(f"r={r[i] / R:5.2f} " + " ".join(f"{PHI[1:, i, j].sum():7.3f}" for j in range(len(ct))))
    for a_idx, lab in [(0, "v=+z (backlit, mu=+1)"), (12, "v=-z (front lit, mu=-1)"), (6, "v=+x (outward side, mu=0)"),
                       (18, "v=-x (inward side, mu=0)"), (24, "v=+y (tangential, mu=0)")]:
        print(f"\nS total for {lab}")
        for i in range(len(r)):
            print(f"r={r[i] / R:5.2f} " + " ".join(f"{St[i, j, a_idx]:7.4f}" for j in range(len(ct))))
        print(f"S diffuse-only for {lab}")
        for i in range(len(r)):
            print(f"r={r[i] / R:5.2f} " + " ".join(f"{S[1:, i, j, a_idx].sum():7.4f}" for j in range(len(ct))))
