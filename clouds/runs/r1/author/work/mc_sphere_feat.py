"""For a sphere MC run, print per bin: tau_s (optical depth toward the sun), enclosure E_b at cube widths b
(fraction of a cube of width b centred on the point that lies inside the sphere), diffuse fluence and
diffuse S for the main views.  ./python3 work/mc_sphere_feat.py sphere_R500 [rows r/R list]
"""
import sys
import numpy as np

SIG = 0.04
name = sys.argv[1]
d = np.load(f"work/mc_out/{name}.npz")
S, PHI = d["S"], d["PHI"]
r, ct, R = d["r"], d["ct"], float(d["R"])
rows = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.95, 0.85, 0.75, 0.55, 0.35, 0.05]
rng = np.random.default_rng(0)
cube = rng.random((4000, 3)) - 0.5


def enclosure(p, b):
    q = p[None, :] + b * cube
    return (np.linalg.norm(q, axis=1) < R).mean()


def tau_s(p):
    rho2 = p[0] ** 2 + p[1] ** 2
    return SIG * (np.sqrt(max(R * R - rho2, 0)) - p[2])


# view index: 0 -> v=+z (backlit mu=+1), 12 -> v=-z (front, mu=-1), 6 -> +x outward, 18 -> -x inward, 24 -> +y
print(f"{name}: tau_R = {R * SIG:.1f}")
print(" r/R  cos   tau_s   E150  E300  E600 |  phi_d  phi_d/4pi | S_front  S_back  S_out  S_in  S_tan  (diffuse only) | S_back_total")
for rr in rows:
    i = int(np.argmin(np.abs(r / R - rr)))
    for j in range(len(ct)):
        st = np.sqrt(1 - ct[j] ** 2)
        p = np.array([r[i] * st, 0.0, r[i] * ct[j]])
        ts = tau_s(p)
        E = [enclosure(p, b) for b in (150, 300, 600)]
        pd = PHI[1:, i, j].sum()
        Sd = S[1:, i, j].sum(0)
        print(f"{r[i] / R:5.2f} {ct[j]:5.2f} {ts:6.2f}   {E[0]:.2f}  {E[1]:.2f}  {E[2]:.2f} | {pd:6.3f}  {pd / (4 * np.pi):7.4f} | "
              f"{Sd[12]:7.4f} {Sd[0]:7.4f} {Sd[6]:6.4f} {Sd[18]:6.4f} {Sd[24]:6.4f} | {S[:, i, j, 0].sum():7.4f}")
    print()
