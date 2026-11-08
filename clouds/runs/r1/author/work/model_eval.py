"""Evaluate a candidate closed-form model of S (per unit sun irradiance) on the MC shapes, using
probe-style features computed exactly as the renderer would (box-blurred density = fraction of a cube
inside the shape, optical depths by midpoint sums of blurred probes along the sun direction).

The model parameters are literals chosen by reasoning (see NETWORK.md); this script only *shows*
model vs MC, it never fits.

./python3 work/model_eval.py sphere_R500 [rows]
./python3 work/model_eval.py slab_D500_t0
"""
import sys
import numpy as np

SIG = 0.04
G1, G2, W1 = 0.9, -0.3, 0.8


def hg(mu, g):
    return (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * mu) ** 1.5)


def phase(mu):
    return W1 * hg(mu, G1) + (1 - W1) * hg(mu, G2)


# ------------------------------------------------------------------ model parameters (chosen, not fitted)
P = dict(
    fine_edges=[0, 10, 25, 50, 90, 150, 240, 370, 550],
    cone_edges=[0, 60, 160, 320, 560, 900, 1400],
    cone_blur=[60, 100, 160, 240, 340, 500],
    enc_blur=[150, 300, 600],
    gs_blur=300, gs_dist=75,
    gv_blur=300, gv_dist=75,
    # fresh (orders 1-5): isotropic part + backscatter lobe (HG -0.3) + forward lobe (HG 0.8)
    F_iso=0.03, F_back=0.19, F_alpha=0.3, F_lobe=0.39, F_lobe_g=0.8, F_lobe_build=1.3,
    # "lit cap" diffuse fluence (orders >= 6): size through E600, cosine law through exp(beta (G_s - 1))
    A_lit=1.25, lit_pow600=0.6, alpha_lit=0.1, beta_gs=2.0,
    # "deep" diffuse fluence: power law in the coarse optical depth toward the sun, depth rise through E150
    K_deep=1.9, deep_tau0=3.0, deep_depth_slope=2.7, deep_depth_off=0.35, deep_pow600=0.3,
    # angular factor for the diffuse light (orders >= 6): exp(k2 mu^2 + k1 mu)
    k2=0.58, k1=0.11,
)

rng = np.random.default_rng(0)
CUBE = rng.random((3000, 3)) - 0.5


class Shape:
    def __init__(self, kind, size):
        self.kind, self.size = kind, size

    def inside(self, q):
        if self.kind == "sphere":
            return np.linalg.norm(q, axis=-1) < self.size
        return (q[..., 2] > 0) & (q[..., 2] < self.size)      # slab: z in [0, D]

    def blurred(self, p, b):
        if b <= 7:
            return self.inside(p[None, :]).astype(float)[0]
        return self.inside(p[None, :] + b * CUBE).mean()

    def tau_exact(self, p, s):
        # optical depth from p toward the sun direction s (unit), exact for uniform shapes
        t = np.linspace(0, 4000, 8001)
        q = p[None, :] + t[:, None] * s[None, :]
        return SIG * self.inside(q).mean() * 4000


def features(shape, p, s, v):
    f = {}
    f["tau_exact"] = shape.tau_exact(p, s)
    e = P["fine_edges"]
    f["tau_f"] = sum(SIG * (b - a) * shape.blurred(p + 0.5 * (a + b) * s, max(b - a, 7)) for a, b in zip(e[:-1], e[1:]))
    e = P["cone_edges"]
    f["tau_c"] = sum(SIG * (b - a) * shape.blurred(p + 0.5 * (a + b) * s, bl)
                     for a, b, bl in zip(e[:-1], e[1:], P["cone_blur"]))
    for b in P["enc_blur"]:
        f[f"E{b}"] = shape.blurred(p, b)
    d, b = P["gs_dist"], P["gs_blur"]
    f["G_s"] = (shape.blurred(p - d * s, b) - shape.blurred(p + d * s, b)) * b / (2 * d)
    d, b = P["gv_dist"], P["gv_blur"]
    # positive when the density increases along the view ray (camera outside looking in)
    f["G_v"] = (shape.blurred(p + d * v, b) - shape.blurred(p - d * v, b)) * b / (2 * d)
    f["mu"] = float(np.dot(v, s))
    return f


def model(f):
    mu = f["mu"]
    direct = np.exp(-f["tau_f"]) * phase(mu)
    a = P["F_alpha"]
    fresh = np.exp(-a * f["tau_f"]) * (P["F_iso"] + P["F_back"] * hg(mu, -0.3)) + \
        (np.exp(-a * f["tau_f"]) - np.exp(-P["F_lobe_build"] * f["tau_f"])) * P["F_lobe"] * hg(mu, P["F_lobe_g"])
    phi_lit = P["A_lit"] * max(f["E600"], 1e-3) ** P["lit_pow600"] * np.exp(-P["alpha_lit"] * f["tau_c"]) * \
        np.exp(P["beta_gs"] * (f["G_s"] - 1))
    depth = max(P["deep_depth_slope"] * f["E150"] - P["deep_depth_off"], 0.0)
    phi_deep = P["K_deep"] * depth * max(f["E600"], 1e-3) ** P["deep_pow600"] / (f["tau_c"] + P["deep_tau0"])
    Sd = (phi_lit + phi_deep) / (4 * np.pi) * np.exp(P["k2"] * mu * mu + P["k1"] * mu)
    return direct, fresh, Sd


name = sys.argv[1]
d = np.load(f"work/mc_out/{name}.npz")
S = d["S"]
if name.startswith("sphere"):
    R = float(d["R"]); shape = Shape("sphere", R)
    r, ct = d["r"], d["ct"]
    s = np.array([0, 0, 1.0])
    rows = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.95, 0.75, 0.55, 0.05]
    views = {"front": (12, np.array([0, 0, -1.0])), "back": (0, np.array([0, 0, 1.0])),
             "in": (18, np.array([-1.0, 0, 0])), "tan": (24, np.array([0, 1.0, 0]))}
    print(f"{name} tau_R={R * SIG:.0f}.  columns: MC diffuse S (orders>=1) | model fresh+Sd  [ratio model/MC]")
    print("  r/R  cos  tau_f  tau_c  E150 E300 E600  G_s |  front: MC  model ratio |  back |  in |  tan")
    for rr in rows:
        i = int(np.argmin(np.abs(r / R - rr)))
        for j in range(len(ct)):
            st = np.sqrt(1 - ct[j] ** 2)
            p = np.array([r[i] * st, 0.0, r[i] * ct[j]])
            line = ""
            for k, (lab, (vi, v)) in enumerate(views.items()):
                f = features(shape, p, s, v)
                direct, fresh, Sd = model(f)
                mc = S[1:, i, j, vi].sum()
                mod = fresh + Sd
                if k == 0:
                    line = (f"{r[i] / R:5.2f} {ct[j]:5.2f} {f['tau_f']:6.2f} {f['tau_c']:6.2f}  {f['E150']:.2f} {f['E300']:.2f} "
                            f"{f['E600']:.2f} {f['G_s']:5.2f} |")
                line += f" {mc:7.4f} {mod:7.4f} [{mod / mc:4.2f}] |"
            print(line)
        print()
else:
    D = float(d["D"]); th = float(d["theta0"]); shape = Shape("slab", D)
    z = d["z"]
    t0 = np.radians(th)
    s = -np.array([np.sin(t0), 0, np.cos(t0)])
    angs = list(d["angs"])
    # views: by angle a: v = (sin a, 0, cos a)
    sel = [0, 60, 90, 180, 240, 270] if th > 0 else [0, 90, 180]
    print(f"{name}: theta0={th}. views by angle a (v=(sin a,0,cos a)); mu = v.s")
    hdr = "  tau   tau_f  tau_c  E150 E300 E600  G_s |" + "".join(f"  a={a:3d}: MC   model [rat] |" for a in sel)
    print(hdr)
    for i in range(0, len(z), 2):
        p = np.array([0, 0, z[i]])
        line = ""
        for k, a in enumerate(sel):
            v = np.array([np.sin(np.radians(a)), 0, np.cos(np.radians(a))])
            f = features(shape, p, s, v)
            direct, fresh, Sd = model(f)
            mc = S[1:, i, angs.index(a)].sum()
            mod = fresh + Sd
            if k == 0:
                line = (f"{z[i] * SIG:5.2f} {f['tau_f']:6.2f} {f['tau_c']:6.2f}  {f['E150']:.2f} {f['E300']:.2f} {f['E600']:.2f} "
                        f"{f['G_s']:5.2f} |")
            line += f" {mc:7.4f} {mod:7.4f} [{mod / mc:4.2f}] |"
        print(line)
