"""Build work/network.json by reasoning: every weight is a physics formula or a chosen literal.

Run:  ./python3 work/build_network.py
No image is read here.  See work/NETWORK.md for the account of every unit.

Notation (KIT section 4/5): x = sample point, v = view ray direction (camera -> point), s = toward-sun
unit vector, mu = v.s.  The network outputs (a_sun, a_sky, a_gnd) per channel; the renderer multiplies them
by E_sun, L_sky, L_gnd.  All our outputs are grey (the medium is achromatic), so each triple repeats one value.
"""
import json, math

SIG = 0.04                   # sigma_max, 1/m
FOURPI = 4 * math.pi
G1, G2, W1 = 0.9, -0.3, 0.8  # phase function lobes


def hg_const(g):
    """log of the (1-g^2)/(4 pi) prefactor of HG(mu; g)."""
    return math.log((1 - g * g) / FOURPI)


# ----------------------------------------------------------------------------- probes
probes, P = [], {}


def probe(name, spec):
    P[name] = len(probes)
    probes.append(spec)


# (1) fine optical depth toward the sun: midpoint rule on geometrically growing segments.
#     The direct beam only matters for tau < ~6, i.e. the first few hundred metres.
#     Blur ~ 0.55 x segment length: wide enough to average the segment, narrow enough that a ray skimming past a
#     billow is not polluted by the air beside it (the forward glow must not appear on every billow edge).
fine_edges = [0, 10, 25, 50, 90, 150, 240, 370, 550, 800, 1150]
fine = []
for a, b in zip(fine_edges[:-1], fine_edges[1:]):
    n = f"fine_{a}_{b}"
    probe(n, {"type": "density", "frame": "light", "dir": [1, 0, 0], "dist": 0.5 * (a + b), "blur": max(0.55 * (b - a), 7)})
    fine.append((n, SIG * (b - a)))
# (2) coarse optical depth toward the sun at the diffusion scale: the blur (>= 150 m, a few transport mean
#     free paths) averages the sun exposure over the neighbourhood that feeds a point by diffusion.
#     Reach 1400 m covers the whole cloud.
cone_edges = [0, 60, 160, 320, 560, 900, 1400]
cone_blur = [150, 200, 280, 380, 500, 700]
cone = []
for (a, b), bl in zip(zip(cone_edges[:-1], cone_edges[1:]), cone_blur):
    n = f"cone_{a}_{b}"
    probe(n, {"type": "density", "frame": "light", "dir": [1, 0, 0], "dist": 0.5 * (a + b), "blur": bl})
    cone.append((n, SIG * (b - a)))
# (Tried and rejected, see NETWORK.md section 5: extra cones tilted 45 deg from s to find nearer sun-lit
#  boundaries.  The boundaries they reach are often not lit (the base's underside, terminator flanks, a base top
#  shadowed by a tower), and a single straight cone could not tell; net effect on the renders was negative.)
# (3) enclosure: box-filtered density around x at three scales
probe("E150", {"type": "density", "frame": "world", "dir": [0, 1, 0], "dist": 0, "blur": 150})
probe("E300", {"type": "density", "frame": "world", "dir": [0, 1, 0], "dist": 0, "blur": 300})
probe("E600", {"type": "density", "frame": "world", "dir": [0, 1, 0], "dist": 0, "blur": 600})
# (4) billow-scale density gradient along the sun direction (cosine of incidence on a lit face)
probe("s_minus", {"type": "density", "frame": "light", "dir": [-1, 0, 0], "dist": 50, "blur": 200})
probe("s_plus", {"type": "density", "frame": "light", "dir": [1, 0, 0], "dist": 50, "blur": 200})
# (5) billow-scale density gradient along the view ray (am I looking into the cloud from outside?)
probe("v_plus", {"type": "density", "frame": "view", "dir": [1, 0, 0], "dist": 50, "blur": 200})
probe("v_minus", {"type": "density", "frame": "view", "dir": [-1, 0, 0], "dist": 50, "blur": 200})
# (6) optical depth straight up (sky) and straight down (ground)
ud_edges = [0, 15, 40, 90, 180, 350, 650]
ud_blur = [15, 25, 50, 90, 170, 300]
up, dn = [], []
for (a, b), bl in zip(zip(ud_edges[:-1], ud_edges[1:]), ud_blur):
    n = f"up_{a}_{b}"
    probe(n, {"type": "density", "frame": "world", "dir": [0, 1, 0], "dist": 0.5 * (a + b), "blur": bl})
    up.append((n, SIG * (b - a)))
    n = f"dn_{a}_{b}"
    probe(n, {"type": "density", "frame": "world", "dir": [0, -1, 0], "dist": 0.5 * (a + b), "blur": bl})
    dn.append((n, SIG * (b - a)))
# (7) vertical large-scale gradient (top surface vs base)
probe("y_below", {"type": "density", "frame": "world", "dir": [0, -1, 0], "dist": 75, "blur": 300})
probe("y_above", {"type": "density", "frame": "world", "dir": [0, 1, 0], "dist": 75, "blur": 300})
# (8) scalars
probe("mu", {"type": "cos_view_sun"})
probe("vy", {"type": "view_up"})
probe("sy", {"type": "sun_up"})
probe("one", {"type": "const", "value": 1.0})

# ----------------------------------------------------------------------------- layer builder
layers = []          # list of dicts with W, b, act
names = [dict(P)]    # name -> index per layer (layer 0 = probes)


def add_layer(units):
    """units: list of (name, act, {input_name: weight}, bias)."""
    prev = names[-1]
    W, b, act, idx = [], [], [], {}
    for name, a, ins, bias in units:
        row = [0.0] * len(prev)
        for k, w in ins.items():
            row[prev[k]] += w
        W.append(row); b.append(bias); act.append(a); idx[name] = len(idx)
    layers.append({"W": W, "b": b, "act": act})
    names.append(idx)


def lin(name, ins, bias=0.0):
    return (name, "linear", ins, bias)


def passthru(*ns):
    return [(n, "linear", {n: 1.0}, 0.0) for n in ns]


EPS = 1e-3
TAU_E, P_E = 11.5, 2.77      # diffusion decay g = 1 / (1 + gamma ((tau + 0.5)/TAU_E)^P_E) of the 'deep' terms
GRAZE = 0.6                  # gamma = 1 + GRAZE (1 - s_y) for the sun (light travelling along the flat base leaks)
LEAK = 0.6                   # the forward glow's decay rate is multiplied by 1 + LEAK * |G_up| (1 - s_y)(1 - |G_s|):
                             # a beam skimming along the base under a low sun loses its glow sideways

# ----------------------------------------------------------------------------- layer 1
GRAD = 200 / (2 * 50)     # (rho(x-d) - rho(x+d)) * blur / (2 d): ~ cos(angle to the surface normal) at a sharp edge
GRAD_UP = 300 / (2 * 75)  # the same for the vertical gradient probes (blur 300, dist 75)
L1 = [
    lin("tau_f", dict(fine)),                       # fine optical depth toward the sun
    lin("tau_c", dict(cone)),                       # diffusion-scale optical depth toward the sun
    lin("tau_up", dict(up)),
    lin("tau_dn", dict(dn)),
    ("log_tf", "log", dict(fine), 0.01),            # log(tau_f + 0.01), for the tau e^-0.55 tau glow build-up
    ("log_tc", "log", dict(cone), 0.5),             # log(tau_c + 0.5)
    ("log_tu", "log", dict(up), 0.5),
    ("log_td", "log", dict(dn), 0.5),
    ("lg9", "log", {"mu": -2 * G1}, 1 + G1 * G1),   # log(1 + g^2 - 2 g mu), g = 0.9
    ("lgm3", "log", {"mu": -2 * G2}, 1 + G2 * G2),  # g = -0.3
    ("lg8", "log", {"mu": -2 * 0.8}, 1 + 0.64),     # g = 0.8 (forward glow of orders 2-5)
    lin("mu", {"mu": 1.0}),
    lin("vy", {"vy": 1.0}),
    lin("Gs", {"s_minus": GRAD, "s_plus": -GRAD}),                 # +1 on the sun-lit cap, -1 on the far side
    # pieces of h = min(1, 2|Gs|): 1 on lit and dark faces, 0 at the terminator (where the outflow vanishes)
    ("aGs", "abs", {"s_minus": GRAD, "s_plus": -GRAD}, 0.0),
    ("rGs_p", "relu", {"s_minus": 2 * GRAD, "s_plus": -2 * GRAD}, -1.0),    # relu(2 Gs - 1)
    ("rGs_m", "relu", {"s_minus": -2 * GRAD, "s_plus": 2 * GRAD}, -1.0),    # relu(-2 Gs - 1)
    lin("Gv", {"v_plus": GRAD, "v_minus": -GRAD}),                 # +1 when looking into the cloud from outside
    ("rGv", "relu", {"v_plus": GRAD, "v_minus": -GRAD}, 0.0),
    ("rGv2", "relu", {"v_plus": 2 * GRAD, "v_minus": -2 * GRAD}, -1.0),     # relu(2 Gv - 1), to cap 2 relu(Gv) at 1
    ("logE600", "log", {"E600": 1.0}, EPS),
    # grazing-sun factor on the diffusion decay: light that travels along the cloud's horizontal base (low sun)
    # leaks through both faces and dies faster than light crossing it (puck MC: 0.37 -> 0.1 from tau 12 to 22 at
    # 8 deg, against 0.5 at tau 14 when lit at 55 deg).  log(1 + GRAZE (1 - s_y)) with GRAZE = 0.6.
    ("lgr", "log", {"sy": -GRAZE}, 1.0 + GRAZE),
    ("log_1msy", "log", {"sy": -1.0}, 1.0 + EPS),                                 # log(1 - s_y)
    ("aGup", "abs", {"y_below": GRAD_UP, "y_above": -GRAD_UP}, 0.0),             # |G_up|
    # boundary-sink factor: fluence rises inward from a free surface; for the cloud's fuzzy boundary the rise is
    # gentler than the sharp-edged MC shapes' (delta+z0)/z0, and a billow (~100 m, tau ~ 4) is lit nearly
    # uniformly (sphere MC): 0.7 at E150=0.3, 1 at 0.5, 1.15 at 0.6, 1.75 at 1
    ("depth", "relu", {"E150": 1.5}, 0.25),
    ("clos", "relu", {"E600": 2.0}, -1.0),                         # closedness: 0 at a surface, 1 >= 300 m inside
    # hemisphere seen by a surface, as a fraction of the irradiance on the facing surface:
    # sky: 1 on tops, 0.7 on vertical flanks (the horizon-bright sky gives a vertical face 0.6-0.85 of E_sky),
    #      0 on the base;  ground (Lambertian, uniform): 1 on the base, 0.5 on flanks, 0 on tops.
    ("sky_a", "relu", {"y_below": 0.7 * GRAD_UP, "y_above": -0.7 * GRAD_UP}, 0.7),    # relu(0.7 + 0.7 Gup)
    ("sky_b", "relu", {"y_below": 0.7 * GRAD_UP, "y_above": -0.7 * GRAD_UP}, -0.3),   # relu(0.7 Gup - 0.3): caps at 1
    ("gndside", "relu", {"y_below": -0.5 * GRAD_UP, "y_above": 0.5 * GRAD_UP}, 0.5),  # relu(0.5 - 0.5 Gup)
]
add_layer(L1)

# ----------------------------------------------------------------------------- layer 2
A_FR = 0.3                   # decay rate (per unit tau_f) of the fresh light's broad part
C_LOBE, A_LOBE = 0.52, 0.55  # forward glow: 0.52 HG(mu; 0.8) tau_f exp(-0.55 tau_f)  (slab MC, orders 1-5 at mu = 1)
L2 = [
    # direct sun, single scattering: T p(mu) = exp(-tau_f) [0.8 HG(.9) + 0.2 HG(-.3)]  (weights 0.8/0.2 in the output)
    ("dir9", "exp", {"tau_f": -1.0, "lg9": -1.5}, hg_const(G1)),
    ("dirm3", "exp", {"tau_f": -1.0, "lgm3": -1.5}, hg_const(G2)),
    # logs of the leak factor's pieces (see layer 3)
    ("log_aGup", "log", {"aGup": 1.0}, EPS),
    ("log_1mGs", "log", {"aGs": -1.0}, 1.0 + EPS),                                # log(1 - |G_s|)
    # logs for the products formed in later exp units
    ("log_rGv", "log", {"rGv": 2.0, "rGv2": -1.0}, EPS),                  # log(min(1, 2 relu(Gv)))
    ("log_h", "log", {"aGs": 2.0, "rGs_p": -1.0, "rGs_m": -1.0}, EPS),     # log(min(1, 2|Gs|))
    ("log_depth", "log", {"depth": 1.0}, EPS),
    ("log_open", "log", {"clos": -1.0}, 1.0 + EPS),                  # log(1 - clos)
    ("log_skyside", "log", {"sky_a": 1.0, "sky_b": -1.0}, EPS),      # log(min(1, 0.7 + 0.7 Gup))
    ("log_gndside", "log", {"gndside": 1.0}, EPS),                   # log(relu(0.5 - 0.5 Gup))
    # (gamma (tau + 0.5)/TAU_E)^P_E: the argument of the diffusion decay 1/(1 + ...) (see layers 3-4)
    ("pw_c", "exp", {"log_tc": P_E, "lgr": 1.0}, -P_E * math.log(TAU_E)),
    ("pw_u", "exp", {"log_tu": P_E}, -P_E * math.log(TAU_E)),
    ("pw_d", "exp", {"log_td": P_E}, -P_E * math.log(TAU_E)),
] + passthru("tau_f", "log_tf", "lg8", "log_1msy", "tau_c", "tau_up", "tau_dn", "lgm3", "mu", "vy", "Gs", "Gv",
             "logE600")
add_layer(L2)

# ----------------------------------------------------------------------------- layer 3
L3 = [
    ("bvp", "exp", {"log_rGv": 1.0, "log_h": 1.0}, 0.0),             # min(1,2relu(Gv)) * min(1,2|Gs|): outflow seen by the camera
    ("tleak", "exp", {"log_tf": 1.0, "log_aGup": 1.0, "log_1msy": 1.0, "log_1mGs": 1.0}, 0.0),   # tau_f * leak
    ("lg1p_c", "log", {"pw_c": 1.0}, 1.0),                            # log(1 + pw): minus this = log g
    ("lg1p_u", "log", {"pw_u": 1.0}, 1.0),
    ("lg1p_d", "log", {"pw_d": 1.0}, 1.0),
] + passthru("dir9", "dirm3", "log_tf", "lg8",
             "log_depth", "log_open", "log_skyside", "log_gndside",
             "tau_f", "tau_c", "tau_up", "tau_dn", "lgm3", "mu", "vy", "Gs", "Gv", "logE600")
add_layer(L3)

# ----------------------------------------------------------------------------- layer 4
# diffuse (orders >= 6) sun light, per unit E_sun: fluence = lit + deep, S = fluence/(4 pi) * boost
A_LIT, LIT_POW600, ALPHA_LIT, BETA_GS = 1.3, 0.6, 0.15, 2.0
# deep: K * depth * E600^0.3 / (1 + gamma ((tau_c+0.5)/11.5)^2.77).  Nearly flat to tau_c ~ 6 (lumpy
# inter-reflection and slab transmission measured in the references and the puck MC), then a smooth power law:
# noon (gamma 1.11) 0.57 at tau_c = 10, 0.13 at 22; afternoon (1.35) 0.2 at 17, 0.065 at 27; golden (1.52)
# 0.43 at 11; sunset (1.58) 0.63 at 8.
K_DEEP, DEEP_POW600 = 1.0, 0.3
KV, KMU = 0.5, 0.11                        # outflow boost exp(0.5 relu(Gv) min(1,2|Gs|)), flow-along-s exp(0.11 mu)
# sky / ground: irradiance on an open surface per unit mean radiance: sky ~ 2.5 L_sky (horizon-bright sky lights
# flanks as much as tops), ground = pi L_gnd on the base, pi/2 on flanks (folded into the 'side' factors).
IRR_SKY, IRR_GND = 2.5, math.pi
FLU_PER_IRR = 3.0                           # fluence just inside an open surface lit by an extended source (slab MC)
SIZE_REF = 0.3 ** LIT_POW600                # E600^0.6 of a typical cloud surface point, normalises the lit terms
K_SKY, K_GND = 10.0, 8.6                    # deep terms carry most of the extended-source light (it diffuses everywhere)
EXT_POW600 = 0.6                            # thin wisps have no multiple-scattering build-up: scale as E600^0.6
A_SKY = (FLU_PER_IRR * IRR_SKY - SIZE_REF * K_SKY) / SIZE_REF   # SIZE_REF*K: what the deep term gives at the surface
A_GND = (FLU_PER_IRR * IRR_GND - SIZE_REF * K_GND) / SIZE_REF
KV_EXT = 0.5                                # outflow boost for extended sources: exp(0.5 Gv)
SINGLE_HEMI, SINGLE_VY = 0.45, 0.62         # int of p over a hemisphere: 0.84 forward, 0.16 back ~ 0.45 exp(+-0.62 vy)
A_SINGLE = 1.3                               # hemispheric transmittance ~ exp(-1.3 tau) (2 E3(tau))
FR_ISO, FR_BACK, FR_DEPTH = 0.033, 0.21, 0.7   # fresh light needs optical thickness to build up: x depth^0.7
L4 = [
    # fresh (orders 1-5): isotropic + backscatter lobe HG(-0.3), decaying as exp(-0.3 tau_f), scaled by depth^0.7
    ("fr_iso", "exp", {"tau_f": -A_FR, "log_depth": FR_DEPTH}, math.log(FR_ISO)),
    ("fr_back", "exp", {"tau_f": -A_FR, "lgm3": -1.5, "log_depth": FR_DEPTH}, math.log(FR_BACK) + hg_const(G2)),
    # forward glow of the low orders: builds up linearly, peaks at tau_f ~ 1.8, dies by tau_f ~ 12; decays faster
    # (x (1 + 0.6 leak)) when the beam skims along a boundary
    ("lobe", "exp", {"log_tf": 1.0, "tau_f": -A_LOBE, "tleak": -A_LOBE * LEAK, "lg8": -1.5},
     math.log(C_LOBE) + hg_const(0.8)),
    ("sun_lit", "exp", {"logE600": LIT_POW600, "Gs": BETA_GS, "tau_c": -ALPHA_LIT, "bvp": KV, "mu": KMU},
     math.log(A_LIT) - BETA_GS),
    ("sun_deep", "exp", {"log_depth": 1.0, "logE600": DEEP_POW600, "lg1p_c": -1.0, "bvp": KV, "mu": KMU},
     math.log(K_DEEP)),
    ("sky_single", "exp", {"vy": SINGLE_VY, "log_open": 1.0, "log_skyside": 1.0, "tau_up": -A_SINGLE},
     math.log(SINGLE_HEMI)),
    ("sky_lit", "exp", {"log_open": 1.0, "log_skyside": 1.0, "logE600": LIT_POW600, "Gv": KV_EXT}, math.log(A_SKY)),
    ("sky_deep", "exp", {"log_depth": 1.0, "logE600": EXT_POW600, "lg1p_u": -1.0, "Gv": KV_EXT}, math.log(K_SKY)),
    ("gnd_single", "exp", {"vy": -SINGLE_VY, "log_open": 1.0, "log_gndside": 1.0, "tau_dn": -A_SINGLE},
     math.log(SINGLE_HEMI)),
    ("gnd_lit", "exp", {"log_open": 1.0, "log_gndside": 1.0, "logE600": LIT_POW600, "Gv": KV_EXT}, math.log(A_GND)),
    ("gnd_deep", "exp", {"log_depth": 1.0, "logE600": EXT_POW600, "lg1p_d": -1.0, "Gv": KV_EXT}, math.log(K_GND)),
] + passthru("dir9", "dirm3")
add_layer(L4)

# ----------------------------------------------------------------------------- output layer
a_sun = {"dir9": W1, "dirm3": 1 - W1, "fr_iso": 1.0, "fr_back": 1.0, "lobe": 1.0,
         "sun_lit": 1 / FOURPI, "sun_deep": 1 / FOURPI}
a_sky = {"sky_single": 1.0, "sky_lit": 1 / FOURPI, "sky_deep": 1 / FOURPI}
a_gnd = {"gnd_single": 1.0, "gnd_lit": 1 / FOURPI, "gnd_deep": 1 / FOURPI}
out = []
for c in "RGB":
    out.append(lin(f"a_sun_{c}", a_sun))
for c in "RGB":
    out.append(lin(f"a_sky_{c}", a_sky))
for c in "RGB":
    out.append(lin(f"a_gnd_{c}", a_gnd))
add_layer(out)

net = {"probes": probes, "layers": layers,
       "_unit_names": [list(n.keys()) for n in names[1:]]}
json.dump(net, open("work/network.json", "w"), indent=1)
n_par = sum(len(l["W"]) * len(l["W"][0]) + len(l["b"]) for l in layers)
print("wrote work/network.json:", len(probes), "probes, layers", [len(l["W"]) for l in layers], n_par, "params")
print("A_SKY %.2f K_SKY %.2f A_GND %.2f K_GND %.2f" % (A_SKY, K_SKY, A_GND, K_GND))
