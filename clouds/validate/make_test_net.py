"""Write the SPEC 6.3 test networks (harness only; they must never reach the author kit).

    uv run python validate/make_test_net.py

validate/test_networks/exact_single_net.json
    Reproduces exact sun single scattering, a_sun = T_sun(x) * p(dot(v, s)), a_sky = a_gnd = 0, from probes and
    activations only:
      probes 0..61  density, light frame, dir (1, 0, 0) (toward the sun), at the midpoints of 62 segments whose widths
                    grow geometrically from W0 m (ratio R) to cover MAX_DIST m; blur = segment width (a box the size
                    of the segment stands in for the line average over it)
      probe  62     cos_view_sun = mu
      layer 1 (3 units): u0 = linear( -sigma_t_max * sum_k w_k d_k ) = log T_sun
                         u1 = log( 1 + g1^2 - 2 g1 mu ),  g1 = 0.9
                         u2 = log( 1 + g2^2 - 2 g2 mu ),  g2 = -0.3
      layer 2 (2 units): e_i = exp( u0 - 1.5 u_i + log(c_i (1 - g_i^2) / 4 pi) ),  c = (0.8, 0.2)
                         = T_sun * c_i * HG(mu; g_i)
      layer 3 (9 outputs, linear): a_sun RGB = e1 + e2; a_sky = a_gnd = 0.
    exact_single_net_noblur.json is the same with blur 0 on every probe (plain midpoint samples): it isolates the
    interpreter from the blur approximation (a blur is a 3D box, so it also averages across the shadow ray).
    So it exercises density probes in the light frame with distance and blur, cos_view_sun, and linear/log/exp units
    with per-unit activation lists.

validate/test_networks/probe_net.json
    density probes only (world frame with 9 blur levels; raw probes 40 m away in the light / view / world frames)
    with identity units, so check_interpreter.py can compare every probe value with numpy.

validate/test_networks/act_<name>.json (one per activation)
    probes: height, view_up, cos_view_sun, density (world frame, dir (0, 1, 0), dist 0, blur 0), const 1.
    layer 1: 9 units of activation <name>, inputs spread over the range where the activation is interesting
    (including negative inputs for log / sqrt / recip and +-100 for exp / sigmoid to hit the +-80 clamp);
    layer 2: identity to the 9 outputs. check_interpreter.py compares every recorded unit with numpy.
"""

from __future__ import annotations

import json
import math

import numpy as np

from common import SCENE_OUT, VALIDATE

N_DENS = 62
W0 = 3.0
MAX_DIST = 2800.0
G = (0.9, -0.3)
C = (0.8, 0.2)
ACTS = ["linear", "relu", "exp", "log", "tanh", "sigmoid", "softplus", "abs", "sin", "square", "sqrt", "recip"]


def segment_ratio():
    lo, hi = 1.0001, 1.5
    for _ in range(200):
        r = 0.5 * (lo + hi)
        tot = W0 * (r ** N_DENS - 1) / (r - 1)
        lo, hi = (r, hi) if tot < MAX_DIST else (lo, r)
    return 0.5 * (lo + hi)


def exact_single_net(sigma):
    r = segment_ratio()
    w = W0 * r ** np.arange(N_DENS)
    edges = np.concatenate([[0.0], np.cumsum(w)])
    mid = 0.5 * (edges[:-1] + edges[1:])
    probes = [{"type": "density", "frame": "light", "dir": [1, 0, 0], "dist": round(float(m), 4),
               "blur": round(float(wk), 4)} for m, wk in zip(mid, w)]
    probes.append({"type": "cos_view_sun"})
    nin = len(probes)
    # layer 1
    W1 = np.zeros((3, nin)); b1 = np.zeros(3)
    W1[0, :N_DENS] = -sigma * w
    for i, g in enumerate(G):
        W1[1 + i, N_DENS] = -2 * g
        b1[1 + i] = 1 + g * g
    # layer 2
    W2 = np.zeros((2, 3)); b2 = np.zeros(2)
    for i, (g, c) in enumerate(zip(G, C)):
        W2[i, 0] = 1.0
        W2[i, 1 + i] = -1.5
        b2[i] = math.log(c * (1 - g * g) / (4 * math.pi))
    # layer 3
    W3 = np.zeros((9, 2)); W3[0:3, :] = 1.0
    net = {"_note": "HARNESS TEST NETWORK (validate/make_test_net.py): exact sun single scattering. Not for the kit.",
           "_segments": {"w0": W0, "ratio": r, "max_dist": float(edges[-1])},
           "probes": probes,
           "layers": [{"W": W1.tolist(), "b": b1.tolist(), "act": ["linear", "log", "log"]},
                      {"W": W2.tolist(), "b": b2.tolist(), "act": "exp"},
                      {"W": W3.tolist(), "b": [0.0] * 9, "act": "linear"}]}
    return net


# pre-activation = A * (probe mix) + B; probe mix j is one of the five probes (cycled), ranges per activation
RANGES = {"linear": (-3, 3), "relu": (-3, 3), "exp": (-6, 4), "log": (-0.5, 3), "tanh": (-4, 4), "sigmoid": (-6, 6),
          "softplus": (-6, 6), "abs": (-3, 3), "sin": (-12, 12), "square": (-3, 3), "sqrt": (-0.5, 4),
          "recip": (-0.5, 3)}
# probe ranges over the scene: height [0,1], view_up [-1,1], cos_view_sun [-1,1], density [0,1], const 1


def act_net(name):
    lo, hi = RANGES[name]
    probes = [{"type": "height"}, {"type": "view_up"}, {"type": "cos_view_sun"},
              {"type": "density", "frame": "world", "dir": [0, 1, 0], "dist": 0, "blur": 0},
              {"type": "const", "value": 1.0}]
    W = np.zeros((9, 5)); b = np.zeros(9)
    span = hi - lo
    # units 0..3: each probe mapped over [lo, hi]; units 4..6: mixes; units 7, 8: fixed constants / clamp tests
    W[0, 0] = span; b[0] = lo                    # height in [0,1]
    W[1, 1] = span / 2; b[1] = lo + span / 2     # view_up in [-1,1]
    W[2, 2] = span / 2; b[2] = lo + span / 2     # cos_view_sun in [-1,1]
    W[3, 3] = span; b[3] = lo                    # density in [0,1]
    W[4, :4] = [span / 3, span / 6, span / 6, span / 3]; b[4] = lo
    W[5, :4] = [-span / 4, span / 4, -span / 4, span / 4]; b[5] = lo + span / 2
    W[6, 2] = span / 2; W[6, 3] = span / 4; b[6] = lo + span / 4
    if name in ("exp", "sigmoid", "softplus"):
        W[7, 4] = 100.0; W[8, 4] = -100.0        # beyond the +-80 clamp
    elif name in ("log", "sqrt", "recip"):
        W[7, 4] = 0.0; W[8, 4] = -1.0            # exactly 0 and negative: the max(x, eps) guards
    else:
        W[7, 4] = hi; W[8, 4] = lo
    return {"_note": f"HARNESS TEST NETWORK (validate/make_test_net.py): activation identity test for '{name}'.",
            "probes": probes,
            "layers": [{"W": W.tolist(), "b": b.tolist(), "act": name},
                       {"W": np.eye(9).tolist(), "b": [0.0] * 9, "act": "linear"}]}


BLURS = [0.0, 7.0, 14.0, 28.0, 56.0, 112.0, 224.0, 448.0, 893.0]
FRAME_PROBES = [("light", [1, 0, 0]), ("light", [0, 1, 0]), ("light", [0, 0, 1]), ("light", [0.6, -0.8, 0]),
                ("view", [1, 0, 0]), ("view", [0, 1, 0]), ("view", [0, 0, 1]), ("view", [0, 0.6, -0.8]),
                ("world", [0, 0, -1])]
FRAME_DIST = 40.0


def probe_net():
    """Density probes only: world frame at the sample point with each blur in BLURS, then raw probes 40 m away in
    the light / view / world frames. One linear identity unit per probe, so inspect records every probe value."""
    probes = [{"type": "density", "frame": "world", "dir": [0, 1, 0], "dist": 0, "blur": b} for b in BLURS]
    probes += [{"type": "density", "frame": f, "dir": d, "dist": FRAME_DIST, "blur": 0} for f, d in FRAME_PROBES]
    n = len(probes)
    W2 = np.zeros((9, n))
    W2[np.arange(9), np.arange(9)] = 1.0
    return {"_note": "HARNESS TEST NETWORK (validate/make_test_net.py): density probe frames and blur levels.",
            "probes": probes,
            "layers": [{"W": np.eye(n).tolist(), "b": [0.0] * n, "act": "linear"},
                       {"W": W2.tolist(), "b": [0.0] * 9, "act": "linear"}]}


def noblur(net):
    n = json.loads(json.dumps(net))
    n["_note"] = n["_note"].replace("exact sun single scattering", "exact sun single scattering, blur 0 everywhere")
    for p in n["probes"]:
        if p["type"] == "density":
            p["blur"] = 0
    return n


def main():
    meta = json.loads((SCENE_OUT / "density.json").read_text())
    d = VALIDATE / "test_networks"
    d.mkdir(parents=True, exist_ok=True)
    net = exact_single_net(meta["sigma_t_max"])
    (d / "exact_single_net.json").write_text(json.dumps(net, indent=1))
    nparams = sum(len(L["b"]) * (1 + len(L["W"][0])) for L in net["layers"])
    print(f"exact_single_net.json: {len(net['probes'])} probes, segments w0 {W0} m x ratio "
          f"{net['_segments']['ratio']:.5f} to {net['_segments']['max_dist']:.0f} m, {nparams} parameters")
    (d / "exact_single_net_noblur.json").write_text(json.dumps(noblur(net), indent=1))
    (d / "probe_net.json").write_text(json.dumps(probe_net(), indent=1))
    print("exact_single_net_noblur.json (same, blur 0), probe_net.json (frames and blur levels)")
    for a in ACTS:
        (d / f"act_{a}.json").write_text(json.dumps(act_net(a), indent=1))
    print(f"act_<name>.json for {len(ACTS)} activations")


if __name__ == "__main__":
    main()
