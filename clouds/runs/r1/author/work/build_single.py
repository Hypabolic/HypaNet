"""Single-scattering-only network: a_sun = T_sun(x) * p(mu), a_sky = a_gnd = 0.

Used only as a tool check and a baseline.  T_sun is estimated by a midpoint
rule along the toward-sun direction with geometrically growing segments.
"""
import json, math

SIG = 0.04                     # sigma_max (1/m)
G1, G2, W1 = 0.9, -0.3, 0.8    # phase function

# segments [a, b] along s; probe at the midpoint with blur ~ segment length
edges = [0, 12, 30, 60, 100, 160, 240, 340, 470, 640]
probes, lens = [], []
for a, b in zip(edges[:-1], edges[1:]):
    mid, ln = 0.5 * (a + b), b - a
    probes.append({"type": "density", "frame": "light", "dir": [1, 0, 0],
                   "dist": mid, "blur": max(ln, 7.0)})
    lens.append(ln)
n_d = len(probes)
probes.append({"type": "cos_view_sun"})
i_mu = n_d
n_in = len(probes)

# layer 1: tau (linear), log(1+g^2-2 g mu) for both g
W1_ = []; b1 = []; act1 = []
W1_.append([SIG * l for l in lens] + [0.0]); b1.append(0.0); act1.append("linear")   # tau
for g in (G1, G2):
    row = [0.0] * n_in; row[i_mu] = -2 * g
    W1_.append(row); b1.append(1 + g * g); act1.append("log")
# layer 2: exp(-tau - 1.5 log u + log((1-g^2)/4pi)) = T * HG
W2 = []; b2 = []
for k, g in enumerate((G1, G2)):
    row = [-1.0, 0.0, 0.0]; row[1 + k] = -1.5
    W2.append(row); b2.append(math.log((1 - g * g) / (4 * math.pi)))
# layer 3: 9 outputs
W3 = [[0.0, 0.0] for _ in range(9)]
for c in range(3):
    W3[c] = [W1, 1 - W1]
net = {"probes": probes,
       "layers": [{"W": W1_, "b": b1, "act": act1},
                  {"W": W2, "b": b2, "act": "exp"},
                  {"W": W3, "b": [0.0] * 9, "act": "linear"}]}
json.dump(net, open("work/single.json", "w"), indent=1)
print("wrote work/single.json", n_in, "probes")
