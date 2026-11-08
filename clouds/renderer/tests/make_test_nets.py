"""Harness-side test networks for the interpreter (NOT for the author kit).

    uv run python renderer/tests/make_test_nets.py      -> renderer/tests/nets/*.json

single_scatter.json  a_sun ~= T_sun(x) p(dot(v, s)) from 63 light-frame density probes along s (midpoint rule,
                     12 m apart, 756 m) and cos_view_sun; HG lobes built with log / exp units. Should render close to
                     render.py --mode exact-single (differences: 756 m truncation and the 12 m quadrature).
activations.json     every activation on a mix of probes (all frames, blur levels, scalar probes), outputs a
                     weighted sum; used by test_interpreter.py to check the shader against a numpy forward pass.
const.json           a_sun = a_sky = a_gnd = 0.05: S = 0.05 (E_sun + L_sky + L_gnd).
"""

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "nets"
SIGMA = 0.04


def hg_const(g):
    return (1 - g * g) / (4 * math.pi)


def single_scatter():
    n, step = 63, 12.0
    probes = [{"type": "density", "frame": "light", "dir": [1, 0, 0], "dist": (k + 0.5) * step, "blur": 0}
              for k in range(n)]
    probes.append({"type": "cos_view_sun"})
    P = len(probes)
    # layer 0: tau (linear), log(1 + g1^2 - 2 g1 mu), log(1 + g2^2 - 2 g2 mu)
    W0 = [[-SIGMA * step] * n + [0.0],
          [0.0] * n + [-2 * 0.9],
          [0.0] * n + [-2 * -0.3]]
    b0 = [0.0, 1 + 0.81, 1 + 0.09]
    # layer 1: exp(-tau - 1.5 log(..) + log(w c)) for both lobes
    W1 = [[1.0, -1.5, 0.0], [1.0, 0.0, -1.5]]
    b1 = [math.log(0.8 * hg_const(0.9)), math.log(0.2 * hg_const(-0.3))]
    # layer 2: 9 outputs, a_sun rgb = sum of lobes
    W2 = [[1.0, 1.0]] * 3 + [[0.0, 0.0]] * 6
    b2 = [0.0] * 9
    return {"probes": probes, "layers": [
        {"W": W0, "b": b0, "act": ["linear", "log", "log"]},
        {"W": W1, "b": b1, "act": "exp"},
        {"W": W2, "b": b2, "act": "linear"}]}


def activations():
    probes = [
        {"type": "density", "frame": "light", "dir": [1, 0, 0], "dist": 30, "blur": 0},
        {"type": "density", "frame": "world", "dir": [0, 1, 0], "dist": 50, "blur": 20},
        {"type": "density", "frame": "view", "dir": [0.6, 0.8, 0], "dist": 25, "blur": 100},
        {"type": "density", "frame": "light", "dir": [0, 0, 1], "dist": 400, "blur": 600},
        {"type": "density", "frame": "world", "dir": [0, 0, 1], "dist": 0, "blur": 893},
        {"type": "cos_view_sun"}, {"type": "sun_up"}, {"type": "view_up"}, {"type": "height"},
        {"type": "const", "value": 0.7},
    ]
    acts = ["linear", "relu", "exp", "log", "tanh", "sigmoid", "softplus", "abs", "sin", "square", "sqrt", "recip"]
    P = len(probes)
    W0, b0 = [], []
    for j, a in enumerate(acts):
        row = [((i * 7 + j * 3) % 11 - 5) / 4.0 for i in range(P)]
        W0.append(row)
        b0.append(((j * 5) % 7 - 3) / 3.0)
    # second layer, 13 units (odd size, tests padding) mixing everything
    W1 = [[((i * 3 + j * 5) % 9 - 4) / 8.0 for i in range(len(acts))] for j in range(13)]
    b1 = [0.1 * j - 0.5 for j in range(13)]
    acts1 = ["tanh", "softplus", "sigmoid", "relu", "linear", "abs", "sin", "sqrt", "square", "exp", "log", "recip", "linear"]
    W2 = [[0.02 * (((i + j) % 5) - 2) for i in range(13)] for j in range(9)]
    b2 = [0.05] * 9
    return {"probes": probes, "layers": [
        {"W": W0, "b": b0, "act": acts},
        {"W": W1, "b": b1, "act": acts1},
        {"W": W2, "b": b2, "act": "softplus"}]}


def const():
    return {"probes": [{"type": "const", "value": 1.0}],
            "layers": [{"W": [[0.05]] * 9, "b": [0.0] * 9, "act": "linear"}]}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("single_scatter", single_scatter), ("activations", activations), ("const", const)]:
        (OUT / f"{name}.json").write_text(json.dumps(fn()))
        print(OUT / f"{name}.json")
