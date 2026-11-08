"""Interpreter self-checks (harness side; not part of the author kit).

    uv run python renderer/tests/make_test_nets.py && uv run python renderer/tests/test_interpreter.py

1. Forward pass: for samples recorded by the shader's debug path (inspect), a float64 numpy forward pass of the
   network from the recorded probe values must reproduce every recorded unit activation and S.
2. Probes: scalar probes and raw (blur 0) density probes in all three frames recomputed in numpy from the
   density grid (GL trilinear convention) must match; blurred probes must be close to a numpy box filter.
3. inspect vs render: the pixel radiance rebuilt from inspect's records equals render.py --frames 1 at that pixel.
4. Interpreter vs exact (SPEC 6.3 style): tests/nets/single_scatter.json renders close to --mode exact-single.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))
from harness_client import DEFAULT_SCENE_DIR, Harness, resolve_pair  # noqa: E402
from inspect_ray import run_inspect  # noqa: E402

NETS = HERE / "nets"
ACTS = {
    "linear": lambda x: x, "relu": lambda x: np.maximum(x, 0), "exp": lambda x: np.exp(np.clip(x, -80, 80)),
    "log": lambda x: np.log(np.maximum(x, 1e-8)), "tanh": np.tanh,
    "sigmoid": lambda x: 1 / (1 + np.exp(-np.clip(x, -80, 80))),
    "softplus": lambda x: np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x))), "abs": np.abs, "sin": np.sin,
    "square": lambda x: x * x, "sqrt": lambda x: np.sqrt(np.maximum(x, 0)), "recip": lambda x: 1 / np.maximum(x, 1e-6),
}
failures = []


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


def forward(net, h0):
    h = np.asarray(h0, np.float64)
    outs = []
    for L in net["layers"]:
        W = np.array(L["W"], np.float32).astype(np.float64)
        b = np.array(L["b"], np.float32).astype(np.float64)
        z = W @ h + b
        acts = L["act"] if isinstance(L["act"], list) else [L["act"]] * len(b)
        h = np.array([ACTS[a](z[j]) for j, a in enumerate(acts)])
        # float32 sin of a large argument is meaningless on any device; mark such units as not comparable
        bad = np.array([a == "sin" and abs(z[j]) > 50 for j, a in enumerate(acts)])
        outs.append((h, bad))
    return outs


class Grid:
    def __init__(self, scene_dir):
        meta = json.loads((scene_dir / "density.json").read_text())
        nx, ny, nz = meta["dims"]
        self.d = np.fromfile(scene_dir / "density.u8", np.uint8).reshape(nz, ny, nx).astype(np.float64) / 255
        self.n = np.array([nx, ny, nz], float)
        self.bmin = np.array(meta["box_min"])
        self.bmax = np.array(meta["box_max"])
        self.vs = np.array(meta["voxel_size"])

    def q(self, x):
        return (np.asarray(x) - self.bmin) / self.vs

    def trilinear(self, x):
        q = self.q(x)
        if (q < 0).any() or (q > self.n).any():
            return 0.0
        u = q - 0.5
        i0 = np.floor(u).astype(int)
        f = u - i0
        val = 0.0
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    ix, iy, iz = (np.clip(i0 + [dx, dy, dz], 0, self.n.astype(int) - 1))
                    w = (f[0] if dx else 1 - f[0]) * (f[1] if dy else 1 - f[1]) * (f[2] if dz else 1 - f[2])
                    val += w * self.d[iz, iy, ix]
        return val

    def box(self, x, w):
        """Mean of the piecewise-constant grid over a centred box w voxels wide (outside = 0)."""
        q = self.q(x)
        if (q < 0).any() or (q > self.n).any():
            return 0.0
        lo, hi = q - w / 2, q + w / 2
        weights = []
        for a in range(3):
            n = int(self.n[a])
            i = np.arange(n)
            weights.append(np.clip(np.minimum(hi[a], i + 1) - np.maximum(lo[a], i), 0, None))
        wx, wy, wz = weights
        return float(np.einsum("zyx,x,y,z->", self.d, wx, wy, wz) / w ** 3)


def camera(shot):
    pos = np.array(shot["position"], float)
    fwd = np.array(shot["look_at"], float) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, shot["up"])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return pos, fwd, right, up


def main():
    grid = Grid(DEFAULT_SCENE_DIR)
    vm = grid.vs.mean()
    with Harness(harness_mode=True) as h:
        shots = h.shots
        # ---- 1 + 2: activations net, several pixels/pairs ----
        net = json.loads((NETS / "activations.json").read_text())
        h.set_network(json.dumps(net))
        worst_act, worst_probe, worst_blur, nrec, nskip = 0.0, 0.0, 0.0, 0, 0
        for pid, px, py in [("front_noon", 330, 170), ("backlit_golden", 300, 140), ("below_golden", 320, 120),
                            ("side_sunset", 250, 200)]:
            shot, preset, ex, _ = resolve_pair(shots, pid)
            r = run_inspect(h, shot, preset, px, py)
            names, rec = r["names"], r["records"]
            P = len(net["probes"])
            s = np.array(shots["presets"][preset]["toward_sun"])
            u = np.cross(s, [0, 1, 0]); u /= np.linalg.norm(u)
            w = np.cross(u, s)
            _, _, _, camup = camera(shots["shots"][shot])
            v = np.array([r["summary"]["dir.x"], r["summary"]["dir.y"], r["summary"]["dir.z"]])
            e2 = np.cross(v, camup); e2 /= np.linalg.norm(e2)
            e3 = np.cross(e2, v)
            for row in rec[:: max(1, len(rec) // 12)]:
                nrec += 1
                probes = row[8 : 8 + P]
                outs = forward(net, probes)
                got = row[8 + P : 8 + P + sum(len(o[0]) for o in outs)]
                ref = np.concatenate([o[0] for o in outs])
                skip = np.concatenate([o[1] for o in outs])
                if outs[-1][1].any() or (len(outs) > 1 and skip[: -9].any()):
                    nskip += 1
                    continue  # downstream of a large-argument sin: not comparable
                err = np.abs(got - ref) / (np.abs(ref) + 1e-4)
                worst_act = max(worst_act, float(err.max()))
                x = row[1:4]
                # scalar probes
                ref_p = {5: v @ s, 6: s[1], 7: v[1], 8: (x[1] - grid.bmin[1]) / (grid.bmax[1] - grid.bmin[1]), 9: 0.7}
                for i, val in ref_p.items():
                    worst_probe = max(worst_probe, abs(probes[i] - val))
                # raw density probe 0: light frame (1,0,0) dist 30
                worst_probe = max(worst_probe, abs(probes[0] - grid.trilinear(x + 30 * s)))
                # blurred probes vs a numpy box filter (approximate by construction)
                for i, (d, dist, blur) in {1: (np.array([0, 1, 0]), 50, 20),
                                           2: (0.6 * v + 0.8 * e2, 25, 100),
                                           3: (w, 400, 600)}.items():
                    worst_blur = max(worst_blur, abs(probes[i] - grid.box(x + dist * d, blur / vm)))
        check("forward pass (numpy float64 vs shader float32, all activations)", worst_act < 1e-3,
              f"max relative error {worst_act:.2e} over {nrec - nskip} samples ({nskip} skipped: sin of |x| > 50 upstream)")
        check("scalar + raw density probes", worst_probe < 3e-3, f"max abs error {worst_probe:.2e}")
        check("blurred density probes ~ box filter", worst_blur < 0.05,
              f"max abs difference {worst_blur:.3f} (pyramid levels interpolate; approximate by design)")

        # ---- 3: inspect vs render at frame 0 ----
        net_ss = (NETS / "single_scatter.json").read_text()
        h.set_network(net_ss)
        worst = 0.0
        for pid, px, py in [("front_noon", 330, 170), ("backlit_golden", 300, 140), ("side_noon", 250, 150)]:
            shot, preset, ex, _ = resolve_pair(shots, pid)
            lin, _, _ = h.render(shot=shot, preset=preset, mode="net", frames=1, budgetMs=40,
                                 exposure=ex)
            r = run_inspect(h, shot, preset, px, py)
            a, b = lin[py, px], r["L"]
            worst = max(worst, float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-6)))
        check("inspect L == render --frames 1 pixel", worst < 1e-4, f"max relative difference {worst:.2e}")

        # ---- 4: single-scatter net vs exact mode ----
        for pid in ["side_noon", "backlit_golden"]:
            shot, preset, ex, _ = resolve_pair(shots, pid)
            kw = dict(shot=shot, preset=preset, frames=16, budgetMs=40, exposure=ex,
                      width=320, height=200)
            h.set_network(net_ss)
            a, _, _ = h.render(mode="net", noBackground=True, **kw)
            b, _, _ = h.render(mode="exact", noBackground=True, **kw)
            rel = np.abs(a - b).mean() / b.mean()
            ratio = a.reshape(-1, 3).mean(0) / b.reshape(-1, 3).mean(0)
            check(f"single_scatter net vs exact, no background ({pid})", rel < 0.03,
                  f"mean |net - exact| / mean exact = {rel:.4f}; mean ratio {np.round(ratio, 4).tolist()}")
    print("ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
