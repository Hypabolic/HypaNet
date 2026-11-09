"""Compile a HypaNet Clouds network.json into straight-line GLSL plus bake passes (embed/DESIGN.md "Compile", "Bake").

    uv run python embed/compile/compile_net.py [NET.json] [--out embed/web/gen] [--no-bake] [--live p0,p1]
        [--res 128x80x128 | --res sun=128x80x128,static=96x60x96,static_coarse=64x40x64] [--coarse-blur 150]

    uv run python embed/compile/compile_net.py --live p0,p1,p2,p3      # the shipped r1 build (embed/web/gen)

Generic over any network the interpreter accepts (renderer/net.js validation). Steps:

1. Fold: const probes go into the biases; a unit whose inputs are all constant becomes a constant (float32, the
   interpreter's activation clamps); a linear unit with one input of weight 1 and bias 0 (layer >= 1) is an alias of
   its input; identical units of one layer (same activation, terms and bias) are computed once.
2. Prune: zero weights, and every unit and probe the 9 outputs do not reach.
3. Probes are classified by what they depend on: density probes in the world frame depend on x only (static), in the
   light frame on x and the sun (sun), in the view frame on the ray (live); cos_view_sun, view_up, sun_up and height
   are analytic and stay inline. --no-bake makes every density probe live; --live forces the listed ones live.
4. The pre-activation of every first-layer unit is split into one linear density combination per class plus the
   analytic terms. Combinations equal up to scale (tau_f and log(tau_f + 0.01); 2 (p19 - p20) and 4 (p19 - p20)) are
   merged. Static and sun combinations go to the channels of RGBA bake textures (grouped by class and resolution
   tier, finest first); live ones are computed inline.

Writes OUT/net_eval.glsl (evalNet, a drop-in for the interpreter's), OUT/bake.glsl (the bake samplers and one
`vec4 bakeK(vec3 x)` per texture) and OUT/manifest.json. The host contracts are in embed/DESIGN.md and at the top of
each generated file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

CLOUDS = Path(__file__).resolve().parents[2]
DEFAULT_NET = CLOUDS / "runs" / "r1" / "author" / "work" / "network.json"
DEFAULT_SCENE = CLOUDS / "scene" / "out"
DEFAULT_OUT = CLOUDS / "embed" / "web" / "gen"

# ---- renderer/net.js constants (keep in step) ----
ACTIVATIONS = ["linear", "relu", "exp", "log", "tanh", "sigmoid", "softplus", "abs", "sin", "square", "sqrt", "recip"]
PROBE_TYPES = ["density", "cos_view_sun", "sun_up", "view_up", "height", "const"]
FRAMES = ["light", "world", "view"]
LIMITS = {"probes": 64, "layers": 6, "units": 128, "params": 50000, "outputs": 9}
BLUR_LEVELS = 8
COMMENT_KEYS = {"name", "comment", "comments", "note", "notes", "description", "label", "doc"}
F32_MAX = 3.4028234663852886e38

# default bake grids per class / tier (nx, ny, nz); see --res. Tuned for r1 on the reference split
# (embed/tests/eval_compiled.py): the final r1 build is `compile_net.py --live p0,p1,p2,p3` with these grids.
DEFAULT_RES = {"sun": (128, 80, 128), "static": (96, 60, 96)}
FRAME_CLASS = {"world": "static", "light": "sun", "view": "live"}
ANALYTIC = {"cos_view_sun": "dot(v, uSun)", "sun_up": "uSun.y", "view_up": "v.y",
            "height": "(x.y - uBoxMin.y) / (uBoxMax.y - uBoxMin.y)"}
CH = "xyzw"
f32 = np.float32
# A baked channel holds sum_i w_i * densProbe_i with densProbe in [0, 1], so sum_i |w_i| bounds it. Half floats end at
# 65504: a RGBA16F texture with a channel bound above this goes to RGBA32F (the interpreter evaluates in float32, and
# an overflowed texel would be +Inf -> NaN downstream, a hole only in baked builds).
HALF_SAFE_BOUND = 6.0e4


class NetworkError(ValueError):
    pass


def fail(msg):
    raise NetworkError(msg)


# ------------------------------------------------------------------------------------------------
# validation: the same rules and normalisation as renderer/net.js validateNetwork

def check_keys(obj, allowed, where):
    for k in obj:
        if k not in allowed and k not in COMMENT_KEYS and not k.startswith("_"):
            fail(f"{where}: unknown key {k!r} (allowed: {', '.join(sorted(allowed))})")


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate(net: dict, voxel_mean: float) -> dict:
    if not isinstance(net, dict):
        fail('network.json must be a JSON object with "probes" and "layers"')
    check_keys(net, {"probes", "layers"}, "network")
    probes_in = net.get("probes")
    if not isinstance(probes_in, list) or not 1 <= len(probes_in) <= LIMITS["probes"]:
        fail(f'"probes" must be a list of 1..{LIMITS["probes"]} probes')
    max_blur = 2 ** (BLUR_LEVELS - 1) * voxel_mean
    probes = []
    for i, p in enumerate(probes_in):
        where = f"probes[{i}]"
        if not isinstance(p, dict) or p.get("type") not in PROBE_TYPES:
            fail(f"{where}: unknown probe {p!r}")
        q = {"index": i, "type": p["type"], "frame": None, "dir": (0.0, 0.0, 0.0), "dist": 0.0, "blur": 0.0,
             "value": 0.0, "lo": 0, "t": 0.0}
        if p["type"] == "density":
            check_keys(p, {"type", "frame", "dir", "dist", "blur"}, where)
            if p.get("frame") not in FRAMES:
                fail(f'{where}: "frame" must be one of {FRAMES}')
            d = p.get("dir")
            if not isinstance(d, list) or len(d) != 3 or not all(is_num(c) for c in d):
                fail(f'{where}: "dir" must be 3 finite numbers')
            n = math.hypot(*d)
            if abs(n - 1) > 1e-3:
                fail(f'{where}: "dir" must be a unit vector (|dir| = {n:.6g})')
            dist, blur = p.get("dist", 0), p.get("blur", 0)
            if not is_num(dist) or dist < 0 or not is_num(blur) or blur < 0:
                fail(f'{where}: "dist" and "blur" must be finite numbers >= 0')
            if abs(dist) > F32_MAX:
                fail(f'{where}: "dist" is outside the float32 range')
            if blur > max_blur * (1 + 1e-9):
                fail(f'{where}: "blur" {blur} m exceeds the widest box filter, {max_blur:.1f} m')
            # net.js: level = log2(blur / voxel) clamped to [0, BLUR_LEVELS-1] in float64, then float32 uniforms
            lv = 0.0 if blur <= voxel_mean else min(math.log2(blur / voxel_mean), BLUR_LEVELS - 1)
            lo = min(math.floor(lv), BLUR_LEVELS - 1)
            t = lv - lo
            if lo == BLUR_LEVELS - 1:
                t = 0.0
            q.update(frame=p["frame"], dir=tuple(float(f32(c / n)) for c in d), dist=float(f32(dist)),
                     blur=float(blur), lo=int(lo), t=float(f32(t)))
        elif p["type"] == "const":
            check_keys(p, {"type", "value"}, where)
            if not is_num(p.get("value")) or abs(p["value"]) > F32_MAX:
                fail(f'{where}: const probe needs a finite float32 "value"')
            q["value"] = float(f32(p["value"]))
        else:
            check_keys(p, {"type"}, where)
        q["label"] = probe_label(q)
        probes.append(q)

    layers_in = net.get("layers")
    if not isinstance(layers_in, list) or not 1 <= len(layers_in) <= LIMITS["layers"]:
        fail(f'"layers" must be a list of 1..{LIMITS["layers"]} layers')
    nin, n_params, layers = len(probes), 0, []
    for k, L in enumerate(layers_in):
        where = f"layers[{k}]"
        if not isinstance(L, dict):
            fail(f"{where} must be an object")
        check_keys(L, {"W", "b", "act"}, where)
        W, b = L.get("W"), L.get("b")
        if not isinstance(W, list) or not 1 <= len(W) <= LIMITS["units"]:
            fail(f'{where}: "W" must be a list of 1..{LIMITS["units"]} rows')
        nout = len(W)
        for j, row in enumerate(W):
            if not isinstance(row, list) or len(row) != nin or not all(is_num(w) and abs(w) <= F32_MAX for w in row):
                fail(f"{where}: W[{j}] must have {nin} finite float32 entries")
        if not isinstance(b, list) or len(b) != nout or not all(is_num(x) and abs(x) <= F32_MAX for x in b):
            fail(f'{where}: "b" must have {nout} finite float32 entries')
        acts = [L.get("act")] * nout if isinstance(L.get("act"), str) else L.get("act")
        if not isinstance(acts, list) or len(acts) != nout or any(a not in ACTIVATIONS for a in acts):
            fail(f'{where}: "act" must be one of {ACTIVATIONS} or a list of {nout} of them')
        n_params += nout * nin + nout
        layers.append({"nin": nin, "nout": nout, "W": np.array(W, np.float64).astype(f32), "b": np.array(b, np.float64).astype(f32),
                       "acts": list(acts)})
        nin = nout
    if layers[-1]["nout"] != LIMITS["outputs"]:
        fail(f"the last layer must have exactly {LIMITS['outputs']} outputs")
    if n_params > LIMITS["params"]:
        fail(f"too many weights and biases: {n_params}")
    return {"probes": probes, "layers": layers, "n_params": n_params}


def fmt_num(x):
    return str(int(x)) if float(x).is_integer() else f"{x:.4g}"


def probe_label(p):
    i = p["index"]
    if p["type"] == "density":
        return (f"p{i}:density[{p['frame']} ({','.join(fmt_num(c) for c in p['dir'])}) dist={fmt_num(p['dist'])} "
                f"blur={fmt_num(p['blur'])}]")
    if p["type"] == "const":
        return f"p{i}:const[{fmt_num(p['value'])}]"
    return f"p{i}:{p['type']}"


# ------------------------------------------------------------------------------------------------
# float32 activations (constant folding) and GLSL emission helpers

def act_f32(a: str, x) -> np.float32:
    x = f32(x)
    with np.errstate(all="ignore"):
        r = {"linear": lambda: x, "relu": lambda: max(x, f32(0)), "exp": lambda: np.exp(np.clip(x, f32(-80), f32(80))),
             "log": lambda: np.log(max(x, f32(1e-8))), "tanh": lambda: np.tanh(np.clip(x, f32(-15), f32(15))),
             "sigmoid": lambda: f32(1) / (f32(1) + np.exp(-np.clip(x, f32(-80), f32(80)))),
             "softplus": lambda: max(x, f32(0)) + np.log(f32(1) + np.exp(-abs(x))), "abs": lambda: abs(x),
             "sin": lambda: np.sin(x), "square": lambda: x * x, "sqrt": lambda: np.sqrt(max(x, f32(0))),
             "recip": lambda: f32(1) / max(x, f32(1e-6))}[a]()
    return f32(r)


def lit(x) -> str:
    """Shortest GLSL float literal that round-trips the float32 value."""
    v = f32(x)
    if not np.isfinite(v):
        raise NetworkError(f"non-finite constant {x}")
    a = abs(float(v))
    if a != 0 and (a < 1e-4 or a >= 1e7):
        s = np.format_float_scientific(v, unique=True, trim="0")
    else:
        s = np.format_float_positional(v, unique=True, trim="0")
    if "." not in s and "e" not in s:
        s += ".0"
    return s


def act_expr(a: str, z: str, helpers: set) -> str:
    if a == "linear":
        return z
    if a in ("softplus", "square"):
        helpers.add(a)
        return f"net_{a}({z})"
    return {"relu": "max({z}, 0.0)", "exp": "exp(clamp({z}, -80.0, 80.0))", "log": "log(max({z}, 1e-8))",
            "tanh": "tanh(clamp({z}, -15.0, 15.0))", "sigmoid": "1.0 / (1.0 + exp(-clamp({z}, -80.0, 80.0)))",
            "abs": "abs({z})", "sin": "sin({z})", "sqrt": "sqrt(max({z}, 0.0))", "recip": "1.0 / max({z}, 1e-6)"}[a].format(z=z)


HELPERS = {"softplus": "float net_softplus(float x) { return max(x, 0.0) + log(1.0 + exp(-abs(x))); }",
           "square": "float net_square(float x) { return x * x; }"}


def lin_sum(terms: list[tuple[float, str]], bias: float = 0.0) -> str:
    """'w0 * a0 + w1 * a1 + b' with unit weights and signs written plainly."""
    out = []
    for w, e in terms:
        w = float(f32(w))
        neg = w < 0
        m = abs(w)
        body = e if m == 1.0 else f"{lit(m)} * {e}"
        out.append(("- " if neg else "+ ") + body)
    if bias != 0 or not out:
        b = float(f32(bias))
        out.append(("- " if b < 0 else "+ ") + lit(abs(b)))
    s = " ".join(out)
    return s[2:] if s.startswith("+ ") else "-" + s[2:]


def probe_pos(p) -> str:
    """Probe point x + dist * d with d in the probe's frame, as the interpreter's probeValue builds it."""
    basis = {"light": ("uSun", "uLu", "uLw"), "view": ("v", "e2", "e3")}.get(p["frame"])
    if p["dist"] == 0:
        return "x"
    if basis is None:
        d = f"vec3({', '.join(lit(c) for c in p['dir'])})"
    else:
        d = lin_sum([(c, b) for c, b in zip(p["dir"], basis) if c != 0])
        if len([c for c in p["dir"] if c != 0]) > 1 or d.startswith("-") or "*" in d:
            d = f"({d})"
    return f"x + {lit(p['dist'])} * {d}"


def probe_call(p) -> str:
    return f"densProbe({probe_pos(p)}, {p['lo']}, {lit(p['t'])})"


def probe_fetches(p) -> int:
    return 1 + (1 if p["t"] > 0 else 0)


# ------------------------------------------------------------------------------------------------
# graph: fold, alias, CSE, prune

def build_graph(norm: dict):
    """Returns (nodes, outputs). A source is ('c', value) | ('p', probe) | ('u', layer, unit).
    nodes[(k, j)] = {act, terms: {src: w}, bias} for every unit that must be computed (before pruning)."""
    probes, layers = norm["probes"], norm["layers"]
    srcs = [("c", p["value"]) if p["type"] == "const" else ("p", p["index"]) for p in probes]
    nodes = {}
    for k, L in enumerate(layers):
        out, seen = [], {}
        for j in range(L["nout"]):
            bias = f32(L["b"][j])
            terms: dict = {}
            for i in np.nonzero(L["W"][j])[0]:
                w, s = float(L["W"][j][i]), srcs[i]
                if s[0] == "c":
                    bias = f32(bias + f32(f32(w) * f32(s[1])))
                else:
                    terms[s] = terms.get(s, 0.0) + w
            terms = {s: float(f32(w)) for s, w in terms.items() if f32(w) != 0}
            a = L["acts"][j]
            if not terms:
                out.append(("c", float(act_f32(a, bias))))
                continue
            if k >= 1 and a == "linear" and bias == 0 and len(terms) == 1 and next(iter(terms.values())) == 1.0:
                out.append(next(iter(terms)))
                continue
            key = (a, tuple(sorted(terms.items())), float(bias))
            if key in seen:
                out.append(seen[key])
                continue
            src = ("u", k, j)
            nodes[(k, j)] = {"act": a, "terms": terms, "bias": float(bias)}
            seen[key] = src
            out.append(src)
        srcs = out
    outputs = srcs
    # prune: keep what the outputs reach
    need, stack = set(), [s for s in outputs if s[0] == "u"]
    while stack:
        s = stack.pop()
        if (s[1], s[2]) in need:
            continue
        need.add((s[1], s[2]))
        stack.extend(t for t in nodes[(s[1], s[2])]["terms"] if t[0] == "u")
    nodes = {kj: n for kj, n in nodes.items() if kj in need}
    return nodes, outputs


# ------------------------------------------------------------------------------------------------

def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(CLOUDS))
    except ValueError:
        return Path(p).name


def parse_res(spec: str | None) -> dict:
    res = dict(DEFAULT_RES)
    if not spec:
        return res
    for part in spec.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
        else:
            k, v = "all", part
        dims = tuple(int(x) for x in v.lower().split("x"))
        if len(dims) != 3 or min(dims) < 1:
            raise SystemExit(f"--res {part!r}: give NXxNYxNZ")
        keys = {"all": ["sun", "static"], "coarse": ["sun_coarse", "static_coarse"]}.get(k, [k])
        for kk in keys:
            if kk not in ("sun", "static", "sun_coarse", "static_coarse"):
                raise SystemExit(f"--res: unknown class {kk!r} (sun, static, sun_coarse, static_coarse, coarse)")
            res[kk] = dims
    return res


def parse_live(spec: str | None, n: int) -> set:
    out = set()
    for tok in (spec or "").replace(" ", "").split(","):
        if tok:
            i = int(tok[1:] if tok.startswith("p") else tok)
            if not 0 <= i < n:
                raise SystemExit(f"--live: probe {tok} out of range")
            out.add(i)
    return out


def compile_net(net_path: Path, scene_dir: Path, out_dir: Path, no_bake=False, live=None, res_spec=None,
                coarse_blur=None, fmt="RGBA16F") -> dict:
    net_bytes = net_path.read_bytes()
    net = json.loads(net_bytes)
    dj = json.loads((scene_dir / "density.json").read_text())
    voxel_mean = (dj["voxel_size"][0] + dj["voxel_size"][1] + dj["voxel_size"][2]) / 3
    dens_sha = dj.get("sha256")
    u8 = scene_dir / dj.get("file", "density.u8")
    if u8.is_file():
        dens_sha = hashlib.sha256(u8.read_bytes()).hexdigest()
    norm = validate(net, voxel_mean)
    probes, layers = norm["probes"], norm["layers"]
    unit_names = net.get("_unit_names") if isinstance(net.get("_unit_names"), list) else None
    live = parse_live(live, len(probes))
    res = parse_res(res_spec)

    def uname(k, j):
        if unit_names and k < len(unit_names) and isinstance(unit_names[k], list) and j < len(unit_names[k]):
            return str(unit_names[k][j])
        return f"L{k}.u{j}"

    nodes, outputs = build_graph(norm)

    def pclass(i):
        p = probes[i]
        if p["type"] != "density":
            return "analytic"
        if no_bake or i in live:
            return "live"
        return FRAME_CLASS[p["frame"]]

    # ---- first-layer combos ----
    combos: list[dict] = []   # {id, cls, coefs: {probe: w}, uses: [...]}

    def add_combo(cls, coefs: dict, unit):
        ps = tuple(sorted(coefs))
        for c in combos:
            if c["cls"] != cls or tuple(sorted(c["coefs"])) != ps:
                continue
            r = coefs[ps[0]] / c["coefs"][ps[0]]
            if all(abs(coefs[p] - r * c["coefs"][p]) <= 1e-6 * abs(coefs[p]) for p in ps):
                c["uses"].append({"unit": unit, "scale": float(f32(r))})
                return c, float(f32(r))
        c = {"id": len(combos), "cls": cls, "coefs": dict(coefs), "uses": [{"unit": unit, "scale": 1.0}]}
        combos.append(c)
        return c, 1.0

    l0_parts = {}   # (0, j) -> ({class: {probe: w}}, [(w, analytic probe)])
    for (k, j), n in sorted(nodes.items()):
        if k != 0:
            continue
        parts, ana = {}, []
        for (_, i), w in sorted(n["terms"].items()):
            c = pclass(i)
            if c == "analytic":
                ana.append((w, i))
            else:
                parts.setdefault(c, {})[i] = w
        l0_parts[(k, j)] = (parts, ana)
    # A baked class with fewer distinct probes than distinct combos bakes the probes themselves (each unit then sums
    # its probe channels inline); otherwise one channel per combo.
    basis = set()
    for cls in ("static", "sun"):
        ps = {i for parts, _ in l0_parts.values() for i in parts.get(cls, {})}
        keys = set()
        for parts, _ in l0_parts.values():
            if cls in parts:
                cf = parts[cls]
                p0 = min(cf)
                keys.add(tuple((i, round(cf[i] / cf[p0], 6)) for i in sorted(cf)))
        if ps and len(ps) < len(keys):
            basis.add(cls)
    l0 = {}   # (0, j) -> {combos: [(combo, scale)], analytic: [(w, probe)]}
    for (k, j), (parts, ana) in l0_parts.items():
        cs = []
        for c in ("static", "sun", "live"):
            if c not in parts:
                continue
            if c in basis:
                cs += [add_combo(c, {i: w}, uname(k, j)) for i, w in sorted(parts[c].items())]
            else:
                cs.append(add_combo(c, parts[c], uname(k, j)))
        l0[(k, j)] = {"combos": cs, "analytic": ana}

    # ---- bake textures: baked combos grouped by (class, tier), finest first, 4 channels each ----
    for c in combos:
        c["label"] = (f"p{next(iter(c['coefs']))}" if c["cls"] in basis else
                      "+".join(dict.fromkeys(u["unit"] for u in c["uses"])))
        c["min_blur"] = min(probes[i]["blur"] for i in c["coefs"])
        c["tier"] = c["cls"]
        if c["cls"] in ("static", "sun") and coarse_blur is not None and f"{c['cls']}_coarse" in res \
                and c["min_blur"] >= coarse_blur:
            c["tier"] = f"{c['cls']}_coarse"
    textures = []
    for tier in ("sun", "sun_coarse", "static", "static_coarse"):
        group = sorted((c for c in combos if c["tier"] == tier), key=lambda c: (c["min_blur"], c["id"]))
        for g in range(0, len(group), 4):
            t = len(textures)
            tex = {"name": f"uBake{t}", "fn": f"bake{t}", "class": tier.split("_")[0], "tier": tier,
                   "dims": list(res[tier]), "format": fmt, "combos": group[g:g + 4]}
            for ch, c in enumerate(tex["combos"]):
                c["tex"], c["ch"] = t, ch
                c["bound"] = float(sum(abs(w) for w in c["coefs"].values()))
            bound = max(c["bound"] for c in tex["combos"])
            if fmt == "RGBA16F" and bound > HALF_SAFE_BOUND:
                tex["format"] = "RGBA32F"
                tex["format_note"] = f"RGBA32F: a channel bound {bound:.6g} exceeds the half-float limit ({HALF_SAFE_BOUND:g})"
                print(f"note: {tex['name']} {tex['format_note']}", file=sys.stderr)
            textures.append(tex)

    # ---- GLSL: net_eval ----
    helpers: set = set()
    live_probes = sorted({i for c in combos if c["cls"] == "live" for i in c["coefs"]})
    ana_probes = sorted({i for v in l0.values() for _, i in v["analytic"]})
    body = []
    if textures:
        body.append("  vec3 bakeTc = (x - uBoxMin) / (uBoxMax - uBoxMin);")
        for t, tex in enumerate(textures):
            what = ", ".join(f"{CH[c['ch']]}: {c['label']}" for c in tex["combos"])
            body.append(f"  vec4 k{t} = texture({tex['name']}, bakeTc);   // {tex['class']} ({what})")
    for i in live_probes:
        body.append(f"  float p{i} = {probe_call(probes[i])};   // {probes[i]['label']}")
    for i in ana_probes:
        body.append(f"  float p{i} = {ANALYTIC[probes[i]['type']]};   // {probes[i]['label']}")

    def combo_ref(c):
        if c["cls"] == "live":
            return f"c{c['id']}"
        return f"k{c['tex']}.{CH[c['ch']]}"

    for c in combos:
        if c["cls"] == "live":
            if len(c["coefs"]) == 1 and next(iter(c["coefs"].values())) == 1.0:
                body.append(f"  float c{c['id']} = p{next(iter(c['coefs']))};")
            else:
                body.append(f"  float c{c['id']} = {lin_sum([(w, f'p{i}') for i, w in sorted(c['coefs'].items())])};")
    var = {}
    n_terms = 0
    for (k, j), n in sorted(nodes.items()):
        name = f"h{k}_{j}"
        var[(k, j)] = name
        if k == 0:
            v = l0[(k, j)]
            terms = [(s, combo_ref(c)) for c, s in v["combos"]] + [(w, f"p{i}") for w, i in v["analytic"]]
        else:
            terms = [(w, var[(s[1], s[2])]) for s, w in sorted(n["terms"].items())]
        n_terms += len(terms)
        body.append(f"  float {name} = {act_expr(n['act'], lin_sum(terms, n['bias']), helpers)};   // {uname(k, j)}")

    def out_ref(s):
        return lit(s[1]) if s[0] == "c" else var[(s[1], s[2])]

    o = [out_ref(s) for s in outputs]
    body.append(f"  aSun = vec3({o[0]}, {o[1]}, {o[2]});")
    body.append(f"  aSky = vec3({o[3]}, {o[4]}, {o[5]});")
    body.append(f"  aGnd = vec3({o[6]}, {o[7]}, {o[8]});")

    net_sha = hashlib.sha256(net_bytes).hexdigest()
    fetch_before = sum(probe_fetches(p) for p in probes if p["type"] == "density")
    fetch_after = sum(probe_fetches(probes[i]) for i in live_probes) + len(textures)
    w_total = sum(L["nin"] * L["nout"] for L in layers)
    w_nonzero = int(sum(np.count_nonzero(L["W"]) for L in layers))
    head = [f"// GENERATED by clouds/embed/compile/compile_net.py from {net_path.name} (sha256 {net_sha[:16]}); do not edit.",
            f"// mode: {'no-bake (every probe live)' if no_bake else 'bake'}; live density probes: "
            f"{', '.join(f'p{i}' for i in live_probes) or 'none'}; {len(textures)} bake texture(s);",
            f"// density fetches per sample: {fetch_before} (interpreter) -> {fetch_after}; "
            f"weights {w_nonzero} nonzero of {w_total} -> {n_terms} terms in {len(nodes)} units.",
            "// Host requirements (embed/DESIGN.md): float densProbe(vec3 x, int lo, float t) with the semantics of",
            "// renderer/shaders/march.frag.glsl; uniforms uSun, uLu, uLw, uBoxMin, uBoxMax; bake.glsl included BEFORE",
            "// this file (it declares the uBakeK samplers). Defines evalNet with the interpreter's signature."]
    net_eval = "\n".join(head + [""] + [HELPERS[h] for h in sorted(helpers)] + ([""] if helpers else []) +
                         ["void evalNet(vec3 x, vec3 v, vec3 e2, vec3 e3, out vec3 aSun, out vec3 aSky, out vec3 aGnd) {"] +
                         body + ["}", ""])

    # ---- GLSL: bake ----
    bake = [f"// GENERATED by clouds/embed/compile/compile_net.py from {net_path.name} (sha256 {net_sha[:16]}); do not edit.",
            f"// Bake textures (cell-centred over the box: texel i at boxMin + (i + 0.5) * size / n), "
            f"{', '.join(sorted({t['format'] for t in textures})) or fmt}, linear.",
            "// Host requirements: densProbe(vec3, int, float); uniforms uSun, uLu, uLw (sun textures), uBoxMin, uBoxMax.",
            f"// {len(textures)} texture(s)" + (": none (every probe is live)." if not textures else ".")]
    for tex in textures:
        bake.append(f"uniform highp sampler3D {tex['name']};")
    for tex in textures:
        bake += ["", f"// {tex['name']}: {tex['class']}, {'x'.join(map(str, tex['dims']))}"]
        bake.append(f"vec4 {tex['fn']}(vec3 x) {{")
        ps = sorted({i for c in tex["combos"] for i in c["coefs"]})
        for i in ps:
            bake.append(f"  float p{i} = {probe_call(probes[i])};   // {probes[i]['label']}")
        vals = [lin_sum([(w, f"p{i}") for i, w in sorted(c["coefs"].items())]) for c in tex["combos"]]
        vals += ["0.0"] * (4 - len(vals))
        bake.append(f"  return vec4({', '.join(vals)});")
        bake.append("}")
    bake_src = "\n".join(bake) + "\n"

    # ---- manifest ----
    def scale_note(c):
        return "; ".join(f"{u['unit']} uses {lit(u['scale'])} x channel" for u in c["uses"])

    def combo_name(c):
        return c["label"]

    manifest = {
        "network": rel(net_path), "network_sha256": net_sha, "density_sha256": dens_sha, "voxel_mean": voxel_mean,
        "mode": "no-bake" if no_bake else "bake",
        "compile_args": {"no_bake": no_bake, "live": sorted(live), "res": {k: list(v) for k, v in res.items()},
                         "coarse_blur": coarse_blur, "format": fmt},
        "live_probes": live_probes, "analytic_probes": ana_probes,
        "forced_live": sorted(live), "probe_basis_classes": sorted(basis),
        "textures": [{"name": t["name"], "fn": t["fn"], "class": t["class"], "tier": t["tier"], "dims": t["dims"],
                      "format": t["format"], **({"format_note": t["format_note"]} if "format_note" in t else {}),
                      "channels": [{"channel": CH[c["ch"]], "combo": combo_name(c), "probes": sorted(c["coefs"]),
                                    "coefs": [float(lit(c["coefs"][i])) for i in sorted(c["coefs"])], "min_blur": c["min_blur"],
                                    "bound": c["bound"], "scale_note": scale_note(c)} for c in t["combos"]]}
                     for t in textures],
        "half_safe_bound": HALF_SAFE_BOUND,
        "live_combos": [{"combo": combo_name(c), "probes": sorted(c["coefs"]), "coefs": [float(lit(c["coefs"][i])) for i in sorted(c["coefs"])],
                         "scale_note": scale_note(c)} for c in combos if c["cls"] == "live"],
        "probes": [{"index": p["index"], "label": p["label"], "class": pclass(p["index"]),
                    "used": p["index"] in live_probes or p["index"] in ana_probes or any(p["index"] in c["coefs"] for c in combos),
                    **({"lo": p["lo"], "t": p["t"], "fetches": probe_fetches(p)} if p["type"] == "density" else {})}
                   for p in probes],
        "stats": {"weights_total": w_total, "weights_nonzero": w_nonzero, "terms_emitted": n_terms,
                  "units_emitted": len(nodes), "units_total": sum(L["nout"] for L in layers),
                  "fetches_per_sample_before": fetch_before, "fetches_per_sample_after": fetch_after,
                  "live_density_probes": len(live_probes), "bake_textures": len(textures),
                  "bake_texels": int(sum(np.prod(t["dims"]) for t in textures)),
                  "bake_fetches_per_texel": {t["name"]: sum(probe_fetches(probes[i]) for i in
                                                            {i for c in t["combos"] for i in c["coefs"]}) for t in textures}},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "net_eval.glsl").write_text(net_eval)
    (out_dir / "bake.glsl").write_text(bake_src)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("net", nargs="?", default=str(DEFAULT_NET), help="network.json (default: Fable's r1 network)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (default embed/web/gen)")
    ap.add_argument("--scene-dir", default=str(DEFAULT_SCENE), help="scene output directory with density.json")
    ap.add_argument("--no-bake", action="store_true", help="every probe live (the exactness test)")
    ap.add_argument("--live", help="probes forced live, e.g. p0,p1")
    ap.add_argument("--res", help="bake grid NXxNYxNZ for all classes, or per class: sun=..,static=..,"
                                  "sun_coarse=..,static_coarse=.. (coarse=.. sets both coarse tiers)")
    ap.add_argument("--coarse-blur", type=float,
                    help="combos whose finest probe blur is >= this (m) go to the class's coarse texture (needs a "
                         "*_coarse res)")
    ap.add_argument("--format", choices=["RGBA16F", "RGBA32F"], default="RGBA16F",
                    help="bake texture format (RGBA32F needs OES_texture_float_linear; for precision tests)")
    a = ap.parse_args(argv)
    try:
        m = compile_net(Path(a.net), Path(a.scene_dir), Path(a.out), no_bake=a.no_bake, live=a.live,
                        res_spec=a.res, coarse_blur=a.coarse_blur, fmt=a.format)
    except NetworkError as e:
        print(f"network error: {e}", file=sys.stderr)
        return 2
    s = m["stats"]
    print(f"{a.net}: {s['units_emitted']} of {s['units_total']} units, {s['terms_emitted']} terms "
          f"({s['weights_nonzero']} nonzero of {s['weights_total']} weights); live density probes "
          f"{['p%d' % i for i in m['live_probes']]}; fetches/sample {s['fetches_per_sample_before']} -> "
          f"{s['fetches_per_sample_after']}; {len(m['textures'])} bake texture(s) -> {a.out}")
    for t in m["textures"]:
        print(f"  {t['name']} {t['class']:6s} {'x'.join(map(str, t['dims']))}: " +
              ", ".join(f"{c['channel']}={c['combo']}" for c in t["channels"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
