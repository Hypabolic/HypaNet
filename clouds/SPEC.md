# HypaNet Clouds: a hand-written light-transport network

A fun side experiment, less rigorous than v0.2–v0.5. Fable (`claude-fable-5-1`) writes, by reasoning alone, the weights of a
small neural network that does the light transport for a volumetric cloud renderer, in the spirit of Disney's
*Deep Scattering* (Kallweit et al., SIGGRAPH Asia 2017), which trained such a network on path-traced data. Nobody trains
this one. Goal: hyper-realistic light scattering through cloud, including the colour change from noon to sunset.

Decisions (Matthew, 2026-11-08): network role = light transport (Deep Scattering style); feedback = Fable may render
and view its own frames and may view path-traced reference images, but gets no fitting tool; Fable cost cap US$500;
one run.

## 1. Division of labour

| Part | Who | Fixed before launch |
|---|---|---|
| Cloud density field, sun and sky model, cameras, tone map | harness | yes |
| Ray marcher along the view ray (transmittance, integration, background) | harness | yes |
| Probe palette (the network's possible inputs) and the MLP interpreter | harness | yes |
| Reference images (Mitsuba 3 volumetric path tracer, same scene) | harness | yes |
| **Which probes to use, the layer sizes, per-neuron activations, every weight and bias** | **Fable** | written in the run |

Fable writes no shader code and cannot change the renderer. Its whole output is `network.json` (plus the scripts that
generated it and a `NETWORK.md` explaining every unit).

## 2. Directory layout (`clouds/`)

```
SPEC.md                 this file
pyproject.toml, uv.lock own uv project (Python 3.12): numpy, pillow, mitsuba==3.9.1, playwright
scene/                  build_scene.py -> scene/out/ (generated, gitignored except small JSON)
renderer/               three.js renderer: index.html, main.js, shaders/*.glsl, vendor/three.module.min.js (pinned)
  render.py             headless CLI (Playwright + Chromium, Metal ANGLE flags) -> PNG / EXR-like float dump
  viewer.html           interactive viewer (orbit camera, sun sliders, progressive accumulation)
reference/              render_ref.py (Mitsuba metal_ad_rgb volpath) -> reference/out/<shot>.npy + .png
validate/               harness self-checks (single-scattering agreement, tone map agreement)
kit/                    author kit source; scripts/prepare_author_dir.sh copies it to ../HypaNet-clouds-author/
scripts/                launch.sh, audit.py, cost tracking
runs/                   archived run outputs after the session (network.json, scripts, NETWORK.md, transcripts)
```

Scene and reference outputs are deterministic; large binaries are gitignored and rebuilt by script. Tone-mapped
reference PNGs are committed.

## 3. Scene (units: metres; y is up)

- **Density grid.** A single cumulus (congestus) cloud, flat-ish base at about 1,500 m altitude, cauliflower billows
  on top. Box roughly 1,800 m × 1,100 m × 1,800 m, grid 256 × 160 × 256 (about 7 m voxels), values `d ∈ [0, 1]`,
  stored as uint8 (`scene/out/density.u8`, x fastest, then y, then z) with `scene/out/density.json`
  (`dims`, `box_min`, `box_max`, `sigma_t_max`). Built procedurally (displaced metaballs + Perlin-Worley FBM erosion,
  height profile), fixed seed. It must look like a real cumulus at a glance.
- **Medium.** Extinction `σ_t(x) = d(x) · sigma_t_max` with `sigma_t_max = 0.04 m⁻¹`. Single-scattering albedo
  `ω = 0.999`, spectrally flat. Phase function (propagation-direction convention, `μ` = cosine between the incoming
  light's travel direction and the scattered direction):
  `p(μ) = 0.8 · HG(μ; 0.9) + 0.2 · HG(μ; −0.3)`, `HG(μ; g) = (1 − g²) / (4π (1 + g² − 2gμ)^{3/2})`.
- **Sun.** Directional (delta) light, toward-sun unit vector `s`. Irradiance `E_sun = 20 · T_atm(h)` (RGB), with
  atmospheric transmittance `T_atm(h) = exp(−(β_R · 8000 + β_M · 1200 + β_O3) · m(h))`, `β_R = (5.802, 13.558,
  33.100) × 10⁻⁶ m⁻¹`, `β_M = 3.996 × 10⁻⁶ m⁻¹ × 1.11` (Mie extinction), ozone optical depth per RGB channel at zenith
  `β_O3` as chosen by scene/build_scene.py (document the values), and Kasten–Young air mass
  `m(h) = 1 / (sin h + 0.50572 (h° + 6.07995)^{−1.6364})`. RGB channels stand for 680, 550 and 440 nm.
- **Sky.** One equirectangular float32 RGB map per sun preset (1024 × 512), from a single-scattering Rayleigh + Mie
  (+ ozone absorption) atmosphere (Nishita style) seen from the cloud's altitude, with no sun disk (the sun is the
  directional light). Lower hemisphere: Lambertian ground, albedo 0.1, constant radiance from the sun + sky irradiance.
  The same map is the background in the renderer and the environment emitter in Mitsuba. The scene also exports, per
  preset, `L_sky` (mean radiance of the upper hemisphere, RGB) and `L_gnd` (ground radiance, RGB).
- **Presets and shots.** `scene/out/shots.json`: sun presets (elevation, azimuth): noon-ish 55°, afternoon 25°,
  golden 8°, sunset 2°. Camera shots: position, look-at, vertical fov, resolution 640 × 400, exposure. At least:
  side view with the sun behind the camera, side view with the sun at 90° (terminator), backlit view toward the sun
  (silver lining), view from below the base, a distant view showing the whole cloud low in the sky. Reference set =
  12 (shot × preset) pairs given to Fable. Held-back set = 6 further pairs, never given to Fable, scored afterwards.
- **Tone map** (identical in GLSL and numpy): `c = exposure · L` (linear RGB), then ACES fit
  `f(x) = clamp((x (2.51x + 0.03)) / (x (2.43x + 0.59) + 0.14), 0, 1)` per channel, then the sRGB OETF, then 8-bit
  rounding. Also dump the linear float image for numeric comparison.

## 4. Renderer (three.js, WebGL2)

A full-screen `RawShaderMaterial` pass (three.js hosts it; the shader is ours). For each pixel:

1. Ray from the camera; intersect the density box; march with step `Δ` (start about 6 m, may grow with distance; max
   512 steps), per-pixel jittered start (hash of pixel + frame index). Density by trilinear sampling of a `Data3DTexture`
   (R8, linear filtering).
2. At each sample with `d > 0.002`, evaluate the network at the sample point to get the in-scattered radiance `S`
   (RGB, below), then accumulate `L += T · ω · S · (1 − exp(−σ_t Δ))` and update `T *= exp(−σ_t Δ)`. Stop when
   `T < 0.002`.
3. Background: `L += T · sky(v)` from the equirect map. Tone map (§3). Frames accumulate (running mean in a float
   target) for a converged still; `render.py` accumulates 16 frames by default, rendering in tiles small enough to
   avoid the macOS GPU watchdog.

**Network output.** 9 numbers `(a_sun RGB, a_sky RGB, a_gnd RGB)`, and `S = a_sun ⊙ E_sun + a_sky ⊙ L_sky + a_gnd ⊙ L_gnd`.
For reference: exact single scattering of sunlight alone is `a_sun = T_sun(x) · p(dot(v, s))`, with `T_sun` the
transmittance from `x` toward the sun, `v` the view ray direction (camera to point) and `s` the toward-sun direction.

**Probes** (the network's possible inputs; Fable picks up to 64, in any order, with repeats allowed):

| Probe | Parameters | Value |
|---|---|---|
| `density` | `frame` ∈ {light, world, view}, unit `dir` (3 numbers in that frame), `dist` (m, ≥ 0), `blur` (m, ≥ 0) | `d` at `x + dist · dir`, from a pre-blurred mip pyramid (blur ≈ box-filter width; 0 = full resolution); 0 outside the box |
| `cos_view_sun` | — | `dot(v, s)` |
| `sun_up` | — | `s.y` (sine of sun elevation) |
| `view_up` | — | `v.y` |
| `height` | — | `(x.y − box_min.y) / (box_max.y − box_min.y)` |
| `const` | `value` | a constant |

Frames: `light` = orthonormal (`s`, `u`, `w`) with `u` = normalised `cross(s, world_up)` (fallback when parallel), `w`
= `cross(u, s)`, components given as (along `s`, along `u`, along `w`); `world` = (x, y, z); `view` = (along `v`,
camera right, camera up).

**MLP.** `network.json`:

```json
{
  "probes": [{"type": "density", "frame": "light", "dir": [1,0,0], "dist": 20, "blur": 8}, {"type": "cos_view_sun"}],
  "layers": [
    {"W": [[...], ...], "b": [...], "act": ["exp", "relu", ...]},
    {"W": [[...]], "b": [...], "act": "linear"}
  ]
}
```

Layer `k` computes `h_k = act(W_k h_{k−1} + b_k)` with `h_0` = probe values. `W_k` is `[out][in]`. `act` is one
activation for the whole layer or one per neuron, from: `linear, relu, exp, log, tanh, sigmoid, softplus, abs, sin,
square, sqrt, recip` (`log(max(x, 1e-8))`, `sqrt(max(x, 0))`, `recip = 1 / max(x, 1e-6)`; `exp` input clamped to
[−80, 80]). The last layer has exactly 9 outputs. Limits: at most 6 layers, at most 128 units per layer, at most 50,000
weights and biases in total. Weights are float32. The renderer loads them into a float texture and evaluates the MLP
in the fragment shader (generic interpreter; no per-network shader generation by the author).

## 5. Reference renderer (Mitsuba 3.9.1, `metal_ad_rgb`, `volpath`)

Same density grid (as a `gridvolume`, same box, trilinear), `σ_t`, albedo, phase (`blendphase` of two `hg` lobes —
check Mitsuba's direction convention so the forward lobe is forward), the directional sun with irradiance `E_sun`, the
sky map as an `envmap` (check orientation and scale against the renderer), same camera (vertical fov). Unlimited depth
with Russian roulette; spp high enough that noise is invisible after tone mapping (record spp and time). Output linear
float `.npy` and the tone-mapped `.png` through the same tone map.

## 6. Validation the harness must pass before launch

1. **Single scattering.** A harness-only mode in the renderer computes `a_sun` exactly (shadow march toward the sun,
   phase as §3, `a_sky = a_gnd = 0`), and Mitsuba renders the same scene with only sun single scattering (max depth 2,
   envmap off in-medium but still the background, or an equivalent split). Linear images must agree within noise
   (report per-shot mean relative error; aim < 3%). This checks the camera, units, phase convention, transmittance,
   integration and the tone map in one go.
2. **Background.** Empty-density render vs Mitsuba: the sky backgrounds must match.
3. **Interpreter.** A small hand-made test network (kept out of the kit) reproducing the exact single-scattering
   `a_sun` from probes (a sum of densities along `s` × distance → `exp`, times an HG built from `cos_view_sun` with
   log/exp units) renders close to mode 1. This proves the probes, frames and activations work.
4. Determinism: same inputs give the same PNG.

## 7. Author kit and run

- Fresh directory outside the repo (`../HypaNet-clouds-author/`), with: the SPEC sections the author needs (scene,
  physics, renderer equation, probes, MLP format, rules — written up as `KIT.md`), the 12 reference shots (PNG and linear
  `.npy`, plus their shot/preset parameters), and tools:
  - `./render NET.json --shot ID [--frames N] [--out PATH]` → tone-mapped PNG (and linear `.npy`); `--compare` makes a
    side-by-side with the reference when one exists;
  - `./inspect NET.json --shot ID --pixel X Y [--samples K]` → per-sample probe values, every unit's activation and
    `S` along that pixel's ray (the "watch the neurons" tool);
  - `python3` with numpy only (to compute weights in the author's own scripts).
- **Rules (in the prompt):** no training, fitting, optimisation or automated search of any kind (no gradient descent,
  least squares, curve fitting, regression, random or grid search over weights or settings, and no fitting to reference
  pixels). Every weight comes from reasoning about the physics and from manual revision after looking at renders, as an
  artist would. No scipy/torch/sklearn. Don't modify the tools. Keep `NETWORK.md` up to date: what every unit computes.
- Session: headless Claude Code, `--model claude-fable-5-1`, max effort, no web, no MCP, no subagents; Bash limited to
  the kit tools and python3; Read allowed (so it can view PNGs). The launcher continues the session after API errors and
  stops at the US$500 cap or when the author ends with `NETWORK COMPLETE`. Long thinking is never a reason to stop.
- Matthew sees and approves the final prompt before launch.
- Afterwards: archive to `runs/r1/`, audit the transcript for fitting or rule breaks, score the 12 reference and 6
  held-back pairs (linear relative error and a side-by-side gallery), and publish the interactive viewer.
