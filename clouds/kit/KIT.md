# HypaNet Clouds: author kit

You will write, by reasoning, every weight of a small neural network that does the light transport inside a
volumetric cloud renderer. The renderer marches rays through one fixed cumulus cloud. At every sample point
inside the cloud it evaluates your network, which says how much light arrives there and scatters toward the
camera. Nobody trains this network. You design it, as an artist and physicist would, from the physics below and
from looking at your own renders next to path-traced reference images.

**Goal:** hyper-realistic light scattering through the cloud, including the colour change from noon to sunset.
The reference images were rendered with a volumetric path tracer (Mitsuba 3) of exactly the same scene, so they
show what the physics produces. Your network will also be judged on other camera views and sun positions of this
same cloud that you are not shown, so it should be right for physical reasons, not tuned to the 12 views.

Everything you need is in this file. `PROMPT.md` is your task statement.

## 1. Your directory

| Path | What it is |
|---|---|
| `KIT.md` | This file |
| `PROMPT.md` | The task statement you were given |
| `references/<pair>.png` | The 12 reference images, tone-mapped (640 × 400) |
| `references/<pair>.npy` | The same images as linear radiance: float32, shape (400, 640, 3), RGB, row 0 at the top |
| `references/<pair>.json` | That pair's camera, sun, exposure and how the reference was rendered (samples, noise) |
| `references/scene.json` | All the scene constants of this file in machine-readable form: medium, sun model, presets, shots, density box |
| `references/sky_<preset>.png` | A preview of each preset's sky map (section 3.4): the sky the cloud sits in, not a reference image |
| `./render`, `./inspect` | Render your network; watch its neurons along one pixel's ray (section 7) |
| `./python3` | Python 3.12 with numpy only, for your own scripts (section 7) |
| `work/` | Your working directory, the only place you can write. Your deliverables go here |
| `tools/`, `.venv/`, `HANDOVER.json` | The kit's guard, the private interpreter and the hand-over record. Read-only |

Everything except `work/` is read-only.

## 2. Units and axes

Metres. World frame: y is up, east = +x, north = −z. Radiance and irradiance are in one consistent linear unit (the
sun's irradiance at the top of the atmosphere is 20). RGB channels stand for 680, 550 and 440 nm.

## 3. The scene

### 3.1 Cloud density

A single cumulus congestus cloud, defined by a density grid `d(x) ∈ [0, 1]`:

| | |
|---|---|
| Box | `box_min = (−900, 1450, −900)`, `box_max = (900, 2550, 900)` |
| Grid | 256 × 160 × 256 voxels (x, y, z), voxel 7.03 × 6.875 × 7.03 m, cell-centred, trilinear interpolation |
| Occupied region | x −759…745, y 1498…2509, z −591…654 m; about 17 % of the voxels are non-zero; largest value d = 0.97 |
| Shape | flat base at about 1,500 m; several towers of different heights (the highest crown near 2,500 m) with cauliflower billows; wispy, eroded lower flanks |

Outside the box `d = 0`. You do not get the grid itself; your network sees it only through density probes
(section 5), and you can look at it through renders and `./inspect`.

### 3.2 Medium

- Extinction `σ_t(x) = d(x) · σ_max`, `σ_max = 0.04 m⁻¹` (so a 25 m path at d = 1 has optical depth 1, and the
  cloud is optically very thick: hundreds of metres of it reach optical depths of tens).
- Single-scattering albedo `ω = 0.999`, the same in every channel (the cloud itself is white; all colour comes from
  the light).
- Phase function, in the propagation-direction convention (`μ` = cosine between the incoming light's travel
  direction and the scattered direction; `μ = 1` means straight on, no deflection):

  `p(μ) = 0.8 · HG(μ; 0.9) + 0.2 · HG(μ; −0.3)`, `HG(μ; g) = (1 − g²) / (4π (1 + g² − 2gμ)^{3/2})`

  `p` integrates to 1 over the sphere (units sr⁻¹). It has a very strong forward peak (`p(1) ≈ 12.1 sr⁻¹`, against `p(0) ≈ 0.018` and
  `p(−1) ≈ 0.044`).

### 3.3 Sun

A directional light (a delta in direction; the disc is not visible), toward-sun unit vector `s`. Irradiance on a
plane facing the sun:

`E_sun = 20 · T_atm(h)` (RGB), `T_atm(h) = exp(−(β_R · 8000 + β_M · 1200 + τ_O3) · m(h))`

with `β_R = (5.802, 13.558, 33.100) × 10⁻⁶ m⁻¹` (Rayleigh), `β_M = 3.996 × 10⁻⁶ × 1.11 m⁻¹` (Mie extinction),
ozone optical depth at zenith `τ_O3 = (0.00975, 0.02822, 0.00128)`, and the Kasten–Young air mass
`m(h) = 1 / (sin h + 0.50572 (h° + 6.07995)^{−1.6364})`, `h` = sun elevation.

Directions: `s = (cos h · sin a, sin h, −cos h · cos a)`, azimuth `a` clockwise from north (−z). Every preset has
azimuth 250° (west-south-west).

### 3.4 Sky and ground

The sky is a single-scattering Rayleigh + Mie (+ ozone) atmosphere seen from 2,000 m altitude, with no sun disc,
stored as one equirectangular map per preset. Below the horizon is a Lambertian ground of albedo 0.1 with one
constant radiance `L_gnd = 0.1/π · (E_sun · sin h + E_sky)`. The renderer uses the map only as the background
behind the cloud. For light inside the cloud, the scene gives two constants per preset:

- `L_sky`: the plain mean radiance of the upper hemisphere of the map (averaged over solid angle, not
  cosine-weighted), RGB;
- `L_gnd`: the ground radiance, RGB.

`E_sky` (the sky's cosine-weighted irradiance on a horizontal plane) is given for reference only.

The reference renders (section 3.7) light the cloud with the full sky map, not only with the constants above, so the
sky's shape matters: it is brightest near the horizon and toward the sun. `references/sky_<preset>.png` shows each
map (512 × 256, the 1024 × 512 map averaged over 2 × 2 texels; tone-mapped like the references with exposure
0.25 noon, 0.25 afternoon, 0.35 golden, 1.0 sunset). Mapping: column `u` and row `v` from the top give azimuth
`360° · u / W` (clockwise from north, so the sun at azimuth 250° is at u ≈ 0.69) and zenith angle `180° · v / H`;
the lower half is the ground (`L_gnd`). Sky radiance (RGB) at the zenith and 1° above the horizon:

| preset | zenith | horizon toward the sun (az 250°) | horizon across (az 160° / 340°) | horizon away from the sun (az 70°) |
|---|---|---|---|---|
| noon | 0.0708, 0.1503, 0.3282 | 0.9049, 1.236, 1.26 | 0.6605, 0.909, 0.9305 | 0.8635, 1.19, 1.219 |
| afternoon | 0.04708, 0.09701, 0.2053 | 1.386, 1.664, 1.479 | 0.6349, 0.8182, 0.7553 | 1.14, 1.468, 1.356 |
| golden | 0.03617, 0.06318, 0.1181 | 2.407, 1.852, 0.9659 | 0.5467, 0.5511, 0.3378 | 1.05, 1.033, 0.6305 |
| sunset | 0.02727, 0.03403, 0.0527 | 2.125, 0.955, 0.1783 | 0.386, 0.2275, 0.04612 | 0.6567, 0.3273, 0.05534 |

### 3.5 The four sun presets

| preset | elevation | `s` (toward sun) | air mass | `E_sun` (RGB) | `L_sky` (RGB) | `L_gnd` (RGB) | `E_sky` (RGB) |
|---|---|---|---|---|---|---|---|
| noon | 55° | (−0.5390, 0.8192, 0.1962) | 1.220 | 18.555, 16.819, 14.363 | 0.1747, 0.3220, 0.5570 | 0.4944, 0.4600, 0.4177 | 0.3341, 0.6753, 1.3569 |
| afternoon | 25° | (−0.8517, 0.4226, 0.3100) | 2.356 | 17.303, 14.313, 10.552 | 0.1914, 0.3278, 0.5235 | 0.2429, 0.2120, 0.1788 | 0.3181, 0.6099, 1.1589 |
| golden | 8° | (−0.9305, 0.1392, 0.3387) | 6.857 | 13.120, 7.554, 3.111 | 0.1786, 0.2483, 0.3266 | 0.06697, 0.04777, 0.03727 | 0.2778, 0.4494, 0.7378 |
| sunset | 2° | (−0.9391, 0.0349, 0.3418) | 19.43 | 6.0545, 1.2664, 0.10245 | 0.1327, 0.1255, 0.1241 | 0.01340, 0.009037, 0.01013 | 0.2096, 0.2397, 0.3147 |

Full precision is in `references/scene.json`.

### 3.6 Cameras

Pinhole cameras: `forward = normalize(look_at − position)`, `right = normalize(cross(forward, up))` with
`up = (0, 1, 0)`, `cam_up = cross(right, forward)`; vertical field of view `fov`. Pixel (i, j), j = 0 at the top, of a
W × H image looks along
`v ∝ forward + (2(i + 0.5)/W − 1) · tan(fov/2) · (W/H) · right + (1 − 2(j + 0.5)/H) · tan(fov/2) · cam_up`
(the renderer also jitters within the pixel from frame to frame, a box pixel filter).

| shot | position | look_at | fov | where the sun is |
|---|---|---|---|---|
| front | (−3100.986, 1150, 1128.666) | (0, 1930, 0) | 30° | behind the camera (the view is lit from the front) |
| side | (−1128.666, 1150, −3100.986) | (0, 1930, 0) | 30° | 90° to the view, on the screen's right (terminator) |
| backlit | (3100.986, 1200, −1128.666) | (0, 1950, 0) | 32° | ahead, behind the cloud (silver lining); about 5–12° from the view axis from afternoon to sunset |
| below | (164.966, 600, −935.567) | (0, 1700, 0) | 65° | ahead and to the right; the camera looks up into the base |
| distant | (−10724.622, 400, −8999.027) | (−9961.493, 487.156, −8358.685) | 12° | behind the camera and to the right; the cloud is 14 km away, low in the sky, horizon near the bottom |

### 3.7 The 12 reference pairs

| pair id | shot | preset | exposure |
|---|---|---|---|
| front_noon | front | noon | 0.315 |
| front_golden | front | golden | 0.5 |
| front_sunset | front | sunset | 2.0 |
| side_noon | side | noon | 0.315 |
| side_afternoon | side | afternoon | 0.3969 |
| side_sunset | side | sunset | 2.5198 |
| backlit_afternoon | backlit | afternoon | 0.1984 |
| backlit_golden | backlit | golden | 0.315 |
| backlit_sunset | backlit | sunset | 1.5874 |
| below_noon | below | noon | 0.25 |
| below_golden | below | golden | 1.0 |
| distant_sunset | distant | sunset | 2.5198 |

Exposures were chosen per pair so the bright parts of the cloud sit just below white. They differ between pairs, so
compare brightness between pairs in the linear `.npy` files, not in the PNGs. The per-channel tone map also shifts
bright saturated colours toward yellow (a sunset orange shows as peach), so judge colour in the `.npy` files too.

**How the references were made.** Mitsuba 3.9.1 `volpath`: the same density grid (trilinear), `σ_t`, albedo and
phase function, the sun as a directional emitter with irradiance `E_sun`, and the sky map as an environment emitter
(so the sky lights the cloud from every direction with its actual radiance, and the ground lights it from below).
Unlimited scattering depth (Russian roulette from depth 256), 12,288–24,576 samples per pixel. Lit parts are clean;
a fine grain remains in dark parts (shadowed bodies, bases, backlit interiors): about 2–4 levels out of 255 after the
tone map, more in the darkest places. Each `references/<pair>.json` gives the figures. Compare averages over small
blocks rather than single pixels in dark regions.

**Agreement of renderer and reference.** The renderer and the path tracer agree on everything except your network:
the camera, the density, the sun direction and irradiance, the phase function, the transmittance, the background
sky and the tone map have been checked against each other, to well under 1 %. The renderer's two cut-offs
(section 4: samples with `d ≤ 0.002` are skipped, the march stops at `T < 0.002`) make it at most about 0.15 %
darker over a whole image than an exact march, and up to about 3 % in the thinnest places such as a backlit rim.
Beyond that, any difference between your render and the reference comes from `S`, the network's output.

## 4. The renderer

A WebGL2 fragment shader (three.js) marches each pixel's ray through the box:

1. Intersect the ray with the box; march with step `Δ = 6 m` from a per-pixel, per-frame jittered start.
   Density by trilinear sampling of the grid.
2. At each sample point `x` with `d(x) > 0.002`, evaluate your network at `x` to get `S` (RGB, below), then
   `L += T · ω · S · (1 − exp(−σ_t Δ))` and `T *= exp(−σ_t Δ)`. Samples with `d ≤ 0.002` are skipped (they neither
   evaluate the network nor change T). Stop when `T < 0.002`.
3. Background: `L += T · sky(v)` from the sky map (`v` = the ray direction).
4. Tone map: `c = exposure · L`, then per channel the ACES fit
   `f(x) = clamp(x (2.51x + 0.03) / (x (2.43x + 0.59) + 0.14), 0, 1)`, then the sRGB transfer function, then 8 bits.
5. Frames accumulate (running mean of the linear image, each frame with new start and sub-pixel jitter). `./render`
   uses 16 frames by default.

**The network's output.** 9 numbers `(a_sun RGB, a_sky RGB, a_gnd RGB)`. The renderer forms

`S = a_sun ⊙ E_sun + a_sky ⊙ L_sky + a_gnd ⊙ L_gnd`

(`⊙` per channel), with the current preset's `E_sun`, `L_sky`, `L_gnd`. `S` is the radiance scattered at `x`
toward the camera per unit of scattering (the integral of the phase function times the radiance arriving at `x`).
`a_sun` has units sr⁻¹; `a_sky` and `a_gnd` are dimensionless. The network has no probe that reads `E_sun`,
`L_sky` or `L_gnd`; the renderer multiplies them in.

For reference, exact single scattering of sunlight alone (no sky light, no multiple scattering) is

`a_sun = T_sun(x) · p(dot(v, s))`, `a_sky = a_gnd = 0`,

with `T_sun(x) = exp(−∫ σ_t)` along the straight path from `x` toward the sun, `v` the view ray direction (from
the camera toward the point) and `s` the toward-sun direction. (Sunlight travels along `−s` and leaves toward the
camera along `−v`, so `μ = dot(v, s)`.)

## 5. Probes: what your network can see

Your network's inputs are up to 64 probes, chosen by you, in any order, repeats allowed. All are evaluated at the
sample point `x`.

| type | parameters | value |
|---|---|---|
| `density` | `frame` ∈ {`light`, `world`, `view`}; unit `dir` (3 numbers, in that frame); `dist` (m, ≥ 0); `blur` (m, ≥ 0) | the density at `x + dist · dir`, box-filtered over a cube of width `blur` (0 = the raw grid, trilinear); 0 outside the box |
| `cos_view_sun` | — | `dot(v, s)` |
| `sun_up` | — | `s.y` (the sine of the sun's elevation) |
| `view_up` | — | `v.y` |
| `height` | — | `(x.y − 1450) / 1100`, i.e. 0 at the bottom of the box and 1 at its top (the cloud base is at about 0.045) |
| `const` | `value` | the constant |

**Frames.** `dir` is given as components along three orthonormal axes:

- `light`: (along `s`, along `u`, along `w`) with `u = normalize(cross(s, world_up))`, `w = cross(u, s)`. So
  `[1, 0, 0]` points toward the sun and `[−1, 0, 0]` away from it; `w` is the "up" direction perpendicular to `s`
  (it has a positive y component).
- `world`: (x, y, z).
- `view`: (along `v`, `e2`, `e3`) with `e2 = normalize(cross(v, cam_up))`, `e3 = cross(e2, v)`. At the centre of the
  image `e2` and `e3` are exactly the camera's right and up; elsewhere they are made perpendicular to that pixel's
  ray. `[1, 0, 0]` points onward along the ray, away from the camera.

**Blur.** The blurred density comes from precomputed centred 3D box filters 2^k voxels wide (k = 0…7; one voxel is
about 6.98 m). `blur` in metres picks the fractional level `log2(blur / 6.98)` and the renderer interpolates
linearly between the two nearest levels. `blur` up to one voxel gives the raw grid; the widest allowed is 128 voxels
≈ 893 m (a larger `blur` is an error). The filter is a cube: it averages density on all sides of the probe point,
not only along `dir`. Measured against an exact box filter it is within about 0.04 (absolute, in units of d).

**Precision.** `dir` must be unit length within 0.001 (it is renormalised). Density values carry the GPU's 8-bit
interpolation precision (about 1e-3).

## 6. The network: `network.json`

```json
{
  "probes": [
    {"type": "density", "frame": "light", "dir": [1, 0, 0], "dist": 20, "blur": 8},
    {"type": "cos_view_sun"}
  ],
  "layers": [
    {"W": [[0.5, -1.0], [2.0, 0.0]], "b": [0.0, 0.1], "act": ["exp", "relu"]},
    {"W": [[...], ...], "b": [...], "act": "linear"}
  ]
}
```

- Layer `k` computes `h_k = act(W_k h_{k−1} + b_k)` with `h_0` = the probe values. `W_k` is `[out][in]`.
- `act` is one activation for the whole layer (a string) or one per unit (a list), from:

  | name | value |
  |---|---|
  | `linear` | `x` |
  | `relu` | `max(x, 0)` |
  | `exp` | `exp(x)`, input clamped to [−80, 80] |
  | `log` | `log(max(x, 1e-8))` |
  | `tanh`, `sigmoid`, `softplus`, `abs`, `sin`, `square` | as named (`softplus = log(1 + exp(x))`) |
  | `sqrt` | `sqrt(max(x, 0))` |
  | `recip` | `1 / max(x, 1e-6)` |

- The last layer has exactly 9 outputs, in the order `a_sun` (R, G, B), `a_sky` (R, G, B), `a_gnd` (R, G, B).
  They are used as they are (no clamping); negative values give negative light.
- Limits: at most 64 probes, 6 layers, 128 units per layer, 50,000 weights and biases in total. All numbers are
  float32 on the GPU.
- Keys must be exactly as above; a misspelt key is an error. You may add comment keys anywhere: `name`, `comment`,
  `comments`, `note`, `notes`, `description`, `label`, `doc`, or any key starting with `_`. They are ignored.
- The renderer validates the file and reports any problem clearly (`./render` prints it).
- **Speed.** The shader evaluates your network at every accepted sample of every pixel, about 2–2.5 million
  evaluations per frame. Rough times for one 640 × 400, 16-frame render on this machine: under 1 s for a network
  of a few dozen parameters, about 3 s for a few hundred, 6.5 s for 64 → 32 → 9 (2.4k parameters), 14 s for
  64 → 64 → 64 → 9 (8.9k), 15–35 s for 64 → 128 → 128 → 9 (26k) and about a minute for 64 → 128 → 128 → 128 → 9
  (42.5k). Each `./render` call also pays about 1 s of start-up. Use `--frames` and `--res` for quick looks.
- **`sin` of large arguments** (|x| above about 50) is imprecise in float32 on the GPU.

## 7. Tools

Run everything from your directory (the Bash tool starts there). The Bash tool runs only `./render`, `./inspect`
and `./python3` commands, one command per call (no pipes, redirection, `&&`, `cd` or other programs). To list or
read files use the Glob, Grep and Read tools; **Read shows you a PNG as an image**, so look at your renders.

### `./render`: render your network

```
./render NET.json --shot PAIR_ID [--frames N] [--res WxH] [--exposure E] [--out PATH] [--compare]
./render NET.json --set reference [--frames N] [--res WxH] [--out DIR/] [--compare]
./render NET.json --shot SHOT --sun SUN [...]                     # free sun for a reference camera
./render NET.json --camera X,Y,Z --look-at X,Y,Z [--fov DEG] --sun SUN [...]   # any camera
```

- `NET.json`: your network file, anywhere in your directory (normally in `work/`).
- `--shot PAIR_ID`: one of the 12 reference pairs (`front_noon` …); its camera, sun and exposure.
- `--set reference`: all 12 reference pairs in one call.
- Free modes: `SHOT` is one of `front`, `side`, `backlit`, `below`, `distant`; `SUN` is a preset (`noon`,
  `afternoon`, `golden`, `sunset`) or `ELEV,AZ` in degrees (elevation 1–89, azimuth clockwise from north; the
  presets have azimuth 250). With `ELEV,AZ`, `E_sun` is computed exactly from section 3.3; `L_sky`, `L_gnd` and the
  sky map are interpolated linearly in elevation between the two neighbouring presets (held at the end presets
  below 2° and above 55°) and the map is rotated to the azimuth. Free views have no reference image; the default
  exposure is the reference exposure for that shot and preset if there is one, otherwise that of the nearest preset
  (noon 0.315, afternoon 0.3969, golden 0.5, sunset 2.0). `--camera` defaults to a 30° field of view.
- `--frames N` (1–256, default 16), `--res WxH` (up to 1280x800; default 640x400, field of view kept),
  `--exposure E` (PNG only; the `.npy` is linear).
- `--out`: a `.png` path for one view, or a directory ending in `/`. Default `work/renders/`, file name
  `<view>_<network file name>.png`. Output must be inside `work/`.
- Writes `<out>.png` (tone-mapped), `<out>.npy` (linear radiance, float32, H × W × 3, row 0 at the top) and
  `<out>.json` (settings, timings, number of network evaluations).
- `--compare` (reference pairs): also writes `<out>_compare.png`, your render beside the reference, and prints the
  linear error `mean|render − ref| / mean(ref)` over the whole image and the per-channel ratio of means
  `mean(render) / mean(ref)` (these include the sky background, which is identical in both). The numbers need the
  full 640 × 400 resolution; at another `--res` only the picture is made.

### `./inspect`: watch the neurons

```
./inspect NET.json --shot PAIR_ID --pixel X Y [--samples K|all] [--frame F] [--res WxH] [--json work/OUT.json]
```

(`--shot SHOT --sun SUN` and `--camera … --look-at … --sun SUN` work as for `./render`.)

For pixel (X, Y) (column, row; (0, 0) top left, at the render's resolution) it re-marches that pixel's ray inside the
shader, with the same jitter as frame F of a render, and prints: the ray direction, where it enters and leaves the
box, the number of accepted samples (d > 0.002), `T` at the end; `E_sun`, `L_sky`, `L_gnd`; the pixel's radiance split
into in-scattering and `T_final · sky`, and its 8-bit value; then, for K samples spread over the ray's
contribution (default 8), the sample position, `d`, `T`, `1 − exp(−σ_t Δ)`, every probe value, every unit's
activation in every layer, the 9 outputs, `S` and the sample's contribution to the pixel. `--json` writes every
sample's full record to a file in `work/`.

A large network prints many lines per sample. Tool output beyond about 30,000 characters is cut short (Claude Code
saves the full text to a file you can Read; that file is the one thing outside this directory you may read), so
prefer a few samples on screen and `--json` plus your own
`./python3` script for the rest.

### `./python3`: your own scripts

`./python3 work/script.py [args]` or `./python3 -c "code"`. Python 3.12 with numpy and the standard library only.
Scripts must be `.py` files. It reads only inside your directory and writes only inside `work/`; it cannot start
other programs, reach the network, change directory, open a directory as a file, take `dir_fd=` arguments,
introspect frames or the garbage collector, set tracing hooks, or re-import or reload a module. Give paths as `str`
or `pathlib.Path`. numpy's least-squares and fitting routines (`numpy.linalg.lstsq`, `numpy.linalg.pinv`,
`numpy.polyfit`, the `numpy.polynomial` fit functions and `.fit` methods) and `statistics.linear_regression` are
disabled, because fitting is not allowed (section 8). Use it to compute weights from your own formulas, to write
`network.json`, to simulate light transport on idealised shapes you build (section 8, rule 2), and to measure your
renders against the references (for example region averages and colour ratios from the `.npy` files). A number you
take from a measurement reaches your network only by you typing it into your script (section 8, rule 1).

## 8. Rules

1. **No training, fitting, optimisation or automated search of any kind.** In particular:
   - no gradient descent or other iterative update of weights toward a target, however it is written;
   - no least squares or regression in any form (including normal equations with `solve`, `inv` or `svd`, or
     `statistics.linear_regression`);
   - no weight, scale or colour computed in code from reference pixels or from the difference between a render and
     a reference (for example `k = ref.mean() / render.mean()`): measure, think, then type the number you choose
     into your script with a comment saying why;
   - no rendering units or terms as separate images and combining those images by computation to choose weights
     (the image is linear in the last layer, so that is a least-squares fit);
   - no sweeps: rendering several values of one setting and keeping whichever scores best is a search, even by
     hand one call at a time, so change a value because you have a reason, then look;
   - no making the network recognise a particular reference shot or preset (by view direction, sun elevation or
     height range) to correct it there: every term must make physical sense for any camera and any sun.

   Every weight comes from your reasoning about the physics and from your own revision after looking at renders
   and numbers, as an artist would: you may look at a render, measure it, think, and change a weight for a reason,
   as many times as you like.
2. What is fine: computing a value from a physical formula (an integral of the phase function, a transmittance, an
   attenuation constant derived from the medium); measuring your renders and the references and looking at them;
   and **simulating light transport yourself** with numpy on idealised shapes you build (for example Monte Carlo
   through a slab or a sphere of this medium) and reading the numbers to inform your choices. Your simulations
   obey rule 1 too: do not fit, regress or solve against their results; read them and type the numbers you choose.
   Their input is shapes you build yourself: not the reference images, and not the cloud's density beyond what
   `./render` and `./inspect` show you.
3. No scipy, torch, sklearn or similar libraries. numpy only.
4. Don't modify or work around the tools or the `./python3` guard, and don't read anything outside this directory,
   except the file Claude Code itself names when it saves a long tool output.
5. Keep `work/NETWORK.md` up to date: what every unit computes and why, layer by layer.

## 9. What you deliver

All in `work/`:

1. **`work/network.json`**: your final network.
2. **The script(s) that generate it**, for example `work/build_network.py`, so that running it (for example
   `./python3 work/build_network.py`) rewrites `work/network.json` exactly. The script that builds `network.json`
   must not read any image (`.npy` or `.png`): every number in it is a literal you chose or a physics formula.
   Every weight should be traceable to a formula or a stated choice in these scripts.
3. **`work/NETWORK.md`**: what the network computes, unit by unit; the physical reasoning behind the design; the
   revisions you made after looking at renders and why; what still differs from the references and why; the command
   that rebuilds `network.json`.

Renders, notes and scratch scripts may stay in `work/` too.
