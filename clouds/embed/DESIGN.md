# HypaNet Clouds embed: compile, bake, drag the sun

Goal: a self-contained web page (three.js, WebGL2) that renders the cloud with Fable's r1 network
(`clouds/runs/r1/author/work/network.json`, unchanged) in real time. A blog reader drags the sun and the cloud
relights immediately. It has to be embeddable in a blog post (an iframe, plus an ES-module `mountClouds()`).

Today the research renderer (`clouds/renderer/`) interprets the network from a weight texture. It evaluates 41
probes per march sample, 37 of them blurred-density reads costing about 74 3D-texture fetches, and it
accumulates hundreds of passes. Two steps make it real-time:

1. **Compile.** Turn `network.json` into straight-line GLSL: drop zero weights, fold constants, and alias
   pass-through units. The output must match the interpreter to float precision.
2. **Bake.** Move every first-layer linear combination of density probes that does not depend on the view
   direction into 3D textures:
   - **static:** world-frame probes, baked once at load;
   - **sun:** light-frame probes, rebaked whenever the sun moves.

   A march sample then reads a few baked texels plus the remaining live probes. Only the view-frame probes have
   to stay live.

## Layout (all new code under `clouds/embed/`)

```
clouds/embed/
  DESIGN.md                this file
  compile/compile_net.py   the compiler (python, run with `uv run python` from clouds/)
  compile/light_table.py   per-sun-elevation lighting table (python, uses scene/sky.py)
  tests/                   verification scripts (python + Playwright, reuse renderer/harness_client.py)
  web/                     the static, embeddable site — everything a blog needs, nothing else
    index.html             full-viewport page for an iframe; query params ?view=&el=&az=
    clouds.js              ES module: export async function mountClouds(container, opts) -> controller
    shaders/*.glsl         hand-written: march, bake driver, sky (transmittance LUT + sky view + sun disk),
                           exposure, tone map, fullscreen vert
    gen/                   compiler output (committed): net_eval.glsl, bake.glsl, manifest.json
    data/                  density.u8.gz (gzip of scene/out/density.u8, ~1.1 MB), density.json, light_table.json
    vendor/                three.js 0.183.2 copied from clouds/renderer/vendor (+ THREE_LICENSE)
```

## Network facts (read these, don't re-derive)

- Probes: 41. Indices 0-9 are light-frame +s at 5…975 m (`tau_f`), 10-15 are light +s (`tau_c`, blur 150-700),
  16 and 18 are world (dist 0, blur 150/600), 19 and 20 are light ∓s at ±50 m blur 200 (`Gs`), 21 and 22 are
  view ±e2 at ±50 m blur 200 (`Gv`), 23-34 are world ±y (`tau_up`/`tau_dn`), 35 and 36 are world ∓y 75 m blur
  300, 37 is cos_view_sun, 38 is view_up, 39 is sun_up, 40 is const 1.
- After pruning zeros, 232 of the 2,957 weights remain. Layers 1-3 are mostly identity pass-throughs.
- First-layer density combinations, grouped by dependency (combos equal up to scale share a channel):
  - **static:** tau_up (6 probes), tau_dn (6), p16, p18, and (p35 − p36).
  - **sun:** tau_f (p0-p9), tau_c (p10-p15), and (p19 − p20).
  - **view (live):** (p21 − p22).

  Everything else in layer 0 is analytic in v·s, v.y or s.y.

## Contracts

### Host shader → compiled code (provided by any host: the research harness and the embed march)

- `float densProbe(vec3 x, int lo, float t)`: blurred density, with exactly the semantics of
  `renderer/shaders/march.frag.glsl`. It returns 0 when the probe point is outside the box and blends pyramid
  levels lo and lo+1 with weight t.
- Uniforms: `uSun` (toward-sun unit vector), `uLu` = normalize(cross(s, +y)) (fallback (1,0,0)),
  `uLw` = cross(uLu, s), `uBoxMin`, `uBoxMax`.
- The blur → (lo, t) mapping is the one in `renderer/net.js`: lv = blur ≤ voxelMean ? 0 :
  min(log2(blur/voxelMean), 7), lo = floor, t = frac, with voxelMean = mean voxel size from `density.json`. Read
  `net.js` and use the identical formula, including the float-rounding behaviour. The compiler bakes lo and t in
  as constants.

### Compiled code → host

`gen/net_eval.glsl` defines
`void evalNet(vec3 x, vec3 v, vec3 e2, vec3 e3, out vec3 aSun, out vec3 aSky, out vec3 aGnd)`. The signature
is identical to the interpreter's, so the file is a drop-in replacement. Activations reproduce the
interpreter's clamps exactly: exp clamps its argument to ±80, log uses max(x, 1e-8).

`gen/bake.glsl` defines one function per bake texture, `vec4 bakeK(vec3 x)` (unused channels are 0). It also
declares the samplers `uniform highp sampler3D uBakeK;`, which `net_eval.glsl` reads with
`texture(uBakeK, (x - uBoxMin) / (uBoxMax - uBoxMin))`. The bake grids are cell-centred over the box: texel i
sits at boxMin + (i + 0.5) · size / n.

`gen/manifest.json` contains:
- `{ network_sha256, density_sha256, voxel_mean, live_probes: [...] }`
- `textures: [{ name: "uBake0", fn: "bake0", class: "static" | "sun", dims: [nx, ny, nz], format: "RGBA16F",
  channels: [{ combo: "...", probes: [...], bound, scale_note }] }]`. `bound` = Σ|coef| bounds the channel
  (densProbe is in [0, 1]); a texture with a bound above 6e4 is written as `"RGBA32F"` instead (half floats overflow
  at 65504), and the host then needs `OES_texture_float_linear`.
- `stats: { weights_total, weights_nonzero, fetches_per_sample_before, fetches_per_sample_after }`

The compiler also has a `--no-bake` mode in which every probe is live. That output must match the interpreter
to float precision, and it is the exactness test.

### Lighting (`compile/light_table.py` → `web/data/light_table.json`)

All lighting depends only on sun elevation; azimuth only rotates the sky. The table runs over elevations
from −10° to 90°: 0.1° steps in [−6°, 8°] and 1° steps elsewhere. Each entry holds:
- `E_sun_lut`: 32 altitudes (y0 = 0, dy = 150), disk-averaged exactly like `video/make_video.py: E_disk`;
- `L_sky`: the mean upper-hemisphere radiance at 2 km (as `frame_light`);
- `L_gnd`: albedo/π · (E_ground · max(s.y, 0) + E_sky) at 0 m.

The page interpolates linearly in elevation and uses the same shader uniforms as the video path (`uEsunLut`,
`uLsky`, `uLgnd`).

The backdrop sky and the sun disk are computed on the GPU from the same model as `scene/sky.py`:
- a transmittance LUT pass, built once (port of `TransmittanceTable`, same parameterisation);
- a sky-view pass rebuilt when the sun moves (port of `sky_radiance` at the camera altitude, equirect, with the
  lower hemisphere set to L_gnd);
- the sun disk per pixel: E_TOA · exp(−tau(cam_h, d·ŷ)) / Ω_sun · limb, where limb is
  (1 − 0.6(1 − μ))/0.8. It is visible only where the march's T ≥ 0.002 and d.y > 0, as in the video shader.

The GPU sky must match the Python `build_sky_map` to within ~2% where the radiance is not tiny.

### Page behaviour

**Interaction.** Pointer drag on the canvas moves the sun: the sun goes to the view ray under the pointer,
with elevation clamped to [−8°, 89°]. A small compass/dome widget sets any azimuth and elevation, including
behind the camera. There are 3-4 view presets (pick good ones from `scene/out/shots.json` and the two video
cameras in `video/make_video.py`). A play button sweeps the sun along an arc. Keyboard arrows also move the
sun. `prefers-reduced-motion` disables auto-play.

**Rendering:**
- **March:** port the harness march (jittered step, d > 0.002 gate, T < 0.002 stop, `L += T·albedo·S·a`), with
  the compiled `evalNet` and E_sun by altitude.
- **Accumulation:** while the sun moves, show a 1-sample-per-pixel temporal blend. When the sun stops,
  progressively accumulate up to N frames, then stop drawing.
- **Resolution:** the internal resolution adapts to hold about 60 fps.
- **Visibility:** pause when off-screen (IntersectionObserver) or when the tab is hidden.
- **Exposure:** auto-exposure follows 0.75 of the log-key change, smoothed over about 0.4 s and computed on the
  GPU without stalling readbacks.
- **Output:** ACES tone map plus sRGB, plus an optional cheap bloom around the sun.

**Loading.** Show a poster or "loading" state, decompress `density.u8.gz` with DecompressionStream, build the
pyramid (reuse `renderer/pyramid.js` logic), run the static bake, then start. Show a clear message when WebGL2
or EXT_color_buffer_float is missing.

**Embedding:**
- `mountClouds(container, { base, view, elevation, azimuth, autoplay })` returns
  `{ setSun(el, az), setView(name), destroy() }`.
- `index.html` fills the iframe.
- No external network requests: three.js is vendored and paths are relative to `base`.

**Test hook.** With `?harness=1`, `window.cloudsHarness` exposes `render({ shot | camera, elevation, azimuth,
res, frames }) → Float32Array linear RGB` and per-frame timing. The tests use it to score the page against
the Mitsuba references and to time it.

## Rules

- Never modify `network.json` or the scoring references. Never open or create held-out sets; the reserved seeds
  are 20261104, 20261109, 20261110 and 20261111.
- Tune choices (bake resolution, which probes stay live, step size) on the **reference** split only. Report the
  held-back split once, at the end, with every number labelled by split.
- The research harness default path (`renderer/`) must stay unchanged: `renderer/tests/test_interpreter.py`
  gives ALL PASS and the reference renders do not change. Any harness hook must be opt-in.
- Don't commit. Don't spend money. Python runs via `uv run python` from `clouds/`.
