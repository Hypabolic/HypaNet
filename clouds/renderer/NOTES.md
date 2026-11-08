# renderer/ notes (SPEC §4 renderer, render / inspect / viewer tools)

## Run it (from `clouds/`)

```
uv run python renderer/render.py NET.json --pair front_noon [--frames 16] [--res 320x200] [--out x.png] [--compare [REF.png]]
uv run python renderer/render.py NET.json --set reference --out renderer/out/r1/        # 12 pairs (or heldback / all)
uv run python renderer/render.py --mode exact-single --pair side_noon [--no-background]  # HARNESS ONLY
uv run python renderer/inspect.py NET.json --pair front_noon --pixel 330 170 [--samples 8|all] [--json out.json] [--harness-exact]
uv run python renderer/serve.py      # then http://127.0.0.1:8765/renderer/viewer.html?scene=/scene/&net=tests/nets/single_scatter.json[&harness=1]
uv run python renderer/tests/make_test_nets.py && uv run python renderer/tests/test_interpreter.py   # self-checks (ALL PASS)
```

`render.py` writes `<out>.png` (the shader's tone map), `<out>.npy` (linear float32 RGB, H×W×3, row 0 at the top)
and `<out>.json` (timings, draws, evaluations per frame, tone-map self-check, GPU string). `--pair` takes any
`<shot>_<preset>` (exposure from `shots.json`); `--shot NAME --preset P` also works. `--compare` with no path
uses `reference/out/<pair>.png` and, when the `.npy` sits next to it, prints the linear error
(mean |render − ref| / mean ref and the per-channel ratio of means) in the footer.

## Files

| file | what |
|---|---|
| `package.json`, `package-lock.json` | pins `three@0.183.2` (released 2026-02-28). `node_modules/` is only the source of the vendored copy |
| `vendor/three.module.min.js`, `vendor/three.core.min.js`, `vendor/OrbitControls.js`, `vendor/THREE_LICENSE` | vendored three.js (the module build imports `three.core.min.js`, so both are needed); no CDN at render time |
| `shaders/march.frag.glsl` | the ray marcher, MLP interpreter, probes, exact mode, inspect debug path (one source, variants by `#define`) |
| `shaders/tonemap.frag.glsl`, `shaders/fullscreen.vert.glsl` | SPEC tone map (twin of `scene/tonemap.py`), full-screen pass |
| `main.js` | `CloudRenderer`: loads scene, builds textures, materials, tile/pass scheduler, readback, inspect |
| `net.js` | `network.json` validation (all limits, clear errors) and packing into the weight texture |
| `pyramid.js` | blurred density pyramid (atlas) |
| `index.html` + `harness.js` | headless page driven by Python through `window.harness` |
| `harness_client.py` | local static server (`/renderer/`, `/scene/` mounts) + Playwright Chromium with the Metal flags |
| `render.py`, `inspect.py` (launcher) + `inspect_ray.py`, `serve.py` | CLI tools |
| `viewer.html` + `viewer.js` | interactive viewer |
| `tests/make_test_nets.py`, `tests/nets/*.json`, `tests/test_interpreter.py` | harness-side test networks and self-checks (keep out of the kit) |

## Choices SPEC leaves open, and deviations

1. **Blur pyramid = precomputed box filters, not a hardware mip chain.** Level k (k = 0..7) is the density
   convolved with a *centred* box `2^k` voxels wide per axis (voxels as piecewise-constant cells; outside the box
   counts as 0); k = 0 is the raw grid. Level k ≥ 1 is stored on a grid of spacing `max(1, 2^k / 4)` voxels with one
   padding cell per side, all levels stacked in one R8 3D atlas (258 × 162 × 774), built in JS at load (~0.5 s).
   `blur` (m) maps to a fractional level `L = log2(blur / 6.979 m)` (6.979 m = mean voxel size), clamped at 0
   (blur ≤ one voxel = raw trilinear) and interpolated linearly between the two neighbouring levels.
   Widest level = 128 voxels = 893 m; a larger `blur` is a validation error. Trilinear interpolation of the
   stored samples widens the effective kernel by ~6 % (std); measured against an exact numpy box filter at
   random sample points: ≤ 0.04 absolute (`test_interpreter.py`). A mip chain would have been grid-aligned
   (position-dependent) and 22 % wider, so I did not use one.
2. **View frame is orthonormalised per pixel:** `(v, e2, e3)` with `e2 = normalize(cross(v, cam_up))`,
   `e3 = cross(e2, v)`. At the image centre these are exactly camera right / camera up (SPEC's wording); off
   centre SPEC's literal frame would not be orthonormal.
3. **Light frame fallback** when `s` is vertical: `u = (1, 0, 0)` (never happens with the presets).
4. **Probe validation:** `dir` must be unit length within 1e-3 (then renormalised); `dist`, `blur` ≥ 0;
   unknown keys are errors (catches typos such as `blurr`); comment keys (`name`, `note`, `comment`,
   `description`, `label`, `doc`, or anything starting with `_`) are ignored. `act` as a string or a per-unit
   list. All SPEC limits (64 probes, 6 layers, 128 units, 50,000 parameters, 9 outputs) give a clear message.
5. **March:** step `Δ = max(6 m, chord / 512)` (6 m for every shot: the longest chord through the box is
   2.77 km); sample k at `t0 + (k + ξ) Δ`, `ξ` hashed from pixel and frame. As SPEC words it, samples with
   `d ≤ 0.002` neither evaluate the network nor update T. Stop at `T < 0.002`; then `L += T · sky(v)`.
6. **Sub-pixel jitter per frame** (box pixel filter), so 16 frames also anti-alias; the Mitsuba reference should
   use `rfilter: box` (as `scene/preview_mitsuba.py` already does).
7. **Sky lookup:** manual bilinear from an RGBA32F texture (u wraps, v clamps), the map's pixel-centre
   convention from `scene/NOTES.md`. The viewer's azimuth slider rotates the lookup.
8. **Exact single-scattering mode (harness only):** at every accepted sample a shadow march toward the sun
   (5 m steps, jittered per sample, stops at optical depth 18), `a_sun = T_sun · p(dot(v, s))`, `a_sky = a_gnd = 0`.
9. **Tone map:** `tonemap.frag.glsl` is `scene/tonemap.py` (ACES fit, sRGB OETF, `floor(255 y + 0.5)`; NaN/inf → 0;
   input clamped at 1e4, where ACES is already 1, so float32 cannot overflow). `render.py` checks every image
   against numpy: at most 1 LSB on 1–6 pixels per image (float32 `pow` at rounding ties).
10. **MLP evaluation** works four units at a time (`vec4 × mat4` from four texels), with activations in
   `vec4[32]` arrays. That was twice as fast as one unit at a time and gives bit-identical output.

## The macOS GPU watchdog (important)

On this machine macOS **silently drops a draw** in which a GPU thread runs too long (reproducibly ~100–130 ms
per pixel, seen with a 42k-parameter network), and **reports no lost context**: the pixels just keep their old
values. After a second failure it ignores the process's GPU work altogether, and for a while also ignores a newly
started Chromium. Total draw time is not the issue: a 466 ms draw of a light network was fine. Measures:

- **Segmented march.** A ray is marched in passes of at most `cap` network evaluations per pixel; the march state
  (step index, T, nacc, L) goes to two float targets and the next pass resumes it. A finalize pass adds
  `T · sky(v)` and folds the frame into the running mean. The result is bit-identical to one uninterrupted march,
  because sample positions are computed from the step index. `cap` adapts so that a pass takes about
  `--budget-ms` (40 ms default). It is cut when a pass is slower and doubled when a pass takes under a quarter of
  that. A draw can never be faster than its slowest thread, so the budget also bounds per-thread time. A tile that
  is still over budget at `cap = 1` is split into four (160 × 100 tiles to start with).
- **Every draw is verified.** It writes a unique stamp into alpha, and the tile is read back and checked. The same
  readback tells which rays have finished. A dropped draw raises `GPU_DROPPED`; `render.py` then restarts the
  browser and retries with a third of the budget (twice at most) instead of saving a wrong image.
- **Loop bounds are uniforms, not constants.** With constant bounds the Metal compiler unrolled the nested
  loops. Separately, the first compile of a new shader variant once took 80–120 s, likely in the OS Metal compiler
  (the CPU was busy with system daemons at the time). Later compiles hit the Metal cache and take about 0.2 s.
  `render.py` reports `compile_ms` separately.

Determinism: re-rendering gives byte-identical `.png` and `.npy` (checked; adaptive passes and tiling do not
change results).

## Validation done here (quick checks; §6 proper is the validate agent's job)

- **Exact mode vs Mitsuba single scattering** (`volpath` max_depth 2, sun only, no sky, 4 × 1024 spp, 320 × 200,
  using `scene/preview_mitsuba.py`'s scene dict; background removed with `--no-background`). Mean ratio
  0.998 / 0.999 / 0.998 (side_noon / front_noon / backlit_golden); 4 × 4-block relative error 0.5 / 0.4 / 0.9 %.
  So the camera, the phase convention (Mitsuba's `hg` forward lobe is forward), the transmittance and the units
  agree.
- **Background vs Mitsuba** (max_depth 1, sky only), on sky pixels: mean ratio within 0.1–0.2 % per channel
  (distant_sunset, below_golden, backlit_sunset). The envmap orientation is right.
- **`test_interpreter.py`** (all pass):
  - numpy float64 forward pass vs shader on recorded probes: ≤ 4e-4 relative.
  - Scalar and raw density probes in all frames vs numpy: ≤ 1.6e-3, i.e. GPU 8-bit filter weights.
  - Blurred probes vs exact box: ≤ 0.04.
  - inspect's rebuilt pixel radiance vs `render --frames 1`: 1.2e-7 relative.
  - `tests/nets/single_scatter.json` (63 light-frame probes + HG from log/exp units) vs exact mode: 0.4 % and
    0.09 % mean abs error.
  - One caveat: `sin` of arguments beyond about ±50 is not comparable in float32, so those samples are skipped.

## Timings (M4 Pro, 640 × 400, 16 frames; render time, plus ~0.8 s browser start-up)

| what | params | time |
|---|---|---|
| exact single scattering (front / side / backlit_golden / side_sunset) | – | 2.4 / 2.8 / 2.5 / 2.4 s |
| `tests/nets/single_scatter.json` | 230 | 3.2 s |
| 64 → 32 → 9 | 2.4k | 6.5 s |
| 64 → 64 → 64 → 9 | 8.9k | 14.3 s |
| 64 → 128 → 128 → 9 | 25k | ≈ 36 s (2.3 s / frame) |
| 64 → 128 → 128 → 128 → 9 | 42.5k | ≈ 57 s (3.6 s / frame) |

About 2.0–2.5 M network evaluations per frame for these shots. The viewer accumulates about 5 frames/s with a
small network.

## inspect

Uses the shader itself (`DEBUG_RAY` variant). Fragment (col, row) re-marches the chosen pixel's ray with the same
jitter as frame `--frame`, then writes record values `4·col … 4·col+3` of accepted sample `row − 1`. Row 0 is a
summary: sample count, `t0`, `t1`, `Δ`, `T_final`, `sky(v)` and `v`. Python rebuilds
`L = Σ T_k ω S_k (1 − e^{−σΔ}) + T_final sky`. `--samples K` prints K samples spread over the ray's cumulative
contribution weight; `--json` dumps every sample.

`renderer/inspect.py` would shadow the standard library's `inspect` for any script run from `renderer/`. It is
therefore a launcher: run as a script, it moves its directory to the end of `sys.path` and runs `inspect_ray.py`;
imported by accident, it loads the real stdlib module. `render.py` also moves its directory to the end of
`sys.path`.

## Viewer

- Orbit camera (OrbitControls, target = the shot's look-at); shot picker (camera, preset, exposure from
  `exposure_table`); preset buttons; exposure slider (log2); resolution 320/640/960 wide.
- `network.json` from a file picker, by drag and drop, or with `?net=URL`. Validation errors are shown on the image.
- Progressive accumulation runs in ≤ 20 ms slices per animation frame, up to 256 frames. It restarts on any change.
- **Sun sliders:** elevation 1–70° and azimuth 0–360°.
  - `E_sun` is recomputed exactly from the SPEC formula, using `shots.json` `sun_model`.
  - `L_sky`, `L_gnd` and the sky map are interpolated linearly in elevation between the two bracketing presets
    (held at the end presets outside 2–55°), and the map is rotated about y to the new azimuth.
  - Nothing else is recomputed. A preset button restores the exact preset.
- `?harness=1` adds the exact single-scattering mode.

## Harness-only features (strip from the author kit)

These are:
- `render.py --mode exact-single` and `--no-background`;
- `inspect.py --harness-exact`;
- the shader's `MODE_EXACT` / `DEBUG_EXACT` paths and `uBgScale`;
- the viewer's `?harness=1`;
- `tests/`.

They are refused unless the page gets `?harness=1`. Python asks for that only when `HARNESS_FEATURES = True` in
`harness_client.py`. The kit copy should set that to `False`, or better, drop those code paths and `tests/` entirely.

## Open problems / notes for others

- **The `side` shot's description is wrong about the sun's side.** For `side`, camera right =
  `cross(forward, up)` ≈ (−0.94, 0, 0.34), and `toward_sun · right` ≈ +1. So the sun is on the screen's **right**,
  not the left as `shots.json` says. Mitsuba renders it the same way as the renderer, so only the description is
  wrong. It belongs to `scene/`; I did not edit it.
- **Heavy networks are slow.** The interpreter is latency-bound: local arrays, one texel fetch per four weights.
  A 25k–42k-parameter network takes 0.5–1 min per 16-frame shot; a few-thousand-parameter one takes under 10 s.
- **Some draws exceed the budget.** Draws occasionally reach 120–190 ms when a new tile is denser than the last
  (throughput-bound, many threads, so per-thread time stays far lower). None has dropped since the segmented march.
- **`sin` precision.** GPU `sin` of large arguments (|x| ≳ 50) is imprecise, as float32 always is. Worth telling the
  author.
