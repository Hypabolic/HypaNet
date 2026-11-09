# HypaNet Clouds embed

A self-contained web page that renders the cloud with Fable's r1 network
(`runs/r1/author/work/network.json`, unchanged) in real time with three.js and WebGL2. Drag the sky and the sun
moves; the cloud relights immediately. Design and contracts: [DESIGN.md](DESIGN.md).

`web/` is the whole site. Copy it anywhere static; it makes no requests outside its own directory.

## Build

Everything in `web/gen/` and `web/data/` is generated and committed. Rebuild from `clouds/` with:

```sh
# 1. compile the network to GLSL (web/gen/: net_eval.glsl, bake.glsl, manifest.json)
uv run python embed/compile/compile_net.py --live p0,p1,p2,p3
# 2. per-sun-elevation light table (web/data/light_table.json; ~1-6 min, 8 workers)
uv run python embed/compile/light_table.py
# 3. density grid (web/data/density.u8.gz + density.json from scene/out/)
uv run python embed/compile/pack_density.py
```

`web/vendor/` is three.js 0.183.2, copied from `renderer/vendor/` with its licence. `web/pyramid.js` is a copy of
`renderer/pyramid.js`, so the blurred-density levels are identical to the research harness's.

## Serve locally

Any static server works. Module scripts and workers need http(s), not `file://`:

```sh
cd clouds/embed/web && python3 -m http.server 8000
# open http://localhost:8000/?view=backlit&el=6&az=250
```

`index.html` takes these query parameters:

| Parameter | Meaning |
|---|---|
| `view` | `front`, `backlit`, `arc` (label "Valley") or `sunset` |
| `el`, `az` | Sun elevation and azimuth in degrees. Azimuth runs clockwise from north; north is -z. |
| `autoplay=1` | Sweeps the sun along the view's arc. Ignored when the reader prefers reduced motion. |
| `harness=1` | Test hook only (see below). |

## Embed

**iframe** (simplest; the page fills the frame):

```html
<iframe src="/clouds/index.html?view=front&el=20&az=250" title="A cloud lit by a neural network"
        style="width:100%;aspect-ratio:16/10;border:0;border-radius:8px" loading="lazy"
        allow="fullscreen"></iframe>
```

**ES module** (into your own element; the UI lives in a shadow root, so your CSS and ours do not mix):

```html
<div id="sky" style="width:100%;aspect-ratio:16/10"></div>
<script type="module">
  import { mountClouds } from '/clouds/clouds.js';
  const clouds = await mountClouds(document.getElementById('sky'), {
    view: 'backlit', elevation: 8, azimuth: 250, autoplay: false,
    // base: '/clouds/'   // default: the directory clouds.js was loaded from
  });
  // clouds.setSun(5, 270); clouds.setView('sunset'); clouds.play(); clouds.pause(); clouds.destroy();
</script>
```

`mountClouds(container, opts)` resolves to `{ setSun(el, az), setView(name), play(), pause(), destroy(), sun, view,
ready, engine }`. It resolves even when loading fails; the failure is shown in the container and `ready` rejects.
`destroy()` frees the WebGL context straight away, so a host may keep the controller around.

To unmount while the page is still loading, pass an `AbortSignal` as `signal`. Aborting it destroys the mount: the
promise resolves to the destroyed controller, `ready` rejects with an `AbortError`, and nothing more is allocated.

Other options:

| Option | Default | Meaning |
|---|---|---|
| `frames` | 96 | Still frames accumulated before the page stops drawing |
| `blend` | 0.3 | Weight of a new frame while the sun moves |
| `targetMs` | 12 | GPU budget for the march in each animation frame |
| `maxDpr` | 2 | Highest device-pixel ratio used |
| `maxPixels` | 2.4e6 | Most internal pixels |
| `dragBake` | 0.5 | Sun-bake grid scale while the sun moves |
| `glare` | true | Draw the glare around the sun |
| `signal` | none | An `AbortSignal` that destroys the mount, also while it is loading |

If clouds.js is imported from another origin, the pyramid is built on the main thread, because a cross-origin
worker is not allowed. Loading then pauses the page for about half a second.

## How it renders

On load the page:

1. inflates `density.u8.gz` with DecompressionStream;
2. builds the blur pyramid in a worker;
3. compiles the shaders;
4. bakes the two static textures once.

For each animation frame:

- **Sun bake.** When the sun moves, the sun bake is redrawn: at a 64x40x64 grid while the sun is moving, then once at
  128x80x128 when it stops.
- **Sky.** The sky map is redrawn on the GPU. It is the transmittance LUT plus the sky view, from the same model as
  `scene/sky.py`: at 512x256 while the sun is moving, then once at 1024x512 when it stops.
- **March.** The march takes one sample per pixel, with the same sampling as the research harness.
  - While the sun moves, frames are blended (weight 0.3). The internal resolution adapts to keep the march near
    12 ms; GPU time comes from EXT_disjoint_timer_query_webgl2, or from frame pacing when that is missing.
  - When the sun stops, the page builds a running mean of 96 frames at full resolution, drawn in strips if one frame
    would be too slow, and then stops drawing.
- **Exposure and tone map.**
  - Auto-exposure runs on the GPU and never reads back. It follows 0.75 of the log-key change, smoothed over about 0.4 s.
  - The image is tone-mapped with ACES and sRGB.
  - The sun disk is drawn only where the march's T >= 0.002.
  - The glare around the sun uses the three Gaussians from `video/make_video.py`, scaled by how much of the disk is
    visible.

The loop pauses when the page is off screen or the tab is hidden. A resize or a device-pixel-ratio change redraws like
a sun move. After a WebGL context loss (a mobile tab in the background, a GPU reset) the page rebuilds its bakes and
targets when the browser restores the context.

Lighting depends only on the sun's elevation. `web/data/light_table.json` holds E_sun by altitude, L_sky and L_gnd;
`web/sky.js` reads it with `lightAt`.

## Sizes

Measured on 2026-11-09:

| File | Bytes | gzip -9 |
|---|---|---|
| data/density.u8.gz | 1,060,310 | (already gzip) |
| vendor/three.core.min.js | 382,617 | 100,384 |
| vendor/three.module.min.js | 359,158 | 84,994 |
| data/light_table.json | 155,807 | 52,960 |
| clouds.js | ~55,000 | ~18,000 |
| everything else (shaders, gen/, sky.js, pyramid*.js, index.html) | ~65,000 | ~20,000 |
| **page load total** | **~2.08 MB** | **~1.34 MB** with a gzip-serving host |

GPU memory is about 18 MB of bake textures, 26 MB of density and pyramid, plus the frame buffers: four RGBA32F
targets of up to 2.4 Mpx each.

## Browser requirements

- WebGL 2 with `EXT_color_buffer_float`, and 3D textures of at least 774 texels along an axis (most GPUs allow 2048).
  A build whose bakes `compile_net.py` had to store as RGBA32F (a baked channel bounded above 6e4, beyond half
  floats; not r1) also needs `OES_texture_float_linear`.
- `DecompressionStream`: Chrome 80+, Firefox 113+, Safari 16.4+.
- ES modules. Module workers are optional; without them the pyramid is built on the main thread.

If WebGL 2 or float render targets are missing, the container shows a short message instead.

## Tests

```sh
uv run python embed/tests/test_page.py [all|load|shots|drag|timing|score]   # writes embed/tests/out/page/
uv run python embed/tests/test_sky.py                                         # GPU sky and light table vs scene/sky.py
uv run python embed/tests/eval_compiled.py exact                              # compiled vs interpreter (harness)
```

`?harness=1` exposes `window.cloudsHarness`:

- `render({ shot | camera, elevation, azimuth, res: [w, h], frames, light, sunDisk })` returns a linear RGB
  `Float32Array` (row 0 at the top) plus per-frame GPU-waited timings. `light: { E_sun, L_sky, L_gnd, sky_h }`
  replaces the light table with a reference preset's lighting.
- `bakeTiming(reps)` times the sun bakes.
- `timing()` returns load, bake and per-frame timings.
- `state()` and `controller` report and drive the interactive view.

`test_page.py score` renders front_noon and backlit_golden through this hook and scores them with
`scripts/score.py`. It also compares them pixel by pixel with the research harness's compiled render.
