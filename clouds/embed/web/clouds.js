// HypaNet Clouds embed: the cloud lit by Fable's r1 network (runs/r1/author/work/network.json, compiled to GLSL by
// embed/compile/compile_net.py), rendered in real time with three.js / WebGL2. See embed/DESIGN.md.
//
//   import { mountClouds } from './clouds.js';
//   const clouds = await mountClouds(document.getElementById('sky'), { view: 'front', elevation: 20, azimuth: 250 });
//   clouds.setSun(5, 270); clouds.setView('backlit'); clouds.destroy();
//
// Per frame: the sun bake (when the sun moved; a coarser grid while it is being dragged), the sky map (when the sun
// moved; a coarser map while it is being dragged), the march (one sample per pixel; a temporal blend while the sun moves, a running mean of up to N frames when
// it stops, then nothing), the GPU auto-exposure and the display pass (tone map, glare). Every path is relative to
// opts.base (default: this module's directory); the page makes no other network requests.
import * as THREE from './vendor/three.module.min.js';
import { SkyGPU, loadSkyShaders, loadLightTable, lightAt, setLightUniforms } from './sky.js';

const RAD = Math.PI / 180;
const BLUR_LEVELS = 8;                // renderer/net.js
const EL_MIN = -8, EL_MAX = 89;       // DESIGN.md: pointer drag clamps the sun's elevation to [-8, 89]
const HEADER = 'precision highp float;\nprecision highp int;\nprecision highp sampler2D;\nprecision highp sampler3D;\n';

// ---- views ----------------------------------------------------------------------------------------------------------
// Cameras from scene/out/shots.json (front, backlit) and video/make_video.py (arc, sunset: look(pos, yaw, pitch, fov)).
// sun: the view's starting sun [elevation, azimuth]; path(x), x in [-1, 1]: the play button's sun arc.
function look(pos, yawDeg, pitchDeg, fov) {
  const y = yawDeg * RAD, p = pitchDeg * RAD;
  const d = [Math.cos(p) * Math.sin(y), Math.sin(p), -Math.cos(p) * Math.cos(y)];
  return { position: pos, look_at: pos.map((v, i) => v + 1000 * d[i]), up: [0, 1, 0], fov_y_deg: fov };
}

export const VIEWS = {
  front: {
    label: 'Front', camera: { position: [-3100.986, 1150.0, 1128.666], look_at: [0, 1930, 0], up: [0, 1, 0], fov_y_deg: 30 },
    sun: [24, 250], path: (x) => [4 + 46 * (1 - x * x), 250 + 60 * x],
  },
  backlit: {
    label: 'Backlit', camera: { position: [3100.986, 1200.0, -1128.666], look_at: [0, 1950, 0], up: [0, 1, 0], fov_y_deg: 32 },
    sun: [8, 250], path: (x) => [-2 + 30 * (1 - x * x), 250 + 30 * x],
  },
  // video/make_video.py "arc": 3.6 km north of the cloud, 300 m up, facing south; the sun passes behind the cloud
  arc: {
    label: 'Valley', camera: look([0, 300, -3600], 180, 18, 50),
    sun: [22, 200], path: (x) => [26 - 21 * x * x, 180 + 40 * x],
  },
  // video/make_video.py "sunset": 20 m up, 4.8 km south-east, facing azimuth 298; the sun sets 28 degrees left of centre
  sunset: {
    label: 'Sunset', camera: look([2955, 20, 3782], 298, 12, 54),
    sun: [6, 270], path: (x) => [11.5 - 16.5 * x, 270],
  },
};
const PLAY_SECONDS = 26;              // one sweep of a view's path (there and back is two)

// ---- small helpers --------------------------------------------------------------------------------------------------
class UnsupportedError extends Error {}
const abortError = () => new DOMException('The cloud was destroyed while it was loading.', 'AbortError');

async function fetchOk(url, kind) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  return kind === 'json' ? r.json() : r.text();
}

/** Bytes of url, reporting (received, total) as they arrive. */
async function fetchBytes(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  const total = Number(r.headers.get('content-length')) || 0;
  if (!r.body || !r.body.getReader) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader();
  const chunks = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    n += value.length;
    onProgress(n, total);
  }
  const out = new Uint8Array(n);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

/** gzip -> bytes with DecompressionStream; bytes that are not gzip (a server that already inflated them) pass through. */
async function gunzip(bytes) {
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
  if (typeof DecompressionStream === 'undefined') throw new UnsupportedError('This browser cannot decompress the cloud (no DecompressionStream).');
  const s = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

/** pyramid.js in a module worker (the loading state keeps animating); on the main thread when workers fail. */
function pyramidAsync(bytes, dims) {
  const local = async () => {
    const { buildPyramid } = await import('./pyramid.js');
    const t0 = performance.now();
    return { pyr: buildPyramid(bytes, dims, BLUR_LEVELS), ms: performance.now() - t0, worker: false };
  };
  return new Promise((resolve) => {
    let w;
    try { w = new Worker(new URL('./pyramid_worker.js', import.meta.url), { type: 'module' }); } catch (e) { resolve(local()); return; }
    w.onmessage = (e) => { w.terminate(); resolve(e.data.error ? local() : { pyr: e.data.pyr, ms: e.data.ms, worker: true }); };
    w.onerror = (e) => { e.preventDefault(); w.terminate(); resolve(local()); };
    const copy = bytes.slice();
    w.postMessage({ bytes: copy, dims, levels: BLUR_LEVELS }, [copy.buffer]);
  });
}

function v3(a) { return new THREE.Vector3(a[0], a[1], a[2]); }
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const wrap360 = (a) => ((a % 360) + 360) % 360;

/** Toward-sun unit vector (scene/sky.py sun_dir): azimuth clockwise from north, north = -z, east = +x. */
export function sunVector(elDeg, azDeg) {
  const e = elDeg * RAD, a = azDeg * RAD;
  return [Math.cos(e) * Math.sin(a), Math.sin(e), -Math.cos(e) * Math.cos(a)];
}

const COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];

// ---- GPU timer (EXT_disjoint_timer_query_webgl2), results read a few frames later ------------------------------------
class GpuTimer {
  constructor(gl) {
    this.gl = gl;
    this.ext = gl.getExtension('EXT_disjoint_timer_query_webgl2');
    this.pending = [];
    this.active = null;
  }
  begin(label, meta) {
    if (!this.ext || this.active) return;
    const q = this.gl.createQuery();
    this.gl.beginQuery(this.ext.TIME_ELAPSED_EXT, q);
    this.active = { q, label, meta };
  }
  end() {
    if (!this.active) return;
    this.gl.endQuery(this.ext.TIME_ELAPSED_EXT);
    this.pending.push(this.active);
    this.active = null;
  }
  /** cb(label, ms, meta) for every finished query, in order. */
  poll(cb) {
    if (!this.ext) return;
    const gl = this.gl;
    const disjoint = gl.getParameter(this.ext.GPU_DISJOINT_EXT);
    while (this.pending.length) {
      const p = this.pending[0];
      if (!gl.getQueryParameter(p.q, gl.QUERY_RESULT_AVAILABLE)) break;
      const ns = gl.getQueryParameter(p.q, gl.QUERY_RESULT);
      gl.deleteQuery(p.q);
      this.pending.shift();
      if (!disjoint) cb(p.label, ns / 1e6, p.meta);
    }
    if (this.pending.length > 32) { this.pending.forEach((p) => gl.deleteQuery(p.q)); this.pending = []; }
  }
}

// =====================================================================================================================
// The renderer (no DOM besides its canvas)
// =====================================================================================================================
export class CloudsEngine {
  /**
   * opts.base: URL of web/ (shaders/, gen/, data/); opts.canvas; opts.harness: keep the drawing buffer for readback.
   * opts.frames: still frames to accumulate (96); opts.blend: weight of a new frame while the sun moves (0.3);
   * opts.targetMs: march budget per animation frame (12); opts.maxDpr (2); opts.maxPixels (internal, 2.4e6);
   * opts.dragBake: sun-bake grid scale while the sun moves (0.5); opts.glare (true); opts.step (6 m, the harness's).
   */
  constructor(opts) {
    this.opts = { frames: 96, blend: 0.3, targetMs: 12, maxDpr: 2, maxPixels: 2.4e6, dragBake: 0.5, glare: true,
      step: 6.0, ...opts };
    this.canvas = opts.canvas;
    this.timing = { load: {}, bakes: { static: [], sunFull: [], sunFast: [] }, frames: [] };
    this.errors = [];
    this.disposed = false;
  }

  /** load() checks this after every await: a mount destroyed while loading allocates nothing more. */
  throwIfDisposed() { if (this.disposed) throw abortError(); }

  /** Throws UnsupportedError when WebGL2 / EXT_color_buffer_float / a large enough 3D texture is missing. */
  initGL() {
    const attrs = { antialias: false, alpha: false, depth: false, stencil: false, premultipliedAlpha: false,
      preserveDrawingBuffer: !!this.opts.harness, powerPreference: 'high-performance' };
    const gl = this.canvas.getContext('webgl2', attrs);
    if (!gl) throw new UnsupportedError('This browser or device does not support WebGL 2, which the cloud needs.');
    if (!gl.getExtension('EXT_color_buffer_float')) throw new UnsupportedError('This GPU cannot render to float textures (EXT_color_buffer_float), which the cloud needs.');
    this.gl = gl;
    this.max3D = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);
    const r = new THREE.WebGLRenderer({ canvas: this.canvas, context: gl });
    r.autoClear = false;
    r.outputColorSpace = THREE.LinearSRGBColorSpace;
    r.debug.onShaderError = (g, program, vs, fs) => {
      const msg = `shader compile/link error:\n${g.getProgramInfoLog(program)}\n${g.getShaderInfoLog(vs)}\n${g.getShaderInfoLog(fs)}`;
      this.errors.push(msg);
      console.error(msg);
    };
    this.renderer = r;
    this.quadScene = new THREE.Scene();
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
    this.quad.frustumCulled = false;
    this.quadScene.add(this.quad);
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.timer = new GpuTimer(gl);
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    this.glInfo = { renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
      max3D: this.max3D, timerQuery: !!this.timer.ext };
  }

  /** Downloads, inflates, builds the pyramid, compiles, bakes. progress(fraction 0..1, label). */
  async load(progress = () => {}) {
    const t0 = performance.now();
    const base = this.opts.base;
    const T = this.timing.load;
    let dl = 0;
    progress(0.02, 'Loading');
    const shaderNames = ['fullscreen.vert.glsl', 'density.glsl', 'march.frag.glsl', 'bake.frag.glsl', 'sunvis.frag.glsl',
      'exposure.frag.glsl', 'tonemap.frag.glsl'];
    const [texts, gen, dj, table, skySrc, gz] = await Promise.all([
      Promise.all(shaderNames.map((f) => fetchOk(`${base}shaders/${f}`, 'text'))),
      Promise.all([fetchOk(`${base}gen/manifest.json`, 'json'), fetchOk(`${base}gen/bake.glsl`, 'text'), fetchOk(`${base}gen/net_eval.glsl`, 'text')]),
      fetchOk(`${base}data/density.json`, 'json'),
      loadLightTable(base),
      loadSkyShaders(base),
      fetchBytes(`${base}data/density.u8.gz`, (n, total) => { dl = total ? Math.min(1, n / total) : 0.5; progress(0.05 + 0.55 * dl, 'Loading'); }),
    ]);
    this.throwIfDisposed();
    T.downloadMs = performance.now() - t0;
    T.densityGzBytes = gz.length;
    const src = Object.fromEntries(shaderNames.map((f, i) => [f, texts[i]]));
    const [manifest, bakeSrc, netSrc] = gen;
    this.manifest = manifest;
    this.table = table;
    this.dj = dj;
    if (manifest.density_sha256 && dj.sha256 && manifest.density_sha256 !== dj.sha256) throw new Error('gen/ was compiled for another density grid (rerun compile_net.py)');
    this.requireExtensions();

    progress(0.62, 'Unpacking');
    let t = performance.now();
    const bytes = await gunzip(gz);
    this.throwIfDisposed();
    T.inflateMs = performance.now() - t;
    const dims = dj.dims;
    if (bytes.length !== dims[0] * dims[1] * dims[2]) throw new Error(`density: ${bytes.length} bytes, expected ${dims[0] * dims[1] * dims[2]}`);
    progress(0.7, 'Blurring');
    t = performance.now();
    const { pyr, ms: pyrMs, worker } = await pyramidAsync(bytes, dims);
    this.throwIfDisposed();
    T.pyramidMs = pyrMs;
    T.pyramidWallMs = performance.now() - t;
    T.pyramidWorker = worker;
    if (Math.max(...pyr.atlasDims, ...dims) > this.max3D) throw new UnsupportedError(`This GPU's 3D textures are too small (${this.max3D}).`);

    progress(0.82, 'Compiling');
    t = performance.now();
    this.setupScene(bytes, pyr, src, bakeSrc, netSrc, skySrc);
    T.uploadMs = performance.now() - t;
    t = performance.now();
    this.compileAll();
    T.compileMs = performance.now() - t;
    progress(0.9, 'Baking');
    t = performance.now();
    this.bakeStatic();
    T.staticBakeMs = performance.now() - t;
    T.totalMs = performance.now() - t0;
    progress(1, 'Ready');
  }

  setupScene(bytes, pyr, src, bakeSrc, netSrc, skySrc) {
    const dj = this.dj, dims = dj.dims;
    const tex3 = (data, nx, ny, nz) => {
      const tx = new THREE.Data3DTexture(data, nx, ny, nz);
      Object.assign(tx, { format: THREE.RedFormat, type: THREE.UnsignedByteType, minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter, wrapS: THREE.ClampToEdgeWrapping, wrapT: THREE.ClampToEdgeWrapping,
        wrapR: THREE.ClampToEdgeWrapping, unpackAlignment: 1, generateMipmaps: false });
      tx.needsUpdate = true;
      return tx;
    };
    this.densTex = tex3(bytes, dims[0], dims[1], dims[2]);
    this.atlasTex = tex3(pyr.atlas, ...pyr.atlasDims);
    const lvlScale = [], lvlOffset = [];
    for (let k = 0; k < BLUR_LEVELS; k++) {
      lvlScale.push(new THREE.Vector4(...pyr.scale[k], 0));
      lvlOffset.push(new THREE.Vector4(...pyr.offset[k], 0));
    }
    this.sky = new SkyGPU(this.renderer, skySrc);
    this.u = {
      uDensity: { value: this.densTex }, uAtlas: { value: this.atlasTex }, uLvlScale: { value: lvlScale }, uLvlOffset: { value: lvlOffset },
      uBoxMin: { value: v3(dj.box_min) }, uBoxMax: { value: v3(dj.box_max) }, uVoxel: { value: v3(dj.voxel_size) },
      uDims: { value: v3(dims) }, uSigmaMax: { value: dj.sigma_t_max }, uAlbedo: { value: dj.albedo },
      uSun: { value: new THREE.Vector3(0, 1, 0) }, uLu: { value: new THREE.Vector3(1, 0, 0) }, uLw: { value: new THREE.Vector3() },
      uCamPos: { value: new THREE.Vector3() }, uCamFwd: { value: new THREE.Vector3() }, uCamRight: { value: new THREE.Vector3() },
      uCamUp: { value: new THREE.Vector3() }, uTan: { value: new THREE.Vector2() }, uRes: { value: new THREE.Vector2() },
      uMaxSteps: { value: 512 }, uFrame: { value: 0 }, uPixelJitter: { value: 1 }, uStep: { value: this.opts.step },
      uPrev: { value: null }, uMeanN: { value: 1 }, uRejectHist: { value: 0 },
      uEsun: { value: new THREE.Vector3() }, uEsunLutOn: { value: 1 }, uEsunLut: { value: Array.from({ length: 32 }, () => new THREE.Vector3()) },
      uEsunLutY: { value: new THREE.Vector2(0, 1) }, uLsky: { value: new THREE.Vector3() }, uLgnd: { value: new THREE.Vector3() },
      uSkyMap: { value: this.sky.skyTexture }, uTransLut: { value: this.sky.lutTexture }, uSunDiskOn: { value: 1 },
    };
    const mat = (frag, uniforms, defines) => new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: src['fullscreen.vert.glsl'],
      fragmentShader: frag, uniforms, defines: defines || {}, depthTest: false, depthWrite: false });
    const dens = src['density.glsl'];
    this.marchMat = mat([HEADER, dens, skySrc.common, bakeSrc, netSrc, src['march.frag.glsl']].join('\n'), this.u);

    // bake targets: every texture at the manifest's grid; sun textures also at a coarser grid used while dragging
    const bakeFrag = [HEADER, dens, bakeSrc, src['bake.frag.glsl']].join('\n');
    const mk = (d, fmt) => {
      const rt = new THREE.WebGL3DRenderTarget(d[0], d[1], d[2], { type: fmt === 'RGBA32F' ? THREE.FloatType : THREE.HalfFloatType,
        format: THREE.RGBAFormat, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false,
        stencilBuffer: false, generateMipmaps: false });
      Object.assign(rt.texture, { wrapS: THREE.ClampToEdgeWrapping, wrapT: THREE.ClampToEdgeWrapping, wrapR: THREE.ClampToEdgeWrapping });
      return rt;
    };
    this.bakes = this.manifest.textures.map((t) => {
      const uniforms = { ...this.u, uBakeDims: { value: new THREE.Vector3() }, uBakeLayer: { value: 0 } };
      for (const o of this.manifest.textures) delete uniforms[o.name];   // never sample a bake target while writing one
      const b = { ...t, full: mk(t.dims, t.format), mat: mat(bakeFrag, uniforms, { BAKE_FN: t.fn }), fullKey: null, fastKey: null };
      if (t.class === 'sun') {
        b.fastDims = t.dims.map((n) => Math.max(8, Math.round(n * this.opts.dragBake)));
        b.fast = this.opts.dragBake < 1 ? mk(b.fastDims, t.format) : null;
      }
      this.u[t.name] = { value: b.full.texture };
      return b;
    });

    // 1 x 1 helpers: sun visibility, exposure ping-pong
    const one = () => new THREE.WebGLRenderTarget(1, 1, { type: THREE.FloatType, format: THREE.RGBAFormat,
      minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false, stencilBuffer: false, generateMipmaps: false });
    this.visRT = one();
    this.expoRT = [one(), one()];
    this.expoFront = 0;
    this.sunvisMat = mat([HEADER, dens, skySrc.common, src['sunvis.frag.glsl']].join('\n'), this.u);
    this.expoMat = mat(HEADER + src['exposure.frag.glsl'], { uImage: { value: null }, uPrevExpo: { value: null },
      uAdapt: { value: 1 }, uBias: { value: 0 }, uFixed: { value: 0 }, uFixedLogE: { value: 0 } });
    this.toneMat = mat(HEADER + src['tonemap.frag.glsl'], { uImage: { value: null }, uHold: { value: null }, uHoldMix: { value: 0 },
      uExpo: { value: null }, uSunVis: { value: this.visRT.texture }, uSunE: { value: new THREE.Vector3() },
      uGlare: { value: this.opts.glare ? 1 : 0 }, uSun: this.u.uSun, uCamFwd: this.u.uCamFwd, uCamRight: this.u.uCamRight,
      uCamUp: this.u.uCamUp, uTan: { value: new THREE.Vector2() }, uOutRes: { value: new THREE.Vector2() }, uFrame: { value: 0 } });
  }

  draw(material, target, rect) {
    this.quad.material = material;
    if (target) {
      target.viewport.set(0, 0, target.width, target.height);
      if (rect) { target.scissor.set(rect[0], rect[1], rect[2], rect[3]); target.scissorTest = true; } else target.scissorTest = false;
    }
    this.renderer.setRenderTarget(target);
    this.renderer.render(this.quadScene, this.quadCam);
  }

  /** Wait for the GPU (a 1-texel readback of target). Harness timing only: the interactive path never stalls. */
  sync(target) {
    const px = new Float32Array(4);
    if (target && target.isWebGL3DRenderTarget) {
      this.renderer.setRenderTarget(target, 0);
      this.gl.readPixels(0, 0, 1, 1, this.gl.RGBA, this.gl.FLOAT, px);
      this.renderer.setRenderTarget(null);
    } else this.renderer.readRenderTargetPixels(target, 0, 0, 1, 1, px);
    return px;
  }

  checkErrors() {
    if (this.errors.length) throw new Error(this.errors.join('\n'));
    if (this.gl.isContextLost()) throw new Error('WebGL context lost');
  }

  /** Extensions beyond initGL's: RGBA32F bakes (compile_net.py picks them when a channel could overflow half floats)
   * are sampled with linear filtering. Again after a context restore (extensions are per context instance). */
  requireExtensions() {
    if (this.manifest && this.manifest.textures.some((t) => t.format === 'RGBA32F') && !this.gl.getExtension('OES_texture_float_linear')) {
      throw new UnsupportedError('This GPU cannot filter float textures (OES_texture_float_linear), which this build of the cloud needs.');
    }
  }

  /**
   * After 'webglcontextrestored'. three.js has re-initialised its state on the same canvas: the Data3DTextures
   * re-upload from their arrays on first use, but every render target comes back empty. Rebuild what is drawn once
   * (transmittance LUT, static bakes, exposure) and invalidate what is drawn on change (sun bakes, sky map, the
   * sun-visibility texel: the caller re-applies the sun). extra: the caller's own render targets.
   */
  restoreGL(extra = []) {
    const gl = this.gl;
    // three.js keeps the 'dispose' listeners it attached before the loss; they would delete GL objects of the lost
    // context (a WebGL warning each). Drop them: objects used again get fresh listeners, the old ones died with the context.
    const sky = this.sky;
    const objs = [this.densTex, this.atlasTex, this.marchMat, this.sunvisMat, this.expoMat, this.toneMat, this.visRT,
      ...this.expoRT, ...(this._scratch || []), ...extra, this.quad.geometry, sky.lutTarget, sky.viewTarget, sky.fastTarget,
      sky.lutMat, sky.viewMat, sky.quad.geometry];
    for (const b of this.bakes) objs.push(b.full, b.fast, b.mat);
    for (const x of objs) for (const y of [x, x && x.texture]) if (y && y._listeners) delete y._listeners.dispose;
    if (!gl.getExtension('EXT_color_buffer_float')) throw new UnsupportedError('This GPU cannot render to float textures (EXT_color_buffer_float), which the cloud needs.');
    this.requireExtensions();
    this.timer = new GpuTimer(gl);   // queries of the lost context are gone
    for (const b of this.bakes) { b.fullKey = null; b.fastKey = null; }
    this.sky.key = null; this.sky.fastKey = null;
    this.sky.buildLut();
    this.expoFront = 0;
    this.compileAll();
    this.bakeStatic();
  }

  /** Compile every program up front with 1-pixel draws (no compile hitch on the first drag). */
  compileAll() {
    const t = this.scratch(1, 1);
    this.u.uRes.value.set(1, 1);
    this.u.uPrev.value = t[1].texture;
    this.draw(this.marchMat, t[0], [0, 0, 1, 1]);
    this.draw(this.sunvisMat, this.visRT);
    for (const b of this.bakes) {
      b.mat.uniforms.uBakeDims.value.set(1, 1, 1);
      this.draw(b.mat, b.full, [0, 0, 1, 1]);
    }
    this.expoMat.uniforms.uImage.value = t[0].texture;
    this.expoMat.uniforms.uPrevExpo.value = this.expoRT[1].texture;
    this.draw(this.expoMat, this.expoRT[0]);
    this.toneMat.uniforms.uImage.value = t[0].texture;
    this.toneMat.uniforms.uHold.value = t[0].texture;
    this.toneMat.uniforms.uExpo.value = this.expoRT[0].texture;
    this.sync(this.expoRT[0]);
    this.checkErrors();
  }

  scratch(w, h) {
    if (this._scratch && this._scratch[0].width === w && this._scratch[0].height === h) return this._scratch;
    if (this._scratch) this._scratch.forEach((t) => t.dispose());
    this._scratch = [this.target(w, h), this.target(w, h)];
    return this._scratch;
  }

  target(w, h) {
    return new THREE.WebGLRenderTarget(w, h, { type: THREE.FloatType, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter, depthBuffer: false, stencilBuffer: false, generateMipmaps: false });
  }

  // ---- bakes ------------------------------------------------------------------------------------------------------
  bakeInto(b, rt, dims) {
    b.mat.uniforms.uBakeDims.value.set(dims[0], dims[1], dims[2]);
    rt.viewport.set(0, 0, dims[0], dims[1]);
    rt.scissorTest = false;
    this.quad.material = b.mat;
    for (let z = 0; z < dims[2]; z++) {
      b.mat.uniforms.uBakeLayer.value = z;
      this.renderer.setRenderTarget(rt, z);
      this.renderer.render(this.quadScene, this.quadCam);
    }
    this.renderer.setRenderTarget(null);
  }

  bakeStatic() {
    for (const b of this.bakes.filter((x) => x.class !== 'sun')) {
      const t = performance.now();
      this.bakeInto(b, b.full, b.dims);
      this.sync(b.full);
      this.timing.bakes.static.push({ name: b.name, ms: performance.now() - t });
      b.fullKey = 'static';
    }
    this.checkErrors();
  }

  sunKey() { return [this.u.uSun.value, this.u.uLu.value, this.u.uLw.value].map((v) => v.toArray().join(',')).join(';'); }

  /**
   * Bake the sun textures for the current sun: the full grid (fast = false) or the drag grid. Binds the result.
   * timed: wait for the GPU and record the time (harness); otherwise the CPU submit time is recorded.
   */
  bakeSun(fast, timed = false) {
    const key = this.sunKey();
    let ms = 0;
    for (const b of this.bakes.filter((x) => x.class === 'sun')) {
      const useFast = fast && b.fast;
      const rt = useFast ? b.fast : b.full;
      if ((useFast ? b.fastKey : b.fullKey) !== key) {
        const t = performance.now();
        if (!timed) this.timer.begin(useFast ? 'sunFast' : 'sunFull');
        this.bakeInto(b, rt, useFast ? b.fastDims : b.dims);
        if (!timed) this.timer.end();
        if (timed) this.sync(rt);
        const dt = performance.now() - t;
        ms += dt;
        if (timed) this.timing.bakes[useFast ? 'sunFast' : 'sunFull'].push(dt);
        if (useFast) b.fastKey = key; else b.fullKey = key;
      }
      this.u[b.name].value = rt.texture;
    }
    return ms;
  }

  sunBakeIsFull() {
    const key = this.sunKey();
    return this.bakes.every((b) => b.class !== 'sun' || (b.fullKey === key && this.u[b.name].value === b.full.texture));
  }

  /** The full sun bake is current for the sun (bound or not): a frame that would bake the drag grid can use it. */
  fullSunBakeCurrent() {
    const key = this.sunKey();
    return this.bakes.every((b) => b.class !== 'sun' || b.fullKey === key);
  }

  // ---- camera and light -------------------------------------------------------------------------------------------
  /** cam: {position, look_at, up, fov_y_deg}; aspect w / h. fit: widen fov_y so the horizontal field never narrows
   * below fov_y * 1.6 (portrait embeds keep the whole cloud); the harness uses fit = false (shots.json exactly). */
  setCamera(cam, aspect, fit = true) {
    const pos = v3(cam.position);
    const fwd = v3(cam.look_at).sub(pos).normalize();
    const right = new THREE.Vector3().crossVectors(fwd, v3(cam.up || [0, 1, 0])).normalize();
    const up = new THREE.Vector3().crossVectors(right, fwd);
    const u = this.u;
    u.uCamPos.value.copy(pos);
    u.uCamFwd.value.copy(fwd);
    u.uCamRight.value.copy(right);
    u.uCamUp.value.copy(up);
    let ty = Math.tan(0.5 * cam.fov_y_deg * RAD);
    if (fit && aspect < 1.6) ty *= 1.6 / aspect;
    u.uTan.value.set(ty * aspect, ty);
    this.cam = cam;
  }

  /**
   * Sun direction and every light that depends on it: light-table lighting (E_sun by altitude, L_sky, L_gnd), the sky
   * map from the camera's altitude, the glare irradiance. override (harness): {E_sun, L_sky, L_gnd, sky_h} replaces
   * the table (constant E_sun, the reference presets' lighting) and sets the sky map's observer altitude.
   * fastSky: draw the coarse sky map (while the sun moves; updateSky(false) draws the full one once it stops).
   */
  setSunLight(el, az, override = null, fastSky = false) {
    const s = sunVector(el, az);
    const u = this.u;
    const sun = v3(s).normalize();
    const lu = new THREE.Vector3().crossVectors(sun, new THREE.Vector3(0, 1, 0));
    if (lu.length() < 1e-6) lu.set(1, 0, 0); else lu.normalize();
    u.uSun.value.copy(sun);
    u.uLu.value.copy(lu);
    u.uLw.value.crossVectors(lu, sun);
    const light = lightAt(this.table, el);
    const camY = u.uCamPos.value.y;
    if (override) {
      u.uEsunLutOn.value = 0;
      u.uEsun.value.set(...override.E_sun);
      u.uLsky.value.set(...override.L_sky);
      u.uLgnd.value.set(...override.L_gnd);
      this.skyArgs = [s, override.sky_h ?? 2000, override.L_gnd];
    } else {
      setLightUniforms(u, light);
      this.skyArgs = [s, camY, light.L_gnd];
    }
    this.updateSky(fastSky);
    // glare: disk-averaged E at the camera's altitude (0 below the table's first altitude: clamp)
    const k = clamp((camY - light.y0) / light.dy, 0, light.E_sun_lut.length - 1);
    const k0 = Math.min(Math.floor(k), light.E_sun_lut.length - 2), f = k - k0;
    const a = light.E_sun_lut[k0], b = light.E_sun_lut[k0 + 1];
    this.toneMat.uniforms.uSunE.value.set(...[0, 1, 2].map((c) => a[c] + (b[c] - a[c]) * f));
    this.light = light;
  }

  /** (Re)draws the sky map for the last setSunLight (only when stale) and binds it: the full map or the coarse one. */
  updateSky(fast) {
    this.sky.update(...this.skyArgs, fast);
    this.u.uSkyMap.value = this.sky.skyTexture;
  }

  skyIsFull() { return this.sky.current === this.sky.viewTarget; }

  /** Visible fraction of the sun disk (for the glare); a 13-ray density march into one texel, no readback. */
  updateSunVis() { this.draw(this.sunvisMat, this.visRT); }

  // ---- march / exposure / display ----------------------------------------------------------------------------------
  /** One march frame from src into dst (RGBA32F, same size): uMeanN = n + 1 (running mean) or 1 / blend. */
  marchFrame(src, dst, frame, meanN, rect = null, rejectHist = false) {
    const u = this.u;
    u.uRejectHist.value = rejectHist ? 1 : 0;
    u.uRes.value.set(dst.width, dst.height);
    u.uFrame.value = frame;
    u.uMeanN.value = meanN;
    u.uPrev.value = src ? src.texture : this.visRT.texture;   // never the target being written (a feedback loop drops the draw)
    this.draw(this.marchMat, dst, rect);
  }

  /** GPU auto-exposure from image (a render target); adapt in (0, 1], 1 snaps. */
  exposure(image, adapt) {
    const m = this.expoMat.uniforms;
    m.uImage.value = image.texture;
    m.uPrevExpo.value = this.expoRT[this.expoFront].texture;
    m.uAdapt.value = adapt;
    this.draw(this.expoMat, this.expoRT[1 - this.expoFront]);
    this.expoFront = 1 - this.expoFront;
  }

  display(image, hold, holdMix, w, h, frame) {
    const m = this.toneMat.uniforms;
    m.uImage.value = image.texture;
    m.uHold.value = (hold || image).texture;
    m.uHoldMix.value = hold ? holdMix : 0;
    m.uExpo.value = this.expoRT[this.expoFront].texture;
    m.uOutRes.value.set(w, h);
    m.uTan.value.copy(this.u.uTan.value);
    m.uFrame.value = frame;
    this.renderer.setRenderTarget(null);
    this.renderer.setViewport(0, 0, w, h);
    this.renderer.setScissorTest(false);
    this.quad.material = this.toneMat;
    this.renderer.render(this.quadScene, this.quadCam);
  }

  dispose() {
    const all = [this.densTex, this.atlasTex, this.marchMat, this.sunvisMat, this.expoMat, this.toneMat, this.visRT,
      ...(this.expoRT || []), ...(this._scratch || [])];
    for (const b of this.bakes || []) all.push(b.full, b.fast, b.mat);
    all.forEach((x) => x && x.dispose());
    if (this.sky) this.sky.dispose();
    if (this.quad) this.quad.geometry.dispose();
    if (this.renderer) this.renderer.dispose();
    // Release the context and its drawing buffer now, not when the host drops its last reference to the controller
    // (each live context holds ~20 MB of GPU memory; past ~16 Chrome force-loses the oldest one).
    if (this.gl && !this.gl.isContextLost()) {
      const lose = this.gl.getExtension('WEBGL_lose_context');
      this.canvas.width = 1; this.canvas.height = 1;
      if (lose) lose.loseContext();
    }
    this.disposed = true;
    this.gl = null; this.renderer = null; this.timer = null;
  }
}

// =====================================================================================================================
// The interactive view: canvas, overlay UI, frame loop
// =====================================================================================================================
const CSS = `
.root { position: absolute; inset: 0; overflow: hidden; background: linear-gradient(#5b7fa8, #a9c1d6 60%, #c9c3b4);
  font: 500 12px/1.35 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: rgba(255,255,255,.94);
  -webkit-tap-highlight-color: transparent; user-select: none; -webkit-user-select: none; outline: none;
  container-type: inline-size; }
.root:focus-visible { box-shadow: inset 0 0 0 2px rgba(255,255,255,.55); }
canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; cursor: grab; touch-action: pan-y pinch-zoom; }
canvas.drag { cursor: grabbing; }
.glass { background: rgba(14,18,26,.36); border: 1px solid rgba(255,255,255,.16); backdrop-filter: blur(10px) saturate(1.2);
  -webkit-backdrop-filter: blur(10px) saturate(1.2); box-shadow: 0 1px 10px rgba(0,0,0,.12); }
.caption { position: absolute; left: 12px; bottom: 12px; padding: 5px 10px; border-radius: 999px; pointer-events: none;
  font-variant-numeric: tabular-nums; letter-spacing: .01em; text-shadow: 0 1px 2px rgba(0,0,0,.35); white-space: nowrap; }
.caption b { font-weight: 600; }
.caption .long { display: inline; }
.hint { position: absolute; left: 50%; top: 14px; transform: translateX(-50%); padding: 5px 11px; border-radius: 999px;
  pointer-events: none; transition: opacity .8s; white-space: nowrap; }
.hint.off { opacity: 0; }
.bar { position: absolute; right: 12px; bottom: 12px; display: flex; align-items: flex-end; gap: 8px; }
.pill { display: flex; align-items: center; gap: 2px; padding: 3px; border-radius: 999px; }
button { all: unset; box-sizing: border-box; cursor: pointer; color: inherit; font: inherit; border-radius: 999px;
  padding: 4px 9px; line-height: 16px; transition: background .15s; }
button:hover { background: rgba(255,255,255,.14); }
button:focus-visible { box-shadow: 0 0 0 2px rgba(255,255,255,.7); }
button[aria-pressed="true"] { background: rgba(255,255,255,.88); color: #1b2230; }
button.icon { width: 26px; height: 26px; padding: 5px; display: grid; place-items: center; }
button.icon svg { width: 14px; height: 14px; fill: currentColor; }
.compass { width: 76px; height: 76px; border-radius: 50%; touch-action: none; cursor: crosshair; flex: none; }
.compass svg { width: 100%; height: 100%; display: block; overflow: visible; }
.veil { position: absolute; inset: 0; display: grid; place-items: center; transition: opacity .6s; }
.veil.off { opacity: 0; pointer-events: none; }
.load { display: grid; gap: 8px; justify-items: center; padding: 12px 16px; border-radius: 12px; min-width: 168px; }
.track { width: 140px; height: 3px; border-radius: 3px; background: rgba(255,255,255,.25); overflow: hidden; }
.fill { height: 100%; width: 0; background: rgba(255,255,255,.92); transition: width .25s; }
.err { max-width: min(80%, 420px); padding: 12px 16px; border-radius: 12px; text-align: center; font-size: 13px; }
/* by the embed's own width (container queries: a narrow blog column on a wide screen). Narrow: the caption sits above
   the view chips (the bar is wider than the space beside it); narrower still the chips go and the caption drops its
   words: "Sun 47.8° · 266° W" */
@container (max-width: 700px) { .caption { bottom: 54px; } }
@container (max-width: 520px) { .views button { padding: 4px 7px; } .compass { width: 64px; height: 64px; } .caption { left: 10px; } }
@container (max-width: 400px) { .views { display: none; } .caption .long { display: none; }
  .caption { bottom: 10px; max-width: calc(100% - 136px); overflow: hidden; text-overflow: ellipsis; box-sizing: border-box; } }
`;

const ICON_PLAY = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9.5-5.5z"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z"/></svg>';

/**
 * Mount the cloud in container (any element; it gets a positioned child that fills it).
 * opts: { base, view, elevation, azimuth, autoplay, frames, glare, harness, signal, onReady, onError } (see
 * CloudsEngine). Resolves after loading to { setSun(el, az), setView(name), play(), pause(), destroy(), engine, ready };
 * a load failure is shown in the container and also rejects `ready` (the controller itself always resolves).
 * signal: an AbortSignal; aborting it destroys the mount, also while it is still loading (the promise then resolves
 * to the destroyed controller, `ready` rejects with an AbortError, and nothing more is allocated on the GPU).
 */
export async function mountClouds(container, opts = {}) {
  const base = opts.base ? new URL(opts.base, document.baseURI).href : new URL('./', import.meta.url).href;
  const o = { view: 'front', autoplay: false, ...opts, base: base.endsWith('/') ? base : base + '/' };
  if (!VIEWS[o.view]) o.view = 'front';
  const reduced = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---- DOM (shadow root: the blog's CSS cannot reach in, ours cannot leak out) ----
  const host = document.createElement('div');
  host.style.cssText = 'display:block;position:relative;width:100%;height:100%;min-height:200px;';
  container.appendChild(host);
  const shadow = host.attachShadow({ mode: 'open' });
  shadow.innerHTML = `<style>${CSS}</style>
<div class="root" tabindex="0" role="application" aria-roledescription="interactive cloud" aria-label="A cloud lit by a hand-written neural network. Drag or use the arrow keys to move the sun.">
  <canvas aria-hidden="true"></canvas>
  <div class="hint glass off">Drag the sky to move the sun</div>
  <div class="caption glass" aria-live="polite"></div>
  <div class="bar">
    <div class="pill glass views" role="group" aria-label="View"></div>
    <div class="pill glass"><button class="icon play" aria-label="Play the sun's arc" title="Play (space)">${ICON_PLAY}</button></div>
    <div class="compass glass" role="slider" tabindex="0" aria-label="Sun position: drag to set azimuth and elevation"></div>
  </div>
  <div class="veil load-veil"><div class="load glass"><div class="label">Loading the cloud</div><div class="track"><div class="fill"></div></div></div></div>
</div>`;
  const $ = (s) => shadow.querySelector(s);
  const root = $('.root'), canvas = $('canvas'), caption = $('.caption'), hint = $('.hint'), playBtn = $('.play');
  const compassEl = $('.compass'), viewsEl = $('.views');
  $('.bar').style.visibility = 'hidden';
  caption.style.visibility = 'hidden';

  const engine = new CloudsEngine({ ...o, canvas });
  const st = {
    view: o.view, el: clamp(o.elevation ?? VIEWS[o.view].sun[0], EL_MIN, EL_MAX), az: wrap360(o.azimuth ?? VIEWS[o.view].sun[1]),
    sunVer: 1, viewVer: 1, drawnSun: 0, drawnView: 0, lastChange: 0, dragging: false, playing: false, playX: -1, playDir: 1,
    visible: true, pageVisible: !document.hidden, raf: 0, destroyed: false, ready: false, lastT: 0, seq: 0,
    cw: 0, ch: 0, moveScale: 0.6, costMsPerMpx: 14, scaleCheck: 0, slowFrames: 0,
    mov: null, movCur: 0, movValid: false, acc: null, accCur: 0, stillN: 0, stillFrame: null, adaptUntil: 0,
    needDisplay: true, snapExposure: true, hinted: false, harnessBusy: false, lost: false,
  };
  let resolveReady, rejectReady;
  const ready = new Promise((res, rej) => { resolveReady = res; rejectReady = rej; });
  ready.catch(() => {});

  // ---- compass widget: zenith at the centre, horizon at 90/98 of the radius, -8 degrees at the rim; north up ----
  const R = 34;
  const rOf = (el) => R * (90 - el) / (90 - EL_MIN);
  compassEl.innerHTML = `<svg viewBox="-38 -38 76 76" aria-hidden="true">
    <circle r="${rOf(0)}" fill="rgba(255,255,255,.06)" stroke="rgba(255,255,255,.55)" stroke-width="1"/>
    <circle r="${rOf(30)}" fill="none" stroke="rgba(255,255,255,.18)" stroke-width=".8"/>
    <circle r="${rOf(60)}" fill="none" stroke="rgba(255,255,255,.18)" stroke-width=".8"/>
    <path class="fov" fill="rgba(255,255,255,.16)"/>
    ${['N', 'E', 'S', 'W'].map((l, i) => { const a = i * 90 * RAD, r = R - 2.5; return `<text x="${(r * Math.sin(a)).toFixed(1)}" y="${(-r * Math.cos(a) + 2.6).toFixed(1)}" text-anchor="middle" font-size="7" font-weight="600" fill="rgba(255,255,255,${l === 'N' ? .95 : .6})">${l}</text>`; }).join('')}
    <circle class="sun" r="4.2" fill="#ffd877" stroke="rgba(80,50,0,.45)" stroke-width=".8"/>
  </svg>`;
  const sunDot = compassEl.querySelector('.sun'), fovPath = compassEl.querySelector('.fov');

  for (const [name, v] of Object.entries(VIEWS)) {
    const b = document.createElement('button');
    b.textContent = v.label;
    b.dataset.view = name;
    b.setAttribute('aria-pressed', String(name === st.view));
    b.onclick = () => { stopPlay(); setView(name); };
    viewsEl.appendChild(b);
  }

  function updateUI() {
    const r = rOf(st.el), a = st.az * RAD;
    sunDot.setAttribute('cx', (r * Math.sin(a)).toFixed(2));
    sunDot.setAttribute('cy', (-r * Math.cos(a)).toFixed(2));
    const u = engine.u;
    if (u) {
      const f = u.uCamFwd.value, yaw = Math.atan2(f.x, -f.z), half = Math.atan(u.uTan.value.x);
      const p = (ang) => `${(rOf(0) * Math.sin(ang)).toFixed(2)} ${(-rOf(0) * Math.cos(ang)).toFixed(2)}`;
      fovPath.setAttribute('d', `M0 0 L${p(yaw - half)} A${rOf(0)} ${rOf(0)} 0 0 1 ${p(yaw + half)} Z`);
    }
    const dir = COMPASS[Math.round(wrap360(st.az) / 22.5) % 16];
    const elTxt = `${st.el < 0 ? '−' : ''}${Math.abs(st.el).toFixed(1)}°`;
    // narrow embeds hide the .long words (CSS): "Sun 47.8° · 266° W"
    caption.innerHTML = `Sun <b>${elTxt}</b><span class="long"> ${st.el < 0 ? 'below the horizon' : 'up'}</span> · `
      + `<span class="long">azimuth </span><b>${Math.round(wrap360(st.az))}°</b> ${dir}`;
    compassEl.setAttribute('aria-valuetext', `elevation ${st.el.toFixed(1)} degrees, azimuth ${Math.round(st.az)} degrees`);
    for (const b of viewsEl.children) b.setAttribute('aria-pressed', String(b.dataset.view === st.view));
    playBtn.innerHTML = st.playing ? ICON_PAUSE : ICON_PLAY;
    playBtn.setAttribute('aria-label', st.playing ? 'Pause' : "Play the sun's arc");
  }

  // ---- state changes ----
  function setSun(el, az) {
    const e = clamp(Number(el), EL_MIN, EL_MAX), a = wrap360(Number(az));
    if (!Number.isFinite(e) || !Number.isFinite(a)) return;
    if (e === st.el && a === st.az) return;
    st.el = e; st.az = a; st.sunVer++;
    updateUI();
    wake();
  }
  function setView(name) {
    if (!VIEWS[name] || name === st.view) return;
    st.view = name; st.viewVer++;
    const [e, a] = VIEWS[name].sun;
    st.el = e; st.az = a; st.sunVer++;
    st.playX = -1; st.playDir = 1;
    st.snapExposure = true;
    updateUI();
    wake();
  }
  function startPlay() {
    if (st.playing) return;
    st.playing = true;
    // start the sweep where the path is closest to the current sun
    const path = VIEWS[st.view].path;
    let best = 1e9;
    for (let x = -1; x <= 1.0001; x += 0.02) {
      const [e, a] = path(x);
      const d = (e - st.el) ** 2 + (((a - st.az + 540) % 360) - 180) ** 2;
      if (d < best) { best = d; st.playX = x; }
    }
    updateUI();
    wake();
  }
  function stopPlay() { if (st.playing) { st.playing = false; updateUI(); } }
  function interacted() {
    if (!st.hinted) { st.hinted = true; hint.classList.add('off'); }
  }

  // ---- pointer: the sun goes to the view ray under the pointer ----
  function sunFromPointer(e) {
    const r = canvas.getBoundingClientRect();
    const u = engine.u;
    const sx = (2 * (e.clientX - r.left) / r.width - 1) * u.uTan.value.x;
    const sy = (1 - 2 * (e.clientY - r.top) / r.height) * u.uTan.value.y;
    const d = u.uCamFwd.value.clone().addScaledVector(u.uCamRight.value, sx).addScaledVector(u.uCamUp.value, sy).normalize();
    setSun(Math.asin(clamp(d.y, -1, 1)) / RAD, Math.atan2(d.x, -d.z) / RAD);
  }
  const onDown = (e) => {
    if (!st.ready || (e.pointerType === 'mouse' && e.button !== 0)) return;
    st.dragging = true;
    canvas.classList.add('drag');
    try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* synthetic events */ }
    stopPlay(); interacted();
    sunFromPointer(e);
  };
  const onMove = (e) => { if (st.dragging) sunFromPointer(e); };
  const onUp = () => { if (st.dragging) { st.dragging = false; canvas.classList.remove('drag'); wake(); } };
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointercancel', onUp);

  // compass: any azimuth and elevation (also behind the camera)
  let compassDrag = false;
  const compassSet = (e) => {
    const r = compassEl.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width * 76 - 38, y = (e.clientY - r.top) / r.height * 76 - 38;
    const rr = Math.hypot(x, y);
    setSun(90 - (90 - EL_MIN) * Math.min(rr, R) / R, Math.atan2(x, -y) / RAD);
  };
  compassEl.addEventListener('pointerdown', (e) => {
    if (!st.ready) return;
    compassDrag = true; st.dragging = true;
    try { compassEl.setPointerCapture(e.pointerId); } catch (err) { /* synthetic events */ }
    stopPlay(); interacted(); compassSet(e); e.preventDefault();
  });
  compassEl.addEventListener('pointermove', (e) => { if (compassDrag) compassSet(e); });
  const compassUp = () => { if (compassDrag) { compassDrag = false; st.dragging = false; wake(); } };
  compassEl.addEventListener('pointerup', compassUp);
  compassEl.addEventListener('pointercancel', compassUp);

  playBtn.addEventListener('click', () => { interacted(); if (st.playing) stopPlay(); else startPlay(); });

  // keyboard: arrows move the sun (shift: 5x), space toggles play
  root.addEventListener('keydown', (e) => {
    if (!st.ready) return;
    const k = e.shiftKey ? 5 : 1;
    let de = 0, da = 0;
    if (e.key === 'ArrowLeft') da = -2 * k; else if (e.key === 'ArrowRight') da = 2 * k;
    else if (e.key === 'ArrowUp') de = k; else if (e.key === 'ArrowDown') de = -k;
    else if (e.key === ' ' && e.target === root) { e.preventDefault(); if (st.playing) stopPlay(); else startPlay(); return; } else return;
    e.preventDefault();
    stopPlay(); interacted();
    st.keyUntil = performance.now() + 250;
    setSun(st.el + de, st.az + da);
  });

  // ---- visibility: pause off screen and in hidden tabs ----
  const io = typeof IntersectionObserver === 'function' ? new IntersectionObserver((es) => {
    st.visible = es[es.length - 1].isIntersecting;
    if (st.visible) wake();
  }) : null;
  if (io) io.observe(host);
  const onVis = () => { st.pageVisible = !document.hidden; if (st.pageVisible) wake(); };
  document.addEventListener('visibilitychange', onVis);
  const ro = typeof ResizeObserver === 'function' ? new ResizeObserver(() => { st.needResize = true; wake(); }) : null;
  if (ro) ro.observe(host);
  // devicePixelRatio changes without a CSS size change (the window moves to another monitor; zoom): a media query
  // for the current ratio fires once when it stops matching, then re-arm it for the new ratio
  let dprQuery = null;
  const onDpr = () => { watchDpr(); st.needResize = true; wake(); };
  function watchDpr() {
    if (dprQuery) dprQuery.removeEventListener('change', onDpr);
    dprQuery = typeof matchMedia === 'function' ? matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`) : null;
    if (dprQuery && dprQuery.addEventListener) dprQuery.addEventListener('change', onDpr); else dprQuery = null;
  }
  watchDpr();

  // ---- WebGL context loss (mobile tab in the background, GPU reset): stop drawing; on restore rebuild every render
  // target (three.js re-creates them empty) and redraw. A context that is not back within 3 s gets a message. ----
  let lostTimer = 0;
  const onLost = (e) => {
    e.preventDefault();   // (three.js does too) allows the restore
    if (st.destroyed) return;
    st.lost = true;
    if (st.raf) { cancelAnimationFrame(st.raf); st.raf = 0; }
    clearTimeout(lostTimer);
    lostTimer = setTimeout(() => { if (st.lost && !st.destroyed) showMessage('The graphics context was lost. Waiting for it to come back…'); }, 3000);
  };
  const onRestored = () => {
    if (st.destroyed || !st.lost) return;
    clearTimeout(lostTimer);
    if (!st.ready) { showError(new Error('the WebGL context was lost while loading; reload the page')); return; }
    try {
      engine.restoreGL([...(st.mov || []), ...(st.acc || [])]);
    } catch (err) { showError(err); return; }
    st.lost = false;
    st.movValid = false; st.stillN = 0; st.stillFrame = null;
    st.drawnSun = 0; st.drawnView = 0;   // re-applies the camera, the sun bakes, the sky map and the sun visibility
    st.snapExposure = true; st.needDisplay = true;
    $('.load-veil').classList.add('off');
    wake();
  };

  // ---- frame loop ----
  function wake() {
    if (st.destroyed || !st.ready || st.raf || st.harnessBusy || st.lost) return;
    if (!st.visible || !st.pageVisible) return;
    st.raf = requestAnimationFrame(frame);
  }

  function resize() {
    st.dpr = window.devicePixelRatio || 1;
    const dpr = Math.min(st.dpr, o.maxDpr || 2);
    const w = Math.max(1, Math.round(host.clientWidth * dpr)), h = Math.max(1, Math.round(host.clientHeight * dpr));
    if (w === st.cw && h === st.ch) return false;
    st.cw = w; st.ch = h;
    canvas.width = w; canvas.height = h;
    engine.setCamera(VIEWS[st.view].camera, w / h);
    st.mov = freeTargets(st.mov); st.acc = freeTargets(st.acc);
    st.movValid = false; st.stillN = 0; st.stillFrame = null;
    updateUI();
    return true;
  }
  const freeTargets = (t) => { if (t) t.forEach((x) => x.dispose()); return null; };
  const stillSize = () => {
    const s = Math.min(1, Math.sqrt((o.maxPixels || engine.opts.maxPixels) / (st.cw * st.ch)));
    return [Math.max(1, Math.round(st.cw * s)), Math.max(1, Math.round(st.ch * s))];
  };
  const moveSize = () => {
    const [w, h] = stillSize();
    return [Math.max(16, Math.round(w * st.moveScale)), Math.max(10, Math.round(h * st.moveScale))];
  };
  function pairAt([w, h], cur) {
    if (cur && cur[0].width === w && cur[0].height === h) return cur;
    freeTargets(cur);
    return [engine.target(w, h), engine.target(w, h)];
  }

  // adaptive internal resolution while the sun moves: hold the march near opts.targetMs per frame
  function adaptScale(now) {
    if (now < st.scaleCheck) return;
    st.scaleCheck = now + 400;
    const [w, h] = stillSize();
    const mpx = w * h / 1e6;
    const s = clamp(Math.sqrt(engine.opts.targetMs / (st.costMsPerMpx * mpx)), 0.3, 1);
    const q = Math.round(s * 20) / 20;
    if (Math.abs(q - st.moveScale) >= 0.05) { st.moveScale = q; }
  }
  function onGpuTime(label, ms, meta) {
    if (label === 'march' && meta.mpx > 0.01) {
      const c = ms / meta.mpx;
      st.costMsPerMpx += (c - st.costMsPerMpx) * 0.25;
    }
    pushTiming({ kind: label, gpuMs: ms, ...meta });
  }
  // diagnostics: the last ~2400 records (~20 s of frames). gpuMs is the page's own timer query and is reliable for
  // 'march' (the pass the adaptive resolution uses); on ANGLE Metal a query around a short pass after the march can
  // also count the march's time
  function pushTiming(rec) {
    const f = engine.timing.frames;
    f.push(rec);
    if (f.length > 3000) f.splice(0, f.length - 2400);
  }

  function frame(now) {
    st.raf = 0;
    if (st.destroyed || !st.visible || !st.pageVisible || st.harnessBusy || st.lost) return;
    const dt = st.lastT ? Math.min(0.1, (now - st.lastT) / 1000) : 1 / 60;
    const frameDelta = st.lastT ? now - st.lastT : 0;
    st.lastT = now;
    const t0 = performance.now();
    engine.timer.poll(onGpuTime);
    let busy = false;
    if (st.needResize || !st.cw || (window.devicePixelRatio || 1) !== st.dpr) {
      st.needResize = false;
      // a resize clears the canvas and the images: draw the next frames like a move (one full-size sample per pixel
      // per frame), so a window being resized never shows the cleared canvas while a strip-split still frame builds
      if (resize()) { st.needDisplay = true; st.lastChange = now; st.scaleCheck = 0; }
    }

    if (st.playing) {
      const speed = 2 / PLAY_SECONDS;
      st.playX += st.playDir * speed * dt;
      if (st.playX > 1) { st.playX = 2 - st.playX; st.playDir = -1; }
      if (st.playX < -1) { st.playX = -2 - st.playX; st.playDir = 1; }
      const [e, a] = VIEWS[st.view].path(st.playX);
      setSun(e, a);
    }

    const viewChanged = st.drawnView !== st.viewVer, sunChanged = st.drawnSun !== st.sunVer;
    if (viewChanged) {
      engine.setCamera(VIEWS[st.view].camera, st.cw / st.ch);
      st.movValid = false;
      updateUI();
    }
    if (sunChanged || viewChanged) {
      engine.setSunLight(st.el, st.az, null, true);   // coarse sky map; the still path draws the full one
      engine.updateSunVis();
      st.drawnSun = st.sunVer; st.drawnView = st.viewVer;
      st.lastChange = now;
      st.stillN = 0; st.stillFrame = null;
    }
    // moving = the sun changed in the last 140 ms (a pointer held still accumulates like a released one)
    const moving = st.playing || now - st.lastChange < 140 || (st.keyUntil && now < st.keyUntil);
    let image = null, hold = null, holdMix = 0, marched = false;

    if (moving) {
      // ---- sun moving: coarse sun bake, one sample per pixel at the adaptive scale, temporal blend ----
      adaptScale(now);
      if (!engine.timer.ext && frameDelta > 0) {
        // no GPU timer: infer the cost from frame pacing (rAF stretches when the GPU is behind)
        if (frameDelta > 24) st.costMsPerMpx *= 1.08; else if (frameDelta < 18) st.costMsPerMpx *= 0.985;
      }
      const sz = moveSize();
      const prevMov = st.mov;
      st.mov = pairAt(sz, st.mov);
      if (st.mov !== prevMov) st.movValid = false;
      engine.bakeSun(!engine.fullSunBakeCurrent());   // the drag grid, unless the full bake is already current
      const src = st.mov[st.movCur], dst = st.mov[1 - st.movCur];
      engine.timer.begin('march', { w: sz[0], h: sz[1], mpx: sz[0] * sz[1] / 1e6, mode: 'moving' });
      engine.marchFrame(src, dst, 1000 + (st.seq++ % 100000), st.movValid ? 1 / engine.opts.blend : 1, null, true);
      engine.timer.end();
      st.movCur = 1 - st.movCur;
      st.movValid = true;
      image = st.mov[st.movCur];
      marched = true; busy = true;
    } else {
      // ---- sun still: full sun bake, running mean of up to opts.frames frames, rendered in strips ----
      const N = engine.opts.frames;
      if (st.stillN < N) {
        if (!engine.sunBakeIsFull()) engine.bakeSun(false);
        if (!engine.skyIsFull()) engine.updateSky(false);
        const sz = stillSize();
        const prevAcc = st.acc;
        st.acc = pairAt(sz, st.acc);
        if (st.acc !== prevAcc) { st.stillN = 0; st.stillFrame = null; }
        if (!st.stillFrame) {
          const mpx = sz[0] * sz[1] / 1e6;
          const strips = clamp(Math.ceil(st.costMsPerMpx * mpx / (engine.opts.targetMs + 2)), 1, 16);
          st.stillFrame = { strip: 0, strips, rows: Math.ceil(sz[1] / strips) };
        }
        const F = st.stillFrame, src = st.acc[st.accCur], dst = st.acc[1 - st.accCur];
        let spent = 0;
        const perStrip = st.costMsPerMpx * sz[0] * F.rows / 1e6;
        do {
          const y = F.strip * F.rows, hh = Math.min(F.rows, sz[1] - y);
          engine.timer.begin('march', { w: sz[0], h: hh, mpx: sz[0] * hh / 1e6, mode: 'still', n: st.stillN });
          engine.marchFrame(src, dst, st.stillN, st.stillN + 1, [0, y, sz[0], hh]);
          engine.timer.end();
          spent += perStrip;
          F.strip++;
        } while (F.strip < F.strips && spent + perStrip <= engine.opts.targetMs + 2);
        if (F.strip >= F.strips) {
          st.accCur = 1 - st.accCur;
          st.stillN++;
          st.stillFrame = null;
          marched = true;
        }
        busy = true;
      }
      if (st.stillN > 0) {
        image = st.acc[st.accCur];
        // cross-fade from the moving image over the first frames (a 1-sample frame would flash noise)
        if (st.movValid && st.mov) { hold = st.mov[st.movCur]; holdMix = Math.max(0, 1 - st.stillN / 6); }
      } else if (st.movValid && st.mov) image = st.mov[st.movCur];
      if (st.stillN >= N) { st.movValid = false; }
    }

    if (image) {
      if (marched || st.snapExposure || now < st.adaptUntil) {
        engine.exposure(image, st.snapExposure ? 1 : 1 - Math.exp(-dt / 0.4));
        if (marched) st.adaptUntil = now + 1200;   // 3 time constants after the last march: 95 % settled
        st.snapExposure = false;
        st.needDisplay = true;
      }
      if (st.needDisplay || marched) {
        engine.display(image, hold, holdMix, st.cw, st.ch, st.seq);
        st.needDisplay = false;
        if (!st.shownFirst) { st.shownFirst = true; engine.timing.load.firstFrameMs = performance.now() - st.mountT0; }
      }
      busy = busy || now < st.adaptUntil;
    }
    pushTiming({ kind: 'cpu', cpuMs: performance.now() - t0, dtMs: frameDelta, moving: !!moving,
      scale: st.moveScale, stillN: st.stillN, cost: st.costMsPerMpx });
    if (busy || moving || st.playing) wake();
  }

  // ---- harness (?harness=1): deterministic renders and timings for the tests ----
  function installHarness() {
    window.cloudsHarness = {
      engine, views: VIEWS,
      /**
       * render({ shot | camera, elevation, azimuth, res: [w, h], frames, light, sunDisk, fit, sunBake }) ->
       * { linear: Float32Array w*h*3 (row 0 top), width, height, timing }. shot: a VIEWS name; camera: shots.json
       * form. light: { E_sun, L_sky, L_gnd, sky_h } replaces the light table (reference presets). sunDisk (true).
       * Frames f = 0..frames-1 use the harness's jitter seeds and running mean, each one waited for (GPU ms).
       */
      async render({ shot, camera, elevation, azimuth, res, frames = 16, light = null, sunDisk = true, fit = false, sunBake = 'full' } = {}) {
        await ready;
        st.harnessBusy = true;
        if (st.raf) { cancelAnimationFrame(st.raf); st.raf = 0; }
        try {
          const cam = camera || (VIEWS[shot] && VIEWS[shot].camera);
          if (!cam) throw new Error(`unknown shot ${shot}`);
          const [w, h] = res || [cam.width || 640, cam.height || 400];
          engine.setCamera(cam, w / h, fit);
          const el = elevation ?? 20, az = azimuth ?? 250;
          engine.setSunLight(el, az, light);
          engine.u.uSunDiskOn.value = sunDisk ? 1 : 0;
          const tb = performance.now();
          const bakeMs = engine.bakeSun(sunBake === 'fast', true);
          engine.updateSunVis();
          engine.sync(engine.visRT);
          const setupMs = performance.now() - tb;
          const tg = engine.scratch(w, h);
          const per = [];
          for (let f = 0; f < frames; f++) {
            const t = performance.now();
            engine.marchFrame(f ? tg[(f + 1) % 2] : null, tg[f % 2], f, f + 1);
            engine.sync(tg[f % 2]);
            per.push(performance.now() - t);
          }
          engine.checkErrors();
          const out = tg[(frames - 1) % 2];
          const b = new Float32Array(w * h * 4);
          engine.renderer.readRenderTargetPixels(out, 0, 0, w, h, b);
          const lin = new Float32Array(w * h * 3);
          for (let y = 0; y < h; y++) {
            const sy = h - 1 - y;
            for (let x = 0; x < w; x++) {
              const i = 4 * (sy * w + x), j = 3 * (y * w + x);
              lin[j] = b[i]; lin[j + 1] = b[i + 1]; lin[j + 2] = b[i + 2];
            }
          }
          engine.u.uSunDiskOn.value = 1;
          const sorted = [...per].sort((p, q) => p - q);
          return { linear: lin, width: w, height: h,
            timing: { frameMs: per, medianFrameMs: sorted[sorted.length >> 1], sunBakeMs: bakeMs, setupMs, gl: engine.glInfo } };
        } finally {
          st.harnessBusy = false;
          st.drawnSun = 0; st.drawnView = 0;   // the interactive view re-applies its own camera and sun
          wake();
        }
      },
      /** Sun bakes timed with a GPU wait: { full: [ms], fast: [ms] } over reps different suns. */
      async bakeTiming(reps = 8) {
        await ready;
        const full = [], fast = [];
        for (let i = 0; i < reps; i++) {
          engine.setSunLight(10 + 3 * i, 200 + 7 * i);
          const a = performance.now(); engine.bakeSun(false, true); full.push(performance.now() - a);
          const b = performance.now(); engine.bakeSun(true, true); fast.push(performance.now() - b);
        }
        st.drawnSun = 0; wake();
        return { full, fast, fullDims: engine.bakes.filter((x) => x.class === 'sun').map((x) => x.dims),
          fastDims: engine.bakes.filter((x) => x.class === 'sun').map((x) => x.fastDims) };
      },
      timing() {
        return { load: engine.timing.load, bakes: engine.timing.bakes, frames: engine.timing.frames.slice(),
          gl: engine.glInfo, state: { cw: st.cw, ch: st.ch, moveScale: st.moveScale, costMsPerMpx: st.costMsPerMpx,
            stillN: st.stillN, moveSize: st.cw ? moveSize() : null, stillSize: st.cw ? stillSize() : null } };
      },
      state() { return { el: st.el, az: st.az, view: st.view, playing: st.playing, stillN: st.stillN, ready: st.ready,
        idle: !st.raf, dragging: st.dragging }; },
      controller,
    };
  }

  const controller = {
    setSun: (el, az) => { stopPlay(); setSun(el, az); },
    setView: (name) => { stopPlay(); setView(name); },
    play: startPlay, pause: stopPlay,
    get sun() { return { elevation: st.el, azimuth: st.az }; },
    get view() { return st.view; },
    engine, ready,
    destroy() {
      if (st.destroyed) return;
      st.destroyed = true;
      if (st.raf) cancelAnimationFrame(st.raf);
      if (io) io.disconnect();
      if (ro) ro.disconnect();
      document.removeEventListener('visibilitychange', onVis);
      if (dprQuery) dprQuery.removeEventListener('change', onDpr);
      if (o.signal) o.signal.removeEventListener('abort', onAbort);
      clearTimeout(lostTimer);
      canvas.removeEventListener('webglcontextlost', onLost);
      canvas.removeEventListener('webglcontextrestored', onRestored);
      freeTargets(st.mov); freeTargets(st.acc);
      engine.dispose();   // also loses the context and shrinks the canvas: GPU memory is freed now
      host.remove();
      if (o.harness && window.cloudsHarness && window.cloudsHarness.engine === engine) delete window.cloudsHarness;
    },
  };

  // ---- load ----
  st.mountT0 = performance.now();
  const fill = $('.fill'), label = $('.label');
  const showMessage = (text) => {
    const v = $('.load-veil');
    v.classList.remove('off');
    v.innerHTML = `<div class="err glass"></div>`;
    v.firstChild.textContent = text;
  };
  const showError = (err) => {
    showMessage(err instanceof UnsupportedError ? err.message : `The cloud could not be drawn: ${err.message || err}`);
    console.error(err);
  };
  const onAbort = () => controller.destroy();
  if (o.signal) {
    if (o.signal.aborted) { controller.destroy(); rejectReady(abortError()); return controller; }
    o.signal.addEventListener('abort', onAbort, { once: true });
  }
  try {
    engine.initGL();
    // after initGL: three.js's own 'webglcontextrestored' listener (registered there) must re-initialise its state
    // before onRestored redraws into the targets
    canvas.addEventListener('webglcontextlost', onLost);
    canvas.addEventListener('webglcontextrestored', onRestored);
    await engine.load((f, text) => {
      fill.style.width = `${Math.round(100 * f)}%`;
      label.textContent = text === 'Ready' ? 'Ready' : `${text} the cloud · ${Math.round(100 * f)}%`;
    });
    if (st.destroyed) throw abortError();
    resize();
    st.ready = true;
    $('.bar').style.visibility = '';
    caption.style.visibility = '';
    $('.load-veil').classList.add('off');
    updateUI();
    if (!reduced && !o.harness) { hint.classList.remove('off'); setTimeout(() => interacted(), 5000); }
    if (o.harness) installHarness();
    if (o.autoplay && !reduced) startPlay();
    resolveReady(controller);
    wake();
    if (o.onReady) o.onReady(controller);
  } catch (err) {
    if (st.destroyed) { rejectReady(err && err.name === 'AbortError' ? err : abortError()); return controller; }
    showError(err);
    rejectReady(err);
    if (o.harness) window.cloudsHarness = { error: String(err && err.message || err), engine };
    if (o.onError) o.onError(err);
  }
  return controller;
}
