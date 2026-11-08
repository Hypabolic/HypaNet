// HypaNet Clouds renderer core (SPEC section 4): three.js hosts full-screen RawShaderMaterial passes.
// Used by index.html (headless harness for render.py / inspect.py) and viewer.html (interactive).
import * as THREE from 'three';
import { validateNetwork, packNetwork, recordLayout, NetworkError, BLUR_LEVELS, DEBUG_HEADER } from './net.js';
import { buildPyramid } from './pyramid.js';

export { NetworkError };

const RAD = Math.PI / 180;

async function fetchOk(url, kind) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  return kind === 'json' ? r.json() : kind === 'text' ? r.text() : r.arrayBuffer();
}

function v3(a) { return new THREE.Vector3(a[0], a[1], a[2]); }

export class CloudRenderer {
  /**
   * opts.canvas: canvas element (viewer) or undefined (an offscreen canvas is created)
   * opts.base: URL of the renderer directory (shaders/), opts.sceneBase: URL of scene/out/
   * opts.harness: enables the harness-only exact single-scattering mode and inspect's exact extras
   */
  constructor(opts = {}) {
    this.opts = { base: './', sceneBase: '../scene/out/', harness: false, ...opts };
    const canvas = opts.canvas || document.createElement('canvas');
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, depth: false, stencil: false,
      preserveDrawingBuffer: true, powerPreference: 'high-performance' });
    this.renderer.autoClear = false;
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.shaderErrors = [];
    this.renderer.debug.onShaderError = (gl, program, vs, fs) => {
      const msg = `shader compile/link error:\n${gl.getProgramInfoLog(program)}\n${gl.getShaderInfoLog(vs)}\n${gl.getShaderInfoLog(fs)}`;
      this.shaderErrors.push(msg);
      console.error(msg);
    };
    this.gl = this.renderer.getContext();
    this.quadScene = new THREE.Scene();
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
    this.quad.frustumCulled = false;
    this.quadScene.add(this.quad);
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.skyCache = new Map();
    this.width = 0;
    this.height = 0;
    this.net = null;
    this.frameIndex = 0;
  }

  glInfo() {
    const gl = this.gl;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    return {
      vendor: ext ? gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
      renderer: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
      version: gl.getParameter(gl.VERSION),
      colorBufferFloat: !!gl.getExtension('EXT_color_buffer_float'),
      max3D: gl.getParameter(gl.MAX_3D_TEXTURE_SIZE),
    };
  }

  async load() {
    const t0 = performance.now();
    const { base, sceneBase } = this.opts;
    const [vert, frag, tone, dj, shots, raw] = await Promise.all([
      fetchOk(base + 'shaders/fullscreen.vert.glsl', 'text'),
      fetchOk(base + 'shaders/march.frag.glsl', 'text'),
      fetchOk(base + 'shaders/tonemap.frag.glsl', 'text'),
      fetchOk(sceneBase + 'density.json', 'json'),
      fetchOk(sceneBase + 'shots.json', 'json'),
      fetchOk(sceneBase + 'density.u8', 'buf'),
    ]);
    this.src = { vert, frag, tone };
    this.densityInfo = dj;
    this.shots = shots;
    const dims = dj.dims;
    const bytes = new Uint8Array(raw);
    if (bytes.length !== dims[0] * dims[1] * dims[2]) throw new Error(`density.u8 has ${bytes.length} bytes, expected ${dims[0] * dims[1] * dims[2]}`);
    this.voxel = dj.voxel_size;
    this.voxelMean = (this.voxel[0] + this.voxel[1] + this.voxel[2]) / 3;

    const dens = new THREE.Data3DTexture(bytes, dims[0], dims[1], dims[2]);
    Object.assign(dens, { format: THREE.RedFormat, type: THREE.UnsignedByteType, minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter, wrapS: THREE.ClampToEdgeWrapping, wrapT: THREE.ClampToEdgeWrapping,
      wrapR: THREE.ClampToEdgeWrapping, unpackAlignment: 1, generateMipmaps: false });
    dens.needsUpdate = true;
    this.densTex = dens;

    const tp = performance.now();
    const pyr = buildPyramid(bytes, dims, BLUR_LEVELS);
    this.pyramidMs = performance.now() - tp;
    this.pyramid = pyr;
    const atlas = new THREE.Data3DTexture(pyr.atlas, ...pyr.atlasDims);
    Object.assign(atlas, { format: THREE.RedFormat, type: THREE.UnsignedByteType, minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter, wrapS: THREE.ClampToEdgeWrapping, wrapT: THREE.ClampToEdgeWrapping,
      wrapR: THREE.ClampToEdgeWrapping, unpackAlignment: 1, generateMipmaps: false });
    atlas.needsUpdate = true;
    this.atlasTex = atlas;

    const lvlScale = [], lvlOffset = [];
    for (let k = 0; k < BLUR_LEVELS; k++) {
      lvlScale.push(new THREE.Vector4(...pyr.scale[k], 0));
      lvlOffset.push(new THREE.Vector4(...pyr.offset[k], 0));
    }
    const dummyNet = new THREE.DataTexture(new Float32Array(4), 1, 1, THREE.RGBAFormat, THREE.FloatType);
    dummyNet.needsUpdate = true;
    const zero4 = () => new THREE.Vector4();
    this.u = {
      uDensity: { value: dens }, uAtlas: { value: atlas }, uLvlScale: { value: lvlScale }, uLvlOffset: { value: lvlOffset },
      uBoxMin: { value: v3(dj.box_min) }, uBoxMax: { value: v3(dj.box_max) }, uVoxel: { value: v3(dj.voxel_size) },
      uDims: { value: v3(dims) }, uSigmaMax: { value: dj.sigma_t_max }, uAlbedo: { value: dj.albedo },
      uCamPos: { value: new THREE.Vector3() }, uCamFwd: { value: new THREE.Vector3() }, uCamRight: { value: new THREE.Vector3() },
      uCamUp: { value: new THREE.Vector3() }, uTan: { value: new THREE.Vector2() }, uRes: { value: new THREE.Vector2() },
      uMaxSteps: { value: 512 }, uMaxShadowSteps: { value: 1024 },
      uFrame: { value: 0 }, uFrame0: { value: 0 }, uStamp: { value: 1 }, uPixelJitter: { value: 1 }, uStep: { value: 6.0 }, uShadowStep: { value: 5.0 },
      uPrev: { value: null },
      uSun: { value: new THREE.Vector3(0, 1, 0) }, uLu: { value: new THREE.Vector3() }, uLw: { value: new THREE.Vector3() },
      uEsun: { value: new THREE.Vector3() }, uLsky: { value: new THREE.Vector3() }, uLgnd: { value: new THREE.Vector3() },
      uSkyA: { value: null }, uSkyB: { value: null }, uSkyMix: { value: 0 }, uSkyRot: { value: 0 }, uBgScale: { value: 1 },
      uNet: { value: dummyNet }, uNumProbes: { value: 0 }, uNumLayers: { value: 0 },
      uProbeA: { value: Array.from({ length: 64 }, zero4) }, uProbeB: { value: Array.from({ length: 64 }, zero4) },
      uLayer: { value: Array.from({ length: 6 }, zero4) },
      uDebugPixel: { value: new THREE.Vector2() },
      uState0: { value: null }, uState1: { value: null }, uSegFirst: { value: 1 }, uEvalCap: { value: 1 },
    };
    this.materials = {};
    this.toneMat = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: vert, fragmentShader: tone,
      uniforms: { uAccum: { value: null }, uExposure: { value: 1 } }, depthTest: false, depthWrite: false });
    this.loadMs = performance.now() - t0;
    return { loadMs: this.loadMs, pyramidMs: this.pyramidMs, gl: this.glInfo(), atlasDims: pyr.atlasDims };
  }

  material(key) {
    if (this.materials[key]) return this.materials[key];
    const defines = {};
    if (key === 'net') { defines.MODE_NET = 1; defines.SEGMENT = 1; }
    else if (key === 'exact') {
      if (!this.opts.harness) throw new Error('exact single-scattering mode is harness-only');
      defines.MODE_EXACT = 1; defines.SEGMENT = 1;
    } else if (key === 'final') { defines.FINALIZE = 1; }
    else if (key === 'debug') { defines.MODE_NET = 1; defines.DEBUG_RAY = 1; }
    else if (key === 'debug_exact') {
      if (!this.opts.harness) throw new Error('exact extras are harness-only');
      defines.MODE_NET = 1; defines.DEBUG_RAY = 1; defines.DEBUG_EXACT = 1;
    } else throw new Error(`unknown material ${key}`);
    const m = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: this.src.vert,
      fragmentShader: this.src.frag, uniforms: this.u, defines, depthTest: false, depthWrite: false });
    this.materials[key] = m;
    return m;
  }

  // ---------------------------------------------------------------------------------------------
  // network
  setNetwork(net) {
    if (typeof net === 'string') {
      try { net = JSON.parse(net); } catch (e) { throw new NetworkError(`network.json is not valid JSON: ${e.message}`); }
    }
    const norm = validateNetwork(net, { voxelMean: this.voxelMean });
    const pk = packNetwork(norm);
    if (this.netTex) this.netTex.dispose();
    const tex = new THREE.DataTexture(pk.data, pk.width, pk.height, THREE.RGBAFormat, THREE.FloatType);
    tex.minFilter = tex.magFilter = THREE.NearestFilter;
    tex.generateMipmaps = false;
    tex.needsUpdate = true;
    this.netTex = tex;
    this.u.uNet.value = tex;
    this.u.uNumProbes.value = norm.probes.length;
    this.u.uNumLayers.value = norm.layers.length;
    norm.probes.forEach((p, i) => {
      this.u.uProbeA.value[i].set(...pk.probeA[i]);
      this.u.uProbeB.value[i].set(...pk.probeB[i]);
    });
    for (let i = norm.probes.length; i < 64; i++) { this.u.uProbeA.value[i].set(0, 0, 0, 0); this.u.uProbeB.value[i].set(0, 0, 0, 0); }
    pk.layerDesc.forEach((d, k) => this.u.uLayer.value[k].set(...d));
    for (let k = pk.layerDesc.length; k < 6; k++) this.u.uLayer.value[k].set(0, 0, 0, 0);
    this.net = norm;
    if (this.tpe) delete this.tpe.net;
    return { probes: norm.probes.length, layers: norm.layers.map((l) => l.nout), params: norm.nParams,
      probeLabels: norm.probes.map((p) => p.label) };
  }

  // ---------------------------------------------------------------------------------------------
  // camera
  setResolution(w, h) {
    this.width = w; this.height = h;
    this.u.uRes.value.set(w, h);
  }

  /** cam: {position, look_at, up, fov_y_deg} (shots.json convention) */
  setCamera(cam) {
    const pos = v3(cam.position);
    const fwd = v3(cam.look_at).sub(pos).normalize();
    const right = new THREE.Vector3().crossVectors(fwd, v3(cam.up || [0, 1, 0])).normalize();
    const up = new THREE.Vector3().crossVectors(right, fwd);
    this.setCameraBasis(pos, fwd, right, up, cam.fov_y_deg);
  }

  setCameraBasis(pos, fwd, right, up, fovY) {
    this.u.uCamPos.value.copy(pos);
    this.u.uCamFwd.value.copy(fwd);
    this.u.uCamRight.value.copy(right);
    this.u.uCamUp.value.copy(up);
    const ty = Math.tan(0.5 * fovY * RAD);
    this.fovY = fovY;
    this.u.uTan.value.set(ty * this.width / this.height, ty);
  }

  // ---------------------------------------------------------------------------------------------
  // light
  async skyTexture(preset) {
    if (this.skyCache.has(preset)) return this.skyCache.get(preset);
    const p = this.shots.presets[preset];
    if (!p) throw new Error(`unknown preset ${preset}`);
    const buf = await fetchOk(this.opts.sceneBase + p.sky_map, 'buf');
    const rgb = new Float32Array(buf);
    const W = 1024, H = 512;
    if (rgb.length !== W * H * 3) throw new Error(`${p.sky_map}: unexpected size ${rgb.length}`);
    const rgba = new Float32Array(W * H * 4);
    for (let i = 0; i < W * H; i++) { rgba[4 * i] = rgb[3 * i]; rgba[4 * i + 1] = rgb[3 * i + 1]; rgba[4 * i + 2] = rgb[3 * i + 2]; rgba[4 * i + 3] = 1; }
    const tex = new THREE.DataTexture(rgba, W, H, THREE.RGBAFormat, THREE.FloatType);
    tex.minFilter = tex.magFilter = THREE.NearestFilter;
    tex.generateMipmaps = false;
    tex.needsUpdate = true;
    this.skyCache.set(preset, tex);
    return tex;
  }

  setSunVector(s) {
    const sun = v3(s).normalize();
    let u = new THREE.Vector3().crossVectors(sun, new THREE.Vector3(0, 1, 0));
    if (u.length() < 1e-6) u.set(1, 0, 0); else u.normalize();
    const w = new THREE.Vector3().crossVectors(u, sun);
    this.u.uSun.value.copy(sun);
    this.u.uLu.value.copy(u);
    this.u.uLw.value.copy(w);
  }

  /** Exactly a preset from shots.json (render.py, inspect.py). */
  async setPreset(name) {
    const p = this.shots.presets[name];
    if (!p) throw new Error(`unknown preset ${name}; known: ${Object.keys(this.shots.presets).join(', ')}`);
    this.setSunVector(p.toward_sun);
    this.u.uEsun.value.set(...p.E_sun);
    this.u.uLsky.value.set(...p.L_sky);
    this.u.uLgnd.value.set(...p.L_gnd);
    const t = await this.skyTexture(name);
    this.u.uSkyA.value = t;
    this.u.uSkyB.value = t;
    this.u.uSkyMix.value = 0;
    this.u.uSkyRot.value = 0;
    this.preset = name;
  }

  /**
   * Viewer: arbitrary sun elevation/azimuth (degrees). E_sun is recomputed exactly from the SPEC formula;
   * L_sky, L_gnd and the sky map are interpolated linearly in elevation between the two bracketing presets
   * (held at the end presets outside their range) and the map is rotated about y to the new azimuth.
   */
  async setSun(elevDeg, azDeg) {
    const sm = this.shots.sun_model;
    const e = elevDeg * RAD, a = azDeg * RAD;
    this.setSunVector([Math.cos(e) * Math.sin(a), Math.sin(e), -Math.cos(e) * Math.cos(a)]);
    const m = 1 / (Math.sin(e) + 0.50572 * Math.pow(elevDeg + 6.07995, -1.6364));
    const E = [0, 1, 2].map((c) => 20 * Math.exp(-(sm.beta_R[c] * 8000 + sm.beta_M_ext * 1200 + sm.tau_O3[c]) * m));
    this.u.uEsun.value.set(...E);
    const ps = Object.entries(this.shots.presets).sort((p, q) => p[1].elevation_deg - q[1].elevation_deg);
    let i = 0;
    while (i < ps.length - 2 && elevDeg > ps[i + 1][1].elevation_deg) i++;
    const [na, pa] = ps[i], [nb, pb] = ps[i + 1];
    const t = Math.min(1, Math.max(0, (elevDeg - pa.elevation_deg) / (pb.elevation_deg - pa.elevation_deg)));
    const lerp3 = (x, y) => x.map((v, c) => v + (y[c] - v) * t);
    this.u.uLsky.value.set(...lerp3(pa.L_sky, pb.L_sky));
    this.u.uLgnd.value.set(...lerp3(pa.L_gnd, pb.L_gnd));
    this.u.uSkyA.value = await this.skyTexture(na);
    this.u.uSkyB.value = await this.skyTexture(nb);
    this.u.uSkyMix.value = t;
    this.u.uSkyRot.value = (azDeg - pa.azimuth_deg) * RAD;
    return { E_sun: E, presets: [na, nb], t };
  }

  // ---------------------------------------------------------------------------------------------
  // render targets
  ensureTargets() {
    const w = this.width, h = this.height;
    if (this.acc && this.acc[0].width === w && this.acc[0].height === h) return;
    if (this.acc) { [...this.acc, ...this.st, this.ldr].forEach((t) => t.dispose()); }
    const opt = { type: THREE.FloatType, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter, depthBuffer: false, stencilBuffer: false, generateMipmaps: false };
    this.acc = [new THREE.WebGLRenderTarget(w, h, opt), new THREE.WebGLRenderTarget(w, h, opt)];
    // march state ping-pong (two attachments each: state, L)
    this.st = [new THREE.WebGLRenderTarget(w, h, { ...opt, count: 2 }), new THREE.WebGLRenderTarget(w, h, { ...opt, count: 2 })];
    this.ldr = new THREE.WebGLRenderTarget(w, h, { ...opt, type: THREE.UnsignedByteType });
    this.accFront = 0;
    this.job = null;
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

  sync(target, x = 0, y = 0) {
    const b = new Float32Array(4);
    this.renderer.readRenderTargetPixels(target, x, y, 1, 1, b);
    return b;
  }

  checkErrors() {
    if (this.shaderErrors.length) throw new Error(this.shaderErrors.join('\n'));
    if (this.gl.isContextLost()) throw new Error('WebGL context lost');
  }

  readTile(target, r, textureIndex = 0) {
    const n = r[2] * r[3] * 4;
    if (!this._vbuf || this._vbuf.length < n) this._vbuf = new Float32Array(n);
    const b = this._vbuf.subarray(0, n);
    this.renderer.readRenderTargetPixels(target, r[0], r[1], r[2], r[3], b, undefined, textureIndex);
    return b;
  }

  nextStamp() {
    this.stamp = ((this.stamp || 0) % 16000000) + 1;
    this.u.uStamp.value = this.stamp;
    return this.stamp;
  }

  static split4(r) {
    const hw = Math.ceil(r[2] / 2), hh = Math.ceil(r[3] / 2);
    return [[r[0], r[1], hw, hh], [r[0] + hw, r[1], r[2] - hw, hh], [r[0], r[1] + hh, hw, r[3] - hh],
      [r[0] + hw, r[1] + hh, r[2] - hw, r[3] - hh]].filter((q) => q[2] > 0 && q[3] > 0);
  }

  /**
   * One frame of one tile, as a generator that yields after every draw:
   * segment passes (each at most ctl.cap network evaluations per pixel) until every ray in the tile is
   * finished, then the finalize pass (background + running mean from src into dst).
   *
   * Why: macOS silently drops a draw in which a GPU thread runs too long (seen from ~100 ms per pixel with a
   * 42k-parameter network), without reporting a lost context, and after repeated failures it ignores the
   * process's GPU work altogether. Capping evaluations per pass bounds every thread's time; the cap adapts
   * so that each pass takes about ctl.budgetMs (cut when slower, doubled when under a quarter). A draw
   * can never take less time than its slowest thread, so the draw budget also bounds per-thread time.
   * Every draw writes a unique stamp to alpha; the tile is read back and checked (this readback also tells
   * which rays are finished). A dropped draw throws GPU_DROPPED.
   * Returns { tooBig } when even cap = 1 is over budget (the caller splits the tile).
   */
  *tileFrame(mode, r, frame, frame0, src, dst, ctl) {
    const seg = this.material(mode), fin = this.material('final');
    const u = this.u;
    u.uFrame.value = frame;
    u.uFrame0.value = frame0;
    let cur = 0, first = true, tooBig = false;
    for (let pass = 0; ; pass++) {
      if (pass > 5000) throw new Error('march did not finish after 5000 passes');
      const sIn = this.st[cur], sOut = this.st[1 - cur];
      u.uSegFirst.value = first ? 1 : 0;
      u.uState0.value = sIn.textures[0];
      u.uState1.value = sIn.textures[1];
      u.uEvalCap.value = ctl.cap;
      const stamp = this.nextStamp();
      const ts = performance.now();
      this.draw(seg, sOut, r);
      const b = this.readTile(sOut, r, 0);
      const dt = performance.now() - ts;
      this.checkErrors();
      let done = true, ok = true, evals = 0;
      for (let i = 0; i < b.length; i += 4) {
        if (b[i + 3] !== stamp) { ok = false; break; }
        if (b[i] >= 0) done = false;
        evals += b[i + 2];
      }
      ctl.draws++;
      ctl.passes++;
      ctl.maxDraw = Math.max(ctl.maxDraw, dt);
      if (this.opts.trace) console.log(`pass rect ${r.join(',')} frame ${frame} cap ${ctl.cap} ${dt.toFixed(1)} ms ok=${ok} done=${done}`);
      if (!ok) throw new Error(`GPU_DROPPED: the GPU dropped a ${r[2]}x${r[3]} draw (${dt.toFixed(0)} ms, up to ${ctl.cap} evaluations per pixel); the browser must be restarted`);
      if (dt > ctl.budgetMs) {
        if (ctl.cap === 1) tooBig = true;
        ctl.cap = Math.max(1, Math.floor(ctl.cap * Math.min(0.5, 0.7 * ctl.budgetMs / dt)));
      } else if (dt < ctl.budgetMs / 4 && !done) ctl.cap = Math.min(1024, ctl.cap * 2);
      first = false;
      cur = 1 - cur;
      if (done) { ctl.evals += evals; break; }
      yield;
    }
    u.uState0.value = this.st[cur].textures[0];
    u.uState1.value = this.st[cur].textures[1];
    u.uPrev.value = src.texture;
    const stamp = this.nextStamp();
    const ts = performance.now();
    this.draw(fin, dst, r);
    const b = this.readTile(dst, r, 0);
    ctl.draws++;
    ctl.maxDraw = Math.max(ctl.maxDraw, performance.now() - ts);
    for (let i = 3; i < b.length; i += 4) if (b[i] !== stamp) throw new Error('GPU_DROPPED: the GPU dropped a finalize draw; the browser must be restarted');
    return { tooBig };
  }

  makeCtl(mode, budgetMs) {
    this.capMem = this.capMem || {};
    return { cap: this.capMem[mode] || 1, budgetMs, draws: 0, passes: 0, maxDraw: 0, evals: 0, mode };
  }

  /**
   * Render `frames` frames into the accumulation target, tile by tile (each tile gets all its frames before
   * the next one; see tileFrame for the per-pass limits). A tile that is over budget even at one evaluation
   * per pass is split into four and restarted. Rects are in GL coordinates (origin bottom-left).
   */
  renderStill({ mode = 'net', frames = 16, budgetMs = 40, frame0 = 0, tile = [160, 100] } = {}) {
    if (mode === 'net' && !this.net) throw new Error('no network loaded');
    this.ensureTargets();
    const compileMs = this.warmUp(mode) + this.warmUp('final');
    const t0 = performance.now();
    const W = this.width, H = this.height;
    const queue = [];
    for (let y = 0; y < H; y += tile[1]) for (let x = 0; x < W; x += tile[0]) queue.push([x, y, Math.min(tile[0], W - x), Math.min(tile[1], H - y)]);
    const ctl = this.makeCtl(mode, budgetMs);
    let tiles = 0, splits = 0;
    while (queue.length) {
      const r = queue.shift();
      let split = false;
      for (let f = 0; f < frames && !split; f++) {
        const gen = this.tileFrame(mode, r, frame0 + f, frame0, this.acc[f % 2], this.acc[(f + 1) % 2], ctl);
        let res;
        do { res = gen.next(); } while (!res.done);
        if (res.value.tooBig && r[2] * r[3] > 64) split = true;
      }
      if (split) { queue.unshift(...CloudRenderer.split4(r)); splits++; } else tiles++;
    }
    this.capMem[mode] = ctl.cap;
    // with every tile getting the same number of frames, the result is in acc[frames % 2]
    this.accFront = frames % 2;
    const ms = performance.now() - t0;
    return { ms, frames, draws: ctl.draws, passes: ctl.passes, tiles, splits, maxDrawMs: ctl.maxDraw, compileMs,
      evalsPerFrame: ctl.evals / frames, finalCap: ctl.cap };
  }

  /** Compile a program (first use) with a 1-pixel draw, so compile time is not counted as a slow tile. */
  warmUp(key) {
    this.compiled = this.compiled || new Set();
    if (this.compiled.has(key)) return 0;
    this.ensureTargets();
    const mat = this.material(key);
    const t = performance.now();
    this.u.uFrame.value = 0;
    this.u.uFrame0.value = 0;
    this.u.uSegFirst.value = 1;
    this.u.uEvalCap.value = 1;
    this.u.uState0.value = this.st[0].textures[0];
    this.u.uState1.value = this.st[0].textures[1];
    this.u.uPrev.value = this.acc[0].texture;
    const target = key === 'final' ? this.acc[1] : this.st[1];
    this.draw(mat, target, [0, 0, 1, 1]);
    this.sync(target, 0, 0);
    this.checkErrors();
    this.compiled.add(key);
    return performance.now() - t;
  }

  /**
   * Viewer: progressive accumulation over the whole image. Each call runs passes for at most budgetMs and
   * returns the number of completed frames in accTarget(). reset (or a size change) starts over.
   */
  renderProgressive(mode, reset, budgetMs = 30, passBudgetMs = 25, tile = [160, 100]) {
    this.ensureTargets();
    if (reset || !this.job) {
      this.warmUp(mode); this.warmUp('final');
      this.job = { gen: this.progressiveJob(mode, tile, this.makeCtl(mode, passBudgetMs)), frames: 0 };
    }
    const t0 = performance.now();
    while (performance.now() - t0 < budgetMs) {
      const r = this.job.gen.next();
      if (r.value === 'frame') this.job.frames++;
    }
    return this.job.frames;
  }

  *progressiveJob(mode, tile, ctl) {
    const W = this.width, H = this.height;
    let tiles = [];
    for (let y = 0; y < H; y += tile[1]) for (let x = 0; x < W; x += tile[0]) tiles.push([x, y, Math.min(tile[0], W - x), Math.min(tile[1], H - y)]);
    for (let f = 0; ; f++) {
      const src = this.acc[this.accFront], dst = this.acc[1 - this.accFront];
      const next = [];
      for (let k = 0; k < tiles.length; k++) {
        const res = yield* this.tileFrame(mode, tiles[k], f, 0, src, dst, ctl);
        if (res.tooBig && tiles[k][2] * tiles[k][3] > 64) next.push(...CloudRenderer.split4(tiles[k])); else next.push(tiles[k]);
        this.capMem[mode] = ctl.cap;
        yield;
      }
      tiles = next;
      this.accFront = 1 - this.accFront;
      yield 'frame';
    }
  }

  accTarget() { return this.acc[this.accFront]; }

  /** Tone map the accumulation into the LDR target (and optionally to the canvas). */
  toneMap(exposure, toCanvas = false) {
    this.toneMat.uniforms.uAccum.value = this.accTarget().texture;
    this.toneMat.uniforms.uExposure.value = exposure;
    if (toCanvas) {
      this.renderer.setSize(this.width, this.height, false);
      this.renderer.setViewport(0, 0, this.width, this.height);
      this.renderer.setScissorTest(false);
      this.draw(this.toneMat, null, null);
    } else {
      this.draw(this.toneMat, this.ldr, null);
    }
  }

  readLinear() {
    const t = this.accTarget();
    const b = new Float32Array(this.width * this.height * 4);
    this.renderer.readRenderTargetPixels(t, 0, 0, this.width, this.height, b);
    return b;
  }

  readLDR() {
    const b = new Uint8Array(this.width * this.height * 4);
    this.renderer.readRenderTargetPixels(this.ldr, 0, 0, this.width, this.height, b);
    return b;
  }

  // ---------------------------------------------------------------------------------------------
  /**
   * Inspect one pixel (column px, row py from the top) for frame `frame`: per accepted sample, the header
   * values, every probe, every unit's activation and S, read from the shader's debug path.
   */
  inspectPixel(px, py, frame = 0, exact = false) {
    if (!this.net) throw new Error('no network loaded');
    const names = recordLayout(this.net, exact);
    const cols = Math.max(3, Math.ceil(names.length / 4));
    const rows = 512 + 1;
    const opt = { type: THREE.FloatType, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter, depthBuffer: false, stencilBuffer: false };
    const rt = new THREE.WebGLRenderTarget(cols, rows, opt);
    this.u.uDebugPixel.value.set(px, py);
    this.u.uFrame.value = frame;
    const mat = this.material(exact ? 'debug_exact' : 'debug');
    const out = new Float32Array(cols * rows * 4);
    // row by row in chunks (each fragment re-marches the ray)
    const chunk = 64;
    for (let y = 0; y < rows; y += chunk) {
      this.draw(mat, rt, [0, y, cols, Math.min(chunk, rows - y)]);
      this.sync(rt, 0, y);
      this.checkErrors();
    }
    this.renderer.readRenderTargetPixels(rt, 0, 0, cols, rows, out);
    rt.dispose();
    return { names, cols, rows, data: out, header: DEBUG_HEADER.length };
  }
}

export function b64(typed) {
  const u8 = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
  let s = '';
  const CH = 0x8000;
  for (let i = 0; i < u8.length; i += CH) s += String.fromCharCode.apply(null, u8.subarray(i, i + CH));
  return btoa(s);
}
