// HypaNet Clouds embed: the sky of scene/sky.py on the GPU (transmittance LUT + equirect sky view), and the
// per-elevation light table (data/light_table.json, built by compile/light_table.py).
//
//   const sky = await SkyGPU.create(renderer, base);       // fetches shaders/sky_*.glsl, builds the LUT once
//   sky.update(sunDir, cameraAltitude, light.L_gnd);       // rebuilds the sky map when any input changed
//   sky.update(sunDir, cameraAltitude, light.L_gnd, true); // ... at the coarse size (while the sun moves)
//   march uniforms: uSkyMap = sky.skyTexture (rebind after update: the full or the coarse map), uTransLut =
//   sky.lutTexture; the march includes sky.common (GLSL)
//   const light = lightAt(table, elevationDeg); setLightUniforms(uniforms, light);
import * as THREE from './vendor/three.module.min.js';

export const TL_NH = 192;             // transmittance LUT nodes in altitude (h = 100 km * (i / 191)^2)
export const TL_NMU = 2048;           // ... and in cos-zenith (mu = -1 + 2 j / 2047)
const HEADER = 'precision highp float;\nprecision highp int;\nprecision highp sampler2D;\n';
const VERT = 'precision highp float;\nin vec3 position;\nvoid main() { gl_Position = vec4(position.xy, 0.0, 1.0); }\n';

async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  return r.text();
}

/** The three sky shader sources: { common, lut, view } (shaders/sky_common.glsl, sky_lut.frag.glsl, sky_view.frag.glsl). */
export async function loadSkyShaders(base = './') {
  const [common, lut, view] = await Promise.all(['sky_common.glsl', 'sky_lut.frag.glsl', 'sky_view.frag.glsl']
    .map((f) => fetchText(`${base}shaders/${f}`)));
  return { common, lut, view };
}

export class SkyGPU {
  /**
   * renderer: THREE.WebGLRenderer (WebGL2 with EXT_color_buffer_float); shaders: loadSkyShaders() result.
   * opts.width x opts.height: sky map size (1024 x 512, the Python map); opts.fastWidth x opts.fastHeight: the
   * coarse map update(..., fast = true) draws instead (512 x 256, ~0.4x the cost; for frames while the sun moves).
   * opts.lutSteps (400) and opts.viewSteps (160): quadrature steps, as scene/sky.py. The LUT is built here, once;
   * after a WebGL context restore call buildLut() again and set key = fastKey = null.
   */
  constructor(renderer, shaders, opts = {}) {
    this.renderer = renderer;
    this.opts = { width: 1024, height: 512, fastWidth: 512, fastHeight: 256, lutSteps: 400, viewSteps: 160, ...opts };
    this.common = shaders.common;
    const tex = { type: THREE.FloatType, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter, depthBuffer: false, stencilBuffer: false, generateMipmaps: false };
    this.lutTarget = new THREE.WebGLRenderTarget(TL_NMU, TL_NH, tex);
    this.viewOpts = { ...tex, wrapS: THREE.RepeatWrapping };
    this.viewTarget = new THREE.WebGLRenderTarget(this.opts.width, this.opts.height, this.viewOpts);
    this.fastTarget = null;            // created on the first fast update
    this.current = this.viewTarget;    // the map the last update drew (or kept)
    const mat = (frag, uniforms) => new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: VERT,
      fragmentShader: HEADER + shaders.common + '\n' + frag, uniforms, depthTest: false, depthWrite: false });
    this.lutMat = mat(shaders.lut, { uSteps: { value: this.opts.lutSteps } });
    this.viewMat = mat(shaders.view, {
      uTransLut: { value: this.lutTarget.texture }, uSunDir: { value: new THREE.Vector3(0, 1, 0) },
      uCamH: { value: 0 }, uLgnd: { value: new THREE.Vector3() }, uSteps: { value: this.opts.viewSteps },
      uSize: { value: new THREE.Vector2(this.opts.width, this.opts.height) },
    });
    this.scene = new THREE.Scene();
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.lutMat);
    this.quad.frustumCulled = false;
    this.scene.add(this.quad);
    this.cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.key = null;
    this.fastKey = null;
    this.buildLut();
  }

  static async create(renderer, base = './', opts = {}) {
    return new SkyGPU(renderer, await loadSkyShaders(base), opts);
  }

  /** RGBA32F, TL_NMU x TL_NH, rgb = optical depth to space (sky_common.glsl: skyTau reads it as uTransLut). */
  get lutTexture() { return this.lutTarget.texture; }

  /** RGBA32F equirect sky map (row 0 = zenith), nearest; the march's sky() reads it with texelFetch (uSkyMap) and
   * takes its size from textureSize. The full map, or the coarse one when the last update was fast and the full map
   * was stale. */
  get skyTexture() { return this.current.texture; }

  /** GLSL to prepend to the march (after its precision statements): uTransLut, skyTau, sunDiskRadiance. */
  get glsl() { return this.common; }

  draw(material, target) {
    const prev = this.renderer.getRenderTarget();
    this.quad.material = material;
    target.viewport.set(0, 0, target.width, target.height);
    target.scissorTest = false;
    this.renderer.setRenderTarget(target);
    this.renderer.render(this.scene, this.cam);
    this.renderer.setRenderTarget(prev);
  }

  buildLut() {
    this.draw(this.lutMat, this.lutTarget);
  }

  /**
   * Re-renders the sky map for toward-sun sunDir (unit, [x, y, z] or Vector3), camera altitude (m) and ground
   * radiance L_gnd ([r, g, b], the lower hemisphere). fast: draw the coarse map instead (unless the full map is
   * already current). Returns false (and draws nothing) when nothing changed; skyTexture is the map to bind.
   */
  update(sunDir, cameraAltitude, L_gnd, fast = false) {
    const s = Array.isArray(sunDir) ? sunDir : [sunDir.x, sunDir.y, sunDir.z];
    const key = [...s, cameraAltitude, ...L_gnd].join(',');
    if (key === this.key) { this.current = this.viewTarget; return false; }
    if (fast && key === this.fastKey) { this.current = this.fastTarget; return false; }
    if (fast && !this.fastTarget) {
      this.fastTarget = new THREE.WebGLRenderTarget(this.opts.fastWidth, this.opts.fastHeight, this.viewOpts);
    }
    const target = fast ? this.fastTarget : this.viewTarget;
    if (fast) this.fastKey = key; else this.key = key;
    const u = this.viewMat.uniforms;
    u.uSunDir.value.set(s[0], s[1], s[2]).normalize();
    u.uCamH.value = cameraAltitude;
    u.uLgnd.value.set(L_gnd[0], L_gnd[1], L_gnd[2]);
    u.uSize.value.set(target.width, target.height);
    this.draw(this.viewMat, target);
    this.current = target;
    return true;
  }

  dispose() {
    this.lutTarget.dispose();
    this.viewTarget.dispose();
    if (this.fastTarget) this.fastTarget.dispose();
    this.lutMat.dispose();
    this.viewMat.dispose();
    this.quad.geometry.dispose();
  }
}

// ---- light table -------------------------------------------------------------------------------------------------

export async function loadLightTable(base = './') {
  const r = await fetch(`${base}data/light_table.json`);
  if (!r.ok) throw new Error(`fetch ${base}data/light_table.json: HTTP ${r.status}`);
  return r.json();
}

/**
 * Lighting for sun elevation elevDeg, linear in elevation between the table's entries (clamped at -10 and 90):
 * { E_sun_lut: 32 x [r, g, b] at y = y0 + k dy, y0, dy, L_sky: [r, g, b], L_gnd: [r, g, b] }.
 * On the 1-degree segments below -6 degrees L_sky and L_gnd are interpolated geometrically: the twilight sky falls
 * ~5x per degree there, and linear interpolation would be up to 28 % high mid-segment (geometric: 1.2 %).
 */
export function lightAt(table, elevDeg) {
  const els = table.elevations;
  const e = Math.min(els[els.length - 1], Math.max(els[0], elevDeg));
  let lo = 0, hi = els.length - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (els[m] <= e) lo = m; else hi = m;
  }
  const t = (e - els[lo]) / (els[hi] - els[lo]);
  const lerp = (a, b) => a.map((v, i) => v + (b[i] - v) * t);
  const geo = (a, b) => (els[lo] < -6 ? a.map((v, i) => (v > 0 && b[i] > 0 ? v * Math.pow(b[i] / v, t) : v + (b[i] - v) * t))
    : lerp(a, b));
  const lut = lerp(table.E_sun_lut[lo], table.E_sun_lut[hi]);
  return {
    E_sun_lut: Array.from({ length: table.lut_n }, (_, k) => lut.slice(3 * k, 3 * k + 3)),
    y0: table.lut_y0, dy: table.lut_dy,
    L_sky: geo(table.L_sky[lo], table.L_sky[hi]),
    L_gnd: geo(table.L_gnd[lo], table.L_gnd[hi]),
  };
}

/** Writes lightAt()'s result into the video-path uniforms: uEsunLut (32 Vector3), uEsunLutY = (y0, 1/dy), uLsky, uLgnd,
 * and uEsunLutOn = 1 when the material has it. */
export function setLightUniforms(u, light) {
  light.E_sun_lut.forEach((v, k) => u.uEsunLut.value[k].set(v[0], v[1], v[2]));
  u.uEsunLutY.value.set(light.y0, 1 / light.dy);
  u.uLsky.value.set(...light.L_sky);
  u.uLgnd.value.set(...light.L_gnd);
  if (u.uEsunLutOn) u.uEsunLutOn.value = 1;
}
