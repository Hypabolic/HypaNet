// Test hooks for embed/web/sky.js (driven by test_sky.py): read back the GPU transmittance LUT and sky maps,
// evaluate skyTau / sunDiskRadiance at arbitrary inputs, interpolate the light table, and time the passes.
import * as THREE from '../web/vendor/three.module.min.js';
import { SkyGPU, loadSkyShaders, loadLightTable, lightAt, TL_NH, TL_NMU } from '../web/sky.js';

const BASE = '../web/';

function b64(f32) {
  const u8 = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}

function unb64(str) {
  const bin = atob(str);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}

const PROBE = `
uniform highp sampler2D uIn;          // N x 1 RGBA32F inputs
uniform int uMode;                    // 0: rgb = skyTau(in.x, in.y); 1: rgb = sunDiskRadiance(in.xyz, uSun, in.w)
uniform vec3 uSun;
out vec4 fragColor;
void main() {
  vec4 a = texelFetch(uIn, ivec2(gl_FragCoord.xy), 0);
  fragColor = vec4(uMode == 0 ? skyTau(a.x, a.y) : sunDiskRadiance(a.xyz, uSun, a.w), 1.0);
}`;

export function installSkyTest() {
  const T = {};
  const read = (rt, w = rt.width, h = rt.height) => {
    const b = new Float32Array(w * h * 4);
    T.renderer.readRenderTargetPixels(rt, 0, 0, w, h, b);
    return b;
  };
  const sync = (rt) => { const b = new Float32Array(4); T.renderer.readRenderTargetPixels(rt, 0, 0, 1, 1, b); };
  const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

  window.skyTest = {
    async init() {
      const canvas = document.createElement('canvas');
      T.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, depth: false, stencil: false,
        powerPreference: 'high-performance' });
      T.errors = [];
      T.renderer.debug.onShaderError = (gl, program, vs, fs) => {
        T.errors.push(`${gl.getProgramInfoLog(program)}\n${gl.getShaderInfoLog(vs)}\n${gl.getShaderInfoLog(fs)}`);
      };
      const gl = T.renderer.getContext();
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      T.shaders = await loadSkyShaders(BASE);
      const t0 = performance.now();
      T.sky = new SkyGPU(T.renderer, T.shaders);
      sync(T.sky.lutTarget);
      const firstLutMs = performance.now() - t0;
      T.table = await loadLightTable(BASE);
      if (T.errors.length) throw new Error(T.errors.join('\n'));
      return { renderer: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
        colorBufferFloat: !!gl.getExtension('EXT_color_buffer_float'), firstLutMs, nh: TL_NH, nmu: TL_NMU };
    },
    readLut() { return b64(read(T.sky.lutTarget)); },
    sky({ s, camH, Lgnd }) {
      T.sky.key = null;
      T.sky.update(s, camH, Lgnd);
      if (T.errors.length) throw new Error(T.errors.join('\n'));
      const rt = T.sky.viewTarget;
      return { w: rt.width, h: rt.height, data: b64(read(rt)) };
    },
    probe({ mode, inputs, sun }) {
      const x = unb64(inputs);
      const n = x.length / 4;
      const tin = new THREE.DataTexture(x, n, 1, THREE.RGBAFormat, THREE.FloatType);
      tin.minFilter = tin.magFilter = THREE.NearestFilter;
      tin.needsUpdate = true;
      const rt = new THREE.WebGLRenderTarget(n, 1, { type: THREE.FloatType, format: THREE.RGBAFormat,
        minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false, generateMipmaps: false });
      const mat = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: T.sky.lutMat.vertexShader,
        fragmentShader: 'precision highp float;\nprecision highp int;\nprecision highp sampler2D;\n' + T.shaders.common + PROBE,
        uniforms: { uIn: { value: tin }, uMode: { value: mode }, uSun: { value: new THREE.Vector3(...(sun || [0, 1, 0])) },
          uTransLut: { value: T.sky.lutTexture } }, depthTest: false, depthWrite: false });
      T.sky.draw(mat, rt);
      if (T.errors.length) throw new Error(T.errors.join('\n'));
      const out = read(rt);
      [tin, rt, mat].forEach((o) => o.dispose());
      return b64(out);
    },
    lightAt(elevs) { return elevs.map((e) => lightAt(T.table, e)); },
    // indicative GPU time per pass: draw + a 1-pixel readback (which waits for the GPU), median over reps;
    // the readback alone is timed and reported separately
    timing({ reps = 10, sizes = [[1024, 512]] } = {}) {
      const res = {};
      const tRead = [], tLut = [];
      for (let i = 0; i < reps; i++) {
        let t = performance.now(); sync(T.sky.lutTarget); tRead.push(performance.now() - t);
        t = performance.now(); T.sky.buildLut(); sync(T.sky.lutTarget); tLut.push(performance.now() - t);
      }
      res.readbackMs = median(tRead);
      res.lutMs = median(tLut);
      for (const [w, h] of sizes) {
        const sky = new SkyGPU(T.renderer, T.shaders, { width: w, height: h });
        const tv = [];
        for (let i = 0; i < reps; i++) {
          const e = (5 + 3 * i) * Math.PI / 180;
          const t = performance.now();
          sky.update([Math.cos(e), Math.sin(e), 0], 300, [0.01, 0.01, 0.01]);
          sync(sky.viewTarget);
          tv.push(performance.now() - t);
        }
        res[`view${w}x${h}Ms`] = median(tv);
        sky.dispose();
      }
      return res;
    },
  };
  window.skyTestLoaded = true;
}
