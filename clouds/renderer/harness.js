// window.harness: the API render.py and inspect.py call through Playwright.
import { CloudRenderer, b64 } from './main.js';

export function installHarness() {
  const params = new URLSearchParams(location.search);
  const state = { r: null, netText: null };
  window.harness = {
    async init() {
      const r = new CloudRenderer({ harness: params.get('harness') === '1', trace: params.get('trace') === '1', sceneBase: params.get('scene') || '../scene/out/' });
      const info = await r.load();
      state.r = r;
      return { ...info, shots: r.shots, voxelMean: r.voxelMean };
    },
    setNetwork(text) {
      state.netText = text;
      return state.r.setNetwork(text);
    },
    async setup({ shot, preset, width, height }) {
      const r = state.r;
      const cam = r.shots.shots[shot];
      if (!cam) throw new Error(`unknown shot ${shot}`);
      r.setResolution(width || cam.width, height || cam.height);
      r.setCamera(cam);
      await r.setPreset(preset);
    },
    async render({ shot, preset, width, height, mode, frames, budgetMs, exposure, frame0, noBackground }) {
      const r = state.r;
      await this.setup({ shot, preset, width, height });
      if (noBackground && !r.opts.harness) throw new Error('noBackground is harness-only');
      r.u.uBgScale.value = noBackground ? 0 : 1;
      const stats = r.renderStill({ mode, frames, budgetMs, frame0: frame0 || 0 });
      const t1 = performance.now();
      r.toneMap(exposure, false);
      const lin = r.readLinear();
      const ldr = r.readLDR();
      stats.readbackMs = performance.now() - t1;
      return { width: r.width, height: r.height, stats, linear: b64(lin), ldr: b64(ldr) };
    },
    async inspect({ shot, preset, width, height, px, py, frame, exact }) {
      const r = state.r;
      await this.setup({ shot, preset, width, height });
      r.u.uBgScale.value = 1;
      const res = r.inspectPixel(px, py, frame || 0, !!exact);
      const u = r.u;
      return { names: res.names, cols: res.cols, rows: res.rows, data: b64(res.data),
        E_sun: u.uEsun.value.toArray(), L_sky: u.uLsky.value.toArray(), L_gnd: u.uLgnd.value.toArray(),
        albedo: u.uAlbedo.value, sun: u.uSun.value.toArray() };
    },
  };
  window.harnessState = state;
  window.harnessLoaded = true;
}
