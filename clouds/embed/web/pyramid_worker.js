// Module worker: builds the density pyramid (pyramid.js) off the main thread, so the loading state keeps animating.
// In: { bytes: Uint8Array (transferred), dims, levels }. Out: buildPyramid()'s result (atlas transferred) + ms.
import { buildPyramid } from './pyramid.js';

self.onmessage = (e) => {
  const { bytes, dims, levels } = e.data;
  try {
    const t0 = performance.now();
    const pyr = buildPyramid(bytes, dims, levels);
    self.postMessage({ pyr, bytes, ms: performance.now() - t0 }, [pyr.atlas.buffer, bytes.buffer]);
  } catch (err) {
    self.postMessage({ error: String(err && err.message || err) });
  }
};
