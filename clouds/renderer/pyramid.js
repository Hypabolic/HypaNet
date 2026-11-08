// Blurred density pyramid for the `density` probe's `blur` parameter (SPEC section 4).
//
// Level 0 is the raw grid (sampled trilinearly). Level k >= 1 is the density convolved with a centred box
// filter of width w_k = 2^k voxels along each axis (the voxels are treated as piecewise-constant cells and
// everything outside the box counts as 0), stored on a grid with spacing f_k = max(1, w_k / 4) voxels and
// one extra cell of padding on each side, so that trilinear interpolation of the stored samples is valid
// right up to the box faces. (Trilinear interpolation between stored samples adds a tent of half-width f_k,
// which widens the effective kernel by about 6% in standard deviation; with f_k = w_k / 2 it was 22%.) Coarse cell j (0-based) of level k is centred at voxel coordinate (j - 0.5) f_k,
// where voxel coordinate q = (p - box_min) / voxel_size (voxel i spans [i, i+1)).
//
// Levels 1..7 are packed one after another along z into a single R8 3D texture (the atlas); the shader maps
// q to atlas texture coordinates with a per-level scale and offset: tc = q * scale + offset.

export function buildPyramid(raw, dims, nLevels) {
  const [nx, ny, nz] = dims;
  const levels = [];
  for (let k = 1; k < nLevels; k++) {
    const w = Math.pow(2, k);
    const f = Math.max(1, w / 4);
    const out = [Math.ceil(nx / f) + 2, Math.ceil(ny / f) + 2, Math.ceil(nz / f) + 2];
    let cur = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) cur[i] = raw[i];
    let cdims = [nx, ny, nz];
    for (let axis = 0; axis < 3; axis++) {
      const r = boxAxis(cur, cdims, axis, f, w, out[axis]);
      cur = r.data;
      cdims = r.dims;
    }
    const bytes = new Uint8Array(cur.length);
    for (let i = 0; i < cur.length; i++) bytes[i] = Math.min(255, Math.max(0, Math.round(cur[i])));
    levels.push({ k, f, w, dims: out, bytes });
  }
  // Atlas: stack along z.
  const AX = Math.max(...levels.map((l) => l.dims[0]));
  const AY = Math.max(...levels.map((l) => l.dims[1]));
  const AZ = levels.reduce((s, l) => s + l.dims[2], 0);
  const atlas = new Uint8Array(AX * AY * AZ);
  let z0 = 0;
  const scale = [[0, 0, 0]];
  const offset = [[0, 0, 0]];
  for (const l of levels) {
    const [lx, ly, lz] = l.dims;
    for (let z = 0; z < lz; z++) {
      for (let y = 0; y < ly; y++) {
        const src = lx * (y + ly * z);
        const dst = AX * (y + AY * (z + z0));
        atlas.set(l.bytes.subarray(src, src + lx), dst);
      }
    }
    // coarse texel coordinate c = q / f + 1 (texel centres at j + 0.5); atlas tc = (c + origin) / A
    scale.push([1 / (l.f * AX), 1 / (l.f * AY), 1 / (l.f * AZ)]);
    offset.push([1 / AX, 1 / AY, (1 + z0) / AZ]);
    l.z0 = z0;
    z0 += lz;
  }
  return { atlas, atlasDims: [AX, AY, AZ], levels: levels.map(({ k, f, w, dims, z0 }) => ({ k, f, w, dims, z0 })), scale, offset };
}

// One separable pass: resample `axis` of `src` (dims) with a centred box of width w (voxels) at spacing f,
// output cell j centred at (j - 0.5) f. Values stay in byte units (0..255).
function boxAxis(src, dims, axis, f, w, nOut) {
  const n = dims[axis];
  const odims = dims.slice();
  odims[axis] = nOut;
  const dst = new Float32Array(odims[0] * odims[1] * odims[2]);
  const sStride = axis === 0 ? 1 : axis === 1 ? dims[0] : dims[0] * dims[1];
  const dStride = axis === 0 ? 1 : axis === 1 ? odims[0] : odims[0] * odims[1];
  const P = new Float64Array(n + 1);
  const line = new Float64Array(n);
  // integral of the piecewise-constant line over [0, x], x clamped to [0, n]
  const F = (x) => {
    if (x <= 0) return 0;
    if (x >= n) return P[n];
    const i = Math.floor(x);
    return P[i] + (x - i) * line[i];
  };
  const lo = new Float64Array(nOut);
  const hi = new Float64Array(nOut);
  for (let j = 0; j < nOut; j++) {
    const c = (j - 0.5) * f;
    lo[j] = c - w / 2;
    hi[j] = c + w / 2;
  }
  // iterate over all lines along `axis`
  const [a0, a1] = axis === 0 ? [1, 2] : axis === 1 ? [0, 2] : [0, 1];
  for (let v = 0; v < dims[a1]; v++) {
    for (let u = 0; u < dims[a0]; u++) {
      const sBase = idx(dims, a0, u, a1, v);
      const dBase = idx(odims, a0, u, a1, v);
      P[0] = 0;
      for (let i = 0; i < n; i++) {
        const val = src[sBase + i * sStride];
        line[i] = val;
        P[i + 1] = P[i] + val;
      }
      for (let j = 0; j < nOut; j++) dst[dBase + j * dStride] = (F(hi[j]) - F(lo[j])) / w;
    }
  }
  return { data: dst, dims: odims };
}

function idx(d, a0, u, a1, v) {
  const c = [0, 0, 0];
  c[a0] = u;
  c[a1] = v;
  return c[0] + d[0] * (c[1] + d[1] * c[2]);
}
