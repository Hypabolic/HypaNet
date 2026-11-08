// network.json validation and packing for the generic MLP interpreter (SPEC section 4).
//
// The shader reads every weight from one RGBA32F texture (width NET_TEX_WIDTH), four units at a time
// (see packNetwork).
// Probe parameters go to two vec4 uniform arrays, layer descriptors to one (see march.frag.glsl).

export const ACTIVATIONS = ['linear', 'relu', 'exp', 'log', 'tanh', 'sigmoid', 'softplus', 'abs', 'sin',
  'square', 'sqrt', 'recip'];
export const PROBE_TYPES = { density: 0, cos_view_sun: 1, sun_up: 2, view_up: 3, height: 4, const: 5 };
export const FRAMES = { light: 0, world: 1, view: 2 };
export const LIMITS = { probes: 64, layers: 6, units: 128, params: 50000, outputs: 9 };
// Blur pyramid: level k (k = 0..BLUR_LEVELS-1) is a centred box filter 2^k voxels wide (k = 0: raw voxels).
export const BLUR_LEVELS = 8;
export const NET_TEX_WIDTH = 1024;
// Record layout of the inspect debug path: 8 header values, then probes, units, S, [exact extras].
export const DEBUG_HEADER = ['t', 'x', 'y', 'z', 'd', 'T', 'dt', 'alpha'];

const COMMENT_KEYS = new Set(['name', 'comment', 'comments', 'note', 'notes', 'description', 'label', 'doc']);

export class NetworkError extends Error {
  constructor(msg) { super(msg); this.name = 'NetworkError'; }
}

function fail(msg) { throw new NetworkError(msg); }

function isNum(x) { return typeof x === 'number' && Number.isFinite(x); }

// The GPU stores every number as float32: a value finite in float64 but beyond float32's range would become Inf.
export const F32_MAX = 3.4028234663852886e38;
function checkF32(x, what) {
  if (Math.abs(x) > F32_MAX) fail(`${what} = ${x} is outside the float32 range (|x| <= 3.4e38); the GPU stores every number as float32`);
}

function checkKeys(obj, allowed, where) {
  for (const k of Object.keys(obj)) {
    if (!allowed.has(k) && !COMMENT_KEYS.has(k) && !k.startsWith('_')) {
      fail(`${where}: unknown key "${k}" (allowed: ${[...allowed].join(', ')}; comment keys such as "name", "note" or keys starting with "_" are ignored)`);
    }
  }
}

function f32(x) { return Math.fround(x); }

/**
 * Validate a parsed network.json and return a normalised description.
 * opts.voxelMean: mean voxel size in metres (blur -> pyramid level).
 */
export function validateNetwork(net, opts) {
  const voxelMean = opts.voxelMean;
  if (net === null || typeof net !== 'object' || Array.isArray(net)) fail('network.json must be a JSON object with "probes" and "layers"');
  checkKeys(net, new Set(['probes', 'layers']), 'network');
  if (!Array.isArray(net.probes)) fail('"probes" must be an array');
  if (net.probes.length < 1) fail('"probes" must list at least one probe');
  if (net.probes.length > LIMITS.probes) fail(`too many probes: ${net.probes.length} (limit ${LIMITS.probes})`);
  const maxBlur = Math.pow(2, BLUR_LEVELS - 1) * voxelMean;

  const probes = net.probes.map((p, i) => {
    const where = `probes[${i}]`;
    if (p === null || typeof p !== 'object' || Array.isArray(p)) fail(`${where} must be an object`);
    if (!(p.type in PROBE_TYPES)) fail(`${where}: unknown type ${JSON.stringify(p.type)} (allowed: ${Object.keys(PROBE_TYPES).join(', ')})`);
    const out = { type: p.type, code: PROBE_TYPES[p.type], frame: 0, dir: [0, 0, 0], dist: 0, blur: 0, value: 0, lvlLo: 0, lvlT: 0 };
    if (p.type === 'density') {
      checkKeys(p, new Set(['type', 'frame', 'dir', 'dist', 'blur']), where);
      if (!(p.frame in FRAMES)) fail(`${where}: "frame" must be one of light, world, view (got ${JSON.stringify(p.frame)})`);
      if (!Array.isArray(p.dir) || p.dir.length !== 3 || !p.dir.every(isNum)) fail(`${where}: "dir" must be 3 finite numbers`);
      const n = Math.hypot(p.dir[0], p.dir[1], p.dir[2]);
      if (Math.abs(n - 1) > 1e-3) fail(`${where}: "dir" must be a unit vector (|dir| = ${n.toPrecision(6)}; tolerance 1e-3)`);
      const dist = p.dist === undefined ? 0 : p.dist;
      const blur = p.blur === undefined ? 0 : p.blur;
      if (!isNum(dist) || dist < 0) fail(`${where}: "dist" must be a finite number >= 0 (metres)`);
      if (!isNum(blur) || blur < 0) fail(`${where}: "blur" must be a finite number >= 0 (metres)`);
      checkF32(dist, `${where}: "dist"`);
      if (blur > maxBlur * (1 + 1e-9)) fail(`${where}: "blur" ${blur} m exceeds the pyramid's widest box filter, ${maxBlur.toFixed(1)} m`);
      out.frame = FRAMES[p.frame];
      out.frameName = p.frame;
      out.dir = p.dir.map((x) => x / n);
      out.dist = dist;
      out.blur = blur;
      // level = log2(blur / voxel) clamped to [0, BLUR_LEVELS-1]; blur <= one voxel -> raw voxels.
      const lv = blur <= voxelMean ? 0 : Math.min(Math.log2(blur / voxelMean), BLUR_LEVELS - 1);
      out.lvlLo = Math.min(Math.floor(lv), BLUR_LEVELS - 1);
      out.lvlT = lv - out.lvlLo;
      if (out.lvlLo === BLUR_LEVELS - 1) out.lvlT = 0;
    } else if (p.type === 'const') {
      checkKeys(p, new Set(['type', 'value']), where);
      if (!isNum(p.value)) fail(`${where}: const probe needs a finite "value"`);
      checkF32(p.value, `${where}: "value"`);
      out.value = p.value;
    } else {
      checkKeys(p, new Set(['type']), where);
    }
    out.label = probeLabel(out, i);
    return out;
  });

  if (!Array.isArray(net.layers)) fail('"layers" must be an array');
  if (net.layers.length < 1) fail('"layers" must have at least one layer');
  if (net.layers.length > LIMITS.layers) fail(`too many layers: ${net.layers.length} (limit ${LIMITS.layers})`);
  let nin = probes.length;
  let nParams = 0;
  const layers = net.layers.map((L, k) => {
    const where = `layers[${k}]`;
    if (L === null || typeof L !== 'object' || Array.isArray(L)) fail(`${where} must be an object`);
    checkKeys(L, new Set(['W', 'b', 'act']), where);
    if (!Array.isArray(L.W) || L.W.length < 1) fail(`${where}: "W" must be a non-empty array of rows ([out][in])`);
    const nout = L.W.length;
    if (nout > LIMITS.units) fail(`${where}: ${nout} units (limit ${LIMITS.units})`);
    L.W.forEach((row, j) => {
      if (!Array.isArray(row) || row.length !== nin) fail(`${where}: W[${j}] must have ${nin} entries (the size of the previous layer${k === 0 ? ', i.e. the number of probes' : ''}), got ${Array.isArray(row) ? row.length : typeof row}`);
      row.forEach((w, i) => { if (!isNum(w)) fail(`${where}: W[${j}][${i}] is not a finite number`); checkF32(w, `${where}: W[${j}][${i}]`); });
    });
    if (!Array.isArray(L.b) || L.b.length !== nout) fail(`${where}: "b" must have ${nout} entries (one per row of W)`);
    L.b.forEach((b, j) => { if (!isNum(b)) fail(`${where}: b[${j}] is not a finite number`); checkF32(b, `${where}: b[${j}]`); });
    let acts;
    if (typeof L.act === 'string') acts = new Array(nout).fill(L.act);
    else if (Array.isArray(L.act)) {
      if (L.act.length !== nout) fail(`${where}: "act" list has ${L.act.length} entries, expected ${nout} (one per unit) or a single string`);
      acts = L.act;
    } else fail(`${where}: "act" must be a string or a list of strings`);
    const actCodes = acts.map((a, j) => {
      const c = ACTIVATIONS.indexOf(a);
      if (c < 0) fail(`${where}: act[${j}] = ${JSON.stringify(a)} is not one of ${ACTIVATIONS.join(', ')}`);
      return c;
    });
    nParams += nout * nin + nout;
    const out = { nin, nout, W: L.W.map((r) => r.map(f32)), b: L.b.map(f32), acts, actCodes };
    nin = nout;
    return out;
  });
  if (layers[layers.length - 1].nout !== LIMITS.outputs) fail(`the last layer must have exactly ${LIMITS.outputs} outputs (a_sun RGB, a_sky RGB, a_gnd RGB); it has ${layers[layers.length - 1].nout}`);
  if (nParams > LIMITS.params) fail(`too many weights and biases: ${nParams} (limit ${LIMITS.params})`);
  return { probes, layers, nParams };
}

function fmt(x) { return Number.isInteger(x) ? String(x) : x.toPrecision(4).replace(/\.?0+$/, ''); }

function probeLabel(p, i) {
  if (p.type === 'density') return `p${i}:density[${p.frameName} (${p.dir.map(fmt).join(',')}) dist=${fmt(p.dist)} blur=${fmt(p.blur)}]`;
  if (p.type === 'const') return `p${i}:const[${fmt(p.value)}]`;
  return `p${i}:${p.type}`;
}

/**
 * Pack a validated network into the texture and uniform arrays used by march.frag.glsl.
 * Units are processed in blocks of four: for layer k and output block j4 (units 4*j4 .. 4*j4+3) there are
 * nin4 = ceil(nin/4) groups of 4 texels, group i holding W[4*j4 + c][4i .. 4i+3] in texel c, then one texel of
 * the 4 biases and one texel of the 4 activation codes. Padding weights, biases and codes are 0 (linear),
 * so padding units output exactly 0.
 */
export function packNetwork(norm) {
  let texels = 0;
  const layerDesc = [];
  for (const L of norm.layers) {
    const nin4 = Math.ceil(L.nin / 4), nout4 = Math.ceil(L.nout / 4);
    layerDesc.push([texels, L.nin, L.nout, nin4]);
    texels += nout4 * (4 * nin4 + 2);
  }
  const height = Math.max(1, Math.ceil(texels / NET_TEX_WIDTH));
  const data = new Float32Array(NET_TEX_WIDTH * height * 4);
  norm.layers.forEach((L, k) => {
    const [off, nin, nout, nin4] = layerDesc[k];
    const stride = 4 * nin4 + 2;
    for (let j = 0; j < nout; j++) {
      const j4 = j >> 2, c = j & 3;
      const base = off + j4 * stride;
      for (let i = 0; i < nin; i++) data[(base + 4 * (i >> 2) + c) * 4 + (i & 3)] = L.W[j][i];
      data[(base + 4 * nin4) * 4 + c] = L.b[j];
      data[(base + 4 * nin4 + 1) * 4 + c] = L.actCodes[j];
    }
  });
  const probeA = norm.probes.map((p) => [p.code, p.frame, p.lvlLo, p.lvlT]);
  const probeB = norm.probes.map((p) => (p.type === 'const' ? [p.value, 0, 0, 0] : [p.dir[0], p.dir[1], p.dir[2], p.dist]));
  return { data, width: NET_TEX_WIDTH, height, layerDesc, probeA, probeB, texels };
}

/** Names of every value in an inspect record (same order as the shader's debug capture indices). */
export function recordLayout(norm, withExact) {
  const names = [...DEBUG_HEADER];
  norm.probes.forEach((p) => names.push(p.label));
  norm.layers.forEach((L, k) => {
    for (let j = 0; j < L.nout; j++) names.push(`L${k}.u${j}:${L.acts[j]}`);
  });
  names.push('S.r', 'S.g', 'S.b');
  if (withExact) names.push('exact.T_sun', 'exact.phase');
  return names;
}
