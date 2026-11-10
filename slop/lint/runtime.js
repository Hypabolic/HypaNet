// Slop transformer runtime (SPEC.md section 7). The same forward pass as ui/model.py, in plain JS: weights come
// from slop-net.bin as float32 and every computation runs in Float64. Bidirectional attention, signed node-distance
// bias (column 127 for the BOS key), no normalisation, logits at each node's EON, argmax ties to the lower class,
// and the pointer rule from the artefact's pointer_head. Reads slop.net/2 (id layout 1 or 2; tensors stored as row
// runs or as sparse cells, see lint/compile.py) and r1's slop.net/1. Node built-ins only.

import { createHash } from 'node:crypto';
import { Vocab, encodeScreen, idLayout, BOS, CLASSES } from './tokenizer.js';

export const MAGIC = 'SLOPNET1';
export const FORMATS = Object.freeze(['slop.net/1', 'slop.net/2']);
export const D = 128;
export const L = 4;
export const H = 8;
export const DH = 16;
export const M = 512;
export const N_CLASSES = 4;
const N_CLUSTERS = 64;
const N_FLAGS = 32;
const NB = 128;
const BOS_COL = 127;
const MAX_DIST = 63;
const QK_SCALE = 0.25;

// Parse slop-net.bin (a Buffer or Uint8Array) into dense Float64 tensors, the vocabulary and the header.
export function loadNet(bytes) {
  const buf = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (buf.subarray(0, 8).toString('latin1') !== MAGIC) throw new Error('not a SLOPNET1 file');
  const hlen = buf.readUInt32LE(8);
  const header = JSON.parse(buf.subarray(12, 12 + hlen).toString('utf8'));
  if (!FORMATS.includes(header.format)) throw new Error('wrong slop-net format tag');
  const v2 = header.format === 'slop.net/2';
  const ids = v2 ? header.ids : 1;
  const layout = idLayout(ids);
  const base = 12 + hlen;
  const tensors = new Map();
  for (const e of header.tensors) {
    const rows = e.shape.length === 1 ? 1 : e.shape[0];
    const cols = e.shape.length === 1 ? e.shape[0] : e.shape[1];
    const t = new Float64Array(rows * cols);
    let pos = base + e.offset;
    const enc = e.enc ?? 'rows';
    if (enc === 'rows') {
      for (const [a, b] of e.rows) {
        for (let r = a; r < b; r++) {
          for (let c = 0; c < cols; c++) {
            t[r * cols + c] = buf.readFloatLE(pos);
            pos += 4;
          }
        }
      }
    } else if (enc === 'cells' && v2) {
      // nrows u32 row indices, then a ceil(cols / 8)-byte column mask per row, then the set cells' float32 values
      const k = e.nrows;
      const mb = (cols + 7) >> 3;
      const idxPos = pos;
      let maskPos = pos + 4 * k;
      pos = maskPos + mb * k;
      let prev = -1;
      for (let i = 0; i < k; i++) {
        const r = buf.readUInt32LE(idxPos + 4 * i);
        if (r <= prev || r >= rows) throw new Error(`${e.name}: row index not ascending or out of range`);
        prev = r;
        for (let c = 0; c < cols; c++) {
          if ((buf[maskPos + (c >> 3)] >> (c & 7)) & 1) {
            t[r * cols + c] = buf.readFloatLE(pos);
            pos += 4;
          }
        }
        maskPos += mb;
      }
    } else throw new Error(`${e.name}: unknown encoding ${enc}`);
    if (pos !== base + e.offset + e.nbytes) throw new Error(`${e.name}: byte count mismatch`);
    tensors.set(e.name, t);
  }
  const tokShape = header.tensors.find((e) => e.name === 'tok_emb.weight')?.shape;
  if (!tokShape || tokShape[0] !== layout.vocabSize || tokShape[1] !== D) {
    throw new Error(`tok_emb.weight shape ${JSON.stringify(tokShape)} does not match id layout ${ids} ([${layout.vocabSize}, ${D}])`);
  }
  const vb = buf.subarray(base + header.vocab.offset, base + header.vocab.offset + header.vocab.nbytes);
  if (base + header.vocab.offset + header.vocab.nbytes !== buf.length) throw new Error('trailing bytes after the vocabulary');
  const vsha = createHash('sha256').update(vb).digest('hex');
  if (vsha !== header.vocab_sha256) throw new Error('vocab sha256 does not match the header');
  const ph = header.pointer_head;
  if (!Array.isArray(ph) || ph.length !== 2 || ph[0] < 1 || ph[0] > L || ph[1] < 0 || ph[1] >= H) throw new Error('bad pointer_head');
  return {
    header,
    tensors,
    ids,
    vocab: Vocab.fromText(vb.toString('utf8'), ids),
    pointerHead: [ph[0], ph[1]],
    sha256: header.artifact_sha256,
  };
}

// y[n x out] = x[n x inDim] W^T + b, with W stored [out x inDim]; the bias is added after the dot product, as in
// numpy's `x @ W.T + b`.
function linear(x, n, inDim, W, b, outDim) {
  const y = new Float64Array(n * outDim);
  for (let i = 0; i < n; i++) {
    const xo = i * inDim;
    for (let o = 0; o < outDim; o++) {
      const wo = o * inDim;
      let s = 0;
      for (let k = 0; k < inDim; k++) s += x[xo + k] * W[wo + k];
      y[i * outDim + o] = s + b[o];
    }
  }
  return y;
}

function addRow(h, pos, table, row) {
  const ho = pos * D;
  const to = row * D;
  for (let d = 0; d < D; d++) h[ho + d] += table[to + d];
}

export function embed(net, w) {
  const n = w.tokens.length;
  if (n === 0 || w.tokens[0] !== BOS || w.node[0] !== 0) throw new Error('a window starts with BOS at node 0');
  const T = net.tensors;
  const h = new Float64Array(n * D);
  const E = T.get('tok_emb.weight');
  for (let t = 0; t < n; t++) for (let d = 0; d < D; d++) h[t * D + d] = E[w.tokens[t] * D + d];
  const P = T.get('off_emb.weight');
  const R = T.get('role_emb.weight');
  const C = T.get('cluster_emb.weight');
  const F = T.get('flag_emb.weight');
  for (let t = 1; t < n; t++) addRow(h, t, P, w.offsets[t]);
  for (let t = 1; t < n; t++) addRow(h, t, R, w.roles[t]);
  for (let t = 1; t < n; t++) addRow(h, t, C, w.clusters[t] % N_CLUSTERS);
  for (let f = 0; f < N_FLAGS; f++) {
    for (let t = 1; t < n; t++) if ((w.flags[t] >>> f) & 1) addRow(h, t, F, f);
  }
  return h;
}

// One window: logits [nNodes][4] at each EON and the pointer head's attention rows at each EON.
export function forward(net, w) {
  const T = net.tensors;
  const n = w.tokens.length;
  const node = w.node;
  let h = embed(net, w);
  const [ptrLayer, ptrHead] = net.pointerHead;
  const eon = w.eon;
  const eonRow = new Map(eon.map((p, m) => [p, m]));
  const ptrAttn = eon.map(() => new Float64Array(n));
  const score = new Float64Array(n);
  for (let li = 0; li < L; li++) {
    const pre = `layers.${li}.`;
    const q = linear(h, n, D, T.get(pre + 'attn.q.weight'), T.get(pre + 'attn.q.bias'), D);
    const k = linear(h, n, D, T.get(pre + 'attn.k.weight'), T.get(pre + 'attn.k.bias'), D);
    const v = linear(h, n, D, T.get(pre + 'attn.v.weight'), T.get(pre + 'attn.v.bias'), D);
    const B = T.get(pre + 'attn.rel_bias');
    const ctx = new Float64Array(n * D);
    for (let hd = 0; hd < H; hd++) {
      const ho = hd * DH;
      for (let i = 0; i < n; i++) {
        let mx = -Infinity;
        for (let j = 0; j < n; j++) {
          let s = 0;
          for (let d = 0; d < DH; d++) s += q[i * D + ho + d] * QK_SCALE * k[j * D + ho + d];
          let c;
          if (node[j] === 0) c = BOS_COL;
          else c = Math.max(-MAX_DIST, Math.min(MAX_DIST, node[j] - node[i])) + MAX_DIST;
          s += B[hd * NB + c];
          score[j] = s;
          if (s > mx) mx = s;
        }
        let sum = 0;
        for (let j = 0; j < n; j++) {
          score[j] = Math.exp(score[j] - mx);
          sum += score[j];
        }
        for (let j = 0; j < n; j++) score[j] /= sum;
        for (let d = 0; d < DH; d++) {
          let acc = 0;
          for (let j = 0; j < n; j++) acc += score[j] * v[j * D + ho + d];
          ctx[i * D + ho + d] = acc;
        }
        if (li === ptrLayer - 1 && hd === ptrHead && eonRow.has(i)) ptrAttn[eonRow.get(i)].set(score);
      }
    }
    const att = linear(ctx, n, D, T.get(pre + 'attn.o.weight'), T.get(pre + 'attn.o.bias'), D);
    for (let x = 0; x < n * D; x++) h[x] += att[x];
    const hid = linear(h, n, D, T.get(pre + 'mlp.fc1.weight'), T.get(pre + 'mlp.fc1.bias'), M);
    for (let x = 0; x < n * M; x++) if (!(hid[x] > 0)) hid[x] = 0;
    const mo = linear(hid, n, M, T.get(pre + 'mlp.fc2.weight'), T.get(pre + 'mlp.fc2.bias'), D);
    for (let x = 0; x < n * D; x++) h[x] += mo[x];
  }
  const hE = new Float64Array(eon.length * D);
  eon.forEach((p, m) => hE.set(h.subarray(p * D, (p + 1) * D), m * D));
  const lg = linear(hE, eon.length, D, T.get('head.weight'), T.get('head.bias'), N_CLASSES);
  const logits = eon.map((_, m) => Array.from(lg.subarray(m * N_CLASSES, (m + 1) * N_CLASSES)));
  return { logits, pointerAttn: ptrAttn };
}

// argmax, ties to the lower index.
export function argmaxClass(logits) {
  let best = 0;
  for (let c = 1; c < logits.length; c++) if (logits[c] > logits[best]) best = c;
  return best;
}

// (target window node or null, BOS weight) for one EON attention row.
export function pointerFromAttn(row, node, selfNode) {
  let nNodes = 0;
  for (const m of node) if (m > nNodes) nNodes = m;
  const sums = new Float64Array(nNodes + 1);
  for (let t = 0; t < row.length; t++) sums[node[t]] += row[t];
  const bos = row[0];
  let best = null;
  let bestVal = -1;
  for (let m = 1; m <= nNodes; m++) {
    if (m === selfNode) continue;
    if (sums[m] > bestVal) { best = m; bestVal = sums[m]; }
  }
  if (bos >= 0.5) best = null;
  return { target: best, bos };
}

export function predictEncoded(net, enc) {
  const preds = new Array(enc.nodes.length).fill(null);
  enc.windows.forEach((w, wi) => {
    const res = forward(net, w);
    w.nodes.forEach((x, m) => {
      const lg = res.logits[m];
      const cls = CLASSES[argmaxClass(lg)];
      const { target, bos } = pointerFromAttn(res.pointerAttn[m], w.node, m + 1);
      const pointer = target !== null && (cls === 'ECHO' || cls === 'NARRATE') ? w.nodes[target - 1] : null;
      preds[x] = { index: x, cls, logits: lg, pointer, bosWeight: bos, window: wi };
    });
  });
  return preds;
}

export function predictScreen(net, screen) {
  return predictEncoded(net, encodeScreen(screen, net.vocab));
}
