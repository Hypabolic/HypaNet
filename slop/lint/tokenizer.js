// Slop tokeniser (SPEC.md section 6) and sequence builder (sections 4 and 7). Mirrors ui/tokenizer.py exactly;
// the rules are documented there. Reads the fixed id table ui/ids.json and the pinned Unicode data
// ui/unicode_tables.json, so both tokenisers share one copy of the role, flag and id lists. Node built-ins only.

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { stem as porterStem } from './porter.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const UI_DIR = process.env.SLOP_UI_DIR || path.join(HERE, '..', 'ui');
export const IDS = JSON.parse(readFileSync(path.join(UI_DIR, 'ids.json'), 'utf8'));
const TABLES = JSON.parse(readFileSync(path.join(UI_DIR, 'unicode_tables.json'), 'utf8'));

export const ROLES = Object.freeze([...IDS.roles]);
export const FLAGS = Object.freeze([...IDS.flags]);
export const CLASSES = Object.freeze([...IDS.classes]);
export const CONTEXT_ROLES = new Set(IDS.context_roles);
export const ROLE_IDS = new Map(ROLES.map((r, i) => [r, i]));
export const FLAG_IDS = new Map(FLAGS.map((f, i) => [f, i]));
if (ROLES.length !== 36 || FLAGS.length !== 28) throw new Error('ids.json: role or flag list has the wrong length');

export const PAD = IDS.special.PAD;
export const BOS = IDS.special.BOS;
export const EON = IDS.special.EON;
export const VAR = IDS.special.VAR;
export const ALT = IDS.special.ALT;
export const LZ = IDS.special.LZ;
const MARKERS = new Map([[IDS.markers.VAR, VAR], [IDS.markers.ALT, ALT]]); // U+E000, U+E001
if (IDS.markers.VAR !== '\uE000' || IDS.markers.ALT !== '\uE001') throw new Error('ids.json: markers must be U+E000 and U+E001');

export const GLYPH_IDS = Object.fromEntries(IDS.glyph_classes.map((g, i) => [g, IDS.glyph_base + i]));
const GLYPH_OF_CHAR = new Map();
for (const [cls, chars] of Object.entries(IDS.glyph_chars)) for (const ch of chars) GLYPH_OF_CHAR.set(ch, cls);
const PUNCT_IDS = new Map(Array.from(IDS.punct_chars).map((ch, i) => [ch, IDS.punct_base + i]));
const NUMBER_BASE = IDS.number_base;
const { YEAR, NUM, DEC } = IDS.number_tokens;
// Word-id layouts (ids.json id_layouts): 1 is r1's (3,584 stems, 6,144 hash buckets), 2 is r2's (16,384 stems,
// 8,192 hash buckets). The constants below are layout 1; a Vocab carries its layout.
export const ID_LAYOUTS = new Map(Object.entries(IDS.id_layouts).map(([k, v]) => [Number(k), Object.freeze({
  version: Number(k), vocabBase: v.vocab_base, vocabMax: v.vocab_max, hashBase: v.hash_base, hashBuckets: v.hash_buckets,
  vocabSize: v.vocab_size,
})]));
export function idLayout(ids = 1) {
  const lay = Number.isInteger(ids) ? ID_LAYOUTS.get(ids) : undefined;
  if (!lay) throw new TokenizerError(`unknown id layout ${JSON.stringify(ids)} (known: ${[...ID_LAYOUTS.keys()].join(', ')})`);
  return lay;
}
export const VOCAB_BASE = IDS.vocab_base;
export const VOCAB_MAX = IDS.vocab_max;
export const HASH_BASE = IDS.hash_base;
export const HASH_BUCKETS = IDS.hash_buckets;
export const V = IDS.vocab_size;
export const MAX_NODE_TOKENS = IDS.max_node_tokens;
export const MAX_WINDOW_NODES = IDS.max_window_nodes;
export const MAX_WINDOW_POSITIONS = IDS.max_window_positions;

const ASSIGNED = TABLES.assigned;
const EMOJI = TABLES.emoji;
const CASEFOLD = new Map(Object.entries(TABLES.casefold).map(([cp, s]) => [Number(cp), s]));

const APOSTROPHES = new Set(["'", '’']);
const HYPHEN = '‐';
const NUMBER = /(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?/y;

const RE_L = /^\p{L}$/u;
const RE_M = /^\p{M}$/u;
const RE_MN_ME = /^[\p{Mn}\p{Me}]$/u;
const RE_UPPER = /^[\p{Lu}\p{Lt}]$/u;
const RE_LOWER = /^\p{Ll}$/u;
const RE_SPACE = /^[\p{Z}\p{Cc}]$/u;
const RE_CF = /^\p{Cf}$/u;

export class TokenizerError extends Error {}

function inRanges(cp, ranges) {
  let lo = 0;
  let hi = ranges.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (cp < ranges[mid][0]) hi = mid - 1;
    else if (cp > ranges[mid][1]) lo = mid + 1;
    else return true;
  }
  return false;
}

const cpOf = (ch) => ch.codePointAt(0);
export const isAssigned = (ch) => inRanges(cpOf(ch), ASSIGNED);
const isRegional = (cp) => cp >= 0x1f1e6 && cp <= 0x1f1ff;
export const isPictograph = (ch) => isRegional(cpOf(ch)) || inRanges(cpOf(ch), EMOJI);
const isUpper = (ch) => RE_UPPER.test(ch);
const isLower = (ch) => RE_LOWER.test(ch);
const isLetter = (ch) => isAssigned(ch) && RE_L.test(ch);

// NFKC on runs of assigned code points; unassigned ones and the markers pass through untouched.
function nfkcProtected(text) {
  let out = '';
  let run = '';
  for (const ch of text) {
    if (isAssigned(ch) && !MARKERS.has(ch)) run += ch;
    else {
      if (run) { out += run.normalize('NFKC'); run = ''; }
      out += ch;
    }
  }
  if (run) out += run.normalize('NFKC');
  return out;
}

// Split one letter run (array of code points) at lower-to-upper changes and before the last capital of an
// upper-case run.
export function camelSplit(run) {
  const parts = [];
  let start = 0;
  for (let p = 1; p < run.length; p++) {
    const a = run[p - 1];
    const b = run[p];
    if ((isLower(a) && isUpper(b)) || (isUpper(a) && isUpper(b) && p + 1 < run.length && isLower(run[p + 1]))) {
      parts.push(run.slice(start, p));
      start = p;
    }
  }
  parts.push(run.slice(start));
  return parts;
}

export function casefold(s) {
  let out = '';
  for (const ch of s) {
    const f = CASEFOLD.get(cpOf(ch));
    out += f !== undefined ? f : ch.toLowerCase();
  }
  return out;
}

// Case folding, NFKD with non-spacing and enclosing marks dropped, apostrophes removed.
export function foldWord(part) {
  let out = '';
  for (const ch of casefold(part).normalize('NFKD')) {
    if (RE_MN_ME.test(ch) || APOSTROPHES.has(ch)) continue;
    out += ch;
  }
  return out;
}

export function hashBucket(stem, ids = 1) {
  const lay = idLayout(ids);
  const digest = createHash('sha256').update(stem, 'utf8').digest();
  return lay.hashBase + (digest.readUInt32BE(0) % lay.hashBuckets);
}

export function numberTokens(text) {
  const out = [];
  const intPart = text.split('.')[0];
  if (intPart.length >= 2 && intPart[0] === '0') out.push(LZ);
  if (text.includes('.')) out.push(DEC);
  else if (text.includes(',')) out.push(NUM);
  else {
    const digits = text.replace(/^0+/, '') || '0';
    if (digits.length <= 2) out.push(NUMBER_BASE + Number(digits));
    else if (digits.length === 4 && Number(digits) >= IDS.year_range[0] && Number(digits) <= IDS.year_range[1]) out.push(YEAR);
    else out.push(NUM);
  }
  return out;
}

// -- vocabulary ------------------------------------------------------------------------------------------------

export class Vocab {
  constructor(stems, ids = 1) {
    this.layout = idLayout(ids);
    if (stems.length > this.layout.vocabMax) {
      throw new TokenizerError(`vocab has ${stems.length} stems (max ${this.layout.vocabMax} in id layout ${ids})`);
    }
    this.stems = [...stems];
    this.ids = new Map();
    stems.forEach((s, n) => {
      if (!s || s !== s.toLowerCase() || /\s/u.test(s)) {
        throw new TokenizerError(`vocab line ${n + 1}: ${JSON.stringify(s)} is empty, not lower-case or has white space`);
      }
      if (this.ids.has(s)) throw new TokenizerError(`vocab line ${n + 1}: ${JSON.stringify(s)} is a duplicate`);
      this.ids.set(s, this.layout.vocabBase + n);
    });
  }

  static fromText(text, ids = 1) {
    const lines = text.split('\n');
    if (lines.length && lines[lines.length - 1] === '') lines.pop();
    return new Vocab(lines, ids);
  }

  wordId(stem) {
    const id = this.ids.get(stem);
    return id !== undefined ? id : hashBucket(stem, this.layout.version);
  }
}

export const EMPTY_VOCAB = new Vocab([]);

// -- one node --------------------------------------------------------------------------------------------------

function emojiEnd(cps, i) {
  const n = cps.length;
  const first = cpOf(cps[i]);
  let j = i + 1;
  if (isRegional(first) && j < n && isRegional(cpOf(cps[j]))) return j + 1;
  while (j < n) {
    const cp = cpOf(cps[j]);
    if (cp === 0xfe0e || cp === 0xfe0f || cp === 0x20e3 || (cp >= 0x1f3fb && cp <= 0x1f3ff) || (cp >= 0xe0020 && cp <= 0xe007f)) j += 1;
    else if (cp === 0x200d && j + 1 < n && isPictograph(cps[j + 1])) j += 2;
    else break;
  }
  return j;
}

export function tokenizeText(text, vocab = EMPTY_VOCAB) {
  const cps = Array.from(nfkcProtected(text));
  const n = cps.length;
  const toks = [];
  const runs = [];
  let i = 0;
  while (i < n) {
    const ch = cps[i];
    if (MARKERS.has(ch)) {
      toks.push({ id: MARKERS.get(ch), kind: 'marker', surface: ch });
      i++;
      continue;
    }
    if (ch === '.' && cps[i + 1] === '.' && cps[i + 2] === '.') {
      let j = i;
      while (j < n && cps[j] === '.') j++;
      toks.push({ id: GLYPH_IDS.ELLIPSIS, kind: 'glyph', surface: cps.slice(i, j).join('') });
      i = j;
      continue;
    }
    if (GLYPH_OF_CHAR.has(ch)) {
      toks.push({ id: GLYPH_IDS[GLYPH_OF_CHAR.get(ch)], kind: 'glyph', surface: ch });
      i++;
      continue;
    }
    if (isPictograph(ch)) {
      const j = emojiEnd(cps, i);
      toks.push({ id: GLYPH_IDS.EMOJI, kind: 'glyph', surface: cps.slice(i, j).join('') });
      i = j;
      continue;
    }
    if (!isAssigned(ch)) {
      toks.push({ id: GLYPH_IDS.SYMBOL, kind: 'glyph', surface: ch });
      i++;
      continue;
    }
    if (RE_SPACE.test(ch) || RE_CF.test(ch) || RE_M.test(ch)) {
      i++;
      continue;
    }
    if (RE_L.test(ch)) {
      let j = i + 1;
      while (j < n) {
        const cj = cps[j];
        if (isAssigned(cj) && (RE_L.test(cj) || RE_M.test(cj) || RE_CF.test(cj))) j++;
        else if (APOSTROPHES.has(cj) && j + 1 < n && isLetter(cps[j + 1])) j++;
        else break;
      }
      const run = cps.slice(i, j).filter((c) => !RE_CF.test(c));
      i = j;
      runs.push(run);
      for (const part of camelSplit(run)) {
        const partStr = part.join('');
        const word = foldWord(partStr);
        if (!word) continue;
        const st = porterStem(word);
        toks.push({ id: vocab.wordId(st), kind: 'word', surface: partStr, word, stem: st });
      }
      continue;
    }
    if (ch >= '0' && ch <= '9') {
      const rest = cps.slice(i).join('');
      NUMBER.lastIndex = 0;
      const m = NUMBER.exec(rest);
      for (const tid of numberTokens(m[0])) toks.push({ id: tid, kind: 'number', surface: m[0] });
      i += m[0].length; // the match is ASCII, so its length in code units equals its length in code points
      continue;
    }
    if (ch === '_') {
      i++;
      continue;
    }
    if (PUNCT_IDS.has(ch)) {
      toks.push({ id: PUNCT_IDS.get(ch), kind: 'punct', surface: ch });
      i++;
      continue;
    }
    if (ch === HYPHEN) {
      toks.push({ id: PUNCT_IDS.get('-'), kind: 'punct', surface: ch });
      i++;
      continue;
    }
    toks.push({ id: GLYPH_IDS.SYMBOL, kind: 'glyph', surface: ch });
    i++;
  }

  const flags = [];
  let nUpper = 0;
  let nLower = 0;
  for (const r of runs) for (const ch of r) { if (isUpper(ch)) nUpper++; else if (isLower(ch)) nLower++; }
  const caps = nUpper >= 2 && nLower === 0;
  if (caps) flags.push('caps');
  if (!caps && runs.length >= 2 && runs.every((r) => isUpper(r[0]))) flags.push('title_case');
  const dropped = Math.max(0, toks.length - MAX_NODE_TOKENS);
  let kept = toks;
  if (dropped) {
    kept = toks.slice(0, MAX_NODE_TOKENS);
    flags.push('truncated');
  }
  return { tokens: kept, ids: kept.map((t) => t.id), flags, dropped };
}

// -- screens and windows ---------------------------------------------------------------------------------------

export function encodeNode(index, node, vocab = EMPTY_VOCAB) {
  const role = node.role;
  if (!ROLE_IDS.has(role)) throw new TokenizerError(`node ${index}: unknown role ${JSON.stringify(role)}`);
  const given = node.flags || [];
  for (const f of given) if (!FLAG_IDS.has(f)) throw new TokenizerError(`node ${index}: unknown flag ${JSON.stringify(f)}`);
  if (typeof node.text !== 'string') throw new TokenizerError(`node ${index}: text must be a string`);
  const tt = tokenizeText(node.text, vocab);
  const names = new Set([...given, ...tt.flags]);
  const flags = FLAGS.filter((f) => names.has(f));
  let mask = 0;
  for (const f of flags) mask |= 1 << FLAG_IDS.get(f);
  const k = node.k;
  if (!Number.isInteger(k) || k < 0) throw new TokenizerError(`node ${index}: cluster index k must be a non-negative integer`);
  return { index, i: node.i ?? index, k, role, roleId: ROLE_IDS.get(role), flags, flagMask: mask >>> 0, text: tt };
}

// Node list indices per window (SPEC.md section 4, Windows).
export function splitWindows(nodes) {
  const runs = [];
  for (const nd of nodes) {
    const last = runs[runs.length - 1];
    if (last && nodes[last[last.length - 1]].k === nd.k) last.push(nd.index);
    else runs.push([nd.index]);
  }
  const cost = (idx) => idx.reduce((s, x) => s + nodes[x].text.tokens.length + 1, 0);
  const windows = [];
  let cur = [];
  for (const run of runs) {
    if (cur.length + run.length <= MAX_WINDOW_NODES && 1 + cost(cur) + cost(run) <= MAX_WINDOW_POSITIONS) {
      cur.push(...run);
      continue;
    }
    if (cur.length) {
      windows.push(cur);
      cur = [];
    }
    if (run.length <= MAX_WINDOW_NODES && 1 + cost(run) <= MAX_WINDOW_POSITIONS) {
      cur = [...run];
      continue;
    }
    for (const x of run) {
      if (cur.length + 1 > MAX_WINDOW_NODES || 1 + cost(cur) + cost([x]) > MAX_WINDOW_POSITIONS) {
        windows.push(cur);
        cur = [];
      }
      cur.push(x);
    }
  }
  if (cur.length) windows.push(cur);
  return windows;
}

export function buildWindow(nodes, idx) {
  const w = { tokens: [BOS], offsets: [0], roles: [-1], clusters: [-1], flags: [0], node: [0], nodes: [], eon: [] };
  idx.forEach((x, n) => {
    const ne = nodes[x];
    const ids = [...ne.text.ids, EON];
    ids.forEach((tid, off) => {
      w.tokens.push(tid);
      w.offsets.push(off);
      w.roles.push(ne.roleId);
      w.clusters.push(ne.k);
      w.flags.push(ne.flagMask);
      w.node.push(n + 1);
    });
    w.nodes.push(x);
    w.eon.push(w.tokens.length - 1);
  });
  return w;
}

export function encodeScreen(screen, vocab = EMPTY_VOCAB) {
  if (screen.schema !== 'slop.screen/1') throw new TokenizerError(`not a slop.screen/1 document (schema ${JSON.stringify(screen.schema)})`);
  if (!Array.isArray(screen.nodes)) throw new TokenizerError('screen has no node list');
  const nodes = screen.nodes.map((nd, x) => encodeNode(x, nd, vocab));
  const windows = splitWindows(nodes).map((idx) => buildWindow(nodes, idx));
  return { id: screen.id ?? null, nodes, windows };
}
