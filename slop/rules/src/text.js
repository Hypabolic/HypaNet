// Text helpers for the rule layer: a small word splitter, the normalised string, branches, arrow tails and the
// index-number device. Self-contained (the tokeniser in lint/ truncates a node at 31 tokens, and the rules need every
// word), but it folds words the same way: NFKC, lower case, marks dropped, inner apostrophes removed.

export const VAR_MARK = '';
export const ALT_MARK = '';

const TOKEN = /()|()|([\p{L}\p{M}]+(?:['’][\p{L}\p{M}]+)*)|(\p{Nd}+(?:[.,]\p{Nd}+)*)|(\s+)|(.)/gsu;

export function foldWord(s) {
  return s.toLowerCase().normalize('NFKD').replace(/\p{M}/gu, '').replace(/['’]/g, '');
}

// Tokens of one string: {k: 'var'|'alt'|'word'|'number'|'other', s: surface, w: folded word}. Words that follow a
// VAR marker are the identifier's words (SPEC.md §4), not copy, and carry `ident: true`. `vw` is the node's `vw` field
// (one count per VAR marker in `text`, in order): the exact number of identifier words (and numbers) after each marker.
// Without it (a screen from an older extractor, or a count that does not match the markers), up to four words after a
// marker, up to the next number or punctuation, are taken as identifier words: a cap, not an exact rule.
const FALLBACK_IDENT_WORDS = 4;
export function tokens(text, vw) {
  const out = [];
  for (const m of text.normalize('NFKC').matchAll(TOKEN)) {
    if (m[1]) out.push({ k: 'var', s: m[1] });
    else if (m[2]) out.push({ k: 'alt', s: m[2] });
    else if (m[3]) out.push({ k: 'word', s: m[3], w: foldWord(m[3]) });
    else if (m[4]) out.push({ k: 'number', s: m[4], w: m[4] });
    else if (m[6]) out.push({ k: 'other', s: m[6] });
  }
  const vars = out.filter((t) => t.k === 'var').length;
  const exact = Array.isArray(vw) && vw.length === vars && vw.every((c) => Number.isInteger(c) && c >= 0);
  let v = 0;
  for (let x = 0; x < out.length; x++) {
    if (out[x].k !== 'var') continue;
    const max = exact ? vw[v++] : FALLBACK_IDENT_WORDS;
    const isIdent = exact ? (t) => t.k === 'word' || t.k === 'number' : (t) => t.k === 'word';
    for (let y = x + 1, n = 0; y < out.length && isIdent(out[y]) && n < max; y++, n++) out[y].ident = true;
  }
  return out;
}

// Word and number tokens that are copy (not identifier words), as folded strings. `vw` as for tokens().
export function literalWords(text, vw) {
  return tokens(text, vw).filter((t) => (t.k === 'word' || t.k === 'number') && !t.ident).map((t) => t.w);
}

// The normalised string used by slop/exact-repeat: folded words and numbers, markers kept, glyphs and punctuation
// dropped, single spaces.
export function normalised(text) {
  return tokens(text)
    .filter((t) => t.k !== 'other')
    .map((t) => (t.k === 'var' ? VAR_MARK : t.k === 'alt' ? ALT_MARK : t.w))
    .join(' ');
}

// The string branches of a conditional (joined by ALT, SPEC.md §4), each trimmed.
export function branches(text) {
  return text.split(ALT_MARK).map((b) => b.trim()).filter((b) => b.length > 0);
}

// The branches of a node as {text, vw}, each branch carrying the slice of the node's `vw` for its own VAR markers
// (undefined when the node has no `vw`).
export function nodeBranches(node) {
  const vw = Array.isArray(node.vw) ? node.vw : null;
  const out = [];
  let v = 0;
  for (const part of node.text.split(ALT_MARK)) {
    const marks = part.split(VAR_MARK).length - 1;
    const text = part.trim();
    if (text.length) out.push({ text, vw: vw ? vw.slice(v, v + marks) : undefined });
    v += marks;
  }
  return out;
}

export function clip(text, n = 48) {
  const t = text.replace(//g, '{…}').replace(//g, ' | ').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
}

// An arrow suffix: a text ending in → ➔ ➜ ➝ ➞ ➡ ⟶ ⟹ ⇒ ⇢ ⇾ ⭢ ⮕ 🡒 (an emoji variation selector allowed after it, "➡️"),
// "->" or "=>", or in › as a separate final token ("Next ›"); otherwise the extractor's `arrow_tail` flag (an arrow
// icon after the text). A text ending in ↗ (an external-link marker, which tells the reader something), ↘, » or a ›
// glued to the last word is not an arrow suffix, even when the extractor flagged it.
const ARROW_TAIL = /(?:[→➔➜➝➞➡⟶⟹⇒⇢⇾⭢⮕🡒]\uFE0F?|->|=>|\s›)\s*$/u;
const NOT_ARROW_TAIL = /[↗↘»›]\s*$/u;
export function hasArrowTail(node) {
  if (ARROW_TAIL.test(node.text)) return true;
  if (NOT_ARROW_TAIL.test(node.text)) return false;
  return (node.flags || []).includes('arrow_tail');
}

// -- the index-number device (slop/index-numbering, R12) ------------------------------------------------------------
//
// Families: `padded` ("01", "02 /", "(001)", "#03", "01 / 04", and as a lead "01 / Connect", "01. Connect",
// "01 / 03 Connect"; a two-digit 10–99 in those shapes is `wide`, which only continues a padded column),
// `fraction` ("3 / 6" as the whole node; a second number after a padded one needs a slash, so "08:00" and "01.03"
// are not devices), `label:<word>` ("Step 2", "Phase 02", "Stage 1 of 3", and as a lead
// "Step 1: Create an account", "Step 2 of 3: Safety talk") and `plain` (a bare "1", "2." or "3)" as the whole node).
// Sources: anti-slop-guides.md:416 (R12: "01 /", "(001)", "Step 2", "Phase 02", "01 / 4"); local-skills.md:239
// (gpt-taste:63, "SECTION 01", "QUESTION 05"); local-skills.md:138 (v2-exp:686, "Stage 1 / Stage 2 / Stage 3").
// Label words left out on purpose: "level", "round", "lesson", "module", "chapter", "part" usually name content (a game
// level, a course unit, a book chapter), which is a fact, not an index.
const LABEL_WORDS = 'step|phase|stage|section|question|no\\.?|nº';
const SEP = '[\\/.:·—–\\-]';
const RE_PADDED = new RegExp(`^[(\\[#]?\\s*(0\\d{1,2})\\s*[)\\]]?\\s*(?:\\/\\s*(\\d{1,3})|${SEP})?\\s*$`, 'u');
const RE_FRACTION = /^(\d{1,2})\s*\/\s*(\d{1,2})$/u;
const RE_LABEL = new RegExp(`^(${LABEL_WORDS})\\s*(\\d{1,3})(?:\\s*(?:of|\\/)\\s*(\\d{1,3}))?\\s*${SEP}?\\s*$`, 'iu');
const RE_PLAIN = /^(\d{1,2})\s*[.)/]?$/u;
const RE_PADDED_LEAD = new RegExp(`^[(\\[#]?(0\\d{1,2})[)\\]]?\\s*(?:\\/\\s*(\\d{1,3})\\s*(?:${SEP}\\s*|\\s+)|${SEP}\\s*|\\s+)(?=\\p{L})`, 'u');
const RE_WIDE = new RegExp(`^(\\d{2})(?:\\s*${SEP}?\\s*$|\\s*(?:${SEP}\\s*|\\s+)(?=(\\p{L})))`, 'u');
const RE_LABEL_LEAD = new RegExp(`^(${LABEL_WORDS})\\s*(\\d{1,3})(?:\\s*(?:of|\\/)\\s*(\\d{1,3}))?\\s*${SEP}\\s*(?=\\p{L})`, 'iu');

const labelWord = (w) => w.toLowerCase().replace(/\.$/, '').replace('nº', 'no');

// {family, value, whole, word?, total?} or null. `total` is the second number of "01 / 04", "3 / 6" or "Step 2 of 4" (null
// when there is none); a total smaller than the value ("05 / 03", "3/2") is not an index device.
export function parseIndex(text) {
  const t = text.normalize('NFKC').trim();
  let m;
  if ((m = RE_PADDED.exec(t)) && (!m[2] || Number(m[2]) >= Number(m[1]))) return { family: 'padded', value: Number(m[1]), whole: true, total: m[2] ? Number(m[2]) : null };
  if ((m = RE_FRACTION.exec(t)) && Number(m[2]) >= Number(m[1])) return { family: 'fraction', value: Number(m[1]), whole: true, total: Number(m[2]) };
  if ((m = RE_LABEL.exec(t))) return { family: `label:${labelWord(m[1])}`, value: Number(m[2]), whole: true, word: labelWord(m[1]), total: m[3] ? Number(m[3]) : null };
  if ((m = RE_PLAIN.exec(t))) return { family: 'plain', value: Number(m[1]), whole: true };
  if ((m = RE_PADDED_LEAD.exec(t)) && (!m[2] || Number(m[2]) >= Number(m[1]))) return { family: 'padded', value: Number(m[1]), whole: false, total: m[2] ? Number(m[2]) : null };
  if ((m = RE_LABEL_LEAD.exec(t))) return { family: `label:${labelWord(m[1])}`, value: Number(m[2]), whole: false, word: labelWord(m[1]), total: m[3] ? Number(m[3]) : null };
  // A two-digit number 10–99 in the same shape as a padded device continues a padded column ("09", "10"): `wide`.
  if ((m = RE_WIDE.exec(t)) && Number(m[1]) >= 10) return { family: 'padded', value: Number(m[1]), whole: !m[2], total: null, wide: true };
  return null;
}
