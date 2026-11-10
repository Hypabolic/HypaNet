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
// VAR marker are the identifier's words (SPEC.md §4), not copy: up to four of them, up to the next punctuation, carry
// `ident: true`. The extractor writes no boundary after them, so this is a cap, not an exact rule.
export function tokens(text) {
  const out = [];
  for (const m of text.normalize('NFKC').matchAll(TOKEN)) {
    if (m[1]) out.push({ k: 'var', s: m[1] });
    else if (m[2]) out.push({ k: 'alt', s: m[2] });
    else if (m[3]) out.push({ k: 'word', s: m[3], w: foldWord(m[3]) });
    else if (m[4]) out.push({ k: 'number', s: m[4], w: m[4] });
    else if (m[6]) out.push({ k: 'other', s: m[6] });
  }
  for (let x = 0; x < out.length; x++) {
    if (out[x].k !== 'var') continue;
    for (let y = x + 1, n = 0; y < out.length && out[y].k === 'word' && n < 4; y++, n++) out[y].ident = true;
  }
  return out;
}

// Word and number tokens that are copy (not identifier words), as folded strings.
export function literalWords(text) {
  return tokens(text).filter((t) => (t.k === 'word' || t.k === 'number') && !t.ident).map((t) => t.w);
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

export function clip(text, n = 48) {
  const t = text.replace(//g, '{…}').replace(//g, ' | ').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
}

// An arrow suffix: the extractor's `arrow_tail` flag (an arrow icon or glyph after the text), or an arrow glyph or
// "->" ending the text. The glyphs are the tokeniser's ARROW class (→ ↗ », SPEC.md §6) plus › ➔ ➜ ⟶ ↘.
const ARROW_TAIL = /(?:[→↗»›➔➜⟶↘]|->)\s*$/u;
export function hasArrowTail(node) {
  return (node.flags || []).includes('arrow_tail') || ARROW_TAIL.test(node.text);
}

// -- the index-number device (slop/index-numbering, R12) ------------------------------------------------------------
//
// Families: `padded` ("01", "02 /", "(001)", "#03", "01 / 04", and as a lead "01 / Connect", "01. Connect"),
// `fraction` ("3 / 6" as the whole node), `label:<word>` ("Step 2", "Phase 02", "Stage 1 of 3", and as a lead
// "Step 1: Create an account") and `plain` (a bare "1", "2." or "3)" as the whole node). Sources:
// anti-slop-guides.md:416 (R12: "01 /", "(001)", "Step 2", "Phase 02", "01 / 4"); local-skills.md:239 (gpt-taste:63,
// "SECTION 01", "QUESTION 05"); local-skills.md:138 (v2-exp:686, "Stage 1 / Stage 2 / Stage 3").
// Label words left out on purpose: "level", "round", "lesson", "module", "chapter", "part" usually name content (a game
// level, a course unit, a book chapter), which is a fact, not an index.
const LABEL_WORDS = 'step|phase|stage|section|question|no\\.?|nº';
const SEP = '[\\/.:·—–\\-]';
const RE_PADDED = new RegExp(`^[(\\[#]?\\s*(0\\d{1,2})\\s*[)\\]]?\\s*(?:${SEP}\\s*(\\d{1,3})?)?\\s*$`, 'u');
const RE_FRACTION = /^(\d{1,2})\s*\/\s*(\d{1,2})$/u;
const RE_LABEL = new RegExp(`^(${LABEL_WORDS})\\s*(\\d{1,3})(?:\\s*(?:of|\\/)\\s*(\\d{1,3}))?\\s*${SEP}?\\s*$`, 'iu');
const RE_PLAIN = /^(\d{1,2})\s*[.)/]?$/u;
const RE_PADDED_LEAD = new RegExp(`^[(\\[#]?(0\\d{1,2})[)\\]]?\\s*(?:${SEP}\\s*|\\s+)(?=\\p{L})`, 'u');
const RE_LABEL_LEAD = new RegExp(`^(${LABEL_WORDS})\\s*(\\d{1,3})(?:\\s*(?:of|\\/)\\s*\\d{1,3})?\\s*${SEP}\\s*(?=\\p{L})`, 'iu');

const labelWord = (w) => w.toLowerCase().replace(/\.$/, '').replace('nº', 'no');

// {family, value, whole, word?} or null.
export function parseIndex(text) {
  const t = text.normalize('NFKC').trim();
  let m;
  if ((m = RE_PADDED.exec(t))) return { family: 'padded', value: Number(m[1]), whole: true };
  if ((m = RE_FRACTION.exec(t))) return { family: 'fraction', value: Number(m[1]), whole: true };
  if ((m = RE_LABEL.exec(t))) return { family: `label:${labelWord(m[1])}`, value: Number(m[2]), whole: true, word: labelWord(m[1]) };
  if ((m = RE_PLAIN.exec(t))) return { family: 'plain', value: Number(m[1]), whole: true };
  if ((m = RE_PADDED_LEAD.exec(t))) return { family: 'padded', value: Number(m[1]), whole: false };
  if ((m = RE_LABEL_LEAD.exec(t))) return { family: `label:${labelWord(m[1])}`, value: Number(m[2]), whole: false, word: labelWord(m[1]) };
  return null;
}
