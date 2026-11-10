// Codex lists and the pattern matcher for the codex rules (slop/codex/llm.md, ui.md). The lists come from
// codex-lists.js, generated from slop/codex/lists/*.txt by rules/make_codex_lists.mjs; each rule picks its sections
// and names the entries it leaves out, with a reason, in its own module.
//
// Pattern syntax (slop/codex/lists/llm_constructions.txt): words are folded like every other rule (src/text.js:
// NFKC, lower case, marks and inner apostrophes dropped, so "it's" is "its"); {x} {y} {z} are one to eight words
// (commas allowed inside, never a sentence end); {w} is exactly one word; {n} is one literal number (an interpolated
// value never counts), optionally followed by k, m, thousand or million; ^ anchors at the start of the text, $ at the
// end (trailing punctuation allowed). Unlike the normalised string, the token stream keeps the punctuation that
// shapes a construction (, . ? ! : ; % + /), so "not just {x}, it's {y}" needs the comma and "^tired of {x}?" the
// question mark. The last word of a pattern also matches the first part of a contraction ("sure! here" matches
// "Sure! Here's"). An em or en dash reads as a comma ("Not just a list — it's a plan"), "…" as a full stop, "&" as
// "and" and "×" as "x". Words after a VAR marker are the identifier's and become one `var` token, which a slot can
// cover but a literal word never matches.

import { CODEX_LISTS } from './codex-lists.js';
import { tokens } from './text.js';

export { CODEX_LISTS };

// Every entry of the named sections of one list, as {entry, id} in file order. `ids` null: every section.
export function sectionEntries(list, ids = null) {
  const l = CODEX_LISTS[list];
  if (!l) throw new Error(`no codex list ${list}`);
  const out = [];
  for (const s of l.sections) if (!ids || ids.includes(s.id)) for (const entry of s.entries) out.push({ entry, id: s.id });
  return out;
}

// Entries of sections minus the excluded ones. Throws if an exclusion names no entry of those sections, so a list
// edit that removes or rewords an entry shows up as a failing build instead of a silent no-op.
export function pick(list, ids, exclude = {}) {
  const all = sectionEntries(list, ids);
  const have = new Set(all.map((e) => e.entry));
  for (const e of Object.keys(exclude)) if (!have.has(e)) throw new Error(`${list}: excluded entry "${e}" is not in sections ${ids}`);
  return all.filter((e) => !(e.entry in exclude));
}

const PUNCT = new Map([[',', ','], ['.', '.'], ['?', '?'], ['!', '!'], [':', ':'], [';', ';'], ['%', '%'], ['+', '+'], ['/', '/'],
  ['—', ','], ['–', ','], ['…', '.'], ['¿', null], ['¡', null]]);
export const SENTENCE_END = new Set(['.', '?', '!']);

// Token stream of one text: {k: 'w'|'n'|'p'|'var', w}. `vw` as for tokens() in text.js.
export function matchTokens(text, vw) {
  const out = [];
  for (const t of tokens(text, vw)) {
    if (t.ident) continue;
    if (t.k === 'var') out.push({ k: 'var', w: '' });
    else if (t.k === 'word') {
      // A contraction keeps its first part too ("here's" is "heres", base "here"), so an entry that ends in "here"
      // matches "Here's".
      const ap = t.s.search(/['’]/u);
      out.push(ap > 0 ? { k: 'w', w: t.w, base: t.w.slice(0, t.s.slice(0, ap).normalize('NFKD').replace(/\p{M}/gu, '').length) } : { k: 'w', w: t.w });
    }
    else if (t.k === 'number') out.push({ k: 'n', w: t.w });
    else if (t.k === 'other') {
      if (t.s === '&') { out.push({ k: 'w', w: 'and' }); continue; }
      if (t.s === '×') { out.push({ k: 'w', w: 'x' }); continue; }
      const p = PUNCT.get(t.s);
      if (!p) continue;
      const last = out[out.length - 1];
      if (last && last.k === 'p' && last.w === p) continue; // "..." is one full stop, "!!" one mark
      out.push({ k: 'p', w: p });
    }
  }
  return out;
}

const SLOT = /\{([xyzwn])\}/g;
const N_SUFFIX = new Set(['k', 'm', 'thousand', 'million']);

// Compile one entry to {src, start, end, toks: [{t: 'lit'|'num'|'p'|'slot', w, s}]}.
export function compile(entry) {
  let src = entry.trim();
  const start = src.startsWith('^');
  if (start) src = src.slice(1);
  const end = src.endsWith('$');
  if (end) src = src.slice(0, -1);
  const toks = [];
  let last = 0;
  const lit = (s) => {
    for (const t of matchTokens(s)) {
      if (t.k === 'w') toks.push({ t: 'lit', w: t.w });
      else if (t.k === 'n') toks.push({ t: 'num', w: t.w });
      else if (t.k === 'p') toks.push({ t: 'p', w: t.w });
    }
  };
  for (const m of src.matchAll(SLOT)) {
    lit(src.slice(last, m.index));
    toks.push({ t: 'slot', s: m[1] });
    last = m.index + m[0].length;
  }
  lit(src.slice(last));
  if (!toks.length) throw new Error(`empty pattern "${entry}"`);
  return { src: entry, start, end, toks };
}

// The first match of a compiled pattern in a token stream: {from, to, slots: {name: [token]}} or null. Slots are lazy.
export function match(pat, ts) {
  const firstWord = ts.findIndex((t) => t.k !== 'p');
  const starts = pat.start ? (firstWord < 0 ? [] : [firstWord]) : ts.map((_, x) => x);
  for (const from of starts) {
    const slots = {};
    const to = step(pat, ts, 0, from, slots);
    if (to !== null) return { from, to, slots };
  }
  return null;
}

function step(pat, ts, p, x, slots) {
  if (p === pat.toks.length) {
    if (pat.end && ts.slice(x).some((t) => t.k !== 'p')) return null;
    return x;
  }
  const tok = pat.toks[p];
  const cur = ts[x];
  if (tok.t === 'lit') {
    const last = p === pat.toks.length - 1;
    return cur && cur.k === 'w' && (cur.w === tok.w || (last && cur.base === tok.w)) ? step(pat, ts, p + 1, x + 1, slots) : null;
  }
  if (tok.t === 'num') return cur && cur.k === 'n' && cur.w === tok.w ? step(pat, ts, p + 1, x + 1, slots) : null;
  if (tok.t === 'p') return cur && cur.k === 'p' && cur.w === tok.w ? step(pat, ts, p + 1, x + 1, slots) : null;
  if (tok.s === 'w') {
    if (!cur || cur.k !== 'w') return null;
    slots.w = [cur];
    return step(pat, ts, p + 1, x + 1, slots);
  }
  if (tok.s === 'n') {
    if (!cur || cur.k !== 'n') return null;
    slots.n = [cur];
    const plain = step(pat, ts, p + 1, x + 1, slots);
    if (plain !== null) return plain;
    const nx = ts[x + 1];
    if (nx && nx.k === 'w' && N_SUFFIX.has(nx.w)) { slots.n = [cur, nx]; return step(pat, ts, p + 1, x + 2, slots); }
    return null;
  }
  // {x} {y} {z}: one to eight words, numbers or interpolations, commas allowed after the first.
  let words = 0;
  for (let y = x; y < ts.length; y++) {
    const t = ts[y];
    if (t.k === 'p') {
      if (t.w !== ',' || y === x) break;
    } else if (++words > 8) break;
    if (t.k === 'p') continue;
    slots[tok.s] = ts.slice(x, y + 1);
    const r = step(pat, ts, p + 1, y + 1, slots);
    if (r !== null) return r;
  }
  delete slots[tok.s];
  return null;
}

// Words of a slot (or of all slots), for guards.
export const slotWords = (m) => Object.values(m.slots).flat().filter((t) => t.k !== 'p');

// Role groups used by the codex rules.
export const TITLE_ROLES = new Set(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'subtitle', 'eyebrow', 'empty_title', 'dialog_title', 'badge']);
// Controls, form parts, values and navigation: a construction there is an option, a value or a label, not prose.
export const NOT_PROSE = new Set(['button', 'link', 'nav', 'tab', 'menu_item', 'breadcrumb', 'label', 'placeholder', 'input_value', 'cell', 'stat']);

export const compileAll = (entries) => entries.map((e) => ({ ...e, pat: compile(e.entry) }));

// Every compiled entry found in a token stream (its first match that `accept` takes), in list order, dropping a hit
// whose span lies inside a longer hit ("unlock" inside "unlock the power of").
export function hits(list, ts, accept = () => true) {
  const found = [];
  for (const e of list) {
    const m = match(e.pat, ts);
    if (m && accept(e, m)) found.push({ ...e, m });
  }
  const len = (h) => h.m.to - h.m.from;
  // Inside a longer hit, or the same span as an earlier one ("(555) 123-4567" and "555-123-4567").
  return found.filter((h, x) => !found.some((o, y) => o !== h && o.m.from <= h.m.from && o.m.to >= h.m.to && (len(o) > len(h) || (len(o) === len(h) && y < x))));
}

// The node's copy as one folded string per branch (for whole-label lists).
export const label = (ts) => ts.filter((t) => t.k !== 'p').map((t) => (t.k === 'var' ? '\u0000' : t.w)).join(' ');
