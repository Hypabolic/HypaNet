// Framework-neutral intermediate representation. Each front end (jsx.ts, html.ts, swift.ts) turns a source file
// into a tree of El and Txt; infer.ts turns that tree into screen nodes. Offsets are UTF-16 indices into the source.

import type { Role } from './schema.ts';

/** Container kinds an element can open. They drive clusters and the in_* flags. */
export type Box =
  | 'root' | 'card' | 'dialog' | 'state' | 'empty' | 'toast' | 'alert'
  | 'list' | 'nav' | 'tablist' | 'menu' | 'breadcrumb' | 'table'
  | 'form' | 'header' | 'footer' | 'section' | 'control';

/** Style evidence an element carries itself (inline style, SwiftUI modifiers); class tokens are read in style.ts. */
export interface Sty {
  caps?: boolean;
  tracked?: boolean;
  small?: boolean; // small text (xs/sm, caption fonts)
  xsmall?: boolean; // the smallest sizes only (text-xs, .caption): a class-stage role signal
  large?: boolean; // display sizes (stat detection)
  mono?: boolean;
  muted?: boolean;
  rounded?: boolean; // fully rounded (pill or dot shape)
  dotSize?: boolean; // tiny width and height
  truncated?: boolean;
  srOnly?: boolean;
  errorColour?: boolean;
  hue?: boolean; // a chromatic text colour (status colours: green, amber, red, sky ...), not a grey or the foreground
  blockDisplay?: boolean; // display: block, flex or grid on an inline tag (it starts its own line)
  flexBox?: boolean; // display: flex or grid (inline or not): each child is laid out as its own item
  titleCase?: boolean;
  heading?: number; // SwiftUI title fonts: a class-stage heading level
  bold?: boolean; // font weight 600 or more (Tailwind font-semibold and bolder, CSS font-weight)
  fontPx?: number; // font size in px when known (Tailwind text-* sizes, a literal CSS font-size)
}

export interface El {
  t: 'el';
  id: number;
  name: string; // lower-case HTML tag for intrinsic elements; component or SwiftUI view name otherwise
  intrinsic: boolean;
  cls: string[]; // static class tokens
  bound: boolean; // class, style or colour bound to an expression (stateful evidence)
  attrs: Record<string, string>; // static attribute values the rules read (role, type, as, variant, aria-level, ...)
  sty: Sty;
  kids: Kid[];
  s: number;
  e: number;
  cond?: boolean; // rendered under a condition
  tmpl?: boolean; // root of a .map() / ForEach template, or of one expanded copy of it
  hint?: Role; // role assigned by the front end from tag or attribute semantics (strong)
  classHint?: Role; // role assigned by the front end at the class stage (SwiftUI fonts)
  box?: Box;
  icon?: boolean;
  arrowIcon?: boolean;
  dot?: boolean;
  control?: boolean; // a form control (input, Toggle, ...)
  textBlock?: boolean; // inline children always merge into one node
  inline?: boolean; // may merge into a surrounding run of text
  noMerge?: boolean; // never merges (pill, kicker, sr-only)
  andGroup?: number; // shown under its own `&&` condition (jsx.ts): its text belongs to that group
  hidden?: boolean; // not drawn (SwiftUI .opacity(0), .hidden()): no nodes
  // filled in by infer.ts
  parent?: El;
}

export interface Txt {
  t: 'txt';
  text: string; // visible text; interpolations already rewritten as VAR_MARK plus words, branches joined with ALT_MARK
  lit: string; // the literal part only (no VAR words, branches split by ALT_MARK), for caps and title_case
  s: number; // span of the visible text (trimmed)
  e: number;
  dyn?: boolean; // contains an interpolation
  alt?: boolean; // chosen by a conditional
  cond?: boolean;
  attr?: Role; // a standalone node from an attribute or modifier (placeholder, title, aria-label, .help())
  bound?: boolean; // the whole text is one expression (text bound to data)
  subs?: [number, number][]; // spans of the array-literal strings substituted into the text (jsx.ts), in text order
  near?: [number, number][]; // the spans of those strings' properties (name: "…")
  copy?: [number, number]; // span of the array element whose expanded copy holds this text
  call?: [number, number]; // span of the call of an inlined local view function (swift.ts): a last-resort loc
  andGroup?: number; // shown under its own `&&` condition (jsx.ts): adjacent texts of different groups are alternatives
  opts?: Opt[]; // the complete renderings of a conditional text, in branch order, '' for an empty branch (≥ 2)
  lspan?: [number, number]; // span of the one literal an in-file constant resolved to (jsx.ts, swift.ts): its loc
}

/** One complete rendering of a conditional text. */
export interface Opt {
  text: string;
  lit: string;
}

/**
 * At most this many complete renderings are written for a text with conditional parts (an optional fragment, a
 * plural ending, a fallback inside a template); beyond it the branches stay inline, joined where they occur.
 */
export const MAX_RENDERINGS = 8;

/**
 * The complete renderings of a sequence of parts, each a list of alternatives: the product in order (the first part
 * varies slowest), or null when there would be more than MAX_RENDERINGS.
 */
export function renderings(parts: Opt[][]): Opt[] | null {
  let out: Opt[] = [{ text: '', lit: '' }];
  for (const p of parts) {
    if (out.length * p.length > MAX_RENDERINGS) return null;
    const next: Opt[] = [];
    for (const a of out) for (const b of p) next.push({ text: a.text + b.text, lit: a.lit + b.lit });
    out = next;
  }
  return out;
}

/** Join complete renderings with ALT: empty and repeated renderings are dropped. */
export function joinRenderings(rs: Opt[]): Opt {
  const seen = new Set<string>();
  const keep: Opt[] = [];
  for (const r of rs) {
    const k = collapse(r.text.replace(/\uE002/gu, ''));
    if (!k || seen.has(k)) continue;
    seen.add(k);
    keep.push(r);
  }
  return { text: keep.map((r) => r.text.trim()).join(ALT_JOIN), lit: keep.map((r) => r.lit.trim()).join(ALT_SEP) };
}

/**
 * Private-use characters in source copy (Material Icons ligature code points, say) are dropped before any marker is
 * written, so that only the extractor's own VAR and ALT code points are markers.
 */
export function stripPUA(s: string): string {
  return s.replace(/[\uE000-\uF8FF\u{F0000}-\u{FFFFD}\u{100000}-\u{10FFFD}]/gu, '');
}

export type Kid = El | Txt;

// Literal arrays: a .map() or ForEach over an array literal in the same file is expanded once per element (jsx.ts,
// swift.ts), at most MAX_EXPAND elements.
export const MAX_EXPAND = 24;

/** A literal value from an array literal. */
export type LV =
  | { k: 'str'; value: string; s: number; e: number; p?: [number, number] } // s, e: inside the quotes; p: property or element
  | { k: 'num'; value: number; raw: string; s: number; e: number; p?: [number, number] }
  | { k: 'bool'; value: boolean }
  | { k: 'nil' }
  | { k: 'obj'; props: Map<string, Slot>; open: boolean; at?: [number, number] } // open: a spread or computed key may add props
  | { k: 'arr'; elems: LV[] };
/** A value known statically (LV), present but not a literal (null), or absent (undefined). */
export type Slot = LV | null | undefined;

/** Does any text in the subtree show a substituted array literal? */
export function hasSubs(k: Kid): boolean {
  return k.t === 'txt' ? !!k.subs : k.kids.some(hasSubs);
}

/** Give every text from an inlined function call the call's span, as a fallback location. */
export function markCall(k: Kid, at: [number, number]): void {
  if (k.t === 'txt') {
    if (!k.call) k.call = at;
  } else for (const c of k.kids) markCall(c, at);
}

/** Give every text in an expanded copy its element's span, unless a nested copy already gave it one. */
export function markCopy(k: Kid, at: [number, number]): void {
  if (k.t === 'txt') {
    if (!k.copy) k.copy = at;
  } else for (const c of k.kids) markCopy(c, at);
}

let nextId = 1;
export function resetIds(): void {
  nextId = 1;
}

export function el(name: string, s: number, e: number, opts: Partial<El> = {}): El {
  return {
    t: 'el', id: nextId++, name, intrinsic: false, cls: [], bound: false, attrs: {}, sty: {}, kids: [], s, e, ...opts,
  };
}

export function txt(text: string, s: number, e: number, opts: Partial<Txt> = {}): Txt {
  return { t: 'txt', text, lit: opts.lit ?? text, s, e, ...opts };
}

// ---------------------------------------------------------------------------------------------------------------
// Text helpers

/** Split an identifier into lower-case words: firstName → first name, HTTPServer → http server, a_b → a b. */
export function identWords(id: string): string {
  const cleaned = id.replace(/^[$_#@]+/, '').replace(/[$]/g, '');
  const parts = cleaned
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .replace(/([A-Za-z])([0-9])/g, '$1 $2')
    .replace(/([0-9])([A-Za-z])/g, '$1 $2')
    .split(/[^A-Za-z0-9]+/)
    .filter(Boolean)
    .map((w) => w.toLowerCase());
  return parts.join(' ');
}

/**
 * Markers written into node text: private-use code points, so that a literal "VAR" or "ALT" in copy (a key cap,
 * say) stays an ordinary word. The tokeniser maps exactly these two code points to its VAR and ALT ids.
 */
export const VAR_MARK = '\uE000';
export const ALT_MARK = '\uE001';
/**
 * Internal only, never written to a screen: closes the identifier words after a VAR marker, so that infer.ts can
 * count them for the node's `vw` field after any amount of concatenation, then strips it.
 */
export const VAR_END = '\uE002';
/** Branch separator in text: the ALT marker between spaces. */
export const ALT_JOIN = ` ${ALT_MARK} `;

/** The text of an interpolation: the VAR marker plus the words of the last identifier segment. */
export function varText(ident: string | null): string {
  const w = ident ? identWords(ident) : '';
  return w ? `${VAR_MARK} ${w}${VAR_END}` : `${VAR_MARK}${VAR_END}`;
}

/** Last segment of a translation key ("home.hero.title" → "title"). */
export function keyLastSegment(key: string): string {
  const segs = key.split(/[.:/]/).filter(Boolean);
  return segs.length ? segs[segs.length - 1] : key;
}

/** A translation-call argument reads as copy (gettext style) rather than a key. */
export function looksLikeCopy(s: string): boolean {
  if (/\s/.test(s.trim())) return true;
  if (/[.:/]/.test(s) && !/[.!?…]$/.test(s)) return false;
  if (/_/.test(s)) return false;
  return /^[A-Z]/.test(s) || /[.!?]$/.test(s);
}

export function collapse(s: string): string {
  return s.replace(/\s+/g, ' ').trim();
}

/** React's JSX whitespace rule: trim lines, drop blank lines, join with a single space. */
export function jsxWhitespace(value: string): string {
  const lines = value.split(/\r\n|\n|\r/);
  let lastNonEmpty = 0;
  for (let i = 0; i < lines.length; i++) if (/[^ \t]/.test(lines[i])) lastNonEmpty = i;
  let out = '';
  for (let i = 0; i < lines.length; i++) {
    const isFirst = i === 0;
    const isLast = i === lines.length - 1;
    let line = lines[i].replace(/\t/g, ' ');
    if (!isFirst) line = line.replace(/^[ ]+/, '');
    if (!isLast) line = line.replace(/[ ]+$/, '');
    if (line) {
      if (i !== lastNonEmpty) line += ' ';
      out += line;
    }
  }
  return out;
}

/** Offsets of the first and last non-whitespace characters of src[s, e). Null when all whitespace. */
export function trimSpan(src: string, s: number, e: number): [number, number] | null {
  let a = s;
  let b = e;
  while (a < b && /\s/.test(src[a])) a++;
  while (b > a && /\s/.test(src[b - 1])) b--;
  return a < b ? [a, b] : null;
}

const MINOR = new Set([
  'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for', 'so', 'yet', 'of', 'to', 'in', 'on', 'at', 'by', 'up', 'as',
  'is', 'if', 'it', 'vs', 'via', 'per', 'with', 'from', 'into', 'onto', 'over',
]);

/** Separates the literal parts of conditional branches; caps and title_case must hold in every branch. */
export const ALT_SEP = ALT_MARK;

const branches = (lit: string): string[] => lit.split(ALT_SEP).filter((b) => /\p{L}/u.test(b));

/** At least two cased letters and every cased letter upper case (in every branch). */
export function isCaps(lit: string): boolean {
  const bs = branches(lit);
  if (bs.length > 1) return bs.every(isCaps);
  const letters = lit.match(/\p{L}/gu) ?? [];
  const cased = letters.filter((c) => c.toLowerCase() !== c.toUpperCase());
  return cased.length >= 2 && cased.every((c) => c === c.toUpperCase());
}

/** Two or more words, each non-minor word capitalised, at least one capitalised word after the first. */
export function isTitleCase(lit: string): boolean {
  const bs = branches(lit);
  if (bs.length > 1) return bs.every(isTitleCase);
  if (isCaps(lit)) return false;
  const words = lit.split(/\s+/).map((w) => w.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '')).filter((w) => /\p{L}/u.test(w));
  if (words.length < 2) return false;
  let laterCap = 0;
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    if (/^\p{N}/u.test(w)) continue;
    const first = w[0];
    const upper = first === first.toUpperCase() && first !== first.toLowerCase();
    if (upper) {
      if (i > 0) laterCap++;
      continue;
    }
    if (i > 0 && MINOR.has(w.toLowerCase())) continue;
    return false;
  }
  return laterCap > 0;
}

const DOT_LEAD = /^[●•◉⬤🟢🔴🟡🟠🟣🔵⚫⚪🟤]/u;
export function hasDotLead(text: string): boolean {
  return DOT_LEAD.test(text.trim());
}

/** Only dot glyphs (and spaces): such an element is a dot, never a node. */
export function isDotOnly(text: string): boolean {
  const t = text.trim();
  return t.length > 0 && /^[●•◉⬤🟢🔴🟡🟠🟣🔵⚫⚪🟤\s]+$/u.test(t);
}

/** No letters, digits or pictographs: separators like "·", "/", "|" are not nodes. */
export function isSeparatorOnly(text: string): boolean {
  return !/[\p{L}\p{N}\p{Extended_Pictographic}★☆✓→↗»€£$¥%&?!\uE000\uE001]/u.test(text);
}

export function isEmojiLead(text: string): boolean {
  const t = text.trim();
  const cp = t.codePointAt(0);
  if (cp === undefined) return false;
  const ch = String.fromCodePoint(cp);
  if (!/\p{Extended_Pictographic}/u.test(ch)) return false;
  if (cp < 0x2000) return false; // © ®
  if (cp >= 0x2100 && cp <= 0x21ff) return false; // letterlike symbols, arrows
  if (cp >= 0x25a0 && cp <= 0x25ff) return false; // geometric shapes
  return true;
}

export function hasArrowTail(text: string): boolean {
  return /(→|↗|↘|»|›|->|⟶|➜|➔|⇢|⇾)\s*$/u.test(text);
}

/**
 * Strip the internal VAR_END marks from a node's text and count, for each VAR marker in order, the identifier words
 * that follow it (the node's `vw` field).
 */
export function varWords(text: string): { text: string; vw: number[] } {
  const vw: number[] = [];
  const re = /\uE000([^\uE000\uE002]*)\uE002/gu;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) vw.push(m[1].split(/\s+/).filter(Boolean).length);
  const n = (text.match(/\uE000/gu) ?? []).length;
  while (vw.length < n) vw.push(0);
  return { text: collapse(text.replace(/\uE002/gu, '')), vw };
}

/** Words, counting each VAR or ALT marker as one. */
export function wordCount(text: string): number {
  return text.split(/\s+/).filter((w) => /[\p{L}\p{N}\uE000\uE001]/u.test(w)).length;
}

/** Mostly a figure: "98%", "12k+", "£4.20", "4.9/5", "1,204". */
export function isFigure(lit: string): boolean {
  const t = lit.trim();
  if (!/\d/.test(t)) return false;
  if (wordCount(t) > 2) return false;
  return /^[~≈<>+\-−£$€¥#]?\s?\d[\d,.\s]*(%|[kKmMbB]\+?|\+|x|×|\/\s?\d+|\s?(ms|s|h|d|GB|MB|TB))?$/u.test(t);
}

// ---------------------------------------------------------------------------------------------------------------
// Hand-built empty states

/**
 * Does a condition (its source text) test that a collection is empty? 'pos' when true means empty (`xs.length === 0`,
 * `!xs.length`, `xs.isEmpty`), 'neg' when true means not empty (`xs.length > 0`, `!xs.isEmpty`), null otherwise.
 */
export function emptyCond(src: string): 'pos' | 'neg' | null {
  const t = src.replace(/\s+/g, '');
  if (/[|&]{2}|,/.test(t)) return null;
  if (/^!\(?[\w$.?]+\.(length|size)\)?$|\.(length|size|count)(===?|<=?)0$|\.(length|size|count)<1$|^0===?[\w$.?]+\.(length|size|count)$|^[\w$.?]*\.isEmpty(\(\))?$|^isEmpty\(|^(?![\w$]*(Not|Non)[A-Z])[\w$]*Empty$/.test(t)) return 'pos';
  if (/\.(length|size|count)(>0|!==?0|>=1)$|^!![\w$.?]+\.(length|size)$|^!(is|has)?[\w$.?]*Empty$|^![\w$.?]+\.isEmpty(\(\))?$/.test(t)) return 'neg';
  return null;
}

/**
 * The branch shown when a collection is empty: a hand-built empty state when it holds one to four lines of text, at
 * least one of them literal copy, and no list, template or other container. Its element (or a wrapper round several) becomes an empty-state container.
 */
export function markEmptyKids(kids: Kid[], s: number, e: number): Kid[] {
  let texts = 0;
  let literal = false;
  let blocked = false;
  const walk = (k: Kid): void => {
    if (k.t === 'txt') {
      if (!k.attr && k.text.trim()) {
        texts++;
        if (/\p{L}/u.test(k.lit)) literal = true;
      }
      return;
    }
    if (k.tmpl || (k.box && k.box !== 'card' && k.box !== 'section')) blocked = true;
    k.kids.forEach(walk);
  };
  kids.forEach(walk);
  // an empty state says so in words of its own: a branch showing only bound values is not one
  if (blocked || !literal || texts < 1 || texts > 4) return kids;
  if (kids.length === 1 && kids[0].t === 'el') {
    kids[0].box = 'empty';
    return kids;
  }
  const w = el('#empty', s, e, { box: 'empty', cond: true });
  w.kids = kids;
  return [w];
}
