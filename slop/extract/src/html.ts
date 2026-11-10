// HTML front end (parse5). Skips <head>, scripts, styles and templates; reads <style> blocks into the class map;
// turns {{ mustache }} interpolations into the VAR marker plus words and strips {% template tags %}.

import { parse, parseFragment } from 'parse5';
import type { El, Kid } from './ir.ts';
import { el, keyLastSegment, stripPUA, txt, varText } from './ir.ts';
import { decorateIntrinsic } from './roles.ts';
import type { Role } from './schema.ts';
import type { CssMap } from './style.ts';
import { parseCss } from './style.ts';

// parse5 tree nodes are handled structurally.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type N = any;

const SKIP = new Set(['head', 'script', 'style', 'template', 'noscript', 'meta', 'link', 'title', 'iframe', 'object', 'math']);
const STATIC_ATTRS = new Set(['role', 'type', 'href', 'for', 'aria-level', 'aria-label', 'variant']);

/** Span of an attribute's value inside the source (between the quotes). */
function attrValueSpan(src: string, loc: N): [number, number] | null {
  if (!loc) return null;
  const raw = src.slice(loc.startOffset, loc.endOffset);
  const eq = raw.indexOf('=');
  if (eq < 0) return null;
  let i = eq + 1;
  while (i < raw.length && /\s/.test(raw[i])) i++;
  const q = raw[i];
  if (q === '"' || q === "'") return [loc.startOffset + i + 1, loc.endOffset - 1];
  return [loc.startOffset + i, loc.endOffset];
}

/** Split text into literal pieces and {{ }} interpolations. Offsets index the raw source. */
function textPieces(src: string, s: number, e: number): Kid[] {
  const raw = src.slice(s, e);
  const out: Kid[] = [];
  const re = /\{\{\s*([\s\S]*?)\s*\}\}|\{%[\s\S]*?%\}|\{#[\s\S]*?#\}/g;
  let last = 0;
  let m: RegExpExecArray | null;
  const pushLit = (a: number, b: number): void => {
    if (b <= a) return;
    const piece = decode(raw.slice(a, b));
    if (piece.trim() === '') {
      out.push(txt(' ', s + a, s + a));
      return;
    }
    let x = a;
    let y = b;
    while (x < y && /\s/.test(raw[x])) x++;
    while (y > x && /\s/.test(raw[y - 1])) y--;
    out.push(txt(piece, s + x, s + y));
  };
  while ((m = re.exec(raw))) {
    pushLit(last, m.index);
    if (m[1] !== undefined) {
      const expr = m[1].split('|')[0].trim();
      const ident = (expr.match(/[A-Za-z_$][\w$]*(?=\s*(\(|$))|[A-Za-z_$][\w$]*$/) ?? [expr])[0];
      const key = /^['"]/.test(expr) ? keyLastSegment(expr.replace(/['"]/g, '')) : ident;
      out.push(txt(` ${varText(key)} `, s + m.index, s + m.index + m[0].length, { lit: ' ', dyn: true, bound: true }));
    }
    last = m.index + m[0].length;
  }
  pushLit(last, raw.length);
  return out;
}

/**
 * Decode character references (named, decimal and hexadecimal) exactly as a browser does in text: parse5 reads the
 * text as the content of a <title> element, which takes references but no tags.
 */
function decode(s: string): string {
  if (!s.includes('&')) return stripPUA(s);
  const frag: N = parseFragment(`<title>${s.replace(/</g, '&lt;')}</title>`);
  const t = frag.childNodes?.[0]?.childNodes?.[0];
  return stripPUA(t?.nodeName === '#text' ? t.value : s);
}

export function htmlToIR(src: string, css: CssMap): { root: El; css: CssMap } {
  const isDoc = /<!doctype|<html[\s>]/i.test(src.slice(0, 2000));
  const doc: N = isDoc ? parse(src, { sourceCodeLocationInfo: true }) : parseFragment(src, { sourceCodeLocationInfo: true });
  const local: CssMap = new Map(css);
  // <style> blocks first, so the class map is complete before roles are read
  const collectStyles = (n: N): void => {
    if (n.nodeName === 'style') {
      for (const c of n.childNodes ?? []) if (c.nodeName === '#text') parseCss(c.value, local);
    }
    for (const c of n.childNodes ?? []) collectStyles(c);
    if (n.content) collectStyles(n.content);
  };
  collectStyles(doc);

  const convert = (n: N): Kid[] => {
    if (n.nodeName === '#text') {
      const loc = n.sourceCodeLocation;
      if (!n.value.trim()) return [txt(' ', loc?.startOffset ?? 0, loc?.startOffset ?? 0)];
      if (!loc) return [txt(stripPUA(n.value), 0, 0)];
      const raw = src.slice(loc.startOffset, loc.endOffset);
      if (/[<>]/.test(raw) && !/[<>]/.test(n.value)) {
        // parse5 can stretch a text node's location over end tags it moved; fall back to the decoded value
        const at = raw.indexOf(n.value.trim().slice(0, 20));
        const s0 = at >= 0 ? loc.startOffset + at : loc.startOffset;
        return [txt(stripPUA(n.value), s0, s0 + n.value.trim().length)];
      }
      return textPieces(src, loc.startOffset, loc.endOffset);
    }
    if (n.nodeName === '#comment' || n.nodeName === '#documentType') return [];
    if (!n.tagName) return (n.childNodes ?? []).flatMap(convert);
    const tag = n.tagName.toLowerCase();
    if (SKIP.has(tag)) return [];
    const loc = n.sourceCodeLocation;
    if (!loc && /^(html|body)$/.test(tag)) return (n.childNodes ?? []).flatMap(convert);
    const e = el(tag, loc?.startOffset ?? 0, loc?.endOffset ?? 0, { intrinsic: true });
    const attrKids: Kid[] = [];
    const attrs: [string, string][] = (n.attrs ?? []).map((a: N) => [a.name, a.value]);
    const type = (attrs.find(([k]) => k === 'type')?.[1] ?? '').toLowerCase();
    for (const [k, v] of attrs) {
      if (k === 'class') e.cls.push(...v.split(/\s+/).filter(Boolean));
      else if (STATIC_ATTRS.has(k)) e.attrs[k] = v;
      if (k === 'hidden') e.attrs.hidden = 'true';
      let role: Role | null = null;
      if (k === 'placeholder') role = 'placeholder';
      else if (k === 'title') role = 'tooltip';
      else if (k === 'aria-label') role = 'aria';
      else if (k === 'alt') role = 'alt';
      else if (k === 'data-tooltip' || k === 'data-tip') role = 'tooltip';
      else if (k === 'value' && tag === 'input') role = /^(submit|button|reset)$/.test(type) ? 'button' : /^(hidden|checkbox|radio|range|color|date|number|file)$/.test(type) ? null : 'input_value';
      if (k === 'style') {
        const decl = v.toLowerCase();
        if (/text-transform\s*:\s*uppercase/.test(decl)) e.sty.caps = true;
        const ls = decl.match(/letter-spacing\s*:\s*(-?[\d.]+)/);
        if (ls) e.sty.tracked = parseFloat(ls[1]) > 0;
        const fs = decl.match(/font-size\s*:\s*([\d.]+)(px|rem|em)/);
        if (fs) {
          const px = fs[2] === 'px' ? parseFloat(fs[1]) : parseFloat(fs[1]) * 16;
          e.sty.small = px <= 14;
          e.sty.xsmall = px <= 12;
          e.sty.large = px >= 24;
          e.sty.fontPx = px;
        }
        const fw = decl.match(/font-weight\s*:\s*(bold(?:er)?|\d+)/);
        if (fw) e.sty.bold = /^bold/.test(fw[1]) || parseFloat(fw[1]) >= 600;
        if (/font-family\s*:[^;]*mono/.test(decl)) e.sty.mono = true;
      }
      if (!role || !v.trim()) continue;
      const span = attrValueSpan(src, loc?.attrs?.[k]);
      if (!span) continue;
      for (const p of textPieces(src, span[0], span[1])) {
        if (p.t === 'txt' && p.text.trim()) attrKids.push({ ...p, attr: role });
      }
    }
    if (tag === 'input' && /^(submit|button|reset)$/.test(type) && !attrs.some(([k]) => k === 'value')) {
      // a submit button with no value shows the browser's default label; nothing to read
    }
    decorateIntrinsic(e);
    if (tag === 'svg') {
      // an icon, unless it draws text (a chart's axis labels and values)
      const kids: Kid[] = (n.childNodes ?? []).flatMap(convert);
      const draws = (k: Kid): boolean => k.t === 'el' && ((k.name === 'text' && k.kids.some((x) => x.t !== 'txt' || x.text.trim() !== '')) || k.kids.some(draws));
      if (kids.some(draws)) {
        e.icon = false;
        e.arrowIcon = false;
        e.kids = attrKids.concat(kids);
      } else e.kids = attrKids;
      return [e];
    }
    if (tag === 'br') return [txt(' ', e.s, e.s)];
    let kids: Kid[] = (n.childNodes ?? []).flatMap(convert);
    if (n.content) kids = kids.concat((n.content.childNodes ?? []).flatMap(convert));
    if (tag === 'textarea') kids = kids.map((k) => (k.t === 'txt' && k.text.trim() ? { ...k, attr: 'input_value' as Role } : k));
    if (e.icon && /material-(icons|symbols)/.test(e.cls.join(' '))) kids = [];
    e.kids = attrKids.concat(kids);
    return [e];
  };

  const root = el('#root', 0, src.length, { box: 'root' });
  root.kids = convert(doc);
  return { root, css: local };
}
