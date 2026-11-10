// Class tokens: Tailwind utilities, the repo's own simple CSS class selectors, semantic class names and the
// slop.config.json overrides. Everything here is read from static class tokens only.

import { lstatSync, readFileSync, readdirSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import type { Sty } from './ir.ts';
import { isRole } from './schema.ts';
import type { Role } from './schema.ts';

/** Style properties of one simple class selector in the repo's CSS. */
export interface CssRule {
  caps?: boolean;
  tracked?: boolean;
  small?: boolean;
  xsmall?: boolean;
  mono?: boolean;
  rounded?: boolean;
  dotSize?: boolean;
  truncated?: boolean;
  srOnly?: boolean;
}
export type CssMap = Map<string, CssRule>;

export interface SlopConfig {
  classRoles: Map<string, Role>;
  componentRoles: Map<string, Role>;
}

export function emptyConfig(): SlopConfig {
  return { classRoles: new Map(), componentRoles: new Map() };
}

// ---------------------------------------------------------------------------------------------------------------
// Tailwind

function pxOf(v: string, unit: string): number {
  const n = parseFloat(v);
  if (unit === 'px') return n;
  return n * 16; // rem and em against a 16px root
}

/** Strip the important marker; return null for variant-prefixed tokens (md:, hover:, dark:), which are ignored. */
export function baseToken(tok: string): string | null {
  if (tok.includes(':')) {
    // arbitrary values may contain ':' inside brackets only
    const outside = tok.replace(/\[[^\]]*\]/g, '');
    if (outside.includes(':')) return null;
  }
  return tok.replace(/^!/, '').replace(/!$/, '');
}

const SIZE_TOK = /^text-(xs|sm|base|lg|xl|[2-9]xl)$/;
const SIZE_ARB = /^text-\[(\d*\.?\d+)(px|rem|em)\]$/;
const ALIGN_TOK = /^text-(left|right|center|justify|start|end|balance|pretty|wrap|nowrap|ellipsis|clip)$/;

/** Style signals of one token. `undefined` means the token says nothing about that property. */
export function tokenSty(tok0: string, css: CssMap): Sty {
  const tok = baseToken(tok0);
  const out: Sty = {};
  if (tok === null) return out;
  let m: RegExpMatchArray | null;
  if ((m = tok.match(SIZE_TOK))) {
    const sz = m[1];
    out.small = sz === 'xs' || sz === 'sm';
    out.xsmall = sz === 'xs';
    out.large = /^[2-9]xl$/.test(sz);
  } else if ((m = tok.match(SIZE_ARB))) {
    const px = pxOf(m[1], m[2]);
    out.small = px <= 14;
    out.xsmall = px <= 12;
    out.large = px >= 24;
  } else if (/^text-/.test(tok) && !ALIGN_TOK.test(tok)) {
    // a colour token
    out.muted = /^text-(muted-foreground|muted|secondary|subtle)$/.test(tok);
    out.errorColour = /^text-(destructive|danger|error)$/.test(tok);
  }
  if (tok === 'uppercase') out.caps = true;
  if (tok === 'normal-case' || tok === 'lowercase') out.caps = false;
  if (tok === 'capitalize') out.titleCase = true;
  if ((m = tok.match(/^tracking-(.+)$/))) {
    const v = m[1];
    if (/^(wide|wider|widest)$/.test(v)) out.tracked = true;
    else if (/^\[(\d*\.?\d+)(em|px|rem)\]$/.test(v)) out.tracked = parseFloat(v.slice(1)) > 0;
    else out.tracked = false;
  }
  if (/^-tracking-/.test(tok)) out.tracked = false;
  if (tok === 'font-mono') out.mono = true;
  if (tok === 'font-sans' || tok === 'font-serif') out.mono = false;
  if (/^rounded(-[a-z]+)?-full$|^rounded-\[(9{3,}px|50%|100%)\]$/.test(tok)) out.rounded = true;
  if (/^(size|h|w)-(0\.5|1|1\.5|2|2\.5|3)$/.test(tok) || /^(size|h|w)-\[(\d*\.?\d+)px\]$/.test(tok) && parseFloat(tok.split('[')[1]) <= 12) {
    out.dotSize = true;
  }
  if (tok === 'truncate' || /^line-clamp-\d+$/.test(tok) || tok === 'text-ellipsis') out.truncated = true;
  if (tok === 'sr-only' || tok === 'visually-hidden' || tok === 'visuallyhidden' || tok === 'screen-reader-text') out.srOnly = true;
  if (tok === 'not-sr-only') out.srOnly = false;
  const rule = css.get(tok);
  if (rule) {
    for (const k of Object.keys(rule) as (keyof CssRule)[]) {
      if (rule[k] !== undefined && (out as Record<string, unknown>)[k] === undefined) (out as Record<string, unknown>)[k] = rule[k];
    }
  }
  return out;
}

/** Merge the signals of all of an element's own tokens, then its explicit style. Later tokens win. */
export function elementSty(cls: string[], own: Sty, css: CssMap): Sty {
  const out: Sty = {};
  for (const t of cls) {
    const s = tokenSty(t, css);
    for (const [k, v] of Object.entries(s)) if (v !== undefined) (out as Record<string, unknown>)[k] = v;
  }
  for (const [k, v] of Object.entries(own)) if (v !== undefined) (out as Record<string, unknown>)[k] = v;
  if (out.rounded && out.dotSize) out.dotSize = true;
  return out;
}

/** Tailwind card surface: rounded corners, a border or shadow, and padding. */
export function isCardSurface(cls: string[]): boolean {
  const toks = cls.map(baseToken).filter((t): t is string => t !== null);
  const rounded = toks.some((t) => /^rounded(-(sm|md|lg|xl|2xl|3xl|\[[^\]]+\]))?$/.test(t));
  const edge = toks.some((t) => /^(border|border-\d|shadow|shadow-(sm|md|lg|xl|2xl)|ring-1)$/.test(t));
  const pad = toks.some((t) => /^p[xy]?-(\d|\[)/.test(t));
  const named = toks.some((t) => /(^|[-_])card$/.test(t) || t === 'card');
  return named || (rounded && edge && pad);
}

// ---------------------------------------------------------------------------------------------------------------
// Semantic class names (BEM and plain CSS) and slop.config.json overrides

const SEMANTIC: [RegExp, Role | 'kicker'][] = [
  [/^(eyebrow|kicker|overline|pretitle|supertitle)$/, 'eyebrow'],
  [/^(subtitle|subheading|subhead|tagline|lede|lead|strapline|standfirst)$/, 'subtitle'],
  [/^(caption)$/, 'caption'],
  [/^(badge|pill|chip|tag)$/, 'badge'],
  [/^(hint|helper|help-text|helptext)$/, 'helper'],
  [/^(tooltip)$/, 'tooltip'],
];

/** The last word of a class name ("hero__eyebrow" → eyebrow, "status-pill" → pill). */
function lastWord(tok: string): string {
  const parts = tok.toLowerCase().split(/__|--|-|_/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : tok.toLowerCase();
}

export function semanticClassRole(cls: string[], cfg: SlopConfig): Role | null {
  for (const t0 of cls) {
    const t = baseToken(t0);
    if (t === null) continue;
    const r = cfg.classRoles.get(t);
    if (r) return r;
  }
  for (const t0 of cls) {
    const t = baseToken(t0);
    if (t === null || t.includes('[')) continue;
    if (/^(text|bg|border|ring|from|to|via|fill|stroke|shadow|outline|decoration|accent|caret|divide|placeholder)-/.test(t)) continue;
    const w = lastWord(t);
    for (const [re, role] of SEMANTIC) {
      if (re.test(w) && role !== 'kicker') return role;
    }
  }
  return null;
}

export function hasDotClass(cls: string[]): boolean {
  return cls.some((t) => {
    const b = baseToken(t);
    return b !== null && /(^|[-_])dot($|[-_])/i.test(b) && !/dotted/i.test(b);
  });
}

// ---------------------------------------------------------------------------------------------------------------
// CSS

/** Parse CSS text into blocks; only simple class selectors (".name", comma lists of them) are kept. */
export function parseCss(text: string, into: CssMap): void {
  const src = text.replace(/\/\*[\s\S]*?\*\//g, ' ');
  let i = 0;
  const walk = (end: number): void => {
    let start = i;
    while (i < end) {
      const c = src[i];
      if (c === '{') {
        const prelude = src.slice(start, i).trim();
        i++;
        const bodyStart = i;
        let depth = 1;
        let j = i;
        while (j < src.length && depth > 0) {
          if (src[j] === '{') depth++;
          else if (src[j] === '}') depth--;
          j++;
        }
        const bodyEnd = j - 1;
        const body = src.slice(bodyStart, bodyEnd);
        if (/^@/.test(prelude)) {
          if (/^@(media|supports|layer|container)\b/.test(prelude)) {
            i = bodyStart;
            walk(bodyEnd);
          }
        } else {
          // declarations at this level only
          const flat = body.replace(/\{[^{}]*\}/g, ' ');
          const sels = prelude.split(',').map((s) => s.trim());
          for (const sel of sels) {
            const m = sel.match(/^\.(-?[_a-zA-Z][\w-]*)$/);
            if (m) addRule(into, m[1], flat);
          }
          if (/\{/.test(body)) {
            // nested blocks (SCSS): only simple nested class selectors
            i = bodyStart;
            walk(bodyEnd);
          }
        }
        i = j;
        start = i;
      } else if (c === ';' || c === '}') {
        i++;
        start = i;
      } else {
        i++;
      }
    }
  };
  walk(src.length);
}

function lengthPx(v: string): number | null {
  const m = v.trim().match(/^(-?\d*\.?\d+)(px|rem|em)?$/);
  if (!m) return null;
  const n = parseFloat(m[1]);
  if (!m[2]) return n === 0 ? 0 : null;
  return m[2] === 'px' ? n : n * 16;
}

function addRule(into: CssMap, name: string, decls: string): void {
  const rule: CssRule = { ...(into.get(name) ?? {}) };
  let width: number | null = null;
  let height: number | null = null;
  for (const d of decls.split(';')) {
    const k = d.indexOf(':');
    if (k < 0) continue;
    const prop = d.slice(0, k).trim().toLowerCase();
    const val = d.slice(k + 1).replace(/!important/, '').trim().toLowerCase();
    if (prop === 'text-transform') rule.caps = val === 'uppercase';
    else if (prop === 'letter-spacing') {
      const px = val.endsWith('em') ? parseFloat(val) * 16 : lengthPx(val);
      rule.tracked = px !== null && px > 0;
    } else if (prop === 'font-size') {
      if (/^(x-small|xx-small|small|smaller)$/.test(val)) {
        rule.small = true;
        rule.xsmall = val !== 'small';
      } else {
        const px = lengthPx(val);
        if (px !== null) {
          rule.small = px <= 14;
          rule.xsmall = px <= 12;
        }
      }
    } else if (prop === 'font-family') rule.mono = /mono|courier|menlo|consolas|monaco/.test(val);
    else if (prop === 'border-radius') rule.rounded = /^(50%|100%|9{3,}px|\d{3,}px|\d{2,}rem)$/.test(val);
    else if (prop === 'width') width = lengthPx(val);
    else if (prop === 'height') height = lengthPx(val);
    else if (prop === 'text-overflow') rule.truncated = val === 'ellipsis';
    else if (prop === '-webkit-line-clamp') rule.truncated = true;
    else if (prop === 'clip' && /rect\(0/.test(val)) rule.srOnly = true;
  }
  if (width !== null && height !== null && width <= 12 && height <= 12 && width > 0) rule.dotSize = true;
  into.set(name, rule);
}

const SKIP_DIRS = new Set(['node_modules', '.next', 'dist', 'build', '.git', 'out', 'coverage', 'vendor', '.turbo', '.vercel', 'Pods', 'DerivedData', '.build']);

/** All .css/.scss/.less files under root (sorted), skipping build output and vendored folders. */
export function findCssFiles(root: string): string[] {
  const out: string[] = [];
  const walk = (dir: string): void => {
    let names: string[];
    try {
      names = readdirSync(dir).sort();
    } catch {
      return;
    }
    for (const n of names) {
      if (SKIP_DIRS.has(n) || n.endsWith('.worktrees') || n.startsWith('.env')) continue;
      const p = join(dir, n);
      let st;
      try {
        st = lstatSync(p);
      } catch {
        continue;
      }
      if (st.isDirectory()) {
        const rel = relative(root, p).split(sep).join('/');
        if (/(^|\/)components\/ui$/.test(rel)) continue;
        walk(p);
      } else if (/\.(css|scss|less)$/.test(n) && st.size < 1_000_000) out.push(p);
    }
  };
  walk(root);
  return out;
}

export function loadRepoCss(root: string): CssMap {
  const map: CssMap = new Map();
  for (const f of findCssFiles(root)) {
    try {
      parseCss(readFileSync(f, 'utf8'), map);
    } catch {
      // unreadable stylesheet: ignored
    }
  }
  return map;
}

export function loadConfig(root: string): SlopConfig {
  const cfg = emptyConfig();
  let raw: string;
  try {
    raw = readFileSync(join(root, 'slop.config.json'), 'utf8');
  } catch {
    return cfg;
  }
  const j = JSON.parse(raw) as { classRoles?: Record<string, string>; componentRoles?: Record<string, string> };
  for (const [k, v] of Object.entries(j.classRoles ?? {})) {
    if (!isRole(v)) throw new Error(`slop.config.json: unknown role "${v}" for class "${k}"`);
    cfg.classRoles.set(k, v);
  }
  for (const [k, v] of Object.entries(j.componentRoles ?? {})) {
    if (!isRole(v)) throw new Error(`slop.config.json: unknown role "${v}" for component "${k}"`);
    cfg.componentRoles.set(k, v);
  }
  return cfg;
}
