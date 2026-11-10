// IR tree → screen nodes: runs of text, roles (tag/attribute → component → class → position), flags, repeated
// subtrees, clusters and section_first/last. Deterministic: depends only on the tree and the style inputs.

import type { Box, El, Kid, Opt, Sty, Txt } from './ir.ts';
import {
  ALT_JOIN, ALT_SEP, MAX_RENDERINGS, collapse, hasArrowTail, hasDotLead, isCaps, isDotOnly, isEmojiLead, isFigure, isSeparatorOnly, isTitleCase,
  joinRenderings, varWords, wordCount,
} from './ir.ts';
import { LISTISH, MONO_TAGS, STATEISH, strongRole } from './roles.ts';
import type { ClusterKind, Flag, Framework, Role } from './schema.ts';
import { CONTEXT_ROLES, FLAGS, HEADING_LEVEL, TRACK_C_ROLES } from './schema.ts';
import type { CssMap, SlopConfig } from './style.ts';
import { elementSty, hasDotClass, semanticClassRole } from './style.ts';

export interface InferCtx {
  fw: Framework;
  css: CssMap;
  cfg: SlopConfig;
}

export interface OutNode {
  i: number;
  k: number;
  role: Role;
  text: string;
  flags: Flag[];
  vw?: number[]; // identifier words after each VAR marker (only when the text has one)
  s: number;
  e: number;
}
export interface OutCluster {
  k: number;
  kind: ClusterKind;
  s: number; // offset of the cluster's line
}

type PosKind = 'kicker' | 'small' | 'muted' | 'none';

interface RNode {
  idx: number;
  text: string;
  lit: string;
  s: number;
  e: number;
  chain: El[]; // root … owner, then common inline wrappers
  attr?: Role;
  dyn: boolean;
  alt: boolean;
  cond: boolean;
  bound: boolean;
  dot: boolean;
  dotBound: boolean;
  icon: boolean;
  arrow: boolean;
  strong: Role | null;
  pos: PosKind | null;
  role: Role;
  flags: Set<Flag>;
  gp: El | null;
  local: El[]; // innermost elements up to and including the first block element
  alts: [number, number][]; // other spans the node may take if [s, e] belongs to an earlier node (expanded arrays)
}

const isHeading = (r: Role): boolean => HEADING_LEVEL[r] !== undefined;

export function infer(root: El, ctx: InferCtx): { nodes: OutNode[]; clusters: OutCluster[] } {
  // ---- parents and per-element style -------------------------------------------------------------------------
  const styOf = new Map<number, Sty>();
  const setParents = (e: El): void => {
    styOf.set(e.id, elementSty(e.cls, e.sty, ctx.css));
    for (const k of e.kids) if (k.t === 'el') {
      k.parent = e;
      setParents(k);
    }
  };
  setParents(root);
  const sty = (e: El): Sty => styOf.get(e.id) ?? {};

  const textMemo = new Map<number, boolean>();
  const hasText = (e: El): boolean => {
    let v = textMemo.get(e.id);
    if (v !== undefined) return v;
    v = e.kids.some((k) => (k.t === 'txt' ? k.text.trim() !== '' : hasText(k)));
    textMemo.set(e.id, v);
    return v;
  };
  const someDesc = (e: El, f: (x: El) => boolean): boolean => f(e) || e.kids.some((k) => k.t === 'el' && someDesc(k, f));
  const isDotEl = (e: El): boolean => {
    if (e.dot) return true;
    const s = sty(e);
    if (s.rounded && s.dotSize) return true;
    if (hasDotClass(e.cls)) return true;
    return false;
  };
  const strongMemo = new Map<number, Role | null>();
  const strongOf = (e: El): Role | null => {
    if (!strongMemo.has(e.id)) strongMemo.set(e.id, strongRole(e, ctx.fw, ctx.cfg));
    return strongMemo.get(e.id) ?? null;
  };
  const blocksMerge = (e: El): boolean => {
    const s = sty(e);
    if (e.noMerge || s.srOnly || s.rounded || (s.caps && s.tracked) || isDotEl(e)) return true;
    if (semanticClassRole(e.cls, ctx.cfg)) return true;
    const r = strongOf(e);
    return r !== null && r !== 'link';
  };
  const isTextBlock = (e: El): boolean => !!e.textBlock || (strongOf(e) !== null && !e.box);

  // A short coloured inline element: in a flex or grid row it is an item of its own (a status word), not part of
  // the row's text.
  const statusWord = (k: El): boolean => {
    if (!sty(k).hue || sty(k).muted) return false;
    const t = k.kids.map((x) => (x.t === 'txt' ? x.text : '')).join(' ');
    return k.kids.every((x) => x.t === 'txt' || !hasText(x)) && wordCount(t) >= 1 && wordCount(t) <= 3;
  };

  // Text-only dot glyph elements are dots, never nodes.
  const dotGlyphEl = (e: El): boolean =>
    e.kids.length > 0 && e.kids.every((k) => k.t === 'txt' && !k.attr) && isDotOnly(e.kids.map((k) => (k as Txt).text).join(''));

  // ---- flatten into runs --------------------------------------------------------------------------------------
  const nodes: RNode[] = [];
  let pendingDot: { scope: El; bound: boolean } | null = null;
  let pendingIcon: { scope: El } | null = null;

  interface Piece {
    txt?: Txt;
    mark?: 'dot' | 'icon' | 'arrow';
    bound?: boolean;
    inl: El[];
  }

  const mkNode = (p: Partial<RNode> & { text: string; lit: string; s: number; e: number; chain: El[] }): RNode => {
    const n: RNode = {
      idx: nodes.length, dyn: false, alt: false, cond: false, bound: false, dot: false, dotBound: false, icon: false,
      arrow: false, strong: null, pos: null, role: 'body', flags: new Set(), gp: null, local: [], alts: [], ...p,
    };
    if (!n.attr && pendingDot && n.chain.includes(pendingDot.scope)) {
      n.dot = true;
      n.dotBound = n.dotBound || pendingDot.bound;
      pendingDot = null;
    }
    if (!n.attr && pendingIcon && n.chain.includes(pendingIcon.scope)) {
      n.icon = true;
      pendingIcon = null;
    }
    nodes.push(n);
    return n;
  };

  // Inside an expanded array copy (jsx.ts) a node sits at the first array literal substituted into its text, else at
  // the copy's array element. Its other literals, their properties, the element and its span in the callback are
  // fallbacks for when that span is taken (see the de-duplication below).
  // A node showing only an in-file constant sits at the constant's literal (its use site if that is taken).
  const place = (pieces: Txt[], s: number, e: number): { s: number; e: number; alts: [number, number][] } => {
    const copy = pieces.find((t) => t.copy)?.copy;
    const call = pieces.find((t) => t.call)?.call;
    const shown = pieces.filter((t) => t.text.trim());
    const lspan = shown.length === 1 ? shown[0].lspan : undefined;
    const cands: [number, number][] = [
      ...pieces.flatMap((t) => t.subs ?? []), ...pieces.flatMap((t) => t.near ?? []), ...(lspan ? [lspan] : []), ...(copy ? [copy] : []), [s, e],
      ...(call ? [call] : []),
    ];
    return { s: cands[0][0], e: cands[0][1], alts: cands.slice(1) };
  };

  const emitAttr = (t: Txt, chain: El[]): void => {
    const text = collapse(t.text);
    if (!text) return;
    mkNode({ text, lit: collapse(t.lit), ...place([t], t.s, t.e), chain, attr: t.attr, dyn: !!t.dyn, alt: !!t.alt, cond: !!t.cond, bound: !!t.bound });
  };

  const visit = (e: El, chain: El[]): void => {
    if (e.hidden) return;
    const ch = chain.concat(e);
    if (dotGlyphEl(e)) {
      pendingDot = { scope: e.parent ?? e, bound: e.bound };
      return;
    }
    const mixed = isTextBlock(e) || e.kids.some((k) => k.t === 'txt' && !k.attr && k.text.trim() !== '');
    let run: Piece[] = [];
    let deferred: { txt: Txt; chain: El[] }[] = [];
    const flush = (): void => {
      const pieces = run;
      const defer = deferred;
      run = [];
      deferred = [];
      buildRun(e, ch, pieces);
      for (const d of defer) emitAttr(d.txt, d.chain);
    };
    const push = (k: Kid, inl: El[]): void => {
      if (k.t === 'txt') {
        if (k.attr) deferred.push({ txt: k, chain: ch.concat(inl) });
        else run.push({ txt: k, inl });
        return;
      }
      if (k.hidden) return;
      if (!hasText(k)) {
        if (someDesc(k, isDotEl)) run.push({ mark: 'dot', bound: someDesc(k, (x) => isDotEl(x) && x.bound), inl });
        else if (someDesc(k, (x) => !!x.arrowIcon)) run.push({ mark: 'arrow', inl });
        else if (someDesc(k, (x) => !!x.icon)) run.push({ mark: 'icon', inl });
        return;
      }
      if (dotGlyphEl(k)) {
        run.push({ mark: 'dot', bound: k.bound, inl });
        return;
      }
      if (mixed && k.inline && !blocksMerge(k) && !k.box && !(sty(e).flexBox && statusWord(k))) {
        for (const kk of k.kids) push(kk, inl.concat(k));
        return;
      }
      flush();
      visit(k, ch.concat(inl));
    };
    for (const k of e.kids) {
      if (k.t === 'txt' && k.attr) {
        flush();
        emitAttr(k, ch);
        continue;
      }
      push(k, []);
    }
    flush();
    if (pendingDot && pendingDot.scope === e) pendingDot = null;
    if (pendingIcon && pendingIcon.scope === e) pendingIcon = null;
  };

  /** Concatenate text pieces with the browser's spacing; `alts`: fragments of different `&&` groups join with ALT. */
  const concatRun = (tpieces: Piece[], alts: boolean): { text: string; lit: string; altJoined: boolean } => {
    let text0 = '';
    let lit0 = '';
    let altJoined = false;
    let prev: Piece | null = null;
    for (const cur of tpieces) {
      const t = cur.txt as Txt;
      if (t.text === '') continue;
      if (prev) {
        const pt = prev.txt as Txt;
        if (alts && pt.andGroup !== undefined && t.andGroup !== undefined && pt.andGroup !== t.andGroup && pt.text.trim() && t.text.trim()) {
          // fragments shown under different && conditions are alternatives, not one line of text
          text0 = `${text0.trimEnd()}${ALT_JOIN}${t.text.trimStart()}`;
          lit0 = `${lit0.trimEnd()}${ALT_SEP}${t.lit.trimStart()}`;
          altJoined = true;
          prev = cur;
          continue;
        }
        const ci = cur.inl;
        const sameEl = prev.inl.length === ci.length && prev.inl.every((x, j) => x === ci[j]);
        // an inline tag displayed as a block starts its own line: the words on either side stay apart
        const pin = prev.inl;
        const crossed = [...pin.filter((x) => !ci.includes(x)), ...ci.filter((x) => !pin.includes(x))];
        const blockEdge = crossed.some((x) => sty(x).blockDisplay);
        if ((blockEdge || (!sameEl && pin.length > 0 && ci.length > 0)) && /[\p{L}\p{N}.:!?)\uE000\uE001\uE002]$/u.test(text0) && /^[\p{L}\p{N}(\uE000\uE001]/u.test(t.text)) {
          text0 += ' ';
          lit0 += ' ';
        }
      }
      text0 += t.text;
      lit0 += t.lit;
      prev = cur;
    }
    return { text: text0, lit: lit0, altJoined };
  };

  /**
   * A run of text with an optional part (a fragment under `&&`, a branch that renders nothing) is written as its
   * complete renderings joined with ALT, the version without the part included ("Saved ALT Saved · draft"), so that
   * the part does not read as always present. Adjacent fragments under different `&&` conditions are one choice of
   * alternatives. Null when nothing is optional, or past MAX_RENDERINGS (the branches then stay inline).
   */
  const renderRun = (tpieces: Piece[]): { text: string; lit: string; altJoined: boolean } | null => {
    const groupPath = (p: Piece): number[] => {
      const out: number[] = [];
      for (const x of p.inl) if (x.andGroup !== undefined) out.push(x.andGroup);
      const t = p.txt as Txt;
      if (t.andGroup !== undefined) out.push(t.andGroup);
      return out;
    };
    let optional = false;
    const withText = (p: Piece, o: Opt): Piece => ({ ...p, txt: { ...(p.txt as Txt), text: o.text, lit: o.lit } });
    // each slot is a list of options, each option the pieces it shows
    const slots = (ps: Piece[], depth: number): Piece[][][] | null => {
      const out: Piece[][][] = [];
      let i = 0;
      while (i < ps.length) {
        if (groupPath(ps[i])[depth] === undefined) {
          const p = ps[i];
          const o = (p.txt as Txt).opts;
          if (o && o.length >= 2) {
            if (o.some((x) => !x.text.trim())) optional = true;
            out.push(o.map((x) => [withText(p, x)]));
          } else out.push([[p]]);
          i++;
          continue;
        }
        const groups: Piece[][] = [];
        let cur: number | undefined;
        let j = i;
        for (; j < ps.length; j++) {
          const g = groupPath(ps[j])[depth];
          if (g === undefined) break;
          if (g !== cur) groups.push([]);
          cur = g;
          groups[groups.length - 1].push(ps[j]);
        }
        const options: Piece[][] = [];
        for (const grp of groups) {
          const sub = slots(grp, depth + 1);
          const combos = sub ? product(sub) : null;
          if (!combos) return null;
          options.push(...combos);
        }
        options.push([]);
        optional = true;
        out.push(options);
        i = j;
      }
      return out;
    };
    const product = (sl: Piece[][][]): Piece[][] | null => {
      let out: Piece[][] = [[]];
      for (const options of sl) {
        if (out.length * options.length > MAX_RENDERINGS) return null;
        out = out.flatMap((a) => options.map((b) => a.concat(b)));
      }
      return out;
    };
    const sl = slots(tpieces, 0);
    if (!sl || !optional) return null;
    const combos = product(sl);
    if (!combos) return null;
    const j = joinRenderings(combos.map((c) => concatRun(c, false)));
    return { text: j.text, lit: j.lit, altJoined: j.text.includes(ALT_SEP) };
  };

  const buildRun = (owner: El, ch: El[], pieces: Piece[]): void => {
    const textIdx = pieces.map((p, i) => (p.txt && p.txt.text.trim() !== '' ? i : -1)).filter((i) => i >= 0);
    if (textIdx.length === 0) {
      const prev = nodes[nodes.length - 1];
      for (const p of pieces) {
        if (p.mark === 'dot') pendingDot = { scope: owner, bound: !!p.bound };
        else if (p.mark === 'arrow' && prev && !prev.attr && prev.chain.includes(owner) && !pendingIcon) {
          prev.arrow = true;
          prev.icon = true;
        } else if (p.mark) pendingIcon = { scope: owner };
      }
      return;
    }
    const first = textIdx[0];
    const last = textIdx[textIdx.length - 1];
    // adjacent pieces from different inline elements with no space between them are separate words
    // (<a><strong>Name</strong><span>Detail</span></a> renders as two lines or two blocks far more often than as one word)
    const tpieces = pieces.filter((p) => p.txt);
    const joined = renderRun(tpieces) ?? concatRun(tpieces, true);
    const text0 = joined.text;
    const lit0 = joined.lit;
    const altJoined = joined.altJoined;
    const tp = tpieces.map((p) => p.txt as Txt);
    const text = collapse(text0);
    const lit = collapse(lit0);
    if (isDotOnly(text)) {
      pendingDot = { scope: owner, bound: tp.some((t) => !!t.bound) };
      return;
    }
    if (isSeparatorOnly(text)) return;
    // common inline wrappers of the text pieces
    let inl = pieces[first].inl;
    for (const i of textIdx) {
      const o = pieces[i].inl;
      let j = 0;
      while (j < inl.length && j < o.length && inl[j] === o[j]) j++;
      inl = inl.slice(0, j);
    }
    const textPieces = textIdx.map((i) => pieces[i].txt as Txt);
    const n = mkNode({
      text, lit, ...place(tp, textPieces[0].s, textPieces[textPieces.length - 1].e), chain: ch.concat(inl),
      dyn: tp.some((t) => !!t.dyn), alt: altJoined || tp.some((t) => !!t.alt), cond: textPieces.every((t) => !!t.cond),
      bound: textPieces.length === 1 && !!textPieces[0].bound,
    });
    for (let i = 0; i < pieces.length; i++) {
      const p = pieces[i];
      if (!p.mark) continue;
      if (p.mark === 'dot' && i < first) {
        n.dot = true;
        n.dotBound = n.dotBound || !!p.bound;
      } else if (p.mark === 'arrow' && i > last) {
        n.arrow = true;
        n.icon = true;
      } else if (p.mark === 'icon' || p.mark === 'arrow') n.icon = true;
    }
  };

  visit(root, []);

  // ---- stage 1–3: strong roles and class signals ---------------------------------------------------------------
  const localChain = (n: RNode): El[] => {
    const out: El[] = [];
    for (let j = n.chain.length - 1; j >= 0; j--) {
      const e = n.chain[j];
      out.push(e);
      if (!e.inline && e.name !== '#prop') break;
    }
    return out; // innermost first
  };
  const mergeSty = (els: El[]): Sty => {
    const out: Sty = {};
    for (const e of els) for (const [k, v] of Object.entries(sty(e))) if (v !== undefined) (out as Record<string, unknown>)[k] = v;
    return out;
  };

  const svgTextRole = (n: RNode): Role => {
    const toks = n.local.flatMap((e) => e.cls).map((t) => t.toLowerCase());
    if (toks.some((t) => /(^|[-_])(num|nums|number|numeric|value|values|data|datum|figure|count|pct|percent)$/.test(t))) return 'cell';
    if (toks.some((t) => /(^|[-_])(axis|tick|ticks|legend|label|labels|title|caption|key)$/.test(t))) return 'label';
    if (/\d/.test(n.lit) && isFigure(n.lit)) return 'cell';
    // a bare value inside a nested template (a grid of cells) is data; at one level it labels a row or column
    if (n.bound && n.chain.filter((e) => e.tmpl).length >= 2) return 'cell';
    return 'label';
  };
  // text that is one bound value named as an error (deepResearchError, model.errorMessage)
  const errorBinding = (n: RNode): boolean =>
    n.bound && /^\uE000 (?:[a-z0-9]+ )*errors?(?: (?:message|messages|text|detail|details|description|msg))?\uE002$/.test(n.text.trim());
  const linkTexts = new Map<number, number>();
  const cardLinkBlock = (n: RNode, j: number): boolean => {
    const link = n.chain[j];
    const between = n.chain.slice(j + 1);
    if (!between.some((x) => !x.name.startsWith('#') && (!x.inline || !!sty(x).blockDisplay))) return false;
    let c = linkTexts.get(link.id);
    if (c === undefined) {
      c = nodes.filter((m) => !m.attr && m.chain.includes(link)).length;
      linkTexts.set(link.id, c);
    }
    return c >= 2;
  };
  const hueCands: RNode[] = [];
  for (const n of nodes) {
    n.local = n.attr ? [n.chain[n.chain.length - 1]] : localChain(n);
    if (n.attr) {
      n.strong = n.attr;
      continue;
    }
    let strongAt = -1;
    for (let j = n.chain.length - 1; j >= 0; j--) {
      const r = strongOf(n.chain[j]);
      if (r) {
        // a card link: a paragraph or other block inside a link that holds several texts keeps its own role; only
        // text directly in the link (or in inline tags) is the link
        if (r === 'link' && ctx.fw !== 'swift' && cardLinkBlock(n, j)) break;
        n.strong = r;
        strongAt = j;
        break;
      }
    }
    const lsty = mergeSty([...n.local].reverse());
    // class stage
    let classRole: Role | null = null;
    let pos: PosKind | null = null;
    for (const e of n.local) {
      if (e.classHint) {
        classRole = e.classHint;
        break;
      }
      const sem = semanticClassRole(e.cls, ctx.cfg);
      if (sem) {
        classRole = sem;
        break;
      }
    }
    // SVG <text> in a chart: axis and legend words are labels, data values are cells (never body)
    if (!classRole && n.chain.some((e) => e.intrinsic && e.name === 'text')) classRole = svgTextRole(n);
    if (!classRole) {
      if (lsty.large && (isFigure(n.lit) || (n.dyn && n.lit.trim() === ''))) classRole = 'stat';
      else if (lsty.rounded && wordCount(n.text) <= 4) classRole = 'badge';
      else if (lsty.caps && lsty.tracked) pos = 'kicker';
      else if (lsty.errorColour && !hasDotLead(n.text) && !n.dot) classRole = 'error';
      else if (errorBinding(n) && !n.chain.some((e) => e.box === 'dialog')) classRole = 'error';
      else if (lsty.xsmall) pos = 'small';
      else if (lsty.muted) pos = 'muted';
    }
    // a short coloured word may be a status badge in a row (decided below, once rows are known); a red one too
    if ((!classRole || (classRole === 'error' && lsty.errorColour)) && (!n.strong || n.strong === 'list_item' || n.strong === 'cell')
      && lsty.hue && !lsty.muted && wordCount(n.text) <= 3 && n.local[0]?.inline && n.local[0]?.intrinsic) hueCands.push(n);
    if (n.strong) {
      // list items and cells are weak containers: a nearer pill or kicker wins
      const weak = n.strong === 'list_item' || n.strong === 'cell' || (n.strong === 'body' && n.chain[strongAt]?.name === 'dd');
      const nearer = strongAt < n.chain.length - 1 || (classRole === 'stat' && weak);
      if (weak && nearer && (classRole === 'badge' || classRole === 'eyebrow' || classRole === 'stat' || pos === 'kicker')) {
        n.strong = null;
      } else continue;
    }
    if (classRole) n.strong = classRole;
    else n.pos = pos ?? 'none';
  }

  // ---- group parents ------------------------------------------------------------------------------------------
  const slots = new Map<number, Set<string>>();
  for (const n of nodes) {
    for (let j = 0; j < n.chain.length; j++) {
      const e = n.chain[j];
      const key = j + 1 < n.chain.length ? `e${n.chain[j + 1].id}` : `n${n.idx}`;
      let s = slots.get(e.id);
      if (!s) slots.set(e.id, (s = new Set()));
      s.add(key);
    }
  }
  for (const n of nodes) {
    n.gp = n.chain[0];
    for (let j = n.chain.length - 1; j >= 0; j--) {
      if ((slots.get(n.chain[j].id)?.size ?? 0) >= 2) {
        n.gp = n.chain[j];
        break;
      }
    }
  }

  // Short status words in a coloured inline element inside a row, after the row's own text, are badges
  // ("Ready" in green beside a file name).
  for (const n of hueCands) {
    let row: El | null = null;
    for (let j = n.chain.length - 1; j >= 0; j--) {
      const e = n.chain[j];
      if ((e.intrinsic && /^(li|tr)$/.test(e.name)) || e.tmpl) {
        row = e;
        break;
      }
    }
    if (!row) continue;
    const r = row;
    const earlier = nodes.some((m) => m.idx < n.idx && m.chain.includes(r) && !m.attr && !(m.strong && CONTEXT_ROLES.has(m.strong)));
    if (earlier) {
      n.strong = 'badge';
      n.pos = null;
    }
  }
  // which nodes sit under each element (before de-duplication; for the kicker rule below)
  const pre = new Map<number, number[]>();
  for (const n of nodes) for (const e of n.chain) {
    let a = pre.get(e.id);
    if (!a) pre.set(e.id, (a = []));
    a.push(n.idx);
  }
  const PANEL: ReadonlySet<Box> = new Set<Box>(['card', 'dialog', 'state', 'alert']);
  /**
   * First node of its card or panel, with at least three more after it: the panel's own heading. (A kicker over a
   * single value is a field label, as in a HUD's "Objective" over the task.)
   */
  const leadsPanel = (n: RNode): boolean => {
    for (let j = n.chain.length - 1; j >= 0; j--) {
      const e = n.chain[j];
      if (!e.box || !PANEL.has(e.box)) continue;
      const u = (pre.get(e.id) ?? []).filter((x) => !nodes[x].attr);
      return u.length >= 4 && u[0] === n.idx;
    }
    return false;
  };

  // ---- stage 4: position ------------------------------------------------------------------------------------
  const cascade = (n: RNode): Sty => mergeSty(n.chain);
  let casc = nodes.map(cascade);
  const isContext = (n: RNode): boolean => (n.strong !== null && CONTEXT_ROLES.has(n.strong)) || !!casc[n.idx].srOnly;
  const neighbour = (n: RNode, dir: 1 | -1): RNode | null => {
    for (let i = n.idx + dir; i >= 0 && i < nodes.length; i += dir) {
      const m = nodes[i];
      if (isContext(m)) continue;
      return m.gp === n.gp ? m : null;
    }
    return null;
  };
  const headingNode = (m: RNode | null): boolean => !!m && m.strong !== null && isHeading(m.strong);
  // the element a node's text sits in (its first block element) and that element's parent
  const blockOf = (n: RNode): El => n.local[n.local.length - 1] ?? n.chain[n.chain.length - 1];
  /** Does content follow the node inside its block's parent? (The next node in reading order sits there.) */
  const contentAfter = (n: RNode): RNode | null => {
    const parent = blockOf(n).parent;
    for (let i = n.idx + 1; i < nodes.length; i++) {
      const m = nodes[i];
      if (isContext(m)) continue;
      return parent && m.chain.includes(parent) ? m : null;
    }
    return null;
  };
  /**
   * An untagged heading: a short line in a large, bold type (Tailwind text-lg or larger with font-semibold or bolder,
   * or the repo CSS's font-size of at least 18px with font-weight 600 or more) with content after it.
   */
  const headingPx = (n: RNode): number => {
    if (n.pos !== 'none' && n.pos !== 'muted') return 0;
    const ls = mergeSty([...n.local].reverse());
    if (!ls.bold || (ls.fontPx ?? 0) < 18) return 0;
    const t = n.text.trim();
    const w = wordCount(t);
    if (w < 1 || w > 10 || /[.,:;]$/.test(t) || isFigure(n.lit)) return 0;
    return ls.fontPx ?? 0;
  };
  const candPx = nodes.map(headingPx);
  /** Content follows the line: not a heading, nor a line in the same or a larger heading type. */
  const untaggedHeading = (n: RNode): Role | null => {
    const px = candPx[n.idx];
    if (!px) return null;
    const next = contentAfter(n);
    if (!next || headingNode(next) || candPx[next.idx] >= px) return null;
    return px >= 24 ? 'h2' : 'h3';
  };
  /** A group label in a sidebar: one to three words over a group of items ("Today" over a list). */
  const sidebarGroupLabel = (n: RNode): boolean => {
    if (!n.chain.some((e) => (e.intrinsic && e.name === 'aside') || /Sidebar/.test(e.name) || e.cls.some((c) => /(^|[-_])sidebar([-_]|$)/i.test(c)))) return false;
    const t = n.text.trim();
    const w = wordCount(t);
    if (w < 1 || w > 3 || !/\p{L}/u.test(n.lit) || /[.!?:…]$/.test(t)) return false;
    const b = blockOf(n);
    const parent = b.parent;
    if (!parent) return false;
    const els = parent.kids.filter((k): k is El => k.t === 'el');
    const sib = els[els.indexOf(b) + 1];
    return !!sib && !sib.inline && !isTextBlock(sib) && nodes.some((m) => m.idx > n.idx && !m.attr && m.chain.includes(sib));
  };
  // untagged headings first, so that the lines around them see a heading
  const uhs = nodes.map(untaggedHeading);
  for (const n of nodes) {
    const uh = uhs[n.idx];
    if (uh) {
      n.strong = uh;
      n.pos = null;
    }
  }
  for (const n of nodes) {
    if (!n.pos) continue;
    const prev = neighbour(n, -1);
    const next = neighbour(n, 1);
    const words = wordCount(n.text);
    const c = casc[n.idx];
    const before = headingNode(next);
    const after = headingNode(prev);
    let r: Role;
    if (!before && sidebarGroupLabel(n)) {
      n.strong = 'label';
      continue;
    }
    switch (n.pos) {
      case 'kicker':
        // above a heading: eyebrow; a lone line under a heading: its tagline; opening a panel: the panel's heading
        if (before) r = 'eyebrow';
        else if (after && !next) r = 'subtitle';
        else if (leadsPanel(n)) r = 'h3';
        else r = 'label';
        break;
      case 'small':
        r = before && words <= 8 ? 'eyebrow' : after ? 'subtitle' : 'caption';
        break;
      case 'muted':
        r = before && words <= 8 ? 'eyebrow' : after ? 'subtitle' : words <= 8 ? 'caption' : 'body';
        break;
      default: {
        const styled = !!(c.small || c.caps || c.tracked || isCaps(n.lit));
        if (before && styled && words <= 8) r = 'eyebrow';
        else if (after) {
          const lone = !next || (next.pos === null && next.strong !== 'body') || headingNode(next);
          r = c.small || c.muted || lone ? 'subtitle' : 'body';
        } else r = 'body';
      }
    }
    n.strong = r;
  }

  // ---- context refinement (nav, tabs, menus, breadcrumbs, dialogs, empty states, toasts) ----------------------
  const nearestBox = (n: RNode, set: ReadonlySet<Box>): El | null => {
    for (let j = n.chain.length - 1; j >= 0; j--) if (n.chain[j].box && set.has(n.chain[j].box as Box)) return n.chain[j];
    return null;
  };
  const NAVISH: ReadonlySet<Box> = new Set<Box>(['nav', 'tablist', 'menu', 'breadcrumb']);
  for (const n of nodes) {
    let r = n.strong as Role;
    if (!n.attr) {
      const nb = nearestBox(n, NAVISH);
      if (nb) {
        const b = nb.box;
        if (b === 'nav' && (r === 'link' || r === 'button' || r === 'list_item' || (ctx.fw === 'swift' && /^(body|label|subtitle|caption)$/.test(r)))) r = 'nav';
        else if (b === 'tablist' && /^(link|button|list_item|body)$/.test(r)) r = 'tab';
        else if (b === 'menu' && /^(link|button|list_item|body)$/.test(r)) r = 'menu_item';
        else if (b === 'breadcrumb' && /^(link|button|list_item|body|nav)$/.test(r)) r = 'breadcrumb';
      }
      // links listed in a page header (a menu built from ul > li > a) are navigation
      if (!nb && r === 'link') {
        const hi = n.chain.findIndex((e) => e.box === 'header');
        if (hi >= 0 && n.chain.some((e, j) => j > hi && e.box === 'list')) r = 'nav';
      }
      const sb = nearestBox(n, STATEISH);
      if (sb) {
        if (sb.box === 'dialog') {
          if (isHeading(r) && r !== 'empty_title') r = 'dialog_title';
          else if (r === 'body' || r === 'subtitle') r = 'dialog_body';
        } else if (sb.box === 'empty') {
          if (isHeading(r) && r !== 'dialog_title') r = 'empty_title';
          else if (r === 'body' || r === 'subtitle' || r === 'caption') r = 'empty_body';
        } else if (sb.box === 'toast') {
          if (!/^(button|link|aria|alt|tooltip|error)$/.test(r)) r = 'toast';
        } else if (sb.box === 'alert' && (sb.attrs.role === 'alert' || sb.attrs.variant === 'destructive')) {
          if (r === 'body' || r === 'subtitle' || r === 'caption') r = 'error';
        }
      }
    }
    if (ctx.fw === 'swift' && r === 'body' && !n.attr && !n.chain.some((e) => e.name === '#value') && nearestBox(n, new Set<Box>(['list']))) r = 'list_item';
    if (TRACK_C_ROLES.has(r)) r = 'other';
    n.role = r;
  }
  // A hand-built empty state with no heading: its first short line is the title, the rest its body.
  {
    const seen = new Set<number>();
    for (const n of nodes) {
      const eb = nearestBox(n, STATEISH);
      if (!eb || eb.box !== 'empty' || seen.has(eb.id)) continue;
      seen.add(eb.id);
      const ms = nodes.filter((m) => !m.attr && m.chain.includes(eb) && !CONTEXT_ROLES.has(m.role) && nearestBox(m, STATEISH) === eb);
      const text = ms.filter((m) => /^(body|empty_body|caption|label|subtitle|helper)$/.test(m.role));
      if (ms.some((m) => isHeading(m.role)) || text.length < 1 || text[0] !== ms[0] || wordCount(text[0].text) > 8) continue;
      text[0].role = 'empty_title';
      for (const m of text.slice(1)) if (m.role !== 'helper' && m.role !== 'label') m.role = 'empty_body';
    }
  }
  // A second line of text inside a button (a description under its label) is a caption; the first stays the button.
  {
    const firstIn = new Set<string>();
    for (const n of nodes) {
      if (n.attr || n.role !== 'button' || casc[n.idx].srOnly || mergeSty([...n.local].reverse()).rounded) continue;
      let b: El | null = null;
      for (let j = n.chain.length - 1; j >= 0; j--) {
        if (strongOf(n.chain[j]) === 'button') {
          b = n.chain[j];
          break;
        }
      }
      if (!b) continue;
      // texts in different branches of a condition inside the button are alternatives, not two lines
      const bi = n.chain.indexOf(b);
      const branch = n.chain.slice(bi + 1).find((e) => e.cond);
      const key = `${b.id}:${branch ? branch.id : ''}`;
      if (branch && [...firstIn].some((k) => k.startsWith(`${b.id}:`) && k !== key && k !== `${b.id}:`)) continue;
      if ([...firstIn].some((k) => k.startsWith(`${b.id}:`))) n.role = 'caption';
      else firstIn.add(key);
    }
  }

  // ---- one node per source span ----------------------------------------------------------------------------------
  // Labels key on loc, so no two nodes may share one. Spans repeat only in expanded array copies: text the copies
  // share (a constant button label, an alt that repeats the title) has one span. Visible nodes choose first, in
  // reading order, then context nodes; each takes its first free span, and a node with none left is dropped, so
  // copy text appears once, in the first copy.
  if (nodes.some((n) => n.alts.length) || new Set(nodes.map((n) => `${n.s}:${n.e}`)).size < nodes.length) {
    const ctxNode = (n: RNode): boolean => CONTEXT_ROLES.has(n.role) || !!casc[n.idx].srOnly;
    const used = new Set<string>();
    const keep = new Set<RNode>();
    for (const n of [...nodes.filter((m) => !ctxNode(m)), ...nodes.filter(ctxNode)]) {
      const free = [[n.s, n.e] as [number, number], ...n.alts].find(([a, b]) => !used.has(`${a}:${b}`));
      if (!free) continue;
      [n.s, n.e] = free;
      used.add(`${n.s}:${n.e}`);
      keep.add(n);
    }
    const kept = nodes.filter((n) => keep.has(n));
    nodes.splice(0, nodes.length, ...kept);
    nodes.forEach((n, j) => {
      n.idx = j;
    });
    casc = nodes.map(cascade);
  }

  const nodesUnder = new Map<number, number[]>();
  for (const n of nodes) for (const e of n.chain) {
    let a = nodesUnder.get(e.id);
    if (!a) nodesUnder.set(e.id, (a = []));
    a.push(n.idx);
  }
  const under = (e: El): number[] => nodesUnder.get(e.id) ?? [];
  // ---- control containers ---------------------------------------------------------------------------------------
  const ctrlMemo = new Map<number, number>();
  const controlsIn = (e: El): number => {
    let v = ctrlMemo.get(e.id);
    if (v === undefined) {
      v = (e.control ? 1 : 0) + e.kids.reduce((a, k) => a + (k.t === 'el' ? controlsIn(k) : 0), 0);
      ctrlMemo.set(e.id, v);
    }
    return v;
  };
  const boundary = (e: El): boolean => !!e.box || e === root || /^(main|article|body|html)$/.test(e.name);
  const markControls = (e: El): void => {
    if (e.control) {
      let c = e;
      // a field is small: stop before a parent that holds headings, actions or more than five nodes
      const fieldLike = (p: El): boolean => {
        const u = under(p);
        return u.length <= 5 && !u.some((x) => isHeading(nodes[x].role) || ((nodes[x].role === 'button' || nodes[x].role === 'link') && !under(c).includes(x)));
      };
      while (c.parent && !boundary(c.parent) && controlsIn(c.parent) === 1 && fieldLike(c.parent)) c = c.parent;
      if (!c.box) c.box = 'control';
    }
    for (const k of e.kids) if (k.t === 'el') markControls(k);
  };
  markControls(root);
  // secondary text inside a field is helper text
  for (const n of nodes) {
    if (n.attr || !/^(caption|body|subtitle)$/.test(n.role)) continue;
    for (let j = n.chain.length - 1; j >= 0; j--) {
      const b = n.chain[j].box;
      if (!b) continue;
      if (b === 'control') n.role = 'helper';
      break;
    }
  }

  // ---- repeated subtrees and card promotion -------------------------------------------------------------------
  const repeatedEls: El[] = [];
  const visitRep = (e: El): void => {
    const groups = new Map<string, El[]>();
    for (const k of e.kids) {
      if (k.t !== 'el') continue;
      const u = under(k);
      if (u.length >= 2) {
        const sig = u.map((i) => nodes[i].role).join(',');
        let g = groups.get(sig);
        if (!g) groups.set(sig, (g = []));
        g.push(k);
      }
      if (k.tmpl && u.length >= 2) repeatedEls.push(k);
    }
    for (const g of groups.values()) if (g.length >= 3) repeatedEls.push(...g);
    for (const k of e.kids) if (k.t === 'el') visitRep(k);
  };
  visitRep(root);
  const repSet = new Set<number>();
  for (const r of repeatedEls) {
    for (const i of under(r)) repSet.add(i);
    const listy = (() => {
      for (let p: El | undefined = r.parent; p; p = p.parent) if (p.box && LISTISH.has(p.box)) return true;
      return false;
    })();
    if (!r.box && !listy && !/^(li|tr|option|dt|dd)$/.test(r.name) && !/Row$|Item$/.test(r.name)) r.box = 'card';
  }

  // ---- flags ----------------------------------------------------------------------------------------------------
  for (const n of nodes) {
    const f = n.flags;
    const detached = !!n.attr && /^(tooltip|aria|alt)$/.test(n.attr);
    const c: Sty = detached ? { srOnly: casc[n.idx].srOnly } : casc[n.idx];
    const lsty: Sty = detached ? {} : mergeSty([...n.local].reverse());
    if (c.caps || isCaps(n.lit)) f.add('caps');
    if (!f.has('caps') && (c.titleCase || isTitleCase(n.lit))) f.add('title_case');
    if (c.tracked) f.add('tracked');
    if (c.small) f.add('small');
    if (c.mono || (!detached && n.chain.some((e) => e.intrinsic && MONO_TAGS.has(e.name)))) f.add('mono');
    const badgeComp = !detached && n.local.some((e) => !e.intrinsic && /(Badge|Chip|Pill|Tag)$/.test(e.name));
    if (lsty.rounded || badgeComp) f.add('pill');
    if (n.dot || hasDotLead(n.text)) f.add('dot');
    if (n.icon) f.add('icon');
    if (isEmojiLead(n.text)) f.add('emoji_lead');
    if (n.arrow || hasArrowTail(n.text)) f.add('arrow_tail');
    if (n.dyn) f.add('dynamic');
    if (n.bound || n.dotBound || n.local.some((e) => e.bound)) f.add('stateful');
    if (n.alt || n.cond || n.chain.some((e) => e.cond)) f.add('conditional');
    for (const e of n.chain) {
      switch (e.box) {
        case 'nav': case 'breadcrumb': f.add('in_nav'); break;
        case 'table': f.add('in_table'); break;
        case 'form': f.add('in_form'); break;
        case 'list': case 'menu': f.add('in_list'); break;
        case 'dialog': f.add('in_dialog'); f.add('in_state'); break;
        case 'state': case 'empty': case 'toast': case 'alert': f.add('in_state'); break;
        case 'header': f.add('in_header'); break;
        case 'footer': f.add('in_footer'); break;
        case 'card': f.add('in_card'); break;
        case 'control': f.add('in_control'); break;
        default: break;
      }
      if (e.tmpl) f.add('in_list');
    }
    if (repSet.has(n.idx)) f.add('repeated');
    if (c.srOnly) f.add('sr_only');
    if (lsty.truncated) f.add('truncated');
  }

  // ---- clusters -------------------------------------------------------------------------------------------------
  const containerOf = (n: RNode): { el: El; kind: ClusterKind } | null => {
    for (let j = n.chain.length - 1; j >= 0; j--) {
      const b = n.chain[j].box;
      if (b === 'card') return { el: n.chain[j], kind: 'card' };
      if (b && STATEISH.has(b)) return { el: n.chain[j], kind: 'state' };
    }
    for (let j = n.chain.length - 1; j >= 0; j--) if (n.chain[j].box === 'control') return { el: n.chain[j], kind: 'control' };
    // lists nest (nav > ul, table > tbody): the outermost list is the cluster
    for (let j = 0; j < n.chain.length; j++) {
      const b = n.chain[j].box;
      if (b && LISTISH.has(b)) {
        for (let x = j + 1; x < n.chain.length; x++) if (n.chain[x].box === 'section') return { el: n.chain[x], kind: 'list' };
        return { el: n.chain[j], kind: 'list' };
      }
    }
    return null;
  };
  const isAncestor = (a: El, b: El): boolean => {
    for (let p: El | undefined = b; p; p = p.parent) if (p === a) return true;
    return false;
  };
  const clusterKey: string[] = new Array(nodes.length);
  const clusterKind: ClusterKind[] = new Array(nodes.length);
  const containers = nodes.map(containerOf);
  let i = 0;
  let seg = 0;
  while (i < nodes.length) {
    const cont = containers[i];
    if (cont) {
      clusterKey[i] = `c${cont.el.id}`;
      clusterKind[i] = cont.kind;
      i++;
      continue;
    }
    // a run of free nodes whose group parents nest
    let j = i + 1;
    while (j < nodes.length && !containers[j]) {
      const a = nodes[j - 1].gp as El;
      const b = nodes[j].gp as El;
      if (!(a === b || isAncestor(a, b) || isAncestor(b, a))) break;
      if (nodes[j - 1].chain[1] !== nodes[j].chain[1]) break;
      j++;
    }
    const run = nodes.slice(i, j);
    for (const [kind, members] of segmentRun(run)) {
      seg++;
      for (const m of members) {
        clusterKey[m.idx] = `f${seg}`;
        clusterKind[m.idx] = kind;
      }
    }
    i = j;
  }
  const outClusters: OutCluster[] = [];
  const kOf: number[] = new Array(nodes.length);
  let prevKey = '';
  const members: RNode[][] = [];
  for (const n of nodes) {
    if (clusterKey[n.idx] !== prevKey) {
      outClusters.push({ k: outClusters.length, kind: clusterKind[n.idx], s: 0 });
      members.push([]);
      prevKey = clusterKey[n.idx];
    }
    kOf[n.idx] = outClusters.length - 1;
    members[members.length - 1].push(n);
  }
  for (let k = 0; k < outClusters.length; k++) {
    const ms = members[k];
    let lca = ms[0].chain;
    for (const m of ms) {
      let x = 0;
      while (x < lca.length && x < m.chain.length && lca[x] === m.chain[x]) x++;
      lca = lca.slice(0, x);
    }
    const top = lca[lca.length - 1];
    outClusters[k].s = top && top !== root ? top.s : ms[0].s;
  }

  // ---- section_first / section_last ------------------------------------------------------------------------------
  const SCOPE_BOX: ReadonlySet<Box> = new Set<Box>(['section', 'card', 'dialog', 'state', 'empty', 'toast', 'alert', 'header', 'footer', 'form', 'nav', 'root']);
  for (const h of nodes) {
    const lv = HEADING_LEVEL[h.role];
    if (lv === undefined || h.flags.has('sr_only')) continue;
    let scope: El = root;
    let scopeAt = 0;
    for (let j = h.chain.length - 2; j >= 0; j--) {
      const e = h.chain[j];
      if ((e.box && SCOPE_BOX.has(e.box)) || e.parent === root) {
        scope = e;
        scopeAt = j;
        break;
      }
    }
    // a sibling heading of the same or higher level bounds the section: the scope is the subtree holding this
    // heading but not the other (footer columns, sibling cards)
    for (let j = h.chain.length - 2; j > scopeAt; j--) {
      const e = h.chain[j];
      const other = under(e).some((x) => x !== h.idx && (HEADING_LEVEL[nodes[x].role] ?? 99) <= lv && !nodes[x].flags.has('sr_only'));
      if (other) {
        scope = h.chain[j + 1];
        break;
      }
    }
    const sec: RNode[] = [];
    for (let x = h.idx + 1; x < nodes.length; x++) {
      const m = nodes[x];
      if (!m.chain.includes(scope)) break;
      const ml = HEADING_LEVEL[m.role];
      if (ml !== undefined && ml <= lv && !m.flags.has('sr_only')) {
        while (sec.length && sec[sec.length - 1].role === 'eyebrow') sec.pop();
        break;
      }
      sec.push(m);
    }
    const real = sec.filter((m) => !CONTEXT_ROLES.has(m.role) && !m.flags.has('sr_only'));
    if (real.length) {
      real[0].flags.add('section_first');
      real[real.length - 1].flags.add('section_last');
    }
  }

  const out: OutNode[] = nodes.map((n) => {
    const { text, vw } = varWords(n.text);
    return {
      i: n.idx,
      k: kOf[n.idx],
      role: n.role,
      text,
      flags: FLAGS.filter((f) => n.flags.has(f)),
      ...(vw.length ? { vw } : {}),
      s: n.s,
      e: n.e,
    };
  });
  return { nodes: out, clusters: outClusters };
}

/** Split a run of free nodes into header / actions / body clusters. */
function segmentRun(run: RNode[]): [ClusterKind, RNode[]][] {
  const out: [ClusterKind, RNode[]][] = [];
  const interactive = (r: Role): boolean => r === 'button' || r === 'link';
  const ctxRole = (n: RNode): boolean => CONTEXT_ROLES.has(n.role) || n.flags.has('sr_only');
  const rest = (ns: RNode[]): void => {
    // maximal blocks of interactive vs other nodes; context nodes join the current block
    let cur: RNode[] = [];
    let curKind: ClusterKind | null = null;
    for (const n of ns) {
      if (ctxRole(n) && curKind) {
        cur.push(n);
        continue;
      }
      const kind: ClusterKind = interactive(n.role) ? 'actions' : 'body';
      if (curKind && kind !== curKind) {
        out.push([curKind, cur]);
        cur = [];
      }
      curKind = kind;
      cur.push(n);
    }
    if (cur.length) out.push([curKind ?? 'body', cur]);
  };
  const LEAD = new Set<Role>(['eyebrow', 'badge', 'caption']);
  const FOLLOW = new Set<Role>(['subtitle', 'badge', 'caption', 'button', 'link', 'helper', 'eyebrow']);
  let i = 0;
  while (i < run.length) {
    let h = -1;
    for (let j = i; j < run.length; j++) if (HEADING_LEVEL[run[j].role] !== undefined) {
      h = j;
      break;
    }
    if (h < 0) {
      rest(run.slice(i));
      break;
    }
    let start = h;
    while (start > i && (LEAD.has(run[start - 1].role) || ctxRole(run[start - 1]))) start--;
    if (start > i) rest(run.slice(i, start));
    let end = h + 1;
    while (end < run.length) {
      const m = run[end];
      if (HEADING_LEVEL[m.role] !== undefined || (m.flags.has('repeated') && !run[h].flags.has('repeated'))) break;
      if (!(FOLLOW.has(m.role) || ctxRole(m))) break;
      // an eyebrow that leads the next heading belongs to it
      if (m.role === 'eyebrow') break;
      end++;
    }
    out.push(['header', run.slice(start, end)]);
    i = end;
  }
  return out;
}
