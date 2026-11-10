// SwiftUI front end (web-tree-sitter + tree-sitter-swift). Reads every `some View` property and function body
// (previews excluded) as view-builder code. Modifiers become style evidence or extra nodes (.navigationTitle,
// .help, .alert, .badge, ...). A ForEach or List over an array literal in the same file is expanded once per element.

import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { Language, Parser } from 'web-tree-sitter';
import type { Node as TSNode } from 'web-tree-sitter';
import type { El, Kid, LV, Slot, Txt } from './ir.ts';
import { ALT_JOIN, ALT_SEP, MAX_EXPAND, el, hasSubs, isFigure, keyLastSegment, looksLikeCopy, markCopy, txt, varText } from './ir.ts';
import { decorateComponent, propRole } from './roles.ts';
import type { Role } from './schema.ts';

let parserPromise: Promise<Parser> | null = null;

export function swiftWasmPath(): string {
  const req = createRequire(import.meta.url);
  const pkg = req.resolve('@repomix/tree-sitter-wasms/package.json');
  return join(dirname(pkg), 'out', 'tree-sitter-swift.wasm');
}

export async function swiftParser(): Promise<Parser> {
  if (!parserPromise) {
    parserPromise = (async () => {
      await Parser.init();
      const lang = await Language.load(swiftWasmPath());
      const p = new Parser();
      p.setLanguage(lang);
      return p;
    })();
  }
  return parserPromise;
}

// ---------------------------------------------------------------------------------------------------------------

const named = (n: TSNode): TSNode[] => n.namedChildren.filter((c): c is TSNode => c !== null);

interface Arg {
  label: string | null;
  value: TSNode;
}
interface Call {
  name: string;
  node: TSNode;
  args: Arg[];
  closures: { label: string | null; node: TSNode }[];
}

function parseSuffix(suffix: TSNode | null): { args: Arg[]; closures: { label: string | null; node: TSNode }[] } {
  const args: Arg[] = [];
  const closures: { label: string | null; node: TSNode }[] = [];
  if (!suffix) return { args, closures };
  let pendingLabel: string | null = null;
  for (const c of named(suffix)) {
    if (c.type === 'value_arguments') {
      for (const a of named(c)) {
        if (a.type !== 'value_argument') continue;
        const lab = a.childForFieldName('name');
        const val = a.childForFieldName('value');
        if (val) args.push({ label: lab ? lab.text : null, value: val });
      }
    } else if (c.type === 'simple_identifier') pendingLabel = c.text;
    else if (c.type === 'lambda_literal') {
      closures.push({ label: pendingLabel, node: c });
      pendingLabel = null;
    }
  }
  return { args, closures };
}

function asCall(n: TSNode): Call | null {
  if (n.type !== 'call_expression') return null;
  const ch = named(n);
  const callee = ch[0];
  const suffix = ch.find((c) => c.type === 'call_suffix') ?? null;
  if (!callee || callee.type !== 'simple_identifier') return null;
  const { args, closures } = parseSuffix(suffix);
  return { name: callee.text, node: n, args, closures };
}

interface Mod {
  name: string;
  args: Arg[];
  closures: { label: string | null; node: TSNode }[];
  node: TSNode;
}

/** Strip trailing modifier calls: Text("a").font(.caption) → base Text("a"), mods [font]. */
function peel(n: TSNode): { base: TSNode; mods: Mod[] } {
  const mods: Mod[] = [];
  let cur = n;
  for (;;) {
    if (cur.type === 'call_expression') {
      const ch = named(cur);
      const callee = ch[0];
      if (callee && callee.type === 'navigation_expression') {
        const target = callee.childForFieldName('target');
        const suf = callee.childForFieldName('suffix');
        const nm = suf ? named(suf).find((c) => c.type === 'simple_identifier') : null;
        if (target && nm) {
          const { args, closures } = parseSuffix(ch.find((c) => c.type === 'call_suffix') ?? null);
          mods.push({ name: nm.text, args, closures, node: cur });
          cur = target;
          continue;
        }
      }
    }
    break;
  }
  return { base: cur, mods: mods.reverse() };
}

// ---------------------------------------------------------------------------------------------------------------
// Text

interface TV {
  text: string;
  lit: string;
  dyn: boolean;
  alt: boolean;
  bound: boolean;
  spans?: [number, number][]; // substituted array literals (see "Literal arrays")
  near?: [number, number][];
}

// ---------------------------------------------------------------------------------------------------------------
// Literal arrays
//
// ForEach(xs) { x in … } and List(xs) { x in … }, where xs is an array literal or a `let` in this file bound to one
// (by a name declared once in the file), of string literals or of initialisers with labelled literal arguments
// (Row(title: "…")), are expanded once per element as in jsx.ts. While a copy is built, SUBST binds the closure's
// parameter (or $0) to the element.

let SUBST: Map<string, Slot> | null = null;
let ARRAYS: Map<string, TSNode[]> = new Map(); // let NAME = [ … ] declarations, and every other declaration of NAME

function swLit(n: TSNode, depth: number): LV | null {
  switch (n.type) {
    case 'line_string_literal': case 'multi_line_string_literal': {
      if (named(n).some((c) => c.type === 'interpolated_expression')) return null;
      const [s, e] = litSpan(n);
      return { k: 'str', value: stringParts(n).text, s, e };
    }
    case 'integer_literal': case 'real_literal': return { k: 'num', value: Number(n.text), raw: n.text, s: n.startIndex, e: n.endIndex };
    case 'boolean_literal': return { k: 'bool', value: n.text === 'true' };
    case 'array_literal': return depth <= 2 ? swArray(n, depth) : null;
    case 'call_expression': {
      const loc = asCall(n);
      if (loc && LOCALISE.test(loc.name)) {
        // String(localized: "…") and the like are copy, as in textOf; a NSLocalizedString key is not
        const a = loc.args.find((x) => x.label === null || x.label === 'localized');
        const v = a ? swLit(a.value, depth) : null;
        return v && v.k === 'str' && (loc.name !== 'NSLocalizedString' || looksLikeCopy(v.value)) ? v : null;
      }
      // Row(title: "…", note: "…") or .init(title: "…")
      const ch = named(n);
      const callee = ch[0];
      if (!callee || !((callee.type === 'simple_identifier' && /^[A-Z]/.test(callee.text)) || (callee.type === 'prefix_expression' && callee.text === '.init'))) return null;
      const { args, closures } = parseSuffix(ch.find((c) => c.type === 'call_suffix') ?? null);
      const props = new Map<string, Slot>();
      let open = closures.length > 0;
      for (const a of args) {
        if (!a.label) {
          open = true;
          continue;
        }
        const v = swLit(a.value, depth + 1);
        if (v && (v.k === 'str' || v.k === 'num')) v.p = [a.value.parent?.startIndex ?? a.value.startIndex, a.value.endIndex];
        props.set(a.label, v);
      }
      return { k: 'obj', props, open };
    }
    default: return n.text === 'nil' ? { k: 'nil' } : null;
  }
}

function swArray(n: TSNode, depth: number): { k: 'arr'; elems: LV[] } | null {
  if (!named(n).some((x) => !/comment/.test(x.type))) return null; // [] is filled in later
  const elems: LV[] = [];
  for (const x of named(n)) {
    if (/comment/.test(x.type)) continue;
    const v = swLit(x, depth + 1);
    if (!v || (v.k !== 'str' && v.k !== 'obj')) return null;
    if (v.k === 'str') v.p = [x.startIndex, x.endIndex];
    else v.at = [x.startIndex, x.endIndex];
    elems.push(v);
  }
  return { k: 'arr', elems };
}

/** A substituted name (x, $0) or a property path from one (x.title). Null when the expression is neither. */
function swResolve(n: TSNode | null): { v: Slot } | null {
  if (!n || !SUBST) return null;
  if (n.type === 'simple_identifier') return SUBST.has(n.text) ? { v: SUBST.get(n.text) } : null;
  if (n.type === 'navigation_expression') {
    const base = swResolve(n.childForFieldName('target'));
    if (!base) return null;
    const suf = n.childForFieldName('suffix');
    const key = suf ? named(suf).find((c) => c.type === 'simple_identifier')?.text ?? null : null;
    const b = base.v;
    if (b === undefined || b === null) return { v: b };
    if (b.k === 'obj' && key) return { v: b.props.has(key) ? b.props.get(key) : b.open ? null : undefined };
    return { v: null };
  }
  return null;
}

/** The elements a ForEach or List data argument stands for, when it is a literal array. */
function swArrayOf(n: TSNode): LV[] | null {
  if (n.type === 'array_literal') return swArray(n, 0)?.elems ?? null;
  const r = swResolve(n);
  if (r) return r.v && r.v.k === 'arr' ? r.v.elems : null;
  let name: string | null = null;
  if (n.type === 'simple_identifier') name = n.text;
  else if (n.type === 'navigation_expression') {
    const target = n.childForFieldName('target');
    const suf = n.childForFieldName('suffix');
    if (target && /^(self|Self|[A-Z]\w*)$/.test(target.text)) name = suf ? named(suf).find((c) => c.type === 'simple_identifier')?.text ?? null : null;
  }
  const decls = name ? ARRAYS.get(name) ?? [] : [];
  return decls.length === 1 && decls[0].type === 'array_literal' ? swArray(decls[0], 0)?.elems ?? null : null;
}

/** Record `let NAME = [ … ]` declarations; any other declaration of the same name makes it ambiguous. */
function collectArrays(root: TSNode): Map<string, TSNode[]> {
  const out = new Map<string, TSNode[]>();
  const walk = (n: TSNode): void => {
    if (n.type === 'property_declaration') {
      const pat = named(n).find((c) => c.type === 'pattern');
      const id = pat && named(pat).length === 1 && named(pat)[0].type === 'simple_identifier' ? named(pat)[0].text : null;
      if (id) {
        const isLet = named(n).some((c) => c.type === 'value_binding_pattern' && c.text === 'let');
        const arr = named(n).find((c) => c.type === 'array_literal');
        let a = out.get(id);
        if (!a) out.set(id, (a = []));
        a.push(isLet && arr ? arr : n);
      }
    } else if (n.type === 'lambda_parameter' || n.type === 'parameter') {
      const id = named(n).find((c) => c.type === 'simple_identifier');
      if (id) {
        let a = out.get(id.text);
        if (!a) out.set(id.text, (a = []));
        a.push(n);
      }
    }
    for (const c of named(n)) walk(c);
  };
  walk(root);
  return out;
}

function slotTV(v: Slot): TV | null {
  if (v === undefined) return { text: '', lit: '', dyn: false, alt: false, bound: false };
  if (v === null) return null;
  const near: [number, number][] | undefined = 'p' in v && v.p ? [v.p] : undefined;
  if (v.k === 'str') return { text: v.value, lit: v.value, dyn: false, alt: false, bound: false, spans: [[v.s, v.e]], near };
  if (v.k === 'num') return { text: v.raw, lit: v.raw, dyn: false, alt: false, bound: false, spans: [[v.s, v.e]], near };
  if (v.k === 'bool' || v.k === 'nil') return { text: '', lit: '', dyn: false, alt: false, bound: false };
  return null;
}

function identOf(n: TSNode | null): string | null {
  if (!n) return null;
  switch (n.type) {
    case 'simple_identifier': return n.text.replace(/^\$/, '');
    case 'navigation_expression': {
      const suf = n.childForFieldName('suffix');
      const nm = suf ? named(suf).find((c) => c.type === 'simple_identifier') : null;
      return nm ? nm.text : identOf(n.childForFieldName('target'));
    }
    case 'call_expression': {
      const c = asCall(n);
      const a0 = c?.args[0]?.value ?? null;
      if (a0 && (a0.type === 'simple_identifier' || a0.type === 'navigation_expression')) return identOf(a0);
      const ch = named(n)[0];
      return ch ? identOf(ch) : null;
    }
    case 'prefix_expression': return identOf(n.childForFieldName('target'));
    case 'nil_coalescing_expression': return identOf(n.childForFieldName('value'));
    case 'postfix_expression': case 'try_expression': case 'await_expression': case 'force_unwrap_expression': case 'optional_chaining_expression':
      return identOf(named(n)[0] ?? null);
    default: return null;
  }
}

const ESC: Record<string, string> = { n: ' ', t: ' ', r: ' ', '"': '"', "'": "'", '\\': '\\', '0': '' };

function stringParts(n: TSNode): TV {
  let text = '';
  let lit = '';
  let dyn = false;
  const subbed: TV[] = [];
  for (const c of named(n)) {
    if (c.type === 'line_str_text' || c.type === 'multi_line_str_text' || c.type === 'raw_str_part' || c.type === 'raw_str_end_part') {
      text += c.text;
      lit += c.text;
    } else if (c.type === 'str_escaped_char') {
      const t = c.text;
      let v: string;
      if (/^\\u\{/.test(t)) v = String.fromCodePoint(parseInt(t.slice(3, -1), 16));
      else v = ESC[t.slice(1)] ?? t.slice(1);
      text += v;
      lit += v;
    } else if (c.type === 'interpolated_expression' || c.type === 'raw_str_interpolation') {
      const v = c.childForFieldName('value') ?? named(c)[0] ?? null;
      const r = swResolve(v);
      const sub = r ? slotTV(r.v) : null;
      if (sub) {
        // "\(row.title)": a substituted literal is copy
        text += sub.text;
        lit += sub.lit;
        subbed.push(sub);
        continue;
      }
      const tv = v && /^(ternary_expression|nil_coalescing_expression)$/.test(v.type) ? textOf(v) : null;
      text += ` ${tv ? tv.text.trim() : varText(identOf(v))} `;
      lit += ' ';
      dyn = dyn || !tv || tv.dyn;
    }
  }
  const spans = subbed.flatMap((t) => t.spans ?? []);
  const near = subbed.flatMap((t) => t.near ?? []);
  return { text, lit, dyn, alt: false, bound: dyn && lit.trim() === '', ...(spans.length ? { spans, near } : {}) };
}

const LOCALISE = /^(String|NSLocalizedString|LocalizedStringKey|LocalizedStringResource)$/;

function textOf(n0: TSNode | null): TV | null {
  if (!n0) return null;
  const n = n0;
  switch (n.type) {
    case 'line_string_literal': case 'multi_line_string_literal': case 'raw_string_literal': return stringParts(n);
    case 'integer_literal': case 'real_literal': return { text: n.text, lit: n.text, dyn: false, alt: false, bound: false };
    case 'simple_identifier': case 'navigation_expression': {
      const r = swResolve(n);
      const sub = r ? slotTV(r.v) : null;
      if (sub) return sub;
      return { text: ` ${varText(identOf(n))} `, lit: ' ', dyn: true, alt: false, bound: true };
    }
    case 'call_expression': {
      const c = asCall(n);
      if (c && LOCALISE.test(c.name)) {
        const a = c.args.find((x) => x.label === null || x.label === 'localized');
        if (a && a.value.type === 'line_string_literal') {
          const sp = stringParts(a.value);
          if (!sp.dyn && !looksLikeCopy(sp.text) && c.name === 'NSLocalizedString') {
            return { text: ` ${varText(keyLastSegment(sp.text))} `, lit: ' ', dyn: true, alt: false, bound: true };
          }
          return sp;
        }
      }
      if (c && c.name === 'Text') {
        const a = c.args[0];
        return a ? textOf(a.value) : null;
      }
      return { text: ` ${varText(identOf(n))} `, lit: ' ', dyn: true, alt: false, bound: true };
    }
    case 'ternary_expression': {
      const a = textOf(n.childForFieldName('if_true'));
      const b = textOf(n.childForFieldName('if_false'));
      if (!a || !b) return null;
      return joinAlt([a, b]);
    }
    case 'nil_coalescing_expression': {
      const a = textOf(n.childForFieldName('value'));
      const b = textOf(n.childForFieldName('if_nil'));
      if (!a || !b) return null;
      return joinAlt([a, b]);
    }
    case 'additive_expression': {
      const a = textOf(n.childForFieldName('lhs'));
      const b = textOf(n.childForFieldName('rhs'));
      if (!a || !b) return null;
      const spans = [...(a.spans ?? []), ...(b.spans ?? [])];
      const near = [...(a.near ?? []), ...(b.near ?? [])];
      return { text: a.text + b.text, lit: a.lit + b.lit, dyn: a.dyn || b.dyn, alt: a.alt || b.alt, bound: false, ...(spans.length ? { spans, near } : {}) };
    }
    case 'prefix_expression': return null;
    default: return null;
  }
}

function joinAlt(parts: TV[]): TV {
  const f = parts.filter((p) => p.text.trim() !== '');
  if (f.length === 1) return { ...f[0], alt: true, bound: true };
  const spans = f.flatMap((p) => p.spans ?? []);
  const near = f.flatMap((p) => p.near ?? []);
  return {
    text: f.map((p) => p.text.trim()).join(ALT_JOIN), lit: f.map((p) => p.lit.trim()).join(ALT_SEP), dyn: f.some((p) => p.dyn), alt: true, bound: true,
    ...(spans.length ? { spans, near } : {}),
  };
}

function litSpan(n: TSNode): [number, number] {
  if (/string_literal$/.test(n.type)) {
    const parts = named(n);
    if (parts.length) return [parts[0].startIndex, parts[parts.length - 1].endIndex];
  }
  return [n.startIndex, n.endIndex];
}

function mkTxt(n: TSNode, src: string, extra: Partial<Txt> = {}): Txt | null {
  const tv = textOf(n);
  if (!tv || tv.text.trim() === '') return null;
  let [s, e] = litSpan(n);
  while (s < e && /\s/.test(src[s])) s++;
  while (e > s && /\s/.test(src[e - 1])) e--;
  return txt(tv.text, s, e, { lit: tv.lit, dyn: tv.dyn, alt: tv.alt, bound: tv.bound, ...(tv.spans ? { subs: tv.spans, near: tv.near } : {}), ...extra });
}

// ---------------------------------------------------------------------------------------------------------------

const CONTAINERS = new Set([
  'NavigationStack', 'NavigationView', 'NavigationSplitView', 'ScrollView', 'VStack', 'HStack', 'ZStack', 'LazyVStack',
  'LazyHStack', 'LazyVGrid', 'LazyHGrid', 'Grid', 'GridRow', 'Group', 'ViewThatFits', 'GeometryReader',
  'ScrollViewReader', 'ControlGroup', 'ToolbarItem', 'ToolbarItemGroup', 'HSplitView', 'VSplitView', 'TabView',
  'Form', 'List', 'Table', 'TimelineView', 'Canvas', 'WithPerceptionTracking',
]);
const SHAPES = new Set(['Circle', 'Capsule', 'Rectangle', 'RoundedRectangle', 'Ellipse', 'Spacer', 'Divider', 'EmptyView', 'Color', 'AsyncImage', 'ProgressViewStyle']);
const NOT_VIEWS = /^(URL|Date|UUID|Color|Font|Binding|State|Data|NSError|Error|Task|DispatchQueue|NotificationCenter|UserDefaults|Array|Dictionary|Set|Int|Double|Float|Bool|String|CGSize|CGPoint|CGRect|EdgeInsets|Animation|LinearGradient|RadialGradient|AngularGradient|Gradient|Image|UIImage|NSImage)$/;

class SwiftBuilder {
  private src: string;
  constructor(src: string) {
    this.src = src;
  }

  wrap(name: string, n: TSNode, kids: Kid[], opts: Partial<El> = {}): El {
    const e = el(name, n.startIndex, n.endIndex, opts);
    e.kids = kids;
    return e;
  }

  /** One copy of a ForEach or List row closure per array element, each a repeated subtree (null as in jsx.ts). */
  expand(content: { node: TSNode }, elems: LV[]): Kid[] | null {
    const params = named(content.node).find((x) => x.type === 'lambda_function_type');
    const p0 = params ? params.descendantsOfType('lambda_parameter')[0] : null;
    const name = p0 ? named(p0).find((x) => x.type === 'simple_identifier')?.text ?? '$0' : '$0';
    const out: Kid[] = [];
    for (const v of elems.slice(0, MAX_EXPAND)) {
      const saved = SUBST;
      SUBST = new Map(saved ?? []);
      SUBST.set(name, v);
      let kids: Kid[];
      try {
        kids = this.closureKids(content);
      } finally {
        SUBST = saved;
      }
      if (!kids.length) continue;
      const at = v.k === 'obj' ? v.at : v.k === 'str' ? v.p : undefined;
      if (at) for (const k of kids) markCopy(k, at);
      if (kids.length === 1 && kids[0].t === 'el') {
        kids[0].tmpl = true;
        out.push(kids[0]);
      } else out.push(this.wrap('#tmpl', content.node, kids, { tmpl: true }));
    }
    return out.some(hasSubs) ? out : null;
  }

  closureKids(c: { node: TSNode } | undefined): Kid[] {
    if (!c) return [];
    const st = named(c.node).find((x) => x.type === 'statements');
    return st ? this.stmts(st) : [];
  }

  stmts(st: TSNode): Kid[] {
    const out: Kid[] = [];
    for (const s of named(st)) out.push(...this.stmt(s));
    return out;
  }

  stmt(s: TSNode): Kid[] {
    switch (s.type) {
      case 'if_statement': {
        const out: Kid[] = [];
        for (const c of named(s)) {
          if (c.type === 'statements') out.push(...this.stmts(c));
          else if (c.type === 'if_statement') out.push(...this.stmt(c));
        }
        return out.map(cond);
      }
      case 'switch_statement': {
        const out: Kid[] = [];
        for (const entry of named(s)) {
          if (entry.type !== 'switch_entry') continue;
          for (const c of named(entry)) if (c.type === 'statements') out.push(...this.stmts(c));
        }
        return out.map(cond);
      }
      case 'control_transfer_statement': {
        const v = named(s).find((c) => c.type !== 'comment');
        return v ? this.view(v) : [];
      }
      case 'property_declaration': case 'comment': case 'multiline_comment': case 'function_declaration': case 'class_declaration':
        return [];
      default:
        return this.view(s);
    }
  }

  /** Kids for a view expression. */
  view(n: TSNode): Kid[] {
    const { base, mods } = peel(n);
    let core: El | null = null;
    let extra: Kid[] = [];
    if (base.type === 'call_expression' && named(base)[0]?.type === 'additive_expression') {
      core = this.textConcat(base);
    } else if (base.type === 'additive_expression') {
      core = this.textConcat(base);
    } else if (base.type === 'ternary_expression') {
      extra = [...this.view(base.childForFieldName('if_true') as TSNode), ...this.view(base.childForFieldName('if_false') as TSNode)].map(cond);
    } else if (base.type === 'call_expression') {
      const c = asCall(base);
      if (c) core = this.build(c);
    }
    if (!core && extra.length === 0 && mods.length === 0) return [];
    if (!core) {
      core = this.wrap('#view', base, extra);
      extra = [];
    }
    return this.applyMods(core, mods);
  }

  textConcat(n: TSNode): El {
    const pieces: Kid[] = [];
    const walk = (x: TSNode): void => {
      if (x.type === 'additive_expression') {
        const l = x.childForFieldName('lhs');
        const r = x.childForFieldName('rhs');
        if (l) walk(l);
        if (r) walk(r);
        return;
      }
      if (x.type === 'call_expression') {
        const ch = named(x);
        if (ch[0]?.type === 'additive_expression') {
          // `Text("a") + Text("b")` parses as (Text("a") + Text)("b")
          const add = ch[0];
          const l = add.childForFieldName('lhs');
          if (l) walk(l);
          const { args } = parseSuffix(ch.find((c) => c.type === 'call_suffix') ?? null);
          const a = args.find((q) => q.label === null || q.label === 'verbatim');
          const t = a ? mkTxt(a.value, this.src) : null;
          if (t) pieces.push(t);
          return;
        }
        const { base } = peel(x);
        const c = asCall(base);
        if (c && c.name === 'Text') {
          const a = c.args.find((q) => q.label === null || q.label === 'verbatim');
          const t = a ? mkTxt(a.value, this.src) : null;
          if (t) pieces.push(t);
        }
        return;
      }
      const t = mkTxt(x, this.src);
      if (t) pieces.push(t);
    };
    walk(n);
    return this.wrap('Text', n, pieces, { textBlock: true });
  }

  labelSlot(hint: Role, kids: Kid[], n: TSNode): El {
    return this.wrap('#slot', n, kids, { hint, textBlock: true });
  }

  argTxt(c: Call, label: string | null, extra: Partial<Txt> = {}): Txt | null {
    const a = c.args.find((x) => x.label === label);
    if (!a) return null;
    // Text("…") passed as an argument
    const inner = asCall(peel(a.value).base);
    if (inner && inner.name === 'Text') {
      const ia = inner.args.find((x) => x.label === null || x.label === 'verbatim');
      return ia ? mkTxt(ia.value, this.src, extra) : null;
    }
    return mkTxt(a.value, this.src, extra);
  }

  /** View content passed as an argument (header: Text("…")). */
  argView(c: Call, label: string): Kid[] {
    const a = c.args.find((x) => x.label === label);
    if (!a) return [];
    if (/string_literal$/.test(a.value.type)) {
      const t = mkTxt(a.value, this.src);
      return t ? [t] : [];
    }
    return this.view(a.value);
  }

  closure(c: Call, label: string | null): { label: string | null; node: TSNode } | undefined {
    return c.closures.find((x) => x.label === label);
  }

  build(c: Call): El | null {
    const n = c.node;
    const name = c.name;
    const title = (): Txt | null => this.argTxt(c, null);
    const iconEl = (): El[] => {
      const sys = c.args.find((a) => a.label === 'systemImage' || a.label === 'image');
      if (!sys) return [];
      const arrow = /arrow|chevron\.right/.test(sys.value.text);
      return [this.wrap('Image', sys.value, [], { icon: true, arrowIcon: arrow })];
    };
    const allClosures = (): Kid[] => c.closures.flatMap((cl) => this.closureKids(cl));
    const t0 = title();
    switch (name) {
      case 'Text': {
        const a = c.args.find((x) => x.label === null || x.label === 'verbatim');
        const t = a ? mkTxt(a.value, this.src) : null;
        return this.wrap('Text', n, t ? [t] : [], { textBlock: true });
      }
      case 'Label': {
        const kids: Kid[] = [...iconEl()];
        if (t0) kids.push(t0);
        else {
          kids.push(...this.closureKids(c.closures.find((x) => x.label === null || x.label === 'title')));
          if (this.closure(c, 'icon')) kids.unshift(this.wrap('Image', n, [], { icon: true }));
        }
        return this.wrap('Label', n, kids, { textBlock: true });
      }
      case 'Button': {
        const kids: Kid[] = [...iconEl()];
        if (t0) kids.push(t0);
        const lab = this.closure(c, 'label') ?? (c.args.some((a) => a.label === 'action') && !t0 ? c.closures.find((x) => x.label === null) : undefined);
        kids.push(...this.closureKids(lab));
        return this.wrap('Button', n, kids);
      }
      case 'Link': case 'ShareLink': case 'NavigationLink': {
        const kids: Kid[] = [...iconEl()];
        if (t0) kids.push(t0);
        const hasDest = c.args.some((a) => a.label === 'destination' || a.label === 'value' || a.label === 'item');
        const lab = this.closure(c, 'label') ?? (!t0 && hasDest ? c.closures.find((x) => x.label === null) : undefined);
        kids.push(...this.closureKids(lab));
        return this.wrap(name === 'ShareLink' ? 'ShareButton' : name, n, kids);
      }
      case 'Toggle': case 'Stepper': case 'DatePicker': case 'ColorPicker': case 'Slider': case 'Gauge': {
        const kids: Kid[] = [];
        if (t0) kids.push(this.labelSlot('label', [t0], n));
        const lab = this.closure(c, 'label') ?? (t0 ? undefined : c.closures.find((x) => x.label === null));
        if (lab) kids.push(this.labelSlot('label', this.closureKids(lab), lab.node));
        return this.wrap(name, n, kids, { control: true });
      }
      case 'TextField': case 'SecureField': case 'TextEditor': {
        const kids: Kid[] = [];
        const prompt = this.argTxt(c, 'prompt', { attr: 'placeholder' });
        if (t0 && prompt) kids.push(this.labelSlot('label', [t0], n), prompt);
        else if (t0) kids.push({ ...t0, attr: 'placeholder' });
        else if (prompt) kids.push(prompt);
        return this.wrap(name, n, kids, { control: true });
      }
      case 'Picker': {
        const kids: Kid[] = [];
        if (t0) kids.push(this.labelSlot('label', [t0], n));
        const lab = this.closure(c, 'label');
        if (lab) kids.push(this.labelSlot('label', this.closureKids(lab), lab.node));
        const content = this.closure(c, null) ?? this.closure(c, 'content');
        if (content) kids.push(this.wrap('#options', content.node, this.closureKids(content), { box: 'menu' }));
        return this.wrap('Picker', n, kids, { control: true });
      }
      case 'Menu': {
        const kids: Kid[] = [];
        if (t0) kids.push(this.wrap('#button', n, [...iconEl(), t0], { hint: 'button', textBlock: true }));
        const lab = this.closure(c, 'label');
        if (lab) kids.push(this.wrap('#button', lab.node, this.closureKids(lab), { hint: 'button' }));
        const content = this.closure(c, null) ?? this.closure(c, 'content');
        if (content) kids.push(this.wrap('#menu', content.node, this.closureKids(content), { box: 'menu', cond: true }));
        return this.wrap('Menu', n, kids);
      }
      case 'DisclosureGroup': {
        const kids: Kid[] = [];
        if (t0) kids.push(this.wrap('#button', n, [t0], { hint: 'button', textBlock: true }));
        const lab = this.closure(c, 'label');
        if (lab) kids.push(this.wrap('#button', lab.node, this.closureKids(lab), { hint: 'button' }));
        const content = this.closure(c, null) ?? this.closure(c, 'content');
        if (content) kids.push(this.wrap('#content', content.node, this.closureKids(content), { cond: true }));
        return this.wrap('DisclosureGroup', n, kids);
      }
      case 'ProgressView': {
        const kids: Kid[] = [];
        if (t0) kids.push(t0);
        kids.push(...this.closureKids(this.closure(c, 'label') ?? this.closure(c, null)));
        return this.wrap('ProgressView', n, kids, { textBlock: true });
      }
      case 'LabeledContent': {
        const kids: Kid[] = [];
        if (t0) kids.push(this.labelSlot('label', [t0], n));
        const v = this.argTxt(c, 'value');
        if (v) kids.push(this.wrap('#value', n, [v], { textBlock: true }));
        kids.push(...this.closureKids(this.closure(c, null) ?? this.closure(c, 'content')));
        return this.wrap('LabeledContent', n, kids);
      }
      case 'Section': {
        const kids: Kid[] = [];
        const header: Kid[] = t0 ? [t0] : [...this.argView(c, 'header'), ...this.closureKids(this.closure(c, 'header'))];
        if (header.length) kids.push(this.labelSlot('h2', header, n));
        kids.push(...this.closureKids(this.closure(c, null) ?? this.closure(c, 'content')));
        const footer = [...this.argView(c, 'footer'), ...this.closureKids(this.closure(c, 'footer'))];
        if (footer.length) kids.push(this.labelSlot('helper', footer, n));
        return this.wrap('Section', n, kids, { box: 'section' });
      }
      case 'GroupBox': {
        const kids: Kid[] = [];
        if (t0) kids.push(this.labelSlot('h3', [t0], n));
        const lab = this.closure(c, 'label');
        if (lab) kids.push(this.labelSlot('h3', this.closureKids(lab), lab.node));
        kids.push(...this.closureKids(this.closure(c, null) ?? this.closure(c, 'content')));
        return this.wrap('GroupBox', n, kids, { box: 'card' });
      }
      case 'ContentUnavailableView': {
        const kids: Kid[] = [];
        if (t0) kids.push(this.labelSlot('empty_title', [t0], n));
        const lab = this.closure(c, 'label') ?? this.closure(c, null);
        if (lab) kids.push(this.labelSlot('empty_title', this.closureKids(lab), lab.node));
        const d = [...this.argView(c, 'description'), ...this.closureKids(this.closure(c, 'description'))];
        if (d.length) kids.push(this.labelSlot('empty_body', d, n));
        kids.push(...this.closureKids(this.closure(c, 'actions')));
        return this.wrap('ContentUnavailableView', n, kids, { box: 'empty' });
      }
      case 'Alert': {
        // the old Alert(title:message:dismissButton:) value
        const kids: Kid[] = [];
        const t = this.argTxt(c, 'title');
        if (t) kids.push(this.labelSlot('dialog_title', [t], n));
        const m = this.argTxt(c, 'message');
        if (m) kids.push(this.labelSlot('dialog_body', [m], n));
        for (const a of c.args) {
          if (!/Button$/.test(a.label ?? '')) continue;
          const inner = a.value.type === 'call_expression' ? parseSuffix(named(a.value).find((x) => x.type === 'call_suffix') ?? null) : null;
          const ta = inner?.args[0];
          const tt = ta ? this.argTxt({ name: '', node: a.value, args: [{ label: null, value: ta.value }], closures: [] }, null) : null;
          if (tt) kids.push(this.wrap('#button', a.value, [tt], { hint: 'button', textBlock: true }));
        }
        return this.wrap('Alert', n, kids, { box: 'dialog' });
      }
      case 'ForEach': {
        const content = this.closure(c, null) ?? this.closure(c, 'content');
        const data = c.args.find((a) => a.label === null)?.value;
        const elems = data && content ? swArrayOf(data) : null;
        const copies = elems && content ? this.expand(content, elems) : null;
        if (copies) return this.wrap('ForEach', n, copies);
        const kids = this.closureKids(content).map((k) => {
          if (k.t === 'el') {
            k.tmpl = true;
            return k;
          }
          return this.wrap('#tmpl', n, [k], { tmpl: true });
        });
        return this.wrap('ForEach', n, kids);
      }
      case 'Image': {
        const sys = c.args.find((a) => a.label === 'systemName');
        return this.wrap('Image', n, [], { icon: !!sys, arrowIcon: !!sys && /arrow|chevron\.right/.test(sys.value.text) });
      }
      default: break;
    }
    if (CONTAINERS.has(name)) {
      const data = name === 'List' ? c.args.find((a) => a.label === null)?.value : undefined;
      const elems = data && c.closures.length === 1 ? swArrayOf(data) : null;
      const copies = elems ? this.expand(c.closures[0], elems) : null;
      if (copies) return this.wrap(name, n, copies, { box: 'list' });
      const kids = allClosures();
      let box: El['box'];
      if (name === 'Form') box = 'form';
      else if (name === 'List') box = 'list';
      else if (name === 'Table') box = 'table';
      const e = this.wrap(name, n, name === 'List' && c.args.length && c.closures.length ? kids.map((k) => (k.t === 'el' ? { ...k, tmpl: true } : k)) : kids, { box });
      return e;
    }
    if (SHAPES.has(name)) return this.wrap(name, n, []);
    if (!/^[A-Z]/.test(name) || NOT_VIEWS.test(name)) return null;
    // a custom view: labelled text arguments by name, closures as content
    const e = this.wrap(name, n, []);
    decorateComponent(e, undefined);
    const kids: Kid[] = [];
    for (const a of c.args) {
      if (!a.label) continue;
      const r = propRole(name, a.label);
      if (!r) continue;
      const t = this.argTxt(c, a.label, { attr: r });
      if (t) kids.push(t);
      else {
        const v = this.view(a.value);
        if (v.length) kids.push(this.labelSlot(r, v, a.value));
      }
    }
    kids.push(...allClosures());
    e.kids = kids;
    return e;
  }

  applyMods(core: El, mods: Mod[]): Kid[] {
    const before: Kid[] = [];
    const after: Kid[] = [];
    const sty = core.sty;
    for (const m of mods) {
      const a0 = m.args[0]?.value;
      const a0t = a0?.text ?? '';
      if (a0 && a0.type === 'prefix_expression' && /^\.[A-Za-z_]\w*$/.test(a0t)) core.cls.push(`${m.name}${a0t}`);
      switch (m.name) {
        case 'font': {
          const f = a0t.match(/\.(largeTitle|title3|title2|title|headline|subheadline|body|callout|footnote|caption2|caption)\b/);
          const size = a0t.match(/size:\s*(\d+(\.\d+)?)/);
          if (f) {
            const v = f[1];
            const lv = v === 'largeTitle' || v === 'title' ? 1 : v === 'title2' ? 2 : v === 'title3' ? 3 : v === 'headline' ? 4 : 0;
            if (lv) {
              const lit = core.kids.filter((k): k is Txt => k.t === 'txt').map((k) => k.lit).join('');
              if (isFigure(lit)) sty.large = true;
              else core.classHint = (`h${lv}`) as Role;
            }
            sty.small = /^(footnote|caption|caption2)$/.test(v);
            sty.xsmall = /^(caption|caption2)$/.test(v);
          }
          if (size) {
            const px = parseFloat(size[1]);
            sty.small = px <= 13;
            sty.xsmall = px <= 12;
            sty.large = px >= 28;
          }
          if (/monospaced/.test(a0t)) sty.mono = true;
          break;
        }
        case 'fontDesign': case 'monospaced':
          if (m.name === 'monospaced' || /monospaced/.test(a0t)) sty.mono = true;
          break;
        case 'textCase': sty.caps = /uppercase/.test(a0t); break;
        case 'tracking': case 'kerning': sty.tracked = parseFloat(a0t) > 0; break;
        case 'foregroundStyle': case 'foregroundColor': case 'fill': case 'tint': {
          if (/^(\.|Color\.)(secondary|tertiary|quaternary|gray|grey)\b/.test(a0t)) sty.muted = true;
          else if (a0 && !/^(\.|Color\.|Color\(|LinearGradient|\.linearGradient)/.test(a0t)) core.bound = true;
          if (a0 && a0.type === 'ternary_expression') core.bound = true;
          break;
        }
        case 'lineLimit': case 'truncationMode': if (!/nil/.test(a0t)) sty.truncated = true; break;
        case 'clipShape': case 'background': case 'overlay': case 'contentShape': {
          if (/Capsule|\.capsule/.test(m.node.text.slice(0, 200)) && m.name !== 'overlay') sty.rounded = true;
          if (m.name === 'overlay' || m.name === 'background') {
            const viewArg = m.args.find((a) => a.label === null);
            if (viewArg && m.name === 'overlay') after.push(...this.view(viewArg.value));
            for (const cl of m.closures) {
              const kids = this.closureKids(cl);
              if (m.name === 'overlay') after.push(...kids);
              else after.push(...kids.filter((k) => k.t === 'el' && kidsHaveText(k)));
            }
          }
          break;
        }
        case 'frame': {
          const w = m.args.find((a) => a.label === 'width');
          const h = m.args.find((a) => a.label === 'height');
          const wv = w ? parseFloat(w.value.text) : NaN;
          const hv = h ? parseFloat(h.value.text) : NaN;
          if (wv <= 12 && (Number.isNaN(hv) || hv <= 12)) {
            sty.dotSize = true;
            if (core.name === 'Circle') core.dot = true;
          }
          break;
        }
        case 'accessibilityAddTraits': if (/isHeader/.test(a0t) && /^(Text|Label)$/.test(core.name)) core.hint = core.hint ?? 'h2'; break;
        case 'navigationTitle': case 'navigationBarTitle': case 'navigationSubtitle': {
          if (!a0) break;
          const role: Role = m.name === 'navigationSubtitle' ? 'subtitle' : 'h1';
          const t = this.modTxt(a0, role);
          if (t) before.push(this.wrap('#navbar', m.node, [t], { box: 'header' }));
          break;
        }
        case 'help': case 'accessibilityLabel': case 'accessibilityHint': case 'badge': {
          if (!a0) break;
          const role: Role = m.name === 'help' ? 'tooltip' : m.name === 'badge' ? 'badge' : 'aria';
          const t = this.modTxt(a0, role);
          if (t) after.push(t);
          break;
        }
        case 'searchable': {
          const p = m.args.find((a) => a.label === 'prompt');
          const t = p ? this.modTxt(p.value, 'placeholder') : null;
          if (t) before.push(this.wrap('#search', m.node, [t], { control: true }));
          break;
        }
        case 'alert': case 'confirmationDialog': {
          const kids: Kid[] = [];
          const ta = m.args.find((a) => a.label === null || a.label === 'title');
          const t = ta ? this.modTxt(ta.value, 'dialog_title') : null;
          if (t) kids.push(t);
          const actions = m.closures.find((x) => x.label === null || x.label === 'actions');
          const message = m.closures.find((x) => x.label === 'message');
          const msgArg = m.args.find((a) => a.label === 'message');
          if (msgArg) {
            const mt = this.modTxt(msgArg.value, 'dialog_body');
            if (mt) kids.push(mt);
          }
          if (message) kids.push(this.labelSlot('dialog_body', this.closureKids(message), message.node));
          if (actions) kids.push(...this.closureKids(actions));
          if (kids.length) after.push(this.wrap(`#${m.name}`, m.node, kids, { box: 'dialog', cond: true }));
          break;
        }
        case 'sheet': case 'fullScreenCover': case 'popover': case 'inspector': {
          const kids = m.closures.flatMap((cl) => this.closureKids(cl));
          if (kids.length) after.push(this.wrap(`#${m.name}`, m.node, kids, { box: 'dialog', cond: true }));
          break;
        }
        case 'toolbar': {
          const kids = m.closures.flatMap((cl) => this.closureKids(cl));
          if (kids.length) after.push(this.wrap('#toolbar', m.node, kids, { box: 'header' }));
          break;
        }
        case 'tabItem': {
          const kids = m.closures.flatMap((cl) => this.closureKids(cl));
          if (kids.length) before.push(this.wrap('#tabItem', m.node, kids, { hint: 'tab', box: 'nav' }));
          break;
        }
        case 'contextMenu': {
          const kids = m.closures.filter((x) => x.label === null || x.label === 'menuItems').flatMap((cl) => this.closureKids(cl));
          if (kids.length) after.push(this.wrap('#contextMenu', m.node, kids, { box: 'menu', cond: true }));
          break;
        }
        case 'swipeActions': case 'safeAreaInset': {
          const kids = m.closures.flatMap((cl) => this.closureKids(cl));
          if (kids.length) after.push(this.wrap(`#${m.name}`, m.node, kids, { cond: m.name === 'swipeActions' }));
          break;
        }
        default: break;
      }
    }
    if (!before.length && !after.length) return [core];
    const g = el('#group', core.s, core.e);
    g.kids = [...before, core, ...after];
    return [g];
  }

  /** Text from a modifier argument: a string, Text("…"), or an expression. */
  modTxt(a: TSNode, role: Role): Txt | null {
    const inner = asCall(peel(a).base);
    if (inner && inner.name === 'Text') {
      const ia = inner.args.find((x) => x.label === null || x.label === 'verbatim');
      return ia ? mkTxt(ia.value, this.src, { attr: role }) : null;
    }
    return mkTxt(a, this.src, { attr: role });
  }
}

function kidsHaveText(k: Kid): boolean {
  if (k.t === 'txt') return k.text.trim() !== '';
  return k.kids.some(kidsHaveText);
}

function cond(k: Kid): Kid {
  k.cond = true;
  return k;
}

function isViewType(n: TSNode): boolean {
  for (const c of named(n)) {
    if (c.type === 'type_annotation' || c.type === 'opaque_type' || c.type === 'user_type') {
      if (/\bView\b|^Text$|AnyView/.test(c.text) && !/ViewModel|ViewController/.test(c.text)) return true;
    }
  }
  return false;
}

export async function swiftToIR(src: string): Promise<El> {
  const parser = await swiftParser();
  const tree = parser.parse(src);
  if (!tree) throw new Error('tree-sitter returned no tree');
  const b = new SwiftBuilder(src);
  ARRAYS = collectArrays(tree.rootNode);
  SUBST = null;
  const root = el('#root', 0, src.length, { box: 'root' });
  const roots: Kid[] = [];
  const walk = (n: TSNode, inPreview: boolean): void => {
    if (n.type === 'macro_invocation' && /^#?Preview/.test(n.text.replace(/^#/, ''))) return;
    if (n.type === 'class_declaration') {
      const inh = named(n).filter((c) => c.type === 'inheritance_specifier').map((c) => c.text).join(' ');
      if (/PreviewProvider/.test(inh)) return;
    }
    if (n.type === 'property_declaration' && !inPreview) {
      const nameNode = named(n).find((c) => c.type === 'pattern');
      const nm = nameNode?.text ?? '';
      const comp = n.childForFieldName('computed_value');
      if (comp && nm !== 'previews' && (isViewType(n) || nm === 'body')) {
        const st = named(comp).find((c) => c.type === 'statements');
        if (st) {
          const kids = b.stmts(st);
          if (kids.length) roots.push(b.wrap('#body', n, kids));
        }
        return;
      }
    }
    if (n.type === 'function_declaration' && isViewType(n)) {
      const body = named(n).find((c) => c.type === 'function_body');
      const st = body ? named(body).find((c) => c.type === 'statements') : undefined;
      if (st) {
        const kids = b.stmts(st);
        if (kids.length) roots.push(b.wrap('#body', n, kids));
      }
      return;
    }
    for (const c of named(n)) walk(c, inPreview);
  };
  walk(tree.rootNode, false);
  tree.delete();
  root.kids = roots;
  return root;
}
