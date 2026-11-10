// SwiftUI front end (web-tree-sitter + tree-sitter-swift). Reads every `some View` property and function body
// (previews excluded) as view-builder code. Modifiers become style evidence or extra nodes (.navigationTitle,
// .help, .alert, .badge, ...). A ForEach or List over an array literal in the same file is expanded once per element.

import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { Language, Parser } from 'web-tree-sitter';
import type { Node as TSNode } from 'web-tree-sitter';
import type { El, Kid, LV, Opt, Slot, Txt } from './ir.ts';
import {
  ALT_JOIN, ALT_SEP, MAX_EXPAND, el, emptyCond, hasSubs, isFigure, joinRenderings, keyLastSegment, looksLikeCopy, markCall, markCopy, markEmptyKids,
  renderings, stripPUA, txt, varText,
} from './ir.ts';
import { ICON_BUTTON, decorateComponent, propRole } from './roles.ts';
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
        // a closure passed by label (`actions: { … }`, `message: { … }`) is content like a trailing closure
        if (val && val.type === 'lambda_literal') closures.push({ label: lab ? lab.text : null, node: val });
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
  opts?: Opt[]; // complete renderings of a conditional text, '' for an empty branch (two or more)
  lspans?: [number, number][]; // in-file constant literals the text shows (see "In-file constants")
}

const EMPTY_TV: TV = { text: '', lit: '', dyn: false, alt: false, bound: false };
const optsOf = (v: TV): Opt[] => v.opts ?? [{ text: v.text.trim() ? v.text : '', lit: v.text.trim() ? v.lit : '' }];

/** Concatenated text ("…\(x ?? "y")…", `a + b`): conditional parts give every complete rendering, as in jsx.ts. */
function concatTV(parts: TV[]): TV {
  const spans = parts.flatMap((p) => p.spans ?? []);
  const near = parts.flatMap((p) => p.near ?? []);
  const lspans = parts.flatMap((p) => p.lspans ?? []);
  const base: TV = {
    text: parts.map((p) => p.text).join(''), lit: parts.map((p) => p.lit).join(''), dyn: parts.some((p) => p.dyn),
    alt: parts.some((p) => p.alt), bound: false, ...(spans.length ? { spans, near } : {}), ...(lspans.length ? { lspans } : {}),
  };
  if (!parts.some((p) => p.opts)) return base;
  const rs = renderings(parts.map(optsOf));
  if (!rs) return base;
  const j = joinRenderings(rs);
  return { ...base, text: j.text, lit: j.lit, alt: j.text.includes(ALT_SEP), bound: true, opts: rs };
}

// ---------------------------------------------------------------------------------------------------------------
// Literal arrays
//
// ForEach(xs) { x in … } and List(xs) { x in … }, where xs is an array literal or a `let` in this file bound to one
// (by a name declared once in the file), of string literals or of initialisers with labelled literal arguments
// (Row(title: "…")), are expanded once per element as in jsx.ts. While a copy is built, SUBST binds the closure's
// parameter (or $0) to the element.

let SUBST: Map<string, Slot> | null = null;

// Local view-builder functions (`private func point(_ icon: String, _ text: String) -> some View`) are read at each
// call in view position, with their parameters bound to the call's arguments: a literal through SUBST (so the node
// sits at the literal in the call), anything else through EXPRS (read in the caller's scope).
interface Bound {
  node: TSNode;
  subst: Map<string, Slot> | null;
  exprs: Map<string, Bound>;
}
interface LocalFunc {
  params: { ext: string | null; name: string; fn: boolean; def: TSNode | null }[];
  body: TSNode;
}
let EXPRS: Map<string, Bound> = new Map();
let FUNCS: Map<string, LocalFunc | null> = new Map();
let INLINED: Set<string> = new Set();
const INLINING: string[] = [];

function withEnv<T>(b: Bound, f: () => T): T {
  const [s0, e0] = [SUBST, EXPRS];
  SUBST = b.subst;
  EXPRS = b.exprs;
  try {
    return f();
  } finally {
    SUBST = s0;
    EXPRS = e0;
  }
}

/** The argument a parameter name is bound to, when the name is a bound parameter of an inlined function. */
function boundArg(n: TSNode | null): Bound | null {
  return n && n.type === 'simple_identifier' ? EXPRS.get(n.text) ?? null : null;
}

/** Every `func NAME(…) -> some View` in the file; a name declared twice is ambiguous and never inlined. */
function collectFuncs(root: TSNode): Map<string, LocalFunc | null> {
  const out = new Map<string, LocalFunc | null>();
  const walk = (n: TSNode): void => {
    if (n.type === 'function_declaration' && isViewType(n)) {
      const nm = n.childForFieldName('name')?.text ?? '';
      const params: LocalFunc['params'] = [];
      for (let i = 0; i < n.namedChildCount; i++) {
        const c = n.namedChild(i);
        if (!c) continue;
        if (c.type === 'parameter') {
          const ext = c.childForFieldName('external_name')?.text ?? null;
          const id = c.childForFieldName('name');
          params.push({ ext, name: id?.text ?? '', fn: /->/.test(c.text), def: null });
        } else if (n.fieldNameForNamedChild(i) === 'default_value' && params.length) params[params.length - 1].def = c;
      }
      const body = named(n).find((c) => c.type === 'function_body');
      const st = body ? named(body).find((c) => c.type === 'statements') : undefined;
      if (nm && st) out.set(nm, out.has(nm) ? null : { params, body: st });
    }
    for (const c of named(n)) walk(c);
  };
  walk(root);
  return out;
}
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

// ---------------------------------------------------------------------------------------------------------------
// In-file constants
//
// A name in text position that is bound in this file to copy is read through to it when every path yields a literal:
// a `let` (local, a member or `static let`, or at the top level) whose value is a string literal or
// `String(localized:)`, a computed property (`var title: String { switch kind { case .a: "A" … } }`) or a function
// returning String whose every branch is one, joined with ALT. A member is found on `self`, `Self`, a type named in
// the file (`Kind.title`), or a value whose type the file states: a parameter or property annotated with it, or the
// closure parameter of a ForEach (or `.map`) over `Kind.allCases`. Names are resolved by scope: a parameter, closure
// parameter or pattern binding of the same name hides the constant. String literals may interpolate (the
// interpolations stay VAR). Anything else keeps the VAR behaviour.

interface SwBinding {
  node: TSNode; // property_declaration, function_declaration, or the binding identifier (parameters, patterns)
  kind: 'prop' | 'func' | 'param';
  scope: TSNode | null; // the node the binding is visible in (null: a member of `type`, or top level)
  type: string | null; // the type a member belongs to (last name component), null for locals and top level
  top: boolean;
}
let BINDINGS: Map<string, SwBinding[]> = new Map();
let CONST_DEPTH = 0;

/** Last component of a type name node (user_type, type_identifier, navigation or identifier), if it looks like a type. */
function typeNameOf(n: TSNode | null | undefined): string | null {
  if (!n) return null;
  if (n.type === 'type_identifier') return n.text;
  if (n.type === 'user_type') {
    const ids = n.descendantsOfType('type_identifier').filter((x): x is TSNode => x !== null);
    return ids.length ? ids[ids.length - 1].text : null;
  }
  if (n.type === 'optional_type') return typeNameOf(named(n)[0]);
  if (n.type === 'simple_identifier') return /^[A-Z]/.test(n.text) ? n.text : null;
  if (n.type === 'navigation_expression') {
    const nm = n.childForFieldName('suffix');
    const id = nm ? named(nm).find((c) => c.type === 'simple_identifier') : null;
    return id && /^[A-Z]/.test(id.text) ? id.text : null;
  }
  return null;
}

/** The type a class_declaration (struct, enum, class, extension) declares or extends. */
function declTypeName(n: TSNode): string | null {
  return typeNameOf(n.childForFieldName('name'));
}

const BINDING_SCOPE = /^(if_statement|guard_statement|switch_entry|for_statement|while_statement|repeat_while_statement|catch_block|do_statement)$/;

function collectBindings(root: TSNode): Map<string, SwBinding[]> {
  const out = new Map<string, SwBinding[]>();
  const add = (name: string, b: SwBinding): void => {
    let a = out.get(name);
    if (!a) out.set(name, (a = []));
    a.push(b);
  };
  const walk = (n: TSNode, type: string | null): void => {
    const parent = n.parent;
    const member = !!parent && /^(class_body|enum_class_body|protocol_body)$/.test(parent.type);
    const top = !!parent && parent.type === 'source_file';
    if (n.type === 'class_declaration') {
      const t = declTypeName(n);
      for (const c of named(n)) walk(c, t);
      return;
    }
    if (n.type === 'property_declaration') {
      for (const pat of named(n).filter((c) => c.type === 'pattern')) {
        const id = pat.childForFieldName('bound_identifier');
        if (id) add(id.text, { node: n, kind: 'prop', scope: member || top ? null : parent, type: member ? type : null, top });
      }
    } else if (n.type === 'function_declaration') {
      const id = n.childForFieldName('name');
      if (id) add(id.text, { node: n, kind: 'func', scope: member || top ? null : parent, type: member ? type : null, top });
      for (const c of named(n)) if (c.type === 'parameter') {
        const pn = c.childForFieldName('name');
        if (pn) add(pn.text, { node: pn, kind: 'param', scope: n, type: null, top: false });
      }
    } else if (n.type === 'lambda_parameter') {
      const pn = n.childForFieldName('name') ?? named(n).find((c) => c.type === 'simple_identifier');
      let lam: TSNode | null = n.parent;
      while (lam && lam.type !== 'lambda_literal') lam = lam.parent;
      if (pn && lam) add(pn.text, { node: pn, kind: 'param', scope: lam, type: null, top: false });
    } else if (n.type === 'pattern' && parent?.type !== 'property_declaration') {
      // if let x, case .a(let x), for x in, catch let x
      const id = n.childForFieldName('bound_identifier');
      if (id) {
        let sc: TSNode | null = n.parent;
        while (sc && !BINDING_SCOPE.test(sc.type)) sc = sc.parent;
        if (sc?.type === 'guard_statement') sc = sc.parent;
        if (sc) add(id.text, { node: id, kind: 'param', scope: sc, type: null, top: false });
      }
    }
    for (const c of named(n)) walk(c, type);
  };
  walk(root, null);
  return out;
}

const within = (outer: TSNode, n: TSNode): boolean => outer.startIndex <= n.startIndex && n.endIndex <= outer.endIndex;

/** The types enclosing a node, innermost first. */
function enclosingTypes(n: TSNode): string[] {
  const out: string[] = [];
  for (let p: TSNode | null = n.parent; p; p = p.parent) {
    if (p.type === 'class_declaration') {
      const t = declTypeName(p);
      if (t) out.push(t);
    }
  }
  return out;
}

/** The binding a bare name refers to at a node, or null when it is unknown or ambiguous. */
function bindingAt(name: string, at: TSNode): SwBinding | null {
  const cands = BINDINGS.get(name) ?? [];
  const local = cands.filter((b) => b.scope && within(b.scope, at));
  if (local.length) {
    const size = Math.min(...local.map((b) => (b.scope as TSNode).endIndex - (b.scope as TSNode).startIndex));
    const inner = local.filter((b) => (b.scope as TSNode).endIndex - (b.scope as TSNode).startIndex === size);
    return inner.length === 1 && inner[0].node.startIndex < at.startIndex ? inner[0] : null;
  }
  for (const t of enclosingTypes(at)) {
    const ms = cands.filter((b) => b.type === t);
    if (ms.length) return ms.length === 1 ? ms[0] : null;
  }
  const tops = cands.filter((b) => b.top);
  return tops.length === 1 ? tops[0] : null;
}

/** A member of a type (by its last name component), declared once in the file. */
function memberOf(type: string, name: string): SwBinding | null {
  const ms = (BINDINGS.get(name) ?? []).filter((b) => b.type === type);
  return ms.length === 1 ? ms[0] : null;
}

/** `Kind.allCases` (possibly filtered, sorted or mapped): Kind. */
function allCasesType(n: TSNode | null | undefined): string | null {
  if (!n) return null;
  if (n.type === 'navigation_expression') {
    const suf = n.childForFieldName('suffix');
    const id = suf ? named(suf).find((c) => c.type === 'simple_identifier') : null;
    if (id?.text === 'allCases') return typeNameOf(n.childForFieldName('target'));
    return allCasesType(n.childForFieldName('target'));
  }
  if (n.type === 'call_expression') return allCasesType(named(n)[0]);
  return null;
}

/** The type of a value the file states: an annotated parameter or property, or a closure over `Kind.allCases`. */
function valueType(n: TSNode): string | null {
  if (n.type !== 'simple_identifier') return null;
  let b: SwBinding | null;
  if (/^\$0$/.test(n.text)) {
    let lam: TSNode | null = n.parent;
    while (lam && lam.type !== 'lambda_literal') lam = lam.parent;
    if (!lam || named(lam).some((c) => c.type === 'lambda_function_type')) return null;
    b = { node: lam, kind: 'param', scope: lam, type: null, top: false };
  } else b = bindingAt(n.text, n);
  if (!b) return null;
  if (b.kind === 'prop') return typeNameOf(named(b.node).find((c) => c.type === 'type_annotation')?.childForFieldName('name'));
  if (b.kind !== 'param') return null;
  const decl = b.node.parent;
  if (decl?.type === 'parameter') return typeNameOf(named(decl).filter((c) => c.type !== 'simple_identifier').pop());
  if (b.scope?.type === 'lambda_literal') {
    // the closure of ForEach(Kind.allCases) { k in … } or Kind.allCases.map { … }
    let call: TSNode | null = b.scope.parent;
    while (call && call.type !== 'call_expression') call = call.parent;
    if (!call) return null;
    const callee = named(call)[0];
    if (callee?.type === 'navigation_expression') return allCasesType(callee);
    const data = asCall(call)?.args.find((a) => a.label === null)?.value;
    return allCasesType(data);
  }
  return null;
}

/** The value nodes every path of a body yields, or null when a path does something else. */
function bodyValues(st: TSNode): TSNode[] | null {
  const items = named(st).filter((c) => !/comment/.test(c.type));
  if (!items.length) return null;
  const out: TSNode[] = [];
  for (let i = 0; i < items.length; i++) {
    const x = items[i];
    const last = i === items.length - 1;
    if (x.type === 'switch_statement' && last) {
      for (const entry of named(x)) {
        if (entry.type !== 'switch_entry') continue;
        const body = named(entry).find((c) => c.type === 'statements');
        const v = body ? bodyValues(body) : null;
        if (!v) return null;
        out.push(...v);
      }
    } else if (x.type === 'if_statement') {
      // `if c { "a" } else { "b" }` last, or `if c { return "a" }` falling through to what follows
      const v = ifValues(x, last);
      if (!v) return null;
      out.push(...v);
    } else if (x.type === 'control_transfer_statement') {
      const v = x.childForFieldName('result') ?? named(x).find((c) => !/comment/.test(c.type));
      if (!v || !/^return\b/.test(x.text)) return null;
      out.push(v);
      return out;
    } else if (last && items.length === 1) out.push(x);
    else return null;
  }
  return out;
}

function ifValues(x: TSNode, last: boolean): TSNode[] | null {
  const ch = named(x);
  const blocks = ch.filter((c) => c.type === 'statements');
  const elseIf = ch.find((c) => c.type === 'if_statement');
  const out: TSNode[] = [];
  for (const b of blocks) {
    const items = named(b).filter((c) => !/comment/.test(c.type));
    // a branch that falls through must return
    if (!last && items[items.length - 1]?.type !== 'control_transfer_statement') return null;
    const v = bodyValues(b);
    if (!v) return null;
    out.push(...v);
  }
  if (elseIf) {
    const v = ifValues(elseIf, last);
    if (!v) return null;
    out.push(...v);
  } else if (last && blocks.length < 2) return null;
  return out;
}

/** Text of a literal value (a string, String(localized:), a ternary of them, nil, or another constant), or null. */
function literalTV(v: TSNode): TV | null {
  if (v.text === 'nil') return EMPTY_TV;
  if (/^(line|multi_line|raw)_string_literal$/.test(v.type)) {
    const tv = stringParts(v);
    return { ...tv, lspans: [litSpan(v)] };
  }
  if (v.type === 'call_expression') {
    const c = asCall(v);
    if (c && LOCALISE.test(c.name)) {
      const a = c.args.find((x) => x.label === null || x.label === 'localized');
      if (!a || !/^(line|multi_line)_string_literal$/.test(a.value.type)) return null;
      const tv = textOf(v);
      return tv ? { ...tv, lspans: [litSpan(a.value)] } : null;
    }
    return constCallTV(v);
  }
  if (v.type === 'ternary_expression') {
    const a = literalTV(v.childForFieldName('if_true') as TSNode);
    const b = a ? literalTV(v.childForFieldName('if_false') as TSNode) : null;
    return a && b ? joinAlt([a, b]) : null;
  }
  if (v.type === 'simple_identifier' || v.type === 'navigation_expression') return constTV(v);
  if (v.type === 'tuple_expression') {
    const ch = named(v).filter((c) => !/comment/.test(c.type));
    return ch.length === 1 ? literalTV(ch[0]) : null;
  }
  return null;
}

/** Text a binding stands for when every path yields a literal. */
function bindingTV(b: SwBinding, call: boolean): TV | null {
  if (CONST_DEPTH > 6) return null;
  let vals: TSNode[] | null = null;
  if (b.kind === 'prop' && !call) {
    const comp = b.node.childForFieldName('computed_value');
    const isLet = named(b.node).some((c) => c.type === 'value_binding_pattern' && /^let\b/.test(c.text));
    if (comp) {
      const st = named(comp).find((c) => c.type === 'statements');
      vals = st ? bodyValues(st) : null;
    } else if (isLet) {
      const v = b.node.childForFieldName('value');
      vals = v ? [v] : null;
    }
  } else if (b.kind === 'func' && call) {
    if (!/->\s*(String|LocalizedStringKey|LocalizedStringResource|Text)\??\s*\{/.test(b.node.text.slice(0, b.node.text.indexOf('{') + 1))) return null;
    const body = named(b.node).find((c) => c.type === 'function_body');
    const st = body ? named(body).find((c) => c.type === 'statements') : undefined;
    vals = st ? bodyValues(st) : null;
  }
  if (!vals || !vals.length || vals.length > MAX_EXPAND) return null;
  CONST_DEPTH++;
  try {
    // the body is read in its own scope: no substitutions from the caller
    const tvs = withEnv({ node: b.node, subst: null, exprs: new Map() }, () => vals!.map(literalTV));
    if (tvs.some((t) => !t)) return null;
    const ts = tvs as TV[];
    if (ts.length === 1) return ts[0];
    const j = joinAlt(ts);
    // several literals: the node sits at its use site
    return { ...j, lspans: undefined };
  } finally {
    CONST_DEPTH--;
  }
}

/** A bare name or member (`title`, `self.title`, `Self.title`, `Kind.title`, `kind.title`) bound to copy. */
function constTV(n: TSNode): TV | null {
  if (n.type === 'simple_identifier') {
    if (/^\$/.test(n.text)) return null;
    const b = bindingAt(n.text, n);
    return b ? bindingTV(b, false) : null;
  }
  if (n.type !== 'navigation_expression') return null;
  const target = n.childForFieldName('target');
  const suf = n.childForFieldName('suffix');
  const name = suf ? named(suf).find((c) => c.type === 'simple_identifier')?.text : undefined;
  if (!target || !name) return null;
  let type: string | null = null;
  if (target.type === 'self_expression' || target.text === 'self' || target.text === 'Self') {
    for (const t of enclosingTypes(n)) {
      const m = memberOf(t, name);
      if (m) return bindingTV(m, false);
    }
    return null;
  }
  type = typeNameOf(target) ?? valueType(target);
  const m = type ? memberOf(type, name) : null;
  return m ? bindingTV(m, false) : null;
}

/** A call of a function in this file (`title(kind)`, `self.title(for: kind)`) whose every branch returns copy. */
function constCallTV(n: TSNode): TV | null {
  const callee = named(n)[0];
  if (!callee) return null;
  if (callee.type === 'simple_identifier') {
    if (/^[A-Z]/.test(callee.text)) return null;
    const b = bindingAt(callee.text, n);
    return b && b.kind === 'func' ? bindingTV(b, true) : null;
  }
  if (callee.type === 'navigation_expression') {
    const target = callee.childForFieldName('target');
    const suf = callee.childForFieldName('suffix');
    const name = suf ? named(suf).find((c) => c.type === 'simple_identifier')?.text : undefined;
    if (!target || !name) return null;
    const types = target.text === 'self' || target.text === 'Self' ? enclosingTypes(n) : [typeNameOf(target) ?? valueType(target)].filter((t): t is string => !!t);
    for (const t of types) {
      const m = memberOf(t, name);
      if (m) return m.kind === 'func' ? bindingTV(m, true) : null;
    }
  }
  return null;
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
  const parts: TV[] = [];
  const plain = (v: string): void => {
    parts.push({ text: v, lit: v, dyn: false, alt: false, bound: false });
  };
  for (const c of named(n)) {
    if (c.type === 'line_str_text' || c.type === 'multi_line_str_text' || c.type === 'raw_str_part' || c.type === 'raw_str_end_part') {
      plain(stripPUA(c.text));
    } else if (c.type === 'str_escaped_char') {
      const t = c.text;
      let v: string;
      if (/^\\u\{/.test(t)) v = String.fromCodePoint(parseInt(t.slice(3, -1), 16));
      else v = ESC[t.slice(1)] ?? t.slice(1);
      plain(stripPUA(v));
    } else if (c.type === 'interpolated_expression' || c.type === 'raw_str_interpolation') {
      // `\(date, format: …)` parses as two interpolations; the labelled one is an argument of the first
      if (c.childForFieldName('name')) continue;
      const v = c.childForFieldName('value') ?? named(c)[0] ?? null;
      const bv = boundArg(v);
      if (bv) {
        const tv = withEnv(bv, () => textOf(bv.node));
        if (tv) {
          parts.push(tv.dyn ? { ...tv, text: ` ${tv.text.trim()} `, lit: ' ', opts: tv.opts?.map((o) => ({ text: o.text.trim() ? ` ${o.text.trim()} ` : '', lit: ' ' })) } : tv);
          continue;
        }
      }
      const r = swResolve(v);
      const sub = r ? slotTV(r.v) : null;
      if (sub) {
        // "\(row.title)": a substituted literal is copy
        parts.push(sub);
        continue;
      }
      const tv = v && /^(ternary_expression|nil_coalescing_expression)$/.test(v.type) ? textOf(v) : null;
      if (tv) parts.push({ ...tv, text: ` ${tv.text.trim()} `, lit: ' ', opts: tv.opts?.map((o) => ({ text: o.text.trim() ? ` ${o.text.trim()} ` : '', lit: ' ' })) });
      else parts.push({ text: ` ${varText(identOf(v))} `, lit: ' ', dyn: true, alt: false, bound: false });
    }
  }
  const out = concatTV(parts);
  const lit = parts.map((p) => p.lit).join('');
  return { ...out, lspans: undefined, bound: out.opts ? out.bound : out.dyn && lit.trim() === '' };
}

const LOCALISE = /^(String|NSLocalizedString|LocalizedStringKey|LocalizedStringResource)$/;

function textOf(n0: TSNode | null): TV | null {
  if (!n0) return null;
  const n = n0;
  switch (n.type) {
    case 'line_string_literal': case 'multi_line_string_literal': case 'raw_string_literal': return stringParts(n);
    case 'integer_literal': case 'real_literal': return { text: n.text, lit: n.text, dyn: false, alt: false, bound: false };
    case 'simple_identifier': case 'navigation_expression': {
      const b = boundArg(n);
      if (b) return withEnv(b, () => textOf(b.node));
      const r = swResolve(n);
      const sub = r ? slotTV(r.v) : null;
      if (sub) return sub;
      const k = constTV(n);
      if (k) return k;
      return { text: ` ${varText(identOf(n))} `, lit: ' ', dyn: true, alt: false, bound: true };
    }
    case 'call_expression': {
      const c = asCall(n);
      if (c && LOCALISE.test(c.name)) {
        const a = c.args.find((x) => x.label === null || x.label === 'localized');
        if (a && /^(line|multi_line)_string_literal$/.test(a.value.type)) {
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
      const k = constCallTV(n);
      if (k) return k;
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
      // `x ?? ""` guards a missing value; the empty fallback is not a rendering of its own
      return joinAlt(b.text.trim() === '' ? [a] : [a, b]);
    }
    case 'additive_expression': {
      const a = textOf(n.childForFieldName('lhs'));
      const b = textOf(n.childForFieldName('rhs'));
      if (!a || !b) return null;
      return concatTV([a, b]);
    }
    case 'prefix_expression': return null;
    case 'tuple_expression': {
      // a parenthesised expression: ("a" + (x ? "b" : "c"))
      const ch = named(n).filter((c) => !/comment/.test(c.type));
      return ch.length === 1 ? textOf(ch[0]) : null;
    }
    default: return n.text === 'nil' ? { text: '', lit: '', dyn: false, alt: false, bound: false } : null;
  }
}

function joinAlt(parts: TV[]): TV {
  const f = parts.filter((p) => p.text.trim() !== '');
  const all = parts.flatMap(optsOf);
  const opts = all.length >= 2 ? all : undefined;
  if (f.length === 0) return { text: '', lit: '', dyn: false, alt: false, bound: false };
  if (f.length === 1) return { ...f[0], alt: true, bound: true, opts };
  const spans = f.flatMap((p) => p.spans ?? []);
  const near = f.flatMap((p) => p.near ?? []);
  const lspans = f.flatMap((p) => p.lspans ?? []);
  return {
    text: f.map((p) => p.text.trim()).join(ALT_JOIN), lit: f.map((p) => p.lit.trim()).join(ALT_SEP), dyn: f.some((p) => p.dyn), alt: true, bound: true,
    ...(spans.length ? { spans, near } : {}), ...(lspans.length ? { lspans } : {}), opts,
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
  const b = boundArg(n);
  if (b) return withEnv(b, () => mkTxt(b.node, src, extra));
  const tv = textOf(n);
  if (!tv || tv.text.trim() === '') return null;
  let [s, e] = litSpan(n);
  while (s < e && /\s/.test(src[s])) s++;
  while (e > s && /\s/.test(src[e - 1])) e--;
  return txt(tv.text, s, e, {
    lit: tv.lit, dyn: tv.dyn, alt: tv.alt, bound: tv.bound, ...(tv.spans ? { subs: tv.spans, near: tv.near } : {}),
    ...(tv.opts && tv.opts.length >= 2 ? { opts: tv.opts } : {}), ...(tv.lspans?.length === 1 ? { lspan: tv.lspans[0] } : {}), ...extra,
  });
}

// ---------------------------------------------------------------------------------------------------------------

/** A SwiftUI text style as a modifier's only argument, or a house style named after one (`.captionBold`). */
const TEXT_STYLE = /^(\.|Font\.)(largeTitle|title3|title2|title|headline|subheadline|body|callout|footnote|caption2|caption)([A-Z]\w*)?$|^\.system\(\.(largeTitle|title3|title2|title|headline|subheadline|body|callout|footnote|caption2|caption)\b/;

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
        // `if xs.isEmpty { … }` (or the else of `if !xs.isEmpty`) is a hand-built empty state
        const blocks = named(s).filter((c) => c.type === 'statements');
        const condText = named(s).filter((c) => c.endIndex <= (blocks[0]?.startIndex ?? s.endIndex) && !/comment/.test(c.type)).map((c) => c.text).join(' ');
        const em = emptyCond(condText);
        for (const c of named(s)) {
          if (c.type === 'statements') {
            const kids = this.stmts(c);
            const empty = (em === 'pos' && c.equals(blocks[0])) || (em === 'neg' && blocks.length > 1 && c.equals(blocks[1]));
            out.push(...(empty ? markEmptyKids(kids, c.startIndex, c.endIndex) : kids));
          } else if (c.type === 'if_statement') out.push(...this.stmt(c));
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
    const bound = boundArg(base);
    if (bound && bound.node.type === 'lambda_literal') {
      // a view-builder closure passed to an inlined function, used as `content`
      extra = withEnv(bound, () => this.closureKids({ node: bound.node }));
    } else if (base.type === 'call_expression' && named(base)[0]?.type === 'additive_expression') {
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
      case 'Tab': {
        // iOS 18 Tab("Recipes", systemImage: …) { content }
        const kids: Kid[] = [];
        if (t0) kids.push(this.wrap('#tabItem', n, [...iconEl(), t0], { hint: 'tab', box: 'nav', textBlock: true }));
        kids.push(...this.closureKids(this.closure(c, null) ?? this.closure(c, 'content')));
        return this.wrap('Tab', n, kids);
      }
      case 'Image': {
        const sys = c.args.find((a) => a.label === 'systemName');
        return this.wrap('Image', n, [], { icon: !!sys, arrowIcon: !!sys && /arrow|chevron\.right/.test(sys.value.text) });
      }
      default: break;
    }
    if (name === 'NavigationSplitView') {
      // the sidebar column lists destinations: navigation
      const kids: Kid[] = [];
      c.closures.forEach((cl, j) => {
        const k = this.closureKids(cl);
        if (j === 0 && (cl.label === null || cl.label === 'sidebar')) kids.push(this.wrap('#sidebar', cl.node, k, { box: 'nav' }));
        else kids.push(...k);
      });
      return this.wrap(name, n, kids);
    }
    if (CONTAINERS.has(name)) {
      const data = name === 'List' ? c.args.find((a) => a.label === null)?.value : undefined;
      const elems = data && c.closures.length === 1 ? swArrayOf(data) : null;
      const copies = elems ? this.expand(c.closures[0], elems) : null;
      if (copies) return this.wrap(name, n, copies, { box: 'list' });
      const kids = allClosures();
      let box: El['box'];
      if (name === 'Form') box = 'form';
      // a List with a selection is a source list (a sidebar's destinations): navigation
      else if (name === 'List') box = c.args.some((a) => a.label === 'selection') ? 'nav' : 'list';
      else if (name === 'Table') box = 'table';
      const e = this.wrap(name, n, name === 'List' && c.args.some((a) => a.label !== 'selection') && c.closures.length ? kids.map((k) => (k.t === 'el' ? { ...k, tmpl: true } : k)) : kids, { box });
      return e;
    }
    if (SHAPES.has(name)) return this.wrap(name, n, []);
    if (/^[a-z_]/.test(name)) {
      const f = FUNCS.get(name);
      if (f && !INLINING.includes(name) && INLINING.length < 4) return this.inline(name, f, c);
      const b = EXPRS.get(name);
      if (b && b.node.type === 'lambda_literal' && !c.args.length) return this.wrap('#content', n, withEnv(b, () => this.closureKids({ node: b.node })));
      return null;
    }
    if (NOT_VIEWS.test(name)) return null;
    // a custom view: its first unlabelled argument when that is copy, labelled text arguments by name, closures as
    // content
    const e = this.wrap(name, n, []);
    decorateComponent(e, undefined);
    const kids: Kid[] = [];
    const hasAction = c.args.some((a) => /^(action|onTap|onSelect|perform|onPress)$/.test(a.label ?? ''))
      || (c.closures.some((cl) => cl.label === null) && !c.closures.some((cl) => this.closureKids(cl).some(kidsHaveText)));
    const iconArg = c.args.some((a) => /^(systemImage|systemName|icon|image|symbol)$/.test(a.label ?? ''));
    const pos = c.args.find((a) => a.label === null);
    const pt = pos && copyArg(pos.value) ? this.argTxt(c, null) : null;
    if (pt) {
      const r: Role | null = ICON_BUTTON.test(name) ? 'aria'
        : /(Button|Btn)$/.test(name) ? 'button'
          : /(Badge|Chip|Tag|Pill)$/.test(name) ? (hasAction ? 'button' : 'badge')
            : /(Title|Heading|Header|HeaderLabel)$/.test(name) ? 'h3'
              : /Link$/.test(name) ? 'link'
                : /Caption$/.test(name) ? 'caption'
                  : /Label$/.test(name) ? 'label' : null;
      if (r === 'aria') kids.push({ ...pt, attr: 'aria' });
      else if (r) kids.push(this.labelSlot(r, [pt], pos?.value ?? n));
      else kids.push(pt);
    }
    // an icon-only button: an icon argument and a `label:` string but no visible title
    const iconOnly = !pt && iconArg && /(Button|Btn)$/.test(name);
    for (const a of c.args) {
      if (!a.label) continue;
      const r = iconOnly && /^(label|labelText)$/.test(a.label) ? 'aria' : propRole(name, a.label);
      if (!r) continue;
      // `content:` and `copy:` usually pass a model or a view in SwiftUI: text only when the argument is copy
      if (/^(content|copy)$/.test(a.label) && !copyArg(a.value)) {
        if (a.value.type === 'lambda_literal' || a.value.type === 'call_expression') {
          const v = this.view(a.value);
          if (v.length) kids.push(this.labelSlot(r, v, a.value));
        }
        continue;
      }
      // an initialiser of a helper type (Copy(failure, …)) is a value passed in, not text or a view on screen
      const callee = a.value.type === 'call_expression' ? named(a.value)[0] : null;
      const helper = callee?.type === 'simple_identifier' && /^[A-Z]/.test(callee.text) && !LOCALISE.test(callee.text) && callee.text !== 'Text' && callee.text !== 'AttributedString';
      if (helper && !looksLikeView(callee.text)) continue;
      const t = helper ? null : this.argTxt(c, a.label, { attr: r });
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

  /** A local view function read at its call, its parameters bound to the call's arguments. */
  inline(name: string, f: LocalFunc, c: Call): El {
    const subst = new Map(SUBST ?? []);
    const exprs = new Map(EXPRS);
    const positional = c.args.filter((a) => a.label === null).map((a) => a.value);
    const trailing = c.closures.filter((cl) => cl.label === null && !c.args.some((a) => a.value.equals(cl.node))).map((cl) => cl.node);
    for (const p of f.params) {
      let arg: TSNode | null | undefined = null;
      if (p.ext === '_') arg = positional.shift();
      else {
        const lab = p.ext ?? p.name;
        arg = c.args.find((a) => a.label === lab)?.value ?? c.closures.find((cl) => cl.label === lab)?.node;
      }
      if (!arg && p.fn) arg = trailing.shift();
      if (!arg) arg = p.def;
      subst.delete(p.name);
      exprs.delete(p.name);
      if (!arg) {
        subst.set(p.name, undefined);
        continue;
      }
      const lv = swLit(arg, 1);
      const r = lv ? null : swResolve(arg);
      if (lv && (lv.k === 'str' || lv.k === 'num' || lv.k === 'bool' || lv.k === 'nil')) subst.set(p.name, lv);
      else if (r && r.v !== null) subst.set(p.name, r.v);
      else exprs.set(p.name, { node: arg, subst: SUBST, exprs: EXPRS });
    }
    INLINING.push(name);
    INLINED.add(name);
    let kids: Kid[];
    try {
      kids = withEnv({ node: c.node, subst, exprs }, () => this.stmts(f.body));
    } finally {
      INLINING.pop();
    }
    for (const k of kids) markCall(k, [c.node.startIndex, c.node.endIndex]);
    return this.wrap(`#${name}`, c.node, kids);
  }

  applyMods(core: El, mods: Mod[]): Kid[] {
    const before: Kid[] = [];
    const after: Kid[] = [];
    const sty = core.sty;
    for (const m of mods) {
      const a0 = m.args[0]?.value;
      const a0t = a0?.text ?? '';
      if (a0 && a0.type === 'prefix_expression' && /^\.[A-Za-z_]\w*$/.test(a0t)) core.cls.push(`${m.name}${a0t}`);
      // A text style is a font whatever the modifier is called (.font(.title), .appFont(.title), .hcFont(.caption)),
      // also as the stem of a house style (.bodyEmphasis, .captionBold).
      const styleArg = m.name !== 'font' && m.args.length === 1 && !m.args[0].label && TEXT_STYLE.test(a0t);
      switch (styleArg ? 'font' : m.name) {
        case 'font': {
          const f = a0t.match(/\.(largeTitle|title3|title2|title|headline|subheadline|body|callout|footnote|caption2|caption)(?![a-z0-9])/);
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
          if (a0 && a0.type !== 'ternary_expression' && /^(\.|Color\.)(red)\b|\b(destructive|danger|error)\b/.test(a0t)) sty.errorColour = true;
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
        case 'labelsHidden': hideLabels(core); break;
        // a view that is not drawn gives no nodes (keyboard-shortcut buttons kept off screen)
        case 'opacity': if (m.args.length === 1 && /^0*(\.0*)?$/.test(a0t) && /\d/.test(a0t)) core.hidden = true; break;
        case 'hidden': if (!m.args.length) core.hidden = true; break;
        case 'keyboardShortcut': core.attrs.keyboardShortcut = a0t || '1'; break;
        case 'accessibilityHidden':
          // hidden from VoiceOver only: the view is still drawn, unless all its text is keyboard-shortcut buttons
          if (/^true$/.test(a0t) && kidsHaveText(core) && !textOutsideShortcut(core)) core.hidden = true;
          break;
        case 'listStyle': if (/sidebar/.test(a0t)) core.box = 'nav'; break;
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

/** An argument that is copy: a string literal (not an SF Symbol name), String(localized:), Text(…), or a literal. */
function copyArg(n0: TSNode): boolean {
  const b = boundArg(n0);
  if (b) return withEnv(b, () => copyArg(b.node));
  const n = n0;
  if (/string_literal$/.test(n.type)) return !/^[a-z0-9]+(\.[a-z0-9]+)+$/.test(stringParts(n).text);
  if (n.type === 'call_expression') {
    const c = asCall(n);
    return !!c && (LOCALISE.test(c.name) || c.name === 'Text');
  }
  const r = swResolve(n);
  return !!r && !!r.v && r.v.k === 'str';
}

const VIEWISH = /^(Text|Label|Image|Button|Link|Toggle|Picker|Menu|Section|GroupBox|ProgressView|Spacer|Divider)$|(View|Button|Label|Text|Badge|Chip|Pill|Tag|Title|Heading|Header|Footer|Row|Card|Banner|Cell|Tile|Image|Icon|Stack|Section|Grid|Group|List)$/;

/** Is a type called in argument position a view (rendered) rather than a value type passed in? */
function looksLikeView(name: string): boolean {
  return VIEWISH.test(name) || CONTAINERS.has(name) || SHAPES.has(name);
}

/** A small leaf view: at most three texts, no control, link, dialog, menu or list. */
function smallLeaf(kids: Kid[]): boolean {
  let texts = 0;
  let ok = true;
  const walk = (k: Kid): void => {
    if (k.t === 'txt') {
      if (!k.attr) texts++;
      return;
    }
    if (k.box || k.control || k.tmpl || /^(Button|Link|NavigationLink|ShareButton|Menu|Toggle|Picker)$/.test(k.name)) ok = false;
    k.kids.forEach(walk);
  };
  kids.forEach(walk);
  return ok && texts >= 1 && texts <= 3;
}

/** `.labelsHidden()`: the labels of the controls inside are not shown; they are their accessible names. */
function hideLabels(e: El): void {
  for (const k of e.kids) {
    if (k.t !== 'el') continue;
    if (k.name === '#slot' && k.hint === 'label') {
      k.hint = 'aria';
      k.kids = k.kids.map((x) => (x.t === 'txt' ? { ...x, attr: 'aria' as Role } : x));
    } else hideLabels(k);
  }
}

/** Does any text sit outside a button with a keyboard shortcut? */
function textOutsideShortcut(k: Kid): boolean {
  if (k.t === 'txt') return !k.attr && k.text.trim() !== '';
  if (k.attrs.keyboardShortcut) return false;
  return k.kids.some(textOutsideShortcut);
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
      if (/\bView\b|^Text$|AnyView|\bToolbarContent\b/.test(c.text) && !/ViewModel|ViewController/.test(c.text)) return true;
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
  BINDINGS = collectBindings(tree.rootNode);
  CONST_DEPTH = 0;
  FUNCS = collectFuncs(tree.rootNode);
  INLINED = new Set();
  EXPRS = new Map();
  SUBST = null;
  const root = el('#root', 0, src.length, { box: 'root' });
  const roots: Kid[] = [];
  const fnRoots: [string, El][] = [];
  // A view whose own name says what it draws (TagPill, StatusBadge): its body takes that role.
  const DEF_ROLE: [RegExp, Role][] = [[/(Badge|Chip|Tag|Pill)$/, 'badge'], [/(Eyebrow|Kicker|Overline)$/, 'eyebrow']];
  const walk = (n: TSNode, inPreview: boolean, typeName: string | null): void => {
    if (n.type === 'macro_invocation' && /^#?Preview/.test(n.text.replace(/^#/, ''))) return;
    if (n.type === 'class_declaration') {
      const inh = named(n).filter((c) => c.type === 'inheritance_specifier').map((c) => c.text).join(' ');
      if (/PreviewProvider/.test(inh)) return;
      const tn = n.childForFieldName('name')?.text ?? null;
      for (const c of named(n)) walk(c, inPreview, tn);
      return;
    }
    const hint = typeName ? DEF_ROLE.find(([re]) => re.test(typeName))?.[1] : undefined;
    if (n.type === 'property_declaration' && !inPreview) {
      const nameNode = named(n).find((c) => c.type === 'pattern');
      const nm = nameNode?.text ?? '';
      const comp = n.childForFieldName('computed_value');
      if (comp && nm !== 'previews' && (isViewType(n) || nm === 'body')) {
        const st = named(comp).find((c) => c.type === 'statements');
        if (st) {
          const kids = b.stmts(st);
          // toolbar content (`some ToolbarContent`) sits in the bar, like .toolbar { }
          const tb = /\bToolbarContent\b/.test(named(n).find((c) => c.type === 'type_annotation')?.text ?? '');
          const leaf = !!hint && nm === 'body' && smallLeaf(kids);
          if (kids.length) roots.push(b.wrap(tb ? '#toolbar' : '#body', n, kids, tb ? { box: 'header' } : leaf ? { hint } : {}));
        }
        return;
      }
    }
    if (n.type === 'function_declaration' && isViewType(n)) {
      const body = named(n).find((c) => c.type === 'function_body');
      const st = body ? named(body).find((c) => c.type === 'statements') : undefined;
      if (st) {
        const kids = b.stmts(st);
        if (kids.length) {
          const r = b.wrap('#body', n, kids);
          roots.push(r);
          fnRoots.push([n.childForFieldName('name')?.text ?? '', r]);
        }
      }
      return;
    }
    for (const c of named(n)) walk(c, inPreview, typeName);
  };
  walk(tree.rootNode, false, null);
  tree.delete();
  // a local view function read at its calls is not also read on its own
  const drop = new Set(fnRoots.filter(([nm]) => INLINED.has(nm)).map(([, r]) => r));
  root.kids = roots.filter((r) => !(r.t === 'el' && drop.has(r)));
  return root;
}
