// JSX / TSX front end (@babel/parser). Every JSX tree in the file becomes a root in source order; toast(),
// alert() and confirm() calls become standalone state roots. Text from expressions becomes the VAR marker plus words.
// A .map() over an array literal of strings or objects in the same file is expanded once per element, with the
// literal values substituted (see "Literal arrays" below).

import { parse } from '@babel/parser';
import type { El, Kid, LV, Slot, Sty, Txt } from './ir.ts';
import {
  ALT_JOIN, ALT_SEP, MAX_EXPAND, el, emptyCond, hasSubs, markEmptyKids, identWords, jsxWhitespace, keyLastSegment, looksLikeCopy, markCopy, trimSpan, txt, varText,
} from './ir.ts';
import { decorateComponent, decorateIntrinsic, propRole } from './roles.ts';
import type { Role } from './schema.ts';

// Babel AST nodes are handled structurally.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type N = any;

interface Env {
  src: string;
  imports: Map<string, string>;
  consts: Map<string, string | null>; // const NAME = "string" (null when declared twice with different values)
  bindings: Map<string, Binding[]>; // every declared name with its scope, for resolving a .map() receiver
  mutated: Set<string>; // names an array method changes in place (push, splice, …) or an element is assigned on
  subst?: Map<string, Slot>; // names bound to array-element values inside an expanded .map() callback
}

const SKIP_KEYS = new Set(['loc', 'start', 'end', 'extra', 'leadingComments', 'trailingComments', 'innerComments', 'range', 'typeAnnotation', 'returnType', 'typeParameters', 'superTypeParameters']);

function unwrap(e: N): N {
  while (e && /^(TSAsExpression|TSNonNullExpression|TSSatisfiesExpression|ParenthesizedExpression|TypeCastExpression|TSTypeAssertion|TSInstantiationExpression)$/.test(e.type)) e = e.expression;
  return e;
}

function isJsx(e: N): boolean {
  return !!e && (e.type === 'JSXElement' || e.type === 'JSXFragment');
}

/** Does the expression contain JSX (not inside nested functions other than the expression itself)? */
function containsJsx(e: N): boolean {
  if (!e || typeof e !== 'object') return false;
  if (isJsx(e)) return true;
  for (const k of Object.keys(e)) {
    if (SKIP_KEYS.has(k)) continue;
    const v = e[k];
    if (Array.isArray(v)) {
      if (v.some((x) => x && typeof x.type === 'string' && containsJsx(x))) return true;
    } else if (v && typeof v.type === 'string' && containsJsx(v)) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------------------------------------------
// Literal arrays
//
// items.map((item) => …) where items is bound in the same file (const, any scope) to an array literal of string or
// object literals is expanded once per element, at most MAX_EXPAND elements, in order. Inside each copy the callback's
// parameters (plain, destructured, or destructured again by a const in the callback body) stand for the element:
// string, number, boolean and null property values are substituted, and a nested array literal one level down can be
// mapped the same way. Anything that is not a literal keeps the VAR behaviour. A substituted string carries the span of
// its literal in the array, which becomes the node's loc; the span of its property and of the whole element are
// fallbacks for a second node that shows the same literal, and the element's span is the loc of copy text that has
// no literal of its own (infer.ts).

interface Binding {
  s: number; // the scope's range
  e: number;
  kind: string; // const, let, var, param, fn, class, import, catch
  init: N | null; // initialiser of a const bound to a plain identifier
}

/** The literal value of an expression, or null. Depth counts from the mapped array (0); arrays nest one level. */
function litValue(e0: N, depth: number): LV | null {
  const e = unwrap(e0);
  if (!e) return null;
  switch (e.type) {
    case 'StringLiteral': return { k: 'str', value: e.value, s: e.start + 1, e: e.end - 1 };
    case 'TemplateLiteral':
      return e.expressions.length === 0 ? { k: 'str', value: e.quasis[0].value.cooked ?? '', s: e.start + 1, e: e.end - 1 } : null;
    case 'NumericLiteral': return { k: 'num', value: e.value, raw: e.extra?.raw ?? String(e.value), s: e.start, e: e.end };
    case 'BooleanLiteral': return { k: 'bool', value: e.value };
    case 'NullLiteral': return { k: 'nil' };
    case 'Identifier': return e.name === 'undefined' ? { k: 'nil' } : null;
    case 'ObjectExpression': {
      const props = new Map<string, Slot>();
      let open = false;
      for (const p of e.properties) {
        const key = p.type === 'ObjectProperty' && !p.computed ? (p.key.type === 'Identifier' ? p.key.name : p.key.type === 'StringLiteral' ? p.key.value : null) : null;
        if (key === null) {
          open = true;
          continue;
        }
        const v = litValue(p.value, depth + 1);
        if (v && (v.k === 'str' || v.k === 'num')) v.p = [p.start, p.end];
        props.set(key, v);
      }
      return { k: 'obj', props, open };
    }
    case 'ArrayExpression': return depth <= 2 ? literalArray(e, depth) : null; // a property of a top-level element
    default: return null;
  }
}

/** An array literal whose elements are all string or object literals, else null. */
function literalArray(e: N, depth: number): { k: 'arr'; elems: LV[] } | null {
  if (!e.elements.length) return null; // [] is filled in later
  const elems: LV[] = [];
  for (const x of e.elements) {
    if (!x || x.type === 'SpreadElement') return null;
    const v = litValue(x, depth + 1);
    if (!v || (v.k !== 'str' && v.k !== 'obj')) return null;
    if (v.k === 'str') v.p = [x.start, x.end];
    else v.at = [x.start, x.end];
    elems.push(v);
  }
  return { k: 'arr', elems };
}

const BLOCKISH = /^(Program|BlockStatement|StaticBlock|For(In|Of)?Statement|SwitchStatement|TSModuleBlock)$/;

/** Every declared name in the file with the range of its scope. */
function collectBindings(program: N): Map<string, Binding[]> {
  const out = new Map<string, Binding[]>();
  const add = (name: string, scope: N, kind: string, init: N | null): void => {
    let a = out.get(name);
    if (!a) out.set(name, (a = []));
    a.push({ s: scope.start, e: scope.end, kind, init });
  };
  const names = (p: N, f: (id: N) => void): void => {
    if (!p) return;
    switch (p.type) {
      case 'Identifier': f(p); break;
      case 'ObjectPattern': for (const q of p.properties) names(q.type === 'RestElement' ? q.argument : q.value, f); break;
      case 'ArrayPattern': for (const q of p.elements) names(q, f); break;
      case 'AssignmentPattern': names(p.left, f); break;
      case 'RestElement': names(p.argument, f); break;
      case 'TSParameterProperty': names(p.parameter, f); break;
      default: break;
    }
  };
  const walk = (n: N, stack: N[]): void => {
    if (!n || typeof n !== 'object') return;
    const near = (re: RegExp): N => {
      for (let j = stack.length - 1; j >= 0; j--) if (re.test(stack[j].type)) return stack[j];
      return program;
    };
    if (n.type === 'VariableDeclaration') {
      const scope = n.kind === 'var' ? near(/Function|^Program$/) : near(BLOCKISH);
      for (const d of n.declarations) {
        const plain = d.id.type === 'Identifier';
        names(d.id, (id) => add(id.name, scope, n.kind, plain && n.kind === 'const' ? d.init : null));
      }
    } else if (/Function/.test(n.type) && n.params) {
      if (n.type === 'FunctionDeclaration' && n.id) add(n.id.name, near(BLOCKISH), 'fn', null);
      else if (n.id) add(n.id.name, n, 'fn', null);
      for (const p of n.params) names(p, (id) => add(id.name, n, 'param', null));
    } else if (n.type === 'ClassDeclaration' && n.id) add(n.id.name, near(BLOCKISH), 'class', null);
    else if (n.type === 'ImportDeclaration') for (const sp of n.specifiers) add(sp.local.name, program, 'import', null);
    else if (n.type === 'CatchClause' && n.param) names(n.param, (id) => add(id.name, n, 'catch', null));
    const next = stack.concat(n);
    for (const k of Object.keys(n)) {
      if (SKIP_KEYS.has(k)) continue;
      const v = n[k];
      if (Array.isArray(v)) v.forEach((x) => walk(x, next));
      else if (v && typeof v.type === 'string') walk(v, next);
    }
  };
  walk(program, []);
  return out;
}

const MUTATORS = /^(push|pop|shift|unshift|splice|sort|reverse|fill|copyWithin)$/;

/** Names whose array is changed in place somewhere in the file (by name only, whatever the scope). */
function collectMutated(program: N): Set<string> {
  const out = new Set<string>();
  const root = (x: N): string | null => {
    const u = unwrap(x);
    if (u?.type === 'Identifier') return u.name;
    if (u?.type === 'MemberExpression' || u?.type === 'OptionalMemberExpression') return root(u.object);
    return null;
  };
  const walk = (n: N): void => {
    if (!n || typeof n !== 'object') return;
    if (n.type === 'CallExpression' || n.type === 'OptionalCallExpression') {
      const c = unwrap(n.callee);
      if ((c.type === 'MemberExpression' || c.type === 'OptionalMemberExpression') && !c.computed && MUTATORS.test(c.property.name)) {
        const r = root(c.object);
        if (r) out.add(r);
      }
    }
    if ((n.type === 'AssignmentExpression' || n.type === 'UpdateExpression')) {
      const t = unwrap(n.left ?? n.argument);
      if (t?.type === 'MemberExpression' || t?.type === 'OptionalMemberExpression') {
        const r = root(t);
        if (r) out.add(r);
      }
    }
    for (const k of Object.keys(n)) {
      if (SKIP_KEYS.has(k)) continue;
      const v = n[k];
      if (Array.isArray(v)) v.forEach(walk);
      else if (v && typeof v.type === 'string') walk(v);
    }
  };
  walk(program);
  return out;
}

/** The binding an identifier refers to: the innermost scope that declares the name; null if none or ambiguous. */
function bindingOf(id: N, env: Env): Binding | null {
  const cands = (env.bindings.get(id.name) ?? []).filter((b) => b.s <= id.start && id.start < b.e);
  if (!cands.length) return null;
  const size = Math.min(...cands.map((b) => b.e - b.s));
  const inner = cands.filter((b) => b.e - b.s === size);
  return inner.length === 1 ? inner[0] : null;
}

/**
 * The static value of an expression inside an expanded copy: a literal, or a path from a substituted name. Null when
 * the expression is neither (it is then handled as before).
 */
function resolve(e0: N, env: Env): { v: Slot } | null {
  const e = unwrap(e0);
  if (!e) return null;
  if (e.type === 'Identifier') {
    if (env.subst?.has(e.name)) return { v: env.subst.get(e.name) };
    return null;
  }
  if (e.type === 'MemberExpression' || e.type === 'OptionalMemberExpression') {
    const base = resolve(e.object, env);
    if (!base) return null;
    const b = base.v;
    if (b === undefined) return { v: undefined };
    if (b === null) return { v: null };
    const key = !e.computed && e.property.type === 'Identifier' ? e.property.name : e.computed && e.property.type === 'StringLiteral' ? e.property.value : null;
    if (b.k === 'obj' && key !== null) return { v: b.props.has(key) ? b.props.get(key) : b.open ? null : undefined };
    if (b.k === 'arr' && e.computed && e.property.type === 'NumericLiteral') return { v: b.elems[e.property.value] };
    return { v: null };
  }
  if (!env.subst) return null;
  const lv = litValue(e, 2);
  return lv ? { v: lv } : null;
}

/** Truthiness of a condition whose operands are all static, else null. */
function staticTruth(e0: N, env: Env): boolean | null {
  const e = unwrap(e0);
  if (!e || !env.subst) return null;
  if (e.type === 'UnaryExpression' && e.operator === '!') {
    const t = staticTruth(e.argument, env);
    return t === null ? null : !t;
  }
  if (e.type === 'LogicalExpression' && (e.operator === '&&' || e.operator === '||')) {
    const a = staticTruth(e.left, env);
    const b = staticTruth(e.right, env);
    if (e.operator === '&&') return a === false || b === false ? false : a === true && b === true ? true : null;
    return a === true || b === true ? true : a === false && b === false ? false : null;
  }
  if (e.type === 'BinaryExpression' && /^([!=]==?|[<>]=?)$/.test(e.operator)) {
    const a = primitive(lengthOf(e.left, env) ?? resolve(e.left, env));
    const b = primitive(lengthOf(e.right, env) ?? resolve(e.right, env));
    if (a === UNKNOWN || b === UNKNOWN) return null;
    if (/^[<>]/.test(e.operator)) {
      if (typeof a !== 'number' || typeof b !== 'number') return null;
      return e.operator === '<' ? a < b : e.operator === '>' ? a > b : e.operator === '<=' ? a <= b : a >= b;
    }
    const loose = e.operator === '==' || e.operator === '!=';
    const eq = loose ? a == b : a === b; // eslint-disable-line eqeqeq
    return e.operator.startsWith('!') ? !eq : eq;
  }
  const r = resolve(e, env);
  if (!r) return null;
  const v = r.v;
  if (v === undefined) return false;
  if (v === null) return null;
  switch (v.k) {
    case 'str': return v.value !== '';
    case 'num': return v.value !== 0;
    case 'bool': return v.value;
    case 'nil': return false;
    default: return true;
  }
}

const UNKNOWN = Symbol('unknown');
/** `xs.length` of a substituted literal array, as a number value (for static conditions only). */
function lengthOf(e0: N, env: Env): { v: Slot } | null {
  const e = unwrap(e0);
  if (!e || !env.subst || (e.type !== 'MemberExpression' && e.type !== 'OptionalMemberExpression') || e.computed || e.property.name !== 'length') return null;
  const r = resolve(e.object, env);
  if (!r?.v || r.v.k !== 'arr') return null;
  const n = r.v.elems.length;
  return { v: { k: 'num', value: n, raw: String(n), s: e.start, e: e.start } };
}

function primitive(r: { v: Slot } | null): string | number | boolean | null | undefined | typeof UNKNOWN {
  if (!r) return UNKNOWN;
  const v = r.v;
  if (v === undefined) return undefined;
  if (v === null) return UNKNOWN;
  if (v.k === 'str' || v.k === 'num' || v.k === 'bool') return v.value;
  if (v.k === 'nil') return null;
  return UNKNOWN;
}

/** The elements a .map() receiver stands for, when it is a literal array (bound in this file or substituted). */
function arrayOf(e0: N, env: Env): LV[] | null {
  const e = unwrap(e0);
  if (!e) return null;
  if (e.type === 'ArrayExpression') return literalArray(e, 0)?.elems ?? null;
  if (e.type === 'Identifier' && !env.subst?.has(e.name)) {
    const b = bindingOf(e, env);
    if (!b || b.kind !== 'const' || !b.init || env.mutated.has(e.name)) return null;
    const init = unwrap(b.init);
    return init.type === 'ArrayExpression' ? literalArray(init, 0)?.elems ?? null : null;
  }
  if (e.type === 'CallExpression' || e.type === 'OptionalCallExpression') {
    const c = unwrap(e.callee);
    if ((c.type === 'MemberExpression' || c.type === 'OptionalMemberExpression') && !c.computed && c.property.name === 'slice') {
      const a = arrayOf(c.object, env);
      const num = (x: N): number | null => {
        const u = unwrap(x);
        if (u?.type === 'NumericLiteral') return u.value;
        if (u?.type === 'UnaryExpression' && u.operator === '-' && u.argument.type === 'NumericLiteral') return -u.argument.value;
        return null;
      };
      const args = e.arguments.map(num);
      if (!a || args.some((x: number | null) => x === null)) return null;
      return a.slice(...(args as number[]));
    }
    return null;
  }
  const r = resolve(e, env);
  return r?.v && r.v.k === 'arr' ? r.v.elems : null;
}

/** Bind a parameter or declaration pattern to a value. */
function bindPattern(p: N, v: Slot, into: Map<string, Slot>): void {
  if (!p) return;
  switch (p.type) {
    case 'Identifier': into.set(p.name, v); break;
    case 'AssignmentPattern': bindPattern(p.left, v === undefined ? litValue(p.right, 1) : v, into); break;
    case 'ObjectPattern':
      for (const q of p.properties) {
        if (q.type === 'RestElement') {
          bindPattern(q.argument, null, into);
          continue;
        }
        const key = !q.computed ? (q.key.type === 'Identifier' ? q.key.name : q.key.type === 'StringLiteral' ? q.key.value : null) : null;
        let sub: Slot = null;
        if (v === undefined) sub = undefined;
        else if (v && v.k === 'obj' && key !== null) sub = v.props.has(key) ? v.props.get(key) : v.open ? null : undefined;
        bindPattern(q.value, sub, into);
      }
      break;
    case 'ArrayPattern':
      p.elements.forEach((q: N, j: number) => bindPattern(q, v && v.k === 'arr' ? v.elems[j] : v === undefined ? undefined : null, into));
      break;
    case 'RestElement': bindPattern(p.argument, null, into); break;
    default: break;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Text from expressions

interface TextVal {
  text: string;
  lit: string;
  dyn: boolean;
  alt: boolean;
  cond: boolean;
  bound: boolean;
  empty?: boolean;
  spans?: [number, number][]; // substituted array literals, in text order
  near?: [number, number][]; // their properties' spans
}

const I18N_CALLEES = new Set(['t', '$t', 'translate', 'tr', '__', '_', 'i18n.t', 'i18next.t', 'intl.formatMessage', 'formatMessage', 'getString', 'localize']);
const PASS_METHODS = new Set(['toString', 'toLocaleString', 'toFixed', 'toPrecision', 'toUpperCase', 'toLowerCase', 'trim', 'padStart', 'padEnd', 'join', 'toLocaleDateString', 'toLocaleTimeString', 'format', 'slice', 'substring']);

function calleeName(c: N): string | null {
  c = unwrap(c);
  if (!c) return null;
  if (c.type === 'Identifier') return c.name;
  if ((c.type === 'MemberExpression' || c.type === 'OptionalMemberExpression') && !c.computed && c.property.type === 'Identifier') {
    const o = calleeName(c.object);
    return o ? `${o}.${c.property.name}` : c.property.name;
  }
  return null;
}

/** The identifier whose words name an interpolation. */
function identOf(e: N): string | null {
  e = unwrap(e);
  if (!e) return null;
  switch (e.type) {
    case 'Identifier': return e.name;
    case 'MemberExpression': case 'OptionalMemberExpression':
      if (!e.computed && e.property.type === 'Identifier') return e.property.name;
      if (e.computed && e.property.type === 'StringLiteral') return e.property.value;
      return identOf(e.object);
    case 'CallExpression': case 'OptionalCallExpression': {
      const c = unwrap(e.callee);
      if ((c.type === 'MemberExpression' || c.type === 'OptionalMemberExpression') && !c.computed && PASS_METHODS.has(c.property.name)) return identOf(c.object);
      const a0 = e.arguments[0];
      if (a0 && /Identifier|MemberExpression/.test(unwrap(a0).type)) return identOf(a0);
      return identOf(c);
    }
    case 'AwaitExpression': return identOf(e.argument);
    case 'NewExpression': return e.arguments[0] ? identOf(e.arguments[0]) : identOf(e.callee);
    case 'ThisExpression': return null;
    default: return null;
  }
}

function lit(text: string): TextVal {
  return { text, lit: text, dyn: false, alt: false, cond: false, bound: false };
}
function vari(e: N): TextVal {
  return { text: ` ${varText(identOf(e))} `, lit: ' ', dyn: true, alt: false, cond: false, bound: true };
}
const EMPTY: TextVal = { text: '', lit: '', dyn: false, alt: false, cond: false, bound: false, empty: true };
const spansOf = (...vs: TextVal[]): [number, number][] | undefined => {
  const out = vs.flatMap((v) => v.spans ?? []);
  return out.length ? out : undefined;
};
const nearOf = (...vs: TextVal[]): [number, number][] | undefined => {
  const out = vs.flatMap((v) => v.near ?? []);
  return out.length ? out : undefined;
};

/** Text of a substituted value: a string or number literal is copy at its own span; other values as before. */
function slotText(v: Slot, e: N): TextVal {
  if (v === undefined) return EMPTY;
  if (v === null) return vari(e);
  switch (v.k) {
    case 'str': return { ...lit(v.value), spans: [[v.s, v.e]], near: v.p ? [v.p] : undefined };
    case 'num': return { ...lit(v.raw), spans: [[v.s, v.e]], near: v.p ? [v.p] : undefined };
    case 'bool': case 'nil': return EMPTY;
    default: return vari(e);
  }
}

/** Text of an expression, or null if it is not text (JSX, objects, functions). */
function textOf(e0: N, env: Env): TextVal | null {
  const e = unwrap(e0);
  if (!e) return null;
  switch (e.type) {
    case 'StringLiteral': return lit(e.value);
    case 'NumericLiteral': return lit(e.extra?.raw ?? String(e.value));
    case 'NullLiteral': case 'BooleanLiteral': return EMPTY;
    case 'Identifier': {
      const r = resolve(e, env);
      if (r) return slotText(r.v, e);
      return e.name === 'undefined' ? EMPTY : vari(e);
    }
    case 'TemplateLiteral': {
      let text = '';
      let l = '';
      let dyn = false;
      const parts: TextVal[] = [];
      for (let i = 0; i < e.quasis.length; i++) {
        const q = e.quasis[i].value.cooked ?? e.quasis[i].value.raw;
        text += q;
        l += q;
        if (i < e.expressions.length) {
          const t = textOf(e.expressions[i], env) ?? vari(e.expressions[i]);
          text += t.text;
          l += t.lit;
          dyn = dyn || t.dyn;
          parts.push(t);
        }
      }
      return { text, lit: l, dyn, alt: false, cond: false, bound: dyn && e.quasis.every((q: N) => !(q.value.cooked ?? '').trim()), spans: spansOf(...parts), near: nearOf(...parts) };
    }
    case 'MemberExpression': case 'OptionalMemberExpression': {
      const r = resolve(e, env);
      return r ? slotText(r.v, e) : vari(e);
    }
    case 'AwaitExpression': return vari(e);
    case 'CallExpression': case 'OptionalCallExpression': {
      const name = calleeName(e.callee);
      if (name && /^(render|get\w*Icon)|Icon$/i.test(name.split('.').pop() ?? '')) return null;
      const callee = unwrap(e.callee);
      if ((callee.type === 'MemberExpression' || callee.type === 'OptionalMemberExpression') && !callee.computed && /^(toUpperCase|toLowerCase|trim)$/.test(callee.property.name)) {
        // a case change of a substituted literal is still that literal
        const inner = textOf(callee.object, env);
        if (inner?.spans && !inner.dyn && !inner.alt) {
          const f = (x: string): string => (callee.property.name === 'toUpperCase' ? x.toUpperCase() : callee.property.name === 'toLowerCase' ? x.toLowerCase() : x.trim());
          return { ...inner, text: f(inner.text), lit: f(inner.lit) };
        }
      }
      if ((callee.type === 'MemberExpression' || callee.type === 'OptionalMemberExpression') && !callee.computed && callee.property.name === 'join') {
        // a literal array of strings joined for display ("Distinction · Graduated top of class")
        const elems = arrayOf(callee.object, env);
        const sep = e.arguments[0] ? unwrap(e.arguments[0]) : null;
        const sepText = !sep ? ',' : sep.type === 'StringLiteral' ? sep.value : null;
        if (elems && sepText !== null && elems.every((x) => x.k === 'str')) {
          const strs = elems as Extract<LV, { k: 'str' }>[];
          return { ...lit(strs.map((x) => x.value).join(sepText)), spans: strs.map((x) => [x.s, x.e] as [number, number]), near: strs.flatMap((x) => (x.p ? [x.p] : [])) };
        }
      }
      if (name && I18N_CALLEES.has(name)) {
        const a0 = unwrap(e.arguments[0]);
        const sub = a0 && a0.type !== 'StringLiteral' ? resolve(a0, env) : null;
        if (sub?.v && sub.v.k === 'str') {
          // a translation key or gettext string from the array: the node sits at the literal
          const v = sub.v;
          const near: [number, number][] | undefined = v.p ? [v.p] : undefined;
          if (looksLikeCopy(v.value)) return { ...lit(v.value), spans: [[v.s, v.e]], near };
          return { text: ` ${varText(keyLastSegment(v.value))} `, lit: ' ', dyn: true, alt: false, cond: false, bound: true, spans: [[v.s, v.e]], near };
        }
        if (a0?.type === 'StringLiteral') {
          if (looksLikeCopy(a0.value)) return lit(a0.value);
          return { text: ` ${varText(keyLastSegment(a0.value))} `, lit: ' ', dyn: true, alt: false, cond: false, bound: true };
        }
        if (a0?.type === 'ObjectExpression') {
          const dm = a0.properties.find((p: N) => p.key?.name === 'defaultMessage' && p.value?.type === 'StringLiteral');
          if (dm) return lit(dm.value.value);
          const id = a0.properties.find((p: N) => p.key?.name === 'id' && p.value?.type === 'StringLiteral');
          if (id) return { text: ` ${varText(keyLastSegment(id.value.value))} `, lit: ' ', dyn: true, alt: false, cond: false, bound: true };
        }
      }
      return vari(e);
    }
    case 'ConditionalExpression': {
      const t = staticTruth(e.test, env);
      if (t !== null) return textOf(t ? e.consequent : e.alternate, env);
      const a = textOf(e.consequent, env);
      const b = textOf(e.alternate, env);
      if (!a || !b) return null;
      return joinAlt([a, b]);
    }
    case 'LogicalExpression': {
      const t = staticTruth(e.left, env);
      if (t !== null) {
        // decided by a substituted value: && keeps the right side when the left is truthy; || and ?? keep the left
        if (e.operator === '&&') return t ? textOf(e.right, env) : EMPTY;
        if (e.operator === '||') return textOf(t ? e.left : e.right, env);
        const l = resolve(e.left, env);
        if (l) return textOf(l.v === undefined || (l.v && l.v.k === 'nil') ? e.right : e.left, env);
      }
      if (e.operator === '&&') {
        const r = textOf(e.right, env);
        if (!r) return null;
        return { ...r, cond: true, bound: false };
      }
      const a = textOf(e.left, env);
      const b = textOf(e.right, env);
      if (!a || !b) return null;
      return joinAlt([a, b]);
    }
    case 'BinaryExpression': {
      if (e.operator !== '+') return vari(e);
      const subStr = (x: N): boolean => {
        const r = resolve(x, env);
        return !!r?.v && r.v.k === 'str';
      };
      const stringy = (x: N): boolean => /^(StringLiteral|TemplateLiteral)$/.test(unwrap(x).type) || subStr(x) || (unwrap(x).type === 'BinaryExpression' && unwrap(x).operator === '+' && (stringy(unwrap(x).left) || stringy(unwrap(x).right)));
      if (!stringy(e.left) && !stringy(e.right)) return vari(e.left);
      const a = textOf(e.left, env);
      const b = textOf(e.right, env);
      if (!a || !b) return null;
      return { text: a.text + b.text, lit: a.lit + b.lit, dyn: a.dyn || b.dyn, alt: a.alt || b.alt, cond: a.cond || b.cond, bound: false, spans: spansOf(a, b), near: nearOf(a, b) };
    }
    case 'UnaryExpression': case 'UpdateExpression': return vari(e.argument ?? e);
    default: return null;
  }
}

function joinAlt(parts: TextVal[]): TextVal {
  const flat: TextVal[] = [];
  for (const p of parts) if (!p.empty && p.text.trim() !== '') flat.push(p);
  if (flat.length === 0) return { ...EMPTY, cond: true };
  if (flat.length === 1) return { ...flat[0], cond: true, bound: true };
  return {
    text: flat.map((p) => p.text.trim()).join(ALT_JOIN),
    lit: flat.map((p) => p.lit.trim()).join(ALT_SEP),
    dyn: flat.some((p) => p.dyn),
    alt: true,
    cond: true,
    bound: true,
    spans: spansOf(...flat),
    near: nearOf(...flat),
  };
}

function txtFrom(v: TextVal, s: number, e: number, src: string, extra: Partial<Txt> = {}): Txt | null {
  if (!v.empty && v.text !== '' && v.text.trim() === '' && !extra.attr) return txt(' ', s, s); // {" "}
  if (v.empty || v.text.trim() === '') return null;
  const sp = trimSpan(src, s, e) ?? [s, e];
  return txt(v.text, sp[0], sp[1], { lit: v.lit, dyn: v.dyn, alt: v.alt, cond: v.cond, bound: v.bound, ...(v.spans ? { subs: v.spans, near: v.near } : {}), ...extra });
}

/** Span of the visible part of a literal: inside the quotes. */
function innerSpan(e: N): [number, number] {
  if (e.type === 'StringLiteral' || e.type === 'TemplateLiteral') return [e.start + 1, e.end - 1];
  return [e.start, e.end];
}

// ---------------------------------------------------------------------------------------------------------------
// Classes and styles

interface ClassInfo {
  tokens: string[];
  bound: boolean;
}

function splitTokens(s: string): string[] {
  return s.split(/\s+/).filter(Boolean);
}

const CLASS_FNS = /^(cn|clsx|classNames|classnames|cx|twMerge|twJoin|cva|tw|mergeClasses|classList)$/;

function classInfo(e0: N, env: Env, inCond = false): ClassInfo {
  const e = unwrap(e0);
  const out: ClassInfo = { tokens: [], bound: false };
  if (!e) return out;
  switch (e.type) {
    case 'StringLiteral': out.tokens = splitTokens(e.value); break;
    case 'TemplateLiteral': {
      for (let i = 0; i < e.quasis.length; i++) {
        let q: string = e.quasis[i].value.cooked ?? '';
        // drop tokens glued to an interpolation (bg-${c}-500)
        if (i > 0 && /^\S/.test(q)) q = q.replace(/^\S+/, '');
        if (i < e.expressions.length && /\S$/.test(q)) {
          q = q.replace(/\S+$/, '');
          out.bound = true;
        }
        out.tokens.push(...splitTokens(q));
        if (i < e.expressions.length) {
          const sub = classInfo(e.expressions[i], env, inCond);
          out.tokens.push(...sub.tokens);
          out.bound = out.bound || sub.bound;
        }
      }
      break;
    }
    case 'CallExpression': {
      const name = calleeName(e.callee);
      if (name && CLASS_FNS.test(name.split('.').pop() ?? '')) {
        for (const a of e.arguments) {
          const sub = classInfo(a, env, inCond);
          out.tokens.push(...sub.tokens);
          out.bound = out.bound || sub.bound;
        }
      }
      break;
    }
    case 'ArrayExpression':
      for (const a of e.elements) {
        const sub = classInfo(a, env, inCond);
        out.tokens.push(...sub.tokens);
        out.bound = out.bound || sub.bound;
      }
      break;
    case 'ConditionalExpression': {
      const a = classInfo(e.consequent, env, true);
      const b = classInfo(e.alternate, env, true);
      out.tokens = a.tokens.filter((t) => b.tokens.includes(t));
      out.bound = true;
      break;
    }
    case 'LogicalExpression': {
      if (e.operator === '&&') {
        out.bound = true;
      } else {
        const a = classInfo(e.left, env, true);
        out.tokens = a.tokens;
        out.bound = true;
      }
      break;
    }
    case 'ObjectExpression':
      for (const p of e.properties) {
        if (p.type !== 'ObjectProperty') continue;
        const v = unwrap(p.value);
        const key = p.key.type === 'StringLiteral' ? p.key.value : p.key.type === 'Identifier' ? p.key.name : null;
        if (!key) continue;
        if (v.type === 'BooleanLiteral' && v.value) out.tokens.push(...splitTokens(key));
        else if (v.type !== 'BooleanLiteral') out.bound = true;
      }
      break;
    case 'MemberExpression': case 'OptionalMemberExpression': {
      const obj = unwrap(e.object);
      if (obj.type === 'Identifier' && /^(styles|classes|css|s|cls|classNames)$/.test(obj.name)) {
        if (!e.computed && e.property.type === 'Identifier') out.tokens.push(e.property.name);
        else if (e.computed && e.property.type === 'StringLiteral') out.tokens.push(e.property.value);
      }
      break;
    }
    case 'Identifier': {
      const v = env.consts.get(e.name);
      if (typeof v === 'string') out.tokens = splitTokens(v);
      break;
    }
    default: break;
  }
  if (inCond) out.tokens = out.tokens.slice();
  return out;
}

const COLOUR_PROPS = /^(color|background|backgroundColor|borderColor|fill|stroke)$/;

function styleInfo(e0: N): { sty: Sty; bound: boolean } {
  const e = unwrap(e0);
  const sty: Sty = {};
  let bound = false;
  if (!e || e.type !== 'ObjectExpression') return { sty, bound };
  for (const p of e.properties) {
    if (p.type !== 'ObjectProperty') continue;
    const key = p.key.type === 'Identifier' ? p.key.name : p.key.type === 'StringLiteral' ? p.key.value : '';
    const v = unwrap(p.value);
    const sv = v.type === 'StringLiteral' ? v.value.toLowerCase() : v.type === 'NumericLiteral' ? String(v.value) : null;
    if (COLOUR_PROPS.test(key) && sv === null) bound = true;
    if (sv === null) continue;
    if (key === 'textTransform') sty.caps = sv === 'uppercase';
    if (key === 'letterSpacing') sty.tracked = parseFloat(sv) > 0;
    if (key === 'fontSize') {
      const n = parseFloat(sv);
      const px = /rem|em/.test(sv) ? n * 16 : n;
      if (!Number.isNaN(px)) {
        sty.small = px <= 14;
        sty.xsmall = px <= 12;
        sty.large = px >= 24;
      }
    }
    if (key === 'fontFamily') sty.mono = /mono|courier|menlo|consolas/.test(sv);
    if (key === 'textOverflow') sty.truncated = sv === 'ellipsis';
  }
  return { sty, bound };
}

// ---------------------------------------------------------------------------------------------------------------
// JSX → IR

const STATIC_ATTRS = new Set(['role', 'type', 'as', 'variant', 'level', 'href', 'htmlFor', 'for', 'aria-level', 'size', 'tone', 'intent', 'aria-label']);
const BOUND_PROPS = /^(variant|color|colour|tone|status|intent|state|severity|kind)$/;
const SKIP_TAGS = new Set(['script', 'style', 'template', 'noscript', 'head', 'meta', 'link', 'title']);

function jsxName(n: N): { name: string; intrinsic: boolean; root: string } {
  if (n.type === 'JSXIdentifier') return { name: n.name, intrinsic: /^[a-z]/.test(n.name) && !n.name.includes('.'), root: n.name };
  if (n.type === 'JSXNamespacedName') return { name: `${n.namespace.name}:${n.name.name}`, intrinsic: true, root: n.namespace.name };
  if (n.type === 'JSXMemberExpression') {
    const parts: string[] = [];
    let x = n;
    while (x.type === 'JSXMemberExpression') {
      parts.unshift(x.property.name);
      x = x.object;
    }
    parts.unshift(x.name);
    if (/^(motion|m|animated|styled)$/.test(parts[0]) && parts.length === 2 && /^[a-z]/.test(parts[1])) {
      return { name: parts[1], intrinsic: true, root: parts[1] };
    }
    return { name: parts.join(''), intrinsic: false, root: parts[0] };
  }
  return { name: '?', intrinsic: false, root: '?' };
}

function attrName(a: N): string {
  return a.name.type === 'JSXNamespacedName' ? `${a.name.namespace.name}:${a.name.name.name}` : a.name.name;
}

class JsxBuilder {
  private env: Env;
  constructor(env: Env) {
    this.env = env;
  }

  element(node: N): El | Kid[] {
    const src = this.env.src;
    if (node.type === 'JSXFragment') return this.children(node.children);
    const open = node.openingElement;
    const { name, intrinsic, root } = jsxName(open.name);
    const e = el(name, node.start, node.end, { intrinsic });
    const attrKids: Kid[] = [];
    let inputType = '';
    for (const a of open.attributes) {
      if (a.type !== 'JSXAttribute') continue;
      const an = attrName(a);
      const v = a.value;
      if (an === 'type' && v?.type === 'StringLiteral') inputType = v.value.toLowerCase();
    }
    // an icon-only button component (an icon prop, no children): its label is the accessible name, not shown
    const iconOnly = !intrinsic && /(Button|Btn)$/.test(name) && !node.children.some((c: N) => c.type !== 'JSXText' || c.value.trim() !== '')
      && open.attributes.some((a: N) => a.type === 'JSXAttribute' && /^(icon|leftIcon|rightIcon|startIcon|endIcon|iconName|systemImage)$/.test(attrName(a)));
    for (const a of open.attributes) {
      if (a.type === 'JSXSpreadAttribute' && !intrinsic) {
        // {...item} in an expanded copy: each literal text prop of the element, as if written out
        const r = resolve(a.argument, this.env);
        if (r?.v && r.v.k === 'obj') {
          for (const [key, pv] of r.v.props) {
            const role = propRole(name, key);
            if (!role || !pv || (pv.k !== 'str' && pv.k !== 'num')) continue;
            const t = txtFrom(slotText(pv, a.argument), a.start, a.end, src, { attr: role });
            if (t) attrKids.push(t);
          }
        }
        continue;
      }
      if (a.type !== 'JSXAttribute') continue;
      const an = attrName(a);
      const v = a.value;
      const expr = v?.type === 'JSXExpressionContainer' ? v.expression : v;
      if (an === 'className' || an === 'class') {
        if (expr && expr.type !== 'JSXEmptyExpression') {
          const ci = classInfo(expr, this.env);
          e.cls.push(...ci.tokens);
          e.bound = e.bound || ci.bound;
        }
        continue;
      }
      if (an === 'style' && expr) {
        const si = styleInfo(expr);
        Object.assign(e.sty, si.sty);
        e.bound = e.bound || si.bound;
        continue;
      }
      if (expr && expr.type === 'StringLiteral' && STATIC_ATTRS.has(an)) e.attrs[an] = expr.value;
      if (!intrinsic && BOUND_PROPS.test(an) && expr && expr.type !== 'StringLiteral' && expr.type !== 'JSXEmptyExpression') e.bound = true;
      if (!intrinsic && expr && expr.type === 'StringLiteral' && /^(variant|level)$/.test(an)) e.attrs[an] = expr.value;
      if (!intrinsic && expr && expr.type === 'NumericLiteral' && an === 'level') e.attrs.level = String(expr.value);
      // text-bearing attributes
      let role: Role | null = null;
      if (intrinsic) {
        if (an === 'placeholder') role = 'placeholder';
        else if (an === 'title' && name !== 'svg') role = 'tooltip';
        else if (an === 'aria-label') role = 'aria';
        else if (an === 'alt') role = 'alt';
        else if (an === 'data-tooltip' || an === 'data-tip') role = 'tooltip';
        else if ((an === 'value' || an === 'defaultValue') && name === 'input') role = /^(submit|button|reset)$/.test(inputType) ? 'button' : (/^(hidden|checkbox|radio|range|color|date|number)$/.test(inputType) ? null : 'input_value');
        else if ((an === 'value' || an === 'defaultValue') && name === 'textarea') role = 'input_value';
      } else {
        role = iconOnly && /^(label|text)$/.test(an) ? 'aria' : propRole(name, an);
        if (!role && expr && expr.type !== 'JSXEmptyExpression' && an !== 'key' && !/^(className|style|ref|as|asChild)$/.test(an) && containsJsx(expr)) {
          const slot = el('#prop', expr.start, expr.end);
          slot.kids = this.exprKids(expr);
          if (slot.kids.length) attrKids.push(slot);
          continue;
        }
      }
      if (!role || !expr || expr.type === 'JSXEmptyExpression') continue;
      if (isJsx(unwrap(expr)) || (containsJsx(expr) && !textOf(expr, this.env))) {
        const slot = el('#prop', expr.start, expr.end, { hint: role, textBlock: true });
        slot.kids = this.exprKids(expr);
        attrKids.push(slot);
        continue;
      }
      const tv = textOf(expr, this.env);
      if (!tv) continue;
      const [s, en] = innerSpan(unwrap(expr));
      const t = txtFrom(tv, s, en, src, { attr: role });
      if (t) attrKids.push(t);
    }
    if (intrinsic) decorateIntrinsic(e);
    else decorateComponent(e, this.env.imports.get(root));
    if (intrinsic && name === 'svg') {
      // an icon, unless it draws text (a chart's axis labels and values)
      const kids = this.children(node.children);
      if (kids.some(drawsText)) {
        e.icon = false;
        e.arrowIcon = false;
        e.kids = attrKids.concat(kids);
      } else e.kids = attrKids;
      return e;
    }
    if (intrinsic && SKIP_TAGS.has(name)) {
      e.kids = attrKids;
      return e;
    }
    if (intrinsic && name === 'br') {
      return [txt(' ', node.start, node.start)];
    }
    let kids = this.children(node.children);
    if (intrinsic && name === 'textarea') {
      kids = kids.map((k) => (k.t === 'txt' ? { ...k, attr: 'input_value' as Role } : k));
    }
    if (intrinsic && e.icon && /material-(icons|symbols)/.test(e.cls.join(' '))) kids = [];
    e.kids = attrKids.concat(kids);
    return e;
  }

  children(children: N[]): Kid[] {
    const out: Kid[] = [];
    const src = this.env.src;
    for (const c of children) {
      if (c.type === 'JSXText') {
        const v = jsxWhitespace(c.value);
        if (!v) continue;
        if (v.trim() === '') {
          out.push(txt(' ', c.start, c.start));
          continue;
        }
        const sp = trimSpan(src, c.start, c.end);
        if (!sp) continue;
        out.push(txt(v, sp[0], sp[1]));
      } else if (c.type === 'JSXExpressionContainer') {
        if (c.expression.type === 'JSXEmptyExpression') continue;
        out.push(...this.exprKids(c.expression));
      } else if (c.type === 'JSXSpreadChild') {
        out.push(...this.exprKids(c.expression));
      } else if (isJsx(c)) {
        const r = this.element(c);
        if (Array.isArray(r)) out.push(...r);
        else out.push(r);
      }
    }
    return out;
  }

  /** Kids produced by an expression in child position. */
  exprKids(e0: N): Kid[] {
    const e = unwrap(e0);
    if (!e) return [];
    const src = this.env.src;
    if (isJsx(e)) {
      const r = this.element(e);
      return Array.isArray(r) ? r : [r];
    }
    switch (e.type) {
      case 'ConditionalExpression': {
        const st = staticTruth(e.test, this.env);
        if (st !== null) return this.exprKids(st ? e.consequent : e.alternate);
        const tv = textOf(e, this.env);
        if (tv) {
          const t = txtFrom(tv, e.start, e.end, src);
          return t ? [t] : [];
        }
        let a = this.exprKids(e.consequent);
        let b = this.exprKids(e.alternate);
        const em = emptyTest(e.test, src);
        if (em === 'pos') a = markEmpty(a, e.consequent);
        else if (em === 'neg') b = markEmpty(b, e.alternate);
        const fa = flatInline(a);
        const fb = flatInline(b);
        if (fa && fb && (fa.text.trim() || fb.text.trim())) {
          // two inline branches in the same run of text: one node, branches joined with ALT
          const parts = [fa, fb].filter((p) => p.text.trim());
          const t = txt(parts.map((p) => p.text.trim()).join(ALT_JOIN), Math.min(...parts.map((p) => p.s)), Math.max(...parts.map((p) => p.e)), {
            lit: parts.map((p) => p.lit.trim()).join(ALT_SEP), dyn: parts.some((p) => p.dyn), alt: parts.length > 1, cond: true, bound: true,
          });
          return [t];
        }
        return [...a, ...b].map(markCond);
      }
      case 'LogicalExpression': {
        const st = staticTruth(e.left, this.env);
        if (st !== null && e.operator === '&&') return st ? this.exprKids(e.right) : [];
        if (st !== null && e.operator === '||') return this.exprKids(st ? e.left : e.right);
        if (e.operator === '&&') {
          const kids = this.exprKids(e.right).map(markCond);
          const em = emptyTest(e.left, this.env.src);
          if (em === 'pos') return markEmpty(kids, e.right);
          const g = nextAndGroup++;
          for (const k of kids) if (k.t === 'txt') k.andGroup = g;
          return kids;
        }
        const tv = textOf(e, this.env);
        if (tv) {
          const t = txtFrom(tv, e.start, e.end, src);
          return t ? [t] : [];
        }
        return [...this.exprKids(e.left), ...this.exprKids(e.right)].map(markCond);
      }
      case 'CallExpression': case 'OptionalCallExpression': {
        const callee = unwrap(e.callee);
        const prop = (callee.type === 'MemberExpression' || callee.type === 'OptionalMemberExpression') && !callee.computed ? callee.property.name : null;
        if (prop && /^(map|flatMap)$/.test(prop) && e.arguments[0]) {
          const elems = arrayOf(callee.object, this.env);
          const fn = unwrap(e.arguments[0]);
          const copies = elems && fn && /Function/.test(fn.type) ? this.expand(fn, elems) : null;
          return copies ?? this.template(e.arguments[0]);
        }
        if (calleeName(callee) === 'Array.from' && e.arguments[1]) return this.template(e.arguments[1]);
        if ((callee.type === 'ArrowFunctionExpression' || callee.type === 'FunctionExpression')) return this.fnKids(callee);
        const name = calleeName(callee) ?? '';
        if (/^React\.Children|^Children\./.test(name)) return [];
        if (/^(render|get\w*Icon)|Icon$/i.test(name.split('.').pop() ?? '')) return [];
        if (e.arguments.some((a: N) => containsJsx(a))) return e.arguments.flatMap((a: N) => (containsJsx(a) ? this.exprKids(a) : []));
        const tv = textOf(e, this.env);
        const t = tv ? txtFrom(tv, e.start, e.end, src) : null;
        return t ? [t] : [];
      }
      case 'ArrowFunctionExpression': case 'FunctionExpression': return this.fnKids(e);
      case 'ArrayExpression': return e.elements.flatMap((x: N) => (x ? this.exprKids(x) : []));
      case 'Identifier': {
        if (/^(children|undefined)$|Icon$|^icon$|Element$|Component$|^(avatar|image|logo|slot|node|element|svg|img)$/i.test(e.name)) return [];
        break;
      }
      case 'MemberExpression': case 'OptionalMemberExpression': {
        const id = identOf(e) ?? '';
        if (/^(children)$|Icon$|^icon$|Element$|Component$|^(avatar|image|logo|slot|node|element|svg|img)$/i.test(id)) return [];
        break;
      }
      default: break;
    }
    const tv = textOf(e, this.env);
    if (!tv) return [];
    const [s, en] = innerSpan(e);
    const t = txtFrom(tv, s, en, src);
    return t ? [t] : [];
  }

  /** Kids returned by a function used as a render prop or IIFE. */
  fnKids(fn: N): Kid[] {
    if (this.env.subst) {
      // inside an expanded copy, the function's own parameters and consts shadow substituted names
      const subst = new Map(this.env.subst);
      for (const p of fn.params) bindPattern(p, null, subst);
      return new JsxBuilder({ ...this.env, subst }).bodyKids(fn);
    }
    return this.bodyKids(fn);
  }

  /**
   * One copy of a .map() callback per array element (at most MAX_EXPAND), each a repeated subtree; null when no copy
   * shows any literal from the array, so that the callback is read once as a template, as before.
   */
  expand(fn: N, elems: LV[]): Kid[] | null {
    const out: Kid[] = [];
    for (const v of elems.slice(0, MAX_EXPAND)) {
      const subst = new Map(this.env.subst ?? []);
      fn.params.forEach((p: N, j: number) => bindPattern(p, j === 0 ? v : null, subst));
      const kids = new JsxBuilder({ ...this.env, subst }).bodyKids(fn);
      if (!kids.length) continue;
      const at = v.k === 'obj' ? v.at : v.k === 'str' ? v.p : undefined;
      if (at) for (const k of kids) markCopy(k, at);
      if (kids.length === 1 && kids[0].t === 'el') {
        kids[0].tmpl = true;
        out.push(kids[0]);
      } else {
        const w = el('#tmpl', fn.start, fn.end, { tmpl: true });
        w.kids = kids;
        out.push(w);
      }
    }
    return out.some(hasSubs) ? out : null;
  }

  bodyKids(fn: N): Kid[] {
    if (fn.body.type !== 'BlockStatement') return this.exprKids(fn.body);
    if (this.env.subst) {
      // const { title } = item; const label = item.cta; … in the callback body
      for (const st of fn.body.body) {
        if (st.type !== 'VariableDeclaration') continue;
        for (const d of st.declarations) {
          const r = st.kind === 'const' && d.init ? resolve(d.init, this.env) : null;
          bindPattern(d.id, r ? r.v : null, this.env.subst);
        }
      }
    }
    const rets: N[] = [];
    const walk = (n: N): void => {
      if (!n || typeof n !== 'object') return;
      if (/Function/.test(n.type) && n !== fn) return;
      if (n.type === 'ReturnStatement' && n.argument) rets.push(n.argument);
      for (const k of Object.keys(n)) {
        if (SKIP_KEYS.has(k)) continue;
        const v = n[k];
        if (Array.isArray(v)) v.forEach(walk);
        else if (v && typeof v.type === 'string') walk(v);
      }
    };
    walk(fn.body);
    const out = rets.flatMap((r) => this.exprKids(r));
    return rets.length > 1 ? out.map(markCond) : out;
  }

  /** A .map() template: each returned element is a repeated item. */
  template(fn0: N): Kid[] {
    const fn = unwrap(fn0);
    if (!fn || !/Function/.test(fn.type)) {
      return [];
    }
    const kids = this.fnKids(fn);
    return kids.map((k) => {
      if (k.t === 'el') {
        k.tmpl = true;
        return k;
      }
      const w = el('#tmpl', k.s, k.e, { tmpl: true });
      w.kids = [k];
      return w;
    });
  }
}

/** Does the subtree hold an SVG <text> element with text in it? */
function drawsText(k: Kid): boolean {
  if (k.t === 'txt') return false;
  if (k.intrinsic && k.name === 'text') return k.kids.some((x) => (x.t === 'txt' ? !x.attr && x.text.trim() !== '' : true));
  return k.kids.some(drawsText);
}

/** Text of kids made only of text and text-less or inline elements (no blocks, no roles), else null. */
function flatInline(kids: Kid[]): { text: string; lit: string; s: number; e: number; dyn: boolean } | null {
  let text = '';
  let lit = '';
  let s = Infinity;
  let e = -Infinity;
  let dyn = false;
  const walk = (k: Kid): boolean => {
    if (k.t === 'txt') {
      if (k.attr) return false;
      text += k.text;
      lit += k.lit;
      if (k.text.trim()) {
        s = Math.min(s, k.s);
        e = Math.max(e, k.e);
      }
      dyn = dyn || !!k.dyn;
      return true;
    }
    if (k.icon || k.dot) return true;
    if (k.name === '#fragment' || (k.intrinsic && /^(span|strong|em|b|i|small|code)$/.test(k.name) && !k.cls.some((c) => /rounded-full|sr-only/.test(c)))) return k.kids.every(walk);
    return false;
  };
  if (!kids.every(walk)) return null;
  if (s === Infinity) return { text: '', lit: '', s: 0, e: 0, dyn };
  return { text, lit, s, e, dyn };
}


let nextAndGroup = 1;

/** Does a condition test that a collection is empty (see ir.ts emptyCond)? */
function emptyTest(test: N, src: string): 'pos' | 'neg' | null {
  return emptyCond(src.slice(test.start, test.end));
}

function markEmpty(kids: Kid[], at: N): Kid[] {
  return markEmptyKids(kids, at.start, at.end);
}

function markCond(k: Kid): Kid {
  if (k.t === 'el') k.cond = true;
  else k.cond = true;
  return k;
}

// ---------------------------------------------------------------------------------------------------------------

const TOAST_CALL = /^(toast|toast\.(success|error|info|warning|message|loading|promise)|sonner\.toast|notify|notifications\.show|enqueueSnackbar)$/;
const DIALOG_CALL = /^(alert|confirm|prompt|window\.alert|window\.confirm|window\.prompt)$/;

export function jsxToIR(src: string, path: string): El {
  const isTs = /\.(tsx|ts|mts|cts)$/.test(path);
  const ast = parse(src, {
    sourceType: 'unambiguous',
    errorRecovery: true,
    plugins: isTs ? ['jsx', 'typescript', 'decorators-legacy'] : ['jsx', 'flow', 'decorators-legacy'],
  });
  const env: Env = { src, imports: new Map(), consts: new Map(), bindings: collectBindings(ast.program), mutated: collectMutated(ast.program) };
  // imports and string constants
  const scan = (n: N): void => {
    if (!n || typeof n !== 'object') return;
    if (n.type === 'ImportDeclaration') {
      for (const s of n.specifiers) env.imports.set(s.local.name, n.source.value);
      return;
    }
    if (n.type === 'VariableDeclarator' && n.id?.type === 'Identifier' && n.init) {
      const init = unwrap(n.init);
      let v: string | null = null;
      if (init.type === 'StringLiteral') v = init.value;
      else if (init.type === 'TemplateLiteral' && init.expressions.length === 0) v = init.quasis[0].value.cooked ?? '';
      else if (init.type === 'CallExpression' && CLASS_FNS.test(calleeName(init.callee) ?? '')) {
        const ci = classInfo(init, env);
        if (!ci.bound) v = ci.tokens.join(' ');
      }
      if (v !== null) {
        const prev = env.consts.get(n.id.name);
        env.consts.set(n.id.name, prev === undefined || prev === v ? v : null);
      }
    }
    for (const k of Object.keys(n)) {
      if (SKIP_KEYS.has(k)) continue;
      const v = n[k];
      if (Array.isArray(v)) v.forEach(scan);
      else if (v && typeof v.type === 'string') scan(v);
    }
  };
  scan(ast.program);

  const b = new JsxBuilder(env);
  const root = el('#root', 0, src.length, { box: 'root' });
  const roots: Kid[] = [];
  // A component whose own name says what it draws (StatusBadge, TagPill, SectionEyebrow): the markup it returns
  // takes that role at the class stage, so a nearer tag still wins.
  const DEF_ROLE: [RegExp, Role][] = [[/(Badge|Chip|Tag|Pill)$/, 'badge'], [/(Eyebrow|Kicker|Overline)$/, 'eyebrow']];
  const addRoot = (k: Kid, state: boolean, fname: string | null): void => {
    if (k.t === 'txt') return; // bare text returned from a function is not markup
    if (state) {
      k.cond = true;
      if (!k.box) k.box = 'state';
    }
    const dr = fname ? DEF_ROLE.find(([re]) => re.test(fname)) : undefined;
    if (dr && !k.classHint) k.classHint = dr[1];
    roots.push(k);
  };
  const walk = (n: N, inIf: boolean, fname: string | null, emptyIf: boolean): void => {
    if (!n || typeof n !== 'object') return;
    if (isJsx(n)) {
      const r = b.element(n);
      for (const k of Array.isArray(r) ? r : [r]) addRoot(k, false, fname);
      return;
    }
    if (/Function/.test(n.type) && n.body) {
      const name = n.id?.name ?? fname;
      if (n.body.type !== 'BlockStatement') {
        if (containsJsx(n.body)) for (const k of b.exprKids(n.body)) addRoot(k, false, name);
        else walk(n.body, false, name, false);
        return;
      }
      walk(n.body, false, name, false);
      return;
    }
    if (n.type === 'ReturnStatement' && n.argument && containsJsx(n.argument)) {
      // an early return when a collection is empty is a hand-built empty state
      const kids = b.exprKids(n.argument);
      for (const k of emptyIf ? markEmpty(kids, n.argument) : kids) addRoot(k, inIf, fname);
      return;
    }
    if (n.type === 'CallExpression') {
      const name = calleeName(n.callee) ?? '';
      const a0 = n.arguments[0] ? unwrap(n.arguments[0]) : null;
      if ((TOAST_CALL.test(name) || DIALOG_CALL.test(name)) && a0) {
        const tv = textOf(a0, env);
        if (tv && !tv.empty) {
          const isErr = /error/.test(name);
          const box = DIALOG_CALL.test(name) ? 'dialog' : 'toast';
          const r = el(`#${name}`, n.start, n.end, { box, cond: true, hint: box === 'dialog' ? 'dialog_body' : isErr ? 'error' : 'toast' });
          const [s, e] = innerSpan(a0);
          const t = txtFrom(tv, s, e, src);
          if (t) {
            r.kids = [t];
            roots.push(r);
          }
          // a description option on toasts
          const opt = n.arguments[1] ? unwrap(n.arguments[1]) : null;
          if (opt?.type === 'ObjectExpression') {
            for (const p of opt.properties) {
              if (p.type === 'ObjectProperty' && p.key?.name === 'description') {
                const dv = textOf(p.value, env);
                const [ds, de] = innerSpan(unwrap(p.value));
                const dt = dv ? txtFrom(dv, ds, de, src) : null;
                if (dt) r.kids.push(el('#description', ds, de, { kids: [dt], textBlock: true }));
              }
            }
          }
        }
      }
    }
    const nextIf = inIf || n.type === 'IfStatement' || n.type === 'SwitchCase' || n.type === 'CatchClause';
    const em = n.type === 'IfStatement' ? emptyTest(n.test, src) : null;
    for (const k of Object.keys(n)) {
      if (SKIP_KEYS.has(k)) continue;
      const v = n[k];
      const name = n.type === 'VariableDeclarator' && k === 'init' && n.id?.type === 'Identifier' ? n.id.name : fname;
      const empty = n.type === 'IfStatement' ? (k === 'consequent' && em === 'pos') || (k === 'alternate' && em === 'neg') : emptyIf && n.type !== 'IfStatement';
      if (Array.isArray(v)) v.forEach((x) => walk(x, nextIf && k !== 'test', name, empty));
      else if (v && typeof v.type === 'string') walk(v, nextIf && k !== 'test', name, empty);
    }
  };
  walk(ast.program, false, null, false);
  root.kids = roots;
  return root;
}

