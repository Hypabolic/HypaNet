/* Slop labeller client. Talks to serve.py; every decision is POSTed and appended to the label file. */
'use strict';

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const keyOf = (loc) => loc.join(':');
const FIELDS = ['loc', 'class', 'target', 'equiv', 'pattern', 'aside', 'partial', 'ct', 'conf', 'role_ok', 'role_fix'];
const INTERACTIVE = new Set(['button', 'link', 'nav', 'tab', 'menu_item', 'breadcrumb']);
const HEADINGS = new Set(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'dialog_title', 'empty_title']);
const INLINE = new Set(['button', 'link', 'nav', 'tab', 'breadcrumb', 'menu_item', 'badge', 'cell']);
const STOP = new Set('the a an of to and or in on for your you our we is are be this that with it its at by from here there now just simply all every'.split(' '));

const S = {
  session: null, schema: null, ref: null, n: 0, data: null, cur: null, mode: 'normal', input: '', staged: null,
  drafts: {}, overlay: null, showOthers: true, advanceTimer: null, guide: null, guideAlarm: false, refTab: 0,
  listHi: 0, dismissed: new Set(),
};

/* ------------------------------------------------------------------ api */

async function api(path, body) {
  const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

/* ------------------------------------------------------------------ helpers */

const nodes = () => S.data.screen.nodes;
const byKey = (k) => nodes().find((x) => keyOf(x.loc) === k);
const byI = (i) => nodes().find((x) => x.i === i);
const isCtx = (nd) => S.schema.context_roles.includes(nd.role) || (nd.flags || []).some((f) => S.schema.context_flags.includes(f));
const labelable = () => nodes().filter((x) => !isCtx(x));
const curNode = () => (S.cur ? byKey(S.cur) : null);
const rec = (k) => S.data.labels[k];
const complete = (r) => !!r && (r.class != null || (r.aside || []).includes('UNRESOLVED'));

function rank(nd) {
  const r = nd.role;
  if (r === 'label') return (nd.flags || []).some((f) => f === 'in_form' || f === 'in_control') ? 1 : 3;
  if (INTERACTIVE.has(r) || r === 'h1' || r === 'dialog_title' || r === 'empty_title') return 1;
  if (/^h[2-6]$/.test(r)) return 2 + (+r[1] - 2) / 10;
  if (['subtitle', 'helper', 'caption'].includes(r)) return 4;
  if (['eyebrow', 'badge', 'tooltip', 'placeholder'].includes(r)) return 5;
  return 3;
}

function stems(text) {
  return new Set(String(text).toLowerCase().split(/[^a-z0-9]+/)
    .filter((w) => w && !STOP.has(w))
    .map((w) => w.replace(/(ing|ed|es|s)$/, '').replace(/e$/, '')));
}
function overlap(a, b) {
  const A = stems(a), B = stems(b);
  if (!A.size || !B.size) return 0;
  let n = 0;
  A.forEach((x) => { if (B.has(x)) n++; });
  return n / Math.min(A.size, B.size);
}

function suggestTarget(nd, cls) {
  let best = null, bestScore = -Infinity;
  const me = rank(nd);
  for (const o of nodes()) {
    if (o === nd) continue;
    if (isCtx(o) && !(nd.role === 'comment' && o.role === 'code')) continue;
    const ov = overlap(nd.text, o.text);
    const dist = Math.abs(o.i - nd.i);
    let s = ov * 10 - dist * 0.3 + (o.k === nd.k ? 2 : 0);
    if (cls === 'ECHO') {
      const r = rank(o);
      if (!(r < me || (r === me && o.i < nd.i))) continue; // the kept copy outranks (§5)
      if (['eyebrow', 'subtitle'].includes(nd.role) && HEADINGS.has(o.role) && o.k === nd.k) s += 5;
      if (['helper', 'placeholder'].includes(nd.role) && o.role === 'label' && o.k === nd.k) s += 5;
      if (nd.role === 'tooltip' && INTERACTIVE.has(o.role)) s += 3;
      if (nd.role === 'empty_body' && o.role === 'empty_title') s += 5;
      if (nd.role === 'comment' && o.role === 'code') s += 8;
    } else {
      s += INTERACTIVE.has(o.role) ? 3 : -2;
    }
    if (s > bestScore) { bestScore = s; best = o; }
  }
  // a candidate already labelled ECHO is itself a flagged copy: point at the copy it keeps (§5)
  for (let hop = 0; cls === 'ECHO' && best && hop < 6; hop++) {
    const r = rec(keyOf(best.loc));
    if (!r || r.class !== 'ECHO' || !r.target) break;
    const t = byKey(keyOf(r.target));
    if (!t || t === nd) break;
    best = t;
  }
  return best;
}

function suggestPattern(nd, cls, tgt) {
  const f = new Set(nd.flags || []);
  if (cls === 'DECOR') return (f.has('dot') || f.has('pill') || f.has('emoji_lead') || nd.role === 'badge' || /^[●•◉✨★]/.test(nd.text)) ? 'R11' : 'R2';
  if (cls === 'NARRATE') return /\b(below|above|here|this (page|screen|section)|following|left|right|one|two|three|four|five|six|seven|eight|nine|ten)\b/i.test(nd.text) ? 'R16' : 'R6';
  if (cls !== 'ECHO') return null;
  if (tgt && tgt.k !== nd.k && !['tooltip', 'helper', 'placeholder'].includes(nd.role)) return 'R14';
  switch (nd.role) {
    case 'eyebrow': case 'badge': return 'R1';
    case 'subtitle': return 'R3';
    case 'helper': case 'tooltip': case 'caption': return 'R5';
    case 'placeholder': return 'R8';
    case 'comment': return 'R18';
    case 'empty_body': return 'R4';
    default:
      if (HEADINGS.has(nd.role)) return 'R9';
      if ((nd.flags || []).includes('section_last')) return 'R17';
      return 'R4';
  }
}

function baseRecord(k) {
  const r = rec(k);
  const nd = byKey(k);
  const b = r ? JSON.parse(JSON.stringify(Object.fromEntries(FIELDS.map((f) => [f, r[f] ?? null]))))
    : { loc: nd.loc, class: null, target: null, equiv: [], pattern: null, aside: [], partial: false, ct: false, conf: 'clear', role_ok: true, role_fix: null };
  b.equiv = b.equiv || [];
  b.aside = b.aside || [];
  if (!r && S.drafts[k]) Object.assign(b, JSON.parse(JSON.stringify(S.drafts[k])));
  return b;
}

/* ------------------------------------------------------------------ saving */

async function save(recs) {
  const body = {
    screen: S.data.screen.id,
    records: recs.map((r) => {
      const o = {};
      FIELDS.forEach((f) => { o[f] = r[f]; });
      if (o.role_ok) delete o.role_fix;
      return o;
    }),
  };
  let res;
  try { res = await api('/api/label', body); } catch (err) { toast(err.message, 'error'); return null; }
  for (const r of res.records) { S.data.labels[keyOf(r.loc)] = r; delete S.drafts[keyOf(r.loc)]; }
  afterWrite(res);
  (res.warnings || []).forEach((w) => toast(w, 'warn'));
  return res;
}

async function saveScreenRecord(kind, value) {
  let res;
  try { res = await api('/api/screen_record', { screen: S.data.screen.id, kind, value }); } catch (err) { toast(err.message, 'error'); return null; }
  S.data[kind] = value;
  afterWrite(res);
  (res.warnings || []).forEach((w) => toast(w, 'warn'));
  return res;
}

function afterWrite(res) {
  S.data.summary = res.summary;
  S.data.done = res.summary.done;
  S.session.screens[S.n] = res.summary;
  handleGuide(res.guide);
  setStatus(`saved ${new Date().toLocaleTimeString()}`);
  renderAll();
  if (res.done_written) onScreenDone();
}

function onScreenDone() {
  toast(`Screen ${S.n + 1} done: every node labelled`, 'ok');
  const N = S.session.screens.length;
  let next = -1;
  for (let d = 1; d <= N; d++) { const m = (S.n + d) % N; if (!S.session.screens[m].done) { next = m; break; } }
  clearTimeout(S.advanceTimer);
  if (next < 0) { toast('All screens in this set are done. Run check.py on the label file.', 'ok'); return; }
  S.advanceTimer = setTimeout(() => loadScreen(next), 450);
}

/* ------------------------------------------------------------------ actions */

async function setClass(k, cls, extra = {}) {
  const b = baseRecord(k);
  const prev = b.class;
  b.class = cls;
  if (prev !== cls) { b.target = null; b.equiv = []; b.pattern = null; }
  Object.assign(b, extra);
  if (cls !== 'KEEP' && b.partial) { b.partial = false; toast('partial dropped: it applies to KEEP only (§4 step 5)', 'warn'); }
  const res = await save([b]);
  if (res && !res.done_written) moveCursor(1);
}

async function modify(k, fn, label) {
  const r = rec(k);
  if (r) {
    const b = baseRecord(k);
    if (fn(b) === false) return;
    if (b.class == null && !b.aside.includes('UNRESOLVED')) { toast('label the node first (k e n d u)', 'warn'); return; }
    await save([b]);
  } else {
    const d = S.drafts[k] || {};
    const b = baseRecord(k);
    if (fn(b) === false) return;
    if (b.aside.includes('UNRESOLVED')) { await save([b]); return; }
    for (const f of ['partial', 'ct', 'conf', 'role_ok', 'role_fix', 'aside']) d[f] = b[f];
    S.drafts[k] = d;
    setStatus(`${label} staged; saved with the label`);
    renderAll();
  }
}

function toggleConf() { const k = S.cur; modify(k, (b) => { b.conf = b.conf === 'clear' ? 'judgement' : 'clear'; }, 'conf'); }
function toggleCt() { const k = S.cur; modify(k, (b) => { b.ct = !b.ct; }, 'ct'); }
function togglePartial() {
  const k = S.cur;
  modify(k, (b) => {
    if (!b.partial && b.class && b.class !== 'KEEP') {
      toast(`partial is KEEP with a new clause (§4 step 5); this node is ${b.class}. Press k, then p.`, 'warn');
      return false;
    }
    b.partial = !b.partial;
  }, 'partial');
}
function toggleUnresolved() {
  const k = S.cur;
  modify(k, (b) => {
    const has = b.aside.includes('UNRESOLVED');
    b.aside = has ? b.aside.filter((a) => a !== 'UNRESOLVED') : [...b.aside, 'UNRESOLVED'];
    if (has && b.class == null) { toast('label the node first, or keep UNRESOLVED', 'warn'); return false; }
  }, 'UNRESOLVED');
}

function moveCursor(d) {
  const L = labelable();
  if (!L.length) return;
  let idx = L.findIndex((x) => keyOf(x.loc) === S.cur);
  idx = idx < 0 ? 0 : Math.max(0, Math.min(L.length - 1, idx + d));
  S.cur = keyOf(L[idx].loc);
  renderCursor();
}

function setCursor(k) {
  const nd = byKey(k);
  if (!nd) return;
  if (isCtx(nd)) { toast(`node ${nd.i} is context only (${nd.role}${(nd.flags || []).includes('sr_only') ? ', sr_only' : ''}); not labelled (§2)`, 'warn'); return; }
  S.cur = k;
  renderCursor();
}

async function loadScreen(n) {
  clearTimeout(S.advanceTimer);
  const N = S.session.screens.length;
  if (n < 0 || n >= N) { toast(n < 0 ? 'first screen' : 'last screen'); return; }
  let d;
  try { d = await api(`/api/screen?n=${n}`); } catch (err) { toast(err.message, 'error'); return; }
  S.n = n; S.data = d; S.drafts = {}; S.mode = 'normal'; S.staged = null; S.input = '';
  S.session.screens[n] = d.summary;
  const L = labelable();
  const first = L.find((x) => !complete(rec(keyOf(x.loc)))) || L[0];
  S.cur = first ? keyOf(first.loc) : null;
  closeDrawer();
  renderAll(true);
  try { localStorage.setItem(`slop-pos-${S.session.set}-${S.session.labeller}`, String(n)); } catch (e) { /* storage off */ }
  if (d.summary.labelled === d.summary.nodes && !d.done) {
    try { const res = await api('/api/done', { screen: d.screen.id }); S.data.done = true; S.session.screens[n] = res.summary; renderHeader(); } catch (err) { toast(err.message, 'error'); }
  }
}

/* ------------------------------------------------------------------ modes */

function startPointer(cls) {
  const nd = curNode();
  if (!nd) return;
  S.mode = 'pointer';
  S.input = '';
  const sug = suggestTarget(nd, cls);
  S.staged = { cls, key: S.cur, suggest: sug ? keyOf(sug.loc) : null };
  renderAll();
}

function pickTarget(tk) {
  const st = S.staged;
  if (tk === st.key) { toast('a node cannot point at itself', 'warn'); return; }
  st.target = tk ? byKey(tk).loc : null;
  startPattern(st.cls, st);
}

function startPattern(cls, staged, reopen = false) {
  const nd = byKey(staged.key);
  const codes = cls === 'KEEP' ? S.schema.k_rules : (S.schema.patterns[cls] || []);
  if (!codes.length) { toast(`${cls} has no pattern codes`, 'warn'); return; }
  const tgt = staged.target ? byKey(keyOf(staged.target)) : null;
  let sug = reopen ? (rec(staged.key) || {}).pattern : suggestPattern(nd, cls, tgt);
  if (!codes.includes(sug)) sug = cls === 'KEEP' ? null : codes[0];
  S.mode = 'pattern';
  S.input = '';
  S.staged = Object.assign(staged, { cls, codes, reopen, hi: Math.max(0, codes.indexOf(sug)) });
  renderAll();
}

async function commitPattern(code) {
  const st = S.staged;
  S.mode = 'normal'; S.staged = null; S.input = '';
  closeDrawer();
  if (st.reopen) {
    await modify(st.key, (b) => { b.pattern = code; }, 'pattern');
  } else {
    await setClass(st.key, st.cls, { target: st.target ?? null, pattern: code });
  }
}

function startEquiv() {
  const r = rec(S.cur);
  if (!r || r.class !== 'ECHO') { toast('equiv lists the other flagged copies of an ECHO stack (§5): label this node ECHO first', 'warn'); return; }
  const sel = new Set((r.equiv || []).map(keyOf));
  if (!sel.size && r.target) {
    for (const [k, o] of Object.entries(S.data.labels)) {
      if (k !== S.cur && o.class === 'ECHO' && o.target && keyOf(o.target) === keyOf(r.target)) sel.add(k);
    }
  }
  S.mode = 'equiv'; S.input = ''; S.staged = { key: S.cur, sel };
  renderAll();
}

function toggleEquiv(k) {
  const st = S.staged;
  const r = rec(st.key);
  if (k === st.key) return;
  if (r.target && k === keyOf(r.target)) { toast('that is the kept copy (the target); equiv lists the other flagged copies', 'warn'); return; }
  if (isCtx(byKey(k))) { toast('context-only node', 'warn'); return; }
  if (st.sel.has(k)) st.sel.delete(k); else st.sel.add(k);
  renderAll();
}

async function commitEquiv() {
  const st = S.staged;
  const r = rec(st.key);
  const M = new Set([...st.sel, st.key]);
  const O = new Set([...(r.equiv || []).map(keyOf), st.key]);
  const skipped = [...M].filter((k) => { const o = rec(k); return o && o.class && o.class !== 'ECHO'; });
  skipped.forEach((k) => M.delete(k));
  const out = [], newly = [];
  for (const k of new Set([...M, ...O])) {
    let b;
    if (M.has(k)) {
      b = baseRecord(k);
      if (b.class == null) { b.class = 'ECHO'; b.target = r.target; b.pattern = r.pattern; newly.push(byKey(k).i); }
      b.equiv = [...M].filter((x) => x !== k).map((x) => byKey(x).loc);
    } else {
      const o = rec(k);
      if (!o || o.class !== 'ECHO') continue;
      b = baseRecord(k);
      b.equiv = b.equiv.filter((x) => !M.has(keyOf(x)));
    }
    out.push(b);
  }
  S.mode = 'normal'; S.staged = null; S.input = '';
  closeDrawer();
  if (skipped.length) toast(`not in the stack (labelled another class): ${skipped.map((k) => '#' + byKey(k).i).join(', ')}`, 'warn');
  const res = await save(out);
  if (res && newly.length) toast(`labelled ${newly.map((i) => '#' + i).join(', ')} ECHO with the same target and pattern (stack)`, 'ok');
}

function startRole() {
  const nd = curNode();
  if (!nd) return;
  S.mode = 'role'; S.staged = { key: S.cur, filter: '', hi: 0 };
  renderDrawer();
  const inp = $('#role-filter');
  if (inp) inp.focus();
}

function roleOptions() {
  const f = S.staged.filter.trim().toLowerCase();
  return S.schema.roles.filter((r) => !f || r.includes(f)).sort((a, b) => (a.startsWith(f) ? 0 : 1) - (b.startsWith(f) ? 0 : 1));
}

async function pickRole(role) {
  const k = S.staged.key;
  const nd = byKey(k);
  S.mode = 'normal'; S.staged = null; closeDrawer();
  await modify(k, (b) => {
    if (role === nd.role) { b.role_ok = true; b.role_fix = null; } else { b.role_ok = false; b.role_fix = role; }
  }, 'role');
}

function startAside() {
  S.mode = 'aside'; S.staged = { key: S.cur, set: new Set(baseRecord(S.cur).aside) };
  renderAll();
}

async function commitAside() {
  const st = S.staged;
  const list = S.schema.node_asides.filter((a) => st.set.has(a));
  S.mode = 'normal'; S.staged = null; closeDrawer();
  await modify(st.key, (b) => {
    if (b.class == null && !list.includes('UNRESOLVED') && rec(st.key)) { toast('label the node first, or keep UNRESOLVED', 'warn'); return false; }
    b.aside = list;
  }, 'asides');
}

function startMissed() { S.mode = 'missed'; S.staged = {}; renderDrawer(); const i = $('#missed-text'); if (i) i.focus(); }
function startScreenAside() { S.mode = 'saside'; S.staged = {}; renderDrawer(); const i = $('#saside-nodes'); if (i) i.focus(); }

function cancelMode() {
  S.mode = 'normal'; S.staged = null; S.input = '';
  closeDrawer();
  renderAll();
}

/* ------------------------------------------------------------------ keyboard */

window.addEventListener('keydown', (e) => {
  if (!S.data) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (S.overlay === 'guide') { if (e.key === 'Enter') ackGuide(); e.preventDefault(); return; }
  if (S.overlay) { overlayKey(e); return; }
  const handler = { normal: normalKey, goto: gotoKey, pointer: pointerKey, pattern: patternKey, equiv: equivKey, role: roleKey, aside: asideKey, missed: formKey, saside: formKey }[S.mode];
  if (handler) handler(e);
});

function consume(e) { e.preventDefault(); e.stopPropagation(); }

function normalKey(e) {
  const k = e.key;
  const nd = curNode();
  const needNode = 'kenduxcptwvar'.includes(k) && k.length === 1;
  if (needNode && !nd) { toast('no labelable node on this screen'); consume(e); return; }
  switch (k) {
    case 'k': setClass(S.cur, 'KEEP'); break;
    case 'u': setClass(S.cur, 'UNSURE'); break;
    case 'e': startPointer('ECHO'); break;
    case 'n': startPointer('NARRATE'); break;
    case 'd': startPattern('DECOR', { key: S.cur }); break;
    case 'x': toggleUnresolved(); break;
    case 'c': toggleConf(); break;
    case 'p': togglePartial(); break;
    case 't': toggleCt(); break;
    case 'w': {
      const r = rec(S.cur);
      if (!r || !r.class || r.class === 'UNSURE') { toast('label the node first; UNSURE has no code', 'warn'); break; }
      startPattern(r.class, { key: S.cur, target: r.target }, true);
      break;
    }
    case 'v': startEquiv(); break;
    case 'a': startAside(); break;
    case 'r': startRole(); break;
    case 'm': startMissed(); break;
    case 's': startScreenAside(); break;
    case 'j': case 'ArrowDown': moveCursor(1); break;
    case 'ArrowUp': moveCursor(-1); break;
    case 'N': case ']': loadScreen(S.n + 1); break;
    case 'P': case '[': loadScreen(S.n - 1); break;
    case '?': openOverlay('help'); break;
    case 'g': openOverlay('ref'); break;
    case 'l': openOverlay('list'); break;
    case '\\': toggleTheme(); break;
    case 'o':
      if (S.session.blind) toast('blind mode: other labellers are never shown');
      else { S.showOthers = !S.showOthers; renderNodes(); }
      break;
    case 'Escape': break;
    default:
      if (/^[0-9]$/.test(k)) { S.mode = 'goto'; S.input = k; renderFooter(); break; }
      return;
  }
  consume(e);
}

function digitInput(e) {
  if (/^[0-9]$/.test(e.key)) { S.input += e.key; return true; }
  if (e.key === 'Backspace') { S.input = S.input.slice(0, -1); return true; }
  return false;
}

function gotoKey(e) {
  if (digitInput(e)) { renderFooter(); consume(e); return; }
  if (e.key === 'Enter') {
    const typed = S.input;
    const nd = typed ? byI(+typed) : null;
    S.mode = 'normal'; S.input = '';
    if (nd) setCursor(keyOf(nd.loc)); else toast(`no node ${typed}`, 'warn');
    renderFooter();
  } else if (e.key === 'Escape') { S.mode = 'normal'; S.input = ''; renderFooter(); }
  consume(e);
}

function pointerKey(e) {
  if (digitInput(e)) { renderFooter(); consume(e); return; }
  const st = S.staged;
  if (e.key === 'Enter') {
    if (S.input) {
      const nd = byI(+S.input);
      if (!nd) { toast(`no node ${S.input}`, 'warn'); S.input = ''; renderFooter(); }
      else pickTarget(keyOf(nd.loc));
    } else pickTarget(st.suggest);
  } else if (e.key === '-') pickTarget(null);
  else if (e.key === 'Escape') cancelMode();
  else if (e.key === '?') openOverlay('help');
  else if (e.key === 'g') openOverlay('ref');
  else return;
  consume(e);
}

function patternKey(e) {
  if (digitInput(e)) { renderDrawer(); renderFooter(); consume(e); return; }
  const st = S.staged;
  const pre = st.cls === 'KEEP' ? 'K' : 'R';
  if (e.key === 'Enter') {
    const code = S.input ? pre + S.input : st.codes[st.hi];
    if (!st.codes.includes(code)) { toast(`${code} is not a ${st.cls} code (${st.codes.join(' ')})`, 'warn'); S.input = ''; renderDrawer(); renderFooter(); }
    else commitPattern(code);
  } else if (e.key === 'ArrowDown' || e.key === 'j') { st.hi = Math.min(st.codes.length - 1, st.hi + 1); S.input = ''; renderDrawer(); }
  else if (e.key === 'ArrowUp') { st.hi = Math.max(0, st.hi - 1); S.input = ''; renderDrawer(); }
  else if (e.key === '-') commitPattern(null);
  else if (e.key === 'Escape') cancelMode();
  else if (e.key === 'g') openOverlay('ref');
  else return;
  consume(e);
}

function equivKey(e) {
  if (digitInput(e)) { renderFooter(); consume(e); return; }
  if (e.key === ' ' || e.key === ',' || (e.key === 'Enter' && S.input)) {
    const nd = byI(+S.input);
    S.input = '';
    if (nd) toggleEquiv(keyOf(nd.loc)); else renderFooter();
  } else if (e.key === 'Enter') commitEquiv();
  else if (e.key === 'Escape') cancelMode();
  else return;
  consume(e);
}

function roleKey(e) {
  const st = S.staged;
  const opts = roleOptions();
  if (e.key === 'Enter') { if (opts[st.hi]) pickRole(opts[st.hi]); consume(e); }
  else if (e.key === 'Escape') { cancelMode(); consume(e); }
  else if (e.key === 'ArrowDown') { st.hi = Math.min(opts.length - 1, st.hi + 1); renderRoleList(); consume(e); }
  else if (e.key === 'ArrowUp') { st.hi = Math.max(0, st.hi - 1); renderRoleList(); consume(e); }
}

function asideKey(e) {
  const st = S.staged;
  const list = S.schema.node_asides;
  if (/^[1-9]$/.test(e.key)) {
    const a = list[+e.key - 1];
    if (a) { if (st.set.has(a)) st.set.delete(a); else st.set.add(a); renderDrawer(); }
  } else if (e.key === 'Enter') commitAside();
  else if (e.key === 'Escape') cancelMode();
  else return;
  consume(e);
}

function formKey(e) {
  if (e.key === 'Escape') { cancelMode(); consume(e); return; }
  if (e.key === 'Enter') { (S.mode === 'missed' ? addMissed : addScreenAside)(); consume(e); }
}

/* ------------------------------------------------------------------ missed and screen asides */

function addMissed() {
  const line = parseInt($('#missed-line').value, 10);
  const text = $('#missed-text').value.trim();
  const cls = $('#missed-class').value;
  if (!(line >= 1) || !text) { toast('missed node needs a line and its text', 'warn'); return; }
  const list = [...(S.data.missed || []), { line, text, class: cls }];
  saveScreenRecord('missed', list).then((r) => { if (r) { renderDrawer(); const i = $('#missed-text'); if (i) i.focus(); } });
}
function removeMissed(idx) {
  const list = (S.data.missed || []).filter((_, n) => n !== idx);
  saveScreenRecord('missed', list).then(() => renderDrawer());
}

function parseNodeList(s) {
  const out = [];
  for (const t of s.split(/[\s,]+/).filter(Boolean)) {
    const nd = byI(+t);
    if (!nd || !/^\d+$/.test(t)) { toast(`no node ${t}`, 'warn'); return null; }
    if (!out.includes(nd)) out.push(nd);
  }
  return out;
}
function addScreenAside() {
  const code = $('#saside-code').value;
  const list = parseNodeList($('#saside-nodes').value);
  if (!list) return;
  if (!list.length) { toast('list the nodes the device appears on', 'warn'); return; }
  const items = [...(S.data.aside_screen || []).filter((it) => it.code !== code), { code, nodes: list.map((n) => n.loc) }];
  saveScreenRecord('aside_screen', items).then((r) => { if (r) { renderDrawer(); const i = $('#saside-nodes'); if (i) i.focus(); } });
}
function removeScreenAside(idx) {
  const items = (S.data.aside_screen || []).filter((_, n) => n !== idx);
  saveScreenRecord('aside_screen', items).then(() => renderDrawer());
}

/* ------------------------------------------------------------------ clicks */

function onNodeClick(k) {
  switch (S.mode) {
    case 'pointer': pickTarget(k); break;
    case 'equiv': toggleEquiv(k); break;
    case 'saside': {
      const nd = byKey(k); const inp = $('#saside-nodes');
      if (inp && nd) { const parts = inp.value.split(/[\s,]+/).filter(Boolean); const s = String(nd.i); inp.value = (parts.includes(s) ? parts.filter((p) => p !== s) : [...parts, s]).join(' '); inp.focus(); }
      break;
    }
    case 'normal': case 'goto': S.mode = 'normal'; S.input = ''; setCursor(k); break;
    default: break;
  }
}
document.addEventListener('click', (e) => {
  const el = e.target.closest('[data-key]');
  if (el && S.data && !S.overlay) { onNodeClick(el.dataset.key); return; }
});

/* ------------------------------------------------------------------ guide */

function handleGuide(g) {
  if (!g) return;
  S.guide = g;
  if (g.changed && !S.guideAlarm) {
    S.guideAlarm = true;
    $('#guide-msg').innerHTML = `Session started with <code>${esc(g.baseline.slice(0, 16))}</code>; the file now hashes to <code>${esc((g.current || 'unreadable').slice(0, 16))}</code>.`;
    openOverlay('guide');
  }
  renderBanners();
}

async function ackGuide() {
  try {
    const g = await api('/api/guide/ack', {});
    S.guideChangedAt = new Date().toLocaleTimeString();
    S.guideAlarm = false;
    S.guide = g;
    closeOverlay();
    S.ref = null;
    renderBanners();
  } catch (err) { toast(err.message, 'error'); }
}

async function pollGuide() {
  try { handleGuide(await api('/api/guide')); } catch (e) { setStatus('server unreachable'); }
}

/* ------------------------------------------------------------------ overlays */

async function openOverlay(name) {
  S.overlay = name;
  $('#overlay').hidden = false;
  $$('#overlay .ov-box').forEach((b) => { b.hidden = b.id !== `ov-${name}`; });
  if (name === 'ref') {
    if (!S.ref) { try { S.ref = await api('/api/reference'); } catch (err) { toast(err.message, 'error'); return; } }
    renderRef();
  }
  if (name === 'list') { S.listHi = S.n; renderList(); }
}
function closeOverlay() { S.overlay = null; $('#overlay').hidden = true; }

const REF_TABS = [['Decision order', '4'], ['K rules', '6'], ['Patterns', '9'], ['Consequence test', '7'], ['Precedence', '5'], ['Classes', '3'], ['Worked cases', '8']];

function overlayKey(e) {
  if (e.key === 'Escape' || (e.key === '?' && S.overlay === 'help') || (e.key === 'g' && S.overlay === 'ref') || (e.key === 'l' && S.overlay === 'list')) { closeOverlay(); consume(e); return; }
  if (S.overlay === 'ref' && /^[1-7]$/.test(e.key)) { S.refTab = +e.key - 1; renderRef(); consume(e); return; }
  if (S.overlay === 'list') {
    const N = S.session.screens.length;
    if (e.key === 'ArrowDown' || e.key === 'j') { S.listHi = Math.min(N - 1, S.listHi + 1); renderList(); }
    else if (e.key === 'ArrowUp') { S.listHi = Math.max(0, S.listHi - 1); renderList(); }
    else if (e.key === 'Enter') { closeOverlay(); loadScreen(S.listHi); }
    else return;
    consume(e);
  }
}
$('#overlay').addEventListener('click', (e) => { if (e.target.id === 'overlay' && S.overlay !== 'guide') closeOverlay(); });

function md(text) {
  const inline = (s) => esc(s).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>').replace(/(^|[\s(])\*([^*]+)\*/g, '$1<em>$2</em>');
  const lines = text.split('\n');
  let html = '', i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (/^#{2,4} /.test(l)) { const h = l.match(/^#+/)[0].length; html += `<h${h}>${inline(l.replace(/^#+ /, ''))}</h${h}>`; i++; continue; }
    if (l.startsWith('|')) {
      const rows = [];
      while (i < lines.length && lines[i].startsWith('|')) { rows.push(lines[i]); i++; }
      const cells = (r) => r.replace(/^\|/, '').replace(/\|\s*$/, '').split('|').map((c) => c.trim());
      html += '<table>';
      rows.forEach((r, n) => {
        if (/^\|[\s:|-]+\|?$/.test(r)) return;
        const t = n === 0 ? 'th' : 'td';
        html += '<tr>' + cells(r).map((c) => `<${t}>${inline(c)}</${t}>`).join('') + '</tr>';
      });
      html += '</table>';
      continue;
    }
    if (/^(- |\d+\. )/.test(l)) {
      const ordered = /^\d+\. /.test(l);
      html += ordered ? '<ol>' : '<ul>';
      while (i < lines.length && /^(- |\d+\. )/.test(lines[i])) {
        let item = lines[i].replace(/^(- |\d+\. )/, ''); i++;
        while (i < lines.length && /^ {2,}\S/.test(lines[i])) { item += ' ' + lines[i].trim(); i++; }
        html += `<li>${inline(item)}</li>`;
      }
      html += ordered ? '</ol>' : '</ul>';
      continue;
    }
    if (!l.trim()) { i++; continue; }
    let p = l; i++;
    while (i < lines.length && lines[i].trim() && !/^(#|\||- |\d+\. )/.test(lines[i])) { p += ' ' + lines[i]; i++; }
    html += `<p>${inline(p)}</p>`;
  }
  return html;
}

function renderRef() {
  $('#ref-tabs').innerHTML = REF_TABS.map(([t], n) => `<button class="${n === S.refTab ? 'on' : ''}" data-tab="${n}"><kbd>${n + 1}</kbd> ${esc(t)}</button>`).join('');
  $$('#ref-tabs button').forEach((b) => { b.onclick = () => { S.refTab = +b.dataset.tab; renderRef(); }; });
  const sec = S.ref.sections[REF_TABS[S.refTab][1]];
  $('#ref-body').innerHTML = sec ? md(sec) : '<p class="muted">Section not found in LABELLING.md.</p>';
}

function renderList() {
  const rows = S.session.screens.map((s, n) => `<tr class="${n === S.listHi ? 'hi' : ''} ${n === S.n ? 'cur' : ''}" data-n="${n}">
    <td>${n + 1}</td><td>${s.done ? '✓' : (s.labelled ? '…' : '')}</td><td class="p">${esc(s.repo || '')}</td><td class="p">${esc(s.path || s.file)}</td>
    <td>${s.labelled} / ${s.nodes}</td></tr>`).join('');
  $('#list-body').innerHTML = `<table class="list-table">${rows}</table>`;
  $$('#list-body tr').forEach((tr) => { tr.onclick = () => { closeOverlay(); loadScreen(+tr.dataset.n); }; });
  const hi = $('#list-body tr.hi');
  if (hi) hi.scrollIntoView({ block: 'nearest' });
}

/* ------------------------------------------------------------------ rendering */

function renderAll(scroll = false) {
  renderHeader();
  renderBanners();
  renderNodes();
  renderPreview();
  if (scroll || !$('#source').dataset.screen || $('#source').dataset.screen !== S.data.screen.id) renderSource();
  else updateSourceMarks();
  renderDrawer();
  renderFooter();
  scrollToCursor();
}

function renderCursor() {
  $$('#nodes .row').forEach((r) => r.classList.toggle('cur', r.dataset.key === S.cur));
  $$('#preview .pv').forEach((r) => r.classList.toggle('cur', r.dataset.key === S.cur));
  updateSourceMarks();
  renderFooter();
  scrollToCursor();
}

function renderHeader() {
  const ss = S.session;
  const done = ss.screens.filter((s) => s.done).length;
  const N = ss.screens.length;
  const sum = S.data.summary;
  $('#sess').textContent = `${ss.set} · ${ss.labeller} · ${ss.blind ? 'blind' : 'compare mode'}`;
  const src = S.data.screen.source || {};
  $('#where').innerHTML = `Screen <strong>${S.n + 1}</strong> of ${N} · <span class="path">${esc(src.repo ? src.repo + '/' : '')}${esc(src.path || S.data.file)}</span>${S.data.done ? ' · <span style="color:var(--KEEP)">done</span>' : ''}`;
  $('#prog-text').textContent = `nodes ${sum.labelled} / ${sum.nodes} · screens done ${done} / ${N}`;
  $('#prog-bar').style.width = `${N ? (100 * done) / N : 0}%`;
  $('#kind').textContent = S.data.kind ? `product: ${S.data.kind}` : 'product kind not recorded (§2a)';
  $('#kind').style.color = S.data.kind ? '' : 'var(--warn-text)';
}

function renderBanners() {
  const out = [];
  const g = S.guide;
  if (g && g.changed && !S.guideAlarm) out.push(['alarm', 'guide-changed', `LABELLING.md changed during this session (${g.baseline.slice(0, 12)} → ${(g.current || '?').slice(0, 12)}); acknowledge with Enter.`]);
  if (S.guideChangedAt) out.push(['alarm', 'guide-acked', `LABELLING.md changed at ${S.guideChangedAt}. Labels before then carry the old hash and will need relabelling before the freeze.`]);
  if (g && g.older_records) out.push(['warn', 'older', `${g.older_records} records in this label file carry a different guide hash from the current LABELLING.md.`]);
  (S.session.screen_problems || []).slice(0, 3).forEach((p, n) => out.push(['warn', 'sp' + n, `Screens: ${p}`]));
  (S.session.guide_problems || []).forEach((p, n) => out.push(['warn', 'gp' + n, `Guide parse: ${p}`]));
  (S.session.bad_lines || []).slice(0, 3).forEach(([ln, e], n) => out.push(['warn', 'bl' + n, `Label file line ${ln}: ${e}`]));
  $('#banners').innerHTML = out.filter(([, id]) => !S.dismissed.has(id)).map(([cls, id, msg]) => `<div class="banner ${cls}"><span>${esc(msg)}</span>${cls === 'warn' ? `<button data-dismiss="${id}">dismiss</button>` : ''}</div>`).join('');
  $$('#banners [data-dismiss]').forEach((b) => { b.onclick = () => { S.dismissed.add(b.dataset.dismiss); renderBanners(); }; });
}

// The extractor's markers (SPEC.md §4): U+E000 stands for an interpolation, U+E001 separates the branches of a
// conditional. The letters VAR and ALT in copy are ordinary text.
const MARKER_CHIPS = {
  '\uE000': '<span class="var" title="interpolated value">VAR</span>',
  '\uE001': '<span class="var alt" title="or (conditional branch)">ALT</span>',
};
function textHtml(t) {
  return esc(t).replace(/[\uE000\uE001]/g, (m) => MARKER_CHIPS[m]);
}

function labelDetail(k) {
  const r = rec(k);
  const d = S.drafts[k];
  const parts = [];
  const src = r || (d ? baseRecord(k) : null);
  if (!src) return '';
  if (src.target) { const t = byKey(keyOf(src.target)); parts.push(`→ #${t ? t.i : '?'}`); }
  else if (src.class === 'ECHO' || src.class === 'NARRATE') parts.push('→ none');
  if (src.equiv && src.equiv.length) parts.push(`≡ ${src.equiv.map((x) => '#' + (byKey(keyOf(x)) || { i: '?' }).i).join(' ')}`);
  if (src.pattern) parts.push(src.pattern);
  if (src.conf === 'judgement') parts.push('judgement');
  if (src.partial) parts.push('partial');
  if (src.ct) parts.push('ct');
  if (src.role_ok === false) parts.push(`role→${src.role_fix}`);
  (src.aside || []).forEach((a) => parts.push(a));
  if (!parts.length) return '';
  return `<div class="det">${r ? esc(parts.join(' · ')) : `<span class="staged">staged: ${esc(parts.join(' · '))}</span>`}</div>`;
}

function chipFor(k) {
  const r = rec(k);
  if (!r) return '<span class="chip empty">—</span>';
  if (r.class == null) return '<span class="chip UNRESOLVED">UNRESOLVED</span>';
  return `<span class="chip ${r.class}">${r.class}</span>`;
}

function modeMarks(k) {
  const c = [];
  if (S.mode === 'pointer' && S.staged && S.staged.suggest === k) c.push('suggest');
  if (S.mode === 'equiv' && S.staged && S.staged.sel.has(k)) c.push('sel');
  return c.join(' ');
}

function renderNodes() {
  const sc = S.data.screen;
  const byK = new Map();
  sc.nodes.forEach((nd) => { if (!byK.has(nd.k)) byK.set(nd.k, []); byK.get(nd.k).push(nd); });
  const kinds = new Map((sc.clusters || []).map((c) => [c.k, c]));
  const others = S.data.others && S.showOthers ? S.data.others : null;
  let html = '';
  let prevK = null;
  for (const nd of sc.nodes) {
    if (nd.k !== prevK) {
      if (prevK !== null) html += '</div>';
      const c = kinds.get(nd.k) || {};
      html += `<div class="cluster"><div class="cluster-h"><span>${esc(c.kind || 'cluster')}</span><span class="faint">k${nd.k}${c.line ? ' · line ' + c.line : ''}</span></div>`;
      prevK = nd.k;
    }
    const k = keyOf(nd.loc);
    const ctx = isCtx(nd);
    const r = rec(k) || (S.drafts[k] ? baseRecord(k) : null);
    const fixed = r && r.role_ok === false;
    const tags = [`<span class="tag role ${fixed ? 'fixed' : ''}">${esc(nd.role)}</span>`];
    if (fixed) tags.push(`<span class="tag fix">${esc(r.role_fix)}</span>`);
    (nd.flags || []).forEach((f) => tags.push(`<span class="tag">${esc(f)}</span>`));
    if (ctx) tags.push('<span class="tag">context only</span>');
    let oth = '';
    if (others) {
      oth = Object.entries(others).map(([who, m]) => {
        const o = m[k];
        if (!o) return '';
        const t = o.target ? byKey(keyOf(o.target)) : null;
        return `<div class="others">${esc(who)}: ${esc(o.class || 'UNRESOLVED')}${t ? ' → #' + t.i : ''}${o.pattern ? ' ' + esc(o.pattern) : ''}</div>`;
      }).join('');
    }
    html += `<div class="row ${k === S.cur ? 'cur' : ''} ${ctx ? 'ctx' : ''} ${modeMarks(k)}" data-key="${esc(k)}">
      <div class="num">${nd.i}</div>
      <div class="main"><div class="text">${textHtml(nd.text)}</div><div class="tags">${tags.join('')}</div>${oth}</div>
      <div class="lab">${ctx ? '' : chipFor(k)}${ctx ? '' : labelDetail(k)}</div></div>`;
  }
  if (prevK !== null) html += '</div>';
  if (!sc.nodes.length) html = '<p class="muted" style="padding:18px">This screen has no nodes.</p>';
  $('#nodes').innerHTML = html;
}

function pvNode(nd) {
  const k = keyOf(nd.loc);
  const f = new Set(nd.flags || []);
  const r = rec(k);
  const cls = ['pv', `pv-${nd.role}`];
  ['caps', 'tracked', 'small', 'mono', 'pill', 'dynamic'].forEach((x) => { if (f.has(x)) cls.push(`f-${x}`); });
  if (f.has('conditional')) cls.push('cond');
  if (r && r.class) cls.push(`lab-${r.class}`);
  if (k === S.cur) cls.push('cur');
  const mm = modeMarks(k);
  if (mm) cls.push(mm);
  let text = textHtml(nd.text);
  if (f.has('dot') && !/^[●•◉·]/.test(nd.text)) text = '<span style="color:var(--KEEP)">●</span> ' + text;
  if (f.has('icon')) text = '◇ ' + text;
  const title = `#${nd.i} ${nd.role}${f.has('conditional') ? ' (conditional)' : ''}`;
  return `<span class="${cls.join(' ')}" data-key="${esc(k)}" title="${esc(title)}">${text}</span>`;
}

function renderPreview() {
  const sc = S.data.screen;
  const kinds = new Map((sc.clusters || []).map((c) => [c.k, c.kind]));
  const groups = [];
  sc.nodes.forEach((nd) => {
    const last = groups[groups.length - 1];
    if (!last || last.k !== nd.k) groups.push({ k: nd.k, nodes: [nd] }); else last.nodes.push(nd);
  });
  const hidden = [];
  let html = '<div class="pv-screen">';
  for (const g of groups) {
    html += `<div class="pv-cluster kind-${esc(kinds.get(g.k) || 'body')}"><span class="pv-kind">${esc(kinds.get(g.k) || '')}</span>`;
    let row = null;
    for (const nd of g.nodes) {
      if (isCtx(nd)) { hidden.push(nd); continue; }
      const inline = INLINE.has(nd.role) || (nd.role === 'tooltip');
      if (inline) {
        if (!row) { html += '<div class="pv-row">'; row = true; }
        html += pvNode(nd);
      } else {
        if (row) { html += '</div>'; row = null; }
        html += pvNode(nd);
      }
    }
    if (row) html += '</div>';
    html += '</div>';
  }
  html += '</div>';
  if (hidden.length) {
    html += `<div class="pv-screen pv-hidden">Not visible (context only): ${hidden.map((nd) => `<span class="pv" data-key="${esc(keyOf(nd.loc))}">#${nd.i} ${esc(nd.role)} “${textHtml(nd.text)}”</span>`).join(' · ')}</div>`;
  }
  $('#preview').innerHTML = html;
}

function renderSource() {
  const src = S.data.source || {};
  const el = $('#source');
  el.dataset.screen = S.data.screen.id;
  const sp = S.data.screen.source || {};
  $('#src-path').textContent = `Source · ${sp.path || ''}`;
  $('#src-note').textContent = sp.commit ? `${sp.repo || ''} @ ${String(sp.commit).slice(0, 10)}` : '';
  if (!src.ok) { el.innerHTML = `<div class="src-missing">Source not shown: ${esc(src.error || 'unavailable')}.<br>Pass <code>--source-root</code> so the visibility rule (§2a) can be checked against the file.</div>`; return; }
  const segs = new Map();
  for (const nd of nodes()) {
    const [l1, c1, l2, c2] = nd.loc;
    for (let L = l1; L <= l2; L++) {
      const line = src.lines[L - 1];
      if (line === undefined) continue;
      const s = L === l1 ? c1 - 1 : 0;
      const e = L === l2 ? c2 - 1 : line.length;
      if (!segs.has(L)) segs.set(L, []);
      segs.get(L).push({ s, e, k: keyOf(nd.loc), i: nd.i, first: L === l1, ctx: isCtx(nd) });
    }
  }
  const out = [];
  src.lines.forEach((line, idx) => {
    const L = idx + 1;
    const list = (segs.get(L) || []).sort((a, b) => a.s - b.s);
    let html = '', pos = 0;
    for (const sg of list) {
      if (sg.s < pos) continue; // overlapping spans: keep the first
      html += esc(line.slice(pos, sg.s));
      html += `<mark class="sn ${sg.ctx ? 'ctx' : ''}" data-key="${esc(sg.k)}" data-n="${sg.first ? sg.i : ''}">${esc(line.slice(sg.s, sg.e)) || ' '}</mark>`;
      pos = sg.e;
    }
    html += esc(line.slice(pos));
    out.push(`<div class="ln" data-line="${L}"><span class="no">${L}</span><span class="code">${html || ' '}</span></div>`);
  });
  el.innerHTML = out.join('');
  updateSourceMarks();
}

function updateSourceMarks() {
  $$('#source mark.sn').forEach((m) => {
    m.classList.toggle('cur', m.dataset.key === S.cur);
    m.classList.toggle('suggest', S.mode === 'pointer' && S.staged && S.staged.suggest === m.dataset.key);
  });
  $$('#source .ln.has-cur').forEach((l) => l.classList.remove('has-cur'));
  const nd = curNode();
  if (nd) for (let L = nd.loc[0]; L <= nd.loc[2]; L++) { const l = $(`#source .ln[data-line="${L}"]`); if (l) l.classList.add('has-cur'); }
}

function scrollWithin(pane, el, center) {
  if (!pane || !el) return;
  const pr = pane.getBoundingClientRect();
  const er = el.getBoundingClientRect();
  if (center) pane.scrollTop += er.top - pr.top - pane.clientHeight / 2 + er.height / 2;
  else if (er.top < pr.top + 8) pane.scrollTop += er.top - pr.top - 8;
  else if (er.bottom > pr.bottom - 8) pane.scrollTop += er.bottom - pr.bottom + 8;
}

function scrollToCursor() {
  if (!S.cur) return;
  const sel = `[data-key="${CSS.escape(S.cur)}"]`;
  scrollWithin($('#nodes-pane'), $(`#nodes .row${sel}`), false);
  scrollWithin($('#preview'), $(`#preview .pv${sel}`), false);
  scrollWithin($('#source'), $(`#source mark${sel}`), true);
}

function closeDrawer() { const d = $('#drawer'); d.hidden = true; d.innerHTML = ''; }

function renderDrawer() {
  const d = $('#drawer');
  const st = S.staged;
  if (S.mode === 'pattern') {
    const info = (S.ref && S.ref.patterns) || {};
    const kr = (S.ref && S.ref.krules) || {};
    const nd = byKey(st.key);
    const typed = S.input ? (st.cls === 'KEEP' ? 'K' : 'R') + S.input : null;
    d.innerHTML = `<h3>${st.reopen ? 'Change code' : st.cls} · node #${nd.i}</h3>
      <div class="sub">${st.target ? `target #${byKey(keyOf(st.target)).i} · ` : ''}Enter accepts the highlighted code · digits pick by number · <kbd>-</kbd> none · Esc cancels</div>
      ${st.codes.map((c, n) => {
        const desc = st.cls === 'KEEP' ? (kr[c] || '') : ((info[c] || {}).desc || '');
        const hi = typed ? typed === c : n === st.hi;
        return `<div class="opt ${hi ? 'hi' : ''}" data-code="${c}"><span class="code">${c}</span><span class="desc">${md1(desc)}</span></div>`;
      }).join('')}`;
    $$('#drawer .opt').forEach((o) => { o.onclick = (e) => { e.stopPropagation(); commitPattern(o.dataset.code); }; });
    d.hidden = false;
    const hiEl = $('#drawer .opt.hi');
    if (hiEl) hiEl.scrollIntoView({ block: 'nearest' });
  } else if (S.mode === 'role') {
    const nd = byKey(st.key);
    d.innerHTML = `<h3>Role of node #${nd.i}</h3>
      <div class="sub">Extractor role <code>${esc(nd.role)}</code>. Pick the right role (role_fix); picking <code>${esc(nd.role)}</code> sets role_ok back to true. Scoring always uses the extractor's role.</div>
      <div class="field"><input type="text" id="role-filter" placeholder="filter roles" autocomplete="off" value="${esc(st.filter)}"></div>
      <div id="role-list"></div>`;
    const inp = $('#role-filter');
    inp.oninput = () => { st.filter = inp.value; st.hi = 0; renderRoleList(); };
    renderRoleList();
    d.hidden = false;
  } else if (S.mode === 'aside') {
    const info = Object.assign({}, (S.ref && S.ref.asides) || {});
    d.innerHTML = `<h3>Asides · node #${byKey(st.key).i}</h3>
      <div class="sub">For the rule layer, never scored against the network. Number keys toggle · Enter saves · Esc cancels.</div>
      ${S.schema.node_asides.map((a, n) => `<div class="opt ${st.set.has(a) ? 'on hi' : ''}" data-a="${a}"><span class="code"><kbd>${n + 1}</kbd> ${a}</span><span class="desc">${md1((info[a] || {}).desc || '')}</span></div>`).join('')}`;
    $$('#drawer .opt').forEach((o) => { o.onclick = (e) => { e.stopPropagation(); const a = o.dataset.a; if (st.set.has(a)) st.set.delete(a); else st.set.add(a); renderDrawer(); }; });
    d.hidden = false;
  } else if (S.mode === 'missed') {
    const nd = curNode();
    const items = S.data.missed || [];
    d.innerHTML = `<h3>Missed nodes</h3>
      <div class="sub">Visible copy the extractor did not emit (extractor recall; missed REDUNDANT counts against end-to-end recall). Enter adds · Esc closes.</div>
      <div class="field"><label>Line</label><input type="number" id="missed-line" min="1" value="${nd ? nd.loc[0] : 1}"></div>
      <div class="field"><label>Text</label><input type="text" id="missed-text" autocomplete="off"></div>
      <div class="field"><label>Class</label><select id="missed-class">${S.schema.classes.map((c) => `<option>${c}</option>`).join('')}</select></div>
      <div class="items">${items.map((it, n) => `<div class="item"><span class="chip ${it.class}">${it.class}</span><span>L${it.line}</span><span>${esc(it.text)}</span><button data-rm="${n}">remove</button></div>`).join('') || '<span class="muted small">None recorded for this screen.</span>'}</div>`;
    $$('#drawer [data-rm]').forEach((b) => { b.onclick = (e) => { e.stopPropagation(); removeMissed(+b.dataset.rm); }; });
    d.hidden = false;
  } else if (S.mode === 'saside') {
    const items = S.data.aside_screen || [];
    const info = Object.assign({}, (S.ref && S.ref.screen_asides) || {}, (S.ref && S.ref.asides) || {});
    d.innerHTML = `<h3>Screen asides</h3>
      <div class="sub">A device repeated across sibling sections, cards or rows (R19), or another rule-layer code spanning several nodes. Type node numbers or click nodes · Enter adds (replaces that code) · Esc closes.</div>
      <div class="field"><label>Code</label><select id="saside-code">${S.schema.screen_asides.map((c) => `<option value="${c}">${c}${info[c] ? ' · ' + esc(info[c].desc.replace(/[`*]/g, '').slice(0, 70)) : ''}</option>`).join('')}</select></div>
      <div class="field"><label>Nodes</label><input type="text" id="saside-nodes" placeholder="e.g. 8 12 16" autocomplete="off"></div>
      <div class="items">${items.map((it, n) => `<div class="item"><strong>${esc(it.code)}</strong><span>${it.nodes.map((l) => '#' + ((byKey(keyOf(l)) || {}).i ?? '?')).join(' ')}</span><button data-rm="${n}">remove</button></div>`).join('') || '<span class="muted small">None recorded for this screen.</span>'}</div>`;
    $$('#drawer [data-rm]').forEach((b) => { b.onclick = (e) => { e.stopPropagation(); removeScreenAside(+b.dataset.rm); }; });
    d.hidden = false;
  } else if (S.mode === 'equiv') {
    const r = rec(st.key);
    const t = r.target ? byKey(keyOf(r.target)) : null;
    d.innerHTML = `<h3>ECHO stack · node #${byKey(st.key).i}</h3>
      <div class="sub">A stack of three or more copies keeps the highest-ranked one${t ? ` (#${t.i})` : ''} and flags each other one; each flagged record lists the others in <code>equiv</code> (§5).
      Click nodes or type a number and Space to toggle · Enter saves every member · Esc cancels. Unlabelled members become ECHO with this node's target and pattern.</div>
      <div class="items">${[...st.sel].map((k) => `<div class="item">#${byKey(k).i} ${textHtml(byKey(k).text)}</div>`).join('') || '<span class="muted small">No other copies selected.</span>'}</div>`;
    d.hidden = false;
  } else {
    closeDrawer();
  }
}

function md1(s) { return esc(s).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>'); }

function renderRoleList() {
  const st = S.staged;
  const nd = byKey(st.key);
  const opts = roleOptions();
  st.hi = Math.min(st.hi, Math.max(0, opts.length - 1));
  $('#role-list').innerHTML = opts.map((r, n) => `<div class="opt ${n === st.hi ? 'hi' : ''}" data-role="${r}"><span class="code">${r}</span><span class="desc">${r === nd.role ? 'extractor role' : ''}</span></div>`).join('');
  $$('#role-list .opt').forEach((o) => { o.onclick = (e) => { e.stopPropagation(); pickRole(o.dataset.role); }; });
  const hi = $('#role-list .opt.hi');
  if (hi) hi.scrollIntoView({ block: 'nearest' });
}

function renderFooter() {
  const nd = curNode();
  const st = S.staged;
  let p = '';
  const inp = `<span class="input">${esc(S.input)}</span>`;
  switch (S.mode) {
    case 'goto': p = `<span class="mode">GO TO</span> node ${inp} <span class="muted">Enter · Esc</span>`; break;
    case 'pointer': p = `<span class="mode">${st.cls}</span> #${byKey(st.key).i} → ${inp} <span class="muted">${st.suggest ? `Enter = #${byKey(st.suggest).i}` : 'type a number'} · click a node · - none · Esc cancel</span>`; break;
    case 'pattern': p = `<span class="mode">${st.cls}</span> ${st.cls === 'KEEP' ? 'K rule' : 'pattern'} ${st.cls === 'KEEP' ? 'K' : 'R'}${inp} <span class="muted">Enter = ${st.codes[st.hi]} · ↑↓ · - none · Esc cancel</span>`; break;
    case 'equiv': p = `<span class="mode">EQUIV</span> toggle node ${inp} <span class="muted">Space toggles · Enter saves · Esc</span>`; break;
    case 'role': p = '<span class="mode">ROLE</span> <span class="muted">type to filter · ↑↓ · Enter · Esc</span>'; break;
    case 'aside': p = '<span class="mode">ASIDE</span> <span class="muted">1–9 toggle · Enter saves · Esc</span>'; break;
    case 'missed': p = '<span class="mode">MISSED</span> <span class="muted">Enter adds · Esc closes</span>'; break;
    case 'saside': p = '<span class="mode">SCREEN ASIDE</span> <span class="muted">Enter adds · Esc closes</span>'; break;
    default:
      p = nd ? `<span class="muted">#${nd.i} ${esc(nd.role)} · <kbd>k</kbd> <kbd>e</kbd> <kbd>n</kbd> <kbd>d</kbd> <kbd>u</kbd> · <kbd>?</kbd> keys · <kbd>g</kbd> guide</span>`
        : '<span class="muted">No labelable nodes on this screen · <kbd>N</kbd> next screen</span>';
  }
  $('#prompt').innerHTML = p;
}

function setStatus(s) { $('#status').textContent = s; }

function toast(msg, kind = '') {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $('#toasts').appendChild(el);
  setTimeout(() => el.remove(), kind === 'error' ? 8000 : kind === 'warn' ? 5500 : 3200);
}

function toggleTheme() {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('slop-theme', root.dataset.theme); } catch (e) { /* storage off */ }
}

/* ------------------------------------------------------------------ boot */

async function boot() {
  try { const t = localStorage.getItem('slop-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage off */ }
  $('#btn-help').onclick = () => openOverlay('help');
  $('#btn-ref').onclick = () => openOverlay('ref');
  $('#btn-list').onclick = () => openOverlay('list');
  $('#btn-theme').onclick = toggleTheme;
  try {
    S.session = await api('/api/session');
  } catch (err) { document.body.innerHTML = `<p style="padding:24px">Cannot reach the labelling server: ${esc(err.message)}</p>`; return; }
  S.schema = S.session.schema;
  S.guide = S.session.guide;
  document.title = `Slop labeller · ${S.session.set}`;
  try { S.ref = await api('/api/reference'); } catch (e) { S.ref = null; }
  await loadScreen(S.session.resume);
  handleGuide(S.session.guide);
  setInterval(pollGuide, 4000);
}

boot();
