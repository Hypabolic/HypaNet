// End-to-end test: drive the labeller UI in headless Chromium with real key events over CDP, label every fixture
// screen, and assert on the app state after each step. Node 22+ (built-in WebSocket and fetch), no packages.
//
//   node slop/labeller/tests/drive_browser.mjs --url http://127.0.0.1:8771/ --chrome PATH --out DIR [--only guide]
//
// Start serve.py on the fixtures first (see run_e2e.sh). The default scenario labels all six fixtures, checks
// resume after a reload and screenshots light and dark themes; `--only guide` checks the guide-change alarm
// (serve.py started with --guide pointing at a scratch copy, passed here as --guide-copy).
import { spawn } from 'node:child_process';
import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, all) => (x.startsWith('--') ? [...a, [x.slice(2), all[i + 1]]] : a), []));
const URL_ = args.url || 'http://127.0.0.1:8771/';
const OUT = args.out || '.';
mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let passed = 0;
const fail = (m) => { console.error(`FAIL ${m}`); process.exitCode = 1; throw new Error(m); };

const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(args.chrome, [`--remote-debugging-port=${port}`, `--user-data-dir=${join(OUT, 'profile')}`, '--window-size=1440,900', '--no-first-run', 'about:blank'], { stdio: 'ignore' });
process.on('exit', () => chrome.kill());

let ws, nextId = 1;
const pending = new Map();
async function connect() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      const page = list.find((t) => t.type === 'page');
      if (page) { ws = new WebSocket(page.webSocketDebuggerUrl); break; }
    } catch (e) { /* not up yet */ }
    await sleep(200);
  }
  if (!ws) fail('chrome did not start');
  await new Promise((r) => ws.addEventListener('open', r));
  ws.addEventListener('message', (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { const { res, rej } = pending.get(m.id); pending.delete(m.id); m.error ? rej(new Error(m.error.message)) : res(m.result); }
  });
}
const send = (method, params = {}) => new Promise((res, rej) => { const id = nextId++; pending.set(id, { res, rej }); ws.send(JSON.stringify({ id, method, params })); });
async function ev(expr) {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
  if (r.exceptionDetails) fail(`eval ${expr}: ${r.exceptionDetails.exception?.description || r.exceptionDetails.text}`);
  return r.result.value;
}
async function idle(extra = 30) {
  for (let i = 0; i < 200; i++) { if ((await ev('window.__inflight || 0')) === 0) break; await sleep(15); }
  await sleep(extra);
}
async function waitFor(expr, desc, ms = 5000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) { if (await ev(expr)) return; await sleep(50); }
  fail(`timeout: ${desc || expr}`);
}
const SPECIAL = { Enter: 13, Escape: 27, ArrowDown: 40, ArrowUp: 38, Backspace: 8, ' ': 32 };
async function key(k) {
  const special = k in SPECIAL && k !== ' ';
  const base = { key: k, windowsVirtualKeyCode: SPECIAL[k] || k.toUpperCase().charCodeAt(0) };
  await send('Input.dispatchKeyEvent', { type: special ? 'rawKeyDown' : 'keyDown', ...base, ...(special ? (k === 'Enter' ? { text: '\r' } : {}) : { text: k }) });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', ...base });
  await idle();
}
async function keys(seq) { for (const k of seq) await key(k); }
async function type(text) { await send('Input.insertText', { text }); await idle(); }
async function expect(expr, desc) {
  const v = await ev(expr);
  if (v !== true) fail(`${desc}: ${expr} -> ${JSON.stringify(v)}`);
  passed++;
  console.log(`ok   ${desc}`);
}
async function shot(name) {
  const r = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(join(OUT, `${name}.png`), Buffer.from(r.data, 'base64'));
}
async function load() {
  await send('Page.navigate', { url: URL_ });
  await waitFor('typeof S !== "undefined" && S.data && S.data.screen', 'app loaded');
  await idle(100);
}

const HELPERS = `
  window.__inflight = 0;
  const __f = window.fetch;
  window.fetch = (...a) => { window.__inflight++; return __f(...a).finally(() => { window.__inflight--; }); };
  window.lab = (i) => S.data.labels[S.data.screen.nodes[i].loc.join(':')];
  window.loc = (i) => S.data.screen.nodes[i].loc.join(':');
  window.curI = () => (S.cur ? S.data.screen.nodes.find((n) => n.loc.join(':') === S.cur).i : null);
  window.tgt = (i) => { const r = lab(i); return r && r.target ? S.data.screen.nodes.find((n) => n.loc.join(':') === r.target.join(':')).i : null; };
  window.eqv = (i) => (lab(i).equiv || []).map((l) => S.data.screen.nodes.find((n) => n.loc.join(':') === l.join(':')).i).sort((a, b) => a - b).join(',');
`;

async function atScreen(n) { await waitFor(`S.n === ${n} && S.data.n === ${n}`, `screen ${n + 1} loaded`, 4000); await idle(80); }

async function scenarioMain() {
  await load();
  await expect('S.n === 0 && S.session.blind === true', 'starts at screen 1, blind');
  await expect('S.data.kind.startsWith("first-person game")', 'product kind shown');
  await expect('S.data.source.ok && document.querySelectorAll("#source mark.sn").length === 7', 'source snippet with 7 node spans');
  await expect('document.querySelectorAll("#preview .pv").length === 7 && curI() === 0', 'preview renders every node; cursor on node 0');
  await shot('01_pause_start_light');

  // ---- screen 1: the pause menu
  await key('e');
  await expect('S.mode === "pointer" && S.staged.suggest === loc(1)', 'ECHO pointer suggests PAUSED');
  await key('Enter');
  await expect('S.mode === "pattern" && S.staged.codes[S.staged.hi] === "R1"', 'pattern picker filtered to ECHO, R1 suggested');
  await shot('02_pause_pattern_picker');
  await key('Enter');
  await expect('lab(0).class === "ECHO" && tgt(0) === 1 && lab(0).pattern === "R1" && lab(0).conf === "clear"', 'eyebrow ECHO -> PAUSED R1');
  await expect('curI() === 1', 'cursor advanced to PAUSED');
  await key('k');
  await keys(['ArrowUp', 'w', '1', '1', 'Enter']);
  await expect('lab(1).class === "KEEP" && lab(1).pattern === "K11"', 'PAUSED KEEP with K11 via w');
  await key('j');
  await keys(['t', 'c']);
  await expect('!lab(2) && S.drafts[loc(2)].ct === true && S.drafts[loc(2)].conf === "judgement"', 'ct and judgement staged on an unlabelled node');
  await key('n');
  await expect('S.staged.suggest === loc(3)', 'NARRATE pointer suggests RESUME');
  await keys(['Enter']);
  await expect('S.staged.codes.join() === "R6,R16" && S.staged.codes[S.staged.hi] === "R6"', 'NARRATE picker shows R6, R16');
  await key('Enter');
  await expect('lab(2).class === "NARRATE" && tgt(2) === 3 && lab(2).pattern === "R6" && lab(2).ct === true && lab(2).conf === "judgement"', 'subtitle NARRATE -> RESUME R6, ct, judgement');
  await keys(['k', 'k']);
  await key('r');
  await expect('S.mode === "role" && document.activeElement.id === "role-filter"', 'role picker open with filter focused');
  await type('menu');
  await key('Enter');
  await expect('!lab(5) && S.drafts[loc(5)].role_ok === false && S.drafts[loc(5)].role_fix === "menu_item"', 'role fix staged');
  await key('m');
  await type('Press Esc to resume');
  await ev('document.querySelector("#missed-line").value = "22"; document.querySelector("#missed-class").value = "NARRATE"; true');
  await key('Enter');
  await expect('S.data.missed.length === 1 && S.data.missed[0].text === "Press Esc to resume" && S.data.missed[0].class === "NARRATE"', 'missed node recorded');
  await key('Escape');
  await key('p');
  await expect('S.drafts[loc(5)].partial === true', 'partial staged');
  await key('p');
  await key('k');
  await expect('lab(5).class === "KEEP" && lab(5).role_ok === false && lab(5).role_fix === "menu_item" && lab(5).partial === false', 'CONTROLS KEEP with role_fix menu_item');
  await expect('S.data.done === false && S.data.summary.labelled === 6', 'not done with one node left');
  await key('k');
  await expect('S.session.screens[0].done === true', 'done record written for screen 1');
  await atScreen(1);
  await expect('true', 'auto-advanced to screen 2 after done');

  // ---- screen 2: profile form; resume check halfway
  await expect('S.data.summary.nodes === 11 && document.querySelectorAll("#nodes .row.ctx").length === 1', 'sr_only node shown as context, not counted');
  await keys(['k', 'k']);
  await keys(['e']);
  await expect('S.staged.suggest === loc(1)', 'subtitle suggests the h2');
  await keys(['Enter']);
  await expect('S.staged.codes[S.staged.hi] === "R3"', 'R3 suggested for a subtitle');
  await keys(['Enter', 'k', 'e']);
  await expect('S.staged.suggest === loc(3)', 'placeholder suggests its label');
  await keys(['Enter']);
  await expect('S.staged.codes[S.staged.hi] === "R8"', 'R8 suggested for a placeholder');
  await keys(['Enter']);
  await load();
  await expect('S.n === 1 && curI() === 5', 'reload resumes at screen 2, first unlabelled node');
  await keys(['k', 'k', 'e', 'Enter']);
  await expect('S.staged.codes[S.staged.hi] === "R5"', 'R5 suggested for a helper');
  await keys(['Enter']);
  await expect('tgt(7) === 5', 'helper ECHO -> Postcode label');
  await keys(['k', 'k']);
  await key('j');
  await expect('curI() === 10', 'j skips nothing labelable past the end');
  await key('k');
  await atScreen(2);

  // ---- screen 3: landing page
  await keys(['u', 'k', 'k', 'k', 'k', 'ArrowUp', 'w', '6', 'Enter', 'j']);
  await expect('lab(0).class === "UNSURE" && lab(4).pattern === "K6" && curI() === 5', 'UNSURE, KEEP K6');
  await keys(['d']);
  await expect('S.staged.codes.join() === "R2,R11" && S.staged.codes[S.staged.hi] === "R2"', 'DECOR picker, R2 for a genre eyebrow');
  await keys(['Enter', 'k', 'n', '-']);
  await expect('S.staged.codes[S.staged.hi] === "R16"', 'R16 suggested for layout narration');
  await keys(['Enter']);
  await expect('lab(7).class === "NARRATE" && lab(7).target === null && lab(7).pattern === "R16"', 'NARRATE with no target');
  for (let c = 0; c < 3; c++) {
    await keys(['d']);
    await expect('S.staged.codes[S.staged.hi] === "R11"', `R11 suggested for dot pill ${c + 1}`);
    await keys(['Enter', 'a', '5', 'Enter', 'k', 'k', 'k']);
  }
  await expect('lab(9).aside.join() === "R12" && lab(13).aside.join() === "R12" && lab(17).aside.join() === "R12"', 'index numbering KEEP with aside R12 (staged then saved)');
  await keys(['k', 'e']);
  await expect('S.staged.suggest === loc(3)', 'repeated CTA suggests the first one');
  await keys(['Enter']);
  await expect('S.staged.codes[S.staged.hi] === "R14"', 'R14 suggested across sections');
  await keys(['Enter']);
  await key('s');
  await type('8 12 16');
  await key('Enter');
  await expect('S.data.aside_screen.length === 1 && S.data.aside_screen[0].code === "R19" && S.data.aside_screen[0].nodes.length === 3', 'screen aside R19 over the three pills');
  await key('Escape');
  await keys(['j', 'e']);
  await expect('S.staged.suggest === loc(3)', 'third CTA suggests the kept first copy, through the labelled second');
  await keys(['Enter', 'Enter', 'v']);
  await expect('S.mode === "equiv" && S.staged.sel.has(loc(21)) && S.staged.sel.size === 1', 'equiv prefilled with the other flagged copy');
  await shot('03_landing_equiv');
  await key('Enter');
  await expect('eqv(21) === "23" && eqv(23) === "21" && tgt(23) === 3', 'equiv written on both flagged copies');
  await key('ArrowUp');
  await shot('04_landing_light');
  await key('\\');
  await shot('05_landing_dark');
  await key('?');
  await shot('06_help_dark');
  await key('Escape');
  await key('g');
  await key('2');
  await shot('07_reference_krules_dark');
  await key('Escape');
  await key('\\');
  await keys(['e', '-']);
  await expect('S.staged.codes[S.staged.hi] === "R17"', 'R17 suggested for a closing summary');
  await key('Enter');
  await atScreen(3);

  // ---- screen 4: invoice list (SwiftUI)
  await keys(['k', 'k', 'k', 'e']);
  await expect('S.staged.suggest === loc(2)', 'empty body suggests the empty title');
  await keys(['Enter']);
  await expect('S.staged.codes[S.staged.hi] === "R4"', 'R4 suggested for an empty-state body');
  await keys(['Enter', 'k', 'k', 'k', 'k', 'k']);
  await atScreen(4);

  // ---- screen 5: status dashboard
  await expect('[...document.querySelectorAll("#nodes-pane .text .var")].map((x) => x.textContent).join() === "VAR,VAR,VAR,VAR" && !document.body.innerText.includes("\\uE000")', 'VAR markers drawn as chips');
  await keys(['k', 'k', 'k', 'k', 'k', 'd', 'Enter', 'k']);
  await expect('lab(5).class === "DECOR" && lab(5).pattern === "R11" && curI() === 8', 'static Live pill DECOR R11; cursor skips the aria node');
  await keys(['k', 'k', 'k', 'k', 'k', 'k']);
  await keys(['x', 'j', 'x', 'j', 'x', 'j']);
  await expect('lab(14).class === null && lab(14).aside.join() === "UNRESOLVED" && lab(16).aside.join() === "UNRESOLVED"', 'VAR-only cells recorded UNRESOLVED');
  await keys(['k', 'n', '-', 'Enter']);
  await atScreen(5);

  // ---- screen 6: onboarding
  await keys(['k', 'k', 'c', 'n', '-', 'Enter', 'n', '-', 'Enter', 'k', 'k', 'k', 'k', 'ArrowUp', 'p', 'j', 't', 'k']);
  await expect('lab(2).conf === "judgement" && lab(7).partial === true && lab(8).ct === true', 'judgement, partial, ct');
  await keys(['k', 'k', 'a', '7', 'Enter', 'n']);
  await expect('S.staged.suggest === loc(9)', 'narration suggests the Continue button');
  await keys(['Enter', 'Enter']);
  await expect('S.session.screens.every((s) => s.done)', 'every screen done');
  await expect('lab(11).aside.join() === "R15" && lab(11).class === "NARRATE"', 'filler aside R15 kept with NARRATE');
}

async function scenarioGuide() {
  await load();
  await ev('window.__h0 = S.guide.current; true');
  await expect('S.overlay === null', 'no alarm at start');
  appendFileSync(args['guide-copy'], '\n<!-- edited during the session -->\n');
  await waitFor('S.overlay === "guide"', 'guide alarm', 7000);
  await expect('!document.querySelector("#ov-guide").hidden', 'guide-change alarm shown');
  await shot('08_guide_alarm');
  await key('k');
  await expect('S.overlay === "guide" && !lab(curI())', 'keys blocked while the alarm is up');
  await key('Enter');
  await expect('S.overlay === null && document.querySelector(".banner.alarm") !== null', 'acknowledged; banner stays');
  await key('k');
  await expect('lab(0).guide_sha256 === S.guide.current && S.guide.current !== window.__h0', 'new label carries the new guide hash');
  await shot('09_guide_banner');
}

await connect();
await send('Page.enable');
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
await send('Page.addScriptToEvaluateOnNewDocument', { source: HELPERS });
try {
  if (args.only === 'guide') await scenarioGuide(); else await scenarioMain();
  console.log(`\n${passed} checks passed`);
} catch (e) {
  console.error(e.message);
  await shot('failure').catch(() => {});
  process.exitCode = 1;
} finally {
  chrome.kill();
  process.exit(process.exitCode || 0);
}
