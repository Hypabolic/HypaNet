// Golden-fixture test: every fixture directory under extract/fixtures/ holds one source file and expected.json.
// The extractor's output must equal expected.json byte for byte, twice in a row. Also checks the schema, coverage
// of every role (bar track C) and every flag, and the pinned Swift grammar.
//
//   npm test                      run all fixtures
//   npm test -- --update NAME...  rewrite expected.json for the named fixtures (review the diff before committing)
//   npm test -- --only NAME...    run only the named fixtures

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { extractFile, frameworkOf, repoContext, serialise } from '../src/extract.ts';
import type { Screen } from '../src/extract.ts';
import { CLUSTER_KINDS, FLAGS, ROLES, TRACK_C_ROLES } from '../src/schema.ts';
import { swiftWasmPath } from '../src/swift.ts';

const here = dirname(fileURLToPath(import.meta.url));
const fixturesDir = join(here, '..', 'fixtures');
const SWIFT_WASM_SHA256 = 'd084eceba50e9a83319f5d171051b53179bd22b6ea6419014900c602d52cfa28';
const MIN_FIXTURES = 60;

const argv = process.argv.slice(2);
const update = argv.includes('--update');
const only = argv.includes('--only');
const names = argv.filter((a) => !a.startsWith('--'));

let failures = 0;
const fail = (msg: string): void => {
  failures++;
  process.stdout.write(`FAIL ${msg}\n`);
};

function validate(name: string, s: Screen): void {
  const roles = new Set<string>(ROLES);
  let prevK = -1;
  s.nodes.forEach((n, idx) => {
    if (n.i !== idx) fail(`${name}: node ${idx} has i=${n.i}`);
    if (!roles.has(n.role) || TRACK_C_ROLES.has(n.role)) fail(`${name}: node ${idx} has role ${n.role}`);
    const order = n.flags.map((f) => FLAGS.indexOf(f));
    if (order.some((o) => o < 0)) fail(`${name}: node ${idx} has an unknown flag`);
    if (order.some((o, j) => j > 0 && o <= order[j - 1])) fail(`${name}: node ${idx} flags out of order`);
    if (n.k < prevK || n.k > prevK + 1) fail(`${name}: node ${idx} cluster ${n.k} after ${prevK}`);
    prevK = n.k;
    const [l1, c1, l2, c2] = n.loc;
    if (!(l1 >= 1 && c1 >= 1 && (l2 > l1 || (l2 === l1 && c2 >= c1)))) fail(`${name}: node ${idx} bad loc ${n.loc}`);
    if (n.text !== n.text.trim() || /\s{2}/.test(n.text) || n.text === '') fail(`${name}: node ${idx} text not normalised`);
    if (/\uE002/.test(n.text)) fail(`${name}: node ${idx} text holds the internal VAR_END mark`);
    // vw: one count per VAR marker, present only when there is one, never more words than follow the marker
    const vars = n.text.split('\uE000').slice(1);
    if (!vars.length && n.vw !== undefined) fail(`${name}: node ${idx} has vw but no VAR marker`);
    if (vars.length) {
      if (!n.vw || n.vw.length !== vars.length) fail(`${name}: node ${idx} vw ${JSON.stringify(n.vw)} for ${vars.length} VAR marker(s)`);
      else n.vw.forEach((w, j) => {
        const after = vars[j].split(/\s+/).filter(Boolean).length;
        if (!Number.isInteger(w) || w < 0 || w > after) fail(`${name}: node ${idx} vw[${j}] = ${w} with ${after} word(s) after the marker`);
      });
    }
  });
  s.clusters.forEach((c, k) => {
    if (c.k !== k) fail(`${name}: cluster ${k} has k=${c.k}`);
    if (!(CLUSTER_KINDS as readonly string[]).includes(c.kind)) fail(`${name}: cluster ${k} kind ${c.kind}`);
    if (!s.nodes.some((n) => n.k === k)) fail(`${name}: cluster ${k} has no nodes`);
  });
}

// pinned grammar
const wasmSha = createHash('sha256').update(readFileSync(swiftWasmPath())).digest('hex');
if (wasmSha !== SWIFT_WASM_SHA256) fail(`tree-sitter-swift.wasm sha256 ${wasmSha} is not the pinned ${SWIFT_WASM_SHA256}`);

const dirs = readdirSync(fixturesDir, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name).sort();
const seenRoles = new Set<string>();
const seenFlags = new Set<string>();
const seenKinds = new Set<string>();
const seenFw = new Set<string>();
let ran = 0;
for (const name of dirs) {
  if ((update || only) && names.length && !names.includes(name)) continue;
  const dir = join(fixturesDir, name);
  const srcs = readdirSync(dir).filter((f) => frameworkOf(f) && f !== 'expected.json').sort();
  if (srcs.length !== 1) {
    fail(`${name}: expected exactly one source file, found ${srcs.length}`);
    continue;
  }
  const ctx = repoContext(dir);
  const meta = { repo: 'slop-fixtures', commit: null, licence: 'MIT' };
  let out1: string;
  let screen: Screen;
  try {
    screen = await extractFile(join(dir, srcs[0]), ctx, meta);
    out1 = serialise(screen);
  } catch (err) {
    fail(`${name}: extractor threw ${(err as Error).stack}`);
    continue;
  }
  const out2 = serialise(await extractFile(join(dir, srcs[0]), repoContext(dir), meta));
  if (out1 !== out2) fail(`${name}: two runs gave different bytes`);
  validate(name, screen);
  ran++;
  for (const n of screen.nodes) {
    seenRoles.add(n.role);
    for (const f of n.flags) seenFlags.add(f);
  }
  for (const c of screen.clusters) seenKinds.add(c.kind);
  seenFw.add(screen.framework);
  const expPath = join(dir, 'expected.json');
  if (update) {
    writeFileSync(expPath, out1);
    process.stdout.write(`updated ${name}\n`);
    continue;
  }
  let expected: string;
  try {
    expected = readFileSync(expPath, 'utf8');
  } catch {
    fail(`${name}: no expected.json`);
    continue;
  }
  if (expected !== out1) {
    const a = expected.split('\n');
    const b = out1.split('\n');
    const lines: string[] = [];
    for (let i = 0; i < Math.max(a.length, b.length); i++) {
      if (a[i] !== b[i]) lines.push(`  line ${i + 1}\n    expected: ${a[i] ?? '(none)'}\n    actual:   ${b[i] ?? '(none)'}`);
    }
    fail(`${name}: output differs from expected.json\n${lines.slice(0, 8).join('\n')}`);
  } else process.stdout.write(`ok   ${name}\n`);
}

// Swift output must not depend on what was parsed before it in the process (tree-sitter-swift's scanner state):
// each file of swift-order/ in a fresh process, then all of them in one process in two orders, give the same bytes.
if (!update && !only) {
  const orderDir = join(here, 'swift-order');
  const files = readdirSync(orderDir).filter((f) => f.endsWith('.swift')).sort();
  const meta = { repo: 'slop-fixtures', commit: null, licence: 'MIT' };
  const single = new Map<string, string>();
  const script = `const { extractFile, repoContext, serialise } = await import(${JSON.stringify(join(here, '..', 'src', 'extract.ts'))});
    process.stdout.write(serialise(await extractFile(process.argv[1], repoContext(process.argv[2]), ${JSON.stringify(meta)})));`;
  for (const f of files) single.set(f, execFileSync(process.execPath, ['--input-type=module', '-e', script, join(orderDir, f), orderDir], { encoding: 'utf8' }));
  for (const order of [files, [...files].reverse()]) {
    for (const f of order) {
      const out = serialise(await extractFile(join(orderDir, f), repoContext(orderDir), meta));
      if (out !== single.get(f)) fail(`swift-order: ${f} after ${order.slice(0, order.indexOf(f)).join(', ') || 'nothing'} differs from a fresh process`);
    }
  }
  if (!/Tap the lid to start \\ue001 Press space to start/.test(single.get('C.swift') ?? '') || /Reset shelf cache/.test(single.get('B.swift') ?? '')) {
    fail('swift-order: #if branches not read as expected');
  }
  process.stdout.write(`ok   swift-order (${files.length} files, two orders)\n`);
}

if (!update && !only) {
  if (ran < MIN_FIXTURES) fail(`only ${ran} fixtures; at least ${MIN_FIXTURES} required`);
  const missingRoles = ROLES.filter((r) => !TRACK_C_ROLES.has(r) && !seenRoles.has(r));
  if (missingRoles.length) fail(`roles never exercised: ${missingRoles.join(' ')}`);
  const missingFlags = FLAGS.filter((f) => !seenFlags.has(f));
  if (missingFlags.length) fail(`flags never exercised: ${missingFlags.join(' ')}`);
  const missingKinds = CLUSTER_KINDS.filter((k) => !seenKinds.has(k));
  if (missingKinds.length) fail(`cluster kinds never exercised: ${missingKinds.join(' ')}`);
  for (const fw of ['tsx', 'html', 'swift']) if (!seenFw.has(fw)) fail(`no ${fw} fixture`);
}
process.stdout.write(`\n${ran} fixtures, ${failures} failure(s)\n`);
process.exit(failures ? 1 : 0);
