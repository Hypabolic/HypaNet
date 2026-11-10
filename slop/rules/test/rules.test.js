// Rule-layer tests: the invented fixtures in fixtures/ (exact findings per screen, positives and near-miss negatives
// for every rule), the text helpers, the CLI, and a run over every other screen in the repo's tests.
//
//   node --test slop/rules/test/

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { encodeScreen } from '../../lint/tokenizer.js';
import { RULE_IDS, RULES, runRules } from '../index.js';
import { buildAll } from '../fixtures/make_fixtures.mjs';
import { isQualifier } from '../src/repeated-chrome.js';
import { hasArrowTail, literalWords, nodeBranches, normalised, parseIndex, tokens } from '../src/text.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const RULES_DIR = path.join(HERE, '..');
const SLOP = path.join(RULES_DIR, '..');
const FIX = path.join(RULES_DIR, 'fixtures');
const CLI = path.join(RULES_DIR, 'cli.js');
const EXPECTED = JSON.parse(readFileSync(path.join(FIX, 'expected.json'), 'utf8'));
const loadFixture = (name) => JSON.parse(readFileSync(path.join(FIX, 'screens', `${name}.json`), 'utf8'));

const compact = (fs) => fs.map((x) => {
  const out = { rule: x.rule, code: x.code };
  if (x.condition) out.condition = x.condition;
  out.nodes = x.nodes;
  if (x.target) out.target = x.target.node;
  return out;
});
const key = (x) => JSON.stringify([RULE_IDS.indexOf(x.rule), x.nodes[0], x.condition ?? '', x.code]);
const sorted = (xs) => [...xs].sort((a, b) => (key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0));

const cli = (args, opts = {}) => spawnSync('node', [CLI, ...args], { encoding: 'utf8', cwd: SLOP, ...opts });

// -- fixtures ------------------------------------------------------------------------------------------------------

test('fixture files are up to date with make_fixtures.mjs', () => {
  for (const [rel, text] of buildAll()) assert.equal(readFileSync(path.join(FIX, rel), 'utf8'), text, rel);
  const onDisk = readdirSync(path.join(FIX, 'screens')).sort();
  assert.deepEqual(onDisk, Object.keys(EXPECTED).map((n) => `${n}.json`).sort());
});

for (const [name, entry] of Object.entries(EXPECTED)) {
  test(`fixture ${name}`, () => {
    const screen = loadFixture(name);
    encodeScreen(screen); // a valid screen for the tokeniser: roles, flags, cluster indices
    const got = runRules(screen);
    assert.deepEqual(sorted(compact(got)), sorted(entry.findings));
    for (const x of got) {
      assert.deepEqual(x.locs, x.nodes.map((i) => screen.nodes[i].loc), 'locs are the flagged nodes\' locs');
      assert.equal(x.scope, x.code === 'R19' ? 'screen' : 'node');
      assert.ok(x.message.length > 10);
    }
  });
}

test('every rule has positive and near-miss negative fixtures, and every repeated-chrome condition fires', () => {
  const names = Object.keys(EXPECTED);
  assert.ok(names.length >= 40, `${names.length} fixtures`);
  for (const rule of RULE_IDS) {
    const about = Object.values(EXPECTED).flatMap((e) => e.about).filter((a) => a.rule === rule);
    assert.ok(about.some((a) => a.kind === 'positive'), `${rule} has no positive fixture`);
    assert.ok(about.some((a) => a.kind === 'negative'), `${rule} has no negative fixture`);
  }
  for (const [name, e] of Object.entries(EXPECTED)) {
    for (const a of e.about) {
      const fires = e.findings.some((x) => x.rule === a.rule);
      if (a.kind === 'positive') assert.ok(fires, `${name} is a positive for ${a.rule} but expects no finding from it`);
    }
    if (e.about.every((a) => a.kind === 'negative')) assert.deepEqual(e.findings, [], `${name} is negative only`);
  }
  const conditions = new Set(Object.values(EXPECTED).flatMap((e) => e.findings.map((x) => x.condition).filter(Boolean)));
  assert.deepEqual([...conditions].sort(), ['arrow-links', 'decorative-dot', 'eyebrow-quota', 'sibling-device', 'state-dots']);
  const codes = new Set(Object.values(EXPECTED).flatMap((e) => e.findings.map((x) => x.code)));
  for (const c of ['R7', 'R8b', 'R9', 'R10', 'R12', 'R14', 'R15', 'R19']) assert.ok(codes.has(c), c);
});

test('the exact-repeat fixtures that need two windows have them', () => {
  for (const name of ['er_cta_pos', 'er_rank_pos', 'er_exempt_neg', 'er_footer_links_neg', 'er_k11_cross_window_neg', 'er_two_dialogs_neg', 'er_conditional_keep_pos',
    'er_k12_error_summary_neg', 'er_k12_icon_tooltip_neg', 'er_site_bar_link_neg', 'er_footer_vs_body_cta_pos']) assert.ok(encodeScreen(loadFixture(name)).windows.length >= 2, name);
  assert.equal(encodeScreen(loadFixture('er_same_window_neg')).windows.length, 1);
});

test('the fixtures use both markers', () => {
  const all = Object.keys(EXPECTED).map(loadFixture).flatMap((s) => s.nodes.map((x) => x.text)).join('');
  assert.ok(all.includes('') && all.includes(''));
});

test('runRules is deterministic and filters by rule id', () => {
  const s = loadFixture('ix_padded_cards_pos');
  assert.deepEqual(runRules(s), runRules(structuredClone(s)));
  assert.deepEqual(runRules(s, { only: ['slop/index-numbering'] }).map((x) => x.rule), ['slop/index-numbering']);
  assert.throws(() => runRules(s, { only: ['slop/nope'] }), /unknown rule/);
  assert.throws(() => runRules({ schema: 'other' }), /slop.screen\/1/);
  assert.deepEqual(RULES.map((r) => r.code), ['R19', 'R7', 'R15', 'R8b', 'R10', 'R12', 'R14']);
});

// -- text helpers --------------------------------------------------------------------------------------------------

test('parseIndex reads the index devices and nothing else', () => {
  const cases = {
    '01': ['padded', 1, true], '02 /': ['padded', 2, true], '(003)': ['padded', 3, true], '#04': ['padded', 4, true],
    '01 / 04': ['padded', 1, true], '05.': ['padded', 5, true], '01 / Connect': ['padded', 1, false],
    '02. Configure the runner': ['padded', 2, false], '3 / 6': ['fraction', 3, true], 'Step 2': ['label:step', 2, true],
    'STEP 2 OF 4': ['label:step', 2, true], 'Phase 02': ['label:phase', 2, true], 'Stage 1 —': ['label:stage', 1, true],
    'Question 05': ['label:question', 5, true], 'No. 3': ['label:no', 3, true], 'Step 1: Create an account': ['label:step', 1, false],
    '1': ['plain', 1, true], '2.': ['plain', 2, true], '3)': ['plain', 3, true],
  };
  for (const [text, [family, value, whole]] of Object.entries(cases)) {
    const d = parseIndex(text);
    assert.ok(d, text);
    assert.deepEqual([d.family, d.value, d.whole], [family, value, whole], text);
  }
  for (const text of ['2026', '1.9 kg', 'Level 3', 'Chapter 2', 'Step 1 is the hardest', '100', '£40', '24/7 support', 'v2', '3 seats left', '1,000',
    '08:00', '01.03', '05 / 03', '3/2'])
    assert.equal(parseIndex(text), null, text);
  assert.equal(parseIndex('01 / 04').total, 4);
  assert.equal(parseIndex('Step 2 of 4').total, 4);
  assert.equal(parseIndex('02 /').total, null);
  // Leads keep their total; "05 / 03 Connect" is not a device.
  assert.deepEqual(['01 / 03 Connect the hub', 'Step 2 of 3: Safety talk'].map((t) => { const d = parseIndex(t); return [d.family, d.value, d.whole, d.total]; }),
    [['padded', 1, false, 3], ['label:step', 2, false, 3]]);
  assert.equal(parseIndex('05 / 03 Connect'), null);
  // A two-digit number 10–99, whole or as a lead, is a `wide` padded device: it only continues a padded column.
  for (const t of ['10 Slipway', '12 months of data', '24 –']) assert.ok(parseIndex(t)?.wide, t);
  assert.equal(parseIndex('10').family, 'plain');
  for (const t of ['09 Fog bell', '3 seats left', '24/7 support', '100 days']) assert.ok(!parseIndex(t)?.wide, t);
});

test('words after a VAR marker are identifier words, not copy', () => {
  assert.deepEqual(literalWords('Hello  first name, please confirm.'), ['hello', 'please', 'confirm']);
  assert.deepEqual(literalWords(' your name'), []);
  assert.deepEqual(literalWords('VAR means variance'), ['var', 'means', 'variance']); // the letters are ordinary words
  assert.equal(tokens('a  b').filter((t) => t.k === 'alt').length, 1);
});

test('the extractor\'s vw gives the exact identifier words; without it (or when it does not fit) the cap applies', () => {
  assert.deepEqual(literalWords(' name please check your inbox', [1]), ['please', 'check', 'your', 'inbox']);
  assert.deepEqual(literalWords(' name please check your inbox'), ['inbox']); // fallback: four words
  assert.deepEqual(literalWords(' upload failed try again please', [5]), []);
  assert.deepEqual(literalWords(' line 2 is due', [2]), ['is', 'due']); // identifier words include numbers
  assert.deepEqual(literalWords(' line 2 is due'), ['2', 'is', 'due']); // the fallback stops at a number
  assert.deepEqual(literalWords('Paid  amount now', [1, 2]), ['paid']); // one count per marker, or the fallback
  assert.deepEqual(literalWords(' a  b c d', [0, 1]), ['a', 'c', 'd']);
  assert.deepEqual(nodeBranches({ text: 'Join  team name  Pay  amount now', vw: [2, 1] }),
    [{ text: 'Join  team name', vw: [2] }, { text: 'Pay  amount now', vw: [1] }]);
  assert.deepEqual(nodeBranches({ text: 'Retry  Wait' }), [{ text: 'Retry', vw: undefined }, { text: 'Wait', vw: undefined }]);
});

test('normalised strings fold case, glyphs, punctuation and apostrophes but keep markers', () => {
  assert.equal(normalised('● No card needed!'), 'no card needed');
  assert.equal(normalised('NO CARD NEEDED'), 'no card needed');
  assert.equal(normalised('Don’t wait'), normalised("dont wait"));
  assert.equal(normalised('Café'), 'cafe');
  assert.equal(normalised(' count left'), ' count left');
});

test('arrow tails come from the flag or a trailing glyph', () => {
  assert.ok(hasArrowTail({ text: 'Docs →', flags: [] }));
  assert.ok(hasArrowTail({ text: 'Docs ->', flags: [] }));
  assert.ok(hasArrowTail({ text: 'Docs', flags: ['arrow_tail'] }));
  assert.ok(!hasArrowTail({ text: '→ Docs', flags: [] }));
  assert.ok(hasArrowTail({ text: 'Next ›', flags: [] }));
  assert.ok(hasArrowTail({ text: 'Docs ⟶', flags: [] }));
  // ↗ marks an external link, » and a glued › are not arrows: the text overrides the extractor's flag.
  for (const text of ['Docs ↗', 'More »', 'Next›', 'Below ↘']) assert.ok(!hasArrowTail({ text, flags: ['arrow_tail'] }), text);
  for (const text of ['Guides ➡️', 'Guides ➡', 'Guides ⇒', 'Guides ➝', 'Guides ➞', 'Guides ⟹', 'Guides =>', 'Guides ⭢', 'Guides 🡒'])
    assert.ok(hasArrowTail({ text, flags: [] }), text);
});

test('K13 qualifier badges: tax status, billing period, unit, currency, "from"', () => {
  for (const t of ['Inc. VAT', '+ VAT', 'excl. tax', 'VAT included', '/month', '/ mo', 'per year', 'a night', 'Billed annually', 'Monthly',
    'per seat', '/user/month', 'Each', 'GBP', 'in EUR', 'From']) assert.ok(isQualifier(t), t);
  for (const t of ['New', 'Most popular', 'Monthly plan', 'Free', 'A list', 'VAT', 'Inc.', 'Best value']) assert.ok(!isQualifier(t), t);
});

// -- CLI -----------------------------------------------------------------------------------------------------------

test('CLI --format json over the fixture directory reproduces expected.json', () => {
  const res = cli([path.join(FIX, 'screens'), '--format', 'json']);
  assert.equal(res.status, 1, res.stderr);
  const report = JSON.parse(res.stdout);
  assert.equal(report.schema, 'slop.rules/1');
  assert.equal(report.files.length, Object.keys(EXPECTED).length);
  assert.deepEqual(report.errors, []);
  for (const f of report.files) {
    const name = path.basename(f.file, '.json');
    assert.deepEqual(sorted(compact(f.findings)), sorted(EXPECTED[name].findings), name);
    for (const x of f.findings) assert.deepEqual(Object.keys(x).filter((k) => !['condition', 'target'].includes(k)), ['rule', 'code', 'scope', 'nodes', 'locs', 'message']);
  }
});

test('CLI exit status, text output, rule filter, skipped and broken files', () => {
  const clean = cli([path.join(FIX, 'screens', 'clean_settings_neg.json')]);
  assert.equal(clean.status, 0);
  assert.match(clean.stdout, /^0 findings in 0 of 1 screen\n$/);
  const text = cli([path.join(FIX, 'screens', 'va_stock_labels_pos.json')]);
  assert.equal(text.status, 1);
  assert.match(text.stdout, /6:7 +R7 +button "Get started →" is a stock label.*slop\/vacuous-action/);
  const only = JSON.parse(cli([path.join(FIX, 'screens'), '--format', 'json', '--rules', 'slop/placeholder-as-label']).stdout);
  assert.deepEqual([...new Set(only.files.flatMap((f) => f.findings.map((x) => x.code)))], ['R8b']);
  assert.equal(cli(['x.json', '--rules', 'slop/nope']).status, 2);

  const dir = mkdtempSync(path.join(os.tmpdir(), 'slop-rules-'));
  writeFileSync(path.join(dir, 'a.json'), readFileSync(path.join(FIX, 'screens', 'pl_body_cluster_pos.json')));
  writeFileSync(path.join(dir, 'other.json'), '{"schema": "slop.rules-expected/1"}');
  writeFileSync(path.join(dir, 'broken.json'), '{"schema": ');
  writeFileSync(path.join(dir, 'notes.txt'), 'not json');
  const res = cli([dir, '--format', 'json']);
  assert.equal(res.status, 2);
  const report = JSON.parse(res.stdout);
  assert.equal(report.files.length, 1);
  assert.equal(report.skipped, 1);
  assert.equal(report.errors.length, 1);
  assert.match(report.errors[0].file, /broken\.json$/);
  assert.deepEqual(report.files[0].findings.map((x) => x.code), ['R8b']);
});

// -- every other screen in the repo's tests ------------------------------------------------------------------------

test('the rules run on every extractor fixture and tokeniser test screen without error, deterministically', () => {
  const dirs = [path.join(SLOP, 'extract', 'fixtures'), path.join(SLOP, 'ui', 'tests', 'screens'), path.join(SLOP, 'labeller', 'fixtures')];
  const a = cli([...dirs, '--format', 'json']);
  const b = cli([...dirs, '--format', 'json']);
  assert.notEqual(a.status, 2, a.stdout);
  assert.equal(a.stdout, b.stdout);
  assert.ok(JSON.parse(a.stdout).files.length >= 100);
});
