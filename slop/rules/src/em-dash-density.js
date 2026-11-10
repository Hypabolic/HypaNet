// slop/em-dash-density (codex L48): em dashes as the default clause connector across a screen. One or two are writing;
// three or more nodes joining clauses with one is the habit of generated copy. Screen-level, severity info. Glyphs:
// codex/lists/llm_glyphs.txt §L48, the em dash and "--" only: a spaced en dash is the ordinary British parenthetical
// dash, and a spaced hyphen is too common in hand-typed copy. A dash counts only between two words (not a lone "—" in
// an empty table cell, not "— Sam" before a byline, not a range of numbers such as "2020 — 2021"), and never in a
// cell, field value or stat.

import { finding } from './finding.js';
import { sectionEntries } from './codex.js';
import { isScored } from './screen.js';

export const id = 'slop/em-dash-density';
export const code = 'L48';
export const severity = 'info';
export const description = 'Em dashes used as the clause connector in three or more strings on one screen.';

export const MIN_NODES = 3;
const DASHES = sectionEntries('llm_glyphs', ['L48']).map((e) => e.entry).filter((e) => e.trim() === e && e.length);
const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const CONNECTOR = new RegExp(`[\\p{L},.)’'"]\\s*(?:${DASHES.map(esc).join('|')})\\s*[\\p{L}(‘'"]`, 'u');
const SKIP_ROLES = new Set(['cell', 'input_value', 'stat']);

export function check(screen) {
  const xs = [];
  screen.nodes.forEach((n, x) => {
    if (isScored(n) && !SKIP_ROLES.has(n.role) && CONNECTOR.test(n.text)) xs.push(x);
  });
  if (xs.length < MIN_NODES) return [];
  return [finding(id, code, screen, xs, `${xs.length} strings on this screen join clauses with an em dash; vary the punctuation or split the sentence`, { scope: 'screen' })];
}
