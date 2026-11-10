// The plain-code rule layer (SPEC.md §8): seven rules over one slop.screen/1 document, no network, no dependencies
// beyond Node and lint/tokenizer.js (for the window split). Scored against the `aside` codes, never credited to the
// network.

import * as exactRepeat from './src/exact-repeat.js';
import * as fillerWords from './src/filler-words.js';
import * as indexNumbering from './src/index-numbering.js';
import * as placeholderAsLabel from './src/placeholder-as-label.js';
import * as repeatedChrome from './src/repeated-chrome.js';
import * as sharedLeadIn from './src/shared-lead-in.js';
import * as vacuousAction from './src/vacuous-action.js';

const MODULES = [repeatedChrome, vacuousAction, fillerWords, placeholderAsLabel, sharedLeadIn, indexNumbering, exactRepeat];

// Rule metadata, in report order: {id, code, description}. A later ESLint plugin maps these ids one to one.
export const RULES = Object.freeze(MODULES.map((m) => Object.freeze({ id: m.id, code: m.code, description: m.description })));
export const RULE_IDS = Object.freeze(RULES.map((r) => r.id));

export class ScreenError extends Error {}

function validate(screen) {
  if (!screen || screen.schema !== 'slop.screen/1') throw new ScreenError('not a slop.screen/1 document');
  if (!Array.isArray(screen.nodes)) throw new ScreenError('screen has no node list');
  screen.nodes.forEach((n, x) => {
    if (typeof n.text !== 'string' || typeof n.role !== 'string') throw new ScreenError(`node ${x}: text and role must be strings`);
  });
}

// Findings for one screen, grouped by rule in RULES order and by first location within a rule. `only` limits the
// run to some rule ids.
export function runRules(screen, { only = null } = {}) {
  validate(screen);
  if (only) {
    const unknown = only.filter((r) => !RULE_IDS.includes(r));
    if (unknown.length) throw new ScreenError(`unknown rule ${unknown.join(', ')}`);
  }
  const out = [];
  for (const m of MODULES) {
    if (only && !only.includes(m.id)) continue;
    const fs = m.check(screen);
    fs.sort((a, b) => cmpLoc(a.locs[0], b.locs[0]));
    out.push(...fs);
  }
  return out;
}

function cmpLoc(a, b) {
  if (!a || !b) return 0;
  for (let p = 0; p < 4; p++) if (a[p] !== b[p]) return a[p] - b[p];
  return 0;
}
