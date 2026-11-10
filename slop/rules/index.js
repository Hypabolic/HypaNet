// The plain-code rule layer (SPEC.md §8): seven rules over one slop.screen/1 document, no network, no dependencies
// beyond Node and lint/tokenizer.js (for the window split). Scored against the `aside` codes, never credited to the
// network. Ten codex rules follow them (slop/codex/llm.md, ui.md): LLM-writing patterns decided by word lists,
// constructions and glyph counts, reported with their codex ids (L.., U..) or, for the leading emoji, aside R11g.
// Their lists are bundled into src/codex-lists.js by make_codex_lists.mjs.

import * as exactRepeat from './src/exact-repeat.js';
import * as fillerWords from './src/filler-words.js';
import * as indexNumbering from './src/index-numbering.js';
import * as placeholderAsLabel from './src/placeholder-as-label.js';
import * as repeatedChrome from './src/repeated-chrome.js';
import * as sharedLeadIn from './src/shared-lead-in.js';
import * as vacuousAction from './src/vacuous-action.js';
import * as chatbotResidue from './src/chatbot-residue.js';
import * as placeholderCopy from './src/placeholder-copy.js';
import * as llmConstruction from './src/llm-construction.js';
import * as stockCta from './src/stock-cta.js';
import * as emptyClaim from './src/empty-claim.js';
import * as promoVocab from './src/promo-vocab.js';
import * as sparkleEmoji from './src/sparkle-emoji.js';
import * as emDashDensity from './src/em-dash-density.js';
import * as exclamationDensity from './src/exclamation-density.js';
import * as pleonasm from './src/pleonasm.js';

const MODULES = [repeatedChrome, vacuousAction, fillerWords, placeholderAsLabel, sharedLeadIn, indexNumbering, exactRepeat,
  chatbotResidue, placeholderCopy, llmConstruction, stockCta, emptyClaim, promoVocab, sparkleEmoji, emDashDensity,
  exclamationDensity, pleonasm];

// Rule metadata, in report order: {id, code, severity, description}. `code` is the rule's pattern or aside code, or the
// range of codex ids its findings carry ("L09-L19"); each finding has its own exact code. `severity` is "error" (never
// meant to ship), "warning" (style, fix it) or "info" (check it: a density or a claim). A later ESLint plugin maps
// these ids one to one.
export const RULES = Object.freeze(MODULES.map((m) => Object.freeze({ id: m.id, code: m.code, severity: m.severity ?? 'warning', description: m.description })));
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
