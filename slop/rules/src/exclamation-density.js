// slop/exclamation-density (codex L51): exclamation marks on success messages, headings and buttons across a screen
// ("You're all set!", "Welcome aboard!", "Let's go!"). One is a voice; three or more strings with one is the hype
// register of generated copy. Screen-level, severity info. Glyph: codex/lists/llm_glyphs.txt §L51 ("!"), counted where
// it ends a sentence (before a space, a closing quote or bracket, or the end), never in code, a cell or a field value.

import { finding } from './finding.js';
import { sectionEntries } from './codex.js';
import { isScored } from './screen.js';

export const id = 'slop/exclamation-density';
export const code = 'L51';
export const severity = 'info';
export const description = 'Exclamation marks in three or more strings on one screen.';

export const MIN_NODES = 3;
if (!sectionEntries('llm_glyphs', ['L51']).some((e) => e.entry === '!')) throw new Error('llm_glyphs §L51 has no "!"');
const BANG = /\p{L}[!¡]*!+(?=$|[\s"'”’)\]])/u;
const SKIP_ROLES = new Set(['cell', 'input_value', 'stat']);

export function check(screen) {
  const xs = [];
  screen.nodes.forEach((n, x) => {
    if (isScored(n) && !SKIP_ROLES.has(n.role) && BANG.test(n.text)) xs.push(x);
  });
  if (xs.length < MIN_NODES) return [];
  return [finding(id, code, screen, xs, `${xs.length} strings on this screen end a sentence with "!"; keep it for the one moment that earns it`, { scope: 'screen' })];
}
