// slop/sparkle-emoji (aside R11g; codex L46): an emoji or sparkle leading a node whose words carry the content
// ("✨ Smart search", "🚀 Launch in minutes", "✅ Unlimited projects"), the bullet style of a pasted chat answer.
// LABELLING.md §6: the node stays KEEP for the network, with aside R11g for the rule layer; an emoji-only node is the
// network's (DECOR, R11) and is not reported here. Severity info. Glyphs: codex/lists/llm_glyphs.txt §L46 (both
// blocks), matched on the raw text with variation selectors ignored. Leading dots and bullets (● • ▸, §L47/L50) are
// slop/repeated-chrome's (R19) and the network's, not this rule's.

import { finding } from './finding.js';
import { sectionEntries } from './codex.js';
import { isScored } from './screen.js';
import { clip, literalWords } from './text.js';

export const id = 'slop/sparkle-emoji';
export const code = 'R11g';
export const severity = 'info';
export const description = 'Emoji or sparkle leading a UI string ("✨ Smart search", "🚀 Launch in minutes").';

const strip = (s) => s.replace(/️/gu, '');
const GLYPHS = [...new Set(sectionEntries('llm_glyphs', ['L46']).map((e) => strip(e.entry.trim())))].sort((a, b) => b.length - a.length);
// A table cell's ✅ / ❌ is the value (included / not included), not decoration.
const SKIP_ROLES = new Set(['cell', 'input_value']);

export function leadingGlyph(text) {
  const t = strip(text).replace(/^[\s]+/u, '');
  return GLYPHS.find((g) => t.startsWith(g)) ?? null;
}

export function check(screen) {
  const out = [];
  screen.nodes.forEach((n, x) => {
    if (!isScored(n) || SKIP_ROLES.has(n.role)) return;
    const g = leadingGlyph(n.text);
    if (!g || !literalWords(n.text, n.vw).length) return;
    out.push(finding(id, code, screen, [x], `${n.role} "${clip(n.text)}" leads with ${g}; let the words carry it`));
  });
  return out;
}
