// slop/promo-vocab (codex L01, L02, L03, L08): promotional vocabulary at a density no plain description has. One
// hype verb or adjective is weak evidence ("transform" in an image editor), so a node is reported only when it holds
// two or more distinct entries ("Seamlessly supercharge your workflow", "A powerful, intuitive editor"); overlapping
// entries count once, the longest ("unlock the power of", not "unlock"). Severity info. List:
// codex/lists/llm_promo_vocab.txt; the code is the section of the node's first entry.

import { finding } from './finding.js';
import { compileAll, hits, matchTokens, pick } from './codex.js';
import { isScored } from './screen.js';
import { clip, nodeBranches } from './text.js';

export const id = 'slop/promo-vocab';
export const code = 'L01-L08';
export const severity = 'info';
export const description = 'Two or more promotional words in one string ("Seamlessly supercharge your workflow").';

export const MIN_DISTINCT = 2;
const LITERAL = 'names a real thing or action in some products';
const R15 = 'reported by slop/filler-words (R15)';
export const EXCLUDE = {
  // L01
  discover: 'a navigation word', navigate: LITERAL, navigating: LITERAL, cultivate: LITERAL,
  // L02
  unique: LITERAL, smart: 'a feature name ("Smart playlists")', intelligent: LITERAL, dynamic: LITERAL, personalised: LITERAL,
  personalized: LITERAL, tailored: LITERAL, curated: LITERAL, crafted: LITERAL, handcrafted: LITERAL, bespoke: LITERAL,
  scalable: LITERAL, 'mission-critical': LITERAL, premier: 'a name ("Premier League")', 'end-to-end': 'end-to-end encryption',
  automatically: 'a fact ("Saves automatically")', securely: LITERAL, reliably: LITERAL, easily: R15, simply: R15,
  // L03: only the decorative nouns count; these name real things
  experience: LITERAL, experiences: LITERAL, solution: LITERAL, solutions: LITERAL, workflow: LITERAL, workflows: LITERAL,
  productivity: LITERAL, potential: LITERAL, landscape: 'an orientation', realm: 'a game world',
};
const LIST = compileAll(pick('llm_promo_vocab', ['L01', 'L02', 'L03', 'L08'], EXCLUDE));
const SKIP_ROLES = new Set(['placeholder', 'input_value', 'cell', 'menu_item']);

export function check(screen) {
  const out = [];
  screen.nodes.forEach((n, x) => {
    if (!isScored(n) || SKIP_ROLES.has(n.role)) return;
    for (const b of nodeBranches(n)) {
      const hs = hits(LIST, matchTokens(b.text, b.vw)).sort((p, q) => p.m.from - q.m.from);
      const distinct = [...new Set(hs.map((h) => h.entry))];
      if (distinct.length >= MIN_DISTINCT) {
        out.push(finding(id, hs[0].id, screen, [x], `${n.role} "${clip(b.text)}": ${distinct.map((e) => `"${e}"`).join(', ')}: promotional words where a fact would do`));
        return;
      }
    }
  });
  return out;
}
