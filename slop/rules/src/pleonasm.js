// slop/pleonasm (codex U51): an acronym followed by the word it already stands for ("PIN number", "PDF format", "ATM
// machine", "FAQ questions", "2FA authentication"). Rule layer only: the repetition is inside one node. Severity
// warning. List: codex/lists/ui_tech_pleonasms.txt (nine entries shared with proselint's RAS-syndrome check, BSD-3-Clause;
// notice in codex/lists/ATTRIBUTION-ui.md).

import { finding } from './finding.js';
import { compileAll, hits, matchTokens, pick } from './codex.js';
import { isScored } from './screen.js';
import { clip, nodeBranches } from './text.js';

export const id = 'slop/pleonasm';
export const code = 'U51';
export const severity = 'warning';
export const description = 'Acronym followed by the word it stands for ("PIN number", "PDF format", "ATM machine").';

export const EXCLUDE = {
  'led light': 'a product name in a shop ("LED light strip")',
  'pin code': 'the Indian postcode is a "PIN code"',
};
const LIST = compileAll(pick('ui_tech_pleonasms', null, EXCLUDE));

export function check(screen) {
  const out = [];
  screen.nodes.forEach((n, x) => {
    if (!isScored(n)) return;
    const found = [];
    for (const b of nodeBranches(n)) for (const h of hits(LIST, matchTokens(b.text, b.vw))) if (!found.includes(h.entry)) found.push(h.entry);
    if (found.length) out.push(finding(id, code, screen, [x], `${n.role} "${clip(n.text)}": ${found.map((e) => `"${e}"`).join(', ')} repeats what the acronym already says`));
  });
  return out;
}
