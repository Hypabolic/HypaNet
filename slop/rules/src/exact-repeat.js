// slop/exact-repeat (R14, or R9 between two titles): the same normalised string in two windows of one file. Repeats
// inside one window are the network's to judge; across windows it cannot see them (SPEC.md §4, Windows).

import { encodeScreen } from '../../lint/tokenizer.js';
import { finding } from './finding.js';
import { HEADINGS, hasFlag, isScored, rank } from './screen.js';
import { clip, literalWords, normalised } from './text.js';

export const id = 'slop/exact-repeat';
export const code = 'R14';
export const description = 'The same normalised string in two windows of one file.';

// Roles that repeat by nature (LABELLING.md K5: navigation, tabs, breadcrumbs, menus and option lists, table values)
// and form furniture that two forms in one file share by design (labels, placeholders, input values). Nodes flagged
// `repeated` are the constant parts of one template stamped on sibling cards or rows (a "Not set" value, a per-section
// "Reset section" button, a "/month" qualifier, K13); repeating them is the template's nature, and a device on every
// sibling is slop/repeated-chrome's to report.
const SKIP_ROLES = new Set(['nav', 'tab', 'breadcrumb', 'menu_item', 'cell', 'input_value', 'label', 'placeholder']);
const TITLES = new Set([...HEADINGS, 'dialog_title', 'empty_title']);
const MIN_WORDS = 2;

// Each group of equal strings that spans two or more windows keeps its highest-ranked copy (the first at equal rank,
// LABELLING.md §5) and flags every copy in a window other than the kept copy's. A pair in which either copy is
// `conditional` is skipped: the two may sit in different branches or states, which are never visible together (§2a).
export function check(screen) {
  const nodes = screen.nodes;
  const enc = encodeScreen(screen);
  const windowOf = new Array(nodes.length);
  enc.windows.forEach((w, wi) => w.nodes.forEach((x) => { windowOf[x] = wi; }));
  if (enc.windows.length < 2) return [];
  const groups = new Map();
  nodes.forEach((n, x) => {
    if (!isScored(n) || SKIP_ROLES.has(n.role) || hasFlag(n, 'repeated')) return;
    if (literalWords(n.text).length < MIN_WORDS) return; // short labels and nodes with no literal words
    const key = normalised(n.text);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(x);
  });
  const out = [];
  for (const xs of groups.values()) {
    if (new Set(xs.map((x) => windowOf[x])).size < 2) continue;
    const keep = xs.reduce((a, b) => (rank(nodes[b]) < rank(nodes[a]) ? b : a));
    for (const x of xs) {
      if (x === keep || windowOf[x] === windowOf[keep]) continue;
      if (hasFlag(nodes[x], 'conditional') || hasFlag(nodes[keep], 'conditional')) continue;
      const pattern = TITLES.has(nodes[x].role) && TITLES.has(nodes[keep].role) ? 'R9' : 'R14';
      out.push(finding(id, pattern, screen, [x],
        `${nodes[x].role} "${clip(nodes[x].text)}" is also the ${nodes[keep].role} at line ${nodes[keep].loc?.[0] ?? '?'}, in another window (${windowOf[keep]}; this is ${windowOf[x]}); say it once`,
        { target: keep }));
    }
  }
  return out;
}
