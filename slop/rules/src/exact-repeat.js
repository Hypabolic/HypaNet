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
const ACTIONS = new Set(['button', 'link']);

// Pairs that never echo each other: a title and a button or link acting on what it names (LABELLING.md K11, "Delete
// playlist?" / "Delete playlist"), and two nodes in different separately mounted states, such as two dialogs (§2a).
function exempt(p, q) {
  if ((TITLES.has(p.role) && ACTIONS.has(q.role)) || (ACTIONS.has(p.role) && TITLES.has(q.role))) return true;
  return hasFlag(p, 'in_state') && hasFlag(q, 'in_state') && p.k !== q.k;
}

// In each group of equal strings that spans two or more windows, `conditional` copies take no part (they may sit in
// different branches or states, never visible together, §2a), as flagged copies or as targets. A copy is flagged when
// a copy that outranks it (LABELLING.md §5: lower rank, or equal rank and earlier) sits in another window, with the
// best such copy as target, unless a copy that outranks it sits in its own window (the network sees that pair) or
// every such pair is exempt (above).
export function check(screen) {
  const nodes = screen.nodes;
  const enc = encodeScreen(screen);
  const windowOf = new Array(nodes.length);
  enc.windows.forEach((w, wi) => w.nodes.forEach((x) => { windowOf[x] = wi; }));
  if (enc.windows.length < 2) return [];
  const groups = new Map();
  nodes.forEach((n, x) => {
    if (!isScored(n) || SKIP_ROLES.has(n.role) || hasFlag(n, 'repeated')) return;
    if (n.role === 'link' && hasFlag(n, 'in_footer')) return; // footer links are navigation (K5)
    if (literalWords(n.text, n.vw).length < MIN_WORDS) return; // short labels and nodes with no literal words
    const key = normalised(n.text);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(x);
  });
  const out = [];
  for (const xs of groups.values()) {
    if (new Set(xs.map((x) => windowOf[x])).size < 2) continue;
    const fixed = xs.filter((x) => !hasFlag(nodes[x], 'conditional'));
    const better = (y, x) => rank(nodes[y]) < rank(nodes[x]) || (rank(nodes[y]) === rank(nodes[x]) && y < x);
    for (const x of fixed) {
      const above = fixed.filter((y) => y !== x && better(y, x) && !exempt(nodes[x], nodes[y]));
      if (!above.length || above.some((y) => windowOf[y] === windowOf[x])) continue;
      const keep = above.reduce((a, b) => (better(b, a) ? b : a));
      const pattern = TITLES.has(nodes[x].role) && TITLES.has(nodes[keep].role) ? 'R9' : 'R14';
      out.push(finding(id, pattern, screen, [x],
        `${nodes[x].role} "${clip(nodes[x].text)}" is also the ${nodes[keep].role} at line ${nodes[keep].loc?.[0] ?? '?'}, in another window (${windowOf[keep]}; this is ${windowOf[x]}); say it once`,
        { target: keep }));
    }
  }
  return out;
}
