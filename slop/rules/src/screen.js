// Screen structure shared by the rules: which nodes are scored, precedence ranks (LABELLING.md §5), sections, and
// groups of sibling items (cards, list rows, form fields).

import { IDS } from '../../lint/tokenizer.js';

const CONTEXT_ROLES = new Set(IDS.context_roles);
export const HEADINGS = new Set(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']);

export const hasFlag = (node, f) => (node.flags || []).includes(f);

// Context-only roles and sr_only nodes are never labelled (LABELLING.md §2), so no rule reports them.
export const isScored = (node) => !CONTEXT_ROLES.has(node.role) && !hasFlag(node, 'sr_only');

// Precedence (LABELLING.md §5): lower is kept. A `label` ranks with the controls only when it belongs to an input
// (the extractor's `in_control`), otherwise with body text.
const RANK1 = new Set(['button', 'link', 'nav', 'tab', 'menu_item', 'breadcrumb', 'h1', 'dialog_title', 'empty_title']);
const RANK3 = new Set(['body', 'list_item', 'dialog_body', 'empty_body', 'toast', 'error', 'stat', 'cell', 'input_value', 'other']);
const RANK4 = new Set(['subtitle', 'helper', 'caption']);
export function rank(node) {
  if (RANK1.has(node.role)) return 10;
  if (node.role === 'label') return hasFlag(node, 'in_control') ? 10 : 30;
  if (HEADINGS.has(node.role)) return 18 + Number(node.role[1]); // h2 20 … h6 24
  if (RANK3.has(node.role)) return 30;
  if (RANK4.has(node.role)) return 40;
  return 50; // eyebrow, badge, tooltip, placeholder
}

// A section heading starts a page section: a heading that is not inside a card, list, table, navigation, dialog or
// state, and not one of a set of repeated siblings. Used by the eyebrow quota and the state-dot count.
const NOT_SECTION = ['in_card', 'repeated', 'in_list', 'in_table', 'in_nav', 'in_dialog', 'in_state'];
export const isSectionHeading = (node) => HEADINGS.has(node.role) && isScored(node) && !NOT_SECTION.some((f) => hasFlag(node, f));

// Section number of every node: 0 before the first section heading, then 1, 2, … from each section heading on.
export function sectionOf(nodes) {
  let s = 0;
  return nodes.map((n) => (isSectionHeading(n) ? ++s : s));
}

// Clusters in reading order as {k, kind, nodes: [list index]} (clusters are contiguous, SPEC.md §4).
export function clusterRuns(screen) {
  const kinds = new Map((screen.clusters || []).map((c) => [c.k, c.kind]));
  const runs = [];
  screen.nodes.forEach((n, x) => {
    const last = runs[runs.length - 1];
    if (last && last.k === n.k) last.nodes.push(x);
    else runs.push({ k: n.k, kind: kinds.get(n.k) ?? 'body', nodes: [x] });
  });
  return runs;
}

// Groups of sibling items. Each group is {source, items: [[list index of scored nodes]]} with two or more items:
//   - `cards`: a run of consecutive `card` clusters, one item per card;
//   - `fields`: a run of consecutive `control` clusters, one item per field;
//   - `rows`: one `list` cluster split into rows (below).
export function siblingGroups(screen) {
  const nodes = screen.nodes;
  const runs = clusterRuns(screen).map((r) => ({ ...r, nodes: r.nodes.filter((x) => isScored(nodes[x])) }));
  const groups = [];
  const consecutive = (kind, source) => {
    let cur = [];
    const flush = () => {
      if (cur.length >= 2) groups.push({ source, items: cur });
      cur = [];
    };
    for (const r of runs) {
      if (!r.nodes.length) continue; // a cluster of context-only nodes neither joins nor breaks a run
      if (r.kind === kind) cur.push(r.nodes);
      else flush();
    }
    flush();
  };
  consecutive('card', 'cards');
  consecutive('control', 'fields');
  for (const r of runs) {
    if (r.kind !== 'list') continue;
    // Rows start at each node with the role of the first node whose role occurs twice or more; nodes before that (a
    // list's own title) belong to no row.
    const count = (role) => r.nodes.filter((x) => nodes[x].role === role).length;
    const start = r.nodes.findIndex((x) => count(nodes[x].role) >= 2);
    if (start < 0) continue;
    const head = nodes[r.nodes[start]].role;
    const items = [];
    for (const x of r.nodes.slice(start)) {
      if (nodes[x].role === head) items.push([x]);
      else items[items.length - 1].push(x);
    }
    if (items.length >= 2) groups.push({ source: 'rows', items });
  }
  return groups;
}

// For each role present in every item of a group, the first node of that role in each item.
export function alignedSets(screen, group) {
  const roles = [];
  for (const x of group.items[0]) if (!roles.includes(screen.nodes[x].role)) roles.push(screen.nodes[x].role);
  const sets = [];
  for (const role of roles) {
    const set = group.items.map((item) => item.find((x) => screen.nodes[x].role === role));
    if (set.every((x) => x !== undefined)) sets.push({ role, nodes: set });
  }
  return sets;
}
