// slop/placeholder-as-label (aside R8b): a placeholder standing in for a missing visible label
// (anti-slop-guides.md:236, GOV.UK: "Do not use placeholder text in place of a label"; :281, NN/g).

import { finding } from './finding.js';
import { clusterRuns, hasFlag, isScored } from './screen.js';
import { clip } from './text.js';

export const id = 'slop/placeholder-as-label';
export const code = 'R8b';
export const description = 'Placeholder used instead of a visible label.';

// A placeholder has a visible label when its `control` cluster holds a scored `label` node, or when the nearest scored
// node before it, skipping helper, tooltip, error and caption text, is a `label` in the same cluster or in no control
// cluster (the extractor puts a <label for> in a body cluster and the input's placeholder in a control cluster of its
// own). An `aria` node or an sr_only label is not visible, so it does not count.
const HINTS = new Set(['helper', 'tooltip', 'error', 'caption']);
function labelBefore(nodes, x, k) {
  for (let y = x - 1; y >= 0; y--) {
    const n = nodes[y];
    if (!isScored(n) || HINTS.has(n.role)) continue;
    return n.role === 'label' && (n.k === k || !hasFlag(n, 'in_control'));
  }
  return false;
}

export function check(screen) {
  const nodes = screen.nodes;
  const out = [];
  for (const run of clusterRuns(screen)) {
    const scored = run.nodes.filter((x) => isScored(nodes[x]));
    scored.forEach((x) => {
      if (nodes[x].role !== 'placeholder') return;
      const labelled = (run.kind === 'control' && scored.some((y) => nodes[y].role === 'label')) || labelBefore(nodes, x, run.k);
      if (!labelled) out.push(finding(id, code, screen, [x], `placeholder "${clip(nodes[x].text)}" stands in for a label: the field has no visible label`));
    });
  }
  return out;
}
