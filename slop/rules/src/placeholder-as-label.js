// slop/placeholder-as-label (aside R8b): a placeholder standing in for a missing visible label
// (anti-slop-guides.md:236, GOV.UK: "Do not use placeholder text in place of a label"; :281, NN/g).

import { finding } from './finding.js';
import { clusterRuns, isScored } from './screen.js';
import { clip } from './text.js';

export const id = 'slop/placeholder-as-label';
export const code = 'R8b';
export const description = 'Placeholder used instead of a visible label.';

// A placeholder has a visible label when its `control` cluster holds a scored `label` node, or, outside a control
// cluster, when the scored node just before it in the same cluster is a `label`. An `aria` node or an sr_only label
// is not visible, so it does not count.
export function check(screen) {
  const nodes = screen.nodes;
  const out = [];
  for (const run of clusterRuns(screen)) {
    const scored = run.nodes.filter((x) => isScored(nodes[x]));
    scored.forEach((x, pos) => {
      if (nodes[x].role !== 'placeholder') return;
      const labelled = run.kind === 'control'
        ? scored.some((y) => nodes[y].role === 'label')
        : pos > 0 && nodes[scored[pos - 1]].role === 'label';
      if (!labelled) out.push(finding(id, code, screen, [x], `placeholder "${clip(nodes[x].text)}" stands in for a label: the field has no visible label`));
    });
  }
  return out;
}
