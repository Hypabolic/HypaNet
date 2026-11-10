// slop/index-numbering (aside R12): index numbers ("01 /", "(002)", "Step 3", "Phase 04") printed over items whose
// order the layout already shows. Taste, not evidence: LABELLING.md §9 notes the tasteskill study found the device
// more common on top-rated sites (research.md:162, 38% against 21%).

import { finding } from './finding.js';
import { isScored } from './screen.js';
import { clip, literalWords, parseIndex } from './text.js';

export const id = 'slop/index-numbering';
export const code = 'R12';
export const description = 'Index numbering over items already in order (K3 exceptions aside).';

// Controls (a pager's "1 2 3"), table cells, figures and inputs carry numbers that are data, not an index device.
const SKIP_ROLES = new Set(['nav', 'tab', 'breadcrumb', 'menu_item', 'link', 'button', 'cell', 'stat', 'input_value', 'placeholder']);
// A run of bare numbers ("1", "2", "3") needs three, the marked devices two.
const MIN_RUN = (family) => (family === 'plain' ? 3 : 2);

// Runs of index devices: same family and role, consecutive values (1, 2, 3 …) in reading order, other nodes between
// them allowed. K3 is applied as far as text can show it: a lone "Step 2 of 4" (the other steps are not on screen)
// never forms a run, and a run of labelled numbers that another node refers to ("as in step 2") is skipped. Whether
// numbered items are steps to be done in order is not decidable here; the rule flags them and the label decides.
export function indexRuns(screen) {
  const nodes = screen.nodes;
  const byKey = new Map();
  nodes.forEach((n, x) => {
    if (!isScored(n) || SKIP_ROLES.has(n.role)) return;
    const d = parseIndex(n.text);
    if (!d) return;
    const key = `${d.family}|${n.role}`;
    if (!byKey.has(key)) byKey.set(key, []);
    byKey.get(key).push({ x, ...d });
  });
  const runs = [];
  for (const devices of byKey.values()) {
    let cur = [];
    const flush = () => {
      if (cur.length >= MIN_RUN(cur[0].family)) runs.push(cur);
      cur = [];
    };
    for (const d of devices) {
      if (cur.length && d.value !== cur[cur.length - 1].value + 1) flush();
      cur.push(d);
    }
    if (cur.length) flush();
  }
  return runs.filter((run) => !referenced(screen, run)).sort((a, b) => a[0].x - b[0].x);
}

function referenced(screen, run) {
  const word = run[0].word;
  if (!word) return false;
  const inRun = new Set(run.map((d) => d.x));
  const refs = new Set(run.map((d) => `${word} ${d.value}`));
  return screen.nodes.some((n, x) => {
    if (inRun.has(x) || !isScored(n)) return false;
    const w = literalWords(n.text).map((s) => s.replace(/^0+(?=\d)/, ''));
    for (let a = 0; a + 1 < w.length; a++) if (refs.has(`${w[a]} ${w[a + 1]}`)) return true;
    return false;
  });
}

export function check(screen) {
  return indexRuns(screen).map((run) => {
    const xs = run.map((d) => d.x);
    const first = screen.nodes[xs[0]];
    return finding(id, code, screen, xs,
      `${xs.length} ${first.role} nodes number their items ("${clip(first.text, 24)}" …); position already gives the order`);
  });
}
