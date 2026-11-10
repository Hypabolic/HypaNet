// slop/repeated-chrome (screen aside R19): the same device stamped across the screen. One finding per condition met
// (per section for state dots), listing the nodes that carry the device:
//   eyebrow-quota   eyebrows above more than ceil(sections / 3) section headings (taste: local-skills.md:121,
//                   design-taste-frontend-v2-experimental SKILL.md:254-256, a landing-page quota; "Hero counts as 1");
//   decorative-dot  any `dot` node that is neither `stateful` nor `dynamic` (local-skills.md:145, v2-experimental
//                   SKILL.md:705);
//   state-dots      more than one `stateful` dot in one section (same line: "limited to one per page section"), not
//                   counting one dot per sibling card or row, each bound to its own item (LABELLING.md K2);
//   arrow-links     an arrow suffix on every link, two links or more (local-skills.md:46, copy.md:126-127;
//                   anti-slop-guides.md:124, Hallmark gate 63); ↗ and » are not arrow suffixes (text.js);
//   sibling-device  the same badge, pill or eyebrow text, an arrow suffix or an index number on most (more than half)
//                   of three or more sibling cards or rows (LABELLING.md §9, R19).

import { finding } from './finding.js';
import { indexRuns } from './index-numbering.js';
import { hasFlag, isScored, isSectionHeading, sectionOf, siblingGroups } from './screen.js';
import { clip, hasArrowTail, normalised } from './text.js';

export const id = 'slop/repeated-chrome';
export const code = 'R19';
export const description = 'Repeated chrome: eyebrow quota, decorative dots, state dots per section, arrows on every link, one device on most siblings.';

// The scored node after x in reading order, or -1.
function nextScored(nodes, x) {
  for (let y = x + 1; y < nodes.length; y++) if (isScored(nodes[y])) return y;
  return -1;
}

function eyebrowQuota(screen) {
  const nodes = screen.nodes;
  const sections = nodes.filter(isSectionHeading).length;
  const eyebrows = [];
  nodes.forEach((n, x) => {
    if (n.role !== 'eyebrow' || !isScored(n)) return;
    const y = nextScored(nodes, x);
    if (y >= 0 && isSectionHeading(nodes[y])) eyebrows.push(x);
  });
  const quota = Math.ceil(sections / 3);
  if (eyebrows.length <= quota) return [];
  return [{ condition: 'eyebrow-quota', xs: eyebrows,
    msg: `${eyebrows.length} eyebrows over ${sections} section headings; the quota is ${quota} (one per three sections)` }];
}

function decorativeDots(screen) {
  const xs = [];
  screen.nodes.forEach((n, x) => { if (isScored(n) && hasFlag(n, 'dot') && !hasFlag(n, 'stateful') && !hasFlag(n, 'dynamic')) xs.push(x); });
  if (!xs.length) return [];
  const t = screen.nodes[xs[0]].text;
  return [{ condition: 'decorative-dot', xs,
    msg: `${xs.length === 1 ? 'a dot' : `${xs.length} dots`} on fixed copy ("${clip(t, 32)}"${xs.length > 1 ? ', …' : ''}); a dot should show live state` }];
}

// K2 wins over the per-section limit: in a group of sibling cards or rows where two or more items carry a stateful dot
// and none carries more than one, each dot shows its own item's state, so those dots are not counted.
function oneDotPerItem(screen, isStateDot) {
  const out = new Set();
  for (const group of siblingGroups(screen)) {
    if (group.source === 'fields') continue;
    const dots = group.items.map((item) => item.filter(isStateDot));
    if (dots.filter((d) => d.length).length >= 2 && dots.every((d) => d.length <= 1)) for (const d of dots) for (const x of d) out.add(x);
  }
  return out;
}

function stateDots(screen) {
  const nodes = screen.nodes;
  const sec = sectionOf(nodes);
  const isStateDot = (x) => isScored(nodes[x]) && hasFlag(nodes[x], 'dot') && hasFlag(nodes[x], 'stateful');
  const perItem = oneDotPerItem(screen, isStateDot);
  const per = new Map();
  nodes.forEach((n, x) => {
    if (!isStateDot(x) || perItem.has(x)) return;
    if (!per.has(sec[x])) per.set(sec[x], []);
    per.get(sec[x]).push(x);
  });
  const out = [];
  for (const [s, xs] of per) {
    if (xs.length < 2) continue;
    out.push({ condition: 'state-dots', xs,
      msg: `${xs.length} live-state dots in one section (${s === 0 ? 'before the first heading' : `section ${s}`}); keep one` });
  }
  return out;
}

function arrowLinks(screen) {
  const links = [];
  screen.nodes.forEach((n, x) => { if (isScored(n) && n.role === 'link') links.push(x); });
  if (links.length < 2 || !links.every((x) => hasArrowTail(screen.nodes[x]))) return [];
  return [{ condition: 'arrow-links', xs: links, msg: `all ${links.length} links end in an arrow; the arrow tells the reader nothing` }];
}

// A rounded-full button, link or field is a control's shape, not a pill device (the extractor sets `pill` from
// `rounded-full`); a `badge` counts whatever its shape.
const CONTROLS = new Set(['button', 'link', 'nav', 'tab', 'menu_item', 'label', 'placeholder', 'input_value']);

// Device signatures of one sibling item (a card or a list row).
function signatures(screen, item, indexed) {
  const sigs = new Map();
  const add = (sig, x) => { if (!sigs.has(sig)) sigs.set(sig, x); };
  for (const x of item) {
    const n = screen.nodes[x];
    if (indexed.has(x)) add(`index:${indexed.get(x)}`, x);
    if ((n.role === 'link' || n.role === 'button') && hasArrowTail(n)) add('arrow', x);
    if (hasFlag(n, 'stateful') || hasFlag(n, 'dynamic')) continue; // bound text differs by item (K2)
    if (n.role === 'eyebrow') add(`eyebrow:${normalised(n.text)}`, x);
    else if (n.role === 'badge' || (hasFlag(n, 'pill') && !CONTROLS.has(n.role))) add(`badge:${normalised(n.text)}`, x);
  }
  return sigs;
}

function siblingDevices(screen, already) {
  const indexed = new Map();
  for (const run of indexRuns(screen)) for (const d of run) indexed.set(d.x, d.family);
  const out = [];
  for (const group of siblingGroups(screen)) {
    if (group.source === 'fields' || group.items.length < 3) continue;
    const counts = new Map();
    for (const item of group.items) {
      for (const [sig, x] of signatures(screen, item, indexed)) {
        if (!counts.has(sig)) counts.set(sig, []);
        counts.get(sig).push(x);
      }
    }
    for (const [sig, xs] of counts) {
      if (xs.length * 2 <= group.items.length) continue;
      if (xs.every((x) => already.has(x))) continue; // the same nodes are already in another R19 finding
      const what = sig === 'arrow' ? 'an arrow suffix' : sig.startsWith('index:') ? 'an index number'
        : `the ${sig.split(':')[0]} "${clip(screen.nodes[xs[0]].text, 24)}"`;
      const kind = group.source === 'cards' ? 'cards' : 'rows';
      out.push({ condition: 'sibling-device', xs, msg: `${what} on ${xs.length} of ${group.items.length} sibling ${kind}` });
    }
  }
  return out;
}

export function check(screen) {
  const hits = [...eyebrowQuota(screen), ...decorativeDots(screen), ...stateDots(screen), ...arrowLinks(screen)];
  const already = new Set(hits.flatMap((h) => h.xs));
  hits.push(...siblingDevices(screen, already));
  return hits.map((h) => finding(id, code, screen, h.xs, h.msg, { condition: h.condition }));
}
