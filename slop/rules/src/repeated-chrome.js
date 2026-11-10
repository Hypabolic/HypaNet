// slop/repeated-chrome (screen aside R19): the same device stamped across the screen. One finding per condition met
// (per section for state dots), listing the nodes that carry the device:
//   eyebrow-quota   eyebrows above more than ceil(sections / 3) section headings (taste: local-skills.md:121,
//                   design-taste-frontend-v2-experimental SKILL.md:254-256, a landing-page quota; "Hero counts as 1");
//   decorative-dot  any `dot` node that is not `stateful` (local-skills.md:145, v2-experimental SKILL.md:705);
//   state-dots      more than one `stateful` dot in one section (same line: "limited to one per page section");
//   arrow-links     an arrow suffix on every link, two links or more (local-skills.md:46, copy.md:126-127;
//                   anti-slop-guides.md:124, Hallmark gate 63);
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
  screen.nodes.forEach((n, x) => { if (isScored(n) && hasFlag(n, 'dot') && !hasFlag(n, 'stateful')) xs.push(x); });
  if (!xs.length) return [];
  const t = screen.nodes[xs[0]].text;
  return [{ condition: 'decorative-dot', xs,
    msg: `${xs.length === 1 ? 'a dot' : `${xs.length} dots`} on fixed copy ("${clip(t, 32)}"${xs.length > 1 ? ', …' : ''}); a dot should show live state` }];
}

function stateDots(screen) {
  const nodes = screen.nodes;
  const sec = sectionOf(nodes);
  const per = new Map();
  nodes.forEach((n, x) => {
    if (!isScored(n) || !hasFlag(n, 'dot') || !hasFlag(n, 'stateful')) return;
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
    else if (n.role === 'badge' || hasFlag(n, 'pill')) add(`badge:${normalised(n.text)}`, x);
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
