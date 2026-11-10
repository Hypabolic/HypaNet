// slop/index-numbering (aside R12): index numbers ("01 /", "(002)", "Step 3", "Phase 04") printed over items whose
// order the layout already shows. Taste, not evidence: LABELLING.md §9 notes the tasteskill study found the device
// more common on top-rated sites (research.md:162, 38% against 21%).

import { finding } from './finding.js';
import { hasFlag, isScored } from './screen.js';
import { clip, literalWords, parseIndex } from './text.js';

export const id = 'slop/index-numbering';
export const code = 'R12';
export const description = 'Index numbering over items already in order (K3 exceptions aside).';

// Controls (a pager's "1 2 3"), table cells, figures and inputs carry numbers that are data, not an index device.
const SKIP_ROLES = new Set(['nav', 'tab', 'breadcrumb', 'menu_item', 'link', 'button', 'cell', 'stat', 'input_value', 'placeholder']);
// A run of bare numbers ("1", "2", "3") needs three, the marked devices two.
const MIN_RUN = (family) => (family === 'plain' ? 3 : 2);

// Runs of index devices: same family and role, consecutive values (1, 2, 3 …) and the same total ("1 / 3", "2 / 3")
// in reading order, other nodes between them allowed. K3 is applied as far as text can show it: a lone "Step 2 of 4"
// (the other steps are not on screen) never forms a run, nor do conditional devices with a total (a wizard showing
// one step heading at a time) or a run that starts above one, and a run that another node refers to ("as in step 2",
// footnote markers) is skipped.
// Whether numbered items are steps to be done in order is not decidable here; the rule flags them and the label
// decides.
export function indexRuns(screen) {
  const nodes = screen.nodes;
  const byKey = new Map();
  // Roles that carry a padded device: there a bare two-digit number ("10") continues the padded column ("09", "10").
  const paddedRoles = new Set();
  nodes.forEach((n) => {
    if (!isScored(n) || SKIP_ROLES.has(n.role)) return;
    const d = parseIndex(n.text);
    if (d && d.family === 'padded' && !d.wide) paddedRoles.add(n.role);
  });
  nodes.forEach((n, x) => {
    if (!isScored(n) || SKIP_ROLES.has(n.role)) return;
    let d = parseIndex(n.text);
    if (!d) return;
    if (d.family === 'plain' && d.value >= 10 && /^\d{2}$/.test(n.text.trim()) && paddedRoles.has(n.role)) d = { family: 'padded', value: d.value, whole: true, total: null, wide: true };
    if (d.wide && !paddedRoles.has(n.role)) return;
    if (d.total && hasFlag(n, 'conditional')) return; // "Step 2 of 4" chosen by a condition: one state at a time (K3)
    // A bare number followed directly by another number device (or by nothing) is a value in a scale or a grid, not an
    // item's index.
    if (d.family === 'plain') {
      const y = nodes.findIndex((m, z) => z > x && isScored(m));
      const next = y < 0 ? null : parseIndex(nodes[y].text);
      if (y < 0 || (next && !next.wide)) return; // a wide number ("12 months") only continues a padded column
    }
    const key = `${d.family}|${n.role}`;
    if (!byKey.has(key)) byKey.set(key, []);
    byKey.get(key).push({ x, ...d });
  });
  const runs = [];
  for (const devices of byKey.values()) {
    let cur = [];
    // An index counts from one: a run that starts higher is a slice of values (hours "06", "07"), or K3's position in
    // a sequence whose earlier members are not on this screen.
    const flush = () => {
      if (cur.length >= MIN_RUN(cur[0].family) && cur[0].value <= 1) runs.push(cur);
      cur = [];
    };
    for (const d of devices) {
      const last = cur[cur.length - 1];
      const continues = last && d.value === last.value + 1 && (d.total ?? null) === (last.total ?? null);
      if (d.wide && !continues) continue; // a wide number only continues a padded column ("09" then "10")
      if (cur.length && !continues) flush();
      cur.push(d);
    }
    if (cur.length) flush();
  }
  return runs.filter((run) => !referenced(screen, run)).sort((a, b) => a[0].x - b[0].x);
}

// K3: another node refers to the numbers. A labelled run is referenced by its word, singular or plural, before one of
// its numbers or a list of them ("go back to step 1", "Steps 2 and 3 …"). An unlabelled run (footnote numbers) is
// referenced when another node carries at least two of its numbers as footnote markers: a digit glued to the word or
// punctuation before it ("£14 a night1", from <sup>1</sup>), a superscript digit or a bracketed "[1]".
const LIST_JOIN = new Set(['and', 'or', 'to', 'through']);
const SUPERSCRIPT = { '¹': 1, '²': 2, '³': 3, '⁴': 4, '⁵': 5, '⁶': 6, '⁷': 7, '⁸': 8, '⁹': 9 };
function markers(text) {
  const out = new Set();
  // After a letter, ")" or "%", or after punctuation that does not follow a digit: "1.5", "9:30" and "4,2" are not markers.
  for (const m of text.matchAll(/(?:[\p{L})%]|(?<!\d)[.,;:])(\d{1,2})(?!\d)(?![\p{L}])/gu)) out.add(Number(m[1]));
  for (const m of text.matchAll(/\[(\d{1,2})\]/gu)) out.add(Number(m[1]));
  for (const c of text) if (SUPERSCRIPT[c]) out.add(SUPERSCRIPT[c]);
  return out;
}
function referenced(screen, run) {
  const inRun = new Set(run.map((d) => d.x));
  const values = new Set(run.map((d) => d.value));
  const word = run[0].word;
  return screen.nodes.some((n, x) => {
    if (inRun.has(x) || !isScored(n)) return false;
    if (!word) return [...markers(n.text)].filter((v) => values.has(v)).length >= 2;
    const w = literalWords(n.text, n.vw).map((s) => s.replace(/^0+(?=\d)/, ''));
    for (let a = 0; a + 1 < w.length; a++) {
      if (w[a] !== word && w[a] !== `${word}s`) continue;
      for (let b = a + 1; b < w.length && (/^\d+$/.test(w[b]) || LIST_JOIN.has(w[b])); b++) if (values.has(Number(w[b]))) return true;
    }
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
