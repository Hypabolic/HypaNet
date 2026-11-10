// slop/shared-lead-in (aside R10): the same opening words on every sibling item, which belong once in the group label
// (anti-slop-guides.md:197, Microsoft: "If all the options have the same introductory text, move that text to the
// group label.").

import { finding } from './finding.js';
import { alignedSets, siblingGroups } from './screen.js';
import { clip, tokens } from './text.js';

export const id = 'slop/shared-lead-in';
export const code = 'R10';
export const description = 'The same lead-in on every sibling item (cards, list rows, options, field labels).';

// Roles whose siblings repeat by nature (table cells, figures, input values, breadcrumb trails) are left out.
const SKIP_ROLES = new Set(['cell', 'stat', 'input_value', 'breadcrumb']);
// A shared lead-in of one word is parallel phrasing ("Add photo" / "Add video"), which the same guide asks for; two
// words on three or more siblings, or three words on a pair, is a lead-in.
const MIN_WORDS = (items) => (items >= 3 ? 2 : 3);

// Leading copy words of a node, up to the first marker or identifier word.
function leadWords(text) {
  const out = [];
  for (const t of tokens(text)) {
    if (t.k === 'var' || t.k === 'alt' || t.ident) break;
    if (t.k === 'word' || t.k === 'number') out.push(t.w);
  }
  return out;
}

export function check(screen) {
  const nodes = screen.nodes;
  const out = [];
  const seen = new Set();
  for (const group of siblingGroups(screen)) {
    for (const set of alignedSets(screen, group)) {
      if (SKIP_ROLES.has(set.role)) continue;
      const key = set.nodes.join(',');
      if (seen.has(key)) continue;
      seen.add(key);
      const texts = set.nodes.map((x) => nodes[x].text.trim());
      if (texts.every((t) => t.endsWith('?'))) continue; // questions open alike by grammar ("How do I …")
      const words = set.nodes.map((x) => leadWords(nodes[x].text));
      let n = 0;
      while (words.every((w) => n < w.length && w[n] === words[0][n])) n++;
      if (n < MIN_WORDS(set.nodes.length)) continue;
      if (!words.every((w) => w.length > n)) continue; // the shared words are the whole of some item: a repeat, not a lead-in
      const lead = words[0].slice(0, n).join(' ');
      out.push(finding(id, code, screen, set.nodes,
        `${set.nodes.length} sibling ${set.role} nodes all open with "${lead}" ("${clip(texts[0], 40)}", …); say it once in the group label`));
    }
  }
  return out;
}
