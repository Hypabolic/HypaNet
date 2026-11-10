// slop/vacuous-action (aside R7): an action label that names a gesture or a stock phrase instead of the outcome.

import { finding } from './finding.js';
import { GESTURE_LEADS, GESTURE_PHRASES, VACUOUS_LABELS } from './lists.js';
import { isScored } from './screen.js';
import { clip, nodeBranches, tokens } from './text.js';

export const id = 'slop/vacuous-action';
export const code = 'R7';
export const description = 'Action label that names a gesture or a stock phrase instead of the outcome ("Learn more", "Click here").';

const ACTIONS = new Set(['button', 'link']);
const has = (words, phrase) => ` ${words.join(' ')} `.includes(` ${phrase} `);

// A named gesture ("double-click", "right click", "long-press") is one word here: "Double-click here to rename" tells
// the reader about a hidden input (LABELLING.md K4), it does not point at a link.
const GESTURE_MODIFIERS = new Set(['double', 'triple', 'right', 'middle', 'long']);
const GESTURES = new Set(['click', 'tap', 'press']);
// Copy words of one branch, a named gesture joined into one word only when a space or a hyphen separates its two parts
// ("double-click", "long press"; not "… is long. Click here").
function copyWords(text, vw) {
  const out = [];
  let joinable = false;
  for (const t of tokens(text, vw)) {
    if ((t.k === 'word' || t.k === 'number') && !t.ident) {
      if (joinable && GESTURES.has(t.w) && GESTURE_MODIFIERS.has(out[out.length - 1])) out[out.length - 1] += t.w;
      else out.push(t.w);
      joinable = true;
    } else if (!(t.k === 'other' && /^[-‐‑]$/u.test(t.s))) joinable = false;
  }
  return out;
}

const FRAGMENT_WORDS = new Set(['more', 'here']);

export function check(screen) {
  const out = [];
  screen.nodes.forEach((n, x) => {
    if (!isScored(n)) return;
    const branches = nodeBranches(n).map((b) => ({ ...b, words: copyWords(b.text, b.vw) }));
    for (const { text: b, words } of branches) {
      const label = words.join(' ');
      let why = null;
      // A lone "more" or "here" beside a longer branch is the tail of a conditional inside the copy ("See fewer ALT
      // more" is `See {x ? "fewer" : "more"}`): the extractor joins the shared words to one branch only. A whole-label
      // branch such as "Submit" in "Sending your form ALT Submit" is still read as a label.
      const fragment = words.length === 1 && FRAGMENT_WORDS.has(words[0]) && branches.some((o) => o.words.length >= 2);
      if (ACTIONS.has(n.role) && !fragment) {
        const hit = VACUOUS_LABELS.find((e) => e.phrase === label && e.roles.includes(n.role));
        if (hit) why = `${n.role} "${clip(b)}" is a stock label; name the outcome or destination`;
        else if (GESTURE_LEADS.some((g) => label.startsWith(`${g} `))) why = `${n.role} "${clip(b)}" names the gesture, not the outcome`;
      }
      if (!why) {
        const g = GESTURE_PHRASES.find((e) => has(words, e.phrase));
        if (g) why = `${n.role} "${clip(b)}" points at a link with "${g.phrase}"; put the destination in the link text`;
      }
      if (why) {
        out.push(finding(id, code, screen, [x], why));
        return; // one finding per node
      }
    }
  });
  return out;
}
