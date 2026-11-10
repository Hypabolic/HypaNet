// slop/vacuous-action (aside R7): an action label that names a gesture or a stock phrase instead of the outcome.

import { finding } from './finding.js';
import { GESTURE_LEADS, GESTURE_PHRASES, VACUOUS_LABELS } from './lists.js';
import { isScored } from './screen.js';
import { clip, literalWords, nodeBranches } from './text.js';

export const id = 'slop/vacuous-action';
export const code = 'R7';
export const description = 'Action label that names a gesture or a stock phrase instead of the outcome ("Learn more", "Click here").';

const ACTIONS = new Set(['button', 'link']);
const has = (words, phrase) => ` ${words.join(' ')} `.includes(` ${phrase} `);

// A named gesture ("double-click", "right click", "long-press") is one word here: "Double-click here to rename" tells
// the reader about a hidden input (LABELLING.md K4), it does not point at a link.
const GESTURE_MODIFIERS = new Set(['double', 'triple', 'right', 'middle', 'long']);
const GESTURES = new Set(['click', 'tap', 'press']);
function joinGestures(words) {
  const out = [];
  for (let a = 0; a < words.length; a++) {
    if (GESTURE_MODIFIERS.has(words[a]) && GESTURES.has(words[a + 1])) out.push(words[a] + words[++a]);
    else out.push(words[a]);
  }
  return out;
}

export function check(screen) {
  const out = [];
  screen.nodes.forEach((n, x) => {
    if (!isScored(n)) return;
    for (const { text: b, vw } of nodeBranches(n)) {
      const words = joinGestures(literalWords(b, vw));
      const label = words.join(' ');
      let why = null;
      if (ACTIONS.has(n.role)) {
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
