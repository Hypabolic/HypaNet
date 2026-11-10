// slop/vacuous-action (aside R7): an action label that names a gesture or a stock phrase instead of the outcome.

import { finding } from './finding.js';
import { GESTURE_LEADS, GESTURE_PHRASES, VACUOUS_LABELS } from './lists.js';
import { isScored } from './screen.js';
import { branches, clip, literalWords } from './text.js';

export const id = 'slop/vacuous-action';
export const code = 'R7';
export const description = 'Action label that names a gesture or a stock phrase instead of the outcome ("Learn more", "Click here").';

const ACTIONS = new Set(['button', 'link']);
const has = (words, phrase) => ` ${words.join(' ')} `.includes(` ${phrase} `);

export function check(screen) {
  const out = [];
  screen.nodes.forEach((n, x) => {
    if (!isScored(n)) return;
    for (const b of branches(n.text)) {
      const words = literalWords(b);
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
