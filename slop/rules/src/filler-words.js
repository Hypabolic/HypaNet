// slop/filler-words (aside R15): words that add length and no meaning to a UI string.

import { finding } from './finding.js';
import { ACTION_ROLES, FILLER, TITLE_LEAD_MAX_WORDS, TITLE_ROLES } from './lists.js';
import { isScored } from './screen.js';
import { branches, clip, literalWords } from './text.js';

export const id = 'slop/filler-words';
export const code = 'R15';
export const description = 'Filler words in a UI string ("please", "simply", "Apply now", "Your Favorites").';

function contains(words, phrase) {
  const p = phrase.split(' ');
  for (let a = 0; a + p.length <= words.length; a++) if (p.every((w, b) => words[a + b] === w)) return true;
  return false;
}

export function check(screen) {
  const out = [];
  screen.nodes.forEach((n, x) => {
    if (!isScored(n)) return;
    const found = [];
    for (const b of branches(n.text)) {
      const words = literalWords(b);
      for (const e of FILLER) {
        if (found.includes(e.phrase)) continue;
        if (e.scope === 'action' && !ACTION_ROLES.has(n.role)) continue;
        if (e.scope === 'title-lead') {
          if (TITLE_ROLES.has(n.role) && words.length <= TITLE_LEAD_MAX_WORDS && words.length >= 2 && words[0] === e.phrase) found.push(e.phrase);
          continue;
        }
        if (contains(words, e.phrase)) found.push(e.phrase);
      }
    }
    if (found.length) {
      const list = found.map((p) => `"${p}"`).join(', ');
      out.push(finding(id, code, screen, [x], `${n.role} "${clip(n.text)}": ${list} ${found.length > 1 ? 'add' : 'adds'} length and no meaning`));
    }
  });
  return out;
}
