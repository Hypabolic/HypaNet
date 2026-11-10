// slop/filler-words (aside R15): words that add length and no meaning to a UI string.

import { finding } from './finding.js';
import { ACTION_ROLES, FILLER, TITLE_LEAD_MAX_WORDS, TITLE_ROLES } from './lists.js';
import { isScored } from './screen.js';
import { clip, literalWords, nodeBranches } from './text.js';

export const id = 'slop/filler-words';
export const code = 'R15';
export const description = 'Filler words in a UI string ("please", "simply", "Apply now", "Your Favorites").';

function contains(words, phrase) {
  const p = phrase.split(' ');
  for (let a = 0; a + p.length <= words.length; a++) if (p.every((w, b) => words[a + b] === w)) return true;
  return false;
}

// Option lists and table values (LABELLING.md K5): there "Very low" … "Very high" names a value on a scale.
const DEGREE = new Set(['very', 'quite']);
const VALUE_ROLES = new Set(['menu_item', 'input_value', 'cell']);

export function check(screen) {
  const out = [];
  screen.nodes.forEach((n, x) => {
    if (!isScored(n)) return;
    const found = [];
    const bs = nodeBranches(n);
    const branches = bs.map((b) => literalWords(b.text, b.vw));
    const branchTexts = bs.map((b) => b.text);
    for (const [bx, words] of branches.entries()) {
      for (const e of FILLER) {
        if (found.includes(e.phrase)) continue;
        if (e.scope === 'action' && !ACTION_ROLES.has(n.role)) continue;
        if (DEGREE.has(e.phrase) && VALUE_ROLES.has(n.role)) continue; // "Very high" on a scale is a value (K5)
        if (e.scope === 'title-lead') {
          if (!TITLE_ROLES.has(n.role) || words[0] !== e.phrase) continue;
          // A title with a comma or a sentence mark is a slogan ("Your brand, your rules"), not the noun phrase
          // "Your Favorites" whose "your" adds nothing.
          if (/[,;:.!?]/u.test(branchTexts[bx])) continue;
          // "Your ALT Shared playlists" is `{mine ? "Your" : "Shared"} playlists`: the extractor joins the words after
          // the conditional to the last branch only, so a lone "your" beside a title branch of two to four words leads
          // a title of that length ("Your playlists").
          const lone = words.length === 1 && branches.some((o) => o.length >= 2 && o.length <= TITLE_LEAD_MAX_WORDS);
          if (lone || (words.length <= TITLE_LEAD_MAX_WORDS && words.length >= 2)) found.push(e.phrase);
          continue;
        }
        // "now" as the first word names something ("Now playing", a time picker's "Now", a /now page); after another
        // word it is the filler of "Book now", except in "Not now", the standard label for putting a prompt off.
        const at = words.indexOf('now');
        if (e.phrase === 'now' ? at > 0 && words[at - 1] !== 'not' : contains(words, e.phrase)) found.push(e.phrase);
      }
    }
    if (found.length) {
      const list = found.map((p) => `"${p}"`).join(', ');
      out.push(finding(id, code, screen, [x], `${n.role} "${clip(n.text)}": ${list} ${found.length > 1 ? 'add' : 'adds'} length and no meaning`));
    }
  });
  return out;
}
