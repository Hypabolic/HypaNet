// slop/llm-construction (codex L09–L19): the sentence and headline templates generated copy reuses ("It's not just a
// list, it's a plan", "Whether you're a beginner or a pro", "From X to Y, Z", "Tired of X?", "Say goodbye to X",
// "Note-taking, reimagined", "Welcome to X", "Everything you need to X", "That's it."). List:
// codex/lists/llm_constructions.txt; the finding's code is the codex id of the section the pattern sits in.
//
// Only patterns with few legitimate uses in shipped UI are kept; the rest are listed in EXCLUDE with the reason.
// Constructions are read in prose roles only (not controls, labels, field values, cells or stats, NOT_PROSE), and some
// only in titles (TITLE_ONLY: headings, subtitles, eyebrows, empty-state and dialog titles, badges). In L09–L11 a slot
// holding a number is a real contrast or range ("not included in the 3-seat plan", "from 9 to 5"), and in L11 so is a
// slot holding a date or time word.

import { finding } from './finding.js';
import { compileAll, hits, matchTokens, NOT_PROSE, pick, slotWords, TITLE_ROLES } from './codex.js';
import { hasFlag, isScored } from './screen.js';
import { clip, nodeBranches } from './text.js';

export const id = 'slop/llm-construction';
export const code = 'L09-L19';
export const severity = 'warning';
export const description = 'Stock LLM construction ("not just X, it\'s Y", "Whether you\'re X or Y", "Say goodbye to X", "X, reimagined").';

const SECTIONS = ['L09', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'L17', 'L18', 'L19'];
export const EXCLUDE = {
  // L09
  '{x} rather than {y}': 'ordinary contrast',
  '{x}, not {y}': 'a real correction ("Archived, not deleted") is a consequence (KEEP)',
  'without the {x}': 'ordinary English ("Use it without the app")',
  'not only {x} but {y}': 'body text uses it for real consequences ("not only removes the file but its versions")',
  // L10
  'whether it\'s {x} or {y}': 'a choice ("Choose whether it\'s public or private")',
  // L11
  '^from {x} to {y}': 'a route or a real range ("From London to Paris"); the comma form is kept',
  // L12
  '^{w}, {w} and {w}.$': 'a real three-item list ("Red, green and blue.")',
  '^{w}, {w}, and {w}.$': 'as above',
  // L13
  '^ready to {x}?': 'a confirmation title ("Ready to publish?"); the CTA-band forms are in slop/stock-cta (L27)',
  '^want to {x}?': 'help text ("Want to change your plan? Contact us.")',
  '^looking for {x}?': 'help text ("Looking for the old dashboard?")',
  '^what if {x}?': 'an FAQ question ("What if I miss a payment?")',
  '^need {x}? {y}': 'help text ("Need help? Contact support.")',
  // L14
  'no more {x}': 'a list end ("No more results")',
  'meet {x}': 'an about page ("Meet the team"); "Meet {x}, {y}" is kept',
  '^introducing {x}': 'a changelog heading',
  // L15
  '{x} so you can {y}': 'a real purpose clause',
  '{x} so that you can {y}': 'as above',
  // L16
  '{x}, ensuring {y}': 'often a real consequence',
  '{x}, allowing you to {y}': 'as above',
  '{x}, enabling you to {y}': 'as above',
  '{x}, helping you {y}': 'as above',
  '{x}, giving you {y}': 'as above',
  // L17
  'all in one {w}': 'ordinary English ("Pay all in one go")',
  // L18
  '{x} serves as {y}': 'a real role ("This address serves as your sign-in")',
  '{x} stands as {y}': 'as above',
  '{x} acts as your {y}': 'as above',
  // L19
  '^just {x}.$': 'a timestamp or answer ("Just now.")',
};
// Patterns read only in titles: as a suffix or a slogan they are a headline device; in a sentence they are often plain.
export const TITLE_ONLY = new Set([
  'more than a {x}', 'more than an {x}', 'stop {x}. start {y}', 'less {x}, more {y}', 'say hello to {x}',
  '{x}, reimagined', '{x}, redefined', '{x}, reinvented', '{x}, made simple', '{x}, made easy', '{x}, simplified', '{x} made simple',
  '{x} the way it should be', '{x} done right', '^welcome to {x}', 'the future of {x}', 'the future is {x}', 'everything you need to {x}',
  '{x} and {y} alike',
]);
const NO_NUMBERS = new Set(['L09', 'L10', 'L11']);
const TIME_WORDS = new Set(['am', 'pm', 'noon', 'midnight', 'morning', 'evening', 'night', 'today', 'tomorrow', 'yesterday',
  'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
  'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
  'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec', 'spring', 'summer', 'autumn', 'winter']);

const LIST = compileAll(pick('llm_constructions', SECTIONS, EXCLUDE));
for (const t of TITLE_ONLY) if (!LIST.some((e) => e.entry === t)) throw new Error(`TITLE_ONLY names "${t}", which is not a kept pattern`);

// A greeting in a dialog or a state view is onboarding ("Welcome to your workspace"); on a page it is the hero opener.
const IN_PLACE = ['in_dialog', 'in_state'];
function accept(node) {
  const role = node.role;
  return (e, m) => {
    if (TITLE_ONLY.has(e.entry) && !TITLE_ROLES.has(role)) return false;
    if (e.entry === '^welcome to {x}' && (role === 'dialog_title' || IN_PLACE.some((f) => hasFlag(node, f)))) return false;
    const words = slotWords(m);
    if (NO_NUMBERS.has(e.id) && words.some((t) => t.k === 'n')) return false;
    if (e.id === 'L11' && words.some((t) => TIME_WORDS.has(t.w))) return false;
    return true;
  };
}

export function check(screen) {
  const out = [];
  screen.nodes.forEach((n, x) => {
    if (!isScored(n) || NOT_PROSE.has(n.role)) return;
    const byCode = new Map();
    for (const b of nodeBranches(n)) {
      for (const h of hits(LIST, matchTokens(b.text, b.vw), accept(n))) {
        if (!byCode.has(h.id)) byCode.set(h.id, []);
        if (!byCode.get(h.id).includes(h.entry)) byCode.get(h.id).push(h.entry);
      }
    }
    for (const [c, entries] of [...byCode].sort(([a], [b]) => (a < b ? -1 : 1)))
      out.push(finding(id, c, screen, [x], `${n.role} "${clip(n.text)}" uses the stock construction ${entries.map((e) => `"${e.replace(/^\^|\$$/g, '')}"`).join(', ')}`));
  });
  return out;
}
