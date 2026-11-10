// slop/stock-cta (codex L26, L27): the call to action every generated landing page ends with. L26: a button or link
// whose whole label is a stock CTA beyond the core that slop/vacuous-action already reports ("Start your journey",
// "Explore features", "Let's go", "Claim your spot"). L27: a closing-band headline ("Ready to get started?", "What are
// you waiting for?", "Start building today") in a heading, subtitle or body. List: codex/lists/llm_cta_cliches.txt.

import { finding } from './finding.js';
import { compileAll, hits, label, matchTokens, NOT_PROSE, pick } from './codex.js';
import { VACUOUS_LABELS } from './lists.js';
import { isScored } from './screen.js';
import { clip, nodeBranches } from './text.js';

export const id = 'slop/stock-cta';
export const code = 'L26-L27';
export const severity = 'warning';
export const description = 'Stock call to action ("Start your journey", "Explore features") or CTA-band headline ("Ready to get started?").';

const R7 = 'already reported by slop/vacuous-action (R7)';
const NOW = '"now" is reported by slop/filler-words (R15)';
const REAL = 'names a real next step (a demo, sales, a waitlist, a purchase)';
export const EXCLUDE_LABELS = {
  'get started': R7, 'learn more': R7, 'read more': R7, 'find out more': R7, 'discover more': R7, 'explore now': R7,
  'explore more': R7, submit: R7, 'click here': R7,
  'get started now': NOW, 'start now': NOW, 'begin now': NOW, 'try it now': NOW, 'sign up now': NOW, 'join now': NOW,
  'unlock now': NOW, 'shop now': NOW, 'buy now': NOW, 'order now': NOW, 'download now': NOW, 'install now': NOW,
  continue: 'a platform convention (rules/src/lists.js)', next: 'as above', go: 'as above',
  'get started for free': 'says what it costs', 'get started free': 'as above', 'start for free': 'as above', 'start free': 'as above',
  'start your free trial': 'names the outcome', 'try for free': 'as above', 'try free': 'as above', 'sign up free': 'as above',
  'sign up for free': 'as above', 'join for free': 'as above', 'claim your free trial': 'as above',
  'watch the demo': REAL, 'book a demo': REAL, 'request a demo': REAL, 'schedule a demo': REAL, 'get a demo': REAL,
  'talk to sales': REAL, 'contact sales': REAL, 'get in touch': REAL, 'join the waitlist': REAL, 'join waitlist': REAL,
  'get early access': REAL, 'get access': REAL, 'see how it works': 'often leads to a real walkthrough',
  "let's talk": 'a personal site\'s contact link', "let's chat": 'as above',
};
const BAND = 'reported by slop/llm-construction or slop/empty-claim';
export const EXCLUDE_HEADLINES = {
  "it's time to": 'a real reminder ("It\'s time to renew your plan")',
  'join the community': 'a real community link',
  'start building': 'a developer tool\'s button',
  "it's your turn": 'a game',
  "don't wait": 'ordinary English',
  'join thousands of': BAND, 'join thousands of happy': BAND, "don't miss out": BAND, 'join the revolution': BAND,
  'experience the difference': BAND, 'your journey starts here': BAND,
};
const VACUOUS = new Set(VACUOUS_LABELS.map((e) => e.phrase));
const LABELS = new Set(pick('llm_cta_cliches', ['L26'], EXCLUDE_LABELS).map((e) => label(matchTokens(e.entry))));
for (const l of LABELS) if (VACUOUS.has(l)) throw new Error(`stock-cta label "${l}" is in VACUOUS_LABELS; exclude it`);
// A headline entry matches anywhere in a prose node: "Ready to get started?" needs its question mark, "ready to transform
// your" is a lead.
const HEADLINES = compileAll(pick('llm_cta_cliches', ['L27'], EXCLUDE_HEADLINES));
const ACTIONS = new Set(['button', 'link']);

export function check(screen) {
  const out = [];
  screen.nodes.forEach((n, x) => {
    if (!isScored(n)) return;
    for (const b of nodeBranches(n)) {
      const ts = matchTokens(b.text, b.vw);
      if (ACTIONS.has(n.role)) {
        const l = label(ts);
        if (LABELS.has(l)) {
          out.push(finding(id, 'L26', screen, [x], `${n.role} "${clip(b.text)}" is a stock call to action; name what happens next`));
          return;
        }
        continue;
      }
      if (NOT_PROSE.has(n.role)) continue;
      const h = hits(HEADLINES, ts)[0];
      if (h) {
        out.push(finding(id, 'L27', screen, [x], `${n.role} "${clip(b.text)}" is a stock CTA-band headline ("${h.entry}")`));
        return;
      }
    }
  });
  return out;
}
