// slop/empty-claim (codex L28, L29, L30, L32): claims generated copy writes because it has no facts to hand: speed and
// ease promises ("Get started in seconds", "No coding required"), round unsourced numbers ("10x faster", "Trusted by
// 10,000+ teams", "Rated 4.9/5"), vague social proof ("Loved by thousands") and false urgency ("Limited time offer").
// Severity info: "check this claim", never an error (codex/lists/llm_claims.txt). List: llm_claims.txt §L28–L30, §L32;
// §L31 (reassurance, "No credit card required") is left to the network (LABELLING.md K6: KEEP once, ECHO repeats).
//
// Read only in promotional slots: headings, subtitles, eyebrows, body, list items, captions, badges, stats, buttons and
// links. A number counts only when it is literal and round (one or two significant figures: 10,000, 2.5, 4.9, 99.9);
// "12,483 teams" reads as a real count, a range ("2-4x faster") as a measured result, and an interpolated count is data. Urgency in a `dynamic` node (a bound date
// or count) is real. A branch that states its source ("in our benchmarks", "in tests", "according to a 2025 survey",
// "measured on", "source:") is a measured claim: its L28–L30 hits are not reported (urgency, L32, still is).

import { finding } from './finding.js';
import { compileAll, hits, matchTokens, pick } from './codex.js';
import { hasFlag, isScored } from './screen.js';
import { clip, nodeBranches } from './text.js';

export const id = 'slop/empty-claim';
export const code = 'L28-L32';
export const severity = 'info';
export const description = 'Unsupported claim: speed promise, round unsourced number, vague social proof or false urgency ("10x faster", "Loved by thousands").';

const FACT = 'often a plain fact in context';
export const EXCLUDE = {
  // L28
  'in seconds': FACT, 'in minutes': FACT, 'in under a minute': FACT, 'in under 5 minutes': FACT, 'in one click': FACT,
  'with one click': FACT, 'with a single click': FACT, 'zero setup': 'often true of developer tools', 'zero config': 'as above',
  'zero configuration': 'as above', 'works out of the box': 'as above', 'out of the box': 'as above', 'plug and play': 'as above',
  'plug-and-play': 'as above', 'just works': 'the longer "it just works" is kept', instantly: FACT, instant: FACT, 'real-time': FACT,
  'in real time': FACT, 'save time': 'a mild, common benefit line', 'hours of work': FACT, 'hours saved': 'a stat label', '10x faster': 'covered by "{n}x faster"',
  // L29
  '{n}% faster': 'a measured result is common', '{n}% more': 'as above', '{n}% less': 'as above', '{n}% of teams': 'survey results',
  '{n}% of users': 'as above', '{n}% uptime': 'an SLA statement', '{n}+ downloads': 'often a real store count', '{n}+ integrations': 'a product fact',
  '{n}+ countries': 'as above',
  // L30
  'trusted by': 'the eyebrow over a logo wall is the network\'s (R2)', 'trusted by teams': 'usually followed by names',
  'loved by': 'as "trusted by"', 'used by teams at': 'followed by names', 'customers love': 'ordinary English',
  'what our customers say': 'a testimonial heading (the network\'s genre kicker)', 'what people are saying': 'as above',
  'hear from our customers': 'as above', 'award-winning': 'can be true and checkable', 'best-rated': 'as above', 'top-rated': 'as above',
  'highly rated': 'as above', '5-star': 'a rating scale', 'five-star': 'as above', 'number one': 'as above', 'the leading': 'as above',
  "the world's leading": 'as above', "the world's first": 'as above', "the world's most": 'as above',
  'industry-leading': 'in llm_promo_vocab (slop/promo-vocab)', 'market-leading': 'as above',
  // L32
  'while it lasts': 'retail stock', 'while supplies last': 'as above', hurry: 'a game', 'last chance': 'a real deadline',
  'ending soon': 'an auction', 'today only': 'a real sale', 'exclusive access': 'a membership fact', 'early bird': 'a ticket type',
  'spots left': 'a bound count', 'seats left': 'as above',
};
const LIST = compileAll(pick('llm_claims', ['L28', 'L29', 'L30', 'L32'], EXCLUDE));
const ROLES = new Set(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'subtitle', 'eyebrow', 'body', 'list_item', 'caption', 'badge', 'stat', 'button', 'link']);

export const SOURCED = /\b(?:in|from|on|per|by|across) (?:our |internal |independent |third-party |public |recent |the |\d{4} |an? )*(?:benchmarks?|tests?|testing|trials?|stud(?:y|ies)|surveys?|measurements?|experiments?|evaluations?)\b|\baccording to\b|\b(?:benchmarked|measured|tested) (?:on|in|with|against|by)\b|\bsource:/iu;

// One or two significant figures, or a run of nines (99.9).
export function isRound(num) {
  const digits = num.replace(/[.,]/g, '').replace(/^0+/, '');
  const sig = digits.replace(/0+$/, '');
  return sig.length <= 2 || /^9+$/.test(sig);
}

function accept(node, ts, sourced) {
  return (e, m) => {
    if (e.id === 'L32' && hasFlag(node, 'dynamic')) return false;
    if (e.id !== 'L32' && sourced) return false;
    if (m.slots.n) {
      if (!isRound(m.slots.n[0].w)) return false;
      // A range ("2-4x faster", "5–20x less") reads as a measured result, not a slogan.
      const before = ts[ts.indexOf(m.slots.n[0]) - 1];
      if (before && before.k === 'n') return false;
    }
    return true;
  };
}

export function check(screen) {
  const out = [];
  screen.nodes.forEach((n, x) => {
    if (!isScored(n) || !ROLES.has(n.role)) return;
    const found = [];
    let first = null;
    for (const b of nodeBranches(n)) {
      const ts = matchTokens(b.text, b.vw);
      for (const h of hits(LIST, ts, accept(n, ts, SOURCED.test(b.text)))) {
        if (!first) first = h.id;
        if (!found.includes(h.entry)) found.push(h.entry);
      }
    }
    if (found.length) out.push(finding(id, first, screen, [x], `${n.role} "${clip(n.text)}": ${found.map((e) => `"${e}"`).join(', ')}: check this claim, or give the fact behind it`));
  });
  return out;
}
