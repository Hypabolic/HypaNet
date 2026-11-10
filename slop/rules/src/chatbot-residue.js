// slop/chatbot-residue (codex L44): the assistant's own voice, or a prompt echo, shipped in a UI string ("Certainly!
// Here's…", "I hope this helps", "Let me know if you'd like any changes", "As an AI language model"). Severity error:
// it is never meant to ship. List: codex/lists/llm_residue.txt §L44, less EXCLUDE. "Lorem ipsum" and "dolor sit amet"
// sit in that section of the list but are template text, not assistant voice: slop/placeholder-copy reports them (L45,
// as CODEX.md does).
//
// Two narrowings keep an AI product's or a personal site's own copy out: "as an ai" counts only as "As an AI, I…" (an
// AI product may say "As an AI, the assistant can make mistakes."), and the reply phrases in SOFT ("Let me know if
// you'd like…", "Here's a revised…") count only outside running prose (body, subtitle, caption), where a person or a
// planner can write them on purpose.

import { finding } from './finding.js';
import { compileAll, hits, matchTokens, pick } from './codex.js';
import { isScored } from './screen.js';
import { clip, nodeBranches } from './text.js';

export const id = 'slop/chatbot-residue';
export const code = 'L44';
export const severity = 'error';
export const description = 'Chatbot voice or prompt text left in a UI string ("Certainly! Here\'s", "I hope this helps", "As an AI language model").';

// Entries of §L44 that are ordinary, deliberate UI copy too often to report.
export const EXCLUDE = {
  'of course!': 'a friendly reply in a chat or game',
  'absolutely!': 'as above',
  'good question': 'quiz and forum copy',
  "you're right to": 'ordinary English',
  'let me know if': 'a personal site\'s contact line; the longer "let me know if you\'d like" is kept',
  'let me know if you need': 'as above',
  'feel free to': 'filler, not residue (codex ui_padding_phrases)',
  'feel free to reach out': 'as above',
  'would you like to': 'a confirmation dialog\'s question ("Would you like to save changes?")',
  'is there anything else': 'a form question ("Is there anything else we should know?")',
  'happy to help': 'support copy',
  "i'd be happy to": 'a personal site\'s first-person copy',
  "i'm happy to help": 'as above',
  'here is a': 'ordinary lead-in ("Here is a summary of your order")',
  "here's a": 'as above',
  'here are some': 'as above ("Here are some ideas to get started")',
  'below is a': 'as above ("Below is a list of your devices")',
  "i don't have access to": 'a sign-in help link ("I don\'t have access to my email")',
  'based on available information': 'legitimate hedging in data products',
  'not publicly available': 'a fact about a profile or file',
  'more detailed breakdown': 'a finance or analytics link ("See a more detailed breakdown")',
  'in this example': 'documentation and tutorials',
  'for illustrative purposes': 'a shop\'s image disclaimer',
  'sample text': 'a font or style preview',
  'example text': 'as above',
  'add your {x} here': 'a field placeholder ("Add your notes here")',
  'your {x} here': 'a drop zone ("Drop your files here")',
  todo: 'a to-do app\'s own word',
  tbd: 'a status value',
  'lorem ipsum': 'template text: slop/placeholder-copy (L45)',
  'dolor sit amet': 'as above',
};
const LIST = compileAll(pick('llm_residue', ['L44'], EXCLUDE));
export const SOFT = new Set(["let me know if you'd like", "here's an updated", "here's the updated", "here's a revised", 'here is the revised',
  'below is the updated']);
export const PROSE_ROLES = new Set(['body', 'subtitle', 'caption']);
for (const e of SOFT) if (!LIST.some((x) => x.entry === e)) throw new Error(`chatbot-residue: SOFT entry "${e}" is not in §L44`);

function accept(node, ts) {
  return (e, m) => {
    if (SOFT.has(e.entry)) return !PROSE_ROLES.has(node.role);
    if (e.entry === 'as an ai') {
      // Only "As an AI, I…" (or "As an AI I…"): the assistant speaking of itself.
      const [a, b] = [ts[m.to], ts[m.to + 1]];
      const isI = (t) => t && t.k === 'w' && (t.w === 'i' || t.base === 'i');
      return isI(a) || (a && a.k === 'p' && a.w === ',' && isI(b));
    }
    return true;
  };
}

export function check(screen) {
  const out = [];
  screen.nodes.forEach((n, x) => {
    if (!isScored(n)) return;
    const found = [];
    for (const b of nodeBranches(n)) {
      const ts = matchTokens(b.text, b.vw);
      for (const h of hits(LIST, ts, accept(n, ts))) if (!found.includes(h.entry)) found.push(h.entry);
    }
    if (found.length) out.push(finding(id, code, screen, [x], `${n.role} "${clip(n.text)}": ${found.map((e) => `"${e}"`).join(', ')} is assistant text left in shipped copy`));
  });
  return out;
}
