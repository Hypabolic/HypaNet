// slop/placeholder-copy (codex L45): template identities and filler never replaced ("Acme Inc", "Jane Doe",
// "(555) 123-4567", "123 Main Street, Anytown", "© 2026 Your Company", "[Your name]"). List: codex/lists/llm_residue.txt
// §L45.
//
// A field's own example is a format hint, not residue (LABELLING.md R8): `placeholder` and `input_value` nodes are never
// reported, nor is a branch that introduces an example ("e.g.", "for example", "such as", "example:"), nor an entry
// right after "like", "not" or "instead of" ("Like you@example.com", "Use your own domain, not example.com."). The
// reserved example domains (example.com, .org, .net) are never reported in a `helper` or `tooltip`, where they
// illustrate a format. Names a real person can have ("John Smith", "Jane Smith") count only as a byline: the name opens
// a short node ("Jane Smith, head of sales"; "— John Smith") or sits beside a quotation, so "Photo by John Smith on
// Unsplash" and "Contact Jane Smith, our head of support." are not reported; "John Doe" and "Jane Doe" always count.
// "Lorem ipsum" and "dolor sit amet" come from §L44 of the list (CODEX.md L45 names them as template text).

import { finding } from './finding.js';
import { compileAll, hits, label, matchTokens, pick } from './codex.js';
import { hasFlag, isScored } from './screen.js';
import { clip, nodeBranches } from './text.js';

export const id = 'slop/placeholder-copy';
export const code = 'L45';
export const severity = 'warning';
export const description = 'Template identity or placeholder text left in shipped copy ("Acme Inc", "Jane Doe", "© Your Company").';

// Slot names: reported only in a copyright line or as the whole text of a header link (the wordmark), since "Company
// name" or "Your logo" is an ordinary field label or settings heading.
export const SLOT_NAMES = ['your company', 'your company name', 'company name', 'your brand', 'brand name', 'product name', 'your product', 'your logo', 'logo'];
export const EXCLUDE = {
  ...Object.fromEntries(SLOT_NAMES.map((s) => [s, 'slot name: copyright lines and header wordmarks only (SLOT_NAMES)'])),
  'sarah m.': 'a plausible real testimonial byline',
  'sarah johnson': 'a plausible real name',
  'alex johnson': 'a plausible real name',
  'founder & ceo': 'a real job title',
  'marketing manager at': 'as above',
  'head of growth at': 'as above',
  'city, state': 'a field label',
  'city, country': 'a field label',
  placeholder: 'a design tool\'s own word',
  '{{name}}': 'template syntax in an HTML file is the template, not residue',
  '{{company}}': 'as above',
};
export const LOREM = ['lorem ipsum', 'dolor sit amet'];
const ALL = [...pick('llm_residue', ['L45'], EXCLUDE), ...pick('llm_residue', ['L44']).filter((e) => LOREM.includes(e.entry))];
if (ALL.filter((e) => LOREM.includes(e.entry)).length !== LOREM.length) throw new Error('placeholder-copy: a LOREM entry is not in §L44');
// Names that are also real people's: byline context only.
export const REAL_NAMES = new Set(['john smith', 'jane smith']);
const EXAMPLE_DOMAIN = /\bexample\.(?:com|org|net)\b/u;
const EXAMPLE_ROLES = new Set(['helper', 'tooltip']);
// Bracketed entries ("[your name]") lose their brackets when folded, so they are matched on the raw text instead.
const RAW = ALL.filter((e) => /[[\]{}]/.test(e.entry)).map((e) => e.entry);
const LIST = compileAll(ALL.filter((e) => !RAW.includes(e.entry)));
const SLOTS = new Set(SLOT_NAMES);

const FORMAT_HINT = /\b(?:e\.\s?g\.?|eg\.|for example|such as|example:|for instance)/iu;
const QUOTE = /["“”«»]/u;
const BYLINE_WORDS = 8;
const COPYRIGHT = /©|\(c\)|\bcopyright\b|all rights reserved/iu;
const SKIP_ROLES = new Set(['placeholder', 'input_value']);
const COPY_NOISE = new Set(['c', 'copyright', 'all', 'rights', 'reserved', 'inc', 'ltd', 'llc', 'limited']);

// A hit right after "like", "not" or "instead of" is an example or a contrast, not leftover text.
function introduced(ts, from) {
  const w = (y) => (ts[y] && ts[y].k === 'w' ? ts[y].w : '');
  return w(from - 1) === 'like' || w(from - 1) === 'not' || (w(from - 2) === 'instead' && w(from - 1) === 'of');
}

// A byline: the name opens the branch (after any dash or quote) in a node of at most BYLINE_WORDS words, or a
// neighbouring node, or this one, holds a quotation.
function byline(screen, x, ts, m) {
  const words = ts.filter((t) => t.k !== 'p');
  if (ts.slice(0, m.from).every((t) => t.k === 'p') && words.length <= BYLINE_WORDS) return true;
  return [x - 1, x, x + 1].some((y) => screen.nodes[y] && QUOTE.test(screen.nodes[y].text));
}

function accept(screen, x, n, ts) {
  return (e, m) => {
    if (introduced(ts, m.from)) return false;
    if (EXAMPLE_ROLES.has(n.role) && EXAMPLE_DOMAIN.test(e.entry)) return false;
    if (REAL_NAMES.has(e.entry)) return byline(screen, x, ts, m);
    return true;
  };
}

export function check(screen) {
  const out = [];
  screen.nodes.forEach((n, x) => {
    if (!isScored(n) || SKIP_ROLES.has(n.role)) return;
    const found = [];
    const add = (e) => { if (!found.includes(e)) found.push(e); };
    for (const b of nodeBranches(n)) {
      if (FORMAT_HINT.test(b.text)) continue;
      const ts = matchTokens(b.text, b.vw);
      for (const h of hits(LIST, ts, accept(screen, x, n, ts))) add(h.entry);
      const low = b.text.toLowerCase();
      for (const r of RAW) if (low.includes(r)) add(r);
      // A slot name: what is left of a copyright line, or a header link's whole text.
      const words = label(ts).split(' ').filter((w) => w && !/^\d+$/.test(w));
      if (COPYRIGHT.test(b.text)) {
        const rest = words.filter((w) => !COPY_NOISE.has(w)).join(' ');
        if (SLOTS.has(rest)) add(rest);
      } else if (n.role === 'link' && hasFlag(n, 'in_header') && SLOTS.has(words.join(' '))) add(words.join(' '));
    }
    if (found.length) out.push(finding(id, code, screen, [x], `${n.role} "${clip(n.text)}": ${found.map((e) => `"${e}"`).join(', ')} looks like template text that was never replaced`));
  });
  return out;
}
