// Fixed word lists for the rule layer (SPEC.md §8). The source lists are short and partly by example, so these are
// written out here; each entry cites the line it rests on, or says it is added. Paths are under slop/research/.
// All entries are folded (lower case, apostrophes removed) and compared word by word.

// slop/vacuous-action (aside R7): a button or link whose whole label is one of these. `roles` limits an entry to
// links where a button with the same words is a platform convention (an overflow "More" menu).
export const VACUOUS_LABELS = [
  { phrase: 'learn more', roles: ['button', 'link'], src: 'anti-slop-guides.md:273 (Mailchimp), :383; local-skills.md:84 (lint.mjs:49)' },
  { phrase: 'read more', roles: ['button', 'link'], src: 'local-skills.md:84 (tasteskill lint.mjs:49)' },
  { phrase: 'find out more', roles: ['button', 'link'], src: 'added: the British form of "learn more"' },
  { phrase: 'discover more', roles: ['button', 'link'], src: 'local-skills.md:84 (lint.mjs:49)' },
  { phrase: 'explore now', roles: ['button', 'link'], src: 'local-skills.md:84 (lint.mjs:49)' },
  { phrase: 'explore more', roles: ['button', 'link'], src: 'added: variant of "discover more" / "explore now"' },
  { phrase: 'click here', roles: ['button', 'link'], src: 'anti-slop-guides.md:241 (GOV.UK), :250 (Apple), :273 (Mailchimp), :283 (NN/g)' },
  { phrase: 'tap here', roles: ['button', 'link'], src: 'added: the touch form of "click here"' },
  { phrase: 'get started', roles: ['button', 'link'], src: 'anti-slop-guides.md:79 (unslop-ui CTA clones), :129 (Hallmark gate 19); local-skills.md:84' },
  { phrase: 'submit', roles: ['button', 'link'], src: 'anti-slop-guides.md:91 (deslop-copy); local-skills.md:84 (lint.mjs:49)' },
  { phrase: 'here', roles: ['link'], src: 'anti-slop-guides.md:241 (GOV.UK: "click here" or "more")' },
  { phrase: 'more', roles: ['link'], src: 'anti-slop-guides.md:241 (GOV.UK)' },
  { phrase: 'this link', roles: ['link'], src: 'added: same failure as "here"' },
  { phrase: 'click', roles: ['button', 'link'], src: 'added: the bare gesture' },
  { phrase: 'tap', roles: ['button', 'link'], src: 'added: the bare gesture' },
];
// Deliberately absent: "OK", "Continue", "Next", "Done", "Cancel", "Close", "Show more", "See more". The first six are
// platform conventions whose meaning the dialog or flow supplies (anti-slop-guides.md:385 rates "Continue" as
// context-dependent); the last two say what they do (expand the list in place).

// A button or link whose label starts with a gesture names the gesture, not the outcome
// (anti-slop-guides.md:143, Impeccable: "not the gesture used to trigger it").
export const GESTURE_LEADS = ['click to', 'tap to', 'click here to', 'tap here to', 'press here to'];

// In any scored role: a gesture pointing at a link the reader cannot see named ("To read the report, click here.").
// Inline links are merged into their paragraph (extract/README.md), so these phrases show up in body text.
export const GESTURE_PHRASES = [
  { phrase: 'click here', src: 'anti-slop-guides.md:382 ("Click here to read the report")' },
  { phrase: 'tap here', src: 'added: touch form' },
  { phrase: 'click this link', src: 'added: same failure' },
  { phrase: 'press here', src: 'added: same failure' },
];

// slop/filler-words (aside R15). anti-slop-guides.md:437 lists "please", "now", "simply", "easily", "your",
// "in order to", "oops"; the other entries come from the sources that line cites.
//   scope `any`: every scored role; `action`: buttons, links, menu items, nav items and tabs only ("Apply now"; in a
//   sentence "now" is often a fact: "Exports now include notes"); `title-lead`: only as the first word of a heading,
//   nav item, tab or menu item of two to four words ("Your Favorites"; in a sentence "your" is ordinary English, and a
//   field label "Your name" is what anti-slop-guides.md:197 recommends over "Type your name").
export const FILLER = [
  { phrase: 'please', scope: 'any', src: 'anti-slop-guides.md:222 (GOV.UK: "please" or "please note"), :437' },
  { phrase: 'simply', scope: 'any', src: 'anti-slop-guides.md:437' },
  { phrase: 'easily', scope: 'any', src: 'anti-slop-guides.md:211 (Microsoft), :437' },
  { phrase: 'quickly', scope: 'any', src: 'anti-slop-guides.md:211 (Microsoft: quite, very, quickly, easily, effectively)' },
  { phrase: 'effectively', scope: 'any', src: 'anti-slop-guides.md:211' },
  { phrase: 'very', scope: 'any', src: 'anti-slop-guides.md:211' },
  { phrase: 'quite', scope: 'any', src: 'anti-slop-guides.md:211' },
  { phrase: 'in order to', scope: 'any', src: 'anti-slop-guides.md:210 (Microsoft), :437' },
  { phrase: 'oops', scope: 'any', src: 'anti-slop-guides.md:253 (Apple), :437' },
  { phrase: 'whoops', scope: 'any', src: 'added: variant of "oops"' },
  { phrase: 'uh oh', scope: 'any', src: 'anti-slop-guides.md:253 (Apple: "uh-oh")' },
  { phrase: 'now', scope: 'action', src: 'anti-slop-guides.md:221 (GOV.UK: "apply" rather than "apply now"), :437' },
  { phrase: 'your', scope: 'title-lead', src: 'anti-slop-guides.md:251 (Apple: "Favorites" over "Your Favorites"), :437' },
  { phrase: 'my', scope: 'title-lead', src: 'anti-slop-guides.md:251 (Apple: "my and your")' },
];
export const ACTION_ROLES = new Set(['button', 'link', 'menu_item', 'nav', 'tab']);
export const TITLE_ROLES = new Set(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'nav', 'tab', 'menu_item', 'empty_title', 'dialog_title']);
export const TITLE_LEAD_MAX_WORDS = 4;
