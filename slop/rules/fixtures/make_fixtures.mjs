#!/usr/bin/env node
// Build the rule layer's test screens (slop.screen/1) and their expected findings from the compact tables below.
//
//   node slop/rules/fixtures/make_fixtures.mjs           write screens/*.json and expected.json
//   node slop/rules/fixtures/make_fixtures.mjs --check   exit 1 if the files on disk differ from what would be written
//
// Hand-written stand-ins for extractor output: roles, flags and clusters are chosen by hand and each node sits on its
// own line, so `loc` is [line, 7, line, 7 + length]. All copy is invented. U+E000 and U+E001 are the extractor's VAR
// and ALT markers (SPEC.md §4); a node with a VAR marker carries the extractor's `vw` (identifier words after each
// marker), except in `fw_markers_mixed`, which keeps the rules' fallback for screens without it. Each fixture names the
// rules it is a positive or a near-miss negative for; a fixture expects exactly the findings listed, from every rule.
// The fixtures marked "review" come from the adversarial review of the rule layer (2026-11-10); those marked b, j and c
// come from the second review the same day (j: the lead's decisions), with c06–c08 guarding its fixes.

import { createHash } from 'node:crypto';
import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FLAGS } from '../../lint/tokenizer.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const V = '';
const A = '';
const EYEBROW = ['caps', 'tracked', 'small'];

const RC = 'slop/repeated-chrome';
const VA = 'slop/vacuous-action';
const FW = 'slop/filler-words';
const PL = 'slop/placeholder-as-label';
const SL = 'slop/shared-lead-in';
const IX = 'slop/index-numbering';
const ER = 'slop/exact-repeat';

// node: [k, role, text, flags, vw?]; finding: [rule, code, [i…], {condition, target}]
const n = (k, role, text, flags = [], vw = undefined) => [k, role, text, flags, vw];
// The same nodes written cluster by cluster: [[kind, [[role, text, flags?, vw?], …]], …] gives {kinds, nodes}.
const clustered = (list) => ({
  kinds: list.map(([kind]) => kind),
  nodes: list.flatMap(([, ns], k) => ns.map(([role, text, flags = [], vw]) => n(k, role, text, flags, vw))),
});
const f = (rule, code, nodes, extra = {}) => ({ rule, code, nodes, ...extra });
const pos = (rule) => ({ rule, kind: 'positive' });
const neg = (rule) => ({ rule, kind: 'negative' });

// Distinct list rows for the long screens that need two windows.
const TREES = ['Alder', 'Birch', 'Cedar', 'Damson', 'Elder', 'Fir', 'Gean', 'Hazel', 'Ivy', 'Juniper', 'Larch', 'Maple'];
// One-node body clusters of distinct notes, to push a screen past one window.
const notes = (count, from = 0) => Array.from({ length: count }, (_, j) => ['body', [['body', `Notes on harbour item ${String.fromCharCode(97 + ((from + j) % 26))}${from + j}`]]]);
const rows = (k, from, count) => Array.from({ length: count }, (_, j) => n(k, 'list_item', `${TREES[(from + j) % TREES.length]} lane, plot ${from + j + 1}`, ['in_list']));

const FIXTURES = [
  // -- slop/repeated-chrome (R19) ---------------------------------------------------------------------------------
  { name: 'rc_eyebrow_quota_pos', fw: 'html', about: [pos(RC)], kinds: ['header', 'header', 'header', 'header'], nodes: [
    n(0, 'eyebrow', 'Built for small bakeries', [...EYEBROW, 'in_header']),
    n(0, 'h1', 'Bake to order, not to waste', ['in_header']),
    n(0, 'subtitle', 'Customers reserve loaves the night before.', ['section_first']),
    n(1, 'eyebrow', 'Ordering', EYEBROW),
    n(1, 'h2', 'Reservations open at six'),
    n(1, 'body', 'Set a cut-off and the shop closes orders for you.', ['section_first', 'section_last']),
    n(2, 'eyebrow', 'Stock', EYEBROW),
    n(2, 'h2', 'Know what is left by noon'),
    n(2, 'body', 'The counter screen shows each loaf as it sells.', ['section_first', 'section_last']),
    n(3, 'eyebrow', 'Payments', EYEBROW),
    n(3, 'h2', 'Card or cash at collection'),
    n(3, 'body', 'Takings reconcile with the till every evening.', ['section_first', 'section_last']),
  ], expect: [f(RC, 'R19', [0, 3, 6, 9], { condition: 'eyebrow-quota' })] },

  { name: 'rc_eyebrow_quota_neg', fw: 'html', about: [neg(RC)], kinds: ['header', 'header', 'header', 'header', 'header', 'header'], nodes: [
    n(0, 'eyebrow', 'Spring term', EYEBROW),
    n(0, 'h1', 'Evening pottery classes'),
    n(0, 'subtitle', 'Six weeks, one wheel each, all clay included.'),
    n(1, 'h2', 'Who teaches'),
    n(1, 'body', 'Two potters who have run the studio since 2009.'),
    n(2, 'eyebrow', 'Thursdays only', EYEBROW),
    n(2, 'h2', 'Raku firing night'),
    n(2, 'body', 'Glazed pieces come out of the kiln glowing.'),
    n(3, 'h2', 'What to bring'),
    n(3, 'body', 'Old clothes and short nails.'),
    n(4, 'h2', 'Where we are'),
    n(4, 'body', 'Unit 4, Tanner’s Yard, behind the station.'),
    n(5, 'h2', 'Booking'),
    n(5, 'body', 'Places are held for 48 hours after you enquire.'),
  ], expect: [] },

  { name: 'rc_pause_overlay_neg', fw: 'tsx', about: [neg(RC), neg(VA)], kinds: ['header', 'actions'], nodes: [
    n(0, 'eyebrow', 'ROUND SUSPENDED', [...EYEBROW, 'in_header']),
    n(0, 'h1', 'PAUSED', ['caps', 'in_header']),
    n(1, 'button', 'CONTINUE', ['caps']),
    n(1, 'button', 'OPTIONS', ['caps']),
    n(1, 'button', 'QUIT TO TITLE', ['caps']),
  ], expect: [] },

  { name: 'rc_card_eyebrows_neg', fw: 'tsx', about: [neg(RC)], kinds: ['header', 'card', 'card', 'card'], nodes: [
    n(0, 'h2', 'Gear for the long weekend'),
    n(1, 'eyebrow', 'Camping', [...EYEBROW, 'in_card', 'repeated']),
    n(1, 'h3', 'Ridgeline two-person tent', ['in_card', 'repeated']),
    n(1, 'body', '1.9 kg packed, pitches in four minutes.', ['in_card', 'repeated']),
    n(2, 'eyebrow', 'Cooking', [...EYEBROW, 'in_card', 'repeated']),
    n(2, 'h3', 'Folding titanium stove', ['in_card', 'repeated']),
    n(2, 'body', 'Boils a litre in three and a half minutes.', ['in_card', 'repeated']),
    n(3, 'eyebrow', 'Lighting', [...EYEBROW, 'in_card', 'repeated']),
    n(3, 'h3', 'Rechargeable lantern', ['in_card', 'repeated']),
    n(3, 'body', 'Forty hours on the low setting.', ['in_card', 'repeated']),
  ], expect: [] },

  { name: 'rc_decorative_dot_pos', fw: 'tsx', about: [pos(RC)], kinds: ['header', 'body'], nodes: [
    n(0, 'h1', 'Harbour Café', ['title_case']),
    n(0, 'badge', 'Open today', ['small', 'pill', 'dot']),
    n(1, 'body', '● Taking bookings', ['dot']),
    n(1, 'body', 'Breakfast until 11, lunch until 3.'),
  ], expect: [f(RC, 'R19', [1, 2], { condition: 'decorative-dot' })] },

  { name: 'rc_stateful_dot_neg', fw: 'tsx', about: [neg(RC)], kinds: ['header', 'actions'], nodes: [
    n(0, 'h2', 'Printer'),
    n(0, 'body', `Ready ${A} Out of paper`, ['dot', 'stateful', 'conditional']),
    n(0, 'caption', `${V} last job`, ['small', 'dynamic', 'stateful'], [2]),
    n(1, 'button', 'Print test page'),
  ], expect: [] },

  // One live dot per sibling row, each bound to its own row: K2, not a finding.
  { name: 'rc_state_dots_rows_neg', fw: 'tsx', about: [neg(RC)], kinds: ['header', 'list'], nodes: [
    n(0, 'h2', 'Service status'),
    n(1, 'list_item', `Bookings ${V} state`, ['dot', 'stateful', 'in_list'], [1]),
    n(1, 'list_item', `Payments ${V} state`, ['dot', 'stateful', 'in_list'], [1]),
    n(1, 'list_item', `Email receipts ${V} state`, ['dot', 'stateful', 'in_list'], [1]),
  ], expect: [] },

  { name: 'rc_state_dots_pos', fw: 'tsx', about: [pos(RC)], kinds: ['header', 'body'], nodes: [
    n(0, 'h2', 'Sync'),
    n(0, 'badge', `${V} connection state`, ['small', 'dot', 'stateful', 'dynamic'], [2]),
    n(1, 'body', `● ${V} queue state`, ['dot', 'stateful', 'dynamic'], [2]),
    n(1, 'body', 'Changes upload when the van is back in range.'),
  ], expect: [f(RC, 'R19', [1, 2], { condition: 'state-dots' })] },

  // Two live dots on every row are not one per item: all of them count.
  { name: 'rc_state_dots_two_per_row_pos', fw: 'tsx', about: [pos(RC)], kinds: ['header', 'list'], nodes: [
    n(0, 'h2', 'Ovens'),
    n(1, 'list_item', 'Deck oven', ['in_list']),
    n(1, 'caption', `${V} heat state`, ['small', 'dot', 'stateful', 'dynamic', 'in_list'], [2]),
    n(1, 'caption', `${V} door state`, ['small', 'dot', 'stateful', 'dynamic', 'in_list'], [2]),
    n(1, 'list_item', 'Proving cabinet', ['in_list']),
    n(1, 'caption', `${V} heat state`, ['small', 'dot', 'stateful', 'dynamic', 'in_list'], [2]),
    n(1, 'caption', `${V} door state`, ['small', 'dot', 'stateful', 'dynamic', 'in_list'], [2]),
  ], expect: [f(RC, 'R19', [2, 3, 5, 6], { condition: 'state-dots' })] },

  { name: 'rc_state_dots_neg', fw: 'tsx', about: [neg(RC)], kinds: ['header', 'header', 'header'], nodes: [
    n(0, 'h2', 'Bookings'),
    n(0, 'body', `${V} bookings state`, ['dot', 'stateful'], [2]),
    n(1, 'h2', 'Payments'),
    n(1, 'body', `${V} payments state`, ['dot', 'stateful'], [2]),
    n(2, 'h2', 'Email receipts'),
    n(2, 'body', `${V} email state`, ['dot', 'stateful'], [2]),
  ], expect: [] },

  { name: 'rc_arrow_links_pos', fw: 'html', about: [pos(RC)], kinds: ['header', 'body', 'actions'], nodes: [
    n(0, 'h2', 'Help for new tenants'),
    n(1, 'body', 'Most questions are answered in the guides below.'),
    n(2, 'link', 'Moving-in checklist →', ['arrow_tail']),
    n(2, 'link', 'Paying the rent', ['icon', 'arrow_tail']),
    n(2, 'link', 'Reporting a repair ›', ['arrow_tail']),
  ], expect: [f(RC, 'R19', [2, 3, 4], { condition: 'arrow-links' })] },

  // ↗ marks an external link (it informs), » and a › glued to the word are not arrow suffixes, even when flagged.
  { name: 'rc_arrow_external_neg', fw: 'html', about: [neg(RC)], kinds: ['header', 'actions'], nodes: [
    n(0, 'h2', 'Further reading'),
    n(1, 'link', 'Council recycling rules ↗', ['arrow_tail']),
    n(1, 'link', 'Bin collection calendar ↗', ['arrow_tail']),
    n(1, 'link', 'All guides »', ['arrow_tail']),
    n(1, 'link', 'Older posts›', ['arrow_tail']),
  ], expect: [] },

  { name: 'rc_arrow_links_neg', fw: 'html', about: [neg(RC)], kinds: ['header', 'actions', 'header'], nodes: [
    n(0, 'h2', 'Opening times'),
    n(1, 'link', 'Bank holiday hours →', ['arrow_tail']),
    n(1, 'link', 'Christmas closures'),
    n(1, 'link', 'Parking at the depot →', ['arrow_tail']),
    n(2, 'h2', 'Contact'),
    n(2, 'link', 'Write to the depot manager →', ['arrow_tail']),
  ], expect: [] },

  { name: 'rc_sibling_badge_pos', fw: 'tsx', about: [pos(RC)], kinds: ['header', 'card', 'card', 'card', 'card'], nodes: [
    n(0, 'h2', 'What changed this month'),
    n(1, 'badge', 'New', ['small', 'pill', 'in_card', 'repeated']),
    n(1, 'h3', 'Split bills by item', ['in_card', 'repeated']),
    n(2, 'badge', 'New', ['small', 'pill', 'in_card', 'repeated']),
    n(2, 'h3', 'Export for the accountant', ['in_card', 'repeated']),
    n(3, 'badge', 'New', ['small', 'pill', 'in_card', 'repeated']),
    n(3, 'h3', 'Recurring invoices', ['in_card', 'repeated']),
    n(4, 'h3', 'Dark theme', ['in_card']),
  ], expect: [f(RC, 'R19', [1, 3, 5], { condition: 'sibling-device' })] },

  { name: 'rc_sibling_badge_neg', fw: 'tsx', about: [neg(RC)], kinds: ['header', 'card', 'card', 'card', 'card'], nodes: [
    n(0, 'h2', 'Tonight’s specials'),
    n(1, 'badge', 'Vegan', ['small', 'pill', 'in_card']),
    n(1, 'h3', 'Roast squash and lentils', ['in_card']),
    n(2, 'badge', 'Gluten free', ['small', 'pill', 'in_card']),
    n(2, 'h3', 'Grilled hake', ['in_card']),
    n(3, 'badge', 'Spicy', ['small', 'pill', 'in_card']),
    n(3, 'h3', 'Lamb shoulder with harissa', ['in_card']),
    n(4, 'badge', 'Chef’s pick', ['small', 'pill', 'in_card']),
    n(4, 'h3', 'Treacle tart', ['in_card']),
  ], expect: [] },

  { name: 'rc_sibling_arrow_rows_pos', fw: 'tsx', about: [pos(RC), neg(SL)], kinds: ['header', 'list', 'body'], nodes: [
    n(0, 'h2', 'Recent statements'),
    n(1, 'list_item', 'March 2026 statement', ['in_list']),
    n(1, 'link', 'Download PDF →', ['in_list', 'arrow_tail']),
    n(1, 'list_item', 'February 2026 statement', ['in_list']),
    n(1, 'link', 'Download PDF →', ['in_list', 'arrow_tail']),
    n(1, 'list_item', 'January 2026 statement', ['in_list']),
    n(1, 'link', 'Download PDF →', ['in_list', 'arrow_tail']),
    n(2, 'link', 'Contact the billing team'),
  ], expect: [f(RC, 'R19', [2, 4, 6], { condition: 'sibling-device' })] },

  { name: 'rc_sibling_stateful_badge_neg', fw: 'tsx', about: [neg(RC), neg(SL)], kinds: ['header', 'card', 'card', 'card'], nodes: [
    n(0, 'h2', 'Invoices'),
    n(1, 'h3', 'Invoice 1041', ['in_card', 'repeated']),
    n(1, 'badge', `Paid ${A} Overdue`, ['small', 'pill', 'stateful', 'conditional', 'in_card', 'repeated']),
    n(2, 'h3', 'Invoice 1042', ['in_card', 'repeated']),
    n(2, 'badge', `Paid ${A} Overdue`, ['small', 'pill', 'stateful', 'conditional', 'in_card', 'repeated']),
    n(3, 'h3', 'Invoice 1043', ['in_card', 'repeated']),
    n(3, 'badge', `Paid ${A} Overdue`, ['small', 'pill', 'stateful', 'conditional', 'in_card', 'repeated']),
  ], expect: [] },

  // -- slop/vacuous-action (R7) -----------------------------------------------------------------------------------
  { name: 'va_stock_labels_pos', fw: 'tsx', about: [pos(VA)], kinds: ['header', 'actions', 'body'], nodes: [
    n(0, 'h2', 'Recipe boxes delivered weekly'),
    n(0, 'subtitle', 'Pick four meals by Thursday for Monday delivery.'),
    n(1, 'button', 'Get started →', ['arrow_tail']),
    n(1, 'link', 'Learn more'),
    n(2, 'h3', 'Our growers'),
    n(2, 'link', 'Read more…'),
  ], expect: [f(VA, 'R7', [2]), f(VA, 'R7', [3]), f(VA, 'R7', [5])] },

  { name: 'va_click_here_pos', fw: 'html', about: [pos(VA)], kinds: ['header', 'body'], nodes: [
    n(0, 'h2', 'Change plan'),
    n(1, 'body', 'To move to the yearly plan, click here.'),
    n(1, 'link', 'here'),
    n(1, 'body', 'Tap here to see what each plan includes.'),
  ], expect: [f(VA, 'R7', [1]), f(VA, 'R7', [2]), f(VA, 'R7', [3])] },

  { name: 'va_gesture_lead_pos', fw: 'tsx', about: [pos(VA)], kinds: ['state', 'state'], nodes: [
    n(0, 'empty_title', 'No photos yet', ['in_state']),
    n(0, 'empty_body', 'Photos added to a listing appear here.', ['in_state']),
    n(0, 'button', 'Tap to add photos', ['in_state']),
    n(1, 'error', 'The upload stopped halfway.', ['in_state']),
    n(1, 'button', 'Click to retry', ['in_state']),
  ], expect: [f(VA, 'R7', [2]), f(VA, 'R7', [4])] },

  { name: 'va_specific_neg', fw: 'tsx', about: [neg(VA)], kinds: ['state', 'actions'], nodes: [
    n(0, 'dialog_title', 'Leave without saving?', ['in_dialog']),
    n(0, 'dialog_body', 'Changes to this draft will be lost.', ['in_dialog']),
    n(0, 'button', 'Leave', ['in_dialog']),
    n(0, 'button', 'Keep editing', ['in_dialog']),
    n(1, 'link', 'Learn more about refunds'),
    n(1, 'button', 'OK'),
    n(1, 'button', 'Continue'),
    n(1, 'button', 'Show more'),
    n(1, 'button', 'More'),
    n(1, 'body', 'Click the map to drop a pin.'),
  ], expect: [] },

  { name: 'va_alt_branch_pos', fw: 'tsx', about: [pos(VA)], kinds: ['header', 'control', 'actions'], nodes: [
    n(0, 'h2', 'Send feedback', ['in_form']),
    n(1, 'label', 'What went wrong?', ['in_form', 'in_control']),
    n(1, 'helper', 'We read every message within a day.', ['small', 'in_form', 'in_control']),
    n(2, 'button', `Sending… ${A} Submit`, ['conditional', 'in_form']),
  ], expect: [f(VA, 'R7', [3])] },

  // -- slop/filler-words (R15) ------------------------------------------------------------------------------------
  { name: 'fw_please_oops_pos', fw: 'tsx', about: [pos(FW)], kinds: ['control', 'state'], nodes: [
    n(0, 'label', 'Postcode', ['in_form', 'in_control']),
    n(0, 'error', 'Please enter a postcode', ['small', 'in_form', 'in_control']),
    n(1, 'toast', 'Oops, that didn’t save. Try again.', ['in_state']),
  ], expect: [f(FW, 'R15', [1]), f(FW, 'R15', [2])] },

  { name: 'fw_now_action_pos', fw: 'tsx', about: [pos(FW)], kinds: ['header', 'actions'], nodes: [
    n(0, 'h2', 'Autumn sale'),
    n(0, 'subtitle', 'Coats and boots at half price until Sunday.'),
    n(1, 'button', 'Shop now'),
    n(1, 'link', 'Book a fitting now'),
  ], expect: [f(FW, 'R15', [2]), f(FW, 'R15', [3])] },

  { name: 'fw_word_boundaries_neg', fw: 'tsx', about: [neg(FW)], kinds: ['header', 'actions'], nodes: [
    n(0, 'h2', 'Now showing'),
    n(0, 'body', 'Exports now include notes and tags.'),
    n(0, 'body', 'Members are pleased to host the snowdrop walk.'),
    n(1, 'button', 'Know your rights'),
    n(1, 'button', 'Nowcast'),
    n(1, 'nav', 'Your', ['in_nav']),
  ], expect: [] },

  { name: 'fw_your_title_pos', fw: 'swift', about: [pos(FW)], kinds: ['list', 'header'], nodes: [
    n(0, 'tab', 'Your Favourites', ['title_case', 'in_nav']),
    n(0, 'tab', 'My Library', ['title_case', 'in_nav']),
    n(0, 'tab', 'Downloads', ['in_nav']),
    n(1, 'h1', 'Your reading list'),
  ], expect: [f(FW, 'R15', [0]), f(FW, 'R15', [1]), f(FW, 'R15', [3])] },

  { name: 'fw_your_neg', fw: 'html', about: [neg(FW)], kinds: ['header', 'control', 'body', 'header'], nodes: [
    n(0, 'h2', 'Plans for your team'),
    n(1, 'label', 'Your name', ['in_form', 'in_control']),
    n(2, 'body', 'Your order ships on Monday.'),
    n(3, 'h2', 'Your account, your data and every setting you can change'),
  ], expect: [] },

  { name: 'fw_adverbs_pos', fw: 'tsx', about: [pos(FW), neg(VA)], kinds: ['header', 'body'], nodes: [
    n(0, 'h2', 'Shared folders'),
    n(0, 'subtitle', 'Simply drag files here to easily share them.'),
    n(1, 'body', 'Setup is very quick and quite painless.'),
    n(1, 'body', 'Move projects in order to keep the archive tidy.'),
  ], expect: [f(FW, 'R15', [1]), f(FW, 'R15', [2]), f(FW, 'R15', [3])] },

  { name: 'fw_markers_mixed', fw: 'tsx', about: [pos(FW), neg(FW)], kinds: ['header', 'actions'], nodes: [
    n(0, 'h2', `${V} your name`, ['dynamic']),
    n(0, 'body', `Hello ${V} first name, please confirm the email address.`, ['dynamic']),
    n(1, 'button', `${V} now playing`, ['dynamic']),
    n(1, 'button', 'Play next'),
  ], expect: [f(FW, 'R15', [1])] },

  // -- slop/placeholder-as-label (R8b) ----------------------------------------------------------------------------
  { name: 'pl_no_label_pos', fw: 'tsx', about: [pos(PL)], kinds: ['header', 'control', 'control'], nodes: [
    n(0, 'h1', 'Recipes'),
    n(1, 'aria', 'Search recipes', ['in_control']),
    n(1, 'placeholder', 'Search recipes…', ['in_control']),
    n(2, 'label', 'Email', ['in_form', 'in_control', 'sr_only']),
    n(2, 'placeholder', 'Email address', ['in_form', 'in_control']),
  ], expect: [f(PL, 'R8b', [2]), f(PL, 'R8b', [4])] },

  { name: 'pl_labelled_neg', fw: 'html', about: [neg(PL)], kinds: ['control', 'control', 'body'], nodes: [
    n(0, 'label', 'Postcode', ['in_form', 'in_control']),
    n(0, 'placeholder', 'e.g. EH1 1YZ', ['in_form', 'in_control']),
    n(1, 'label', 'Card number', ['in_form', 'in_control']),
    n(1, 'helper', 'The long number on the front', ['small', 'in_form', 'in_control']),
    n(1, 'placeholder', '1234 5678 9012 3456', ['in_form', 'in_control']),
    n(2, 'label', 'Notes for the driver', ['in_form']),
    n(2, 'placeholder', 'Leave by the side gate', ['in_form']),
  ], expect: [] },

  { name: 'pl_body_cluster_pos', fw: 'html', about: [pos(PL)], kinds: ['body'], nodes: [
    n(0, 'h2', 'Join the newsletter'),
    n(0, 'placeholder', 'you@example.com'),
    n(0, 'button', 'Subscribe'),
  ], expect: [f(PL, 'R8b', [1])] },

  // -- slop/shared-lead-in (R10) ----------------------------------------------------------------------------------
  { name: 'sl_rows_pos', fw: 'tsx', about: [pos(SL)], kinds: ['header', 'list'], nodes: [
    n(0, 'h2', 'Notifications'),
    n(1, 'list_item', 'Email me when a booking is made', ['in_list']),
    n(1, 'list_item', 'Email me when a booking is cancelled', ['in_list']),
    n(1, 'list_item', 'Email me when a review is posted', ['in_list']),
  ], expect: [f(SL, 'R10', [1, 2, 3])] },

  { name: 'sl_rows_titled_pos', fw: 'html', about: [pos(SL)], kinds: ['list'], nodes: [
    n(0, 'label', 'Before you travel', ['in_list']),
    n(0, 'list_item', 'Check that the ferry is running', ['in_list']),
    n(0, 'list_item', 'Check that the road is open', ['in_list']),
    n(0, 'list_item', 'Check that the hut has space', ['in_list']),
  ], expect: [f(SL, 'R10', [1, 2, 3])] },

  { name: 'sl_cards_pos', fw: 'tsx', about: [pos(SL), neg(FW)], kinds: ['header', 'card', 'card', 'card'], nodes: [
    n(0, 'h2', 'One place for the whole shop'),
    n(1, 'h3', 'Track your stock', ['in_card', 'repeated']),
    n(1, 'body', 'Counts update as each sale rings through.', ['in_card', 'repeated']),
    n(2, 'h3', 'Track your orders', ['in_card', 'repeated']),
    n(2, 'body', 'Suppliers confirm delivery dates in the app.', ['in_card', 'repeated']),
    n(3, 'h3', 'Track your staff rota', ['in_card', 'repeated']),
    n(3, 'body', 'Swap shifts without a group chat.', ['in_card', 'repeated']),
  ], expect: [f(SL, 'R10', [1, 3, 5])] },

  { name: 'sl_field_pair_pos', fw: 'swift', about: [pos(SL)], kinds: ['header', 'control', 'control'], nodes: [
    n(0, 'h2', 'Alerts', ['in_form']),
    n(1, 'label', 'Notify me when someone replies', ['in_form', 'in_control']),
    n(2, 'label', 'Notify me when someone follows me', ['in_form', 'in_control']),
  ], expect: [f(SL, 'R10', [1, 2])] },

  { name: 'sl_parallel_neg', fw: 'tsx', about: [neg(SL)], kinds: ['list', 'list', 'control', 'control'], nodes: [
    n(0, 'list_item', 'Add photo', ['in_list']),
    n(0, 'list_item', 'Add video', ['in_list']),
    n(0, 'list_item', 'Add note', ['in_list']),
    n(1, 'list_item', 'How do I change my plan?', ['in_list']),
    n(1, 'list_item', 'How do I cancel?', ['in_list']),
    n(1, 'list_item', 'How do I get a receipt?', ['in_list']),
    n(2, 'label', 'Email address', ['in_form', 'in_control']),
    n(3, 'label', 'Email format', ['in_form', 'in_control']),
  ], expect: [] },

  { name: 'sl_whole_and_cells_neg', fw: 'html', about: [neg(SL)], kinds: ['list', 'list'], nodes: [
    n(0, 'cell', 'Paid on 3 May', ['in_table', 'in_list']),
    n(0, 'cell', 'Paid on 9 May', ['in_table', 'in_list']),
    n(0, 'cell', 'Paid on 14 May', ['in_table', 'in_list']),
    n(1, 'list_item', 'Track your stock', ['in_list']),
    n(1, 'list_item', 'Track your stock levels', ['in_list']),
    n(1, 'list_item', 'Track your stock weekly', ['in_list']),
  ], expect: [] },

  // -- slop/index-numbering (R12) ---------------------------------------------------------------------------------
  { name: 'ix_padded_cards_pos', fw: 'tsx', about: [pos(IX), pos(RC)], kinds: ['header', 'card', 'card', 'card'], nodes: [
    n(0, 'h2', 'How the co-op works'),
    n(1, 'caption', '01 /', ['small', 'mono', 'in_card', 'repeated']),
    n(1, 'h3', 'Join for a pound', ['in_card', 'repeated']),
    n(2, 'caption', '02 /', ['small', 'mono', 'in_card', 'repeated']),
    n(2, 'h3', 'Vote at the spring meeting', ['in_card', 'repeated']),
    n(3, 'caption', '03 /', ['small', 'mono', 'in_card', 'repeated']),
    n(3, 'h3', 'Share the surplus', ['in_card', 'repeated']),
  ], expect: [f(RC, 'R19', [1, 3, 5], { condition: 'sibling-device' }), f(IX, 'R12', [1, 3, 5])] },

  { name: 'ix_step_leads_pos', fw: 'tsx', about: [pos(IX), neg(SL)], kinds: ['header', 'list'], nodes: [
    n(0, 'h2', 'Getting paid'),
    n(1, 'list_item', 'Step 1: Connect the bank', ['in_list']),
    n(1, 'list_item', 'Step 2: Send the first invoice', ['in_list']),
    n(1, 'list_item', 'Step 3: Chase late payers automatically', ['in_list']),
  ], expect: [f(RC, 'R19', [1, 2, 3], { condition: 'sibling-device' }), f(IX, 'R12', [1, 2, 3])] },

  { name: 'ix_lone_step_neg', fw: 'html', about: [neg(IX)], kinds: ['header', 'control', 'control', 'actions'], nodes: [
    n(0, 'caption', 'Step 2 of 4', ['small']),
    n(0, 'h1', 'Delivery address'),
    n(1, 'label', 'Address line 1', ['in_form', 'in_control']),
    n(2, 'label', 'Town or city', ['in_form', 'in_control']),
    n(3, 'button', 'Continue', ['in_form']),
  ], expect: [] },

  { name: 'ix_referenced_neg', fw: 'tsx', about: [neg(IX), neg(RC)], kinds: ['header', 'card', 'card', 'card', 'body'], nodes: [
    n(0, 'h2', 'Moving banks'),
    n(1, 'eyebrow', 'Step 1', [...EYEBROW, 'in_card', 'repeated']),
    n(1, 'h3', 'Export from the old bank', ['in_card', 'repeated']),
    n(2, 'eyebrow', 'Step 2', [...EYEBROW, 'in_card', 'repeated']),
    n(2, 'h3', 'Wait for confirmation', ['in_card', 'repeated']),
    n(3, 'eyebrow', 'Step 3', [...EYEBROW, 'in_card', 'repeated']),
    n(3, 'h3', 'Import the file', ['in_card', 'repeated']),
    n(4, 'body', 'If the export fails, go back to step 1 and pick CSV.'),
  ], expect: [] },

  { name: 'ix_plain_pos', fw: 'tsx', about: [pos(IX)], kinds: ['header', 'card', 'card', 'card'], nodes: [
    n(0, 'h2', 'Three ways to give'),
    n(1, 'caption', '1', ['small', 'in_card', 'repeated']),
    n(1, 'h3', 'Monthly', ['in_card', 'repeated']),
    n(2, 'caption', '2', ['small', 'in_card', 'repeated']),
    n(2, 'h3', 'One-off', ['in_card', 'repeated']),
    n(3, 'caption', '3', ['small', 'in_card', 'repeated']),
    n(3, 'h3', 'In a will', ['in_card', 'repeated']),
  ], expect: [f(RC, 'R19', [1, 3, 5], { condition: 'sibling-device' }), f(IX, 'R12', [1, 3, 5])] },

  { name: 'ix_plain_neg', fw: 'tsx', about: [neg(IX)], kinds: ['header', 'body', 'list', 'body'], nodes: [
    n(0, 'h2', 'Support desk'),
    n(1, 'stat', '1'),
    n(1, 'caption', 'open ticket', ['small']),
    n(1, 'stat', '2'),
    n(1, 'caption', 'agents online', ['small']),
    n(1, 'stat', '3'),
    n(1, 'caption', 'queues', ['small']),
    n(2, 'link', '1', ['in_nav', 'in_list']),
    n(2, 'link', '2', ['in_nav', 'in_list']),
    n(2, 'link', '3', ['in_nav', 'in_list']),
    n(3, 'caption', '2', ['small']),
    n(3, 'body', 'Seats left'),
    n(3, 'caption', '3', ['small']),
    n(3, 'body', 'Days to go'),
  ], expect: [] },

  { name: 'ix_fraction_pos', fw: 'tsx', about: [pos(IX), pos(RC)], kinds: ['header', 'card', 'card', 'card', 'card'], nodes: [
    n(0, 'h2', 'Studio tour'),
    n(1, 'caption', '1 / 4', ['small', 'in_card', 'repeated']),
    n(1, 'alt', 'Kiln room at dusk', ['in_card', 'repeated']),
    n(2, 'caption', '2 / 4', ['small', 'in_card', 'repeated']),
    n(2, 'alt', 'Glaze shelves', ['in_card', 'repeated']),
    n(3, 'caption', '3 / 4', ['small', 'in_card', 'repeated']),
    n(3, 'alt', 'Wheel corner', ['in_card', 'repeated']),
    n(4, 'caption', '4 / 4', ['small', 'in_card', 'repeated']),
    n(4, 'alt', 'Drying racks', ['in_card', 'repeated']),
  ], expect: [f(RC, 'R19', [1, 3, 5, 7], { condition: 'sibling-device' }), f(IX, 'R12', [1, 3, 5, 7])] },

  // -- slop/exact-repeat (R14, R9) --------------------------------------------------------------------------------
  { name: 'er_cta_pos', fw: 'tsx', about: [pos(ER)], kinds: ['header', 'list', 'list', 'header'], nodes: [
    n(0, 'h1', 'Rota planning for small kitchens'),
    n(0, 'subtitle', 'No card needed for the first month'),
    n(0, 'button', 'Start a free month'),
    ...rows(1, 0, 35),
    ...rows(2, 35, 35),
    n(3, 'h2', 'Rota planning for small kitchens'),
    n(3, 'subtitle', 'No card needed for the first month'),
    n(3, 'button', 'Start a free month'),
  ], expect: [f(ER, 'R9', [73], { target: 0 }), f(ER, 'R14', [74], { target: 1 }), f(ER, 'R14', [75], { target: 2 })] },

  { name: 'er_same_window_neg', fw: 'tsx', about: [neg(ER)], kinds: ['header', 'body'], nodes: [
    n(0, 'h2', 'Gift cards'),
    n(0, 'subtitle', 'Spend them in any of the three shops.'),
    n(1, 'body', 'Spend them in any of the three shops.'),
  ], expect: [] },

  { name: 'er_exempt_neg', fw: 'tsx', about: [neg(ER)],
    kinds: ['list', 'control', 'card', 'card', 'card', 'body', 'list', 'list', 'list', 'control', 'card', 'card', 'card', 'body'], nodes: [
      n(0, 'nav', 'Menu', ['in_nav', 'in_list', 'in_header']),
      n(0, 'nav', 'Opening times', ['in_nav', 'in_list', 'in_header']),
      n(0, 'nav', 'Private dining', ['in_nav', 'in_list', 'in_header']),
      n(1, 'label', 'Postcode', ['in_form', 'in_control']),
      n(1, 'placeholder', 'e.g. G1 1AA', ['in_form', 'in_control']),
      n(2, 'h3', 'Window table', ['in_card', 'repeated']),
      n(2, 'button', 'Reserve this table', ['in_card', 'repeated']),
      n(3, 'h3', 'Garden table', ['in_card', 'repeated']),
      n(3, 'button', 'Reserve this table', ['in_card', 'repeated']),
      n(4, 'h3', 'Counter seats', ['in_card', 'repeated']),
      n(4, 'button', 'Reserve this table', ['in_card', 'repeated']),
      n(5, 'body', 'Kitchen closes at ten tonight', ['conditional']),
      ...rows(6, 0, 35),
      ...rows(7, 35, 35),
      n(8, 'nav', 'Opening times', ['in_nav', 'in_list', 'in_footer']),
      n(8, 'nav', 'Private dining', ['in_nav', 'in_list', 'in_footer']),
      n(9, 'label', 'Postcode', ['in_form', 'in_control']),
      n(9, 'placeholder', 'e.g. G1 1AA', ['in_form', 'in_control']),
      n(10, 'h3', 'Booth', ['in_card', 'repeated']),
      n(10, 'button', 'Reserve this table', ['in_card', 'repeated']),
      n(11, 'h3', 'Chef’s counter', ['in_card', 'repeated']),
      n(11, 'button', 'Reserve this table', ['in_card', 'repeated']),
      n(12, 'h3', 'Terrace', ['in_card', 'repeated']),
      n(12, 'button', 'Reserve this table', ['in_card', 'repeated']),
      n(13, 'body', 'Kitchen closes at ten tonight', ['conditional']),
    ], expect: [] },

  { name: 'er_rank_pos', fw: 'tsx', about: [pos(ER)], kinds: ['header', 'list', 'list', 'body'], nodes: [
    n(0, 'h2', 'Rooms'),
    n(0, 'subtitle', 'Breakfast is served from seven until ten.'),
    ...rows(1, 0, 35),
    ...rows(2, 35, 35),
    n(3, 'body', 'Breakfast is served from seven until ten.'),
  ], expect: [f(ER, 'R14', [1], { target: 72 })] },

  { name: 'er_footer_links_neg', fw: 'tsx', about: [neg(ER)], ...clustered([
    ['header', [['h1', 'Allotment waiting list'], ['link', 'Read the tenancy rules']]],
    ...notes(70),
    ['body', [['link', 'Read the tenancy rules', ['in_footer']], ['link', 'Contact the site warden', ['in_footer']]]],
  ]), expect: [] },

  // -- identifier words: the extractor's `vw` ---------------------------------------------------------------------
  // "very", "please" and "now" follow a one-word identifier; the four-word fallback would swallow them (review a03).
  { name: 'fw_vw_var_masks_pos', fw: 'tsx', about: [pos(FW)], ...clustered([
    ['header', [['h2', 'Uploads']]],
    ['body', [['body', `${V} count files uploaded very quickly`, ['dynamic'], [1]],
              ['body', `${V} name please check your inbox`, ['dynamic'], [1]]]],
    ['actions', [['button', `Pay ${V} amount now`, ['dynamic'], [1]]]],
  ]), expect: [f(FW, 'R15', [1]), f(FW, 'R15', [2]), f(FW, 'R15', [3])] },

  // Each branch of a conditional reads its own slice of `vw`: "now" in the second branch is copy.
  { name: 'fw_vw_branches_pos', fw: 'tsx', about: [pos(FW)], ...clustered([
    ['header', [['h2', 'Checkout']]],
    ['actions', [['button', `Join ${V} team name ${A} Pay ${V} amount now`, ['dynamic', 'conditional'], [2, 1]]]],
  ]), expect: [f(FW, 'R15', [1])] },

  // A five-word identifier ends in "please": the identifier's words, not copy (the fallback would flag it).
  { name: 'fw_vw_long_ident_neg', fw: 'tsx', about: [neg(FW)], ...clustered([
    ['header', [['h2', 'Upload failed']]],
    ['body', [['body', `${V} upload failed try again please`, ['dynamic'], [5]]]],
  ]), expect: [] },

  // -- from the adversarial review (screens a01–a15, a03 above) ---------------------------------------------------
  // a01: a rounded-full "Add to basket" button on every card is a control, not a pill device.
  { name: 'rc_pill_button_cards_neg', fw: 'tsx', about: [neg(RC)], ...clustered([
    ['header', [['h2', 'Kitchen kit']]],
    ...[['Teapot', 'Cast iron, 1.2 litres'], ['Kettle', 'Copper, whistling'], ['Toaster', 'Four slots, long']].map(([t, b]) =>
      ['card', [['h3', t, ['in_card', 'repeated']], ['body', b, ['in_card', 'repeated']], ['button', 'Add to basket', ['pill', 'in_card', 'repeated']]]]),
  ]), expect: [] },

  // a02: a <label for> sits in a body cluster and its input's placeholder in a control cluster of its own.
  { name: 'pl_label_for_input_pos', fw: 'tsx', about: [pos(PL), neg(PL)], ...clustered([
    ['header', [['h2', 'Delivery address']]],
    ['body', [['label', 'Email', ['in_form']]]],
    ['control', [['placeholder', 'name@example.com', ['in_form', 'in_control']]]],
    ['body', [['label', 'Postcode', ['in_form']], ['helper', 'For example, SW1A 1AA', ['in_form']]]],
    ['control', [['placeholder', 'SW1A 1AA', ['in_form', 'in_control']]]],
    ['control', [['placeholder', 'Search addresses', ['in_form', 'in_control']]]],
  ]), expect: [f(PL, 'R8b', [6])] },

  // a04: a wizard shows one "Step N of 3" heading at a time (K3).
  { name: 'ix_wizard_conditional_neg', fw: 'tsx', about: [neg(IX)], ...clustered([
    ['header', [['h2', 'Step 1 of 3', ['conditional']], ['subtitle', 'Tell us about the shop', ['conditional']]]],
    ['header', [['h2', 'Step 2 of 3', ['conditional']], ['subtitle', 'Connect a payment account', ['conditional']]]],
    ['header', [['h2', 'Step 3 of 3', ['conditional']], ['subtitle', 'Choose a shipping zone', ['conditional']]]],
    ['actions', [['button', 'Continue']]],
  ]), expect: [] },

  // a05: times, dates and recipe fractions are values, not index devices.
  { name: 'ix_times_dates_fractions_neg', fw: 'tsx', about: [neg(IX), neg(RC)], ...clustered([
    ['header', [['h2', 'Saturday programme']]],
    ['list', [['caption', '08:00', ['in_list']], ['list_item', 'Doors open', ['in_list']],
              ['caption', '09:00', ['in_list']], ['list_item', 'Opening talk', ['in_list']],
              ['caption', '10:30', ['in_list']], ['list_item', 'Workshops', ['in_list']]]],
    ['header', [['h2', 'Past events']]],
    ['body', [['caption', '01.03'], ['body', 'Spring fair'], ['caption', '02.03'], ['body', 'Seed swap']]],
    ['header', [['h2', 'Pancakes']]],
    ['body', [['other', '1/2'], ['body', 'cup of milk'], ['other', '2/3'], ['body', 'cup of flour']]],
  ]), expect: [] },

  // a06: the page's "Delete playlist" button and a dialog titled "Delete playlist?" in another window (K11).
  { name: 'er_k11_cross_window_neg', fw: 'tsx', about: [neg(ER)], ...clustered([
    ['header', [['h1', 'Road trip mix']]],
    ['actions', [['button', 'Delete playlist']]],
    ...notes(70),
    ['state', [['dialog_title', 'Delete playlist?', ['in_dialog', 'in_state']],
               ['dialog_body', 'Its 40 tracks stay in your library.', ['in_dialog', 'in_state']],
               ['button', 'Cancel', ['in_dialog', 'in_state']]]],
  ]), expect: [] },

  // a07: the same body in two different dialogs, never visible together (LABELLING.md §2a).
  { name: 'er_two_dialogs_neg', fw: 'tsx', about: [neg(ER)], ...clustered([
    ['header', [['h1', 'Exports']]],
    ['state', [['dialog_title', 'Delete this export?', ['in_dialog', 'in_state']],
               ['dialog_body', 'This cannot be undone.', ['in_dialog', 'in_state']]]],
    ...notes(70),
    ['state', [['dialog_title', 'Close your account?', ['in_dialog', 'in_state']],
               ['dialog_body', 'This cannot be undone.', ['in_dialog', 'in_state']]]],
  ]), expect: [] },

  // a08: a conditional h1 is no target; the two always-visible h2s in different windows still repeat (R9).
  { name: 'er_conditional_keep_pos', fw: 'tsx', about: [pos(ER)], ...clustered([
    ['header', [['h1', 'Plans for growing teams', ['conditional']]]],
    ...notes(70, 0),
    ['header', [['h2', 'Plans for growing teams']]],
    ...notes(70, 100),
    ['header', [['h2', 'Plans for growing teams']]],
  ]), expect: [f(ER, 'R9', [142], { target: 71 })] },

  // a09: "Address line N" is a numbered series and "Up to N users" a qualifier on a figure (K13), not lead-ins.
  { name: 'sl_numbered_series_neg', fw: 'tsx', about: [neg(SL)], ...clustered([
    ['header', [['h2', 'Billing address']]],
    ['control', [['label', 'Address line 1', ['in_form', 'in_control']]]],
    ['control', [['label', 'Address line 2', ['in_form', 'in_control']]]],
    ['control', [['label', 'Address line 3', ['in_form', 'in_control']]]],
    ['header', [['h2', 'Plans']]],
    ...[['Solo', 'Up to 5 users'], ['Team', 'Up to 20 users'], ['Studio', 'Up to 100 users']].map(([t, b]) =>
      ['card', [['h3', t, ['in_card', 'repeated']], ['body', b, ['in_card', 'repeated']]]]),
  ]), expect: [] },

  // a10: "Now" first names something (a /now page, a "Now playing" tab, a time picker); only "Book now" is filler.
  { name: 'fw_now_as_content_pos', fw: 'tsx', about: [pos(FW), neg(FW)], ...clustered([
    ['header', [['nav', 'Now', ['in_nav']], ['nav', 'Writing', ['in_nav']]]],
    ['actions', [['tab', 'Now playing'], ['tab', 'Queue']]],
    ['control', [['label', 'Start time', ['in_control']], ['button', 'Now', ['in_control']]]],
    ['actions', [['button', 'Book now']]],
  ]), expect: [f(FW, 'R15', [6])] },

  // a11: "double-click here" and "long-press here" name a hidden gesture (K4); "click here" points at a link.
  { name: 'va_hidden_gesture_pos', fw: 'tsx', about: [pos(VA), neg(VA)], ...clustered([
    ['header', [['h2', 'Layers']]],
    ['body', [['helper', 'Double-click here to rename a layer.'],
              ['tooltip', 'Long-press here to reorder'],
              ['body', 'To read the full report, click here.']]],
  ]), expect: [f(VA, 'R7', [3])] },

  // a12: a dot before interpolated live text shows live state (K2); the fixed "All systems normal" dot does not.
  { name: 'rc_dot_live_text_pos', fw: 'tsx', about: [pos(RC), neg(RC)], ...clustered([
    ['header', [['h2', 'Team']]],
    ['body', [['body', `● ${V} count online`, ['dot', 'dynamic'], [1]],
              ['body', `${V} name is editing`, ['dot', 'dynamic'], [1]],
              ['body', '● All systems normal', ['dot']]]],
  ]), expect: [f(RC, 'R19', [3], { condition: 'decorative-dot' })] },

  // a13: <label for> fields come out as a run of labels in one body cluster; they share "Emergency contact".
  { name: 'sl_label_for_run_pos', fw: 'tsx', about: [pos(SL)], ...clustered([
    ['header', [['h2', 'Next of kin']]],
    ['body', [['label', 'Emergency contact name', ['in_form']],
              ['label', 'Emergency contact phone', ['in_form']],
              ['label', 'Emergency contact email', ['in_form']]]],
  ]), expect: [f(SL, 'R10', [1, 2, 3])] },

  // a14: a 1–5 rating scale and calendar days are bare numbers next to bare numbers: values, not numbering.
  { name: 'ix_scale_and_days_neg', fw: 'tsx', about: [neg(IX)], ...clustered([
    ['header', [['h2', 'How likely are you to recommend us?']]],
    ['body', [1, 2, 3, 4, 5].map((d) => ['caption', String(d)])],
    ['header', [['h2', 'March']]],
    ['body', [1, 2, 3, 4, 5, 6, 7].map((d) => ['other', String(d)])],
  ]), expect: [] },

  // a15: the lead-in sits in the second body of each card.
  { name: 'sl_second_body_pos', fw: 'tsx', about: [pos(SL)], ...clustered([
    ['header', [['h2', 'Plans']]],
    ...[['Solo', '£9 a month', 'Perfect for freelancers on the move'], ['Team', '£29 a month', 'Perfect for small studios'],
      ['Agency', '£99 a month', 'Perfect for agencies with many clients']].map(([t, p, b]) =>
      ['card', [['h3', t, ['in_card', 'repeated']], ['body', p, ['in_card', 'repeated']], ['body', b, ['in_card', 'repeated']]]]),
  ]), expect: [f(SL, 'R10', [3, 6, 9])] },

  // -- from the second review (2026-11-10: screens b01–b18, j01–j04, c01–c05) and guards for its fixes (c06–c08) ----
  // b01: a dot shown only while a data condition holds (`{live && <span><Dot/>Live</span>}`) beside a fixed heading is
  // bound to data (K2); the extractor gives it `conditional`, not `stateful`.
  { name: 'rc_conditional_dot_neg', fw: 'tsx', about: [neg(RC)], ...clustered([
    ['header', [['h2', 'Harbour webcam'], ['subtitle', 'Live', ['dot', 'conditional', 'section_first']]]],
    ['body', [['body', 'The picture refreshes every minute.']]],
  ]), expect: [] },

  // b02: `See {x ? "fewer" : "more"}` comes out "See fewer ALT more": the lone "more" is the tail of the conditional,
  // not a label. `{open ? "Hide" : "Learn"} more` is "Hide ALT Learn more", whose branch "Learn more" is a true R7.
  { name: 'va_alt_fragment_pos', fw: 'tsx', about: [pos(VA), neg(VA)], ...clustered([
    ['header', [['h2', 'Moorings']]],
    ['list', [['list_item', 'Pontoon A, berth 4', ['in_list']], ['list_item', 'Pontoon B, berth 9', ['in_list']]]],
    ['actions', [['link', `See fewer ${A} more`, ['conditional']], ['button', `Hide ${A} Learn more`, ['conditional']]]],
  ]), expect: [f(VA, 'R7', [4])] },

  // b03: a wizard shows one "Step N of 3: …" heading at a time; the lead form keeps its total, so the conditional skip
  // applies (K3).
  { name: 'ix_wizard_lead_total_neg', fw: 'tsx', about: [neg(IX)], ...clustered([
    ['header', [['h2', 'Step 1 of 3: Your boat', ['conditional']]]],
    ['header', [['h2', 'Step 2 of 3: Berth dates', ['conditional']]]],
    ['header', [['h2', 'Step 3 of 3: Payment', ['conditional']]]],
    ['actions', [['button', 'Continue']]],
  ]), expect: [] },

  // b04: an error summary item (a link in the alert) and the inline error it links to are both KEEP (K12), here in
  // different windows.
  { name: 'er_k12_error_summary_neg', fw: 'tsx', about: [neg(ER)], ...clustered([
    ['state', [['h2', 'There is a problem', ['in_state']],
               ['link', 'Enter your date of birth', ['in_list', 'in_state']],
               ['link', 'Choose how to pay the key deposit', ['in_list', 'in_state']]]],
    ['header', [['h1', 'Apply for a plot']]],
    ['control', [['label', 'Date of birth', ['in_form', 'in_control']], ['error', 'Enter your date of birth', ['in_form', 'in_control']]]],
    ...notes(70),
    ['control', [['label', 'Key deposit method', ['in_form', 'in_control']], ['error', 'Choose how to pay the key deposit', ['in_form', 'in_control']]]],
  ]), expect: [] },

  // b05: the tooltip is the only visible name of an icon-only button (K12); the dialog it opens names the same action
  // (K11 between the control and the dialog).
  { name: 'er_k12_icon_tooltip_neg', fw: 'tsx', about: [neg(ER)], ...clustered([
    ['header', [['h1', 'Harbour mix', ['in_header']], ['tooltip', 'Delete playlist', ['in_header']], ['tooltip', 'Clear listening history', ['in_header']]]],
    ...notes(70),
    ['state', [['dialog_title', 'Delete this playlist?', ['in_dialog', 'in_state']],
               ['dialog_body', 'The songs stay in your library.', ['in_dialog', 'in_state']],
               ['button', 'Delete playlist', ['in_dialog', 'in_state']],
               ['button', 'Keep it', ['in_dialog', 'in_state']]]],
    ['state', [['dialog_title', 'Clear listening history?', ['in_dialog', 'in_state']],
               ['button', 'Clear', ['in_dialog', 'in_state']]]],
  ]), expect: [] },

  // b06: the hour labels of a day grid, "06" … "11", are values: the padded column continues past "09" and starts above
  // one, so it is not an index.
  { name: 'ix_hour_gutter_neg', fw: 'tsx', about: [neg(IX), neg(RC)], ...clustered([
    ['header', [['h2', 'Tuesday at the boatyard']]],
    ...['06', '07', '08', '09', '10', '11'].map((h) => ['card', [['caption', h, ['small', 'in_card', 'repeated']], ['body', 'Slip free for hire', ['in_card', 'repeated']]]]),
  ]), expect: [] },

  // b07: another node refers to the numbers in the plural ("Steps 2 and 3 …", K3).
  { name: 'ix_steps_plural_reference_neg', fw: 'tsx', about: [neg(IX)], ...clustered([
    ['header', [['h2', 'Launch a boat']]],
    ['body', [['h3', 'Step 1'], ['body', 'Book the slipway at the office.']]],
    ['body', [['h3', 'Step 2'], ['body', 'Check the trailer lights.']]],
    ['body', [['h3', 'Step 3'], ['body', 'Wash the hull after launching.'], ['body', 'Steps 2 and 3 can be done in either order.']]],
  ]), expect: [] },

  // b08: footnote numbers referenced by markers in the price line (K3, K13); `<sup>1</sup>` comes out glued, "night1".
  { name: 'ix_footnote_markers_neg', fw: 'tsx', about: [neg(IX), neg(RC)], ...clustered([
    ['header', [['h2', 'Mooring fees'], ['subtitle', 'From £14 a night1, or £300 a season2, plus harbour dues3.']]],
    ['card', [['caption', '1', ['small', 'in_card', 'repeated']], ['caption', 'Boats up to eight metres.', ['small', 'in_card', 'repeated']]]],
    ['card', [['caption', '2', ['small', 'in_card', 'repeated']], ['caption', 'April to October only.', ['small', 'in_card', 'repeated']]]],
    ['card', [['caption', '3', ['small', 'in_card', 'repeated']], ['caption', 'Set by the council each spring.', ['small', 'in_card', 'repeated']]]],
  ]), expect: [] },

  // b09: "Very low" … "Very high" in an option list, and "Very good" in a table, are values on a scale (K5).
  { name: 'fw_scale_values_neg', fw: 'tsx', about: [neg(FW)], ...clustered([
    ['control', [['label', 'Wind risk', ['in_form', 'in_control']],
                 ['menu_item', 'Very low', ['in_form', 'in_list', 'in_control']], ['menu_item', 'Low', ['in_form', 'in_list', 'in_control']],
                 ['menu_item', 'High', ['in_form', 'in_list', 'in_control']], ['menu_item', 'Very high', ['in_form', 'in_list', 'in_control']]]],
    ['list', [['cell', 'Hull', ['in_table', 'in_list']], ['cell', 'Very good', ['in_table', 'in_list']],
              ['cell', 'Sails', ['in_table', 'in_list']], ['cell', 'Fair', ['in_table', 'in_list']]]],
  ]), expect: [] },

  // b10: a badge chosen per card by a condition (`{c.isNew ? "New" : "Updated"}`) can differ between siblings (K2).
  { name: 'rc_conditional_badge_neg', fw: 'tsx', about: [neg(RC)], ...clustered([
    ['header', [['h2', 'Route cards']]],
    ['card', [['h3', 'Green slab', ['in_card']], ['badge', `New ${A} Updated`, ['small', 'pill', 'conditional', 'in_card']]]],
    ['card', [['h3', 'Red overhang', ['in_card']], ['badge', `New ${A} Updated`, ['small', 'pill', 'conditional', 'in_card']]]],
    ['card', [['h3', 'Blue crack', ['in_card']], ['badge', `New ${A} Updated`, ['small', 'pill', 'conditional', 'in_card']]]],
  ]), expect: [] },

  // b11: "long" and "double" end a sentence here, so "Click here" and "Tap here" are not named gestures; the third node
  // is the hidden-input case (K4).
  { name: 'va_gesture_sentences_pos', fw: 'html', about: [pos(VA), neg(VA)], ...clustered([
    ['header', [['h2', 'Climbing wall']]],
    ['body', [['body', 'The queue for the wall is long. Click here to join it.'],
              ['body', 'Weekend points are double. Tap here to claim them.'],
              ['body', 'Double-click a route card to rename it.']]],
  ]), expect: [f(VA, 'R7', [1]), f(VA, 'R7', [2])] },

  // b12: arrow suffixes outside the first list (➡️ with U+FE0F, ⇒, ➝); the extractor sets no `arrow_tail` for them.
  { name: 'rc_arrow_glyphs_pos', fw: 'html', about: [pos(RC)], ...clustered([
    ['header', [['h2', 'Guides']]],
    ['actions', [['link', 'Belay basics ➡️'], ['link', 'Knots for beginners ⇒'], ['link', 'Grading explained ➝']]],
  ]), expect: [f(RC, 'R19', [1, 2, 3], { condition: 'arrow-links' })] },

  // b13, a known false negative: a <legend> comes out as a `label` in a body cluster, exactly like a <label for>, so it
  // counts as the label of the first field after it ("Street"). "Town" and "Postcode" are flagged.
  { name: 'pl_fieldset_legend_pos', fw: 'tsx', about: [pos(PL)], ...clustered([
    ['header', [['h2', 'Collection']]],
    ['body', [['label', 'Collection address', ['in_form']]]],
    ['control', [['placeholder', 'Street', ['in_form', 'in_control']]]],
    ['control', [['placeholder', 'Town', ['in_form', 'in_control']]]],
    ['control', [['placeholder', 'Postcode', ['in_form', 'in_control']]]],
  ]), expect: [f(PL, 'R8b', [3]), f(PL, 'R8b', [4])] },

  // b14, a known false negative: `<label for="nick">Nickname</label><input id="nick"/>` gives no node for the input, so
  // the next field's placeholder takes "Nickname" as its label. The screen cannot tell this from a <label for> whose
  // input has the placeholder.
  { name: 'pl_label_for_textless_neg', fw: 'tsx', about: [neg(PL)], ...clustered([
    ['header', [['h2', 'Crew details']]],
    ['body', [['label', 'Nickname']]],
    ['control', [['placeholder', 'Phone number', ['in_control']]]],
  ]), expect: [] },

  // b15: a padded column "01" … "12" is one device; "10"–"12" continue it.
  { name: 'ix_padded_past_nine_pos', fw: 'tsx', about: [pos(IX), pos(RC)], ...clustered([
    ['header', [['h2', 'Harbour mix']]],
    ['list', ['Harbour lights', 'Low tide waltz', 'Gull song', 'Ropewalk', 'Salt on the sill', 'Night ferry', 'Lantern row', 'Breakwater', 'Fog bell', 'Slipway', 'Mooring line', 'Pilot boat']
      .map((t, j) => ['list_item', `${String(j + 1).padStart(2, '0')} ${t}`, ['in_list']])],
  ]), expect: [f(RC, 'R19', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], { condition: 'sibling-device' }), f(IX, 'R12', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])] },

  // b16: `{mine ? "Your" : "Shared"} playlists` comes out "Your ALT Shared playlists"; the lone "Your" leads "Your
  // playlists".
  { name: 'fw_alt_lone_your_pos', fw: 'tsx', about: [pos(FW)], ...clustered([
    ['header', [['h2', `Your ${A} Shared playlists`, ['conditional']]]],
    ['list', [['list_item', 'Harbour mix', ['in_list']], ['list_item', 'Night ferry', ['in_list']]]],
  ]), expect: [f(FW, 'R15', [0])] },

  // b17: "01 / 03 Connect the hub", a padded device with a total used as a lead.
  { name: 'ix_padded_total_lead_pos', fw: 'tsx', about: [pos(IX), pos(RC)], ...clustered([
    ['header', [['h2', 'Set up the tide sensor']]],
    ['card', [['h3', '01 / 03 Connect the hub', ['in_card']], ['body', 'Plug it in near the router.', ['in_card']]]],
    ['card', [['h3', '02 / 03 Pair the sensors', ['in_card']], ['body', 'Hold each one by the hub.', ['in_card']]]],
    ['card', [['h3', '03 / 03 Name the berths', ['in_card']], ['body', 'Names show on the alerts.', ['in_card']]]],
  ]), expect: [f(RC, 'R19', [1, 3, 5], { condition: 'sibling-device' }), f(IX, 'R12', [1, 3, 5])] },

  // b18: "Step 1 of 3" (whole) then "Step 2 of 3: …" (lead) keep one total, so the run holds all three.
  { name: 'ix_mixed_totals_pos', fw: 'tsx', about: [pos(IX), pos(RC)], ...clustered([
    ['header', [['h2', 'Join the club']]],
    ['card', [['h3', 'Step 1 of 3', ['in_card']], ['body', 'Pick a membership.', ['in_card']]]],
    ['card', [['h3', 'Step 2 of 3: Safety talk', ['in_card']], ['body', 'Twenty minutes at the desk.', ['in_card']]]],
    ['card', [['h3', 'Step 3 of 3: First climb', ['in_card']], ['body', 'An instructor belays you.', ['in_card']]]],
  ]), expect: [f(RC, 'R19', [1, 3, 5], { condition: 'sibling-device' }), f(IX, 'R12', [1, 3, 5])] },

  // j01 (lead's decision): footer links are left out of the arrow-links count.
  { name: 'rc_arrow_links_footer_pos', fw: 'tsx', about: [pos(RC)], ...clustered([
    ['header', [['h2', 'Tide tables']]],
    ['actions', [['link', 'Read the tide tables →', ['arrow_tail']], ['link', 'See mooring prices →', ['arrow_tail']]]],
    ['body', [['link', 'Terms', ['in_footer']], ['link', 'Privacy', ['in_footer']]]],
  ]), expect: [f(RC, 'R19', [1, 2], { condition: 'arrow-links' })] },

  // j02 (lead's decision): a site-bar link (in_header, in a cluster with no heading) is navigation chrome (K5), like a
  // footer link, so a later "Sign in" link does not repeat it.
  { name: 'er_site_bar_link_neg', fw: 'tsx', about: [neg(ER)], ...clustered([
    ['header', [['link', 'Harbourside Allotments', ['in_header']], ['link', 'Sign in', ['in_header']]]],
    ...notes(70),
    ['body', [['h2', 'Questions about plots'], ['link', 'Sign in']]],
  ]), expect: [] },

  // j03 (lead's decision): a tax-status qualifier on each price card is KEEP in every copy (K13), not a device.
  { name: 'rc_qualifier_badge_neg', fw: 'tsx', about: [neg(RC)], ...clustered([
    ['header', [['h2', 'Berth plans']]],
    ['card', [['h3', 'Day', ['in_card']], ['stat', '£14', ['in_card']], ['badge', 'Inc. VAT', ['small', 'pill', 'in_card']]]],
    ['card', [['h3', 'Month', ['in_card']], ['stat', '£180', ['in_card']], ['badge', 'Inc. VAT', ['small', 'pill', 'in_card']]]],
    ['card', [['h3', 'Season', ['in_card']], ['stat', '£300', ['in_card']], ['badge', 'Inc. VAT', ['small', 'pill', 'in_card']]]],
  ]), expect: [] },

  // j04 (lead's decision): the filler list stays as it is; "quickly" in an instruction still fires (the rule layer is a
  // cheap baseline).
  { name: 'fw_quickly_instruction_pos', fw: 'tsx', about: [pos(FW)], ...clustered([
    ['header', [['h2', 'Cradle lock']]],
    ['body', [['body', 'Turn the winch quickly to lock the cradle.']]],
  ]), expect: [f(FW, 'R15', [1])] },

  // c01: `vw` sliced per ALT branch with two markers: "now" after identifier words in the second branch, and "Your crew"
  // as a two-word branch after an identifier-only branch.
  { name: 'fw_vw_alt_mix_pos', fw: 'tsx', about: [pos(FW)], ...clustered([
    ['header', [['h2', `${V} crew name ${A} Your crew`, ['conditional', 'dynamic'], [2]]]],
    ['actions', [['button', `Pay ${V} amount due ${A} Pay ${V} balance now`, ['conditional', 'dynamic'], [2, 1]]]],
  ]), expect: [f(FW, 'R15', [0]), f(FW, 'R15', [1])] },

  // c02: one stateful dot per row is K2, and the section badge's dot is one dot.
  { name: 'rc_row_dots_header_dot_neg', fw: 'tsx', about: [neg(RC)], ...clustered([
    ['header', [['h2', 'Pontoons'], ['badge', `${V} overall state`, ['small', 'dot', 'stateful', 'dynamic'], [2]]]],
    ['list', [['list_item', 'Pontoon A', ['in_list']], ['caption', `${V} power state`, ['dot', 'stateful', 'dynamic', 'in_list'], [2]],
              ['list_item', 'Pontoon B', ['in_list']], ['caption', `${V} power state`, ['dot', 'stateful', 'dynamic', 'in_list'], [2]],
              ['list_item', 'Pontoon C', ['in_list']], ['caption', `${V} power state`, ['dot', 'stateful', 'dynamic', 'in_list'], [2]]]],
  ]), expect: [] },

  // c03: fraction runs with different totals do not join; each is reported.
  { name: 'ix_totals_split_pos', fw: 'tsx', about: [pos(IX)], ...clustered([
    ['header', [['h2', 'Gallery']]],
    ['body', [['caption', '1 / 2'], ['body', 'Slipway at dawn'], ['caption', '2 / 2'], ['body', 'Slipway at dusk']]],
    ['body', [['caption', '1 / 3'], ['body', 'Pier'], ['caption', '2 / 3'], ['body', 'Boathouse'], ['caption', '3 / 3'], ['body', 'Crane']]],
  ]), expect: [f(IX, 'R12', [1, 3]), f(IX, 'R12', [5, 7, 9])] },

  // c04: a footer link is skipped; a body call to action repeated across windows is still R14.
  { name: 'er_footer_vs_body_cta_pos', fw: 'tsx', about: [pos(ER), neg(ER)], ...clustered([
    ['actions', [['button', 'Book a guided climb']]],
    ...notes(70),
    ['actions', [['button', 'Book a guided climb']]],
    ['body', [['link', 'Book a guided climb', ['in_footer']]]],
  ]), expect: [f(ER, 'R14', [71], { target: 0 })] },

  // c05: when the ternary is the whole label (`{sending ? "Sending your form" : "Submit"}`), "Submit" is still R7; the
  // fragment reading covers only a lone "more" or "here".
  { name: 'va_whole_branch_submit_pos', fw: 'tsx', about: [pos(VA)], ...clustered([
    ['control', [['label', 'Message', ['in_form', 'in_control']]]],
    ['actions', [['button', `Sending your form ${A} Submit`, ['conditional', 'in_form']]]],
  ]), expect: [f(VA, 'R7', [1])] },

  // c06: inside a conditional block (`{show && …}`) every node is `conditional`, so the flag alone is not a node's own
  // condition: a fixed dot and a fixed pill on every card are still R19.
  { name: 'rc_conditional_block_pos', fw: 'tsx', about: [pos(RC)], ...clustered([
    ['header', [['h2', 'Winter berths']]],
    ['body', [['body', 'Harbour office open', ['dot', 'conditional']], ['body', 'Ask at the hut for the gate key.', ['conditional']]]],
    ...['Dinghy', 'Keelboat', 'Catamaran'].map((t) => ['card', [['h3', t, ['conditional', 'in_card', 'repeated']], ['badge', 'New', ['pill', 'conditional', 'in_card', 'repeated']]]]),
  ]), expect: [f(RC, 'R19', [1], { condition: 'decorative-dot' }), f(RC, 'R19', [4, 6, 8], { condition: 'sibling-device' })] },

  // c07: a two-digit number with words ("12 months …") after a bare number is not a device that makes the bare number
  // a value; it only continues a padded column.
  { name: 'ix_plain_before_wide_pos', fw: 'tsx', about: [pos(IX)], ...clustered([
    ['header', [['h2', 'Fit the tide gauge']]],
    ['body', [['caption', '1'], ['body', 'Fix the mast to the pier rail.'], ['caption', '2'], ['body', 'Point the sensor at open water.'],
              ['caption', '3'], ['body', '12 months of readings fit on the card.']]],
  ]), expect: [f(IX, 'R12', [1, 3, 5])] },

  // c08: decimals and times are not footnote markers ("4.1", "4.2" do not refer to notes 1 and 2).
  { name: 'ix_decimal_not_marker_pos', fw: 'tsx', about: [pos(IX)], ...clustered([
    ['header', [['h2', 'Visitor moorings']]],
    ['body', [['caption', '1'], ['body', 'Members only.'], ['caption', '2'], ['body', 'Weekdays from 9:30.'], ['caption', '3'], ['body', 'Closed in July.'],
              ['body', 'Rated 4.1 in spring and 4.2 in autumn.']]],
  ]), expect: [f(IX, 'R12', [1, 3, 5])] },

  // -- a clean screen: no rule fires ------------------------------------------------------------------------------
  { name: 'clean_settings_neg', fw: 'swift', about: [neg(RC), neg(VA), neg(FW), neg(PL), neg(SL), neg(IX), neg(ER)],
    kinds: ['header', 'control', 'control', 'actions'], nodes: [
      n(0, 'h1', 'Settings'),
      n(1, 'label', 'Display name', ['in_form', 'in_control']),
      n(1, 'helper', 'Shown on receipts and the staff rota.', ['small', 'in_form', 'in_control']),
      n(2, 'label', 'Time zone', ['in_form', 'in_control']),
      n(2, 'menu_item', 'London', ['in_form', 'in_list', 'in_control']),
      n(2, 'menu_item', 'Lisbon', ['in_form', 'in_list', 'in_control']),
      n(2, 'menu_item', 'New York', ['in_form', 'in_list', 'in_control', 'title_case']),
      n(3, 'button', 'Save changes', ['in_form']),
      n(3, 'button', 'Cancel', ['in_form']),
    ], expect: [] },
];

// -- writing -------------------------------------------------------------------------------------------------------

const EXT = { tsx: 'tsx', html: 'html', swift: 'swift' };
const j = (v) => JSON.stringify(v).replace(/[]/g, (c) => `\\u${c.charCodeAt(0).toString(16)}`);
const obj = (pairs) => `{${pairs.map(([k, v]) => `${j(k)}: ${v}`).join(', ')}}`;
const arr = (xs) => `[${xs.map(j).join(', ')}]`;

// The extractor's serialiser layout (extract/src/screen.ts): one cluster and one node per line.
function serialise(s) {
  const src = obj([['repo', j(s.source.repo)], ['path', j(s.source.path)], ['commit', j(s.source.commit)], ['licence', j(s.source.licence)]]);
  const cl = s.clusters.map((c) => obj([['k', j(c.k)], ['kind', j(c.kind)], ['line', j(c.line)]]));
  const nd = s.nodes.map((x) => obj([['i', j(x.i)], ['k', j(x.k)], ['role', j(x.role)], ['text', j(x.text)], ['flags', arr(x.flags)], ['loc', arr(x.loc)],
    ...(x.vw ? [['vw', arr(x.vw)]] : [])]));
  const list = (xs) => (xs.length ? `[\n  ${xs.join(',\n  ')}]` : '[]');
  return `{"schema": ${j(s.schema)}, "id": ${j(s.id)}, "framework": ${j(s.framework)},\n "source": ${src},\n "clusters": ${list(cl)},\n "nodes": ${list(nd)}}\n`;
}

function build(fx) {
  const nodes = fx.nodes.map(([k, role, text, flags, vw], i) => {
    for (const fl of flags) if (!FLAGS.includes(fl)) throw new Error(`${fx.name}: unknown flag ${fl}`);
    if (vw && vw.length !== text.split(V).length - 1) throw new Error(`${fx.name}: node ${i} has ${vw.length} vw counts for its VAR markers`);
    const line = 3 + i + k; // a blank line before each cluster
    const node = { i, k, role, text, flags: FLAGS.filter((fl) => flags.includes(fl)), loc: [line, 7, line, 7 + Array.from(text).length] };
    if (vw) node.vw = vw;
    return node;
  });
  const clusters = fx.kinds.map((kind, k) => {
    const first = nodes.find((x) => x.k === k);
    if (!first) throw new Error(`${fx.name}: cluster ${k} has no nodes`);
    return { k, kind, line: first.loc[0] - 1 };
  });
  const p = `src/${fx.name}.${EXT[fx.fw]}`;
  const id = createHash('sha256').update(`slop.screen/1\0slop-rules-fixtures\0${p}\0${JSON.stringify(fx.nodes.map((x) => (x[4] ? x : x.slice(0, 4))))}`).digest('hex').slice(0, 16);
  const screen = { schema: 'slop.screen/1', id, framework: fx.fw, source: { repo: 'slop-rules-fixtures', path: p, commit: null, licence: 'MIT' }, clusters, nodes };
  const findings = fx.expect.map((e) => {
    const out = { rule: e.rule, code: e.code };
    if (e.condition) out.condition = e.condition;
    out.nodes = e.nodes;
    if (e.target !== undefined) out.target = e.target;
    return out;
  });
  return { screen, entry: { about: fx.about, findings } };
}

export function buildAll() {
  const names = new Set();
  const files = new Map();
  const expected = {};
  for (const fx of FIXTURES) {
    if (names.has(fx.name)) throw new Error(`duplicate fixture ${fx.name}`);
    names.add(fx.name);
    const { screen, entry } = build(fx);
    files.set(`screens/${fx.name}.json`, serialise(screen));
    expected[fx.name] = entry;
  }
  const lines = Object.entries(expected).map(([k, v]) => `  ${JSON.stringify(k)}: ${JSON.stringify(v)}`);
  files.set('expected.json', `{\n${lines.join(',\n')}\n}\n`);
  return files;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const files = buildAll();
  const check = process.argv.includes('--check');
  const onDisk = new Set(readdirSync(path.join(HERE, 'screens'), { withFileTypes: false }).map((x) => `screens/${x}`));
  let stale = 0;
  for (const [rel, text] of files) {
    const abs = path.join(HERE, rel);
    let old = null;
    try { old = readFileSync(abs, 'utf8'); } catch { /* new file */ }
    onDisk.delete(rel);
    if (old === text) continue;
    stale++;
    if (check) process.stderr.write(`stale: ${rel}\n`);
    else { mkdirSync(path.dirname(abs), { recursive: true }); writeFileSync(abs, text); }
  }
  for (const rel of onDisk) {
    stale++;
    if (check) process.stderr.write(`extra: ${rel}\n`);
    else rmSync(path.join(HERE, rel));
  }
  if (check) process.exit(stale ? 1 : 0);
  process.stdout.write(`${files.size - 1} screens, ${stale} file${stale === 1 ? '' : 's'} written or removed\n`);
}
