#!/usr/bin/env node
// Build the rule layer's test screens (slop.screen/1) and their expected findings from the compact tables below.
//
//   node slop/rules/fixtures/make_fixtures.mjs           write screens/*.json and expected.json
//   node slop/rules/fixtures/make_fixtures.mjs --check   exit 1 if the files on disk differ from what would be written
//
// Hand-written stand-ins for extractor output: roles, flags and clusters are chosen by hand and each node sits on its
// own line, so `loc` is [line, 7, line, 7 + length]. All copy is invented. U+E000 and U+E001 are the extractor's VAR
// and ALT markers (SPEC.md §4). Each fixture names the rules it is a positive or a near-miss negative for; a fixture
// expects exactly the findings listed, from every rule.

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

// node: [k, role, text, flags]; finding: [rule, code, [i…], {condition, target}]
const n = (k, role, text, flags = []) => [k, role, text, flags];
const f = (rule, code, nodes, extra = {}) => ({ rule, code, nodes, ...extra });
const pos = (rule) => ({ rule, kind: 'positive' });
const neg = (rule) => ({ rule, kind: 'negative' });

// Distinct list rows for the long screens that need two windows.
const TREES = ['Alder', 'Birch', 'Cedar', 'Damson', 'Elder', 'Fir', 'Gean', 'Hazel', 'Ivy', 'Juniper', 'Larch', 'Maple'];
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
    n(0, 'caption', `${V} last job`, ['small', 'dynamic', 'stateful']),
    n(1, 'button', 'Print test page'),
  ], expect: [] },

  { name: 'rc_state_dots_pos', fw: 'tsx', about: [pos(RC)], kinds: ['header', 'list'], nodes: [
    n(0, 'h2', 'Service status'),
    n(1, 'list_item', `Bookings ${V} state`, ['dot', 'stateful', 'in_list']),
    n(1, 'list_item', `Payments ${V} state`, ['dot', 'stateful', 'in_list']),
    n(1, 'list_item', `Email receipts ${V} state`, ['dot', 'stateful', 'in_list']),
  ], expect: [f(RC, 'R19', [1, 2, 3], { condition: 'state-dots' })] },

  { name: 'rc_state_dots_neg', fw: 'tsx', about: [neg(RC)], kinds: ['header', 'header', 'header'], nodes: [
    n(0, 'h2', 'Bookings'),
    n(0, 'body', `${V} bookings state`, ['dot', 'stateful']),
    n(1, 'h2', 'Payments'),
    n(1, 'body', `${V} payments state`, ['dot', 'stateful']),
    n(2, 'h2', 'Email receipts'),
    n(2, 'body', `${V} email state`, ['dot', 'stateful']),
  ], expect: [] },

  { name: 'rc_arrow_links_pos', fw: 'html', about: [pos(RC)], kinds: ['header', 'body', 'actions'], nodes: [
    n(0, 'h2', 'Help for new tenants'),
    n(1, 'body', 'Most questions are answered in the guides below.'),
    n(2, 'link', 'Moving-in checklist →', ['arrow_tail']),
    n(2, 'link', 'Paying the rent', ['icon', 'arrow_tail']),
    n(2, 'link', 'Reporting a repair ↗', ['arrow_tail']),
  ], expect: [f(RC, 'R19', [2, 3, 4], { condition: 'arrow-links' })] },

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
  const nd = s.nodes.map((x) => obj([['i', j(x.i)], ['k', j(x.k)], ['role', j(x.role)], ['text', j(x.text)], ['flags', arr(x.flags)], ['loc', arr(x.loc)]]));
  const list = (xs) => (xs.length ? `[\n  ${xs.join(',\n  ')}]` : '[]');
  return `{"schema": ${j(s.schema)}, "id": ${j(s.id)}, "framework": ${j(s.framework)},\n "source": ${src},\n "clusters": ${list(cl)},\n "nodes": ${list(nd)}}\n`;
}

function build(fx) {
  const nodes = fx.nodes.map(([k, role, text, flags], i) => {
    for (const fl of flags) if (!FLAGS.includes(fl)) throw new Error(`${fx.name}: unknown flag ${fl}`);
    const line = 3 + i + k; // a blank line before each cluster
    return { i, k, role, text, flags: FLAGS.filter((fl) => flags.includes(fl)), loc: [line, 7, line, 7 + Array.from(text).length] };
  });
  const clusters = fx.kinds.map((kind, k) => {
    const first = nodes.find((x) => x.k === k);
    if (!first) throw new Error(`${fx.name}: cluster ${k} has no nodes`);
    return { k, kind, line: first.loc[0] - 1 };
  });
  const p = `src/${fx.name}.${EXT[fx.fw]}`;
  const id = createHash('sha256').update(`slop.screen/1\0slop-rules-fixtures\0${p}\0${JSON.stringify(fx.nodes)}`).digest('hex').slice(0, 16);
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
