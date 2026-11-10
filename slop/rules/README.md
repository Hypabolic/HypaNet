# Rule layer

The plain-code rules of `SPEC.md` §8. They read one `slop.screen/1` document (the extractor's output, `SPEC.md` §4)
and report the patterns that are counts or fixed word lists: repeated chrome, vacuous action labels, filler words,
placeholders used as labels, shared lead-ins, index numbering and exact repeats across windows. They ship with the
linter beside the network rule and are scored separately against the `aside` codes of `LABELLING.md`; the network
gets no credit for them.

Ten **codex rules** follow them: the LLM-writing patterns of the slop codex (`codex/llm.md`, `codex/ui.md`) that a
word list, a construction or a glyph count decides, so they belong in plain code and not in the network (codex
`llm.md` §5 finding 1: most LLM markers are not redundancy). They report with the codex id of the pattern (L01–L52,
U01–U52), or with aside R11g for the leading emoji, and carry a severity. See "Codex rules" below.

The code is plain ES modules with no dependencies beyond Node. It imports `lint/tokenizer.js` only for the window
split that `slop/exact-repeat` needs; every other rule uses its own small word splitter (`src/text.js`), because the
tokeniser cuts a node at 31 tokens and the rules need every word.

## Running it

```
node slop/rules/cli.js <screen.json|dir> [...] [--format text|json] [--rules id,id] [--list]
```

- A directory is walked recursively for `.json` files. Files whose `schema` is not `slop.screen/1` are skipped and
  counted; files that do not parse are reported as errors and the run carries on.
- `--format text` (the default) prints one line per finding: `line:col  code  message  rule`, with `(+N more)` when
  the finding lists more than one node.
- `--format json` prints one document:

  ```json
  {"schema": "slop.rules/1",
   "files": [{"file": "…/hero.json", "screen": "b3f1…", "findings": [
     {"rule": "slop/vacuous-action", "code": "R7", "scope": "node", "nodes": [3], "locs": [[16, 11, 16, 21]],
      "message": "link \"Learn more\" is a stock label; name the outcome or destination"}]}],
   "skipped": 0, "errors": []}
  ```

  Each finding has `rule`, `code` (the pattern or aside code), `scope` (`screen` for R19, a screen-level aside;
  `node` otherwise), `nodes` (the flagged nodes' `i`), `locs` (their `loc`, in the same order) and `message`.
  `slop/repeated-chrome` adds `condition`; `slop/exact-repeat` adds `target`, `{"node", "loc"}` of the copy that is
  kept.
- `--rules` runs only the named rules; `--list` prints the rule ids.
- Exit status: 0 no findings, 1 findings, 2 a file could not be read or checked (the other files are still reported).
- `scope` is `screen` for R19 and for the two per-screen counts (L48, L51), `node` otherwise.

From Python, run it with `subprocess` and read stdout as JSON (`rules/tests/test_rules_cli.py` does this). From
JavaScript, `import { runRules, RULES } from './rules/index.js'`; `runRules(screen, {only})` returns the findings
for one screen and `RULES` lists `{id, code, severity, description}` in report order, for the ESLint packaging later
(§14). `severity` is `error` (never meant to ship), `warning` (style: fix it) or `info` (check it: a density or a
claim); the seven original rules are `warning`.

## Shared definitions

- **Scored nodes.** Context-only roles (`aria`, `alt`, `code`, `comment`) and `sr_only` nodes are never labelled
  (`LABELLING.md` §2), so no rule reports them. An `aria` label or an `sr_only` label never counts as a visible label.
- **Branches.** Text joined by the ALT marker (U+E001) is split into its branches, and the word rules check each
  branch on its own.
- **Identifier words.** The words that follow a VAR marker (U+E000) are the identifier's, never matched as copy. The
  extractor's `vw` field (`SPEC.md` §4: one count per marker, in order) says exactly how many words and numbers follow
  each marker; each ALT branch reads the counts of its own markers. When a node has no `vw`, or its length differs
  from the number of markers, up to four words after each marker, up to the next number or punctuation, are taken as
  identifier words instead (a cap for screens from an older extractor). The letters "VAR" and "ALT" are ordinary
  words.
- **Normalised string.** NFKC, lower case, accents dropped, inner apostrophes removed, glyphs and punctuation dropped,
  markers kept, words and numbers joined by single spaces.
- **Section heading.** An `h1`–`h6` not flagged `in_card`, `repeated`, `in_list`, `in_table`, `in_nav`, `in_dialog`
  or `in_state`. A node's section runs from the last section heading before it.
- **Sibling groups.** A run of two or more consecutive `card` clusters (one item per card); a run of two or more
  consecutive `control` clusters (one item per field); a run of two or more form labels in reading order (scored
  `label` nodes flagged `in_form` or `in_control`, each item a label with the `helper`, `tooltip`, `error`,
  `placeholder` and `input_value` nodes after it, the run ending at any other scored node: a `<label for>` lands in a
  body cluster and its input in a control cluster of its own); and the rows of one `list` cluster, a new row starting
  at each node with the role of the first node whose role occurs at least twice (nodes before it, such as the list's
  own title, belong to no row). Within a group, the aligned sets of a role are the first node of that role in every
  item, then the second when every item has a second, and so on, for each role that every item has.
- **Precedence** (`LABELLING.md` §5, lower is kept): controls, a `label` flagged `in_control`, `h1`, `dialog_title`
  and `empty_title` 10; `h2`–`h6` 20–24; body roles and any other `label` 30; `subtitle`, `helper`, `caption` 40;
  `eyebrow`, `badge`, `tooltip`, `placeholder` 50.

## The rules

### `slop/repeated-chrome` (screen aside R19)

One finding per condition met, listing the nodes that carry the device; state dots give one finding per section.

| Condition | Fires when |
|---|---|
| `eyebrow-quota` | the eyebrows directly above a section heading (the next scored node is one) number more than ceil(section headings / 3). Taste: design-taste-frontend-v2-experimental SKILL.md:254-256, a landing-page quota, "Hero counts as 1". |
| `decorative-dot` | any scored node has `dot` and is not bound to data (below) (v2-experimental SKILL.md:705; a dot before interpolated text, or shown only while a data condition holds, shows live state, K2). |
| `state-dots` | one section holds two or more nodes with both `dot` and `stateful`, not counting one dot per sibling item (same line: "limited to one per page section"). K2 wins over the limit: in a card or row group where two or more items carry a stateful dot and no item carries more than one, each dot shows its own item's state, and those dots are not counted. |
| `arrow-links` | there are two or more `link` nodes outside the footer (footer links, `in_footer`, are navigation and are not counted) and every one has an arrow suffix (tasteskill copy.md:126-127; Hallmark gate 63). An arrow suffix is the text ending in → ➔ ➜ ➝ ➞ ➡ ⟶ ⟹ ⇒ ⇢ ⇾ ⭢ ⮕ 🡒 (an emoji variation selector after it allowed, "➡️"), `->` or `=>`, or in › as a separate final token ("Next ›"); otherwise the `arrow_tail` flag (an arrow icon). A text ending in ↗ (an external-link marker, which informs), ↘, » or a › glued to the last word is not an arrow suffix, even when the extractor set `arrow_tail`. |
| `sibling-device` | in a card or row group of three or more items, one device sits on more than half of them: the same normalised text in a `badge`, or in a `pill` node that is not a control (`button`, `link`, `nav`, `tab`, `menu_item`, `label`, `placeholder`, `input_value`: the extractor sets `pill` from `rounded-full`, the usual shape of a button), that is not bound to data and is not a K13 qualifier (below); the same text in an `eyebrow` that is not bound to data; an arrow suffix on a `link` or `button`; or a member of a `slop/index-numbering` run. Dropped when every node it lists is already in another R19 finding. From `LABELLING.md` R19 ("the same eyebrow, pill, dot, arrow suffix or index-number device on most sibling sections, cards or rows"). |

A node is **bound to data** (K2) when it is `stateful` or `dynamic`, or when it is `conditional` by a condition of
its own: its text holds ALT branches, or another scored node of its card or row (for `sibling-device`) or of its
cluster (for `decorative-dot`) is not `conditional`. The extractor flags every node inside a conditional block (an
`&&` section, a dialog, an early-return branch) `conditional`, so the flag alone does not show a per-node condition: a
fixed "New" pill on every card of a `{show && …}` block is still a device.

A **K13 qualifier** is a badge or pill whose whole text is a tax status ("Inc. VAT", "+ VAT", "VAT included", "before
tax"), a billing period ("/month", "per year", "a night", "monthly", "billed annually"), a unit ("per seat",
"/user/month", "each"), a currency ("GBP", "in EUR") or "from" (`QUALIFIERS` in `src/lists.js`). It is `KEEP` in
every copy, so it is not a repeated device.

Known false positive: a static legend ("● Healthy ● Degraded ● Down", K2) has `dot` without `stateful`, so
`decorative-dot` fires on it; the screen JSON does not say which dots form a legend. Likewise the extractor sets
`arrow_tail` for an `ExternalLink` or `ArrowUpRight` icon, which the rule cannot tell from a plain arrow icon, so
icon-marked external links still count as arrow suffixes.

### `slop/vacuous-action` (aside R7)

A node is flagged, once, when any branch of it

- is a `button` or `link` whose whole label (its copy words joined by spaces) is in `VACUOUS_LABELS` for that role;
- is a `button` or `link` whose label starts with a gesture lead: "click to", "tap to", "click here to", "tap here
  to", "press here to";
- in any scored role, contains a gesture phrase: "click here", "tap here", "click this link", "press here" (inline
  links are merged into their paragraph, so "To change it, click here." is one body node).

A named gesture (double, triple, right, middle or long, then click, tap or press, separated by a space or a hyphen
only) is read as one word first: "Double-click here to rename a layer." tells the reader about a hidden input
(`LABELLING.md` K4) and is not flagged, while "The queue is long. Click here to join it." is.

A lone "more" or "here" branch beside a branch of two or more words is not read as a label: `See {x ? "fewer" :
"more"}` comes out "See fewer ALT more", the extractor joining the shared words to one branch only. A whole-label
branch ("Submit" in "Sending your form ALT Submit") is still a label.

`VACUOUS_LABELS` (15): learn more, read more, find out more, discover more, explore now, explore more, click here,
tap here, get started, submit, click, tap (buttons and links); here, more, this link (links only, since a "More"
button is often an overflow menu). Left out on purpose: OK, Continue, Next, Done, Cancel, Close (platform
conventions whose meaning the dialog or flow supplies) and Show more, See more (they say what they do).

### `slop/filler-words` (aside R15)

One finding per node, naming every filler phrase found in its branches (`FILLER`, 14 entries):

- in any scored role: please, simply, easily, quickly, effectively, very, quite, in order to, oops, whoops, uh oh;
- in a `button`, `link`, `menu_item`, `nav` or `tab` only, and not as the first word: now ("Book now"; in a sentence
  it is often a fact, and first it names something: a "Now" page, a "Now playing" tab, a time picker's "Now"; after
  "not" it is the meaning of "Not now", the standard label for putting a prompt off);
- only as the first word of an `h1`–`h6`, `nav`, `tab`, `menu_item`, `empty_title` or `dialog_title` of two to four
  words: your, my ("Your Favourites"). A field label "Your name" is not flagged: Microsoft recommends it over "Type
  your name". A branch that is "your" or "my" alone counts when another branch has two to four words: `{mine ?
  "Your" : "Shared"} playlists` comes out "Your ALT Shared playlists", whose first branch reads "Your playlists". A
  title with a comma or a sentence mark is a slogan, not a noun phrase, and is not flagged ("Your brand, your rules").

"very" and "quite" are not flagged in a `menu_item`, `input_value` or `cell`: there "Very low" … "Very high" names a
value on a scale (K5).

Matching is on whole words, so "pleased", "snowdrop" and "Nowcast" do not match.

### `slop/placeholder-as-label` (aside R8b)

A `placeholder` is flagged when its field has no visible label: its `control` cluster holds no scored `label` node,
and the nearest scored node before it, skipping `helper`, `tooltip`, `error` and `caption` nodes, is not a `label` in
the same cluster or in no control cluster (a `<label for>` lands in a body cluster and its input's placeholder in a
control cluster of its own; a label that belongs to another field's control cluster does not count).

Known false negatives, both for want of information in the screen: a `<legend>` is a `label` in a body cluster, like
a `<label for>`, so it counts as the label of the first field after it (`pl_fieldset_legend_pos`); and an input with no
text (`<label for="a">…</label><input id="a"/>`) gives no node, so the next field's placeholder takes that label
(`pl_label_for_textless_neg`).

### `slop/shared-lead-in` (aside R10)

For each aligned set in each sibling group (roles `cell`, `stat`, `input_value` and `breadcrumb` aside), take each
node's leading copy words up to the first marker or identifier word. One finding lists the whole set when

- the common leading words number at least two for three or more items, or at least three for a pair (a single shared
  word is parallel phrasing, "Add photo" / "Add video"), and
- every item has at least one word after the shared ones (otherwise the items repeat rather than share a lead-in), and
- the next word is not a number in every item ("Address line 1", "Up to 5 users": a numbered series, or a qualifier
  on a figure, K13), and
- not every item ends with "?" (questions open alike by grammar).

### `slop/index-numbering` (aside R12)

An index device is the whole text of a node, or its lead before a separator and a word:

| Family | Examples |
|---|---|
| `padded` | "01", "02 /", "(003)", "#04", "01 / 04" (a second number needs a slash, so "08:00" and "01.03" are not devices); as a lead "01 / Connect", "02. Configure", "01 / 03 Connect"; and, only to continue a padded column of the same role, a two-digit number 10–99 in the same shapes ("10", "10 Slipway"), so "01" … "12" is one run |
| `fraction` | "3 / 6" (the whole node) |
| `label:<word>` | step, phase, stage, section, question, no.: "Step 2", "PHASE 02", "Stage 1 of 3"; as a lead "Step 1: Create an account", "Step 2 of 3: Safety talk" (the total kept) |
| `plain` | "1", "2.", "3)" (the whole node), unless the next scored node is itself a device (a two-digit number with words after it, "12 months …", does not count) or there is none (a rating scale, calendar days) |

Nodes in `nav`, `tab`, `breadcrumb`, `menu_item`, `link`, `button`, `cell`, `stat`, `input_value` and `placeholder`
are skipped (a pager's "1 2 3", figures, table values). A run is a sequence of devices of one family and role whose
values go up by one, with the same total, in reading order, other nodes between them allowed, and that starts at 0 or
1 (an index counts from one: "06", "07" … are hours, and a run starting higher is K3's position in a sequence whose
earlier members are not on this screen); one finding lists each run of at least two devices (three for `plain`). A
total smaller than the value ("05 / 03", "3/2") is not a device, and a fraction or labelled device with a total ("1/2"
then "2/3") does not run on into one with another total. K3 is applied as far as text shows it: a lone "Step 2 of 4"
never forms a run, nor do `conditional` devices with a total (a wizard showing one "Step N of 3" heading at a time),
and a run is dropped when another node refers to it: for a labelled run, its word, singular or plural, before one of
its numbers or a list of them ("go back to step 1", "Steps 2 and 3 can wait"); for an unlabelled run (footnote
numbers), at least two of its numbers as footnote markers in one other node: a digit glued to a letter, ")" or "%", or
to punctuation that does not follow a digit ("a night1," from `<sup>1</sup>`; "4.1" and "9:30" are not markers), a
superscript digit or "[1]". Known false positive: day-first dates written "01/03", "02/03" read as padded devices with
a total and can form a run. Whether the items are steps to be done in order cannot be read from the text, so the rule
flags them and the label decides. "Level", "round", "lesson", "module", "chapter" and "part" are not label words: they
usually name content.

### `slop/exact-repeat` (R14; R9 between two titles)

The screen is split into windows exactly as the tokeniser does (`SPEC.md` §4). Nodes are grouped by normalised
string, leaving out `nav`, `tab`, `breadcrumb`, `menu_item`, `cell`, `input_value`, `label` and `placeholder` nodes,
footer links (`link` flagged `in_footer`, navigation by K5), site-bar links (`link` flagged `in_header` in a
cluster with no heading: a brand link or a top-bar "Sign in"; a hero's call to action in a header cluster with a
heading is still compared), nodes flagged `repeated` (the constant parts of one
template on sibling cards or rows, such as a per-row "Not set" or "/month") and nodes with fewer than two copy words.
In a group that spans two or more windows, `conditional` copies take no part, as flagged copies or as targets (they
may never be visible with the others, `LABELLING.md` §2a). Two copies are exempt from each other when one is a title
(`h1`–`h6`, `dialog_title`, `empty_title`) and the other a `button` or `link` (K11: "Delete playlist?" and "Delete
playlist"); when one is an `error` and the other a `link` or `list_item` flagged `in_state` (K12: an error summary item
and the inline error it links to); when one is a `tooltip` and the other a `button`, `link` or title (K12: the tooltip
names an icon-only control, and K11 holds between the control and the dialog it opens); or when both are flagged
`in_state` and sit in different clusters (two separately mounted states, such as
two dialogs, §2a). A copy is flagged when a non-exempt copy that outranks it (lower rank, or equal rank and earlier)
sits in another window, with the best such copy as `target`, unless a non-exempt copy that outranks it sits in its
own window: that repeat is within one window and left to the network. The code is R9 when both copies are titles, R14
otherwise.

## Codex rules

Lists come from `slop/codex/lists/*.txt`. `node rules/make_codex_lists.mjs` bundles the seven lists the rules read into
`src/codex-lists.js` (sections, entries, licence line and the source file's sha256), so the linter reads no files at
run time and the kit carries the lists with `rules/src/`; `--check` reports a stale bundle and a test fails on one.
`llm_excess_vocab.txt` (CC BY-SA 4.0) is not bundled. Each rule names the sections it reads and, in an `EXCLUDE`
table, every entry it leaves out with the reason (an entry with ordinary, deliberate uses in shipped UI); an exclusion
that names no list entry fails at import, so a list edit cannot silently void it.

**Matching** (`src/codex.js`). The list syntax of `llm_constructions.txt`: words folded as elsewhere (NFKC, lower
case, marks and inner apostrophes dropped: "it's" is "its"); `{x}` `{y}` `{z}` one to eight words (commas allowed
inside, never a sentence end); `{w}` one word; `{n}` one literal number, optionally followed by k, m, thousand or
million (an interpolated value never counts); `^` and `$` anchor at the start and end of the text. The token stream
keeps the punctuation that shapes a construction (`, . ? ! : ; % + /`), so "not just {x}, it's {y}" needs its comma
and "^tired of {x}?" its question mark; an em or en dash reads as a comma, "…" as a full stop, "&" as "and". The last
word of a pattern also matches the first part of a contraction ("sure! here" matches "Sure! Here's"). Identifier
words after a VAR marker become one token that a slot may cover and a literal word never matches. Each ALT branch is
matched on its own. A hit inside a longer hit counts once, as the longer.

**Roles.** "Prose roles" are all scored roles except controls, labels, field values, cells and stats (`button link
nav tab menu_item breadcrumb label placeholder input_value cell stat`). "Titles" are `h1`–`h6`, `subtitle`,
`eyebrow`, `empty_title`, `dialog_title` and `badge`.

| Rule | Codes | Severity | Fires on |
|---|---|---|---|
| `slop/chatbot-residue` | L44 | error | any scored node containing an entry of `llm_residue.txt` §L44 ("Certainly!", "Sure! Here's", "I hope this helps", "Let me know if you'd like", "As an AI language model", "As of my last update"). 29 entries are excluded as ordinary UI copy ("Would you like to…" in a dialog, "Is there anything else…" on a form, "Here are some…", "Below is a…", "for illustrative purposes", "Todo"), and "Lorem ipsum" / "dolor sit amet" are left to `slop/placeholder-copy`. "As an AI" counts only as "As an AI, I…" (an AI product may write "As an AI, the assistant can make mistakes."), and the reply phrases in `SOFT` ("Let me know if you'd like…", "Here's a revised…", "Here's the updated…") not in `body`, `subtitle` or `caption`, where a person or a planner writes them on purpose. |
| `slop/placeholder-copy` | L45 | warning | an entry of §L45 ("Acme Inc", "Jane Doe", "(555) 123-4567", "123 Main Street", "Anytown", "example.com", "[Your name]"; bracketed entries on the raw text). Never in `placeholder` or `input_value`, nor in a branch that gives an example ("e.g.", "for example", "such as"): a format hint is not residue. Slot names ("Your company", "Company name", "Logo") only in a copyright line or as a header link's whole text (the wordmark). Plausible real names and job titles are excluded. Also "Lorem ipsum" and "dolor sit amet" (in §L44 of the list; CODEX.md L45 names them as template text). Not reported: an entry right after "like", "not" or "instead of" ("Like you@example.com"); the reserved example domains (example.com, .org, .net) in a `helper` or `tooltip`; "John Smith" and "Jane Smith" except as a byline (opening a node of at most eight words, or beside a quotation), so "Photo by John Smith on Unsplash" is not reported, while "John Doe" and "Jane Doe" always are. |
| `slop/llm-construction` | L09–L19 (the section's id) | warning | a kept pattern of `llm_constructions.txt` in a prose role: negative parallelism ("It's not just X, it's Y", "No X, no Y, just Z"), the audience hedge ("Whether you're X or Y", "for teams of all sizes"), the false range ("From X to Y, Z"), tricolon slogans ("Simple. Smart. Yours."), rhetorical openers ("Tired of X?", "X? We've got you covered"), the pivot ("Say goodbye to X", "X, reimagined"), "so you can focus on what matters", the trailing-participle tails ("…, making it easy to", "…, so nothing slips through the cracks"), grandiose openers and closers ("In today's fast-paced world", "The future of X is here", "Everything you need, all in one place"), significance framing ("at the heart of") and staccato closers ("That's it.", "The best part?"). 28 patterns are excluded (`EXCLUDE`: real contrasts such as "X, not Y", help text such as "Need help? …" and "Want to …?", "Meet the team", "No more results", "Just now."). Suffix and slogan forms ("X, reimagined", "made simple", "Welcome to X", "The future of X", "Everything you need to X", "X and Y alike") count in titles only, and "Welcome to X" not in a dialog or state view (onboarding). In L09–L11 a slot holding a number is a real contrast or range, and in L11 so is a date or time word ("From 9 to 5"). |
| `slop/stock-cta` | L26, L27 | warning | L26: a `button` or `link` whose whole label is a stock CTA of `llm_cta_cliches.txt` §L26 beyond `slop/vacuous-action`'s list ("Start your journey", "Explore features", "See it in action", "Let's go", "Claim your spot", "Get yours"). Labels R7 or R15 already report, labels that state the cost ("Start for free") and real next steps ("Book a demo", "Contact sales", "Join the waitlist") are excluded. L27: a CTA-band headline of §L27 in a prose role ("Ready to get started?", "What are you waiting for?", "Ready when you are", "Start building today"). |
| `slop/empty-claim` | L28, L29, L30, L32 | info | a claim of `llm_claims.txt` in a promotional role (titles, `body`, `list_item`, `caption`, `stat`, `button`, `link`): speed and ease promises ("Get started in seconds", "No coding required", "Up and running in minutes"), invented precision ("10x faster", "Trusted by 10,000+ teams", "Rated 4.9/5"), vague social proof ("Loved by thousands", "Don't just take our word for it") and false urgency ("Limited time offer", "Act now"). A `{n}` counts only when the number is round (one or two significant figures, or a run of nines) and not the top of a range ("2-4x faster" is a measured result); urgency in a `dynamic` node is real. A branch that states its source ("in our benchmarks", "in tests", "according to…", "measured on…", "source:") is a measured claim: only its urgency is reported. §L31 reassurance ("No credit card required") is the network's (K6). |
| `slop/promo-vocab` | L01, L02, L03, L08 (first hit) | info | two or more distinct entries of `llm_promo_vocab.txt` in one branch ("Seamlessly supercharge your planning", "A powerful, intuitive editor"). One word is weak evidence; words with literal uses (smart, unique, tailored, automatically, workflow, experience, solution, landscape, …) and R15's filler are excluded. Not in `placeholder`, `input_value`, `cell` or `menu_item`. |
| `slop/sparkle-emoji` | R11g | info | an emoji of `llm_glyphs.txt` §L46 leading a node with words ("✨ Smart search", "✅ Unlimited projects"); variation selectors ignored. Not in a `cell` or `input_value` (a table's ✅ is the value); an emoji-only node is the network's (R11); leading dots are `slop/repeated-chrome`'s. |
| `slop/em-dash-density` | L48 (screen) | info | three or more nodes on a screen using an em dash (or "--") between two words as a clause connector. A spaced en dash (the British parenthetical dash), a spaced hyphen, a lone dash in a cell, a dash before a byline and a range of numbers ("2019 — 2021") do not count; nor do cells, field values and stats. |
| `slop/exclamation-density` | L51 (screen) | info | three or more nodes on a screen ending a sentence with "!" (§L51), not counting cells, field values and stats. |
| `slop/pleonasm` | U51 | warning | an acronym followed by the word it stands for (`ui_tech_pleonasms.txt`: "PIN number", "PDF format", "ATM machine", "FAQ questions", "2FA authentication"); "LED light" (a product) and "PIN code" (the Indian postcode) are excluded. |

Not implemented here, by the codex's own mapping: patterns whose redundant instances are the network's (L33–L43,
L47), patterns that need meaning (L35 paraphrase, L41, L43), the case test (L49, `LABELLING.md` §2: case never
decides) and the weak single-word lists (L04–L06 excess vocabulary, L20–L25 filler and hedges, whose words carry the
consequence in UI copy, `codex/ui.md` §5 note 7). Firing rates as a guide: on the 20,000 generated dev screens
(362,131 nodes) the codex rules fire 977 times, all on planted LLM patterns in a spot check; on the 22 real reference
screens, never.

## Lists and their sources

Every list is in `src/lists.js` (and the index-number grammar in `src/text.js`), each entry citing the line of
`research/anti-slop-guides.md` or `research/local-skills.md` it rests on, or marked as added. Added entries: "find out
more", "explore more", "tap here", "this link", "click", "tap" (vacuous labels); "tap here", "click this link", "press
here" (gesture phrases); the five gesture leads; "whoops" (filler); and the K13 qualifiers (`QUALIFIERS`), which rest
on `LABELLING.md` K13 rather than the research notes.

## Tests and fixtures

```
npm run test:rules                 # in slop/: node --test rules/test/rules.test.js
uv run pytest rules/tests          # the CLI called from Python
node rules/make_codex_lists.mjs    # after editing codex/lists/*.txt (--check: is the bundle stale?)
```

`fixtures/make_fixtures.mjs` holds 118 invented screens as compact tables and writes `fixtures/screens/*.json` (in the
extractor's serialised layout, with the VAR and ALT markers and `vw`) and `fixtures/expected.json`, which lists, for
each screen, the rules it is a positive or near-miss negative for and the exact findings expected from every rule.
Fifteen of them are the screens of the adversarial review of 2026-11-10, one per bug it found, and 30 come from the
second review the same day: 18 found bugs (b01–b18; b13 and b14 record known false negatives), four the lead's
decisions (j01–j04: footer links out of the arrow count, site-bar links out of exact-repeat, K13 qualifier badges out
of sibling-device, the filler list unchanged), five controls (c01–c05) and three guards for the fixes (c06–c08).
Twenty more (`cr_`, `ph_`, `lc_`, `sc_`, `ec_`, `pv_`, `se_`, `ed_`, `ex_`, `pn_`) are a positive and a near-miss
negative for each codex rule; the negatives carry the exclusions (real contrasts, help text, onboarding greetings,
measured ranges, year ranges, format hints). `fw_markers_mixed` has no `vw`, to keep the fallback covered. Run
`node rules/fixtures/make_fixtures.mjs` after editing the tables and `--check` to see whether the files are stale; the
tests fail on a stale file. The tests also run every rule over the extractor fixtures, the tokeniser test screens and
the labeller fixtures, twice, and check that the output is the same. All copy in the fixtures is invented; never paste
strings from a real repo.

## Not done here

The ESLint plugin: under ESLint the rules need the extractor to turn the linted file into a screen first, and the
extractor's Swift parser loads asynchronously while ESLint rules run synchronously. `RULES` carries the ids, codes
and descriptions the plugin will map one to one; packaging is part of `SPEC.md` §14.
