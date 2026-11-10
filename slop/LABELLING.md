# HypaNet Slop: labelling guide

The definition of "redundant copy" for the slop experiment (`SPEC.md`). Matthew labels with it, the second labeller
labels with it, Codex's OOD generator and rules engine are briefed with it, and Fable receives it in the authoring
packet. It is built from the research notes in `research/`. Pattern codes are those of `research/anti-slop-guides.md`
§D; `research/local-skills.md` §14 numbers its rules differently.

Worked examples here are invented, the motivating pause menu, or from the reference repos (HypaChef, Mosaic,
PersonalHomepage). Strings from sealed repos must never appear in this file. This draft was written by a harness
session that had read the unredacted corpus note, so the file must pass the sealed n-gram check (`SPEC.md` §9) before
it is frozen.

## 1. The principle

Each visible piece of copy should do one job that no other piece on the screen already does. Redundancy is repeated
**information**, not repeated **words**: the style guides (Microsoft Writing Style Guide, Shopify, frontend-design)
require one name for one thing everywhere, and the slop catalogues (Wikipedia "Signs of AI writing", unslop rule 11)
list swapping in synonyms as a tell. So "OPERATION ON HOLD" above "PAUSED" is redundant although the two share no
word, and a "Playlists" heading over a "New playlist" button is not, although they share a stem.

**Deletion test** (deslop-ui, Hallmark gate 54): delete the node and read the screen again. If the reader loses no
fact, destination, identifier, constraint, consequence, live state or instruction, the node is redundant. In the
procedure (§4) this test comes last, after duplicates, K rules and narration have been settled.

## 2. What is labelled

Every node the extractor emits for a screen (`SPEC.md` §4), one record per node. Not labelled (context only, never
scored): `aria`, `alt`, `code` (the code line in the comment track), nodes flagged `sr_only`, and nodes with no literal
words (only `VAR` and identifier words, such as an unresolved translation key; recorded with `aside: ["UNRESOLVED"]`
in the screen's count). Letter case and styling never decide a label: game menus use capitals by convention. Styling
is evidence of a role (an eyebrow), never of a class.

### 2a. Evidence and visibility

Label from the screen JSON and its source file only, plus the product's kind as named in the repo README (one line,
recorded per repo before labelling; for a fixture, its fixture note). Do not use other files, the running app, or
knowledge of the game or page behind an overlay.

Nodes are visible together unless they sit in different branches of the same conditional, or in separately mounted
states (loading, error, empty, ready, or two different dialogs). Nodes in different branches never echo each other. A
target must be a node in the same window (`SPEC.md` §4). If the repeated information lives only in another file, label
the node as if that file did not exist, and record `aside: ["X-FILE"]`.

## 3. Classes

| Class | Meaning | Target |
|---|---|---|
| `KEEP` | Does a job no other node does, or is one of the K cases in §6 | none |
| `ECHO` | Restates information already shown by another node visible with it (§2a) | the node it repeats (required when one node holds the information; `null` when it is spread over several) |
| `NARRATE` | Explains how the visible UI works, what a visible control will do, or what the user can already see (layout, a literal count of items that are all visible, "below you'll find") | the control or node it explains, or `null` |
| `DECOR` | Carries no information: a label that only names what comes next or the section's genre (§6 K1), a dot or pill on static content, meta strips whose items are not facts about this product or page (decorative coordinates, local time, "Founded 2025" on a product launched last year) | none |

`ECHO`, `NARRATE` and `DECOR` together are **REDUNDANT**. `UNSURE` may be used when the labeller cannot decide after
§4; those nodes are counted and reported but not scored. A version number on an about, settings or release screen is
`KEEP`.

## 4. Decision procedure (in order)

1. **Context only.** Skip the roles and nodes listed in §2.
2. **Duplicates.** If another node visible with this one (§2a) gives this node's information, compare ranks (§5). If
   this node ranks higher (or equal and earlier), it is not an echo of that node: treat the other copy as deleted
   and go on. If it ranks lower, it is an `ECHO` candidate with that node as target.
3. **K rules.** If a K rule in §6 applies, label `KEEP` and stop.
4. **Narration.** If the node describes the visible interface, its layout, a literal count of visible items, or what
   a visible control does, apply the consequence test (§7). If the test fails, the node is a `NARRATE` candidate
   with the control or node it explains as target.
5. **Partial.** A candidate from step 2 or 4 that also carries a clause which would be `KEEP` on its own under steps
   3–4 is `KEEP` with `partial: true` ("Completed research outputs. Not used as grounding input."). Clauses that are
   themselves narration or restatement do not count as new information. Otherwise the candidate takes its label.
6. **Deletion test, last.** For a node with no candidate label: delete it. If the reader loses a fact, destination,
   identifier, constraint, consequence or live state that steps 2–5 did not already account for, label `KEEP`;
   otherwise `DECOR`.

Set `conf: "clear"` when the steps decide it at once and `conf: "judgement"` when a step was arguable. Set `ct: true`
whenever §7 decided the label, in either direction.

## 5. Precedence: which copy of a duplicate is flagged

From Microsoft's UI text guide: "leave full text in main instructions and interactive controls, and remove any
redundancy from the other places." Higher rank is kept; the lower-ranked node is labelled and points at the higher.

1. Interactive controls (`button`, `link`, `nav`, `tab`, `menu_item`, `breadcrumb`, `label` of an input) and main
   headings (`h1`, `dialog_title`, `empty_title`)
2. `h2` > `h3` > … > `h6`
3. `body`, `list_item`, `dialog_body`, `empty_body`, `toast`, `error`, `stat`, `cell`, `input_value`, a `label` not
   attached to a control, `other`
4. `subtitle`, `helper`, `caption`
5. `eyebrow`, `badge`, `tooltip`, `placeholder`

`comment` is always the flagged side and points at `code`. Equal rank: flag the later node in reading order. Across
sections, the higher-ranked copy is kept; at equal rank the first occurrence is kept and later copies point at it.
A stack of three or more nodes naming the same thing keeps the highest-ranked one, flags each other one, and lists
the other flagged copies in each record's `equiv`.

## 6. KEEP rules (never redundant)

- **K1** An eyebrow or caption that names a product or content category (Camping, Security, Recipes), a parent
  section in a multi-page document or flow ("Section 3: Your details", a GOV.UK caption), or a date or issue number.
  K1 does not cover **section-genre words**, which name the kind of section rather than its subject: Features,
  Pricing, FAQ, About, About us, Overview, How it works, Testimonials, Contact, Get started, Products, Services, Our
  team, Why us, Capabilities, and their close variants (this list is frozen with the guide). A section-genre eyebrow
  is `ECHO` → the heading when the heading contains the same word or its stem, and `DECOR` otherwise.
- **K2** A dot, pill or badge whose text or colour is bound to data (`stateful`), that differs between sibling items,
  or that is part of a legend ("Healthy / Degraded / Down"). Constant words beside a bound dot are `KEEP`: the node
  carries live state.
- **K3** Numbering when the items are steps to be done in order, when it gives position in a sequence whose other
  members are not on this screen ("Step 2 of 4"), or when another node refers to the number. Other index numbering is
  `KEEP` for the network with `aside: ["R12"]` for the rule layer.
- **K4** Helper text that answers a real question: format, limit, eligibility, timing, scope, what happens to data, a
  hidden input (§7) ("Every quantity is scaled for this many." under a servings stepper).
- **K5** Repeated values in tables, navigation, breadcrumbs, tabs, menus and form option lists; a nav item or tab that
  names the current page.
- **K6** One global notice (a single disclaimer, a single "No card needed" beside the primary sign-up).
- **K7** A confirmation body that states a consequence ("This also deletes 12 saved exports.").
- **K8** An empty or error state body that gives the cause or the next action.
- **K9** Consistent naming: the same term reused for the same thing (h2 "Playlists" and button "New playlist").
- **K10** Comments: licence headers, pragmas (`prettier-ignore`, `eslint-disable`), doc comments on an exported API
  that state a contract, issue or RFC links, comments that say why.
- **K11** A heading and a control that acts on the thing the heading names are never echoes of each other: a question
  title and its confirming button ("Delete playlist?" / "Delete playlist"), a state heading and the action that leaves
  that state ("Paused" / "Resume"), a page `h1` and the nav item, tab or breadcrumb for the same page.
- **K12** Accessibility and validation copies: an error summary item and the inline error it links to; a tooltip or
  `title` on a control with no visible text, or on a `truncated` node; a label that is also the page heading (GOV.UK
  label-as-heading).
- **K13** A qualifier attached to each of several sibling figures or offers (unit, currency, billing period, tax
  status, "from", footnote marker), and legal or regulatory text that must appear beside each item it qualifies, in
  every copy.
- **K14** Branches of one conditional or plural form are alternatives, never echoes of each other.

**Static status claims.** A status word in fixed markup ("● Online", "● Live", "Available", "Now in beta", no data
binding) is a claim that cannot change. If it is the only statement of a real, dated fact (a single "Early access"
pill on the one product in early access), apply the deletion test to its words. Otherwise it is `DECOR` (R11).
**Glyph on an informative node.** A dot, sparkle or emoji glyph that leads a node whose words are `KEEP` leaves the
node `KEEP`, with `aside: ["R11g"]` for the rule layer.

## 7. The consequence test (NARRATE against K4)

Copy that states what a visible control does is `KEEP` only if both hold: (a) the consequence is not what the
control's label and the kind of product (§2a) already lead the user to expect, and (b) the user would act
differently, or could act at all, for knowing it: loss, cost, irreversibility, timing, scope, privacy, or an input or
gesture that no visible control, label or platform convention exposes (keyboard shortcut, long-press, drag, swipe
action, multi-select modifier). Otherwise it is `NARRATE`. Ordinary platform behaviour (scrolling, hover
highlighting, tapping a visible button, a link opening a page) is never non-obvious. Every label this test decides
gets `ct: true` and is scored in the stretch column, not the headline (`SPEC.md` §10).

## 8. Worked cases

The motivating pause menu (an AI-built game UI; fixture note: "first-person game"):

| Node | Role | Label | Target | Pattern | Why (step) |
|---|---|---|---|---|---|
| OPERATION ON HOLD | eyebrow | ECHO | PAUSED | R1 | Step 2: the h1 (rank 1) gives the same state; nothing new survives step 5 (clear) |
| PAUSED | h1 | KEEP | | | Step 2: it outranks the eyebrow; step 3: K11 against RESUME; step 6: deleting it loses the stated state |
| Resume recaptures the mouse. | subtitle | NARRATE | RESUME | R6 | Step 4, `ct: true`: the screen is a game pause menu; resuming play with the mouse is the label's ordinary meaning; the line names no loss, cost, setting or hidden input (judgement) |
| RESUME, SETTINGS, CONTROLS, RESTART MISSION | button | KEEP | | | Step 6: controls |

Note on PAUSED: the user underlined it too. Microsoft's guide allows dropping a main instruction that is "obvious
from the context of the UI", but that context is the frozen game behind the overlay, which §2a rules out as
evidence. This guide labels it `KEEP`; `SPEC.md` §16 asks Matthew to confirm.

The contrast case, HypaChef's pause overlay: "Paused" / "Timers are stopped until you resume." / "Resume cooking".
The body is `KEEP` (step 4 then K4, `ct: true`, judgement): in a cooking app the timers track real food, so whether
they keep running while paused has a cost.

Short cases (invented):

| Copy | Label (step) |
|---|---|
| eyebrow "Help" over h2 "Help centre" | eyebrow ECHO → h2 (R1; step 2) |
| eyebrow "FEATURES" over h2 "Tools for small kitchens" | eyebrow DECOR (R2; K1 genre word not in the heading; step 6) |
| eyebrow "Features" over h2 "Features that scale" | eyebrow ECHO → h2 (R1; K1 genre word in the heading) |
| eyebrow "Camping" over h2 "Trailhead 2 tent" | KEEP (K1) |
| h2 "Settings" / subtitle "Manage your settings here." | subtitle ECHO → h2 (R3; step 2) |
| label "Postcode" / helper "Enter your postcode" | helper ECHO → label (R5) |
| label "Email" / helper "We send the receipt here" | helper KEEP (K4) |
| button "Add line" with tooltip "Add line" | tooltip ECHO → button (R5) |
| icon-only button with tooltip "Add line" | tooltip KEEP (K12) |
| label "Name" / placeholder "Name" | placeholder ECHO → label (R8) |
| dialog_title "Delete playlist?" / button "Delete playlist" | both KEEP (K11) |
| h1 "Billing" and tab "Billing" in the same file | both KEEP (K11) |
| error summary "Enter a postcode" and the inline error "Enter a postcode" | both KEEP (K12) |
| "● New" pill on all six feature cards | each pill DECOR (R11); screen aside R19 |
| "● Online" whose text and dot colour are fixed in the markup | DECOR (R11; static status claim) |
| "● Online" whose dot colour is bound to the connection state | KEEP (K2) |
| "Early access" pill on the one product in early access | KEEP (static claim, sole statement of a fact; step 6) |
| h3 "✨ Smart search" | KEEP, `aside: ["R11g"]` |
| "01 /", "02 /", "03 /" above three feature cards in a row | each KEEP, `aside: ["R12"]` (K3) |
| "Step 2 of 4" above a form | KEEP (K3) |
| h2 "Setup" / subtitle "Four quick stages to your first invoice." above a 4-step list | subtitle NARRATE (R16; literal count of visible items) |
| "340 invoices" above a paged table | KEEP (count bound to data) |
| "/month" on each of three pricing cards | each KEEP (K13) |
| empty state "No invoices" / "There are no invoices yet." | body ECHO → title (R4) |
| empty state "No invoices" / "Create an invoice to see it here." | body KEEP (K8) |
| "Free for 21 days" under four sign-up buttons | first KEEP (K6), later three ECHO → first (R14) |
| helper "Hold Shift to select a range." | KEEP (K4, hidden input; `ct: true`) |
| sub "Scroll to explore" | NARRATE (R6; ordinary platform behaviour; `ct: true`) |
| `# Open the cache` above `cache = open_cache(path)` | comment ECHO → code (R18) |
| `{/* Price table */}` above `<table>` | comment ECHO → code (R18) |

## 9. Pattern codes

Labels carry the pattern code for per-pattern recall. Codes outside the network's scope go in `aside` (recorded for
the plain-code rule layer, `SPEC.md` §8, and never scored against the network).

| Code | Pattern | Class |
|---|---|---|
| R1 | Kicker or eyebrow restating the heading | ECHO |
| R2 | Decorative label: names only what comes next, or the section's genre (K1 list); meta strip | DECOR |
| R3 | Subtitle restating the heading | ECHO |
| R4 | First or last line of a section, or an empty-state body, restating its heading | ECHO |
| R5 | Helper, tooltip, card body or bold lead-in restating its label | ECHO |
| R6 | Explaining the obvious (a self-evident control or state; usually `ct: true`) | NARRATE |
| R8 | Placeholder repeating the label | ECHO |
| R9 | Title repeated across window, page, hero and card headings (not nav, tabs, breadcrumbs: K5, K11) | ECHO |
| R11 | Decorative dot, pill, badge or emoji marker; static status claim | DECOR |
| R14 | The same fact or call to action repeated across sections | ECHO |
| R16 | Copy describing the layout, the design or the screen itself, including a literal count of items all visible | NARRATE |
| R17 | Closing line summarising what was just shown | ECHO |
| R18 | Comment restating the next code line | ECHO |
| aside R7 | Vacuous action label ("Click here", "Learn more") | rule layer |
| aside R8b | Placeholder used instead of a label | rule layer |
| aside R10 | Same lead-in on every sibling item | rule layer |
| aside R11g | Dot, sparkle or emoji glyph leading a KEEP node | rule layer |
| aside R12 | Index numbering not covered by K3 (taste: tasteskill research.md:162 found it more common on top-rated sites, 38% against 21%) | rule layer |
| aside R13 | Forced triplet that restates nothing else on screen | not scored in r1 |
| aside R15 | Filler words ("please", "now", "simply") | rule layer |
| aside X-FILE | Repeats information held only in another file | reported |
| aside UNRESOLVED | Node with no literal words (§2) | reported |
| screen aside R19 | Repeated device: the same eyebrow, pill, dot, arrow suffix or index-number device on most sibling sections, cards or rows (the labeller lists the node ids; `SPEC.md` §5) | rule layer (`slop/repeated-chrome`) |

Out of scope entirely: promotional vocabulary, invented statistics, em-dash density, default visual styling.
