# MODEL.md — the hand-written slop network

Rebuild: `./python3 work/build.py` (reads `work/lexicon/*.txt`, writes `vocab.txt`, `artifact.npz`, `meta.json`;
also `directions.json` for `./inspect` and `dims.json`, the full name → dimension table). It prints the artefact
hash; running it twice gives the same hash. Pointer head: layer 3, head 0.

Size: 1,077 vocabulary stems, 128 residual dimensions, 8 + 3 + 1 attention heads in layers 1-3 (layer 4 attention
is all zero), 378 + 54 + 28 + 3 MLP units, about 141,000 non-zero parameters (most of them the ±1 concept codes of
the 6,144 hash buckets). The cluster embedding is all zero.

No weight, threshold or vocabulary entry was fitted or searched: each number is derived in a comment in `build.py`
from what the mechanism computes (code-match scores, counts that are integers or halves), and each revision below
was made for a stated reason and then looked at.

## 1. The idea in one paragraph

A node is redundant when everything informative in it is already given by a higher-ranked node that is visible
with it. The network measures exactly that. Every word token carries a **concept code** (synonyms share one code)
and an **information weight** (0 for function and filler words, 1 for content and unknown words). Layer 1 asks, per
token, "is my concept in a rank 1-3 node just before / just after me?". The layer-1 MLP turns that into per-token
*novelty* (informative and not covered) and *anchor hits*. Layer 2 sums these per node at the node's `EON`, and
separately checks whether an earlier node has exactly the same bag of concepts (a repeated title, call to action
or fact). Layer 3 is a small set of rule units, one per pattern of `LABELLING.md`, each an AND of a few booleans
with vetoes. Unknown words are informative by default, runtime text (`VAR`, `dynamic`) vetoes every rule, and
roles the guide never flags (button, nav, tab, menu item, cell, stat, …) have no rule at all: the default is `KEEP`.

## 2. Lexicon (`work/lexicon/`, written by hand from general UI vocabulary; 1,077 stems in `vocab.txt`)

| File | Meaning | Effect |
|---|---|---|
| `function.txt` | articles, pronouns, auxiliaries, prepositions | no code, weight 0: invisible to matching and novelty |
| `generic.txt` | filler that adds nothing when restating (all, every, manage, view, see, enter, receive, here, things, welcome, …) | code, weight 0 |
| `semi.txt` | half-weight words (only, new, set, plan, back, add, create, account, data, choose, adjust, …) | weight 0.5: one alone is not news, two are |
| `groups.txt` | one synonym group per line (settings/preferences/options; pause/hold/frozen; login/sign-in/log; photo/picture/image/avatar; edit/change/update; …), about 240 groups | members share one concept code |
| `deictic.txt` | below, underneath, above, bottom | layout pointers (R16/R6), weight 0 |
| `uinoun.txt` | button, page, card, list, option, tab, count, … | interface nouns (R16/R6); not narration novelty |
| `action.txt` | click, tap, press, choose, toggle, enable … | R6 |
| `gesture.txt` | scroll, swipe | both an action and an interface word, so "Scroll to explore" is narration on its own |
| `promo.txt` | new, fresh, hot, popular, featured, trending, top, best, premium, exclusive, limited, updated, improved | the words of a decorative pill (R11 repeated pill) |
| `timeword.txt` | today, now, tonight, current(ly) | filler in a sentence, but a date category as a kicker (K1) |
| `narrverb.txt` | explain, cover, describe, outline … | R16 |
| `genre.txt` | section-genre words (features, pricing, faq, testimonials, contact, products, services, capabilities, benefits, questions, start) | no "class-external" information (R2) |
| `status.txt` | online, live, ready, running, connected, open, synced, available, active, operational, system(s), normal | static status claims (R11) |
| `summary.txt` | nutshell, short, covers, recap, summary, sum, wrap, that, those | closing lines (R17); weight 0 |
| `unit.txt` | per, month, year, user, seat, billed, vat … | K13 qualifiers: a repeated price qualifier is not a "repeated fact" |
| `dem.txt` | here, these, this | R16 |

In `build.py`: number words two…twelve and the ordinals second…fifth share the code of the digit token;
`n s e w` are compass letters. Everything not in the lexicon (every hash bucket) is a content word: own code,
weight 1. Conjunctions that open a new clause (if, unless, because, but, until, since, before, after, while) are
deliberately *not* function words: an unmatched one counts as information ("…, if you have one"). Negations
(no, not, none, nothing, zero, never, empty, n't forms) are one synonym group with weight 1, so "No albums so
far; none at all" is covered by "No shared albums" but an unmatched negation is news. Known homograph traps
handled: "one" stems to "on", "use" to "us" (both function words); "may" is left a content word because of the
month; "personalise" (stem "person"), "organise" ("organis" = organisation) and "activate" ("activ" = active) are
left out of the filler / action lists.

## 3. Residual allocation (128 dims; names in `work/dims.json`)

| Dims | At tokens | At `EON` |
|---|---|---|
| `code0..14` | ±1 concept code (0 for function words) | after L1 the node's code sum; after L2 the difference to the nearest earlier candidate copy (0 = same bag) |
| `one`, `is_eon`, `is_bos`, `first` | constants / position marks | |
| `wi`, `mt`, `wn`, `wx` | information weight; has-a-code; narration weight; class-external weight | per-node sums (the `wn` dim instead holds the boolean `b_xs`) |
| `num var deic uin act narr status summ dem promo coord unit` | lexical class flags | per-node counts |
| `m_p m_ph m_pl m_pc` | concept found in a rank 1-3 node at −4..−1 (any / heading or error title / label / control) | 0 from the heads; three of these dims are reused for the booleans `b_novp2`, `b_novx1`, `b_word` |
| `m_n m_nc`, `m_n1h`, `m_g` | found at +1..+3 (any / control); in the heading at +1; in any earlier heading, button or text line | 0 |
| `mc0..11` | "matched code": the token's code if it was matched through a gate that fits its role, else 0 | the sum: the pointer's query |
| `nov_p nov_n nov_x` | novelty against previous / next / both (narration weight) | per-node sums |
| `hit_p hit_n hit_nc hit_nc0` | role-gated anchor hits | per-node sums |
| `g_*`, `r_button`, `r_unfl`, `r_err` | role group of the node (role embedding) | same |
| `f_*` | flags used by rules and key eligibility | same |
| `b_*` | 0, except three dims reused for the pointer flags `rf`, `rb`, `f_any` | booleans (static from L1 MLP, dynamic from L2 MLP) |
| `norm2`, `dl1` | 0 | squared length of the code sum (10 bits); L1 distance of the duplicate comparison |
| `o_echo o_narr o_decor` | 0 | class evidence read by the output head |

The cluster embedding is unused (all zero): locality comes from the signed node distance in the attention bias.
Dims are shared between token positions and `EON` positions on purpose ("per-token value at tokens, per-node sum
at `EON`"); the six reuses named above pair a token-only quantity with an `EON`-only one, and every MLP unit is
gated to the kind of position it is meant for.

## 4. Layers

**Layer 1 attention** (matching heads: score = 6 per agreeing minus 6 per disagreeing code bit over 14 bits, so an
identical code scores 84 and any other at most 72; `BOS` = 79 is the threshold; eligible keys only)
- H0 *previous*: keys in rank 1-3 nodes (headings, labels, buttons, links, body, error) at −4..−1; +3 for anchors.
  Never keys: dynamic, sr-only. A key inside a state cluster is invisible to a query outside one (§2a).
- H1 *next* (+1..+3, same keys), H2 *next heading* (+1 only, headings), H3 *any earlier* heading / button / text
  line that is not conditional, dynamic, sr-only, in nav or in a table (only feeds the pointer query).
- H4 / H7 reverse matches for the pointer: `rf` = this token is repeated by any later lower-rank or
  duplicate-class node; `rb` = by an eyebrow, tooltip or narrating line in the three nodes before.
- H5 static per-node counts, H6 per-node code sum → `EON` (count heads: 1e-4 per own token, the rest on `BOS`,
  ×1e4 on the way out; only `EON` queries, only token keys).

**Layer 1 MLP** (token units are switched off at `EON`, `EON` units at tokens)
- `nov_p = relu(wi − 2 m_p)`, `nov_n`, `nov_x = relu(wn − 2 m_p − 2 m_n)`; for tooltips filler words count too.
- hits: `relu(m_class + role − 1 − num)` per role group (subtitle/helper/caption/placeholder/small body → heading
  or label before; tooltip → control before / after; list line, state body, card label → heading before; eyebrow →
  heading at +1; narration → anything named below; subtitle whose *first* word is a control below).
- matched code `mc_i = relu(±code_i + 3·m + 3·role − 6)` through four gates (lower-rank roles → previous; eyebrow
  → next heading; tooltip/narration → control below; heading and link → any earlier copy).
- `f_any = clamp(rf + rb)`, the pointer's key gate.
- static booleans from the counts (`b_var`, `b_wx`, `b_status`, `b_deic`, …), each two ReLUs.
- `norm2 = Σ x²` over 10 code-sum entries, exact for |x| ≤ 11 by `x² = |x| + 2 Σ_k relu(|x| − k)`.

**Layer 2 attention**
- H0 per-node sums of the token features; H2 per-node sum of the matched code.
- H1 *duplicate head*: `EON` of A → `EON` of the earlier eligible node B nearest to A in code sum
  (score 3·(cs_A·cs_B − |cs_B|²/2), i.e. −1.5·|cs_A − cs_B|² up to a constant) and subtracts cs_B from A's code
  dims. Eligible: heading → earlier heading (main heading → earlier main heading only); link → earlier button;
  helper/caption/subtitle → earlier text line; badge → earlier badge, card label → earlier heading. Never
  conditional, dynamic, sr-only, nav or table keys; and, except for the badge / card-label class, never a key
  inside repeated sibling items (two list entries with the same title are data).

**Layer 2 MLP**: dynamic booleans (`b_hitp`, `b_novp` …) and `dl1 = Σ|code_i|` at `EON`.

**Layer 3 attention**, H0 = the pointer head (writes nothing): `EON` of A → token u with score
2·(mc-sum of A · code of u, 11 bits: 22 per shared concept) − 24 unless u is flagged "repeated by a node that could
be restating it" (`rf`: by any later lower-rank or duplicate-class node; `rb`: by an eyebrow, tooltip or narrating
line just before; an eyebrow only accepts `rb`) + 3 (anchor node) + 6 (u has the role an earlier copy of A would
have) + 3 (A has an action verb or a layout word and u is a button) − 0.05·|distance| (cap −0.6) − 1 (u later than
A); `EON` keys excluded; `BOS` 7. Distance and direction only break ties: attention is summed per node, so a far
node holding two of A's concepts must still beat a near node holding one.

**Why this head is the pointer** (`meta.json`: `[3, 0]`): it is the only head whose attention from an `EON` is
meant to land on another node as a whole. The duplicate head (layer 2) also points at a node, but only for exact
copies; the layer-1 heads work per token. The pointer needs the per-node matched-code sum, which exists after the
layer-2 count head, so it sits in layer 3.

**Layer 3 MLP**: rules (`relu(Σ positives − (need−1) − 100·Σ vetoes − soft)`). Evidence added per fired rule:
3 = word for word (same bag as another node; tooltip inside its control), 2.5 = shared word and nothing
informative left over, 2 = lexicon only. Narration rules also take 1 off the ECHO evidence, so narration
wording wins the class when both fire.

| Rule | Fires when | Evidence |
|---|---|---|
| R1 | eyebrow, a word matches the heading at +1, at most one other informative word | 2.5 |
| R3/R5/R8 | subtitle, helper, caption, placeholder: a word matches the heading/label before and nothing informative is left over | 2.5 |
| R5 | tooltip: every word is in the control before/after | 3 |
| R3 | subtitle with at least two filler words and no informative word at all | 2 |
| R5 | small body (stat-card note), list line in a card: as R3 against heading/label | 2.5 |
| R4 | empty_body, dialog_body, error (not inside a control): as R3 against the heading (or error title) | 2.5 |
| R9 | heading (with at least one word that is not a number) with the same bag of concepts as an earlier heading (`dl1` = 0) | 3 |
| R5 | label (not in a control or table) equal to an earlier heading and matching a heading within 4 nodes | 3 |
| R14 | link (≥ 2 words) equal to an earlier button; helper/caption/subtitle with ≥ 2 non-unit informative words equal to an earlier text line; never conditional, never a repeated item | 3 |
| R17 | last body line of a section with a summary cue and at most one informative word (two cues: two words) | 2 |
| R2 | eyebrow (or tracked label) with no information outside genre/filler/function words; decorative coordinate | 2 |
| R11 | pill/dot/badge whose only information is a status word, not stateful/conditional/in a table/repeated item | 2 |
| R11 | a badge made of promo-pill words ("New", "Fresh", "Popular") equal to an earlier badge | 3 |
| R16 | narration roles: "below/above" + interface noun, count, or something named below; "this/these/here" + two interface words; + count + interface noun; with at most one other informative word. "below/above" alone when nothing new at all is said ("Below you'll find our plans.") | 2 |
| R6 | action verb + interface noun / "below" / something named below (scroll and swipe are both); helper "toggle/enable <label>"; subtitle starting with a control's name | 2 |

Also in the L3 MLP: a near-miss unit (+0.25 ECHO for "shared word, exactly one informative word left over"), which
only orders KEEP nodes for the precision–recall curve. **Layer 4 MLP**: three units cap each class's evidence at its
strongest rule level (ECHO 3, DECOR 3, NARRATE 2), so two rules on one node are not double evidence.

Vetoes on every rule: any `VAR`/`ALT` marker, the `dynamic` flag.

**Output head**: logits `(0, o_echo − 0.5, o_narr − 0.5, o_decor − 0.5)`. Margins are +2.5 / +2.0 / +1.5 by the
evidence level; the −0.5 means a rule must fire to at least a quarter before a node is flagged. All of these are
set from the mechanism, none from scores.

## 5. Checks used while revising

- `./score work --set dev` after each change (to see what the change did, never to pick a number).
- `work/scratch/flagged.py examples/reference`: the calls on the 22 unlabelled real screens, read by eye.
- `work/probes/*.probe.txt` (→ JSON by `work/scratch/mkprobe.py`): eight screens I wrote in the phrasing of real
  component-library and AI-built UIs, labelled by my reading of `LABELLING.md`: a settings form, a landing page,
  app screens (dashboard, empty states, dialogs, login card, tooltips), a page of hard negatives, a SwiftUI-style
  settings screen with a game HUD and menus, a personal site with a SaaS shell, chat and product page, a page of
  natural slop, and a page of "tricky keeps" (lines that share a word with their anchor but add something).
  Lines I would not want to be held to are marked `?` and not scored. Plus `work/probes/edge.json`
  (`work/scratch/mkedge.py`): three windows, empty texts, a 40-word node, number-only headings.
- `work/scratch/ptrdebug.py`, `tokdebug.py`: raw pointer scores and per-token features for one node.
- `./lint` on the pause-menu source and on a dev screen: the JavaScript runtime gives the same calls and margins.

## 6. Revisions after scoring

1. First build (token-level "repeated earlier" matching for R9/R14): dev F1 0.728, 74 false alarms. The false
   alarms were headings and links whose words were each found in *different* earlier nodes ("Add a box" from "Add
   your first order" and "Weekly veg box"). Replaced by the duplicate head, which compares whole bags of concepts.
2. Eyebrow hits restricted to the heading directly after (+1): "Last cask" was matching a heading two nodes on.
3. State pairing in the local heads: a table-header caption after an empty state was matching the empty title.
4. Tooltip must repeat its control word for word ("Edit album" is not "New album"); helper/caption no longer
   echo a *button* ("Saved" after "Save changes").
5. The duplicate head's bans were first −60 like the matching heads; a code-sum dot product is unbounded, so
   ineligible keys and the node itself won. Bans are now −1000s. After this: dev F1 0.769, 9 false alarms.
6. Lexicon: summary cues weigh 0; "features", "offers", "included" are filler; "plan" is half weight; ordinals
   share digit codes; "tell" joins notify; team/group merged; error lines are anchors (an error title followed by
   an error description); a subtitle of filler only is flagged; an eyebrow of function words only is decor.
   "may" removed from the function words (it made "Added in May" equal to "Added"); "appear" and "people" kept as
   content after "Albums other people share with you appear here" (K8) was flagged. Dev F1 0.823, 6 false alarms.
7. Pointer: three designs. (a) flags "this token is repeated by a neighbour" + distance: not specific to the
   querying node (69%). (b) the node's whole code sum as query: noisy for long nodes, and lost after the duplicate
   head overwrites it. (c) current: the sum of the codes of the tokens *that were matched* (`mc`), which is specific
   and almost noise-free: 90%.
8. Reference screens (unlabelled, real): two false alarms found by reading the calls. "Lancaster University" twice
   in an education list (both `repeated`): keys inside repeated siblings are now banned for headings. Field label
   "Required approver" nine nodes below card heading "Approval required" (same two stems): a card label must now
   also match a heading within four nodes.
9. Duplicate head selection used a length penalty proportional to the word count; with anti-correlated codes the
   true copy lost to an unrelated node ("Set up in under 7 minutes." → "9 October 2026"). Now the exact squared
   length of the key's code sum is computed in the L1 MLP, so the head picks the nearest code sum.
10. Probes (my own screens): 9 of 19 natural redundancies were missed at first. Fixes, all lexical: greetings are
    filler, "back" / "add" / "create" / "account" / "data" / "choose" are half weight, "receive" is filler,
    "enable" is an action verb, "systems" / "normal" are status words; a new narration rule for "enter your email
    below" (a layout word plus something named below). "activate" and "following" removed again: their stems
    ("activ", "follow") made "Active sessions" and "Columns follow the order on screen" narration.
11. Narration novelty allowance tightened from 1.5 to 1 ("…within 24 hours" is information); status rule not on
    repeated siblings (K2); graded evidence (3 / 2.5 / 2) and NARRATE preferred when narration wording is present.
12. About 60 more stems of general synonym groups (edit/change/update, view/see/show, log = sign in, avatar,
    role/position, amount/total, title/headline, guide/manual, level/stage …). Dev F1 0.834, 5 false alarms; probes
    16 of 19 with no false alarm on 170 KEEP nodes; reference screens: only the two pause-menu nodes flagged.
13. Kicker rules (R2): "today" / "now" count as class-external information (a sidebar group "TODAY" is a date
    category, K1) while staying filler inside sentences; the status rule now measures "information besides status
    words" separately (`b_xs`), is off on list lines (a dotted list is a legend, K2) and on repeated siblings.
    "about" and "why" moved from function words to coded filler, and the pure-filler eyebrow rule again needs one
    coded word, so "About", "About us", "Why us" are decor but "For you" is kept.
14. Near-miss ranking (+0.25, never a call), caps in the L4 MLP (two rules on one node are not double evidence).
15. A second batch of general synonym groups (about 75: shop/store, buy, stock/inventory, money/cash, bug/defect,
    repo/repository, spreadsheet/sheet, quiz/exam, employee/worker, reward/prize, plugin/extension, licence/license,
    centre/center, programme/program, survey/poll, demo, tip/hint, play/playback, …). `./tok --vocab` then showed
    Porter merges that put unrelated words in one row; fixed by leaving out the rarer reading: customize (=
    customer), application (= applicant), important (= import), manager (= manage), presentation (= present),
    standings (= stand), heading (= head), activity (= active), quite (= quit), authorize (= author).
    Dev unchanged (F1 0.834, 5 false alarms, pointer 91%); probes: 19 of 23, no false alarm on 217 KEEP nodes.
16. A sixth probe (personal site, SaaS shell, chat, product page) gave one false alarm: tag badge "TypeScript" on
    two project cards was called a repeated decorative pill. The repeated-badge rule now needs the badge to consist
    of promo-pill words (`promo.txt`); the class dim came from folding "scroll / swipe" into action + interface
    words. Probes now: 20 of 24 redundant nodes, no false alarm on 306 KEEP nodes.
17. A seventh probe of natural slop (eyebrows over section headings, restating subtitles, placeholders, empty
    states, dialogs) found lexical gaps: "address", "name", "number", "date" are now half weight ("Enter your email
    address" under "Email"), "read" and "occur" are filler, and a layout word with nothing new at all is narration.
    Two fillers I added were wrong and are reverted: "meet" (dev: "Meet at Swansea." is an instruction) and
    "product" (probe: "Receive emails about new products, features, and more." says what the emails are about).
    An audit of the filler list removed "already" and "taken" (they state a condition: "Username already taken").
18. An edge-case screen (three windows, empty texts, a 40-word node, number-only headings) through `./lint`: no
    crash, same calls as the reference; two findings. Headings "12" and "12" were called an echo: the repeated
    heading rule now needs a word that is not a number. Repeated headings pointed at the nearest heading sharing
    one word rather than at the full copy: the pointer's distance term is now only a tie-break (0.05 per node,
    capped at 0.6). Dev pointer accuracy 92%; dev F1 0.836 with 5 false alarms; probes 47 of 52 with no false
    alarm on 335 KEEP nodes.
19. With distance reduced to a tie-break the pointer's code noise showed (probes 78%): a node that matched three
    concepts has noise of sd 11 against 22 per true token, and some token on a 300-token screen reaches it. Fixes,
    one at a time: (a) keys that no candidate restater repeats lose 24 (`rf` / `rb` flags from two more matching
    heads); a first version *rewarded* flagged keys and tokens with both flags then beat the true target; (b)
    the "action verb → button" bonus cut from 32 to 3 and the "later node" penalty from 4 to 1 (a helper with
    "turn" was sent to a far button); (c) helper / caption / subtitle words no longer join the query through a
    match in a far heading ("account" in "Voice commands for your account." pulled the pointer to the "Account"
    heading), at the price that a repeated fact (R14 text) gets no target. Probes 37 of 37, dev 93%.
20. Eighth probe ("tricky keeps", 65 firm KEEP lines): one false alarm, eyebrow "For teams" called decor because
    "team" was a genre word (for "Our team"). "team" is far more often a plan name or an audience than a section
    genre, so it is no longer one; an "Our team" eyebrow is now kept.
21. The kicker-styled label rule is off inside navigation (a sidebar group label such as "GENERAL" structures the
    nav; the examples' decorative labels are all outside nav). No example changed.

Final state (artefact `e3e0b1d08f24…`): dev F1 0.836 (precision 0.984, recall 0.726, 5 false alarms in 6,785 KEEP
nodes; on `clear` labels recall 0.87, on `judgement` labels 0.46), four-class macro-F1 0.897, pointer 94%; the
`ct` column 32 of 38 with no false alarm; my eight probes 47 of 52 redundant nodes with no false alarm on 405 KEEP
nodes, pointer 37 of 37; the 22 real reference screens: only the two pause-menu nodes are flagged (eyebrow
"OPERATION ON HOLD" → ECHO → "PAUSED", subtitle "Resume recaptures the mouse." → NARRATE → "RESUME").

## 7. Known failures

On the dev examples (116 misses, 5 false alarms at the time of writing):

- **Paraphrase with no shared concept** (most misses; labelled `judgement` in the examples): "Tweak how it plays."
  for "Options", "Run ended" for "Game over", "Something only you know" for "Password", "You have been defeated",
  domain synonyms (fares/tickets, canoes/boats, brews/batches, plantings/sowings). The lexicon holds general UI
  synonym groups only; anything else is an unknown content word and therefore information.
- **Restatement plus one unknown word**: "Everything about your batches in one place." under "At a glance",
  "Hive settings for this account, listed.", "Extra login protection." One unmatched content word is treated as
  new information (that is what protects "Reading history for every branch" and most K4 helpers), so these stay.
- **Negative emphasis tails**: "…; nothing else.", "…, no older.", "not a penny": an unmatched negation counts as
  information.
- **Subset duplicates**: h3 "Board booking" under h2 "Board booking, on every plan"; h2 "Connected apps" under
  h1 "Integrations". The duplicate head needs the same bag of concepts, filler words included; this is what keeps
  "Album details" under "Album", and step headings that reuse words, from being flagged.
- **Repeated button** (R14 button → button) is never flagged: in real files the same button text in two places is
  usually two separately mounted views. A repeated *link* is flagged only against an earlier button.
- **Label / stat repeated** (R14 "Foot passengers", "4,817"), the first copy of a repeated badge, "CAM 11 · REC".
- False alarms: an eyebrow "Settings" over "Preferences" (same concept, labelled a parent section, K1); a helper
  with action words that describes an effect ("Turns rapid button presses into a single hold."); a card heading
  "Sessions" equal to an earlier heading; caption "In Aberystwyth".

By design, to be checked on real screens:

- The rule for a subtitle that begins with a control's name ("Resume recaptures the mouse.") fires on any subtitle
  whose first word is the label of a control within three nodes and that has at most two other content words. It
  exists for the motivating example (a `ct` case) and can flag an instruction such as "Save your changes before
  leaving." above a "Save" button.
- A subtitle made only of filler ("Everything you need, all in one place.") is flagged even with no heading match.
- Word order and negation scope are invisible: the model compares bags of concepts.
- Porter merges remain (active/activity, book/booking, objective/object, organ/organization).
- Pointer: about 93% on dev. It cannot require that *all* of a node's concepts sit in one target (attention sums
  per token), so with three or more matched concepts an unrelated flagged token can win on code noise. A repeated
  fact (R14 text) and narration that names no control get no target. For lexicon-only calls (R16, R17, R2) the
  guide wants no target, but the head still points at a node when the line shares a word with a neighbour.
- Stretch (`ct`) rules are deliberately narrow; "Press Confirm once you are happy." (two unknown words) is missed.
