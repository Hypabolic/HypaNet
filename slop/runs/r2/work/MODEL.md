# MODEL.md — the hand-written slop network (round 2)

Rebuild:

```
./python3 work/select_thesaurus.py    # resources/thesaurus + hand lexicon -> work/thesaurus_groups.tsv
./python3 work/build.py               # lexicon + thesaurus_groups.tsv -> vocab.txt, artifact.npz, meta.json ("ids": 2)
```

`build.py` also writes `directions.json` and `dims.json` for `./inspect`. It prints the artefact hash; running it
twice gives the same hash (`9fa3f37d61f7…`). `select_thesaurus.py` only needs re-running when a lexicon file
changes (its output is kept in `work/`). **Pointer head: layer 4, head 0** (`meta.json`: `[4, 0]`).

Size: 5,336 vocabulary stems in id layout 2 (2,134 from the hand lexicon, 3,202 from the thesaurus), about 1,820
synonym groups, 10,451 concept codes, 128 residual dimensions, 8 + 5 + 2 + 1 attention heads, 354 + 100 + 43 + 7
MLP units, about 262,000 non-zero parameters. The cluster embedding is all zero.

No weight, threshold or vocabulary entry was fitted or searched: each number is derived in a comment in `build.py`
from what the mechanism computes (code-match scores, counts that are integers, halves or quarters), and each
revision below was made for a stated reason and then looked at. Where a change was tried and taken back, the
reason the reasoning was wrong is given.

## 1. The idea in one paragraph

A node is redundant when everything informative in it is already given by a higher-ranked node that is visible
with it. The network measures exactly that. Every word token carries a **concept code** (synonyms share one code)
and an **information weight** (0 for function words and restating frames, 0.5 for weak words, 1 for content words,
for markers of a consequence / condition / time / audience, and for unknown words). Layer 1 asks, per token, "is
my concept in a rank 1-3 node just before / just after me?". The layer-1 MLP turns that into per-token *novelty*
(informative and not covered) and *anchor hits*. Layer 2 sums these per node at the node's `EON`, checks whether an
earlier node has exactly the same bag of stems (a repeated title, call to action or fact), and asks per token
whether its concept occurs anywhere else in the window. Layer 3 is a set of rule units, one per pattern of
`LABELLING.md`, each an AND of a few booleans with vetoes; layer 4 adds the two rules that need the window-wide
coverage, caps, and holds the pointer head. Unknown words are informative by default, runtime text (`VAR`, `ALT`,
`dynamic`) vetoes every rule, and roles the guide never flags (button, nav, tab, menu item, cell, stat, …) have no
rule at all: the default is `KEEP`.

**What round 2 changed in the idea.** RESULTS.md: every round-1 false alarm on real copy was a full sentence that
shares a concept with its anchor and then adds something. A shared concept is still required, but what the
sentence says *besides* is now weighed as the codex's residue test (its §7) describes:

- consequence, condition, time and audience markers (`also only until unless because if when after before again
  still never must each everyone off plus …`) are never function words: unmatched, each is worth a full content
  word;
- half-weight words (`will so too then always other someone people new save access data plan start …`) need a
  second one;
- the scope pronouns `everything everywhere` are news inside a confirmation body (K7: the action reaches further
  than the title's object); `all every both whole` are frames;
- an **imperative lead** (a node that opens with an instruction verb) is news against a heading or title (K8, U14:
  "Pick a playlist." beside the list "Playlists") and only an input lead against a field label (R5/R8: "Choose a
  size." under "Size"); the network tells the two anchors apart (`hit_p` / `hit_pl` and the value 64, section 4);
- the restating frames of the codex (`manage view configure browse track everything about, in one place …`)
  weigh nothing in either direction;
- a **subtitle** is weighed against its own anchor only (the two nodes before it): a heading three or four nodes
  back belongs to the section or card before ("Everything in Free, plus:" under the plan "Pro");
- duplicates (R9, R14) need the same *stems*, not only the same concepts, so that broad synonym groups do not turn
  a section "Members" on a page "Team" into a repeated title;
- narration needs interface wording: a gesture or control-mechanics verb, or an ordinary verb (`choose`, `select`,
  `hit`) whose line names a control below in full and says nothing else.

## 2. Lexicon

### 2a. Hand lexicon (`work/lexicon/`; general UI vocabulary from my own knowledge, written with the codex lists and the thesaurus open; no file has a comment syntax)

| File | Meaning | Effect |
|---|---|---|
| `function.txt` | articles, pronouns, auxiliaries, prepositions | no code, weight 0: invisible to matching and novelty |
| `particle.txt` | in out up down off on over under | a code (so "Sign in" and "Sign up" are different bags), weight 0, never a hit ("in one place" does not restate "Sign in"), never narration novelty |
| `marker.txt` | consequence / condition / time / audience markers: the closed-class words of the codex's `ui_consequence_markers.txt` (future, scope, time, condition and reason blocks) and of the obligation frames in `ui_instruction_markers.txt`; plus `each`, `off`, `down`, `plus` | weight 1 always (takes precedence over every other weight class); never grouped by the thesaurus rule |
| `scope.txt` | everything everywhere | weight 0, but 1 in a `dialog_body` |
| `frame.txt` | restating frames and filler: generic verbs (manage, view, see, find, browse, track, configure, confirm, …), generic nouns (thing, item, place, app, page, …), quantifiers (all, every, any, more, …), intensifiers (`ui_filler_words.txt`, `ui_padding_phrases.txt`), the words of `ui_restating_frames.txt`, praise / hype / honesty adjectives, interjections | code, weight 0 |
| `semi.txt` | half-weight words | weight 0.5 |
| `ilead.txt` | instruction / input lead verbs (`ui_instruction_markers.txt` imperative leads, `ui_input_lead_verbs.txt`): add create make choose select pick enter type fill provide give go try turn switch tell let save tap click press visit upload search write insert paste drop specify start describe. Left out: the frame verbs (manage, view, check, update, find, …), "confirm" (a restating frame), "set" ("Settings …" opens many subtitles) | imperative lead (section 4) |
| `groups_actions.txt`, `groups_objects.txt`, `groups_states.txt`, `groups_more.txt`, `groups_general.txt` | one synonym group per line, about 650 groups (delete/remove/erase/discard…; settings/preferences/options/configure…; user/member/people/team/workspace…; message/email/inbox/chat…; order/purchase/buy; search/find/results/match; pay/payment/billing/invoice; library/collection/saved; end/finish/stop/over/defeated; no/none/nothing/empty/clear/without…; …). Earlier files win a conflict; `build.py` prints any conflict | members share one concept code |
| `action.txt` / `actionweak.txt` / `actionpart.txt` | narration cue weight 1 (click, tap, press, scroll, swipe, hover, toggle, pinch) / 0.5 (choose, select, pick, switch, turn, hit, enable, disable, expand, collapse) / 0.25 (on, off): `ui_narration_verbs.txt`, gesture and control-mechanics blocks | R6: a whole cue (0.75), or half a cue with a control named in full |
| `gesture.txt`, `deictic.txt`, `uinoun.txt`, `narrverb.txt`, `dem.txt` | scroll / swipe; below, underneath, beneath, bottom (1) and above (0.5); interface nouns; "this page lists / shows / lets …"; here, this, these | R16 / R6 |
| `narrpad.txt` | padding of a narrating line: ready, done, happy, satisfied ("when you're ready", "once you are happy": `ui_padding_phrases.txt`); continue, proceed, start(ed) ("to continue", "to proceed", "to get started": `ui_narration_verbs.txt`, outcome narration) and the same purpose tails finish, begin, ahead | no weight in the narration novelty only; everywhere else these words keep their weight and group |
| `hid.txt` | hidden-input cues: hold, long, shift, ctrl, cmd, alt, esc, keyboard, shortcut, double, drag, left, right (`llm_narration.txt`, hidden-input block) | veto every narration rule (K4) |
| `aux.txt` | clause markers: auxiliaries, modals, relative pronouns and their contractions | −100 each in the list-shape sum (layer 4) |
| `kick.txt` / `kickhalf.txt` | section-genre kicker weight 1 (features, pricing, faq, overview, about, why, details, highlights, introduction, …) / 0.5 (how + works, our + team, key + facts, the + basics, get + in + touch, selected + work, …): the K1 list of `LABELLING.md`, `ui_genre_kickers.txt` blocks A and B and `llm_kickers.txt` L38 / L39, as far as they are section genres; content categories of those lists (blog, news, guide, results, projects, security, …) stay content | R2: an eyebrow whose kicker weights reach 1 and that has no other information |
| `status.txt` | static status words: online live ready running connected open synced available active operational system(s) normal healthy stable (`ui_static_status_claims.txt`, availability and system-health blocks; release-stage words such as beta / alpha / early access are left out: the guide keeps a sole "Early access" pill) | R11 |
| `promo.txt` | new fresh hot popular featured trending top best premium exclusive limited updated improved recommended | R11 repeated promo pill |
| `summary.txt`, `summarystrict.txt`, `summaryhalf.txt`, `summarymost.txt` | summary cues, weighted: loose closing cues 1 (that, those, short, covers, sum, wrap, taken, upshot, plus, besides, nutshell, all); a whole summary frame 4 (summary, overview, glance, snapshot, recap, rundown, lowdown, gist, place); parts: everything 2 + about 2, together 3 (+ all 1) (`llm_filler.txt` L34, `ui_narration_verbs.txt` self-reference block) | R17 (any cue), layer-4 summary-line rule (≥ 4) |
| `meta.txt`, `metahalf.txt` | meta-strip words: sector sys cam rec lat lon latitude longitude coordinates (1); local 0.5, time 0.25 | R2 meta strip |
| `navgenre.txt` | contents navigation menu sitemap page | R2 heading inside navigation |
| `unit.txt`, `timeword.txt` | price qualifiers (K13); today / now (filler in a sentence, a date category as a kicker) | as in round 1 |

In `build.py`: number words two…twelve and the ordinals second…fifth share the code of the digit token; compass
letters (n, w 0.5; s, e 0.25) and the degree sign (0.25) carry meta-strip weight; every single letter is marked for
the monogram rule; the comma, "&" and "and" are list cues. `NUM` and `DEC` tokens each have one code (a figure
repeated beside its stat is covered; a number is never a hit). Negations (no, not, none, nothing, zero, never,
empty, clear, without, n't forms) are one synonym group with weight 1 ("clear" 0.5), so "Your inbox is clear." is
covered by "No messages" but an unmatched negation is news.

Weight classes are applied in this order: marker (1), frame / deictic / scope / particle (0), half weight (0.5),
summary cue (0), everything else and every hash bucket (1). The *narration* weight of a word is its weight, except
0 for interface nouns, action cues, narration verbs, number words, narration padding and particles.

What must stay apart was checked three ways: (1) by hand while writing the groups (enable / disable, show / hide,
public / private, lock / unlock, connect / disconnect, increase / decrease, small / big, win / lose, start / stop,
first / last, valid / invalid, required / optional, upload / download and the like are separate groups or
ungrouped); (2) `work/scratch/antcheck.py` lists every pair of stems that shares a concept although WordNet calls
them antonyms: after separating borrow / lend and question / answer it reports none in 1,820 groups; (3) Porter
merges: `work/scratch/lexcheck.py` and a listing of the other WordNet words of every weight-class stem. Left out
of the frame list because their stem is another word's: excellent (Excel), powerful (power), outstanding
(outstanding invoices), exceptional (except), remarkable (remark), lovely (love), literally (litre), practically
(practice), effectively (effect), largely (large), polished (Polish), basically (Basic). "organise" stays a
content word (organisation); "activate", "manager", "application / apply" as in round 1.

### 2b. Thesaurus (`work/select_thesaurus.py` → `work/thesaurus_groups.tsv`)

Read: `resources/thesaurus/out/stems.tsv` and `stem_links.tsv`. The rule was fixed by reasoning about what the
thesaurus is good for, never by a score; I read its output and struck out 47 stems by hand
(`work/thesaurus_exclude.txt`).

- A *strict synonym* pair shares a WordNet synset **and** is a mutual Moby link (both thesauri agree, as the
  README and the codex §9.2 advise), and is not a WordNet antonym pair.
- Only *specific* stems take part: at most 3 WordNet synsets, alphabetic, at least 3 letters. Words with many
  senses are exactly where a thesaurus wanders off ("clear", "set", "order", "save"); those are grouped by hand.
- **Rule A**: a specific stem outside the hand lexicon joins the group of a specific hand-lexicon stem it is a
  strict synonym of (the group with most qualifying members wins, a tie drops it, never an antonym of a member,
  never into the closed classes `function / particle / marker / scope`, number words or single letters); it
  inherits that member's weight class (frame or half weight) and nothing else. 64 stems into 38 groups.
- **Rule B**: the remaining specific stems form star clusters of strict synonyms (visited by number of strict
  partners, then alphabetically; a head takes its unassigned partners; nothing is chained through a member).
  1,904 stems under 1,234 heads, weight 1.

What I did not take, and why: **hypernym links** (canoe → boat, van → vehicle): a hyponym is more specific than
its hypernym, and a K1 category eyebrow is usually exactly such a hyponym ("Vans" over "Every vehicle and every
driver, in one app"); **Moby-only links** (loose associates); **anything through a polysemous word**: a first
version of rule A let any hand word attract members through its three commonest senses, and "kid", "fair", "firm",
"void" and "deck" pulled in "banter", "carnival", "unwavering", "excrete" and "bedeck". The thesaurus therefore
widens the long tail (rare, specific words). The common UI vocabulary, where the recall is, is the hand lexicon;
for it the thesaurus was a reference: `work/scratch/suggest.py` lists, per hand group, the strict synonyms I had
not written, and I took about 45 of them by hand into `groups_general.txt` (rewrite, transmit, interrupt,
substitute, assess / evaluate, transform, rebate, critique, altitude, intent, …).

Codex lists read while writing the hand lexicon (none is read by `build.py`): `ui_consequence_markers.txt`,
`ui_instruction_markers.txt`, `ui_restating_frames.txt`, `ui_input_lead_verbs.txt`, `ui_narration_verbs.txt`,
`ui_genre_kickers.txt`, `ui_static_status_claims.txt`, `ui_status_genre_titles.txt`, `ui_filler_words.txt`,
`ui_padding_phrases.txt`, `ui_empty_state_templates.txt`, `llm_kickers.txt`, `llm_narration.txt`, `llm_filler.txt`.
Selection rule: single words (or the words of a phrase, with fractional weights when only the whole phrase
counts) whose class holds in UI copy in general. Left out: entries the codex marks as beyond the guide
(status-genre titles as DECOR: the reference key keeps "Something Went Wrong"; vague warnings), open-class
consequence words (they are content words anyway), and promotional vocabulary (rule-layer territory) except where
a word is plain filler inside a restatement.

## 3. Residual allocation (128 dims; names in `work/dims.json`)

Three kinds of dimension: **both** (a per-token value at tokens, the per-node sum at `EON`), **tok** (tokens only)
and **eon** (`EON` only). Tok and eon dimensions share physical dimensions (`tokname|eonname` in `dims.json`);
every MLP unit is gated to the kind of position it is meant for, and attention readers of a tok / eon dimension
only see keys or queries of that kind.

| Dims | Kind | Content |
|---|---|---|
| `code0..13` | both | ±1 concept code (0 for function words); at `EON` after L1 the node's code sum, after L2 the difference to the nearest earlier candidate copy |
| `code14` | both | 1 + 4·j for the j-th stem of a synonym group: synonyms differ here, so only the same *stems* make the same bag (read by `dl1` only) |
| `one`, `is_eon`, `first` | both | constants / position marks |
| `wi mt wx num deic uin act narr status summ dem promo unit kick meta tiny` | both | lexical weights and class sums (L2 count head 4) |
| `var hid aux` | both | marker count; hidden-input cue count; the signed list-shape sum (L2 count head 0) |
| `mc0..10` | both | matched code (the token's code if it was matched through a gate that fits its role); the sum is the pointer's query |
| `nov_p nov_n nov_x hit_p hit_pl hit_n hit_nc hit_nc0` | both | novelty against the nodes before / after (for subtitles and body lines: against the whole window) / narration novelty; hits on a heading / a label before, the heading after (for a control: its words that no narrating line before it repeats), something named below, a control named first |
| `m_any` | both | the concept occurs in another node of the window (layer 2) |
| `g_* r_*` (17), `f_*` (14) | both | role groups and flags |
| `wn ws il nohit m_p m_pn m_ph m_pl m_pc m_n m_nc m_n1h m_g rf rb f_any nov_ct` | tok | narration weight, scope weight, instruction lead, particle; match outputs (`m_pn`: in the two nodes before); pointer key flags; narration novelty against the controls below |
| `b_*` (19 static, 12 dynamic), `novg`, `b_list`, `novc`, `b_ctl`, `dl1`, `norm2`, `o_echo o_narr o_decor` | eon | booleans, window novelty, list shape, narration novelty against the controls below, "a control below is named in full", duplicate distance, squared code-sum length, class evidence |

The cluster embedding is unused: locality comes from the signed node distance in the attention bias.

## 4. Layers

**Layer 1 attention** (matching heads: score = 6 per agreeing minus 6 per disagreeing code bit over 14 bits, so an
identical code scores 84 and any other at most 72; `BOS` = 79 is the threshold; eligible keys only)
- H0 *previous*: keys in rank 1-3 nodes (headings, labels, buttons, stats, links, body, error) at −4..−1; +3 for
  anchors. Never keys: dynamic, sr-only. A key inside a state cluster is invisible to a query outside one (§2a).
- H5 *own anchor*: the same keys at −2..−1 only (`m_pn`), used for subtitles.
- H1 *next* (+1..+3, same keys), H2 *next heading* (+1 only, headings), H3 *any earlier* heading / button / text
  line that is not conditional, dynamic, sr-only, in nav or in a table (feeds the pointer query).
- H4 / H7 reverse matches (`rf`, `rb`): the token is repeated by a later lower-rank node / by an eyebrow, tooltip
  or narrating line in the three nodes before. Used as pointer keys; `rb` also tells a control which of its words
  the line above it names.
- H6 per-node code sum → `EON`.

**Layer 1 MLP** (token units are switched off at `EON`, `EON` units at tokens)
- `nov_p = relu(wi − 2 m_p)`; for a subtitle `relu(wi − 2 m_pn)` (its heading is the node directly before it;
  the other roles sit inside a form, card, list or state whose heading can be three or four back).
  `nov_n` likewise against the next nodes, for eyebrows and tooltips only;
  `nov_x = relu(wn − 2 m_p − 2 m_n)`; `nov_ct = relu(wn − 2 m_nc)` (against controls below only); for tooltips
  filler words count too.
- scope pronoun in a `dialog_body`: `relu(ws + r_dlg − 1 − 2 m_p)` more novelty.
- **imperative lead**: the first token, if it is an instruction verb, adds **64** to `nov_p` instead of its own
  weight: in an empty / error / dialog body unless the title has the verb; in a subtitle, body line, list line,
  caption, helper or placeholder unless a node just before or a control just after has it. 64 is out of reach of
  ordinary novelty (≤ 31), so two booleans read the same sum: `b_novp` (≥ 0.75: the imperative counts) for
  heading-anchored rules and `b_novpl` (the 64 taken out) for label-anchored ones.
- hits: `relu(m_class + role − 1 − num − nohit)` per role group; `hit_p` on a heading (or error title; for a
  tooltip the control), `hit_pl` on a label. For a button or link: `relu(mt − 2 rb)` per token, so the sum at its
  `EON` is 0 exactly when every coded word of its label is repeated by a narrating line just before it.
- matched code through five gates (round 1's four, and for a repeated fact: a full-weight helper / caption /
  subtitle word found in an earlier text line and not just before), pointer gate `f_any`.
- `norm2` (squared length of the code sum, for the duplicate head).

**Layer 2 attention**
- H0 per-node sums of the token features (and of `var`, `hid`, `aux`, `nov_ct` → `novc`); H4 per-node sums of the
  16 lexical class dims; H2 per-node sum of the matched code.
- H1 *duplicate head*: `EON` of A → `EON` of the earlier eligible node B nearest to A in code sum; subtracts cs_B
  from A's code dims (classes as round 1: heading → earlier heading, link → earlier button, text line → earlier
  text line, badge → earlier badge, card label → earlier heading; never conditional, dynamic, sr-only, nav or
  table keys, nor repeated sibling items except for badges).
- H3 *anywhere*: token → a token with the same concept in any other node (static text, not sr-only) → `m_any`.
  `EON` keys are banned with −4000: in this layer an `EON` holds a code sum, whose product with a code is
  unbounded (revision 59).

**Layer 2 MLP**: static booleans from the class sums (`b_act` ≥ 0.75 cue weight, `b_deic`, `b_uin`, `b_kick`,
`b_mt2`, `b_mt3`, `b_word`, …); dynamic booleans (the two hit thresholds sit below a third because a word found in
both the heading and the label above splits its attention); `b_novpl`; `b_list` (list-shape sum ≥ 2: two list
cues and no clause marker); `dl1 = Σ|code_i|` over 15 dims; and the token unit for the window novelty of
subtitles, body and list lines: `relu(wi − 2 m_p − 2 m_any)` into the `nov_n` dimension, which layer 1 leaves
empty for those roles.

**Layer 3 attention**
- H1 a count head that sums the window novelty into `novg`.
- H2 the *named control* head: `EON` of a line → the `EON` of a button or link in the next three nodes that has a
  word (`b_word`) and no uncovered word (`b_hitn` = 0); never dynamic or sr-only, never from outside a state into
  one. Scores are exact (eligible 0, anything else ≤ −60, `BOS` −30); it writes the key's constant 1 into `b_ctl`.

**Layer 3 MLP**: rules at `EON`, each `relu(Σ positives − (need−1) − 100·Σ vetoes − soft)`. Evidence per fired
rule: 3 = word for word, 2.5 = shared word and nothing informative left over, 2 = lexicon only.

| Rule | Fires when | Evidence |
|---|---|---|
| R1 | eyebrow, a word matches the heading at +1, at most one other informative word | 2.5 |
| R3/R5 | subtitle, helper: a word matches the **heading** before, nothing informative left over, no imperative lead | 2.5 |
| R5/R8 | subtitle, helper, placeholder: the same against the **label** before; an imperative lead is ignored. A placeholder echoes labels only | 2.5 |
| R5 | caption: as both; under a heading it also needs two coded words (a lone noun under a heading that contains it is a column header) | 2.5 |
| R5 | tooltip: every word is in the control before/after | 3 |
| R5 | small body (stat-card note) against heading or label; list line in a card against the heading | 2.5 |
| R4 | first body line of a section against its heading, with nothing at all left over (not even a half-weight word) | 2.5 |
| R4 | empty_body, dialog_body, error (not inside a control): as R3 against the title | 2.5 |
| R3 / R17 | filler lines: a subtitle, or the last body / list line of a section (not conditional, not in a state cluster), with ≥ 2 coded words and weight 0; or ≥ 3 coded words and novelty < 0.75 against the nodes before | 2 |
| R9, R5 label, R14 link / fact | same bag of **stems** as an earlier eligible node (`dl1` = 0) | 3 |
| R17 | last body / list line with a summary cue and at most one informative word | 2 |
| R2 | eyebrow (or tracked label outside nav and controls) whose kicker weights reach 1 and that has no other information | 2 |
| R2 | caption / helper / eyebrow that is a meta strip: meta weight ≥ 0.5 beside a number, and nothing but numbers | 2 |
| R2 | heading inside navigation made of navigation-genre words ("Contents", "On this page") | 2 |
| R11 | static pill holding a single letter | 2 |
| R11 | badge, or pill / dot on a text role, whose only information is a status word; not stateful / conditional / in nav or table / repeated | 2 |
| R11 | a badge with a promo word and nothing else, equal to an earlier badge | 3 |
| R16 | layout word + interface noun or count (≤ 1 new word); layout word + something named below; a layout word and nothing new at all; "this / here" + two interface words, or + one count + interface noun (both switched off by a second number) | 2 |
| R6 | a whole action cue (≥ 0.75) + an interface noun, a layout word or a word of a node below (≤ 1 new word); a whole cue + label or heading hit (helper, caption) | 2 |
| R6 | half a cue (choose, select, pick, hit, switch, turn …) + a control below named in full (`b_ctl`) + no other word of narration weight (`novc` = 0) + no number ("Choose Save changes to continue.") | 2 |
| R6 | subtitle starting with the name of a control below, at most 2.5 new words, no number ("Resume recaptures the mouse.") | 2 |

All narration rules: vetoed by a hidden-input cue; −10 on ECHO so narration wording wins the class.

**Layer 4 attention**: H0 the pointer head (below). It writes nothing.

**Layer 4 MLP**
- caps on the layer-3 evidence (ECHO 3, DECOR 3, NARRATE 2: two rules on one node are not double evidence);
- the **summary line**: a subtitle that carries a whole summary frame (cue weight ≥ 4: "… at a glance", "… in one
  place", "an overview / a summary …", "everything about …", "… all together") and whose informative words all
  occur elsewhere in the window (`novg` = 0) → ECHO, evidence 2 ("A summary of your orders." over cards about
  orders; "All your saved recipes in one place." under "Library");
- the **enumerating intro**: a subtitle shaped like a list (`b_list`), with no number, whose listed words are on
  the screen: covered − 2·uncovered ≥ 4 informative words (covered = `wi − novg`, uncovered = `novg`) → ECHO,
  evidence 2 ("Strong grounding across design, research, testing and delivery." over
  a list whose entries name those subjects [real example replaced before publishing]). An imperative lead costs 2.
  Both are graded units clamped to one firing; neither has a single target. They sit beside the caps, so their 2
  comes on top of capped layer-3 evidence (a subtitle that restates its heading *and* is a summary line of the
  screen ends at 4.5); that only orders nodes that are flagged anyway.

Vetoes on every rule: any `VAR`/`ALT` marker, the `dynamic` flag.

**Output head**: logits `(0, o_echo − 0.5, o_narr − 0.5, o_decor − 0.5)`: a rule must fire to at least a quarter
before a node is flagged; margins are evidence − 0.5.

**Pointer** (`meta.json`: `[4, 0]`; the round-1 design, moved from layer 3 to layer 4 with the same weights):
`EON` of A → token u with score 2·(matched-code sum of A · code of u, 11 bits: 22 per shared concept) − 24 unless
u is flagged as repeated by a node that could be restating it + 3 (anchor node) + 6 (u has the role an earlier copy
of A would have) + 3 (A has an action cue or a layout word and u is a button) − 0.05·|distance| (cap −0.6) − 1 (u
later than A); `BOS` 7. It needs the per-node matched-code sum, which exists after the layer-2 count head, and in
layer 4 it also sees the `b_act` that the layer-3 rule "half a cue + a control named in full" sets, so that such
a line points at the button it names and not at a title that shares the word. Everything else it reads is the
same in layers 3 and 4. Calls made without a matched word (filler lines, the layer-4 rules, lexicon-only DECOR /
NARRATE) have no target.

## 5. The fourth layer

Round 1 left it empty. Considered for it:

1. a second matching pass on secondary senses of a stem: rejected. One code per stem with the UI sense chosen by
   hand is simpler, and a wrong second sense is a false-alarm source;
2. a "related topic" match (login ~ account, tools ~ integrations): rejected. A subtitle naming specifics of its
   heading's topic is information;
3. **window-wide coverage** for lines that restate what several other nodes show (LABELLING §3: an echo "spread
   over several" nodes, target null). This cannot be done with the local matching of layer 1 and needs four steps
   (anywhere match → token novelty → node sum → rule), so it runs through layers 2, 3 and 4: the anywhere head in
   L2 attention, the token novelty in the L2 MLP, the node sum in L3 attention, the two rules in the L4 MLP.

(3) is what is built. It is deliberately narrow, because on pages that repeat their nouns everywhere coverage
alone says nothing: the summary-line rule demands a whole summary frame, the enumerating-intro rule the shape of a
list with no clause marker and no number. Both first versions were wider and were cut back after reading what they
flagged (revisions 37 and 41).

The layer also made a second three-step chain possible: control token → "is my whole label named above?" (L1,
L2) → the line above learns it (L3 attention) → rule (L3 MLP) → the pointer reads the result (L4 attention).
That is why the pointer head now sits in layer 4.

## 6. Checks used while revising

- `./score work --set dev` and `--set reference` after each change (to see what the change did).
- `work/probes/*.probe.txt` → JSON by `work/scratch/mkprobe.py`: round 1's eight probes plus seven written for
  round 2: `r2_keep` (about 135 KEEP lines: sentences that share a concept with their title / heading / control
  and add a consequence, an instruction or a reason: the three RESULTS.md categories and codex §7 pairs),
  `r2_para`, `r2_para2`, `r2_para3` (restatements in other words: the RESULTS.md miss examples, codex L35 / L41 /
  U04-U34 / U47 examples and my own, by slot and kind of product), `r2_real` and `r2_real2` (about 220 lines of
  ordinary product copy: stat cards, forms, dialogs, settings rows, footers, a paywall, a permission prompt, pricing cards, error
  and empty pages; `r2_real2` was written before it was run and gave revision 61), `r2_narr` (instructions that
  name a button in a purpose clause, against narration of a button). Lines I would not want to be held to are
  marked `?`.
- Five source files through the real extractor and runtime (`./lint work/probes/src_*.tsx|swift`): a settings
  page, a SwiftUI cooking screen, a landing page, a research panel, a personal site. They confirm the roles the
  rules expect and gave revision 45 and the "Selected work" kicker; the JavaScript runtime gives the same calls
  and margins as the reference model.
- `work/scratch/dbg.py FILE NODE` (per-token features, EON sums, booleans, pointer), `blockers.py` (which tokens
  carry the leftover novelty of missed nodes), `farcov.py` (nodes whose only cover is three or four nodes back),
  `bykind.py`, `wrong.py`, `listrole.py`, `calls.py`, `lexcheck.py`, `antcheck.py`, `thes.py`, `suggest.py`. All
  of these only print; none writes anything `build.py` reads.

## 7. Revisions

### Round 1 (kept as written then)

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

Round-1 final state (artefact `e3e0b1d08f24…`): on the 400 dev screens of round 1 F1 0.836 with 5 false alarms; on
this kit's 1,200 dev screens F1 0.771 (precision 0.938, recall 0.655, 56 false alarms in 20,288 KEEP nodes).

### Round 2

22. **Id layout 2 and the dimension kinds.** `build.py` moved to layout 2; token-only and EON-only dimensions
    share physical dimensions, which made room for the new quantities. Rule units are gated to `EON`. Concept
    codes are handed out by enumeration (numbers, vocabulary concepts, hash buckets: about 10,450 of the 16,384
    values of the 14-bit code), so no two concepts share a code.
23. **Residue instead of overlap** (RESULTS.md false alarms 1-3, codex §7 and §9.1): marker words out of the
    function list with weight 1; half-weight "will / so / too / then / always / other(s) / someone"; the filler
    list cut down to the codex's restating frames and padding. Reason: "All items in this list will also be
    removed." under "Remove this list?" had novelty 0 in round 1 because "all", "also", "will" and "items" all
    weighed nothing.
24. **Scope words.** First "all / every / each / everything …" weighed 1 inside state bodies. Wrong twice: in
    empty bodies they are the frame ("You're all caught up", "You don't have any …"), so the weight was limited to
    `dialog_body`; and dev's "All 8 boats will be deleted." under "Delete 8 boats?" showed that a determiner on the
    title's own noun adds nothing, so only the pronouns `everything / everywhere` keep it ("Everything in it will
    be deleted."). "each" became a marker instead ("Each sighting can have this on or off." is K4).
25. **Imperative lead.** First as +1 novelty in subtitle / body / caption roles and never in helpers. Dev then
    lost "Add your workspace URL here." (a caption restating its *label*). The anchor decides, not the role, so
    hits were split into heading hits and label hits and the imperative writes 64 into the novelty sum, which the
    label-anchored rules subtract (`b_novpl`). The lead verb's own weight is cancelled so "Upload a photo for your
    profile." under "Profile picture" leaves nothing. A gerund or third person in first position has the same
    stem ("Searching for a match.") and is treated the same: that only ever keeps a node.
26. **Kicker weights** (R2): round 1 flagged any eyebrow made of zero-weight words, so the filler eyebrow "Update"
    (a content category in dev) was DECOR. Now an eyebrow needs kicker weight ≥ 1 from explicit genre words; halves
    cover two-word kickers ("How it works", "Key facts", "Our team", "The basics", "Under the hood") without
    making "Team" or "Print" a genre word. Dev R2 recall 75% → 94%.
27. **Meta strips** (R2): coordinates needed decimal + degree + compass; "N 50.74" and "Local time 16:45" were
    missed. One weighted sum now (see `META` in `build.py`), and "nothing but numbers besides". "s" and "e" weigh
    less because "2.5 s" is a duration.
28. **Particles** carry a code: in round 1 "Sign in" and "Sign up" were the same bag (one coded word each), so two
    such headings were a repeated title. A particle is never a hit.
29. **Captions**: a caption needs two coded words to echo a heading (dev: 13 column-header captions such as "Bed"
    under "Busiest beds" were flagged in round 1).
30. **Action cue weights**: "Our class picked it up without any training." and "Pick the evidence you saw and the
    right breeding code is filled in." were narration because pick / choose / turn counted as gestures. They now
    weigh half and need a second cue ("switch … on or off"); "drag" moved to the hidden-input cues, which veto
    narration ("Long-press a row to select several.", "Drag photos here …"). (This cost most of the stretch
    column's recall; revision 57 is the narrow way back.)
31. **Promo-pill rule** needed a promo word positively: with particles coded, the badges "On" / "Off" on sibling
    rows were "a badge with no information besides promo words".
32. **Same stems, not same concepts, for duplicates**: bit 14 of the code is stem-specific. With broader groups a
    section "Members" under a page "Team" (probe) and "Summary" under "Overview" (dev) became repeated titles; the
    guide keeps the synonym case when the page has other sections.
33. **Filler lines**: "Everything you've saved, in one spot." has one half-weight word; the filler-only rule now
    also fires with ≥ 3 coded words and novelty < 0.75 against the previous nodes, for subtitles and closing lines.
34. **Narration wins the class** needed −10 on ECHO, not −1, once two echo rules could fire on one node.
35. **Lexicon, hand part**: rewritten into five group files. Decisions worth recording: team / member / people /
    user are one concept ("Members" / "The people on your team"); order / purchase / buy; search / find / result /
    match; "clear" joins the negation group at half weight ("Your inbox is clear."); plan / tier / package join
    subscription and "plan" is half weight ("The Plus plan" under "Plus"); "recent / latest", "prefer(red)",
    "standard" half weight; praise, hype and honesty adjectives are frames.
36. **Thesaurus rule**: see 2b. Rule A first accepted any hand word as an anchor through its top three senses and
    produced junk; both sides must now be specific. Hypernym merges were rejected by reasoning (K1 categories).
37. **Layer 4: summary lines** (section 5). First version: any summary cue, subtitles and closing lines, novelty
    against the window = 0. On dev it flagged plan descriptions ("For small clubs that share games.": "that" was
    a cue) and closing facts on pricing pages, where every noun recurs: 11 new false alarms. Now subtitles only
    and a whole summary frame (cue weight ≥ 4); the loose cues stay for the layer-3 closing-line rule.
38. An implementation fault worth remembering: in layer 2 an `EON` carries its code sum, so a matching head run
    there also fires for `EON` queries and `EON` keys. `m_any` got a dimension of its own and `EON` keys are banned
    (the ban itself was still too small: revision 59).
39. **Shared flag dimensions can be 2** (`pill` + `dot`, `in_nav` + `in_table`): an empty badge with both flags
    was a "single-letter pill" because the 2 stood in for the missing boolean. Rules that need such a flag are now
    graded on it behind G-weighted booleans (`graded_rule`).
40. **Meta-strip words** cut to HUD dressing and coordinates (sector, sys, cam, rec, lat, lon, local time): "index"
    ("Indexing…" is a live status), "node", "edition", "gmt / utc" ("All times in UTC" is information) came out,
    and a meta strip now always needs a number beside the meta word.
41. **Layer 4: the enumerating intro** (reference screen: the Education subtitle, an echo "spread over several"
    nodes, R14). First as "heading hit, ≥ 4 covered words, ≤ 1 uncovered, no clause marker": on dev it fired on
    pricing-page facts with no auxiliary ("Cancel free until 24 hours before.", "7,882 plot holders used Furrow
    last month."): 12 false alarms. Now the node must have the shape of a list (two list cues, no clause marker)
    and no number, and uncovered words count double against it; no dev false alarm. A heading hit was required
    at first and then dropped: what makes an enumeration redundant is that the screen shows the listed things,
    not that it shares a word with its heading (dev unchanged either way). "academic" joined the education group.
42. Domain groups for the kinds of product in `examples/README.md` (cooking: method / instructions, prepare /
    prep, pantry / larder, allergen, diet; research: corpus, ingest; games: respawn; personal sites: post / article
    / writing, project / portfolio), written as general vocabulary.
43. **Hit thresholds**: with hits split by anchor, a word found in both the heading and the label before it
    ("login" under h2 "Sign-in" and label "Two-step sign-in") shares its attention between two keys, so each hit
    was 0.5 and neither boolean fired. The two hit thresholds now sit below a third.
44. **Placeholders echo labels only** (R8). A placeholder that shares a word with the section heading and has no
    label of its own ("Recipe name" under "New recipe") is the field's only name. Under a field label a caption
    needs only one word ("Your handle." under "Username"); the two-word condition is for headings.
45. **End-to-end sources** (`work/probes/src_*.tsx|swift` through `./lint`): the extractor's roles are what the
    rules expect (CardDescription → subtitle, Section footer → helper, alert message → dialog_body,
    ContentUnavailableView → empty_title / empty_body). Found there: "All your saved recipes in one place." under
    "Library" was kept ("recipes" occurs only in the empty state, which the anywhere head could not see, and
    "saved" was news). The anywhere head lost its state pairing (a noun the page's own empty state uses is a noun
    the page is about) and save / saved joined library / collection ("Everything you've saved …").
46. **Lexicon review for precision** (every zero- and half-weight word read once more against sentences where it
    would carry the point): "moment" back to the loading group at weight 1 ("One moment…" as the last line of a
    loading state was a filler line), and the two closing-line filler rules skip conditional and state nodes;
    "left" / "right" are hidden-input cues ("Swipe left on a row to archive it."); "start" half weight ("You
    haven't started any chats yet."); long / time join duration ("How long it takes to cook." under "Cook time");
    "data" and "content" tried at weight 1 for K7 and put back to 0.5 ("Your data is safe and protected." under
    "Secure by default" is the codex's own L41 echo, and a real loss sentence has "will", "permanently" or a loss
    verb besides).
47. **Antonym audit** (`work/scratch/antcheck.py`): borrow / lend and question / answer separated (answer now
    sits with reply / response); small / big, which round 1 had as one "size" concept, split by hand.
48. **Game-menu vocabulary** (dev recall on game screens was 0.46; the motivating fixture is a game menu):
    "over" keeps its particle class but takes the code of end / finish / stop, so "Game over" is restated by
    "Run ended" and "The end"; defeat / defeated / died / dead join that concept (lose / lost / loss stay apart:
    "changes will be lost" is a consequence); beat joins win; matchmaking joins loading / wait; "game", "play" and
    "match" are half weight ("Configure the game." under "Settings"). Game screens: 0.70.
49. Three merges from codex examples and dev: billing / bill / invoice with pay / payment (codex U47: "Billing" /
    "Payment details and invoices for your plan."); workspace with team / member; choice with choose / select.
50. **A stat is an anchor**: stat nodes share the button's key dimension, so a note that repeats the stat's
    figure and its label ("Foot passengers: 4,753" under "4,753") is covered. Dev R5 50% → 56%; a stat still has
    no rule of its own. The pointer for these notes goes to the label (two matched words) rather than the stat;
    a pointer bonus "a line with a number → the stat" was tried and reverted (most lines with a number restate a
    heading that carries the number: dev pointer 91.0% → 89.9%).
51. **Pointer for repeated facts** (R14 text): full-weight helper / caption / subtitle words now join the pointer
    query through a match in an earlier text line, but only when they were not matched just before (round 1 left
    them out entirely because a word shared with the label above pulled the pointer to a far heading; a frame
    verb matched far away did the same until the gate was limited to full-weight words). Dev pointer 89.7% → 92%.
52. "confirm" is a frame word ("Confirm: the pet will be archived." restates "Archive this pet?").
53. **"off" and "down" weigh 1** (still particles: never a hit): "Notifications are off" under "Notifications"
    is a state, and "20% off" a fact. "on" cannot follow: it is also the stem of "one" and a preposition.
    Narration weight is 0 for every action cue, the weak ones and "on / off" included (the split of the action
    list had dropped that).
54. **First body line of a section** (R4): the rule reads the novelty raw (nothing at all left over): "Your
    recent activity will show up here." under "Recent activity" is that section's only statement that it is
    empty. "receive" is half weight ("Messages you receive will show up here." implies the next action, K8).
55. **Narration, three dev false alarms by template**: "above" is half a layout word ("on the Plus plan and
    above" = or higher); "plus" got weight (see 62); the two R16 rules with "here / this" are graded on 2 − number
    tokens, so "Here are four of your 16 key numbers; …" (a partial count, a fact) is kept.
56. About 45 strict synonyms picked by hand from `suggest.py`'s listing (2b), and "selected / recent" as half
    kickers ("Selected work" over "Projects", from the personal-site source probe).
57. **Narration with half a cue: a control named in full** (codex U11: "a button label named inside the body
    plus a narration verb"). After revision 30 the stretch column found 60% of what round 1 found 87% of, because
    "Choose Save changes to continue." has only half a cue. "Choose" is not interface wording by itself, so the
    evidence has to be that the verb's object *is* the control. Word order is invisible to a bag of concepts; what
    stands in for it took three versions, each read on a probe written for it (`r2_narr`):
    (a) "a word of a control below, the control's whole label named (`b_ctl`), at most one new word": flagged
    "Choose a photo to upload." above [Upload] — the one new word was the real object;
    (b) "no new word at all" (`nov_x` = 0): still flagged it under the heading "Profile photo", which covers
    "photo";
    (c) now: no word of narration weight outside the *controls below* (`nov_ct` → `novc`, read raw), and no number.
    "Choose a photo to upload.", "Choose a plan to upgrade.", "Select a workspace to continue.", "Pick a
    playlist." above [New playlist] and "Switch to annual and save 20%." are kept; the stretch column is at 81%
    with no false alarm. The new pieces: the control's own tokens count their words that no narrating line above
    repeats (L1), the layer-3 named-control head carries "a control below has none left" back to the line, and
    the rule is graded on 4·cue − 1. The dimension came from the boolean `b_novx3`, whose one rule now reads the
    novelty raw (11 − 4·`nov_x`), deciding exactly as before.
58. **Narration padding and particles** weigh nothing in the narration novelty (`narrpad.txt`; "down" in "Scroll
    down to see more boats." is the gesture's direction). "when" and "once" keep their weight: "Press Confirm once
    you are happy." passes on the one word a whole-cue line may add, a real condition ("… when the upload has
    finished") does not, and a half-cue line with either is kept (see known failures).
59. **A fault in the anywhere head** (found because one dev subtitle changed its call when only the *order* of
    the vocabulary changed): `EON` keys were banned with the usual −60, but in layer 2 an `EON` holds a code sum,
    and the sum of an eight-word node lines up with some word's code about one time in a hundred. An unknown word
    ("studio" in "Profile for your whole studio, in one place.") was then "found elsewhere" by chance and the
    summary-line rule fired. The ban is −4000 now (above any code-sum product); no call depends on the
    vocabulary order through this head any more.
60. **Pointer head in layer 4** (same weights), so that it sees the `b_act` set by the rule of revision 57: such a
    line now points at the button it names rather than at a title sharing the word (dev pointer back to 92%). A
    bonus for button tokens that a narrating line just before repeats was tried and taken out: it changed nothing
    on dev or the probes.
61. **A subtitle is weighed against its own anchor.** `r2_real2` (written before it was run): "Everything in
    Free, plus:" under the plan "Pro" was called an echo of the heading "Free" of the card before, three nodes
    back, and "plus" weighed a half. A subtitle's heading is the node directly before it, so the novelty of a
    subtitle (and its imperative test) now reads a match head that sees the two nodes before only (`m_pn`). The
    head came from moving the sums of the lexical classes, and the booleans made from them, from layer 1 to layer
    2 (nothing before layer 3 reads them). First I gave *every* role except list lines the short reach; that lost
    what the wide reach is right for: an empty or dialog body restating its title in the words of the page
    heading ("The tutors list has nothing in it." under "Nothing here yet" on the page "Tutors"), and a field in
    a card ("Name of your project" under "Name" in "Create project"). `farcov.py` showed both sides: 315 KEEP
    nodes on dev have a word covered only three or four nodes back, nearly always by a neighbouring item.
62. **"plus" is a marker** (weight 1): as a conjunction it announces an addition, like "also"; as a plan name
    it is a content word, covered when the heading has it. (It weighed 0 as a summary cue at first, then a half.)

**Final state** (artefact `9fa3f37d61f7…`; round-1 column: the round-1 artefact on this kit, as measured at the
start of the round):

| | Round 1 artefact | Round 2 |
|---|---|---|
| dev (1,200 screens) F1 / precision / recall | 0.771 / 0.938 / 0.655 | 0.813 / 0.982 / 0.694 |
| dev false alarms in 20,288 KEEP nodes | 56 | 16 |
| dev four-class macro-F1 | 0.855 | 0.903 |
| dev `ct` column precision / recall | 0.977 / 0.869 | 1.000 / 0.808 |
| dev pointer accuracy | 92.6% | 92.0% |
| reference (4 redundant in 146 headline nodes) | 1 found, 0 false alarms | 4 found, 0 false alarms; the `ct` node too |
| my probes (15 files) | — | 166 of 171 redundant, 0 flags on 956 KEEP |

The three invented false alarms of RESULTS.md ("All items in this list will also be removed.", "Pick a
playlist.", "Your profile is public, so anyone can find it.") are kept, and its invented misses ("Everything
you've saved, in one spot.", "THE LOWDOWN", "Your inbox is clear.", "The name people see.", "The people on your
team", "Things you've bought", "Dishes you've saved") are flagged.

## 8. Known failures

On the dev examples (393 misses, 16 false alarms):

- **Paraphrase with no lexical link** (most misses, labelled `judgement` there): "Tweak how it plays." for
  "Options", "Something only you know" for "Password", definitional helpers and placeholders ("Proof you can
  hold a rope." for "Belay certificate", "What your team is called" for "Workspace name"), domain synonyms the
  lexicon does not hold (fares / tickets, canoes / boats, brews / batches), "Level failed" for "Game over",
  "could not be completed" for "failed".
- **Restatement plus one unknown word**: "Extra login protection.", "Files you've downloaded.", "Everything about
  your batches in one place." when "batches" is nowhere else in the window. One unmatched content word is
  information (that is what protects "Reading history for every branch" and most K4 helpers).
- **Emphasis tails with a marker or negation**: "…; nothing else.", "…, no older.", "Only the last 60 days of
  classes.", "Skip a week for everyone." Markers always count; that is the price of keeping "Only you can see
  these."
- **Imperative restatements**: "Start another one." under "Add pet", "Give this visit a new name." under "Rename
  visit": an imperative lead is news against a title.
- **Narration with an ordinary verb and anything else in the line**: "Select Save when you are ready.", "Make
  your changes, then choose Leave queue.", "Enter your details and hit Save." (a marker or an instruction clause
  besides the button's name); narration that names no control ("Select an option to begin.", "Use the menu to
  navigate."); a button whose name is a hidden-input word ("Press Delete shift to get started.").
- **Subset and synonym duplicates** of headings (h3 "Board booking" under h2 "Board booking, on every plan"; h2
  "Connected apps" under h1 "Integrations"); repeated buttons, labels and stats (never flagged: in real files the
  same button text in two places is usually two separately mounted views, as in the reference screens);
  paraphrased repeated facts ("Cancel any time." / "Leave any time, no notice period.").
- False alarms that remain on dev: repeated headings the generator keeps (h2 = h1 in some list screens, stat-card
  headings), stat-card captions that repeat part of their heading ("In Aberystwyth"), a sentence repeated as
  subtitle and helper, an eyebrow "Settings" over "Preferences" (labelled a parent section).

By design, to be checked on real screens:

- "X is on." under a heading or label X ("Notifications are on.") is called an echo: "on" weighs nothing.
- Nodes in different branches of one conditional are matched like any neighbours unless they are joined with
  `ALT` or one is in a state cluster and the other is not: a caption in one branch can be called an echo of a
  heading in the other. The flags do not say which conditional a node belongs to.
- A helper, caption, placeholder or list line is weighed against rank 1-3 nodes up to four back, and those can
  belong to the section before when its own heading is nearer than that (only subtitles have the short reach). A
  subtitle that comes more than two nodes after its heading is weighed against nothing and kept.
- The filler-line rules flag a line with no hit at all ("We're here to help.", "View all" if it is extracted as a
  subtitle); the tracked-label kicker rule flags a kicker-styled label made of genre words even where it works
  as a field label ("Summary" over a summary).
- The `ct` rule for a subtitle that begins with a control's name ("Resume recaptures the mouse.") can flag an
  instruction such as "Save your changes before leaving." above a "Save" button; the half-cue rule flags a line
  made only of an ordinary verb, padding and a button's whole name even where it is the state's next action
  ("Choose New project to begin." in an empty state: the examples label these NARRATE, K8 could argue).
- Word order and negation scope are invisible (bags of concepts); Porter merges remain (active / activity, book /
  booking, organ / organization, operation / operational, ad / added).
- Hash buckets: two different unknown words fall into the same bucket once in 8,192 pairs and then match.
- Pointer: about 92% on dev. It compares 11 code bits, so with two or three matched words an unrelated flagged
  token sometimes outscores the true one (which token, depends on the order of the vocabulary); of two buttons
  with the same word the earlier wins, also for a narrating line that means the one below it; calls without a
  matched word have no target; a stat note points at the stat's label; a repeated fact can point at a nearer
  node that shares a word.
- Class calls depend on the order of the vocabulary only through the duplicate head's 10-bit comparison (two
  different bags with the same code sum are possible in principle; none was seen).
