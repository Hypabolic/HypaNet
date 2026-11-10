# Slop codex: LLM writing patterns and UI copy patterns

A catalogue of redundant, decorative and generated-sounding copy in user interfaces, for the slop linter's network,
its plain-code rule layer and anyone writing UI copy. It has two parts with stable ids:

- **Part L (L01–L52), LLM writing markers**: what a large language model does when it writes app and website copy,
  from research on LLM vocabulary and practitioner catalogues of "AI slop".
- **Part U (U01–U52), UI copy patterns**: what established UI-writing guidance (Microsoft, Apple, Material, Google,
  GOV.UK, Mailchimp, Shopify, Atlassian, Nielsen Norman Group) and the prose linters say about redundant, decorative
  and narrating copy, and how to tell it from copy that does a job.

`LABELLING.md` (the label guide) is the definition of the classes, and this codex does not change it. Where an entry
goes beyond the guide it says **beyond the guide**: the guide does not settle that case, and labels follow the
guide, not this file. All examples are invented for the codex. Style guides are quoted only in short phrases with a
link. Word and phrase lists are in `lists/` (§11); sources, licences and what was taken from each are in `SOURCES.md`
(cited by key, e.g. [wp-signs]) and, for imported entries, in `lists/ATTRIBUTION-ui.md` and each list's header.

## Contents

1. How to read an entry
2. How the codex maps onto `LABELLING.md`'s classes and the rule layer
3. Index of patterns
4. Why models write like this (causes A–H)
5. Part L: LLM writing markers (groups L-A to L-F)
6. Part U: UI copy patterns (groups U-A to U-H)
7. Decision aid: sentences that add a consequence, instruction or reason
8. Button labels repeated across views
9. Detection notes
10. What the prose linters cover
11. The lists
12. Out of scope
13. Short source keys of Part U

## 1. How to read an entry

**Part L** entries give **What** (the pattern), **Why** (cause letters from §4), **Examples**, **Detection**,
**Maps to** and **Sources**. Detection tags:

- `lexical`: a word or phrase list decides it.
- `pattern`: a construction regex over folded words (`lists/llm_constructions.txt` syntax).
- `structural`: needs roles, positions or sibling structure from the screen JSON.
- `context`: needs meaning (a paraphrase, an obvious consequence) and so the network or a human.
- `glyph`: decided on raw text before folding.

**Part U** entries give:

Each pattern has: the **guidance** it rests on (short quote and link), **examples** (redundant first, then the useful
contrast), **redundant or useful** (the test that separates them), **detect** (cues for the network or the rule
layer), and **maps to**: the class (`ECHO`, `NARRATE`, `DECOR`, `KEEP`), the R code or aside from LABELLING §9, and
who owns it (network, rule layer, out of scope). "Anchor" means the node a lower-ranked node would echo (LABELLING §5
rank order: controls and main headings, then h2–h6, then body-like roles, then subtitle/helper/caption, then
eyebrow/badge/tooltip/placeholder).

The short keys (MS-UIT, APL-A, NN-CD, …) are listed in §13.

## 2. How the codex maps onto `LABELLING.md`'s classes and the rule layer

The linter has two halves. The **network** gives every visible node one class: `KEEP`, or one of the three redundant
classes `ECHO` (restates information a visible higher-ranked node already gives), `NARRATE` (explains the visible
interface) and `DECOR` (carries no information). The **rule layer** is plain code that reports what word lists,
regexes and counts decide (the asides of `LABELLING.md` §9: vacuous actions R7, placeholder-as-label R8b, shared
lead-ins R10, index numbering R12, filler R15, repeated chrome R19, glyph leads R11g), scored apart from the network.

- A pattern is network territory only when its typical instance is **redundant** under `LABELLING.md` §4. A
  promotional word is not redundancy: "Seamless sync" is slop, but deleting it loses the fact that the app syncs.
  Most LLM markers (Part L groups L-A to L-D, L-F) are rule-layer findings or claims no linter can check, and the
  network should not flag a node because it contains hype words.
- Every entry's **Maps to** line names the class its redundant instances take, the R code or K rule it rests on, and
  its owner. By the class their redundant (or, for guards, useful) instances take:
  - `ECHO`: L12, L14, L17, L25, L28, L31, L33, L34, L35, L38, L40, L41, L42, L43, L52, U01, U02, U04, U05, U06, U07, U08, U10, U15, U18, U19, U20, U21, U23, U28, U31, U34, U36, U46, U47.
  - `NARRATE`: L20, L36, L37, U11, U15, U24, U30, U32, U37, U38, U39, U41.
  - `DECOR`: L05, L07, L17, L19, L20, L27, L30, L34, L38, L39, L46, L47, L50, U02, U03, U12, U16, U42.
  - `KEEP` (guards against false alarms, or the class of the useful contrast): L13, L26, L27, L31, L40, L42, L47, U09, U10, U13, U14, U19, U25, U26, U28, U35, U36, U40, U43, U45, U48.
  - No class (rule layer or out of scope only): L01, L02, L03, L04, L06, L08, L09, L10, L11, L15, L16, L18, L21, L22, L23, L24, L29, L32, L44, L45, L48, L49, L51, U17, U22, U27, U29, U33, U44, U49, U50, U51, U52.
- Seven Part U entries are **KEEP guards**, not slop: U09, U13, U14, U25, U35, U40, U43 and U48. They describe the
  copy a detector must not flag: a sentence that restates and adds, a consequence body, an instruction, a useful
  helper, an empty state with a cause or next action, a hidden-input hint, a bound status and consistent term reuse.
- The lists are **cues, not verdicts**. A narration word in a sentence that also states a consequence is still
  `KEEP`; §7 is the procedure. Lists are written as natural lower-case words and phrases, folded the way the rule
  layer folds text; the network sees Porter stems instead, so a list entry reaches it only through its stem.
- **Signal strength.** Lexical markers are weak one at a time: every catalogue says so ([wp-signs] "one or two may be
coincidental"; [gptzero] term lists underperform a bag-of-words baseline). The rule layer should score density (several
distinct markers in one node or one screen, or one marker repeated across siblings), and the strongest single-node
signals are structural (L09, L10, L13, L14, L17) or residue (L44, L45).
- **Era drift.** The vocabulary moves with model generations. Wikipedia's era lists ([wp-signs]) show "delve" peaking
in 2023 and falling away in 2025, and a newer set ("quietly", "steady", "practical", "clearer", "matters") for models
from mid-2026. Em dashes were the best-known tell until about September 2026, after which newer models are often tuned
not to emit them at all. A list frozen today will age; the structural patterns age more slowly than the words.

## 3. Index of patterns

### Part L

| Id | Pattern | Detection | Home | Network class of redundant instances | List |
|---|---|---|---|---|---|
| L01 | Hype verbs | lexical | rules | none | llm_promo_vocab |
| L02 | Inflating adjectives, stacked adverbs | lexical | rules | none | llm_promo_vocab |
| L03 | Decorative abstract nouns | lexical (weak) | rules | none | llm_promo_vocab |
| L04 | "Delve"-class excess vocabulary | lexical | rules | none | llm_excess_vocab |
| L05 | Significance puffery | lexical | rules | DECOR (rare) | llm_excess_vocab |
| L06 | Era drift vocabulary | lexical (weak) | rules | none | llm_excess_vocab |
| L07 | Empty personalisation | pattern | rules | DECOR as a lone subtitle | llm_constructions |
| L08 | "AI-powered" headline | lexical, glyph | rules | none | llm_promo_vocab |
| L09 | Negative parallelism | pattern | rules | none | llm_constructions |
| L10 | Audience-spanning hedge | pattern | rules | none | llm_constructions |
| L11 | False range | pattern | rules | none | llm_constructions |
| L12 | Rule of three | pattern, structural | rules (R13) | ECHO when it restates | llm_constructions |
| L13 | Rhetorical question then answer | pattern | rules | KEEP over its button (K11) | llm_constructions |
| L14 | Before/after pivot, "reimagined" | pattern | rules | ECHO → wordmark (L33) | llm_constructions |
| L15 | Mirrored benefit clause | pattern, structural | rules (R10) | none | llm_constructions |
| L16 | Trailing participle | pattern | rules (span) | none | llm_constructions |
| L17 | Grandiose openers and closers | pattern | rules | ECHO → hero (R14) / DECOR | llm_constructions |
| L18 | Copula substitutes | pattern | rules | none | llm_constructions, llm_excess_vocab |
| L19 | Staccato fragments | pattern | rules | DECOR as a lone node | llm_constructions |
| L20 | Throat-clearing openers | lexical | rules | NARRATE / DECOR | llm_filler |
| L21 | Emphasis crutches | lexical | rules (R15+) | none | llm_filler |
| L22 | Connective filler | lexical | rules | none | llm_filler |
| L23 | Capability hedges | lexical | rules | none | llm_hedges |
| L24 | Vague attribution | lexical | rules; out of scope (truth) | none | llm_hedges |
| L25 | Chirpy system messages | lexical | rules (R15+) | ECHO if it restates (R4) | llm_hedges |
| L26 | Stock CTA labels | lexical | rules (R7) | KEEP | llm_cta_cliches |
| L27 | CTA-band headline | lexical, structural | rules | KEEP (K11) / DECOR | llm_cta_cliches |
| L28 | Speed and ease promises | lexical | rules; out of scope (truth) | ECHO on repeats (R14) | llm_claims |
| L29 | Invented precision | pattern | rules; out of scope (truth) | none | llm_claims |
| L30 | Vague social proof | lexical | network + rules | DECOR | llm_claims |
| L31 | Reassurance boilerplate | lexical, count | network + rules | KEEP once (K6), ECHO repeats | llm_claims |
| L32 | False urgency | lexical | rules; out of scope (truth) | none | llm_claims |
| L33 | Product-name greeting | pattern, structural | network | ECHO → wordmark | llm_constructions |
| L34 | Summary closers | lexical, structural | network | ECHO (null) / DECOR | llm_filler |
| L35 | Paraphrasing subtitle | context | network | ECHO → heading | (thesaurus) |
| L36 | Captions narrating controls | lexical cue, context | network | NARRATE → control | llm_narration |
| L37 | Layout self-description | lexical cue, structural | network | NARRATE | llm_narration |
| L38 | Genre eyebrows and pills | lexical, structural | network | ECHO / DECOR | llm_kickers |
| L39 | Editorial-label kickers | lexical, structural | network | DECOR | llm_kickers |
| L40 | Question-form genre headings | lexical | rules (low) | KEEP, ECHO on duplicates | llm_kickers |
| L41 | Bold lead-in restated | context | network | ECHO → title | (thesaurus) |
| L42 | Model-voice FAQ | context, structural | network | ECHO / KEEP partial | — |
| L43 | Synonym cycling | context | network + rules | ECHO | (thesaurus) |
| L44 | Chatbot residue | lexical | rules (error) | none | llm_residue |
| L45 | Placeholder identities | lexical | rules | none | llm_residue |
| L46 | Emoji bullets and sparkles | glyph | rules (R11g) | DECOR if emoji-only | llm_glyphs |
| L47 | Static status pills | lexical, structural | network | DECOR / KEEP if bound | llm_kickers |
| L48 | Em dash habit | glyph, count | rules | none | llm_glyphs |
| L49 | Title Case, all caps | case test | rules | none (case never decides) | — |
| L50 | Middle-dot strips, arrows | glyph, structural | rules + network | DECOR meta strip | llm_glyphs |
| L51 | Exclamation marks | glyph, count | rules | none | llm_glyphs |
| L52 | Metronomic siblings | structural | rules | ECHO per restating body | — |

### Part U

| Id | Pattern | Class | Home |
|---|---|---|---|
| U01 | Kicker restating the heading | ECHO | network |
| U02 | Section-genre kicker | ECHO, DECOR | network, rule layer |
| U03 | Decorative label strip | DECOR | network |
| U04 | Subtitle restating the heading | ECHO | network |
| U05 | Generic-verb subtitle | ECHO | network |
| U06 | Heading echoed in the section's first or last line | ECHO | network |
| U07 | Title repeated across chrome | ECHO | network |
| U08 | Restating closer | ECHO | network |
| U09 | Subtitle that restates and adds | KEEP | network |
| U10 | Dialog body restating the title question | ECHO, KEEP | network |
| U11 | Body explaining the buttons | NARRATE | network |
| U12 | Vague warning | DECOR | network |
| U13 | Consequence body (the false-alarm guard) | KEEP | network |
| U14 | Instruction sentence | KEEP | network |
| U15 | Instruction that restates a visible control | ECHO, NARRATE | network |
| U16 | Status-genre title | DECOR | network |
| U17 | "Warning:" and "Note:" prefixes | - | network, rule layer |
| U18 | Success or status body restating its title | ECHO | network |
| U19 | Error body restating its title | ECHO, KEEP | network |
| U20 | Placeholder repeating the label | ECHO | network |
| U21 | Placeholder as instruction plus label | ECHO | network |
| U22 | Placeholder used instead of a label | - | rule layer |
| U23 | Helper restating the label | ECHO | network |
| U24 | Helper narrating the control's mechanics | NARRATE | network |
| U25 | Useful helper (format, limit, eligibility, timing, scope, data use) | KEEP | network |
| U26 | Toggle description covering both states | KEEP | network, rule layer |
| U27 | Instructional verb in a label | - | network, rule layer |
| U28 | Tooltip repeating the visible label | ECHO, KEEP | network |
| U29 | Vacuous action label | - | network, rule layer |
| U30 | Gesture lead or pointer | NARRATE | network, rule layer |
| U31 | CTA repeated across sections | ECHO | network |
| U32 | Control plus caption saying what it does | NARRATE | network |
| U33 | Same lead-in on every sibling | - | rule layer |
| U34 | Empty body restating the title | ECHO | network |
| U35 | Empty body with cause or next action | KEEP | network |
| U36 | Loading narration | ECHO, KEEP | network, rule layer |
| U37 | Copy about the screen itself | NARRATE | network |
| U38 | Literal count of visible items | NARRATE | network |
| U39 | Ordinary platform behaviour | NARRATE | network |
| U40 | Hidden-input hint | KEEP | network |
| U41 | Tour narration of the visible UI | NARRATE | network |
| U42 | Static status claim or decorative pill | DECOR | network |
| U43 | Bound status | KEEP | network |
| U44 | Repeated device on every sibling | - | rule layer |
| U45 | Index numbering | KEEP | network, rule layer |
| U46 | Same fact repeated across sections | ECHO | network |
| U47 | Synonym-cycled echo | ECHO | network |
| U48 | Consistent term reuse | KEEP | network |
| U49 | Filler words | - | rule layer |
| U50 | Padding phrases | - | rule layer |
| U51 | Redundant acronym phrases | - | network, out of scope |
| U52 | Promotional and unverifiable claims | - | out of scope |

## 4. Why models write like this

The causes recur across patterns, so each entry cites them by letter.

- **A. The template prior.** Training data holds millions of near-identical SaaS landing pages, starter kits and
  templates. Asked for "a landing page", a model returns the average one: hero, eyebrow, three features, stats, CTA
  band, each slot filled with that slot's most typical words ([unslop-ui] §4 "the codified section order";
  [tasteskill] "the average of every landing page").
- **B. Preference tuning.** Raters reward text that sounds thorough, polished and positive. Juzek and Ward's focal
  words point at preference tuning rather than pre-training as the likeliest source of "delve"-class words
  ([juzek]); the same pressure rewards upbeat adjectives and reassurance.
- **C. The instruction-tuned register.** Instruction-tuned models write a dense, noun-heavy, participle-rich style
  (present participial clauses at about 2-5 times the human rate, nominalisations at 1.5-2 times; base models are
  close to human rates) ([reinhart]).
- **D. Slot filling.** A component with a slot gets the slot filled: every heading gets an eyebrow and a subtitle,
  every toggle a caption, every card an icon. With no fact to put there, the model restates the neighbour or names
  the genre. This is where most network-class redundancy comes from.
- **E. Prompt framing.** Prompts say "compelling", "engaging", "modern", "friendly". The model obeys with the words
  that most often co-occur with those descriptions: hype verbs, exclamation marks, emoji.
- **F. Chat-register leakage.** The model's assistant voice ("Certainly!", "Here's a", "I hope this helps"), its
  hedges and its summaries leak into strings meant for end users ([wp-signs] §Collaborative communication).
- **G. Variation pressure.** Repetition penalties and style training discourage repeating a word, so models cycle
  synonyms ([wp-signs] §Lexical diversity/elegant variation). In UI this hides echoes behind paraphrase.
- **H. No facts to hand.** The model does not know the price, the user count or the setup time, so it writes
  plausible ones ("10,000+ teams", "in minutes") or generic praise in their place ([deslop] "a concrete fact beats an
  adjective").

## 5. Part L: LLM writing markers

### L-A. Vocabulary

#### L01 Hype verbs
- **What:** Verbs that promise transformation without saying what changes: unlock, unleash, supercharge, elevate,
  empower, streamline, transform, revolutionise, harness, leverage, level up, reimagine.
- **Why:** A, B, E. They are the most frequent predicate of "your [workflow/productivity/potential]" in template copy.
- **Examples:** "Unlock your team's full potential." / "Supercharge your invoicing." / "Elevate every meeting with
  smart notes." / "Harness the power of your data."
- **Detection:** `lexical`. Strongest with a possessive abstract object ("your potential", "your workflow"). Weak
  alone: "transform" is literal in an image editor, "streamline" in a logistics tool.
- **Maps to:** rules, `lists/llm_promo_vocab.txt` §L01; density finding, not an error. Network: none (a hype verb in a
  node that also restates its heading is still judged ECHO on the restatement, L35).
- **Sources:** [unslop-ui] §8 "Adjective reservoir"; [deslop]; [sathya]; [kobak] (elevate, harness, unlocking as
  style words); [tasteskill] §6.

#### L02 Inflating adjectives and stacked adverbs
- **What:** Adjectives that grade rather than describe (seamless, effortless, robust, powerful, cutting-edge,
  world-class, intuitive, all-in-one) and adverbs stacked on every verb (seamlessly, effortlessly, instantly).
- **Why:** A, B. They are safe positives that fit any product, so they dominate the average.
- **Examples:** "A seamless, intuitive experience." / "Effortlessly track every expense." / "Robust, enterprise-grade
  security." / "Blazing-fast search."
- **Detection:** `lexical`. Count distinct entries per node; two or more in a short node is a strong signal. "Robust"
  and "secure" are legitimate in technical specs when a measurable claim follows.
- **Maps to:** rules, `llm_promo_vocab.txt` §L02. Network: none.
- **Sources:** [unslop-ui] §8 "Adjective reservoir", "Adverb stacking"; [wp-wtw] MOS:PUFFERY; [deslop]; [stop-slop]
  "Adverbs"; [kobak] (seamless, seamlessly, exceptional, remarkable, innovative).

#### L03 Decorative abstract nouns
- **What:** Nouns that stand where a concrete object should: journey, experience, solution, ecosystem, landscape,
  workflow, productivity, potential, possibilities, peace of mind.
- **Why:** C (nominal style), A.
- **Examples:** "Your wellness journey starts here." / "Solutions for every workflow." / "A better hiring experience."
- **Detection:** `lexical`, weak: most of these are ordinary words. Report only beside L01/L02 entries or in a
  hero/subtitle slot.
- **Maps to:** rules, `llm_promo_vocab.txt` §L03 (density only). Network: none.
- **Sources:** [wp-signs] WP:AIVOCAB ("landscape", "tapestry" as abstract nouns); [aismells]; [reinhart]
  (nominalisations).

#### L04 "Delve"-class excess vocabulary
- **What:** Words measured as over-used by LLMs in large corpora: delve, tapestry, realm, testament, pivotal,
  intricate, meticulous, underscore, showcase, boasts, vibrant, garner, bolster, multifaceted, commendable.
- **Why:** B. The measured lists come from abstracts, reviews and encyclopaedia edits; they transfer to blog-style
  and "about" copy more than to buttons and labels.
- **Examples:** "Delve into your analytics." / "A vibrant tapestry of recipes." / "Meticulously crafted for creators."
  / "Our platform boasts over 40 integrations."
- **Detection:** `lexical`. Rare in UI strings, so a single hit in a heading or subtitle is worth reporting; in body
  text use density. Do not expand with synonyms: [wp-signs] warns that overuse of a word does not imply overuse of its
  synonyms.
- **Maps to:** rules, `lists/llm_excess_vocab.txt` (CC BY-SA 4.0). Network: none.
- **Sources:** [kobak]; [juzek]; [liang]; [wp-signs] WP:AIVOCAB; [paech-sf] (essay bigrams confirm "ever evolving
  landscape", "unlock full potential").

#### L05 Significance puffery
- **What:** Phrases that assert importance instead of showing it: "plays a pivotal role", "is a testament to",
  "underscores the importance of", "sets the stage for", "leaves a lasting impact".
- **Why:** B, C. Wikipedia lists this as the first content sign of AI text.
- **Examples:** "Good sleep plays a pivotal role in your health." (in a sleep tracker's onboarding) / "Our app is a
  testament to simple design." / "Reminders that underscore what matters."
- **Detection:** `lexical` (phrase list). Strong when it appears in product copy at all.
- **Maps to:** rules, `llm_excess_vocab.txt` §C phrases. Network: none; if the whole node only restates the section
  heading's importance, it is DECOR by the deletion test.
- **Sources:** [wp-signs] "Undue emphasis on significance"; [unslop-ui] §8 "Promotional puffery"; [paech-sf] ("plays
  crucial role", "testament enduring power").

#### L06 Era drift: newer-model vocabulary
- **What:** The words newer models over-use once the old ones are suppressed. Wikipedia's mid-2026 list: challenge,
  clearer, dependable, echoed, foster, leverage, matters, multifaceted, practical, prioritize, quietly, steady,
  universally; with shorter sentences of about 8-20 words and more full stops.
- **Why:** B with suppression: vendors tune out the notorious tells ("delve", em dashes) and the probability mass moves
  to plain-sounding words.
- **Examples:** "Quietly powerful budgeting." / "A steady, dependable way to plan your week." / "Practical tools
  for clearer finances."
- **Detection:** `lexical`, low precision: these are plain words. Useful only as a density feature with L01-L05, and
  "quietly" as an adverb on a product claim is the most distinctive.
- **Maps to:** rules, `llm_excess_vocab.txt` §C (GPT-6 era). Network: none. Revisit each round; expect this list to
  move again.
- **Sources:** [wp-signs] WP:AIVOCAB era lists and §Overuse of em dashes.

#### L07 Empty personalisation
- **What:** Claims of fit that name no user and no workflow: "designed with you in mind", "built for the way you
  work", "your money, your way", "tailored to your needs".
- **Why:** A, E ("friendly", "personal").
- **Examples:** "Built for the way you cook." / "Your calendar, your way." / "Personalised insights, tailored to
  you."
- **Detection:** `pattern` (`llm_constructions.txt` L17 entries) plus `lexical` (tailored, personalised).
- **Maps to:** rules. Network: a subtitle that only says this under a heading is DECOR by the deletion test (it carries
  no fact); a hero line that also names the product's job is KEEP.
- **Sources:** [unslop-ui] §8 "Empty personalization".

#### L08 "AI-powered" as the headline
- **What:** The technology named as the value: "AI-powered", "Harness the power of AI", a sparkle beside every AI
  feature.
- **Why:** A (2024-2026 template pages), E.
- **Examples:** "AI-powered meal planning." / "Supercharged by AI." / "✨ Smart suggestions, powered by AI."
- **Detection:** `lexical` (`llm_promo_vocab.txt` §L08) plus `glyph` (✨, L46).
- **Maps to:** rules. Network: none; the pill form ("Now with AI") is L47.
- **Sources:** [unslop-ui] §5 "Hero pill badge", §7 sparkle, §8 "AI-powered".

### L-B. Constructions

#### L09 Negative parallelism and the correction
- **What:** Defining by denial: "It's not just X, it's Y", "More than a Y", "Not X. Y.", "No X, no Y, just Z", and
  the reversed "Y rather than X".
- **Why:** B, C. The construction sounds insightful and is cheap to produce; it pre-empts a misconception nobody had.
- **Examples:** "It's not just a to-do list. It's your second brain." / "More than a calendar." / "No spreadsheets, no
  guesswork, just answers."
- **Detection:** `pattern` (L09 block). High precision in headings and subtitles; in body text a real contrast
  ("not included in the free plan") is common, so require the X and Y halves to be noun phrases with no number.
- **Maps to:** rules. Network: none, unless one half restates a visible node (then ECHO on the node).
- **Sources:** [wp-signs] WP:AIPARALLEL; [stop-slop] "Binary contrasts", "Negative listing"; [unslop-ui] §8;
  [tasteskill] "The correction"; [paech-as] regex "not ... but".

#### L10 Audience-spanning hedge
- **What:** "Whether you're X or Y", "for teams of all sizes", "from freelancers to enterprises": naming every
  audience so as to commit to none.
- **Why:** A, B (inclusive, safe).
- **Examples:** "Whether you're a beginner or a pro, we've got you covered." / "For teams of all sizes." / "From solo
  creators to global brands."
- **Detection:** `pattern` (L10). Very high precision at node start.
- **Maps to:** rules. Network: none.
- **Sources:** [unslop-ui] §8 "Audience-spanning hedge"; [deslop]; [wp-signs] lead image caption ("Whether powered by
  ...").

#### L11 False range
- **What:** "From X to Y" where X and Y are not ends of a scale, used to sound comprehensive.
- **Why:** C, A.
- **Examples:** "From grocery lists to wedding plans, Tally keeps it all." / "Everything from invoices to insights."
- **Detection:** `pattern` (L11) at node start; weak inside sentences ("from 9 to 5" is a real range; skip when X or Y
  is a number or date).
- **Maps to:** rules. Network: none.
- **Sources:** HypaNet generalisation of the [wp-signs] WP:AIPUFFERY example ("From its scenic landscapes to its
  historical landmarks").

#### L12 Rule of three and the tricolon tagline
- **What:** Three parallel words or clauses where the content has fewer distinct ideas: "Fast. Simple. Secure.",
  "Plan, track and grow", three features, three stats, three steps.
- **Why:** A, B; three reads as complete.
- **Examples:** "Simple. Smart. Yours." / "Capture, organise and share your ideas." / "Track it. Split it. Settle it."
- **Detection:** `pattern` (L12) for the one-word sentence form, high precision; `structural` for three-of-everything
  (exactly three sibling cards whose bodies share a template, L52).
- **Maps to:** rules (existing aside R13). Network: when the triplet restates the heading or the hero line in other
  words, the triplet node is ECHO (R13, `LABELLING.md` §4 step 2); otherwise KEEP for the network.
- **Sources:** [wp-signs] WP:RO3; [stop-slop]; [humanizer] §6; [unslop-ui] §8; [deslop]; [tasteskill];
  [anti-slop-guides] R13.

#### L13 Rhetorical question, then the answer
- **What:** A question the reader did not ask, answered at once: "Tired of X? Meet Y.", "Ready to X?", "What if X?".
- **Why:** A (problem-agitate-solve copy), E ("engaging").
- **Examples:** "Tired of losing receipts? Meet Stash." / "What if budgeting felt easy?" / "Need a hand? We've got you
  covered."
- **Detection:** `pattern` (L13), high precision at node start.
- **Maps to:** rules. Network: a question heading over the button that answers it is KEEP under K11 (see L27); a
  question-form genre heading is L40.
- **Sources:** [stop-slop] "Rhetorical setups"; [deslop] "a question in every heading"; [tasteskill] "A question the
  reader did not ask".

#### L14 Before/after pivot and "X, reimagined"
- **What:** "Say goodbye to X", "No more X", "Gone are the days of X"; and the reinvention suffix "X, reimagined /
  redefined / made simple", "Meet X", "Introducing X".
- **Why:** A (launch-page idiom), B.
- **Examples:** "Say goodbye to messy group chats." / "Note-taking, reimagined." / "Meet Ledger, your new finance
  hub." / "No more missed renewals."
- **Detection:** `pattern` (L14), high precision.
- **Maps to:** rules. Network: "Meet [product]" in a hero whose wordmark is visible is ECHO → the wordmark (L33).
- **Sources:** [unslop-ui] §8 "Before/after pivots"; [tasteskill] "X, reimagined", "Meet ..."; [stop-slop] "stops
  being X and starts being Y".

#### L15 Mirrored benefit clause
- **What:** Every feature written as "X so you can Y", often with the same Y ("focus on what matters").
- **Why:** A, D (feature-benefit template applied per card).
- **Examples:** "Automatic backups, so you can focus on what matters." / "Shared lists so you can shop together." /
  "Smart reminders so you never miss a beat."
- **Detection:** `pattern` (L15) per node; `structural` when two or more sibling cards share the clause (a shared
  lead-in or tail, existing aside R10).
- **Maps to:** rules (R10 for the sibling form). Network: the "so you can focus on what matters" tail is filler, not a
  node of its own, so the network does not see it.
- **Sources:** [unslop-ui] §8 "Mirrored feature/benefit pairs".

#### L16 Trailing participle "analysis"
- **What:** A main clause followed by ", making it easy to...", ", ensuring...", ", empowering you to...": a second
  clause that asserts a benefit without evidence.
- **Why:** C (participial clauses at 2-5 times the human rate in instruction-tuned models).
- **Examples:** "Exports run nightly, ensuring your data is always safe." / "Templates load instantly, making it easy
  to start." / "Tags sync across devices, giving you peace of mind."
- **Detection:** `pattern` (L16). The first clause is often a real fact (KEEP); only the tail is slop.
- **Maps to:** rules (a span finding inside a KEEP node). Network: none; never let the tail push a factual node to
  REDUNDANT.
- **Sources:** [wp-signs] "Superficial analyses"; [reinhart].

#### L17 Grandiose openers and closers
- **What:** "In today's fast-paced world", "The future of X is here", "Join the revolution", "Experience the
  difference", "Everything you need to X".
- **Why:** A, B.
- **Examples:** "The future of hiring is here." / "Experience the difference today." / "Everything you need to run your
  studio."
- **Detection:** `pattern` (L17), high precision.
- **Maps to:** rules. Network: a closing-band headline that repeats the hero's claim is ECHO → the hero heading (R14);
  "Everything you need, in one place." after a feature grid is DECOR (a summary closer, L34).
- **Sources:** [unslop-ui] §8 "Grandiose closers", "Context-padding openers", "Hero copy clichés"; [tasteskill];
  [stop-slop] "In a world where".

#### L18 Copula substitutes
- **What:** "serves as", "acts as your", "stands as", "at the heart of" where "is" would do.
- **Why:** C, B (varied verbs read as fluent).
- **Examples:** "Inbox serves as your central hub." / "The dashboard acts as your command centre." / "At the heart
  of Plotline is a simple idea."
- **Detection:** `pattern` (L18). Moderate precision.
- **Maps to:** rules. Network: none.
- **Sources:** [wp-signs] "Avoidance of basic copulatives"; [aismells].

#### L19 Staccato emphasis fragments
- **What:** One- or two-word sentences used for drama: "That's it.", "No fluff.", "The best part?", "Period."
- **Why:** B (punchy reads as confident), A (2025-2026 landing pages).
- **Examples:** "One app. Every bill. That's it." / "The best part? It's free." / "No fluff. Just results."
- **Detection:** `pattern` (L19) for the closers; a count of sentences under three words per node for the general
  form.
- **Maps to:** rules. Network: a standalone "That's it." node is DECOR (no information).
- **Sources:** [stop-slop] "Dramatic fragmentation", "Emphasis crutches"; [humanizer] §2.

### L-C. Filler and hedges

#### L20 Throat-clearing openers
- **What:** Lead-ins before the point: "Here's how it works:", "Let's get started!", "When it comes to X", "Look no
  further".
- **Why:** F, C.
- **Examples:** "Let's get you set up!" (above a sign-in form) / "Here's how it works:" (above a numbered list) /
  "When it comes to budgeting, clarity is key."
- **Detection:** `lexical` (`llm_filler.txt` §L20), node start.
- **Maps to:** rules. Network: a lead-in that names what follows ("Here's how it works:") is NARRATE (R16, it narrates
  the visible list); a cheerful "Let's get started!" over a form is DECOR.
- **Sources:** [stop-slop] "Throat-clearing openers"; [unslop-ui] §8 "Context-padding openers".

#### L21 Emphasis crutches and intensifiers
- **What:** truly, really, incredibly, "and so much more", "like never before", "at your fingertips", "with just a few
  clicks", "it's worth noting".
- **Why:** B, E.
- **Examples:** "Truly effortless sharing." / "Your files, right at your fingertips." / "Calendars, notes and so much
  more."
- **Detection:** `lexical` (§L21). The existing R15 rule already covers a few fillers from the guide (easily, simply, please and now).
- **Maps to:** rules (extends R15). Network: none.
- **Sources:** [stop-slop] "Emphasis crutches", "Adverbs"; [wp-wtw] MOS:EDITORIALIZING; [unslop-ui] §8.

#### L22 Connective filler
- **What:** Sentence openers from essay prose: Additionally, Furthermore, Moreover, Notably, Ultimately.
- **Why:** C; [wp-signs] lists "Additionally" at sentence start as an AI vocabulary word.
- **Examples:** "Additionally, you can export to PDF." / "Ultimately, it's about your peace of mind."
- **Detection:** `lexical`, sentence start only. Rare in UI; one hit in helper or body text is a fair signal.
- **Maps to:** rules. Network: none.
- **Sources:** [wp-signs] WP:AIVOCAB; [aismells] "connective filler".

#### L23 Capability hedges
- **What:** "can help you", "designed to", "aims to", "a wide range of", "myriad": claims softened until unfalsifiable.
- **Why:** B (raters punish over-claiming), H (no facts).
- **Examples:** "Pocket can help you save more." / "A wide range of templates." / "Designed to make meetings better."
- **Detection:** `lexical` (`llm_hedges.txt` §L23), only in promotional slots. "May take up to 24 hours" in helper text
  is a real constraint and KEEP (`LABELLING.md` K4).
- **Maps to:** rules. Network: none.
- **Sources:** [stop-slop] "no softeners, no hedges"; [reinhart] (register); HypaNet's own UI framing.

#### L24 Vague attribution and borrowed authority
- **What:** "Experts agree", "Studies show", "Industry-leading", "Trusted by industry leaders" with no names.
- **Why:** H, B.
- **Examples:** "Backed by science." / "Experts recommend journaling daily." / "Industry-leading accuracy."
- **Detection:** `lexical` (§L24). A named source or link beside it clears the finding.
- **Maps to:** rules ("check this claim"). Network: none; out of scope for truth.
- **Sources:** [wp-signs] "Vague attributions"; [wp-wtw] MOS:WEASEL; [unslop-ui] §8 "Fabricated social proof".

#### L25 Chirpy and apologetic system messages
- **What:** "Oops! Something went wrong.", "Hang tight!", "Don't worry,", "We've got you covered", "Your privacy
  matters to us."
- **Why:** E ("friendly"), F.
- **Examples:** "Oops! Something went wrong." (as the whole error) / "Hang tight, we're loading your data!" / "Don't
  worry, your progress is saved."
- **Detection:** `lexical` (§L25). The useful finding is the combination of a softener and no cause or next action.
- **Maps to:** rules (extends R15 "oops"). Network: an error or empty body that only softens and gives no cause or
  action is not covered by K8; if it restates the title it is ECHO (R4); otherwise KEEP. "Don't worry, your progress
  is saved." is KEEP: the clause after the softener is a consequence.
- **Sources:** [unslop-ui] §5 toast; [anti-slop-guides] R15 (Apple, GOV.UK).

### L-D. Calls to action and claims

#### L26 Stock CTA labels
- **What:** Buttons that name no outcome: Get started, Learn more, Try it free, Start your journey, Explore features,
  Join the waitlist.
- **Why:** A (the hero always has a solid and a ghost button), D.
- **Examples:** "Get started" + "Learn more" in a hero / "Start your journey" on a fitness app / "Explore features" on
  a one-feature product.
- **Detection:** `lexical`, whole-label match on button and link (`llm_cta_cliches.txt` §L26). Existing rule
  slop/vacuous-action (R7) covers the core; this list widens it.
- **Maps to:** rules (R7). Network: KEEP (controls rank first, `LABELLING.md` §5).
- **Sources:** [unslop-ui] §5, §8 "CTA clones"; [hallmark] gate 19; [deslop]; [tasteskill] §4.

#### L27 CTA-band question and closer headlines
- **What:** The closing band's heading: "Ready to get started?", "What are you waiting for?", "Start building today",
  above a button that says the same.
- **Why:** A (the final CTA band mirrors the hero), D.
- **Examples:** "Ready to get started?" over [Get started] / "Ready to transform your workflow?" over [Start free
  trial] / "Join thousands of happy cooks." over [Sign up].
- **Detection:** `lexical` (§L27) plus `structural` (heading immediately followed by a single primary button).
- **Maps to:** rules, as an R7-family finding on the list hit. Network: under the current guide a question heading
  and the button that answers it are KEEP (K11, "a question title and its confirming button"), so "Ready to get
  started?" over [Get started] is KEEP for the network even though it is slop. A band heading that is neither the
  button's action nor a fact ("Ready to transform your workflow?" over [Start free trial]) is DECOR by the deletion
  test (`conf: judgement`). Open question for the guide: K11 was written for confirmation dialogs; the guide does not say
  whether it covers marketing CTA bands, so the answer key follows K11 as written.
- **Sources:** [unslop-ui] §4 "Final CTA band mirrors the hero"; [hallmark] gate 69.

#### L28 Speed and ease promises
- **What:** "Get started in seconds", "Set up in minutes", "No coding required", "in just a few clicks", "Works out
  of the box".
- **Why:** H, A.
- **Examples:** "Create your first invoice in seconds." / "No coding required." / "Live before lunch."
- **Detection:** `lexical` (`llm_claims.txt` §L28).
- **Maps to:** rules ("check this claim"). Network: KEEP once if it is a real constraint statement; a repeat on the
  same screen is ECHO (R14). Out of scope for truth.
- **Sources:** [unslop-ui] §8 "Hero copy clichés"; [deslop].

#### L29 Invented precision
- **What:** Round, unsourced numbers: "10x faster", "99.9% uptime", "10,000+ happy customers", "4.9/5", "Save 5
  hours a week".
- **Why:** H; models know numbers make copy credible and have none.
- **Examples:** "Trusted by 10,000+ teams." / "3x faster reporting." / "Rated 4.9/5 by our users."
- **Detection:** `pattern` (§L29 with `{n}`) plus a round-mantissa test; clears when a source, date or link is on the
  screen.
- **Maps to:** rules ("check this claim"); out of scope for truth. Network: a stat row whose numbers are all constant
  in markup is not DECOR by itself; it is a claim, so KEEP.
- **Sources:** [deslop] slop_scan.py "fake-stats"; [unslop-ui] §4 "Stat band", §8 "round vanity stats"; [tasteskill]
  §5.

#### L30 Vague social proof
- **What:** "Trusted by teams worldwide", "Loved by thousands", "Join thousands of happy users", "What our customers
  say" over unnamed quotes.
- **Why:** H, A.
- **Examples:** "Loved by creators everywhere." / "Trusted by industry leaders" over a logo strip / "Join thousands
  of happy planners."
- **Detection:** `lexical` (§L30).
- **Maps to:** network for the eyebrow form: "TRUSTED BY" over a logo wall is DECOR (R2; it names the section's genre,
  [hallmark] gate 54). As a sentence with no names it is DECOR by the deletion test. Rules: list hit.
- **Sources:** [unslop-ui] §5, §8; [hallmark] gate 54; [wp-buzz].

#### L31 Reassurance boilerplate
- **What:** "No credit card required", "Cancel anytime", "No spam, ever", "Bank-level security", repeated under every
  button.
- **Why:** A, D.
- **Examples:** "No credit card required." under four sign-up buttons / "No spam. Unsubscribe anytime." under a
  newsletter field / "Your data is safe with us." with no detail.
- **Detection:** `lexical` (§L31) plus a per-screen count.
- **Maps to:** network: one copy beside the primary action is KEEP (K6), later copies ECHO → the first (R14). "Your
  data is safe with us." with no specifics is DECOR. Rules: count.
- **Sources:** [unslop-ui] §4, §5; `LABELLING.md` K6.

#### L32 False urgency
- **What:** "Limited time", "Only 3 spots left", "Don't miss out" on a static page.
- **Why:** A (conversion templates), E.
- **Examples:** "Early-bird pricing ends soon!" with no date / "Only a few spots left." / "Don't miss out."
- **Detection:** `lexical` (§L32); stronger with no date or number bound to data.
- **Maps to:** rules; out of scope for truth.
- **Sources:** [unslop-ui] §8 "Fake urgency".

### L-E. Slot-filling redundancy (the network's territory)

#### L33 Product-name greeting
- **What:** "Welcome to [Product]", "Meet [Product]", "[Product]: the X for Y" in the hero, under a nav bar that
  already shows the wordmark.
- **Why:** A, D. The hero slot wants a headline and the product name is the safest one.
- **Examples:** "Welcome to Penny Jar" under the Penny Jar wordmark / "Meet Tally" / "Ledgerly: invoicing for
  freelancers".
- **Detection:** `pattern` (L17 "welcome to", L14 "meet") plus `structural` (the name matches the wordmark or page
  title).
- **Maps to:** network: the heading is ECHO → the wordmark when the rest of it is "Welcome to" / "Meet"; with a real
  description after the colon it is KEEP `partial: true` (`LABELLING.md` §4 step 5). "Welcome back, Sam" with a bound
  name is KEEP (identifier). Rules: R9 for chrome duplication.
- **Sources:** [tasteskill] "Welcome to ...", "A headline that needs its colon"; [anti-slop-guides] R9.

#### L34 Summary closers and "in a nutshell" lines
- **What:** A line that summarises what the reader just saw: "Everything you need, in one place.", "In short:
  less admin, more cooking.", "That's it. That's the whole setup."
- **Why:** F, C. Wikipedia lists section summaries as a long-standing chatbot habit.
- **Examples:** after three feature cards: "All your money, in one place." / after a three-step list: "Simple as
  that." / "In a nutshell: Tally does the maths."
- **Detection:** `lexical` (`llm_filler.txt` §L34) plus `structural` (last node of a section, no new noun).
- **Maps to:** network: ECHO with target `null` when it restates several nodes (`LABELLING.md` §3), DECOR when it
  carries nothing (R17). Rules: list hit.
- **Sources:** [wp-signs] "Section summaries"; [humanizer] §2; [stop-slop]; [anti-slop-guides] R17.

#### L35 Subtitle that restates the heading in other words
- **What:** A supporting line that paraphrases the heading instead of adding a fact, usually swapping in synonyms.
- **Why:** D, G. The subtitle slot must be filled; variation pressure makes the restatement a paraphrase, which is
  why string overlap misses it.
- **Examples:** "Simple pricing" / "Pricing that's clear and easy to understand." / "Your library" / "All your saved
  books in one place." / "Stay on track" / "Keep your goals moving forward."
- **Detection:** `context`. Needs synonym and paraphrase knowledge: the r1 model's ~170 synonym groups missed most of
  these. Lexical cues that the subtitle is filler: L01-L03 words, "in one place", "easy to", "here".
- **Maps to:** network ECHO → heading (R3). Partial KEEP when it adds a scope, number or constraint ("Pricing that's
  clear: £4 a month, no tiers.").
- **Sources:** [deslop] "A subtitle that repeats the heading"; [tasteskill] "never repeats the headline in other
  words"; [wp-signs] elegant variation; [anti-slop-guides] R3.

#### L36 Captions narrating controls
- **What:** Helper or caption text that says what a visible control does: "Click the button below to save your
  changes", "Use the toggle to turn reminders on or off", "Tap any card to see details".
- **Why:** D (every control gets a caption), F (the assistant explains).
- **Examples:** "Toggle on to receive email updates." under an "Email updates" toggle / "Click Save to store your
  edits." / "Use the search bar to find a recipe." under a search field labelled "Search recipes".
- **Detection:** `lexical` cues (`llm_narration.txt` §L36) plus `structural` (a control nearby whose label shares the
  object) plus `context` for the consequence test. The hidden-input cues in the same file ("hold Shift",
  "long-press", "drag to reorder") point the other way.
- **Maps to:** network NARRATE → the control (R6), `ct: true`. Caution from r1: sentences that add a consequence or an
  instruction the controls do not show (an invented case of the same shape: "Every photo in the album goes
  with it." under "Delete album?") are KEEP (K7, K4); the cue words must never outweigh a consequence clause.
- **Sources:** [unslop-ui] §8 "Over-explained microcopy/tooltips"; [anti-slop-guides] R6 (GOV.UK,
  Microsoft); `LABELLING.md` §7.

#### L37 Layout and self-description
- **What:** Copy about the screen rather than its subject: "Below you'll find your recent orders", "Here's everything
  at a glance", "This page shows your settings", "A clean, minimal layout for your projects".
- **Why:** F (assistant voice describes its output), D.
- **Examples:** "Below are your saved searches." above the list / "Your week at a glance." over a week view titled
  "This week" / "A clean, simple space for your notes."
- **Detection:** `lexical` cues (`llm_narration.txt` §L37) plus `structural` (the thing described is visible).
- **Maps to:** network NARRATE (R16), target the described node or `null`. A literal count of visible items ("Your 3
  saved searches") is NARRATE; a count bound to data is KEEP.
- **Sources:** [humanizer] §25; [tasteskill] "Explainer captions that narrate the design"; [stop-slop]
  "Meta-commentary"; [anti-slop-guides] R16.

#### L38 Eyebrow pills and section-genre kickers
- **What:** Small text over a heading that names the section's genre or announces the heading: FEATURES, HOW IT WORKS,
  WHY US, a "✨ New" pill, "TRUSTED BY".
- **Why:** A, D. The hero and every section in the template have an eyebrow slot.
- **Examples:** "FEATURES" over "Everything your kitchen needs" / "How it works" over "How Tally works" / "PRICING"
  over "Simple pricing".
- **Detection:** `lexical` (`llm_kickers.txt` §L38a/L38b, whole node) plus `structural` (role eyebrow or badge, placed
  before a heading).
- **Maps to:** network: ECHO → heading when the heading shares the word or stem, DECOR otherwise (`LABELLING.md` K1,
  R1/R2). §L38b is a candidate extension of K1's frozen list, not part of it.
- **Sources:** [unslop-ui] §1; [hallmark] gate 54; [krebs]; [anti-ai-slop]; [anti-slop-guides] R1, R2.

#### L39 Editorial-label kickers
- **What:** The second-order eyebrow that appears once "FEATURES" is banned: labels dressed as editorial ("Field
  notes", "Under the hood", "The details", "The fine print", "Our philosophy", "The short version").
- **Why:** A, plus the anti-slop instructions themselves: models told not to write genre eyebrows reach for the next
  nearest kicker.
- **Examples:** "Under the hood" over "Built on open standards" / "The fine print" over a pricing FAQ / "Field notes"
  over a blog list.
- **Detection:** `lexical` (§L39) plus `structural` as L38.
- **Maps to:** network DECOR (R2) unless it names a real category or parent section (K1: "Section 3: Your details").
- **Sources:** [tasteskill] "Labels pretending to be editorial"; [anti-ai-slop] (second-order slop).

#### L40 Question-form genre headings
- **What:** Section headings that are genre words posed as questions: "Why choose us?", "How does it work?", "What
  makes us different?", "Questions?".
- **Why:** A, E.
- **Examples:** "Why choose Pennywise?" over three feature cards / "How does it work?" over three numbered steps /
  "Got questions?" over an FAQ titled "FAQ".
- **Detection:** `lexical` (§L40) plus `structural`.
- **Maps to:** as a heading it is the section's title, so KEEP for the network unless another node says the same
  ("Got questions?" over "FAQ": the eyebrow-or-heading of lower rank is ECHO). Rules: list hit, low severity.
- **Sources:** [deslop] "a question in every heading"; [unslop-ui] §8 "Model-voice FAQ"; [tasteskill].

#### L41 Bold lead-in restated by its body
- **What:** The inline-header list as UI: a card or list item whose title is restated by its body ("Lightning fast"
  / "Experience blazing-fast performance.").
- **Why:** F (the chat answer's "**Label:** text" shape), D, G.
- **Examples:** "Secure by default" / "Your data is protected with robust security." / "Easy sharing" / "Share
  effortlessly with anyone." / "Offline mode" / "Work offline, anytime."
- **Detection:** `context` (paraphrase), helped by L01-L03 cues in the body. String overlap catches the verbatim half.
- **Maps to:** network ECHO → the card title (R5 / R4).
- **Sources:** [wp-signs] "Inline-header vertical lists"; [humanizer] §19; [unslop-ui] §8 "bold-lead-in lists";
  [anti-slop-guides] R5.

#### L42 Model-voice FAQ
- **What:** FAQ answers that open by echoing the question ("Is Tally right for me? Tally is right for anyone who...")
  and questions no customer asked.
- **Why:** F, C.
- **Examples:** "Can I cancel anytime?" / "Yes, you can cancel anytime." / "Is my data secure?" / "Your data's
  security is our top priority."
- **Detection:** `context` plus `structural` (question node followed by answer node); the first clause of the answer
  overlaps the question.
- **Maps to:** network: the restating first sentence is not a separate node, so the answer is KEEP `partial: true`
  when it goes on to a fact and ECHO → the question when it does not ("Your data's security is our top priority.").
- **Sources:** [unslop-ui] §8 "Model-voice FAQ".

#### L43 Synonym cycling across nodes
- **What:** One thing called by several names on one screen (projects, boards, workspaces), and its mirror: one
  fact restated in several slots with different words.
- **Why:** G.
- **Examples:** tab "Boards", heading "Your workspaces", button "New project" for the same object / eyebrow "ON HOLD"
  over "PAUSED".
- **Detection:** `context`. Needs a thesaurus with UI senses and the screen's own term choices. It is the reason r1
  misses paraphrase echoes.
- **Maps to:** network: the restating node is ECHO (`LABELLING.md` §1: redundancy is repeated information, not
  repeated words); the naming inconsistency itself is a rules finding (the inverse of K9), out of the network's
  classes.
- **Sources:** [wp-signs] "Lexical diversity/elegant variation"; [deslop] "One term, one meaning";
  [anti-slop-guides] §C.

### L-F. Residue, glyphs and typography

#### L44 Chatbot residue
- **What:** The assistant's own voice shipped in UI strings: "Certainly! Here's", "I hope this helps", "Let me know
  if you'd like", "Great question!", "As an AI".
- **Why:** F.
- **Examples:** an empty state reading "Sure! Here's a friendly message for when there are no items." / a tooltip
  ending "Let me know if you'd like any changes!"
- **Detection:** `lexical` (`llm_residue.txt` §L44). The highest-precision list in the codex. Allow chat products'
  transcript strings and fixtures.
- **Maps to:** rules (error severity). Network: none.
- **Sources:** [wp-signs] §Collaborative communication, §Hedging disclaimers.

#### L45 Placeholder identities and template text
- **What:** Unreplaced template content: Acme Inc, Jane Doe, jane@example.com, (555) 123-4567, Lorem ipsum, "[Your
  company]", "Your logo".
- **Why:** H, A.
- **Examples:** testimonial "— Sarah M., Marketing Manager" / footer "© 2026 Your Company. All rights reserved." /
  contact "123 Main Street, Anytown".
- **Detection:** `lexical` (§L45). Allow format hints in placeholders ("e.g. name@example.com").
- **Maps to:** rules. Network: none.
- **Sources:** [unslop-ui] §8 "Placeholder identities left in"; [deslop] slop_scan.py "placeholder"; [wp-signs]
  §Phrasal templates and placeholder text.

#### L46 Emoji bullets, sparkles and emoji icons
- **What:** Emoji leading headings, list items and buttons (🚀 Fast setup, ✅ Unlimited projects, ✨ Smart search),
  emoji as nav icons, the sparkle for anything AI.
- **Why:** F (chat answers use emoji bullets), E ("fun"), A.
- **Examples:** "🚀 Launch in minutes" / "✨ AI summaries" / "🔒 Bank-level security" / a feature list where every
  line starts with ✅.
- **Detection:** `glyph` (`llm_glyphs.txt` §L46) on raw text; count per screen and per sibling group.
- **Maps to:** rules (aside R11g). Network: a glyph on a KEEP node leaves it KEEP (`LABELLING.md` §6); an emoji-only
  node with no live state is DECOR (R11).
- **Sources:** [wp-signs] /Style "Emoji as formatting"; [fountain]; [unslop-ui] §7; [deslop].

#### L47 Static status pills and launch badges
- **What:** "● Live", "Now in beta", "New", "Available for work", "Now with AI" in fixed markup.
- **Why:** A, D.
- **Examples:** "● All systems operational" hard-coded in a footer / "NEW" on every feature card / "Now in beta" on a
  product that left beta.
- **Detection:** `lexical` (`llm_kickers.txt` §L47) plus `structural` (is the text or colour bound to data?).
- **Maps to:** network DECOR (R11) for static claims, KEEP when bound to data (K2) or the sole statement of a real fact.
  Already defined in `LABELLING.md` §6; listed here for the LLM motive.
- **Sources:** [unslop-ui] §5; [hallmark] gates 63, 66; [tasteskill]; [fountain] "Status dots that don't mean
  anything".

#### L48 The em dash habit
- **What:** Em dashes as the default clause connector, several per screen; after September 2026 often replaced by a
  spaced hyphen or by full stops everywhere.
- **Why:** B, C. Notorious enough that vendors now suppress it ([wp-signs]); Claude models still used more em dashes
  than professional writers in a July 2026 study the page cites.
- **Examples:** "Plan your week — without the stress." / "Fast — really fast." / "Export anywhere - PDF, CSV or
  Sheets."
- **Detection:** `glyph`, count per screen; one or two are writing, many are a habit ([tasteskill]). Weak alone and
  model-dependent.
- **Maps to:** rules (screen-level count). Network: none.
- **Sources:** [wp-signs] §Overuse of em dashes; [unslop-ui] §8; [tasteskill] §6; [anti-ai-slop].

#### L49 Title Case and all-caps labels
- **What:** Title Case On Every Heading And Button, and short tracked all-caps labels.
- **Why:** A (template defaults), F (chat headings).
- **Examples:** "Get Started For Free" / "Manage Your Account Settings" / "WHY TEAMS CHOOSE US".
- **Detection:** case test on raw text, skipping proper nouns and acronyms.
- **Maps to:** rules only. `LABELLING.md` §2: case and styling never decide a class.
- **Sources:** [wp-signs] /Style "Title case"; [unslop-ui] §8; [hallmark] gate 67.

#### L50 Middle-dot strips, arrow suffixes and slash strings
- **What:** "Fast · Private · Open source", "Learn more →", "Design / Build / Ship", "Est. 2025 · London".
- **Why:** A (template punctuation), D (meta strip slots).
- **Examples:** "Free · No signup · Works offline" under a hero / "See pricing →" / "Product / Design / Engineering"
  as a decorative strip.
- **Detection:** `glyph` (`llm_glyphs.txt` §L47/L50) plus `structural` (strip with no links).
- **Maps to:** rules; network DECOR when the strip's items are not facts about this product (`LABELLING.md` §3 DECOR,
  "meta strips"), KEEP when they are (a byline with a date).
- **Sources:** [hallmark] gate 63; [tasteskill] "Furniture text"; [anti-ai-slop].

#### L51 Exclamation marks and hype punctuation
- **What:** "You're all set!", "Welcome aboard!", "Let's go!", exclamation marks on success toasts and headings.
- **Why:** E, F.
- **Examples:** "Profile updated successfully!" / "Awesome! You're in!" / "Get ready to cook!"
- **Detection:** `glyph`, count per screen; more than one per screen in non-game UI is a fair signal.
- **Maps to:** rules. Network: "Successfully" and "!" do not change the class; "Profile updated successfully!" is
  KEEP as a toast unless the screen already shows the update (then ECHO).
- **Sources:** [deslop] "exclamation marks everywhere"; [unslop-ui] §5 toast.

#### L52 Metronomic sibling uniformity
- **What:** Every card the same length and shape: icon, two-word title, one 12-word sentence, the same clause
  structure, three of them.
- **Why:** A, D.
- **Examples:** three cards "Smart reminders / Never miss a bill again with smart reminders.", "Easy budgets / Create
  budgets effortlessly in seconds.", "Secure sync / Keep your data safe with secure sync."
- **Detection:** `structural`: sibling bodies with near-equal word counts, a shared template, and each body
  containing its own title's words.
- **Maps to:** rules (screen-level, with R10 and R13); each body that restates its title is network ECHO (L41).
- **Sources:** [unslop-ui] §4 "Feature soup", §8 "Metronomic rhythm"; [tasteskill] "Three parallel items
  everywhere".

## 6. Part U: UI copy patterns

### U-A. Headings, kickers and subtitles

#### U01 Kicker restating the heading
- **Guidance:** MS-UIT "Omit control labels that restate the main instruction" and "eliminate duplicate words and
  statements, both within and across controls"; REI Cedar "Don't use eyebrows as headings" (RES B9).
- **Examples:** eyebrow "ON HOLD" over h1 "Paused"; eyebrow "Your plan" over h2 "Plan"; eyebrow "Invoices" over h2
  "Invoices for May". Useful: eyebrow "Cycling" over h2 "Gravel tyre 40 mm" (a category the heading does not name).
- **Redundant or useful:** redundant when the eyebrow's information is in the heading, in the same words or a
  synonym; useful when it names a category, parent section, date or issue number the heading lacks (K1).
- **Detect:** eyebrow role or styling (small, tracked caps) directly above a heading; concept overlap with the
  heading after synonym folding. Paraphrase is the hard case (U47).
- **Maps to:** `ECHO` → heading, R1. Network.

#### U02 Section-genre kicker
- **Guidance:** LABELLING K1 genre list; GOV.UK "Headings should describe the purpose of the text that follows" (UK-UI).
- **Examples:** eyebrow "FEATURES" over h2 "Tools for small kitchens" (`DECOR`); eyebrow "Pricing" over h2 "Simple
  pricing" (`ECHO`); eyebrow "Kind words" over h2 "Trusted by 300 bakeries" (`DECOR`, a Testimonials variant).
- **Redundant or useful:** a genre word names the kind of section, which the section's content already shows. It is
  `ECHO` when the heading shares its stem or is itself a genre word of the same kind, `DECOR` otherwise.
- **Detect:** whole-text match against `lists/ui_genre_kickers.txt` (Block A frozen, Block B candidates).
- **Maps to:** `ECHO` → heading or `DECOR`, R1/R2. Network; the list feeds the rule layer and the concept codes.

#### U03 Decorative label strip
- **Guidance:** MS-UIT "Don't use all capital letters for emphasis ... users tend to regard it as 'screaming'"; Hallmark
  gates 54, 66, 67 and tasteskill "No tiny uppercase letter-spaced labels" (RES R2).
- **Examples:** "// EST. 2024 — LONDON — 51.5°N" under a hero; "THE BIG IDEA" above every slide title; "Section 02"
  above the only section on a page. Useful: "Updated 4 March" on a policy page (a dated fact).
- **Redundant or useful:** delete it; if no fact about this product or page is lost, it is decoration.
- **Detect:** mono or tracked-caps styling, separators (`//`, `—`, `/`), coordinates, local time, founding year;
  nodes with no verb and no product noun.
- **Maps to:** `DECOR`, R2. Network.

#### U04 Subtitle restating the heading
- **Guidance:** MS-UIT supplemental instructions: "Don't repeat the main instruction with slightly different wording.
  Instead, omit the supplemental instruction if there is nothing more to add"; SHOP "Avoid duplicating content, such
  as adding a page title under the page Title bar".
- **Examples:** h1 "Simple pricing" / "Pricing that is simple to understand."; h2 "Your library" / "All the books in
  your library." Useful: h1 "Simple pricing" / "£6 a month per editor. Viewers are free."
- **Redundant or useful:** subtract the heading's concepts and the restating frames (`lists/ui_restating_frames.txt`);
  if nothing is left, `ECHO`. If a scope, figure, condition or audience is left, `KEEP` (with `partial: true` when the
  rest restates; LABELLING step 5, the "Reading history for every branch" case).
- **Detect:** subtitle role under a heading; residue test (§7 step 2) rather than raw overlap.
- **Maps to:** `ECHO` → heading, R3. Network.

#### U05 Generic-verb subtitle
- **Guidance:** MS-UIT "Specific verbs (examples: connect, save, install) are more meaningful to users than generic
  ones (examples: configure, manage, set)".
- **Examples:** h1 "Notifications" / "Manage your notification settings."; h2 "Team" / "View and manage your team.";
  h2 "Integrations" / "Configure your integrations here." Useful: h2 "Team" / "Admins can invite people and change
  roles." (audience and permission).
- **Redundant or useful:** the frame "manage / view / configure / control your" plus the heading noun says only that
  the section is about the heading; a subtitle that names who can do what, or a limit, is new.
- **Detect:** first words in the settings block of `ui_restating_frames.txt`, remainder covered by the heading.
- **Maps to:** `ECHO` → heading, R3. Network; a strong, cheap cue (common in AI-built settings pages).

#### U06 Heading echoed in the section's first or last line
- **Guidance:** UK-UI "Headings should describe the purpose of the text that follows - they should not be part of the
  text"; humanizer and Hallmark gate 69 (RES R4).
- **Examples:** h2 "Offline mode" / body "Offline mode lets you work offline."; h2 "Fast exports" ending "Exports,
  made fast." Useful: h2 "Offline mode" / "Edits made without a connection sync when you reconnect."
- **Redundant or useful:** the first sentence restates when its residue (§7) is empty; a closing line restates when
  it adds no fact the section lacks (see U08).
- **Detect:** first or last body sentence of a section; residue against the section heading.
- **Maps to:** `ECHO` → heading, R4. Network.

#### U07 Title repeated across chrome
- **Guidance:** SHOP "adding a page title under the page Title bar"; MS-UIT "if the title bar caption and icon are
  displayed prominently near the top of the window, you can hide the title bar caption"; MS-UIT "For dialog boxes,
  don't include the words 'dialog' or 'progress' in the titles".
- **Examples:** page h1 "Billing" then card h2 "Billing" then card h3 "Billing details" (the h2 and h3 echo); dialog
  title "Rename dialog". Useful: tab "Billing" and h1 "Billing" (K11, both kept); breadcrumb ending "Billing" (K5).
- **Redundant or useful:** navigation, tabs and breadcrumbs repeat the page name by design (K5, K11); headings at
  different levels inside the page that repeat it are echoes, keeping the higher rank (LABELLING §5).
- **Detect:** exact or stem match between heading levels in one window; skip nav, tab and breadcrumb roles.
- **Maps to:** `ECHO` → higher heading, R9. Network.

#### U08 Restating closer
- **Guidance:** MS-UIT "Too much text discourages reading"; MAT "Only communicate the essential details"; humanizer
  and stop-slop (RES R17).
- **Examples:** three feature cards followed by "Everything you need, in one place."; a setup list ending "That's
  all it takes." Useful: a closing line with a fact the section lacks ("Setup takes about ten minutes.").
- **Redundant or useful:** a closer that summarises or cheers is `ECHO` (target `null` when it sums several nodes); a
  closer with a new figure or condition is `KEEP`.
- **Detect:** last node of a section; summary frames ("everything you need", "in one place", "that's it");
  coverage by the union of the section's nodes.
- **Maps to:** `ECHO` (often target `null`), R17. Network.

#### U09 Subtitle that restates and adds
- **Guidance:** MS-UIT supplemental instructions may provide "Qualifying information that helps users decide how to
  act on the main instruction".
- **Examples:** h3 "Invoices" / "Invoices for every client, newest first." ("every client" and the order are new);
  h2 "Storage" / "Storage used across all your devices."
- **Redundant or useful:** useful; the restating half does not make it redundant (LABELLING step 5).
- **Detect:** residue contains a scope or ordering marker (`ui_consequence_markers.txt`, scope block).
- **Maps to:** `KEEP`, `partial: true`. Network: this is a false-alarm guard, not a slop pattern.

### U-B. Dialogs, confirmations and messages

#### U10 Dialog body restating the title question
- **Guidance:** MS-CONF "Don't repeat the main instruction with slightly different wording"; APL-A "Include
  informative text only if it adds value"; ATL-MD "Body copy for a modal dialog should contain only valuable and
  relevant information".
- **Examples:** title "Discard this draft?" / body "Are you sure you want to discard this draft?"; title "Sign out?" /
  body "You're about to sign out." Useful: title "Sign out?" / body "Downloads in progress will stop."
- **Redundant or useful:** §7 decides it: strip the frame and the title's words; an empty residue is an echo.
- **Detect:** dialog_body under dialog_title; frames from `ui_restating_frames.txt` (confirmation block).
- **Maps to:** `ECHO` → title, R3 (R4 when it is the body's first sentence and a second sentence adds facts: label
  the sentences together as `KEEP`, `partial: true`, because nodes are labelled whole).

#### U11 Body explaining the buttons
- **Guidance:** APL-A "Avoid explaining alert buttons. If your alert text and button titles are clear, you don't need
  to explain what the buttons do"; MAT "A dialog's purpose should be communicated by its title and button text".
- **Examples:** "Select Delete to delete the file, or Cancel to keep it."; "Press Continue to proceed." Useful (rare):
  "Choose Keep both to save the new file with a number added." (says what a non-obvious option does).
- **Redundant or useful:** explaining a button whose label already states its result is narration; explaining an
  option whose result is not what its label leads people to expect is KEEP (LABELLING §7 consequence test).
- **Detect:** a button label quoted or named inside the body plus a narration verb (`ui_narration_verbs.txt`).
- **Maps to:** `NARRATE` → the button, R6, `ct: true`. Network.

#### U12 Vague warning
- **Guidance:** MS-CONF "Provide specific, useful information" and "Provide choices, not dilemmas"; NN-CD "Be specific
  and inform users about the consequence of their action".
- **Examples:** "This action may have unintended consequences."; "Please proceed with caution."; "This could affect
  your account." Useful: "This also cancels 2 upcoming bookings."
- **Redundant or useful:** a warning that names no object, scope, time or cost tells the reader nothing they can act
  on; under LABELLING it has no echo target and no narration, so the deletion test (step 6) decides: nothing lost.
- **Detect:** hedge modal (may, could, might) plus abstract nouns (consequences, issues, changes) and no marker from
  the scope, time, cost or audience blocks of `ui_consequence_markers.txt`.
- **Maps to:** `DECOR`, `conf: judgement` (**beyond the guide**: LABELLING has no R code for it; a labeller who reads it as narration
  records R6). Network.

#### U13 Consequence body (the false-alarm guard)
- **Guidance:** MS-CONF supplemental instruction: "Explain any non-obvious reasons why the user might not want to
  proceed", listing "data loss or financial loss", "System access or integrity", "Privacy or control over
  confidential information" and "Actions that are irreversible"; MS-CONF "use the term permanently to indicate that an
  action can't be undone"; MAT "consider what information is needed to explain an operation and any critical
  consequence"; NN-CD "A confirmation dialog must restate the user's request and explain what the computer is about to
  do"; APL-A "describe what happened, the context in which it happened, and why".
- **Examples:** "Discard this draft?" / "All 6 photos in it will be deleted."; "Delete workspace?" / "Members lose
  access to its boards straight away."; "Cancel subscription?" / "You keep Pro features until 14 March."
- **Redundant or useful:** useful whenever the residue carries one of the nine kinds of new information in §7, even
  though the sentence repeats the title's noun. This is the category r1 flagged wrongly.
- **Detect:** §7 and §9.1.
- **Maps to:** `KEEP` (K7; `partial: true` when a restating clause comes with it). Network.

#### U14 Instruction sentence
- **Guidance:** MS-UIT main instructions "explain concisely what users should do in a given window or page" and
  "communicate the user's objective rather than focusing just on manipulating the UI"; Atlassian error messages "Must
  include clear resolution steps" (ATL-CS); NN-ERR "offer some potential remedies".
- **Examples:** an empty preview pane saying "Choose a recipe to see its steps."; under "Can't connect" the body "Check
  that the printer is on and on the same Wi-Fi network."; a help sentence "Use the code in the email we just sent."
- **Redundant or useful:** an instruction is useful when its verb or object is not already the label of a visible
  control next to it (§7 step 4 and U15).
- **Detect:** sentence opens with a verb from `ui_instruction_markers.txt` or carries an obligation frame; the object
  is not the anchor label.
- **Maps to:** `KEEP` (K8 for empty and error bodies; main-instruction role otherwise; K4 for helpers). Network.

#### U15 Instruction that restates a visible control
- **Guidance:** MS-UIT "Omit instructional verbs that apply to all controls of the given type ... it goes without
  saying that users need to type into a text box control or that users need to click a link"; UK-UI "Fix the
  interface so it does not need explaining".
- **Examples:** "Save your changes." above a "Save changes" button; "Select a size." above a picker labelled "Size";
  "Click the button below to continue." above "Continue". Useful: "Save before 18:00 to include it in today's batch."
- **Redundant or useful:** the instruction's verb is the control's generic verb and its object is the control's label
  (or a synonym), with no condition, time or reason added.
- **Detect:** instruction lead + object covered by the adjacent control or label; gesture words point to NARRATE.
- **Maps to:** `ECHO` → the label (R5) when it restates a label; `NARRATE` → the control (R6, `ct: true`) when it
  tells the reader to operate a visible control or names its gesture. Network.

#### U16 Status-genre title
- **Guidance:** APL-A 'Avoid writing a title that doesn't convey useful information — like "Error" or "Error 329347
  occurred"'; NN-ERR "Generic messages such as An error occurred lack context".
- **Examples:** dialog title "Error" over "We couldn't save the file."; toast title "Success!" over "Your profile was
  updated."; banner "Heads up" over "Maintenance on Sunday from 02:00." Useful: a toast "Saved" with no body (it is the
  only statement of the new state).
- **Redundant or useful:** the title outranks the body, so it is not an echo of it; it fails the deletion test when
  the body says what happened (step 6).
- **Detect:** whole-text match against `lists/ui_status_genre_titles.txt` on a title role with a body sibling.
- **Maps to:** `DECOR`. Network (cheap and precise).

#### U17 "Warning:" and "Note:" prefixes
- **Guidance:** MS-CONF 'Don't use "warning" or "caution" in the text. If users need to proceed with caution, indicate
  this using a warning icon instead'; MS-UIT "There is no need to add, for example, the term WARNING in all capital
  letters"; UK-AZ "There's usually no need to say 'please' or 'please note'".
- **Examples:** "Warning: this deletes the file for everyone."; "Note: changes may take a few minutes."
- **Redundant or useful:** the prefix is filler; the sentence after it is labelled on its own (both examples are
  `KEEP`: audience, timing).
- **Detect:** leading token from `ui_status_genre_titles.txt` followed by a colon.
- **Maps to:** rule layer, aside R15. Not scored by the network.

#### U18 Success or status body restating its title
- **Guidance:** ATL-ES "Avoid repeating content from the title"; UK-UI "it's fine to say 'Application complete' rather
  than 'Thank you for your application'".
- **Examples:** title "Payment received" / "We've received your payment."; title "Invite sent" / "Your invite has been
  sent successfully." Useful: title "Invite sent" / "Ana has 7 days to accept it."
- **Redundant or useful:** §7 residue test; "successfully", "has been" and thanks carry nothing.
- **Detect:** title + body sharing the event noun and a past-participle verb; empty-status block of
  `ui_restating_frames.txt`.
- **Maps to:** `ECHO` → title, R4. Network.

#### U19 Error body restating its title
- **Guidance:** NN-ERR "Merely stating the problem is also not enough; offer some potential remedies"; APL-W "be clear
  about what someone can do to fix it"; UK-UI "There's no need to say 'sorry' in validation error messages".
- **Examples:** "Upload failed" / "Your upload could not be completed."; "Payment declined" / "Sorry, your payment
  was declined." Useful: "Upload failed" / "Files over 20 MB can't be uploaded. Compress it or split it."
- **Redundant or useful:** a cause or a remedy makes it `KEEP` (K8); a restated failure does not.
- **Detect:** residue test; instruction markers and cause markers ("because", "over", "expired") point to KEEP.
- **Maps to:** `ECHO` → title, R4; `KEEP` K8 otherwise. Network.

### U-C. Forms

#### U20 Placeholder repeating the label
- **Guidance:** UK-TI "Do not use placeholder text in place of a label, or for hints or examples"; NN-PH
  "placeholders should not be used as replacements for field labels"; APL-TF "Because placeholder text disappears
  when people start typing, it can also be useful to include a separate label".
- **Examples:** label "Email" / placeholder "Email"; label "Company name" / placeholder "Company". Useful (per Apple):
  label "Email" / placeholder "name@example.com" (format example).
- **Redundant or useful:** a placeholder that names the field is an echo; one that gives a format example is not
  (GOV.UK would still move it to hint text, but that is a design call, not redundancy).
- **Detect:** placeholder role; whole-text stem match with the attached label.
- **Maps to:** `ECHO` → label, R8. Network.

#### U21 Placeholder as instruction plus label
- **Guidance:** MS-UIT "Omit instructional verbs that apply to all controls of the given type"; NN-FORM "placeholder
  text causes many usability problems, and is best avoided".
- **Examples:** label "Postcode" / placeholder "Enter your postcode"; label "Country" / placeholder "Select a
  country..."; label "Search recipes" / placeholder "Search recipes...". Useful: label "Code" / placeholder "6 digits".
- **Redundant or useful:** strip a lead from `lists/ui_input_lead_verbs.txt`, then "your/a/the" and an ellipsis; if
  the label is what remains, echo.
- **Detect:** as stated; the r1 lexicon should treat the lead verbs as zero-information for this pair.
- **Maps to:** `ECHO` → label, R8. Network.

#### U22 Placeholder used instead of a label
- **Guidance:** UK-TI "All text inputs must have labels, and in most cases the label should be visible"; ATL-CS
  "Avoid placeholder text and never use it for critical information".
- **Examples:** a sign-up form whose only text per field is a placeholder ("Full name", "Work email").
- **Redundant or useful:** not redundant (it is the only copy); an accessibility and memory problem.
- **Detect:** input with placeholder and no label or accessible name.
- **Maps to:** rule layer, aside R8b (`slop/placeholder-as-label`).

#### U23 Helper restating the label
- **Guidance:** MS-UIT "Don't have supplemental explanations that merely restate the label for consistency"; UK-UI
  "only add help text if user research shows that you need it" and "there's usually no need to say 'This is the total
  cost'".
- **Examples:** label "Display name" / helper "Your display name."; label "Phone" / helper "Enter your phone number.";
  label "Notes" / helper "Any notes you want to add."
- **Redundant or useful:** echo when the residue is empty; see U25 for what makes a helper useful.
- **Detect:** helper role attached to a label; residue after stripping input leads and restating frames.
- **Maps to:** `ECHO` → label, R5. Network.

#### U24 Helper narrating the control's mechanics
- **Guidance:** UK-UI "If you find yourself having to explain how the user interface works, that's a sign something
  has gone wrong"; MS-STY "Don't use input-specific verbs, such as click or swipe".
- **Examples:** toggle "Mention alerts" / caption "Switch mention alerts on or off."; select "Theme" / helper "Pick a
  theme from the dropdown."; slider "Volume" / helper "Drag the slider to change the volume."
- **Redundant or useful:** describing what every control of that type does is narration; describing a hidden input
  is K4 (U40).
- **Detect:** control-mechanics block of `ui_narration_verbs.txt` + the label's noun.
- **Maps to:** `NARRATE` → the control, R6, `ct: true`. Network.

#### U25 Useful helper (format, limit, eligibility, timing, scope, data use)
- **Guidance:** UK-TI "Use hint text for help that's relevant to the majority of users, like how their information
  will be used, or where to find it"; NN-FORM "state the exact instructions" and "Don't make users guess your obscure
  password requirements"; APL-W "Choose a password with at least 8 characters".
- **Examples:** label "Email" / "We send the receipt here."; "At least 12 characters."; "You'll find this on the back
  of your card."; "Only visible to you."
- **Redundant or useful:** useful by definition (K4); it answers a question the label leaves open.
- **Detect:** limit and number markers, audience markers, "we send / we use / we store", "on the back of", "only".
- **Maps to:** `KEEP` (K4). Network false-alarm guard.

#### U26 Toggle description covering both states
- **Guidance:** APL-W "Describe what it does when turned on, and people can infer the opposite ... It isn't necessary
  to tell you that a timer won't start when this setting is off".
- **Examples:** "When on, new photos back up over Wi-Fi. When off, they don't." Useful version: "New photos back up
  over Wi-Fi."
- **Redundant or useful:** the on-state clause is information (K4); the off-state clause restates it.
- **Detect:** "when on ... when off" or "if enabled ... if disabled" pairs.
- **Maps to:** `KEEP`, `partial: true` for the network; the off clause is a rule-layer candidate (**beyond the guide**:
  aside R15-style "both states").

#### U27 Instructional verb in a label
- **Guidance:** MS-UIT example "Type your name" becomes "Your name"; MS-UIT "Keep labels brief".
- **Examples:** labels "Enter your email", "Select your country", "Type a message".
- **Redundant or useful:** labels rank first and are never flagged; the verb is filler inside the label.
- **Detect:** label starting with a lead from `ui_input_lead_verbs.txt`.
- **Maps to:** rule layer (aside R15 family). Not scored by the network.

#### U28 Tooltip repeating the visible label
- **Guidance:** NN-TT "Tooltips with obvious or redundant text are not beneficial to users" and its example "a button
  with the label Add new line also had a tooltip with the text Add new line"; NN-TT "Tooltips are best when they
  provide additional explanation for a form field unfamiliar to some users".
- **Examples:** button "Export" with tooltip "Export"; link "Settings" with title "Go to settings". Useful: icon-only
  button with tooltip "Export" (K12); tooltip "Exports exclude archived rows." (K4).
- **Redundant or useful:** echo when the control already shows the same words; useful when the control has no
  visible text, is truncated, or the tooltip adds a fact.
- **Detect:** tooltip role; compare with the control's visible text; check `icon_only` or `truncated`.
- **Maps to:** `ECHO` → control, R5; `KEEP` K12 or K4. Network.

### U-D. Actions and links

#### U29 Vacuous action label
- **Guidance:** APL-W 'just saying "Send" often works better than "Let's do it!"' and 'avoid using "Click here" in
  favor of more descriptive words'; MC-WE 'Don't say things like "Click here!" or "Click for more information" or
  "Read this."'; NN-LNK "avoid generic links such as Read more, or Click here"; APL-A "Avoid using OK as the default
  button title unless the alert is purely informational"; MS-UIT "Prefer specific labels over generic ones".
- **Examples:** "Learn more", "Get started", "Click here", "Submit" on a payment form. Useful: "See pricing", "Start a
  7-day trial", "Pay £12".
- **Redundant or useful:** not a redundancy pattern (controls are never flagged); a quality defect.
- **Detect:** whole-label match against `lists/ui_generic_ctas.txt` (the context-dependent block needs the dialog
  type).
- **Maps to:** rule layer, aside R7 (`slop/vacuous-action`). Not scored by the network.

#### U30 Gesture lead or pointer
- **Guidance:** UK-AZ "Don't use 'click' when talking about user interfaces because not all users click"; MS-STY
  "use generic verbs that work with any input method"; NN-LNK "there are no clicks on touch-screen devices".
- **Examples:** link "Click here to download the report"; body "To read the guide, tap here."
- **Redundant or useful:** the gesture is filler on a control; in a sentence that only points at a visible link it is
  narration.
- **Detect:** the rule layer's gesture lead and phrase lists; gesture block of `ui_narration_verbs.txt`.
- **Maps to:** aside R7 on controls; `NARRATE` R6 for a body sentence whose only content is the pointer. Rule layer
  and network.

#### U31 CTA repeated across sections
- **Guidance:** RES R14 (unslop-ui "Final CTA band mirrors the hero"); SHOP "presenting the same content in different
  ways on the same page"; MC-WE "If you can, avoid repeating links from the body text in related articles".
- **Examples:** "Start free trial" in the hero, after features and in the footer band of one page; "Book a demo"
  button and "Schedule a demo" link side by side.
- **Redundant or useful:** see §8; repeated per-item actions and separately mounted views are not echoes.
- **Detect:** equal-label (or synonym) controls in one window, across sections, same destination where known.
- **Maps to:** `ECHO` → first occurrence, R14. Network (needs the window's other sections in view).

#### U32 Control plus caption saying what it does
- **Guidance:** APL-A "Avoid explaining alert buttons"; MS-UIT "Use supplemental explanations when controls require
  more information than can be conveyed by their label. But don't use a supplemental explanation if one isn't
  necessary".
- **Examples:** button "Export CSV" / caption "Exports your data as a CSV file."; toggle "Dark mode" / "Turns on dark
  mode." Useful: button "Export CSV" / "Includes archived rows."; command link "Reset device" / "Removes all apps and
  data from this phone."
- **Redundant or useful:** LABELLING §7 consequence test: useful only if the effect is not what the label leads
  people to expect and the reader would act differently for knowing it.
- **Detect:** caption adjacent to a control; capability frames from `ui_restating_frames.txt`; residue empty.
- **Maps to:** `NARRATE` → control, R6, `ct: true`. Network.

#### U33 Same lead-in on every sibling
- **Guidance:** MS-UIT "If all the options have the same introductory text, move that text to the group label" and
  "focus the label text on the differences among the options".
- **Examples:** options "Email me when someone comments", "Email me when someone replies", "Email me when someone
  follows me".
- **Redundant or useful:** each item is still the only statement of its difference; the shared prefix is the waste.
- **Detect:** shared leading n-gram across sibling items (`slop/shared-lead-in`).
- **Maps to:** rule layer, aside R10.

### U-E. Empty and loading states

#### U34 Empty body restating the title
- **Guidance:** ATL-ES "Avoid repeating content from the title"; ATL-CS 'Avoid: "Nothing here", "No data found"'.
- **Examples:** "No invoices" / "There are no invoices yet."; "No results" / "We couldn't find any results."
- **Redundant or useful:** template match plus nothing else; the title already says it.
- **Detect:** `lists/ui_empty_state_templates.txt` with `{x}` bound to the title's noun or a synonym.
- **Maps to:** `ECHO` → empty title, R4. Network.

#### U35 Empty body with cause or next action
- **Guidance:** ATL-ES "Include the reason for the empty state and where they can go next"; NN-ES "Tell the user what
  could be displayed, and how to populate the area with that content"; APL-W "guide people on actions they can take".
- **Examples:** "No invoices" / "Invoices you send from a quote appear here."; "No results" / "Try a shorter search or
  remove a filter."; "No messages" / "Messages older than 90 days are deleted."
- **Redundant or useful:** useful (K8); "will appear here" alone is the judgement case (list header).
- **Detect:** instruction markers or cause markers in the residue.
- **Maps to:** `KEEP` (K8). Network false-alarm guard.

#### U36 Loading narration
- **Guidance:** MS-UIT progress dialogs: "use a gerund phrase briefly explaining the operation in progress, ending with
  an ellipsis"; UK-AZ "There's usually no need to say 'please'".
- **Examples:** spinner text "Loading your dashboard, please wait..." (KEEP as live state; "please wait" is filler);
  under it "Your dashboard is loading." (echo of the first line).
- **Redundant or useful:** one statement of progress is live state; a second one restating it is an echo.
- **Detect:** in a loading state branch (`in_state`), duplicate progress verbs.
- **Maps to:** first `KEEP`; repeats `ECHO`, R4; "please wait" aside R15.

### U-F. Narration and self-reference

#### U37 Copy about the screen itself
- **Guidance:** GDS "Don't use directional language to orient the reader, such as above, below, or right-hand side";
  UK-UI "Fix the interface so it does not need explaining"; tasteskill "Explainer captions that narrate the design"
  (RES R16).
- **Examples:** "Below you'll find your recent orders."; "This dashboard gives you an overview of your account.";
  "A clean, simple layout for your projects." Useful: "Orders from the last 90 days." (scope).
- **Redundant or useful:** describing the layout, the design or the screen's purpose is narration; stating the scope
  or source of what is shown is information.
- **Detect:** self-reference and directional blocks of `ui_narration_verbs.txt`.
- **Maps to:** `NARRATE`, R16 (target the section or `null`). Network.

#### U38 Literal count of visible items
- **Guidance:** LABELLING R16; MC-WE "Number lists when the order is important ... Don't use numbers when the list's
  order doesn't matter".
- **Examples:** "Three ways to save" above three visible cards; "Here are 4 quick steps:" above 4 steps. Useful: "340
  invoices" above a paged table (a bound count of items not all visible).
- **Redundant or useful:** a count the reader can see is narration; a bound or partial count is a fact.
- **Detect:** number word or digit + plural noun; compare with sibling count; check binding.
- **Maps to:** `NARRATE`, R16. Network.

#### U39 Ordinary platform behaviour
- **Guidance:** LABELLING §7 ("scrolling, hover highlighting, tapping a visible button, a link opening a page" are
  never non-obvious); NN-OVL "Users cannot be expected to read a manual before using your app".
- **Examples:** "Scroll to explore"; "Tap a card to open it"; "Hover over a chart for details" when the chart shows
  its values on screen.
- **Redundant or useful:** narration unless the gesture is hidden (U40).
- **Detect:** gesture block of `ui_narration_verbs.txt` + a visible target.
- **Maps to:** `NARRATE`, R6, `ct: true`. Network.

#### U40 Hidden-input hint
- **Guidance:** LABELLING §7(b) (keyboard shortcut, long-press, drag, swipe action, multi-select modifier); NN-OVL
  "there are times when it is helpful to the user to get a nudge in the right direction".
- **Examples:** "Hold Shift to select a range."; "Press ⌘K to search from anywhere."; "Swipe left on a row to archive
  it."
- **Redundant or useful:** useful (K4): no visible control exposes the input.
- **Detect:** key names, modifiers, "long-press", "swipe left/right on" + an object without a visible button for it.
- **Maps to:** `KEEP` (K4, `ct: true`). Network false-alarm guard.

#### U41 Tour narration of the visible UI
- **Guidance:** NN-OVL "Bombarding users with frequent hint screens causes them to dismiss hints more quickly" and
  advice to focus on one interaction "rather than attempting to explain every possible area of the user interface".
- **Examples:** spotlight "This is your dashboard. Here you can see your stats."; "This button creates a new project."
  over a "New project" button. Useful: "Projects are private until you share them." (privacy).
- **Redundant or useful:** pointing at what is visible is narration; a fact the screen does not show is not.
- **Detect:** onboarding roles (popover, spotlight); "this is", "here you can".
- **Maps to:** `NARRATE`, R16 or R6. Network.

### U-G. Markers, numbering and repetition

#### U42 Static status claim or decorative pill
- **Guidance:** GOV.UK Tag "Tags are only used to indicate a status" and "The more you add, the harder it is for users
  to remember them" (RES B4); Hallmark gate 63 (RES R11).
- **Examples:** "● Available for work" fixed in a portfolio header; "New" pill on every card; "AI-powered" badge on a
  heading that says "AI summaries".
- **Redundant or useful:** fixed, undated, undifferentiating claims are decoration; see U43.
- **Detect:** `lists/ui_static_status_claims.txt` + pill/badge role + no `stateful` flag.
- **Maps to:** `DECOR`, R11. Network.

#### U43 Bound status
- **Guidance:** LABELLING K2.
- **Examples:** "● Online" whose dot colour is bound to the connection; "Overdue" on one invoice row of ten; a legend
  "Healthy / Degraded / Down".
- **Redundant or useful:** useful (live state or a difference between siblings).
- **Detect:** `stateful` flag, sibling variation.
- **Maps to:** `KEEP` (K2). Network false-alarm guard.

#### U44 Repeated device on every sibling
- **Guidance:** GOV.UK Tag (above); RES R19.
- **Examples:** the same arrow suffix "→" on all six card links; "01 /", "02 /"... above every section; a sparkle
  emoji before every feature heading.
- **Detect:** `slop/repeated-chrome`.
- **Maps to:** screen aside R19, rule layer.

#### U45 Index numbering
- **Guidance:** MC-WE "Number lists when the order is important, like when you're describing steps of a process" and
  "Don't use numbers when the list's order doesn't matter".
- **Examples:** "01", "02", "03" over three unordered feature cards (aside R12); "Step 2 of 4" above a form (K3).
- **Maps to:** `KEEP` (K3) for the network; aside R12 for the rule layer.

#### U46 Same fact repeated across sections
- **Guidance:** SHOP "presenting the same content in different ways on the same page"; LABELLING K6 (one global
  notice) and K13 (per-item qualifiers).
- **Examples:** "No card needed" under four sign-up buttons (first `KEEP`, later `ECHO`); "/month" on each price card
  (each `KEEP`, K13).
- **Detect:** identical or synonymous sentences in different sections of one window; exempt qualifier lists
  (the rule layer's qualifier list).
- **Maps to:** `ECHO` → first, R14. Network.

#### U47 Synonym-cycled echo
- **Guidance:** SHOP "Avoid using multiple synonyms to describe a specific thing, action, or concept" and "identify
  and eliminate synonyms for key concepts and actions"; LABELLING §1 ("OPERATION ON HOLD" above "PAUSED").
- **Examples:** h2 "Billing" / subtitle "Payment details and invoices for your plan." (billing ≈ payments + invoices);
  empty title "No bookings" / "You have no reservations yet."; button "Remove" / tooltip "Delete".
- **Redundant or useful:** the same as U01–U34; the words differ but the information does not.
- **Detect:** synonym groups (§9.2); the main recall gap in r1.
- **Maps to:** `ECHO` with the pattern's own R code. Network.

#### U48 Consistent term reuse
- **Guidance:** MS-STY "Use one term consistently to represent one concept" (RES B2); SHOP "Use a single noun, verb,
  or phrase to describe a specific thing".
- **Examples:** h2 "Playlists" and button "New playlist"; dialog title "Delete playlist?" and button "Delete playlist".
- **Redundant or useful:** useful (K9, K11); sharing a word is not sharing information.
- **Detect:** guard: a heading and a control acting on it never echo each other.
- **Maps to:** `KEEP`. Network false-alarm guard.

### U-H. Word level

#### U49 Filler words
- **Guidance:** UK-UI "You do not usually need the word 'now'" and "There's usually no need to say 'please' or 'please
  note'"; APL-W "Possessive pronouns like my and your are often unnecessary"; MS-STY "Edit out you can when it isn't
  necessary".
- **Examples:** "Please enter your email now"; "Simply drag your files"; "Oops! Something went wrong."
- **Maps to:** rule layer, aside R15 (`slop/filler-words`; candidates in `lists/ui_filler_words.txt`).

#### U50 Padding phrases
- **Guidance:** MS-STY "Don't use two or three words when one will do" (RES B2); UK-AZ "in order to, usually not
  needed - do not use it"; MS-STY "Avoid weak phrasing like there is, there are, and there were".
- **Examples:** "Please be aware that at this time exports are in the process of being updated."
- **Maps to:** rule layer (**beyond the guide**: an R15 extension using `lists/ui_padding_phrases.txt`).

#### U51 Redundant acronym phrases
- **Guidance:** proselint `redundancy.ras_syndrome` (§10).
- **Examples:** "Enter your PIN number"; "Download as PDF format".
- **Maps to:** out of scope for the network (inside one node); rule-layer candidate (`lists/ui_tech_pleonasms.txt`).

#### U52 Promotional and unverifiable claims
- **Guidance:** GDS "avoid superlatives like best, simplest, fastest, never, and always"; MC-VT "we avoid distractions
  like fluffy metaphors and cheap plays to emotion"; NN-MC "Avoid broad and generic headings".
- **Maps to:** out of scope for redundancy (LABELLING §9 last line); see Part 1 (`llm_promo_vocab.txt`,
  `llm_claims.txt`).

## 7. Decision aid: sentences that add a consequence, instruction or reason

**Why this section exists.** In r1 nearly all real-set false alarms were full sentences in a lower-ranked slot that
repeated their title's noun and added something: a consequence under a confirmation title, a short instruction in an
empty pane, a help sentence. The network counted the repeated noun and missed the added clause. The guidance is
unanimous that this is exactly the copy a dialog or helper exists for:

- MS-CONF: the supplemental instruction should "Explain any non-obvious reasons why the user might not want to
  proceed" (data or financial loss, access, privacy, irreversibility); "Good confirmations never state the obvious;
  they should communicate something users need to be aware of"; and, the other side, "Don't repeat the main
  instruction with slightly different wording".
- NN-CD: "Be specific and inform users about the consequence of their action".
- MAT: "consider what information is needed to explain an operation and any critical consequence".
- APL-A: "Include informative text only if it adds value" and "Avoid explaining alert buttons".
- ATL-ES: "Avoid repeating content from the title" and "Include the reason for the empty state and where they can go
  next".

**Scope.** A sentence or clause in a `dialog_body`, `subtitle`, `helper`, `caption`, `empty_body`, `body`, `toast` or
`error` node that sits with a title, heading, label or control naming the same thing. The procedure below is LABELLING
§4 steps 2–6 written out for this case; it adds no new rule.

**Step 1. Find the anchor.** The highest-ranked node visible with the sentence (LABELLING §2a) that names the same thing: the
dialog title, the empty title, the section heading, the field label, or the control the sentence talks about.

**Step 2. Subtract what the anchor already gives.** Remove from the sentence:
1. restating frames (`lists/ui_restating_frames.txt`: "are you sure you want to", "you're about to", "this will",
   "manage your", "there are no");
2. the anchor's words, their stems and synonyms, including near-synonyms of the anchor's verb (discard, delete,
   remove, erase, clear; leave, exit, close; cancel, stop; send, submit);
3. the ordinary meaning of the anchor's action, that is, what LABELLING §7(a) calls the consequence "the control's
   label and the kind of product already lead the user to expect" ("Leave without saving?" already says unsaved work
   is lost);
4. filler (`lists/ui_filler_words.txt`).

**Step 3. Read what is left (the residue).** If it carries any of these nine kinds of information, the sentence is
`KEEP`. Markers for each are in `lists/ui_consequence_markers.txt` and `lists/ui_instruction_markers.txt`; a marker
counts only if the anchor does not already contain it.

| # | Kind | Marker examples | Invented example (anchor / sentence) |
|---|---|---|---|
| 1 | Scope beyond the anchor's object | all, every, also, including, both, a number ("6 photos") | "Discard this draft?" / "All 6 photos in it will be deleted." |
| 2 | Irreversibility | permanently, can't be undone, can't be recovered | "Delete account?" / "This can't be undone." |
| 3 | Recovery path | restore, undo, Bin, for 30 days | "Archive project?" / "You can restore it from Archive at any time." |
| 4 | Time | until, after, before, immediately, at the end of | "Cancel subscription?" / "You keep Pro features until 14 March." |
| 5 | Cost or quota | charged, refund, counts towards, free | "Add a seat?" / "You'll be charged £6 today, pro rata." |
| 6 | Audience or privacy | visible to, notified, members, shared with | "Remove Sam from the project?" / "Sam won't be notified." |
| 7 | Condition or exception | only, unless, if, except | caption under "Export" / "Exports exclude archived rows." |
| 8 | Reason or cause | because, so that, for security, why it appeared | "Sign in again" / "Your session ended after 30 minutes of inactivity." |
| 9 | Instruction or next action | an imperative or obligation with an object, place or input not given by a visible control's label | empty preview pane / "Choose a recipe to see its steps." |

Label: `KEEP` by K7 (confirmation consequence), K8 (empty or error body with cause or next action) or K4 (helper
answering a real question). Add `partial: true` when the node also contains a restating clause ("Are you sure? All 6
photos will be deleted."). Add `ct: true` when the sentence states what a visible control does (LABELLING §7 decided
it). Use `conf: judgement` only when the marker is weak (kind 1 with an implied scope, or kind 4 with a vague time).

**Step 4. Empty residue: the sentence restates.**
- It restates a title, heading or label → `ECHO` → that node (R3 for a subtitle or dialog body, R4 for a section's
  first line or an empty or status body, R5 for a helper, tooltip or card body).
- It explains what a visible control does, names its gesture, or names ordinary platform behaviour → `NARRATE` → that
  control (R6, `ct: true`).

**Step 5. Residue with no new information** (a vague hedge, an emotion, a promotional aside: "This may have
consequences.", "We'd hate to see you go.") → LABELLING step 6, the deletion test: normally `DECOR`, `conf: judgement`.

**Instructions: three cases that look alike.**

| Sentence | Where | Label | Why |
|---|---|---|---|
| "Choose a recipe to see its steps." | empty detail pane beside a recipe list | `KEEP` K8 | Says what to do to fill the pane; no control is labelled with it |
| "Choose a size." | helper under a picker labelled "Size" | `ECHO` → label, R5 | Generic verb for that control + its label (MS-UIT "Omit instructional verbs that apply to all controls") |
| "Tap Save to store your edits." | above a "Save" button | `NARRATE` → button, R6, `ct: true` | Operates a visible control by its own label |
| "Save before 18:00 to include it in today's batch." | above a "Save" button | `KEEP` K4, `ct: true` | Adds a time and a consequence |
| "Scroll to see every plan." | under a pricing heading | `NARRATE`, R6, `ct: true` | Ordinary platform behaviour |
| "Press ⌘K to search from anywhere." | footer hint | `KEEP` K4, `ct: true` | Hidden input |

**Worked pairs (all invented).**

| Anchor | Sentence | Residue | Label |
|---|---|---|---|
| title "Discard this draft?" | "Are you sure you want to discard this draft?" | none | `ECHO` → title, R3 |
| title "Discard this draft?" | "The draft will be discarded." | none | `ECHO` → title, R3 |
| title "Discard this draft?" | "All 6 photos in it will be deleted." | scope ("all 6 photos") | `KEEP` K7 |
| title "Discard this draft?" | "You can't undo this." | irreversibility | `KEEP` K7 |
| title "Leave without saving?" | "Your changes will be lost." | none (the title's ordinary meaning) | `ECHO` → title, R3, `conf: judgement` |
| title "Leave without saving?" | "Changes to 3 recipes will be lost." | scope (count) | `KEEP` K7 |
| title "Delete workspace?" | "Select Delete to continue or Cancel to go back." | none (explains the buttons) | `NARRATE` → Delete, R6, `ct: true` |
| title "Delete workspace?" | "Members lose access to its boards straight away." | audience, time | `KEEP` K7 |
| title "Turn off backups?" | "Photos taken after today won't be copied to the cloud." | time, scope | `KEEP` K7 |
| title "Payment failed" | "Your payment didn't go through." | none | `ECHO` → title, R4 |
| title "Payment failed" | "Your card has expired. Update it in Billing." | cause, instruction | `KEEP` K8 |
| h2 "Sharing" | "Manage your sharing settings." | none | `ECHO` → h2, R3 |
| h2 "Sharing" | "Only people you invite can see this album." | condition, audience | `KEEP` |
| button "Export CSV" | "Exports your data as a CSV file." | none | `NARRATE` → button, R6, `ct: true` |
| empty title "No results" | "Try fewer words or remove a filter." | instruction | `KEEP` K8 |
| empty title "No results" | "We couldn't find anything." | none | `ECHO` → title, R4 |

**Two traps.**
- *A marker inside the anchor does not count.* Title "Permanently delete 4 files?" / body "These files will be
  permanently deleted." has an empty residue: `ECHO`.
- *Sentence-ness is not a verdict.* Full sentences are KEEP more often than fragments, but "Manage your settings
  here." is a full sentence and an echo. Use the residue, not the full stop.

## 8. Button labels repeated across views

Controls rank first (LABELLING §5), so a control is flagged only as a later copy of an equal-ranked control.

| Situation | Label | Rests on |
|---|---|---|
| Same label in separately mounted states, dialogs or routes | never echoes; `aside: ["X-FILE"]` if the other copy is only in another file | LABELLING §2a |
| Per-item action on every card or row ("Edit", "Add to basket", "Remove") | `KEEP` each: each acts on a different item; deleting one loses that item's action | deletion test; K5 for tables and menus |
| Dialog title and its confirming button ("Delete playlist?" / "Delete playlist") | `KEEP` both | K11 |
| Page h1 and the nav item, tab or breadcrumb for the page | `KEEP` both | K5, K11 |
| Hero CTA repeated in a closing band or between sections of one page | first `KEEP`, later `ECHO` → first, R14 | LABELLING R14; RES R14 |
| Same action under two wordings side by side ("Book a demo" button and "Schedule a demo" link) | later or lower `ECHO`, R14 (synonym-cycled, U47) | SHOP synonyms |
| Two "Learn more" links to different places | not echoes (different destinations); aside R7 on both | NN-LNK "when users see the same link twice on the same page they assume that it goes to the same place" |
| Header CTA and hero CTA with the same words | open question: LABELLING does not settle whether a header CTA is navigation (K5); label `conf: judgement` or `UNSURE` | **beyond the guide** |
| Top and bottom "Save" on one long form | `conf: judgement`; `ECHO` → first, R14, when both are visible together; `KEEP` when they are in different scroll regions the labeller cannot see together | **beyond the guide** |

## 9. Detection notes

Items 1–7 come from the UI-guidance survey, items 8–15 from the LLM-writing research.

1. **Residue, not overlap.** The r1 network's layer-1 question ("is my concept in a nearby higher-rank node") is right for
   fragments and wrong for sentences: a sentence can contain the anchor's concept and still add a clause. The r2 net
   needs a second quantity, the information weight of the sentence's concepts NOT found in the anchor, and should
   count only the residue as evidence for KEEP. Practical form: give the marker classes in
   `ui_consequence_markers.txt` and `ui_instruction_markers.txt` their own concept codes with high information
   weight; give the restating frames (`ui_restating_frames.txt`) zero weight in both directions (they neither prove a
   restatement nor count as residue); and zero the anchor-verb synonyms when the anchor is a confirmation title.
2. **Thesaurus, with guard rails.** r1 missed most paraphrase echoes because its synonym groups were small. A full open
   thesaurus raises recall but merges senses ("clear" = transparent and = empty; "save" = keep and = store), so: (a)
   build UI groups by role (action verbs: delete/remove/erase/discard/trash/clear; state words: paused/on hold/
   stopped/suspended; place nouns: settings/preferences/options); (b) never put a marker word in the same group as
   its negation or opposite (can/cannot, will/won't, public/private, on/off), since that erases exactly the
   information §7 relies on; (c) prefer groups that two thesauri agree on.
3. **Slot pairs.** Every catalogue entry is a pair of roles: eyebrow↔heading (U01–U02), subtitle↔heading (U04–U05,
   U09), body↔title in dialogs, empty and status states (U10–U19, U34–U35), placeholder↔label (U20–U21),
   helper↔label (U23–U26), tooltip↔control (U28), caption↔control (U32), control↔control across sections (U31). The
   pair decides which list applies.
4. **Imperatives.** A sentence-initial verb from `ui_instruction_markers.txt` marks an instruction; compare its object
   with the adjacent label (U15). Gesture words (`ui_narration_verbs.txt`) next to a visible control mark NARRATE;
   next to a key name or modifier they mark a hidden input (U40).
5. **Whole-text lists.** U02, U16 and U42 are whole-node matches on short nodes (`ui_genre_kickers.txt`,
   `ui_status_genre_titles.txt`, `ui_static_status_claims.txt`). They are precise and cheap; the binding (`stateful`)
   decides U42 against U43.
6. **Counts.** U38 needs the number of visible siblings; a number word that equals it is narration, one that exceeds
   it is a fact.
7. **What not to learn from the prose linters.** Their weasel, hedge and tense rules flag "will", "all", "also",
   "only", "might" and "enough" (§10). In UI copy these are the consequence. Do not import them as slop cues.
8. **Most LLM markers are not redundancy.** Of 52 patterns, 13 have the network as their home (L30, L31, L33-L39,
   L41-L43, L47) and 14 more produce a redundant label in some instances (L05, L07, L12, L14, L17, L19, L20, L25, L27,
   L28, L40, L46, L50, L52); the other 25 belong to the rule layer alone or are claims a linter cannot verify. A useful anti-slop linter therefore needs both halves, and the network should not be pushed to
   flag a node because it contains hype words: that would make it a promo-word detector and break `LABELLING.md`.
9. **The excess-vocabulary studies do not measure UI copy.** Kobak, Juzek and Liang measured abstracts and reviews;
   Wikipedia measured encyclopaedia edits. Their words ("delve", "underscore", "intricate") are rare in buttons and
   headings. The UI-specific vocabulary (unlock, seamless, supercharge, effortless, journey) comes from practitioner
   catalogues, mainly [unslop-ui] §8 and [deslop], with no frequency data behind it. A real measurement on generated
   versus human UI copy would be worth more than any list here.
10. **Paech's slop lists are fiction lists.** slop-forensics and antislop-sampler (MIT and Apache-2.0) are dominated
   by character names, fiction imagery and essay topics; almost nothing transfers to UI. Their essay-domain n-grams
   only confirm the marketing words already listed. antislop-vllm has no licence and was not used; gptslop is AGPL and
   was not used.
11. **The vocabulary drifts.** Wikipedia's era lists and its em-dash note (suppressed in newer models from about
   September 2026) mean a lexical list ages within a year. Structural patterns (L09-L19, L33-L43) are more stable
   targets for a hand-written network.
12. **Second-order slop.** Banning the obvious tell produces the next one: editorial-label kickers replace "FEATURES"
   (L39), plain "quietly"/"steady" vocabulary replaces "delve" (L06), spaced hyphens and full stops replace em dashes
   (L48). Lists should be versioned per round.
13. **Paraphrase is the core gap for r2.** L35, L41 and L43 are the most common redundancy LLMs produce, and they are
   exactly the echoes r1 missed: the model varies its words on purpose (cause G). This argues for the full thesaurus
   with UI-sense groups, plus a "filler subtitle" cue built from L01-L03 and the "in one place / at a glance / easy to"
   phrases.
14. **The r1 false alarms fit a known trap.** r1 flagged sentences that add a consequence or instruction. L36's cue
   words ("click", "use the", "select") appear in genuine instructions too; the codex records the hidden-input cues
   and the consequence clause as the counterweight, so r2 should give a consequence clause (will be deleted, can't be
   undone, also removes, takes up to) priority over any narration cue.
15. **Highest-precision rules to add first:** L44 chatbot residue, L45 placeholders, L10 "Whether you're", L09
   negative parallelism in headings, L14 "X, reimagined" / "Say goodbye to", L13 rhetorical question openers, L29
   round unsourced numbers. Each is a regex over folded text with few legitimate uses in shipped UI.

## 10. What the prose linters cover

All of these lint one string at a time. None compares a node with its title, label or neighbour, so none covers the
slot-pair redundancy that is the network's job (§6 groups U-A to U-G); they cover the word level (§6 group U-H). Licences checked on the
published packages on 2026-11-11.

| Tool | Licence | Rules relevant to UI copy | Fit |
|---|---|---|---|
| Vale (https://vale.sh), Microsoft package (https://github.com/errata-ai/Microsoft) | MIT | `Wordiness` (phrase swaps, links MS-STY simple words), `Adverbs` (remove), `UIVerbs` ("Use 'select' instead of the input-specific verb"), `Vocab`, `Ellipses`, `HeadingPunctuation`, `We`, `FirstPerson` | U30, U49, U50 |
| Vale Google package (https://github.com/errata-ai/Google) | MIT | `Will` (flags every "will", after developers.google.com/style/tense), `ExcessiveClaims`, `Timeless` (currently, latest, soon), `Exclamation`, `WordList` | U52; `Will` conflicts with §7 kind 4 |
| Vale write-good and proselint packages (https://github.com/errata-ai/write-good, https://github.com/errata-ai/proselint) | MIT; BSD-3-Clause | write-good: passive, weasel, "so" at the start, "there is/are", too wordy, clichés; proselint: `Needless`, `RASSyndrome`, `Hedging`, `Very`, `Cliches`, `CorporateSpeak` | U49–U51 |
| write-good (https://github.com/btford/write-good) | MIT | checks `passive`, `illusion` (repeated word), `so`, `thereIs`, `weasel`, `adverb`, `tooWordy`, `cliches`, `eprime` | U49, U50; `illusion` catches "the the", not cross-node echo |
| proselint (https://github.com/amperser/proselint) | BSD-3-Clause | `redundancy.ras_syndrome`, `redundancy.misc` (Garner and After the Deadline pleonasm lists), hedging, weasel words, uncomparables | U51; the `misc` lists were not imported: one is taken from Garner's Modern American Usage (a copyrighted book) and the other from After the Deadline, so their licence lineage is unclear |
| alex (https://github.com/get-alex/alex), retext-equality | MIT | insensitive and inconsiderate wording | out of scope |
| retext-simplify (https://github.com/retextjs/retext-simplify) | MIT | wordy phrases with simpler swaps, some marked removable (`omit`) | U50 (8 phrases imported) |
| retext-intensify with words/fillers, words/hedges, words/weasels | MIT | fillers, hedges, weasels | U49 (26 fillers imported); hedges and weasels not imported (scope and consequence words) |
| retext-repeated-words, retext-redundant-acronyms, retext-passive | MIT | "the the", "ATM machine", passive voice | U51; passive voice is out of scope |

Imported entries and full licence notices: `lists/ATTRIBUTION-ui.md`.

## 11. The lists

All in `lists/`, one entry per line, `#` comments (each file's header gives its use, format, provenance and licence),
lower case, folded like the rule layer before matching; 3,257 entries in 24 files (counts exclude comments).
`llm_excess_vocab.txt` is CC BY-SA 4.0 as a whole (it includes Wikipedia lists); the other files are MIT, with the
notices of imported MIT and BSD entries kept in their headers and in `lists/ATTRIBUTION-ui.md`.

| File | Entries | Use | Patterns |
|---|---:|---|---|
| `llm_claims.txt` | 184 | speed and ease promises, invented precision, social proof, reassurance, urgency | L28, L29, L30, L31, L32 |
| `llm_constructions.txt` | 143 | sentence and headline templates with `{x}` `{w}` `{n}` slots | L09–L19 |
| `llm_cta_cliches.txt` | 101 | stock CTA labels and CTA-band headlines | L26, L27 |
| `llm_excess_vocab.txt` | 272 | excess vocabulary measured in research corpora (CC BY-SA 4.0 file) | L04, L05, L06, L18 |
| `llm_filler.txt` | 131 | throat-clearing openers, emphasis crutches, connectives, summary closers | L20, L21, L22, L34 |
| `llm_glyphs.txt` | 88 | emoji, glyphs and punctuation habits (raw text, before folding) | L46, L48, L50, L51 |
| `llm_hedges.txt` | 123 | capability hedges, vague attribution, chirpy and apologetic softeners | L23, L24, L25 |
| `llm_kickers.txt` | 188 | eyebrow genre words, editorial labels, question-form genre headings, status pills | L38, L39, L40, L47 |
| `llm_narration.txt` | 154 | narration cues, and hidden-input cues that point towards KEEP | L36, L37 |
| `llm_promo_vocab.txt` | 221 | hype verbs, inflating adjectives, abstract nouns, AI-as-headline words | L01, L02, L03, L08 |
| `llm_residue.txt` | 124 | chatbot residue and placeholder identities | L44, L45 |
| `ui_consequence_markers.txt` | 348 | KEEP evidence: scope, irreversibility, recovery, time, cost, audience, condition, reason, state change | U09, U13, U25, §7 |
| `ui_instruction_markers.txt` | 163 | instruction leads and obligation and remedy frames | U14, U15, U19, U35, §7 |
| `ui_restating_frames.txt` | 121 | frames that wrap a restatement (zero information) | U04, U05, U10, U18, U23, U32 |
| `ui_narration_verbs.txt` | 196 | gestures, control mechanics, directional and self-reference words | U11, U24, U30, U37–U41 |
| `ui_genre_kickers.txt` | 192 | section-genre eyebrows (Block A frozen K1, Block B candidates) | U01, U02 |
| `ui_status_genre_titles.txt` | 66 | message-genre titles and prefixes | U16, U17 |
| `ui_static_status_claims.txt` | 69 | pill and badge status words | U42, U43 |
| `ui_input_lead_verbs.txt` | 77 | placeholder and helper leads | U20, U21, U23, U27 |
| `ui_empty_state_templates.txt` | 49 | empty-state templates with `{x}` | U34, U35 |
| `ui_generic_ctas.txt` | 79 | vacuous and context-dependent action labels | U29, U30 |
| `ui_filler_words.txt` | 61 | filler words (Block B from words/fillers, MIT) | U49 |
| `ui_padding_phrases.txt` | 75 | padding phrases (Block B from retext-simplify, MIT) | U50 |
| `ui_tech_pleonasms.txt` | 32 | redundant acronym phrases | U51 |

## 12. Out of scope

Promotional vocabulary, invented statistics, em-dash density and default visual styling (LABELLING §9) are Part L's
territory. Passive voice, sentence length, capitalisation style, Oxford commas and inclusive-language rules are style
choices the linters above already cover and say nothing about redundancy. Vacuous labels (U29), placeholder-as-label
(U22), shared lead-ins (U33), repeated devices (U44), index numbering (U45), filler and padding (U49–U50) and
pleonasms (U51) belong to the rule layer and are never scored against the network.

## 13. Short source keys of Part U

Full entries, licences and access dates are in `SOURCES.md`; Part L cites `SOURCES.md` keys in square brackets
directly.

| Key | Source |
|---|---|
| MS-UIT | Microsoft Win32 UX guidelines, User interface text, https://learn.microsoft.com/en-us/windows/win32/uxguide/text-ui |
| MS-CONF | Microsoft Win32 UX guidelines, Confirmations, https://learn.microsoft.com/en-us/windows/win32/uxguide/mess-confirm |
| MS-STY | Microsoft Writing Style Guide: Top 10 tips, https://learn.microsoft.com/en-us/style-guide/top-10-tips-style-voice ; Describing interactions with the UI, https://learn.microsoft.com/en-us/style-guide/procedures-instructions/describing-interactions-with-ui ; Use simple words, https://learn.microsoft.com/en-us/style-guide/word-choice/use-simple-words-concise-sentences |
| APL-W, APL-A, APL-TF | Apple Human Interface Guidelines: Writing, https://developer.apple.com/design/human-interface-guidelines/writing ; Alerts, …/alerts ; Text fields, …/text-fields |
| MAT | Google, Material's Communication Principles: Intro to UX Writing (codelab), https://codelabs.developers.google.com/codelabs/material-communication-guidance |
| GDS | Google developer documentation style guide: UI elements, https://developers.google.com/style/ui-elements ; Excessive claims, https://developers.google.com/style/excessive-claims |
| UK-UI, UK-AZ, UK-TI | GOV.UK Service Manual, Writing for user interfaces, https://www.gov.uk/service-manual/design/writing-for-user-interfaces ; GOV.UK A to Z style guide, https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/style-guides/a-to-z-style-guide/ ; GOV.UK Design System, Text input, https://design-system.service.gov.uk/components/text-input/ |
| MC-WE, MC-VT | Mailchimp Content Style Guide: Web elements, https://styleguide.mailchimp.com/web-elements/ ; Voice and tone, https://styleguide.mailchimp.com/voice-and-tone/ |
| SHOP | Shopify app design guidelines, Content (Polaris content guidance), https://shopify.dev/docs/apps/design/content |
| ATL-ES, ATL-MD, ATL-CS | Atlassian Design System: Empty state, https://atlassian.design/foundations/content/designing-messages/empty-state ; Modal dialog usage, https://atlassian.design/components/modal-dialog/usage ; Content standards digest, https://atlassian.design/llms-content.txt |
| NN-PH, NN-TT, NN-CD, NN-ERR, NN-FORM, NN-OVL, NN-LNK, NN-ES, NN-MC | Nielsen Norman Group: placeholders https://www.nngroup.com/articles/form-design-placeholders/ ; tooltips https://www.nngroup.com/articles/tooltip-guidelines/ ; confirmation dialogs https://www.nngroup.com/articles/confirmation-dialog/ ; error messages https://www.nngroup.com/articles/error-message-guidelines/ ; web forms https://www.nngroup.com/articles/web-form-design/ ; instructional overlays https://www.nngroup.com/articles/mobile-instructional-overlay/ ; writing links https://www.nngroup.com/articles/writing-links/ ; empty states https://www.nngroup.com/articles/empty-state-interface-design/ ; microcontent https://www.nngroup.com/articles/microcontent-how-to-write-headlines-page-titles-and-subject-lines/ |
| RES | [anti-slop-guides]: an earlier HypaNet survey of anti-slop guides (REI Cedar, Hallmark, tasteskill, unslop-ui and others); not included with the codex |
