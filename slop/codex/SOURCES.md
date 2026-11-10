# Slop codex: sources

Every outside source the codex (`CODEX.md`, `lists/*.txt`) draws on, with its URL, its licence and what was
taken from it. Append-safe: each entry has a stable key in square brackets; add new entries at the end of the
relevant section, and never renumber. The codex cites sources by key.

"Imported" means entries were copied into a codex file (allowed only under a licence that permits it, with the
notice kept). "Cited" means the codex describes or names the source's finding, quotes at most a short phrase, and
writes its own examples. All pages were read on 2026-11-11 unless noted. Downloaded files were kept in a scratch
directory outside the repo and treated as data.

## LLM writing markers (CODEX.md Part L)

### [wp-signs] Wikipedia, "Wikipedia:Signs of AI writing" (WikiProject AI Cleanup advice page)
- URL: https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing (wikitext read via `action=raw`); subpage
  https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing/Style
- Licence: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). Attribution: Wikipedia contributors.
- Used: the catalogue structure (significance puffery, superficial participle analyses, vague attributions, AI
  vocabulary by model era, copula avoidance, negative parallelisms, rule of three, title case, boldface, inline-header
  lists, emoji as formatting, curly quotes, collaborative communication, placeholder text, section summaries, em
  dashes, elegant variation, promotional language as a historical indicator). Imported: the short "words to watch"
  word lists into `lists/llm_excess_vocab.txt` section C (that file is therefore CC BY-SA 4.0); stock phrases
  named in the collaborative-communication box are listed, with attribution, in `lists/llm_residue.txt`. Notable
  claims taken from the page and used as stated: newer models (from about September 2026) are often tuned not to use
  em dashes; a July 2026 study it cites found only Claude used em dashes more than professional writers; promotional
  tone became less common in chatbot output after about 2024 unless the user asks for it; the page warns that overuse
  of a word does not imply overuse of its synonyms.

### [wp-wtw] Wikipedia, "Manual of Style/Words to watch" (MOS:PUFFERY, MOS:WEASEL, MOS:EDITORIALIZING)
- URL: https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Words_to_watch
- Licence: CC BY-SA 4.0. Cited only: categories (puffery, weasel attributions, editorialising adverbs) informed
  `llm_promo_vocab.txt`, `llm_hedges.txt` and `llm_filler.txt`; entries there are stock English chosen by HypaNet,
  not a copy of the boxes.

### [wp-buzz] Wikipedia, "Wikipedia:Marketing buzzspeak" (essay, WP:BUZZ / WP:AIBUZZ)
- URL: https://en.wikipedia.org/wiki/Wikipedia:Marketing_buzzspeak
- Licence: CC BY-SA 4.0. Cited: "thought leader", "cutting edge", "award-winning" as buzzspeak; the essay's point
  that buzzspeak predates LLMs and should be judged on content, not on who wrote it.

### [kobak] Kobak, González-Márquez, Horvát, Lause (2025), "Delving into LLM-assisted writing in biomedical publications through excess vocabulary"
- Paper: Science Advances 11(27), eadt3813, https://www.science.org/doi/full/10.1126/sciadv.adt3813 ; preprint
  https://arxiv.org/abs/2406.07016 (earlier title "Delving into ChatGPT usage in academic writing through excess
  vocabulary").
- Data: https://github.com/berenslab/llm-excess-vocab , file `results/excess_words.csv` (900 excess words 2013-2024,
  annotated content/style).
- Licence: MIT, Copyright (c) 2024 Dmitry Kobak, Rita González-Márquez (repo LICENSE).
- Imported: 216 of the 407 rows annotated `style`, function and procedure words removed, into
  `lists/llm_excess_vocab.txt` section D with the MIT notice. Cited: the excess-frequency method and the finding that
  style words (delves, showcasing, underscores) carry the LLM signal while content words track topics.

### [juzek] Juzek and Ward (2025), "Why Does ChatGPT 'Delve' So Much? Exploring the Sources of Lexical Overrepresentation in Large Language Models"
- URL: https://aclanthology.org/2025.coling-main.426 ; https://arxiv.org/abs/2412.11385 (arXiv HTML shows CC BY-SA 4.0)
- Imported: the 21 focal words of Appendix A Table 2 into `lists/llm_excess_vocab.txt` section A (a list of facts,
  attributed). Cited: their tentative conclusion that preference tuning (RLHF), not training data or architecture,
  is the likeliest driver.

### [liang] Liang et al. (2024), "Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews"
- URL: https://arxiv.org/abs/2403.07183 (ICML 2024); arXiv perpetual non-exclusive licence.
- Imported: only the three adjectives named in the Figure 1 caption (commendable, meticulous, intricate) with their
  fold changes, as facts. The paper's full top-100 adjective and adverb lists (Appendices A and B) were not copied.
- Press coverage used for "innovative", "comprehensive": The Register, 2024-03-19,
  https://www.theregister.com/2024/03/19/ai_researchers_reviewing_peers/ (cited).

### [reinhart] Reinhart et al. (2025), "Do LLMs write like humans? Variation in grammatical and rhetorical styles" (PNAS)
- URL: https://arxiv.org/abs/2410.16107 ; CMU summary https://www.cmu.edu/dietrich/news/news-stories/2025/large-language-models-writing-text
- Cited: instruction-tuned models use present participial clauses at about 2-5 times and nominalisations at about
  1.5-2 times the human rate; base models are close to human rates, so instruction tuning drives the noun-heavy style.
  Supports codex L16 (trailing participles) and L03 (abstract nouns). Multipliers taken from the arXiv version.

### [paech-sf] Sam Paech, slop-forensics
- URL: https://github.com/sam-paech/slop-forensics (files `data/slop_list*.json`,
  `results_by_domain/essays/slop_lists/slop_list*.json`)
- Licence: MIT, Copyright (c) 2025 Sam Paech.
- Used: inspected only. The lists are over-represented words and stop-word-stripped n-grams from creative writing and
  essays; most entries are character names, fiction imagery or essay topics. The few marketing-relevant bigrams and
  trigrams ("unlock full potential", "ever evolving landscape", "cutting edge technology", "valuable insights",
  "navigate complexities", "unwavering commitment", "paradigm shift") confirmed the direction of the hand-written
  `llm_promo_vocab.txt` and the Wikipedia/Kobak entries of `llm_excess_vocab.txt`; nothing was imported verbatim.
  Finding recorded in CODEX.md §9.

### [paech-as] Sam Paech, antislop-sampler
- URL: https://github.com/sam-paech/antislop-sampler (files `slop_phrases_2025-04-07.json`,
  `slop_words_2025-04-07.json`, `slop_regexes.txt`)
- Licence: Apache-2.0.
- Used: inspected only. The phrase list is fiction-dominated ("voice barely above a whisper"); its regex file targets
  "not ... but" and "each/every ... a" constructions, which `llm_constructions.txt` L09 covers in its own syntax.
  Nothing imported.

### [paech-vllm] Sam Paech, antislop-vllm
- URL: https://github.com/sam-paech/antislop-vllm
- Licence: none stated in the repository (GitHub reports no licence). Not downloaded, not used.

### [eqbench] EQ-Bench (creative-writing slop score)
- URL: https://github.com/EQ-bench/EQ-Bench (MIT). Cited only as the origin of Paech's slop-scoring work; no list used.

### [stop-slop] Hardik Pandya, stop-slop (agent skill)
- URL: https://github.com/hardikpandya/stop-slop (`references/phrases.md`, `references/structures.md`)
- Licence: MIT, Copyright (c) 2025 Hardik Pandya.
- Used: categories (throat-clearing openers, emphasis crutches, business jargon, adverbs, meta-commentary, binary
  contrasts, negative listing, dramatic fragmentation, rhetorical setups, false agency). A handful of stock phrases
  it names ("at its core", "let that sink in", "make no mistake", "here's the thing") appear in
  `lists/llm_filler.txt`; the MIT notice is recorded here and in that file's header.

### [unslop-ui] Claudiu Sararu, unslop-ui-skill, TELLS.md
- URL: https://github.com/claudiusararu/unslop-ui-skill/blob/main/TELLS.md
- Licence: MIT, Copyright (c) 2026 Claudiu Sararu.
- Used: §8 "Copy, microcopy & voice" is the most UI-specific catalogue found (adjective reservoir, audience-spanning
  hedge, before/after pivots, "AI-powered" headlines, CTA clones, mirrored feature/benefit pairs, fabricated social
  proof, model-voice FAQ, fake urgency, empty personalisation, grandiose closers, placeholder identities); §1, §4, §5
  and §7 for eyebrows, CTA band mirroring the hero, hero pill badge, emoji icons and the sparkle. Patterns are
  described in HypaNet's words; stock phrases it names are listed in the lists files under MIT.

### [deslop] samigiman, anti-slop-skills: deslop-copy, deslop-ui slop-catalog and slop_scan.py
- URL: https://github.com/samigiman/anti-slop-skills (`skills/deslop-copy/SKILL.md`,
  `skills/deslop-ui/references/slop-catalog.md`, `skills/deslop-ui/scripts/slop_scan.py`)
- Licence: MIT, Copyright (c) 2026 anti-slop-skills contributors.
- Used: the English symptom list, structural symptoms (triplets, a question in every heading, a subtitle that repeats
  the heading, invented numbers, emoji bullets, exclamation marks), the "fact beats an adjective" rule, and the
  regex categories of `slop_scan.py` (`slop-copy`, `fake-stats`, `placeholder`). The script was read as text and not
  run.

### [humanizer] blader, humanizer (SKILL.md v4)
- URL: https://github.com/blader/humanizer ; Licence: MIT. Cited via [anti-slop-guides] A3 (closers,
  forced triads, bold labels, heading repeated in the first sentence, writing about the document).

### [hallmark] Nutlope, Hallmark, gates.md
- URL: https://github.com/Nutlope/hallmark/blob/main/skills/hallmark/references/gates.md ; Licence: MIT. Cited via
  [anti-slop-guides] A8 (gates 19, 54, 63, 66, 67, 69).

### [impeccable] Paul Bakaus, Impeccable
- URL: https://github.com/pbakaus/impeccable ; Licence: Apache-2.0. Cited via [anti-slop-guides] A9.

### [tasteskill] tasteskill v2 (design-taste-frontend), references/copy.md and tells.md
- Local install `~/.agents/skills/design-taste-frontend` (public project "taste-skill"); read 2026-11-11. Licence not
  recorded in the install, so cited only, in short phrases: "Things that read as generated" (genre verbs and
  adjectives, editorial labels, furniture text, em dash as a habit, explainer captions, step labels, three parallel
  items) and the headline shapes to avoid (mad-lib, tricolon, correction, "X, reimagined", "Welcome to ...", a
  question the reader did not ask).

### [anti-ai-slop] Vinayak-Shukla-03, anti-ai-slop
- URL: https://github.com/Vinayak-Shukla-03/anti-ai-slop ; Licence: MIT. Cited via [anti-slop-guides] A7
  (kicker on every slide, second-order slop such as mono micro-labels).

### [llmstrip] llmstrip
- URL: https://github.com/HugoLopes45/llmstrip ; Licence: MIT, Copyright (c) 2026 llmstrip contributors.
- Inspected `cli/src/rules/text.rs`: its word rules are tiered by Kobak's excess ratio and add "sycophantic opener"
  rules. Nothing imported; it confirmed the tiering approach used in CODEX.md (strong vs. weak lexical signals).

### [deslop-vsc] AUAggy, DeSlop (VS Code extension)
- URL: https://github.com/AUAggy/deslop ; Licence: MIT. Inspected only; nothing imported.

### [gptslop] AlpinDale, gptslop
- URL: https://github.com/AlpinDale/gptslop ; Licence: AGPL-3.0. Not downloaded and not used (copyleft terms
  unsuitable for a list file in an MIT repo).

### [aismells] "aismells", Delve and vocab tics
- URL: https://aismells.pages.dev/delve-and-vocab-tics.html ; author and licence not stated. Cited only (connective
  filler, copula substitutes, "load-bearing" as a recent tic); nothing copied.

### [sathya] Sathya, "Avoid 'landing page words'" (Microcopy Examples newsletter, 2024-12-23)
- URL: https://microcopyexamples.substack.com/p/avoid-landing-page-words ; all rights reserved. Cited only: unlock,
  unleash, exceed, empower, game-changer, supercharge as hollow landing-page words.

### [fountain] Jeff Humble, "7 Signs a UI Has Been Vibe Coded" (The Fountain Institute)
- URL: https://www.thefountaininstitute.com/blog/signs-vibe-coded-ui ; all rights reserved. Cited only: emojis used as
  icons, bullets and navigation items; status dots that don't mean anything.

### [krebs] Adrian Krebs, "Scoring Show HN pages for AI design patterns"
- URL: https://adriankrebs.ch/blog/design-slop/ ; tool https://github.com/AdrianKrebs/ai-design-checker (MIT). Cited via
  [anti-slop-guides] A11 (base rates; badge above the hero H1; all-caps section labels).

### [gptzero] GPTZero, "AI Vocabulary"
- URL: https://gptzero.me/ai-vocabulary and https://gptzero.me/news/most-common-ai-vocabulary/ ; proprietary
  (© GPTZero). Cited only: the method (phrases used by AI 10x to 200x+ more than humans over 3.3 million texts) and
  its caveat; the table did not render and nothing was copied. A BEA 2025 evaluation
  (https://preview.aclanthology.org/landing_page/2025.bea-1.71.pdf) found such term lists underperform a
  bag-of-words classifier, which supports treating lexical lists as weak evidence.

## UI copy guidance and prose linters (CODEX.md Part U)

Keys here are cited in `CODEX.md` Part U by the short source keys of its §13 table (MS-UIT, APL-A, …); the bracketed key
below names the entry each short key points to. Style guides are proprietary unless stated: cited only, short phrases
quoted, no list or example copied. All pages read on 2026-11-11.

### [ms-uit] Microsoft, Windows (Win32) UX guidelines, "User Interface Text" (CODEX.md key: MS-UIT)
- URL: https://learn.microsoft.com/en-us/windows/win32/uxguide/text-ui (written for Windows 7; Microsoft notes much
  still applies). © Microsoft. Cited: "Eliminate redundancy", "Remove redundant text" and the keep-full-text-in-
  main-instructions-and-controls rule; supplemental instructions "elaborate on the main instruction without simply
  re-wording it"; "Don't repeat the main instruction with slightly different wording"; "Omit control labels that
  restate the main instruction"; "Omit instructional verbs that apply to all controls of the given type"; "Don't have
  supplemental explanations that merely restate the label for consistency"; "If all the options have the same
  introductory text, move that text to the group label"; "Prefer specific labels over generic ones"; specific versus
  generic verbs; no "dialog" or "progress" in titles; the WARNING-in-capitals remark.

### [ms-conf] Microsoft, Windows (Win32) UX guidelines, "Confirmations" (CODEX.md key: MS-CONF)
- URL: https://learn.microsoft.com/en-us/windows/win32/uxguide/mess-confirm . © Microsoft. Cited: "Good confirmations
  never state the obvious"; supplemental instruction should "Explain any non-obvious reasons why the user might not
  want to proceed" (data or financial loss, system access, privacy, irreversibility); "use the term permanently to
  indicate that an action can't be undone"; "Don't use 'warning' or 'caution' in the text"; "Provide specific, useful
  information"; "Provide choices, not dilemmas"; the "are you sure" connotation table. Basis of the §3 decision aid.

### [ms-style] Microsoft Writing Style Guide (CODEX.md key: MS-STY)
- URLs: https://learn.microsoft.com/en-us/style-guide/top-10-tips-style-voice ("Prune every excess word", "Edit out
  you can when it isn't necessary", "Avoid weak phrasing like there is, there are, and there were");
  https://learn.microsoft.com/en-us/style-guide/procedures-instructions/describing-interactions-with-ui ("Don't use
  input-specific verbs, such as click or swipe", input-neutral verb table);
  https://learn.microsoft.com/en-us/style-guide/word-choice/use-simple-words-concise-sentences (via
  [anti-slop-guides] B2: "Make every word count", "Use one term consistently to represent one concept").
  © Microsoft. Cited only.

### [apple-hig] Apple Human Interface Guidelines: Writing, Alerts, Text fields (CODEX.md key: APL-W, APL-A, APL-TF)
- URLs: https://developer.apple.com/design/human-interface-guidelines/writing ;
  https://developer.apple.com/design/human-interface-guidelines/alerts ;
  https://developer.apple.com/design/human-interface-guidelines/text-fields (read from the pages' DocC JSON). © Apple.
  Cited: "Check each word to be sure it needs to be there"; 'just saying "Send" often works better than "Let's do
  it!"'; 'avoid using "Click here"'; possessive pronouns; "Describe what it does when turned on, and people can infer
  the opposite"; empty states ("guide people on actions they can take"); error messages and interjections; hint text
  examples; alerts: "Include informative text only if it adds value", "Avoid explaining alert buttons", "describe what
  happened, the context in which it happened, and why", the "Error" title example, OK only for informational alerts;
  text fields: placeholder disappears, so keep a label.

### [material-ux] Google, "Material's Communication Principles: Intro to UX Writing" (codelab) (CODEX.md key: MAT)
- URL: https://codelabs.developers.google.com/codelabs/material-communication-guidance (m3.material.io pages render
  client-side and could not be read; the codelab carries Material's writing guidance). © Google. Cited: "Only
  communicate the essential details for a specific context"; dialogs: "consider what information is needed to explain
  an operation and any critical consequence"; "A dialog's purpose should be communicated by its title and button
  text"; "To refer to a UI control or element, mention it using its label text".

### [google-devstyle] Google developer documentation style guide (CODEX.md key: GDS)
- URLs: https://developers.google.com/style/ui-elements ("Don't use directional language to orient the reader, such as
  above, below, or right-hand side"; "To refer to a button, use the button's label");
  https://developers.google.com/style/excessive-claims ("avoid superlatives like best, simplest, fastest, never, and
  always"); https://developers.google.com/style/tense (the basis of Vale's `Google.Will`). The guide's text is
  CC BY 4.0 per its site footer; only short phrases are quoted.

### [govuk] GOV.UK Service Manual, A to Z style guide and Design System (CODEX.md key: UK-UI, UK-AZ, UK-TI)
- URLs: https://www.gov.uk/service-manual/design/writing-for-user-interfaces ;
  https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/style-guides/a-to-z-style-guide/ ;
  https://design-system.service.gov.uk/components/text-input/ . Open Government Licence v3.0 (content) and MIT
  (Design System code). Cited: "If you find yourself having to explain how the user interface works, that's a sign
  something has gone wrong"; "Fix the interface so it does not need explaining"; help text "only ... if user research
  shows that you need it"; "now", "please", "sorry"; "Don't use 'click'"; "in order to"; placeholders not "in place of a
  label, or for hints or examples"; hint text "like how their information will be used, or where to find it".

### [mailchimp] Mailchimp Content Style Guide (CODEX.md key: MC-WE, MC-VT)
- URLs: https://styleguide.mailchimp.com/web-elements/ ; https://styleguide.mailchimp.com/voice-and-tone/ . © Intuit
  Mailchimp. Cited: "Button copy should always include verbs"; 'Don't say things like "Click here!"'; avoid repeating
  body links in related articles; number lists only when order matters; "we avoid distractions like fluffy metaphors
  and cheap plays to emotion".

### [shopify-content] Shopify app design guidelines, "Content" (Polaris content guidance) (CODEX.md key: SHOP)
- URL: https://shopify.dev/docs/apps/design/content (the old polaris-react.shopify.com content pages now redirect to shopify.dev).
  © Shopify. Cited: "Avoid duplicating content, such as adding a page title under the page Title bar"; "presenting
  the same content in different ways on the same page"; "Avoid using multiple synonyms"; CTAs "start with a strong
  verb".

### [atlassian] Atlassian Design System (CODEX.md key: ATL-ES, ATL-MD, ATL-CS)
- URLs: https://atlassian.design/foundations/content/designing-messages/empty-state ("Avoid repeating content from the
  title"; "Include the reason for the empty state and where they can go next");
  https://atlassian.design/components/modal-dialog/usage ("Body copy for a modal dialog should contain only valuable
  and relevant information"); https://atlassian.design/llms-content.txt (content standards digest: "Avoid placeholder
  text and never use it for critical information"; empty-state avoid list). © Atlassian. Cited only.

### [nngroup-ui] Nielsen Norman Group articles (CODEX.md key: NN-*)
- URLs: https://www.nngroup.com/articles/form-design-placeholders/ ; https://www.nngroup.com/articles/tooltip-guidelines/ ;
  https://www.nngroup.com/articles/confirmation-dialog/ ; https://www.nngroup.com/articles/error-message-guidelines/ ;
  https://www.nngroup.com/articles/web-form-design/ ; https://www.nngroup.com/articles/mobile-instructional-overlay/ ;
  https://www.nngroup.com/articles/writing-links/ ; https://www.nngroup.com/articles/empty-state-interface-design/ ;
  https://www.nngroup.com/articles/microcontent-how-to-write-headlines-page-titles-and-subject-lines/ . © Nielsen
  Norman Group. Cited: placeholders not as labels; the "Add new line" tooltip example; "Be specific and inform users
  about the consequence of their action"; "Do not use confirmation dialogs for routine actions"; "offer some potential
  remedies"; "Generic messages such as An error occurred lack context"; instructional overlays and dismissal; generic
  link text and repeated link labels; empty states ("what could be displayed, and how to populate the area").

### [vale] Vale prose linter and style packages
- URLs: https://vale.sh ; https://github.com/errata-ai/Microsoft (MIT, Copyright (c) 2018-2019 Joseph Kato);
  https://github.com/errata-ai/Google (MIT); https://github.com/errata-ai/write-good (MIT);
  https://github.com/errata-ai/proselint (BSD-3-Clause). Read from the repositories' tarballs. Cited: rule names and
  what they flag (`Microsoft.Wordiness`, `Microsoft.UIVerbs`, `Microsoft.Adverbs`, `Google.Will`,
  `Google.ExcessiveClaims`, `Google.Timeless`, `proselint.RASSyndrome`, …). Nothing imported.

### [write-good] write-good
- URL: https://github.com/btford/write-good (npm `write-good` 1.0.8). MIT. Cited: its checks (passive, illusion, so,
  thereIs, weasel, adverb, tooWordy, cliches, eprime). Nothing imported.

### [proselint] proselint
- URL: https://github.com/amperser/proselint . BSD-3-Clause, Copyright © 2014–2015 Jordan Suchow, Michael Pacer and
  Lara A. Ross. Cited: `redundancy.ras_syndrome`, `redundancy.misc`, hedging, weasel words. Imported: nine RAS phrases
  shared with `lists/ui_tech_pleonasms.txt` (notice in `lists/ATTRIBUTION-ui.md`). Not imported: the Garner and After
  the Deadline pleonasm files (licence lineage unclear).

### [alex] alex and retext-equality
- URLs: https://github.com/get-alex/alex ; https://github.com/retextjs/retext-equality . MIT. Cited only, as out of
  scope (insensitive language).

### [retext-simplify] retext-simplify
- URL: https://github.com/retextjs/retext-simplify (npm 8.0.0, `lib/patterns.js`). MIT, Copyright (c) 2016 Titus
  Wormer. Imported: 8 phrases marked `omit: true` into `lists/ui_padding_phrases.txt` Block B (notice in
  `lists/ATTRIBUTION-ui.md`).

### [words-fillers] words/fillers, with retext-intensify, words/hedges and words/weasels
- URLs: https://github.com/words/fillers (npm 2.0.1); https://github.com/retextjs/retext-intensify ;
  https://github.com/words/hedges ; https://github.com/words/weasels . All MIT, Copyright (c) 2014 Titus Wormer.
  Imported: 26 of the 83 `fillers` words into `lists/ui_filler_words.txt` Block B (notice in
  `lists/ATTRIBUTION-ui.md`). Not imported: hedges and weasels, which include UI scope and consequence words (all,
  also, only, might, enough).

### [retext-misc] retext-repeated-words, retext-redundant-acronyms, retext-passive
- URLs: https://github.com/retextjs/retext-repeated-words ; https://github.com/retextjs/retext-redundant-acronyms ;
  https://github.com/retextjs/retext-passive . MIT. Cited only.

### [anti-slop-guides] HypaNet research note "Anti-slop guides" (internal; not included with the codex)
- Internal. Cited for sources read in the earlier sweep and not re-read here: REI Cedar eyebrow guidance, GOV.UK Tag,
  Hallmark gates, tasteskill, unslop-ui, humanizer, PEP 8.
