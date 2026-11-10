# Anti-slop guidance for redundant UI copy: source sweep

Compiled 2026-11-10 as source material for the written definition of "redundant copy" that Fable 5.2 will work from in the slop experiment. Method: web search plus direct reads of each primary page (WebFetch, or raw GitHub files fetched with curl into a scratch folder). No paid model API was called. Every quote below is verbatim from the page named and under 25 words. Where a page could only be seen through a search snippet, that is said.

Two kinds of source are mixed here and should be weighed differently:

- **Agent skills and slop catalogues** (section A). Written in 2025 and 2026 by practitioners, mostly opinion, sometimes backed by small studies. They name the AI-specific patterns (eyebrows, dot pills, "01 /" numbering) that the older style guides never needed to.
- **Established UX-writing and content-design guidance** (section B). Microsoft, GOV.UK, Apple, Shopify, Mailchimp, NN/g, PEP 8. These predate AI slop and give the general principle (say a thing once, in the slot where it does work) plus the legitimate exceptions.

Pattern codes (R1 to R18) refer to the taxonomy at the end.

Sources that could not be read: Material Design (m3.material.io and m2.material.io render client-side; WebFetch returned only the page title), Atlassian Design (empty render), Strunk on bartleby.com (HTTP 403; quoted from the widely reproduced 1918 text via search results). The Fluent 2 badge page returned 404.

---

## A. Agent skills, prompt packs and slop catalogues

### A1. Wikipedia: Signs of AI writing
URL: https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing (WikiProject AI Cleanup; advice page, not policy)

- Rule of three: "LLMs often use this structure to make superficial analyses appear more comprehensive." (§3.5)
- "This sign is stronger when it appears in contexts where most people would not bother" (§3.5)
- Negative parallelisms: "Not just X, but also Y", "Not X, but Y", "Y rather than X" (§3.3 headings)
- Outline-like conclusions: "Such paragraphs usually appear at the end of articles with a rigid outline structure" (§2.7)
- Style: "Using title case instead of sentence case in section titles." / "Overuse of boldface in prose" / "Frequently presenting information in inline-header vertical lists." / "Affixing emoji at the beginning of section titles or list entries." (§4)
- Elegant variation (historical, §13.8): "Older AI models may include a repetition penalty, meant to discourage it from reusing words too often."
- "Section summaries" is listed as a historical indicator (§13.3); its text was beyond what could be read.
- Caveat from the page: the signs are observations, not proof, and the page warns against scrubbing symptoms.

Supports: R13 (triplets), R17 (restating closers and summaries), R5 (inline-header lists where the label restates the line), R11 (emoji as decoration). Also the warning that synonym cycling is itself a tell, which matters for the definition (see section C).

### A2. stop-slop (Hardik Pandya)
URL: https://github.com/hardikpandya/stop-slop (SKILL.md, references/phrases.md, references/structures.md)

- "Throat-clearing openers. Remove these announcement phrases. State the content directly." (phrases.md)
- "Any 'here's what/this/that' construction is throat-clearing before the point. Cut it and state the point."
- Emphasis crutches: "These add no meaning. Delete them." ("Let that sink in.", "Full stop.")
- Meta-commentary: "Remove self-referential asides. The essay should move, not announce its own structure."
- Rhetorical setups: "'Here's what I mean:' | Redundant preview"
- Rhythm: "Three-item lists | Use two items or one"
- Scoring question for density: "Anything cuttable?"

Supports: R13, R15 (filler), R16 (meta copy), R17 (closers). Prose-focused; nothing UI-specific.

### A3. humanizer (blader), v4.0.0
URL: https://github.com/blader/humanizer (SKILL.md)

- Staging: "The sentence signals importance instead of adding a fact, with a contrast that only adds weight or a one-line closer that repeats the point."
- §2 closers: watch for "a one-sentence paragraph that restates the paragraph before it"; "Cut a closer that repeats, including one that explains an example the reader just saw."
- §6 forced triads: "Ideas arrive in threes to sound complete, whether the meaning has three parts or not." "Keep three real items when the meaning needs three."
- §19 bold labels: "vertical lists give every item a bold label and a colon ... Turn a labeled list into prose when the labels carry no information"
- §20 decorative headings: "the document opens with a top-level heading that repeats its own title." "A heading written for effect ... should name what the section holds"
- §24 "A heading repeated in the first sentence": "A heading is followed by a one-line paragraph that restates it before the real content begins. Remove the repeated sentence."
- §25 writing about the document: "a legend, layout, or order the reader can already see ('the table below compares', ...)"
- §26 "Re-explaining what the reader knows".

Supports: R3, R4, R5, R13, R16, R17. The single best prose source for heading-to-body redundancy; it also gives keep-conditions (a short sentence is fine "when it carries a new fact").

### A4. unslop (local skill, installed at ~/.agents/skills/unslop/SKILL.md)
Public origin not recorded in the install. Rule numbers are stable ids.

- 11 "Synonym cycling. Protagonist, main character, central figure, hero all in one paragraph. Pick one, repeat it."
- 16 "Inline-header lists. The tell is a bold label and colon that restates the line"; a bold lead-in "followed by genuinely new detail ... is fine, not a tell."
- 27 "if the sentence could appear unchanged in another project's docs, it says nothing about this one. Cut it."
- 10 "Rule of three. Don't force ideas into groups of three. Use the natural number."

Supports: R5, R13, and the lexical-versus-semantic point in section C (repeat the term, not the idea).

### A5. unslop-ui-skill, TELLS.md (claudiusararu)
URL: https://github.com/claudiusararu/unslop-ui-skill/blob/main/TELLS.md (about 100 tells in 9 categories; says it was compiled from a "6-agent research pass" plus review)

- Root cause: "don't add decorative slots that carry no information (pills, eyebrows, blobs, count-ups, one-side stripes)"
- "Uppercase eyebrow over every section (`text-xs uppercase tracking-widest text-muted` 'WHY US', 'FEATURES'). Fix: none by default; let the heading lead."
- "Decorative rule/hairline + spaced-caps kicker above the heading - the subtle cousin of the pill"
- "Hero pill badge (`rounded-full border px-3 py-1` 'Now with AI' + sparkle + dot). Fix: a real changelog link or nothing."
- "Final CTA band mirrors the hero (same cadence, same two buttons, 'No credit card required')."
- "The rule-of-three as a tic: 'Fast, secure, and reliable', 'Simple. Powerful. Yours.'"
- "Model-voice FAQ ... answers that echo the question"
- "CTA clones: 'Get Started', 'Learn More', 'Sign Up Now', 'Try it Free'."
- "Mirrored feature/benefit pairs: 'X so you can Y' on every feature."
- "Over-explained microcopy/tooltips ('Click here to add a new item to your list of items')."
- "Title Case On Every Heading." / "bold-lead-in lists ('**Speed:** ...')"

Supports: R1, R2, R6, R7, R10, R11, R13, R14, R5.

### A6. anti-slop-skills (samigiman): deslop-copy and deslop-ui
URLs: https://github.com/samigiman/anti-slop-skills/blob/main/skills/deslop-copy/SKILL.md ; https://github.com/samigiman/anti-slop-skills/blob/main/skills/deslop-ui/references/slop-catalog.md

- deslop-copy rule: "Every sentence either tells the user something or helps them do something. If it does neither, it goes."
- Structural symptoms: "Triplets everywhere ('Fast. Reliable. Professional.')." / "A subtitle that repeats the heading." / "a question in every heading ('Ready?')"
- Microcopy: "Button = verb + object ... Not a lone 'Continue', 'OK', 'Submit'." / "Empty state = why it is empty + one action."
- "One term, one meaning: 'draft' is never also 'project' elsewhere."
- slop-catalog: "Duplicate logo in header and hero." / "A 'sample / pending approval' note under every section → one global note."
- "An icon next to every heading ... → icons only where they carry meaning." / "ALL-CAPS labels with wide tracking everywhere."
- Visual-pass question: "What would be lost if you deleted it? Nothing → delete it."
- Ships a scanner, `skills/deslop-ui/scripts/slop_scan.py` (regex-level, per its README).

Supports: R3, R7, R9, R13, R14, R2. The deletion test is the cleanest one-line operational definition found.

### A7. anti-ai-slop (Vinayak-Shukla-03)
URLs: https://github.com/Vinayak-Shukla-03/anti-ai-slop (SKILL.md, references/presentations.md, references/web-ui.md)

- "Kicker on every slide. A tiny uppercase, wide-tracked, accent-colored eyebrow label above the heading ... This is the most repeated AI slide motif."
- Fix: "use a section marker only at real section breaks — never one per slide."
- "Fake variety tricks ... decorative numbered corner badges ('02'…'09') to simulate design variation."
- "The corner page-counter + arrow-nav chrome — '01 / 10', a dot row"
- "A repeated wordmark + a 'sample / figures illustrative' disclaimer at the foot of every slide." / "The self-aware disclaimer ... on every screen."
- "Mono micro-labels on everything — column headers, eyebrows, timestamps all in a monospace face"
- "No forced rule-of-three. No em-dash pile-ups or decorative '·' middots."
- "The 'badge above the H1' + 'numbered 1·2·3 step row' + 'three feature cards' reflex"

Supports: R1, R2, R11, R12, R13, R14. Important nuance: it names the "second-order" slop that appears once first-order tells are banned (mono micro-labels replacing tracked-caps eyebrows).

### A8. Hallmark (Nutlope), gates.md
URL: https://github.com/Nutlope/hallmark/blob/main/skills/hallmark/references/gates.md

A browser-run linter with numbered gates; "M" gates are mechanical. Most directly reusable source for a lint spec.

- Gate 54, the eyebrow: "short inert type before a heading that announces it, in any geometry (stacked, beside, pill, ordinal, category tag over a card, `TRUSTED BY` over logos)"
- Gate 54 legal list: "`<figcaption>`, `<th>`, `<label>`, a nav or tab label, a byline or `<time>`, a unit after its figure, a real filename, a live `role="status"` chip"
- The eyebrow test: "Small type before a heading survives only when it carries a fact, a destination or an identifier the reader would otherwise lack."
- "Delete the line and read the page: if nothing was lost but a name for what comes next, it was an eyebrow."
- Gate 63, template punctuation: "meta strings joined with middle dots (`A · B · C`), an arrow glued to link or button text (`Learn more →`), decorative status dots before labels"
- "The middle dot as a universal separator, the arrow as a universal suffix and the coloured dot as a universal prefix are the punctuation of generated pages."
- Gate 63 legal: "one dot in a byline, a caption or a live state"
- Gate 66, decoration text: "a version badge or `BETA` pill in the hero ... a `01 / 4` count on tiles"; "A fact stays; a decoration goes."
- Gate 67: "The tracked-caps label: a label, caption, table head, nav link, tag, chip or any text at 15px or under set uppercase"
- Gate 69, the echo: "one fact (a price, a time, a date, a count) printed in four or more sections ... or a section's last line restating its heading"
- Gate 19 (copy): "a stock second action beside the primary (*Learn more*, *Get started*, *Book a demo*)"
- Gate 58, screen budget: "About 80 words per 900px window, labels and cells included"

Supports: R1, R2, R4, R7, R11, R12, R14. Gates 54, 63 and 69 are near-ready rule definitions with explicit exceptions.

### A9. Impeccable (pbakaus)
URLs: https://github.com/pbakaus/impeccable (README); https://github.com/pbakaus/impeccable/blob/main/.agents/skills/impeccable/reference/clarify.md ; detector source crates/foundation/src/rules/

- README: "A slop detector. 62 deterministic rules flag the tells above ... No LLM, no API key." (precedent for the lint-time constraint)
- The detector source has named checks `checkHeroEyebrow`, `checkKickerAboveHeading`, a numbered-label check and a repeated-text check, with skip lists for nav, tables, forms, breadcrumbs and `aria-hidden`.
- Code comment: Tailwind's "`tracking-widest` (0.1em) at 12px ... is the most common eyebrow setting"; kicker floor "0.06em".
- clarify.md audit list: "redundant headings, intros, helper text, and confirmations"
- "Say each idea once. If the heading already explains the state, the introduction should add new information or disappear."
- "Helper text answers an implicit question instead of restating the control."
- "Labels should describe what will happen, not the gesture used to trigger it."
- "Routine success should be brief." / "The final copy is as short as it can be without removing meaning or recovery."

Supports: R1, R3, R5, R6, R7, R12. The skip-list design (nav, tables, forms) is a useful precedent for where repetition is legitimate.

### A10. tasteskill v2, design-taste-frontend (local install at ~/.agents/skills/design-taste-frontend; public project "taste-skill")
Files: SKILL.md, references/copy.md, references/tells.md

- House rule: "No eyebrows. No small label above a heading, no pill badge above the headline, no 'New ·' chip."
- "No index numbering. No 01 / 02 / 03, no (001), no 'Step 2'." Instead: "A real `<ol>` if order matters."
- Supporting line: "It never repeats the headline in other words."
- "Step labels that duplicate numbers: 'Step 1', 'Phase 02'."
- "Explainer captions that narrate the design: 'A clean, minimal layout for…'."
- "status lines such as 'Available for work' with a dot, strings of words separated by middle dots or slashes."
- tells.md: "Decorative status dot ('Available for work', pulsing 'live') | lint: bounce | Remove"
- "A headline that needs its colon: 'Ledgerly: invoicing for freelancers'. Drop the name, it is in the navigation."
- Buttons: "The primary action keeps one name everywhere on the page."
- Numbers (the skill's own study of 247 gallery sites, October 2026; share of sites using the device, panel score 7.5 or higher versus under 6.5): icon feature cards 1% vs 28%, pill badges 14% vs 34%, tiny tracked caps 16% vs 57%. The same note says monospace labels (44% vs 24%) and index numbering (38% vs 24%) were more common on the higher-scored sites, and are banned as house taste, not on evidence.

Supports: R1, R2, R3, R9, R11, R12, R16. The honest split between measured and taste-based bans is worth copying into our spec.

### A11. Adrian Krebs, "Scoring 500 Show HN pages for AI design patterns"
URL: https://adriankrebs.ch/blog/design-slop/ (tool: github.com/AdrianKrebs/ai-design-checker)

- Patterns include "Badge right above the hero H1", "Numbered '1, 2, 3' step sequences", "All-caps headings and section labels", "Identical feature cards, each with an icon on top", "Stat banner rows".
- Numbers (1,590 Show HN submissions, headless-browser scoring): 22% (347) had 4 or more patterns, 32% (508) had 2 to 3, 46% (735) had 0 to 1. Author's manual QA put false positives at about 5 to 10%.
- Inconsistencies on the page: the title says 500 pages, the intro about 1,400, the results 1,590; the text says 16 patterns but lists 18.

Supports: R1, R11, R12, R13 (as base-rate evidence that these patterns are common, not that they are redundant).

### A12. Shorter catalogues
- TeneX Studio, "AI slop: 8 signs": https://tenex.studio/en/blog/ai-slop-8-signs/ . Eyebrow: "a tiny uppercase letter-spaced label sitting immediately above an oversized headline" / "Or the same shape rendered as a pill chip." Numbering: 01, 02, 03 "above three steps already in reading order." Cards: "Same size, same white fill, same icon chip, same bold label, same grey paragraph." Supports R1, R11, R12.
- solodesign.cc, "AI design slop: the tells": https://solodesign.cc/blog/ai-design-slop-the-tells/ . "Eyebrow chrome": "The uppercase label with a dot in front and a thin line trailing after it" and "decoration dressed up as design." Supports R1, R11.
- GitHub issues seen only as search snippets (low weight): gbyo/qbsheet #377 "Reduce eyebrow-label overuse"; enorm-labs/event-junkie #1239 "Spend the eyebrow label once per page, not above every title". They show the "once, where it earns its place" fix in practice.

---

## B. Established UX-writing and content-design guidance

### B1. Microsoft Windows UX guidelines: User interface text
URL: https://learn.microsoft.com/en-us/windows/win32/uxguide/text-ui (written for Windows 7; Microsoft notes much still applies)

The strongest single source on redundant UI copy.

- "Review each window and eliminate duplicate words and statements, both within and across controls."
- "Don't avoid important text; be explicit wherever necessary, but don't be redundant and don't explain the obvious."
- "Too much text discourages reading; the eye tends to skip right over it"
- "Look for redundant text in window titles, main instructions, supplemental instructions, content areas, command links, and commit buttons."
- "Generally, leave full text in main instructions and interactive controls, and remove any redundancy from the other places."
- Supplemental instructions: "Don't repeat the main instruction with slightly different wording. Instead, omit the supplemental instruction if there is nothing more to add."
- "Don't feel obliged to use main instruction text if adding it would only be redundant or obvious from the context of the UI."
- "Omit control labels that restate the main instruction."
- "Don't have supplemental explanations that merely restate the label for consistency."
- "If all the options have the same introductory text, move that text to the group label."
- "Omit instructional verbs that apply to all controls of the given type." Example: "Type your name" becomes "Your name".
- "For dialog boxes, don't include the words 'dialog' or 'progress' in the titles."
- "Don't use the title to explain the dialog box's purpose that's the purpose of the main instructions."
- Users "tend to read static text only when they think they need to" and "once users have decided what to do, they will immediately stop reading".
- "Don't use all capital letters for emphasis ... users tend to regard it as 'screaming.'"
- "Prefer specific labels over generic ones." (OK becomes Publish)

Supports: R3, R5, R6, R7, R9, R10, R15, R2 (caps). Gives the priority rule for which copy of a duplicate to keep.

### B2. Microsoft Writing Style Guide: Use simple words, concise sentences
URL: https://learn.microsoft.com/en-us/style-guide/word-choice/use-simple-words-concise-sentences

- "Make every word count." / "remove words that don't add substance."
- "Don't use two or three words when one will do." (in order to becomes to)
- "Omit unnecessary adverbs" (examples given: quite, very, quickly, easily, effectively)
- "Use one term consistently to represent one concept."
- Counterweight: "Use your judgment to avoid sounding abrupt or unfriendly."

Supports: R15; and the rule that repeating a term is correct (section C).

### B3. GOV.UK Service Manual: Writing for user interfaces
URL: https://www.gov.uk/service-manual/design/writing-for-user-interfaces

- "Space is at a premium with user interfaces. So put the important words first and drop any unnecessary words."
- "You do not usually need the word 'now'." ... "just say 'apply' rather than 'apply now'."
- "There's usually no need to say 'please' or 'please note'."
- "If you find yourself having to explain how the user interface works, that's a sign something has gone wrong."
- Help text: "only add help text if user research shows that you need it."
- "Headings should describe the purpose of the text that follows - they should not be part of the text."
- "In fact, it can help to remove duplication."

Supports: R6, R15, R4, R5.

### B4. GOV.UK Design System: text input, question pages, tag
URLs: https://design-system.service.gov.uk/components/text-input/ ; https://design-system.service.gov.uk/patterns/question-pages/ ; https://design-system.service.gov.uk/components/tag/

- Hint text: "Keep hint text to a single short sentence, without any full stops." / "Do not use hint text to explain anything that's longer than a short, simple sentence."
- "Do not use placeholder text in place of a label, or for hints or examples"
- Label as heading: "users of screen readers will only hear the contents once." (the system's own anti-duplication device)
- Question pages: "Do not use the same page heading across multiple pages."
- Legitimate kicker: for a higher-level section "you can use the `govuk-caption` style" above the heading (a caption that names the section, not the page's content).
- Tag: "Tags are only used to indicate a status." / "The more you add, the harder it is for users to remember them."
- Tag: a single tag can do the work, because "if something does not have a tag, that means it's incomplete".
- Tag: "Do not use colour alone to convey information"
- Links (search snippet only, guidance.publishing.service.gov.uk): "make it descriptive and front-load it with relevant terms instead of using something generic like 'click here' or 'more'."

Supports: R5, R8, R9, R11 (status pills only for real status, and only where absence is not enough), R7, and the legitimate-kicker exception to R1.

### B5. Apple Human Interface Guidelines: Writing
URL: https://developer.apple.com/design/human-interface-guidelines/writing (read via the page's JSON data; change log dated December 16, 2025)

- "Check each word to be sure it needs to be there. If you can use fewer words, do so."
- "Prioritize clarity and avoid the temptation to be too cute or clever with your labels." ("Send" over "Let's do it!")
- "For links, avoid using 'Click here' in favor of more descriptive words or phrases"
- "Possessive pronouns like my and your are often unnecessary to establish context." ("Favorites" over "Your Favorites")
- Settings: "Describe what it does when turned on, and people can infer the opposite."
- Errors: "Interjections like 'oops!' or 'uh-oh' are typically unnecessary and can sound insincere."
- Empty states: "guide people on actions they can take, and give them a button or link to do so if possible."
- Note: Apple endorses hint/placeholder text for format ("name@example.com"), with labels on all fields.

Supports: R7, R15, R6 (describe one state, not both), R8 (placeholder as format hint is fine; as label is not).

### B6. Shopify app design guidelines: Content (Polaris content guidance now lives here)
URL: https://shopify.dev/docs/apps/design/content

- "Avoid duplicating content, such as adding a page title under the page Title bar"
- Also flags "presenting the same content in different ways on the same page."
- "Use a single noun, verb, or phrase to describe a specific thing, action, or concept."
- "Avoid using multiple synonyms to describe a specific thing, action, or concept."
- "For calls to action (CTAs), start with a strong verb that describes the action."

Supports: R9, R14, R7; and the anti-synonym rule in section C.

### B7. Mailchimp Content Style Guide
URLs: https://styleguide.mailchimp.com/web-elements/ ; https://styleguide.mailchimp.com/writing-for-accessibility/

- Links: "Try to avoid 'click here' or 'learn more.'" / "Links should provide information on the associated action or destination."
- Buttons: "Button copy should always include verbs."
- "If you can, avoid repeating links from the body text in related articles."
- "Don't include preceding articles (a, an, the, our) when you link text."

Supports: R7, R14, R15.

### B8. Nielsen Norman Group
- Placeholders: https://www.nngroup.com/articles/form-design-placeholders/ . "placeholders should not be used as replacements for field labels." Disappearing hints "strain users' short-term memory". Supports R8.
- Tooltips: https://www.nngroup.com/articles/tooltip-guidelines/ . "Tooltips with obvious or redundant text are not beneficial to users." Example: "a button with the label Add new line also had a tooltip with the text Add new line." "If you can't think of particularly helpful content, don't offer a tooltip." Supports R5 (the cleanest published example of label-tooltip echo).
- Links: https://www.nngroup.com/articles/writing-links/ . "We advised against Click here links as early as 1996." "people mostly look at the first 2 words of a link." "when users see the same link twice on the same page they assume that it goes to the same place." Supports R7.
- Empty states: https://www.nngroup.com/articles/empty-state-interface-design/ . "Tell the user what could be displayed, and how to populate the area with that content." Supports the clean side of R6 examples.
- Concise, scannable, objective (Morkes and Nielsen, 1997): https://www.nngroup.com/articles/concise-scannable-and-objective-how-to-write-for-the-web/ . Against a promotional control version of the same site: concise text scored 58% higher on measured usability, scannable 47%, objective 27%, all three combined 124%. "users detest anything that seems like marketing fluff or overly hyped language". Supports the cost of padding in general.
- How little users read (Weinreich, Obendorf, Herder, Mayer; data collected 2005, published 2008; 25 participants, 59,573 page views logged, about 45,237 after filtering): https://www.nngroup.com/articles/how-little-do-users-read/ . "users have time to read at most 28% of the words during an average visit; 20% is more likely." Supports the case that redundant copy costs attention.
- No NN/g article on eyebrows or kickers was found.

### B9. REI Cedar design system: Eyebrow
URL: https://cedar.rei.com/guidelines/typography/eyebrow

- "Do use eyebrows for categories or groupings." / "Do use eyebrows to categorize data."
- "Don't use eyebrows for brand names." / "Don't use eyebrows as headings."

The legitimate-use counterpoint for R1: an eyebrow that names a category the heading does not (e.g. "Camping" over a product name) is information, not redundancy.

### B10. PEP 8 (code comments)
URL: https://peps.python.org/pep-0008/

- "Inline comments are unnecessary and in fact distracting if they state the obvious." Example given: `x = x + 1  # Increment x`, versus the useful `# Compensate for border`.
- "Comments that contradict the code are worse than no comments."

Supports R18 (the code-comment analogue of label restatement).

### B11. Strunk, The Elements of Style (1918), "Omit needless words"
Read via secondary reproductions; bartleby.com returned 403.

- "A sentence should contain no unnecessary words, a paragraph no unnecessary sentences"
- Classic example pair: "the question as to whether" becomes "whether".

Supports R15 and the general principle. Note that commentators dispute how strictly to apply it, which is why the taxonomy rates judgement.

---

## C. What the sources imply for the definition

1. **Redundancy is semantic and slot-based, not lexical.** Microsoft, Shopify and the deslop-copy skill all require repeating the same term for the same concept, and Wikipedia and unslop treat synonym cycling as an AI tell. So "Paused" heading plus a "Resume" button is fine; "OPERATION ON HOLD" above "PAUSED" is not, even though the two share no word. The detector should score repeated *information* across slots, and should not reward paraphrase.

2. **Most rules are about a pair of slots.** The recurring pairs are eyebrow and heading, heading and subtitle, heading and first sentence, section closer and heading, label and helper text, label and tooltip, label and placeholder, window or title bar and page heading, card title and card body, sibling items and their shared prefix, hero and closing CTA band. Microsoft gives the keep rule when two slots duplicate: keep the full text in the main instruction and the interactive control, cut it elsewhere.

3. **Some patterns are decorative slots rather than duplicates.** Eyebrows that only name what comes next, dot pills on static content, "01 /" numbering over items already in order, version pills. Hallmark's test generalises: delete it; if no fact, destination or identifier was lost, it was decoration. Several of these are partly visual (tracked caps, a coloured dot). A text-only model will need the element role and a few style features (uppercase, letter-spacing, font size relative to the heading, leading glyph such as "●") as input, or it will miss them.

4. **Exceptions the sources agree on.** An eyebrow that carries a category, a section name in a multi-page flow (GOV.UK caption), a byline, a date or a unit; a dot that shows live state; a real ordered list; helper text that answers an implicit question (format, eligibility, consequence); repeated values in tables, navigation, menus and forms (Impeccable's skip list); one honest global disclaimer.

5. **The motivating pause menu, read against the sources.**
   - "OPERATION ON HOLD" above "PAUSED": a themed paraphrase of the heading in a tracked-caps eyebrow (R1). Hallmark gate 54 and the tasteskill house rule both cover it; the deletion test loses nothing.
   - "Resume recaptures the mouse.": explains what a self-evident control does (R6). GOV.UK: "If you find yourself having to explain how the user interface works, that's a sign something has gone wrong." It is the highest-judgement item of the three, since mouse capture is a real consequence; a reasonable clean version moves it into the control if it matters ("Resume (captures mouse)") or drops it.
   - "PAUSED" itself the user underlined too: in a pause overlay the state is already shown by the frozen game and the menu; Microsoft's "Don't feel obliged to use main instruction text if adding it would only be redundant or obvious from the context of the UI" covers it. Clean version: one plain heading "Paused" with the four buttons, or the buttons alone.

6. **Neighbouring patterns that are not redundancy.** Promotional adjectives ("seamless", "unlock"), invented statistics, em-dash density, gradient text and default fonts appear in every catalogue but are not redundant copy. They are listed in the taxonomy only where they double as padding (R13 triplets, R17 closers) and otherwise left out of scope.

---

## D. Taxonomy of redundancy patterns (consolidated, deduplicated)

Judgement: **low** means two strings or a fixed list decide it; **medium** means it needs the element role and some semantic comparison; **high** means people disagree and context decides.

### R1. Kicker echo (eyebrow that announces the heading)
Definition: small text above or beside a heading (eyebrow, kicker, overline, pill, "TRUSTED BY") that restates, paraphrases or merely names what the heading already says.
- Slop: "OPERATION ON HOLD" over "PAUSED". Clean: "Paused".
- Slop: "FEATURES" over "Everything your team needs". Clean: "What it does" (or the heading alone).
Sources: Hallmark gate 54; tasteskill house rules and tells.md; unslop-ui TELLS.md; anti-ai-slop presentations.md; TeneX; solodesign; Krebs ("Badge right above the hero H1"); Microsoft ("Omit control labels that restate the main instruction"). Exceptions: REI Cedar categories, GOV.UK caption.
Judgement: medium (paraphrase detection; category versus restatement).

### R2. Decorative label slot (inert kicker or tracked-caps chrome)
Definition: a label that carries no fact, destination or identifier and exists to look designed; often tiny, uppercase, letter-spaced or monospace, repeated above every section.
- Slop: "THE BIG IDEA" above every slide title. Clean: no label; a section marker only at real section breaks.
- Slop: "// 2026 EDITION" mono strip under the hero. Clean: delete, or state the fact in a sentence if it is one.
Sources: Hallmark gates 54, 66, 67; anti-ai-slop (kicker on every slide; mono micro-labels); tasteskill ("No tiny uppercase letter-spaced labels"; 16% vs 57% in its 247-site study); unslop-ui TELLS.md; Microsoft (all caps as "screaming"); Krebs ("All-caps headings and section labels").
Judgement: medium (needs style features; the fact-or-not test is semantic).

### R3. Subtitle echo (subhead or standfirst restating the heading)
Definition: a supporting line directly under a heading that says the heading again in other words instead of adding a fact.
- Slop: "Simple pricing" / "Pricing that's simple and easy to understand." Clean: "Simple pricing" / "£9 a month per editor. Viewers are free."
- Slop: "Settings" / "Manage your settings here." Clean: "Settings" alone.
Sources: deslop-copy ("A subtitle that repeats the heading."); tasteskill copy.md ("It never repeats the headline in other words."); Microsoft ("Don't repeat the main instruction with slightly different wording."); Impeccable clarify.md ("Say each idea once").
Judgement: medium.

### R4. Heading echo in body (first sentence or closing line restates the heading)
Definition: a section whose first sentence, or last line, restates its own heading.
- Slop: "Offline mode" / "Offline mode lets you work offline." Clean: "Offline mode" / "Edits made without a connection sync when you reconnect."
- Slop: section "Fast exports" ending "Exports, made fast." Clean: end on the last fact.
Sources: humanizer §24; Hallmark gate 69 ("a section's last line restating its heading"); GOV.UK ("Headings should describe ... they should not be part of the text").
Judgement: low to medium (string overlap catches most).

### R5. Label restatement (helper text, tooltip, card body or bold lead-in repeating its label)
Definition: secondary text attached to a control, card or list item that repeats the control's or item's own label.
- Slop: button "Add new line" with tooltip "Add new line". Clean: no tooltip.
- Slop: field "Email" with helper "Enter your email address". Clean: "Email" with helper "We send the receipt here" or none. Also "**Speed:** Speed is improved" becomes "Exports finish in under 2 s."
Sources: NN/g tooltip guidelines; Microsoft ("Don't have supplemental explanations that merely restate the label for consistency"); Impeccable ("Helper text answers an implicit question instead of restating the control"); unslop rule 16; humanizer §19; Wikipedia (inline-header lists); GOV.UK (help text only if research shows the need).
Judgement: low (near-verbatim) to medium (paraphrase).

### R6. Explaining the obvious (instructions for self-evident controls or states)
Definition: text that tells the user how to operate a standard control, or what an action they can already see will do, when the UI already shows it.
- Slop: "Resume recaptures the mouse." under a Resume button. Clean: drop it, or "Resume (captures mouse)" if the capture surprises people.
- Slop: "Click the button below to save your changes." above "Save". Clean: "Save changes".
Sources: GOV.UK ("If you find yourself having to explain how the user interface works, that's a sign something has gone wrong."); Microsoft ("don't explain the obvious"; "Omit instructional verbs that apply to all controls"); Apple (describe the on state only); unslop-ui ("Over-explained microcopy/tooltips").
Judgement: high (whether a consequence is obvious depends on the user and product).

### R7. Vacuous action label (generic or gesture-based button and link text)
Definition: an action label that names the gesture or a stock phrase instead of the outcome, so surrounding copy has to explain it.
- Slop: "Click here to read the report". Clean: "Read the 2026 report".
- Slop: "Get started" / "Learn more" pair. Clean: "Start a free trial" plus a plain text link "See pricing".
Sources: NN/g writing links; GOV.UK links; Apple; Mailchimp; Microsoft ("Prefer specific labels over generic ones"); Impeccable ("not the gesture used to trigger it"); Hallmark gate 19; unslop-ui ("CTA clones"); deslop-copy ("Not a lone 'Continue', 'OK', 'Submit'").
Judgement: low (fixed list of offenders), medium for whether context already makes "Continue" clear.

### R8. Placeholder echo (placeholder repeating or replacing the label)
Definition: placeholder text that duplicates the visible label, or stands in for a missing label.
- Slop: label "Name", placeholder "Name". Clean: label "Name", no placeholder, or an example "e.g. Ada Lovelace" if format matters.
- Slop: no label, placeholder "Search products...". Clean: visible label or accessible name "Search products", placeholder empty.
Sources: GOV.UK text input; NN/g placeholders; Impeccable ("placeholders are examples, not labels"); Apple (placeholder for format hints, with labels).
Judgement: low.

### R9. Chrome duplication (title repeated across window, tab, nav, page and hero)
Definition: the same name or title printed in two structural slots that the user sees together.
- Slop: tab "Billing" then page heading "Billing" then card heading "Billing details". Clean: tab "Billing", card heading "Payment method".
- Slop: dialog title "Delete project dialog". Clean: title "Delete project?" with button "Delete project".
Sources: Shopify ("adding a page title under the page Title bar"); Microsoft (no "dialog" or "progress" in titles; hide caption when shown prominently); deslop-ui ("Duplicate logo in header and hero"); tasteskill ("Drop the name, it is in the navigation"); GOV.UK (label as heading so screen readers hear it once).
Judgement: medium (a breadcrumb or tab legitimately repeats a page name).

### R10. Shared prefix across siblings (repeated lead-in on every item)
Definition: options, cards or list items that all start with the same words, which belong once in the group label.
- Slop: "Notify me when someone comments" / "Notify me when someone mentions me". Clean: group "Notify me when someone" with options "Comments" / "Mentions me".
- Slop: three cards "Easily export…", "Easily share…", "Easily sync…". Clean: "Export to CSV", "Share by link", "Sync every 5 min".
Sources: Microsoft ("If all the options have the same introductory text, move that text to the group label."); unslop-ui ("Mirrored feature/benefit pairs: 'X so you can Y' on every feature"); humanizer §7 (repeated sentence openings).
Judgement: low (prefix overlap is mechanical); exceptions for deliberate parallel phrasing (Microsoft also asks for parallel labels).

### R11. Decorative status marker (dot pill, badge or emoji with no live state)
Definition: a coloured dot, pill or badge ("● Live", "● New", "Now with AI ✨", "BETA") placed as ornament on content that has no changing state, or on every item so it no longer distinguishes anything.
- Slop: "● Available for work" in a portfolio header. Clean: "Taking projects from January" in the contact line, or nothing.
- Slop: "● New" on all six feature cards. Clean: no badge, or one badge on the one feature shipped this month.
Sources: Hallmark gate 63 ("decorative status dots before labels"; legal for "a live state") and gate 66 (`BETA` pill in the hero); tasteskill tells.md ("Decorative status dot"); unslop-ui ("Hero pill badge ... + sparkle + dot"; copy-paste status pill); solodesign ("eyebrow chrome"); GOV.UK tag (status only; fewer is better; absence can carry the meaning); Wikipedia (emoji in titles and list entries).
Judgement: medium (needs to know whether the state is real and whether it differentiates).

### R12. Decorative numbering (index kickers duplicating visible order)
Definition: "01 /", "(001)", "Step 2", "Phase 02" or "01 / 4" counters printed above items whose order is already shown by position, or where order does not matter.
- Slop: "01 / Connect", "02 / Configure", "03 / Ship" as three cards in a row. Clean: an ordered list "Connect your repo", "Pick a branch", "Deploy".
- Slop: "Step 1: Step one, create an account". Clean: "Create an account".
Sources: tasteskill ("No index numbering"; "Step labels that duplicate numbers"); TeneX ("above three steps already in reading order"); Krebs ("Numbered '1, 2, 3' step sequences"); anti-ai-slop (numbered corner badges, "01 / 10" counters); Hallmark gate 66 (`01 / 4` on tiles); Impeccable numbered-label check.
Judgement: low to medium (a real multi-step flow is a legitimate use).

### R13. Forced triplet (tricolon tagline or three-part padding)
Definition: three parallel words or clauses where the content has fewer distinct ideas, typically restating one claim three ways.
- Slop: "Fast. Simple. Secure." Clean: "Loads in 0.4 s on a 3G connection."
- Slop: "Plan, build and ship your ideas." Clean: "Turn a spec into a deployed site."
Sources: Wikipedia §3.5; stop-slop; humanizer §6; unslop rule 10; unslop-ui TELLS.md; deslop-copy; tasteskill copy.md; anti-ai-slop.
Judgement: medium (three real items are fine; the test is whether each adds a distinct fact).

### R14. Cross-section echo (same fact or CTA repeated across the page)
Definition: one fact, reassurance or call to action printed in several sections, or a closing band that mirrors the hero.
- Slop: "No credit card required" under four buttons. Clean: once, beside the primary sign-up.
- Slop: "Sample data, names are made up" at the foot of every screen. Clean: one global note.
Sources: Hallmark gate 69 (one fact in four or more sections); unslop-ui ("Final CTA band mirrors the hero"); deslop-ui ("one global note"); anti-ai-slop (per-slide disclaimer); Shopify ("presenting the same content in different ways on the same page"); Mailchimp (avoid repeating body links in related articles).
Judgement: low (count repeats) to medium (tables and ledgers repeat by nature).

### R15. Filler words in UI strings
Definition: words that add length and no meaning in a label or message: "please", "now", "simply", "easily", "your", "in order to", "oops".
- Slop: "Please click Apply now". Clean: "Apply".
- Slop: "Your Favorites". Clean: "Favorites".
Sources: GOV.UK ("please", "now"); Apple (possessive pronouns; "oops!"); Microsoft style guide (adverbs, "in order to"); stop-slop (adverbs); Strunk.
Judgement: low (fixed lists), with a tone caveat from Microsoft.

### R16. Self-describing copy (text about the interface or document instead of its subject)
Definition: copy that narrates the layout, the design or the document's own structure.
- Slop: "A clean, minimal layout for your projects." Clean: list the projects.
- Slop: "The table below compares the plans." Clean: the table, with a heading "Plans".
Sources: tasteskill ("Explainer captions that narrate the design"); humanizer §25; stop-slop (meta-commentary); Microsoft (don't explain the obvious).
Judgement: medium.

### R17. Restating closer or summary
Definition: a closing line, paragraph or section that summarises what the reader has just seen, or a one-liner that only adds emphasis.
- Slop: three feature cards followed by "Everything you need, all in one place." Clean: end after the cards.
- Slop: "That's it. That's the whole setup." Clean: end on the last step.
Sources: humanizer §2; stop-slop (emphasis crutches, dramatic fragmentation); Wikipedia §2.7 and §13.3; unslop rule 25.
Judgement: medium.

### R18. Comment restating code (code comments and docstrings)
Definition: a comment or docstring that says what the next line already says.
- Slop: `x = x + 1  # Increment x`. Clean: no comment, or `# Compensate for border`.
- Slop: `/** Gets the user. */ getUser()`. Clean: no docstring, or one that states a non-obvious contract ("Returns null for deleted users").
Sources: PEP 8; humanizer §26 (re-explaining what the reader knows), by analogy.
Judgement: low (token overlap with the code) to medium.

### Out of scope (catalogued as slop, but not redundancy)
Promotional vocabulary ("unlock", "seamless"), invented statistics and testimonials, em-dash density, negative parallelisms ("not just X, it's Y") where both halves are distinct, gradient and font defaults. Sources: all of section A, plus NN/g's 1997 finding against promotional style. A linter could carry these as separate rules; the redundancy model should not be trained against them.
