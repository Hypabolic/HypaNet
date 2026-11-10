# Slop smell: prior art and baselines

Survey date: 2026-11-10. Written for the `slop/` experiment, in which Fable 5.2 (`claude-fable-5-2`) hand-writes a small model that flags redundant, explanatory or repetitive copy in user interfaces (later, possibly, code comments and documents), to run deterministically inside a linter with no LLM at lint time.

Every number below carries its source. Where a source was a secondary summary rather than the primary paper or page, it says so. Licences marked "check" were not confirmed from the licence file itself.

The target, restated so the survey has something to measure against. In the motivating pause menu:

| Element | Text | Why it is slop |
| --- | --- | --- |
| Eyebrow (small, letter-spaced) | OPERATION ON HOLD | Says what the heading says, in more words |
| Heading | PAUSED | Fine on its own; redundant with the eyebrow |
| Subtitle | Resume recaptures the mouse. | Explains a button the player is about to press; restates the RESUME label |
| Buttons | RESUME / SETTINGS / CONTROLS / RESTART MISSION | Fine |

The user's wider framing: repetition within the UI (headings, labels, "dot highlights" such as "● Live", "● New"). So the task has two parts that prior art treats very differently:

1. **Cross-element redundancy**: one string restates another string on the same screen (eyebrow ≈ heading, subtitle ≈ button, helper text ≈ label).
2. **Decorative repetition**: a pattern repeated across a screen or page (an eyebrow above every section, a status dot on every row, a pill on every card).

---

## 1. Guidance that already describes the target

These are the closest things to a spec. None is a model; most are rules for a human or an agent.

### Microsoft, "User Interface Text" (Windows 7 UX guidelines)

<https://learn.microsoft.com/en-us/windows/win32/uxguide/text-ui> (source: [MicrosoftDocs win32 repo](https://github.com/MicrosoftDocs/win32); licence of the docs repo: check, MicrosoftDocs repos are usually CC BY 4.0 for prose).

The most direct statement of cross-element redundancy found. Quoted from the page:

- "Review each window and eliminate duplicate words and statements, both within and across controls."
- "Look for redundant text in window titles, main instructions, supplemental instructions, content areas, command links, and commit buttons. Generally, leave full text in main instructions and interactive controls, and remove any redundancy from the other places."
- Supplemental instructions: "Don't repeat the main instruction with slightly different wording."
- Control labels: "Omit control labels that restate the main instruction."
- Supplemental explanations: "Don't have supplemental explanations that merely restate the label for consistency."
- "Don't use all capital letters for emphasis ... There is no need to add, for example, the term WARNING in all capital letters."
- "Don't feel obliged to use main instruction text if adding it would only be redundant or obvious from the context of the UI."

It also gives the precedence rule a detector needs: **keep the text on interactive controls and the main instruction; flag the copy that repeats them elsewhere.** In the pause menu that maps to: keep PAUSED and RESUME, flag OPERATION ON HOLD and "Resume recaptures the mouse." The guide is a manual review procedure; it ships no checker.

### Anti-slop agent skills (2025–2026)

The user asked whether such skills exist. They do, and several name the exact patterns in the pause menu. They are written as instructions to a generating agent, not as detectors, and none publishes a precision figure.

| Skill / tool | Where | What it says about the target | Licence |
| --- | --- | --- | --- |
| taste-skill (`design-taste-frontend`, v1/v2/v2-experimental), Leonxlnx | <https://github.com/Leonxlnx/taste-skill>; installed locally at `~/.claude/skills/design-taste-frontend*` | v2 house rule: "No eyebrows. No small label above a heading, no pill badge above the headline, no 'New ·' chip." Also "No tiny uppercase letter-spaced labels." v2-experimental: "Maximum 1 eyebrow per 3 sections" (counted mechanically as `≤ ceil(sections/3)`), "ZERO decorative status dots by default ... Only acceptable when conveying real semantic state", "NO micro-meta-sentences under eyebrows", and a mandatory copy self-audit of every visible string. v2 reports, from its own study of 247 gallery sites, tiny tracked caps on 16% of sites scored ≥ 7.5 against 57% of sites under 6.5, and pill badges 14% against 34% (source: the skill's SKILL.md; the panel scores are the skill author's, not independently checked). | check repo |
| Hallmark (`sloplint`), Nutlope | <https://github.com/Nutlope/hallmark> | Catalogue of AI-UI anti-patterns with a `hallmark audit` command and a linter that checks "the eyebrow above every heading" and tracked-capital labels. A secondary summary ([instagit](https://instagit.com/Nutlope/hallmark/what-are-the-named-ai-slop-anti-patterns.md)) lists "Eyebrow on every section" as a major pattern. | MIT (repo README) |
| `unslop`, local skill | `~/.claude/skills/unslop/SKILL.md` | 33 numbered prose rules; relevant ones: rule 16 "Inline-header lists ... a bold label and colon that restates the line", rule 23 filler phrases, rule 27 "if the sentence could appear unchanged in another project's docs, it says nothing about this one". Prose-level, not UI-structure-level. | local |
| Wikipedia, "Signs of AI writing" | <https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing> | Editor-maintained catalogue (AI vocabulary, rule of three, negative parallelisms, section summaries, excessive boldface, and more). Source of most of the rule sets below. Not about UI. | CC BY-SA 4.0 |
| vale-signs-of-ai-writing, ammil industries | <https://github.com/ammil-industries/vale-signs-of-ai-writing>; write-up <https://ammil.industries/signs-of-ai-writing-a-vale-ruleset> | Vale rules derived from the Wikipedia page in three tiers (error / warning / suggestion). The author publishes no precision figures and says the suggestion tier is the most prone to false positives. | CC BY-SA 4.0 |
| slop-guard, Eric Tramel | <https://github.com/eric-tramel/slop-guard>; write-up <https://eric-tramel.github.io/blog/2026-02-18-slop-guard> | Regex prose linter, score 0–100, about 80 vocabulary words and about 50 assistant phrases (author's counts), plus structure rules (repeated bold-header blocks, runs of bullets, "X, not Y" contrast pairs). Its own calibration: 5 of 7 human essays from 1986–2015 scored "clean" (81–91), Norvig 63, and Hamming's 14,000-word transcript 39 ("heavy"), a false positive the author attributes to length. Blog-length prose, not UI. | check repo |
| Others seen, not reviewed in depth | [anti-slop-frontend (Pi)](https://pi.dev/packages/@blackbelt-technology/anti-slop-frontend), [deslop](https://skillselion.com/skills/viktorbezdek/skillstack/deslop), [anti-ui-slop](https://www.skills.sh/discountry/ritmex-skills/anti-ui-slop), [no-ai-design-slop](https://www.ui-skills.com/skills/mengto/no-ai-design-slop), [wystack anti-patterns](https://cdn.jsdelivr.net/npm/@wystack/agent-kit@0.12.0/wystack-agent-kit/references/anti-patterns.md) | The Pi skill states the eyebrow cap as a pass/fail count (`eyebrow count > ceil(sections/3)`). deslop audits button labels, error messages, empty states, tooltips and onboarding text. One design-system page ([sportingscouter Eyebrow](https://design-system.sportingscouter.com/eyebrow/)) states the rule from the other side: an eyebrow "may never carry information that appears nowhere else". | varies |

**What this means for the spec.** The skills give a usable label vocabulary and two kinds of rule:

- *Count rules* (eyebrows per section, decorative dots per section, pills per card) are structural and can be checked by a few lines of code once the linter knows which node is an eyebrow. A network adds little here.
- *Restatement rules* (eyebrow restates heading; subtitle restates a button; supplemental text restates the main instruction) need a judgement of meaning overlap between two short strings. No skill or tool surveyed implements that check; they leave it to the agent's reading. This is where a hand-written model could earn its place.

---

## 2. Prose and copy linters

All of these work on a sentence or a phrase at a time, by phrase list, regex or simple metric. None compares one UI string with another.

| Tool | URL | Redundancy coverage | Licence |
| --- | --- | --- | --- |
| Vale (engine) | <https://vale.sh>, package hub <https://vale.sh/hub/>, explorer <https://vale.sh/explorer> | Engine for YAML rules (existence, substitution, occurrence, repetition, sequence, metric). Its `repetition` check catches repeated adjacent tokens only. | MIT |
| Vale Microsoft | <https://github.com/errata-ai/Microsoft> | `Microsoft.Wordiness` (substitution list, suggestion level). The explorer listed 47 rules at the time of a search summary; the repo listing said 40 (counts from search summaries, check). | MIT |
| Vale Google | <https://github.com/errata-ai/Google> | No dedicated wordiness or redundancy rule found. | MIT |
| Vale write-good | <https://vale.sh/hub/> | 8 rules (explorer count, via search summary): passive voice, weasel words, "so" at sentence start, adverbs, clichés, "there is/are", too wordy. Port of the npm `write-good` package. | MIT |
| Vale proselint / proselint CLI | <https://github.com/amperser/proselint> | `redundancy.garner` (Garner's Modern American Usage list, e.g. "while at the same time" → "while"), `redundancy.misc`, `redundancy.ras_syndrome` ("ATM machine"), `nordquist.redundancy`. Source file for v0.14.0: <https://sources.debian.org/src/python3-proselint/0.14.0-2/proselint/checks/redundancy/misc.py>. | BSD-3-Clause |
| Vale Readability | <https://vale.sh/hub/> | Flesch-Kincaid, Gunning Fog and other metrics. No redundancy rule. | MIT |
| GDS content style guide for Vale | <https://github.com/alphagov/gds-vale-styles> | GOV.UK content style as Vale rules; code MIT, the guide text Crown copyright under the Open Government Licence v3.0 (per the repo). | MIT / OGL v3.0 |
| alex | <https://github.com/get-alex/alex> | Insensitive and inconsiderate wording only. Not relevant to redundancy. | MIT |
| retext plugins (unified) | <https://unifiedjs.com/explore/keyword/retext>; [retext-redundant-acronyms](https://unifiedjs.com/explore/package/retext-redundant-acronyms/); [retext-simplify](https://www.npmjs.com/package/retext-simplify); retext-repeated-words | Redundant acronyms ("ATM machine"), wordy phrases ("be advised"), immediately repeated words ("the the"). | MIT |
| textlint | <https://textlint.github.io> | Rule framework; the redundancy rules are mostly Japanese (`textlint-rule-ja-no-redundant-expression`, `no-doubled-joshi`, `no-doubled-conjunction`). A `textlint-rule-preset-ai-writing` exists for Japanese AI-writing tells ([textlint-ja](https://github.com/textlint-ja/textlint-rule-preset-ai-writing)). | MIT |
| LanguageTool | Rule pages, e.g. [EN_REDUNDANCY_REPLACE](https://community.languagetool.org/rule/show/EN_REDUNDANCY_REPLACE?lang=en-GB), [ADD_AN_ADDITIONAL](https://community.languagetool.org/rule/show/ADD_AN_ADDITIONAL?lang=en) | Category "Redundant Phrases" (ID `REDUNDANCY`, type style), pattern rules like "tuna fish" → "tuna". A forum thread (undated) says about 161 redundancy rules, with the wordiness category inactive by default ([forum](https://forum.languagetool.org/t/wordiness-keeping-within-your-word-count-by-removing-or-reducing-a-sentence/2991)). | LGPL-2.1 |
| Hemingway Editor | <https://hemingwayapp.com/help/docs/highlighted-issues> | Highlights hard sentences, adverbs (any "-ly" word, hence false positives such as "Kelly"), passive voice and complex words. No redundancy check. Proprietary. | proprietary |

UI-string-aware ESLint rules exist, but only to find strings, not to judge them: `i18next/no-literal-string` ([eslint-plugin-i18next](https://github.com/edvardchen/eslint-plugin-i18next/blob/HEAD/docs/rules/no-literal-string.md), JSX text by default from v6), `react/jsx-no-literals`, and `@intlify/vue-i18n/no-raw-text` ([docs](https://eslint-plugin-vue-i18n.intlify.dev/rules/no-raw-text)). These are useful plumbing: the slop linter can reuse their JSX-text extraction to get the strings and their element tree.

**Takeaway.** Off-the-shelf linters cover "redundant phrase inside one sentence" (tuna fish, ATM machine) and nothing else. On the pause menu, none would fire: "OPERATION ON HOLD", "PAUSED" and "Resume recaptures the mouse." contain no listed phrase. A fair rules baseline therefore has to be written for this task (see §8).

---

## 3. Redundant code comments

### Research

| Work | URL | What it does | Numbers (source) |
| --- | --- | --- | --- |
| Steidl, Hummel, Jürgens 2013, "Quality analysis of source code comments", ICPC | <https://teamscale.com/hubfs/26978363/Publications/2013-quality-analysis-of-source-code-comments.pdf> | Seven comment categories; a quality model over consistency, completeness, coherence (comment vs code) and usefulness. Coherence is measured by lexical similarity between a member comment and the member name: the original "trivial comment" detector. | 830 Java comments from 12 projects and 500 C++ comments (per a later survey, check against paper) |
| Pascarella and Bacchelli 2017, "Classifying code comments in Java open-source software systems", MSR | <https://arxiv.org/abs/1703.09131> (check) | Taxonomy of 16 comment types and a supervised classifier; public dataset. | see paper |
| NLBSE tool competition (code comment classification) | <https://nlbse2024.github.io/tools>, <https://nlbse2023.github.io/tools> | Sentence-level classification of class comments in Java, Python, Pharo. Categories (summary, usage, pointer, deprecation, ownership, …) do not include "redundant". | NLBSE'24: 82,089 sentences, 19 categories; STACC average F1 0.74 against a 0.31 baseline (competition reports, via search summary) |
| Louis, Dash, Barr, Sutton 2018, "Deep Learning to Detect Redundant Method Comments" (CRAIC) | <https://arxiv.org/abs/1806.04616> | Treats a comment as redundant when it is predictable from the code ("comment entailment"); an LSTM seq2seq model scores P(comment ∣ code), low perplexity = redundant. No manual labels for training. | Trained on 3M method-comment sentence pairs from the GitHub Java Corpus (14,785 projects). Human study: 500 pairs from 45 projects, 4 annotators, pairwise Cohen's κ 0.1–0.29; all four agreed on 95 of 500. Mean perplexity 9.50 for majority "entails" (n=237) vs 115.73 "not entailed" (n=39). `@param` sentences averaged 6.49 perplexity, `@throws` 2.50, non-Javadoc about 15.58. (ar5iv full text) |
| Jabrayilzade et al. 2024, "Taxonomy of inline code comment smells", EMSE | <https://link.springer.com/10.1007/s10664-023-10425-5> | 11 smell types; "obvious" (restates the code) is one, alongside misleading, commented-out code, irrelevant, task, too much information, attribution, beautification, non-local, vague, and no comment on non-obvious code. Survey respondents said some smells, "obvious" included, can help some readers. | 2,447 manually labelled inline comments (journal; a follow-up says 2,448); data on figshare |
| Öztaş et al. 2025, "Towards Automated Detection of Inline Code Comment Smells" | <https://arxiv.org/abs/2504.18956> | ML classifiers and GPT-4 on the Jabrayilzade data (9 of 11 categories labelled), augmented with code context. | Random Forest 69% accuracy, MCC 0.44; GPT-4 34% → 55% accuracy with code added (abstract). Per-class "obvious" results not in the abstract. Paper licence CC BY-NC-ND 4.0; data licence not stated. |
| Çağlar, Çağlar, Tüzün 2025, "Automated Inline Comment Smell Detection and Repair with LLMs", ASE | <https://conf.researchr.org/details/ase-2025/ase-2025-papers/36/Automated-Inline-Comment-Smell-Detection-and-Repair-with-Large-Language-Models> | GPT-4o-mini, o3-mini, DeepSeek-V3, Codestral-2501, zero- and few-shot. | Best median: o3-mini few-shot, macro-F1 0.41, MCC 0.50, accuracy 0.72 (paper abstract, via search summary) |

### Tools

- **ESLint core, Pylint, Ruff**: no rule for comments that restate code. Ruff's comment rules are about `noqa` directives (e.g. RUF100) and commented-out code (ERA001); CodeQL has commented-out-code queries for several languages; Checkstyle `JavadocMethod` checks for *missing* `@param` tags, the opposite of redundancy.
- **New third-party ESLint plugins aimed at AI-written comments** (small, recent, not evaluated; listings only): [eslint-plugin-deslop](https://www.pkgstats.com/pkg:eslint-plugin-deslop) (restating comments, trivial JSDoc such as "Gets the user name", auto-fix), [eslint-plugin-meaningful-docs](https://www.pkgstats.com/pkg:eslint-plugin-meaningful-docs) (`no-obvious-comments`, `no-restating-comments`), [eslint-plugin-ai-guardrails](https://eslint-ai-guardrails.vercel.app/) ("Set x to 5" above `const x = 5`), [@dannote/sloplint](https://www.pkgstats.com/pkg:@dannote/sloplint) (ast-grep, 8 languages: obvious, narrator and step comments). All appear to be token-overlap heuristics; none publishes precision.

**Takeaway.** Comment redundancy is studied and has one labelled dataset with an "obvious" class (Jabrayilzade, 2,447 comments). Two warnings carry over to UI copy: human agreement on "redundant" is low (CRAIC κ 0.1–0.29), and strong LLMs score modestly on smell classes (macro-F1 0.41 at best in the ASE 2025 study). The comment track is a reasonable phase 2, with Jabrayilzade's "obvious" labels as an external test, subject to its licence.

---

## 4. AI-text and slop detection: what precision to expect

| Work | URL | Finding relevant to this experiment |
| --- | --- | --- |
| Liang et al. 2023, "GPT detectors are biased against non-native English writers", *Patterns* | <https://arxiv.org/abs/2304.02819> | 7 detectors, 91 TOEFL essays: average false-positive rate 61.22%; 18 of 91 flagged by all 7, 89 of 91 by at least one; native 8th-grade essays classified near-perfectly. Cause: low perplexity in plainer writing. Prompting GPT to enrich the language cut the false-positive rate by over 49% (summary of the paper). |
| Shaib, Chakrabarty, Garcia-Olano, Wallace 2025, "Measuring AI 'Slop' in Text" | <https://arxiv.org/abs/2509.19163> | 11-code taxonomy in three themes: information utility (density, relevance), information quality (factuality, bias), style quality (repetition, templatedness, coherence, fluency, verbosity, word complexity, tone). Three professional copy-editors; on the shared subset, binary "slop" Cohen's κ was −0.15, 0.29 and 0.06 across pairs (Gwet's AC1 0.12, 0.42, 0.28). Zero-shot LLMs against a 34% human positive rate: GPT-5 κ 0.01, precision 0.38, recall 0.12. A linear model over automatic metrics reached AUPRC 0.52 (news, prevalence 0.25). (arXiv v2 HTML) |
| "Antislop" (arXiv 2510.15061, 2025) | <https://arxiv.org/abs/2510.15061>; [antislop-sampler](https://github.com/sam-paech/antislop-sampler) | Mines over-represented words and phrases from model output against human baselines and suppresses them at decode time (8,000+ patterns, per a summary). Source of the EQ-Bench "slop score". About generation, not detection, but a method for building phrase lists from frequency ratios. MIT. |

**Implications.**

1. Do not frame the detector as "AI-written or not". Authorship detection has well-documented false positives on plain writing (Liang: 61.22% average on TOEFL essays). Short, plain UI strings are the lowest-perplexity text there is. The target must be a property of the text itself (this string restates that string; this label is decorative), which a human can verify on the screen.
2. Broad "slop" labels have near-chance agreement (Shaib: κ −0.15 to 0.29). Narrow, structural labels do better there: factuality AC1 0.76, bias 0.67, structure 0.51. The spec should define a few narrow classes (restates-sibling, restates-control, decorative-label, decorative-dot, explains-the-obvious) with written decision rules, and measure the label author's agreement with a second labeller on a sample before scoring anything.
3. Score with precision first. A linter that cries wolf gets switched off. Report per-element false-positive rate on clean screens alongside F1, as HypaNet already reports FA (false-alarm) rates.

---

## 5. Similarity of short strings without pretrained embeddings

Fable must write any lexical knowledge into the weights by hand. Prior art shows how far lexicon-based methods go on sentence-pair paraphrase.

| Method | URL | Result on MSRP (Microsoft Research Paraphrase Corpus test set; ACL wiki table) |
| --- | --- | --- |
| tf-idf cosine baseline | [ACL wiki](https://aclweb.org/aclwiki/Paraphrase_Identification_(State_of_the_art)) | 65.4% accuracy, F1 75.3% (test set is 66.5% positive) |
| Mihalcea, Corley, Strapparava 2006: six WordNet word-similarity measures and two corpus measures, combined | [AAAI 2006](https://mlanthology.org/aaai/2006/mihalcea2006aaai-corpus) | 70.3%, F1 81.3% |
| Islam and Inkpen 2007: semantic plus string similarity | ACL wiki | 72.6%, F1 81.3% |
| Fernando and Stevenson 2008: WordNet JCN similarity in a matrix | ACL wiki | 74.1%, F1 82.4% |
| Best listed (supervised TF-KLD, Ji and Eisenstein 2013) | ACL wiki | 80.4%, F1 85.9% |

Building blocks Fable can write by hand:

- **Stemming.** Porter (1980) suffix stripping, <https://tartarus.org/martin/PorterStemmer/> (free to use). In a byte- or character-level network this is a set of suffix-matching circuits; for UI copy a short suffix list (‑s, ‑es, ‑ed, ‑ing, ‑ion, ‑er) probably covers most cases.
- **Synonym groups.** Open English WordNet, <https://github.com/globalwordnet/english-wordnet>, CC BY 4.0 with Princeton's WordNet 3.1 notice kept (Princeton licence: <https://wordnet.princeton.edu/node/10>). UI copy uses a small vocabulary, so a few hundred concept groups (pause / hold / suspend / paused; resume / continue / return; settings / options / preferences; delete / remove) may cover most redundancy. Fable can write these groups from its own knowledge; WordNet is a reference for checking coverage, and any group copied from it carries attribution.
- **Phrase paraphrases.** PPDB 1.0 (Ganitkevitch et al. 2013, <https://aclanthology.org/N13-1092/>): 8 million lexical and 73 million phrasal English paraphrase pairs, "freely available" from paraphrase.org (licence: check). Useful for coverage checks, too large to hand-write.
- **Identifier splitting** for the comment track: camelCase and underscore splitting, then stem matching against comment words (standard in Steidl 2013's coherence metric). One thesis reports plain camelCase splitting correct on about 18% of identifiers in one system against 93% for a dictionary method ([Polytechnique Montréal](https://publications.polymtl.ca/294); single data point, in French).

**Takeaway.** The UI problem is easier than MSRP: strings are 1–8 words, the vocabulary is narrow, and the pairs to compare come from the element tree (eyebrow vs the heading it sits above; subtitle vs its buttons), not from all pairs on the page. A hand-written stem-plus-synonym-group overlap with role-aware thresholds is a plausible design, and the same design written as plain code is the rules baseline.

---

## 6. Sources of natural UI text, with licences

None of these sets carries redundancy labels. Natural strings serve as mostly-clean negatives and as raw material; positives have to be constructed (see §8).

| Source | URL | What it contains | Licence | Use here |
| --- | --- | --- | --- | --- |
| AOSP app strings (`res/values/strings.xml`: Settings, Launcher3, PackageInstaller, setupdesign, …) | e.g. <https://android.googlesource.com/platform/packages/apps/Settings/+/c5dfd3e/res/values/strings.xml> | Thousands of labels, titles, summaries and dialog texts with translator comments, grouped by screen via resource names | Apache-2.0 (file headers) | Yes. Best legally clean source of mature, edited UI copy. |
| Firefox en-US strings | <https://github.com/mozilla-l10n/firefox-l10n-source> | All Firefox UI strings in Fluent (`.ftl`), with message IDs that group strings by dialog | MPL-2.0 (repo README) | Yes, with notice. Fluent groups (title, description, button in one message) give element roles directly. |
| Rico (Deka et al. 2017) | <https://www.interactionmining.org> (official site; the `rico.html` page returned 404 on 2026-11-10); research summary <https://research.google/pubs/an-early-rico-retrospective-three-years-of-uses-for-a-mobile-app-dataset/> | 66,261 UI screens from about 9.7k Android apps with view hierarchies (screens count from the [HF loader](https://huggingface.co/datasets/shunk031/Rico)) | Unconfirmed: the HF loader says "unknown"; a Voxel51 mirror says CC BY 4.0 and excludes commercial redistribution without permission ([card](https://docs.voxel51.com/dataset_zoo/datasets_hf/rico.html)) | Research use, local only. Do not redistribute strings until the licence is confirmed. |
| Rico semantic annotations (Liu et al. 2018, "Learning Design Semantics for Mobile Apps", UIST) | via the Voxel51 card above | 20,386 unique button text strings clustered into 197 text-button concepts (card figures) | as Rico | A ready-made concept inventory for button synonyms (the "resume / continue" groups). |
| Screen2Words and Google Screen Annotation (Rico-derived) | <https://github.com/google-research-datasets/screen_annotation>; [RICO-Screen2Words](https://huggingface.co/datasets/Billyyer/RICO-Screen2Words-Rename) | Screen summaries; element annotations with text for a Rico subset | CC BY 4.0 | Yes, with attribution. |
| WebUI (Wu et al. 2023, CHI) | <https://uimodeling.github.io>, <https://arxiv.org/abs/2301.13280> | About 400,000 rendered web pages with accessibility trees, layout boxes and computed styles (paper). Computed styles matter: letter-spacing and font size identify eyebrows. | Not stated on the project page; crawled third-party sites | Local research only, not for release. |
| GOV.UK Design System | <https://design-system.service.gov.uk/components/button/>, <https://github.com/alphagov/govuk-design-system> | Component guidance and example copy | Code MIT; guidance Crown copyright, OGL v3.0 (per gds-vale-styles README, check) | Yes, guidance and examples. |
| Mailchimp Content Style Guide | <https://styleguide.mailchimp.com/> | Voice, tone and UI-copy guidance | CC BY-NC 4.0 (site footer) | Read as guidance; do not ship its text in a commercial tool. |
| Microsoft Terminology Collection | <https://learn.microsoft.com/en-us/globalization/reference/microsoft-terminology>; HF [ms_terms](https://huggingface.co/datasets/microsoft/ms_terms) | Terms and short UI labels with definitions (`.tbx`) | HF card says MS-PL (check against Microsoft's terms) | Possible, after checking. Full UI-string downloads were subscriber-only. |
| Jabrayilzade inline comments (comment track) | figshare, linked from the EMSE paper | 2,447 labelled comments including "obvious" | check figshare licence | External test for phase 2 if the licence allows. |

---

## 7. What is genuinely new here

1. **Cross-element restatement in a UI tree, checked deterministically.** Every linter in §2 judges one sentence at a time against a phrase list. Every skill in §1 asks an agent to judge restatement by reading. Microsoft's 2009 guidance describes the check (eliminate duplicate statements "within and across controls") as a manual review. No tool found compares an eyebrow with its heading, or a subtitle with its buttons, by meaning.
2. **Role-aware precedence.** Using the Microsoft rule (keep interactive controls and the main instruction; flag the copy that repeats them) gives a direction to each flag, which a pure similarity score does not.
3. **A hand-written network with lexical knowledge written in, not learned.** Prior lexicon methods (§5) used WordNet as an external database at run time. Here the stem and synonym knowledge lives in the weights, written by Fable, and the whole thing runs inside ESLint with no model download and no API call.
4. **Honest limits on novelty.** Decorative repetition (eyebrow and dot counts) is already specified as count rules by taste-skill and Hallmark, and is checkable with a few lines of code. The network should not get credit for it; the spec should leave those rules in plain code, or include them in the rules baseline too.

One labelling caution from the motivating example. "Resume recaptures the mouse." is redundant in the sense that it explains a button the player is about to press, but it does carry one fact the RESUME label does not (mouse capture). Under Microsoft's rules it is "explaining the obvious" rather than a restatement. The label definitions must say which of these the detector is meant to flag, or agreement will be as low as Shaib et al. measured.

---

## 8. Recommendation

### Baselines (in priority order)

1. **Rules engine written by Codex, in an isolated session, from the spec only** (house pattern from v0.2–v0.5). It should implement: stemmed token overlap (Jaccard or containment) between each element and its parent heading, sibling elements and the controls in the same container, with role-aware thresholds; a hand-listed synonym table of the same budget as Fable's; and the count rules (eyebrows per section, decorative dots per section). This is the fair rules baseline. Report alongside it, as a floor, **off-the-shelf Vale** with Microsoft + proselint + write-good + vale-signs-of-ai-writing at default settings, which should score near zero recall on the target; that number shows the gap is real rather than asserted.
2. **Same architecture trained by gradient descent** on the dev split at several data budgets × 3 seeds (house pattern: v0.2's 5 budgets × 3 seeds, v0.3's 32/128/512/2048/all). This answers the HypaNet question directly: how written weights compare with trained ones at each label budget.
3. **Prompted small local LLM** as a reference that cannot run in a linter: Qwen3.5-0.8B in the existing `envs/qwen` environment (no API spend; about 450 tok/s per MPS process, 3 workers about 850 tok/s, per the v0.4 machine notes), given the element tree and the label definitions, zero-shot and few-shot. Shaib et al. 2025 (GPT-5 zero-shot κ 0.01 on broad slop) and Çağlar et al. 2025 (o3-mini few-shot macro-F1 0.41 on comment smells) suggest it will not be strong, but it marks where a general model stands on the narrow labels. A larger local model is optional; no paid API.
4. **Trivial floors**: always-clean, and exact-match-after-stemming between parent and child. Cheap, and they keep every other number honest.

### Data that can be used legally

- **Natural negatives and raw strings**: AOSP `strings.xml` (Apache-2.0) and Firefox Fluent strings (MPL-2.0), both redistributable with notice, both grouped by screen or dialog. Screen2Words / Screen Annotation (CC BY 4.0) for screen-level structure.
- **Local research only, not redistributed**: Rico view hierarchies (licence unconfirmed) and WebUI (licence not stated; contains third-party site text). Useful for checking false-positive rates on many natural screens.
- **Guidance, not data**: Microsoft UI text guide, GOV.UK (OGL v3.0), Mailchimp (CC BY-NC 4.0: read, do not copy into the shipped tool), taste-skill and Hallmark rule text.
- **Positives must be constructed.** No public set labels cross-element redundancy in UIs. Two generators, as in v0.3–v0.5: one written for the main split and an independent one written by Codex (which Fable never sees) for OOD testing. Each composes screens (eyebrow, heading, subtitle, buttons, status pills) from natural strings and injects restatements at known places, with templates drawn from the patterns named in §1. Hand-label a small set of screens built by AI coding agents (such as the pause menu) as a separate natural-positive test, kept sealed under the usual seed rules.
- **Comment track (phase 2)**: Jabrayilzade et al. "obvious" labels as an external test, if the figshare licence allows; CRAIC's predictability framing is the method to compare against.

### Scoring

Precision-first: per-element false-alarm rate on clean screens (AOSP and Firefox dialogs) reported next to F1 on constructed positives, with a second labeller's agreement on 100–200 elements reported before any model is scored, so the ceiling is known.
