# Local skills: anti-slop rules about copy

Survey date: 2026-11-10. Source: the skills installed on this machine. No network, no paid API.

## Scope and method

Searched:

- `~/.claude/skills/` (most entries are symlinks into `~/.agents/skills/`; paths below use `~/.agents/skills/` because that is where the files live).
- `~/.claude/plugins/cache/claude-plugins-official/` and `~/.claude/plugins/marketplaces/claude-plugins-official/` (the `frontend-design` SKILL.md in the marketplace and in cache 1.1.1 is byte-identical).
- `~/.claude/skills/synced/` (Anthropic document skills: docx, pdf, pptx and so on; nothing about UI copy).
- CLAUDE.md files under `~/.claude`: only `~/.claude/plugins/cache/claude-plugins-official/mattpocock-skills/1.3.1/CLAUDE.md`. There is no `~/.claude/CLAUDE.md`.

Grep terms: slop, generic, filler, redundant, eyebrow, kicker, badge, pill, tagline, subtitle, helper text, microcopy, comment, AI-made, tells, lorem, emoji, em dash, repeat, duplicate, label, caption, dot, narrate, explainer.

Classes used below:

- **(a)** redundancy or repetition of copy. In scope for the slop model: an element that restates another element, labels what is already obvious, narrates the interface, or is the same chrome stamped on every section (eyebrows, kickers, dot pills, step labels).
- **(b)** other copy quality. Adjacent: hype words, em dashes, emoji, clichés, title case, invented proof, length caps.
- **(c)** visual only. Out of scope: colour, shadow, layout families, motion, fonts.

Skills with nothing relevant: `in-your-own-words` (restate the user's goal; no copy rules), `full-output-enforcement` (its "No recap, no repetition" at line 41 is about resuming truncated output, not copy), `sharpen`, `principle-*` (except one line noted under comments), the Anthropic synced document skills, the LSP plugins.

A scope warning that applies to most of what follows: the taste skills were written for **marketing pages**, not app or game screens. `design-taste-frontend/SKILL.md:3` says "Not for dashboards or app screens", and `technical-writing/SKILL.md:95` says "Product UI strings are not documentation". The pause-menu example sits outside both. Their rules still transfer, but the thresholds (eyebrows per section, hero element counts) were set on landing pages.

---

## 1. design-taste-frontend (tasteskill v2, the current default)

Files: `~/.agents/skills/design-taste-frontend/`. This is the richest source and the only one with working detectors (a static linter and a headless-browser probe).

### references/copy.md

| Line | Rule (quoted, short) | Class |
| --- | --- | --- |
| 35 | "A section with no claim is filler; delete it." | (a), judgement only |
| 103 | "The mad-lib: 'The AI-powered platform for modern teams'." | (b) |
| 104 | "The tricolon: 'Fast. Simple. Secure.'" | (b) |
| 105 | "The correction: 'It's not just a tool, it's a workflow'." | (b) |
| 106 | "'X, reimagined / redefined / made simple'. 'Welcome to …'." | (b) |
| 108-109 | "A headline that needs its colon: 'Ledgerly: invoicing for freelancers'. Drop the name, it is in the navigation." | **(a)**, concrete |
| 116-118 | "**Supporting line.** … It never repeats the headline in other words." | **(a)**, concrete; the closest rule to the user's complaint |
| 121-122 | "Read every sentence and ask whether a different company could print it unchanged; if so, sharpen it or cut it." | (b), judgement only |
| 124-125 | "One to three words. The primary action keeps one name everywhere on the page." | (a), but it asks *for* repetition (see conflicts) |
| 126-127 | "No arrow glyph on every link and button" | (a) repeated chrome, concrete |
| 158-159 | "Labels pretending to be editorial: 'Field notes', 'The details', 'Our philosophy' … A section heading can simply be 'Pricing'." | (b), lexicon |
| 160-162 | "Furniture text: … 'Scroll', '(Scroll to explore)', fake photo credits, status lines such as 'Available for work' with a dot, strings of words separated by middle dots or slashes." | **(a)** for the scroll cue and the dot status line; (b) for the rest |
| 163 | "The em dash as a habit. One or two on a page read as writing; ten read as a model." | (b) |
| 164 | "Explainer captions that narrate the design: 'A clean, minimal layout for…'." | **(a)**, concrete |
| 165 | "Step labels that duplicate numbers: 'Step 1', 'Phase 02'." | **(a)**, concrete |
| 166-167 | "Three parallel items everywhere … Use the number the content has." | (b) |
| 176 | "No all-caps for anything longer than three words" | (b) |

### references/tells.md

| Line | Rule | Class |
| --- | --- | --- |
| 33 | "Emoji as icons or bullets / banned" | (b) |
| 41 | "Eyebrow / kicker: tiny tracked uppercase label above every heading / banned / Fold it into the heading, or delete" | **(a)**, concrete |
| 42 | "Pill badge above the hero headline ('New', 'Beta', 'Introducing') / banned / Say it in the supporting sentence" | **(a)**, concrete |
| 48 | "Decorative status dot ('Available for work', pulsing 'live') / Remove" | **(a)** (dot highlights), concrete |
| 62 | "'Field notes', 'Quietly trusted by', 'On our desks': performative labels" | (b) |
| 75 | "Index numbering: 01 / 02 / 03, (001), 'Step 02', numbered section markers / banned" | (a) where it duplicates order; otherwise (c) |
| 78 | "Agency micro-meta: local time, coordinates, weather, version stamps, '©2026 — Scroll', middle-dot strings" | (b) |
| 79 | "Tiny uppercase letter-spaced captions scattered around the viewport / banned (3 or more)" | (a) as repeated chrome; (c) as styling |
| 99-101 | Copy tells flagged by lint: genre verbs, the correction, tricolon headlines, "Get started" and "Learn more", em dashes, lorem ipsum | (b) |

### SKILL.md house rules (lines 290-307)

- 296: "**No eyebrows.** No small label above a heading, no pill badge above the headline, no 'New ·' chip." Fix: "Put the information in the heading or the first sentence, or drop it." **(a)**, concrete.
- 298: "**No index numbering.** No 01 / 02 / 03, no (001), no 'Step 2'." (a)/(c).
- 303: "**No tiny uppercase letter-spaced labels.**" (c), but this is how eyebrows look.
- 306: "**No emoji**" (b).

### references/layout.md

- 106-107: "A repeated module is for archives only … Anywhere else, repetition reads as a kit." (c), layout repetition rather than copy.

### Existing detectors (useful for labelling and as a baseline)

- `scripts/lint.mjs:39` FAIL `eyebrow`: class-name regex `eyebrow|kicker|overline|pretitle|pre-title|suptitle|sup-title|tagline-label|section-label|section-tag`. Static, cheap, purely lexical on class names.
- `scripts/lint.mjs:37` FAIL `numbering`: text nodes `>01<`, `"0[1-9]"`, `Step|Phase|No.|Chapter 0?\d`.
- `scripts/lint.mjs:48` WARN `copy`: hype-word regex (unlock, seamless, elevate, … "not just X, it's Y", "more than a").
- `scripts/lint.mjs:49` WARN `cta-copy`: `Learn more|Get started|Click here|Read more|Discover more|Explore now|Submit` as whole element text.
- `scripts/lint.mjs:55` WARN `bounce`: `animate-(bounce|ping|pulse)` (the pulsing live dot).
- `scripts/lint.mjs:86` WARN `em-dash`: more than 4 em dashes across all files, comments excluded.
- `scripts/lib/probe.mjs:180-204` geometric eyebrow: a text block with font size at most min(18px, half the heading size), 2 to 44 characters, sitting above a heading within max(72px, 1.1 x heading size) capped at 140px, horizontally overlapping it, alone on its row, not inside nav, footer, button, a top header or a pinned layer. It also records whether the label sits in a pill (radius at least 10px, width under 420px, height under 56px, filled or bordered).
- `scripts/lib/rules.mjs:50-52` turns any eyebrow into a FAIL; `:62-63` FAILs 3 or more micro-labels (size at most 14px, uppercase, tracking at least 0.05em, not in nav, button or link).

### Numbers the skill reports (references/research.md)

All from the skill author's study, not checked here. Sample: 247 gallery sites; 235 usable for the panel; 202 measurable for typography (research.md:33-36). "Top" is panel score 7.5 or higher (82 measurable sites); "weak" is under 6.5 (23 measurable sites). The panel is model judgement from still frames, "not human jurors and not the real Awwwards scores" (research.md:30-31).

| research.md line | Device | Top | Weak |
| --- | --- | --- | --- |
| 137 | Pill badges (panel) | 14% | 34% |
| 138 | Pill-shaped label above a heading (measured) | 5% | 26% |
| 139 | Tiny uppercase letter-spaced labels, any (measured) | 16% | 57% |
| 140 | Same, three or more | 6% | 43% |
| 145 | Emoji in the page | 10% | 26% |
| 146 | Eyebrow labels (panel) | 32% | 45% |
| 162 | Index numbering (panel), more common on strong sites | 38% | 21% |

research.md:38-40: the geometric eyebrow detector over-counts, firing on 60% of the 202 measurable sites against 35% observed by the panel. research.md:234: across the same 202 sites, "eyebrow-like label" is the joint most frequent FAIL (60%, tied with headlines over two lines). Takeaway for HypaNet: geometry alone is a weak eyebrow classifier; whether a label is slop depends on its words against the heading's words, which is the part the slop model should add.

---

## 2. design-taste-frontend-v2-experimental (old single-file v2)

File: `~/.agents/skills/design-taste-frontend-v2-experimental/SKILL.md`. The most rule-dense source on labels and repetition, and the only one with count quotas.

| Line | Rule | Class |
| --- | --- | --- |
| 226 | CTA wrap ban: "3 words max for primary CTAs, ideally 1-2" | (b) |
| 227 | "**NO DUPLICATE CTA INTENT** … 'Get in touch' + 'Contact us' + 'Let's talk' … pick ONE label and use it everywhere on the page (nav, hero, footer)." | **(a)**, concrete |
| 236 | Hero subtext "max **20 words**" | (b) |
| 239-243 | Hero stack, max 4 text elements: "Eyebrow … OR brand strip OR neither - pick zero or one", headline, subtext, CTAs | **(a)**, concrete count |
| 244 | "BANNED in the hero: tiny tagline below CTAs …, trust micro-strip …, pricing teaser …" | (a) |
| 245 | "If you have an eyebrow AND a tagline below CTAs in the same hero, drop the tagline. … One small text element per hero, max." | **(a)**, concrete |
| 253 | "**EYEBROW RESTRAINT** … Every AI-built site puts an eyebrow above EVERY section header, producing the same templated rhythm." Signature: `text-[11px] uppercase tracking-[0.18em]` | **(a)** |
| 254-256 | "Maximum 1 eyebrow per 3 sections. Hero counts as 1." "If section A has an eyebrow, the next 2 sections cannot have one." Mechanical check: count > ceil(sectionCount / 3) fails. | **(a)**, concrete quota |
| 257 | "drop it entirely. The headline alone is enough. … the section's location on the page already categorizes it; no label needed." | **(a)** |
| 258 | "SPLIT-HEADER BAN … left big headline + right small explainer paragraph" | (a)/(c) |
| 281 | "**LOGO-ONLY rule** … Do NOT print industry / category labels below each logo" | **(a)**, concrete |
| 302 | "short headline (≤ 8 words) + short sub-paragraph (≤ 25 words)" | (b) |
| 321-326 | Copy self-audit: grammatically broken, unclear referents, "AI hallucination", "an LLM trying to sound thoughtful" | (b), judgement |
| 331 | "One copy register per page." | (b) |
| 339 | "use real typographic quotes … Not straight ASCII" | (b) |
| 657 | "NO version labels in the hero. `V0.6`, `v2.0`, `BETA`, `INVITE-ONLY PREVIEW` … banned as default eyebrows." | **(a)**, lexicon |
| 658 | "NO 'Brand · No. 01'-style sub-eyebrows." | **(a)** |
| 661 | "NO section-number eyebrows. `00 / INDEX`, `001 · Capabilities` … Eyebrows should name the topic in plain language, not enumerate." | **(a)**, concrete |
| 662 | "NO `01 / 4`-style pagination on images or bento tiles. If the user can count, they don't need the label." | **(a)**, concrete |
| 664 | "NO 'Index of Work, 2018 - 2026'-style range labels as eyebrows." | (a) |
| 667 | "The middle-dot (`·`) is rationed. Maximum 1 per line in metadata strips." | (b), concrete count |
| 668 | "NO decorative colored status dots on every list/nav/badge. … Acceptable only when the dot conveys actual semantic state … and is used sparingly." | **(a)** (dot highlights) |
| 682 | "NO 'From the field' / 'Field notes' / 'Currently on the bench' … poetic labels" | (b), lexicon |
| 685 | "**NO micro-meta-sentences under eyebrows.** Sentences like 'Each of these is a feature we ship today, not a roadmap promise. The list will stay short on purpose.' … are clutter." | **(a)**, the best match for "Resume recaptures the mouse." |
| 686 | "NO generic step labels. 'Stage 1 / Stage 2 / Stage 3' … The actual step content is the label." | **(a)**, concrete |
| 689 | "NO pills/labels/tags overlaid on images." | (a)/(c) |
| 690 | "NO photo-credit captions as decoration." | (b) |
| 695 | "NO decoration text strip at hero bottom. … `BRAND. MOTION. SPATIAL.`, `DESIGN · BUILD · SHIP`" | (a)/(b) |
| 696 | "NO floating top-right sub-text in section headings." | (c) |
| 703 | Locale, time and weather strips "banned for 99% of briefs" | (b) |
| 704 | "**Scroll cues are banned.** `Scroll`, `↓ scroll`, `Scroll to explore` … They know what scroll is." | **(a)**, concrete lexicon; explains the obvious |
| 705 | "**ZERO decorative status dots by default.** A coloured dot before nav items, before list rows, before badges, before status labels is a Tell. Only acceptable when conveying real semantic state … and limited to one per page section." | **(a)**, concrete quota |
| 709-723 | Em dash "COMPLETELY banned", en dash too | (b) |
| 955, 958, 973, 980, 981 | Pre-flight checklist repeats 254-256, 227, 685, 661, 705 | (a) |

Visual-only items in the same file (glass, purple, bento cell counts, zigzag caps at 251-262, 617-650) are (c).

---

## 3. unslop

File: `~/.agents/skills/unslop/SKILL.md`. Prose-oriented, not UI. Rule numbers are stable ids (line 13).

| Line | Rule | Class |
| --- | --- | --- |
| 17 | 3. "Superficial -ing phrases. 'highlighting...', 'ensuring...'" | (b) |
| 22 | 7. AI vocabulary list (delve, crucial, pivotal, showcase, testament …) | (b), lexicon |
| 23 | 8. "'serves as', 'stands as', 'boasts'. Say 'is' or 'has'." | (b) |
| 24 | 9. "'Not just X, but Y.'" | (b) |
| 25 | 10. "Rule of three." | (b) |
| 26 | 11. "**Synonym cycling.** Protagonist, main character, central figure, hero all in one paragraph. Pick one, repeat it." | (a), asks *for* lexical repetition (see conflicts) |
| 31 | 13. "Avoid em dashes entirely." | (b) |
| 32 | 14. Colon as mid-sentence connector | (b) |
| 34 | 16. "**Inline-header lists.** The tell is a bold label and colon that restates the line: '**Performance:** Performance improved...'. … A bold lead-in that … is followed by genuinely new detail … is fine" | **(a)**, concrete, with a stated negative case |
| 35 | 17. "Title case headings. Use sentence case." | (b) |
| 36 | 18. "Decorative emojis." | (b) |
| 37 | 19. "Curly quotes. Use straight quotes." | (b) |
| 41 | 20. Chatbot phrases | (b) |
| 46 | 23. Filler phrases ("In order to", "It is important to note that") | (b), but close to (a): words that add nothing |
| 48 | 25. Generic conclusions | (b) |
| 56 | 27. "if the sentence could appear unchanged in another project's docs, it says nothing about this one. Cut it." | (b), judgement |

---

## 4. technical-writing

File: `~/.agents/skills/technical-writing/SKILL.md`. Docs, not UI, and it says so at line 95.

| Line | Rule | Class |
| --- | --- | --- |
| 13 | "**Cut every word that does no work.**" | (a), principle |
| 59 | "don't start consecutive sentences with the same phrase." | **(a)**, concrete |
| 61 | "Headings carry the point, not just the topic" | (b) |
| 72 | "Pick one word per action and stick to it" | (a), pro-repetition |
| 85 | "Replace an em dash with a new sentence." | (b) |
| 88 | "Call each thing by one name, everywhere." | (a), pro-repetition |
| 93 | Delegates the slop catalogue to `unslop` | n/a |
| 95 | "Product UI strings are not documentation." | scope note |

---

## 5. frontend-design (official plugin, v1.1.1)

File: `~/.claude/plugins/cache/claude-plugins-official/frontend-design/1.1.1/skills/frontend-design/SKILL.md` (same text in the marketplace copy).

| Line | Rule | Class |
| --- | --- | --- |
| 26 | "Accenting just a single word or phrase in a headline" | (c) |
| 27 | "Using all caps for labels." | (b) |
| 28 | "Adding unnecessary typographic labels above content." | **(a)** |
| 30 | "Structural devices like outlines, borders, numbering, eyebrows, dividers, labels, etc., encode useful information about the content rather than decorate it. … check the content really is a sequence." | **(a)**, principle; allows labels that inform |
| 43 | "template chrome that appears whatever the subject: a tracked-out ALL-CAPS eyebrow label above every heading; meta strings joined with middle dots ('A · B · C'); labels built as 'WORD — fragment' with a spaced em dash; … a '→' appended to link and button text." | **(a)**, concrete patterns |
| 63 | "Words appear in a design for one reason: to make it easier to understand and use." | (a), principle |
| 65 | "Name things by what users will understand … not by how the system is built." | (b) |
| 67 | "An action keeps the same name through the whole flow, so the button that says 'Publish' produces a toast that says 'Published.'" | (a), pro-repetition |
| 69 | "Errors don't apologize, and they are never vague" | (b) |
| 71 | "plain verbs, sentence case, no filler … Let each written element do exactly one job." | **(a)**, the cleanest one-line definition of the target |

---

## 6. stitch-design-taste

Files: `~/.agents/skills/stitch-design-taste/SKILL.md`, `DESIGN.md`.

| File:line | Rule | Class |
| --- | --- | --- |
| SKILL.md:59, DESIGN.md:67 | "**No Filler Text:** 'Scroll to explore', 'Swipe down' … and any instructional UI chrome are BANNED. The user knows how to scroll." | **(a)** |
| DESIGN.md:69 | "**CTA Restraint:** Maximum one primary CTA button. No secondary 'Learn more' links. No redundant micro-copy below the headline" | **(a)**, names the exact pattern |
| SKILL.md:111, DESIGN.md:113 | "No filler UI text: 'Scroll to explore', 'Swipe down', 'Discover more below'" | (a), lexicon |
| DESIGN.md:57, SKILL.md:67 | "Helper text optional" | neutral; see conflicts |
| SKILL.md:98, DESIGN.md:102 | No emojis | (b) |
| SKILL.md:108-110, DESIGN.md:114-116 | No generic names; no "Elevate", "Seamless", "Unleash", "Next-Gen" | (b) |
| DESIGN.md:95 | "Pulse on status dots" | (c), conflicts with the dot bans |

---

## 7. gpt-taste

File: `~/.agents/skills/gpt-taste/SKILL.md`.

| Line | Rule | Class |
| --- | --- | --- |
| 7 | LLMs "use cheap meta-labels ('QUESTION 05', 'SECTION 01')" | (a) |
| 11 | "DO NOT USE EMOJIS IN YOUR CODE, COMMENTS, OR OUTPUT." | (b) |
| 40 | "Do NOT use arbitrary floating stamp/badge icons on the text. Do NOT use pill-tags under the hero." | **(a)** |
| 63 | "**The Meta-Label Ban:** BANNED FOREVER are labels like 'SECTION 01', 'SECTION 04', 'QUESTION 05', 'ABOUT US'." | **(a)**, concrete lexicon |
| 74 | "Label Sweep … Confirm no cheap meta-labels" | (a) |
| 23 | Navigation as "floating glass pill" | (c), conflicts with v2 |

---

## 8. design-taste-frontend-v1

File: `~/.agents/skills/design-taste-frontend-v1/SKILL.md`.

| Line | Rule | Class |
| --- | --- | --- |
| 26 | "NEVER use emojis in code, markup, text content, or alt text." | (b) |
| 64 | "Label MUST sit above input. Helper text is optional but should exist in markup." | (b), pro-helper-text |
| 70 | Perpetual micro-animations on "status dots" | (c), conflicts |
| 117-121 | No "John Doe", fake round numbers, "Acme", no "Elevate", "Seamless", "Unleash", "Next-Gen" | (b) |
| 214 | "'breathing' status indicators … a pop-up notification badge" | (c), conflicts |

---

## 9. redesign-existing-projects

File: `~/.agents/skills/redesign-existing-projects/SKILL.md`.

| Line | Rule | Class |
| --- | --- | --- |
| 84 | "Never use 'Elevate', 'Seamless', 'Unleash', 'Next-Gen', 'Game-changer', 'Delve', 'Tapestry', or 'In the world of...'" | (b) |
| 85 | "Exclamation marks in success messages. Remove them." | (b) |
| 86 | "'Oops!' error messages. Be direct" | (b) |
| 90-91 | Lorem ipsum; title case on every header | (b) |
| 97 | "Pill-shaped 'New' and 'Beta' badges. Try square badges, flags, or plain text labels." | (a), but restyles rather than removes |
| 99 | "3-card carousel testimonials with dots" | (c) |
| 122 | "Commented-out dead code. Remove all debug artifacts before shipping." | code comments, see section 13 |

---

## 10. minimalist-ui, high-end-visual-design, industrial-brutalist-ui

| File:line | Rule | Class |
| --- | --- | --- |
| minimalist-ui/SKILL.md:20 | No emojis | (b) |
| minimalist-ui/SKILL.md:21 | No "John Doe", "Acme Corp", "Lorem Ipsum" | (b) |
| minimalist-ui/SKILL.md:22 | No "Elevate", "Seamless", "Unleash", "Next-Gen", "Game-changer", "Delve" | (b) |
| minimalist-ui/SKILL.md:52-53 | "Tags & Status Badges: Pill-shaped …, very small typography (`text-xs`), uppercase with wide tracking" | (c), but *prescribes* the pill that v2 bans |
| high-end-visual-design/SKILL.md:52 | "**Eyebrow Tags:** Precede major H1/H2s with a microscopic, pill-shaped badge (`rounded-full px-3 py-1 text-[10px] uppercase tracking-[0.2em]`)." | (a), **mandates** the eyebrow |
| industrial-brutalist-ui/SKILL.md:42 | "Exclusively uppercase. Used for all metadata, navigation, unit IDs, and coordinates." | (b)/(c) |
| industrial-brutalist-ui/SKILL.md:80 | "randomized string data (e.g., `REV 2.6`, `UNIT / D-01`) to simulate active mechanical processes" | (a), **mandates** decorative micro-meta |

---

## 11. image-to-code, imagegen-frontend-web, imagegen-frontend-mobile, brandkit

These generate images first, so their copy rules are brief, but several name the label clutter directly.

| File:line | Rule | Class |
| --- | --- | --- |
| image-to-code/SKILL.md:34 | Defaults include "tiny pills, labels, tags, system markers, and fake interface jargon" | (a) |
| image-to-code/SKILL.md:612-630 | "Do not clutter the design with tiny UI extras": "unnecessary pills", "pseudo-system markers", "meaningless small metadata rows", "filler chips", "tiny badges everywhere", "'00 orchestration layer'", "tiny technical status pills", "filler operator/control-room labels that exist only to look complex" | **(a)** |
| image-to-code/SKILL.md:921 | "giant heading + weak tiny subcopy" | (c)/(a); this is the shape of the pause menu |
| image-to-code/SKILL.md:927-934 | unleash, elevate, revolutionize, next-gen, seamless, "transformative platform" | (b) |
| image-to-code/SKILL.md:943-947 | "pseudo-enterprise control labels", "decorative system markers", "filler status microcopy" | **(a)** |
| imagegen-frontend-web/SKILL.md:541 | "giant heading + weak tiny subcopy" | (c)/(a) |
| imagegen-frontend-web/SKILL.md:547-556 | Copy-vibe bans incl. "powerful solution" | (b) |
| imagegen-frontend-web/SKILL.md:577 | "auto-play-style hero dots with no semantic purpose" | (c) |
| imagegen-frontend-web/SKILL.md:580 | "three identical stat columns (99% satisfaction, $10 saved, ∞ scale)" | (b) |
| imagegen-frontend-mobile/SKILL.md:735-741 | "elevate your life", "unlock your potential", "smarter than ever", "transform your day" | (b) |
| imagegen-frontend-mobile/SKILL.md:750-753 | "too many pills", "too many badges", "too many tiny labels", "fake system markers" | **(a)** |
| imagegen-frontend-mobile/SKILL.md:1074-1079 | "lorem ipsum overload", "fake inspirational filler", "overloaded onboarding explanations", "overly technical filler labels" | (a)/(b) |
| brandkit/SKILL.md:619-636 | "Taglines should be short and specific." Good examples include "Your network. Our watch.", "Nothing random." Avoid "buzzword soup" | (b); the examples clash with v2's slogan bans |

---

## 12. CLAUDE.md files

- `~/.claude/plugins/cache/claude-plugins-official/mattpocock-skills/1.3.1/CLAUDE.md:25`: "No em-dashes anywhere in this repo's prose (… code comments). … never do a blind character substitution." (b). Applies only to that repo.
- No `~/.claude/CLAUDE.md` exists.

---

## 13. Code comments: no-comments and related

### no-comments

Files: `~/.agents/skills/no-comments/SKILL.md` (24 lines), `references/comment-sicko.md` (32 lines). `disable-model-invocation: true`, so it only runs as `/no-comments`.

What it is: a process, not a rule list. SKILL.md:19 spawns a "Comment Sicko" subagent; SKILL.md:20-23 have the parent audit its report, run `/how` or `/why` on doubtful keeps, fix root causes, and offer to encode constraint comments as a type, test or lint before deleting them. Nothing in it is deterministic; every keep or kill can depend on an LLM reading nearby code.

Rules on redundant comments (comment-sicko.md):

- :12 kill list: "Narration, banners, commented-out corpses, workaround sermons." Narration (a comment restating what the next line does) is the code-comment analogue of the UI subtitle restating the heading: **(a)**. Banners and commented-out code are concrete and lexically detectable.
- :14-20 the only keeps:
  - "Legal or license headers."
  - "Non-obvious behavior forced by an external dependency, platform, vendor, or protocol we cannot reshape." Surprises in our own code are killed and the symbol marked `MUST KILL` for a rename or refactor.
  - "`// prettier-ignore`. Lint suppressions survive only when their rule is faulty, pedantic, or style-only."
  - "Doc comments that define a public API contract."
  - "Issue or RFC links that explain a constraint code cannot express."
- :22 "When I am not sure a keep clause applies, the comment dies." Recall over precision.
- :24 `eslint-disable`, `@ts-ignore`, `@ts-expect-error` "stink"; kill if the rule protects correctness.
- :26 "`IMPORTANT`, `do not remove`, `too risky`, `fine for now`, and long justifications are scent, not conviction."
- :28 "A long justification without a proven keep-list exception is a confession."

What transfers to a deterministic linter:

- The keep list is a good **negative-label set**: licence headers, `prettier-ignore` and similar pragmas, doc comments on exported symbols, comments that are mostly an issue or RFC URL. These can be recognised without reasoning and should never be flagged.
- Commented-out code and banner comments are concrete positives.
- "Narration" is the in-scope semantic class: a comment whose content words mostly reappear in the identifiers of the next statement (`// increment counter` above `counter++`). That is a lexical-overlap test, the same shape as subtitle-restates-heading.
- The "when unsure, kill" posture is wrong for a linter that people must trust. A lint rule should flag only high-confidence narration and leave "workaround sermons" (which need code understanding) out of scope.

### Related comment rules elsewhere

- `~/.agents/skills/poteto-mode/SKILL.md:131`: "Keep a comment only for a non-obvious *why* the code can't show. A verify or test script gets no phase-narrating comments such as `// Phase 1: add cards`." (a), concrete for the "Phase N" / "Step N" pattern.
- `~/.agents/skills/principle-fix-root-causes/SKILL.md:20`: "If a workaround needs a paragraph-long comment to justify it, the code is wrong." Judgement.
- `~/.agents/skills/redesign-existing-projects/SKILL.md:122`: commented-out dead code.
- `~/.agents/skills/gpt-taste/SKILL.md:11`: no emojis in comments. (b)

---

## 14. In-scope rules concrete enough to become labelling rules

Each candidate names its sources. "Concrete" means two annotators applying the wording would mostly agree without discussing it.

| Id | Labelling rule | Sources | Pause-menu example |
| --- | --- | --- | --- |
| R1 | **Eyebrow above heading.** A short label (2 to 44 characters, smaller than half the heading size) directly above a heading. Slop when its words are a paraphrase, a category or a restatement of the heading, or generic ("SELECTED WORK", "OVERVIEW"). Not slop when it carries information absent from the heading (a date, a real section name in a long document). | tells.md:41; SKILL.md:296; v2-exp:253-257; frontend-design:28, 43; high-end-visual-design:52 (the opposite view); geometry from probe.mjs:184-203 | "OPERATION ON HOLD" above "PAUSED": positive (paraphrase) |
| R2 | **Subtitle restates heading.** A supporting line under a heading that says the heading again in other words. | copy.md:116-118; stitch DESIGN.md:69 | not this example |
| R3 | **Explainer narrates the interface.** Copy that tells the user what a visible control does or how to use the page when the control's own label already says it ("Scroll to explore", "Click below to get started", "Resume recaptures the mouse."). | copy.md:160-162, 164; v2-exp:685, 704; stitch SKILL.md:59, DESIGN.md:67, 69, 113 | "Resume recaptures the mouse.": positive. Borderline: it is a fact about pointer capture, but it explains the Resume button below it |
| R4 | **Inline label restates its line.** A bold or prefixed label whose words reappear in the text it labels ("**Performance:** Performance improved…"). Negative case stated: a lead-in followed by new detail. | unslop:34 | n/a |
| R5 | **Step or index label duplicates order.** "Step 1", "Phase 02", "01 / 4", "SECTION 01", "QUESTION 05" where position already shows order. | copy.md:165; tells.md:75; v2-exp:661, 662, 686; gpt-taste:63; poteto-mode:131 (comments) | n/a |
| R6 | **Meta-label names the section's own type.** "ABOUT US", "FEATURES", "OVERVIEW" over a section whose heading or content already says what it is. | gpt-taste:63; v2-exp:257 ("the section's location on the page already categorizes it") | n/a |
| R7 | **Decorative dot or pill label.** "● Live", "● New", "Beta", "Available for work", "v0.6", "INVITE-ONLY" with no backing state. Quota from v2-exp:705: at most one per section, and only for real state. | tells.md:42, 48; copy.md:161; v2-exp:657, 668, 705; gpt-taste:40; image-to-code:612-630; imagegen-frontend-mobile:750-753 | n/a (the menu has none) |
| R8 | **Repeated chrome across the UI.** The same label device (eyebrow, kicker, dot pill, arrow glyph) on most sections. Countable: eyebrows > ceil(sections / 3); arrow on every link; dot before every row. | v2-exp:254-256, 668; frontend-design:43; copy.md:126-127; tells.md:79 (3 or more micro-labels) | n/a (one screen) |
| R9 | **Duplicate intent, different words.** Several CTAs or labels for the same action phrased differently ("Get in touch", "Contact us", "Let's talk"). | v2-exp:227; copy.md:125; frontend-design:67 | n/a |
| R10 | **Name repeated from navigation.** Headline of the form "Brand: description" when the brand is in the nav. | copy.md:108-109 | n/a |
| R11 | **Hero small-text count.** More than one small text element (eyebrow, tagline, trust strip, pricing teaser) around the main heading. | v2-exp:239-245 | Eyebrow plus subtitle around "PAUSED" is two: positive by this count |
| R12 | **Logo or item with category caption.** Category word under each logo ("Stripe / payments"). | v2-exp:281 | n/a |
| R13 | **Consecutive sentences open the same way.** | technical-writing:59 | n/a |
| R14 | **Narrating code comment.** Comment whose content words mostly appear in the next statement; banners; commented-out code; "Phase N" comments. Never flag: licence headers, pragmas, doc comments on exported symbols, issue/RFC links. | comment-sicko.md:12, 16-20; poteto-mode:131; redesign-existing-projects:122 | n/a |

Not concrete enough to label directly (judgement calls; useful as a guiding principle in the spec, not as a rule): copy.md:35 (section with no claim), copy.md:121-122 and unslop:56 (could another company print it), frontend-design:71 ("each written element do exactly one job"), v2-exp:321-326 (self-audit), technical-writing:13 (cut every word that does no work), principle-fix-root-causes:20.

Gap: no local skill directly covers the case where an eyebrow and a heading are **synonyms of each other** ("OPERATION ON HOLD" / "PAUSED"). v2 bans all eyebrows; v2-exp rations them by count; frontend-design keeps them if they inform. None tests the eyebrow's words against the heading's words. Likewise, no rule covers helper text that explains a button sitting directly below it; R3 is pieced together from scroll-cue and micro-meta rules. These two gaps are where a learned or hand-written semantic model adds something the existing regex and geometry checks cannot.

---

## 15. Conflicts between rules

1. **Eyebrows: ban, ration, require or allow.** v2 bans every eyebrow (SKILL.md:296; tells.md:41; rules.mjs:50 FAIL). v2-exp allows one per three sections and zero or one in the hero (253-256, 240). high-end-visual-design:52 requires a pill eyebrow above every major H1/H2. frontend-design:30 allows labels that "encode useful information". v2-exp:661 says eyebrows "should name the topic in plain language", so it assumes some exist. A labelling spec has to pick one: the defensible choice is frontend-design's test (informative or not), with v2-exp's quota as the repetition rule.
2. **Lexical repetition is required, not slop.** unslop:26 ("Pick one, repeat it"), technical-writing:72, 88, frontend-design:67 ("Publish" / "Published") and copy.md:125 / v2-exp:227 (one label per action everywhere) all want the same word reused. So the target is not repeated words. It is the same content restated in different words (copy.md:118), or the same decorative device stamped everywhere (v2-exp:253). A detector that fires on lexical overlap alone would punish consistent naming; the labelling rules need to separate "same name for the same thing" (good) from "second element adds nothing" (slop).
3. **Status dots.** Removed or zero by default in v2 tells.md:48 and v2-exp:668, 705. Animated by default in v1:70, v1:214 and stitch DESIGN.md:95 ("Pulse on status dots"). v2-exp allows one per section for real state.
4. **Pill badges.** Banned in v2 tells.md:42 and gpt-taste:40. Restyled, not removed, in redesign-existing-projects:97. Prescribed in minimalist-ui:52-53 for "Tags & Status Badges".
5. **Decorative micro-meta.** Banned in v2 tells.md:78, v2-exp:658, 695 and image-to-code:612-630. Required in industrial-brutalist-ui:80 ("randomized string data … `REV 2.6`, `UNIT / D-01`").
6. **Index numbering.** Total ban in v2 (SKILL.md:298) although research.md:162 shows it more common on strong sites (38% against 21%, panel). frontend-design:30 allows it for real sequences.
7. **Helper text.** v1:64 wants it "in markup"; stitch DESIGN.md:57 calls it optional; stitch DESIGN.md:69 bans "redundant micro-copy below the headline". Form helper text that states a constraint ("8 characters minimum") is informative and must be a negative example.
8. **Em dashes.** v2 allows one or two per page (copy.md:163; lint threshold more than 4 at lint.mjs:86). v2-exp:709-723, unslop:31 and the mattpocock CLAUDE.md:25 ban them outright; technical-writing:85 replaces them with a sentence. (b), but any shared spec has to choose.
9. **Quote marks.** unslop:37 wants straight quotes; copy.md:171 and v2-exp:339 want curly ones.
10. **Case.** unslop:35 and redesign-existing-projects:91 want sentence case; frontend-design:27 bans all-caps labels; industrial-brutalist-ui:33, 42 want uppercase throughout. Game UIs (the motivating example) use all-caps buttons by convention, so case should not count towards the slop label.
11. **Taglines.** brandkit:619-636 offers "Your network. Our watch." as a good tagline; copy.md:103-106 and unslop:25 treat that slogan shape as a tell.
12. **Comments.** comment-sicko.md:19 keeps doc comments that define a public API; poteto-mode:131 keeps only a "non-obvious why", which would remove doc comments that restate the signature. comment-sicko.md:22 kills when unsure; a linter should do the opposite.

---

## 16. What this means for the slop spec

- The local skills already supply cheap lexical and geometric detectors (lint.mjs, probe.mjs). They cannot tell an informative label from a redundant one; the v2 author's own data puts the geometric eyebrow detector at 60% of 202 sites against 35% by panel judgement (research.md:38-40).
- The in-scope class splits into three detectable shapes: (i) one element restating a neighbour (R1, R2, R3, R4, R10, R14), (ii) a label that duplicates something structural such as order or section type (R5, R6, R12), and (iii) the same device repeated across the UI (R7, R8, R9, R11). Shape (i) needs a comparison between two strings and is where a small hand-written model by Fable would earn its place; shapes (ii) and (iii) are mostly counting and lexicons.
- Lexical repetition of a name is a negative example, by four independent skills (conflict 2).
- None of these sources cover app or game screens directly, and two say so. The pause menu is the first test case outside their scope.
