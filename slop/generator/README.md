# Dev generator

The harness's synthetic generator for the slop experiment (`SPEC.md` §9 "Dev generator", §12.4). It writes realistic
UI source files with planted redundant copy, runs the real extractor over them and joins the planted labels to the
extracted nodes by source span. It builds `dev` and `val`, and later `test_gen`. Nothing here enters the author kit:
the tables, templates and synonym lists stay in this directory.

## Running it

```
uv run python -m generator.build dev        # 20,000 screens, seed 20261112             -> datasets/dev/
uv run python -m generator.build val        #  2,000 screens, seed 20261112 (val)       -> datasets/val/
uv run python -m generator.build test_gen   #  2,000 screens, seed 20261113             -> datasets/test_gen/
uv run python -m generator.build dev --n 500 --out DIR   # a smaller build elsewhere
```

Run from `slop/`, with Node 22.18 or later and the pinned packages installed (`npm ci`). A set directory holds:

- `screens.jsonl`: one `slop.screen/1` screen per line, exactly as the extractor wrote it.
- `<set>.gen.jsonl`: the label records (`SPEC.md` §5), with labeller `gen`, the current guide's sha256 and a fixed
  time (`2026-11-12T00:00:00Z`). Every labelled node gets a record, `KEEP` unless planted; a screen with a repeated
  device gets a screen-level R19 aside; every screen ends with a done record. `labeller/check.py` reads the file
  with `--screens` set to the same directory.
- `sources.jsonl`: the generated files (path, framework, styling idiom, source text and any CSS module).
- `<set>.meta.jsonl`: harness only, one line per screen: its kind, whether it is clean, and its attempt number. Nothing
  shipped to a model (the author kit, the disclosed screens, a trained baseline's input) may read this file; the
  screen and source files carry neither field, and source paths (`<set>/<index>/<folder>/<Name>.<ext>`) name no kind.
- `manifest.json` and `SHA256SUMS`: seeds, the hashes of the generator, extractor and guide, the data files' hashes,
  and counts per class, pattern, K rule, aside, kind, framework and idiom.

The data files are gitignored; the manifest and the sums are not. The same set, seed and generator give the same
bytes.

`test_gen` refuses to build unless `slop/registry.yaml` exists and its `generator.sha256` equals the hash of this
directory's code and tables (`build.generator_sha256()`: every `.py` and `.ts` file here, tests excluded), and
unless `dev` and `val` are built. Screens whose normalised text, or whose node set at Jaccard 0.8 or more, matches a
`dev` or `val` screen are regenerated and counted.

## How a screen is made

1. A seeded random stream for (seed, namespace, index, attempt) picks the screen kind (marketing sections, app
   pages, settings and forms, dialogs, empty states, dashboards, game menus and HUDs), the framework (TSX mostly,
   HTML, SwiftUI) and whether the screen is clean (about 15%: nothing redundant planted, hard negatives kept).
2. A composer (`compose.py`) builds an abstract tree of elements whose copy sits in slots, and labels each slot as
   it goes. Each screen is about one topic: a product from `tables.py` (27 invented domains, ten games) with its two
   nouns, their synonyms, its people and its group (`lexicon.py`). A planted slot is written as a restatement,
   narration or decoration of a named node and points at it; every other slot adds a fact about what its heading or
   label names. K-rule hard negatives are `KEEP` records that name the rule in `pattern`. Copies of one fact form a
   stack and list each other in `equiv`.
3. A renderer writes the source: TSX in Tailwind, CSS-module, shadcn-like or inline-style idioms (with `.map()` over
   literal arrays, conditionals and interpolations), HTML documents or partials (BEM classes with a style block,
   Tailwind, inline styles), or a SwiftUI view (stacks, `Form`, `List`, `ForEach` over literal arrays, alerts as
   modifiers, `.navigationTitle`). The writer records where each slot's visible characters land.
4. `extract_batch.ts` runs the real extractor (`extract/src/extract.ts`) over the files.
5. `join.py` matches slots to nodes by span. Any planted span that is not a node, any node that came from no slot
   and any span used twice stops the build with an error. A screen whose extracted role contradicts its planted
   pattern (an R1 node that is not an eyebrow, say), whose pointer runs against the precedence of `LABELLING.md` §5,
   or whose pointer crosses a window boundary is regenerated with the next attempt number; so is a screen whose copy
   has a grammar seam (`core.seams`: "a" or "an" before the wrong sound, a doubled word, a bad plural such as
   "branchs"). The manifest counts these under `regenerated`.

R18 (comment restating code) and K10 (comments) belong to track C, which is deferred, so they are not planted; the
extractor emits no comment nodes in r1.

## Keeping the task, not the generator, learnable (v2)

A blind check of `val` v1 found surface features that decided the label. v2 removes them:

- **Role.** Every role the guide lets be redundant carries planted positives and hard negatives: h1 and h2 (a page
  title repeated by an app bar, a navigation title or a section heading, R9), h3 and h4 (card titles repeating their
  section or card, R9), `empty_title` and `dialog_title` (R9), `list_item` (a bullet that only names its card, R4; a
  closing bullet, R17), `label` (a group label that only names what follows, R2; a figure label restating its card,
  R5; a repeated figure, R14), `stat` (R14), `link` (a repeated call to action, R14) and `error` (an alert body
  restating its title, R4). `cell`, `tab`, `nav`, `menu_item` and `breadcrumb` stay `KEEP`: repeated values there are
  K5, and a tab or crumb naming the page is K11. Eyebrows that start with "Your" are field-style kickers (K1) as often
  as restatements.
- **Shared stems.** Restatements come with the node's own words and without them (synonyms per noun, field label
  paraphrases, verb synonyms for confirmations); informative copy often repeats the heading's noun (a placeholder that
  gives an example after the label is `KEEP`, `partial`; helpers on scope, limit or privacy name their field;
  consequences name the action; category eyebrows may sit over headings that name the category, K1).
- **Coherence.** Settings pages hold the sections, toggles and selects their title names; dialogs that rename take a
  name; empty states name the page's noun; pricing, quotes and FAQs use the product's people, group and nouns.
- **String pool.** Copy is assembled from slots (nouns, synonyms, figures, dates, times, people, places), status and
  genre words are used on both sides, and figures in tables and stats take fresh digits.
- **Widgets.** Tables have 2 to 7 columns, dashboards vary their range tabs and draw a chart (no text) beside a
  legend of real series, icon buttons draw the icon their tooltip names, and badges, notices and update lines vary.

## Context, not wording, decides the class (v3)

A blind check of `val` v2 found single features that were one class, grammar seams, fixed widgets and a classifier
that reached REDUNDANT F1 0.86 from a node's own text, role and flags. v3 changes:

- **Grammar.** Articles and plurals come from `core.article` and `core.plural` ("an order", "travelling parties",
  "heads of department"); templates that could double a word are guarded, and the build regenerates any screen that
  still has a seam.
- **The same sentence on both sides.** Facts (`lexicon.FACTS`: a limit, a price, a time, a window) come with a
  heading that states the figure, a topic heading that does not, lines that only repeat the figure and lines that add
  another fact. Under the figure heading a repeating line is R3, R4 or R5 (with digits, commas and semicolons); under
  the topic heading the same line is KEEP. Fields and toggles work the same way: "Bio (up to 160 characters)" makes
  "No more than 160 characters." R5, plain "Bio" makes it K4; "No invoices since 3 May" makes a dated body R4, "No
  invoices" makes it KEEP. Restatement frames ("Manage your …", "There are no …", "Are you sure …") also open KEEP
  partials that add a fact, and facts open with the same verbs ("Browse …", "Keep track of …").
- **Single features with both classes.** Semicolons, mono upper-case meta lines (versions, builds, issues and dates
  are KEEP), conditional empty-state bodies (R4 planted), placeholders with "e.g." (R8 "e.g. your email address"; and
  bare examples without "e.g."), dialog bodies ending in "?" (a consequence and a question, K7), group labels (an item
  name or a date, K1, beside the R2 ones), card titles inside grids (R9) and glyphs on informative badges (K2, aside
  R11g).
- **Tone.** Short KEEP lines that add one fact without a figure, drawn from pools that fit where they sit; KEEP copy
  that uses the words restatements use ("total", "want", "sure", "confirm", "together", "below", "that"); restatements
  that do not address the reader; more templates for each KEEP line that used to recur.
- **Widgets.** Footer columns draw their headings and links from pools; legal lines carry random company, VAT and
  charity numbers.
- **Coherence.** Plan cards take seats, sites and limits from one ladder, so a card's subtitle and its list agree; a
  spotlight's subtitle is about its feature; facts are only used by products they suit; counts in dialogs are three
  or more.
- **Guide decisions** (`LABELLING.md` §8): one "Most popular" pill on one plan card is K2 judgement; a genre-variant
  eyebrow ("Word of mouth") over a heading that neither shares its stem nor is a genre word is DECOR R2; "Switch X on
  or off." under toggle X is NARRATE R6 with `ct`; a near-synonym h2 heading the page's only section is R9 judgement
  (a second section is added otherwise); a heading plus a qualifier ("Shared budgets: new in 4.2", "Hives this
  week" under "Hives", a tooltip "New hive (N)") is KEEP with `partial`.

An exact repeat at equal rank flags the later node in reading order; when the composer cannot know that order (a
SwiftUI navigation title is read first), `build.orient_duplicates` swaps the label onto the later copy.

## Tests

`uv run pytest generator/tests` (also part of `uv run pytest` in `slop/`): two builds of the same seed are
byte-identical, namespaces differ, every node is labelled and `labeller/check.py` finds no errors, every plantable
pattern and K rule occurs, the screen and source files name no kind and carry no clean flag, sixteen roles carry both
classes, an equal-rank repeat flags the later node, a moved span is reported, `test_gen` refuses without a frozen
registry, the duplicate check matches near copies, `core.article`, `core.plural` and `core.seams` handle their hard
cases, and no node of the built screens or of `datasets/dev` (when built) has a grammar seam.
