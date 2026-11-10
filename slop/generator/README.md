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
- `sources.jsonl`: the generated files (path, kind, framework, styling idiom, clean or not, source text and any CSS
  module).
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
   it goes. A planted slot is written as a restatement, narration or decoration of a named node and points at it;
   every other slot comes from tables of copy that adds information (`tables.py`: 27 invented products, ten games,
   heading concepts with paraphrases, settings, notices). K-rule hard negatives are `KEEP` records that name the rule
   in `pattern`. Copies of one fact form a stack and list each other in `equiv`.
3. A renderer writes the source: TSX in Tailwind, CSS-module, shadcn-like or inline-style idioms (with `.map()` over
   literal arrays, conditionals and interpolations), HTML documents or partials (BEM classes with a style block,
   Tailwind, inline styles), or a SwiftUI view (stacks, `Form`, `List`, `ForEach` over literal arrays, alerts as
   modifiers, `.navigationTitle`). The writer records where each slot's visible characters land.
4. `extract_batch.ts` runs the real extractor (`extract/src/extract.ts`) over the files.
5. `join.py` matches slots to nodes by span. Any planted span that is not a node, any node that came from no slot
   and any span used twice stops the build with an error. A screen whose extracted role contradicts its planted
   pattern (an R1 node that is not an eyebrow, say), whose pointer runs against the precedence of `LABELLING.md` §5,
   or whose pointer crosses a window boundary is regenerated with the next attempt number; the manifest counts these
   under `regenerated`.

R18 (comment restating code) and K10 (comments) belong to track C, which is deferred, so they are not planted; the
extractor emits no comment nodes in r1.

## Tests

`uv run pytest generator/tests` (also part of `uv run pytest` in `slop/`): two builds of the same seed are
byte-identical, namespaces differ, every node is labelled and `labeller/check.py` finds no errors, every plantable
pattern and K rule occurs, a moved span is reported, `test_gen` refuses without a frozen registry, and the duplicate
check matches near copies.
