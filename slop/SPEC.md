# HypaNet Slop: a hand-written detector for redundant UI copy

Fable 5.2 (`claude-fable-5-2`) writes, by reasoning alone, every weight of a small transformer that flags redundant
copy in user interfaces: an eyebrow that restates its heading, a subtitle that explains the button below it, dot
pills and badges on static content, the same fact repeated across sections. The network has to run inside a linter
(a CLI and an ESLint rule): deterministic, on CPU, no LLM at lint time. Nobody trains it. The question is the
HypaNet one: how written weights compare with trained ones and with rules, on a task that needs lexical knowledge
(paraphrase between short strings) that Fable must write into the embedding table itself.

Motivating example (a game pause menu built by an AI): eyebrow "OPERATION ON HOLD", heading "PAUSED", subtitle
"Resume recaptures the mouse.", buttons RESUME / SETTINGS / CONTROLS / RESTART MISSION. The user underlined the
first three, and widened the target to "repetition within the UI: headings, labels, dot highlights".

Decisions (Matthew, 2026-11-10): go ahead; author Fable 5.2, not 5.1; scope includes repetition across the UI as
well as explanatory copy; public anti-slop guidance informs the definition (`LABELLING.md`); Matthew adjudicates
and hand-labels; Matthew approves the final prompt, the cap and the effort level before any launch, smoke launch
included; no money is spent before that. Review of this spec: `research/review-log.md`.
Further decisions (Matthew, 2026-11-10, after review): full labelling plan (§5, §9); the heading of a self-evident
overlay stays `KEEP` (PAUSED); track C is deferred to a later run; the run uses `--effort xhigh`, not `max`.

## 1. Scope

- **Track U (headline): UI copy** in JSX/TSX, HTML and SwiftUI source files.
- **Track C (code comments): deferred to a later run** (decision 2026-11-10). Everything below marked for track C
  (the `comment` and `code` roles, the Python parser, `comments_real`, the track C column) is not built for r1.
- Documents (long prose) are out of scope for r1.
- Out of scope for the network (catalogued as slop, but not redundancy): promotional vocabulary, invented
  statistics, em-dash density, visual styling. Count rules and fixed word lists go in a plain-code rule layer (§8)
  that ships with the linter and earns the network no credit.
- **Stretch, not headline:** copy whose label rests on the consequence test (`LABELLING.md` §7), which needs to
  know what a kind of product is expected to do. These nodes carry `ct: true` and are scored in their own column
  (§10). The motivating subtitle is one of them.

## 2. Division of labour

| Part | Who | Fixed before launch |
|---|---|---|
| Extractor: parsers, roles, style flags, clusters, screen JSON (§4) | harness (Claude Code, Opus 5.5) | yes, frozen before sealed labelling |
| Normaliser, stemmer, tokeniser, reserved ids (§6) | harness | yes |
| Transformer interpreter, Python reference and JS runtime with parity tests (§7) | harness | yes |
| Plain-code rule layer (§8); dev generator with exact labels (§9); `KIT.md`; Codex briefs | harness, in a fresh session that has not read the unredacted corpus note (§9) | yes |
| OOD generator (§9); rules-engine baseline (§11) | Codex, each in its own isolated session | yes |
| `LABELLING.md`; labels on the calibration, reference and sealed real sets | Matthew (guide drafted by the harness) | yes |
| Second labels on the calibration pool and 40 sealed screens (§5) | a second person | yes |
| **Vocabulary, lexicon, every weight and bias, choice of pointer head** | **Fable 5.2** | written in the run |
| CLI and ESLint packaging of the final network (§14) | harness | after the run |

Fable writes no parser or runtime code. Its output is `artifact.npz`, `vocab.txt`, `meta.json`, the build script that
produces them and `MODEL.md` explaining every unit.

## 3. Directory layout (`slop/`)

```
SPEC.md, LABELLING.md    this spec; the label definition (both shipped in the packet)
.gitignore               written before any set is built (below)
research/                source notes, never in the packet; corpus-scout.md is redacted (§9)
pyproject.toml, uv.lock  Python 3.12: numpy, torch (baselines and parity only), pyyaml
package.json, lock       Node 22: @babel/parser, parse5, web-tree-sitter + tree-sitter-swift, tree-sitter-python (pinned)
extract/ ui/ lint/       extractor (TypeScript) and golden fixtures; tokenizer.py, porter.py, model.py (numpy
                         reference); runtime.js, tokenizer.js, porter.js, compile.py, CLI, eslint-plugin/, parity tests
rules/ generator/        plain-code rule layer; dev generator (never in the packet)
labeller/                local labelling tool
labels/                  label records per set (gitignored; sha256 committed at freeze)
commissions/             ood_generator/, rules_baseline/ (Codex: BRIEF, hashes, logs, deliverable)
datasets/ registry.yaml  manifests, split id lists and sha256 (data gitignored); freeze record
kit/                     KIT.md, prompt.md, prompt_continue.md, PROMPT_APPROVED.sha256, settings.json, lib/, tools/,
                         make_settings.py, kit_build.py, leakage.py (all new)
scripts/                 all new: prepare_author_dir.sh, launch.sh, audit.py, cost.py, freeze.py, build_sets.py,
                         build_sealed.py, public_to_screen.py, score.py, sealed_eval.py
runs/smoke/ runs/r1/     smoke-launch archive; run archive
```

`slop/.gitignore` ignores `datasets/**/*.jsonl*`, `datasets/**/*.npz`, `labels/`, `commissions/*/deliverable/data/`
and `kit/sealed_salt`. Only manifests, split id lists and sha256 are committed; label files are committed as hashes
only, because `missed` records hold source text. The repo is public (`git@github.com:Hypabolic/HypaNet.git`) and
commits are pushed straight away, so nothing in `slop/` may quote a private repo.

Author directory: `../HypaNet-slop-author/`; smoke directory: `../HypaNet-slop-smoke/`. Clouds r1 is complete; its
launcher refuses while either exists (`clouds/scripts/launch.sh:95`), and that is left as is. Clouds files are not
edited. `slop/kit/make_settings.py` (from `clouds/kit/make_settings.py`) writes `kit/settings.json`:
`defaultMode: dontAsk`; Read/Glob/Grep/Edit/Write denied on the repo, on every other `HypaNet*` sibling (the smoke
directory included), on every entry of `~/.claude` except the session's own project directory (so
`~/.claude/projects/-Users-matthew-Development-hypabolic-HypaNet/**` is denied), on `/tmp/claude-501`, on the private
note directory (§9) and on every repo that supplies `sealed_real`, `public_clean` or `comments_real`;
`autoMemoryEnabled: false`; `switchModelsOnFlag: false`. The launcher runs `make_settings.py --check` and refuses on
any gap. `./tok`, `./lint`, `./score` and `./inspect` resolve their file arguments and refuse anything outside the
author directory.

## 4. Input: screens, clusters, nodes

**Screen** = the visible text of one source file (one component or page), in reading order. A **cluster** is a group
of nodes a reader takes in as one unit; its `kind` is one of a closed list: `header` (eyebrow, heading, subtitle,
badges, actions before the first repeated child), `actions`, `control` (label, placeholder, helper, tooltip), `state`
(empty state, dialog, alert, toast), `card`, `list`, `body` (anything else). The model sees the whole window, so
repetition across clusters is in view.

Screen JSON (`slop.screen/1`), one per file, written by the extractor at labelling time and at lint time alike:

```json
{"schema": "slop.screen/1", "id": "b3f1…", "framework": "tsx",
 "source": {"repo": "game-ui", "path": "src/PauseMenu.tsx", "commit": "4e2a…", "licence": "private"},
 "clusters": [{"k": 0, "kind": "header", "line": 11}, {"k": 1, "kind": "actions", "line": 16}],
 "nodes": [
  {"i": 0, "k": 0, "role": "eyebrow", "text": "OPERATION ON HOLD", "flags": ["caps", "tracked", "small"], "loc": [12, 9, 12, 26]},
  {"i": 1, "k": 0, "role": "h1", "text": "PAUSED", "flags": ["caps"], "loc": [13, 9, 13, 15]},
  {"i": 2, "k": 0, "role": "subtitle", "text": "Resume recaptures the mouse.", "flags": ["small"], "loc": [14, 9, 14, 37]},
  {"i": 3, "k": 1, "role": "button", "text": "RESUME", "flags": ["caps"], "loc": [16, 11, 16, 17]}]}
```

`loc` is `[line, col, endLine, endCol]`, 1-based. Label records key on `loc`, not on `i`. A node whose text holds
a VAR marker also carries `vw`, one integer per marker in order: how many identifier words follow that marker
(`"text": "Due in  eta hours hours", "vw": [2]`); the tokeniser ignores it.

**Roles** (closed list, 36; ids fixed in `ui/tokenizer.py`): `h1 h2 h3 h4 h5 h6 eyebrow subtitle body list_item
caption label helper placeholder input_value button link nav tab breadcrumb menu_item badge tooltip empty_title
empty_body dialog_title dialog_body toast error stat cell aria alt comment code other`. `aria`, `alt` and `code` are
context only: the model reads them, nothing is scored on them.

**Flags** (closed list, 28 of 32 slots): `caps title_case tracked small mono pill dot icon emoji_lead arrow_tail
dynamic stateful conditional in_nav in_table in_form in_list in_dialog in_header in_footer in_card in_state in_control
section_first section_last repeated sr_only truncated`. `dot` marks a leading dot glyph, or a text-less dot element
directly before the node; a text-less dot is never a node. `stateful` marks text or colour bound to an expression;
`dynamic` marks interpolated text; `conditional` marks text chosen by a ternary or `if`. `section_first` and
`section_last` mark the first and last text node under a heading; `repeated` marks a node inside one of three or more
sibling subtrees with the same role sequence (sibling cards, rows). `sr_only` nodes are context only (`LABELLING.md`
§2).

**Role inference** (deterministic, in this order): tag and attribute (`h1`–`h6`, `label`, `button`, `a`, `th`,
`figcaption`, `legend`, `placeholder`, `title`, `aria-label`, `alt`); component name (`*Title`, `*Description`,
`Badge|Chip|Tag|Pill`, `EmptyState[title|description]`, SwiftUI `.navigationTitle`, `Section` header and footer,
`.help()`); class tokens (Tailwind `uppercase` + `tracking-*`, `text-xs`, `rounded-full`, `text-muted-foreground`;
`letter-spacing`, `text-transform` and `font-size` from simple class selectors in the repo's own CSS); position (small
text directly before a heading is an eyebrow; text directly after is a subtitle). A project may add class-to-role
entries in `slop.config.json`; per-repo maps used for the experiment are frozen before labelling. Interpolations
become the `VAR` marker (U+E000) plus the last identifier segment's words; string branches of a conditional are joined
with the `ALT` marker (U+E001); the letters "VAR" and "ALT" are ordinary words. A `.map()` or `ForEach` over an array
literal of strings or objects in the same file is expanded once per element (at most 24), its literal values
substituted, each node at the `loc` of the literal it shows.
Comments (track C) pair a line or JSX comment with the next code statement, in JSX/TSX, Swift and Python.

**Windows.** A screen longer than 64 nodes or 1,024 positions (`BOS` and every `EON` included) is split at cluster
boundaries into windows that each fit; a single cluster that does not fit is split at node boundaries, and its nodes
keep their cluster index. Repetition across windows or files is out of the network's reach (the rule layer catches
exact repeats).

## 5. Labels

Defined in `LABELLING.md`: four classes `KEEP`, `ECHO` (restates another node; pointer required), `NARRATE`
(explains the visible UI; pointer when it explains one node), `DECOR` (carries no information). REDUNDANT = any of
the last three. One record per labelled node in `labels/<set>.jsonl`:

```json
{"screen": "b3f1…", "loc": [12, 9, 12, 26], "class": "ECHO", "target": [13, 9, 13, 15], "equiv": [], "pattern": "R1",
 "aside": [], "partial": false, "ct": false, "conf": "clear", "role_ok": true, "labeller": "mg",
 "guide_sha256": "…", "time": "2026-11-12T10:41:00Z"}
```

`class` may also be `UNSURE` (counted, not scored). `equiv` lists the other copies a stack of equivalent nodes holds
(`LABELLING.md` §5). `ct: true` marks a label decided by the consequence test (§1). `role_ok: false` records an
extractor role error (with `role_fix`); scoring always uses the extractor's roles, because that is what the linter
sees. Screen-level asides are one record per screen: `{"screen": …, "aside_screen": [{"code": "R19", "nodes":
[…]}]}`. Missing nodes are recorded per screen as `{"screen": …, "missed": [{"line": …, "text": …, "class": …}]}`:
missed nodes are reported as extractor recall, and missed REDUNDANT nodes count as false negatives in an
**end-to-end recall** line beside node recall. A missed record may carry a `pattern`. Other records written by the
labelling tool (`slop/labeller/`):

- **Done record.** `{"screen", "done": true, "nodes": N, …}` is written once every node on a screen is labelled.
- **K rules.** A `KEEP` record may name the K rule that decided it in `pattern` (`K1`–`K14`).
- **Screen-level fields.** Screen asides list their nodes by `loc`. Every screen-level record carries `labeller`, `guide_sha256` and `time`.
- **Nodes with no literal words.** These get `class: null` with `aside: ["UNRESOLVED"]`. Scorers skip them and count them.
- **Product kind.** Each repo's product kind (`LABELLING.md` §2a) is recorded in `datasets/kinds.json`.
- **Screen order.** The set builder writes each set's screen order (by hash rank) for the tool's `--order` flag.

**Agreement.** Measured on a calibration pool that belongs to no scored set: all of `reference_real` plus up to 40
screens drawn by hash (seed 20261116, namespace `calibration`) from repos in no split: HypaVPN, lean-ctx,
detectorists, HyperCubeDemo, HypaDev, StreetSweeperPro and the Hypa-Private wiki pages (aitutaki-game is Unreal C++,
which the extractor does not read). A second labeller, a person, labels the pool blind to Matthew's labels. If binary
κ is below 0.6, the guide is revised once using only the pool, both labellers relabel the pool, and both rounds are
reported. If `UNSURE` exceeds 5% of the pool's nodes, the guide is not frozen. Then `LABELLING.md` is frozen: every
label record's `guide_sha256` must equal the frozen guide's hash, and `slop/scripts/freeze.py --check` refuses
otherwise, so any set labelled before a revision is relabelled. After the freeze the second labeller labels 40
`sealed_real` screens (namespace `agreement`) blind with the frozen guide. That measures agreement on the headline set
and is never used to revise the guide.

Reported for the pool and for the 40 sealed screens: Cohen's κ, Gwet's AC1, and positive and negative agreement, each
with a 95% interval that resamples screens; for REDUNDANT against KEEP and for the four classes; with `UNSURE` as its
own category and again with `UNSURE` nodes dropped; on all nodes and on nodes whose role is not
`button`/`link`/`nav`/`tab`/`menu_item`; pointer agreement on nodes both call `ECHO`; `clear` and `judgement` subsets
by Matthew's `conf`. The second labeller's REDUNDANT F1 and false-alarm rate against Matthew on the 40 sealed screens
are the human ceiling, in the same table as the systems (§10). (Shaib et al. 2025 measured binary "slop" κ of −0.15
to 0.29 among three copy-editors; the narrow classes here are meant to do better, and this check says whether they
do.)

## 6. Tokeniser (`ui/tokenizer.py`, mirrored by `lint/tokenizer.js`)

Per node: NFKC; glyphs classified before anything else (`DOT` ●•◉, `MIDDOT` ·, `ARROW` → ↗ », `DASH` — –,
`ELLIPSIS` …, `CHECK` ✓, `STAR` ★, `SPARKLE` ✨, `EMOJI` any other pictograph, `SYMBOL` the rest); the `caps` and
`title_case` flags computed; letter runs split at lower-to-upper case changes, before the last capital of an
upper-case run followed by a lower-case letter, and at underscores (`createEngine` → create engine, `HTTPServer` →
http server, `navigationTitle` → navigation title, `create_engine` → create engine; `iPhone` → i phone is accepted);
then case folding; NFKD with combining marks dropped; split into words (letters, with inner apostrophes removed),
numbers and punctuation; hyphens split words. Words are stemmed with Porter's algorithm as in his definitive ANSI C implementation (three marked departures from the 1980 paper; `ui/tests/data/porter_SOURCE.txt`). Numbers:
0–99 have their own ids, a leading zero adds `LZ` ("01" = `LZ 1`), 1900–2099 is `YEAR`, other integers `NUM`,
decimals `DEC`. A node keeps at most 31 tokens before its end token; longer text is cut and gets the `truncated` flag.
The extractor sets `caps` and `title_case` from styling (CSS `text-transform`, Tailwind `uppercase`) and its literal
text, the tokeniser computes them again from the node's text alone, and a node carries each flag if either side sets
it (the two are ORed).

| Ids | Use |
|---|---|
| 0–511 | 0–5 `PAD`, `BOS`, `EON` (end of node; the readout position), `VAR`, `ALT`, `LZ`; 64–95 glyph classes; 96–191 ASCII punctuation; 192–319 numbers 0–99, `YEAR`, `NUM`, `DEC`; the rest spare (roles and flags enter as embeddings, §7) |
| 512–4,095 | Fable's vocabulary: line *n* (from 0) of `vocab.txt` is id 512 + *n*; at most 3,584 stems |
| 4,096–10,239 | hash buckets for any other stem: 4,096 + (first 4 bytes of sha256(stem), big-endian, mod 6,144) |

Two unrelated out-of-vocabulary stems share a bucket with probability 1/6,144 per pair; `KIT.md` states this.
`vocab.txt` is UTF-8, one Porter stem per line, unique, lowercase, no spaces. Fable chooses its entries; the
tokeniser only looks them up. Porter merges some UI words (new/news → `new`; general/generate/generic → `gener`);
`./tok --vocab` shows every merge (§13).

## 7. Model (`ui/model.py`, numpy reference; `lint/runtime.js`)

The v0.3 transformer (`src/agent_loop_lab/v03/model.py`) with three changes: attention is bidirectional, the block
bias uses signed node distance, and role, cluster and flag embeddings are added to every token of a node.

- Sequence: `BOS`, then for each node its word tokens followed by `EON`.
- Embedding: `h0[t] = E[tok_t] + P[off_t] + Rr[role_t] + Cc[cluster_t mod 64] + Σ_{f ∈ flags_t} F[f]`, with
  `off_t` the position inside the node (0–31). `BOS` takes `E[BOS]` only.
- Layer ℓ = 1..4: `h += Attn(h)`, then `h += W2 relu(W1 h + b1) + b2`. No normalisation, no dropout.
- Attention: 8 heads of width 16; score `q·k / 4 + B_ℓ[h, c(i, j)]` over all positions (no causal mask), with
  `c = 127` for the `BOS` key, else `c = clamp(node_j − node_i, −63, 63) + 63`. Softmax over keys, maximum
  subtracted; padding masked.
- **Precision.** Weights are stored as float32; every computation (embedding sums, q·k, softmax, MLP, logits) runs
  in float64, in `ui/model.py` and `lint/runtime.js` alike. `KIT.md` states this.
- Output: `logits = W_out h + b_out` at each node's `EON`, classes `(KEEP, ECHO, NARRATE, DECOR)`; argmax, ties to
  the lower index. The linter reports nodes whose argmax is not `KEEP`.
- **Pointer.** `meta.json` names one `pointer_head: [layer, head]` (layer 1–4, head 0–7). For a flagged node, sum
  that head's attention weights from the node's `EON` over each other node's tokens; the target is the node with
  the largest sum (ties to the earlier node in reading order), or none if the weight on `BOS` is at least 0.5.
  Reported for `ECHO` and `NARRATE` only.

| Tensor | Shape |
|---|---|
| `tok_emb` E | 10,240 × 128 |
| `off_emb` P, `role_emb` Rr, `cluster_emb` Cc, `flag_emb` F | 32 × 128, 48 × 128, 64 × 128, 32 × 128 |
| per layer: `rel_bias` B; `q`, `k`, `v`, `o` weight and bias; `fc1`; `fc2` | 8 × 128; 128 × 128 and 128 each; 512 × 128 + 512; 128 × 512 + 128 |
| `head` | 4 × 128 + 4 |

Total 2,128,900 parameters, 61.6% of them in `tok_emb` (computed from the shapes above; v0.3 had 2,128,932). Every
tensor starts at 0.0.

**Artefact** (`work/`): `artifact.npz` (uncompressed, float32, exactly the tensor names above, each shape as listed,
all finite), `vocab.txt`, `meta.json` (`{"format": "slop.artifact/1", "pointer_head": [L, H]}`). Its sha256 is the
sha256 of the three files' sha256s in that order. `lint/compile.py` turns it into `slop-net.bin`: the bytes
`SLOPNET1`, a little-endian u32 header length, a JSON header (tensor names, shapes, byte offsets, artefact sha256,
vocab sha256), float32 little-endian data, then `vocab.txt`. About 8.5 MB dense; zero rows are stored sparse.

## 8. Plain-code rule layer (`rules/`, no network credit)

Written by the harness, shipped as separate ESLint rules, scored separately against the `aside` codes:
`slop/repeated-chrome` (scored against the screen-level R19 labels; conditions: eyebrows above more than
ceil(sections / 3) section headings, a section heading being one outside cards, lists, tables, navigation, dialogs,
states and repeated siblings, a quota marked taste (tasteskill v2-experimental, landing pages); any decorative dot,
that is a `dot` flag on a node not bound to data (bound: `stateful`, `dynamic`, or `conditional` by a condition of its
own rather than an enclosing block's); more than one real-state dot per section (v2-experimental SKILL.md:705), where
K2 wins: one stateful dot on each of several sibling cards or rows, each bound to its own item, is not counted; an
arrow suffix on every link outside the footer, when there are two or more, an arrow being → or a similar arrow glyph,
`->` or `=>` (or › as a separate final token), not ↗, which marks an external link, nor »; the same badge, pill or
eyebrow text, arrow suffix or index number on more than half of three or more sibling cards or rows, as
`LABELLING.md` R19 describes, a K13 qualifier badge (tax status, billing period, currency, unit) excepted),
`slop/vacuous-action` (R7 list), `slop/filler-words` (R15 list), `slop/placeholder-as-label` (R8b),
`slop/shared-lead-in` (R10), `slop/index-numbering` (R12) and `slop/exact-repeat` (the same normalised string of two
or more words in two windows of one file, outside navigation, menus, tables, form fields, footer links and site-bar
links (a header link in a cluster with no heading) (K5), nodes flagged `repeated`, `conditional` copies, and pairs
that K11, K12 or separately mounted states (§2a) exempt; the lower-ranked copy by `LABELLING.md` §5 is flagged). The
rules read identifier words from the node's `vw` (§4). The lists and exact conditions are in `rules/README.md`.

## 9. Data, splits and sealing

| Set | Source | Size | Seed (namespace) | Who sees it |
|---|---|---|---|---|
| `dev` | harness generator | 20,000 screens | 20261112 | trained baselines; 400 screens disclosed to Fable and Codex |
| `val` | harness generator | 2,000 screens | 20261112 (`val`) | trained baselines' early stopping (not T-400m); reported |
| `test_gen` | harness generator | 2,000 screens | 20261113 | sealed |
| `ood_codex` | Codex generator | 2,000 screens | 20261114 | sealed |
| `reference_real` | HypaChef, Mosaic, PersonalHomepage, plus the pause-menu fixture | about 50 screens | 20261116 (`reference`) | disclosed to Fable and Codex |
| `sealed_real` (headline) | Atomic-Private, MemoryOS, SandyScouts, hypabolic-homepage, fpp-quilting-generator, HypaAlarm, CoastalQuilts, odysseus, plus the sealed sibling fixture | about 250 screens | 20261116 (`sealed`) | sealed |
| `public_clean` | AOSP `strings.xml` dialogs and settings (Apache-2.0), Firefox Fluent dialogs (MPL-2.0) | about 300 screens | 20261116 (`public`) | sealed; unlabelled |
| `comments_real` | odysseus Python, JSX section comments in sealed repos | about 200 pairs | 20261116 (`comments`) | sealed |

Selection hashes are sha256(namespace, seed, path). Seeds 20261112–20261116 are new (a grep of `src/`, `scripts/`,
`experiments/`, `clouds/` and the README found none of them). At freeze, `test_gen`, `ood_codex`, `sealed_real`,
`public_clean` and `comments_real` go in the README's Sealed data table; 20261112 (dev, val), 20261115 (bootstrap)
and 20261116's `reference`, `calibration` and `agreement` namespaces go in the paragraph of used seeds below it.
Reserved and never used here: 20261104 and 20261106–20261111, 20261201, 20261202.

- **Firewall for sealed copy.** `research/corpus-scout.md` §3 quotes about 60 clusters from the repos that become
  `sealed_real` and `comments_real`, with provisional calls. Before any generator, rule-layer or kit file is written,
  that section moves out of the repo to `~/Development/hypabolic/slop-private/corpus-scout.full.md` (denied in the
  kit settings) and is replaced by a redacted version (repo, path, line, pattern code; no copy, no calls). The
  generator, rule layer, `KIT.md` and Codex briefs are written in a fresh harness session that has never read the
  unredacted note. The labellers do not read it before labelling `sealed_real`; each sealed label file records that.
  `LABELLING.md` was drafted by a session that did read it, so it is covered by the leakage check below and by
  Matthew's review.
- **Dev generator** (`slop/generator/`). Emits source files (TSX, HTML, SwiftUI) plus label records, then runs the
  real extractor, so labels are joined to extracted nodes by source span; a planted span with no node stops the build,
  and a screen whose extracted roles contradict its planted pattern or precedence, or whose copy has a grammar seam
  (a/an before the wrong sound, a doubled word, a bad plural), is regenerated and counted. Screen kinds: marketing
  sections, app pages, settings and forms, dialogs, empty states, dashboards, game menus and HUDs.
  It plants every scored pattern of `LABELLING.md` §9 with an exact class, pattern and target, and every K rule as
  hard negatives, with clean screens mixed in; comment pairs, R18 and K10 belong to track C and are not built for r1.
  Each screen is built around one product, so kept copy adds a fact about what its heading or label names. Restating
  copy and informative copy each come both with and without words shared with the node they sit under, every role the
  guide lets be redundant carries both classes, and copy is assembled from slots (nouns, synonyms, figures, names), so
  that no role, stem overlap or frequent string decides the class. Restatements carry the figures, dates and limits
  of what they restate, and the same sentence occurs as a restatement under a heading or label that states its fact
  and as `KEEP` under one that does not, so that the class depends on the context, not on the sentence. A screen's
  kind and whether it is clean are written only to a harness sidecar (`<set>.meta.jsonl`), never to the screen or
  source files, and source paths name no kind. Its templates and synonym tables never enter the packet and pass the leakage check.
- **OOD generator (Codex).** Briefed with `LABELLING.md` and §4 only. It emits TSX, HTML and SwiftUI source files with
  span-keyed labels; the harness runs the real extractor on them (0 unmatched spans, §12.4). It never sees the
  harness generator, `dev`, `val`, the reference labels or Fable's output.
- **Disclosure to Fable.** 400 `dev` screens ranked by sha256(seed, screen id), taken in order until each scored
  pattern has 10 positive nodes and each K rule 10 negatives, then filled to 400. All of `reference_real` with
  Matthew's labels.
- **Real sets.** Repos are split whole; Atomic-Private (three apps, one house style) counts as one repo for the cap
  and the bootstrap. Excluded before selection (`slop/scripts/build_sets.py`): vendored primitives
  (`components/ui/**`, `vendor/**`, any file byte-identical to a shadcn registry file), `*.worktrees`,
  `DesignRefs/**` where a `design_handoff_*` copy exists, `plan/prototype/**` where it mirrors `src/`, test, spec and
  story files, files git does not track, and any `reference_real` or calibration file byte-identical to a file in a
  sealed repo (a sha256 check found 54 PersonalHomepage UI files byte-identical to files in sealed repos; the sealed
  files are hashed as bytes only, into a gitignored list). Screens are then deduplicated across all real sets by sha256 of the normalised node-text
  sequence: a screen whose hash, or whose node set at Jaccard ≥ 0.8, matches a `reference_real` screen is dropped from
  `sealed_real`, and within a set only the first by hash rank is kept. The manifest records how many screens each rule
  dropped. Screens with at least two nodes that are not context only (`LABELLING.md` §2) are ranked by hash; no repo
  gives more than 25% of the nodes of `sealed_real` or of the pairs of `comments_real`, or more than 40% of the
  nodes of `reference_real`'s repo screens. The calibration screens take the same cap: no repo gives more than 40%
  of their nodes, and a shortfall from 40 screens is reported, never filled by relaxing the cap. In `reference_real`
  and the calibration screens no directory gives more than 6 screens, taken in hash-rank order, so that one folder of
  similar components (PersonalHomepage's blog figures, HypaVPN's `UI`) cannot fill the set. MemoryOS is third-party (Apache-2.0) and is credited. ACCACloner
  and the 3DConnexions SDK are excluded. Private repo text is never committed or published; manifests hold repo,
  path, commit and hashes.
- **Sealed sibling fixture.** Before launch Matthew obtains a second AI-built game or app overlay (eyebrow, heading,
  subtitle, buttons; built by a coding agent of his choice), labels it and adds it to `sealed_real`'s manifest.
  Nobody shows it to Fable.
- **`public_clean`** is not labelled. AOSP and Firefox dialogs become screen JSON through
  `slop/scripts/public_to_screen.py` (resource-id suffix → role map frozen in `registry.yaml`; one dialog or
  preference screen = one screen), reported as a different input path from the linter's. Flags from every system are
  pooled, deduplicated by node, shuffled and adjudicated by Matthew blind to which systems raised them; per-system
  adjudicated false-alarm rates are computed afterwards.
- **Sealing.** Every sealed file and every label file is hashed and the hashes committed before Fable's network
  exists. The sealed run is one-time: `slop/scripts/sealed_eval.py` (new) writes the repo-root marker
  `artifacts/reports/slop_sealed.started.json` with a random token before opening any sealed file and refuses if it
  or the report exists (pattern from `src/agent_loop_lab/v03/sealed.py`).
- **Leakage check** (new, `slop/kit/leakage.py`; `src/agent_loop_lab/v03/packet.py` supplies only the sentinel
  pattern). At sealing time `slop/scripts/build_sealed.py` writes `sealed_ngrams.json`: salted sha256 (salt in the
  gitignored `kit/sealed_salt`) of every normalised whole node string of 2 or more words, and of every 3-word run, in
  `sealed_real`, `comments_real`, `test_gen` and `ood_codex` (§6 tokeniser, stems). The check compares hashes only and
  opens no sealed file. It fails on any match in `KIT.md`, `LABELLING.md`, `examples/` or the generator tables, and on
  any generator sentinel or Codex generator text in the packet. Matches on stop-word-only n-grams, and on phrases
  quoted from the public sources in `research/anti-slop-guides.md` or `research/local-skills.md`, are listed for
  Matthew to waive one by one; each waiver is recorded with its reason.

## 10. Metrics

Every number names its split. **Primary (one number):** REDUNDANT-against-KEEP F1 on `sealed_real`, node level, at
each system's shipped operating point (Fable: `minMargin` 0). **Co-primary:** the false-alarm rate (KEEP nodes
flagged / KEEP nodes). Nodes with `ct: true` or `UNSURE` are left out of both and counted. For every system that gives
a score (Fable's margin, the trained logit, the rules engine's overlap) also: precision at the recall of the Codex
rules engine and at the recall of the exact-stem floor, and the precision-recall curve. **Linter view:** the share of
`sealed_real` files with no REDUNDANT label that get at least one finding, and false findings per 100 KEEP nodes per
file. Also: precision and recall; end-to-end recall (§5); four-class macro-F1 and confusion matrix; recall per pattern
code; the `ct` column (`ct: true` nodes scored on their own); the human ceiling (§5); **pointer accuracy** = among
nodes labelled `ECHO` or `NARRATE` with a non-null target and predicted non-KEEP, the share whose predicted target is
the labelled target or a node in its `equiv` list, also on the intersection of nodes every system flags; results on
`clear` and `judgement` nodes; every metric per repo and with each repo left out in turn; `public_clean` flags per 100
nodes and adjudicated false alarms; track C as its own column.

**Lexicon overlap.** Recall on `ECHO` nodes is split by whether the (node, target) stem pair, or a pair sharing a stem
with it, occurs in any disclosed screen (`examples/dev`, `examples/reference`). Also reported: the share of Fable's
`vocab.txt` stems that occur in the disclosed screens against the share in a matched sample of 400 undisclosed `dev`
screens.

**Fixtures.** The pause-menu fixture node by node, labelled "disclosed to Fable", beside the sealed sibling fixture
(§9) node by node.

Size: parameters, non-zero parameters, `slop-net.bin` bytes. Speed: median and p95 milliseconds per file for the JS
CLI on one CPU thread (M4 Pro, Node version recorded), over `sealed_real`.

Intervals: 95% bootstrap, 10,000 resamples, seed 20261115; real sets resample repos, then screens within each repo;
generated sets resample screens. Paired differences for: written against the Codex rules engine (headline on
`sealed_real`; on `ood_codex` it carries the note "rules engine shares a model with this generator", and on
`test_gen` Fable's result carries "same model family as the generator"), against T-400m and against trained on all
of `dev`, and against each LLM baseline. This list is frozen in `registry.yaml`. No pass mark is set; the comparisons
are the result.

## 11. Baselines

1. **Rules engine (Codex, isolated).** Brief = `KIT.md`, `LABELLING.md`, the same `examples/dev` (400) and
   `examples/reference` directories, and the same `./score` and `./lint` tools on the disclosed sets, under the same
   no-fitting and no-search rule as Fable. Stemmed overlap between each node and the higher-precedence nodes in its
   cluster and screen, role-aware thresholds, a hand-written synonym table of at most as many stems as Fable's
   `vocab.txt` plus lexicon, and the `NARRATE` and `DECOR` cues it can state as rules. Codex's token spend and number
   of revisions are reported beside Fable's.
2. **Same architecture trained by gradient descent** (class loss plus a pointer loss on layer 4, head 0). The matched
   arm, **T-400m**, trains on exactly the 400 disclosed `dev` screens plus `reference_real` (extracted the same way),
   with a fixed epoch count chosen before training, never `val`. The data-scaling curve uses nested random draws from
   `dev` of 100, 1,600, 6,400 and 20,000 screens with early stopping on `val`. Torch seeds 0, 1, 2 for each (15
   checkpoints). Vocabulary: the 3,584 most frequent stems in its training screens. Learning rate, epochs, batch size
   and pointer-loss weight are fixed in `registry.yaml` before any training run.
3. **Prompted LLMs** (cannot run in a linter; optional, run only if Matthew approves the cost and the sending of
   sealed private copy to the API). Claude Haiku 5.5 and Claude Opus 5.5, once on each sealed set. Each gets
   `LABELLING.md` plus the `reference_real` screens with labels as a cached few-shot prefix. Opus 5.5's result is
   marked "same model as the guide's drafter". Unparseable or refused outputs are scored as KEEP and counted. Prompt
   texts and their sha256 are in `registry.yaml`. Estimate (harness, not measured): about 4,750 sealed screens, a
   25,000-token cached prefix, 1,000 fresh input and 400 output tokens per screen; at list prices on 2026-11-10
   (platform.claude.com/docs/en/about-claude/pricing: Opus 5.5 US$4 input, US$0.20 cache hit, US$20 output; Haiku
   5.5 US$0.10, US$0.01, US$0.50 per MTok) about US$80–90 for Opus 5.5 and about US$3 for Haiku 5.5. Also
   Qwen3.5-0.8B locally in `envs/qwen` at no API cost, with the same prefix.
4. **Floors:** always-KEEP; exact-stem match (flag the lower-precedence node of any pair in a cluster whose stem sets
   are equal or nested; point at the other); Vale with Microsoft, proselint, write-good and vale-signs-of-ai-writing
   at default settings on each node's text.

## 12. Validation before launch

1. **Extractor.** At least 60 golden fixtures over TSX, HTML, SwiftUI and Python comment pairs (pause menu included)
   with expected nodes, roles, flags and clusters; all pass. On `reference_real`, Matthew's `role_ok` must be true for
   at least 90% of nodes, or the extractor is fixed and the reference relabelled. The extractor is then frozen (hash in
   `registry.yaml`) before any `sealed_real` or `comments_real` labelling begins; sealed `role_ok` and missed-node
   records are reported only and never used to change it.
2. **Tokeniser.** Porter output matches Martin Porter's published word list and output; 10 golden identifier cases
   (camelCase, PascalCase, upper-case runs, snake_case, digits) pass; Python and JS give identical ids on every node
   of `dev` and `val`.
3. **Interpreter.** A harness-written test network (kept out of the kit) implements the exact-stem floor inside the
   transformer, with a pointer head. Its predictions and pointers must equal the floor's on at least 99% of `val`
   nodes whose floor pair lies within the test network's construction reach (7 nodes); agreement on all `val` nodes
   is reported beside it (96.8% at the first build, every miss a repeated table or list value further apart). The
   check tests the interpreter, not the reach of a hand-built network. The JS runtime must match the numpy model: identical argmax and pointer on every `val` node, logits
   within 1e-6 × max(1, |logit|).
4. **Generators.** Every planted node is found by span (0 unmatched); counts per class and pattern are reported.
   `test_gen` screens whose normalised text hash, or whose node set at Jaccard ≥ 0.8, matches any `dev` or `val`
   screen are regenerated, and the count is reported. `ood_codex` screens containing any node string from
   `LABELLING.md`, `KIT.md` or the 400 disclosed screens are dropped; the build refuses if more than 2% are dropped.
   Matthew labels 30 harness-generator screens (`val` namespace) and 30 Codex-generator screens (seed 20261114,
   namespace `check`, not in `ood_codex`) blind to the planted labels; κ and REDUNDANT F1 of each generator's labels
   against Matthew are reported, and a generator below κ 0.6 is revised before sealing.
5. **Labels.** Agreement (§5) is measured and reported, and the guide frozen, before any sealed labelling.
6. **Freeze.** `slop/scripts/freeze.py` (new, after `scripts/v04_freeze.py`) writes `slop/registry.yaml`: the spec,
   guide, extractor, tokeniser, model, runtime, rule layer, kit, env pins, set manifests, label hashes,
   `slop/scripts/score.py`, `sealed_eval.py` and the bootstrap code, the T-400m and scaling hyperparameters, LLM prompt
   texts and model ids, and the §10 list of paired comparisons; `--check` passes. Before the sealed run it is extended
   once with every baseline artefact (the Codex rules engine, all 15 trained checkpoints and their configs, the
   Vale/proselint configs). `sealed_eval.py` refuses unless every listed hash matches, scores every system in the
   single opening, and runs the LLM baselines inside the same attempt token, hashing their raw outputs before scoring.
   Determinism: same input, same bytes.
7. **Launcher.** Claude Code at least 2.1.321 (the first version with `claude-fable-5-2`; this machine has 2.1.322 from
   2026-11-10). After Matthew approves the prompt and the US$200 cap, one smoke launch at `xhigh`:
   mode `smoke` runs in `../HypaNet-slop-smoke/` (prepared like the real directory and denied in the r1 settings),
   logs to `runs/smoke/logs/` and passes `--max-budget-usd 10`. Its spend can pass US$10 by about one response, up
   to about US$21 (`research/fable-5-2.md` §8). It measures tokens per turn and refusals, and tests
   `showThinkingSummaries: true` in `kit/settings.json`, reporting how many thinking blocks have text. Afterwards the
   smoke directory and its `~/.claude/projects` entry are moved aside; `start` still requires a pristine r1 author
   directory and an empty project directory.

## 13. Author kit and run

- Fresh directory `../HypaNet-slop-author/`, built by `slop/scripts/prepare_author_dir.sh` on the Clouds pattern
  (read-only except `work/`; `HANDOVER.json` with every file's sha256, re-verified before and after each launch). It
  holds `KIT.md` (§1, §4–§7 and the rules; it says the `ct` column is a stretch case and the headline does not depend
  on it), `LABELLING.md`, `examples/dev/` (400 screens with labels), `examples/reference/` (screens, labels and source
  snippets), and `kit/lib/` with read-only copies of `porter.py`, `tokenizer.py` and the numpy `model.py`. `examples/`
  also holds each screen in a compact text form, one line per node (`i k role flags | text | CLASS→target PATTERN
  conf`), and `examples/INDEX.txt` (screen id, kind, node count, patterns present). Tools:
  - `./tok FILE|--text "…"`: normalisation, stems, ids (vocabulary or hash bucket) and flags per node; `--words FILE`
    prints word → stem → id for a word list; `--vocab work/vocab.txt` reports duplicates, entries that change when
    stemmed again, and stems that more than one lexicon word maps to;
  - `./lint FILE` (`.tsx`, `.html`, `.swift`, `.py` or screen JSON in the author directory): the real linter's output;
  - `./score ART --set dev|reference`: a summary table (per class and per pattern), pointer accuracy and at most 40
    wrong nodes with their text, filtered by `--class`, `--pattern`, `--screen` and paged by `--offset`;
    `--files work/probes/*.json` scores screen JSON the author wrote (labels optional; per-node logits and pointer
    printed when there are none); `--against ART2` lists nodes whose class or pointer differs between two artefacts;
  - `./inspect ART FILE --node K`: the node's `EON` row by default (embedding parts, each head's attention grouped by
    source node, projections onto the named directions in `work/directions.json`, `{name: [dims] | {dim: weight}}`,
    and the logits); per-token rows need `--tokens`, MLP units `--units TOP_K` (default 0); `--layer` and `--head`
    filter the attention table;
  - `./python3`: numpy only, under `slop/kit/tools/py_guard.py` (a copy of `clouds/kit/tools/py_guard.py`, hashed in
    `HANDOVER.json`; no `lstsq`, `pinv`, `polyfit`, `.fit`).
  Each tool's output is capped at 20,000 characters and says when it truncates.
- **Deliverables in `work/`:** `build.py` (rewrites the artefact exactly; may read only files under `work/` and import
  `kit/lib/`, never `examples/`), `vocab.txt` (written by `build.py`, which may stem `work/lexicon/*.txt` surface
  words), optional hand-written `lexicon/*.txt`, `artifact.npz`, `meta.json`, and `MODEL.md`: the residual
  allocation, what every head and MLP group computes, the lexicon and its synonym groups, the revisions made after
  scoring and why, and known failures.
- **Rules (in the prompt):** no training, fitting, optimisation or automated search of any kind (no gradient descent,
  regression, least squares, or sweeps: scoring several settings and keeping the best is a search, even by hand);
  no code that reads the examples to compute any weight, vocabulary entry or synonym group; every vocabulary row and
  synonym group must be general UI vocabulary from the author's own knowledge, not a key for one disclosed screen; no
  WordNet or other lexical resource (none is in the kit); numpy only; don't modify or work around the tools; keep
  `MODEL.md` current; write each file as soon as it is settled.
- **Launch** (`slop/scripts/launch.sh`, from `clouds/scripts/launch.sh`): `claude -p --model claude-fable-5-2
  --effort xhigh`, `--settings kit/settings.json --setting-sources project`, `--permission-mode dontAsk
  --permission-prompts none`, `--strict-mcp-config`, `--tools Read,Glob,Grep,Write,Edit,Bash`, Agent, Workflow, web,
  Artifact and Skill tools disallowed, `--disable-slash-commands`, `--max-budget-usd` = remaining cap,
  `--output-format stream-json --verbose`. No `--fallback-model`; `CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK=1`,
  `CLAUDE_CODE_NO_MODEL_FALLBACK=1`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`, `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`,
  `CLAUDE_CODE_DISABLE_CLAUDE_MDS=1`; the parent-session environment scrub of the Clouds launcher, which unsets
  `CLAUDE_CODE_EFFORT_LEVEL` (it outranks `--effort`, and Claude Code's own Fable 5.2 default is `medium`);
  `runs/r1/logs/NAME.launch.json` written before claude starts; `start` refuses unless `kit/PROMPT_APPROVED.sha256`
  holds the sha256 of `kit/prompt.md` (written by Matthew), and below Claude Code 2.1.321. `continue K` resumes after
  API errors with a fixed technical prompt. A launch ends only at the cap or when the author's last message ends with
  `MODEL COMPLETE`; long thinking is never a reason to stop. The prompt carries no budget wording (Anthropic's Fable
  5.2 prompting guide reports that a large stated budget raised spend) and asks for files and decisions, not for
  reasoning in the reply.
- **Cost cap: US$200** for Fable, smoke launch included, for Matthew to approve. Basis (`research/fable-5-2.md` §7–8):
  the three Fable 5.1 runs repriced at Fable 5.2 rates come to US$45.26–60.62 (the upper figure is Clouds r1
  including its aborted-launch charge, scaled); allowing 1.3× tokens gives an expected US$45–80; about 2× the dearest
  run's tokens gives US$100–130; one response can overshoot the cap by about US$21, and the smoke launch costs up to
  about US$21. US$200 covers the high case, one overshoot and the smoke launch (130 + 21 + 21 = 172). Codex and
  LLM-baseline spend are approved separately.
- **Afterwards:** archive to `runs/r1/`; audit the transcript (`slop/scripts/audit.py`: `MODEL = "claude-fable-5-2"`,
  completion regex `^\s*MODEL COMPLETE\s*$`, refusals counted by `stop_details.category`, allowed commands, fitting
  patterns, slop paths in `MARKERS`; `slop/scripts/cost.py`: `CAP_USD = 200.0`, `PRICES = {"claude-fable-5-2": (8.0,
  40.0, 10.0, 16.0, 0.20)}` ([PZ] in `research/fable-5-2.md` §2), launch names `smoke*`, `r1`, `r1.c<K>` charged
  against the one cap); rebuild the artefact from `build.py` in a fresh guard and compare hashes; run the JS–numpy
  parity test on Fable's artefact over `val` and the disclosed sets and report any mismatch (sealed sets are scored
  with the numpy reference); write `INTEGRITY.md`, with the share of thinking blocks that have text; commit the
  archive (transcripts and `examples/reference/` redacted to ids and hashes unless Matthew clears those repos, §16 Q5)
  before any sealed scoring; run the sealed evaluation once.

## 14. Deliverable: the linter

`slop/lint/` becomes an npm package with a `slop` CLI and `eslint-plugin-slop`. The network rule is
`slop/redundant-copy` (options: `classes`, default all three; `minMargin`, default 0, the gap between the top
non-KEEP logit and KEEP); the §8 rules sit beside it. Same extractor, tokeniser and runtime as the experiment.
Example output on the pause-menu fixture, which is disclosed to Fable:

```
$ slop src/PauseMenu.tsx
src/PauseMenu.tsx
  12:9  ECHO     eyebrow "OPERATION ON HOLD" restates h1 "PAUSED" (13:9)       slop/redundant-copy
  14:9  NARRATE  subtitle "Resume recaptures the mouse." explains button "RESUME" (16:11)   slop/redundant-copy
2 findings in 1 file · network slop-r1 3f2a9c1e · 4 ms
```

(The NARRATE finding is the `ct` stretch case; the headline score does not depend on it.) `--format json` prints
`{"file", "line", "col", "class", "role", "text", "target": {"line", "col", "role", "text"}, "margin", "rule",
"network"}` per finding; `--format sarif` is also supported. Under ESLint the same message appears at the node's
location.

## 15. Risks and what would make this fail

- **Labels.** If κ stays low after one revision, every score has a low ceiling; `NARRATE` is the likely weak spot.
- **World knowledge.** Whether a stated consequence is expected depends on the kind of product (`LABELLING.md` §7).
  A lexicon cannot carry that, so consequence-test labels are a stretch column.
- **Concentration.** `sealed_real` leans on two repos with one house style (shadcn and Tailwind); about 250 screens
  over 8 repos gives wide repo-level intervals, and the leave-one-repo-out table shows how much any one repo moves.
- **Generator fit.** v0.5 Phase B scored macro-F1 0.959 on sealed test with 31.4% false alarms on sealed OOD (README);
  a network written against 400 generated screens can fail the same way on real copy.
- **Shared priors.** Fable and the harness generator are both Claude models; Codex wrote both the OOD generator and
  the rules engine. §10 notes each case.
- **Coverage.** A paraphrase outside Fable's vocabulary falls in a hash bucket and is missed; an extractor role error
  (an eyebrow read as body text) becomes a network error.
- **Easy credit.** The exact-stem floor and the rule layer show how much of any score cheap repeats explain.
- **Fable 5.2** has no HypaNet run yet: token use at `xhigh` and `frontier_llm` refusals are unknown until the smoke launch.

## 16. Open questions for Matthew

1. ~~Is PAUSED redundant?~~ Decided 2026-11-10: `KEEP` (K11).
2. Who is the second labeller? It must be a person. If a model is used instead, it is not Opus 5.5 or a Fable model,
   its κ is reported as "model agreement", that model is not also a baseline, and the sealed screens it reads go to
   its API (cost and privacy as in §11.3).
3. ~~Effort~~ Decided 2026-11-10: `xhigh`. The US$200 cap is still to be approved with the final prompt.
4. ~~Track C~~ Decided 2026-11-10: deferred to a later run.
5. May labelled strings from Atomic-Private (private) and MemoryOS (third-party) be quoted in the write-up, or only
   aggregates? May HypaChef, Mosaic and PersonalHomepage copy (`reference_real`) appear in the committed transcripts
   and kit archive?
6. Run the optional LLM baselines (about US$80–90 for Opus 5.5, about US$3 for Haiku 5.5, §11.3), sending sealed
   private copy to the API?
7. Accept that the motivating subtitle ("Resume recaptures the mouse.") is scored in the `ct` stretch column, not
   the headline?
8. Supply the sealed sibling fixture (§9) before launch.
