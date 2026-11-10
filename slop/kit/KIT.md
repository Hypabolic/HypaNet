# HypaNet Slop: author kit

You will write, by reasoning, every weight of a small transformer that finds redundant copy in user interfaces: an
eyebrow that restates its heading, a subtitle that restates the heading above it or explains the button below it, a
dot pill on static content, the same fact given twice on one screen. The network runs inside a linter (a command-line
tool and an ESLint rule): deterministic, on CPU, with no language model at lint time. Nobody trains it. You choose its
vocabulary, write its lexicon into the embedding table, and set every weight and bias, from your own knowledge of UI
writing and from what the tools show you.

The question behind the experiment is how weights written by reasoning compare with weights trained by gradient
descent, and with hand-written rules, on a task that needs lexical knowledge: telling that two short strings say the
same thing ("OPERATION ON HOLD" and "PAUSED") or do not ("Playlists" and "New playlist"). That knowledge has to live in
the weights you write.

Everything you need is in this file and in `LABELLING.md`, which defines what counts as redundant. `PROMPT.md` is
your task statement.

## 1. Your directory

| Path | What it is |
|---|---|
| `KIT.md` | This file |
| `PROMPT.md` | The task statement you were given |
| `LABELLING.md` | The definition of the labels: classes, decision procedure, precedence, K rules, pattern codes. It is the answer key's definition |
| `examples/dev/` | 400 generated screens with exact labels (section 9) |
| `examples/reference/` | Real screens from three projects plus the pause-menu fixture, each with its whole source file, and labels to `LABELLING.md` when the kit includes their answer key (section 9; `examples/README.md` says whether it does) |
| `examples/INDEX.txt` | One line per example screen: set, screen id, kind, framework, node count, window count, codes present |
| `kit/lib/` | Read-only copies of the code the linter runs: `porter.py`, `tokenizer.py`, `model.py` and the two tables `tokenizer.py` reads (`ids.json`, `unicode_tables.json`) |
| `./tok`, `./lint`, `./score`, `./inspect`, `./python3` | The tools (section 10) |
| `work/` | Your working directory and the only place you can write. Your deliverables go here |
| `tools/`, `.venv/`, `HANDOVER.json` | The kit's guard, the private interpreter and the hand-over record (the hash of every file you were given). Read-only |

Everything except `work/` is read-only.

## 2. Scope

- **UI copy** in JSX/TSX, HTML and SwiftUI source files. One source file is one **screen**.
- Out of scope for the network, though catalogued as slop elsewhere: promotional vocabulary, invented statistics,
  em-dash density, visual styling. Count rules, fixed word lists and repeats your network cannot see (vacuous "Learn
  more" links, filler words, the same lead-in on every item, index numbering, a device repeated on most sibling
  cards, a placeholder standing in for a missing label, the same string repeated in two windows of one file) belong
  to a separate plain-code rule layer that ships beside your network and earns it no credit. In the labels most of
  these appear as `aside` codes (`LABELLING.md` §9); the node's own class is still whatever the guide's procedure
  gives.
- **Stretch case.** Some labels rest on the consequence test (`LABELLING.md` §7): whether a stated consequence is
  what this kind of product leads a user to expect. Those nodes carry `ct: true` and are scored in their own column.
  **The headline score does not depend on them.** The motivating example's subtitle "Resume recaptures the mouse."
  is one.

## 3. The task: four classes

`LABELLING.md` is the definition; read it in full. In short, every scored node has one class:

| Class | Meaning | Pointer |
|---|---|---|
| `KEEP` | Does a job no other node on the screen does, or is one of the K cases (`LABELLING.md` §6) | none |
| `ECHO` | Restates information another node visible with it already gives | the node it repeats (none when the information is spread over several) |
| `NARRATE` | Explains the visible UI: what a visible control does, the layout, a literal count of visible items | the node it explains, when there is one |
| `DECOR` | Carries no information: a section-genre label, a dot or pill on static content, a meta strip | none |

`ECHO`, `NARRATE` and `DECOR` together are **REDUNDANT**. Redundancy is repeated *information*, not repeated words:
an eyebrow "OPERATION ON HOLD" over a heading "PAUSED" is `ECHO` with no shared word, and a heading "Playlists" over a
button "New playlist" is `KEEP` although they share a stem. Of two copies of the same information, the one with the
lower rank (`LABELLING.md` §5: controls and main headings first, eyebrows, badges, tooltips and placeholders last) is
the one flagged, and it points at the other.

The motivating example (a game pause menu built by an AI; it is one of the reference screens):

| Node | Role | Label |
|---|---|---|
| OPERATION ON HOLD | eyebrow | `ECHO` → PAUSED (R1) |
| PAUSED | h1 | `KEEP` |
| Resume recaptures the mouse. | subtitle | `NARRATE` → RESUME (R6, `ct: true`) |
| RESUME, SETTINGS, CONTROLS, RESTART MISSION | button | `KEEP` |

**Not scored** (context only; the network reads them, nothing is scored on them): nodes with role `aria` or `alt`,
nodes flagged `sr_only`, and nodes whose text has no literal words (only `VAR` and identifier words; their label is
`null` with `aside: ["UNRESOLVED"]`). The roles `comment` and `code` exist in the id table but no node in this run
has them.

## 4. Input: screens, clusters, nodes

A screen is the visible text of one source file in reading order, written by the extractor as JSON
(`slop.screen/1`). The same extractor runs at lint time.

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

- `loc` is `[line, col, endLine, endCol]`, 1-based. Label records name nodes by `loc`.
- **Clusters.** A cluster is a group of nodes a reader takes in as one unit; `k` is its index in the screen, in
  reading order. Its `kind` is one of `header` (eyebrow, heading, subtitle, badges, actions before the first body
  text or repeated child), `actions`, `control` (label, placeholder, helper, tooltip of one field), `state` (empty
  state, dialog, alert, toast), `card`, `list`, `body`. **The network receives the cluster index (`k mod 64`), not
  the kind.** Nodes of one cluster share `k`.
- **Roles** (closed list; the number is the row of `role_emb.weight`): 0 `h1`, 1 `h2`, 2 `h3`, 3 `h4`, 4 `h5`,
  5 `h6`, 6 `eyebrow`, 7 `subtitle`, 8 `body`, 9 `list_item`, 10 `caption`, 11 `label`, 12 `helper`,
  13 `placeholder`, 14 `input_value`, 15 `button`, 16 `link`, 17 `nav`, 18 `tab`, 19 `breadcrumb`, 20 `menu_item`,
  21 `badge`, 22 `tooltip`, 23 `empty_title`, 24 `empty_body`, 25 `dialog_title`, 26 `dialog_body`, 27 `toast`,
  28 `error`, 29 `stat`, 30 `cell`, 31 `aria`, 32 `alt`, 33 `comment`, 34 `code`, 35 `other`. Rows 36–47 are never
  used.
- **Flags** (closed list; the number is the row of `flag_emb.weight`): 0 `caps`, 1 `title_case`, 2 `tracked`
  (letter-spaced), 3 `small`, 4 `mono`, 5 `pill`, 6 `dot` (a leading dot glyph, or a text-less dot element directly
  before the node), 7 `icon`, 8 `emoji_lead`, 9 `arrow_tail`, 10 `dynamic` (interpolated text), 11 `stateful` (text,
  colour, class or dot bound to an expression), 12 `conditional` (chosen by a ternary, `&&`, `if`, `switch`, an early
  return, a dialog or sheet), 13 `in_nav`, 14 `in_table`, 15 `in_form`, 16 `in_list`, 17 `in_dialog`, 18 `in_header`,
  19 `in_footer`, 20 `in_card`, 21 `in_state`, 22 `in_control`, 23 `section_first`, 24 `section_last` (first and last
  node under a heading), 25 `repeated` (inside one of three or more sibling subtrees with the same role sequence, or a
  `.map()`/`ForEach` template), 26 `sr_only`, 27 `truncated`. Rows 28–31 are never used.
- **How roles are inferred** (deterministically, so expect some errors on real code): tag and attribute (`h1`–`h6`,
  `label`, `button`, `a`, `th`, `figcaption`, `legend`, `placeholder`, `title`, `aria-label`, `alt`); component and
  prop names (`*Title`, `*Description`, `Badge|Chip|Tag|Pill`, `EmptyState`, SwiftUI `.navigationTitle`, `Section`
  header and footer, `.help()`); class tokens (Tailwind `uppercase` with `tracking-*`, `text-xs`, `rounded-full`,
  `text-muted-foreground`, and the repo's own simple CSS); position (small text directly before a heading is an
  eyebrow, directly after it a subtitle). Scoring always uses the extractor's roles, because that is what the linter
  sees.
- **Interpolations and branches.** An interpolation becomes the `VAR` marker (U+E000) followed by the words of the
  last identifier segment: `{user.firstName}` gives `VAR first name`, `t("account.settingsTitle")` gives
  `VAR settings title`. The string branches of a conditional are joined with the `ALT` marker (U+E001):
  `Online ALT Offline`, `VAR n file ALT VAR n files`. The letters "VAR" and "ALT" typed in copy are ordinary words. A
  node with `VAR` also carries `vw` (how many identifier words follow each marker); the tokeniser ignores it.
- A `.map()` or `ForEach` over a literal array in the same file is expanded once per element (at most 24).
- **Windows.** The network sees a window of at most 64 nodes and 1,024 positions (`BOS` and every `EON` counted). A
  longer screen is split at cluster boundaries into windows that each fit (a cluster too big for a window is split at
  node boundaries). Nodes in different windows never see each other, and a labelled target is always in the same
  window as its node. Most screens fit one window.

## 5. Labels in the examples

One JSON record per labelled node:

```json
{"screen": "b3f1…", "loc": [12, 9, 12, 26], "class": "ECHO", "target": [13, 9, 13, 15], "equiv": [], "pattern": "R1",
 "aside": [], "partial": false, "ct": false, "conf": "clear", "role_ok": true}
```

- `class`: `KEEP`, `ECHO`, `NARRATE`, `DECOR`, `UNSURE` (counted, not scored), or `null` for an unresolved node (with
  `aside: ["UNRESOLVED"]`) or a contested one (below).
- `target`: the `loc` of the node it repeats or explains, or `null`. `equiv`: the other flagged copies when three or
  more nodes say the same thing (`LABELLING.md` §5); a pointer at any of them, or at the target, counts as right.
- `pattern`: the R code for a redundant node, or the K rule that kept it (`K1`–`K14`), or `null`. `aside`: rule-layer
  codes (not scored against the network). `partial: true`: a restatement kept because it adds a clause.
- `ct: true`: decided by the consequence test (stretch column). `conf`: `clear` or `judgement` (or `contested`).
- `role_ok: false` (with `role_fix`, the role the node should have had): the extractor's role is wrong. Scoring
  still uses the extractor's role, because that is what the linter sees.
- Contested (reference screens only): `class: null`, `conf: "contested"` and `candidates`, the two readings the
  answer key could not settle between (shown as A and B in the `.txt` view). Contested nodes are not scored.
- Screen-level records: `{"screen", "done": true, "nodes": N}`; `{"screen", "aside_screen": [{"code": "R19",
  "nodes": [locs]}]}`; `{"screen", "missed": [{"line", "text", "class"}]}` (copy the extractor did not emit).

## 6. Tokeniser (`kit/lib/tokenizer.py`; the linter runs an exact JavaScript mirror)

Per node text, in order:

1. NFKC normalisation.
2. Glyphs are classified before anything else: `DOT` (● • ◉), `MIDDOT` (·), `ARROW` (→ ↗ »), `DASH` (— –),
   `ELLIPSIS` (… or three or more full stops), `CHECK` (✓), `STAR` (★), `SPARKLE` (✨), `EMOJI` (any other
   pictograph, with its modifiers), `SYMBOL` (anything else that is not a letter, digit, ASCII punctuation or space).
3. Letter runs are split into words at lower-to-upper case changes, before the last capital of an upper-case run that
   is followed by a lower-case letter, and at underscores (`createEngine` → create engine, `HTTPServer` → http server,
   `create_engine` → create engine). Hyphens split words. Then case folding, NFKD with combining marks dropped,
   apostrophes removed inside words, and Porter stemming (Martin Porter's ANSI C reference implementation).
4. Numbers: 0–99 have their own ids; a leading zero adds `LZ` first ("01" = `LZ 1`); 1900–2099 is `YEAR`; other
   integers and grouped numbers ("1,234") are `NUM`; decimals are `DEC`.
5. ASCII punctuation has one id per character. White space separates tokens and has no id.
6. A node keeps at most 31 tokens; a longer one is cut and gets the `truncated` flag.
7. The tokeniser computes `caps` (at least two capitals and no lower-case letter) and `title_case` (not caps, at least
   two letter runs, every run starting with a capital) from the text; the extractor sets them from styling and text.
   A node has each flag if either side sets it.

**Ids.** The table has 10,240 rows:

| Ids | Use |
|---|---|
| 0–5 | `PAD`, `BOS` (1), `EON` (2, end of node: the readout position), `VAR` (3), `ALT` (4), `LZ` (5) |
| 64–73 | glyph classes, in the order of step 2 (`DOT` = 64 … `SYMBOL` = 73) |
| 96–127 | the 32 ASCII punctuation characters, one id each, in code-point order (`!` = 96 … `~` = 127) |
| 192–291 | numbers 0–99 (id = 192 + n); 292 `YEAR`, 293 `NUM`, 294 `DEC` |
| other ids below 512 | never produced |
| 512–4,095 | **your vocabulary**: the entry on line *n* of `work/vocab.txt` (counted from 1, as the Read tool and `./tok` count) is id 511 + *n*; at most 3,584 stems |
| 4,096–10,239 | hash buckets: any stem not in `vocab.txt` gets 4,096 + (first 4 bytes of sha256 of the UTF-8 stem, big-endian, mod 6,144) |

Every row of `tok_emb.weight` is yours to write, the hash-bucket rows included. Two unrelated out-of-vocabulary stems
share a bucket with probability 1/6,144 per pair, so within one screen two different unknown stems almost never share
a row, and the same unknown word in two nodes always gets the same id.

**`vocab.txt`**: UTF-8, one Porter stem per line, unique, lower case, no spaces. You choose its entries; the tokeniser
only looks them up, so an entry is only ever used if some word stems to exactly it. Write stems, not words: find a
word's stem with `./tok --words` (an entry that is not lower case or that does not tokenise to itself as one word is
a dead row, and `./tok --vocab` reports it). Porter's stemmer is not idempotent, so a real stem can change when stemmed
a second time (pause → `paus`, but `paus` → `pau`); `./tok --vocab` lists those entries as notes, not problems, and
they are fine when some word stems to them (`./tok --words` shows whether one does). Porter merges some words: new/news → `new`; general/generate/generic → `gener`;
settings/setting → `set`. British and American spellings can stem differently (organisation → `organis`, organization
→ `organ`). `./tok --words` and `./tok --vocab` show every stem and merge.

## 7. The model (`kit/lib/model.py`; the linter runs an exact JavaScript mirror)

A 4-layer transformer with bidirectional attention, no normalisation and no dropout.

**Sequence** (one per window): `BOS`, then for each node its tokens followed by `EON`. Window node numbers: `BOS` is
node 0, the window's nodes are 1, 2, … in reading order. Every token of a node, its `EON` included, carries the
node's role, cluster and flags. `off` is the position inside the node: 0 for the first token, and the node's token
count (0–31) at its `EON`. For example "OPERATION ON HOLD" then "PAUSED" is `BOS oper on hold EON paus EON` with
offsets `0 0 1 2 3 0 1`.

**Embedding**, at every position except `BOS`:

`h0[t] = E[tok_t] + P[off_t] + Rr[role_t] + Cc[k_t mod 64] + Σ_{f ∈ flags_t} F[f]`

and at `BOS`: `h0 = E[1]` only.

**Each layer ℓ = 1…4:** `h += Attn_ℓ(h)`, then `h += W2 relu(W1 h + b1) + b2`.

**Attention.** 128-wide residual, 8 heads of width 16. With row vectors and `[out, in]` weight matrices:

```
q = h @ Wq.T + bq,  k = h @ Wk.T + bk,  v = h @ Wv.T + bv          (each [n, 128]; head j uses dims 16j … 16j+15)
score[j, a, b] = (q_j[a] · k_j[b]) / 4 + B[j, c(a, b)]               (query a, key b, every position: no mask)
c(a, b) = 127                                    if key b is BOS
        = clamp(node_b − node_a, −63, 63) + 63   otherwise (63 = the same node; 64 = the next node; 62 = the previous)
attn = softmax over b (maximum subtracted first)
h += concat_j(attn_j @ v_j) @ Wo.T + bo                            (heads concatenated in order 0…7)
```

The `BOS` position is a query too (its row is computed and updated like any other) and a key for every query.

**MLP:** `h += relu(h @ W1.T + b1) @ W2.T + b2`, 512 hidden units.

**Output.** At each node's `EON` only: `logits = h @ Wout.T + bout`, classes in the order `(KEEP, ECHO, NARRATE,
DECOR)`. The predicted class is the argmax; ties go to the lower index, so all-zero logits mean `KEEP`. The linter
reports every node whose argmax is not `KEEP`. Its margin (the top non-KEEP logit minus the KEEP logit) is used as a
score for the precision–recall curves (section 12); the linter's default threshold is margin > 0.

**Pointer.** `work/meta.json` names one `pointer_head: [layer, head]` (layer 1–4, head 0–7). For a node predicted
`ECHO` or `NARRATE`, take that head's attention row (in that layer) from the node's `EON`, and sum it over each other
node's positions (its tokens and its `EON`). The target is the node with the largest sum (ties to the earlier node),
or none if the weight on `BOS` is at least 0.5. The node's own positions are excluded. A node predicted `DECOR` has
no pointer.

**Precision.** Weights are stored as float32; every computation (embedding sums, q·k, softmax, MLP, logits) runs in
float64, in `model.py` and in the JavaScript runtime alike. The two agree to about 1e-6 of a logit's size, so do not
let a decision hinge on a smaller difference.

**Tensors** (`artifact.npz`; names exactly as below; layers are numbered 0–3 in names and 1–4 in `pointer_head`):

| Name | Shape | Symbol |
|---|---|---|
| `tok_emb.weight` | 10240 × 128 | E |
| `off_emb.weight` | 32 × 128 | P |
| `role_emb.weight` | 48 × 128 | Rr |
| `cluster_emb.weight` | 64 × 128 | Cc |
| `flag_emb.weight` | 32 × 128 | F |
| `layers.{0-3}.attn.rel_bias` | 8 × 128 | B (row = head, column = c) |
| `layers.{0-3}.attn.q.weight`, `.k.weight`, `.v.weight`, `.o.weight` | 128 × 128 each | Wq, Wk, Wv, Wo |
| `layers.{0-3}.attn.q.bias`, `.k.bias`, `.v.bias`, `.o.bias` | 128 each | bq, bk, bv, bo |
| `layers.{0-3}.mlp.fc1.weight`, `.fc1.bias` | 512 × 128, 512 | W1, b1 |
| `layers.{0-3}.mlp.fc2.weight`, `.fc2.bias` | 128 × 512, 128 | W2, b2 |
| `head.weight`, `head.bias` | 4 × 128, 4 | Wout, bout |

2,128,900 parameters in all, 61.6% of them in `tok_emb.weight`. Every tensor starts at 0.0. You do not have to use
every layer, head or unit; anything left at zero contributes nothing (and zero rows are not stored in the compiled
file).

## 8. The artefact

`work/` holds three files that are the network:

- `artifact.npz`: uncompressed, float32, exactly the tensors of section 7 with those shapes, all finite;
- `vocab.txt`: as in section 6;
- `meta.json`: `{"format": "slop.artifact/1", "pointer_head": [L, H]}`.

The artefact's sha256 is the sha256 of the three files' hex sha256 digests concatenated in that order. The harness
compiles it into the linter's binary (`slop-net.bin`) unchanged.

Write them with `model.save_artifact`, which validates names, shapes, dtype and finiteness and writes byte-for-byte
reproducible files (`vocab.txt` included). From `work/build.py`:

```python
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK.parent / "kit" / "lib"))
import model as mdl        # numpy reference; imports tokenizer and porter
import tokenizer as tk     # tk.tokenize_text(text, vocab), tk.ROLE_IDS, tk.FLAG_IDS, tk.Vocab, ...
import porter              # porter.stem(word)

T = mdl.zero_tensors()     # {name: float32 zeros of the exact shape}, in order
# ... set T["tok_emb.weight"][...] and the rest ...
sha = mdl.save_artifact(WORK, T, vocab_stems, (layer, head))   # writes artifact.npz, vocab.txt, meta.json
```

`mdl.load_artifact("work")` and `mdl.predict_screen(art, screen_dict)` run the reference forward pass in your own
scripts; `mdl.forward(art, window, collect=True)` keeps every residual and attention map. `tk.encode_screen(screen,
art.vocab)` gives the windows exactly as the network sees them. To turn a lexicon word into the stem the linter will
produce, use `tk.tokenize_text(word).tokens` (it applies the case splitting, folding and stemming of section 6).

## 9. The examples

`examples/dev/` holds 400 screens from a program that generates TSX, HTML and SwiftUI source files from the guide
and labels them exactly; the files went through the real extractor. They were taken in a fixed hash order until every
scored pattern had at least 10 redundant nodes and every K rule at least 10 kept ones, then filled to 400. They cover
the patterns and the hard negatives, but their phrasing is narrower than real copy, so treat them as a check on your
reasoning, not as the distribution you are judged on.

`examples/reference/` holds real screens from three projects plus the pause-menu fixture, with their whole source
files. When the kit includes their answer key, each has labels to `LABELLING.md` (some nodes may be contested,
section 5); otherwise they are unlabelled and `./score --set reference` is unavailable. `examples/README.md` says
which.

For each screen, in its set's directory:

- `<id>.screen.json`: the screen JSON (section 4);
- `<id>.labels.jsonl`: its label records (section 5; for reference screens only when the answer key is included);
- `<id>.txt`: a compact view, one line per node: `i k role flags | text | CLASS→target PATTERN conf` (target as the
  target node's `i`);
- reference screens only: `<id>.source.<ext>`, the whole source file. Its styling can differ from the screen file's
  (the project's stylesheets are not included), so `./lint` on it can give a node a different flag or role; the
  screen file is the one that counts.

You are not shown any other screen: not the rest of the generated sets, and none of the real screens you are scored
on.

## 10. Tools

Run everything from your directory (the Bash tool starts there). The Bash tool runs only `./tok`, `./lint`,
`./score`, `./inspect` and `./python3`, one command per call (no pipes, redirection, `&&`, `cd` or other programs).
List and read files with the Glob, Grep and Read tools. Each tool refuses file arguments outside this directory, and
caps its output at 20,000 characters, saying when it cuts. `ART` below is an artefact directory (normally `work`; a
second artefact you build for comparison can live in a subdirectory of `work/`). Where a tool takes `FILE`, it is
screen JSON or a `.tsx`, `.jsx`, `.html` or `.swift` source file, which the real extractor turns into a screen first.
Every tool has `--help`.

### `./tok`: what the tokeniser does

```
./tok FILE [--node K] [--vocab-file work/vocab.txt]  per node its text, flags and tokens, then its windows
./tok --text "TEXT" [--vocab-file work/vocab.txt]    one node's text
./tok --words FILE [--vocab-file work/vocab.txt]     a word list: word -> folded word -> stem -> id, for every word
./tok --vocab work/vocab.txt [--lexicon DIR]         checks a vocabulary (below)
```

Each word token prints as surface → folded word → stem → id and what the id is (`vocab line N`, counted from 1, or
`hash bucket N`); other tokens print their kind and id. Without `--vocab-file` every stem shows its hash bucket.
`--node K` shows only the node whose `i` is K. `--vocab` reports as problems duplicates, entries that are not clean
stems and entries that do not tokenise to themselves; as notes, entries that change when stemmed again (fine when
some word stems to them, section 6); and, reading every word of every `*.txt` file in the lexicon directory
(default `lexicon/` beside the vocabulary; it has no comment syntax: every word in the files counts), each stem that
more than one lexicon word maps to, marking stems missing from `vocab.txt`.

### `./lint`: the real linter

```
./lint FILE [--art DIR] [--nodes] [--no-net] [--no-rules] [--emit work/X.json] [--format text|json]
```

Runs the real extractor (for a source file), your network through the shipped JavaScript runtime (on the artefact in
`work/`, or `--art DIR`; none if `work/artifact.npz` does not exist yet) and the plain-code rule layer, and prints the
findings as the shipped linter does, with each network finding's margin (section 7):

```
examples/reference/4649cb8941e87ac2.source.tsx  (screen 00e9930e9b0df44e, 7 nodes, 1 window)
  6:76    ECHO     eyebrow "OPERATION ON HOLD" restates h1 "PAUSED" (7:47)   margin +2.000   slop/redundant-copy
  8:48    NARRATE  subtitle "Resume recaptures the mouse." explains button "RESUME" (11:38)   margin +2.000   slop/redundant-copy
2 network findings, 0 rule findings · network 5aeba5ec · 20 ms
```

Lines whose rule is not `slop/redundant-copy` are the rule layer's, not your network's: `slop/exact-repeat` (the same
string in two windows of one file, R14 or R9), `slop/placeholder-as-label` (R8b), `slop/vacuous-action` (R7),
`slop/filler-words` (R15), `slop/shared-lead-in` (R10), `slop/index-numbering` (R12) and `slop/repeated-chrome`
(R19). `--no-rules` hides them and `--no-net` hides the network. `--nodes` adds every node with its role, flags,
call, pointer and logits; `--emit work/X.json` also saves the extracted screen (a starting point for a probe);
`--format json` prints everything as one JSON object.

### `./score`: score an artefact on the examples

```
./score ART --set dev|reference [--class C] [--pattern P] [--screen ID] [--offset N]
./score ART --files work/probes/a.json [work/probes/b.json ...] [--screen NAME]
./score ART --set dev|reference --against ART2 [--class C] [--pattern P] [--offset N]
```

- `--set`: a summary table (REDUNDANT F1 and false-alarm rate as in section 12, with `ct` nodes left out; the `ct`
  column; the confusion matrix; recall per pattern code, `ct` nodes included and counted; false alarms per K rule),
  pointer accuracy, and at most 40 wrong nodes with their text; `--class`, `--pattern` and `--screen` filter the list
  and `--offset` pages through it. `--set reference` works only when the kit includes the reference labels.
- `--files`: screen JSON you wrote yourself. Labels are optional: put them in `<name>.labels.jsonl` beside
  `<name>.json` (or `<name>.screen.json`), or in a `"labels"` list inside the screen; records as in section 5 (at
  least `loc` and `class`). Labelled files are scored as above; for unlabelled ones it prints each node's class,
  logits, pointer and `BOS` weight.
- `--against ART2`: lists the nodes whose class or pointer differs between the two artefacts.

A probe screen needs, per node, `i`, `k`, `role`, `text` and `loc` (four integers), and optionally `flags`; the
tools say which field is missing or wrong.

### `./inspect`: look inside the network at one node

```
./inspect ART FILE --node K [--tokens] [--units TOP_K] [--layer L] [--head H] [--directions FILE]
```

`K` is the node's `i`. By default it prints the node's `EON` row: the embedding parts (token, offset, role, cluster,
flags), each head's attention in each layer grouped by source node, with what the head writes, the residual's
projections onto your named directions, and the logits. `--tokens` adds the rows of the node's other tokens;
`--units TOP_K` lists the TOP_K most active MLP units per layer (default 0); `--layer` and `--head` filter the
attention table. The directions come from `work/directions.json` (or `--directions FILE`), which is yours to write:
`{"name": [dims] | {"dim": weight, ...}, ...}`; the projection is Σ weight × h[dim] (a plain list means weight 1 for
each dimension). Without the file it prints each vector's length and its six largest dimensions.

### `./python3`: your own scripts

`./python3 work/script.py [args]` or `./python3 -c "code"`. Python 3.12 with numpy and the standard library only.
Scripts must be `.py` files. It reads only inside your directory and writes only inside `work/`; it cannot start other
programs, reach the network, change directory, introspect frames or the garbage collector, set tracing hooks, or
re-import or reload a module. numpy's least-squares and fitting routines (`numpy.linalg.lstsq`, `numpy.linalg.pinv`,
`numpy.polyfit`, the `numpy.polynomial` fit functions and `.fit` methods) and `statistics.linear_regression` are
disabled, because fitting is not allowed (section 11).

## 11. Rules

1. **No training, fitting, optimisation or automated search of any kind**: no gradient descent or other iterative
   update of weights toward a target, however it is written; no regression or least squares in any form; no sweeps.
   Scoring several settings and keeping whichever scores best is a search, even by hand one call at a time: change a
   value because you have a reason, then look. Comparing a change with the version before it, to see what the change
   did, is fine, and so is reverting it once you can say why your reasoning was wrong; trying values you have no
   separate reason for (0.5, 1, 2, 4 …) and keeping whichever scores best is not. Setting a threshold or an output
   bias where the dev score is highest, by eye or in code, is a search: set it from what the mechanism computes (for
   example the margin that one shared stem gives) and say so in `MODEL.md`.
2. **No code that reads the examples to compute any weight, vocabulary entry or synonym group.** No counting words in
   the examples to choose the vocabulary, no thresholds or biases computed in code from scores or from the network's
   outputs on any screen. Measure with the tools, think, then type the number you choose into `build.py` with a
   comment saying why. Building weights in code from your own vectors and lists (orthonormalising a code book you
   chose, an outer product of two of your own directions) is construction, not fitting. Fixed pseudo-random vectors
   from a seeded numpy generator are a construction too; trying several seeds and keeping the best is a search.
3. **Every vocabulary entry and synonym group is general UI vocabulary from your own knowledge**, not a key for one
   disclosed screen. Seeing a gap in an example and adding the general group it belongs to is fine; adding the one
   word pair that fixes that screen is not. There is no WordNet or other lexical resource in the kit, and none may be
   used.
4. **numpy only.** No scipy, torch, sklearn or similar libraries.
5. **Don't modify or work around the tools or the `./python3` guard**, and don't read anything outside this
   directory, except the file Claude Code itself names when it saves a long tool output.
6. **Keep `work/MODEL.md` current**, and write each file with its own tool call as soon as it is settled, so that work
   is never lost if a response is cut short.

What is fine: reading every example and its labels, scoring and inspecting your network as often as you like,
writing your own probe screens in `work/probes/` to test one mechanism at a time, measuring anything, and revising a
weight for a reason you can state, as many times as you like.

## 12. What is scored

Your network is scored once, after the session, on **real screens from other projects that you are never shown**,
labelled to `LABELLING.md`, with the extractor and linter you have here.

- **Headline:** REDUNDANT-against-KEEP F1, node level, at the linter's default threshold (argmax). **Co-primary:** the
  false-alarm rate, KEEP nodes flagged divided by KEEP nodes. Any redundant class counts as a hit for these two (an
  `ECHO` predicted as `DECOR` is still a hit). Nodes with `ct: true`, `UNSURE` or a contested label are left out of
  both and counted; context-only and unresolved nodes are never scored.
- **Also reported:** precision and recall; four-class macro-F1 and the confusion matrix; recall per pattern code;
  precision–recall curves from your margin; the share of files with no redundant node that get at least one finding;
  results on `clear` and `judgement` nodes; every metric per project and with each project left out in turn;
  false alarms on further unlabelled real screens, judged against `LABELLING.md`.
- **Pointer accuracy:** among nodes labelled `ECHO` or `NARRATE` with a target and predicted as anything but `KEEP`,
  the share whose predicted target is the labelled target or a node in its `equiv` list.
- **The `ct` column:** the `ct: true` nodes scored on their own. A stretch case; the headline does not depend on it.
- **Lexicon overlap:** recall on `ECHO` nodes is split by whether the pair of stems involved occurs in any example
  screen, and the share of your `vocab.txt` stems that occur in the examples is compared with the share in other
  generated screens you were not shown.
- Also scored on held-out generated screens, from the generator behind `examples/dev` and from an independent one.
- Also reported: parameters, non-zero parameters, the compiled file's size and lint speed.

The same architecture trained by gradient descent and a hand-written rules engine (and possibly prompted language
models) are scored on the same screens beside it. No pass mark is set; the comparisons are the result.

## 13. What you deliver

All in `work/`:

1. **`build.py`**: rewrites the artefact exactly (`./python3 work/build.py`). It may read only files under `work/`
   that you wrote by hand (your lexicon and tables), never a file that a script produced from the examples, from
   scores or from the network's outputs, and it may import `kit/lib/`; it never reads `examples/`. Every number in it
   is a literal you chose, a formula, or a construction from your own vectors and lists, with a comment saying why.
2. **`vocab.txt`**, written by `build.py`, which may stem the surface words of your lexicon files.
3. **`lexicon/*.txt`** (optional): your hand-written word lists and synonym groups, in any layout `build.py` reads.
4. **`artifact.npz`** and **`meta.json`**, written by `build.py`.
5. **`MODEL.md`**: the residual allocation (which dimensions carry what); what every head and every MLP group
   computes; the lexicon and its synonym groups; the choice of pointer head; the revisions you made after scoring and
   why; known failures; the command that rebuilds the artefact. Keep it current as you go, not only at the end.

Probe screens, scratch scripts and notes may stay in `work/` too.
