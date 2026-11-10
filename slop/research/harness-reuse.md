# HypaNet Slop: what the existing harness gives us

Research note for the slop experiment: Fable 5.2 (`claude-fable-5-2`) writes, by reasoning alone, every weight of a
small model that flags redundant, explanatory or repetitive UI copy (eyebrows that restate the heading, subtitles that
explain a button, dot-highlight pills, repeated labels). The model has to run inside a linter: deterministic, no LLM at
lint time. This note lists what can be lifted from the doom-loop harness (v0.2 to v0.5) and from HypaNet Clouds, and
what has to be new. Paths are relative to the repo root; `file:N` is a line number at commit `1237ffe`.

Nothing here was run against a paid API. No sealed or held-out data was opened.

## 0. Summary

| Piece | Reuse | Change needed |
|---|---|---|
| Author-dir isolation, settings, hand-over record | `clouds/scripts/prepare_author_dir.sh`, `clouds/kit/kit_build.py`, `clouds/kit/make_settings.py` | Names and paths; deny the new sibling in the Clouds settings too |
| Launcher (flags, cap, continue loop, no-init handling) | `clouds/scripts/launch.sh` | `MODEL`, price table, effort check, smoke test |
| Packet leakage check | `src/agent_loop_lab/v03/packet.py` | New sentinel and generator-word lists |
| Transformer | `src/agent_loop_lab/v03/model.py` | Class names; either a two-copy input or a non-causal flag; pointer read from a designated head |
| Tokeniser | none as is | New: Unicode, case folding, trailing punctuation, UI role tokens |
| Guards | `src/agent_loop_lab/v03/guards.py` (torch) or `clouds/kit/tools/py_guard.py` (numpy) | None for the guards themselves |
| Static source rules and transcript audit | `src/agent_loop_lab/v03/source_rules.py`, `clouds/scripts/audit.py` | Model id, markers, allowed commands |
| Freeze registry, sealed sets, one-time eval | `src/agent_loop_lab/v03/registry.py`, `scripts/v05c_build_heldout.py`, `src/agent_loop_lab/v03/sealed.py` | New frozen-file list and new seeds |
| Codex commissions | `experiments/commissions/` pattern | New briefs: OOD UI-copy generator, ESLint-style rules baseline |
| Neuron inspection | `src/agent_loop_lab/v03/inspect_cli.py` (`heads` groups attention by source block) | A `--node` front end over UI files |
| Linter runtime | precedent: `clouds/embed/compile/compile_net.py` (network to GLSL) | New: a JS forward pass and a parity test |

## 1. Authoring packet, author-dir isolation and launch flags

### 1.1 Two generations of the pattern

**v0.3 (packet).** The author sees only a hashed packet built by `src/agent_loop_lab/v03/packet.py:1-30`: the
`[packet]`-marked spec sections, verbatim copies of `model.py`, `tokenizer.py`, `renderer.py`, a zero state-dict
template, `AUTHOR_INSTRUCTIONS.md`, a grammar sheet and `PACKET_MANIFEST.json`. The packet sha256 is the sha256 of the
manifest bytes, and the build is byte-deterministic. `verify_packet` (`packet.py:654`) and `leakage_problems`
(`packet.py:586`) refuse a packet that contains any `SPEC_SENTINELS` phrase (`packet.py:77`, text from withheld
sections) or any `GENERATOR_WORDS` word (`packet.py:85`) outside `SPEC.md`. The author's Python runs through
`experiments/v0_3/author_kit/py`, a private venv under the AUTHOR_RUN guard via `tools/launch.py`
(`src/agent_loop_lab/v03/kit.py:1-16` installs a standalone copy of both guard modules).

**Clouds (kit plus hand-over record).** Newer and tighter; the better base for slop.
`clouds/scripts/prepare_author_dir.sh:15-25` builds `../HypaNet-clouds-author/` outside the repo with `KIT.md`,
`PROMPT.md`, the reference material, `tools/py_guard.py`, thin wrappers `./render`, `./inspect` (they call
`kit/tools/author_tool.py` on the harness side with the repo's uv env, `:85-92`) and `./python3` (private numpy-only
venv under the guard, `:93-100`). Everything is `chmod a-w` except `work/` (`:101-104`); the author dir itself is
`555` (`:112`). `kit/kit_build.py record` writes `HANDOVER.json` (sha256 of every handed-over file, the venv and the
harness files the tools run) into the author dir and into `clouds/runs/r1/HANDOVER.json` (`:105-113`), and the
launcher re-verifies it before and after every launch. Preparation refuses once a session exists (`:56-59`), so the
record is frozen at first launch.

`clouds/kit/make_settings.py` generates `kit/settings.json`: allow Read/Glob/Grep in the author dir, Write/Edit in
`work/`, Bash only for the three wrappers, and Read of the session's own `~/.claude/projects/<dir>/` (Claude Code saves
large tool outputs there; v0.5 Phase C lost a read to a blanket deny, `make_settings.py:1-12`). It denies web tools,
the repo, every `HypaNet*` sibling, Claude scratch and temp roots, each other `~/.claude` entry by name (deny beats
allow, so a blanket deny is not possible), and common shell commands by name (`BASH_DENY`, `make_settings.py:30,78`).

For slop: take the Clouds kit pattern whole and add the v0.3 packet leakage check (sentinels from the generator's
injection templates and from the withheld scoring sections, so the packet cannot leak the slop recipes the data
generator uses).

### 1.2 Exact launch flags (`clouds/scripts/launch.sh`)

The invocation (`launch.sh:224-235`):

```sh
claude -p "$PROMPT" \
  --model claude-fable-5-1 --effort max \
  --session-id $SID            # start;  --resume $SID on continue
  --settings kit/settings.json --setting-sources project \
  --permission-mode dontAsk --permission-prompts none \
  --strict-mcp-config \
  --tools Read,Glob,Grep,Write,Edit,Bash \
  --disallowedTools Agent Workflow WebFetch WebSearch Artifact Skill \
  --disable-slash-commands \
  --max-budget-usd "$REMAINING" \
  --output-format stream-json --verbose \
  < /dev/null > logs/NAME.stream.jsonl 2> logs/NAME.stderr.log
```

Environment (`launch.sh:208-219`): a list of parent-session variables is unset (`CLAUDECODE`, `CLAUDE_CODE_*` session
and messaging ids, `ANTHROPIC_MODEL`, `PYTHONPATH`, `VIRTUAL_ENV`, ...); then `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`,
`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`, `CLAUDE_CODE_NO_MODEL_FALLBACK=1`, `CLAUDE_CODE_DISABLE_CLAUDE_MDS=1`; cwd is the
author dir.

- **No fallback.** Clouds passes no `--fallback-model` and sets `CLAUDE_CODE_NO_MODEL_FALLBACK=1`; `clouds/scripts/audit.py:49,317`
  flags any assistant message from another model. v0.3 still had `--fallback-model claude-opus-5-5`
  (`scripts/v03_launch.sh:41-43,138-139`); do not copy that.
- **Cost cap.** `scripts/cost.py --check` refuses at the cap or with an unresolved launch (`launch.sh:126-133`).
  `--max-budget-usd` gets the remaining cap on start and on continue (`:158-164`): Claude Code's `total_cost_usd` is
  cumulative over `--resume`, so passing only the remainder can never overspend (`:26-31`). A launch record with that
  ceiling is written before claude starts (`:203-206`), so a killed launch is charged its ceiling. Clouds r1 used
  US$75.77 of US$500 (commit `c2d0cf4`); v0.3 Phase A used US$42.73 and Phase B US$21.89
  (`src/agent_loop_lab/v04/cost.py`, docstring).
- **Continue loop.** `launch.sh continue K` (K = 1..30, `:59-60`) resumes the same session with the fixed
  `kit/prompt_continue.md` ("purely technical continuation ... no feedback of any kind"). It refuses after
  `NETWORK COMPLETE` (`:150-152`). A launch with no `system/init` event is moved aside as `NAME.noinit.<UTC>.*` and
  charged nothing (`:243-253`). Long thinking is never a reason to stop: only the cap, an error or `NETWORK COMPLETE`
  ends a launch (`:18-19`). v0.3 also had `resubmit N ERROR_FILE` (N <= 3, runner error text only, at most 2,000
  characters; `scripts/v03_launch.sh:105-121`); keep it if slop has a separate build runner.
- **Pre-launch refusals** (`launch.sh:33-41,64-124`): author dir inside the repo; `HANDOVER.json` mismatch or a test
  hand-over; settings missing the repo deny or a `HypaNet*` sibling; an unfilled `{PLACEHOLDER}` in a prompt;
  `kit/PROMPT_APPROVED.sha256` not equal to the sha256 of `kit/prompt.md` (Matthew approves the prompt by writing it);
  the session's own `~/.claude/projects/` dir not empty. v0.3 also refused while pytest's temp dir existed
  (`scripts/v03_launch.sh:34-40`), because it held spec copies and rendered data; slop's tests will create such files
  too, so keep that check.

### 1.3 What changes for `--model claude-fable-5-2`

Facts from the task brief (platform.claude.com models overview, 2026-11-10): US$8 / US$40 per MTok input / output,
adaptive thinking always on, default effort `high`, 1M context, 128K output. The installed claude-api skill reference
is cached 2026-10-06 and lists Fable 5.1 only (US$10 / US$50, effort `low` to `max`), so nothing below about 5.2 beyond
those facts is confirmed.

1. `MODEL="claude-fable-5-2"` in the launcher (`clouds/scripts/launch.sh:55`) and in the audit (`clouds/scripts/audit.py:49`).
2. **Price table.** `clouds/scripts/cost.py:36-37` has only `claude-fable-5-1: (10.0, 50.0, 12.5, 20.0, 0.25)`
   (input, output, cache write 5 min, cache write 1 h, cache read). An unknown model is priced at `MAX_PRICE`, which is
   the 5.1 row, so an unchanged table over-charges 5.2 by 25% and the cap stays safe. Add a 5.2 row once its cache
   prices are confirmed. Scaling 5.1's ratios would give (8.0, 40.0, 10.0, 16.0, 0.20); that is an inference, not a
   published price. At list prices the same token counts cost 80% of a 5.1 run.
3. **Effort.** Clouds and v0.3 ran `--effort max`. 5.2's default is `high`; whether `max` is accepted for 5.2 is not
   in the facts above. Confirm on the models page before launch. A rejected flag most likely fails before the
   `init` event, which the launcher moves aside at no charge (`launch.sh:243-253`), but that is untested.
4. Thinking needs no flag (Claude Code sends adaptive). `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000` stays valid.
5. Claude Code here is 2.1.311; check that it passes the new id through (`launch.sh check` does not call the API).
6. Refusal fallbacks: keep `CLAUDE_CODE_NO_MODEL_FALLBACK=1` and the audit's single-model check, so a server-side
   fallback to another model shows up as a violation rather than silently.
7. Rename the role in a new roles file (`experiments/roles.yaml:4-8` names `claude-fable-5-1` with an Opus fallback).
8. **Cross-impact on Clouds.** `clouds/scripts/launch.sh:86-96` refuses any `HypaNet*` sibling it does not know, so
   creating `../HypaNet-slop-author` makes a future Clouds launch refuse until its deny list is extended. Slop's
   settings must deny `HypaNet-clouds-author*`, `HypaNet-v0*`, `HypaNet-v1*`, `HypaNet-m*` the same way.

## 2. The v0.3 transformer as a slop model

### 2.1 What it is (`src/agent_loop_lab/v03/model.py`)

Causal transformer, no normalisation, no dropout, no absolute positions (`model.py:7-17`). Constants
(`model.py:44-52`): V = 10,240, d = 128, L = 4, H = 8, d_h = 16, MLP m = 512, R = 64 distance buckets plus a BOS
column, O = 192 within-block offsets, 4 classes. 2,128,932 parameters (`:89`), of which the token embedding is
1,310,720 (61.6%). All parameters start at exactly 0.0 and construction draws no random number (`:92-108, 384-394`).

- Inputs: `tokens`, `event_ord` (0 for BOS, k+1 for every token of block k), `offset` (position inside the block);
  `h0 = E[tokens] + P[offset]`.
- Attention score: `q.k / 4 + B[h, c(i,j)]`, with `c = R` for the BOS key and `min(event_ord_i - event_ord_j, 63)`
  otherwise, and `j <= i` only (`:12-14`). So the only positional signals are the offset inside a block and the block
  distance, capped at 63.
- Ops: embedding lookup, linear maps, softmax attention, ReLU MLP, residual adds. Nothing else.
- Output: `logits = W_out h` at every position, read at each block's `EOE` (`:17`); argmax ties go to index 0
  (`:20-21, 397-399`).
- `activations()` (`:181-209`) returns the embedding, both residuals per layer, attention maps for chosen query rows,
  and logits: the hook every inspection tool uses.
- Appending blocks leaves earlier logits bit-for-bit unchanged (fixed 128-row blocks, `:23-31`), which matters for an
  incremental linter.

### 2.2 Can it take (role token, word tokens) per UI node with per-node outputs?

Yes, without changing the maths, if a **UI node is a block**: block k = `ROLE` token, then the node's word tokens,
then `EOE`; `event_ord` = node index in document order; the label is read at that node's `EOE`. Structure the
transformer has no notion of (depth, parent role, sibling index, "inside a button") can be written into the block as
extra tokens (`DEPTH_2`, `PARENT_HEADER`, digits), as v0.3 wrote tool names and fields into rendered text. Blocks may
hold at most 191 tokens plus `EOE` (`tokenizer.py:136-137`); UI strings fit, document paragraphs need splitting.

Three limits need a decision:

1. **Causality.** A node sees only earlier nodes. In the motivating pause menu, the eyebrow "OPERATION ON HOLD" comes
   before "PAUSED", so the eyebrow's own `EOE` cannot know it is redundant. Options:
   - (a) flag the *later* member of a redundant pair and point back at the earlier one; the linter reports the pair.
     No code change, but the label then depends on order, which suits "repeats X" and not "explains the obvious".
   - (b) **two-copy input**: feed the node list twice and read labels at the second copy's `EOE`s, which see the whole
     first copy. No model change; twice the tokens. A node and its own first copy are N blocks apart, so for N > 63
     nodes the bias cannot single out the copy by distance alone (the author would match by content instead).
   - (c) **minimal code change**: a `bidirectional` constructor flag that drops the `j > i` mask (`model.py:232, 306`)
     and the BOS rule keeps working. Breaks the bit-for-bit prefix property (a full re-lint per file is cheap anyway)
     and the existing causal test. Recommended if the spec is written fresh; (b) if `model.py` must stay byte-identical.
2. **Classes.** `N_CLASSES = 4` and `CLASS_NAMES` are hard-coded (`model.py:52,58`), as are the head shape and
   `PARAM_COUNT`. Four classes would keep every shape: for example `CLEAN`, `RESTATES` (repeats or paraphrases another
   node), `EXPLAINS_OBVIOUS` (copy that narrates an action the control already names), `DECORATIVE` (kicker, eyebrow
   or dot pill that carries no information). Only the name tuple changes.
3. **Pointer head.** Not present, and not needed as new parameters. Designate one (layer, head) in the spec as the
   pointer; the pointer of node t is the source block with the most attention mass from `EOE_t` in that head. That is
   exactly what `inspect heads` already computes (`inspect_cli.py:187-192`, `index_add_` of attention over
   `event_ord`). The scorer reads it through `activations(query_positions=...)`. The spec must also say how the author
   can signal "no pointer" (for example mass on BOS above a threshold).

### 2.3 Tokeniser (`src/agent_loop_lab/v03/tokenizer.py`): must be new

The v0.3 tokeniser is tied to agent logs and fails on UI copy:

- Id layout (`:19-54`): 0-6 specials (`PAD, BOS, EOE, NL` and three line markers), 7-127 reserved words (55 used, all
  tool names and log fields, `:26-38`), 128-223 printable ASCII punctuation, 224-1333 digit groups, 1334-2047 unused,
  2048-10239 sha256-hashed word buckets (8,192, `:52-68`).
- **Non-ASCII raises** `TokenizerError` (`:109-113`): the dot glyph "●", "—", "…", curly quotes, "→", emoji and
  accented letters all fail. Dot highlights are exactly what the user wants caught.
- **Case-sensitive hashing**: "PAUSED", "Paused" and "paused" land in three unrelated buckets. Eyebrows and buttons
  are often upper-case, so repetition across case would need three separate hand-written embeddings per word.
- The word regex `[A-Za-z_][A-Za-z0-9_.\-/]*` (`:56`) keeps a trailing full stop inside the word, so "mouse." and
  "mouse" are different tokens.
- No lexicon: synonyms ("ON HOLD" and "PAUSED", "Resume" and "Continue") share nothing. Whatever paraphrase knowledge
  the model has must be written by Fable into `E` rows for a closed word list, with hash collisions (8,192 buckets)
  as noise.

A slop tokeniser should keep the v0.3 shape (standard library only, frozen, shipped in the packet, spans for
reporting) and add: Unicode normalisation (NFKC) and case folding with a separate `CAPS` / `Title` flag token;
trailing punctuation split off; glyph classes as reserved ids (`DOT_GLYPH`, `EMOJI`, `ARROW`, `DASH`, `ELLIPSIS`);
reserved role tokens (`h1..h6`, `eyebrow`, `subtitle`, `button`, `link`, `label`, `badge`, `pill`, `tooltip`,
`placeholder`, `aria_label`, `title_attr`, `toast`, `empty_state`, `comment`, `doc_para`); and the 714 unused ids
(1334-2047) as spare reserved space.

### 2.4 Size at lint time

2,128,932 float32 weights are 8.5 MB. v0.3 Phase B set 156,501 non-zero weights (README, Versions), so a sparse
export would be far smaller. A JS forward pass is about one file (embedding, 4 x attention + MLP, head). The Clouds
embed already compiles a hand-written network into shader code (`clouds/embed/compile/compile_net.py`); the slop
equivalent compiles the artefact into a `Float32Array` blob plus a fixed JS interpreter, with a parity test against
the torch model on the val set (argmax and pointer equality, logits within a tolerance, since float summation order
differs). A smaller fixed shape (d = 64, 2 layers, fewer hash buckets) is a spec decision; the v0.3 shape is reusable
as is.

An alternative is a Clouds-style MLP over harness-defined probes (`clouds/SPEC.md:92-124`), such as "token overlap
with the nearest heading". That is cheap and transparent, but it moves the detection into hand-engineered features,
which v0.3 deliberately removed. The transformer path keeps the question the same as v0.3: can the author build the
circuits itself.

## 3. Guards and audits to reuse

| File | What it does | Use in slop |
|---|---|---|
| `src/agent_loop_lab/guards.py:505` `no_training_guard` | v0.2 tripwire: no grad mode, no `.backward`/autograd engine, no `torch.optim`, no forward-mode AD, no threads or processes, no import of generator / dataset / eval / baseline packages (`guards.py:1-40`) | Inner layer of every author run and build, unchanged |
| `src/agent_loop_lab/v03/guards.py` `build_guard` (`:710`), `author_run_guard` (`:715`) | Adds an RNG ban (stdlib, numpy, torch, random nn.functional), a compute ban in BUILD (`matmul`, `@`, `softmax`, `nn.functional`, `linalg`, model forward), a `sys.monitoring` identity monitor for calls reached around patched names, I/O policy, C-level file-reader tripwires (`:1-75`) | Reuse unchanged if the slop model is torch; pass the slop model class as `model_classes` and the slop packet files as `read_roots` |
| `src/agent_loop_lab/v03/kit.py` | Installs standalone copies of both guards in the author's `tools/` | Reuse; point `LAUNCHER_SOURCE` at the slop kit |
| `src/agent_loop_lab/v03/runner.py:1-30` | Build runner: packet verify, required files, static rules, `build()` in a fresh `python -I -B` under BUILD, 300 s CPU, archive with hashes | Reuse with slop paths and deliverable names |
| `src/agent_loop_lab/v03/source_rules.py:1-40` | Static rules: forbidden imports, RNG APIs, literal-size limits (256 elements, 512 characters, 200 KB), introspection | Reuse unchanged |
| `clouds/kit/tools/py_guard.py:1-22` | Numpy-only `./python3`: reads inside the author dir, writes in `work/`, no subprocess, network, ctypes or introspection; `lstsq`, `pinv`, `polyfit`, `.fit` and `statistics.linear_regression` raise | Use instead if the author's tools are numpy-only (for example a weights-as-JSON format) |
| `clouds/scripts/audit.py:1-30` | Transcript audit: allowed tools, one model, Bash must be a single allowed command, paths outside the author dir, writes outside `work/`, source rules over every Write/Edit body, fitting / search patterns as "review" findings, guard refusals, cost, `NETWORK COMPLETE` | Base for `slop/scripts/audit.py`; update `MODEL`, `BASH_OK`, `MARKERS` (add `slop/data`, `heldout`, `sealed`, the generator package names) |
| `scripts/v03_audit_session.py:1-20` | v0.3 version of the same, plus static scan of every executed script | Reference for torch-specific checks |

The audit is a regex screen; `clouds/scripts/audit.py:21-22` already says a person or LLM must read every author
source afterwards. Clouds r1's integrity review is the template (`clouds/runs/r1/INTEGRITY.md`, commit `851dda7`).

## 4. Freeze and seal procedure

- **Freeze registry.** `src/agent_loop_lab/v03/registry.py:35` lists `FROZEN_FILES`; `compute_freeze` (`:96`) hashes
  each one's raw bytes (LF only) and rebuilds the packet in a temp dir to record its sha256; the record's own
  `freeze_sha256` is the sha256 of the canonical JSON. `scripts/v03_freeze.py:1-11`: print, `--write` (refuses to
  replace a differing registry without `--force`), `--check` (non-zero on any mismatch). Slop needs
  `slop/registry.yaml` with the spec, tokeniser, renderer, model, guards, runner, kit and env pins.
- **Sealed sets.** Data files are gitignored; only manifests and hashes are committed (README, Sealed data). The doom-loop
  sealed hashes were committed (`51c9fa8`) before any network existed. `scripts/v05c_build_heldout.py:1-22` is the
  most complete builder: fixed seeds and sizes in the official mode, refuses to overwrite, deterministic gzip with no
  timestamps, a committed manifest with every file's sha256, the rows' sha256, the token encoding's sha256 and the
  generator / renderer / tokeniser source hashes, disjointness checks, and a check that no generated id falls in a
  reserved seed namespace (computed from the seed formulas, no data file opened).
- **One-time evaluation.** `src/agent_loop_lab/v03/sealed.py:14-16,137-145`: `write_attempt_record` creates
  `artifacts/reports/*.started.json` exclusively with a random token before any sealed file is opened; sealed loaders
  require that token; the script refuses if the marker or report exists.
- **Seeds.** Used or reserved (README, Sealed data): 20261104, 20261106, 20261107, 20261108, 20261109, 20261110,
  20261111, 20261201, 20261202. A grep of `src/`, `scripts/`, `experiments/`, the README and the Clouds spec finds no
  use of 20261112 to 20261129, so slop could take 20261112 (Phase B disclosure), 20261113 (sealed in-distribution
  test) and 20261114 (sealed Codex-generator test). Record them in the slop spec and in the README table before use.

## 5. Codex-isolated pieces

`experiments/roles.yaml:9-16` and `experiments/commissions/`:

- Each commission has a directory with `BRIEF.md` and `BRIEF.sha256` (every input hashed), the Codex run logs and last
  messages, revision requests (`REVISION_1.md`, `REVISION_2.md`) and `DELIVERABLE.sha256`.
- Codex ran in an isolated directory outside the repo (a scratchpad), with only the brief and copies of shared
  contracts (`experiments/commissions/ood_generator/README.md:1-7`). The run header shows codex-cli 0.166.0, model
  `gpt-6.1-sol`, `approval: never`, `sandbox: workspace-write`, and one instruction: read `BRIEF.md` and carry it out
  using only files in that directory (`codex_run_v1.log:1-13`).
- The harness measured the output and wrote revision requests, but did not edit Codex's code
  (`ood_generator/README.md:9-15`). The OOD generator took 3 versions (v1 at low effort was too narrow).
- The rules baseline got the same information as the weight author and no data, with an explicit no-fitting rule
  (`rules_baseline/BRIEF.md`).

For slop, two commissions fit the same mould: an **independent UI-copy generator** (different templates, frameworks
and phrasing, same label contract, never seeing the harness generator) for the OOD test, and an **ESLint-style rules
baseline** (string overlap, role-pair rules, glyph rules) written from the spec alone. That baseline is also the
obvious thing the shipped linter is compared with.

## 6. Watching the neurons

| Existing | What it shows | Slop equivalent |
|---|---|---|
| `python -m agent_loop_lab.v03.inspect_cli inspect log SESSION` (`inspect_cli.py:1-21`) | Tokens with ids per block and logits at each `EOE` | `./inspect FILE.ui.json`: per node, role, tokens (with ids and hash bucket), class logits, pointer |
| `inspect heads SESSION --event T [--layer L] [--top K]` (`:167-200`) | Per head, attention mass from `EOE_T` grouped by source block, and top tokens | `./inspect FILE --node K`: the same grouped by source node, which doubles as the pointer view |
| `inspect model A.dlw` (`:203`) | Non-zero blocks per tensor | Unchanged on the slop artefact |
| Clouds `./inspect NET.json --shot ID --pixel X Y` (`clouds/kit/tools/author_tool.py:313-352`) | Every probe and unit activation along one pixel's ray | Add `--residual` to print named residual subspaces (from the author's own allocation table) per layer for one node |

Ship these as author-dir wrappers that call a harness-side `author_tool.py`, as Clouds does, so the author cannot
point them at data outside its directory. A `./lint FILE` wrapper (the JS runtime on the author's own inputs) lets
Fable see what the linter will report.

## 7. Proposed layout (`slop/`, mirroring `clouds/`)

```
slop/
  SPEC.md                    task, labels, model, tokeniser, packet rules, phases, scoring, budget
  pyproject.toml, uv.lock    own uv project (torch pinned as the root env, numpy, pyyaml)
  registry.yaml              freeze record (scripts/freeze.py --write / --check)
  research/                  this note; anti-slop guidance survey; label taxonomy
  ui/                        frozen, shipped in the packet: tokenizer.py, renderer.py (UI tree -> node blocks), model.py
  generator/                 harness UI-copy generator with latent slop flags (the label oracle); never in the packet
  commissions/               ood_generator/, rules_baseline/  (Codex; BRIEF, hashes, logs, deliverable)
  datasets/                  manifests and splits only; data gitignored
  kit/                       KIT.md, prompt.md, prompt_continue.md, PROMPT_APPROVED.sha256, settings.json,
                             make_settings.py, kit_build.py, packet.py (leakage check), tools/author_tool.py,
                             tools/ guard copies
  scripts/                   prepare_author_dir.sh, launch.sh, audit.py, cost.py, freeze.py, build_heldout.py,
                             run_build.py, score.py, sealed_eval.py
  lint/                      compile_artifact.py (artefact -> weights blob), runtime.js, eslint-plugin/, cli,
                             parity tests against the torch model
  inspect/                   inspect CLI (log, node, model)
  runs/r1/                   HANDOVER.json, SESSION_ID, logs/, author/ (archived work/), score/, INTEGRITY.md
```

Author dir: `../HypaNet-slop-author/` (`KIT.md`, `PROMPT.md`, `packet/`, `examples/`, `./python3` or `./py`,
`./inspect`, `./lint`, `tools/`, `.venv/`, `work/`).

Not covered here: the anti-slop guidance that should become the label taxonomy (the user expects existing anti-slop
skills to help; several are installed locally, for example `design-taste-frontend` and `redesign-existing-projects`,
and they belong in a separate survey under `research/`).
