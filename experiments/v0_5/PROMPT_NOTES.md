# v0.5 prompt: provenance and deviations

`PROMPT.txt` is the one message the v0.5 author (claude-fable-5-1, no tools) receives as turn 1 of each run.
It is built by `scripts/v05_build_prompt.py` and checked by `tests/test_v05_build_prompt.py`.

```
uv run python scripts/v05_build_prompt.py            # writes PROMPT.txt and PROMPT.sha256
uv run python scripts/v05_build_prompt.py --check    # rebuilds in memory; exit 1 if either file differs
```

- Size: 73,542 characters (73,713 bytes, 1,816 lines, 10,502 words). About 31,800 characters of that are
  the three Python sources. The estimate is about 22,300 tokens, likely between 19,000 and 25,000. There is no
  local Claude tokeniser, so the estimate uses 3.3 characters per token, because source code and id tables
  tokenise densely. The exact figure is `usage.input_tokens` of the first result event of r1.
- sha256: see `PROMPT.sha256` (`<sha256>  PROMPT.txt`). The file has **no trailing newline**, so its bytes are
  exactly the prompt as sent. `claude -p "$(cat PROMPT.txt)"` sends the same bytes, because `$(...)` strips only
  trailing newlines. Under 1 MB, it fits in argv on macOS.

## 1. Inputs and checks

The prompt is assembled only from the frozen v0.3 packet, `experiments/v0_3/packet_handed_over/`, plus the
fixed framing text in the script. The script never reads datasets, reports, weights, transcripts or anything
written by an earlier author. Before it assembles anything it checks:

1. `PACKET_MANIFEST.json` has sha256 `72d2bebf…45f2` (`PACKET_SHA256` in the script). That value is the same
   as `packet_sha256` in `experiments/registry_v0_3.yaml`.
2. Every file listed in the manifest matches its sha256 there.
3. The packet's `model.py`, `tokenizer.py` and `renderer.py` are byte-identical to `src/agent_loop_lab/v03/`,
   which is the code the scorer runs. So the source shown to the author is the code that scores the network.

Cuts and edits are exact. `SPEC.md` and `AUTHOR_INSTRUCTIONS.md` are split at named heading lines. Each split
must be a full partition of the file, so nothing is dropped silently, and every left-out piece is listed in §3.
Every textual edit is a replace-once that fails unless its text occurs exactly once (§4). The token-id table is
generated from the packet `tokenizer.py`, which is loaded from the verified file. The tensor table is generated
from the verified `state_dict_template.json`. Neither is typed by hand.

## 2. Section-by-section provenance

The prompt follows the Exp 5 prompt's order: the task first, then the environment and its exact source, then
the input, the network, scoring, and how to write the weights.

| # | Prompt part | Source |
|---|---|---|
| 1 | `YOUR TASK: …` | Exp 5 line 1 (`YOUR TASK: write, from scratch, the weights of a policy network …`), adapted (§5 D1–D4) |
| 2 | `THE TASK: …` (lead paragraph) | New. Exp 5 `THE GAME:` lead-paragraph style. The facts come from SPEC §1, §1.2, §6.1 and §4.2 (tool list) |
| 3 | `## 1. Objective` | SPEC.md `## 1. Objective and hypotheses`, edited (§4 E1, E2) |
| 4 | `### 1.2 What a good result does and does not show` | SPEC.md §1.2, verbatim |
| 5 | `## 6. Windows, gate, labels` | SPEC.md §6, edited (§4 E3) |
| 6 | `## v0.2 oracle rules and scenario families (unchanged in v0.3)` (v0.2 §6.1–§6.3, §7.1) | SPEC.md, same heading, edited (§4 E4, E5) |
| 7 | `THE LOGS (the network's input): …` (lead) | New. Takes the place of Exp 5 `THE INPUT` |
| 8 | `## 4. Log rendering (frozen)` (§4.1–§4.6) | SPEC.md §4, verbatim |
| 9 | `## 5. Tokeniser (frozen)` | SPEC.md §5, verbatim |
| 10 | `Token ids (the tables of tokenizer.py, listed):` | Generated from packet `tokenizer.py` (`LINE_MARKERS`, `RESERVED_IDS`, `PUNCT_IDS`, `DIGIT_IDS`, `HASH_BASE`) (§5 D7) |
| 11 | `## Operation records (RawOp dicts, the renderer's input)` | AUTHOR_INSTRUCTIONS.md, same section, edited (§4 E6, E7) |
| 12 | `The renderer's exact source (renderer.py):` + source | packet `renderer.py`, verbatim, in a python fence. Exp 5 included the game's exact source the same way |
| 13 | `The tokeniser's exact source (tokenizer.py):` + source | packet `tokenizer.py`, verbatim |
| 14 | `# Grammar sheet (spec v0.3 §4-§5)` (all 16 example blocks) | packet `GRAMMAR_SHEET.md`, verbatim |
| 15 | `THE NETWORK (the same architecture as the trained network it will be compared with): …` | Exp 5 `THE NETWORK (the same architecture as the reinforcement-learning networks it will be compared with)`, including `EVERY WEIGHT STARTS AT 0; you set the weights you need.` |
| 16 | `### 2.2 Target model (immutable within a run)` | SPEC.md §2.2, verbatim |
| 17 | Tensor layout and the list of all 56 tensors | Generated from packet `state_dict_template.json`. The layout column restates §2.2 and model.py (§5 D8). Exp 5 counterpart: `Row i of W1 holds the weights from input i …` |
| 18 | `The model's exact source (model.py):` + source | packet `model.py`, verbatim |
| 19 | `HOW YOUR NETWORK IS SCORED: …` | Exp 5 `HOW YOUR NETWORK IS SCORED:`, rewritten for this task (§5 D9) |
| 20 | `HOW TO WRITE THE WEIGHTS (numbers only; nothing is run). …` | Exp 5 `HOW TO WRITE THE WEIGHTS`. The language is unchanged except for the multi-reply reading rule (§5 D2, D3, D10) |

## 3. Packet content left out, and why

The v0.3 Phase A author got the packet plus a launch prompt
(`experiments/v0_3/author_kit/prompt_v03_phase_a.md`). v0.5 keeps every **task fact** and drops what only
makes sense for a tool-using author who delivers files:

| Left out | Why |
|---|---|
| SPEC.md file header (`# Agent doom-loop detector, spec v0.3: authoring specification` and the paragraph naming `AUTHOR_INSTRUCTIONS.md` / `GRAMMAR_SHEET.md`) | It describes the packet files. There are no files in v0.5 |
| SPEC.md §1 hypothesis table (H1 Design / H2 Compilation / H3 End-to-end) | It defines the experiment's claims through `reference_policy.py` and agreement with it. There is no reference policy in this arm. Exp 5 states no hypotheses either |
| SPEC.md `## 9. Direct-weight construction protocol` (§9.1 deliverables `reference_policy.py`, `build_weights.py`, `CIRCUITS.md`, `ITERATIONS.md`; §9.2 Phase A rules and cost cap; §9.3 Phase B) | Tool-only: these are build rules, file deliverables, self-check iterations, phases and spend caps. The Exp-5-style YOUR TASK and HOW TO WRITE THE WEIGHTS replace them. The cost cap is a harness rule and is not stated (D4) |
| AUTHOR_INSTRUCTIONS.md, every section except "Operation records": "Your directory", "Running Python" (`./py`, author guard), "Deliverables and pinned interfaces", "What the harness does with your files", "Phase A", "Phase B" | Tool or file protocol. The one task fact that matters for writing weights by hand is that weights are float32. The prompt states it ("stored as float32"), and model.py and §2.2 state it too |
| `state_dict_template.json` as a file (`construction` field: "start from model.zero_model() … build() …") | Its names, shapes and parameter count are in the generated tensor table. The `construction` text is about `build_weights.py` |
| `PACKET_MANIFEST.json` | Integrity metadata, not task information |
| The v0.3 launch prompt (`prompt_v03_phase_a.md`) | Tool protocol and deliverables. Its task sentence ("write, by hand and with no optimiser and no labelled examples, the parameters of a small fixed transformer that reads rendered agent-log text and classifies the agent's recent behaviour into four classes") is carried by YOUR TASK |

**Why the "Operation records" section is kept.** The renderer source reads these records, and the table gives
the meaning of their fields. Examples: `changed` lists every tracked file whose text differs; `dropped` lists
the withheld observations, and a dropped digest adds `[workspace untracked]`; the outcome-to-status mapping is
given. These are facts about what the log shows. Without them the author could reconstruct only part of this
from `renderer.py`. The section's first sentence names `renderer.render_session`, which is in the source given.

## 4. Exact edits to packet text

| ID | Where | Old | New | Why |
|---|---|---|---|---|
| E1 | SPEC §1 heading | `## 1. Objective and hypotheses` | `## 1. Objective` | The hypothesis table is dropped (§3) |
| E2 | SPEC §1 | the table starting `\| ID \| Claim \| Measured as \|` (5 lines) and the blank line after it | (removed) | §3 |
| E3 | SPEC §6 | `- **Oracle rules.** The packet carries v0.2 §6.1–§6.3 (latent state definitions, parameters, rules) and the family names in v0.2 §7.1, as the v0.2 packet did.` | `- **Oracle rules.** The labels come from the v0.2 oracle. Its §6.1–§6.3 (latent state definitions, parameters, rules) and the family names in its §7.1 follow.` | No packet. Same information |
| E4 | v0.2 §6.2 | `These values live in \`oracle_config.v0_2.yaml\` and are frozen at M0.` (its paragraph) | (removed) | Names a file and a milestone the author cannot see. The values are in the table above it |
| E5 | v0.2 §7.1 heading | `(names only; this list is in the authoring packet)` | `(names only)` | No packet |
| E6 | Operation records, lead | `error line numbers of SPEC.md §4.3)` | `error line numbers of §4.3)` | No files. §4.3 is in the prompt |
| E7 | Operation records, `changed` row | `Rendered as hunk lines (SPEC.md §4.3)` | `Rendered as hunk lines (§4.3)` | Same |

Nothing else in the packet text is changed. The three Python sources and the grammar sheet are byte-exact and
still say "ships in the authoring packet". That is how the frozen files read, and the author is told they are
the exact sources. The section numbering is the packet's own: v0.3 §6 and v0.2 §6.1–§6.3 both appear, as they
did for the v0.3 author. In the scoring paragraph, "§6" means v0.3 §6 (the gate) and "§6.3" means the v0.2
labelling rules. Both resolve to a unique heading in the prompt (`## 6.` and `#### 6.3`).

## 5. Deviations from the Exp 5 prompt, and why

| ID | Exp 5 | v0.5 | Reason |
|---|---|---|---|
| D1 | `This is a single response and you will get no feedback.` | `You will get no feedback: nothing is run, and the network is scored only after you have finished. You may take as many replies as you need. After each reply you will receive only the message "Continue." (it carries no feedback, and nothing is run in between); after a reply that was cut off by a length limit you may instead receive a fixed notice that it was cut off. If a reply is cut off, continue in your next reply; if the cut fell inside an edits block, open a new edits block and write the cut statement again in full.` | Matthew's decision: the run does not have to be a single response, and the output is not limited. "Continue." is the only message the harness sends. It is fixed and carries no information. The "fixed notice" clause is there because Claude Code sends its own recovery message after a cut inside a turn (D13); the prompt must not promise that only "Continue." ever arrives. The cut-off instruction matches the reader's rule (D3); an earlier draft said "continue where you left off", which an author could read as "carry on the list without reopening a block" (review, 6 Nov) |
| D2 | `Only the LAST edits block of your reply is read` | `Every edits block of every reply is read and applied, in the order written, so you can write the network in parts over several replies and change earlier statements later; do not write an edits block you do not want applied.` | Multi-reply. The warning stops illustrative blocks in the design description from being applied by accident |
| D3 | (none) | `If a reply is cut off inside an edits block, the statements before the cut are applied and the statement that was cut off is skipped: open a new edits block in your next reply and write that statement again in full (whatever message follows the cut). An edits block never continues into the next reply, and lines outside edits blocks are not read.` | A reply can stop at the per-reply output limit (stop_reason max_tokens). The reader skips the incomplete trailing statement, so the author has to know to rewrite it. "Whatever message follows the cut" covers Claude Code's own recovery message (D13), whose default text says "Pick up mid-thought" and so conflicts with this rule; the prompt states which rule the reader applies |
| D4 | (none) | `End your final reply with a line that is exactly END OF WEIGHTS (outside any edits block).` Stated in YOUR TASK and again at the end of HOW TO WRITE THE WEIGHTS | The harness needs a finish signal. "Outside any edits block" keeps the marker from being read as an unreadable statement. **No output budget, turn cap or cost cap is stated anywhere.** The harness's safety cap (e.g. 40 continuations) and the US$500 cap are harness facts. Stating them would set a budget, which Matthew ruled out |
| D5 | `You have no tools: no code execution, no calculator, nothing to run.` | `… nothing to run, no files to read; everything you are given is in this message.` | The v0.3 author worked from files. This makes clear that none exist |
| D6 | `First describe your design briefly (the strategy, and what the hidden units compute), then write the edits block.` | `First describe your design briefly (the strategy, and what the attention heads and MLP units compute), then write the edits blocks.` | Transformer instead of MLP. Blocks are plural (D2) |
| D7 | `THE INPUT (600 numbers …): … Input number i = (row * 10 + column) * 6 + channel` | §4 and §5 verbatim, then a generated **token-id table**: specials 0–6, reserved words 7–54, punctuation 128–158, the digit-group rule (224 + g, 234 + value, 334 + value) and the hashed-word rule | Exp 5 states the input layout explicitly. With tools, the v0.3 author could read every id off `tokenizer.py` in one line of code. Without tools, computing `PUNCT_IDS` by hand is error-prone. The table is a deterministic function of the frozen source and adds no information. The hashed-word rule is stated as in §5. No hashed id is listed beyond those already in the grammar sheet |
| D8 | `Row i of W1 holds the weights from input i …; column j of a weight matrix belongs to unit j of the next layer.` | A layout table (torch names; `[out, in]`; head h = rows 16h..16h+15 of q/k/v, columns 16h..16h+15 of o; rel_bias columns 0..63 event distance, 64 BOS) and every tensor's name and shape | Same purpose, for this model. It restates §2.2 and model.py (`_split_heads`, `nn.Linear`, the bias table). The addressing paragraph under HOW TO WRITE THE WEIGHTS gives the same facts in Exp 5's "Useful strided ranges" position. Addendum (review, 6 Nov): that paragraph said "rows h*16 : h*16+16", which reads like index arithmetic the reader rejects; it now says "rows 16h to 16h+15 … (for head 3: 48:64)" |
| D9 | `HOW YOUR NETWORK IS SCORED: its mean score per episode over 1,000 episodes … It is compared with networks of the same architecture trained by reinforcement learning on this game.` | Held-out sessions from the same simulator. One causal forward pass per session. Class = argmax at each block's EOE (ties to the first index). Window = blocks t-7..t, so windows ending before block 7 are not scored. Scored = passes the gate and has one of the four oracle classes, so INSUFFICIENT_EVIDENCE is not scored. Metric: macro-F1 (mean of the four per-class F1), 95% bootstrap CI over sessions. The false-alarm rate (PRODUCTIVE/RECOVERY windows predicted STALLED/OSCILLATING) is also reported. "compared with a network of the same architecture trained on labelled windows" | These are definitions of the scoring procedure, the counterpart of Exp 5's scoring rule. They match `scripts/v03_score_validation.py`, `agent_loop_lab/v03/scoring.py`, `windows.window_endpoints` and the `metrics.DEFINITIONS` false-alarm definition. The gate is stated in the packet's own words (§6, "withheld observations"), with no more detail than the v0.3 author got. **No data statistic is given**: no split name, session or window counts, class balance, or any score of a trained or earlier network. The two populations (`excluding_disclosed`, `all_val`) are a reporting detail and are not mentioned |
| D10 | Edits examples on W1/W2/W3/b1 | Examples on `tok_emb.weight`, `off_emb.weight`, `layers.0.attn.q.weight`, `layers.0.attn.rel_bias` (an event-distance range and the BOS column), `layers.1.mlp.fc1.bias` (stride), `layers.1.mlp.fc2.weight` (`*=`), `head.bias`. Plus `stored as float32` | Same statement forms as Exp 5, on our tensor names. Every example's indices are in range (tested). The examples are syntax demonstrations with neutral values. Any example hints a little at where things live. These hint no more than Exp 5's "column 0 of channel 3" did |
| D10a | (Exp 5's tensors were small and its indices never needed rules) | After `Ranges: …`: `Indices are plain non-negative integers counted from 0: no negative indices and no arithmetic (write 48:64, not 3*16:4*16); every index and every range end must lie inside its axis (a range is not clamped to the axis); a:, :b and ::s are accepted.` | Added after review (6 Nov, blocker). The reader rejects negative indices, range ends past the axis and arithmetic, where Python/torch would wrap, clamp or compute. Without tools and feedback an author used to torch would lose those statements silently. `tests/test_v05_build_prompt.py::test_index_rules_stated_and_honoured_by_the_reader` checks the sentence and the reader's behaviour together. Exponent notation (`1e-3`) is accepted by the reader but not mentioned: "plain decimal numbers" (Exp 5's words) is enough to write any value, and accepting more than the prompt promises cannot lose a statement |
| D11 | THE GAME (rules plus env source) | THE TASK (SPEC §1, §1.2, §6, the v0.2 oracle rules and families) and THE LOGS (renderer and tokeniser source, operation records, grammar sheet) | Our "environment" is the oracle (its rules, which is all the v0.3 author got; the simulator's code was never in the packet) plus the renderer and tokeniser (their exact source) |
| D12 | System prompt: `Answer the user's request.` (Codex `model_instructions_file`, which **replaces** the model instructions; nothing else in context) | `--system-prompt "Answer the user's request."`, but Claude Code 2.1.311 in `-p` mode still sends more (smoke test, haiku, 6 Nov; `agent_loop_lab/v05/session.py` docstring): (1) the system prompt is two blocks, the fixed line `You are a Claude agent, built on Anthropic's Claude Agent SDK.` and then `Answer the user's request.`; (2) a system-role turn with the working directory (a fresh `run-XXXX` directory in the per-user temp dir, named so it says nothing), "not a git repository", platform, shell, OS version and a fixed note about downloaded files; (3) the model's name, id and knowledge cutoff, and today's date; (4) a user-role reminder carrying the account e-mail (`atomic@hypabolic.com`; its domain names the organisation behind the project). `CLAUDE_CODE_DISABLE_ATTACHMENTS` does not remove (2)–(4); only `--bare` does, and `--bare` does not accept the OAuth login (needs `ANTHROPIC_API_KEY` or a `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`). No CLAUDE.md, memory, skills, hooks, MCP servers, slash commands or tools reach the model (audited per run) | **Open decision for Matthew**: accept (1)–(4) as a recorded deviation, or set up a token and launch with `V05_BARE=1` (implemented, untested: run `scripts/v05_smoke.py` with it first). None of (1)–(4) is task information, but (4) is a small identity leak that Exp 5 did not have |
| D13 | (Codex: one response, nothing in between) | When a reply hits the per-reply output limit, Claude Code continues it **inside the same turn** with its own fixed message, at most 3 times per turn (`un=3` in the binary), and there is no setting to turn this off. In this build the message is `Output token limit hit. Resume directly — no apology, no recap of what you were doing. Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces.`, or, if a server flag selects it after a thinking-only cut, `[Your previous response reached the output limit with no visible output. Please continue and produce a brief user-visible response or tool call right away.]` (all six texts in the binary are in `session.CC_RECOVERY_TEXTS`). Only after the 3rd in-turn recovery does the turn end and the harness send "Continue." | This is a message the harness did not send, and it carries instructions ("pick up mid-thought", "break remaining work into smaller pieces", "brief") but no information about the task or the network. v0.3 Phase A hit it twice (same model, same limit). How it is handled: the prompt says a cut may be followed by "a fixed notice" and that the cut-off rule holds "whatever message follows the cut" (D1, D3); the launcher keeps each API message of a turn as its own part (`replies/NNN.p<K>.md`, with `cut_off` per part in `INDEX.json`) and the reader reads each part as a reply, so the cut-off rule applies at every cut, inside a turn too (a statement cut mid-number is skipped, never applied truncated); lines the author continues outside a block after such a cut are counted in `apply.json` (`outside_block_statement_lines`). The audit reports every recovery message: as a **violation** by default, as a recorded deviation with `--allow-cc-recovery`. **Open decision for Matthew**: accept the in-turn recovery as a recorded deviation (then audit with `--allow-cc-recovery`), or treat any run that has one as failing the premise |
| D14 | (Exp 5: one response; its length is the only limit) | No output limit is stated or imposed, but the **context window** is the real cap: compaction is off (a compaction summary would be a message the harness did not send) and every thinking block is kept (`clear_thinking` `keep: all`), so the prompt plus every reply and its thinking must fit in claude-fable-5-1's window, 1,000,000 tokens in Claude Code 2.1.311's model table. That is about 7 full-length (128,000-token) API messages after the ~22,000-token prompt, in-turn recoveries included. When it is full, Claude Code refuses the next request without calling the API; the runner stops the run as `context_limit` (final, no retries) and the weights written so far are scored. The 40-continuation cap and the US$500 cap are safety caps that the context window will normally reach first | Recorded so the 40-turn cap is not presented as the binding limit. Nothing about it is stated in the prompt (D4) |

What stays exactly as in Exp 5: no training, data, labels, loss or gradients; "you choose every number
yourself"; no tools; no feedback; the description first, then the edits; every weight starts at 0; "the same
architecture as the networks it will be compared with"; the edits language (single entries, explicit lists
over ranges, scalar fills, `a:b`, `a:b:s`, `:`, 2-D lists for 2-D blocks, `*=` `+=` `-=`, `#` comments, lists
continuing over lines, statements in order with later ones overwriting, unreadable statements skipped, "plain
decimal numbers (|v| <= 1e6)"); "numbers only; nothing is run".

### Information beyond the v0.3 Phase A packet

The Exp 5 form states how the network is scored (D9). Four facts in HOW YOUR NETWORK IS SCORED were not stated
in the v0.3 Phase A packet, which said only "macro-F1 and false-alarm rate" and "network against ground truth"
(SPEC.md §1 H1/H3, §6). They are definitions of the scoring procedure, taken from the scoring code, and none is
a data statistic (no counts, class balance or scores):

1. windows labelled INSUFFICIENT_EVIDENCE are not scored;
2. windows ending before block 7 (counting from 0) are not scored;
3. the confidence interval is a 95% bootstrap over sessions;
4. the network "is compared with a network of the same architecture trained on labelled windows" (Exp 5's
   "compared with networks of the same architecture trained by reinforcement learning", for this task).

The v0.3 author could have inferred (1) and (2) from the packet's §6 and window definition, and with tools
could test anything; the v0.5 author cannot, so the Exp 5 form states them.

## 6. What the prompt promises the reader and the harness

The edits reader and the run harness must honour these, because the author is told them:

- Turn 1 sends `PROMPT.txt` byte-exact. Every later turn sends exactly `Continue.`, with nothing appended
  (no error text, no counts, no results). The only other message the author can receive is Claude Code's fixed
  recovery notice after a cut inside a turn (D13), which the prompt mentions as "a fixed notice".
- **Every** ```` ```edits ```` block of **every** assistant reply in the run is applied in order: reply 1's
  blocks first, then reply 2's, and so on. A reply is one API message: a turn that Claude Code continued
  inside itself is read as its parts. Text outside edits blocks is ignored (statement-like lines there are
  counted in `apply.json`).
- A block cut off by the end of a reply (no closing fence) counts up to its last complete statement. The
  incomplete trailing statement is skipped and recorded. "Cut off" is known per reply from the transcript
  (`INDEX.json` `cut_off`); when it is not known the reader treats the reply as possibly cut. A block left
  unclosed at the end of a reply that ended normally is read to its end.
- Indices: non-negative literal integers, inside the axis, no arithmetic, no clamping (D10a).
- An unreadable statement is skipped and recorded, and the rest still apply. Values are plain decimals with
  |v| <= 1e6, stored as float32.
- The run ends when a reply ends with a line that is exactly `END OF WEIGHTS`. If a safety cap ends it
  first, the statements written so far are scored, and this is recorded. The prompt does not mention caps.
- Settled by the reader (`agent_loop_lab/v05/edits.py`, recorded in every `apply.json` under `reader`):
  (a) exponent forms such as `1e-3` are accepted, though the prompt promises only plain decimals; (b) `+=`,
  `-=` and `*=` take a list of the block's shape as well as a scalar; (c) `[ [ … ], … ]` (a space after the
  first bracket) is accepted.
