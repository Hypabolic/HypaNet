# Agent Doom Loop Detector — Weight-Space Programming of Qwen3.5-0.8B

**Specification:** v0.4 (successor to v0.3; every v0.3 and v0.2 result stays as reported)
**Status:** DRAFT r3, 2026-11-06 (r2 2026-11-06, r1 2026-11-05). r3 records Matthew's second direct decision of 2026-11-06, taken before G1: "Val only, no sealed" (D15, §9, §10). r2 records his first, to drop the training baselines and all cloud compute (D3, D12, §11). It is written from `agent_doom_loop_detector_plan_v0_4.md` (every recommendation accepted, including its "Revisions after M0") and the WSP kernel contract `docs/V04_WSP_KERNEL.md`. It is frozen at G1 (§16), before any weight author sees it.
**Purpose:** Measure whether a reasoning LLM can **edit a pretrained model's weights**, with no optimiser, no search and no labelled examples, so that Qwen3.5-0.8B, read through a fixed one-word prompt, classifies an AI coding agent's recent behaviour into the four v0.2 patterns; and compare that edit with building a network from zero (v0.3), and with the model as it comes. (The comparison with ordinary fine-tuning was dropped before G1: §11.)

Sections marked **[packet]** go into the authoring packet. All other sections are withheld from the weight author.

---

## 0. Changes from v0.3

| # | Area | v0.3 | v0.4 |
|---|---|---|---|
| 1 | Model | Hand-built 4-layer transformer, 2,128,932 parameters, all zero at the start | Pretrained Qwen3.5-0.8B (text model, 752,393,024 parameters), pinned revision, fp32 |
| 2 | What the author writes | Every weight, from zero | An edit to the pretrained weights: the replaced tensors only (§8) |
| 3 | Input | v0.3 word-level tokeniser, `BOS`/`EOE`, event ordinals | The same rendered text (§4), read by Qwen's own tokeniser behind a fixed prompt (§5) |
| 4 | Readout | 4-way head at each `EOE` | Verbaliser: the four label-word logits after a fixed suffix, one pass per session with cache branching (§5) |
| 5 | Forward passes | Free, inside author scripts | A separate, metered tool, one request per Bash call, 2,000,000 tokens per phase (§7) |
| 6 | Reference policy | `reference_policy.py`, H2 fidelity | None. Q5 (mechanism check on counterfactual pairs, not run in v0.4: D15) replaces H2 |
| 7 | Labels, windows, gate, metrics | — | Unchanged: the same sessions, oracle labels, gate, scored-window set and metric code |
| 8 | Primary sealed set | v0.3 test | **None in v0.4** (D15): the result is on val, harness-only, with a quick look on a fixed 1,000-session val subset first. The **fresh sealed test set** (§9), v0.3 test, OOD and counterfactual stay sealed for a later version |
| 9 | Trained baselines | Phase C transformers on CPU | None new: no Qwen-probe or Qwen-LoRA. The trained v0.3 C columns are re-scored on val, and v0.2 C-all is cited, as PREREGISTRATION §1 states (§11) |

### 0.1 Decisions (accepted 2026-11-05; D3 and D12 changed by Matthew on 2026-11-06, and D7, D13 and D15 by his second decision that day)

| # | Decision | Value |
|---|---|---|
| D1 | Base model | `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`; every file's sha256 pinned (`envs/qwen/hypaqwen/model.py` `FILE_SHA256`); offline load; SDPA attention; fp32 |
| D2 | Environment | `envs/qwen`, a separate uv project: Python 3.12, torch 2.14.1, transformers 5.22.1. The root v0.3 environment (torch 2.2.2) stays frozen. Neither environment imports the other's packages |
| D3 | Machine | Apple M4 Pro, 24 GB. Everything runs here, on local MPS in fp32 (about 445 tok/s): the author's forward server and the harness-only val runs (there is no sealed scoring run, D15). The CPU replay check runs on this Mac's CPU. Nothing is rented (Matthew, 2026-11-06) |
| D4 | Interface | `w1-state-full`, chosen with no labels by the pre-registered prompt choice as revised (`experiments/v0_4/prompt_choice/README.md`, Revision 1); frozen as `hypaqwen.interface.FROZEN`, `spec_sha256` `5922eb4500122921ac7d07e3d35599b95cb10b462c4f50b89e5d6bb989437879` |
| D5 | Label words | ` productive`, ` recovering`, ` stalled`, ` looping` (each one Qwen token; ` oscillating` is two) |
| D6 | Label-word rows | The embedding is tied to the LM head. Editing the four label rows is allowed and reported, with the input-side effect flagged (§7.4) |
| D7 | Fresh sealed test | 3,996 sessions, 333 per family, master seed 20261106, v0.2 generator unchanged; registration `6a778d1`, manifest `8475556`. Built and sealed; not opened or scored in v0.4 (D15), so it stays available to a later version |
| D8 | OOD | No OOD-2. v0.3's OOD set stays sealed and is not scored in v0.4 (D15) |
| D9 | Weight author | A fresh `claude-fable-5-1` session (fallback `claude-opus-5-5`), effort max, headless `claude -p`, in a new directory `../HypaNet-v04-author`; it has never seen generator, oracle or dataset code |
| D10 | Caps | Session cost: WSP-A US$500, WSP-B US$250, WSP-F US$250, accounted by session-id delta of the cumulative `total_cost_usd` (§13). Forward budget: 2,000,000 tokens per phase, 50,000 per request |
| D11 | WSP-F | Included as a separate, clearly labelled column after WSP-B; never instead of it |
| D12 | Trained baselines | None (Matthew, 2026-11-06: "Drop the training baselines"). There is no Qwen-probe or Qwen-LoRA column at any budget; this replaces the plan's accepted LoRA budgets (32 / 128 / 512 / 2,048 / all, plus 200 windows) |
| D13 | Re-scoring v0.3 on arm64 | The v0.3 val dry run matched every official check exactly on this machine, so the v0.3 A, B and C networks and the constant priors are re-scored on val locally with no tolerance (PREREGISTRATION §1) |
| D14 | v0.3 cost | Corrected to US$64.62 (A $42.73, B $21.89); `scripts/v03_cost.py` double-counted resumed sessions |
| D15 | Sealed scoring | None in v0.4 (Matthew, 2026-11-06: "Val only, no sealed"; "I need a quick answer to how the base qwen vs modified qwen performs, not 90 hours"). The result is base Qwen against the WSP edits on val, excluding the 200 disclosed sessions, harness-only. Each edit is first read out and scored paired against Qwen-0 on the quick-look subset `datasets/v0_4/VAL_QUICK.txt` (1,000 val sessions by hash key, fixed before any edit existed), reported beside, never instead of, full val. The fresh test set, v0.3 test, v0.3 OOD and the counterfactual split are not opened. Q2-OOD and the Q5 verdict are not tested |

The adjudicator may reverse any decision. Reversing one after G1 creates a new registered version.

---

## 1. Objective [packet]

At every tool call of an AI coding agent's session, the model classifies the pattern of the agent's most recent eight operations as one of four classes:

1. **PRODUCTIVE**: evidence of forward progress or materially new investigation.
2. **RECOVERY**: a bounded retry or repair after a transient or diagnosed failure, still within policy or recently resolved.
3. **STALLED**: repeated ineffective operations, with no observed progress and no state-reversion cycle.
4. **OSCILLATING**: recurring reversal between previously held workspace or execution states, with no net progress.

The model is **Qwen3.5-0.8B** (§3), a pretrained language model. It reads the session's rendered log (§4) behind a fixed instruction, and after each tool-call block a fixed suffix asks for the agent's state. The class is whichever of four label words (` productive`, ` recovering`, ` stalled`, ` looping`) gets the highest next-token logit (§5). The prompt is frozen. **Only the weights may change.**

Your task is to edit Qwen's weights, by reasoning, so that this readout classifies the agent's state well. You may read every weight, run metered forward passes on texts you write yourself, and save direct, reasoned edits (§7). You may not train, fit or search (§7.6). You see no labelled data.

### 1.1 What a good result does and does not show

- **The logs are synthetic, in one fixed grammar.** There is one header line per tool call, with a tool-specific name and canonical result syntax. Tool output and assistant text are absent, and file content is replaced by opaque content ids. A good result shows that a pretrained model can be programmed to perceive patterns in this grammar. It does not show robustness to real transcripts.
- **Diff content is opaque.** Whether an edit repeats, undoes or restores an earlier change is visible from the ids. Whether it is correct code is not. Progress shows only through later validation results, as in the real world.
- **The prompt carries the class definitions.** The four definitions above are in the frozen instruction the model reads, so an edit can build on what the released model already makes of them.
- **Each decision sees only the past.** The label after block t never sees block t+1.
- An edit that changes few weights and whose mechanism can be stated and checked is worth more than an opaque one with the same score.

---

## 2. Questions, hypotheses and arms

### 2.1 Questions

Each is a research target, not a pass/fail gate, as in v0.3. Comparisons are always at **equal labelled data**.

| ID | Question | What would support it |
|---|---|---|
| Q1 | Can Qwen do the task as it comes? | Qwen-0 clearly above the constant prior on val; its gap to v0.3 A on val is the value of hand-building |
| Q2 | Does WSP on Qwen beat building from zero? | WSP-B above v0.3 B on val (excluding the disclosed sessions) at the same 200 labelled windows; WSP-A against v0.3 A (0 labels). The OOD comparison (v0.3 B 0.908) is not tested in v0.4 (D15) |
| Q3 | Does WSP beat ordinary fine-tuning of the same model? | Not tested in v0.4: the training baselines were dropped before G1 (§11) |
| Q4 | Does pretraining fix the trained models' OOD collapse? | Not tested in v0.4: the training baselines were dropped before G1 (§11) |
| Q5 | Is a WSP edit explainable and small? | Not tested in v0.4: the counterfactual check (§10.4) needs the sealed counterfactual split (D15). The size of every edit, and which of its claimed mechanisms are eligible for the check, are reported |

### 2.2 Arms

Every Qwen arm starts from the same checkpoint (D1) and uses the verbaliser readout of §5, so they differ only in the weights.

| Arm | Built by | Labelled data | Answers |
|---|---|---|---|
| **Qwen-0** | Nobody: the released weights and the frozen prompt | 0 | Q1; the base every edit is compared with |
| **WSP-A** | The weight author edits the weights in a sealed session; forward passes only on its own texts | 0 | Reported against Qwen-0 and v0.3 A; size |
| **WSP-B** | One revision of WSP-A after the same 200 disclosed windows as v0.3 Phase B (§14.1) | 200 windows | Q2 (vs v0.3 B); reported against Qwen-0; size |
| **WSP-F** (separate column) | WSP-B plus a labelled forward-pass budget on dev with accept or reject, as in the blog's Exp 8 (§14.2) | 200 windows + the budget | Matches the published method; reported beside WSP-B, never instead of it |
| v0.3 and cited | v0.3 A / B / C and the constant priors, re-scored on val; v0.2 columns and the rules engine, cited | as reported | Baselines |

The v0.3 reference policy (H1) and H2 fidelity have no v0.4 counterpart: Qwen's edited weights have no written reference policy. Q5 replaces H2.

---

## 3. Target model (frozen) [packet]

**Qwen3.5-0.8B** (`Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`), text model only. The vision encoder and the multi-token-prediction head (`mtp.*`) are not used and are not loaded. `packet/qwen/config.json` is the checkpoint's own config; its `text_config` is what runs.

- **Precision.** The checkpoint stores 284 tensors in bf16 and 36 in fp32. The model runs on an exact fp32 upcast of them, and every edit is fp32.
- **Size.** 320 text-model tensors, `model.language_model.*`, 752,393,024 parameters. Hidden size `d = 1024`. Vocabulary 248,320.
- **Embedding and head.** `embed_tokens.weight` `[248320, 1024]` is **tied** to the LM head (one tensor, no bias): the logit of token `v` at a position is `E[v] · h`, where `h` is the final-normed residual.
- **Layers.** 24 decoder layers, numbered 0–23. Each is pre-norm and residual:
  `h ← h + mixer(input_layernorm(h))`, then `h ← h + mlp(post_attention_layernorm(h))`.
  The mixer is **Gated DeltaNet** (`linear_attention`) in layers 0–2, 4–6, 8–10, 12–14, 16–18 and 20–22, and **full attention** in every fourth layer: 3, 7, 11, 15, 19 and 23.
- **Norms.** RMSNorm with `eps = 1e-6`. `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm` and the final `norm` are **zero-centred**: `x · rsqrt(mean(x²) + eps) · (1 + w)`. The DeltaNet output norm (`linear_attn.norm.weight`, `[128]`) is a gated RMSNorm, `rmsnorm(x) · w · silu(z)`, per value head.
- **MLP** (every layer): SwiGLU, `down_proj(silu(gate_proj x) ⊙ up_proj x)`; `gate_proj`, `up_proj` `[3584, 1024]`, `down_proj` `[1024, 3584]`; no biases.
- **Full-attention layers:** 8 query heads and 2 key/value heads (grouped: query heads 0–3 share KV head 0, 4–7 share KV head 1), head dimension 256.
  - `q_proj` `[4096, 1024]`: per query head, 256 query rows followed by 256 output-gate rows.
  - `k_proj`, `v_proj` `[512, 1024]`; `q_norm`, `k_norm` `[256]` applied per head before RoPE.
  - Rotary position embedding on the first 64 of each head's 256 dimensions (`partial_rotary_factor` 0.25, `rope_theta` 1e7); the other 192 carry no position.
  - Causal softmax attention with scale `256^-0.5`. The head outputs are multiplied by `sigmoid(gate)` and projected by `o_proj` `[1024, 2048]`.
- **Gated DeltaNet layers:** 16 key heads and 16 value heads, each of dimension 128.
  - `in_proj_qkv` `[6144, 1024]`: rows `[q (2048) | k (2048) | v (2048)]`, followed by a causal depthwise convolution (`conv1d.weight` `[6144, 1, 4]`, kernel 4, no bias) and SiLU.
  - q and k are L2-normalised per head; q is scaled by `128^-0.5`.
  - `in_proj_b` `[16, 1024]` gives the write strength `β = sigmoid(b)` per head and token; `in_proj_a` `[16, 1024]` with `A_log` `[16]` and `dt_bias` `[16]` gives the decay `g = −exp(A_log) · softplus(a + dt_bias)`.
  - Each head keeps a 128 × 128 state `S`. Per token: `S ← exp(g) S`, then the delta rule `S ← S + β k (v − Sᵀk)ᵀ`, and the output is `Sᵀq`.
  - `in_proj_z` `[2048, 1024]` is the output gate of the gated norm; `out_proj` `[1024, 2048]` writes back to the residual.
- **No absolute position embedding.** Position enters only through RoPE in the six full-attention layers and through order in the DeltaNet state.

The kernel (§7) serves this metadata as `ARCH`, `NAMES` and `SHAPES`.

---

## 4. Log rendering (frozen) [packet]

The log text is the frozen **v0.3 renderer**'s output (`packet/renderer.py`), unchanged. The renderer produces one **event block** per tool call; each block is one piece of the prompt (§5). The token budgets in this section are counted in the renderer's own **word-level tokens** (§4.7), not in Qwen tokens.

### 4.1 Layout

A session's log is one **event block** per event, in order. Each block is a header line plus up to 6 body lines, joined by newlines, all ASCII (non-ASCII characters become `?` before any marker is inserted). The truncation marker is the ASCII word `TRUNC`.

### 4.2 Header line

```
{HH:MM:SS.mmm} {tool} {fields…} -> {status} in={input_tokens} out={output_tokens}
```

| Tool | Fields (budget in tokens) | Status |
|---|---|---|
| `read_file` | `path={p}` [`lines={a}-{b}`] | `ok` / `error` |
| `search` | `scope={s} query="{q}"` (query ≤ 12) [`limit={n}`] | `ok` |
| `list_dir` | `path={p}` | `ok` |
| `edit_file` | `path={p}` | `ok` / `error` |
| `write_file` | `path={p}` | `ok` / `error` |
| `revert` | `paths={p1},{p2}…` (≤ 16) | `ok` |
| `run_tests` | `suite={s}` [`tests={t1},{t2}…`] (≤ 24) | `ok` or `FAILED` |
| `build`, `lint` | `target={t}` (`?` if the suite was dropped) | `ok` or `FAILED` |
| `http_call` | `{METHOD} https://{host}{path_template}` [`body={json}`] (body ≤ 16) `req={hex8}` | `ok` / `error` |
| `poll_status` | `resource={r}` [`backoff={b}`] | `ready` / `pending` |
| `plan`, `ask`, `yield` | `action={action_type}` | `ok` |

- **Validation counts.** For any event with a validation result, the status is followed by `{passed}/{total} passed`.
- **Field budgets.** A field over its budget is cut and ends with `TRUNC`.
- **The suffix is never cut.** The header's suffix is `-> {status} … in= out=`.
- **Header limit.** The whole header is at most 64 tokens, `TRUNC` included.

### 4.3 Body lines

These are emitted in this order, at most 6 lines in total:

1. **Hunk lines.** One line per changed file, in path order:
   ```
     @ {path} {h_old} > {h_new}
   ```
   - The diff is `difflib.unified_diff(before.split("\n"), after.split("\n"), n=0)`.
   - `h_x` is `h` plus the first 10 hex characters of `sha256(session_id + ":hunk:" + path + ":" + text)`, where `text` is the file's removed lines (for `h_old`) or added lines (for `h_new`), taken across all hunks in diff order and joined with `\n`.
   - An empty side renders as `h0`.
   - The same change in the same file and session always gets the same ids. An undo swaps them, and a return to earlier content reuses an earlier id.
   - The number of hunks inside a file is not shown. At most `min(5, 6 − tail lines)` file lines are shown. If more files changed, the last shown line is `  @ TRUNC {k}`, where `k` is the number of files not shown.
   - Hunk lines are still shown when workspace digests were dropped.
2. **Error line** (ERROR outcome only):
   ```
     ! {error_class} at {error_location}:{line} ({noise})
   ```
   - `{line}` is `int(sha256(session_id+":"+error_location)[:4],16) % 900 + 1`, so it is stable across repeats.
   - `{noise}` is `pid {n}` or `0x{hex8}`, drawn from the event's render RNG.
3. **Workspace line** (`[workspace untracked]`), emitted when either digest was dropped.

### 4.4 Missing observations

Each dropped field renders as `?` in its own position, for example `in=? out=812`, `?/18 passed`, `suite=?`, or `! ? at ?:? (…)` when the failure was dropped.

### 4.5 Noise

- **Render RNG.** Each event has its own render RNG, `random.Random(int(sha256(canonical_json(RawOp) + ":render")[:16], 16))`. It supplies `req=` and `{noise}`.
- **Timestamps.** These are parsed with `datetime.fromisoformat` and rendered in UTC as `HH:MM:SS.mmm`.

### 4.6 Bound

A block has at most 64 header tokens, then for each of up to 6 body lines at most 21 tokens (a line marker plus at most 20), 190 word-level tokens in all. The renderer enforces the per-line cap with `TRUNC`.

### 4.7 Word-level tokens (`packet/tokenizer.py`)

The budgets above count the tokens of the frozen v0.3 tokeniser, which the renderer imports for this purpose only. It scans a block left to right, dropping spaces and tabs: a newline followed by `@`, `!` or `[` is one line-marker token; any other newline is one token; a word (`[A-Za-z_][A-Za-z0-9_.\-/]*`) is one token; a run of digits is split left to right into groups of three, one token each; any other printable character is one token. `packet/GRAMMAR_SHEET.md` shows these tokens for one invented block per tool. It is v0.3's sheet, unchanged: its `BOS`, `EOE`, `event_ord` and network-readout sentences describe v0.3's model and do not apply here; only the block text and the word-level truncation budgets matter. Block strings sent to `./fwd` contain no `EOE`.

**Qwen does not read these tokens.** Qwen reads each block's text through its own tokeniser (§5). The word-level tokens only fix where the renderer truncates.

---

## 5. Task interface (frozen) [packet]

The interface is fixed. `packet/INTERFACE.md` gives the exact strings and token ids, generated from the harness's definition; where it and this section differ, `INTERFACE.md` is right.

- **Interface name:** `w1-state-full`. **`spec_sha256`:** `5922eb4500122921ac7d07e3d35599b95cb10b462c4f50b89e5d6bb989437879`: the sha256 of the JSON object `{"block_template", "label_words", "prefix", "suffix"}` (sorted keys, `ensure_ascii=False`).
- **Prefix**, once at the start of every session: Qwen's chat format, a system message holding the instruction and the four class definitions, then the opening of the user turn. As a JSON string:
  ```
  "<|im_start|>system\nYou monitor a coding agent. You will read the log of its tool calls, one block per call: a header line (time, tool, arguments, result, token counts) and up to six indented lines for changed files (opaque content ids before > after), errors and untracked workspace notes.\nAfter the log, give the agent's state at the most recent tool call as one word:\nproductive: evidence of forward progress or materially new investigation.\nrecovering: a bounded retry or repair after a transient or diagnosed failure, still within policy or recently resolved.\nstalled: repeated ineffective operations, with no observed progress and no state-reversion cycle.\nlooping: recurring reversal between previously held workspace or execution states, with no net progress.<|im_end|>\n<|im_start|>user\nTool-call log:\n"
  ```
- **Block template:** `"{block}\n"`. Each rendered block (§4), followed by one newline, is one piece.
- **Suffix**, after the last block of a window: the end of the user turn, then the assistant turn with thinking switched off, then the answer cue. As a JSON string:
  ```
  "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nState:"
  ```
- **Label words**, in class order (index = logit index):

  | Index | Class | Label word | Qwen token id |
  |---|---|---|---|
  | 0 | PRODUCTIVE | ` productive` | 25399 |
  | 1 | RECOVERY | ` recovering` | 39923 |
  | 2 | STALLED | ` stalled` | 65552 |
  | 3 | OSCILLATING | ` looping` | 60477 |

  Each label word is a single token with its leading space. None of the four ids occurs in the prefix or the suffix.
- **Tokenisation.** The prefix, each piece and the suffix are tokenised **separately** with Qwen's tokeniser (`packet/qwen/tokenizer.json`, no special tokens added) and concatenated as token ids. The prefix is 167 tokens and the suffix 11. Tokenising pieces separately fixes the token stream at block boundaries.
- **Readout.** The prompt for the window ending at block `t` is `prefix + piece_0 + … + piece_t + suffix`. The four **label logits** are the next-token logits of the four label ids at the suffix's last token (the `:` of `State:`), that is `E[label_id] · h`, with `h` the final-normed residual there. The **class** is the argmax over the four label logits; ties go to the first index (all-equal logits mean PRODUCTIVE). The other 248,316 tokens of the vocabulary do not affect the class.
- **One pass per session.** The session is run once: prefix, then blocks in order. After each block that ends a window, the model's cache (the full-attention keys and values and every DeltaNet convolution and recurrent state) is branched, the suffix is run on the branch and read, and the branch is thrown away. The result equals a separate prompt per window up to floating-point reassociation, and the label after block `t` never sees block `t+1`.
- **Frozen.** The prefix, the block template, the suffix, the label words and the readout are fixed. Only weights may change.
- **Precision.** Scoring runs the same readout in fp32. It may run on a different device from your forward passes, so a class decided by a margin of a few thousandths of a logit is not reliable.

---

## 6. Windows, gate, labels

These are unchanged from v0.3 §6 (v0.2 §5–§6). Windows, the scored mask, labels and populations are v0.3's, row for row: the readout after block *t* is the prediction for the window ending at *t*. Every Qwen column is scored on exactly the 87,176 val windows the v0.3 report used (a quick look on the windows of its 1,000 sessions). No other split is scored in v0.4 (§9).

**[packet]**

- **Windows.** The class at block *t* describes the window of the agent's most recent eight operations ending at block *t* (fewer at the start of a session).
- **Oracle rules.** The packet carries v0.2 §6.1–§6.3 (latent state definitions, parameters, rules) and the family names in v0.2 §7.1, as the v0.2 and v0.3 packets did.
- **Gate rule.** Windows with withheld observations are not scored.

---

## 7. Weight-space programming protocol [packet]

`packet/AUTHOR_INSTRUCTIONS.md` is the operational protocol: exact commands, function signatures, request and response formats. This section states the rules.

### 7.1 Two processes

- **The model server** holds the model (fp32), your edit state, the forward-pass meter, the edit journal and the ledger. It runs outside your directory, belongs to the harness, and never runs your code. The edits, the meter and the journal persist across your scripts and across continuations of your session.
- **Your scripts** run in short-lived, guarded Python interpreters (`./py`). They can read weights and save edits through the kernel client (`from wsp import …`). They cannot run the model, and they cannot reach the forward socket.
- **Forward passes are a separate tool,** `./fwd request.json`, one request per Bash call. Each request is metered, logged and stamped with the edit-state hash. So "try an edit, run the model, keep the best" can never be a loop inside a script: every model run is a visible tool call.

### 7.2 Kernel operations (`./py` scripts)

- **Metadata:** `NAMES` (the 320 tensor names), `SHAPES`, `ARCH` (sizes, layer types, head counts, `rms_norm_eps`, the label ids, the edit scope, the load quota), `LABEL_IDS`, `CLASS_NAMES`, and `tok` (Qwen's tokeniser, from `packet/qwen/tokenizer.json`).
- **Reading:** `load(name)` (your edit if there is one, else the original), `load_original(name)`, `label_rows(original=False)` (the four label rows, `[4, 1024]`).
- **Editing:** `save_edit(name, tensor)` replaces a whole tensor; `save_label_rows(rows)` replaces only the four label rows; `drop_edit(name)` restores one tensor; `edited()`; `edit_hash()`. A save returns a receipt: changed entries, rows and columns, `‖ΔW‖_F / ‖W‖_F`, the rank of `ΔW`, the bf16-rounded changed count, the tensor's sha256 and the new edit hash. Saving a tensor equal to its original removes that tensor's edit, so the edit hash depends only on the effective weights.
- **Analysis helpers** (plain torch, weight-only; none saves anything): `svd`, `head_slice`, `top_writers`, `project_out`, `rank1`, `zero_head`, `label_lens`, `dla`, `diff_means`, `control_direction`.
- **Weight-load quota.** `load` and `load_original` of a layer tensor (anything but the embedding and the final norm) draw on a per-phase quota of distinct tensor bytes: half of the layer tensors' bytes, 996,224,640 bytes. Loading a tensor again is free. The quota exists so that scripts cannot assemble the whole model and run unmetered forward passes; plan which tensors you need.
- **Budget:** `budget()` gives `{tokens_used, tokens_left, requests}`.

### 7.3 Forward passes (`./fwd`)

- A request names the weights (`edited`, `original` or `both`), gives one or more sessions as lists of rendered block strings with the block indices that end a window, and optionally asks for hidden states (layers 0–24: the residual after that many layers, 0 = embeddings, 24 = before the final norm), MLP activations (layers 0–23: the input of `down_proj`) and the top-k token ids (k ≤ 20) at each readout position.
- Texts run through exactly the readout of §5: the frozen prefix and suffix, the same tokenisation, one pass per session.
- The response gives, per window: the four label logits, the full-vocabulary log-normaliser, the class (label argmax), the full-vocabulary argmax token id and, if asked, the top-k ids (ids only, never vocabulary strings). For `both` it adds the edited-minus-original deltas and the number of class flips. Requested activations are written as `.npy` arrays in your output directory.
- **Cost** is the number of tokens run: per session the prefix, the blocks up to its last window, and one suffix per window; doubled for `both`. It is charged when the request completes.

### 7.4 Edit scope and refusals

- **In scope:** the 320 `model.language_model.*` tensors, with two exceptions:
  - `model.language_model.embed_tokens.weight`: only the four label-word rows (`LABEL_IDS`) may change, through `save_label_rows` (or a `save_edit` that changes no other row).
  - `model.language_model.norm.weight` (the final norm) is refused: scaling it is a global temperature knob.
- **Refused:** `mtp.*` and the vision encoder (not loaded); non-finite values; a shape or dtype mismatch (edits are fp32); and **rescalings**. With `new = c·W + R` and `c = ⟨new, W⟩ / ⟨W, W⟩`, an edit is refused when `|c − 1| > 1e-3` and `‖R‖ < 0.5 · ‖(c − 1)·W‖`: an exact multiple, a multiple plus a small change, or zeroing a tensor (`c = 0`). For the zero-centred norms (`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`) the same test also runs on the effective gain `1 + w`. For the label rows it also runs on the row-centred `[4, 1024]` block and on the changed rows when two or more changed, so a common rescaling of the rows, with or without a shared offset, is refused. Changing a single label row's scale is an edit to that class's readout direction and is allowed.
- **Tied label rows.** The four label rows are both the readout directions and the input embeddings of those four tokens. They do not occur in the frozen prefix or suffix, so the input-side effect reaches only texts that contain the label words themselves. The build record flags every label-row edit.
- A refused save changes nothing.

### 7.5 Budget, meter and ledger

- **Phase budget:** 2,000,000 tokens of forward passes. **Per request:** at most 50,000 tokens. A request that would exceed either is refused before it runs and costs nothing; so does an invalid request.
- The model runs at roughly 450 tokens per second, plus a fixed cost of tens of milliseconds per window, so the whole budget is a little over an hour of model time.
- **Ledger.** Every request that reaches the model server, run or refused, gets a ledger line in the harness: call id, time, tokens, cumulative tokens, the edit hash, the sha256 of each session's token ids and windows, the layers requested and the request file. Cite call ids in `EDIT.md` and `ITERATIONS.md`. A request file `./fwd` rejects as invalid (exit status 2) is not sent: it costs nothing, has no call id and appears only in your session log.
- **Audit.** The harness reviews the ledger, every script you ran and the edit journal. It counts how many distinct edit states were run on the same set of texts; more than three goes to the adjudicator as possible search.

### 7.6 Method: banned and allowed

**Banned:**

- backward passes, autograd and optimisers;
- fitting weights to a loss, in closed form or iteratively (regression, least squares, pseudo-inverse fits, gradient steps);
- any automated or numerical search over candidate weights, including keep-the-best loops, random restarts and parameter sweeps scored against an objective, whether the loop is a script or a sequence of tool calls;
- distillation of any model's outputs into the weights.

**Allowed:** direct, reasoned edits, each with a stated mechanism. For example:

- a rank-1 update along a direction you have identified;
- projecting a direction out of a matrix;
- writing a readout direction into the label-word rows;
- reading weights, SVDs, attributions and activations of your own texts, and deriving directions from them (means, differences of means, projections), then testing an edit with a few forward passes and revising it by reasoning.

A loop that tries candidate weights and keeps the best is search.

### 7.7 WSP-A (zero labels)

- **Data:** none. Forward passes run only on texts you write yourself: hand-written block text, or operation records you build and render with `packet/renderer.py`. You receive no labelled data, no dataset and no score.
- **Cost cap:** US$500 of session spend for the whole phase, continuations and build-rejection resubmissions included.
- **Result:** the first valid artefact you submit as final.

### 7.8 Later phases

A one-revision phase (WSP-B) may follow in the same directory, with its own instructions, budget and cost cap. It starts only if you are told it has.

---

## 8. Deliverables and build checks [packet]

All in `output/wsp-a/` (WSP-A):

1. **`build_edit.py`** defines `def build(orig: Mapping[str, Tensor], consts: Mapping[str, Tensor]) -> dict[str, Tensor]`. `orig` maps every tensor name to its **original** fp32 weights (each access hands out a fresh copy); `consts` holds the optional constants file. It returns **only the replaced tensors**, each full-size, fp32, finite and different from the original; the embedding, if edited, is returned whole with only its four label rows changed. It runs under the build guard: matrix products and `torch.linalg` are allowed (except least squares and pseudo-inverses); calling any model or `nn.Module`, `torch.nn.functional`, softmax, random numbers, autograd, optimisers, threads, subprocesses, sockets and file access are refused. At most 200 KB; at most 300 s of CPU on one thread.
2. **`edit_constants.safetensors`** (optional, at most 8 MiB): named fp32 tensors, such as directions derived from forward-pass activations. `EDIT.md` must cite, for each constant, the ledger call ids it was derived from.
3. **`EDIT.md`**: what was changed, in which tensors and why; the mechanism claimed for each change, stated so that it can be checked; the ledger call ids behind every vector. It ends with a section headed exactly `## Mechanisms`: one Markdown table with the columns `id | tensors | target class | claim`, in that order, one row per claimed mechanism. `tensors` lists the full names of the edited tensors that implement it (comma-separated), or `label rows` for the four label-word rows; `target class` is one of the four class names; `claim` is one sentence. Each row is checked on its own by reverting exactly its tensors to the original weights. A row is checked only if every tensor it names is edited, it does not name the whole edit and it shares no tensor with another row, so an edit of a single tensor leaves no row that can be checked.
4. **`ITERATIONS.md`**: every iteration in order: the edit hash, a summary of the change, the reason, and the forward-pass call ids that tested it with what they showed.

**Build checks.** When you state that a submission is final, the harness:

- runs `build()` from a private copy of `build_edit.py` and the constants, alone, in a fresh interpreter on CPU (one thread, deterministic algorithms);
- validates the result against the edit scope and refusals of §7.4;
- rebuilds it in a **second** fresh interpreter with a different hash seed and requires **byte-identical** tensors;
- requires the build to **equal the model server's final journalled edit state**, tensor for tensor and byte for byte. Whatever you last saved through the kernel is what `build()` must reproduce, and vice versa;
- archives the edit, an exact diff, a build record (per tensor: changed entries, rows and columns, `‖ΔW‖_F / ‖W‖_F`, rank and bf16-rounded changed count; the overall changed fraction; the label-row flag) and your whole phase directory.

A rejected build returns only the harness's error text.

---

## 9. Data, splits and sealing

- **Training and disclosure data are unchanged.** Dev (12,000 sessions) trains no v0.4 arm: only WSP-F reads its labels, in aggregate and on the holdout (§14.2). Val (4,004 sessions) is for harness-only validation, with the 200 disclosed sessions left out of the primary population.
- **Task export** (`scripts/v04_export_tasks.py`, root environment). `datasets/v0_4/{split}.tasks.jsonl.gz` holds rendered blocks and window positions with no labels; `datasets/v0_4/{split}.labels.npz` holds the window rows, built exactly as v0.3's scoring rows. Only dev and val can be exported; v0.4 exports no other split.
- **Fresh sealed test set** (D7, `experiments/v0_4/fresh_test/REGISTRATION.json`, `datasets/v0_4/fresh_test/MANIFEST_fresh_test.json`): 86,412 windows, 83,874 gate-passing. It was rebuilt byte-identically under a second PYTHONHASHSEED. The manifest holds no label statistics. It is **not scored in v0.4** (D15) and stays sealed for a later version.
- **v0.3 test, OOD and counterfactual** stay sealed and are not scored in v0.4 either. The counterfactual split, which v0.3 scored before its sealed attempt, is sealed here.
- **Sealing.** Nobody opens, lists or decompresses `datasets/**/test.*`, `datasets/**/ood.*`, `datasets/**/counterfactual.*` or anything under `datasets/v0_4/fresh_test/` in v0.4; the scoring bridge only re-hashes the fresh-test files' bytes against their committed manifest. Val labels and dev labels are read only by harness scoring code. No val number is ever shown to the weight author. No Qwen output is compared with a label before G1.
- **The result is on val** (D15): `excluding_disclosed` is primary, `all_val` beside. **Quick look:** `datasets/v0_4/VAL_QUICK.txt`, the 1,000 val sessions with the smallest `sha256("20261106:v04-val-quick:{session_id}")` after the 200 disclosed sessions are removed, chosen before any edit existed. Every edit is scored on it first, paired against Qwen-0, then on full val; the full-val numbers are the result of record. Val is not a sealed set (the harness reads models out on it, v0.3 used it for early stopping and the Phase B disclosure), so the v0.4 result is weaker evidence than a sealed one. A later version may score the archived v0.4 edits on the sealed sets under its own pre-registration.

---

## 10. Scoring

### 10.1 Bridge

`src/agent_loop_lab/v04/scoring.py` turns Qwen logits into the v0.3 report. Val rows are checked against v0.3's `rows_sha256`; columns are aligned strictly; metrics come from `eval.m4.score_columns`, as v0.3 Phase C. H2 is not applicable. The constant priors and the v0.3 networks re-scored on val (`scripts/v04_v03_val_columns.py`, exact against the v0.3 report) join the same call, and the bridge reports paired macro-F1 differences over the shared bootstrap draws. A column read out on a subset is accepted only when its subset is `VAL_QUICK.txt` by sha256 and its sessions are exactly that file's; all Qwen columns of one report cover the same sessions.

### 10.2 Metrics

v0.2 §10.1 as used by v0.3: macro-F1, per-class precision and recall, false-loop alarm rate, confusion and session-bootstrap CIs (counterfactual consistency is not computed in v0.4: it needs the sealed counterfactual split). Also: tokens per session, readout latency, and for WSP the changed fraction of weights and the per-tensor relative Frobenius norm.

### 10.3 Backend

Every Qwen run is local and fp32 on this Mac's MPS: the harness-only val runs and the author's forward server, fixed at G1. torch's deterministic-algorithms mode is no guarantee on MPS, so the harness also replays 100 val sessions on this Mac's CPU for Qwen-0 and WSP-B and reports how many argmaxes agree with MPS.

### 10.4 Q5 mechanism check

For each mechanism `EDIT.md` claims, the check runs on the counterfactual pairs with the edit applied and then ablated (the claimed tensors reverted to the original). The pass statistic is pre-registered at G1, but it is **not run in v0.4** (D15): the counterfactual split stays sealed. v0.4 reports each edit's size and which claimed mechanisms are eligible, with no Q5 verdict.

### 10.5 Validation before WSP-B

Val scores of WSP-A are computed for the harness only and are never shown to the author: first the quick look on `VAL_QUICK.txt`, then full val. The first archived WSP-A artefact is the WSP-A result; a later relaunch is a new registered version, reported alongside the first with the number of prior launches.

---

## 11. Trained baselines

Removed before G1 (Matthew, 2026-11-06: "Drop the training baselines"). There is no Qwen-probe and no Qwen-LoRA column at any budget, and no cloud GPU. Q3 and Q4 are not tested. The trained columns of the report are the v0.3 C columns, re-scored on val, and v0.2 C-all, cited (PREREGISTRATION §1). The heading stays so that section numbers do not move.

---

## 12. Compute

- **Forward server** (WSP): local MPS, fp32, about 445 tok/s. The 2M-token budget is about 75 min of model time.
- **Harness-only val runs:** local MPS; one full val pass takes about 5 h (7.87M readout tokens; 1,726 tokens per session with the prefix), a quick look on the 1,000-session subset about a quarter of that.
- **v0.3 columns on val:** this Mac's CPU, in the frozen v0.3 environment.
- **Sealed scoring:** none in v0.4 (D15).
- **CPU replay check:** this Mac's CPU, fp32.
- **Nothing is rented** (Matthew, 2026-11-06).
- **Build runner:** CPU, one thread, deterministic algorithms, RLIMIT_CPU 300 s, wall timeout 600 s.

---

## 13. Roles, isolation and session harness

| Role | Who |
|---|---|
| Harness implementer and adjudicator stand-in | This Claude Code session and its workflow agents |
| Weight author | D9. It holds only the v0.4 packet |

**Packet** (`scripts/v04_build_packet.py`, `src/agent_loop_lab/v04/packet.py`; the packet sha256 is the sha256 of `PACKET_MANIFEST.json`):

- `SPEC.md`: the [packet] sections of this spec, then v0.2 §6.1–§6.3 and the §7.1 family names, exactly as v0.3's packet carried them;
- `AUTHOR_INSTRUCTIONS.md`, `INTERFACE.md` (generated from `hypaqwen/interface.py`), `GRAMMAR_SHEET.md` (v0.3's, byte-identical);
- `renderer.py`, `tokenizer.py` (verbatim v0.3);
- `qwen/config.json`, `qwen/tokenizer.json`, `qwen/tokenizer_config.json` (verbatim from the pinned snapshot, sha256-checked). No weights: the kernel serves them.
- No generator, oracle, dataset, scoring or report content; no dataset paths; no import of the harness packages. A leakage check (v0.3's sentinels and generator words plus v0.4's) refuses the build otherwise.

**Author directory** (`../HypaNet-v04-author`, created fresh, never reused): `packet/` and `tools/` read-only; `tools/hn_kernel/` (client, `./fwd`, protocol, launcher) and `tools/hn_guard/` installed and verified by the harness; `py` and `fwd` wrappers (0555); the author venv (`envs/qwen/scripts/make_author_venv.sh`); `.kernel/` sockets; `output/wsp-a/` the only writable tree.

**Launch** (`scripts/v04_launch.sh`): `claude -p` with the tool allow-list `Read,Glob,Grep,Write,Edit,Bash`, `--strict-mcp-config` with no MCP servers, `--permission-mode dontAsk`, a PreToolUse hook that rejects any Bash command not matching `^\./(py|fwd) [^;&|<>`$\n]+$`, and `--max-budget-usd` set to the remaining cap. Deny-lists cover the repo, every sibling `HypaNet*` directory, harness scratch and Claude project data; no web, MCP, subagents or skills; auto-memory off.

**Cost.** By session id, as the delta of the cumulative `total_cost_usd` between launches of that session; no new launch once the phase's sum reaches its cap.

**Audit:** forbidden markers (the repo, `envs/qwen`, `datasets`, `.cache/huggingface`, `experiments/v0_4`); an AST rule flagging a loop body that calls `save_edit` or `drop_edit` and writes a `./fwd` request; the ledger cross-check (budget; distinct edit hashes run on identical text sets, more than three to the adjudicator); the kernel tools' and guard's hashes.

**Logs:** `experiments/v0_4/logs/` holds the server ledger `<phase>.ledger.jsonl`, the kernel log `<phase>.kernel.jsonl`, the journal `<phase>/journal/` and the session transcripts.

---

## 14. WSP-B and WSP-F

### 14.1 WSP-B (one revision)

The same 200 windows and oracle labels as v0.3 Phase B (the selection rule never looks at a model's output). `network_class` comes from WSP-A's verbaliser; `reference_policy_class` is dropped. Format tag `v04-wsp-b-disclosure/1`. The server's `forward_disclosed` returns label logits and the argmax only on those windows (no activations or top-k), for at most two distinct non-empty edit states; `forward` refuses any session overlapping a disclosed window. One submitted revision, `REVISION.md` citing the disclosed window ids. Cost cap US$250, forward budget 2M tokens.

### 14.2 WSP-F (blog-style, separate column)

WSP-B plus 50 `score_dev` calls of at most 128 dev sessions each, with accept or reject. A final paired check on 500 dev sessions picked by hash key, outside the v0.3 Phase C 2,048-session budget subset, held by the harness (`holdout_check`, once). Cost cap US$250. Dev labels live only in the WSP-F server. The WSP-F archive is the last v0.4 artefact; no sealed evaluation waits on it (D15).

---

## 15. Verification

- **Interface:** `spec_sha256` equals D4; the packet's `INTERFACE.md` is regenerated from `interface.py` and its token ids match the pinned tokeniser.
- **Readout:** one pass equals a separate prompt per window on MPS and CPU; label-row edits reach the readout.
- **Kernel:** every §7.4 refusal; the meter refuses over budget and over the per-request cap; the journal and ledger replay on restart; `./py` cannot reach the forward socket.
- **Guards:** one test per banned path on torch 2.14.1.
- **Build runner:** byte-identical rebuild under two hash seeds; journal equality; a fake `build_edit.py` round trip.
- **Packet:** builds twice byte-identically; leakage refusal; qwen files' sha256.
- **Kit tests:** non-kit tools absent from the init tool list; the hook rejects compound commands.

---

## 16. Milestones

| Milestone | Contents |
|---|---|
| **M0** | Environment, readout, token audit, fresh sealed test set, prompt choice, scoring bridge, WSP kernel |
| **G1** | Pre-registration frozen and committed: this spec, the interface, the WSP rules, the Q5 statistic (not run in v0.4), the quick-look subset, the backend (local MPS, fp32), the packet hash (`registry_v0_4.yaml`) |
| **M1** | Qwen-0 on val (quick-look subset and full), harness-only |
| **M2** | WSP-A: sealed session; edit archived; quick look, then full val; the 200-window disclosure refreshed |
| **M3** | WSP-B, then WSP-F; each archived, then quick look and full val |
| **G2** | Removed (D15): no sealed evaluation waits on the archives |
| **M4** | Removed (D15): no sealed evaluation in v0.4; the sealed sets stay sealed for a later version |
| **M5** | Report on val: one joint table with v0.3 (re-scored on val) and v0.2 (cited) |

A poor result is still a result, provided it is attributed to the interface, the edit, capacity or distribution shift.
