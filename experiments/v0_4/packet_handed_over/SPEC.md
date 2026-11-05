# Agent doom-loop detector, spec v0.4: authoring specification (weight-space programming)

This file holds the sections of the v0.4 specification that are given to the weight author,
followed by the oracle rules and scenario-family names of spec v0.2, which v0.4 keeps unchanged.
`AUTHOR_INSTRUCTIONS.md` is the operational protocol; `INTERFACE.md` gives the frozen prompt and its
token ids; `GRAMMAR_SHEET.md` shows the log grammar with invented examples.

## 1. Objective

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

## 3. Target model (frozen)

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

## 4. Log rendering (frozen)

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

## 5. Task interface (frozen)

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

## 6. Windows, gate, labels

- **Windows.** The class at block *t* describes the window of the agent's most recent eight operations ending at block *t* (fewer at the start of a session).
- **Oracle rules.** The packet carries v0.2 §6.1–§6.3 (latent state definitions, parameters, rules) and the family names in v0.2 §7.1, as the v0.2 and v0.3 packets did.
- **Gate rule.** Windows with withheld observations are not scored.

## 7. Weight-space programming protocol

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

## 8. Deliverables and build checks

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

## v0.2 oracle rules and scenario families (unchanged in v0.3 and v0.4)

The labels are produced by the v0.2 oracle from the simulator's latent state. Its definitions,
parameters and rules (v0.2 §6.1-§6.3) and the names of the scenario families (v0.2 §7.1) are
reproduced here. "Encoded features" and "the 120-vector" refer to v0.2's input; v0.4, like v0.3,
gives the model the rendered log instead.

#### 6.1 Latent state (simulator-internal, never encoded)

Per session the simulator maintains:

- `progress p_t ∈ [0, 1]`: hidden objective completion.
- `knowledge K_t`: a set of latent facts about the fault. An operation may add facts. A new fact is "material narrowing".
- `workspace W_t`: real simulated file contents. Digests are computed from these.
- `episodes`: failure episodes with latent cause (`transient`, `persistent`, `diagnosed_repair`), a retry counter and a resolution flag.
- Per-operation latent effect: one of `PROGRESS`, `NEW_FACT`, `RESOLVES_EPISODE`, `IN_POLICY_RETRY`, `IN_POLICY_POLL`, `REVERSION`, `REDUNDANT`, `NONE`.

`REDUNDANT` means the operation repeats work already reflected in latent state. It re-acquires a fact already in `K_t`, re-applies an edit already present in `W_t`, re-runs validation on an unchanged workspace, or is an out-of-policy retry or poll. Redundancy is defined against `K_t` and `W_t`, **not** against `intent_fingerprint`, so it stays independent of the encoded features. `NONE` means the operation is novel but yielded nothing, for example a first read of a file that turns out to be irrelevant.

An episode's **initial failing operation** has effect `NONE`, because the attempt itself was reasonable. Its in-policy retries are `IN_POLICY_RETRY`. Retries beyond `R_max` are `REDUNDANT`.

An operation is **stall-ineffective** iff its latent effect is `REDUNDANT` or `REVERSION`. An operation is **ineffective** iff it is stall-ineffective or has effect `NONE`.

A **reversion** is an agent mutation whose latent after-state equals a workspace state held earlier in the window **and** that undoes an earlier agent mutation. External resets don't count.

#### 6.2 Pre-registered oracle parameters

| Symbol | Value | Meaning |
|---|---:|---|
| `ε_p` | 0.02 | Progress tolerance over a window |
| `F_prod` | 2 | Minimum new facts in the window for investigative PRODUCTIVE |
| `N_stall` | 5 | Minimum ineffective operations of 8 for STALLED |
| `N_grey` | 3 | Lower bound of the stall grey zone |
| `N_rev` | 2 | Minimum reversions in the window for OSCILLATING |
| `R_max` | 3 | Maximum retries of one intent within an episode while still in policy |
| `P_max` | 5 | Maximum polls of an external dependency (with backoff) while still in policy |
| `k_rec` | 4 | Recency horizon for RECOVERY |

These values live in `oracle_config.v0_2.yaml` and are frozen at M0.

#### 6.3 Window labelling rules (evaluated in order; first match wins)

Let `Δp` be the progress change across the window, `F` the number of new facts, `I` the number of **stall-ineffective** operations (`REDUNDANT` + `REVERSION`) and `V` the number of reversions.

1. **INSUFFICIENT_EVIDENCE (instrumentation):** the generator deliberately dropped a required observation (scenario 12).
2. **OSCILLATING:** `V ≥ N_rev` and `Δp ≤ ε_p`.
3. **STALLED:** `Δp ≤ ε_p`, `F = 0`, `I ≥ N_stall`, the final event is stall-ineffective, **and** no episode is in policy at the endpoint.
4. **RECOVERY:** an episode is active at the endpoint with retries ≤ `R_max` (or polls ≤ `P_max`), **or** an episode resolved within the last `k_rec` events.
5. **PRODUCTIVE:** `Δp > ε_p` or `F ≥ F_prod`.
6. **INSUFFICIENT_EVIDENCE (ambiguous):** everything else. This includes the grey zone `N_grey ≤ I < N_stall` with no progress, windows with a single new fact and no progress, and aimless novel activity (mostly `NONE` effects). Novel but fruitless exploration is never labelled STALLED. Repetition of already-done work is what defines a stall.

This ordering gives `OSCILLATING > STALLED > RECOVERY > PRODUCTIVE`. The recency bound in rule 4 stops a long-resolved retry from overriding later productive work.

Labels depend on latent causal state. Observable features are emitted independently by the event layer. Labels are never defined in terms of encoded features. For example, there must be no rule of the form "STALLED iff `repeated_intent`".

#### 7.1 Scenario families (names only; this list is in the authoring packet)

1. Incremental productive repair: failing tests, targeted edits, fewer failures, eventually green.
2. Productive exploration: several inspections and diagnostics before a meaningful change.
3. Transient external failure: repeated timeout, then success, then productive work.
4. Bounded backoff or retry: a recoverable failure with one or two controlled retries.
5. Repeated failed test invocation with unchanged artefact and unchanged failure.
6. Repeated edits that change files but don't improve validation.
7. A/B/A/B workspace oscillation between competing edits.
8. Tool oscillation: two operations repeatedly undo each other's effects.
9. Benign lookalike: repeated test runs with genuine incremental improvement.
10. Benign lookalike: repeated reads or polls while an external dependency legitimately recovers.
11. Rising context or token expenditure without useful work. Cost is weak supporting evidence, not a label shortcut.
12. Under-instrumented sessions, for gate coverage.
