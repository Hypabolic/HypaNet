# Agent doom-loop detector, spec v0.3: authoring specification

This file holds the sections of the v0.3 specification that are given to the weight author,
followed by the oracle rules and scenario-family names of spec v0.2, which v0.3 keeps unchanged.
`AUTHOR_INSTRUCTIONS.md` is the operational protocol; `GRAMMAR_SHEET.md` shows the log grammar
with invented examples.

## 1. Objective and hypotheses

At every event of an AI coding agent's session, the model classifies the pattern of the agent's most recent eight operations as one of four classes:

1. **PRODUCTIVE**: evidence of forward progress or materially new investigation.
2. **RECOVERY**: a bounded retry or repair after a transient or diagnosed failure, still within policy or recently resolved.
3. **STALLED**: repeated ineffective operations, with no observed progress and no state-reversion cycle.
4. **OSCILLATING**: recurring reversal between previously held workspace or execution states, with no net progress.

The model reads the session's rendered log. No hand-written feature extractor sits between the log and the network. There is only a fixed renderer (§4) and a fixed tokeniser (§5).

| ID | Claim | Measured as |
|---|---|---|
| **H1 Design** | An LLM can write, without labelled examples, a doom-loop policy over the log that matches the hidden ground truth | `reference_policy.py` against ground truth: macro-F1 and false-alarm rate |
| **H2 Compilation** | The same LLM can compile that policy into the fixed transformer's weights | Argmax agreement between the network and `reference_policy.py` on benchmark windows. Needs no labels |
| **H3 End-to-end** | The hand-programmed transformer is a useful log classifier | Network against ground truth, including out-of-distribution scenarios |

### 1.2 What a good result does and does not show

- **The logs are synthetic, in one fixed grammar.** There is one header line per tool call, with a family-specific tool name and canonical result syntax. Tool output and assistant text are absent, and diff content is replaced by opaque hunk ids. A good score shows hand-programmed perception over this grammar. It does not show robustness to real transcripts.
- **Segmentation is given, not perceived.** `EOE` tokens, the `event_ord` input and an attention bias indexed by event distance make "same event" and "within the last 8 events" one scalar table each. This mirrors transcripts where each tool call is a record. It is a task-aligned inductive bias, and it computes no content signal.
- **The tokeniser is fixed and word-level.** Paths and identifiers are single tokens, and numbers are digit groups. It computes no repeat, state or delta signal.
- **Diff hunks are opaque ids.** Whether an edit repeats, undoes or restores an earlier change is visible. Whether it is correct code is not. In the real world, progress shows only through later validation, and that is the case here too.
- v0.3 makes no claim that hand-programming beats training.

### 2.2 Target model (immutable within a run)

Notation: `V = 10,240`, `d = 128`, `L = 4`, `H = 8`, `d_h = 16`, `m = 512`, `R = 64` event-distance buckets plus one BOS column, `O = 192` offsets.

- **Inputs:**
  - `tokens: int64[n]`;
  - `event_ord: int64[n]` (0 for `BOS`; *k*+1 for every token of the *k*-th rendered event block);
  - `offset: int64[n]`, the 0-based offset within the event block. `BOS` has offset 0, and the tokeniser **raises** if any offset is ≥ `O`.
- **Embedding:** `h⁰ = E[tokens] + P[offset]`, where `E ∈ ℝ^{V×d}` and `P ∈ ℝ^{O×d}`.
- **Each layer ℓ = 1..4**, residual with no normalisation:
  - `q = h W_qᵀ + b_q`, `k = h W_kᵀ + b_k`, `v = h W_vᵀ + b_v`, each `W ∈ ℝ^{d×d}`, split into `H` heads of width `d_h`.
  - `score[h,i,j] = q_{h,i}·k_{h,j} / 4 + B_ℓ[h, c(i,j)]` for `j ≤ i`, and `−∞` for `j > i`. Here `c(i,j) = R` (the BOS column) if `j` is the BOS token, and otherwise `min(event_ord_i − event_ord_j, R−1)`. `B_ℓ ∈ ℝ^{H×(R+1)}`.
  - `a = softmax_j(score)` and `o = concat_h(a_h v_h) W_oᵀ + b_o`; then `h ← h + o`.
  - `h ← h + W₂ ReLU(W₁ h + b₁) + b₂`, where `W₁ ∈ ℝ^{m×d}` and `W₂ ∈ ℝ^{d×m}`.
- **Head:** `logits_i = W_out h_i + b_out` with `W_out ∈ ℝ^{4×d}`. Logits are computed at every position and read only at `EOE` tokens.
- **Classes and ties:** class order `[PRODUCTIVE, RECOVERY, STALLED, OSCILLATING]`. Argmax ties go to the first index, so all-zero logits mean PRODUCTIVE.
- **Parameters:**
  - `E` 1,310,720 + `P` 24,576;
  - per layer: attention 66,048, `B` 520 and MLP 131,712, giving 198,280; four layers give 793,120;
  - head 516.
  - **Total 2,128,932.**
- **Initial state:** all zero. CPU, float32, `torch.no_grad()`.
- **Excluded:** normalisation, dropout, absolute token positions, recurrence, gating and a learned tokeniser.

## 4. Log rendering (frozen)

### 4.1 Layout

The log is `BOS`, then one **event block** per event in order. Each block is a header line plus up to 6 body lines, all ASCII (non-ASCII characters become `?` before any marker is inserted). The truncation marker is the ASCII word `TRUNC`.

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
- **The suffix is never cut.** The suffix is `-> {status} … in= out=`.
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
   - Hunk lines are still shown when workspace digests were dropped. Those windows are not scored, but later windows may refer back to them.
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

- **Render RNG.** Each event has its own render RNG, `random.Random(int(sha256(canonical_json(RawOp) + ":render")[:16], 16))`. It supplies `req=` and `{noise}`. It is independent of the generators and the oracle.
- **Timestamps.** These are parsed with `datetime.fromisoformat` and rendered in UTC as `HH:MM:SS.mmm`.

### 4.6 Bound

A block has at most 64 header tokens, then for each of up to 6 body lines one line-marker token plus at most 20 tokens, then `EOE`. That is at most 64 + 6·21 + 1 = 191 < `O`. The renderer enforces the per-line cap with `TRUNC`.

## 5. Tokeniser (frozen)

A deterministic left-to-right scan of each block. Spaces and tabs are dropped.

| Rule | Token |
|---|---|
| A newline followed (after spaces) by `@`, `!` or `[` | one reserved line-marker id `NL@`, `NL!` or `NL[` (the character is consumed) |
| Any other newline | `NL` |
| `WORD = [A-Za-z_][A-Za-z0-9_.\-/]*` | one token: a reserved id if the word is reserved, otherwise a hashed id |
| `NUM = [0-9]+` | split left to right into groups of 3 digits (the last group may be shorter); each group is a dedicated id |
| Any other printable character | one punctuation id |

**Vocabulary (V = 10,240):**

| Ids | Content |
|---|---|
| 0 | `PAD` |
| 1 | `BOS` |
| 2 | `EOE` |
| 3 | `NL` |
| 4–6 | `NL@`, `NL!`, `NL[` |
| 7–127 | reserved words, frozen in `tokenizer.py`: all tool names; the words `path lines scope query limit bytes paths suite tests target body req resource backoff action in out ok error FAILED passed ready pending at pid workspace untracked TRUNC https GET POST PUT PATCH DELETE` |
| 128–223 | printable ASCII punctuation |
| 224–1,333 | digit groups `0`–`9`, `00`–`99`, `000`–`999` |
| 1,334–2,047 | unused |
| 2,048–10,239 | other words: `2048 + int(sha256(word)[:8],16) mod 8192` |

The tokeniser returns `tokens`, `event_ord`, `offset`, the indices of the `EOE` tokens, and the **surface strings** (each token's text before hashing).

## 6. Windows, gate, labels

- **Oracle rules.** The packet carries v0.2 §6.1–§6.3 (latent state definitions, parameters, rules) and the family names in v0.2 §7.1, as the v0.2 packet did.
- **Gate rule.** Windows with withheld observations are not scored.

## 9. Direct-weight construction protocol

### 9.1 Deliverables

1. **`reference_policy.py`.** It defines `reference_policy(tokens: list[str], event_ord: list[int]) -> int`, the class at the **last** event block of the given prefix. `tokens` holds the tokeniser's surface strings, and `event_ord` its event ordinals. It uses the standard library only, is pure and deterministic, and takes at most 0.5 s of CPU per call. A timeout counts as a disagreement.
2. **`build_weights.py`.** It defines `build()`, which starts from `zero_model()`, sets parameters by closed-form rule or explicit constants, and returns the model.
   - It may import `torch`, `math`, `hashlib`, the packet `model` and the packet `tokenizer`, but may only call the tokeniser on literal word lists.
   - It may not import `renderer` or `reference_policy`.
   - Quasi-random vectors must come from `hashlib`.
   - Forbidden: `random`, `numpy.random`, `torch.rand*`, `torch.Generator`, all of `torch.nn.functional`, `softmax`, `matmul`/`@`, `mm`/`bmm`/`addmm`/`addr`/`ger`/`baddbmm`, `einsum`, `linalg`, model forward calls, I/O, and the `base64`, `struct`, `zlib`, `pickle` and `array` modules.
   - `outer` and `kron` are allowed.
   - No list or tuple literal may have more than 256 elements, and no string literal more than 512 characters. The file is at most 200 KB.
   - The build must finish in under 300 s of CPU.
3. **`CIRCUITS.md`.** It covers:
   - the residual-subspace allocation;
   - each head's role (query, key and value subspaces, and the bias pattern);
   - MLP units grouped by circuit;
   - which `reference_policy` branch each logit implements.
   - It must also account for every non-zero tensor block.
4. **`ITERATIONS.md`.** Every self-check iteration, with its diff, reason and inputs.

### 9.2 Phase A (zero labels)

- **Allowed:** forward passes and `reference_policy` calls on logs the author writes: hand-written text, or `RawOp` sequences it builds and renders with the packet renderer. Direct, reasoned edits.
- **Forbidden:** any other data, and automated search. That includes fitting, gradient steps, random restarts, parameter sweeps scored against an objective, and distilling anything into the weights, the author's own `reference_policy` included.
- **Cost cap:** US$500 of session spend, summed over every Phase A launch. Technical continuations (≤ 9) and build-rejection resubmissions (≤ 3, error text only) count against it.
- **Result:** the first valid artefact submitted as final.

### 9.3 Phase B (one diagnostic revision)

The author receives 200 validation windows: 50 per oracle class, at most one window per session. They are chosen by ascending `sha256(f"20261105:v03-phase-b-disclosure:{session_id}:{endpoint}")` within each class, with classes taken in canonical order and sessions already used skipped. For each window the author gets:
- the rendered prefix of blocks 0..t, with block t−7 marked;
- the oracle label;
- the Phase A network's argmax class at `EOE_t`;
- the `reference_policy` class.

Nothing else is disclosed. The author makes one submitted revision in the same session, with every iteration logged. Cost cap US$250.

Phase A is reported as 0 labelled examples. Phase B is reported as 200 labelled examples and 1 revision.

## v0.2 oracle rules and scenario families (unchanged in v0.3)

The labels are produced by the v0.2 oracle from the simulator's latent state. Its definitions,
parameters and rules (v0.2 §6.1-§6.3) and the names of the scenario families (v0.2 §7.1) are
reproduced here. "Encoded features" and "the 120-vector" refer to v0.2's input, which v0.3
replaces with the rendered log.

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
