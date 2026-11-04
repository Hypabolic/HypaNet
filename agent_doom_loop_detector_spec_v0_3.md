# Agent Doom Loop Detector — Hand-Programmed Transformer on Raw Logs

**Specification:** v0.3 (successor to v0.2; v0.2 continues to completion as a baseline)
**Status:** DRAFT r2, 2026-11-05, written under delegated authority (§0.1) after a four-lens review (§0.2). It is frozen at V0c (§14), before any weight author sees it.
**Purpose:** Measure whether a reasoning LLM can write, **with no optimiser and no labelled examples**, the parameters of a small fixed transformer that reads **rendered agent log text** and classifies the agent's recent behaviour into the four v0.2 patterns.

Sections marked **[packet]** go into the authoring packet. All other sections are withheld from the weight author.

---

## 0. Changes from v0.2

| # | Area | v0.2 | v0.3 |
|---|---|---|---|
| 1 | Network input | 120 hand-engineered features | Tokenised rendered log text of the session prefix (§4–§5) |
| 2 | Architecture | MLP 120→24→12→4, 3,256 parameters | Causal transformer: 4 layers, d=128, 8 heads, event-distance bias, no normalisation. **2,128,932 parameters** (§2.2) |
| 3 | Inference unit | One forward pass per window | One causal forward pass per session, read at each event's `EOE` token |
| 4 | Engineered signals | `repeated_*`, `state_revisited`, `validation_delta`, … | None. The network sees tool calls, arguments, results, error lines, opaque diff-hunk ids and token counts |
| 5 | Labels, windows, gate | — | Unchanged: the same sessions, oracle labels, gate and scored-window set |
| 6 | Phase A | A-strict plus A-selfcheck | Phase A only (zero labels; forward passes on author-written inputs allowed), with a cost cap |
| 7 | Reference policy | Function of the 120-vector | Function of the tokeniser's surface tokens (§9.1) |
| 8 | Phase C | Trained MLPs | Trained transformers of the same architecture, capped to fit CPU |

### 0.1 Decisions made under delegated authority

The adjudicator was away. On 2026-11-05 they instructed: "continue working until we have that working hand programmed transformer", "try and get fable building v0.3 as soon as possible". They had already chosen synthetic logs and asked for v0.2 to be finished.

| # | Decision | Value |
|---|---|---|
| E1 | Log source | v0.2 generators plus a raw-operation side capture, rendered by a frozen renderer |
| E2 | Numbers | Digit groups of up to 3, left to right, each with a dedicated id |
| E3 | Positions | Within-event offset embedding, plus a per-head scalar bias over clamped event distance, with a dedicated BOS column. No absolute token position |
| E4 | Width | d=128, 8 heads × 16, MLP 512, 4 layers |
| E5 | Phase A cost cap | US$500, summed as `total_cost_usd` over all `result` events of every Phase A launch transcript. No new launch once the sum reaches the cap; any overshoot within a launch is reported |
| E6 | Weight author | A fresh `claude-fable-5-1` session (fallback `claude-opus-5-5`) in a new directory `../HypaNet-v03-author` |
| E7 | Sealed timing | v0.2 sealed test/OOD scoring may run once V0c is committed. v0.3 sealed scoring runs once, after the v0.3 Phase B archive, or after Phase A is declared failed. Neither is ever shown to a v0.3 author |
| E8 | Phase C budgets | 32 / 128 / 512 sessions × 3 seeds; 2,048 × 1; all-dev × 1, capped at 3 epochs |
| E9 | Diff rendering | Each changed file is shown as one pair of opaque per-session ids (removed lines > added lines), not code text, with no hunk count and no byte size (§4.3). The generators' code text encodes each edit's latent kind (`+1` and a guard line for fixes, `1000+n` constants for wrong variants, `cleanup pass N` for cosmetic edits, `-ok` for data fixes), which would leak the label. Real code semantics are out of scope for v0.3 |
| E10 | Feasibility spike | Replaced by a paper feasibility analysis (review lens 1: circuit sketch for same-failure-within-8, plus a head and dimension budget of about 26–29 of 32 heads). This is for speed. The opaque hunk lines (E9) remove the deepest stage the analysis found (inverse-patch detection on raw code lines) |
| E11 | Validation scoring before Phase B | Allowed for the harness only (as in v0.2 M3). The first archived Phase A artefact is the v0.3 Phase A result. Any further author launch is a new registered version, v0.3.k, which is reported alongside the first with the number of prior launches. It never replaces the first |

The adjudicator may reverse any of these. Reversing one after V0c creates a new registered version.

### 0.2 Review record

The four-lens review (hand-programmability, data and rendering, integrity, meaning) ran on 2026-11-05. Adopted in r1:

| Finding | Change |
|---|---|
| Critical: patch text leaks latent kind | E9 |
| Revert digest exposed (`to=<digest>`) | Removed |
| Bodies only for edit/write | Every event that changes tracked files shows its hunks (revert, OOD installs) |
| Truncation and field rules | Per-field budgets; the suffix is never cut |
| Dropped observations | A per-field `?` |
| Noise | Render noise seeded per event |
| Line numbers | Stable line numbers |
| Markers | ASCII truncation marker; line-marker tokens `NL@` `NL!` `NL[` |
| BOS | A dedicated BOS bias column |
| Offsets | Offset overflow raises an error |
| Reference policy | Takes surface tokens |
| Guard extensions | Allow `outer`/`kron`; ban the functional, `mm` and softmax families; literal-size limits; guarded author `py` |
| Author directory | A new, sealed author directory |
| Packet | Exclusions extended |
| Phase B | One window per session |
| Grammar-fuzzer sweep | Dropped (H2 is measured on benchmark windows only) |
| Disclosure | Disclosure bullets in §1.2 |
| r2 (data audit, dev): `bytes=` on `write_file` and the per-file hunk count both separated progress from non-progress edits (generator artefacts) | `bytes=` removed; one line per changed file |
| r2: reserved words also occur as values (`query`, `limit`, `path`, `tests`, `backoff`, `target`) | Accepted: in a value position they get the reserved id |

---

## 1. Objective and hypotheses [packet]

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

---

## 2. Architecture

### 2.1 Boundary

```text
Generator ──► OracleRecord (sealed labels) ─────────────────────────────────────┐
    │                                                                            │
    ├─► CanonicalEvent[] ──► v0.2 runtime gate (non-neural) ──► scored mask       │
    └─► RawOp[] ──► renderer (§4) ──► text ──► tokeniser (§5) ──► ids, event_ord, offset
                                                │                      │
                                      reference_policy        transformer (hand-written)
                                                └──► class per EOE ◄───┘ logits per EOE ──► evaluation
```

The gate runs on canonical events and only decides which windows are scored. The network sees the whole rendered prefix, including blocks with dropped observations (rendered as `?`).

### 2.2 Target model (immutable within a run) [packet]

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

### 2.3 Capacity notes (withheld)

- **Query–key dimensions.** Any one head compares at most 16 query–key dimensions, or 15 with a threshold or flag dimension.
- **Single-token identity.** ±1 codes over 15 dimensions give a false-match rate of about 2⁻¹⁵ per pair, with the threshold set between dot products 15 and 13 and a large scale. A k-part identity gets about 15/k dimensions per part.
- **Multi-token identities.** Test ids with spaces, http path templates and search queries need per-token matching plus an aggregation stage, or a chosen representative token.
- **No LayerNorm.** Magnitudes accumulate, and large values in one subspace can saturate later softmaxes.
- **BOS as a sink.** BOS is in its own bias column, so it serves as a constant attention sink.

Use these notes to diagnose Phase A failures.

---

## 3. Raw operation record (`RawOp`)

One record per event is captured from the generator. It is the information a real tool-call log would carry. The capture draws no random numbers and changes no canonical event (§7.1).

| Field | Notes |
|---|---|
| `family`, `target`, `args` | As passed to `intent_fingerprint` |
| `outcome` | SUCCESS / ERROR / NEUTRAL |
| `error_class`, `error_location` | As passed to `failure_fingerprint`. Present even if the canonical fingerprint was dropped |
| `validation` | `{suite, passed, total}`, true values, or null |
| `changed` | `[{path, before, after}]`: every tracked file whose text differs between the start and end of the operation, for any family, sorted by path. `before`/`after` is null when the file is absent |
| `extra` | Family-specific observable detail, for example the first failing test id |
| `input_tokens`, `output_tokens`, `timestamp` | As emitted on the canonical event (null if dropped) |
| `dropped` | A subset of `{input_tokens, output_tokens, digest_before, digest_after, failure, validation_suite, validation_passed, validation_total}`, mapped from each engine's own names |

A `RawOp` never carries latent effect, progress, episode state, family or scenario id, or anything derived from the oracle.

---

## 4. Log rendering (frozen) [packet]

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

---

## 5. Tokeniser (frozen) [packet]

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

---

## 6. Windows, gate, labels

These are unchanged from v0.2 §5–§6. The prediction for endpoint *t* is the argmax at `EOE_t`. A window is scored if it is gate-passing and its oracle label is not INSUFFICIENT, which is the same set as in v0.2.

**[packet]**

- **Oracle rules.** The packet carries v0.2 §6.1–§6.3 (latent state definitions, parameters, rules) and the family names in v0.2 §7.1, as the v0.2 packet did.
- **Gate rule.** Windows with withheld observations are not scored.

---

## 7. Data

### 7.1 Raw capture

- **Engines.** Both engines gain a `RawOp` side output. The change is recorded with its diff under `experiments/v0_3/raw_capture/`.
- **Acceptance.** Regenerated event and oracle files are byte-identical to the committed v0.2 manifests, and no RNG draw is changed.

### 7.2 Rendered datasets

`datasets/v0_3/{split}.raw.jsonl.gz` for dev, val, test, counterfactual and OOD, plus `MANIFEST_v0_3.json`, which holds:
- the v0.2 manifest hashes;
- the source hashes of the renderer and tokeniser;
- token statistics per split.

### 7.3 Rendering audits (dev only; reported)

1. **Revisit evidence.** For each REVERSION event, check whether some `h_new` of the event equals an `h_old` or `h_new` of an earlier event in the window. Report the fraction of OSCILLATING windows whose reversions all show such evidence.
2. **Marker audit.** For behaviour-changing MUTATE events, fit a logistic classifier on hunk count, changed-path count and tool. Report its macro-F1 for PROGRESS/RESOLVES vs NONE/REVERSION relative to the constant prior.
3. **Bucket collisions.** Report distinct words that share a hashed id within one session.
4. **Length.** Report the distribution of tokens per session.

---

## 8. Roles and independence

| Role | Who |
|---|---|
| Harness implementer and adjudicator stand-in | This Claude Code session and its workflow agents |
| Weight author | A **fresh** Fable 5.1 session in `../HypaNet-v03-author`, holding only the v0.3 packet. It has not seen generator or oracle code, datasets, eval output, the v0.2 author's directory or the reviews |
| OOD generator | Codex (v0.2). The raw-capture edit is recorded |

**Isolation:**
- Read, Grep and Glob are denied on the repo, on every sibling `HypaNet*` directory, on harness scratch and on Claude project data.
- No web, MCP, subagents or skills.
- Auto-memory is off.
- The author's `py` wrapper runs every script under the anti-training guard. The guard allows forward passes and bans autograd, `torch.optim`, random-number APIs, and imports of the generator, dataset and eval packages.
- The audit flags any path outside the author directory, and scans every executed script.

**Packet:**
- the [packet] sections of this spec;
- the v0.2 oracle rules and family names;
- `renderer.py`, `tokenizer.py` and `model.py`. `rawop.py` is not included because its docstring describes the generators. The renderer takes plain dicts, and `AUTHOR_INSTRUCTIONS.md` lists the RawOp dict fields of §3;
- the zero `state_dict` template;
- `AUTHOR_INSTRUCTIONS.md`;
- a grammar sheet with one invented example block per tool. No word in it may occur in the generator vocabularies, which is checked by a test.

---

## 9. Direct-weight construction protocol [packet]

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
- **Cost cap:** E5. Technical continuations (≤ 9) and build-rejection resubmissions (≤ 3, error text only) count against it.
- **Result:** the first valid artefact submitted as final.

### 9.3 Phase B (one diagnostic revision)

The author receives 200 validation windows: 50 per oracle class, at most one window per session. They are chosen by ascending `sha256(f"20261105:v03-phase-b-disclosure:{session_id}:{endpoint}")` within each class, with classes taken in canonical order and sessions already used skipped. For each window the author gets:
- the rendered prefix of blocks 0..t, with block t−7 marked;
- the oracle label;
- the Phase A network's argmax class at `EOE_t`;
- the `reference_policy` class.

Nothing else is disclosed. The author makes one submitted revision in the same session, with every iteration logged. Cost cap US$250.

Phase A is reported as 0 labelled examples. Phase B is reported as 200 labelled examples and 1 revision.

---

## 9.5 Phase C (trained baselines)

- **Setup:** same architecture and tokeniser; standard random initialisation; cross-entropy on scored `EOE` positions; Adam with lr 3e-4; batches of 8 sessions; early stopping on validation macro-F1.
- **Budgets:** E8.
- **Optional ablation:** no `event_ord`, with a log-bucketed token-distance bias instead (512 sessions and all-dev, 1 seed each).
- **Also reported:** the v0.2 networks, the rules engine, the v0.3 reference policy and the constant priors.

## 10. Evaluation

- **Metrics:** v0.2 §10.1 metrics: macro-F1, per-class precision and recall, false-loop alarm rate, confusion, H2 fidelity, counterfactual consistency and session-bootstrap CIs. Also CPU latency per session and the non-zero parameter count.
- **Validation:** metrics exclude the Phase B disclosed sessions.
- **Columns:** v0.3 A, v0.3 B, v0.3 reference policy, v0.2 columns, Phase C, rules and constant prior.
- **Timing:** sealed scoring follows E7.
- **Research targets (not gates):**
  - fidelity ≥ 0.95 against the author's own policy, on validation windows outside the disclosed sessions;
  - sealed macro-F1 well above the constant prior;
  - stated gaps to v0.2 A-selfcheck and to the trained transformer.

## 11. Verification

- **Renderer:** byte-identical output for the same input; the caps hold; golden tests for each family; no latent field reachable.
- **Tokeniser:**
  - vocabulary ranges, digit grouping, reserved ids and line markers are correct;
  - an offset at or above `O` raises;
  - no reserved word is a generator vocabulary or error-class string.
- **Model:**
  - shapes and 2,128,932 parameter slots, all zero at the start;
  - **causal check:** appending events never changes earlier `EOE` logits (bitwise);
  - BOS bias column; event-distance clamp.
- **Raw capture:** byte-identity, as in §7.1.
- **Anti-training:** the §9.1 bans are enforced by the runner's source scan and the runtime compute ban, plus the guarded author `py`.
- **Leakage:**
  - the packet manifest contains no generator, dataset, oracle-record or review files;
  - grammar-sheet words are disjoint from the generator vocabularies.

## 12. CLI

- `inspect log`: the rendered log with ids and per-`EOE` logits.
- `inspect heads --event t`: per-head attention from `EOE_t`, grouped by source event.
- `inspect model`: non-zero blocks with their `CIRCUITS.md` labels.

## 13. Layout

`src/agent_loop_lab/v03/{rawop,renderer,tokenizer,model,packet,runner,audits}.py`, `experiments/v0_3/`, `weight_program/v0_3/{phase-a,phase-b}/`, `datasets/v0_3/` and `experiments/registry_v0_3.yaml`.

## 14. Milestones

| Milestone | Contents |
|---|---|
| **V0a** | Raw capture (byte-identity), renderer, tokeniser, model, guards, packet, author kit, dev audits |
| **V0c** | Freeze and register: spec, packet hash, prompts, settings, `py` wrapper, scoring scripts and seeds, the Phase B rule, Phase C configuration, and the dataset manifest |
| **V2** | Phase A |
| **V3** | Phase B |
| **V4** | Phase C, then the joint sealed evaluation with v0.2, then the report |

## 15. Definition of done

1. The packet rebuilds to the same hash.
2. The artefacts rebuild byte-identically.
3. Logs and guards show no optimiser, search or unauthorised labels.
4. The circuits are traceable with `inspect`.
5. The comparison table can be reproduced.

A poor result is still a result, provided it is attributed to policy, compilation, rendering, capacity or distribution shift.
