# v0.4 WSP kernel: design contract

_Draft, 2026-11-05. This file decides the open design points in the plan's "Rules for Fable's weight-space programming". It is the contract the kernel, guards and build runner are built against. G1 freezes it._

## 1. Principles

1. **Two processes.** A harness-owned **model server** holds the weights, the edit state, the token meter and the ledger. It never executes author code. The author's code runs in fresh, guarded, short-lived **client** interpreters (`./py script.py`). The clients never load a model.
2. **Forward passes are a separate tool.** The author's scripts can read weights and save edits. They cannot run the model. A forward pass is one `./fwd request.json` command per Bash call, metered and logged with the edit-state hash.

   So "try edit, run model, keep the best" cannot be a loop in a script. Every candidate costs a visible tool call, and the audit sees each one. This follows the blog, which kept forward in its own block, one request per agent turn.
3. **Everything the harness keeps sits outside the author directory.** That covers the meter, the edit journal and the ledger. Continuations reload them, so neither the budget nor the edits reset.
4. **fp32 throughout.** Weights are an exact fp32 upcast of the pinned BF16/F32 checkpoint (`hypaqwen/model.py` FILE_SHA256). Edits are fp32. Scoring never casts edits to bf16. The bf16-rounded change count is reported alongside.

## 2. Edit scope

Allowed tensors: `model.language_model.*` (the 320 text-model tensors), with these exceptions:

- `model.language_model.embed_tokens.weight`: only the 4 label-word rows (`LABEL_IDS`), through `save_label_rows`. The embeddings are tied, so the build record flags the input-side effect. It lists every occurrence of those ids in the frozen prefix and suffix.
- `model.language_model.norm.weight` (the final, zero-centred `(1 + w)` norm) is refused. Scaling it is a global temperature knob.
- Refused: `mtp.*`, `model.visual.*`, any non-finite value, a shape or dtype mismatch, and a whole-tensor scalar multiple of the original (`W' = c·W`).

## 3. Model server (`envs/qwen/hypaqwen/wsp/server.py`)

- **Process.** Started by the launcher before the author session, outside the author sandbox. It loads the model once with `hypaqwen.model.load(device, "fp32")`. The device is pre-registered: MPS on this Mac, about 445 tok/s.
- **Transport.** It serves a length-prefixed JSON + raw-tensor-bytes protocol over a Unix domain socket or FIFOs in `AUTHOR_DIR/.kernel/`. The client guard permits only that path.
- **State.**
  - The current edits are `{name: fp32 tensor}`. Pristine copies are kept only for edited tensors.
  - The edit-state hash is sha256 over sorted `(name, bytes)`. The empty state has a fixed hash.
  - An edit journal sits in a harness dir (`experiments/v0_4/logs/<phase>/journal/`). The server writes it after each save or drop and replays it on start.
- **Ops for the client** (any number per `./py` run):

  | Op | Behaviour |
  | --- | --- |
  | `names` / `shapes` / `arch` | static metadata |
  | `load(name)` | current tensor: the edit, else the original |
  | `load_original(name)` | always the original |
  | `save_edit(name, tensor)` | validates §2 and returns an `EditReceipt` (below) |
  | `save_label_rows(rows[4, 1024])` | writes only the four label rows; reported separately |
  | `drop_edit(name)` | removes one edit |
  | `edited()` | names of the edited tensors |
  | `edit_hash()` | the edit-state hash |
  | `budget()` | `{tokens_used, tokens_left, requests}` |

  `EditReceipt` carries `name`, `changed_entries`, `changed_rows`, `changed_cols`, `rel_frobenius`, `delta_rank` (singular values > 1e-6·‖ΔW‖₂) and `sha256`.

- **Op for `./fwd` only: `forward(request)`.** The request names `model: "edited" | "original" | "both"` and gives `sessions: [{id, blocks: [str], windows: [int]}]` plus `return: {hidden: [layers], mlp_act: [layers], topk: int ≤ 20}`. The rules:
  - Texts are run through `hypaqwen.readout.Verbaliser` with the frozen interface and the exact scoring tokenisation.
  - The cost is the sum of `WindowReadout.tokens`, doubled for `both`. It is charged on completion. The server refuses a request that would cross the phase budget (2,000,000 tokens) or exceed 50,000 tokens by itself.
  - The response gives, per window: the 4 label logits, the full-vocabulary logsumexp, the argmax and the top-k token ids (ids only, never vocabulary strings). For `both` it adds the logit deltas and an argmax-flip count. Requested activations are written as `.npy` under `output/<phase>/fwd/<k>/`.
  - Each request appends a ledger line to `experiments/v0_4/logs/<phase>.ledger.jsonl`: call id, UTC time, tokens, cumulative tokens, edit hash, sha256 of each session's text, layers requested and the request file path.
- **WSP-B only: `forward_disclosed(window_ids, model)`.** It runs the 200 disclosed windows from the read-only disclosure copy and returns **label logits and argmax only**. No hidden states, MLP activations or top-k come back. That prevents a closed-form 200-window centroid probe. It is metered like `forward`.
- **WSP-F only: `score_dev(session_ids)`** (at most 128 sessions, at most 50 calls) and `holdout_check()` (once). Dev labels live only in the WSP-F server.

## 4. Author client (`tools/hn_kernel/client.py`, copied into the author dir)

`from wsp import load, load_original, save_edit, save_label_rows, drop_edit, edited, edit_hash, budget, NAMES, SHAPES, ARCH, LABEL_IDS, CLASS_NAMES, tok`

- `tok` is a `tokenizers.Tokenizer` built from `packet/qwen/tokenizer.json` before the guard is entered.
- There is no `forward` in the client.
- Weight-only analysis helpers are plain torch and live in the client:
  - `svd(name, k)`
  - `head_slice(layer, head, kind)`
  - `top_writers(name, direction, k)`
  - `project_out(W, u, cols=None, alpha=1.0)` = `W − α·u ⊗ (mask ⊙ (uᵀW))`
  - `rank1(W, a, b)`
  - `zero_head(layer, head)`
  - `label_lens(h)` and `dla(vec)`, which apply the final `(1 + w)` RMSNorm and the label rows to vectors the author got from `./fwd` output files
  - `diff_means(a, b)`
  - `control_direction(seed, like)` for norm-matched random controls, the only RNG allowed

## 5. Guards (`tools/hn_guard/guards_v04.py`, on top of the v0.2/v0.3 guards; torch 2.14.1)

**KERNEL_CLIENT** (`./py`) is v0.3 AUTHOR_RUN with these changes:

- Matmul and `torch.linalg` (svd, eigh, qr, norm) are allowed. `lstsq` and `pinv` are refused.
- These imports are refused: `transformers`, `hypaqwen`, `peft`, `accelerate`, `agent_loop_lab`, `sklearn`, `torch._dynamo`, `torch.compile`, `torch.export`, `torch.func`, `torch.optim`.
- Backward, `autograd.grad`, enabling grad mode and the functional grad transforms are refused.
- I/O is an allow-list:
  - reads under the author dir and the interpreter install;
  - writes under `output/<phase>/`;
  - connections only to the kernel endpoint.
  - No subprocesses, threads or other sockets.
- Every patched torch internal is re-verified on 2.14.1, with one test per banned path.

**BUILD_EDIT** (`build_edit.py` in the build runner) is v0.3 BUILD minus the matmul and linalg ban. It still refuses:

- any model or `nn.Module` call;
- RNG, autograd and optimisers;
- threads and subprocesses;
- all I/O except its declared constants file.

## 6. Deliverables and build (`envs/qwen/hypaqwen/wsp/runner.py`)

- **`build_edit.py`.** It exposes `def build(orig: Mapping[str, Tensor], consts: Mapping[str, Tensor]) -> dict[str, Tensor]`, returning only the replaced tensors. `orig` hands out clones. `consts` comes from an optional `edit_constants.safetensors` of at most 8 MB, holding activation-derived vectors; EDIT.md must cite each one's ledger call ids.
- **How the runner runs it:**
  - a fresh `-I -B` child, under BUILD_EDIT;
  - CPU, `torch.set_num_threads(1)`, `torch.use_deterministic_algorithms(True)`;
  - float64 accumulation where the author asks for it, cast to fp32 at the end;
  - RLIMIT_CPU and a wall timeout.
- **Checks:**
  - names inside §2 scope; shapes, fp32, finite; no autograd state;
  - a second fresh child must rebuild it byte-identically;
  - the result must equal the server's final journalled edit state tensor for tensor, or the build is refused.
- **Archive.**
  - `BUILD_RECORD.json`: per tensor, changed entries, changed rows and columns, ‖ΔW‖_F/‖W‖_F and the bf16-rounded changed count; the overall changed fraction; the label-row flag; platform, torch and BLAS.
  - An exact diff (`edit_diff.safetensors`: changed indices and values per tensor, with sha256), so the artefact does not depend on recomputation.
  - `attempts/NNNN.json`.
- **Prose files:** `EDIT.md` (what changed, where, the mechanism claimed and the ledger ids behind each vector), plus `ITERATIONS.md` (WSP-A) or `REVISION.md` (WSP-B).

## 7. Session harness (built after §3–6)

- **Launch** (`scripts/v04_launch.sh`):
  - `claude -p --tools Read,Glob,Grep,Write,Edit,Bash` (an allow-list), `--strict-mcp-config` with no MCP servers, and `--permission-mode dontAsk`;
  - a PreToolUse hook that rejects any Bash command not matching `^\./(py|fwd) [^;&|<>`$\n]+$`;
  - `--max-budget-usd` set to the remaining cap.
- **Cost.** Account by session id, as a delta of the cumulative `total_cost_usd`. v0.3's `scripts/v03_cost.py` double-counted resumed sessions. Caps: WSP-A US$500, WSP-B US$250, WSP-F US$250.
- **Audit:**
  - forbidden markers: the repo, `envs/qwen`, `datasets`, `.cache/huggingface`, `experiments/v0_4`;
  - an AST rule flagging a loop body that calls `save_edit` or `drop_edit` and writes a `./fwd` request;
  - a ledger cross-check: budget, and how many distinct edit hashes were run on identical text sets (more than 3 goes to the adjudicator).
- **Kit tests:** `ListAgents`, `SendMessage` and other non-kit tools are absent from the init tool list; the hook rejects compound commands; the meter refuses over budget; a fake `build_edit.py` rebuilds byte-identically.
