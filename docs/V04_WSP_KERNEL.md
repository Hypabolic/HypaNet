# v0.4 WSP kernel: design contract

_Frozen at G1, 2026-11-05 (`experiments/v0_4/PREREGISTRATION.md`, hashed in `experiments/registry_v0_4.yaml`). This file decides the open design points in the plan's "Rules for Fable's weight-space programming". It is the contract the kernel, guards and build runner were built against. The rules the kernel added beyond the first draft (listed in `experiments/v0_4/STATUS.md`) are folded in below and marked **(G1)**. A change after G1 is a registered revision._

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
- **(G1) Rescaling refusal.** "Scalar multiple" is a tolerance, not exact equality, which `W' = c·W + noise` would evade. With `new = c·W + R` and `c = ⟨new, W⟩ / ⟨W, W⟩`, an edit is refused when `|c − 1| > 1e-3` **and** `‖R‖ < 0.5·‖(c − 1)·W‖` (`c = 0` covers zeroing a tensor). For the zero-centred `x·(1 + w)` norms (`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`) the test also runs on the effective gain `1 + w`. For the label rows it also runs on the row-centred `[4, d]` block and, when two or more rows changed, on the changed rows, which refuses a common rescaling of the rows with or without a shared offset. The rule lives in one place (`hypaqwen/wsp/edit_state.py`, `rescaling`): the server applies it to every save and the runner to every built tensor, so the two cannot disagree.
- **(G1) Edit state is a function of the weights.** Saving a tensor equal to its original removes that tensor's edit, so the edit hash never depends on history. The embedding's edit is held and hashed as its four label rows (`[4, d]`, `LABEL_IDS` order).

## 3. Model server (`envs/qwen/hypaqwen/wsp/server.py`)

- **Process.** Started by the launcher before the author session, outside the author sandbox. It loads the model once with `hypaqwen.model.load(device, "fp32")`. The device is pre-registered: MPS on this Mac, about 445 tok/s.
- **Transport.** It serves a length-prefixed JSON + raw-tensor-bytes protocol over a Unix domain socket or FIFOs in `AUTHOR_DIR/.kernel/`. The client guard permits only that path.
- **(G1) Two sockets.** `kernel.sock` serves `./py` scripts (metadata, load, save, drop, `edited`, `edit_hash`, `budget`; no forward). `fwd.sock` serves only the `./fwd` tool (`forward`, `forward_disclosed` in WSP-B, `budget`, `edit_hash`). The client guard permits `./py` only the kernel socket, so a script cannot run the model even by writing frames by hand. One connection is served at a time; a malformed frame or a failure while serving drops that connection only.
- **(G1) Command line.** `python -m hypaqwen.wsp.server --phase P --author-dir D --log-dir L [--device mps] [--budget 2000000] [--request-cap 50000] [--disclosed FILE]`. It binds `D/.kernel/kernel.sock` and `D/.kernel/fwd.sock`, prints one JSON `ready` line and exits with status 143 on SIGTERM.
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
  - **(G1) Disclosed-overlap refusal.** In a phase that holds a disclosure copy, a session is refused when its text overlaps any disclosed window: an identical window prefix, or a share of at least **0.6** of the window's blocks or of its 16-character shingles, overall or in its last four blocks. So `forward` cannot return the activations `forward_disclosed` withholds. Blocks after a session's last requested window are refused (they would never run).
  - The response gives, per window: the 4 label logits, the full-vocabulary logsumexp, the argmax and the top-k token ids (ids only, never vocabulary strings). For `both` it adds the logit deltas and an argmax-flip count. Requested activations are written as `.npy` under `output/<phase>/fwd/<k>/`.
  - Each request appends a ledger line to `experiments/v0_4/logs/<phase>.ledger.jsonl`: call id, UTC time, tokens, cumulative tokens, edit hash, sha256 of each session's text, layers requested and the request file path.
- **WSP-B only: `forward_disclosed(window_ids, model)`.** It runs the 200 disclosed windows from the read-only disclosure copy and returns **label logits and argmax only**. No hidden states, MLP activations or top-k come back. That prevents a closed-form 200-window centroid probe. It is metered like `forward`.
  - **(G1, confirmed)** At most **2** distinct non-empty edit states may be evaluated on the disclosed windows per phase: the WSP-A edit WSP-B starts from, and one revision. `original` is not limited. This is what "sees the 200 disclosed windows once" means for a model that can be re-run.
- **(G1) Weight-load quota.** `load` / `load_original` of layer tensors (everything except the embedding and the final norm) draw on a per-phase quota of distinct tensor bytes: **0.5** of the layer weights. Reloading a tensor is free. The quota is per phase, not per run, because tensors can be cached under `output/<phase>/` and re-read by the next run; it exists so that no series of `./py` scripts can assemble the whole model and run unmetered forward passes. `arch()` reports the quota.
- **(G1) Kernel log.** Every kernel op (load, save, drop, metadata) appends a line to `experiments/v0_4/logs/<phase>.kernel.jsonl` with the connection id, tensor name and bytes. It is the quota's record. The ledger is the meter's record: tokens used and requests are re-read from it on start, and each line holds, per session, the sha256 of exactly what ran (block token ids and windows), a short hash per block for near-duplicate audits and, when windows are disclosed, the session's overlap with them. A refused request is logged too.
- **WSP-F only: `score_dev(session_ids)`** (at most 128 sessions, at most 50 calls) and `holdout_check()` (once). Dev labels live only in the WSP-F server. **(G1)** The full WSP-F rules (aggregate-only responses, opaque pool ids and no dev text, the 500-session holdout disjoint from the 2,048 budget subset, accept only if strictly better than WSP-B on the holdout) are in `experiments/v0_4/PREREGISTRATION.md` §7.3. They are frozen; these two ops are implemented before WSP-F launches, by the one additive registry revision §7.3 pre-registers (new files plus named dispatch changes in `server.py`, `author/protocol.py`, `author/client.py` and `author/fwd.py`; the ops exist only in phase `wsp-f`).

## 4. Author client (`tools/hn_kernel/client.py`, copied into the author dir)

**(G1) Tools are copied read-only.** `hypaqwen.wsp.install.install(dest)` copies exactly `KERNEL_FILES` into `tools/hn_kernel/` with files 0444 and the directory 0555, so Write/Edit can neither change a file nor plant a module (`socket.py`, `numpy/`) beside them; the guard package goes to `tools/hn_guard/` the same way. `install.verify(dest)` runs before every `./py` and `./fwd` exec and in the audit: the directory holds exactly those files, none is a symlink or writable, and each matches the package source's sha256. Anything else raises `KernelTampered`. This matters because `./fwd` runs without the `./py` guard, and `./py` imports `client` before it enters the guard.

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

**(G1) Violations end the run with status 3.** Every refusal raises `GuardViolation`, is counted and prints `HN_GUARD: refused: ...` to a private duplicate of the stderr the guard found at install, so a later `dup2` or close of fd 2 cannot divert it. A swallowed violation still fails the process with status 3 however it ends (`atexit`, `finish()`, `os._exit`, `os.abort`). `install(mode, *, author_dir, phase, kernel_endpoint=None, consts_path=None)` is permanent: there is no uninstall. **Launchers** give the child's stderr as a pipe and treat a run as clean only when `hn_guard.clean_run(returncode, stderr)` holds: status 0 **and** no `HN_GUARD:` line.

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
- **(G1) Mechanism table.** `EDIT.md` ends with a `## Mechanisms` table, one row per claimed mechanism: id, the tensors (or `label rows`) that implement it, the target class (one of the four class names) and the claim in one sentence. The Q5 check (`PREREGISTRATION.md` §8) ablates exactly those tensors; a row is evaluated only if its tensors are a strict subset of the edit's and disjoint from every other row's, and an edit with no such row fails Q5. The exact format (a section headed exactly `## Mechanisms`, one Markdown table with the columns `id | tensors | target class | claim` in that order, tensors as comma-separated full names or `label rows`) is stated in both phase prompts, in the packet's `AUTHOR_INSTRUCTIONS.md` and in spec §8, and parsed by `agent_loop_lab.v04.mechanisms`; `scripts/v04_run_build.py` writes a preview (`MECHANISMS.json`: the rows and which are eligible) to the archive and warns the operator, but never refuses a build over it.
- **(G1) Archive records** (`scripts/v04_run_build.py`, after the runner): `FREEZE.json` (the registry `freeze_sha256` the phase ran under and every launch record's, `PREREGISTRATION.md` §7.3), `BUILD_RECORD.sha256` (the archive hash WSP-B names) and `MECHANISMS.json`. The freeze record fails closed. A real build (the real log dir, or an archive under `weight_program/v0_4/`) refuses to start unless the registry is `experiments/registry_v0_4.yaml`, `scripts/v04_freeze.py --check` passes and the phase has at least one launch record, every one under the registry's **G1 kernel freeze**: the G1 freeze, or a freeze after additive revisions only (a revision that changes an existing file, the WSP-F ops, ends that set; `agent_loop_lab.v04.handover.kernel_freezes`). The WSP-A archive is refused by the disclosure builder, the WSP-B prepare script and every WSP-B launch unless its `FREEZE.json` is such a record against the current registry, and unless its bytes are the ones `DISCLOSURE_PRIVATE.json` recorded (also in `wsp_b/HANDOVER.json`).
- **(G1) The WSP-A column** for the disclosure is read out with the archived edit applied: `hypaqwen.run_readout --split val --device mps --dtype fp32 --edit ARCHIVE` (`runner.load_edit`, then `runner.apply_edit`); its `meta["edit"]` names the archive's `edit_state_sha256` and `BUILD_RECORD.json` sha256, and `meta["model"]` must be the pinned model.
- **(G1) Refusals.** A refused build writes `attempts/NNNN.json`; only its author-facing `error` text (at most 2,000 characters, no harness paths, author exceptions by type and line only) goes back to the author, in at most 3 resubmissions per phase.

## 7. Session harness (built after §3–6)

- **(G1) Author dir layout** (`../HypaNet-v04-author`, made fresh by the prepare script, never reused): `packet/` (read-only), `tools/hn_kernel/` and `tools/hn_guard/` (read-only, §4), the `py` and `fwd` wrappers (0555: `py` runs `.venv/bin/python -I -B tools/hn_kernel/run_py.py`, `fwd` runs `.venv/bin/python -I -B tools/hn_kernel/fwd.py`), `.venv/` (`envs/qwen/scripts/make_author_venv.sh`), `.kernel/` (the sockets) and `output/<phase>/`, the only writable tree.
- **Launch** (`scripts/v04_launch.sh`):
  - `claude -p --tools Read,Glob,Grep,Write,Edit,Bash` (an allow-list), `--strict-mcp-config` with no MCP servers, and `--permission-mode dontAsk`;
  - a PreToolUse hook that rejects any Bash command not matching `^\./(py|fwd) [^;&|<>`$\n]+$`;
  - `--max-budget-usd` set to the remaining cap;
  - **(G1)** the session permissions deny Write and Edit under `tools/`, on the `py` and `fwd` wrappers, on `.venv/` and on `.kernel/`; `install.verify` (§4) is what still holds if that is misconfigured.
- **(G1) Further launch rules** (as built, `experiments/v0_4/KIT_TESTS.md`):
  - **Claude flags.** Besides the above: `--restricted` (file tools confined to the working directory; the settings' hook still applies), `--settings <rendered> --setting-sources project`, `--permission-prompts none`, `--disable-slash-commands`, `--output-format stream-json --verbose`; a fresh `--session-id` for WSP-A, `--resume` of the same session for every later launch. The environment is scrubbed of every variable that links the session to a parent Claude Code session, with `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`. A session dir with a `CLAUDE.md` on any ancestor, or `~/.claude/CLAUDE.md`, is refused.
  - **Settings.** The phase template (`author_kit/settings_<phase>.json`) is rendered for each attempt; its one `@DENY_ROOTS@` entry expands to Read/Glob/Grep/Edit/Write deny rules for the repo, every sibling of every ancestor of the author dir up to `~`, `~/.claude`, `~/.cache`, `~/.config`, `~/.local`, `~/Library` and the temp roots. The launcher checks the rendered rules (`agent_loop_lab.v04.kit.required_rules`), since `claude -p` silently ignores an invalid settings file.
  - **Hook install location.** The hook is `hook_bash.py`, installed read-only as `AUTHOR_DIR/tools/hn_hook/hook_bash.py` and verified with the other tools (§4). Its command names no harness path (a blocked call shows the session the command line): `"$CLAUDE_PROJECT_DIR"/.venv/bin/python -I -B "$CLAUDE_PROJECT_DIR"/tools/hn_hook/hook_bash.py <phase>`. A hook exiting with anything but 0 or 2 fails open, so every internal error is an exit 2, and the launcher runs the hook self-test (33 cases) on the rendered settings before each launch.
  - **Hook word rules.** Beyond the regex (applied with `fullmatch`, so a trailing newline does not pass): every argument is made of `[A-Za-z0-9_./=:,+@%-]` and separated by single spaces (no quotes, globs, `~`, braces, backslashes, tabs); no argument is absolute or has a `..` component; `./fwd` takes exactly one `.json` inside `output/<phase>/`; `./py` takes a `.py` inside `output/<phase>/` (or `-m py_compile` of one); `./py -c`, any other `-m` and `run_in_background` are refused (`claude -p` waits for background tasks). Write/Edit targets resolve inside `output/<phase>/` and outside its `fwd/` (compared case-insensitively); Read/Glob/Grep paths resolve inside the author dir and outside `.kernel/`.
  - **Per-attempt files.** Everything written before claude starts is per attempt (`ATTEMPT = NAME.<utc>`: the rendered settings and the server's ready line and log), so a failure before claude leaves the launch re-runnable. Only the transcript `NAME.stream.jsonl` marks a launch as done; it is never overwritten. A launch with no init event (nothing was spent) is moved aside as `ATTEMPT.noinit.*`. Every attempt is appended to `LAUNCHES.jsonl`.
  - **Killed-launch charging.** Just before claude starts, `NAME.launch.json` records the `--max-budget-usd` ceiling (and the registry `freeze_sha256`). A launch with no result event (killed, or stopped by the watchdog) is charged that ceiling, or an operator figure from `COST_OVERRIDES.json`, never US$0 and never a later resume's total: the cumulative total restarts from the last normal exit.
  - **Watchdog.** claude runs in the background under a wall-clock limit (`V04_WALL_SECONDS`, default 48 h): SIGINT, then SIGTERM after 60 s, then SIGKILL after 30 s more. The model server gets SIGTERM after the session (SIGKILL after 60 s); `caffeinate` keeps the Mac awake while the server runs.
  - **Harness scratch.** No `pytest-of-$USER` may exist in any temp root (the real per-user temp dir included) while a session runs.
- **(G1) WSP-B launch.** `scripts/v04_prepare_phase_b.sh` (after the WSP-A archive and `scripts/v04_build_disclosure.py`) copies `DISCLOSURE.json` and `DISCLOSURE.sha256` into `AUTHOR_DIR/disclosure/` (0444/0555), makes `output/wsp-a/` read-only, creates `output/wsp-b/scratch/`, verifies the tools (the same files in every phase, never reinstalled) and seeds the WSP-B journal with a byte-identical copy of the WSP-A journal, whose final edit hash must be the archived edit's: the server replays it, so WSP-B starts from the archived WSP-A edit (`PREREGISTRATION.md` §6) with a new ledger, budget and load quota. `scripts/v04_launch.sh wsp-b` re-checks the archive, the disclosure, the copy and the seeded journal on every WSP-B launch (`agent_loop_lab.v04.handover`), starts the server with `--phase wsp-b --disclosed experiments/v0_4/DISCLOSED_WINDOWS.json` (blocks 0..t of each disclosed window, exactly the disclosed text), requires the first WSP-B server to report the WSP-A edit hash, and resumes the WSP-A session with `prompt_wsp_b.md` (the archive's `BUILD_RECORD.json` sha256, its edit hash and the disclosure sha256 filled in). The real prepare refuses unless `experiments/v0_4/HANDOVER.json` names the author dir (not a kit dir), before anything is written. Kit rehearsals (`V04_KIT=1`, with or without the registry) give both scripts their own archive, disclosure root and log dir (`V04_WSP_A_ARCHIVE`, `V04_DISCLOSURE_ROOT`, `V04_LOGDIR`; the prepare script also `V04_HANDOVER`), never the real ones or a handover record inside the repo; without a registry they accept only an unfrozen kit archive (`--kit-unfrozen`).
- **Cost.** Account by session id, as a delta of the cumulative `total_cost_usd`. v0.3's `scripts/v03_cost.py` double-counted resumed sessions. Caps: WSP-A US$500, WSP-B US$250, WSP-F US$250.
- **Audit:**
  - forbidden markers: the repo, `envs/qwen`, `datasets`, `.cache/huggingface`, `experiments/v0_4`;
  - an AST rule flagging a loop body that calls `save_edit` or `drop_edit` and writes a `./fwd` request;
  - a ledger cross-check: budget, and how many distinct edit hashes were run on identical text sets (more than 3 goes to the adjudicator).
- **Kit tests:** `ListAgents`, `SendMessage` and other non-kit tools are absent from the init tool list; the hook rejects compound commands; the meter refuses over budget; a fake `build_edit.py` rebuilds byte-identically.
