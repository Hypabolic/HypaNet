# Weight-author instructions (spec v0.4, weight-space programming)

You are the **weight author**. Your inputs are the files in `packet/` and nothing else. You work in
your own directory; you never read anything outside it.

`SPEC.md` defines the task, the model, the log grammar, the rules and the deliverables. This file is
the operational protocol: the exact commands, functions, request formats and checks. Where the two
seem to differ on *how* something is run, built or archived, follow this file. `INTERFACE.md` gives
the frozen prompt and its token ids.

## Your directory

| Path | What it is |
|---|---|
| `packet/` | This packet (read-only). `PACKET_MANIFEST.json` holds the sha256 of every file |
| `packet/SPEC.md`, `packet/INTERFACE.md` | The task and the frozen prompt |
| `packet/renderer.py` | The frozen log renderer (`render_session`, `render_event`); it reads plain dicts (below) |
| `packet/tokenizer.py` | The renderer's word-level tokenizer, used only for its truncation budgets (not Qwen's) |
| `packet/GRAMMAR_SHEET.md` | One invented example block per tool, with its word-level tokens. It is v0.3's: its `BOS`, `EOE`, `event_ord` and network-readout sentences describe v0.3's model and do not apply here; only the block text and the word-level truncation budgets matter. Block strings sent to `./fwd` contain no `EOE` |
| `packet/qwen/` | Qwen3.5-0.8B's `config.json`, `tokenizer.json` and `tokenizer_config.json`. No weights: the kernel serves them |
| `tools/hn_kernel/` | The kernel client (`wsp`), the `./fwd` tool and the `./py` launcher (read-only) |
| `tools/hn_guard/` | The guard every `./py` script runs under (read-only) |
| `py`, `fwd` | The two commands you run |
| `.kernel/` | The kernel's sockets. Do not touch |
| `output/wsp-a/` | The only writable tree: put your scripts and request files in `output/wsp-a/scratch/`; `./fwd` writes its results to `output/wsp-a/fwd/`; the deliverables go at the top level |

Never write inside `packet/` or `tools/`, and do not edit `output/wsp-a/fwd/`. The harness
archives your whole phase directory, `scratch/` and `fwd/` included.

## How the model is held

- A **model server**, run by the harness outside your directory, holds Qwen3.5-0.8B in fp32, your
  current edit, the forward-pass meter, the edit journal and the ledger. It never runs your code.
- Your `./py` scripts talk to it through the kernel client (`from wsp import ...`): they read weights
  and save or drop edits. They cannot run the model.
- `./fwd request.json` runs the model, under your current edit or the original weights, on texts you
  write. It is metered and logged. It is the only way to run the model.
- The edit lives in the server: it persists across `./py` runs and across continuations of your
  session. `edited()` and `edit_hash()` tell you what it is now.

## Commands

**One command per Bash call.** Only commands that start with `./py ` or `./fwd ` are allowed, and
compound commands are refused: no `;`, `&&`, `||`, pipes, redirection (`>`, `<`), backticks, `$` or
`echo`. Write scripts and request files with the file-writing tool, then run them:

```
./py output/wsp-a/scratch/inspect_heads.py
./fwd output/wsp-a/scratch/req_0001.json
```

Never call `./fwd` from a script, and never write a script that saves edits and request files in a
loop: every forward pass is its own, visible tool call.

## Running Python (`./py`)

- `./py FILE [ARGS...]` runs `FILE` with the `.venv` interpreter: Python 3.12, `torch==2.14.1`,
  `numpy==2.5.4`, `safetensors==0.8.0` and `tokenizers==0.23.3`, on CPU. Nothing else is installed,
  and nothing can be installed.
- The module path is `tools/hn_kernel/` (for `wsp`), then `packet/` (`renderer`, `tokenizer`), then
  the script's directory, then that directory's parent. So a script in `output/wsp-a/scratch/` can
  `import renderer` and `import build_edit`.
- Every script runs under the kernel-client guard. **Allowed:** reading weights through `wsp`,
  matrix products and `torch.linalg` (`svd`, `eigh`, `qr`, `norm`, ...), numpy, reading files under
  your directory, writing files under `output/wsp-a/` (use `numpy.save` / `numpy.load` for `.npy`
  and `safetensors.torch.save_file` / `load_file` for `.safetensors`), and the one connection to
  the kernel socket that `wsp` opens.
- **Refused, and fatal to the run** (the run exits with status 3 and prints `HN_GUARD: refused: ...`,
  even if your code catches the error):
  - autograd: grad mode, `.backward()`, `torch.autograd.grad`, `requires_grad_(True)`, forward-mode
    AD, `torch.func` and `torch.vmap`; optimisers (`torch.optim`); compilers (`torch.compile`,
    TorchScript, `torch.export`), and `torch.use_deterministic_algorithms`, which imports one;
  - least squares and pseudo-inverses (`torch.linalg.lstsq`, `pinv`, `numpy.linalg.lstsq`, ...);
  - random numbers (`random`, `numpy.random`, `torch.rand*`, `torch.randn`, `torch.normal`,
    `torch.Generator`, ...). The renderer's own seeded per-event noise source is exempt; for a random
    control direction use `control_direction` (below); for quasi-random vectors use `hashlib`;
  - imports of `transformers`, `peft`, `accelerate`, `sklearn` and `scipy.optimize`;
  - threads, subprocesses, signals, `ctypes`, `mmap`, sockets other than the kernel's, reading
    outside your directory and the interpreter install, writing outside `output/wsp-a/`, and
    `torch.load` of zip-format files.

## The kernel client: `from wsp import ...`

```python
from wsp import (load, load_original, label_rows, save_edit, save_label_rows, drop_edit, edited,
                 edit_hash, budget, load_quota, NAMES, SHAPES, ARCH, LABEL_IDS, CLASS_NAMES, tok,
                 svd, head_slice, top_writers, project_out, rank1, zero_head, label_lens, dla,
                 diff_means, control_direction)
```

Tensor names are the checkpoint's, `model.language_model.*`. Every tensor comes back as a CPU fp32
`torch.Tensor`, a copy you may modify.

### Metadata

| Name | Value |
|---|---|
| `NAMES` | The 320 tensor names, a tuple |
| `SHAPES` | `{name: shape tuple}` |
| `ARCH` | A dict: `hidden_size`, `intermediate_size`, `num_hidden_layers`, `vocab_size`, `layer_types`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `attn_output_gate`, the `linear_*` DeltaNet sizes, `full_attention_interval`, `rms_norm_eps`, `tie_word_embeddings`, `hidden_act`, `label_ids`, `class_names`, the edit scope and the load quota |
| `LABEL_IDS` | The four label token ids, class order |
| `CLASS_NAMES` | `("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING")` |
| `tok` | A `tokenizers.Tokenizer` from `packet/qwen/tokenizer.json`. Use `tok.encode(text, add_special_tokens=False).ids`, as the readout does (`INTERFACE.md`) |

### Reading and editing

| Function | Semantics |
|---|---|
| `load(name) -> Tensor` | The current tensor: your edit if there is one, else the original |
| `load_original(name) -> Tensor` | Always the original |
| `label_rows(original=False) -> Tensor` | The four label rows of the tied embedding, `[4, 1024]`, class order |
| `save_edit(name, tensor) -> EditReceipt` | Replace a whole tensor (fp32, same shape, finite). Refused outside the edit scope (below); a tensor equal to its original removes that tensor's edit. For the embedding, only the label rows may differ (prefer `save_label_rows`) |
| `save_label_rows(rows) -> EditReceipt` | Write only the four label rows, `rows` `[4, 1024]` fp32 in class order |
| `drop_edit(name) -> bool` | Restore one tensor; `False` if it was not edited |
| `edited() -> list[str]` | Names of the edited tensors |
| `edit_hash() -> str` | The edit-state hash (sha256 over the sorted edited tensors; the empty edit has a fixed hash). Every forward pass is stamped with it |
| `budget() -> dict` | `{"tokens_used", "tokens_left", "requests"}` of the forward-pass budget |
| `load_quota() -> dict` | `{"bytes_used", "bytes_left", "bytes_quota", "tensors"}` of the weight-load quota |

`EditReceipt` fields: `name`, `changed_entries`, `changed_rows`, `changed_cols`, `rel_frobenius`
(`‖ΔW‖_F / ‖W‖_F`), `delta_rank` (singular values of `ΔW` above `1e-6 · ‖ΔW‖₂`), `sha256` (of the
saved tensor), `label_rows` (true for a label-row edit; then every count is over the `[4, 1024]`
block), `bf16_changed_entries` and `edit_hash` (after the save).

Errors are exceptions: `EditRefused`, `BudgetRefused`, `RequestError`, `QuotaRefused`, all
subclasses of `KernelError`. A refused request changes neither the edit nor the meter.

**Weight-load quota.** `load` / `load_original` of a layer tensor (every tensor but the embedding and
the final norm) that you have not loaded before in this phase draws on a quota of
996,224,640 bytes of distinct tensors, half of the layer weights. Loading the same tensor
again is free; the embedding, the final norm and `label_rows` never count. The helpers `svd`,
`head_slice`, `top_writers` and `zero_head` call `load`. When the quota is used up, `QuotaRefused`.

### Weight-only helpers (plain torch; none saves anything)

| Function | Semantics |
|---|---|
| `svd(name, k)` | Top-`k` `(U [m, k], S [k], Vh [k, n])` of the current tensor reshaped to `[shape[0], -1]`, computed in float64, returned fp32 |
| `head_slice(layer, head, kind, W=None)` | `HeadSlice(name, dim, start, stop, weight)`: one head's block, `weight == W.narrow(dim, start, stop - start)` (`W` defaults to `load(name)`). Full-attention layers: `head` is a query head 0..7 and `kind` one of `q`, `k`, `v`, `gate` (rows) or `o` (columns of `o_proj`); `k` and `v` give its shared KV head (`head // 4`). DeltaNet layers: `head` is a value head 0..15 and `kind` one of `q`, `k`, `v`, `z`, `a`, `b` (rows) or `o` (columns of `out_proj`) |
| `zero_head(layer, head)` | `(name, W')`: the layer's output projection with that head's input columns zeroed |
| `top_writers(name, direction, k=10)` | `(indices, signed scores)` of the `k` units whose weight vectors align most with `direction`, by absolute score: columns if `direction` has `shape[0]` entries (what each input unit writes), else rows |
| `project_out(W, u, cols=None, alpha=1.0)` | `W − alpha · û (mask ⊙ (ûᵀW))` with `û = u / ‖u‖` in the row space; `cols` limits it to those columns |
| `rank1(W, a, b)` | `W + a bᵀ` |
| `label_lens(h, original=False)` | The four label logits a residual `h [..., 1024]` would give: the final `(1 + w)` RMSNorm, then the label rows. On a `./fwd` `hidden` array at layer 24 it reproduces the readout |
| `dla(vec, resid=None, original=False)` | Direct label attribution of a residual contribution: `(vec ⊙ (1 + w)) / rms(resid) @ rowsᵀ` (RMS factor 1 without `resid`) |
| `diff_means(a, b)` | `mean(a, 0) − mean(b, 0)` |
| `control_direction(seed, like)` | A seeded random tensor with `like`'s shape and Frobenius norm, for controls. The only randomness allowed |

### Edit scope

- In scope: the 320 `model.language_model.*` tensors, except:
  - `model.language_model.embed_tokens.weight`: only the four label rows (`LABEL_IDS`) may change;
  - `model.language_model.norm.weight` (the final norm): refused.
- Refused: non-finite values; a shape or dtype mismatch; and a rescaling. With `new = c·W + R`,
  `c = ⟨new, W⟩ / ⟨W, W⟩`, an edit is refused when `|c − 1| > 1e-3` and `‖R‖ < 0.5 · ‖(c − 1)·W‖`
  (this includes zeroing a whole tensor); for `input_layernorm`, `post_attention_layernorm`, `q_norm`
  and `k_norm` (which multiply by `1 + w`) also on the gain `1 + w`; for the label rows also on the
  row-centred `[4, 1024]` block and on the changed rows when two or more changed. Rescaling one label
  row alone is allowed.
- The label rows are tied: they are also the input embeddings of the four label tokens (`INTERFACE.md`).

## Forward passes (`./fwd`)

`./fwd output/wsp-a/scratch/req_0001.json`. The request file must be a regular file inside your
directory (not a symlink, not in `.kernel/`), UTF-8 JSON, at most 8 MiB:

```json
{
  "model": "edited",
  "sessions": [
    {"id": "a1", "blocks": ["<block 0 text>", "<block 1 text>", "<block 2 text>"], "windows": [1, 2]}
  ],
  "return": {"hidden": [12, 24], "mlp_act": [20], "topk": 5}
}
```

- `model`: `"edited"` (your current edit), `"original"` or `"both"`.
- `sessions`: a non-empty list. Each has exactly `id` (a distinct string of 1 to 128 characters),
  `blocks` (a non-empty list of rendered block strings, one per tool call, as
  `renderer.render_session` returns them, without the trailing newline) and `windows` (block
  indices, strictly increasing, each ending a window to read out; the last must be the last block,
  because blocks after it would never run).
- `return` (optional): `hidden`, distinct layers in 0..24 (the residual after that many layers at the
  readout position: 0 = embeddings, 24 = before the final norm); `mlp_act`, distinct layers in 0..23
  (`silu(gate_proj x) ⊙ up_proj x`, the input of `down_proj`); `topk`, an int in 0..20. No other keys.
- Cost: per session 167 prefix tokens + the Qwen tokens of blocks 0..last window +
  11 per window; twice that for `both`. At most 50,000 tokens per request and
  2,000,000 per phase. A request over either, or invalid, is refused and costs nothing.

The result goes to `output/wsp-a/fwd/<call id>/`: `response.json`, `request.json` (a copy) and one
`.npy` per requested activation. `response.json`:

```json
{
  "call_id": 7, "phase": "wsp-a", "model": "both", "edit_hash": "<sha256>", "tokens": 1234,
  "budget": {"tokens_used": 56789, "tokens_left": 1943211, "requests": 7},
  "class_names": ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"],
  "sessions": [
    {"id": "a1", "windows": [1, 2],
     "edited":   {"label_logits": [[...4], [...4]], "logsumexp": [..], "label_argmax": [0, 2],
                  "argmax": [25399, 65552], "topk": [[...5], [...5]]},
     "original": {"...": "the same fields"},
     "delta": {"label_logits": [[...4], [...4]], "logsumexp": [..], "label_argmax_flips": 1,
               "argmax_flips": 1}}
  ],
  "request_file": "output/wsp-a/scratch/req_0001.json",
  "arrays": {"edited.s0.hidden.L24": {"file": "edited.s0.hidden.L24.npy", "shape": [2, 1024]}}
}
```

- Per window, in `windows` order: `label_logits` (the four label logits, class order),
  `logsumexp` (the full-vocabulary log-normaliser), `label_argmax` (the class index), `argmax` (the
  full-vocabulary argmax token id) and, if asked, `topk` (token ids only).
- `delta` (only for `both`): edited minus original, and the counts of windows whose class
  (`label_argmax_flips`) or full-vocabulary argmax (`argmax_flips`) changed.
- Arrays: `<model>.s<session index>.<hidden|mlp_act>.L<layer>.npy`, shape `[n_windows, 1024]` for
  `hidden` and `[n_windows, 3584]` for `mlp_act`, fp32, at the readout position of each window.
- `./fwd` prints a short summary (call id, tokens, budget, class indices); it never prints vocabulary
  strings. Exit status 0 on success, 1 if the server refused the request, 2 if the request file was
  invalid.
- The harness's ledger records every request that reaches the model server, run or refused: call
  id, time, tokens, cumulative tokens, edit hash, the sha256 of each session's token ids and
  windows, the layers requested and the request file. Cite call ids in `EDIT.md` and
  `ITERATIONS.md`. A request file `./fwd` rejects as invalid (exit status 2) is not sent: it costs
  nothing, has no call id and appears only in your session log.
- Scoring runs the same readout in fp32, possibly on another device: do not rely on label-logit
  margins of a few thousandths.

## Operation records (`RawOp` dicts, the renderer's input)

`renderer.render_session(ops, session_id)` returns one block string per operation; `ops` is a list
of plain dicts, one per tool call, with these keys (a missing key reads as null; `session_id` is any
string, and it seeds the hunk ids and error line numbers of SPEC.md §4.3):

| Key | Type | Meaning |
|---|---|---|
| `family` | str | The tool name: `read_file`, `search`, `list_dir`, `edit_file`, `write_file`, `revert`, `run_tests`, `build`, `lint`, `http_call`, `poll_status`, `plan`, `ask`, `yield` |
| `target` | str, list or dict | What the call acts on: the path (`read_file`, `list_dir`, `edit_file`, `write_file`), the scope (`search`), the list of paths (`revert`), the suite (`run_tests`), the target (`build`, `lint`), `{"method", "host", "path_template"}` (`http_call`), the resource (`poll_status`) |
| `args` | dict | Call arguments the header shows: `line_start` / `line_end` (`read_file`), `content` (`write_file`; not shown in the header), `query` / `limit` (`search`), `tests` (a list, `run_tests`), `body` (`http_call`), `backoff` (`poll_status`), `action_type` (`plan`, `ask`, `yield`) |
| `outcome` | str | `SUCCESS`, `ERROR` or `NEUTRAL`. The status reads `error` for `ERROR` and `ok` otherwise; `run_tests` / `build` / `lint` read `FAILED` / `ok`, and `poll_status` reads `ready` / `error` / `pending` |
| `error_class`, `error_location` | str or null | The error line of an `ERROR` outcome: `! {error_class} at {error_location}:{line} (...)` |
| `validation` | `{"suite", "passed", "total"}` or null | The validation result; shown as `{passed}/{total} passed` |
| `changed` | list of `{"path", "before", "after"}` | Every tracked file whose text differs between the start and the end of the call, sorted by path; `before` / `after` is the full text, or null when the file is absent. Rendered as hunk lines (SPEC.md §4.3) |
| `extra` | dict or null | Tool-specific detail that the header does not show |
| `input_tokens`, `output_tokens` | int or null | Shown as `in=` and `out=` |
| `timestamp` | str or null | ISO 8601 (`2026-03-01T09:14:03.250+00:00`); shown in UTC as `HH:MM:SS.mmm` |
| `dropped` | sorted list of str | Observations withheld on this call, each shown as `?` in its own position: any of `digest_after`, `digest_before`, `failure`, `input_tokens`, `output_tokens`, `validation_passed`, `validation_suite`, `validation_total` (a dropped digest adds the `[workspace untracked]` line) |

Each string `renderer.render_session` returns is one block for `./fwd`.

## Method

- **Banned:** backward passes, autograd and optimisers; fitting weights to a loss, in closed form
  or iteratively (regression, least squares, pseudo-inverse fits, gradient steps); any automated or
  numerical search over candidate weights, including keep-the-best loops, random restarts and sweeps
  scored against an objective, whether the loop is a script or a run of tool calls; distilling any
  model's outputs into the weights.
- **Allowed:** direct, reasoned edits with a stated mechanism: a rank-1 update along a direction you
  identified, projecting a direction out of a matrix, writing a readout direction into the label
  rows; reading weights, SVDs, attributions and activations of your own texts and deriving directions
  from them (means, differences of means, projections); testing an edit with a few forward passes and
  revising it by reasoning.
- A loop that tries candidate weights and keeps the best is search. The harness counts how many
  distinct edit states were run on the same texts; more than three goes to the adjudicator.

## Deliverables

All in `output/wsp-a/`.

### `build_edit.py`

```python
import torch


def build(orig, consts):
    W = orig["model.language_model.layers.20.mlp.down_proj.weight"]  # a fresh fp32 copy of the original
    u = consts["u_stall"]                                                # from edit_constants.safetensors
    ...
    return {"model.language_model.layers.20.mlp.down_proj.weight": W_new}
```

- A module-level `build(orig, consts)`. `orig` is a read-only mapping from each of the 320 tensor
  names to its **original** fp32 weights; every access hands out a fresh copy. `consts` maps the
  names in `edit_constants.safetensors` to their tensors (empty if there is no such file).
- Return a `dict` of **only the replaced tensors**: each full-size, fp32, finite, on CPU, with no
  autograd state, and different from the original. To edit the label rows, return the whole
  embedding (`orig["model.language_model.embed_tokens.weight"]` with only rows `LABEL_IDS` changed).
  Every returned tensor must pass the edit-scope rules above.
- It runs in a fresh interpreter on CPU with one thread and `torch.use_deterministic_algorithms(True)`,
  under the build guard: matrix products and `torch.linalg` are allowed (not `lstsq` or `pinv`);
  refused are calling any model or `nn.Module`, every `torch.nn.functional` function, softmax, random
  numbers, autograd, optimisers, compilers, threads, subprocesses, sockets and every file access
  (constants come only through `consts`). Use the standard library, `torch` and `numpy`; do not
  import `wsp` (there is no kernel at build time), `renderer` or your scratch modules.
- At most 200 KB; at most 300 s of CPU (600 s wall).
- **It must reproduce the server's edit exactly.** The harness requires the built tensors to equal
  the model server's final journalled edit state, tensor for tensor and byte for byte, and builds
  twice, in two fresh interpreters with different hash seeds, requiring identical bytes. So apply
  your final edit by running `build()` itself from a `./py` script and saving what it returns:

  ```python
  from collections.abc import Mapping

  import torch
  from safetensors.torch import load_file
  from wsp import NAMES, edited, drop_edit, load_original, save_edit, save_label_rows, LABEL_IDS
  import build_edit

  torch.set_num_threads(1)  # the build's thread count


  class Orig(Mapping):  # like the harness's orig: every name, a fresh original copy per access
      def __getitem__(self, name):  # loads a tensor only when build() asks for it
          if name not in NAMES:
              raise KeyError(name)
          return load_original(name)

      def __iter__(self):  # sorted, as the harness's
          return iter(sorted(NAMES))

      def __len__(self):
          return len(NAMES)


  consts = load_file("output/wsp-a/edit_constants.safetensors")  # or {} if you have none
  new = build_edit.build(Orig(), consts)
  for name in edited():
      if name not in new:
          drop_edit(name)
  for name, t in new.items():
      if name == "model.language_model.embed_tokens.weight":
          save_label_rows(t[list(LABEL_IDS)])
      else:
          save_edit(name, t)
  ```

  Do not add `torch.use_deterministic_algorithms(True)` to this script: the `./py` guard refuses it
  (above). On CPU that mode does not change matrix products, `torch.linalg` or elementwise
  arithmetic, but it fills the memory of `torch.empty` and the other `empty*` constructors with
  NaN, so never read a tensor's entries before writing them. Results that depend on thread count
  or summation order are fragile: prefer elementwise arithmetic and `torch.outer`, or accumulate in
  float64 and cast to fp32 at the end.

### `edit_constants.safetensors` (optional)

At most 8,388,608 bytes (8 MiB) of named fp32 tensors, for vectors that cannot be written
as closed-form rules, such as directions derived from `./fwd` activations. Write it from a `./py`
script with `safetensors.torch.save_file`. `EDIT.md` must cite, for every constant, the ledger call
ids of the forward passes it was derived from, and how.

### `EDIT.md`

- What changed: every edited tensor, with its rows, columns or rank.
- The mechanism claimed for each change, stated so that it can be checked: what the edited
  component reads, what it writes, and what the readout should do differently on which kind of
  window.
- For every vector or constant, the ledger call ids behind it.
- It **ends with a section headed exactly `## Mechanisms`**: one Markdown table with exactly these
  four columns, in this order, and one row per mechanism you claim:

  ```text
  ## Mechanisms

  | id | tensors | target class | claim |
  | --- | --- | --- | --- |
  | M1 | model.language_model.layers.20.mlp.down_proj.weight | STALLED | One sentence: what it makes the readout do on which windows. |
  | M2 | label rows | OSCILLATING | One sentence. |
  ```

  - `id`: a short unique id (`M1`, `M2`, ...).
  - `tensors`: the full names (as in `NAMES`) of the edited tensors that implement the mechanism,
    comma-separated, or `label rows` for the four label-word rows of the embedding.
  - `target class`: exactly one of `PRODUCTIVE`, `RECOVERY`, `STALLED`, `OSCILLATING`: the class
    the mechanism makes the model recognise.
  - `claim`: one sentence.

  The harness checks each row on its own: with your whole edit the model should name the target
  class on windows of that class more often than the original model does, without naming it more
  often on windows of other classes; with only that row's tensors reverted to the original weights,
  most of that improvement should go away. A row is checked only if every tensor it names is edited,
  it does not name the whole edit, and it shares no tensor with any other row. So give each mechanism
  its own tensors, give every edited tensor to at most one row, and note that an edit of a single
  tensor (or of the label rows alone) leaves no row that can be checked.

### `ITERATIONS.md`

Every iteration in order: the edit hash, a summary of the change, the reason, and the forward-pass
call ids that tested it with what they showed (the texts you used, or how you built them).

## What the harness does with your files

You do not build, archive, hash or commit anything. When you state that a submission is final, the
harness reads `output/wsp-a/` and:

1. checks the packet and the kernel tools against their registered hashes;
2. requires `build_edit.py`, `EDIT.md` and `ITERATIONS.md`;
3. copies `build_edit.py` and `edit_constants.safetensors` **alone** into a fresh private directory
   and runs `build()` there in a fresh interpreter under the build guard, so `build()` must not
   depend on any other file of yours;
4. validates the result (names in scope, shapes, fp32, finite, no unchanged tensor, label rows only,
   no rescaling), rebuilds it in a second fresh interpreter and requires identical bytes;
5. requires it to equal the server's final journalled edit state;
6. archives the edit, an exact diff and a build record (per tensor: changed entries, rows and
   columns, `‖ΔW‖_F / ‖W‖_F`, rank, the bf16-rounded changed count; the overall changed fraction; the
   label-row flag), with your whole phase directory.

If a build is rejected you receive only the harness's error text, and you may fix what it names.
For an exception raised by your code that text gives the exception type and the line numbers in your
file, not its message, so reproduce the failure with `./py` to see the message.

## WSP-A (zero labels)

- **Allowed:** forward passes on texts you write yourself (hand-written block text, or operation
  records you build and render with `packet/renderer.py`), reading weights, and direct, reasoned
  edits. Compare the readout under your edit with the original (`"model": "both"`) and revise.
- **Forbidden:** any other data, and any training, fitting, search or distillation (Method above).
  You receive no labelled data and no score.
- Log every iteration in `ITERATIONS.md`.
- A cost cap applies to the whole of WSP-A, including continuations and resubmissions. The result is
  the first valid artefact you submit as final.

## Later phases

A one-revision phase (WSP-B, in `output/wsp-b/`) starts only if you are told it has, with its own
instructions, forward-pass budget and cost cap.
