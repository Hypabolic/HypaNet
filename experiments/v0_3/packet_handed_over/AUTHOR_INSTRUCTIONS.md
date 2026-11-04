# Weight-author instructions (spec v0.3, §9)

You are the **weight author**. Your inputs are the files in `packet/` and nothing else. You work in
your own directory; you never read anything outside it.

`SPEC.md` defines the task, the model, the log grammar and the deliverables. This file is the
operational protocol: the exact interfaces the harness calls, the rules it enforces and how to run
things. Where the two seem to differ on *how* something is built, run or archived, follow this file.

## Your directory

| Path | What it is |
|---|---|
| `packet/` | This packet (read-only). `PACKET_MANIFEST.json` holds the sha256 of every file |
| `packet/model.py` | The fixed transformer (`DoomLoopTransformer`, `zero_model`, `PARAM_SHAPES`, constants) |
| `packet/tokenizer.py` | The frozen tokenizer (`encode_session`, `tokenize_block`, `word_id`, vocabulary constants) |
| `packet/renderer.py` | The frozen renderer (`render_session`, `render_event`); it reads plain dicts (below) |
| `packet/state_dict_template.json` | Tensor names, shapes and parameter count |
| `packet/GRAMMAR_SHEET.md` | One invented example block per tool, with its tokens |
| `tools/` | The `./py` launcher and its guard (read-only) |
| `output/v03-phase-a/` | Phase A deliverables; put scratch scripts in `output/v03-phase-a/scratch/` |
| `output/v03-phase-b/` | Phase B deliverables (only when Phase B starts) |

Never write inside `packet/` or `tools/`. The harness archives your whole phase directory,
including `scratch/`.

## Running Python

Run every script with the wrapper, from your directory:

```
./py output/v03-phase-a/scratch/check_logs.py
./py -m py_compile output/v03-phase-a/build_weights.py
```

- `./py` runs `.venv` Python 3.12 with `torch==2.2.2` and `numpy<2` (CPU, float32). No other
  packages are installed, and nothing can be installed.
- The module path is `packet/` first, then the script's directory, then that directory's parent.
  So `import model`, `import tokenizer` and `import renderer` work everywhere, and a
  script in `output/v03-phase-a/scratch/` can `import build_weights` and `import reference_policy`.
- Every script runs under the author guard. Forward passes of the model, matrix products and
  `reference_policy` calls are allowed. Refused, and fatal to the run: autograd (grad mode,
  `.backward()`, `torch.autograd`), `torch.optim`, any random-number API (`random`,
  `numpy.random`, `torch.rand*`, `torch.normal`, `torch.Generator`, in-place `.normal_()` and the
  like, dropout), threads and subprocesses, network access, reading or listing anything outside your
  directory, and writing outside your directory. The renderer's own seeded per-event noise
  source is the one allowed use of `random.Random`. For quasi-random vectors use `hashlib`.
- **One command per Bash call.** Only commands that start with `./py ` are allowed, and compound
  commands are refused: no `;`, `&&`, `||`, pipes, redirection (`>`, `<`), backticks, `$(...)` or
  `echo`. Write a script to a file with the file-writing tool, then run it with `./py <file>`.

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

## Deliverables and pinned interfaces

Put them in `output/v03-phase-a/` (Phase A) or `output/v03-phase-b/` (Phase B).

### `reference_policy.py`

```python
def reference_policy(tokens: list[str], event_ord: list[int]) -> int:
    ...
```

- `tokens` holds the tokenizer's **surface strings** for a session prefix (`Encoded.surface`:
  `"<BOS>"` first, `"<EOE>"` at the end of every block, `"\n@"` / `"\n!"` / `"\n["` / `"\n"`
  for line markers), and `event_ord` the matching event ordinals (`Encoded.event_ord`). The prefix
  always ends with the `<EOE>` of the block being classified.
- Return an `int` (not a `bool`) in `0..3`: `PRODUCTIVE`, `RECOVERY`, `STALLED`, `OSCILLATING`. It
  states your intended policy for the **last** block of the prefix.
- Standard library only; it may not import `tokenizer`, `model`, `renderer`, `torch` or `numpy`.
  Pure and deterministic: no I/O, no clock, no randomness, no `global` or `nonlocal` statement, and
  no function may change module-level state (no item assignment, `.append()`, `.update()` and the
  like on a module-level list, dict or set, no attribute assignment on a module-level name).
  Module-level constants are fine; build per-call tables inside the function.
- Both Python deliverables: no interpreter internals. Dunder names other than `__name__`,
  `__doc__`, `__all__`, `__init__`, `__post_init__`, `__repr__` and the like are refused
  (`__file__`, `__dict__`, `__globals__`, `__builtins__`, `__class__`, ...), as are strings that
  spell them, `vars()` / `locals()` / `globals()`, and `getattr` / `setattr` / `hasattr` /
  `delattr` unless the attribute name is a literal string (or the loop variable over a literal
  tuple of strings). Special methods other than `__init__`, `__post_init__` and `__repr__` may not
  be defined.
- The harness calls `reference_policy` only in a fresh interpreter under the build guard (no I/O,
  no randomness), with the CPU limit per call.
- At most 0.5 s of CPU per call. The harness smoke-calls it once, on the tokens of the grammar
  sheet's `read_file` block, as a type and time check (the value is not recorded). A timeout in
  evaluation counts as a disagreement.

### `build_weights.py`

```python
import torch
import model
import tokenizer


def build() -> model.DoomLoopTransformer:
    m = model.zero_model()
    with torch.no_grad():
        ...  # explicit constants and closed-form rules
    return m
```

- A module-level `build()` with no arguments that starts from `model.zero_model()`, sets
  parameters by closed-form rule or explicit constants, and returns that model. Keep every tensor's
  name, shape and float32 dtype; every value must be finite.
- Imports allowed: `torch` and `torch.nn`, `model`, `tokenizer`, and the standard-library modules
  `math`, `hashlib`, `itertools`, `functools`, `collections`, `dataclasses`, `typing`, `string`,
  `re`. Nothing else (in particular not `numpy`, `renderer` or `reference_policy`).
- The tokenizer may be called only on literal word lists: arguments must be literals (`"path"`,
  `["path", "lines"]`), a module-level name assigned once to a literal list or tuple of strings, or
  the loop variable of a `for` or comprehension over one. Its constants (`V`, `EOE`,
  `RESERVED_IDS`, ...) can be read freely.
- Quasi-random vectors must come from `hashlib` (for example the bits of `sha256(name)`).
- Allowed: indexing and index assignment, slicing, `index_put_`, `fill_`, `copy_`, `clone`,
  `torch.zeros` / `ones` / `full` / `tensor` / `arange` / `eye`, elementwise and scalar
  arithmetic, reductions, `torch.outer` / `Tensor.outer` and `torch.kron`.
- Forbidden (static scan and run-time tripwires, even inside `try`): `random`, `numpy.random`,
  `torch.rand*`, `torch.Generator` and every other random-number API; all of
  `torch.nn.functional`; `softmax` / `log_softmax`; `matmul` and the `@` operator; `mm`, `bmm`,
  `addmm`, `addr`, `ger`, `baddbmm`, `addbmm`, `addmv`, `mv`, `dot`, `vdot`, `inner`, `tensordot`,
  `einsum`, `chain_matmul`, `bilinear`, `torch.linalg`; `torch.ops`, `torch._C`; calling the model
  or any `nn.Module` (`m(x)`) and any `forward` / `forward_batch` / `activations` / `_run` call;
  autograd and `torch.optim`; I/O of any kind (`open`, `print` to files, `pathlib`, `os`, `sys`,
  `io`, `subprocess`); `eval`, `exec`, `compile`, `__import__`, `importlib`, `globals`; the
  modules `base64`, `struct`, `zlib`, `pickle`, `array`, `marshal`, `json`. Names containing
  `matmul`, `linalg`, `einsum` or `softmax` are refused too, so do not use them for your own
  variables.
- Size limits: no list, tuple, set or dict literal with more than 256 elements, no string or bytes
  literal longer than 512 characters (docstrings count, so keep long prose in `CIRCUITS.md`), and
  the file is at most 200 KB.
- The build must finish within 300 s of CPU, and two builds in two fresh interpreters must give
  byte-identical weights.
- Never evaluate the network, or any part of it, inside `build_weights.py`, however it is written
  (for example a forward pass assembled from elementwise products and sums). Weights follow from
  your stated policy by assignment or closed-form rules. Run forward passes only from your own
  scratch scripts.

### `CIRCUITS.md`

- The residual-subspace allocation (which dimensions carry what).
- Each head's role: its query, key and value subspaces and its event-distance bias pattern.
- MLP units grouped by circuit.
- Which `reference_policy` branch each logit implements.
- An account of every non-zero tensor block.

### `ITERATIONS.md` (Phase A) / `REVISION.md` (Phase B)

- `ITERATIONS.md`: every self-check iteration, with a summary of the diff, the reason and the
  inputs used (each hand-written log, or the operation-record sequence you rendered).
- `REVISION.md`: what was disclosed to you, and every change with its reason and the disclosed
  windows that motivated it.

## What the harness does with your files

You do not build, archive, hash or commit anything. When you state that a submission is final,
the harness reads your phase directory and:

1. checks the packet against its registered hash;
2. requires `build_weights.py`, `reference_policy.py`, `CIRCUITS.md` and `ITERATIONS.md`
   (Phase A) or `REVISION.md` (Phase B);
3. runs the static rules above on both Python files;
4. copies `build_weights.py` **alone** (with the packet) into a fresh private directory and runs
   `build()` from there in a fresh interpreter under the build guard (the bans above, no I/O, 300 s
   CPU), validates the state dict (exact tensor names in order, shapes, float32, finite, no
   autograd state) and rebuilds in a second fresh interpreter, requiring identical bytes. So
   `build()` must not depend on any other file of yours;
5. smoke-calls `reference_policy` once (above), again from a private copy of that file alone;
6. archives the artefact, its sha256 and your whole phase directory, `scratch/` included.

If a build is rejected you receive only the harness's error text, and you may fix what it names.
For an exception raised by your code that text gives the exception type and the line numbers in
your file, not the exception message, so reproduce the failure with `./py` to see the message.

## Phase A (zero labels)

- **Allowed:** forward passes of your network and calls of your `reference_policy` on logs you
  write yourself: hand-written block text, or `RawOp` sequences you build and render with
  `packet/renderer.py`. Compare the two and revise weights and policy by direct, reasoned edits.
- **Forbidden:** any other data; and automated search of any kind: fitting, gradient steps,
  random restarts, parameter sweeps scored against an objective, or distilling anything into the
  weights (your own `reference_policy` included). A loop that tries candidate weights and keeps the
  best is search.
- Log every iteration in `ITERATIONS.md`.
- A cost cap applies to the whole of Phase A, including continuations and resubmissions. The
  result is the first valid artefact you submit as final.

## Phase B (one diagnostic revision)

Phase B starts only if you are told it has. You then receive a fixed report of 200 validation
windows (for each: the rendered prefix with the window's first block marked, the true label,
your Phase A network's class and your `reference_policy`'s class) and nothing else. You make one
submitted revision in `output/v03-phase-b/`, logging every iteration in `REVISION.md`. Phase B has
its own cost cap.
