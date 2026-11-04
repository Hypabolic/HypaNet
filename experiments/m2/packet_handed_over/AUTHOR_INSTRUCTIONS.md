# Weight-author instructions (spec v0.2, §9)

You are the **weight author**. You work in a fresh context. Your only inputs are
the files in this packet directory and your own output directory. Do not open
anything else: no repository code, no generator or oracle code, no datasets and
no evaluation output.

`SPEC_AUTHORING.md` defines the experiment, the features and what your
deliverables must contain (§9.1). This file is the operational protocol: the
environment, where you write, the exact interfaces the harness calls and the
rules it enforces. Where the two seem to differ on *how* something is built,
run or archived, follow this file. In particular, `build_weights.py` does not
write the artefact itself: the harness saves it (see below).

## Packet contents

| File | Contents |
|---|---|
| `SPEC_AUTHORING.md` | The experiment specification prepared for the weight author |
| `model.py` | The immutable network: the harness model module with its two schema constants written inline, so it runs standalone from this directory |
| `feature_index.json` | Window slot x feature -> flat input index, feature names and definitions, class order |
| `state_dict_template.json` | Tensor names, shapes, parameter count; the initial state is all zeros |
| `AUTHOR_INSTRUCTIONS.md` | This file |
| `MANIFEST.json` | sha256 of every file above and the packet hash |

The network is `Linear(120, 24) -> ReLU -> Linear(24, 12) -> ReLU -> Linear(12, 4)`,
float32 on CPU. Its input is 8 events x 15 features, oldest event first, laid
out as `feature_index.json` describes (`flat_index = slot * 15 + feature_index`,
where the slot is called `position` in `feature_index.json`; slot 0 is the
oldest event, slot 7 the window endpoint). Its output is four raw
logits in the canonical class order
`[PRODUCTIVE, RECOVERY, STALLED, OSCILLATING]` (indices 0, 1, 2, 3). The
prediction is the argmax, and ties go to the first maximal index, so an
all-zero output predicts PRODUCTIVE.

## Environment

- Python 3.12, `torch==2.2.2`, `numpy>=1.26,<2`. CPU only, float32.
- The packet's `model.py` runs standalone: it imports only the Python standard
  library, `numpy` and `torch`. Do not install or import the harness package.
- To run your files, put the packet directory on the module path and turn off
  bytecode caching, for example
  `PYTHONPATH=<packet directory> PYTHONDONTWRITEBYTECODE=1 python <file>`
  (or `python -B`). Never write anything into the packet directory, not even a
  `__pycache__/` folder: the packet is verified against `MANIFEST.json`, and a
  stray entry makes it fail verification.

## Where to write

The adjudicator gives you a separate output directory, outside the packet, with
one subdirectory per phase:

- `<output>/phase-a-strict/`
- `<output>/phase-a-selfcheck/`
- `<output>/phase-b-revision/`

Write your deliverables only there, never inside the packet. You read only the
packet and your own output directory.

You do not archive, hash or commit anything. The adjudicator's harness reads
your phase directory, checks and runs your files, saves the artefact
(`artifact.dlw`), hashes it, copies your files into the archive and records
everything in a build record; the adjudicator git-commits the archive. The
A-strict artefact is built and committed by the harness **before A-selfcheck
begins**.

## Deliverables and pinned interfaces

### `reference_policy.py`

```python
def reference_policy(x: tuple[float, ...]) -> int:
    ...
```

- `x` is a tuple of exactly 120 Python floats, each exactly representable in
  float32, ordered as in `feature_index.json`.
- Return an `int` (not a `bool`) in `0..3`, using the canonical class order
  above. This function states your intended policy.
- Standard library only, deterministic, no I/O (no files, console, network or
  clock), no data access, no global mutable state (no `global` statements; do
  not mutate module-level objects).
- Smoke call: the harness loads the file under the guard and
  calls `reference_policy` exactly once, on the all-zero 120-tuple, as a type
  check (it must return an `int` in `0..3`); the return value is not recorded.
  It is not a behavioural test, and it is the only call the harness makes.

### `build_weights.py`

```python
import torch
from model import zero_model, DoomLoopMLP


def build() -> DoomLoopMLP:
    m = zero_model()
    with torch.no_grad():
        ...  # literal assignments or closed-form rules
    return m
```

- Define a module-level function `build()` that takes no arguments and returns
  a `DoomLoopMLP` from the packet's `model.py`. Import it as
  `from model import zero_model, DoomLoopMLP`; the harness puts the packet
  directory first on `sys.path` before loading your file.
- Start from `zero_model()` (every tensor exactly 0.0) and set `Linear.weight`
  and `Linear.bias` entries by literal assignment or closed-form rules, under
  `torch.no_grad()`. Deterministic loops that fill known circuits are fine.
  Every tensor must keep its name, shape and float32 dtype, and every value
  must be finite.
- No forward passes, matrix products or linear-algebra calls anywhere in this
  file (the compute ban below). `build()` writes numbers into the tensors; it
  does not compute with the network or fit anything. Elementwise and scalar
  arithmetic on tensors and arrays, and index assignment, are fine. In
  A-selfcheck, run your forward passes from a separate script of your own,
  never from `build_weights.py` (not even its `__main__` block).
- No module-level side effects beyond imports, definitions and constants. The
  harness executes the whole file under the anti-training guard, so
  module-level code is checked exactly like `build()`.
- Do not write or read any file. The harness saves the artefact from the model
  `build()` returns, so do not call `save_artifact` or `torch.save`.
- Optionally, an `if __name__ == "__main__":` block at the end of the file may
  print sparsity statistics. The import above does not bind the name `model`,
  so import `sparsity` inside the block:

  ```python
  if __name__ == "__main__":
      from model import sparsity
      print(sparsity(build()))
  ```

  The harness does not run that block, but it is checked by the same rules as
  the rest of the file.
- Running `build()` twice, in two fresh interpreters, must give byte-identical
  weights: the harness rebuilds in a second interpreter and compares the
  artefact bytes.

### `CIRCUITS.md` (spec §9.1)

The purpose of every non-zero hidden unit, grouped by circuit, plus which
`reference_policy.py` branch each output logit implements.

### Phase-specific files

- A-selfcheck: `ITERATIONS.md`, the iteration log. For every iteration record a
  summary of the diff, the reason for it, and the inputs you used (each
  hand-constructed input, or a description of the enumeration you ran).
- Phase B: `REVISION.md`, describing your one revision and exactly what was
  disclosed to you.

The harness rejects a phase directory that is missing any required file:
A-strict needs `reference_policy.py`, `build_weights.py` and `CIRCUITS.md`;
A-selfcheck needs those plus `ITERATIONS.md`; Phase B needs those plus
`REVISION.md`.

## What the harness rejects

These are all the checks your files go through. A build that trips any of them
is rejected with a message naming the file, line and rule. Follow them and an
honest build will pass.

**Allowed:** arithmetic reasoning, explicit constants, structured matrix
assignment, deterministic loops that fill known circuits, `torch.no_grad()`,
`torch.zeros`/`torch.tensor`/indexing/`fill_`/`copy_`, elementwise and scalar
`torch` and `numpy` arithmetic, and the deterministic `torch.nn.init` functions
listed below.

**Never inside `build_weights.py`, however it is written:** evaluating the
network, or any part of it, on inputs. That includes a forward pass assembled
from elementwise multiplication and sums, which the tripwires below cannot
detect. Weights must follow from your stated policy by assignment or a
closed-form rule, never from computing outputs and adjusting. In A-selfcheck,
forward passes belong only in your own separate scripts.

**Forbidden in Phases A and B (spec §9.1):** labelled examples (except Phase B's
disclosed set), fitting, distillation, autograd, `.backward()`, `torch.optim`,
numerical or evolutionary search over weights (including random search), and
importing or reading the generator, dataset or evaluation packages or files.

### Static source scan of `build_weights.py` (`scan_source`)

Any occurrence is a rejection, wherever it appears in the file (including the
`__main__` block; comments and docstrings are not scanned):

- Importing (with `import`, `from ... import`, `importlib.import_module` or
  `__import__`) any of: `torch.optim`, `torch.autograd`, `torch.func`,
  `torch._C`, `functorch`, `scipy.optimize`, `sklearn`, or their submodules;
  `from torch import optim / autograd / func / _C`; the harness packages
  `agent_loop_lab.synth`, `agent_loop_lab.synth_ood`, `agent_loop_lab.datasets`,
  `agent_loop_lab.eval`, `agent_loop_lab.baselines`; any relative import that
  names `synth*`, `datasets`, `eval` or `baselines`.
- Random numbers of any kind: importing `random`, `numpy.random`,
  `torch.random` or `torch.distributions` (also `from numpy import random`,
  `from torch import random / distributions`); any use of `np.random`,
  `numpy.random`, `torch.random` or `torch.distributions`; these torch
  functions, called or imported by name: `torch.rand`, `torch.randn`,
  `torch.randint`, `torch.randperm`, `torch.rand_like`, `torch.randn_like`,
  `torch.randint_like`, `torch.normal`, `torch.bernoulli`, `torch.multinomial`,
  `torch.poisson`, `torch.Generator`, `torch.manual_seed`, `torch.seed`,
  `torch.initial_seed`, `torch.get_rng_state`, `torch.set_rng_state`; and the
  in-place RNG methods `.normal_`, `.uniform_`, `.random_`, `.bernoulli_`,
  `.exponential_`, `.cauchy_`, `.log_normal_`, `.geometric_`.
- `torch.nn.init`: only `zeros_`, `ones_`, `constant_`, `eye_`, `dirac_` and
  `calculate_gain` are allowed. Any other `init.<name>` (for example
  `kaiming_uniform_`, `xavier_normal_`) is rejected.
- Optimisers: any attribute or name `optim` (so `torch.optim.anything`), and
  any method call named `.step(` (do not name your own methods `step`).
- Autograd: any call named `backward(` or `.backward(`, `run_backward(`,
  `autograd.grad`, `autograd.backward`, `autograd.functional`,
  `autograd.forward_ad`, `autograd.Function`, any use of `torch.func` or
  `torch._C`, `_execution_engine`, and `_enter_dual_level(`.
- Grad mode: `enable_grad(`, `set_grad_enabled(` or `_set_grad_enabled(` with
  anything but the literal `False`, `inference_mode(False)` and
  `_InferenceMode(False)`.
- `requires_grad`: any keyword argument `requires_grad=` other than the literal
  `False`, `requires_grad_(` with anything but the literal `False`, and any
  assignment to `.requires_grad`.
- `nn.Parameter(...)` construction unless it passes `requires_grad=False`
  (you never need it: assign into the existing parameters instead).
- `getattr`, `setattr` or `delattr` on `torch` with a literal name from the
  lists above.
- Any string literal, or path assembled from literals, that contains a
  `datasets` path segment or a segment starting with `synth`.
- Source that does not parse.

### Harness import and I/O rules

`build_weights.py`:

- may import only `model` (the packet's), `torch`, `numpy` and standard-library
  modules, never with a relative import;
- may not import any module on the banned list below;
- may not call `open`, `exec`, `eval`, `compile`, `__import__`, `input` or
  `breakpoint`;
- may not call any function or method named `load`, `loadtxt`, `genfromtxt`,
  `fromfile`, `memmap`, `from_file`, `save`, `savez`, `savez_compressed`,
  `savetxt`, `tofile`, `read_text`, `read_bytes`, `write_text`, `write_bytes`,
  `save_artifact` or `load_artifact` (no file reads or writes of any kind);
- must define a module-level `def build()`.

`reference_policy.py`:

- may import only standard-library modules, never with a relative import, and
  none on the banned list (`numpy` and `torch` are not allowed here);
- may not call `open`, `exec`, `eval`, `compile`, `__import__`, `input` or
  `breakpoint`, and may not contain a `global` statement;
- must define a module-level `def reference_policy(...)`.

Banned modules (and their submodules) for both files: `os`, `sys`,
`subprocess`, `socket`, `importlib`, `pathlib`, `io`, `builtins`, `ctypes`,
`threading`, `_thread`, `multiprocessing`, `concurrent`, `asyncio`, `random`,
`secrets`, `time`, `datetime`, `uuid`, `pickle`, `marshal`, `shelve`, `shutil`,
`tempfile`, `glob`, `urllib`, `http`, `ssl`, `select`, `selectors`, `signal`,
`mmap`, `fcntl`, `sqlite3`, `csv`, `gzip`, `bz2`, `lzma`, `zipfile`, `tarfile`,
`logging`, `atexit`, `gc`, `inspect`, `runpy`, `pkgutil`, `zipimport`, `site`,
`sysconfig`, `platform`, `getpass`, `webbrowser`, `resource`, `tracemalloc`,
`faulthandler`, `code`, `codeop`. Safe standard-library modules such as `math`,
`fractions`, `itertools`, `functools`, `operator`, `collections`, `typing` and
`dataclasses` are fine.

### Compute ban (`build_weights.py`)

`build()` may not run forward passes, matrix products or linear-algebra
solvers: literal assignment and closed-form rules never need them, and using
them would amount to fitting. Elementwise and scalar arithmetic and index
assignment stay allowed.

Static check, anywhere in the file (including the `__main__` block): the `@`
operator (also `@=`); any `.forward(` or `.dot(` call (so `numpy.dot` and the
`ndarray.dot` method are rejected); and any name, attribute or import
containing `linalg`, `lstsq`, `pinv`, `einsum` or `matmul` (so `torch.linalg`,
`numpy.linalg`, `torch.pinverse` and `__matmul__` are all rejected).

At run time, while the module-level code and `build()` run, each of these is
replaced by a tripwire that rejects the build, even if your code catches the
error: `torch.nn.functional.linear`, `nn.Linear.forward`, calling any
`nn.Module` (for example `m(x)`), `torch.matmul`, `torch.mm`, `torch.bmm`,
`torch.addmm`, `torch.addmv`, `torch.mv`, `torch.einsum`, `torch.tensordot`,
`torch.inner`, `torch.outer`, `torch.dot`, every `torch.linalg` function,
`Tensor.__matmul__`, `Tensor.__rmatmul__`, `Tensor.matmul`, `Tensor.mm`,
`Tensor.mv`, `Tensor.dot`, `Tensor.bmm`, `Tensor.addmm`, `Tensor.addmv`,
`Tensor.inner`, `Tensor.outer`, every
`numpy.linalg` function, `numpy.matmul`, `numpy.dot`, `numpy.einsum`,
`numpy.tensordot`, `numpy.inner`, `numpy.outer` and `numpy.vdot`.

### Runtime checks: anti-training guard (spec §11) and harness checks

Both files are executed, and `build()` is called, inside the harness's
anti-training guard and its I/O tripwire, with `torch.no_grad()` in force throughout. The run is
rejected if, at any point (including inside a `try` block that catches the
error):

- a `torch.optim` optimiser of any kind is constructed;
- `Tensor.backward`, `torch.autograd.backward`, `torch.autograd.grad` or the
  autograd engine's `run_backward` is called, or a new autograd engine is
  created;
- grad mode is switched on by any route (`torch.enable_grad()`,
  `torch.set_grad_enabled(True)`, `torch._C._set_grad_enabled(True)`,
  `torch.inference_mode(False)`, or anything else that leaves grad mode on);
- forward-mode AD is entered (`torch.autograd.forward_ad`, `torch.func.jvp`);
  note that `torch.func` transforms such as `vmap` fail under the guard;
- a thread or a process is started by any route (`threading`, `_thread`,
  `multiprocessing`, `subprocess`, `os.system`, `os.fork`, `os.forkpty`,
  `os.exec*`, `os.posix_spawn`, `os.spawn*`);
- one of the forbidden `agent_loop_lab` packages above is imported;
- a call listed under the compute ban is made while `build_weights.py`'s
  module-level code or `build()` runs;
- a file other than Python code is opened, a file is opened for writing, the
  filesystem is changed (remove, rename, mkdir, rmdir, chmod, link) or a
  socket or URL request is made;
- the returned model, or the harness model loaded from its `state_dict`, has a
  parameter with `requires_grad=True`, a `.grad`, or a `grad_fn`;
- the `state_dict` does not have exactly the template's tensor names in order,
  with the template's shapes, dtype float32 and finite values;
- `reference_policy` fails its smoke call: the harness
  calls `reference_policy` exactly once, on the all-zero 120-tuple, as a type
  check (it must return an `int` in `0..3`); the return value is not recorded.

## Phase A-strict (§9.2): zero execution

Output directory: `<output>/phase-a-strict/`

- You may run only shape, dtype and finiteness checks on the built
  `state_dict` (for example by calling `build()` and inspecting the tensors,
  or printing `sparsity(build())` from the `__main__` block above), plus
  syntax checks of `reference_policy.py` (for example `python -m py_compile`).
- No forward passes of the network on any input, and no calls of
  `reference_policy` on any input. The harness's smoke call is the only one: it
  calls `reference_policy` exactly once, on the all-zero 120-tuple, as a type
  check (it must return an `int` in `0..3`); the return value is not recorded.
- Deliver `reference_policy.py`, `build_weights.py` and `CIRCUITS.md`. The
  harness builds, archives and commits the first valid artefact with its
  sha256 **before anything else runs**.

## Phase A-selfcheck (§9.3): zero labels

Output directory: `<output>/phase-a-selfcheck/`

- Starts from the archived A-strict deliverables, in the same authoring
  context. Copy them into the A-selfcheck directory and revise them there;
  never edit the A-strict directory after it is archived.
- You may run forward passes and `reference_policy` **on inputs you construct
  yourself**: hand-written windows, or enumerations of your own policy's
  branch conditions. Do this from your own separate scripts, not from
  `build_weights.py` (see the compute ban). You may compare the two and revise the weights and the
  policy by direct, reasoned edits.
- Not allowed: generator output, datasets, labels, evaluation output, or
  automated weight search of any kind.
- Log every iteration in `ITERATIONS.md` (diff summary, reason, inputs used).
- Deliver the final `reference_policy.py`, `build_weights.py`, `CIRCUITS.md`
  and `ITERATIONS.md`. The harness archives the final artefact.

Both A-strict and A-selfcheck are reported, and both count as 0 labelled
examples. A-selfcheck is the headline Phase A result.

## Phase B (§9.4): one diagnostic revision

Output directory: `<output>/phase-b-revision/`

This phase starts only when the adjudicator gives you the fixed diagnostic
report: 200 windows from the validation split, each with features, oracle
label, network prediction and reference-policy prediction. The base is the
A-selfcheck deliverables. You may make **one** revision to the weights and to
`reference_policy.py`, using that report and nothing else. Record the revision
and exactly what was disclosed in `REVISION.md`, and deliver it with the
updated `reference_policy.py`, `build_weights.py` and `CIRCUITS.md`. This phase
is reported as 200 labelled examples with one revision.
