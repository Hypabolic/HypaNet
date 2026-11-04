"""Authoring-packet builder and verifier (spec v0.2 §8, §9, §11).

The authoring packet is the only input the weight author sees. It contains:

- ``SPEC_AUTHORING.md``: the spec minus §0 (incl. §0.1), §2.3, §7.2 and §7.3;
- ``model.py``: the immutable model module, made standalone (see below);
- ``feature_index.json``: (position, feature) -> flat input index, names, definitions, class order;
- ``state_dict_template.json``: tensor names, shapes, parameter count, all-zero initial state;
- ``AUTHOR_INSTRUCTIONS.md``: deliverables, phase rules and forbidden actions (spec §9);
- ``MANIFEST.json``: per-file sha256, ``packet_sha256`` and the source spec sha256.

Everything is deterministic: no timestamps, no absolute paths, sorted JSON keys,
LF line endings. Building twice gives byte-identical files. Sources with CR line
endings are refused, because their hashes would not match an LF checkout.

Standalone ``model.py`` (spec §8 overrides the M0 contract's "byte copy"): the
harness module does ``from .schema import CLASS_NAMES, INPUT_DIM``, which cannot
be imported from the packet directory, and importing the harness package instead
would expose the generator and oracle to the weight author. The builder replaces
that one line with a marked block of literal constants taken from
``agent_loop_lab.schema``. The replacement is exactly invertible, so
``verify_packet`` can prove the packet's ``model.py`` is the source module
(``source_model_sha256``) with only those constants inlined. If the source
module ever stops importing from ``schema``, it is copied byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .model import PARAM_COUNT, PARAM_SHAPES
from .schema import (
    CATEGORY_ORDER,
    CLASS_NAMES,
    EVENTS_PER_WINDOW,
    FEATURE_DEFINITIONS,
    FEATURE_NAMES,
    FEATURES_PER_EVENT,
    INPUT_DIM,
    OUTCOME_ORDER,
    SPEC_VERSION,
    flat_index,
)

PACKET_FORMAT_VERSION = 1

EXCLUDED_SECTIONS = ("## 0.", "### 2.3", "### 7.2", "### 7.3")  # by heading prefix
REQUIRED_SECTIONS = ("## 4.", "## 6.", "## 9.")
CAPACITY_HINT_PHRASES = ("per position", "Design note on capacity", "Generator requirements")

SPEC_FILE = "SPEC_AUTHORING.md"
MODEL_FILE = "model.py"
FEATURE_INDEX_FILE = "feature_index.json"
STATE_DICT_TEMPLATE_FILE = "state_dict_template.json"
INSTRUCTIONS_FILE = "AUTHOR_INSTRUCTIONS.md"
MANIFEST_FILE = "MANIFEST.json"

CONTENT_FILES = (
    SPEC_FILE,
    MODEL_FILE,
    FEATURE_INDEX_FILE,
    STATE_DICT_TEMPLATE_FILE,
    INSTRUCTIONS_FILE,
)
PACKET_FILES = CONTENT_FILES + (MANIFEST_FILE,)

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(\S+)")
_FENCE_RE = re.compile(r"^ {0,3}(```|~~~)")
_RULE_RE = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_FORBIDDEN_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+(?:agent_loop_lab\.)?(?:synth\w*|datasets)\b", re.MULTILINE
)
# Imports that only resolve inside the harness package (relative, or of agent_loop_lab itself).
_NON_STANDALONE_IMPORT_RE = re.compile(r"^\s*(?:from\s+\.|(?:from|import)\s+agent_loop_lab\b)", re.MULTILINE)

SCHEMA_IMPORT_LINE = "from .schema import CLASS_NAMES, INPUT_DIM"
_SCHEMA_IMPORT_RE = re.compile(r"^" + re.escape(SCHEMA_IMPORT_LINE) + r"$", re.MULTILINE)


class PacketError(RuntimeError):
    """Raised when the packet cannot be built or fails verification."""


# -- helpers ----------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _pretty_json(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def packet_digest(file_hashes: dict[str, str]) -> str:
    """sha256 of canonical JSON of the sorted ``[name, sha256]`` list."""
    return _sha256(_canonical_json(sorted([name, sha] for name, sha in file_hashes.items())))


def _headings(lines: list[str]) -> dict[int, tuple[int, str]]:
    """Map line index -> (level, key) for ATX headings outside fenced code blocks.

    ``key`` is ``"<hashes> <first token>"``, e.g. ``"### 2.3"`` or ``"## 0."``.
    """
    out: dict[int, tuple[int, str]] = {}
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(line)
        if m:
            out[i] = (len(m.group(1)), f"{m.group(1)} {m.group(2)}")
    return out


def section_keys(text: str) -> list[str]:
    """Heading keys (``"## 4."``, ``"### 2.3"``...) in document order, ignoring code fences."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    h = _headings(lines)
    return [h[i][1] for i in sorted(h)]


def _ends_with_rule(lines: list[str]) -> bool:
    for line in reversed(lines):
        if line.strip() == "":
            continue
        return bool(_RULE_RE.match(line))
    return False


def _normalise_for_phrase_search(text: str) -> str:
    t = text.replace("*", "").replace("`", "").lower()
    t = re.sub(r"[-\s]+", " ", t)
    return t


def find_capacity_hints(text: str) -> list[str]:
    """Capacity-hint phrases present in ``text`` (case, emphasis, hyphen and whitespace insensitive)."""
    norm = _normalise_for_phrase_search(text)
    return [p for p in CAPACITY_HINT_PHRASES if _normalise_for_phrase_search(p) in norm]


def _has_cr(data: bytes) -> bool:
    return b"\r" in data


# -- standalone model.py ---------------------------------------------------------


def inlined_constants_block() -> str:
    """The block that replaces ``SCHEMA_IMPORT_LINE`` in the packet's ``model.py`` (no trailing newline)."""
    return "\n".join(
        [
            f"# --- authoring packet: inlined from agent_loop_lab.schema, replacing `{SCHEMA_IMPORT_LINE}` ---",
            f"INPUT_DIM = {INPUT_DIM!r}",
            f"CLASS_NAMES = {tuple(CLASS_NAMES)!r}",
            "# --- end of inlined constants ---",
        ]
    )


def standalone_model_source(model_text: str) -> str:
    """Make the harness ``model.py`` importable on its own (see module docstring).

    Raises ``PacketError`` if the import line occurs more than once, if the
    source already contains the inlined block, or if any other harness-only
    import remains.
    """
    block = inlined_constants_block()
    if block in model_text:
        raise PacketError(f"{MODEL_FILE}: source already contains the inlined-constants block")
    hits = len(_SCHEMA_IMPORT_RE.findall(model_text))
    if hits > 1:
        raise PacketError(f"{MODEL_FILE}: {SCHEMA_IMPORT_LINE!r} occurs {hits} times")
    out = _SCHEMA_IMPORT_RE.sub(lambda _m: block, model_text)
    if _NON_STANDALONE_IMPORT_RE.search(out):
        raise PacketError(f"{MODEL_FILE}: imports from the harness package that the packet cannot satisfy")
    return out


def original_model_source(packet_model_text: str) -> str:
    """Inverse of ``standalone_model_source``."""
    return packet_model_text.replace(inlined_constants_block(), SCHEMA_IMPORT_LINE)


# -- spec stripping ----------------------------------------------------------


def strip_spec(spec_text: str) -> str:
    """Remove the excluded sections (§0 incl. §0.1, §2.3, §7.2, §7.3).

    Each excluded section runs from its heading up to (not including) the next
    heading of the same or higher level. A trailing ``---`` rule inside the
    removed range is kept, because it separates the surrounding sections,
    unless the kept text already ends with a rule (it then only separated the
    removed section and is dropped). Nothing else is changed apart from
    normalising line endings to LF and ending with exactly one newline.
    """
    text = spec_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    heads = _headings(lines)
    order = sorted(heads)
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        head = heads.get(i)
        if head is not None and head[1] in EXCLUDED_SECTIONS:
            level = head[0]
            end = next((j for j in order if j > i and heads[j][0] <= level), n)
            removed = lines[i:end]
            last = max((k for k, line in enumerate(removed) if line.strip()), default=None)
            if last is not None and _RULE_RE.match(removed[last]) and not _ends_with_rule(out):
                out.extend(removed[last:])
            i = end
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out).rstrip("\n") + "\n"


# -- packet content ------------------------------------------------------------


def feature_index_document() -> dict[str, Any]:
    entries = []
    for position in range(EVENTS_PER_WINDOW):
        for fi, name in enumerate(FEATURE_NAMES):
            entries.append(
                {
                    "flat_index": flat_index(position, name),
                    "position": position,
                    "feature_index": fi,
                    "feature": name,
                }
            )
    return {
        "spec_version": SPEC_VERSION,
        "input_dim": INPUT_DIM,
        "events_per_window": EVENTS_PER_WINDOW,
        "features_per_event": FEATURES_PER_EVENT,
        "layout": (
            "flat_index = position * features_per_event + feature_index; "
            "position 0 is the oldest event of the window, position 7 the newest (the window endpoint)"
        ),
        "features": [
            {"feature_index": fi, "name": name, "definition": FEATURE_DEFINITIONS[name]}
            for fi, name in enumerate(FEATURE_NAMES)
        ],
        "entries": entries,
        "category_order": [c.value for c in CATEGORY_ORDER],
        "outcome_order": [o.value for o in OUTCOME_ORDER],
        "class_order": list(CLASS_NAMES),
    }


def state_dict_template_document() -> dict[str, Any]:
    tensors = []
    for name, shape in PARAM_SHAPES.items():
        numel = 1
        for d in shape:
            numel *= d
        tensors.append({"name": name, "shape": list(shape), "numel": numel})
    return {
        "spec_version": SPEC_VERSION,
        "tensors": tensors,
        "param_count": PARAM_COUNT,
        "dtype": "float32",
        "device": "cpu",
        "initial_state": "all zeros",
        "construction": (
            "start from model.zero_model() (every tensor exactly 0.0), assign entries under torch.no_grad(), "
            "and return the model from build(); the harness saves the artefact (see AUTHOR_INSTRUCTIONS.md)"
        ),
    }


AUTHOR_INSTRUCTIONS = """\
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
"""


# -- build / verify -------------------------------------------------------------


def _relative_files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink())


def _relative_entries(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def _is_leaky_path(rel: str) -> bool:
    return any(part == "datasets" or part.startswith("synth") for part in Path(rel).parts)


def build_packet(spec_path: str | Path, model_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Build the packet into ``out_dir`` and return the manifest (also written to MANIFEST.json).

    ``out_dir`` may be absent, empty, or a previous packet. Anything else in it
    is refused, because a stray file would end up in front of the weight author.
    """
    spec_path, model_path, out = Path(spec_path), Path(model_path), Path(out_dir)
    if out.exists():
        if not out.is_dir():
            raise PacketError(f"{out} exists and is not a directory")
        stray = [e for e in _relative_entries(out) if e not in PACKET_FILES]
        if stray:
            raise PacketError(f"{out} contains entries that are not packet files: {stray}")
    spec_bytes = spec_path.read_bytes()
    model_bytes = model_path.read_bytes()
    for label, data in (("spec", spec_bytes), ("model", model_bytes)):
        if _has_cr(data):
            raise PacketError(
                f"{label} source has CR line endings; check it out with LF (.gitattributes eol=lf) "
                "so its sha256 matches the recorded one"
            )

    contents: dict[str, bytes] = {
        SPEC_FILE: strip_spec(spec_bytes.decode("utf-8")).encode("utf-8"),
        MODEL_FILE: standalone_model_source(model_bytes.decode("utf-8")).encode("utf-8"),
        FEATURE_INDEX_FILE: _pretty_json(feature_index_document()),
        STATE_DICT_TEMPLATE_FILE: _pretty_json(state_dict_template_document()),
        INSTRUCTIONS_FILE: AUTHOR_INSTRUCTIONS.encode("utf-8"),
    }
    assert tuple(contents) == CONTENT_FILES
    hashes = {name: _sha256(data) for name, data in contents.items()}
    manifest: dict[str, Any] = {
        "packet_format_version": PACKET_FORMAT_VERSION,
        "spec_version": SPEC_VERSION,
        "source_spec_sha256": _sha256(spec_bytes),
        "source_model_sha256": _sha256(model_bytes),
        "excluded_sections": list(EXCLUDED_SECTIONS),
        "files": dict(sorted(hashes.items())),
        "packet_sha256": packet_digest(hashes),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        (out / name).write_bytes(data)
    (out / MANIFEST_FILE).write_bytes(_pretty_json(manifest))

    problems = verify_packet(out)
    if problems:
        raise PacketError("built packet failed verification: " + "; ".join(problems))
    return manifest


def _check_spec_text(text: str) -> list[str]:
    problems = []
    keys = section_keys(text)
    for ex in EXCLUDED_SECTIONS:
        if ex in keys:
            problems.append(f"{SPEC_FILE}: excluded section heading present: {ex!r}")
    for req in REQUIRED_SECTIONS:
        if req not in keys:
            problems.append(f"{SPEC_FILE}: required section heading missing: {req!r}")
    return problems


def _check_feature_index(doc: Any) -> list[str]:
    if doc != feature_index_document():
        return [f"{FEATURE_INDEX_FILE}: does not match schema feature order/definitions"]
    return []


def _check_state_dict_template(doc: Any) -> list[str]:
    if doc != state_dict_template_document():
        return [f"{STATE_DICT_TEMPLATE_FILE}: does not match model.PARAM_SHAPES / PARAM_COUNT"]
    return []


def verify_packet(out_dir: str | Path) -> list[str]:
    """Return a list of problems with the packet in ``out_dir``; ``[]`` means OK."""
    out = Path(out_dir)
    if not out.is_dir():
        return [f"packet directory not found: {out}"]
    problems: list[str] = []

    on_disk = _relative_entries(out)
    files_on_disk = set(_relative_files(out))
    for e in on_disk:
        if e not in PACKET_FILES:
            problems.append(f"stray entry: {e}")
        if _is_leaky_path(e):
            problems.append(f"entry from a generator/dataset path: {e}")
    for name in PACKET_FILES:
        if name not in files_on_disk:
            problems.append(f"missing file: {name}")

    manifest: dict[str, Any] | None = None
    if MANIFEST_FILE in files_on_disk:
        try:
            manifest = json.loads((out / MANIFEST_FILE).read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            problems.append(f"{MANIFEST_FILE}: unreadable ({exc})")
    if manifest is not None:
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(files, dict):
            problems.append(f"{MANIFEST_FILE}: missing 'files' mapping")
        else:
            names = sorted(files)
            if names != sorted(CONTENT_FILES):
                problems.append(f"{MANIFEST_FILE}: file list {names} != expected {sorted(CONTENT_FILES)}")
            for name in names:
                if _is_leaky_path(name):
                    problems.append(f"{MANIFEST_FILE}: lists a generator/dataset path: {name}")
            for name in CONTENT_FILES:
                if name in files and name in files_on_disk:
                    actual = _sha256((out / name).read_bytes())
                    if actual != files[name]:
                        problems.append(f"hash mismatch: {name} (manifest {files[name]}, actual {actual})")
            if all(isinstance(v, str) for v in files.values()):
                if manifest.get("packet_sha256") != packet_digest(files):
                    problems.append(f"{MANIFEST_FILE}: packet_sha256 does not match its file list")
            for key in ("source_spec_sha256", "source_model_sha256", "spec_version"):
                if key not in manifest:
                    problems.append(f"{MANIFEST_FILE}: missing {key!r}")

    for name in CONTENT_FILES:
        if name in files_on_disk and _has_cr((out / name).read_bytes()):
            problems.append(f"{name}: contains CR line endings (packet files are LF only)")

    def _read_text(name: str) -> str | None:
        if name not in files_on_disk:
            return None
        try:
            return (out / name).read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"{name}: not UTF-8")
            return None

    spec_text = _read_text(SPEC_FILE)
    if spec_text is not None:
        problems.extend(_check_spec_text(spec_text))
    for name in (SPEC_FILE, INSTRUCTIONS_FILE):
        text = spec_text if name == SPEC_FILE else _read_text(name)
        if text is not None:
            for phrase in find_capacity_hints(text):
                problems.append(f"{name}: contains capacity-hint phrase {phrase!r}")
    instructions = _read_text(INSTRUCTIONS_FILE)
    if instructions is not None and instructions != AUTHOR_INSTRUCTIONS:
        problems.append(f"{INSTRUCTIONS_FILE}: does not match the builder's AUTHOR_INSTRUCTIONS text")
    model_text = _read_text(MODEL_FILE)
    if model_text is not None:
        if _FORBIDDEN_IMPORT_RE.search(model_text):
            problems.append(f"{MODEL_FILE}: imports a generator/dataset module")
        if _NON_STANDALONE_IMPORT_RE.search(model_text):
            problems.append(f"{MODEL_FILE}: not standalone (imports from the harness package)")
        if model_text.count(inlined_constants_block()) > 1:
            problems.append(f"{MODEL_FILE}: inlined-constants block occurs more than once")
        source_sha = manifest.get("source_model_sha256") if isinstance(manifest, dict) else None
        if isinstance(source_sha, str):
            restored = _sha256(original_model_source(model_text).encode("utf-8"))
            if restored != source_sha:
                problems.append(
                    f"{MODEL_FILE}: differs from the source model beyond the inlined constants "
                    f"(source_model_sha256 {source_sha}, restored {restored})"
                )
    for name, check in ((FEATURE_INDEX_FILE, _check_feature_index), (STATE_DICT_TEMPLATE_FILE, _check_state_dict_template)):
        text = _read_text(name)
        if text is None:
            continue
        try:
            problems.extend(check(json.loads(text)))
        except json.JSONDecodeError as exc:
            problems.append(f"{name}: invalid JSON ({exc})")
    return problems
