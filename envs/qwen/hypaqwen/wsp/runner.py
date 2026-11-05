"""v0.4 WSP build runner (docs/V04_WSP_KERNEL.md §2, §6), adapted from the v0.3 runner.

:func:`run_build` turns the weight author's deliverables into a checked, immutable archive:

1. **Inputs.** ``build_edit.py`` (a regular file of at most 200 KB that parses and defines a
   module-level ``def build``) and, optionally, ``edit_constants.safetensors`` (at most 8 MiB, a
   readable safetensors file). Only these two files are staged, byte snapshots taken once, into a
   fresh private directory per build. The prose deliverables are required (:func:`required_prose`):
   ``EDIT.md`` always, plus ``ITERATIONS.md`` for a WSP-A phase or ``REVISION.md`` for a WSP-B
   phase; they are not staged but copied into the archive, the record notes whether each constant
   is named in ``EDIT.md``, and every file of the author directory is hashed into the record.
2. **Build.** A fresh interpreter of this environment, ``python -P -s -B`` (``-I -B`` without
   ``-E``, which would ignore PYTHONHASHSEED; the environment is built from scratch, so the only
   PYTHON* variable is ``PYTHONHASHSEED``), cwd the staged copy, one BLAS / OpenMP thread and no
   CUDA device, sets ``torch.set_num_threads(1)`` and ``torch.use_deterministic_algorithms(True)``,
   checks the pinned checkpoint's sha256 (``hypaqwen.model.FILE_SHA256``), reads the 320 text-model
   tensors with safetensors and upcasts them to fp32 (exact), reads the constants, sets
   ``RLIMIT_CPU`` to :data:`BUILD_CPU_BUDGET_S` beyond that start-up, binds every callable the
   post-build path uses, and only then installs the BUILD_EDIT guard (``from hn_guard.guards_v04
   import install``; ``envs/qwen/hypaqwen/wsp/author`` on ``sys.path``) for the rest of the process.
   Under the guard it executes the staged ``build_edit.py`` and calls ``build(orig, consts)``; both
   mappings are read-only and hand out clones. The returned value must be a ``dict`` of ``str`` to
   plain CPU ``torch.Tensor`` (float32, strided, no ``requires_grad``, ``grad``, ``grad_fn`` or
   Python attributes). After ``build()`` the child refuses a changed thread count or
   deterministic-algorithms setting, an active torch function / dispatch mode and, with the real
   guard, any recorded guard violation (even one the build caught). It sends the tensors back
   through a file opened before the guard (raw fp32 bytes, sorted by name) and reports through a
   second such file, never through stdout / stderr, keeps every author object alive and leaves with
   ``os._exit`` (no author finaliser runs). The parent never executes author code, applies a
   wall-clock timeout of :data:`BUILD_TIMEOUT_S`, keeps only the last
   :data:`CHILD_OUTPUT_TAIL_BYTES` of the child's output (bytes, decoded with replacement), and
   trusts nothing the child reports except the transfer file, which it re-hashes and validates.
3. **Validation (parent, against the parent's own read of the checkpoint).** Every name is one
   of the 320 text-model tensors and not the final norm (``mtp.*`` and ``model.visual.*`` are
   refused by name); shapes match; every value is finite; every returned tensor differs from the
   original; the embedding (``embed_tokens.weight``, tied to the output layer) changes only in
   the four label-word rows (``LABEL_IDS`` of the frozen interface); and no tensor's change is a
   whole-tensor rescaling of the original. That last rule is a tolerance, not exact equality
   (which ``W' = c·W + noise`` would evade), and it is the model server's rule, shared through
   :func:`hypaqwen.wsp.edit_state.rescaling` so the two cannot disagree: with ``new = c·W + R``
   it is refused when ``|c − 1| > 1e-3`` and ``||R|| < 0.5·||(c − 1)·W||`` (``c = 0`` is zeroing a
   tensor); for the zero-centred ``x·(1 + w)`` norms (``input_layernorm``,
   ``post_attention_layernorm``, ``q_norm``, ``k_norm``) also on the effective gain ``1 + w``; for
   the label rows also on the row-centred block and on the changed rows.
4. **Journal (required).** The built edit must equal ``journal_state`` (the model server's final
   journalled edit state) tensor for tensor, byte for byte (a journalled embedding may be the full
   tensor or its four label rows, in ``LABEL_IDS`` order), and its edit-state hash
   (:mod:`hypaqwen.wsp.edit_state`, the server's rule) must equal the journal's final edit hash when
   one is given (:func:`load_journal` replays a journal directory and returns it).
   ``journal_state=None`` is refused unless ``allow_no_journal=True`` (tests only).
5. **Rebuild.** The same build runs in a second fresh interpreter with a different PYTHONHASHSEED
   (:data:`HASH_SEEDS`); its transfer bytes must be identical, so a build that depends on str hash
   order is refused instead of passing by chance.
6. **Archive** (written only after every check; ``BUILD_RECORD.json`` last; an existing record is
   never overwritten unless ``force=True``):

   - ``edit.safetensors``: every edited tensor in full fp32, except the embedding, stored as its
     four label rows under :data:`LABEL_ROWS_KEY` (the full tensor would be 1 GB);
   - ``edit_diff.safetensors``: per tensor the exact changed flat indices (int64, C order) and new
     values (fp32), with each edited tensor's and original's sha256 in the metadata, so the edit
     never depends on recomputation;
   - copies of ``build_edit.py``, the constants and the prose files;
   - ``BUILD_RECORD.json`` (canonical JSON): per tensor the changed entries, rows and columns
     (dimension 0 and the flattened trailing dimensions), ``||dW||_F / ||W||_F``, ``cos(dW, W)``
     (and ``cos(dW, 1 + W)`` for the zero-centred norms), the rank of ``dW`` above the fp32
     storage-noise floor (and by the bare 1e-6 rule), the bf16-rounded changed count and sha256s;
     the overall changed fraction of all 752,393,024 text-model parameters; the edit-state hash
     by the server's rule; the label-row flag with every occurrence of the label ids in the frozen
     interface prefix and suffix; the platform, torch and BLAS, the guard module's identity
     (sha256, STUB flag) as the parent reads them; the base model revision and checkpoint sha256;
     and the sha256 of every input file.

Both files are re-read after writing and checked against the validated tensors.

**Attempts.** Every call, built or refused, writes ``attempt_dir/NNNN.json`` (next free index;
default ``archive_dir/attempts``). That record is the harness copy: it holds the author-facing
``error`` and, separately, ``operator_detail`` (child output tail, harness tracebacks), which is
never shown to the author. The ``BuildError`` text is the author-facing part: an exception raised by
author code is reported as its type (builtin classes only) and author-file line numbers, without
its message; harness paths are replaced by placeholders; an unexpected harness failure is reported
only as such; the text is capped at :data:`MAX_ERROR_CHARS`.

**Guard.** ``install(mode, *, author_dir, phase, kernel_endpoint=None, consts_path=None)`` must
install the guard for the rest of the child and return ``None``. A guard that returns a context
manager (or anything else) is refused: the harness code after ``build()`` runs in the same
interpreter, and author code could otherwise reach it unguarded. A guard module that declares
``STUB = True`` is refused unless ``allow_stub_guard=True`` (tests only).

:func:`load_edit` refuses an archive built without the journal check or under the STUB guard
(``allow_unverified=True`` for tests). :func:`apply_edit` loads an archived edit into a
``hypaqwen`` fp32 model in place (``param.data.copy_``), keeping the tied embedding tied.

CLI: ``python -m hypaqwen.wsp.runner --author-output D --archive A --journal J``.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

from .. import MODEL_ID, MODEL_REVISION, REPO
from .. import model as qmodel
from . import edit_state

__all__ = [
    "BuildError",
    "BUILD_FILE",
    "CONSTS_FILE",
    "PROSE_FILES",
    "EDIT_FILE",
    "DIFF_FILE",
    "RECORD_FILE",
    "ATTEMPTS_DIR",
    "LABEL_ROWS_KEY",
    "EMBED",
    "FINAL_NORM",
    "BUILD_CPU_BUDGET_S",
    "BUILD_TIMEOUT_S",
    "MAX_CONSTS_BYTES",
    "MAX_BUILD_FILE_BYTES",
    "MAX_ERROR_CHARS",
    "run_build",
    "load_edit",
    "edit_from_diff",
    "apply_edit",
    "label_info",
    "load_originals",
    "load_journal",
    "load_journal_state",
    "required_prose",
    "main",
]

RECORD_FORMAT_VERSION = 2
TRANSFER_FORMAT = "hn-wsp-transfer-1"
EDIT_FORMAT = "hn-wsp-edit-1"
DIFF_FORMAT = "hn-wsp-diff-1"

BUILD_FILE = "build_edit.py"
CONSTS_FILE = "edit_constants.safetensors"
PROSE_FILES = ("EDIT.md", "ITERATIONS.md", "REVISION.md")
EDIT_FILE = "edit.safetensors"
DIFF_FILE = "edit_diff.safetensors"
RECORD_FILE = "BUILD_RECORD.json"
ATTEMPTS_DIR = "attempts"
INPUTS_DIR = "inputs"

WEIGHTS_FILE = "model.safetensors-00001-of-00001.safetensors"
TEXT_PREFIX = "model.language_model."
EMBED = TEXT_PREFIX + "embed_tokens.weight"
FINAL_NORM = TEXT_PREFIX + "norm.weight"
LABEL_ROWS_KEY = "label_rows:" + EMBED
# Qwen3_5RMSNorm weights (x * (1 + w)) below the final norm; checked against the model in the tests.
ZERO_CENTRED_NORM_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.(input_layernorm|post_attention_layernorm|self_attn\.q_norm|"
    r"self_attn\.k_norm)\.weight$")
SAFETENSORS_META_KEY = "hypanet"  # one key: safetensors writes a multi-key metadata map in hash order

RANK_RTOL = 1e-6  # same rule as the server's EditReceipt.delta_rank
RANK_NOISE_K = 4.0  # margin on the fp32 storage-noise floor of delta_rank (see _delta_rank)
FP32_EPS = float(torch.finfo(torch.float32).eps)

BUILD_CPU_BUDGET_S = 300
BUILD_TIMEOUT_S = 600
MAX_CONSTS_BYTES = 8 * 1024 * 1024
MAX_BUILD_FILE_BYTES = 200 * 1024
MAX_ERROR_CHARS = 2000
MAX_RETURNED_TENSORS = 400
_CHUNK = 1 << 24
CHILD_OUTPUT_TAIL_BYTES = 8000  # the operator detail keeps the last this many bytes of the child's output
MAX_CHILD_RESULT_BYTES = 1 << 20
CHILD_PYTHON_FLAGS = ("-P", "-s", "-B")  # = -I -B with the environment built from scratch (see _child_env)
HASH_SEEDS = (0, 1)  # PYTHONHASHSEED of the build and of the rebuild check: hash-order dependence is refused

ENV_ROOT = Path(__file__).resolve().parents[2]  # envs/qwen
GUARD_ROOT = Path(__file__).resolve().parent / "author"
GUARD_MODULE = "hn_guard.guards_v04"
GUARD_FILE = GUARD_ROOT / "hn_guard" / "guards_v04.py"
_STAGE_PREFIX = "hn_wsp_stage_"
_ATTEMPT_RE = re.compile(r"^(\d{4})\.json$")
_PHASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class BuildError(RuntimeError):
    """The deliverables were refused, or the archive could not be written.

    ``str(exc)`` is the author-facing text; ``detail`` (operator only) holds raw child output."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail


# -- small helpers ------------------------------------------------------------------------


def _sha256(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_sha256(t: torch.Tensor) -> str:
    """sha256 of a tensor's raw C-order bytes."""
    return _sha256(memoryview(t.detach().contiguous().cpu().numpy()).cast("B"))


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


# -- the pinned checkpoint ---------------------------------------------------------------


def _weights_path(root: Path) -> Path:
    return root / WEIGHTS_FILE


def _layout(weights: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    """Every tensor of the checkpoint: name -> (shape, safetensors dtype)."""
    from safetensors import safe_open

    out = {}
    with safe_open(str(weights), "pt") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            out[k] = (tuple(sl.get_shape()), sl.get_dtype())
    return out


def _text_names(layout: Mapping[str, Any]) -> list[str]:
    return sorted(k for k in layout if k.startswith(TEXT_PREFIX))


def load_originals(weights: Path, names: list[str] | None = None) -> dict[str, torch.Tensor]:
    """Text-model tensors of the checkpoint as fp32 (an exact upcast of BF16; F32 as stored)."""
    from safetensors import safe_open

    out: dict[str, torch.Tensor] = {}
    with safe_open(str(weights), "pt") as f:
        for k in (names if names is not None else [k for k in f.keys() if k.startswith(TEXT_PREFIX)]):
            out[k] = f.get_tensor(k).to(torch.float32)
    return out


_LABEL_CACHE: dict[str, dict[str, Any]] = {}


def label_info(root: Path | None = None) -> dict[str, Any]:
    """Label ids of the frozen interface and every occurrence of them in its prefix and suffix."""
    root = root or qmodel.snapshot_dir()
    key = str(root)
    if key not in _LABEL_CACHE:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer

        from ..interface import FROZEN, build_interface, spec_sha256

        iface = build_interface(AutoTokenizer.from_pretrained(root), FROZEN)
        by_label = []
        for word, lid in zip(FROZEN.label_words, iface.label_ids, strict=True):
            by_label.append({
                "word": word,
                "id": lid,
                "prefix_positions": [i for i, t in enumerate(iface.prefix_ids) if t == lid],
                "suffix_positions": [i for i, t in enumerate(iface.suffix_ids) if t == lid],
            })
        _LABEL_CACHE[key] = {
            "interface": FROZEN.name,
            "interface_sha256": spec_sha256(FROZEN),
            "label_ids": list(iface.label_ids),
            "label_words": list(FROZEN.label_words),
            "prefix_tokens": len(iface.prefix_ids),
            "suffix_tokens": len(iface.suffix_ids),
            "occurrences": by_label,
        }
    return _LABEL_CACHE[key]


# -- error text (v0.3 hygiene) -----------------------------------------------------------


_AUTHOR_FILES: set[str] = set()  # realpaths of the executed author copies (for traceback line numbers)


def _safe_type_name(obj: Any) -> str:
    """The class name if it is a builtin or a torch class, otherwise a fixed phrase."""
    import builtins

    cls = obj if isinstance(obj, type) else type(obj)
    if getattr(builtins, cls.__name__, None) is cls:
        return cls.__name__
    mod = getattr(cls, "__module__", "") or ""
    if mod.split(".")[0] == "torch" and cls.__name__.isidentifier() and len(cls.__name__) <= 40:
        if any(getattr(sub, cls.__name__, None) is cls for sub in (torch, torch.nn)):
            return f"{mod}.{cls.__name__}"
    return "an object of a non-builtin class"


def _author_lines(exc: BaseException) -> str:
    lines = []
    for fr in traceback.extract_tb(exc.__traceback__):
        try:
            fn = os.path.realpath(fr.filename)
        except (OSError, ValueError):
            continue
        if fn in _AUTHOR_FILES and fr.lineno is not None:
            lines.append(f"{os.path.basename(fn)}:{fr.lineno}")
    if not lines:
        return "outside the author file"
    return "at " + " < ".join(reversed(lines[-5:]))


def _author_exception(exc: BaseException) -> str:
    return f"{_safe_type_name(exc)} raised {_author_lines(exc)} (the message is not disclosed)"


def _redact(message: str, places: Mapping[str, Path | None]) -> str:
    pairs: list[tuple[str, str]] = []
    for placeholder, path in places.items():
        if path is None:
            continue
        for form in {str(path), os.path.realpath(path)}:
            if Path(form).is_absolute() and len(form) > 1:
                pairs.append((form, placeholder))
    for form, placeholder in sorted(pairs, key=lambda fp: -len(fp[0])):
        message = message.replace(form, placeholder)
    return re.sub(r"[^\s'\"]*" + re.escape(_STAGE_PREFIX) + r"[^/\s'\"]*", "<stage>", message)


def _cap(message: str) -> str:
    if len(message) <= MAX_ERROR_CHARS:
        return message
    return message[: MAX_ERROR_CHARS - 40] + f" ... [cut: {len(message)} characters]"


# -- the child (fresh interpreter) ---------------------------------------------------------


class _ReadOnlyTensors(Mapping[str, torch.Tensor]):
    """A read-only mapping that hands out clones, so a build cannot alter the harness copy by accident.

    (Deliberate tampering gains nothing: the parent checks the result against its own read of the
    checkpoint.)"""

    __slots__ = ("_t",)

    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        self._t = dict(tensors)

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._t[name].clone()

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._t))

    def __len__(self) -> int:
        return len(self._t)

    def __contains__(self, name: object) -> bool:
        return name in self._t


def _exec_author_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    code = compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    sys.modules[name] = module
    exec(code, module.__dict__)  # noqa: S102 - runs under the BUILD_EDIT guard
    return module


def _step(label: str, fn: Any, violation: type[BaseException]) -> Any:
    try:
        return fn()
    except BuildError:
        raise
    except violation as exc:
        raise BuildError(f"{label}: rejected by the BUILD_EDIT guard: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - author code may raise anything
        raise BuildError(f"{label}: {_author_exception(exc)}") from exc


def _check_returned(result: Any, known: Mapping[str, Any]) -> list[tuple[str, torch.Tensor]]:
    """Type and autograd checks on what ``build`` returned (scope, values: the parent)."""
    if type(result) is not dict:
        raise BuildError(f"{BUILD_FILE}: build() returned {_safe_type_name(result)}, not a dict")
    if len(result) > MAX_RETURNED_TENSORS:
        raise BuildError(f"{BUILD_FILE}: build() returned {len(result)} entries (at most {MAX_RETURNED_TENSORS})")
    items = []
    total = 0
    for k, v in dict.items(result):
        if type(k) is not str or len(k) > 256:
            raise BuildError(f"{BUILD_FILE}: build() returned a key that is not a str of at most 256 characters")
        what = k if k in known else "a returned key that is not a text-model tensor name (not quoted)"
        if type(v) is not torch.Tensor:
            raise BuildError(f"{BUILD_FILE}: build()[{what}] is {_safe_type_name(v)}, not a plain torch.Tensor")
        if object.__getattribute__(v, "__dict__"):
            raise BuildError(f"{BUILD_FILE}: build()[{what}] carries Python attributes; return plain tensors")
        if v.requires_grad or v.grad_fn is not None or (v.is_leaf and v.grad is not None):
            raise BuildError(f"{BUILD_FILE}: build()[{what}] carries autograd state (requires_grad, grad or grad_fn)")
        if v.layout is not torch.strided or v.device.type != "cpu" or v.is_quantized or v.is_nested:
            raise BuildError(f"{BUILD_FILE}: build()[{what}] must be a dense (strided) CPU tensor")
        if v.dtype is not torch.float32:
            raise BuildError(f"{BUILD_FILE}: build()[{what}] has dtype {v.dtype}, not torch.float32")
        total += v.numel()
        items.append((k, v))
    if total > sum(math.prod(s) for s, _ in known.values()):
        raise BuildError(f"{BUILD_FILE}: build() returned more values than the text model has")
    return sorted(items, key=lambda kv: kv[0])


def _write_all(fd: int, data: memoryview, write: Any = os.write) -> None:
    while len(data):
        n = write(fd, data)
        data = data[n:]


def _transfer_bytes(items: list[tuple[str, torch.Tensor]], prims: tuple[Any, ...]) -> tuple[list[memoryview], str]:
    """The transfer file's chunks (an 8-byte header length, a JSON header, raw fp32 bytes of every tensor,
    sorted by name) and their sha256. ``prims`` are the callables bound before any author code ran."""
    sha256_new, dumps, numpy_of = prims
    arrays, entries, offset = [], [], 0
    for name, t in items:
        a = numpy_of(t)
        arrays.append(memoryview(a).cast("B"))
        entries.append({"name": name, "shape": list(t.shape), "offset": offset, "nbytes": a.nbytes})
        offset += a.nbytes
    header = dumps({"format": TRANSFER_FORMAT, "tensors": entries}, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False).encode("utf-8")
    h = sha256_new()
    chunks = [memoryview(len(header).to_bytes(8, "little")), memoryview(header), *arrays]
    for mv in chunks:
        h.update(mv)
    return chunks, h.hexdigest()


def _platform_info() -> dict[str, Any]:
    """Facts of this interpreter and its packages (the parent records its own: the child runs the same
    executable with the same packages, and nothing the child reports is trusted)."""
    import platform

    import numpy
    import safetensors

    blas = [ln.strip() for ln in torch.__config__.show().splitlines() if "BLAS" in ln or "LAPACK" in ln]
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "safetensors": safetensors.__version__,
        "blas": blas,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "device": "cpu",
    }


def _guard_handle_kind(handle: Any) -> str:
    if handle is None:
        return "None (installed for the rest of the process)"
    if hasattr(handle, "__enter__") and hasattr(handle, "__exit__"):
        return "a context manager"
    return _safe_type_name(handle)


def _fresh_numpy(t: torch.Tensor) -> Any:
    """A C-order numpy copy of ``t``'s values (never a view of the author's storage)."""
    return t.detach().resolve_neg().resolve_conj().contiguous().numpy().copy()


def _child_build(author_dir: Path, weights: Path, weights_sha: str, consts_path: Path | None, phase: str,
                 cpu_budget: int, transfer_fd: int, keep: list[Any]) -> dict[str, Any]:
    """Runs in the fresh interpreter. The guard is installed for the rest of the process before any author
    code runs (a guard that returns a context manager is refused), so the harness code after build()
    also runs under it. Everything that code calls is bound here first, every author object is kept
    alive in ``keep`` until ``os._exit`` (no author finaliser runs), and the parent trusts nothing this
    function reports except the transfer file, whose bytes it re-hashes and validates itself."""
    import resource

    from safetensors.torch import load_file

    # Bound before any author code: module globals and library attributes can be replaced by it.
    prims = (hashlib.sha256, json.dumps, _fresh_numpy)
    transfer_bytes, write_all, check_returned = _transfer_bytes, _write_all, _check_returned
    step, exec_author = _step, _exec_author_module
    os_write, os_close = os.write, os.close
    get_threads, det_enabled = torch.get_num_threads, torch.are_deterministic_algorithms_enabled
    tf_stack = getattr(torch._C, "_len_torch_function_stack", None)
    td_stack = getattr(torch._C, "_len_torch_dispatch_stack", None)
    BE = BuildError

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.dont_write_bytecode = True
    if qmodel._sha256(weights) != weights_sha:
        return {"error": "the pinned checkpoint failed its sha256 check in the build interpreter"}
    base = load_originals(weights)  # before the guard: the build itself reads no file
    consts = load_file(str(consts_path)) if consts_path is not None else {}
    used = resource.getrusage(resource.RUSAGE_SELF)
    soft = int(math.ceil(used.ru_utime + used.ru_stime)) + cpu_budget
    resource.setrlimit(resource.RLIMIT_CPU, (soft, soft + 5))

    sys.path.insert(0, str(GUARD_ROOT))
    guard = importlib.import_module(GUARD_MODULE)
    if Path(guard.__file__).resolve() != GUARD_FILE.resolve():
        return {"error": "could not import the harness guard module"}
    violation = getattr(guard, "TrainingViolation", None) or getattr(guard, "GuardViolation", None)
    if not (isinstance(violation, type) and issubclass(violation, BaseException)):
        violation = type("_NoViolation", (Exception,), {})
    mode = getattr(guard, "BUILD_EDIT", "build_edit")
    installed_mode = getattr(guard, "installed_mode", None)
    violation_count = getattr(guard, "violation_count", None)
    author_file = (author_dir / "output" / phase / BUILD_FILE).resolve()
    _AUTHOR_FILES.add(str(author_file))
    orig, cmap = _ReadOnlyTensors(base), _ReadOnlyTensors(consts)
    known = {k: (tuple(v.shape), "F32") for k, v in base.items()}
    keep.extend((orig, cmap, base, consts))
    stub_calls = None

    gc.disable()  # no author finaliser runs from a collection
    try:
        with contextlib.redirect_stdout(sys.stderr):
            try:
                handle = guard.install(mode, author_dir=str(author_dir), phase=phase, kernel_endpoint=None,
                                       consts_path=None if consts_path is None else str(consts_path))
                if handle is not None:
                    # A context-manager guard would leave the harness code after build() unguarded.
                    keep.append(handle)
                    return {"error": "the BUILD_EDIT guard's install() returned "
                                     f"{_guard_handle_kind(handle)}; the build runner needs a guard installed "
                                     "for the rest of the process (install() returning None)"}
                if callable(installed_mode) and installed_mode() != mode:
                    return {"error": "the BUILD_EDIT guard does not report itself installed after install()"}
                if getattr(guard, "STUB", False) is True:
                    stub_calls = [dict(c) for c in getattr(guard, "CALLS", [])]
                module = step(f"{BUILD_FILE} (module-level code)",
                              lambda: exec_author("build_edit", author_file), violation)
                keep.append(module)
                build = step(f"{BUILD_FILE}: build lookup", lambda: getattr(module, "build", None), violation)
                keep.append(build)
                if not callable(build):
                    raise BE(f"{BUILD_FILE}: does not define a callable build()")
                result = step(f"{BUILD_FILE}: build()", lambda: build(orig, cmap), violation)
                keep.append(result)
            except violation as exc:
                keep.append(exc)
                raise BE(f"rejected by the BUILD_EDIT guard: {exc}") from exc
        # From here on no author code is called on purpose; the guard stays installed regardless.
        if (tf_stack is not None and tf_stack() != 0) or (td_stack is not None and td_stack() != 0):
            raise BE(f"{BUILD_FILE}: build() left a torch function or dispatch mode active")
        if get_threads() != 1 or det_enabled() is not True:
            raise BE(f"{BUILD_FILE}: build() changed the build interpreter's thread count or its "
                     "deterministic-algorithms setting")
        if callable(violation_count) and violation_count() != 0:
            raise BE(f"{BUILD_FILE}: the BUILD_EDIT guard refused {violation_count()} action(s) during the build "
                     "(a refusal that the build caught still fails it)")
        items = check_returned(result, known)
        keep.append(items)
        chunks, sha = transfer_bytes(items, prims)
    except BE as exc:
        keep.append(exc)
        return {"error": str(exc)}
    for mv in chunks:
        write_all(transfer_fd, mv, os_write)
    os_close(transfer_fd)
    body: dict[str, Any] = {"ok": True, "transfer_sha256": sha}
    if stub_calls is not None:
        body["stub_calls"] = [
            {"mode": c.get("mode"), "phase": c.get("phase"),
             "author_dir_is_stage": c.get("author_dir") == str(author_dir),
             "consts_path_given": c.get("consts_path") is not None,
             "kernel_endpoint": c.get("kernel_endpoint")}
            for c in stub_calls
        ]
    return body


def _child_main(argv: list[str]) -> int:
    """``build AUTHOR_DIR WEIGHTS WEIGHTS_SHA CONSTS|- PHASE CPU_BUDGET TRANSFER RESULT``."""
    dumps, os_write, os_close, os_exit, write_all = json.dumps, os.write, os.close, os._exit, _write_all
    flush_out, flush_err = sys.stdout.flush, sys.stderr.flush
    keep: list[Any] = []  # every author object, alive until os._exit: no author finaliser ever runs
    result_path = Path(argv[-1])
    try:
        rfd = os.open(result_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError:
        os_exit(3)
    try:
        if argv[0] != "build" or len(argv) != 9:
            body: dict[str, Any] = {"error": "bad child arguments"}
        else:
            tfd = os.open(argv[7], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            consts = None if argv[4] == "-" else Path(argv[4])
            body = _child_build(Path(argv[1]), Path(argv[2]), argv[3], consts, argv[5], int(argv[6]), tfd, keep)
    except BaseException as exc:  # noqa: BLE001 - reported to the operator only
        keep.append(exc)
        try:
            traceback.print_exc(file=sys.stderr)
        except BaseException:  # noqa: BLE001
            pass
        body = {"error": "the fresh interpreter failed outside the guarded steps"}
    ok = body.get("ok") is True
    try:
        write_all(rfd, memoryview(dumps(body).encode("utf-8")), os_write)
        os_close(rfd)
    finally:
        for flush in (flush_out, flush_err):
            try:
                flush()
            except BaseException:  # noqa: BLE001
                pass
        os_exit(0 if ok else 1)
    return 1  # pragma: no cover


# -- parent: staging and running a child ------------------------------------------------------


def _child_env(hash_seed: int) -> dict[str, str]:
    """A minimal environment built from scratch. The only PYTHON* variable is PYTHONHASHSEED (the child
    runs with ``-P -s`` rather than ``-I``, because ``-I`` implies ``-E``, which ignores PYTHONHASHSEED;
    with no other PYTHON* variable present the isolation is the same)."""
    env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG") if k in os.environ}
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
                "PYTHONHASHSEED": str(int(hash_seed))})
    return env


def _run_capped(cmd: list[str], *, cwd: Path, env: Mapping[str, str], timeout: float,
                cap: int = CHILD_OUTPUT_TAIL_BYTES) -> tuple[int, str]:
    """Run ``cmd`` with stdout and stderr on one pipe, keeping only the last ``cap`` bytes (a reader
    thread drains the pipe, so memory stays bounded however much the child prints). Returns the exit
    status and the kept tail, decoded with ``errors="replace"``. Raises ``subprocess.TimeoutExpired``."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=dict(env), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    tail = bytearray()
    total = [0]

    def drain() -> None:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read1(1 << 16)
            if not chunk:
                return
            total[0] += len(chunk)
            tail.extend(chunk)
            if len(tail) > 2 * cap:
                del tail[: len(tail) - cap]

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    finally:
        reader.join(timeout=10)
        if proc.stdout is not None:
            proc.stdout.close()
    text = bytes(tail[-cap:]).decode("utf-8", errors="replace")
    if total[0] > cap:
        text = f"[child output: {total[0]} bytes, last {cap} kept]\n" + text
    return rc, text


@contextlib.contextmanager
def _staged(files: Mapping[str, bytes], phase: str) -> Iterator[Path]:
    """``files`` staged at ``author/output/<phase>/``, the only place BUILD_EDIT lets a build read them."""
    with tempfile.TemporaryDirectory(prefix=_STAGE_PREFIX) as tmp:
        root = Path(tmp).resolve()
        (root / "author" / "output" / phase).mkdir(parents=True)
        for name, data in files.items():
            (root / "author" / "output" / phase / name).write_bytes(data)
        (root / "out").mkdir()
        yield root


def _build_in_child(stage: Path, weights: Path, phase: str, cpu_budget: int, timeout: float,
                    label: str, hash_seed: int) -> tuple[dict[str, Any], Path]:
    """One build in a fresh interpreter. Only ``transfer_sha256`` of its report is used, and only after the
    transfer file is re-hashed here: author code ran in that process and could have written the report."""
    author = stage / "author"
    consts = author / "output" / phase / CONSTS_FILE
    transfer, result_path = stage / "out" / "edit.transfer", stage / "out" / "result.json"
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "from hypaqwen.wsp.runner import _child_main; sys.exit(_child_main(sys.argv[2:]))")
    cmd = [sys.executable, *CHILD_PYTHON_FLAGS, "-c", code, str(ENV_ROOT), "build", str(author), str(weights),
           qmodel.FILE_SHA256[WEIGHTS_FILE], str(consts) if consts.is_file() else "-", phase, str(cpu_budget),
           str(transfer), str(result_path)]
    try:
        rc, output = _run_capped(cmd, cwd=author, env=_child_env(hash_seed), timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise BuildError(f"{label}: timed out after {timeout:.0f} s (wall clock)") from exc
    detail = output.strip()[-CHILD_OUTPUT_TAIL_BYTES:] or None
    result: Any = None
    if result_path.is_file() and result_path.stat().st_size <= MAX_CHILD_RESULT_BYTES:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            result = None
    if isinstance(result, dict) and isinstance(result.get("error"), str):
        raise BuildError(f"{label}: {_cap(result['error'])}", detail)
    if rc in (-signal.SIGXCPU, -signal.SIGKILL, 128 + signal.SIGXCPU):
        raise BuildError(f"{label}: CPU budget exceeded (limit {cpu_budget} s)", detail)
    if rc != 0 or not isinstance(result, dict) or result.get("ok") is not True:
        # The child's own output is not quoted: author code can print to it.
        raise BuildError(f"{label}: fresh interpreter failed (exit {rc}) without a harness verdict", detail)
    sha = result.get("transfer_sha256")
    if not isinstance(sha, str) or not transfer.is_file() or _file_sha256(transfer) != sha:
        raise BuildError(f"{label}: the edit transfer file is missing or differs from the child's report", detail)
    return {"transfer_sha256": sha, "stub_calls": result.get("stub_calls")}, transfer


def _file_sha256(path: Path) -> str:
    return qmodel._sha256(path)


def _read_transfer(path: Path) -> dict[str, torch.Tensor]:
    """Parse the transfer file. It is untrusted (author code ran in the process that wrote it): any
    malformation is a refusal, and its values are validated afterwards like any returned tensor."""
    size = path.stat().st_size
    buf = bytearray(size)
    with open(path, "rb") as f:
        if f.readinto(buf) != size:
            raise BuildError("the edit transfer file could not be read")
    try:
        n = int.from_bytes(buf[:8], "little")
        if size < 8 or 8 + n > size:
            raise ValueError("header")
        header = json.loads(bytes(buf[8: 8 + n]).decode("utf-8"))
        if not isinstance(header, dict) or header.get("format") != TRANSFER_FORMAT or not isinstance(
                header.get("tensors"), list) or len(header["tensors"]) > MAX_RETURNED_TENSORS:
            raise ValueError("format")
        out: dict[str, torch.Tensor] = {}
        base = 8 + n
        expect = 0
        for e in header["tensors"]:
            name, shape = e["name"], e["shape"]
            if (type(name) is not str or len(name) > 256 or name in out or type(shape) is not list or len(shape) > 8
                    or not all(type(d) is int and 0 <= d < 1 << 40 for d in shape)):
                raise ValueError("entry")
            numel = math.prod(shape)
            if e["offset"] != expect or e["nbytes"] != 4 * numel or base + expect + e["nbytes"] > size:
                raise ValueError("extent")
            out[name] = (torch.frombuffer(buf, dtype=torch.float32, count=numel, offset=base + expect).view(shape)
                         if numel else torch.empty(shape, dtype=torch.float32))
            expect += e["nbytes"]
        if base + expect != size:
            raise ValueError("size")
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise BuildError("the edit transfer file is malformed") from exc
    return out


# -- parent: validation and statistics ------------------------------------------------------------


def _changed_indices(new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    """Flat C-order indices where the fp32 bit patterns differ."""
    a, b = new.reshape(-1).view(torch.int32), old.reshape(-1).view(torch.int32)
    parts = [torch.nonzero(a[i: i + _CHUNK] != b[i: i + _CHUNK]).flatten() + i for i in range(0, a.numel(), _CHUNK)]
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)


def _sq_norm64(t: torch.Tensor) -> float:
    flat = t.reshape(-1)
    return float(sum(flat[i: i + _CHUNK].double().square().sum().item() for i in range(0, flat.numel(), _CHUNK)))


def _delta_rank(delta: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor | None, row_idx: torch.Tensor,
                col_idx: torch.Tensor | None, new_vals: torch.Tensor) -> dict[str, Any]:
    """Rank of ``dW`` on its changed rows x columns, two ways.

    ``rank``: singular values above ``max(RANK_RTOL * s_max, floor)``, where ``floor`` is the fp32 storage
    noise: each stored entry is rounded by at most half an ulp (<= eps32/2 * |w|), and the spectral norm
    of such an m x n rounding matrix stays below ``eps32 * rms|w| * (sqrt(m) + sqrt(n))`` (the
    random-matrix scale with the worst per-entry error); :data:`RANK_NOISE_K` adds a margin. Without
    the floor a small exact rank-1 edit of a real layer reports near-full rank (rounding noise of about
    3e-8 exceeds 1e-6 * s_max once s_max < 0.03). ``rank_rtol_only``: the bare RANK_RTOL rule (the
    server's ``EditReceipt.delta_rank``)."""
    out = {"rank": 0, "rank_rtol_only": 0, "threshold": 0.0}
    if delta.numel() == 0:
        return out
    if cols is None:
        return {"rank": 1, "rank_rtol_only": 1, "threshold": 0.0}
    sub = torch.zeros(rows.numel(), cols.numel(), dtype=torch.float64)
    sub[torch.searchsorted(rows, row_idx), torch.searchsorted(cols, col_idx)] = delta
    s = torch.linalg.svdvals(sub)
    if not s.numel() or not s[0] > 0:
        return out
    m, n = sub.shape
    rms = math.sqrt(float(new_vals.double().square().mean().item()))
    floor = RANK_NOISE_K * FP32_EPS * rms * (math.sqrt(m) + math.sqrt(n))
    threshold = max(RANK_RTOL * float(s[0]), floor)
    return {"rank": int((s > threshold).sum().item()), "rank_rtol_only": int((s > RANK_RTOL * s[0]).sum().item()),
            "threshold": threshold}


def _is_zero_centred_norm(name: str) -> bool:
    """A Qwen3_5RMSNorm weight: the layer computes ``x * (1 + w)`` (the gated linear-attention norm,
    Qwen3_5RMSNormGated, computes ``x * w`` and is not one)."""
    return name == FINAL_NORM or ZERO_CENTRED_NORM_RE.match(name) is not None


def _tensor_stats(name: str, new: torch.Tensor, old: torch.Tensor) -> dict[str, Any]:
    """Exact change statistics of one tensor, from its changed entries (``idx``/``values`` are kept for the diff)."""
    idx = _changed_indices(new, old)
    flat_new, flat_old = new.reshape(-1), old.reshape(-1)
    vals, ovals = flat_new[idx], flat_old[idx]
    delta = vals.double() - ovals.double()
    sq_old, sq_delta = _sq_norm64(old), float(delta.square().sum().item())
    dot = float((delta * ovals.double()).sum().item())
    if new.dim() >= 2:
        inner = new.numel() // new.shape[0]
        row_idx, col_idx = idx // inner, idx % inner
        rows, cols = torch.unique(row_idx), torch.unique(col_idx)
    else:
        row_idx, col_idx, rows, cols = idx, None, idx, None
    bf16 = int((vals.to(torch.bfloat16).view(torch.int16) != ovals.to(torch.bfloat16).view(torch.int16)).sum().item())
    norm_old, norm_delta = math.sqrt(sq_old), math.sqrt(sq_delta)
    rank = _delta_rank(delta, rows, cols, row_idx, col_idx, vals)
    out = {
        "shape": list(new.shape),
        "numel": new.numel(),
        "changed_entries": int(idx.numel()),
        "changed_rows": int(rows.numel()),
        "changed_cols": None if cols is None else int(cols.numel()),
        "rel_frobenius": norm_delta / norm_old if norm_old > 0 else None,
        "delta_cos_original": dot / (norm_delta * norm_old) if norm_delta > 0 and norm_old > 0 else None,
        "delta_rank": rank["rank"],
        "delta_rank_rtol_only": rank["rank_rtol_only"],
        "delta_rank_threshold": rank["threshold"],
        "bf16_changed_entries": bf16,
        "sha256": _tensor_sha256(new),
        "original_sha256": _tensor_sha256(old),
        "_idx": idx,
        "_values": vals.clone(),
        "_rows": rows,
    }
    if _is_zero_centred_norm(name):
        # The effective gain is 1 + w: a scalar multiple of it, w' = c (1 + w) - 1, has dW = (c - 1)(1 + w).
        gain = old.double().reshape(-1) + 1.0
        full = torch.zeros_like(gain)
        full[idx] = delta
        ng = float(gain.norm().item())
        out["zero_centred_norm"] = True
        out["delta_cos_effective_gain"] = (float((full * gain).sum().item()) / (norm_delta * ng)
                                           if norm_delta > 0 and ng > 0 else None)
    return out


def _validate(built: Mapping[str, torch.Tensor], layout: Mapping[str, tuple[tuple[int, ...], str]],
              originals: Mapping[str, torch.Tensor], label_ids: list[int]) -> dict[str, dict[str, Any]]:
    """§2 scope and value checks (module docstring, step 3); returns per-tensor statistics."""
    text = set(_text_names(layout))
    unknown = [k for k in built if k not in layout]
    if unknown:
        raise BuildError(f"{BUILD_FILE}: build() returned {len(unknown)} name(s) that are not tensors of the "
                         "checkpoint (not quoted); only model.language_model.* tensors may be edited")
    outside = sorted(k for k in built if k not in text)
    if outside:
        raise BuildError(f"{BUILD_FILE}: outside the edit scope (mtp.* and model.visual.* are refused): {outside}")
    if FINAL_NORM in built:
        raise BuildError(f"{BUILD_FILE}: {FINAL_NORM} (the final (1 + w) norm, a global temperature) is refused")
    for k, t in built.items():
        if tuple(t.shape) != layout[k][0]:
            raise BuildError(f"{BUILD_FILE}: {k} has shape {list(t.shape)}, the checkpoint's is {list(layout[k][0])}")
    for k in sorted(built):
        if not bool(torch.isfinite(built[k]).all()):
            raise BuildError(f"{BUILD_FILE}: {k} has non-finite values")
    stats: dict[str, dict[str, Any]] = {}
    for k in sorted(built):
        s = _tensor_stats(k, built[k], originals[k])
        if s["changed_entries"] == 0:
            raise BuildError(f"{BUILD_FILE}: {k} is returned unchanged; build() returns only the replaced tensors")
        if k == EMBED:
            other = sorted(set(s["_rows"].tolist()) - set(label_ids))
            if other:
                raise BuildError(f"{BUILD_FILE}: {EMBED}: only the {len(label_ids)} label-word rows {label_ids} may "
                                 f"change; {len(other)} other row(s) changed")
        if k == EMBED:
            why = edit_state.rescaling(k, built[k][label_ids], originals[k][label_ids])
        else:
            why = edit_state.rescaling(k, built[k], originals[k])
        if why is not None:
            raise BuildError(f"{BUILD_FILE}: {k}: {why}; refused ({edit_state.RESCALING_RULE})")
        stats[k] = s
    return stats


def _normalise_journal(journal: Mapping[str, torch.Tensor], label_ids: list[int],
                       originals: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in journal.items():
        if not isinstance(v, torch.Tensor):
            raise BuildError("the journalled edit state holds a value that is not a tensor")
        v = v.detach().to("cpu")
        if k == EMBED and k in originals and v.dtype == torch.float32 and tuple(v.shape) == (
                len(label_ids), originals[k].shape[1]):
            full = originals[k].clone()  # a journalled label-row edit: the four rows in LABEL_IDS order
            full[label_ids] = v
            v = full
        out[k] = v
    return out


def _check_journal(built: Mapping[str, torch.Tensor], journal: Mapping[str, torch.Tensor]) -> None:
    missing, extra = sorted(set(journal) - set(built)), sorted(set(built) - set(journal))
    differ = []
    for k in sorted(set(built) & set(journal)):
        j, b = journal[k], built[k]
        if j.dtype != torch.float32 or tuple(j.shape) != tuple(b.shape) or not torch.equal(
                j.contiguous().view(torch.int32), b.contiguous().view(torch.int32)):
            differ.append(k)
    if missing or extra or differ:
        raise BuildError(
            "the build differs from the model server's final journalled edit state: "
            f"journalled but not built {missing}; built but not journalled {extra}; different bytes {differ}"
        )


# -- parent: inputs, archive, attempts ---------------------------------------------------------


def _read_inputs(author_dir: Path) -> dict[str, bytes]:
    """Byte snapshots of ``build_edit.py``, the constants and the prose files present, with the static checks."""
    if not author_dir.is_dir():
        raise BuildError("author output directory not found")
    src = author_dir / BUILD_FILE
    if src.is_symlink() or not src.is_file():
        raise BuildError(f"{BUILD_FILE} is missing (or is not a regular file)")
    snap = {BUILD_FILE: src.read_bytes()}
    if len(snap[BUILD_FILE]) > MAX_BUILD_FILE_BYTES:
        raise BuildError(f"{BUILD_FILE} is larger than {MAX_BUILD_FILE_BYTES} bytes")
    try:
        tree = ast.parse(snap[BUILD_FILE].decode("utf-8"), filename=BUILD_FILE)
    except (UnicodeDecodeError, SyntaxError, ValueError) as exc:
        line = getattr(exc, "lineno", None)
        raise BuildError(f"{BUILD_FILE} does not parse as UTF-8 Python" + (f" (line {line})" if line else "")) from exc
    if not any(isinstance(n, ast.FunctionDef) and n.name == "build" for n in tree.body):
        raise BuildError(f"{BUILD_FILE} does not define a module-level def build(orig, consts)")
    consts = author_dir / CONSTS_FILE
    if consts.exists() or consts.is_symlink():
        if consts.is_symlink() or not consts.is_file():
            raise BuildError(f"{CONSTS_FILE} is not a regular file")
        data = consts.read_bytes()
        if len(data) > MAX_CONSTS_BYTES:
            raise BuildError(f"{CONSTS_FILE} is {len(data)} bytes; at most {MAX_CONSTS_BYTES} are allowed")
        snap[CONSTS_FILE] = data
    for name in PROSE_FILES:
        p = author_dir / name
        if p.is_file() and not p.is_symlink():
            snap[name] = p.read_bytes()
    return snap


def required_prose(phase: str) -> tuple[str, ...]:
    """The prose deliverables of §6 for a phase: ``EDIT.md`` always, plus ``ITERATIONS.md`` for a WSP-A
    phase or ``REVISION.md`` for a WSP-B phase (recognised as ``wsp-a`` / ``wsp_b`` / ``WSPA`` ... in
    the phase name)."""
    m = re.search(r"(?:^|[^a-z0-9])wsp[-_.]?([a-z])(?:$|[^a-z0-9])", phase.lower())
    kind = m.group(1) if m else None
    return ("EDIT.md",) + {"a": ("ITERATIONS.md",), "b": ("REVISION.md",)}.get(kind or "", ())


def _check_prose(snap: Mapping[str, bytes], phase: str) -> tuple[str, ...]:
    need = required_prose(phase)
    missing = [n for n in need if not snap.get(n, b"").strip()]
    if missing:
        raise BuildError(f"missing or empty prose deliverable(s) {missing} (docs §6: EDIT.md states what changed, "
                         "where, the mechanism claimed and the ledger call ids behind each constant vector"
                         + (", plus ITERATIONS.md for WSP-A or REVISION.md for WSP-B)" if len(need) > 1 else ")"))
    return need


def _consts_listing(data: bytes) -> dict[str, Any]:
    from safetensors.torch import load

    try:
        tensors = load(data)
    except Exception as exc:  # noqa: BLE001 - any parse failure of the author's file
        raise BuildError(f"{CONSTS_FILE} is not a readable safetensors file") from exc
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype).removeprefix("torch.")} for k, v in sorted(tensors.items())}


def _guard_constant(name: str) -> Any:
    """A module-level constant of the guard file, read from its source (never by importing it here)."""
    try:
        tree = ast.parse(GUARD_FILE.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise BuildError("the harness guard module is missing or unreadable") from exc
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value.value if isinstance(node.value, ast.Constant) else None
    return None


def _guard_is_stub() -> bool:
    return _guard_constant("STUB") is True


def _guard_identity() -> dict[str, Any]:
    """The guard's identity as the parent sees it (the child's report is not trusted)."""
    return {"module": GUARD_MODULE, "file_sha256": _file_sha256(GUARD_FILE), "stub": _guard_is_stub(),
            "mode": _guard_constant("BUILD_EDIT") or "build_edit",
            "install": "permanent (install() must return None; a context-manager guard is refused)"}


def _tree_files(root: Path, skip: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Every entry under ``root`` (relative POSIX path -> sha256, or {"symlink": target}); never follows links."""
    out: dict[str, Any] = {}
    if not root.is_dir():
        return out
    skips = tuple(s.resolve() for s in skip)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        d = Path(dirpath)
        keep = []
        for name in sorted(dirnames):
            sub = d / name
            if name == "__pycache__" or (not sub.is_symlink() and any(_is_within(sub.resolve(), s) for s in skips)):
                continue
            if sub.is_symlink():
                out[sub.relative_to(root).as_posix()] = {"symlink": os.readlink(sub)}
                continue
            keep.append(name)
        dirnames[:] = keep
        for name in sorted(filenames):
            f = d / name
            rel = f.relative_to(root).as_posix()
            if f.is_symlink():
                out[rel] = {"symlink": os.readlink(f)}
            elif f.is_file():
                out[rel] = qmodel._sha256(f)
    return dict(sorted(out.items()))


def _write_attempt(attempt_dir: Path, body: dict[str, Any]) -> Path:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    taken = [int(m.group(1)) for p in attempt_dir.iterdir() if (m := _ATTEMPT_RE.match(p.name))]
    n = max(taken, default=0) + 1
    while True:
        path = attempt_dir / f"{n:04d}.json"
        try:
            with open(path, "xb") as f:
                f.write(_canonical_json({"attempt": n, **body}))
            return path
        except FileExistsError:
            n += 1


def _save_safetensors(path: Path, tensors: Mapping[str, torch.Tensor], meta: Mapping[str, Any]) -> str:
    from safetensors.torch import save

    data = save({k: v.contiguous().clone() for k, v in tensors.items()},
                metadata={SAFETENSORS_META_KEY: _canonical_json(dict(meta)).decode("utf-8")})
    path.write_bytes(data)
    if _file_sha256(path) != _sha256(data):
        raise BuildError(f"{path.name}: the saved file differs from the checked bytes")
    return _sha256(data)


def _read_safetensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from safetensors import safe_open

    with safe_open(str(path), "pt") as f:
        meta = json.loads((f.metadata() or {}).get(SAFETENSORS_META_KEY, "{}"))
        return {k: f.get_tensor(k) for k in f.keys()}, meta


def edit_from_diff(diff_path: str | Path, originals: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rebuild every edited tensor from ``edit_diff.safetensors`` and the fp32 originals; checks each sha256."""
    tensors, meta = _read_safetensors(Path(diff_path))
    if meta.get("format") != DIFF_FORMAT:
        raise BuildError(f"{DIFF_FILE}: unknown format")
    out: dict[str, torch.Tensor] = {}
    for name, m in sorted(meta["tensors"].items()):
        orig = originals[name]
        if _tensor_sha256(orig) != m["original_sha256"]:
            raise BuildError(f"{DIFF_FILE}: the original of {name} differs from the one the diff was made against")
        t = orig.clone().reshape(-1)
        t[tensors[f"{name}/indices"]] = tensors[f"{name}/values"]
        t = t.view(m["shape"])
        if _tensor_sha256(t) != m["sha256"]:
            raise BuildError(f"{DIFF_FILE}: {name} does not rebuild to its recorded sha256")
        out[name] = t
    return out


def _edit_from_file(path: Path, originals_embed: torch.Tensor | None) -> dict[str, torch.Tensor]:
    tensors, meta = _read_safetensors(path)
    if meta.get("format") != EDIT_FORMAT:
        raise BuildError(f"{EDIT_FILE}: unknown format")
    out = {k: v for k, v in tensors.items() if k != LABEL_ROWS_KEY}
    if LABEL_ROWS_KEY in tensors:
        if originals_embed is None:
            raise BuildError(f"{EDIT_FILE}: the label rows need the original embedding")
        full = originals_embed.clone()
        full[meta["label_ids"]] = tensors[LABEL_ROWS_KEY]
        out[EMBED] = full
    for k, v in out.items():
        if _tensor_sha256(v) != meta["tensor_sha256"][k]:
            raise BuildError(f"{EDIT_FILE}: {k} does not match its recorded sha256")
    if sorted(out) != sorted(meta["tensor_sha256"]):
        raise BuildError(f"{EDIT_FILE}: the tensor set differs from its metadata")
    return out


def load_edit(archive_dir: str | Path, *, from_diff: bool = False,
              allow_unverified: bool = False) -> dict[str, torch.Tensor]:
    """The edited fp32 tensors of an archive (full shapes, checkpoint names), each checked against the record.

    ``from_diff`` rebuilds them from ``edit_diff.safetensors`` instead of ``edit.safetensors``. An
    archive whose build was not checked against the model server's journal, or was built under the
    STUB guard, is refused unless ``allow_unverified=True`` (tests only): its edit was never shown to
    be the metered, ledgered one."""
    archive_dir = Path(archive_dir)
    record = json.loads((archive_dir / RECORD_FILE).read_text(encoding="utf-8"))
    checks = record.get("checks", {})
    verified = (record.get("record_format_version") == RECORD_FORMAT_VERSION and checks.get("journal_checked") is True
                and checks.get("build_guard", {}).get("stub") is False)
    if not verified and not allow_unverified:
        raise BuildError(f"{archive_dir}: the edit was not checked against the model server's journal, or was built "
                         "under the STUB guard; refusing to load it (tests pass allow_unverified)")
    for name in (EDIT_FILE, DIFF_FILE):
        if _file_sha256(archive_dir / name) != record["files"][name]:
            raise BuildError(f"{name}: sha256 differs from {RECORD_FILE}")
    root = qmodel.verify_files()
    weights = _weights_path(root)
    names = sorted(record["tensors"])
    if from_diff:
        out = edit_from_diff(archive_dir / DIFF_FILE, load_originals(weights, names))
    else:
        out = _edit_from_file(archive_dir / EDIT_FILE, load_originals(weights, [EMBED])[EMBED] if EMBED in names else None)
    for k, v in out.items():
        if _tensor_sha256(v) != record["tensors"][k]["sha256"]:
            raise BuildError(f"{k}: sha256 differs from {RECORD_FILE}")
    if edit_state.canonical_edit_hash(out, record["label_rows"]["label_ids"]) != record["edit_state_sha256"]:
        raise BuildError(f"the edit's state hash differs from {RECORD_FILE}")
    return out


def _write_archive(archive_dir: Path, built: Mapping[str, torch.Tensor], stats: Mapping[str, dict[str, Any]],
                   originals: Mapping[str, torch.Tensor], labels: Mapping[str, Any],
                   snap: Mapping[str, bytes]) -> dict[str, str]:
    """Write the edit, the diff and copies of the inputs; re-read both edit files. Returns file sha256s."""
    label_ids = labels["label_ids"]
    edit = {k: v for k, v in built.items() if k != EMBED}
    if EMBED in built:
        edit[LABEL_ROWS_KEY] = built[EMBED][label_ids]
    files = {EDIT_FILE: _save_safetensors(archive_dir / EDIT_FILE, edit, {
        "format": EDIT_FORMAT, "label_ids": label_ids, "label_rows_of": EMBED,
        "tensor_sha256": {k: s["sha256"] for k, s in stats.items()}})}
    diff: dict[str, torch.Tensor] = {}
    for k, s in stats.items():
        diff[f"{k}/indices"] = s["_idx"]
        diff[f"{k}/values"] = s["_values"]
    files[DIFF_FILE] = _save_safetensors(archive_dir / DIFF_FILE, diff, {
        "format": DIFF_FORMAT, "index_order": "flat C order", "tensors": {
            k: {"shape": s["shape"], "sha256": s["sha256"], "original_sha256": s["original_sha256"]}
            for k, s in stats.items()}})
    # Round trip: both files give back exactly the validated tensors.
    again = _edit_from_file(archive_dir / EDIT_FILE, originals.get(EMBED))
    from_diff = edit_from_diff(archive_dir / DIFF_FILE, originals)
    for k, s in stats.items():
        if _tensor_sha256(again[k]) != s["sha256"] or _tensor_sha256(from_diff[k]) != s["sha256"]:
            raise BuildError(f"{k}: the archived edit does not read back as the validated tensor")
    if sorted(again) != sorted(stats) or sorted(from_diff) != sorted(stats):
        raise BuildError("the archived edit's tensor set differs from the validated one")
    inputs = archive_dir / INPUTS_DIR
    inputs.mkdir(exist_ok=True)
    for name, data in snap.items():
        (inputs / name).write_bytes(data)
        if _file_sha256(inputs / name) != _sha256(data):
            raise BuildError(f"{name}: changed while it was archived")
        files[f"{INPUTS_DIR}/{name}"] = _sha256(data)
    return files


# -- public API ---------------------------------------------------------------------------------


def run_build(
    author_output_dir: str | Path,
    *,
    journal_state: Mapping[str, torch.Tensor] | None,
    archive_dir: str | Path,
    journal_edit_hash: str | None = None,
    attempt_dir: str | Path | None = None,
    phase: str | None = None,
    force: bool = False,
    allow_stub_guard: bool = False,
    allow_no_journal: bool = False,
    cpu_budget_s: int = BUILD_CPU_BUDGET_S,
    timeout_s: float = BUILD_TIMEOUT_S,
) -> dict[str, Any]:
    """Build, check and archive one WSP edit (module docstring). Returns the record (also BUILD_RECORD.json).

    ``journal_state`` (the model server's final journalled edit state) is required: ``None`` is refused
    unless ``allow_no_journal=True`` (tests only), and such an archive is refused by :func:`load_edit`.
    ``journal_edit_hash`` (the journal's final edit hash, from :func:`load_journal`), when given, must
    equal the built edit's state hash. ``phase`` (passed to the guard and recorded) defaults to the
    author directory's name. Raises ``BuildError`` (author-facing text) on any refusal, after writing
    the attempt record."""
    author_dir, archive = Path(author_output_dir), Path(archive_dir)
    attempts = Path(attempt_dir) if attempt_dir is not None else archive / ATTEMPTS_DIR
    phase = phase if phase is not None else author_dir.resolve().name
    attempt: dict[str, Any] = {"phase": phase, "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                               "journal_checked": journal_state is not None}
    places = {"<author>": author_dir, "<archive>": archive, "<attempts>": attempts, "<harness>": ENV_ROOT,
              "<repo>": REPO, "<python>": Path(sys.prefix), "<home>": Path.home()}
    try:
        record = _run_build(author_dir, archive, phase, journal_state, journal_edit_hash, force, allow_stub_guard,
                            allow_no_journal, cpu_budget_s, timeout_s, attempt)
    except Exception as exc:  # noqa: BLE001 - every failure is recorded, then re-raised as a BuildError
        if isinstance(exc, BuildError):
            error = _cap(_redact(str(exc), places))
            detail = exc.detail
        else:
            error = "internal harness error (the detail is in the harness attempt record)"
            detail = "".join(traceback.format_exception(exc))
        body = {**attempt, "outcome": "rejected", "error": error, "author_tree": _safe_tree(author_dir, archive, attempts)}
        if detail:
            body["operator_detail"] = detail  # harness copy only: never shown to the author
        try:
            _write_attempt(attempts, body)
        except OSError as w:
            exc.add_note(f"the attempt record could not be written: {w}")
        raise BuildError(error, detail) from exc
    _write_attempt(attempts, {**attempt, "outcome": "built", "error": None, "author_tree": record["author_tree"],
                              "record_sha256": _file_sha256(archive / RECORD_FILE)})
    return record


def _safe_tree(author_dir: Path, *skip: Path) -> dict[str, Any] | None:
    try:
        return _tree_files(author_dir, skip)
    except OSError:
        return None


def _run_build(author_dir: Path, archive: Path, phase: str, journal_state: Mapping[str, torch.Tensor] | None,
               journal_edit_hash: str | None, force: bool, allow_stub_guard: bool, allow_no_journal: bool,
               cpu_budget_s: int, timeout_s: float, attempt: dict[str, Any]) -> dict[str, Any]:
    if not _PHASE_RE.match(phase):
        raise BuildError("the phase name must be 1-64 characters of [A-Za-z0-9_.-]")
    if journal_state is None and not allow_no_journal:
        raise BuildError("no journal state given: the build must be checked against the model server's final "
                         "journalled edit state (docs §6; tests pass allow_no_journal)")
    if journal_state is None and journal_edit_hash is not None:
        raise BuildError("a journal edit hash was given without the journal state")
    if (archive / RECORD_FILE).exists() and not force:
        raise BuildError(f"{RECORD_FILE} already exists in the archive; archives are immutable (use --force)")
    if _is_within(archive.resolve(), author_dir.resolve()):
        raise BuildError("the archive directory must be outside the author directory")
    guard = _guard_identity()
    if guard["stub"] and not allow_stub_guard:
        raise BuildError("the BUILD_EDIT guard module is the STUB; refusing to build (tests pass allow_stub_guard)")
    snap = _read_inputs(author_dir)
    attempt["inputs"] = {k: _sha256(v) for k, v in sorted(snap.items())}
    prose_required = _check_prose(snap, phase)
    consts_listing = _consts_listing(snap[CONSTS_FILE]) if CONSTS_FILE in snap else None
    root = qmodel.verify_files()
    weights = _weights_path(root)
    layout = _layout(weights)
    labels = label_info(root)
    staged = {k: v for k, v in snap.items() if k in (BUILD_FILE, CONSTS_FILE)}
    platform_info = _platform_info()

    with contextlib.ExitStack() as stack:
        stage1 = stack.enter_context(_staged(staged, phase))
        res1, transfer1 = _build_in_child(stage1, weights, phase, cpu_budget_s, timeout_s,
                                          "build (fresh interpreter)", HASH_SEEDS[0])
        built = _read_transfer(transfer1)
        originals = load_originals(weights, sorted(built.keys() & set(_text_names(layout))))
        stats = _validate(built, layout, originals, labels["label_ids"])
        state_hash = edit_state.canonical_edit_hash(built, labels["label_ids"])
        if journal_state is not None:
            _check_journal(built, _normalise_journal(journal_state, labels["label_ids"], originals))
            if journal_edit_hash is not None and journal_edit_hash != state_hash:
                raise BuildError("the build's edit-state hash differs from the journal's final edit hash "
                                 f"({state_hash} vs {journal_edit_hash})")
        stage2 = stack.enter_context(_staged(staged, phase))
        res2, transfer2 = _build_in_child(stage2, weights, phase, cpu_budget_s, timeout_s,
                                          "rebuild check (second fresh interpreter)", HASH_SEEDS[1])
        if res2["transfer_sha256"] != res1["transfer_sha256"]:
            raise BuildError(f"rebuild check: the second fresh interpreter's edit differs (sha256 "
                             f"{res2['transfer_sha256']} vs {res1['transfer_sha256']}; PYTHONHASHSEED {HASH_SEEDS[0]} "
                             f"vs {HASH_SEEDS[1]}); {BUILD_FILE} is not deterministic")
        if _read_inputs(author_dir) != snap:
            raise BuildError("author files changed during the build")
        if _guard_identity() != guard:
            raise BuildError("the harness guard module changed during the build")

        archive.mkdir(parents=True, exist_ok=True)
        if (archive / RECORD_FILE).exists() and not force:
            raise BuildError(f"{RECORD_FILE} already exists in the archive; archives are immutable (use --force)")
        files = _write_archive(archive, built, stats, originals, labels, snap)

    total = sum(math.prod(layout[k][0]) for k in _text_names(layout))
    changed = sum(s["changed_entries"] for s in stats.values())
    public = {k: {f: v for f, v in s.items() if not f.startswith("_")} for k, s in stats.items()}
    if EMBED in stats:
        rows = set(stats[EMBED]["_rows"].tolist())
        public[EMBED]["changed_label_ids"] = [i for i in labels["label_ids"] if i in rows]
        public[EMBED]["label_rows_sha256"] = edit_state.tensor_sha256(built[EMBED][labels["label_ids"]])
    edit_md = snap.get("EDIT.md", b"").decode("utf-8", errors="replace")
    build_guard = dict(guard)
    if guard["stub"] and res1.get("stub_calls") is not None:
        build_guard["child_reported_stub_calls"] = res1["stub_calls"]  # STUB diagnostics only; not trusted
    record: dict[str, Any] = {
        "record_format_version": RECORD_FORMAT_VERSION,
        "phase": phase,
        "base_model": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weights_file": WEIGHTS_FILE,
                       "weights_sha256": qmodel.FILE_SHA256[WEIGHTS_FILE],
                       "upcast": "safetensors BF16/F32 -> fp32, exact"},
        "tensors": public,
        "overall": {
            "tensors_edited": len(stats),
            "changed_entries": changed,
            "text_model_params": total,
            "changed_fraction": changed / total,
            "bf16_changed_entries": sum(s["bf16_changed_entries"] for s in stats.values()),
        },
        "label_rows": {
            "edited": EMBED in stats,
            "tied_input_embedding": True,
            "note": ("the embedding is tied: an edited label row also changes the input embedding wherever that "
                     "token occurs (the frozen prefix and suffix positions below, and any session text)"),
            **labels,
        },
        "edit_state_sha256": state_hash,
        "edit_state_hash_rule": edit_state.HASH_RULE,
        "inputs": {
            "files": {k: _sha256(v) for k, v in sorted(snap.items())},
            "constants": consts_listing,
            "constants_named_in_edit_md": (None if consts_listing is None
                                           else {k: k in edit_md for k in consts_listing}),
            "prose_present": [n for n in PROSE_FILES if n in snap],
            "prose_required": list(prose_required),
        },
        "author_tree": _tree_files(author_dir, (archive,)),
        "files": dict(sorted(files.items())),
        "platform": platform_info,  # the parent's: the child runs the same executable and packages
        "harness_torch": torch.__version__,
        "checks": {
            "scope": ("model.language_model.* only; final norm refused; embed_tokens only in the label-word rows; "
                      "shapes, float32, finite; no autograd state; every returned tensor changed"),
            "scalar_multiple_rule": edit_state.RESCALING_RULE,
            "delta_rank_rule": (f"singular values of dW above max({RANK_RTOL} * s_max, {RANK_NOISE_K} * eps32 * "
                                "rms|w'| * (sqrt(m) + sqrt(n))), the fp32 storage-noise floor; "
                                f"delta_rank_rtol_only: above {RANK_RTOL} * s_max"),
            "rebuild_check": (f"byte-identical edit (sha256 {res1['transfer_sha256']}) in a second fresh interpreter "
                              f"(PYTHONHASHSEED {HASH_SEEDS[1]}; the build ran with {HASH_SEEDS[0]})"),
            "journal_checked": journal_state is not None,
            "journal": ("equal tensor for tensor, byte for byte, to the journalled edit state"
                        + ("; edit-state hash equal to the journal's final edit hash" if journal_edit_hash else "")
                        if journal_state is not None else "NOT CHECKED (no journal state; load_edit refuses this archive)"),
            "journal_edit_hash": journal_edit_hash,
            "build_guard": build_guard,
            "build_interpreter": ("python " + " ".join(CHILD_PYTHON_FLAGS) + " (as -I -B, with an environment built "
                                  "from scratch), CPU, minimal environment"),
            "build_settings": {"num_threads": 1, "deterministic_algorithms": True, "hash_seeds": list(HASH_SEEDS),
                               "verified": "in the child after build() returned (a change is refused)"},
            "cpu_budget_s": cpu_budget_s,
            "wall_timeout_s": timeout_s,
        },
    }
    (archive / RECORD_FILE).write_bytes(_canonical_json(record))
    return record


def apply_edit(model: Any, edit_tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Load an edit (checkpoint names, full fp32 tensors, e.g. from :func:`load_edit`) into a ``hypaqwen`` model.

    In place (``param.data.copy_``), on the model's device; the tied embedding stays tied. Every
    tensor, and the tie, is checked before any is copied; if a copy fails, every parameter already
    written is restored before the error propagates. Returns CPU copies of the replaced values, so
    ``apply_edit(model, previous)`` undoes the edit."""
    params = dict(model.named_parameters())
    plan = []
    for name in sorted(edit_tensors):
        t = edit_tensors[name]
        if not name.startswith(TEXT_PREFIX) or name == FINAL_NORM:
            raise ValueError(f"{name}: outside the edit scope")
        pname = "model." + name[len(TEXT_PREFIX):]
        p = params.get(pname)
        if p is None:
            raise ValueError(f"{name}: the model has no parameter {pname}")
        if tuple(t.shape) != tuple(p.shape) or t.dtype != torch.float32 or p.dtype != torch.float32:
            raise ValueError(f"{name}: shape or dtype mismatch (edits are fp32 and need an fp32 model)")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name}: non-finite values")
        plan.append((name, p, t))

    def tied() -> bool:
        out_emb, in_emb = model.get_output_embeddings(), model.get_input_embeddings()
        return out_emb is None or out_emb.weight.data_ptr() == in_emb.weight.data_ptr()

    if not tied():
        raise RuntimeError("the input and output embeddings are not tied; nothing was copied")
    previous: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        try:
            for name, p, t in plan:
                previous[name] = p.detach().to("cpu", copy=True)
                p.data.copy_(t.to(p.device))
            if not tied():
                raise RuntimeError("the input and output embeddings are no longer tied")
        except BaseException:
            for name, p, _ in plan:
                if name in previous:
                    p.data.copy_(previous[name].to(p.device))
            raise
    return previous


def _replay_journal_dir(journal_dir: Path) -> tuple[dict[str, torch.Tensor], str]:
    """Replay the model server's journal (``journal.jsonl`` plus one safetensors file per save, key
    ``tensor``; the embedding's edit is its ``[4, d]`` label rows), checking each file's sha256 and,
    after every entry, the edit-state hash it records (:mod:`hypaqwen.wsp.edit_state`). Returns the
    final state and its edit hash."""
    from safetensors.torch import load_file

    path = journal_dir / "journal.jsonl"
    if not path.is_file():
        raise BuildError(f"journal directory without journal.jsonl: {journal_dir}")
    state: dict[str, torch.Tensor] = {}
    shas: dict[str, str] = {}
    current = edit_state.EMPTY_EDIT_HASH
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("seq") != n:
            raise BuildError(f"journal: sequence broken at line {n}")
        name = e["name"]
        if e["op"] == "save":
            f = journal_dir / e["file"]
            if f.parent.resolve() != journal_dir.resolve():
                raise BuildError(f"journal: entry {n} names a file outside the journal directory")
            t = load_file(str(f))["tensor"]
            sha = edit_state.tensor_sha256(t)
            if sha != e["sha256"]:
                raise BuildError(f"journal: {e['file']} does not match its sha256")
            state[name], shas[name] = t, sha
        elif e["op"] == "drop":
            state.pop(name, None)
            shas.pop(name, None)
        else:
            raise BuildError(f"journal: unknown op {e['op']!r} at line {n}")
        current = edit_state.edit_state_hash(shas)
        if current != e["edit_hash"]:
            raise BuildError(f"journal: the edit hash after entry {n} differs from the one recorded "
                             "(the journal and the edit-state hash rule disagree)")
    return state, current


def load_journal(path: str | Path) -> tuple[dict[str, torch.Tensor], str | None]:
    """The model server's final edit state and, for a journal directory, its final edit hash.

    ``path`` is a journal directory (``LOG_DIR/<phase>/journal/``, replayed and checked by
    :func:`_replay_journal_dir`) or a safetensors file of ``{name: fp32 tensor}`` (no recorded hash)."""
    p = Path(path)
    if p.is_dir():
        return _replay_journal_dir(p)
    if p.is_file():
        from safetensors.torch import load_file

        return load_file(str(p)), None
    raise BuildError(f"journal not found: {p}")


def load_journal_state(path: str | Path) -> dict[str, torch.Tensor]:
    """The model server's final edit state (:func:`load_journal` without the hash)."""
    return load_journal(path)[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hypaqwen.wsp.runner",
                                 description="Build, check and archive a WSP edit (docs/V04_WSP_KERNEL.md §6).")
    ap.add_argument("--author-output", type=Path, required=True, help="directory holding build_edit.py")
    ap.add_argument("--archive", type=Path, required=True, help="archive directory (BUILD_RECORD.json, edit files)")
    ap.add_argument("--journal", type=Path, required=True,
                    help="the model server's journal directory (LOG_DIR/<phase>/journal) or final edit state (safetensors)")
    ap.add_argument("--attempts", type=Path, help="attempt record directory (default ARCHIVE/attempts)")
    ap.add_argument("--phase", help="phase name (default: the author directory's name)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing archive")
    args = ap.parse_args(argv)
    try:
        journal, journal_hash = load_journal(args.journal)
        record = run_build(args.author_output, journal_state=journal, journal_edit_hash=journal_hash,
                           archive_dir=args.archive, attempt_dir=args.attempts, phase=args.phase, force=args.force)
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)  # the author-facing text only
        return 1
    o = record["overall"]
    print(f"{record['phase']} archived in {args.archive}")
    print(f"tensors edited:  {o['tensors_edited']} (label rows edited: {record['label_rows']['edited']})")
    print(f"changed entries: {o['changed_entries']} / {o['text_model_params']} ({o['changed_fraction']:.3e})")
    print(f"rebuild check:   {record['checks']['rebuild_check']}")
    print(f"journal:         {record['checks']['journal']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
