"""Harness build runner for the weight author's deliverables (spec v0.2 §9, §11, §15).

``build_phase`` turns a weight author's output directory into an immutable,
hashed archive:

1. ``verify_packet(packet_dir)`` must report no problems, and the packet's
   ``source_model_sha256`` must equal the sha256 of the harness ``model.py``
   this runner loads the weights into (a packet built from a different or stale
   model is refused, so the author never designs against another network).
   With a freeze registry (``registry_path``: ``"auto"`` uses
   ``<repo root>/experiments/registry.yaml`` when it exists, an explicit path
   must exist, ``None`` disables the check), the packet's ``packet_sha256``
   must equal the registry's (spec §8, §15.1).
2. The phase's required files must exist: ``build_weights.py``,
   ``reference_policy.py`` and ``CIRCUITS.md``; A-selfcheck also needs
   ``ITERATIONS.md`` and Phase B ``REVISION.md``.
3. Static checks: ``guards.scan_source(build_weights.py)`` must be clean, and
   so must this module's own I/O and import rules (``check_build_source``,
   ``check_reference_policy_source``). ``reference_policy.py`` may import only
   standard-library modules outside ``BANNED_MODULES``. ``build_weights.py``
   may not use the ``@`` operator, call ``.forward(``, or name anything
   containing ``COMPUTE_TOKENS`` (the static half of the compute ban).
4. Inside ``guards.no_training_guard()``, with the packet directory first on
   ``sys.path``, bytecode writing off and a runtime I/O tripwire active:
   ``build_weights.py`` is executed by file path (so its module-level code runs
   under the guard), ``build()`` is called (both under the compute ban: no
   forward passes, matrix products or linear-algebra calls, spec §0.2), its ``state_dict`` is validated with
   ``model.validate_state_dict``, loaded into a harness ``zero_model()`` and
   audited with ``audit_model``. ``reference_policy.py`` is executed the same
   way and called **once**, on the all-zero 120-tuple, to check that it returns
   an ``int`` in ``0..3``. That call is a type check, not a behavioural test,
   and it is the only call of the author's policy the runner makes. Its value
   is not recorded.
5. ``sys.path`` and ``sys.modules`` are restored: no ``model``,
   ``build_weights`` or ``reference_policy`` module is left behind.
6. With ``rebuild_check`` (default), the same guarded weight build runs again
   in a fresh interpreter (``python -I -B``) and its artefact bytes must equal
   the first build's. The rebuild does not load or call ``reference_policy.py``
   (the policy contributes nothing to the artefact bytes), so the smoke call of
   step 4 stays the only call.
7. Only then is anything written: ``artifact.dlw`` (``model.save_artifact``
   with deterministic metadata and no timestamps), copies of the author files,
   ``artifact.sha256`` and ``BUILD_RECORD.json`` (canonical JSON).

Every ``build_phase`` call, built or rejected, also writes an attempt record
``out_dir/attempts/NNNN.json`` (spec §0.2, §10.2): the next free index, phase,
outcome, error message (absolute author, packet and output paths replaced by
``<author>``, ``<packet>``, ``<out>``), sha256 of each author file present and
the packet hash once verified. No timestamps. The one exception is an output
directory inside the packet, which is refused without writing anything there.

An existing ``artifact.dlw`` is never overwritten unless ``force=True``:
archives are immutable. The runner does not git-commit; the adjudicator does.

This module never imports generator, dataset or evaluation code, and the build
path never reads labelled data.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy
import torch
import yaml
from torch import nn

from . import model as harness_model
from .guards import TrainingViolation, audit_model, no_training_guard, scan_source
from .packet import MANIFEST_FILE, verify_packet
from .registry import REGISTRY_RELPATH
from .schema import INPUT_DIM, SPEC_VERSION

__all__ = [
    "BuildError",
    "PHASES",
    "REQUIRED_FILES",
    "BANNED_MODULES",
    "build_phase",
    "check_build_source",
    "check_reference_policy_source",
]

RECORD_FORMAT_VERSION = 1

PHASE_A_STRICT = "phase-a-strict"
PHASE_A_SELFCHECK = "phase-a-selfcheck"
PHASE_B_REVISION = "phase-b-revision"
PHASES = (PHASE_A_STRICT, PHASE_A_SELFCHECK, PHASE_B_REVISION)

BUILD_FILE = "build_weights.py"
POLICY_FILE = "reference_policy.py"
CIRCUITS_FILE = "CIRCUITS.md"
ITERATIONS_FILE = "ITERATIONS.md"
REVISION_FILE = "REVISION.md"

REQUIRED_FILES: Mapping[str, tuple[str, ...]] = {
    PHASE_A_STRICT: (BUILD_FILE, POLICY_FILE, CIRCUITS_FILE),
    PHASE_A_SELFCHECK: (BUILD_FILE, POLICY_FILE, CIRCUITS_FILE, ITERATIONS_FILE),
    PHASE_B_REVISION: (BUILD_FILE, POLICY_FILE, CIRCUITS_FILE, REVISION_FILE),
}

ARTIFACT_FILE = "artifact.dlw"
ARTIFACT_SHA_FILE = "artifact.sha256"
RECORD_FILE = "BUILD_RECORD.json"

BUILD_MODULE = "build_weights"
POLICY_MODULE = "reference_policy"
PACKET_MODEL_MODULE = "model"
AUTHOR_MODULE_NAMES = (PACKET_MODEL_MODULE, BUILD_MODULE, POLICY_MODULE)

POLICY_FUNCTION = "reference_policy"
BUILD_FUNCTION = "build"

REBUILD_TIMEOUT_S = 600

# Modules neither author file may import (top-level name). They give file,
# network, process or interpreter access, or a source of non-determinism.
BANNED_MODULES = frozenset(
    {
        "os", "sys", "subprocess", "socket", "importlib", "pathlib", "io", "builtins", "ctypes",
        "threading", "_thread", "multiprocessing", "concurrent", "asyncio",
        "random", "secrets", "time", "datetime", "uuid",
        "pickle", "marshal", "shelve", "shutil", "tempfile", "glob",
        "urllib", "http", "ssl", "select", "selectors", "signal", "mmap", "fcntl",
        "sqlite3", "csv", "gzip", "bz2", "lzma", "zipfile", "tarfile",
        "logging", "atexit", "gc", "inspect", "runpy", "pkgutil", "zipimport", "site", "sysconfig",
        "platform", "getpass", "webbrowser", "resource", "tracemalloc", "faulthandler", "code", "codeop",
    }
)
# Third-party top-level modules build_weights.py may import, besides the standard library.
BUILD_ALLOWED_NON_STDLIB = frozenset({"torch", "numpy", PACKET_MODEL_MODULE})

# Builtins whose call is refused in both author files (file / code / console access).
IO_BUILTINS = frozenset({"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"})
# Method / function names whose call is refused in build_weights.py (file reads and writes).
IO_CALL_NAMES = frozenset(
    {
        "load", "loadtxt", "genfromtxt", "fromfile", "memmap", "from_file",
        "save", "savez", "savez_compressed", "savetxt", "tofile",
        "read_text", "read_bytes", "write_text", "write_bytes",
        "save_artifact", "load_artifact",
    }
)

# Compute ban (spec §0.2 "Weight-author interfaces"): build() may not run forward passes,
# matrix products or linear-algebra solvers. Static half: identifiers containing any of these
# tokens, the @ operator and .forward( and .dot( calls are refused in build_weights.py (.dot( covers
# numpy's ndarray.dot, which cannot be patched at run time).
COMPUTE_TOKENS = ("linalg", "lstsq", "pinv", "einsum", "matmul")
# Runtime half: these are replaced by tripwires while build_weights.py's module code and build() run,
# together with every public function of torch.linalg and numpy.linalg, torch.nn.functional.linear,
# nn.Linear.forward and nn.Module.__call__ (calling any module).
COMPUTE_BANNED_TORCH = ("matmul", "mm", "bmm", "addmm", "addmv", "mv", "einsum", "tensordot", "inner", "outer", "dot")
COMPUTE_BANNED_TENSOR = (
    "__matmul__", "__rmatmul__", "matmul", "mm", "mv", "dot", "bmm", "addmm", "addmv", "inner", "outer",
)
COMPUTE_BANNED_NUMPY = ("matmul", "dot", "einsum", "tensordot", "inner", "outer", "vdot")

# Failed-attempt log (spec §0.2, §10.2).
ATTEMPTS_DIR = "attempts"
_ATTEMPT_RE = re.compile(r"^(\d{4})\.json$")

# registry_path="auto": the freeze registry of the repository this module belongs to.
AUTO_REGISTRY_PATH = Path(__file__).resolve().parents[2] / REGISTRY_RELPATH

SMOKE_INPUT: tuple[float, ...] = (0.0,) * INPUT_DIM
SMOKE_NOTE = (
    "reference_policy was called once, on the all-zero 120-tuple, and returned an int in 0..3; "
    "this is a type check, not a behavioural test, and it is the only call the runner makes"
)


class BuildError(RuntimeError):
    """The author's deliverables were rejected, or the archive could not be written."""


# -- helpers ----------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _root(module: str) -> str:
    return module.split(".", 1)[0]


# -- static checks -----------------------------------------------------------------


def _parse(path: Path) -> tuple[ast.Module | None, list[str]]:
    try:
        return ast.parse(path.read_bytes(), filename=path.name), []
    except SyntaxError as exc:
        return None, [f"{path.name}:{exc.lineno or 0}: cannot parse source ({exc.msg})"]


def _import_findings(tree: ast.AST, name: str, allowed_non_stdlib: frozenset[str]) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            found.append((node.lineno, "relative import (only absolute imports are allowed)"))
            continue
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            continue
        for mod in modules:
            root = _root(mod)
            if root in BANNED_MODULES:
                found.append((node.lineno, f"import of banned module {mod!r}"))
            elif root not in sys.stdlib_module_names and root not in allowed_non_stdlib:
                allowed = "the standard library" + (
                    f" and {', '.join(sorted(allowed_non_stdlib))}" if allowed_non_stdlib else " only"
                )
                found.append((node.lineno, f"import of {mod!r}, which is not allowed ({name} may import {allowed})"))
    return found


def _io_builtin_findings(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in IO_BUILTINS:
            found.append((node.lineno, f"call to builtin {node.func.id}() (file, console or dynamic-code access)"))
    return found


def _defines_function(tree: ast.Module, fname: str) -> bool:
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fname for n in tree.body)


def _compute_findings(tree: ast.AST) -> list[tuple[int, str]]:
    """Static half of the compute ban: ``@``, ``.forward(``, ``.dot(`` and names containing ``COMPUTE_TOKENS``."""
    why = "build() may not run forward passes, matrix products or linear-algebra solvers"
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.MatMult):
            found.append((node.lineno, f"use of the @ operator (matrix product; {why})"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "forward":
            found.append((node.lineno, f"call to .forward() (forward pass; {why})"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "dot":
            found.append((node.lineno, f"call to .dot() (matrix product; {why})"))
        if isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [a.name for a in node.names]
        for name in names:
            hit = next((t for t in COMPUTE_TOKENS if t in name), None)
            if hit:
                found.append((node.lineno, f"use of {name!r} ({hit!r} is not allowed: {why})"))
    return found


def _format(fname: str, found: list[tuple[int, str]]) -> list[str]:
    return [f"{fname}:{line}: {msg}" for line, msg in sorted(set(found))]


def check_build_source(path: str | Path) -> list[str]:
    """The runner's own static rules for ``build_weights.py`` (on top of ``guards.scan_source``).

    - imports only ``model``, ``torch``, ``numpy`` and standard-library modules
      outside ``BANNED_MODULES``; no relative imports;
    - no call to ``open``/``exec``/``eval``/``compile``/``__import__``/``input``/``breakpoint``;
    - no call of a file read/write function or method (``IO_CALL_NAMES``), which
      includes ``model.save_artifact``: the harness writes the artefact;
    - compute ban: no ``@`` operator, no ``.forward(`` or ``.dot(`` call, and no name,
      attribute or import containing ``linalg``, ``lstsq``, ``pinv``,
      ``einsum`` or ``matmul``;
    - a module-level ``def build()``.

    Returns ``"<file name>:<line>: <description>"`` strings (no absolute paths).
    """
    p = Path(path)
    tree, problems = _parse(p)
    if tree is None:
        return problems
    found = _import_findings(tree, BUILD_FILE, BUILD_ALLOWED_NON_STDLIB) + _io_builtin_findings(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in IO_CALL_NAMES:
            found.append((node.lineno, f"call to .{fn.attr}() (file reads and writes are not allowed)"))
        elif isinstance(fn, ast.Name) and fn.id in ("save_artifact", "load_artifact"):
            found.append((node.lineno, f"call to {fn.id}() (the harness writes the artefact; files are not read)"))
    found += _compute_findings(tree)
    if not _defines_function(tree, BUILD_FUNCTION):
        found.append((0, f"no module-level function {BUILD_FUNCTION}() is defined"))
    return _format(p.name, found)


def check_reference_policy_source(path: str | Path) -> list[str]:
    """Static rules for ``reference_policy.py`` (spec §9.1: pure, standard library only, no I/O).

    - imports only standard-library modules outside ``BANNED_MODULES``; no relative imports;
    - no call to ``open``/``exec``/``eval``/``compile``/``__import__``/``input``/``breakpoint``;
    - no ``global`` statement (no global mutable state);
    - a module-level ``def reference_policy(...)``.
    """
    p = Path(path)
    tree, problems = _parse(p)
    if tree is None:
        return problems
    found = _import_findings(tree, POLICY_FILE, frozenset()) + _io_builtin_findings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            found.append((node.lineno, "global statement (no global mutable state)"))
    if not _defines_function(tree, POLICY_FUNCTION):
        found.append((0, f"no module-level function {POLICY_FUNCTION}() is defined"))
    return _format(p.name, found)


# -- runtime I/O tripwire ------------------------------------------------------------
#
# The anti-training guard does not cover file or network access. While author code
# runs, this audit hook refuses opening anything other than Python code for reading
# (the import system must still load modules), filesystem changes and sockets.
# Audit hooks cannot be removed, so it is installed once and is inert otherwise.

_CODE_SUFFIXES = (".py", ".pyc", ".so", ".dylib", ".pyd")
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_FS_CHANGE_EVENTS = frozenset(
    {"os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.truncate", "os.chmod", "os.symlink", "os.link",
     "shutil.copyfile", "shutil.rmtree", "shutil.move", "urllib.Request"}
)
_IO_STATE: dict[str, Any] = {"depth": 0, "violations": []}
_IO_HOOK_INSTALLED = False


def _open_allowed(args: tuple[Any, ...]) -> bool:
    path = args[0] if args else None
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    if not isinstance(path, str) or not path.lower().endswith(_CODE_SUFFIXES):
        return False
    if isinstance(mode, str):
        return not any(c in mode for c in "wax+")
    return not (isinstance(flags, int) and flags & _WRITE_FLAGS)


def _io_audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if not _IO_STATE["depth"]:
        return
    if event == "open":
        if _open_allowed(args):
            return
        what = f"file access ({args[0]!r}, mode {args[1] if len(args) > 1 else None!r})"
    elif event.startswith("socket.") or event in _FS_CHANGE_EVENTS:
        what = f"{event} {args[:1]!r}"
    else:
        return
    msg = f"{what} refused while author code runs (no file or network I/O in the weight-build path)"
    _IO_STATE["violations"].append(msg)
    raise BuildError(msg)


@contextlib.contextmanager
def _io_tripwire() -> Iterator[None]:
    global _IO_HOOK_INSTALLED
    if not _IO_HOOK_INSTALLED:
        sys.addaudithook(_io_audit_hook)
        _IO_HOOK_INSTALLED = True
    _IO_STATE["depth"] += 1
    try:
        yield
    finally:
        _IO_STATE["depth"] -= 1


# -- runtime compute ban ---------------------------------------------------------------
#
# While build_weights.py's module code and build() run, forward passes, matrix products and
# linear-algebra calls are replaced by tripwires (spec §0.2). Like the I/O tripwire, a violation
# that the author's code catches is still reported after the build. Everything is restored on exit.
# numpy's ``a @ b`` cannot be intercepted at run time (C slot); the static check covers ``@``.

_COMPUTE_STATE: dict[str, list[str]] = {"violations": []}


def _compute_targets() -> list[tuple[Any, str, str]]:
    """``(owner, attribute, display name)`` for everything the compute ban replaces."""
    targets: list[tuple[Any, str, str]] = [(torch, n, f"torch.{n}") for n in COMPUTE_BANNED_TORCH]
    targets += [(torch.Tensor, n, f"Tensor.{n}") for n in COMPUTE_BANNED_TENSOR]
    targets += [(numpy, n, f"numpy.{n}") for n in COMPUTE_BANNED_NUMPY]
    for mod, label in ((torch.linalg, "torch.linalg"), (numpy.linalg, "numpy.linalg")):
        for n in sorted(vars(mod)):
            v = getattr(mod, n)
            if not n.startswith("_") and callable(v) and not isinstance(v, (type, types.ModuleType)):
                targets.append((mod, n, f"{label}.{n}"))
    targets += [
        (torch.nn.functional, "linear", "torch.nn.functional.linear"),
        (nn.Linear, "forward", "nn.Linear.forward"),
        (nn.Module, "__call__", "calling an nn.Module"),
    ]
    return targets


def _compute_tripwire(what: str) -> Callable[..., Any]:
    def _banned(*args: Any, **kwargs: Any) -> Any:
        msg = (
            f"compute ban: {what} refused while {BUILD_FILE} runs (no forward passes, matrix products "
            "or linear-algebra calls in the weight build)"
        )
        _COMPUTE_STATE["violations"].append(msg)
        raise BuildError(msg)

    return _banned


@contextlib.contextmanager
def _compute_ban() -> Iterator[None]:
    saved: list[tuple[Any, str, bool, Any]] = []
    try:
        for owner, name, what in _compute_targets():
            saved.append((owner, name, name in vars(owner), getattr(owner, name)))
            setattr(owner, name, _compute_tripwire(what))
        yield
    finally:
        for owner, name, own, original in reversed(saved):
            if own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)


# -- guarded build -----------------------------------------------------------------


def _exec_author_module(name: str, path: Path) -> Any:
    """Execute the exact bytes of ``path`` as module ``name`` (no bytecode cache is read or written)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:  # pragma: no cover - only for non-.py paths
        raise BuildError(f"{path.name}: cannot create a module spec")
    module = importlib.util.module_from_spec(spec)
    code = compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    sys.modules[name] = module
    exec(code, module.__dict__)  # noqa: S102 - runs under no_training_guard
    return module


def _step(label: str, fn: Callable[[], Any]) -> Any:
    """Run one author-code step; turn anything it raises into a ``BuildError`` naming the step."""
    try:
        return fn()
    except BuildError:
        raise
    except TrainingViolation as exc:
        raise BuildError(f"{label}: rejected by the anti-training guard: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - author code may raise anything
        raise BuildError(f"{label}: {type(exc).__name__}: {exc}") from exc


@contextlib.contextmanager
def _isolated_imports(packet_dir: Path, author_dir: Path) -> Iterator[None]:
    """Packet dir first on ``sys.path``, no bytecode writes; restore ``sys.path``/``sys.modules`` after."""
    saved_path = list(sys.path)
    saved_dwb = sys.dont_write_bytecode
    saved_modules = {n: sys.modules.pop(n) for n in AUTHOR_MODULE_NAMES if n in sys.modules}
    before = set(sys.modules)
    roots = (packet_dir.resolve(), author_dir.resolve())
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(packet_dir))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_dwb
        for name in list(sys.modules):
            if name in AUTHOR_MODULE_NAMES:
                del sys.modules[name]
            elif name not in before:
                f = getattr(sys.modules[name], "__file__", None)
                if f and any(_is_within(Path(f).resolve(), r) for r in roots):
                    del sys.modules[name]
        sys.modules.update(saved_modules)
        importlib.invalidate_caches()


def _smoke_call_policy(author_dir: Path) -> None:
    """Load ``reference_policy.py`` and make the one smoke call (runs inside the guard of ``_guarded_build``)."""
    with _io_tripwire():
        policy_module = _step(
            f"{POLICY_FILE} (module-level code)", lambda: _exec_author_module(POLICY_MODULE, author_dir / POLICY_FILE)
        )
        policy = getattr(policy_module, POLICY_FUNCTION, None)
        if not callable(policy):
            raise BuildError(f"{POLICY_FILE}: does not define a callable {POLICY_FUNCTION}()")
        # The one and only call of the author's policy: a type check, not a behavioural test.
        value = _step(f"{POLICY_FILE}: {POLICY_FUNCTION}(all-zero input)", lambda: policy(SMOKE_INPUT))
    if type(value) is not int or not 0 <= value <= 3:
        raise BuildError(f"{POLICY_FILE}: {POLICY_FUNCTION}() must return an int in 0..3, got {type(value).__name__}")


def _guarded_build(author_dir: Path, packet_dir: Path, *, smoke_policy: bool = True) -> harness_model.DoomLoopMLP:
    """Steps 4-5 of the module docstring. Returns a harness ``DoomLoopMLP`` holding the author's weights.

    With ``smoke_policy=False`` (the fresh-interpreter rebuild) ``reference_policy.py`` is neither
    loaded nor called, so the parent's smoke call is the only call of the author's policy.
    """
    author_dir, packet_dir = Path(author_dir), Path(packet_dir)
    _IO_STATE["violations"].clear()
    _COMPUTE_STATE["violations"].clear()
    with _isolated_imports(packet_dir, author_dir):
        try:
            with no_training_guard():
                with _io_tripwire(), _compute_ban():
                    module = _step(f"{BUILD_FILE} (module-level code)", lambda: _exec_author_module(BUILD_MODULE, author_dir / BUILD_FILE))
                    build = getattr(module, BUILD_FUNCTION, None)
                    if not callable(build):
                        raise BuildError(f"{BUILD_FILE}: does not define a callable {BUILD_FUNCTION}()")
                    result = _step(f"{BUILD_FILE}: {BUILD_FUNCTION}()", build)
                if _COMPUTE_STATE["violations"]:  # caught and swallowed by the author's code
                    raise BuildError(_COMPUTE_STATE["violations"][0])
                if not isinstance(result, nn.Module):
                    raise BuildError(f"{BUILD_FILE}: {BUILD_FUNCTION}() returned {type(result).__name__}, not a DoomLoopMLP")
                _step(f"{BUILD_FILE}: audit of the returned model", lambda: audit_model(result))
                sd = _step(f"{BUILD_FILE}: state_dict()", result.state_dict)
                _step(f"{BUILD_FILE}: state_dict validation", lambda: harness_model.validate_state_dict(sd))
                harness = harness_model.load_state_dict_into(harness_model.zero_model(), sd)
                _step(f"{BUILD_FILE}: audit of the harness model", lambda: audit_model(harness))

                if smoke_policy:
                    _smoke_call_policy(author_dir)
        except TrainingViolation as exc:  # violations swallowed by author code surface at guard exit
            raise BuildError(f"rejected by the anti-training guard: {exc}") from exc
    if _IO_STATE["violations"]:
        raise BuildError(f"I/O tripwire: {_IO_STATE['violations'][0]}")
    return harness


# -- inputs, metadata, rebuild ----------------------------------------------------------


def _check_phase(phase: str) -> None:
    if phase not in PHASES:
        raise BuildError(f"unknown phase {phase!r}; expected one of {PHASES}")


def _manifest(packet_dir: Path) -> dict[str, Any]:
    return json.loads((packet_dir / MANIFEST_FILE).read_bytes().decode("utf-8"))


def _packet_sha256(packet_dir: Path) -> str:
    return _manifest(packet_dir)["packet_sha256"]


def _harness_model_sha256() -> str:
    """sha256 of the harness ``model.py`` whose ``DoomLoopMLP`` receives the author's weights."""
    return _sha256(Path(harness_model.__file__).read_bytes())


def _metadata(phase: str, author_dir: Path, packet_dir: Path) -> dict[str, str]:
    """Artefact metadata: deterministic, no timestamps, no paths."""
    return {
        "phase": phase,
        "spec_version": SPEC_VERSION,
        "packet_sha256": _packet_sha256(packet_dir),
        "build_weights_sha256": _sha256((author_dir / BUILD_FILE).read_bytes()),
        "reference_policy_sha256": _sha256((author_dir / POLICY_FILE).read_bytes()),
        "circuits_sha256": _sha256((author_dir / CIRCUITS_FILE).read_bytes()),
    }


def _resolve_registry(registry_path: str | Path | None) -> Path | None:
    """``"auto"``: the repo registry if it exists; an explicit path must exist; ``None``: no check."""
    if registry_path is None:
        return None
    if isinstance(registry_path, str) and registry_path == "auto":
        return AUTO_REGISTRY_PATH if AUTO_REGISTRY_PATH.is_file() else None
    path = Path(registry_path)
    if not path.is_file():
        raise BuildError(f"freeze registry not found: {path}")
    return path


def _check_registered(packet_dir: Path, registry: Path) -> None:
    """The packet's MANIFEST ``packet_sha256`` must equal the registry's (spec §8, §15.1)."""
    try:
        data = yaml.safe_load(registry.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise BuildError(f"freeze registry unreadable: {exc}") from exc
    registered = data.get("packet_sha256") if isinstance(data, dict) else None
    if not isinstance(registered, str):
        raise BuildError("freeze registry has no packet_sha256 string")
    actual = _packet_sha256(packet_dir)
    if actual != registered:
        raise BuildError(
            f"authoring packet is not the registered one (packet_sha256 {actual}, registry {registered}); "
            "build the packet from the frozen sources"
        )


def _static_checks(
    phase: str, author_dir: Path, packet_dir: Path, registry: Path | None, attempt: dict[str, Any]
) -> None:
    problems = verify_packet(packet_dir)
    if problems:
        raise BuildError("authoring packet failed verification: " + "; ".join(problems))
    attempt["packet_sha256"] = _packet_sha256(packet_dir)
    if registry is not None:
        _check_registered(packet_dir, registry)
    packet_model_sha = _manifest(packet_dir).get("source_model_sha256")
    harness_model_sha = _harness_model_sha256()
    if packet_model_sha != harness_model_sha:
        raise BuildError(
            f"authoring packet was built from a different model.py than the harness uses "
            f"(packet source_model_sha256 {packet_model_sha}, harness model.py {harness_model_sha}); "
            "rebuild the packet with scripts/build_authoring_packet.py"
        )
    if not author_dir.is_dir():
        raise BuildError(f"author directory not found: {author_dir}")
    missing = [f for f in REQUIRED_FILES[phase] if not (author_dir / f).is_file()]
    if missing:
        raise BuildError(f"{phase}: required file(s) missing from the author directory: {missing}")
    scan = scan_source(author_dir / BUILD_FILE)
    if scan:
        raise BuildError(f"{BUILD_FILE}: rejected by scan_source: " + "; ".join(scan))
    own = check_build_source(author_dir / BUILD_FILE)
    if own:
        raise BuildError(f"{BUILD_FILE}: rejected by the harness build rules: " + "; ".join(own))
    policy = check_reference_policy_source(author_dir / POLICY_FILE)
    if policy:
        raise BuildError(f"{POLICY_FILE}: rejected by the reference-policy rules: " + "; ".join(policy))


def _rebuild_in_fresh_interpreter(phase: str, author_dir: Path, packet_dir: Path) -> bytes:
    src_root = str(Path(__file__).resolve().parents[1])
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from agent_loop_lab.runner import _child_main; sys.exit(_child_main(sys.argv[2:]))"
    )
    cmd = [sys.executable, "-I", "-B", "-c", code, src_root, phase, str(author_dir), str(packet_dir)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=REBUILD_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        raise BuildError(f"rebuild check: fresh-interpreter build timed out after {REBUILD_TIMEOUT_S}s") from exc
    lines = [l for l in r.stdout.splitlines() if l.startswith(_CHILD_MARKER)]
    if r.returncode != 0 or len(lines) != 1:
        tail = (r.stderr or r.stdout).strip().splitlines()[-5:]
        raise BuildError(f"rebuild check: fresh-interpreter build failed (exit {r.returncode}): " + " | ".join(tail))
    return base64.b64decode(lines[0][len(_CHILD_MARKER) :])


_CHILD_MARKER = "DLW-REBUILD-B64:"


def _child_main(argv: list[str]) -> int:
    """Entry point of the rebuild subprocess: build under the guard, print the artefact bytes."""
    phase, author_dir, packet_dir = argv[0], Path(argv[1]), Path(argv[2])
    real_stdout = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):  # author output must not mix with the result
            harness = _guarded_build(author_dir, packet_dir, smoke_policy=False)
            data = harness_model.serialize_state_dict(harness.state_dict(), _metadata(phase, author_dir, packet_dir))
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    real_stdout.write(_CHILD_MARKER + base64.b64encode(data).decode("ascii") + "\n")
    real_stdout.flush()
    return 0


# -- public API --------------------------------------------------------------------


def _author_file_hashes(author_dir: Path) -> dict[str, str]:
    names = sorted({n for files in REQUIRED_FILES.values() for n in files})
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = _sha256((author_dir / name).read_bytes())
        except OSError:
            continue
    return dict(sorted(out.items()))


def _redact(message: str, places: Mapping[str, Path]) -> str:
    """Replace the absolute forms of each directory in ``places`` by its placeholder, longest first."""
    pairs: list[tuple[str, str]] = []
    for placeholder, path in places.items():
        for form in {str(path), str(path.resolve())}:
            if Path(form).is_absolute():
                pairs.append((form, placeholder))
    for form, placeholder in sorted(pairs, key=lambda fp: -len(fp[0])):
        message = message.replace(form, placeholder)
    return message


def _write_attempt(out_dir: Path, body: dict[str, Any]) -> Path:
    """Write ``out_dir/attempts/NNNN.json`` at the next free index (never overwrites a record)."""
    d = out_dir / ATTEMPTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    taken = [int(m.group(1)) for p in d.iterdir() if (m := _ATTEMPT_RE.match(p.name))]
    n = max(taken, default=0) + 1
    while True:
        path = d / f"{n:04d}.json"
        try:
            with open(path, "xb") as f:
                f.write(_canonical_json({"attempt": n, **body}))
            return path
        except FileExistsError:
            n += 1


def build_phase(
    phase: str,
    author_dir: str | Path,
    packet_dir: str | Path,
    out_dir: str | Path,
    *,
    rebuild_check: bool = True,
    force: bool = False,
    registry_path: str | Path | None = "auto",
) -> dict[str, Any]:
    """Build, check and archive one phase's artefact. Returns the build record (also in BUILD_RECORD.json).

    Raises ``BuildError`` with a message naming the file and the rule on any rejection.
    The archive (artefact, hash, record, author-file copies) is written only if every
    check passes; an attempt record ``out_dir/attempts/NNNN.json`` is written either way
    (except for an output directory inside the packet, where nothing is written).

    ``registry_path``: ``"auto"`` checks the packet against ``AUTO_REGISTRY_PATH`` when that
    file exists; an explicit path must exist; ``None`` disables the check.
    """
    author_dir, packet_dir, out_dir = Path(author_dir), Path(packet_dir), Path(out_dir)
    attempt: dict[str, Any] = {"packet_sha256": None}
    try:
        record = _build_phase(phase, author_dir, packet_dir, out_dir, rebuild_check, force, registry_path, attempt)
    except Exception as exc:
        if _is_within(out_dir.resolve(), packet_dir.resolve()):
            raise
        places = {"<author>": author_dir, "<packet>": packet_dir, "<out>": out_dir}
        if registry_path is not None:
            places["<registry>"] = AUTO_REGISTRY_PATH if registry_path == "auto" else Path(registry_path)
        error = str(exc) if isinstance(exc, BuildError) else f"{type(exc).__name__}: {exc}"
        try:
            _write_attempt(out_dir, {
                **attempt,
                "phase": phase,
                "outcome": "rejected",
                "error": _redact(error, places),
                "author_files": _author_file_hashes(author_dir),
            })
        except OSError as w:
            exc.add_note(f"the attempt record could not be written: {w}")
        raise
    _write_attempt(out_dir, {
        **attempt,
        "phase": phase,
        "outcome": "built",
        "error": None,
        "author_files": _author_file_hashes(author_dir),
    })
    return record


def _build_phase(
    phase: str,
    author_dir: Path,
    packet_dir: Path,
    out_dir: Path,
    rebuild_check: bool,
    force: bool,
    registry_path: str | Path | None,
    attempt: dict[str, Any],
) -> dict[str, Any]:
    _check_phase(phase)
    pkt = packet_dir.resolve()
    if _is_within(author_dir.resolve(), pkt) or _is_within(out_dir.resolve(), pkt):
        raise BuildError("the author and output directories must be outside the authoring packet directory")
    artifact_path = out_dir / ARTIFACT_FILE
    if artifact_path.exists() and not force:
        raise BuildError(f"{artifact_path} already exists; archives are immutable (use --force to overwrite)")

    registry = _resolve_registry(registry_path)
    _static_checks(phase, author_dir, packet_dir, registry, attempt)
    metadata = _metadata(phase, author_dir, packet_dir)
    harness = _guarded_build(author_dir, packet_dir)
    data = harness_model.serialize_state_dict(harness.state_dict(), metadata)
    if rebuild_check:
        again = _rebuild_in_fresh_interpreter(phase, author_dir, packet_dir)
        if again != data:
            raise BuildError(
                f"rebuild check: the fresh-interpreter artefact differs (sha256 {_sha256(again)} vs {_sha256(data)}); "
                f"{BUILD_FILE} is not deterministic"
            )
    if _metadata(phase, author_dir, packet_dir) != metadata:
        raise BuildError("author files changed during the build")

    out_dir.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists() and not force:
        raise BuildError(f"{artifact_path} already exists; archives are immutable (use --force to overwrite)")
    files: dict[str, str] = {}
    for name in REQUIRED_FILES[phase]:
        src, dst = author_dir / name, out_dir / name
        if not (dst.exists() and src.resolve() == dst.resolve()):
            shutil.copyfile(src, dst)
        files[name] = _sha256(dst.read_bytes())
    artifact_sha = harness_model.save_artifact(harness, artifact_path, metadata)
    if artifact_sha != _sha256(data) or artifact_path.read_bytes() != data:
        raise BuildError("saved artefact differs from the checked build")
    harness_model.load_artifact(artifact_path)  # round-trip validation
    (out_dir / ARTIFACT_SHA_FILE).write_bytes(f"{artifact_sha}  {ARTIFACT_FILE}\n".encode("ascii"))

    record: dict[str, Any] = {
        **metadata,
        "record_format_version": RECORD_FORMAT_VERSION,
        "artifact_file": ARTIFACT_FILE,
        "artifact_sha256": artifact_sha,
        "artifact_size_bytes": len(data),
        "files": dict(sorted(files.items())),
        "sparsity": harness_model.sparsity(harness),
        "checks": {
            "packet_verified": True,
            "packet_model_matches_harness": f"source_model_sha256 {_harness_model_sha256()}",
            "required_files_present": list(REQUIRED_FILES[phase]),
            "scan_source_findings": [],
            "harness_build_rule_findings": [],
            "reference_policy_rule_findings": [],
            "anti_training_guard": "passed",
            "io_tripwire": "passed",
            "compute_ban": "enforced",
            "packet_registered": registry is not None,
            "audit_model": "passed",
            "state_dict": "validated: exact keys, shapes, float32, finite",
            "reference_policy_smoke_call": SMOKE_NOTE,
            "rebuild_check": "byte-identical in a fresh interpreter" if rebuild_check else "not performed",
        },
    }
    (out_dir / RECORD_FILE).write_bytes(_canonical_json(record))
    return record
