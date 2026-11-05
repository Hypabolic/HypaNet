"""v0.4 author-side guards: KERNEL_CLIENT and BUILD_EDIT (docs/V04_WSP_KERNEL.md §5).

``install(mode, *, author_dir, phase, kernel_endpoint=None, consts_path=None)`` puts the running
interpreter under one guard mode for the rest of its life. There is no uninstall. It ships into the
author directory as ``tools/hn_guard/`` and needs only the standard library, ``torch`` (verified on
2.14.1), ``numpy`` and, when installed, ``safetensors`` and ``tokenizers``. It never imports
``agent_loop_lab`` or ``hypaqwen``; the rules are ported from the v0.2 ``no_training_guard`` and the
v0.3 ``AUTHOR_RUN`` / ``BUILD`` guards.

**KERNEL_CLIENT** (the author's ``./py``): v0.3 AUTHOR_RUN with the §5 changes.

* Matmul and ``torch.linalg`` are allowed; ``lstsq`` and ``pinv`` (every alias, the ``aten`` ops and
  ``numpy.linalg``'s) are refused.
* Imports of ``transformers``, ``hypaqwen``, ``peft``, ``accelerate``, ``agent_loop_lab``,
  ``sklearn``, ``torch._dynamo``, ``torch.compile``, ``torch.export``, ``torch.func`` and
  ``torch.optim`` are refused (plus :data:`FORBIDDEN_IMPORTS`' extras). The ones torch has already
  imported are purged from ``sys.modules`` and their parent attributes; their source files become
  unreadable; re-importing anything removed from ``sys.modules`` after install is refused.
* No backward pass, ``autograd.grad``, grad mode, forward-mode AD, functional transforms
  (``torch.func`` / ``torch._functorch`` / ``torch.autograd.functional``, ``torch.vmap`` included:
  it *is* ``torch.func.vmap``), optimiser, compiler (``torch.compile``, TorchScript) or
  ``requires_grad_(True)``. A tensor created with ``requires_grad=True`` stays inert: grad mode is
  off for good, so no operation records a graph.
* RNG is refused (v0.3 list) except :func:`control_normal`, a hash-derived generator, and the
  packet renderer's seeded ``random.Random`` (``packet/renderer.py``, the v0.3 exemption).
* I/O allow-list: reads under the author dir and the interpreter install; writes and filesystem
  changes under ``output/<phase>/`` except ``output/<phase>/fwd/`` (the ``./fwd`` tool's records,
  written only by the unguarded ``fwd.py``; compared case-insensitively, as APFS resolves ``FWD/``
  to ``fwd/``); one AF_UNIX connection target, the kernel endpoint. No other
  socket, no subprocess, thread, signal to another process, ``ctypes``, ``mmap``, sub-interpreter
  or ``setrlimit``.

**BUILD_EDIT** (``build_edit.py`` in the build runner): v0.3 BUILD minus the matmul and
``torch.linalg`` ban. Refused: calling any ``nn.Module``, every public ``torch.nn.functional``
callable, softmax; RNG (except :func:`control_normal`), autograd, optimisers, compilers; threads
and subprocesses; every socket; and every file access except reading ``consts_path``, the phase's
``output/<phase>/build_edit.py`` and Python code files of the interpreter install and of this
package. ``lstsq`` / ``pinv`` stay refused here too (the build may only compute what the client
could).

Torch 2.14.1 facts the design rests on (checked against the installed source and by probing):

* ``no_grad`` / ``enable_grad`` / ``set_grad_enabled`` / ``inference_mode`` look up
  ``torch._C._set_grad_enabled`` / ``_InferenceMode`` at call time; ``set_grad_enabled(True)``
  enables grad in ``__init__``; no torch module binds those names at import.
* ``Tensor.backward`` -> ``torch.autograd.backward``; that and ``autograd.grad`` call the module
  global ``_engine_run_backward`` (from ``autograd.graph``) -> ``Variable._execution_engine``
  (type ``torch._C._EngineBase``, alias ``_ImperativeEngine``), whose only reference is that class
  attribute.
* ``torch.func`` is ``torch._functorch.{apis, eager_transforms}``; ``grad`` enters
  ``torch.enable_grad``; forward AD goes through ``torch._functorch.predispatch``, which bound
  ``torch._C._enter_dual_level`` at import (so that name is patched there too).
* TorchScript compiles ``torch.set_grad_enabled(True)`` and ``torch.autograd.grad`` into C++: a
  ``torch.jit.CompilationUnit`` function computes a gradient under ``no_grad`` and restores grad
  mode before returning. ``torch.ops.aten.{set_grad_enabled, backward, grad}`` are JIT operators
  callable from Python. Hence the compiler and operator bans.
* ``import torch`` loads ``torch.func``, ``torch.optim``, ``torch.export`` and ``torch.jit``, not
  ``torch._dynamo`` (lazy, via ``torch.__getattr__``).
* Deleting a single-phase extension such as ``torch._C`` from ``sys.modules`` and importing it again
  rebuilds it from the interpreter's saved copy of its original dict: re-imports are refused.
* ``torch.save`` to an ASCII path writes from C++ (``PyTorchFileWriter``); ``safetensors`` 0.8
  ``safe_open`` / ``serialize_file`` and ``tokenizers`` ``from_file`` / ``save`` do Rust file I/O;
  none raises an audit event.

How it works:

* **Attribute patches** replace the public and private names (``Tensor.backward``,
  ``torch.autograd.{grad, backward, _engine_run_backward}``, ``Variable._execution_engine`` by a
  proxy, ``torch._C._ImperativeEngine``, ``torch._C._set_grad_enabled`` (no-op for ``False``),
  ``torch._C._InferenceMode`` (``inference_mode(False)`` re-enables grad), ``_enter_dual_level``,
  the ``enable_grad`` / ``set_grad_enabled`` methods, ``Optimizer.__init__``, the functorch
  transforms, ``torch.compile``, the TorchScript compilers, the RNG and I/O names). Every
  replacement is fail-safe: it never performs the forbidden action, even if this module's globals
  are overwritten.
* **An audit hook** (``sys.addaudithook``, cannot be removed) enforces I/O, sockets, processes,
  threads, imports (the C import path), ``ctypes``, ``mmap``, ``sys.settrace`` / ``setprofile``,
  ``gc`` introspection, ``__code__`` assignment and sub-interpreters.
* **A ``sys.monitoring`` tool** refuses CALLs whose target is the original of any patched callable,
  a ``torch._C`` alias of one, a banned ``torch.ops`` / JIT operator, an RNG object's method, a
  file reader without an audit event, ``type.__subclasses__``, or an extension-module load from
  outside the install; PY_START on the original torch autograd / optimiser / compiler functions;
  and after every C call it checks that grad mode is still off (if not, the process exits with
  status 3 at once). The ``sys.monitoring`` mutators are then replaced, so the tool cannot be
  freed, re-pointed or switched off.
* **Irreversibility.** The hook and the monitor callbacks are closures over immutable values only
  (strings, tuples, frozensets of ids, C builtins) and contain no global-name lookups (a test checks
  their bytecode), so overwriting module globals or builtins cannot change them, and a frame that
  reaches them through a traceback exposes nothing writable. The originals are kept alive (for
  stable ids) by a callback registered for a monitoring event that is never enabled.
* **Violations.** Every refusal raises :class:`GuardViolation` (an ``ImportError`` or
  ``PermissionError`` subclass where callers expect those), is counted, and prints
  ``HN_GUARD: refused: ...`` at once to a private duplicate of the stderr that ``install`` found
  (so a later ``os.dup2`` or close of fd 2 cannot divert it). A swallowed violation still fails the
  process with status 3, however it ends: an ``atexit`` check, :func:`finish` (for launchers),
  ``os._exit`` and ``os.abort`` all exit with status 3 once any violation was recorded.
* **The exit and stderr paths.** ``os._exit`` / ``os.abort`` / ``os.close`` / ``socket.close`` are
  replaced by wrappers that hold no original; they ask a second audit hook, which never raises and
  never runs Python code, to do the work (so no frame on a traceback holds an exit or close
  function). ``os.dup2``, ``os.closerange`` and ``os.login_tty`` are refused. Closing or opening a
  file object or a ``kqueue`` on fd 2 or the private copy is refused; the monitor also converts a
  direct call of any ``posix._exit`` / ``abort`` (even of a re-created ``posix``) into status 3.
  ``_imp.create_builtin`` / ``create_dynamic`` of an already loaded module (a fresh ``atexit``
  whose ``_clear`` would drop the check, a fresh ``posix``) is refused.
* **Launchers** give the child's stderr as a pipe and treat the run as clean only when
  :func:`clean_run` holds: status 0 *and* no ``HN_GUARD:`` line on stderr.

Residual risks (a tripwire against moderately evasive code, not a sandbox; the audit's AST rules,
the ledger cross-check and the runner's rebuild check are the other half):

* C-level trampolines (``map``, ``functools.partial``, ``sorted(key=)`` ...) call C functions
  without a CALL event. They need an original to call, and the originals are reachable only through
  ``object.__subclasses__()`` or ``_imp.create_builtin`` (a fresh ``posix``, ``atexit``) itself
  called through a trampoline (direct calls are refused), and a graph needs grad mode, which is
  re-checked after every C call. A trampolined ``select.kqueue.fromfd`` (a C type, not patchable)
  can close the private stderr copy: the ``HN_GUARD`` lines are then lost, the status 3 is not.
* The markers are tamper-evident only on a pipe: on a regular-file stderr ``os.lseek`` (not audited)
  lets later writes overwrite them. Again the status 3 survives.
* A deliberately stateful ``__fspath__`` on a monkeypatched ``pathlib`` (the hook checks
  ``PurePath.__fspath__``/``__str__`` are the originals) or a ``dir_fd`` argument (directory
  descriptors cannot be opened) are the only known ways to make the hook check a different path
  from the one the OS uses; ``builtins.open`` converts path objects to ``str`` before the check.
* ``os.stat`` / ``os.path.exists`` are not audited by CPython: file *metadata* outside the allow-list
  stays visible. The interpreter's own teardown after ``atexit`` runs with the hook still active.
* KERNEL_CLIENT allows matmul, so a hand-written forward pass over loaded weights is not stopped
  here (doc §1: the audit and adjudicator handle that).
* ``torch.load`` of zip-format files is refused (its C++ reader takes paths); ``torch.save`` is
  redirected to the legacy format through the audited ``open``. ``shutil.rmtree`` (which opens
  directory descriptors) is refused.
* Cost: the CALL monitor makes pure-Python call loops about 15-20x slower; tensor operations,
  matmul and ``torch.linalg`` run at full speed. ``install`` takes about 20 ms.
"""

from __future__ import annotations

import _imp
import _io
import _posixsubprocess
import _random
import _socket
import atexit
import builtins
import errno
import hashlib
import importlib.machinery
import io
import itertools
import json
import operator
import os
import pathlib
import posix
import random
import socket
import sys
import threading
import types
from importlib.abc import MetaPathFinder
from typing import Any, Callable, Iterable, NoReturn, Sequence

import numpy as np
import numpy.linalg  # noqa: F401  (loaded so its names exist to patch)
import numpy.random  # noqa: F401
import torch
import torch.autograd.forward_ad  # noqa: F401
import torch.autograd.functional  # noqa: F401
import torch.serialization
from torch import nn

__all__ = [
    "KERNEL_CLIENT",
    "BUILD_EDIT",
    "MODES",
    "EXIT_VIOLATION",
    "VERIFIED_TORCH",
    "FORBIDDEN_IMPORTS",
    "RUNNER_PACKAGES",
    "GuardViolation",
    "ImportViolation",
    "IOViolation",
    "TrainingViolation",
    "install",
    "installed_mode",
    "violation_count",
    "finish",
    "clean_run",
    "control_normal",
]

KERNEL_CLIENT = "KERNEL_CLIENT"
BUILD_EDIT = "BUILD_EDIT"
MODES = (KERNEL_CLIENT, BUILD_EDIT)
EXIT_VIOLATION = 3
VERIFIED_TORCH = "2.14.1"  # every patched internal below was checked against this version's source

# §5 list first; then the same purpose (no autograd, optimisers or compilers) and interpreter escapes:
# readline reads files in C, sub-interpreters start without audit hooks, the _test* modules poke memory.
FORBIDDEN_IMPORTS: tuple[str, ...] = (
    "transformers", "hypaqwen", "peft", "accelerate", "agent_loop_lab", "sklearn",
    "torch._dynamo", "torch.compile", "torch.export", "torch.func", "torch.optim",
    "functorch", "torch._export", "torch._inductor", "torch.onnx", "scipy.optimize",
    "readline", "_xxsubinterpreters", "_xxinterpchannels", "_xxinterpqueues", "_interpreters",
    "_interpchannels", "_interpqueues", "_testcapi", "_testinternalcapi", "_testmultiphase",
    "_testsinglephase", "_testbuffer", "_testimportmultiple", "_testclinic", "_ctypes_test",
    "_tkinter", "tkinter", "_curses", "_curses_panel", "curses", "cffi", "_cffi_backend",
)
# BUILD_EDIT runs inside the harness's build runner (hypaqwen code): refused as new imports, not purged.
RUNNER_PACKAGES: tuple[str, ...] = ("hypaqwen", "agent_loop_lab")

RENDERER_RELPATH = "packet/renderer.py"  # KERNEL_CLIENT: may build its seeded random.Random (v0.3 rule)
BUILD_FILE = "build_edit.py"
FWD_DIR = "fwd"  # KERNEL_CLIENT: output/<phase>/fwd/ holds the ./fwd records; no guarded write there
CODE_SUFFIXES = (".py", ".pyc", ".so", ".dylib", ".pth")


class GuardViolation(RuntimeError):
    """An operation refused by the v0.4 guard."""


class ImportViolation(GuardViolation, ImportError):
    """A refused import (an ``ImportError`` so ``try: import x`` fallbacks keep working)."""


class IOViolation(GuardViolation, PermissionError):
    """A refused file, socket or process operation (a ``PermissionError``, errno EPERM)."""

    _EPERM = errno.EPERM  # a class attribute: the hook raises this class, so no global lookups here

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.errno, self.strerror = self._EPERM, msg


TrainingViolation = GuardViolation  # the v0.2 / v0.3 name, for launchers that catch it


# -- refusals from patched names (reachable, fail-safe) -------------------------------------------


def _refuse(what: str, kind: type[GuardViolation] = GuardViolation) -> NoReturn:
    """Record a violation through the audit hook (count + stderr) and raise."""
    sys.audit("hn_guard.violation", what)
    raise kind(f"{what} refused by the hn_guard v0.4 guard")


def _raiser(what: str, kind: type[GuardViolation] = GuardViolation) -> Callable[..., Any]:
    def blocked(*args: Any, **kwargs: Any) -> Any:
        _refuse(what, kind)

    blocked.__name__ = blocked.__qualname__ = "blocked_" + "".join(c if c.isalnum() else "_" for c in what)
    return blocked


def _blocked_class(what: str, kind: type[GuardViolation] = GuardViolation) -> type:
    def __new__(cls: type, *args: Any, **kwargs: Any) -> Any:  # noqa: N807
        _refuse(what, kind)

    return type("Blocked_" + "".join(c if c.isalnum() else "_" for c in what), (), {"__new__": __new__})


def _make_fd_wrappers() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """The ``os._exit`` / ``os.abort`` / ``os.close`` (and ``socket.close``) replacements. They hold no
    original, only ``sys.audit``: the actions hook (:func:`_make_actions_hook`) exits or closes, so
    a ``__closure__`` or a frame of theirs gives nothing that skips the checks."""
    audit, index_, int_t, oserror, strerror = sys.audit, operator.index, int, OSError, posix.strerror
    refuse, IOV = _hn_refuse, IOViolation

    def _exit(status: Any) -> None:
        audit("hn_guard.exit", status)  # exits: status 3 if any violation was recorded

    def abort() -> None:
        audit("hn_guard.abort")  # exits with status 3 if any violation was recorded, else aborts

    def close(fd: Any) -> None:
        fd = int_t(index_(fd))
        box: list[int] = []
        audit("hn_guard.close", fd, box)
        if box:
            e = box[0]
            if e == -1:
                refuse(audit, IOV, "closing the guard's stderr (fd 2 or its private copy)")
            raise oserror(e, strerror(e))

    return _exit, abort, close


_WHY_GRAD = "no autograd: weights are edited by reasoning, never fitted"
_WHY_RNG = "random numbers come only from hn_guard.control_normal"
_WHY_IO = "a file reader or writer without an audit event"
_WHY_BUILD = "no model, nn.Module or functional calls in build_edit.py"


def _guarded_set_grad_enabled_C(mode: Any) -> None:
    # Grad mode is already off for the life of the process: False needs no call, True is refused.
    if mode:
        _refuse(f"grad mode enabled (torch._C._set_grad_enabled(True)); {_WHY_GRAD}")


def _sge_init(self: Any, mode: Any) -> None:
    self.prev = False
    self.mode = False
    if mode:
        _refuse(f"torch.set_grad_enabled(True); {_WHY_GRAD}")


def _sge_enter(self: Any) -> None:
    if self.mode:
        _refuse(f"torch.set_grad_enabled(True) entered; {_WHY_GRAD}")


class _NoInferenceMode:
    """Stands in for ``torch._C._InferenceMode``: ``True`` is a no-op guard, ``False`` (which would
    switch grad mode on) is refused. A class, so torch's ``Optional[_InferenceMode]`` annotations work."""

    def __new__(cls, mode: bool = True) -> Any:
        if not mode:
            _refuse(f"inference_mode(False) (re-enables grad mode); {_WHY_GRAD}")
        return object.__new__(cls)

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class _NoEngine:
    """Stands in for ``Variable._execution_engine``; holds no reference to the real engine."""

    def __getattr__(self, name: str) -> Any:
        _refuse(f"autograd engine .{name}; {_WHY_GRAD}")


def _guarded_requires_grad_(base: Callable[..., Any]) -> Callable[..., Any]:
    def requires_grad_(self: Any, requires_grad: bool = True) -> Any:
        if requires_grad:
            _refuse(f"Tensor.requires_grad_(True); {_WHY_GRAD}")
        return base(self, False)

    return requires_grad_


def _guarded_open(orig: Callable[..., Any]) -> Callable[..., Any]:
    """``builtins.open`` / ``io.open``: convert a path object to ``str`` before the audited open, so the
    audit hook checks exactly the string the OS gets."""

    def open(file: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: A001
        if not isinstance(file, (str, bytes, int)):
            file = os.fspath(file)
        return orig(file, *args, **kwargs)

    return open


def _guarded_torch_save(orig: Callable[..., Any]) -> Callable[..., Any]:
    """``torch.save`` to a path writes from C++ (``PyTorchFileWriter``, no audit event). Save the legacy
    format through a Python file instead, which the audit hook checks."""

    def save(obj: Any, f: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs["_use_new_zipfile_serialization"] = False
        if isinstance(f, (str, bytes, os.PathLike)):
            with builtins.open(os.fspath(f), "wb") as fh:
                return orig(obj, fh, *args, **kwargs)
        return orig(obj, f, *args, **kwargs)

    return save


class _SafeSlice:
    def __init__(self, t: Any) -> None:
        self._t = t

    def get_shape(self) -> list[int]:
        return list(self._t.shape)

    def get_dtype(self) -> str:
        return str(self._t.dtype)

    def __getitem__(self, idx: Any) -> Any:
        return self._t[idx]


class _SafeOpen:
    """Pure-Python ``safetensors.safe_open``: reads the whole file with the audited ``open`` (the Rust
    ``safe_open`` opens and maps files with no audit event)."""

    def __init__(self, filename: Any, framework: str, device: Any = "cpu", backend: Any = None) -> None:
        import safetensors.numpy
        import safetensors.torch

        fw = str(framework)
        if fw not in ("pt", "torch", "np", "numpy"):
            raise ValueError(f"framework {fw!r} is not supported by the guarded safe_open")
        with builtins.open(filename, "rb") as fh:
            data = fh.read()
        n = int.from_bytes(data[:8], "little")
        header = json.loads(data[8 : 8 + n])
        self._meta = header.pop("__metadata__", None)
        self._order = [k for k, _ in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])]
        if fw in ("pt", "torch"):
            tensors = safetensors.torch.load(data)
            if str(device) != "cpu":
                tensors = {k: v.to(device) for k, v in tensors.items()}
        else:
            tensors = safetensors.numpy.load(data)
        self._tensors = tensors

    def __enter__(self) -> _SafeOpen:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def keys(self) -> list[str]:
        return sorted(self._tensors)

    def offset_keys(self) -> list[str]:
        return list(self._order)

    def metadata(self) -> dict[str, str] | None:
        return self._meta

    def get_tensor(self, name: str) -> Any:
        return self._tensors[name]

    def get_tensors(self) -> dict[str, Any]:
        return dict(self._tensors)

    def get_slice(self, name: str) -> _SafeSlice:
        return _SafeSlice(self._tensors[name])


def _guarded_save_file(serialize: Callable[..., bytes]) -> Callable[..., None]:
    def save_file(tensors: Any, filename: Any, metadata: Any = None) -> None:
        data = serialize(tensors, metadata)
        with builtins.open(filename, "wb") as fh:
            fh.write(data)

    return save_file


# -- the hash-derived RNG (the one allowed source of random numbers) --------------------------------

_CTRL_CHUNK = 1 << 20


def control_normal(seed: int, shape: Sequence[int] | torch.Size, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Standard-normal values from SHAKE-256 of ``seed`` (Box-Muller), bit-identical on every platform
    and torch version. The only source of random numbers under either guard mode (the client's
    ``control_direction`` builds on it)."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("control_normal: seed must be an int")
    dims = tuple(int(d) for d in shape)
    if any(d < 0 for d in dims):
        raise ValueError("control_normal: negative dimension")
    n = 1
    for d in dims:
        n *= d
    pairs = (n + 1) // 2
    out = np.empty(2 * pairs, dtype=np.float64)
    tag = f"hn_guard.control_normal.v1|{seed}|".encode()
    done = 0
    k = 0
    while done < pairs:
        m = min(_CTRL_CHUNK, pairs - done)
        raw = hashlib.shake_256(tag + str(k).encode()).digest(16 * m)
        u = (np.frombuffer(raw, dtype="<u8") >> np.uint64(11)).astype(np.float64) * (2.0**-53)
        u1 = 1.0 - u[0::2]  # (0, 1]
        r = np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u[1::2]
        out[2 * done : 2 * (done + m) : 2] = r * np.cos(theta)
        out[2 * done + 1 : 2 * (done + m) : 2] = r * np.sin(theta)
        done += m
        k += 1
    return torch.from_numpy(out[:n].reshape(dims).copy()).to(dtype)


# -- pure helpers for the hook and the monitor --------------------------------------------------------
#
# No global names and no closure: their behaviour depends only on their arguments, so overwriting this
# module's globals cannot change them (tests/test_wsp_guards.py checks the bytecode).


def _hn_realpath(path: str, cwd: str, lstat: Any, readlink: Any) -> str | None:
    """``posixpath.realpath(path, strict=False)`` from captured ``posix`` primitives; None if a
    symlink loop or an unreadable link makes it undecidable."""
    if not path.startswith("/"):
        path = cwd + "/" + path
    todo = path.split("/")
    todo.reverse()
    out: list[str] = []
    hops = 0
    while todo:
        name = todo.pop()
        if name == "" or name == ".":
            continue
        if name == "..":
            if out:
                out.pop()
            continue
        out.append(name)
        cand = "/" + "/".join(out)
        try:
            is_link = (lstat(cand).st_mode & 0o170000) == 0o120000
        except:  # noqa: E722 - missing component: the rest is taken literally (no global names here)
            is_link = False
        if not is_link:
            continue
        out.pop()
        hops = hops + 1
        if hops > 64:
            return None
        try:
            target = readlink(cand)
        except:  # noqa: E722
            return None
        if target.startswith("/"):
            out = []
        parts = target.split("/")
        parts.reverse()
        todo.extend(parts)
    return "/" + "/".join(out)


def _hn_under(path: str, roots: tuple[str, ...]) -> bool:
    for r in roots:
        if path == r or path.startswith(r + "/"):
            return True
    return False


def _hn_path(p: Any, t: Any) -> Any:
    """A path argument as ``str``; an ``int`` descriptor as itself; None if unsupported. ``t`` is the
    captured type table (see ``_type_table``)."""
    tp = t[0](p)
    if tp is t[1]:  # str
        return p
    if tp is t[3]:  # int (a descriptor)
        return p
    if tp is t[2]:  # bytes
        return p.decode("utf-8", "surrogateescape")
    if tp is t[4]:  # os.DirEntry: a C __fspath__
        return t[5](p)
    if tp is t[6] or tp is t[7]:  # PosixPath / PurePosixPath with pathlib's own __fspath__ / __str__
        pv = t[8]
        if pv.get("__fspath__") is t[9] and pv.get("__str__") is t[10] and tp.__fspath__ is t[9] and tp.__str__ is t[10]:
            r = t[5](p)
            if t[0](r) is t[1]:
                return r
        return None
    if t[1] in tp.__mro__:  # a str subclass: the OS uses its str value, never its __fspath__
        return t[11](p)
    return None


def _hn_refuse(audit: Any, kind: Any, what: str) -> None:
    audit("hn_guard.violation", what)
    raise kind(what + " refused by the hn_guard v0.4 guard")


def _hn_mro_ids(tp: Any, id_: Any) -> list[int]:
    out = []
    for k in tp.__mro__:
        out.append(id_(k))
    return out


def _type_table() -> tuple[Any, ...]:
    return (
        type, str, bytes, int, posix.DirEntry, posix.fspath, pathlib.PosixPath, pathlib.PurePosixPath,
        vars(pathlib.PurePath), vars(pathlib.PurePath)["__fspath__"], vars(pathlib.PurePath)["__str__"], str.__str__,
    )


# -- the audit hook ---------------------------------------------------------------------------------

_LIST_EVENTS = frozenset({"os.listdir", "os.scandir", "os.getxattr", "os.listxattr"})
# event -> positions of path arguments, and of dir_fd arguments (which must be None)
_FS_CHANGE = {
    "os.remove": ((0,), (1,)),
    "os.rmdir": ((0,), (1,)),
    "os.mkdir": ((0,), (2,)),
    "os.rename": ((0, 1), (2, 3)),
    "os.truncate": ((0,), ()),
    "os.chmod": ((0,), (2,)),
    "os.chown": ((0,), (3,)),
    "os.utime": ((0,), (3,)),
    "os.chflags": ((0,), ()),
    "os.setxattr": ((0,), ()),
    "os.removexattr": ((0,), ()),
}
_OS_ALLOWED = frozenset({"os.chdir", "os.putenv", "os.unsetenv", "os.walk", "os.fwalk", "os.lockf", "os.add_dll_directory"})
_ALWAYS_REFUSED = frozenset({
    "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork", "os.forkpty",
    "os.startfile", "os.kill", "os.killpg", "signal.pthread_kill", "signal.pidfd_send_signal", "pty.spawn",
    "resource.setrlimit", "resource.prlimit", "_thread.start_new_thread", "_thread.start_joinable_thread",
    "cpython.PyInterpreterState_New", "sys.settrace", "sys.setprofile", "gc.get_objects", "gc.get_referrers",
    "gc.get_referents", "mmap.__new__", "urllib.Request", "webbrowser.open", "sqlite3.connect",
    "sqlite3.connect/handle", "sqlite3.enable_load_extension", "sqlite3.load_extension", "dbm.open",
    "syslog.openlog", "syslog.syslog", "os.symlink", "os.link", "os.mkfifo", "os.mknod",
})


def _make_audit_hook(
    *, mode: str, read_roots: tuple[str, ...], read_exact: frozenset[str], code_roots: tuple[str, ...],
    write_roots: tuple[str, ...], endpoint: str | None, endpoint_is_dir: bool, forbid_paths: tuple[str, ...],
    forbidden: tuple[str, ...], snapshot: frozenset[str], modules: dict[str, Any], counter: Any, err_fd: int,
    protected: frozenset[int], nowrite_roots: tuple[str, ...] = (),
) -> tuple[Callable[[str, tuple[Any, ...]], None], Callable[[], None]]:
    """The audit hook and the atexit check, closures over immutable values and C builtins only.

    ``nowrite_roots`` (lower-cased realpaths) are carved out of ``write_roots``: no write or filesystem
    change under them, whatever the case of the path (a case-insensitive volume maps ``FWD`` to ``fwd``).

    Markers go to ``err_fd`` (a private duplicate of stderr); ``counter`` is shared with the actions
    hook and the monitor. The hook holds no exit or close function: a frame of it seen on a traceback
    gives nothing that ends the process without the status-3 check."""
    is_client = mode == KERNEL_CLIENT
    t = _type_table()
    type_, str_t, int_t, tuple_t, list_t = type, str, int, tuple, list
    repr_, len_ = repr, len
    realpath, under, coerce = _hn_realpath, _hn_under, _hn_path
    lstat, readlink, getcwd, write, audit = posix.lstat, posix.readlink, posix.getcwd, posix.write, sys.audit
    note = counter.__next__
    IOV, IMV, GV = IOViolation, ImportViolation, GuardViolation
    list_events, fs_change, os_allowed, refused = _LIST_EVENTS, _FS_CHANGE, _OS_ALLOWED, _ALWAYS_REFUSED
    code_sfx = CODE_SUFFIXES
    af_unix = socket.AF_UNIX
    w_flags = posix.O_WRONLY | posix.O_RDWR | posix.O_APPEND | posix.O_CREAT | posix.O_TRUNC
    label = "KERNEL_CLIENT" if is_client else "BUILD_EDIT"
    stdout, stderr = sys.__stdout__, sys.__stderr__
    lower = str.lower
    nowrite = tuple(lower(r) for r in nowrite_roots)

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if event == "hn_guard.violation":
            note()
            try:
                write(err_fd, ("HN_GUARD: refused: " + str_t(args[0])[:400] + "\n").encode("utf-8", "replace"))
            except:  # noqa: E722  (a closed or full stderr loses the line, never the status 3)
                pass
            return None
        msg = None
        exc = IOV
        if event == "open" or event in list_events:
            target = args[0] if args else None
            if target is None and event != "open":
                target = "."
            p = coerce(target, t)
            if type_(p) is int_t:
                if p in protected:
                    msg = "a file object on the guard's stderr (fd 2 or its private copy)"
                else:
                    return None  # an already open descriptor
            if msg is not None:
                pass
            elif p is None:
                msg = "file access through an unsupported path object " + repr_(type_(target))
            else:
                rp = realpath(p, getcwd(), lstat, readlink)
                mode_arg = args[1] if len_(args) > 1 else None
                flags = args[2] if len_(args) > 2 else 0
                is_write = False
                if event == "open":
                    if type_(mode_arg) is str_t:
                        is_write = ("w" in mode_arg) or ("a" in mode_arg) or ("x" in mode_arg) or ("+" in mode_arg)
                    if type_(flags) is int_t and flags & w_flags:
                        is_write = True
                kernel = endpoint is not None and rp is not None and (
                    rp == endpoint or (endpoint_is_dir and rp.startswith(endpoint + "/") and rp.count("/") == endpoint.count("/") + 1)
                )
                if rp is None:
                    msg = "file access to an undecidable path (symlink loop)"
                elif under(rp, forbid_paths):
                    msg = "reading the source of a forbidden module " + rp[:200]
                elif event == "open" and mode_arg is None and not kernel:
                    try:
                        is_dir = (lstat(rp).st_mode & 0o170000) == 0o040000
                    except:  # noqa: E722
                        is_dir = False
                    if is_dir:
                        msg = "opening a directory descriptor " + rp[:200]
                if msg is None:
                    if is_write:
                        if not (is_client and (kernel or under(rp, write_roots))):
                            msg = "write to " + rp[:200] + (" (writes go under output/<phase>/)" if is_client else " (no writes in build_edit.py)")
                        elif not kernel and under(lower(rp), nowrite):
                            msg = "write to " + rp[:200] + " (output/<phase>/fwd/ is written only by ./fwd)"
                    elif is_client:
                        if not (kernel or under(rp, read_roots)):
                            msg = "read of " + rp[:200] + " (outside the author directory and the interpreter install)"
                    elif not (rp in read_exact or (under(rp, code_roots) and (event != "open" or rp.endswith(code_sfx)))):
                        msg = "read of " + rp[:200] + " (build_edit.py reads only its constants file)"
        elif event in refused:
            msg = event + (" (no subprocesses, threads, tracing, gc introspection, mmap or links)")
        elif event in fs_change:
            if not is_client:
                msg = event + " (no filesystem changes in build_edit.py)"
            else:
                pos, fds = fs_change[event]
                for i in fds:  # CPython reports an unset dir_fd as -1
                    if len_(args) > i and args[i] is not None and args[i] != -1:
                        msg = event + " with dir_fd"
                for i in pos:
                    if msg is None and len_(args) > i:
                        p = coerce(args[i], t)
                        rp = realpath(p, getcwd(), lstat, readlink) if type_(p) is str_t else None
                        if rp is None or not under(rp, write_roots):
                            msg = event + " on " + (rp or repr_(type_(args[i])))[:200] + " (outside output/<phase>/)"
                        elif under(lower(rp), nowrite):
                            msg = event + " on " + rp[:200] + " (output/<phase>/fwd/ is written only by ./fwd)"
        elif event.startswith("socket."):
            if is_client and event == "socket.__new__" and len_(args) > 1 and args[1] == af_unix:
                return None
            if is_client and event == "socket.connect" and endpoint is not None and len_(args) > 1:
                p = coerce(args[1], t)
                rp = realpath(p, getcwd(), lstat, readlink) if type_(p) is str_t else None
                if rp is not None and (rp == endpoint or (endpoint_is_dir and rp.startswith(endpoint + "/"))):
                    return None
            msg = event + " (the only connection allowed is the kernel endpoint)"
        elif event.startswith("ctypes."):
            msg = event + " (no ctypes)"
        elif event.startswith("os."):
            if event in os_allowed:
                return None
            msg = event + " (not on the guard's list of allowed os operations)"
        elif event == "import":
            name = args[0] if args else ""
            if type_(name) is str_t:
                for pre in forbidden:
                    if name == pre or name.startswith(pre + "."):
                        msg = "import of forbidden module " + name
                if msg is None and name in snapshot and name not in modules:
                    msg = "re-import of " + name + " after it was removed from sys.modules"
            exc = IMV
        elif event == "object.__setattr__":
            if len_(args) > 1 and args[1] == "__code__":
                msg = "assignment to a function's __code__"
            exc = GV
        elif event == "hn_guard.count":
            if args and type_(args[0]) is list_t:
                r = repr_(counter)  # "count(N)": reading it does not advance it
                args[0].append(int_t(r[6:-1]))
            return None
        elif event == "hn_guard.mode":
            if args and type_(args[0]) is list_t:
                args[0].append(label)
            return None
        if msg is None:
            return None
        note()
        try:
            write(err_fd, ("HN_GUARD: refused: " + msg + "\n").encode("utf-8", "replace"))
        except:  # noqa: E722
            pass
        raise exc(msg + " refused by the hn_guard " + label + " guard")

    def at_exit() -> None:
        for s in (stdout, stderr):
            try:
                s.flush()
            except:  # noqa: E722
                pass
        audit("hn_guard.exit_check")  # the actions hook exits with status 3 if any violation was recorded

    return hook, at_exit


_ACTIONS = frozenset({"hn_guard.exit", "hn_guard.abort", "hn_guard.close", "hn_guard.exit_check", "hn_guard.fatal"})


def _make_actions_hook(
    *, counter: Any, err_fd: int, protected: frozenset[int], originals: tuple[Any, Any, Any],
) -> Callable[[str, tuple[Any, ...]], None]:
    """The second audit hook, the only holder of the original ``_exit`` / ``abort`` / ``close``.

    It never raises and calls only C functions on exact ints and bytes, so no frame of it is ever on a
    traceback or under attacker code. ``hn_guard.exit`` / ``abort`` / ``exit_check`` / ``fatal``: once
    any violation was recorded (or for ``fatal``) it writes a marker to ``err_fd`` and exits with
    status 3; otherwise it exits with the given status, aborts, or (``exit_check``) returns.
    ``hn_guard.close``: closes the descriptor unless it is protected, reporting into the given list.
    ``originals``: ``posix._exit``, ``posix.abort`` and ``posix.close`` taken before they were patched."""
    type_, int_t, list_t, len_, repr_ = type, int, list, len, repr
    write = posix.write
    exit_, abort_, close_ = originals
    acts, fatal, ebadf, status_v, oserror = _ACTIONS, _FATAL, errno.EBADF, EXIT_VIOLATION, OSError

    def actions(event: str, args: tuple[Any, ...]) -> None:
        if event not in acts:  # the event name is always an exact str (CPython builds it from C)
            return None
        if event == "hn_guard.close":
            if len_(args) == 2 and type_(args[1]) is list_t:
                fd = args[0]
                if type_(fd) is not int_t or fd < 0 or fd > 2147483647:
                    args[1].append(ebadf)
                elif fd in protected:
                    args[1].append(-1)
                else:
                    try:
                        close_(fd)
                    except oserror as e:
                        en = e.errno
                        args[1].append(en if type_(en) is int_t else ebadf)
            return None
        r = repr_(counter)  # "count(N)"
        if event == "hn_guard.fatal":
            msg = fatal
        elif r != "count(0)":
            msg = ("HN_GUARD: " + r[6:-1] + " violation(s) during this run; exit status 3\n").encode()
        elif event == "hn_guard.abort":
            abort_()
            return None
        elif event == "hn_guard.exit":
            code = args[0] if len_(args) == 1 else 1
            if type_(code) is not int_t or code < -2147483648 or code > 2147483647:
                code = 1
            exit_(code)
            return None
        else:
            return None  # exit_check with no violation: the interpreter finishes normally
        try:
            write(err_fd, msg)
        except:  # noqa: E722
            pass
        exit_(status_v)
        return None

    return actions


# -- the sys.monitoring tool -------------------------------------------------------------------------

_MON_MUTATORS = ("use_tool_id", "free_tool_id", "register_callback", "set_events", "set_local_events", "restart_events")
_TOOL_PREFERENCE = (3, 4, 5, 2, 1, 0)
_FATAL = b"HN_GUARD: grad mode was found enabled (a path the patches do not cover); exiting with status 3\n"
# Builtin (C) callables refused by name wherever they are reached (v0.3 IO list + process/subclass escapes).
BANNED_BUILTIN_NAMES: frozenset[str] = frozenset({
    "from_file", "import_ir_module", "_load_for_lite_interpreter", "_load_mobile_module_from_file",
    "_load_jit_module_from_file", "_get_model_bytecode_version", "_get_mobile_model_contained_types",
    "_get_model_ops_and_info", "_backport_for_mobile", "_load_mobile_module_from_bytes_file",
    "_import_ir_module_from_package", "fork_exec", "__subclasses__", "mkfifo", "mknod",
    "dup2", "closerange", "login_tty",  # could divert or close the guard's stderr
})
EXIT_NAMES: frozenset[str] = frozenset({"_exit", "abort"})  # posix functions that end the process without atexit
FD_NAMES: frozenset[str] = frozenset({"close", "fromfd"})  # module / class functions that close or adopt a descriptor
TOKENIZERS_IO_NAMES: frozenset[str] = frozenset({"from_file", "read_file", "save", "from_pretrained", "train"})
# torch.ops / JIT operator base names refused in both modes (grad mode, autograd, forward AD, lstsq / pinv, RNG).
OP_BANNED: frozenset[str] = frozenset({
    "set_grad_enabled", "backward", "grad", "_make_dual", "_unpack_dual", "_fw_primal", "linalg_lstsq",
    "linalg_pinv", "pinverse", "lstsq", "from_file", "rand", "randn", "randint", "randperm", "rand_like",
    "randn_like", "randint_like", "normal", "normal_", "bernoulli", "bernoulli_", "multinomial", "poisson",
    "binomial", "_standard_gamma", "_sample_dirichlet", "uniform", "uniform_", "random", "random_",
    "exponential", "exponential_", "cauchy", "cauchy_", "log_normal", "log_normal_", "geometric",
    "geometric_", "dropout", "dropout_", "native_dropout", "feature_dropout", "alpha_dropout",
    "feature_alpha_dropout", "rrelu", "rrelu_", "rrelu_with_noise",
})
OP_BANNED_BUILD: frozenset[str] = frozenset({"softmax", "log_softmax", "_softmax", "_log_softmax", "linear"})


def _make_monitor(
    *, banned: frozenset[int], sge_id: int, inference_id: int, op_banned: frozenset[str], op_types: tuple[type, ...],
    instance_types: frozenset[int], script_types: frozenset[int], script_names: frozenset[str],
    rng_ids: frozenset[int], rng_py_ids: frozenset[int], rng_c_types: frozenset[int], rng_ok: frozenset[str],
    np_rng_types: frozenset[int], tok_types: frozenset[int], mon_ns: Any, imp_mod: Any, snapshot: frozenset[str],
    modules: dict[str, Any], forbidden: tuple[str, ...], ext_roots: tuple[str, ...], forbid_paths: tuple[str, ...],
    spec_type: type, counter: Any, protected: frozenset[int],
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """CALL / C_RETURN / PY_START callbacks, closures over immutable values and C builtins only.

    They hold no exit function: grad mode found on, or a direct ``_exit`` / ``abort`` after a
    violation, is ended by the actions hook (``hn_guard.fatal`` / ``hn_guard.exit``)."""
    is_grad = torch._C.is_grad_enabled
    audit = sys.audit
    id_, type_, str_t, vars_, getattr_, int_t, repr_ = id, type, str, vars, getattr, int, repr
    builtin_t, method_t, missing = types.BuiltinFunctionType, types.MethodType, sys.monitoring.MISSING
    module_t = types.ModuleType
    GV, IOV = GuardViolation, IOViolation
    exit_names, fd_names, status_v = EXIT_NAMES, FD_NAMES, EXIT_VIOLATION
    mutators = frozenset(_MON_MUTATORS)
    banned_names, tok_names = BANNED_BUILTIN_NAMES, TOKENIZERS_IO_NAMES
    realpath, under, refuse, mro_ids = _hn_realpath, _hn_under, _hn_refuse, _hn_mro_ids
    lstat, readlink, getcwd = posix.lstat, posix.readlink, posix.getcwd
    rng_callers = rng_ok | {random.__file__}
    no_rng = " (random numbers come only from hn_guard.control_normal)"

    def on_call(code: Any, offset: int, c: Any, arg0: Any) -> None:
        if is_grad():
            audit("hn_guard.fatal")  # exits with status 3
        i = id_(c)
        if i in banned:
            if i == sge_id:
                if arg0 is missing or arg0:
                    refuse(audit, GV, "grad mode enabled through an original torch._C._set_grad_enabled reference")
                return None
            if i == inference_id:
                if arg0 is not missing and not arg0:
                    refuse(audit, GV, "inference_mode(False) through an original torch._C._InferenceMode reference")
                return None
            nm = getattr_(c, "__qualname__", None) or getattr_(c, "__name__", "?")
            refuse(audit, GV, "call of a banned original (" + str_t(nm)[:120] + ")")
        tc = type_(c)
        if tc is method_t:  # a bound Python function, or a pybind11 instancemethod bound to an object
            f = c.__func__
            if id_(f) in banned:
                refuse(audit, GV, "call of a banned original method")
            if id_(f) in rng_ids:
                refuse(audit, GV, "numpy random generator method" + no_rng)
            if id_(f) in rng_py_ids and code.co_filename not in rng_callers:
                refuse(audit, GV, "random.Random method" + no_rng)
            name = str_t(getattr_(f, "__name__", ""))
            s = c.__self__
            for k in mro_ids(type_(s), id_):
                if k in instance_types:
                    refuse(audit, GV, "method " + name + " of a banned engine, file reader or compiler object")
                if k in script_types and name in script_names:
                    refuse(audit, GV, "TorchScript " + name)
                if k in np_rng_types:
                    refuse(audit, GV, "numpy random generator method " + name + no_rng)
                if k in tok_types and name in tok_names:
                    refuse(audit, GV, "tokenizers file I/O " + name)
            return None
        if tc is builtin_t:
            name = c.__name__
            s = c.__self__
            if name in banned_names:
                refuse(audit, GV, "call of " + name + " (a file reader, process launcher or subclass enumeration)")
            if s is mon_ns and name in mutators:
                refuse(audit, GV, "sys.monitoring." + name)
            st = type_(s)
            if st is module_t and name in exit_names and repr_(counter) != "count(0)":
                audit("hn_guard.exit", status_v)  # any posix._exit / abort, even of a re-created posix
            if (st is module_t or st is type_) and name in fd_names and type_(arg0) is int_t and arg0 in protected:
                refuse(audit, IOV, name + " on the guard's stderr (fd 2 or its private copy)")
            ids = mro_ids(s, id_) if st is type_ else mro_ids(st, id_)
            for k in ids:
                if k in instance_types:
                    refuse(audit, GV, "method " + name + " of a banned engine, file reader or compiler object")
                if k in script_types and name in script_names:
                    refuse(audit, GV, "TorchScript " + name)
                if k in np_rng_types:
                    refuse(audit, GV, "numpy random generator method " + name + no_rng)
                if k in tok_types and name in tok_names:
                    refuse(audit, GV, "tokenizers file I/O " + name)
                if k in rng_c_types and code.co_filename not in rng_callers:
                    refuse(audit, GV, "random.Random method " + name + no_rng)
            if name in op_banned and st.__name__.startswith("pybind11"):
                refuse(audit, GV, "JIT operator " + name)
            if s is imp_mod and (name == "create_dynamic" or name == "create_builtin"):
                if type_(arg0) is not spec_type or "name" in vars_(spec_type) or "origin" in vars_(spec_type):
                    refuse(audit, GV, "_imp." + name + " with a non-standard spec")
                d = vars_(arg0)
                mod = d.get("name")
                if type_(mod) is not str_t:
                    refuse(audit, GV, "_imp." + name + " without a module name")
                for pre in forbidden:
                    if mod == pre or mod.startswith(pre + "."):
                        refuse(audit, GV, "loading the forbidden module " + mod)
                if mod in snapshot and mod not in modules:
                    refuse(audit, GV, "re-creating " + mod + " after it was removed from sys.modules")
                if mod in modules:
                    refuse(audit, GV, "a second instance of the loaded module " + mod)
                if name == "create_dynamic":
                    origin = d.get("origin")
                    rp = realpath(origin, getcwd(), lstat, readlink) if type_(origin) is str_t else None
                    if rp is None or not under(rp, ext_roots) or under(rp, forbid_paths):
                        refuse(audit, GV, "loading an extension module from outside the interpreter install")
            return None
        if i in rng_ids:
            refuse(audit, GV, "numpy random generator method" + no_rng)
        if i in rng_py_ids and code.co_filename not in rng_callers:
            refuse(audit, GV, "random.Random method" + no_rng)
        if tc is type_:
            for k in mro_ids(c, id_):
                if k in rng_c_types and code.co_filename not in rng_ok:
                    refuse(audit, GV, "random.Random construction" + no_rng)
            return None
        if tc in op_types:
            q = getattr_(c, "_qualified_op_name", None)
            if q is None:
                q = getattr_(getattr_(c, "_overloadpacket", None), "_qualified_op_name", None) or getattr_(c, "_name", "")
            base = str_t(q).split("::")[-1].split(".")[0]
            if base in op_banned:
                refuse(audit, GV, "torch.ops operator " + base)
        return None

    def on_c_return(code: Any, offset: int, c: Any, arg0: Any) -> None:
        if is_grad():
            audit("hn_guard.fatal")  # exits with status 3
        return None

    def on_py_start(code: Any, offset: int) -> None:
        refuse(audit, GV, "execution of the original " + str_t(code.co_qualname)[:120])

    return on_call, on_c_return, on_py_start


# -- patch targets ---------------------------------------------------------------------------------

RANDOM_MODULE_FUNCS: tuple[str, ...] = (
    "random", "uniform", "randint", "randrange", "choice", "choices", "shuffle", "sample", "seed",
    "gauss", "normalvariate", "lognormvariate", "expovariate", "vonmisesvariate", "gammavariate",
    "betavariate", "paretovariate", "weibullvariate", "triangular", "getrandbits", "randbytes",
    "binomialvariate", "setstate",
)
TORCH_RNG_FUNCS: tuple[str, ...] = (
    "rand", "randn", "randint", "randperm", "rand_like", "randn_like", "randint_like", "normal",
    "bernoulli", "multinomial", "poisson", "binomial", "_standard_gamma", "_sample_dirichlet", "manual_seed",
    "seed", "set_rng_state", "dropout", "alpha_dropout", "feature_alpha_dropout", "feature_dropout", "rrelu",
    "rrelu_", "native_dropout",
)
TORCH_RANDOM_MODULE_FUNCS: tuple[str, ...] = ("manual_seed", "seed", "set_rng_state", "fork_rng")
TENSOR_RNG_METHODS: tuple[str, ...] = (
    "normal_", "uniform_", "random_", "bernoulli_", "exponential_", "cauchy_", "log_normal_",
    "geometric_", "bernoulli", "multinomial",
)
FUNCTIONAL_RNG: tuple[str, ...] = (
    "dropout", "dropout1d", "dropout2d", "dropout3d", "alpha_dropout", "feature_alpha_dropout",
    "rrelu", "rrelu_", "gumbel_softmax",
)
LSTSQ_PINV_NAMES: tuple[str, ...] = ("linalg_lstsq", "linalg_pinv", "pinverse")
FUNCTORCH_APIS: tuple[str, ...] = ("grad", "grad_and_value", "vmap")
FUNCTORCH_EAGER: tuple[str, ...] = (
    "vjp", "jvp", "jacrev", "jacfwd", "hessian", "linearize", "grad_and_value_impl", "grad_impl",
    "_autograd_grad", "_vjp_with_argnums", "_jvp_with_argnums",
)
TORCH_FUNC_NAMES: tuple[str, ...] = ("grad", "grad_and_value", "vjp", "jvp", "jacrev", "jacfwd", "hessian", "linearize", "vmap")
AUTOGRAD_FUNCTIONAL: tuple[str, ...] = ("jacobian", "hessian", "vjp", "jvp", "hvp", "vhp")
JIT_PY_NAMES: tuple[str, ...] = (
    "script", "trace", "trace_module", "fork", "_fork", "freeze", "optimize_for_inference", "CompilationUnit",
    "load", "save",
)
JIT_C_NAMES: tuple[str, ...] = (
    "_jit_script_compile", "_jit_script_class_compile", "_jit_script_compile_overload", "_jit_script_interface_compile",
    "_create_function_from_trace", "_create_function_from_trace_with_dict", "_create_module_with_type",
    "_jit_trace_graph", "_jit_trace_module", "fork", "CompilationUnit",
)
SCRIPT_MODULE_NAMES: frozenset[str] = frozenset({
    "_define", "_create_method_from_trace", "_create_method_from_trace_with_dict", "save", "_save_for_mobile",
    "save_to_buffer", "_save_to_buffer_for_mobile",
})
PROFILER_C_NAMES: tuple[str, ...] = ("_enable_profiler", "_enable_profiler_legacy", "_prepare_profiler")

_HELPER_MODULES = ("typing", "builtins", "torch._jit_internal", "torch.overrides", "torch._C")
_BUILTIN_VALUES = tuple(vars(builtins).values())


def _public_callables(mod: Any, classes: bool = False) -> list[str]:
    """Public callables defined by ``mod``'s own library (v0.3 rule): no re-exported typing helpers,
    builtins, torch-function dispatch helpers or, unless ``classes``, classes."""
    out = []
    for n in sorted(vars(mod)):
        if n.startswith("_"):
            continue
        v = vars(mod)[n]
        if isinstance(v, types.ModuleType) or not callable(v):
            continue
        if getattr(v, "__module__", None) in _HELPER_MODULES or any(v is b for b in _BUILTIN_VALUES):
            continue
        if isinstance(v, type) and not classes:
            continue
        out.append(n)
    return out


# (owner, attribute, replacement, ban the original by identity)
_Target = tuple[Any, str, Any, bool]


def _rng_targets(mode: str) -> list[_Target]:
    t: list[_Target] = []
    for n in RANDOM_MODULE_FUNCS:
        if hasattr(random, n):
            t.append((random, n, _raiser(f"random.{n}; {_WHY_RNG}"), False))  # bound to _inst: the monitor rule covers it
    for n in TORCH_RNG_FUNCS:
        if hasattr(torch, n):
            t.append((torch, n, _raiser(f"torch.{n}; {_WHY_RNG}"), True))
    t.append((torch, "Generator", _blocked_class(f"torch.Generator; {_WHY_RNG}"), True))
    for n in TORCH_RANDOM_MODULE_FUNCS:
        if hasattr(torch.random, n):
            t.append((torch.random, n, _raiser(f"torch.random.{n}; {_WHY_RNG}"), True))
    for n in TENSOR_RNG_METHODS:
        if hasattr(torch.Tensor, n):
            t.append((torch.Tensor, n, _raiser(f"Tensor.{n}; {_WHY_RNG}"), True))
    for n in _public_callables(np.random, classes=True):
        v = vars(np.random)[n]
        rep = _blocked_class(f"numpy.random.{n}; {_WHY_RNG}") if isinstance(v, type) else _raiser(f"numpy.random.{n}; {_WHY_RNG}")
        t.append((np.random, n, rep, True))
    t.append((os, "urandom", _raiser(f"os.urandom; {_WHY_RNG}"), True))
    if mode == KERNEL_CLIENT:
        for n in FUNCTIONAL_RNG:
            if hasattr(nn.functional, n):
                t.append((nn.functional, n, _raiser(f"torch.nn.functional.{n}; {_WHY_RNG}"), True))
    return t


def _grad_targets() -> list[_Target]:
    from torch.autograd import grad_mode
    from torch.autograd import graph as autograd_graph
    from torch.autograd import variable as variable_module

    t: list[_Target] = [
        (torch._C, "_set_grad_enabled", _guarded_set_grad_enabled_C, True),
        (torch._C, "_InferenceMode", _NoInferenceMode, True),
        (torch._C, "_enter_dual_level", _raiser(f"forward-mode AD (torch._C._enter_dual_level); {_WHY_GRAD}"), True),
        (grad_mode.set_grad_enabled, "__init__", _sge_init, True),
        (grad_mode.set_grad_enabled, "__enter__", _sge_enter, True),
        (grad_mode.enable_grad, "__enter__", _raiser(f"torch.enable_grad(); {_WHY_GRAD}"), True),
        (torch.Tensor, "backward", _raiser(f"Tensor.backward; {_WHY_GRAD}"), True),
        (torch.Tensor, "requires_grad_", _guarded_requires_grad_(torch._C.TensorBase.requires_grad_), False),
        (torch.autograd, "backward", _raiser(f"torch.autograd.backward; {_WHY_GRAD}"), True),
        (torch.autograd, "grad", _raiser(f"torch.autograd.grad; {_WHY_GRAD}"), True),
        (torch.autograd, "_engine_run_backward", _raiser(f"the autograd engine; {_WHY_GRAD}"), True),
        (autograd_graph, "_engine_run_backward", _raiser(f"the autograd engine; {_WHY_GRAD}"), True),
        (variable_module.Variable, "_execution_engine", _NoEngine(), False),
        (torch._C, "_ImperativeEngine", _blocked_class(f"a new autograd engine; {_WHY_GRAD}"), True),
        (variable_module, "ImperativeEngine", _blocked_class(f"a new autograd engine; {_WHY_GRAD}"), True),
    ]
    for n in AUTOGRAD_FUNCTIONAL:
        t.append((torch.autograd.functional, n, _raiser(f"torch.autograd.functional.{n}; {_WHY_GRAD}"), True))
    # 2.14 routes forward AD through torch._functorch.predispatch, which bound torch._C._enter_dual_level at import.
    dual = _raiser(f"forward-mode AD (enter_dual_level); {_WHY_GRAD}")
    t.append((torch.autograd.forward_ad, "enter_dual_level", dual, True))
    pre = sys.modules.get("torch._functorch.predispatch")
    if pre is not None:
        for n in ("_enter_dual_level", "_enter_dual_level_impl"):
            if n in vars(pre):
                t.append((pre, n, dual, True))
    # torch.vmap is torch.func.vmap (§5 refuses torch.func); in 2.14 it also goes through the export pre-dispatch path.
    vmap = _raiser("torch.vmap (torch.func.vmap; torch.func is refused)")
    t.append((torch, "vmap", vmap, True))
    apis = sys.modules.get("torch._functorch.apis")
    eager = sys.modules.get("torch._functorch.eager_transforms")
    func = sys.modules.get("torch.func")
    for mod, names, label in ((apis, FUNCTORCH_APIS, "torch.func"), (eager, FUNCTORCH_EAGER, "torch.func"), (func, TORCH_FUNC_NAMES, "torch.func")):
        if mod is not None:
            for n in names:
                if n in vars(mod):
                    t.append((mod, n, _raiser(f"{label}.{n} (a functional autograd transform); {_WHY_GRAD}"), True))
    optim = sys.modules.get("torch.optim.optimizer")
    if optim is not None:
        t.append((optim.Optimizer, "__init__", _raiser(f"a torch.optim optimiser; {_WHY_GRAD}"), True))
    for n in PROFILER_C_NAMES:
        if hasattr(torch._C._autograd, n):
            t.append((torch._C._autograd, n, _raiser(f"the autograd profiler (writes trace files from C++); {_WHY_IO}"), True))
    return t


def _compiler_targets() -> list[_Target]:
    why = "no compilers: TorchScript and torch.compile can run autograd outside Python"
    t: list[_Target] = [(torch, "compile", _raiser(f"torch.compile; {why}"), True)]
    export = sys.modules.get("torch.export")
    if export is not None and hasattr(export, "export"):
        t.append((export, "export", _raiser(f"torch.export.export; {why}"), True))
    for mod_name in ("torch.jit", "torch.jit._script", "torch.jit._trace", "torch.jit._async", "torch.jit._freeze", "torch.jit._serialization"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for n in JIT_PY_NAMES:
            if n in vars(mod):
                v = vars(mod)[n]
                rep = _blocked_class(f"{mod_name}.{n}; {why}") if isinstance(v, type) else _raiser(f"{mod_name}.{n}; {why}")
                t.append((mod, n, rep, True))
    for n in JIT_C_NAMES:
        if hasattr(torch._C, n):
            v = getattr(torch._C, n)
            rep = _blocked_class(f"torch._C.{n}; {why}") if isinstance(v, type) else _raiser(f"torch._C.{n}; {why}")
            t.append((torch._C, n, rep, True))
    return t


def _io_targets() -> list[_Target]:
    t: list[_Target] = [
        (builtins, "open", _guarded_open(_io.open), False),
        (io, "open", _guarded_open(_io.open), False),
        (torch, "from_file", _raiser(f"torch.from_file; {_WHY_IO}", IOViolation), True),
        (torch, "save", _guarded_torch_save(torch.serialization.save), False),
        (torch.serialization, "save", _guarded_torch_save(torch.serialization.save), False),
        (torch._C, "PyTorchFileReader", _blocked_class(f"torch._C.PyTorchFileReader (zip-format torch.load); {_WHY_IO}", IOViolation), True),
        (torch._C, "PyTorchFileWriter", _blocked_class(f"torch._C.PyTorchFileWriter; {_WHY_IO}", IOViolation), True),
        (_posixsubprocess, "fork_exec", _raiser("process creation (_posixsubprocess.fork_exec)", IOViolation), True),
        (os, "mkfifo", _raiser("os.mkfifo", IOViolation), True),
        (os, "mknod", _raiser("os.mknod", IOViolation), True),
        (atexit, "_clear", _raiser("atexit._clear (the guard's exit check)"), True),
        (atexit, "unregister", _raiser("atexit.unregister (the guard's exit check)"), True),
        (atexit, "_run_exitfuncs", _raiser("atexit._run_exitfuncs"), True),
    ]
    # the exit status and the stderr markers: wrappers that hold no original (not banned: the actions
    # hook calls the originals; the monitor checks direct calls of them by name), and refused dup2s
    ex, ab, cl = _make_fd_wrappers()
    for owner in (posix, os):
        t += [(owner, "_exit", ex, False), (owner, "abort", ab, False), (owner, "close", cl, False)]
        for n in ("dup2", "closerange", "login_tty"):
            if hasattr(owner, n):
                t.append((owner, n, _raiser(f"os.{n} (it could divert or close the guard's stderr)", IOViolation), True))
    for owner in (_socket, socket):
        if hasattr(owner, "close"):
            t.append((owner, "close", cl, False))
    for cls in (torch.UntypedStorage, torch.TypedStorage):
        if hasattr(cls, "from_file"):
            t.append((cls, "from_file", _raiser(f"{cls.__name__}.from_file; {_WHY_IO}", IOViolation), True))
    for n in sorted(BANNED_BUILTIN_NAMES):
        if n not in ("from_file", "fork_exec", "__subclasses__", "mkfifo", "mknod") and hasattr(torch._C, n):
            t.append((torch._C, n, _raiser(f"torch._C.{n}; {_WHY_IO}", IOViolation), True))
    st = sys.modules.get("safetensors")
    if st is not None:
        import safetensors.numpy
        import safetensors.torch

        for mod in (st, st._safetensors_rust, safetensors.torch, safetensors.numpy):
            if "safe_open" in vars(mod):
                t.append((mod, "safe_open", _SafeOpen, True))
            if "serialize_file" in vars(mod):
                t.append((mod, "serialize_file", _raiser(f"safetensors serialize_file; {_WHY_IO}", IOViolation), True))
        t.append((safetensors.torch, "save_file", _guarded_save_file(safetensors.torch.save), True))
        t.append((safetensors.numpy, "save_file", _guarded_save_file(safetensors.numpy.save), True))
    for mod_name in [m for m in list(sys.modules) if m == "tokenizers" or m.startswith("tokenizers.")]:
        mod = sys.modules[mod_name]
        for v in list(vars(mod).values()):
            if isinstance(v, type) and str(getattr(v, "__module__", "")).startswith("tokenizers"):
                for n in TOKENIZERS_IO_NAMES:
                    if n in vars(v):
                        t.append((v, n, _raiser(f"tokenizers {v.__name__}.{n}; {_WHY_IO}", IOViolation), True))
    return t


def _lstsq_targets() -> list[_Target]:
    why = "lstsq / pinv fit weights in closed form"
    t: list[_Target] = [
        (torch.linalg, "lstsq", _raiser(f"torch.linalg.lstsq; {why}"), True),
        (torch.linalg, "pinv", _raiser(f"torch.linalg.pinv; {why}"), True),
        (torch, "pinverse", _raiser(f"torch.pinverse; {why}"), True),
        (torch.Tensor, "pinverse", _raiser(f"Tensor.pinverse; {why}"), True),
        (np.linalg, "lstsq", _raiser(f"numpy.linalg.lstsq; {why}"), True),
        (np.linalg, "pinv", _raiser(f"numpy.linalg.pinv; {why}"), True),
    ]
    return t


def _build_targets() -> list[_Target]:
    t: list[_Target] = []
    F = nn.functional
    for n in _public_callables(F):
        t.append((F, n, _raiser(f"torch.nn.functional.{n}; {_WHY_BUILD}"), True))
    for n in ("softmax", "log_softmax"):
        t.append((torch, n, _raiser(f"torch.{n}; {_WHY_BUILD}"), True))
        t.append((torch.Tensor, n, _raiser(f"Tensor.{n}; {_WHY_BUILD}"), True))
        if hasattr(torch.special, n):
            t.append((torch.special, n, _raiser(f"torch.special.{n}; {_WHY_BUILD}"), True))
    for n in ("__call__", "_call_impl", "_wrapped_call_impl"):
        if n in vars(nn.Module):
            t.append((nn.Module, n, _raiser(f"calling an nn.Module ({n}); {_WHY_BUILD}"), True))
    return t


def _targets(mode: str) -> list[_Target]:
    t = _grad_targets() + _compiler_targets() + _io_targets() + _lstsq_targets() + _rng_targets(mode)
    if mode == BUILD_EDIT:
        t += _build_targets()
    seen: set[tuple[int, str]] = set()
    out = []
    for owner, name, rep, ban in t:  # first wins
        if (id(owner), name) not in seen:
            seen.add((id(owner), name))
            out.append((owner, name, rep, ban))
    return out


def _callables_of(values: Iterable[Any]) -> list[Any]:
    """Callables among class-dict values, each with its unwrapped function (``staticmethod``,
    ``classmethod`` and pybind11's ``instancemethod``, whose bound form calls the inner builtin)."""
    out = []
    for v in values:
        inner = getattr(v, "__func__", None) if not isinstance(v, types.MethodType) else None
        for x in (v, inner):
            if x is not None and callable(x):
                out.append(x)
    return out


def _alias_originals(names: Iterable[str]) -> list[Any]:
    """The ``torch._C`` aliases of banned names (functions and ``TensorBase`` method descriptors)."""
    owners = [torch._C._VariableFunctions, torch._C._nn, torch._C._linalg, torch._C._special, torch._C.TensorBase]
    out = []
    for owner in owners:
        for n in names:
            v = vars(owner).get(n) if isinstance(owner, type) else getattr(owner, n, None)
            if v is not None and callable(v):
                out.append(v)
    return out


# -- install --------------------------------------------------------------------------------------


def installed_mode() -> str | None:
    """The mode this process was put under, or None (asked of the audit hook, not of a module global)."""
    box: list[str] = []
    sys.audit("hn_guard.mode", box)
    return box[0] if box else None


def violation_count() -> int:
    """Violations recorded so far (0 when no guard is installed). The count cannot be lowered."""
    box: list[int] = []
    sys.audit("hn_guard.count", box)
    return box[0] if box else 0


def finish(code: int = 0) -> NoReturn:
    """End the process without interpreter teardown: flush stdio, then exit with ``code``, or with
    :data:`EXIT_VIOLATION` if any violation was recorded (even one the script caught). Under the
    guard ``os._exit`` is the guard's wrapper, which enforces the same rule on any caller."""
    for s in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        try:
            s.flush()
        except Exception:  # noqa: BLE001
            pass
    os._exit(EXIT_VIOLATION if violation_count() else int(code))


def clean_run(returncode: int, stderr: str | bytes) -> bool:
    """For launchers of a guarded child (stderr given as a pipe): the run was clean only if it exited
    with status 0 *and* printed no ``HN_GUARD:`` line. Either signal alone means a violation."""
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, (bytes, bytearray)) else str(stderr)
    return returncode == 0 and "HN_GUARD:" not in text


_FD_TARGET_NAMES = frozenset({"_exit", "abort", "close", "dup2", "closerange", "login_tty"})


def _swap_module_aliases(swaps: dict[int, tuple[Any, Any]]) -> None:
    """Other modules' own references to the replaced exit / close / dup2 functions (``pty`` does
    ``from os import close``) get the replacement too, so no loaded module keeps an original."""
    for mod in list(sys.modules.values()):
        d = getattr(mod, "__dict__", None) if isinstance(mod, types.ModuleType) else None
        if type(d) is not dict:
            continue
        for k, v in list(d.items()):
            hit = swaps.get(id(v))
            if hit is not None and hit[0] is v:
                d[k] = hit[1]


def _realdir(p: str | os.PathLike[str], what: str) -> str:
    r = os.path.realpath(os.fspath(p))
    if not os.path.isdir(r):
        raise ValueError(f"{what} {p!r} is not a directory")
    if r == "/":
        raise ValueError(f"{what} may not be the filesystem root")
    return r


def _install_roots() -> tuple[str, ...]:
    out = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix) if p}
    out.discard("/")
    return tuple(sorted(out))


def _matches(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _module_locations(names: Sequence[str]) -> tuple[str, ...]:
    """Source locations (package dirs or module files) of the forbidden modules that are installed."""
    out: set[str] = set()
    for name in names:
        parent, _, leaf = name.rpartition(".")
        try:
            if parent:
                pmod = sys.modules.get(parent)
                if pmod is None or not hasattr(pmod, "__path__"):
                    continue
                spec = importlib.machinery.PathFinder.find_spec(name, list(pmod.__path__))
            else:
                spec = importlib.machinery.PathFinder.find_spec(name)
        except Exception:  # noqa: BLE001 - an odd finder never blocks install
            spec = None
        if spec is None:
            continue
        if spec.submodule_search_locations:
            out.update(os.path.realpath(p) for p in spec.submodule_search_locations)
        elif spec.origin and os.path.isabs(spec.origin):
            out.add(os.path.realpath(spec.origin))
    return tuple(sorted(out))


def _purge(prefixes: Sequence[str]) -> list[str]:
    """Remove imported forbidden modules from ``sys.modules`` and from their parents' attributes."""
    gone = sorted(n for n in list(sys.modules) if _matches(n, prefixes))
    for n in gone:
        mod = sys.modules.pop(n, None)
        parent, _, leaf = n.rpartition(".")
        pmod = sys.modules.get(parent) if parent else None
        if pmod is not None and vars(pmod).get(leaf) is mod:
            delattr(pmod, leaf)
    return gone


class _ForbiddenImportFinder(MetaPathFinder):
    """First on ``sys.meta_path``: refuses forbidden names and re-imports through ``importlib``. A
    convenience layer only: the audit hook (C import path), the source-path rule and the monitor's
    ``_imp`` rule hold if it is removed."""

    def __init__(self, check: Callable[[str], str | None]) -> None:
        self._check = check

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        why = self._check(fullname)
        if why is not None:
            _refuse(why, ImportViolation)
        return None

    def invalidate_caches(self) -> None:  # pragma: no cover - importlib protocol
        return None


def _make_import_check(forbidden: tuple[str, ...], snapshot: frozenset[str], modules: dict[str, Any]) -> Callable[[str], str | None]:
    def check(fullname: str) -> str | None:
        for pre in forbidden:
            if fullname == pre or fullname.startswith(pre + "."):
                return f"import of forbidden module {fullname!r}"
        if fullname in snapshot and fullname not in modules:
            return f"re-import of {fullname!r} after it was removed from sys.modules"
        return None

    return check


def _validate(mode: str, author_dir: Any, phase: Any, kernel_endpoint: Any, consts_path: Any) -> tuple[str, str, str | None, str | None]:
    if mode not in MODES:
        raise ValueError(f"unknown guard mode {mode!r}; expected one of {MODES}")
    if not isinstance(phase, str) or not phase or "/" in phase or phase.startswith(".") or "\0" in phase:
        raise ValueError(f"phase must be a plain directory name, got {phase!r}")
    adir = _realdir(author_dir, "author_dir")
    ep = None
    if kernel_endpoint is not None:
        if mode != KERNEL_CLIENT:
            raise ValueError("kernel_endpoint is only for KERNEL_CLIENT (the build never talks to the kernel)")
        ep = os.path.realpath(os.fspath(kernel_endpoint))
        if ep == "/":
            raise ValueError("kernel_endpoint may not be the filesystem root")
    cp = None
    if consts_path is not None:
        if mode != BUILD_EDIT:
            raise ValueError("consts_path is only for BUILD_EDIT")
        cp = os.path.realpath(os.fspath(consts_path))
        if not os.path.isfile(cp):
            raise ValueError(f"consts_path {consts_path!r} is not a file")
    return adir, phase, ep, cp


def install(
    mode: str,
    *,
    author_dir: str | os.PathLike[str],
    phase: str,
    kernel_endpoint: str | os.PathLike[str] | None = None,
    consts_path: str | os.PathLike[str] | None = None,
) -> None:
    """Put this interpreter under ``mode`` for the rest of its life (module docstring).

    ``author_dir``: the author directory (KERNEL_CLIENT reads anywhere under it; BUILD_EDIT reads only
    ``output/<phase>/build_edit.py`` there). ``phase``: names ``output/<phase>/``, the only writable
    tree (KERNEL_CLIENT). ``kernel_endpoint``: KERNEL_CLIENT only, the server's AF_UNIX socket path, or a
    directory whose direct children (FIFOs or sockets) may be opened or connected to; None allows no
    socket. ``consts_path``: BUILD_EDIT only, the declared constants file, the one data file it may read.

    Call it from the main thread with no other Python thread running, after everything the guarded
    code needs (``torch``, ``numpy``, ``safetensors``, the tokenizer) is loaded.
    """
    if installed_mode() is not None:
        _refuse(f"install() under an installed {installed_mode()} guard (it cannot be changed or removed)")
    adir, phase, ep, cp = _validate(mode, author_dir, phase, kernel_endpoint, consts_path)
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise GuardViolation("install the guard from the main thread with no other Python thread running")
    if not torch.__version__.startswith(VERIFIED_TORCH):
        raise GuardViolation(f"hn_guard v0.4 was verified on torch {VERIFIED_TORCH}, not {torch.__version__}")
    for opt in ("safetensors", "safetensors.torch", "safetensors.numpy", "tokenizers"):
        try:  # loaded before patching, so their names exist and later use needs no file reads
            __import__(opt)
        except ImportError:
            pass
    try:  # the markers go to a private copy of stderr, so a later dup2 / close of fd 2 cannot divert them
        err_fd = posix.dup(2)
    except OSError as e:
        raise GuardViolation("install the guard with an open stderr (fd 2): its markers go there") from e
    protected = frozenset({2, err_fd})
    originals = (posix._exit, posix.abort, posix.close)  # held only by the actions hook once patched
    counter = itertools.count()  # violations; shared by the audit hook, the actions hook and the monitor
    torch.set_grad_enabled(False)
    sys.dont_write_bytecode = True  # lazy imports would otherwise try to write __pycache__ (refused writes)

    # policy (all realpaths, all immutable)
    guard_dir = os.path.dirname(os.path.realpath(__file__))
    install_roots = _install_roots()
    out_root = os.path.join(adir, "output", phase)
    forbidden = FORBIDDEN_IMPORTS
    purge_prefixes = forbidden if mode == KERNEL_CLIENT else tuple(p for p in forbidden if not _matches(p, RUNNER_PACKAGES))
    forbid_paths = tuple(p for p in _module_locations(forbidden) if p not in install_roots)
    for name in sorted(n for n in sys.modules if _matches(n, purge_prefixes)):
        f = getattr(sys.modules[name], "__file__", None)
        if f and name.count(".") <= 1:
            loc = os.path.dirname(f) if os.path.basename(f) == "__init__.py" else f
            forbid_paths += (os.path.realpath(loc),)
    forbid_paths = tuple(sorted(set(forbid_paths)))
    rng_ok = frozenset()
    if mode == KERNEL_CLIENT:
        r = os.path.join(adir, RENDERER_RELPATH)
        rng_ok = frozenset({r, os.path.realpath(r), os.path.normpath(os.path.join(os.fspath(author_dir), RENDERER_RELPATH))})

    targets = _targets(mode)
    _purge(purge_prefixes)
    modules = sys.modules
    snapshot = frozenset(modules)

    # attribute patches; originals kept (ids stay valid) only by the keeper below
    keep: list[Any] = []
    banned: set[int] = set()
    py_codes: list[types.CodeType] = []
    torch_dir = os.path.dirname(os.path.realpath(torch.__file__)) + "/"
    fd_swaps: dict[int, tuple[Any, Any]] = {}  # the exit / close / dup2 originals -> their replacements
    for owner, name, rep, ban in targets:
        own = vars(owner)
        original = own[name] if name in own else getattr(owner, name, None)
        setattr(owner, name, rep)
        if owner in (posix, os, _socket, socket) and name in _FD_TARGET_NAMES and original is not None:
            fd_swaps[id(original)] = (original, rep)
        if not ban or original is None:
            continue
        func = original.__func__ if isinstance(original, (staticmethod, classmethod, types.MethodType)) else original
        for obj in {id(original): original, id(func): func}.values():
            keep.append(obj)
            banned.add(id(obj))
        code = getattr(func, "__code__", None) if isinstance(func, types.FunctionType) else None
        if code is not None and code.co_qualname == func.__qualname__ and os.path.realpath(code.co_filename).startswith(torch_dir):
            py_codes.append(code)
    _swap_module_aliases(fd_swaps)
    alias_names = set(TORCH_RNG_FUNCS) | set(TENSOR_RNG_METHODS) | set(LSTSQ_PINV_NAMES) | {"from_file"}
    if mode == BUILD_EDIT:
        alias_names |= {"softmax", "log_softmax", "_softmax", "_log_softmax"}
    for v in _alias_originals(alias_names):
        keep.append(v)
        banned.add(id(v))
    sge = next(o for o in keep if getattr(o, "__name__", None) == "_set_grad_enabled" and isinstance(o, types.BuiltinFunctionType))
    inf = next(o for o in keep if getattr(o, "__name__", None) == "_InferenceMode" and isinstance(o, type))
    engine_type = next(o for o in keep if isinstance(o, type) and o.__name__ == "_EngineBase")
    instance_types = {engine_type}
    for o in keep:
        if isinstance(o, type) and o.__name__ in ("PyTorchFileReader", "PyTorchFileWriter", "CompilationUnit", "safe_open"):
            instance_types.add(o)
    for tp in list(instance_types):
        for v in _callables_of(vars(tp).values()):
            keep.append(v)
            banned.add(id(v))
    script_types = {torch._C.ScriptModule}
    for v in _callables_of(vars(torch._C.ScriptModule)[n] for n in SCRIPT_MODULE_NAMES if n in vars(torch._C.ScriptModule)):
        keep.append(v)
        banned.add(id(v))
    np_rng_types = {np.random.RandomState, np.random.Generator, np.random.BitGenerator, np.random.SeedSequence}
    rng_ids: set[int] = set()
    for tp in list(np_rng_types):
        for k in tp.__mro__:
            if k is not object:
                for v in _callables_of(vars(k).values()):
                    keep.append(v)
                    rng_ids.add(id(v))
    rng_py_ids = {id(v) for k in (random.Random, random.SystemRandom) for v in vars(k).values() if isinstance(v, types.FunctionType)}
    rng_py_ids |= {id(v) for v in vars(_random.Random).values() if callable(v)}
    keep.extend(v for k in (random.Random, random.SystemRandom, _random.Random) for v in vars(k).values())
    tok_types: set[type] = set()
    for mod_name in [m for m in list(sys.modules) if m == "tokenizers" or m.startswith("tokenizers.")]:
        for v in vars(sys.modules[mod_name]).values():
            if isinstance(v, type) and str(getattr(v, "__module__", "")).startswith("tokenizers"):
                tok_types.add(v)
    op_banned = OP_BANNED | (OP_BANNED_BUILD if mode == BUILD_EDIT else frozenset())
    from torch._ops import OpOverload, OpOverloadPacket

    # the import finder, the audit hook, the exit check, then the monitor (last: install itself is unmonitored)
    sys.meta_path.insert(0, _ForbiddenImportFinder(_make_import_check(forbidden, snapshot, modules)))
    hook, at_exit = _make_audit_hook(
        mode=mode,
        read_roots=(adir,) + install_roots if mode == KERNEL_CLIENT else (),
        read_exact=frozenset({cp} if cp else set()) | ({os.path.join(out_root, BUILD_FILE)} if mode == BUILD_EDIT else frozenset()),
        code_roots=install_roots + (guard_dir,) if mode == BUILD_EDIT else (),
        write_roots=(out_root,) if mode == KERNEL_CLIENT else (),
        endpoint=ep,
        endpoint_is_dir=bool(ep and os.path.isdir(ep)),
        forbid_paths=forbid_paths,
        forbidden=forbidden,
        snapshot=snapshot,
        modules=modules,
        counter=counter,
        err_fd=err_fd,
        protected=protected,
        nowrite_roots=(os.path.join(out_root, FWD_DIR),) if mode == KERNEL_CLIENT else (),
    )
    sys.addaudithook(hook)
    sys.addaudithook(_make_actions_hook(counter=counter, err_fd=err_fd, protected=protected, originals=originals))
    atexit.register(at_exit)  # registered before atexit._clear / unregister are patched (they are, in targets)

    on_call, on_c_return, on_py_start = _make_monitor(
        banned=frozenset(banned),
        sge_id=id(sge),
        inference_id=id(inf),
        op_banned=op_banned,
        op_types=(OpOverload, OpOverloadPacket),
        instance_types=frozenset(id(tp) for tp in instance_types),
        script_types=frozenset(id(tp) for tp in script_types),
        script_names=SCRIPT_MODULE_NAMES,
        rng_ids=frozenset(rng_ids),
        rng_py_ids=frozenset(rng_py_ids),
        rng_c_types=frozenset({id(_random.Random)}),
        rng_ok=rng_ok,
        np_rng_types=frozenset(id(tp) for tp in np_rng_types),
        tok_types=frozenset(id(tp) for tp in tok_types),
        mon_ns=sys.monitoring,
        imp_mod=_imp,
        snapshot=snapshot,
        modules=modules,
        forbidden=forbidden,
        ext_roots=install_roots,
        forbid_paths=forbid_paths,
        spec_type=importlib.machinery.ModuleSpec,
        counter=counter,
        protected=protected,
    )
    keep_all = tuple(keep)

    def keeper() -> Any:  # registered for an event that is never enabled: holds the originals, never runs
        return keep_all

    _install_monitor(on_call, on_c_return, on_py_start, keeper, py_codes)


def _install_monitor(on_call: Any, on_c_return: Any, on_py_start: Any, keeper: Any, py_codes: list[types.CodeType]) -> None:
    mon = sys.monitoring
    E = mon.events
    for tid in _TOOL_PREFERENCE:
        if mon.get_tool(tid) is None:
            break
    else:
        raise GuardViolation("no free sys.monitoring tool id for the guard")
    mon.use_tool_id(tid, "hn_guard.v04")
    mon.register_callback(tid, E.CALL, on_call)
    mon.register_callback(tid, E.C_RETURN, on_c_return)
    mon.register_callback(tid, E.C_RAISE, on_c_return)
    mon.register_callback(tid, E.PY_START, on_py_start)
    mon.register_callback(tid, E.BRANCH, keeper)
    for code in py_codes:
        mon.set_local_events(tid, code, E.PY_START)
    mon.set_events(tid, E.CALL | E.C_RETURN | E.C_RAISE)
    # From here on nobody can free, re-point or switch off the tool (the originals are dropped).
    for n in _MON_MUTATORS:
        setattr(mon, n, _raiser(f"sys.monitoring.{n} (the guard's monitor cannot be changed)"))
