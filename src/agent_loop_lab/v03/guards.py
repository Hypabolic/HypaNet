"""v0.3 guards: BUILD and AUTHOR_RUN modes (spec v0.3 §8, §9.1, §11 "Anti-training").

Both modes run inside the v0.2 ``no_training_guard`` (imported, not modified),
which already makes training impossible: no grad mode, no ``.backward`` or
autograd engine, no ``torch.optim`` optimiser, no forward-mode AD, no new
thread or process, and no import of ``agent_loop_lab.{synth, synth_ood,
datasets, eval, baselines}``. On top of that, this module adds:

* **RNG ban (both modes).** The module-level functions of ``random`` (which use
  the hidden global generator), ``random.SystemRandom``, every public callable
  of ``numpy.random``, the ``torch`` random functions (``rand``, ``randn``,
  ``randint``, ``randperm``, ``*_like``, ``normal``, ``bernoulli``,
  ``multinomial``, ``poisson``, ``manual_seed``, ``seed``, ``set_rng_state``,
  ``torch.Generator``), the in-place RNG ``Tensor`` methods (``normal_``,
  ``uniform_``, ...) and the random ``torch.nn.functional`` calls (dropout
  family, ``rrelu``, ``gumbel_softmax``) are replaced by tripwires.
  Constructing ``random.Random`` is refused too, except when the direct caller
  is one of ``rng_allowed_files``: the AUTHOR_RUN launcher passes the packet's
  ``renderer.py``, whose per-event render RNG is a seeded ``random.Random``
  derived from sha256 of the event (spec §4.5). Imports of ``random`` or
  ``numpy.random`` are not blocked, because ``torch`` and ``numpy`` import them
  themselves; their *use* is what is refused.
* **Compute ban (BUILD only, spec §9.1).** Every public callable of
  ``torch.nn.functional`` and ``torch.linalg``; ``torch.softmax`` /
  ``log_softmax`` (also ``torch.special``'s and the ``Tensor`` methods);
  ``matmul`` and the ``@`` operator (``Tensor.__matmul__`` / ``__rmatmul__`` /
  ``__imatmul__``); ``mm``, ``bmm``, ``addmm``, ``addr``, ``ger``, ``baddbmm``,
  ``addbmm``, ``addmv``, ``mv``, ``dot``, ``vdot``, ``inner``, ``tensordot``,
  ``einsum``, ``chain_matmul``, ``bilinear`` (functions and in-place / method
  forms); the numpy equivalents; calling any ``nn.Module``
  (``nn.Module.__call__``); and ``forward`` / ``forward_batch`` /
  ``activations`` / ``_run`` of every class in ``model_classes`` (the runner
  passes the packet's and the harness's ``DoomLoopTransformer``).
  ``torch.outer`` / ``Tensor.outer`` / ``torch.kron`` stay allowed, as do
  indexing, index assignment, ``index_put_``, elementwise arithmetic,
  ``clone``, ``zeros``, ``tensor(...)``, reductions and ``torch.embedding``.
* **Identity monitor (both modes).** Attribute patches cannot affect references
  reached around the patched names (``torch._C._VariableFunctions.matmul``,
  ``torch._C._TensorBase.matmul``, ``torch.ops.aten.mm``, a bound
  ``model.forward``). A ``sys.monitoring`` CALL monitor (a second tool id, next
  to v0.2's grad-mode monitor) refuses calls from code outside this module
  whose target is the original of a banned callable, a ``torch._C`` alias of
  one, an ``aten`` op overload of a banned name, or a bound banned method.
* **Imports.** BUILD also forbids ``agent_loop_lab.v03.renderer`` and the
  packet's bare ``renderer`` and ``reference_policy`` modules (spec §9.1).
  AUTHOR_RUN forbids only the v0.2 set (the author renders logs with the
  packet renderer).
* **I/O.** BUILD: every ``open`` is refused except a read of a Python code
  file (``.py``, ``.pyc``, ``.so``, ...) that lies under the interpreter's
  install (``sys.prefix``, ``sys.base_prefix``, ...) or under one of the
  ``read_roots`` the caller passes (the runner passes the packet directory and
  the private copy of the one author file it executes); filesystem changes and
  sockets are refused too. AUTHOR_RUN (when ``io_root`` is given): reading or
  listing a path under one of ``io_deny`` that is not under ``io_root`` is
  refused, and so is any write or filesystem change outside ``io_root``.
* **File reads that raise no ``open`` audit event (both modes).** C-level
  readers are tripwired: ``torch.from_file``, ``UntypedStorage.from_file`` /
  ``TypedStorage.from_file`` (and the ``torch._C`` / ``torch.ops.aten``
  aliases), ``torch.jit.load``, ``torch._C.import_ir_module`` and the other
  path-taking ``torch._C`` loaders (:data:`IO_BANNED_BUILTIN_NAMES`), and
  ``torch._C.PyTorchFileReader`` given a path (a file object is allowed, which
  is how ``torch.load`` calls it). Every ``ctypes.*`` audit event (``dlopen``,
  ``dlsym``, foreign calls, raw memory reads) and ``mmap.__new__`` are refused.
  BUILD also refuses ``torch.load`` and numpy's ``fromfile`` / ``load`` /
  ``memmap`` / ``loadtxt`` / ``genfromtxt`` / ``fromregex``.

Violations that the guarded code catches still fail the guard on exit. Like
the v0.2 guard this is a tripwire, not a sandbox: the runner's and the
audit's static source rules (``source_rules``) are the other half.

The module also works as a standalone copy in the weight author's ``tools/``
directory (``kit.install_tools``), where the v0.2 guard is imported from the
sibling ``guards_v02`` module instead of ``agent_loop_lab.guards``.
"""

from __future__ import annotations

import os
import random as _random
import sys
import threading
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch
from torch import nn

if __package__ == "agent_loop_lab.v03":
    from ..guards import TrainingViolation, audit_model, guard_active, no_training_guard
else:  # standalone copy (author tools/): the v0.2 guard sits next to this file
    from .guards_v02 import TrainingViolation, audit_model, guard_active, no_training_guard  # type: ignore

__all__ = [
    "BUILD",
    "AUTHOR_RUN",
    "MODES",
    "TrainingViolation",
    "audit_model",
    "v03_guard",
    "build_guard",
    "author_run_guard",
    "v03_guard_active",
    "BUILD_FORBIDDEN_IMPORTS",
    "COMPUTE_BANNED_TORCH",
    "COMPUTE_BANNED_TENSOR",
    "COMPUTE_BANNED_NUMPY",
    "COMPUTE_ALLOWED",
    "RANDOM_MODULE_FUNCS",
    "TORCH_RNG_FUNCS",
    "TENSOR_RNG_METHODS",
    "FUNCTIONAL_RNG",
    "MODEL_FORWARD_METHODS",
    "IO_BANNED_BUILTIN_NAMES",
    "MAX_VIOLATION_PATH_CHARS",
]

BUILD = "build"
AUTHOR_RUN = "author_run"
MODES = (BUILD, AUTHOR_RUN)

# Extra import prefixes refused in BUILD (on top of the v0.2 set).
BUILD_FORBIDDEN_IMPORTS: tuple[str, ...] = ("agent_loop_lab.v03.renderer", "renderer", "reference_policy")

# -- RNG ---------------------------------------------------------------------------

RANDOM_MODULE_FUNCS: tuple[str, ...] = (
    "random", "uniform", "randint", "randrange", "choice", "choices", "shuffle", "sample", "seed",
    "gauss", "normalvariate", "lognormvariate", "expovariate", "vonmisesvariate", "gammavariate",
    "betavariate", "paretovariate", "weibullvariate", "triangular", "getrandbits", "randbytes",
    "binomialvariate", "setstate",
)
TORCH_RNG_FUNCS: tuple[str, ...] = (
    "rand", "randn", "randint", "randperm", "rand_like", "randn_like", "randint_like", "normal",
    "bernoulli", "multinomial", "poisson", "manual_seed", "seed", "set_rng_state", "dropout",
    "alpha_dropout", "feature_alpha_dropout", "feature_dropout", "rrelu", "native_dropout",
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

# -- compute ban (BUILD) -------------------------------------------------------------

COMPUTE_BANNED_TORCH: tuple[str, ...] = (
    "matmul", "mm", "bmm", "addmm", "addr", "ger", "baddbmm", "addbmm", "addmv", "mv", "dot", "vdot",
    "inner", "tensordot", "einsum", "chain_matmul", "softmax", "log_softmax", "bilinear",
)
COMPUTE_BANNED_TENSOR: tuple[str, ...] = (
    "__matmul__", "__rmatmul__", "__imatmul__", "matmul", "mm", "bmm", "addmm", "addmm_", "addr", "addr_",
    "ger", "baddbmm", "baddbmm_", "addbmm", "addbmm_", "addmv", "addmv_", "mv", "dot", "vdot", "inner",
    "softmax", "log_softmax",
)
COMPUTE_BANNED_NUMPY: tuple[str, ...] = ("matmul", "dot", "einsum", "tensordot", "inner", "vdot")
COMPUTE_ALLOWED: tuple[str, ...] = ("outer", "kron")
MODEL_FORWARD_METHODS: tuple[str, ...] = ("forward", "forward_batch", "activations", "_run")
# aten op base names refused through torch.ops in BUILD (besides the names above).
_ATEN_COMPUTE_EXTRA = ("linear", "_softmax", "_log_softmax", "addmm_", "baddbmm_", "addbmm_", "addmv_", "addr_")

# -- file readers that bypass the ``open`` audit event (both modes) ------------------------
# Builtin (C) functions with these names are refused by the identity monitor wherever they are reached.
IO_BANNED_BUILTIN_NAMES: frozenset[str] = frozenset(
    {"from_file", "import_ir_module", "_load_for_lite_interpreter", "_load_mobile_module_from_file",
     "_load_jit_module_from_file", "_get_model_bytecode_version", "_get_mobile_model_contained_types",
     "_get_model_ops_and_info", "_backport_for_mobile", "_load_mobile_module_from_bytes_file"}
)
_IO_WHY = "no file reads that bypass the I/O policy"
_NUMPY_IO_BUILD = ("fromfile", "load", "memmap", "loadtxt", "genfromtxt", "fromregex")
MAX_VIOLATION_PATH_CHARS = 120

_THIS_FILE = __file__


@dataclass
class _Patch:
    owner: Any
    name: str
    had_own: bool
    original: Any
    replacement: Any


@dataclass
class _State:
    mode: str
    thread: int
    violations: list[str] = field(default_factory=list)
    patches: list[_Patch] = field(default_factory=list)
    banned_ids: set[int] = field(default_factory=set)
    banned_keep: list[Any] = field(default_factory=list)  # keeps ids stable
    banned_funcs: set[Any] = field(default_factory=set)  # plain functions (bound-method targets)
    aten_names: frozenset[str] = frozenset()
    rng_allowed_files: frozenset[str] = frozenset()
    io_root: str | None = None
    io_deny: tuple[str, ...] = ()
    read_roots: tuple[str, ...] = ()
    monitor_tool: int | None = None


_LOCK = threading.RLock()
_STATE: _State | None = None


def v03_guard_active() -> bool:
    return _STATE is not None


def _violation(msg: str) -> TrainingViolation:
    st = _STATE
    if st is not None:
        st.violations.append(msg)
    return TrainingViolation(msg)


def _mode_label(mode: str) -> str:
    return "BUILD guard" if mode == BUILD else "AUTHOR_RUN guard"


def _raiser(what: str, why: str) -> Callable[..., Any]:
    def _blocked(*args: Any, **kwargs: Any) -> Any:
        st = _STATE
        label = _mode_label(st.mode) if st is not None else "v0.3 guard"
        raise _violation(f"{what} refused under the {label} ({why})")

    _blocked.__name__ = "blocked_" + "".join(c if c.isalnum() else "_" for c in what)
    _blocked.__qualname__ = _blocked.__name__
    return _blocked


def _blocked_class(what: str, why: str) -> type:
    def __new__(cls: type, *args: Any, **kwargs: Any) -> Any:  # noqa: N807
        st = _STATE
        label = _mode_label(st.mode) if st is not None else "v0.3 guard"
        raise _violation(f"{what} refused under the {label} ({why})")

    return type("Blocked_" + "".join(c if c.isalnum() else "_" for c in what), (), {"__new__": __new__})


def _guarded_file_reader(orig: Any) -> Callable[..., Any]:
    """``torch._C.PyTorchFileReader`` reads a path in C++ (no audit event); a file object goes through
    Python's audited ``open`` first, so only that form is allowed."""

    def PyTorchFileReader(arg: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: N802
        if isinstance(arg, (str, bytes, os.PathLike)):
            st = _STATE
            label = _mode_label(st.mode) if st is not None else "v0.3 guard"
            raise _violation(f"torch._C.PyTorchFileReader on a path refused under the {label} ({_IO_WHY})")
        return orig(arg, *args, **kwargs)

    return PyTorchFileReader


_RNG_WHY = "random numbers are forbidden; derive quasi-random values from hashlib"
_COMPUTE_WHY = "no forward passes, matrix products, softmax or linear-algebra calls in the weight build"


def _guarded_random_init(orig: Callable[..., Any]) -> Callable[..., Any]:
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: N807
        st = _STATE
        if st is not None:
            caller = sys._getframe(1).f_code.co_filename
            try:
                caller = os.path.realpath(caller)
            except (OSError, ValueError):
                pass
            if caller not in st.rng_allowed_files:
                raise _violation(
                    f"random.Random construction (from {os.path.basename(caller)}) refused under the "
                    f"{_mode_label(st.mode)} ({_RNG_WHY})"
                )
        orig(self, *args, **kwargs)

    return __init__


_HELPER_MODULES = ("typing", "builtins", "torch._jit_internal", "torch.overrides", "torch._C")


def _public_callables(mod: Any, classes: bool = False) -> list[str]:
    """Public callables defined by ``mod``'s own library: no re-exported typing helpers, builtins,
    torch-function dispatch helpers (``has_torch_function*``, ``handle_torch_function``) or, unless
    ``classes``, classes (``F.Tensor``)."""
    out = []
    for n in sorted(vars(mod)):
        if n.startswith("_"):
            continue
        v = vars(mod)[n]
        if isinstance(v, types.ModuleType) or not callable(v):
            continue
        # re-exported typing helpers and builtins (e.g. F.Optional, F.DType = int) are not the module's own
        if getattr(v, "__module__", None) in _HELPER_MODULES or any(v is b for b in _BUILTIN_VALUES):
            continue
        if isinstance(v, type) and not classes:
            continue
        out.append(n)
    return out


import builtins as _builtins  # noqa: E402

_BUILTIN_VALUES = tuple(vars(_builtins).values())


def _patch_targets(mode: str, model_classes: Sequence[type]) -> list[tuple[Any, str, Any]]:
    """(owner, attribute, replacement) triples for the mode."""
    t: list[tuple[Any, str, Any]] = []
    # RNG: stdlib random
    for n in RANDOM_MODULE_FUNCS:
        if hasattr(_random, n):
            t.append((_random, n, _raiser(f"random.{n}", _RNG_WHY)))
    t.append((_random.Random, "__init__", _guarded_random_init(vars(_random.Random)["__init__"])))
    t.append((_random.SystemRandom, "__init__", _raiser("random.SystemRandom", _RNG_WHY)))
    # RNG: torch
    for n in TORCH_RNG_FUNCS:
        if hasattr(torch, n):
            t.append((torch, n, _raiser(f"torch.{n}", _RNG_WHY)))
    t.append((torch, "Generator", _blocked_class("torch.Generator", _RNG_WHY)))
    for n in TORCH_RANDOM_MODULE_FUNCS:
        if hasattr(torch.random, n):
            t.append((torch.random, n, _raiser(f"torch.random.{n}", _RNG_WHY)))
    for n in TENSOR_RNG_METHODS:
        if hasattr(torch.Tensor, n):
            t.append((torch.Tensor, n, _raiser(f"Tensor.{n}", _RNG_WHY)))
    # RNG: numpy (if installed)
    np_random = sys.modules.get("numpy.random")
    if np_random is not None:
        for n in _public_callables(np_random, classes=True):
            v = vars(np_random)[n]
            rep = _blocked_class(f"numpy.random.{n}", _RNG_WHY) if isinstance(v, type) else _raiser(f"numpy.random.{n}", _RNG_WHY)
            t.append((np_random, n, rep))
    # C-level file readers (both modes): no open audit event is raised for these
    t.append((torch, "from_file", _raiser("torch.from_file", _IO_WHY)))
    for cls in (torch.UntypedStorage, torch.TypedStorage):
        if hasattr(cls, "from_file"):
            t.append((cls, "from_file", _raiser(f"{cls.__name__}.from_file", _IO_WHY)))
    # torch._C.StorageBase is an immutable C type: its from_file (the original UntypedStorage.from_file, one
    # object) is refused by identity and by IO_BANNED_BUILTIN_NAMES in the monitor.
    jit = getattr(torch, "jit", None)
    if jit is not None and hasattr(jit, "load"):
        t.append((jit, "load", _raiser("torch.jit.load", _IO_WHY)))
    for n in sorted(IO_BANNED_BUILTIN_NAMES):
        if n != "from_file" and hasattr(torch._C, n):
            t.append((torch._C, n, _raiser(f"torch._C.{n}", _IO_WHY)))
    if hasattr(torch._C, "PyTorchFileReader"):
        t.append((torch._C, "PyTorchFileReader", _guarded_file_reader(torch._C.PyTorchFileReader)))
    F = torch.nn.functional
    if mode == BUILD:
        for owner in (torch, getattr(torch, "serialization", None)):
            if owner is not None and hasattr(owner, "load"):
                t.append((owner, "load", _raiser(f"{owner.__name__}.load", "no file I/O in the weight build")))
        np_io = sys.modules.get("numpy")
        if np_io is not None:
            for n in _NUMPY_IO_BUILD:
                v = getattr(np_io, n, None)
                if v is not None:
                    rep = _blocked_class(f"numpy.{n}", "no file I/O in the weight build") if isinstance(v, type) else _raiser(f"numpy.{n}", "no file I/O in the weight build")
                    t.append((np_io, n, rep))
        for n in _public_callables(F):
            t.append((F, n, _raiser(f"torch.nn.functional.{n}", _COMPUTE_WHY)))
        for n in COMPUTE_BANNED_TORCH:
            if hasattr(torch, n):
                t.append((torch, n, _raiser(f"torch.{n}", _COMPUTE_WHY)))
        for n in ("softmax", "log_softmax"):
            if hasattr(torch.special, n):
                t.append((torch.special, n, _raiser(f"torch.special.{n}", _COMPUTE_WHY)))
        for n in COMPUTE_BANNED_TENSOR:
            if hasattr(torch.Tensor, n):
                t.append((torch.Tensor, n, _raiser(f"Tensor.{n}" + (" (the @ operator)" if "matmul" in n and n.startswith("__") else ""), _COMPUTE_WHY)))
        for n in _public_callables(torch.linalg):
            t.append((torch.linalg, n, _raiser(f"torch.linalg.{n}", _COMPUTE_WHY)))
        np_mod = sys.modules.get("numpy")
        if np_mod is not None:
            for n in COMPUTE_BANNED_NUMPY:
                if hasattr(np_mod, n):
                    t.append((np_mod, n, _raiser(f"numpy.{n}", _COMPUTE_WHY)))
            np_linalg = sys.modules.get("numpy.linalg")
            if np_linalg is not None:
                for n in _public_callables(np_linalg):
                    t.append((np_linalg, n, _raiser(f"numpy.linalg.{n}", _COMPUTE_WHY)))
        t.append((nn.Module, "__call__", _raiser("calling an nn.Module", _COMPUTE_WHY)))
        for cls in model_classes:
            for n in MODEL_FORWARD_METHODS:
                if n in vars(cls):
                    t.append((cls, n, _raiser(f"{cls.__name__}.{n}", _COMPUTE_WHY)))
    else:
        for n in FUNCTIONAL_RNG:
            if hasattr(F, n):
                t.append((F, n, _raiser(f"torch.nn.functional.{n}", _RNG_WHY)))
    # de-duplicate (owner, name): first wins
    seen: set[tuple[int, str]] = set()
    out = []
    for owner, name, rep in t:
        key = (id(owner), name)
        if key in seen:
            continue
        seen.add(key)
        out.append((owner, name, rep))
    return out


def _install(st: _State, targets: list[tuple[Any, str, Any]]) -> None:
    for owner, name, rep in targets:
        had_own = name in vars(owner)
        original = vars(owner)[name] if had_own else getattr(owner, name, None)
        setattr(owner, name, rep)
        st.patches.append(_Patch(owner, name, had_own, original if had_own else None, rep))
        if original is not None:
            st.banned_keep.append(original)
            st.banned_ids.add(id(original))
            func = getattr(original, "__func__", original)
            if isinstance(func, types.FunctionType):
                st.banned_funcs.add(func)


def _restore(st: _State) -> list[str]:
    problems: list[str] = []
    for p in reversed(st.patches):
        try:
            current = vars(p.owner).get(p.name, None)
            if current is not p.replacement:
                problems.append(f"patch on {getattr(p.owner, '__name__', p.owner)!s}.{p.name} was tampered with")
            if p.had_own:
                setattr(p.owner, p.name, p.original)
            elif p.name in vars(p.owner):
                delattr(p.owner, p.name)
        except Exception as exc:  # pragma: no cover - defensive
            problems.append(f"failed to restore {p.name}: {exc!r}")
    st.patches.clear()
    return problems


def _alias_originals(st: _State, names: Iterable[str]) -> None:
    """Add the ``torch._C`` aliases of banned names (functions and Tensor method descriptors)."""
    owners = [torch._C._VariableFunctions, torch._C._nn, torch._C._linalg, torch._C._special, torch._C._TensorBase]
    for owner in owners:
        for n in names:
            v = vars(owner).get(n) if isinstance(owner, type) else getattr(owner, n, None)
            if v is not None and callable(v):
                st.banned_keep.append(v)
                st.banned_ids.add(id(v))


# -- identity monitor ---------------------------------------------------------------

_MON = sys.monitoring
_MONITOR_NAME = "agent_loop_lab.v03_guard"
_MONITOR_PREFERENCE = (4, 3, 2, 5, 1, 0)
_OP_TYPES: tuple[type, ...] = ()
try:  # torch.ops overloads
    from torch._ops import OpOverload, OpOverloadPacket

    _OP_TYPES = (OpOverload, OpOverloadPacket)
except Exception:  # pragma: no cover
    pass


def _op_base_name(op: Any) -> str:
    q = getattr(op, "_qualified_op_name", None)
    if q is None:
        q = getattr(getattr(op, "_overloadpacket", None), "_qualified_op_name", "") or ""
    return str(q).split("::")[-1]


def _monitor_call(code: Any, offset: int, callable_: Any, arg0: Any) -> Any:
    st = _STATE
    if st is None or code.co_filename == _THIS_FILE:
        return None
    if id(callable_) in st.banned_ids:
        name = getattr(callable_, "__qualname__", None) or getattr(callable_, "__name__", repr(callable_))
        raise _violation(f"call of banned {name} (reached around the patched name) refused under the {_mode_label(st.mode)}")
    tc = type(callable_)
    if tc is types.BuiltinFunctionType and getattr(callable_, "__name__", None) in IO_BANNED_BUILTIN_NAMES:
        raise _violation(f"call of {callable_.__name__} (a file reader without an audit event) refused under the {_mode_label(st.mode)}")
    if tc is types.MethodType and callable_.__func__ in st.banned_funcs:
        raise _violation(f"call of banned bound method {callable_.__func__.__qualname__} refused under the {_mode_label(st.mode)}")
    if _OP_TYPES and tc in _OP_TYPES and _op_base_name(callable_) in st.aten_names:
        raise _violation(f"call of banned aten op {_op_base_name(callable_)} (torch.ops) refused under the {_mode_label(st.mode)}")
    return None


def _install_monitor(st: _State) -> None:
    for tid in _MONITOR_PREFERENCE:
        if _MON.get_tool(tid) is None:
            break
    else:
        raise TrainingViolation("cannot start the v0.3 guard: no free sys.monitoring tool id")
    _MON.use_tool_id(tid, _MONITOR_NAME)
    st.monitor_tool = tid
    _MON.register_callback(tid, _MON.events.CALL, _monitor_call)
    _MON.set_events(tid, _MON.events.CALL)


def _uninstall_monitor(st: _State) -> list[str]:
    tid = st.monitor_tool
    if tid is None:
        return []
    st.monitor_tool = None
    problems: list[str] = []
    try:
        if _MON.get_tool(tid) != _MONITOR_NAME:
            return [f"v0.3 identity monitor (sys.monitoring tool {tid}) was freed or replaced"]
        if _MON.get_events(tid) != _MON.events.CALL:
            problems.append("v0.3 identity monitor events were changed")
        _MON.set_events(tid, 0)
        if _MON.register_callback(tid, _MON.events.CALL, None) is not _monitor_call:
            problems.append("v0.3 identity monitor callback was replaced")
        _MON.free_tool_id(tid)
    except Exception as exc:  # pragma: no cover
        problems.append(f"failed to remove the v0.3 identity monitor: {exc!r}")
    return problems


# -- I/O audit hook -------------------------------------------------------------------

_CODE_SUFFIXES = (".py", ".pyc", ".so", ".dylib", ".pyd", ".pth")
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_FS_CHANGE_EVENTS = frozenset(
    {"os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.truncate", "os.chmod", "os.symlink", "os.link",
     "os.chown", "os.utime", "shutil.copyfile", "shutil.copytree", "shutil.rmtree", "shutil.move"}
)
_LIST_EVENTS = frozenset({"os.listdir", "os.scandir", "glob.glob", "glob.glob/2", "os.walk"})
_HOOK_INSTALLED = False


def _is_write(args: tuple[Any, ...]) -> bool:
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    if isinstance(mode, str):
        return any(c in mode for c in "wax+")
    return bool(isinstance(flags, int) and flags & _WRITE_FLAGS)


def _abspath(p: Any) -> str | None:
    if isinstance(p, bytes):
        p = os.fsdecode(p)
    if isinstance(p, os.PathLike):
        p = os.fspath(p)
    if not isinstance(p, str):
        return None
    try:
        return os.path.realpath(os.path.join(os.getcwd(), p))
    except (OSError, ValueError):
        return os.path.abspath(p)


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _short(path: Any) -> str:
    r = repr(path)
    return r if len(r) <= MAX_VIOLATION_PATH_CHARS else r[: MAX_VIOLATION_PATH_CHARS - 3] + "..."


def _install_roots() -> tuple[str, ...]:
    """The interpreter's stdlib and site-packages roots (realpath)."""
    out = []
    for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        if p:
            out.append(os.path.realpath(p))
    return tuple(sorted(set(out)))


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    st = _STATE
    if st is None:
        return
    if event.startswith("ctypes.") or event == "mmap.__new__":
        raise _violation(f"{event} refused under the {_mode_label(st.mode)} (no ctypes or mmap access)")
    if st.mode == BUILD:
        if event == "open":
            target = args[0] if args else None
            if isinstance(target, int):
                return  # an already open descriptor (os.fdopen)
            path = _abspath(target)
            if (
                path is not None
                and path.lower().endswith(_CODE_SUFFIXES)
                and not _is_write(args)
                and any(_under(path, r) for r in st.read_roots)
            ):
                return
            raise _violation(
                f"file access ({_short(target)}) refused under the BUILD guard (no file I/O in the weight build; only "
                "Python code of the interpreter install and the packet can be read)"
            )
        # Directory listings are not refused here: the import system lists sys.path entries.
        if event in _FS_CHANGE_EVENTS or event.startswith("socket.") or event == "urllib.Request":
            raise _violation(f"{event} refused under the BUILD guard (no file or network I/O in the weight build)")
        return
    # AUTHOR_RUN
    if event.startswith("socket.") or event == "urllib.Request":
        raise _violation(f"{event} refused under the AUTHOR_RUN guard (no network access)")
    if st.io_root is None:
        return
    if event == "open" or event in _LIST_EVENTS:
        target = args[0] if args else None
        if isinstance(target, int) or target is None:
            return
        path = _abspath(target)
        if path is None:
            return
        inside = _under(path, st.io_root)
        if not inside and any(_under(path, d) for d in st.io_deny):
            raise _violation(f"access to {_short(path)} refused under the AUTHOR_RUN guard (outside the author directory)")
        if event == "open" and _is_write(args) and not inside:
            raise _violation(f"write to {_short(path)} refused under the AUTHOR_RUN guard (writes stay in the author directory)")
        return
    if event in _FS_CHANGE_EVENTS:
        for a in args[:2]:
            path = _abspath(a)
            if path is not None and not _under(path, st.io_root):
                raise _violation(f"{event} on {_short(path)} refused under the AUTHOR_RUN guard (outside the author directory)")


def _ensure_hook() -> None:
    global _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit_hook)
        _HOOK_INSTALLED = True


# -- public context managers ------------------------------------------------------------


def _real(paths: Iterable[str | os.PathLike[str]]) -> frozenset[str]:
    return frozenset(os.path.realpath(os.fspath(p)) for p in paths)


@contextmanager
def v03_guard(
    mode: str,
    *,
    model_classes: Sequence[type] = (),
    rng_allowed_files: Iterable[str | os.PathLike[str]] = (),
    io_root: str | os.PathLike[str] | None = None,
    io_deny: Iterable[str | os.PathLike[str]] = (),
    read_roots: Iterable[str | os.PathLike[str]] = (),
    extra_forbidden: Sequence[str] = (),
) -> Iterator[None]:
    """Run a block under the v0.2 anti-training guard plus the v0.3 ``mode`` rules (module docstring).

    ``model_classes``: classes whose forward methods are refused (BUILD).
    ``rng_allowed_files``: files whose code may construct a seeded ``random.Random`` (AUTHOR_RUN).
    ``io_root`` / ``io_deny``: AUTHOR_RUN file policy (no policy when ``io_root`` is None).
    ``read_roots``: BUILD only: files or directories, besides the interpreter install, whose Python code
    may be read (imports of the packet modules, the executed author file).
    """
    global _STATE
    if mode not in MODES:
        raise ValueError(f"unknown guard mode {mode!r}; expected one of {MODES}")
    with _LOCK:
        if _STATE is not None:
            raise TrainingViolation("a v0.3 guard is already active; refusing to nest")
    forbidden = tuple(extra_forbidden) + (BUILD_FORBIDDEN_IMPORTS if mode == BUILD else ())
    try:  # imported before patching so its module objects exist (its own import draws no numbers)
        import numpy.random  # noqa: F401
        import numpy.linalg  # noqa: F401
    except ImportError:  # pragma: no cover - numpy is optional for the guard
        pass
    body_exc: BaseException | None = None
    with no_training_guard(extra_forbidden=forbidden):
        st = _State(
            mode=mode,
            thread=threading.get_ident(),
            rng_allowed_files=_real(rng_allowed_files) if mode == AUTHOR_RUN else frozenset(),
            io_root=os.path.realpath(os.fspath(io_root)) if (io_root is not None and mode == AUTHOR_RUN) else None,
            io_deny=tuple(sorted(_real(io_deny))) if mode == AUTHOR_RUN else (),
            read_roots=tuple(sorted(set(_install_roots()) | _real(read_roots))) if mode == BUILD else (),
        )
        targets = _patch_targets(mode, model_classes)
        names = set(TORCH_RNG_FUNCS) | set(TENSOR_RNG_METHODS) | {"from_file"}
        if mode == BUILD:
            names |= set(COMPUTE_BANNED_TORCH) | set(COMPUTE_BANNED_TENSOR) | set(_ATEN_COMPUTE_EXTRA)
        st.aten_names = frozenset(names)
        with _LOCK:
            _ensure_hook()
            try:
                _install(st, targets)
                _alias_originals(st, names)
                _STATE = st
                _install_monitor(st)  # last: set-up is not monitored
            except BaseException:
                _STATE = None
                _uninstall_monitor(st)
                _restore(st)
                raise
        try:
            yield
        except BaseException as exc:
            body_exc = exc
            raise
        finally:
            with _LOCK:
                _STATE = None
                problems = _uninstall_monitor(st)
                problems += _restore(st)
                if st.violations:
                    problems.insert(0, f"{len(st.violations)} violation(s) occurred inside the v0.3 guard: {st.violations}")
            if problems and body_exc is None:
                raise TrainingViolation("; ".join(problems))


def build_guard(model_classes: Sequence[type] = (), **kw: Any) -> Any:
    """BUILD mode (weight build): RNG + compute ban + no I/O; see the module docstring."""
    return v03_guard(BUILD, model_classes=model_classes, **kw)


def author_run_guard(**kw: Any) -> Any:
    """AUTHOR_RUN mode (the author's ``./py``): forward passes allowed; no autograd, optimiser or RNG."""
    return v03_guard(AUTHOR_RUN, **kw)
