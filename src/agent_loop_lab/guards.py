"""Anti-training guard for the direct-weight path (spec v0.2 §9.1, §11).

``no_training_guard`` makes the weight-build path fail loudly if it tries to
train: create a ``torch.optim`` optimiser, run reverse- or forward-mode
autograd, enable grad mode, record an autograd graph, start a thread or a
process, or import a forbidden package (generator, datasets, eval,
baselines).

It is a tripwire, not a sandbox. It catches ordinary and moderately
evasive training code. Code that deliberately edits this module's private
state, or that bound torch internals before the guard was entered, can still
get past an in-process guard. The build runner should therefore enter the
guard in a fresh interpreter before any user code is imported.

How it intercepts torch 2.2.2 (checked against the installed source):

* ``torch.no_grad`` / ``torch.enable_grad`` / ``torch.set_grad_enabled`` all
  call ``torch._C._set_grad_enabled`` by attribute lookup at call time
  (``torch/autograd/grad_mode.py``). ``enable_grad`` never goes through the
  public ``torch.set_grad_enabled``. The guard patches the class methods
  (so every alias such as ``torch.autograd.grad_mode.enable_grad``, and any
  pre-captured reference to those classes, is covered) **and** the
  ``torch._C`` chokepoint.
* ``torch.inference_mode(False)`` re-enables grad mode inside ``no_grad`` via
  the C++ ``torch._C._InferenceMode`` RAII guard, bypassing
  ``_set_grad_enabled``. That constructor is wrapped as well.
* ``Tensor.backward`` -> ``torch.autograd.backward`` and
  ``torch.autograd.grad`` all end in
  ``torch.autograd.variable.Variable._execution_engine.run_backward``. The
  public names are patched and the engine is replaced by a proxy, which also
  covers modules that bound ``grad``/``backward`` at import time. The engine
  class (``torch._C._ImperativeEngine`` and its alias
  ``torch.autograd.variable.ImperativeEngine``) is patched so a fresh engine
  cannot be built by name.
* Forward-mode AD (``torch.autograd.forward_ad``, ``torch.func.jvp``) works
  under ``no_grad``; every entry needs ``torch._C._enter_dual_level``, which
  is patched.
* ``torch.optim.Optimizer.__init__`` is patched; every built-in optimiser
  (SGD, Adam, ...) reaches it via ``super().__init__``. ``__new__`` is
  deliberately *not* patched: assigning and then deleting ``__new__`` on a
  class leaves CPython's ``tp_new`` slot pointing at ``slot_tp_new``, after
  which ``object.__new__`` rejects constructor arguments, i.e. it cannot be
  restored cleanly.
* Attribute patches cannot affect references bound before the guard started
  (``from torch._C import _set_grad_enabled``, a stored engine instance), nor
  originals reached through a patch's closure. A ``sys.monitoring`` CALL
  monitor therefore runs while the guard is active. Before every call made
  from code outside this module it (a) refuses, by raising, calls to the
  original grad-enabling bindings (``_set_grad_enabled(True)``,
  ``_InferenceMode(False)``, ``_enter_dual_level``), construction of an
  autograd engine and ``run_backward`` on any engine, from any thread; and
  (b) on the guard's thread, checks grad mode itself. If grad mode is found
  on, the monitor records a violation and forces it back off. It does not
  raise there, because a ``with`` block's ``__exit__`` is also a CALL and
  raising would skip the guard's own clean-up. The check also runs when each
  C function returns (C_RETURN), so grad mode enabled through a C trampoline
  such as ``functools.partial`` is switched off before the next operation.
  Removing or altering the monitor is detected on exit.
* Grad mode is thread-local, so a thread started inside the guard would run
  with grad enabled. A permanent audit hook (installed on first use, inert
  when no guard is active) refuses thread and process creation
  (``_thread.start_new_thread``, ``os.fork``, ``os.forkpty``,
  ``os.posix_spawn``, ``os.spawn``, ``os.exec``, ``os.system``,
  ``subprocess.Popen``) while a guard is active, whatever reference was used.
  ``_posixsubprocess.fork_exec``, which ``multiprocessing``'s spawn path calls
  directly without an audit event, is patched too.
  Threads that were already running when the guard started are not covered
  beyond the global patches and the monitor's identity checks.
* No saved-tensors hook is installed: ``torch.func`` transforms (including
  harmless ``vmap``) refuse to run while one is active.

Everything is restored on exit, including after an exception. Violations
that the guarded code catches and swallows still fail the guard on exit.
Guards nest (an inner guard can add ``extra_forbidden`` prefixes); a guard
may not be entered from a second thread while one is active.
"""

from __future__ import annotations

import _posixsubprocess
import ast
import re
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.abc import MetaPathFinder
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, TypeVar

import torch
from torch import nn

from .model import DoomLoopMLP

T = TypeVar("T")

__all__ = [
    "TrainingViolation",
    "FORBIDDEN_MODULE_PREFIXES",
    "no_training_guard",
    "run_guarded",
    "audit_model",
    "scan_source",
    "guard_active",
]


class TrainingViolation(RuntimeError):
    """A forbidden training / data-access operation on the direct-weight path."""


FORBIDDEN_MODULE_PREFIXES: tuple[str, ...] = (
    "agent_loop_lab.synth",
    "agent_loop_lab.synth_ood",
    "agent_loop_lab.datasets",
    "agent_loop_lab.eval",
    "agent_loop_lab.baselines",
)


def _matches(name: str, prefixes: Sequence[str]) -> str | None:
    """Dotted-boundary prefix match: ``a.b`` matches ``a.b`` and ``a.b.c``, not ``a.bc``."""
    for p in prefixes:
        if name == p or name.startswith(p + "."):
            return p
    return None


def _normalise_prefixes(extra: Sequence[str]) -> tuple[str, ...]:
    if isinstance(extra, str):
        raise TypeError("extra_forbidden must be a sequence of module prefixes, not a str")
    out = []
    for p in extra:
        if not isinstance(p, str) or not p.strip(".").strip():
            raise ValueError(f"invalid forbidden module prefix: {p!r}")
        out.append(p.strip().strip("."))
    return tuple(out)


# -- guard state ---------------------------------------------------------------


@dataclass
class _Patch:
    owner: Any
    name: str
    had_own: bool  # attribute was in owner.__dict__ (else restore by deleting)
    original: Any
    replacement: Any


@dataclass
class _ActiveGuard:
    thread: int
    depth: int = 1
    prefix_stack: list[tuple[str, ...]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    patches: list[_Patch] = field(default_factory=list)
    finder: Any = None
    grad_ctx: Any = None
    monitor_tool: int | None = None
    monitor_events: int = 0  # as reported by get_events (3.12 folds C_RETURN/C_RAISE into CALL)

    def forbidden(self) -> tuple[str, ...]:
        out = list(FORBIDDEN_MODULE_PREFIXES)
        for ps in self.prefix_stack:
            out.extend(ps)
        return tuple(out)


_LOCK = threading.RLock()
_ACTIVE: _ActiveGuard | None = None


def guard_active() -> bool:
    """True while a ``no_training_guard`` is active (any thread)."""
    return _ACTIVE is not None


def _violation(msg: str) -> TrainingViolation:
    """Record a violation on the active guard and return the exception to raise."""
    active = _ACTIVE
    if active is not None:
        active.violations.append(msg)
    return TrainingViolation(msg)


# -- replacements ----------------------------------------------------------------


class _ForbiddenImportFinder(MetaPathFinder):
    """Raises instead of letting any finder resolve a forbidden module."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        active = _ACTIVE
        if active is not None:
            hit = _matches(fullname, active.forbidden())
            if hit is not None:
                raise _violation(f"import of forbidden module {fullname!r} (prefix {hit!r}) under no_training_guard")
        return None

    def invalidate_caches(self) -> None:  # pragma: no cover - importlib protocol
        return None


class _EngineProxy:
    """Stands in for ``Variable._execution_engine``; blocks ``run_backward``."""

    def __init__(self, real: Any) -> None:
        self._real = real

    def run_backward(self, *args: Any, **kwargs: Any) -> Any:
        raise _violation("autograd engine run_backward called under no_training_guard")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _GuardedEngineClass:
    """Stands in for ``torch._C._ImperativeEngine``: building an engine is refused."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        raise _violation("construction of a fresh autograd engine (torch._C._ImperativeEngine) under no_training_guard")


# -- originals, bound when this module is imported ---------------------------------
#
# The monitor compares call targets against these, so it recognises the real
# bindings however the guarded code obtained them.

from torch.autograd.variable import Variable as _Variable  # noqa: E402

_ORIG_SET_GRAD_ENABLED = torch._C._set_grad_enabled
_ORIG_INFERENCE_MODE = torch._C._InferenceMode
_ORIG_ENTER_DUAL_LEVEL = torch._C._enter_dual_level
_ENGINE_TYPE: type = type(_Variable._execution_engine)  # torch._C._EngineBase
_ENGINE_RUN_BACKWARD = vars(_ENGINE_TYPE)["run_backward"]
_IS_GRAD_ENABLED = torch._C.is_grad_enabled
_THIS_FILE = __file__


def _truthy(value: Any) -> bool:
    try:
        return bool(value)
    except Exception:  # noqa: BLE001 - unknown objects count as "enable"
        return True


# -- grad-mode monitor (sys.monitoring) --------------------------------------------

_MON = sys.monitoring
_MONITOR_NAME = "agent_loop_lab.no_training_guard"
_MONITOR_EVENTS = _MON.events.CALL | _MON.events.C_RETURN | _MON.events.C_RAISE
_MONITOR_TOOL_PREFERENCE = (3, 4, 2, 5, 1, 0)  # ids 3 and 4 have no conventional owner


def _forbidden_call(callable_: Any, arg0: Any) -> str | None:
    """Name the forbidden operation if ``callable_(arg0, ...)`` is one."""
    if callable_ is _ORIG_SET_GRAD_ENABLED:
        if arg0 is not _MON.MISSING and _truthy(arg0):
            return "grad mode enabled through an unpatched torch._C._set_grad_enabled reference"
        return None
    if callable_ is _ORIG_INFERENCE_MODE:
        if arg0 is not _MON.MISSING and not _truthy(arg0):
            return "inference_mode(False) (re-enables grad) through an unpatched torch._C._InferenceMode reference"
        return None
    if callable_ is _ORIG_ENTER_DUAL_LEVEL:
        return "forward-mode AD through an unpatched torch._C._enter_dual_level reference"
    if callable_ is _ENGINE_TYPE:
        return "construction of a fresh autograd engine (torch._C._ImperativeEngine)"
    if callable_ is _ENGINE_RUN_BACKWARD:
        if isinstance(arg0, _ENGINE_TYPE):
            return "autograd engine run_backward called through an unpatched engine"
        return None
    try:
        if (
            type(callable_) is type(_ORIG_SET_GRAD_ENABLED)  # builtin bound method
            and getattr(callable_, "__name__", None) == "run_backward"
            and isinstance(getattr(callable_, "__self__", None), _ENGINE_TYPE)
        ):
            return "autograd engine run_backward called through an unpatched engine"
    except Exception:  # noqa: BLE001 - never let an odd callable break the monitor
        return None
    return None


def _monitor_call(code: Any, offset: int, callable_: Any, arg0: Any) -> None:
    """CALL event: refuse forbidden targets; on the guard thread, check grad mode."""
    active = _ACTIVE
    if active is None or code.co_filename == _THIS_FILE:
        return None
    msg = _forbidden_call(callable_, arg0)
    if msg is not None:
        raise _violation(f"{msg} under no_training_guard")
    _check_grad_mode(active)
    return None


def _monitor_return(code: Any, offset: int, callable_: Any, arg0: Any) -> None:
    """C_RETURN / C_RAISE event: grad mode may have been switched on by C code."""
    active = _ACTIVE
    if active is None or code.co_filename == _THIS_FILE:
        return None
    _check_grad_mode(active)
    return None


def _check_grad_mode(active: _ActiveGuard) -> None:
    # Record, then force grad mode off. Never raise here: this also runs for a
    # ``with`` block's ``__exit__`` call, and raising would skip the guard's exit.
    if threading.get_ident() == active.thread and _IS_GRAD_ENABLED():
        _ORIG_SET_GRAD_ENABLED(False)
        _violation("grad mode was found enabled under no_training_guard (switched on by a path the patches do not cover); forced back off")


def _install_monitor(active: _ActiveGuard) -> None:
    for tid in _MONITOR_TOOL_PREFERENCE:
        if _MON.get_tool(tid) is None:
            break
    else:
        raise TrainingViolation("cannot start no_training_guard: no free sys.monitoring tool id for the grad-mode monitor")
    _MON.use_tool_id(tid, _MONITOR_NAME)
    active.monitor_tool = tid
    _MON.register_callback(tid, _MON.events.CALL, _monitor_call)
    _MON.register_callback(tid, _MON.events.C_RETURN, _monitor_return)
    _MON.register_callback(tid, _MON.events.C_RAISE, _monitor_return)
    _MON.set_events(tid, _MONITOR_EVENTS)
    active.monitor_events = _MON.get_events(tid)


def _uninstall_monitor(active: _ActiveGuard) -> list[str]:
    """Remove the monitor. Returns tamper problems."""
    tid = active.monitor_tool
    if tid is None:
        return []
    active.monitor_tool = None
    problems: list[str] = []
    try:
        if _MON.get_tool(tid) != _MONITOR_NAME:
            # Freed (and maybe reused) by someone else: leave that tool alone.
            return [f"grad-mode monitor (sys.monitoring tool {tid}) was freed or replaced"]
        if _MON.get_events(tid) != active.monitor_events:
            problems.append(f"grad-mode monitor (sys.monitoring tool {tid}) events were changed")
        _MON.set_events(tid, 0)
        expected = {
            _MON.events.CALL: _monitor_call,
            _MON.events.C_RETURN: _monitor_return,
            _MON.events.C_RAISE: _monitor_return,
        }
        for event, fn in expected.items():
            if _MON.register_callback(tid, event, None) is not fn:
                problems.append(f"grad-mode monitor (sys.monitoring tool {tid}) callback was replaced")
        _MON.free_tool_id(tid)
    except Exception as exc:  # pragma: no cover - defensive
        problems.append(f"failed to remove grad-mode monitor: {exc!r}")
    return problems


# -- thread / process creation (audit hook) ---------------------------------------

_AUDIT_BLOCKED = {
    "_thread.start_new_thread": "thread creation",
    "os.fork": "process creation",
    "os.forkpty": "process creation",
    "os.posix_spawn": "process creation",
    "os.spawn": "process creation",
    "os.exec": "process creation",
    "os.system": "process creation",
    "subprocess.Popen": "process creation",
}
_AUDIT_HOOK_INSTALLED = False


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    # Audit hooks cannot be removed, so this stays installed for the life of
    # the interpreter. It does nothing unless a guard is active.
    if _ACTIVE is None:
        return
    what = _AUDIT_BLOCKED.get(event)
    if what is not None:
        raise _violation(
            f"{what} ({event}) under no_training_guard: grad mode is thread-local and a new"
            " thread or process would run outside the guard"
        )


def _ensure_audit_hook() -> None:
    global _AUDIT_HOOK_INSTALLED
    if not _AUDIT_HOOK_INSTALLED:
        sys.addaudithook(_audit_hook)
        _AUDIT_HOOK_INSTALLED = True


def _raiser(what: str) -> Callable[..., Any]:
    def _blocked(*args: Any, **kwargs: Any) -> Any:
        raise _violation(f"{what} called under no_training_guard")

    _blocked.__name__ = f"blocked_{what.replace('.', '_').replace('(', '').replace(')', '')}"
    _blocked.__qualname__ = _blocked.__name__
    return _blocked


def _make_patches() -> list[tuple[Any, str, Any]]:
    """(owner, attribute, replacement) triples, in installation order."""
    import torch.autograd.forward_ad  # noqa: F401  (ensure loaded so its references are real)
    from torch.autograd import grad_mode
    from torch.autograd import variable as variable_module
    from torch.autograd.variable import Variable

    orig_set_grad_C = torch._C._set_grad_enabled
    orig_inference_C = torch._C._InferenceMode
    orig_sge_init = grad_mode.set_grad_enabled.__dict__["__init__"]
    orig_sge_enter = grad_mode.set_grad_enabled.__dict__["__enter__"]

    def guarded_set_grad_enabled_C(mode: bool) -> None:
        if mode:
            raise _violation("grad mode enabled (torch._C._set_grad_enabled(True)) under no_training_guard")
        return orig_set_grad_C(mode)

    def guarded_sge_init(self: Any, mode: bool) -> None:
        if mode:
            raise _violation("torch.set_grad_enabled(True) under no_training_guard")
        return orig_sge_init(self, mode)

    def guarded_sge_enter(self: Any) -> None:
        if self.mode:
            raise _violation("torch.set_grad_enabled(True) entered under no_training_guard")
        return orig_sge_enter(self)

    def guarded_enable_grad_enter(self: Any) -> None:
        raise _violation("torch.enable_grad() under no_training_guard")

    class GuardedInferenceMode:
        # A class (not a function) so ``Optional[torch._C._InferenceMode]``
        # annotations evaluated inside torch keep working.
        def __new__(cls, mode: bool = True) -> Any:
            if not mode:
                raise _violation("inference_mode(False) (re-enables grad) under no_training_guard")
            return orig_inference_C(mode)

    def optimizer_init(self: Any, *args: Any, **kwargs: Any) -> None:
        raise _violation(f"torch.optim optimiser {type(self).__name__} constructed under no_training_guard")

    Optimizer = torch.optim.Optimizer
    return [
        (torch._C, "_set_grad_enabled", guarded_set_grad_enabled_C),
        (torch._C, "_InferenceMode", GuardedInferenceMode),
        (torch._C, "_enter_dual_level", _raiser("forward-mode AD (torch._C._enter_dual_level)")),
        (grad_mode.set_grad_enabled, "__init__", guarded_sge_init),
        (grad_mode.set_grad_enabled, "__enter__", guarded_sge_enter),
        (grad_mode.enable_grad, "__enter__", guarded_enable_grad_enter),
        (Optimizer, "__init__", optimizer_init),
        (torch.Tensor, "backward", _raiser("Tensor.backward")),
        (torch.autograd, "backward", _raiser("torch.autograd.backward")),
        (torch.autograd, "grad", _raiser("torch.autograd.grad")),
        (Variable, "_execution_engine", _EngineProxy(Variable._execution_engine)),
        (torch._C, "_ImperativeEngine", _GuardedEngineClass),
        (variable_module, "ImperativeEngine", _GuardedEngineClass),
        # multiprocessing's spawn path calls this directly and raises no audit event.
        (_posixsubprocess, "fork_exec", _raiser("process creation (_posixsubprocess.fork_exec)")),
    ]


def _install(active: _ActiveGuard) -> None:
    for owner, name, replacement in _make_patches():
        had_own = name in vars(owner)
        original = vars(owner)[name] if had_own else None
        setattr(owner, name, replacement)
        active.patches.append(_Patch(owner, name, had_own, original, replacement))


def _restore(active: _ActiveGuard) -> list[str]:
    """Undo every patch (always, in reverse order). Returns tamper problems."""
    problems: list[str] = []
    for p in reversed(active.patches):
        try:
            current = vars(p.owner).get(p.name, None)
            if current is not p.replacement:
                problems.append(f"patch on {getattr(p.owner, '__name__', p.owner)!s}.{p.name} was tampered with")
            if p.had_own:
                setattr(p.owner, p.name, p.original)
            elif p.name in vars(p.owner):
                delattr(p.owner, p.name)
        except Exception as exc:  # pragma: no cover - defensive; keep restoring the rest
            problems.append(f"failed to restore {p.name}: {exc!r}")
    active.patches.clear()
    if active.finder is not None:
        found = False
        for i in range(len(sys.meta_path) - 1, -1, -1):
            if sys.meta_path[i] is active.finder:
                del sys.meta_path[i]
                found = True
        if not found:
            problems.append("forbidden-import finder was removed from sys.meta_path")
        active.finder = None
    return problems


def _preimported(prefixes: Sequence[str]) -> list[str]:
    return sorted(n for n in list(sys.modules) if _matches(n, prefixes) is not None)


@contextmanager
def no_training_guard(extra_forbidden: Sequence[str] = ()) -> Iterator[None]:
    """Run a block with training and forbidden imports made impossible.

    Raises ``TrainingViolation`` on entry if a forbidden module is already
    imported, during the block on any forbidden operation, and on exit if a
    violation was swallowed, grad mode ended up enabled, a forbidden module
    appeared in ``sys.modules`` or a patch was tampered with.
    """
    global _ACTIVE
    extra = _normalise_prefixes(extra_forbidden)
    me = threading.get_ident()

    with _LOCK:
        outer = _ACTIVE
        if outer is not None:
            if outer.thread != me:
                raise TrainingViolation("no_training_guard is already active in another thread; refusing to nest")
            combined = outer.forbidden() + extra
            pre = _preimported(combined)
            if pre:
                raise TrainingViolation(f"forbidden modules already imported on guard entry: {pre}")
            outer.depth += 1
            outer.prefix_stack.append(extra)
            mark = len(outer.violations)
            nested = True
        else:
            pre = _preimported(FORBIDDEN_MODULE_PREFIXES + extra)
            if pre:
                raise TrainingViolation(f"forbidden modules already imported on guard entry: {pre}")
            active = _ActiveGuard(thread=me, prefix_stack=[extra])
            mark = 0
            nested = False
            try:
                _ensure_audit_hook()
                active.grad_ctx = torch.no_grad()
                active.grad_ctx.__enter__()
                active.finder = _ForbiddenImportFinder()
                sys.meta_path.insert(0, active.finder)
                _ACTIVE = active
                _install(active)
                _install_monitor(active)  # last: the guard's own set-up is not monitored
            except BaseException:
                _ACTIVE = None
                _uninstall_monitor(active)
                _restore(active)
                if active.grad_ctx is not None:
                    active.grad_ctx.__exit__(None, None, None)
                raise

    body_exc: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        body_exc = exc
        raise
    finally:
        with _LOCK:
            active = _ACTIVE
            assert active is not None
            problems: list[str] = []
            swallowed = active.violations[mark:]
            if swallowed:
                problems.append(f"{len(swallowed)} training violation(s) occurred inside the guard: {swallowed}")
            if torch.is_grad_enabled():
                problems.append("grad mode is enabled at guard exit")
            if nested:
                active.prefix_stack.pop()
                active.depth -= 1
                # Violations stay recorded so the outer guard also fails even if
                # the outer body swallows the inner guard's exception.
            else:
                late = _preimported(active.forbidden())
                if late:
                    problems.append(f"forbidden modules present in sys.modules at guard exit: {late}")
                # Order matters: remove the monitor and restore torch *before*
                # leaving no_grad, because no_grad.__exit__ calls
                # set_grad_enabled(prev) and prev is normally True, which the
                # patched path and the monitor would both reject.
                _ACTIVE = None
                problems.extend(_uninstall_monitor(active))
                problems.extend(_restore(active))
                active.grad_ctx.__exit__(None, None, None)
            if problems and body_exc is None:
                raise TrainingViolation("; ".join(problems))


def run_guarded(fn: Callable[[], T]) -> T:
    """Call ``fn()`` under ``no_training_guard``; audit the result if it is a model."""
    with no_training_guard():
        result = fn()
        if isinstance(result, nn.Module):
            audit_model(result)
    return result


def audit_model(model: DoomLoopMLP) -> None:
    """Raise ``TrainingViolation`` if any parameter requires grad, holds a
    ``.grad`` or is a non-leaf (was produced by a recorded graph)."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"audit_model expects an nn.Module, got {type(model).__name__}")
    problems: list[str] = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            problems.append(f"{name}: requires_grad=True")
        if p.grad is not None:
            problems.append(f"{name}: has .grad")
        if p.grad_fn is not None:
            problems.append(f"{name}: has grad_fn (autograd graph)")
    for name, b in model.named_buffers():
        if b.requires_grad or b.grad_fn is not None:
            problems.append(f"buffer {name}: participates in autograd")
    if problems:
        raise TrainingViolation("model audit failed: " + "; ".join(problems))


# -- static source audit ---------------------------------------------------------

_FORBIDDEN_IMPORT_ROOTS = (
    "torch.optim",
    "torch.autograd",
    "torch.func",
    "torch._C",
    "functorch",
    "scipy.optimize",
    "sklearn",
)
_RANDOM_MODULES = ("numpy.random", "torch.random", "torch.distributions")
_FORBIDDEN_LEAVES = ("synth", "synth_ood", "datasets", "eval", "baselines")
_TORCH_FORBIDDEN_SUBMODULES = {"optim", "autograd", "func", "_C"}
_RANDOM_TORCH_FUNCS = {
    "rand",
    "randn",
    "randint",
    "randperm",
    "rand_like",
    "randn_like",
    "randint_like",
    "normal",
    "bernoulli",
    "multinomial",
    "poisson",
    "Generator",
    "manual_seed",
    "seed",
    "initial_seed",
    "get_rng_state",
    "set_rng_state",
}
# In-place tensor methods that fill a tensor from the RNG.
_RNG_INPLACE_METHODS = {
    "normal_",
    "uniform_",
    "random_",
    "bernoulli_",
    "exponential_",
    "cauchy_",
    "log_normal_",
    "geometric_",
}
# torch.nn.init functions that are deterministic; every other initialiser draws from the RNG.
_DETERMINISTIC_INIT = {"zeros_", "ones_", "constant_", "eye_", "dirac_", "calculate_gain"}
_AUTOGRAD_ATTRS = {"grad", "backward", "functional", "forward_ad", "Function"}
_FORBIDDEN_PATH_RE = re.compile(r"(?:^|[/\\])(datasets|synth[A-Za-z0-9_\-]*)(?:[/\\]|$)")
_IMPORT_FUNCS = {"import_module", "__import__"}
_PATH_JOIN_FUNCS = {"join", "Path", "PurePath", "PosixPath", "PurePosixPath", "WindowsPath", "PureWindowsPath"}


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _module_violation(mod: str) -> str | None:
    hit = _matches(mod, FORBIDDEN_MODULE_PREFIXES) or _matches(mod, _FORBIDDEN_IMPORT_ROOTS)
    if hit:
        return f"import of forbidden module {mod!r}"
    if mod == "random" or mod.startswith("random.") or _matches(mod, _RANDOM_MODULES):
        return f"import of random-number module {mod!r} (search is forbidden)"
    return None


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            ids.add(id(node.value))
    return ids


def _fold_str(node: ast.AST) -> str | None:
    """Constant-fold a string or path built only from literal pieces.

    Handles ``'a' + 'b'``, f-strings of literals, ``Path('a') / 'b'``,
    ``os.path.join('a', 'b')`` and ``Path('a', 'b')``. Path pieces are joined
    with ``/``. Returns ``None`` if any piece is not a literal.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
        left, right = _fold_str(node.left), _fold_str(node.right)
        if left is None or right is None:
            return None
        return left + right if isinstance(node.op, ast.Add) else f"{left}/{right}"
    if isinstance(node, ast.JoinedStr):
        out = []
        for part in node.values:
            if isinstance(part, ast.FormattedValue):
                if part.format_spec is not None:
                    return None
                piece = _fold_str(part.value)
            else:
                piece = _fold_str(part)
            if piece is None:
                return None
            out.append(piece)
        return "".join(out)
    if isinstance(node, ast.Call) and not node.keywords and node.args:
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else None
        pieces = [_fold_str(a) for a in node.args]
        if any(x is None for x in pieces):
            return None
        if name in _PATH_JOIN_FUNCS:
            return "/".join(pieces)  # type: ignore[arg-type]
        if name == "joinpath" and isinstance(fn, ast.Attribute):
            head = _fold_str(fn.value)
            return None if head is None else "/".join([head, *pieces])  # type: ignore[list-item]
    return None


def _bound_names(tree: ast.AST, module: str, names: set[str] | None) -> dict[str, str]:
    """Local names bound by ``from <module> import X [as Y]`` -> ``module.X``."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not node.level and node.module == module:
            for alias in node.names:
                if names is None or alias.name in names:
                    out[alias.asname or alias.name] = f"{module}.{alias.name}"
    return out


def scan_source(path: str | Path) -> list[str]:
    """Static AST audit of a weight-build source file.

    Returns ``"<path>:<line>: <description>"`` strings sorted by line; an
    empty list means no findings. Unparseable source is itself a finding.

    This is a pre-check alongside the runtime guard, not a proof: code built
    to evade it (``exec``, ``getattr`` with computed names) can get past any
    AST scan, and the runtime guard is what has to catch it.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(p))
    except SyntaxError as exc:
        return [f"{p}:{exc.lineno or 0}: cannot parse source ({exc.msg}); audit failed closed"]

    found: set[tuple[int, str]] = set()

    def flag(node: ast.AST, msg: str) -> None:
        found.add((getattr(node, "lineno", 0), msg))

    docstrings = _docstring_nodes(tree)
    random_aliases = _bound_names(tree, "torch", _RANDOM_TORCH_FUNCS)
    random_aliases.update(
        {k: v for k, v in _bound_names(tree, "torch.nn.init", None).items() if v.rsplit(".", 1)[1] not in _DETERMINISTIC_INIT}
    )
    # Only the outermost foldable expression is reported, once.
    inner_folds: set[int] = set()

    for node in ast.walk(tree):
        # imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                v = _module_violation(alias.name)
                if v:
                    flag(node, v)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = (node.module or "").split(".") + [a.name for a in node.names]
                bad = [x for x in parts if x and (x in _FORBIDDEN_LEAVES or x.startswith("synth"))]
                if bad:
                    flag(node, f"relative import of forbidden package {bad[0]!r}")
                continue
            mod = node.module or ""
            v = _module_violation(mod)
            if v:
                flag(node, v)
            for alias in node.names:
                full = f"{mod}.{alias.name}"
                if mod == "torch" and alias.name in _TORCH_FORBIDDEN_SUBMODULES:
                    flag(node, f"import of forbidden module 'torch.{alias.name}'")
                elif mod == "torch" and alias.name in ("random", "distributions"):
                    flag(node, f"import of random-number module 'torch.{alias.name}' (search is forbidden)")
                elif mod in ("numpy", "np") and alias.name == "random":
                    flag(node, "import of random-number module 'numpy.random' (search is forbidden)")
                elif full in random_aliases.values():
                    flag(node, f"import of {full} (random search is forbidden)")
                elif _matches(full, FORBIDDEN_MODULE_PREFIXES) and not v:
                    flag(node, f"import of forbidden module {full!r}")

        # attribute references
        elif isinstance(node, ast.Attribute):
            base = _dotted(node.value)
            last = base.split(".")[-1] if base is not None else None
            if node.attr == "optim":
                flag(node, "use of torch.optim ('optim.')")
            elif last == "autograd" and node.attr in _AUTOGRAD_ATTRS:
                flag(node, f"use of autograd.{node.attr}")
            elif node.attr == "random" and base in ("np", "numpy", "torch"):
                flag(node, f"use of {base}.random (search is forbidden)")
            elif base == "torch" and node.attr in _RANDOM_TORCH_FUNCS:
                flag(node, f"use of torch.{node.attr} (random search is forbidden)")
            elif base == "torch" and node.attr == "distributions":
                flag(node, "use of torch.distributions (random search is forbidden)")
            elif last == "init" and node.attr not in _DETERMINISTIC_INIT:
                flag(node, f"use of nn.init.{node.attr} (random initialiser; search is forbidden)")
            elif node.attr in _RNG_INPLACE_METHODS:
                flag(node, f"use of in-place RNG method .{node.attr} (random search is forbidden)")
            elif base == "torch" and node.attr == "func":
                flag(node, "use of torch.func (functional autograd)")
            elif base == "torch" and node.attr == "_C":
                flag(node, "use of torch._C (private autograd / grad-mode bindings)")
            elif node.attr == "_execution_engine":
                flag(node, "use of the autograd engine (_execution_engine)")
            elif node.attr == "requires_grad" and isinstance(node.ctx, ast.Store):
                flag(node, "assignment to .requires_grad")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id == "optim":
                flag(node, "use of torch.optim ('optim.')")
            elif node.id in random_aliases:
                flag(node, f"use of {random_aliases[node.id]} (random search is forbidden)")

        # calls
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else None
            if name == "backward":
                flag(node, "call to .backward(")
            elif name == "run_backward":
                flag(node, "call to run_backward( (autograd engine)")
            elif name == "step" and isinstance(fn, ast.Attribute):
                flag(node, "call to .step( (optimiser step)")
            elif name == "requires_grad_":
                arg = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "requires_grad"), None)
                if arg is None or not (isinstance(arg, ast.Constant) and arg.value is False):
                    flag(node, "call to requires_grad_(True)")
            elif name == "enable_grad":
                flag(node, "call to enable_grad()")
            elif name in ("set_grad_enabled", "_set_grad_enabled"):
                arg = node.args[0] if node.args else None
                if not (isinstance(arg, ast.Constant) and arg.value is False):
                    flag(node, f"call to {name}(True or non-literal)")
            elif name in ("inference_mode", "_InferenceMode") and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value is False:
                flag(node, f"call to {name}(False) (re-enables grad)")
            elif name == "_enter_dual_level":
                flag(node, "call to _enter_dual_level (forward-mode AD)")
            elif name == "Parameter" and not any(k.arg == "requires_grad" for k in node.keywords):
                arg = node.args[1] if len(node.args) > 1 else None
                if not (isinstance(arg, ast.Constant) and arg.value is False):
                    flag(node, "nn.Parameter(...) without requires_grad=False (defaults to requires_grad=True)")
            elif name in _IMPORT_FUNCS and node.args and (modname := _fold_str(node.args[0])) is not None:
                v = _module_violation(modname)
                if v:
                    flag(node, f"dynamic {v}")
            elif (
                name in ("getattr", "setattr", "delattr")
                and isinstance(fn, ast.Name)
                and len(node.args) >= 2
                and _dotted(node.args[0]) == "torch"
                and (attr := _fold_str(node.args[1])) is not None
                and (attr in _TORCH_FORBIDDEN_SUBMODULES or attr in _RANDOM_TORCH_FUNCS or attr in ("random", "distributions"))
            ):
                flag(node, f"use of torch.{attr} via {name}()")
            for kw in node.keywords:
                if kw.arg == "requires_grad" and name != "requires_grad_":
                    if not (isinstance(kw.value, ast.Constant) and kw.value.value is False):
                        flag(node, "requires_grad=True")

        # string literals that name forbidden data locations
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            m = _FORBIDDEN_PATH_RE.search(node.value)
            if m:
                flag(node, f"reference to forbidden data path {node.value!r} ({m.group(1)})")
        # ...and strings / paths assembled from literal pieces
        elif isinstance(node, (ast.BinOp, ast.JoinedStr, ast.Call)) and id(node) not in inner_folds:
            folded = _fold_str(node)
            if folded is not None:
                for sub in ast.walk(node):
                    if sub is not node:
                        inner_folds.add(id(sub))
                m = _FORBIDDEN_PATH_RE.search(folded)
                if m:
                    flag(node, f"reference to forbidden data path {folded!r} ({m.group(1)})")

    return [f"{p}:{line}: {msg}" for line, msg in sorted(found)]
