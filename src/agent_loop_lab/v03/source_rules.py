"""Static source rules for the v0.3 weight author's files (spec v0.3 §9.1, §11 "Anti-training").

Rule sets are separate functions so that the runner and the session audit can
combine them:

* ``forbidden_import_findings``: harness packages (generator, OOD generator,
  datasets, eval, baselines and anything else under ``agent_loop_lab``) and
  generator / dataset paths. Used on every author ``.py`` file.
* ``rng_findings``: random-number APIs (``random``, ``numpy.random``,
  ``secrets``, ``torch.rand*`` / ``normal`` / ``bernoulli`` / ``multinomial`` /
  ``Generator`` / ``manual_seed``, in-place RNG methods, random
  ``nn.init`` initialisers). Used on every author ``.py`` file.
* ``literal_size_findings``: list / tuple / set / dict literal with more than
  256 elements, str / bytes literal (docstrings included) longer than 512
  characters, file larger than 200 KB.
* ``build_weights_findings``: everything §9.1 says about ``build_weights.py``:
  the import whitelist, forbidden modules and builtins, the static half of the
  compute ban, tokeniser calls on literal word lists only, the literal limits
  and a module-level ``def build()``.
* ``reference_policy_findings``: standard library only (no ``tokenizer``,
  ``model``, ``torch`` or ``numpy``), no I/O, no global state (no ``global`` /
  ``nonlocal``, no mutation of a module-level container or attribute from a
  function), a module-level ``def reference_policy(...)``.
* ``introspection_findings`` (both deliverables, strict): every dunder name or
  attribute except a small allowlist (``__name__``, ``__init__``, ...;
  ``__file__``, ``__globals__``, ``__builtins__``, ``__subclasses__``,
  ``__dict__`` and the rest are refused), frame and traceback internals
  (``f_globals``, ``tb_frame``, ...), string constants that spell such a name
  (after folding ``'a' + 'b'`` concatenations), ``getattr`` / ``setattr`` /
  ``delattr`` / ``hasattr`` / ``operator.attrgetter`` / ``methodcaller`` with a
  name that is not a literal, any other use of those functions, ``vars`` /
  ``locals`` / ``globals``, and dunder method definitions (``__del__``, ...)
  other than ``__init__`` / ``__post_init__`` / ``__repr__``.
* ``author_script_findings``: the forbidden-import and RNG rules, plus file
  readers that bypass the ``./py`` I/O policy (``from_file``, ``ctypes``,
  ``_ctypes``, ``mmap``) and the interpreter-internals subset of the
  introspection rules (``__globals__``, ``__builtins__``, ``__subclasses__``,
  frame internals, computed ``getattr`` names).

Every finding is ``"<file name>:<line>: <description>"``. This is a pre-check
beside the runtime guards, not a proof: computed ``getattr`` names or ``exec``
of generated text get past any AST scan, and the runtime guard has to catch
them.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Iterable

__all__ = [
    "ALLOWED_DUNDERS",
    "INTERNALS_ATTRS",
    "introspection_findings",
    "mutable_state_findings",
    "BUILD_ALLOWED_IMPORTS",
    "BUILD_FORBIDDEN_MODULES",
    "POLICY_BANNED_MODULES",
    "MAX_COLLECTION_ELEMENTS",
    "MAX_STRING_CHARS",
    "MAX_FILE_BYTES",
    "forbidden_import_findings",
    "rng_findings",
    "literal_size_findings",
    "build_weights_findings",
    "reference_policy_findings",
    "author_script_findings",
    "check_build_weights",
    "check_reference_policy",
]

MAX_COLLECTION_ELEMENTS = 256
MAX_STRING_CHARS = 512
MAX_FILE_BYTES = 200 * 1024

# Top-level modules build_weights.py may import (spec §9.1 plus harmless stdlib for closed-form construction).
BUILD_ALLOWED_IMPORTS = frozenset(
    {"__future__", "torch", "math", "hashlib", "itertools", "functools", "collections", "dataclasses", "typing",
     "string", "re", "model", "tokenizer"}
)
# torch submodules build_weights.py may import (everything else under torch is refused).
BUILD_ALLOWED_TORCH_SUBMODULES = frozenset({"torch", "torch.nn"})
BUILD_FORBIDDEN_MODULES = frozenset(
    {"random", "numpy", "base64", "struct", "zlib", "pickle", "array", "marshal", "io", "os", "sys", "subprocess",
     "pathlib", "importlib", "renderer", "reference_policy", "secrets", "builtins", "ctypes", "shutil", "tempfile",
     "socket", "gzip", "bz2", "lzma", "zipfile", "tarfile", "codecs", "binascii", "json", "csv", "sqlite3",
     "shelve", "mmap", "threading", "multiprocessing", "concurrent", "asyncio", "runpy", "inspect", "gc",
     "agent_loop_lab"}
)
# reference_policy.py: standard library only, minus these (I/O, processes, non-determinism, introspection).
POLICY_BANNED_MODULES = frozenset(
    {"os", "sys", "subprocess", "socket", "importlib", "pathlib", "io", "builtins", "ctypes", "threading", "_thread",
     "multiprocessing", "concurrent", "asyncio", "random", "secrets", "time", "datetime", "uuid", "pickle", "marshal",
     "shelve", "shutil", "tempfile", "glob", "urllib", "http", "ssl", "select", "selectors", "signal", "mmap",
     "fcntl", "sqlite3", "csv", "gzip", "bz2", "lzma", "zipfile", "tarfile", "logging", "atexit", "gc", "inspect",
     "runpy", "pkgutil", "zipimport", "site", "sysconfig", "platform", "getpass", "webbrowser", "resource",
     "tracemalloc", "faulthandler", "code", "codeop", "base64", "struct", "zlib", "array", "fileinput", "linecache",
     "tokenize", "pdb", "trace", "cProfile", "profile", "timeit", "weakref", "copyreg", "_ctypes", "_io",
     "_posixsubprocess", "posix", "nt", "pty", "termios", "readline", "rlcompleter", "ensurepip", "venv",
     "zipapp", "_imp", "_frozen_importlib", "_frozen_importlib_external"}
)
POLICY_NON_STDLIB_REFUSED = frozenset({"tokenizer", "model", "torch", "numpy", "renderer", "rawop", "build_weights"})

IO_BUILTINS = frozenset({"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"})
_IO_METHODS = frozenset(
    {"load", "loadtxt", "genfromtxt", "fromfile", "memmap", "from_file", "save", "savez", "savez_compressed",
     "savetxt", "tofile", "read_text", "read_bytes", "write_text", "write_bytes", "save_artifact", "load_artifact"}
)

# -- interpreter internals (both deliverables strict; author scripts a subset) ---------------
# Dunder names that do not reach interpreter internals.
ALLOWED_DUNDERS = frozenset({"__name__", "__doc__", "__all__", "__init__", "__post_init__", "__qualname__",
                             "__module__", "__main__", "__repr__", "__annotations__", "__slots__"})
ALLOWED_DUNDER_DEFS = frozenset({"__init__", "__post_init__", "__repr__"})
# Attributes (dunder or not) that reach globals, builtins, frames, code objects or the class graph.
INTERNALS_ATTRS = frozenset(
    {"__globals__", "__builtins__", "__subclasses__", "__bases__", "__base__", "__mro__", "__code__",
     "__closure__", "__getattribute__", "__self__", "__func__", "__dict__", "__class__", "__loader__", "__spec__",
     "__import__", "__file__", "__cached__", "__path__", "__traceback__", "__reduce__", "__reduce_ex__",
     "__wrapped__", "__defaults__", "__kwdefaults__", "f_globals", "f_builtins", "f_locals", "f_back", "f_code",
     "tb_frame", "tb_next", "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code", "co_consts",
     "co_names"}
)
# The subset flagged in the author's scratch scripts (spec §8 session audit).
SCRIPT_INTERNALS = frozenset(
    {"__globals__", "__builtins__", "__subclasses__", "__bases__", "__base__", "__mro__", "__code__",
     "__closure__", "__getattribute__", "__self__", "f_globals", "f_builtins", "f_locals", "f_back", "tb_frame",
     "gi_frame", "cr_frame", "ag_frame"}
)
ATTR_FUNCS = frozenset({"getattr", "setattr", "delattr", "hasattr"})
ATTR_FACTORIES = frozenset({"attrgetter", "methodcaller"})
NAMESPACE_FUNCS = frozenset({"vars", "locals", "globals"})
SCRIPT_READER_ATTRS = frozenset({"from_file", "PyTorchFileReader", "import_ir_module"})
SCRIPT_READER_MODULES = frozenset({"ctypes", "_ctypes", "mmap"})
_DUNDER_RE = re.compile(r"__[A-Za-z0-9_]+__")
_MUTATING_METHODS = frozenset(
    {"append", "extend", "insert", "pop", "popitem", "clear", "update", "setdefault", "add", "discard", "remove",
     "sort", "reverse", "appendleft", "extendleft", "popleft", "rotate", "difference_update",
     "intersection_update", "symmetric_difference_update", "__setitem__", "__delitem__", "write", "fromkeys"}
)
_MUTABLE_FACTORIES = frozenset({"list", "dict", "set", "bytearray", "defaultdict", "Counter", "OrderedDict", "deque",
                                "ChainMap", "array", "SimpleNamespace"})

# Static half of the compute ban (spec §9.1). Attribute or name uses of these are refused in build_weights.py.
COMPUTE_ATTRS = frozenset(
    {"matmul", "mm", "bmm", "addmm", "addmm_", "addr", "addr_", "ger", "baddbmm", "baddbmm_", "addbmm", "addbmm_",
     "addmv", "addmv_", "mv", "dot", "vdot", "inner", "tensordot", "einsum", "chain_matmul", "softmax",
     "log_softmax", "bilinear", "linalg", "functional", "forward", "forward_batch", "activations", "_run",
     "__matmul__", "__rmatmul__", "__imatmul__", "ops", "_C", "_VariableFunctions", "_TensorBase"}
)
COMPUTE_SUBSTRINGS = ("matmul", "linalg", "einsum", "softmax")

TORCH_RNG_NAMES = frozenset(
    {"rand", "randn", "randint", "randperm", "rand_like", "randn_like", "randint_like", "normal", "bernoulli",
     "multinomial", "poisson", "Generator", "manual_seed", "seed", "set_rng_state", "get_rng_state", "initial_seed",
     "fork_rng", "random", "distributions"}
)
RNG_INPLACE_METHODS = frozenset(
    {"normal_", "uniform_", "random_", "bernoulli_", "exponential_", "cauchy_", "log_normal_", "geometric_"}
)
DETERMINISTIC_INIT = frozenset({"zeros_", "ones_", "constant_", "eye_", "dirac_", "calculate_gain"})
_RNG_MODULES = ("random", "secrets", "numpy.random", "torch.random", "torch.distributions")
_HARNESS_LEAVES = ("synth", "synth_ood", "datasets", "eval", "baselines", "oracle_config", "labeller", "counterfactual")


def _fmt(name: str, found: Iterable[tuple[int, str]]) -> list[str]:
    return [f"{name}:{line}: {msg}" for line, msg in sorted(set(found))]


def _matches(mod: str, prefixes: Iterable[str]) -> bool:
    return any(mod == p or mod.startswith(p + ".") for p in prefixes)


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _imported_modules(tree: ast.AST) -> list[tuple[ast.AST, str, list[str]]]:
    """(node, module, imported names) for every import; relative imports give module '.'."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((node, a.name, []))
        elif isinstance(node, ast.ImportFrom):
            mod = "." * node.level + (node.module or "")
            out.append((node, mod, [a.name for a in node.names]))
    return out


def _parse(path: Path) -> tuple[ast.Module | None, list[tuple[int, str]]]:
    try:
        return ast.parse(path.read_bytes(), filename=path.name), []
    except SyntaxError as exc:
        return None, [(exc.lineno or 0, f"cannot parse source ({exc.msg})")]


# -- shared rule sets -------------------------------------------------------------


def forbidden_import_findings(tree: ast.AST) -> list[tuple[int, str]]:
    """Harness-package imports (``agent_loop_lab`` and the generator / dataset / eval leaves) and data paths."""
    found: list[tuple[int, str]] = []
    for node, mod, names in _imported_modules(tree):
        root = mod.lstrip(".").split(".")[0]
        parts = [p for p in mod.lstrip(".").split(".") if p] + names
        if root == "agent_loop_lab":
            found.append((node.lineno, f"import of harness package {mod!r} (the author sees only the packet)"))
        elif any(p in _HARNESS_LEAVES or p.startswith("synth") for p in parts):
            found.append((node.lineno, f"import naming a generator/dataset/eval module ({mod!r} {names})"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else None
            if name in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant):
                v = str(node.args[0].value)
                if v.split(".")[0] == "agent_loop_lab" or any(p in _HARNESS_LEAVES or p.startswith("synth") for p in v.split(".")):
                    found.append((node.lineno, f"dynamic import of {v!r}"))
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for seg in node.value.replace("\\", "/").split("/"):
                if seg in ("datasets", "synth", "synth_ood", "weight_program") or seg.startswith("HypaNet"):
                    if "/" in node.value:
                        found.append((node.lineno, f"string names a harness/data path ({node.value[:80]!r})"))
                        break
    return found


def rng_findings(tree: ast.AST) -> list[tuple[int, str]]:
    """Random-number APIs (spec §9.1, §11): any import or use."""
    found: list[tuple[int, str]] = []
    torch_rng_aliases: dict[str, str] = {}
    for node, mod, names in _imported_modules(tree):
        m = mod.lstrip(".")
        if _matches(m, _RNG_MODULES):
            found.append((node.lineno, f"import of random-number module {m!r}"))
        if m in ("numpy", "torch") and "random" in names:
            found.append((node.lineno, f"import of {m}.random"))
        if m == "torch" and "distributions" in names:
            found.append((node.lineno, "import of torch.distributions"))
        if m == "torch":
            for n in names:
                if n in TORCH_RNG_NAMES:
                    torch_rng_aliases[n] = f"torch.{n}"
                    found.append((node.lineno, f"import of torch.{n} (random numbers are forbidden)"))
        if m == "torch.nn.init" and any(n not in DETERMINISTIC_INIT for n in names):
            found.append((node.lineno, "import of a random torch.nn.init initialiser"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            base = _dotted(node.value)
            last = base.split(".")[-1] if base else None
            if node.attr == "random" and base in ("np", "numpy", "torch"):
                found.append((node.lineno, f"use of {base}.random"))
            elif base == "torch" and node.attr in TORCH_RNG_NAMES:
                found.append((node.lineno, f"use of torch.{node.attr} (random numbers are forbidden)"))
            elif node.attr in RNG_INPLACE_METHODS:
                found.append((node.lineno, f"use of in-place RNG method .{node.attr}"))
            elif last == "init" and node.attr not in DETERMINISTIC_INIT:
                found.append((node.lineno, f"use of nn.init.{node.attr} (random initialiser)"))
            elif base in ("random", "secrets"):
                found.append((node.lineno, f"use of {base}.{node.attr}"))
            elif node.attr in ("default_rng", "RandomState", "SystemRandom"):
                found.append((node.lineno, f"use of {node.attr}"))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in torch_rng_aliases:
            found.append((node.lineno, f"use of {torch_rng_aliases[node.id]}"))
        elif isinstance(node, ast.keyword) and node.arg == "generator":
            found.append((getattr(node.value, "lineno", 0), "generator= argument (random numbers are forbidden)"))
    return found


def literal_size_findings(tree: ast.AST, source_bytes: int | None = None) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    if source_bytes is not None and source_bytes > MAX_FILE_BYTES:
        found.append((0, f"file is {source_bytes} bytes; the limit is {MAX_FILE_BYTES} bytes (200 KB)"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and len(node.elts) > MAX_COLLECTION_ELEMENTS:
            kind = type(node).__name__.lower()
            found.append((node.lineno, f"{kind} literal with {len(node.elts)} elements (limit {MAX_COLLECTION_ELEMENTS})"))
        elif isinstance(node, ast.Dict) and len(node.keys) > MAX_COLLECTION_ELEMENTS:
            found.append((node.lineno, f"dict literal with {len(node.keys)} entries (limit {MAX_COLLECTION_ELEMENTS})"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)) and len(node.value) > MAX_STRING_CHARS:
            kind = "str" if isinstance(node.value, str) else "bytes"
            found.append((node.lineno, f"{kind} literal of {len(node.value)} characters (limit {MAX_STRING_CHARS}; docstrings count)"))
    return found


def _io_findings(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in IO_BUILTINS:
                found.append((node.lineno, f"call to builtin {fn.id}() (file, console or dynamic-code access)"))
            elif isinstance(fn, ast.Attribute) and fn.attr in _IO_METHODS:
                found.append((node.lineno, f"call to .{fn.attr}() (file reads and writes are not allowed)"))
        elif isinstance(node, ast.Name) and node.id in ("__builtins__", "globals", "__loader__", "__spec__"):
            found.append((node.lineno, f"use of {node.id} (interpreter internals)"))
    return found


def _fold_str(node: ast.AST) -> str | None:
    """The value of a str constant, a ``+`` concatenation, an f-string or ``"".join`` of foldable parts."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _fold_str(node.left), _fold_str(node.right)
        return a + b if a is not None and b is not None else None
    if isinstance(node, ast.JoinedStr):
        parts = [_fold_str(v) if not isinstance(v, ast.FormattedValue) else _fold_str(v.value) for v in node.values]
        return "".join(parts) if all(p is not None for p in parts) else None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join"
            and len(node.args) == 1 and isinstance(node.args[0], (ast.List, ast.Tuple))):
        sep = _fold_str(node.func.value)
        parts = [_fold_str(e) for e in node.args[0].elts]
        return sep.join(parts) if sep is not None and all(p is not None for p in parts) else None  # type: ignore[arg-type]
    return None


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    out: dict[int, ast.AST] = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            out[id(c)] = p
    return out


def _loop_literal_strs(name: str, node: ast.AST, parents: dict[int, ast.AST], module_lits: dict[str, list[str]]) -> list[str] | None:
    """The strings a loop variable ``name`` ranges over, when the loop iterates a literal of str constants."""
    cur = parents.get(id(node))
    while cur is not None:
        gens: list[tuple[ast.AST, ast.AST]] = []
        if isinstance(cur, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            gens = [(g.target, g.iter) for g in cur.generators]
        elif isinstance(cur, ast.For):
            gens = [(cur.target, cur.iter)]
        for target, it in gens:
            if isinstance(target, ast.Name) and target.id == name:
                if isinstance(it, (ast.List, ast.Tuple)):
                    if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in it.elts):
                        return [e.value for e in it.elts]  # type: ignore[union-attr]
                    return None
                if isinstance(it, ast.Name) and it.id in module_lits:
                    return module_lits[it.id]
                return None
        cur = parents.get(id(cur))
    return None


def _module_str_literals(tree: ast.Module) -> dict[str, list[str]]:
    """Module-level names assigned exactly once, to a list/tuple of str constants."""
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name,)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            counts[node.id] = counts.get(node.id, 0) + 1
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = node.value
            if (isinstance(v, (ast.List, ast.Tuple)) and counts.get(node.targets[0].id) == 1
                    and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in v.elts)):
                out[node.targets[0].id] = [e.value for e in v.elts]  # type: ignore[union-attr]
    return out


def _bad_name(name: str, strict: bool) -> bool:
    if name in (INTERNALS_ATTRS if strict else SCRIPT_INTERNALS):
        return True
    return strict and bool(_DUNDER_RE.fullmatch(name)) and name not in ALLOWED_DUNDERS


def introspection_findings(tree: ast.Module, strict: bool = True) -> list[tuple[int, str]]:
    """Reaching interpreter internals (globals, builtins, frames, the class graph) or computing attribute names.

    ``strict`` (``build_weights.py``, ``reference_policy.py``): the full rule set. Otherwise (author scripts):
    :data:`SCRIPT_INTERNALS` and computed attribute names only."""
    found: list[tuple[int, str]] = []
    why = "no access to interpreter internals"
    parents = _parents(tree)
    lits = _module_str_literals(tree)
    call_funcs: set[int] = set()

    def literal_name_ok(arg: ast.AST) -> bool:
        # a plain str constant (a concatenation is a computed name), or the loop variable over a literal of them
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            v = arg.value
            return not _bad_name(v, strict) and not (strict and "__" in v)
        if isinstance(arg, ast.Name):
            vals = _loop_literal_strs(arg.id, arg, parents, lits)
            return vals is not None and not any(_bad_name(x, strict) or (strict and "__" in x) for x in vals)
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            fname = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
            if fname in ATTR_FUNCS and isinstance(fn, ast.Name):
                call_funcs.add(id(fn))
                if len(node.args) < 2 or not literal_name_ok(node.args[1]):
                    found.append((node.lineno, f"{fname}() with a computed or internal attribute name ({why})"))
            elif fname in ATTR_FACTORIES:
                call_funcs.add(id(fn))
                if not node.args or not all(literal_name_ok(a) for a in node.args[:1 if fname == "methodcaller" else None]):
                    found.append((node.lineno, f"{fname}() with a computed or internal attribute name ({why})"))
            elif strict and fname in NAMESPACE_FUNCS and isinstance(fn, ast.Name):
                call_funcs.add(id(fn))
                found.append((node.lineno, f"call to {fname}() ({why})"))
        elif isinstance(node, ast.Attribute):
            if _bad_name(node.attr, strict):
                found.append((node.lineno, f"use of attribute {node.attr!r} ({why})"))
        elif isinstance(node, ast.Name):
            if _bad_name(node.id, strict):
                found.append((node.lineno, f"use of {node.id!r} ({why})"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if strict and _DUNDER_RE.fullmatch(node.name) and node.name not in ALLOWED_DUNDER_DEFS:
                found.append((node.lineno, f"definition of special method {node.name} (only __init__, __post_init__ and "
                              "__repr__ may be defined)"))
        elif isinstance(node, ast.alias):
            leaf = node.name.split(".")[-1]
            if _bad_name(leaf, strict) or (node.asname and _bad_name(node.asname, strict)):
                found.append((getattr(node, "lineno", 0), f"import of {node.name!r} ({why})"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and id(node) not in call_funcs:
            if node.id in ATTR_FUNCS or (strict and node.id in NAMESPACE_FUNCS):
                found.append((node.lineno, f"reference to {node.id} outside a call with a literal name ({why})"))
        elif isinstance(node, ast.Attribute) and node.attr in ATTR_FACTORIES and id(node) not in call_funcs:
            found.append((node.lineno, f"reference to {node.attr} outside a call with a literal name ({why})"))
        v = _fold_str(node) if isinstance(node, (ast.Constant, ast.BinOp, ast.JoinedStr)) else None
        if v is not None and isinstance(parents.get(id(node)), (ast.BinOp, ast.JoinedStr)):
            continue  # judged as part of the enclosing concatenation
        if v is not None:
            hits = [m for m in _DUNDER_RE.findall(v) if _bad_name(m, strict)]
            hits += [w for w in (INTERNALS_ATTRS if strict else SCRIPT_INTERNALS) if not w.startswith("__") and w == v]
            if hits:
                found.append((getattr(node, "lineno", 0), f"string spells an interpreter internal ({hits[0]!r}; {why})"))
    return found


def mutable_state_findings(tree: ast.Module) -> list[tuple[int, str]]:
    """Mutation of module-level state from inside a function: a module-level container (subscript or attribute
    store / delete, augmented assignment, mutating method call) or an attribute of any module-level name."""
    module_names: set[str] = set()
    mutable: set[str] = set()
    for node in tree.body:
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                module_names.add((a.asname or a.name).split(".")[0])
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    module_names.add(n.id)
                    if value is not None and _is_mutable_value(value):
                        mutable.add(n.id)
    found: list[tuple[int, str]] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        local = _local_names(fn)
        for node in ast.walk(fn):
            root = None
            what = ""
            if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                root, what = _root_name(node.value), "assignment into"
                if isinstance(node, ast.Attribute) and root is not None and root not in local and root in module_names:
                    found.append((node.lineno, f"attribute assignment on module-level {root!r} inside a function (no mutable global state)"))
                    continue
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, (ast.Subscript, ast.Attribute)):
                root, what = _root_name(node.target.value), "augmented assignment into"
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _MUTATING_METHODS:
                root, what = _root_name(node.func.value), f".{node.func.attr}() on"
            if root is not None and root not in local and root in mutable:
                found.append((node.lineno, f"{what} module-level container {root!r} inside a function (no mutable global state)"))
    return found


def _is_mutable_value(v: ast.AST) -> bool:
    if isinstance(v, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
        return True
    if isinstance(v, ast.Call):
        f = v.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        return name in _MUTABLE_FACTORIES
    return False


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _local_names(fn: ast.AST) -> set[str]:
    out: set[str] = set()
    args = fn.args  # type: ignore[attr-defined]
    for a in args.posonlyargs + args.args + args.kwonlyargs + [x for x in (args.vararg, args.kwarg) if x]:
        out.add(a.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            out.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not fn:
            out.add(node.name)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                out.add((a.asname or a.name).split(".")[0])
    return out


def _script_reader_findings(tree: ast.AST) -> list[tuple[int, str]]:
    """File readers that bypass the ./py I/O policy: ``from_file`` and friends, ``ctypes`` / ``mmap``."""
    found: list[tuple[int, str]] = []
    why = "a file read that bypasses the ./py I/O policy"
    for node, mod, names in _imported_modules(tree):
        parts = mod.lstrip(".").split(".")
        if parts[0] in SCRIPT_READER_MODULES or any(n in SCRIPT_READER_MODULES or n in SCRIPT_READER_ATTRS for n in names):
            found.append((node.lineno, f"import of {mod!r} {names} ({why})"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (node.attr in SCRIPT_READER_ATTRS or node.attr in SCRIPT_READER_MODULES):
            found.append((node.lineno, f"use of {node.attr!r} ({why})"))
        elif isinstance(node, ast.Name) and (node.id in SCRIPT_READER_ATTRS or node.id in SCRIPT_READER_MODULES):
            found.append((node.lineno, f"use of {node.id!r} ({why})"))
        else:
            v = _fold_str(node) if isinstance(node, (ast.Constant, ast.BinOp, ast.JoinedStr)) else None
            if v is not None and (v in SCRIPT_READER_MODULES or v in SCRIPT_READER_ATTRS):
                found.append((getattr(node, "lineno", 0), f"string {v!r} ({why})"))
    return found


def _defines(tree: ast.Module, fname: str) -> bool:
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fname for n in tree.body)


# -- build_weights.py --------------------------------------------------------------


def _import_whitelist_findings(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node, mod, names in _imported_modules(tree):
        if mod.startswith("."):
            found.append((node.lineno, "relative import (only absolute imports are allowed)"))
            continue
        root = mod.split(".")[0]
        if root in BUILD_FORBIDDEN_MODULES or mod == "numpy.random":
            found.append((node.lineno, f"import of forbidden module {mod!r}"))
            continue
        if root not in BUILD_ALLOWED_IMPORTS:
            found.append((node.lineno, f"import of {mod!r} is not allowed (build_weights.py may import {', '.join(sorted(BUILD_ALLOWED_IMPORTS - {'__future__'}))})"))
            continue
        if root == "torch":
            if mod not in BUILD_ALLOWED_TORCH_SUBMODULES:
                found.append((node.lineno, f"import of {mod!r} (only torch and torch.nn may be imported)"))
            for n in names:
                if mod == "torch" and n not in ("nn", "Tensor", "no_grad", "zeros", "tensor", "float32", "outer", "kron", "inference_mode"):
                    if n in COMPUTE_ATTRS or any(s in n for s in COMPUTE_SUBSTRINGS) or n in ("optim", "autograd", "func", "_C", "linalg", "special", "ops"):
                        found.append((node.lineno, f"import of torch.{n} (compute ban / training)"))
                if mod == "torch.nn" and n == "functional":
                    found.append((node.lineno, "import of torch.nn.functional (compute ban)"))
    return found


def _compute_findings(tree: ast.AST) -> list[tuple[int, str]]:
    why = "no forward passes, matrix products, softmax or linear-algebra calls in build_weights.py"
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.MatMult):
            found.append((node.lineno, f"use of the @ operator ({why})"))
        names: list[str] = []
        if isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, ast.Name):
            names = [node.id]
        for n in names:
            if n in COMPUTE_ATTRS:
                found.append((node.lineno, f"use of {n!r} ({why})"))
            else:
                hit = next((s for s in COMPUTE_SUBSTRINGS if s in n), None)
                if hit:
                    found.append((node.lineno, f"use of {n!r} (contains {hit!r}; {why})"))
    # Calling a module instance (m(x)) cannot be told from a function call statically;
    # the BUILD guard refuses nn.Module.__call__ at run time.
    return found


def _tokenizer_aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
    """(names bound to the tokenizer module, names bound to tokenizer members)."""
    mods: set[str] = set()
    members: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "tokenizer":
                    mods.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "tokenizer" and not node.level:
            for a in node.names:
                members.add(a.asname or a.name)
    return mods, members


def _literal_word_names(tree: ast.Module) -> set[str]:
    """Module-level names assigned exactly once, to a list/tuple of str constants."""
    counts: dict[str, int] = {}
    literal: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, (ast.For, ast.comprehension)):
            targets = [node.target]
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    counts[n.id] = counts.get(n.id, 0) + 1
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = node.value
            if isinstance(v, (ast.List, ast.Tuple)) and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in v.elts):
                literal.add(node.targets[0].id)
    return {n for n in literal if counts.get(n) == 1}


def _is_literal(node: ast.AST, words: set[str]) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple)):
        return all(_is_literal(e, words) for e in node.elts)
    if isinstance(node, ast.Name) and node.id in words:
        return True
    return False


def _tokenizer_call_findings(tree: ast.Module) -> list[tuple[int, str]]:
    mods, members = _tokenizer_aliases(tree)
    if not mods and not members:
        return []
    words = _literal_word_names(tree)
    parents: dict[int, ast.AST] = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parents[id(c)] = p

    def loop_literal(name: str, node: ast.AST) -> bool:
        cur = parents.get(id(node))
        while cur is not None:
            gens: list[tuple[ast.AST, ast.AST]] = []
            if isinstance(cur, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                gens = [(g.target, g.iter) for g in cur.generators]
            elif isinstance(cur, ast.For):
                gens = [(cur.target, cur.iter)]
            for target, it in gens:
                if isinstance(target, ast.Name) and target.id == name:
                    return _is_literal(it, words)
                # for i, w in enumerate(<literal word list>)
                if (
                    isinstance(target, ast.Tuple) and len(target.elts) == 2
                    and isinstance(target.elts[1], ast.Name) and target.elts[1].id == name
                    and isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "enumerate"
                    and len(it.args) == 1 and not it.keywords
                ):
                    return _is_literal(it.args[0], words)
            cur = parents.get(id(cur))
        return False

    def arg_ok(a: ast.AST) -> bool:
        if _is_literal(a, words):
            return True
        return isinstance(a, ast.Name) and loop_literal(a.id, a)

    def rooted(fn: ast.AST) -> bool:
        d = _dotted(fn)
        if d is None:
            return False
        root = d.split(".")[0]
        return root in mods or root in members

    found: list[tuple[int, str]] = []
    call_funcs: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and rooted(node.func):
            call_funcs.add(id(node.func))
            args = list(node.args) + [k.value for k in node.keywords]
            bad = [a for a in args if not arg_ok(a)]
            if bad:
                found.append((node.lineno, "tokenizer call on a non-literal argument (build_weights.py may call the "
                              "tokenizer only on literal word lists, or on the loop variable over one)"))
    for node in ast.walk(tree):
        # passing a tokenizer function around (map(t.word_id, xs)) instead of calling it on a literal
        if isinstance(node, (ast.Attribute, ast.Name)) and id(node) not in call_funcs and isinstance(getattr(node, "ctx", None), ast.Load):
            d = _dotted(node)
            if d is None:
                continue
            parts = d.split(".")
            if isinstance(node, ast.Attribute) and parts[0] in mods and len(parts) == 2 and parts[1] in (
                "word_id", "digit_groups", "tokenize_block", "encode_session", "prefix"
            ):
                parent = parents.get(id(node))
                if not (isinstance(parent, ast.Attribute)):
                    found.append((node.lineno, f"reference to tokenizer function {d} outside a call on a literal"))
            elif isinstance(node, ast.Name) and node.id in members and node.id in (
                "word_id", "digit_groups", "tokenize_block", "encode_session", "prefix"
            ):
                found.append((node.lineno, f"reference to tokenizer function {node.id} outside a call on a literal"))
    return found


def build_weights_findings(tree: ast.Module, source_bytes: int | None = None) -> list[tuple[int, str]]:
    """All static rules for ``build_weights.py`` (spec §9.1)."""
    found = _import_whitelist_findings(tree)
    found += forbidden_import_findings(tree)
    found += rng_findings(tree)
    found += _io_findings(tree)
    found += _compute_findings(tree)
    found += _tokenizer_call_findings(tree)
    found += introspection_findings(tree, strict=True)
    found += literal_size_findings(tree, source_bytes)
    if not _defines(tree, "build"):
        found.append((0, "no module-level function build() is defined"))
    return found


def reference_policy_findings(tree: ast.Module, source_bytes: int | None = None) -> list[tuple[int, str]]:
    """Static rules for ``reference_policy.py`` (spec §9.1: standard library only, pure, no I/O)."""
    found: list[tuple[int, str]] = []
    for node, mod, _names in _imported_modules(tree):
        if mod.startswith("."):
            found.append((node.lineno, "relative import (only absolute imports are allowed)"))
            continue
        root = mod.split(".")[0]
        if root in POLICY_NON_STDLIB_REFUSED:
            found.append((node.lineno, f"import of {mod!r} (reference_policy.py is a pure function of its arguments: standard library only)"))
        elif root in POLICY_BANNED_MODULES:
            found.append((node.lineno, f"import of banned module {mod!r} (no I/O, processes or non-determinism)"))
        elif root not in sys.stdlib_module_names and root != "__future__":
            found.append((node.lineno, f"import of {mod!r}, which is not in the standard library"))
    found += forbidden_import_findings(tree)
    found += rng_findings(tree)
    found += _io_findings(tree)
    found += introspection_findings(tree, strict=True)
    found += mutable_state_findings(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            found.append((node.lineno, f"{type(node).__name__.lower()} statement (no mutable global state)"))
    if source_bytes is not None and source_bytes > MAX_FILE_BYTES:
        found.append((0, f"file is {source_bytes} bytes; the limit is {MAX_FILE_BYTES} bytes (200 KB)"))
    if not _defines(tree, "reference_policy"):
        found.append((0, "no module-level function reference_policy() is defined"))
    return found


def author_script_findings(path: str | Path) -> list[str]:
    """Rules for any author ``.py`` file (the session audit's scan): forbidden imports, RNG APIs, file readers
    that bypass the ./py I/O policy (``from_file``, ``ctypes``, ``mmap``) and interpreter internals
    (``__globals__``, ``__builtins__``, frames, computed ``getattr`` names)."""
    p = Path(path)
    tree, problems = _parse(p)
    if tree is None:
        return _fmt(p.name, problems)
    return _fmt(p.name, forbidden_import_findings(tree) + rng_findings(tree) + _script_reader_findings(tree)
                + introspection_findings(tree, strict=False))


def check_build_weights(path: str | Path) -> list[str]:
    p = Path(path)
    data = p.read_bytes()
    tree, problems = _parse(p)
    if tree is None:
        return _fmt(p.name, problems)
    return _fmt(p.name, build_weights_findings(tree, len(data)))


def check_reference_policy(path: str | Path) -> list[str]:
    p = Path(path)
    data = p.read_bytes()
    tree, problems = _parse(p)
    if tree is None:
        return _fmt(p.name, problems)
    return _fmt(p.name, reference_policy_findings(tree, len(data)))
