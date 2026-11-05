"""``./py SCRIPT [ARGS...]``: one author script under the KERNEL_CLIENT guard (docs/V04_WSP_KERNEL.md §4-5).

    ./py output/<phase>/scratch/check.py [ARGS...]
    ./py -m py_compile output/<phase>/build_edit.py      (a syntax check; nothing is executed or written)

Installed as ``tools/hn_kernel/run_py.py`` (read-only, checked by ``hypaqwen.wsp.install.verify_tools``) and run
by the ``./py`` wrapper as ``.venv/bin/python -I -B tools/hn_kernel/run_py.py``. The v0.3 launcher
(``experiments/v0_3/author_kit/launch.py``) put the author's directories on ``sys.path``; this one never
does, because everything it runs before the guard is unguarded:

1. ``sys.path`` is scrubbed before the first non-builtin import, as in ``fwd.py``: ``''``, this directory
   and every entry inside the author dir are dropped, except the interpreter install (the author venv,
   ``AUTHOR_DIR/.venv``, read-only). A ``json.py`` or ``numpy/`` planted in ``output/`` or ``tools/``
   is therefore never imported.
2. The tools directories must be read-only (``tools/``, ``tools/hn_kernel/``, ``tools/hn_guard/``), so
   nothing can be planted next to the files loaded by path below.
3. The standard library, ``numpy``, ``torch``, ``safetensors`` and ``tokenizers`` are imported; the
   kernel client is loaded by explicit path (``client.py`` as ``hn_kernel_client``, which loads
   ``protocol.py``) and builds ``tok`` from ``packet/qwen/tokenizer.json``; ``wsp.py`` is loaded by path
   as the module ``wsp``, so ``from wsp import ...`` resolves to this installed client and nothing else;
   the guard package is loaded by path as ``hn_guard``.
4. The phase is ``$HN_PHASE`` (set by the launcher), else the single ``output/<phase>/`` directory (the
   single writable one if there are several). The script must be a ``.py`` file inside
   ``output/<phase>/``; ``-c`` and every ``-m`` other than ``py_compile`` are refused.
5. ``hn_guard.install(KERNEL_CLIENT, author_dir=AUTHOR_DIR, phase=..., kernel_endpoint=AUTHOR_DIR/.kernel/kernel.sock)``.
   From here on the guard holds for the life of the process: reads only under the author dir and the
   interpreter install, writes only under ``output/<phase>/``, one socket (the kernel), no subprocess,
   thread, autograd, optimiser or RNG (except ``wsp.control_direction``).
6. Only then is author code reachable. A finder appended at the end of ``sys.meta_path`` (after every
   standard finder, so it can never shadow the standard library or an installed package) resolves a
   top-level ``import name`` to ``name.py`` in the script's directory, then each parent up to
   ``output/<phase>/``, then ``packet/``. The script runs with ``runpy.run_path`` as ``__main__``.
7. The process ends with ``hn_guard.finish(code)``: the script's exit status, 1 after an uncaught
   exception, and 3 whenever the guard recorded a violation (even one the script caught).

Exit status 2: the command line, the phase or the tools layout was refused before any author code ran.
"""

# No ``from __future__`` import: it would be the first module looked up on sys.path (see fwd.py).
import os  # frozen, already loaded at start-up
import sys  # builtin

_HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))  # AUTHOR_DIR (this file is AUTHOR_DIR/tools/hn_kernel/run_py.py)


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _install_roots() -> list:
    return [os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)
            if p and os.path.realpath(p) != os.sep]


def _scrub_sys_path() -> None:
    """Drop '' (the cwd), this directory and the author dir from sys.path, keeping the interpreter install."""
    installs = _install_roots()
    keep = []
    for p in sys.path:
        r = os.path.realpath(p) if p else ""
        if not p or _inside(r, _HERE):
            continue
        if _inside(r, ROOT) and not any(_inside(r, i) for i in installs):
            continue
        keep.append(p)
    sys.path[:] = keep


if not __package__:  # run as the ./py launcher (as a package module, the harness's own sys.path is left alone)
    _scrub_sys_path()

import importlib.abc  # noqa: E402
import importlib.util  # noqa: E402
import re  # noqa: E402
import runpy  # noqa: E402
import stat  # noqa: E402
import traceback  # noqa: E402
import contextlib  # noqa: E402,F401  (protocol's and client's imports, loaded before them)
import json  # noqa: E402,F401
import math  # noqa: E402,F401
import socket  # noqa: E402,F401
import struct  # noqa: E402,F401
import typing  # noqa: E402,F401
import atexit  # noqa: E402,F401
from pathlib import Path  # noqa: E402,F401

import numpy  # noqa: E402,F401
import torch  # noqa: E402,F401

try:  # optional in the venv; loaded before the guard so later use needs no file reads
    import safetensors  # noqa: E402,F401
    import safetensors.torch  # noqa: E402,F401
except ImportError:  # pragma: no cover
    pass
try:
    import tokenizers  # noqa: E402,F401
except ImportError:  # pragma: no cover
    pass

PHASE_ENV = "HN_PHASE"
AUTHOR_ENV = "HN_AUTHOR_DIR"
SOCKET_ENV = "HN_KERNEL_SOCKET"
PHASE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
MODULE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
USAGE = "usage: ./py output/<phase>/SCRIPT.py [ARGS...]  |  ./py -m py_compile output/<phase>/FILE.py"


class Refused(Exception):
    """The command line, phase or tools layout is not allowed; nothing of the author's ran."""


def _check_tools() -> None:
    """tools/, tools/hn_kernel/ and tools/hn_guard/ are real, read-only directories (nothing can be planted)."""
    tools = os.path.dirname(_HERE)
    for d in (tools, _HERE, os.path.join(tools, "hn_guard")):
        try:
            st = os.lstat(d)
        except FileNotFoundError:
            raise Refused(f"{os.path.relpath(d, ROOT)}/ is missing") from None
        if not stat.S_ISDIR(st.st_mode):
            raise Refused(f"{os.path.relpath(d, ROOT)} is not a directory")
        if st.st_mode & 0o222:
            raise Refused(f"{os.path.relpath(d, ROOT)}/ is writable; the kernel tools may not run")


def _load_path(key: str, path: str, package_dir: str = None):
    """``path`` loaded by explicit file location as ``sys.modules[key]`` (a package when ``package_dir``)."""
    mod = sys.modules.get(key)
    if mod is not None:
        return mod
    kw = {"submodule_search_locations": [package_dir]} if package_dir else {}
    spec = importlib.util.spec_from_file_location(key, path, **kw)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[key]
        raise
    return mod


def resolve_phase(root: str, env: str = None) -> str:
    """``$HN_PHASE`` if set, else the single (writable) directory under ``output/``."""
    out = os.path.join(root, "output")
    if env:
        if not PHASE_RE.fullmatch(env):
            raise Refused(f"bad phase name {env!r}")
        if not os.path.isdir(os.path.join(out, env)):
            raise Refused(f"output/{env}/ does not exist")
        return env
    try:
        names = sorted(n for n in os.listdir(out) if PHASE_RE.fullmatch(n) and os.path.isdir(os.path.join(out, n)))
    except FileNotFoundError:
        raise Refused("output/ does not exist") from None
    if len(names) > 1:
        names = [n for n in names if os.access(os.path.join(out, n), os.W_OK)]
    if len(names) != 1:
        raise Refused(f"cannot tell the phase: output/ holds {len(names)} candidate directories and {PHASE_ENV} is unset")
    return names[0]


def resolve_script(arg: str, root: str, phase: str) -> str:
    """The real path of a ``.py`` file inside ``output/<phase>/``, or ``Refused``."""
    phase_dir = os.path.realpath(os.path.join(root, "output", phase))
    path = os.path.realpath(arg if os.path.isabs(arg) else os.path.join(os.getcwd(), arg))
    if not _inside(path, phase_dir) or path == phase_dir:
        raise Refused(f"scripts must lie inside output/{phase}/: {arg}")
    if not path.endswith(".py"):
        raise Refused(f"not a .py file: {arg}")
    try:
        st = os.stat(path)
    except FileNotFoundError:
        raise Refused(f"no such script: {arg}") from None
    if not stat.S_ISREG(st.st_mode):
        raise Refused(f"not a regular file: {arg}")
    return path


def parse_argv(argv: list, root: str, phase: str) -> tuple:
    """``("run", script, args)`` or ``("compile", file, [])``; ``Refused`` for anything else."""
    if not argv or argv[0] in ("-h", "--help"):
        raise Refused(USAGE)
    if argv[0] == "-c":
        raise Refused("./py -c is not supported: write the code to a file under output/<phase>/ and run that")
    if argv[0] == "-m":
        if len(argv) != 3 or argv[1] != "py_compile":
            raise Refused("the only -m form is: ./py -m py_compile output/<phase>/FILE.py")
        return "compile", resolve_script(argv[2], root, phase), []
    if argv[0].startswith("-"):
        raise Refused(f"unknown option {argv[0]!r}; {USAGE}")
    return "run", resolve_script(argv[0], root, phase), list(argv[1:])


class AuthorModules(importlib.abc.MetaPathFinder):
    """Top-level ``import name`` -> ``name.py`` in ``dirs`` (searched in order). Installed after the guard and
    after every other finder, so it only ever sees names nothing else could import."""

    def __init__(self, dirs):
        self.dirs = tuple(dirs)

    def find_spec(self, name, path=None, target=None):
        if path is not None or not MODULE_RE.fullmatch(name):
            return None
        for d in self.dirs:
            f = os.path.join(d, name + ".py")
            if os.path.isfile(f):
                return importlib.util.spec_from_file_location(name, f)
        return None


def module_dirs(script: str, root: str, phase: str) -> list:
    """The script's directory, each parent up to ``output/<phase>/``, then ``packet/``."""
    phase_dir = os.path.realpath(os.path.join(root, "output", phase))
    dirs, d = [], os.path.dirname(script)
    while _inside(d, phase_dir):
        dirs.append(d)
        if d == phase_dir:
            break
        d = os.path.dirname(d)
    packet = os.path.join(root, "packet")
    return dirs + ([packet] if os.path.isdir(packet) else [])


def _prepare(argv: list) -> tuple:
    """Everything before the guard. Returns ``(guard module, phase, action, path, args)``."""
    env_root = os.environ.get(AUTHOR_ENV)
    if env_root and os.path.realpath(env_root) != os.path.realpath(ROOT):
        raise Refused(f"{AUTHOR_ENV} does not name the directory these tools are installed in")
    _check_tools()
    phase = resolve_phase(ROOT, os.environ.get(PHASE_ENV))
    action, path, args = parse_argv(argv, ROOT, phase)
    endpoint = os.path.join(ROOT, ".kernel", "kernel.sock")
    os.environ[AUTHOR_ENV] = ROOT  # the client reads both at import / connect time
    os.environ[SOCKET_ENV] = endpoint  # the one socket the guard permits
    os.environ[PHASE_ENV] = phase
    client = _load_path("hn_kernel_client", os.path.join(_HERE, "client.py"))
    client._cache["tok"] = client.init_tok()  # reads packet/qwen/tokenizer.json, before the guard
    _load_path("wsp", os.path.join(_HERE, "wsp.py"))
    gdir = os.path.join(os.path.dirname(_HERE), "hn_guard")
    guard = _load_path("hn_guard", os.path.join(gdir, "__init__.py"), package_dir=gdir)
    return guard, phase, action, path, args, endpoint


def main(argv: list) -> int:
    try:
        guard, phase, action, path, args, endpoint = _prepare(argv)
    except Refused as exc:
        print(f"./py: refused: {exc}", file=sys.stderr)
        return 2
    guard.install(guard.KERNEL_CLIENT, author_dir=ROOT, phase=phase, kernel_endpoint=endpoint)
    code = 0
    try:
        if action == "compile":
            with open(path, "rb") as f:
                compile(f.read(), path, "exec", dont_inherit=True)
            print(f"./py: {os.path.relpath(path, ROOT)} compiles")
        else:
            sys.meta_path.append(AuthorModules(module_dirs(path, ROOT, phase)))
            sys.argv = [path] + args  # runpy sets argv[0] to the script's path in any case
            try:
                runpy.run_path(path, run_name="__main__")
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException:  # noqa: BLE001 - the script's own traceback; the guard has printed any refusal
        traceback.print_exc()
        code = 1
    guard.finish(code)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
