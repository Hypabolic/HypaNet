"""./python3: run your own Python (numpy only) under the kit's guard.

    ./python3 work/make_network.py [ARGS...]
    ./python3 -c "import numpy as np; print(np.pi)"

The interpreter is this directory's private venv (Python 3.12, numpy only), started in isolated mode. The guard:

* scripts must be .py files inside this directory;
* reads and directory listings: only inside this directory (plus the interpreter's own files and read-only
  system directories); paths are str or pathlib paths; opening a directory as a file and dir_fd= arguments
  are refused;
* writes, renames, removals and other filesystem changes: only inside work/;
* no subprocesses, no network, no ctypes, no symlinks or hard links, no changing directory;
* no frame or garbage-collector introspection (sys._getframe, gc.get_objects, ...), no tracing hooks, no
  re-import or reload of a module that was loaded at start-up;
* no scipy, torch, sklearn or other fitting / optimisation / autodiff libraries (they are not installed either);
* numpy's least-squares and fitting routines (numpy.linalg.lstsq, numpy.linalg.pinv, numpy.polyfit, the
  numpy.polynomial fit functions and .fit methods) and statistics.linear_regression raise an error: no fitting
  of any kind is allowed in this run.

A refused operation raises GuardError (a PermissionError) inside your script.
"""

import os
import sys

sys.dont_write_bytecode = True


class GuardError(PermissionError):
    pass


def _install(root):
    """Patch the fitting routines and dir_fd functions; return the audit hook and the import finder.

    Everything the hook uses is bound as a keyword default (immutable values, C functions, read-only views), and
    the hook has no globals and no closure cells, so nothing the author's code can reach (this module's globals,
    builtins, os, posixpath, ...) changes what it decides. The hook is never stored where Python code can find it.
    """
    import pathlib
    import posix
    import shutil
    import stat as stat_mod
    import statistics

    import numpy as np
    import numpy.lib._polynomial_impl as poly_impl
    import numpy.linalg as npl
    import numpy.linalg._linalg as npl_impl
    import numpy.ma  # noqa: F401  (loaded now so nothing imported later holds an unpatched reference)
    import numpy.polynomial as P
    import numpy.polynomial._polybase as PB
    import numpy.polynomial.polyutils as PU

    # the C functions the hook uses, bound before anything below wraps them
    c_fspath, c_getcwd, c_lstat, c_stat, c_readlink = posix.fspath, posix.getcwd, posix.lstat, posix.stat, \
        posix.readlink
    system_read = ("/System/Library", "/usr/lib", "/usr/share", "/dev", "/private/etc", "/etc",
                   "/private/var/db/timezone")
    roots = {os.path.realpath(root)}
    for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix, *sys.path, *system_read):
        if p and os.path.exists(p):
            roots.add(os.path.realpath(p))
    read_roots = tuple(sorted(r.rstrip("/") for r in roots))
    work_real = os.path.realpath(os.path.join(root, "work"))

    # --- fitting routines refused, including numpy's private originals and its dispatch registry ----------------
    def refuse(name):
        def f(*a, _Err=GuardError, **k):
            raise _Err(f"./python3 guard: {name} is a fitting routine; no fitting of any kind is allowed "
                             "in this run (KIT.md, Rules)")
        f.__name__ = f.__qualname__ = name.rsplit(".", 1)[-1]
        return f

    originals = []
    for mod, names in ((npl, ("lstsq", "pinv")), (npl_impl, ("lstsq", "pinv")), (np, ("polyfit",)),
                       (poly_impl, ("polyfit",)), (PU, ("_fit",)), (statistics, ("linear_regression",))):
        for n in names:
            if hasattr(mod, n):
                originals.append(getattr(mod, n))
                setattr(mod, n, refuse(f"{mod.__name__}.{n}"))
    umath = getattr(npl_impl, "_umath_linalg", None)
    if umath is not None and hasattr(umath, "lstsq"):
        umath.lstsq = refuse("numpy.linalg._umath_linalg.lstsq")
    for modname in ("polynomial", "chebyshev", "legendre", "laguerre", "hermite", "hermite_e"):
        mod = getattr(P, modname)
        for fn in ("polyfit", "chebfit", "legfit", "lagfit", "hermfit", "hermefit"):
            if hasattr(mod, fn):
                originals.append(getattr(mod, fn))
                setattr(mod, fn, refuse(f"numpy.polynomial.{modname}.{fn}"))
    PB.ABCPolyBase.fit = classmethod(lambda cls, *a, _f=refuse("numpy.polynomial .fit"), **k: _f())
    try:
        from numpy._core import overrides
        reg = getattr(overrides, "ARRAY_FUNCTIONS", None)
        if isinstance(reg, set):
            for f in originals:
                reg.discard(f)
    except ImportError:
        pass
    del originals
    shutil._use_fd_functions = False  # its fd-based rmtree opens directories, which the hook refuses

    # --- dir_fd= refused (defence in depth: a directory cannot be opened as a file either) -----------------------
    def no_dir_fd(fn):
        # the original is a keyword default, not a closure cell: reading __kwdefaults__ of a wrapper is refused
        def wrapped(*a, _fn=fn, _Err=GuardError, **k):
            for key, v in k.items():
                if key.endswith("dir_fd") and v is not None:
                    raise _Err(f"./python3 guard: {_fn.__name__}({key}=...) refused (paths are checked "
                               "against the current directory)")
            return _fn(*a, **k)
        wrapped.__name__ = wrapped.__qualname__ = fn.__name__
        return wrapped

    wrappers = []
    for fn in list(os.supports_dir_fd):
        w = no_dir_fd(fn)
        wrappers.append(w)
        for mod in (os, posix):
            if getattr(mod, fn.__name__, None) is fn:
                setattr(mod, fn.__name__, w)
    os.supports_dir_fd = set()

    # re-executing these files (re-import after deletion from sys.modules, importlib.reload) would restore the
    # originals patched above
    protected_files = frozenset(f for m in (
        npl_impl, poly_impl, PU, PB, statistics, shutil, pathlib, os,
        *(getattr(P, n) for n in ("polynomial", "chebyshev", "legendre", "laguerre", "hermite", "hermite_e")))
        if getattr(m, "__file__", None) for f in (m.__file__, os.path.realpath(m.__file__)))
    loaded_at_start = frozenset(sys.modules)

    refused_imports = frozenset({
        "scipy", "torch", "sklearn", "jax", "jaxlib", "tensorflow", "keras", "numba", "autograd", "cvxpy", "optuna",
        "statsmodels", "lmfit", "pymoo", "nevergrad", "hyperopt", "skopt", "deap", "theano", "mlx", "tinygrad",
        "ctypes", "_ctypes", "cffi", "sqlite3", "_sqlite3", "socket", "_socket", "ssl", "urllib.request",
        "http.client", "subprocess", "multiprocessing", "concurrent", "readline", "_tkinter", "tkinter", "_curses",
        "curses", "_testcapi", "_testinternalcapi", "_testmultiphase", "_testsinglephase", "_xxsubinterpreters",
        "_xxinterpchannels", "_interpreters", "_posixsubprocess",
    })
    fs_change = {
        "os.remove": (0,), "os.rmdir": (0,), "os.mkdir": (0,), "os.rename": (0, 1), "os.chmod": (0,), "os.chown": (0,),
        "os.chflags": (0,), "os.lchflags": (0,), "os.utime": (0,), "os.truncate": (0,), "shutil.copyfile": (1,),
        "shutil.copytree": (1,), "shutil.rmtree": (0,), "shutil.move": (0, 1), "shutil.make_archive": (0,),
        "os.setxattr": (0,), "os.removexattr": (0,),
    }
    refused_events = (
        "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "pty.",
        "os.kill", "os.killpg", "socket.", "ctypes.", "sqlite3.", "webbrowser.", "os.symlink", "os.link",
        "sys.addaudithook", "os.startfile", "_posixsubprocess", "urllib.", "ftplib.", "smtplib.", "imaplib.",
        "poplib.", "nntplib.", "telnetlib.", "http.", "sys.remote_exec", "os.add_dll_directory", "os.chdir",
        "os.fchdir", "gc.get_", "sys.settrace", "sys.setprofile", "sys.monitoring", "os.posix_openpt",
    )
    refused_exact = frozenset({"sys._getframe", "sys._current_frames", "sys._current_exceptions"})
    listing = frozenset({"os.listdir", "os.scandir", "glob.glob", "glob.glob/2", "os.walk", "os.fwalk"})
    pure = pathlib.PurePath.__dict__  # a read-only live view: later changes to PurePath show in it
    path_ok = (pure.get("__fspath__"), pure.get("__str__"), pure.get("_str"))
    protected = (GuardError, pathlib.PurePath, pathlib.Path, pathlib.PosixPath, pathlib.PurePosixPath,
                 *(f for f in path_ok if f is not None), *wrappers)

    def hook(event, args, *, _str=str, _bytes=bytes, _int=int, _bool=bool, _type=type, _isinstance=isinstance,
             _len=len, _range=range, _getattr=getattr, _OSError=OSError, _Err=GuardError, _fspath=c_fspath,
             _getcwd=c_getcwd, _lstat=c_lstat, _stat=c_stat, _readlink=c_readlink,
             _S_ISLNK=stat_mod.S_ISLNK, _S_ISDIR=stat_mod.S_ISDIR, _DirEntry=os.DirEntry,
             _path_types=(pathlib.PosixPath, pathlib.PurePosixPath), _pure=pure, _path_ok=path_ok,
             _sub_dicts=tuple(t.__dict__ for t in (pathlib.Path, pathlib.PosixPath, pathlib.PurePosixPath)),
             _read_roots=read_roots, _work=work_real, _fs_change=fs_change, _refused_events=refused_events, _refused_exact=refused_exact,
             _refused_imports=refused_imports, _listing=listing, _protected_files=protected_files,
             _loaded=loaded_at_start, _protected=protected,
             _write_flags=os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC):
        if event == "open":
            what = "open"
        elif event in _fs_change:
            what = "change"
        elif event in _listing:
            what = "list"
        elif event == "exec":
            fn = _getattr(args[0], "co_filename", None) if args else None
            if _type(fn) is _str and fn in _protected_files:
                raise _Err(f"./python3 guard: re-executing {fn} refused (no reload or re-import of the modules "
                           "the guard patches)")
            return
        elif event == "import":
            name = args[0] if args else None
            if _type(name) is _str:
                parts = name.split(".")
                for i in _range(1, _len(parts) + 1):
                    if ".".join(parts[:i]) in _refused_imports:
                        raise _Err(f"./python3 guard: import of {name} refused")
                if name in _loaded:
                    raise _Err(f"./python3 guard: re-import of {name} refused (it was removed from sys.modules)")
            return
        elif event == "object.__setattr__" or event == "object.__delattr__" or event == "object.__getattr__":
            obj = args[0] if args else None
            for o in _protected:
                if obj is o:
                    raise _Err(f"./python3 guard: {event} on the guard's objects refused")
            return
        else:
            if event in _refused_exact:
                raise _Err(f"./python3 guard: {event} refused (no frame or gc introspection)")
            for prefix in _refused_events:
                if event.startswith(prefix):
                    raise _Err(f"./python3 guard: {event} refused (no subprocesses, network, ctypes, links, chdir, "
                               "frame or gc introspection, or tracing)")
            return

        if what == "open":
            idx = (0,)
            mode = args[1] if _len(args) > 1 else None
            flags = args[2] if _len(args) > 2 else 0
            write = False
            if _type(mode) is _str:
                for c in "wax+":
                    if c in mode:
                        write = True
            if _isinstance(flags, _int) and flags & _write_flags:
                write = True
        elif what == "change":
            idx = _fs_change[event]
            write = True
        else:
            idx = (0,)
            write = False
        for i in idx:
            t = args[i] if i < _len(args) else None
            if t is None:
                if what == "list":
                    t = "."
                elif what == "open":
                    continue
            if _isinstance(t, _int) and not _isinstance(t, _bool):
                if what == "change":
                    raise _Err(f"./python3 guard: {event} on a file descriptor refused")
                continue  # an fd: it came from an open the guard checked
            # the path as a plain str, from immutable data only
            if _isinstance(t, _str):
                s = "".join((t,))
            elif _isinstance(t, _bytes):
                s = b"".join((t,)).decode("utf-8", "surrogateescape")
            elif _type(t) is _DirEntry:
                s = "".join((_fspath(t),))
            elif _type(t) in _path_types:
                ok = (_pure.get("__fspath__"), _pure.get("__str__"), _pure.get("_str")) == _path_ok
                for d in _sub_dicts:
                    if "__fspath__" in d or "__str__" in d or "_str" in d:
                        ok = False
                if not ok:
                    raise _Err("./python3 guard: pathlib was modified; path objects refused")
                s = _fspath(t)  # PurePath caches its str on first use, so open() sees the same string
                if _isinstance(s, _str):
                    s = "".join((s,))
                else:
                    raise _Err("./python3 guard: unusable path object")
            else:
                raise _Err(f"./python3 guard: {event}: give the path as a str or a pathlib.Path, not "
                           f"{_type(t).__name__}")
            # resolve it as os.path.realpath does, with the C functions bound above
            if not s.startswith("/"):
                s = _getcwd() + "/" + s
            pending = s.split("/")
            pending.reverse()
            out = []
            links = 0
            while pending:
                c = pending.pop()
                if c == "" or c == ".":
                    continue
                if c == "..":
                    if out:
                        out.pop()
                    continue
                cand = "/" + "/".join(out + [c])
                try:
                    islink = _S_ISLNK(_lstat(cand).st_mode)
                except _OSError:
                    islink = False
                if islink:
                    links += 1
                    if links > 40:
                        raise _Err(f"./python3 guard: too many symbolic links in {s!r}")
                    target = _readlink(cand)
                    if target.startswith("/"):
                        out = []
                    more = target.split("/")
                    more.reverse()
                    pending.extend(more)
                else:
                    out.append(c)
            p = "/" + "/".join(out)
            if write:
                if not (p == _work or p.startswith(_work + "/")):
                    raise _Err(f"./python3 guard: {'write to' if what == 'open' else event + ' on'} {s!r} refused "
                               "(writes and filesystem changes stay in work/)")
                continue
            inside = False
            for r in _read_roots:
                if p == r or p.startswith(r + "/"):
                    inside = True
            if not inside:
                raise _Err(f"./python3 guard: {'read of' if what == 'open' else event + ' of'} {s!r} refused "
                           "(reads stay in this directory)")
            if what == "open":
                try:
                    isdir = _S_ISDIR(_stat(p).st_mode)
                except _OSError:
                    isdir = False
                if isdir:
                    raise _Err(f"./python3 guard: opening the directory {s!r} as a file refused (list it with "
                               "os.listdir)")

    class RefuseFinder:
        @staticmethod
        def find_spec(name, path=None, target=None, _refused=refused_imports, _Err=ImportError, _len=len,
                      _range=range):
            parts = name.split(".")
            for i in _range(1, _len(parts) + 1):
                if ".".join(parts[:i]) in _refused:
                    raise _Err(f"./python3 guard: import of {name} refused (numpy only; no fitting, optimisation, "
                               "subprocess, network or ctypes modules)", name=name)
            return None

    return hook, RefuseFinder()


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__, file=sys.stderr)
        return 2
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if argv[0] == "-c":
        if len(argv) < 2:
            print("./python3 -c needs code", file=sys.stderr)
            return 2
        code, script = argv[1], None
        sys.argv = ["-c"] + argv[2:]
        sys.path.insert(0, os.getcwd())
    elif argv[0].startswith("-"):
        print(f"./python3: option {argv[0]} is not supported (use ./python3 SCRIPT.py or ./python3 -c CODE)",
              file=sys.stderr)
        return 2
    else:
        script = os.path.realpath(argv[0])
        if not os.path.isfile(script):
            print(f"./python3: no such script: {argv[0]}", file=sys.stderr)
            return 2
        if not (script == root or script.startswith(root.rstrip("/") + "/")):
            print("./python3: scripts must be inside this directory", file=sys.stderr)
            return 2
        if not script.endswith(".py"):
            print(f"./python3: scripts must be .py files: {argv[0]}", file=sys.stderr)
            return 2
        code = None
        sys.argv = [argv[0]] + argv[1:]
        sys.path.insert(0, os.path.dirname(script))
    here = os.path.dirname(os.path.realpath(__file__))
    if here in sys.path:
        sys.path.remove(here)
    hook, finder = _install(root)  # imports numpy before the hook (numpy's own start-up loads ctypes)
    import builtins
    import runpy
    import types

    run_path, new_module, _sys, _exit = runpy.run_path, types.ModuleType, sys, SystemExit
    # forget this module's globals: the author's code can reach this module, but nothing in it matters any more
    g = globals()
    for k in list(g):
        if k not in ("__builtins__", "__name__"):
            del g[k]
    del g
    _sys.meta_path.insert(0, finder)
    _sys.addaudithook(hook)
    del hook, finder
    try:
        if script is None:
            mod = new_module("__main__")  # a fresh __main__: the guard module is not the code's namespace
            mod.__builtins__ = builtins
            _sys.modules["__main__"] = mod
            exec(compile(code, "<string>", "exec"), mod.__dict__)
        else:
            run_path(script, run_name="__main__")
    except _exit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
