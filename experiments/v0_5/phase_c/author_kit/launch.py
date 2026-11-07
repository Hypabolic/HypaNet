"""Author-side launcher behind ``./py`` (v0.5 Phase C; spec v0.3 §8): runs one script under the AUTHOR_RUN guard.

    ./py SCRIPT.py [ARGS...]
    ./py -m MODULE [ARGS...]          (for example: ./py -m py_compile output/v05-phase-c/scratch/look.py,
                                       ./py -m v05_load check)

The module path is ``packet/`` first, then ``tools/lib/`` (the Phase C helper ``v05_load`` and its ``hn_v05``
package), then the script's directory, then its parent (with ``-m``: ``packet/``, ``tools/lib/``, then the
current directory). Everything runs inside ``hn_guard.guards_v03.author_run_guard``: forward passes allowed;
autograd, ``torch.optim``, random-number APIs, threads, subprocesses, network access, reads or listings outside
this directory and writes outside it are refused, and so are file readers that bypass those checks
(``torch.from_file``, storage ``from_file``, ``torch.jit.load``, ``ctypes``, ``mmap``). The packet renderer is the
only code that may construct its seeded ``random.Random``.

Phase C tightens the v0.3 launcher (the guard itself is unchanged):

* reads and listings: everything outside this directory is denied except the Python interpreter (its install,
  its import path) and the read-only system directories in :data:`SYSTEM_READ`; the v0.3 list (siblings, Claude
  data and scratch, the temp roots) denied only a few roots and left the rest of the disk readable;
* writes and filesystem changes (chmod, rename, link, remove, ...) only inside ``output/v05-phase-c/``;
  ``torch.save`` to a path (a C-level writer with no audit event) only there too;
* modules that read or write files below the audit layer are refused (:data:`REFUSED_MODULES`), and so is
  every ``sqlite3`` audit event.
"""

import os
import runpy
import sys
import tempfile
import traceback

try:
    import fcntl
except ImportError:  # pragma: no cover - not macOS / POSIX
    fcntl = None

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)
PACKET = os.path.join(ROOT, "packet")
LIB = os.path.join(HERE, "lib")
OUTPUT = os.path.join(os.path.realpath(ROOT), "output", "v05-phase-c")
# Read-only system directories the interpreter or torch may read lazily (e.g. platform.mac_ver). Not /System as a
# whole: /System/Volumes/Data is the data volume, so /System/Volumes/Data/Users/... reaches every user file under
# a path that realpath does not map back to /Users/...
SYSTEM_READ = ("/System/Library", "/usr", "/bin", "/sbin", "/dev", "/private/etc")
EXTRA_DENY = ("/usr/local", "/System/Volumes", "/Volumes")
# C extensions that open files without an audit event: readline.read_history_file reads any text file, _dbm and
# _sqlite3 create or read database files, Tcl (_tkinter) and curses can open files.
REFUSED_MODULES = frozenset({"readline", "_dbm", "_gdbm", "_sqlite3", "_tkinter", "_curses", "_curses_panel"})

sys.path.insert(0, HERE)
from hn_guard.guards_v03 import (  # noqa: E402
    _FS_CHANGE_EVENTS,
    TrainingViolation,
    _abspath,
    _is_write,
    _violation,
    author_run_guard,
)

sys.path.remove(HERE)
_START_PATH = list(sys.path)


def _under(path, root):
    return path == root or path.startswith(root.rstrip("/") + "/")


def _v03_deny_roots():
    """Read/list deny roots: the author dir's siblings (the repo among them), Claude data, Claude scratch and the
    system temp roots (harness scratch such as pytest-of-$USER lives there). The author dir itself stays readable."""
    roots = [os.path.dirname(ROOT), os.path.expanduser("~/.claude"), "/private/var/folders", "/var/folders"]
    if hasattr(os, "getuid"):
        roots += [f"/private/tmp/claude-{os.getuid()}", f"/tmp/claude-{os.getuid()}"]
    try:
        roots.append(os.path.realpath(tempfile.gettempdir()))
    except (OSError, FileNotFoundError):
        pass
    for var in ("TMPDIR", "TEMP", "TMP"):
        if os.environ.get(var):
            roots.append(os.path.realpath(os.environ[var]))
    return roots


def _keep_roots(root=ROOT, path=None, prefixes=None, system=SYSTEM_READ):
    """Realpaths that stay readable: the author dir, the interpreter's prefixes and import path, SYSTEM_READ."""
    if path is None:
        path = _START_PATH
    if prefixes is None:
        prefixes = (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)
    keep = {os.path.realpath(root)}
    for p in (*prefixes, *path, *system):
        if p and os.path.exists(p):
            keep.add(os.path.realpath(p))
    return sorted(keep)


def _deny_roots(root=ROOT, keep=None, listdir=os.listdir):
    """Every path outside ``keep`` is denied: for each directory on the way down to a keep root, each of its
    entries that is neither on the way to a keep root nor inside one; the author dir's own ancestors (its parent
    holds the repo) unless they lead to the interpreter; EXTRA_DENY; the v0.3 roots. A root that would contain the
    interpreter is dropped (the author dir itself is exempt in the guard). A directory on the way that cannot be
    listed is an error: the launcher refuses to run rather than leave its entries readable."""
    root = os.path.realpath(root)
    if keep is None:
        keep = _keep_roots(root)
    others = [k for k in keep if k != root]
    deny = set()
    for k in keep:
        a = k
        while os.path.dirname(a) != a:
            parent = os.path.dirname(a)
            for name in listdir(parent):  # OSError propagates: fail closed
                c = os.path.realpath(os.path.join(parent, name))
                if not any(_under(x, c) or _under(c, x) for x in keep):
                    deny.add(c)
            a = parent
    a = root
    while os.path.dirname(a) != a:
        a = os.path.dirname(a)
        if not any(_under(x, a) for x in others):
            deny.add(a)
    deny.update(EXTRA_DENY)
    deny.update(os.path.realpath(r) for r in _v03_deny_roots())
    return sorted(d for d in deny if not any(_under(x, d) for x in others))


class _RefuseModules:
    """sys.meta_path finder: the modules in REFUSED_MODULES cannot be imported."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        if name in REFUSED_MODULES or name.split(".")[0] in REFUSED_MODULES:
            raise ImportError(f"{name} is refused by ./py (it reads or writes files below the guard's checks)",
                              name=name)
        return None


# filesystem-change audit events (Python 3.12): (index of a path argument, index of its dir_fd or None) for every
# path the change writes; "entry" events act on the directory entry (a symlink is not followed)
_FS_PATHS = {
    "os.remove": ([(0, 1)], True), "os.rmdir": ([(0, 1)], True), "os.mkdir": ([(0, 2)], True),
    "os.rename": ([(0, 2), (1, 3)], True), "os.link": ([(0, 2), (1, 3)], True), "os.symlink": ([(1, 2)], True),
    "os.chmod": ([(0, 2)], False), "os.chown": ([(0, 3)], False), "os.utime": ([(0, 3)], False),
    "os.truncate": ([(0, None)], False), "shutil.copyfile": ([(1, None)], False),
    "shutil.copytree": ([(1, None)], False), "shutil.rmtree": ([(0, 1)], True),
    "shutil.move": ([(0, None), (1, None)], True),
}


def _fd_path(fd):
    """The path of an open descriptor (macOS F_GETPATH), or None."""
    try:
        return os.fsdecode(fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024)).rstrip(b"\0"))
    except (AttributeError, OSError, ValueError):
        return None


def _fs_target(args, i, j, entry):
    """Absolute path written by argument i (dir_fd at j), symlinks resolved except an entry's last component."""
    t = args[i] if i < len(args) else None
    if isinstance(t, int):
        return _fd_path(t)
    if isinstance(t, os.DirEntry):
        t = t.path
    if isinstance(t, bytes):
        t = os.fsdecode(t)
    if isinstance(t, os.PathLike):
        t = os.fspath(t)
    if not isinstance(t, str):
        return None
    d = args[j] if j is not None and j < len(args) else None
    if isinstance(d, int) and d >= 0 and not os.path.isabs(t):
        base = _fd_path(d)
        if base is None:
            return None
        t = os.path.join(base, t)
    t = os.path.abspath(t)
    if entry:
        return os.path.join(os.path.realpath(os.path.dirname(t)), os.path.basename(t))
    return os.path.realpath(t)


def _phase_c_hook(event, args):
    if event.startswith("sqlite3."):
        raise _violation(f"{event} refused under the Phase C launcher (no sqlite3)")
    if event == "open":
        target = args[0] if args else None
        if target is None or isinstance(target, int) or not _is_write(args):
            return
        path = _abspath(target)
        if path is None or not _under(path, OUTPUT):
            raise _violation(f"write to {target!r} refused under the Phase C launcher (writes stay in "
                                    "output/v05-phase-c/)")
        return
    if event in _FS_CHANGE_EVENTS:
        spots, entry = _FS_PATHS.get(event, ([(0, None)], False))
        for i, j in spots:
            path = _fs_target(args, i, j, entry)
            if path is None or not _under(path, OUTPUT):
                raise _violation(f"{event} on {args[i] if i < len(args) else None!r} refused under the Phase C "
                                 "launcher (filesystem changes stay in output/v05-phase-c/)")


def _guard_torch_writer():
    """torch.save to a file name writes through torch._C.PyTorchFileWriter in C++, with no audit event."""
    import torch

    orig = torch._C.PyTorchFileWriter

    def PyTorchFileWriter(target, *args, **kwargs):  # noqa: N802
        if isinstance(target, (str, bytes, os.PathLike)):
            path = _abspath(target)
            if path is None or not _under(path, OUTPUT):
                raise _violation(f"torch.save to {target!r} refused under the Phase C launcher (writes stay "
                                        "in output/v05-phase-c/)")
        return orig(target, *args, **kwargs)

    torch._C.PyTorchFileWriter = PyTorchFileWriter


def _tighten():
    loaded = sorted(REFUSED_MODULES & set(sys.modules))
    if loaded:
        raise _violation(f"refused modules already imported before the guard: {loaded}")
    sys.meta_path.insert(0, _RefuseModules())
    _guard_torch_writer()
    sys.addaudithook(_phase_c_hook)


def main(argv):
    if not argv or argv[0] in ("-h", "--help") or argv[0] == "-c" or (argv[0] == "-m" and len(argv) < 2):
        print(__doc__, file=sys.stderr)
        return 2
    if argv[0] == "-m":
        module, rest = argv[1], argv[2:]
        extra = [PACKET, LIB, os.getcwd()]
        sys.argv = [module] + rest
    else:
        script = os.path.realpath(argv[0])
        if not os.path.isfile(script):
            print(f"./py: no such script: {argv[0]}", file=sys.stderr)
            return 2
        module, script_dir = None, os.path.dirname(script)
        parent = os.path.dirname(script_dir)
        extra = [PACKET, LIB, script_dir] + ([parent] if parent.startswith(ROOT + os.sep) else [])
        sys.argv = [argv[0]] + argv[1:]
    try:
        deny = _deny_roots()
    except OSError as exc:
        print(f"./py: cannot compute the read deny list ({exc}); refusing to run", file=sys.stderr)
        return 2
    for p in reversed(extra):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    code = 0
    try:
        with author_run_guard(
            io_root=ROOT,
            io_deny=deny,
            rng_allowed_files=[os.path.join(PACKET, "renderer.py")],
        ):
            _tighten()
            try:
                if module is not None:
                    runpy.run_module(module, run_name="__main__", alter_sys=True)
                else:
                    runpy.run_path(script, run_name="__main__")
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except TrainingViolation as exc:
        print(f"AUTHOR GUARD: refused: {exc}", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001 - show the script's own traceback
        traceback.print_exc()
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
