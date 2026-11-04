"""Author-side launcher behind ``./py`` (spec v0.3 §8): runs one script under the AUTHOR_RUN guard.

    ./py SCRIPT.py [ARGS...]
    ./py -m MODULE [ARGS...]          (for example: ./py -m py_compile output/v03-phase-a/build_weights.py)

The module path is ``packet/`` first, then the script's directory, then its parent (with ``-m``:
``packet/``, then the current directory). Everything runs
inside ``hn_guard.guards_v03.author_run_guard``: forward passes allowed; autograd, ``torch.optim``,
random-number APIs, threads, subprocesses, network access, reads or listings outside this directory
(its siblings, Claude data and scratch, the system temp roots) and writes outside it are refused, and
so are file readers that bypass those checks (``torch.from_file``, storage ``from_file``,
``torch.jit.load``, ``ctypes``, ``mmap``). The packet renderer is the only code that may construct its
seeded ``random.Random``.
"""

import os
import runpy
import sys
import tempfile
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)
PACKET = os.path.join(ROOT, "packet")

sys.path.insert(0, HERE)
from hn_guard.guards_v03 import TrainingViolation, author_run_guard  # noqa: E402

sys.path.remove(HERE)


def _deny_roots():
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


def main(argv):
    if not argv or argv[0] in ("-h", "--help") or argv[0] == "-c" or (argv[0] == "-m" and len(argv) < 2):
        print(__doc__, file=sys.stderr)
        return 2
    if argv[0] == "-m":
        module, rest = argv[1], argv[2:]
        extra = [PACKET, os.getcwd()]
        sys.argv = [module] + rest
    else:
        script = os.path.realpath(argv[0])
        if not os.path.isfile(script):
            print(f"./py: no such script: {argv[0]}", file=sys.stderr)
            return 2
        module, script_dir = None, os.path.dirname(script)
        parent = os.path.dirname(script_dir)
        extra = [PACKET, script_dir] + ([parent] if parent.startswith(ROOT + os.sep) else [])
        sys.argv = [argv[0]] + argv[1:]
    for p in reversed(extra):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    code = 0
    try:
        with author_run_guard(
            io_root=ROOT,
            io_deny=_deny_roots(),
            rng_allowed_files=[os.path.join(PACKET, "renderer.py")],
        ):
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
