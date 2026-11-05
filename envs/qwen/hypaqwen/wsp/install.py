"""Install and verify the author-side tools (``tools/``, docs/V04_WSP_KERNEL.md §4, §7).

``./fwd`` runs without the ``./py`` guard, and the ``./py`` launcher (``tools/hn_kernel/run_py.py``)
imports ``client`` and the guard package before it enters the guard (for ``init_tok``). Code in
those directories therefore runs unguarded, so the author must not be able to change it or add to it:

* ``install(dest)`` copies exactly ``KERNEL_FILES`` from this package's ``author/`` directory and
  makes the files (0444) and the directory (0555) read-only, so Write/Edit can neither change a file
  nor plant a new module (``socket.py``, ``numpy/``) next to them.
* ``verify(dest)`` is what the launcher calls before every session and the audit calls at the end: the
  directory holds exactly ``KERNEL_FILES``, none is a symlink or writable, and each matches the sha256
  of the package source. Anything else raises ``KernelTampered``. (``run_py.py`` itself re-checks at
  every ``./py`` run that the tools directories are read-only; it cannot check hashes against a copy
  it does not have.)
* ``install_tools(tools)`` / ``verify_tools(tools)`` do the same for the whole ``tools/`` tree:
  ``hn_kernel/`` (``KERNEL_FILES``), ``hn_guard/`` (``GUARD_FILES``) and, when a hook source is given,
  ``hn_hook/hook_bash.py`` (the session's PreToolUse hook, from the harness kit
  ``experiments/v0_4/author_kit/``). ``tools/`` itself is read-only and holds nothing else.

The launcher (§7) also denies Write/Edit under ``tools/`` in the session's permissions and in its
hook; this check is what still holds if that is ever misconfigured.

CLI (harness env, ``uv run --project envs/qwen``)::

    python -m hypaqwen.wsp.install install  TOOLS_DIR [--hook FILE]   # prints the manifest (JSON)
    python -m hypaqwen.wsp.install verify   TOOLS_DIR [--hook FILE]   # exit 1 if tampered
    python -m hypaqwen.wsp.install manifest [--hook FILE]             # what an install would hold
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Mapping

SOURCE_DIR = Path(__file__).resolve().parent / "author"
GUARD_SOURCE_DIR = SOURCE_DIR / "hn_guard"
KERNEL_FILES = ("client.py", "fwd.py", "protocol.py", "run_py.py", "wsp.py")
GUARD_FILES = ("__init__.py", "guards_v04.py")
KERNEL_DIR, GUARD_DIR, HOOK_DIR = "hn_kernel", "hn_guard", "hn_hook"
HOOK_FILE = "hook_bash.py"
FILE_MODE, DIR_MODE = 0o444, 0o555


class KernelTampered(RuntimeError):
    """The installed kernel tools differ from the harness's copy. Nothing from them may run."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict[str, str]:
    """``{file: sha256}`` of the harness's own copy of ``tools/hn_kernel/``."""
    return {name: _sha256(SOURCE_DIR / name) for name in KERNEL_FILES}


def _sources(hook: Path | str | None) -> dict[str, dict[str, Path]]:
    """``{subdir: {file: source path}}`` of a full ``tools/`` install."""
    out = {KERNEL_DIR: {n: SOURCE_DIR / n for n in KERNEL_FILES},
           GUARD_DIR: {n: GUARD_SOURCE_DIR / n for n in GUARD_FILES}}
    if hook is not None:
        out[HOOK_DIR] = {HOOK_FILE: Path(hook)}
    return out


def tools_manifest(hook: Path | str | None = None) -> dict[str, str]:
    """``{"hn_kernel/client.py": sha256, ...}`` of what ``install_tools`` copies."""
    return {f"{d}/{n}": _sha256(src) for d, files in _sources(hook).items() for n, src in files.items()}


def _install_dir(dest: Path, files: Mapping[str, Path]) -> None:
    if dest.is_symlink():
        raise KernelTampered(f"{dest} is a symlink")
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty")
    for name, src in files.items():
        shutil.copyfile(src, dest / name)
        os.chmod(dest / name, FILE_MODE)
    os.chmod(dest, DIR_MODE)


def _check_ro_dir(dest: Path) -> None:
    try:
        st = os.lstat(dest)
    except FileNotFoundError:
        raise KernelTampered(f"{dest} does not exist") from None
    if not stat.S_ISDIR(st.st_mode):
        raise KernelTampered(f"{dest} is not a directory (or is a symlink)")
    if st.st_mode & 0o222:
        raise KernelTampered(f"{dest} is writable")


def _verify_dir(dest: Path, want: Mapping[str, str]) -> None:
    """``dest`` is a read-only directory holding exactly ``want`` (``{file: sha256}``), read-only regular files."""
    _check_ro_dir(dest)
    present = sorted(os.listdir(dest))
    if present != sorted(want):
        extra = sorted(set(present) - set(want))
        missing = sorted(set(want) - set(present))
        raise KernelTampered(f"{dest}: unexpected entries {extra}, missing {missing}")
    for name, sha in want.items():
        st = os.lstat(dest / name)
        if not stat.S_ISREG(st.st_mode):
            raise KernelTampered(f"{dest / name} is not a regular file")
        if st.st_mode & 0o222:
            raise KernelTampered(f"{dest / name} is writable")
        if _sha256(dest / name) != sha:
            raise KernelTampered(f"{dest / name} does not match the harness copy")


def install(dest: Path | str) -> dict[str, str]:
    """Copy the kernel tools into ``dest`` (new or empty), read-only. Returns the manifest."""
    dest = Path(dest)
    _install_dir(dest, {n: SOURCE_DIR / n for n in KERNEL_FILES})
    verify(dest)
    return source_manifest()


def verify(dest: Path | str) -> None:
    """Raise ``KernelTampered`` unless ``dest`` is exactly the read-only, unmodified install."""
    _verify_dir(Path(dest), source_manifest())


def install_tools(tools: Path | str, *, hook: Path | str | None = None) -> dict[str, str]:
    """Install ``tools/`` (new or empty): ``hn_kernel/``, ``hn_guard/`` and, with ``hook``, ``hn_hook/``.

    Every file is 0444 and every directory, ``tools/`` included, 0555. Returns ``tools_manifest(hook)``.
    """
    tools = Path(tools)
    if tools.is_symlink():
        raise KernelTampered(f"{tools} is a symlink")
    tools.mkdir(parents=True, exist_ok=True)
    if any(tools.iterdir()):
        raise FileExistsError(f"{tools} is not empty")
    for sub, files in _sources(hook).items():
        _install_dir(tools / sub, files)
    os.chmod(tools, DIR_MODE)
    verify_tools(tools, hook=hook)
    return tools_manifest(hook)


def verify_tools(tools: Path | str, *, hook: Path | str | None = None) -> None:
    """Raise ``KernelTampered`` unless ``tools`` is exactly the read-only, unmodified ``install_tools`` tree."""
    tools = Path(tools)
    _check_ro_dir(tools)
    sources = _sources(hook)
    present = sorted(os.listdir(tools))
    if present != sorted(sources):
        raise KernelTampered(f"{tools}: holds {present}, expected exactly {sorted(sources)}")
    for sub, files in sources.items():
        _verify_dir(tools / sub, {n: _sha256(src) for n, src in files.items()})


def uninstall(dest: Path | str) -> None:
    """Make ``dest`` (a ``tools/`` tree or one of its directories) writable again and remove it (tests)."""
    dest = Path(dest)
    if not dest.exists():
        return
    for d, _dirs, files in os.walk(dest):
        os.chmod(d, 0o755)
        for f in files:
            if not os.path.islink(os.path.join(d, f)):
                os.chmod(os.path.join(d, f), 0o644)
    shutil.rmtree(dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hypaqwen.wsp.install", description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("install", "verify", "manifest"))
    ap.add_argument("tools", type=Path, nargs="?", help="the author dir's tools/ directory")
    ap.add_argument("--hook", type=Path, help="the PreToolUse hook source (experiments/v0_4/author_kit/hook_bash.py)")
    args = ap.parse_args(argv)
    if args.action == "manifest":
        print(json.dumps(tools_manifest(args.hook), indent=2, sort_keys=True))
        return 0
    if args.tools is None:
        ap.error(f"{args.action} needs TOOLS_DIR")
    try:
        if args.action == "install":
            print(json.dumps(install_tools(args.tools, hook=args.hook), indent=2, sort_keys=True))
        else:
            verify_tools(args.tools, hook=args.hook)
            print(f"tools verified: {args.tools}")
    except (KernelTampered, FileExistsError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
