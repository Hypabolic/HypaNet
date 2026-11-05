"""Install and verify the author-side kernel tools (``tools/hn_kernel/``, docs/V04_WSP_KERNEL.md §4).

``./fwd`` runs without the ``./py`` guard, and the ``./py`` wrapper imports ``client`` before it
enters the guard (for ``init_tok``). Code in that directory therefore runs unguarded, so the
author must not be able to change it or add to it:

* ``install(dest)`` copies exactly ``KERNEL_FILES`` from this package's ``author/`` directory and
  makes the files (0444) and the directory (0555) read-only, so Write/Edit can neither change a file
  nor plant a new module (``socket.py``, ``numpy/``) next to them.
* ``verify(dest)`` is what the ``./py`` and ``./fwd`` wrappers call before every exec, and the audit
  calls at the end: the directory holds exactly ``KERNEL_FILES``, none is a symlink or writable, and
  each matches the sha256 of the package source. Anything else raises ``KernelTampered``.

The launcher (§7) also denies Write/Edit under ``tools/`` in the session's permissions; this
check is what still holds if that is ever misconfigured.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent / "author"
KERNEL_FILES = ("client.py", "fwd.py", "protocol.py", "wsp.py")
FILE_MODE, DIR_MODE = 0o444, 0o555


class KernelTampered(RuntimeError):
    """The installed kernel tools differ from the harness's copy. Nothing from them may run."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict[str, str]:
    """``{file: sha256}`` of the harness's own copy."""
    return {name: _sha256(SOURCE_DIR / name) for name in KERNEL_FILES}


def install(dest: Path | str) -> dict[str, str]:
    """Copy the kernel tools into ``dest`` (new or empty), read-only. Returns the manifest."""
    dest = Path(dest)
    if dest.is_symlink():
        raise KernelTampered(f"{dest} is a symlink")
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty")
    for name in KERNEL_FILES:
        shutil.copyfile(SOURCE_DIR / name, dest / name)
        os.chmod(dest / name, FILE_MODE)
    os.chmod(dest, DIR_MODE)
    verify(dest)
    return source_manifest()


def verify(dest: Path | str) -> None:
    """Raise ``KernelTampered`` unless ``dest`` is exactly the read-only, unmodified install."""
    dest = Path(dest)
    try:
        st = os.lstat(dest)
    except FileNotFoundError:
        raise KernelTampered(f"{dest} does not exist") from None
    if not stat.S_ISDIR(st.st_mode):
        raise KernelTampered(f"{dest} is not a directory (or is a symlink)")
    if st.st_mode & 0o222:
        raise KernelTampered(f"{dest} is writable")
    present = sorted(os.listdir(dest))
    if present != sorted(KERNEL_FILES):
        extra = sorted(set(present) - set(KERNEL_FILES))
        missing = sorted(set(KERNEL_FILES) - set(present))
        raise KernelTampered(f"{dest}: unexpected entries {extra}, missing {missing}")
    want = source_manifest()
    for name in KERNEL_FILES:
        st = os.lstat(dest / name)
        if not stat.S_ISREG(st.st_mode):
            raise KernelTampered(f"{dest / name} is not a regular file")
        if st.st_mode & 0o222:
            raise KernelTampered(f"{dest / name} is writable")
        if _sha256(dest / name) != want[name]:
            raise KernelTampered(f"{dest / name} does not match the harness copy")


def uninstall(dest: Path | str) -> None:
    """Make ``dest`` writable again and remove it (end of a session, tests)."""
    dest = Path(dest)
    if not dest.exists():
        return
    os.chmod(dest, 0o755)
    for p in dest.iterdir():
        if not p.is_symlink():
            os.chmod(p, 0o644)
    shutil.rmtree(dest)
