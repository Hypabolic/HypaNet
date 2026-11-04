"""Install the v0.3 weight author's ``tools/`` directory (spec v0.3 §8 "Isolation").

``install_tools(tools_dir)`` writes a self-contained copy of the AUTHOR_RUN guard
that works in the author's private venv, which has only ``torch`` and ``numpy``:

    tools/launch.py              experiments/v0_3/author_kit/launch.py (byte copy)
    tools/hn_guard/__init__.py   empty
    tools/hn_guard/guards_v02.py src/agent_loop_lab/guards.py with its one harness import
                                 (``from .model import DoomLoopMLP``, used only in an annotation)
                                 replaced by ``from torch.nn import Module as DoomLoopMLP``
    tools/hn_guard/guards_v03.py src/agent_loop_lab/v03/guards.py (byte copy); in a package other
                                 than ``agent_loop_lab.v03`` it imports the v0.2 guard from
                                 ``.guards_v02``

Usage: ``python -m agent_loop_lab.v03.kit install-tools TOOLS_DIR`` prints the sha256 of every
file written, as JSON.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

__all__ = ["V02_IMPORT_LINE", "STANDALONE_IMPORT_LINE", "standalone_v02_guards", "install_tools", "LAUNCHER_SOURCE"]

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[2]
V02_GUARDS = _HERE.parent / "guards.py"
V03_GUARDS = _HERE / "guards.py"
LAUNCHER_SOURCE = REPO_ROOT / "experiments" / "v0_3" / "author_kit" / "launch.py"

V02_IMPORT_LINE = "from .model import DoomLoopMLP"
STANDALONE_IMPORT_LINE = "from torch.nn import Module as DoomLoopMLP  # standalone copy (author tools/): annotation only"


def standalone_v02_guards(text: str) -> str:
    lines = text.split("\n")
    hits = [i for i, line in enumerate(lines) if line == V02_IMPORT_LINE]
    if len(hits) != 1:
        raise RuntimeError(f"v0.2 guards.py: expected exactly one {V02_IMPORT_LINE!r} line, found {len(hits)}")
    lines[hits[0]] = STANDALONE_IMPORT_LINE
    out = "\n".join(lines)
    if "from ." in out:
        raise RuntimeError("v0.2 guards.py has another relative import; the standalone copy would not import")
    return out


def install_tools(tools_dir: str | Path, launcher: str | Path | None = None) -> dict[str, str]:
    tools = Path(tools_dir)
    pkg = tools / "hn_guard"
    pkg.mkdir(parents=True, exist_ok=False)
    files = {
        pkg / "__init__.py": b"",
        pkg / "guards_v02.py": standalone_v02_guards(V02_GUARDS.read_text(encoding="utf-8")).encode("utf-8"),
        pkg / "guards_v03.py": V03_GUARDS.read_bytes(),
        tools / "launch.py": Path(launcher or LAUNCHER_SOURCE).read_bytes(),
    }
    for path, data in files.items():
        path.write_bytes(data)
    return {p.relative_to(tools).as_posix(): hashlib.sha256(d).hexdigest() for p, d in sorted(files.items())}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "install-tools":
        print(__doc__, file=sys.stderr)
        return 2
    print(json.dumps(install_tools(argv[1]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
