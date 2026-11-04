"""Shape / dtype / finiteness check for a weight author's build (spec v0.2 §9.2).

Usage (from the author working directory):
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/shape_check.py output/phase-a-strict

Imports ``build_weights.py`` from the given phase directory with ``packet/`` on the
module path, calls ``build()`` and checks the returned model's state_dict against the
packet's ``PARAM_SHAPES``. It performs **no forward pass** and never calls
``reference_policy``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    here = Path(__file__).resolve().parent.parent
    phase_dir = Path(argv[1]).resolve()
    sys.path.insert(0, str(here / "packet"))
    import model  # the packet's model.py

    spec = importlib.util.spec_from_file_location("build_weights", phase_dir / "build_weights.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    net = mod.build()
    sd = net.state_dict()
    try:
        model.validate_state_dict(sd)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1
    print("OK: keys, shapes, float32 dtype and finiteness match the template")
    for name, row in model.sparsity(net).items():
        print(f"  {name:12s} nonzero {row['nonzero']:5d} / {row['slots']:5d}  density {row['density']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
