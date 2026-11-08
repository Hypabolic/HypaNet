"""inspect: per-sample probes, activations and S along one pixel's ray. See inspect_ray.py for usage.

    uv run python renderer/inspect.py NET.json --pair front_noon --pixel 320 200 [--samples 8|all]

A script named inspect.py shadows the standard library's `inspect` module for any script run from this
directory (Python puts the script's directory first on sys.path). So: when run as a script, this launcher moves
its directory to the end of sys.path and runs inspect_ray.py; when imported as `inspect` by accident, it loads
the real standard-library module in its place.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    import runpy

    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here] + [_here]
    runpy.run_path(os.path.join(_here, "inspect_ray.py"), run_name="__main__")
else:  # imported as `inspect`: become the standard library module
    import importlib.util
    import sysconfig

    _spec = importlib.util.spec_from_file_location("inspect", os.path.join(sysconfig.get_paths()["stdlib"], "inspect.py"))
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["inspect"] = _mod
    _spec.loader.exec_module(_mod)
    globals().update({k: v for k, v in _mod.__dict__.items() if k not in ("__name__", "__file__", "__spec__", "__loader__")})
