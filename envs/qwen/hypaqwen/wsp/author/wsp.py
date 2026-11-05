"""``from wsp import ...``: the name author scripts import the kernel client under (docs/V04_WSP_KERNEL.md §4).

A thin alias of ``client``. Functions and classes are copied here; the lazy metadata (``NAMES``,
``SHAPES``, ``ARCH``, ``LABEL_IDS``, ``CLASS_NAMES``, ``tok``) is forwarded on access, so importing
this module does not connect to the kernel.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _sibling(name: str):
    """``<this dir>/<name>.py`` loaded by explicit path, never through sys.path (shared via sys.modules).

    So ``from wsp import ...`` reuses the client the ``./py`` wrapper loaded before the guard.
    """
    import importlib.util

    key = f"hn_kernel_{name}"
    mod = sys.modules.get(key)
    if mod is None:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).resolve().parent / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[key]
            raise
    return mod


if __package__:
    from . import client as _client
else:  # imported as the top-level module ``wsp`` from tools/hn_kernel/
    _client = _sibling("client")

__all__ = list(_client.__all__)
globals().update({n: getattr(_client, n) for n in __all__ if n not in _client._LAZY})


def __getattr__(name: str) -> Any:
    return getattr(_client, name)
