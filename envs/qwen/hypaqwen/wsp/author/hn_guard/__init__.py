"""HypaNet v0.4 author-side guards (docs/V04_WSP_KERNEL.md §5); ships into the author dir as ``tools/hn_guard/``.

Self-contained: the standard library, ``torch``, ``numpy`` and (optionally) ``safetensors`` and
``tokenizers``. See ``guards_v04`` for the contract.
"""

from .guards_v04 import (
    BUILD_EDIT,
    EXIT_VIOLATION,
    KERNEL_CLIENT,
    MODES,
    GuardViolation,
    ImportViolation,
    IOViolation,
    TrainingViolation,
    clean_run,
    control_normal,
    finish,
    install,
    installed_mode,
    violation_count,
)

__all__ = [
    "BUILD_EDIT",
    "EXIT_VIOLATION",
    "KERNEL_CLIENT",
    "MODES",
    "GuardViolation",
    "ImportViolation",
    "IOViolation",
    "TrainingViolation",
    "clean_run",
    "control_normal",
    "finish",
    "install",
    "installed_mode",
    "violation_count",
]
