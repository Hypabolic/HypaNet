"""v0.5 Phase C revision applier: Phase B's weights + ``REVISION.edits`` -> the revised state dict and a report.

Phase C (experiments/v0_5/phase_c/) improves the network of v0.5 Phase B (weight_program/v0_5/phase-b/artifact.dlw),
which is itself r1's network with Phase B's ``REVISION.edits`` applied on top. The Phase C author delivers a new
``REVISION.edits`` and the harness applies it **in order on top of Phase B's weights**.

The applier is Phase B's (:func:`agent_loop_lab.v05.revision.apply_revision`, imported and not modified): it has
always taken the base state dict as an argument, so the language, the file rules and the build verdict
(:func:`build_problems`) are exactly Phase B's. This module only names the base in the report
(:data:`REVISION_C_POLICY`, ``revision_version`` :data:`REVISION_C_VERSION`) so a Phase C apply report cannot be
mistaken for a Phase B one. Phase B's module, its report and its vendored copy stay byte for byte as they were.

This module imports only :mod:`.revision`, so a byte copy of it next to the copies of the reader and of
``revision.py`` works in the author's ``tools/lib/hn_v05/`` package (agent_loop_lab.v05.kit_c).
"""

from __future__ import annotations

from typing import Any, Mapping

from .revision import REVISION_FILE, REVISION_POLICY, REVISION_VERSION, apply_revision, build_problems

REVISION_C_VERSION = "v05c-revision-1"
REVISION_C_POLICY = {
    **REVISION_POLICY,
    "name": "agent_loop_lab.v05.revision_c",
    "version": REVISION_C_VERSION,
    "applier": f"agent_loop_lab.v05.revision.apply_revision ({REVISION_VERSION}, unchanged)",
    "base": ("v0.5 Phase B's weights (r1's weights with Phase B's REVISION.edits applied); statements apply in "
             "order on top of them"),
    "empty_file": "allowed: Phase B's network unchanged",
}

__all__ = [
    "REVISION_C_POLICY",
    "REVISION_C_VERSION",
    "REVISION_FILE",
    "apply_revision_c",
    "build_problems",
]


def apply_revision_c(base: Mapping[str, Any], text: str):
    """``(state dict, report)``: ``text``'s statements applied in order to a copy of ``base`` (Phase B's weights).

    Exactly :func:`agent_loop_lab.v05.revision.apply_revision`; the report names the Phase C base and version."""
    sd, report = apply_revision(base, text)
    report["revision"] = REVISION_C_POLICY
    report["revision_version"] = REVISION_C_VERSION
    return sd, report
