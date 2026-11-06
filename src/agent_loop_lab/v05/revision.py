"""v0.5 Phase B revision applier: r1's weights + ``REVISION.edits`` -> the revised state dict and a report.

Phase B (experiments/v0_5/phase_b/) improves the network r1 wrote. The author delivers ``REVISION.edits``,
statements of the v0.5 edits language, and the harness applies them **in order on top of r1's weights** (not
from zero) with the frozen reader's own lexer, parser and apply step (:mod:`agent_loop_lab.v05.edits`, imported,
not modified). The rules are the reader's (indices, ranges, list shapes, numbers, ``=``/``+=``/``-=``/``*=``,
``#`` comments, statements continuing over lines while a ``[`` is open or after a trailing operator or comma),
with these Phase B rules for a file:

* the whole file is the body of one edits block: it holds statements, comments and blank lines only. It has no
  fence lines (a line of three or more backticks or tildes) and no ``END OF WEIGHTS`` line; each such line is a
  structural problem;
* the block ends at the end of the file and is never "cut off": a statement still open there (an unclosed ``[``,
  a trailing operator or comma) is an error, as in a block that was closed normally;
* the reader skips a statement it cannot read or apply and goes on (that is what it did for r1); the Phase B
  **build** is stricter: :func:`build_problems` lists every skipped statement and every structural line, and the
  harness (scripts/v05b_run_build.py) rejects a revision that has any. The author checks the same thing with
  ``./py -m v05_load check`` before finishing;
* an empty file (or comments only) is allowed: the revised network is then r1's network unchanged.

This module imports only :mod:`.edits` (plus numpy and torch), so a byte copy of it next to a copy of the reader
works in the author's ``tools/lib/hn_v05/`` package (agent_loop_lab.v05.kit_b). The serialised artefact
(DLW3) is written by the build script, not here.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Mapping

import numpy as np
import torch

from .edits import (
    END_MARKER,
    MAX_ERRORS_LISTED,
    PARAM_COUNT,
    PARAM_SHAPES,
    POLICY,
    READER_VERSION,
    Block,
    EditError,
    _apply,
    _FENCE,
    _split_lines,
    parse_statement,
    split_statements,
)

REVISION_VERSION = "v05b-revision-1"
REVISION_FILE = "REVISION.edits"
CLOSED_BY = "end_of_file"
PROBLEMS_LISTED = 50

REVISION_POLICY = {
    "name": "agent_loop_lab.v05.revision",
    "version": REVISION_VERSION,
    "base": "r1's weights (the frozen reader's artefact of r1's replies); statements apply in order on top of them",
    "file": "the whole file is the body of one edits block, closed at the end of the file (never cut off)",
    "fence_and_end_marker_lines": "not allowed (structural problem)",
    "skipped_statements": "the reader skips them; the Phase B build rejects a revision with any",
    "empty_file": "allowed: r1's network unchanged",
}

__all__ = [
    "REVISION_FILE",
    "REVISION_POLICY",
    "REVISION_VERSION",
    "apply_revision",
    "build_problems",
    "revision_block",
]


def revision_block(text: str) -> tuple[Block, list[dict[str, Any]]]:
    """The file as one edits block, and its structural problems (fence or END OF WEIGHTS lines, left out)."""
    block = Block(reply=1, index_in_reply=1, open_line=0, close_line=None, closed_by=CLOSED_BY)
    structural: list[dict[str, Any]] = []
    for line in _split_lines(text):
        if _FENCE.match(line.text):
            structural.append({"line": line.number, "text": line.text[:120],
                               "reason": "fence line: REVISION.edits holds statements only (no ``` fences)"})
            continue
        if line.text.strip() == END_MARKER:
            structural.append({"line": line.number, "text": line.text[:120],
                               "reason": "END OF WEIGHTS line: not used in REVISION.edits"})
            continue
        block.lines.append(line)
    return block, structural


def _state_from(base: Mapping[str, Any]) -> "OrderedDict[str, np.ndarray]":
    keys = list(base.keys())
    if keys != list(PARAM_SHAPES.keys()):
        raise ValueError("base state dict keys differ from the v0.3 template (names or order)")
    out: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for k, shape in PARAM_SHAPES.items():
        v = base[k]
        arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        if tuple(arr.shape) != shape:
            raise ValueError(f"base {k}: shape {tuple(arr.shape)} != {shape}")
        if arr.dtype != np.float32:
            raise ValueError(f"base {k}: dtype {arr.dtype} != float32")
        if not bool(np.isfinite(arr).all()):
            raise ValueError(f"base {k}: non-finite values")
        out[k] = np.array(arr, dtype=np.float32, copy=True)
    return out


def apply_revision(base: Mapping[str, Any], text: str) -> tuple["OrderedDict[str, torch.Tensor]", dict[str, Any]]:
    """Apply the statements of ``text`` (a REVISION.edits file) in order to a copy of ``base``.

    ``base``: a v0.3 state dict (torch tensors or numpy arrays, float32, template order). Returns the revised
    state dict (new tensors; ``base`` is not changed) and the report. Never raises for a bad statement: it is
    skipped and recorded, as the reader does; :func:`build_problems` turns the report into the build's verdict.
    """
    before = _state_from(base)
    state = OrderedDict((k, a.copy()) for k, a in before.items())
    block, structural = revision_block(text)
    errors: list[dict[str, Any]] = []
    n_err = 0
    applied = 0
    entries = 0
    statements = 0
    touched: set[str] = set()
    for raw in split_statements(block):
        statements += 1
        where = {"line": raw.first_line, "end_line": raw.last_line, "statement": raw.text}
        try:
            if raw.broken:
                raise EditError(raw.broken)
            st = parse_statement(raw.tokens)
            written = _apply(state, st)
        except EditError as exc:
            n_err += 1
            if len(errors) < MAX_ERRORS_LISTED:
                errors.append({**where, "reason": str(exc)})
            continue
        applied += 1
        entries += written
        touched.add(st.name)
    changed = {k: int(np.count_nonzero(state[k] != before[k])) for k in PARAM_SHAPES}
    nonzero_before = {k: int(np.count_nonzero(a)) for k, a in before.items()}
    nonzero_after = {k: int(np.count_nonzero(a)) for k, a in state.items()}
    report: dict[str, Any] = {
        "reader": POLICY,
        "reader_version": READER_VERSION,
        "revision": REVISION_POLICY,
        "revision_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "lines": len(block.lines) + len(structural),
        "structural_problems": structural,
        "edit_report": {
            "statements": statements,
            "applied": applied,
            "skipped": n_err,
            "entries_written": entries,
            "errors": errors,
            "errors_listed": len(errors),
            "tensors_touched": [k for k in PARAM_SHAPES if k in touched],
        },
        "changed_entries": {k: v for k, v in changed.items() if v},
        "changed_entries_total": sum(changed.values()),
        "nonzero_before_total": sum(nonzero_before.values()),
        "nonzero_after_total": sum(nonzero_after.values()),
        "nonzero_after": nonzero_after,
        "max_abs_after": {k: float(np.abs(a).max(initial=0.0)) for k, a in state.items()},
        "parameters": PARAM_COUNT,
    }
    sd = OrderedDict((k, torch.from_numpy(a.copy())) for k, a in state.items())
    return sd, report


def build_problems(report: Mapping[str, Any], limit: int = PROBLEMS_LISTED) -> list[str]:
    """Why the Phase B build would reject this revision (empty list: it would be accepted).

    One line per structural line and per skipped statement, ``line N: reason: statement``; at most ``limit``
    lines, then a count of the rest."""
    out: list[str] = []
    for p in report.get("structural_problems") or []:
        out.append(f"line {p['line']}: {p['reason']}")
    er = report.get("edit_report") or {}
    for e in er.get("errors") or []:
        lines = f"line {e['line']}" if e.get("end_line") in (None, e["line"]) else f"lines {e['line']}-{e['end_line']}"
        stmt = " ".join(str(e.get("statement", "")).split())
        out.append(f"{lines}: {e['reason']}: {stmt[:160]}")
    unlisted = int(er.get("skipped") or 0) - len(er.get("errors") or [])
    total = len(out) + max(0, unlisted)
    if len(out) > limit:
        out = out[:limit]
    if total > len(out):
        out.append(f"... and {total - len(out)} more")
    return out
