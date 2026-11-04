"""Frozen 15-feature-per-event encoder (spec v0.2 §4; M0 contracts §B).

``prefix`` is always the *session prefix* up to and including the window
endpoint. The window is the last ``EVENTS_PER_WINDOW`` events of the prefix.
Events before the window are used **only** to find validation baselines
(features 12-13). Nothing after the endpoint is ever seen, by construction.

Missing observations are never mapped to 0: the encoder raises
:class:`EncodingError` instead (spec §4 "Missing data"). The runtime gate in
``windows.py`` rejects such windows before they reach the encoder.

This module must not import ``windows.py``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .schema import (
    CATEGORY_ORDER,
    EVENTS_PER_WINDOW,
    FEATURE_INDEX,
    FEATURE_NAMES,
    FEATURES_PER_EVENT,
    INPUT_DIM,
    OUTCOME_ORDER,
    CanonicalEvent,
    Outcome,
)

__all__ = [
    "EncodingError",
    "TOKEN_LOG_SCALE",
    "TOKEN_SATURATION",
    "encode_event_matrix",
    "encode_window",
    "describe_matrix",
    "history_complete",
    "previous_validation_result",
    "relative_token_cost",
    "validation_baseline",
]


class EncodingError(ValueError):
    """Raised when a prefix cannot be encoded without inventing a value."""


TOKEN_SATURATION = 100000
TOKEN_LOG_SCALE = math.log1p(TOKEN_SATURATION)

_CAT_COLS = tuple(FEATURE_INDEX[f"cat_{c.value}"] for c in CATEGORY_ORDER)
_OUT_COLS = tuple(FEATURE_INDEX[f"out_{o.value}"] for o in OUTCOME_ORDER)
_I_REPEATED_INTENT = FEATURE_INDEX["repeated_intent"]
_I_REPEATED_FAILURE = FEATURE_INDEX["repeated_failure"]
_I_WORKSPACE_CHANGED = FEATURE_INDEX["workspace_changed"]
_I_STATE_REVISITED = FEATURE_INDEX["state_revisited"]
_I_VALIDATION_COMPARABLE = FEATURE_INDEX["validation_comparable"]
_I_VALIDATION_DELTA = FEATURE_INDEX["validation_delta"]
_I_TOKEN_COST = FEATURE_INDEX["relative_token_cost"]


# ---------------------------------------------------------------------------
# Shared helpers (also used by windows.py's gate so both agree exactly)
# ---------------------------------------------------------------------------


def history_complete(prefix: Sequence[CanonicalEvent]) -> bool:
    """True iff the prefix is known to start at the session's first operation.

    Uses the first event's ``provenance.history_complete``. A prefix whose
    first event is not ``sequence_index == 0`` is treated as truncated even if
    flagged complete (such a sequence is rejected by ``validate_sequence``);
    this stops a bare window slice from silently losing its baselines.
    """
    if not prefix:
        return False
    first = prefix[0]
    return bool(first.provenance.history_complete) and first.sequence_index == 0


def previous_validation_result(prefix: Sequence[CanonicalEvent], k: int) -> CanonicalEvent | None:
    """Most recent event before position ``k`` of ``prefix`` that produced a
    validation result from the same (non-None) suite as ``prefix[k]``.

    Returns ``None`` if ``prefix[k]`` produced no validation result, has no
    suite id, or no earlier same-suite result exists in the prefix.

    This is a plain lookup: earlier results whose suite id is unobserved are
    skipped. Features 12-13 and the gate use :func:`validation_baseline`,
    which treats such a result as making the baseline unobserved.
    """
    ev = prefix[k]
    if not ev.validation_produced or ev.validation_suite_id is None:
        return None
    suite = ev.validation_suite_id
    for j in range(k - 1, -1, -1):
        prev = prefix[j]
        if prev.validation_produced and prev.validation_suite_id == suite:
            return prev
    return None


def validation_baseline(prefix: Sequence[CanonicalEvent], k: int) -> tuple[CanonicalEvent | None, bool]:
    """Baseline for features 12-13 of ``prefix[k]``, as ``(baseline, observed)``.

    Scans backward from ``k - 1`` for the most recent validation result from the
    same suite as ``prefix[k]``. ``observed`` is False (baseline unobserved,
    spec §4 "Missing data" / §5.2 item 2) when:

    * an earlier validation result whose ``validation_suite_id`` is unobserved
      (None) is reached before any same-suite result: it may itself be the most
      recent same-suite result, so neither existence (feature 12) nor value
      (feature 13) of the baseline is known;
    * the most recent same-suite result has unobserved passed/total;
    * no earlier same-suite result exists and the history is truncated.

    Otherwise ``observed`` is True and ``baseline`` is the same-suite result,
    or None when none exists in a complete history (comparable = 0).
    ``prefix[k]`` must have produced a validation result with a suite id.
    """
    ev = prefix[k]
    if not ev.validation_produced or ev.validation_suite_id is None:
        raise ValueError("validation_baseline needs a validation result with an observed suite id")
    suite = ev.validation_suite_id
    for j in range(k - 1, -1, -1):
        prev = prefix[j]
        if not prev.validation_produced:
            continue
        if prev.validation_suite_id is None:
            return None, False
        if prev.validation_suite_id == suite:
            observed = prev.validation_passed is not None and prev.validation_total is not None
            return prev, observed
    return None, history_complete(prefix)


def relative_token_cost(input_tokens: int, output_tokens: int) -> float:
    """``min(1, log1p(in + out) / log1p(100000))`` in float64. 0 tokens -> 0.0."""
    total = int(input_tokens) + int(output_tokens)
    if total < 0:
        raise EncodingError("token counts must be non-negative")
    return min(1.0, math.log1p(total) / TOKEN_LOG_SCALE)


# ---------------------------------------------------------------------------
# Precondition checks (mirror the gate; never zero-fill)
# ---------------------------------------------------------------------------


def _check_window_event(ev: CanonicalEvent, pos: int) -> None:
    where = f"window position {pos} (sequence_index {ev.sequence_index})"
    if ev.operation_category is None or ev.outcome is None:
        raise EncodingError(f"{where}: missing category or outcome")
    if ev.input_tokens is None or ev.output_tokens is None:
        raise EncodingError(f"{where}: missing token count")
    if ev.workspace_state_before is None or ev.workspace_state_after is None:
        raise EncodingError(f"{where}: missing workspace digest")
    if ev.outcome is Outcome.ERROR and ev.failure_fingerprint is None:
        raise EncodingError(f"{where}: ERROR without failure_fingerprint")
    if ev.validation_produced and (
        ev.validation_suite_id is None or ev.validation_passed is None or ev.validation_total is None
    ):
        raise EncodingError(f"{where}: validation result with unobserved suite/passed/total")


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def encode_event_matrix(prefix: Sequence[CanonicalEvent]) -> np.ndarray:
    """Encode the last 8 events of ``prefix`` as a float32 ``(8, 15)`` matrix.

    Rows are window positions, oldest first; columns follow ``FEATURE_NAMES``.
    """
    events = list(prefix)
    n = len(events)
    if n < EVENTS_PER_WINDOW:
        raise EncodingError(f"prefix has {n} events; need at least {EVENTS_PER_WINDOW}")
    start = n - EVENTS_PER_WINDOW
    window = events[start:]
    for pos, ev in enumerate(window):
        _check_window_event(ev, pos)

    mat = np.zeros((EVENTS_PER_WINDOW, FEATURES_PER_EVENT), dtype=np.float32)
    first_before = window[0].workspace_state_before

    for pos, ev in enumerate(window):
        row = mat[pos]
        earlier = window[:pos]

        # 0-4 category one-hot, 5-7 outcome one-hot
        row[_CAT_COLS[CATEGORY_ORDER.index(ev.operation_category)]] = 1.0
        row[_OUT_COLS[OUTCOME_ORDER.index(ev.outcome)]] = 1.0

        # 8 repeated_intent (window-local)
        if any(e.intent_fingerprint == ev.intent_fingerprint for e in earlier):
            row[_I_REPEATED_INTENT] = 1.0

        # 9 repeated_failure (window-local, ERROR vs earlier ERROR)
        if ev.outcome is Outcome.ERROR and any(
            e.outcome is Outcome.ERROR and e.failure_fingerprint == ev.failure_fingerprint for e in earlier
        ):
            row[_I_REPEATED_FAILURE] = 1.0

        # 10 workspace_changed
        changed = ev.workspace_state_before != ev.workspace_state_after
        if changed:
            row[_I_WORKSPACE_CHANGED] = 1.0

            # 11 state_revisited (window-local, includes window[0].before)
            seen = {first_before}
            for e in earlier:
                seen.add(e.workspace_state_before)
                seen.add(e.workspace_state_after)
            if ev.workspace_state_after in seen:
                row[_I_STATE_REVISITED] = 1.0

        # 12-13 validation_comparable / validation_delta (session prefix)
        if ev.validation_produced:
            k = start + pos
            prev, observed = validation_baseline(events, k)
            if not observed:
                raise EncodingError(
                    f"window position {pos}: '{ev.validation_suite_id}' baseline unobserved (truncated "
                    "history, unobserved earlier suite id, or unobserved passed/total)"
                )
            if prev is not None:
                cur = ev.validation_passed / ev.validation_total
                base = prev.validation_passed / prev.validation_total
                delta = min(1.0, max(-1.0, cur - base))
                row[_I_VALIDATION_COMPARABLE] = 1.0
                row[_I_VALIDATION_DELTA] = np.float32(delta)

        # 14 relative_token_cost
        row[_I_TOKEN_COST] = np.float32(relative_token_cost(ev.input_tokens, ev.output_tokens))

    return mat


def encode_window(prefix: Sequence[CanonicalEvent]) -> np.ndarray:
    """Flattened float32 ``(120,)`` input vector; equals ``encode_event_matrix(prefix).reshape(-1)``."""
    vec = np.ascontiguousarray(encode_event_matrix(prefix).reshape(-1), dtype=np.float32)
    assert vec.shape == (INPUT_DIM,)
    return vec


def describe_matrix(matrix: np.ndarray) -> str:
    """Human-readable table: one row per window position, columns named by ``FEATURE_NAMES``."""
    m = np.asarray(matrix)
    if m.shape == (INPUT_DIM,):
        m = m.reshape(EVENTS_PER_WINDOW, FEATURES_PER_EVENT)
    if m.shape != (EVENTS_PER_WINDOW, FEATURES_PER_EVENT):
        raise ValueError(f"expected shape (8, 15) or (120,), got {m.shape}")
    widths = [max(len(name), 6) for name in FEATURE_NAMES]
    header = "pos | " + " | ".join(name.rjust(w) for name, w in zip(FEATURE_NAMES, widths))
    lines = [header, "-" * len(header)]
    for pos in range(EVENTS_PER_WINDOW):
        cells = []
        for col, w in enumerate(widths):
            v = float(m[pos, col])
            text = f"{v:.0f}" if v in (0.0, 1.0) else f"{v:+.4f}" if col == _I_VALIDATION_DELTA else f"{v:.4f}"
            cells.append(text.rjust(w))
        lines.append(f"{pos:>3} | " + " | ".join(cells))
    return "\n".join(lines)
