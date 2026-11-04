"""Windowing, sequence validation and the runtime evidence gate (spec v0.2 §5; M0 contracts §B).

The gate is purely about instrumentation: it makes **no semantic judgement**.
A window of eight novel, fully instrumented INSPECT events passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Sequence

import numpy as np

from .encode import encode_window, validation_baseline
from .schema import EVENTS_PER_WINDOW, INPUT_DIM, CanonicalEvent, Outcome

__all__ = [
    "INSUFFICIENT_EVIDENCE",
    "PASS",
    "SequenceError",
    "GateReason",
    "GateDecision",
    "WindowResult",
    "validate_sequence",
    "gate",
    "window_endpoints",
    "iter_windows",
    "encode_session",
]

INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
PASS = "PASS"


class SequenceError(ValueError):
    """Raised when an event list is not one dense, ordered (session, agent, task) stream."""


class GateReason(str, Enum):
    TOO_SHORT = "TOO_SHORT"
    MISSING_TOKENS = "MISSING_TOKENS"
    MISSING_WORKSPACE_DIGEST = "MISSING_WORKSPACE_DIGEST"
    MISSING_FAILURE_FINGERPRINT = "MISSING_FAILURE_FINGERPRINT"
    MISSING_VALIDATION_FIELDS = "MISSING_VALIDATION_FIELDS"
    UNOBSERVED_VALIDATION_BASELINE = "UNOBSERVED_VALIDATION_BASELINE"


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reasons: tuple[tuple[int, GateReason], ...] = ()

    def __post_init__(self) -> None:
        if self.passed != (len(self.reasons) == 0):
            raise ValueError("GateDecision.passed must be True iff there are no reasons")

    @property
    def result(self) -> str:
        return PASS if self.passed else INSUFFICIENT_EVIDENCE


@dataclass(frozen=True)
class WindowResult:
    endpoint_index: int  # sequence_index of the window's last event
    gate: GateDecision
    features: np.ndarray | None  # (120,) float32 when gate passed, else None


def validate_sequence(events: Sequence[CanonicalEvent]) -> None:
    """Raise :class:`SequenceError` unless ``events`` is one dense ordered stream."""
    events = list(events)
    if not events:
        return
    for ev in events:
        if not isinstance(ev, CanonicalEvent):
            raise SequenceError(f"not a CanonicalEvent: {type(ev).__name__}")
    first = events[0]
    for name in ("session_id", "agent_id", "task_id"):
        values = {getattr(ev, name) for ev in events}
        if len(values) != 1:
            raise SequenceError(f"mixed {name}: {sorted(values)}")
    ids = [ev.event_id for ev in events]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise SequenceError(f"duplicate event_id(s): {dup}")
    for prev, cur in zip(events, events[1:]):
        if cur.sequence_index != prev.sequence_index + 1:
            raise SequenceError(
                f"sequence_index must increase by exactly 1: {prev.sequence_index} -> {cur.sequence_index}"
            )
    if first.provenance.history_complete and first.sequence_index != 0:
        raise SequenceError(
            f"history_complete session must start at sequence_index 0, starts at {first.sequence_index}"
        )


def gate(prefix: Sequence[CanonicalEvent]) -> GateDecision:
    """Runtime evidence gate (spec §5.2) for the window ending at the last event of ``prefix``."""
    events = list(prefix)
    n = len(events)
    if n < EVENTS_PER_WINDOW:
        return GateDecision(False, ((-1, GateReason.TOO_SHORT),))
    start = n - EVENTS_PER_WINDOW
    reasons: set[tuple[int, GateReason]] = set()
    for pos in range(EVENTS_PER_WINDOW):
        k = start + pos
        ev = events[k]
        if ev.input_tokens is None or ev.output_tokens is None:
            reasons.add((pos, GateReason.MISSING_TOKENS))
        if ev.workspace_state_before is None or ev.workspace_state_after is None:
            reasons.add((pos, GateReason.MISSING_WORKSPACE_DIGEST))
        if ev.outcome is Outcome.ERROR and ev.failure_fingerprint is None:
            reasons.add((pos, GateReason.MISSING_FAILURE_FINGERPRINT))
        if ev.validation_produced:
            if ev.validation_suite_id is None or ev.validation_passed is None or ev.validation_total is None:
                reasons.add((pos, GateReason.MISSING_VALIDATION_FIELDS))
            if ev.validation_suite_id is not None:
                _, observed = validation_baseline(events, k)
                if not observed:
                    reasons.add((pos, GateReason.UNOBSERVED_VALIDATION_BASELINE))
    ordered = tuple(sorted(reasons, key=lambda r: (r[0], r[1].value)))
    return GateDecision(not ordered, ordered)


def window_endpoints(events: Sequence[CanonicalEvent]) -> list[int]:
    """List positions ``t >= 7`` that end a full window."""
    return list(range(EVENTS_PER_WINDOW - 1, len(events)))


def iter_windows(events: Sequence[CanonicalEvent]) -> Iterator[WindowResult]:
    """Validate once, then gate and (if passed) encode each stride-1 window.

    Each window sees only ``events[:t+1]``; nothing after the endpoint is read.
    """
    events = list(events)
    validate_sequence(events)
    for t in window_endpoints(events):
        prefix = events[: t + 1]
        decision = gate(prefix)
        features = encode_window(prefix) if decision.passed else None
        yield WindowResult(endpoint_index=prefix[-1].sequence_index, gate=decision, features=features)


def encode_session(events: Sequence[CanonicalEvent]) -> tuple[list[int], np.ndarray]:
    """Endpoints (``sequence_index``) of passing windows and their stacked ``(n, 120)`` float32 features."""
    endpoints: list[int] = []
    rows: list[np.ndarray] = []
    for w in iter_windows(events):
        if w.gate.passed:
            endpoints.append(w.endpoint_index)
            rows.append(w.features)
    if not rows:
        return endpoints, np.zeros((0, INPUT_DIM), dtype=np.float32)
    return endpoints, np.ascontiguousarray(np.stack(rows), dtype=np.float32)
