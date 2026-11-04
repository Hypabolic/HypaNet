"""Shared test helpers for building canonical events and sessions.

Not a generator: these are hand-specified fixtures for unit tests only.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from agent_loop_lab.schema import CanonicalEvent, OperationCategory, Outcome, Provenance


def h(label: str) -> str:
    """Deterministic 64-hex digest for a readable label (e.g. h('state-A'))."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


DEFAULT_PROVENANCE = Provenance(source="test:factory", history_complete=True)

# Sentinel for "argument not given", so that an explicit ``after=None`` can mean
# "unobserved" while leaving ``after`` out means "same as ``before``".
_UNSET: Any = object()


def make_event(
    i: int,
    category: str | OperationCategory = "INSPECT",
    outcome: str | Outcome = "SUCCESS",
    *,
    intent: str | None = None,
    failure: str | None = None,
    before: str | None = "S0",
    after: str | None = _UNSET,
    validation: tuple[str, int, int] | None = None,
    tokens: tuple[int | None, int | None] = (100, 50),
    family: str | None = None,
    session_id: str = "sess-1",
    provenance: Provenance = DEFAULT_PROVENANCE,
    **overrides: Any,
) -> CanonicalEvent:
    """Build one event.

    ``intent`` / ``failure`` / ``before`` / ``after`` are readable labels hashed
    with :func:`h`; pass ``None`` for an unobserved value. Leaving ``after`` out
    makes it equal to ``before`` (no workspace change); an explicit
    ``after=None`` makes the after-digest unobserved. ``validation`` is
    ``(suite, passed, total)``.
    When ``outcome == 'ERROR'`` and ``failure`` is not given, a per-event unique
    failure fingerprint is used.
    """
    cat = OperationCategory(category)
    out = Outcome(outcome)
    if after is _UNSET:
        after = before
    if out is Outcome.ERROR and failure is None and "failure_fingerprint" not in overrides:
        failure = f"auto-failure-{i}"
    kw: dict[str, Any] = dict(
        session_id=session_id,
        agent_id="agent-1",
        task_id="task-1",
        event_id=f"{session_id}-e{i}",
        sequence_index=i,
        timestamp=f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z",
        operation_family=family or cat.value.lower(),
        operation_category=cat,
        outcome=out,
        intent_fingerprint=h(intent if intent is not None else f"intent-{i}"),
        failure_fingerprint=h(failure) if (failure is not None and out is Outcome.ERROR) else None,
        workspace_state_before=h(before) if before is not None else None,
        workspace_state_after=h(after) if after is not None else None,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        provenance=provenance,
    )
    if validation is not None:
        suite, passed, total = validation
        kw.update(
            validation_produced=True,
            validation_suite_id=suite,
            validation_passed=passed,
            validation_total=total,
        )
    kw.update(overrides)
    return CanonicalEvent(**kw)


def session(specs: Iterable[dict[str, Any]], **common: Any) -> list[CanonicalEvent]:
    """Build a dense session from a list of per-event kwargs dicts."""
    return [make_event(i, **{**common, **spec}) for i, spec in enumerate(specs)]


def novel_inspections(n: int = 8) -> list[CanonicalEvent]:
    """n distinct INSPECT/SUCCESS events, no workspace change, no validation."""
    return session([{"category": "INSPECT", "outcome": "SUCCESS", "intent": f"read-{i}"} for i in range(n)])
