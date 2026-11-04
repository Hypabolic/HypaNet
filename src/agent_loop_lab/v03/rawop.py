"""Raw operation record (spec v0.3 §3) and the generators' capture side channel (§7.1).

Each generator records, for every emitted ``CanonicalEvent``, the concrete
operation it executed: the exact ``target`` and ``args`` it passed to
``intent_fingerprint``, the ``error_class`` / ``error_location`` it passed to
``failure_fingerprint``, the true validation result, the file text before and
after a mutation, and the token counts and timestamp *as emitted*. Recording
draws no random numbers and changes no generator state.

A ``RawOp`` never carries latent effect, progress, episode state, the
scenario family or id, fault ids, facts or anything else derived from the
oracle. :data:`FIELDS` is the complete whitelist, and :meth:`RawOp.from_dict`
rejects any other key.

``dropped`` uses one vocabulary for both engines (:data:`DROPPED_VOCAB`, spec v0.3 §3):
``input_tokens output_tokens digest_before digest_after failure validation_suite
validation_passed validation_total``. See :data:`STANDARD_DROP_MAP` / :data:`OOD_DROP_MAP`.

``changed`` lists every tracked file whose text differs between the start and the end of
the operation, any family, sorted by path: ``[{"path", "before", "after"}]`` with ``None``
for an absent file. ``dropped`` lists only
observations the generator deliberately withheld on this event. It does not
include the baseline-propagation marks that the oracle adds in
``OracleRecord.dropped_observations`` (those are oracle-derived).

Capture: :func:`capture_raw` opens a sink. While one is open, each engine's
``finish`` hands the sink ``(session_id, events, raw)`` for the session it
seals. The sink is module state, never an attribute of a session, so it is
invisible to ``Session.fork`` (deepcopy) and to latent snapshots.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, fields
from typing import Any, Iterator, Mapping, Sequence

__all__ = [
    "DROPPED_VOCAB",
    "FIELDS",
    "OUTCOMES",
    "STANDARD_DROP_MAP",
    "OOD_DROP_MAP",
    "RawOp",
    "RawOpError",
    "RawCapture",
    "capture_raw",
    "emit_finished",
    "normalise_standard_drop",
    "normalise_ood_drop",
    "json_safe",
]

DROPPED_VOCAB: tuple[str, ...] = (
    "digest_after", "digest_before", "failure", "input_tokens", "output_tokens",
    "validation_passed", "validation_suite", "validation_total",
)
OUTCOMES: tuple[str, ...] = ("SUCCESS", "ERROR", "NEUTRAL")

STANDARD_DROP_MAP: Mapping[str, tuple[str, ...]] = {
    "tokens": ("input_tokens", "output_tokens"),
    "input_tokens": ("input_tokens",),
    "output_tokens": ("output_tokens",),
    "digests": ("digest_before", "digest_after"),
    "digest_before": ("digest_before",),
    "digest_after": ("digest_after",),
    "failure": ("failure",),
    "validation_counts": ("validation_passed", "validation_total"),
    "validation_suite": ("validation_suite",),
}
OOD_DROP_MAP: Mapping[str, tuple[str, ...]] = {
    "tokens": ("input_tokens",),  # the OOD engine nulls input_tokens only
    "workspace": ("digest_before", "digest_after"),
    "failure": ("failure",),
    "validation_suite": ("validation_suite",),
    "validation_passed": ("validation_passed",),
    "validation_total": ("validation_total",),
}


class RawOpError(ValueError):
    """A RawOp field is missing, unknown or not JSON-safe."""


def json_safe(obj: Any, where: str = "$") -> Any:
    """A plain-JSON copy of ``obj`` (tuples -> lists). Rejects floats that are not
    finite, non-str keys and any non-JSON type."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise RawOpError(f"non-finite float at {where}")
        return obj
    if isinstance(obj, (list, tuple)):
        return [json_safe(v, f"{where}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise RawOpError(f"non-str key {k!r} at {where}")
            out[k] = json_safe(v, f"{where}.{k}")
        return out
    raise RawOpError(f"{type(obj).__name__} at {where} is not JSON-safe")


def _norm(drops: Sequence[str], table: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    out: set[str] = set()
    for d in drops:
        if d not in table:
            raise RawOpError(f"unknown drop field {d!r}; expected one of {sorted(table)}")
        out.update(table[d])
    return tuple(sorted(out))


def normalise_standard_drop(drops: Sequence[str]) -> tuple[str, ...]:
    """Standard-engine ``drop=`` fields -> sorted :data:`DROPPED_VOCAB` entries."""
    return _norm(drops, STANDARD_DROP_MAP)


def normalise_ood_drop(dropped: str | None, *, is_error: bool, produced: bool) -> tuple[str, ...]:
    """OOD-engine ``dropped=`` -> sorted :data:`DROPPED_VOCAB` entries.

    Only observations the canonical event would otherwise have carried count:
    the OOD engine accepts ``dropped="failure"`` on a passing run and
    validation drops on a non-validation event, where nothing is withheld.
    """
    if dropped is None:
        return ()
    if dropped == "failure" and not is_error:
        return ()
    if dropped.startswith("validation") and not produced:
        return ()
    return _norm((dropped,), OOD_DROP_MAP)


@dataclass(frozen=True)
class RawOp:
    """One executed operation (spec v0.3 §3). See the module docstring."""

    family: str
    target: Any
    args: dict[str, Any]
    outcome: str
    error_class: str | None = None
    error_location: str | None = None
    validation: dict[str, Any] | None = None
    changed: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    timestamp: str | None = None
    dropped: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family:
            raise RawOpError("family must be a non-empty str")
        if self.outcome not in OUTCOMES:
            raise RawOpError(f"outcome {self.outcome!r} not in {OUTCOMES}")
        if not isinstance(self.args, dict):
            raise RawOpError("args must be a dict")
        if self.changed is not None:
            for c in self.changed:
                if set(c) != {"path", "before", "after"}:
                    raise RawOpError(f"changed entry keys {sorted(c)}")
        if self.validation is not None and set(self.validation) != {"suite", "passed", "total"}:
            raise RawOpError(f"validation keys {sorted(self.validation)} != ['passed', 'suite', 'total']")
        for d in self.dropped:
            if d not in DROPPED_VOCAB:
                raise RawOpError(f"dropped entry {d!r} not in {DROPPED_VOCAB}")
        if list(self.dropped) != sorted(set(self.dropped)):
            raise RawOpError("dropped must be sorted and unique")

    @classmethod
    def make(cls, *, family: str, target: Any, args: Mapping[str, Any], outcome: Any,
             error_class: str | None = None, error_location: str | None = None,
             validation: tuple[str | None, int | None, int | None] | None = None,
             changed: Sequence[Mapping[str, Any]] | None = None,
             extra: Mapping[str, Any] | None = None, input_tokens: int | None = None,
             output_tokens: int | None = None, timestamp: str | None = None,
             dropped: Sequence[str] = ()) -> "RawOp":
        """Build from engine values: copies ``target``/``args``/``extra`` to plain
        JSON (so later engine mutation can't alias), ``outcome`` enum -> str."""
        val = None
        if validation is not None:
            suite, passed, total = validation
            val = {"suite": suite, "passed": passed, "total": total}
        return cls(
            family=family,
            target=json_safe(target, "target"),
            args=json_safe(dict(args), "args"),
            outcome=getattr(outcome, "value", outcome),
            error_class=error_class,
            error_location=error_location,
            validation=val,
            changed=sorted(({"path": c["path"], "before": c["before"], "after": c["after"]} for c in changed),
                           key=lambda c: c["path"]) if changed else [],
            extra=json_safe(dict(extra), "extra") if extra else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=timestamp,
            dropped=tuple(sorted(set(dropped))),
        )

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["dropped"] = list(self.dropped)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RawOp":
        keys = set(d)
        if keys != set(FIELDS):
            raise RawOpError(f"RawOp keys: unknown {sorted(keys - set(FIELDS))}, missing {sorted(set(FIELDS) - keys)}")
        kw = dict(d)
        kw["dropped"] = tuple(kw["dropped"])
        return cls(**kw)


FIELDS: tuple[str, ...] = tuple(f.name for f in fields(RawOp))


# ---------------------------------------------------------------------------
# Capture side channel
# ---------------------------------------------------------------------------


class RawCapture:
    """Sessions sealed while this sink is open, by session id, in finish order.

    Several sealed sessions may share an id (the OOD counterfactual branches
    do, and a family could seal a session it then discards). :meth:`pop`
    therefore matches on the identity of the events list the caller received,
    so raw ops are never paired with the wrong events.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, list[tuple[list, list[RawOp]]]] = {}
        self.finished = 0

    def _add(self, session_id: str, events: list, raw: Sequence[RawOp]) -> None:
        self._by_id.setdefault(session_id, []).append((events, list(raw)))
        self.finished += 1

    def pop(self, session_id: str, events: list | None = None) -> list[RawOp]:
        """Raw ops of the sealed session whose ``finish`` returned ``events``
        (identity); with ``events=None``, of the latest one under ``session_id``."""
        entries = self._by_id.get(session_id)
        if not entries:
            raise RawOpError(f"no captured raw ops for session {session_id!r}")
        if events is None:
            i = len(entries) - 1
        else:
            for i in range(len(entries) - 1, -1, -1):
                if entries[i][0] is events:
                    break
            else:
                raise RawOpError(f"session {session_id!r}: no captured raw ops for this events list")
        _, raw = entries.pop(i)
        if not entries:
            del self._by_id[session_id]
        return raw

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_id.values())

    def pending(self) -> list[str]:
        return sorted(self._by_id)


_SINKS: list[RawCapture] = []


@contextlib.contextmanager
def capture_raw() -> Iterator[RawCapture]:
    """Collect the raw ops of every session sealed inside the ``with`` block."""
    sink = RawCapture()
    _SINKS.append(sink)
    try:
        yield sink
    finally:
        _SINKS.remove(sink)


def emit_finished(session_id: str, events: list, raw: Sequence[RawOp]) -> None:
    """Engine hook, called by ``finish``: hand a sealed session to open sinks (no-op if none)."""
    for sink in _SINKS:
        sink._add(session_id, events, raw)
