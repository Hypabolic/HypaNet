"""Canonical event and oracle contracts (spec v0.2 §3, §6).

``CanonicalEvent`` is the *public inference object*: it must never carry labels,
scenario/family IDs, latent progress, retry-policy state or future outcomes.
``OracleRecord`` is the sealed ground-truth object produced by the simulator;
nothing on the inference path may import or consume it.

Unknown observations are represented as ``None`` and are never coerced to a
default value (spec §4 "Missing data").
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Mapping

SPEC_VERSION = "0.2"

EVENTS_PER_WINDOW = 8
FEATURES_PER_EVENT = 15
INPUT_DIM = EVENTS_PER_WINDOW * FEATURES_PER_EVENT  # 120


# Frozen per-event feature order (spec §4). Index within a 15-feature block.
FEATURE_NAMES: tuple[str, ...] = (
    "cat_INSPECT",
    "cat_MUTATE",
    "cat_VALIDATE",
    "cat_EXTERNAL",
    "cat_CONTROL",
    "out_SUCCESS",
    "out_ERROR",
    "out_NEUTRAL",
    "repeated_intent",
    "repeated_failure",
    "workspace_changed",
    "state_revisited",
    "validation_comparable",
    "validation_delta",
    "relative_token_cost",
)
FEATURE_INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}
assert len(FEATURE_NAMES) == FEATURES_PER_EVENT

FEATURE_DEFINITIONS: dict[str, str] = {
    "cat_INSPECT": "1 iff operation_category == INSPECT",
    "cat_MUTATE": "1 iff operation_category == MUTATE",
    "cat_VALIDATE": "1 iff operation_category == VALIDATE",
    "cat_EXTERNAL": "1 iff operation_category == EXTERNAL",
    "cat_CONTROL": "1 iff operation_category == CONTROL",
    "out_SUCCESS": "1 iff outcome == SUCCESS",
    "out_ERROR": "1 iff outcome == ERROR",
    "out_NEUTRAL": "1 iff outcome == NEUTRAL",
    "repeated_intent": "1 iff intent_fingerprint equals that of an earlier event in this window",
    "repeated_failure": "1 iff outcome == ERROR and failure_fingerprint equals that of an earlier ERROR event in this window",
    "workspace_changed": "1 iff workspace_state_before != workspace_state_after",
    "state_revisited": (
        "1 iff workspace_changed == 1 and workspace_state_after equals any before- or after-digest of an "
        "earlier event in this window (including the first event's before-state)"
    ),
    "validation_comparable": (
        "1 iff this event produced a validation result and an earlier result from the same "
        "validation_suite_id exists anywhere earlier in the session"
    ),
    "validation_delta": (
        "if validation_comparable: clip(passed/total - prev_passed/prev_total, -1, 1) against the most recent "
        "earlier result from that suite; else 0"
    ),
    "relative_token_cost": "min(1, log1p(input_tokens + output_tokens) / log1p(100000)); 0 for a verified zero-token step",
}


def flat_index(position: int, feature: str | int) -> int:
    """Index into the flattened 120-vector for window position 0..7 (oldest first)."""
    idx = FEATURE_INDEX[feature] if isinstance(feature, str) else int(feature)
    if not (0 <= position < EVENTS_PER_WINDOW and 0 <= idx < FEATURES_PER_EVENT):
        raise IndexError(f"position {position} / feature {feature} out of range")
    return position * FEATURES_PER_EVENT + idx


class OperationCategory(str, Enum):
    INSPECT = "INSPECT"
    MUTATE = "MUTATE"
    VALIDATE = "VALIDATE"
    EXTERNAL = "EXTERNAL"
    CONTROL = "CONTROL"


CATEGORY_ORDER: tuple[OperationCategory, ...] = (
    OperationCategory.INSPECT,
    OperationCategory.MUTATE,
    OperationCategory.VALIDATE,
    OperationCategory.EXTERNAL,
    OperationCategory.CONTROL,
)


class Outcome(str, Enum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    NEUTRAL = "NEUTRAL"


OUTCOME_ORDER: tuple[Outcome, ...] = (Outcome.SUCCESS, Outcome.ERROR, Outcome.NEUTRAL)


class ClassLabel(IntEnum):
    """Canonical network output order (spec §2.2). Index == logit position."""

    PRODUCTIVE = 0
    RECOVERY = 1
    STALLED = 2
    OSCILLATING = 3


CLASS_NAMES: tuple[str, ...] = tuple(c.name for c in ClassLabel)


class OracleLabel(str, Enum):
    """Oracle window label (spec §6.3). Not a network output."""

    PRODUCTIVE = "PRODUCTIVE"
    RECOVERY = "RECOVERY"
    STALLED = "STALLED"
    OSCILLATING = "OSCILLATING"
    INSUFFICIENT_INSTRUMENTATION = "INSUFFICIENT_INSTRUMENTATION"  # rule 1
    INSUFFICIENT_AMBIGUOUS = "INSUFFICIENT_AMBIGUOUS"  # rule 6

    @property
    def is_scored(self) -> bool:
        return self.name in ClassLabel.__members__

    def to_class(self) -> ClassLabel:
        if not self.is_scored:
            raise ValueError(f"{self.value} has no network class")
        return ClassLabel[self.name]


class LatentEffect(str, Enum):
    """Per-operation latent effect (spec §6.1). Oracle-only."""

    PROGRESS = "PROGRESS"
    NEW_FACT = "NEW_FACT"
    RESOLVES_EPISODE = "RESOLVES_EPISODE"
    IN_POLICY_RETRY = "IN_POLICY_RETRY"
    IN_POLICY_POLL = "IN_POLICY_POLL"
    REVERSION = "REVERSION"
    REDUNDANT = "REDUNDANT"
    NONE = "NONE"


class SchemaError(ValueError):
    """Raised when an event or record violates the canonical contract."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


def _opt_nonneg_int(name: str, value: Any) -> None:
    if value is None:
        return
    _require(isinstance(value, int) and not isinstance(value, bool), f"{name} must be int or None")
    _require(value >= 0, f"{name} must be >= 0")


def _iso_timestamp(name: str, value: str) -> None:
    """ISO-8601 as accepted by ``datetime.fromisoformat`` (Python 3.12), plus a trailing ``Z`` for UTC.

    Only validated, never normalised: the string is kept as given.
    """
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(text)
    except ValueError as exc:
        raise SchemaError(f"{name} must be an ISO-8601 timestamp, got {value!r}") from exc


_HEX64 = re.compile(r"[0-9a-f]{64}")


def _opt_hex(name: str, value: Any) -> None:
    """Canonical digest: exactly 64 lowercase hex chars (``hashlib.sha256().hexdigest()``).

    Digests are compared by plain string equality downstream, so non-canonical
    spellings (uppercase, ``0x`` prefix, ``_`` separators, whitespace) are rejected.
    """
    if value is None:
        return
    _require(
        isinstance(value, str) and _HEX64.fullmatch(value) is not None,
        f"{name} must be a 64-char lowercase sha256 hex digest or None",
    )


@dataclass(frozen=True)
class Provenance:
    """Where an event came from and how well it was instrumented.

    ``history_complete`` is True iff the event stream for this session is known
    to start at the session's first operation (no truncated prefix). The gate
    uses it to decide whether a missing validation baseline is *unobserved*
    rather than genuinely absent (spec §5.2).
    """

    source: str  # e.g. "simulator:0.2.0" or "adapter:otel-genai:1"
    history_complete: bool = True
    instrumentation_flags: tuple[str, ...] = ()
    trace_id: str | None = None
    span_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require(isinstance(self.source, str) and self.source != "", "provenance.source required")
        _require(isinstance(self.history_complete, bool), "provenance.history_complete must be bool")
        object.__setattr__(self, "instrumentation_flags", tuple(self.instrumentation_flags))
        object.__setattr__(self, "span_ids", tuple(self.span_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "history_complete": self.history_complete,
            "instrumentation_flags": list(self.instrumentation_flags),
            "trace_id": self.trace_id,
            "span_ids": list(self.span_ids),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Provenance":
        allowed = {f.name for f in fields(cls)}
        extra = set(d) - allowed
        _require(not extra, f"unknown provenance fields: {sorted(extra)}")
        return cls(
            source=d["source"],
            history_complete=d.get("history_complete", True),
            instrumentation_flags=tuple(d.get("instrumentation_flags", ())),
            trace_id=d.get("trace_id"),
            span_ids=tuple(d.get("span_ids", ())),
        )


@dataclass(frozen=True)
class CanonicalEvent:
    """One completed logical agent operation (spec §3.1).

    Validation fields: ``validation_produced`` is True iff this operation
    produced a comparable validation result. When True, suite id / passed /
    total are expected (``None`` means unobserved, which the gate rejects).
    When False they must all be ``None``. A validation result implies category
    VALIDATE.
    """

    session_id: str
    agent_id: str
    task_id: str
    event_id: str
    sequence_index: int
    timestamp: str
    operation_family: str
    operation_category: OperationCategory
    outcome: Outcome
    intent_fingerprint: str
    provenance: Provenance
    failure_fingerprint: str | None = None
    workspace_state_before: str | None = None
    workspace_state_after: str | None = None
    validation_produced: bool = False
    validation_suite_id: str | None = None
    validation_passed: int | None = None
    validation_total: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "agent_id", "task_id", "event_id", "timestamp", "operation_family"):
            v = getattr(self, name)
            _require(isinstance(v, str) and v != "", f"{name} must be a non-empty str")
        _iso_timestamp("timestamp", self.timestamp)
        _require(
            isinstance(self.sequence_index, int) and not isinstance(self.sequence_index, bool) and self.sequence_index >= 0,
            "sequence_index must be int >= 0",
        )
        # Coerce enums given as strings, reject anything else.
        try:
            object.__setattr__(self, "operation_category", OperationCategory(self.operation_category))
            object.__setattr__(self, "outcome", Outcome(self.outcome))
        except ValueError as exc:
            raise SchemaError(str(exc)) from exc
        _require(isinstance(self.provenance, Provenance), "provenance must be a Provenance")
        _opt_hex("intent_fingerprint", self.intent_fingerprint)
        _require(self.intent_fingerprint is not None, "intent_fingerprint required")
        _opt_hex("failure_fingerprint", self.failure_fingerprint)
        _require(
            self.failure_fingerprint is None or self.outcome is Outcome.ERROR,
            "failure_fingerprint only allowed when outcome == ERROR",
        )
        _opt_hex("workspace_state_before", self.workspace_state_before)
        _opt_hex("workspace_state_after", self.workspace_state_after)
        _opt_nonneg_int("validation_passed", self.validation_passed)
        _opt_nonneg_int("validation_total", self.validation_total)
        _opt_nonneg_int("input_tokens", self.input_tokens)
        _opt_nonneg_int("output_tokens", self.output_tokens)
        _require(isinstance(self.validation_produced, bool), "validation_produced must be bool")
        if self.validation_produced:
            _require(
                self.operation_category is OperationCategory.VALIDATE,
                "validation_produced requires operation_category == VALIDATE",
            )
            if self.validation_total is not None:
                _require(self.validation_total >= 1, "validation_total must be >= 1")
            if self.validation_passed is not None and self.validation_total is not None:
                _require(self.validation_passed <= self.validation_total, "validation_passed > validation_total")
        else:
            _require(
                self.validation_suite_id is None and self.validation_passed is None and self.validation_total is None,
                "validation fields must be None when validation_produced is False",
            )

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Enum):
                v = v.value
            elif isinstance(v, Provenance):
                v = v.to_dict()
            out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CanonicalEvent":
        allowed = {f.name for f in fields(cls)}
        extra = set(d) - allowed
        _require(not extra, f"unknown CanonicalEvent fields (possible oracle leakage): {sorted(extra)}")
        kwargs = dict(d)
        kwargs["provenance"] = Provenance.from_dict(d["provenance"])
        return cls(**kwargs)

    def to_json(self) -> str:
        """Canonical JSON: sorted keys, no insignificant whitespace, UTF-8 safe."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "CanonicalEvent":
        return cls.from_dict(json.loads(s))

    @property
    def step_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class OracleRecord:
    """Sealed ground truth for one session (spec §6). Never on the inference path.

    ``event_effects[i]`` is the latent effect of the event with sequence_index i.
    ``window_labels`` maps window endpoint sequence_index -> OracleLabel.
    ``dropped_observations`` lists sequence indices where the generator
    deliberately dropped a required observation (scenario 12, rule 1).
    """

    session_id: str
    generator_version: str
    seed: int
    scenario_family: str
    event_effects: tuple[LatentEffect, ...]
    window_labels: Mapping[int, OracleLabel] = field(default_factory=dict)
    dropped_observations: tuple[int, ...] = ()
    latent_trace: Mapping[str, Any] = field(default_factory=dict)  # free-form diagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "scenario_family": self.scenario_family,
            "event_effects": [e.value for e in self.event_effects],
            "window_labels": {str(k): v.value for k, v in sorted(self.window_labels.items())},
            "dropped_observations": list(self.dropped_observations),
            "latent_trace": dict(self.latent_trace),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OracleRecord":
        return cls(
            session_id=d["session_id"],
            generator_version=d["generator_version"],
            seed=int(d["seed"]),
            scenario_family=d["scenario_family"],
            event_effects=tuple(LatentEffect(e) for e in d["event_effects"]),
            window_labels={int(k): OracleLabel(v) for k, v in d.get("window_labels", {}).items()},
            dropped_observations=tuple(d.get("dropped_observations", ())),
            latent_trace=dict(d.get("latent_trace", {})),
        )


# Field names that must never appear on the public inference object.
FORBIDDEN_INFERENCE_FIELDS: frozenset[str] = frozenset(
    {
        "label",
        "labels",
        "window_labels",
        "scenario",
        "scenario_id",
        "scenario_family",
        "family",
        "latent_effect",
        "event_effects",
        "progress",
        "latent_progress",
        "retry_policy",
        "oracle",
        "future_outcome",
    }
)
