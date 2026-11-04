"""Per-event latent annotation contract (spec v0.2 §6.1).

Every generator (standard ``synth`` and the independently written ``synth_ood``)
emits one :class:`LatentStep` per canonical event. The window labeller
(:mod:`agent_loop_lab.synth.labeller`) derives oracle labels from these steps
alone, so labels mean the same thing whichever generator produced the session.

Nothing here is visible to the inference path: steps live in
``OracleRecord.latent_trace`` only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from ..schema import LatentEffect, OracleRecord

EPISODE_KINDS = ("retry", "poll")


class LatentError(ValueError):
    """Raised when a latent trace violates the contract."""


@dataclass(frozen=True)
class LatentStep:
    """Latent state for one event, as known to the simulator.

    Fields:
        sequence_index: the event's ``sequence_index``.
        effect: §6.1 effect of this operation. ``REVERSION`` is set when the
            operation is a reversion relative to the window ending at this
            event; the labeller recomputes reversions per window from the
            workspace fields below, so this value is informational.
        progress_after: latent objective completion ``p`` after the operation, in [0, 1].
        new_facts: number of facts added to ``K`` by this operation (>= 0).
        ws_before / ws_after: opaque ids of the latent workspace content state
            before / after the operation (equal when nothing changed). These are
            latent: they stay correct even when the observed digests are dropped.
        agent_mutation: the agent's own operation changed the workspace
            (external resets are ``False``).
        undoes_prior_mutation: this mutation undoes an earlier agent mutation.
        open_episode_kind: ``"retry"`` / ``"poll"`` for the failure episode that
            is open (unresolved) **after** this operation, else ``None``.
        open_episode_attempts: retries (kind ``retry``) or polls (kind ``poll``)
            made so far in that open episode, counted after this operation; the
            initial failing operation counts 0.
        resolved_episode_here: this operation resolved a failure episode.
        dropped: the generator deliberately dropped a required observation on
            this event (scenario 12, rule 1).
    """

    sequence_index: int
    effect: LatentEffect
    progress_after: float
    new_facts: int
    ws_before: str
    ws_after: str
    agent_mutation: bool = False
    undoes_prior_mutation: bool = False
    open_episode_kind: str | None = None
    open_episode_attempts: int = 0
    resolved_episode_here: bool = False
    dropped: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect", LatentEffect(self.effect))
        if not isinstance(self.sequence_index, int) or self.sequence_index < 0:
            raise LatentError("sequence_index must be int >= 0")
        if not (0.0 <= float(self.progress_after) <= 1.0):
            raise LatentError(f"progress_after out of [0,1]: {self.progress_after}")
        if not isinstance(self.new_facts, int) or self.new_facts < 0:
            raise LatentError("new_facts must be int >= 0")
        if not (isinstance(self.ws_before, str) and isinstance(self.ws_after, str) and self.ws_before and self.ws_after):
            raise LatentError("ws_before/ws_after must be non-empty str")
        if self.open_episode_kind is not None and self.open_episode_kind not in EPISODE_KINDS:
            raise LatentError(f"open_episode_kind must be one of {EPISODE_KINDS} or None")
        if self.open_episode_kind is None and self.open_episode_attempts != 0:
            raise LatentError("open_episode_attempts must be 0 when no episode is open")
        if self.open_episode_attempts < 0:
            raise LatentError("open_episode_attempts must be >= 0")
        if self.undoes_prior_mutation and not self.agent_mutation:
            raise LatentError("undoes_prior_mutation requires agent_mutation")
        if self.agent_mutation and self.ws_before == self.ws_after:
            raise LatentError("agent_mutation requires ws_before != ws_after")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["effect"] = self.effect.value
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LatentStep":
        return cls(**{**d, "effect": LatentEffect(d["effect"])})


def latent_trace(steps: Sequence[LatentStep], initial_progress: float) -> dict[str, Any]:
    """The ``OracleRecord.latent_trace`` payload for a session."""
    if not (0.0 <= float(initial_progress) <= 1.0):
        raise LatentError("initial_progress out of [0,1]")
    return {"initial_progress": float(initial_progress), "steps": [s.to_dict() for s in steps]}


def steps_from_record(record: OracleRecord) -> tuple[list[LatentStep], float]:
    """Recover (steps, initial_progress) from an OracleRecord."""
    trace = record.latent_trace
    try:
        steps = [LatentStep.from_dict(d) for d in trace["steps"]]
        p0 = float(trace["initial_progress"])
    except (KeyError, TypeError) as exc:
        raise LatentError(f"malformed latent_trace: {exc}") from exc
    return steps, p0
