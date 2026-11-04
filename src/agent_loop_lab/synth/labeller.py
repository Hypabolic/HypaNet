"""Window labeller: spec v0.2 §6.3 evaluated over latent steps (§6.1, §6.2).

Shared by every generator. Labels depend only on the latent trace and the
frozen oracle config, never on encoded features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..oracle_config import OracleConfig, load_oracle_config
from ..schema import EVENTS_PER_WINDOW, LatentEffect, OracleLabel
from .latent import LatentError, LatentStep


@dataclass(frozen=True)
class WindowStats:
    """§6.3 quantities for one window (for labelling and diagnostics)."""

    start: int  # list position of the window's first event
    end: int  # list position of the window's last event (endpoint)
    delta_p: float
    facts: int  # F
    stall_ineffective: int  # I = REDUNDANT + window reversions
    reversions: int  # V
    final_stall_ineffective: bool
    episode_in_policy_at_end: bool
    resolved_recently: bool  # an episode resolved within the last k_rec events
    dropped: bool
    none_effects: int  # count of NONE effects (diagnostics: aimless activity)


def validate_steps(steps: Sequence[LatentStep], cfg: OracleConfig) -> None:
    """Check internal consistency of a session's latent trace."""
    for i, s in enumerate(steps):
        if i > 0 and s.sequence_index != steps[i - 1].sequence_index + 1:
            raise LatentError(f"non-dense sequence_index at position {i}")
        if i > 0 and s.ws_before != steps[i - 1].ws_after:
            raise LatentError(f"ws_before of step {s.sequence_index} != ws_after of previous step")
        limit = cfg.r_max if s.open_episode_kind == "retry" else cfg.p_max
        if s.effect is LatentEffect.IN_POLICY_RETRY and not (s.open_episode_kind == "retry" and 1 <= s.open_episode_attempts <= cfg.r_max):
            raise LatentError(f"step {s.sequence_index}: IN_POLICY_RETRY outside an in-policy retry episode")
        if s.effect is LatentEffect.IN_POLICY_POLL and not (s.open_episode_kind == "poll" and 1 <= s.open_episode_attempts <= cfg.p_max):
            raise LatentError(f"step {s.sequence_index}: IN_POLICY_POLL outside an in-policy poll episode")
        if s.effect is LatentEffect.RESOLVES_EPISODE and not s.resolved_episode_here:
            raise LatentError(f"step {s.sequence_index}: RESOLVES_EPISODE without resolved_episode_here")
        if s.open_episode_kind is not None and s.open_episode_attempts > limit and s.effect in (
            LatentEffect.IN_POLICY_RETRY,
            LatentEffect.IN_POLICY_POLL,
        ):
            raise LatentError(f"step {s.sequence_index}: out-of-policy attempt marked in-policy")
        if s.effect is LatentEffect.PROGRESS and i > 0 and not s.progress_after > steps[i - 1].progress_after:
            raise LatentError(f"step {s.sequence_index}: PROGRESS without a progress increase")
        if s.effect is LatentEffect.NEW_FACT and s.new_facts < 1:
            raise LatentError(f"step {s.sequence_index}: NEW_FACT with new_facts == 0")


def window_stats(
    steps: Sequence[LatentStep], initial_progress: float, end: int, cfg: OracleConfig
) -> WindowStats:
    """Compute §6.3 quantities for the window whose endpoint is list position ``end``."""
    if end < EVENTS_PER_WINDOW - 1 or end >= len(steps):
        raise IndexError(f"no full window ends at position {end}")
    start = end - EVENTS_PER_WINDOW + 1
    window = steps[start : end + 1]
    p_before = steps[start - 1].progress_after if start > 0 else float(initial_progress)
    delta_p = window[-1].progress_after - p_before

    seen = {window[0].ws_before}
    rev_flags: list[bool] = []
    for s in window:
        is_rev = s.agent_mutation and s.undoes_prior_mutation and s.ws_after in seen
        rev_flags.append(is_rev)
        seen.add(s.ws_before)
        seen.add(s.ws_after)
    stall_flags = [s.effect is LatentEffect.REDUNDANT or r for s, r in zip(window, rev_flags)]

    last = window[-1]
    in_policy = last.open_episode_kind is not None and last.open_episode_attempts <= (
        cfg.r_max if last.open_episode_kind == "retry" else cfg.p_max
    )
    recent = window[-cfg.k_rec :] if cfg.k_rec > 0 else []
    return WindowStats(
        start=start,
        end=end,
        delta_p=delta_p,
        facts=sum(s.new_facts for s in window),
        stall_ineffective=sum(stall_flags),
        reversions=sum(rev_flags),
        final_stall_ineffective=stall_flags[-1],
        episode_in_policy_at_end=in_policy,
        resolved_recently=any(s.resolved_episode_here for s in recent),
        dropped=any(s.dropped for s in window),
        none_effects=sum(s.effect is LatentEffect.NONE for s in window),
    )


def label_from_stats(st: WindowStats, cfg: OracleConfig) -> OracleLabel:
    """§6.3 rules 1-6, first match wins."""
    if st.dropped:  # rule 1
        return OracleLabel.INSUFFICIENT_INSTRUMENTATION
    if st.reversions >= cfg.n_rev and st.delta_p <= cfg.eps_p:  # rule 2
        return OracleLabel.OSCILLATING
    if (  # rule 3
        st.delta_p <= cfg.eps_p
        and st.facts == 0
        and st.stall_ineffective >= cfg.n_stall
        and st.final_stall_ineffective
        and not st.episode_in_policy_at_end
    ):
        return OracleLabel.STALLED
    if st.episode_in_policy_at_end or st.resolved_recently:  # rule 4
        return OracleLabel.RECOVERY
    if st.delta_p > cfg.eps_p or st.facts >= cfg.f_prod:  # rule 5
        return OracleLabel.PRODUCTIVE
    return OracleLabel.INSUFFICIENT_AMBIGUOUS  # rule 6


def label_session(
    steps: Sequence[LatentStep], initial_progress: float, cfg: OracleConfig | None = None
) -> dict[int, OracleLabel]:
    """Labels for every full window, keyed by the endpoint's ``sequence_index``."""
    cfg = cfg or load_oracle_config()
    validate_steps(steps, cfg)
    return {
        steps[t].sequence_index: label_from_stats(window_stats(steps, initial_progress, t, cfg), cfg)
        for t in range(EVENTS_PER_WINDOW - 1, len(steps))
    }
