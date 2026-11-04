"""Adversarial tests for the shared window labeller and latent-step contract.

Targets ``agent_loop_lab.synth.labeller`` and ``agent_loop_lab.synth.latent``
against spec v0.2 §6.1-§6.3 and §0.2 (r2), with the M1 contract
(``docs/M1_CONTRACTS.md``) where the spec is silent.

Two layers:

* ``label_from_stats`` on hand-built ``WindowStats``: the §6.3 rule order and
  every precedence pair, in isolation.
* ``window_stats`` / ``label_session`` on hand-constructed ``LatentStep``
  sequences, built by a small chained builder that keeps ``ws_before`` /
  ``sequence_index`` consistent so ``validate_steps`` accepts every trace that
  is not deliberately malformed. Tests assert on the computed ``WindowStats``
  fields as well as the label, so a failure says which quantity went wrong.

All sequences are hand-specified; nothing here generates sessions or looks at
encoded features.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from agent_loop_lab.oracle_config import FROZEN_V0_2, OracleConfig, load_oracle_config
from agent_loop_lab.schema import EVENTS_PER_WINDOW, LatentEffect, OracleLabel, OracleRecord
from agent_loop_lab.synth.labeller import (
    WindowStats,
    label_from_stats,
    label_session,
    validate_steps,
    window_stats,
)
from agent_loop_lab.synth.latent import (
    EPISODE_KINDS,
    LatentError,
    LatentStep,
    latent_trace,
    steps_from_record,
)

E = LatentEffect
L = OracleLabel
CFG = FROZEN_V0_2
EPS = CFG.eps_p
JUST_ABOVE_EPS = math.nextafter(EPS, 1.0)


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


class Session:
    """Chained builder for a consistent latent trace.

    ``ws_before`` of each step is the previous ``ws_after``; ``sequence_index``
    is dense from ``start_index``; ``progress_after`` carries over unless set.
    ``agent_mutation`` defaults to "the workspace changed".
    """

    def __init__(self, *, start_index: int = 0, ws: str = "S0", initial_progress: float = 0.0):
        self.steps: list[LatentStep] = []
        self.initial_progress = initial_progress
        self._next = start_index
        self._ws = ws
        self._p = initial_progress

    # core ------------------------------------------------------------------
    def op(
        self,
        effect: LatentEffect = E.NONE,
        *,
        to: str | None = None,
        progress: float | None = None,
        facts: int = 0,
        undo: bool = False,
        mutation: bool | None = None,
        episode: tuple[str, int] | None = None,
        resolved: bool = False,
        dropped: bool = False,
    ) -> "Session":
        ws_before = self._ws
        ws_after = ws_before if to is None else to
        if mutation is None:
            mutation = ws_after != ws_before
        p = self._p if progress is None else progress
        kind, attempts = episode if episode is not None else (None, 0)
        self.steps.append(
            LatentStep(
                sequence_index=self._next,
                effect=effect,
                progress_after=p,
                new_facts=facts,
                ws_before=ws_before,
                ws_after=ws_after,
                agent_mutation=mutation,
                undoes_prior_mutation=undo,
                open_episode_kind=kind,
                open_episode_attempts=attempts,
                resolved_episode_here=resolved,
                dropped=dropped,
            )
        )
        self._next += 1
        self._ws = ws_after
        self._p = p
        return self

    # shortcuts -------------------------------------------------------------
    def none(self, n: int = 1, **kw) -> "Session":
        """Novel but fruitless operations (e.g. first read of an irrelevant file)."""
        for _ in range(n):
            self.op(E.NONE, **kw)
        return self

    def redundant(self, n: int = 1, **kw) -> "Session":
        """Operations that repeat work already reflected in K / W."""
        for _ in range(n):
            self.op(E.REDUNDANT, **kw)
        return self

    def fact(self, k: int = 1, **kw) -> "Session":
        return self.op(E.NEW_FACT, facts=k, **kw)

    def progress_to(self, p: float, **kw) -> "Session":
        return self.op(E.PROGRESS, progress=p, **kw)

    def edit(self, to: str, effect: LatentEffect = E.NONE, *, undo: bool = False, **kw) -> "Session":
        return self.op(effect, to=to, undo=undo, mutation=True, **kw)

    def fail(self, **kw) -> "Session":
        """Initial failing operation of a would-be episode: NONE, no episode open (M1 §1)."""
        return self.op(E.NONE, **kw)

    def retry(self, attempts: int, **kw) -> "Session":
        eff = E.IN_POLICY_RETRY if 1 <= attempts <= CFG.r_max else E.REDUNDANT
        return self.op(eff, episode=("retry", attempts), **kw)

    def poll(self, attempts: int, **kw) -> "Session":
        eff = E.IN_POLICY_POLL if 1 <= attempts <= CFG.p_max else E.REDUNDANT
        return self.op(eff, episode=("poll", attempts), **kw)

    def resolve(self, **kw) -> "Session":
        return self.op(E.RESOLVES_EPISODE, resolved=True, **kw)

    # evaluation ------------------------------------------------------------
    def labels(self, cfg: OracleConfig = CFG) -> dict[int, OracleLabel]:
        return label_session(self.steps, self.initial_progress, cfg)

    def label_at(self, end: int, cfg: OracleConfig = CFG) -> OracleLabel:
        """Label of the window ending at list position ``end`` (default: last)."""
        return label_from_stats(self.stats(end, cfg), cfg)

    def stats(self, end: int | None = None, cfg: OracleConfig = CFG) -> WindowStats:
        validate_steps(self.steps, cfg)
        end = len(self.steps) - 1 if end is None else end
        return window_stats(self.steps, self.initial_progress, end, cfg)

    def last_label(self, cfg: OracleConfig = CFG) -> OracleLabel:
        return self.labels(cfg)[self.steps[-1].sequence_index]


def stats(**kw) -> WindowStats:
    """A neutral WindowStats (rule 6 by default) with overrides."""
    base = dict(
        start=0,
        end=7,
        delta_p=0.0,
        facts=0,
        stall_ineffective=0,
        reversions=0,
        final_stall_ineffective=False,
        episode_in_policy_at_end=False,
        resolved_recently=False,
        dropped=False,
        none_effects=0,
    )
    base.update(kw)
    return WindowStats(**base)


def step(**kw) -> LatentStep:
    base = dict(sequence_index=0, effect=E.NONE, progress_after=0.0, new_facts=0, ws_before="S0", ws_after="S0")
    base.update(kw)
    return LatentStep(**base)


# Condition bundles for the stats-level precedence matrix.
DROPPED = dict(dropped=True)
OSC = dict(reversions=2)  # with delta_p = 0.0
STALL = dict(stall_ineffective=5, final_stall_ineffective=True)  # with delta_p = 0.0, facts = 0
REC_ACTIVE = dict(episode_in_policy_at_end=True)
REC_RESOLVED = dict(resolved_recently=True)
PROD_DP = dict(delta_p=0.5)
PROD_F = dict(facts=2)


# --------------------------------------------------------------------------
# Config sanity: the tests below assume the frozen §6.2 values.
# --------------------------------------------------------------------------


def test_frozen_parameters_match_spec_6_2():
    assert CFG == OracleConfig(eps_p=0.02, f_prod=2, n_stall=5, n_grey=3, n_rev=2, r_max=3, p_max=5, k_rec=4)
    assert load_oracle_config() == CFG
    assert EVENTS_PER_WINDOW == 8
    assert EPISODE_KINDS == ("retry", "poll")


# --------------------------------------------------------------------------
# label_from_stats: each rule alone
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, expected",
    [
        (DROPPED, L.INSUFFICIENT_INSTRUMENTATION),  # rule 1
        (OSC, L.OSCILLATING),  # rule 2
        (STALL, L.STALLED),  # rule 3
        (REC_ACTIVE, L.RECOVERY),  # rule 4, active in policy
        (REC_RESOLVED, L.RECOVERY),  # rule 4, resolved recently
        (PROD_DP, L.PRODUCTIVE),  # rule 5, progress
        (PROD_F, L.PRODUCTIVE),  # rule 5, investigative
        ({}, L.INSUFFICIENT_AMBIGUOUS),  # rule 6
    ],
    ids=["r1", "r2", "r3", "r4-active", "r4-resolved", "r5-dp", "r5-facts", "r6"],
)
def test_each_rule_alone(kw, expected):
    assert label_from_stats(stats(**kw), CFG) is expected


# --------------------------------------------------------------------------
# label_from_stats: precedence pairs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "other",
    [OSC, STALL, REC_ACTIVE, REC_RESOLVED, PROD_DP, PROD_F, {**OSC, **STALL, **REC_ACTIVE, **REC_RESOLVED, **PROD_F}],
    ids=["osc", "stalled", "rec-active", "rec-resolved", "prod-dp", "prod-f", "everything"],
)
def test_rule1_beats_every_other_rule(other):
    assert label_from_stats(stats(**DROPPED, **other), CFG) is L.INSUFFICIENT_INSTRUMENTATION


@pytest.mark.parametrize(
    "pair, expected",
    [
        ({**OSC, **STALL}, L.OSCILLATING),
        ({**OSC, **REC_ACTIVE}, L.OSCILLATING),
        ({**OSC, **REC_RESOLVED}, L.OSCILLATING),
        ({**OSC, **PROD_F}, L.OSCILLATING),  # F >= F_prod does not rescue an oscillation
        ({**STALL, **REC_RESOLVED}, L.STALLED),  # a recent resolution does not outrank a stall
        ({**REC_ACTIVE, **PROD_DP}, L.RECOVERY),
        ({**REC_ACTIVE, **PROD_F}, L.RECOVERY),
        ({**REC_RESOLVED, **PROD_DP}, L.RECOVERY),  # ratified k_rec trade-off (§0 row 8)
        ({**REC_RESOLVED, **PROD_F}, L.RECOVERY),
    ],
    ids=[
        "osc>stalled",
        "osc>rec-active",
        "osc>rec-resolved",
        "osc>prod-f",
        "stalled>rec-resolved",
        "rec-active>prod-dp",
        "rec-active>prod-f",
        "rec-resolved>prod-dp",
        "rec-resolved>prod-f",
    ],
)
def test_precedence_pairs(pair, expected):
    assert label_from_stats(stats(**pair), CFG) is expected


def test_osc_requires_no_net_progress():
    """Rule 2 needs Δp ≤ ε; with Δp > ε the window falls through to PRODUCTIVE."""
    assert label_from_stats(stats(**OSC, **PROD_DP), CFG) is L.PRODUCTIVE
    assert label_from_stats(stats(reversions=8, delta_p=JUST_ABOVE_EPS), CFG) is L.PRODUCTIVE


def test_in_policy_episode_blocks_stalled_rather_than_losing_to_it():
    """'No episode in policy at the endpoint' is a STALLED precondition, so rule 4 catches it."""
    assert label_from_stats(stats(**STALL, **REC_ACTIVE), CFG) is L.RECOVERY
    assert label_from_stats(stats(**STALL, **REC_ACTIVE, **REC_RESOLVED), CFG) is L.RECOVERY


def test_stalled_and_productive_are_mutually_exclusive():
    """STALLED needs F = 0 and Δp ≤ ε; PRODUCTIVE needs F ≥ 2 or Δp > ε. No window satisfies both:
    adding either PRODUCTIVE condition to a stall makes it PRODUCTIVE, not STALLED."""
    assert label_from_stats(stats(**STALL, **PROD_DP), CFG) is L.PRODUCTIVE
    assert label_from_stats(stats(**STALL, **PROD_F), CFG) is L.PRODUCTIVE


# --------------------------------------------------------------------------
# label_from_stats: numeric boundaries
# --------------------------------------------------------------------------


def test_eps_boundary_on_stats():
    assert label_from_stats(stats(delta_p=EPS), CFG) is L.INSUFFICIENT_AMBIGUOUS  # 0.02 is not > 0.02
    assert label_from_stats(stats(delta_p=JUST_ABOVE_EPS), CFG) is L.PRODUCTIVE
    assert label_from_stats(stats(delta_p=EPS, **OSC), CFG) is L.OSCILLATING  # Δp ≤ ε holds at equality
    assert label_from_stats(stats(delta_p=EPS, **STALL), CFG) is L.STALLED
    assert label_from_stats(stats(delta_p=JUST_ABOVE_EPS, **STALL), CFG) is L.PRODUCTIVE
    assert label_from_stats(stats(delta_p=-1.0, **STALL), CFG) is L.STALLED  # regressions are ≤ ε


@pytest.mark.parametrize("facts, expected", [(0, L.INSUFFICIENT_AMBIGUOUS), (1, L.INSUFFICIENT_AMBIGUOUS), (2, L.PRODUCTIVE), (9, L.PRODUCTIVE)])
def test_f_prod_boundary_on_stats(facts, expected):
    assert label_from_stats(stats(facts=facts), CFG) is expected


def test_single_fact_blocks_stalled():
    assert label_from_stats(stats(**STALL, facts=1), CFG) is L.INSUFFICIENT_AMBIGUOUS


@pytest.mark.parametrize("i, expected", [(0, L.INSUFFICIENT_AMBIGUOUS), (3, L.INSUFFICIENT_AMBIGUOUS), (4, L.INSUFFICIENT_AMBIGUOUS), (5, L.STALLED), (8, L.STALLED)])
def test_n_stall_boundary_on_stats(i, expected):
    assert label_from_stats(stats(stall_ineffective=i, final_stall_ineffective=True), CFG) is expected


def test_final_event_must_be_stall_ineffective_on_stats():
    assert label_from_stats(stats(stall_ineffective=8, final_stall_ineffective=False), CFG) is L.INSUFFICIENT_AMBIGUOUS


@pytest.mark.parametrize("v, expected", [(0, L.INSUFFICIENT_AMBIGUOUS), (1, L.INSUFFICIENT_AMBIGUOUS), (2, L.OSCILLATING), (4, L.OSCILLATING)])
def test_n_rev_boundary_on_stats(v, expected):
    assert label_from_stats(stats(reversions=v), CFG) is expected


def test_label_from_stats_uses_passed_config():
    loose = dataclasses.replace(CFG, eps_p=0.5, f_prod=3, n_stall=2, n_grey=1, n_rev=1)
    assert label_from_stats(stats(delta_p=0.4), loose) is L.INSUFFICIENT_AMBIGUOUS
    assert label_from_stats(stats(facts=2), loose) is L.INSUFFICIENT_AMBIGUOUS
    assert label_from_stats(stats(stall_ineffective=2, final_stall_ineffective=True), loose) is L.STALLED
    assert label_from_stats(stats(reversions=1), loose) is L.OSCILLATING


# --------------------------------------------------------------------------
# Step level: rules 1-6 on hand-constructed traces
# --------------------------------------------------------------------------


def _osc_window(test_effect=E.NONE, *, test_facts: int = 0, initial_progress: float = 0.0, first_edit=None):
    """A / revert / A / revert with a test after each edit. V = 3 (the first edit is novel)."""
    s = Session(initial_progress=initial_progress)
    if first_edit is None:
        s.edit("A")
    else:
        s.edit("A", first_edit[0], progress=first_edit[1])
    s.op(test_effect, facts=test_facts)
    s.edit("S0", E.REVERSION, undo=True)
    s.op(test_effect, facts=test_facts)
    s.edit("A", E.REVERSION, undo=True)
    s.op(test_effect, facts=test_facts)
    s.edit("S0", E.REVERSION, undo=True)
    s.op(test_effect, facts=test_facts)
    return s


def _stalled_window():
    return Session(initial_progress=0.5).redundant(8)


def _recovery_window():
    return Session().none(6).fail().retry(1)


def _productive_window():
    return Session().none().fact().progress_to(0.5, to="S1", mutation=True).none(5)


def _ambiguous_window():
    return Session().none(8)


SCENARIOS = {
    "osc": (_osc_window, L.OSCILLATING),
    "stalled": (_stalled_window, L.STALLED),
    "recovery": (_recovery_window, L.RECOVERY),
    "productive": (_productive_window, L.PRODUCTIVE),
    "ambiguous": (_ambiguous_window, L.INSUFFICIENT_AMBIGUOUS),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_step_level_scenarios_get_their_rule(name):
    build, expected = SCENARIOS[name]
    s = build()
    assert len(s.steps) == 8
    assert s.labels() == {7: expected}


@pytest.mark.parametrize("pos", [0, 3, 7])
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_rule1_dropped_anywhere_in_window_beats_everything(name, pos):
    build, _ = SCENARIOS[name]
    s = build()
    s.steps[pos] = dataclasses.replace(s.steps[pos], dropped=True)
    assert s.stats().dropped is True
    assert s.labels() == {7: L.INSUFFICIENT_INSTRUMENTATION}


def test_rule1_only_for_windows_containing_the_dropped_event():
    s = Session(initial_progress=0.5).redundant(3).op(E.REDUNDANT, dropped=True).redundant(12)
    labels = s.labels()
    # Dropped event at index 3 is inside windows ending 7..10.
    assert [labels[t] for t in range(7, 11)] == [L.INSUFFICIENT_INSTRUMENTATION] * 4
    assert [labels[t] for t in range(11, 16)] == [L.STALLED] * 5


# --------------------------------------------------------------------------
# Step level: precedence pairs
# --------------------------------------------------------------------------


def test_step_osc_beats_stalled():
    s = _osc_window(E.REDUNDANT)  # tests re-run on states already validated
    st = s.stats()
    assert (st.reversions, st.stall_ineffective, st.final_stall_ineffective, st.facts) == (3, 7, True, 0)
    assert s.labels() == {7: L.OSCILLATING}


def test_step_osc_beats_recovery_active_episode():
    s = Session()
    s.edit("A").none().edit("S0", E.REVERSION, undo=True).none()
    s.edit("A", E.REVERSION, undo=True).none().edit("S0", E.REVERSION, undo=True)
    s.retry(1)  # the test that failed after the edit before is retried, in policy
    st = s.stats()
    assert st.episode_in_policy_at_end and st.reversions == 3
    assert s.labels() == {7: L.OSCILLATING}


def test_step_osc_beats_recovery_recent_resolution():
    s = Session()
    s.edit("A").none().edit("S0", E.REVERSION, undo=True).fail()
    s.retry(1).resolve().edit("A", E.REVERSION, undo=True).edit("S0", E.REVERSION, undo=True)
    st = s.stats()
    assert st.resolved_recently and st.reversions == 3
    assert s.labels() == {7: L.OSCILLATING}


def test_step_osc_beats_productive_by_facts():
    s = _osc_window(E.NEW_FACT, test_facts=1)
    st = s.stats()
    assert st.facts == 4 and st.reversions == 3 and st.delta_p == 0.0
    assert s.labels() == {7: L.OSCILLATING}


def test_step_oscillation_with_net_progress_is_productive():
    s = _osc_window(initial_progress=0.0, first_edit=(E.PROGRESS, 0.5))
    st = s.stats()
    assert st.reversions == 3 and st.delta_p == 0.5
    assert s.labels() == {7: L.PRODUCTIVE}


def test_step_oscillation_at_exactly_eps_is_still_oscillating():
    s = _osc_window(initial_progress=0.0, first_edit=(E.PROGRESS, 0.02))
    st = s.stats()
    assert st.delta_p == EPS
    assert s.labels() == {7: L.OSCILLATING}


def test_step_stalled_beats_recent_resolution():
    """Doomed retries, a success, then redundant re-reads: resolution at endpoint-3 does not beat STALLED."""
    s = Session(initial_progress=0.5).fail().retry(1).retry(2).retry(3)
    s.retry(4).retry(5).retry(6).retry(7).resolve().redundant(3)
    st = s.stats()
    assert st.resolved_recently and not st.episode_in_policy_at_end
    assert (st.stall_ineffective, st.final_stall_ineffective) == (7, True)
    assert s.last_label() is L.STALLED


def test_step_recovery_beats_productive_active_episode():
    s = Session().progress_to(0.25, to="S1").progress_to(0.5, to="S2").none(4).fail().retry(2)
    assert s.stats().delta_p == 0.5
    assert s.labels() == {7: L.RECOVERY}


def test_step_productive_beats_ambiguous():
    s = Session().fact().fact().none(6)
    assert s.stats().facts == 2
    assert s.labels() == {7: L.PRODUCTIVE}


# --------------------------------------------------------------------------
# eps_p and Δp semantics
# --------------------------------------------------------------------------


def test_delta_p_exactly_eps_is_not_productive():
    s = Session(initial_progress=0.0).progress_to(0.02, to="S1").none(7)
    st = s.stats()
    assert st.delta_p == 0.02 == EPS
    assert s.labels() == {7: L.INSUFFICIENT_AMBIGUOUS}


def test_delta_p_just_above_eps_is_productive():
    s = Session(initial_progress=0.0).progress_to(JUST_ABOVE_EPS, to="S1").none(7)
    assert s.stats().delta_p > EPS
    assert s.labels() == {7: L.PRODUCTIVE}


def test_delta_p_exactly_eps_still_allows_stalled():
    s = Session(initial_progress=0.0).progress_to(0.02, to="S1").redundant(7)
    st = s.stats()
    assert st.delta_p == EPS and st.stall_ineffective == 7
    assert s.labels() == {7: L.STALLED}


def test_delta_p_just_above_eps_blocks_stalled():
    s = Session(initial_progress=0.0).progress_to(JUST_ABOVE_EPS, to="S1").redundant(7)
    assert s.labels() == {7: L.PRODUCTIVE}


def test_first_window_uses_initial_progress_not_first_step():
    def build(p0):
        return Session(initial_progress=p0).progress_to(0.5, to="S1").none(7)

    assert build(0.0).stats().delta_p == 0.5
    assert build(0.0).labels() == {7: L.PRODUCTIVE}
    low = build(0.49)
    assert low.stats().delta_p == pytest.approx(0.01)
    assert low.labels() == {7: L.INSUFFICIENT_AMBIGUOUS}


def test_later_windows_use_progress_before_the_window():
    """Progress credited at index 0 is in window 7 but before window 8."""
    s = Session(initial_progress=0.0).progress_to(0.5, to="S1").none(8)
    assert s.stats(7).delta_p == 0.5
    assert s.stats(8).delta_p == 0.0
    assert s.labels() == {7: L.PRODUCTIVE, 8: L.INSUFFICIENT_AMBIGUOUS}


def test_progress_rise_inside_window_counts_from_first_event():
    """A rise on the window's first event counts (p_before is the progress before that event)."""
    s = Session(initial_progress=0.0).none().progress_to(0.5, to="S1").none(7)
    assert s.stats(8).delta_p == 0.5  # window 1..8, rise at index 1
    assert s.labels()[8] is L.PRODUCTIVE


def test_delta_p_is_net_change_not_peak():
    s = Session(initial_progress=0.5).progress_to(0.75, to="S1").edit("S2", progress=0.5).none(6)
    assert s.stats().delta_p == 0.0
    assert s.labels() == {7: L.INSUFFICIENT_AMBIGUOUS}


def test_negative_delta_p_can_stall():
    s = Session(initial_progress=0.5).edit("S1", progress=0.25).redundant(7)
    st = s.stats()
    assert st.delta_p == -0.25
    assert s.labels() == {7: L.STALLED}


def test_characterise_eps_comparison_is_binary_float():
    """Characterisation, not a spec requirement: Δp is a float subtraction, so 0.5 -> 0.52 gives
    0.020000000000000018 > 0.02 and the window is PRODUCTIVE. Only progress values whose
    difference is exact (e.g. 0.0 -> 0.02) sit exactly on the boundary."""
    s = Session(initial_progress=0.5).progress_to(0.52, to="S1").none(7)
    assert s.stats().delta_p > EPS
    assert s.labels() == {7: L.PRODUCTIVE}


# --------------------------------------------------------------------------
# F_prod
# --------------------------------------------------------------------------


def test_one_new_fact_is_ambiguous():
    s = Session().fact().none(7)
    assert s.stats().facts == 1
    assert s.labels() == {7: L.INSUFFICIENT_AMBIGUOUS}


def test_two_facts_from_one_step_is_productive():
    s = Session().fact(2).none(7)
    assert s.stats().facts == 2
    assert s.labels() == {7: L.PRODUCTIVE}


def test_new_facts_on_a_progress_step_count_toward_f():
    """new_facts is recorded whatever the effect (M1 §1); F sums it."""
    s = Session(initial_progress=0.5).edit("S1", progress=0.25).progress_to(0.5, to="S2", facts=2).none(6)
    st = s.stats()
    assert st.delta_p == 0.0 and st.facts == 2
    assert s.labels() == {7: L.PRODUCTIVE}


def test_one_new_fact_blocks_stalled():
    s = Session(initial_progress=0.5).fact().redundant(7)
    st = s.stats()
    assert (st.facts, st.stall_ineffective, st.final_stall_ineffective) == (1, 7, True)
    assert s.labels() == {7: L.INSUFFICIENT_AMBIGUOUS}


def test_facts_outside_the_window_do_not_count():
    s = Session().fact(2).none(8)
    assert s.stats(7).facts == 2 and s.stats(8).facts == 0
    assert s.labels() == {7: L.PRODUCTIVE, 8: L.INSUFFICIENT_AMBIGUOUS}


# --------------------------------------------------------------------------
# N_stall, grey zone, final-event requirement, NONE effects
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_none, n_red, expected",
    [
        (5, 3, L.INSUFFICIENT_AMBIGUOUS),  # grey zone lower bound N_grey = 3
        (4, 4, L.INSUFFICIENT_AMBIGUOUS),  # grey zone upper end, I = N_stall - 1
        (3, 5, L.STALLED),  # I = N_stall
        (0, 8, L.STALLED),
    ],
)
def test_n_stall_boundary(n_none, n_red, expected):
    s = Session(initial_progress=0.5).none(n_none).redundant(n_red)
    st = s.stats()
    assert st.stall_ineffective == n_red and st.none_effects == n_none and st.final_stall_ineffective
    assert s.labels() == {7: expected}


def test_n_stall_interleaved_order():
    s = Session(initial_progress=0.5)
    for _ in range(3):
        s.redundant().none()
    s.redundant(2)  # R N R N R N R R: I = 5, final R
    assert s.stats().stall_ineffective == 5
    assert s.labels() == {7: L.STALLED}


def test_final_event_must_be_stall_ineffective():
    s = Session(initial_progress=0.5).redundant(7).none()
    st = s.stats()
    assert st.stall_ineffective == 7 and not st.final_stall_ineffective
    assert s.labels() == {7: L.INSUFFICIENT_AMBIGUOUS}


@pytest.mark.parametrize("final", ["retry", "poll", "fact", "resolve"])
def test_final_effective_or_episode_event_blocks_stalled(final):
    s = Session(initial_progress=0.5).redundant(7)
    if final == "retry":
        s.retry(1)
    elif final == "poll":
        s.poll(1)
    elif final == "fact":
        s.fact()
    else:
        s.resolve()
    assert not s.stats().final_stall_ineffective
    assert s.last_label() is not L.STALLED


def test_final_event_stall_ineffective_via_window_reversion():
    s = Session(initial_progress=0.5).edit("S1").redundant(6).edit("S0", E.REVERSION, undo=True)
    st = s.stats()
    assert (st.stall_ineffective, st.reversions, st.final_stall_ineffective) == (7, 1, True)
    assert s.labels() == {7: L.STALLED}


def test_eight_novel_fruitless_inspections_are_never_stalled():
    s = Session(initial_progress=0.5).none(8)
    st = s.stats()
    assert (st.stall_ineffective, st.none_effects, st.reversions) == (0, 8, 0)
    assert s.labels() == {7: L.INSUFFICIENT_AMBIGUOUS}


def test_eight_novel_fruitless_edits_are_never_stalled():
    s = Session(initial_progress=0.5)
    for i in range(8):
        s.edit(f"N{i}")
    st = s.stats()
    assert (st.stall_ineffective, st.reversions, st.none_effects) == (0, 0, 8)
    assert s.labels() == {7: L.INSUFFICIENT_AMBIGUOUS}


def test_long_aimless_session_never_stalled():
    s = Session(initial_progress=0.3)
    for i in range(40):
        if i % 3 == 0:
            s.edit(f"N{i}")
        else:
            s.none()
    labels = s.labels()
    assert len(labels) == 33
    assert set(labels.values()) == {L.INSUFFICIENT_AMBIGUOUS}


def test_none_ops_never_count_toward_i():
    s = Session(initial_progress=0.5).none(4).redundant(4)
    assert s.stats().stall_ineffective == 4
    s2 = Session(initial_progress=0.5).redundant(4).none(4)
    assert s2.stats().stall_ineffective == 4
    assert s.labels() == {7: L.INSUFFICIENT_AMBIGUOUS}
    assert s2.labels() == {7: L.INSUFFICIENT_AMBIGUOUS}


def test_in_policy_retries_and_polls_do_not_count_toward_i():
    s = Session(initial_progress=0.5).redundant(4).fail().retry(1).retry(2).retry(3)
    assert s.stats().stall_ineffective == 4
    p = Session(initial_progress=0.5).redundant(2).fail().poll(1).poll(2).poll(3).poll(4).poll(5)
    assert p.stats().stall_ineffective == 2


def test_stored_reversion_effect_alone_is_not_stall_ineffective():
    """The labeller recomputes reversions; a REVERSION effect label without the structure is ignored."""
    s = Session(initial_progress=0.5).redundant(7).op(E.REVERSION)  # no mutation, no undo
    st = s.stats()
    assert (st.stall_ineffective, st.reversions, st.final_stall_ineffective) == (7, 0, False)
    assert s.labels() == {7: L.INSUFFICIENT_AMBIGUOUS}


# --------------------------------------------------------------------------
# Episode in policy at the endpoint
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, attempts, in_policy, expected",
    [
        ("retry", 1, True, L.RECOVERY),
        ("retry", 3, True, L.RECOVERY),  # R_max
        ("retry", 4, False, L.STALLED),
        ("retry", 9, False, L.STALLED),
        ("poll", 1, True, L.RECOVERY),
        ("poll", 5, True, L.RECOVERY),  # P_max
        ("poll", 6, False, L.STALLED),
    ],
)
def test_endpoint_episode_policy_boundary_blocks_stalled(kind, attempts, in_policy, expected):
    """Field-level discriminator: a stall-ineffective endpoint (a redundant re-read made while the
    episode is still open) with everything else identical; only the attempt count changes."""
    s = Session(initial_progress=0.5).redundant(7).redundant(episode=(kind, attempts))
    st = s.stats()
    assert st.episode_in_policy_at_end is in_policy
    assert (st.stall_ineffective, st.final_stall_ineffective, st.facts) == (8, True, 0)
    assert s.labels() == {7: expected}


def test_poll_budget_is_not_the_retry_budget():
    """Poll attempts 4-5 are in policy although they exceed R_max."""
    s = Session(initial_progress=0.5).redundant(7).redundant(episode=("poll", 4))
    assert s.stats().episode_in_policy_at_end
    r = Session(initial_progress=0.5).redundant(7).redundant(episode=("retry", 4))
    assert not r.stats().episode_in_policy_at_end


def test_realistic_retry_episode_recovery_then_stalled():
    """Family-5 shape: initial failure, three in-policy retries, then doomed retries."""
    s = Session(initial_progress=0.5).none(4).fail().retry(1).retry(2).retry(3)
    for a in range(4, 9):
        s.retry(a)
    assert [x.effect for x in s.steps[8:]] == [E.REDUNDANT] * 5
    labels = s.labels()
    assert labels[7] is L.RECOVERY  # endpoint is retry 3: in policy
    assert labels[8] is L.INSUFFICIENT_AMBIGUOUS  # retry 4: out of policy, I = 1
    assert labels[11] is L.INSUFFICIENT_AMBIGUOUS  # I = 4 (retries 4..7)
    assert labels[12] is L.STALLED  # I = 5 (retries 4..8)
    assert s.stats(12).stall_ineffective == 5


def test_realistic_poll_episode_recovery_then_stalled():
    s = Session(initial_progress=0.5).none(2).fail()
    for a in range(1, 11):
        s.poll(a)
    labels = s.labels()
    assert labels[7] is L.RECOVERY  # poll 5
    assert labels[8] is L.INSUFFICIENT_AMBIGUOUS  # poll 6: out of policy, I = 1
    assert labels[12] is L.STALLED  # polls 6..10 redundant, I = 5


def test_abandoned_episode_mid_window_does_not_block_stalled():
    """Only the endpoint's episode state matters."""
    s = Session(initial_progress=0.5).fail().retry(1).retry(2).redundant(5)
    st = s.stats()
    assert not st.episode_in_policy_at_end and st.stall_ineffective == 5
    assert s.labels() == {7: L.STALLED}


def test_characterise_open_episode_with_zero_attempts_is_in_policy():
    """Characterisation of a contract hazard (reported): latent.py documents 'the initial failing
    operation counts 0', and validate_steps accepts an open episode with 0 attempts. The labeller
    treats it as in policy, so an ordinary first test failure would become RECOVERY. M1 §1 forbids
    generators from opening the episode before the first exact retry; this test pins what happens
    if one does."""
    s = Session(initial_progress=0.5).none(7).op(E.NONE, episode=("retry", 0))
    assert s.stats().episode_in_policy_at_end
    assert s.labels() == {7: L.RECOVERY}


# --------------------------------------------------------------------------
# RECOVERY recency (k_rec)
# --------------------------------------------------------------------------


def _resolution_session(after: str = "none") -> Session:
    s = Session(initial_progress=0.0).none(5).fail().retry(1).resolve()  # resolve at index 7
    if after == "none":
        s.none(4)
    else:
        for i, p in enumerate((0.2, 0.4, 0.6, 0.8)):
            s.progress_to(p, to=f"P{i}")
    return s


def test_resolution_recency_boundary():
    s = _resolution_session()
    assert [s.stats(t).resolved_recently for t in range(7, 12)] == [True, True, True, True, False]
    labels = s.labels()
    assert labels[7] is L.RECOVERY  # resolution at the endpoint
    assert labels[10] is L.RECOVERY  # endpoint - 3
    assert labels[11] is L.INSUFFICIENT_AMBIGUOUS  # endpoint - 4: no longer recent


def test_resolution_then_productive_work():
    s = _resolution_session("productive")
    labels = s.labels()
    assert s.stats(10).delta_p > EPS
    assert labels[10] is L.RECOVERY  # ratified: resolution within k_rec outranks progress
    assert labels[11] is L.PRODUCTIVE


def test_k_rec_zero_disables_recency():
    """Guards the ``window[-0:]`` slice trap: k_rec = 0 must mean no recent events, not all 8."""
    cfg0 = dataclasses.replace(CFG, k_rec=0)
    s = _resolution_session()
    assert not s.stats(7, cfg0).resolved_recently
    assert s.labels(cfg0)[7] is L.INSUFFICIENT_AMBIGUOUS


def test_k_rec_eight_covers_whole_window():
    cfg8 = dataclasses.replace(CFG, k_rec=8)
    s = _resolution_session().none(4)  # resolve at index 7, steps 0..15
    labels = s.labels(cfg8)
    assert labels[11] is L.RECOVERY  # resolution at endpoint - 4
    assert labels[14] is L.RECOVERY  # endpoint - 7 (window start)
    assert labels[15] is L.INSUFFICIENT_AMBIGUOUS  # resolution left the window
    assert s.labels()[11] is L.INSUFFICIENT_AMBIGUOUS  # frozen k_rec = 4 for contrast


def test_resolution_flag_drives_recency():
    s = Session().none(7).resolve()
    assert s.stats().resolved_recently
    assert s.labels() == {7: L.RECOVERY}


# --------------------------------------------------------------------------
# Window-relative reversion
# --------------------------------------------------------------------------


def test_reverting_to_first_events_before_state_counts():
    s = Session().edit("S1").none(6).edit("S0", E.REVERSION, undo=True)
    st = s.stats()
    assert st.reversions == 1 and st.stall_ineffective == 1


def test_reversion_whose_matching_state_falls_out_no_longer_counts():
    """Step 7 reverts to S0; S0 is the before-state of window 7 but not held anywhere in window 8.
    The stored REVERSION effect is informational and is ignored for window 8."""
    s = Session(initial_progress=0.5).edit("S1").none(6).edit("S0", E.REVERSION, undo=True).none()
    assert s.stats(7).reversions == 1
    st8 = s.stats(8)
    assert (st8.reversions, st8.stall_ineffective) == (0, 0)


def test_reversion_to_intermediate_state_counts():
    s = Session().edit("S1").edit("S2").none(5).edit("S1", E.REVERSION, undo=True)
    assert s.stats().reversions == 1


def test_reversion_to_state_held_only_before_the_window_does_not_count():
    """Matching state S0 exists earlier in the session but not in this window (engine: effect NONE)."""
    s = Session(initial_progress=0.5).edit("S1").redundant(7).edit("S0", E.NONE, undo=True)
    st = s.stats(8)  # window 1..8 starts at S1
    assert (st.reversions, st.stall_ineffective, st.final_stall_ineffective) == (0, 7, False)
    assert s.labels()[8] is L.INSUFFICIENT_AMBIGUOUS


def test_external_reset_to_seen_state_does_not_count():
    s = Session().edit("S1").none(5).op(E.NONE, to="S0", mutation=False).none()
    st = s.stats()
    assert st.reversions == 0 and st.stall_ineffective == 0


def test_external_reset_cannot_claim_undo():
    with pytest.raises(LatentError, match="undoes_prior_mutation requires agent_mutation"):
        step(ws_after="S1", agent_mutation=False, undoes_prior_mutation=True)


def test_non_undoing_mutation_to_seen_state_does_not_count():
    s = Session().edit("S1").none(5).edit("S0", undo=False).none()
    st = s.stats()
    assert st.reversions == 0 and st.stall_ineffective == 0


def test_undo_flag_to_novel_state_does_not_count():
    s = Session().edit("S1").none(5).edit("S9", undo=True).none()
    assert s.stats().reversions == 0


def test_reversion_recomputed_from_workspace_not_stored_effect():
    """The REVERSION effect is informational: structure decides."""
    s = Session().edit("S1").none(5).edit("S0", E.NONE, undo=True).none()
    assert s.stats().reversions == 1


def test_single_intentional_revert_is_not_oscillating():
    """Family-1 lookalike: one revert among new edits, no net progress, V = 1 < N_rev."""
    s = Session(initial_progress=0.5).edit("A").none().edit("S0", E.REVERSION, undo=True).none()
    s.edit("B").none().edit("C").none()
    st = s.stats()
    assert st.reversions == 1
    assert s.labels() == {7: L.INSUFFICIENT_AMBIGUOUS}


def test_abab_oscillation_counts_each_undo():
    s = _osc_window()
    st = s.stats()
    assert (st.reversions, st.stall_ineffective, st.delta_p) == (3, 3, 0.0)
    assert s.labels() == {7: L.OSCILLATING}


def test_two_reversions_is_oscillating():
    s = Session(initial_progress=0.5).edit("A").none(2).edit("S0", E.REVERSION, undo=True).none(2)
    s.edit("A", E.REVERSION, undo=True).none()
    assert s.stats().reversions == 2
    assert s.labels() == {7: L.OSCILLATING}


def test_oscillation_ends_as_states_slide_out():
    s = Session(initial_progress=0.5).edit("A").edit("S0", E.REVERSION, undo=True).edit("A", E.REVERSION, undo=True)
    s.none(13)
    labels = s.labels()
    assert s.stats(7).reversions == 2
    # Window 8 starts at index 1 (state A): A->S0 at 1 does not match (S0 not yet held), S0->A at 2 does.
    assert s.stats(8).reversions == 1
    assert labels[7] is L.OSCILLATING
    assert labels[8] is L.INSUFFICIENT_AMBIGUOUS


def test_characterise_progress_effect_reversion_counts_as_stall_ineffective():
    """Characterisation of a contract ambiguity (reported): spec §6.1 defines a reversion
    structurally, and the labeller counts every structural reversion in both V and I. M1 §1's
    precedence (PROGRESS > REVERSION) gives this undo-of-a-regression the stored effect PROGRESS;
    a literal 'stall-ineffective iff effect is REDUNDANT or REVERSION' reading would give I = 4 and
    INSUFFICIENT_AMBIGUOUS instead of STALLED."""
    s = Session(initial_progress=0.5).redundant(4).none(2)
    s.edit("X", progress=0.25)  # regression-introducing edit
    s.edit("S0", E.PROGRESS, undo=True, progress=0.5)  # revert fixes the regression
    st = s.stats()
    assert (st.delta_p, st.facts, st.reversions) == (0.0, 0, 1)
    assert (st.stall_ineffective, st.final_stall_ineffective) == (5, True)
    assert s.labels() == {7: L.STALLED}


# --------------------------------------------------------------------------
# label_session: keys, windows, config, purity
# --------------------------------------------------------------------------


def test_fewer_than_eight_steps_gives_no_labels():
    assert label_session([], 0.0, CFG) == {}
    assert Session().none(7).labels() == {}


def test_exactly_one_window_for_eight_steps():
    assert list(Session().none(8).labels()) == [7]


def test_labels_keyed_by_endpoint_sequence_index():
    s = Session(start_index=5).none(10)
    labels = s.labels()
    assert list(labels) == [12, 13, 14]
    assert all(isinstance(v, OracleLabel) for v in labels.values())


def test_window_stats_positions_are_list_positions():
    s = Session(start_index=100).none(9)
    st = s.stats(8)
    assert (st.start, st.end) == (1, 8)
    with pytest.raises(IndexError):
        window_stats(s.steps, 0.0, 6, CFG)
    with pytest.raises(IndexError):
        window_stats(s.steps, 0.0, 9, CFG)


def test_label_session_default_config_is_frozen_config():
    s = _osc_window()
    s2 = _resolution_session()
    assert label_session(s.steps, s.initial_progress) == label_session(s.steps, s.initial_progress, CFG)
    assert label_session(s2.steps, 0.0) == label_session(s2.steps, 0.0, CFG)


def test_label_session_is_pure_and_deterministic():
    s = _resolution_session("productive")
    before = [x.to_dict() for x in s.steps]
    a = label_session(s.steps, 0.0, CFG)
    b = label_session(tuple(s.steps), 0.0, CFG)
    assert a == b
    assert [x.to_dict() for x in s.steps] == before


def test_label_session_validates():
    s = Session().none(8)
    s.steps[3] = dataclasses.replace(s.steps[3], sequence_index=99)
    with pytest.raises(LatentError, match="non-dense"):
        label_session(s.steps, 0.0, CFG)


def test_label_session_validates_with_passed_config():
    s = Session().none(6).fail().retry(1).retry(2)
    assert s.labels()[8] is L.RECOVERY
    with pytest.raises(LatentError, match="IN_POLICY_RETRY outside"):
        s.labels(dataclasses.replace(CFG, r_max=1))


# --------------------------------------------------------------------------
# validate_steps
# --------------------------------------------------------------------------


def _bad(mutate):
    s = Session(initial_progress=0.5).none(3)
    mutate(s)
    return s.steps


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: [step(sequence_index=0), step(sequence_index=2)], "non-dense"),
        (lambda: [step(sequence_index=3), step(sequence_index=3)], "non-dense"),
        (lambda: [step(sequence_index=3), step(sequence_index=2)], "non-dense"),
        (lambda: [step(sequence_index=0, ws_after="S0"), step(sequence_index=1, ws_before="S9", ws_after="S9")], "ws_before"),
        (lambda: [step(effect=E.IN_POLICY_RETRY)], "IN_POLICY_RETRY outside"),
        (lambda: [step(effect=E.IN_POLICY_RETRY, open_episode_kind="poll", open_episode_attempts=1)], "IN_POLICY_RETRY outside"),
        (lambda: [step(effect=E.IN_POLICY_RETRY, open_episode_kind="retry", open_episode_attempts=0)], "IN_POLICY_RETRY outside"),
        (lambda: [step(effect=E.IN_POLICY_RETRY, open_episode_kind="retry", open_episode_attempts=4)], "IN_POLICY_RETRY outside"),
        (lambda: [step(effect=E.IN_POLICY_POLL)], "IN_POLICY_POLL outside"),
        (lambda: [step(effect=E.IN_POLICY_POLL, open_episode_kind="retry", open_episode_attempts=1)], "IN_POLICY_POLL outside"),
        (lambda: [step(effect=E.IN_POLICY_POLL, open_episode_kind="poll", open_episode_attempts=0)], "IN_POLICY_POLL outside"),
        (lambda: [step(effect=E.IN_POLICY_POLL, open_episode_kind="poll", open_episode_attempts=6)], "IN_POLICY_POLL outside"),
        (lambda: [step(effect=E.RESOLVES_EPISODE)], "RESOLVES_EPISODE without"),
        (lambda: [step(progress_after=0.5), step(sequence_index=1, effect=E.PROGRESS, progress_after=0.5)], "PROGRESS without"),
        (lambda: [step(progress_after=0.5), step(sequence_index=1, effect=E.PROGRESS, progress_after=0.4)], "PROGRESS without"),
        (lambda: [step(effect=E.NEW_FACT, new_facts=0)], "NEW_FACT with new_facts == 0"),
    ],
    ids=[
        "gap",
        "duplicate",
        "decreasing",
        "ws-chain",
        "retry-no-episode",
        "retry-in-poll-episode",
        "retry-attempts-0",
        "retry-attempts-over-rmax",
        "poll-no-episode",
        "poll-in-retry-episode",
        "poll-attempts-0",
        "poll-attempts-over-pmax",
        "resolves-without-flag",
        "progress-equal",
        "progress-decrease",
        "new-fact-zero",
    ],
)
def test_validate_steps_errors(build, match):
    with pytest.raises(LatentError, match=match):
        validate_steps(build(), CFG)


def test_validate_steps_error_mid_session_reports_position():
    s = Session().none(5)
    s.steps.append(step(sequence_index=5, ws_before="S0", ws_after="S0", effect=E.NEW_FACT, new_facts=0))
    with pytest.raises(LatentError, match="step 5"):
        validate_steps(s.steps, CFG)


@pytest.mark.parametrize(
    "kw",
    [
        dict(effect=E.IN_POLICY_RETRY, open_episode_kind="retry", open_episode_attempts=1),
        dict(effect=E.IN_POLICY_RETRY, open_episode_kind="retry", open_episode_attempts=3),
        dict(effect=E.IN_POLICY_POLL, open_episode_kind="poll", open_episode_attempts=1),
        dict(effect=E.IN_POLICY_POLL, open_episode_kind="poll", open_episode_attempts=5),
        dict(effect=E.REDUNDANT, open_episode_kind="retry", open_episode_attempts=4),
        dict(effect=E.REDUNDANT, open_episode_kind="poll", open_episode_attempts=6),
        dict(effect=E.RESOLVES_EPISODE, resolved_episode_here=True),
        dict(effect=E.NEW_FACT, new_facts=1),
        dict(effect=E.PROGRESS, progress_after=0.0),  # first step is not compared with anything
    ],
)
def test_validate_steps_accepts_boundaries(kw):
    validate_steps([step(**kw)], CFG)


def test_validate_steps_accepts_empty_and_consistent_sessions():
    validate_steps([], CFG)
    for build, _ in SCENARIOS.values():
        validate_steps(build().steps, CFG)
    validate_steps(_resolution_session("productive").steps, CFG)


def test_validate_steps_uses_config_budgets():
    tight = dataclasses.replace(CFG, r_max=1, p_max=1)
    validate_steps([step(effect=E.IN_POLICY_RETRY, open_episode_kind="retry", open_episode_attempts=1)], tight)
    with pytest.raises(LatentError):
        validate_steps([step(effect=E.IN_POLICY_RETRY, open_episode_kind="retry", open_episode_attempts=2)], tight)
    with pytest.raises(LatentError):
        validate_steps([step(effect=E.IN_POLICY_POLL, open_episode_kind="poll", open_episode_attempts=2)], tight)


# --------------------------------------------------------------------------
# LatentStep validation and serialisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        dict(sequence_index=-1),
        dict(sequence_index=1.0),
        dict(sequence_index="1"),
        dict(progress_after=-0.01),
        dict(progress_after=1.01),
        dict(progress_after=float("nan")),
        dict(progress_after=float("inf")),
        dict(new_facts=-1),
        dict(new_facts=1.0),
        dict(ws_before=""),
        dict(ws_after=""),
        dict(ws_before=None),
        dict(ws_after=7),
        dict(open_episode_kind="backoff", open_episode_attempts=1),
        dict(open_episode_kind="", open_episode_attempts=0),
        dict(open_episode_attempts=1),  # attempts without an open episode
        dict(open_episode_kind="retry", open_episode_attempts=-1),
        dict(undoes_prior_mutation=True),  # requires agent_mutation
        dict(agent_mutation=True),  # requires ws_before != ws_after
    ],
)
def test_latent_step_rejects(kw):
    with pytest.raises(LatentError):
        step(**kw)


def test_latent_step_rejects_unknown_effect():
    with pytest.raises(ValueError):
        step(effect="STUCK")


def test_latent_step_accepts_boundaries_and_coerces_effect():
    assert step(progress_after=0.0).progress_after == 0.0
    assert step(progress_after=1.0).progress_after == 1.0
    s = step(effect="REDUNDANT")
    assert s.effect is E.REDUNDANT
    for kind in EPISODE_KINDS:
        assert step(open_episode_kind=kind, open_episode_attempts=0).open_episode_kind == kind
    assert step(ws_after="S1", agent_mutation=True, undoes_prior_mutation=True).undoes_prior_mutation
    assert step(ws_after="S1", agent_mutation=False).ws_after == "S1"  # external change


def test_latent_step_is_frozen():
    s = step()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.effect = E.PROGRESS  # type: ignore[misc]


def test_latent_step_dict_round_trip():
    s = step(
        sequence_index=4,
        effect=E.REVERSION,
        progress_after=0.75,
        new_facts=3,
        ws_before="A",
        ws_after="B",
        agent_mutation=True,
        undoes_prior_mutation=True,
        open_episode_kind="poll",
        open_episode_attempts=2,
        resolved_episode_here=True,
        dropped=True,
    )
    d = s.to_dict()
    assert set(d) == {f.name for f in dataclasses.fields(LatentStep)}
    assert d["effect"] == "REVERSION" and isinstance(d["effect"], str)
    assert LatentStep.from_dict(d) == s
    assert LatentStep.from_dict(json.loads(json.dumps(d))) == s
    assert LatentStep.from_dict({**d, "effect": E.REVERSION}) == s


def test_latent_step_dict_round_trip_whole_sessions():
    for build, _ in SCENARIOS.values():
        steps = build().steps
        assert [LatentStep.from_dict(json.loads(json.dumps(x.to_dict()))) for x in steps] == steps


def test_latent_trace_payload():
    s = _resolution_session()
    t = latent_trace(s.steps, 0)
    assert set(t) == {"initial_progress", "steps"}
    assert t["initial_progress"] == 0.0 and isinstance(t["initial_progress"], float)
    assert t["steps"] == [x.to_dict() for x in s.steps]
    json.dumps(t)  # JSON-serialisable


@pytest.mark.parametrize("p0", [-0.01, 1.01, float("nan")])
def test_latent_trace_rejects_bad_initial_progress(p0):
    with pytest.raises(LatentError):
        latent_trace([], p0)


def _record(trace) -> OracleRecord:
    return OracleRecord(
        session_id="s-1",
        generator_version="test",
        seed=1,
        scenario_family="test",
        event_effects=(),
        latent_trace=trace,
    )


def test_steps_from_record_round_trip_through_json():
    s = _resolution_session("productive")
    steps = s.steps
    rec = OracleRecord(
        session_id="s-1",
        generator_version="test",
        seed=7,
        scenario_family="test",
        event_effects=tuple(x.effect for x in steps),
        window_labels=label_session(steps, 0.0, CFG),
        latent_trace=latent_trace(steps, 0.0),
    )
    rec2 = OracleRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
    got_steps, p0 = steps_from_record(rec2)
    assert got_steps == steps and p0 == 0.0
    assert label_session(got_steps, p0, CFG) == rec2.window_labels


def test_steps_from_record_preserves_initial_progress():
    s = Session(initial_progress=0.49).progress_to(0.5, to="S1").none(7)
    got, p0 = steps_from_record(_record(latent_trace(s.steps, 0.49)))
    assert p0 == 0.49
    assert label_session(got, p0, CFG) == {7: L.INSUFFICIENT_AMBIGUOUS}


@pytest.mark.parametrize(
    "trace",
    [
        {},
        {"initial_progress": 0.0},
        {"steps": []},
        {"initial_progress": 0.0, "steps": None},
        {"initial_progress": 0.0, "steps": ["not-a-dict"]},
        {"initial_progress": 0.0, "steps": [{"sequence_index": 0}]},  # missing effect
        {"initial_progress": 0.0, "steps": [{**step().to_dict(), "colour": "red"}]},  # unknown key
        {"initial_progress": 0.0, "steps": [{k: v for k, v in step().to_dict().items() if k != "ws_after"}]},
        {"initial_progress": None, "steps": []},
        {"initial_progress": 0.0, "steps": [{**step().to_dict(), "progress_after": 2.0}]},  # invalid value
    ],
    ids=[
        "empty",
        "no-steps",
        "no-initial-progress",
        "steps-none",
        "step-not-dict",
        "step-missing-effect",
        "step-unknown-key",
        "step-missing-field",
        "initial-progress-none",
        "step-invalid-value",
    ],
)
def test_steps_from_record_malformed(trace):
    with pytest.raises(LatentError):
        steps_from_record(_record(trace))
