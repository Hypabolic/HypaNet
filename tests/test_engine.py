"""Engine effect rules (M1_CONTRACTS §1, §2, §7 "Engine")."""

from __future__ import annotations

import random

import pytest

from agent_loop_lab.fingerprint import workspace_digest
from agent_loop_lab.oracle_config import load_oracle_config
from agent_loop_lab.schema import LatentEffect as E
from agent_loop_lab.schema import OracleLabel, Outcome
from agent_loop_lab.synth.engine import (
    GENERATOR_VERSION,
    PROVENANCE_SOURCE,
    Session,
    SessionLengthError,
    competing,
    cosmetic,
    fix,
    neutral,
    undo,
    wrong,
)
from agent_loop_lab.synth.labeller import label_session, validate_steps
from agent_loop_lab.synth.world import Resource, Service, World
from agent_loop_lab.windows import iter_windows
from tests import synth_helpers as H

CFG = load_oracle_config()
SUITE = "tests/unit"
P = "src/app/parser.py"
C = "src/app/cache.py"
F1_TESTS = ["tests/unit/test_parser.py::test_header_a", "tests/unit/test_parser.py::test_header_b"]
F2_TESTS = ["tests/unit/test_cache.py::test_lookup_a", "tests/unit/test_cache.py::test_lookup_b"]
HEALTHY = "tests/unit/test_parser.py::test_split"


def small_world() -> World:
    return World.from_spec(
        language="python",
        files={
            P: ["parse_header", "split_rows", "merge_ranges"],
            C: ["lookup_key", "evict"],
            "README.md": ["intro", "usage", "faq"],
            "docs/runbook.md": ["oncall", "deploy"],
            "config/settings.yaml": ["parser.limit", "cache.ttl"],
            "scripts/release.sh": ["main"],
        },
        faults=[
            dict(id="F1", path=P, block="parse_header", suite=SUITE, tests=F1_TESTS, error_class="KeyError",
                 clues=[("read", "config/settings.yaml", "parser.limit"), ("lint", "ruff")]),
            dict(id="F2", path=C, block="lookup_key", suite=SUITE, tests=F2_TESTS, error_class="TypeError",
                 clues=[("http", "sentry.io", "/api/0/issues/{issue_id}/events/")]),
        ],
        healthy_tests={SUITE: [(HEALTHY, [f"{P}:split_rows"])]},
        services=[
            Service("api.example.com", "transient", 2, "TimeoutError"),
            Service("down.example.com", "persistent", 0, "HTTP503"),
            Service("auth.example.com", "needs_fix", 0, "HTTP401", requires="F2"),
            Service("sentry.io", "transient", 1, "HTTP502"),
        ],
        resources=[Resource("ci/pipeline/1", 3), Resource("deploy/never", None)],
    )


def new_session(seed: int = 0, world: World | None = None) -> Session:
    return Session(random.Random(seed), "eng-1", CFG, family="engine_test", world=world or small_world())


def effects(results) -> list[E]:
    return [r.effect for r in results]


# -- world / event plumbing ----------------------------------------------------


def test_digests_are_workspace_digest_and_inspect_before_equals_after():
    s = new_session()
    r0 = s.read(P)
    assert r0.event.workspace_state_before == r0.event.workspace_state_after == workspace_digest(s.world.texts())
    r1 = s.edit(P, neutral())
    assert r1.event.workspace_state_before == r0.event.workspace_state_after
    assert r1.event.workspace_state_after == workspace_digest(s.world.texts()) != r1.event.workspace_state_before
    assert r1.step.agent_mutation and r1.step.ws_after == r1.event.workspace_state_after
    assert r0.event.provenance.source == PROVENANCE_SOURCE == f"simulator:{GENERATOR_VERSION}"


def test_tests_pass_iff_faults_fixed_and_no_regression():
    s = new_session()
    r = s.run_tests(SUITE)
    assert (r.passed, r.total, r.outcome) == (1, 5, Outcome.ERROR)
    assert r.failing_test == F2_TESTS[0] and r.error_class == "TypeError"  # first failing test in suite order
    s.edit(C, fix("F2"))
    s.edit(P, fix("F1"))
    assert s.run_tests(SUITE).outcome is Outcome.SUCCESS
    s.edit(P, wrong(regress=True), block="split_rows")
    r = s.run_tests(SUITE)
    assert r.failed and r.failing_test == HEALTHY and (r.passed, r.total) == (4, 5)
    assert s.progress == pytest.approx(0.75)  # (2 fixed - 0.5 * 1 regression) / 2


def test_same_edit_repeated_gives_same_intent_and_different_edits_differ():
    s = new_session()
    a = s.edit(C, competing("F2", "A"))
    b = s.edit(C, competing("F2", "B"))
    a2 = s.edit(C, competing("F2", "A"))
    b2 = s.edit(C, competing("F2", "B"))
    assert a.event.intent_fingerprint != b.event.intent_fingerprint
    assert a2.event.intent_fingerprint != a.event.intent_fingerprint  # A from orig vs A from B: different patches
    assert b2.event.intent_fingerprint == b.event.intent_fingerprint  # identical patch (A -> B) both times


# -- rule 1: episodes ------------------------------------------------------------


def test_episode_opens_on_first_exact_retry_only_and_goes_out_of_policy():
    s = new_session()
    rs = [s.run_tests(SUITE) for _ in range(5)]
    assert effects(rs) == [E.NEW_FACT, E.IN_POLICY_RETRY, E.IN_POLICY_RETRY, E.IN_POLICY_RETRY, E.REDUNDANT]
    assert [(r.step.open_episode_kind, r.step.open_episode_attempts) for r in rs] == [
        (None, 0), ("retry", 1), ("retry", 2), ("retry", 3), ("retry", 4)]


def test_initial_failure_without_new_fact_is_none():
    s = new_session()
    r = s.http("GET", "api.example.com", "/v1/x")
    assert r.failed and r.effect is E.NONE and r.step.open_episode_kind is None


def test_no_episode_when_retry_is_not_immediate():
    s = new_session()
    rs = [s.run_tests(SUITE), s.read("README.md"), s.run_tests(SUITE)]
    assert effects(rs) == [E.NEW_FACT, E.NONE, E.REDUNDANT]
    assert rs[2].step.open_episode_kind is None


def test_open_episode_tolerates_one_unrelated_op_and_closes_at_the_second():
    s = new_session()
    s.run_tests(SUITE)
    s.run_tests(SUITE)
    r = s.read("README.md")
    assert r.step.open_episode_kind == "retry" and r.step.open_episode_attempts == 1
    r = s.run_tests(SUITE)
    assert r.effect is E.IN_POLICY_RETRY and r.step.open_episode_attempts == 2
    s.read("README.md", (1, 2))
    r = s.read("docs/runbook.md")
    assert r.step.open_episode_kind is None
    assert s.run_tests(SUITE).effect is E.REDUNDANT


def test_text_only_edit_does_not_close_but_behaviour_change_does():
    s = new_session()
    s.run_tests(SUITE, F2_TESTS)
    s.run_tests(SUITE, F2_TESTS)
    r = s.edit(C, cosmetic("comment"))
    assert r.effect is E.REDUNDANT and r.step.open_episode_kind == "retry"
    assert s.run_tests(SUITE, F2_TESTS).effect is E.IN_POLICY_RETRY
    r = s.edit(P, neutral())
    assert r.effect is E.NONE and r.step.open_episode_kind is None


def test_diagnosed_repair_keeps_episode_open_and_retry_resolves():
    s = new_session()
    rs = [s.run_tests(SUITE, F2_TESTS), s.run_tests(SUITE, F2_TESTS), s.read(C, block="lookup_key"),
          s.edit(C, fix("F2")), s.run_tests(SUITE, F2_TESTS)]
    assert effects(rs) == [E.NEW_FACT, E.IN_POLICY_RETRY, E.NEW_FACT, E.PROGRESS, E.RESOLVES_EPISODE]
    assert rs[3].step.open_episode_kind == "retry" and rs[3].step.progress_after == 0.5
    assert rs[4].step.resolved_episode_here and rs[4].step.open_episode_kind is None and rs[4].ok


def test_http_transient_recovers_persistent_goes_redundant():
    s = new_session()
    rs = [s.http("GET", "api.example.com", "/v1/x") for _ in range(3)]
    assert effects(rs) == [E.NONE, E.IN_POLICY_RETRY, E.RESOLVES_EPISODE]
    assert [r.outcome for r in rs] == [Outcome.ERROR, Outcome.ERROR, Outcome.SUCCESS]
    assert s.http("GET", "api.example.com", "/v1/x").effect is E.REDUNDANT  # identical successful call again

    s = new_session()
    rs = [s.http("POST", "down.example.com", "/v1/charges", body='{"a": 1}') for _ in range(5)]
    assert effects(rs) == [E.NONE, E.IN_POLICY_RETRY, E.IN_POLICY_RETRY, E.IN_POLICY_RETRY, E.REDUNDANT]
    assert rs[-1].step.open_episode_kind == "retry" and rs[-1].step.open_episode_attempts == 4
    assert len({r.event.failure_fingerprint for r in rs}) == 1


def test_http_needs_fix_is_a_diagnosed_repair():
    s = new_session()
    rs = [s.http("GET", "auth.example.com", "/oauth/token"), s.http("GET", "auth.example.com", "/oauth/token"),
          s.edit(C, fix("F2")), s.http("GET", "auth.example.com", "/oauth/token")]
    assert effects(rs) == [E.NONE, E.IN_POLICY_RETRY, E.PROGRESS, E.RESOLVES_EPISODE]


def test_poll_recovers_within_policy_and_doomed_poll_goes_redundant():
    s = new_session()
    rs = [s.poll("ci/pipeline/1") for _ in range(4)]
    assert effects(rs) == [E.NONE, E.IN_POLICY_POLL, E.IN_POLICY_POLL, E.RESOLVES_EPISODE]
    assert [r.outcome for r in rs] == [Outcome.NEUTRAL] * 3 + [Outcome.SUCCESS]
    assert s.poll("ci/pipeline/1").effect is E.REDUNDANT

    s = new_session()
    rs = [s.poll("deploy/never", backoff=2 ** i) for i in range(7)]
    assert effects(rs) == [E.NONE] + [E.IN_POLICY_POLL] * 5 + [E.REDUNDANT]
    assert rs[-1].step.open_episode_kind == "poll" and rs[-1].step.open_episode_attempts == 6


def test_new_episode_replaces_open_one():
    s = new_session()
    s.run_tests(SUITE, F2_TESTS)
    s.run_tests(SUITE, F2_TESTS)
    r = s.http("GET", "down.example.com", "/status")
    assert r.effect is E.NONE and r.step.open_episode_kind == "retry" and r.step.open_episode_attempts == 1
    r = s.http("GET", "down.example.com", "/status")
    assert r.effect is E.IN_POLICY_RETRY and r.step.open_episode_attempts == 1
    r = s.run_tests(SUITE, F2_TESTS)  # the old episode is gone: an unrelated, redundant re-run
    assert r.effect is E.REDUNDANT and r.step.open_episode_attempts == 1


def test_resolution_with_new_fact_is_resolves_and_records_facts():
    s = new_session()
    rs = [s.http("GET", "sentry.io", "/api/0/issues/{issue_id}/events/") for _ in range(2)]
    assert effects(rs) == [E.NONE, E.RESOLVES_EPISODE]
    assert rs[1].step.new_facts == 1 and s.knows("F2:c1")


# -- rules 2-4: progress, facts, redundancy -----------------------------------------


def test_progress_is_credited_to_the_fix_not_the_test_run():
    s = new_session()
    rs = [s.run_tests(SUITE), s.edit(C, fix("F2")), s.run_tests(SUITE)]
    assert effects(rs) == [E.NEW_FACT, E.PROGRESS, E.NEW_FACT]  # second run newly reveals F1's location
    assert s.run_tests(SUITE, F2_TESTS).effect is E.NONE


def test_facts_and_rereads():
    s = new_session()
    rs = [
        s.read(P, block="parse_header"),  # F1:code
        s.read(P),  # whole file: its only fact is known -> REDUNDANT although never read
        s.read("README.md"),  # novel, no facts
        s.read("README.md"),  # same content again
        s.search("src", "parse_header"),  # F1:loc
        s.search("src", "PARSE_header  "),  # same normalised query, same content
        s.lint("ruff"),  # reveals F1:c1, fails
        s.read("config/settings.yaml", block="parser.limit"),  # F1:c2... (config clue)
    ]
    assert effects(rs) == [E.NEW_FACT, E.REDUNDANT, E.NONE, E.REDUNDANT, E.NEW_FACT, E.REDUNDANT, E.NEW_FACT, E.NEW_FACT]
    assert rs[6].failed and rs[6].event.failure_fingerprint is not None


def test_reread_after_text_change_is_new_content():
    s = new_session()
    s.read("README.md")
    s.edit("README.md", cosmetic("comment"), block="intro")
    assert s.read("README.md").effect is E.NONE


def test_validation_redundant_on_behaviourally_unchanged_workspace():
    s = new_session()
    rs = [s.run_tests(SUITE, [HEALTHY]), s.edit(P, cosmetic("rename_local"), block="split_rows"),
          s.run_tests(SUITE, [HEALTHY]), s.edit(P, neutral(), block="merge_ranges"), s.run_tests(SUITE, [HEALTHY])]
    assert effects(rs) == [E.NONE, E.REDUNDANT, E.REDUNDANT, E.NONE, E.NONE]


def test_cosmetic_and_reapplied_fix_are_redundant_and_noop_format_is_not_a_mutation():
    s = new_session()
    assert s.edit(P, fix("F1")).effect is E.PROGRESS
    r = s.edit(P, fix("F1"))  # re-adding an existing guard: text changes, behaviour doesn't
    assert r.effect is E.REDUNDANT and r.step.agent_mutation
    r = s.edit(P, cosmetic("format"))
    assert r.effect is E.REDUNDANT and r.step.agent_mutation
    r = s.edit(P, cosmetic("format"))
    assert r.effect is E.REDUNDANT and not r.step.agent_mutation and r.step.ws_before == r.step.ws_after


def test_control_redundant_without_new_information():
    s = new_session()
    assert effects([s.plan("decompose"), s.plan("decompose")]) == [E.NONE, E.REDUNDANT]
    s = new_session()
    assert effects([s.plan("decompose"), s.read("README.md"), s.plan("decompose")]) == [E.NONE, E.NONE, E.NONE]


# -- rule 5: reversion -----------------------------------------------------------------


@pytest.mark.parametrize("gap,expected", [(6, E.REVERSION), (7, E.NONE)])
def test_reversion_only_within_trailing_window(gap, expected):
    s = new_session()
    s.edit(P, neutral(), block="merge_ranges")
    for _ in range(gap):
        s.novel_decoy()
    r = s.edit(P, undo())
    assert r.step.undoes_prior_mutation and r.step.agent_mutation
    assert r.effect is expected
    steps = s.steps
    validate_steps(steps, CFG)
    if len(steps) >= 8:
        from agent_loop_lab.synth.labeller import window_stats
        st = window_stats(steps, 0.0, len(steps) - 1, CFG)
        assert st.reversions == (1 if expected is E.REVERSION else 0)


def test_precedence_progress_over_reversion_and_reversion_over_redundant():
    s = new_session()
    s.edit(P, fix("F1"))
    s.edit(P, wrong(regress=True), block="split_rows")
    r = s.edit(P, undo())  # removes the regression: p rises and it reverts
    assert r.effect is E.PROGRESS and r.step.undoes_prior_mutation
    s.edit(C, cosmetic("rename_local"))
    r = s.edit(C, undo())  # text-only and a reversion
    assert r.effect is E.REVERSION


def test_abab_alternation_reverts_without_progress():
    s = new_session()
    rs = []
    for label in "ABABAB":
        rs.append(s.edit(C, competing("F2", label)))
        rs.append(s.run_tests(SUITE, F2_TESTS))
    assert all(r.step.progress_after == 0.0 for r in rs)
    edits = rs[0::2]
    assert effects(edits) == [E.NONE, E.NONE] + [E.REVERSION] * 4
    passes = [r.passed for r in rs[1::2]]
    assert passes == [1, 1, 1, 1, 1, 1] and len({r.failing_test for r in rs[1::2]}) == 2
    labels = label_session(s.steps, 0.0, CFG)
    assert labels[len(rs) - 1] is OracleLabel.OSCILLATING


# -- drops, truncation, fork, determinism ---------------------------------------------


def _assert_gate_agrees(events, record):
    labels = record.window_labels
    for w in iter_windows(events):
        assert w.gate.passed == (labels[w.endpoint_index] is not OracleLabel.INSUFFICIENT_INSTRUMENTATION), w.endpoint_index


def _filler(s: Session, n: int) -> None:
    for _ in range(n):
        s.novel_decoy()


def test_dropped_counts_propagate_to_next_same_suite_run_and_gate_agrees():
    s = new_session()
    _filler(s, 3)
    r = s.run_tests(SUITE, drop=("validation_counts",))
    assert r.event.validation_passed is None and r.step.dropped
    _filler(s, 9)
    s.run_tests(SUITE, [HEALTHY])  # different selection, same suite: baseline is the dropped run
    _filler(s, 9)
    events, record = s.finish(min_events=8)
    assert record.dropped_observations == (3, 13)
    assert record.latent_trace["drops"]["propagated"] == [13]
    _assert_gate_agrees(events, record)
    assert any(v is OracleLabel.INSUFFICIENT_INSTRUMENTATION for v in record.window_labels.values())


@pytest.mark.parametrize("field", ["tokens", "input_tokens", "digests", "digest_after", "failure"])
def test_single_field_drops_reject_exactly_their_windows(field):
    s = new_session()
    _filler(s, 9)
    r = s.run_tests(SUITE, drop=(field,))
    assert r.step.dropped
    _filler(s, 9)
    events, record = s.finish(min_events=8)
    assert record.dropped_observations == (9,)
    _assert_gate_agrees(events, record)
    H.assert_record_consistent(events, record, "engine_test")


def test_truncated_history_marks_first_observed_same_suite_run_dropped():
    s = new_session()
    s.run_tests(SUITE)
    _filler(s, 3)
    s.run_tests(SUITE, [HEALTHY])  # second suite-run... observed after truncation, baseline cut off
    _filler(s, 10)
    s.run_tests(SUITE, [HEALTHY])  # baseline is the previous (observed) run: fine
    _filler(s, 4)
    events, record = s.finish(truncate_before=2, min_events=8)
    assert events[0].sequence_index == 2 and not events[0].provenance.history_complete
    assert record.dropped_observations == (4,)
    assert record.latent_trace["initial_progress"] == 0.0
    _assert_gate_agrees(events, record)
    H.assert_record_consistent(events, record, "engine_test")


def test_bad_drop_requests_are_rejected():
    s = new_session()
    with pytest.raises(ValueError):
        s.read("README.md", drop=("failure",))
    with pytest.raises(ValueError):
        s.read("README.md", drop=("validation_counts",))
    with pytest.raises(ValueError):
        s.read("README.md", drop=("nonsense",))


def test_length_bounds_enforced():
    s = new_session()
    _filler(s, 5)
    with pytest.raises(SessionLengthError):
        s.finish()


def test_fork_shares_prefix_then_diverges_on_one_variable():
    s = new_session(seed=7)
    s.run_tests(SUITE)
    s.read(C, block="lookup_key")
    a, b = s.fork("pair-a"), s.fork("pair-b")
    b.world.services["api.example.com"].mode = "persistent"
    ra = [a.http("GET", "api.example.com", "/v1/x") for _ in range(3)]
    rb = [b.http("GET", "api.example.com", "/v1/x") for _ in range(3)]
    for ea, eb in zip(a.events[:2], b.events[:2]):
        da, db = ea.to_dict(), eb.to_dict()
        for k in ("session_id", "event_id", "provenance"):
            da.pop(k), db.pop(k)
        assert da == db
    assert a.steps[:2] == b.steps[:2]
    # Same random draws while outcomes agree (a timeout and a success take different base durations).
    assert [r.event.timestamp for r in ra[:2]] == [r.event.timestamp for r in rb[:2]]
    assert [r.event.input_tokens for r in ra[:2]] == [r.event.input_tokens for r in rb[:2]]
    assert effects(ra)[-1] is E.RESOLVES_EPISODE and effects(rb)[-1] is E.IN_POLICY_RETRY
    assert all(e.session_id == "pair-a" for e in a.events)


def test_session_is_deterministic_and_rng_only():
    def run(seed):
        s = Session(random.Random(seed), "det", CFG, family="engine_test", n_faults=3)
        s.run_tests(next(iter(s.world.suites)))
        for f in s.world.faults.values():
            s.read(f.path, block=f.block)
            s.edit(f.path, fix(f))
        _filler(s, 6)
        events, record = s.finish()
        return [e.to_json() for e in events], record.to_dict()

    assert run(11) == run(11)
    assert run(11) != run(12)


def test_generated_world_vocabulary_varies():
    langs, hosts, suites = set(), set(), set()
    for seed in range(40):
        w = World.generate(random.Random(seed))
        langs.add(w.language)
        hosts.update(w.vocab.hosts)
        suites.update(w.suites)
        assert 1 <= len(w.faults) <= 4
        for f in w.faults.values():
            assert len(f.tests) >= 2 and f.loc_fact in f.facts and f.code_fact in f.facts
    assert len(langs) == 6 and len(hosts) >= 15 and len(suites) >= 10


# -- cost, drops and family registry do not touch latent effects ------------------


def _scripted(seed: int, *, mult: float = 1.0, drop_at: int | None = None):
    s = new_session(seed)
    s.cost_multiplier = mult
    ops = [lambda: s.run_tests(SUITE), lambda: s.read(C, block="lookup_key"), lambda: s.edit(C, fix("F2")),
           lambda: s.run_tests(SUITE), lambda: s.http("GET", "api.example.com", "/v1/x"),
           lambda: s.http("GET", "api.example.com", "/v1/x"), lambda: s.poll("ci/pipeline/1"), lambda: s.plan("checkpoint")]
    for i, op in enumerate(ops * 2):
        if i == drop_at:
            s.read("README.md", drop=("tokens",))
        else:
            op()
    return s.finish(min_events=8)


def test_cost_multiplier_changes_tokens_only():
    ev1, rec1 = _scripted(5)
    ev8, rec8 = _scripted(5, mult=8.0)
    assert rec1.to_dict() == rec8.to_dict()
    assert [e.timestamp for e in ev1] == [e.timestamp for e in ev8]
    assert all(b.input_tokens > a.input_tokens for a, b in zip(ev1, ev8))


def test_drop_does_not_perturb_the_random_stream():
    s1, s2 = new_session(9), new_session(9)
    for s, drop in ((s1, ()), (s2, ("tokens",))):
        s.run_tests(SUITE)
        s.read("README.md", drop=drop)
        s.edit(C, fix("F2"))
        s.run_tests(SUITE)
    for a, b in zip(s1.events, s2.events):
        da, db = a.to_dict(), b.to_dict()
        if a.sequence_index == 1:
            assert db["input_tokens"] is None and db["output_tokens"] is None
            for k in ("input_tokens", "output_tokens"):
                da.pop(k), db.pop(k)
        assert da == db
    assert [st.effect for st in s1.steps] == [st.effect for st in s2.steps]
    assert [st.dropped for st in s2.steps] == [False, True, False, False]


def test_family_registry():
    from agent_loop_lab.synth.families import FAMILY_NAMES, generate_session, get_family

    assert len(FAMILY_NAMES) == 12 and FAMILY_NAMES[:2] == ("incremental_repair", "exploration")
    with pytest.raises(KeyError):
        get_family("nope")
    for name in FAMILY_NAMES[:2]:
        assert get_family(name).FAMILY == name
    _, rec = generate_session("exploration", 123, "reg-1", CFG)
    assert rec.seed == 123 and rec.generator_version == GENERATOR_VERSION


# -- the STALLED recipes in docs/M1_ENGINE.md ---------------------------------------


def test_cosmetic_churn_with_non_immediate_reruns_stalls():
    s = new_session()
    s.run_tests(SUITE)  # NEW_FACT
    for _ in range(8):
        s.edit(C, cosmetic("rename_local"))
        s.run_tests(SUITE)
    _, rec = s.finish(min_events=8)
    assert all(e is E.REDUNDANT for e in rec.event_effects[1:])
    labels = [rec.window_labels[t] for t in sorted(rec.window_labels)]
    assert labels[0] is OracleLabel.INSUFFICIENT_AMBIGUOUS  # contains the revealing run (F = 1)
    assert all(lab is OracleLabel.STALLED for lab in labels[1:])


def test_retries_past_r_max_with_rereads_go_recovery_then_stalled():
    s = new_session()
    s.read("README.md")
    s.run_tests(SUITE)
    s.run_tests(SUITE)
    for _ in range(8):
        s.read("README.md")
        s.run_tests(SUITE)
    _, rec = s.finish(min_events=8)
    steps = s.steps
    assert [st.open_episode_attempts for st in steps[2::2]] == list(range(1, 10))
    labels = rec.window_labels
    assert labels[7] is OracleLabel.RECOVERY  # attempts 3 at the endpoint: still in policy
    # Attempt 4 is out of policy, but the window still holds the revealing run (F = 1): ambiguous.
    assert labels[8] is OracleLabel.INSUFFICIENT_AMBIGUOUS
    assert all(labels[t] is OracleLabel.STALLED for t in range(9, len(steps)))
