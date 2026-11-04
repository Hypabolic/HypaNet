"""Revision acceptance audit on the commissioned seeds 0 through 49."""

from collections import Counter
from itertools import combinations, groupby
import json
import random

import pytest

from agent_loop_lab.oracle_config import load_oracle_config
from agent_loop_lab.fingerprint import intent_fingerprint
from agent_loop_lab.schema import LatentEffect as E, OracleLabel as L, Outcome
from agent_loop_lab.synth.labeller import label_session, validate_steps
from agent_loop_lab.synth.latent import steps_from_record
from agent_loop_lab.synth_ood import FAMILIES
from agent_loop_lab.synth_ood.engine import Engine, PackageResolver
from agent_loop_lab.windows import GateReason, gate, validate_sequence


CFG = load_oracle_config("oracle_config.v0_2.yaml")


@pytest.fixture(scope="module")
def corpus():
    # Reuse these sessions across audits; no encoder runs or seed searching.
    return {name: [generate(random.Random(seed), "revision-audit", CFG)
                   for seed in range(50)] for name, generate in FAMILIES.items()}


def test_families_have_distinct_sessions_and_operation_multisets(corpus):
    shapes = {
        name: {tuple((e.operation_category, e.outcome) for e in events)
               for events, _ in sessions}
        for name, sessions in corpus.items()
    }
    operations = {
        name: Counter(e.operation_family for events, _ in sessions for e in events)
        for name, sessions in corpus.items()
    }
    for left, right in combinations(corpus, 2):
        # Compare every cross-seed pair as well as matching seed indices.
        assert shapes[left].isdisjoint(shapes[right]), (left, right)
        assert operations[left] != operations[right], (left, right)


def _package_target(method, version=None):
    return {"method": method, "host": PackageResolver.host,
            "path_template": f"/cedar/{version}" if method == "GET" else "/install/cedar"}


def _package_request(e, method, version=None):
    args = {"body": json.dumps({"version": version})} if method == "POST" else {}
    e.perform("http_call", _package_target(method, version), args)


@pytest.mark.parametrize("version", ["1", "2", "3"])
def test_package_builds_depend_on_fetched_artifacts_pins_and_abi(version):
    e = Engine(random.Random(7), "package-causality", CFG, dependencies=True)
    initial = e.repo.digest()
    _package_request(e, "POST", version)
    assert e.events[-1].outcome == Outcome.ERROR
    assert e.repo.digest() == initial
    _package_request(e, "GET", version)
    assert e.repo.digest() == initial
    assert e.steps[-1].effect == E.NEW_FACT
    _package_request(e, "GET", version)
    assert e.steps[-1].effect == E.REDUNDANT
    e.write(PackageResolver.lock_path, f"package=cedar\nversion={version}\n")
    e.perform("build", "cedar:dependency-link")
    assert e.events[-1].validation_passed == 0  # A pin alone cannot install.
    assert e.repo.progress() == 0
    _package_request(e, "POST", version)
    install = e.steps[-1]
    e.perform("build", "cedar:dependency-link")
    assert e.events[-1].validation_total == 1
    assert e.events[-1].validation_passed == int(version == "3")
    assert (e.events[-1].outcome == Outcome.SUCCESS) == (version == "3")
    assert (install.effect == E.PROGRESS) == (version == "3")
    if version != "3":
        # Change the actual application's ABI contract, not a scripted result.
        e.write(PackageResolver.manifest_path, f"required_abi={version}\n")
        assert e.steps[-1].effect == E.PROGRESS
        e.perform("build", "cedar:dependency-link")
        assert e.events[-1].outcome == Outcome.SUCCESS
    satisfied = e.repo.progress()
    e.write(PackageResolver.lock_path, "package=cedar\nversion=missing\n")
    e.perform("build", "cedar:dependency-link")
    assert e.events[-1].outcome == Outcome.ERROR
    assert e.repo.progress() < satisfied
    e.write(PackageResolver.lock_path, f"package=cedar\nversion={version}\n")
    _package_request(e, "DELETE")
    e.perform("build", "cedar:dependency-link")
    assert e.events[-1].outcome == Outcome.ERROR
    _package_request(e, "POST", version)
    assert e.steps[-1].effect == E.PROGRESS
    assert e.repo.progress() == satisfied
    validate_steps(e.steps, CFG)
    validate_sequence(e.events)


def test_lockfile_loops_restore_installed_states_then_really_resolve(corpus):
    uninstall = intent_fingerprint("http_call", _package_target("DELETE"))
    repair_routes = set()
    for events, record in corpus["lockfile_churn"]:
        steps, _ = steps_from_record(record)
        builds = [e for e in events if e.operation_family == "build"]
        assert len(builds) >= 7
        assert all(e.outcome == Outcome.ERROR and e.validation_passed == 0
                   for e in builds[:-1])
        assert builds[-1].outcome == Outcome.SUCCESS
        assert builds[-1].validation_passed == builds[-1].validation_total == 1
        assert any(e.operation_family == "http_call" and s.effect == E.REVERSION
                   for e, s in zip(events, steps))
        assert any(e.operation_family == "http_call" and s.effect == E.PROGRESS
                   for e, s in zip(events, steps))
        start, end = builds[0].sequence_index, builds[-2].sequence_index
        assert len({s.progress_after for s in steps[start:end+1]}) == 1
        repair_routes.add(any(e.intent_fingerprint == uninstall for e in events))
    assert repair_routes == {False, True}


@pytest.mark.parametrize("name", FAMILIES)
def test_structural_variety_over_fifty_seeds(corpus, name):
    shapes = [tuple((event.operation_family, event.outcome) for event in events)
              for events, _ in corpus[name]]
    assert len(set(shapes)) >= 40
    # Also vary operation order, rather than only the length of repeated runs.
    assert len({tuple(key for key, _ in groupby(shape)) for shape in shapes}) >= 10
    assert len({len(shape) for shape in shapes}) >= 5


@pytest.mark.parametrize("name", FAMILIES)
def test_every_session_mixes_scored_behaviours(corpus, name):
    for events, record in corpus[name]:
        assert 20 <= len(events) <= 90
        assert len({label for label in record.window_labels.values() if label.is_scored}) >= 2
        steps, p0 = steps_from_record(record)
        validate_sequence(events)
        validate_steps(steps, CFG)
        assert label_session(steps, p0, CFG) == record.window_labels


def _contains_in_order(labels, expected):
    remaining = iter(labels)
    return all(any(actual == desired for actual in remaining) for desired in expected)


@pytest.mark.parametrize("name,expected", [
    ("shard_migration", [L.PRODUCTIVE, L.RECOVERY, L.PRODUCTIVE]),
    ("fixture_archaeology", [L.PRODUCTIVE, L.STALLED, L.PRODUCTIVE]),
    ("lease_reconnect", [L.PRODUCTIVE, L.RECOVERY, L.PRODUCTIVE]),
    ("index_watermark", [L.PRODUCTIVE, L.RECOVERY, L.PRODUCTIVE]),
    ("pinned_validation", [L.PRODUCTIVE, L.RECOVERY, L.STALLED, L.PRODUCTIVE]),
    ("annotation_spiral", [L.STALLED, L.PRODUCTIVE]),
    ("serializer_duel", [L.OSCILLATING, L.PRODUCTIVE]),
    ("merge_ping_pong", [L.PRODUCTIVE, L.OSCILLATING, L.PRODUCTIVE]),
    ("lockfile_churn", [L.PRODUCTIVE, L.OSCILLATING, L.PRODUCTIVE]),
])
def test_behaviour_transitions_follow_real_work(corpus, name, expected):
    for _, record in corpus[name]:
        assert _contains_in_order(record.window_labels.values(), expected)
        steps, _ = steps_from_record(record)
        # The resumed productive work includes actual repairs, not only facts.
        last_problem = max(t for t, label in record.window_labels.items()
                           if label == expected[-2])
        assert any(steps[i].progress_after > steps[i-1].progress_after
                   for i in range(last_problem + 1, len(steps)))


def test_ambiguous_boundary_windows_are_retained(corpus):
    families = {name for name, sessions in corpus.items()
                if any(L.INSUFFICIENT_AMBIGUOUS in record.window_labels.values()
                       for _, record in sessions)}
    assert len(families) >= 3
    # Depending on phase alignment, recovery can turn directly into a stall.
    # Require ambiguous boundaries in the corpus without forcing every seed
    # to traverse the same label schedule.
    assert any(_contains_in_order(record.window_labels.values(),
                                  [L.PRODUCTIVE, L.INSUFFICIENT_AMBIGUOUS,
                                   L.STALLED, L.PRODUCTIVE])
               for _, record in corpus["pinned_validation"])


@pytest.mark.parametrize("opponent,label", [
    ("annotation_spiral", L.STALLED), ("serializer_duel", L.OSCILLATING),
])
def test_two_benign_lookalike_pairs(corpus, opponent, label):
    for (good_events, good), (bad_events, bad) in zip(
            corpus["shard_migration"], corpus[opponent]):
        # These are full, scored eight-event windows, not just similar counts.
        good_window, bad_window = good_events[9:17], bad_events[9:17]
        assert [(e.operation_category, e.outcome) for e in good_window] == [
            (e.operation_category, e.outcome) for e in bad_window]
        assert good.window_labels[16] == L.PRODUCTIVE
        assert bad.window_labels[16] == label
        good_steps, _ = steps_from_record(good)
        bad_steps, _ = steps_from_record(bad)
        assert good_steps[16].progress_after > good_steps[8].progress_after
        assert bad_steps[16].progress_after == bad_steps[8].progress_after
        assert gate(good_events[:17]).passed and gate(bad_events[:17]).passed


def test_telemetry_samples_several_behaviours_and_every_drop_kind(corpus):
    clean_classes = set()
    external_tools = set()
    for events, record in corpus["telemetry_gaps"]:
        steps, _ = steps_from_record(record)
        assert tuple(i for i, step in enumerate(steps) if step.dropped) == record.dropped_observations
        reasons = set()
        classes = set()
        for t in range(7, len(events)):
            decision = gate(events[:t+1])
            reasons.update(reason for _, reason in decision.reasons)
            dropped = any(step.dropped for step in steps[t-7:t+1])
            assert decision.passed == (not dropped)
            assert (record.window_labels[t] == L.INSUFFICIENT_INSTRUMENTATION) == dropped
            if decision.passed and record.window_labels[t].is_scored:
                classes.add(record.window_labels[t])
        assert reasons == set(GateReason) - {GateReason.TOO_SHORT}
        assert L.PRODUCTIVE in classes and len(classes) >= 2
        clean_classes.update(classes)
        external_tools.update(e.operation_family for e in events
                              if e.operation_family in ("http_call", "poll_status"))
        assert gate(events[:max(record.dropped_observations)+9]).passed
        for field in ("validation_suite_id", "validation_passed", "validation_total"):
            assert any(e.validation_produced and getattr(e, field) is None for e in events)
        assert gate(events[:7]).reasons == ((-1, GateReason.TOO_SHORT),)
    assert clean_classes == {L.PRODUCTIVE, L.RECOVERY, L.STALLED, L.OSCILLATING}
    assert external_tools == {"http_call", "poll_status"}


@pytest.mark.parametrize("field", ["validation_suite", "validation_passed", "validation_total"])
def test_lost_validation_baseline_marks_only_affected_observations(field):
    clean = Engine(random.Random(7), "baseline", CFG)
    lossy = Engine(random.Random(7), "baseline", CFG)
    for e in (clean, lossy):
        e.test()
        e.test(dropped=field if e is lossy else None)
        e.test()
        e.test()
        for i in range(8):
            e.perform("plan", args={"action_type": f"review:{i}"})
    assert [s.sequence_index for s in lossy.steps if s.dropped] == [1, 2]
    assert all(getattr(lossy.events[2], f) is not None for f in
               ("validation_suite_id", "validation_passed", "validation_total"))
    for a, b in zip(clean.steps, lossy.steps):
        assert {k: v for k, v in a.to_dict().items() if k != "dropped"} == {
            k: v for k, v in b.to_dict().items() if k != "dropped"}
    for t in range(7, len(lossy.events)):
        assert gate(lossy.events[:t+1]).passed == (t >= 10)
    assert clean.repo.files == lossy.repo.files
