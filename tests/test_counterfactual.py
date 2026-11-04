"""Counterfactual pairs (M1_CONTRACTS §4, §7 "Counterfactuals")."""

from __future__ import annotations

import collections
import functools
import json
import random

import numpy as np
import pytest

from agent_loop_lab.schema import EVENTS_PER_WINDOW, OracleLabel
from agent_loop_lab.synth.counterfactual import (
    INTENDED,
    KINDS,
    branch_point,
    generate_pair,
    generate_pair_from_seed,
    latent_snapshot,
    normalise_identity,
    post_branch_labels,
)
from agent_loop_lab.synth.latent import steps_from_record
from agent_loop_lab.windows import encode_session
from tests import synth_helpers as H

N = H.N_SEEDS
MIN_DIRECTION = 45  # of 50 seeds per kind
CF_KEYS = {"pair_id", "kind", "branch", "branch_index", "variable"}


def _seeds(kind: str) -> list[int]:
    return H.seeds(f"counterfactual:{kind}", N)


@functools.lru_cache(maxsize=None)
def _pairs(kind: str) -> tuple:
    return tuple(generate_pair_from_seed(kind, sd, f"cf-{kind}-{k:03d}", H.cfg()) for k, sd in enumerate(_seeds(kind)))


def _flatten(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _flatten(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _flatten(v, f"{path}[{i}]")
    else:
        yield path, x


def _diff(x, y) -> list[str]:
    fx, fy = dict(_flatten(x)), dict(_flatten(y))
    return sorted(p for p in set(fx) | set(fy) if fx.get(p, object()) != fy.get(p, object()))


def _record_json(rec) -> str:
    return json.dumps(rec.to_dict(), sort_keys=True, separators=(",", ":"))


# -- contract -------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_counterfactual_metadata(kind):
    for (ea, ra), (eb, rb) in _pairs(kind):
        ca, cb = ra.latent_trace["counterfactual"], rb.latent_trace["counterfactual"]
        assert CF_KEYS <= set(ca) and CF_KEYS <= set(cb)
        assert (ca["branch"], cb["branch"]) == ("a", "b")
        for key in ("pair_id", "kind", "branch_index", "variable", "contrast_endpoints"):
            assert ca[key] == cb[key]
        assert ca["kind"] == kind and ra.scenario_family == rb.scenario_family == kind
        assert ca["value"] != cb["value"]
        assert ra.session_id != rb.session_id and ra.seed == rb.seed != 0
        assert all(e.session_id == ra.session_id for e in ea) and all(e.session_id == rb.session_id for e in eb)
        bi = ca["branch_index"]
        assert 0 < bi < min(len(ea), len(eb))


def test_unknown_kind():
    with pytest.raises(KeyError):
        generate_pair("nope", random.Random(0), "x", H.cfg())


# -- identical prefix -------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_identical_prefix(kind):
    for (ea, ra), (eb, rb) in _pairs(kind):
        bi = ra.latent_trace["counterfactual"]["branch_index"]
        # Events: byte-identical once b's session identity (re-stamped by fork) is mapped to a's.
        assert [normalise_identity(e_b, e_a).to_json() for e_a, e_b in zip(ea[:bi], eb[:bi])] == \
               [e.to_json() for e in ea[:bi]]
        assert all(e_a.agent_id == e_b.agent_id and e_a.task_id == e_b.task_id for e_a, e_b in zip(ea, eb))
        # Latent steps: byte-identical.
        sa, pa = steps_from_record(ra)
        sb, pb = steps_from_record(rb)
        assert pa == pb
        assert [json.dumps(s.to_dict(), sort_keys=True) for s in sa[:bi]] == \
               [json.dumps(s.to_dict(), sort_keys=True) for s in sb[:bi]]
        assert ra.event_effects[:bi] == rb.event_effects[:bi]
        # Windows wholly inside the prefix: same labels and byte-identical encoded features.
        pre = [t for t in ra.window_labels if t < bi]
        assert {t: ra.window_labels[t] for t in pre} == {t: rb.window_labels[t] for t in pre}
        if bi >= EVENTS_PER_WINDOW:
            epa, xa = encode_session(ea[:bi])
            epb, xb = encode_session(eb[:bi])
            assert epa == epb and np.asarray(xa, dtype=np.float32).tobytes() == np.asarray(xb, dtype=np.float32).tobytes()
        # ... and the two sessions really do diverge after the branch.
        assert [s.to_dict() for s in sa[bi:]] != [s.to_dict() for s in sb[bi:]]


# -- exactly one latent variable ------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_exactly_one_latent_variable_differs(kind):
    for k, sd in enumerate(_seeds(kind)[:20]):
        bp = branch_point(kind, random.Random(sd), f"cf-{k}", H.cfg(), seed=sd)
        assert bp.a.n == bp.b.n == bp.branch_index
        state_a = {"session": latent_snapshot(bp.a), "policy": bp.policy_a}
        state_b = {"session": latent_snapshot(bp.b), "policy": bp.policy_b}
        diff = _diff(state_a, state_b)
        assert len(diff) == 1, (kind, sd, diff)
        if kind == "retry_success_vs_doomed":  # world variable: the service's persistence
            host = bp.ctx["host"]
            assert diff == [f"/session/world/services/{json.dumps(host)}/mode"]
            assert bp.a.world.services[host].mode == bp.value_a == "transient"
            assert bp.b.world.services[host].mode == bp.value_b == "persistent"
            assert bp.policy_a == bp.policy_b
        else:  # policy variable: the agent's next patch
            assert diff == [f"/policy/{bp.variable}"]
            assert bp.policy_a[bp.variable] == bp.value_a and bp.policy_b[bp.variable] == bp.value_b


def test_branch_point_matches_generate_pair():
    for kind in KINDS:
        sd = _seeds(kind)[0]
        (ea, ra), (eb, rb) = generate_pair_from_seed(kind, sd, "p", H.cfg())
        bp = branch_point(kind, random.Random(sd), "p", H.cfg(), seed=sd)
        assert bp.branch_index == ra.latent_trace["counterfactual"]["branch_index"]
        assert [e.to_json() for e in bp.a.events] == [e.to_json() for e in ea[: bp.branch_index]]
        assert [e.to_json() for e in bp.b.events] == [e.to_json() for e in eb[: bp.branch_index]]


# -- labels after the branch --------------------------------------------------------


def _direction_ok(kind, ra, rb) -> bool:
    benign, bad = INTENDED[kind]
    la, lb = post_branch_labels(ra).values(), post_branch_labels(rb).values()
    return any(v in benign for v in la) and bad not in la and bad in lb


@pytest.mark.parametrize("kind", KINDS)
def test_labels_differ_in_intended_direction(kind):
    benign, bad = INTENDED[kind]
    ok = contrast = 0
    mix = {"a": collections.Counter(), "b": collections.Counter()}
    for (ea, ra), (eb, rb) in _pairs(kind):
        ok += _direction_ok(kind, ra, rb)
        cf = ra.latent_trace["counterfactual"]
        contrast += bool(cf["contrast_endpoints"])
        for t in cf["contrast_endpoints"]:
            assert ra.window_labels[t] in benign and rb.window_labels[t] is bad
        for br, rec in (("a", ra), ("b", rb)):
            mix[br].update(v.value for v in post_branch_labels(rec).values() if v.is_scored)
    assert ok >= MIN_DIRECTION, f"{kind}: direction held for {ok}/{N}; mix {mix}"
    assert contrast >= N // 2, f"{kind}: contrast endpoints in {contrast}/{N}"  # measured ~66-100%
    # Pooled post-branch scored labels: a is dominated by the benign labels, b's degenerate label beats a's count.
    assert mix["a"][bad.value] == 0 or mix["a"][bad.value] * 20 < sum(mix["a"].values())
    assert mix["b"][bad.value] > mix["a"][bad.value]


# -- determinism and validity ---------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_deterministic(kind):
    for sd in _seeds(kind)[:4]:
        p1 = generate_pair_from_seed(kind, sd, "d", H.cfg())
        p2 = generate_pair_from_seed(kind, sd, "d", H.cfg())
        for (e1, r1), (e2, r2) in zip(p1, p2):
            assert [e.to_json() for e in e1] == [e.to_json() for e in e2]
            assert _record_json(r1) == _record_json(r2)


@pytest.mark.parametrize("kind", KINDS)
def test_events_and_records_valid(kind):
    for k, ((ea, ra), (eb, rb)) in enumerate(_pairs(kind)):
        sd = _seeds(kind)[k]
        for ev, rec in ((ea, ra), (eb, rb)):
            H.assert_events_valid(ev)
            H.assert_no_oracle_fields(ev)
            H.assert_record_consistent(ev, rec, kind, seed=sd)
            H.assert_digests_match_latent(ev, rec)
            H.assert_gate_all_pass(ev, rec)
            assert OracleLabel.INSUFFICIENT_INSTRUMENTATION not in rec.window_labels.values()


# -- identity hygiene (spec §3.1: no scenario ids / future outcomes on events) --------


@pytest.mark.parametrize("kind", KINDS)
def test_session_ids_do_not_reveal_branch_kind_or_pair(kind):
    import re

    for (ea, ra), (eb, rb) in _pairs(kind):
        pair_id = ra.latent_trace["counterfactual"]["pair_id"]
        for ev, rec in ((ea, ra), (eb, rb)):
            sid = rec.session_id
            assert re.fullmatch(r"cf-[0-9a-f]{16}", sid), sid
            assert pair_id not in sid and kind not in sid
            assert not sid.endswith(("-a", "-b"))
            for e in ev:
                assert e.session_id == sid
                assert e.event_id == f"{sid}-e{e.sequence_index:03d}"
                for ident in (e.event_id, e.provenance.trace_id, *e.provenance.span_ids):
                    assert pair_id not in ident and kind not in ident
                    assert "-a-" not in ident and "-b-" not in ident
        assert ra.session_id != rb.session_id
