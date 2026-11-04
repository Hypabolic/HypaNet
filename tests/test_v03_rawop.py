"""v0.3 RawOp record and the generators' raw-capture side channel (spec v0.3 §3, §7.1)."""

from __future__ import annotations

import dataclasses
import difflib
import json
import random

import pytest

from agent_loop_lab.datasets.build import (
    DEFAULT_MASTER_SEED,
    counterfactual_seed,
    pair_id_for,
    session_id_for,
    session_seed,
)
from agent_loop_lab.datasets.ood import ood_seed, ood_session_id
from agent_loop_lab.fingerprint import FAMILY_CATEGORY, failure_fingerprint, intent_fingerprint
from agent_loop_lab.schema import OperationCategory, Outcome
from agent_loop_lab.synth import counterfactual as cf
from agent_loop_lab.synth.engine import Session, cosmetic, fix
from agent_loop_lab.synth.families import FAMILY_NAMES, generate_session
from agent_loop_lab.synth_ood import FAMILIES as OOD_FAMILIES
from agent_loop_lab.synth_ood.counterfactuals import counterfactual_pairs
from agent_loop_lab.v03.rawop import (
    DROPPED_VOCAB,
    FIELDS,
    OOD_DROP_MAP,
    STANDARD_DROP_MAP,
    RawOp,
    RawOpError,
    capture_raw,
    normalise_ood_drop,
    normalise_standard_drop,
)
from agent_loop_lab.v03.build_raw import alignment_errors
from agent_loop_lab.v03.renderer import render_event, render_session
from tests import synth_helpers as H

EXPECTED_FIELDS = (
    "family", "target", "args", "outcome", "error_class", "error_location", "validation", "changed",
    "extra", "input_tokens", "output_tokens", "timestamp", "dropped",
)
EXPECTED_DROPPED_VOCAB = {"input_tokens", "output_tokens", "digest_before", "digest_after", "failure",
                          "validation_suite", "validation_passed", "validation_total"}
_EXTRA_KEYS = {"failing_test", "status"}


def _dev_seeds(family: str, n: int) -> list[int]:
    return [session_seed(DEFAULT_MASTER_SEED, family, k) for k in range(n)]


def _gen(family: str, seed: int):
    sid = session_id_for(seed)
    with capture_raw() as cap:
        events, record = generate_session(family, seed, sid, H.cfg())
    raw = cap.pop(sid, events)
    assert len(cap) == 0
    return events, record, raw


def _check_against_events(events, raw, *, ood: bool = False) -> None:
    """Every RawOp reproduces its event's fingerprints and observed fields (r1 schema)."""
    assert len(raw) == len(events)
    for r, e in zip(raw, events):
        # the builder's own alignment checks (drops <-> nulls, validation, changed <-> digests)
        assert alignment_errors(r, e) == [], (e.event_id, alignment_errors(r, e))
        # target/args are exactly what intent_fingerprint was given
        assert intent_fingerprint(r.family, r.target, r.args) == e.intent_fingerprint
        d = set(r.dropped)
        # failure: present on every ERROR event, even when the canonical fingerprint was dropped
        if e.outcome is Outcome.ERROR:
            assert r.error_class is not None and r.error_location is not None
            fp = failure_fingerprint(r.family, r.error_class, r.error_location)
            if "failure" in d:
                assert e.failure_fingerprint is None
            else:
                assert e.failure_fingerprint == fp
        else:
            assert r.error_class is None and r.error_location is None and "failure" not in d
        # validation: the true values, even when dropped canonically
        if e.validation_produced:
            v = r.validation
            assert v is not None and v["suite"] is not None and v["passed"] is not None and v["total"] is not None
        else:
            assert r.validation is None
        # changed: sorted by path, every entry a real change
        ch = r.changed
        assert isinstance(ch, list)
        assert [c["path"] for c in ch] == sorted({c["path"] for c in ch})
        assert all(c["before"] != c["after"] for c in ch)
        if FAMILY_CATEGORY[r.family] is not OperationCategory.MUTATE and not ood:
            assert ch == []  # standard engine: only mutations touch files
        if r.outcome == "ERROR" and FAMILY_CATEGORY[r.family] is OperationCategory.MUTATE:
            assert ch == []  # a failed mutation changes nothing
        if r.family == "write_file" and e.outcome is Outcome.SUCCESS and ch:
            assert [c["path"] for c in ch] == [r.target] and ch[0]["after"] == r.args["content"]
        elif r.family == "edit_file" and e.outcome is Outcome.SUCCESS:
            path = r.target
            if ch:
                assert [c["path"] for c in ch] == [path]
                prev, new = ch[0]["before"], ch[0]["after"]
                assert prev is not None and new is not None
            else:
                prev = new = None
            patch = "" if not ch else "\n".join(difflib.unified_diff(prev.split("\n"), new.split("\n"),
                                                                    f"a/{path}", f"b/{path}", lineterm="", n=0))
            assert (patch or f"--- a/{path}\n+++ b/{path}\n@@ no change @@") == r.args["patch"]
        elif r.family == "revert":
            assert {c["path"] for c in ch} <= set(r.target) and "revision" in r.args
        if r.extra is not None:
            assert set(r.extra) <= _EXTRA_KEYS
        # JSON round trip
        assert RawOp.from_dict(json.loads(json.dumps(r.to_dict()))) == r


def _no_latent_strings(record, raw) -> None:
    blob = json.dumps([r.to_dict() for r in raw])
    for f in record.latent_trace["world"]["faults"]:
        for fact in f["facts"]:
            assert json.dumps(fact) not in blob, fact
        assert f"fix:{f['id']}" not in blob and f"competing:{f['id']}" not in blob
    assert record.scenario_family not in {r.family for r in raw}


# -- record ---------------------------------------------------------------------


def test_field_whitelist():
    assert FIELDS == EXPECTED_FIELDS
    forbidden = ("effect", "progress", "episode", "scenario", "latent", "oracle", "label", "fact", "fault",
                 "seed", "session", "attempt", "redundant", "window")
    for name in FIELDS:
        assert not any(w in name for w in forbidden), name


def test_round_trip_and_strict_keys():
    r = RawOp.make(family="run_tests", target="unit", args={"tests": ("a", "b")}, outcome=Outcome.ERROR,
                   error_class="AssertionError", error_location="t::a", validation=("unit", 3, 5),
                   extra={"failing_test": "t::a"}, input_tokens=10, output_tokens=None,
                   timestamp="2026-01-01T00:00:00.000Z", dropped=("output_tokens",))
    d = r.to_dict()
    assert list(d) == list(EXPECTED_FIELDS)
    assert d["args"] == {"tests": ["a", "b"]} and d["outcome"] == "ERROR"
    assert d["validation"] == {"suite": "unit", "passed": 3, "total": 5} and d["changed"] == []
    s = json.dumps(d, sort_keys=True)
    assert RawOp.from_dict(json.loads(s)) == r
    with pytest.raises(RawOpError):
        RawOp.from_dict({**d, "effect": "PROGRESS"})
    d2 = dict(d)
    del d2["timestamp"]
    with pytest.raises(RawOpError):
        RawOp.from_dict(d2)
    for legacy in ("workspace", "tokens", "validation_counts"):
        with pytest.raises(RawOpError):
            RawOp.from_dict({**d, "dropped": [legacy]})
    with pytest.raises(RawOpError):
        RawOp.from_dict({**d, "prev_text": None})
    with pytest.raises(RawOpError):
        RawOp.from_dict({**d, "changed": [{"path": "a", "before": None}]})
    with pytest.raises(RawOpError):
        RawOp.from_dict({**d, "dropped": ["output_tokens", "input_tokens"]})  # not sorted
    with pytest.raises(RawOpError):
        RawOp.make(family="x", target=None, args={"k": {1, 2}}, outcome="SUCCESS")


def test_make_copies_args():
    args = {"tests": ["a"]}
    r = RawOp.make(family="run_tests", target="u", args=args, outcome="SUCCESS")
    args["tests"].append("b")
    assert r.args == {"tests": ["a"]}


# -- dropped normalisation -----------------------------------------------------


def test_standard_drop_mapping():
    assert normalise_standard_drop(()) == ()
    assert normalise_standard_drop(("tokens",)) == ("input_tokens", "output_tokens")
    assert normalise_standard_drop(("input_tokens",)) == ("input_tokens",)
    assert normalise_standard_drop(("output_tokens",)) == ("output_tokens",)
    assert normalise_standard_drop(("output_tokens", "input_tokens")) == ("input_tokens", "output_tokens")
    assert normalise_standard_drop(("digests",)) == ("digest_after", "digest_before")
    assert normalise_standard_drop(("digest_before",)) == ("digest_before",)
    assert normalise_standard_drop(("digest_after",)) == ("digest_after",)
    assert normalise_standard_drop(("failure",)) == ("failure",)
    assert normalise_standard_drop(("validation_counts", "validation_suite")) == (
        "validation_passed", "validation_suite", "validation_total")
    assert set(STANDARD_DROP_MAP) == {"tokens", "input_tokens", "output_tokens", "digests", "digest_before",
                                      "digest_after", "failure", "validation_counts", "validation_suite"}
    with pytest.raises(RawOpError):
        normalise_standard_drop(("workspace",))


def test_ood_drop_mapping():
    kw = dict(is_error=True, produced=True)
    assert normalise_ood_drop(None, **kw) == ()
    assert normalise_ood_drop("tokens", **kw) == ("input_tokens",)  # the OOD engine nulls input_tokens only
    assert normalise_ood_drop("workspace", **kw) == ("digest_after", "digest_before")
    assert normalise_ood_drop("failure", **kw) == ("failure",)
    assert normalise_ood_drop("validation_suite", **kw) == ("validation_suite",)
    assert normalise_ood_drop("validation_passed", **kw) == ("validation_passed",)
    assert normalise_ood_drop("validation_total", **kw) == ("validation_total",)
    assert set(OOD_DROP_MAP) == {"tokens", "workspace", "failure", "validation_suite", "validation_passed",
                                 "validation_total"}
    # nothing withheld -> nothing recorded
    assert normalise_ood_drop("failure", is_error=False, produced=True) == ()
    assert normalise_ood_drop("validation_total", is_error=False, produced=False) == ()
    with pytest.raises(RawOpError):
        normalise_ood_drop("digests", **kw)


def test_dropped_vocab():
    assert set(DROPPED_VOCAB) == EXPECTED_DROPPED_VOCAB and list(DROPPED_VOCAB) == sorted(DROPPED_VOCAB)
    for table in (STANDARD_DROP_MAP, OOD_DROP_MAP):
        assert set().union(*table.values()) <= EXPECTED_DROPPED_VOCAB


# -- standard engine capture ------------------------------------------------------


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_standard_capture(family):
    for seed in _dev_seeds(family, 3):
        events, record, raw = _gen(family, seed)
        _check_against_events(events, raw)
        _no_latent_strings(record, raw)


def test_capture_is_observe_only():
    """Generating with and without an open sink gives byte-identical events and records."""
    for family in FAMILY_NAMES:
        seed = _dev_seeds(family, 1)[0]
        ev1, rec1 = generate_session(family, seed, session_id_for(seed), H.cfg())
        ev2, rec2, _ = _gen(family, seed)
        assert [e.to_dict() for e in ev1] == [e.to_dict() for e in ev2]
        assert json.dumps(rec1.to_dict(), sort_keys=True) == json.dumps(rec2.to_dict(), sort_keys=True)


def test_no_sink_no_capture_leak():
    seed = _dev_seeds("exploration", 1)[0]
    generate_session("exploration", seed, session_id_for(seed), H.cfg())  # no sink open: must not fail
    with capture_raw() as cap:
        pass
    assert len(cap) == 0


def test_dropped_fields_under_instrumented():
    seen: set[str] = set()
    for seed in _dev_seeds("under_instrumented", 40):
        events, record, raw = _gen("under_instrumented", seed)
        _check_against_events(events, raw)
        for r in raw:
            seen.update(r.dropped)
        deliberate = {i for i, _ in record.latent_trace["drops"]["fields"]}
        assert {e.sequence_index for e, r in zip(events, raw) if r.dropped} == deliberate
    assert seen == set(DROPPED_VOCAB)


def test_truncated_session():
    found = 0
    for seed in _dev_seeds("under_instrumented", 80):
        events, record, raw = _gen("under_instrumented", seed)
        m = record.latent_trace["drops"]["truncated_before"]
        if not m:
            continue
        found += 1
        assert events[0].sequence_index == m
        _check_against_events(events, raw)
    assert found > 0


def test_session_truncate_slices_raw():
    rng = random.Random(7)
    s = Session(rng, "s-trunc", H.cfg(), family="t", seed=7, n_faults=1)
    f = next(iter(s.world.faults.values()))
    for _ in range(4):
        s.novel_decoy()
    s.run_tests(f.suite)
    s.read(f.path, block=f.block)
    s.edit(f.path, cosmetic("comment"), block=f.block)
    s.write(f.path, fix(f))
    s.run_tests(f.suite)
    s.write("src/new_module.py")
    s.revert("src/new_module.py")
    while s.n < 18:
        s.novel_decoy()
    full = s.raw_ops
    assert len(full) == s.n
    with capture_raw() as cap:
        events, _ = s.finish(truncate_before=5)
    raw = cap.pop("s-trunc", events)
    assert raw == full[5:] == s.raw_ops
    _check_against_events(events, raw)
    create = next(r for r in full if r.family == "write_file" and r.target == "src/new_module.py")
    assert len(create.changed) == 1 and create.changed[0]["before"] is None
    assert create.changed[0]["after"] == create.args["content"]
    rev = next(r for r in full if r.family == "revert")
    assert rev.changed == [{"path": "src/new_module.py", "before": create.args["content"], "after": None}]
    assert "revision" in rev.args


@pytest.mark.parametrize("kind", cf.KINDS)
def test_counterfactual_fork(kind):
    for k in range(3):
        seed = counterfactual_seed(DEFAULT_MASTER_SEED, kind, k)
        pid = pair_id_for(seed)
        bp = cf.branch_point(kind, random.Random(seed), pid, H.cfg(), seed=seed)
        assert [r.to_dict() for r in bp.a.raw_ops] == [r.to_dict() for r in bp.b.raw_ops]
        assert len(bp.a.raw_ops) == bp.branch_index
        with capture_raw() as cap:
            (ev_a, rec_a), (ev_b, rec_b) = cf.generate_pair_from_seed(kind, seed, pid, H.cfg())
        raw_a, raw_b = cap.pop(rec_a.session_id, ev_a), cap.pop(rec_b.session_id, ev_b)
        assert len(cap) == 0
        _check_against_events(ev_a, raw_a)
        _check_against_events(ev_b, raw_b)
        bi = rec_a.latent_trace["counterfactual"]["branch_index"]
        assert [r.to_dict() for r in raw_a[:bi]] == [r.to_dict() for r in raw_b[:bi]]


# -- OOD engine capture -----------------------------------------------------------


@pytest.mark.parametrize("family", list(OOD_FAMILIES))
def test_ood_capture(family):
    for k in range(3):
        seed = ood_seed(DEFAULT_MASTER_SEED, family, k)
        sid = ood_session_id(seed)
        with capture_raw() as cap:
            events, record = OOD_FAMILIES[family](random.Random(seed), sid, H.cfg())
        raw = cap.pop(sid, events)
        _check_against_events(events, raw, ood=True)
        for r in raw:
            if r.family == "write_file" and r.changed:
                assert r.changed == [{"path": r.target, "before": r.changed[0]["before"],
                                      "after": r.args["content"]}]
            if r.family == "poll_status":
                assert r.extra == {"status": "pending" if r.outcome == "NEUTRAL" else "ready"}


def test_ood_capture_is_observe_only():
    for family, gen in OOD_FAMILIES.items():
        seed = ood_seed(DEFAULT_MASTER_SEED, family, 0)
        ev1, rec1 = gen(random.Random(seed), "o-x", H.cfg())
        with capture_raw():
            ev2, rec2 = gen(random.Random(seed), "o-x", H.cfg())
        assert [e.to_dict() for e in ev1] == [e.to_dict() for e in ev2]
        assert json.dumps(rec1.to_dict(), sort_keys=True) == json.dumps(rec2.to_dict(), sort_keys=True)


def test_ood_telemetry_gaps_dropped():
    seed = ood_seed(DEFAULT_MASTER_SEED, "telemetry_gaps", 0)
    with capture_raw() as cap:
        events, _ = OOD_FAMILIES["telemetry_gaps"](random.Random(seed), "o-t", H.cfg())
    raw = cap.pop("o-t", events)
    _check_against_events(events, raw, ood=True)
    drops = [r.dropped for r in raw if r.dropped]
    assert ("input_tokens",) in drops and ("digest_after", "digest_before") in drops
    assert ("validation_suite",) in drops
    assert ("validation_passed",) in drops and ("validation_total",) in drops
    tok = next(r for r in raw if r.dropped == ("input_tokens",))
    assert tok.input_tokens is None and tok.output_tokens is not None  # OOD nulls input only


def test_ood_counterfactual_pairs_share_ids():
    with capture_raw() as cap:
        pairs = counterfactual_pairs(random.Random(11), "o-cf", H.cfg())
    for (ev_a, rec_a), (ev_b, rec_b) in pairs.values():
        assert rec_a.session_id == rec_b.session_id  # OOD branches share ids: matched by events identity
        raw_a, raw_b = cap.pop(rec_a.session_id, ev_a), cap.pop(rec_b.session_id, ev_b)
        _check_against_events(ev_a, raw_a, ood=True)
        _check_against_events(ev_b, raw_b, ood=True)
        n = 0
        while n < min(len(ev_a), len(ev_b)) and ev_a[n] == ev_b[n]:
            n += 1
        assert n >= 9 and [r.to_dict() for r in raw_a[:n]] == [r.to_dict() for r in raw_b[:n]]
    assert len(cap) == 0


def test_rawop_is_frozen():
    r = RawOp.make(family="plan", target=None, args={"action_type": "x"}, outcome="SUCCESS")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.family = "ask"  # type: ignore[misc]


# -- builder (small, unverified) -------------------------------------------------


def test_build_raw_small(tmp_path):
    from agent_loop_lab.datasets.build import iter_sessions
    from agent_loop_lab.datasets.ood import iter_ood_sessions
    from agent_loop_lab.v03.build_raw import RAW_SPLITS, build_raw, iter_raw, raw_path

    regen, out = tmp_path / "regen", tmp_path / "v0_3"
    m = build_raw(out, v02_dir=tmp_path / "unused", tmp_dir=regen, sessions=24, pairs_per_kind=1,
                  ood_sessions_per_family=1, verify=False)
    assert all(v == 0 for v in m["unclaimed_finished_sessions"].values())
    assert sum(m["sessions"][s] for s in ("dev", "val", "test")) == 24
    assert m["sessions"]["counterfactual"] == 2 * len(cf.KINDS) and m["sessions"]["ood"] == len(OOD_FAMILIES)
    for split in RAW_SPLITS:
        assert raw_path(out, split).exists()
        canon = iter_ood_sessions(regen) if split == "ood" else iter_sessions(regen, split)
        pairs = list(zip(iter_raw(out, split), canon, strict=True))
        assert len(pairs) == m["sessions"][split]
        for (sid, raw), (events, record) in pairs:
            assert sid == record.session_id
            _check_against_events(events, raw, ood=split == "ood")
    assert not (out / "MANIFEST_v0_3.json").exists()  # unverified builds write no manifest


# -- rendering engine-produced RawOps (spec v0.3 §4.3-§4.4) ------------------------


def _hunk_lines(block: str) -> list[str]:
    return [ln for ln in block.split("\n")[1:] if ln.startswith("  @ ")]


def _first_std(family: str, pred, n: int = 40):
    for seed in _dev_seeds(family, n):
        events, record, raw = _gen(family, seed)
        for r in raw:
            if pred(r):
                return record.session_id, r
    raise AssertionError(f"no matching RawOp in {n} {family} sessions")


def _ood_ops(family: str, n: int = 5):
    for k in range(n):
        seed = ood_seed(DEFAULT_MASTER_SEED, family, k)
        sid = ood_session_id(seed)
        with capture_raw() as cap:
            events, _ = OOD_FAMILIES[family](random.Random(seed), sid, H.cfg())
        yield sid, cap.pop(sid, events)


def _first_ood(family: str, pred):
    for sid, raw in _ood_ops(family):
        for r in raw:
            if pred(r):
                return sid, r
    raise AssertionError(f"no matching RawOp in {family}")


def test_render_edit_has_hunk_line():
    sid, r = _first_std("cosmetic_churn", lambda r: r.family == "edit_file" and r.changed)
    hunks = _hunk_lines(render_event(r.to_dict(), sid))
    assert hunks and all(h.split()[1] == r.target for h in hunks if "TRUNC" not in h)


def test_render_revert_has_hunk_line_and_no_digest():
    sid, r = _first_std("tool_oscillation", lambda r: r.family == "revert" and r.changed)
    block = render_event(r.to_dict(), sid)
    assert _hunk_lines(block)
    header = block.split("\n")[0]
    assert " revert paths=" in header and "to=" not in header and r.args["revision"] not in block


def test_render_ood_install_has_hunk_line():
    def is_install(r):
        return (r.family == "http_call" and r.target["path_template"] == "/install/cedar" and r.changed)
    sid, r = _first_ood("lockfile_churn", is_install)
    hunks = _hunk_lines(render_event(r.to_dict(), sid))
    assert hunks and hunks[0].split()[1] == "vendor/cedar.artifact"


def _header(r, sid) -> str:
    return render_event(r.to_dict(), sid).split("\n")[0]


def test_render_standard_validation_counts_drop():
    sid, r = _first_std("under_instrumented", lambda r: r.validation is not None
                        and {"validation_passed", "validation_total"} <= set(r.dropped), n=80)
    assert " ?/? passed " in _header(r, sid)


def test_render_ood_validation_partial_drops():
    sid, r = _first_ood("telemetry_gaps", lambda r: r.dropped == ("validation_passed",))
    assert f" ?/{r.validation['total']} passed " in _header(r, sid)
    sid, r = _first_ood("telemetry_gaps", lambda r: r.dropped == ("validation_total",))
    assert f" {r.validation['passed']}/? passed " in _header(r, sid)


def test_render_tokens_drop_standard_and_ood():
    sid, r = _first_std("under_instrumented", lambda r: {"input_tokens", "output_tokens"} <= set(r.dropped),
                        n=80)
    assert _header(r, sid).endswith(" in=? out=?")
    sid, r = _first_ood("telemetry_gaps", lambda r: r.dropped == ("input_tokens",))
    assert r.output_tokens is not None and _header(r, sid).endswith(f" in=? out={r.output_tokens}")


def test_render_engine_session_deterministic():
    seed = _dev_seeds("abab_oscillation", 1)[0]
    _, record, raw = _gen("abab_oscillation", seed)
    dicts = [r.to_dict() for r in raw]
    assert render_session(dicts, record.session_id) == render_session(json.loads(json.dumps(dicts)),
                                                                      record.session_id)
