"""v0.4 fresh sealed test: plan, disjointness, deterministic build, cleanup on failure, seal gate."""

from __future__ import annotations

import gzip
import json

import pytest

from agent_loop_lab.v04 import fresh_test as F


def test_plan_order_and_counts():
    plan = F.fresh_plan(7, 2)
    assert len(plan) == 24
    fams = [p[0] for p in plan]
    assert fams[0] == fams[1] and fams[1] != fams[2]  # family order, then k
    assert [p[1] for p in plan[:2]] == [0, 1]
    assert all(p[3] == f"s-{p[2]:016x}" for p in plan)


def test_registered_seed_is_disjoint_from_v02():
    plan = F.fresh_plan()
    assert len(plan) == 3996
    F.check_disjoint(plan, F.existing_session_ids())


def test_collision_refused():
    plan = F.fresh_plan(7, 1)
    with pytest.raises(F.FreshTestError, match="already exist"):
        F.check_disjoint(plan, {plan[3][3]})


def test_generator_unchanged():
    g = F.check_generator()
    assert g["generator_version"] == "0.2.0"


def test_build_deterministic_and_aligned(tmp_path):
    a = F.build_fresh_test(tmp_path / "a", master_seed=7, per_family=2)
    b = F.build_fresh_test(tmp_path / "b", master_seed=7, per_family=2)
    assert a == b
    assert a["sessions"] == 24 and set(a["family_sessions"].values()) == {2}
    assert {m["lines"] for m in a["files"].values()} == {24}
    with gzip.open(tmp_path / "a" / "test.raw.jsonl.gz", "rt") as fr, \
            gzip.open(tmp_path / "a" / "standard" / "test.events.jsonl.gz", "rt") as fe:
        for lr, le in zip(fr, fe, strict=True):
            r, e = json.loads(lr), json.loads(le)
            assert r["session_id"] == e["session_id"] and len(r["raw"]) == len(e["events"])
    assert "label" not in json.dumps({k: v for k, v in a.items() if k != "identity"}).lower()


def test_failure_removes_outputs(tmp_path, monkeypatch):
    from agent_loop_lab.synth import families

    real = families.generate_session
    calls = {"n": 0}

    def flaky(*args, **kw):
        calls["n"] += 1
        if calls["n"] == 5:
            raise RuntimeError("boom")
        return real(*args, **kw)

    monkeypatch.setattr(families, "generate_session", flaky)
    with pytest.raises(Exception, match="boom"):
        F.build_fresh_test(tmp_path, master_seed=7, per_family=1)
    assert not list(tmp_path.rglob("*.jsonl.gz"))


def test_verify_files_detects_change(tmp_path):
    built = F.build_fresh_test(tmp_path, master_seed=7, per_family=1)
    F.verify_files(built, tmp_path)
    p = tmp_path / "test.raw.jsonl.gz"
    p.write_bytes(p.read_bytes() + b"\0")
    with pytest.raises(F.FreshTestError, match="sha256"):
        F.verify_files(built, tmp_path)


def test_content_sha256_is_the_uncompressed_stream(tmp_path):
    import hashlib

    built = F.build_fresh_test(tmp_path, master_seed=7, per_family=1)
    for rel, meta in built["files"].items():
        with gzip.open(tmp_path / rel, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == meta["content_sha256"]
