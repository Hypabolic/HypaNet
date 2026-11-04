"""OOD holdout dataset build (spec §7.3): determinism, disjointness, manifest integrity, no leakage."""

from __future__ import annotations

import gzip
import json

from agent_loop_lab.datasets.build import file_sha256
from agent_loop_lab.datasets.ood import build_ood, iter_ood_sessions, ood_paths
from agent_loop_lab.schema import FORBIDDEN_INFERENCE_FIELDS


def _build(tmp_path, name):
    out = tmp_path / name
    return out, build_ood(out, master_seed=7, sessions_per_family=2)


def test_ood_build_is_byte_identical(tmp_path):
    a, ma = _build(tmp_path, "a")
    b, mb = _build(tmp_path, "b")
    assert ma == mb
    for pa, pb in zip(ood_paths(a), ood_paths(b)):
        assert pa.read_bytes() == pb.read_bytes()


def test_ood_manifest_hashes_and_ids(tmp_path):
    out, m = _build(tmp_path, "x")
    for rel, h in m["files"].items():
        assert file_sha256(out / rel) == h
    sessions = list(iter_ood_sessions(out))
    assert len(sessions) == m["sessions"] == 2 * len(m["families"])
    ids = [rec.session_id for _, rec in sessions]
    assert len(set(ids)) == len(ids) and all(i.startswith("o-") for i in ids)
    for events, rec in sessions:
        assert all(e.session_id == rec.session_id for e in events)
        assert rec.generator_version == "ood-0.1" and rec.scenario_family.startswith("ood:")


def test_ood_events_file_has_no_oracle_fields(tmp_path):
    out, _ = _build(tmp_path, "y")
    with gzip.open(ood_paths(out)[0], "rt") as f:
        for line in f:
            d = json.loads(line)
            assert set(d) == {"session_id", "events"}
            for e in d["events"]:
                assert not (set(e) & FORBIDDEN_INFERENCE_FIELDS)
