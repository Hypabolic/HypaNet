"""Dataset build (M1_CONTRACTS §5, §7 "Datasets")."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from pathlib import Path

import pytest

from agent_loop_lab.datasets import build as B
from agent_loop_lab.schema import FORBIDDEN_INFERENCE_FIELDS, CanonicalEvent, OracleRecord
from agent_loop_lab.synth import counterfactual as cf
from agent_loop_lab.synth.families import FAMILY_NAMES

SMALL = 132  # 11 per family: 7/2/2 under the rounding rule
PAIRS = 2
SEED = 20261104


@pytest.fixture(scope="module")
def two_builds(tmp_path_factory):
    out = []
    for name in ("a", "b"):
        root = tmp_path_factory.mktemp(f"build_{name}")
        res = B.build(root / "v0_2", master_seed=SEED, sessions=SMALL, pairs_per_kind=PAIRS,
                      reports_dir=root / "reports")
        out.append((root, res))
    return out


def _all_files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_small_build_byte_identical(two_builds):
    (ra, _), (rb, _) = two_builds
    fa, fb = _all_files(ra), _all_files(rb)
    assert sorted(fa) == sorted(fb)
    expected = {f"v0_2/standard/{s}.{k}.jsonl.gz" for s in (*B.SPLITS, "counterfactual") for k in ("events", "oracle")}
    expected |= {"v0_2/MANIFEST.json", "v0_2/splits.json", "reports/m1_dataset_stats.json",
                 "reports/m1_collisions_dev.json"}
    assert set(fa) == expected
    for name in fa:
        assert fa[name] == fb[name], name


def test_gzip_header_is_deterministic(two_builds):
    (ra, _), _ = two_builds
    raw = (ra / "v0_2/standard/dev.events.jsonl.gz").read_bytes()
    assert raw[:2] == b"\x1f\x8b"
    assert raw[4:8] == b"\x00\x00\x00\x00"  # mtime 0
    assert raw[3] & 0x08 == 0  # FNAME flag unset: no filename in header


def test_no_session_shared_across_splits(two_builds):
    (ra, res), _ = two_builds
    splits = json.loads((ra / "v0_2/splits.json").read_text())["splits"]
    assert list(splits) == list(B.SPLITS)
    all_ids = [i for s in B.SPLITS for i in splits[s]]
    assert len(all_ids) == len(set(all_ids)) == SMALL
    for s in B.SPLITS:
        file_ids = [rec.session_id for _, rec in B.iter_sessions(ra / "v0_2", s)]
        assert file_ids == splits[s]
    # counterfactual sessions are separate from every standard split
    with gzip.open(ra / "v0_2/standard/counterfactual.oracle.jsonl.gz", "rt") as f:
        cf_ids = [json.loads(line)["session_id"] for line in f]
    assert len(cf_ids) == 2 * PAIRS * 3 and not set(cf_ids) & set(all_ids)


def test_manifest_hashes_match_files(two_builds):
    (ra, _), _ = two_builds
    out = ra / "v0_2"
    manifest = json.loads((out / "MANIFEST.json").read_text())
    data_files = sorted(p.relative_to(out).as_posix() for p in (out / "standard").iterdir())
    assert sorted(manifest["files"]) == data_files
    for rel, meta in manifest["files"].items():
        data = (out / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == meta["sha256"], rel
        assert len(data) == meta["bytes"]
        with gzip.open(out / rel, "rt") as f:
            assert sum(1 for _ in f) == meta["lines"]
    assert manifest["master_seed"] == SEED
    assert manifest["generator_version"]
    std = manifest["standard"]
    assert sum(c["sessions"] for c in std["splits"].values()) == SMALL
    for s in B.SPLITS:
        recs = [rec for _, rec in B.iter_sessions(out, s)]
        c = std["splits"][s]
        assert c["sessions"] == len(recs)
        assert c["windows"] == sum(len(r.window_labels) for r in recs) == sum(c["window_labels"].values())
        assert c["families"] == {f: sum(r.scenario_family == f for r in recs) for f in FAMILY_NAMES}


def test_split_proportions_per_family(two_builds):
    (ra, res), _ = two_builds
    fam_split = res.manifest["standard"]["family_split_sessions"]
    assert list(fam_split) == list(FAMILY_NAMES)
    for f in FAMILY_NAMES:
        assert fam_split[f] == {"dev": 7, "val": 2, "test": 2}
    for s in B.SPLITS:
        recs = [rec for _, rec in B.iter_sessions(ra / "v0_2", s)]
        for f in FAMILY_NAMES:
            assert sum(r.scenario_family == f for r in recs) == fam_split[f][s]


def test_events_files_carry_no_oracle_fields(two_builds):
    (ra, _), _ = two_builds
    public = {f for f in CanonicalEvent.__dataclass_fields__}
    pair_ids = {B.pair_id_for(B.counterfactual_seed(SEED, kind, k)) for kind in cf.KINDS for k in range(PAIRS)}
    for split in (*B.SPLITS, "counterfactual"):
        with gzip.open(ra / f"v0_2/standard/{split}.events.jsonl.gz", "rt") as f:
            for line in f:
                d = json.loads(line)
                assert set(d) == {"session_id", "events"}
                for e in d["events"]:
                    assert set(e) <= public
                    assert not set(e) & FORBIDDEN_INFERENCE_FIELDS
                    assert "scenario_family" not in json.dumps(e)
                    CanonicalEvent.from_dict(e)
                    # ids are opaque: no family/kind name, pair id or branch marker
                    idents = (e["session_id"], e["event_id"], e["agent_id"], e["task_id"],
                              e["provenance"]["trace_id"], *e["provenance"]["span_ids"])
                    for ident in idents:
                        assert not any(name in ident for name in (*FAMILY_NAMES, *cf.KINDS)), ident
                        assert not any(pid in ident for pid in pair_ids), ident
                        assert "-a-" not in ident and "-b-" not in ident, ident
                    assert not e["session_id"].endswith(("-a", "-b"))


def test_oracle_records_match_seed_formula(two_builds):
    (ra, _), _ = two_builds
    seen = {}
    for s in B.SPLITS:
        for events, rec in B.iter_sessions(ra / "v0_2", s):
            assert rec.session_id == B.session_id_for(rec.seed)
            seen.setdefault(rec.scenario_family, set()).add(rec.seed)
            assert events[0].session_id == rec.session_id
            assert len(rec.event_effects) == len(events)
    for f, n in B.family_counts(SMALL, FAMILY_NAMES).items():
        assert seen[f] == {B.session_seed(SEED, f, k) for k in range(n)}


def test_counterfactual_file_is_paired(two_builds):
    (ra, res), _ = two_builds
    with gzip.open(ra / "v0_2/standard/counterfactual.oracle.jsonl.gz", "rt") as f:
        recs = [OracleRecord.from_dict(json.loads(line)) for line in f]
    assert len(recs) == 2 * PAIRS * 3
    for a, b in zip(recs[::2], recs[1::2]):
        ca, cb = a.latent_trace["counterfactual"], b.latent_trace["counterfactual"]
        assert ca["pair_id"] == cb["pair_id"] and ca["kind"] == cb["kind"]
        assert {ca["branch"], cb["branch"]} == {"a", "b"}
    assert res.manifest["counterfactual"]["kinds"].keys() == {
        "retry_success_vs_doomed", "improving_vs_identical_failures", "productive_vs_abab"}


def test_counterfactual_line_order_does_not_reveal_branch(two_builds):
    """Within each pair the line order is drawn from the pair seed, not fixed a-then-b."""
    (ra, _), _ = two_builds
    with gzip.open(ra / "v0_2/standard/counterfactual.oracle.jsonl.gz", "rt") as f:
        recs = [OracleRecord.from_dict(json.loads(line)) for line in f]
    expected, got = [], []
    for kind in cf.KINDS:
        for k in range(PAIRS):
            expected.append(B.counterfactual_branch_order(B.counterfactual_seed(SEED, kind, k)))
    for first, second in zip(recs[::2], recs[1::2]):
        got.append((first.latent_trace["counterfactual"]["branch"], second.latent_trace["counterfactual"]["branch"]))
    assert got == expected
    assert ("b", "a") in got and ("a", "b") in got
    # the order rule is a pure function of the pair seed
    assert all(B.counterfactual_branch_order(s) in {("a", "b"), ("b", "a")} for s in range(64))
    assert {B.counterfactual_branch_order(s) for s in range(64)} == {("a", "b"), ("b", "a")}


def test_reports_written(two_builds):
    (ra, res), _ = two_builds
    stats = json.loads((ra / "reports/m1_dataset_stats.json").read_text())
    coll = json.loads((ra / "reports/m1_collisions_dev.json").read_text())
    assert stats["sessions"]["total"] == SMALL
    assert stats["checks"]["gate_all_pass_families_1_11"] is True
    assert stats["checks"]["gate_oracle_agree_all_families"] is True
    assert coll["audit_split"] == "dev" and "cross_split" in coll


# -- pure rules ---------------------------------------------------------------


def test_seed_formula():
    expected = int.from_bytes(hashlib.sha256(b"20261104:exploration:7").digest()[:8], "big")
    assert B.session_seed(20261104, "exploration", 7) == expected


def test_family_counts_even_with_remainder_in_order():
    c = B.family_counts(20_000, FAMILY_NAMES)
    assert sum(c.values()) == 20_000
    assert [c[f] for f in FAMILY_NAMES] == [1667] * 8 + [1666] * 4
    assert B.family_counts(13, FAMILY_NAMES)[FAMILY_NAMES[0]] == 2


@pytest.mark.parametrize("n", [0, 1, 5, 10, 11, 1666, 1667, 999])
def test_split_assignment_proportions_and_determinism(n):
    c1, c2 = B.split_cuts(n)
    assert (c1, c2) == ((6 * n + 5) // 10, (8 * n + 5) // 10)
    a = B.assign_splits(SEED, "cosmetic_churn", n)
    assert a == B.assign_splits(SEED, "cosmetic_churn", n)
    assert (a.count("dev"), a.count("val"), a.count("test")) == (c1, c2 - c1, n - c2)
    if n:
        assert abs(a.count("dev") - 0.6 * n) <= 0.5 and abs(a.count("val") - 0.2 * n) <= 1


def test_failed_build_leaves_no_partial_files(tmp_path, monkeypatch):
    from agent_loop_lab.synth import families

    real = families.generate_session

    def boom(family, seed, session_id, cfg=None):
        if family == FAMILY_NAMES[3]:
            raise ValueError("synthetic failure")
        return real(family, seed, session_id, cfg)

    monkeypatch.setattr(families, "generate_session", boom)
    out = tmp_path / "v0_2"
    with pytest.raises(B.BuildError, match="synthetic failure"):
        B.build(out, sessions=48, pairs_per_kind=0)
    assert not [p for p in out.rglob("*") if p.is_file()]


def test_split_assignment_keyed_on_master_seed_and_family():
    base = B.assign_splits(SEED, "exploration", 200)
    assert B.assign_splits(SEED + 1, "exploration", 200) != base
    assert B.assign_splits(SEED, "bounded_retry", 200) != base
    # independent of ambient RNG state
    random.seed(0)
    assert B.assign_splits(SEED, "exploration", 200) == base
