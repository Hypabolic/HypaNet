"""v0.4 BUDGETS.json: format, sizes, disjointness, the v0.3 rules, determinism, the v0.3 disclosure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_loop_lab.baselines import phase_c_v03 as pc
from agent_loop_lab.v03 import sealed as v03_sealed
from agent_loop_lab.v04 import budgets as B
from agent_loop_lab.v04 import export, registry

needs_val = pytest.mark.skipif(not export.labels_path("val").is_file(), reason="needs scripts/v04_export_tasks.py")


@pytest.fixture(scope="module")
def doc():
    return B.load(B.BUDGETS_PATH)


@pytest.fixture(scope="module")
def ids():
    return B.split_ids()


def test_committed_file_format(doc):
    assert set(doc) == {"format", "budgets", "es_val", "disclosed_200w", "wspf_holdout", "sha256"}
    assert doc["format"] == "v04-budgets/1"
    assert list(doc["budgets"]) == ["128", "2048", "32", "512", "all"]  # sorted keys, as rendered
    assert all(set(w) == {"session_id", "row_pos"} and isinstance(w["row_pos"], int) for w in doc["disclosed_200w"])
    for name in B.LIST_NAMES:
        assert doc["sha256"][name] == B.canonical_sha256(B._list(doc, name))


def test_no_label_in_file():
    text = B.BUDGETS_PATH.read_text()
    for word in ("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING", "INSUFFICIENT", "label", "family"):
        assert word not in text


def test_sizes(doc, ids):
    dev, _ = ids
    assert len(dev) == 12000
    assert {k: len(v) for k, v in doc["budgets"].items()} == {"32": 32, "128": 128, "512": 512, "2048": 2048,
                                                              "all": 12000}
    assert len(doc["es_val"]) == 500 and len(doc["disclosed_200w"]) == 200 and len(doc["wspf_holdout"]) == 500


def test_budgets_are_v03_phase_c_subsets(doc, ids):
    dev, _ = ids
    for name in B.BUDGET_NAMES:
        assert doc["budgets"][name] == pc.budget_subset(dev, "all" if name == "all" else int(name))
        assert doc["budgets"]["all"][: len(doc["budgets"][name])] == doc["budgets"][name]
    assert sorted(doc["budgets"]["all"]) == sorted(dev)


def test_v03_training_logs_reproduced(doc):
    for name in B.BUDGET_NAMES:
        log = json.loads((B.V03_PHASE_C_LOGS / name / "seed0" / "training_log.json").read_text())
        assert log["budget_sessions_sha256"] == pc.ids_sha256(doc["budgets"][name])
        assert log["es_set"]["sha256_of_ids_in_order"] == pc.ids_sha256(doc["es_val"])


def test_disjointness(doc, ids):
    dev, val = set(ids[0]), set(ids[1])
    hold, es = set(doc["wspf_holdout"]), set(doc["es_val"])
    disclosed = {w["session_id"] for w in doc["disclosed_200w"]}
    assert hold <= dev and es <= val and disclosed <= val and len(disclosed) == 200
    for name in ("32", "128", "512", "2048"):
        assert not hold & set(doc["budgets"][name])
    assert hold <= set(doc["budgets"]["all"])  # 'all' is every dev session: no holdout can avoid it
    assert not es & disclosed


def test_holdout_rule(doc, ids):
    dev, _ = ids
    excl = set(pc.budget_subset(dev, 2048))
    want = [s for s in sorted(dev, key=B.holdout_key) if s not in excl][:500]
    assert doc["wspf_holdout"] == want
    assert B.holdout_key("s-x") == __import__("hashlib").sha256(b"20261105:v04-wsp-f-holdout:s-x").hexdigest()


def test_es_set_rule(doc, ids):
    _, val = ids
    disclosed = {w["session_id"] for w in doc["disclosed_200w"]}
    assert doc["es_val"] == pc.es_session_ids(val, disclosed)


def test_disclosure_equals_v03_private(doc):
    rec = json.loads(v03_sealed.PRIVATE_DISCLOSURE.read_text())
    private = sorted(rec["windows"], key=lambda w: w["id"])
    assert [(w["session_id"], w["block"]) for w in private] == [(w["session_id"], w["row_pos"])
                                                                 for w in doc["disclosed_200w"]]
    assert {w["session_id"] for w in doc["disclosed_200w"]} == v03_sealed.private_disclosed()


def test_load_refuses_tampering(doc, tmp_path: Path):
    bad = json.loads(json.dumps(doc))
    bad["budgets"]["32"][0], bad["budgets"]["32"][1] = bad["budgets"]["32"][1], bad["budgets"]["32"][0]
    p = tmp_path / "BUDGETS.json"
    p.write_text(B.render(bad))
    with pytest.raises(B.BudgetsError, match="32"):
        B.load(p)
    bad = {**doc, "labels": []}
    p.write_text(B.render(bad))
    with pytest.raises(B.BudgetsError, match="not a"):
        B.load(p)


def test_holdout_refuses_small_pool():
    with pytest.raises(B.BudgetsError, match="outside"):
        B.wspf_holdout([f"s-{i:04d}" for i in range(2100)])


def test_registry_requires_budgets():
    assert "datasets/v0_4/BUDGETS.json" in registry.REQUIRED_FILES


@needs_val
def test_rebuild_is_deterministic_and_current(doc):
    a, b = B.build(), B.build()
    assert B.render(a) == B.render(b) == B.BUDGETS_PATH.read_text()
    assert B.check() == []


@needs_val
def test_check_flags_a_stale_file(tmp_path: Path):
    p = tmp_path / "BUDGETS.json"
    p.write_text(B.BUDGETS_PATH.read_text().replace('"format": "v04-budgets/1"', '"format": "v04-budgets/0"'))
    assert any("not a" in x for x in B.check(p)) and any("rebuild" in x for x in B.check(p))


@needs_val
def test_disclosure_mismatch_refused(tmp_path: Path):
    rec = json.loads(v03_sealed.PRIVATE_DISCLOSURE.read_text())
    rec["windows"][0]["block"] += 1
    p = tmp_path / "DISCLOSURE_PRIVATE.json"
    p.write_text(json.dumps(rec))
    with pytest.raises(B.BudgetsError, match="differs from DISCLOSURE_PRIVATE"):
        B.build(private_disclosure=p)
