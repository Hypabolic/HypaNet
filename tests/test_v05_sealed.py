"""v0.5 paired val comparison and one-time sealed evaluation (src/agent_loop_lab/v05/sealed.py,
scripts/v05_sealed_eval.py).

Synthetic fixtures, committed records (manifests, reports, the v0.5 artefacts) and the first few
**validation** sessions only. No test hashes, opens or lists a test, OOD or counterfactual file, and
the sealed mode is never run: an autouse fixture points the official reports and attempt records at
tmp paths and makes writing an attempt record or hashing the sealed files fail.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from agent_loop_lab.eval import m4
from agent_loop_lab.eval.collisions import WindowRow
from agent_loop_lab.schema import CLASS_NAMES, OracleLabel
from agent_loop_lab.v03 import scoring
from agent_loop_lab.v03 import sealed as S
from agent_loop_lab.v04 import scoring as v04_scoring
from agent_loop_lab.v05 import scoring as v05_scoring
from agent_loop_lab.v05 import sealed as V

REPO = Path(__file__).resolve().parents[1]
SCORED = [OracleLabel(n) for n in CLASS_NAMES]
N_RES = 300


def _load_script():
    path = REPO / "scripts" / "v05_sealed_eval.py"
    spec = importlib.util.spec_from_file_location("_test_v05_sealed_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ev():
    return _load_script()


@pytest.fixture(autouse=True)
def _no_sealed(monkeypatch, tmp_path):
    """Official paths -> tmp; writing an attempt record or hashing sealed files fails the test."""
    monkeypatch.setattr(V, "SEALED_REPORT", tmp_path / "reports" / "v05_sealed.json")
    monkeypatch.setattr(V, "SEALED_MARKER", tmp_path / "reports" / "v05_sealed.started.json")
    monkeypatch.setattr(S, "SEALED_MARKER", tmp_path / "reports" / "v03_sealed.started.json")

    def boom(*a, **k):
        raise AssertionError("a test reached the sealed stage")

    monkeypatch.setattr(S, "write_attempt_record", boom)
    monkeypatch.setattr(S, "verify_sealed_files", boom)


# ---------------------------------------------------------------------------
# CLI guards and the one-time rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [
    [],
    ["--sealed"],
    ["--sealed", "--confirm-one-time", "--limit-sessions", "5"],
    ["--sealed", "--confirm-one-time", "--report-dir", "x"],
    ["--sealed", "--confirm-one-time", "--no-counterfactual"],
    ["--sealed", "--confirm-one-time", "--force"],
    ["--dry-run-val"],
    ["--dry-run-val", "--report-dir", "artifacts/reports"],
    ["--dry-run-val", "--report-dir", "x", "--confirm-one-time"],
    ["--dry-run-val", "--report-dir", "x", "--limit-sessions", "0"],
    ["--dry-run-val", "--sealed", "--confirm-one-time", "--report-dir", "x"],
    ["--dry-run-val", "--report-dir", "x", "--workers", "0"],
])
def test_cli_guards(ev, argv, monkeypatch):
    monkeypatch.chdir(REPO)
    with pytest.raises(SystemExit):
        ev.parse_args(argv)


def test_report_dir_guard_resolves_paths(ev, monkeypatch, tmp_path):
    monkeypatch.chdir(REPO / "artifacts")
    with pytest.raises(SystemExit):
        ev.parse_args(["--dry-run-val", "--report-dir", "reports"])
    a = ev.parse_args(["--dry-run-val", "--report-dir", str(tmp_path), "--limit-sessions", "3"])
    assert a.dry_run_val and a.limit_sessions == 3 and not a.sealed


def _must_not_run(*a, **k):
    raise AssertionError("inputs were touched after a refusal")


@pytest.mark.parametrize("which", ["report", "marker"])
def test_sealed_refuses_when_report_or_attempt_exists(ev, monkeypatch, which):
    target = V.SEALED_REPORT if which == "report" else V.SEALED_MARKER
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}")
    monkeypatch.setattr(S, "load_columns", _must_not_run)
    monkeypatch.setattr(V, "load_v05_columns", _must_not_run)
    monkeypatch.setattr(S, "build_split", _must_not_run)
    assert ev.run(ev.parse_args(["--sealed", "--confirm-one-time"])) == 2
    assert any("runs once" in r for r in V.refuse_reasons())


def test_sealed_refuses_without_the_v03_sealed_report(ev, monkeypatch, tmp_path):
    monkeypatch.setattr(V, "V03_SEALED_REPORT", tmp_path / "missing" / "v03_sealed.json")
    monkeypatch.setattr(S, "load_columns", _must_not_run)
    assert ev.run(ev.parse_args(["--sealed", "--confirm-one-time"])) == 2
    assert any("v03_sealed.json missing" in r for r in V.refuse_reasons())


def test_the_real_v03_sealed_report_does_not_block_and_v05_paths_are_its_own():
    committed = REPO / "experiments" / "v0_3" / "sealed" / "README.md"  # stands in for a committed pre-registration
    assert V.refuse_reasons(V.SEALED_REPORT, V.SEALED_MARKER, S.REPORT_DIR / "v03_sealed.json", committed) == []
    assert V.SEALED_REPORT.name == "v05_sealed.json" and V.SEALED_MARKER.name == "v05_sealed.started.json"


def test_sealed_refuses_without_a_committed_pre_registration(ev, monkeypatch, tmp_path):
    assert V.pre_registration_problem(REPO / "experiments" / "v0_3" / "sealed" / "README.md") is None
    assert "missing" in V.pre_registration_problem(tmp_path / "README.md")
    (tmp_path / "README.md").write_text("draft")
    assert "outside the repository" in V.pre_registration_problem(tmp_path / "README.md")
    real_run = subprocess.run
    for failing, want in (("ls-files", "not committed"), ("diff", "differs from HEAD")):
        monkeypatch.setattr(subprocess, "run", lambda cmd, *a, _f=failing, **k: subprocess.CompletedProcess(
            cmd, 1 if _f in cmd else real_run(cmd, *a, **k).returncode))
        assert want in V.pre_registration_problem(REPO / "experiments" / "v0_3" / "sealed" / "README.md")
    monkeypatch.setattr(subprocess, "run", real_run)
    monkeypatch.setattr(V, "PRE_REGISTRATION", "experiments/v0_5/sealed/MISSING_README.md")
    monkeypatch.setattr(S, "load_columns", _must_not_run)
    assert any("pre-registered" in r for r in V.refuse_reasons())
    assert ev.run(ev.parse_args(["--sealed", "--confirm-one-time"])) == 2
    assert V.code_hashes()[V.PRE_REGISTRATION] is None


def test_input_failure_stops_before_attempt_record(ev, monkeypatch):
    def bad(*a, **k):
        raise S.CheckFailed("v0.5 B network: BUILD_RECORD.json artifact_sha256 differs from the artefact")

    monkeypatch.setattr(V, "refuse_reasons", lambda *a, **k: [])  # as if the pre-registration were committed
    monkeypatch.setattr(V, "load_v05_columns", bad)
    monkeypatch.setattr(S, "load_columns", lambda *a, **k: S.Columns([], {}, {}, {}, {}, {}, {}))
    monkeypatch.setattr(S, "build_split", _must_not_run)
    assert ev.main(["--sealed", "--confirm-one-time"]) == 4
    assert not V.SEALED_MARKER.exists() and not V.SEALED_REPORT.exists()


def test_dry_run_refuses_existing_report(ev, tmp_path, monkeypatch):
    (tmp_path / ev.DRY_REPORT_NAME).write_text("{}")
    monkeypatch.setattr(S, "load_columns", _must_not_run)
    assert ev.run(ev.parse_args(["--dry-run-val", "--report-dir", str(tmp_path)])) == 2


def test_previous_attempts_use_the_v05_name(tmp_path):
    (tmp_path / "v05_sealed.attempt1.json").write_text(json.dumps({"started_utc": "t1"}))
    (tmp_path / "v03_sealed.attempt1.json").write_text(json.dumps({"started_utc": "other"}))
    prev = V.previous_attempts(tmp_path)
    assert [p["file"] for p in prev] == ["v05_sealed.attempt1.json"]


# ---------------------------------------------------------------------------
# Inputs (committed records and the v0.5 artefacts; no sealed file)
# ---------------------------------------------------------------------------


def test_v05_columns_load_and_tie_to_their_records():
    v05 = V.load_v05_columns()
    assert [s.name for s in v05.specs] == [V.R1, V.V05B]
    for n in V.V05_NETWORKS:
        want = (n.weight_dir / "artifact.sha256").read_text().split()[0]
        info = v05.info[n.column]
        assert info["artifact_sha256"] == want == json.loads(n.val_report.read_text())["inputs"]["artifact"]["artifact_sha256"]
        assert S.load_net(next(s for s in v05.specs if s.name == n.column)) is not None
    assert v05.info[V.V05B]["labelled_examples"] == 200 and v05.info[V.R1]["labelled_examples"] == 0


def _copy_net(src: V.V05Network, root: Path) -> V.V05Network:
    d = root / src.weight_dir.name
    d.mkdir(parents=True)
    for f in ("artifact.dlw", "artifact.sha256", "apply.json", "BUILD_RECORD.json"):
        if (src.weight_dir / f).is_file():
            shutil.copy(src.weight_dir / f, d / f)
    return V.V05Network(src.column, d, src.val_report, src.run, src.phase, src.labelled)


def test_v05_columns_reject_changed_records(tmp_path):
    r1, b = V.V05_NETWORKS
    bad_sha = _copy_net(r1, tmp_path / "a")
    (bad_sha.weight_dir / "artifact.sha256").write_text("0" * 64 + "  artifact.dlw\n")
    with pytest.raises(S.CheckFailed, match="v0.5 r1 network"):
        V.load_v05_columns([bad_sha])
    bad_build = _copy_net(b, tmp_path / "b")
    rec = json.loads((bad_build.weight_dir / "BUILD_RECORD.json").read_text())
    rec["artifact_sha256"] = "1" * 64
    (bad_build.weight_dir / "BUILD_RECORD.json").write_text(json.dumps(rec))
    with pytest.raises(S.CheckFailed, match="BUILD_RECORD"):
        V.load_v05_columns([bad_build])
    wrong_report = _copy_net(b, tmp_path / "c")
    wrong_report = V.V05Network(b.column, wrong_report.weight_dir, r1.val_report, None, b.phase, b.labelled)
    with pytest.raises(S.CheckFailed, match="scored another artefact"):
        V.load_v05_columns([wrong_report])
    wrong_phase = _copy_net(r1, tmp_path / "d")
    wrong_phase = V.V05Network(r1.column, wrong_phase.weight_dir, r1.val_report, "r1", "v05-phase-b", r1.labelled)
    with pytest.raises(S.CheckFailed, match="phase"):
        V.load_v05_columns([wrong_phase])


def test_v05_b_lineage_is_r1(tmp_path, monkeypatch):
    v05 = V.load_v05_columns()
    assert v05.info[V.V05B]["base_artifact_sha256"] == v05.info[V.R1]["artifact_sha256"]
    r1, b = V.V05_NETWORKS
    with pytest.raises(S.CheckFailed, match="base v0.5 r1 network is not loaded"):
        V.load_v05_columns([b])
    r1c, bc = _copy_net(r1, tmp_path / "a"), _copy_net(b, tmp_path / "b")
    rec = json.loads((bc.weight_dir / "BUILD_RECORD.json").read_text())
    rec["base_artifact_sha256"] = "4" * 64
    (bc.weight_dir / "BUILD_RECORD.json").write_text(json.dumps(rec))
    with pytest.raises(S.CheckFailed, match="base_artifact_sha256 in BUILD_RECORD.json is not"):
        V.load_v05_columns([r1c, bc])
    real = v05_scoring.load_run_artifact

    def other_base(*a, **k):
        spec, art = real(*a, **k)
        if art["artifact_metadata"].get("phase") == b.phase:
            art = {**art, "artifact_metadata": {**art["artifact_metadata"], "base_artifact_sha256": "5" * 64}}
        return spec, art

    monkeypatch.setattr(v05_scoring, "load_run_artifact", other_base)
    with pytest.raises(S.CheckFailed, match="base_artifact_sha256 in artifact_metadata is not"):
        V.load_v05_columns()


def test_v05_disclosure_is_the_v03_set():
    rec = V.check_v05_disclosure(S.private_disclosed())
    assert rec["windows"] == 200 and rec["same_sessions_as_v03"]
    with pytest.raises(S.CheckFailed, match="differ"):
        V.check_v05_disclosure(S.private_disclosed() - {next(iter(S.private_disclosed()))})


def test_registered_sealed_hashes_equal_the_v03_sealed_run_without_reading_sealed_files(monkeypatch):
    opened = []
    real = S.sha256_file
    monkeypatch.setattr(S, "sha256_file", lambda p: opened.append(Path(p)) or real(p))
    v02 = json.loads(S.V02_REPORT.read_text())
    records = S.verify_sealed_hash_records(v02)
    assert opened == []
    v03 = json.loads(V.V03_SEALED_REPORT.read_text())
    cols = S.Columns([], {}, {}, {}, {}, {}, v03["inputs"]["columns"])
    data = {k: v03["inputs"]["data_sha256"][k] for k in ("datasets/v0_3/val.raw.jsonl.gz",)}
    assert V.check_against_v03_sealed_inputs(v03, cols, data, records)["match"]
    bad = dict(records)
    bad[next(iter(bad))] = "0" * 64
    with pytest.raises(S.CheckFailed, match="sealed_hash_records"):
        V.check_against_v03_sealed_inputs(v03, cols, data, bad)
    real_code = S.code_hashes
    monkeypatch.setattr(S, "code_hashes", lambda: {**real_code(), "src/agent_loop_lab/eval/m4.py": "3" * 64})
    with pytest.raises(S.CheckFailed, match="code_sha256"):
        V.check_against_v03_sealed_inputs(v03, cols, data, records)
    monkeypatch.setattr(S, "code_hashes", real_code)
    other = json.loads(json.dumps(v03["inputs"]["columns"]))
    other["phase_c"]["v0.3 C-all seed0"]["artifact_sha256"] = "2" * 64
    with pytest.raises(S.CheckFailed, match="columns"):
        V.check_against_v03_sealed_inputs(v03, S.Columns([], {}, {}, {}, {}, {}, other), data, records)


# ---------------------------------------------------------------------------
# Scoring (synthetic)
# ---------------------------------------------------------------------------


def _rows(n_sessions: int, per: int, seed: int = 0) -> list[WindowRow]:
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_sessions):
        for e in range(per):
            lab = SCORED[int(rng.integers(0, 4))] if rng.random() > 0.1 else OracleLabel.INSUFFICIENT_AMBIGUOUS
            out.append(WindowRow(f"s{s:03d}", f"fam{s % 3}", "val", 7 + e, lab, bool(rng.random() > 0.05), None))
    return out


def _synthetic(n_sessions=40, per=6, seed=0):
    rows = _rows(n_sessions, per, seed)
    ids = sorted({r.session_id for r in rows})
    sidx = {sid: i for i, sid in enumerate(ids)}
    data = scoring.ValData(ids, [f"fam{i % 3}" for i in range(len(ids))],
                           [scoring.EncodedSession([0], [0], [0], [0], ["BOS"]) for _ in ids], rows,
                           np.array([sidx[r.session_id] for r in rows]), np.array([r.endpoint for r in rows]))
    sd = S.SplitData("val", data)
    gate = np.array([r.gate_passed for r in rows])
    truth = np.array([int(r.label.to_class()) if r.label.is_scored else 0 for r in rows])
    rng = np.random.default_rng(seed + 1)

    def noisy(p):
        return np.where(gate, np.where(rng.random(len(rows)) < p, truth, rng.integers(0, 4, len(rows))), -1)

    groups = {"v0.3 C-all": [V.V03C_ALL]}
    cols = S.Columns([], {}, {}, {short: (f"{short} network", f"{short} reference_policy") for short, _ in S.PHASES},
                     groups, {}, {})
    preds, raw = {V.R1: noisy(0.2), V.V05B: noisy(0.9)}, {}
    for short, _ in S.PHASES:
        net = noisy(0.8)
        ref = np.where(gate, net, -1)
        ref[np.flatnonzero(gate)[:3]] = -2
        preds[f"{short} network"] = net
        preds[f"{short} reference_policy"] = scoring.fill_policy_failures(rows, net, ref)
        raw[short] = ref
    preds[V.V03C_ALL] = noisy(0.85)
    preds[S.PRIOR_COLUMNS[0]] = np.where(gate, 0, -1)
    preds[V.PRIOR] = np.where(gate, 0, -1)
    return sd, S.SplitPredictions(preds, raw, {}, {}), cols, frozenset({"s000", "s005", "s013"})


def test_paired_metric_matches_v04_paired_macro_f1_and_point_values():
    sd, sp, cols, _ = _synthetic()
    rows = sd.data.rows
    want = v04_scoring.paired_macro_f1(rows, sp.preds, V.PAIRS, n_resamples=N_RES)
    assert V.paired_metric(rows, sp.preds, V.PAIRS, "macro_f1", n_resamples=N_RES) == want
    sc = m4.score_columns(rows, sp.preds, {}, {}, n_resamples=N_RES, per_family=False)["overall"]["columns"]
    fa = V.paired_metric(rows, sp.preds, V.PAIRS, "false_alarm_rate", n_resamples=N_RES)
    for name, (a, b) in V.PAIRS.items():
        assert want[name]["value"] == pytest.approx(sc[a]["macro_f1"]["value"] - sc[b]["macro_f1"]["value"], abs=1e-15)
        assert fa[name]["value"] == pytest.approx(sc[a]["false_alarm_rate"]["value"] - sc[b]["false_alarm_rate"]["value"],
                                                  abs=1e-15)
        lo, hi = fa[name]["ci95"]
        assert lo <= hi
    assert want["v0.5 B network - v0.5 r1 network"]["value"] > 0


def test_score_all_adds_v05_columns_without_changing_v03_blocks():
    sd, sp, cols, disclosed = _synthetic()
    sc = V.score_all(sd, sp, cols, disclosed, n_resamples=N_RES)
    v03_only = S.SplitPredictions({k: v for k, v in sp.preds.items() if k not in (V.R1, V.V05B)}, sp.raw_ref, {}, {})
    base = S.score_split(sd, v03_only, cols, disclosed, n_resamples=N_RES)
    rec = V.compare_with_v03_sealed("val", sc, base, cols)
    assert rec["reproduced"] and rec["items_compared"] > 0
    for pop in ("excluding_disclosed", "all_val"):
        p = sc["populations"][pop]
        assert {V.R1, V.V05B} <= set(p["overall"]["columns"])
        assert set(p["paired_differences"]) == set(V.PAIRED_METRICS)
        assert set(p["paired_differences"]["macro_f1"]) == set(V.PAIRS)
    table = V.summary(sc, cols)
    assert set(table["all_val"]["paired"]) == set(V.PAIRS) and V.V05B in table["all_val"]["columns"]
    # the official v0.3 path still matches the m4 blocks with the v0.5 columns present
    ok, _ = S.scoring_path_check(sd, sp, sc, disclosed, n_resamples=N_RES)
    assert all(v["match"] for v in ok.values())


def test_v03_sealed_reproduction_is_recorded_not_fatal_and_flags_pairs():
    sd, sp, cols, disclosed = _synthetic()
    sc = V.score_all(sd, sp, cols, disclosed, n_resamples=N_RES)
    recorded = json.loads(json.dumps(sc))
    recorded["populations"]["all_val"]["per_family"]["fam1"]["columns"][V.V03B]["macro_f1"]["value"] += 1e-12
    recorded["populations"]["excluding_disclosed"]["policy_failures"]["v0.3 A"]["total"] += 1
    rec = V.compare_with_v03_sealed("val", sc, recorded, cols)
    assert not rec["reproduced"] and rec["not_reproduced"] == ["policy_failures", V.V03B]
    assert rec["first_difference"][V.V03B] == "val all_val per_family/fam1/macro_f1/value"
    V.caveat_pairs(sc, rec)
    pd = sc["populations"]["all_val"]["paired_differences"]
    assert "caveat" in pd["macro_f1"]["v0.5 B network - v0.3 B network"]
    assert "caveat" in pd["false_alarm_rate"]["v0.5 B network - v0.3 B network"]
    assert "caveat" not in pd["macro_f1"]["v0.5 B network - v0.3 A network"]
    assert V.compare_with_v03_sealed("test", sc, None, cols) == {
        "compared": False, "reason": "v03_sealed.json has no scored test split"}


def _fake_v05_report(sd, sp, column, disclosed, path: Path) -> V.V05Columns:
    body = v05_scoring.score_network(sd.data, sp.preds[column], sp.preds[V.V03A], disclosed=disclosed,
                                     n_resamples=N_RES)
    rep = {**body, "nonzero_parameters": {"nonzero": 5, "slots": 10, "density": 0.5},
           "inputs": {"artifact": {"artifact_sha256": "a" * 64}, "val_rows_sha256": S.rows_sha256(sd.data.rows),
                      "dataset": None}}
    path.write_text(json.dumps(rep))
    return V.V05Columns([], {column: {"artifact_sha256": "a" * 64,
                                      "nonzero_parameters": {"nonzero": 5, "slots": 10, "density": 0.5}}}, {},
                        {column: path})


def test_v05_val_check_reproduces_the_v05_report_path(tmp_path):
    """The m4 blocks of a v0.5 column equal what v05.scoring (the v0.5 val report path) wrote."""
    sd, sp, cols, disclosed = _synthetic()
    sc = V.score_all(sd, sp, cols, disclosed, n_resamples=N_RES)
    for column in (V.R1, V.V05B):
        v05 = _fake_v05_report(sd, sp, column, disclosed, tmp_path / f"{column}.json")
        rec = V.v05_val_check(sd, sp, sc, v05, disclosed, n_resamples=N_RES)
        assert rec[column]["match"] and rec[column]["blocks_compared"] > 20
    path = tmp_path / f"{V.V05B}.json"
    rep = json.loads(path.read_text())
    rep["excluding_disclosed"]["per_family"]["fam2"]["network"]["confusion_matrix"]["counts"][0][1] += 1
    path.write_text(json.dumps(rep))
    with pytest.raises(S.CheckFailed, match="v0.5 B network: differs .* excluding_disclosed/per_family/fam2/network"):
        V.v05_val_check(sd, sp, sc, v05, disclosed, n_resamples=N_RES)
    rep = json.loads(path.read_text())
    rep["excluding_disclosed"]["per_family"]["fam2"]["network"]["confusion_matrix"]["counts"][0][1] -= 1
    rep["all_val"]["paired_macro_f1"]["v0.5 network - v0.3 A network"]["ci95"][0] += 1e-9
    path.write_text(json.dumps(rep))
    with pytest.raises(S.CheckFailed, match="paired_macro_f1"):
        V.v05_val_check(sd, sp, sc, v05, disclosed, n_resamples=N_RES)


# ---------------------------------------------------------------------------
# Predictions (first validation sessions)
# ---------------------------------------------------------------------------


def test_predict_all_runs_v03_path_then_v05_networks(monkeypatch):
    """v0.3's predict_split unchanged (networks, then each policy alone), then the v0.5 networks; the v0.5
    classes are those of the v0.5 val scorer's path."""
    data = scoring.load_val(limit=4)
    sd = S.SplitData("val", data)
    events: list[str] = []

    def fake_nets(specs, data, pool, **k):
        events.append("v03 nets")
        return {s.name: (np.zeros(len(data.rows), dtype=np.int64), {}) for s in specs}

    def fake_eval(policy_file, inputs):
        events.append("policy")
        return [1] * len(inputs)

    real_v05 = v05_scoring.network_predictions

    def spy_v05(specs, data, pool, **k):
        events.append("v05 nets")
        return real_v05(specs, data, pool, **k)

    monkeypatch.setattr(S, "network_predictions", fake_nets)
    monkeypatch.setattr(v05_scoring, "network_predictions", spy_v05)
    nets = [S.NetSpec(f"{short} network", "state_dict", "x", "0") for short, _ in S.PHASES]
    pols = {short: Path(f"/nonexistent/{short}.py") for short, _ in S.PHASES}
    cols = S.Columns(nets, pols, {}, {}, {}, {}, {})
    v05 = V.load_v05_columns()
    sp, logits = V.predict_all(sd, cols, v05, {"majority_index": 0}, None, workers=0, evaluate=fake_eval)
    assert events[0] == "v03 nets" and events[-1] == "v05 nets" and events.count("v05 nets") == 1
    assert list(sp.preds)[:2] == [V.R1, V.V05B]
    assert set(logits) == {V.R1, V.V05B} and logits[V.R1]["events"] > 0
    want = real_v05(v05.specs, data, None, workers=0)
    gate = np.array([r.gate_passed for r in data.rows])
    for name in (V.R1, V.V05B):
        assert np.array_equal(sp.preds[name], want[name].pred) and np.all(sp.preds[name][gate] >= 0)


# ---------------------------------------------------------------------------
# run() end to end with fakes: no data, no sealed file; the order of the sealed steps
# ---------------------------------------------------------------------------


class _Harness:
    """Every data-touching step of ``run`` replaced by a recording fake."""

    def __init__(self, monkeypatch, tmp_path):
        self.calls: list[tuple[str, str | None]] = []
        sd, sp, cols, disclosed = _synthetic(n_sessions=12, per=3)
        self.sd, self.sp, self.cols, self.disclosed = sd, sp, cols, disclosed
        prior = {"class_counts": [1, 2, 3, 4], "majority_index": 3}
        v02 = tmp_path / "v02.json"
        v02.write_text(json.dumps({"dev_prior": prior}))
        v03 = tmp_path / "v03_sealed.json"
        v03.write_text(json.dumps({"dev_prior": prior, "splits": {},
                                   "checks": {"identity_v02": {s: {"rows_sha256": f"rows-{s}"} for s in S.SEALED_SPLITS}}}))
        self.sealed_records = {"datasets/v0_3/test.raw.jsonl.gz": "h1", "datasets/v0_3/ood.raw.jsonl.gz": "h2"}
        self.attempt: S.Attempt | None = None
        rec = self.rec

        def build_split(split, *, limit=None, attempt=None):
            rec("build_split", split)
            if split in S.SEALED_SPLITS:
                assert attempt is not None and attempt == self.attempt and attempt.marker.is_file()
            return S.SplitData(split, sd.data)

        def write_attempt_record(record, marker=None):
            rec("write_attempt_record")
            assert marker == V.SEALED_MARKER
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(record))
            self.attempt = S.Attempt(marker, "token")
            return self.attempt

        def verify_sealed_files(attempt):
            rec("verify_sealed_files")
            assert attempt == self.attempt
            return dict(self.sealed_records)

        def identity_check(sd_, v02_, *, limit=None, attempt=None):
            rec("identity_check", sd_.split)
            return {"rows_sha256": f"rows-{sd_.split}"}

        def predict_all(sd_, *a, **k):
            rec("predict_all", sd_.split)
            return sp, {}

        def score_all(sd_, *a, **k):
            rec("score_all", sd_.split)
            return {"populations": {}}

        fakes = {
            (V, "refuse_reasons"): lambda *a, **k: [],
            (V, "previous_attempts"): lambda *a, **k: [],
            (S, "git_state"): lambda *a, **k: {},
            (S, "load_columns"): lambda *a, **k: rec("load_columns") or cols,
            (V, "load_v05_columns"): lambda *a, **k: rec("load_v05_columns") or V.V05Columns([], {}, {}, {}),
            (S, "verify_unsealed_inputs"): lambda splits=(): rec("verify_unsealed_inputs") or {},
            (S, "V02_REPORT"): v02,
            (V, "V03_SEALED_REPORT"): v03,
            (S, "verify_sealed_hash_records"): lambda *a, **k: dict(self.sealed_records),
            (V, "check_against_v03_sealed_inputs"): lambda *a, **k: rec("v03_sealed_inputs") or {"match": True},
            (S, "dev_prior"): lambda: dict(prior),
            (S, "private_disclosed"): lambda *a, **k: disclosed,
            (V, "check_v05_disclosure"): lambda *a, **k: {},
            (S, "make_pool"): lambda workers: contextlib.nullcontext(),
            (S, "build_split"): build_split,
            (scoring, "disclosed_sessions"): lambda rows: disclosed,
            (v04_scoring, "VAL_ROWS_SHA256"): S.rows_sha256(sd.data.rows),
            (S, "identity_check"): identity_check,
            (V, "predict_all"): predict_all,
            (V, "score_all"): score_all,
            (S, "scoring_path_check"): lambda *a, **k: ({}, {}),
            (S, "official_val_check"): lambda *a, **k: rec("official_val_check") or {},
            (V, "v05_val_check"): lambda *a, **k: rec("v05_val_check") or {},
            (S, "phase_c_es_check"): lambda *a, **k: rec("phase_c_es_check") or {},
            (S, "prior_confirmation"): lambda *a, **k: {},
            (V, "compare_with_v03_sealed"): lambda split, *a, **k: {"compared": True, "reproduced": True,
                                                                   "not_reproduced": [], "first_difference": {},
                                                                   "items_compared": 1},
            (V, "compare_counterfactual_with_v03_sealed"): lambda *a, **k: {"compared": False, "reason": "fake"},
            (V, "summary"): lambda *a, **k: {},
            (S, "counterfactual_scores"): lambda *a, **k: {},
            (S, "latency_sample"): lambda data: ([], []),
            (S, "isolated_latency"): lambda *a, **k: rec("isolated_latency") or {},
            (S, "write_attempt_record"): write_attempt_record,
            (S, "verify_sealed_files"): verify_sealed_files,
        }
        for (mod, name), fake in fakes.items():
            monkeypatch.setattr(mod, name, fake)

    def rec(self, what: str, split: str | None = None) -> None:
        self.calls.append((what, split))

    def index(self, what: str, split: str | None = None) -> int:
        return next(i for i, c in enumerate(self.calls) if c[0] == what and (split is None or c[1] == split))


@pytest.mark.parametrize("limit", [None, 5])
def test_dry_run_never_asks_for_a_sealed_split(ev, monkeypatch, tmp_path, limit):
    h = _Harness(monkeypatch, tmp_path)
    monkeypatch.setattr(S, "write_attempt_record", _must_not_run)
    monkeypatch.setattr(S, "verify_sealed_files", _must_not_run)
    argv = ["--dry-run-val", "--report-dir", str(tmp_path / "out")] + ([] if limit is None else ["--limit-sessions", "5"])
    assert ev.main(argv) == 0
    assert not any(split in S.SEALED_SPLITS for _, split in h.calls)
    assert [c for c in h.calls if c[0] == "build_split"] == [("build_split", "val"), ("build_split", "counterfactual")]
    assert ("phase_c_es_check", None) in h.calls if limit is None else ("phase_c_es_check", None) not in h.calls
    rep = json.loads((tmp_path / "out" / ev.DRY_REPORT_NAME).read_text())
    assert rep["official"] is False and rep["attempt"] is None and set(rep["splits"]) == {"val", "counterfactual"}
    assert not V.SEALED_MARKER.exists() and not V.SEALED_REPORT.exists()


def test_sealed_order_attempt_record_before_any_sealed_access(ev, monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path)
    assert ev.main(["--sealed", "--confirm-one-time"]) == 0
    first_sealed = min(i for i, (what, split) in enumerate(h.calls)
                       if what == "verify_sealed_files" or split in S.SEALED_SPLITS)
    attempt = h.index("write_attempt_record")
    assert h.calls.count(("write_attempt_record", None)) == 1
    for step in ("load_columns", "load_v05_columns", "verify_unsealed_inputs", "v03_sealed_inputs",
                 "official_val_check", "v05_val_check", "phase_c_es_check", "isolated_latency"):
        assert h.index(step) < attempt, step
    assert h.index("predict_all", "counterfactual") < attempt
    assert attempt < first_sealed == h.index("verify_sealed_files")
    assert h.index("identity_check", "test") < h.index("predict_all", "test")
    rep = json.loads(V.SEALED_REPORT.read_text())
    assert rep["official"] is True and rep["attempt"]["number"] == 1
    assert set(rep["splits"]) == {"test", "ood", "val", "counterfactual"}
    assert all(rep["checks"]["identity_v02"][s]["matches_v03_sealed_rows"] for s in S.SEALED_SPLITS)


@pytest.mark.parametrize("failing", ["phase_c_es_check", "v05_val_check", "official_val_check"])
def test_val_check_failure_stops_before_attempt_record(ev, monkeypatch, tmp_path, failing):
    h = _Harness(monkeypatch, tmp_path)

    def fail(*a, **k):
        raise S.CheckFailed(f"{failing}: differs")

    monkeypatch.setattr(V if failing == "v05_val_check" else S, failing, fail)
    assert ev.main(["--sealed", "--confirm-one-time"]) == 4
    assert ("write_attempt_record", None) not in h.calls and ("verify_sealed_files", None) not in h.calls
    assert not any(split in S.SEALED_SPLITS for _, split in h.calls)
    assert not V.SEALED_MARKER.exists() and not V.SEALED_REPORT.exists()


def test_sealed_rows_differing_from_v03_is_a_spent_attempt_without_metrics(ev, monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path)
    monkeypatch.setattr(S, "identity_check", lambda sd_, *a, **k: {"rows_sha256": "other"})
    assert ev.main(["--sealed", "--confirm-one-time"]) == 3
    assert V.SEALED_MARKER.exists() and not V.SEALED_REPORT.exists()
    assert ("predict_all", "test") not in h.calls
