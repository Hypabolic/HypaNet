"""v0.3 one-time sealed evaluation tooling (src/agent_loop_lab/v03/sealed.py, scripts/v03_sealed_eval.py).

Synthetic fixtures and the first few **validation** sessions only. No test hashes, opens or lists a
test or OOD file, and the sealed mode is never run: an autouse fixture points the official report and
attempt record at tmp paths and makes writing an attempt record fail.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agent_loop_lab.eval import m4
from agent_loop_lab.eval.collisions import WindowRow
from agent_loop_lab.schema import CLASS_NAMES, OracleLabel
from agent_loop_lab.v03 import scoring
from agent_loop_lab.v03 import sealed as S
from agent_loop_lab.v03.model import zero_model

REPO = Path(__file__).resolve().parents[1]
SCORED = [OracleLabel(n) for n in CLASS_NAMES]


def _load_script():
    path = REPO / "scripts" / "v03_sealed_eval.py"
    spec = importlib.util.spec_from_file_location("_test_v03_sealed_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ev():
    return _load_script()


@pytest.fixture(autouse=True)
def _no_sealed(monkeypatch, tmp_path):
    """Official paths -> tmp; writing an attempt record or hashing sealed files fails the test."""
    monkeypatch.setattr(S, "SEALED_REPORT", tmp_path / "reports" / "v03_sealed.json")
    monkeypatch.setattr(S, "SEALED_MARKER", tmp_path / "reports" / "v03_sealed.started.json")

    def boom(*a, **k):
        raise AssertionError("a test reached the sealed stage")

    monkeypatch.setattr(S, "write_attempt_record", boom)
    monkeypatch.setattr(S, "verify_sealed_files", boom)


def _rows(n_sessions: int, per: int, split: str = "val", seed: int = 0) -> list[WindowRow]:
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_sessions):
        for e in range(per):
            lab = SCORED[int(rng.integers(0, 4))] if rng.random() > 0.1 else OracleLabel.INSUFFICIENT_AMBIGUOUS
            out.append(WindowRow(f"s{s:03d}", f"fam{s % 3}", split, 7 + e, lab, bool(rng.random() > 0.05), None))
    return out


# ---------------------------------------------------------------------------
# CLI guards
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


@pytest.mark.parametrize("which", ["report", "marker"])
def test_sealed_refuses_when_report_or_attempt_exists(ev, monkeypatch, which):
    target = S.SEALED_REPORT if which == "report" else S.SEALED_MARKER
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}")

    def must_not_run(*a, **k):
        raise AssertionError("inputs were touched after a refusal")

    monkeypatch.setattr(S, "load_columns", must_not_run)
    monkeypatch.setattr(S, "build_split", must_not_run)
    args = ev.parse_args(["--sealed", "--confirm-one-time"])
    assert ev.run(args) == 2
    assert S.refuse_reasons()


def test_sealed_hash_failure_stops_before_attempt_record(ev, monkeypatch):
    """A failed input check returns non-zero and writes no attempt record."""
    def bad_columns(*a, **k):
        raise S.CheckFailed("Phase C 32/seed0: model.pt sha256 != model.sha256")

    monkeypatch.setattr(S, "load_columns", bad_columns)
    assert ev.main(["--sealed", "--confirm-one-time"]) == 4
    assert not S.SEALED_MARKER.exists() and not S.SEALED_REPORT.exists()


def test_dry_run_refuses_existing_report(ev, tmp_path, monkeypatch):
    (tmp_path / ev.DRY_REPORT_NAME).write_text("{}")

    def must_not_run(*a, **k):
        raise AssertionError("ran")

    monkeypatch.setattr(S, "load_columns", must_not_run)
    assert ev.run(ev.parse_args(["--dry-run-val", "--report-dir", str(tmp_path)])) == 2


# ---------------------------------------------------------------------------
# Sealed access
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("split", S.SEALED_SPLITS)
def test_sealed_splits_need_an_attempt_record(split, tmp_path, monkeypatch):
    def must_not_open(*a, **k):
        raise AssertionError("a sealed file would be opened")

    monkeypatch.setattr(S, "_v02_sessions", must_not_open)
    monkeypatch.setattr(m4, "load_split", must_not_open)
    with pytest.raises(S.SealedAccessError):
        S.build_split(split)
    with pytest.raises(S.SealedAccessError):
        S.v02_window_rows(split)
    with pytest.raises(S.SealedAccessError):
        S.verify_unsealed_inputs((split,))
    marker = tmp_path / "marker.json"
    marker.write_text(json.dumps({"attempt_token": "right"}))
    with pytest.raises(S.SealedAccessError):
        S.build_split(split, attempt=S.Attempt(marker, "wrong"))
    with pytest.raises(S.SealedAccessError):
        S._encode_split(split, attempt=S.Attempt(tmp_path / "missing.json", "right"))
    # a valid token still refuses a session limit, before anything is opened
    with pytest.raises(S.SealedAccessError):
        S.build_split(split, limit=5, attempt=S.Attempt(marker, "right"))


def test_verify_sealed_files_checks_the_token_first(tmp_path, monkeypatch):
    monkeypatch.undo()  # the real verify_sealed_files; it must refuse before hashing anything
    marker = tmp_path / "m.json"
    marker.write_text(json.dumps({"attempt_token": "t"}))
    opened = []
    monkeypatch.setattr(S, "sha256_file", lambda p: opened.append(p) or "x")
    with pytest.raises(S.SealedAccessError):
        S.verify_sealed_files(S.Attempt(marker, "other"))
    assert opened == []


def test_attempt_record_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.undo()  # the real write_attempt_record, tmp marker only
    marker = tmp_path / "rec.json"
    a = S.write_attempt_record({"x": 1}, marker)
    doc = json.loads(marker.read_text())
    assert doc["x"] == 1 and doc["attempt_token"] == a.token and len(a.token) == 32
    S._check_attempt("test", a)  # valid token passes
    with pytest.raises(FileExistsError):
        S.write_attempt_record({"x": 2}, marker)


def test_previous_attempts_are_collected(tmp_path):
    (tmp_path / "v03_sealed.attempt1.json").write_text(json.dumps({"started_utc": "t1"}))
    prev = S.previous_attempts(tmp_path)
    assert [p["file"] for p in prev] == ["v03_sealed.attempt1.json"] and prev[0]["content"]["started_utc"] == "t1"


# ---------------------------------------------------------------------------
# Input hashes
# ---------------------------------------------------------------------------


def _phase_c_run(root: Path, budget, seed, *, sha=None, smoke=False, log_sha=None):
    d = root / str(budget) / f"seed{seed}"
    d.mkdir(parents=True)
    torch.save(zero_model().state_dict(), d / "model.pt")
    real = hashlib.sha256((d / "model.pt").read_bytes()).hexdigest()
    (d / "model.sha256").write_text(f"{sha or real}  model.pt\n")
    log = {"model_sha256": log_sha or real, "smoke": smoke, "phase": "v03-phase-c", "budget": budget, "seed": seed,
           "train_positions": 10, "budget_sessions": 32, "pool_sessions": 30, "epochs_run": 2, "best_epoch": 1,
           "best_es_macro_f1": 0.5, "stopped": "epoch_cap", "environment": {"torch_threads": 2}}
    (d / "training_log.json").write_text(json.dumps(log))
    return real


def test_phase_c_hash_checks(tmp_path):
    real = _phase_c_run(tmp_path / "ok", 32, 0)
    spec, log, info = S.check_phase_c_run(32, 0, tmp_path / "ok")
    assert spec.sha256 == real and info["labelled_examples"] == 10 and spec.name == "v0.3 C-32 seed0"
    assert S.load_net(spec) is not None
    _phase_c_run(tmp_path / "bad_sha", 32, 0, sha="0" * 64)
    with pytest.raises(S.CheckFailed, match="model.sha256"):
        S.check_phase_c_run(32, 0, tmp_path / "bad_sha")
    _phase_c_run(tmp_path / "bad_log", 32, 0, log_sha="1" * 64)
    with pytest.raises(S.CheckFailed, match="training log"):
        S.check_phase_c_run(32, 0, tmp_path / "bad_log")
    _phase_c_run(tmp_path / "smoke", 32, 0, smoke=True)
    with pytest.raises(S.CheckFailed, match="registered"):
        S.check_phase_c_run(32, 0, tmp_path / "smoke")
    _phase_c_run(tmp_path / "wrong_seed", 32, 0)
    (tmp_path / "wrong_seed" / "32" / "seed0").rename(tmp_path / "wrong_seed" / "32" / "seed1")
    with pytest.raises(S.CheckFailed, match="budget/seed"):
        S.check_phase_c_run(32, 1, tmp_path / "wrong_seed")


def test_load_net_rejects_a_changed_file(tmp_path):
    _phase_c_run(tmp_path, 32, 0)
    spec, _, _ = S.check_phase_c_run(32, 0, tmp_path)
    (tmp_path / "32" / "seed0" / "model.pt").write_bytes(b"changed")
    with pytest.raises(S.CheckFailed, match="changed"):
        S.load_net(spec)


def test_unsealed_input_mismatch_fails(monkeypatch):
    real = S.expected_hashes

    def wrong(split, man=None):
        h = real(split, man)
        return {k: ("0" * 64 if k.endswith(f"{split}.raw.jsonl.gz") else v) for k, v in h.items()}

    monkeypatch.setattr(S, "expected_hashes", wrong)
    with pytest.raises(S.CheckFailed, match="val.raw.jsonl.gz"):
        S.verify_unsealed_inputs(("val",))


def test_sealed_hash_records_cross_check_without_reading_sealed_files(monkeypatch):
    opened = []
    real = S.sha256_file
    monkeypatch.setattr(S, "sha256_file", lambda p: opened.append(Path(p)) or real(p))
    v02 = json.loads(S.V02_REPORT.read_text())
    got = S.verify_sealed_hash_records(v02)
    assert set(got) == {f"datasets/v0_3/{s}.raw.jsonl.gz" for s in S.SEALED_SPLITS} | {
        "datasets/v0_2/standard/test.events.jsonl.gz", "datasets/v0_2/standard/test.oracle.jsonl.gz",
        "datasets/v0_2/ood/ood.events.jsonl.gz", "datasets/v0_2/ood/ood.oracle.jsonl.gz"}
    assert opened == []
    bad = json.loads(json.dumps(v02))
    bad["inputs"]["data_sha256"]["standard/test.oracle.jsonl.gz"] = "0" * 64
    with pytest.raises(S.CheckFailed, match="m4_sealed_v02"):
        S.verify_sealed_hash_records(bad)


# ---------------------------------------------------------------------------
# Scoring and aggregation (synthetic)
# ---------------------------------------------------------------------------


def _cols(groups=None) -> S.Columns:
    fid = {short: (f"{short} network", f"{short} reference_policy") for short, _ in S.PHASES}
    return S.Columns([], {}, {}, fid, groups or {}, {}, {})


def test_phase_c_seed_aggregation():
    rows = _rows(40, 6)
    rng = np.random.default_rng(1)
    names = [f"v0.3 C-32 seed{s}" for s in range(3)]
    preds = {n: rng.integers(0, 4, len(rows)) for n in names}
    preds["v0.3 C-2048 seed0"] = rng.integers(0, 4, len(rows))
    groups = {"v0.3 C-32": names, "v0.3 C-2048": ["v0.3 C-2048 seed0"]}
    sc = m4.score_columns(rows, preds, {}, groups, n_resamples=200)
    vals = [sc["overall"]["columns"][n]["macro_f1"]["value"] for n in names]
    g = sc["overall"]["groups"]["v0.3 C-32"]["macro_f1"]
    assert g["per_seed"] == vals
    assert g["mean"] == pytest.approx(np.mean(vals)) and g["sd"] == pytest.approx(np.std(vals, ddof=1))
    lo, hi = g["seed_mean_ci95"]
    assert lo <= g["mean"] <= hi
    one = sc["overall"]["groups"]["v0.3 C-2048"]["macro_f1"]
    assert one["sd"] is None and one["mean"] == sc["overall"]["columns"]["v0.3 C-2048 seed0"]["macro_f1"]["value"]
    table = S.summary_table(sc, _cols(groups))
    assert table["v0.3 C-32 mean±sd"]["macro_f1"]["sd"] == g["sd"]
    assert table["v0.3 C-2048 mean±sd"]["macro_f1"]["sd"] is None


def _synthetic_split(n_sessions=30, per=6):
    rows = _rows(n_sessions, per)
    ids = sorted({r.session_id for r in rows})
    sidx = {sid: i for i, sid in enumerate(ids)}
    r_sess = np.array([sidx[r.session_id] for r in rows])
    r_pos = np.array([r.endpoint for r in rows])
    fams = [f"fam{i % 3}" for i in range(len(ids))]
    sessions = [scoring.EncodedSession([0], [0], [0], [0], ["BOS"]) for _ in ids]
    data = scoring.ValData(ids, fams, sessions, rows, r_sess, r_pos)
    return S.SplitData("val", data)


def test_m4_blocks_equal_scoring_score_with_policy_failures():
    sd = _synthetic_split()
    rows = sd.data.rows
    rng = np.random.default_rng(3)
    gate = np.array([r.gate_passed for r in rows])
    preds, raw = {}, {}
    for short, _ in S.PHASES:
        net = np.where(gate, rng.integers(0, 4, len(rows)), -1)
        ref = np.where(gate, rng.integers(0, 4, len(rows)), -1)
        ref[np.flatnonzero(gate)[:5]] = -2  # policy failures
        preds[f"{short} network"] = net
        preds[f"{short} reference_policy"] = scoring.fill_policy_failures(rows, net, ref)
        raw[short] = ref
    sp = S.SplitPredictions(preds, raw, {}, {})
    disclosed = frozenset({"s000", "s004", "s011"})
    sc = S.score_split(sd, sp, _cols(), disclosed, n_resamples=300)
    assert set(sc["populations"]) == {"excluding_disclosed", "all_val"}
    assert sc["populations"]["excluding_disclosed"]["overall"]["sessions"] == 27
    rec, bodies = S.scoring_path_check(sd, sp, sc, disclosed, n_resamples=300)
    assert all(v["match"] for v in rec.values()) and set(bodies) == {p for _, p in S.PHASES}
    assert sc["populations"]["all_val"]["policy_failures"]["v0.3 A"]["total"] == 5
    # a perturbed m4 block is caught
    sc["populations"]["all_val"]["overall"]["columns"]["v0.3 B network"]["macro_f1"]["value"] += 1e-12
    with pytest.raises(S.CheckFailed, match="v03-phase-b all_val"):
        S.scoring_path_check(sd, sp, sc, disclosed, n_resamples=300)


def test_non_val_split_has_one_population():
    sd = _synthetic_split()
    sd.split = "counterfactual"
    assert list(S.population_masks(sd, frozenset({"s000"}))) == ["all"]


def test_identity_check_detects_a_changed_row(monkeypatch):
    sd = _synthetic_split()
    changed = list(sd.data.rows)
    r = changed[3]
    changed[3] = WindowRow(r.session_id, r.family, r.split, r.endpoint,
                           OracleLabel.STALLED if r.label != OracleLabel.STALLED else OracleLabel.RECOVERY,
                           r.gate_passed, None)
    monkeypatch.setattr(S, "v02_window_rows", lambda *a, **k: list(sd.data.rows))
    rec = S.identity_check(sd, None, limit=None)
    assert rec["matches_v02_load_split"] and rec["rows_sha256"] == S.rows_sha256(sd.data.rows)
    monkeypatch.setattr(S, "v02_window_rows", lambda *a, **k: changed)
    with pytest.raises(S.CheckFailed, match="differ from v0.2"):
        S.identity_check(sd, None)


def test_complete_pairs_and_first_difference():
    meta = {"a1": {"pair_id": "p1", "branch": "a"}, "b1": {"pair_id": "p1", "branch": "b"},
            "a2": {"pair_id": "p2", "branch": "a"}}
    assert set(S.complete_pairs(meta)) == {"a1", "b1"}
    assert S.first_difference({"x": [1, {"y": 2.0}]}, {"x": [1, {"y": 2.0}]}) is None
    assert S.first_difference({"x": [1, {"y": 2.0}]}, {"x": [1, {"y": 2.5}]}) == "/x[1]/y"
    assert S.first_difference({"x": 1}, {"x": 1.0}) == "/x"
    assert S.first_difference({"x": 1}, {"z": 1}) is not None


# ---------------------------------------------------------------------------
# Real validation data (first sessions only)
# ---------------------------------------------------------------------------


def test_generic_builder_matches_load_val_on_val_prefix():
    n = 12
    ref = scoring.load_val(limit=n)
    got = S._encode_split("val", limit=n)
    assert got.data.session_ids == ref.session_ids and got.data.families == ref.families
    assert S.row_tuples(got.data.rows) == S.row_tuples(ref.rows)
    assert np.array_equal(got.data.row_pos, ref.row_pos) and np.array_equal(got.data.row_session, ref.row_session)
    for a, b in zip(got.data.sessions, ref.sessions):
        assert (a.tokens, a.event_ord, a.offset, a.eoe_index, a.surface) == (b.tokens, b.event_ord, b.offset, b.eoe_index, b.surface)


def test_val_and_counterfactual_rows_match_v02_loader():
    for split in ("val", "counterfactual"):
        sd = S.build_split(split, limit=10)
        rec = S.identity_check(sd, None, limit=10)
        assert rec["matches_v02_load_split"]
        if split == "counterfactual":
            assert all(m["pair_id"] for m in sd.meta.values())


def test_parallel_network_predictions_match_scoring_path(tmp_path):
    data = scoring.load_val(limit=6)
    model = zero_model()
    for p in model.parameters():  # a non-trivial deterministic model
        with torch.no_grad():
            p.copy_(torch.linspace(-0.05, 0.05, p.numel()).reshape(p.shape))
    path = tmp_path / "m.pt"
    torch.save(model.state_dict(), path)
    spec = S.NetSpec("m", "state_dict", str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    want, _ = scoring.network_predictions(S.load_net(spec), data)
    with S.make_pool(2) as pool:
        got = S.network_predictions([spec], data, pool, workers=2, chunk_sessions=2)
    assert np.array_equal(got["m"][0], want)
    assert got["m"][1]["sessions"] == 6 and got["m"][1]["workers"] == 2


def test_dev_prior_matches_v02_report():
    prior = S.dev_prior()
    v02 = json.loads(S.V02_REPORT.read_text())
    assert prior["class_counts"] == v02["dev_prior"]["class_counts"]
    assert prior["majority_index"] == v02["dev_prior"]["majority_index"]


def test_latency_sample_and_isolated_latency(tmp_path):
    data = scoring.load_val(limit=7)
    ids, encs = S.latency_sample(data, n=3)
    assert ids == [data.session_ids[i] for i in (0, 3, 6)] and len(encs) == 3
    torch.save(zero_model().state_dict(), tmp_path / "z.pt")
    spec = S.NetSpec("z", "state_dict", str(tmp_path / "z.pt"), hashlib.sha256((tmp_path / "z.pt").read_bytes()).hexdigest())
    lat = S.isolated_latency([spec], encs)["z"]
    assert lat["sessions"] == 3 and lat["workers"] == 1 and lat["definition"].startswith("isolated")


def test_policies_run_after_networks_and_one_at_a_time(monkeypatch):
    """Same measurement conditions as score_phase: networks first, then each policy alone."""
    import threading

    data = scoring.load_val(limit=6)
    sd = S.SplitData("val", data)
    events: list[str] = []
    lock = threading.Lock()
    active: dict[str, int] = {}

    def fake_nets(specs, data, pool, **k):
        events.append("nets")
        return {s.name: (np.zeros(len(data.rows), dtype=np.int64), {}) for s in specs}

    def fake_eval(policy_file, inputs):
        with lock:
            assert events and events[0] == "nets", "a policy ran before network inference finished"
            other = [p for p, n in active.items() if n and p != str(policy_file)]
            assert not other, f"policies overlapped: {policy_file} with {other}"
            active[str(policy_file)] = active.get(str(policy_file), 0) + 1
            events.append(str(policy_file))
        try:
            return [1] * len(inputs)
        finally:
            with lock:
                active[str(policy_file)] -= 1

    monkeypatch.setattr(S, "network_predictions", fake_nets)
    nets = [S.NetSpec(f"{short} network", "state_dict", "x", "0") for short, _ in S.PHASES]
    pols = {short: Path(f"/nonexistent/{short}.py") for short, _ in S.PHASES}
    cols = S.Columns(nets, pols, {}, {}, {}, {}, {})
    msgs: list[str] = []
    sp = S.predict_split(sd, cols, {"majority_index": 0}, None, workers=1, progress=msgs.append, evaluate=fake_eval)
    pol_events = [e for e in events if e != "nets"]
    first_b = pol_events.index(str(pols["v0.3 B"]))
    assert all(e == str(pols["v0.3 A"]) for e in pol_events[:first_b])
    assert all(e == str(pols["v0.3 B"]) for e in pol_events[first_b:])
    gate = np.array([r.gate_passed for r in data.rows])
    assert all(np.array_equal(sp.raw_ref[s][gate], np.ones(int(gate.sum()))) for s in pols)
    assert any("0 failures" in m for m in msgs)
