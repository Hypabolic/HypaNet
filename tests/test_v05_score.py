"""v0.5 validation scoring (src/agent_loop_lab/v05/scoring.py, scripts/v05_score_validation.py).

Synthetic sessions and fake artefacts under tmp_path only: no test loads the validation split.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agent_loop_lab.eval import metrics
from agent_loop_lab.eval.collisions import WindowRow
from agent_loop_lab.schema import CLASS_NAMES, OracleLabel
from agent_loop_lab.v03 import scoring as v03_scoring
from agent_loop_lab.v03 import tokenizer
from agent_loop_lab.v03.artifact import save_artifact
from agent_loop_lab.v03.model import zero_model
from agent_loop_lab.v05 import scoring

REPO = Path(__file__).resolve().parents[1]
SCORED = [OracleLabel(n) for n in CLASS_NAMES]
N_RES = 200


def _load_script(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sv():
    return _load_script("v05_score_validation")


def _session_blocks(n: int) -> list[str]:
    out = []
    for k in range(n):
        tool = "edit_file" if k % 4 == 1 else "read_file"
        b = f"00:00:{k:02d}.000 {tool} path=src/mod{k % 3}.py -> ok in=12 out=34"
        if k % 4 == 1:
            b += f"\n  @ src/mod{k % 3}.py h{k:010x} > h{k + 1:010x}"
        out.append(b)
    return out


def synthetic_data(n_sessions: int = 12, n_events: int = 11, seed: int = 0) -> v03_scoring.ValData:
    rng = np.random.default_rng(seed)
    ids, fams, sessions, rows, r_sess, r_pos = [], [], [], [], [], []
    for s in range(n_sessions):
        sid = f"sess-{s:03d}"
        ids.append(sid)
        fams.append(f"fam_{s % 2}")
        enc = tokenizer.encode_session(_session_blocks(n_events))
        sessions.append(v03_scoring.EncodedSession(list(enc.tokens), list(enc.event_ord), list(enc.offset),
                                                   list(enc.eoe_index), list(enc.surface)))
        for pos in range(7, n_events):
            u = rng.random()
            label = OracleLabel.INSUFFICIENT_AMBIGUOUS if u < 0.1 else SCORED[int(rng.integers(0, 4))]
            gate = not (u > 0.95)
            if not gate:
                label = OracleLabel.INSUFFICIENT_INSTRUMENTATION
            rows.append(WindowRow(sid, fams[-1], "val", pos, label, gate, None))
            r_sess.append(s)
            r_pos.append(pos)
    return v03_scoring.ValData(ids, fams, sessions, rows, np.array(r_sess), np.array(r_pos), {"cache_key": {"fake": True}})


def head_bias_model(bias: list[float]):
    m = zero_model()
    with torch.no_grad():
        m.head.bias.copy_(torch.tensor(bias))
    return m


def fake_run(d: Path, model=None, **meta) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    sha = save_artifact(model if model is not None else zero_model(), d / "artifact.dlw", meta)
    (d / "artifact.sha256").write_text(f"{sha}  artifact.dlw\n")
    return d


# -- run names, artefacts ------------------------------------------------------------------


def test_split_is_val():
    assert scoring.SPLIT == "val" == v03_scoring.SPLIT


@pytest.mark.parametrize("bad", ["", "../r1", "r1/x", "-r", "a" * 65, "r 1"])
def test_bad_run_names(bad):
    with pytest.raises(scoring.ScoringError):
        scoring.check_run_name(bad)
    assert scoring.check_run_name("r1") == "r1"


def test_run_dir_needs_artifact_and_sha(tmp_path):
    with pytest.raises(FileNotFoundError):
        scoring.run_dir("r1", tmp_path)
    fake_run(tmp_path / "r1")
    assert scoring.run_dir("r1", tmp_path) == tmp_path / "r1"


def test_read_sha256_file(tmp_path):
    h = "a" * 64
    (tmp_path / "x.sha256").write_text(f"{h}  artifact.dlw\n")
    assert scoring.read_sha256_file(tmp_path / "x.sha256", "artifact.dlw") == h
    (tmp_path / "y.sha256").write_text(f"{h}\n")
    assert scoring.read_sha256_file(tmp_path / "y.sha256", "artifact.dlw") == h
    with pytest.raises(scoring.ScoringError, match="no digest"):
        scoring.read_sha256_file(tmp_path / "x.sha256", "other.dlw")
    (tmp_path / "z.sha256").write_text("nothex  artifact.dlw\n")
    with pytest.raises(scoring.ScoringError, match="not a sha256"):
        scoring.read_sha256_file(tmp_path / "z.sha256", "artifact.dlw")


def test_load_run_artifact_checks_sha_and_run(tmp_path):
    d = fake_run(tmp_path / "r1", head_bias_model([0.0, 0.0, 1.0, 0.0]), run="r1")
    spec, info = scoring.load_run_artifact(d / "artifact.dlw", d / "artifact.sha256", run="r1")
    assert spec.name == scoring.NET_COLUMN and spec.sha256 == info["artifact_sha256"]
    assert info["nonzero_parameters"]["nonzero"] == 1 and info["nonzero_per_tensor"]["head.bias"] == 1
    assert info["artifact_metadata"] == {"run": "r1"}
    with pytest.raises(scoring.ScoringError, match="metadata run"):
        scoring.load_run_artifact(d / "artifact.dlw", d / "artifact.sha256", run="r2")
    (d / "artifact.sha256").write_text(f"{'0' * 64}  artifact.dlw\n")
    with pytest.raises(scoring.ScoringError, match="sha256"):
        scoring.load_run_artifact(d / "artifact.dlw", d / "artifact.sha256")


def test_official_artifact_must_name_the_run(tmp_path):
    d = fake_run(tmp_path / "r1", head_bias_model([0.0, 0.0, 1.0, 0.0]), phase="A")  # e.g. a copied v0.3 artefact
    with pytest.raises(scoring.ScoringError, match="metadata run None"):
        scoring.load_run_artifact(d / "artifact.dlw", d / "artifact.sha256", run="r1")
    scoring.load_run_artifact(d / "artifact.dlw", d / "artifact.sha256")  # dry runs do not need it


def test_check_apply_report(tmp_path):
    d = fake_run(tmp_path / "r1", run="r1")
    sha = (d / "artifact.sha256").read_text().split()[0]
    with pytest.raises(scoring.ScoringError, match="apply.json missing"):
        scoring.check_apply_report(d, sha)
    (d / "apply.json").write_text(json.dumps({"artifact": {"sha256": "0" * 64}}))
    with pytest.raises(scoring.ScoringError, match="artifact sha256"):
        scoring.check_apply_report(d, sha)
    (d / "apply.json").write_text(json.dumps({"edit_report": {}}))
    with pytest.raises(scoring.ScoringError, match="no artifact.sha256"):
        scoring.check_apply_report(d, sha)
    (d / "apply.json").write_text(json.dumps({"artifact": {"sha256": sha}}))
    assert scoring.check_apply_report(d, sha) == sha


def _stub_val(monkeypatch, data):
    from functools import partial

    from agent_loop_lab.v03 import sealed

    monkeypatch.setattr(scoring.v03_scoring, "load_val", lambda *a, **k: data)
    monkeypatch.setattr(scoring.v04_scoring, "VAL_ROWS_SHA256", sealed.rows_sha256(data.rows))
    monkeypatch.setattr(scoring.v04_scoring, "checked_disclosed", lambda rows: frozenset(data.session_ids[:2]))
    monkeypatch.setattr(scoring, "score_network", partial(scoring.score_network, n_resamples=N_RES))


def test_cli_writes_a_dry_run_report(sv, tmp_path, monkeypatch, capsys):
    data = synthetic_data(n_sessions=6, n_events=10, seed=7)
    _stub_val(monkeypatch, data)
    d = fake_run(tmp_path / "r1", head_bias_model([0.0, 0.0, 1.0, 0.0]), run="r1")
    out_dir = tmp_path / "out"
    assert sv.main(["r1", "--artifact", str(d / "artifact.dlw"), "--report-dir", str(out_dir), "--skip-v03-a",
                    "--workers", "0"]) == 0
    rep = json.loads((out_dir / "v05_val_r1.json").read_text())
    assert rep["official"] is False and rep["split"] == "val" and rep["run"] == "r1"
    assert rep["inputs"]["artifact"]["sha256_checked_against"] == str(d / "artifact.sha256")
    assert rep["v03_phase_a_logits"] is None and rep["disclosed_sessions"]["count"] == 2
    for pop in scoring.POPULATIONS:
        assert set(rep["summary"][pop]["network_f1_per_class"]) == set(CLASS_NAMES)
    assert not (out_dir / "v05_val_r1.json.tmp").exists()
    assert "macro-F1" in capsys.readouterr().out


def test_report_refuses_another_encoding_than_v03_a(tmp_path, monkeypatch):
    data = synthetic_data(n_sessions=4, n_events=9, seed=8)  # cache_key {"fake": True}
    _stub_val(monkeypatch, data)
    a = fake_run(tmp_path / "phase-a")
    a_sha = (a / "artifact.sha256").read_text().split()[0]
    rep = tmp_path / "rep.json"
    rep.write_text(json.dumps({"inputs": {"artifact_sha256": a_sha, "dataset": {"fake": False}}}))
    monkeypatch.setattr(scoring, "V03_A_REPORT", rep)
    orig = scoring.v03_a_spec
    monkeypatch.setattr(scoring, "v03_a_spec", lambda: orig(a, rep))
    d = fake_run(tmp_path / "r1", run="r1")
    with pytest.raises(scoring.ScoringError, match="cache key"):
        scoring.report("r1", artifact=d / "artifact.dlw", sha_file=d / "artifact.sha256", official=False,
                       workers=0, progress=lambda m: None)


def test_v03_a_spec_checks_report(tmp_path):
    d = fake_run(tmp_path / "phase-a")
    sha = (d / "artifact.sha256").read_text().split()[0]
    rep = tmp_path / "rep.json"
    rep.write_text(json.dumps({"inputs": {"artifact_sha256": sha}}))
    spec, info = scoring.v03_a_spec(d, rep)
    assert spec.name == scoring.V03_A_COLUMN and info["artifact_sha256"] == sha
    rep.write_text(json.dumps({"inputs": {"artifact_sha256": "f" * 64}}))
    with pytest.raises(scoring.ScoringError, match="inputs.artifact_sha256"):
        scoring.v03_a_spec(d, rep)


def test_archived_v03_a_matches_its_report():
    spec, _ = scoring.v03_a_spec()
    assert spec.sha256 == json.loads(scoring.V03_A_REPORT.read_text())["inputs"]["artifact_sha256"]


# -- predictions ---------------------------------------------------------------------------


def test_logit_diagnostics():
    lg = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 2.0, 0.0], [1.0, 3.0, 0.0, -5.0], [float("nan"), 0, 0, 0],
                       [float("inf"), 0, 0, 0]])
    assert scoring.logit_diagnostics(lg) == (5, 2, 2, 1, 5.0)


def test_network_predictions_in_process(tmp_path):
    data = synthetic_data(n_sessions=5)
    d0 = fake_run(tmp_path / "zero")
    d2 = fake_run(tmp_path / "stalled", head_bias_model([0.0, 0.0, 1.0, 0.0]))
    s0, _ = scoring.load_run_artifact(d0 / "artifact.dlw", d0 / "artifact.sha256")
    s2, _ = scoring.load_run_artifact(d2 / "artifact.dlw", d2 / "artifact.sha256")
    s2 = type(s2)(scoring.V03_A_COLUMN, s2.kind, s2.path, s2.sha256)
    res = scoring.network_predictions([s0, s2], data, None, chunk_sessions=2)
    assert set(res) == {scoring.NET_COLUMN, scoring.V03_A_COLUMN}
    z, st = res[scoring.NET_COLUMN], res[scoring.V03_A_COLUMN]
    assert np.all(z.pred == 0) and np.all(st.pred == 2)  # all-zero logits tie -> PRODUCTIVE
    n_events = sum(len(s.eoe_index) for s in data.sessions)
    assert z.logits["events"] == n_events == z.logits["tied_top_events"] == z.logits["all_equal_events"]
    assert z.logits["nonfinite_events"] == 0 and z.logits["max_abs_finite_logit"] == 0.0
    assert st.logits["tied_top_events"] == 0 and st.logits["max_abs_finite_logit"] == 1.0
    assert z.latency["sessions"] == len(data.sessions)


def test_network_predictions_match_v03_path(tmp_path):
    data = synthetic_data(n_sessions=4)
    g = torch.Generator().manual_seed(0)
    m = zero_model()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(0.05 * torch.randn(p.shape, generator=g))
    d = fake_run(tmp_path / "rand", m)
    spec, _ = scoring.load_run_artifact(d / "artifact.dlw", d / "artifact.sha256")
    mine = scoring.network_predictions([spec], data, None)[scoring.NET_COLUMN].pred
    ref, _ = v03_scoring.network_predictions(m.eval(), data)
    assert np.array_equal(mine, ref)


# -- scoring -------------------------------------------------------------------------------


def _preds(data, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4, size=len(data.rows)).astype(np.int64)


def test_score_network_blocks_are_v03_metrics():
    data = synthetic_data(n_sessions=30, n_events=14, seed=3)
    net, a = _preds(data, 1), _preds(data, 2)
    disclosed = frozenset(data.session_ids[:3])
    body = scoring.score_network(data, net, a, disclosed=disclosed, n_resamples=N_RES)
    excl = [i for i, r in enumerate(data.rows) if r.session_id not in disclosed]
    for pop, idx in (("excluding_disclosed", excl), ("all_val", list(range(len(data.rows))))):
        sub = [data.rows[i] for i in idx]
        ref = metrics.score_report(sub, net[idx], a[idx], n_resamples=N_RES)
        rep = body[pop]
        assert rep["overall"]["network"] == ref["overall"]["network"]
        assert rep["overall"][scoring.A_BLOCK] == ref["overall"]["reference_policy"]
        assert rep["overall"][scoring.AGREE_BLOCK] == ref["overall"]["h2_fidelity"]
        for fam in ref["per_family"]:
            assert rep["per_family"][fam]["network"] == ref["per_family"][fam]["network"]
        assert "reference_policy" not in rep["overall"] and "policy_failures" not in rep
        assert "h2_fidelity" not in rep["definitions"] and scoring.AGREE_BLOCK in rep["definitions"]
        pm = rep["paired_macro_f1"]
        diff = pm[f"{scoring.NET_COLUMN} - {scoring.V03_A_COLUMN}"]
        nv, av = ref["overall"]["network"]["macro_f1"]["value"], ref["overall"]["reference_policy"]["macro_f1"]["value"]
        assert diff["value"] == pytest.approx(nv - av, abs=1e-12)
        s = body["summary"][pop]
        assert s["network"]["macro_f1"]["value"] == nv and s[scoring.A_BLOCK]["macro_f1"]["value"] == av
        assert set(s["network_f1_per_class"]) == set(CLASS_NAMES)
    assert body["disclosed_sessions"]["count"] == 3
    assert body["v03_phase_a_reproduction"] is None
    # the network block does not depend on the slot v0.3 gave the reference policy
    alone = scoring.score_network(data, net, None, disclosed=disclosed, n_resamples=N_RES)
    assert alone["excluding_disclosed"]["overall"]["network"] == body["excluding_disclosed"]["overall"]["network"]
    assert scoring.A_BLOCK not in alone["excluding_disclosed"]["overall"]
    assert list(alone["excluding_disclosed"]["paired_macro_f1"]) == [f"{scoring.NET_COLUMN} - {scoring.PRIOR_COLUMN}"]


def test_all_zero_network_equals_prior():
    data = synthetic_data(n_sessions=20, n_events=12, seed=5)
    net = np.zeros(len(data.rows), dtype=np.int64)
    body = scoring.score_network(data, net, None, disclosed=frozenset(), n_resamples=N_RES)
    d = body["all_val"]["paired_macro_f1"][f"{scoring.NET_COLUMN} - {scoring.PRIOR_COLUMN}"]
    assert d["value"] == 0.0 and d["ci95"] == [0.0, 0.0]


def test_score_network_refuses_missing_prediction():
    data = synthetic_data(n_sessions=6)
    net = _preds(data, 0)
    i = next(i for i, r in enumerate(data.rows) if r.gate_passed)
    net[i] = -1
    with pytest.raises(scoring.ScoringError, match="no v0.5 network prediction"):
        scoring.score_network(data, net, None, disclosed=frozenset(), n_resamples=N_RES)


def _recorded_a(data, a, disclosed):
    """A v0.3-style val report body with ``a`` as the network (as v03_score_validation writes it)."""
    return v03_scoring.score(data, a, a, n_resamples=N_RES, disclosed=disclosed)


def test_v03_a_reproduction_check():
    data = synthetic_data(n_sessions=24, n_events=13, seed=7)
    net, a = _preds(data, 1), _preds(data, 2)
    disclosed = frozenset(data.session_ids[:2])
    rec = _recorded_a(data, a, disclosed)
    body = scoring.score_network(data, net, a, disclosed=disclosed, recorded_a=rec, n_resamples=N_RES)
    r = body["v03_phase_a_reproduction"]
    assert r == {"reproduced": True, "first_difference": None, "blocks_compared": 2 * (1 + 2)}
    # one changed A class: recorded, not fatal, and the paired difference is flagged
    a2 = a.copy()
    i = next(i for i, row in enumerate(data.rows) if row.gate_passed and row.label.is_scored)
    a2[i] = (a2[i] + 1) % 4
    body = scoring.score_network(data, net, a2, disclosed=disclosed, recorded_a=rec, n_resamples=N_RES)
    r = body["v03_phase_a_reproduction"]
    assert not r["reproduced"] and r["first_difference"].startswith("val ")
    assert "caveat" in body["all_val"]["paired_macro_f1"][f"{scoring.NET_COLUMN} - {scoring.V03_A_COLUMN}"]
    # a different population or bootstrap is fatal
    bad = json.loads(json.dumps(rec))
    bad["all_val"]["bootstrap"]["seed"] = 1
    with pytest.raises(scoring.ScoringError, match="bootstrap"):
        scoring.score_network(data, net, a, disclosed=disclosed, recorded_a=bad, n_resamples=N_RES)
    bad = json.loads(json.dumps(rec))
    bad["disclosed_sessions"]["sha256_of_sorted_ids"] = "0" * 64
    with pytest.raises(scoring.ScoringError, match="disclosed"):
        scoring.score_network(data, net, a, disclosed=disclosed, recorded_a=bad, n_resamples=N_RES)


# -- CLI guards (none of these loads val) ---------------------------------------------------


def test_cli_guards(sv, tmp_path, capsys):
    with pytest.raises(SystemExit):
        sv.main(["r1", "--artifact", str(tmp_path / "a.dlw")])
    with pytest.raises(SystemExit):
        sv.main(["r1", "--report-dir", str(tmp_path)])
    d = fake_run(tmp_path / "r1")
    with pytest.raises(SystemExit):
        sv.main(["r1", "--artifact", str(d / "artifact.dlw"), "--report-dir", str(REPO / "artifacts" / "reports")])
    with pytest.raises(SystemExit):
        sv.main(["../r1", "--artifact", str(d / "artifact.dlw"), "--report-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        sv.main(["r1", "--skip-v03-a"])
    with pytest.raises(SystemExit):
        sv.main(["r1", "--split", "test"])
    assert sv.main(["no-such-run-v05-test"]) == 2
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "v05_val_r1.json").write_text("{}")
    assert sv.main(["r1", "--artifact", str(d / "artifact.dlw"), "--report-dir", str(out_dir)]) == 2
    assert "refusing" in capsys.readouterr().err
    assert (out_dir / "v05_val_r1.json").read_text() == "{}"


def test_cli_sha_file_beside(sv, tmp_path):
    d = fake_run(tmp_path / "r1")
    assert sv._sha_file_beside(d / "artifact.dlw") == d / "artifact.sha256"
    other = tmp_path / "loose.dlw"
    other.write_bytes((d / "artifact.dlw").read_bytes())
    assert sv._sha_file_beside(other) is None


def test_network_predictions_in_pool_match_in_process(tmp_path):
    from agent_loop_lab.v03 import sealed

    data = synthetic_data(n_sessions=3)
    d = fake_run(tmp_path / "r1", head_bias_model([0.0, 0.5, 0.0, 1.0]))
    spec, _ = scoring.load_run_artifact(d / "artifact.dlw", d / "artifact.sha256")
    local = scoring.network_predictions([spec], data, None, chunk_sessions=1)[scoring.NET_COLUMN]
    with sealed.make_pool(1) as pool:
        pooled = scoring.network_predictions([spec], data, pool, workers=1, chunk_sessions=1)[scoring.NET_COLUMN]
    assert np.array_equal(local.pred, pooled.pred) and np.all(pooled.pred == 3)
    assert local.logits == pooled.logits and pooled.latency["workers"] == 1
