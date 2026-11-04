"""v0.3 validation scoring and Phase B disclosure tooling (src/agent_loop_lab/v03/scoring.py,
scripts/v03_score_validation.py, scripts/v03_build_disclosure.py).

Fake archived phases only (zero model, trivial policies) under tmp_path. Most tests use synthetic
sessions; the integration tests read the first few validation sessions through ``load_val(limit=...)``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from agent_loop_lab.eval import metrics
from agent_loop_lab.eval.collisions import WindowRow
from agent_loop_lab.schema import CLASS_NAMES, OracleLabel
from agent_loop_lab.v03 import scoring, tokenizer
from agent_loop_lab.v03.artifact import save_artifact
from agent_loop_lab.v03.model import zero_model

REPO = Path(__file__).resolve().parents[1]
SCORED = [OracleLabel(n) for n in CLASS_NAMES]


def _load_script(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sv():
    return _load_script("v03_score_validation")


@pytest.fixture(scope="module")
def dl():
    return _load_script("v03_build_disclosure")


def _block(k: int, tool: str = "read_file") -> str:
    return f"00:00:{k:02d}.000 {tool} path=src/mod{k % 3}.py -> ok in=12 out=34"


def _session_blocks(n: int) -> list[str]:
    out = []
    for k in range(n):
        b = _block(k, "edit_file" if k % 4 == 1 else "read_file")
        if k % 4 == 1:
            b += f"\n  @ src/mod{k % 3}.py h{k:010x} > h{k + 1:010x}"
        out.append(b)
    return out


def synthetic_data(n_sessions: int = 12, n_events: int = 11, seed: int = 0) -> scoring.ValData:
    rng = np.random.default_rng(seed)
    ids, fams, sessions, rows, r_sess, r_pos = [], [], [], [], [], []
    for s in range(n_sessions):
        sid = f"sess-{s:03d}"
        ids.append(sid)
        fams.append(f"fam_{s % 2}")
        enc = tokenizer.encode_session(_session_blocks(n_events))
        sessions.append(scoring.EncodedSession(list(enc.tokens), list(enc.event_ord), list(enc.offset),
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
    return scoring.ValData(ids, fams, sessions, rows, np.array(r_sess), np.array(r_pos), {"cache_key": {"fake": True}})


def fake_archive(d: Path, phase: str = "v03-phase-a", policy: str = "def reference_policy(tokens, event_ord):\n    return 0\n",
                 model=None, **meta_overrides) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    pol = d / "reference_policy.py"
    pol.write_text(policy)
    meta = {
        "phase": phase,
        "spec_version": "0.3",
        "packet_sha256": yaml.safe_load((REPO / "experiments" / "registry_v0_3.yaml").read_text())["packet_sha256"],
        "reference_policy_sha256": hashlib.sha256(pol.read_bytes()).hexdigest(),
        **meta_overrides,
    }
    sha = save_artifact(model if model is not None else zero_model(), d / "artifact.dlw", meta)
    (d / "artifact.sha256").write_text(f"{sha}  artifact.dlw\n")
    return d


def fake_evaluate(value):
    def evaluate(policy_file, inputs):
        out = []
        for toks, eo in inputs:
            assert toks[0] == "<BOS>" and toks[-1] == "<EOE>" and len(toks) == len(eo)
            out.append(value(toks, eo) if callable(value) else value)
        return out

    return evaluate


# -- split ----------------------------------------------------------------------------------


def test_split_is_hard_coded(sv, dl):
    assert scoring.SPLIT == "val" and metrics.ALLOWED_SPLITS == frozenset({"val"})
    import inspect

    assert "split" not in inspect.signature(scoring.load_val).parameters
    for main in (sv.main, dl.main):
        with pytest.raises(SystemExit):
            main(["v03-phase-a", "--split", "test"] if main is sv.main else ["--split", "test"])
    rows = [WindowRow("s", "f", "test", 7, OracleLabel.PRODUCTIVE, True, None)]
    with pytest.raises(ValueError, match="may not be scored"):
        scoring.disclosure_selection(rows, per_class=1)


def test_cli_override_guards(sv, dl, tmp_path):
    with pytest.raises(SystemExit):
        sv.main(["v03-phase-a", "--weight-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        sv.main(["v03-phase-a", "--weight-dir", str(tmp_path), "--report-dir", str(REPO / "artifacts" / "reports")])
    with pytest.raises(SystemExit):
        dl.main(["--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        dl.main(["--weight-dir", str(tmp_path), "--out-dir", str(REPO / "experiments" / "v0_3" / "phase_b")])
    with pytest.raises(SystemExit):
        sv.main(["phase-a-strict"])


# -- disclosure rule ------------------------------------------------------------------------


def test_disclosure_key_format():
    assert scoring.disclosure_key("abc", 9) == hashlib.sha256(b"20261105:v03-phase-b-disclosure:abc:9").hexdigest()


def _rule_reference(rows, per_class):
    used, chosen = set(), []
    for c in CLASS_NAMES:
        cands = sorted((scoring.disclosure_key(r.session_id, r.endpoint), i) for i, r in enumerate(rows)
                       if r.gate_passed and r.label.value == c)
        n = 0
        for k, i in cands:
            if rows[i].session_id in used or n == per_class:
                continue
            used.add(rows[i].session_id)
            chosen.append((k, i))
            n += 1
    return sorted(chosen)


def test_disclosure_selection_one_window_per_session_across_classes():
    data = synthetic_data(n_sessions=60, n_events=20, seed=4)
    chosen = scoring.disclosure_selection(data.rows, per_class=5)
    assert chosen == _rule_reference(data.rows, 5)
    assert [k for k, _ in chosen] == sorted(k for k, _ in chosen)
    sids = [data.rows[i].session_id for _, i in chosen]
    assert len(set(sids)) == len(sids) == 20
    labels = [data.rows[i].label.value for _, i in chosen]
    assert all(labels.count(c) == 5 for c in CLASS_NAMES)
    for _, i in chosen:
        assert data.rows[i].gate_passed and data.rows[i].label.is_scored
    # a later class skips sessions an earlier class used: the per-class rule alone would pick differently
    per_class_only = {}
    for c in CLASS_NAMES:
        cands = sorted((scoring.disclosure_key(r.session_id, r.endpoint), r.session_id) for r in data.rows
                       if r.gate_passed and r.label.value == c)
        seen, take = set(), []
        for k, s in cands:
            if s not in seen:
                seen.add(s)
                take.append(k)
        per_class_only[c] = take[:5]
    assert sorted(k for v in per_class_only.values() for k in v) != [k for k, _ in chosen]
    assert scoring.disclosed_sessions(data.rows, per_class=5) == frozenset(sids)


def test_disclosure_selection_raises_when_short():
    data = synthetic_data(n_sessions=6, n_events=12)
    with pytest.raises(ValueError, match="need"):
        scoring.disclosure_selection(data.rows, per_class=5)


def test_disclosed_prefix_round_trip():
    blocks = _session_blocks(12)
    blocks[3] = "? read_file path=x -> ok in=? out=1\n  ! KeyError at x:12 (pid 1234)\n  [ workspace untracked]"
    text = scoring.render_disclosed_prefix(blocks, 9)
    lines = text.split("\n")
    assert lines.count(scoring.WINDOW_MARKER) == 1
    assert lines[lines.index(scoring.WINDOW_MARKER) + 1] == blocks[2].split("\n")[0]
    assert scoring.parse_disclosed_prefix(text) == (blocks[:10], 2)
    assert text.endswith(blocks[9])
    first = scoring.render_disclosed_prefix(blocks, 7)
    assert first.split("\n")[0] == scoring.WINDOW_MARKER
    with pytest.raises(ValueError):
        scoring.render_disclosed_prefix(blocks, 6)


# -- archive loading ------------------------------------------------------------------------


def test_phase_dir_and_load_phase_checks(tmp_path):
    with pytest.raises(ValueError, match="not a v0.3 phase"):
        scoring.phase_dir("phase-a-strict")
    with pytest.raises(FileNotFoundError):
        scoring.phase_dir("v03-phase-a", tmp_path / "missing")
    assert scoring.WEIGHT_ROOT / "phase-a" == REPO / "weight_program" / "v0_3" / "phase-a"
    d = fake_archive(tmp_path / "ok")
    model, meta, pol, shas = scoring.load_phase("v03-phase-a", scoring.phase_dir("v03-phase-a", d))
    assert pol == d / "reference_policy.py" and meta["phase"] == "v03-phase-a"
    with pytest.raises(ValueError, match="phase"):
        scoring.load_phase("v03-phase-b", d)
    bad = fake_archive(tmp_path / "badsha")
    (bad / "artifact.sha256").write_text("0" * 64 + "  artifact.dlw\n")
    with pytest.raises(ValueError, match="sha256"):
        scoring.load_phase("v03-phase-a", bad)
    swapped = fake_archive(tmp_path / "swapped")
    (swapped / "reference_policy.py").write_text("def reference_policy(tokens, event_ord):\n    return 1\n")
    with pytest.raises(ValueError, match="reference_policy_sha256"):
        scoring.load_phase("v03-phase-a", swapped)
    unreg = fake_archive(tmp_path / "unreg", packet_sha256="f" * 64)
    with pytest.raises(ValueError, match="registered"):
        scoring.load_phase("v03-phase-a", unreg)
    io = fake_archive(tmp_path / "io", policy="import os\n\ndef reference_policy(tokens, event_ord):\n    return 0\n")
    with pytest.raises(ValueError, match="static rules"):
        scoring.load_phase("v03-phase-a", io)


# -- predictions and metrics ----------------------------------------------------------------


def test_zero_model_predicts_productive_everywhere():
    data = synthetic_data(n_sessions=3)
    net, lat = scoring.network_predictions(zero_model(), data)
    assert (net == 0).all() and lat["sessions"] == 3


def test_policy_predictions_inputs_and_failures():
    data = synthetic_data(n_sessions=5)

    def value(toks, eo):
        n_events = eo[-1]
        return None if n_events == 9 else (n_events % 4)

    ref = scoring.policy_predictions(Path("unused.py"), data, evaluate=fake_evaluate(value), chunk_sessions=2)
    for i, r in enumerate(data.rows):
        pos = int(data.row_pos[i])
        if not r.gate_passed:
            assert ref[i] == -1
        elif pos + 1 == 9:
            assert ref[i] == -2
        else:
            assert ref[i] == (pos + 1) % 4
    with pytest.raises(ValueError):
        scoring.policy_predictions(Path("unused.py"), data, workers=3, evaluate=fake_evaluate(0))


def test_policy_prefix_ends_at_eoe_t():
    data = synthetic_data(n_sessions=1, n_events=10)
    seen = []

    def value(toks, eo):
        seen.append((len(toks), eo[-1]))
        return 0

    scoring.policy_predictions(Path("unused.py"), data, evaluate=fake_evaluate(value))
    enc = data.sessions[0]
    expected = [(enc.eoe_index[int(data.row_pos[i])] + 1, int(data.row_pos[i]) + 1)
                for i, r in enumerate(data.rows) if r.gate_passed]
    assert seen == expected


def test_policy_predictions_real_guarded_child(tmp_path):
    data = synthetic_data(n_sessions=3, n_events=9)
    pol = tmp_path / "reference_policy.py"
    pol.write_text(
        "def reference_policy(tokens, event_ord):\n"
        "    if event_ord[-1] == 9:\n"
        "        while True:\n"
        "            pass\n"
        "    return len(tokens) % 4\n"
    )
    ref = scoring.policy_predictions(pol, data, chunk_sessions=2)
    for i, r in enumerate(data.rows):
        if not r.gate_passed:
            assert ref[i] == -1
        elif int(data.row_pos[i]) == 8:
            assert ref[i] == -2  # timeout
        else:
            assert ref[i] == (data.sessions[0].eoe_index[int(data.row_pos[i])] + 1) % 4


def test_fill_policy_failures_rule():
    rows = [WindowRow("s", "f", "val", 7, OracleLabel.PRODUCTIVE, True, None),
            WindowRow("s", "f", "val", 8, OracleLabel.RECOVERY, True, None),
            WindowRow("s", "f", "val", 9, OracleLabel.INSUFFICIENT_AMBIGUOUS, True, None),
            WindowRow("s", "f", "val", 10, OracleLabel.STALLED, True, None)]
    net = np.array([1, 0, 0, 2])
    ref = np.array([-2, -2, -2, 3])
    assert scoring.fill_policy_failures(rows, net, ref).tolist() == [2, 2, 1, 3]


def test_score_without_failures_matches_metrics_and_counts_failures():
    data = synthetic_data(n_sessions=10, n_events=14, seed=2)
    rng = np.random.default_rng(1)
    gate = np.array([r.gate_passed for r in data.rows])
    net = np.where(gate, rng.integers(0, 4, len(data.rows)), -1)
    ref = np.where(gate, rng.integers(0, 4, len(data.rows)), -1)
    disclosed = frozenset({"sess-001", "sess-004"})
    rep = scoring.score(data, net, ref, n_resamples=50, disclosed=disclosed)
    direct = metrics.score_report(data.rows, net, ref, n_resamples=50, seed=metrics.BOOTSTRAP_SEED)
    for k in ("overall", "per_family", "population"):
        assert rep["all_val"][k] == direct[k]
    keep = [i for i, r in enumerate(data.rows) if r.session_id not in disclosed]
    sub = metrics.score_report([data.rows[i] for i in keep], net[keep], ref[keep], n_resamples=50,
                               seed=metrics.BOOTSTRAP_SEED)
    assert rep["excluding_disclosed"]["overall"] == sub["overall"]
    assert rep["excluding_disclosed"]["population"]["sessions"] == 8
    assert rep["summary"]["all_val"]["policy_failures"] == 0
    # failures: every gate-passing ref fails -> H2 agreement 0, policy accuracy 0
    failed = np.where(gate, -2, -1)
    rep2 = scoring.score(data, net, failed, n_resamples=20, disclosed=disclosed)
    o = rep2["all_val"]["overall"]
    assert o["h2_fidelity"]["all"]["agree"] == 0
    assert o["reference_policy"]["accuracy"]["value"] == 0.0
    assert rep2["all_val"]["policy_failures"]["total"] == int(gate.sum())
    assert rep2["all_val"]["policy_failures"]["calls"] == int(gate.sum())


def test_score_phase_end_to_end_synthetic(sv, tmp_path, monkeypatch):
    data = synthetic_data(n_sessions=8, n_events=12, seed=5)
    d = fake_archive(tmp_path / "arch")
    monkeypatch.setattr(scoring, "disclosed_sessions", lambda rows: frozenset({"sess-000"}))
    rep = sv.score_phase("v03-phase-a", data, weight_dir=d, n_resamples=20, progress=lambda m: None)
    assert rep["official"] is False and rep["split"] == "val"
    for pop in ("excluding_disclosed", "all_val"):
        net = rep[pop]["overall"]["network"]
        assert all(net["per_class"][c]["predicted"] == 0 for c in CLASS_NAMES[1:])
        assert net["false_alarm_rate"]["value"] == 0.0
        assert rep[pop]["overall"]["h2_fidelity"]["all"]["agreement"]["value"] == 1.0
    assert rep["nonzero_parameters"]["nonzero"] == 0
    assert rep["disclosed_sessions"]["count"] == 1


# -- disclosure builder ---------------------------------------------------------------------


def test_build_disclosure_no_leakage(dl, tmp_path, monkeypatch):
    data = synthetic_data(n_sessions=40, n_events=14, seed=7)
    blocks = {sid: _session_blocks(14) for sid in data.session_ids}
    monkeypatch.setattr(dl, "_blocks_for", lambda ids: {s: blocks[s] for s in ids})
    real = scoring.disclosure_selection
    monkeypatch.setattr(scoring, "disclosure_selection", lambda rows: real(rows, per_class=3))
    d = fake_archive(tmp_path / "arch", policy="def reference_policy(tokens, event_ord):\n    return 2\n")
    public, private = dl.build(data, d)
    assert [w["id"] for w in public["windows"]] == [f"V{n:03d}" for n in range(1, 13)]
    assert set(public) == {"format", "class_names", "window_marker", "fields", "windows"}
    for w in public["windows"]:
        assert set(w) == {"id", "prefix", "oracle_label", "network_class", "reference_policy_class"}
        assert w["network_class"] == "PRODUCTIVE" and w["reference_policy_class"] == "STALLED"
        parsed, marked = scoring.parse_disclosed_prefix(w["prefix"])
        assert marked == len(parsed) - 8
    text = json.dumps(public)
    for sid in data.session_ids:
        assert sid not in text
    for fam in set(data.families):
        assert fam not in text
    assert [p["key"] for p in private["windows"]] == sorted(p["key"] for p in private["windows"])
    assert len({p["session_id"] for p in private["windows"]}) == 12
    for w, p in zip(public["windows"], private["windows"]):
        assert w["id"] == p["id"] and w["oracle_label"] == p["oracle_label"]
        i = next(i for i, r in enumerate(data.rows) if r.session_id == p["session_id"] and r.endpoint == p["endpoint"])
        assert w["prefix"] == scoring.render_disclosed_prefix(blocks[p["session_id"]], int(data.row_pos[i]))


def test_build_disclosure_refuses_overwrite(dl, tmp_path):
    out = tmp_path / "out"
    (out / "disclosure").mkdir(parents=True)
    (out / "disclosure" / "DISCLOSURE.json").write_text("{}")
    d = fake_archive(tmp_path / "arch")
    assert dl.main(["--weight-dir", str(d), "--out-dir", str(out)]) == 2


def test_disclosure_paths_are_the_frozen_launchers(dl, sv):
    """README §2 Records (r1): the builder writes where scripts/v03_launch.sh phase-b reads."""
    launcher = (REPO / "scripts" / "v03_launch.sh").read_text()
    pub, sha, prv = dl.output_paths(dl.OUT_ROOT)
    assert pub == REPO / "experiments" / "v0_3" / "disclosure" / "DISCLOSURE.json"
    assert sha == REPO / "experiments" / "v0_3" / "disclosure" / "DISCLOSURE.sha256"
    for p in (pub, sha):
        assert f'"$REPO/{p.relative_to(REPO).as_posix()}"' in launcher
    assert prv == sv.PRIVATE_DISCLOSURE
    assert prv.parent != pub.parent  # the private record is never inside the handed-over directory
    prep = (REPO / "scripts" / "v03_prepare_phase_b.sh").read_text()
    assert 'SRC="$REPO/experiments/v0_3/disclosure"' in prep


def test_build_disclosure_refuses_official_out_dir(dl, tmp_path):
    d = fake_archive(tmp_path / "arch")
    with pytest.raises(SystemExit):
        dl.main(["--weight-dir", str(d), "--out-dir", str(REPO / "experiments" / "v0_3")])


# -- integration on the first validation sessions ------------------------------------------


def test_load_val_limit_matches_windows_gate():
    from agent_loop_lab.datasets.build import iter_sessions
    from agent_loop_lab.eval.collisions import session_windows

    data = scoring.load_val(limit=15)
    assert len(data.sessions) == 15
    expected = []
    for n, (events, record) in enumerate(iter_sessions(scoring.V02_DATA, "val")):
        if n == 15:
            break
        expected.extend(session_windows(events, record, "val"))
    assert [(r.session_id, r.endpoint, r.label, r.gate_passed, r.family) for r in data.rows] == \
           [(r.session_id, r.endpoint, r.label, r.gate_passed, r.family) for r in expected]
    for s, enc in enumerate(data.sessions):
        idx = data.rows_of(s)
        assert len(enc.eoe_index) == int(data.row_pos[idx].max()) + 1
        assert enc.tokens[enc.eoe_index[-1]] == tokenizer.EOE


def test_zero_model_on_real_val_prefix(sv, tmp_path, monkeypatch):
    data = scoring.load_val(limit=10)
    monkeypatch.setattr(scoring, "disclosed_sessions", lambda rows: frozenset({data.session_ids[0]}))
    d = fake_archive(tmp_path / "arch")
    rep = sv.score_phase("v03-phase-a", data, weight_dir=d, n_resamples=20, progress=lambda m: None)
    o = rep["all_val"]["overall"]
    assert o["network"]["per_class"]["PRODUCTIVE"]["predicted"] == o["network"]["scored_windows"]
    assert o["h2_fidelity"]["all"]["agreement"]["value"] == 1.0
    assert rep["all_val"]["policy_failures"]["total"] == 0
