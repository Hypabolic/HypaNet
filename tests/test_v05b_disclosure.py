"""v0.5 Phase B disclosure builder (scripts/v05b_build_disclosure.py).

Synthetic sessions, fake run directories and fake v0.3 records under tmp_path only: no test loads
the validation split or reads experiments/v0_5/runs/.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agent_loop_lab.eval.collisions import WindowRow
from agent_loop_lab.schema import CLASS_NAMES, OracleLabel
from agent_loop_lab.v03 import scoring as v03_scoring
from agent_loop_lab.v03 import tokenizer
from agent_loop_lab.v03.artifact import save_artifact
from agent_loop_lab.v03.model import zero_model
from agent_loop_lab.v05 import scoring as v05_scoring

REPO = Path(__file__).resolve().parents[1]
SCORED = [OracleLabel(n) for n in CLASS_NAMES]
PER_CLASS = 3
N_EVENTS = 14


def _load_script(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def db():
    return _load_script("v05b_build_disclosure")


def _session_blocks(n: int) -> list[str]:
    out = []
    for k in range(n):
        tool = "edit_file" if k % 4 == 1 else "read_file"
        b = f"00:00:{k:02d}.000 {tool} path=src/mod{k % 3}.py -> ok in=12 out=34"
        if k % 4 == 1:
            b += f"\n  @ src/mod{k % 3}.py h{k:010x} > h{k + 1:010x}"
        out.append(b)
    return out


def synthetic_data(n_sessions: int = 40, seed: int = 7) -> v03_scoring.ValData:
    rng = np.random.default_rng(seed)
    ids, fams, sessions, rows, r_sess, r_pos = [], [], [], [], [], []
    enc = tokenizer.encode_session(_session_blocks(N_EVENTS))
    for s in range(n_sessions):
        sid = f"sess-{s:03d}"
        ids.append(sid)
        fams.append(f"fam_{s % 2}")
        sessions.append(v03_scoring.EncodedSession(list(enc.tokens), list(enc.event_ord), list(enc.offset),
                                                   list(enc.eoe_index), list(enc.surface)))
        for pos in range(7, N_EVENTS):
            u = rng.random()
            label = OracleLabel.INSUFFICIENT_AMBIGUOUS if u < 0.1 else SCORED[int(rng.integers(0, 4))]
            gate = not (u > 0.95)
            if not gate:
                label = OracleLabel.INSUFFICIENT_INSTRUMENTATION
            rows.append(WindowRow(sid, fams[-1], "val", pos, label, gate, None))
            r_sess.append(s)
            r_pos.append(pos)
    return v03_scoring.ValData(ids, fams, sessions, rows, np.array(r_sess), np.array(r_pos), {"cache_key": {"fake": True}})


@pytest.fixture()
def synth(db, monkeypatch):
    """Synthetic val data, its blocks, and v0.3-style records of the same selection (with v0.3 'predictions')."""
    data = synthetic_data()
    blocks = {sid: _session_blocks(N_EVENTS) for sid in data.session_ids}
    monkeypatch.setattr(db, "_blocks_for", lambda ids: {s: blocks[s] for s in ids})
    real = v03_scoring.disclosure_selection
    monkeypatch.setattr(v03_scoring, "disclosure_selection", lambda rows: real(rows, per_class=PER_CLASS))
    pub, prv = [], []
    for n, (key, i) in enumerate(real(data.rows, per_class=PER_CLASS), start=1):
        r = data.rows[i]
        vid = f"V{n:03d}"
        pub.append({"id": vid, "prefix": v03_scoring.render_disclosed_prefix(blocks[r.session_id], int(data.row_pos[i])),
                    "oracle_label": r.label.to_class().name, "network_class": "STALLED",
                    "reference_policy_class": "RECOVERY"})
        prv.append({"id": vid, "session_id": r.session_id, "endpoint": r.endpoint, "block": int(data.row_pos[i]),
                    "family": r.family, "oracle_label": r.label.to_class().name, "key": key})
    return data, blocks, pub, prv


def _write_v03_records(d: Path, pub: list, prv: list) -> tuple[Path, Path, Path]:
    d.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps({"format": "v03-phase-b-disclosure/1", "windows": pub}, indent=1, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    (d / "DISCLOSURE.json").write_bytes(raw)
    (d / "DISCLOSURE.sha256").write_text(f"{digest}  DISCLOSURE.json\n")
    (d / "DISCLOSURE_PRIVATE.json").write_text(json.dumps({"disclosure_sha256": digest, "windows": prv}))
    return d / "DISCLOSURE.json", d / "DISCLOSURE.sha256", d / "DISCLOSURE_PRIVATE.json"


def _model(cls: int | None = None):
    m = zero_model()
    if cls is not None:
        with torch.no_grad():
            m.head.bias[cls] = 1.0
    m.eval()
    return m


def fake_run(d: Path, model=None, run: str | None = None, apply_sha: str | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    meta = {"experiment": "v0.5", "builder": "edits reader", "run": run or d.name}
    sha = save_artifact(model if model is not None else zero_model(), d / "artifact.dlw", meta)
    (d / "artifact.sha256").write_text(f"{sha}  artifact.dlw\n")
    (d / "apply.json").write_text(json.dumps({"artifact": {"file": "artifact.dlw", "sha256": apply_sha or sha}}))
    return d


SOURCE = {"run": "r9", "artifact_sha256": "ab" * 32}


# -- build ---------------------------------------------------------------------------------


def test_build_public_shape_and_no_leakage(db, synth):
    data, _blocks, pub, prv = synth
    public, private = db.build(data, _model(3), SOURCE, pub, prv)
    assert set(public) == {"format", "class_names", "window_marker", "network", "selection", "reference_policy",
                           "fields", "windows"}
    assert public["format"] == db.FORMAT and public["class_names"] == list(CLASS_NAMES)
    assert public["network"]["run"] == "r9" and public["network"]["artifact_sha256"] == "ab" * 32
    assert set(public["fields"]) == {"prefix", "oracle_label", "network_class"}
    assert "no reference_policy_class" in public["reference_policy"]
    assert [w["id"] for w in public["windows"]] == [f"V{n:03d}" for n in range(1, 4 * PER_CLASS + 1)]
    for w, ref in zip(public["windows"], pub):
        assert set(w) == {"id", "prefix", "oracle_label", "network_class"}
        assert w["prefix"] == ref["prefix"] and w["oracle_label"] == ref["oracle_label"]
        assert w["network_class"] == "OSCILLATING"  # from the model, never v0.3's STALLED
        parsed, marked = v03_scoring.parse_disclosed_prefix(w["prefix"])
        assert marked == len(parsed) - 8
    text = json.dumps(public)
    for sid in data.session_ids:
        assert sid not in text
    for fam in set(data.families):
        assert fam not in text
    assert "reference_policy_class\"" not in text.replace("no reference_policy_class field", "")
    assert [p["session_id"] for p in private["windows"]] == [p["session_id"] for p in prv]
    assert private["source"] == SOURCE


def test_build_network_class_follows_the_model(db, synth):
    data, _blocks, pub, prv = synth
    public, _ = db.build(data, _model(), SOURCE, pub, prv)
    assert {w["network_class"] for w in public["windows"]} == {"PRODUCTIVE"}  # all-zero: ties go to index 0
    public2, _ = db.build(data, _model(1), SOURCE, pub, prv)
    assert {w["network_class"] for w in public2["windows"]} == {"RECOVERY"}


@pytest.mark.parametrize("field,value", [("endpoint", 999), ("session_id", "sess-999"), ("key", "0" * 64)])
def test_build_refuses_different_selection(db, synth, field, value):
    data, _blocks, pub, prv = synth
    prv = [dict(w) for w in prv]
    prv[4][field] = value
    with pytest.raises(db.DisclosureError, match="first at V005"):
        db.build(data, _model(), SOURCE, pub, prv)
    with pytest.raises(db.DisclosureError, match="selected windows differ"):
        db.build(data, _model(), SOURCE, pub, prv[:-1])


def test_build_refuses_different_prefix_or_label(db, synth):
    data, _blocks, pub, prv = synth
    bad = [dict(w) for w in pub]
    bad[2]["prefix"] += " "
    with pytest.raises(db.DisclosureError, match="V003: prefix differs"):
        db.build(data, _model(), SOURCE, bad, prv)
    bad = [dict(w) for w in pub]
    bad[6]["oracle_label"] = "RECOVERY" if bad[6]["oracle_label"] != "RECOVERY" else "STALLED"
    with pytest.raises(db.DisclosureError, match="V007: oracle_label differs"):
        db.build(data, _model(), SOURCE, bad, prv)


def test_build_refuses_rerendered_blocks_that_differ(db, synth, monkeypatch):
    data, blocks, pub, prv = synth
    other = {s: _session_blocks(N_EVENTS - 1) + ["00:00:59.000 read_file path=x -> ok in=1 out=1"] for s in blocks}
    monkeypatch.setattr(db, "_blocks_for", lambda ids: {s: other[s] for s in ids})
    with pytest.raises(db.DisclosureError, match="re-rendered blocks differ"):
        db.build(data, _model(), SOURCE, pub, prv)


# -- inputs --------------------------------------------------------------------------------


def test_load_v03_reference_checks_hashes_and_drops_predictions(db, synth, tmp_path):
    _data, _blocks, pub, prv = synth
    paths = _write_v03_records(tmp_path / "v03", pub, prv)
    ref_pub, ref_prv, hashes = db.load_v03_reference(*paths)
    assert all(set(w) == {"id", "prefix", "oracle_label"} for w in ref_pub)  # v0.3 predictions never read
    assert ref_prv[0]["session_id"] == prv[0]["session_id"]
    assert hashes["v03_disclosure_sha256"] == paths[1].read_text().split()[0]
    paths[1].write_text("0" * 64 + "  DISCLOSURE.json\n")
    with pytest.raises(db.DisclosureError, match="sha256"):
        db.load_v03_reference(*paths)
    paths = _write_v03_records(tmp_path / "v03b", pub, prv)
    rec = json.loads(paths[2].read_text())
    rec["disclosure_sha256"] = "1" * 64
    paths[2].write_text(json.dumps(rec))
    with pytest.raises(db.DisclosureError, match="disclosure_sha256"):
        db.load_v03_reference(*paths)


def test_real_v03_reference_has_200_windows(db):
    """The committed v0.3 records load and agree (hashes only; no val data)."""
    pub, prv, hashes = db.load_v03_reference()
    assert len(pub) == len(prv) == 200 == 4 * v03_scoring.PER_CLASS
    assert len({w["session_id"] for w in prv}) == 200
    assert hashes["v03_disclosure_sha256"] == db.V03_SHA.read_text().split()[0]


def test_load_run_checks(db, tmp_path):
    d = fake_run(tmp_path / "r1", _model(2))
    model, src = db.load_run(d)
    assert src["run"] == "r1" and src["dry_run"] is False
    assert src["artifact_sha256"] == (d / "artifact.sha256").read_text().split()[0]
    assert int(model.head.bias.argmax()) == 2
    bad = fake_run(tmp_path / "r2", run="r1")
    with pytest.raises(v05_scoring.ScoringError, match="metadata run"):
        db.load_run(bad)
    bad = fake_run(tmp_path / "r3", apply_sha="c" * 64)
    with pytest.raises(v05_scoring.ScoringError, match="apply.json"):
        db.load_run(bad)
    bad = fake_run(tmp_path / "r4")
    (bad / "artifact.sha256").write_text("d" * 64 + "  artifact.dlw\n")
    with pytest.raises(v05_scoring.ScoringError, match="sha256"):
        db.load_run(bad)
    bad = fake_run(tmp_path / "r5")
    (bad / "apply.json").unlink()
    with pytest.raises(db.DisclosureError, match="apply.json missing"):
        db.load_run(bad)


def test_load_any_artifact_needs_sha_file(db, tmp_path):
    d = tmp_path / "a"
    d.mkdir()
    save_artifact(zero_model(), d / "artifact.dlw", {"phase": "x"})
    with pytest.raises(db.DisclosureError, match="missing"):
        db.load_any_artifact(d / "artifact.dlw")
    d = fake_run(tmp_path / "b")
    (d / "apply.json").unlink()
    _model_, src = db.load_any_artifact(d / "artifact.dlw")
    assert src["dry_run"] is True and src["run"] is None


# -- CLI -----------------------------------------------------------------------------------


def test_cli_argument_guards(db, tmp_path):
    run = fake_run(tmp_path / "r1")
    for argv in ([], [str(run), "--artifact", str(run / "artifact.dlw"), "--out", str(tmp_path / "o")],
                 ["--artifact", str(run / "artifact.dlw")],
                 ["--artifact", str(run / "artifact.dlw"), "--out", str(db.OUT_ROOT)],
                 ["--artifact", str(run / "artifact.dlw"), "--out", str(db.OUT_ROOT / "x")],
                 [str(run), "--out", str(REPO / "experiments" / "v0_3" / "x")],
                 [str(run), "--out", str(REPO / "experiments" / "v0_5" / "runs" / "x")],
                 [str(run), "--out", str(run / "x")],
                 ["--split", "test"]):
        with pytest.raises(SystemExit):
            db.main(argv)


def test_cli_refuses_overwrite(db, tmp_path):
    run = fake_run(tmp_path / "r1")
    out = tmp_path / "out"
    (out / "disclosure").mkdir(parents=True)
    (out / "disclosure" / "DISCLOSURE.sha256").write_text("x")
    assert db.main([str(run), "--out", str(out)]) == 2
    assert (out / "disclosure" / "DISCLOSURE.sha256").read_text() == "x"


def test_cli_end_to_end(db, synth, tmp_path, monkeypatch, capsys):
    data, _blocks, pub, prv = synth
    v03 = _write_v03_records(tmp_path / "v03", pub, prv)
    monkeypatch.setattr(db, "V03_PUBLIC", v03[0])
    monkeypatch.setattr(db, "V03_SHA", v03[1])
    monkeypatch.setattr(db, "V03_PRIVATE", v03[2])
    monkeypatch.setattr(v03_scoring, "load_val", lambda **kw: data)
    run = fake_run(tmp_path / "r1", _model(2))
    out = tmp_path / "out"
    assert db.main([str(run), "--out", str(out)]) == 0
    pub_p, sha_p, prv_p = db.output_paths(out)
    raw = pub_p.read_bytes()
    assert sha_p.read_text() == f"{hashlib.sha256(raw).hexdigest()}  DISCLOSURE.json\n"
    public = json.loads(raw)
    assert {w["network_class"] for w in public["windows"]} == {"STALLED"}
    assert public["network"]["run"] == "r1"
    private = json.loads(prv_p.read_text())
    assert private["disclosure_sha256"] == hashlib.sha256(raw).hexdigest()
    assert private["v03_disclosure_sha256"] == v03[1].read_text().split()[0]
    assert prv_p.parent != pub_p.parent  # the private record is never in the handed-over directory
    out2 = tmp_path / "out2"
    assert db.main([str(run), "--out", str(out2)]) == 0
    assert db.output_paths(out2)[0].read_bytes() == raw  # deterministic
    assert db.main([str(run), "--out", str(out)]) == 2


def test_cli_mismatch_writes_nothing(db, synth, tmp_path, monkeypatch):
    data, _blocks, pub, prv = synth
    prv = [dict(w) for w in prv]
    prv[0]["endpoint"] = 999
    v03 = _write_v03_records(tmp_path / "v03", pub, prv)
    for name, p in zip(("V03_PUBLIC", "V03_SHA", "V03_PRIVATE"), v03):
        monkeypatch.setattr(db, name, p)
    monkeypatch.setattr(v03_scoring, "load_val", lambda **kw: data)
    run = fake_run(tmp_path / "r1")
    out = tmp_path / "out"
    assert db.main([str(run), "--out", str(out)]) == 1
    assert not any(p.exists() for p in db.output_paths(out))
