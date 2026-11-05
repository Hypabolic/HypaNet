"""v0.4 WSP-B disclosure (PREREGISTRATION §7.1): agent_loop_lab.v04.disclosure and scripts/v04_build_disclosure.py.

No dataset file is read. The 200 windows v0.3 already disclosed (experiments/v0_3/disclosure/DISCLOSURE.json and
DISCLOSURE_PRIVATE.json, public to the author since v0.3 Phase B) stand in for the val rows: their prefixes give
the blocks, their private records the rows. The WSP-A column and archive are fakes; nothing is scored."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from agent_loop_lab.eval.collisions import WindowRow
from agent_loop_lab.schema import CLASS_NAMES, OracleLabel
from agent_loop_lab.v03 import scoring as v03_scoring
from agent_loop_lab.v04 import disclosure as D

REPO = Path(__file__).resolve().parents[1]
V03 = REPO / "experiments" / "v0_3"
EDIT_HASH = "e" * 64


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture(scope="module")
def v03():
    pub = json.loads((V03 / "disclosure" / "DISCLOSURE.json").read_text())
    prv = json.loads((V03 / "DISCLOSURE_PRIVATE.json").read_text())
    return pub, prv


def _inputs(v03, pred_of=lambda n: n % 4):
    """Rows, positions, a fake column argmax and the blocks of the 200 v0.3 disclosed windows."""
    pub, prv = v03
    rows, pos, pred, blocks, wpos = [], [], [], {}, {}
    for n, (p, w) in enumerate(zip(pub["windows"], prv["windows"])):
        b, marked = v03_scoring.parse_disclosed_prefix(p["prefix"])
        assert marked + 7 == len(b) - 1 == w["block"]
        rows.append(WindowRow(session_id=w["session_id"], family=w["family"], split="val", endpoint=w["endpoint"],
                              label=OracleLabel(w["oracle_label"]), gate_passed=True, vector=None))
        pos.append(w["block"])
        pred.append(pred_of(n))
        blocks[w["session_id"]] = b + ["later block, never disclosed"]
        wpos[w["session_id"]] = [w["block"]]
    source = {"wsp_a": {"edit_state_sha256": EDIT_HASH, "build_record_sha256": "b" * 64}, "column": {"sha256": "c" * 64}}
    return rows, np.array(pos), np.array(pred), blocks, wpos, source


def test_reproduces_the_v03_windows_with_the_wsp_a_class(v03):
    pub3, prv3 = v03
    rows, pos, pred, blocks, wpos, source = _inputs(v03)
    public, windows, private = D.build(rows, pos, pred, blocks, wpos, pub3, prv3, source)
    assert public["format"] == "v04-wsp-b-disclosure/1" and public["class_names"] == list(CLASS_NAMES)
    assert public["window_marker"] == "### window start"
    assert set(public["fields"]) == {"prefix", "oracle_label", "network_class"}
    assert EDIT_HASH in public["fields"]["network_class"]
    assert len(public["windows"]) == 200
    for n, (w, ref) in enumerate(zip(public["windows"], pub3["windows"])):
        assert set(w) == {"id", "prefix", "oracle_label", "network_class"}  # reference_policy_class is dropped
        assert w["id"] == ref["id"] == f"V{n + 1:03d}"
        assert w["prefix"] == ref["prefix"] and w["oracle_label"] == ref["oracle_label"]
        assert w["network_class"] == CLASS_NAMES[n % 4]
    assert {c: sum(w["oracle_label"] == c for w in public["windows"]) for c in CLASS_NAMES} == {c: 50 for c in CLASS_NAMES}
    text = json.dumps(public)
    assert not any(w["session_id"] in text for w in prv3["windows"])  # no session id, family or endpoint
    assert "reference_policy" not in text and "family" not in text
    assert [w["session_id"] for w in private["windows"]] == [w["session_id"] for w in prv3["windows"]]
    assert private["format"] == "v04-wsp-b-disclosure/1-private" and private["source"] == source


def test_windows_file_is_what_the_server_reads(v03):
    """hypaqwen.wsp.server.read_disclosed: {"windows": [{"id", "blocks", "pos"}]}, 0 <= pos < len(blocks)."""
    rows, pos, pred, blocks, wpos, source = _inputs(v03)
    public, windows, _ = D.build(rows, pos, pred, blocks, wpos, *v03, source)
    seen = set()
    for w, p in zip(windows["windows"], public["windows"]):
        assert {"id", "blocks", "pos"} <= set(w) and w["id"] == p["id"] and w["id"] not in seen
        seen.add(w["id"])
        assert w["pos"] == len(w["blocks"]) - 1  # blocks 0..t only: exactly the disclosed text
        assert v03_scoring.parse_disclosed_prefix(p["prefix"]) == (w["blocks"], w["pos"] - 7)
        assert "later block, never disclosed" not in w["blocks"]


def test_build_is_deterministic_and_the_files_hash_each_other(v03, tmp_path):
    rows, pos, pred, blocks, wpos, source = _inputs(v03)
    a = D.build(rows, pos, pred, blocks, wpos, *v03, source)
    b = D.build(rows, pos, pred, blocks, wpos, *v03, source)
    assert [D.dump(x) for x in a] == [D.dump(x) for x in b]
    shas = D.write_outputs(tmp_path, *a)
    paths = D.output_paths(tmp_path)
    assert _sha(paths["public"].read_bytes()) == shas["disclosure_sha256"]
    assert paths["sha"].read_text() == f"{shas['disclosure_sha256']}  DISCLOSURE.json\n"
    assert D.sha_field(paths["sha"]) == shas["disclosure_sha256"]
    prv = json.loads(paths["private"].read_text())
    assert prv["disclosure_sha256"] == shas["disclosure_sha256"]
    assert prv["disclosed_windows_sha256"] == _sha(paths["windows"].read_bytes())
    again = tmp_path / "again"
    D.write_outputs(again, *b)
    for k in paths:
        assert D.output_paths(again)[k].read_bytes() == paths[k].read_bytes()


def test_any_difference_from_v03_is_refused(v03):
    pub3, prv3 = v03
    rows, pos, pred, blocks, wpos, source = _inputs(v03)
    bad_prv = json.loads(json.dumps(prv3))
    bad_prv["windows"][5]["key"] = "0" * 64
    with pytest.raises(D.DisclosureError, match="V006: the selection differs"):
        D.build(rows, pos, pred, blocks, wpos, pub3, bad_prv, source)
    sid = rows[0].session_id  # rows are in v0.3 id order; V001's session
    altered = {**blocks, sid: [blocks[sid][0] + " x", *blocks[sid][1:]]}
    with pytest.raises(D.DisclosureError, match="prefix or label differs"):
        D.build(rows, pos, pred, altered, wpos, pub3, prv3, source)
    with pytest.raises(D.DisclosureError, match="does not end a window"):
        D.build(rows, pos, pred, blocks, {**wpos, sid: [0]}, pub3, prv3, source)
    with pytest.raises(D.DisclosureError, match="no prediction"):
        D.build(rows, pos, np.where(np.arange(200) == 3, -1, pred), blocks, wpos, pub3, prv3, source)
    with pytest.raises(D.DisclosureError, match="windows selected"):
        D.build(rows, pos, pred, blocks, wpos, pub3, {**prv3, "windows": prv3["windows"][:-1]}, source)


def test_v03_disclosure_files_agree():
    pub_b = (V03 / "disclosure" / "DISCLOSURE.json").read_bytes()
    pub, prv = json.loads(pub_b), json.loads((V03 / "DISCLOSURE_PRIVATE.json").read_text())
    sha_file = D.sha_field(V03 / "disclosure" / "DISCLOSURE.sha256")
    D.check_v03_disclosure(pub, prv, _sha(pub_b), sha_file)
    with pytest.raises(D.DisclosureError):
        D.check_v03_disclosure(pub, prv, "0" * 64, sha_file)


# -- the WSP-A archive and column -----------------------------------------------------------------------------


FREEZE_SHA = "f" * 64


def freeze_json(freeze_sha: str | None = FREEZE_SHA, **over) -> dict:
    """What scripts/v04_run_build.py writes (handover.freeze_record) for a WSP-A build with two launches."""
    rec = {"phase": "wsp-a", "freeze_sha256": freeze_sha, "unfrozen": freeze_sha is None, "consistent": True,
           "launches": {"wsp-a": freeze_sha, "wsp-a.c1": freeze_sha}}
    return {**rec, **over}


def fake_archive(d: Path, *, phase: str = "wsp-a", stub: bool = False, freeze: bool = True,
                 freeze_sha: str | None = FREEZE_SHA) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in D.EDIT_FILES:
        (d / name).write_bytes(name.encode())
        files[name] = _sha(name.encode())
    rec = {"record_format_version": 2, "phase": phase, "edit_state_sha256": EDIT_HASH, "files": files,
           "tensors": {"model.language_model.layers.3.mlp.down_proj.weight": {}},
           "checks": {"journal_checked": True, "build_guard": {"stub": stub}}}
    (d / D.RECORD_FILE).write_text(json.dumps(rec))
    (d / D.RECORD_SHA_FILE).write_text(f"{_sha((d / D.RECORD_FILE).read_bytes())}  {D.RECORD_FILE}\n")
    if freeze:
        (d / D.FREEZE_FILE).write_text(json.dumps(freeze_json(freeze_sha)))
    return d


def test_wsp_a_archive_check(tmp_path):
    info = D.check_wsp_a_archive(fake_archive(tmp_path / "a"))
    assert info["edit_state_sha256"] == EDIT_HASH and info["freeze_sha256"] == FREEZE_SHA
    assert info["freeze_launches"] == {"wsp-a": FREEZE_SHA, "wsp-a.c1": FREEZE_SHA} and info["unfrozen"] is False
    assert info["freeze_json_sha256"] == _sha((tmp_path / "a" / D.FREEZE_FILE).read_bytes())
    assert info["build_record_sha256"] == _sha((tmp_path / "a" / D.RECORD_FILE).read_bytes())
    for kw, why in (({"phase": "wsp-b"}, "phase"), ({"stub": True}, "real guard"), ({"freeze": False}, "FREEZE.json")):
        with pytest.raises(D.DisclosureError, match=why):
            D.check_wsp_a_archive(fake_archive(tmp_path / str(kw), **kw))
    d = fake_archive(tmp_path / "tampered")
    (d / "edit.safetensors").write_bytes(b"other")
    with pytest.raises(D.DisclosureError, match="edit.safetensors"):
        D.check_wsp_a_archive(d)
    d = fake_archive(tmp_path / "record")
    (d / D.RECORD_FILE).write_text((d / D.RECORD_FILE).read_text() + " ")
    with pytest.raises(D.DisclosureError, match="BUILD_RECORD.sha256"):
        D.check_wsp_a_archive(d)
    with pytest.raises(D.DisclosureError, match="not archived"):
        D.check_wsp_a_archive(tmp_path / "none")


@pytest.mark.parametrize("text, why", [
    ("{}", "phase"), ("[]", "not a freeze record"), ('"x"', "not a freeze record"), ("not json", "unreadable"),
    (json.dumps(freeze_json(None)), "no registry freeze_sha256"),
    (json.dumps(freeze_json(unfrozen=True)), "built unfrozen"),
    (json.dumps(freeze_json(consistent=False)), "another freeze"),
    (json.dumps({**freeze_json(), "consistent": None}), "another freeze"),
    (json.dumps(freeze_json(launches={})), "no launch records"),
    (json.dumps(freeze_json(launches=None)), "no launch records"),
    (json.dumps(freeze_json(launches={"wsp-a": None})), "without a freeze_sha256"),
    (json.dumps(freeze_json(phase="wsp-b")), "phase"),
])
def test_freeze_record_fails_closed(tmp_path, text, why):
    """PREREGISTRATION §7.3: an archive whose FREEZE.json does not show a frozen, consistent run is refused."""
    d = fake_archive(tmp_path / "a")
    (d / D.FREEZE_FILE).write_text(text)
    with pytest.raises(D.DisclosureError, match=why):
        D.check_wsp_a_archive(d)


def test_unfrozen_archive_only_in_kit_mode(tmp_path):
    d = fake_archive(tmp_path / "a", freeze_sha=None)
    with pytest.raises(D.DisclosureError, match="no registry freeze_sha256"):
        D.check_wsp_a_archive(d)
    assert D.check_wsp_a_archive(d, allow_unfrozen=True)["unfrozen"] is True
    (d / D.FREEZE_FILE).write_text("{}")
    with pytest.raises(D.DisclosureError):  # still a record, even in kit mode
        D.check_wsp_a_archive(d, allow_unfrozen=True)


PINS = {"id": "Qwen/Qwen3.5-0.8B", "revision": "r" * 40, "file_sha256": {"model.safetensors": "m" * 64}}
META = {"split": "val", "limit": None, "device": "mps", "dtype": "fp32", "model": PINS,
        "edit": {"edit_state_sha256": EDIT_HASH, "build_record_sha256": "b" * 64},
        "interface": {"spec_sha256": "i" * 64}}


def test_column_must_be_the_archived_edit_on_mps_fp32():
    D.check_column(META, EDIT_HASH, interface_spec_sha256="i" * 64)
    D.check_column({**{k: v for k, v in META.items() if k != "edit"}, "edit_state_sha256": EDIT_HASH}, EDIT_HASH)
    for bad, why in (({"device": "cpu"}, "mps"), ({"limit": 20}, "full val"), ({"edit": {}}, "edit state"),
                     ({"edit": {"edit_state_sha256": "0" * 64}}, "edit state")):
        with pytest.raises(D.DisclosureError, match=why):
            D.check_column({**META, **bad}, EDIT_HASH)
    with pytest.raises(D.DisclosureError, match="frozen interface"):
        D.check_column(META, EDIT_HASH, interface_spec_sha256="j" * 64)


def test_column_must_name_the_pinned_model_and_the_archive():
    D.check_column(META, EDIT_HASH, model_pins=PINS, build_record_sha256="b" * 64)
    for meta in ({**META, "model": {**PINS, "revision": "s" * 40}}, {k: v for k, v in META.items() if k != "model"}):
        with pytest.raises(D.DisclosureError, match="pinned"):
            D.check_column(meta, EDIT_HASH, model_pins=PINS)
    with pytest.raises(D.DisclosureError, match="BUILD_RECORD.json"):
        D.check_column(META, EDIT_HASH, build_record_sha256="c" * 64)
    with pytest.raises(D.DisclosureError, match="BUILD_RECORD.json"):  # only the fallback hash: no archive named
        D.check_column({**{k: v for k, v in META.items() if k != "edit"}, "edit_state_sha256": EDIT_HASH}, EDIT_HASH,
                       build_record_sha256="b" * 64)


def test_the_cli_checks_the_column_against_the_real_model_pins():
    """The builder passes registry.model_pins: the meta hypaqwen.run_readout writes (its "model" key) must match."""
    from agent_loop_lab.v04 import registry as R

    src = (REPO / "scripts" / "v04_build_disclosure.py").read_text()
    assert "model_pins=R.model_pins(REPO)" in src and 'build_record_sha256=wsp_a["build_record_sha256"]' in src
    run_readout = (REPO / "envs" / "qwen" / "hypaqwen" / "run_readout.py").read_text()
    assert '"model": {"id": MODEL_ID, "revision": MODEL_REVISION, "file_sha256": qmodel.FILE_SHA256}' in run_readout
    assert set(R.model_pins(REPO)) == {"id", "revision", "file_sha256"}


# -- CLI ------------------------------------------------------------------------------------------------------


def _cli():
    spec = importlib.util.spec_from_file_location("v04_build_disclosure", REPO / "scripts" / "v04_build_disclosure.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_cli_refuses_a_missing_column_and_the_official_dir_as_out_dir(tmp_path, capsys):
    m = _cli()
    assert m.main(["--column", str(tmp_path / "none.npz"), "--out-dir", str(tmp_path / "o")]) == 2
    assert "does not exist" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        m.main(["--column", str(tmp_path / "none.npz"), "--out-dir", str(REPO / "experiments" / "v0_4" / "x")])


def test_cli_writes_once_then_needs_force(v03, tmp_path, monkeypatch, capsys):
    m = _cli()
    rows, pos, pred, blocks, wpos, source = _inputs(v03)
    monkeypatch.setattr(m, "load_inputs", lambda args: {
        "rows": rows, "row_pos": pos, "pred": pred, "blocks": blocks, "window_pos": wpos, "v03_public": v03[0],
        "v03_private": v03[1], "source": source})
    col = tmp_path / "col.npz"
    col.write_bytes(b"fake")
    out = tmp_path / "o"
    argv = ["--column", str(col), "--out-dir", str(out)]
    assert m.main(argv) == 0
    first = {k: p.read_bytes() for k, p in D.output_paths(out).items()}
    assert "{'PRODUCTIVE': 50, 'RECOVERY': 50, 'STALLED': 50, 'OSCILLATING': 50}" in capsys.readouterr().out
    assert m.main(argv) == 2
    assert m.main(argv + ["--force"]) == 0
    assert {k: p.read_bytes() for k, p in D.output_paths(out).items()} == first
    monkeypatch.setattr(m, "load_inputs", lambda args: (_ for _ in ()).throw(D.DisclosureError("WSP-A column: x")))
    assert m.main(argv + ["--force"]) == 2 and "refusing: WSP-A column" in capsys.readouterr().err


def test_cli_refuses_scoring_and_freeze_errors_without_a_traceback(tmp_path, monkeypatch, capsys):
    from agent_loop_lab.v04 import handover as H
    from agent_loop_lab.v04.scoring import ScoringError

    m = _cli()
    col = tmp_path / "col.npz"
    col.write_bytes(b"fake")
    argv = ["--column", str(col), "--out-dir", str(tmp_path / "o")]
    for exc in (ScoringError("col.npz: not a full val run (split=val, limit=20)"), H.HandoverError("freeze x")):
        monkeypatch.setattr(m, "load_inputs", lambda args, exc=exc: (_ for _ in ()).throw(exc))
        assert m.main(argv) == 2 and f"refusing: {exc}" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_cli_refuses_a_limited_column_before_anything_is_written(tmp_path, monkeypatch, capsys):
    """The real load path: scoring.load_column's ScoringError (a column with limit=20) is a refusal, exit 2."""
    m = _cli()
    from agent_loop_lab.v04 import scoring as S

    monkeypatch.setattr(m, "check_archive", lambda *a, **k: ({"edit_state_sha256": EDIT_HASH,
                                                               "build_record_sha256": "b" * 64}, {"freeze_sha256": None}))
    lab = {"session_ids": np.array(["s1"]), "row_pos": np.array([3])}
    monkeypatch.setattr(S, "load_rows", lambda data_dir: ([], lab))
    col = tmp_path / "col.npz"
    np.savez(col, label_logits=np.zeros((1, 4)), logsumexp=np.zeros(1), session_ids=np.array(["s1"]),
             row_pos=np.array([3]), meta=json.dumps({"split": "val", "limit": 20}))
    assert m.main(["--column", str(col), "--out-dir", str(tmp_path / "o")]) == 2
    assert "refusing: col.npz: not a complete val run" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_cli_archive_freeze_policy(tmp_path, monkeypatch):
    """Without the registry only --kit-unfrozen (with --out-dir, a kit archive) accepts an unfrozen archive; with
    a registry the archive's freezes must be its G1 kernel freeze (handover.check_archive_freeze)."""
    from agent_loop_lab.v04 import handover as H

    m = _cli()
    none = tmp_path / "none.yaml"
    unfrozen = fake_archive(tmp_path / "u", freeze_sha=None)
    with pytest.raises(D.DisclosureError, match="no registry freeze_sha256"):
        m.check_archive(unfrozen, registry=none)
    assert m.check_archive(unfrozen, registry=none, kit_unfrozen=True)[1]["unfrozen"] is True
    with pytest.raises(H.HandoverError, match="no registry"):
        m.check_archive(fake_archive(tmp_path / "f"), registry=none, kit_unfrozen=True)
    col = tmp_path / "col.npz"
    col.write_bytes(b"x")
    for argv in (["--kit-unfrozen"], ["--kit-unfrozen", "--out-dir", str(tmp_path / "o"), "--wsp-a-archive",
                                      str(D.WSP_A_ARCHIVE)]):
        with pytest.raises(SystemExit):
            m.main(["--column", str(col)] + argv)
    monkeypatch.setattr(m, "REGISTRY", tmp_path / "registry_v0_4.yaml")
    m.REGISTRY.write_text("x")
    with pytest.raises(SystemExit):  # refused once a registry exists
        m.main(["--column", str(col), "--kit-unfrozen", "--out-dir", str(tmp_path / "o"), "--wsp-a-archive",
                str(unfrozen)])


# -- deviation D1: a readout of only the disclosed sessions -------------------------------------------------


def test_disclosed_sessions_file_is_the_v03_disclosed_sessions(v03):
    ids = D.DISCLOSED_SESSIONS.read_text().split()
    assert _sha(D.DISCLOSED_SESSIONS.read_bytes()) == D.DISCLOSED_SESSIONS_SHA256
    assert ids == sorted({w["session_id"] for w in v03[1]["windows"]}) and len(ids) == 200


def test_column_may_be_the_disclosed_sessions_subset_only():
    D.check_column({**META, "subset": {"file": "DISCLOSED_SESSIONS.txt", "sha256": D.DISCLOSED_SESSIONS_SHA256}},
                   EDIT_HASH)
    with pytest.raises(D.DisclosureError, match="other than DISCLOSED_SESSIONS.txt"):
        D.check_column({**META, "subset": {"file": "VAL_QUICK.txt", "sha256": "b" * 64}}, EDIT_HASH)


def test_cli_maps_a_disclosed_sessions_column_onto_the_val_rows(tmp_path, monkeypatch):
    """Rows of the subset's sessions get the column's argmax; every other row gets -1 (no prediction)."""
    from agent_loop_lab.v04 import scoring as S

    m = _cli()
    sub = tmp_path / "SUB.txt"
    sub.write_text("s2\ns1\n")
    monkeypatch.setattr(D, "DISCLOSED_SESSIONS", sub)
    monkeypatch.setattr(D, "DISCLOSED_SESSIONS_SHA256", _sha(sub.read_bytes()))
    monkeypatch.setattr(m, "check_archive", lambda *a, **k: ({"edit_state_sha256": EDIT_HASH,
                                                               "build_record_sha256": "b" * 64}, {"freeze_sha256": None}))
    lab = {"session_ids": np.array(["s0", "s1", "s2"]), "families": np.array(["f", "f", "f"]),
           "row_session": np.array([0, 1, 1, 2]), "row_pos": np.array([7, 7, 9, 8]),
           "row_endpoint": np.zeros(4), "row_label": np.zeros(4), "row_gate": np.ones(4, bool)}
    monkeypatch.setattr(S, "load_rows", lambda data_dir: ([object()] * 4, lab))
    monkeypatch.setattr(D, "check_column", lambda *a, **k: None)
    monkeypatch.setattr(D, "read_tasks", lambda path, ids: ({}, {}))
    monkeypatch.setattr(D, "check_v03_disclosure", lambda *a: None)
    logits = np.eye(4)[[2, 3, 1]]
    col = tmp_path / "col.npz"
    meta = {"split": "val", "limit": None, "subset": {"file": "SUB.txt", "sha256": _sha(sub.read_bytes())}}
    np.savez(col, label_logits=logits, logsumexp=np.zeros(3), session_ids=np.array(["s1", "s2"]),
             row_pos=np.array([7, 9, 8]), meta=json.dumps(meta))
    root = tmp_path / "v03"
    (root / "disclosure").mkdir(parents=True)
    (root / "disclosure" / "DISCLOSURE.json").write_text("{}")
    (root / "disclosure" / "DISCLOSURE.sha256").write_text("0" * 64 + "  DISCLOSURE.json\n")
    monkeypatch.setattr(D, "V03_ROOT", root)
    for name in ("val.tasks.jsonl.gz", "val.labels.npz"):
        (tmp_path / name).write_bytes(b"fake")
    args = m.argparse.Namespace(column=col, wsp_a_archive=tmp_path, data_dir=tmp_path, kit_unfrozen=False)
    (root / "DISCLOSURE_PRIVATE.json").write_text(json.dumps({"windows": [{"session_id": "s1"}, {"session_id": "s0"}]}))
    with pytest.raises(D.DisclosureError, match="not the v0.3 disclosed sessions"):
        m.load_inputs(args)
    (root / "DISCLOSURE_PRIVATE.json").write_text(json.dumps({"windows": [{"session_id": "s1"}, {"session_id": "s2"}]}))
    inp = m.load_inputs(args)
    assert inp["pred"].tolist() == [-1, 2, 3, 1]
