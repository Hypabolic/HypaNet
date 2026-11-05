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


def fake_archive(d: Path, *, phase: str = "wsp-a", stub: bool = False, freeze: bool = True) -> Path:
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
        (d / D.FREEZE_FILE).write_text(json.dumps({"freeze_sha256": "f" * 64}))
    return d


def test_wsp_a_archive_check(tmp_path):
    info = D.check_wsp_a_archive(fake_archive(tmp_path / "a"))
    assert info["edit_state_sha256"] == EDIT_HASH and info["freeze_sha256"] == "f" * 64
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


META = {"split": "val", "limit": None, "device": "mps", "dtype": "fp32", "edit": {"edit_state_sha256": EDIT_HASH},
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
