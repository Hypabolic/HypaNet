"""v0.5 Phase C data and held-out evaluation (src/agent_loop_lab/v05/phase_c_data.py, phase_c_eval.py,
scripts/v05c_build_disclosure.py, scripts/v05c_build_heldout.py, scripts/v05c_heldout_eval.py).

Small synthetic sizes in tmp dirs, scratch master seeds (never a reserved or a Phase C role seed for generated
data), committed artefacts, and the first few validation sessions only. No test opens, hashes or lists a sealed
test / OOD file or the real Phase C held-out data; an autouse fixture points the official report and attempt
record at tmp paths.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from agent_loop_lab.datasets import build as dsb
from agent_loop_lab.datasets.ood import ood_seed, ood_session_id
from agent_loop_lab.eval.collisions import WindowRow, session_windows
from agent_loop_lab.schema import CLASS_NAMES, OracleLabel
from agent_loop_lab.v03 import renderer, tokenizer
from agent_loop_lab.v03 import scoring as v03_scoring
from agent_loop_lab.v03 import sealed as S
from agent_loop_lab.v03.artifact import save_artifact
from agent_loop_lab.v03.model import zero_model
from agent_loop_lab.v05 import phase_c_data as P
from agent_loop_lab.v05 import phase_c_eval as E

REPO = Path(__file__).resolve().parents[1]
SCRATCH = 7  # a master seed no Phase C role, the sealed sets or the diagnosis uses
N_RES = 200


def _load_script(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _official_paths_to_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "HELDOUT_REPORT", tmp_path / "official" / "v05c_heldout.json")
    monkeypatch.setattr(E, "HELDOUT_MARKER", tmp_path / "official" / "v05c_heldout.started.json")


# ---------------------------------------------------------------------------
# Family rule
# ---------------------------------------------------------------------------


def test_family_rule_gives_the_pinned_split():
    seen, unseen = P.family_split()
    assert (seen, unseen) == (P.SEEN_FAMILIES, P.UNSEEN_FAMILIES)
    assert P.checked_family_split() == (seen, unseen)
    from agent_loop_lab.synth_ood import FAMILIES

    names = sorted(FAMILIES, key=lambda n: hashlib.sha256(f"20261109:v05c-family:{n}".encode()).hexdigest())
    assert tuple(names[:6]) == seen and tuple(names[6:]) == unseen
    assert len(seen) == 6 and len(unseen) == 4 and not set(seen) & set(unseen)
    assert set(seen) | set(unseen) == set(FAMILIES)


def test_family_rule_depends_only_on_names():
    names = [f"fam{i}" for i in range(9)]
    a = P.family_split(names)
    assert a == P.family_split(list(reversed(names)))
    assert len(a[0]) == 6 and len(a[1]) == 3
    with pytest.raises(ValueError):
        P.family_split(["a", "b", "c"])


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", sorted(P.FORBIDDEN_SEEDS))
@pytest.mark.parametrize("role", sorted(P.ROLE_SEEDS))
def test_reserved_seeds_are_refused(seed, role):
    with pytest.raises(P.SeedError, match="never be used"):
        P.check_seed(seed, role)


def test_role_seeds():
    assert P.ROLE_SEEDS == {"disclosure": 20261109, "heldout_ood": 20261110, "heldout_indist": 20261111}
    for role, seed in P.ROLE_SEEDS.items():
        assert P.check_seed(seed, role) == seed
        for other in P.ROLE_SEEDS:
            if other != role:
                with pytest.raises(P.SeedError, match=f"the {role} seed"):
                    P.check_seed(seed, other)
    with pytest.raises(P.SeedError):
        P.check_seed(True, "disclosure")
    with pytest.raises(ValueError):
        P.check_seed(SCRATCH, "nope")
    assert {20261104, 20261107, 20261108} == set(P.FORBIDDEN_SEEDS)


@pytest.mark.parametrize("seed", [20261104, 20261107, 20261108, 20261110])
def test_generators_refuse_before_generating(seed):
    with pytest.raises(P.SeedError):
        next(P.generate_ood(seed, P.SEEN_FAMILIES, 1, role="disclosure"))
    with pytest.raises(P.SeedError):
        P.disclosure_candidates(seed, 1)


@pytest.mark.parametrize("kw", [{"ood_seed_": 20261104}, {"ood_seed_": 20261109}, {"indist_seed": 20261107},
                                {"indist_seed": 20261110}])
def test_build_heldout_refuses_seeds(tmp_path, kw):
    with pytest.raises(P.SeedError):
        P.build_heldout(tmp_path / "d", ood_per_family=1, indist_sessions=1, **kw)
    assert not (tmp_path / "d").exists()


def test_disclosure_only_from_seen_families():
    with pytest.raises(P.DataError, match="seen families"):
        P.disclosure_candidates(SCRATCH, 1, families=(P.UNSEEN_FAMILIES[0],))


# ---------------------------------------------------------------------------
# Generation and encoding
# ---------------------------------------------------------------------------


def test_generation_matches_dataset_builders_and_v02_windows():
    ss = list(P.generate_ood(SCRATCH, ("merge_ping_pong",), 2, role="disclosure"))
    assert [s.session_id for s in ss] == [ood_session_id(ood_seed(SCRATCH, "merge_ping_pong", k)) for k in range(2)]
    assert all(s.record.seed == s.seed and s.family == P.ood_row_family("merge_ping_pong") for s in ss)
    ind = list(P.generate_indist(SCRATCH, 13))
    from agent_loop_lab.synth.families import FAMILY_NAMES

    want = [dsb.session_id_for(dsb.session_seed(SCRATCH, f, k)) for f in FAMILY_NAMES
            for k in range(dsb.family_counts(13, FAMILY_NAMES)[f])]
    assert [s.session_id for s in ind] == want
    enc = P.encode_sessions(ss + ind, keep_blocks=True)
    d = enc.data
    for n, s in enumerate(ss + ind):
        blocks = renderer.render_session([r.to_dict() for r in s.raw], s.session_id)
        assert enc.blocks[s.session_id] == blocks
        assert d.sessions[n].tokens == list(tokenizer.encode_session(blocks).tokens)
        mine = [d.rows[i] for i in d.rows_of(n)]
        v2 = session_windows(s.events, s.record)
        assert S.row_tuples(mine) == S.row_tuples(v2)
    assert enc.rows_sha256 == S.rows_sha256(d.rows)


# ---------------------------------------------------------------------------
# Re-timing of OOD sessions
# ---------------------------------------------------------------------------

_CLOCK = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d{3}) ")


def _hours(lines) -> list[str]:
    """The hour field of every block header among ``lines`` (a block starts with a line not starting with a
    space; its header begins with the rendered clock ``HH:MM:SS.mmm``)."""
    out = []
    for line in lines:
        m = _CLOCK.match(line)
        if m:
            out.append(m.group(1))
    return out


def test_retime_offsets_are_deterministic_and_in_range():
    ids = [ood_session_id(ood_seed(SCRATCH, f, k)) for f in P.ood_families() for k in range(50)]
    offs = [P.retime_offset_s(sid) for sid in ids]
    assert offs == [P.retime_offset_s(sid) for sid in ids]
    for sid, off in zip(ids, offs):
        assert isinstance(off, int) and 0 <= off < 86400
        want = int.from_bytes(hashlib.sha256(f"v05c-retime:{sid}".encode()).digest()[:8], "big") % 86400
        assert off == want
    assert len(set(offs)) > 0.95 * len(offs)  # per-session, not one offset for all
    assert len({o // 3600 for o in offs}) == 24  # 500 offsets cover every hour of the day
    assert P.RETIME_RULE["offset_formula"] == \
        'int.from_bytes(sha256(f"v05c-retime:{session_id}").digest()[:8], "big") % 86400'


@pytest.fixture(scope="module")
def shifted_pairs():
    """One session per OOD family (scratch seed), un-shifted and as generate_ood gives it."""
    fams = P.ood_families()
    plain = list(P._generate_ood_unshifted(SCRATCH, fams, 2, role="disclosure"))
    shifted = list(P.generate_ood(SCRATCH, fams, 2, role="disclosure"))
    return plain, shifted


def _ts(ts: str):
    from datetime import datetime

    return datetime.fromisoformat(ts)


def test_retime_shifts_every_timestamp_by_one_offset_and_nothing_else(shifted_pairs):
    import dataclasses
    from datetime import timedelta

    from agent_loop_lab.v03.build_raw import _check_aligned

    plain, shifted = shifted_pairs
    assert [s.session_id for s in plain] == [s.session_id for s in shifted]
    for a, b in zip(plain, shifted):
        off = P.retime_offset_s(a.session_id)
        assert (a.family, a.k, a.seed, a.record) == (b.family, b.k, b.seed, b.record)
        assert len(a.events) == len(b.events) and len(a.raw) == len(b.raw)
        for x, y in (*zip(a.events, b.events), *zip(a.raw, b.raw)):
            assert _ts(y.timestamp) - _ts(x.timestamp) == timedelta(seconds=off)
            assert dataclasses.replace(y, timestamp=x.timestamp) == x  # every other field unchanged
            assert len(y.timestamp) == len(x.timestamp) and y.timestamp[-6:] == x.timestamp[-6:]
        ta, tb = [_ts(e.timestamp) for e in a.events], [_ts(e.timestamp) for e in b.events]
        assert [q - p for p, q in zip(ta, ta[1:])] == [q - p for p, q in zip(tb, tb[1:])]  # gaps
        assert tb == sorted(tb)
        # the generator's own clock started at 2026-03-14 00:00 UTC; the shifted one starts `off` seconds later
        assert ta[0].hour == 0 and ta[0].date().isoformat() == "2026-03-14"
        assert (tb[0] - timedelta(seconds=off)) == ta[0]
        _check_aligned(b.session_id, b.events, b.raw)
        assert [r.timestamp for r in b.raw] == [e.timestamp for e in b.events]


def test_retime_leaves_window_rows_and_labels_unchanged(shifted_pairs):
    plain, shifted = shifted_pairs
    ea, eb = P.encode_sessions(plain, keep_blocks=True), P.encode_sessions(shifted, keep_blocks=True)
    assert S.row_tuples(ea.data.rows) == S.row_tuples(eb.data.rows)
    assert ea.rows_sha256 == eb.rows_sha256
    assert ea.encoding_sha256 != eb.encoding_sha256  # the rendered clock (and so the tokens) differ
    for a, b in zip(plain, shifted):
        ba, bb = ea.blocks[a.session_id], eb.blocks[b.session_id]
        assert len(ba) == len(bb)
        ha = [_hours(blk.split("\n")[:1])[0] for blk in ba]
        hb = [_hours(blk.split("\n")[:1])[0] for blk in bb]
        assert set(ha) <= {"00", "01"}  # un-shifted: every header in the first hour(s) of the day
        off = P.retime_offset_s(a.session_id)
        sec = int(ba[0][0:2]) * 3600 + int(ba[0][3:5]) * 60 + int(ba[0][6:8])
        assert hb[0] == f"{(sec + off) // 3600 % 24:02d}"


def test_retime_spreads_the_rendered_hour(shifted_pairs):
    plain, shifted = shifted_pairs
    hp = _hours(line for s in P.encode_sessions(plain, keep_blocks=True).blocks.values() for b in s
                for line in b.split("\n"))
    hs = _hours(line for s in P.encode_sessions(shifted, keep_blocks=True).blocks.values() for b in s
                for line in b.split("\n"))
    assert set(hp) <= {"00", "01"} and hp.count("00") > 0.9 * len(hp)
    assert len(set(hs)) >= 8 and hs.count("00") < 0.5 * len(hs)


def test_indist_sessions_are_not_retimed():
    from agent_loop_lab.synth.families import FAMILY_NAMES, generate_session

    s = next(P.generate_indist(SCRATCH, 13))
    events, _ = generate_session(FAMILY_NAMES[0], s.seed, s.session_id, P.load_oracle_config())
    assert [e.timestamp for e in s.events] == [e.timestamp for e in events]


def test_retime_refuses_a_timestamp_that_does_not_round_trip():
    with pytest.raises(P.DataError, match="round-trip"):
        P._shift_ts("2026-03-14T00:00:00Z", 5, "x")
    assert P._shift_ts(None, 5, "x") is None
    assert P._shift_ts("2026-03-14T23:59:59.500000+00:00", 2, "x") == "2026-03-15T00:00:01.500000+00:00"


# ---------------------------------------------------------------------------
# OOD disclosure
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def candidates():
    return P.disclosure_candidates(SCRATCH, 6)


def _model(seed: int | None):
    m = zero_model()
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    m.eval()
    return m


SRC = {"artifact_sha256": "0" * 64, "artifact_path": "x.dlw", "phase": None, "dry_run": True}


def test_ood_disclosure_selection_invariants(candidates):
    enc, meta = candidates
    pub, prv = P.build_ood_disclosure(enc, meta, _model(None), SRC, master_seed=SCRATCH, per_family=6, per_class=3)
    wins = pub["windows"]
    assert [w["id"] for w in wins] == [f"W{n:03d}" for n in range(1, 13)]
    assert all(set(w) == {"id", "prefix", "oracle_label", "network_class"} for w in wins)
    assert {c: sum(w["oracle_label"] == c for w in wins) for c in CLASS_NAMES} == {c: 3 for c in CLASS_NAMES}
    sids = [w["session_id"] for w in prv["windows"]]
    assert len(set(sids)) == len(sids)
    assert [w["key"] for w in prv["windows"]] == sorted(w["key"] for w in prv["windows"])
    rows = {(r.session_id, r.endpoint): r for r in enc.data.rows}
    for w, p in zip(wins, prv["windows"]):
        r = rows[(p["session_id"], p["endpoint"])]
        assert r.gate_passed and r.label.is_scored and r.label.to_class().name == w["oracle_label"]
        assert p["key"] == v03_scoring.disclosure_key(p["session_id"], p["endpoint"])
        assert p["family"] in P.SEEN_FAMILIES
        blocks, start = v03_scoring.parse_disclosed_prefix(w["prefix"])
        assert w["prefix"].count(v03_scoring.WINDOW_MARKER) == 1
        assert blocks == enc.blocks[p["session_id"]][: p["block"] + 1] and start == p["block"] - 7
        assert w["network_class"] == "PRODUCTIVE"  # the all-zero network
    # no family name, session id or seed anywhere in the handed-over record
    text = json.dumps(pub)
    from agent_loop_lab.synth.families import FAMILY_NAMES
    from agent_loop_lab.synth_ood import FAMILIES

    for s in (*FAMILIES, *FAMILY_NAMES, *meta, *(m["seed"] for m in meta.values())):
        assert str(s) not in text
    header = json.dumps({k: v for k, v in pub.items() if k != "windows"})
    assert "session_id" not in header and "seed" not in header  # (session, endpoint) is named only in the rule text
    for word in ("retim", "re-tim", "clock", "86400", "shift"):  # the re-timing is not revealed
        assert word not in header.lower()
    assert prv["retime"] == P.RETIME_RULE
    assert all(p["retime_offset_s"] == P.retime_offset_s(p["session_id"]) for p in prv["windows"])
    hours = _hours(line for w in wins for line in w["prefix"].split("\n"))
    assert len(set(hours)) >= 4 and hours.count("00") < 0.5 * len(hours)


def test_ood_disclosure_selection_uses_no_model_output(candidates):
    enc, meta = candidates
    a_pub, a_prv = P.build_ood_disclosure(enc, meta, _model(None), SRC, master_seed=SCRATCH, per_family=6, per_class=3)
    b_pub, b_prv = P.build_ood_disclosure(enc, meta, _model(3), SRC, master_seed=SCRATCH, per_family=6, per_class=3)
    assert a_prv["windows"] == b_prv["windows"]
    assert [(w["prefix"], w["oracle_label"]) for w in a_pub["windows"]] == \
        [(w["prefix"], w["oracle_label"]) for w in b_pub["windows"]]
    assert [w["network_class"] for w in a_pub["windows"]] != [w["network_class"] for w in b_pub["windows"]]


def test_leak_check_catches_family_and_session(candidates):
    enc, meta = candidates
    sid = next(iter(meta))
    forbidden = P.forbidden_strings(meta, P.DISCLOSURE_SEED)
    assert P.leak_problems(f"x {sid} y", forbidden) == sorted({sid, sid[2:]})
    assert P.leak_problems("merge_ping_pong", forbidden) == ["merge_ping_pong"]
    assert P.leak_problems("20261109", forbidden) == ["20261109"]
    assert P.leak_problems("nothing here", forbidden) == []


def test_ood_disclosure_header_names_b_only_for_b(candidates):
    enc, meta = candidates
    pub, _ = P.build_ood_disclosure(enc, meta, _model(None), SRC, master_seed=SCRATCH, per_family=6, per_class=3)
    assert pub["network"]["phase"] is None and "dry-run" in pub["network"]["description"]
    b_src = {"artifact_sha256": P.B_SHA256, "phase": P.B_PHASE, "dry_run": False}
    pub, _ = P.build_ood_disclosure(enc, meta, _model(None), b_src, master_seed=SCRATCH, per_family=6, per_class=3)
    assert pub["network"]["phase"] == P.B_PHASE and "Phase B" in pub["network"]["description"]
    header = json.dumps({k: v for k, v in pub.items() if k != "windows"})
    for word in ("unseen", "held out", "held-out", "evaluation", "false alarm", "distribution"):
        assert word not in header.lower()
    assert not re.search(r"(?<![a-z])ood(?![a-z])", header, re.IGNORECASE)


def test_ood_disclosure_passes_the_kit_screen(candidates, tmp_path):
    """The builder's handed-over W record passes kit_c.check_disclosure (the screen v05c_prepare_author_dir.sh
    runs before copying), including the network_class check against the artefact the header names."""
    from agent_loop_lab.v05 import kit_b, kit_c

    enc, meta = candidates
    art = tmp_path / "artifact.dlw"
    sha = save_artifact(zero_model(), art, {"phase": "stand-in"})
    src = {"artifact_sha256": sha, "artifact_path": str(art), "phase": None, "dry_run": True}
    pub, _ = P.build_ood_disclosure(enc, meta, _model(None), src, master_seed=SCRATCH, per_family=6, per_class=3)
    assert set(pub) == kit_b.DISCLOSURE_TOP_KEYS
    path = tmp_path / kit_c.DISCLOSURES["W"][0]
    path.write_text(json.dumps(pub, indent=1, sort_keys=True) + "\n")
    out = kit_c.check_disclosure(path, ids="W", expect=len(pub["windows"]), artifact=art)
    assert out["windows"] == 12 and out["network_class_checked_against"] == sha
    with_source = dict(pub, source="x")
    path.write_text(json.dumps(with_source))
    with pytest.raises(kit_c.KitError, match="unexpected top-level keys"):
        kit_c.check_disclosure(path, ids="W", expect=None)


def test_relabel_checks_phase_b_windows():
    db = _load_script("v05c_build_disclosure")
    b_wins = [{"id": "V001", "prefix": "p1", "oracle_label": "STALLED"},
              {"id": "V002", "prefix": "p2", "oracle_label": "RECOVERY"}]
    pub = {"format": "v05-phase-b-disclosure/1", "network": {"run": None}, "fields": {"network_class": "old"},
           "windows": [dict(w, network_class="PRODUCTIVE") for w in b_wins]}
    prv = {"format": "x", "same_windows_as": "v0.3", "source": {}}
    src = {"artifact_sha256": P.B_SHA256, "phase": P.B_PHASE, "dry_run": False}
    out, oprv = db.relabel(pub, prv, src, b_wins)
    assert out["format"] == db.FORMAT and out["network"]["phase"] == P.B_PHASE
    assert "Phase B" in out["fields"]["network_class"] and P.B_SHA256 in out["fields"]["network_class"]
    assert "v0.5 Phase B" in oprv["same_windows_as"]
    bad = [dict(b_wins[0]), dict(b_wins[1], prefix="other")]
    with pytest.raises(db.DisclosureError, match="prefix"):
        db.relabel(pub, prv, src, bad)
    with pytest.raises(db.DisclosureError, match="ids"):
        db.relabel(pub, prv, src, b_wins[:1])


@pytest.mark.parametrize("argv", [["--artifact", "x.dlw"], ["--sessions-per-family", "3"], ["--part", "ood"],
                                  ["--out", "experiments/v0_5/phase_c/dry"], ["--out", "experiments/v0_3/x"],
                                  ["--out", "weight_program/x"], ["--out", "X", "--sessions-per-family", "0"]])
def test_disclosure_script_argument_refusals(argv):
    db = _load_script("v05c_build_disclosure")
    argv = [str(REPO / a) if a.startswith(("experiments", "weight_program")) else a for a in argv]
    with pytest.raises(SystemExit) as e:
        db.main(argv)
    assert e.value.code == 2


def test_disclosure_script_refuses_overwrite(tmp_path):
    db = _load_script("v05c_build_disclosure")
    (tmp_path / "disclosure").mkdir()
    (tmp_path / "disclosure" / "DISCLOSURE_W.json").write_text("{}")
    assert db.main(["--out", str(tmp_path), "--part", "ood"]) == 2


# ---------------------------------------------------------------------------
# Held-out sets, disjointness and manifest hashing
# ---------------------------------------------------------------------------


def test_reserved_ids_cover_disclosure_and_sealed_namespaces(tmp_path):
    res = P.reserved_session_ids(5)
    every = frozenset().union(*res.values())
    for seed in (20261104, 20261107, 20261108, 20261109):
        for f in P.SEEN_FAMILIES:
            assert ood_session_id(ood_seed(seed, f, 4)) in every
    from agent_loop_lab.synth.families import FAMILY_NAMES

    assert dsb.session_id_for(dsb.session_seed(20261104, FAMILY_NAMES[0], 3)) in every
    assert ood_session_id(ood_seed(20261108, FAMILY_NAMES[0], 3)) in every  # the diagnosis' in-dist arm
    held = {ood_session_id(ood_seed(P.HELDOUT_OOD_SEED, f, k)) for f in P.ood_families() for k in range(5)}
    held |= {dsb.session_id_for(dsb.session_seed(P.HELDOUT_INDIST_SEED, f, k)) for f in FAMILY_NAMES for k in range(5)}
    assert P.check_disjoint(held, "held-out", res)["overlap"] == 0
    priv = tmp_path / "priv.json"
    priv.write_text(json.dumps({"windows": [{"session_id": "o-0000000000000001"}]}))
    res2 = P.reserved_session_ids(1, disclosure_private=priv)
    with pytest.raises(P.DataError, match="also in disclosed windows"):
        P.check_disjoint(["o-0000000000000001"], "held-out", res2)


@pytest.fixture(scope="module")
def tiny_heldout(tmp_path_factory):
    """Tiny held-out sets in a tmp dir, scratch master seeds 11 / 12 (not the real held-out seeds)."""
    root = tmp_path_factory.mktemp("heldout")
    man = P.build_heldout(root / "data", ood_seed_=11, indist_seed=12, ood_per_family=2, indist_sessions=13,
                          reserved_k=50)
    (root / "HELDOUT_MANIFEST.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    return root, man


def test_heldout_manifest_records_hashes_and_split(tiny_heldout):
    root, man = tiny_heldout
    assert man["official"] is False and man["format"] == P.MANIFEST_FORMAT
    assert man["family_rule"]["seen"] == list(P.SEEN_FAMILIES) and man["family_rule"]["unseen"] == list(P.UNSEEN_FAMILIES)
    ood, ind = man["sets"]["heldout_ood"], man["sets"]["heldout_indist"]
    assert ood["sessions"] == 20 and ind["sessions"] == 13 and ood["master_seed"] == 11
    for name in P.SET_NAMES:
        e = man["sets"][name]
        assert len(e["files"]) == 3
        for rel, meta in e["files"].items():
            assert meta["sha256"] == dsb.file_sha256(root / "data" / rel)
    assert man["disjointness"]["heldout_indist"]["overlap"] == 0
    assert "held-out set heldout_ood" in man["disjointness"]["heldout_indist"]["disjoint_from"]
    assert any("20261109" in s for s in man["disjointness"]["heldout_ood"]["disjoint_from"])
    assert ood["retime"] == P.RETIME_RULE and "retime" not in ind
    assert P.manifest_problems(man, official=False) == []
    probs = P.manifest_problems(man, official=True)
    assert any("dry-run" in p for p in probs) and any("pre-registered" in p for p in probs)


def test_manifest_problems_flag_changes(tiny_heldout):
    _, man = tiny_heldout
    bad = json.loads(json.dumps(man))
    bad["sets"]["heldout_ood"]["master_seed"] = 20261104
    assert any("never be used" in p for p in P.manifest_problems(bad, official=False))
    bad = json.loads(json.dumps(man))
    bad["family_rule"]["seen"] = list(P.UNSEEN_FAMILIES)
    assert any("family split" in p for p in P.manifest_problems(bad, official=False))
    bad = json.loads(json.dumps(man))
    bad["source_sha256"]["synth_ood (OOD generator)"] = "0" * 64
    assert any("synth_ood" in p for p in P.manifest_problems(bad, official=False))
    bad = json.loads(json.dumps(man))
    del bad["sets"]["heldout_ood"]["retime"]
    assert any("re-timing" in p for p in P.manifest_problems(bad, official=False))
    bad = json.loads(json.dumps(man))
    bad["sets"]["heldout_indist"]["retime"] = P.RETIME_RULE
    assert any("not re-timed" in p for p in P.manifest_problems(bad, official=False))


def test_heldout_ood_files_hold_retimed_sessions(tiny_heldout):
    """The written OOD set holds the re-timed sessions (alignment re-checked on reading), the in-distribution set
    the standard generator's own timestamps."""
    from datetime import timedelta

    root, _ = tiny_heldout
    ood = list(P.iter_set(root / "data", "heldout_ood"))
    plain = {s.session_id: s for s in P._generate_ood_unshifted(11, P.ood_families(), 2, role="heldout_ood")}
    assert [s.session_id for s in ood] == list(plain)
    for s in ood:
        off = timedelta(seconds=P.retime_offset_s(s.session_id))
        assert [_ts(e.timestamp) - off for e in s.events] == [_ts(e.timestamp) for e in plain[s.session_id].events]
        assert s.record == plain[s.session_id].record
    ind = list(P.iter_set(root / "data", "heldout_indist"))
    assert [[e.timestamp for e in s.events] for s in ind] == \
        [[e.timestamp for e in s.events] for s in P.generate_indist(12, 13)]


def test_real_w_disclosure_hours_are_spread():
    """The committed DISCLOSURE_W.json: the rendered hour field of its block headers is spread over the day, and
    its private record carries the re-timing rule and each window's offset (no generation; reads two files)."""
    pub_p = REPO / "experiments" / "v0_5" / "phase_c" / "disclosure" / "DISCLOSURE_W.json"
    prv_p = REPO / "experiments" / "v0_5" / "phase_c" / "DISCLOSURE_OOD_PRIVATE.json"
    if not (pub_p.is_file() and prv_p.is_file()):
        pytest.skip("the real W disclosure is not built")
    pub, prv = json.loads(pub_p.read_text()), json.loads(prv_p.read_text())
    last = []
    for w in pub["windows"]:
        hs = _hours(w["prefix"].split("\n"))
        assert hs
        last.append(hs[-1])
    assert len(last) == 200 and len(set(last)) >= 16 and last.count("00") <= 30
    assert prv["retime"] == P.RETIME_RULE
    assert all(p["retime_offset_s"] == P.retime_offset_s(p["session_id"]) for p in prv["windows"])
    head = json.dumps({k: v for k, v in pub.items() if k != "windows"}).lower()
    assert not any(w in head for w in ("retim", "re-tim", "clock", "86400", "shift"))


def test_heldout_build_is_byte_reproducible(tiny_heldout, tmp_path):
    root, man = tiny_heldout
    again = P.build_heldout(tmp_path / "data", ood_seed_=11, indist_seed=12, ood_per_family=2, indist_sessions=13,
                            reserved_k=50)
    assert again["sets"] == man["sets"]


def test_write_set_refuses_overwrite(tiny_heldout):
    root, _ = tiny_heldout
    with pytest.raises(P.DataError, match="overwrite"):
        P.write_set(root / "data", "heldout_ood", iter(()))


def test_load_set_needs_attempt_and_checks_hashes(tiny_heldout, tmp_path):
    root, man = tiny_heldout
    entry = man["sets"]["heldout_ood"]
    with pytest.raises(S.SealedAccessError):
        P.load_set(root / "data", "heldout_ood", entry, attempt=None)
    att = S.write_attempt_record({"test": True}, tmp_path / "marker.json")
    sd = P.load_set(root / "data", "heldout_ood", entry, attempt=att)
    assert sd.split == "heldout_ood" and len(sd.data.session_ids) == 20
    assert S.rows_sha256(sd.data.rows) == entry["rows_sha256"]
    with pytest.raises(S.SealedAccessError):
        P.load_set(root / "data", "heldout_ood", entry, attempt=S.Attempt(att.marker, "wrong"))
    bad = dict(entry, rows_sha256="0" * 64)
    with pytest.raises(P.DataError, match="rows_sha256"):
        P.load_set(root / "data", "heldout_ood", bad, attempt=att)
    bad = dict(entry, files={k: dict(v, sha256="0" * 64) for k, v in entry["files"].items()})
    with pytest.raises(P.DataError, match="sha256 differs"):
        P.verify_set_files(root / "data", "heldout_ood", bad)


def test_population_masks_partition_ood_families(tiny_heldout, tmp_path):
    root, man = tiny_heldout
    att = S.write_attempt_record({"test": True}, tmp_path / "m.json")
    sd = P.load_set(root / "data", "heldout_ood", man["sets"]["heldout_ood"], attempt=att)
    m = E.population_masks(sd)
    assert set(m) == {"unseen", "seen", "all"}
    assert np.all(m["unseen"] ^ m["seen"]) and m["all"].all()
    fams = {r.family for r, u in zip(sd.data.rows, m["unseen"]) if u}
    assert fams == {P.ood_row_family(f) for f in P.UNSEEN_FAMILIES}
    sd2 = P.load_set(root / "data", "heldout_indist", man["sets"]["heldout_indist"], attempt=att)
    assert set(E.population_masks(sd2)) == {"all"}


def test_heldout_script_argument_refusals(tmp_path):
    hb = _load_script("v05c_build_heldout")
    for argv in (["--ood-per-family", "2"], ["--data-root", str(tmp_path)],
                 ["--data-root", str(REPO / "datasets" / "v0_5c" / "x"), "--manifest", str(tmp_path / "m.json")],
                 ["--data-root", str(tmp_path), "--manifest", str(REPO / "experiments/v0_5/phase_c/heldout/x.json")]):
        with pytest.raises(SystemExit) as e:
            hb.main(argv)
        assert e.value.code == 2
    (tmp_path / "m.json").write_text("{}")
    assert hb.main(["--data-root", str(tmp_path / "d"), "--manifest", str(tmp_path / "m.json")]) == 2


# ---------------------------------------------------------------------------
# Evaluation refusals and columns
# ---------------------------------------------------------------------------


def test_refuse_reasons(tmp_path):
    rep, mk = tmp_path / "r.json", tmp_path / "m.json"
    committed = "experiments/v0_5/sealed/README.md"  # any committed, unchanged file
    kw = dict(manifest=committed, disclosure_private=committed, code_files=(committed,))
    assert E.refuse_reasons(rep, mk, pre_registration=committed, **kw) == []
    rep.write_text("{}")
    mk.write_text("{}")
    r = E.refuse_reasons(rep, mk, pre_registration=committed, **kw)
    assert len(r) == 2 and all("runs once" in x for x in r)
    r = E.refuse_reasons(tmp_path / "a", tmp_path / "b", pre_registration=tmp_path / "nope.md", **kw)
    assert len(r) == 1 and "missing" in r[0]
    untracked = REPO / "experiments" / "v0_5" / "phase_c" / "heldout" / "_untracked_test_file.md"
    untracked.write_text("x")
    try:
        r = E.refuse_reasons(tmp_path / "a", tmp_path / "b", pre_registration=untracked, **kw)
        assert len(r) == 1 and "not committed" in r[0]
        # the scoring code must be committed unchanged too
        r = E.refuse_reasons(tmp_path / "a", tmp_path / "b", pre_registration=committed, manifest=committed,
                             disclosure_private=committed, code_files=(committed, untracked))
        assert len(r) == 1 and "not committed (scoring code)" in r[0]
    finally:
        untracked.unlink()
    assert set(E.CODE_FILES) == {*P.SCORING_CODE_FILES, E.PRE_REGISTRATION, E.MANIFEST}
    assert {"scripts/v05c_heldout_eval.py", "src/agent_loop_lab/v05/phase_c_eval.py",
            "src/agent_loop_lab/v05/phase_c_data.py"} <= set(P.SCORING_CODE_FILES)


def test_manifest_pins_scoring_code(tiny_heldout):
    _, man = tiny_heldout
    assert man["scoring_code_sha256"] == P.scoring_code_sha256()
    assert all(v is not None for v in man["scoring_code_sha256"].values())
    pinned = [p for p in P.manifest_problems(man, official=True) if "scoring code" in p]
    assert pinned == []
    bad = json.loads(json.dumps(man))
    bad["scoring_code_sha256"]["src/agent_loop_lab/v05/phase_c_eval.py"] = "0" * 64
    assert any("scoring code differs" in p and "phase_c_eval.py" in p for p in P.manifest_problems(bad, official=True))
    assert not any("scoring code" in p for p in P.manifest_problems(bad, official=False))
    del bad["scoring_code_sha256"]
    assert any("does not pin the scoring code" in p for p in P.manifest_problems(bad, official=True))


def test_prelaunch_problems(tiny_heldout, tmp_path):
    root, _ = tiny_heldout
    committed = "experiments/v0_5/sealed/README.md"
    kw = dict(pre_registration=committed, disclosure_private=committed, code_files=(committed,))
    # a committed .json that is not an official held-out manifest does not freeze the evaluation
    r = E.prelaunch_problems(manifest="experiments/registry_v0_3.yaml", **kw)
    assert len(r) == 1 and "not JSON" in r[0]
    r = E.prelaunch_problems(manifest=tmp_path / "nope.json", **kw)
    assert len(r) == 1 and "missing" in r[0]
    # the dry-run manifest is outside the repo (not committed) and fails manifest_problems(official=True)
    r = E.prelaunch_problems(manifest=root / "HELDOUT_MANIFEST.json", **kw)
    assert any("outside the repository" in x for x in r) and any("dry-run" in x for x in r)
    assert E.prelaunch_problems.__defaults__ is None and E.prelaunch_problems.__kwdefaults__["manifest"] == E.MANIFEST


def test_dry_run_dir_problem():
    assert E.dry_run_dir_problem(REPO / "artifacts" / "reports") is not None
    assert E.dry_run_dir_problem(REPO / "artifacts" / "reports" / "sub") is not None
    assert E.dry_run_dir_problem(REPO / "artifacts") is None


@pytest.mark.parametrize("argv", [
    ["--heldout"],
    ["--heldout", "--confirm-one-time", "--phase-c-artifact", "x.dlw"],
    ["--heldout", "--confirm-one-time", "--report-dir", "x"],
    ["--heldout", "--confirm-one-time", "--limit-sessions", "3"],
    ["--dry-run-val"],
    ["--dry-run-val", "--report-dir", str(REPO / "artifacts" / "reports")],
    ["--dry-run-val", "--report-dir", "x", "--confirm-one-time"],
    ["--dry-run-val", "--report-dir", "x", "--limit-sessions", "0"],
    ["--dry-run-val", "--heldout", "--report-dir", "x"],
])
def test_eval_script_argument_refusals(argv):
    ev = _load_script("v05c_heldout_eval")
    with pytest.raises(SystemExit) as e:
        ev.parse_args(argv)
    assert e.value.code == 2


def test_eval_script_refuses_when_report_or_marker_exists(monkeypatch, tmp_path):
    ev = _load_script("v05c_heldout_eval")
    E.HELDOUT_MARKER.parent.mkdir(parents=True, exist_ok=True)
    E.HELDOUT_MARKER.write_text("{}")
    monkeypatch.setattr(E, "load_eval_columns", lambda *a, **k: pytest.fail("columns loaded after a refusal"))
    assert ev.main(["--heldout", "--confirm-one-time"]) == 2


def _fake_phase_c(d: Path, *, phase="v05-phase-c", base=P.B_SHA256, build_base=None) -> str:
    d.mkdir(parents=True)
    sha = save_artifact(zero_model(), d / "artifact.dlw", {"phase": phase, "base_artifact_sha256": base})
    (d / "artifact.sha256").write_text(f"{sha}  artifact.dlw\n")
    (d / "apply.json").write_text(json.dumps({"artifact": {"sha256": sha}}))
    (d / "BUILD_RECORD.json").write_text(json.dumps({"artifact_sha256": sha,
                                                     "base_artifact_sha256": base if build_base is None else build_base}))
    return sha


def test_load_phase_c_checks(tmp_path):
    with pytest.raises(S.CheckFailed, match="missing"):
        E.load_phase_c(tmp_path / "none", base_sha256=P.B_SHA256)
    sha = _fake_phase_c(tmp_path / "ok")
    spec, info = E.load_phase_c(tmp_path / "ok", base_sha256=P.B_SHA256)
    assert spec.name == E.C and info["artifact_sha256"] == sha and info["stand_in"] is False
    _fake_phase_c(tmp_path / "phase", phase="v05-phase-b")
    with pytest.raises(S.CheckFailed, match="phase"):
        E.load_phase_c(tmp_path / "phase", base_sha256=P.B_SHA256)
    _fake_phase_c(tmp_path / "base", base="1" * 64)
    with pytest.raises(S.CheckFailed, match="base_artifact_sha256"):
        E.load_phase_c(tmp_path / "base", base_sha256=P.B_SHA256)
    _fake_phase_c(tmp_path / "build", build_base="1" * 64)
    with pytest.raises(S.CheckFailed, match="BUILD_RECORD"):
        E.load_phase_c(tmp_path / "build", base_sha256=P.B_SHA256)


@pytest.fixture(scope="module")
def stand_in_cols():
    return E.load_eval_columns(REPO / "weight_program" / "v0_5" / "phase-b" / "artifact.dlw")


def test_columns_and_pairs(stand_in_cols):
    cols = stand_in_cols
    assert [s.name for s in cols.specs] + [E.PRIOR] == list(E.COLUMN_ORDER)
    assert cols.phase_c_stand_in and cols.info[E.C]["stand_in"] is True and "STAND-IN" in cols.info[E.C]["WARNING"]
    assert E.PRIMARY_PAIRS == ("v0.5 Phase C network - v0.5 B network", "v0.5 Phase C network - v0.3 B network")
    assert E.PAIRS[E.PRIMARY_PAIRS[0]] == (E.C, E.B) and E.PAIRS[E.PRIMARY_PAIRS[1]] == (E.C, E.V03B)
    assert E.COMPARISON_ORDER[0] == ("heldout_ood", "unseen")


def test_official_needs_real_phase_c_and_heldout(stand_in_cols, tmp_path):
    with pytest.raises(ValueError, match="official"):
        E.evaluate(stand_in_cols, heldout=True, official=True, report_path=tmp_path / "r.json",
                   marker_path=tmp_path / "m.json")
    assert not (tmp_path / "m.json").exists()


def test_heldout_evaluation_end_to_end(stand_in_cols, tiny_heldout, tmp_path, monkeypatch):
    """Tiny held-out sets, the first 3 val sessions, a stand-in Phase C (B): order, marker, populations, pairs."""
    root, man = tiny_heldout
    order: list[str] = []
    real_write, real_load = S.write_attempt_record, P.load_set

    def write(record, marker=None):
        order.append("marker")
        return real_write(record, marker)

    def load(*a, **k):
        order.append(f"load {a[1]}")
        return real_load(*a, **k)

    monkeypatch.setattr(S, "write_attempt_record", write)
    monkeypatch.setattr(P, "load_set", load)
    rep_path, marker = tmp_path / "rep.json", tmp_path / "marker.json"
    rep = E.evaluate(stand_in_cols, heldout=True, official=False, report_path=rep_path, marker_path=marker,
                     data_root=root / "data", manifest_path=root / "HELDOUT_MANIFEST.json",
                     disclosure_private=tmp_path / "none.json", val_limit=3, n_resamples=N_RES)
    assert order == ["marker", "load heldout_ood", "load heldout_indist"]
    assert marker.is_file() and rep_path.is_file()
    assert json.loads(rep_path.read_text())["official"] is False
    assert rep["phase_c_stand_in"] is True and rep["notes"][0].startswith("WARNING: STAND-IN")
    assert list(rep["sets"]) == ["heldout_ood", "heldout_indist", "val"]
    assert list(rep["sets"]["heldout_ood"]["populations"]) == ["unseen", "seen", "all"]
    assert list(rep["sets"]["val"]["populations"]) == ["excluding_disclosed", "all_val"]
    for s, pops in rep["sets"].items():
        for pop, sc in pops["populations"].items():
            assert list(sc["overall"]["columns"]) == list(E.COLUMN_ORDER)
            d = sc["paired_differences"]["macro_f1"][E.PRIMARY_PAIRS[0]]
            assert d["value"] in (0.0, None)  # the stand-in is B itself
    assert rep["checks"]["disjointness"]["heldout_ood"]["overlap"] == 0
    # a second run refuses before scoring anything: the attempt record exists
    with pytest.raises(S.CheckFailed, match="runs once"):
        E.evaluate(stand_in_cols, heldout=True, official=False, report_path=tmp_path / "rep2.json",
                   marker_path=marker, data_root=root / "data", manifest_path=root / "HELDOUT_MANIFEST.json",
                   disclosure_private=tmp_path / "none.json", val_limit=3, n_resamples=N_RES)


def test_heldout_evaluation_rejects_tampered_data_before_marker(stand_in_cols, tiny_heldout, tmp_path):
    root, man = tiny_heldout
    bad = json.loads(json.dumps(man))
    rel = next(iter(bad["sets"]["heldout_indist"]["files"]))
    bad["sets"]["heldout_indist"]["files"][rel]["sha256"] = "0" * 64
    mp = tmp_path / "man.json"
    mp.write_text(json.dumps(bad))
    with pytest.raises(P.DataError, match="sha256 differs"):
        E.evaluate(stand_in_cols, heldout=True, official=False, report_path=tmp_path / "r.json",
                   marker_path=tmp_path / "m.json", data_root=root / "data", manifest_path=mp,
                   disclosure_private=tmp_path / "none.json", val_limit=3, n_resamples=N_RES)
    assert not (tmp_path / "m.json").exists() and not (tmp_path / "r.json").exists()


def test_disclosure_private_must_be_seen_families(tmp_path):
    p = tmp_path / "priv.json"
    p.write_text(json.dumps({"master_seed": P.DISCLOSURE_SEED,
                             "windows": [{"session_id": "o-1", "family": P.UNSEEN_FAMILIES[0]}]}))
    with pytest.raises(S.CheckFailed, match="not seen"):
        E.check_disclosure_private(p)
    p.write_text(json.dumps({"master_seed": 1, "windows": []}))
    with pytest.raises(S.CheckFailed, match="master seed"):
        E.check_disclosure_private(p)
