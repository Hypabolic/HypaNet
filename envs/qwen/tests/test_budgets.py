"""BUDGETS.json reader (PREREGISTRATION §4, §5, §7): format, recomputed orders, hashes, seeds, disclosure rule.

Synthetic splits and labels only (the ``world`` fixture).
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from hypaqwen import budgets as B


def _bad(doc, edit):
    d = copy.deepcopy(doc)
    edit(d)
    return d


def test_valid_document_loads(world):
    b = B.load(world["budgets"])
    assert [len(b.sessions(n)) for n in B.SESSION_BUDGETS] == [32, 128, 512, 2048, 2600]
    assert b.size("200w") == 200 and b.size("all") == 2600 and len(b.es_val) == 500 and len(b.wspf_holdout) == 500
    assert b.file_sha256 and len(b.disclosed_sessions) == 200
    assert b.sessions("32") == b.sessions("all")[:32]
    with pytest.raises(B.BudgetError):
        b.sessions("200w")


def test_seeds_and_names_match_preregistration():
    assert B.BUDGET_NAMES == ("32", "128", "512", "2048", "all", "200w")
    assert B.LORA_SEEDS == {"32": (0, 1, 2), "128": (0, 1, 2), "512": (0, 1, 2), "2048": (0,), "all": (0,),
                            "200w": (0, 1, 2)}


def _swap(lst, i, j):
    lst[i], lst[j] = lst[j], lst[i]


def _resha(d):
    sha = d["sha256"]
    for k in B.SESSION_BUDGETS:
        sha[k] = B.ids_sha256(d["budgets"][k])
    sha["es_val"] = B.ids_sha256(d["es_val"])
    sha["wspf_holdout"] = B.ids_sha256(d["wspf_holdout"])
    sha["disclosed_200w"] = B.windows_sha256(B.Window(w["session_id"], w["row_pos"]) for w in d["disclosed_200w"])


@pytest.mark.parametrize("edit, match", [
    (lambda d: d.update(format="v04-budgets/0"), "format"),
    (lambda d: d.pop("wspf_holdout"), "keys"),
    (lambda d: d["budgets"].pop("2048"), "budgets"),
    (lambda d: (_swap(d["budgets"]["all"], 0, 1), _resha(d)), "order"),
    (lambda d: (d["budgets"].__setitem__("32", d["budgets"]["all"][1:33]), _resha(d)), "first 32"),
    (lambda d: (d["budgets"]["128"].__setitem__(5, d["budgets"]["128"][4]), _resha(d)), "duplicate"),
    (lambda d: (_swap(d["es_val"], 0, 1), _resha(d)), "ES_KEY"),
    (lambda d: (d["es_val"].__setitem__(-1, d["disclosed_200w"][0]["session_id"]),
                d.__setitem__("es_val", sorted(d["es_val"], key=B.es_key)), _resha(d)), "disclosed"),
    (lambda d: (d["wspf_holdout"].__setitem__(0, d["budgets"]["2048"][0]), _resha(d)), "holdout|HOLDOUT"),
    (lambda d: (d["disclosed_200w"].pop(), _resha(d)), "200"),
    (lambda d: (d["disclosed_200w"].__setitem__(1, {**d["disclosed_200w"][1],
                                                   "session_id": d["disclosed_200w"][0]["session_id"]}), _resha(d)),
     "one window per session"),
    (lambda d: (d["disclosed_200w"].__setitem__(0, {**d["disclosed_200w"][0], "row_pos": -1}), _resha(d)), "row_pos"),
    (lambda d: d["sha256"].__setitem__("512", "0" * 64), "sha256"),
    (lambda d: d["sha256"].pop("es_val"), "sha256"),
])
def test_invalid_documents_are_refused(world, edit, match):
    with pytest.raises(B.BudgetError, match=match):
        B.parse(_bad(world["doc"], edit))


def test_split_checks(world):
    b = B.load(world["budgets"])
    dev_ids = [s["session_id"] for s in world["dev"]]
    val_ids = [s["session_id"] for s in world["val"]]
    B.check_dev_ids(b, dev_ids)
    B.check_val_ids(b, val_ids)
    with pytest.raises(B.BudgetError):
        B.check_dev_ids(b, dev_ids[:-1])
    with pytest.raises(B.BudgetError):
        B.check_val_ids(b, val_ids[1:] + ["vXXXX"])  # changes the ES order
    B.check_disclosure(b, world["val_labels"])
    lab = {k: v.copy() for k, v in world["val_labels"].items()}
    first = b.disclosed_200w[0]
    s = [str(x) for x in lab["session_ids"]].index(first.session_id)
    lab["row_gate"][(lab["row_session"] == s) & (lab["row_pos"] == first.row_pos)] = False
    with pytest.raises(B.BudgetError):
        B.check_disclosure(b, lab)


def test_disclosure_rule_by_hand():
    """Two classes' candidates with known keys: gated and INSUFFICIENT rows never count; one window per session."""
    sids = ["a", "b", "c", "d", "e", "f", "g", "h"]
    rows = []  # (session index, pos, endpoint, label, gate)
    for s in range(len(sids)):
        for pos in (7, 8):
            rows.append((s, pos, pos + 1, (s + pos) % 4, True))
    rows.append((0, 9, 10, 4, True))  # INSUFFICIENT
    rows.append((1, 9, 10, 0, False))  # gate-rejected
    arr = np.array(rows, dtype=np.int64)
    lab = {"session_ids": np.array(sids), "row_session": arr[:, 0], "row_pos": arr[:, 1], "row_endpoint": arr[:, 2],
           "row_label": arr[:, 3].astype(np.int8), "row_gate": arr[:, 4].astype(bool)}
    got = B.disclosure_rows(lab, per_class=1)
    # Reference written out longhand: per class in order, the smallest key among unused sessions.
    used, chosen = set(), []
    for c in range(4):
        cands = sorted((B.disclosure_key(sids[s], e), s, p) for s, p, e, l, g in rows if g and l == c)
        for k, s, p in cands:
            if sids[s] not in used:
                used.add(sids[s])
                chosen.append((k, B.Window(sids[s], p)))
                break
    assert got == [w for _, w in sorted(chosen)]
    assert all(not (w.session_id == "b" and w.row_pos == 9) for w in got)


def test_load_reads_the_file(tmp_path, world):
    p = tmp_path / "BUDGETS.json"
    p.write_text(json.dumps(world["doc"]))
    assert B.load(p).budgets == B.load(world["budgets"]).budgets
