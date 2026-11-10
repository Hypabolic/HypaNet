"""kit/kit_build.py: kit/lib copies, the disclosure rule, the compact text form (SPEC §9, §13)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

SLOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SLOP / "kit"))
import kit_build as kb  # noqa: E402

HAVE_DEV = all((kb.DEV_DIR / f).is_file() for f in ("screens.jsonl", "dev.gen.jsonl", "dev.meta.jsonl"))


def test_lib_copies_match_ui_and_guard_matches_clouds():
    assert kb.lib_problems() == []


def test_rank_key_is_the_recorded_rule():
    assert kb.rank_key("abc") == hashlib.sha256(b"disclose:20261112:abc").hexdigest()


def test_norm_and_seq_hash():
    assert kb.norm("  Don’t STOP  now ") == "don't stop now"
    assert kb.norm("Due in eta hours") == "due in eta hours"
    a = {"nodes": [{"text": "Paused"}, {"text": "Resume"}]}
    b = {"nodes": [{"text": "PAUSED "}, {"text": "resume"}]}
    assert kb.seq_hash(a) == kb.seq_hash(b)
    assert kb.seq_hash(a) != kb.seq_hash({"nodes": [{"text": "Resume"}, {"text": "Paused"}]})


SCREEN = {
    "schema": "slop.screen/1", "id": "0123456789abcdef", "framework": "tsx",
    "source": {"repo": "x", "path": "a.tsx", "commit": None, "licence": None},
    "clusters": [{"k": 0, "kind": "header", "line": 1}, {"k": 1, "kind": "actions", "line": 5}],
    "nodes": [
        {"i": 0, "k": 0, "role": "eyebrow", "text": "OPERATION ON HOLD", "flags": ["caps"], "loc": [1, 1, 1, 18]},
        {"i": 1, "k": 0, "role": "h1", "text": "PAUSED", "flags": [], "loc": [2, 1, 2, 7]},
        {"i": 2, "k": 0, "role": "subtitle", "text": "Due in eta hours", "flags": [], "loc": [3, 1, 3, 9]},
        {"i": 3, "k": 1, "role": "aria", "text": "Close", "flags": [], "loc": [4, 1, 4, 6]},
        {"i": 4, "k": 1, "role": "button", "text": "RESUME", "flags": ["caps"], "loc": [5, 1, 5, 7]},
    ],
}
LABELS = [
    {"screen": SCREEN["id"], "loc": [1, 1, 1, 18], "class": "ECHO", "target": [2, 1, 2, 7], "equiv": [],
     "pattern": "R1", "aside": [], "partial": False, "ct": False, "conf": "clear"},
    {"screen": SCREEN["id"], "loc": [2, 1, 2, 7], "class": "KEEP", "target": None, "pattern": "K11", "conf": "clear"},
    {"screen": SCREEN["id"], "loc": [3, 1, 3, 9], "class": "NARRATE", "target": None, "pattern": "R6",
     "conf": "judgement", "ct": True, "aside": ["R15"], "partial": True},
    {"screen": SCREEN["id"], "loc": [5, 1, 5, 7], "class": "KEEP", "target": None, "pattern": None, "conf": "clear",
     "equiv": [[1, 1, 1, 18]]},
    {"screen": SCREEN["id"], "aside_screen": [{"code": "R19", "nodes": [[1, 1, 1, 18], [5, 1, 5, 7]]}]},
]


def test_compact_form():
    text = kb.compact(SCREEN, LABELS, "# head")
    assert text.splitlines() == [
        "# head",
        "-- cluster 0 header (line 1)",
        '0 0 eyebrow caps | "OPERATION ON HOLD" | ECHO→1 R1 clear',
        '1 0 h1 - | "PAUSED" | KEEP K11 clear',
        '2 0 subtitle - | "Due in ⟨VAR⟩eta hours" | NARRATE→- R6 judgement ct partial aside:R15',
        "-- cluster 1 actions (line 5)",
        '3 1 aria - | "Close" | (context)',
        '4 1 button caps | "RESUME" | KEEP clear equiv:0',
        "screen aside R19: nodes 0, 4",
    ]
    unl = kb.compact(SCREEN, None, "# head")
    assert unl.count("(unlabelled)") == 5


def test_codes_present_and_strip_record():
    assert kb.codes_present(LABELS) == "K11 R1 R6 aside:R15 screen:R19"
    assert kb.codes_present(None) == "(unlabelled)"
    full = {**LABELS[0], "labeller": "gen", "guide_sha256": "x", "time": "t"}
    assert set(kb.strip_record(full)) <= set(kb.NODE_KEYS)
    assert "labeller" not in kb.strip_record(LABELS[4]) and "aside_screen" in kb.strip_record(LABELS[4])


@pytest.mark.skipif(not HAVE_DEV, reason="dev set not built (datasets/dev is gitignored)")
def test_disclosure_rule():
    sel = kb.select_dev()
    assert len(sel["ids"]) == kb.N_DEV == len(set(sel["ids"]))
    assert sel["ids"] == sorted(sel["ids"], key=kb.rank_key)
    for code in kb.SCORED_PATTERNS + kb.K_RULES:
        assert sel["counts"][code] >= kb.QUOTA, code
    assert sel["pass1_screens"] <= kb.N_DEV
    held = kb.heldout_hashes()
    chosen = set(sel["ids"])
    for s in kb.iter_jsonl(kb.DEV_DIR / "screens.jsonl"):
        if s["id"] in chosen:
            assert kb.seq_hash(s) not in held
    assert kb.select_dev()["ids"] == sel["ids"]  # deterministic


CONTESTED = {"screen": SCREEN["id"], "loc": [3, 1, 3, 9], "class": None, "target": None, "pattern": None, "equiv": [],
             "partial": False, "ct": False, "conf": "contested", "aside": [], "labeller": "final", "source": "contested",
             "adjudication": [{"labeller": "a", "class": "KEEP"}],
             "candidates": [{"labeller": "a", "class": "NARRATE", "target": [5, 1, 5, 7], "pattern": "R6",
                             "conf": "judgement", "ct": True, "partial": False, "equiv": [], "aside": [],
                             "reason": "explains the button"},
                            {"labeller": "b", "class": "KEEP", "target": None, "pattern": "K4", "conf": "judgement",
                             "ct": True, "partial": False, "equiv": [], "aside": [], "reason": "states a cost"}]}


def test_contested_records_keep_their_readings_and_nothing_else():
    r = kb.strip_record(CONTESTED)
    assert set(r) <= set(kb.NODE_KEYS) and "source" not in r and "adjudication" not in r
    assert [set(c) for c in r["candidates"]] == [set(kb.CAND_KEYS)] * 2
    assert [c["class"] for c in r["candidates"]] == ["NARRATE", "KEEP"]
    agreed = {**LABELS[0], "candidates": CONTESTED["candidates"], "source": "agreed"}
    assert "candidates" not in kb.strip_record(agreed)
    line = kb.compact(SCREEN, [LABELS[0], r], "# head").splitlines()[4]
    assert line == '2 0 subtitle - | "Due in ⟨VAR⟩eta hours" | CONTESTED A:NARRATE→4 R6 ct | B:KEEP K4 ct'


def test_author_copy_of_the_guide():
    text = kb.kit_guide()
    for word in kb.GUIDE_FORBIDDEN:
        assert word.lower() not in text.lower(), word
    assert text.startswith(kb.GUIDE_PREAMBLE) and "\n## 1. The principle\n" in text
    guide = kb.GUIDE.read_text(encoding="utf-8")
    assert text[len(kb.GUIDE_PREAMBLE):].count("\n") == guide[guide.index(kb.GUIDE_PREAMBLE_END):].count("\n")
    for old, new in kb.GUIDE_EDITS:
        assert old not in text and new in text


def test_author_record_view():
    rec = {k: k for k in kb.AUTHOR_RECORD_KEYS} | {"harness": {}, "inputs": {}, "examples": {}, "test_root": "/x"}
    assert set(kb.author_record(rec)) == set(kb.AUTHOR_RECORD_KEYS)
