"""Tokeniser rules (SPEC.md §6) and window building (§4)."""

from __future__ import annotations

import hashlib
import json

import pytest

from conftest import SCREENS_DIR
from ui import tokenizer as tk

SPEC_ROLES = (
    "h1 h2 h3 h4 h5 h6 eyebrow subtitle body list_item caption label helper placeholder input_value button link nav tab "
    "breadcrumb menu_item badge tooltip empty_title empty_body dialog_title dialog_body toast error stat cell aria alt "
    "comment code other"
).split()
SPEC_FLAGS = (
    "caps title_case tracked small mono pill dot icon emoji_lead arrow_tail dynamic stateful conditional in_nav in_table "
    "in_form in_list in_dialog in_header in_footer in_card in_state in_control section_first section_last repeated sr_only "
    "truncated"
).split()


def words(text: str) -> list[str]:
    return [t.word for t in tk.tokenize_text(text).tokens if t.kind == "word"]


def ids(text: str, vocab: tk.Vocab = tk.EMPTY_VOCAB) -> list[int]:
    return tk.tokenize_text(text, vocab).ids


def g(name: str) -> int:
    return tk.GLYPH_IDS[name]


def test_fixed_lists_match_the_spec():
    assert list(tk.ROLES) == SPEC_ROLES and len(SPEC_ROLES) == 36
    assert list(tk.FLAGS) == SPEC_FLAGS and len(SPEC_FLAGS) == 28
    assert (tk.PAD, tk.BOS, tk.EON, tk.VAR, tk.ALT, tk.LZ) == (0, 1, 2, 3, 4, 5)
    assert min(tk.GLYPH_IDS.values()) == 64 and max(tk.GLYPH_IDS.values()) <= 95
    assert min(tk.PUNCT_IDS.values()) == 96 and max(tk.PUNCT_IDS.values()) <= 191 and len(tk.PUNCT_IDS) == 32
    assert tk.NUMBER_BASE == 192 and max(tk.YEAR, tk.NUM, tk.DEC) <= 319
    assert (tk.VOCAB_BASE, tk.HASH_BASE, tk.HASH_BUCKETS, tk.V) == (512, 4096, 6144, 10240)


# Ten-plus golden identifier cases (SPEC.md §12.2): camelCase, PascalCase, upper-case runs, snake_case, digits.
IDENTIFIERS = [
    ("createEngine", ["create", "engine"]),
    ("HTTPServer", ["http", "server"]),
    ("navigationTitle", ["navigation", "title"]),
    ("create_engine", ["create", "engine"]),
    ("iPhone", ["i", "phone"]),
    ("UserProfileCard", ["user", "profile", "card"]),
    ("XMLHttpRequest", ["xml", "http", "request"]),
    ("getHTTPResponseCode", ["get", "http", "response", "code"]),
    ("SCREAMING_SNAKE_CASE", ["screaming", "snake", "case"]),
    ("__private__", ["private"]),
    ("PAUSED", ["paused"]),
    ("ABCs", ["ab", "cs"]),
]


@pytest.mark.parametrize("text,expected", IDENTIFIERS)
def test_identifier_splitting(text, expected):
    assert words(text) == expected


def test_identifiers_with_digits():
    toks = tk.tokenize_text("utf8Encode v2Api 3rd").tokens
    assert [(t.kind, t.word or t.surface) for t in toks] == [
        ("word", "utf"), ("number", "8"), ("word", "encode"), ("word", "v"), ("number", "2"), ("word", "api"),
        ("number", "3"), ("word", "rd"),
    ]


def test_glyph_classes():
    assert ids("● • ◉") == [g("DOT")] * 3
    assert ids("·") == [g("MIDDOT")]
    assert ids("→ ↗ »") == [g("ARROW")] * 3
    assert ids("— –") == [g("DASH")] * 2
    assert ids("… ... ....") == [g("ELLIPSIS")] * 3
    assert ids("..") == [tk.PUNCT_IDS["."]] * 2
    assert ids("✓ ★ ✨") == [g("CHECK"), g("STAR"), g("SPARKLE")]
    assert ids("👍🏽 🇬🇧 ❤️‍🔥 ⌚ ⭐") == [g("EMOJI")] * 5
    assert ids("© ®") == [g("SYMBOL")] * 2  # pictographs below U+2300 are SYMBOL
    assert ids("½") == [tk.NUMBER_BASE + 1, g("SYMBOL"), tk.NUMBER_BASE + 2]  # NFKC gives 1, U+2044, 2
    assert ids("\U0001fae9") == [g("EMOJI")]  # unassigned in Unicode 15.0 but Extended_Pictographic
    assert ids("\U00010d50") == [g("SYMBOL")]  # a letter added after Unicode 15.0: SYMBOL in both tokenisers


def test_nfkc_fold_and_marks():
    assert words("Straße café naïve ﬁle ＡＢＣ") == ["strasse", "cafe", "naive", "file", "abc"]
    assert words("Don't they’re O'Neil 'quoted'") == ["dont", "theyre", "oneil", "quoted"]
    assert ids("'quoted'")[0] == tk.PUNCT_IDS["'"]
    assert ids("sign-in a‐b") == [ids("sign")[0], tk.PUNCT_IDS["-"], ids("in")[0], ids("a")[0], tk.PUNCT_IDS["-"], ids("b")[0]]
    assert words("zero​width soft­hyphen") == ["zerowidth", "softhyphen"]


def test_numbers():
    N = tk.NUMBER_BASE
    cases = {
        "0": [N], "7": [N + 7], "99": [N + 99], "00": [tk.LZ, N], "01": [tk.LZ, N + 1], "007": [tk.LZ, N + 7],
        "100": [tk.NUM], "1899": [tk.NUM], "1900": [tk.YEAR], "2026": [tk.YEAR], "2099": [tk.YEAR], "2100": [tk.NUM],
        "3.14": [tk.DEC], "0.5": [tk.DEC], "05.50": [tk.LZ, tk.DEC], "1,000": [tk.NUM], "1,000,000.50": [tk.DEC],
        "1,999": [tk.NUM], "123456789012345678901234567890": [tk.NUM],
    }
    for text, expected in cases.items():
        assert ids(text) == expected, text
    assert ids("12,34") == [N + 12, tk.PUNCT_IDS[","], N + 34]
    assert ids("v2.1.0") == [ids("v")[0], tk.DEC, tk.PUNCT_IDS["."], N]


def test_case_flags():
    assert tk.tokenize_text("OPERATION ON HOLD").flags == ["caps"]
    assert tk.tokenize_text("OK").flags == ["caps"]
    assert tk.tokenize_text("A").flags == []
    assert tk.tokenize_text("Delete Playlist").flags == ["title_case"]
    assert tk.tokenize_text("Delete playlist").flags == []
    assert tk.tokenize_text("Settings").flags == []
    assert tk.tokenize_text("\ue000 firstName").flags == []  # markers do not count
    assert tk.tokenize_text("Press ALT").flags == ["title_case"]  # the letters ALT are an ordinary word


def test_markers():
    assert tk.IDS["markers"] == {"VAR": "\ue000", "ALT": "\ue001"}
    assert ids("\ue000 userName \ue001") == [tk.VAR] + ids("user name") + [tk.ALT]
    assert ids("a\ue000b\ue001c") == [ids("a")[0], tk.VAR, ids("b")[0], tk.ALT, ids("c")[0]]  # markers need no spaces
    assert ids("\ue000\ue000") == [tk.VAR, tk.VAR]
    # the letters VAR and ALT are ordinary words, whatever their case
    for text in ("VAR", "ALT", "VAR ALT", "Var", "alt", "VARIANT ALTITUDE"):
        assert tk.VAR not in ids(text) and tk.ALT not in ids(text), text
    assert words("Hold ALT and press VAR") == ["hold", "alt", "and", "press", "var"]
    assert tk.tokenize_text("ALT").flags == ["caps"]
    # no other private-use code point is a marker
    assert ids("\ue002 \uf8ff") == [g("SYMBOL"), g("SYMBOL")]
    # NFKC leaves the markers alone, also beside text it does change, and a combining mark does not attach to them
    assert ids("\ufb01\ue000\uff21") == ids("fi") + [tk.VAR] + ids("A")
    assert ids("\ue001\u0301x") == [tk.ALT] + ids("x")
    assert tk._nfkc_protected("\ue000\u0301\ue001") == "\ue000\u0301\ue001"


def test_truncation():
    tt = tk.tokenize_text(" ".join(f"word{i}" for i in range(40)))
    assert len(tt.tokens) == 31 and tt.dropped == 49 and tt.flags == ["truncated"]
    assert len(tk.tokenize_text(" ".join(["a"] * 31)).tokens) == 31
    assert "truncated" not in tk.tokenize_text(" ".join(["a"] * 31)).flags


def test_vocab_and_hash_ids():
    v = tk.Vocab(["save", "set", "paus"])
    assert ids("Save settings PAUSED", v) == [512, 513, 514]
    digest = hashlib.sha256("resum".encode("utf-8")).digest()
    assert ids("Resume") == [4096 + int.from_bytes(digest[:4], "big") % 6144]
    assert ids("Resume", v) == ids("Resume")
    assert tk.Vocab.from_text("a\nb\n").stems == ["a", "b"]
    for bad in (["a", "a"], ["A"], ["two words"], [""], ["x"] * 0 + [f"s{i}" for i in range(3585)]):
        with pytest.raises(tk.TokenizerError):
            tk.Vocab(bad)


def test_id_layouts():
    l1, l2 = tk.LAYOUTS[1], tk.LAYOUTS[2]
    assert (l1.vocab_base, l1.vocab_max, l1.hash_base, l1.hash_buckets, l1.vocab_size) == (512, 3584, 4096, 6144, 10240)
    assert (l2.vocab_base, l2.vocab_max, l2.hash_base, l2.hash_buckets, l2.vocab_size) == (512, 16384, 16896, 8192, 25088)
    assert tk.layout(1) is l1 and tk.EMPTY_VOCAB.layout is l1 and tk.EMPTY_VOCABS[2].layout is l2
    for bad in (0, 3, "2", True, None):
        with pytest.raises(tk.TokenizerError):
            tk.layout(bad)
    digest = int.from_bytes(hashlib.sha256("resum".encode("utf-8")).digest()[:4], "big")
    assert tk.hash_bucket("resum") == 4096 + digest % 6144 == tk.hash_bucket("resum", 1)
    assert tk.hash_bucket("resum", 2) == 16896 + digest % 8192
    assert ids("Resume", tk.EMPTY_VOCABS[2]) == [16896 + digest % 8192]
    # ids below 512 do not move; a vocabulary line has the same id in both layouts
    v1, v2 = tk.Vocab(["save", "set"]), tk.Vocab(["save", "set"], 2)
    assert ids("Save settings: 42 ● ...", v1) == ids("Save settings: 42 ● ...", v2)
    big = [f"w{i}" for i in range(16384)]
    assert tk.Vocab(big, 2).word_id("w16383") == 16895
    for stems, lay in ((big[:3585], 1), (big + ["x"], 2)):
        with pytest.raises(tk.TokenizerError, match="max"):
            tk.Vocab(stems, lay)
    for w in ("resum", "configur", "zzzz", "naïve"):
        assert l2.hash_base <= tk.hash_bucket(w, 2) < l2.vocab_size and l1.hash_base <= tk.hash_bucket(w) < l1.vocab_size


def test_vocab_load_takes_the_layout_from_meta_beside_it(tmp_path):
    (tmp_path / "vocab.txt").write_text("save\n")
    assert tk.Vocab.load(tmp_path / "vocab.txt").ids_version == 1
    assert tk.meta_ids({"format": "slop.artifact/1"}) == 1 and tk.meta_ids({"ids": 2}) == 2
    (tmp_path / "meta.json").write_text(json.dumps({"format": "slop.artifact/1", "pointer_head": [4, 0], "ids": 2}))
    assert tk.Vocab.load(tmp_path / "vocab.txt").ids_version == 2
    assert tk.Vocab.load(tmp_path / "vocab.txt", ids=1).ids_version == 1


def _screen(nodes):
    return {"schema": "slop.screen/1", "id": "t", "nodes": [{"i": i, "loc": [1, 1, 1, 1], "flags": [], **n} for i, n in enumerate(nodes)]}


def test_sequence_layout():
    enc = tk.encode_screen(_screen([
        {"k": 0, "role": "eyebrow", "text": "NEW", "flags": ["small"]},
        {"k": 0, "role": "h1", "text": "Projects"},
        {"k": 3, "role": "button", "text": "Create project"},
    ]))
    w = enc.windows[0]
    assert w.tokens[0] == tk.BOS and w.roles[0] == -1 and w.clusters[0] == -1 and w.flags[0] == 0 and w.node[0] == 0
    assert w.eon == [2, 4, 7] and [w.tokens[p] for p in w.eon] == [tk.EON] * 3
    assert w.offsets == [0, 0, 1, 0, 1, 0, 1, 2]
    assert w.node == [0, 1, 1, 2, 2, 3, 3, 3]
    assert w.roles[1:] == [tk.ROLE_IDS["eyebrow"]] * 2 + [tk.ROLE_IDS["h1"]] * 2 + [tk.ROLE_IDS["button"]] * 3
    assert w.clusters[1:] == [0, 0, 0, 0, 3, 3, 3]
    caps_small = (1 << tk.FLAG_IDS["caps"]) | (1 << tk.FLAG_IDS["small"])
    assert w.flags[1] == caps_small and enc.nodes[0].flags == ["caps", "small"]
    assert enc.nodes[2].flags == []


def test_bad_screens():
    with pytest.raises(tk.TokenizerError):
        tk.encode_screen({"schema": "other", "nodes": []})
    with pytest.raises(tk.TokenizerError):
        tk.encode_screen(_screen([{"k": 0, "role": "headline", "text": "x"}]))
    with pytest.raises(tk.TokenizerError):
        tk.encode_screen(_screen([{"k": 0, "role": "h1", "text": "x", "flags": ["bold"]}]))
    with pytest.raises(tk.TokenizerError):
        tk.encode_screen(_screen([{"k": -1, "role": "h1", "text": "x"}]))


def _check_windows(enc: tk.EncodedScreen) -> None:
    seen = []
    for w in enc.windows:
        assert len(w.nodes) <= 64 and len(w.tokens) <= 1024
        seen.extend(w.nodes)
        assert len(w.tokens) == 1 + sum(len(enc.nodes[x].text.tokens) + 1 for x in w.nodes)
    assert seen == list(range(len(enc.nodes)))


def test_windows_split_a_long_cluster_at_node_boundaries():
    enc = tk.encode_screen(json.loads((SCREENS_DIR / "big_list_71.json").read_text()))
    _check_windows(enc)
    assert [len(w.nodes) for w in enc.windows] == [2, 64, 7]  # header alone, then the 71-item list in two parts
    assert {enc.nodes[x].k for x in enc.windows[1].nodes} == {1} and {enc.nodes[x].k for x in enc.windows[2].nodes} == {1}


def test_windows_split_at_cluster_boundaries():
    for name in ("faq_long_24.json", "office_cards_82.json"):
        enc = tk.encode_screen(json.loads((SCREENS_DIR / name).read_text()))
        _check_windows(enc)
        assert len(enc.windows) == 2
        for a, b in zip(enc.windows, enc.windows[1:]):
            assert enc.nodes[a.nodes[-1]].k != enc.nodes[b.nodes[0]].k  # no cluster is cut


def test_every_test_screen_encodes(test_screens):
    for s in test_screens:
        _check_windows(tk.encode_screen(s))
