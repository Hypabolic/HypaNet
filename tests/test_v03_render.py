"""v0.3 renderer (spec §4) and tokeniser (§5): goldens, caps, vocabulary, and a property pass over all dev.

The dev pass renders every dev session once (module fixture, ~2 min) and checks the §4.6 caps, line
markers, hunk-id formula, revert headers and the absence of 64-hex strings. No other split is rendered.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agent_loop_lab.v03 import renderer as R
from agent_loop_lab.v03 import tokenizer as T
from agent_loop_lab.v03.rawop import RawOp

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "datasets" / "v0_3"
SRC = REPO / "src" / "agent_loop_lab"
SID = "s-golden"
HEX64 = re.compile(r"[0-9a-fA-F]{64}")
HUNK_LINE = re.compile(r"^  @ (\S+) (h0|h[0-9a-f]{10}) > (h0|h[0-9a-f]{10})$")


def _ids(text: str) -> list[int]:
    return [t for t, _ in T.tokenize_block(text)]


def _spec_hunk_id(session_id: str, path: str, lines: list[str]) -> str:
    """Spec §4.3, written independently of the renderer."""
    if not lines:
        return "h0"
    return "h" + hashlib.sha256((session_id + ":hunk:" + path + ":" + "\n".join(lines)).encode()).hexdigest()[:10]


def _spec_hunks(before, after) -> list[tuple[list[str], list[str]]]:
    """Spec §4.3 hunks (removed, added) of ``difflib.unified_diff(..., n=0)``, independent of the renderer."""
    import difflib

    a = [] if before is None else before.split("\n")
    b = [] if after is None else after.split("\n")
    out: list[tuple[list[str], list[str]]] = []
    for ln in list(difflib.unified_diff(a, b, lineterm="", n=0))[2:]:
        if ln.startswith("@@"):
            out.append(([], []))
        elif ln[:1] in "-+":
            out[-1][0 if ln[0] == "-" else 1].append(ln[1:])
    return out


# -- goldens (one hand-written RawOp per tool family) ------------------------------

GOLDEN_OPS = {
    "read_file": dict(family="read_file", target="pkg/alpha.py", args={"line_start": 3, "line_end": 14},
                      outcome="SUCCESS", input_tokens=1234, output_tokens=56,
                      timestamp="2026-03-14T09:05:07.250000+00:00"),
    "search": dict(family="search", target="pkg", args={"query": "frobnicate widget", "limit": 20},
                   outcome="SUCCESS", input_tokens=88, output_tokens=7, timestamp="2026-03-14T10:00:00+01:00"),
    "edit_file": dict(family="edit_file", target="pkg/alpha.py",
                      args={"patch": "--- a/pkg/alpha.py\n+++ b/pkg/alpha.py\n@@ -2 +2 @@\n-x = 1\n+x = 2"},
                      outcome="SUCCESS",
                      changed=[{"path": "pkg/alpha.py", "before": "a\nx = 1\nb", "after": "a\nx = 2\nb"}],
                      input_tokens=500, output_tokens=1000, timestamp="2026-03-14T09:05:08.000Z"),
    "revert": dict(family="revert", target=["pkg/alpha.py", "pkg/beta.py"], args={"revision": "f" * 64},
                   outcome="SUCCESS",
                   changed=[{"path": "pkg/alpha.py", "before": "a\nx = 2\nb", "after": "a\nx = 1\nb"},
                            {"path": "pkg/beta.py", "before": "new", "after": None}],
                   input_tokens=10, output_tokens=20, timestamp="2026-03-14T09:05:09.999Z"),
    "run_tests": dict(family="run_tests", target="unit", args={"tests": ["t/test_a.py::test_one"]}, outcome="ERROR",
                      error_class="AssertionError", error_location="t/test_a.py::test_one",
                      validation=("unit", 3, 1234567), extra={"failing_test": "t/test_a.py::test_one"},
                      input_tokens=42, output_tokens=None, timestamp="2026-03-14T09:05:10Z",
                      dropped=("output_tokens",)),
    "http_call": dict(family="http_call",
                      target={"method": "POST", "host": "svc.example.test", "path_template": "/v1/items/{id}"},
                      args={"body": '{"name": "zed"}'}, outcome="ERROR", error_class="HTTP503",
                      error_location="svc.example.test/v1/items/{id}", input_tokens=None, output_tokens=None,
                      timestamp="2026-03-14T09:05:11Z", dropped=("failure", "input_tokens", "output_tokens")),
    "poll_status": dict(family="poll_status", target="ci/run/7", args={"backoff": 4}, outcome="NEUTRAL",
                        extra={"status": "pending"}, input_tokens=5, output_tokens=6,
                        timestamp="2026-03-14T09:05:12Z", dropped=("digest_after", "digest_before")),
    "write_file": dict(family="write_file", target="pkg/new.py", args={"content": "abc\n"}, outcome="SUCCESS",
                       changed=[{"path": "pkg/new.py", "before": None, "after": "abc\n"}], input_tokens=1,
                       output_tokens=2, timestamp="2026-03-14T09:05:13Z"),
    "build": dict(family="build", target="wheel", args={}, outcome="ERROR", error_class="BuildFail",
                  error_location="pkg/setup.cfg", input_tokens=3, output_tokens=4, timestamp="2026-03-14T09:05:14Z"),
    "plan": dict(family="plan", target=None, args={"action_type": "sketch"}, outcome="SUCCESS", input_tokens=9,
                 output_tokens=9, timestamp="2026-03-14T09:05:15Z"),
}

GOLDEN_TEXT = {
    "read_file": "09:05:07.250 read_file path=pkg/alpha.py lines=3-14 -> ok in=1234 out=56",
    "search": '09:00:00.000 search scope=pkg query="frobnicate widget" limit=20 -> ok in=88 out=7',
    "edit_file": ("09:05:08.000 edit_file path=pkg/alpha.py -> ok in=500 out=1000\n"
                  "  @ pkg/alpha.py h04875873af > h4e1f288b44"),
    "revert": ("09:05:09.999 revert paths=pkg/alpha.py,pkg/beta.py -> ok in=10 out=20\n"
               "  @ pkg/alpha.py h4e1f288b44 > h04875873af\n"
               "  @ pkg/beta.py hfed0fbb59f > h0"),
    "run_tests": ("09:05:10.000 run_tests suite=unit tests=t/test_a.py::test_one -> FAILED 3/1234567 passed "
                  "in=42 out=?\n"
                  "  ! AssertionError at t/test_a.py::test_one:660 (0x02d8659a)"),
    "http_call": ('09:05:11.000 http_call POST https://svc.example.test/v1/items/{id} body={"name": "zed"} '
                  "req=c42ebbde -> error in=? out=?\n"
                  "  ! ? at ?:? (0xc7172210)"),
    "poll_status": ("09:05:12.000 poll_status resource=ci/run/7 backoff=4 -> pending in=5 out=6\n"
                    "  [ workspace untracked]"),
    "write_file": ("09:05:13.000 write_file path=pkg/new.py -> ok in=1 out=2\n"
                   "  @ pkg/new.py h0 > h5b4cc071c2"),
    "build": ("09:05:14.000 build target=wheel -> FAILED in=3 out=4\n"
              "  ! BuildFail at pkg/setup.cfg:661 (pid 3779)"),
    "plan": "09:05:15.000 plan action=sketch -> ok in=9 out=9",
}

GOLDEN_IDS = {
    "read_file": [243, 143, 239, 143, 241, 141, 584, 7, 21, 146, 7438, 22, 146, 227, 140, 248, 140, 147, 38, 36, 146,
                  457, 228, 37, 146, 290],
    "search": [243, 143, 234, 143, 234, 141, 334, 8, 23, 146, 8831, 24, 146, 129, 8358, 2254, 129, 25, 146, 254, 140,
               147, 38, 36, 146, 322, 37, 146, 231],
    "edit_file": [243, 143, 239, 143, 242, 141, 334, 10, 21, 146, 7438, 140, 147, 38, 36, 146, 834, 37, 146, 434, 224,
                  4, 7438, 7475, 147, 7743],
    "revert": [243, 143, 239, 143, 243, 141, 1333, 12, 27, 146, 7438, 139, 4371, 140, 147, 38, 36, 146, 244, 37, 146,
               254, 4, 7438, 7743, 147, 7475, 4, 4371, 6509, 147, 5861],
    "run_tests": [243, 143, 239, 143, 244, 141, 334, 13, 28, 146, 9660, 29, 146, 2506, 143, 143, 4119, 140, 147, 40,
                  227, 142, 457, 790, 231, 41, 36, 146, 276, 37, 146, 148, 5, 9529, 44, 2506, 143, 143, 4119, 143,
                  994, 135, 224, 6154, 136],
    "http_call": [243, 143, 239, 143, 245, 141, 334, 16, 51, 49, 143, 142, 142, 3680, 155, 3367, 157, 31, 146, 155,
                  129, 7039, 129, 143, 129, 6272, 129, 157, 32, 146, 9555, 140, 147, 39, 36, 146, 148, 37, 146, 148,
                  5, 148, 44, 148, 143, 148, 135, 224, 7998, 136],
    "poll_status": [243, 143, 239, 143, 246, 141, 334, 17, 33, 146, 5311, 34, 146, 228, 140, 147, 43, 36, 146, 229, 37,
                    146, 230, 6, 46, 47, 152],
    "write_file": [243, 143, 239, 143, 247, 141, 334, 11, 21, 146, 9170, 140, 147, 38, 36, 146, 225, 37,
                   146, 226, 4, 9170, 5861, 147, 7442],
    "build": [243, 143, 239, 143, 248, 141, 334, 14, 30, 146, 6753, 140, 147, 40, 36, 146, 227, 37, 146, 228, 5, 8170,
              44, 8664, 143, 995, 135, 45, 711, 233, 136],
    "plan": [243, 143, 239, 143, 249, 141, 334, 18, 35, 146, 7011, 140, 147, 38, 36, 146, 233, 37, 146, 233],
}


def _golden(name: str) -> dict:
    return RawOp.make(**GOLDEN_OPS[name]).to_dict()


def _ref_tokenize(text: str) -> list[int]:
    """Spec §5 written independently of tokenizer.py (single-line text, no newlines)."""
    reserved = {w: 7 + i for i, w in enumerate(T.TOOL_NAMES + T.FIELD_AND_STATUS_WORDS)}
    punct = [chr(c) for c in range(33, 127) if not chr(c).isalnum() and chr(c) != "_"]
    groups = [str(i) for i in range(10)] + [f"{i:02d}" for i in range(100)] + [f"{i:03d}" for i in range(1000)]
    out = []
    for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_.\-/]*|[0-9]+|[^ \t]", text):
        s = m.group(0)
        if s[0].isalpha() or s[0] == "_":
            out.append(reserved.get(s, 2048 + int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 8192))
        elif s[0].isdigit():
            out.extend(224 + groups.index(s[i:i + 3]) for i in range(0, len(s), 3))
        else:
            out.append(128 + punct.index(s))
    return out


@pytest.mark.parametrize("name", sorted(GOLDEN_OPS))
def test_golden_text_and_ids(name):
    block = R.render_event(_golden(name), SID)
    assert block == GOLDEN_TEXT[name]
    assert _ids(block) == GOLDEN_IDS[name]
    # independent reference tokeniser agrees line by line (line markers handled separately)
    lines = block.split("\n")
    ref = _ref_tokenize(lines[0])
    for ln in lines[1:]:
        assert ln[:2] == "  " and ln[2] in "@![" and ln[3] == " "
        ref += [T.LINE_MARKERS[ln[2]]] + _ref_tokenize(ln[4:])
    assert ref == GOLDEN_IDS[name]


def test_golden_formulas():
    """The golden hunk ids and error line numbers follow the spec formulas."""
    assert _spec_hunk_id(SID, "pkg/alpha.py", ["x = 1"]) == "h04875873af"
    assert _spec_hunk_id(SID, "pkg/alpha.py", ["x = 2"]) == "h4e1f288b44"
    assert _spec_hunk_id(SID, "pkg/beta.py", ["new"]) == "hfed0fbb59f"
    assert _spec_hunk_id(SID, "pkg/new.py", ["abc", ""]) == "h5b4cc071c2"
    line = int(hashlib.sha256(f"{SID}:t/test_a.py::test_one".encode()).hexdigest()[:4], 16) % 900 + 1
    assert line == 660
    # req= comes from the event's render RNG
    d = _golden("http_call")
    seed = int(hashlib.sha256((R.canonical_json(d) + ":render").encode()).hexdigest()[:16], 16)
    assert f"req={random.Random(seed).getrandbits(32):08x}" in R.render_event(d, SID)


def test_golden_families_cover_grammar():
    assert set(GOLDEN_OPS) >= {"read_file", "search", "edit_file", "revert", "run_tests", "http_call",
                               "poll_status", "write_file", "build", "plan"}


# -- determinism -------------------------------------------------------------------


def test_deterministic_same_process():
    ops = [_golden(n) for n in sorted(GOLDEN_OPS)]
    a = R.render_session(ops, SID)
    b = R.render_session(json.loads(json.dumps(ops)), SID)
    assert a == b == R.render_session(ops, SID)
    assert T.encode_session(a) == T.encode_session(b)


def test_deterministic_across_hash_seeds():
    code = ("import json,sys; from agent_loop_lab.v03 import renderer as R, tokenizer as T; "
            "ops=json.load(sys.stdin); b=R.render_session(ops,'s'); print(json.dumps([b, T.encode_session(b).tokens]))")
    ops = json.dumps([_golden(n) for n in sorted(GOLDEN_OPS)])
    outs = {subprocess.run([sys.executable, "-c", code], input=ops, capture_output=True, text=True, check=True,
                           env={"PYTHONHASHSEED": seed, "PATH": ""}).stdout for seed in ("1", "2", "12345")}
    assert len(outs) == 1


def test_render_rng_per_event():
    """Render noise is seeded per event: changing one event leaves the others' noise alone."""
    ops = [_golden("http_call"), _golden("run_tests")]
    a = R.render_session(ops, SID)
    ops2 = [dict(ops[0], input_tokens=7), ops[1]]
    b = R.render_session(ops2, SID)
    assert a[1] == b[1] and a[0] != b[0]


# -- caps (synthetic worst cases) ----------------------------------------------------


def _long_op(**over) -> dict:
    words = [f"w{i}x" for i in range(200)]
    base = dict(family="run_tests", target="suite_" + "_".join(words[:3]),
                args={"tests": [f"t/{w}.py::case {w} {w}" for w in words[:40]]}, outcome="ERROR",
                error_class="E " * 30, error_location=" ".join(words[:50]), validation=("s", 123456, 7891011),
                changed=[{"path": f"p/{i}.py", "before": "a\nb\nc\nd", "after": "A\nb\nC\nd"} for i in range(9)],
                input_tokens=123456789, output_tokens=987654321, timestamp="2026-01-01T00:00:00Z",
                dropped=("digest_after", "digest_before"))
    base.update(over)
    return RawOp.make(**base).to_dict()


def _check_caps(block: str) -> None:
    lines = block.split("\n")
    assert len(T.tokenize_block(lines[0])) <= R.HEADER_MAX_TOKENS
    assert len(lines) - 1 <= R.BODY_MAX_LINES
    for ln in lines[1:]:
        assert len(T.tokenize_block("\n" + ln)) <= R.LINE_MAX_TOKENS + 1
    assert len(T.tokenize_block(block)) + 1 <= 191 < T.O


def test_caps_worst_case_block():
    block = R.render_event(_long_op(), SID)
    _check_caps(block)
    lines = block.split("\n")
    assert lines[0].endswith(" -> FAILED 123456/7891011 passed in=123456789 out=987654321")  # suffix intact
    assert R.TRUNC in lines[0]
    # 9 changed files (one line each, r2) + error + workspace: 4 file lines, the 4th a TRUNC count, then ! and [
    assert [ln[:3] for ln in lines[1:]] == ["  @"] * 4 + ["  !", "  ["]
    assert lines[4] == "  @ TRUNC 6"
    assert lines[5].endswith(R.TRUNC) and len(T.tokenize_block("\n" + lines[5])) == R.LINE_MAX_TOKENS + 1


def test_hunk_truncation_without_tail():
    d = _long_op(outcome="SUCCESS", error_class=None, error_location=None, dropped=())
    lines = R.render_event(d, SID).split("\n")[1:]
    assert len(lines) == 5 and lines[-1] == "  @ TRUNC 5" and all(HUNK_LINE.match(x) for x in lines[:4])
    d = _long_op(outcome="SUCCESS", error_class=None, error_location=None, dropped=(),
                 changed=[{"path": f"p/{i}.py", "before": "a", "after": "b"} for i in range(5)])
    lines = R.render_event(d, SID).split("\n")[1:]
    assert len(lines) == 5 and all(HUNK_LINE.match(x) for x in lines)  # exactly 5: no TRUNC


@pytest.mark.parametrize("field,family,args,budget", [
    ("query", "search", {"query": " ".join(f"q{i}" for i in range(40))}, 12),
    ("tests", "run_tests", {"tests": [f"t{i}" for i in range(40)]}, 24),
    ("body", "http_call", {"body": json.dumps({f"k{i}": i for i in range(40)})}, 16),
])
def test_field_budgets(field, family, args, budget):
    target = {"method": "GET", "host": "h.test", "path_template": "/x"} if family == "http_call" else "t"
    d = RawOp.make(family=family, target=target, args=args, outcome="SUCCESS", input_tokens=1, output_tokens=2,
                   timestamp="2026-01-01T00:00:00Z").to_dict()
    header = R.render_event(d, SID)
    m = re.search(field + r'="?(.*?) TRUNC', header)
    assert m, header
    assert len(T.tokenize_block(m.group(1))) == budget - 1  # budget tokens including TRUNC
    assert header.endswith(" -> ok in=1 out=2")


def test_paths_budget_and_revert_header():
    paths = [f"p{i}/f.py" for i in range(30)]
    d = RawOp.make(family="revert", target=paths, args={"revision": "a" * 64}, outcome="SUCCESS",
                   input_tokens=1, output_tokens=1, timestamp="2026-01-01T00:00:00Z").to_dict()
    header = R.render_event(d, SID)
    value = header.split(" paths=", 1)[1].split(" -> ")[0]
    assert value.endswith(" TRUNC") and len(T.tokenize_block(value)) == 16
    assert "to=" not in header and "a" * 64 not in header and "revision" not in header


def test_truncated_field_keeps_source_spacing():
    d = RawOp.make(family="run_tests", target="u", args={"tests": [f"a/b.py::case {i}" for i in range(20)]},
                   outcome="SUCCESS", timestamp="2026-01-01T00:00:00Z").to_dict()
    header = R.render_event(d, SID)
    assert "tests=a/b.py::case 0,a/b.py::case 1," in header and " : : " not in header


def test_non_ascii_and_embedded_newlines():
    for q in ("café crème", "a\n@ b\n! c\n[ d", "tab\there"):
        d = RawOp.make(family="search", target="s", args={"query": q * 5}, outcome="SUCCESS", input_tokens=1,
                       output_tokens=1, timestamp="2026-01-01T00:00:00Z").to_dict()
        block = R.render_event(d, SID)
        assert block.isascii() and "\n" not in block  # one header line, no forged body line
        _check_caps(block)
    d = _long_op(error_location="lib/über.py\nfake", error_class="Err☃")
    block = R.render_event(d, SID)
    assert block.isascii() and "lib/?ber.py?fake" in block
    _check_caps(block)


def test_line_markers():
    block = R.render_event(_long_op(), SID)
    pairs = T.tokenize_block(block)
    markers = [t for t, _ in pairs if t in (T.NL, T.NL_AT, T.NL_BANG, T.NL_BRACKET)]
    assert markers == [T.NL_AT] * 4 + [T.NL_BANG, T.NL_BRACKET]
    assert T.tokenize_block("x\n  @ y\n! z\n[ w\nv") == [
        (T.word_id("x"), "x"), (T.NL_AT, "\n@"), (T.word_id("y"), "y"), (T.NL_BANG, "\n!"), (T.word_id("z"), "z"),
        (T.NL_BRACKET, "\n["), (T.word_id("w"), "w"), (T.NL, "\n"), (T.word_id("v"), "v")]


def test_missing_observations():
    d = RawOp.make(family="run_tests", target="u", args={}, outcome="SUCCESS", validation=("u", 5, 9),
                   timestamp=None, dropped=("validation_suite", "validation_total")).to_dict()
    assert R.render_event(d, SID) == "? run_tests suite=? -> ok 5/? passed in=? out=?"


# -- hunk ids ------------------------------------------------------------------------


def test_hunk_id_stability_and_undo_swap():
    a, b, c = "x\nline A\ny", "x\nline B\ny", "x\nline C\ny"
    seq = [(a, b), (b, c), (c, b), (b, a), (a, b)]
    ops = [RawOp.make(family="edit_file", target="m.py", args={"patch": ""}, outcome="SUCCESS",
                      changed=[{"path": "m.py", "before": u, "after": v}], timestamp="2026-01-01T00:00:00Z").to_dict()
           for u, v in seq]
    ids = [HUNK_LINE.match(blk.split("\n")[1]).groups()[1:] for blk in R.render_session(ops, "s1")]
    hA, hB, hC = (_spec_hunk_id("s1", "m.py", [t]) for t in ("line A", "line B", "line C"))
    assert ids == [(hA, hB), (hB, hC), (hC, hB), (hB, hA), (hA, hB)]
    assert ids[2] == ids[1][::-1] and ids[4] == ids[0]  # undo swaps; a return reuses the earlier ids
    # session- and path-scoped
    other = HUNK_LINE.match(R.render_session(ops[:1], "s2")[0].split("\n")[1]).groups()[1:]
    assert other != ids[0]
    moved = RawOp.make(family="edit_file", target="n.py", args={"patch": ""}, outcome="SUCCESS",
                       changed=[{"path": "n.py", "before": a, "after": b}]).to_dict()
    assert HUNK_LINE.match(R.render_event(moved, "s1").split("\n")[1]).groups()[1:] != ids[0]


def test_hunk_empty_sides():
    d = RawOp.make(family="write_file", target="n.py", args={"content": "q"}, outcome="SUCCESS",
                   changed=[{"path": "n.py", "before": None, "after": "q"}]).to_dict()
    assert R.render_event(d, "s").split("\n")[1] == f"  @ n.py h0 > {_spec_hunk_id('s', 'n.py', ['q'])}"
    d = RawOp.make(family="revert", target=["n.py"], args={}, outcome="SUCCESS",
                   changed=[{"path": "n.py", "before": "q", "after": None}]).to_dict()
    assert R.render_event(d, "s").split("\n")[1] == f"  @ n.py {_spec_hunk_id('s', 'n.py', ['q'])} > h0"


# -- tokeniser -----------------------------------------------------------------------


def test_offset_bound_raises():
    ok_block = " ".join(["a"] * 191)  # 191 tokens + EOE = offsets 0..191
    enc = T.encode_session([ok_block])
    assert max(enc.offset) == 191 < T.O
    with pytest.raises(T.TokenizerError):
        T.encode_session([ok_block + " a"])  # EOE would sit at offset 192
    with pytest.raises(T.TokenizerError):
        T.encode_session(["x", ok_block + " a"])  # any block, not just the first


def test_encode_layout():
    enc = T.encode_session(["a b", "c"])
    assert enc.tokens[0] == T.BOS and enc.event_ord == [0, 1, 1, 1, 2, 2] and enc.offset == [0, 0, 1, 2, 0, 1]
    assert enc.eoe_index == [3, 5] and [enc.tokens[i] for i in enc.eoe_index] == [T.EOE, T.EOE]
    assert enc.surface == ["<BOS>", "a", "b", "<EOE>", "c", "<EOE>"]
    p = T.prefix(enc, 0)
    assert p.tokens == enc.tokens[:4] and p.eoe_index == [3]


def test_digit_grouping():
    assert T.digit_groups("1234567") == ["123", "456", "7"]
    assert T.digit_groups("0007") == ["000", "7"] and T.digit_groups("42") == ["42"]
    assert _ids("1234567") == [T.DIGIT_IDS["123"], T.DIGIT_IDS["456"], T.DIGIT_IDS["7"]]
    assert T.DIGIT_IDS["0"] == 224 and T.DIGIT_IDS["9"] == 233 and T.DIGIT_IDS["00"] == 234
    assert T.DIGIT_IDS["99"] == 333 and T.DIGIT_IDS["000"] == 334 and T.DIGIT_IDS["999"] == 1333
    assert T.DIGIT_IDS["7"] != T.DIGIT_IDS["07"] != T.DIGIT_IDS["007"]
    assert _ids("a1b2") == [T.word_id("a1b2")]  # digits inside a word stay in the word
    assert _ids("12ab") == [T.DIGIT_IDS["12"], T.word_id("ab")]


def test_vocabulary_ranges():
    rid = list(T.RESERVED_IDS.values())
    assert len(set(rid)) == len(rid) == len(T.RESERVED_WORDS) and all(7 <= i <= 127 for i in rid)
    assert len(set(T.RESERVED_WORDS)) == len(T.RESERVED_WORDS)
    assert set(T.TOOL_NAMES) <= set(T.RESERVED_WORDS)
    spec_words = ("path lines scope query limit bytes paths suite tests target body req resource backoff action in "
                  "out ok error FAILED passed ready pending at pid workspace untracked TRUNC https GET POST PUT PATCH "
                  "DELETE").split()
    assert set(T.RESERVED_WORDS) == set(T.TOOL_NAMES) | set(spec_words)
    pid = list(T.PUNCT_IDS.values())
    assert len(set(pid)) == len(pid) and all(128 <= i <= 223 for i in pid)
    assert all(not c.isalnum() and c != "_" and 33 <= ord(c) < 127 for c in T.PUNCT_IDS)
    did = list(T.DIGIT_IDS.values())
    assert len(set(did)) == len(did) == 1110 and min(did) == 224 and max(did) == 1333
    assert (T.PAD, T.BOS, T.EOE, T.NL, T.NL_AT, T.NL_BANG, T.NL_BRACKET) == (0, 1, 2, 3, 4, 5, 6)
    for w in ("alpha", "x", "_y", "a.b/c-d"):
        i = T.word_id(w)
        assert 2048 <= i < T.V and i == 2048 + int(hashlib.sha256(w.encode()).hexdigest()[:8], 16) % 8192
    for w in T.RESERVED_WORDS:
        assert T.word_id(w) == T.RESERVED_IDS[w]


def test_tokenize_spans_match_source():
    text = "12:05 http_call GET https://a.b/c{d} body={\"k\": 1234} -> ok\n  @ p h0 > h1"
    for tid, s, a, b in T.tokenize_spans(text):
        if tid in (T.NL_AT, T.NL_BANG, T.NL_BRACKET):
            assert text[a] == "\n" and text[b - 1] == s[1]
        else:
            assert text[a:b] == s
    assert [(t, s) for t, s, _, _ in T.tokenize_spans(text)] == T.tokenize_block(text)
    with pytest.raises(T.TokenizerError):
        T.tokenize_block("café")


# -- reserved words vs generator vocabulary (spec §11) --------------------------------

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-/]*")
# Grammar words the generators pass in their grammar position by design (family names, HTTP methods).
_GRAMMAR = set(T.TOOL_NAMES) | {"GET", "POST", "PUT", "PATCH", "DELETE"}
# Reserved words that ALSO occur as generator vocabulary. Frozen: the test fails if this set grows or shrinks.
# Rendered in value positions (dev counts in artifacts/reports/v03_dev_audits.json):
#   path, limit, query   world.NOUNS (test-case and search words), PATH_TEMPLATES, http body keys
#   backoff              world.MODULES (test-case words, search queries)
#   target               http body key (families/transient_external.py)
#   tests                directory name (list_dir path=tests)
# Not rendered (code text, control-flow modes or status strings that the renderer derives itself):
#   ok, out, req, ready, pending, suite
KNOWN_RESERVED_VOCAB_COLLISIONS = frozenset(
    {"path", "limit", "query", "backoff", "target", "tests", "ok", "out", "req", "ready", "pending", "suite"})


def _generator_vocabulary() -> dict[str, set[str]]:
    """Words of the string literals in synth/ and synth_ood/ that can be generator vocabulary.

    Excluded as code rather than vocabulary: docstrings, raise/assert messages, f-strings, dict keys,
    subscripts, comparison operands, ``drop=``/``dropped=``/``family=``-style keyword values, the family
    argument of the engines' operation calls, and string-method arguments.
    """
    code_kw = {"drop", "dropped", "family", "to", "kind", "rel_kind", "mode"}
    family_calls = {"perform", "_commit", "_finish_mutation", "_mutate", "_package_request"}
    str_methods = {"get", "pop", "setdefault", "startswith", "endswith", "split", "partition", "join", "format",
                   "getattr", "hasattr"}
    out: dict[str, set[str]] = {}
    for sub in ("synth", "synth_ood"):
        for p in sorted((SRC / sub).rglob("*.py")):
            tree = ast.parse(p.read_text(encoding="utf-8"))
            skip: set[int] = set()

            def drop_all(node):
                skip.update(id(c) for c in ast.walk(node))

            for n in ast.walk(tree):
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body \
                        and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
                    skip.add(id(n.body[0].value))
                if isinstance(n, (ast.Raise, ast.JoinedStr, ast.Assert, ast.Compare)):
                    drop_all(n)
                elif isinstance(n, ast.Dict):
                    for k in n.keys:
                        if k is not None:
                            drop_all(k)
                elif isinstance(n, ast.Subscript):
                    drop_all(n.slice)
                elif isinstance(n, ast.keyword) and n.arg in code_kw:
                    drop_all(n.value)
                elif isinstance(n, ast.Call):
                    name = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
                    if name in str_methods and n.args:
                        drop_all(n.args[0])
                    if name in family_calls and n.args:
                        drop_all(n.args[0])
            for n in ast.walk(tree):
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip:
                    for w in _WORD.findall(n.value):
                        out.setdefault(w, set()).add(f"{p.relative_to(SRC)}:{n.lineno}")
    return out


def _world_pool_words() -> set[str]:
    from agent_loop_lab.synth import world

    words: set[str] = set()

    def walk(v):
        if isinstance(v, str):
            words.update(_WORD.findall(v))
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(k)
                walk(x)
        elif isinstance(v, (tuple, list, set, frozenset)):
            for x in v:
                walk(x)
        elif hasattr(v, "__dataclass_fields__"):
            for f in v.__dataclass_fields__:
                walk(getattr(v, f))

    for name in dir(world):
        if name.lstrip("_").isupper():
            walk(getattr(world, name))
    return words


def test_reserved_words_vs_generator_vocabulary():
    vocab = _generator_vocabulary()
    pools = _world_pool_words()
    assert {"path", "limit", "query", "backoff"} <= pools  # world.NOUNS / MODULES are scanned
    collisions = (set(vocab) | pools) & set(T.RESERVED_WORDS) - _GRAMMAR
    assert collisions == KNOWN_RESERVED_VOCAB_COLLISIONS, {w: sorted(vocab.get(w, ()))[:3] for w in collisions}


def _error_classes() -> set[str]:
    from agent_loop_lab.synth import world
    from agent_loop_lab.datasets.build import DEFAULT_MASTER_SEED
    from agent_loop_lab.datasets.ood import ood_seed, ood_session_id
    from agent_loop_lab.synth_ood import FAMILIES as OOD_FAMILIES
    from agent_loop_lab.v03.rawop import capture_raw
    from tests import synth_helpers as H

    classes = set(world.HTTP_ERRORS) | set(world.AUTH_ERRORS)
    for lang in world.LANGUAGES.values():
        classes |= set(lang.errors) | set(lang.lint_errors) | set(lang.build_errors)
    for fam, gen in OOD_FAMILIES.items():
        for k in range(3):
            seed = ood_seed(DEFAULT_MASTER_SEED, fam, k)
            sid = ood_session_id(seed)
            with capture_raw() as cap:
                events, _ = gen(random.Random(seed), sid, H.cfg())
            classes |= {r.error_class for r in cap.pop(sid, events) if r.error_class}
    return classes


def test_error_classes_disjoint_from_reserved_words():
    classes = _error_classes()
    assert {"ReplicaDivergence", "PeerReset", "AssertionError"} & classes
    words = {w for c in classes for w in _WORD.findall(c)}
    assert not words & set(T.RESERVED_WORDS), sorted(words & set(T.RESERVED_WORDS))


# -- property pass over every dev session (spec §4.6, §11) ----------------------------


@pytest.fixture(scope="module")
def dev_pass():
    if not (DATA / "dev.raw.jsonl.gz").exists():
        pytest.skip("datasets/v0_3/dev.raw.jsonl.gz not built")
    from agent_loop_lab.v03.build_raw import iter_raw

    st = dict(sessions=0, blocks=0, max_header=0, max_lines=0, max_line=0, max_block=0, max_session=0,
              hex64=0, plain_nl=0, bad_body=0, bad_hunk_ids=0, hunk_lines=0, revert_headers=0,
              revert_bad=0, ascii_bad=0, error_classes=set(), nondeterministic=0)
    for n, (sid, raw) in enumerate(iter_raw(DATA, "dev")):
        dicts = [r.to_dict() for r in raw]
        blocks = R.render_session(dicts, sid)
        if n % 500 == 0 and R.render_session(dicts, sid) != blocks:
            st["nondeterministic"] += 1
        enc = T.encode_session(blocks)  # raises on offset >= O
        st["sessions"] += 1
        st["max_session"] = max(st["max_session"], len(enc.tokens))
        prev = 0
        for e in enc.eoe_index:
            st["max_block"] = max(st["max_block"], e - prev)
            prev = e
        st["plain_nl"] += sum(t == T.NL for t in enc.tokens)
        for r, blk in zip(raw, blocks):
            st["blocks"] += 1
            if r.error_class:
                st["error_classes"].add(r.error_class)
            if HEX64.search(blk):
                st["hex64"] += 1
            if not blk.isascii():
                st["ascii_bad"] += 1
            lines = blk.split("\n")
            st["max_header"] = max(st["max_header"], len(T.tokenize_block(lines[0])))
            st["max_lines"] = max(st["max_lines"], len(lines) - 1)
            for ln in lines[1:]:
                st["max_line"] = max(st["max_line"], len(T.tokenize_block("\n" + ln)))
                if ln[:4] not in ("  @ ", "  ! ", "  [ "):
                    st["bad_body"] += 1
            if r.family == "revert":
                st["revert_headers"] += 1
                if " to=" in lines[0] or "revision" in lines[0] or str(r.args.get("revision")) in blk:
                    st["revert_bad"] += 1
            # hunk lines agree with the spec formula, computed here from the raw diff
            expect = []
            for ch in r.changed:  # r2: one line per changed file, removed vs added lines across all hunks
                hs = _spec_hunks(ch["before"], ch["after"])
                if not hs:
                    continue
                rem = [x for a, _ in hs for x in a]
                add = [x for _, b in hs for x in b]
                expect.append(f"  @ {ch['path']} {_spec_hunk_id(sid, ch['path'], rem)} > "
                              f"{_spec_hunk_id(sid, ch['path'], add)}")
            got = [ln for ln in lines[1:] if ln.startswith("  @ ")]
            st["hunk_lines"] += len(got)
            if got and got[-1].startswith("  @ TRUNC "):
                ok = got[:-1] == expect[: len(got) - 1] and int(got[-1].split()[-1]) == len(expect) - len(got) + 1
            else:
                ok = got == expect
            st["bad_hunk_ids"] += not ok
    return st


def test_dev_caps_and_markers(dev_pass):
    st = dev_pass
    print(f"\nv0.3 dev render pass: {st['sessions']} sessions, {st['blocks']} blocks; max header "
          f"{st['max_header']}, body lines {st['max_lines']}, body line {st['max_line']} (incl marker), block "
          f"{st['max_block']} (incl EOE), session {st['max_session']} tokens")
    assert st["sessions"] == 12000
    assert st["max_header"] <= 64 and st["max_lines"] <= 6 and st["max_line"] <= 21 and st["max_block"] <= 191
    assert st["plain_nl"] == 0 and st["bad_body"] == 0 and st["ascii_bad"] == 0 and st["nondeterministic"] == 0
    assert st["hunk_lines"] > 0 and st["bad_hunk_ids"] == 0


def test_dev_no_hex64_and_revert_headers(dev_pass):
    assert dev_pass["hex64"] == 0
    assert dev_pass["revert_headers"] > 0 and dev_pass["revert_bad"] == 0


def test_dev_error_classes_disjoint_from_reserved(dev_pass):
    words = {w for c in dev_pass["error_classes"] for w in _WORD.findall(c)}
    assert words and not words & set(T.RESERVED_WORDS)


def test_dev_maxima_match_manifest(dev_pass):
    m = json.loads((DATA / "MANIFEST_v0_3.json").read_text(encoding="utf-8"))
    stats = (m.get("token_statistics") or {}).get("splits", {}).get("dev")
    if stats is None:
        pytest.skip("manifest has no token statistics yet")
    assert m["renderer_source_sha256"] == hashlib.sha256((SRC / "v03" / "renderer.py").read_bytes()).hexdigest()
    assert m["tokeniser_source_sha256"] == hashlib.sha256((SRC / "v03" / "tokenizer.py").read_bytes()).hexdigest()
    assert stats["tokens_per_session"]["max"] == dev_pass["max_session"]
    assert stats["tokens_per_block"]["max"] == dev_pass["max_block"]
