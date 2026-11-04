"""Tests for spec v0.2 §3.2 fingerprint canonicalisation (contracts §A)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

from agent_loop_lab import fingerprint as fp
from agent_loop_lab.fingerprint import (
    FAMILY_CATEGORY,
    FingerprintError,
    canonical_json,
    failure_fingerprint,
    intent_fingerprint,
    normalise_location,
    normalise_message,
    normalise_path,
    sha256_hex,
    workspace_digest,
)
from agent_loop_lab.schema import OperationCategory


# ---------------------------------------------------------------------------
# Independent reference (stdlib only; never calls the module under test)
# ---------------------------------------------------------------------------


def _ref_cj(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _ref_h(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def _ref_intent(family, target, args) -> str:
    return _ref_h(_ref_cj({"family": family, "target": target, "args": args}))


HTTP_GET_T = {"method": "GET", "host": "api.example.com", "path_template": "/v1/items/{id}"}
HTTP_POST_T = {"method": "POST", "host": "api.example.com", "path_template": "/v1/items"}

# Minimal valid (target, args) per family, used for parametrised property tests.
MINIMAL: dict[str, tuple[object, dict]] = {
    "read_file": ("a.py", {}),
    "search": (".", {"query": "foo"}),
    "list_dir": (".", {}),
    "edit_file": ("a.py", {"patch": "+x"}),
    "write_file": ("a.py", {"content": "x"}),
    "revert": (["a.py"], {}),
    "run_tests": ("unit", {}),
    "build": ("app", {}),
    "lint": ("src", {}),
    "http_call": (HTTP_POST_T, {"body": "x"}),
    "poll_status": ("job-1", {}),
    "plan": (None, {"action_type": "decompose"}),
    "ask": (None, {"action_type": "clarify"}),
    "yield": (None, {"action_type": "handoff"}),
}
ALL_FAMILIES = sorted(MINIMAL)

GLOBAL_EXCLUDED = (
    "timestamp",
    "attempt",
    "attempt_number",
    "retry",
    "retry_count",
    "timeout",
    "request_id",
    "reasoning",
    "free_text",
    "text",
)

FAMILY_EXCLUDED = {
    "search": ("limit",),
    "run_tests": ("verbosity", "parallelism"),
    "build": ("flags",),
    "lint": ("flags",),
    "poll_status": ("backoff",),
}


# ---------------------------------------------------------------------------
# FAMILY_CATEGORY matches the §3.2 table
# ---------------------------------------------------------------------------


def test_family_category_matches_spec_table():
    I, M, V, E, C = (
        OperationCategory.INSPECT,
        OperationCategory.MUTATE,
        OperationCategory.VALIDATE,
        OperationCategory.EXTERNAL,
        OperationCategory.CONTROL,
    )
    expected = {
        "read_file": I,
        "search": I,
        "list_dir": I,
        "edit_file": M,
        "write_file": M,
        "revert": M,
        "run_tests": V,
        "build": V,
        "lint": V,
        "http_call": E,
        "poll_status": E,
        "plan": C,
        "ask": C,
        "yield": C,
    }
    assert dict(FAMILY_CATEGORY) == expected
    assert set(FAMILY_CATEGORY) == set(MINIMAL)


def test_family_category_is_read_only():
    with pytest.raises(TypeError):
        FAMILY_CATEGORY["new_family"] = OperationCategory.INSPECT  # type: ignore[index]


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_every_category_family_is_accepted(family):
    target, args = MINIMAL[family]
    h = intent_fingerprint(family, target, args)
    assert len(h) == 64 and int(h, 16) >= 0


# ---------------------------------------------------------------------------
# Golden behaviour: literal hex computed once with a stdlib-only script
# ---------------------------------------------------------------------------

GOLDEN = {
    "read_file": "d71eea88542ecfc9eba02a96da9405711898cde6d1de7794e1479be4c3cce093",
    "search": "8cd4c64ebb6748fcee757d8f2719fcee1fe5fbe2eb607ff13d6324a6f325b35e",
    "list_dir": "f366e64c1c2586f5a5fdeeeb8886dbdcada031221608b67c598e2851432d0201",
    "edit_file": "34c09a249c0ad08bccedbf9b52515ba43570d38b9dfc639bc229ae8612b2bb29",
    "write_file": "bf1f880835e9fbbe34bae6fa58f735fba5b6d862c8197461db5813e56766d416",
    "revert": "4665bf77fa7e1955028855f29b29e3bd1d430462f90aad837bf13e9ac9cf6e7d",
    "run_tests": "4f7fe4f7e02d3aabbf49e41052ba443ffda6780db806644571fc279c68f26786",
    "build": "34f277a87bbf70200b311bf3613fbe82ec6ab5b7fc45894da92115dd4279a95e",
    "lint": "e1c5aabb551b88f782f7479b8e8884f4920bf2fff71deb82f387c2617eadfdff",
    "http_get": "dd74dfec1a2da6361e2513cb63dd9680a80cfb7a08132b24499d70f1af675565",
    "http_post": "d4f2077d42eb8b93585beeecd05d160a7e7dc2fc8fef789b9f66dc2b69963128",
    "poll_status": "f79945a821ecd6fc63ea6be839c44aca1dd6316100b6396af6a0c129eccf031a",
    "plan": "1175be294903d7e8ea792b2f5538cc1c4d79ab6a32959011cd18522a5ff3877e",
    "ask": "bd04f3f2c9c01e6d070cfec545ad609e80a6d469298c9b108e23fc01307e0495",
    "yield": "a65b0e8ec2682079057fef69adb195af9543ba289d64d81742688ad2410d3a6c",
    "failure": "8505a445cc5b01496efb8325d52dd470829640e89886f57ecbe2764aefec7d39",
    "workspace": "2df3b8e6aefca56974ebc5cbf4aa20219f5660ca660513966c2aeea9ead63d85",
}

# Messy-but-equivalent inputs: each must normalise to the golden payload.
GOLDEN_CALLS = {
    "read_file": ("read_file", "./src//app.py", {"line_start": 10, "line_end": 20, "timestamp": "t"}),
    "search": ("search", "src/", {"query": "  DEF   Café\tMAIN ", "limit": 50}),
    "list_dir": ("list_dir", "src\\pkg\\", {}),
    "edit_file": ("edit_file", "src/app.py", {"patch": "-old  \r\n+new\t\r\n\r\n"}),
    "write_file": ("write_file", "notes.txt", {"content": "hello\n"}),
    "revert": ("revert", ["src/b.py", "./a.py", "a.py"], {"revision": "abc123"}),
    "run_tests": ("run_tests", " unit ", {"tests": ["t::b", "t::a", "t::b"], "verbosity": 2}),
    "build": ("build", " //app:bin ", {"flags": ["-O2"]}),
    "lint": ("lint", "src", None),
    "http_get": (
        "http_call",
        {"method": "get", "host": "API.Example.COM", "path_template": "/v1/items/{id}"},
        {"body": "ignored", "request_id": "r-1"},
    ),
    "http_post": (
        "http_call",
        {"method": "post", "host": "api.example.com", "path_template": "/v1/items"},
        {"body": '{ "b": [1, 2],\n  "a": 1 }'},
    ),
    "poll_status": ("poll_status", " job-42 ", {"backoff": 2.5, "attempt": 3}),
    "plan": ("plan", None, {"action_type": "Decompose", "free_text": "I will..."}),
    "ask": ("ask", None, {"action_type": "CLARIFY"}),
    "yield": ("yield", None, {"action_type": "handoff", "reasoning": "done"}),
}


@pytest.mark.parametrize("key", sorted(GOLDEN_CALLS))
def test_intent_golden(key):
    family, target, args = GOLDEN_CALLS[key]
    assert intent_fingerprint(family, target, args) == GOLDEN[key]


def test_failure_golden():
    assert failure_fingerprint("run_tests", " AssertionError ", "tests/test_a.py::test_x:42") == GOLDEN["failure"]


def test_workspace_golden():
    assert workspace_digest({"src/b.py": b"B", "./a.txt": "A"}) == GOLDEN["workspace"]


# ---------------------------------------------------------------------------
# Non-circular: independently recompute expected hex in-test with hashlib+json
# ---------------------------------------------------------------------------


def test_independent_recompute_read_file():
    expected = _ref_intent("read_file", "src/app.py", {"line_range": [3, None]})
    assert intent_fingerprint("read_file", "src/./app.py", {"line_start": 3}) == expected
    whole = _ref_intent("read_file", "src/app.py", {"line_range": [None, None]})
    assert intent_fingerprint("read_file", "src/app.py") == whole


def test_independent_recompute_search_non_ascii():
    expected = _ref_intent("search", "docs/ünï", {"query": "naïve résumé"})
    assert intent_fingerprint("search", "docs\\ünï", {"query": "NAÏVE\n RÉSUMÉ"}) == expected
    # ensure_ascii=False is pinned: the escaped form would differ.
    escaped = hashlib.sha256(
        json.dumps(
            {"family": "search", "target": "docs/ünï", "args": {"query": "naïve résumé"}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert expected != escaped


def test_independent_recompute_edit_file():
    patch_digest = hashlib.sha256(b"@@ -1 +1 @@\n-a\n+b").hexdigest()
    expected = _ref_intent("edit_file", "x/y.py", {"patch_sha256": patch_digest})
    assert intent_fingerprint("edit_file", "x/z/../y.py", {"patch": "@@ -1 +1 @@\r\n-a \n+b\r\n\n\n"}) == expected


def test_independent_recompute_http_post_json_body():
    body_digest = hashlib.sha256(_ref_cj({"name": "x", "tags": ["a", "b"]})).hexdigest()
    expected = _ref_intent(
        "http_call",
        {"method": "PATCH", "host": "h.io", "path_template": "/u/{id}"},
        {"body_sha256": body_digest},
    )
    got = intent_fingerprint(
        "http_call",
        {"method": "patch", "host": "H.IO", "path_template": "/u/{id}"},
        {"body": b'{"tags": ["a", "b"], "name": "x"}'},
    )
    assert got == expected


def test_independent_recompute_failure_and_workspace():
    expected_f = _ref_h(_ref_cj({"family": "build", "error_class": "LinkError", "location": "src/pkg/mod.c"}))
    assert failure_fingerprint("build", "LinkError", "/home/u/proj/src/pkg/mod.c:12:3") == expected_f
    expected_w = _ref_h(_ref_cj([["a", _ref_h(b"1")], ["b/c", _ref_h(b"2")]]))
    assert workspace_digest({"b/c": "2", "a": b"1"}) == expected_w


# ---------------------------------------------------------------------------
# Excluded-argument insensitivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ALL_FAMILIES)
@pytest.mark.parametrize("key", GLOBAL_EXCLUDED)
def test_global_excluded_args_ignored(family, key):
    target, args = MINIMAL[family]
    base = intent_fingerprint(family, target, args)
    for value in ("v1", 7, None, ["x"], {"nested": True}):
        assert intent_fingerprint(family, target, {**args, key: value}) == base


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_all_global_excluded_together(family):
    target, args = MINIMAL[family]
    noisy = {**args, **{k: f"noise-{k}" for k in GLOBAL_EXCLUDED}}
    assert intent_fingerprint(family, target, noisy) == intent_fingerprint(family, target, args)


@pytest.mark.parametrize("family,key", [(f, k) for f, ks in FAMILY_EXCLUDED.items() for k in ks])
def test_family_excluded_args_ignored(family, key):
    target, args = MINIMAL[family]
    base = intent_fingerprint(family, target, args)
    assert intent_fingerprint(family, target, {**args, key: 1}) == base
    assert intent_fingerprint(family, target, {**args, key: ["a", "b"]}) == base


@pytest.mark.parametrize("key", sorted({k for ks in FAMILY_EXCLUDED.values() for k in ks}))
def test_family_excluded_args_rejected_on_other_families(key):
    owners = {f for f, ks in FAMILY_EXCLUDED.items() if key in ks}
    for family in ALL_FAMILIES:
        if family in owners:
            continue
        target, args = MINIMAL[family]
        with pytest.raises(FingerprintError):
            intent_fingerprint(family, target, {**args, key: 1})


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "PUT", "DELETE"])
def test_http_idempotent_body_ignored(method):
    t = {"method": method, "host": "h", "path_template": "/p"}
    base = intent_fingerprint("http_call", t)
    assert intent_fingerprint("http_call", t, {"body": "a"}) == base
    assert intent_fingerprint("http_call", t, {"body": b"\xff\xfe"}) == base
    assert intent_fingerprint("http_call", t, {"body": None}) == base
    assert base == _ref_intent("http_call", t, {})


@pytest.mark.parametrize("family", [f for f in ALL_FAMILIES if not MINIMAL[f][1]])
def test_args_none_equals_empty(family):
    target = MINIMAL[family][0]
    assert intent_fingerprint(family, target, None) == intent_fingerprint(family, target, {})


# ---------------------------------------------------------------------------
# Included-argument sensitivity
# ---------------------------------------------------------------------------


def test_read_file_line_range_sensitivity():
    a = intent_fingerprint("read_file", "a.py", {"line_start": 1, "line_end": 10})
    b = intent_fingerprint("read_file", "a.py", {"line_start": 1, "line_end": 11})
    c = intent_fingerprint("read_file", "a.py", {"line_start": 2, "line_end": 10})
    d = intent_fingerprint("read_file", "a.py")
    assert len({a, b, c, d}) == 4
    assert intent_fingerprint("read_file", "a.py", {"line_start": None, "line_end": None}) == d
    assert intent_fingerprint("read_file", "b.py") != d


def test_search_query_sensitivity_and_normalisation():
    a = intent_fingerprint("search", "src", {"query": "foo bar"})
    assert intent_fingerprint("search", "src", {"query": "  FOO\t\n  bar "}) == a
    assert intent_fingerprint("search", "src", {"query": "foobar"}) != a
    assert intent_fingerprint("search", "tests", {"query": "foo bar"}) != a


def test_edit_file_patch_sensitivity():
    a = intent_fingerprint("edit_file", "a.py", {"patch": "+x = 1"})
    assert intent_fingerprint("edit_file", "a.py", {"patch": "+x = 2"}) != a
    assert intent_fingerprint("edit_file", "b.py", {"patch": "+x = 1"}) != a
    # leading whitespace is significant
    assert intent_fingerprint("edit_file", "a.py", {"patch": " +x = 1"}) != a


def test_edit_file_patch_normalisation():
    canonical = intent_fingerprint("edit_file", "a.py", {"patch": "a\nb"})
    for variant in ("a \r\nb\t\r\n\r\n", "a\rb", "a\r\nb", "a  \nb\n\n\n", "a\nb   \n \n\t\n"):
        assert intent_fingerprint("edit_file", "a.py", {"patch": variant}) == canonical, repr(variant)
    # interior blank lines are preserved
    assert intent_fingerprint("edit_file", "a.py", {"patch": "a\n\nb"}) != canonical
    # leading blank lines are preserved
    assert intent_fingerprint("edit_file", "a.py", {"patch": "\na\nb"}) != canonical


def test_write_file_content_sensitivity():
    a = intent_fingerprint("write_file", "a.txt", {"content": "hello"})
    assert intent_fingerprint("write_file", "a.txt", {"content": b"hello"}) == a
    assert intent_fingerprint("write_file", "a.txt", {"content": "hello\n"}) != a  # raw bytes, no normalisation
    assert intent_fingerprint("write_file", "a.txt", {"content": "héllo"}) == _ref_intent(
        "write_file", "a.txt", {"content_sha256": _ref_h("héllo".encode("utf-8"))}
    )


def test_revert_target_and_revision():
    a = intent_fingerprint("revert", ["b.py", "a.py"], {"revision": "r1"})
    assert intent_fingerprint("revert", ("a.py", "./b.py", "a.py"), {"revision": "r1"}) == a
    assert intent_fingerprint("revert", ["a.py"], {"revision": "r1"}) != a
    assert intent_fingerprint("revert", ["a.py", "b.py"], {"revision": "r2"}) != a
    no_rev = intent_fingerprint("revert", ["a.py", "b.py"])
    assert no_rev != a
    assert intent_fingerprint("revert", ["a.py", "b.py"], {"revision": None}) == no_rev


def test_run_tests_order_insensitive_and_selection_sensitive():
    a = intent_fingerprint("run_tests", "unit", {"tests": ["t::c", "t::a", "t::b"]})
    assert intent_fingerprint("run_tests", "unit", {"tests": ("t::b", "t::c", "t::a", "t::a")}) == a
    assert intent_fingerprint("run_tests", "unit", {"tests": iter(["t::a", "t::b", "t::c"])}) == a
    assert intent_fingerprint("run_tests", "unit", {"tests": ["t::a", "t::b"]}) != a
    assert intent_fingerprint("run_tests", "integration", {"tests": ["t::a", "t::b", "t::c"]}) != a
    whole = intent_fingerprint("run_tests", "unit")
    assert intent_fingerprint("run_tests", "unit", {"tests": None}) == whole
    assert intent_fingerprint("run_tests", "unit", {"tests": []}) != whole  # empty selection != whole suite


def test_run_tests_suite_id_is_not_a_path():
    # Stripped only; not path-normalised.
    assert intent_fingerprint("run_tests", "./unit") != intent_fingerprint("run_tests", "unit")
    assert intent_fingerprint("run_tests", "  unit\n") == intent_fingerprint("run_tests", "unit")


def test_build_lint_poll_targets():
    assert intent_fingerprint("build", "app") != intent_fingerprint("build", "lib")
    assert intent_fingerprint("build", "app") != intent_fingerprint("lint", "app")
    assert intent_fingerprint("poll_status", "job-1") != intent_fingerprint("poll_status", "job-2")


def test_control_action_type_sensitivity():
    assert intent_fingerprint("plan", None, {"action_type": "a"}) != intent_fingerprint("plan", None, {"action_type": "b"})
    assert intent_fingerprint("plan", None, {"action_type": "a"}) != intent_fingerprint("ask", None, {"action_type": "a"})


# ---------------------------------------------------------------------------
# http_call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["POST", "PATCH"])
def test_http_non_idempotent_body_included(method):
    t = {"method": method, "host": "h", "path_template": "/p"}
    a = intent_fingerprint("http_call", t, {"body": "alpha"})
    assert intent_fingerprint("http_call", t, {"body": "beta"}) != a
    assert intent_fingerprint("http_call", t, {"body": b"alpha"}) == a
    none = intent_fingerprint("http_call", t)
    assert intent_fingerprint("http_call", t, {"body": None}) == none
    assert none == _ref_intent("http_call", t, {"body_sha256": None})
    assert intent_fingerprint("http_call", t, {"body": ""}) != none


def test_http_json_body_canonicalised():
    t = HTTP_POST_T
    a = intent_fingerprint("http_call", t, {"body": '{"a": 1, "b": {"y": 2, "x": [true, null]}}'})
    b = intent_fingerprint("http_call", t, {"body": '{\n  "b": {"x": [true,null], "y": 2},\n  "a": 1\n}'})
    assert a == b
    assert intent_fingerprint("http_call", t, {"body": '{"a": 2, "b": {"y": 2, "x": [true, null]}}'}) != a
    # JSON scalars are JSON too
    assert intent_fingerprint("http_call", t, {"body": " 5 "}) == intent_fingerprint("http_call", t, {"body": "5"})


def test_http_non_json_body_hashed_raw():
    t = HTTP_POST_T
    raw = "a=1&b=2"
    expected = _ref_intent("http_call", t, {"body_sha256": _ref_h(raw)})
    assert intent_fingerprint("http_call", t, {"body": raw}) == expected
    # whitespace matters for non-JSON bodies
    assert intent_fingerprint("http_call", t, {"body": "a=1&b=2 "}) != expected
    # non-UTF-8 bytes fall back to raw
    expected_bin = _ref_intent("http_call", t, {"body_sha256": _ref_h(b"\xff\x00\xfe")})
    assert intent_fingerprint("http_call", t, {"body": b"\xff\x00\xfe"}) == expected_bin


def test_http_json_body_with_nan_or_infinity_hashed_raw():
    # NaN / Infinity are not RFC 8259 JSON: "anything else as raw bytes".
    t = HTTP_POST_T
    for raw in ('{"x": NaN}', "Infinity", '[-Infinity, 1]', '{"a": 1.5, "b": NaN}'):
        expected = _ref_intent("http_call", t, {"body_sha256": _ref_h(raw)})
        assert intent_fingerprint("http_call", t, {"body": raw}) == expected, raw


@pytest.mark.parametrize(
    "raw,canonical",
    [
        ('{"price": 1.5}', b'{"price":1.5}'),
        ("[1e3]", b"[1e3]"),
        (' { "b" : [ -0.25 , 2 ] , "a" : 1.5E+2 } ', b'{"a":1.5E+2,"b":[-0.25,2]}'),
        ('{"t": 0.7, "m": "gpt", "n": {"z": 1.0, "y": null}}', b'{"m":"gpt","n":{"y":null,"z":1.0},"t":0.7}'),
    ],
)
def test_http_json_body_with_floats_canonicalised_with_verbatim_literals(raw, canonical):
    # Float literals keep their exact source text (no float repr is involved),
    # but key order and insignificant whitespace are canonicalised.
    t = HTTP_POST_T
    expected = _ref_intent("http_call", t, {"body_sha256": _ref_h(canonical)})
    assert intent_fingerprint("http_call", t, {"body": raw}) == expected


def test_http_json_body_with_floats_key_order_and_whitespace_insensitive():
    t = HTTP_POST_T
    a = intent_fingerprint("http_call", t, {"body": '{"a":1.5,"b":2}'})
    assert intent_fingerprint("http_call", t, {"body": '{"b": 2, "a": 1.5}'}) == a
    assert intent_fingerprint("http_call", t, {"body": b'{\n  "a" : 1.5,\n  "b" : 2\n}'}) == a
    patch = {"method": "PATCH", "host": "h", "path_template": "/p"}
    assert intent_fingerprint("http_call", patch, {"body": '{"temperature":0.2,"model":"m"}'}) == intent_fingerprint(
        "http_call", patch, {"body": '{"model": "m", "temperature": 0.2}'}
    )
    # values still matter
    assert intent_fingerprint("http_call", t, {"body": '{"a":2.5,"b":2}'}) != a


def test_http_json_float_literal_text_is_kept_verbatim():
    t = HTTP_POST_T
    one_five = intent_fingerprint("http_call", t, {"body": "[1.5]"})
    assert intent_fingerprint("http_call", t, {"body": "[1.50]"}) != one_five
    assert intent_fingerprint("http_call", t, {"body": "[15e-1]"}) != one_five
    # an int literal and a float literal are different bodies
    assert intent_fingerprint("http_call", t, {"body": "[1]"}) != intent_fingerprint("http_call", t, {"body": "[1.0]"})


def test_http_json_float_literal_does_not_collide_with_tagged_objects():
    t = HTTP_POST_T
    plain = intent_fingerprint("http_call", t, {"body": "1.5"})
    for lookalike in ('{"$num":"1.5"}', '"1.5"', '["1.5"]'):
        assert intent_fingerprint("http_call", t, {"body": lookalike}) != plain, lookalike


def test_http_target_normalisation():
    a = intent_fingerprint("http_call", {"method": "get", "host": "API.X.COM", "path_template": "/A/{id}"})
    b = intent_fingerprint("http_call", {"method": "GET", "host": "api.x.com", "path_template": "/A/{id}"})
    assert a == b
    # path_template is kept as given (case-sensitive)
    c = intent_fingerprint("http_call", {"method": "GET", "host": "api.x.com", "path_template": "/a/{id}"})
    assert c != a
    assert intent_fingerprint("http_call", {"method": "HEAD", "host": "api.x.com", "path_template": "/A/{id}"}) != a


@pytest.mark.parametrize(
    "target",
    [
        {"method": "TRACE", "host": "h", "path_template": "/"},
        {"method": "CONNECT", "host": "h", "path_template": "/"},
        {"method": "GET ", "host": "h", "path_template": "/"},
        {"method": "GET", "host": "h"},
        {"method": "GET", "host": "h", "path_template": "/", "query": "x"},
        {"method": 1, "host": "h", "path_template": "/"},
        "GET h /",
        ["GET", "h", "/"],
    ],
)
def test_http_bad_target_rejected(target):
    with pytest.raises(FingerprintError):
        intent_fingerprint("http_call", target)


def test_http_bad_body_type_rejected():
    with pytest.raises(FingerprintError):
        intent_fingerprint("http_call", HTTP_POST_T, {"body": {"a": 1}})
    with pytest.raises(FingerprintError):
        intent_fingerprint("http_call", HTTP_GET_T, {"body": {"a": 1}})


# ---------------------------------------------------------------------------
# Rejections: unknown keys, unknown family, targets, arg types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_unknown_arg_key_rejected(family):
    target, args = MINIMAL[family]
    with pytest.raises(FingerprintError):
        intent_fingerprint(family, target, {**args, "definitely_unknown": 1})


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_non_str_arg_key_rejected(family):
    target, args = MINIMAL[family]
    with pytest.raises(FingerprintError):
        intent_fingerprint(family, target, {**args, 1: "x"})


def test_args_must_be_mapping():
    with pytest.raises(FingerprintError):
        intent_fingerprint("list_dir", ".", [("timeout", 1)])  # type: ignore[arg-type]


@pytest.mark.parametrize("family", ["READ_FILE", "readfile", "shell", "", "Read_File", None, 3])
def test_unknown_family_rejected(family):
    with pytest.raises(FingerprintError):
        intent_fingerprint(family, "a.py")  # type: ignore[arg-type]


@pytest.mark.parametrize("family", [f for f in ALL_FAMILIES if f not in ("plan", "ask", "yield")])
def test_target_required(family):
    _, args = MINIMAL[family]
    with pytest.raises(FingerprintError):
        intent_fingerprint(family, None, args)


@pytest.mark.parametrize("family", ["plan", "ask", "yield"])
def test_control_families_reject_target(family):
    with pytest.raises(FingerprintError):
        intent_fingerprint(family, "x", {"action_type": "a"})
    with pytest.raises(FingerprintError):
        intent_fingerprint(family, "", {"action_type": "a"})


@pytest.mark.parametrize("family", ["plan", "ask", "yield"])
def test_control_families_require_str_action_type(family):
    with pytest.raises(FingerprintError):
        intent_fingerprint(family)
    with pytest.raises(FingerprintError):
        intent_fingerprint(family, None, {"action_type": None})
    with pytest.raises(FingerprintError):
        intent_fingerprint(family, None, {"action_type": 5})


@pytest.mark.parametrize(
    "family,target",
    [
        ("read_file", 5),
        ("read_file", ["a.py"]),
        ("list_dir", b"src"),
        ("run_tests", "   "),
        ("run_tests", 7),
        ("build", ""),
        ("lint", "  "),
        ("poll_status", ""),
        ("revert", "a.py"),  # bare str, not a list
        ("revert", []),
        ("revert", ["a.py", 3]),
        ("revert", ["../x"]),
        ("edit_file", "/etc/passwd"),
    ],
)
def test_bad_targets_rejected(family, target):
    _, args = MINIMAL[family]
    with pytest.raises(FingerprintError):
        intent_fingerprint(family, target, args)


@pytest.mark.parametrize(
    "family,args",
    [
        ("read_file", {"line_start": "1"}),
        ("read_file", {"line_start": True}),
        ("read_file", {"line_end": 1.0}),
        ("search", {}),
        ("search", {"query": None}),
        ("search", {"query": 3}),
        ("edit_file", {}),
        ("edit_file", {"patch": None}),
        ("edit_file", {"patch": b"+x"}),
        ("write_file", {}),
        ("write_file", {"content": None}),
        ("write_file", {"content": 3}),
        ("revert", {"revision": 3}),
        ("run_tests", {"tests": "t::a"}),  # bare str would iterate chars
        ("run_tests", {"tests": ["t::a", 3]}),
        ("run_tests", {"tests": 5}),
        ("run_tests", {"tests": {"t::a": 1}}),
    ],
)
def test_bad_included_args_rejected(family, args):
    target, _ = MINIMAL[family]
    with pytest.raises(FingerprintError):
        intent_fingerprint(family, target, args)


# ---------------------------------------------------------------------------
# normalise_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("a/b.py", "a/b.py"),
        ("./a/b.py", "a/b.py"),
        ("a//b.py", "a/b.py"),
        ("a/./b.py", "a/b.py"),
        ("a//./b/", "a/b"),
        ("a\\b\\c.py", "a/b/c.py"),
        ("a/../b", "b"),
        ("a/b/../../c", "c"),
        ("./././x", "x"),
        (".", "."),
        ("./", "."),
        ("a/..", "."),
        ("ab:c/d", "ab:c/d"),
        ("ünï/çödé.py", "ünï/çödé.py"),
    ],
)
def test_normalise_path(raw, expected):
    assert normalise_path(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "/etc/passwd",
        "/",
        "\\abs\\path",
        "\\\\server\\share\\x",
        "//server/share",
        "C:\\Users\\x",
        "C:/Users/x",
        "C:foo",
        "c:",
        "..",
        "../x",
        "a/../../b",
        "./../x",
        "a\\..\\..\\b",
        "",
        "a\x00b",
    ],
)
def test_normalise_path_rejects(raw):
    with pytest.raises(FingerprintError):
        normalise_path(raw)


@pytest.mark.parametrize("raw", [None, 3, b"a", ["a"]])
def test_normalise_path_rejects_non_str(raw):
    with pytest.raises(FingerprintError):
        normalise_path(raw)  # type: ignore[arg-type]


def test_equivalent_paths_give_equal_intents():
    base = intent_fingerprint("read_file", "src/pkg/mod.py")
    for variant in ("./src/pkg/mod.py", "src\\pkg\\mod.py", "src//pkg/./mod.py", "src/x/../pkg/mod.py"):
        assert intent_fingerprint("read_file", variant) == base


# ---------------------------------------------------------------------------
# failure_fingerprint / normalise_location / normalise_message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ("src/a.py", "src/a.py"),
        ("src/a.py:42", "src/a.py"),
        ("src/a.py:42:7", "src/a.py"),
        ("tests/test_a.py::test_x", "tests/test_a.py::test_x"),
        ("tests/test_a.py::TestK::test_x[1-2]", "tests/test_a.py::TestK::test_x[1-2]"),
        ("tests/test_a.py::test_x:42", "tests/test_a.py::test_x"),
        ("libfoo.so+0x1a2b", "libfoo.so"),
        ("frame at 0xDEADbeef", "frame at"),
        ("/build/out/bin/app@0x7ffee000", "app"),
        ("a.py:0x1f", "a.py"),
        ("a.py:12+0x1f", "a.py"),
        # hex-like text inside identifiers is not an address
        ("tests/test_0x1.py::t", "tests/test_0x1.py::t"),
        ("src/x0xff.py", "src/x0xff.py"),
        ("tests/test_a.py::test_p[0x1f]", "tests/test_a.py::test_p[0x1f]"),
        ("tests/test_a.py::test_p[0x1f]:7", "tests/test_a.py::test_p[0x1f]"),
        # absolute: keep from the leftmost /src/, /tests/ or /lib/ anchor (inclusive)
        ("/home/u/proj/src/pkg/mod.py:12", "src/pkg/mod.py"),
        ("/home/u/proj/tests/test_a.py::test_x", "tests/test_a.py::test_x"),
        ("/home/u/proj/tests/unit/test_a.py::test_x:9", "tests/unit/test_a.py::test_x"),
        ("/usr/lib/python3.12/json/decoder.py:355", "lib/python3.12/json/decoder.py"),
        ("/opt/app/src/vendor/lib/z.py", "src/vendor/lib/z.py"),  # leftmost anchor wins
        ("/opt/app/src/z.py", "src/z.py"),
        ("/r/src//pkg/./x/../m.py:3", "src/pkg/m.py"),
        ("/opt/app/src", "src"),  # anchor must be a directory -> basename
        ("/opt/app/main.py:3:1", "main.py"),  # no anchor -> basename
        ("C:\\work\\proj\\src\\pkg\\m.py:10", "src/pkg/m.py"),
        ("C:\\work\\proj\\m.py", "m.py"),
        ("\\\\server\\share\\tests\\t.py", "tests/t.py"),
        # relative: lexically normalised like normalise_path, but never raises
        ("  src/a.py:42  ", "src/a.py"),
        ("./src/a.py:3", "src/a.py"),
        ("src//a.py:3", "src/a.py"),
        ("src/./pkg/../a.py", "src/a.py"),
        ("src\\a.py:3", "src/a.py"),
        ("./tests//test_a.py::test_x:4", "tests/test_a.py::test_x"),
        ("../outside/a.py:1", "../outside/a.py"),
        ("", ""),
    ],
)
def test_normalise_location(raw, expected):
    assert normalise_location(raw) == expected


def test_failure_fingerprint_line_and_address_insensitive():
    base = failure_fingerprint("run_tests", "AssertionError", "tests/test_a.py::test_x")
    assert failure_fingerprint("run_tests", "AssertionError", "tests/test_a.py::test_x:10") == base
    assert failure_fingerprint("run_tests", "AssertionError", "tests/test_a.py::test_x:99:3") == base
    assert failure_fingerprint("run_tests", "  AssertionError\n", "tests/test_a.py::test_x") == base

    seg1 = failure_fingerprint("build", "Segfault", "/build/out/bin/app@0x7ffee000")
    seg2 = failure_fingerprint("build", "Segfault", "/build/out/bin/app@0x0040abcd")
    assert seg1 == seg2

    abs1 = failure_fingerprint("read_file", "FileNotFoundError", "/home/alice/repo/src/m.py:3")
    abs2 = failure_fingerprint("read_file", "FileNotFoundError", "/Users/bob/work/repo/src/m.py:77")
    assert abs1 == abs2


@pytest.mark.parametrize(
    "absolute",
    [
        "/home/u/repo/src/a.py:10:5",
        "/Users/bob/work/repo/src/a.py:99",
        "C:\\work\\repo\\src\\a.py:3",
        "/home/u/repo/src/../src/a.py",
    ],
)
@pytest.mark.parametrize("relative", ["src/a.py", "src/a.py:10", "./src/a.py:3", "src//a.py:3", "src\\a.py"])
def test_failure_fingerprint_absolute_and_relative_forms_agree(absolute, relative):
    assert failure_fingerprint("test", "E", absolute) == failure_fingerprint("test", "E", relative)


def test_failure_fingerprint_absolute_and_relative_node_ids_agree():
    rel = failure_fingerprint("run_tests", "AssertionError", "tests/unit/test_a.py::test_x")
    assert failure_fingerprint("run_tests", "AssertionError", "/ci/build/tests/unit/test_a.py::test_x:9") == rel


def test_failure_fingerprint_distinguishes_files_sharing_a_basename():
    locs = ["/r/src/util.py", "/r/tests/util.py", "/r/lib/util.py", "/opt/app/src/vendor/lib/z.py", "/opt/app/src/z.py"]
    fps = {failure_fingerprint("test", "E", loc) for loc in locs}
    assert len(fps) == len(locs)


def test_failure_fingerprint_hex_like_identifiers_do_not_collide():
    a = failure_fingerprint("run_tests", "AssertionError", "tests/test_0x1.py::t")
    b = failure_fingerprint("run_tests", "AssertionError", "tests/test_0x2.py::t")
    assert a != b
    p1 = failure_fingerprint("run_tests", "AssertionError", "tests/test_a.py::test_p[0x1f]")
    p2 = failure_fingerprint("run_tests", "AssertionError", "tests/test_a.py::test_p[0x2e]")
    assert p1 != p2
    assert failure_fingerprint("build", "E", "a.py:0x1f") == failure_fingerprint("build", "E", "a.py")


def test_failure_fingerprint_sensitivity():
    base = failure_fingerprint("run_tests", "AssertionError", "tests/test_a.py::test_x")
    assert failure_fingerprint("run_tests", "TypeError", "tests/test_a.py::test_x") != base
    assert failure_fingerprint("run_tests", "AssertionError", "tests/test_a.py::test_y") != base
    assert failure_fingerprint("build", "AssertionError", "tests/test_a.py::test_x") != base
    assert failure_fingerprint("run_tests", "AssertionError") != base
    assert failure_fingerprint("run_tests", "AssertionError", None) == _ref_h(
        _ref_cj({"family": "run_tests", "error_class": "AssertionError", "location": None})
    )


def test_failure_fingerprint_rejects_bad_types():
    with pytest.raises(FingerprintError):
        failure_fingerprint("run_tests", None)  # type: ignore[arg-type]
    with pytest.raises(FingerprintError):
        failure_fingerprint("run_tests", "E", 42)  # type: ignore[arg-type]
    with pytest.raises(FingerprintError):
        failure_fingerprint(None, "E")  # type: ignore[arg-type]


def test_normalise_message():
    a = normalise_message("Timeout after 30.5s at 0x7ffe12 in /home/u/x.py (2026-11-04T10:22:33Z), retry 3")
    b = normalise_message("Timeout after 12s at 0xabc in /tmp/other/y.py (2025-01-01T00:00:00+01:00), retry 9")
    assert a == b
    for leaked in ("30", "0x", "/home", "2026", "x.py"):
        assert leaked not in a
    # identifiers keep embedded digits
    assert "py312" in normalise_message("module py312 failed")
    assert normalise_message("C:\\Users\\bob\\f.txt missing") == normalise_message("D:\\x\\g.txt missing")


def test_normalise_message_is_not_part_of_any_hash():
    # failure_fingerprint has no message parameter at all.
    import inspect as _inspect

    assert list(_inspect.signature(failure_fingerprint).parameters) == ["family", "error_class", "location"]


# ---------------------------------------------------------------------------
# workspace_digest
# ---------------------------------------------------------------------------


def test_workspace_digest_order_independent():
    items = [("a.py", "1"), ("src/b.py", b"2"), ("docs/c.md", "3")]
    d1 = workspace_digest(dict(items))
    d2 = workspace_digest(dict(reversed(items)))
    d3 = workspace_digest(dict([items[1], items[0], items[2]]))
    assert d1 == d2 == d3


def test_workspace_digest_paths_normalised_and_content_sensitive():
    base = workspace_digest({"src/a.py": "x"})
    assert workspace_digest({"./src//a.py": "x"}) == base
    assert workspace_digest({"src\\a.py": b"x"}) == base
    assert workspace_digest({"src/a.py": "y"}) != base
    assert workspace_digest({"src/b.py": "x"}) != base
    assert workspace_digest({}) == _ref_h(b"[]")


@pytest.mark.parametrize(
    "files",
    [
        {"a.py": "1", "./a.py": "2"},
        {"src/a.py": "1", "src\\a.py": "1"},
        {"x/../a.py": "1", "a.py": "1"},
    ],
)
def test_workspace_digest_duplicate_paths_rejected(files):
    with pytest.raises(FingerprintError):
        workspace_digest(files)


def test_workspace_digest_rejects_bad_input():
    with pytest.raises(FingerprintError):
        workspace_digest({"/abs/a.py": "1"})
    with pytest.raises(FingerprintError):
        workspace_digest({"../a.py": "1"})
    with pytest.raises(FingerprintError):
        workspace_digest({"a.py": 1})  # type: ignore[dict-item]
    with pytest.raises(FingerprintError):
        workspace_digest([("a.py", "1")])  # type: ignore[arg-type]


def test_workspace_digest_is_valid_schema_hex():
    from tests.factories import make_event

    d = workspace_digest({"a.py": "1"})
    ev = make_event(0, before=None, after=None, workspace_state_before=d, workspace_state_after=d)
    assert ev.workspace_state_before == d


# ---------------------------------------------------------------------------
# canonical_json / sha256_hex
# ---------------------------------------------------------------------------


def test_canonical_json_format():
    assert canonical_json({"b": 1, "a": [1, (2, 3)], "c": "é"}) == '{"a":[1,[2,3]],"b":1,"c":"é"}'.encode("utf-8")
    assert canonical_json({"z": {"y": None, "x": True}}) == b'{"z":{"x":true,"y":null}}'
    assert canonical_json((1, 2)) == canonical_json([1, 2])


@pytest.mark.parametrize(
    "obj",
    [1.0, 0.5, float("nan"), float("inf"), {"a": 1.5}, [1, [2.0]], (0.0,), {"a": {"b": -0.0}}],
)
def test_canonical_json_rejects_floats(obj):
    with pytest.raises((FingerprintError, TypeError)):
        canonical_json(obj)


@pytest.mark.parametrize("obj", [{1: "a"}, {("a",): 1}, {1, 2}, b"bytes", object(), {"a": {1, 2}}])
def test_canonical_json_rejects_non_json(obj):
    with pytest.raises((FingerprintError, TypeError)):
        canonical_json(obj)


def test_canonical_json_int_key_does_not_collide_with_str_key():
    with pytest.raises(FingerprintError):
        canonical_json({1: "x"})
    assert canonical_json({"1": "x"}) == b'{"1":"x"}'


def test_canonical_json_raises_fingerprint_error_specifically():
    with pytest.raises(FingerprintError):
        canonical_json({"a": 1.5})
    assert issubclass(FingerprintError, ValueError)


@pytest.mark.parametrize(
    "call",
    [
        lambda: canonical_json({"a": "\udcff"}),
        lambda: canonical_json({"\ud800": 1}),
        lambda: sha256_hex("\udcff"),
        lambda: intent_fingerprint("read_file", "src/\udcff.py"),
        lambda: intent_fingerprint("search", ".", {"query": "\ud800"}),
        lambda: intent_fingerprint("run_tests", "unit", {"tests": ["\udcff"]}),
        lambda: intent_fingerprint("write_file", "a", {"content": "\udcff"}),
        lambda: intent_fingerprint("edit_file", "a", {"patch": "+\udcff"}),
        lambda: intent_fingerprint("http_call", HTTP_POST_T, {"body": "\udcff"}),
        lambda: intent_fingerprint("http_call", {"method": "GET", "host": "h\udcff", "path_template": "/"}),
        lambda: workspace_digest({"a\udcff": b"x"}),
        lambda: workspace_digest({"a": "\udcff"}),
        lambda: failure_fingerprint("run_tests", "E", "src/\udcff.py"),
        lambda: failure_fingerprint("run_tests", "E\udcff"),
    ],
)
def test_lone_surrogates_rejected_as_fingerprint_error(call):
    with pytest.raises(FingerprintError):
        call()


def test_http_json_body_with_escaped_surrogate_hashed_raw():
    # Valid JSON (an escaped lone surrogate) with no UTF-8 canonical form:
    # falls back to the raw bytes rather than crashing.
    raw = '"\\udcff"'
    assert len(raw) == 8
    expected = _ref_intent("http_call", HTTP_POST_T, {"body_sha256": _ref_h(raw)})
    assert intent_fingerprint("http_call", HTTP_POST_T, {"body": raw}) == expected
    nested = '{"b": 1.5, "a": ["\\ud800"]}'
    expected_nested = _ref_intent("http_call", HTTP_POST_T, {"body_sha256": _ref_h(nested)})
    assert intent_fingerprint("http_call", HTTP_POST_T, {"body": nested}) == expected_nested


def test_sha256_hex():
    assert sha256_hex("") == hashlib.sha256(b"").hexdigest()
    assert sha256_hex("é") == hashlib.sha256("é".encode("utf-8")).hexdigest()
    assert sha256_hex(b"abc") == sha256_hex("abc")
    with pytest.raises(FingerprintError):
        sha256_hex(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

_DETERMINISM_SNIPPET = r"""
from agent_loop_lab.fingerprint import intent_fingerprint, failure_fingerprint, workspace_digest
out = [
    intent_fingerprint("revert", ["z.py", "a.py", "m/n.py", "b.py"], {"revision": "r"}),
    intent_fingerprint("run_tests", "unit", {"tests": {"t::z", "t::a", "t::m", "t::b"}}),
    intent_fingerprint("http_call", {"method": "POST", "host": "h", "path_template": "/"},
                       {"body": '{"z": 1, "a": {"q": [3, 2], "b": null}}'}),
    failure_fingerprint("run_tests", "E", "/x/src/a.py:3"),
    workspace_digest({"z": "1", "a": "2", "m/q": b"3", "b": "4"}),
]
print("\n".join(out))
"""


def test_determinism_in_process():
    calls = [GOLDEN_CALLS[k] for k in sorted(GOLDEN_CALLS)]
    first = [intent_fingerprint(*c) for c in calls]
    second = [intent_fingerprint(*c) for c in calls]
    assert first == second


def test_determinism_across_processes_and_hash_seeds():
    outputs = []
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        res = subprocess.run(
            [sys.executable, "-c", _DETERMINISM_SNIPPET], capture_output=True, text=True, env=env, check=True
        )
        outputs.append(res.stdout)
    assert outputs[0] == outputs[1] == outputs[2]
    assert len(outputs[0].split()) == 5


def test_module_exports_contract_names():
    for name in (
        "FingerprintError",
        "canonical_json",
        "sha256_hex",
        "normalise_path",
        "intent_fingerprint",
        "failure_fingerprint",
        "normalise_location",
        "normalise_message",
        "workspace_digest",
        "FAMILY_CATEGORY",
    ):
        assert hasattr(fp, name)
