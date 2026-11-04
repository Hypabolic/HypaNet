"""Fingerprint canonicalisation (spec v0.2 §3.2, contracts §A).

Frozen with the encoder: these rules silently determine ``repeated_intent``,
``repeated_failure`` and ``state_revisited``, so any change creates a new
registered spec version.

* ``intent_fingerprint``  = sha256(canonical_json({family, target, args}))
* ``failure_fingerprint`` = sha256(canonical_json({family, error_class, location}))
* ``workspace_digest``    = sha256(canonical_json(sorted([path, sha256(content)])))

Every rejection surfaces as :class:`FingerprintError` (including strings that
cannot be encoded as UTF-8, e.g. lone surrogates).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

from agent_loop_lab.schema import OperationCategory

__all__ = [
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
    "GLOBAL_EXCLUDED_ARGS",
    "IDEMPOTENT_METHODS",
    "NON_IDEMPOTENT_METHODS",
]


class FingerprintError(ValueError):
    """Raised when an input cannot be canonicalised under the frozen §3.2 rules."""


# ---------------------------------------------------------------------------
# Family table (spec §3.2)
# ---------------------------------------------------------------------------

_I = OperationCategory.INSPECT
_M = OperationCategory.MUTATE
_V = OperationCategory.VALIDATE
_E = OperationCategory.EXTERNAL
_C = OperationCategory.CONTROL

FAMILY_CATEGORY: Mapping[str, OperationCategory] = MappingProxyType(
    {
        "read_file": _I,
        "search": _I,
        "list_dir": _I,
        "edit_file": _M,
        "write_file": _M,
        "revert": _M,
        "run_tests": _V,
        "build": _V,
        "lint": _V,
        "http_call": _E,
        "poll_status": _E,
        "plan": _C,
        "ask": _C,
        "yield": _C,
    }
)

GLOBAL_EXCLUDED_ARGS: frozenset[str] = frozenset(
    {
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
    }
)

IDEMPOTENT_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})
NON_IDEMPOTENT_METHODS: frozenset[str] = frozenset({"POST", "PATCH"})

# family -> (included arg keys, family-specific excluded arg keys)
_FAMILY_ARGS: Mapping[str, tuple[frozenset[str], frozenset[str]]] = MappingProxyType(
    {
        "read_file": (frozenset({"line_start", "line_end"}), frozenset()),
        "search": (frozenset({"query"}), frozenset({"limit"})),
        "list_dir": (frozenset(), frozenset()),
        "edit_file": (frozenset({"patch"}), frozenset()),
        "write_file": (frozenset({"content"}), frozenset()),
        "revert": (frozenset({"revision"}), frozenset()),
        "run_tests": (frozenset({"tests"}), frozenset({"verbosity", "parallelism"})),
        "build": (frozenset(), frozenset({"flags"})),
        "lint": (frozenset(), frozenset({"flags"})),
        # ``body`` is included for every method at the key-check level; whether it
        # reaches the payload depends on the method (see _http_args).
        "http_call": (frozenset({"body"}), frozenset()),
        "poll_status": (frozenset(), frozenset({"backoff"})),
        "plan": (frozenset({"action_type"}), frozenset()),
        "ask": (frozenset({"action_type"}), frozenset()),
        "yield": (frozenset({"action_type"}), frozenset()),
    }
)
assert set(_FAMILY_ARGS) == set(FAMILY_CATEGORY)

_CONTROL_FAMILIES = frozenset({"plan", "ask", "yield"})


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _canon(obj: Any, where: str = "$") -> Any:
    """Recursive pre-pass: tuples -> lists, Mappings -> dicts; reject floats, sets,
    non-str keys and anything else that is not plain JSON."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        raise FingerprintError(f"float at {where}: fingerprints must not depend on float repr")
    if isinstance(obj, (list, tuple)):
        return [_canon(v, f"{where}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise FingerprintError(f"non-str key {k!r} at {where}")
            out[k] = _canon(v, f"{where}.{k}")
        return out
    raise FingerprintError(f"unsupported type {type(obj).__name__} at {where}")


def _utf8(text: str, what: str) -> bytes:
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise FingerprintError(f"{what} is not encodable as UTF-8 (lone surrogate?): {exc}") from None


def canonical_json(obj: Any) -> bytes:
    """Sorted keys, UTF-8, no insignificant whitespace. Floats are rejected."""
    text = json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return _utf8(text, "canonical JSON")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = _utf8(data, "sha256_hex input")
    elif isinstance(data, (bytearray, memoryview)):
        data = bytes(data)
    elif not isinstance(data, bytes):
        raise FingerprintError(f"sha256_hex expects bytes or str, got {type(data).__name__}")
    return hashlib.sha256(data).hexdigest()


_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def normalise_path(path: str) -> str:
    """Repo-relative POSIX path, resolved lexically. The repo root is ``"."``."""
    if not isinstance(path, str):
        raise FingerprintError(f"path must be str, got {type(path).__name__}")
    if path == "":
        raise FingerprintError("empty path")
    if "\x00" in path:
        raise FingerprintError("NUL byte in path")
    p = path.replace("\\", "/")
    if p.startswith("/"):
        raise FingerprintError(f"absolute path not allowed: {path!r}")
    if _DRIVE_RE.match(p):
        raise FingerprintError(f"drive-letter path not allowed: {path!r}")
    parts: list[str] = []
    for seg in p.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if not parts:
                raise FingerprintError(f"path escapes the repository root: {path!r}")
            parts.pop()
            continue
        parts.append(seg)
    return "/".join(parts) if parts else "."


# ---------------------------------------------------------------------------
# Intent fingerprint
# ---------------------------------------------------------------------------


def _require_str(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise FingerprintError(f"{what} must be str, got {type(value).__name__}")
    return value


def _stripped_id(value: Any, what: str) -> str:
    s = _require_str(value, what).strip()
    if not s:
        raise FingerprintError(f"{what} must be non-empty")
    return s


def _opt_line(value: Any, what: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise FingerprintError(f"{what} must be int or None, got {type(value).__name__}")
    return int(value)


def _str_list(value: Any, what: str) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable) or isinstance(value, Mapping):
        raise FingerprintError(f"{what} must be an iterable of str (not a bare string)")
    items = list(value)
    for item in items:
        _require_str(item, f"{what} element")
    return items


def _normalise_patch(patch: str) -> str:
    text = patch.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


class _NonCanonicalJSON(ValueError):
    pass


class _JsonNumberLiteral:
    """A JSON float literal kept as its exact source text (never converted to float)."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


def _reject_json_constant(_literal: str) -> Any:
    raise _NonCanonicalJSON(_literal)


def _body_canonical_text(obj: Any) -> str:
    """Canonical JSON text for a parsed body.

    Byte-identical to :func:`canonical_json` on float-free values; float
    literals are emitted verbatim from the source text.
    """
    if isinstance(obj, _JsonNumberLiteral):
        return obj.text
    if obj is None or isinstance(obj, (bool, int, str)):
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, list):
        return "[" + ",".join(_body_canonical_text(v) for v in obj) + "]"
    if isinstance(obj, dict):
        return (
            "{"
            + ",".join(
                json.dumps(k, ensure_ascii=False) + ":" + _body_canonical_text(obj[k]) for k in sorted(obj)
            )
            + "}"
        )
    raise AssertionError(type(obj))  # pragma: no cover - json.loads yields only the above


def _body_sha256(body: Any) -> str | None:
    """sha256 of the normalised body (contracts §A, http_call row).

    Bodies that parse as strict (RFC 8259) JSON are hashed in canonical JSON
    form: sorted keys, no insignificant whitespace, UTF-8. Float literals keep
    their exact source text (``1.5`` and ``1.50`` differ), so no float repr is
    involved. Anything else is hashed as raw bytes: non-UTF-8 bytes, non-JSON
    text, the non-standard ``NaN`` / ``Infinity`` / ``-Infinity`` constants,
    and JSON whose canonical form is not UTF-8-encodable (escaped lone
    surrogates).
    """
    if body is None:
        return None
    if isinstance(body, str):
        raw = _utf8(body, "http body")
    elif isinstance(body, (bytes, bytearray)):
        raw = bytes(body)
    else:
        raise FingerprintError(f"http body must be str, bytes or None, got {type(body).__name__}")
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            parse_float=_JsonNumberLiteral,
            parse_constant=_reject_json_constant,
        )
        canonical = _body_canonical_text(parsed).encode("utf-8")
    except (ValueError, UnicodeError, RecursionError):
        return sha256_hex(raw)
    return sha256_hex(canonical)


def _http_target(target: Any) -> dict[str, str]:
    if not isinstance(target, Mapping):
        raise FingerprintError("http_call target must be a mapping {method, host, path_template}")
    keys = set(target)
    expected = {"method", "host", "path_template"}
    if keys != expected:
        raise FingerprintError(
            f"http_call target keys must be exactly {sorted(expected)}; got {sorted(map(str, keys))}"
        )
    method = _require_str(target["method"], "http_call method").upper()
    if method not in IDEMPOTENT_METHODS and method not in NON_IDEMPOTENT_METHODS:
        raise FingerprintError(f"unknown http method {target['method']!r}")
    host = _require_str(target["host"], "http_call host").lower()
    path_template = _require_str(target["path_template"], "http_call path_template")
    return {"method": method, "host": host, "path_template": path_template}


def _normalise_target(family: str, target: Any) -> Any:
    if family in _CONTROL_FAMILIES:
        if target is not None:
            raise FingerprintError(f"{family} takes no target")
        return None
    if target is None:
        raise FingerprintError(f"{family} requires a target")
    if family in ("read_file", "search", "list_dir", "edit_file", "write_file"):
        return normalise_path(_require_str(target, f"{family} target"))
    if family == "revert":
        if not isinstance(target, (list, tuple)):
            raise FingerprintError("revert target must be a list of paths")
        paths = sorted({normalise_path(_require_str(p, "revert path")) for p in target})
        if not paths:
            raise FingerprintError("revert target must name at least one path")
        return paths
    if family in ("run_tests", "build", "lint", "poll_status"):
        return _stripped_id(target, f"{family} target")
    if family == "http_call":
        return _http_target(target)
    raise AssertionError(family)  # pragma: no cover - table is exhaustive


def _included_args(family: str, target: Any, args: Mapping[str, Any]) -> dict[str, Any]:
    if family == "read_file":
        return {
            "line_range": [
                _opt_line(args.get("line_start"), "line_start"),
                _opt_line(args.get("line_end"), "line_end"),
            ]
        }
    if family == "search":
        query = args.get("query")
        if query is None:
            raise FingerprintError("search requires args['query']")
        return {"query": " ".join(_require_str(query, "query").lower().split())}
    if family == "edit_file":
        patch = args.get("patch")
        if patch is None:
            raise FingerprintError("edit_file requires args['patch']")
        return {"patch_sha256": sha256_hex(_normalise_patch(_require_str(patch, "patch")))}
    if family == "write_file":
        content = args.get("content")
        if content is None:
            raise FingerprintError("write_file requires args['content']")
        if not isinstance(content, (str, bytes, bytearray)):
            raise FingerprintError("write_file content must be str or bytes")
        return {"content_sha256": sha256_hex(content)}
    if family == "revert":
        revision = args.get("revision")
        return {"revision": None if revision is None else _require_str(revision, "revision")}
    if family == "run_tests":
        tests = args.get("tests")
        return {"tests": None if tests is None else sorted(set(_str_list(tests, "tests")))}
    if family == "http_call":
        if target["method"] in NON_IDEMPOTENT_METHODS:
            return {"body_sha256": _body_sha256(args.get("body"))}
        body = args.get("body")
        if body is not None and not isinstance(body, (str, bytes, bytearray)):
            raise FingerprintError("http body must be str, bytes or None")
        return {}  # idempotent: body accepted and ignored
    if family in _CONTROL_FAMILIES:
        action_type = args.get("action_type")
        if action_type is None:
            raise FingerprintError(f"{family} requires args['action_type']")
        return {"action_type": _require_str(action_type, "action_type").lower()}
    return {}  # list_dir, build, lint, poll_status


def intent_fingerprint(family: str, target: Any = None, args: Mapping[str, Any] | None = None) -> str:
    if not isinstance(family, str) or family not in FAMILY_CATEGORY:
        raise FingerprintError(f"unknown operation family {family!r}")
    if args is None:
        args = {}
    if not isinstance(args, Mapping):
        raise FingerprintError("args must be a mapping or None")
    included, excluded = _FAMILY_ARGS[family]
    for key in args:
        if not isinstance(key, str):
            raise FingerprintError(f"non-str arg key {key!r}")
        if key not in included and key not in excluded and key not in GLOBAL_EXCLUDED_ARGS:
            raise FingerprintError(f"arg {key!r} is neither included nor excluded for family {family!r}")
    norm_target = _normalise_target(family, target)
    payload = {"family": family, "target": norm_target, "args": _included_args(family, norm_target, args)}
    return sha256_hex(canonical_json(payload))


# ---------------------------------------------------------------------------
# Failure fingerprint
# ---------------------------------------------------------------------------

# A standalone hex address: not part of a larger identifier such as
# ``test_0x1.py`` or ``x0xff``.
_HEX_RE = re.compile(r"(?<![A-Za-z0-9_])0x[0-9a-fA-F]+(?![A-Za-z0-9_])")
# In locations, also drop the separator that introduced the address
# (``a.py:0x1f``, ``libfoo.so+0x1a2b``, ``app@0x7ffee000``).
_LOCATION_HEX_RE = re.compile(r"[:@+]?(?<![A-Za-z0-9_])0x[0-9a-fA-F]+(?![A-Za-z0-9_])")
_LINE_COL_RE = re.compile(r"(?<!:)(?::\d+){1,2}$")
_ABS_RE = re.compile(r"^(?:/|[A-Za-z]:/)")
_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")
_ANCHORS = frozenset({"src", "tests", "lib"})


def _lexical_segments(path: str) -> list[str]:
    """Collapse ``""`` / ``"."`` segments and resolve ``..`` lexically.

    Like :func:`normalise_path`, but never raises: a ``..`` that cannot be
    popped is kept, because a location is a free-form identifier.
    """
    parts: list[str] = []
    for seg in path.split("/"):
        if seg in ("", "."):
            continue
        if seg == ".." and parts and parts[-1] != "..":
            parts.pop()
            continue
        parts.append(seg)
    return parts


def normalise_location(location: str | None) -> str | None:
    """Normalised failure location (spec §3.2: normalised repo-relative POSIX paths).

    * strip surrounding whitespace and a trailing ``:<line>`` / ``:<line>:<col>``;
    * split off a pytest node id (``path::name``); the node id is kept intact;
    * remove standalone hex addresses (with their ``:``/``@``/``+`` separator)
      from the path part;
    * lexically normalise the path (``\\`` -> ``/``, collapse ``//`` and ``/./``,
      resolve ``..``), so ``./src/a.py`` and ``src//a.py`` equal ``src/a.py``;
    * for absolute paths, keep everything from the **leftmost** ``src``,
      ``tests`` or ``lib`` directory segment (inclusive), so
      ``/home/u/repo/src/a.py`` equals the repo-relative ``src/a.py``;
      without such an anchor, keep only the basename.
    """
    if location is None:
        return None
    if not isinstance(location, str):
        raise FingerprintError(f"location must be str or None, got {type(location).__name__}")
    s = _LINE_COL_RE.sub("", location.strip())
    path, sep, rest = s.partition("::")
    path = _LOCATION_HEX_RE.sub("", path).strip()
    path = _LINE_COL_RE.sub("", path)
    path = path.replace("\\", "/")
    if _ABS_RE.match(path):
        segs = _lexical_segments(_DRIVE_PREFIX_RE.sub("", path))
        segs = [seg for seg in segs if seg != ".."]  # ".." above the filesystem root
        anchor = next((i for i, seg in enumerate(segs[:-1]) if seg in _ANCHORS), None)
        if anchor is not None:
            path = "/".join(segs[anchor:])
        else:
            path = segs[-1] if segs else ""
    elif path:
        segs = _lexical_segments(path)
        path = "/".join(segs) if segs else "."
    return path + sep + rest


def failure_fingerprint(family: str, error_class: str, location: str | None = None) -> str:
    _require_str(family, "family")
    payload = {
        "family": family,
        "error_class": _require_str(error_class, "error_class").strip(),
        "location": normalise_location(location),
    }
    return sha256_hex(canonical_json(payload))


_ISO_TS_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
)
_ABS_PATH_IN_TEXT_RE = re.compile(r"(?:^|(?<=[\s\"'(=,\[]))(?:/|[A-Za-z]:[\\/])[^\s\"'(),\[\]]*")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?")
_WS_RE = re.compile(r"\s+")


def normalise_message(message: str) -> str:
    """Strip numbers, hex addresses, ISO timestamps and absolute paths.

    Diagnostics only: never part of any hash.
    """
    s = _require_str(message, "message")
    s = _ISO_TS_RE.sub("<TS>", s)
    s = _HEX_RE.sub("<ADDR>", s)
    s = _ABS_PATH_IN_TEXT_RE.sub("<PATH>", s)
    s = _NUMBER_RE.sub("<N>", s)
    return _WS_RE.sub(" ", s).strip()


# ---------------------------------------------------------------------------
# Workspace digest
# ---------------------------------------------------------------------------


def workspace_digest(files: Mapping[str, bytes | str]) -> str:
    if not isinstance(files, Mapping):
        raise FingerprintError("files must be a mapping of path -> content")
    entries: dict[str, str] = {}
    for p, content in files.items():
        norm = normalise_path(p)
        if norm in entries:
            raise FingerprintError(f"duplicate normalised path {norm!r} (from {p!r})")
        if not isinstance(content, (bytes, bytearray, str)):
            raise FingerprintError(f"content for {p!r} must be bytes or str")
        entries[norm] = sha256_hex(content)
    return sha256_hex(canonical_json(sorted([path, digest] for path, digest in entries.items())))
