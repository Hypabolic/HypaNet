"""Latent world for the standard generator (M1_CONTRACTS §2).

A :class:`World` is the simulator's ground truth for one session:

* **Repository.** Files are modelled as a header plus ordered named *blocks*
  (functions in source files, sections/keys/entries in data files). Each block
  has a latent behaviour id (``"orig"``, ``"fix"``, a wrong/neutral/competing
  variant id) and a cosmetic style (a rename/comment counter and a formatted
  flag). Rendering is deterministic and real text, so
  ``fingerprint.workspace_digest(world.texts())`` is the latent workspace id.
  Cosmetic changes alter text only; behaviour lives in the behaviour ids.
* **Faults.** 1-4 per task. Each has a location (path, block), clue facts that
  reveal it, a correct fix (behaviour ``"fix"`` at its block), an error class
  and the tests that depend on it.
* **Tests.** A test passes iff every fault it depends on is fixed (or a
  competing variant happens to satisfy it) and no regressing variant is active
  on a block it covers.
* **External services and poll resources** with latent availability.

Everything here is latent (oracle side). Nothing in this module is visible on
the inference path; only the engine turns world state into canonical events.
Determinism: the only randomness is the ``random.Random`` passed to
:meth:`World.generate`; rendering and variant text are derived with sha256,
never ``hash()``.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..fingerprint import normalise_path, workspace_digest

__all__ = [
    "Language",
    "LANGUAGES",
    "LANGUAGE_NAMES",
    "BlockState",
    "FileModel",
    "Clue",
    "Fault",
    "TestCase",
    "Variant",
    "Service",
    "Resource",
    "Vocab",
    "World",
    "SERVICE_MODES",
    "HOSTS",
    "PATH_TEMPLATES",
    "OBS_ENDPOINTS",
    "POLL_RESOURCES",
    "HTTP_ERRORS",
    "PLAN_ACTIONS",
    "ASK_ACTIONS",
    "YIELD_ACTIONS",
    "DECOY_QUERIES",
]

# ---------------------------------------------------------------------------
# Vocabulary pools
# ---------------------------------------------------------------------------

PACKAGES = (
    "core", "api", "billing", "auth", "ingest", "storage", "search", "notify", "report", "sync",
    "cli", "web", "worker", "metrics", "geo", "media", "catalog", "orders", "identity", "pipeline",
)
MODULES = (
    "parser", "tokenizer", "scheduler", "cache", "router", "session", "invoice", "retry_policy",
    "config_loader", "serializer", "rate_limiter", "migrations", "uploader", "search_index", "notifier",
    "ledger", "csv_import", "pagination", "webhook", "queue_worker", "feature_flags", "permissions",
    "image_resize", "diff_engine", "templates", "locale", "money", "timezone", "checksum", "backoff",
    "cursor", "manifest", "archive", "dedupe", "thumbnailer", "email_render", "audit_log", "quota",
    "shard_map", "graph_walk", "slugify", "currency", "export_job", "token_bucket", "chunker",
)
VERBS = (
    "parse", "normalize", "compute", "apply", "refresh", "validate", "merge", "encode", "decode",
    "resolve", "render", "schedule", "flush", "split", "format", "load", "store", "lookup",
    "reconcile", "compact", "expand", "sanitize", "convert", "estimate", "rank", "dispatch", "batch",
)
NOUNS = (
    "header", "path", "total", "discount", "token", "payload", "ranges", "cursor", "timestamp",
    "locale", "amount", "chunk", "record", "key", "window", "limit", "offset", "slug", "query",
    "entry", "row", "manifest", "digest", "quota", "address", "schema", "retries", "deadline",
)
SCENARIOS = (
    "empty_input", "unicode", "negative_values", "large_batch", "missing_key", "trailing_slash",
    "leap_year", "timezone_offset", "duplicate_entries", "null_field", "max_length", "zero_division",
    "nested_objects", "round_trip", "concurrent_access", "expired_token", "utf8_bom", "sorted_order",
    "boundary", "rounding", "default_config", "retry_after", "partial_page", "case_folding",
)
KEYWORDS = (
    "overflow", "timeout", "encoding", "off_by_one", "null", "precision", "ordering", "stale",
    "boundary", "locale", "escape", "truncation", "race", "cache_key", "dst",
)
DECOY_QUERIES = (
    "TODO", "FIXME", "deprecated", "logger", "import json", "def main", "class Config",
    "feature flag", "__all__", "copyright", "version", "env var", "metrics.increment", "lru_cache",
    "async def", "unsafe", "panic!", "interface{}", "assert_eq", "@Override", "require_relative",
)
DECOY_DOCS = (
    "README.md", "CHANGELOG.md", "CONTRIBUTING.md", "docs/architecture.md", "docs/runbook.md",
    "docs/adr/0003-caching.md", "docs/api.md", ".github/workflows/ci.yml", "Dockerfile", "Makefile",
    "scripts/release.sh", "scripts/bootstrap.sh", ".editorconfig", "docs/onboarding.md",
    "deploy/helm/values.yaml", "deploy/terraform/main.tf",
)
HOSTS = (
    "api.github.com", "registry.npmjs.org", "pypi.org", "api.stripe.com", "hooks.slack.com",
    "s3.us-east-1.amazonaws.com", "auth.internal.acme.dev", "staging-api.acme.io", "api.openai-proxy.acme.dev",
    "maps.googleapis.com", "ci.acme.dev", "artifacts.acme.dev", "vault.acme.internal", "db-proxy.internal",
    "api.sendgrid.com", "storage.googleapis.com", "graphql.acme.io", "payments.sandbox.acme.io",
    "localhost:8080", "search.internal.acme.dev",
)
PATH_TEMPLATES = (
    "/repos/{owner}/{repo}/pulls", "/v1/charges", "/v1/users/{id}", "/api/v2/search", "/health",
    "/v1/jobs/{job_id}", "/oauth/token", "/simple/{package}/", "/graphql", "/v2/objects/{bucket}/{key}",
    "/api/v1/orders/{order_id}", "/v3/mail/send", "/internal/flags", "/v1/secrets/{path}",
    "/api/v1/query", "/v1/embeddings", "/status", "/v1/webhooks/{hook_id}/deliveries",
)
OBS_ENDPOINTS = (
    ("sentry.io", "/api/0/issues/{issue_id}/events/"),
    ("grafana.acme.dev", "/api/datasources/proxy/{id}/query"),
    ("logs.acme.dev", "/v1/search"),
    ("ci.acme.dev", "/api/v4/jobs/{job_id}/trace"),
    ("honeycomb.io", "/1/query_results/{dataset}/{id}"),
    ("api.datadoghq.com", "/api/v2/logs/events/search"),
)
POLL_RESOURCES = (
    "ci/pipeline/{n}", "deploy/staging/{svc}", "k8s/rollout/{svc}", "db/migration/{n}", "job/{hex}",
    "cache/warmup/{svc}", "index/rebuild/{name}", "container/{svc}/health", "build/artifact/{hex}",
    "queue/drain/{name}", "dns/propagation/{svc}", "replica/sync/{n}",
)
SERVICE_NAMES = ("api", "worker", "web", "search", "billing", "gateway", "scheduler", "ledger")
HTTP_ERRORS = (
    "TimeoutError", "ConnectionResetError", "HTTP503", "HTTP502", "HTTP504", "HTTP429", "HTTP500",
    "SSLError", "DNSResolutionError", "ConnectionRefusedError",
)
AUTH_ERRORS = ("HTTP401", "HTTP403", "HTTP404", "HTTP400")
PLAN_ACTIONS = (
    "decompose", "prioritise", "choose_next_step", "reflect", "summarise_findings", "verify_plan",
    "review_diff", "estimate_scope", "list_hypotheses", "checkpoint",
)
ASK_ACTIONS = ("clarify_requirement", "confirm_destructive", "request_credentials", "confirm_scope")
YIELD_ACTIONS = ("done", "handoff", "blocked", "partial_result")
HELPERS = ("clamp", "coerce", "scale", "trim", "shift", "pick", "bound", "fold", "mask", "pad_to")
SERVICE_MODES = ("ok", "transient", "persistent", "needs_fix")


@dataclass(frozen=True)
class Language:
    name: str
    ext: str
    source_path: str  # format keys: pkg, mod, Mod, org
    test_path: str  # format keys: pkg, mod, Mod, tag, org
    test_case: str  # format keys: case, Case, cAse, words, mod
    suite_id: str  # format key: tag
    suite_tags: tuple[str, ...]
    fn_case: str  # "snake" | "camel" | "pascal"
    errors: tuple[str, ...]
    configs: tuple[str, ...]
    lint_targets: tuple[str, ...]
    build_targets: tuple[str, ...]
    lint_errors: tuple[str, ...]
    build_errors: tuple[str, ...]


LANGUAGES: dict[str, Language] = {
    "python": Language(
        "python", ".py", "src/{pkg}/{mod}.py", "tests/{tag}/test_{mod}.py", "test_{case}", "tests/{tag}",
        ("unit", "integration", "regression", "smoke"), "snake",
        ("AssertionError", "KeyError", "TypeError", "ValueError", "IndexError", "AttributeError",
         "ZeroDivisionError", "UnicodeDecodeError", "RecursionError", "OverflowError"),
        ("pyproject.toml", "setup.cfg", "config/settings.yaml", ".env.example", "tox.ini"),
        ("ruff", "mypy", "flake8"), ("wheel", "sdist", "docs"),
        ("F821", "E501", "B006", "mypy:arg-type", "mypy:union-attr"), ("BuildBackendError", "SphinxWarning"),
    ),
    "typescript": Language(
        "typescript", ".ts", "src/{pkg}/{cmod}.ts", "src/{pkg}/__tests__/{cmod}.{tag}.test.ts", "{mod} {words}",
        "{tag}", ("unit", "integration", "e2e", "contract"), "camel",
        ("TypeError", "AssertionError", "RangeError", "ReferenceError", "JestAssertionError", "SyntaxError"),
        ("package.json", "tsconfig.json", ".eslintrc.json", "jest.config.ts", "config/default.json"),
        ("eslint", "tsc", "prettier"), ("bundle", "tsc-build", "storybook"),
        ("no-unused-vars", "TS2345", "TS2532", "no-floating-promises"), ("TS2307", "RollupError"),
    ),
    "go": Language(
        "go", ".go", "internal/{pkg}/{mod}.go", "internal/{pkg}/{mod}_{tag}_test.go", "Test{Case}",
        "./internal/...:{tag}", ("unit", "integration", "race"), "pascal",
        ("RuntimePanic", "AssertionFailed", "NilPointerDereference", "IndexOutOfRange", "DeadlineExceeded",
         "ErrUnexpectedEOF"),
        ("go.mod", "config.yaml", ".golangci.yml", "Makefile.env"),
        ("golangci-lint", "go-vet", "staticcheck"), ("go-build", "goreleaser"),
        ("SA4006", "errcheck", "govet:copylocks", "ineffassign"), ("CompileError", "UndefinedSymbol"),
    ),
    "rust": Language(
        "rust", ".rs", "src/{pkg}/{mod}.rs", "tests/{mod}_{tag}.rs", "{case}", "{tag}",
        ("unit", "integration", "doc", "property"), "snake",
        ("AssertionFailed", "Panic", "ParseIntError", "Utf8Error", "IndexOutOfBounds", "ArithmeticOverflow"),
        ("Cargo.toml", "config/app.toml", "rustfmt.toml", "clippy.toml"),
        ("clippy", "rustfmt-check"), ("cargo-build", "cargo-doc"),
        ("clippy::unwrap_used", "clippy::needless_borrow", "dead_code"), ("E0308", "E0382"),
    ),
    "java": Language(
        "java", ".java", "src/main/java/com/{org}/{pkg}/{Mod}.java", "src/test/java/com/{org}/{pkg}/{Mod}{Tag}Test.java",
        "{cAse}", "{tag}Tests", ("unit", "integration", "contract"), "camel",
        ("NullPointerException", "IllegalArgumentException", "IllegalStateException", "AssertionFailedError",
         "ArrayIndexOutOfBoundsException", "NumberFormatException", "ClassCastException"),
        ("pom.xml", "build.gradle", "src/main/resources/application.yml", "checkstyle.xml"),
        ("checkstyle", "spotbugs", "errorprone"), ("gradle-assemble", "mvn-package"),
        ("MagicNumber", "NP_NULL_ON_SOME_PATH", "UnusedImports"), ("CompilationFailure", "MissingSymbol"),
    ),
    "ruby": Language(
        "ruby", ".rb", "lib/{pkg}/{mod}.rb", "spec/{pkg}/{mod}_{tag}_spec.rb", "{mod} {words}", "spec:{tag}",
        ("unit", "request", "feature"), "snake",
        ("NoMethodError", "ArgumentError", "KeyError", "TypeError", "ExpectationNotMetError", "ZeroDivisionError",
         "NameError"),
        ("Gemfile", "config/database.yml", ".rubocop.yml", "config/application.rb"),
        ("rubocop", "standardrb", "brakeman"), ("rake-build", "bundle-package"),
        ("Style/StringLiterals", "Lint/UselessAssignment", "Security/Open"), ("LoadError", "GemNotFound"),
    ),
}
LANGUAGE_NAMES: tuple[str, ...] = tuple(LANGUAGES)
_EXT_LANG = {spec.ext: name for name, spec in LANGUAGES.items()}
ORGS = ("acme", "globex", "initech", "umbrella", "hooli")


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------


def _hint(*parts: Any) -> int:
    """Deterministic integer from parts (sha256; never Python's salted ``hash``)."""
    data = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def _camel(snake: str) -> str:
    head, *rest = snake.split("_")
    return head + "".join(w.capitalize() for w in rest)


def _pascal(snake: str) -> str:
    return "".join(w.capitalize() for w in snake.split("_"))


def _fn_name(lang: Language, snake: str) -> str:
    if lang.fn_case == "camel":
        return _camel(snake)
    if lang.fn_case == "pascal":
        return _pascal(snake)
    return snake


def _ident(name: str) -> str:
    out = re.sub(r"[^0-9A-Za-z_]", "_", name)
    return out if out and not out[0].isdigit() else f"_{out}"


def _under(path: str, scope: str) -> bool:
    return scope == "." or path == scope or path.startswith(scope.rstrip("/") + "/")


# ---------------------------------------------------------------------------
# Repository model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockState:
    """Latent state of one block: behaviour (semantics) plus cosmetic style (text only)."""

    behaviour: str = "orig"
    style: int = 0
    formatted: bool = False


@dataclass
class FileModel:
    path: str
    lang: str | None  # language name for code files, else None (data file)
    blocks: tuple[str, ...]
    states: dict[str, BlockState]
    role: str = "source"  # source | test | config | log | doc | fixture | note

    def snapshot(self) -> dict[str, BlockState]:
        return dict(self.states)


@dataclass(frozen=True)
class Clue:
    """How a fact can be revealed.

    ``kind``: ``read`` (path + block), ``search`` (query substring + scope),
    ``lint`` / ``build`` (target), ``http`` (host + path template), or
    ``test`` (a failing test whose first failure is the fault; implicit).
    """

    fact: str
    fault: str
    kind: str
    path: str | None = None
    block: str | None = None
    query: str | None = None
    scope: str | None = None
    target: str | None = None
    host: str | None = None
    path_template: str | None = None


@dataclass(frozen=True)
class Fault:
    id: str
    path: str
    block: str
    symbol: str  # searchable name of the faulty block
    error_class: str
    suite: str
    tests: tuple[str, ...]
    facts: tuple[str, ...]  # every clue fact id, loc and code first

    @property
    def loc_fact(self) -> str:
        return f"{self.id}:loc"

    @property
    def code_fact(self) -> str:
        return f"{self.id}:code"


@dataclass(frozen=True)
class TestCase:
    id: str
    suite: str
    faults: tuple[str, ...]  # faults this test depends on
    covers: tuple[tuple[str, str], ...]  # (path, block) whose regressions break it


@dataclass(frozen=True)
class Variant:
    """A non-original, non-fix behaviour id."""

    id: str
    kind: str  # wrong | neutral | competing | new
    serial: int
    regress: bool = False
    error_class: str | None = None
    fault: str | None = None  # competing variants: the fault they compete on
    passes: tuple[str, ...] = ()  # competing variants: fault tests they happen to satisfy


@dataclass
class Service:
    """External dependency. ``mode``: ok | transient (fails the first
    ``fail_attempts`` calls) | persistent (never recovers) | needs_fix (fails
    until fault ``requires`` is fixed: a diagnosed-repair cause)."""

    host: str
    mode: str = "ok"
    fail_attempts: int = 0
    error_class: str = "TimeoutError"
    requires: str | None = None
    calls: int = 0

    def __post_init__(self) -> None:
        if self.mode not in SERVICE_MODES:
            raise ValueError(f"unknown service mode {self.mode!r}")
        if self.mode == "needs_fix" and not self.requires:
            raise ValueError("needs_fix service requires a fault id")


@dataclass
class Resource:
    """Pollable resource. Not ready for the first ``ready_after`` polls, ready
    afterwards; ``ready_after=None`` never becomes ready (persistent)."""

    id: str
    ready_after: int | None = 0
    polls: int = 0


@dataclass
class Vocab:
    """Per-world picks from the global pools (so sessions vary)."""

    hosts: list[str]
    path_templates: list[str]
    resources: list[str]
    decoy_queries: list[str]
    plan_actions: list[str]
    services: list[str]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_CODE_SYNTAX: dict[str, dict[str, Any]] = {
    "python": dict(cm="#", head=lambda mod: [f'"""{mod} module."""', "", "from __future__ import annotations", "", "import logging", "", "log = logging.getLogger(__name__)", ""],
                   open="def {name}({p}, opts=None):", decl="    {local}{sp}={sp}{p}", stmt="    {local}{sp}={sp}{expr}",
                   guard=["    if {p} is None:", "        return {local}"], ret="    return {local}", close=[""], ind="    "),
    "typescript": dict(cm="//", head=lambda mod: [f"// {mod}.ts", "import { strict as assert } from 'node:assert';", ""],
                       open="export function {name}({p}: Input, opts?: Options): Output {{", decl="  let {local}{sp}={sp}{p};", stmt="  {local}{sp}={sp}{expr};",
                       guard=["  if ({p} == null) {{", "    return {local};", "  }}"], ret="  return {local};", close=["}", ""], ind="  "),
    "go": dict(cm="//", head=lambda mod: [f"package {mod.split('_')[0]}", "", 'import "fmt"', ""],
               open="func {name}({p} Input) Output {{", decl="\t{local}{sp}:={sp}{p}", stmt="\t{local}{sp}={sp}{expr}",
               guard=["\tif {p} == nil {{", "\t\treturn {local}", "\t}}"], ret="\treturn {local}", close=["}", ""], ind="\t"),
    "rust": dict(cm="//", head=lambda mod: [f"//! {mod}", "use crate::prelude::*;", ""],
                 open="pub fn {name}({p}: &Input) -> Output {{", decl="    let mut {local}{sp}={sp}{p}.clone();", stmt="    {local}{sp}={sp}{expr};",
                 guard=["    if {p}.is_empty() {{", "        return {local};", "    }}"], ret="    {local}", close=["}", ""], ind="    "),
    "java": dict(cm="//", head=lambda mod: [f"package com.acme.{mod};", "", "import java.util.Objects;", "", f"public final class {_pascal(mod)} {{", ""],
                 open="    public static Output {name}(Input {p}) {{", decl="        var {local}{sp}={sp}{p};", stmt="        {local}{sp}={sp}{expr};",
                 guard=["        if ({p} == null) {{", "            return {local};", "        }}"], ret="        return {local};", close=["    }", ""], ind="        ",
                 tail=["}"]),
    "ruby": dict(cm="#", head=lambda mod: ["# frozen_string_literal: true", "", f"module {_pascal(mod)}"],
                 open="  def self.{name}({p})", decl="    {local}{sp}={sp}{p}", stmt="    {local}{sp}={sp}{expr}",
                 guard=["    return {local} if {p}.nil?"], ret="    {local}", close=["  end", ""], ind="    ", tail=["end"]),
}

_PARAMS = ("value", "data", "item", "req", "payload", "src", "entry", "row")
_LOCALS = ("result", "acc", "out", "buf", "current", "tmp", "state", "res")


class World:
    """Latent ground truth for one session. Construct with :meth:`generate`
    (randomised) or :meth:`from_spec` (explicit, for tests and hand-built scenarios)."""

    def __init__(
        self,
        language: str,
        files: dict[str, FileModel],
        faults: dict[str, Fault],
        tests: dict[str, TestCase],
        suites: dict[str, tuple[str, ...]],
        clues: list[Clue],
        services: dict[str, Service],
        resources: dict[str, Resource],
        vocab: Vocab,
        lint_targets: tuple[str, ...],
        build_targets: tuple[str, ...],
    ) -> None:
        self.language = language
        self.lang = LANGUAGES[language]
        self.files = files
        self.faults = faults
        self.tests = tests
        self.suites = suites
        self.clues = clues
        self.services = services
        self.resources = resources
        self.vocab = vocab
        self.lint_targets = lint_targets
        self.build_targets = build_targets
        self.variants: dict[str, Variant] = {}
        self._serial = 0
        self._competing: dict[tuple[str, str], str] = {}
        self._render_cache: dict[str, tuple[str, dict[str, tuple[int, int]]]] = {}
        self._ws_cache: tuple[str, str] | None = None
        self._scope_cache: dict[str, str] = {}
        self._dirs_cache: list[str] | None = None
        self._block_fault: dict[tuple[str, str], str] = {(f.path, f.block): f.id for f in faults.values()}

    # -- construction ----------------------------------------------------

    @classmethod
    def from_spec(
        cls,
        *,
        language: str = "python",
        files: Mapping[str, Sequence[str]],
        faults: Sequence[Mapping[str, Any]],
        healthy_tests: Mapping[str, Sequence[tuple[str, Sequence[str]]]] | None = None,
        services: Sequence[Service] = (),
        resources: Sequence[Resource] = (),
        vocab: Vocab | None = None,
        lint_targets: Sequence[str] | None = None,
        build_targets: Sequence[str] | None = None,
    ) -> "World":
        """Explicit construction.

        ``files``: path -> block names (all blocks start in behaviour ``orig``).
        ``faults``: dicts with ``id, path, block, suite, tests, error_class`` and
        optional ``clues`` (list of tuples: ``("read", path, block)``,
        ``("search", query[, scope])``, ``("lint", target)``, ``("build", target)``,
        ``("http", host, path_template)``) and ``symbol`` (default: the block name).
        Every fault automatically gets a ``loc`` fact (revealed by its first
        failing test, or by searching its symbol in a scope containing its path)
        and a ``code`` fact (revealed by reading its block).
        ``healthy_tests``: suite -> [(test_id, ["path:block", ...] covered)].
        Test files named by the test ids are created automatically.
        """
        lang = LANGUAGES[language]
        fmodels: dict[str, FileModel] = {}

        def add_file(path: str, blocks: Iterable[str], role: str | None = None) -> None:
            path = normalise_path(path)
            blocks = tuple(dict.fromkeys(blocks))
            if path in fmodels:
                merged = tuple(dict.fromkeys(fmodels[path].blocks + blocks))
                fmodels[path].blocks = merged
                fmodels[path].states = {b: fmodels[path].states.get(b, BlockState()) for b in merged}
                return
            ext = "." + path.rsplit(".", 1)[-1] if "." in path.rsplit("/", 1)[-1] else ""
            code_lang = _EXT_LANG.get(ext)
            fmodels[path] = FileModel(path, code_lang, blocks, {b: BlockState() for b in blocks}, role or _role_for(path, code_lang))

        for path, blocks in files.items():
            add_file(path, blocks)

        fault_objs: dict[str, Fault] = {}
        clue_list: list[Clue] = []
        tests: dict[str, TestCase] = {}
        suite_members: dict[str, list[str]] = {}

        def add_test(tid: str, suite: str, fault_ids: tuple[str, ...], covers: tuple[tuple[str, str], ...]) -> None:
            if tid in tests:
                raise ValueError(f"duplicate test id {tid!r}")
            tests[tid] = TestCase(tid, suite, fault_ids, covers)
            suite_members.setdefault(suite, []).append(tid)
            tpath, _, case = tid.partition("::")
            add_file(tpath, [_ident(case or "case")], "test")

        for spec in faults:
            fid = spec["id"]
            path = normalise_path(spec["path"])
            block = spec["block"]
            if path not in fmodels or block not in fmodels[path].blocks:
                raise ValueError(f"fault {fid}: block {path}:{block} not in files")
            symbol = spec.get("symbol", block)
            scope = path.split("/", 1)[0] if "/" in path else "."
            facts = [f"{fid}:loc", f"{fid}:code"]
            clue_list.append(Clue(f"{fid}:loc", fid, "search", query=symbol.lower(), scope=scope, path=path))
            clue_list.append(Clue(f"{fid}:code", fid, "read", path=path, block=block))
            for i, c in enumerate(spec.get("clues", ()), start=1):
                fact = f"{fid}:c{i}"
                facts.append(fact)
                kind = c[0]
                if kind == "read":
                    add_file(c[1], [c[2]])
                    clue_list.append(Clue(fact, fid, "read", path=normalise_path(c[1]), block=c[2]))
                elif kind == "search":
                    sc = c[2] if len(c) > 2 else "."
                    clue_list.append(Clue(fact, fid, "search", query=" ".join(c[1].lower().split()), scope=sc, path=path))
                elif kind in ("lint", "build"):
                    clue_list.append(Clue(fact, fid, kind, target=c[1], path=path))
                elif kind == "http":
                    clue_list.append(Clue(fact, fid, "http", host=c[1].lower(), path_template=c[2]))
                else:
                    raise ValueError(f"unknown clue kind {kind!r}")
            suite = spec["suite"]
            tids = tuple(spec["tests"])
            if len(tids) < 2:
                raise ValueError(f"fault {fid}: needs >= 2 dependent tests (competing variants split them)")
            for tid in tids:
                add_test(tid, suite, (fid,), ((path, block),))
            fault_objs[fid] = Fault(fid, path, block, symbol, spec["error_class"], suite, tids, tuple(facts))

        for suite, entries in (healthy_tests or {}).items():
            for tid, covers in entries:
                cov = tuple((normalise_path(c.rsplit(":", 1)[0]), c.rsplit(":", 1)[1]) for c in covers)
                add_test(tid, suite, (), cov)

        suites = {s: tuple(sorted(ids)) for s, ids in sorted(suite_members.items())}
        vocab = vocab or Vocab(list(HOSTS[:4]), list(PATH_TEMPLATES[:4]), [], list(DECOY_QUERIES[:6]), list(PLAN_ACTIONS[:4]), list(SERVICE_NAMES[:3]))
        svc = {s.host.lower(): s for s in services}
        res = {r.id: r for r in resources}
        files_sorted = {p: fmodels[p] for p in sorted(fmodels)}
        return cls(
            language, files_sorted, fault_objs, tests, suites, clue_list, svc, res, vocab,
            tuple(lint_targets or lang.lint_targets), tuple(build_targets or lang.build_targets),
        )

    @classmethod
    def generate(
        cls,
        rng: random.Random,
        *,
        language: str | None = None,
        n_faults: int | None = None,
        extra_clues: tuple[int, int] = (0, 2),
        clue_kinds: Sequence[str] = ("config", "log", "keyword", "lint", "build", "http"),
        n_modules: int | None = None,
        n_suites: int | None = None,
        n_decoys: int | None = None,
        tests_per_fault: tuple[int, int] = (2, 3),
        single_suite: bool = False,
    ) -> "World":
        """Randomised world: 1-4 faults across several modules, suites, configs, docs and logs.

        ``extra_clues``: (min, max) cause clues per fault beyond ``loc``/``code``,
        drawn from ``clue_kinds``. ``single_suite``: put every fault's tests in
        the first suite (one suite reveals every fault), else each fault picks one.
        """
        lang_name = language or rng.choice(LANGUAGE_NAMES)
        lang = LANGUAGES[lang_name]
        nf = n_faults if n_faults is not None else rng.choice((1, 2, 2, 3, 3, 4))
        if not 1 <= nf <= 4:
            raise ValueError("n_faults must be in 1..4")
        org = rng.choice(ORGS)
        pkgs = rng.sample(PACKAGES, rng.randint(1, 3))
        nm = n_modules if n_modules is not None else rng.randint(max(3, nf + 1), 8)
        mods = rng.sample(MODULES, nm)
        ns = n_suites if n_suites is not None else rng.randint(1, min(3, len(lang.suite_tags)))
        tags = rng.sample(lang.suite_tags, ns)
        suite_ids = [lang.suite_id.format(tag=t) for t in tags]

        files: dict[str, list[str]] = {}
        src_blocks: list[tuple[str, str, str]] = []  # (path, block, mod)
        mod_pkg: dict[str, str] = {}
        for mod in mods:
            pkg = rng.choice(pkgs)
            mod_pkg[mod] = pkg
            path = lang.source_path.format(pkg=pkg, mod=mod, cmod=_camel(mod), Mod=_pascal(mod), org=org)
            names: list[str] = []
            for _ in range(rng.randint(2, 5)):
                nm_ = _fn_name(lang, f"{rng.choice(VERBS)}_{rng.choice(NOUNS)}")
                if nm_ not in names:
                    names.append(nm_)
            files[path] = names
            src_blocks.extend((path, b, mod) for b in names)

        def test_file(mod: str, tag: str) -> str:
            return lang.test_path.format(pkg=mod_pkg[mod], mod=mod, cmod=_camel(mod), Mod=_pascal(mod), tag=tag, Tag=tag.capitalize(), org=org)

        def case_id(mod: str, snake_case: str) -> str:
            return lang.test_case.format(
                case=snake_case, Case=_pascal(snake_case), cAse=_camel(snake_case), words=snake_case.replace("_", " "), mod=mod
            )

        used_ids: set[str] = set()

        def new_test_id(mod: str, tag: str, stem: str) -> str:
            base = f"{test_file(mod, tag)}::{case_id(mod, stem)}"
            tid, k = base, 2
            while tid in used_ids:
                tid = f"{base}_{k}" if lang.name != "typescript" else f"{base} {k}"
                k += 1
            used_ids.add(tid)
            return tid

        fault_sites = rng.sample(src_blocks, nf)
        faults: list[dict[str, Any]] = []
        for i, (path, block, mod) in enumerate(fault_sites, start=1):
            fid = f"F{i}"
            tag_i = 0 if single_suite or ns == 1 else rng.randrange(ns)
            tag = tags[tag_i]
            ntests = rng.randint(*tests_per_fault)
            snake_block = re.sub(r"(?<!^)(?=[A-Z])", "_", block).lower()
            tids = [new_test_id(mod, tag, f"{snake_block}_{sc}") for sc in rng.sample(SCENARIOS, ntests)]
            k_extra = rng.randint(*extra_clues)
            kinds = rng.sample(list(clue_kinds), min(k_extra, len(clue_kinds)))
            clues: list[tuple[Any, ...]] = []
            noun = rng.choice(NOUNS)
            for kind in kinds:
                if kind == "config":
                    cfg_path = rng.choice(lang.configs)
                    clues.append(("read", cfg_path, f"{mod}.{noun}"))
                elif kind == "log":
                    log_path = rng.choice(("logs/app.log", f"logs/{mod}.log", f"logs/ci-{rng.randint(100, 999)}.log"))
                    clues.append(("read", log_path, f"{block}@{rng.randint(1000, 9999)}"))
                elif kind == "keyword":
                    clues.append(("search", f"{noun} {rng.choice(KEYWORDS)}", path.split("/", 1)[0]))
                elif kind == "lint":
                    clues.append(("lint", rng.choice(lang.lint_targets)))
                elif kind == "build":
                    clues.append(("build", rng.choice(lang.build_targets)))
                elif kind == "http":
                    host, tmpl = rng.choice(OBS_ENDPOINTS)
                    clues.append(("http", host, tmpl))
                else:
                    raise ValueError(f"unknown clue kind {kind!r}")
            faults.append(dict(id=fid, path=path, block=block, suite=suite_ids[tag_i], tests=tids,
                               error_class=rng.choice(lang.errors), clues=clues))

        healthy: dict[str, list[tuple[str, list[str]]]] = {}
        for tag, sid in zip(tags, suite_ids):
            for _ in range(rng.randint(3, 9)):
                path, block, mod = rng.choice(src_blocks)
                snake_block = re.sub(r"(?<!^)(?=[A-Z])", "_", block).lower()
                tid = new_test_id(mod, tag, f"{snake_block}_{rng.choice(SCENARIOS)}")
                healthy.setdefault(sid, []).append((tid, [f"{path}:{block}"]))

        nd = n_decoys if n_decoys is not None else rng.randint(6, 12)
        for doc in rng.sample(DECOY_DOCS, min(nd, len(DECOY_DOCS))):
            files.setdefault(doc, [f"section_{j}" for j in range(rng.randint(2, 4))])
        for cfg_path in rng.sample(lang.configs, rng.randint(1, len(lang.configs))):
            files.setdefault(cfg_path, [f"{rng.choice(mods)}.{rng.choice(NOUNS)}" for _ in range(rng.randint(1, 3))])
        for mod in rng.sample(mods, min(2, len(mods))):
            files.setdefault(f"tests/fixtures/{mod}.json", [f"case_{j}" for j in range(rng.randint(1, 3))])

        def fill(pattern: str) -> str:
            return pattern.format(n=rng.randint(10, 9999), svc=rng.choice(SERVICE_NAMES), hex=f"{rng.getrandbits(24):06x}",
                                  name=rng.choice(MODULES))

        vocab = Vocab(
            hosts=rng.sample(HOSTS, 5),
            path_templates=rng.sample(PATH_TEMPLATES, 5),
            resources=[fill(p) for p in rng.sample(POLL_RESOURCES, 4)],
            decoy_queries=rng.sample(DECOY_QUERIES, 8),
            plan_actions=rng.sample(PLAN_ACTIONS, 5),
            services=rng.sample(SERVICE_NAMES, 3),
        )
        return cls.from_spec(language=lang_name, files=files, faults=faults, healthy_tests=healthy, vocab=vocab)

    # -- rendering and digests -------------------------------------------

    def _render(self, path: str) -> tuple[str, dict[str, tuple[int, int]]]:
        cached = self._render_cache.get(path)
        if cached is not None:
            return cached
        fm = self.files[path]
        lines: list[str] = []
        spans: dict[str, tuple[int, int]] = {}
        if fm.lang is not None:
            syn = _CODE_SYNTAX[fm.lang]
            stem = path.rsplit("/", 1)[-1].split(".")[0]
            lines.extend(syn["head"](stem))
            for b in fm.blocks:
                lo = len(lines) + 1
                lines.extend(self._render_code_block(fm, b, syn))
                spans[b] = (lo, len(lines))
            lines.extend(syn.get("tail", []))
        else:
            lines.extend(_data_header(path))
            for b in fm.blocks:
                lo = len(lines) + 1
                lines.extend(self._render_data_block(fm, b))
                spans[b] = (lo, len(lines))
        text = "\n".join(lines) + "\n"
        self._render_cache[path] = (text, spans)
        return text, spans

    def _variant_number(self, path: str, block: str, behaviour: str) -> int:
        if behaviour in ("orig", "fix"):
            return _hint(path, block, "orig") % 900 + 1
        return 1000 + self.variants[behaviour].serial

    def _render_code_block(self, fm: FileModel, block: str, syn: Mapping[str, Any]) -> list[str]:
        st = fm.states[block]
        h = _hint(fm.path, block)
        p = _PARAMS[h % len(_PARAMS)]
        local = _LOCALS[(h >> 8) % len(_LOCALS)] + (str(st.style) if st.style else "")
        sp = " " if st.formatted else ""
        helper = HELPERS[_hint(fm.path, block, st.behaviour) % len(HELPERS)]
        k = self._variant_number(fm.path, block, st.behaviour)
        expr = f"{helper}({local},{sp}{k})" if st.behaviour != "fix" else f"{helper}({local},{sp}{k}{sp}+{sp}1)"
        name = block if fm.role != "test" else _ident(block)
        out = [syn["open"].format(name=name, p=p)]
        if st.style:
            out.append(f"{syn['ind']}{syn['cm']} cleanup pass {st.style}")
        out.append(syn["decl"].format(local=local, p=p, sp=sp))
        if st.behaviour == "fix":
            out.extend(g.format(p=p, local=local) for g in syn["guard"])
        out.append(syn["stmt"].format(local=local, expr=expr, sp=sp))
        out.append(syn["ret"].format(local=local))
        out.extend(syn["close"])
        return out

    def _render_data_block(self, fm: FileModel, block: str) -> list[str]:
        st = fm.states[block]
        k = self._variant_number(fm.path, block, st.behaviour)
        value = f"v{k}" + ("-ok" if st.behaviour == "fix" else "")
        sp = " " if st.formatted else ""
        note = [f"# revised {st.style}"] if st.style else []
        ext = fm.path.rsplit(".", 1)[-1] if "." in fm.path.rsplit("/", 1)[-1] else ""
        if ext in ("toml", "cfg", "ini", "env", "example", "tf"):
            return note + [f"[{block}]", f"value{sp}={sp}\"{value}\"", ""]
        if ext in ("yaml", "yml"):
            return note + [f"{block}:", f"  value:{sp or ' '}{value}" + (" " if st.formatted else ""), ""]
        if ext == "json":
            return [f'  "{block}":{sp}{{"value":{sp}"{value}"}},' + (f'  "_rev": {st.style}' if st.style else "")]
        if ext == "log":
            return [f"[2026-01-0{_hint(block) % 9 + 1}T0{_hint(block, 1) % 10}:00:00Z] ERROR {block}:{sp}code={value}" + (f" rev={st.style}" if st.style else "")]
        if ext == "md":
            return [f"## {block.replace('_', ' ').title()}", "", f"Details{sp}:{sp}{value}."] + ([f"_Revision {st.style}._"] if st.style else []) + [""]
        return note + [f"# {block}", f"{block.upper().replace('.', '_')}{sp}={sp}{value}", ""]

    def text(self, path: str) -> str:
        return self._render(path)[0]

    def spans(self, path: str) -> dict[str, tuple[int, int]]:
        return dict(self._render(path)[1])

    def line_count(self, path: str) -> int:
        return self.text(path).count("\n")

    def region(self, path: str, lo: int | None, hi: int | None) -> str:
        lines = self.text(path).split("\n")
        lo_i = 1 if lo is None else max(1, lo)
        hi_i = len(lines) if hi is None else hi
        return "\n".join(lines[lo_i - 1 : hi_i])

    def texts(self) -> dict[str, str]:
        return {p: self.text(p) for p in self.files}

    def _digests(self) -> tuple[str, str]:
        if self._ws_cache is None:
            ws = workspace_digest(self.texts())
            beh = hashlib.sha256(
                repr(sorted((p, b, fm.states[b].behaviour) for p, fm in self.files.items() for b in fm.blocks)).encode()
            ).hexdigest()
            self._ws_cache = (ws, beh)
        return self._ws_cache

    def ws_id(self) -> str:
        """Latent workspace content id == ``fingerprint.workspace_digest`` of the true contents."""
        return self._digests()[0]

    def behaviour_key(self) -> str:
        """Id of the behavioural workspace (cosmetic text ignored)."""
        return self._digests()[1]

    def scope_digest(self, scope: str) -> str:
        d = self._scope_cache.get(scope)
        if d is None:
            d = workspace_digest({p: self.text(p) for p in self.files if _under(p, scope)} or {"__empty__": ""})
            self._scope_cache[scope] = d
        return d

    # -- mutation --------------------------------------------------------

    def _invalidate(self, path: str) -> None:
        self._render_cache.pop(path, None)
        self._ws_cache = None
        self._scope_cache = {}
        if path not in self.files:
            self._dirs_cache = None

    def set_states(self, path: str, states: Mapping[str, BlockState]) -> None:
        fm = self.files[path]
        if set(states) != set(fm.blocks):
            raise ValueError(f"state blocks for {path} must match file blocks")
        fm.states = {b: states[b] for b in fm.blocks}
        self._invalidate(path)

    def add_file(self, path: str, blocks: Sequence[str], behaviour: str, role: str = "note") -> None:
        path = normalise_path(path)
        if path in self.files:
            raise ValueError(f"{path} exists")
        ext = "." + path.rsplit(".", 1)[-1] if "." in path.rsplit("/", 1)[-1] else ""
        fm = FileModel(path, _EXT_LANG.get(ext), tuple(blocks), {b: BlockState(behaviour) for b in blocks}, role)
        self.files[path] = fm
        self.files = {p: self.files[p] for p in sorted(self.files)}
        self._dirs_cache = None
        self._invalidate(path)

    def remove_file(self, path: str) -> None:
        del self.files[path]
        self._invalidate(path)

    def new_variant(self, kind: str, *, regress: bool = False, fault: str | None = None, passes: Sequence[str] = ()) -> str:
        self._serial += 1
        vid = f"{kind[0]}{self._serial}"
        errs = self.lang.errors
        err = errs[_hint(vid, kind) % len(errs)] if regress else None
        self.variants[vid] = Variant(vid, kind, self._serial, regress, err, fault, tuple(passes))
        return vid

    def competing_variant(self, fault_id: str, label: str) -> str:
        """Behaviour id of competing alternative ``label`` for a fault.

        Competing alternatives never fix the fault (progress does not move) and
        cause no regression. Alternative number ``i`` (in order of first use)
        satisfies the fault's tests at positions ``j`` with ``(j + i) % 2 == 0``,
        so alternatives A and B make different tests pass, never all of them.
        """
        key = (fault_id, label)
        if key not in self._competing:
            f = self.faults[fault_id]
            i = sum(1 for k in self._competing if k[0] == fault_id)
            passes = tuple(t for j, t in enumerate(f.tests) if (j + i) % 2 == 0)
            self._competing[key] = self.new_variant("competing", fault=fault_id, passes=passes)
        return self._competing[key]

    # -- latent queries --------------------------------------------------

    def behaviour(self, path: str, block: str) -> str:
        return self.files[path].states[block].behaviour

    def fault_at(self, path: str, block: str) -> str | None:
        return self._block_fault.get((path, block))

    def is_fixed(self, fault_id: str) -> bool:
        f = self.faults[fault_id]
        return self.behaviour(f.path, f.block) == "fix"

    def fixed_faults(self) -> list[str]:
        return [fid for fid in self.faults if self.is_fixed(fid)]

    def unfixed_faults(self) -> list[Fault]:
        return [f for f in self.faults.values() if not self.is_fixed(f.id)]

    def regressions(self) -> list[str]:
        out = []
        for p, fm in self.files.items():
            for b in fm.blocks:
                v = self.variants.get(fm.states[b].behaviour)
                if v is not None and v.regress:
                    out.append(f"reg:{p}:{b}")
        return out

    def causes(self) -> set[str]:
        """Active problem ids: unfixed faults and active regressions."""
        return {f.id for f in self.unfixed_faults()} | set(self.regressions())

    def progress(self) -> float:
        """``p`` = (fixed faults - 0.5 * active regressions) / n_faults, clipped to [0, 1]."""
        n = len(self.faults)
        if n == 0:
            return 0.0
        p = (len(self.fixed_faults()) - 0.5 * len(self.regressions())) / n
        return min(1.0, max(0.0, p))

    def test_failure(self, test_id: str) -> tuple[str, str, str] | None:
        """``None`` if the test passes, else ``(error_class, revealed fact, cause id)``."""
        t = self.tests[test_id]
        for fid in t.faults:
            f = self.faults[fid]
            b = self.behaviour(f.path, f.block)
            if b == "fix":
                continue
            v = self.variants.get(b)
            if v is not None and v.kind == "competing" and test_id in v.passes:
                continue
            return f.error_class, f.loc_fact, fid
        for path, block in t.covers:
            if path not in self.files:
                continue
            v = self.variants.get(self.behaviour(path, block))
            if v is not None and v.regress:
                return v.error_class or "AssertionError", f"reg:{path}:{block}:{v.id}", f"reg:{path}:{block}"
        return None

    def run_suite(self, suite: str, selection: Sequence[str] | None) -> tuple[int, int, tuple[str, str, str, str] | None, set[str]]:
        """``(passed, total, first_failure, causes)``; first_failure = (test_id, error_class, fact, cause)."""
        if suite not in self.suites:
            raise ValueError(f"unknown suite {suite!r}")
        ids = self.suites[suite] if selection is None else tuple(sorted(set(selection)))
        for tid in ids:
            if tid not in self.tests or self.tests[tid].suite != suite:
                raise ValueError(f"test {tid!r} is not in suite {suite!r}")
        passed, first, causes = 0, None, set()
        for tid in ids:
            fail = self.test_failure(tid)
            if fail is None:
                passed += 1
                continue
            causes.add(fail[2])
            if first is None:
                first = (tid, fail[0], fail[1], fail[2])
        return passed, len(ids), first, causes

    def read_clues(self, path: str, lo: int | None, hi: int | None) -> list[Clue]:
        spans = self.spans(path) if path in self.files else {}
        out = []
        for c in self.clues:
            if c.kind != "read" or c.path != path or c.block not in spans:
                continue
            b_lo, b_hi = spans[c.block]
            if (lo is None or b_hi >= lo) and (hi is None or b_lo <= hi):
                out.append(c)
        return out

    def search_clues(self, scope: str, query: str) -> list[Clue]:
        q = " ".join(query.lower().split())
        return [c for c in self.clues if c.kind == "search" and c.query in q and c.path is not None and _under(c.path, scope)]

    def tool_clues(self, kind: str, target: str) -> list[Clue]:
        return [c for c in self.clues if c.kind == kind and c.target == target]

    def http_clues(self, host: str, path_template: str) -> list[Clue]:
        return [c for c in self.clues if c.kind == "http" and c.host == host.lower() and c.path_template == path_template]

    def clues_for(self, fault_id: str) -> list[Clue]:
        return [c for c in self.clues if c.fault == fault_id]

    def decoy_paths(self) -> list[str]:
        """Files carrying no read clue and no fault (novel reads of these yield nothing)."""
        clue_paths = {c.path for c in self.clues if c.kind == "read"}
        fault_paths = {f.path for f in self.faults.values()}
        return [p for p in self.files if p not in clue_paths and p not in fault_paths]

    def dirs(self) -> list[str]:
        if self._dirs_cache is None:
            out = {"."}
            for p in self.files:
                parts = p.split("/")[:-1]
                for i in range(1, len(parts) + 1):
                    out.add("/".join(parts[:i]))
            self._dirs_cache = sorted(out)
        return list(self._dirs_cache)

    def listing(self, path: str) -> list[str] | None:
        if path != "." and path not in self.dirs():
            return None
        prefix = "" if path == "." else path + "/"
        entries = set()
        for p in self.files:
            if p.startswith(prefix):
                rest = p[len(prefix) :]
                entries.add(rest.split("/", 1)[0] + ("/" if "/" in rest else ""))
        return sorted(entries)

    # -- external --------------------------------------------------------

    def service(self, host: str) -> Service:
        h = host.lower()
        if h not in self.services:
            self.services[h] = Service(h)
        return self.services[h]

    def add_service(self, host: str, mode: str = "ok", *, fail_attempts: int = 0, error_class: str | None = None,
                    requires: str | None = None) -> Service:
        err = error_class or (AUTH_ERRORS[0] if mode == "needs_fix" else "TimeoutError")
        s = Service(host.lower(), mode, fail_attempts, err, requires)
        self.services[s.host] = s
        return s

    def add_resource(self, resource_id: str, ready_after: int | None = 0) -> Resource:
        r = Resource(resource_id, ready_after)
        self.resources[resource_id] = r
        return r

    def resource(self, resource_id: str) -> Resource:
        if resource_id not in self.resources:
            self.resources[resource_id] = Resource(resource_id, 0)
        return self.resources[resource_id]

    def summary(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "faults": [
                {"id": f.id, "path": f.path, "block": f.block, "suite": f.suite, "tests": list(f.tests),
                 "error_class": f.error_class, "facts": list(f.facts)}
                for f in self.faults.values()
            ],
            "suites": {s: len(ids) for s, ids in self.suites.items()},
            "n_files": len(self.files),
            "services": {h: {"mode": s.mode, "fail_attempts": s.fail_attempts, "requires": s.requires}
                         for h, s in sorted(self.services.items())},
            "resources": {r.id: r.ready_after for r in self.resources.values()},
        }


def _role_for(path: str, code_lang: str | None) -> str:
    if code_lang is not None:
        return "test" if ("test" in path.rsplit("/", 1)[-1].lower() or path.startswith(("tests/", "spec/"))) else "source"
    if path.endswith(".log"):
        return "log"
    if path.endswith(".md"):
        return "doc"
    if "fixtures" in path:
        return "fixture"
    return "config"


def _data_header(path: str) -> list[str]:
    name = path.rsplit("/", 1)[-1]
    if name.endswith(".json"):
        return ["{", f'  "_file": "{name}",']
    if name.endswith(".md"):
        return [f"# {name[:-3].replace('-', ' ').replace('_', ' ').title()}", ""]
    if name.endswith(".log"):
        return [f"=== {name} ==="]
    return [f"# {name}", ""]
