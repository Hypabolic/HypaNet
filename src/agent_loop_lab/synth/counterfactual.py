"""Counterfactual session pairs (M1_CONTRACTS §4, spec v0.2 §7.2).

A pair is two sessions that share an identical prefix, both the emitted events
and the full latent state (world, facts ``K``, episodes, history, RNG), and
then branch on exactly one causal latent variable. Both continuations run the
*same* scripted-agent policy, which reacts only to what the agent observes
(outcomes, failing test ids) and to the policy parameters drawn before the
branch. Every effect after the branch is decided by the engine from latent
state, and labels come from :func:`labeller.label_session`. Nothing here looks
at encoded features or fingerprints.

Branch ``"a"`` is the benign continuation and ``"b"`` the degenerate one:

``retry_success_vs_doomed``
    Variable ``service_mode`` (world): the external service behind a failing
    call is ``transient`` (recovers after ``k <= R_max`` failing calls) in
    ``a`` and ``persistent`` in ``b``. The prefix ends with the initial failing
    call. The policy retries immediately while the call fails (up to a cap
    beyond ``R_max``), then either does the remaining repair work or gives up.
    Intended: RECOVERY then PRODUCTIVE in ``a``; STALLED in ``b`` (attempts
    beyond ``R_max`` are REDUNDANT).
``improving_vs_identical_failures``
    Variable ``edit_fixes`` (the agent's patch): in an edit-then-test loop on
    one suite, each edit at the location the last failing run revealed is the
    fault's real fix in ``a`` and a text-only (cosmetic) edit in ``b``. The
    prefix ends with the failing run that reveals the first location.
    Intended: PRODUCTIVE in ``a``; STALLED in ``b`` (cosmetic edits and
    re-runs on a behaviourally unchanged workspace are REDUNDANT, r2 reading).
``productive_vs_abab``
    Variable ``next_edit`` (the agent's patch): after trying competing
    alternatives A then B on a fault (both fail its tests), the next edit is a
    new, correct fix in ``a`` and a return to A in ``b``. The policy keeps
    editing while the fault's tests fail, alternating between the
    alternatives it has. Intended: PRODUCTIVE in ``a``; OSCILLATING in ``b``.

Each record carries ``latent_trace["counterfactual"]`` with the contract keys
``pair_id``, ``kind``, ``branch``, ``branch_index`` (the first event after the
branch, i.e. the prefix is ``events[:branch_index]``) and ``variable``, plus
``value`` (this branch's value of the variable) and ``contrast_endpoints``
(window endpoints present in both sessions, at or after ``branch_index``,
where ``a``'s label is in the benign set and ``b``'s is the degenerate label).

The prefix events of ``b`` equal ``a``'s except for the session identity
fields that :meth:`Session.fork` re-stamps (``session_id``, ``event_id``,
``provenance.trace_id`` / ``span_ids``); see :func:`normalise_identity`.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..oracle_config import OracleConfig, load_oracle_config
from ..schema import EVENTS_PER_WINDOW, CanonicalEvent, OracleLabel, OracleRecord
from .engine import MIN_EVENTS, OpResult, Session, competing, cosmetic, fix
from .world import HOSTS, HTTP_ERRORS, Fault

__all__ = [
    "KINDS",
    "INTENDED",
    "BranchPoint",
    "branch_point",
    "branch_session_id",
    "generate_pair",
    "generate_pair_from_seed",
    "normalise_identity",
    "latent_snapshot",
    "post_branch_labels",
    "contrast_endpoints",
]

KINDS: tuple[str, ...] = (
    "retry_success_vs_doomed",
    "improving_vs_identical_failures",
    "productive_vs_abab",
)

# kind -> (benign labels expected in branch a, degenerate label expected in branch b)
INTENDED: Mapping[str, tuple[frozenset[OracleLabel], OracleLabel]] = {
    "retry_success_vs_doomed": (frozenset({OracleLabel.RECOVERY, OracleLabel.PRODUCTIVE}), OracleLabel.STALLED),
    "improving_vs_identical_failures": (frozenset({OracleLabel.PRODUCTIVE}), OracleLabel.STALLED),
    "productive_vs_abab": (frozenset({OracleLabel.PRODUCTIVE}), OracleLabel.OSCILLATING),
}

_VARIABLES: Mapping[str, tuple[str, Any, Any]] = {
    "retry_success_vs_doomed": ("service_mode", "transient", "persistent"),
    "improving_vs_identical_failures": ("edit_fixes", True, False),
    "productive_vs_abab": ("next_edit", "fix", "revert"),
}

_TEXT_COSMETICS = ("rename_local", "comment", "whitespace", "docstring")  # never "format": idempotent
_METHODS = ("GET", "GET", "GET", "POST", "PUT", "PATCH", "DELETE")


@dataclass
class BranchPoint:
    """Both sessions right after the branch variable was set, before any
    post-branch operation. ``policy_a`` / ``policy_b`` are the continuation
    parameters; they differ only in ``variable`` when it is a policy variable."""

    kind: str
    pair_id: str
    variable: str
    value_a: Any
    value_b: Any
    branch_index: int
    a: Session
    b: Session
    policy_a: dict[str, Any]
    policy_b: dict[str, Any]
    ctx: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared scripted-agent helpers (operate only through the engine)
# ---------------------------------------------------------------------------


def _neutral(s: Session, rng: random.Random, k: int, used_plans: list[str]) -> None:
    """``k`` novel, fruitless operations (NONE): decoys and first plans."""
    for _ in range(k):
        left = [a for a in s.world.vocab.plan_actions if a not in used_plans]
        roll = rng.random()
        if roll < 0.2 and left:
            a = rng.choice(left)
            used_plans.append(a)
            s.plan(a)
        elif roll < 0.4:
            s.novel_decoy(kinds=("list",))
        else:
            s.novel_decoy(kinds=("read", "search"))


def _locate(s: Session, rng: random.Random, f: Fault) -> None:
    """Reveal a fault's location by a symbol search (NEW_FACT)."""
    if not s.knows(f.loc_fact):
        scope = f.path.split("/", 1)[0] if rng.random() < 0.7 else "."
        s.search(scope, f.symbol)


def _repair(s: Session, rng: random.Random, f: Fault, use_write: bool) -> OpResult:
    """Productive repair of one fault: locate, read, fix, targeted verification."""
    _locate(s, rng, f)
    if not s.knows(f.code_fact):
        s.read(f.path, block=f.block, pad=rng.randint(0, 4))
    if use_write:
        s.write(f.path, fix(f))
    else:
        s.edit(f.path, fix(f))
    return s.run_tests(f.suite, list(f.tests))


def _pad(s: Session) -> None:
    while s.n < MIN_EVENTS:
        s.novel_decoy()


def _verify(s: Session, lint: bool, skip: tuple[str, ...] = ()) -> None:
    """Wrap-up after the work is done: run every suite once (green when all
    faults are fixed) and optionally lint. ``skip``: suites the agent has just
    run on the current workspace (re-running them would be REDUNDANT)."""
    for suite in s.world.suites:
        if suite not in skip and s.room(2):
            s.run_tests(suite)
    if lint and s.room(2):
        s.lint(s.world.lint_targets[0])


def _fault_of(s: Session, test_id: str | None) -> Fault | None:
    if test_id is None:
        return None
    fids = s.world.tests[test_id].faults
    return s.world.faults[fids[0]] if fids else None


def _free_host(s: Session, rng: random.Random) -> str:
    w = s.world
    taken = set(w.services) | {c.host.lower() for c in w.clues if c.host}
    pool = [h for h in w.vocab.hosts if h.lower() not in taken]
    if not pool:
        pool = [h for h in HOSTS if h.lower() not in taken]
    return rng.choice(pool).lower()


# ---------------------------------------------------------------------------
# Kind 1: retry_success_vs_doomed
# ---------------------------------------------------------------------------


def _prefix_retry(s: Session, rng: random.Random, cfg: OracleConfig) -> tuple[dict, dict]:
    w = s.world
    used_plans: list[str] = []
    host = _free_host(s, rng)
    tmpl = rng.choice(w.vocab.path_templates)
    method = rng.choice(_METHODS)
    body = None
    if method in ("POST", "PATCH"):
        body = json.dumps({"ref": f"r-{rng.getrandbits(16):04x}", "dry_run": rng.random() < 0.5}, sort_keys=True)
    k = rng.randint(1, cfg.r_max)  # failing calls in total before recovery (transient branch)
    w.add_service(host, "transient", fail_attempts=k, error_class=rng.choice(HTTP_ERRORS))
    faults = list(w.faults.values())
    n_pre = rng.randint(0, len(faults) - 1)  # faults repaired before the outage; >= 1 left for afterwards
    policy = {
        "cap": rng.randint(cfg.r_max + 6, cfg.r_max + 9),  # retries before giving up
        "use_write": rng.random() < 0.2,
        "wrap": rng.choice(("yield", "ask_yield", "none")),
        "verify_lint": rng.random() < 0.3,
    }
    _neutral(s, rng, rng.randint(1 if n_pre == 0 else 0, 4), used_plans)
    for f in faults[:n_pre]:
        _repair(s, rng, f, policy["use_write"])
    if rng.random() < 0.4:
        _neutral(s, rng, rng.randint(1, 2), used_plans)
    r = s.http(method, host, tmpl, body=body)  # initial failure (NONE); arms the retry candidate
    assert r.failed
    ctx = {"host": host, "method": method, "tmpl": tmpl, "body": body, "fail_attempts": k}
    return policy, ctx


def _set_retry(b: Session, ctx: dict, value: str) -> None:
    b.world.services[ctx["host"]].mode = value  # the one latent variable


def _continue_retry(s: Session, policy: dict, ctx: dict) -> None:
    rng = s.rng
    attempts = 0
    r = None
    while attempts < policy["cap"]:
        r = s.http(ctx["method"], ctx["host"], ctx["tmpl"], body=ctx["body"])
        attempts += 1
        if r.ok:
            break
    if r is not None and r.ok:
        for f in s.world.unfixed_faults():
            if not s.room(4):
                break
            _repair(s, rng, f, policy["use_write"])
        _verify(s, policy["verify_lint"])
        if policy["wrap"] != "none" and s.room(1):
            s.yield_("done")
    else:
        if policy["wrap"] == "ask_yield":
            s.ask("confirm_scope")
        if policy["wrap"] != "none":
            s.yield_("blocked")
    _pad(s)


# ---------------------------------------------------------------------------
# Kind 2: improving_vs_identical_failures
# ---------------------------------------------------------------------------


def _prefix_improving(s: Session, rng: random.Random, cfg: OracleConfig) -> tuple[dict, dict]:
    w = s.world
    used_plans: list[str] = []
    nf = len(w.faults)
    suite = next(iter(w.faults.values())).suite
    read_rate = rng.choice((0.0, 0.0, 0.3, 0.6))  # re-read the failing block before an edit
    read_first = read_rate > 0 or rng.random() < 0.5  # the agent reads the first location before editing
    policy = {
        "cap": rng.randint(max(4, nf), 7),  # edit/test iterations while the suite fails
        "use_write": rng.random() < 0.2,
        "styles": [rng.choice(_TEXT_COSMETICS) for _ in range(7)],
        "reads": [rng.random() < read_rate for _ in range(7)],  # read the failing block before the edit
        "pads": [rng.randint(0, 3) for _ in range(7)],
        "suffix": rng.choice(("yield", "none")),
        "verify_lint": rng.random() < 0.3,
    }
    fixed_ops = 1 + int(read_first) + 2 * nf + int(policy["suffix"] == "yield")
    _neutral(s, rng, max(MIN_EVENTS - fixed_ops, rng.randint(0, 6)), used_plans)
    r = s.run_tests(suite)  # reveals the first fault's location (NEW_FACT)
    assert r.failed
    f = _fault_of(s, r.failing_test)
    if read_first and f is not None:
        s.read(f.path, block=f.block, pad=rng.randint(0, 6))
    ctx = {"suite": suite, "failing_test": r.failing_test}
    return policy, ctx


def _continue_improving(s: Session, policy: dict, ctx: dict) -> None:
    failing = ctx["failing_test"]
    r: OpResult | None = None
    for i in range(policy["cap"]):
        f = _fault_of(s, failing)
        if f is None or s.world.is_fixed(f.id):
            break  # failing for a reason outside the fault set: never re-run unchanged
        if policy["reads"][i]:
            s.read(f.path, block=f.block, pad=policy["pads"][i])  # NEW_FACT on a new location, else REDUNDANT
        change = fix(f) if policy["edit_fixes"] else cosmetic(policy["styles"][i])
        if policy["use_write"]:
            s.write(f.path, change, block=f.block)
        else:
            s.edit(f.path, change, block=f.block)
        r = s.run_tests(ctx["suite"])
        failing = r.failing_test
        if r.ok:
            break
    if r is not None and r.ok:
        _verify(s, policy["verify_lint"], skip=(ctx["suite"],))
    if policy["suffix"] == "yield" and s.room(1):
        s.yield_("done" if (r is not None and r.ok) else "partial_result")
    _pad(s)


# ---------------------------------------------------------------------------
# Kind 3: productive_vs_abab
# ---------------------------------------------------------------------------


def _prefix_abab(s: Session, rng: random.Random, cfg: OracleConfig) -> tuple[dict, dict]:
    w = s.world
    used_plans: list[str] = []
    faults = list(w.faults.values())
    target = faults[rng.randrange(len(faults))]
    others = [f for f in faults if f.id != target.id]
    policy = {
        "cap": rng.randint(4, 8),  # post-branch edits while the target's tests fail
        "use_write": rng.random() < 0.2,
        "labels": ("A", "B"),
        "giveup": rng.choice(("yield", "ask_yield", "none")),
        "verify_lint": rng.random() < 0.3,
    }
    n_pre = rng.randint(0, len(others))
    _neutral(s, rng, rng.randint(0, 4), used_plans)
    for f in others[:n_pre]:
        _repair(s, rng, f, policy["use_write"])
    if rng.random() < 0.5:
        _neutral(s, rng, rng.randint(1, 2), used_plans)
    if rng.random() < 0.6:
        s.run_tests(target.suite, list(target.tests))  # reveals the location (NEW_FACT)
    else:
        _locate(s, rng, target)
    s.read(target.path, block=target.block, pad=rng.randint(0, 6))
    r = None
    for lab in policy["labels"]:  # two competing alternatives, both fail the fault's tests
        change = competing(target, lab)
        if policy["use_write"]:
            s.write(target.path, change)
        else:
            s.edit(target.path, change)
        r = s.run_tests(target.suite, list(target.tests))
        assert r.failed
    ctx = {"target": target.id}
    return policy, ctx


def _continue_abab(s: Session, policy: dict, ctx: dict) -> None:
    rng = s.rng
    w = s.world
    target = w.faults[ctx["target"]]
    labels = policy["labels"]
    r: OpResult | None = None
    for k in range(policy["cap"]):
        if k == 0 and policy["next_edit"] == "fix":
            change = fix(target)  # a new edit: the real fix
        else:
            change = competing(target, labels[k % len(labels)])  # back to an alternative tried before
        if policy["use_write"]:
            s.write(target.path, change)
        else:
            s.edit(target.path, change)
        r = s.run_tests(target.suite, list(target.tests))
        if r.ok:
            break
    if r is not None and r.ok:
        for f in w.unfixed_faults():
            if not s.room(4):
                break
            _repair(s, rng, f, policy["use_write"])
        _verify(s, policy["verify_lint"])
        if policy["giveup"] != "none" and s.room(1):
            s.yield_("done")
    else:
        if policy["giveup"] == "ask_yield":
            s.ask("clarify_requirement")
        if policy["giveup"] != "none":
            s.yield_("blocked")
    _pad(s)


# ---------------------------------------------------------------------------
# Kind table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Kind:
    world_kwargs: Callable[[random.Random], dict]
    prefix: Callable[[Session, random.Random, OracleConfig], tuple[dict, dict]]
    continue_: Callable[[Session, dict, dict], None]
    set_world: Callable[[Session, dict, Any], None] | None  # world variable setter, else a policy variable


_KIND_TABLE: Mapping[str, _Kind] = {
    "retry_success_vs_doomed": _Kind(
        lambda rng: {"n_faults": rng.choice((1, 2, 2, 3)), "extra_clues": (0, 1)},
        _prefix_retry, _continue_retry, _set_retry,
    ),
    "improving_vs_identical_failures": _Kind(
        lambda rng: {"n_faults": rng.choice((2, 3, 3, 4)), "extra_clues": (0, 1), "single_suite": True,
                     "tests_per_fault": rng.choice(((2, 3), (2, 4)))},
        _prefix_improving, _continue_improving, None,
    ),
    "productive_vs_abab": _Kind(
        lambda rng: {"n_faults": rng.choice((1, 2, 2, 3)), "extra_clues": (0, 1),
                     "single_suite": rng.random() < 0.5, "tests_per_fault": rng.choice(((2, 3), (2, 4), (3, 4)))},
        _prefix_abab, _continue_abab, None,
    ),
}


def branch_session_id(pair_id: str, branch: str) -> str:
    """Opaque session id of ``branch`` (``"a"``/``"b"``) of pair ``pair_id``.

    ``"cf-"`` + the first 16 hex digits of ``sha256(f"{pair_id}:{branch}")``.
    It carries neither the pair id, the kind nor the branch (spec §3.1), and it
    is derived by hashing, not from ``rng``, so no generator draw changes. The
    branch and pair linkage live only in ``latent_trace["counterfactual"]``.
    """
    if branch not in ("a", "b"):
        raise ValueError(f"branch must be 'a' or 'b', got {branch!r}")
    return "cf-" + hashlib.sha256(f"{pair_id}:{branch}".encode()).hexdigest()[:16]


def branch_point(kind: str, rng: random.Random, pair_id: str, cfg: OracleConfig | None = None, *,
                 seed: int | None = None) -> BranchPoint:
    """Build the shared prefix, fork it, and set the branch variable.

    All continuation parameters are drawn from ``rng`` before the fork, so
    ``policy_a`` and ``policy_b`` agree except on a policy variable. Session
    ``a`` keeps ``rng``; ``b`` gets a copy in the same state.
    """
    if kind not in _KIND_TABLE:
        raise KeyError(f"unknown counterfactual kind {kind!r}; expected one of {KINDS}")
    cfg = cfg or load_oracle_config()
    spec = _KIND_TABLE[kind]
    variable, va, vb = _VARIABLES[kind]
    s = Session(rng, branch_session_id(pair_id, "a"), cfg, family=kind, seed=seed, **spec.world_kwargs(rng))
    policy, ctx = spec.prefix(s, rng, cfg)
    branch_index = s.n
    s.meta.update(counterfactual_policy=json.loads(json.dumps(policy)), counterfactual_ctx=json.loads(json.dumps(ctx)))
    b = s.fork(branch_session_id(pair_id, "b"))
    policy_a, policy_b = dict(policy), dict(policy)
    if spec.set_world is not None:
        spec.set_world(b, ctx, vb)  # a already holds va (set up in the prefix)
    else:
        policy_a[variable], policy_b[variable] = va, vb
    return BranchPoint(kind, pair_id, variable, va, vb, branch_index, s, b, policy_a, policy_b, ctx)


def post_branch_labels(record: OracleRecord, branch_index: int | None = None) -> dict[int, OracleLabel]:
    """Window labels whose window holds at least one post-branch event."""
    bi = record.latent_trace["counterfactual"]["branch_index"] if branch_index is None else branch_index
    return {e: lab for e, lab in record.window_labels.items() if e >= bi}


def contrast_endpoints(kind: str, rec_a: OracleRecord, rec_b: OracleRecord, branch_index: int) -> list[int]:
    """Endpoints present in both sessions where ``a`` is benign and ``b`` degenerate."""
    benign, bad = INTENDED[kind]
    la, lb = post_branch_labels(rec_a, branch_index), post_branch_labels(rec_b, branch_index)
    return sorted(e for e in la if e in lb and la[e] in benign and lb[e] is bad)


def generate_pair(kind: str, rng: random.Random, pair_id: str, cfg: OracleConfig | None = None, *,
                  seed: int | None = None
                  ) -> tuple[tuple[list[CanonicalEvent], OracleRecord], tuple[list[CanonicalEvent], OracleRecord]]:
    """``((events_a, record_a), (events_b, record_b))`` for one counterfactual pair.

    Session ids are opaque (:func:`branch_session_id`); ``scenario_family``
    is ``kind``; ``seed`` (optional) is stamped on both records.
    """
    bp = branch_point(kind, rng, pair_id, cfg, seed=seed)
    spec = _KIND_TABLE[kind]
    out = []
    for sess, policy in ((bp.a, bp.policy_a), (bp.b, bp.policy_b)):
        spec.continue_(sess, policy, bp.ctx)
        out.append(sess.finish())
    (ev_a, rec_a), (ev_b, rec_b) = out
    contrast = contrast_endpoints(kind, rec_a, rec_b, bp.branch_index)
    stamped = []
    for branch, value, (ev, rec) in (("a", bp.value_a, (ev_a, rec_a)), ("b", bp.value_b, (ev_b, rec_b))):
        cf = {"pair_id": pair_id, "kind": kind, "branch": branch, "branch_index": bp.branch_index,
              "variable": bp.variable, "value": value, "contrast_endpoints": contrast}
        stamped.append((ev, dataclasses.replace(rec, latent_trace={**rec.latent_trace, "counterfactual": cf})))
    return stamped[0], stamped[1]


def generate_pair_from_seed(kind: str, seed: int, pair_id: str, cfg: OracleConfig | None = None):
    """:func:`generate_pair` with ``random.Random(seed)``; ``seed`` is stamped on both records."""
    return generate_pair(kind, random.Random(seed), pair_id, cfg, seed=seed)


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------


def normalise_identity(event: CanonicalEvent, like: CanonicalEvent) -> CanonicalEvent:
    """``event`` re-stamped with ``like``'s session identity (the fields
    :meth:`Session.fork` changes), for byte comparison of shared prefixes."""
    return dataclasses.replace(
        event, session_id=like.session_id, event_id=like.event_id,
        provenance=dataclasses.replace(event.provenance, trace_id=like.provenance.trace_id,
                                       span_ids=like.provenance.span_ids),
    )


_SNAPSHOT_SKIP = frozenset({"session_id", "trace_id", "_events"})  # identity / emitted stream (compared separately)


def _canon(x: Any) -> Any:
    if isinstance(x, enum.Enum):
        return x.value
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, random.Random):
        return _canon(x.getstate())
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _canon(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, Mapping):
        items = [(json.dumps(_canon(k), sort_keys=True), _canon(v)) for k, v in x.items()]
        return dict(sorted(items))
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted((_canon(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))
    if hasattr(x, "__dict__"):
        return {k: _canon(v) for k, v in sorted(vars(x).items())
                if k not in _SNAPSHOT_SKIP and not k.endswith("_cache")}
    raise TypeError(f"cannot snapshot {type(x).__name__}")


def latent_snapshot(s: Session) -> dict[str, Any]:
    """Canonical, JSON-able latent state of a session (world, facts, episodes,
    histories, RNG state), excluding session identity and the emitted events."""
    return _canon(s)
