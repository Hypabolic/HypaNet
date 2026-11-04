"""Operation engine for the standard generator (M1_CONTRACTS §1, §2).

Every operation a family performs goes through a :class:`Session`. The session
mutates the latent :class:`~agent_loop_lab.synth.world.World`, decides the
operation's latent effect **causally from latent state** (facts ``K``,
behavioural vs text workspace, faults, tests, services, episodes), and emits
one ``(CanonicalEvent, LatentStep)`` pair. :meth:`Session.finish` returns
``(events, OracleRecord)`` with labels from :func:`labeller.label_session`.

Effect rules (applied in this order; see docs/M1_ENGINE.md for the full guide):

1. Episodes. A failure (ERROR, or a NEUTRAL not-ready poll) arms a candidate.
   Only the *immediately following* operation, if it is the same operation on
   the same relevant state, opens an episode (attempt 1). Once open, further
   same-key operations are attempts; one unrelated operation in between is
   tolerated, the 2nd consecutive unrelated operation closes it, and so does a
   non-repair operation that changes the episode's relevant state. A fix of
   one of the episode's causes (a *diagnosed repair*) keeps it open and lets
   the next same-key operation count as an attempt. Attempts ``1..R_max``
   (``P_max`` for polls) are in policy; later ones are REDUNDANT; the attempt
   that succeeds RESOLVES_EPISODE. The initial failing operation is not an
   attempt; its effect comes from rules 2-6.
2. Progress: ``p`` rose -> PROGRESS (credited to the fixing mutation).
3. Facts: new facts added to ``K`` -> NEW_FACT.
4. Redundancy (op-specific, §6.1 with the r2 reading), out-of-policy attempts.
5. Reversion: a mutation back to a workspace state held earlier that is
   within its trailing 8-event window -> REVERSION; ``undoes_prior_mutation``
   is set whenever the after-state was held earlier in the session.
6. Otherwise NONE.

Precedence: RESOLVES_EPISODE > PROGRESS > NEW_FACT > REVERSION >
IN_POLICY_RETRY/POLL > REDUNDANT > NONE.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import random
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

from ..fingerprint import FAMILY_CATEGORY, failure_fingerprint, intent_fingerprint, normalise_path, workspace_digest
from ..oracle_config import OracleConfig, load_oracle_config
from ..v03.rawop import RawOp, emit_finished, normalise_standard_drop
from ..schema import (
    EVENTS_PER_WINDOW,
    CanonicalEvent,
    LatentEffect,
    OperationCategory,
    OracleRecord,
    Outcome,
    Provenance,
)
from .labeller import label_session
from .latent import LatentStep, latent_trace
from .world import ASK_ACTIONS, Clue, Fault, World, YIELD_ACTIONS

__all__ = [
    "GENERATOR_VERSION",
    "PROVENANCE_SOURCE",
    "MIN_EVENTS",
    "MAX_EVENTS",
    "DROP_FIELDS",
    "COSMETIC_KINDS",
    "Change",
    "fix",
    "cosmetic",
    "competing",
    "wrong",
    "neutral",
    "undo",
    "CostProfile",
    "OpResult",
    "Session",
    "SessionLengthError",
]

GENERATOR_VERSION = "0.2.0"
PROVENANCE_SOURCE = f"simulator:{GENERATOR_VERSION}"
MIN_EVENTS = 12
MAX_EVENTS = 60

DROP_FIELDS = (
    "tokens", "input_tokens", "output_tokens", "digests", "digest_before", "digest_after",
    "failure", "validation_counts", "validation_suite",
)
COSMETIC_KINDS = ("format", "rename_local", "comment", "whitespace", "docstring")

_C = OperationCategory
BASE_TOKENS: Mapping[OperationCategory, tuple[float, float]] = {
    _C.INSPECT: (1800.0, 120.0),
    _C.MUTATE: (2600.0, 420.0),
    _C.VALIDATE: (1400.0, 90.0),
    _C.EXTERNAL: (700.0, 60.0),
    _C.CONTROL: (3000.0, 480.0),
}
BASE_SECONDS: Mapping[OperationCategory, float] = {
    _C.INSPECT: 1.5, _C.MUTATE: 4.0, _C.VALIDATE: 18.0, _C.EXTERNAL: 2.5, _C.CONTROL: 6.0,
}
_EPOCH = datetime(2026, 1, 1)


class SessionLengthError(ValueError):
    """A finished session is outside ``MIN_EVENTS..MAX_EVENTS``."""


# ---------------------------------------------------------------------------
# Changes (what an edit/write does to latent behaviour)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Change:
    kind: str  # fix | cosmetic | competing | wrong | neutral | undo
    fault: str | None = None
    label: str | None = None
    style: str | None = None
    regress: bool = False

    def describe(self) -> str:
        if self.kind == "fix":
            return f"fix:{self.fault}"
        if self.kind == "competing":
            return f"competing:{self.fault}:{self.label}"
        if self.kind == "cosmetic":
            return f"cosmetic:{self.style}"
        if self.kind == "wrong":
            return "wrong" + (":regress" if self.regress else "")
        return self.kind


def _changed(before: Mapping[str, str | None], after: Mapping[str, str | None]) -> list[dict]:
    """v0.3 raw capture: files whose text differs, sorted by path (None = absent)."""
    return [{"path": p, "before": before.get(p), "after": after.get(p)}
            for p in sorted(set(before) | set(after)) if before.get(p) != after.get(p)]


def _fid(fault: Fault | str) -> str:
    return fault.id if isinstance(fault, Fault) else str(fault)


def fix(fault: Fault | str) -> Change:
    """The fault's correct behavioural patch. On an already-fixed block it
    re-adds the existing guard: text changes, behaviour doesn't (REDUNDANT)."""
    return Change("fix", fault=_fid(fault))


def cosmetic(kind: str = "rename_local") -> Change:
    """Text-only change (REDUNDANT under the r2 reading). ``format`` is
    idempotent and file-wide; the other kinds always produce new text."""
    if kind not in COSMETIC_KINDS:
        raise ValueError(f"cosmetic kind must be one of {COSMETIC_KINDS}")
    return Change("cosmetic", style=kind)


def competing(fault: Fault | str, label: str) -> Change:
    """Competing alternative ``label`` (e.g. "A"/"B") at the fault's block:
    never fixes it, no progress; re-applying the current label is REDUNDANT."""
    return Change("competing", fault=_fid(fault), label=str(label))


def wrong(regress: bool = False) -> Change:
    """Novel behaviour change that doesn't help (NONE). ``regress=True`` also
    breaks the tests covering the block and lowers ``p``."""
    return Change("wrong", regress=regress)


def neutral() -> Change:
    """Novel behaviour change with no effect on tests (NONE)."""
    return Change("neutral")


def undo() -> Change:
    """Reverse patch of the most recent agent mutation of this file."""
    return Change("undo")


# ---------------------------------------------------------------------------
# Costs and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostProfile:
    """Session cost profile: tokens ~ lognormal(median * scale * category multiplier, sigma)."""

    scale: float = 1.0
    sigma: float = 0.45
    category: Mapping[str, float] = field(default_factory=lambda: {c.value: 1.0 for c in OperationCategory})

    @classmethod
    def draw(cls, rng: random.Random, *, scale: float | None = None) -> "CostProfile":
        s = scale if scale is not None else rng.lognormvariate(0.0, 0.4)
        sigma = rng.uniform(0.25, 0.7)
        cat = {c.value: rng.lognormvariate(0.0, 0.25) for c in OperationCategory}
        return cls(s, sigma, cat)


@dataclass(frozen=True)
class OpResult:
    """What one operation produced. ``failing_test`` / ``error_class`` are what
    the agent observes on a failing run; latent data stays in ``step``."""

    event: CanonicalEvent
    step: LatentStep
    failing_test: str | None = None
    error_class: str | None = None
    passed: int | None = None
    total: int | None = None

    @property
    def outcome(self) -> Outcome:
        return self.event.outcome

    @property
    def effect(self) -> LatentEffect:
        return self.step.effect

    @property
    def new_facts(self) -> int:
        return self.step.new_facts

    @property
    def ok(self) -> bool:
        return self.event.outcome is Outcome.SUCCESS

    @property
    def failed(self) -> bool:
        return self.event.outcome is Outcome.ERROR

    @property
    def not_ready(self) -> bool:
        return self.event.outcome is Outcome.NEUTRAL


@dataclass
class _Episode:
    kind: str  # retry | poll
    key: tuple
    rel_kind: str
    relevant: str
    causes: frozenset
    attempts: int = 0
    unrelated: int = 0
    repaired: bool = False


@dataclass
class _Candidate:
    kind: str
    key: tuple
    rel_kind: str
    relevant: str
    causes: frozenset


@dataclass
class _LogEntry:
    desc: str
    vsuite: str | None  # latent suite if the op produced a validation result
    dropped: tuple[str, ...]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    """One simulated agent session against a seeded world.

    Families call operations in order; each returns an :class:`OpResult`.
    ``cost_multiplier`` scales the tokens of every later operation (use it
    for rising cost); ops also take ``cost=`` (one-off multiplier) and
    ``drop=`` (fields from :data:`DROP_FIELDS` to drop on this event).
    """

    def __init__(
        self,
        rng: random.Random,
        session_id: str,
        cfg: OracleConfig | None = None,
        *,
        family: str,
        world: World | None = None,
        cost_profile: CostProfile | None = None,
        seed: int | None = None,
        **world_kwargs: Any,
    ) -> None:
        if world is not None and world_kwargs:
            raise TypeError("pass either a world or World.generate kwargs, not both")
        self.rng = rng
        self.cfg = cfg or load_oracle_config()
        self.family = family
        self.seed = seed
        self.session_id = session_id
        self.world = world if world is not None else World.generate(rng, **world_kwargs)
        self.cost = cost_profile if cost_profile is not None else CostProfile.draw(rng)
        self.cost_multiplier = 1.0
        self.meta: dict[str, Any] = {}
        self.agent_id = f"agent-{rng.getrandbits(32):08x}"
        self.task_id = f"task-{rng.getrandbits(40):010x}"
        self.trace_id = _sha(f"trace:{session_id}")[:32]
        self._ms = rng.randint(0, 300) * 86_400_000 + rng.randint(0, 86_399) * 1000
        self._events: list[CanonicalEvent] = []
        self._steps: list[LatentStep] = []
        self._log: list[_LogEntry] = []
        self._raw: list[RawOp] = []  # v0.3 side channel, aligned with _events (no latent data)
        self._finished_raw: list[RawOp] | None = None
        self._K: dict[str, int] = {}
        self._seen_content: set[tuple] = set()
        self._seen_obs: set[tuple] = set()
        self._last_validation: dict[tuple, str] = {}
        self._ext_ok: set[tuple] = set()
        self._last_poll: dict[str, str] = {}
        self._control: dict[tuple, int] = {}
        self._epoch = 0
        self._episode: _Episode | None = None
        self._candidate: _Candidate | None = None
        self._ws_held: set[str] = {self.world.ws_id()}
        self._stacks: dict[str, list[dict | None]] = {}
        self._initial: dict[str, dict | None] = {p: fm.snapshot() for p, fm in self.world.files.items()}
        self._p0 = self.world.progress()
        self._p = self._p0
        self._finished = False

    # -- introspection (latent; families may branch on it) ----------------

    @property
    def n(self) -> int:
        return len(self._events)

    @property
    def progress(self) -> float:
        return self._p

    @property
    def events(self) -> list[CanonicalEvent]:
        return list(self._events)

    @property
    def steps(self) -> list[LatentStep]:
        return list(self._steps)

    @property
    def raw_ops(self) -> list[RawOp]:
        """v0.3 raw operation records, aligned with :attr:`events` (after
        :meth:`finish`, with the returned events, i.e. after any truncation)."""
        return list(self._finished_raw if self._finished_raw is not None else self._raw)

    @property
    def episode(self) -> tuple[str, int] | None:
        """``(kind, attempts)`` of the open episode, else None."""
        ep = self._episode
        return None if ep is None else (ep.kind, ep.attempts)

    def room(self, k: int = 1, max_events: int = MAX_EVENTS) -> bool:
        """True if ``k`` more events fit under ``max_events``."""
        return self.n + k <= max_events

    def knows(self, fact: str) -> bool:
        return fact in self._K

    def unknown_clues(self, fault: Fault | str) -> list[Clue]:
        """Clues of the fault whose fact is not yet in ``K`` (first clue per fact)."""
        out, seen = [], set()
        for c in self.world.clues_for(_fid(fault)):
            if c.fact not in self._K and c.fact not in seen:
                seen.add(c.fact)
                out.append(c)
        return out

    def first_failure(self, suite: str, tests: Sequence[str] | None = None) -> tuple[str, str, str, str] | None:
        """Latent ``(test_id, error_class, fact, cause)`` of the first failing test, or None."""
        return self.world.run_suite(suite, tests)[2]

    # -- inspection -------------------------------------------------------

    def read(self, path: str, lines: tuple[int, int] | None = None, *, block: str | None = None, pad: int = 0,
             cost: float | None = None, drop: Sequence[str] = ()) -> OpResult:
        """``read_file``. ``block=`` reads the block's current line span (plus ``pad`` lines)."""
        path = normalise_path(path)
        w = self.world
        if path not in w.files:
            lo, hi = lines if lines else (None, None)
            return self._commit(
                "read_file", path, {"line_start": lo, "line_end": hi}, key=("read_file", path, lo, hi), rel_kind="text",
                outcome=Outcome.ERROR, failure=("read_file", "FileNotFoundError", path), desc=f"read {path} (missing)",
                cost=cost, drop=drop,
            )
        n_lines = w.line_count(path)
        if block is not None:
            b_lo, b_hi = w.spans(path)[block]
            lo, hi = max(1, b_lo - pad), min(n_lines, b_hi + pad)
        elif lines is not None:
            lo, hi = int(lines[0]), int(lines[1])
        else:
            lo = hi = None
        clues = w.read_clues(path, lo, hi)
        content = ("read", path, lo, hi, _sha(w.region(path, lo, hi)))
        span = (hi or n_lines) - (lo or 1) + 1
        return self._inspect(
            "read_file", path, {"line_start": lo, "line_end": hi}, ("read_file", path, lo, hi), clues, content,
            hint=0.4 + span / 60.0, desc=f"read {path}:{lo}-{hi}", cost=cost, drop=drop,
        )

    def search(self, scope: str, query: str, *, limit: int | None = None, cost: float | None = None,
               drop: Sequence[str] = ()) -> OpResult:
        scope = normalise_path(scope)
        q = " ".join(query.lower().split())
        w = self.world
        clues = w.search_clues(scope, q)
        content = ("search", scope, q, w.scope_digest(scope))
        args: dict[str, Any] = {"query": query}
        if limit is not None:
            args["limit"] = limit
        return self._inspect("search", scope, args, ("search", scope, q), clues, content, hint=0.8,
                             desc=f"search {scope} {q!r}", cost=cost, drop=drop)

    def list_dir(self, path: str = ".", *, cost: float | None = None, drop: Sequence[str] = ()) -> OpResult:
        path = normalise_path(path)
        listing = self.world.listing(path)
        if listing is None:
            return self._commit("list_dir", path, {}, key=("list_dir", path), rel_kind="text", outcome=Outcome.ERROR,
                                failure=("list_dir", "FileNotFoundError", path), desc=f"ls {path} (missing)",
                                cost=cost, drop=drop)
        return self._inspect("list_dir", path, {}, ("list_dir", path), [], ("list", path, tuple(listing)), hint=0.3,
                             desc=f"ls {path}", cost=cost, drop=drop)

    def inspect_clue(self, clue: Clue, **opts: Any) -> OpResult:
        """Run the inspection that reveals ``clue`` (read / search / lint / build / http GET)."""
        if clue.kind == "read":
            return self.read(clue.path, block=clue.block, pad=self.rng.randint(0, 4), **opts)
        if clue.kind == "search":
            return self.search(clue.scope or ".", clue.query, **opts)
        if clue.kind == "lint":
            return self.lint(clue.target, **opts)
        if clue.kind == "build":
            return self.build(clue.target, **opts)
        if clue.kind == "http":
            return self.http("GET", clue.host, clue.path_template, **opts)
        raise ValueError(f"cannot inspect clue kind {clue.kind!r}")

    def novel_decoy(self, kinds: Sequence[str] = ("read", "list", "search"), **opts: Any) -> OpResult:
        """A novel, fruitless inspection (effect NONE): an unread decoy file or
        region, an unlisted directory, or a search that matches no clue."""
        w = self.world
        cands: list[tuple[str, Any]] = []
        if "read" in kinds:
            for p in w.decoy_paths():
                key = ("read", p, None, None, _sha(w.region(p, None, None)))
                if key not in self._seen_content:
                    cands.append(("read", p))
        if "list" in kinds:
            for d in w.dirs():
                lst = w.listing(d)
                if ("list", d, tuple(lst)) not in self._seen_content:
                    cands.append(("list", d))
        if "search" in kinds:
            for q in w.vocab.decoy_queries:
                for sc in (".", "src", "docs"):
                    qn = " ".join(q.lower().split())
                    if ("search", sc, qn, w.scope_digest(sc)) not in self._seen_content and not w.search_clues(sc, qn):
                        cands.append(("search", (sc, q)))
        if not cands:  # fall back to a fresh region of a decoy file
            p = self.rng.choice(w.decoy_paths() or sorted(w.files))
            n_lines = w.line_count(p)
            for _ in range(50):
                lo = self.rng.randint(1, max(1, n_lines))
                hi = min(n_lines, lo + self.rng.randint(0, 20))
                if ("read", p, lo, hi, _sha(w.region(p, lo, hi))) not in self._seen_content and not w.read_clues(p, lo, hi):
                    return self.read(p, (lo, hi), **opts)
            raise RuntimeError("no novel decoy inspection left")
        kind, arg = cands[self.rng.randrange(len(cands))]
        if kind == "read":
            return self.read(arg, **opts)
        if kind == "list":
            return self.list_dir(arg, **opts)
        return self.search(arg[0], arg[1], **opts)

    # -- mutation ---------------------------------------------------------

    def edit(self, path: str, change: Change, *, block: str | None = None, cost: float | None = None,
             drop: Sequence[str] = ()) -> OpResult:
        """``edit_file`` with a unified-diff patch derived from the real text change."""
        return self._mutate("edit_file", normalise_path(path), change, block, cost, drop)

    def write(self, path: str, change: Change | None = None, *, block: str | None = None, cost: float | None = None,
              drop: Sequence[str] = ()) -> OpResult:
        """``write_file``: apply ``change`` as a whole-file rewrite, or create a
        new file (``change=None``; a novel behaviour, effect NONE)."""
        return self._mutate("write_file", normalise_path(path), change, block, cost, drop)

    def revert(self, paths: str | Sequence[str], to: str = "last", *, cost: float | None = None,
               drop: Sequence[str] = ()) -> OpResult:
        """``revert``: ``to="last"`` undoes the most recent mutation of each path,
        ``to="initial"`` restores the session-start content."""
        if isinstance(paths, str):
            paths = [paths]
        paths = sorted({normalise_path(p) for p in paths})
        if to not in ("last", "initial"):
            raise ValueError("to must be 'last' or 'initial'")
        w = self.world
        before = self._mut_snapshot()
        pre_texts = {p: (w.text(p) if p in w.files else None) for p in paths}  # v0.3 raw capture (no rng)
        for p in paths:
            if to == "last":
                stack = self._stacks.get(p)
                if not stack:
                    raise ValueError(f"nothing to revert for {p}")
                self._restore(p, stack.pop())
            else:
                self._push(p)
                self._restore(p, self._initial.get(p))
        target = {p: w.text(p) for p in paths if p in w.files}
        revision = workspace_digest(target) if target else _sha("deleted:" + ",".join(paths))
        post_texts = {p: (w.text(p) if p in w.files else None) for p in paths}
        return self._finish_mutation("revert", list(paths), {"revision": revision}, ("revert", tuple(paths), revision),
                                     before, f"revert {','.join(paths)} to {to}", cost, drop,
                                     changed=_changed(pre_texts, post_texts))

    def _push(self, path: str) -> None:
        fm = self.world.files.get(path)
        self._stacks.setdefault(path, []).append(None if fm is None else fm.snapshot())

    def _restore(self, path: str, snap: dict | None) -> None:
        w = self.world
        if snap is None:
            if path in w.files:
                w.remove_file(path)
            return
        if path not in w.files:
            raise ValueError(f"cannot restore missing file {path}")
        w.set_states(path, snap)

    def _mut_snapshot(self) -> tuple[str, str, set[str], dict[str, str]]:
        w = self.world
        return w.ws_id(), w.behaviour_key(), w.causes(), {}

    def _mutate(self, family: str, path: str, change: Change | None, block: str | None, cost: float | None,
                drop: Sequence[str]) -> OpResult:
        w = self.world
        exists = path in w.files
        if not exists and (family == "edit_file" or (change is not None and change.kind != "undo")):
            patch = f"--- a/{path}\n+++ b/{path}\n@@ missing @@"
            args = {"patch": patch} if family == "edit_file" else {"content": patch}
            return self._commit(family, path, args, key=(family, path, _sha(patch)), rel_kind="text", outcome=Outcome.ERROR,
                                failure=(family, "FileNotFoundError", path), desc=f"{family} {path} (missing)",
                                cost=cost, drop=drop)
        before = self._mut_snapshot()
        old_text = w.text(path) if exists else ""
        if change is None:
            if exists:
                raise ValueError(f"write to existing {path} needs a change")
            vid = w.new_variant("new")
            self._stacks.setdefault(path, []).append(None)
            w.add_file(path, [block or "main"], vid)
            desc = "create"
        elif change.kind == "undo":
            stack = self._stacks.get(path)
            if not stack:
                raise ValueError(f"nothing to undo for {path}")
            self._restore(path, stack.pop())
            desc = "undo"
        else:
            new_states = self._plan_change(path, change, block)
            if new_states != w.files[path].states:
                self._push(path)
                w.set_states(path, new_states)
            desc = change.describe()
        new_exists = path in w.files
        new_text = w.text(path) if new_exists else ""
        if family == "edit_file":
            patch = "\n".join(difflib.unified_diff(old_text.split("\n"), new_text.split("\n"), f"a/{path}", f"b/{path}",
                                                   lineterm="", n=0)) or f"--- a/{path}\n+++ b/{path}\n@@ no change @@"
            args: dict[str, Any] = {"patch": patch}
            key = (family, path, _sha(patch))
        else:
            args = {"content": new_text}
            key = (family, path, _sha(new_text))
        return self._finish_mutation(family, path, args, key, before, f"{family} {path} {desc}", cost, drop,
                                     hint=0.6 + abs(len(new_text) - len(old_text)) / 2000.0,
                                     changed=_changed({path: old_text if exists else None},
                                                      {path: new_text if new_exists else None}))

    def _plan_change(self, path: str, change: Change, block: str | None) -> dict:
        w = self.world
        fm = w.files[path]
        states = dict(fm.states)
        if change.kind in ("fix", "competing"):
            f = w.faults[change.fault]
            if f.path != path:
                raise ValueError(f"fault {f.id} is in {f.path}, not {path}")
            b = f.block
            st = states[b]
            target = "fix" if change.kind == "fix" else w.competing_variant(f.id, change.label)
            states[b] = replace(st, style=st.style + 1) if st.behaviour == target else replace(st, behaviour=target)
            return states
        if change.kind == "cosmetic":
            if change.style == "format":
                return {b: replace(s, formatted=True) for b, s in states.items()}
            b = block or fm.blocks[0]
            states[b] = replace(states[b], style=states[b].style + 1)
            return states
        if change.kind in ("wrong", "neutral"):
            if block is None:
                open_faults = [b for b in fm.blocks if w.fault_at(path, b) and not w.is_fixed(w.fault_at(path, b))]
                block = open_faults[0] if open_faults else fm.blocks[0]
            vid = w.new_variant(change.kind, regress=change.regress and change.kind == "wrong")
            states[block] = replace(states[block], behaviour=vid)
            return states
        raise ValueError(f"unknown change kind {change.kind!r}")

    def _finish_mutation(self, family: str, target: Any, args: dict, key: tuple, before: tuple, desc: str,
                         cost: float | None, drop: Sequence[str], hint: float = 1.0, *,
                         changed: list[dict] | None = None) -> OpResult:
        w = self.world
        ws_b, beh_b, causes_b, _ = before
        beh_changed = w.behaviour_key() != beh_b
        repaired = frozenset(causes_b - w.causes())
        return self._commit(
            family, target, args, key=key, rel_kind="text", outcome=Outcome.SUCCESS, ws_before=ws_b,
            redundant=not beh_changed, beh_changed=beh_changed, repaired=repaired, desc=desc, cost=cost, drop=drop,
            hint=hint, changed=changed,
        )

    # -- validation -------------------------------------------------------

    def run_tests(self, suite: str, tests: Sequence[str] | None = None, *, cost: float | None = None,
                  drop: Sequence[str] = ()) -> OpResult:
        w = self.world
        passed, total, first, causes = w.run_suite(suite, tests)
        sel = None if tests is None else tuple(sorted(set(tests)))
        vkey = ("run_tests", suite, sel)
        beh = w.behaviour_key()
        redundant = self._last_validation.get(vkey) == beh
        self._last_validation[vkey] = beh
        args = {"tests": list(sel)} if sel is not None else {}
        facts = [first[2]] if first else []
        res = self._commit(
            "run_tests", suite, args, key=vkey, rel_kind="beh", outcome=Outcome.ERROR if first else Outcome.SUCCESS,
            failure=("run_tests", first[1], first[0]) if first else None, causes=frozenset(causes), facts=facts,
            redundant=redundant, validation=(suite, passed, total), obs=("val", vkey, passed, total, first and first[0]),
            desc=f"test {suite}{'' if sel is None else f' [{len(sel)}]'} -> {passed}/{total}",
            hint=0.6 + total / 40.0 + (0.8 if first else 0.0), cost=cost, drop=drop,
            extra={"failing_test": first[0]} if first else None,
        )
        return replace(res, failing_test=first[0] if first else None, error_class=first[1] if first else None,
                       passed=passed, total=total)

    def build(self, target: str, *, cost: float | None = None, drop: Sequence[str] = ()) -> OpResult:
        return self._tool("build", target, cost, drop)

    def lint(self, target: str, *, cost: float | None = None, drop: Sequence[str] = ()) -> OpResult:
        return self._tool("lint", target, cost, drop)

    def _tool(self, family: str, target: str, cost: float | None, drop: Sequence[str]) -> OpResult:
        w = self.world
        failing = [c for c in w.tool_clues(family, target) if not w.is_fixed(c.fault)]
        vkey = (family, target)
        beh = w.behaviour_key()
        redundant = self._last_validation.get(vkey) == beh
        self._last_validation[vkey] = beh
        failure = None
        err = None
        if failing:
            pool = w.lang.lint_errors if family == "lint" else w.lang.build_errors
            err = pool[int(_sha(failing[0].fault + family)[:8], 16) % len(pool)]
            failure = (family, err, failing[0].path)
        res = self._commit(
            family, target, {}, key=vkey, rel_kind="beh", outcome=Outcome.ERROR if failing else Outcome.SUCCESS,
            failure=failure, causes=frozenset(c.fault for c in failing), facts=[c.fact for c in failing],
            redundant=redundant, obs=(family, target, tuple(c.fact for c in failing)),
            desc=f"{family} {target} -> {'fail' if failing else 'ok'}", hint=0.7, cost=cost, drop=drop,
        )
        return replace(res, error_class=err)

    # -- external ---------------------------------------------------------

    def http(self, method: str, host: str, path_template: str, *, body: str | None = None, cost: float | None = None,
             drop: Sequence[str] = ()) -> OpResult:
        w = self.world
        m = method.upper()
        svc = w.service(host)
        svc.calls += 1
        if svc.mode == "ok":
            up = True
        elif svc.mode == "transient":
            up = svc.calls > svc.fail_attempts
        elif svc.mode == "needs_fix":
            up = w.is_fixed(svc.requires)
        else:
            up = False
        key = ("http_call", m, svc.host, path_template, body if m in ("POST", "PATCH") else None)
        target = {"method": m, "host": svc.host, "path_template": path_template}
        args = {"body": body} if body is not None else {}
        if not up:
            causes = frozenset({svc.requires}) if svc.mode == "needs_fix" else frozenset({f"svc:{svc.host}"})
            res = self._commit(
                "http_call", target, args, key=key, rel_kind=f"ext:{svc.host}", outcome=Outcome.ERROR,
                failure=("http_call", svc.error_class, f"{svc.host}{path_template}"), causes=causes,
                obs=("http", key, "err"), desc=f"http {m} {svc.host}{path_template} -> {svc.error_class}",
                hint=0.6, cost=cost, drop=drop, seconds=6.0,
            )
            return replace(res, error_class=svc.error_class)
        clues = w.http_clues(svc.host, path_template)
        facts = [c.fact for c in clues]
        new = [f for f in dict.fromkeys(facts) if f not in self._K]
        redundant = not new and key in self._ext_ok
        self._ext_ok.add(key)
        return self._commit(
            "http_call", target, args, key=key, rel_kind=f"ext:{svc.host}", outcome=Outcome.SUCCESS, facts=facts,
            redundant=redundant, obs=("http", key, "ok"), desc=f"http {m} {svc.host}{path_template} -> ok",
            hint=1.0 + 0.5 * len(clues), cost=cost, drop=drop,
        )

    def poll(self, resource: str, *, backoff: int | None = None, cost: float | None = None,
             drop: Sequence[str] = ()) -> OpResult:
        """``poll_status``: NEUTRAL while not ready, SUCCESS once ready."""
        r = self.world.resource(resource)
        r.polls += 1
        ready = r.ready_after is not None and r.polls > r.ready_after
        status = "ready" if ready else "pending"
        redundant = self._last_poll.get(resource) == status
        self._last_poll[resource] = status
        args = {"backoff": backoff} if backoff is not None else {}
        return self._commit(
            "poll_status", resource, args, key=("poll_status", resource), rel_kind=f"res:{resource}",
            outcome=Outcome.SUCCESS if ready else Outcome.NEUTRAL, poll=True, redundant=redundant,
            obs=("poll", resource, status), desc=f"poll {resource} -> {status}", hint=0.5, cost=cost, drop=drop,
            seconds=1.0 + (backoff or 0), extra={"status": status},
        )

    # -- control ----------------------------------------------------------

    def plan(self, action_type: str, **opts: Any) -> OpResult:
        return self._control_op("plan", action_type, **opts)

    def ask(self, action_type: str = ASK_ACTIONS[0], **opts: Any) -> OpResult:
        return self._control_op("ask", action_type, **opts)

    def yield_(self, action_type: str = YIELD_ACTIONS[0], **opts: Any) -> OpResult:
        return self._control_op("yield", action_type, **opts)

    def _control_op(self, family: str, action_type: str, *, cost: float | None = None, drop: Sequence[str] = ()) -> OpResult:
        key = (family, action_type.lower())
        redundant = self._control.get(key) == self._epoch
        self._control[key] = self._epoch
        return self._commit(family, None, {"action_type": action_type}, key=key, rel_kind="text", outcome=Outcome.SUCCESS,
                            redundant=redundant, desc=f"{family} {action_type}", cost=cost, drop=drop)

    # -- shared inspection core ------------------------------------------

    def _inspect(self, family: str, target: Any, args: dict, key: tuple, clues: Iterable[Clue], content: tuple,
                 *, hint: float, desc: str, cost: float | None, drop: Sequence[str]) -> OpResult:
        facts = list(dict.fromkeys(c.fact for c in clues))
        new = [f for f in facts if f not in self._K]
        redundant = not new and (content in self._seen_content or bool(facts))
        self._seen_content.add(content)
        return self._commit(family, target, args, key=key, rel_kind="text", outcome=Outcome.SUCCESS, facts=facts,
                            redundant=redundant, obs=content, desc=desc, hint=hint, cost=cost, drop=drop)

    # -- the effect engine -----------------------------------------------

    def _relevant(self, rel_kind: str) -> str:
        if rel_kind == "beh":
            return self.world.behaviour_key()
        if rel_kind == "text":
            return self.world.ws_id()
        return rel_kind

    def _commit(
        self,
        family: str,
        target: Any,
        args: dict,
        *,
        key: tuple,
        rel_kind: str,
        outcome: Outcome,
        ws_before: str | None = None,
        failure: tuple[str, str, str] | None = None,
        causes: frozenset = frozenset(),
        facts: Sequence[str] = (),
        redundant: bool = False,
        beh_changed: bool = False,
        repaired: frozenset = frozenset(),
        poll: bool = False,
        validation: tuple[str, int, int] | None = None,
        obs: tuple | None = None,
        desc: str = "",
        hint: float = 1.0,
        cost: float | None = None,
        drop: Sequence[str] = (),
        seconds: float | None = None,
        changed: list[dict] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> OpResult:
        if self._finished:
            raise RuntimeError("session already finished")
        drop = tuple(drop)
        for d in drop:
            if d not in DROP_FIELDS:
                raise ValueError(f"unknown drop field {d!r}; expected one of {DROP_FIELDS}")
        if "failure" in drop and outcome is not Outcome.ERROR:
            raise ValueError("can only drop the failure fingerprint of an ERROR event")
        if any(d.startswith("validation") for d in drop) and validation is None:
            raise ValueError("can only drop validation fields of a validation event")

        w = self.world
        idx = len(self._events)
        ws_after = w.ws_id()
        ws_before = ws_after if ws_before is None else ws_before
        mutation = ws_before != ws_after
        p_before, p_after = self._p, w.progress()
        new_facts = [f for f in dict.fromkeys(facts) if f not in self._K]
        for f in new_facts:
            self._K[f] = idx
        relevant = self._relevant(rel_kind)
        failing = outcome is Outcome.ERROR or (poll and outcome is Outcome.NEUTRAL)

        # Rule 1: episodes.
        attempt = resolved = False
        ep, cand = self._episode, self._candidate
        if ep is not None and key == ep.key and (relevant == ep.relevant or ep.repaired):
            attempt = True
        elif cand is not None and key == cand.key and relevant == cand.relevant:
            ep = _Episode(cand.kind, cand.key, cand.rel_kind, cand.relevant, cand.causes)
            attempt = True
        if attempt:
            ep.attempts += 1
            ep.unrelated = 0
            ep.repaired = False
            ep.relevant = relevant
            if failing:
                self._episode = ep
            else:
                resolved = True
                self._episode = None
        elif ep is not None:
            if repaired & ep.causes:
                ep.repaired = True
            elif self._relevant(ep.rel_kind) != ep.relevant:
                self._episode = None
            else:
                ep.unrelated += 1
                if ep.unrelated >= 2:
                    self._episode = None
        self._candidate = None
        if failing and not (attempt and self._episode is not None):
            self._candidate = _Candidate("poll" if poll else "retry", key, rel_kind, relevant, frozenset(causes))
        ep_after = self._episode
        limit = self.cfg.r_max if (ep is not None and ep.kind == "retry") else self.cfg.p_max
        in_policy = attempt and not resolved and ep is not None and ep.attempts <= limit

        # Rule 5: reversion (window seen-set as in labeller.window_stats).
        undoes = mutation and ws_after in self._ws_held
        window_seen: set[str] = set()
        for s in self._steps[-(EVENTS_PER_WINDOW - 1):]:
            window_seen.add(s.ws_before)
            window_seen.add(s.ws_after)
        if not self._steps:
            window_seen.add(ws_before)
        reversion = undoes and ws_after in window_seen

        # Precedence.
        if resolved:
            effect = LatentEffect.RESOLVES_EPISODE
        elif p_after > p_before:
            effect = LatentEffect.PROGRESS
        elif new_facts:
            effect = LatentEffect.NEW_FACT
        elif reversion:
            effect = LatentEffect.REVERSION
        elif in_policy:
            effect = LatentEffect.IN_POLICY_POLL if ep.kind == "poll" else LatentEffect.IN_POLICY_RETRY
        elif redundant or attempt:
            effect = LatentEffect.REDUNDANT
        else:
            effect = LatentEffect.NONE

        # Information epoch (for control-op redundancy).
        informative = bool(new_facts) or p_after != p_before or beh_changed or resolved
        if obs is not None and obs not in self._seen_obs:
            self._seen_obs.add(obs)
            informative = True
        if informative:
            self._epoch += 1

        self._ws_held.add(ws_before)
        self._ws_held.add(ws_after)
        self._p = p_after

        step = LatentStep(
            sequence_index=idx, effect=effect, progress_after=p_after, new_facts=len(new_facts),
            ws_before=ws_before, ws_after=ws_after, agent_mutation=mutation, undoes_prior_mutation=undoes,
            open_episode_kind=ep_after.kind if ep_after else None,
            open_episode_attempts=ep_after.attempts if ep_after else 0,
            resolved_episode_here=resolved, dropped=bool(drop),
        )
        event = self._build_event(idx, family, target, args, outcome, failure, ws_before, ws_after, validation,
                                  hint, cost, drop, seconds)
        self._events.append(event)
        self._steps.append(step)
        self._log.append(_LogEntry(desc, validation[0] if validation else None, drop))
        # v0.3 side channel: the concrete operation as executed and as emitted.
        # Built from values computed above; no world or rng access.
        err = failure is not None and outcome is Outcome.ERROR
        self._raw.append(RawOp.make(
            family=family, target=target, args=args, outcome=outcome,
            error_class=failure[1] if err else None, error_location=failure[2] if err else None,
            validation=validation, changed=changed, extra=extra,
            input_tokens=event.input_tokens, output_tokens=event.output_tokens, timestamp=event.timestamp,
            dropped=normalise_standard_drop(drop),
        ))
        return OpResult(event, step)

    def _build_event(self, idx: int, family: str, target: Any, args: dict, outcome: Outcome,
                     failure: tuple[str, str, str] | None, ws_before: str, ws_after: str,
                     validation: tuple[str, int, int] | None, hint: float, cost: float | None,
                     drop: tuple[str, ...], seconds: float | None) -> CanonicalEvent:
        cat = FAMILY_CATEGORY[family]
        mult = self.cost.scale * self.cost.category[cat.value] * self.cost_multiplier * (cost if cost is not None else 1.0)
        if outcome is Outcome.ERROR:
            mult *= 1.3
        mi, mo = BASE_TOKENS[cat]
        tin = max(1, min(400_000, int(round(mi * mult * hint * self.rng.lognormvariate(0.0, self.cost.sigma)))))
        tout = max(1, min(400_000, int(round(mo * mult * self.rng.lognormvariate(0.0, self.cost.sigma)))))
        base = seconds if seconds is not None else BASE_SECONDS[cat]
        self._ms += max(1, int(base * 1000 * self.rng.lognormvariate(0.0, 0.6)))
        t = _EPOCH + timedelta(milliseconds=self._ms)
        ts = f"{t:%Y-%m-%dT%H:%M:%S}.{t.microsecond // 1000:03d}Z"
        fp = failure_fingerprint(*failure) if (failure and outcome is Outcome.ERROR) else None
        d = set(drop)
        kw: dict[str, Any] = dict(
            session_id=self.session_id, agent_id=self.agent_id, task_id=self.task_id,
            event_id=f"{self.session_id}-e{idx:03d}", sequence_index=idx, timestamp=ts,
            operation_family=family, operation_category=cat, outcome=outcome,
            intent_fingerprint=intent_fingerprint(family, target, args),
            provenance=Provenance(PROVENANCE_SOURCE, True, (), self.trace_id, (f"{self.trace_id[:16]}{idx:04x}",)),
            failure_fingerprint=None if "failure" in d else fp,
            workspace_state_before=None if d & {"digests", "digest_before"} else ws_before,
            workspace_state_after=None if d & {"digests", "digest_after"} else ws_after,
            input_tokens=None if d & {"tokens", "input_tokens"} else tin,
            output_tokens=None if d & {"tokens", "output_tokens"} else tout,
        )
        if validation is not None:
            suite, passed, total = validation
            kw.update(
                validation_produced=True,
                validation_suite_id=None if "validation_suite" in d else suite,
                validation_passed=None if "validation_counts" in d else passed,
                validation_total=None if "validation_counts" in d else total,
            )
        return CanonicalEvent(**kw)

    # -- branching and finishing -----------------------------------------

    def fork(self, session_id: str) -> "Session":
        """Deep copy (world, rng state, history, episodes) under a new session id.

        The emitted prefix is re-stamped with the new ``session_id``/``event_id``;
        everything else, including the latent state and future random draws, is
        identical until the two sessions diverge. The v0.3 raw ops are copied
        unchanged by the deepcopy: they carry no session identity.
        """
        new = copy.deepcopy(self)
        new.session_id = session_id
        new.trace_id = _sha(f"trace:{session_id}")[:32]
        new._events = [
            replace(e, session_id=session_id, event_id=f"{session_id}-e{e.sequence_index:03d}",
                    provenance=replace(e.provenance, trace_id=new.trace_id,
                                       span_ids=(f"{new.trace_id[:16]}{e.sequence_index:04x}",)))
            for e in new._events
        ]
        return new

    def finish(self, *, truncate_before: int = 0, min_events: int = MIN_EVENTS,
               max_events: int = MAX_EVENTS) -> tuple[list[CanonicalEvent], OracleRecord]:
        """Seal the session.

        ``truncate_before=m`` drops the first ``m`` events from the observed
        stream (history truncation): the remaining events keep their
        ``sequence_index`` and get ``provenance.history_complete=False``.
        Validation events whose baseline is thereby unobserved (as §5.2 / the
        gate define it) are marked ``dropped`` in the latent trace, as are those
        whose baseline is a validation event with dropped suite or counts.
        ``event_effects`` is aligned with the returned events list.
        """
        if self._finished:
            raise RuntimeError("session already finished")
        m = int(truncate_before)
        if not 0 <= m < len(self._events):
            raise ValueError("truncate_before out of range")
        events = self._events[m:]
        steps = self._steps[m:]
        log = self._log[m:]
        raw = self._raw[m:]
        if m:
            events = [replace(e, provenance=replace(e.provenance, history_complete=False)) for e in events]
            p0 = self._steps[m - 1].progress_after
        else:
            p0 = self._p0
        if not min_events <= len(events) <= max_events:
            raise SessionLengthError(f"{self.family}: session has {len(events)} events, outside {min_events}..{max_events}")

        deliberate = [i for i, e in enumerate(log) if e.dropped]
        propagated = []
        for k, entry in enumerate(log):
            if entry.vsuite is None or entry.dropped:
                continue
            unobserved = m > 0
            for j in range(k - 1, -1, -1):
                prev = log[j]
                if prev.vsuite is None:
                    continue
                if "validation_suite" in prev.dropped:
                    unobserved = True
                    break
                if prev.vsuite == entry.vsuite:
                    unobserved = "validation_counts" in prev.dropped
                    break
            if unobserved:
                propagated.append(k)
        dropped = set(deliberate) | set(propagated)
        steps = [replace(s, dropped=True) if i in dropped else s for i, s in enumerate(steps)]
        labels = label_session(steps, p0, self.cfg)
        trace = latent_trace(steps, p0)
        trace.update(
            world=self.world.summary(),
            meta=dict(self.meta),
            ops=[e.desc for e in log],
            drops={
                "deliberate": [steps[i].sequence_index for i in deliberate],
                "propagated": [steps[i].sequence_index for i in propagated],
                "fields": [[steps[i].sequence_index, list(log[i].dropped)] for i in deliberate],
                "truncated_before": m,
            },
        )
        record = OracleRecord(
            session_id=self.session_id,
            generator_version=GENERATOR_VERSION,
            seed=int(self.seed) if self.seed is not None else 0,
            scenario_family=self.family,
            event_effects=tuple(s.effect for s in steps),
            window_labels=labels,
            dropped_observations=tuple(steps[i].sequence_index for i in sorted(dropped)),
            latent_trace=trace,
        )
        self._finished = True
        self._finished_raw = raw
        emit_finished(self.session_id, events, raw)
        return events, record
