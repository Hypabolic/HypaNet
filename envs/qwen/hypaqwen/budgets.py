"""Labelled budgets of the probe and LoRA arms (PREREGISTRATION §4, §5, §7.2, §7.3), read from ``BUDGETS.json``.

The file is written harness-side in the frozen v0.3 environment (which holds the v0.3 rules:
``baselines.phase_c_v03.budget_subset`` and ``es_session_ids``, ``v03.scoring.disclosure_selection``)
and read here, because ``envs/qwen`` never imports ``agent_loop_lab``. Format ``v04-budgets/1``::

    {"format": "v04-budgets/1",
     "budgets": {"32": [ids], "128": [ids], "512": [ids], "2048": [ids], "all": [ids]},  # v0.3 order
     "es_val": [500 val ids],                     # the v0.3 Phase C ES set, in ES-key order
     "disclosed_200w": [{"session_id": id, "row_pos": int}, ... 200],   # V001..V200, key order
     "wspf_holdout": [500 dev ids],               # §7.3, in holdout-key order
     "sha256": {name: hex}}

``sha256`` holds, for every list, the writer's convention (``agent_loop_lab.v04.budgets.canonical_sha256``):
the sha256 of the list's canonical JSON as it stands in the file (``sort_keys=True``, ``","``/``":"``
separators, ``ensure_ascii=False``, UTF-8), so ``disclosed_200w`` hashes as its list of
``{"row_pos", "session_id"}`` objects. (The writer separately checks each list against v0.3's
newline-joined ``ids_sha256`` training logs; that hash is not stored here.) Names: the five budget
names, ``es_val``, ``disclosed_200w``, ``wspf_holdout``; every one is required.

:func:`load` checks everything that can be checked without labels or other splits: the orders are
recomputed from their hash keys (``all`` is ascending ``sha256(BUDGET_KEY)``, each budget its first
*k*; ``es_val`` ascending ``sha256(ES_KEY)``; the holdout is exactly the first 500 of ``all`` outside
the 2,048 subset by ``sha256(HOLDOUT_KEY)``), sizes, duplicates and the hashes. With the split's ids
or labels in hand, :func:`check_dev_ids`, :func:`check_val_ids` and :func:`check_disclosure` verify
the rest (the dev population, the ES rule with disclosed sessions removed first, and the v0.3
Phase B disclosure rule re-run on the val labels; only the disclosure path may read val labels).

Budget names: ``"32"``, ``"128"``, ``"512"``, ``"2048"``, ``"all"`` (dev sessions) and ``"200w"``
(the 200 disclosed val windows). Seeds (§1): LoRA 0,1,2 / 0,1,2 / 0,1,2 / 0 / 0 and 0,1,2 for 200w;
the probe is deterministic and has no seed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import V04_DATA

__all__ = [
    "FORMAT", "PATH", "SESSION_BUDGETS", "BUDGET_NAMES", "BUDGET_SIZES", "LORA_SEEDS", "BUDGET_KEY", "ES_KEY",
    "HOLDOUT_KEY", "DISCLOSURE_KEY", "ES_SESSIONS", "HOLDOUT_SESSIONS", "DISCLOSED_WINDOWS", "PER_CLASS",
    "BudgetError", "Window", "Budgets", "canonical_sha256", "budget_key", "es_key", "holdout_key",
    "disclosure_key", "parse", "load", "check_dev_ids", "check_val_ids", "check_disclosure", "disclosure_rows",
]

FORMAT = "v04-budgets/1"
PATH = V04_DATA / "BUDGETS.json"
SESSION_BUDGETS = ("32", "128", "512", "2048", "all")
BUDGET_NAMES = (*SESSION_BUDGETS, "200w")
BUDGET_SIZES = {"32": 32, "128": 128, "512": 512, "2048": 2048}
LORA_SEEDS: dict[str, tuple[int, ...]] = {"32": (0, 1, 2), "128": (0, 1, 2), "512": (0, 1, 2), "2048": (0,),
                                          "all": (0,), "200w": (0, 1, 2)}  # PREREGISTRATION §1
BUDGET_KEY = "20261105:v03-phase-c-budget:{session_id}"  # v0.3 phase_c_v03.BUDGET_KEY
ES_KEY = "20261105:v03-phase-c-es:{session_id}"  # v0.3 phase_c_v03.ES_KEY
HOLDOUT_KEY = "20261105:v04-wsp-f-holdout:{session_id}"  # PREREGISTRATION §7.3
DISCLOSURE_KEY = "20261105:v03-phase-b-disclosure:{session_id}:{endpoint}"  # v0.3 scoring.disclosure_key
ES_SESSIONS = 500
HOLDOUT_SESSIONS = 500
DISCLOSED_WINDOWS = 200
PER_CLASS = 50
N_CLASSES = 4
_SHA_NAMES = (*SESSION_BUDGETS, "es_val", "disclosed_200w", "wspf_holdout")


class BudgetError(ValueError):
    pass


@dataclass(frozen=True)
class Window:
    session_id: str
    row_pos: int


@dataclass(frozen=True)
class Budgets:
    budgets: Mapping[str, tuple[str, ...]]
    es_val: tuple[str, ...]
    disclosed_200w: tuple[Window, ...]
    wspf_holdout: tuple[str, ...]
    sha256: Mapping[str, str]
    file_sha256: str | None = None

    def sessions(self, budget: str) -> tuple[str, ...]:
        """Dev session ids of a session budget, in v0.3 order (``200w`` has windows, not sessions)."""
        if budget not in SESSION_BUDGETS:
            raise BudgetError(f"{budget!r} is not a session budget {SESSION_BUDGETS}")
        return self.budgets[budget]

    def size(self, budget: str) -> int:
        """``b`` of the epoch cap (§5): budget sessions, or 200 for ``200w``."""
        return len(self.disclosed_200w) if budget == "200w" else len(self.sessions(budget))

    @property
    def disclosed_sessions(self) -> frozenset[str]:
        return frozenset(w.session_id for w in self.disclosed_200w)


def _key(template: str, **kw) -> str:
    return hashlib.sha256(template.format(**kw).encode("utf-8")).hexdigest()


def budget_key(session_id: str) -> str:
    return _key(BUDGET_KEY, session_id=session_id)


def es_key(session_id: str) -> str:
    return _key(ES_KEY, session_id=session_id)


def holdout_key(session_id: str) -> str:
    return _key(HOLDOUT_KEY, session_id=session_id)


def disclosure_key(session_id: str, endpoint: int) -> str:
    return _key(DISCLOSURE_KEY, session_id=session_id, endpoint=int(endpoint))


def canonical_sha256(obj) -> str:
    """sha256 of ``obj``'s canonical JSON: the writer's ``agent_loop_lab.v04.budgets.canonical_sha256``."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def _ids(value, what: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(s, str) and s for s in value):
        raise BudgetError(f"{what}: expected a list of non-empty session id strings")
    if len(set(value)) != len(value):
        raise BudgetError(f"{what}: duplicate session ids")
    return tuple(value)


def _windows(value) -> tuple[Window, ...]:
    if not isinstance(value, list):
        raise BudgetError("disclosed_200w: expected a list")
    out = []
    for i, w in enumerate(value):
        if not isinstance(w, dict) or set(w) != {"session_id", "row_pos"}:
            raise BudgetError(f"disclosed_200w[{i}]: expected {{'session_id', 'row_pos'}}")
        sid, pos = w["session_id"], w["row_pos"]
        if not isinstance(sid, str) or not sid or not isinstance(pos, int) or isinstance(pos, bool) or pos < 0:
            raise BudgetError(f"disclosed_200w[{i}]: bad session_id or row_pos")
        out.append(Window(sid, pos))
    if len(out) != DISCLOSED_WINDOWS:
        raise BudgetError(f"disclosed_200w: {len(out)} windows, need {DISCLOSED_WINDOWS}")
    if len({w.session_id for w in out}) != len(out):
        raise BudgetError("disclosed_200w: more than one window per session")
    return tuple(out)


def parse(doc: Mapping, *, file_sha256: str | None = None) -> Budgets:
    """Validate a decoded ``BUDGETS.json`` (see the module docstring) and return it."""
    if not isinstance(doc, Mapping) or doc.get("format") != FORMAT:
        raise BudgetError(f"format is {doc.get('format') if isinstance(doc, Mapping) else None!r}, not {FORMAT!r}")
    want = {"format", "budgets", "es_val", "disclosed_200w", "wspf_holdout", "sha256"}
    if set(doc) != want:
        raise BudgetError(f"keys {sorted(doc)} != {sorted(want)}")
    raw = doc["budgets"]
    if not isinstance(raw, Mapping) or set(raw) != set(SESSION_BUDGETS):
        raise BudgetError(f"budgets: keys must be exactly {SESSION_BUDGETS}")
    budgets = {name: _ids(raw[name], f"budgets[{name!r}]") for name in SESSION_BUDGETS}
    full = budgets["all"]
    if list(full) != sorted(full, key=budget_key):
        raise BudgetError("budgets['all'] is not in ascending sha256(BUDGET_KEY) order")
    for name, k in BUDGET_SIZES.items():
        if len(full) < k:
            raise BudgetError(f"budgets['all'] has {len(full)} sessions, fewer than {k}")
        if budgets[name] != full[:k]:
            raise BudgetError(f"budgets[{name!r}] is not the first {k} of budgets['all']")
    es_val = _ids(doc["es_val"], "es_val")
    if len(es_val) != ES_SESSIONS:
        raise BudgetError(f"es_val: {len(es_val)} sessions, need {ES_SESSIONS}")
    if list(es_val) != sorted(es_val, key=es_key):
        raise BudgetError("es_val is not in ascending sha256(ES_KEY) order")
    disclosed = _windows(doc["disclosed_200w"])
    if {w.session_id for w in disclosed} & set(es_val):
        raise BudgetError("es_val contains a disclosed session (they are removed first)")
    holdout = _ids(doc["wspf_holdout"], "wspf_holdout")
    outside = [s for s in full if s not in set(budgets["2048"])]
    if list(holdout) != sorted(outside, key=holdout_key)[:HOLDOUT_SESSIONS]:
        raise BudgetError("wspf_holdout is not the first 500 dev sessions outside the 2,048 subset by HOLDOUT_KEY")
    sha = doc["sha256"]
    if not isinstance(sha, Mapping) or set(sha) != set(_SHA_NAMES):
        raise BudgetError(f"sha256: keys must be exactly {sorted(_SHA_NAMES)}")
    raw_lists = {**{n: raw[n] for n in SESSION_BUDGETS}, "es_val": doc["es_val"],
                 "disclosed_200w": doc["disclosed_200w"], "wspf_holdout": doc["wspf_holdout"]}
    got = {name: canonical_sha256(raw_lists[name]) for name in _SHA_NAMES}
    for name in _SHA_NAMES:
        if sha[name] != got[name]:
            raise BudgetError(f"sha256[{name!r}] {sha[name]} != {got[name]} recomputed")
    return Budgets(budgets, es_val, disclosed, holdout, dict(sha), file_sha256)


def load(path: Path = PATH) -> Budgets:
    data = Path(path).read_bytes()
    return parse(json.loads(data), file_sha256=hashlib.sha256(data).hexdigest())


def check_dev_ids(b: Budgets, dev_ids: Sequence[str]) -> None:
    """``all`` is exactly the dev population (``dev_ids``: the dev export's session ids)."""
    if set(b.budgets["all"]) != set(dev_ids) or len(dev_ids) != len(b.budgets["all"]):
        raise BudgetError("budgets['all'] is not the dev export's session set")


def check_val_ids(b: Budgets, val_ids: Sequence[str]) -> None:
    """The ES set is the v0.3 rule re-run: val ids by ES key, disclosed sessions removed first, first 500."""
    vs = set(val_ids)
    if not b.disclosed_sessions <= vs:
        raise BudgetError("a disclosed session is not in the val export")
    order = [s for s in sorted(val_ids, key=es_key) if s not in b.disclosed_sessions]
    if list(b.es_val) != order[:ES_SESSIONS]:
        raise BudgetError("es_val differs from the v0.3 ES rule on the val export")


def disclosure_rows(labels: Mapping[str, np.ndarray], per_class: int = PER_CLASS) -> list[Window]:
    """The v0.3 Phase B disclosure rule (``v03.scoring.disclosure_selection``) on the export's val labels.

    Classes in canonical order; within a class ascending key; a window is skipped if its session was
    already used by any class; the chosen windows sorted by key. Only gate-passing scored rows count.
    """
    sids = [str(s) for s in labels["session_ids"]]
    r_sess, r_pos, r_end = labels["row_session"], labels["row_pos"], labels["row_endpoint"]
    r_lab, r_gate = labels["row_label"].astype(np.int64), labels["row_gate"].astype(bool)
    by_class: dict[int, list[tuple[str, int]]] = {c: [] for c in range(N_CLASSES)}
    for i in np.flatnonzero(r_gate & (r_lab < N_CLASSES)).tolist():
        by_class[int(r_lab[i])].append((disclosure_key(sids[int(r_sess[i])], int(r_end[i])), i))
    used: set[str] = set()
    chosen: list[tuple[str, int]] = []
    for c in range(N_CLASSES):
        taken = 0
        for k, i in sorted(by_class[c]):
            sid = sids[int(r_sess[i])]
            if sid in used:
                continue
            chosen.append((k, i))
            used.add(sid)
            taken += 1
            if taken == per_class:
                break
        if taken < per_class:
            raise BudgetError(f"class {c}: only {taken} windows from unused sessions, need {per_class}")
    chosen.sort()
    return [Window(sids[int(r_sess[i])], int(r_pos[i])) for _, i in chosen]


def check_disclosure(b: Budgets, val_labels: Mapping[str, np.ndarray]) -> None:
    """``disclosed_200w`` is the v0.3 Phase B disclosure, in key order (V001..V200)."""
    if list(b.disclosed_200w) != disclosure_rows(val_labels):
        raise BudgetError("disclosed_200w differs from the v0.3 disclosure rule on the val labels")
