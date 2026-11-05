"""v0.4 labelled-budget session lists (PREREGISTRATION.md §7): ``datasets/v0_4/BUDGETS.json``.

Harness only; runs in the frozen v0.3 environment. The Qwen environment reads the written file (it
never imports this package), so the WSP-F server (§7.3, implemented after G1) and any other reader
use the same lists.

The file was written before G1 for the probe and LoRA columns, which were dropped on 2026-11-06
(PREREGISTRATION.md header, §4, §5). Its data are unchanged: v0.4 uses ``disclosed_200w`` (§7.1) and
``wspf_holdout`` (§7.3); ``budgets`` and ``es_val`` are no longer an input of any column, and
``budgets["2048"]`` is kept because the holdout is defined against it:

* ``budgets`` ``{"32", "128", "512", "2048", "all"}``: exactly the v0.3 Phase C subsets,
  :func:`agent_loop_lab.baselines.phase_c_v03.budget_subset` over the dev ids of
  ``datasets/v0_2/splits.json`` (ascending ``sha256("20261105:v03-phase-c-budget:{id}")``, first
  *k*; ``all`` is every dev id in that order, so each budget is a prefix of ``all``);
* ``es_val``: the 500-session val early-stopping set of v0.3 Phase C,
  :func:`agent_loop_lab.baselines.phase_c_v03.es_session_ids`, disclosed sessions removed first;
* ``disclosed_200w``: the 200 WSP-B windows (§7.1), :func:`agent_loop_lab.v03.scoring.disclosure_selection`
  on the val rows of :func:`agent_loop_lab.v04.scoring.load_rows` (which checks them against v0.3's
  ``rows_sha256``), in key order (``V001``-``V200``), each as ``{"session_id", "row_pos"}``;
  ``row_pos`` is the export's row position (``labels.npz`` ``row_pos``: the block index of the
  window's last event, v0.3's ``block``);
* ``wspf_holdout`` (§7.3): the first 500 dev ids by ascending
  ``sha256("20261105:v04-wsp-f-holdout:{id}")`` that are not in the 2,048-session budget, so it is
  disjoint from every budget except ``all`` (which is all of dev: no 500 dev sessions can avoid it).

``sha256`` maps each list's name to the sha256 of its canonical JSON (sorted keys, ``","``/``":"``
separators, UTF-8). The file holds no label: the val labels are read only to select the disclosed
windows, by the label-blind-to-models rule v0.3 used.

:func:`build` reproduces v0.3 before it returns: every budget's ``budget_sessions_sha256`` and the ES
set's ``sha256_of_ids_in_order`` from the committed v0.3 Phase C training logs, and the disclosed
windows' sessions, endpoints and blocks from ``experiments/v0_3/DISCLOSURE_PRIVATE.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..baselines import phase_c_v03 as pc
from ..v03 import scoring as v03_scoring
from ..v03 import sealed as v03_sealed
from . import export

__all__ = ["FORMAT", "BUDGET_NAMES", "HOLDOUT_KEY", "HOLDOUT_SESSIONS", "HOLDOUT_EXCLUDES", "BUDGETS_PATH",
           "BudgetsError", "canonical_sha256", "holdout_key", "wspf_holdout", "disclosed_windows", "build", "render",
           "check", "load"]

FORMAT = "v04-budgets/1"
BUDGET_NAMES: tuple[str, ...] = tuple(str(b) for b in pc.BUDGETS)  # "32", "128", "512", "2048", "all"
HOLDOUT_KEY = "20261105:v04-wsp-f-holdout:{session_id}"
HOLDOUT_SESSIONS = 500
HOLDOUT_EXCLUDES = 2048  # the largest budget below "all"; the smaller budgets are its prefixes
LIST_NAMES: tuple[str, ...] = (*BUDGET_NAMES, "es_val", "disclosed_200w", "wspf_holdout")

REPO = export.REPO
SPLITS_PATH = REPO / "datasets" / "v0_2" / "splits.json"
BUDGETS_PATH = export.V04_DATA / "BUDGETS.json"
V03_PHASE_C_LOGS = REPO / "experiments" / "phase-c-trained" / "v0_3"


class BudgetsError(ValueError):
    pass


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def holdout_key(session_id: str) -> str:
    return hashlib.sha256(HOLDOUT_KEY.format(session_id=session_id).encode("utf-8")).hexdigest()


def _budget(name: str) -> int | str:
    return "all" if name == "all" else int(name)


def split_ids(path: Path = SPLITS_PATH) -> tuple[list[str], list[str]]:
    """``(dev ids, val ids)`` from ``datasets/v0_2/splits.json`` (a committed manifest)."""
    doc = json.loads(path.read_text())
    return list(doc["splits"]["dev"]), list(doc["splits"]["val"])


def wspf_holdout(dev_ids: Sequence[str], n: int = HOLDOUT_SESSIONS) -> list[str]:
    """§7.3: the first ``n`` dev ids by :func:`holdout_key` outside the 2,048-session budget."""
    excluded = set(pc.budget_subset(dev_ids, HOLDOUT_EXCLUDES))
    order = [s for s in sorted(dev_ids, key=holdout_key) if s not in excluded]
    if len(order) < n:
        raise BudgetsError(f"only {len(order)} dev sessions outside the {HOLDOUT_EXCLUDES} budget, need {n}")
    return order[:n]


def disclosed_windows(rows: Sequence[Any], lab: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The 200 disclosed val windows in key order, as ``{"session_id", "row_pos"}``.

    ``rows`` and ``lab`` are :func:`agent_loop_lab.v04.scoring.load_rows`' output; row *i* of ``rows``
    is row *i* of the export, so its position is ``lab["row_pos"][i]``.
    """
    pos = lab["row_pos"]
    if len(pos) != len(rows):
        raise BudgetsError(f"{len(rows)} rows but {len(pos)} row positions")
    return [{"session_id": rows[i].session_id, "row_pos": int(pos[i])}
            for _, i in v03_scoring.disclosure_selection(rows)]


def _check_private_disclosure(chosen: Sequence[dict[str, Any]], rows: Sequence[Any],
                              lab: Mapping[str, Any], path: Path | None) -> None:
    """The selection must be v0.3 Phase B's: same sessions, and per id the same endpoint and block."""
    rec = json.loads((v03_sealed.PRIVATE_DISCLOSURE if path is None else path).read_text())
    private = sorted(rec["windows"], key=lambda w: w["id"])
    if frozenset(w["session_id"] for w in chosen) != v03_sealed.private_disclosed(path):
        raise BudgetsError("disclosed sessions differ from experiments/v0_3/DISCLOSURE_PRIVATE.json")
    index = {(r.session_id, int(p)): r for r, p in zip(rows, lab["row_pos"].tolist())}
    if len(private) != len(chosen):
        raise BudgetsError(f"{len(chosen)} disclosed windows, DISCLOSURE_PRIVATE.json has {len(private)}")
    for w, c in zip(private, chosen):
        r = index[(c["session_id"], c["row_pos"])]
        if (w["session_id"], int(w["block"]), int(w["endpoint"])) != (c["session_id"], c["row_pos"], r.endpoint):
            raise BudgetsError(f"{w['id']}: {c} (endpoint {r.endpoint}) differs from DISCLOSURE_PRIVATE.json "
                               f"(session {w['session_id']}, block {w['block']}, endpoint {w['endpoint']})")


def _check_v03_logs(budgets: Mapping[str, list[str]], es: list[str], log_root: Path | None) -> None:
    """Each list must hash as v0.3 Phase C's training logs recorded (``ids_sha256``, in order)."""
    root = V03_PHASE_C_LOGS if log_root is None else log_root
    for name, ids in budgets.items():
        log = json.loads((root / name / "seed0" / "training_log.json").read_text())
        if (log["budget_sessions"], log["budget_sessions_sha256"]) != (len(ids), pc.ids_sha256(ids)):
            raise BudgetsError(f"budget {name}: differs from the v0.3 Phase C training log")
        if log["es_set"]["sha256_of_ids_in_order"] != pc.ids_sha256(es):
            raise BudgetsError(f"budget {name}: ES set differs from the v0.3 Phase C training log")


def _check_lists(doc: Mapping[str, Any], dev_ids: Sequence[str], val_ids: Sequence[str]) -> None:
    b = doc["budgets"]
    dev, val = set(dev_ids), set(val_ids)
    for name in BUDGET_NAMES:
        want = len(dev_ids) if name == "all" else int(name)
        if len(b[name]) != want or len(set(b[name])) != want or not set(b[name]) <= dev:
            raise BudgetsError(f"budget {name}: {len(b[name])} ids, want {want} distinct dev ids")
        if b["all"][: want] != b[name]:
            raise BudgetsError(f"budget {name} is not a prefix of the 'all' order")
    disclosed = {w["session_id"] for w in doc["disclosed_200w"]}
    es, hold = doc["es_val"], doc["wspf_holdout"]
    n_disclosed = v03_scoring.PER_CLASS * 4  # 50 per class, 4 classes
    if len(disclosed) != n_disclosed or len(doc["disclosed_200w"]) != n_disclosed or not disclosed <= val:
        raise BudgetsError("disclosed windows: want 200 distinct val sessions")
    if len(set(es)) != pc.ES_SESSIONS or not set(es) <= val or set(es) & disclosed:
        raise BudgetsError("es_val: want 500 distinct val ids, none disclosed")
    if len(set(hold)) != HOLDOUT_SESSIONS or not set(hold) <= dev:
        raise BudgetsError("wspf_holdout: want 500 distinct dev ids")
    for name in BUDGET_NAMES[:-1]:
        if set(hold) & set(b[name]):
            raise BudgetsError(f"wspf_holdout overlaps budget {name}")


def build(*, splits_path: Path = SPLITS_PATH, data_dir: Path = export.V04_DATA, private_disclosure: Path | None = None,
          v03_logs: Path | None = None) -> dict[str, Any]:
    """The BUDGETS.json document, checked against v0.3 before it is returned."""
    from . import scoring  # imported here: it needs only the val labels, read once

    dev_ids, val_ids = split_ids(splits_path)
    rows, lab = scoring.load_rows(data_dir)
    if set(str(s) for s in lab["session_ids"]) != set(val_ids):
        raise BudgetsError("val export sessions differ from datasets/v0_2/splits.json")
    budgets = {name: pc.budget_subset(dev_ids, _budget(name)) for name in BUDGET_NAMES}
    chosen = disclosed_windows(rows, lab)
    _check_private_disclosure(chosen, rows, lab, private_disclosure)
    es = pc.es_session_ids(val_ids, {w["session_id"] for w in chosen})
    _check_v03_logs(budgets, es, v03_logs)
    doc: dict[str, Any] = {"format": FORMAT, "budgets": budgets, "es_val": es, "disclosed_200w": chosen,
                           "wspf_holdout": wspf_holdout(dev_ids)}
    _check_lists(doc, dev_ids, val_ids)
    doc["sha256"] = {name: canonical_sha256(_list(doc, name)) for name in LIST_NAMES}
    return doc


def _list(doc: Mapping[str, Any], name: str) -> Any:
    return doc["budgets"][name] if name in BUDGET_NAMES else doc[name]


def render(doc: Mapping[str, Any]) -> str:
    return json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def load(path: Path = BUDGETS_PATH) -> dict[str, Any]:
    """Read BUDGETS.json and verify its format and every list's recorded sha256."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("format") != FORMAT or set(doc) != {"format", "budgets", "es_val", "disclosed_200w", "wspf_holdout",
                                                     "sha256"}:
        raise BudgetsError(f"{path}: not a {FORMAT} file")
    if set(doc["budgets"]) != set(BUDGET_NAMES) or set(doc["sha256"]) != set(LIST_NAMES):
        raise BudgetsError(f"{path}: budget or sha256 names differ from {LIST_NAMES}")
    for name in LIST_NAMES:
        if canonical_sha256(_list(doc, name)) != doc["sha256"][name]:
            raise BudgetsError(f"{path}: {name} does not match its recorded sha256")
    return doc


def check(path: Path = BUDGETS_PATH, **build_kw: Any) -> list[str]:
    """Rebuild and compare byte for byte; returns the problems (empty when the file is current)."""
    if not path.is_file():
        return [f"{path} is missing"]
    problems: list[str] = []
    try:
        load(path)
    except BudgetsError as e:
        problems.append(str(e))
    want = render(build(**build_kw))
    if path.read_text(encoding="utf-8") != want:
        problems.append(f"{path} differs from a rebuild")
    return problems
