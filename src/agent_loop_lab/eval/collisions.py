"""Feature-collision audit (spec v0.2 §6.4; M1_CONTRACTS §6).

Works on in-memory ``(events, OracleRecord)`` sessions. Labels are read from
``record.window_labels`` and are never recomputed here; the input vectors come
from the frozen runtime path (``windows.iter_windows``: gate, then encode).

Population. Only windows that pass the runtime gate **and** carry a scored
oracle label (PRODUCTIVE / RECOVERY / STALLED / OSCILLATING) are grouped.
Gate-rejected windows have no input vector; ambiguous and instrumentation
windows are not scored. Both are counted separately in ``population``.

Grouping. Windows are grouped by the exact bytes of their float32 (120,)
input vector (``ndarray.tobytes()``). The group id in the report is the
sha256 of those bytes.

Definitions (also written into every report under ``definitions``):

* A *collision group* is a group whose windows carry more than one distinct
  scored label.
* Collision rate of a class (or family) = scored windows of that class (or
  family) that lie in a collision group / all scored windows of that class
  (or family) in the population.
* Bayes ceiling: predict each group's majority label (ties broken by
  canonical class order, i.e. the lowest ``ClassLabel`` index wins), then
  score that prediction against every window's own label. Accuracy is
  correct / windows. Macro-F1 averages per-class F1 over the canonical
  classes with support > 0 in the population. Majority predictions are
  always a label present in their group, so this equals averaging over the
  union of true and predicted classes. A class with no predictions has
  precision 0 (and so F1 0).

Everything is computed from integer counts and is independent of the order in
which sessions are supplied. Outputs are plain JSON-serialisable dicts.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..schema import CLASS_NAMES, CanonicalEvent, ClassLabel, OracleLabel, OracleRecord
from ..windows import iter_windows

__all__ = [
    "CANONICAL_SPLITS",
    "WindowRow",
    "session_windows",
    "iter_split_rows",
    "split_order",
    "class_of",
    "majority_label",
    "bayes_ceiling",
    "collision_audit",
    "cross_split_overlap",
    "collision_report",
    "DEFINITIONS",
]

CANONICAL_SPLITS: tuple[str, ...] = ("dev", "val", "test")

DEFINITIONS: dict[str, str] = {
    "population": "windows that pass the runtime gate and carry a scored oracle label",
    "group_key": "exact bytes of the float32 (120,) input vector; group id = sha256 of those bytes",
    "collision_group": "a group whose windows carry more than one distinct scored label",
    "collision_rate": "scored windows lying in a collision group / scored windows in the population (per class, per family, overall)",
    "majority_label": "most frequent scored label in a group; ties broken by canonical class order (lowest index wins)",
    "bayes_accuracy": "windows whose label equals their group's majority label / windows in the population",
    "bayes_macro_f1": "mean per-class F1 of the majority-label predictor over canonical classes with support > 0; precision is 0 for a class never predicted",
    "cross_split": "a vector is shared when the same input-vector bytes occur in windows of two or more splits",
}


# ---------------------------------------------------------------------------
# Window rows: one gate + encode pass per session, shared by every audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowRow:
    """One stride-1 window of a session, with its gate result and oracle label."""

    session_id: str
    family: str
    split: str | None
    endpoint: int  # sequence_index of the window's last event
    label: OracleLabel
    gate_passed: bool
    vector: bytes | None  # float32 (120,) bytes when the gate passed, else None

    @property
    def scored(self) -> bool:
        return self.label.is_scored

    @property
    def in_population(self) -> bool:
        """Gate-passing and oracle-scored: the windows the collision audit groups."""
        return self.gate_passed and self.label.is_scored


def session_windows(
    events: Sequence[CanonicalEvent], record: OracleRecord, split: str | None = None
) -> list[WindowRow]:
    """Gate and encode every full window of one session (``windows.iter_windows``).

    The label of each window is ``record.window_labels[endpoint]``. Raises
    ``ValueError`` if the record does not belong to the events or lacks a
    label for some window.
    """
    events = list(events)
    if events and events[0].session_id != record.session_id:
        raise ValueError(f"record {record.session_id!r} does not match events of session {events[0].session_id!r}")
    rows: list[WindowRow] = []
    for w in iter_windows(events):
        try:
            label = record.window_labels[w.endpoint_index]
        except KeyError:
            raise ValueError(f"session {record.session_id!r}: no oracle label for window endpoint {w.endpoint_index}") from None
        rows.append(
            WindowRow(
                session_id=record.session_id,
                family=record.scenario_family,
                split=split,
                endpoint=w.endpoint_index,
                label=OracleLabel(label),
                gate_passed=w.gate.passed,
                vector=w.features.tobytes() if w.gate.passed else None,
            )
        )
    return rows


def iter_split_rows(
    sessions_by_split: Mapping[str, Iterable[tuple[Sequence[CanonicalEvent], OracleRecord]]],
) -> Iterator[WindowRow]:
    """Window rows of every session, split by split in :func:`split_order`."""
    for split in split_order(sessions_by_split):
        for events, record in sessions_by_split[split]:
            yield from session_windows(events, record, split)


def split_order(splits: Iterable[str]) -> list[str]:
    """Canonical splits first (dev, val, test), then any others sorted."""
    names = list(dict.fromkeys(splits))
    return [s for s in CANONICAL_SPLITS if s in names] + sorted(s for s in names if s not in CANONICAL_SPLITS)


# ---------------------------------------------------------------------------
# Bayes ceiling arithmetic
# ---------------------------------------------------------------------------


def class_of(label: ClassLabel | OracleLabel | str | int) -> ClassLabel:
    """Coerce a scored label (ClassLabel, scored OracleLabel, class name or index)."""
    if isinstance(label, ClassLabel):
        return label
    if isinstance(label, OracleLabel):
        return label.to_class()
    if isinstance(label, str):
        return OracleLabel(label).to_class()
    if isinstance(label, int) and not isinstance(label, bool):
        return ClassLabel(label)
    raise TypeError(f"not a scored label: {label!r}")


def majority_label(counts: Mapping[ClassLabel, int]) -> ClassLabel:
    """Most frequent class; ties go to the lowest canonical index."""
    best = None
    for c in ClassLabel:  # canonical order: first maximum wins ties
        n = counts.get(c, 0)
        if n > 0 and (best is None or n > counts[best]):
            best = c
    if best is None:
        raise ValueError("majority of an empty group")
    return best


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _bayes_from_counts(groups: Iterable[Mapping[ClassLabel, int]]) -> dict[str, Any]:
    support = Counter()
    predicted = Counter()
    correct = Counter()
    n_groups = 0
    for counts in groups:
        n_groups += 1
        maj = majority_label(counts)
        size = sum(counts.values())
        for c, n in counts.items():
            support[c] += n
        predicted[maj] += size
        correct[maj] += counts[maj]
    total = sum(support.values())
    per_class: dict[str, dict[str, Any]] = {}
    f1s: list[float] = []
    for c in ClassLabel:
        tp, pred, sup = correct[c], predicted[c], support[c]
        precision = tp / pred if pred else 0.0
        recall = tp / sup if sup else None
        if sup:
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            f1s.append(f1)
        else:
            f1 = None
        per_class[c.name] = {
            "support": sup,
            "predicted": pred,
            "correct": tp,
            "precision": precision if pred else (0.0 if sup else None),
            "recall": recall,
            "f1": f1,
        }
    n_correct = sum(correct.values())
    return {
        "windows": total,
        "groups": n_groups,
        "correct": n_correct,
        "accuracy": _ratio(n_correct, total),
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "macro_f1_classes": [c.name for c in ClassLabel if support[c]],
        "per_class": per_class,
    }


def bayes_ceiling(groups: Iterable[Iterable[ClassLabel | OracleLabel | str | int] | Mapping[Any, int]]) -> dict[str, Any]:
    """Accuracy and macro-F1 of predicting each group's majority label.

    ``groups`` is an iterable of groups; each group is either a sequence of
    scored labels (one per window) or a mapping label -> count. See the module
    docstring for the exact conventions.
    """

    def counts(g: Any) -> Counter:
        out: Counter = Counter()
        if isinstance(g, Mapping):
            for k, n in g.items():
                if n:
                    out[class_of(k)] += int(n)
        else:
            for lab in g:
                out[class_of(lab)] += 1
        if not out:
            raise ValueError("empty group")
        return out

    return _bayes_from_counts(counts(g) for g in groups)


# ---------------------------------------------------------------------------
# Collision audit
# ---------------------------------------------------------------------------


class _Group:
    __slots__ = ("labels", "families", "examples")

    def __init__(self) -> None:
        self.labels: Counter = Counter()
        self.families: Counter = Counter()
        self.examples: list[tuple[str, int, str, str]] = []  # (session_id, endpoint, family, label)


class CollisionAccumulator:
    """Streams window rows into collision groups (exact vector bytes)."""

    def __init__(self, max_examples: int = 5) -> None:
        if max_examples < 0:
            raise ValueError("max_examples must be >= 0")
        self.max_examples = max_examples
        self.groups: dict[bytes, _Group] = {}
        self.windows_total = 0
        self.gate_rejected = 0
        self.unscored_gate_passed: Counter = Counter()
        self.class_totals: Counter = Counter()
        self.family_totals: Counter = Counter()

    def add(self, row: WindowRow) -> None:
        self.windows_total += 1
        if not row.gate_passed:
            self.gate_rejected += 1
            return
        if not row.label.is_scored:
            self.unscored_gate_passed[row.label] += 1
            return
        if row.vector is None:
            raise ValueError(f"gate-passing window {row.session_id}@{row.endpoint} has no vector")
        cls = row.label.to_class()
        g = self.groups.get(row.vector)
        if g is None:
            g = self.groups[row.vector] = _Group()
        g.labels[cls] += 1
        g.families[row.family] += 1
        self.class_totals[cls] += 1
        self.family_totals[row.family] += 1
        if self.max_examples:
            g.examples.append((row.session_id, row.endpoint, row.family, cls.name))
            if len(g.examples) > 4 * self.max_examples:
                g.examples.sort()
                del g.examples[self.max_examples :]

    def report(self) -> dict[str, Any]:
        keyed = sorted(((hashlib.sha256(k).hexdigest(), g) for k, g in self.groups.items()), key=lambda kg: kg[0])
        collision_class: Counter = Counter()
        collision_family: Counter = Counter()
        collision_groups: list[dict[str, Any]] = []
        sizes: Counter = Counter()
        for digest, g in keyed:
            size = sum(g.labels.values())
            sizes[size] += 1
            if len(g.labels) < 2:
                continue
            for c, n in g.labels.items():
                collision_class[c] += n
            for f, n in g.families.items():
                collision_family[f] += n
            collision_groups.append(
                {
                    "vector_sha256": digest,
                    "windows": size,
                    "label_counts": {c.name: g.labels[c] for c in ClassLabel if g.labels[c]},
                    "majority": majority_label(g.labels).name,
                    "family_counts": {f: g.families[f] for f in sorted(g.families)},
                    "examples": [
                        {"session_id": s, "endpoint": e, "family": f, "label": lab}
                        for s, e, f, lab in sorted(g.examples)[: self.max_examples]
                    ],
                }
            )
        collision_groups.sort(key=lambda d: (-d["windows"], d["vector_sha256"]))
        scored = sum(self.class_totals.values())
        in_coll = sum(collision_class.values())
        return {
            "definitions": dict(DEFINITIONS),
            "population": {
                "windows_total": self.windows_total,
                "gate_rejected": self.gate_rejected,
                "gate_passed_unscored": {
                    lab.value: self.unscored_gate_passed[lab] for lab in OracleLabel if not lab.is_scored
                },
                "scored_gate_passed": scored,
            },
            "groups": len(self.groups),
            "singleton_groups": sizes.get(1, 0),
            "group_size_histogram": {str(k): sizes[k] for k in sorted(sizes)},
            "collision_groups_count": len(collision_groups),
            "windows_in_collision_groups": in_coll,
            "collision_rate": _ratio(in_coll, scored),
            "per_class": {
                name: {
                    "windows": self.class_totals[ClassLabel[name]],
                    "in_collision_groups": collision_class[ClassLabel[name]],
                    "collision_rate": _ratio(collision_class[ClassLabel[name]], self.class_totals[ClassLabel[name]]),
                }
                for name in CLASS_NAMES
            },
            "per_family": {
                f: {
                    "windows": self.family_totals[f],
                    "in_collision_groups": collision_family[f],
                    "collision_rate": _ratio(collision_family[f], self.family_totals[f]),
                }
                for f in sorted(self.family_totals)
            },
            "bayes_ceiling": _bayes_from_counts(g.labels for _, g in keyed),
            "collision_groups": collision_groups,
        }


def collision_audit(rows: Iterable[WindowRow], *, max_examples: int = 5) -> dict[str, Any]:
    """§6.4 collision audit over window rows (typically one split's)."""
    acc = CollisionAccumulator(max_examples)
    for r in rows:
        acc.add(r)
    return acc.report()


# ---------------------------------------------------------------------------
# Cross-split identical vectors
# ---------------------------------------------------------------------------


class OverlapAccumulator:
    """Tracks, per input vector, the windows of each split it occurs in.

    Two populations are reported: all gate-passing windows (any oracle label:
    the input is what can leak) and the scored subset.
    """

    def __init__(self) -> None:
        # vector -> split -> Counter(label)
        self.vectors: dict[bytes, dict[str, Counter]] = {}

    def add(self, row: WindowRow) -> None:
        if not row.gate_passed:
            return
        if row.split is None:
            raise ValueError(f"window {row.session_id}@{row.endpoint} has no split")
        if row.vector is None:
            raise ValueError(f"gate-passing window {row.session_id}@{row.endpoint} has no vector")
        per_split = self.vectors.get(row.vector)
        if per_split is None:
            per_split = self.vectors[row.vector] = {}
        per_split.setdefault(row.split, Counter())[row.label] += 1

    def _population(self, splits: list[str], scored_only: bool) -> dict[str, Any]:
        windows = Counter()
        distinct = Counter()
        windows_shared = Counter()
        distinct_shared = Counter()
        pair_vectors = Counter()
        pair_windows: dict[tuple[str, str], Counter] = {}
        multi = 0
        conflicting = 0
        for per_split in self.vectors.values():
            present: dict[str, int] = {}
            labels: set[str] = set()
            for s, lab_counts in per_split.items():
                n = sum(c for lab, c in lab_counts.items() if lab.is_scored or not scored_only)
                if n:
                    present[s] = n
                    labels.update(lab.value for lab, c in lab_counts.items() if c and (lab.is_scored or not scored_only))
            if not present:
                continue
            for s, n in present.items():
                windows[s] += n
                distinct[s] += 1
            if len(present) < 2:
                continue
            multi += 1
            scored_labels = {lab for lab in labels if OracleLabel(lab).is_scored}
            if len(scored_labels) > 1:
                conflicting += 1
            for s, n in present.items():
                windows_shared[s] += n
                distinct_shared[s] += 1
            ordered = [s for s in splits if s in present]
            for i, a in enumerate(ordered):
                for b in ordered[i + 1 :]:
                    pair_vectors[(a, b)] += 1
                    pw = pair_windows.setdefault((a, b), Counter())
                    pw[a] += present[a]
                    pw[b] += present[b]
        pairs = {}
        for i, a in enumerate(splits):
            for b in splits[i + 1 :]:
                pw = pair_windows.get((a, b), Counter())
                pairs[f"{a}|{b}"] = {
                    "shared_distinct_vectors": pair_vectors[(a, b)],
                    f"windows_in_{a}": pw[a],
                    f"windows_in_{b}": pw[b],
                }
        return {
            "vectors_in_multiple_splits": multi,
            "vectors_in_multiple_splits_with_conflicting_scored_labels": conflicting,
            "per_split": {
                s: {
                    "windows": windows[s],
                    "distinct_vectors": distinct[s],
                    "windows_with_vector_in_another_split": windows_shared[s],
                    "distinct_vectors_in_another_split": distinct_shared[s],
                    "shared_window_rate": _ratio(windows_shared[s], windows[s]),
                }
                for s in splits
            },
            "pairs": pairs,
        }

    def report(self, splits: Iterable[str] | None = None) -> dict[str, Any]:
        seen = {s for per in self.vectors.values() for s in per}
        names = split_order(list(splits) + sorted(seen - set(splits)) if splits is not None else seen)
        return {
            "definition": DEFINITIONS["cross_split"],
            "splits": names,
            "gate_passed": self._population(names, scored_only=False),
            "scored": self._population(names, scored_only=True),
        }


def cross_split_overlap(rows: Iterable[WindowRow], splits: Iterable[str] | None = None) -> dict[str, Any]:
    """Count identical input vectors occurring in more than one split.

    ``rows`` must carry ``split``. ``splits`` (optional) fixes which splits
    appear in the report even if they have no windows.
    """
    acc = OverlapAccumulator()
    for r in rows:
        acc.add(r)
    return acc.report(splits)


def collision_report(
    sessions_by_split: Mapping[str, Iterable[tuple[Sequence[CanonicalEvent], OracleRecord]]],
    *,
    audit_split: str = "dev",
    max_examples: int = 5,
) -> dict[str, Any]:
    """Collision audit of ``audit_split`` plus cross-split overlap, in one encode pass.

    This is the content of ``artifacts/reports/m1_collisions_dev.json``.
    """
    if audit_split not in sessions_by_split:
        raise KeyError(f"audit split {audit_split!r} not in {sorted(sessions_by_split)}")
    coll = CollisionAccumulator(max_examples)
    overlap = OverlapAccumulator()
    for row in iter_split_rows(sessions_by_split):
        overlap.add(row)
        if row.split == audit_split:
            coll.add(row)
    return {
        "audit_split": audit_split,
        "collisions": coll.report(),
        "cross_split": overlap.report(sessions_by_split.keys()),
    }
