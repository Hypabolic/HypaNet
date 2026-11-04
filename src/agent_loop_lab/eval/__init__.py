"""M1 audits over in-memory sessions (M1_CONTRACTS §6; spec v0.2 §6.4).

This package is a forbidden prefix under the anti-training guard; the weight
build path must never import it.

Inputs are ``(events, OracleRecord)`` pairs grouped by split, e.g.
``{"dev": [...], "val": [...], "test": [...]}``. Any iterable works, so a
builder can stream sessions. Labels are read from the records, never
recomputed. Every report is a deterministic, JSON-serialisable dict.

* :func:`collisions.collision_report` -> ``artifacts/reports/m1_collisions_dev.json``
  (collision audit of one split plus cross-split identical-vector counts);
* :func:`dataset_stats.dataset_stats` -> ``artifacts/reports/m1_dataset_stats.json``;
* :func:`audit_dataset` computes both in a single gate/encode pass;
* :func:`to_json` serialises a report reproducibly.

Lower-level pieces: ``session_windows`` (one session's window rows),
``collision_audit`` / ``cross_split_overlap`` over rows, ``bayes_ceiling``
over label groups, and the streaming accumulators.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

from ..schema import CanonicalEvent, OracleRecord
from .collisions import (
    CollisionAccumulator,
    OverlapAccumulator,
    WindowRow,
    bayes_ceiling,
    collision_audit,
    collision_report,
    cross_split_overlap,
    session_windows,
    split_order,
)
from .dataset_stats import DatasetStatsAccumulator, dataset_stats

__all__ = [
    "CollisionAccumulator",
    "DatasetStatsAccumulator",
    "OverlapAccumulator",
    "WindowRow",
    "audit_dataset",
    "bayes_ceiling",
    "collision_audit",
    "collision_report",
    "cross_split_overlap",
    "dataset_stats",
    "session_windows",
    "split_order",
    "to_json",
]


def audit_dataset(
    sessions_by_split: Mapping[str, Iterable[tuple[Sequence[CanonicalEvent], OracleRecord]]],
    *,
    audit_split: str = "dev",
    max_examples: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(dataset_stats_report, collision_report)`` from one pass over the sessions.

    Equal to ``dataset_stats(...)`` and ``collision_report(...)`` called
    separately, but each window is gated and encoded once.
    """
    if audit_split not in sessions_by_split:
        raise KeyError(f"audit split {audit_split!r} not in {sorted(sessions_by_split)}")
    stats = DatasetStatsAccumulator(max_examples)
    coll = CollisionAccumulator(max_examples)
    overlap = OverlapAccumulator()
    for split in split_order(sessions_by_split):
        for events, record in sessions_by_split[split]:
            rows = session_windows(events, record, split)
            stats.add_session(split, events, record, rows)
            for row in rows:
                overlap.add(row)
                if split == audit_split:
                    coll.add(row)
    collisions = {
        "audit_split": audit_split,
        "collisions": coll.report(),
        "cross_split": overlap.report(sessions_by_split.keys()),
    }
    return stats.report(), collisions


def to_json(report: Mapping[str, Any]) -> str:
    """Reproducible JSON text (2-space indent, insertion order, no NaN, trailing newline)."""
    return json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
