"""Dataset statistics (M1_CONTRACTS §6).

Over in-memory ``(events, OracleRecord)`` sessions grouped by split:

* class priors per split (scored oracle labels; all six oracle labels counted);
* per-family label mix;
* gate pass rate per family, and gate/oracle agreement per family. A window
  *agrees* when the gate rejects it exactly when its oracle label is
  INSUFFICIENT_INSTRUMENTATION. Families 1-11 must pass the gate on 100% of
  windows; family 12 (``under_instrumented``) must agree on every window. The
  library reports these as ``checks``; it does not raise;
* session-length distribution (per split, per family, overall);
* window counts (total, gate-passed, scored, scored and gate-passed).

The output is a plain JSON-serialisable dict, built in fixed orders from
integer counts, so it does not depend on the order sessions are supplied in.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from ..schema import CLASS_NAMES, CanonicalEvent, OracleLabel, OracleRecord
from .collisions import WindowRow, session_windows, split_order

__all__ = ["UNDER_INSTRUMENTED", "DatasetStatsAccumulator", "dataset_stats", "length_summary"]

UNDER_INSTRUMENTED = "under_instrumented"  # family 12 (M1_CONTRACTS §3)
_LABELS: tuple[OracleLabel, ...] = tuple(OracleLabel)


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def length_summary(hist: Mapping[int, int]) -> dict[str, Any]:
    """min / max / mean / median and the full histogram of session lengths."""
    n = sum(hist.values())
    if not n:
        return {"sessions": 0, "min": None, "max": None, "mean": None, "median": None, "histogram": {}}
    keys = sorted(k for k in hist if hist[k])

    def kth(k: int) -> int:  # 0-based order statistic
        acc = 0
        for v in keys:
            acc += hist[v]
            if k < acc:
                return v
        raise AssertionError("unreachable")

    median = kth(n // 2) if n % 2 else (kth(n // 2 - 1) + kth(n // 2)) / 2
    return {
        "sessions": n,
        "min": keys[0],
        "max": keys[-1],
        "mean": sum(k * hist[k] for k in keys) / n,
        "median": median,
        "histogram": {str(k): hist[k] for k in keys},
    }


class _Cell:
    """Counts for one (split) or (family) or (split, family) cell."""

    __slots__ = ("sessions", "lengths", "labels", "windows", "gate_passed", "scored_gate_passed", "disagree", "disagree_examples")

    def __init__(self) -> None:
        self.sessions = 0
        self.lengths: Counter = Counter()
        self.labels: Counter = Counter()
        self.windows = 0
        self.gate_passed = 0
        self.scored_gate_passed = 0
        self.disagree = 0
        self.disagree_examples: list[tuple[str, int]] = []

    def add_session(self, n_events: int, rows: Sequence[WindowRow], max_examples: int) -> None:
        self.sessions += 1
        self.lengths[n_events] += 1
        for r in rows:
            self.windows += 1
            self.labels[r.label] += 1
            self.gate_passed += r.gate_passed
            self.scored_gate_passed += r.gate_passed and r.label.is_scored
            if r.gate_passed == (r.label is OracleLabel.INSUFFICIENT_INSTRUMENTATION):
                self.disagree += 1
                if max_examples:
                    self.disagree_examples.append((r.session_id, r.endpoint))
                    if len(self.disagree_examples) > 4 * max_examples:
                        self.disagree_examples.sort()
                        del self.disagree_examples[max_examples:]

    def window_counts(self) -> dict[str, int]:
        scored = sum(self.labels[lab] for lab in _LABELS if lab.is_scored)
        return {
            "windows": self.windows,
            "gate_passed": self.gate_passed,
            "scored": scored,
            "scored_gate_passed": self.scored_gate_passed,
        }

    def label_block(self) -> dict[str, Any]:
        scored = sum(self.labels[lab] for lab in _LABELS if lab.is_scored)
        return {
            "label_counts": {lab.value: self.labels[lab] for lab in _LABELS},
            "scored_windows": scored,
            "scored_priors": {name: _ratio(self.labels[OracleLabel(name)], scored) for name in CLASS_NAMES},
        }

    def gate_block(self, max_examples: int) -> dict[str, Any]:
        return {
            "windows": self.windows,
            "gate_passed": self.gate_passed,
            "gate_pass_rate": _ratio(self.gate_passed, self.windows),
            "gate_oracle_disagreements": self.disagree,
            "gate_oracle_agreement_rate": _ratio(self.windows - self.disagree, self.windows),
            "disagreement_examples": [
                {"session_id": s, "endpoint": e} for s, e in sorted(self.disagree_examples)[:max_examples]
            ],
        }


class DatasetStatsAccumulator:
    """Streams sessions (or pre-computed window rows) into dataset statistics."""

    def __init__(self, max_examples: int = 5) -> None:
        if max_examples < 0:
            raise ValueError("max_examples must be >= 0")
        self.max_examples = max_examples
        self.by_split: dict[str, _Cell] = {}
        self.by_family: dict[str, _Cell] = {}
        self.by_split_family: dict[tuple[str, str], _Cell] = {}
        self.total = _Cell()
        self._session_ids: dict[str, str] = {}

    def add_session(
        self,
        split: str,
        events: Sequence[CanonicalEvent],
        record: OracleRecord,
        rows: Sequence[WindowRow] | None = None,
    ) -> list[WindowRow]:
        """Add one session. ``rows`` (from ``session_windows``) avoids re-encoding. Returns the rows."""
        if rows is None:
            rows = session_windows(events, record, split)
        prev = self._session_ids.get(record.session_id)
        if prev is not None:
            raise ValueError(f"session {record.session_id!r} added twice (splits {prev!r} and {split!r})")
        self._session_ids[record.session_id] = split
        fam = record.scenario_family
        for cell in (
            self.by_split.setdefault(split, _Cell()),
            self.by_family.setdefault(fam, _Cell()),
            self.by_split_family.setdefault((split, fam), _Cell()),
            self.total,
        ):
            cell.add_session(len(events), rows, self.max_examples)
        return list(rows)

    def report(self) -> dict[str, Any]:
        splits = split_order(self.by_split)
        families = sorted(self.by_family)
        mx = self.max_examples
        gate = {f: self.by_family[f].gate_block(mx) for f in families}
        others = [f for f in families if f != UNDER_INSTRUMENTED]
        checks = {
            "gate_all_pass_families_1_11": all(gate[f]["gate_passed"] == gate[f]["windows"] for f in others),
            "families_below_full_gate_pass": [f for f in others if gate[f]["gate_passed"] != gate[f]["windows"]],
            "under_instrumented_present": UNDER_INSTRUMENTED in gate,
            "under_instrumented_gate_oracle_agree": (
                gate[UNDER_INSTRUMENTED]["gate_oracle_disagreements"] == 0 if UNDER_INSTRUMENTED in gate else None
            ),
            "gate_oracle_agree_all_families": all(g["gate_oracle_disagreements"] == 0 for g in gate.values()),
        }
        return {
            "definitions": {
                "scored_priors": "fraction of scored oracle windows (PRODUCTIVE/RECOVERY/STALLED/OSCILLATING) per class",
                "gate_oracle_agreement": "a window agrees iff the gate rejects it exactly when its oracle label is INSUFFICIENT_INSTRUMENTATION",
                "windows": "every full stride-1 window of a session (list position >= 7)",
            },
            "splits": splits,
            "families": families,
            "sessions": {
                "total": self.total.sessions,
                "per_split": {s: self.by_split[s].sessions for s in splits},
                "per_family": {f: self.by_family[f].sessions for f in families},
                "per_split_family": {
                    s: {f: self.by_split_family[(s, f)].sessions for f in families if (s, f) in self.by_split_family}
                    for s in splits
                },
            },
            "window_counts": {
                "total": self.total.window_counts(),
                "per_split": {s: self.by_split[s].window_counts() for s in splits},
                "per_family": {f: self.by_family[f].window_counts() for f in families},
            },
            "class_priors": {
                "total": self.total.label_block(),
                "per_split": {s: self.by_split[s].label_block() for s in splits},
            },
            "family_label_mix": {f: self.by_family[f].label_block() for f in families},
            "gate": {"per_family": gate},
            "under_instrumented_gate_oracle": gate.get(UNDER_INSTRUMENTED),
            "session_length": {
                "total": length_summary(self.total.lengths),
                "per_split": {s: length_summary(self.by_split[s].lengths) for s in splits},
                "per_family": {f: length_summary(self.by_family[f].lengths) for f in families},
            },
            "checks": checks,
        }


def dataset_stats(
    sessions_by_split: Mapping[str, Iterable[tuple[Sequence[CanonicalEvent], OracleRecord]]],
    *,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Dataset statistics report; the content of ``artifacts/reports/m1_dataset_stats.json``."""
    acc = DatasetStatsAccumulator(max_examples)
    for split in split_order(sessions_by_split):
        for events, record in sessions_by_split[split]:
            acc.add_session(split, events, record)
    return acc.report()

