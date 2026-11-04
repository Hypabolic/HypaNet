"""Lookalike requirement (spec v0.2 §7.2; M1_CONTRACTS §3).

Each required family pair must produce at least one shared 8-event
category/outcome sequence that carries opposite scored oracle labels in the
two families. The search runs over seeds of the real family generators, with
an early exit; labels are the stored oracle labels (latent behaviour), never
derived from encoded features.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

import pytest

from agent_loop_lab.oracle_config import load_oracle_config
from agent_loop_lab.schema import EVENTS_PER_WINDOW
from agent_loop_lab.synth.families import generate_session
from agent_loop_lab.synth.latent import LatentStep, steps_from_record

CoSeq = tuple[tuple[str, str], ...]
MAX_SEEDS = 200  # per family; the probe finds every pair within 20
BATCH = 10


def _count(seq: CoSeq, cats: Sequence[str], outcome: str | None = None) -> int:
    return sum(1 for c, o in seq if c in cats and (outcome is None or o == outcome))


def _edit_test_alternation(seq: CoSeq) -> bool:
    return _count(seq, ("MUTATE",), "SUCCESS") >= 2 and _count(seq, ("VALIDATE",), "ERROR") >= 2


def _repeated_errors(seq: CoSeq) -> bool:
    return _count(seq, ("VALIDATE", "EXTERNAL"), "ERROR") >= 3


def _several_edits(seq: CoSeq) -> bool:
    return _count(seq, ("MUTATE",)) >= 3


def _single_revert(window: Sequence[LatentStep]) -> bool:
    return sum(s.undoes_prior_mutation for s in window) == 1


def _any(window: Sequence[LatentStep]) -> bool:
    return True


@dataclass(frozen=True)
class Pair:
    name: str
    family_a: str
    label_a: str
    family_b: str
    label_b: str
    shape: Callable[[CoSeq], bool]  # on the shared category/outcome sequence
    latent_b: Callable[[Sequence[LatentStep]], bool] = _any  # on family B's window steps


PAIRS = [
    Pair("6_vs_9", "cosmetic_churn", "STALLED", "improving_tests", "PRODUCTIVE", _edit_test_alternation),
    Pair("5_vs_4", "repeated_test_failure", "STALLED", "bounded_retry", "RECOVERY", _repeated_errors),
    Pair("7_vs_1", "abab_oscillation", "OSCILLATING", "incremental_repair", "PRODUCTIVE", _several_edits, _single_revert),
]


def _seed(family: str, k: int) -> int:
    return int.from_bytes(hashlib.sha256(f"lookalike:{family}:{k}".encode()).digest()[:8], "big")


def _index_batch(family: str, label: str, ks: range, shape, latent, out: dict[CoSeq, tuple[int, int]]) -> None:
    """Add windows of ``family`` labelled ``label`` (and passing the filters) to ``out``: seq -> (seed, endpoint)."""
    cfg = load_oracle_config()
    for k in ks:
        sd = _seed(family, k)
        events, record = generate_session(family, sd, f"{family}-{sd:016x}", cfg)
        steps, _ = steps_from_record(record)
        co = [(e.operation_category.value, e.outcome.value) for e in events]
        for t in range(EVENTS_PER_WINDOW - 1, len(events)):
            if record.window_labels[events[t].sequence_index].value != label:
                continue
            seq = tuple(co[t - EVENTS_PER_WINDOW + 1 : t + 1])
            if seq in out or not shape(seq) or not latent(steps[t - EVENTS_PER_WINDOW + 1 : t + 1]):
                continue
            out[seq] = (sd, events[t].sequence_index)


def find_lookalike(pair: Pair) -> tuple[CoSeq, tuple[int, int], tuple[int, int]] | None:
    a: dict[CoSeq, tuple[int, int]] = {}
    b: dict[CoSeq, tuple[int, int]] = {}
    for lo in range(0, MAX_SEEDS, BATCH):
        ks = range(lo, lo + BATCH)
        _index_batch(pair.family_a, pair.label_a, ks, pair.shape, _any, a)
        _index_batch(pair.family_b, pair.label_b, ks, pair.shape, pair.latent_b, b)
        shared = sorted(set(a) & set(b))
        if shared:
            seq = shared[0]
            return seq, a[seq], b[seq]
    return None


@pytest.mark.parametrize("pair", PAIRS, ids=[p.name for p in PAIRS])
def test_lookalike_pair_shares_sequence_with_opposite_labels(pair: Pair):
    assert pair.label_a != pair.label_b
    found = find_lookalike(pair)
    assert found is not None, (
        f"{pair.family_a} ({pair.label_a}) and {pair.family_b} ({pair.label_b}) share no qualifying "
        f"8-event category/outcome sequence within {MAX_SEEDS} seeds each"
    )
    seq, (sd_a, end_a), (sd_b, end_b) = found
    # Re-check both windows from fresh generations: same string, opposite stored labels.
    cfg = load_oracle_config()
    for fam, sd, end, label in ((pair.family_a, sd_a, end_a, pair.label_a), (pair.family_b, sd_b, end_b, pair.label_b)):
        events, record = generate_session(fam, sd, f"{fam}-{sd:016x}", cfg)
        pos = [e.sequence_index for e in events].index(end)
        window = events[pos - EVENTS_PER_WINDOW + 1 : pos + 1]
        assert tuple((e.operation_category.value, e.outcome.value) for e in window) == seq
        assert record.window_labels[end].value == label
        assert record.window_labels[end].is_scored
