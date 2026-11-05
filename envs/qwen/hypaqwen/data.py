"""Readers for the v0.4 task exports (``agent_loop_lab.v04.export``)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np

from . import V04_DATA

ALLOWED_SPLITS = frozenset({"dev", "val"})


class Task(NamedTuple):
    session_id: str
    blocks: list[str]
    window_pos: list[int]


def _check(split: str) -> None:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"only {sorted(ALLOWED_SPLITS)} are exported before the sealed evaluation, not {split!r}")


def iter_tasks(split: str, data_dir: Path = V04_DATA) -> Iterator[Task]:
    _check(split)
    with gzip.open(data_dir / f"{split}.tasks.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            yield Task(d["session_id"], d["blocks"], d["window_pos"])


def load_labels(split: str, data_dir: Path = V04_DATA) -> dict[str, np.ndarray]:
    """Window rows (harness side only: never pass these to a weight author)."""
    _check(split)
    with np.load(data_dir / f"{split}.labels.npz") as z:
        return {k: z[k] for k in z.files}
