"""Split and dataset tooling (M1_CONTRACTS §5).

This package is a forbidden prefix under the anti-training guard; the weight
build path must never import it. See :mod:`.build` for seeding, split rule,
file layout and manifest contents.
"""

from __future__ import annotations

from .build import (
    DEFAULT_MASTER_SEED,
    GZIP_LEVEL,
    PAIRS_PER_KIND,
    SPLITS,
    STANDARD_SESSIONS,
    BuildError,
    BuildResult,
    assign_splits,
    build as build_datasets,
    counterfactual_seed,
    data_path,
    family_counts,
    file_sha256,
    iter_sessions,
    session_id_for,
    session_seed,
    split_cuts,
    write_reports,
)

__all__ = [
    "DEFAULT_MASTER_SEED",
    "GZIP_LEVEL",
    "PAIRS_PER_KIND",
    "SPLITS",
    "STANDARD_SESSIONS",
    "BuildError",
    "BuildResult",
    "assign_splits",
    "build_datasets",
    "counterfactual_seed",
    "data_path",
    "family_counts",
    "file_sha256",
    "iter_sessions",
    "session_id_for",
    "session_seed",
    "split_cuts",
    "write_reports",
]
