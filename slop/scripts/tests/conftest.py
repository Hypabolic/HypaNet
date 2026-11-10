"""Fixtures for the freeze, sealed-selection and launcher tests: a copy of the frozen slop files in a temporary root."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

SLOP = Path(__file__).resolve().parents[2]
# what scripts/freeze.py hashes (no data, no labels, no node_modules)
FROZEN_GLOBS = ["SPEC.md", "LABELLING.md", "pyproject.toml", "uv.lock", "package.json", "package-lock.json", ".npmrc",
                "extract/cli.js", "extract/src/**/*.ts", "ui/*.py", "ui/*.json", "lint/*.js", "lint/*.py",
                "rules/index.js", "rules/cli.js", "rules/README.md", "rules/src/**/*", "generator/*.py",
                "generator/*.ts", "scripts/*.py", "scripts/*.sh"]


def copy_frozen(dest: Path) -> Path:
    for pat in FROZEN_GLOBS:
        for p in SLOP.glob(pat):
            if p.is_file() and "__pycache__" not in p.parts:
                t = dest / p.relative_to(SLOP)
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, t)
    return dest


@pytest.fixture
def fake_slop(tmp_path) -> Path:
    """A slop-like root with the frozen files (no data sets, no labels, no registry)."""
    return copy_frozen(tmp_path / "slop")
