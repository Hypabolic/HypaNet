"""Shared pytest helpers: the test screens and a runner for the Node test helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

SLOP = Path(__file__).resolve().parent
SCREENS_DIR = SLOP / "ui" / "tests" / "screens"


def load_test_screens() -> list[dict]:
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(SCREENS_DIR.glob("*.json"))]


def run_node(script: str | Path, payload: object) -> object:
    """Run a Node helper with JSON on stdin and return its JSON output."""
    res = subprocess.run(["node", str(script)], input=json.dumps(payload), capture_output=True, text=True, cwd=SLOP)
    if res.returncode != 0:
        raise AssertionError(f"node {script} failed:\n{res.stderr}")
    return json.loads(res.stdout)


@pytest.fixture(scope="session")
def test_screens() -> list[dict]:
    screens = load_test_screens()
    assert len(screens) >= 30
    return screens
