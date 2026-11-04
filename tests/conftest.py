"""Shared pytest fixtures.

The anti-training guard refuses to start if a harness-forbidden module
(``agent_loop_lab.synth*``, ``.datasets``, ``.eval``, ``.baselines``) is already
imported. Other test modules legitimately import those packages, so for tests
that exercise the guard we set the forbidden modules aside for the duration of
the test and restore the *same* module objects afterwards (keeping class
identity intact for every other test).
"""

from __future__ import annotations

import sys

import pytest

import agent_loop_lab
from agent_loop_lab.guards import FORBIDDEN_MODULE_PREFIXES

_GUARD_TEST_MODULES = {"tests.test_guards", "tests.test_runner", "tests.test_cli"}


@pytest.fixture(autouse=True)
def _isolate_forbidden_modules(request):
    if request.module.__name__ not in _GUARD_TEST_MODULES:
        yield
        return
    saved = {m: mod for m, mod in sys.modules.items() if m.startswith(FORBIDDEN_MODULE_PREFIXES)}
    saved_attrs = {}
    for name in saved:
        del sys.modules[name]
        parent, _, child = name.rpartition(".")
        if parent == "agent_loop_lab" and child in vars(agent_loop_lab):
            saved_attrs[child] = vars(agent_loop_lab)[child]
            delattr(agent_loop_lab, child)
    try:
        yield
    finally:
        for name in [m for m in sys.modules if m.startswith(FORBIDDEN_MODULE_PREFIXES)]:
            del sys.modules[name]
        for child in [c for c in vars(agent_loop_lab) if f"agent_loop_lab.{c}".startswith(FORBIDDEN_MODULE_PREFIXES)]:
            delattr(agent_loop_lab, child)
        sys.modules.update(saved)
        for child, mod in saved_attrs.items():
            setattr(agent_loop_lab, child, mod)
