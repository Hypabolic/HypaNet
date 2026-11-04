"""Scenario families (M1_CONTRACTS §3). One module per family, named as below.

Each module exposes ``FAMILY = "<name>"`` and
``generate(rng, session_id, cfg) -> (events, OracleRecord)``.
Modules are imported lazily so families can be written in parallel.
"""

from __future__ import annotations

import dataclasses
import importlib
import random
from types import ModuleType

from ...oracle_config import OracleConfig, load_oracle_config
from ...schema import CanonicalEvent, OracleRecord

__all__ = ["FAMILY_NAMES", "get_family", "generate_session"]

FAMILY_NAMES: tuple[str, ...] = (
    "incremental_repair",
    "exploration",
    "transient_external",
    "bounded_retry",
    "repeated_test_failure",
    "cosmetic_churn",
    "abab_oscillation",
    "tool_oscillation",
    "improving_tests",
    "polling_recovery",
    "rising_cost",
    "under_instrumented",
)


def get_family(name: str) -> ModuleType:
    """Import ``agent_loop_lab.synth.families.<name>`` and check its contract."""
    if name not in FAMILY_NAMES:
        raise KeyError(f"unknown family {name!r}; expected one of {FAMILY_NAMES}")
    mod = importlib.import_module(f"{__name__}.{name}")
    if getattr(mod, "FAMILY", None) != name or not callable(getattr(mod, "generate", None)):
        raise TypeError(f"family module {name} must define FAMILY = {name!r} and generate(rng, session_id, cfg)")
    return mod


def generate_session(
    family: str, seed: int, session_id: str, cfg: OracleConfig | None = None
) -> tuple[list[CanonicalEvent], OracleRecord]:
    """Generate one session with ``random.Random(seed)`` and stamp ``seed`` on the record."""
    cfg = cfg or load_oracle_config()
    mod = get_family(family)
    events, record = mod.generate(random.Random(seed), session_id, cfg)
    if record.scenario_family != family or record.session_id != session_id:
        raise ValueError(f"family {family}: record has family {record.scenario_family!r} / session {record.session_id!r}")
    return events, dataclasses.replace(record, seed=int(seed))
