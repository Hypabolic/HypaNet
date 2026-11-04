"""Pre-registered oracle parameters (spec v0.2 §6.2).

The frozen values live in ``oracle_config.v0_2.yaml`` at the repository root.
``load_oracle_config`` validates the exact key set, types and ranges, and
rejects duplicate keys, so a typo, a silently added parameter or a repeated one
fails loudly instead of changing labels.

File layout::

    spec_version: "0.2"
    approved: "2026-11-04"
    parameters:
      eps_p: 0.02
      f_prod: 2
      ...
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schema import SPEC_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "oracle_config.v0_2.yaml"

TOP_LEVEL_KEYS = frozenset({"spec_version", "approved", "parameters"})
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class OracleConfigError(ValueError):
    """Raised when the oracle config file or values violate the contract."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys.

    Plain ``yaml.safe_load`` keeps the last of two equal keys, so a repeated
    parameter would silently override the value a reader sees first.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:  # unhashable key: the base constructor reports it
                continue
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark, f"found duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True)
class OracleConfig:
    eps_p: float
    f_prod: int
    n_stall: int
    n_grey: int
    n_rev: int
    r_max: int
    p_max: int
    k_rec: int

    def __post_init__(self) -> None:
        eps = self.eps_p
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise OracleConfigError(f"eps_p must be a number, got {type(eps).__name__}")
        eps = float(eps)
        if not math.isfinite(eps) or not (0.0 < eps < 1.0):
            raise OracleConfigError(f"eps_p must satisfy 0 < eps_p < 1, got {self.eps_p!r}")
        object.__setattr__(self, "eps_p", eps)
        for f in fields(self):
            if f.name == "eps_p":
                continue
            v = getattr(self, f.name)
            if not _is_int(v):
                raise OracleConfigError(f"{f.name} must be an int, got {type(v).__name__} ({v!r})")
        if not (0 <= self.n_grey < self.n_stall <= 8):
            raise OracleConfigError(
                f"require 0 <= n_grey < n_stall <= 8, got n_grey={self.n_grey}, n_stall={self.n_stall}"
            )
        if not (1 <= self.n_rev <= 8):
            raise OracleConfigError(f"require 1 <= n_rev <= 8, got {self.n_rev}")
        if not (0 <= self.k_rec <= 8):
            raise OracleConfigError(f"require 0 <= k_rec <= 8, got {self.k_rec}")
        # Not in the M0 contract's range list, but values outside these make the
        # §6.3 rules degenerate (F >= 0 is always true; negative budgets).
        if self.f_prod < 1:
            raise OracleConfigError(f"require f_prod >= 1, got {self.f_prod}")
        if self.r_max < 0:
            raise OracleConfigError(f"require r_max >= 0, got {self.r_max}")
        if self.p_max < 0:
            raise OracleConfigError(f"require p_max >= 0, got {self.p_max}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PARAMETER_KEYS = frozenset(f.name for f in fields(OracleConfig))

FROZEN_V0_2 = OracleConfig(eps_p=0.02, f_prod=2, n_stall=5, n_grey=3, n_rev=2, r_max=3, p_max=5, k_rec=4)


def _check_keys(where: str, d: Mapping[str, Any], expected: frozenset[str]) -> None:
    keys = set(d)
    missing = sorted(expected - keys)
    extra = sorted(str(k) for k in keys - expected)
    if missing or extra:
        raise OracleConfigError(f"{where}: missing keys {missing}, unexpected keys {extra}")


def parse_oracle_config(data: Any) -> OracleConfig:
    """Validate an already-parsed YAML document and return the config."""
    if not isinstance(data, dict):
        raise OracleConfigError(f"oracle config must be a mapping, got {type(data).__name__}")
    _check_keys("oracle config", data, TOP_LEVEL_KEYS)
    if data["spec_version"] != SPEC_VERSION:
        raise OracleConfigError(
            f"spec_version must be the string {SPEC_VERSION!r} (quote it in YAML), got {data['spec_version']!r}"
        )
    approved = data["approved"]
    if not isinstance(approved, str) or not _DATE_RE.match(approved):
        raise OracleConfigError(f"approved must be a quoted 'YYYY-MM-DD' string, got {approved!r}")
    params = data["parameters"]
    if not isinstance(params, dict):
        raise OracleConfigError(f"parameters must be a mapping, got {type(params).__name__}")
    _check_keys("parameters", params, PARAMETER_KEYS)
    return OracleConfig(**params)


def load_oracle_config(path: str | Path | None = None) -> OracleConfig:
    """Load and validate the oracle config (default: repo-root ``oracle_config.v0_2.yaml``)."""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise OracleConfigError(f"cannot read oracle config {p}: {exc}") from exc
    try:
        data = yaml.load(text, Loader=_UniqueKeySafeLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise OracleConfigError(f"invalid YAML in {p}: {exc}") from exc
    return parse_oracle_config(data)
