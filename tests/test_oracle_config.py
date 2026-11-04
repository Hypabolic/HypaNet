"""Oracle config: frozen values (spec §6.2) and strict loader validation."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
import yaml

from agent_loop_lab.oracle_config import (
    DEFAULT_CONFIG_PATH,
    FROZEN_V0_2,
    OracleConfig,
    OracleConfigError,
    load_oracle_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "agent_doom_loop_detector_spec_v0_2.md"

# Spec §6.2 values, written out by hand so FROZEN_V0_2 and the YAML can't drift together.
SPEC_6_2 = {
    "eps_p": 0.02,
    "f_prod": 2,
    "n_stall": 5,
    "n_grey": 3,
    "n_rev": 2,
    "r_max": 3,
    "p_max": 5,
    "k_rec": 4,
}
SYMBOLS = {
    "ε_p": "eps_p",
    "F_prod": "f_prod",
    "N_stall": "n_stall",
    "N_grey": "n_grey",
    "N_rev": "n_rev",
    "R_max": "r_max",
    "P_max": "p_max",
    "k_rec": "k_rec",
}


def _valid_doc() -> dict:
    return {"spec_version": "0.2", "approved": "2026-11-04", "parameters": dict(SPEC_6_2)}


def _write(tmp_path: Path, doc) -> Path:
    p = tmp_path / "oracle.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=True), encoding="utf-8")
    return p


def _with_params(**overrides) -> dict:
    doc = _valid_doc()
    doc["parameters"].update(overrides)
    return doc


def test_frozen_values_equal_spec_6_2_literals():
    assert dataclasses.asdict(FROZEN_V0_2) == SPEC_6_2


def test_frozen_values_equal_spec_6_2_table():
    """Parse the §6.2 table from the real spec file."""
    text = SPEC_PATH.read_text(encoding="utf-8")
    start = text.index("### 6.2")
    end = text.index("### 6.3")
    table = {}
    for m in re.finditer(r"^\| `([^`]+)` \| ([0-9.]+) \|", text[start:end], re.MULTILINE):
        sym, val = m.group(1), m.group(2)
        table[SYMBOLS[sym]] = float(val) if "." in val else int(val)
    assert table == SPEC_6_2
    assert dataclasses.asdict(FROZEN_V0_2) == table


def test_repo_yaml_loads_to_frozen_values():
    assert DEFAULT_CONFIG_PATH == REPO_ROOT / "oracle_config.v0_2.yaml"
    cfg = load_oracle_config()
    assert cfg == FROZEN_V0_2
    assert dataclasses.asdict(cfg) == SPEC_6_2
    assert load_oracle_config(DEFAULT_CONFIG_PATH) == FROZEN_V0_2
    assert load_oracle_config(str(DEFAULT_CONFIG_PATH)) == FROZEN_V0_2


def test_repo_yaml_records_version_and_approval_date():
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert raw["spec_version"] == "0.2"
    assert raw["approved"] == "2026-11-04"


def test_config_is_frozen_and_typed():
    with pytest.raises(dataclasses.FrozenInstanceError):
        FROZEN_V0_2.n_stall = 6  # type: ignore[misc]
    assert isinstance(FROZEN_V0_2.eps_p, float)
    for name in ("f_prod", "n_stall", "n_grey", "n_rev", "r_max", "p_max", "k_rec"):
        assert type(getattr(FROZEN_V0_2, name)) is int


def test_valid_tmp_file_loads(tmp_path):
    assert load_oracle_config(_write(tmp_path, _valid_doc())) == FROZEN_V0_2


# -- key set ------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(SPEC_6_2))
def test_missing_parameter_rejected(tmp_path, key):
    doc = _valid_doc()
    del doc["parameters"][key]
    with pytest.raises(OracleConfigError, match=key):
        load_oracle_config(_write(tmp_path, doc))


def test_extra_parameter_rejected(tmp_path):
    with pytest.raises(OracleConfigError, match="unexpected"):
        load_oracle_config(_write(tmp_path, _with_params(n_extra=1)))


@pytest.mark.parametrize("key", ["spec_version", "approved", "parameters"])
def test_missing_top_level_key_rejected(tmp_path, key):
    doc = _valid_doc()
    del doc[key]
    with pytest.raises(OracleConfigError, match=key):
        load_oracle_config(_write(tmp_path, doc))


def test_extra_top_level_key_rejected(tmp_path):
    doc = _valid_doc()
    doc["notes"] = "hello"
    with pytest.raises(OracleConfigError, match="unexpected"):
        load_oracle_config(_write(tmp_path, doc))


def test_flat_layout_rejected(tmp_path):
    doc = {"spec_version": "0.2", "approved": "2026-11-04", **SPEC_6_2}
    with pytest.raises(OracleConfigError):
        load_oracle_config(_write(tmp_path, doc))


# -- top-level values -----------------------------------------------------------


@pytest.mark.parametrize("version", ["0.1", "0.3", 0.2, None])
def test_wrong_spec_version_rejected(tmp_path, version):
    doc = _valid_doc()
    doc["spec_version"] = version
    with pytest.raises(OracleConfigError, match="spec_version"):
        load_oracle_config(_write(tmp_path, doc))


def test_unquoted_yaml_values_rejected(tmp_path):
    p = tmp_path / "oracle.yaml"
    params = "\n".join(f"  {k}: {v}" for k, v in SPEC_6_2.items())
    p.write_text(f"spec_version: 0.2\napproved: \"2026-11-04\"\nparameters:\n{params}\n", encoding="utf-8")
    with pytest.raises(OracleConfigError, match="spec_version"):
        load_oracle_config(p)
    p.write_text(f"spec_version: \"0.2\"\napproved: 2026-11-04\nparameters:\n{params}\n", encoding="utf-8")
    with pytest.raises(OracleConfigError, match="approved"):
        load_oracle_config(p)


@pytest.mark.parametrize("approved", ["04/11/2026", "", 20261104])
def test_bad_approval_date_rejected(tmp_path, approved):
    doc = _valid_doc()
    doc["approved"] = approved
    with pytest.raises(OracleConfigError, match="approved"):
        load_oracle_config(_write(tmp_path, doc))


@pytest.mark.parametrize("content", ["", "- 1\n- 2\n", "just a string\n", "spec_version: [\n"])
def test_non_mapping_or_invalid_yaml_rejected(tmp_path, content):
    p = tmp_path / "oracle.yaml"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(OracleConfigError):
        load_oracle_config(p)


def test_parameters_not_a_mapping_rejected(tmp_path):
    doc = _valid_doc()
    doc["parameters"] = [1, 2, 3]
    with pytest.raises(OracleConfigError, match="parameters"):
        load_oracle_config(_write(tmp_path, doc))


def test_missing_file_rejected(tmp_path):
    with pytest.raises(OracleConfigError, match="cannot read"):
        load_oracle_config(tmp_path / "nope.yaml")


# -- types ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,value",
    [
        ("f_prod", 2.0),
        ("n_stall", "5"),
        ("n_grey", True),
        ("n_rev", None),
        ("r_max", 3.5),
        ("p_max", [5]),
        ("k_rec", False),
        ("eps_p", "0.02"),
        ("eps_p", True),
        ("eps_p", None),
    ],
)
def test_bad_types_rejected(tmp_path, key, value):
    with pytest.raises(OracleConfigError, match=key):
        load_oracle_config(_write(tmp_path, _with_params(**{key: value})))


# -- ranges ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"eps_p": 0.0},
        {"eps_p": -0.01},
        {"eps_p": float("nan")},
        {"eps_p": float("inf")},
        {"eps_p": 1.0},
        {"n_grey": -1},
        {"n_grey": 5},  # == n_stall
        {"n_grey": 6},  # > n_stall
        {"n_stall": 9, "n_grey": 3},
        {"n_stall": 0, "n_grey": 0},
        {"n_rev": 0},
        {"n_rev": 9},
        {"k_rec": -1},
        {"k_rec": 9},
        {"f_prod": 0},
        {"r_max": -1},
        {"p_max": -1},
    ],
)
def test_out_of_range_rejected(tmp_path, overrides):
    with pytest.raises(OracleConfigError):
        load_oracle_config(_write(tmp_path, _with_params(**overrides)))


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_grey": 0, "n_stall": 1},
        {"n_grey": 7, "n_stall": 8},
        {"n_rev": 1},
        {"n_rev": 8},
        {"k_rec": 0},
        {"k_rec": 8},
        {"eps_p": 0.5},
    ],
)
def test_range_boundaries_accepted(tmp_path, overrides):
    cfg = load_oracle_config(_write(tmp_path, _with_params(**overrides)))
    for k, v in overrides.items():
        assert getattr(cfg, k) == v


def test_constructor_validates_too():
    with pytest.raises(OracleConfigError):
        OracleConfig(eps_p=0.02, f_prod=2, n_stall=5, n_grey=5, n_rev=2, r_max=3, p_max=5, k_rec=4)
    assert isinstance(OracleConfigError("x"), ValueError)


def test_duplicate_parameter_key_rejected(tmp_path):
    """safe_load keeps the last duplicate; the loader must refuse it instead."""
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").rstrip("\n") + "\n  eps_p: 0.5\n"
    p = tmp_path / "dup.yaml"
    p.write_text(text, encoding="utf-8")
    assert yaml.safe_load(text)["parameters"]["eps_p"] == 0.5  # what plain safe_load would do
    with pytest.raises(OracleConfigError, match="duplicate key 'eps_p'"):
        load_oracle_config(p)


def test_duplicate_top_level_key_rejected(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8") + 'spec_version: "0.2"\n', encoding="utf-8")
    with pytest.raises(OracleConfigError, match="duplicate key 'spec_version'"):
        load_oracle_config(p)
