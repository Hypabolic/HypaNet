"""Freeze registry: deterministic freeze, packet hash, mismatch detection, repo-level check."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

import agent_loop_lab.packet as packet_mod
from agent_loop_lab.packet import build_packet
from agent_loop_lab.registry import (
    FROZEN_FILES,
    PACKET_KEYS,
    REGISTRY_KEYS,
    REGISTRY_RELPATH,
    RegistryError,
    check_registry,
    compute_freeze,
    freeze_digest,
    packet_hashes,
    write_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = "agent_doom_loop_detector_spec_v0_2.md"
MODEL = "src/agent_loop_lab/model.py"

PYPROJECT = "pyproject.toml"
UV_LOCK = "uv.lock"

# Real copies: the packet sources (spec, model), the environment pins and files
# whose content is settled. The rest, including runner.py and oracle_config.py
# (edited elsewhere), get deterministic placeholders. Bytes are snapshotted once
# per test session so a concurrent edit in the working tree can't make two
# roots differ.
_COPY = (
    SPEC,
    PYPROJECT,
    UV_LOCK,
    "oracle_config.v0_2.yaml",
    "src/agent_loop_lab/schema.py",
    MODEL,
    "src/agent_loop_lab/packet.py",
    "src/agent_loop_lab/guards.py",
)


@pytest.fixture(scope="session")
def frozen_bytes() -> dict[str, bytes]:
    out = {}
    for rel in FROZEN_FILES:
        if rel in _COPY:
            out[rel] = (REPO_ROOT / rel).read_bytes()
        else:
            out[rel] = f'"""placeholder for {rel}"""\n'.encode("utf-8")
    return out


@pytest.fixture(autouse=True)
def _temp_under_tmp_path(tmp_path, monkeypatch):
    """Keep the registry's temporary packet builds inside pytest's tmp_path."""
    d = tmp_path / "systmp"
    d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    return d


def _make_root(root: Path, frozen: dict[str, bytes]) -> Path:
    for rel, data in frozen.items():
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    return root


@pytest.fixture
def make_root(frozen_bytes):
    return lambda root: _make_root(root, frozen_bytes)


@pytest.fixture
def root(tmp_path, make_root) -> Path:
    return make_root(tmp_path / "repo")


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _non_packet(problems: list[str]) -> list[str]:
    return [p for p in problems if not p.startswith("authoring packet")]


def _rewrite(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


# -- contract -------------------------------------------------------------------


def test_frozen_files_contract():
    assert FROZEN_FILES == (
        "agent_doom_loop_detector_spec_v0_2.md",
        "pyproject.toml",
        "uv.lock",
        "oracle_config.v0_2.yaml",
        "src/agent_loop_lab/oracle_config.py",
        "src/agent_loop_lab/schema.py",
        "src/agent_loop_lab/fingerprint.py",
        "src/agent_loop_lab/encode.py",
        "src/agent_loop_lab/windows.py",
        "src/agent_loop_lab/model.py",
        "src/agent_loop_lab/packet.py",
        "src/agent_loop_lab/guards.py",
        "src/agent_loop_lab/runner.py",
    )
    assert REGISTRY_RELPATH == "experiments/registry.yaml"
    assert PACKET_KEYS == ("packet_sha256", "packet_source_spec_sha256")
    assert REGISTRY_KEYS == {"spec_version", "files", "packet_sha256", "packet_source_spec_sha256", "freeze_sha256"}


# -- compute_freeze -------------------------------------------------------------


def test_compute_freeze_is_deterministic(root, tmp_path, make_root):
    f1 = compute_freeze(root)
    f2 = compute_freeze(root)
    f3 = compute_freeze(make_root(tmp_path / "other"))
    assert f1 == f2 == f3
    assert set(f1) == REGISTRY_KEYS
    assert f1["spec_version"] == "0.2"
    assert list(f1["files"]) == list(FROZEN_FILES)
    for rel, sha in f1["files"].items():
        assert sha == hashlib.sha256((root / rel).read_bytes()).hexdigest()


def test_compute_freeze_records_packet_hash(root, tmp_path):
    record = compute_freeze(root)
    manifest = build_packet(root / SPEC, root / MODEL, tmp_path / "packet")
    assert record["packet_sha256"] == manifest["packet_sha256"]
    assert record["packet_source_spec_sha256"] == manifest["source_spec_sha256"]
    assert record["packet_source_spec_sha256"] == record["files"][SPEC]
    assert packet_hashes(root) == {k: record[k] for k in PACKET_KEYS}


def test_freeze_sha256_covers_full_record(root):
    record = compute_freeze(root)
    body = {k: v for k, v in record.items() if k != "freeze_sha256"}
    expected = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert record["freeze_sha256"] == freeze_digest(record) == freeze_digest(body) == expected
    for key in ("spec_version", *PACKET_KEYS):
        changed = dict(record, **{key: "x"})
        assert freeze_digest(changed) != record["freeze_sha256"], key
    changed = dict(record, files=dict(record["files"], **{SPEC: "0" * 64}))
    assert freeze_digest(changed) != record["freeze_sha256"]
    # Key order of the record is irrelevant.
    assert freeze_digest(dict(reversed(list(record.items())))) == record["freeze_sha256"]


def test_compute_freeze_changes_with_content(root):
    before = compute_freeze(root)
    p = root / "src/agent_loop_lab/encode.py"
    p.write_bytes(p.read_bytes() + b" ")
    after = compute_freeze(root)
    assert after["freeze_sha256"] != before["freeze_sha256"]
    changed = {k for k in FROZEN_FILES if before["files"][k] != after["files"][k]}
    assert changed == {"src/agent_loop_lab/encode.py"}
    assert {k: after[k] for k in PACKET_KEYS} == {k: before[k] for k in PACKET_KEYS}


def test_packet_hash_follows_packet_sources(root):
    before = compute_freeze(root)
    m = root / MODEL
    m.write_bytes(m.read_bytes() + b"# extra comment\n")
    after = compute_freeze(root)
    assert after["packet_sha256"] != before["packet_sha256"]
    assert after["packet_source_spec_sha256"] == before["packet_source_spec_sha256"]
    s = root / SPEC
    s.write_bytes(s.read_bytes() + b"\nAn extra trailing paragraph.\n")
    after2 = compute_freeze(root)
    assert after2["packet_sha256"] != after["packet_sha256"]
    assert after2["packet_source_spec_sha256"] == hashlib.sha256(s.read_bytes()).hexdigest()


def test_compute_freeze_writes_nothing_under_root(root):
    before = _tree(root)
    compute_freeze(root)
    assert _tree(root) == before


@pytest.mark.parametrize("rel", ["src/agent_loop_lab/windows.py", "src/agent_loop_lab/runner.py"])
def test_compute_freeze_missing_file_raises(root, rel):
    (root / rel).unlink()
    with pytest.raises(FileNotFoundError, match=Path(rel).name):
        compute_freeze(root)


def test_compute_freeze_unbuildable_packet_raises(root):
    # A model module that can't be made standalone for the packet.
    (root / MODEL).write_text("from .encode import encode_window\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="authoring packet could not be built"):
        compute_freeze(root)
    with pytest.raises(RegistryError, match="authoring packet could not be built"):
        write_registry(root)
    assert not (root / REGISTRY_RELPATH).exists()


def test_compute_freeze_default_root_is_cwd(root, monkeypatch):
    monkeypatch.chdir(root)
    assert compute_freeze() == compute_freeze(root)


# -- write / check --------------------------------------------------------------


def test_write_then_check_ok(root):
    path = write_registry(root)
    assert path == root / "experiments" / "registry.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data == compute_freeze(root)
    assert set(data) == REGISTRY_KEYS
    assert check_registry(root) == []
    # Writing an identical registry again is a no-op.
    before = path.read_bytes()
    assert write_registry(root) == path
    assert path.read_bytes() == before
    # Checking (which rebuilds the packet) writes nothing under root.
    tree = _tree(root)
    assert check_registry(root) == []
    assert _tree(root) == tree


def test_check_detects_modified_frozen_file(root):
    write_registry(root)
    p = root / MODEL
    p.write_bytes(p.read_bytes().replace(b"HIDDEN1 = 24", b"HIDDEN1 = 25"))
    problems = check_registry(root)
    assert len(problems) == 2, problems
    assert "modified" in problems[0] and MODEL in problems[0]
    assert problems[1].startswith("authoring packet packet_sha256 differs")


@pytest.mark.parametrize("rel", FROZEN_FILES)
def test_check_detects_one_byte_change_in_each_frozen_file(root, rel):
    write_registry(root)
    p = root / rel
    p.write_bytes(p.read_bytes() + b"\n")
    problems = check_registry(root)
    assert any("modified" in x and rel in x for x in problems), problems
    assert all(rel in x for x in _non_packet(problems)), problems
    # Only a packet source can move the packet hashes.
    if rel not in (SPEC, MODEL):
        assert _non_packet(problems) == problems


def _bump_torch_pin(text: str, field: str) -> str:
    if field == "lock":  # the resolved torch entry in uv.lock
        old = 'name = "torch"\nversion = "2.2.2"\n'
        new = 'name = "torch"\nversion = "2.2.3"\n'
    else:  # the requirement in pyproject.toml
        old, new = '"torch==2.2.2"', '"torch==2.2.3"'
    assert text.count(old) == 1, f"expected exactly one torch pin in the {field} copy"
    return text.replace(old, new)


@pytest.mark.parametrize(("rel", "field"), [(UV_LOCK, "lock"), (PYPROJECT, "requirement")])
def test_check_detects_modified_environment_pin(root, rel, field):
    """[R3] A changed environment pin (uv.lock or pyproject.toml) is a frozen-file change."""
    write_registry(root)
    before = compute_freeze(root)
    p = root / rel
    p.write_text(_bump_torch_pin(p.read_text(encoding="utf-8"), field), encoding="utf-8")
    problems = check_registry(root)
    assert len(problems) == 1, problems
    assert problems[0].startswith(f"frozen file modified: {rel} (registry {before['files'][rel]}, now ")
    after = compute_freeze(root)
    assert {k: after[k] for k in PACKET_KEYS} == {k: before[k] for k in PACKET_KEYS}
    assert {k for k in FROZEN_FILES if before["files"][k] != after["files"][k]} == {rel}


def test_check_detects_modified_oracle_loader(root):
    """[R3] The oracle-config loader is frozen alongside the YAML it validates."""
    rel = "src/agent_loop_lab/oracle_config.py"
    write_registry(root)
    p = root / rel
    p.write_bytes(p.read_bytes() + b"# drift\n")
    problems = check_registry(root)
    assert len(problems) == 1 and problems[0].startswith(f"frozen file modified: {rel}"), problems


@pytest.mark.parametrize("rel", ["pyproject.toml", "uv.lock", "src/agent_loop_lab/oracle_config.py"])
def test_compute_freeze_requires_new_frozen_files(root, rel):
    (root / rel).unlink()
    with pytest.raises(FileNotFoundError, match=Path(rel).name.replace(".", r"\.")):
        compute_freeze(root)


def test_repo_frozen_files_present_and_lf():
    """Every frozen file exists in the working tree and is LF only, so the M0 freeze can be written."""
    for rel in FROZEN_FILES:
        f = REPO_ROOT / rel
        assert f.is_file(), rel
        assert b"\r" not in f.read_bytes(), rel


@pytest.mark.parametrize("rel", ["src/agent_loop_lab/packet.py", "src/agent_loop_lab/guards.py"])
def test_check_detects_modified_packet_builder_and_guard(root, rel):
    write_registry(root)
    p = root / rel
    p.write_bytes(p.read_bytes().replace(b"def ", b"def  ", 1))
    problems = check_registry(root)
    assert len(problems) == 1, problems
    assert problems[0].startswith(f"frozen file modified: {rel}")


def test_check_rebuilds_packet_and_detects_builder_drift(root, monkeypatch):
    """A change in what the imported builder produces is caught even if no frozen file changed."""
    write_registry(root)
    monkeypatch.setattr(packet_mod, "AUTHOR_INSTRUCTIONS", packet_mod.AUTHOR_INSTRUCTIONS + "\nOne more line.\n")
    problems = check_registry(root)
    assert len(problems) == 1, problems
    assert problems[0].startswith("authoring packet packet_sha256 differs")


def test_check_detects_tampered_packet_hash(root):
    path = write_registry(root)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["packet_sha256"] = "0" * 64
    _rewrite(path, data)
    problems = check_registry(root)
    assert any(x.startswith("authoring packet packet_sha256 differs") for x in problems), problems
    assert any("freeze_sha256" in x for x in problems), problems
    assert len(problems) == 2, problems


def test_check_detects_tampered_packet_source_spec_hash(root):
    path = write_registry(root)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["packet_source_spec_sha256"] = "1" * 64
    data["freeze_sha256"] = freeze_digest(data)  # a consistent forgery is still caught by the rebuild
    _rewrite(path, data)
    problems = check_registry(root)
    assert len(problems) == 1, problems
    assert problems[0].startswith("authoring packet packet_source_spec_sha256 differs")


def test_check_detects_missing_packet_keys(root):
    path = write_registry(root)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    del data["packet_sha256"]
    _rewrite(path, data)
    problems = check_registry(root)
    assert any("registry keys" in x for x in problems), problems
    assert any("'packet_sha256' must be a sha256 string" in x for x in problems), problems


def test_check_reports_unbuildable_packet(root):
    write_registry(root)
    (root / MODEL).write_text("from .encode import encode_window\n", encoding="utf-8")
    problems = check_registry(root)
    assert any("modified" in x and MODEL in x for x in problems), problems
    assert any(x.startswith("authoring packet could not be built") for x in problems), problems


def test_check_detects_missing_frozen_file(root):
    write_registry(root)
    (root / "oracle_config.v0_2.yaml").unlink()
    problems = check_registry(root)
    assert problems == [problems[0]] and "missing" in problems[0] and "oracle_config.v0_2.yaml" in problems[0]


def test_check_missing_packet_source_skips_rebuild(root):
    write_registry(root)
    (root / MODEL).unlink()
    problems = check_registry(root)
    assert problems == [f"frozen file missing: {MODEL}"], problems


def test_check_detects_tampered_registry(root):
    path = write_registry(root)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["files"]["src/agent_loop_lab/schema.py"] = "0" * 64
    _rewrite(path, data)
    problems = check_registry(root)
    assert any("schema.py" in x for x in problems)
    assert any("freeze_sha256" in x for x in problems)


def test_check_detects_registry_missing_entry_and_wrong_version(root):
    path = write_registry(root)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    del data["files"]["src/agent_loop_lab/runner.py"]
    data["spec_version"] = "0.1"
    _rewrite(path, data)
    problems = check_registry(root)
    assert any("not in registry" in x and "runner.py" in x for x in problems)
    assert any("spec_version" in x for x in problems)


def test_check_without_registry_reports_problem(root):
    problems = check_registry(root)
    assert len(problems) == 1 and "not found" in problems[0]


def test_check_malformed_registry(root):
    (root / "experiments").mkdir()
    (root / REGISTRY_RELPATH).write_text("- just\n- a list\n", encoding="utf-8")
    assert check_registry(root) == ["registry is not a mapping"]


def test_write_refuses_to_move_a_freeze(root):
    path = write_registry(root)
    original = path.read_bytes()
    p = root / "src/agent_loop_lab/fingerprint.py"
    p.write_text('"""changed"""\n', encoding="utf-8")
    with pytest.raises(RegistryError, match="overwrite"):
        write_registry(root)
    assert path.read_bytes() == original
    write_registry(root, overwrite=True)
    assert check_registry(root) == []


def test_freeze_script(root, _temp_under_tmp_path):
    script = REPO_ROOT / "scripts" / "freeze.py"
    env = {**os.environ, "TMPDIR": str(_temp_under_tmp_path)}

    def run(*args):
        return subprocess.run(
            [sys.executable, str(script), "--root", str(root), *args], capture_output=True, text=True, env=env
        )

    # Compared with the subprocess's own output, not with this process's
    # compute_freeze: the child imports packet.py from disk afresh.
    r = run()
    assert r.returncode == 0, r.stderr
    record = json.loads(r.stdout)
    assert set(record) == REGISTRY_KEYS
    assert not (root / REGISTRY_RELPATH).exists(), "printing must not write"
    assert run("--check").returncode == 1
    r = run("--write")
    assert r.returncode == 0, r.stderr
    assert yaml.safe_load((root / REGISTRY_RELPATH).read_text(encoding="utf-8")) == record
    assert f"packet sha256: {record['packet_sha256']}" in r.stdout
    assert f"freeze sha256: {record['freeze_sha256']}" in r.stdout
    assert run("--check").returncode == 0
    (root / "src/agent_loop_lab/windows.py").write_text("changed\n", encoding="utf-8")
    r = run("--check")
    assert r.returncode == 1 and "windows.py" in r.stderr
    assert run("--write").returncode == 1
    assert run("--write", "--force").returncode == 0
    assert run("--check").returncode == 0
    (root / "src/agent_loop_lab/runner.py").unlink()
    r = run()
    assert r.returncode == 1 and "runner.py" in r.stderr


def test_repo_registry_matches():
    """Armed once experiments/registry.yaml is written at the end of M0."""
    if not (REPO_ROOT / REGISTRY_RELPATH).exists():
        pytest.skip("experiments/registry.yaml not written yet (written at M0 sign-off)")
    assert check_registry(REPO_ROOT) == []


# -- line endings ---------------------------------------------------------------


def _to_crlf(root: Path) -> None:
    for rel in FROZEN_FILES:
        f = root / rel
        f.write_bytes(f.read_bytes().replace(b"\n", b"\r\n"))


def test_compute_freeze_refuses_crlf(root):
    p = root / "oracle_config.v0_2.yaml"
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(RegistryError, match="CR line endings"):
        compute_freeze(root)
    with pytest.raises(RegistryError, match="CR line endings"):
        write_registry(root)


def test_check_names_crlf_checkout(root):
    """A CRLF checkout (e.g. core.autocrlf=true) is reported as such, not as a content change."""
    write_registry(root)
    _to_crlf(root)
    problems = check_registry(root)
    assert len(problems) == len(FROZEN_FILES), problems
    assert all("CRLF line endings" in x for x in problems), problems
    assert not any("modified" in x for x in problems)


def test_gitattributes_pins_lf():
    text = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    lines = [line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert ["*", "text=auto", "eol=lf"] in lines
    assert ["*.dlw", "binary"] in lines
