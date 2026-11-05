"""v0.4 freeze registry (``agent_loop_lab.v04.registry``, ``scripts/v04_freeze.py``).

Every test that writes runs on a fake root in a temporary directory: the required files are
placeholders, except the three files the registry parses or executes (``hypaqwen/__init__.py``,
``interface.py``, ``model.py``), which are copied from the repo, and a fake packet builder.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from agent_loop_lab.v04 import registry as R

REPO = Path(__file__).resolve().parents[1]
_FAKE_BUILDER = textwrap.dedent('''\
    import argparse, hashlib, json, pathlib
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True)
    spec = pathlib.Path("agent_doom_loop_detector_spec_v0_4.md").read_bytes()
    body = json.dumps({"format": "fake-packet", "spec": hashlib.sha256(spec).hexdigest()}, sort_keys=True).encode()
    (out / "PACKET_MANIFEST.json").write_bytes(body)
    print("packet sha256: " + hashlib.sha256(body).hexdigest())
''')
_REAL = (R.PKG_INIT, R.INTERFACE_FILE, R.MODEL_FILE)


def _load_script():
    spec = importlib.util.spec_from_file_location("v04_freeze_script", REPO / "scripts" / "v04_freeze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _put(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    base = tmp_path / "repo"
    for rel in R.REQUIRED_FILES:
        _put(base, rel, f"placeholder {rel}\n")
    for rel in _REAL:
        (base / rel).write_bytes((REPO / rel).read_bytes())
    _put(base, R.PACKET_BUILDER, _FAKE_BUILDER)
    for rel in ("envs/qwen/hypaqwen/wsp/server.py", "envs/qwen/hypaqwen/wsp/author/client.py",
                "envs/qwen/hypaqwen/wsp/author/AUTHOR_VENV.md", "experiments/v0_4/author_kit/prompt_wsp_a.md",
                "experiments/v0_4/author_kit/hooks/bash_hook.py", "scripts/v04_launch.sh",
                "src/agent_loop_lab/v04/registry.py"):
        _put(base, rel, f"# {rel}\n")
    _put(base, "envs/qwen/hypaqwen/wsp/__pycache__/server.cpython-312.pyc", "junk")
    return base


def test_write_then_check_ok(root: Path) -> None:
    path, rec = R.write_registry(root)
    assert path == root / R.REGISTRY_RELPATH
    assert R.check_registry(root) == []
    files = rec["files"]
    assert "experiments/v0_4/author_kit/hooks/bash_hook.py" in files
    assert "envs/qwen/hypaqwen/wsp/author/AUTHOR_VENV.md" in files
    assert not any("__pycache__" in f for f in files)
    assert rec["interface_spec_sha256"] == R.INTERFACE_SPEC_SHA256
    assert rec["model"]["revision"].startswith("2fc06364")
    assert set(rec["model"]["file_sha256"]) >= {"model.safetensors-00001-of-00001.safetensors", "tokenizer.json"}
    assert rec["freeze_sha256"] == R.freeze_digest(rec)


def test_write_is_deterministic(root: Path, tmp_path: Path) -> None:
    R.write_registry(root)
    first = (root / R.REGISTRY_RELPATH).read_bytes()
    R.write_registry(root, overwrite=True)
    assert (root / R.REGISTRY_RELPATH).read_bytes() == first
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    (copy / R.REGISTRY_RELPATH).unlink()
    R.write_registry(copy)
    assert (copy / R.REGISTRY_RELPATH).read_bytes() == first


def test_write_refuses_existing_without_force(root: Path) -> None:
    R.write_registry(root)
    with pytest.raises(R.RegistryError, match="exists"):
        R.write_registry(root)
    assert _load_script().main(["--root", str(root), "--write"]) == 1
    assert _load_script().main(["--root", str(root), "--write", "--force"]) == 0


def test_check_catches_changed_file(root: Path, capsys) -> None:
    R.write_registry(root)
    (root / "envs/qwen/hypaqwen/wsp/server.py").write_text("# changed\n")
    problems = R.check_registry(root)
    assert problems == ["frozen file modified: envs/qwen/hypaqwen/wsp/server.py"]
    assert _load_script().main(["--root", str(root), "--check"]) == 1
    assert "MISMATCH: frozen file modified: envs/qwen/hypaqwen/wsp/server.py" in capsys.readouterr().err


def test_check_catches_missing_file(root: Path) -> None:
    R.write_registry(root)
    (root / "experiments/v0_4/PREREGISTRATION.md").unlink()
    (root / "experiments/v0_4/author_kit/prompt_wsp_a.md").unlink()
    problems = R.check_registry(root)
    assert "frozen file missing: experiments/v0_4/PREREGISTRATION.md" in problems
    assert "frozen file missing: experiments/v0_4/author_kit/prompt_wsp_a.md" in problems


def test_check_catches_added_file(root: Path) -> None:
    R.write_registry(root)
    _put(root, "scripts/v04_sealed_eval.py", "# written after G1\n")
    assert R.check_registry(root) == ["not in registry: scripts/v04_sealed_eval.py"]


def test_check_catches_registry_edit(root: Path) -> None:
    R.write_registry(root)
    path = root / R.REGISTRY_RELPATH
    data = yaml.safe_load(path.read_text())
    data["files"]["docs/V04_WSP_KERNEL.md"] = "0" * 64
    path.write_text(R.render(data))
    problems = R.check_registry(root)
    assert "frozen file modified: docs/V04_WSP_KERNEL.md" in problems
    assert "registry freeze_sha256 does not match its record" in problems


def test_check_catches_freeze_digest_and_packet_edits(root: Path) -> None:
    R.write_registry(root)
    path = root / R.REGISTRY_RELPATH
    data = yaml.safe_load(path.read_text())
    data["packet_sha256"] = "f" * 64
    data["freeze_sha256"] = R.freeze_digest(data)  # a consistent forgery still fails the rebuild
    path.write_text(R.render(data))
    assert any(p.startswith("packet_sha256 differs") for p in R.check_registry(root))
    data["model"]["revision"] = "deadbeef"
    data["freeze_sha256"] = "0" * 64
    path.write_text(R.render(data))
    problems = R.check_registry(root)
    assert any(p.startswith("model differs") for p in problems)
    assert "registry freeze_sha256 does not match its record" in problems


def test_check_catches_packet_change(root: Path) -> None:
    R.write_registry(root)
    (root / "agent_doom_loop_detector_spec_v0_4.md").write_text("changed spec\n")
    problems = R.check_registry(root)
    assert "frozen file modified: agent_doom_loop_detector_spec_v0_4.md" in problems
    assert any(p.startswith("packet_sha256 differs") for p in problems)


def test_check_without_registry(root: Path) -> None:
    assert R.check_registry(root)[0].startswith("registry not found")
    assert _load_script().main(["--root", str(root), "--check"]) == 1


def test_write_refuses_without_packet_builder(root: Path, capsys) -> None:
    (root / R.PACKET_BUILDER).unlink()
    with pytest.raises(R.RegistryError, match="v04_build_packet.py"):
        R.write_registry(root)
    assert not (root / R.REGISTRY_RELPATH).exists()
    assert _load_script().main(["--root", str(root), "--write"]) == 1
    assert "frozen file missing: scripts/v04_build_packet.py" in capsys.readouterr().err


def test_write_refuses_lying_packet_builder(root: Path) -> None:
    _put(root, R.PACKET_BUILDER, _FAKE_BUILDER.replace('hashlib.sha256(body).hexdigest()', '"a" * 64'))
    with pytest.raises(R.RegistryError, match="hashes to"):
        R.write_registry(root)


def test_write_refuses_empty_glob(root: Path) -> None:
    (root / "scripts/v04_launch.sh").unlink()
    with pytest.raises(R.RegistryError, match=r"glob matches no file: scripts/v04_\*\.sh"):
        R.write_registry(root)


def test_write_refuses_other_interface(root: Path) -> None:
    p = root / R.INTERFACE_FILE
    p.write_text(p.read_text().replace('"State:", "full")', '"Answer:", "full")', 1))
    with pytest.raises(R.RegistryError, match="pre-registered"):
        R.write_registry(root)


def test_write_refuses_symlink(root: Path, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.py"
    target.write_text("# outside\n")
    (root / "scripts/v04_extra.py").symlink_to(target)
    with pytest.raises(R.RegistryError, match="symlink"):
        R.write_registry(root)


def test_repo_pins_and_globs() -> None:
    """On the real tree: the interface hash is the pre-registered one, the model pins parse, and the
    globs cover every file the G1 task names. Nothing is written."""
    assert R.interface_spec_sha256(REPO) == R.INTERFACE_SPEC_SHA256
    pins = R.model_pins(REPO)
    assert pins["id"] == "Qwen/Qwen3.5-0.8B" and pins["revision"].startswith("2fc06364")
    files, _ = R.frozen_files(REPO)
    for rel in ("envs/qwen/hypaqwen/interface.py", "envs/qwen/hypaqwen/readout.py", "envs/qwen/hypaqwen/model.py",
                "envs/qwen/hypaqwen/data.py", "envs/qwen/hypaqwen/run_readout.py",
                "envs/qwen/hypaqwen/wsp/server.py", "envs/qwen/hypaqwen/wsp/author/hn_guard/guards_v04.py",
                "envs/qwen/scripts/make_author_venv.sh", "envs/qwen/uv.lock", "docs/V04_WSP_KERNEL.md",
                "experiments/v0_4/PREREGISTRATION.md", "scripts/v04_freeze.py", "src/agent_loop_lab/v04/registry.py",
                "src/agent_loop_lab/v04/scoring.py", "experiments/v0_4/fresh_test/REGISTRATION.json",
                "artifacts/reports/v04_prompt_choice_revision1.json"):
        assert rel in files, rel
    assert "_hn_v04_frozen_interface" not in sys.modules
