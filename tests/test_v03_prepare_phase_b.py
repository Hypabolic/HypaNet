"""scripts/v03_prepare_phase_b.sh: Phase B disclosure handover (experiments/v0_3/phase_b/README.md §2).

The script is copied into a fake repo under tmp_path (it locates the repo from its own path), with a fake
author directory next to it. The real repo and the real author directory are never touched.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "v03_prepare_phase_b.sh"
P_SHA_LINE = "packet_sha256: {}\n"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _make_writable(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            if not p.is_symlink():
                p.chmod(p.stat().st_mode | stat.S_IWUSR | stat.S_IRUSR | (stat.S_IXUSR if p.is_dir() else 0))


@pytest.fixture
def env(tmp_path):
    repo, author = tmp_path / "HypaNet", tmp_path / "HypaNet-v03-author"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, repo / "scripts" / SCRIPT.name)
    disc = repo / "experiments" / "v0_3" / "disclosure"
    disc.mkdir(parents=True)
    (disc / "DISCLOSURE.json").write_text('{"windows": []}\n')
    (disc / "DISCLOSURE.sha256").write_text(f"{_sha(disc / 'DISCLOSURE.json')}  DISCLOSURE.json\n")
    (repo / "experiments" / "v0_3" / "phase_b").mkdir()
    kit = repo / "experiments" / "v0_3" / "author_kit"
    kit.mkdir()
    (kit / "settings_v03_phase_b.json").write_text("{}\n")
    arch = repo / "weight_program" / "v0_3" / "phase-a"
    arch.mkdir(parents=True)
    (arch / "artifact.dlw").write_bytes(b"artefact")
    (arch / "artifact.sha256").write_text(f"{_sha(arch / 'artifact.dlw')}  artifact.dlw\n")
    (author / "packet").mkdir(parents=True)
    (author / "packet" / "PACKET_MANIFEST.json").write_text('{"packet": 1}\n')
    (repo / "experiments" / "registry_v0_3.yaml").write_text(
        "files: {}\n" + P_SHA_LINE.format(_sha(author / "packet" / "PACKET_MANIFEST.json")))
    (author / "tools").mkdir()
    (author / "tools" / "launch.py").write_text("")
    (author / "py").write_text("#!/bin/sh\n")
    (author / "py").chmod(0o755)
    (author / "output" / "v03-phase-a").mkdir(parents=True)
    (author / "output" / "v03-phase-a" / "artifact.dlw").write_bytes(b"artefact")
    yield repo, author
    _make_writable(tmp_path)


def _run(repo: Path, author: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["sh", str(repo / "scripts" / SCRIPT.name), str(author)],
                          capture_output=True, text=True, timeout=60)


def test_syntax():
    assert subprocess.run(["sh", "-n", str(SCRIPT)], capture_output=True).returncode == 0


def test_hands_over_disclosure(env):
    repo, author = env
    r = _run(repo, author)
    assert r.returncode == 0, r.stderr
    src, dst = repo / "experiments" / "v0_3" / "disclosure", author / "disclosure"
    assert sorted(p.name for p in dst.iterdir()) == ["DISCLOSURE.json", "DISCLOSURE.sha256"]
    for name in ("DISCLOSURE.json", "DISCLOSURE.sha256"):
        assert (dst / name).read_bytes() == (src / name).read_bytes()
        assert stat.S_IMODE((dst / name).stat().st_mode) == 0o444
    assert stat.S_IMODE(dst.stat().st_mode) == 0o555
    for d in ("output/v03-phase-a", "tools", "packet"):
        for p in [author / d, *(author / d).rglob("*")]:
            assert not p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH), p
    assert (author / "output" / "v03-phase-b").is_dir()
    rec = (repo / "experiments" / "v0_3" / "phase_b" / "HANDOVER.json").read_text()
    assert _sha(src / "DISCLOSURE.json") in rec and '"prompt_v03_phase_b_sha256": "missing"' in rec
    # Idempotent when the copy is byte-identical.
    r2 = _run(repo, author)
    assert r2.returncode == 0, r2.stderr
    assert "already handed over" in r2.stdout


def test_refuses_bad_disclosure_sha(env):
    repo, author = env
    (repo / "experiments" / "v0_3" / "disclosure" / "DISCLOSURE.sha256").write_text("0" * 64 + "  DISCLOSURE.json\n")
    r = _run(repo, author)
    assert r.returncode == 2 and "does not match" in r.stderr
    assert not (author / "disclosure").exists()


def test_refuses_missing_disclosure(env):
    repo, author = env
    shutil.rmtree(repo / "experiments" / "v0_3" / "disclosure")
    r = _run(repo, author)
    assert r.returncode == 2 and "DISCLOSURE.json missing" in r.stderr


def test_refuses_bad_artifact_and_packet(env):
    repo, author = env
    (repo / "weight_program" / "v0_3" / "phase-a" / "artifact.dlw").write_bytes(b"other")
    r = _run(repo, author)
    assert r.returncode == 2 and "artifact.dlw does not match" in r.stderr
    (repo / "weight_program" / "v0_3" / "phase-a" / "artifact.dlw").write_bytes(b"artefact")
    (author / "packet" / "PACKET_MANIFEST.json").write_text("{}\n")
    r = _run(repo, author)
    assert r.returncode == 2 and "packet_sha256" in r.stderr
    assert not (author / "disclosure").exists()


def test_refuses_differing_or_extra_copy(env):
    repo, author = env
    (author / "disclosure").mkdir()
    (author / "disclosure" / "DISCLOSURE.json").write_text("{}\n")
    shutil.copy(repo / "experiments" / "v0_3" / "disclosure" / "DISCLOSURE.sha256", author / "disclosure")
    r = _run(repo, author)
    assert r.returncode == 2 and "differs" in r.stderr
    shutil.copy(repo / "experiments" / "v0_3" / "disclosure" / "DISCLOSURE.json", author / "disclosure")
    (author / "disclosure" / "DISCLOSURE_PRIVATE.json").write_text("{}\n")
    r = _run(repo, author)
    assert r.returncode == 2 and "unexpected file" in r.stderr


def test_refuses_nonempty_phase_b_output(env):
    repo, author = env
    (author / "output" / "v03-phase-b").mkdir()
    (author / "output" / "v03-phase-b" / "x").write_text("")
    r = _run(repo, author)
    assert r.returncode == 2 and "not empty" in r.stderr


def test_refuses_author_dir_inside_repo(env):
    repo, author = env
    inside = repo / "author"
    shutil.copytree(author, inside)
    r = _run(repo, inside)
    assert r.returncode == 2 and "outside the repo" in r.stderr
