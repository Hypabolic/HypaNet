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


# One file per required component (their real names belong to the components).
_COMPONENT_FILES = (
    "scripts/v04_build_disclosure.py",
    "scripts/v04_audit_session.py",
    "experiments/v0_4/author_kit/prompt_wsp_b.md",
    "experiments/v0_4/author_kit/hook_bash.py",
)


def _placeholder(rel: str) -> str:
    return f"# placeholder {rel}\n"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    base = tmp_path / "repo"
    for rel in R.REQUIRED_FILES + _COMPONENT_FILES:
        _put(base, rel, _placeholder(rel))
    for rel in _REAL:
        (base / rel).write_bytes((REPO / rel).read_bytes())
    _put(base, R.PACKET_BUILDER, _FAKE_BUILDER)
    for rel in ("envs/qwen/hypaqwen/wsp/server.py", "envs/qwen/hypaqwen/wsp/author/client.py",
                "envs/qwen/hypaqwen/wsp/author/AUTHOR_VENV.md", "src/agent_loop_lab/v04/registry.py"):
        _put(base, rel, f"# {rel}\n")
    _put(base, "envs/qwen/hypaqwen/wsp/__pycache__/server.cpython-312.pyc", "junk")
    _put(base, "envs/qwen/.venv/lib/site.py", "# never frozen\n")
    return base


def _check(root: Path) -> list[str]:
    return R.check_registry(root, check_v03=False)


def test_write_then_check_ok(root: Path) -> None:
    path, rec = R.write_registry(root)
    assert path == root / R.REGISTRY_RELPATH
    assert _check(root) == []
    files = rec["files"]
    assert "experiments/v0_4/author_kit/hook_bash.py" in files
    assert not any(".venv" in f for f in files)
    assert rec["revisions"] == []
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
    problems = _check(root)
    assert problems == ["frozen file modified: envs/qwen/hypaqwen/wsp/server.py"]
    assert _load_script().main(["--root", str(root), "--check"]) == 1
    assert "MISMATCH: frozen file modified: envs/qwen/hypaqwen/wsp/server.py" in capsys.readouterr().err


def test_check_catches_missing_file(root: Path) -> None:
    R.write_registry(root)
    (root / "experiments/v0_4/PREREGISTRATION.md").unlink()
    (root / "experiments/v0_4/author_kit/prompt_wsp_a.md").unlink()
    problems = _check(root)
    assert "frozen file missing: experiments/v0_4/PREREGISTRATION.md" in problems
    assert "frozen file missing: experiments/v0_4/author_kit/prompt_wsp_a.md" in problems


def test_check_catches_added_file(root: Path) -> None:
    R.write_registry(root)
    _put(root, "scripts/v04_sealed_eval.py", "# written after G1\n")
    _put(root, "envs/qwen/hypaqwen/recipes/constants.json", "{}\n")  # non-.py data under hypaqwen
    _put(root, "src/agent_loop_lab/v04/sub/mod.py", "# a v04 subpackage\n")
    _put(root, "scripts/v04_notes.yaml", "note: 1\n")
    assert _check(root) == ["not in registry: envs/qwen/hypaqwen/recipes/constants.json",
                            "not in registry: scripts/v04_notes.yaml",
                            "not in registry: scripts/v04_sealed_eval.py",
                            "not in registry: src/agent_loop_lab/v04/sub/mod.py"]


def test_check_catches_registry_edit(root: Path) -> None:
    R.write_registry(root)
    path = root / R.REGISTRY_RELPATH
    data = yaml.safe_load(path.read_text())
    data["files"]["docs/V04_WSP_KERNEL.md"] = "0" * 64
    path.write_text(R.render(data))
    problems = _check(root)
    assert "frozen file modified: docs/V04_WSP_KERNEL.md" in problems
    assert "registry freeze_sha256 does not match its record" in problems


def test_check_catches_freeze_digest_and_packet_edits(root: Path) -> None:
    R.write_registry(root)
    path = root / R.REGISTRY_RELPATH
    data = yaml.safe_load(path.read_text())
    data["packet_sha256"] = "f" * 64
    data["freeze_sha256"] = R.freeze_digest(data)  # a consistent forgery still fails the rebuild
    path.write_text(R.render(data))
    assert any(p.startswith("packet_sha256 differs") for p in _check(root))
    data["model"]["revision"] = "deadbeef"
    data["freeze_sha256"] = "0" * 64
    path.write_text(R.render(data))
    problems = _check(root)
    assert any(p.startswith("model differs") for p in problems)
    assert "registry freeze_sha256 does not match its record" in problems


def test_check_catches_packet_change(root: Path) -> None:
    R.write_registry(root)
    (root / "agent_doom_loop_detector_spec_v0_4.md").write_text("changed spec\n")
    problems = _check(root)
    assert "frozen file modified: agent_doom_loop_detector_spec_v0_4.md" in problems
    assert any(p.startswith("packet_sha256 differs") for p in problems)


def test_check_without_registry(root: Path) -> None:
    assert _check(root)[0].startswith("registry not found")
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
    shutil.rmtree(root / "experiments/v0_4/author_kit")
    with pytest.raises(R.RegistryError, match=r"glob matches no file: experiments/v0_4/author_kit/\*\*/\*"):
        R.write_registry(root)


@pytest.mark.parametrize("rel, name", [
    ("scripts/v04_build_disclosure.py", "WSP-B disclosure builder"),
    ("scripts/v04_audit_session.py", "session audit"),
    ("experiments/v0_4/author_kit/prompt_wsp_b.md", "WSP-B prompt"),
    ("experiments/v0_4/author_kit/hook_bash.py", "author-kit command hook"),
])
def test_write_refuses_missing_component(root: Path, rel: str, name: str) -> None:
    (root / rel).unlink()
    with pytest.raises(R.RegistryError, match=f"required component missing: {name}"):
        R.write_registry(root)
    assert not (root / R.REGISTRY_RELPATH).exists()


def test_write_refuses_missing_harness_file(root: Path) -> None:
    (root / "envs/qwen/hypaqwen/wsp/author/run_py.py").unlink()
    with pytest.raises(R.RegistryError, match="frozen file missing: envs/qwen/hypaqwen/wsp/author/run_py.py"):
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


def test_write_refuses_symlinked_directory(root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _put(outside, "outside_extra.py", "# outside the tree\n")
    (root / "envs/qwen/hypaqwen/more").symlink_to(outside, target_is_directory=True)
    files, problems = R.frozen_files(root)
    assert "symlink under a frozen tree: envs/qwen/hypaqwen/more" in problems
    assert not any("outside_extra" in f for f in files)
    with pytest.raises(R.RegistryError, match="symlink under a frozen tree"):
        R.write_registry(root)


# ---------------------------------------------------------------------------
# Root-env import closure
# ---------------------------------------------------------------------------


def _wire_imports(root: Path) -> None:
    for pkg in ("src/agent_loop_lab", "src/agent_loop_lab/eval", "src/agent_loop_lab/v04"):
        _put(root, f"{pkg}/__init__.py", "")
    _put(root, "src/agent_loop_lab/v04/scoring.py", "from ..eval import m4\n\ndef f():\n    from ..eval.extra import X\n")
    _put(root, "src/agent_loop_lab/eval/m4.py", "from agent_loop_lab.eval import helper\n")
    _put(root, "src/agent_loop_lab/eval/helper.py", "X = 1\n")
    _put(root, "src/agent_loop_lab/eval/extra.py", "X = 2\n")


def test_import_closure_is_frozen(root: Path) -> None:
    _wire_imports(root)
    _, rec = R.write_registry(root)
    for rel in ("src/agent_loop_lab/eval/helper.py", "src/agent_loop_lab/eval/extra.py",
                "src/agent_loop_lab/__init__.py", "src/agent_loop_lab/eval/__init__.py"):
        assert rel in rec["files"], rel
    assert _check(root) == []
    (root / "src/agent_loop_lab/eval/m4.py").write_text("from agent_loop_lab.eval import helper\nY = 0\n")
    (root / "src/agent_loop_lab/eval/helper.py").write_text("X = 3\n")
    assert _check(root) == ["frozen file modified: src/agent_loop_lab/eval/helper.py",
                            "frozen file modified: src/agent_loop_lab/eval/m4.py"]


def test_import_closure_new_import_is_caught(root: Path) -> None:
    _wire_imports(root)
    R.write_registry(root)
    _put(root, "src/agent_loop_lab/eval/later.py", "Z = 1\n")
    (root / "src/agent_loop_lab/eval/helper.py").write_text("from . import later\nX = 1\n")
    problems = _check(root)
    assert "not in registry: src/agent_loop_lab/eval/later.py" in problems
    assert "frozen file modified: src/agent_loop_lab/eval/helper.py" in problems


def test_import_closure_refuses_unresolved_import(root: Path) -> None:
    _wire_imports(root)
    _put(root, "src/agent_loop_lab/v04/scoring.py", "from ..eval import m4\nfrom ..nowhere import thing\n")
    with pytest.raises(R.RegistryError, match="imports agent_loop_lab.nowhere, which is not a file"):
        R.write_registry(root)


# ---------------------------------------------------------------------------
# Additive revisions
# ---------------------------------------------------------------------------


def test_revise_adds_files_and_chains(root: Path) -> None:
    _, g1 = R.write_registry(root)
    _put(root, "scripts/v04_sealed_eval.py", "# written after G1 against §3.4\n")
    _, rec = R.revise_registry(root, reason="sealed evaluation script (§3.4)", date="2026-11-20")
    rev = rec["revisions"][-1]
    assert rev["added"] == {"scripts/v04_sealed_eval.py": R._sha256(b"# written after G1 against \xc2\xa73.4\n")}
    assert rev["changed"] == {} and rev["previous_freeze_sha256"] == g1["freeze_sha256"]
    assert _check(root) == []
    assert R.g1_freeze_sha256(rec) == g1["freeze_sha256"]
    # A second revision chains onto the first.
    _put(root, "envs/qwen/hypaqwen/wsp/wsp_f.py", "# WSP-F ops\n")
    (root / "envs/qwen/hypaqwen/wsp/server.py").write_text("# server with wsp-f dispatch\n")
    _, rec2 = R.revise_registry(root, reason="WSP-F ops (§7.3)", allow_change=["envs/qwen/hypaqwen/wsp/server.py"],
                                date="2026-11-21")
    assert rec2["revisions"][-1]["previous_freeze_sha256"] == rec["freeze_sha256"]
    assert set(rec2["revisions"][-1]["changed"]) == {"envs/qwen/hypaqwen/wsp/server.py"}
    assert _check(root) == []
    assert R.g1_freeze_sha256(rec2) == g1["freeze_sha256"]
    assert _load_script().main(["--root", str(root), "--revise", "--reason", "x"]) == 1  # nothing to revise


def test_revise_refuses_unlisted_change_removal_and_packet(root: Path) -> None:
    R.write_registry(root)
    before = (root / R.REGISTRY_RELPATH).read_bytes()
    (root / "envs/qwen/hypaqwen/wsp/server.py").write_text("# changed\n")
    with pytest.raises(R.RegistryError, match="modified without --allow-change: envs/qwen/hypaqwen/wsp/server.py"):
        R.revise_registry(root, reason="x", date="2026-11-20")
    with pytest.raises(R.RegistryError, match="allow-change docs/V04_WSP_KERNEL.md names a file that did not change"):
        R.revise_registry(root, reason="x", allow_change=["envs/qwen/hypaqwen/wsp/server.py",
                                                          "docs/V04_WSP_KERNEL.md"], date="2026-11-20")
    (root / "envs/qwen/hypaqwen/wsp/server.py").write_text("# envs/qwen/hypaqwen/wsp/server.py\n")
    (root / "envs/qwen/hypaqwen/wsp/author/client.py").unlink()
    with pytest.raises(R.RegistryError, match="frozen file removed: envs/qwen/hypaqwen/wsp/author/client.py"):
        R.revise_registry(root, reason="x", date="2026-11-20")
    _put(root, "envs/qwen/hypaqwen/wsp/author/client.py", "# envs/qwen/hypaqwen/wsp/author/client.py\n")
    (root / "agent_doom_loop_detector_spec_v0_4.md").write_text("changed spec\n")  # changes the packet
    with pytest.raises(R.RegistryError, match="packet_sha256 differs"):
        R.revise_registry(root, reason="x", allow_change=["agent_doom_loop_detector_spec_v0_4.md"],
                          date="2026-11-20")
    with pytest.raises(R.RegistryError, match="needs a reason"):
        R.revise_registry(root, reason="  ", date="2026-11-20")
    assert (root / R.REGISTRY_RELPATH).read_bytes() == before


def test_check_catches_rewritten_revision_history(root: Path) -> None:
    R.write_registry(root)
    (root / "envs/qwen/hypaqwen/wsp/server.py").write_text("# wsp-f dispatch\n")
    R.revise_registry(root, reason="WSP-F", allow_change=["envs/qwen/hypaqwen/wsp/server.py"], date="2026-11-21")
    path = root / R.REGISTRY_RELPATH
    data = yaml.safe_load(path.read_text())
    # Hide the change: record it as an addition-free revision that only says the file was never changed.
    data["revisions"][-1]["changed"] = {}
    data["freeze_sha256"] = R.freeze_digest(data)
    path.write_text(R.render(data))
    problems = _check(root)
    assert "revision 1: the record it revised does not hash to its previous_freeze_sha256" in problems
    with pytest.raises(R.RegistryError, match="not intact"):
        R.revise_registry(root, reason="y", date="2026-11-22")


def test_revise_is_deterministic(root: Path, tmp_path: Path) -> None:
    R.write_registry(root)
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    for base in (root, copy):
        _put(base, "scripts/v04_sealed_eval.py", "# same\n")
        R.revise_registry(base, reason="sealed eval", date="2026-11-20")
    assert (root / R.REGISTRY_RELPATH).read_bytes() == (copy / R.REGISTRY_RELPATH).read_bytes()


def test_check_runs_the_v03_registry_check(root: Path) -> None:
    R.write_registry(root)
    problems = R.check_registry(root)
    assert problems and all(p.startswith("v0.3 registry: ") for p in problems)
    assert any("registry not found" in p for p in problems)


def test_repo_pins_and_globs() -> None:
    """On the real tree: the interface hash is the pre-registered one, the model pins parse, and the
    globs and the import closure cover every file the G1 task and the review name. Nothing is written."""
    assert R.interface_spec_sha256(REPO) == R.INTERFACE_SPEC_SHA256
    pins = R.model_pins(REPO)
    assert pins["id"] == "Qwen/Qwen3.5-0.8B" and pins["revision"].startswith("2fc06364")
    files, problems = R.frozen_files(REPO)
    assert not [p for p in problems if "imports" in p or "symlink" in p or "parse" in p], problems
    for rel in ("envs/qwen/hypaqwen/interface.py", "envs/qwen/hypaqwen/readout.py", "envs/qwen/hypaqwen/model.py",
                "envs/qwen/hypaqwen/data.py", "envs/qwen/hypaqwen/run_readout.py",
                "envs/qwen/hypaqwen/wsp/server.py", "envs/qwen/hypaqwen/wsp/author/hn_guard/guards_v04.py",
                "envs/qwen/scripts/make_author_venv.sh", "envs/qwen/uv.lock", "docs/V04_WSP_KERNEL.md",
                "experiments/v0_4/PREREGISTRATION.md", "scripts/v04_freeze.py", "src/agent_loop_lab/v04/registry.py",
                "src/agent_loop_lab/v04/scoring.py", "experiments/v0_4/fresh_test/REGISTRATION.json",
                "artifacts/reports/v04_prompt_choice_revision1.json",
                # the root-env scoring, subset and disclosure path (import closure and required files)
                "src/agent_loop_lab/schema.py", "src/agent_loop_lab/eval/m4.py", "src/agent_loop_lab/eval/metrics.py",
                "src/agent_loop_lab/eval/collisions.py", "src/agent_loop_lab/v03/scoring.py",
                "src/agent_loop_lab/v03/sealed.py", "src/agent_loop_lab/v03/renderer.py",
                "src/agent_loop_lab/baselines/phase_c_v03.py", "src/agent_loop_lab/baselines/phase_c.py",
                "src/agent_loop_lab/synth/counterfactual.py", "src/agent_loop_lab/synth/families/abab_oscillation.py",
                "datasets/v0_4/EXPORT_MANIFEST.json", "datasets/v0_2/splits.json", "datasets/v0_3/MANIFEST_v0_3.json",
                "artifacts/reports/v03_sealed.json", "experiments/v0_3/phase_c/README.md",
                "envs/qwen/tests/test_wsp_server.py"):
        assert rel in files, rel
    assert "_hn_v04_frozen_interface" not in sys.modules
