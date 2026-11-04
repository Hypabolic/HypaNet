"""v0.3 author kit (spec v0.3 §8): standalone ./py launcher, session audit, cost summary, settings."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_loop_lab.v03.kit import install_tools
from agent_loop_lab.v03.packet import build_packet

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "experiments" / "v0_3" / "author_kit"
AUDIT = REPO / "scripts" / "v03_audit_session.py"
COST = REPO / "scripts" / "v03_cost.py"


@pytest.fixture(scope="module")
def author_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("kit") / "HypaNet-v03-author"
    (root / "output" / "v03-phase-a" / "scratch").mkdir(parents=True)
    build_packet(REPO / "agent_doom_loop_detector_spec_v0_3.md", REPO / "agent_doom_loop_detector_spec_v0_2.md",
                 REPO / "src" / "agent_loop_lab" / "v03", root / "packet")
    install_tools(root / "tools")
    return root


def _run(author_dir: Path, script: str, *args: str) -> subprocess.CompletedProcess:
    p = author_dir / "output" / "v03-phase-a" / "scratch" / "s.py"
    p.write_text(script)
    return subprocess.run([sys.executable, "-I", "-B", str(author_dir / "tools" / "launch.py"), str(p), *args],
                          cwd=author_dir, capture_output=True, text=True, timeout=300)


FORWARD = """\
import sys
import torch, model, tokenizer, renderer
m = model.zero_model()
enc = tokenizer.encode_session(["09:00:00.000 read_file path=zork -> ok in=1 out=2"])
logits = m(torch.tensor(enc.tokens), torch.tensor(enc.event_ord), torch.tensor(enc.offset))
print("shape", tuple(logits.shape), "argmax", int(model.predict(logits[enc.eoe_index[-1]])))
renderer.render_rng({"family": "read_file"}).random()
what = sys.argv[1] if len(sys.argv) > 1 else ""
if what == "rand":
    torch.rand(2)
elif what == "random":
    import random
    random.random()
elif what == "backward":
    (torch.ones(2, requires_grad=True) * 2).sum().backward()
elif what == "optim":
    torch.optim.SGD([torch.zeros(1)], lr=0.1)
elif what == "sibling":
    open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").getcwd()), "x.txt"), "w")
elif what == "subprocess":
    __import__("subprocess").run(["true"])
elif what == "from_file":
    sib = __import__("os").path.join(__import__("os").path.dirname(__import__("os").getcwd()), "secret.txt")
    print(bytes(torch.from_file(sib, size=6, dtype=torch.uint8).tolist()))
elif what == "storage_from_file":
    sib = __import__("os").path.join(__import__("os").path.dirname(__import__("os").getcwd()), "secret.txt")
    print(bytes(torch.UntypedStorage.from_file(sib, False, 6).tolist()))
elif what == "ctypes":
    import ctypes
    libc = ctypes.CDLL(None)
    print(libc.getpid())
elif what == "tmp":
    import tempfile, os
    print(os.listdir(os.path.realpath(tempfile.gettempdir())))
"""


def test_launcher_allows_forward_and_renderer_rng(author_dir):
    r = _run(author_dir, FORWARD)
    assert r.returncode == 0, r.stderr
    assert "shape (" in r.stdout and "argmax 0" in r.stdout


@pytest.mark.parametrize("what", ["rand", "random", "backward", "optim", "sibling", "subprocess", "from_file",
                                  "storage_from_file", "ctypes", "tmp"])
def test_launcher_refuses(author_dir, what):
    (author_dir.parent / "secret.txt").write_text("SECRET")
    r = _run(author_dir, FORWARD, what)
    assert r.returncode == 3, (r.returncode, r.stderr[-500:])
    assert "AUTHOR GUARD: refused" in r.stderr
    assert "SECRET" not in r.stdout


def test_launcher_module_mode_and_no_inline_code(author_dir):
    target = author_dir / "output" / "v03-phase-a" / "ok.py"
    target.write_text("x = 1\n")
    r = subprocess.run([sys.executable, "-I", "-B", str(author_dir / "tools" / "launch.py"), "-m", "py_compile", str(target)],
                       cwd=author_dir, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "-I", "-B", str(author_dir / "tools" / "launch.py"), "-c", "print(1)"],
                       cwd=author_dir, capture_output=True, text=True, timeout=300)
    assert r.returncode == 2


def test_standalone_guard_has_no_harness_import(author_dir):
    v02 = (author_dir / "tools" / "hn_guard" / "guards_v02.py").read_text()
    assert "from .model import" not in v02 and "from torch.nn import Module as DoomLoopMLP" in v02
    v03 = (author_dir / "tools" / "hn_guard" / "guards_v03.py").read_bytes()
    assert v03 == (REPO / "src" / "agent_loop_lab" / "v03" / "guards.py").read_bytes()


# -- audit ------------------------------------------------------------------------------


def _transcript(path: Path, calls: list[tuple[str, dict]], denied: tuple[int, ...] = (), cost: float = 1.25) -> Path:
    lines = []
    for i, (name, inp) in enumerate(calls):
        lines.append({"type": "assistant", "message": {"model": "claude-fable-5-1", "content": [
            {"type": "tool_use", "id": f"t{i}", "name": name, "input": inp}]}})
    lines.append({"type": "result", "total_cost_usd": cost,
                  "permission_denials": [{"tool_use_id": f"t{i}"} for i in denied]})
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return path


def _audit(log: Path, author: Path) -> tuple[int, dict]:
    r = subprocess.run([sys.executable, str(AUDIT), str(log), str(author), "v03-phase-a"], capture_output=True, text=True)
    return r.returncode, json.loads(log.with_suffix(".audit.json").read_text())


def _audit_dir(tmp_path: Path) -> Path:
    a = tmp_path / "HypaNet-v03-author"
    (a / "output" / "v03-phase-a" / "scratch").mkdir(parents=True)
    (a / "packet").mkdir()
    return a


def test_audit_clean_session(tmp_path):
    a = _audit_dir(tmp_path)
    (a / "output" / "v03-phase-a" / "scratch" / "c.py").write_text("import sys, json\nimport model, renderer\n")
    log = _transcript(tmp_path / "s.stream.jsonl", [
        ("Read", {"file_path": str(a / "packet" / "SPEC.md")}),
        ("Read", {"file_path": "packet/AUTHOR_INSTRUCTIONS.md"}),
        ("Write", {"file_path": str(a / "output" / "v03-phase-a" / "scratch" / "c.py"), "content": "x"}),
        ("Bash", {"command": "./py output/v03-phase-a/scratch/c.py"}),
        ("Bash", {"command": "./py -m py_compile output/v03-phase-a/build_weights.py"}),
    ])
    code, rep = _audit(log, a)
    assert code == 0, rep["violations"] + rep["file_findings"]
    assert rep["cost_usd"] == 1.25 and rep["models"] == ["claude-fable-5-1"]
    assert "output/v03-phase-a/scratch/c.py" in rep["scanned_files"]


@pytest.mark.parametrize("call,fragment", [
    (("Read", {"file_path": "/Users/matthew/Development/hypabolic/HypaNet/src/agent_loop_lab/synth/world.py"}), "outside"),
    (("Read", {"file_path": "../HypaNet/README.md"}), "outside the author dir"),
    (("Grep", {"pattern": "x", "path": "/Users/matthew/.claude/projects"}), ".claude"),
    (("Bash", {"command": "ls .."}), "does not start with './py '"),
    (("Bash", {"command": "./py a.py; echo hi"}), "compound"),
    (("Bash", {"command": "./py a.py > out.txt"}), "compound"),
    (("Bash", {"command": "./py -c 'print(1)'"}), "-c"),
    (("Bash", {"command": "./py packet/renderer.py"}), "outside output/"),
    (("Write", {"file_path": "packet/model.py", "content": ""}), "write outside output/v03-phase-a/"),
    (("Edit", {"file_path": "tools/launch.py", "old_string": "a", "new_string": "b"}), "write outside"),
    (("WebFetch", {"url": "https://example.com"}), "outside the author kit"),
])
def test_audit_flags(tmp_path, call, fragment):
    a = _audit_dir(tmp_path)
    log = _transcript(tmp_path / "s.stream.jsonl", [call])
    code, rep = _audit(log, a)
    assert code == 1
    assert any(fragment in v for v in rep["violations"]), rep["violations"]


def test_audit_refused_attempts_do_not_fail(tmp_path):
    a = _audit_dir(tmp_path)
    log = _transcript(tmp_path / "s.stream.jsonl", [("Bash", {"command": "ls .."})], denied=(0,))
    code, rep = _audit(log, a)
    assert code == 0 and rep["violations"] and rep["executed_violations"] == 0


def test_audit_scans_output_scripts(tmp_path):
    a = _audit_dir(tmp_path)
    (a / "output" / "v03-phase-a" / "scratch" / "search.py").write_text("import random\nimport torch\nx = torch.randn(3)\n")
    log = _transcript(tmp_path / "s.stream.jsonl", [])
    code, rep = _audit(log, a)
    assert code == 1 and any("random" in f for f in rep["file_findings"])


# -- cost ---------------------------------------------------------------------------------


def test_cost_sums_and_cap(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    _transcript(logs / "v03-phase-a.stream.jsonl", [], cost=300.0)
    _transcript(logs / "v03-phase-a.c1.stream.jsonl", [], cost=150.0)
    _transcript(logs / "v03-phase-b.stream.jsonl", [], cost=10.0)
    r = subprocess.run([sys.executable, str(COST), "--logs", str(logs), "--phase", "v03-phase-a", "--check"], capture_output=True, text=True)
    s = json.loads(r.stdout)
    assert r.returncode == 0 and s["phases"]["v03-phase-a"]["total_usd"] == 450.0 and s["total_usd"] == 460.0
    _transcript(logs / "v03-phase-a.r1.stream.jsonl", [], cost=60.0)
    r = subprocess.run([sys.executable, str(COST), "--logs", str(logs), "--phase", "v03-phase-a", "--check"], capture_output=True, text=True)
    s = json.loads(r.stdout)
    assert r.returncode == 3 and s["phases"]["v03-phase-a"]["overshoot_usd"] == 10.0


# -- settings -------------------------------------------------------------------------------


@pytest.mark.parametrize("phase", ["a", "b"])
def test_settings_isolation(phase):
    s = json.loads((KIT / f"settings_v03_phase_{phase}.json").read_text())
    assert s["autoMemoryEnabled"] is False
    allow, deny = s["permissions"]["allow"], s["permissions"]["deny"]
    assert [a for a in allow if a.startswith("Bash")] == ["Bash(./py:*)"]
    writes = [a for a in allow if a.startswith(("Write", "Edit"))]
    assert writes == [f"Write(./output/v03-phase-{phase}/**)", f"Edit(./output/v03-phase-{phase}/**)"]
    for root in ("//Users/matthew/Development/hypabolic/HypaNet/**", "//Users/matthew/Development/hypabolic/HypaNet-m2-author/**",
                 "//private/tmp/claude-501/**", "//tmp/claude-501/**", "//Users/matthew/.claude/projects/**",
                 "//private/var/folders/**", "//var/folders/**"):
        for tool in ("Read", "Grep", "Glob", "Edit", "Write"):
            assert f"{tool}({root})" in deny
    for local in ("./packet/**", "./tools/**"):
        assert f"Write({local})" in deny and f"Edit({local})" in deny
    assert not any("HypaNet-v03-author" in d or "HypaNet*" in d for d in deny)


def test_launch_script_flags():
    text = (REPO / "scripts" / "v03_launch.sh").read_text()
    for flag in ("--setting-sources project", "--strict-mcp-config", "--permission-mode dontAsk",
                 "--disallowedTools Agent Workflow WebFetch WebSearch Artifact Skill", "claude-fable-5-1",
                 "claude-opus-5-5", "V03_EFFORT:-max", "128000", "CLAUDE_CODE_DISABLE_AUTO_MEMORY=1",
                 "v03_audit_session.py", "v03_cost.py"):
        assert flag in text, flag
    assert os.access(REPO / "scripts" / "v03_launch.sh", os.X_OK)
    assert "pytest-of-$USER_NAME" in text and "error cap" in text


def test_launcher_deny_roots_cover_system_temp():
    text = (KIT / "launch.py").read_text()
    assert "tempfile.gettempdir()" in text and '"/private/var/folders"' in text and '"/var/folders"' in text


@pytest.mark.parametrize("src,fragment", [
    ("import torch\nx = torch.from_file('/x', size=3)\n", "from_file"),
    ("import ctypes\n", "ctypes"),
    ("def f():\n    return f.__globals__\n", "__globals__"),
    ("import torch\ndef f(n):\n    return getattr(torch, n)\n", "getattr"),
])
def test_audit_flags_unaudited_readers_and_internals(tmp_path, src, fragment):
    a = _audit_dir(tmp_path)
    (a / "output" / "v03-phase-a" / "scratch" / "x.py").write_text(src)
    code, rep = _audit(_transcript(tmp_path / "s.stream.jsonl", []), a)
    assert code == 1 and any(fragment in f for f in rep["file_findings"]), rep["file_findings"]
