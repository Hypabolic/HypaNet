"""The ``./py`` launcher (``tools/hn_kernel/run_py.py``) and the tools install, against a fake kernel server.

Every run is a fresh ``python -I -B tools/hn_kernel/run_py.py`` child, as the ``./py`` wrapper runs it. The
author directory lives under ``/tmp`` (macOS's 104-byte AF_UNIX path limit) with the tools installed
read-only by ``hypaqwen.wsp.install.install_tools``; a socket server in this process answers the kernel ops
the scripts use. No model is loaded.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from hypaqwen.wsp import install as I
from hypaqwen.wsp.author import fwd as F
from hypaqwen.wsp.author import protocol as P
from hypaqwen.wsp.author import run_py as R

PHASE = "p1"
EDIT_HASH = "e" * 64


class FakeKernel:
    """Answers ``edit_hash``, ``names`` and ``budget`` on the kernel socket, one connection at a time."""

    def __init__(self, path: Path) -> None:
        self.sock = P.listen(path)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                while True:
                    try:
                        header, _ = P.recv_msg(conn)
                    except Exception:  # noqa: BLE001 - the client closed the connection
                        break
                    result = {"edit_hash": EDIT_HASH, "names": ["model.language_model.norm.weight"],
                              "budget": {"tokens_used": 0, "tokens_left": 10, "requests": 0}}.get(header.get("op"))
                    P.send_msg(conn, {"ok": True, "result": result})

    def close(self) -> None:
        self.sock.close()


@pytest.fixture
def author():
    import tokenizers

    root = Path(os.path.realpath(tempfile.mkdtemp(prefix="hnr", dir="/tmp")))
    ad = root / "a"
    (ad / "output" / PHASE / "scratch").mkdir(parents=True)
    (ad / "packet" / "qwen").mkdir(parents=True)
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel({"hello": 0, "[UNK]": 1}, unk_token="[UNK]"))
    tok.save(str(ad / "packet" / "qwen" / "tokenizer.json"))
    (ad / "packet" / "helper_in_packet.py").write_text("VALUE = 'packet'\n")
    I.install_tools(ad / "tools")
    kernel = FakeKernel(ad / ".kernel" / "kernel.sock")
    try:
        yield ad
    finally:
        kernel.close()
        I.uninstall(ad / "tools")
        for d, _dirs, _files in os.walk(root):
            os.chmod(d, 0o755)
        shutil.rmtree(root, ignore_errors=True)


def run(ad: Path, *argv: str, phase: str | None = PHASE, cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "HN_"))}
    env["HN_AUTHOR_DIR"] = str(ad)
    if phase is not None:
        env["HN_PHASE"] = phase
    return subprocess.run([sys.executable, "-I", "-B", str(ad / "tools" / "hn_kernel" / "run_py.py"), *argv],
                          cwd=cwd or ad, env=env, capture_output=True, text=True, timeout=180)


def script(ad: Path, rel: str, text: str) -> str:
    p = ad / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return rel


def test_guard_installed_and_wsp_is_the_installed_client(author):
    rel = script(author, f"output/{PHASE}/scratch/ok.py", (
        "import sys\n"
        "import hn_guard\n"
        "from wsp import NAMES, edit_hash, budget, tok\n"  # NAMES first: connecting must keep the pre-guard tok
        "print('mode', hn_guard.installed_mode())\n"
        "print('hash', edit_hash())\n"
        "print('names', NAMES)\n"
        "print('tok', tok.encode('hello').ids)\n"
        "print('wsp', sys.modules['wsp'].__file__)\n"
        "print('argv', sys.argv)\n"))
    p = run(author, rel, "x", "--k=1")
    assert p.returncode == 0, p.stderr
    assert "mode KERNEL_CLIENT" in p.stdout and f"hash {EDIT_HASH}" in p.stdout
    assert "tok [0]" in p.stdout and "names ('model.language_model.norm.weight',)" in p.stdout
    assert f"wsp {author / 'tools' / 'hn_kernel' / 'wsp.py'}" in p.stdout
    assert f"argv ['{author / rel}', 'x', '--k=1']" in p.stdout  # runpy sets argv[0] to the script's path
    assert "HN_GUARD" not in p.stderr


PLANTED = ("json", "socket", "re", "runpy", "traceback", "stat", "struct", "contextlib", "math", "typing", "atexit",
           "numpy", "torch", "safetensors", "tokenizers", "importlib", "wsp", "client", "protocol", "hn_guard",
           "__future__", "random", "pathlib")


def test_planted_modules_are_never_imported(author, tmp_path):
    marker = Path(tempfile.mkdtemp(prefix="hnm", dir="/tmp")) / "PLANTED_CODE_RAN"
    try:
        for d in (author, author / "output" / PHASE, author / "output" / PHASE / "scratch"):
            for name in PLANTED:
                (d / f"{name}.py").write_text(f"open({str(marker)!r}, 'a').write({name!r} + '\\n')\n")
        rel = script(author, f"output/{PHASE}/scratch/use.py", (
            "import json, re, socket, struct, math\n"
            "from wsp import edit_hash\n"
            "print('hash', edit_hash())\n"))
        for cwd in (author, author / "output" / PHASE / "scratch"):
            p = run(author, os.path.relpath(author / rel, cwd), cwd=cwd)
            assert p.returncode == 0, p.stderr
            assert f"hash {EDIT_HASH}" in p.stdout
        assert not marker.exists(), marker.read_text()
    finally:
        shutil.rmtree(marker.parent, ignore_errors=True)


def test_author_helper_modules_resolve_after_the_standard_finders(author):
    script(author, f"output/{PHASE}/common.py", "VALUE = 'phase dir'\n")
    script(author, f"output/{PHASE}/scratch/helper.py", "VALUE = 'scratch'\n")
    script(author, f"output/{PHASE}/scratch/json.py", "raise SystemExit('the author json.py ran')\n")
    rel = script(author, f"output/{PHASE}/scratch/main.py", (
        "import common, helper, helper_in_packet, json\n"
        "print(common.VALUE, helper.VALUE, helper_in_packet.VALUE, json.__file__.endswith('json/__init__.py'))\n"))
    p = run(author, rel)
    assert p.returncode == 0, p.stderr
    assert "phase dir scratch packet True" in p.stdout


def test_a_caught_violation_still_exits_3(author):
    rel = script(author, f"output/{PHASE}/scratch/sneaky.py", (
        "try:\n"
        "    open('/etc/hosts').read()\n"
        "except Exception as e:\n"
        "    print('caught', type(e).__name__)\n"))
    p = run(author, rel)
    assert p.returncode == 3, (p.returncode, p.stderr)
    assert "caught" in p.stdout and "HN_GUARD: refused" in p.stderr


def test_exit_status_and_tracebacks(author):
    assert run(author, script(author, f"output/{PHASE}/scratch/five.py", "raise SystemExit(5)\n")).returncode == 5
    p = run(author, script(author, f"output/{PHASE}/scratch/boom.py", "1 / 0\n"))
    assert p.returncode == 1 and "ZeroDivisionError" in p.stderr


@pytest.mark.parametrize("argv, why", [
    (["packet/helper_in_packet.py"], "inside output/p1/"),
    (["output/p2/x.py"], "inside output/p1/"),
    ([f"output/{PHASE}/../../packet/helper_in_packet.py"], "inside output/p1/"),
    ([f"output/{PHASE}/scratch/missing.py"], "no such script"),
    ([f"output/{PHASE}/scratch/notes.txt"], "not a .py file"),
    (["-c", "print(1)"], "-c is not supported"),
    (["-m", "http.server"], "only -m form"),
    (["--version"], "unknown option"),
    ([], "usage"),
])
def test_refused_command_lines(author, argv, why):
    (author / "output" / "p2").mkdir()
    (author / "output" / "p2" / "x.py").write_text("print('ran')\n")
    p = run(author, *argv)
    assert p.returncode == 2 and why in p.stderr, (p.returncode, p.stderr)
    assert "ran" not in p.stdout


def test_symlinked_script_outside_output_is_refused(author):
    outside = Path(tempfile.mkdtemp(prefix="hno", dir="/tmp"))
    try:
        (outside / "x.py").write_text("print('ran')\n")
        (author / "output" / PHASE / "scratch" / "link.py").symlink_to(outside / "x.py")
        p = run(author, f"output/{PHASE}/scratch/link.py")
        assert p.returncode == 2 and "inside output/p1/" in p.stderr
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def test_py_compile_checks_syntax_without_running_or_writing(author):
    good = script(author, f"output/{PHASE}/build_edit.py", "print('ran')\ndef build(orig, consts):\n    return {}\n")
    p = run(author, "-m", "py_compile", good)
    assert p.returncode == 0 and "compiles" in p.stdout and "ran" not in p.stdout, p.stderr
    assert not (author / "output" / PHASE / "__pycache__").exists()
    bad = script(author, f"output/{PHASE}/bad.py", "def f(:\n")
    p = run(author, "-m", "py_compile", bad)
    assert p.returncode == 1 and "SyntaxError" in p.stderr


def test_phase_from_the_output_dir(author):
    rel = script(author, f"output/{PHASE}/scratch/ph.py", "import os\nprint('phase', os.environ['HN_PHASE'])\n")
    p = run(author, rel, phase=None)
    assert p.returncode == 0 and f"phase {PHASE}" in p.stdout, p.stderr
    (author / "output" / "p0").mkdir()
    os.chmod(author / "output" / "p0", 0o555)  # an earlier phase, made read-only: the writable one is chosen
    p = run(author, rel, phase=None)
    assert p.returncode == 0 and f"phase {PHASE}" in p.stdout, p.stderr
    os.chmod(author / "output" / "p0", 0o755)
    p = run(author, rel, phase=None)
    assert p.returncode == 2 and "cannot tell the phase" in p.stderr
    p = run(author, rel, phase="../x")
    assert p.returncode == 2 and "bad phase name" in p.stderr


def test_writable_tools_refuse_to_run(author):
    rel = script(author, f"output/{PHASE}/scratch/ok2.py", "print('ran')\n")
    os.chmod(author / "tools" / "hn_kernel", 0o755)
    try:
        p = run(author, rel)
        assert p.returncode == 2 and "writable" in p.stderr and "ran" not in p.stdout
    finally:
        os.chmod(author / "tools" / "hn_kernel", 0o555)


def test_mismatched_author_dir_env_is_refused(author, tmp_path):
    rel = script(author, f"output/{PHASE}/scratch/ok3.py", "print('ran')\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "HN_"))}
    env.update(HN_AUTHOR_DIR=str(tmp_path), HN_PHASE=PHASE)
    p = subprocess.run([sys.executable, "-I", "-B", str(author / "tools" / "hn_kernel" / "run_py.py"), rel],
                       cwd=author, env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 2 and "does not name the directory" in p.stderr


# -- sys.path scrubbing with the venv inside the author dir (the fixed layout: AUTHOR_DIR/.venv) ---------------


def _scrub(mod, monkeypatch, author: Path, here: Path, path: list[str]) -> list[str]:
    venv = author / ".venv"
    monkeypatch.setattr(sys, "path", list(path))
    for attr in ("prefix", "exec_prefix"):
        monkeypatch.setattr(sys, attr, str(venv))
    if mod is R:
        monkeypatch.setattr(R, "ROOT", str(author))
        monkeypatch.setattr(R, "_HERE", str(here))
    else:
        monkeypatch.setenv("HN_AUTHOR_DIR", str(author))
        monkeypatch.setattr(F, "_HERE", str(here))
    mod._scrub_sys_path()
    return list(sys.path)


@pytest.mark.parametrize("mod", [R, F], ids=["run_py", "fwd"])
def test_scrub_keeps_the_author_venv_and_nothing_else_of_the_author_dir(mod, monkeypatch, tmp_path):
    author = tmp_path.resolve() / "a"
    here = author / "tools" / "hn_kernel"
    site = author / ".venv" / "lib" / "python3.12" / "site-packages"
    for d in (here, site, author / "output" / PHASE):
        d.mkdir(parents=True)
    base = "/opt/python/lib/python3.12"
    got = _scrub(mod, monkeypatch, author, here, ["", str(here), str(author), str(author / "output" / PHASE), base,
                                                  str(site)])
    assert got == [base, str(site)]


# -- install_tools / verify_tools ------------------------------------------------------------------------------


def test_install_tools_is_read_only_and_verified(tmp_path):
    hook = tmp_path / "hook_bash.py"
    hook.write_text("# hook\n")
    tools = tmp_path / "a" / "tools"
    try:
        man = I.install_tools(tools, hook=hook)
        assert set(man) == {f"hn_kernel/{n}" for n in I.KERNEL_FILES} | {f"hn_guard/{n}" for n in I.GUARD_FILES} \
            | {"hn_hook/hook_bash.py"}
        assert "hn_kernel/run_py.py" in man
        I.verify_tools(tools, hook=hook)
        with pytest.raises(I.KernelTampered, match="expected exactly"):
            I.verify_tools(tools)  # without the hook source the hn_hook/ dir is unexpected
        with pytest.raises(PermissionError):
            (tools / "socket.py").write_text("x")
        hook.write_text("# changed\n")
        with pytest.raises(I.KernelTampered, match="does not match"):
            I.verify_tools(tools, hook=hook)
        hook.write_text("# hook\n")
        os.chmod(tools, 0o755)
        with pytest.raises(I.KernelTampered, match="writable"):
            I.verify_tools(tools, hook=hook)
        (tools / "extra").mkdir()
        os.chmod(tools, 0o555)
        with pytest.raises(I.KernelTampered, match="expected exactly"):
            I.verify_tools(tools, hook=hook)
    finally:
        I.uninstall(tools)
    assert not tools.exists()


def test_install_cli(tmp_path):
    tools = tmp_path / "t"
    try:
        p = subprocess.run([sys.executable, "-m", "hypaqwen.wsp.install", "install", str(tools)], capture_output=True,
                           text=True, timeout=60)
        assert p.returncode == 0 and "hn_kernel/run_py.py" in json.loads(p.stdout), p.stderr
        p = subprocess.run([sys.executable, "-m", "hypaqwen.wsp.install", "verify", str(tools)], capture_output=True,
                           text=True, timeout=60)
        assert p.returncode == 0, p.stderr
        os.chmod(tools / "hn_guard", 0o755)
        p = subprocess.run([sys.executable, "-m", "hypaqwen.wsp.install", "verify", str(tools)], capture_output=True,
                           text=True, timeout=60)
        assert p.returncode == 1 and "writable" in p.stderr
    finally:
        I.uninstall(tools)
