"""kit/make_settings.py: the author session's settings (SPEC §3, §13)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SLOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SLOP / "kit"))
import make_settings as ms  # noqa: E402

R1 = ms.DEFAULT_AUTHOR
R2 = ms.R2_AUTHOR
SMOKE = ms.SMOKE


def reads(settings: dict) -> set[str]:
    return {r for r in settings["permissions"]["deny"] if r.startswith("Read(")}


def rd(p) -> str:
    return f"Read(/{p}/**)"


def test_r1_settings():
    s = ms.build(R1)
    assert s["autoMemoryEnabled"] is False and s["switchModelsOnFlag"] is False
    assert s["showThinkingSummaries"] is True and s["permissions"]["defaultMode"] == "dontAsk"
    deny = reads(s)
    uid = os.getuid()
    for p in (ms.REPO, SMOKE, ms.PRIVATE_NOTES, f"/tmp/claude-{uid}", f"/private/tmp/claude-{uid}"):
        assert rd(p) in deny, p
    for _, root in ms.SEALED_REPOS:
        assert rd(root) in deny
    for r in ms.REFERENCE_REPOS + ms.CALIBRATION_REPOS:
        assert rd(r.root) in deny
    assert rd(ms.HOME / "Development/hypabolic/HypaChef") in deny  # whole repo above a recorded subdir
    assert rd(R1) not in deny
    own = ms.HOME / ".claude" / "projects" / ms.project_name(R1)
    assert rd(own) not in deny and f"Read(/{own}/**)" in s["permissions"]["allow"]
    assert rd(ms.HOME / ".claude" / "projects" / ms.project_name(ms.REPO)) in deny
    allow = s["permissions"]["allow"]
    for t in ms.TOOLS:
        assert f"Bash(./{t}:*)" in allow
        assert not any(r.startswith(f"Bash({t}") for r in s["permissions"]["deny"])
    assert [r for r in allow if r.startswith(("Write", "Edit"))] == ["Write(./work/**)", "Edit(./work/**)"]
    for p in ("./KIT.md", "./LABELLING.md", "./examples/**", "./kit/**", "./lint"):
        assert f"Write({p})" in s["permissions"]["deny"] and f"Edit({p})" in s["permissions"]["deny"]


def test_smoke_settings_keep_the_smoke_dir_readable_and_deny_r1():
    s = ms.build(SMOKE)
    deny = reads(s)
    assert rd(SMOKE) not in deny and rd(str(SMOKE) + "*") not in deny
    assert rd(R1) in deny
    own = ms.project_name(SMOKE)
    assert not any(own in r for r in deny)


def test_check_round_trip(tmp_path):
    out = tmp_path / "settings.json"
    run = lambda *a: subprocess.run([sys.executable, str(SLOP / "kit" / "make_settings.py"), *a],  # noqa: E731
                                    capture_output=True, text=True)
    assert run(str(R1), "--out", str(out)).returncode == 0
    assert run("--check", str(R1), "--out", str(out)).returncode == 0
    d = json.loads(out.read_text())
    d["permissions"]["deny"].remove(rd(SMOKE))
    out.write_text(json.dumps(d))
    r = run("--check", str(R1), "--out", str(out))
    assert r.returncode == 1 and "MISSING" in r.stdout
    assert run(str(R1), "--out", str(out)).returncode == 0
    d = json.loads(out.read_text())
    d["switchModelsOnFlag"] = True
    out.write_text(json.dumps(d))
    r = run("--check", str(R1), "--out", str(out))
    assert r.returncode == 1 and "WRONG" in r.stdout


@pytest.mark.skipif(not (SLOP / "scripts" / "launch_checks.py").is_file(), reason="launcher checks not present")
@pytest.mark.parametrize("author", [R1, SMOKE, R2])
def test_launcher_accepts_the_settings(tmp_path, author):
    out = tmp_path / "settings.json"
    out.write_text(json.dumps(ms.build(author)))
    r = subprocess.run([sys.executable, str(SLOP / "scripts" / "launch_checks.py"), "settings", str(out), str(author)],
                       capture_output=True, text=True, cwd=SLOP)
    assert r.returncode == 0, r.stdout + r.stderr


def test_codex_and_claude_json_are_denied_and_runs_have_their_own_settings_file():
    for author in (R1, SMOKE):
        deny = reads(ms.build(author))
        assert rd(ms.CODEX_HOME) in deny and f"Read(/{ms.CLAUDE_JSON})" in deny
    assert ms.default_out(R1) == SLOP / "kit" / "settings.json"
    assert ms.default_out(SMOKE) == SLOP / "kit" / "settings.smoke.json"


def test_r2_settings_deny_r1_and_keep_resources_read_only():
    s = ms.build(R2)
    deny = reads(s)
    assert rd(R1) in deny and rd(SMOKE) in deny and rd(ms.REPO) in deny
    assert rd(R2) not in deny and rd(str(R2) + "*") not in deny
    own = ms.HOME / ".claude" / "projects" / ms.project_name(R2)
    assert f"Read(/{own}/**)" in s["permissions"]["allow"] and rd(own) not in deny
    assert rd(ms.HOME / ".claude" / "projects" / ms.project_name(R1)) in deny
    for t in ("Edit", "Write"):
        assert f"{t}(./resources/**)" in s["permissions"]["deny"]
    assert ms.default_out(R2) == SLOP / "kit" / "settings.r2.json"
    assert rd(R2) in reads(ms.build(R1))
