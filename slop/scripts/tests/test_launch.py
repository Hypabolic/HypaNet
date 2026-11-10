"""scripts/launch.sh with a stub `claude` on PATH (no model is called): flags, environment scrub, refusals, budget
arithmetic, smoke, start and continue K, and the no-init path.

Each test builds a fake tree in its temporary directory: ROOT/HypaNet/slop (copies of the launcher and its Python
helpers, stub kit/make_settings.py and kit/kit_build.py, a prompt, settings), ROOT/HypaNet-slop-author and
ROOT/HypaNet-slop-smoke (prepared dirs with a stub ./tok), ROOT/home (HOME, with an empty ~/.claude/projects) and
ROOT/bin/claude (the stub: records argv, environment and the launch records present when it starts, and writes
stream-json with a result line carrying total_cost_usd). Nothing outside ROOT is created.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SLOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SLOP / "scripts"))
import build_sets  # noqa: E402

UID = os.getuid()
EXPECTED_FLAGS_HEAD = ["--model", "claude-fable-5-2", "--effort", "xhigh"]
EXPECTED_FLAGS_TAIL = ["--settings", None, "--setting-sources", "project", "--permission-mode", "dontAsk",
                       "--permission-prompts", "none", "--strict-mcp-config",
                       "--tools", "Read,Glob,Grep,Write,Edit,Bash",
                       "--disallowedTools", "Agent", "Workflow", "WebFetch", "WebSearch", "Artifact", "Skill",
                       "--disable-slash-commands", "--max-budget-usd", None, "--output-format", "stream-json",
                       "--verbose"]
PARENT_ENV = {"CLAUDECODE": "1", "CLAUDE_CODE_EFFORT_LEVEL": "medium", "CLAUDE_EFFORT": "high",
              "CLAUDE_CODE_SESSION_ID": "parent", "CLAUDE_CODE_ENTRYPOINT": "cli", "CLAUDE_PID": "1",
              "CLAUDE_CODE_MESSAGING_TOKEN": "t", "CLAUDE_CODE_SOME_FUTURE_LINK": "1", "CLAUDE_PROJECT_DIR": "/x",
              "CLAUDE_CODE_SUBAGENT_EFFORT": "max", "ANTHROPIC_MODEL": "other", "PYTHONPATH": "/x",
              "VIRTUAL_ENV": "/x", "CLAUDE_CODE_OAUTH_TOKEN": "kept"}
REQUIRED_ENV = {"CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK": "1", "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
                "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1"}

STUB_CLAUDE = r'''#!{python}
import json, os, pathlib, sys
argv = sys.argv[1:]
if argv == ["--version"]:
    print(os.environ.get("STUB_CLAUDE_VERSION", "2.1.322 (Claude Code)"))
    sys.exit(0)
rec = pathlib.Path(os.environ["STUB_RECORD_DIR"])
rec.mkdir(parents=True, exist_ok=True)
n = len(list(rec.glob("call-*.json")))
slop = pathlib.Path(os.environ["STUB_SLOP"])
launch_records = sorted(p.name for p in slop.glob("runs/*/logs/*.launch.json"))
(rec / f"call-{n}.json").write_text(json.dumps({"argv": argv, "env": dict(os.environ), "cwd": os.getcwd(),
                                                "launch_records": launch_records}))
if os.environ.get("STUB_NOINIT"):
    print("API Error: stub refuses to start", file=sys.stderr)
    sys.exit(1)
sid = argv[argv.index("--session-id") + 1] if "--session-id" in argv else argv[argv.index("--resume") + 1]
out = lambda ev: print(json.dumps(ev, separators=(",", ":")))
out({"type": "system", "subtype": "init", "session_id": sid, "model": "claude-fable-5-2"})
out({"type": "assistant", "message": {"id": f"m{n}", "model": "claude-fable-5-2", "content": [
    {"type": "thinking", "thinking": "", "signature": "s"}, {"type": "thinking", "thinking": "plan", "signature": "t"},
    {"type": "text", "text": "working"}], "usage": {"input_tokens": 10, "output_tokens": 120}}})
if os.environ.get("STUB_REFUSAL"):
    out({"type": "assistant", "message": {"id": f"r{n}", "model": "claude-fable-5-2", "content": [],
         "stop_reason": "refusal", "stop_details": {"category": "frontier_llm"}, "usage": {"output_tokens": 0}}})
if not os.environ.get("STUB_KILLED"):
    out({"type": "result", "subtype": os.environ.get("STUB_SUBTYPE", "success"), "session_id": sid,
         "total_cost_usd": float(os.environ.get("STUB_TOTAL", "1.0")), "num_turns": 2,
         "result": os.environ.get("STUB_RESULT", "Files written.\n\nMODEL COMPLETE"),
         "modelUsage": {"claude-fable-5-2": {}}})
'''
STUB_KIT_BUILD = r'''import pathlib, sys
cmd, author, record = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
problems = []
if (author / "HANDOVER.json").read_text() != record.read_text():
    problems.append("HANDOVER.json differs from the record")
if (author / "TAMPERED").exists():
    problems.append("a handed-over file changed")
import json
seed = json.loads(record.read_text()).get("work_seed")
have = sorted(p.name for p in (author / "work").iterdir())
if "--pristine" in sys.argv and seed is None and have:
    problems.append("work/ is not empty")
if "--pristine" in sys.argv and seed is not None and have != sorted(seed):
    problems.append("work/ is not the seed")
for p in problems:
    print("PROBLEM:", p)
sys.exit(1 if problems else 0)
'''
STUB_LEAKAGE = r'''import pathlib, sys
if "--require-sealed" not in sys.argv or "--run" not in sys.argv:
    print("LEAK: the launcher must pass --require-sealed and --run")
    sys.exit(1)
if (pathlib.Path(__file__).parent / "LEAK").exists():
    print("LEAK: sealed-ngrams - 'not-run'")
    sys.exit(1)
print("leakage check: 0 problem(s), 0 waived")
'''
STUB_MAKE_SETTINGS = r'''import pathlib, sys
if (pathlib.Path(__file__).parent / "GAP").exists():
    print("MISSING: Read(//somewhere/**)")
    sys.exit(1)
print("settings cover every current entry")
'''


class Tree:
    def __init__(self, root: Path):
        self.root = root
        self.repo = root / "HypaNet"
        self.slop = self.repo / "slop"
        self.home = root / "home"
        self.author = root / "HypaNet-slop-author"
        self.smoke = root / "HypaNet-slop-smoke"
        self.r2 = root / "HypaNet-slop-author-r2"
        self.calls = root / "calls"

    # ----------------------------------------------------------------------------------------------- building
    def build(self) -> "Tree":
        (self.slop / "scripts").mkdir(parents=True)
        for f in ("launch.sh", "cost.py", "audit.py", "launch_checks.py", "build_sets.py"):
            shutil.copyfile(SLOP / "scripts" / f, self.slop / "scripts" / f)
        for f in ("pyproject.toml", "uv.lock"):
            shutil.copyfile(SLOP / f, self.slop / f)
        (self.slop / ".venv").symlink_to(SLOP / ".venv")
        kit = self.slop / "kit"
        kit.mkdir()
        (kit / "kit_build.py").write_text(STUB_KIT_BUILD)
        (kit / "make_settings.py").write_text(STUB_MAKE_SETTINGS)
        (kit / "leakage.py").write_text(STUB_LEAKAGE)
        (kit / "prompt.md").write_text("Write the network. End with MODEL COMPLETE on its own line.\n")
        (kit / "prompt_continue.md").write_text("The previous launch stopped on an API error. Carry on.\n")
        (kit / "prompt_r2.md").write_text("Revise the network. End with MODEL COMPLETE on its own line.\n")
        (self.home / ".claude" / "projects").mkdir(parents=True)
        for run, d in (("r1", self.author), ("smoke", self.smoke), ("r2", self.r2)):
            rec = {"kind": "hypanet-slop-author", "test": False, "dir": str(d)}
            if run == "r2":
                rec["work_seed"] = ["build.py", "MODEL.md"]
            record = json.dumps(rec, indent=1) + "\n"
            (self.slop / "runs" / run).mkdir(parents=True)
            (self.slop / "runs" / run / "HANDOVER.json").write_text(record)
            (d / "work").mkdir(parents=True)
            if run == "r2":
                (d / "work" / "build.py").write_text("# r1's build\n")
                (d / "work" / "MODEL.md").write_text("# r1's model\n")
            (d / "HANDOVER.json").write_text(record)
            shutil.copyfile(kit / ("prompt_r2.md" if run == "r2" else "prompt.md"), d / "PROMPT.md")
            (d / "tok").write_text("#!/bin/sh\necho \"tok: $*\"\n")
            (d / "tok").chmod(0o755)
        bin_ = self.root / "bin"
        bin_.mkdir()
        (bin_ / "claude").write_text(STUB_CLAUDE.replace("{python}", sys.executable))
        (bin_ / "claude").chmod(0o755)
        self.settings(self.author)
        return self

    def settings(self, author: Path, **override) -> None:
        a = lambda p: "/" + str(p)  # noqa: E731
        deny = [f"Read({a(self.repo)}/**)", f"Read(//tmp/claude-{UID}/**)", f"Read(//private/tmp/claude-{UID}/**)",
                f"Read({a(self.home / 'Development/hypabolic/slop-private')}/**)",
                f"Read({a(self.home / '.codex')}/**)", f"Read({a(self.home / '.claude.json')})",
                f"Read({a(self.home / '.claude/projects' / re.sub(r'[^A-Za-z0-9]', '-', str(self.repo)))}/**)"]
        for _, r in build_sets.SEALED_REPOS:
            deny.append(f"Read({a(self.home / r.relative_to(Path.home()))}/**)")
        for d in sorted(self.root.glob("HypaNet*")):
            if d not in (self.repo, author):
                deny.append(f"Read({a(d)}/**)")
        own = re.sub(r"[^A-Za-z0-9]", "-", str(author))
        allow = ["Read(./**)", "Glob(./**)", "Grep(./**)", "Write(./work/**)", "Edit(./work/**)",
                 "Bash(./tok:*)", "Bash(./lint:*)", "Bash(./score:*)", "Bash(./inspect:*)", "Bash(./python3:*)",
                 f"Read({a(self.home / '.claude/projects' / own)}/**)"]
        s = {"autoMemoryEnabled": False, "switchModelsOnFlag": False, "showThinkingSummaries": True,
             "permissions": {"defaultMode": "dontAsk", "allow": allow, "deny": deny + override.pop("extra_deny", [])}}
        s.update(override)
        name = ("settings.smoke.json" if author.name.startswith("HypaNet-slop-smoke") else "settings.r2.json"
                if author.name.startswith("HypaNet-slop-author-r2") else "settings.json")
        (self.slop / "kit" / name).write_text(json.dumps(s, indent=1))

    def smoke_done(self, total: str = "0") -> None:
        """A finished smoke transcript (start refuses without one, SPEC §12.7)."""
        logs = self.slop / "runs" / "smoke" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        evs = [{"type": "system", "subtype": "init", "session_id": "smoke-sid", "model": "claude-fable-5-2"},
               {"type": "result", "subtype": "success", "session_id": "smoke-sid", "total_cost_usd": float(total),
                "num_turns": 1, "result": "done"}]
        (logs / "smoke.stream.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs))

    def approve(self) -> None:
        h = hashlib.sha256((self.slop / "kit" / "prompt.md").read_bytes()).hexdigest()
        (self.slop / "kit" / "PROMPT_APPROVED.sha256").write_text(f"{h}  kit/prompt.md\n")

    def approve_r2(self, cap: str | None = "250") -> None:
        h = hashlib.sha256((self.slop / "kit" / "prompt_r2.md").read_bytes()).hexdigest()
        (self.slop / "kit" / "PROMPT_R2_APPROVED.sha256").write_text(f"{h}  kit/prompt_r2.md\n")
        if cap is not None:
            (self.slop / "runs" / "r2" / "CAP_USD").write_text(cap + "\n")
        self.settings(self.r2)

    # ----------------------------------------------------------------------------------------------- running
    def run(self, *args: str, **stub_env: str) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "ANTHROPIC", "PYTEST", "UV_"))}
        env.update(PARENT_ENV)
        env.update({"HOME": str(self.home), "PATH": f"{self.root / 'bin'}:{os.environ['PATH']}",
                    "UV_CACHE_DIR": str(Path(os.environ.get("UV_CACHE_DIR", Path.home() / ".cache" / "uv"))),
                    "STUB_RECORD_DIR": str(self.calls), "STUB_SLOP": str(self.slop)})
        env.update(stub_env)
        return subprocess.run(["sh", str(self.slop / "scripts" / "launch.sh"), *args], capture_output=True,
                              text=True, env=env, timeout=120)

    def call(self, i: int = -1) -> dict:
        files = sorted(self.calls.glob("call-*.json"), key=lambda p: int(p.stem.split("-")[1]))
        return json.loads(files[i].read_text())

    def ncalls(self) -> int:
        return len(list(self.calls.glob("call-*.json"))) if self.calls.is_dir() else 0

    def launch(self, run: str, name: str) -> dict:
        return json.loads((self.slop / "runs" / run / "logs" / f"{name}.launch.json").read_text())


@pytest.fixture
def tree(tmp_path) -> Tree:
    return Tree(tmp_path / "kit-test").build()


def refused(res: subprocess.CompletedProcess, text: str) -> bool:
    return res.returncode == 2 and "refusing:" in res.stderr and text in res.stderr


# ----------------------------------------------------------------------------------------------- the claude flags


def test_every_claude_flag_exists_in_the_installed_help():
    claude = shutil.which("claude")
    if not claude:
        pytest.skip("claude not installed")
    help_text = subprocess.run([claude, "--help"], capture_output=True, text=True, timeout=60).stdout
    for flag in ["-p", "--model", "--effort", "--session-id", "--resume"] + [
            f for f in EXPECTED_FLAGS_HEAD + EXPECTED_FLAGS_TAIL if f and f.startswith("--")]:
        assert re.search(rf"(^|[\s,]){re.escape(flag)}([\s,<]|$)", help_text, re.M), flag
    assert "dontAsk" in help_text and '"none"' in help_text and "xhigh" in help_text and "stream-json" in help_text


# ----------------------------------------------------------------------------------------------- check and refusals


def test_check_passes_and_launches_nothing(tree):
    res = tree.run("check")
    assert res.returncode == 0, res.stderr
    assert "check passed: would launch r1 model=claude-fable-5-2 fallback=none effort=xhigh max_budget_usd=200.00" \
        in res.stdout
    assert "prompt approved" in res.stdout and ": no -- smoke and start refuse" in res.stdout
    assert "smoke launch: not yet run -- start refuses until it has" in res.stdout
    assert tree.ncalls() == 0
    tree.settings(tree.smoke)
    res = tree.run("check", "smoke")
    assert res.returncode == 0, res.stderr
    assert "would launch smoke" in res.stdout and "max_budget_usd=10.00" in res.stdout
    # check writes nothing: no logs dir, no tool-check log
    assert not (tree.slop / "runs" / "r1" / "logs").exists() and not (tree.slop / "runs" / "smoke" / "logs").exists()


def test_start_refuses_before_the_smoke_launch(tree):
    tree.approve()
    assert refused(tree.run("start"), "no smoke transcript")
    tree.smoke_done()
    assert tree.run("start", STUB_TOTAL="1").returncode == 0
    assert tree.ncalls() == 1


def test_refuses_a_failing_leakage_check(tree):
    tree.approve()
    tree.smoke_done()
    (tree.slop / "kit" / "LEAK").write_text("")
    assert refused(tree.run("start"), "leakage.py --require-sealed fails")
    assert refused(tree.run("check"), "leakage.py --require-sealed fails")
    assert tree.ncalls() == 0


def test_start_and_smoke_refuse_without_approval(tree):
    res = tree.run("start")
    assert refused(res, "kit/prompt.md is not approved")
    tree.settings(tree.smoke)
    assert refused(tree.run("smoke"), "kit/prompt.md is not approved")
    (tree.slop / "kit" / "PROMPT_APPROVED.sha256").write_text("0" * 64 + "\n")
    assert refused(tree.run("smoke"), "kit/prompt.md is not approved")
    assert tree.ncalls() == 0


def test_refuses_an_old_or_unreadable_claude(tree):
    tree.approve()
    assert refused(tree.run("start", STUB_CLAUDE_VERSION="2.1.320 (Claude Code)"), "older than 2.1.321")
    assert refused(tree.run("start", STUB_CLAUDE_VERSION="garbage"), "older than 2.1.321")
    assert tree.run("check", STUB_CLAUDE_VERSION="2.1.321 (Claude Code)").returncode == 0
    assert tree.ncalls() == 0


def test_refuses_settings_gaps(tree):
    tree.approve()
    (tree.slop / "kit" / "GAP").write_text("")
    assert refused(tree.run("start"), "does not match kit/make_settings.py")
    (tree.slop / "kit" / "GAP").unlink()
    tree.settings(tree.author, switchModelsOnFlag=True)
    res = tree.run("start")
    assert refused(res, "lacks a required rule") and "switchModelsOnFlag" in res.stderr
    tree.settings(tree.author)
    s = json.loads((tree.slop / "kit" / "settings.json").read_text())
    for rule in (f"Read(/{tree.home / '.codex'}/**)", f"Read(/{tree.home / '.claude.json'})"):
        s["permissions"]["deny"].remove(rule)
        (tree.slop / "kit" / "settings.json").write_text(json.dumps(s))
        res = tree.run("start")
        assert refused(res, "lacks a required rule") and ("codex" in res.stderr or ".claude.json" in res.stderr)
        tree.settings(tree.author)
        s = json.loads((tree.slop / "kit" / "settings.json").read_text())
    (tree.root / "HypaNet-other").mkdir()
    res = tree.run("start")
    assert refused(res, "lacks a required rule") and "HypaNet-other" in res.stderr
    tree.settings(tree.author)
    tree.settings(tree.author, extra_deny=[f"Read(/{tree.root}/HypaNet-slop-*/**)"])
    res = tree.run("start")
    assert refused(res, "lacks a required rule") and "author dir itself" in res.stderr
    assert tree.ncalls() == 0


def test_refuses_a_dir_that_is_not_pristine_or_not_real(tree):
    tree.approve()
    (tree.author / "work" / "build.py").write_text("x = 1\n")
    assert refused(tree.run("start"), "(pristine: work/ must be empty)")
    (tree.author / "work" / "build.py").unlink()
    (tree.author / "TAMPERED").write_text("")
    assert refused(tree.run("start"), "does not verify")
    (tree.author / "TAMPERED").unlink()
    rec = tree.slop / "runs" / "r1" / "HANDOVER.json"
    rec.write_text(rec.read_text().replace('"test": false', '"test": true'))
    assert refused(tree.run("start"), "TEST hand-over")
    rec.write_text(rec.read_text().replace('"test": true', '"test": false'))
    own = tree.home / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(tree.author))
    own.mkdir(parents=True)
    (own / "old.jsonl").write_text("{}\n")
    assert refused(tree.run("start"), "is not empty")
    shutil.rmtree(own)
    (tree.author / "PROMPT.md").write_text("another prompt\n")
    assert refused(tree.run("start"), "PROMPT.md differs")
    shutil.copyfile(tree.slop / "kit" / "prompt.md", tree.author / "PROMPT.md")
    (tree.slop / "kit" / "prompt_continue.md").write_text("Carry on with {THING}.\n")
    assert refused(tree.run("start"), "unfilled placeholder")
    assert tree.ncalls() == 0


def test_refuses_an_author_dir_inside_the_repo(tree):
    inner = tree.repo / "HypaNet-slop-author"
    shutil.copytree(tree.author, inner)
    assert refused(tree.run("check", str(inner)), "outside the repo")


# ----------------------------------------------------------------------------------------------- launches


def test_start_flags_environment_and_records(tree):
    tree.approve()
    tree.smoke_done()
    res = tree.run("start", STUB_TOTAL="37.5", STUB_REFUSAL="1")
    assert res.returncode == 0, res.stderr + res.stdout
    c = tree.call()
    argv = c["argv"]
    assert argv[0] == "-p" and argv[1] == (tree.slop / "kit" / "prompt.md").read_text().rstrip("\n")
    assert argv[2:6] == EXPECTED_FLAGS_HEAD
    assert argv[6] == "--session-id" and re.fullmatch(r"[0-9a-f-]{36}", argv[7])
    tail = argv[8:]
    assert len(tail) == len(EXPECTED_FLAGS_TAIL)
    for want, have in zip(EXPECTED_FLAGS_TAIL, tail):
        assert want is None or want == have, (want, have)
    assert tail[1] == str(tree.slop / "kit" / "settings.json")
    assert tail[tail.index("--max-budget-usd") + 1] == "200.00"
    assert "--fallback-model" not in argv
    env = c["env"]
    for k, v in REQUIRED_ENV.items():
        assert env.get(k) == v, k
    for k in PARENT_ENV:
        if k == "CLAUDE_CODE_OAUTH_TOKEN":
            assert env.get(k) == "kept"
        else:
            assert k not in env, k
    assert not [k for k in env if k.startswith("CLAUDE") and k not in REQUIRED_ENV and k != "CLAUDE_CODE_OAUTH_TOKEN"]
    assert c["cwd"] == str(tree.author.resolve())
    assert c["launch_records"] == ["r1.launch.json"]  # written before claude started
    rec = tree.launch("r1", "r1")
    assert rec["model"] == "claude-fable-5-2" and rec["effort"] == "xhigh" and rec["fallback"] is None
    assert rec["max_budget_usd"] == 200.0 and rec["remaining_cap_usd"] == 200.0 and rec["prompt_approved"] == "yes"
    assert rec["session_id"] == argv[7] and rec["claude_version"].startswith("2.1.322")
    assert (tree.slop / "runs" / "r1" / "SESSION_ID").read_text().strip() == argv[7]
    logs = tree.slop / "runs" / "r1" / "logs"
    launches = [json.loads(x) for x in (logs / "LAUNCHES.jsonl").read_text().splitlines()]
    assert launches[-1]["name"] == "r1" and launches[-1]["started"] is True
    audit = json.loads((logs / "r1.audit.json").read_text())
    assert audit["model_complete"] is True and audit["refusals"] == {"frontier_llm": 1}
    assert audit["thinking"] == {"blocks": 2, "with_text": 1, "share_with_text": 0.5}
    assert audit["launch_cost_usd"] == 37.5 and audit["run_remaining_usd"] == 162.5
    assert "MODEL COMPLETE: the run is over" in res.stdout
    # a second start is refused: the session exists
    assert refused(tree.run("start"), "transcripts exist")


def test_budget_arithmetic_smoke_start_and_continue(tree):
    tree.approve()
    tree.settings(tree.smoke)
    res = tree.run("smoke", STUB_TOTAL="12.345", STUB_RESULT="stopped", STUB_SUBTYPE="error_max_budget_usd")
    assert res.returncode == 0, res.stderr
    argv = tree.call()["argv"]
    assert argv[argv.index("--max-budget-usd") + 1] == "10.00"
    assert tree.call()["cwd"] == str(tree.smoke.resolve())
    smoke_argv = tree.call()["argv"]
    assert smoke_argv[smoke_argv.index("--settings") + 1] == str(tree.slop / "kit" / "settings.smoke.json")
    assert tree.launch("smoke", "smoke")["max_budget_usd"] == 10.0
    assert "SMOKE LAUNCH DONE" in res.stdout
    assert refused(tree.run("smoke"), "one smoke launch")
    # start: the remaining cap after the smoke spend, rounded down to the cent
    tree.settings(tree.author)
    res = tree.run("start", STUB_TOTAL="50", STUB_RESULT="stopped on an API error")
    assert res.returncode == 0, res.stderr
    argv = tree.call()["argv"]
    sid = argv[argv.index("--session-id") + 1]
    assert argv[argv.index("--max-budget-usd") + 1] == "187.65"
    assert "without MODEL COMPLETE" in res.stdout and "a reply with no tool call" in res.stdout
    # continue K: in order only, resuming the r1 session with the fixed continue prompt and the remaining cap
    assert refused(tree.run("continue", "2"), "r1.c1 has not run")
    assert refused(tree.run("continue", "31"), "K must be 1..30")
    (tree.author / "work" / "build.py").write_text("x = 1\n")  # continue does not need a pristine dir
    res = tree.run("continue", "1", STUB_TOTAL="60", STUB_RESULT="stopped again")
    assert res.returncode == 0, res.stderr
    argv = tree.call()["argv"]
    assert argv[1] == (tree.slop / "kit" / "prompt_continue.md").read_text().rstrip("\n")
    assert argv[argv.index("--resume") + 1] == sid and "--session-id" not in argv
    assert argv[argv.index("--max-budget-usd") + 1] == "137.65"
    rec = tree.launch("r1", "r1.c1")
    assert rec["session_total_before_usd"] == 50.0 and rec["max_budget_usd"] == 137.65
    # a killed continuation (no result event) is charged its ceiling
    res = tree.run("continue", "2", STUB_KILLED="1")
    assert res.returncode == 0, res.stderr
    summary = json.loads(subprocess.run(
        [sys.executable, str(tree.slop / "scripts" / "cost.py"), "--logs", str(tree.slop / "runs/smoke/logs"),
         "--logs", str(tree.slop / "runs/r1/logs")], capture_output=True, text=True).stdout)
    assert summary["launches"]["r1.c2"]["charged_as"] == "max_budget_usd ceiling"
    assert summary["launches"]["r1.c2"]["cost_usd"] == 127.65
    assert summary["launches"]["r1.c1"]["cost_usd"] == pytest.approx(10.0)
    assert summary["total_usd"] == pytest.approx(12.345 + 60 + 127.65)
    # the cap is now spent: nothing more launches
    assert refused(tree.run("continue", "3"), "cap")


def test_continue_after_model_complete_is_refused(tree):
    tree.approve()
    tree.smoke_done()
    assert tree.run("start", STUB_TOTAL="20").returncode == 0
    assert refused(tree.run("continue", "1"), "MODEL COMPLETE")


def test_continue_needs_a_session(tree):
    assert refused(tree.run("continue", "1"), "no r1 session yet")


def test_less_than_a_dollar_left_is_refused(tree):
    tree.approve()
    logs = tree.slop / "runs" / "r1" / "logs"
    logs.mkdir(parents=True)
    (logs / "PRIOR_SPEND.json").write_text(json.dumps([{"usd": 199.5, "what": "test"}]))
    assert refused(tree.run("start"), "less than US$1")


def test_no_init_event_is_moved_aside_and_not_charged(tree):
    tree.approve()
    tree.smoke_done()
    res = tree.run("start", STUB_NOINIT="1")
    assert res.returncode == 1 and "NO INIT EVENT" in res.stderr
    logs = tree.slop / "runs" / "r1" / "logs"
    assert not list(logs.glob("*.stream.jsonl")) and not (tree.slop / "runs" / "r1" / "SESSION_ID").exists()
    assert list(logs.glob("r1.noinit.*.launch.json")) and list(logs.glob("r1.noinit.*.SESSION_ID"))
    res = tree.run("start", STUB_TOTAL="5")
    assert res.returncode == 0, res.stderr
    assert tree.launch("r1", "r1")["max_budget_usd"] == 200.0


# ----------------------------------------------------------------------------------------------- round 2


def test_r2_needs_its_cap_and_its_own_approval(tree):
    tree.settings(tree.r2)
    assert refused(tree.run("--run", "r2", "check"), "runs/r2/CAP_USD missing")
    assert not (tree.slop / "runs" / "r2" / "CAP_USD").exists()  # the launcher never writes it
    (tree.slop / "runs" / "r2" / "CAP_USD").write_text("250\n")
    res = tree.run("--run", "r2", "check")
    assert res.returncode == 0, res.stderr
    assert "would launch r2" in res.stdout and "max_budget_usd=250.00" in res.stdout
    assert "smoke launch: not required for r2" in res.stdout and "PROMPT_R2_APPROVED.sha256" in res.stdout
    tree.approve()  # r1's approval does not approve r2's prompt
    assert refused(tree.run("--run", "r2", "start"), "kit/prompt_r2.md is not approved")
    assert refused(tree.run("--run", "r2", "smoke"), "run r2 has no smoke launch")
    assert refused(tree.run("--run", "r3", "check"), "--run must be r1 or r2")
    assert tree.ncalls() == 0


def test_r2_and_r1_dirs_are_not_interchangeable(tree):
    tree.approve()
    tree.approve_r2()
    assert refused(tree.run("--run", "r2", "check", str(tree.author)), "starts with HypaNet-slop-author-r2")
    assert refused(tree.run("check", str(tree.r2)), "is an r2 author dir (use --run r2)")
    assert tree.ncalls() == 0


def test_r2_start_and_continue(tree):
    tree.approve_r2()
    # r1's spend is not charged to r2's cap
    logs1 = tree.slop / "runs" / "r1" / "logs"
    logs1.mkdir(parents=True)
    (logs1 / "PRIOR_SPEND.json").write_text(json.dumps([{"usd": 199.5, "what": "r1"}]))
    (tree.r2 / "work" / "extra.py").write_text("x = 1\n")
    assert refused(tree.run("--run", "r2", "start"), "(pristine: work/ must be exactly the recorded seed")
    (tree.r2 / "work" / "extra.py").unlink()
    res = tree.run("--run", "r2", "start", STUB_TOTAL="40", STUB_RESULT="stopped on an API error")
    assert res.returncode == 0, res.stderr + res.stdout
    c = tree.call()
    argv = c["argv"]
    assert argv[1] == (tree.slop / "kit" / "prompt_r2.md").read_text().rstrip("\n")
    assert argv[argv.index("--settings") + 1] == str(tree.slop / "kit" / "settings.r2.json")
    assert argv[argv.index("--max-budget-usd") + 1] == "250.00"
    assert c["cwd"] == str(tree.r2.resolve()) and c["launch_records"] == ["r2.launch.json"]
    sid = argv[argv.index("--session-id") + 1]
    assert (tree.slop / "runs" / "r2" / "SESSION_ID").read_text().strip() == sid
    rec = tree.launch("r2", "r2")
    assert rec["max_budget_usd"] == 250.0 and rec["prompt_approved"] == "yes"
    audit = json.loads((tree.slop / "runs" / "r2" / "logs" / "r2.audit.json").read_text())
    assert audit["run"] == "r2" and audit["run_remaining_usd"] == 210.0
    assert "launch.sh --run r2 continue <K>" in res.stdout
    assert refused(tree.run("--run", "r2", "start"), "transcripts exist")
    (tree.r2 / "work" / "extra.py").write_text("x = 1\n")  # continue does not need the seed
    res = tree.run("--run", "r2", "continue", "1", STUB_TOTAL="55")
    assert res.returncode == 0, res.stderr
    argv = tree.call()["argv"]
    assert argv[1] == (tree.slop / "kit" / "prompt_continue.md").read_text().rstrip("\n")
    assert argv[argv.index("--resume") + 1] == sid and argv[argv.index("--max-budget-usd") + 1] == "210.00"
    assert tree.launch("r2", "r2.c1")["session_total_before_usd"] == 40.0
    assert "MODEL COMPLETE: the run is over. Archive work/ to runs/r2/" in res.stdout
    assert refused(tree.run("--run", "r2", "continue", "2"), "MODEL COMPLETE")
    assert not (tree.slop / "runs" / "r1" / "SESSION_ID").exists()


def test_r2_continue_needs_an_r2_session(tree):
    tree.approve_r2()
    assert refused(tree.run("--run", "r2", "continue", "1"), "no r2 session yet")
