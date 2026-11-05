"""v0.4 session harness (docs/V04_WSP_KERNEL.md §7): the PreToolUse hook, cost by session-id delta, the session
audit, the settings templates and the kit-transcript check; at G1 also the freeze records, the WSP-B handover
(agent_loop_lab.v04.handover, scripts/v04_prepare_phase_b.sh) and the launcher's wsp-b mode, statically and on
fakes. No model, no claude and no envs/qwen import: the hook and the kernel sources are read or loaded by file
path only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent_loop_lab.v04 import audit as A
from agent_loop_lab.v04 import cost as C
from agent_loop_lab.v04 import kit as K

REPO = Path(__file__).resolve().parents[1]
HOOK = K.load_hook()
PHASE = "wsp-a"
OUT = f"output/{PHASE}"


@pytest.fixture
def author(tmp_path):
    a = tmp_path.resolve() / "HypaNet-v04-author"
    for d in (f"{OUT}/scratch", "packet", "tools", ".kernel"):
        (a / d).mkdir(parents=True)
    return a


# -- the hook --------------------------------------------------------------------------------------------------

BASH_CASES = [
    (f"./py {OUT}/scratch/x.py", True),
    (f"./py {OUT}/scratch/x.py --k=3 a,b 1e-3 +x @y %z", True),
    (f"./py {OUT}/build_edit.py", True),
    (f"./py ./{OUT}/scratch/x.py", True),
    (f"./py -m py_compile {OUT}/build_edit.py", True),
    (f"./fwd {OUT}/scratch/req_01.json", True),
    (f"./fwd {OUT}/r.json", True),
    (f"./py {OUT}/x.py ; ls ..", False),
    (f"./py {OUT}/x.py;ls", False),
    (f"./py {OUT}/x.py && cat ../HypaNet/README.md", False),
    (f"./py {OUT}/x.py || true", False),
    (f"./py {OUT}/x.py | cat", False),
    (f"./py {OUT}/x.py > {OUT}/o.txt", False),
    (f"./py {OUT}/x.py < /etc/hosts", False),
    (f"./py {OUT}/x.py 2>&1", False),
    (f"./py {OUT}/x.py &", False),
    (f"./py {OUT}/x.py\nls", False),
    (f"./py {OUT}/x.py\n", False),
    (f"./py {OUT}/x.py\r", False),
    (f"./py\t{OUT}/x.py", False),
    (f"./py {OUT}/x.py\tls", False),
    (f"./py `echo {OUT}/x.py`", False),
    (f"./py $(echo {OUT}/x.py)", False),
    ("./py ${HOME}/x.py", False),
    (f"./py {OUT}/x.py $HOME", False),
    ("./py -c print(1)", False),
    ("./py -c 'import os'", False),
    (f"./py -m http.server {OUT}/x.py", False),
    (f"./py -m py_compile {OUT}/x.py extra", False),
    ("./py -m py_compile packet/renderer.py", False),
    ("./py packet/renderer.py", False),
    ("./py tools/hn_kernel/client.py", False),
    (f"./py {OUT}/../../packet/x.py", False),
    (f"./py {OUT}/scratch/../../wsp-b/x.py", False),
    ("./py /etc/x.py", False),
    ("./py output/wsp-b/x.py", False),
    (f"./py {OUT}", False),
    (f"./py {OUT}/notes.txt", False),
    (f"./py {OUT}/*.py", False),
    (f"./py {OUT}/x?.py", False),
    (f"./py {OUT}/{{a,b}}.py", False),
    ("./py ~/x.py", False),
    (f"./py '{OUT}/x.py'", False),
    (f'./py "{OUT}/x.py"', False),
    (f"./py {OUT}/x.py # comment", False),
    (f"./py {OUT}/x\\ y.py", False),
    (f"./py  {OUT}/x.py", False),
    (f"./py {OUT}/x.py ", False),
    (f" ./py {OUT}/x.py", False),
    (f"./fwd {OUT}/a.json {OUT}/b.json", False),
    (f"./fwd {OUT}/a.txt", False),
    ("./fwd ../x.json", False),
    ("./fwd packet/x.json", False),
    ("./fwd -h", False),
    ("./py", False),
    ("./py ", False),
    ("./fwd", False),
    ("py x.py", False),
    ("ls ..", False),
    ("cat packet/SPEC.md", False),
    (f"python {OUT}/x.py", False),
    (f"./pyx {OUT}/x.py", False),
    (f"../py {OUT}/x.py", False),
    (f"HN_PHASE=wsp-b ./py {OUT}/x.py", False),
    (f"env ./py {OUT}/x.py", False),
]


@pytest.mark.parametrize("command, allowed", BASH_CASES)
def test_hook_bash_rule(author, command, allowed):
    why = HOOK.check_bash(command, str(author), PHASE)
    assert (why is None) == allowed, why


def test_hook_refuses_background_bash_and_fwd_records_in_any_case(author):
    """Review findings: claude -p waits for background tasks; APFS resolves FWD/ to fwd/."""
    cmd = f"./py {OUT}/scratch/x.py"
    assert HOOK.decide(_ev(author, "Bash", command=cmd), str(author), PHASE) is None
    for flag in (True, "true", 1):
        why = HOOK.decide(_ev(author, "Bash", command=cmd, run_in_background=flag), str(author), PHASE)
        assert why and "run_in_background" in why
    assert HOOK.decide(_ev(author, "Bash", command=cmd, run_in_background=False), str(author), PHASE) is None
    for d in ("fwd", "FWD", "Fwd"):
        why = HOOK.check_write(str(author / OUT / d / "1" / "response.json"), str(author), PHASE)
        assert why and "fwd/" in why, d
    assert HOOK.check_write(str(author / OUT / "fwd_notes.md"), str(author), PHASE) is None


def test_hook_regex_is_the_contract_regex():
    assert HOOK.COMMAND_RE.pattern == r"\./(py|fwd) [^;&|<>`$\n]+"
    # fullmatch, not match with $: a trailing newline must not slip through
    assert HOOK.COMMAND_RE.fullmatch("./py x.py\n") is None


def _ev(author, tool, cwd=None, **inp):
    return {"hook_event_name": "PreToolUse", "cwd": str(cwd or author), "tool_name": tool, "tool_input": inp}


def test_hook_file_tools(author, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (author / OUT / "scratch" / "link").symlink_to(outside)
    d = lambda tool, **inp: HOOK.decide(_ev(author, tool, **inp), str(author), PHASE)  # noqa: E731
    assert d("Write", file_path=str(author / OUT / "EDIT.md"), content="x") is None
    assert d("Write", file_path=f"{OUT}/scratch/a.py", content="x") is None  # relative to the author dir
    assert d("Edit", file_path=str(author / OUT / "build_edit.py"), old_string="a", new_string="b") is None
    for bad in ("packet/SPEC.md", "tools/hn_kernel/client.py", "py", "fwd", ".venv/bin/python", "output/wsp-b/x",
                f"{OUT}/fwd/1/response.json", f"{OUT}/../wsp-b/x", f"{OUT}/scratch/link/x.py", str(outside / "x"),
                OUT, "CLAUDE.md", ".claude/settings.json"):
        assert d("Write", file_path=bad, content="x") is not None, bad
        assert d("Edit", file_path=bad, old_string="a", new_string="b") is not None, bad
    assert d("NotebookEdit", notebook_path="packet/x.ipynb", new_source="x") is not None
    assert d("Read", file_path=str(author / "packet" / "SPEC.md")) is None
    for bad in (str(tmp_path / "x"), str(author.parent / "HypaNet" / "README.md"), "/etc/hosts",
                f"{OUT}/scratch/link/secret", ".kernel/kernel.sock", "../x"):
        assert d("Read", file_path=bad) is not None, bad
    assert d("Read") is not None
    assert d("Glob", pattern="**/*.py") is None
    assert d("Glob", pattern="*.md", path="packet") is None
    for pat, path in (("../*", None), ("/Users/**", None), ("~/x", None), ("*", ".."), ("*", "/etc")):
        assert d("Glob", pattern=pat, **({"path": path} if path else {})) is not None, (pat, path)
    assert d("Grep", pattern="import", path="packet") is None
    assert d("Grep", pattern="import") is None
    assert d("Grep", pattern="x", path="..") is not None
    assert d("Grep", pattern="x", glob="../**") is not None
    for tool in ("WebFetch", "WebSearch", "Agent", "Task", "ListAgents", "SendMessage", "Skill", "mcp__x__y"):
        assert d(tool) is not None, tool
    assert HOOK.decide(_ev(author, "Read", cwd=author.parent, file_path=str(author / "packet" / "SPEC.md")),
                       str(author), PHASE) is not None
    assert HOOK.decide(_ev(author, "Read", file_path="packet/SPEC.md"), str(author), "../x") is not None


def _run_hook(author, stdin: str, phase: str = PHASE):
    return subprocess.run([sys.executable, "-I", "-B", str(K.HOOK_SOURCE), phase, str(author)], input=stdin,
                          text=True, capture_output=True, timeout=60)


def test_hook_cli_exit_codes_fail_closed(author):
    ok = _run_hook(author, json.dumps(_ev(author, "Bash", command=f"./py {OUT}/x.py")))
    assert ok.returncode == 0 and ok.stdout == "" and ok.stderr == ""
    bad = _run_hook(author, json.dumps(_ev(author, "Bash", command=f"./py {OUT}/x.py ; ls ..")))
    assert bad.returncode == 2 and bad.stderr.startswith("refused:")
    for stdin in ("", "not json", "[]", json.dumps({"cwd": str(author)}), json.dumps(_ev(author, "Bash"))):
        p = _run_hook(author, stdin)
        assert p.returncode == 2, stdin
    p = subprocess.run([sys.executable, "-I", "-B", str(K.HOOK_SOURCE)], input="{}", text=True, capture_output=True)
    assert p.returncode == 2  # no phase argument: blocked, not a crash that would fail open
    # the reasons never echo a path outside the author dir
    for e in (_ev(author, "Read", file_path="/Users/someone/secret"), _ev(author, "Bash", command="cat /etc/hosts")):
        p = _run_hook(author, json.dumps(e))
        assert p.returncode == 2 and "/Users/someone" not in p.stderr and "/etc/hosts" not in p.stderr


def test_hook_installed_location_finds_its_author_dir(author):
    d = author / "tools" / "hn_hook"
    d.mkdir()
    shutil.copyfile(K.HOOK_SOURCE, d / "hook_bash.py")
    p = subprocess.run([sys.executable, "-I", "-B", str(d / "hook_bash.py"), PHASE], text=True, capture_output=True,
                       input=json.dumps(_ev(author, "Bash", command=f"./fwd {OUT}/r.json")), cwd=author)
    assert p.returncode == 0, p.stderr


# -- settings ----------------------------------------------------------------------------------------------------

KNOWN_KEYS = {"autoMemoryEnabled", "disableBypassPermissionsMode", "env", "permissions", "hooks"}


@pytest.mark.parametrize("phase", ["wsp-a", "wsp-b"])
def test_settings_templates(phase):
    path = K.settings_template(phase)
    t = json.loads(path.read_text())
    assert set(t) <= KNOWN_KEYS and t["autoMemoryEnabled"] is False and t["env"] == {"HN_PHASE": phase}
    assert t["permissions"]["defaultMode"] == "dontAsk"
    assert t["permissions"]["deny"].count(K.DENY_MARKER) == 1
    req = K.required_rules(phase)
    assert t["permissions"]["allow"] == req["allow"]  # nothing else is allowed
    assert set(req["deny"]) <= set(t["permissions"]["deny"])
    cmd = K.hook_command(t)
    assert cmd == ('"$CLAUDE_PROJECT_DIR"/.venv/bin/python -I -B "$CLAUDE_PROJECT_DIR"/tools/hn_hook/hook_bash.py '
                   + phase)
    assert "/" + "Users" not in json.dumps(t) and str(REPO) not in json.dumps(t)  # no machine paths in the kit
    if phase == "wsp-b":
        for r in ("Edit(./output/wsp-a/**)", "Write(./output/wsp-a/**)", "Edit(./disclosure/**)"):
            assert r in t["permissions"]["deny"]


def test_render_settings_denies_every_sibling_and_harness_root(tmp_path):
    home = tmp_path.resolve() / "home"
    parent = home / "Development" / "hypabolic"
    author = parent / "HypaNet-v04-author"
    repo = parent / "HypaNet"
    for d in (author, repo, parent / "OtherProject", home / "Development" / "elsewhere", home / ".claude",
              home / ".cache" / "huggingface", home / "Documents"):
        d.mkdir(parents=True)
    (parent / ".DS_Store").write_text("x")
    system = ["/tmp", "/private/tmp", "/var/folders-test", "/private/var/folders-test"]
    s = K.render_settings(K.settings_template("wsp-a"), author_dir=author, repo=repo, home=home, system=system)
    deny = s["permissions"]["deny"]
    assert K.DENY_MARKER not in deny
    for root in (repo, parent / "OtherProject", home / "Development" / "elsewhere", home / ".claude",
                 home / ".cache", home / "Documents", *map(Path, system)):
        for tool in K.FILE_TOOLS:
            assert f"{tool}(/{root}/**)" in deny, (tool, root)
    assert f"Read(/{parent / '.DS_Store'})" in deny  # a file: no /**
    assert not any(str(author) in r for r in deny)  # the author dir and its ancestors are never denied
    assert not any(f"/{p}/**)" in r for p in (home, home / "Development", parent) for r in deny)
    assert set(K.required_rules("wsp-a")["deny"]) <= set(deny)


def test_render_refuses_a_deny_root_over_the_author_dir(tmp_path):
    author = tmp_path.resolve() / "a"
    author.mkdir()
    with pytest.raises(ValueError, match="contains the author dir"):
        K.render_settings(K.settings_template("wsp-a"), author_dir=author, repo=tmp_path / "r", home=tmp_path / "h",
                          system=[str(tmp_path)])


def test_kit_wrappers_and_prompts():
    for w in ("py", "fwd"):
        text = (K.KIT_DIR / w).read_text()
        assert text.startswith("#!/bin/sh\n") and "-I -B" in text and str(REPO) not in text
    assert "tools/hn_kernel/run_py.py" in (K.KIT_DIR / "py").read_text()
    assert "tools/hn_kernel/fwd.py" in (K.KIT_DIR / "fwd").read_text()
    p = (K.KIT_DIR / "prompt_wsp_a.md").read_text()
    for need in ("packet/SPEC.md", "packet/AUTHOR_INSTRUCTIONS.md", "output/wsp-a/", "ITERATIONS.md",
                 "one `./fwd` request per Bash call", "final"):
        assert need in p, need
    for name in ("prompt_continue.md", "prompt_resubmit.md"):
        assert "{PHASE}" in (K.KIT_DIR / name).read_text()
    assert "{ERROR_TEXT}" in (K.KIT_DIR / "prompt_resubmit.md").read_text()


def test_launcher_records_are_per_attempt_and_written_in_order():
    """Review findings: a failure before claude blocked the launch name for good; a killed launcher recorded
    no ceiling; a background job or a hung server kept everything alive."""
    text = (REPO / "scripts" / "v04_launch.sh").read_text()
    assert subprocess.run(["sh", "-n", str(REPO / "scripts" / "v04_launch.sh")]).returncode == 0
    pos = lambda needle: text.index(needle)  # noqa: E731
    for side in ('SETTINGS="$LOGDIR/$ATTEMPT.settings.json"', 'READY="$LOGDIR/$ATTEMPT.server.out"',
                 'SERVER_ERR="$LOGDIR/$ATTEMPT.server.log"'):
        assert side in text
    assert "$NAME.settings.json" not in text and "$NAME.server.out" not in text
    ready_wait = pos("until grep -q '\"ready\": true'")
    launch_record = pos('> "$LOGDIR/$NAME.launch.json"')
    resub_copy = pos('cp "$ERROR_FILE" "$RESUB/$TARGET.r$N.error.txt"')
    claude = pos('claude -p "$PROMPT"')
    assert ready_wait < launch_record < claude and ready_wait < resub_copy < claude
    assert pos("check-disclosure") < pos('SID="$(cat "$SID_FILE")"\n    RESUB=')
    assert text.count('echo "$SID" > "$SID_FILE"') == 1 and pos('echo "$SID" > "$SID_FILE"') > pos("record_session() {")
    assert "ATTEMPT.noinit.stream.log" in text and '"started": false' in text
    assert "V04_WALL_SECONDS" in text and "stop_pid \"$SERVER_PID\" TERM 60" in text and "kill -KILL" in text
    assert '--max-budget-usd "$REMAINING"' in text and '"max_budget_usd": %s' in text


def test_kit_requests_follow_the_forward_schema():
    files = K.kit_files(PHASE)
    for name in (f"{OUT}/scratch/kt_req.json", f"{OUT}/scratch/kt_req_big.json"):
        req = json.loads(files[name])
        assert set(req) <= {"model", "sessions", "return"}
        for s in req["sessions"]:
            assert set(s) == {"id", "blocks", "windows"} and s["windows"][-1] == len(s["blocks"]) - 1
    for rel, text in files.items():
        if rel.endswith(".py"):
            compile(text, rel, "exec")


# -- cost by session-id delta --------------------------------------------------------------------------------------


def _launch(logs: Path, name: str, *results: tuple[str, float]) -> None:
    lines = [json.dumps({"type": "system", "subtype": "init", "session_id": results[0][0] if results else "x"})]
    for sid, total in results:
        lines.append(json.dumps({"type": "result", "subtype": "success", "session_id": sid, "total_cost_usd": total,
                                 "modelUsage": {"claude-fable-5-1": {"costUSD": total}}}))
    (logs / f"{name}.stream.jsonl").write_text("\n".join(lines) + "\n")


def test_cost_is_a_per_session_delta(tmp_path):
    logs = tmp_path
    _launch(logs, "wsp-a", ("S", 10.0))
    _launch(logs, "wsp-a.c1", ("S", 25.5))  # resumed: cumulative
    _launch(logs, "wsp-a.r1", ("S", 30.0))
    _launch(logs, "wsp-b", ("S", 42.0))  # WSP-B resumes the WSP-A session
    _launch(logs, "wsp-f", ("F", 7.0))  # a fresh session
    s = C.summary(logs)
    got = {n: r["cost_usd"] for n, r in s["launches"].items()}
    assert got == {"wsp-a": 10.0, "wsp-a.c1": 15.5, "wsp-a.r1": 4.5, "wsp-b": 12.0, "wsp-f": 7.0}
    assert s["phases"]["wsp-a"]["total_usd"] == 30.0 and s["phases"]["wsp-b"]["total_usd"] == 12.0
    assert s["phases"]["wsp-a"]["remaining_usd"] == 470.0 and s["total_usd"] == 49.0
    assert C.check(s, "wsp-a") == (0, "wsp-a: US$30.0 spent, US$470.0 of US$500.0 left")


def test_cost_multiple_results_in_one_launch_take_the_last(tmp_path):
    _launch(tmp_path, "wsp-a", ("S", 1.0), ("S", 3.0))
    _launch(tmp_path, "wsp-a.c1", ("S", 4.0))
    s = C.summary(tmp_path)
    assert s["launches"]["wsp-a"]["cost_usd"] == 3.0 and s["launches"]["wsp-a.c1"]["cost_usd"] == 1.0


def _killed(logs: Path, name: str, *, model: str = "claude-fable-5-1", turns: int = 3, ceiling: float | None = None,
            after_result: tuple[str, float] | None = None) -> None:
    """A launch SIGKILLed mid-session: assistant turns with per-message usage, no (final) result event."""
    lines = [{"type": "system", "subtype": "init", "session_id": "T"}]
    if after_result:
        lines.append({"type": "result", "session_id": after_result[0], "total_cost_usd": after_result[1]})
    for i in range(turns):
        usage = {"input_tokens": 10, "cache_read_input_tokens": 100_000, "output_tokens": 5,
                 "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1000}}
        for _ in range(2):  # one message spans several assistant events with the same id and usage
            lines.append({"type": "assistant", "message": {"id": f"m{i}", "model": model, "usage": usage, "content": []}})
    (logs / f"{name}.stream.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    if ceiling is not None:
        (logs / f"{name}.launch.json").write_text(json.dumps({"name": name, "max_budget_usd": ceiling}) + "\n")


def test_cost_cap(tmp_path):
    _launch(tmp_path, "wsp-b", ("S", 250.0))
    assert C.check(C.summary(tmp_path), "wsp-b")[0] == C.EXIT_CAP


def test_killed_launch_is_charged_its_ceiling_and_a_resume_does_not_cover_it(tmp_path):
    """Review finding: a resumed launch reports only its own turns, so a killed launch must be charged itself."""
    _launch(tmp_path, "wsp-a", ("T", 100.0))
    _killed(tmp_path, "wsp-a.c1", ceiling=400.0)  # the launcher wrote the record before claude started
    _launch(tmp_path, "wsp-a.c2", ("T", 100.3))  # the resume restarts from the last saved total
    s = C.summary(tmp_path)
    c1 = s["launches"]["wsp-a.c1"]
    assert not c1["resolved"] and c1["cost_usd"] == 400.0 and c1["charged_as"] == "max_budget_usd ceiling"
    assert s["launches"]["wsp-a.c2"]["cost_usd"] == pytest.approx(0.3)
    assert s["phases"]["wsp-a"]["unresolved"] == ["wsp-a.c1"] and s["phases"]["wsp-a"]["total_usd"] == pytest.approx(500.3)
    assert C.check(s, "wsp-a")[0] == C.EXIT_CAP  # never "US$0 and carry on"


def test_killed_launch_without_a_record_is_a_lower_bound_never_zero(tmp_path):
    _launch(tmp_path, "wsp-a", ("T", 10.0))
    _killed(tmp_path, "wsp-a.c1", turns=3)
    s = C.summary(tmp_path)
    c1 = s["launches"]["wsp-a.c1"]
    # input side only, per distinct message: 10 x $10 + 1000 x $20 (1 h write) + 100,000 x $0.25, per MTok
    assert c1["cost_usd"] == pytest.approx(3 * (10 * 10 + 1000 * 20 + 100_000 * 0.25) / 1e6)
    assert c1["charged_as"].startswith("lower bound") and s["phases"]["wsp-a"]["lower_bound_only"] == ["wsp-a.c1"]
    code, msg = C.check(s, "wsp-a")
    assert code == C.EXIT_UNRESOLVED and "lower bound" in msg
    assert C.check(s, "wsp-a", accept_unresolved=True)[0] == 0
    assert s["phases"]["wsp-a"]["total_usd"] > 10.0
    (tmp_path / C.OVERRIDES).write_text(json.dumps({"wsp-a.c1": {"usd": 37.5, "reason": "console usage page"}}))
    s = C.summary(tmp_path)
    assert s["launches"]["wsp-a.c1"]["cost_usd"] == 37.5 and C.check(s, "wsp-a")[0] == 0
    assert s["phases"]["wsp-a"]["unresolved"] == ["wsp-a.c1"]  # still listed


def test_turns_after_the_last_result_make_a_launch_unresolved(tmp_path):
    _killed(tmp_path, "wsp-a", turns=2, ceiling=50.0, after_result=("T", 4.0))
    s = C.summary(tmp_path)
    rec = s["launches"]["wsp-a"]
    assert not rec["resolved"] and rec["turns_after_last_result"] == 2 and rec["cost_usd"] == 50.0
    _killed(tmp_path, "wsp-a", turns=2, ceiling=1.0, after_result=("T", 4.0))
    assert C.summary(tmp_path)["launches"]["wsp-a"]["cost_usd"] == 4.0  # never below what the results say


def test_cost_script_exits_4_on_a_lower_bound(tmp_path):
    _killed(tmp_path, "wsp-a", turns=1)
    run = lambda *a: subprocess.run([sys.executable, str(REPO / "scripts" / "v04_cost.py"), "--logs", str(tmp_path), *a],  # noqa: E731
                                    capture_output=True, text=True)
    assert run("--phase", "wsp-a", "--remaining").returncode == 4
    p = run("--phase", "wsp-a", "--remaining", "--accept-unresolved")
    assert p.returncode == 0 and float(p.stdout) < 500.0


@pytest.mark.skipif(not (REPO / "experiments" / "v0_3" / "logs" / "v03-phase-b.stream.jsonl").is_file(),
                    reason="v0.3 transcripts absent")
def test_price_table_reproduces_logged_costs():
    for log in sorted((REPO / "experiments" / "v0_3" / "logs").glob("*.stream.jsonl")):
        for ev in C._events(log):
            for model, u in (ev.get("modelUsage") or {}).items() if ev.get("type") == "result" else ():
                p = C.PRICES_PER_MTOK[model]
                est = (u["inputTokens"] * p[0] + u["outputTokens"] * p[1] + u["cacheCreationEphemeral5mInputTokens"] * p[2]
                       + u["cacheCreationEphemeral1hInputTokens"] * p[3] + u["cacheReadInputTokens"] * p[4]) / 1e6
                assert est == pytest.approx(u["costUSD"], rel=1e-6), (log.name, model)


def test_cost_script_remaining_and_check(tmp_path):
    _launch(tmp_path, "wsp-a", ("S", 123.456))
    run = lambda *a: subprocess.run([sys.executable, str(REPO / "scripts" / "v04_cost.py"), "--logs", str(tmp_path), *a],  # noqa: E731
                                    capture_output=True, text=True)
    p = run("--phase", "wsp-a", "--remaining")
    assert p.returncode == 0 and p.stdout.strip() == "376.54"
    _launch(tmp_path, "wsp-a.c1", ("S", 500.0))
    assert run("--phase", "wsp-a", "--check").returncode == 3
    assert run("--phase", "wsp-a", "--remaining").returncode == 3


V03_LOGS = REPO / "experiments" / "v0_3" / "logs"


@pytest.mark.skipif(not (V03_LOGS / "v03-phase-b.stream.jsonl").is_file(), reason="v0.3 transcripts absent")
def test_delta_accounting_reproduces_the_corrected_v03_cost():
    """Plan revision 11: v0.3's hand-written work cost US$42.73 (A) and US$21.89 (B), not US$122.24."""
    s = C.account(sorted(V03_LOGS.glob("*.stream.jsonl")), phases=("v03-phase-a", "v03-phase-b"))
    assert round(s["phases"]["v03-phase-a"]["total_usd"], 2) == 42.73
    assert round(s["phases"]["v03-phase-b"]["total_usd"], 2) == 21.89


# -- the audit ------------------------------------------------------------------------------------------------------


def _install_fake_tools(author: Path) -> None:
    src = A.AUTHOR_SRC
    for sub, files in (("hn_kernel", [p.name for p in src.glob("*.py") if p.name != "__init__.py"]),
                       ("hn_guard", ["__init__.py", "guards_v04.py"])):
        (author / "tools" / sub).mkdir(parents=True)
        for f in files:
            shutil.copyfile((src / f) if sub == "hn_kernel" else (src / "hn_guard" / f), author / "tools" / sub / f)
    (author / "tools" / "hn_hook").mkdir()
    shutil.copyfile(K.HOOK_SOURCE, author / "tools" / "hn_hook" / "hook_bash.py")
    for w in ("py", "fwd"):
        shutil.copyfile(K.KIT_DIR / w, author / w)
    for d, _dirs, files in os.walk(author / "tools"):
        for f in files:
            os.chmod(os.path.join(d, f), 0o444)


def _transcript(path: Path, author: Path, calls: list[tuple[str, dict, bool]], results: dict[str, str] | None = None):
    lines = [{"type": "system", "subtype": "init", "cwd": str(author), "tools": K.KIT_TOOLS, "mcp_servers": [],
              "permissionMode": "dontAsk", "model": "claude-fable-5-1"}]
    denials = []
    for i, (tool, inp, denied) in enumerate(calls):
        tid = f"t{i}"
        lines.append({"type": "assistant", "message": {"model": "claude-fable-5-1", "content": [
            {"type": "tool_use", "id": tid, "name": tool, "input": inp}]}})
        lines.append({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": tid, "content": (results or {}).get(tid, "ok"), "is_error": denied}]}})
        if denied:
            denials.append({"tool_name": tool, "tool_use_id": tid, "tool_input": inp})
    lines.append({"type": "result", "subtype": "success", "session_id": "S", "total_cost_usd": 1.0,
                  "permission_denials": denials, "modelUsage": {"claude-opus-5-5": {}}})
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")


def test_audit_flags_on_a_synthetic_transcript(author, tmp_path):
    _install_fake_tools(author)
    (author / OUT / "scratch" / "ok.py").write_text("from wsp import load\nprint(load('x').shape)\n")
    repo = REPO
    calls = [
        ("Read", {"file_path": str(author / "packet" / "SPEC.md")}, False),
        ("Bash", {"command": f"./py {OUT}/scratch/ok.py"}, False),
        ("Bash", {"command": f"./fwd {OUT}/scratch/r.json"}, False),
        ("Bash", {"command": f"./py {OUT}/scratch/ok.py ; ls .."}, True),  # refused: listed, not executed
        ("Bash", {"command": "ls .."}, False),  # executed: a violation
        ("Write", {"file_path": str(author / "packet" / "SPEC.md"), "content": "x"}, False),
        ("Read", {"file_path": str(repo / "agent_doom_loop_detector_plan_v0_4.md")}, True),
        ("Read", {"file_path": f"{author.parent}/HypaNet/datasets/v0_4/val.jsonl"}, False),
        ("WebFetch", {"url": "https://example.com"}, False),
        ("Read", {"file_path": str(author / "packet" / "AUTHOR_INSTRUCTIONS.md")}, False),
    ]
    log = tmp_path / "wsp-a.stream.jsonl"
    _transcript(log, author, calls, results={"t9": f"see {repo}/envs/qwen/hypaqwen/model.py"})
    rep = A.audit(log, author, PHASE, repo=repo)
    v = rep["violations"]
    executed = [x for x in v if x.startswith("[EXECUTED]")]
    assert any("Bash" in x and "ls .." in x for x in executed)
    assert any(x.startswith("[refused] Bash") for x in v)  # the compound command, blocked
    assert any("Write" in x and "packet/SPEC.md" in x for x in executed)
    assert any(x.startswith("[refused] Read") and "refers to" in x for x in v)
    assert any("datasets/" in x for x in executed) and any("'HypaNet'" in x for x in executed)
    assert any("WebFetch" in x and "outside the author kit" in x for x in executed)
    assert any("the session was shown" in x and str(repo) in x for x in executed)  # a broken seal
    allowed = ("scratch/ok.py", "scratch/r.json", "AUTHOR_INSTRUCTIONS")
    assert not [x for x in executed if any(a in x for a in allowed)], executed  # the kit calls pass
    assert rep["fwd_calls_executed"] == 1 and len(rep["permission_denials"]) == 2
    assert set(rep["models"]) == {"claude-fable-5-1", "claude-opus-5-5"}
    assert f"{OUT}/scratch/ok.py" in rep["scanned_files"] and not rep["file_findings"]


def test_audit_executed_then_deleted_script_and_transcript_versions(author, tmp_path):
    """Review finding: a script run and then deleted was skipped silently."""
    _install_fake_tools(author)
    gone = f"{OUT}/scratch/gone.py"
    calls = [
        ("Write", {"file_path": str(author / gone), "content": LOOP_BAD}, False),
        ("Bash", {"command": f"./py {gone}"}, False),
        ("Write", {"file_path": str(author / OUT / "scratch" / "kept.py"), "content": "print(1)\n"}, False),
        ("Edit", {"file_path": str(author / OUT / "scratch" / "kept.py"), "old_string": "print(1)",
                  "new_string": "exec('print(2)')"}, False),
        ("Bash", {"command": f"./py {OUT}/scratch/kept.py"}, False),
    ]
    (author / OUT / "scratch" / "kept.py").write_text("print(1)\n")  # rewritten back on disk afterwards
    log = tmp_path / "wsp-a.stream.jsonl"
    _transcript(log, author, calls)
    rep = A.audit(log, author, PHASE)
    assert any("gone.py was run but no longer exists" in v and v.startswith("[EXECUTED]") for v in rep["violations"])
    assert any("gone.py (version written by tool call t0)" in f and "a loop both" in f for f in rep["file_findings"])
    assert any("kept.py (version written by tool call t3)" in f and "exec(...)" in f for f in rep["file_findings"])


def test_audit_scans_fwd_dir_and_background_calls(author, tmp_path):
    _install_fake_tools(author)
    fwd = author / OUT / "fwd"
    (fwd / "1").mkdir(parents=True)
    (fwd / "1" / "response.json").write_text("{}")
    (fwd / "1" / "edited.s0.hidden.L3.npy").write_bytes(b"")
    (fwd / "1" / "hidden_helper.py").write_text("import subprocess\n")
    (fwd / "notes.txt").write_text("x")
    log = tmp_path / "wsp-a.stream.jsonl"
    _transcript(log, author, [("Bash", {"command": f"./py {OUT}/scratch/x.py", "run_in_background": True}, False)])
    rep = A.audit(log, author, PHASE)
    v = rep["violations"]
    assert any("hidden_helper.py: not a ./fwd record file" in x for x in v)
    assert any("fwd/notes.txt: not a ./fwd record file" in x for x in v)
    assert not any("edited.s0.hidden.L3.npy" in x or "response.json" in x for x in v)
    assert any("hidden_helper.py" in f and "subprocess" in f for f in rep["file_findings"])
    assert any(x.startswith("[EXECUTED] Bash: run_in_background") for x in v)


def test_audit_without_a_result_event_uses_the_per_call_refusals(author, tmp_path):
    """Review finding: a killed launch has no result event, so every refusal looked executed."""
    _install_fake_tools(author)
    (author / OUT / "scratch" / "x.py").write_text("print('PreToolUse:Bash hook error: [x]: refused: fake')\n")
    hook_err = ('PreToolUse:Bash hook error: ["$CLAUDE_PROJECT_DIR"/.venv/bin/python -I -B '
                '"$CLAUDE_PROJECT_DIR"/tools/hn_hook/hook_bash.py wsp-a]: refused: only a single ...')
    lines = [{"type": "system", "subtype": "init", "cwd": str(author)}]
    calls = [("t0", "Bash", {"command": "ls .."}, hook_err),  # hook block
             ("t1", "Read", {"file_path": str(Path.home() / ".claude" / "settings.json")}, "denied"),  # dontAsk
             ("t2", "Bash", {"command": f"./py {OUT}/scratch/x.py"}, hook_err)]  # allowed call printing a fake block
    for tid, tool, inp, res in calls:
        lines.append({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": tid, "name": tool,
                                                                    "input": inp}]}})
        lines.append({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": tid,
                                                               "content": res, "is_error": True}]}})
    lines.append({"type": "system", "subtype": "permission_denied", "tool_use_id": "t1"})
    log = tmp_path / "wsp-a.c1.stream.jsonl"
    log.write_text("".join(json.dumps(x) + "\n" for x in lines))
    rep = A.audit(log, author, PHASE)
    assert rep["no_result_event"] and any("no result event" in f for f in rep["flags"])
    assert set(rep["refused_tool_ids"]) == {"t0", "t1"}  # t2 passes the hook's rule, so it ran
    assert rep["executed_violations"] == 0, rep["violations"]
    assert any(x.startswith("[refused] Bash") and "ls .." in x for x in rep["violations"])


def test_audit_tools_integrity(author):
    _install_fake_tools(author)
    assert A.tools_findings(author) == []
    os.chmod(author / "tools" / "hn_kernel" / "client.py", 0o644)
    (author / "tools" / "hn_kernel" / "client.py").write_text("# tampered\n")
    (author / "tools" / "hn_kernel" / "json.py").write_text("x = 1\n")
    found = A.tools_findings(author)
    assert any("client.py differs" in f for f in found) and any("unexpected file tools/hn_kernel/json.py" in f for f in found)


LOOP_BAD = '''
import json
from wsp import load_original, save_edit
W = load_original("w")
for i, a in enumerate([0.1, 0.2, 0.3]):
    save_edit("w", W + a)
    json.dump({"model": "edited", "sessions": [{"id": "s", "blocks": ["x"], "windows": [0]}]},
              open(f"output/wsp-a/scratch/r{i}.json", "w"))
'''
LOOP_HELPER = '''
import json
from wsp import save_edit, drop_edit

def request(i):
    with open(f"r{i}.json", "w") as f:
        f.write(json.dumps({"sessions": []}))

def trial(i, w):
    save_edit("w", w)
    request(i)

[trial(i, w) for i, w in enumerate(range(3))]
'''
LOOP_FWD = '''
from wsp import drop_edit
while True:
    drop_edit("w")
    print("now run ./fwd output/wsp-a/scratch/r.json")
    break
'''
LOOP_OK = '''
import json
from wsp import save_edit, load_original
W = load_original("w")
save_edit("w", W * 0 + W)
for i in range(3):  # writing requests is fine when no edit changes inside the loop
    json.dump({"sessions": [{"id": str(i), "blocks": ["x"], "windows": [0]}]}, open(f"r{i}.json", "w"))
for n in ["a", "b"]:  # editing is fine when the loop writes no request
    save_edit(n, load_original(n))
'''


LOOP_ALIAS = '''
import json
from wsp import save_edit as keep
for i in range(3):
    keep("w", i)
    json.dump({"sessions": []}, open(f"r{i}.json", "w"))
'''
LOOP_MODULE_ALIAS = '''
import json
import wsp as W
k = W.save_edit
for i in range(3):
    k("w", i)
    json.dump({"sessions": []}, open(f"r{i}.json", "w"))
'''
LOOP_GETATTR = '''
import json, wsp
for i in range(3):
    getattr(wsp, "save_edit")("w", i)
    json.dump({"sessions": []}, open(f"r{i}.json", "w"))
'''
LOOP_FOLDED_KEY = '''
import json
from wsp import save_edit
K = "sess" + "ions"
for i in range(3):
    save_edit("w", i)
    json.dump({K: []}, open(f"r{i}.json", "w"))
'''


@pytest.mark.parametrize("src, n", [(LOOP_BAD, 1), (LOOP_HELPER, 1), (LOOP_FWD, 1), (LOOP_OK, 0), (LOOP_ALIAS, 1),
                                    (LOOP_MODULE_ALIAS, 1), (LOOP_GETATTR, 1), (LOOP_FOLDED_KEY, 1)],
                         ids=["inline", "through-helpers", "invokes-fwd", "allowed", "import-alias", "module-alias",
                              "getattr", "folded-key"])
def test_ast_loop_rule(src, n):
    import ast

    found = A.loop_findings(ast.parse(src))
    assert len(found) == n, found


def test_loop_rule_follows_helpers_in_sibling_modules():
    """Review finding: the closure stayed inside one file, while ./py resolves sibling helpers at run time."""
    import ast

    helper = ast.parse("import json\nfrom wsp import save_edit as s\n"
                       "def step(i):\n    s('w', i)\n    json.dump({'sessions': []}, open(f'r{i}.json', 'w'))\n")
    main = ast.parse("from helper import step as go\nimport helper\nfor i in range(3):\n    go(i)\n"
                     "while False:\n    helper.step(1)\n")
    code = A.AuthorCode({"helper": [helper], "main": [main]})
    assert len(A.loop_findings(main, code, "main")) == 2
    assert A.loop_findings(main) == []  # without the module table the helper is invisible (the old behaviour)


def test_loop_rule_computed_keys_are_flagged_for_the_adjudicator():
    import ast

    tree = ast.parse("import json\nfrom wsp import save_edit\nfor i, k in enumerate(KEYS):\n    save_edit('w', i)\n"
                     "    json.dump({k: []}, open(f'r{i}.json', 'w'))\n")
    assert A.loop_findings(tree) == [] and len(A.loop_suspicions(tree)) == 1


def test_dynamic_code_is_a_finding(tmp_path):
    p = tmp_path / "dyn.py"
    p.write_text("import importlib\nexec(open('output/wsp-a/scratch/payload.txt').read())\n"
                 "m = importlib.import_module('sub' + 'process')\n__import__('socket')\n"
                 "open('output/wsp-a/scratch/gen.py', 'w').write('x')\n"
                 "import importlib.util\nspec = importlib.util.spec_from_file_location('g', 'g.py')\n")
    found = A.script_findings(p)
    for need in ("exec(...)", "computed module name", "__import__('socket')", "writes a .py file", "spec_from_file_location"):
        assert any(need in f for f in found), (need, found)


def test_static_scan_of_author_scripts(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("import subprocess\nimport torch.optim\nfrom torch.autograd import grad\nimport socket\n"
                 "import torch\nx = torch.ones(2, requires_grad=True)\ny = x.sum()\ny.backward()\n"
                 "torch.linalg.lstsq(x, x)\nimport random\nrandom.random()\nimport re\nre.compile('a')\n")
    found = A.script_findings(p)
    for need in ("subprocess", "torch.optim", "torch.autograd", "socket", "requires_grad", "backward", "lstsq", "random"):
        assert any(need in f for f in found), (need, found)
    assert not any("re.compile" in f for f in found)
    ok = tmp_path / "ok.py"
    ok.write_text("import json, math\nimport torch\nfrom wsp import load, svd, control_direction\n"
                  "U, S, V = torch.linalg.svd(load('w'))\nprint(json.dumps({'s': float(S[0])}))\n")
    assert A.script_findings(ok) == []


def _ledger(logs: Path, phase: str, entries: list[dict]) -> None:
    cum = 0
    lines = []
    for i, e in enumerate(entries, 1):
        tok = e.get("tokens", 100) if e.get("status", "ok") == "ok" else 0
        cum += tok
        lines.append({"call_id": i, "op": "forward", "status": "ok", "tokens": tok, "cumulative_tokens": cum,
                      "model": "edited", "session_sha256": ["a", "b"], "edit_hash": "e0", **e})
    (logs / f"{phase}.ledger.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (logs / f"{phase}.kernel.jsonl").write_text(
        json.dumps({"seq": 1, "op": "load", "name": "w", "status": "ok", "quota": True, "quota_used": 4096}) + "\n"
        + json.dumps({"seq": 2, "op": "save_edit", "name": "w", "status": "refused"}) + "\n")


def test_ledger_cross_check(tmp_path):
    _ledger(tmp_path, PHASE, [{"edit_hash": f"e{i}"} for i in range(4)]
            + [{"edit_hash": "e9", "session_sha256": ["c"]}, {"status": "refused", "error": "BudgetRefused: x"},
               {"model": "original", "edit_hash": "e7"}])
    s = A.ledger_summary(tmp_path, PHASE)
    assert s["status"] == {"ok": 6, "refused": 1} and s["tokens_ok"] == 600 and not s["over_budget"]
    assert len(s["text_sets_over_edit_limit"]) == 1
    assert next(iter(s["text_sets_over_edit_limit"].values()))["distinct_edit_hashes"] == 4
    assert s["kernel_ops"] == {"load": 1, "save_edit": 1} and s["kernel_refused"] == {"save_edit": 1}
    assert s["quota_tensors_loaded"] == 1 and s["quota_bytes_used"] == 4096
    s = A.ledger_summary(tmp_path, PHASE, budget=500, request_cap=99)
    assert s["over_budget"] and len(s["requests_over_cap"]) == 6


def test_ledger_overlap_clusters_catch_one_swapped_session_per_request(tmp_path):
    """Review finding: changing one session per request split the exact-set grouping."""
    sets = [["a", "b", "c", "d"], ["a", "b", "c", "e"], ["a", "b", "f", "e"], ["a", "g", "f", "e"], ["x", "y"]]
    _ledger(tmp_path, PHASE, [{"edit_hash": f"e{i}", "session_sha256": s} for i, s in enumerate(sets)])
    s = A.ledger_summary(tmp_path, PHASE)
    assert s["text_sets_over_edit_limit"] == {}
    (c,) = s["overlap_clusters_over_edit_limit"]
    assert c["distinct_edit_hashes"] == 4 and c["calls"] == [1, 2, 3, 4]
    assert s["max_edit_states_per_session_text"] == 4


def test_audit_script_end_to_end(author, tmp_path):
    _install_fake_tools(author)
    (author / OUT / "scratch" / "loop.py").write_text(LOOP_BAD)
    log = tmp_path / "wsp-a.stream.jsonl"
    _transcript(log, author, [("Bash", {"command": f"./py {OUT}/scratch/loop.py"}, False)])
    _ledger(tmp_path, PHASE, [{"edit_hash": f"e{i}"} for i in range(5)])
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "v04_audit_session.py"), str(log), str(author), PHASE,
                        "--logs", str(tmp_path)], capture_output=True, text=True)
    assert p.returncode == 1, p.stdout + p.stderr
    rep = json.loads((tmp_path / "wsp-a.stream.audit.json").read_text())
    assert any("loop.py" in f and "a loop both" in f for f in rep["file_findings"])
    assert any("distinct edit states" in f for f in rep["flags"]) and rep["executed_violations"] == 0


# -- the kit-transcript check ----------------------------------------------------------------------------------------


def _kit_transcript(path: Path, author: Path, flip: str | None = None) -> None:
    prompt_steps = []
    out_lines = [{"type": "system", "subtype": "init", "cwd": str(author), "tools": K.KIT_TOOLS, "mcp_servers": [],
                  "permissionMode": "dontAsk", "model": "claude-haiku-5-5"}]
    rel_repo = os.path.relpath(REPO, author)
    steps = [
        ("Read", {"file_path": str(author / "packet" / "SPEC.md")}, False, "spec"),
        ("Write", {"file_path": str(author / OUT / "scratch" / "kt_note.txt"), "content": "kit"}, False, "ok"),
        ("Bash", {"command": f"./py {OUT}/scratch/kt_wsp.py"}, False, "KT_WSP_OK"),
        ("Bash", {"command": f"./py {OUT}/scratch/kt_guard.py"}, False, "KT_GUARD read refused\nKT_GUARD_DONE"),
        ("Bash", {"command": f"./fwd {OUT}/scratch/kt_req.json"}, False, "fwd call 1: model=edited"),
        ("Bash", {"command": f"./fwd {OUT}/scratch/kt_req_big.json"}, False, "fwd: refused (BudgetRefused): x"),
        ("Bash", {"command": f"./py {OUT}/scratch/kt_edit.py"}, False, "KT_EDIT 1 abc abc"),
        ("Read", {"file_path": str(author / rel_repo / "pyproject.toml")}, True, "denied"),
        ("Write", {"file_path": str(author / "packet" / "SPEC.md"), "content": "x"}, True, "denied"),
        ("Write", {"file_path": str(author / "tools" / "hn_kernel" / "kt.py"), "content": "x"}, True, "denied"),
        ("Write", {"file_path": str(author / "py"), "content": "x"}, True, "denied"),
        ("Write", {"file_path": str(author / OUT / "fwd" / "kt.json"), "content": "{}"}, True, "denied"),
        ("Bash", {"command": f"./py {OUT}/scratch/kt_wsp.py ; ls .."}, flip != "compound", "denied"),
        ("Bash", {"command": "ls .."}, True, "denied"),
    ]
    denials = []
    for i, (tool, inp, denied, res) in enumerate(steps):
        tid = f"k{i}"
        out_lines.append({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": tid, "name": tool,
                                                                        "input": inp}]}})
        out_lines.append({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": tid,
                                                                   "content": res}]}})
        if denied:
            denials.append({"tool_use_id": tid})
    out_lines.append({"type": "result", "permission_denials": denials, "session_id": "K", "total_cost_usd": 0.01})
    path.write_text("\n".join(json.dumps(x) for x in out_lines) + "\n")
    del prompt_steps


def test_check_kit_session(author, tmp_path):
    log = tmp_path / "k.stream.jsonl"
    _kit_transcript(log, author)
    rep = K.check_kit_session(log, author, PHASE)
    assert rep["ok"], rep["problems"]
    _kit_transcript(log, author, flip="compound")
    rep = K.check_kit_session(log, author, PHASE)
    assert not rep["ok"] and any("compound ;" in p for p in rep["problems"])


def test_kit_prompt_names_every_required_probe(author):
    text = K.kit_prompt(author, PHASE)
    for need in (f"./py {OUT}/scratch/kt_wsp.py", f"./fwd {OUT}/scratch/kt_req_big.json", "packet/SPEC.md", "ls ..",
                 f"./py {OUT}/scratch/kt_edit.py"):
        assert need in text


def _run_build_module():
    spec = importlib.util.spec_from_file_location("v04_run_build", REPO / "scripts" / "v04_run_build.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_run_build_error_text_comes_from_the_attempt_record(tmp_path):
    """Review finding: stderr after the runner's ERROR: line (uv / interpreter warnings) is never disclosed."""
    m = _run_build_module()
    att = tmp_path / "attempts"
    att.mkdir()
    (att / "0001.json").write_text(json.dumps({"attempt": 1, "error": "old rejection"}))
    before = m.attempt_files(att)
    assert m.attempt_error_text(att, before) is None  # the runner wrote nothing this time
    (att / "0002.json").write_text(json.dumps({"attempt": 2, "error": "build() raised KeyError at line 3 ",
                                               "operator_detail": f"{REPO}/envs/qwen/.venv/lib/x.py warning"}))
    assert m.attempt_error_text(att, before) == "build() raised KeyError at line 3"


def test_disclosure_screen(author, tmp_path):
    clean = "build() raised KeyError at line 3 of <author>/build_edit.py"
    assert K.disclosure_leaks(clean, author) == []
    assert K.disclosure_leaks(f"see {author}/output/wsp-a/build_edit.py and HypaNet-v04-author", author) == []
    for bad in (f"ResourceWarning in {REPO}/envs/qwen/.venv/lib/python3.12/site-packages/x.py",
                f"{Path.home()}/.cache/huggingface/hub", "/private/tmp/claude-501/x/tasks/1.output",
                "/var/folders/ab/T/pytest-of-matthew", "datasets/v0_4/val.jsonl", f"{author.parent}/HypaNet/README.md"):
        assert K.disclosure_leaks(bad, author), bad
    f = tmp_path / "err.txt"
    f.write_text(f"warning: {REPO}/envs/qwen\n")
    p = subprocess.run([sys.executable, "-m", "agent_loop_lab.v04.kit", "check-disclosure", str(f), "--author-dir",
                        str(author)], capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 1 and "DISCLOSURE LEAK" in p.stderr
    f.write_text(clean + "\n")
    p = subprocess.run([sys.executable, "-m", "agent_loop_lab.v04.kit", "check-disclosure", str(f), "--author-dir",
                        str(author)], capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 0, p.stderr


def test_fwd_records_are_cross_checked_against_the_ledger(author, tmp_path):
    m = _run_build_module()
    _ledger(tmp_path, PHASE, [{"edit_hash": "e0"}, {"edit_hash": "e1"}])
    out = author / OUT
    for k, h in ((1, "e0"), (2, "e1")):
        (out / "fwd" / str(k)).mkdir(parents=True)
        (out / "fwd" / str(k) / "response.json").write_text(json.dumps(
            {"call_id": k, "model": "edited", "edit_hash": h, "tokens": 100}))
        (out / "fwd" / str(k) / "request.json").write_text("{}")
    rep = m.fwd_crosscheck(out, tmp_path / f"{PHASE}.ledger.jsonl")
    assert rep["records_checked"] == 2 and rep["problems"] == []
    (out / "fwd" / "2" / "response.json").write_text(json.dumps({"call_id": 2, "model": "edited", "edit_hash": "FORGED",
                                                                  "tokens": 100}))
    (out / "fwd" / "7").mkdir()
    (out / "fwd" / "7" / "response.json").write_text(json.dumps({"forged": True}))
    (out / "fwd" / "1" / "hidden_helper.py").write_text("x = 1\n")
    probs = m.fwd_crosscheck(out, tmp_path / f"{PHASE}.ledger.jsonl")["problems"]
    assert any("fwd/2" in x and "edit_hash" in x for x in probs) and any("fwd/7" in x and "no ok ledger" in x for x in probs)
    assert any("hidden_helper.py" in x for x in probs)


# -- G1: freeze records, WSP-B handover and launch (PREREGISTRATION §6, §7.1, §7.3) -----------------------------

from agent_loop_lab.v04 import disclosure as D  # noqa: E402
from agent_loop_lab.v04 import handover as H  # noqa: E402
from agent_loop_lab.v04.registry import freeze_digest  # noqa: E402

from tests.test_v04_disclosure import EDIT_HASH, V03, _inputs, fake_archive  # noqa: E402


def fake_registry(path: Path, revisions: tuple[str, ...] = (), *, seed: str = "0") -> str:
    """A self-consistent registry (G1 record plus ``revisions``: "add" a new file or "change" an existing one, as
    scripts/v04_freeze.py --revise writes them); returns its freeze_sha256."""
    import yaml

    rec = {"spec_version": "0.4", "globs": ["g/**"], "files": {"a": seed * 64}, "revisions": []}
    rec["freeze_sha256"] = freeze_digest(rec)
    for n, kind in enumerate(revisions, start=1):
        files = dict(rec["files"])
        added, changed = {}, {}
        if kind == "add":
            added[f"new{n}"] = files[f"new{n}"] = f"{n:064x}"
        else:
            changed["a"] = {"before": files["a"], "after": f"{n + 100:064x}"}
            files["a"] = changed["a"]["after"]
        rev = {"date": "2026-11-06", "reason": kind, "previous_freeze_sha256": rec["freeze_sha256"],
               "globs_before": rec["globs"], "added": added, "changed": changed}
        rec = {**rec, "files": files, "revisions": rec["revisions"] + [rev]}
        rec["freeze_sha256"] = freeze_digest(rec)
    path.write_text(yaml.safe_dump(rec))
    return rec["freeze_sha256"]


def test_freeze_info_and_record(tmp_path):
    reg = tmp_path / "registry_v0_4.yaml"
    sha = fake_registry(reg)
    info = H.freeze_info(reg)
    assert info["freeze_sha256"] == info["g1_freeze_sha256"] == sha and info["revisions"] == 0
    logs = tmp_path / "logs"
    logs.mkdir()
    for name, f in (("wsp-a", sha), ("wsp-a.c1", sha), ("wsp-a.r1", sha), ("wsp-b", "x" * 64)):
        (logs / f"{name}.launch.json").write_text(json.dumps({"name": name, "freeze_sha256": f}))
    (logs / "wsp-a.20261106T000000Z.noinit.launch.json").write_text(json.dumps({"freeze_sha256": "y"}))
    rec = H.freeze_record(logs, "wsp-a", reg, required=True)
    assert rec["launches"] == {"wsp-a": sha, "wsp-a.c1": sha, "wsp-a.r1": sha} and rec["consistent"] is True
    assert H.freeze_record(logs, "wsp-b", reg, required=True)["consistent"] is False
    assert H.freeze_record(logs, "wsp-f", reg, required=True)["consistent"] is False  # no launch record at all
    with pytest.raises(H.HandoverError, match="missing"):
        H.freeze_record(logs, "wsp-a", tmp_path / "none.yaml", required=True)
    assert H.freeze_record(logs, "wsp-a", tmp_path / "none.yaml", required=False)["unfrozen"] is True
    reg.write_text(reg.read_text().replace("spec_version: '0.4'", "spec_version: '0.5'"))
    with pytest.raises(H.HandoverError, match="does not match"):
        H.freeze_info(reg)
    p = subprocess.run([sys.executable, "-m", "agent_loop_lab.v04.handover", "freeze-sha", str(reg)], cwd=REPO,
                       capture_output=True, text=True)
    assert p.returncode == 2 and "refusing" in p.stderr


def _journal(d: Path, hashes: list[str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for n, h in enumerate(hashes, start=1):
        (d / f"{n:06d}.safetensors").write_bytes(f"t{n}".encode())
        lines.append(json.dumps({"seq": n, "op": "save", "name": "x", "edit_hash": h, "file": f"{n:06d}.safetensors"}))
    (d / "journal.jsonl").write_text("".join(x + "\n" for x in lines))


def test_seed_journal_starts_wsp_b_in_the_wsp_a_state(tmp_path):
    logs = tmp_path / "logs"
    _journal(logs / "wsp-a" / "journal", ["h1", EDIT_HASH])
    with pytest.raises(H.HandoverError, match="not the archived edit"):
        H.seed_journal(logs, "other")
    got = H.seed_journal(logs, EDIT_HASH)
    assert got["seeded"] == "now" and got["final_edit_hash"] == EDIT_HASH and got["entries"] == 2
    assert H.seed_journal(logs, EDIT_HASH)["seeded"] == "already"
    b = logs / "wsp-b" / "journal"
    assert (b / "journal.jsonl").read_bytes() == (logs / "wsp-a" / "journal" / "journal.jsonl").read_bytes()
    with open(b / "journal.jsonl", "a") as f:  # a WSP-B save after the seed: still seeded, never re-seeded
        f.write(json.dumps({"seq": 3, "op": "drop", "name": "x", "edit_hash": "h3"}) + "\n")
    assert H.check_seeded_journal(logs / "wsp-a" / "journal", b)["final_edit_hash"] == "h3"
    with pytest.raises(H.HandoverError, match="WSP-B entries"):
        H.seed_journal(logs, EDIT_HASH)
    (b / "000001.safetensors").write_bytes(b"forged")
    with pytest.raises(H.HandoverError, match="differs"):
        H.check_seeded_journal(logs / "wsp-a" / "journal", b)
    logs2 = tmp_path / "logs2"
    _journal(logs2 / "wsp-a" / "journal", [EDIT_HASH])
    (logs2 / "wsp-b.ledger.jsonl").write_text("{}\n")
    with pytest.raises(H.HandoverError, match="WSP-B has started"):
        H.seed_journal(logs2, EDIT_HASH)


def test_kernel_freezes_follow_additive_revisions_only(tmp_path):
    """PREREGISTRATION §7.3, §13: G1 and every freeze after an additive revision ran the G1 kernel; a revision that
    changes an existing file (the WSP-F ops) and everything after it did not."""
    import yaml

    from agent_loop_lab.v04.registry import chain_problems

    reg = tmp_path / "registry_v0_4.yaml"
    sha = fake_registry(reg, ("add", "add", "change", "add"))
    rec = yaml.safe_load(reg.read_text())
    assert chain_problems(rec) == []
    info = H.freeze_info(reg)
    chain = [r["previous_freeze_sha256"] for r in rec["revisions"]] + [sha]
    assert info["g1_freeze_sha256"] == chain[0] and info["kernel_freezes"] == chain[:3]
    assert H.kernel_freezes({"freeze_sha256": "x", "revisions": []}) == ["x"]


def test_archive_freeze_must_be_the_registry_g1_kernel(tmp_path):
    reg = tmp_path / "registry_v0_4.yaml"
    sha = fake_registry(reg, ("add",))
    g1 = H.freeze_info(reg)["g1_freeze_sha256"]
    for freeze_sha in (sha, g1):  # the current freeze, or the G1 one it additively revised
        a = D.check_wsp_a_archive(fake_archive(tmp_path / freeze_sha[:8], freeze_sha=freeze_sha))
        assert H.check_archive_freeze(a, registry=reg)["freeze_sha256"] == sha
    a = D.check_wsp_a_archive(fake_archive(tmp_path / "other", freeze_sha="9" * 64))
    with pytest.raises(H.HandoverError, match="not the registry's G1 kernel freeze"):
        H.check_archive_freeze(a, registry=reg)
    mixed = {**D.check_wsp_a_archive(fake_archive(tmp_path / "m", freeze_sha=sha)),
             "freeze_launches": {"wsp-a": sha, "wsp-a.c1": "9" * 64}}
    with pytest.raises(H.HandoverError, match="launch wsp-a.c1"):
        H.check_archive_freeze(mixed, registry=reg)
    with pytest.raises(H.HandoverError, match="no launch records"):
        H.check_archive_freeze({**mixed, "freeze_launches": {}}, registry=reg)
    with pytest.raises(H.HandoverError, match="no registry"):
        H.check_archive_freeze(a, registry=None)
    changed = tmp_path / "changed.yaml"
    fake_registry(changed, ("change",))  # the WSP-F revision: its freeze is not the G1 kernel
    a = D.check_wsp_a_archive(fake_archive(tmp_path / "f", freeze_sha=H.freeze_info(changed)["freeze_sha256"]))
    with pytest.raises(H.HandoverError, match="G1 kernel"):
        H.check_archive_freeze(a, registry=changed)


@pytest.fixture
def wsp_b(tmp_path):
    """A fake registry, a WSP-A archive that ran under it and a disclosure built from v0.3's 200 windows against
    it. Returns ``(archive, disclosure root, logs)``; the registry is ``tmp_path / "registry_v0_4.yaml"``."""
    sha = fake_registry(tmp_path / "registry_v0_4.yaml")
    arch = fake_archive(tmp_path / "archive", freeze_sha=sha)
    info = D.check_wsp_a_archive(arch)
    v03 = (json.loads((V03 / "disclosure" / "DISCLOSURE.json").read_text()),
           json.loads((V03 / "DISCLOSURE_PRIVATE.json").read_text()))
    rows, pos, pred, blocks, wpos, _ = _inputs(v03)
    root = tmp_path / "disc"
    D.write_outputs(root, *D.build(rows, pos, pred, blocks, wpos, *v03, {"wsp_a": info}))
    logs = tmp_path / "logs"
    _journal(logs / "wsp-a" / "journal", ["h1", EDIT_HASH])
    return arch, root, logs


def test_wsp_b_inputs(wsp_b, tmp_path):
    arch, root, logs = wsp_b
    reg = tmp_path / "registry_v0_4.yaml"
    got = H.wsp_b_inputs(arch, root, registry=reg)
    assert got["freeze_sha256"] == H.freeze_info(reg)["freeze_sha256"] == got["wsp_a"]["freeze_sha256"]
    assert got["A_EDIT_HASH"] == EDIT_HASH and got["windows"] == 200
    assert got["A_SHA256"] == hashlib.sha256((arch / "BUILD_RECORD.json").read_bytes()).hexdigest()
    assert got["DISCLOSURE_SHA256"] == D.sha_field(root / "disclosure" / "DISCLOSURE.sha256")
    author = tmp_path / "author"
    (author / "disclosure").mkdir(parents=True)
    with pytest.raises(H.HandoverError, match="byte-identical copy"):
        H.wsp_b_inputs(arch, root, registry=reg, author_dir=author)
    for n in ("DISCLOSURE.json", "DISCLOSURE.sha256"):
        shutil.copy(root / "disclosure" / n, author / "disclosure" / n)
    H.wsp_b_inputs(arch, root, registry=reg, author_dir=author)
    with pytest.raises(H.HandoverError, match="seeds it"):
        H.wsp_b_inputs(arch, root, registry=reg, author_dir=author, logs=logs)
    H.seed_journal(logs, EDIT_HASH)
    assert H.wsp_b_inputs(arch, root, registry=reg, author_dir=author,
                          logs=logs)["journal"]["wsp_a_final_edit_hash"] == EDIT_HASH
    (author / "disclosure" / "extra.txt").write_text("x")
    with pytest.raises(H.HandoverError, match="other files"):
        H.wsp_b_inputs(arch, root, registry=reg, author_dir=author)


def test_wsp_b_inputs_fail_closed_on_the_freeze(wsp_b, tmp_path):
    """Review finding: any FREEZE.json used to pass; now it must be frozen, consistent, the registry's G1 kernel
    freeze, and the bytes the disclosure was built against."""
    arch, root, _ = wsp_b
    reg = tmp_path / "registry_v0_4.yaml"
    good = (arch / "FREEZE.json").read_bytes()
    with pytest.raises(H.HandoverError, match="no registry"):
        H.wsp_b_inputs(arch, root, registry=None)
    with pytest.raises(H.HandoverError, match="no registry"):
        H.wsp_b_inputs(arch, root, registry=None, allow_unfrozen=True)  # a frozen archive needs its registry
    for text in ("{}", "[]", json.dumps({"freeze_sha256": None, "unfrozen": True, "consistent": False})):
        (arch / "FREEZE.json").write_text(text)
        with pytest.raises(H.HandoverError):
            H.wsp_b_inputs(arch, root, registry=reg)
    rec = json.loads(good)
    (arch / "FREEZE.json").write_text(json.dumps({**rec, "launches": {**rec["launches"], "wsp-a.c1": "9" * 64}}))
    with pytest.raises(H.HandoverError, match="G1 kernel freeze"):
        H.wsp_b_inputs(arch, root, registry=reg)
    (arch / "FREEZE.json").write_bytes(good + b" ")  # still valid, but not what the disclosure recorded
    with pytest.raises(H.HandoverError, match="FREEZE.json changed after the disclosure"):
        H.wsp_b_inputs(arch, root, registry=reg)
    (arch / "FREEZE.json").write_bytes(good)
    other = tmp_path / "other.yaml"
    fake_registry(other, ("add",), seed="1")  # another G1 record: the archive's freeze is not in its chain
    with pytest.raises(H.HandoverError, match="G1 kernel freeze"):
        H.wsp_b_inputs(arch, root, registry=other)
    p = subprocess.run([sys.executable, "-m", "agent_loop_lab.v04.handover", "wsp-b-inputs", "--archive", str(arch),
                        "--disclosure-root", str(root)], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 2 and "--registry" in p.stderr  # one of --registry / --kit-unfrozen is required
    p = subprocess.run([sys.executable, "-m", "agent_loop_lab.v04.handover", "wsp-b-inputs", "--archive", str(arch),
                        "--disclosure-root", str(root), "--registry", str(reg)], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_wsp_b_inputs_kit_unfrozen(tmp_path):
    arch = fake_archive(tmp_path / "archive", freeze_sha=None)
    info = D.check_wsp_a_archive(arch, allow_unfrozen=True)
    v03 = (json.loads((V03 / "disclosure" / "DISCLOSURE.json").read_text()),
           json.loads((V03 / "DISCLOSURE_PRIVATE.json").read_text()))
    rows, pos, pred, blocks, wpos, _ = _inputs(v03)
    root = tmp_path / "disc"
    D.write_outputs(root, *D.build(rows, pos, pred, blocks, wpos, *v03, {"wsp_a": info}))
    assert H.wsp_b_inputs(arch, root, registry=None, allow_unfrozen=True)["freeze_sha256"] is None
    with pytest.raises(H.HandoverError, match="registry freeze_sha256"):
        H.wsp_b_inputs(arch, root, registry=None)


def test_wsp_b_inputs_refuse_mismatches(wsp_b, tmp_path):
    arch, root, _ = wsp_b
    reg = tmp_path / "registry_v0_4.yaml"
    w = root / "DISCLOSED_WINDOWS.json"
    w.write_bytes(w.read_bytes().replace(b'"pos": 8', b'"pos": 9', 1))
    with pytest.raises(H.HandoverError, match="DISCLOSED_WINDOWS.json is not the one"):
        H.wsp_b_inputs(arch, root, registry=reg)
    other = fake_archive(tmp_path / "other", freeze_sha=H.freeze_info(reg)["freeze_sha256"])
    rec = json.loads((other / "BUILD_RECORD.json").read_text())
    (other / "BUILD_RECORD.json").write_text(json.dumps({**rec, "edit_state_sha256": "d" * 64}))
    (other / "BUILD_RECORD.sha256").write_text(hashlib.sha256((other / "BUILD_RECORD.json").read_bytes()).hexdigest())
    with pytest.raises(H.HandoverError, match="another WSP-A archive"):
        H.wsp_b_inputs(other, root, registry=reg)
    (root / "disclosure" / "DISCLOSURE.sha256").write_text("0" * 64 + "  DISCLOSURE.json\n")
    with pytest.raises(H.HandoverError, match="does not match"):
        H.wsp_b_inputs(arch, root, registry=reg)


def test_wsp_b_prompt_renders_every_placeholder(wsp_b, tmp_path):
    arch, root, _ = wsp_b
    reg = tmp_path / "registry_v0_4.yaml"
    tmpl = (K.KIT_DIR / "prompt_wsp_b.md").read_text()
    p = subprocess.run([sys.executable, "-m", "agent_loop_lab.v04.handover", "render-wsp-b-prompt", "--template",
                        str(K.KIT_DIR / "prompt_wsp_b.md"), "--archive", str(arch), "--disclosure-root", str(root),
                        "--registry", str(reg)], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    out = p.stdout
    vals = H.wsp_b_inputs(arch, root, registry=reg)
    for k in H.PROMPT_KEYS:
        assert "{" + k + "}" in tmpl and "{" + k + "}" not in out and vals[k] in out
    assert K.disclosure_leaks(out, Path("/nonexistent/HypaNet-v04-author")) == []
    with pytest.raises(H.HandoverError, match="no value"):
        H.render_prompt(tmpl, {**vals, "A_EDIT_HASH": None})
    with pytest.raises(H.HandoverError, match="placeholder"):
        H.render_prompt(tmpl.replace("{DISCLOSURE_SHA256}", "x"), vals)


def test_wsp_b_prompt_states_the_v04_rules():
    p = (K.KIT_DIR / "prompt_wsp_b.md").read_text()
    for need in ("disclosure/DISCLOSURE.json", "output/wsp-b/", "output/wsp-a/", "REVISION.md", "EDIT.md",
                 "build_edit.py", "**one revision**", "2 distinct edit states", '"disclosed_windows"',
                 "label-word logits and their argmax **only**", "`Vnnn`", "## Mechanisms",
                 "| id | tensors | target class | claim |", "validation", "oracle_label", "network_class",
                 "final", "./fwd output/wsp-b/"):
        assert need in p, need
    assert "reference_policy" not in p and str(REPO) not in p and "/" + "Users" not in p
    for name in ("prompt_wsp_a.md", "prompt_wsp_b.md"):
        text = (K.KIT_DIR / name).read_text()
        assert "## Mechanisms" in text and "| id | tensors | target class | claim |" in text
        assert "`label rows`" in text and "`PRODUCTIVE`, `RECOVERY`, `STALLED`, `OSCILLATING`" in text


def test_packet_and_spec_state_the_mechanisms_table():
    from agent_loop_lab.v04 import mechanisms as MECH
    from agent_loop_lab.v04 import packet as P

    text = P.AUTHOR_INSTRUCTIONS
    assert "## Mechanisms" in text and "| id | tensors | target class | claim |" in text
    example = text[text.index("```text\n  ## Mechanisms"):]
    example = "\n".join(x[2:] for x in example.split("\n")[1:8])
    rows = MECH.parse(example)["rows"]  # the packet's own example parses
    assert [r["target_class"] for r in rows] == ["STALLED", "OSCILLATING"] and rows[1]["tensors"] == ["label rows"]
    spec = (REPO / "agent_doom_loop_detector_spec_v0_4.md").read_text()
    s8 = spec[spec.index("## 8. Deliverables and build checks [packet]"):spec.index("## 9. ")]
    assert "## Mechanisms" in s8 and "`id | tensors | target class | claim`" in s8


def real_part(text: str) -> str:
    return text[text.index('  LOGDIR="$REAL_LOGDIR"'):text.index("DEVICE=")]


def test_launcher_wsp_b_mode_is_real():
    text = (REPO / "scripts" / "v04_launch.sh").read_text()
    assert subprocess.run(["sh", "-n", str(REPO / "scripts" / "v04_launch.sh")]).returncode == 0
    assert "WSP-B is not ready" not in text
    pos = lambda needle: text.index(needle)  # noqa: E731
    case_b = text[pos('  wsp-b)\n    TARGET="wsp-b"'):pos('  continue)\n    TARGET')]
    assert 'SESSION_ARGS="--resume $SID"' in case_b and "render-wsp-b-prompt" in case_b and "prompt_wsp_b.md" in case_b
    assert "--session-id" not in case_b
    # every launch of the phase checks the inputs before the server starts, and the server gets --disclosed
    assert pos('if [ "$TARGET" = wsp-b ]; then\n  check_kit_wsp_b\n') < pos("  B_INPUTS=") \
        < pos("hypaqwen.wsp.server --phase")
    assert "wsp-b-inputs --archive" in text and '--logs "$LOGDIR"' in text
    # the archive's freeze is checked against the registry (real: always; kit: when it exists)
    assert text.count("$REG_ARGS") == 2 and 'REG_ARGS="--registry $REPO/experiments/registry_v0_4.yaml"' in real_part(text)
    assert text.count("check_kit_wsp_b\n") == 2 and "V04_KIT=1 may not use the real WSP-A archive" in text
    assert 'DISCLOSED_ARGS="--disclosed $DISCLOSED"' in text and '"$REQUEST_CAP" $DISCLOSED_ARGS)' in text
    assert pos('grep -q "\\"edit_hash\\": \\"$B_A_EDIT\\""') > pos("until grep -q '\"ready\": true'")
    assert "-perm -u+w" in text and "output/wsp-a" in text
    # the freeze: read after --check (real) and written into the launch record before claude starts
    assert pos("scripts/v04_freeze.py --check") < pos('FREEZE_SHA="$(cd "$REPO" && uv run python -m '
                                                      'agent_loop_lab.v04.handover freeze-sha')
    assert '"freeze_sha256": %s' in text and text.count('"$FREEZE_JSON"') == 2
    assert pos('> "$LOGDIR/$NAME.launch.json"') < pos('claude -p "$PROMPT"')
    real = text[pos('  LOGDIR="$REAL_LOGDIR"'):pos("DEVICE=")]
    assert "V04_WSP_A_ARCHIVE" in text and "V04_WSP_A_ARCHIVE" not in real  # kit-only overrides
    assert "V04_DISCLOSURE_ROOT" in text and "V04_DISCLOSURE_ROOT" not in real


def test_prepare_phase_b_script():
    path = REPO / "scripts" / "v04_prepare_phase_b.sh"
    text = path.read_text()
    assert subprocess.run(["sh", "-n", str(path)]).returncode == 0 and os.access(path, os.X_OK)
    text = text[text.index("\nset -eu\n"):]  # the code, after the header comment
    pos = lambda needle: text.index(needle)  # noqa: E731
    order = ["scripts/v04_freeze.py --check", "wsp-b-inputs --archive", "seed-journal --logs",
             'cp "$SRC/DISCLOSURE.json" "$SRC/DISCLOSURE.sha256" "$DST/"', 'chmod -R a-w "$AUTHOR_DIR/output/wsp-a"',
             'mkdir -p "$OUT_B/scratch"', "hypaqwen.wsp.install verify", "hook-selftest --phase wsp-b",
             'HANDOVER="$REPO/experiments/v0_4/wsp_b/HANDOVER.json"']
    assert [pos(x) for x in order[:-1]] == sorted(pos(x) for x in order[:-1]) and order[-1] in text
    assert "install install" not in text  # the tools are the same files in every phase: verified only
    assert text.count("\ncp ") + text.count(" cp ") == 1  # only the two disclosure files go into the author dir
    assert 'chmod 444 "$DST/DISCLOSURE.json" "$DST/DISCLOSURE.sha256"' in text and 'chmod 555 "$DST"' in text
    assert "trap 'chmod 555" in text and "V04_PREPARE_UNFROZEN" in text
    # frozen: the author dir must be the handed-over one, checked before any write (seed, copy, chmod, record)
    check = pos('h["author_dir"] == sys.argv[2] and h["unfrozen_kit_test_dir"] is False')
    first_write = min(pos("seed-journal --logs"), pos('chmod u+w "$AUTHOR_DIR"'), pos('mkdir -p "$(dirname "$HANDOVER")"'))
    assert check < first_write and pos('elif [ -f "$REGISTRY" ]; then') < check
    assert text.count("$REG_ARGS") == 1 and 'REG_ARGS="--kit-unfrozen"' in text


def _fake_author_dir(d: Path) -> Path:
    for f in ("packet/PACKET_MANIFEST.json", "py", "fwd", ".venv/bin/python", "tools/hn_kernel/run_py.py",
              "tools/hn_hook/hook_bash.py"):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text("")
    (d / "output" / "wsp-a").mkdir(parents=True)
    return d


@pytest.mark.parametrize("which, why", [
    ("logs", "real log dir"), ("archive", "real WSP-A archive"), ("disc", "real disclosure root"),
    ("handover", "into the repo"), ("missing", "existing directories"),
])
def test_prepare_phase_b_kit_mode_never_uses_the_real_locations(tmp_path, which, why):
    """Kit mode (V04_KIT=1) works with or without the registry; it refuses the real archive, disclosure root, log
    dir and a handover record inside the repo before anything is checked or written."""
    author = _fake_author_dir(tmp_path / "author")
    for n in ("archive", "disc", "logs"):
        (tmp_path / n).mkdir()
    real = {"logs": REPO / "experiments" / "v0_4" / "logs", "archive": REPO / "weight_program" / "v0_4" / "wsp-a",
            "disc": REPO / "experiments" / "v0_4"}
    env = {**os.environ, "V04_KIT": "1", "V04_WSP_A_ARCHIVE": str(tmp_path / "archive"),
           "V04_DISCLOSURE_ROOT": str(tmp_path / "disc"), "V04_LOGDIR": str(tmp_path / "logs"),
           "V04_HANDOVER": str(tmp_path / "handover.json")}
    if which in real:
        if not real[which].is_dir():
            pytest.skip(f"{real[which]} does not exist in this checkout")
        env[{"logs": "V04_LOGDIR", "archive": "V04_WSP_A_ARCHIVE", "disc": "V04_DISCLOSURE_ROOT"}[which]] = str(real[which])
    elif which == "handover":
        env["V04_HANDOVER"] = str(REPO / "experiments" / "v0_4" / "kit_handover.json")
    else:
        env["V04_LOGDIR"] = str(tmp_path / "nonexistent")
    p = subprocess.run(["sh", str(REPO / "scripts" / "v04_prepare_phase_b.sh"), str(author)], env=env,
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 2 and why in p.stderr, p.stderr
    assert not (author / "disclosure").exists() and not (tmp_path / "logs" / "wsp-b").exists()


def test_run_build_freeze_and_archive_records(tmp_path):
    m = _run_build_module()
    assert m.is_real_build(m.REAL_LOGS, tmp_path) and m.is_real_build(tmp_path, m.REAL_ARCHIVES / "wsp-a")
    assert not m.is_real_build(tmp_path / "logs", tmp_path / "archive")
    arch = tmp_path / "archive"
    (arch / "author_output").mkdir(parents=True)
    tensors = {"model.language_model.layers.3.mlp.down_proj.weight": {}, "model.language_model.embed_tokens.weight": {}}
    (arch / "BUILD_RECORD.json").write_text(json.dumps({"tensors": tensors}))
    (arch / "author_output" / "EDIT.md").write_text(
        "# Edit\n\n## Mechanisms\n\n| id | tensors | target class | claim |\n| --- | --- | --- | --- |\n"
        "| M1 | `model.language_model.layers.3.mlp.down_proj.weight` | STALLED | x |\n| M2 | label rows | RECOVERY | y |\n")
    reg = tmp_path / "registry_v0_4.yaml"
    sha = fake_registry(reg)
    freeze = H.freeze_record(tmp_path, "wsp-a", reg, required=True)
    extra = m.write_records(arch, freeze)
    assert json.loads((arch / "FREEZE.json").read_text())["freeze_sha256"] == sha
    assert D.sha_field(arch / "BUILD_RECORD.sha256") == extra["build_record_sha256"] \
        == hashlib.sha256((arch / "BUILD_RECORD.json").read_bytes()).hexdigest()
    mech = json.loads((arch / "MECHANISMS.json").read_text())
    assert mech["evaluated"] == 2 and mech["q5_possible"]
    assert mech["units"] == ["label rows", "model.language_model.layers.3.mlp.down_proj.weight"]


def test_run_build_real_build_freeze_fails_closed(tmp_path, monkeypatch):
    """Review finding: a real build took any --registry matching its own digest, with zero or inconsistent
    launches; now it must be experiments/registry_v0_4.yaml, pass --check and have consistent launch records."""
    m = _run_build_module()
    reg = tmp_path / "registry_v0_4.yaml"
    sha = fake_registry(reg)
    logs = tmp_path / "logs"
    logs.mkdir()
    with pytest.raises(H.HandoverError, match="not "):
        m.real_build_freeze(reg, logs, "wsp-a")  # not the repo's registry
    monkeypatch.setattr(m, "REGISTRY", reg)
    monkeypatch.setattr(m, "freeze_check", lambda: "MISMATCH: x")
    with pytest.raises(H.HandoverError, match="--check failed"):
        m.real_build_freeze(reg, logs, "wsp-a")
    monkeypatch.setattr(m, "freeze_check", lambda: None)
    with pytest.raises(H.HandoverError, match="no launch record"):
        m.real_build_freeze(reg, logs, "wsp-a")
    (logs / "wsp-a.launch.json").write_text(json.dumps({"freeze_sha256": sha}))
    (logs / "wsp-a.c1.launch.json").write_text(json.dumps({"freeze_sha256": "9" * 64}))
    with pytest.raises(H.HandoverError, match="another registry freeze"):
        m.real_build_freeze(reg, logs, "wsp-a")
    (logs / "wsp-a.c1.launch.json").write_text(json.dumps({"freeze_sha256": sha}))
    rec = m.real_build_freeze(reg, logs, "wsp-a")
    assert rec["consistent"] and rec["unfrozen"] is False and rec["launches"] == {"wsp-a": sha, "wsp-a.c1": sha}
    arch = tmp_path / "archive"
    arch.mkdir()
    (arch / "BUILD_RECORD.json").write_text(json.dumps({"tensors": {}}))
    m.write_records(arch, {**rec, "phase": "wsp-a"})
    assert D.check_freeze_record(json.loads((arch / "FREEZE.json").read_text()), arch / "FREEZE.json") == rec["launches"]


def test_run_build_refuses_a_real_build_without_the_registry(tmp_path, capsys):
    m = _run_build_module()
    author = tmp_path / "author"
    (author / "output" / "wsp-a").mkdir(parents=True)
    logs = tmp_path / "logs"
    (logs / "wsp-a" / "journal").mkdir(parents=True)
    target = m.REAL_ARCHIVES / "test-never-created"
    assert m.main(["wsp-a", "--author-dir", str(author), "--logs", str(logs), "--archive", str(target),
                   "--registry", str(tmp_path / "none.yaml")]) == 2
    assert "PREREGISTRATION §7.3" in capsys.readouterr().err and not target.exists()
