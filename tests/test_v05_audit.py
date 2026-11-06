"""v0.5 audit (agent_loop_lab.v05.audit) and turn loop (agent_loop_lab.v05.runner) on synthetic streams.

The events follow the shapes of a real ``claude -p --output-format stream-json --verbose`` transcript and
session file (Claude Code 2.1.311 smoke test, 6 Nov 2026). Nothing here starts claude.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest

from agent_loop_lab.v05 import runner as v05_runner
from agent_loop_lab.v05.audit import audit_run
from agent_loop_lab.v05.cost import summary as cost_summary
from agent_loop_lab.v05.session import (
    CLI_PREFIXES, CONTINUE, END_MARKER, MODEL, SYSTEM_PROMPT, claude_argv, child_env, ends_with_marker, parse_turn,
    printable_argv, project_slug,
)

PROMPT = "THE TASK\nWrite every weight.\n```edits\ntok_emb.weight[0, 0] = 1\n```\n"
WORKDIR = "/private/var/folders/xx/T/run-abc"
SID = "11111111-2222-3333-4444-555555555555"
SLEPT: list[float] = []
RECOVERY = ("Output token limit hit. Resume directly \u2014 no apology, no recap of what you were doing. "
            "Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces.")


def NO_SLEEP(seconds: float) -> None:  # noqa: N802 - the runner's sleep, recorded instead of waited
    SLEPT.append(seconds)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------- synthetic events
def init_ev(sid=SID, model=MODEL, cwd=WORKDIR, **over):
    ev = {"type": "system", "subtype": "init", "cwd": cwd, "session_id": sid, "tools": [], "mcp_servers": [],
          "model": model, "permissionMode": "dontAsk", "slash_commands": [], "apiKeySource": "none",
          "claude_code_version": "2.1.311", "output_style": "default",
          "agents": ["claude", "Explore", "general-purpose", "Plan", "web-fetch"], "skills": [],
          "plugins": [{"name": "cc-plugin-sec-default", "path": "builtin", "source": "cc-plugin-sec-default@builtin"}],
          "additional_directories": []}
    ev.update(over)
    return ev


def asst_ev(text, mid="msg_1", sid=SID, model=MODEL, block_type="text", **block):
    b = {"type": block_type, **({"text": text} if block_type == "text" else block)}
    return {"type": "assistant", "message": {"model": model, "id": mid, "type": "message", "role": "assistant",
                                             "content": [b], "stop_reason": None, "usage": {"input_tokens": 2}},
            "parent_tool_use_id": None, "session_id": sid}


def result_ev(total, sid=SID, stop="end_turn", out_tokens=1000, is_error=False, subtype="success"):
    return {"type": "result", "subtype": subtype, "is_error": is_error, "stop_reason": stop, "session_id": sid,
            "total_cost_usd": total, "num_turns": 1,
            "usage": {"input_tokens": 10, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 50,
                      "output_tokens": out_tokens, "output_tokens_details": {"thinking_tokens": out_tokens // 2}},
            "modelUsage": {MODEL: {"costUSD": total}}, "permission_denials": [], "terminal_reason": "completed"}


def write_jsonl(path: Path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


def session_entries(user_texts, *, sid=SID, model=MODEL, attachments=None, system=None, tools=None,
                    meta_texts=(), stops=("max_tokens", "end_turn")):
    ev = []
    atts = attachments if attachments is not None else [
        ("environment", "system", "<system-reminder>\n# Environment\n - Primary working directory: " + WORKDIR
         + "\n</system-reminder>", {}),
        ("model", "system", "<system-reminder>\nYou are powered by the model named Fable 5.1.\n</system-reminder>", {}),
        ("session_context", "user", "<system-reminder>\nAs you answer the user's questions, you can use the "
         "following context:\n# userEmail\nThe user's email address is a@b.c.\n</system-reminder>",
         {"context": {"userEmail": "The user's email address is a@b.c."}}),
        ("date", "system", "<system-reminder>\nToday's date is 2026-11-06.\n</system-reminder>", {}),
        ("credential_org", None, None, {"organizationUuid": "x"}),
    ]
    for i, text in enumerate(user_texts):
        ev.append({"type": "user", "isSidechain": False, "message": {"role": "user", "content": text},
                   "sessionId": sid})
        if i == 0:
            for at, role, txt, extra in atts:
                e = {"type": "attachment", "isSidechain": False, "attachment": {"type": at, **extra}, "sessionId": sid}
                if txt is not None:
                    e["rendered"] = [{"content": txt}]
                    e["renderedRole"] = role
                ev.append(e)
            ev.append({"type": "api-request-shape", "sessionId": sid,
                       "shape": {"system": [{"type": "text", "text": t} for t in (system or [CLI_PREFIXES[0], SYSTEM_PROMPT])],
                                 "tools": tools or []}})
            ev.append({"type": "api-request-blob", "sessionId": sid, "message": {"role": "user", "content": [
                {"type": "text", "text": atts[2][2] + "\n"} if len(atts) > 2 and atts[2][2] else {"type": "text", "text": text},
                {"type": "text", "text": text}]}})
            ev.append({"type": "api-request-blob", "sessionId": sid,
                       "message": {"role": "system", "content": [{"type": "text", "text": "# Environment\n..."}]}})
        else:
            ev.append({"type": "api-request-blob", "sessionId": sid,
                       "message": {"role": "user", "content": [{"type": "text", "text": text}]}})
        ev.append({"type": "api-request", "sessionId": sid, "params": {"model": model, "max_tokens": 128000,
                                                                        "output_config": {"effort": "max"}}})
        ev.append({"type": "assistant", "isSidechain": False, "sessionId": sid,
                   "message": {"model": model, "id": f"msg_{i + 1}", "role": "assistant",
                               "content": [{"type": "text", "text": "x"}],
                               "stop_reason": stops[i] if i < len(stops) else "end_turn"}})
    for t in meta_texts:
        ev.append({"type": "user", "isMeta": True, "isSidechain": False, "message": {"role": "user", "content": t},
                   "sessionId": sid})
    return ev


def build_run(root: Path, *, turns=None, session=None, run="r1", status="finished", launch_msgs=None, sid=SID):
    """A run on disk: PROMPT.txt, run.json, turn transcripts with launch records, a session snapshot."""
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "PROMPT.txt").write_text(PROMPT)
    (root / "PROMPT.sha256").write_text(sha(PROMPT) + "  PROMPT.txt\n")
    turns = turns if turns is not None else [
        [init_ev(), asst_ev("I will write the embeddings.\n```edits\ntok_emb.weight[0, 0] = 1\n"),
         result_ev(5.0, stop="max_tokens")],
        [init_ev(), asst_ev("```edits\nhead.weight[0, 0] = 2\n```\n" + END_MARKER + "\n", mid="msg_2"),
         result_ev(7.5)],
    ]
    for n, events in enumerate(turns, 1):
        write_jsonl(root / "logs" / f"{run}.t{n:03d}.stream.jsonl", events)
        msg = (launch_msgs or {}).get(n, PROMPT if n == 1 else CONTINUE)
        argv = printable_argv(claude_argv(session_id=sid, resume=n > 1, message=msg, max_budget_usd=60.0))
        (root / "logs" / f"{run}.t{n:03d}.launch.json").write_text(json.dumps(
            {"name": f"{run}.t{n:03d}", "message_sha256": sha(msg), "resume": n > 1, "max_budget_usd": 60.0,
             "argv": argv}))
    sess = session if session is not None else session_entries([PROMPT] + [CONTINUE] * (len(turns) - 1))
    write_jsonl(root / "logs" / f"{run}.session.jsonl", sess)
    rec = {"run": run, "status": status, "model": MODEL, "session_id": sid, "workdir": WORKDIR,
           "prompt_sha256": sha(PROMPT), "abandoned_session_ids": [], "turns": []}
    (root / "runs" / run).mkdir(parents=True, exist_ok=True)
    (root / "runs" / run / "run.json").write_text(json.dumps(rec))
    return root


# ---------------------------------------------------------------- audit
def test_clean_run_passes_and_records_everything(tmp_path):
    rep = audit_run(build_run(tmp_path), "r1")
    assert rep["violations"] == [], rep["violations"]
    assert rep["ok"] and rep["end_of_weights_given"]
    assert rep["turns_accepted"] == 2 and rep["continuation_turns"] == 1
    assert rep["cut_off_turns"] == [1]
    assert [t["stop_reason"] for t in rep["turns"]] == ["max_tokens", "end_turn"]
    assert rep["usage_total"]["output_tokens"] == 2000 and rep["usage_total"]["thinking_tokens"] == 1000
    # cumulative total_cost_usd per session: charged 5.0 + 2.5, not 12.5
    assert rep["cost"]["run_usd"] == pytest.approx(7.5)
    assert rep["cost"]["naive_sum_all_runs_usd"] == pytest.approx(12.5)
    sf = rep["session_file"]
    assert sf["system_blocks"] == [CLI_PREFIXES[0], SYSTEM_PROMPT]
    assert sf["attachments"]["environment"] == 1 and sf["attachments"]["session_context"] == 1
    assert [u["is_prompt"] for u in sf["user_messages"]] == [True, False]
    assert rep["init"][0]["tools"] == [] and rep["init"][0]["agents"]


def test_init_listing_tools_mcp_or_skills_fails(tmp_path):
    turns = [[init_ev(tools=["Bash"], mcp_servers=[{"name": "x", "status": "connected"}], skills=["s"],
                      slash_commands=["/x"]),
              asst_ev("done\n" + END_MARKER), result_ev(1.0)]]
    rep = audit_run(build_run(tmp_path, turns=turns), "r1")
    text = " ".join(rep["violations"])
    for k in ("tools", "mcp_servers", "skills", "slash_commands"):
        assert f"init lists {k}" in text
    assert not rep["ok"]


def test_non_builtin_plugin_other_model_and_cwd_fail(tmp_path):
    turns = [[init_ev(model="claude-opus-5-5", cwd="/elsewhere",
                      plugins=[{"name": "p", "source": "p@marketplace"}]),
              asst_ev("x\n" + END_MARKER, model="claude-opus-5-5"), result_ev(1.0)]]
    rep = audit_run(build_run(tmp_path, turns=turns), "r1")
    text = " ".join(rep["violations"])
    assert "non-builtin plugin" in text and "init model" in text and "init cwd" in text
    assert "from model 'claude-opus-5-5'" in text


def test_tool_use_block_fails(tmp_path):
    turns = [[init_ev(), asst_ev("", block_type="tool_use", name="Bash", id="t1", input={}),
              asst_ev("x\n" + END_MARKER), result_ev(1.0)]]
    rep = audit_run(build_run(tmp_path, turns=turns), "r1")
    assert any("tool_use block (Bash)" in v for v in rep["violations"])


def test_wrong_continuation_message_fails(tmp_path):
    sess = session_entries([PROMPT, "Continue. Your edits had an error on line 3."])
    rep = audit_run(build_run(tmp_path, session=sess), "r1")
    assert any("user message 2 is not exactly" in v for v in rep["violations"])
    assert any("neither the prompt nor Continue." in v for v in rep["violations"])


def test_first_message_not_prompt_or_prompt_twice_fails(tmp_path):
    rep = audit_run(build_run(tmp_path, session=session_entries(["hello", CONTINUE])), "r1")
    assert any("first user message is not exactly PROMPT.txt" in v for v in rep["violations"])
    rep = audit_run(build_run(tmp_path, session=session_entries([PROMPT, PROMPT])), "r1")
    assert any("sent more than once" in v for v in rep["violations"])


def test_launch_record_with_other_message_fails(tmp_path):
    rep = audit_run(build_run(tmp_path, launch_msgs={2: "Continue. (val F1 0.3)"}), "r1")
    assert any("launch message sha256" in v for v in rep["violations"])


def test_missing_launch_record_and_session_snapshot_fail(tmp_path):
    root = build_run(tmp_path)
    (root / "logs" / "r1.t002.launch.json").unlink()
    (root / "logs" / "r1.session.jsonl").unlink()
    rep = audit_run(root, "r1")
    text = " ".join(rep["violations"])
    assert "no launch record" in text and "no session snapshot" in text


def test_cc_recovery_is_a_violation_unless_allowed(tmp_path):
    rec_text = ("Output token limit hit. Resume directly — no apology, no recap of what you were doing. "
                "Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces.")
    turns = [
        [init_ev(), asst_ev("part one ```edits\na[0] = 1\n"),
         {"type": "user", "message": {"role": "user", "content": rec_text}, "session_id": SID, "isMeta": True},
         asst_ev("a[1] = 2\n```\n" + END_MARKER, mid="msg_1b"), result_ev(3.0)],
    ]
    sess = session_entries([PROMPT], meta_texts=[rec_text])
    root = build_run(tmp_path, turns=turns, session=sess)
    rep = audit_run(root, "r1")
    assert not rep["ok"]
    assert any("2 API messages in one turn" in v for v in rep["violations"])
    assert any("Claude Code recovery message" in v for v in rep["violations"])
    rep = audit_run(root, "r1", allow_cc_recovery=True)
    assert rep["ok"], rep["violations"]
    assert len(rep["deviations"]) >= 2
    p = parse_turn(root / "logs" / "r1.t001.stream.jsonl")
    assert p["reply_parts"] == ["part one ```edits\na[0] = 1\n", "a[1] = 2\n```\n" + END_MARKER]


def test_other_injected_user_message_is_always_a_violation(tmp_path):
    turns = [[init_ev(), {"type": "user", "message": {"role": "user", "content": "hint: use layer 2"},
                          "session_id": SID}, asst_ev("x\n" + END_MARKER), result_ev(1.0)]]
    rep = audit_run(build_run(tmp_path, turns=turns, session=session_entries([PROMPT])), "r1", allow_cc_recovery=True)
    assert any("harness did not send" in v for v in rep["violations"])


def test_forbidden_context_fails(tmp_path):
    atts = [("nested_memory", "user", "<system-reminder>\nContents of /x/CLAUDE.md\n</system-reminder>", {}),
            ("session_context", "user", "<system-reminder>\nAs you answer the user's questions, you can use the "
             "following context:\n# userEmail\nx\n</system-reminder>", {"context": {"userEmail": "x", "gitStatus": "y"}})]
    rep = audit_run(build_run(tmp_path, session=session_entries([PROMPT, CONTINUE], attachments=atts)), "r1")
    text = " ".join(rep["violations"])
    assert "attachment type 'nested_memory' rendered" in text
    assert "CLAUDE.md" in text
    assert "session_context carries ['gitStatus']" in text


def test_system_blocks_and_tools_in_request_fail(tmp_path):
    sess = session_entries([PROMPT, CONTINUE], system=["You are Claude Code.", SYSTEM_PROMPT, "# Environment"],
                           tools=[{"name": "Bash"}])
    rep = audit_run(build_run(tmp_path, session=sess), "r1")
    text = " ".join(rep["violations"])
    assert "are not the CLI line plus the system prompt" in text and "offered tools" in text


def test_end_marker_early_or_missing_fails(tmp_path):
    turns = [[init_ev(), asst_ev("x\n" + END_MARKER), result_ev(1.0)],
             [init_ev(), asst_ev("more", mid="msg_2"), result_ev(2.0)]]
    rep = audit_run(build_run(tmp_path, turns=turns), "r1")
    text = " ".join(rep["violations"])
    assert "END OF WEIGHTS before the last turn" in text
    assert "run.json says finished" in text


def test_session_id_mismatch_fails(tmp_path):
    other = str(uuid.uuid4())
    turns = [[init_ev(), asst_ev("a"), result_ev(1.0, stop="max_tokens")],
             [init_ev(sid=other), asst_ev("b\n" + END_MARKER, mid="m2", sid=other), result_ev(1.0, sid=other)]]
    rep = audit_run(build_run(tmp_path, turns=turns), "r1")
    assert any("accepted turns use session ids" in v for v in rep["violations"])


def test_failed_attempt_is_recorded_and_checked(tmp_path):
    root = build_run(tmp_path)
    write_jsonl(root / "logs" / "r1.t002.fail1.stream.jsonl",
                [init_ev(), result_ev(5.2, is_error=True, subtype="error_during_execution", out_tokens=0)])
    (root / "logs" / "r1.t002.fail1.launch.json").write_text(json.dumps({"message_sha256": sha("Continue!")}))
    rep = audit_run(root, "r1")
    assert [a["attempt"] for a in rep["failed_attempts"]] == ["r1.t002.fail1"]
    assert any("retry with a different message" in v for v in rep["violations"])


def test_end_marker_detection():
    assert ends_with_marker("x\n" + END_MARKER + "\n") == (True, True)
    assert ends_with_marker("x\n  " + END_MARKER + "  \n\n") == (True, False)
    assert ends_with_marker("x " + END_MARKER) == (False, False)
    assert ends_with_marker(END_MARKER + "\nmore") == (False, False)


def test_command_and_environment():
    argv = claude_argv(session_id=SID, resume=True, message=CONTINUE, max_budget_usd=12.345)
    assert argv[:2] == ["claude", "-p"] and argv[-2:] == ["--", CONTINUE]
    assert argv[argv.index("--tools") + 1] == "" and "--fallback-model" not in argv
    assert argv[argv.index("--model") + 1] == MODEL and argv[argv.index("--effort") + 1] == "max"
    assert argv[argv.index("--system-prompt") + 1] == SYSTEM_PROMPT
    assert argv[argv.index("--resume") + 1] == SID and "--session-id" not in argv
    assert argv[argv.index("--max-budget-usd") + 1] == "12.35"
    for f in ("--strict-mcp-config", "--safe-mode", "--disable-slash-commands", "--verbose"):
        assert f in argv
    env = child_env({"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "x", "HOME": "/h", "PYTHONPATH": "/p"})
    assert "CLAUDECODE" not in env and "CLAUDE_CODE_SESSION_ID" not in env and "PYTHONPATH" not in env
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "128000" and env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert env["CLAUDE_CODE_NO_MODEL_FALLBACK"] == "1" and env["HOME"] == "/h"


# ---------------------------------------------------------------- the turn loop with a fake claude
class FakeClaude:
    """Writes a transcript and appends to the session file the way claude -p does; replies are scripted.

    Each script item: ("reply", text, stop_reason, cumulative_cost); ("parts", [text, ...], stop_reason,
    cumulative_cost) (Claude Code continued a cut reply inside the turn with its recovery message);
    ("fail", cost[, synthetic text[, terminal_reason]]) (no model output); or ("interrupt",) (init only, then
    KeyboardInterrupt)."""

    def __init__(self, home: Path, script):
        self.home, self.script, self.calls = home, list(script), []

    def __call__(self, argv, cwd, env, out, err, wall):
        self.calls.append(argv)
        step = self.script.pop(0)
        sid = argv[argv.index("--resume" if "--resume" in argv else "--session-id") + 1]
        msg = argv[-1]
        sess = self.home / ".claude" / "projects" / project_slug(cwd) / f"{sid}.jsonl"
        sess.parent.mkdir(parents=True, exist_ok=True)
        entries = [{"type": "user", "message": {"role": "user", "content": msg}, "sessionId": sid}]
        if "--resume" not in argv:
            entries.append({"type": "api-request-shape", "sessionId": sid, "shape": {
                "system": [{"type": "text", "text": CLI_PREFIXES[0]}, {"type": "text", "text": SYSTEM_PROMPT}],
                "tools": []}})
        entries.append({"type": "api-request-blob", "sessionId": sid,
                        "message": {"role": "user", "content": [{"type": "text", "text": msg}]}})
        entries.append({"type": "api-request", "sessionId": sid, "params": {"model": MODEL, "max_tokens": 128000,
                                                                             "output_config": {"effort": "max"}}})
        events = [init_ev(sid=sid, cwd=str(cwd))]
        Path(err).write_text("")
        if step[0] == "interrupt":
            write_jsonl(Path(out), events)
            with open(sess, "a") as f:
                f.write("".join(json.dumps(e) + "\n" for e in entries))
            raise KeyboardInterrupt
        if step[0] == "fail":
            res = result_ev(step[1] if len(step) > 1 else 0.0, sid=sid, is_error=True,
                            subtype="error_during_execution", out_tokens=0)
            if len(step) > 3:
                res["terminal_reason"] = step[3]
            events += [{"type": "assistant", "message": {"model": "<synthetic>", "id": "e", "content": [
                {"type": "text", "text": step[2] if len(step) > 2 else "API Error: 529 overloaded"}]},
                "session_id": sid}, res]
        elif step[0] == "parts":
            _, texts, stop, total = step
            for k, text in enumerate(texts):
                mid = f"msg_{len(self.calls)}_{k}"
                if k:
                    events.append({"type": "user", "isSynthetic": True, "session_id": sid,
                                   "message": {"role": "user", "content": [{"type": "text", "text": RECOVERY}]}})
                    entries.append({"type": "user", "isMeta": True, "sessionId": sid,
                                    "message": {"role": "user", "content": RECOVERY}})
                    entries.append({"type": "api-request", "sessionId": sid, "params": {
                        "model": MODEL, "max_tokens": 128000, "output_config": {"effort": "max"}}})
                events.append(asst_ev(text, mid=mid, sid=sid))
                entries.append({"type": "assistant", "sessionId": sid, "message": {
                    "model": MODEL, "id": mid, "content": [{"type": "text", "text": text}],
                    "stop_reason": "max_tokens" if k < len(texts) - 1 else stop}})
            events.append(result_ev(total, sid=sid, stop=stop))
        else:
            _, text, stop, total = step
            mid = f"msg_{len(self.calls)}"
            events += [asst_ev(text, mid=mid, sid=sid), result_ev(total, sid=sid, stop=stop)]
            entries.append({"type": "assistant", "sessionId": sid, "message": {
                "model": MODEL, "id": mid, "content": [{"type": "text", "text": text}], "stop_reason": stop}})
        write_jsonl(Path(out), events)
        with open(sess, "a") as f:
            f.write("".join(json.dumps(e) + "\n" for e in entries))
        return 1 if step[0] == "fail" else 0


@pytest.fixture
def kit(tmp_path, monkeypatch):
    root = tmp_path / "v0_5"
    root.mkdir()
    (root / "PROMPT.txt").write_text(PROMPT)
    (root / "PROMPT.sha256").write_text(sha(PROMPT) + "  PROMPT.txt\n")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    work = tmp_path / "tmp"
    work.mkdir()
    monkeypatch.setattr(v05_runner, "workdir_base", lambda: str(work))
    monkeypatch.setattr(v05_runner, "PATH_WORDS", ())  # pytest's tmp_path names the test
    monkeypatch.setattr(v05_runner, "pytest_scratch", lambda: [])  # pytest's own scratch exists during tests
    monkeypatch.setattr(v05_runner, "context_files_on_path", lambda cwd: [])
    monkeypatch.setattr(v05_runner, "claude_version", lambda: "test")
    return root, home


def test_runner_conversation_retry_and_audit(kit):
    root, home = kit
    fake = FakeClaude(home, [
        ("reply", "design...\n```edits\ntok_emb.weight[0, 0] = 1\n", "max_tokens", 6.0),
        ("fail", 6.1),
        ("reply", "```edits\nhead.weight[0, 0] = 2\n```\n" + END_MARKER + "\n", "end_turn", 8.0),
    ])
    rec = v05_runner.run("r1", root=root, launch=fake, log=lambda s: None, sleep=NO_SLEEP)
    assert rec["status"] == "finished" and len(rec["turns"]) == 2
    assert [c[-1] for c in fake.calls] == [PROMPT, CONTINUE, CONTINUE]
    assert "--session-id" in fake.calls[0] and all("--resume" in c for c in fake.calls[1:])
    logs = root / "logs"
    assert (logs / "r1.t002.fail1.stream.jsonl").exists() and (logs / "r1.t002.fail1.launch.json").exists()
    assert (root / "runs" / "r1" / "replies" / "001.md").read_text().startswith("design")
    idx = json.loads((root / "runs" / "r1" / "replies" / "INDEX.json").read_text())
    assert [(u["turn"], u["cut_off"], u["end_marker"]) for u in idx["units"]] == [(1, True, False), (2, False, True)]
    assert rec["turns"][1]["failed_attempts"] == ["r1.t002.fail1"]
    rep = audit_run(root, "r1")
    assert rep["ok"], rep["violations"]
    assert rep["cost"]["run_usd"] == pytest.approx(8.0)
    with pytest.raises(v05_runner.Refusal):
        v05_runner.run("r1", root=root, launch=fake, log=lambda s: None, sleep=NO_SLEEP)
    assert v05_runner.run("r1", root=root, resume=True, launch=fake, log=lambda s: None, sleep=NO_SLEEP)["status"] == "finished"


def test_runner_resumes_after_interrupt_without_resending(kit):
    root, home = kit
    fake = FakeClaude(home, [("reply", "part 1", "max_tokens", 2.0), ("interrupt",)])
    with pytest.raises(KeyboardInterrupt):
        v05_runner.run("r2", root=root, launch=fake, log=lambda s: None, sleep=NO_SLEEP)
    assert not (root / "runs" / "r2" / ".lock").exists()
    fake.script = [("reply", "done\n" + END_MARKER, "end_turn", 3.0)]
    rec = v05_runner.run("r2", root=root, resume=True, launch=fake, log=lambda s: None, sleep=NO_SLEEP)
    assert rec["status"] == "finished"
    # turn 1 was not sent again; the interrupted turn 2 (no model output) became a failed attempt and was retried
    assert [c[-1] for c in fake.calls] == [PROMPT, CONTINUE, CONTINUE]
    assert (root / "logs" / "r2.t002.fail1.stream.jsonl").exists()
    rep = audit_run(root, "r2")
    assert rep["ok"], rep["violations"]


def test_runner_turn_001_retry_uses_a_new_session(kit):
    root, home = kit
    fake = FakeClaude(home, [("fail", 0.5), ("reply", "w\n" + END_MARKER, "end_turn", 1.0)])
    rec = v05_runner.run("r3", root=root, launch=fake, log=lambda s: None, sleep=NO_SLEEP)
    assert rec["status"] == "finished" and len(rec["abandoned_session_ids"]) == 1
    s1 = fake.calls[0][fake.calls[0].index("--session-id") + 1]
    s2 = fake.calls[1][fake.calls[1].index("--session-id") + 1]
    assert s1 != s2 and rec["session_id"] == s2
    assert audit_run(root, "r3")["ok"]


def test_runner_stops_at_retry_cap_and_cost_cap(kit, monkeypatch):
    root, home = kit
    fake = FakeClaude(home, [("reply", "a", "end_turn", 1.0)] + [("fail", 1.0)] * 4)
    rec = v05_runner.run("r1", root=root, launch=fake, log=lambda s: None, sleep=NO_SLEEP)
    assert rec["status"] == "technical_stop" and len(fake.calls) == 5
    # the cap is summed over all runs: an r2 launch that reached US$500 stops r3 before it sends anything
    write_jsonl(root / "logs" / "r2.t001.stream.jsonl", [init_ev(sid="s2"), asst_ev("x", sid="s2"), result_ev(500.0, sid="s2")])
    assert cost_summary(root / "logs")["cap"]["reached"]
    fake2 = FakeClaude(home, [])
    rec = v05_runner.run("r3", root=root, launch=fake2, log=lambda s: None, sleep=NO_SLEEP)
    assert rec["status"] == "cost_cap" and fake2.calls == []


def test_runner_continuation_cap(kit, monkeypatch):
    root, home = kit
    monkeypatch.setattr(v05_runner, "CONTINUATION_CAP", 2)
    fake = FakeClaude(home, [("reply", f"t{i}", "max_tokens", float(i)) for i in range(1, 6)])
    rec = v05_runner.run("r1", root=root, launch=fake, log=lambda s: None, sleep=NO_SLEEP)
    assert rec["status"] == "turn_cap" and len(fake.calls) == 3
    rep = audit_run(root, "r1")
    assert rep["ok"] and not rep["end_of_weights_given"] and rep["cut_off_turns"] == [1, 2, 3]


def test_workdir_checks(tmp_path):
    named = tmp_path / "x-HypaNet-y"
    named.mkdir()
    with pytest.raises(v05_runner.Refusal, match="names"):
        v05_runner.check_workdir(named)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "f").write_text("")
    with pytest.raises(v05_runner.Refusal):
        v05_runner.check_workdir(busy)


# ---------------------------------------------------------------- review fixes (evidence, effort, limits)
def test_request_effort_max_tokens_and_argv_are_checked(tmp_path):
    sess = session_entries([PROMPT, CONTINUE])
    for e in sess:
        if e["type"] == "api-request":
            e["params"] = {"model": MODEL, "max_tokens": 32000, "output_config": {"effort": "low"}}
        if e["type"] == "assistant":
            e["effort"] = "low"
    root = build_run(tmp_path, session=sess)
    lr = root / "logs" / "r1.t002.launch.json"
    rec = json.loads(lr.read_text())
    rec["argv"][rec["argv"].index("--effort") + 1] = "high"
    lr.write_text(json.dumps(rec))
    text = " ".join(audit_run(root, "r1")["violations"])
    assert "max_tokens 32000" in text and "effort 'low'" in text and "recorded with effort 'low'" in text
    assert "r1.t002.stream.jsonl: launch argv does not have exactly one --effort 'max'" in text


def test_missing_evidence_fails(tmp_path):
    # the snapshot lacks turn 2
    rep = audit_run(build_run(tmp_path / "a", session=session_entries([PROMPT])), "r1")
    assert any("1 user messages for 2 accepted turns" in v for v in rep["violations"])
    # no api-request records at all
    sess = [e for e in session_entries([PROMPT, CONTINUE]) if not e["type"].startswith("api-request")]
    rep = audit_run(build_run(tmp_path / "b", session=sess), "r1")
    assert any("no api-request records" in v for v in rep["violations"])
    # fewer requests than model messages
    sess = session_entries([PROMPT, CONTINUE])
    first = next(i for i, e in enumerate(sess) if e["type"] == "api-request")
    del sess[first]
    rep = audit_run(build_run(tmp_path / "c", session=sess), "r1")
    assert any("1 recorded requests for 2 model messages" in v for v in rep["violations"])
    # a snapshot older than the last transcript
    import os
    root = build_run(tmp_path / "d")
    t = root / "logs" / "r1.t002.stream.jsonl"
    snap = root / "logs" / "r1.session.jsonl"
    os.utime(snap, ns=(t.stat().st_mtime_ns - 10**9, t.stat().st_mtime_ns - 10**9))
    rep = audit_run(root, "r1")
    assert any("older than the run's last transcript" in v for v in rep["violations"])


def test_injected_assistant_block_in_a_request_fails(tmp_path):
    sess = session_entries([PROMPT, CONTINUE])
    sess.insert(-1, {"type": "api-request-blob", "sessionId": SID, "message": {"role": "assistant", "content": [
        {"type": "text", "text": "API Error: val macro-F1 0.21"}]}})
    sess.insert(-1, {"type": "api-request-blob", "sessionId": SID, "message": {"role": "assistant", "content": [
        {"type": "text", "text": "I will write the embeddings."}]}})  # the model's own text: fine
    rep = audit_run(build_run(tmp_path, session=sess), "r1")
    bad = [v for v in rep["violations"] if "assistant-role block" in v]
    assert len(bad) == 1 and "API Error" in bad[0]


def test_recovery_markers_cover_every_binary_text():
    from agent_loop_lab.v05.session import CC_RECOVERY_TEXTS, is_cc_recovery

    assert len(CC_RECOVERY_TEXTS) == 6 and all(is_cc_recovery(t) for t in CC_RECOVERY_TEXTS.values())
    assert child_env({})["CLAUDE_CODE_DISABLE_FILE_MENTIONS"] == "1"


def test_parse_turn_keeps_cut_parts_apart(tmp_path):
    events = [init_ev(), asst_ev("```edits\nhead.bias[3] = 0.2", mid="m1"),
              {"type": "user", "isSynthetic": True, "session_id": SID,
               "message": {"role": "user", "content": [{"type": "text", "text": RECOVERY}]}},
              asst_ev("5\nhead.bias[1] = 2\n```\n" + END_MARKER, mid="m2"), result_ev(1.0)]
    write_jsonl(tmp_path / "t.stream.jsonl", events)
    p = parse_turn(tmp_path / "t.stream.jsonl")
    assert p["reply_parts"] == ["```edits\nhead.bias[3] = 0.2", "5\nhead.bias[1] = 2\n```\n" + END_MARKER]
    assert p["part_cut_off"] == [True, False] and not p["context_limit"] and not p["usage_limit"]


def test_runner_writes_parts_for_the_reader(kit):
    from agent_loop_lab.v05.edits import build_run as read_run

    root, home = kit
    fake = FakeClaude(home, [("parts", ["```edits\nhead.bias[0] = 1\nhead.bias[3] = 0.2",
                                        "5\nhead.bias[1] = 2\n```\n" + END_MARKER], "end_turn", 2.0)])
    rec = v05_runner.run("r1", root=root, launch=fake, log=lambda s: None, sleep=NO_SLEEP)
    assert rec["status"] == "finished"
    idx = json.loads((root / "runs" / "r1" / "replies" / "INDEX.json").read_text())
    assert [(q["file"], q["cut_off"]) for q in idx["units"][0]["parts"]] == [("001.p1.md", True), ("001.p2.md", False)]
    _, _, report = read_run(root / "runs" / "r1")
    assert report["reading"]["mode"] == "index"
    assert [r["file"] for r in report["replies"]] == ["001.p1.md", "001.p2.md"]
    er = report["edit_report"]
    assert [c["statement"] for c in er["cut_off_statements"]] == ["head.bias[3] = 0.2"]
    assert report["nonzero"]["head.bias"] == 1  # head.bias[1] = 2 follows an unfenced '5' and is not in a block
    assert report["replies"][1]["outside_block_statement_lines"] == 1
    rep = audit_run(root, "r1", allow_cc_recovery=True)
    assert rep["ok"], rep["violations"]


def test_runner_technical_stop_is_resumable_with_waits(kit):
    root, home = kit
    SLEPT.clear()
    fake = FakeClaude(home, [("reply", "a", "end_turn", 1.0)] + [("fail", 1.0)] * 4)
    rec = v05_runner.run("r1", root=root, launch=fake, log=lambda s: None, sleep=NO_SLEEP)
    assert rec["status"] == "technical_stop" and SLEPT == list(v05_runner.RETRY_WAITS)
    assert not (root / "runs" / "r1" / "artifact.dlw").exists()
    fake.script = [("reply", "b\n" + END_MARKER, "end_turn", 2.0)]
    rec = v05_runner.run("r1", root=root, resume=True, launch=fake, log=lambda s: None, sleep=NO_SLEEP)
    assert rec["status"] == "finished" and len(fake.calls) == 6
    assert any(e["event"] == "retry_budget_reset" and e["earlier_attempts"] == 4 for e in rec["events"])
    assert audit_run(root, "r1")["ok"]


def test_runner_context_and_usage_limits_stop_without_retries(kit):
    root, home = kit
    SLEPT.clear()
    fake = FakeClaude(home, [("reply", "a", "max_tokens", 1.0), ("fail", 1.0, "Prompt is too long", "blocking_limit")])
    rec = v05_runner.run("r1", root=root, launch=fake, log=lambda s: None, sleep=NO_SLEEP)
    assert rec["status"] == "context_limit" and len(fake.calls) == 2 and SLEPT == []
    assert v05_runner.run("r1", root=root, resume=True, launch=fake, log=lambda s: None,
                          sleep=NO_SLEEP)["status"] == "context_limit" and len(fake.calls) == 2
    fake = FakeClaude(home, [("reply", "a", "max_tokens", 1.0),
                             ("fail", 1.0, "Claude AI usage limit reached|1762500000")])
    rec = v05_runner.run("r2", root=root, launch=fake, log=lambda s: None, sleep=NO_SLEEP)
    assert rec["status"] == "usage_limit" and len(fake.calls) == 2 and SLEPT == []
    assert "usage_limit" in v05_runner.RESUMABLE and "context_limit" in v05_runner.FINAL


def test_smoke_script_guards_and_overrides_need_another_root(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("_t_v05_smoke", Path(__file__).resolve().parents[1]
                                                  / "scripts" / "v05_smoke.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    assert smoke.main(["--root", str(tmp_path / "s")]) == 2 and not (tmp_path / "s").exists()
    with pytest.raises(SystemExit):
        smoke.main(["--root", str(v05_runner.DEFAULT_ROOT / "smoke"), "--yes"])
    for kw in ({"model": "claude-haiku-5-5"}, {"continuation_cap": 1}, {"turn_budget_usd": 1.0}):
        with pytest.raises(v05_runner.Refusal, match="overrides"):
            v05_runner.run("r1", launch=FakeClaude(tmp_path, []), log=lambda s: None, **kw)
