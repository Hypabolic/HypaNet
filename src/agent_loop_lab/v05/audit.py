"""Audit of one v0.5 run (scripts/v05_audit_session.py): what the author saw, what it did, what it cost.

Inputs (ROOT = experiments/v0_5): ``ROOT/runs/<run>/run.json``, every turn transcript of the run
(``logs/<run>.tNNN.stream.jsonl``) and every failed attempt (``logs/<run>.tNNN.fail<K>.stream.jsonl``,
``.noinit<K>.stream.log``), their ``.launch.json`` records, ``logs/<run>.session.jsonl`` (the snapshot of
Claude Code's session file, which records every API request: system blocks, tools and each message sent) and
``ROOT/PROMPT.txt``.

Violations (exit 1):

* an init event that lists a tool, an MCP server, a slash command or a skill, a plugin that is not built in
  (``<name>@builtin``), another model, or another working directory; no init event, or more than one;
* any tool_use / server_tool_use / mcp_tool_use block or tool_result anywhere;
* a turn with more than one API message or a user message in the stream (Claude Code continuing a cut reply
  on its own with its recovery text) - reported as ``cc_recovery`` and downgraded to a deviation with
  ``allow_cc_recovery``; anything else injected is always a violation;
* a launch record whose message hash is not PROMPT.txt (turn 001) or ``Continue.`` (later turns), a missing
  launch record, turns not numbered 1..N, accepted turns in different sessions or not in the run's session;
* an assistant message from another model;
* END OF WEIGHTS before the last turn, or a ``finished`` run whose last turn lacks it;
* in the session file: a first user message that is not exactly PROMPT.txt (a match up to leading and trailing
  whitespace, if Claude Code trims the message, is accepted and recorded), a later user message that is not
  exactly ``Continue.``, the prompt sent twice, a request whose system blocks are not the CLI line plus
  "Answer the user's request." or that offers tools, a request for another model, an attachment type outside
  :data:`~agent_loop_lab.v05.session.CC_CONTEXT_ATTACHMENTS` that is rendered to the model, a
  ``session_context`` with keys other than ``userEmail``, a user-role block that is not the prompt,
  ``Continue.``, a recognised Claude Code context reminder or a recovery text, and any context text that
  carries CLAUDE.md, memory, skills or git status (:data:`FORBIDDEN_CONTEXT_MARKERS`);
* a request whose ``max_tokens`` is not :data:`~agent_loop_lab.v05.session.MAX_OUTPUT_TOKENS` or whose
  ``output_config.effort`` is not ``max``, or a model message recorded with another effort; a launch record
  whose argv lacks ``--model MODEL``, ``--effort max``, ``--tools ""`` or ``--system-prompt "Answer the
  user's request."``;
* an assistant-role text block in a recorded request that is not text the model wrote in this run (a synthetic
  "API Error" message, or anything else, put back into the conversation as if the model had said it);
* missing evidence: no session snapshot, no ``api-request`` records in it, fewer recorded requests than model
  messages, fewer user messages in it than accepted turns (or more than turns plus failed attempts that
  started a session), or a snapshot older than the run's last transcript.

Recorded (not violations): the init event's fields (tools, mcp_servers, slash_commands, skills, plugins,
agents, cwd, model, apiKeySource), Claude Code's own context (system prompt blocks, environment, model
identity, date and e-mail attachments, exact rendered texts), stop reasons per message (``max_tokens`` = cut
off), whether END OF WEIGHTS was given, retries, usage summed per turn and in total, and cost (session-id
delta, agent_loop_lab.v05.cost).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_loop_lab.v05 import cost as v05_cost
from agent_loop_lab.v05.session import (
    CC_CONTEXT_ATTACHMENTS, CLI_PREFIXES, CONTINUE, EFFORT, END_MARKER, MAX_OUTPUT_TOKENS, MODEL,
    SESSION_CONTEXT_KEYS, SYSTEM_PROMPT, is_cc_recovery, parse_turn, read_events, sha256_file, sha256_text,
)

TOOL_BLOCKS = ("tool_use", "server_tool_use", "mcp_tool_use", "tool_result")
FORBIDDEN_CONTEXT_MARKERS = (
    "CLAUDE.md", "CLAUDE.local.md", "claudeMd", "MEMORY.md", "auto memory", "auto-memory", "# Memory",
    "Contents of /", "skills are available", "Skill tool", "gitStatus", "git status", "Current branch:",
    "hypanet", "HypaNet", "doom-loop", "doom loop",
)
# The Claude Code context reminders seen in the smoke test (user role); anything else in a user-role block fails.
KNOWN_USER_REMINDER_PREFIXES = (
    "<system-reminder>\nAs you answer the user's questions, you can use the following context:\n# userEmail\n",
)
USAGE_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def _usage_of(result: dict | None) -> dict:
    u = (result or {}).get("usage") or {}
    out = {k: int(u.get(k) or 0) for k in USAGE_KEYS}
    out["thinking_tokens"] = int(((u.get("output_tokens_details") or {}).get("thinking_tokens")) or 0)
    return out


def _add(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def _blocks(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _markers(text: str) -> list[str]:
    return [m for m in FORBIDDEN_CONTEXT_MARKERS if m in text]


def audit_init(init: dict | None, *, model: str, workdir: str | None, where: str) -> tuple[dict, list[str]]:
    v: list[str] = []
    if init is None:
        return {}, [f"{where}: no init event"]
    rec = {k: init.get(k) for k in ("tools", "mcp_servers", "slash_commands", "skills", "plugins", "agents", "cwd",
                                    "model", "permissionMode", "apiKeySource", "claude_code_version",
                                    "output_style", "additional_directories")}
    for k in ("tools", "mcp_servers", "slash_commands", "skills"):
        val = init.get(k)
        if val:  # absent or empty is fine
            v.append(f"{where}: init lists {k}: {val}")
    for p in init.get("plugins") or []:
        src = p.get("source", "") if isinstance(p, dict) else str(p)
        if not str(src).endswith("@builtin"):
            v.append(f"{where}: init lists a non-builtin plugin: {p}")
    if init.get("additional_directories"):
        v.append(f"{where}: init lists additional directories: {init['additional_directories']}")
    if init.get("model") != model:
        v.append(f"{where}: init model {init.get('model')!r} != {model!r}")
    if workdir is not None and init.get("cwd") != workdir:
        v.append(f"{where}: init cwd {init.get('cwd')!r} != run workdir {workdir!r}")
    return rec, v


def audit_stream(p: dict, *, model: str, where: str) -> tuple[list[str], list[str]]:
    """``(violations, cc_recovery)`` of one parsed transcript (tools, injected messages, models)."""
    v, rec = [], []
    for tu in p["tool_uses"]:
        v.append(f"{where}: {tu['type']} block ({tu['name']})")
    for ue in p["user_events"]:
        if "tool_result" in ue["blocks"]:
            continue
        if is_cc_recovery(ue["text"]):
            rec.append(f"{where}: Claude Code recovery message: {ue['text'][:200]!r}")
        else:
            v.append(f"{where}: a user message in the stream that the harness did not send: {ue['text'][:200]!r}")
    if p["assistant_messages"] > 1:
        rec.append(f"{where}: {p['assistant_messages']} API messages in one turn (Claude Code continued the reply)")
    for m in p["messages"]:
        if not m["synthetic"] and m["model"] != model:
            v.append(f"{where}: assistant message {m['id']} from model {m['model']!r}")
        if m.get("parent_tool_use_id"):
            v.append(f"{where}: assistant message {m['id']} from a subagent")
    if p["unreadable_lines"]:
        v.append(f"{where}: {p['unreadable_lines']} unreadable transcript lines")
    return v, rec


def audit_session_file(path: Path, *, prompt: str, model: str, session_id: str | None,
                       assistant_texts: list[str] | None = None) -> dict:
    """The session snapshot: user messages, attachments and every recorded API request.

    ``assistant_texts``: the texts of the model's own messages in the run's transcripts; when given, every
    assistant-role text block of a recorded request must be part of one of them."""
    out: dict[str, Any] = {"path": str(path), "violations": [], "cc_recovery": [], "user_messages": [],
                           "attachments": {}, "attachment_texts": [], "system_blocks": [], "system_turn_texts": [],
                           "user_reminders": [], "requests": 0, "request_models": [], "request_params": [],
                           "assistant_stop_reasons": [], "assistant_message_ids": 0, "other_attachments": [],
                           "request_log": False}
    v = out["violations"]
    if not path.is_file():
        v.append(f"no session snapshot at {path}: the user messages and the context cannot be checked")
        return out
    events, bad = read_events(path)
    if bad:
        out["unreadable_lines"] = bad
    seen_sys, seen_rem, seen_att = set(), set(), set()
    stop_by_id: dict[str, Any] = {}
    real_ids: set[str] = set()
    for ev in events:
        t = ev.get("type")
        if session_id and ev.get("sessionId") not in (None, session_id):
            v.append(f"session file entry from another session: {ev.get('sessionId')}")
        if ev.get("isSidechain"):
            v.append(f"session file has a sidechain (subagent) entry: {t}")
        if t == "user":
            msg = ev.get("message") or {}
            blocks = _blocks(msg.get("content"))
            if any(b.get("type") in TOOL_BLOCKS for b in blocks):
                v.append("session file: a tool_result user message")
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            if ev.get("isMeta"):
                if is_cc_recovery(text):
                    out["cc_recovery"].append(f"session file: Claude Code recovery message: {text[:200]!r}")
                else:
                    v.append(f"session file: a meta user message the harness did not send: {text[:200]!r}")
                continue
            trimmed = text != prompt and text.strip() == prompt.strip()
            out["user_messages"].append({"sha256": sha256_text(text), "chars": len(text),
                                         "is_prompt": text == prompt or trimmed, "prompt_whitespace_trimmed": trimmed,
                                         "is_continue": text == CONTINUE})
        elif t == "attachment":
            att = ev.get("attachment") or {}
            at = att.get("type")
            out["attachments"][at] = out["attachments"].get(at, 0) + 1
            rendered = [r.get("content", "") for r in ev.get("rendered") or [] if isinstance(r, dict)]
            for txt in rendered:
                if txt not in seen_att:
                    seen_att.add(txt)
                    out["attachment_texts"].append({"type": at, "role": ev.get("renderedRole"), "text": txt})
                    if _markers(txt):
                        v.append(f"session file: attachment {at} carries {_markers(txt)}")
            if at not in CC_CONTEXT_ATTACHMENTS:
                if rendered:
                    v.append(f"session file: attachment type {at!r} rendered to the model")
                else:
                    out["other_attachments"].append(at)
            if at == "session_context":
                keys = sorted((att.get("context") or {}).keys())
                extra = [k for k in keys if k not in SESSION_CONTEXT_KEYS]
                if extra:
                    v.append(f"session file: session_context carries {extra}")
        elif t == "assistant":
            msg = ev.get("message") or {}
            mid = msg.get("id")
            if msg.get("model") not in (model, "<synthetic>"):
                v.append(f"session file: assistant message from model {msg.get('model')!r}")
            if msg.get("model") != "<synthetic>":
                for k in ("effort", "perTurnEffort"):
                    if k in ev and ev[k] != EFFORT:
                        v.append(f"session file: assistant message {mid} recorded with {k} {ev[k]!r}")
            if any(b.get("type") in TOOL_BLOCKS for b in _blocks(msg.get("content"))):
                v.append("session file: a tool_use block")
            if mid and msg.get("model") != "<synthetic>":
                real_ids.add(mid)
            if mid and msg.get("stop_reason"):
                stop_by_id[mid] = msg.get("stop_reason")
            elif mid:
                stop_by_id.setdefault(mid, None)
        elif t == "api-request-shape":
            out["request_log"] = True
            shape = ev.get("shape") or {}
            sys_texts = [b.get("text", "") for b in _blocks(shape.get("system"))]
            for s in sys_texts:
                if s not in seen_sys:
                    seen_sys.add(s)
                    out["system_blocks"].append(s)
            if SYSTEM_PROMPT not in sys_texts or any(s not in CLI_PREFIXES + (SYSTEM_PROMPT,) for s in sys_texts):
                v.append(f"session file: system blocks {sys_texts!r} are not the CLI line plus the system prompt")
            if shape.get("tools"):
                v.append(f"session file: a request offered tools: {[x.get('name') for x in shape['tools']]}")
        elif t == "api-request":
            out["requests"] += 1
            params = ev.get("params") or {}
            out["request_models"].append(params.get("model"))
            out["request_params"].append({k: params.get(k) for k in ("max_tokens", "thinking", "output_config")})
            if params.get("model") != model:
                v.append(f"session file: a request for model {params.get('model')!r}")
            if params.get("max_tokens") != MAX_OUTPUT_TOKENS:
                v.append(f"session file: a request with max_tokens {params.get('max_tokens')!r} != {MAX_OUTPUT_TOKENS}")
            effort = (params.get("output_config") or {}).get("effort")
            if effort != EFFORT:
                v.append(f"session file: a request with effort {effort!r} != {EFFORT!r}")
            if params.get("tools"):
                v.append("session file: a request with tools")
        elif t == "api-request-blob":
            msg = ev.get("message") or {}
            role = msg.get("role")
            for b in _blocks(msg.get("content")):
                bt = b.get("type")
                if bt in TOOL_BLOCKS:
                    v.append(f"session file: a {bt} block in a request")
                    continue
                if bt != "text":
                    continue
                txt = b.get("text", "")
                if role == "assistant":
                    if txt.startswith("API Error") or (
                            assistant_texts is not None and txt.strip() and not any(txt in a for a in assistant_texts)):
                        v.append(f"session file: an assistant-role block in a request that the model did not write "
                                 f"in this run: {txt[:200]!r}")
                    continue
                if role == "system":
                    if txt not in seen_sys:
                        seen_sys.add(txt)
                        out["system_turn_texts"].append(txt)
                    if _markers(txt):
                        v.append(f"session file: a system turn carries {_markers(txt)}")
                elif role == "user":
                    if txt in (prompt, prompt.strip(), CONTINUE):
                        continue
                    if is_cc_recovery(txt):
                        out["cc_recovery"].append(f"request: Claude Code recovery text: {txt[:200]!r}")
                    elif txt.startswith(KNOWN_USER_REMINDER_PREFIXES):
                        if txt not in seen_rem:
                            seen_rem.add(txt)
                            out["user_reminders"].append(txt)
                        if _markers(txt):
                            v.append(f"session file: a user-role reminder carries {_markers(txt)}")
                    else:
                        v.append(f"session file: a user-role block that is neither the prompt nor Continue.: "
                                 f"{txt[:200]!r}")
    out["assistant_stop_reasons"] = list(stop_by_id.values())
    out["assistant_message_ids"] = len(real_ids)
    um = out["user_messages"]
    if not um:
        v.append("session file: no user message")
    else:
        if not um[0]["is_prompt"]:
            v.append("session file: the first user message is not exactly PROMPT.txt")
        if sum(1 for u in um if u["is_prompt"]) > 1:
            v.append("session file: PROMPT.txt was sent more than once")
        for i, u in enumerate(um[1:], 2):
            if not u["is_continue"]:
                v.append(f"session file: user message {i} is not exactly {CONTINUE!r} (sha256 {u['sha256']})")
    if not out["request_log"] or not out["requests"]:
        v.append("session file: no api-request records; the system blocks and request contents cannot be checked")
    elif out["requests"] < len(real_ids):
        v.append(f"session file: {out['requests']} recorded requests for {len(real_ids)} model messages")
    return out


def audit_launch_argv(argv: Any, *, model: str, where: str) -> list[str]:
    """The flags that fix the author's model, effort, tools and system prompt, from a launch record's argv."""
    if not isinstance(argv, list):
        return [f"{where}: launch record has no argv"]
    want = (("--model", model), ("--effort", EFFORT), ("--tools", ""), ("--system-prompt", SYSTEM_PROMPT))
    v = []
    for flag, val in want:
        idx = [i for i, a in enumerate(argv) if a == flag]
        if len(idx) != 1 or idx[0] + 1 >= len(argv) or argv[idx[0] + 1] != val:
            v.append(f"{where}: launch argv does not have exactly one {flag} {val!r}")
    if "--fallback-model" in argv:
        v.append(f"{where}: launch argv has --fallback-model")
    return v


def audit_run(root: Path, run: str, *, allow_cc_recovery: bool = False) -> dict:
    root = Path(root)
    logs = root / "logs"
    rec_path = root / "runs" / run / "run.json"
    violations: list[str] = []
    rec = json.loads(rec_path.read_text()) if rec_path.is_file() else None
    if rec is None:
        violations.append(f"no run record {rec_path}")
    prompt_path = root / "PROMPT.txt"
    prompt = prompt_path.read_bytes().decode("utf-8") if prompt_path.is_file() else ""
    prompt_sha = sha256_file(prompt_path) if prompt_path.is_file() else None
    if rec is not None and prompt_sha != rec.get("prompt_sha256"):
        violations.append(f"PROMPT.txt sha256 {prompt_sha} != the run's {rec.get('prompt_sha256')}")
    model = (rec or {}).get("model") or MODEL
    workdir = (rec or {}).get("workdir")
    cont_sha = sha256_text(CONTINUE)

    accepted = sorted(p for p in logs.glob(f"{run}.t*.stream.jsonl") if p.name.count(".") == 3)
    failed = sorted(set(logs.glob(f"{run}.t*.fail*.stream.jsonl")) | set(logs.glob(f"{run}.t*.noinit*.stream.log")))
    numbers = [int(p.name.split(".")[1][1:]) for p in accepted]
    if numbers != list(range(1, len(numbers) + 1)):
        violations.append(f"turns are not numbered 1..N: {numbers}")
    turns, inits, cc_recovery = [], [], []
    assistant_texts: list[str] = []
    usage_total: dict = {}
    sids: set[str] = set()
    for n, path in zip(numbers, accepted):
        where = path.name
        p = parse_turn(path)
        init_rec, v = audit_init(p["init"], model=model, workdir=workdir, where=where)
        violations += v
        if p["init_events"] > 1:
            violations.append(f"{where}: {p['init_events']} init events")
        inits.append({"turn": n, **init_rec})
        v, r = audit_stream(p, model=model, where=where)
        violations += v
        cc_recovery += r
        sids.update(p["session_ids"])
        assistant_texts += [m["text"] for m in p["messages"] if not m["synthetic"] and m["text"]]
        if len(p["session_ids"]) != 1:
            violations.append(f"{where}: session ids {p['session_ids']}")
        lr_path = path.with_name(path.name.replace(".stream.jsonl", ".launch.json"))
        lr = json.loads(lr_path.read_text()) if lr_path.is_file() else None
        want = prompt_sha if n == 1 else cont_sha
        if lr is None:
            violations.append(f"{where}: no launch record")
        else:
            if lr.get("message_sha256") != want:
                violations.append(f"{where}: launch message sha256 {lr.get('message_sha256')} != "
                                  f"{'PROMPT.txt' if n == 1 else 'Continue.'} ({want})")
            if bool(lr.get("resume")) != (n > 1):
                violations.append(f"{where}: resume={lr.get('resume')} for turn {n}")
            violations += audit_launch_argv(lr.get("argv"), model=model, where=where)
        if p["end_marker"] and n != numbers[-1]:
            violations.append(f"{where}: END OF WEIGHTS before the last turn")
        u = _usage_of(p["result"])
        usage_total = _add(usage_total, u)
        res = p["result"] or {}
        turns.append({"turn": n, "transcript": path.name, "session_ids": p["session_ids"],
                      "assistant_messages": p["assistant_messages"], "stop_reason": p["stop_reason"],
                      "stop_reasons": p["stop_reasons"], "cut_off": p["stop_reason"] == "max_tokens",
                      "result_subtype": res.get("subtype"), "is_error": res.get("is_error"),
                      "has_result_event": p["result"] is not None, "models": p["models"],
                      "model_usage_models": sorted((res.get("modelUsage") or {}).keys()),
                      "reply_chars": len(p["reply_text"]), "end_marker": p["end_marker"],
                      "end_marker_exact": p["end_marker_exact"], "usage": u,
                      "synthetic_messages": p["synthetic_messages"],
                      "message_sha256": lr.get("message_sha256") if lr else None})
        extra_models = [m for m in (res.get("modelUsage") or {}) if m != model]
        if extra_models:
            turns[-1]["flag"] = f"modelUsage lists other models {extra_models} (auxiliary calls?)"
    if rec is not None and sids and sids != {rec.get("session_id")}:
        violations.append(f"accepted turns use session ids {sorted(sids)}, the run's is {rec.get('session_id')}")
    attempts = []
    for path in failed:
        where = path.name
        p = parse_turn(path)
        _, v = audit_init(p["init"], model=model, workdir=workdir, where=where) if p["init"] else ({}, [])
        violations += v
        v, r = audit_stream(p, model=model, where=where)
        violations += v
        cc_recovery += r
        stem = path.name.rsplit(".stream.", 1)[0]
        lr_path = logs / f"{stem}.launch.json"
        lr = json.loads(lr_path.read_text()) if lr_path.is_file() else None
        n = int(path.name.split(".")[1][1:])
        want = prompt_sha if n == 1 else cont_sha
        if lr is not None and lr.get("message_sha256") != want:
            violations.append(f"{where}: retry with a different message (sha256 {lr.get('message_sha256')})")
        if lr is not None and "argv" in lr:
            violations += audit_launch_argv(lr.get("argv"), model=model, where=where)
        assistant_texts += [m["text"] for m in p["messages"] if not m["synthetic"] and m["text"]]
        u = _usage_of(p["result"])
        usage_total = _add(usage_total, u)
        attempts.append({"attempt": stem, "turn": n, "init": p["init"] is not None, "session_ids": p["session_ids"],
                         "result": p["result"], "synthetic_messages": p["synthetic_messages"], "usage": u})
    if not accepted:
        violations.append("no accepted turn transcript")
    finished = bool(turns) and turns[-1]["end_marker"]
    if rec is not None and rec.get("status") == "finished" and not finished:
        violations.append("run.json says finished but the last turn does not end with END OF WEIGHTS")
    snap = logs / f"{run}.session.jsonl"
    sess = audit_session_file(snap, prompt=prompt, model=model, session_id=(rec or {}).get("session_id"),
                              assistant_texts=assistant_texts)
    violations += sess.pop("violations")
    if snap.is_file():
        # a turn > 1 retried after a failure that reached the API sends Continue. again in the same session
        same_session_failures = sum(1 for a in attempts if a["init"] and a["turn"] > 1)
        n_user = len(sess["user_messages"])
        if not len(turns) <= n_user <= len(turns) + same_session_failures:
            violations.append(f"session file: {n_user} user messages for {len(turns)} accepted turns and "
                              f"{same_session_failures} failed attempts after turn 001 (a stale or partial snapshot?)")
        newest = max([p.stat().st_mtime_ns for p in accepted + failed], default=0)
        if snap.stat().st_mtime_ns < newest:
            violations.append("session file: the snapshot is older than the run's last transcript")
    cc_recovery += sess.pop("cc_recovery")
    deviations: list[str] = []
    if cc_recovery:
        if allow_cc_recovery:
            deviations += cc_recovery
        else:
            violations += cc_recovery
    costs = v05_cost.summary(logs)
    run_launches = {k: v for k, v in costs["launches"].items() if k == run or k.startswith(run + ".")}
    report = {
        "run": run,
        "status": (rec or {}).get("status"),
        "model": model,
        "session_id": (rec or {}).get("session_id"),
        "abandoned_session_ids": (rec or {}).get("abandoned_session_ids", []),
        "workdir": workdir,
        "prompt_sha256": prompt_sha,
        "continue_sha256": cont_sha,
        "turns_accepted": len(turns),
        "continuation_turns": max(0, len(turns) - 1),
        "end_of_weights_given": finished,
        "end_marker": END_MARKER,
        "cut_off_turns": [t["turn"] for t in turns if t["cut_off"]],
        "turns": turns,
        "failed_attempts": attempts,
        "init": inits,
        "session_file": sess,
        "usage_total": usage_total,
        "cost": {"run_usd": round(sum(v["cost_usd"] for v in run_launches.values()), 6), "launches": run_launches,
                 "all_runs": costs["cap"], "naive_sum_all_runs_usd": costs["naive_sum_usd"]},
        "cc_recovery_allowed": allow_cc_recovery,
        "deviations": deviations,
        "violations": violations,
        "ok": not violations,
    }
    return report
