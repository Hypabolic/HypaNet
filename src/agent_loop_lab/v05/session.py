"""The v0.5 author session: the exact ``claude -p`` command, its environment, and how a turn transcript is read.

v0.5 replicates the premise of "Experiment 5" (one prompt, no tools, no feedback, every weight starts at 0) with
one deliberate deviation: a run is a conversation. Turn 001 sends experiments/v0_5/PROMPT.txt in a new session;
while the last reply does not end with the line ``END OF WEIGHTS``, the harness resumes the same session with
the fixed message :data:`CONTINUE` and nothing else (no feedback, no error text, no results).

Context. The author's context must hold only :data:`SYSTEM_PROMPT` and the user messages. The command
(:func:`claude_argv`) therefore uses ``--tools ""`` (no built-in tool), ``--system-prompt`` (replaces Claude
Code's default system prompt, and with it the environment, cwd, git and memory sections),
``--strict-mcp-config`` without ``--mcp-config`` (no MCP server), ``--safe-mode`` (no CLAUDE.md, skills,
plugins, hooks, MCP servers, custom commands or agents, output styles), ``--disable-slash-commands``,
``--setting-sources project`` in a fresh empty directory (so no user or local settings: no user hooks,
plugins or permissions) plus a ``--settings`` JSON that turns auto-memory and hooks off, and
``--permission-mode dontAsk --permission-prompts none``. ``--bare`` is not used: it reads only
ANTHROPIC_API_KEY, and this machine authenticates with OAuth. The environment (:func:`child_env`) removes
every variable that links the session to a parent Claude Code session and sets
CLAUDE_CODE_DISABLE_AUTO_MEMORY, CLAUDE_CODE_DISABLE_CLAUDE_MDS, CLAUDE_CODE_DISABLE_ATTACHMENTS (no
system-reminder attachments), CLAUDE_CODE_DISABLE_FILE_MENTIONS (the prompt has lines starting with
``@dataclass``; no @-mention lookup), CLAUDE_CODE_NO_MODEL_FALLBACK (no silent model substitution; there is no
``--fallback-model`` either), CLAUDE_CODE_MAX_EFFORT_REMINDER=0, DISABLE_AUTO_COMPACT / DISABLE_COMPACT (a
compaction summary would be a message the harness did not send) and CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000.

What Claude Code still adds on its own (smoke test with claude-haiku-5-5 and these exact flags, Claude Code
2.1.311, 6 Nov 2026; the session file records every API request: ``api-request-shape`` = system and tools,
``api-request-blob`` = each message sent):

* the system prompt is two blocks, a fixed CLI line (:data:`CLI_PREFIXES`; for ``-p`` without
  ``--append-system-prompt`` it is "You are a Claude agent, built on Anthropic's Claude Agent SDK.") and then
  :data:`SYSTEM_PROMPT`; ``tools`` is empty;
* attachments (:data:`CC_CONTEXT_ATTACHMENTS`) that CLAUDE_CODE_DISABLE_ATTACHMENTS does not remove: the
  environment (working directory, "not a git repository", platform, shell, OS version and a fixed note about
  downloaded files), the model identity line (name, id, knowledge cutoff) and today's date, sent as one
  system-role turn, and the account's e-mail address as a user-role system-reminder (``session_context``).
  Claude Code skips these only in ``--bare`` mode, which needs ANTHROPIC_API_KEY / an OAuth token in the
  environment (``claude setup-token``): set V05_BARE=1 to use it (untested here). No CLAUDE.md, memory,
  skills, slash commands, hooks, MCP servers or tools got in.

When a reply hits the output limit (stop_reason ``max_tokens``), Claude Code 2.1.311 continues it inside the
same turn, up to 3 times per turn, with its own fixed recovery message (an ``isMeta`` user message; the texts are
:data:`CC_RECOVERY_TEXTS`, the one used by default is "Output token limit hit. Resume directly ... Pick up
mid-thought ..."). There is no setting that turns this off. It is a message the harness did not send; the audit
reports it (agent_loop_lab.v05.audit) and PROMPT_NOTES.md records it (D13). :func:`parse_turn` keeps each API
message of a turn apart (``reply_parts``) with whether it was cut off (``part_cut_off``: followed by a recovery
message, or the turn's last message with stop_reason ``max_tokens``), so the edits reader can apply the cut-off
rule at every cut. The session file (``~/.claude/projects/<cwd slug>/<session>.jsonl``) is snapshotted into the
logs after every turn.

The context window is the real limit on how much the author can write: compaction is off and every thinking
block is kept (``clear_thinking`` ``keep: all``), so the conversation (prompt, every reply with its thinking)
must fit in Fable's window (1,000,000 tokens in Claude Code 2.1.311's model table). When it no longer does,
Claude Code refuses the next request without calling the API (``terminal_reason`` ``blocking_limit`` or
``prompt_too_long``, a synthetic "Prompt is too long" message); :func:`parse_turn` reports it as
``context_limit`` and the runner stops with that status.

A turn transcript (``RUN.tNNN.stream.jsonl``, ``--output-format stream-json --verbose``) is read with
:func:`parse_turn`: init event, assistant messages (grouped by message id, synthetic API-error messages kept
apart), tool_use blocks, the result event, stop reasons, usage, and the reply text.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

RUNS = ("r1", "r2", "r3")
MODEL = "claude-fable-5-1"
EFFORT = "max"
SYSTEM_PROMPT = "Answer the user's request."
CONTINUE = "Continue."
END_MARKER = "END OF WEIGHTS"
MAX_OUTPUT_TOKENS = 128000
CONTINUATION_CAP = 40  # continuation turns per run (turns 002..041); a technical safety cap, recorded
RETRY_CAP = 3  # technical retries per turn (same message, recorded)
TURN_BUDGET_USD = 60.0  # --max-budget-usd per turn (below the remaining cap); a killed turn is charged this
SETTINGS = {"autoMemoryEnabled": False, "disableAllHooks": True, "includeCoAuthoredBy": False}
SYNTHETIC_MODEL = "<synthetic>"
CLI_PREFIXES = (
    "You are a Claude agent, built on Anthropic's Claude Agent SDK.",
    "You are Claude Code, Anthropic's official CLI for Claude.",
    "You are Claude Code, Anthropic's official CLI for Claude, running within the Claude Agent SDK.",
)
# Attachment types Claude Code adds in -p mode that cannot be switched off without --bare (recorded, see above);
# credential_org and prompt_snapshot are bookkeeping and are not rendered to the model.
CC_CONTEXT_ATTACHMENTS = ("environment", "model", "date", "session_context", "credential_org", "prompt_snapshot")
SESSION_CONTEXT_KEYS = ("userEmail",)
# Claude Code 2.1.311's own continuation texts after an output-limit cut (fragments that identify them).
CC_RECOVERY_MARKERS = (
    "Output token limit hit",
    "Your response was cut off because it exceeded the output token limit",
    "response was truncated",
    "reached the output limit",
    "cut off mid-stream",
)
# The exact texts in the 2.1.311 binary (its list M = [iAt, aAt, Ykn, UZo, K3t, q3t]); in -p mode the nudge is iAt,
# or Ykn when the server flag tengu_tingly_pike selects "no_visible_output" after a thinking-only cut (aAt is gated
# off in this build). Recorded for PROMPT_NOTES.md D13.
CC_RECOVERY_TEXTS = {
    "resume (iAt, default)": "Output token limit hit. Resume directly \u2014 no apology, no recap of what you were "
                             "doing. Pick up mid-thought if that is where the cut happened. Break remaining work into "
                             "smaller pieces.",
    "thinking discarded (aAt)": "Output token limit hit while you were still reasoning, before this response produced "
                                "any text or tool call. That reasoning was not kept and is not part of this "
                                "conversation, so there is nothing to resume. Continue from the conversation above: "
                                "reason much more briefly and produce your answer or next tool call well within the "
                                "output limit.",
    "no visible output (Ykn)": "[Your previous response reached the output limit with no visible output. Please "
                               "continue and produce a brief user-visible response or tool call right away.]",
    "cut off, rewrite the line (UZo)": "Your response was cut off because it exceeded the output token limit. Continue "
                                       "from where you left off. What you write next is shown as a separate block "
                                       "below the text so far, so begin at the start of the line, table row, list "
                                       "item or sentence that was cut, writing it again in full, and if you were "
                                       "inside a code block or a table, open it again first (the same code fence and "
                                       "language, the table's header row). Repeat nothing before that point and do "
                                       "not comment on the interruption. Keep going until the response is complete.",
    "mid-stream resume (K3t)": "Your response above was cut off mid-stream. Resume directly from where it stops "
                               "\u2014 no apology, no recap. "
                               "If none of it survived, answer the request from the start.",
    "mid-stream restart (q3t)": "Your response above was cut off mid-stream and only your next message is delivered. "
                                "Write the complete response again from the start \u2014 no apology, no mention of "
                                "the cut-off.",
}
CC_RECOVERY_LIMIT = 3  # in-turn recoveries per turn (the binary's un=3)
CONTEXT_WINDOW_TOKENS = 1_000_000  # claude-fable-5-1 in Claude Code 2.1.311's model table
CONTEXT_LIMIT_REASONS = ("blocking_limit", "prompt_too_long")
CONTEXT_LIMIT_TEXT = "prompt is too long"
# A failure of this kind is not retried at once: the run stops (resumable) until the window resets.
USAGE_LIMIT_TEXTS = ("usage limit", "rate limit", "rate_limit", "limit reached", "limit will reset", "api error: 429")
# Variables that would link the child to a parent Claude Code session or change Python's behaviour.
SCRUB_ENV = (
    "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_CODE_BRIDGE_SESSION_ID", "CLAUDE_CODE_SESSION_ID", "CLAUDE_PID",
    "CLAUDE_PROJECT_DIR", "CLAUDE_EFFORT", "CLAUDE_CODE_SESSION_ATTENDED", "CLAUDE_CODE_EXECPATH",
    "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_SIMPLE", "CLAUDE_CODE_SAFE_MODE", "CLAUDE_CODE_EFFORT_LEVEL",
    "CLAUDE_CODE_SUBAGENT_MODEL", "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL", "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV",
)
SET_ENV = {
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(MAX_OUTPUT_TOKENS),
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_ATTACHMENTS": "1",
    "CLAUDE_CODE_DISABLE_FILE_MENTIONS": "1",  # the prompt has '@dataclass' lines; no @-mention lookup
    "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",
    "CLAUDE_CODE_MAX_EFFORT_REMINDER": "0",
    "DISABLE_AUTO_COMPACT": "1",
    "DISABLE_COMPACT": "1",
}


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def turn_name(run: str, n: int) -> str:
    """``r1.t001``: the launch name of turn n (the transcript is ``<name>.stream.jsonl``)."""
    return f"{run}.t{n:03d}"


def claude_argv(*, session_id: str, resume: bool, message: str, max_budget_usd: float, model: str = MODEL,
                effort: str = EFFORT, bare: bool = False) -> list[str]:
    """The exact command of one turn. The message is the last argument, after ``--`` (no shell, no stdin)."""
    session = ["--resume", session_id] if resume else ["--session-id", session_id]
    return [
        "claude", "-p",
        *(["--bare"] if bare else []),
        "--model", model, "--effort", effort,
        *session,
        "--tools", "",
        "--system-prompt", SYSTEM_PROMPT,
        "--strict-mcp-config",
        "--safe-mode",
        "--disable-slash-commands",
        "--setting-sources", "project",
        "--settings", json.dumps(SETTINGS, sort_keys=True),
        "--permission-mode", "dontAsk", "--permission-prompts", "none",
        "--max-budget-usd", f"{max_budget_usd:.2f}",
        "--output-format", "stream-json", "--verbose",
        "--", message,
    ]


def printable_argv(argv: list[str]) -> list[str]:
    """argv with the message replaced by its length and sha256 (for records)."""
    *head, msg = argv
    return head + [f"<message: {len(msg)} chars, sha256 {sha256_text(msg)}>"]


def child_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for k in SCRUB_ENV:
        env.pop(k, None)
    env.update(SET_ENV)
    return env


def project_slug(cwd: str | Path) -> str:
    """Claude Code's project folder name for a working directory (non-alphanumerics become '-')."""
    return re.sub(r"[^A-Za-z0-9]", "-", str(cwd))


def find_session_file(session_id: str, cwd: str | Path | None = None, home: Path | None = None) -> Path | None:
    """The session's JSONL under ~/.claude/projects (the cwd's folder first, then any project folder)."""
    root = (home or Path.home()) / ".claude" / "projects"
    if cwd is not None:
        p = root / project_slug(cwd) / f"{session_id}.jsonl"
        if p.is_file():
            return p
    hits = sorted(root.glob(f"*/{session_id}.jsonl")) if root.is_dir() else []
    return hits[0] if hits else None


def read_events(log: Path) -> tuple[list[dict], int]:
    """``(events, unreadable lines)`` of a stream-json transcript."""
    out, bad = [], 0
    for line in Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(ev, dict):
            out.append(ev)
        else:
            bad += 1
    return out, bad


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def is_cc_recovery(text: str) -> bool:
    return any(m in text for m in CC_RECOVERY_MARKERS)


def ends_with_marker(text: str) -> tuple[bool, bool]:
    """``(finished, exact)``: the reply's last non-blank line is END OF WEIGHTS (``exact``: with no surrounding
    whitespace on that line and nothing but newlines after it)."""
    lines = text.rstrip().splitlines()
    if not lines:
        return False, False
    last = lines[-1]
    return last.strip() == END_MARKER, last == END_MARKER and text.rstrip("\n").endswith(END_MARKER)


def parse_turn(log: Path) -> dict:
    """Everything the runner and the audit need from one turn transcript.

    ``messages``: the API messages in order, each ``{"id", "model", "synthetic", "text", "blocks",
    "stop_reason", "usage"}`` (stream-json emits one event per content block, with the message id repeated).
    ``reply_text``: the text of the non-synthetic messages, joined by a newline between messages (a view for
    people; the edits reader reads ``reply_parts`` separately when there is more than one).
    ``reply_parts`` / ``part_cut_off``: each non-synthetic message's text, and whether it was cut off (a
    Claude Code recovery message follows it in the stream, or it is the last one and the turn's stop_reason is
    ``max_tokens``; ``None`` when the last one has no known stop_reason, e.g. a killed turn).
    ``context_limit``: Claude Code refused a request because the conversation no longer fits the context window.
    ``usage_limit``: the failure text names a rate or usage limit.
    ``completed``: at least one non-synthetic assistant message (the model answered); otherwise the turn is a
    technical failure (API error, start-up failure, killed before any output).
    """
    events, bad = read_events(log)
    init = [e for e in events if e.get("type") == "system" and e.get("subtype") == "init"]
    results = [e for e in events if e.get("type") == "result"]
    msgs: dict[str, dict] = {}
    order: list[str] = []
    tool_uses: list[dict] = []
    user_events: list[dict] = []
    other_types: dict[str, int] = {}
    sids: set[str] = set()
    last_mid: str | None = None
    for i, ev in enumerate(events):
        if isinstance(ev.get("session_id"), str):
            sids.add(ev["session_id"])
        t = ev.get("type")
        if t == "assistant" and isinstance(ev.get("message"), dict):
            m = ev["message"]
            mid = str(m.get("id") or f"_noid{i}")
            if mid not in msgs:
                msgs[mid] = {"id": mid, "model": m.get("model"), "synthetic": m.get("model") == SYNTHETIC_MODEL,
                             "blocks": [], "text_parts": [], "stop_reason": None, "usage": None,
                             "parent_tool_use_id": ev.get("parent_tool_use_id"),
                             "is_api_error": bool(ev.get("isApiErrorMessage") or ev.get("error")),
                             "followed_by_recovery": False}
                order.append(mid)
            rec = msgs[mid]
            if not rec["synthetic"]:
                last_mid = mid
            for b in m.get("content") or []:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                rec["blocks"].append(bt)
                if bt == "text":
                    rec["text_parts"].append(b.get("text", ""))
                elif bt in ("tool_use", "server_tool_use", "mcp_tool_use"):
                    tool_uses.append({"message": mid, "type": bt, "name": b.get("name")})
            if m.get("stop_reason"):
                rec["stop_reason"] = m["stop_reason"]
            if isinstance(m.get("usage"), dict):
                rec["usage"] = m["usage"]
        elif t == "user":
            msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
            content = msg.get("content")
            blocks = [b.get("type") for b in content if isinstance(b, dict)] if isinstance(content, list) else ["text"]
            if "tool_result" in blocks:
                tool_uses.append({"message": None, "type": "tool_result", "name": None})
            text = _content_text(content)
            if last_mid is not None and is_cc_recovery(text):
                msgs[last_mid]["followed_by_recovery"] = True
            user_events.append({"text": text, "blocks": blocks,
                                "is_meta": bool(ev.get("isMeta")), "is_replay": bool(ev.get("isReplay"))})
        elif t not in ("system", "result"):
            other_types[str(t)] = other_types.get(str(t), 0) + 1
    messages = []
    for mid in order:
        r = msgs[mid]
        r["text"] = "".join(r.pop("text_parts"))
        messages.append(r)
    real = [m for m in messages if not m["synthetic"]]
    last_result = results[-1] if results else {}
    stop = last_result.get("stop_reason") or (real[-1]["stop_reason"] if real else None)
    if real and real[-1]["stop_reason"] is None and stop:
        real[-1]["stop_reason"] = stop
    reply = "\n".join(m["text"] for m in real if m["text"] != "")
    finished, exact = ends_with_marker(reply)
    part_cut: list[bool | None] = []
    for k, m in enumerate(real):
        if m["followed_by_recovery"] or m["stop_reason"] == "max_tokens":
            part_cut.append(True)
        elif k < len(real) - 1:
            part_cut.append(True)  # Claude Code only starts another API message in a tool-less turn after a cut
        elif m["stop_reason"] in ("end_turn", "stop_sequence"):
            part_cut.append(False)
        else:
            part_cut.append(None)
    failure_text = " ".join([m["text"] for m in messages if m["synthetic"]]
                            + [json.dumps(last_result.get("errors") or "")]).lower()
    context_limit = (last_result.get("terminal_reason") in CONTEXT_LIMIT_REASONS
                     or CONTEXT_LIMIT_TEXT in failure_text)
    usage_limit = any(t in failure_text for t in USAGE_LIMIT_TEXTS)
    return {
        "events": len(events),
        "unreadable_lines": bad,
        "init": init[0] if init else None,
        "init_events": len(init),
        "session_ids": sorted(sids),
        "messages": messages,
        "assistant_messages": len(real),
        "synthetic_messages": [m["text"][:500] for m in messages if m["synthetic"]],
        "tool_uses": tool_uses,
        "user_events": user_events,
        "other_event_types": other_types,
        "result_events": len(results),
        "result": {k: last_result.get(k) for k in ("subtype", "is_error", "stop_reason", "num_turns", "duration_ms",
                                                    "total_cost_usd", "usage", "modelUsage", "permission_denials",
                                                    "terminal_reason", "errors")} if results else None,
        "stop_reason": stop,
        "stop_reasons": [m["stop_reason"] for m in real],
        "models": sorted({m["model"] for m in real if isinstance(m.get("model"), str)}),
        "reply_text": reply,
        "reply_parts": [m["text"] for m in real],
        "part_cut_off": part_cut,
        "context_limit": bool(context_limit),
        "usage_limit": bool(usage_limit),
        "completed": bool(real),
        "end_marker": finished,
        "end_marker_exact": exact,
    }
