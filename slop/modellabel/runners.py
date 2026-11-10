"""The two model labellers, each run as a fresh, tool-less, blind CLI process per screen.

Opus 5.5 through Claude Code (`claude -p`) and GPT-6.1 Sol through Codex (`codex exec`). Neither process sees the
other's labels: each starts in its own empty working directory outside any repository, with no tools, and its
prompt holds only the guide and the one screen. The flags and what they isolate are listed in README.md.

`call()` runs one process and returns a CallResult; `run_with_retry()` adds the one retry on an invalid reply.
Nothing here decides what to label or where to write; run.py does.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from modellabel import common as c

# Parent-session variables that would leak into a child Claude Code process (the Clouds launcher's scrub list).
# CLAUDE_CODE_EFFORT_LEVEL outranks --effort, so it must go.
SCRUB_ENV = [
    "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_CODE_BRIDGE_SESSION_ID", "CLAUDE_CODE_SESSION_ID", "CLAUDE_PID",
    "CLAUDE_PROJECT_DIR", "CLAUDE_EFFORT", "CLAUDE_CODE_SESSION_ATTENDED", "CLAUDE_CODE_EXECPATH",
    "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_SIMPLE", "CLAUDE_CODE_SAFE_MODE", "CLAUDE_CODE_EFFORT_LEVEL",
    "CLAUDE_CODE_SUBAGENT_MODEL", "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL",
    "CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV",
    "UV_PROJECT_ENVIRONMENT", "CODEX_SANDBOX", "CODEX_SANDBOX_NETWORK_DISABLED",
]
OPUS_ENV = {
    "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK": "1",  # a refusal is recorded, never answered by another model
    "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000",
}
# Every built-in tool family, denied on top of `--tools ""` (which already removes them all).
OPUS_DENY = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit", "WebFetch", "WebSearch", "Agent",
             "Task", "Workflow", "Skill", "Artifact", "TodoWrite", "ToolSearch"]
# Codex features that give the model a tool (from `codex features list`, codex-cli 0.160.1; every name below is
# listed there as stable and on by default).
CODEX_DISABLE = ["shell_tool", "unified_exec", "apps", "plugins", "multi_agent", "image_generation", "browser_use",
                 "computer_use", "view_image", "hooks", "skill_search", "tool_suggest", "sleep_tool", "goals",
                 "code_mode_host", "browser_use_external", "browser_use_full_cdp_access", "in_app_browser",
                 "remote_plugin"]

# A refusal sentence counts only in a reply that holds no JSON at all ("I won't restate node 2; here it is: {..."
# is a broken reply, not a refusal).
REFUSAL_TEXT = re.compile(r"\b(I can(?:'|no)t help with|I(?:'m| am) (?:not able|unable) to (?:help|assist)|"
                          r"I won't|I cannot (?:help|assist|comply))\b", re.I)


def is_refusal_text(text) -> bool:
    return isinstance(text, str) and "{" not in text and bool(REFUSAL_TEXT.search(text))


@dataclass
class CallResult:
    status: str  # ok | refused | error | budget | truncated
    text: str | None = None
    cost_usd: float | None = None
    usage: dict | None = None
    exit_code: int | None = None
    stop_reason: str | None = None
    tool_calls: int = 0
    error: str | None = None
    stderr_tail: str = ""
    duration_s: float = 0.0
    started: str = ""
    ended: str = ""
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def clean_env(base=None, add=None) -> dict:
    env = dict(os.environ if base is None else base)
    for k in SCRUB_ENV:
        env.pop(k, None)
    env.update(add or {})
    return env


def default_cwd(labeller: str) -> Path:
    """An empty directory outside any repository, one per labeller and fixed across calls (a stable working
    directory keeps everything the CLI derives from it identical between screens)."""
    return Path(tempfile.gettempdir()).resolve() / "slop-modellabel" / labeller


def prepare_cwd(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    stray = [p.name for p in path.iterdir()]
    if stray:
        raise RuntimeError(f"working directory {path} is not empty ({', '.join(stray[:5])}); the labeller must start "
                           "in an empty directory")
    probe = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if probe.returncode == 0:
        raise RuntimeError(f"working directory {path} is inside a git repository ({probe.stdout.strip()})")
    return path


def _tail(s, n: int = 2000) -> str:
    if isinstance(s, bytes):
        s = s.decode("utf-8", errors="replace")
    s = s or ""
    return s[-n:]


def run_process(argv: list, stdin_text: str, cwd: Path, env: dict, timeout_s: float):
    """Run one CLI in its own process group; on a timeout (or an interrupt of this process) the whole group is killed,
    so no labeller keeps running after the harness gives up on it. Returns (returncode, stdout, stderr, timed_out);
    output is decoded as UTF-8 with replacement, so stray bytes never crash a run."""
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env,
                         start_new_session=True)
    try:
        out, err = p.communicate(stdin_text.encode("utf-8"), timeout=timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill_group(p)
        out, err = p.communicate()
        timed_out = True
    except BaseException:
        _kill_group(p)
        p.communicate()
        raise
    dec = lambda b: (b or b"").decode("utf-8", errors="replace")  # noqa: E731
    return p.returncode, dec(out), dec(err), timed_out


def _kill_group(p):
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        p.kill()


_VERSIONS: dict = {}


def cli_version(bin_path) -> str | None:
    """`<cli> --version` (no model call), cached per binary; recorded in every raw record so a change of CLI between
    the pool and sealed_real is visible and refused (run.py)."""
    key = str(bin_path)
    if key not in _VERSIONS:
        try:
            p = subprocess.run([key, "--version"], capture_output=True, timeout=30)
            out = (p.stdout or b"").decode("utf-8", errors="replace").strip()
            _VERSIONS[key] = out.splitlines()[0] if p.returncode == 0 and out else None
        except (OSError, subprocess.TimeoutExpired):
            _VERSIONS[key] = None
    return _VERSIONS[key]


class Runner:
    key = ""  # opus | codex
    labeller = ""
    model = ""
    effort = "high"
    costs_money = False

    def __init__(self, bin_path=None, cwd=None, timeout_s: float = 3600.0, env=None):
        self.bin = bin_path or shutil.which(self.default_bin, path=(env or os.environ).get("PATH"))
        if not self.bin:
            raise FileNotFoundError(f"{self.default_bin} is not on PATH")
        self.cwd = Path(cwd) if cwd else default_cwd(self.labeller)
        self.timeout_s = timeout_s
        self.base_env = env

    def describe(self) -> dict:
        return {"labeller": self.labeller, "model": self.model, "effort": self.effort, "bin": str(self.bin),
                "cli_version": cli_version(self.bin), "cwd": str(self.cwd)}

    def call(self, prefix: str, body: str, schema_path: Path) -> CallResult:
        raise NotImplementedError


class OpusRunner(Runner):
    key = "opus"
    labeller = c.LABELLERS["opus"]
    model = "claude-opus-5-5"
    default_bin = "claude"
    costs_money = True

    def __init__(self, max_budget_usd: float = 2.0, safe_mode: bool = True, **kw):
        super().__init__(**kw)
        self.max_budget_usd = max_budget_usd
        self.safe_mode = safe_mode

    def argv(self, prefix: str) -> list:
        a = [str(self.bin), "-p",
             "--model", self.model, "--effort", self.effort,
             "--output-format", "json",
             "--system-prompt", prefix,
             "--tools", "",
             "--disallowedTools", *OPUS_DENY,
             "--permission-mode", "dontAsk", "--permission-prompts", "none",
             "--setting-sources", "project",
             "--settings", json.dumps({"autoMemoryEnabled": False}),
             "--strict-mcp-config",
             "--disable-slash-commands",
             "--no-session-persistence",
             "--max-budget-usd", f"{self.max_budget_usd:.2f}"]
        if self.safe_mode:
            a.append("--safe-mode")
        return a

    def describe(self) -> dict:
        d = super().describe()
        d.update(max_budget_usd=self.max_budget_usd, safe_mode=self.safe_mode,
                 flags=[x for x in self.argv("<prefix>")[1:]], env=OPUS_ENV, scrubbed=SCRUB_ENV)
        return d

    def call(self, prefix: str, body: str, schema_path: Path = None) -> CallResult:
        self.cwd.mkdir(parents=True, exist_ok=True)
        env = clean_env(self.base_env, OPUS_ENV)
        started, t0 = c.lk.now_iso(), time.monotonic()
        rc, out, err, timed_out = run_process(self.argv(prefix), body, self.cwd, env, self.timeout_s)
        if timed_out:
            return CallResult("error", error=f"timed out after {self.timeout_s:.0f}s", exit_code=rc,
                              stderr_tail=_tail(err), started=started, ended=c.lk.now_iso(),
                              duration_s=round(time.monotonic() - t0, 2), extra={"stdout_tail": _tail(out)})
        r = parse_claude_output(out)
        r.exit_code, r.stderr_tail = rc, _tail(err)
        r.started, r.ended, r.duration_s = started, c.lk.now_iso(), round(time.monotonic() - t0, 2)
        if r.status == "ok" and rc != 0:
            r.status, r.error = "error", f"claude exited {rc}"
        return r


def parse_claude_output(stdout: str) -> CallResult:
    """Parse `claude -p --output-format json`: one result object with `result`, `total_cost_usd`, `usage`."""
    try:
        d = json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else None
    except (json.JSONDecodeError, IndexError):
        d = None
    if not isinstance(d, dict):
        return CallResult("error", error="claude printed no JSON result", extra={"stdout_tail": _tail(stdout)})
    cost = d.get("total_cost_usd")
    cost = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None
    text = d.get("result")
    r = CallResult("ok", text=text if isinstance(text, str) else None, cost_usd=cost,
                   usage=d.get("usage") if isinstance(d.get("usage"), dict) else None,
                   stop_reason=d.get("stop_reason") if isinstance(d.get("stop_reason"), str) else None,
                   extra={k: d.get(k) for k in ("subtype", "is_error", "num_turns", "session_id", "terminal_reason",
                                                "safety_stops", "stop_details", "modelUsage", "api_error_status",
                                                "permission_denials") if k in d})
    sub = d.get("subtype") if isinstance(d.get("subtype"), str) else ""
    stop_details = d.get("stop_details") if isinstance(d.get("stop_details"), dict) else {}
    ss = d.get("safety_stops")
    safety = len(ss) if isinstance(ss, (list, dict)) else ss if isinstance(ss, (int, float)) else 0
    turns = d.get("num_turns") if isinstance(d.get("num_turns"), int) else 1
    denials = d.get("permission_denials") if isinstance(d.get("permission_denials"), list) else []
    if r.stop_reason == "refusal" or stop_details.get("category") or (safety or 0) > 0:
        r.status, r.error = "refused", f"refusal (stop_reason {r.stop_reason}, {stop_details or ss or ''})"
    elif sub.startswith("error_max_budget"):
        r.status, r.error = "budget", f"per-call budget reached ({sub})"
    elif r.stop_reason in ("max_tokens", "model_context_window_exceeded"):
        # the same prompt would hit the same limit, so this is not retried within the run (README: Retry)
        r.status, r.error = "truncated", f"reply cut off (stop_reason {r.stop_reason})"
    elif d.get("is_error") or sub.startswith("error"):
        r.status, r.error = "error", f"claude reported {sub or 'an error'}: {_tail(str(d.get('result')), 300)}"
    elif turns > 1 or denials:
        r.tool_calls = max(0, turns - 1) + len(denials)
    return r


class CodexRunner(Runner):
    key = "codex"
    labeller = c.LABELLERS["codex"]
    model = "gpt-6.1-sol"
    default_bin = "codex"

    def argv(self, schema_path: Path, out_file: Path) -> list:
        a = [str(self.bin), "exec",
             "-m", self.model, "-c", f'model_reasoning_effort="{self.effort}"',
             "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
             "--ignore-user-config", "--ignore-rules",
             "-c", 'web_search="disabled"']
        for f in CODEX_DISABLE:
            a += ["--disable", f]
        a += ["--output-schema", str(schema_path), "--output-last-message", str(out_file), "--json",
              "--color", "never", "-C", str(self.cwd), "-"]
        return a

    def describe(self) -> dict:
        d = super().describe()
        d.update(flags=self.argv(Path("<schema.json>"), Path("<last-message>"))[1:], scrubbed=SCRUB_ENV)
        return d

    def call(self, prefix: str, body: str, schema_path: Path) -> CallResult:
        self.cwd.mkdir(parents=True, exist_ok=True)
        env = clean_env(self.base_env)
        started, t0 = c.lk.now_iso(), time.monotonic()
        with tempfile.TemporaryDirectory(prefix="slop-codex-out-") as td:
            out_file = Path(td) / "last_message.json"
            rc, out, err, timed_out = run_process(self.argv(schema_path, out_file), prefix + "\n\n" + body,
                                                  self.cwd, env, self.timeout_s)
            if timed_out:
                return CallResult("error", error=f"timed out after {self.timeout_s:.0f}s", exit_code=rc,
                                  stderr_tail=_tail(err), started=started, ended=c.lk.now_iso(),
                                  duration_s=round(time.monotonic() - t0, 2))
            last = out_file.read_bytes().decode("utf-8", errors="replace") if out_file.exists() else None
        r = parse_codex_events(out, last)
        r.exit_code, r.stderr_tail = rc, _tail(err)
        r.started, r.ended, r.duration_s = started, c.lk.now_iso(), round(time.monotonic() - t0, 2)
        if r.status == "ok" and rc != 0:
            r.status, r.error = "error", f"codex exited {rc}"
        return r


# Allow-list: any other item type in the event stream (command_execution, mcp_tool_call, web_search, file_change,
# collab_tool_call, dynamic_tool_call, image_view, or a type this harness has never seen) is a tool use, whether the
# item completed or only started.
SAFE_ITEMS = {"agent_message", "reasoning", "error", "todo_list"}


def parse_codex_events(stdout: str, last_message: str | None) -> CallResult:
    """Parse `codex exec --json` JSONL events (thread.started, turn.started, item.started/updated/completed,
    turn.completed with usage, turn.failed, error) plus the --output-last-message file."""
    usage, errors, tools, agent_msgs, kinds, seen, models = {}, [], [], [], {}, set(), set()
    for n, line in enumerate((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        t = ev.get("type") if isinstance(ev.get("type"), str) else ""
        kinds[t] = kinds.get(t, 0) + 1
        for k in ("model", "model_slug"):
            if isinstance(ev.get(k), str):
                models.add(ev[k])
        if t == "turn.completed":
            u = ev.get("usage") if isinstance(ev.get("usage"), dict) else {}
            for k, v in u.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    usage[k] = usage.get(k, 0) + v
        elif t == "turn.failed":
            e = ev.get("error")
            errors.append(e.get("message") if isinstance(e, dict) and isinstance(e.get("message"), str)
                          else json.dumps(e))
        elif t == "error":
            errors.append(ev.get("message") if isinstance(ev.get("message"), str) else json.dumps(ev))
        elif t.startswith("item."):
            item = ev.get("item") if isinstance(ev.get("item"), dict) else {}
            it = item.get("type") or item.get("item_type")
            key = item.get("id") if item.get("id") is not None else ("line", n)
            if it not in SAFE_ITEMS:
                if key not in seen:
                    seen.add(key)
                    tools.append(str(it))
            elif it == "agent_message" and t == "item.completed" and isinstance(item.get("text"), str):
                agent_msgs.append(item["text"])
    text = last_message if last_message is not None and last_message.strip() else (agent_msgs[-1] if agent_msgs
                                                                                    else None)
    r = CallResult("ok", text=text, usage=usage or None, tool_calls=len(tools),
                   extra={"events": kinds, "tools": tools, **({"models": sorted(models)} if models else {})})
    joined = " ".join(str(x) for x in errors)
    if errors and re.search(r"refus|content[_ ]?(policy|filter)|safety|cyber_policy|policy_violation", joined, re.I):
        r.status, r.error = "refused", f"refusal: {joined[:300]}"
    elif errors and not text:
        r.status, r.error = "error", joined[:500]
    elif is_refusal_text(text):
        r.status, r.error = "refused", f"refusal text: {text[:200]}"
    elif text is None:
        r.status, r.error = "error", "codex gave no final message"
    return r


# ---------------------------------------------------------------- spend cap


class Budget:
    """Hard caps on Opus spend (Claude Code's total_cost_usd).

    - `cap_usd` caps this invocation; `total_cap_usd` caps everything in the ledger (`labels/raw/spend.jsonl`: every
      set, phase and run, `prior_usd` at the start of this one).
    - A call launches only if the spend so far plus the expected cost of every call in flight, this one included, stays
      within both caps. Expected cost = mean of the calls with a reported cost, or `per_call_max` before any.
    - When a launch does not fit while calls are in flight, it waits for them (their cost may make room); it is refused,
      and launches stop for good, only when nothing is in flight and it still does not fit.
    - A call that reports no cost (timeout, garbled output, crash) may still have been billed: it is charged at
      `per_call_max` and marked estimated.
    - Every launch is written to the ledger before the process starts and every charge after it ends, so a launch with
      no charge (this process was killed mid-call) counts at its reservation (`per_call_max`) in `ledger_total`.
    Overshoot of a cap is bounded by the in-flight calls' excess over the expected cost, at most workers ×
    `per_call_max`, provided no single call costs more than `per_call_max` (claude's --max-budget-usd is checked between
    turns, so one turn can pass it: README "Spend cap")."""

    def __init__(self, cap_usd, per_call_max: float, ledger: Path | None = None, total_cap_usd=None,
                 prior_usd: float = 0.0):
        self.cap = cap_usd
        self.total_cap = total_cap_usd
        self.prior = float(prior_usd or 0.0)
        self.per_call_max = per_call_max
        self.spent = 0.0
        self.reported = 0.0
        self.calls = 0
        self.paid = 0
        self.inflight = 0
        self.estimated = 0
        self.stopped = False
        self.ledger = ledger
        self.cond = threading.Condition()

    def expected(self) -> float:
        """Mean cost of the calls that reported a cost so far, or the per-call maximum before any has."""
        return self.reported / self.paid if self.paid else self.per_call_max

    def _fits(self) -> bool:
        need = (self.inflight + 1) * self.expected()
        if self.cap is not None and self.spent + need > self.cap + 1e-9:
            return False
        if self.total_cap is not None and self.prior + self.spent + need > self.total_cap + 1e-9:
            return False
        return True

    def try_reserve(self, block: bool = True, note: dict | None = None):
        """A launch id (truthy) when the call may start, else None."""
        with self.cond:
            while True:
                if self.stopped:
                    return None
                if self._fits():
                    self.inflight += 1
                    lid = uuid.uuid4().hex
                    self._append({"event": "launch", "launch": lid, "reserve_usd": self.per_call_max, **(note or {})})
                    return lid
                if self.inflight == 0 or not block:
                    self.stopped = True
                    self.cond.notify_all()
                    return None
                self.cond.wait()

    def release(self, cost, note: dict | None = None, launch: str | None = None):
        estimated = not (isinstance(cost, (int, float)) and not isinstance(cost, bool))
        charge = self.per_call_max if estimated else float(cost)
        with self.cond:
            self.inflight = max(0, self.inflight - 1)
            self.spent += charge
            if not estimated:
                self.reported += charge
                if charge:
                    self.paid += 1
            else:
                self.estimated += 1
            self.calls += 1
            self._append({"event": "charge", "launch": launch, "cost_usd": charge, "estimated": estimated,
                          **(note or {})})
            self.cond.notify_all()

    def _append(self, rec: dict):
        if self.ledger is None:
            return
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"time": c.lk.now_iso(), **rec}, ensure_ascii=False) + "\n"
        with open(self.ledger, "a+b") as f:
            f.seek(0, os.SEEK_END)
            if f.tell():
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":  # a torn last line (a killed writer) must not swallow this one
                    line = "\n" + line
            f.write(line.encode("utf-8"))


class NoBudget(Budget):
    def __init__(self):
        super().__init__(None, 0.0)


def ledger_summary(ledger: Path) -> dict:
    """Charges (reported and estimated) plus launches that never got a charge, each at its reservation."""
    out = {"charged_usd": 0.0, "estimated_usd": 0.0, "orphan_launches": 0, "orphan_usd": 0.0, "bad_lines": 0}
    launches, charged = {}, set()
    p = Path(ledger)
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if not isinstance(r, dict):
                    raise ValueError
            except ValueError:
                out["bad_lines"] += 1
                continue
            if r.get("event") == "launch":
                launches[r.get("launch")] = float(r.get("reserve_usd") or 0)
                continue
            try:
                v = float(r.get("cost_usd") or 0)
            except (TypeError, ValueError):
                out["bad_lines"] += 1
                continue
            out["charged_usd"] += v
            if r.get("estimated"):
                out["estimated_usd"] += v
            if r.get("launch"):
                charged.add(r["launch"])
    for lid, res in launches.items():
        if lid not in charged:
            out["orphan_launches"] += 1
            out["orphan_usd"] += res
    out["total_usd"] = out["charged_usd"] + out["orphan_usd"]
    return out


def ledger_total(ledger: Path) -> float:
    return ledger_summary(ledger)["total_usd"]


# ---------------------------------------------------------------- one screen, one retry


RETRY_NOTE = ("\n\nYour previous reply to this message failed validation:\n{errors}\n"
              "Reply again with the complete JSON object for this screen, fixing these problems.")
CHECK_NOTE = ("\n\nYour previous reply to this message was valid, but these labels look inconsistent with the guide:"
              "\n{errors}\nCheck each against the guide (§4, §5). Reply again with the complete JSON object for this "
              "screen; keep a label unchanged if the guide supports it.")


def run_with_retry(runner: Runner, prefix: str, body: str, schema_path: Path, validate, budget: Budget,
                   note: dict | None = None) -> dict:
    """Call the runner; on an unparseable or invalid reply, call once more with the errors appended. A reply that is
    valid but has soft issues (validate's optional third list, e.g. an ECHO that points at a lower-ranked node,
    LABELLING.md §5) is also called once more with them listed, and the second reply is accepted with the issues kept
    as warnings. A refusal is recorded and not retried; a CLI error, a cut-off reply or a per-call budget stop is
    recorded and left for the next run. A crash inside the call is recorded as an error, and an Opus call is always
    charged (at --per-call-usd when the CLI reported no cost).

    validate(obj) -> (errors, warnings[, soft]). Returns {"status", "reply", "attempts", "cost_usd", "error"} where
    status is ok, invalid, refused, error, budget, truncated or capped (the spend cap stopped a launch)."""
    attempts, cost = [], 0.0
    prompt_body = body
    fallback = None  # a valid first reply that was only sent back for soft issues

    def done(status, error):
        if fallback is not None:  # the check-again call failed: the valid first reply stands
            return {"status": "ok", "reply": fallback[0], "attempts": attempts, "cost_usd": cost, "error": None,
                    "warnings": fallback[1] + [f"the check-again call ended {status}: {error}"]}
        return {"status": status, "reply": None, "attempts": attempts, "cost_usd": cost, "error": error}

    for n in (1, 2):
        lid = None
        if runner.costs_money:
            lid = budget.try_reserve(note={**(note or {}), "attempt": n})
            if not lid:
                return done("capped", "spend cap reached")
        r = None
        try:
            r = runner.call(prefix, prompt_body, schema_path)
        except Exception as e:  # noqa: BLE001 - one call's crash must not lose its charge or stop the run
            r = CallResult("error", error=f"harness error in the call: {type(e).__name__}: {e}")
        finally:
            if runner.costs_money:
                budget.release(r.cost_usd if r is not None else None,
                               {**(note or {}), "attempt": n, "status": r.status if r is not None else "crash"}, lid)
        cost += r.cost_usd if r.cost_usd is not None else (budget.per_call_max if runner.costs_money else 0.0)
        att = {"n": n, **r.to_json(), "errors": [], "warnings": [], "soft": [],
               "retry_note": None if n == 1 else prompt_body[len(body):]}
        attempts.append(att)
        if r.status != "ok":
            return done(r.status, r.error)
        errors = []
        if r.tool_calls:
            errors.append(f"the reply used {r.tool_calls} tool call(s); labelling must use no tools or files")
        obj, perr = c.extract_json(r.text)
        if perr:
            if is_refusal_text(r.text):
                att["errors"] = [perr]
                return done("refused", f"refusal text: {r.text[:200]}")
            errors.append(perr)
        warnings, soft = [], []
        if obj is not None and not errors:
            res = validate(obj)
            errors, warnings = list(res[0]), list(res[1])
            soft = list(res[2]) if len(res) > 2 else []
        att["errors"], att["warnings"], att["soft"] = errors, warnings, soft
        if not errors and (not soft or n == 2):
            return {"status": "ok", "reply": obj, "attempts": attempts, "cost_usd": cost, "error": None,
                    "warnings": warnings + soft}
        if errors:
            prompt_body = body + RETRY_NOTE.format(errors="\n".join(f"- {e}" for e in errors[:40]))
        else:
            fallback = (obj, warnings + soft)
            prompt_body = body + CHECK_NOTE.format(errors="\n".join(f"- {e}" for e in soft[:40]))
    return done("invalid", "; ".join(attempts[-1]["errors"][:5]))


def make_runner(key: str, **kw) -> Runner:
    if key == "opus":
        return OpusRunner(**kw)
    if key == "codex":
        kw.pop("max_budget_usd", None)
        kw.pop("safe_mode", None)
        return CodexRunner(**kw)
    raise ValueError(key)
