"""The v0.5 turn loop behind scripts/v05_run.py (and scripts/v05_launch.sh).

A run (r1, r2 or r3) is one conversation with the author model (agent_loop_lab.v05.session). Turn 001 sends
PROMPT.txt in a new session (``--session-id``); every later turn resumes it (``--resume``) with exactly
``Continue.``. The loop stops when a reply ends with the line ``END OF WEIGHTS`` (``finished``), after
:data:`~agent_loop_lab.v05.session.CONTINUATION_CAP` continuation turns (``turn_cap``), when the conversation no
longer fits the model's context window (``context_limit``: Claude Code refused the request; not retried), when the
US$500 cap is reached (``cost_cap``) or a killed launch's cost is unknown (``cost_unresolved``), when a failure
names a rate or usage limit (``usage_limit``: stopped at once, resume later), or when a turn failed technically
more than :data:`~agent_loop_lab.v05.session.RETRY_CAP` times (``technical_stop``). :data:`FINAL` statuses end the
run: the weights written so far are what gets scored. :data:`RESUMABLE` statuses can be continued with
``resume=True``; a resume gives the pending turn a fresh retry budget (recorded as ``retry_budget_reset``).
Technical retries wait :data:`RETRY_WAITS` seconds first (recorded as ``retry_wait``).

Files (ROOT = experiments/v0_5):

* ``ROOT/logs/<run>.tNNN.stream.jsonl`` the turn's transcript (never overwritten), ``.stderr.log``, and
  ``.launch.json`` (written before claude starts: message hash, session, ``--max-budget-usd`` ceiling, argv);
* a technical failure (no assistant message from the model: API error, start-up failure, killed before any
  output) is moved aside as ``<run>.tNNN.fail<K>.*`` (a session was started: its cost counts) or
  ``<run>.tNNN.noinit<K>.stream.log`` etc. (no init event: nothing was sent) and the same turn is retried with
  the same message, at most RETRY_CAP times; a turn-001 retry after the session was created uses a new session
  id (the abandoned one is recorded), since the prompt may not be sent twice in one conversation;
* ``ROOT/logs/LAUNCHES.jsonl`` one line per attempt; ``ROOT/logs/<run>.session.jsonl`` a snapshot of Claude
  Code's session file after every turn (the exact requests sent, for the audit);
* ``ROOT/runs/<run>/run.json`` the run record, ``ROOT/runs/<run>/replies/NNN.md`` each turn's reply text (the
  text of its API messages joined by a newline: a view for people), ``NNN.p<K>.md`` each API message of a turn
  that has more than one (Claude Code continued a cut reply inside the turn), and ``replies/INDEX.json``: for
  every turn, the files the edits reader reads (the parts when there are several, else ``NNN.md``) in order,
  each with ``cut_off`` (true: ended at the output limit; false: ended normally; null: unknown).

Resumable: rerunning with ``resume=True`` reads what is on disk, finishes the bookkeeping of the last turn and
sends the next one; a turn whose transcript exists is never sent again. The run's working directory is a fresh
empty directory in the system temp root, kept for the whole run (``--resume`` looks the session up by it).
Before every turn: the cost cap, no ``pytest-of-$USER`` in the temp roots, PROMPT.txt against PROMPT.sha256,
the working directory still empty and free of CLAUDE.md on its path.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from agent_loop_lab.v05 import cost as v05_cost
from agent_loop_lab.v05.session import (
    CONTINUATION_CAP, CONTINUE, EFFORT, MODEL, RETRY_CAP, RUNS, SET_ENV, SYSTEM_PROMPT, TURN_BUDGET_USD,
    child_env, claude_argv, find_session_file, parse_turn, printable_argv, sha256_file, sha256_text, turn_name,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "experiments" / "v0_5"
TURN_WALL_SECONDS = 6 * 3600
FINAL = ("finished", "turn_cap", "context_limit", "cost_cap")
RESUMABLE = ("cost_unresolved", "technical_stop", "usage_limit")
TERMINAL = FINAL + RESUMABLE
RETRY_WAITS = (120, 600, 1800)  # seconds before technical retry 1, 2, 3 of a turn


class Refusal(RuntimeError):
    pass


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def read_sha_file(path: Path) -> str:
    if not path.is_file() or not path.read_text().strip():
        raise Refusal(f"{path} missing or empty")
    h = path.read_text().split()[0].strip().lower()
    if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
        raise Refusal(f"{path} does not hold a sha256")
    return h


def verify_prompt(root: Path) -> tuple[str, str]:
    """``(prompt text, sha256)`` of ROOT/PROMPT.txt after checking it against ROOT/PROMPT.sha256."""
    p = root / "PROMPT.txt"
    if not p.is_file():
        raise Refusal(f"{p} missing")
    want = read_sha_file(root / "PROMPT.sha256")
    got = sha256_file(p)
    if got != want:
        raise Refusal(f"{p} sha256 {got} does not match PROMPT.sha256 {want}")
    text = p.read_bytes().decode("utf-8")
    if not text.strip():
        raise Refusal(f"{p} is empty")
    if len(text.encode("utf-8")) > 800_000:
        raise Refusal(f"{p} is too long to pass as one argument ({len(text.encode('utf-8'))} bytes)")
    return text, got


def temp_roots() -> list[str]:
    roots = [os.path.realpath(tempfile.gettempdir()), os.environ.get("TMPDIR") or "/tmp", "/tmp", "/private/tmp"]
    try:
        out = subprocess.run(["getconf", "DARWIN_USER_TEMP_DIR"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            roots.append(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    seen, out_roots = set(), []
    for r in roots:
        rp = os.path.realpath(r)
        if rp not in seen:
            seen.add(rp)
            out_roots.append(rp)
    return out_roots


def pytest_scratch() -> list[str]:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    if not user:
        try:
            import pwd
            user = pwd.getpwuid(os.getuid()).pw_name
        except (ImportError, KeyError):
            user = "unknown"
    return [os.path.join(r, f"pytest-of-{user}") for r in temp_roots() if os.path.exists(os.path.join(r, f"pytest-of-{user}"))]


def context_files_on_path(cwd: Path) -> list[str]:
    """CLAUDE.md / CLAUDE.local.md / .claude directories in the working directory or any of its parents."""
    hits = []
    for d in [cwd, *cwd.parents]:
        for name in ("CLAUDE.md", "CLAUDE.local.md", ".claude", "AGENTS.md"):
            p = d / name
            if p.exists() and os.path.realpath(p) != os.path.realpath(Path.home() / ".claude"):
                hits.append(str(p))
    return hits


def workdir_base() -> str:
    """Where run directories are made: the per-user system temp dir (``getconf DARWIN_USER_TEMP_DIR``), not
    $TMPDIR, which inside a Claude Code session points at a scratch path named after the repo. Claude Code shows
    the working directory to the model (its environment attachment), so the path must not say anything."""
    try:
        out = subprocess.run(["getconf", "DARWIN_USER_TEMP_DIR"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip() and os.path.isdir(out.stdout.strip()):
            return os.path.realpath(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return os.path.realpath("/tmp")


PATH_WORDS = ("hypanet", "hypabolic", "doom", "claude", "weight", "v0_5", "v05", "agent_loop")


def check_workdir(cwd: Path) -> None:
    real = Path(os.path.realpath(cwd))
    if str(real).startswith(str(REPO) + os.sep) or real == REPO:
        raise Refusal(f"working directory {real} is inside the repo")
    words = [w for w in PATH_WORDS if w in str(real).lower()]
    if words:
        raise Refusal(f"working directory {real} names {words}; Claude Code shows this path to the model")
    if any(real.iterdir()):
        raise Refusal(f"working directory {real} is not empty: {sorted(p.name for p in real.iterdir())[:10]}")
    hits = context_files_on_path(real)
    if hits:
        raise Refusal(f"context files on the working directory's path: {hits}")


def claude_version() -> str | None:
    try:
        out = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=60,
                             env=child_env(), stdin=subprocess.DEVNULL)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class Layout:
    def __init__(self, root: Path, run: str):
        if run not in RUNS:
            raise Refusal(f"unknown run {run!r} (expected one of {RUNS})")
        self.root, self.run = Path(root), run
        self.logs = self.root / "logs"
        self.run_dir = self.root / "runs" / run
        self.replies = self.run_dir / "replies"
        self.record = self.run_dir / "run.json"
        self.lock = self.run_dir / ".lock"
        self.launches = self.logs / "LAUNCHES.jsonl"
        self.session_snapshot = self.logs / f"{run}.session.jsonl"

    def transcript(self, n: int) -> Path:
        return self.logs / f"{turn_name(self.run, n)}.stream.jsonl"

    def sidecar(self, name: str, suffix: str) -> Path:
        return self.logs / f"{name}{suffix}"

    def attempts(self, n: int) -> list[str]:
        """Names of the moved-aside attempts of turn n (failK and noinitK), in order."""
        base = turn_name(self.run, n)
        names = set()
        for p in self.logs.glob(f"{base}.fail*.stream.jsonl"):
            names.add(p.name[: -len(".stream.jsonl")])
        for p in self.logs.glob(f"{base}.noinit*.stream.log"):
            names.add(p.name[: -len(".stream.log")])
        return sorted(names, key=lambda s: int("".join(c for c in s.rsplit(".", 1)[-1] if c.isdigit()) or 0))

    def existing_transcripts(self) -> list[Path]:
        return sorted(self.logs.glob(f"{self.run}.t*.stream.jsonl"))


def _move_aside(lay: Layout, n: int, kind: str) -> str:
    base = turn_name(lay.run, n)
    k = len(lay.attempts(n)) + 1
    new = f"{base}.{kind}{k}"
    stream_suffix = ".stream.jsonl" if kind == "fail" else ".stream.log"
    os.replace(lay.transcript(n), lay.sidecar(new, stream_suffix))
    for suf in (".stderr.log", ".launch.json"):
        src = lay.sidecar(base, suf)
        if src.exists():
            os.replace(src, lay.sidecar(new, suf))
    return new


def _snapshot_session(lay: Layout, rec: dict) -> None:
    sf = find_session_file(rec["session_id"], rec["workdir"])
    if sf is not None:
        shutil.copyfile(sf, lay.session_snapshot)
        rec["session_file"] = str(sf)
    else:
        rec["events"].append({"utc": utc(), "event": "session_snapshot_missing", "session_id": rec["session_id"]})
    for sid in rec.get("abandoned_session_ids", []):
        f = find_session_file(sid, rec["workdir"])
        if f is not None:
            shutil.copyfile(f, lay.logs / f"{lay.run}.abandoned-{sid}.session.jsonl")


def _write_replies(lay: Layout, rec: dict, parsed: dict[int, dict]) -> None:
    lay.replies.mkdir(parents=True, exist_ok=True)
    units = []
    for n in sorted(parsed):
        p = parsed[n]
        f = lay.replies / f"{n:03d}.md"
        f.write_text(p["reply_text"], encoding="utf-8")
        cuts = p["part_cut_off"]
        parts = []
        if len(p["reply_parts"]) > 1:
            for k, (txt, cut) in enumerate(zip(p["reply_parts"], cuts), 1):
                pf = lay.replies / f"{n:03d}.p{k}.md"
                pf.write_text(txt, encoding="utf-8")
                parts.append({"file": pf.name, "chars": len(txt), "sha256": sha256_text(txt), "cut_off": cut})
        units.append({"turn": n, "file": f.name, "parts": parts, "chars": len(p["reply_text"]),
                      "sha256": sha256_text(p["reply_text"]), "stop_reason": p["stop_reason"],
                      "cut_off": cuts[-1] if cuts else None, "end_marker": p["end_marker"]})
    _write_json(lay.replies / "INDEX.json", {
        "run": lay.run, "status": rec.get("status"),
        "reading_order": "every ```edits block of every unit, in this order (turn order, then position in the file); "
                         "a unit with parts is read as its parts, each a reply of its own (one API message each)",
        "join": "NNN.md joins a turn's parts with a newline, for people only; the reader reads the parts",
        "units": units})


def _turn_entry(lay: Layout, n: int, p: dict, message_sha: str) -> dict:
    res = p["result"] or {}
    return {"turn": n, "name": turn_name(lay.run, n), "message": "PROMPT.txt" if n == 1 else CONTINUE,
            "message_sha256": message_sha, "transcript": lay.transcript(n).name, "session_ids": p["session_ids"],
            "failed_attempts": lay.attempts(n), "assistant_messages": p["assistant_messages"],
            "stop_reason": p["stop_reason"], "cut_off": p["stop_reason"] == "max_tokens",
            "api_messages_cut_off": p["part_cut_off"],
            "result_subtype": res.get("subtype"), "is_error": res.get("is_error"),
            "has_result_event": p["result"] is not None, "session_total_cost_usd": res.get("total_cost_usd"),
            "reply_chars": len(p["reply_text"]), "end_marker": p["end_marker"],
            "end_marker_exact": p["end_marker_exact"], "models": p["models"]}


def _launch(argv: list[str], cwd: Path, env: dict, out: Path, err: Path, wall: int) -> int:
    """Run claude in the foreground of this process group under a wall-clock watchdog; returns its exit code."""
    with open(out, "x", encoding="utf-8") as fo, open(err, "x", encoding="utf-8") as fe:
        proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe)
        t0 = time.monotonic()
        try:
            while proc.poll() is None:
                if time.monotonic() - t0 > wall:
                    fe.write(f"\n[v05 runner] WATCHDOG: turn ran {wall}s; SIGINT\n")
                    proc.send_signal(signal.SIGINT)
                    try:
                        proc.wait(60)
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                        try:
                            proc.wait(30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                    break
                time.sleep(2)
        except KeyboardInterrupt:
            # claude got the same SIGINT (same process group); let it exit, keep its transcript, then stop
            try:
                proc.wait(90)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        return proc.wait()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run(run_name: str, *, resume: bool = False, root: Path = DEFAULT_ROOT, model: str | None = None,
        bare: bool | None = None, wall: int = TURN_WALL_SECONDS,
        launch: Callable[[list[str], Path, dict, Path, Path, int], int] = _launch,
        log: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep,
        continuation_cap: int | None = None, turn_budget_usd: float | None = None) -> dict:
    """Run (or resume) one run until a terminal status; returns the run record."""
    lay = Layout(root, run_name)
    overrides = (model, continuation_cap, turn_budget_usd)
    if any(x is not None for x in overrides) and Path(root).resolve() == DEFAULT_ROOT.resolve():
        raise Refusal("model / cap / budget overrides are only allowed with a non-default root (kit and smoke tests)")
    lay.logs.mkdir(parents=True, exist_ok=True)
    have = lay.existing_transcripts() or list(lay.logs.glob(f"{run_name}.t*.stream.log"))
    if not resume and (have or lay.record.exists()):
        raise Refusal(f"{run_name} already has transcripts or a run record; use --resume to continue it")
    if resume and not lay.record.exists():
        raise Refusal(f"{run_name} has no run record to resume ({lay.record})")
    lay.run_dir.mkdir(parents=True, exist_ok=True)
    if lay.lock.exists():
        try:
            pid = int(lay.lock.read_text().split()[0])
        except (ValueError, IndexError):
            pid = -1
        if pid > 0 and _pid_alive(pid):
            raise Refusal(f"{run_name} is locked by running process {pid} ({lay.lock})")
    lay.lock.write_text(f"{os.getpid()} {utc()}\n")
    try:
        return _loop(lay, resume=resume, model=model, bare=bare, wall=wall, launch=launch, log=log, sleep=sleep,
                     continuation_cap=CONTINUATION_CAP if continuation_cap is None else continuation_cap,
                     turn_budget_usd=TURN_BUDGET_USD if turn_budget_usd is None else turn_budget_usd)
    finally:
        try:
            lay.lock.unlink()
        except FileNotFoundError:
            pass


def _new_record(lay: Layout, prompt_sha: str, model: str, bare: bool, continuation_cap: int,
                turn_budget_usd: float) -> dict:
    workdir = Path(os.path.realpath(tempfile.mkdtemp(prefix="run-", dir=workdir_base())))
    return {"run": lay.run, "status": "running", "utc_created": utc(), "model": model, "effort": EFFORT,
            "system_prompt": SYSTEM_PROMPT, "continue_message": CONTINUE, "bare": bare,
            "prompt_sha256": prompt_sha, "session_id": str(uuid.uuid4()), "abandoned_session_ids": [],
            "workdir": str(workdir), "continuation_cap": continuation_cap, "retry_cap": RETRY_CAP,
            "retry_waits_s": list(RETRY_WAITS), "retry_offsets": {}, "turn_budget_usd": turn_budget_usd,
            "env_set": SET_ENV, "claude_version": claude_version(),
            "turns": [], "events": [{"utc": utc(), "event": "created"}]}


def _loop(lay: Layout, *, resume: bool, model: str | None, bare: bool | None, wall: int, launch, log, sleep,
          continuation_cap: int, turn_budget_usd: float) -> dict:
    prompt, prompt_sha = verify_prompt(lay.root)
    if resume:
        rec = json.loads(lay.record.read_text())
        if rec["prompt_sha256"] != prompt_sha:
            raise Refusal(f"PROMPT.txt changed since the run started ({rec['prompt_sha256']} -> {prompt_sha})")
        if bare is not None and bool(bare) != bool(rec["bare"]):
            raise Refusal(f"V05_BARE differs from the run's ({rec['bare']})")
        if model is not None and model != rec["model"]:
            raise Refusal(f"model differs from the run's ({rec['model']})")
        rec["events"].append({"utc": utc(), "event": "resumed", "status_was": rec["status"]})
        if rec["status"] in FINAL:
            log(f"{lay.run}: already {rec['status']}")
            return rec
        rec["status"] = "running"
        reset_retries = True
        continuation_cap = rec.get("continuation_cap", continuation_cap)
        turn_budget_usd = rec.get("turn_budget_usd", turn_budget_usd)
    else:
        rec = _new_record(lay, prompt_sha, model or MODEL, bool(bare), continuation_cap, turn_budget_usd)
        reset_retries = False
    offsets: dict = rec.setdefault("retry_offsets", {})
    workdir = Path(rec["workdir"])
    if not workdir.exists():
        workdir.mkdir(parents=True)
        rec["events"].append({"utc": utc(), "event": "workdir_recreated"})
    if (rec["bare"] and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"))):
        raise Refusal("bare mode needs ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN in the environment")
    _write_json(lay.record, rec)
    parsed: dict[int, dict] = {}
    n = 1
    while True:
        t = lay.transcript(n)
        if t.exists():
            p = parse_turn(t)
            if p["init"] is None:
                name = _move_aside(lay, n, "noinit")
                rec["events"].append({"utc": utc(), "event": "attempt_noinit", "attempt": name})
                log(f"{lay.run} turn {n:03d}: no init event (nothing sent); moved aside as {name}")
                _write_json(lay.record, rec)
                continue
            if not p["completed"]:
                name = _move_aside(lay, n, "fail")
                rec["events"].append({"utc": utc(), "event": "attempt_failed", "attempt": name,
                                      "result": p["result"], "synthetic": p["synthetic_messages"],
                                      "context_limit": p["context_limit"], "usage_limit": p["usage_limit"]})
                log(f"{lay.run} turn {n:03d}: technical failure (no model output); moved aside as {name}")
                _write_json(lay.record, rec)
                if p["context_limit"]:
                    rec["status"] = "context_limit"
                    log(f"{lay.run} turn {n:03d}: the conversation no longer fits the context window; stopping")
                    break
                if p["usage_limit"]:
                    rec["status"] = "usage_limit"
                    log(f"{lay.run} turn {n:03d}: a rate or usage limit; stopping (resume later with --resume)")
                    break
                continue
            parsed[n] = p
            entry = _turn_entry(lay, n, p, prompt_sha if n == 1 else sha256_text(CONTINUE))
            rec["turns"] = [e for e in rec["turns"] if e["turn"] != n] + [entry]
            rec["turns"].sort(key=lambda e: e["turn"])
            _write_replies(lay, rec, parsed)
            _write_json(lay.record, rec)
            if p["end_marker"]:
                rec["status"] = "finished"
                break
            n += 1
            continue
        # turn n has not been sent
        if n - 1 > continuation_cap:
            rec["status"] = "turn_cap"
            log(f"{lay.run}: continuation cap ({continuation_cap}) reached without END OF WEIGHTS")
            break
        if reset_retries:
            reset_retries = False
            if lay.attempts(n):
                offsets[str(n)] = len(lay.attempts(n))
                rec["events"].append({"utc": utc(), "event": "retry_budget_reset", "turn": n,
                                      "earlier_attempts": offsets[str(n)]})
        retries = len(lay.attempts(n)) - int(offsets.get(str(n), 0))
        if retries > RETRY_CAP:
            rec["status"] = "technical_stop"
            log(f"{lay.run} turn {n:03d}: {RETRY_CAP} technical retries used; stopping (resumable)")
            break
        if retries > 0:
            wait = RETRY_WAITS[min(retries, len(RETRY_WAITS)) - 1]
            rec["events"].append({"utc": utc(), "event": "retry_wait", "turn": n, "retry": retries, "seconds": wait})
            _write_json(lay.record, rec)
            log(f"{lay.run} turn {n:03d}: waiting {wait}s before technical retry {retries}")
            sleep(wait)
        s = v05_cost.summary(lay.logs)
        code, msg = v05_cost.check(s)
        if code:
            rec["status"] = "cost_cap" if code == v05_cost.EXIT_CAP else "cost_unresolved"
            log(f"{lay.run}: {msg}")
            break
        budget = min(turn_budget_usd, int(s["cap"]["remaining_usd"] * 100) / 100)
        if budget < 0.01:
            rec["status"] = "cost_cap"
            break
        scratch = pytest_scratch()
        if scratch:
            raise Refusal(f"harness scratch exists: {scratch}; remove it and do not run the test suite during a run")
        prompt, prompt_sha2 = verify_prompt(lay.root)
        if prompt_sha2 != rec["prompt_sha256"]:
            raise Refusal("PROMPT.txt changed during the run")
        check_workdir(workdir)
        if n == 1 and find_session_file(rec["session_id"], workdir) is not None:
            # turn 001 failed after the session was created: the prompt may not be sent twice in one session
            rec["abandoned_session_ids"].append(rec["session_id"])
            rec["session_id"] = str(uuid.uuid4())
            rec["events"].append({"utc": utc(), "event": "new_session_for_turn_001_retry",
                                  "abandoned": rec["abandoned_session_ids"][-1], "session_id": rec["session_id"]})
        message = prompt if n == 1 else CONTINUE
        argv = claude_argv(session_id=rec["session_id"], resume=n > 1, message=message, max_budget_usd=budget,
                           model=rec["model"], bare=rec["bare"])
        name = turn_name(lay.run, n)
        launch_rec = {"name": name, "run": lay.run, "turn": n, "attempt": len(lay.attempts(n)) + 1,
                      "session_id": rec["session_id"], "resume": n > 1, "message_sha256": sha256_text(message),
                      "message_chars": len(message), "max_budget_usd": budget, "model": rec["model"],
                      "effort": EFFORT, "workdir": str(workdir), "argv": printable_argv(argv), "utc_start": utc(),
                      "spent_before_usd": s["cap"]["total_usd"]}
        _write_json(lay.sidecar(name, ".launch.json"), launch_rec)
        _write_json(lay.record, rec)
        log(f"{lay.run} turn {n:03d}: launching ({'prompt' if n == 1 else CONTINUE}) session={rec['session_id']} "
            f"budget=US${budget:.2f} spent=US${s['cap']['total_usd']}")
        rc = None
        try:
            rc = launch(argv, workdir, child_env(), t, lay.sidecar(name, ".stderr.log"), wall)
        finally:
            _append_jsonl(lay.launches, {**launch_rec, "utc_end": utc(), "rc": rc})
            try:
                _snapshot_session(lay, rec)
            except OSError as exc:
                rec["events"].append({"utc": utc(), "event": "session_snapshot_failed", "error": str(exc)})
            _write_json(lay.record, rec)
        log(f"{lay.run} turn {n:03d}: claude exited {rc}")
    rec["events"].append({"utc": utc(), "event": "stopped", "status": rec["status"]})
    rec["utc_stopped"] = utc()
    rec["finished"] = rec["status"] == "finished"
    rec["turns_sent"] = len(rec["turns"])
    rec["cost"] = v05_cost.summary(lay.logs)["cap"]
    _write_replies(lay, rec, parsed)
    _write_json(lay.record, rec)
    return rec
