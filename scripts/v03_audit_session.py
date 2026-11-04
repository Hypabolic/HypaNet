"""Audit a v0.3 weight-author session transcript (spec v0.3 §8 "Isolation", §9.2-9.3).

    uv run python scripts/v03_audit_session.py LOG.stream.jsonl [AUTHOR_DIR] [PHASE]

AUTHOR_DIR defaults to ../HypaNet-v03-author, PHASE to v03-phase-a. Extends scripts/m2_audit_session.py:

- any file path (absolute, or relative that resolves outside AUTHOR_DIR) or command that refers
  outside AUTHOR_DIR is a violation, as is any mention of the repo, harness scratch, Claude data or
  another HypaNet* directory;
- every Bash command must start with ``./py `` and be a single command (no ``;``, ``&&``, ``||``,
  ``|``, redirection, backticks, ``$(`` or newlines); ``./py -c`` is refused;
- Write / Edit / NotebookEdit only inside ``output/<PHASE>/``;
- scripts run with ``./py`` must lie inside ``output/`` (``-m py_compile`` / ``-m <module>`` excepted);
- every ``.py`` file under ``AUTHOR_DIR/output/`` and every executed script is scanned statically with
  ``agent_loop_lab.v03.source_rules.author_script_findings`` (forbidden imports and RNG APIs);
- refused tool calls (permission denials), the models used and the launch cost are recorded.

Writes LOG.audit.json next to the log and exits 1 if any executed violation (or file finding) exists.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO / "src"))

from agent_loop_lab.v03.source_rules import author_script_findings  # noqa: E402

DEFAULT_AUTHOR_DIR = REPO.parent / "HypaNet-v03-author"
PHASES = ("v03-phase-a", "v03-phase-b")
PATH_KEYS = ("file_path", "path", "notebook_path", "pattern", "command", "glob")
FORBIDDEN_MARKERS = (
    str(REPO) + "/", str(REPO) + '"', "/private/tmp/claude-", "/tmp/claude-", "/.claude/", "~/.claude",
    "HypaNet/src", "HypaNet/datasets", "HypaNet-m2-author", "HypaNet-m3", "weight_program",
)
COMPOUND_RE = re.compile(r"(;|&&|\|\||\||>|<|`|\$\(|\n)")


def tool_calls(log: Path):
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        msg = ev.get("message") or {}
        for block in (msg.get("content") or []) if isinstance(msg, dict) else []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                yield ev, block
        if ev.get("type") == "result":
            yield ev, None


def _resolve(target: str, author_dir: Path) -> str:
    p = target if os.path.isabs(target) else os.path.join(str(author_dir), target)
    return os.path.normpath(p)


def _inside(path: str, root: Path) -> bool:
    r = str(root).rstrip("/")
    return path == r or path.startswith(r + "/")


def audit(log: Path, author_dir: Path, phase: str) -> dict:
    author_dir = author_dir.resolve() if author_dir.exists() else Path(os.path.normpath(str(author_dir)))
    out_dir = author_dir / "output"
    phase_dir = out_dir / phase
    violations: list[str] = []
    calls: list[dict] = []
    models: set[str] = set()
    denials: list = []
    cost = 0.0
    executed_scripts: set[str] = set()
    events = list(tool_calls(log))
    denied_ids = {d.get("tool_use_id") for ev, b in events if b is None for d in ev.get("permission_denials") or []}
    for ev, block in events:
        if block is None:
            denials.extend(ev.get("permission_denials") or [])
            if isinstance(ev.get("total_cost_usd"), (int, float)):
                cost += float(ev["total_cost_usd"])
            continue
        tag = "[refused] " if block.get("id") in denied_ids else "[EXECUTED] "
        model = (ev.get("message") or {}).get("model")
        if model:
            models.add(model)
        name, inp = block.get("name"), block.get("input") or {}
        text = " ".join(str(inp.get(k, "")) for k in PATH_KEYS)
        calls.append({"tool": name, "input": {k: inp[k] for k in PATH_KEYS if k in inp}})
        for marker in FORBIDDEN_MARKERS:
            if marker in text:
                violations.append(tag + f"{name}: refers to {marker!r}: {text[:200]}")
        for m in re.findall(r"(?:^|(?<=[\s=\"'(]))(/[^\s'\";|&)]+)", text):
            if not _inside(os.path.normpath(m), author_dir) and not m.startswith(("/dev/null",)):
                violations.append(tag + f"{name}: absolute path outside the author dir: {m}")
        for key in ("file_path", "path", "notebook_path"):
            t = inp.get(key)
            if isinstance(t, str) and t and not _inside(_resolve(t, author_dir), author_dir):
                violations.append(tag + f"{name}: path resolves outside the author dir: {t}")
        target = str(inp.get("file_path") or inp.get("notebook_path") or "")
        if name in ("Write", "Edit", "NotebookEdit", "MultiEdit"):
            if not _inside(_resolve(target, author_dir), phase_dir):
                violations.append(tag + f"{name}: write outside output/{phase}/: {target}")
        if name == "Bash":
            cmd = str(inp.get("command", ""))
            stripped = cmd.strip()
            if not stripped.startswith("./py "):
                violations.append(tag + f"Bash command does not start with './py ': {cmd[:200]}")
            elif COMPOUND_RE.search(stripped):
                violations.append(tag + f"Bash compound command or redirection: {cmd[:200]}")
            else:
                argv = stripped.split()[1:]
                if argv and argv[0] == "-c":
                    violations.append(tag + f"Bash ./py -c (inline code is not allowed): {cmd[:200]}")
                elif argv and argv[0] != "-m":
                    script = _resolve(argv[0], author_dir)
                    if not _inside(script, out_dir):
                        violations.append(tag + f"script outside output/: {argv[0]}")
                    if tag == "[EXECUTED] ":
                        executed_scripts.add(script)
                elif len(argv) >= 3 and argv[0] == "-m" and argv[1] == "py_compile":
                    pass
        elif name not in ("Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit", "MultiEdit", "TodoWrite"):
            violations.append(tag + f"tool {name!r} is outside the author kit")
    file_findings: list[str] = []
    scanned: list[str] = []
    paths = set(executed_scripts)
    if out_dir.is_dir():
        paths |= {str(p) for p in out_dir.rglob("*.py")}
    for p in sorted(paths):
        if os.path.isfile(p) and _inside(p, out_dir):
            scanned.append(os.path.relpath(p, author_dir))
            file_findings += [f"{os.path.relpath(p, author_dir)}: {f}" for f in author_script_findings(p)]
        elif os.path.isfile(p):
            violations.append(f"[EXECUTED] script outside output/ was run: {p}")
    executed = [v for v in violations if v.startswith("[EXECUTED]")]
    return {
        "phase": phase,
        "log": str(log),
        "author_dir": str(author_dir),
        "models": sorted(models),
        "cost_usd": round(cost, 6),
        "tool_calls": len(calls),
        "permission_denials": denials,
        "violations": violations,
        "executed_violations": len(executed),
        "scanned_files": scanned,
        "file_findings": file_findings,
        "calls": calls,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    log = Path(argv[1])
    author_dir = Path(argv[2]) if len(argv) > 2 else DEFAULT_AUTHOR_DIR
    phase = argv[3] if len(argv) > 3 else "v03-phase-a"
    if phase not in PHASES:
        print(f"unknown phase {phase!r}; expected one of {PHASES}", file=sys.stderr)
        return 2
    report = audit(log, author_dir, phase)
    out = log.with_suffix(".audit.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"audit: {report['tool_calls']} tool calls, models {report['models']}, cost ${report['cost_usd']}, "
        f"{len(report['permission_denials'])} denials, {len(report['violations'])} violations, "
        f"{len(report['file_findings'])} file findings -> {out}"
    )
    for v in report["violations"]:
        print("  VIOLATION:", v)
    for f in report["file_findings"]:
        print("  FILE FINDING:", f)
    print(f"  executed violations: {report['executed_violations']} (refused attempts are listed but were blocked)")
    return 1 if report["executed_violations"] or report["file_findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
