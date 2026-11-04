"""Audit a weight-author session transcript (spec v0.2 §8, §9.2-9.3).

    uv run python scripts/m2_audit_session.py LOG.stream.jsonl AUTHOR_DIR PHASE

Scans every tool call in the stream-json transcript and reports:
- any file path or command that refers outside AUTHOR_DIR (repo, harness scratch, Claude project data);
- writes inside packet/ (or, in A-selfcheck, inside output/phase-a-strict/);
- Bash commands other than the phase's allowed ./py invocations (A-strict: py_compile and shape_check only);
- refused tool calls (permission denials) and the model actually used.
Writes LOG.audit.json next to the log and exits 1 if any violation was found.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FORBIDDEN_MARKERS = (str(REPO) + "/", str(REPO) + '"', "/private/tmp/claude-501", "/.claude/projects", "HypaNet/src", "HypaNet/datasets")
PATH_KEYS = ("file_path", "path", "notebook_path", "pattern", "command", "glob")


def tool_calls(log: Path):
    for line in log.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = ev.get("message") or {}
        for block in msg.get("content") or [] if isinstance(msg, dict) else []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                yield ev, block
        if ev.get("type") == "result":
            yield ev, None


def main(argv: list[str]) -> int:
    log, author_dir, phase = Path(argv[1]), Path(argv[2]).resolve(), argv[3]
    violations, calls, models, denials = [], [], set(), []
    events = list(tool_calls(log))
    denied_ids = {d.get("tool_use_id") for ev, b in events if b is None for d in ev.get("permission_denials") or []}
    for ev, block in events:
        if block is None:
            for d in ev.get("permission_denials") or []:
                denials.append(d)
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
            p = Path(m)
            if p.is_absolute() and not str(p).startswith(str(author_dir)) and not m.startswith(("/dev/null", "/usr/bin", "/bin")):
                violations.append(tag + f"{name}: absolute path outside author dir: {m}")
        target = str(inp.get("file_path", ""))
        if name in ("Write", "Edit", "NotebookEdit") and "/packet/" in target:
            violations.append(tag + f"{name}: write inside packet: {target}")
        if phase == "phase-a-selfcheck" and name in ("Write", "Edit") and "/output/phase-a-strict/" in target:
            violations.append(tag + f"{name}: edit of archived A-strict deliverable: {target}")
        if name == "Bash":
            cmd = str(inp.get("command", "")).strip()
            if phase == "phase-a-strict":
                ok = cmd.startswith("./py -m py_compile ") or cmd == "./py tools/shape_check.py output/phase-a-strict"
            else:
                ok = cmd.startswith("./py ")
            if not ok:
                violations.append(tag + f"Bash outside phase allowance: {cmd[:200]}")
    report = {
        "phase": phase,
        "log": str(log),
        "models": sorted(models),
        "tool_calls": len(calls),
        "permission_denials": denials,
        "violations": violations,
        "calls": calls,
    }
    out = log.with_suffix(".audit.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"audit: {len(calls)} tool calls, models {sorted(models)}, {len(denials)} denials, {len(violations)} violations -> {out}")
    for v in violations:
        print("  VIOLATION:", v)
    executed = [v for v in violations if v.startswith("[EXECUTED]")]
    report["executed_violations"] = len(executed)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"  executed violations: {len(executed)} (refused attempts are listed but were blocked)")
    return 1 if executed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
