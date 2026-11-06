"""Audit one v0.5 run: every turn transcript, failed attempt, launch record and the session snapshot.

    uv run python scripts/v05_audit_session.py RUN [--root DIR] [--allow-cc-recovery]

RUN is r1, r2 or r3; DIR defaults to experiments/v0_5. See agent_loop_lab.v05.audit for every rule: the init
event of each turn (tools, MCP servers, slash commands, skills, plugins, model, cwd), no tool use anywhere,
one API message per turn, PROMPT.txt as the first user message and exactly "Continue." for every later one
(launch records and the session file), one session id, the model of every message and request, the system
blocks and the context Claude Code added, stop reasons (max_tokens = cut off), END OF WEIGHTS, usage and cost.

Writes DIR/logs/RUN.audit.json and exits 1 on any violation. --allow-cc-recovery turns Claude Code's own
continuation of a cut reply (its fixed "output token limit" message) from a violation into a recorded
deviation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO / "src"))

from agent_loop_lab.v05.audit import audit_run  # noqa: E402
from agent_loop_lab.v05.session import RUNS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", choices=RUNS)
    ap.add_argument("--root", type=Path, default=REPO / "experiments" / "v0_5")
    ap.add_argument("--allow-cc-recovery", action="store_true")
    args = ap.parse_args(argv)
    report = audit_run(args.root, args.run, allow_cc_recovery=args.allow_cc_recovery)
    out = args.root / "logs" / f"{args.run}.audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    sf = report["session_file"]
    print(f"audit {args.run}: status {report['status']}, {report['turns_accepted']} turns "
          f"({report['continuation_turns']} continuations), {len(report['failed_attempts'])} failed attempts, "
          f"END OF WEIGHTS {'given' if report['end_of_weights_given'] else 'NOT given'}, "
          f"cut off: {report['cut_off_turns']}, cost US${report['cost']['run_usd']} "
          f"(all runs US${report['cost']['all_runs']['total_usd']} of US${report['cost']['all_runs']['cap_usd']}) -> {out}")
    print(f"  usage: {report['usage_total']}")
    print(f"  context added by Claude Code: system blocks {sf['system_blocks']}, attachments {sf['attachments']}")
    for d in report["deviations"]:
        print("  DEVIATION:", d)
    for v in report["violations"]:
        print("  VIOLATION:", v)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
