"""Pre-launch smoke test of the v0.5 harness: the real PROMPT.txt, turn 001 plus one "Continue." resume, a cheap model.

    uv run python scripts/v05_smoke.py --root DIR --yes [--model claude-haiku-5-5] [--turn-budget-usd 1.00]

Runs agent_loop_lab.v05.runner.run exactly as scripts/v05_run.py does (same command, environment, working
directory rule, transcripts and session snapshot), except: the root is DIR (outside the repo; PROMPT.txt and
PROMPT.sha256 are copied there), the model is a cheap one, the continuation cap is 1 and each turn's
--max-budget-usd is small. It then audits the run (agent_loop_lab.v05.audit, Claude Code's recovery messages
allowed and listed) and prints what reached the model on each turn: the attachment types, the system blocks,
the user-role reminders and anything the audit flags. Use it before r1 to see the --resume path and the 73 KB
prompt go through Claude Code once. It spends money (a few US cents to about US$2 with haiku), so it refuses
without --yes. The v0.5 cost cap does not count it (its logs are under DIR).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO / "src"))

from agent_loop_lab.v05.audit import audit_run  # noqa: E402
from agent_loop_lab.v05.runner import DEFAULT_ROOT, Refusal, run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="a new directory outside the repo")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--turn-budget-usd", type=float, default=1.0)
    ap.add_argument("--yes", action="store_true", help="really launch claude (spends money)")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    if root == DEFAULT_ROOT.resolve() or str(root).startswith(str(REPO) + "/"):
        ap.error("--root must be outside the repo")
    if root.exists() and any(root.iterdir()):
        ap.error(f"{root} is not empty")
    if not args.yes:
        print("refusing: pass --yes to launch claude (this spends money)", file=sys.stderr)
        return 2
    root.mkdir(parents=True, exist_ok=True)
    for name in ("PROMPT.txt", "PROMPT.sha256"):
        shutil.copyfile(DEFAULT_ROOT / name, root / name)
    try:
        rec = run("r1", root=root, model=args.model, continuation_cap=1, turn_budget_usd=args.turn_budget_usd,
                  log=lambda s: print(s, flush=True))
    except Refusal as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    rep = audit_run(root, "r1", allow_cc_recovery=True)
    (root / "logs" / "r1.audit.json").write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    sf = rep["session_file"]
    print(f"status {rec['status']}, {rep['turns_accepted']} turns, cost US${rep['cost']['run_usd']}")
    print("attachments:", json.dumps(sf["attachments"], sort_keys=True))
    print("other (not rendered) attachments:", sf["other_attachments"])
    print("system blocks:", json.dumps(sf["system_blocks"]))
    print("user-role reminders:", len(sf["user_reminders"]))
    print("request params:", json.dumps(sf["request_params"][:4]))
    for d in rep["deviations"]:
        print("deviation:", d)
    for v in rep["violations"]:
        print("VIOLATION:", v)
    print(f"audit {'ok' if rep['ok'] else 'FAILED'}: {root / 'logs' / 'r1.audit.json'}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
