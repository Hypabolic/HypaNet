"""Run (or resume) one v0.5 author conversation, then audit it and read its replies into weights.

    uv run python scripts/v05_run.py RUN [--resume] [--no-post]

RUN is r1, r2 or r3. Without --resume it refuses if the run already has a transcript or a run record; with
--resume it continues an interrupted run at the next turn (a turn whose transcript exists is never sent again).
See agent_loop_lab.v05.runner for the loop, its stop conditions and its files, and agent_loop_lab.v05.session
for the exact claude command and environment. V05_BARE=1 adds --bare (needs ANTHROPIC_API_KEY or
CLAUDE_CODE_OAUTH_TOKEN); it is fixed when the run is created. V05_TURN_WALL_SECONDS (default 21600) is the
per-turn watchdog.

After the loop stops (any status): scripts/v05_audit_session.py RUN. Only when the status is final
(finished, turn_cap, context_limit, cost_cap: agent_loop_lab.v05.runner.FINAL) also
scripts/v05_apply_reply.py experiments/v0_5/runs/RUN, which writes the artefact that gets scored and refuses to
overwrite it; a resumable stop (technical_stop, usage_limit, cost_unresolved) writes no artefact, so a later
--resume can still finish the run. Exit code: 0 if the run finished and the audit is clean, 1 if the audit found
violations, 2 on a refusal, 3 if the run stopped without END OF WEIGHTS, 4 if it stopped resumably.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO / "src"))

from agent_loop_lab.v05.runner import DEFAULT_ROOT, FINAL, TURN_WALL_SECONDS, Refusal, run  # noqa: E402
from agent_loop_lab.v05.session import RUNS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", choices=RUNS)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-post", action="store_true", help="skip the audit and the reply reader")
    args = ap.parse_args(argv)
    bare = os.environ.get("V05_BARE")
    wall = int(os.environ.get("V05_TURN_WALL_SECONDS") or TURN_WALL_SECONDS)
    try:
        rec = run(args.run, resume=args.resume, root=DEFAULT_ROOT, bare=None if bare is None else bare == "1",
                  wall=wall, log=lambda s: print(s, flush=True))
    except Refusal as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    print(f"{args.run}: {rec['status']} after {len(rec['turns'])} turns; cost so far {rec.get('cost')}")
    final = rec["status"] in FINAL
    code = 0 if rec["status"] == "finished" else (3 if final else 4)
    if args.no_post:
        return code
    audit_rc = subprocess.run([sys.executable, str(REPO / "scripts" / "v05_audit_session.py"), args.run],
                              cwd=REPO).returncode
    apply = REPO / "scripts" / "v05_apply_reply.py"
    if not final:
        print(f"{args.run}: stopped resumably ({rec['status']}); no artefact written. Continue it with --resume.",
              file=sys.stderr)
    elif apply.is_file():
        rc = subprocess.run([sys.executable, str(apply), str(DEFAULT_ROOT / "runs" / args.run)], cwd=REPO).returncode
        if rc:
            print(f"scripts/v05_apply_reply.py exited {rc}", file=sys.stderr)
    else:
        print(f"scripts/v05_apply_reply.py not found; read the replies with it later "
              f"({DEFAULT_ROOT / 'runs' / args.run / 'replies' / 'INDEX.json'})", file=sys.stderr)
    if audit_rc:
        print("AUDIT FOUND VIOLATIONS (see experiments/v0_5/logs/%s.audit.json)" % args.run, file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
