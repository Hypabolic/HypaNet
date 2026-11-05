"""Audit a v0.4 weight-author session transcript (docs/V04_WSP_KERNEL.md §7 "Audit").

    uv run python scripts/v04_audit_session.py LOG.stream.jsonl [AUTHOR_DIR] [PHASE] [--logs DIR]
                                               [--budget N] [--request-cap N]

AUTHOR_DIR defaults to ../HypaNet-v04-author, PHASE to wsp-a, DIR (the model server's ledger and kernel log)
to experiments/v0_4/logs. See agent_loop_lab.v04.audit for every rule: forbidden markers and paths, the
PreToolUse hook's Bash rule (the audit loads the hook itself), Write/Edit only in output/<PHASE>/, tools
outside the kit, harness paths shown to the session, tool and wrapper integrity, v0.3's author-script rules
plus v0.4's, the AST loop rule, and the ledger cross-check (budget, per-request cap, edit states per text set
for the adjudicator, refused requests, kernel ops and the weight-load quota).

Writes LOG.audit.json next to the log (the .stream.jsonl suffix becomes .stream.audit.json) and exits 1 if any
executed violation or file finding exists. Adjudication flags are printed but do not fail the audit.
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

from agent_loop_lab.v04.audit import PHASE_BUDGET, REQUEST_CAP, audit  # noqa: E402

DEFAULT_AUTHOR_DIR = REPO.parent / "HypaNet-v04-author"
PHASES = ("wsp-a", "wsp-b", "wsp-f")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("author_dir", type=Path, nargs="?", default=DEFAULT_AUTHOR_DIR)
    ap.add_argument("phase", nargs="?", default="wsp-a", choices=PHASES)
    ap.add_argument("--logs", type=Path, default=REPO / "experiments" / "v0_4" / "logs")
    ap.add_argument("--budget", type=int, default=PHASE_BUDGET)
    ap.add_argument("--request-cap", type=int, default=REQUEST_CAP)
    args = ap.parse_args(argv)
    report = audit(args.log, args.author_dir, args.phase, log_dir=args.logs, budget=args.budget,
                   request_cap=args.request_cap)
    out = args.log.with_suffix(".audit.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    led = report["ledger"] or {}
    print(f"audit: {sum(report['tool_calls'].values())} tool calls {report['tool_calls']}, models {report['models']}, "
          f"{len(report['permission_denials'])} denials, {len(report['violations'])} violations, "
          f"{len(report['file_findings'])} file findings, {len(report['flags'])} flags -> {out}")
    if led:
        print(f"  ledger: {led['forward_requests']} requests {led['status']}, {led['tokens_ok']} / {led['budget']} tokens, "
              f"{led['text_sets']} text sets, kernel ops {led['kernel_ops']}, {led['quota_tensors_loaded']} quota tensors "
              f"({led['quota_bytes_used']} bytes)")
    for v in report["violations"]:
        print("  VIOLATION:", v)
    for f in report["file_findings"]:
        print("  FILE FINDING:", f)
    for f in report["flags"]:
        print("  FLAG (adjudicator):", f)
    print(f"  executed violations: {report['executed_violations']} (refused attempts are listed but were blocked)")
    return 1 if report["executed_violations"] or report["file_findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
