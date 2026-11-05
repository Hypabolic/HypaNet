"""Score Qwen readout columns on val with the unchanged v0.3 metrics (``agent_loop_lab.v04.scoring``).

Usage:
    uv run python scripts/v04_score_validation.py --arm qwen0 --column "Qwen-0=PATH.npz" [--column ...]
        [--group "LoRA-512=LoRA-512 s0,LoRA-512 s1,LoRA-512 s2"] [--report-dir DIR] [--force]

Writes artifacts/reports/v04_val_<arm>.json (official), or DIR/v04_val_<arm>.json with --report-dir
(unofficial). Refuses to overwrite an official report without --force. Harness only: these numbers
are never shown to a weight author.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_loop_lab.v04 import scoring  # noqa: E402

OFFICIAL_DIR = REPO_ROOT / "artifacts" / "reports"


def _pairs(items: list[str], what: str) -> dict[str, str]:
    out = {}
    for it in items:
        name, sep, val = it.partition("=")
        if not sep or not name or not val:
            raise SystemExit(f"--{what} must be NAME=VALUE, got {it!r}")
        out[name] = val
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--column", action="append", required=True)
    ap.add_argument("--group", action="append", default=[])
    ap.add_argument("--report-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    official = a.report_dir is None
    out_dir = OFFICIAL_DIR if official else a.report_dir
    if not official and out_dir.resolve() == OFFICIAL_DIR.resolve():
        raise SystemExit("--report-dir may not be artifacts/reports (that is the official location)")
    out = out_dir / f"v04_val_{a.arm}.json"
    if official and out.exists() and not a.force:
        raise SystemExit(f"{out} exists; pass --force to replace an official report")
    columns = {k: Path(v) for k, v in _pairs(a.column, "column").items()}
    groups = {k: [c.strip() for c in v.split(",")] for k, v in _pairs(a.group, "group").items()}
    rep = scoring.report(a.arm, columns, groups or None, official=official)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
