"""Score Qwen readout columns on val with the unchanged v0.3 metrics (``agent_loop_lab.v04.scoring``).

Usage:
    uv run python scripts/v04_score_validation.py --arm qwen0 --column "Qwen-0=PATH.npz" [--column ...]
        [--v03 experiments/v0_4/columns/v03.val.npz] [--subset FILE] [--diff "MINUEND,SUBTRAHEND" ...]
        [--group "NAME=COLUMN,COLUMN,..."] [--report-dir DIR] [--force]

Full val columns write artifacts/reports/v04_val_<arm>.json. Columns read out on the quick-look
subset (``hypaqwen.run_readout --sessions datasets/v0_4/VAL_QUICK.txt``) write
artifacts/reports/v04_val_quick_<arm>.json; their subset must be datasets/v0_4/VAL_QUICK.txt by
sha256 (another subset file, given with --subset, is accepted only with --report-dir, as
v04_val_subset_<arm>.json). Every Qwen column of one report must cover the same sessions.
With --report-dir DIR the report goes to DIR and is unofficial. Refuses to overwrite an official
report without --force.

The two constant priors are always added; --v03 adds the v0.3 columns (scripts/v04_v03_val_columns.py).
Paired macro-F1 differences: each --diff "A,B" is A - B over the shared bootstrap draws; without
--diff, every Qwen column minus Qwen-0 and Qwen-0 minus the dev-majority prior. The differences
are printed to stderr. Harness only: these numbers are never shown to a weight author.
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


def _diffs(items: list[str]) -> dict[str, tuple[str, str]] | None:
    if not items:
        return None
    out = {}
    for it in items:
        a, sep, b = it.partition(",")
        if not sep or not a.strip() or not b.strip():
            raise SystemExit(f"--diff must be MINUEND,SUBTRAHEND, got {it!r}")
        out[scoring.pair_name(a.strip(), b.strip())] = (a.strip(), b.strip())
    return out


def report_name(arm: str, sub: dict | None) -> str:
    if sub is None:
        return f"v04_val_{arm}.json"
    return f"v04_val_quick_{arm}.json" if sub["val_quick"] else f"v04_val_subset_{arm}.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--column", action="append", required=True)
    ap.add_argument("--v03", type=Path, default=None, help="v0.3 val columns (scripts/v04_v03_val_columns.py)")
    ap.add_argument("--subset", type=Path, default=None, help="subset file (default datasets/v0_4/VAL_QUICK.txt)")
    ap.add_argument("--diff", action="append", default=[], help="MINUEND,SUBTRAHEND (repeatable)")
    ap.add_argument("--group", action="append", default=[])
    ap.add_argument("--report-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    official = a.report_dir is None
    out_dir = OFFICIAL_DIR if official else a.report_dir
    if not official and out_dir.resolve() == OFFICIAL_DIR.resolve():
        raise SystemExit("--report-dir may not be artifacts/reports (that is the official location)")
    columns = {k: Path(v) for k, v in _pairs(a.column, "column").items()}
    try:
        sub = scoring.coverage({k: scoring.column_meta(p) for k, p in columns.items()}, a.subset)
    except scoring.ScoringError as exc:
        raise SystemExit(f"refusing: {exc}") from None
    if official and sub is not None and not sub["val_quick"]:
        raise SystemExit("refusing: an official subset report must be on datasets/v0_4/VAL_QUICK.txt; use --report-dir")
    out = out_dir / report_name(a.arm, sub)
    if official and out.exists() and not a.force:
        raise SystemExit(f"{out} exists; pass --force to replace an official report")
    groups = {k: [c.strip() for c in v.split(",")] for k, v in _pairs(a.group, "group").items()}
    rep = scoring.report(a.arm, columns, groups or None, v03_columns=a.v03, subset_file=a.subset,
                         pairs=_diffs(a.diff), official=official)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out} ({rep['coverage']})", file=sys.stderr)
    for pop in ("excluding_disclosed", "all_val"):
        for name, d in rep[pop].get("paired_macro_f1", {}).items():
            ci = "n/a" if d["ci95"] is None else f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
            val = "n/a" if d["value"] is None else f"{d['value']:+.4f}"
            print(f"  {pop:20s} {name:40s} macro-F1 diff {val}  95% CI {ci}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
