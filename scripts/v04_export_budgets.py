"""Write the v0.4 labelled-budget session lists, ``datasets/v0_4/BUDGETS.json`` (PREREGISTRATION.md §4, §5, §7).

Usage:
    uv run python scripts/v04_export_budgets.py [--out datasets/v0_4/BUDGETS.json] [--data-dir datasets/v0_4]
    uv run python scripts/v04_export_budgets.py --check

Runs in the frozen v0.3 environment (``agent_loop_lab.v04.budgets``). Reads the dev and val ids of
``datasets/v0_2/splits.json`` and the val window rows of ``DATA_DIR/val.labels.npz`` (for the
disclosure rule only); writes the probe/LoRA budget subsets, the val ES set, the 200 disclosed
windows and the WSP-F holdout, each with its sha256. No label is written. The build reproduces v0.3
(Phase C logs, DISCLOSURE_PRIVATE.json) or refuses. ``--check`` rebuilds and compares byte for byte.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_loop_lab.v04 import budgets  # noqa: E402
from agent_loop_lab.v04.export import V04_DATA  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=budgets.BUDGETS_PATH)
    ap.add_argument("--data-dir", type=Path, default=V04_DATA, help="where val.labels.npz is (default datasets/v0_4)")
    ap.add_argument("--check", action="store_true", help="rebuild and compare with --out; write nothing")
    args = ap.parse_args(argv)
    if args.check:
        problems = budgets.check(args.out, data_dir=args.data_dir)
        for p in problems:
            print(f"BUDGETS: {p}", file=sys.stderr)
        print("BUDGETS.json: ok" if not problems else "BUDGETS.json: FAILED", file=sys.stderr)
        return 1 if problems else 0
    doc = budgets.build(data_dir=args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(budgets.render(doc), encoding="utf-8")
    tmp.replace(args.out)
    sizes = {**{k: len(v) for k, v in doc["budgets"].items()},
             **{k: len(doc[k]) for k in ("es_val", "disclosed_200w", "wspf_holdout")}}
    print(f"{args.out}: {sizes}", file=sys.stderr)
    print(f"sha256 {budgets.canonical_sha256(doc)} (canonical JSON of the whole document)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
