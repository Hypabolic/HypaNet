"""Export the v0.4 task files for the Qwen environment (plan v0.4, Task interface).

Usage:
    uv run python scripts/v04_export_tasks.py [--splits dev,val] [--out datasets/v0_4]

Runs in the frozen v0.3 environment. Writes OUT/{split}.tasks.jsonl.gz (rendered blocks and window
positions, no labels) and OUT/{split}.labels.npz (window rows), then OUT/EXPORT_MANIFEST.json.
Only dev and val are allowed; v0.4 exports no sealed split (they stay sealed, PREREGISTRATION §3.4).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_loop_lab.v04.export import V04_DATA, check_split, export_split  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="dev,val")
    ap.add_argument("--out", type=Path, default=V04_DATA)
    args = ap.parse_args(argv)
    splits = [s for s in args.splits.split(",") if s]
    for s in splits:
        check_split(s)
    man_path = args.out / "EXPORT_MANIFEST.json"
    manifest = json.loads(man_path.read_text()) if man_path.is_file() else {"splits": {}}
    for s in splits:
        t0 = time.monotonic()
        entry = export_split(s, out_dir=args.out)
        manifest["splits"][s] = entry
        print(f"{s}: {entry['sessions']} sessions, {entry['windows']} windows "
              f"({entry['scored_windows']} scored) in {time.monotonic() - t0:.1f}s", file=sys.stderr)
    man_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
