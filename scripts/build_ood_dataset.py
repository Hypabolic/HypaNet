"""Build the OOD holdout set (spec v0.2 §7.3) and its audit report.

    uv run python scripts/build_ood_dataset.py [--out datasets/v0_2] [--sessions-per-family 200] [--reports artifacts/reports]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_loop_lab.datasets.build import DEFAULT_MASTER_SEED
from agent_loop_lab.datasets.ood import build_ood, iter_ood_sessions

REPO = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "datasets" / "v0_2"))
    ap.add_argument("--sessions-per-family", type=int, default=200)
    ap.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    ap.add_argument("--reports", default=str(REPO / "artifacts" / "reports"))
    a = ap.parse_args(argv)
    m = build_ood(a.out, master_seed=a.master_seed, sessions_per_family=a.sessions_per_family)
    print(f"OOD: {m['sessions']} sessions, labels {m['window_label_counts']}")
    for k, v in m["files"].items():
        print(f"  {k}  {v}")
    if a.reports:
        from agent_loop_lab.eval import audit_dataset, to_json

        stats, collisions = audit_dataset({"ood": iter_ood_sessions(a.out)}, audit_split="ood")
        rd = Path(a.reports)
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "m1_ood_stats.json").write_text(to_json(stats), encoding="utf-8")
        (rd / "m1_ood_collisions.json").write_text(to_json(collisions), encoding="utf-8")
        print(f"  reports -> {rd}/m1_ood_stats.json, m1_ood_collisions.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
