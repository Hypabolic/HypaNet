"""Spec v0.3 §7.3 rendering audits, on DEV ONLY.

Usage:
    uv run python scripts/v03_audits.py [--v03 datasets/v0_3] [--v02 datasets/v0_2] [--out artifacts/reports/v03_dev_audits.json]

Renders every dev session, reads the dev oracle records (latent effects, window labels), and writes
the audit report. Never reads or renders val, test, counterfactual or OOD.
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

from agent_loop_lab.v03.audits import run_dev_audits  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v03", type=Path, default=REPO_ROOT / "datasets" / "v0_3")
    ap.add_argument("--v02", type=Path, default=REPO_ROOT / "datasets" / "v0_2")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "reports" / "v03_dev_audits.json")
    args = ap.parse_args(argv)
    t0 = time.monotonic()

    def say(msg: str) -> None:
        print(f"[{time.monotonic() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)

    rep = run_dev_audits(args.v03, args.v02, progress=say)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    rv, mk, hc, ln, lk = (rep["revisit_evidence"], rep["marker_audit"], rep["hashed_bucket_collisions"],
                          rep["length"], rep["leakage_scan"])
    print(f"v0.3 dev audits: {rep['sessions']} sessions, {rep['events']} events -> {args.out}")
    print(f"(1) revisit evidence: {rv['reversion_events_with_evidence']}/{rv['reversion_events']} REVERSION events "
          f"= {rv['fraction_reversion_events_with_evidence']}; OSCILLATING windows all-evidenced "
          f"{rv['oscillating_windows_all_reversions_evidenced']}/{rv['oscillating_windows_with_reversions']} = "
          f"{rv['fraction_oscillating_windows_all_evidenced']} strict in-window, "
          f"{rv['oscillating_windows_all_reversions_event_evidenced']}/{rv['oscillating_windows_with_reversions']} = "
          f"{rv['fraction_oscillating_windows_all_event_evidenced']} per-event windows "
          f"({rv['oscillating_windows']} OSCILLATING windows)")
    print(f"(2) marker audit: n={mk['n']} pos={mk['positive_rate']} macro-F1@0.5={mk['macro_f1_threshold_0_5']} "
          f"macro-F1@base-rate={mk['macro_f1_threshold_base_rate']} prior={mk['macro_f1_constant_prior']} "
          f"AUC={mk['auc']}")
    for name in ("ablation_without_tool", "ablation_tool_only"):
        ab = mk[name]
        print(f"    {name}: macro-F1@0.5={ab['macro_f1_threshold_0_5']} @base-rate={ab['macro_f1_threshold_base_rate']} "
              f"AUC={ab['auc']}")
    print(f"(3) bucket collisions: {hc['sessions_with_collision']} sessions ({hc['fraction_sessions_with_collision']}), "
          f"{hc['colliding_word_pairs']} pairs, word rate {hc['word_collision_rate']}")
    print(f"(4) tokens/session {ln['tokens_per_session']}; tokens/block {ln['tokens_per_block_all']}")
    for fam, d in ln["tokens_per_block_by_family"].items():
        print(f"      {fam:12s} {d}")
    print(f"(5) leakage: clean={lk['clean']} markers={lk['marker_counts']} non-grammar body lines="
          f"{lk['non_grammar_body_lines']} file-text lines={lk['file_text_lines_found_in_blocks']}")
    print(f"    reserved field words outside a field position: {rep['reserved_field_words_outside_field_position']}")
    say("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
