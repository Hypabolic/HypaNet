"""F19b (post Figure 23): "Structural rules travel; behavioural ones don't".

Run from the repo root:  uv run python scripts/blog/figures/f19b.py

For each of v0.5 B's "wasted effort" (INEFF) rules, the share of operations the rule flagged that the labeller
(the oracle's latent per-operation effect) also marks ineffective, on a fresh in-distribution batch and a fresh
OOD batch. Plus the side note: how often INEFF fires against the labeller, and what the 7,175 false STALLED
windows of the fresh OOD batch look like.

Reads the per-window diagnostic rows written by experiments/v0_5/ood_analysis/analyse.py (@ 369beae; gitignored,
regenerate with `uv run python experiments/v0_5/ood_analysis/analyse.py --per-family 100` for the fresh OOD
batch, master seed 20261107, and `--indist --per-family 40` for the fresh in-distribution batch, seed 20261108).
No sealed, held-out or fresh_test data is opened; no evaluator or scorer is run. Every number is checked against
experiments/v0_5/ood_analysis/FINDINGS.md (§2 and §3 @ 82cc5ec) before it is written.

Counting, as FINDINGS.md §3: an operation is the event at one position of one session (windows overlap, so each
operation is counted once). Only operations the network flags INEFF are attributed to rules. The labeller marks an
operation ineffective when its latent effect is REDUNDANT or REVERSION. EDR and g_REP are counted only where they
fire alone; g_SAME wherever it fires; operations with a reversion (REV) are excluded from the other rules' counts.
NOOP never fires in the fresh OOD batch.

Shown percentages are whole numbers, except one that would round up to 100% (one decimal). The recount matches
FINDINGS.md except g_REP in distribution: 79.48% here (shown 79%), printed as 80% in FINDINGS.md.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F19b.json"
GENERATED_BY = "scripts/blog/figures/f19b.py"

DIAG = ROOT / "experiments" / "v0_5" / "ood_analysis"
ROWS_OOD = DIAG / "rows.jsonl.gz"
ROWS_IN = DIAG / "rows_indist.jsonl.gz"
FINDINGS = DIAG / "FINDINGS.md"
ANALYSE = DIAG / "analyse.py"
REVISION = ROOT / "weight_program" / "v0_5" / "phase-b" / "REVISION.md"
SOURCES = [FINDINGS, ANALYSE, REVISION]
SOURCE_COMMIT = "82cc5ec"
ANALYSE_COMMIT = "369beae"
DP = 5

INEFFECTIVE = {"REDUNDANT", "REVERSION"}

# Rule definitions, in plain words (REVISION.md §3, FINDINGS.md §3).
RULES = [
    dict(id="REV", kind="structural", name="Undo",
         definition="a hunk that returns a file to an earlier state (kept from the no-tools draft)",
         counting="any"),
    dict(id="EDR", kind="behavioural", name="Edit after a repeat",
         definition="a file change within 3 operations after a repeated test result",
         counting="alone"),
    dict(id="g_REP", kind="behavioural", name="Repeated target",
         definition="a repeated read, search, plan or build target",
         counting="alone"),
    dict(id="g_SAME", kind="behavioural", name="Repeated error line",
         definition="a failure whose error line was seen in the last 15 operations",
         counting="any"),
]

# FINDINGS.md §2-§3, as printed there (percent). The export refuses to write if the rows disagree.
FINDINGS_PCT = {
    "REV": (99.5, 96), "EDR": (84, 28), "g_REP": (80, 36), "g_SAME": (62, 44),
    "ineff_net": (47, 57), "ineff_oracle": (38, 29),
}


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def r(x: float) -> float:
    return round(float(x), DP)


def load_rows(p: Path) -> list[dict]:
    if not p.exists():
        raise SystemExit(f"{rel(p)} is missing: regenerate it with experiments/v0_5/ood_analysis/analyse.py")
    with gzip.open(p, "rt") as fh:
        return [json.loads(line) for line in fh]


def operations(rows: list[dict]) -> dict[tuple[str, int], dict]:
    """Every operation once: windows overlap, so key each event by (session, position)."""
    ops: dict[tuple[str, int], dict] = {}
    for row in rows:
        n = len(row["events"])
        for k, ev in enumerate(row["events"]):
            ops[(row["sid"], row["pos"] - n + 1 + k)] = ev
    return ops


def rule_ops(ops: dict, rule: dict) -> list[dict]:
    """Operations the network flagged INEFF that this rule fired on (FINDINGS.md §3's attribution)."""
    rid = rule["id"]
    flagged = [e for e in ops.values() if e["net_ineff"]]
    if rid == "REV":
        return [e for e in flagged if "REV" in e["parts"]]
    if rule["counting"] == "alone":
        return [e for e in flagged if e["parts"] == [rid]]
    return [e for e in flagged if rid in e["parts"] and "REV" not in e["parts"]]


def agreement(ev: list[dict]) -> dict:
    n = len(ev)
    agree = sum(e["effect"] in INEFFECTIVE for e in ev)
    effects = Counter(e["effect"] for e in ev)
    return {
        "operations": n,
        "agree": agree,
        "share": r(agree / n) if n else None,
        "labeller_effects": dict(effects.most_common()),
    }


def shown_pct(share: float, findings: float, mismatches: list[str], what: str) -> str:
    """The rows' share as a percentage, checked against FINDINGS.md. A difference of up to one point (rounding)
    is allowed and recorded; anything larger stops the export. Whole numbers, except one that would read 100%."""
    s = f"{share * 100:.0f}"
    if s == "100" and share < 1:
        s = f"{share * 100:.1f}"
    if abs(share * 100 - findings) > 1.0:
        raise SystemExit(f"{what}: rows give {share * 100:.2f}% but FINDINGS.md says {findings}%")
    if float(s) != float(findings):
        mismatches.append(f"{what}: rows give {share * 100:.2f}% (shown {s}%); FINDINGS.md prints {findings}%")
    return s


def batch(rows: list[dict], name: str, seed: int, generator: str) -> dict:
    ops = operations(rows)
    n = len(ops)
    return {
        "name": name,
        "generator": generator,
        "master_seed": seed,
        "sessions": len({row["sid"] for row in rows}),
        "families": len({row["family"] for row in rows}),
        "windows": len(rows),
        "operations": n,
        "ineff_net": r(sum(e["net_ineff"] for e in ops.values()) / n),
        "ineff_oracle": r(sum(e["effect"] in INEFFECTIVE for e in ops.values()) / n),
        "_ops": ops,
    }


def build() -> dict:
    rin = load_rows(ROWS_IN)
    rood = load_rows(ROWS_OOD)
    b_in = batch(rin, "fresh in-distribution batch", 20261108, "standard generator (the original simulator)")
    b_ood = batch(rood, "fresh OOD batch", 20261107, "Codex's independently written generator (synth_ood)")

    mismatches: list[str] = []
    rules = []
    for rule in RULES:
        a_in = agreement(rule_ops(b_in["_ops"], rule))
        a_ood = agreement(rule_ops(b_ood["_ops"], rule))
        f_in, f_ood = FINDINGS_PCT[rule["id"]]
        rules.append({
            **rule,
            "indist": {**a_in, "shown": shown_pct(a_in["share"], f_in, mismatches, f"{rule['id']} in-distribution")},
            "ood": {**a_ood, "shown": shown_pct(a_ood["share"], f_ood, mismatches, f"{rule['id']} OOD")},
        })

    noop_in = agreement([e for e in b_in["_ops"].values() if e["net_ineff"] and e["parts"] == ["NOOP"]])
    noop_ood = len([e for e in b_ood["_ops"].values() if "NOOP" in e["parts"]])

    for key, b, idx in (("ineff_net", b_in, 0), ("ineff_oracle", b_in, 0), ("ineff_net", b_ood, 1), ("ineff_oracle", b_ood, 1)):
        b[f"{key}_shown"] = shown_pct(b[key], FINDINGS_PCT[key][idx], mismatches, f"{key} {b['name']}")

    # The fresh OOD batch's false STALLED windows (FINDINGS.md §2).
    fs = [row for row in rood if row["label"] == "PRODUCTIVE" and row["v05b"] == "STALLED"]
    productive = sum(row["label"] == "PRODUCTIVE" for row in rood)
    false_stalled = {
        "windows": len(fs),
        "productive_windows": productive,
        "rate": r(len(fs) / productive),
        "labeller_progress": r(sum(row["oracle"]["dp"] > 0 for row in fs) / len(fs)),
        "labeller_no_ineffective": r(sum(row["oracle"]["I"] == 0 for row in fs) / len(fs)),
        "labeller_stall_count_met": sum(row["oracle"]["I"] >= 5 and row["oracle"]["final_I"] for row in fs),
        "network_count_min": min(row["net_I"] for row in fs),
        "network_count_max": max(row["net_I"] for row in fs),
        "v03b_productive": r(sum(row["v03b"] == "PRODUCTIVE" for row in fs) / len(fs)),
    }
    checks = {"windows": 7175, "rate": 0.328, "labeller_progress": 0.92, "labeller_no_ineffective": 0.61}
    for k, want in checks.items():
        got = false_stalled[k]
        if (isinstance(want, int) and got != want) or (isinstance(want, float) and round(got, len(str(want)) - 2) != want):
            raise SystemExit(f"false STALLED {k}: rows give {got}, FINDINGS.md says {want}")

    for b in (b_in, b_ood):
        b.pop("_ops")

    return {
        "_provenance": {
            "source": [rel(p) for p in SOURCES],
            "sha256": {rel(p): sha256(p) for p in SOURCES + [ROWS_IN, ROWS_OOD]},
            "source_commit": SOURCE_COMMIT,
            "analyse_commit": ANALYSE_COMMIT,
            "git_commit": git_head(),
            "generated_by": GENERATED_BY,
            "notes": [
                "rows.jsonl.gz and rows_indist.jsonl.gz are gitignored; analyse.py @ 369beae regenerates them "
                "(fresh OOD batch: synth_ood, master seed 20261107, 100 sessions per family; fresh in-distribution "
                "batch: standard generator, seed 20261108, 40 per family). Their sha256 is recorded above.",
                "No sealed, held-out or fresh_test data was opened; no evaluator or scorer was run.",
                "agreement = share of the operations the network flagged INEFF, attributed to a rule, that the "
                "labeller marks ineffective (latent effect REDUNDANT or REVERSION); each operation counted once "
                "across overlapping windows",
                "EDR and g_REP counted where they fire alone; g_SAME wherever it fires; operations with REV excluded "
                "from the other rules (FINDINGS.md §3)",
                "every shown percentage is checked against FINDINGS.md §2-§3 before writing (within one point); "
                "differences beyond rounding are listed in findings_mismatches",
            ],
            "findings_mismatches": mismatches,
        },
        "figure": "F19b",
        "title": "Structural rules travel; behavioural ones don't",
        "network": "v0.5 B (the no-tools revision)",
        "batches": {"indist": b_in, "ood": b_ood},
        "rules": rules,
        "noop": {"indist": noop_in, "ood_operations": noop_ood},
        "false_stalled": false_stalled,
    }


def main() -> int:
    data = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
