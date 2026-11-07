"""F26 (post Figure 14): "The prompt that always said stuck", the v0.4 label-free prompt choice.

Run from the repo root:  uv run python scripts/blog/figures/f26.py

Reads only the two committed prompt-choice reports (the first run, which chose by median label-word mass alone,
and Revision 1, which multiplies that mass by the normalised entropy of the predicted-class shares). Both reports
were computed on 100 dev sessions with no labels. No dataset, evaluator or scorer is opened here.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F26.json"
GENERATED_BY = "scripts/blog/figures/f26.py"

FIRST = ROOT / "artifacts" / "reports" / "v04_prompt_choice.json"
REV1 = ROOT / "artifacts" / "reports" / "v04_prompt_choice_revision1.json"
README = ROOT / "experiments" / "v0_4" / "prompt_choice" / "README.md"
SOURCES = [FIRST, REV1, README]
SOURCE_COMMIT = "a616b93"
CLASSES = ("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING")
DP = 5


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def r(x: float) -> float:
    return round(float(x), DP)


def norm_entropy(shares: dict[str, float]) -> float:
    ps = [shares[c] for c in CLASSES if shares[c] > 0]
    return -sum(p * math.log(p) for p in ps) / math.log(len(CLASSES))


def build() -> dict:
    first = json.loads(FIRST.read_text())
    rev = json.loads(REV1.read_text())
    rev_by = {c["name"]: c for c in rev["candidates"]}

    candidates = []
    for c in first["candidates"]:
        name = c["name"]
        rv = rev_by[name]
        # The revision reads its two factors from the first run's report; check they agree.
        if abs(rv["median_mass"] - c["mass"]["median"]) > 1e-9:
            raise SystemExit(f"F26: median mass mismatch for {name}")
        ent = norm_entropy(c["predicted_share"])
        if abs(ent - rv["share_entropy"]) > 2e-3:  # shares are rounded to 4 dp in the first report
            raise SystemExit(f"F26: entropy mismatch for {name}: {ent} vs {rv['share_entropy']}")
        word_set, cue, style = name.split("-")
        candidates.append({
            "name": name,
            "word_set": word_set,
            "answer_cue": c["answer_cue"],
            "style": style,
            "label_words": c["label_words"],
            "prefix_tokens": c["prefix_tokens"],
            "windows": c["windows"],
            "median_mass": r(c["mass"]["median"]),
            "mass_p10": r(c["mass"]["p10"]),
            "mass_p90": r(c["mass"]["p90"]),
            "share_entropy": r(rv["share_entropy"]),
            "score": r(rv["score"]),
            "predicted_share": {k: r(v) for k, v in c["predicted_share"].items()},
        })

    # The first rule: highest median mass, ties to the earlier candidate.
    first_choice = max(candidates, key=lambda c: (c["median_mass"], -candidates.index(c)))["name"]
    if first_choice != first["chosen"]:
        raise SystemExit(f"F26: first-run choice {first_choice} != report {first['chosen']}")
    ranked = sorted(candidates, key=lambda c: -c["score"])
    if ranked[0]["name"] != rev["chosen"]:
        raise SystemExit("F26: revision choice mismatch")
    windows = {c["windows"] for c in candidates}
    if len(windows) != 1:
        raise SystemExit("F26: candidates scored different window counts")

    return {
        "_provenance": {
            "source": [rel(p) for p in SOURCES],
            "sha256": {rel(p): sha256(p) for p in SOURCES},
            "source_commit": SOURCE_COMMIT,
            "git_commit": git_head(),
            "generated_by": GENERATED_BY,
            "notes": [
                "split: dev (100 sessions with the smallest sha256('v04-prompt-choice:{session_id}'), every window); "
                "no labels, gate results or families were read by either rule",
                "median_mass: median over windows of the probability Qwen puts on the four label words together",
                "share_entropy: entropy of the predicted-class shares divided by log 4 (0 = one class always, 1 = uniform)",
                "score (Revision 1) = median_mass x share_entropy",
            ],
        },
        "figure": "F26",
        "split": "dev",
        "sessions": first["sessions"],
        "windows": windows.pop(),
        "labels_used": 0,
        "first_rule": "highest median label-word mass",
        "first_choice": first["chosen"],
        "revised_rule": "highest median label-word mass x normalised entropy of predicted-class shares",
        "revised_choice": rev["chosen"],
        "runner_up": ranked[1]["name"],
        "candidates": candidates,
    }


def main() -> None:
    data = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {rel(OUT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
