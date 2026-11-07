"""F27 (post Figure 17): "Two looks, one revision", WSP-B's edit states and its two runs on the disclosed windows.

Run from the repo root:  uv run python scripts/blog/figures/f27.py

Reads only the committed WSP-B forward-pass ledger (which edit state each call ran under, and on what) and Fable's
committed revision notes (REVISION.md, EDIT.md) for the accuracies it reported on the 200 disclosed windows.
The ledger fixes the structure (two edit states on the disclosed windows; the final state only on Fable's own set R);
every accuracy is a checked quotation from REVISION.md. No evaluator, scorer or dataset is opened.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F27.json"
GENERATED_BY = "scripts/blog/figures/f27.py"

LEDGER = ROOT / "experiments" / "v0_4" / "logs" / "wsp-b.ledger.jsonl"
REV = ROOT / "weight_program" / "v0_4" / "wsp-b" / "inputs" / "REVISION.md"
EDIT = ROOT / "weight_program" / "v0_4" / "wsp-b" / "inputs" / "EDIT.md"
SOURCES = [LEDGER, REV, EDIT]
COMMIT = "d177712"

A = "74576466ac6d22c6cc6d7545e4162a0ae2daf38a6049624e232caf635910079a"
R = "ac483550ce6d5b858e0150059d1631e4418e1c2856f36affbd6ddd9d7fcabde7"
F = "0ea2eb774a950389f6cd60da4e39b413cd300c47af353a59b2f035363d6bf142"


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def norm_text(s: str) -> str:
    return " ".join(s.replace("`", "").replace("*", "").split())


def quoted(path: Path, needle: str) -> str:
    if norm_text(needle) not in norm_text(path.read_text()):
        raise AssertionError(f"{rel(path)} does not contain: {needle!r}")
    return needle


def span(ids: list[int]) -> str:
    ids = sorted(ids)
    assert ids == list(range(ids[0], ids[-1] + 1)), ids
    return f"{ids[0]}-{ids[-1]}" if len(ids) > 1 else str(ids[0])


def ledger_runs() -> dict:
    """Per edit hash: which calls ran on the disclosed windows and which on Fable's own texts."""
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    out: dict[str, dict] = {}
    for row in rows:
        assert row["status"] == "ok", row["call_id"]
        h = row["edit_hash"]
        d = out.setdefault(h, {"disclosed": [], "own": [], "disclosed_windows": 0, "first_utc": row["utc"],
                               "max_disclosed_counter": 0})
        if row["op"] == "forward_disclosed":
            d["disclosed"].append(row["call_id"])
            d["disclosed_windows"] += len(row["windows"])
            d["max_disclosed_counter"] = max(d["max_disclosed_counter"], row["disclosed_edit_hashes"])
            d.setdefault("disclosed_utc", [row["utc"], row["utc"]])[1] = row["utc"]
        else:
            d["own"].append(row["call_id"])
    return {"rows": rows, "by_hash": out}


def build() -> dict:
    led = ledger_runs()
    by = led["by_hash"]
    assert set(by) == {A, R, F}, set(by)
    # Structure the figure asserts: two edit states on the disclosed windows, the final state never on them.
    assert by[A]["disclosed_windows"] == 200 and by[R]["disclosed_windows"] == 200, (by[A]["disclosed_windows"], by[R]["disclosed_windows"])
    assert by[F]["disclosed"] == [], "final state ran on disclosed windows"
    assert max(r.get("disclosed_edit_hashes", 0) for r in led["rows"]) == 2
    assert by[R]["own"] == [24, 25] and by[F]["own"] == [39, 40]

    calls = lambda ids: f"calls {span(ids)}"  # noqa: E731

    states = [
        {
            "id": "wspa", "hash": A[:8], "name": "WSP-A", "kind": "run",
            "what": "the zero-label edit, as handed over",
            "disclosed": {
                "slot": 1, "accuracy": 0.315, "calls": calls(by[A]["disclosed"]), "utc": by[A]["disclosed_utc"],
                "recall": {"PRODUCTIVE": 0.06, "RECOVERY": 0.18, "STALLED": 0.46, "OSCILLATING": 0.56},
                "quote": quoted(REV, "WSP-A on the 200 windows: accuracy 0.315; recall PROD 0.06, RECO 0.18, STAL 0.46, OSCI 0.56"),
            },
            "base_qwen_disclosed": {"accuracy": 0.240, "quote": quoted(REV, "The original model: 0.240 with PROD recall 0.00")},
        },
        {
            "id": "offline", "kind": "rejected", "names": ["b1", "b2", "b3", "b4"],
            "what": "gated RECOVERY-neuron designs, rejected by reasoning, never applied",
            "quote": quoted(REV, "gated RECOVERY neuron variants; rejected by reasoning (no gate separation from STALLED, noise amplification), never applied"),
        },
        {
            "id": "revision", "hash": R[:8], "name": "Revision", "kind": "run",
            "what": "productive default +4.2, linear RECOVERY term, layer 21 reverted, both neurons re-gated",
            "own": {"accuracy": 0.42, "baseline": 0.19, "calls": calls(by[R]["own"]), "set": "set R (64 sessions Fable wrote and labelled)",
                    "quote": quoted(REV, "revision before the STALLED rescale: acc 0.42 vs 0.19 original")},
            "disclosed": {
                "slot": 2, "accuracy": 0.425, "calls": calls(by[R]["disclosed"]), "utc": by[R]["disclosed_utc"],
                "recall": {"PRODUCTIVE": 0.20, "RECOVERY": 0.36, "STALLED": 0.70, "OSCILLATING": 0.44},
                "fixed": 45, "broken": 23,
                "quote": quoted(REV, "acc 0.425 (recall 0.20 / 0.36 / 0.70 / 0.44), 45 WSP-A errors fixed, 23 broken"),
            },
            "stalled_gain": {"measured_median": 18.1, "designed": 5,
                             "quote": quoted(REV, "median gain of +18.1 on the disclosed STALLED windows (designed: +5)")},
            "stalled_wins_wrongly": {"oscillating": 21, "recovery": 17,
                                     "quote": quoted(REV, "made STALLED win on 21 oscillating")},
        },
        {
            "id": "final", "hash": F[:8], "name": "Final (WSP-B)", "kind": "calculated",
            "what": "STALLED wire × 0.2764, calibrated to a median +5 on the same disclosed windows",
            "factor": 0.2764,
            "own": {"accuracy": 0.44, "baseline": 0.19, "calls": calls(by[F]["own"]), "set": "set R (64 sessions Fable wrote and labelled)",
                    "quote": quoted(REV, "final state: acc 0.44 vs 0.19")},
            "disclosed": {
                "slot": 3, "locked": True, "offline_estimate": 0.440,
                "recall": {"PRODUCTIVE": 0.22, "RECOVERY": 0.40, "STALLED": 0.68, "OSCILLATING": 0.46},
                "quote": quoted(REV, "Offline on the same logits (scratch/sim_rescale.py) the effect is 0.425 → 0.440 (recall 0.22 / 0.40 / 0.68 / 0.46)"),
                "lock_quote": quoted(REV, "Edit states run on the disclosed windows: 74576466… and ac483550… (two)."),
            },
        },
    ]

    return {
        "_provenance": {
            "source": [rel(p) for p in SOURCES],
            "sha256": {rel(p): sha256(p) for p in SOURCES},
            "commit": COMMIT,
            "git_commit": git_head(),
            "generated_by": GENERATED_BY,
        },
        "figure": "F27",
        "title": "Two looks, one revision",
        "disclosed": {"windows": 200, "per_class": 50, "metric": "accuracy (= mean per-class recall, since the classes are balanced)",
                      "split": "200 disclosed val windows, 50 per class; in-sample",
                      "quote": quoted(REV, "200 validation windows `V001`…`V200`, 50 per oracle class")},
        "limit": {"edit_states_on_disclosed": 2,
                  "rule": "at most two edit states on the disclosed windows (WSP-A and one revision), logits only"},
        "states": states,
        "meter": {"requests": len(led["rows"]), "quote": quoted(REV, "Meter: 40 requests, 1,439,522 tokens of 2,000,000.")},
    }


def main() -> None:
    obj = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{rel(OUT)} {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    sys.exit(main())
