"""F7 (post Figure 11): "Labels against macro-F1", plus the early-stopping inset for the largest run.

Run from the repo root:  uv run python scripts/blog/figures/f7.py

Reuses the F7 builder in scripts/blog/export_figures.py unchanged (committed as blog_figures/F7.json @ bf164d1) and
adds one block, "es_inset": the largest trained run's (v0.3 C-all seed0) early-stopping macro-F1 after each epoch,
copied from its committed training log. That score is on a fixed 500-session val subset (disclosed sessions removed),
not on a sealed split, and is optimistic by construction (it chose the checkpoint). No evaluator, scorer or
dataset is opened.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F7.json"
GENERATED_BY = "scripts/blog/figures/f7.py (builder: scripts/blog/export_figures.py f7, plus es_inset)"
LOG = ROOT / "experiments" / "phase-c-trained" / "v0_3" / "all" / "seed0" / "training_log.json"


def _export_figures():
    spec = importlib.util.spec_from_file_location("export_figures", ROOT / "scripts" / "blog" / "export_figures.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def es_inset() -> dict:
    log = json.loads(LOG.read_text())
    assert log["budget"] == "all" and log["seed"] == 0 and not log["smoke"]
    evals = [{"epoch": e["epoch"], "macro_f1": round(e["es_macro_f1"], 5)} for e in log["es_evaluations"]]
    es = log["es_set"]
    out = {
        "run": "v0.3 C-all seed0",
        "labelled_windows": log["train_positions"],
        "epoch_cap": log["epoch_cap"],
        "epochs_run": log["epochs_run"],
        "stopped": log["stopped"],
        "best_epoch": log["best_epoch"],
        "evaluations": evals,
        "es_set": {"sessions": es["sessions"], "windows": es["windows"], "disclosed_excluded": es["disclosed_excluded"]},
        "split": "early-stopping subset: 500 val sessions (9,350 windows), disclosed sessions removed; not sealed",
        "source": str(LOG.relative_to(ROOT)),
    }
    assert [round(e["macro_f1"], 3) for e in evals] == [0.830, 0.912, 0.934]
    assert out["epoch_cap"] == 3 and out["stopped"] == "epoch_cap" and es["sessions"] == 500 and es["windows"] == 9350
    return out


def build() -> dict:
    ef = _export_figures()
    obj = ef.f7(ef.load(ef.V03), ef.load(ef.V02))
    obj["es_inset"] = es_inset()
    prov = obj["_provenance"]
    rel = str(LOG.relative_to(ROOT))
    prov["source"].append(rel)
    prov["sha256"][rel] = hashlib.sha256(LOG.read_bytes()).hexdigest()
    prov["generated_by"] = GENERATED_BY
    prov["source_commit"] = {s: _git("log", "-1", "--format=%H", "--", s) for s in prov["source"]}
    # The caption's numbers, checked against the built data.
    means = {m["budget"]: m for m in obj["c_means"]}
    assert [round(means[b]["test"], 3) for b in ("32", "128", "512", "2048", "all")] == [0.347, 0.450, 0.707, 0.755, 0.923]
    assert [round(means[b]["ood"], 3) for b in ("32", "128", "512", "2048", "all")] == [0.241, 0.287, 0.305, 0.400, 0.488]  # manifest says 0.288; the mean is 0.28750 - 4e-6
    return obj


def main() -> None:
    obj = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{OUT.relative_to(ROOT)} {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    sys.exit(main())
