"""M4 Phase C: train the 15 supervised baselines (spec v0.2 §9.5; experiments/m4/README.md §1).

    uv run python scripts/m4_train_phase_c.py [--force] [--budgets 32 128 ...] [--seeds 0 1 2]

Reads the dev split (training) and the val split (early stopping) only. Writes
``experiments/phase-c-trained/v0_2/<budget>/seed<seed>/{model.dlw, model.sha256, training_log.json}``
and ``experiments/phase-c-trained/v0_2/SUMMARY.json``. Refuses to overwrite an
existing run directory unless ``--force``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(REPO / "src"))

from agent_loop_lab.baselines import phase_c  # noqa: E402
from agent_loop_lab.datasets.build import iter_sessions  # noqa: E402
from agent_loop_lab.eval.collisions import session_windows  # noqa: E402
from agent_loop_lab.model import save_artifact, sparsity  # noqa: E402

DATASET_DIR = REPO / "datasets" / "v0_2"
OUT_ROOT = REPO / "experiments" / "phase-c-trained" / "v0_2"
TRAIN_SPLITS = ("dev", "val")  # the only splits this script may read
THREADS = 2


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(split: str):
    if split not in TRAIN_SPLITS:
        raise ValueError(f"Phase C training may not read split {split!r}")
    rows = []
    for events, record in iter_sessions(DATASET_DIR, split):
        rows.extend(session_windows(events, record, split))
    return rows


def run_dir(budget: int | str, seed: int, root: Path = OUT_ROOT) -> Path:
    return root / str(budget) / f"seed{seed}"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--budgets", nargs="*", default=[str(b) for b in phase_c.BUDGETS])
    ap.add_argument("--seeds", nargs="*", type=int, default=list(phase_c.SEEDS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    budgets = [b if b == "all" else int(b) for b in args.budgets]

    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    targets = [(b, s) for b in budgets for s in args.seeds]
    for b, s in targets:
        d = run_dir(b, s)
        if (d / "model.dlw").exists() and not args.force:
            print(f"refusing: {d} exists (pass --force)", file=sys.stderr)
            return 2

    t0 = time.time()
    dev_rows = load_rows("dev")
    val_rows = load_rows("val")
    print(f"loaded dev {len(dev_rows)} / val {len(val_rows)} windows in {time.time() - t0:.1f}s", flush=True)
    splits_doc = json.loads((DATASET_DIR / "splits.json").read_text())
    dev_ids = splits_doc["splits"]["dev"]
    if set(dev_ids) != {r.session_id for r in dev_rows}:
        raise SystemExit("dev session ids in splits.json do not match the dev data file")
    x_val, y_val, _ = phase_c.windows_to_arrays(val_rows)

    code_sha = {
        "src/agent_loop_lab/baselines/phase_c.py": _sha(Path(phase_c.__file__).resolve()),
        "scripts/m4_train_phase_c.py": _sha(Path(__file__).resolve()),
    }
    data_sha = {
        f"standard/{s}.{k}.jsonl.gz": _sha(DATASET_DIR / "standard" / f"{s}.{k}.jsonl.gz")
        for s in TRAIN_SPLITS
        for k in ("events", "oracle")
    }
    env = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    summary_runs = []
    for b, s in targets:
        subset = phase_c.budget_subset(dev_ids, b)
        x, y, _ = phase_c.windows_to_arrays(dev_rows, set(subset))
        t1 = time.time()
        res = phase_c.train_run(x, y, x_val, y_val, seed=s, budget=b)
        d = run_dir(b, s)
        d.mkdir(parents=True, exist_ok=True)
        meta = {
            "phase": "phase-c",
            "spec_version": "0.2",
            "budget_sessions": b,
            "budget_sessions_count": len(subset),
            "seed": s,
            "train_windows": res.train_windows,
            "best_epoch": res.best_epoch,
            "best_val_macro_f1": res.best_val_macro_f1,
        }
        sha = save_artifact(res.model, d / "model.dlw", meta)
        (d / "model.sha256").write_text(f"{sha}  model.dlw\n")
        log = {
            **res.log(),
            "budget_sessions_count": len(subset),
            "budget_sessions_sha256": hashlib.sha256("\n".join(subset).encode()).hexdigest(),
            "first_sessions": subset[:5],
            "protocol": {
                "init": "torch.manual_seed(seed); DoomLoopMLP() (PyTorch default nn.Linear init)",
                "loss": "cross_entropy (mean, unweighted)",
                "optimiser": f"Adam(lr={phase_c.LR}), PyTorch defaults",
                "batch_size": phase_c.BATCH_SIZE,
                "max_epochs": phase_c.MAX_EPOCHS,
                "early_stopping": f"val macro-F1, strict improvement, patience {phase_c.PATIENCE}, keep best",
                "shuffle": "torch.randperm(n, generator=torch.Generator().manual_seed(seed)) per epoch",
                "budget_order": f"ascending sha256({phase_c.BUDGET_KEY!r}).hexdigest() over splits.json dev ids",
            },
            "val_windows": int(len(y_val)),
            "model_sha256": sha,
            "sparsity": sparsity(res.model),
            "code_sha256": code_sha,
            "data_sha256": data_sha,
            "environment": env,
            "wall_seconds": time.time() - t1,
        }
        (d / "training_log.json").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
        summary_runs.append({k: log[k] for k in ("budget", "seed", "train_windows", "budget_sessions_count",
                                                 "epochs_run", "best_epoch", "best_val_macro_f1", "model_sha256")})
        print(f"budget {b} seed {s}: {res.train_windows} windows, epochs {res.epochs_run}, best epoch "
              f"{res.best_epoch}, val macro-F1 {res.best_val_macro_f1:.4f} ({time.time() - t1:.1f}s)", flush=True)

    summary_path = OUT_ROOT / "SUMMARY.json"
    old = json.loads(summary_path.read_text())["runs"] if summary_path.exists() else []
    merged = {(str(r["budget"]), r["seed"]): r for r in old}
    merged.update({(str(r["budget"]), r["seed"]): r for r in summary_runs})
    order = {str(b): i for i, b in enumerate(phase_c.BUDGETS)}
    runs = sorted(merged.values(), key=lambda r: (order.get(str(r["budget"]), 99), r["seed"]))
    summary_path.write_text(json.dumps({"runs": runs, "code_sha256": code_sha, "environment": env},
                                       indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
