"""v0.3 Phase C: train one transformer baseline (spec v0.3 §9.5, E8; experiments/v0_3/phase_c/README.md).

    uv run python scripts/v03_train_phase_c.py --budget 32 --seed 0 [--out DIR] [--force]
    uv run python scripts/v03_train_phase_c.py --budget 32 --seed 0 --smoke --out /some/scratch/dir

Reads dev (training; cache ``datasets/v0_3/dev.encoded.pt``) and val (early stopping; cache
``datasets/v0_3/val.encoded.pt``) only. Writes ``<out>/{model.pt, model.sha256, training_log.json}``,
default ``experiments/phase-c-trained/v0_3/<budget>/seed<seed>/``. Refuses to overwrite a run unless
``--force``. ``--smoke`` caps training at 2 epochs and must write outside
``experiments/phase-c-trained/``; smoke outputs are not Phase C results.
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

from agent_loop_lab.baselines import phase_c_v03 as pc  # noqa: E402

OUT_ROOT = REPO / "experiments" / "phase-c-trained" / "v0_3"
SMOKE_CAP = 2


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_out(budget: int | str, seed: int) -> Path:
    return OUT_ROOT / str(budget) / f"seed{seed}"


def _parse_budget(s: str) -> int | str:
    return "all" if s == "all" else int(s)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--budget", required=True, type=_parse_budget, help="32, 128, 512, 2048 or all")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true", help=f"cap at {SMOKE_CAP} epochs; output outside {OUT_ROOT}")
    ap.add_argument("--threads", type=int, default=pc.THREADS,
                    help=f"torch threads (registered value {pc.THREADS}; any other value is recorded and reported)")
    args = ap.parse_args(argv)

    b, s = args.budget, args.seed
    if not args.smoke and (b, s) not in pc.SCHEDULE:
        print(f"refusing: (budget {b}, seed {s}) is not in the registered schedule {list(pc.SCHEDULE)}", file=sys.stderr)
        return 2
    if b not in pc.EPOCH_CAPS:
        print(f"refusing: unknown budget {b!r}", file=sys.stderr)
        return 2
    if args.smoke:
        if args.out is None or _inside(args.out, REPO / "experiments" / "phase-c-trained"):
            print("refusing: --smoke needs an explicit --out outside experiments/phase-c-trained/", file=sys.stderr)
            return 2
        cap = SMOKE_CAP
    else:
        cap = pc.EPOCH_CAPS[b]
    out = args.out if args.out is not None else default_out(b, s)
    if (out / "model.pt").exists() and not args.force:
        print(f"refusing: {out / 'model.pt'} exists (pass --force)", file=sys.stderr)
        return 2

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731

    t0 = time.perf_counter()
    dev = pc.load_dev()
    t_dev = time.perf_counter() - t0
    log(f"dev: {len(dev.session_ids)} sessions ({'built' if dev.inputs['built_now'] else 'cached'}, {t_dev:.1f}s)")
    splits_doc = json.loads((REPO / "datasets" / "v0_2" / "splits.json").read_text())
    dev_ids = splits_doc["splits"]["dev"]
    if set(dev_ids) != set(dev.session_ids):
        raise SystemExit("dev session ids in splits.json do not match the dev data file")

    from agent_loop_lab.v03 import scoring

    t1 = time.perf_counter()
    val = scoring.load_val()
    disclosed = scoring.disclosed_sessions(val.rows)
    es = pc.es_set(val, disclosed=disclosed)
    t_val = time.perf_counter() - t1
    log(f"val: ES set {len(es.session_ids)} sessions, {es.windows} windows ({t_val:.1f}s)")

    subset = pc.budget_subset(dev_ids, b)
    pool = pc.train_sessions(dev, subset)
    log(f"budget {b} seed {s}: {len(subset)} sessions, pool {len(pool)}, cap {cap} epochs, "
        f"ES at epochs {pc.es_eval_epochs(cap)}")

    t2 = time.perf_counter()
    res = pc.train_run(pool, lambda m: pc.es_macro_f1(m, es), seed=s, cap=cap,
                       record_session_times=args.smoke, progress=log)
    wall = time.perf_counter() - t2

    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.pt"
    torch.save(res.state_dict, model_path)
    sha = _sha(model_path)

    code = {p: _sha(REPO / p) for p in (
        "src/agent_loop_lab/baselines/phase_c_v03.py", "scripts/v03_train_phase_c.py",
        "src/agent_loop_lab/v03/model.py", "src/agent_loop_lab/v03/infer.py", "src/agent_loop_lab/v03/scoring.py",
        "src/agent_loop_lab/baselines/phase_c.py", "experiments/v0_3/phase_c/README.md")}
    lengths = [int(ts.tokens.shape[0]) for ts in pool]
    doc = {
        "phase": "v03-phase-c-smoke" if args.smoke else "v03-phase-c",
        "spec_version": "0.3",
        "smoke": args.smoke,
        "budget": b,
        "budget_sessions": len(subset),
        "budget_sessions_sha256": pc.ids_sha256(subset),
        "first_sessions": subset[:5],
        "pool_tokens": {"total": int(sum(lengths)), "mean": float(np.mean(lengths)), "max": int(max(lengths))},
        **res.log(),
        "es_set": {"sessions": len(es.session_ids), "windows": es.windows,
                   "sha256_of_ids_in_order": pc.ids_sha256(es.session_ids),
                   "sha256_of_sorted_ids": pc.ids_sha256(sorted(es.session_ids)),
                   "disclosed_excluded": len(disclosed)},
        "protocol": {
            "init": "torch.manual_seed(seed); DoomLoopTransformer(); modules() order: nn.Linear.reset_parameters, "
                    f"Embedding normal_(0, {pc.EMB_STD}); rel_bias 0",
            "loss": "pooled mean cross-entropy over the batch's gate-passing scored EOE positions",
            "optimiser": f"Adam(lr={pc.LR}), PyTorch defaults, no weight decay, no schedule",
            "grad_clip": f"clip_grad_norm_(max_norm={pc.GRAD_CLIP})",
            "batch_sessions": pc.BATCH_SESSIONS,
            "sequence_handling": "per-session forward + backward, gradients accumulated, one step per batch",
            "shuffle": "torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed)) per epoch",
            "budget_order": f"ascending sha256({pc.BUDGET_KEY!r})",
            "es_order": f"ascending sha256({pc.ES_KEY!r}), disclosed removed, first {pc.ES_SESSIONS}",
            "es_schedule": f"epoch % ceil(cap/{pc.MAX_ES_EVALS}) == 0 or epoch == cap; patience "
                           f"{pc.PATIENCE_EVALS} evaluations; strict improvement; keep best",
            "registered_epoch_cap": pc.EPOCH_CAPS[b],
        },
        "model_sha256": sha,
        "wall_seconds": {"train_run": wall, "load_dev": t_dev, "load_val_and_es": t_val},
        "code_sha256": code,
        "data_sha256": {"dev": dev.inputs["cache_key"], "val": val.inputs["cache_key"]},
        "environment": {
            "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
            "registered_threads": pc.THREADS,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }
    (out / "training_log.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    (out / "model.sha256").write_text(f"{sha}  model.pt\n")  # written last: marks a complete run
    log(f"done: best epoch {res.best_epoch}, ES macro-F1 {res.best_es_macro_f1:.4f}, {res.steps} steps, "
        f"{wall:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
