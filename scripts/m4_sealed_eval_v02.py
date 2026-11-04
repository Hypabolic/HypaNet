"""M4 sealed evaluation of every v0.2 column (spec v0.2 §10; experiments/m4/README.md §3).

    uv run python scripts/m4_sealed_eval_v02.py --dry-run-val   # val only, scratch report
    uv run python scripts/m4_sealed_eval_v02.py                 # the one sealed run

The sealed run scores test, OOD, val (reference) and the counterfactual set
once and writes ``artifacts/reports/m4_sealed_v02.json``. It refuses to start
if that report or its start marker ``m4_sealed_v02.started.json`` exists,
unless ``--force`` (a forced rerun records the earlier marker). The marker is
written before any sealed file is opened.

``--dry-run-val`` runs the same pipeline on val only (no test, OOD or
counterfactual file is opened, the sampler uses 2,000 windows) and writes
``artifacts/reports/m4_dryrun_val_v02.json``. It never reads or writes the
sealed marker or report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(REPO / "src"))

from agent_loop_lab.baselines import phase_c  # noqa: E402
from agent_loop_lab.eval import m4  # noqa: E402
from agent_loop_lab.eval import metrics as M  # noqa: E402
from agent_loop_lab.eval.collisions import collision_audit  # noqa: E402
from agent_loop_lab.eval.sampler import SAMPLER_SEED, SAMPLER_WINDOWS, sample_windows  # noqa: E402
from agent_loop_lab.model import artifact_sha256, load_artifact, predict_classes, sparsity  # noqa: E402
from agent_loop_lab.runner import check_reference_policy_source  # noqa: E402
from agent_loop_lab.schema import CLASS_NAMES, INPUT_DIM, ClassLabel  # noqa: E402

DATASET_DIR = REPO / "datasets" / "v0_2"
WEIGHT_PROGRAM = REPO / "weight_program"
PHASE_C_ROOT = REPO / "experiments" / "phase-c-trained" / "v0_2"
RULES_DIR = REPO / "experiments" / "commissions" / "rules_baseline"
REPORT_DIR = REPO / "artifacts" / "reports"
SEALED_REPORT = REPORT_DIR / "m4_sealed_v02.json"
SEALED_MARKER = REPORT_DIR / "m4_sealed_v02.started.json"
DRY_REPORT = REPORT_DIR / "m4_dryrun_val_v02.json"
THREADS = 2
PHASES = ("phase-a-strict", "phase-a-selfcheck", "phase-b-revision")
PHASE_SHORT = {"phase-a-strict": "A-strict", "phase-a-selfcheck": "A-selfcheck", "phase-b-revision": "B"}
LABELLED = {"A-strict": (0, 0), "A-selfcheck": (0, None), "B": (200, 1)}  # (labelled examples, revisions)
DRY_SAMPLER_WINDOWS = 2_000
AUTHOR_PATHS = ("experiments/v0_3/SESSION_ID", "experiments/v0_3/logs/")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _git() -> dict[str, Any]:
    def run(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()

    return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "status_porcelain": [ln for ln in run("status", "--porcelain").splitlines()
                                 if not any(x in ln for x in AUTHOR_PATHS)],
            "status_note": "lines for the running v0.3 author session's paths are filtered out"}


def _say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


class Column:
    def __init__(self, name: str, kind: str, fn: Callable[[np.ndarray, list], np.ndarray], info: dict[str, Any],
                 model: Any = None, policy: Callable | None = None) -> None:
        self.name, self.kind, self.fn, self.info, self.model, self.policy = name, kind, fn, info, model, policy


def _exec_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)  # noqa: S102
    return module


def _policy_fn(policy: Callable, label: str) -> Callable[[np.ndarray, list], np.ndarray]:
    def fn(x: np.ndarray, tuples: list) -> np.ndarray:
        out = np.empty(len(tuples), dtype=np.int64)
        for i, t in enumerate(tuples):
            c = policy(t)
            if type(c) is not int or not 0 <= c <= 3:
                raise ValueError(f"{label} returned {c!r}")
            out[i] = c
        return out

    return fn


def _net_fn(model: Any) -> Callable[[np.ndarray, list], np.ndarray]:
    return lambda x, tuples: predict_classes(model, x).astype(np.int64)


def load_columns() -> tuple[list[Column], dict[str, tuple[str, str]], dict[str, list[str]], dict[str, Any]]:
    cols: list[Column] = []
    fid_pairs: dict[str, tuple[str, str]] = {}
    groups: dict[str, list[str]] = {}
    inputs: dict[str, Any] = {"weight_program": {}, "phase_c": {}, "rules": {}}
    for phase in PHASES:
        d = WEIGHT_PROGRAM / phase
        recorded = (d / "artifact.sha256").read_text().split()[0]
        actual = artifact_sha256(d / "artifact.dlw")
        if actual != recorded:
            raise SystemExit(f"{phase}: artifact sha256 {actual} != recorded {recorded}")
        model, meta = load_artifact(d / "artifact.dlw")
        model.eval()
        pol_path = d / "reference_policy.py"
        problems = check_reference_policy_source(pol_path)
        if problems:
            raise SystemExit(f"{pol_path}: {problems}")
        policy = _exec_module(pol_path, f"_m4_ref_{phase.replace('-', '_')}").reference_policy
        short = PHASE_SHORT[phase]
        ex, rev = LABELLED[short]
        if rev is None:
            text = (d / "ITERATIONS.md").read_text()
            rev = sum(1 for line in text.splitlines() if line.startswith("## Iteration"))
        info = {"labelled_examples": ex, "revisions_or_iterations": rev, "artifact_sha256": actual,
                "reference_policy_sha256": _sha(pol_path)}
        cols.append(Column(f"{short} network", "network", _net_fn(model), {**info, "sparsity": sparsity(model)}, model=model))
        cols.append(Column(f"{short} reference_policy", "policy", _policy_fn(policy, f"{phase} reference_policy"), info,
                           policy=policy))
        fid_pairs[short] = (f"{short} network", f"{short} reference_policy")
        inputs["weight_program"][phase] = info
    for b in phase_c.BUDGETS:
        names = []
        for s in phase_c.SEEDS:
            d = PHASE_C_ROOT / str(b) / f"seed{s}"
            dlw = d / "model.dlw"
            if not dlw.is_file():
                raise SystemExit(f"Phase C artefact missing: {dlw}")
            recorded = (d / "model.sha256").read_text().split()[0]
            actual = artifact_sha256(dlw)
            if actual != recorded:
                raise SystemExit(f"{dlw}: sha256 {actual} != recorded {recorded}")
            log = json.loads((d / "training_log.json").read_text())
            if log["model_sha256"] != actual:
                raise SystemExit(f"{d}: training log sha256 mismatch")
            model, _ = load_artifact(dlw)
            model.eval()
            name = f"C-{b} seed{s}"
            info = {"labelled_examples": log["train_windows"], "budget_sessions": log["budget_sessions_count"],
                    "epochs_run": log["epochs_run"], "best_epoch": log["best_epoch"],
                    "best_val_macro_f1_training_log": log["best_val_macro_f1"], "artifact_sha256": actual,
                    "sparsity": sparsity(model)}
            cols.append(Column(name, "network", _net_fn(model), info, model=model))
            names.append(name)
            inputs["phase_c"][name] = {"artifact_sha256": actual, "training_log_sha256": _sha(d / "training_log.json")}
        groups[f"C-{b}"] = names
    # Rules engine, sha-checked against its deliverables list.
    expected = {}
    for line in (RULES_DIR / "DELIVERABLES.sha256").read_text().splitlines():
        if line.strip():
            h, f = line.split()
            expected[f] = h
    for f in ("rules.py", "RULES.md"):
        if _sha(RULES_DIR / f) != expected[f]:
            raise SystemExit(f"rules baseline {f}: sha256 does not match DELIVERABLES.sha256")
    rules = _exec_module(RULES_DIR / "rules.py", "_m4_rules").rules_classify
    inputs["rules"] = {"rules.py_sha256": expected["rules.py"], "RULES.md_sha256": expected["RULES.md"]}
    cols.append(Column("rules", "policy", _policy_fn(rules, "rules_classify"), {"labelled_examples": 0}, policy=rules))
    return cols, fid_pairs, groups, inputs


def dev_prior() -> dict[str, Any]:
    rows, _ = m4.load_split(DATASET_DIR, "dev")
    counts = np.zeros(len(ClassLabel), dtype=np.int64)
    for r in rows:
        if r.gate_passed and r.label.is_scored:
            counts[int(r.label.to_class())] += 1
    maj = int(np.argmax(counts))  # first maximum: canonical tie rule
    return {"class_counts": {c: int(counts[i]) for i, c in enumerate(CLASS_NAMES)}, "majority": CLASS_NAMES[maj],
            "majority_index": maj, "scored_windows": int(counts.sum())}


def add_priors(cols: list[Column], prior: dict[str, Any]) -> None:
    maj = prior["majority_index"]
    cols.append(Column("prior dev-majority", "prior", lambda x, t, c=maj: np.full(x.shape[0], c, np.int64),
                       {"labelled_examples": prior["scored_windows"], "class": prior["majority"],
                        "note": "dev label counts only"}))
    cols.append(Column("prior always-PRODUCTIVE", "prior", lambda x, t: np.zeros(x.shape[0], np.int64),
                       {"labelled_examples": 0, "class": "PRODUCTIVE"}))


# ---------------------------------------------------------------------------
# Per split
# ---------------------------------------------------------------------------


def predict_all(rows: list, cols: list[Column]) -> dict[str, np.ndarray]:
    idx = np.array([i for i, r in enumerate(rows) if r.gate_passed], dtype=np.int64)
    x = np.frombuffer(b"".join(rows[i].vector for i in idx), dtype=np.float32).reshape(len(idx), INPUT_DIM)
    tuples = [tuple(v) for v in x.astype(np.float64).tolist()]
    out = {}
    for c in cols:
        p = np.full(len(rows), -1, dtype=np.int64)
        p[idx] = c.fn(x, tuples)
        out[c.name] = p
    return out


def sampler_fidelity(cols: list[Column], n: int) -> dict[str, Any]:
    x = sample_windows(n, SAMPLER_SEED)
    tuples = [tuple(v) for v in x.astype(np.float64).tolist()]
    out: dict[str, Any] = {"windows": n, "seed": SAMPLER_SEED, "x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                           "phases": {}}
    by_name = {c.name: c for c in cols}
    for short in PHASE_SHORT.values():
        net = by_name[f"{short} network"].fn(x, tuples)
        ref = by_name[f"{short} reference_policy"].fn(x, tuples)
        agree = net == ref
        per_ref = {}
        for k, cname in enumerate(CLASS_NAMES):
            m = ref == k
            per_ref[cname] = {"windows": int(m.sum()), "agreement": float(agree[m].mean()) if m.any() else None}
        confusion = [[int(((ref == a) & (net == b)).sum()) for b in range(4)] for a in range(4)]
        out["phases"][short] = {"agreement": float(agree.mean()), "agree": int(agree.sum()),
                                "by_reference_class": per_ref,
                                "confusion_ref_rows_net_cols": confusion}
    return out


def latency(cols: list[Column]) -> dict[str, Any]:
    x = sample_windows(1024, SAMPLER_SEED + 1)
    t1 = torch.from_numpy(x[:1].copy())
    t1024 = torch.from_numpy(x.copy())
    tuples = [tuple(v) for v in x.astype(np.float64).tolist()]
    out: dict[str, Any] = {"torch_threads": torch.get_num_threads(), "unit": "microseconds",
                           "method": "batch 1: 50 warm-up then median of 1,000 calls; batch 1024: 10 warm-up then median of 200 calls; forward + argmax under no_grad; Python policies: mean per-window over 1,024 windows"}
    for c in cols:
        if c.kind == "network":
            res = {}
            with torch.no_grad():
                for label, t, warm, reps in (("batch_1", t1, 50, 1000), ("batch_1024", t1024, 10, 200)):
                    for _ in range(warm):
                        torch.argmax(c.model(t), dim=-1)
                    ts = []
                    for _ in range(reps):
                        a = time.perf_counter_ns()
                        torch.argmax(c.model(t), dim=-1)
                        ts.append(time.perf_counter_ns() - a)
                    res[label] = float(np.median(ts)) / 1e3
            out[c.name] = res
        elif c.kind == "policy":
            a = time.perf_counter_ns()
            for tt in tuples:
                c.policy(tt)
            out[c.name] = {"per_window": (time.perf_counter_ns() - a) / len(tuples) / 1e3}
    return out


def score_split(split: str, cols: list[Column], fid_pairs, groups, n_resamples: int) -> tuple[dict[str, Any], dict]:
    t0 = time.time()
    rows, meta = m4.load_split(DATASET_DIR, split)
    _say(f"{split}: {len(rows)} windows loaded ({time.time() - t0:.1f}s)")
    preds = predict_all(rows, cols)
    _say(f"{split}: predictions done ({time.time() - t0:.1f}s)")
    out: dict[str, Any] = {"gate": m4.gate_report(rows)}
    if split == "counterfactual":
        out["counterfactual_consistency"] = m4.counterfactual_consistency(rows, meta, preds, n_resamples=n_resamples)
    else:
        out["scores"] = m4.score_columns(rows, preds, fid_pairs, groups, n_resamples=n_resamples)
        if split in ("test", "ood"):
            coll = collision_audit(rows, max_examples=3)
            coll["collision_groups"] = coll["collision_groups"][:20]
            out["collisions"] = coll
    _say(f"{split}: scored ({time.time() - t0:.1f}s)")
    return out, preds


def summary_table(splits: dict[str, Any], cols: list[Column], groups) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for split, res in splits.items():
        if "scores" not in res:
            continue
        o = res["scores"]["overall"]
        t = {}
        for c in cols:
            b = o["columns"][c.name]
            t[c.name] = {
                "macro_f1": [b["macro_f1"]["value"], b["macro_f1"]["ci95"]],
                "false_alarm_rate": [b["false_alarm_rate"]["value"], b["false_alarm_rate"]["ci95"]],
                "recall": {k: b["per_class"][k]["recall"]["value"] for k in CLASS_NAMES},
            }
        for g in groups:
            gg = o["groups"][g]
            t[f"{g} mean±sd"] = {m: {"mean": gg[m]["mean"], "sd": gg[m]["sd"], "seed_mean_ci95": gg[m]["seed_mean_ci95"]}
                                 for m in ("macro_f1", "false_alarm_rate")}
        for short, f in o["h2_fidelity"].items():
            t[f"{short} network"]["h2_fidelity_all"] = [f["all"]["agreement"]["value"], f["all"]["agreement"]["ci95"]]
        table[split] = t
    return table


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run-val", action="store_true", help="val only; scratch report; sealed files untouched")
    ap.add_argument("--force", action="store_true", help="rerun the sealed evaluation (recorded)")
    args = ap.parse_args(argv)
    torch.set_num_threads(THREADS)
    started = _now()
    previous = []
    if not args.dry_run_val:
        for p in (SEALED_REPORT, SEALED_MARKER):
            if p.exists():
                if not args.force:
                    print(f"refusing: {p.relative_to(REPO)} exists; the sealed evaluation runs once "
                          f"(pass --force to rerun, which is recorded)", file=sys.stderr)
                    return 2
                previous.append({"file": p.relative_to(REPO).as_posix(), "content": json.loads(p.read_text())
                                 if p == SEALED_MARKER else {"sha256": _sha(p)}})
    git = _git()
    _say("loading columns (archived artefacts, Phase C, rules)")
    cols, fid_pairs, groups, inputs = load_columns()
    _say("dev prior")
    prior = dev_prior()
    add_priors(cols, prior)
    code_sha = {p: _sha(REPO / p) for p in (
        "scripts/m4_sealed_eval_v02.py", "src/agent_loop_lab/eval/m4.py", "src/agent_loop_lab/eval/metrics.py",
        "src/agent_loop_lab/eval/sampler.py", "src/agent_loop_lab/eval/collisions.py",
        "src/agent_loop_lab/baselines/phase_c.py", "experiments/m4/README.md")}
    manifest = json.loads((DATASET_DIR / "MANIFEST.json").read_text())
    data_sha: dict[str, Any] = {"MANIFEST.json": _sha(DATASET_DIR / "MANIFEST.json")}
    def check_standard(split: str) -> None:
        for kind in ("events", "oracle"):
            rel = f"standard/{split}.{kind}.jsonl.gz"
            h = _sha(DATASET_DIR / rel)
            if manifest["files"][rel]["sha256"] != h:
                raise SystemExit(f"{rel}: sha256 does not match MANIFEST.json")
            data_sha[rel] = h

    # Unsealed splits first (val, counterfactual), so a failure there cannot spend the sealed run.
    unsealed = ["val"] if args.dry_run_val else ["val", "counterfactual"]
    n_sampler = DRY_SAMPLER_WINDOWS if args.dry_run_val else SAMPLER_WINDOWS
    for split in ("dev", *unsealed):
        check_standard(split)
    split_results: dict[str, Any] = {}
    for split in unsealed:
        split_results[split], _ = score_split(split, cols, fid_pairs, groups, M.BOOTSTRAP_RESAMPLES)
    _say(f"sampler fidelity ({n_sampler} windows)")
    sampler = sampler_fidelity(cols, n_sampler)
    _say("latency")
    lat = latency(cols)
    json.dumps([split_results, sampler, lat], allow_nan=False)  # fail before the marker if not serialisable
    if not args.dry_run_val:
        marker = {"started_utc": started, "sealed_opened_utc": _now(), "git": git, "previous": previous,
                  "inputs": inputs, "code_sha256": code_sha}
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        SEALED_MARKER.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
        _say(f"marker written: {SEALED_MARKER.relative_to(REPO)} (no sealed file opened before this)")
        check_standard("test")
        ood_manifest = json.loads((DATASET_DIR / "OOD_MANIFEST.json").read_text())
        data_sha["OOD_MANIFEST.json"] = _sha(DATASET_DIR / "OOD_MANIFEST.json")
        for rel, h in ood_manifest["files"].items():
            if _sha(DATASET_DIR / rel) != h:
                raise SystemExit(f"{rel}: sha256 does not match OOD_MANIFEST.json")
            data_sha[rel] = h
        for split in ("test", "ood"):
            split_results[split], _ = score_split(split, cols, fid_pairs, groups, M.BOOTSTRAP_RESAMPLES)
        split_results = {k: split_results[k] for k in ("test", "ood", "val", "counterfactual")}

    # Consistency: Phase C val macro-F1 must equal the training log's best value.
    consistency = {}
    vo = split_results["val"]["scores"]["overall"]["columns"]
    for c in cols:
        if c.name.startswith("C-"):
            got = vo[c.name]["macro_f1"]["value"]
            want = c.info["best_val_macro_f1_training_log"]
            consistency[c.name] = {"val_macro_f1_eval": got, "training_log": want, "match": abs(got - want) < 1e-9}
    if not all(v["match"] for v in consistency.values()):
        _say(f"WARNING: Phase C val macro-F1 differs from a training log: "
             f"{ {k: v for k, v in consistency.items() if not v['match']} }")
    report = {
        "report": "m4_dryrun_val_v02" if args.dry_run_val else "m4_sealed_v02",
        "dry_run": bool(args.dry_run_val),
        "spec_version": "0.2",
        "pre_registration": "experiments/m4/README.md",
        "authority": "spec v0.3 E7: v0.2 sealed test/OOD scoring may run once V0c is committed (e05f371)",
        "started_utc": started,
        "finished_utc": _now(),
        "git": git,
        "previous_runs": previous,
        "environment": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                        "platform": platform.platform(), "torch_threads": torch.get_num_threads()},
        "inputs": {**inputs, "data_sha256": data_sha, "code_sha256": code_sha},
        "dev_prior": prior,
        "columns": {c.name: {"kind": c.kind, **c.info} for c in cols},
        "phase_c_groups": groups,
        "notes": [
            "val is reference only: Phase C early-stopped on val, and Phase B's 200 disclosed windows are val windows",
            "dev-majority prior equals always-PRODUCTIVE when the dev majority is PRODUCTIVE",
            "columns without a reference policy (Phase C, rules, priors) have no H2 fidelity",
        ],
        "summary": summary_table(split_results, cols, groups),
        "splits": split_results,
        "h2_fidelity_sampler": sampler,
        "latency": lat,
        "phase_c_val_consistency": consistency,
    }
    out = DRY_REPORT if args.dry_run_val else SEALED_REPORT
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n")
    _say(f"wrote {out.relative_to(REPO)}")
    for split, t in report["summary"].items():
        print(f"== {split}")
        for name, v in t.items():
            if "macro_f1" in v and isinstance(v["macro_f1"], list):
                print(f"  {name:28s} F1 {v['macro_f1'][0]:.4f} {v['macro_f1'][1]}  FA {v['false_alarm_rate'][0]:.4f}"
                      + (f"  fid {v['h2_fidelity_all'][0]:.4f}" if "h2_fidelity_all" in v else ""))
            else:
                print(f"  {name:28s} F1 {v['macro_f1']['mean']:.4f}±{v['macro_f1']['sd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
