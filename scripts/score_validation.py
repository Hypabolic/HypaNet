"""M3 validation scoring of archived weight-program phases (experiments/m3/README.md §1).

    uv run python scripts/score_validation.py PHASE [PHASE ...]

PHASE names an archived directory ``weight_program/<PHASE>/`` (for example
``phase-a-strict``, ``phase-a-selfcheck``). For each phase the archived
``artifact.dlw`` (checked against ``artifact.sha256``) and ``reference_policy.py``
are loaded from that directory, never from the author directory, and scored on
the **validation** split only. The split is hard-coded (:data:`SPLIT`); there
is no option, argument or environment variable that changes it. The sealed test
and OOD sets are scored once, in M4, by a separate script.

Writes ``artifacts/reports/m3_val_<PHASE>.json`` (refuses to overwrite unless
``--force``). The reports are for the adjudicator only: no aggregate validation
metric is ever given to the weight author.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:  # pragma: no cover - uv run puts the package on the path
    sys.path.insert(0, str(REPO / "src"))

from agent_loop_lab.eval import metrics  # noqa: E402
from agent_loop_lab.eval.collisions import WindowRow  # noqa: E402
from agent_loop_lab.model import artifact_sha256, load_artifact, predict_classes  # noqa: E402
from agent_loop_lab.runner import check_reference_policy_source  # noqa: E402
from agent_loop_lab.schema import INPUT_DIM, SPEC_VERSION  # noqa: E402

SPLIT = "val"  # hard-coded: README §1 ("the scoring script hard-codes {"val"} and has no override")
assert frozenset({SPLIT}) == metrics.ALLOWED_SPLITS
DATASET_DIR = REPO / "datasets" / "v0_2"
WEIGHT_PROGRAM = REPO / "weight_program"
REPORT_DIR = REPO / "artifacts" / "reports"
PHASE_RE = re.compile(r"^phase-[a-z0-9]+(?:-[a-z0-9]+)*$")
REPORT_NAME = "m3_val_{phase}.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_dir(phase: str, root: Path = WEIGHT_PROGRAM) -> Path:
    """The archived directory of ``phase``; refuses names that are not plain phase ids."""
    if not PHASE_RE.match(phase):
        raise ValueError(f"not a phase name: {phase!r}")
    d = root / phase
    for f in ("artifact.dlw", "artifact.sha256", "reference_policy.py"):
        if not (d / f).is_file():
            raise FileNotFoundError(f"{d / f} missing (only archived phases are scored)")
    return d


def load_phase(d: Path) -> tuple[Any, dict, Callable[[tuple[float, ...]], int], dict[str, str]]:
    """Archived network and reference policy of one phase directory, sha-checked."""
    dlw = d / "artifact.dlw"
    recorded = (d / "artifact.sha256").read_text().split()[0]
    actual = artifact_sha256(dlw)
    if actual != recorded:
        raise ValueError(f"{dlw}: sha256 {actual} != recorded {recorded}")
    model, meta = load_artifact(dlw)
    model.eval()
    pol_path = d / "reference_policy.py"
    problems = check_reference_policy_source(pol_path)
    if problems:
        raise ValueError(f"{pol_path} fails the reference_policy static checks: {problems}")
    name = f"_m3_reference_policy_{hashlib.sha256(str(d).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, pol_path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(pol_path.read_bytes(), str(pol_path), "exec", dont_inherit=True), module.__dict__)  # noqa: S102
    policy = getattr(module, "reference_policy")
    shas = {"artifact_sha256": actual, "reference_policy_sha256": _sha256_file(pol_path)}
    return model, meta, policy, shas


def predictions(rows: Sequence[WindowRow], model: Any, policy: Callable) -> tuple[np.ndarray, np.ndarray]:
    """Network and reference-policy class per row (-1 for gate-rejected rows)."""
    net = np.full(len(rows), -1, dtype=np.int64)
    ref = np.full(len(rows), -1, dtype=np.int64)
    idx = [i for i, r in enumerate(rows) if r.gate_passed]
    if not idx:
        return net, ref
    x = np.stack([np.frombuffer(rows[i].vector, dtype=np.float32) for i in idx])
    if x.shape[1] != INPUT_DIM:
        raise ValueError(f"vectors of width {x.shape[1]} != {INPUT_DIM}")
    net[idx] = predict_classes(model, x).astype(np.int64)
    for i, v in zip(idx, x):
        c = policy(tuple(v.astype(np.float64).tolist()))
        if type(c) is not int or not 0 <= c <= 3:
            raise ValueError(f"reference_policy returned {c!r} on {rows[i].session_id}@{rows[i].endpoint}")
        ref[i] = c
    return net, ref


def _summary(rep: dict) -> dict[str, Any]:
    o = rep["overall"]

    def f(stat: dict) -> dict:
        return {"value": stat["value"], "ci95": stat["ci95"]}

    return {
        model: {
            "macro_f1": f(o[model]["macro_f1"]),
            "false_alarm_rate": f(o[model]["false_alarm_rate"]),
            "accuracy": f(o[model]["accuracy"]),
        }
        for model in ("network", "reference_policy")
    } | {"h2_fidelity_all": f(o["h2_fidelity"]["all"]["agreement"])}


def score_phase(phase: str, rows: Sequence[WindowRow], dataset_inputs: dict, *, root: Path = WEIGHT_PROGRAM,
                n_resamples: int = metrics.BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    d = phase_dir(phase, root)
    model, meta, policy, shas = load_phase(d)
    net, ref = predictions(rows, model, policy)
    rep = metrics.score_report(rows, net, ref, n_resamples=n_resamples, seed=metrics.BOOTSTRAP_SEED)
    return {
        "report": "m3_val",
        "spec_version": SPEC_VERSION,
        "phase": phase,
        "split": SPLIT,
        "inputs": {
            "artifact": f"weight_program/{phase}/artifact.dlw",
            "reference_policy": f"weight_program/{phase}/reference_policy.py",
            **shas,
            "artifact_metadata": meta,
            **dataset_inputs,
            "scoring_code_sha256": {
                "scripts/score_validation.py": _sha256_file(Path(__file__).resolve()),
                "src/agent_loop_lab/eval/metrics.py": _sha256_file(Path(metrics.__file__).resolve()),
            },
        },
        "summary": _summary(rep),
        **rep,
    }


def dataset_inputs(out_dir: Path) -> dict[str, Any]:
    std = out_dir / "standard"
    return {
        "dataset_manifest_sha256": _sha256_file(out_dir / "MANIFEST.json"),
        "val_files_sha256": {
            f"{SPLIT}.{k}.jsonl.gz": _sha256_file(std / f"{SPLIT}.{k}.jsonl.gz") for k in ("events", "oracle")
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("phases", nargs="+", help="archived phase directories under weight_program/")
    ap.add_argument("--force", action="store_true", help="overwrite existing reports")
    args = ap.parse_args(argv)
    targets = []
    for phase in dict.fromkeys(args.phases):
        phase_dir(phase)
        out = REPORT_DIR / REPORT_NAME.format(phase=phase)
        if out.exists() and not args.force:
            print(f"refusing: {out} exists (pass --force to overwrite)", file=sys.stderr)
            return 2
        targets.append((phase, out))
    t0 = time.time()
    rows = metrics.load_rows(DATASET_DIR, SPLIT)
    inputs = dataset_inputs(DATASET_DIR)
    print(f"{SPLIT}: {len(rows)} windows loaded in {time.time() - t0:.1f}s")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for phase, out in targets:
        t1 = time.time()
        rep = score_phase(phase, rows, inputs)
        out.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
        s = rep["summary"]
        print(f"{phase}: {time.time() - t1:.1f}s -> {out.relative_to(REPO)}")
        for model in ("network", "reference_policy"):
            m = s[model]
            print(
                f"  {model:16s} macro-F1 {m['macro_f1']['value']:.4f} {m['macro_f1']['ci95']}"
                f"  false-alarm {m['false_alarm_rate']['value']:.4f} {m['false_alarm_rate']['ci95']}"
            )
        h = s["h2_fidelity_all"]
        print(f"  h2 fidelity      {h['value']:.4f} {h['ci95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
