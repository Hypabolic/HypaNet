"""v0.4 cloud GPU runner (PREREGISTRATION.md §3.1, §3.3, §4, §5): the label-free side of the H100 work.

Run on the rented machine through ``envs/qwen/scripts/cloud_run.sh`` (which builds the uv env from
``uv.lock`` and fetches the pinned model), after copying the repo subset :func:`bundle` packs. It
runs named jobs, each a separate process through :func:`exec_job`, which applies §3.1's settings
before the job's own code and checks them after:

* ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (set by the shell, checked here), ``torch.use_deterministic_algorithms(True)``,
  TF32 off for matmul and cuDNN, ``set_float32_matmul_precision("highest")``, cuDNN benchmark off,
  SDPA restricted to the math backend; refused when ``flash-linear-attention`` or ``causal-conv1d`` is
  importable (Gated DeltaNet must run its torch implementation). fp32 is each CLI's default dtype;
  the LoRA recipe's bf16 autocast is its own (§5).
* Jobs (:func:`plan_jobs`), all on dev or val only: probe features (dev, val); a probe fit and a val
  prediction per budget (32, 128, 512, 2048, all, 200w); LoRA training per budget × seed (§1: 3 / 3 /
  3 / 1 / 1, and 3 for 200w) and a val readout of every trained model; and the §3.3 CPU replay:
  the 100 val sessions with the smallest ``sha256("20261105:v04-cpu-replay:{session_id}")`` read out
  on CPU and on CUDA for Qwen-0, LoRA-all (seed 0), probe-all and, when its archive is given, WSP-B,
  with the window-level argmax agreement (below 99.5% is a reported caveat, never a score change).
  The replay runs the arms' own CLIs in a copy of the package whose export holds only those 100
  sessions' task lines (no labels), so CPU and CUDA run the same code on the same text.
* The sealed scoring (fresh test, v0.3 test, OOD, counterfactual) is **not** here: it belongs to
  ``scripts/v04_sealed_eval.py`` after G2. :func:`check_args` refuses any ``--split`` other than dev or
  val and any dataset path of a sealed split, and :func:`check_data_tree` refuses to start when the
  copied ``datasets/`` holds any file outside :data:`ALLOWED_DATA`.

The runner never reads a label and computes no score. It writes ``RUN_MANIFEST.json`` (source commit,
the registry's ``freeze_sha256`` when present, environment and GPU, settings, model and data hashes,
each job's record with its outputs' sha256) and ``COPY_BACK.json`` (what to bring home). A job whose
record is ``ok`` and whose outputs still hash as recorded is not rerun; a failed job is not retried
unless asked (§10: a diverged LoRA run is reported as failed, never re-run with other settings).

CLIs (``python -m hypaqwen.cloud ...``): ``plan`` (the dry run: jobs and a cost estimate), ``run``,
``exec`` (internal), ``replay-subset``, ``replay-report``, ``readout-edit``, ``compare``, ``env``,
``fetch-model``, ``bundle``. The probe and LoRA CLIs it calls are fixed:
``hypaqwen.probe features|fit|predict`` and ``hypaqwen.lora train|readout``.
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import MODEL_ID, MODEL_REVISION, REPO, V04_DATA

FORMAT = "v04-cloud-run/1"
SPLITS = ("dev", "val")
SEALED_SPLITS = ("test", "ood", "counterfactual", "fresh_test")
BUDGETS = ("32", "128", "512", "2048", "all", "200w")
SEEDS: dict[str, tuple[int, ...]] = {"32": (0, 1, 2), "128": (0, 1, 2), "512": (0, 1, 2), "2048": (0,), "all": (0,),
                                     "200w": (0, 1, 2)}  # §1 (v0.3 E8; 200w: 0, 1, 2)
EPOCH_CAPS = {"32": 60, "128": 48, "512": 12, "2048": 3, "all": 3, "200w": 31}  # §5, used only by the estimate
POOL_SESSIONS = {"32": 32, "128": 128, "512": 512, "2048": 2048, "all": 11864, "200w": 200}  # §5 (v0.3 all-dev pool)
ES_SESSIONS = 500
MAX_ES_EVALS = 10
REPLAY_KEY = "20261105:v04-cpu-replay:{session_id}"
REPLAY_SESSIONS = 100
REPLAY_ARMS = ("qwen0", "wsp-b", "lora-all", "probe-all")
AGREEMENT_CAVEAT = 0.995
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_CUDA_PREFIX = "13.0"  # uv.lock's torch 2.14.1 manylinux_2_28_x86_64 wheel (PyPI) depends on cuda-toolkit 13.0
GPU_NAME = "H100"
GPU_MIN_MEMORY_GIB = 79
BANNED_MODULES = ("fla", "causal_conv1d")  # flash-linear-attention, causal-conv1d (§3.1)
JOB_MODULES = frozenset({"hypaqwen.probe", "hypaqwen.lora", "hypaqwen.run_readout", "hypaqwen.cloud"})
CLOUD_SUBCOMMANDS = frozenset({"replay-subset", "replay-report", "readout-edit"})  # what `exec` may run of this module
REGISTRY = REPO / "experiments" / "registry_v0_4.yaml"
QWEN_DIR = REPO / "envs" / "qwen"
# The only dataset files the cloud machine may hold (label files: dev trains, val for ES and the 200 windows).
ALLOWED_DATA = (
    "datasets/v0_4/dev.tasks.jsonl.gz",
    "datasets/v0_4/dev.labels.npz",
    "datasets/v0_4/val.tasks.jsonl.gz",
    "datasets/v0_4/val.labels.npz",
    "datasets/v0_4/EXPORT_MANIFEST.json",
    "datasets/v0_4/BUDGETS.json",
    "datasets/v0_3/MANIFEST_v0_3.json",
    "datasets/v0_2/splits.json",
    "datasets/v0_2/MANIFEST.json",
)  # no manifest of a sealed split either (fresh test, OOD): nothing on the cloud machine needs one
REQUIRED_DATA = ALLOWED_DATA[:6]
_SEALED_PATH = re.compile(r"(?:^|/)(?:" + "|".join(SEALED_SPLITS) + r")(?:[./]|$)")


class CloudError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Refusal rules
# ---------------------------------------------------------------------------


def check_split(split: str) -> None:
    if split not in SPLITS:
        raise CloudError(f"the cloud runner reads only {list(SPLITS)}, not {split!r}; sealed scoring is "
                         "scripts/v04_sealed_eval.py after G2")


def check_args(argv: Sequence[str]) -> None:
    """Refuse a job argv naming a split other than dev/val, or a dataset path of a sealed split."""
    for i, a in enumerate(argv):
        if a == "--split":
            if i + 1 >= len(argv):
                raise CloudError("--split without a value")
            check_split(argv[i + 1])
        elif a.startswith("--split="):
            check_split(a.split("=", 1)[1])
        if "datasets" in Path(a).parts and _SEALED_PATH.search(a.split("datasets", 1)[1]):
            raise CloudError(f"{a}: a sealed dataset path")


def check_data_tree(root: Path) -> dict[str, str]:
    """Every file under ``root/datasets`` must be in :data:`ALLOWED_DATA`; returns {path: sha256}.

    Run on the cloud copy only (``run`` calls it after the GPU check), never on the home repo."""
    base = root / "datasets"
    found: dict[str, str] = {}
    extra: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        for d in dirnames:
            if (Path(dirpath) / d).is_symlink():
                extra.append(str((Path(dirpath) / d).relative_to(root)))
        for f in filenames:
            rel = (Path(dirpath) / f).relative_to(root).as_posix()
            if rel not in ALLOWED_DATA or (Path(dirpath) / f).is_symlink():
                extra.append(rel)
            else:
                found[rel] = _sha256_file(root / rel)
    if extra:
        raise CloudError("refusing: the cloud copy holds dataset files outside the allow-list "
                         f"(sealed splits never leave the home machine): {sorted(extra)[:10]}")
    missing = [p for p in REQUIRED_DATA if p not in found]
    if missing:
        raise CloudError(f"missing dataset files: {missing}")
    return dict(sorted(found.items()))


# ---------------------------------------------------------------------------
# §3.1 settings, environment record
# ---------------------------------------------------------------------------


def apply_determinism() -> dict[str, Any]:
    """§3.1's settings for this process (CPU or CUDA); returns :func:`settings`."""
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise CloudError(f"CUBLAS_WORKSPACE_CONFIG must be {CUBLAS_WORKSPACE_CONFIG} before torch starts "
                         f"(got {os.environ.get('CUBLAS_WORKSPACE_CONFIG')!r})")
    check_banned()
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return settings()


def settings() -> dict[str, Any]:
    import torch

    return {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "sdpa": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                 "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                 "cudnn": torch.backends.cuda.cudnn_sdp_enabled(), "math": torch.backends.cuda.math_sdp_enabled()},
        "torch_threads": torch.get_num_threads(),
    }


REQUIRED_SETTINGS = {"CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG, "deterministic_algorithms": True,
                     "matmul_allow_tf32": False, "cudnn_allow_tf32": False, "float32_matmul_precision": "highest",
                     "sdpa": {"flash": False, "mem_efficient": False, "cudnn": False, "math": True}}


def settings_problems(s: Mapping[str, Any]) -> list[str]:
    return [f"{k}={s.get(k)!r}, want {v!r}" for k, v in REQUIRED_SETTINGS.items() if s.get(k) != v]


def check_banned() -> None:
    found = [m for m in BANNED_MODULES if importlib.util.find_spec(m) is not None]
    if found:
        raise CloudError(f"{found} importable; §3.1 runs Gated DeltaNet's torch implementation (uninstall them)")


def _versions() -> dict[str, str | None]:
    from importlib import metadata

    out = {}
    for p in ("torch", "transformers", "peft", "accelerate", "safetensors", "numpy", "tokenizers", "huggingface-hub"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def _run_text(cmd: Sequence[str], cwd: Path | None = None) -> str | None:
    try:
        r = subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def gpu_record() -> dict[str, Any]:
    import torch

    rec: dict[str, Any] = {"torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                           "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}
    if rec["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        rec.update(device_count=torch.cuda.device_count(), name=p.name, capability=f"{p.major}.{p.minor}",
                   total_memory_gib=round(p.total_memory / 2**30, 2))
    rec["nvidia_smi"] = _run_text(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                   "--format=csv,noheader"])
    return rec


def env_record() -> dict[str, Any]:
    return {
        "python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "versions": _versions(), "uv": _run_text(["uv", "--version"]),
        "uv_lock_sha256": _sha256_file(QWEN_DIR / "uv.lock"),
        "pyproject_sha256": _sha256_file(QWEN_DIR / "pyproject.toml"),
        "banned_modules_importable": [m for m in BANNED_MODULES if importlib.util.find_spec(m) is not None],
        "gpu": gpu_record(),
    }


def preflight(*, allow_other_gpu: bool = False) -> dict[str, Any]:
    """Refuse unless this is the §3.1 machine: CUDA build of the lock's torch on one H100 80GB."""
    import torch

    if torch.version.cuda is None or not torch.cuda.is_available():
        raise CloudError(f"no CUDA (torch {torch.__version__}, torch.version.cuda={torch.version.cuda}); "
                         "the cloud runner runs on the rented H100 only (the dry run is `plan`)")
    if not str(torch.version.cuda).startswith(EXPECTED_CUDA_PREFIX):
        raise CloudError(f"torch.version.cuda {torch.version.cuda} is not the lock's CUDA {EXPECTED_CUDA_PREFIX} wheel")
    g = gpu_record()
    if not allow_other_gpu and (GPU_NAME not in g["name"] or g["total_memory_gib"] < GPU_MIN_MEMORY_GIB):
        raise CloudError(f"GPU {g['name']} ({g['total_memory_gib']} GiB) is not one H100 80GB (§3.1); "
                         "--allow-other-gpu runs it unofficially")
    check_banned()
    return g


def source_record(root: Path = REPO) -> dict[str, Any]:
    """Git commit of the copy (or SOURCE.json from :func:`bundle`), and the registry's freeze sha256."""
    src: dict[str, Any] = {}
    commit = _run_text(["git", "rev-parse", "HEAD"], cwd=root)
    if commit:
        src = {"git_commit": commit, "from": "git",
               "git_dirty": bool(_run_text(["git", "status", "--porcelain", "--", "envs/qwen"], cwd=root))}
    elif (root / "SOURCE.json").is_file():
        src = {**json.loads((root / "SOURCE.json").read_text()), "from": "SOURCE.json"}
    reg = root / "experiments" / "registry_v0_4.yaml"
    src["registry"] = {"present": reg.is_file()}
    if reg.is_file():
        m = re.search(r"^freeze_sha256:\s*['\"]?([0-9a-f]{64})", reg.read_text(), re.MULTILINE)
        src["registry"].update(file_sha256=_sha256_file(reg), freeze_sha256=m.group(1) if m else None)
    return src


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@dataclass
class Rates:
    """Assumptions of the cost estimate (override on the ``plan`` command line)."""

    readout_s_per_session: float = 1.35  # one val session, one window at a time on an H100 (plan: 1.4-2 h per val pass)
    train_factor: float = 3.0  # a training pass (forward + backward) over a session vs a readout pass
    window_200w_factor: float = 0.5  # a 200w training example is a session cut at its window
    probe_fit_hours: float = 0.05  # per budget below 2048; 2048: 2x, all: 10x (31 float64 L-BFGS fits)
    cpu_replay_hours: float = 0.5  # per arm, 100 sessions on CPU (§3.3)
    parallel: int = 4  # concurrent GPU jobs on the one H100 (the readout is launch-bound)
    parallel_efficiency: float = 0.8
    usd_per_hour: float = 3.0  # on-demand H100 80GB SXM; check the provider's page
    cap_usd: float = 100.0  # §3.1


@dataclass
class Job:
    name: str
    module: str
    args: list[str]
    device: str  # "cuda", "cpu" or "none"
    deps: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()  # relative to the run's out dir
    root: str = "repo"  # "repo" (the copy) or "replay" (the 100-session copy)
    hours: float = 0.0
    copy: str = "required"  # "required", "large" (optional to bring back) or "none"
    threads: int | None = None

    def argv(self) -> list[str]:
        return [self.module, *self.args]


def _readout_hours(sessions: int, r: Rates) -> float:
    return sessions * r.readout_s_per_session / 3600


def _lora_hours(budget: str, r: Rates) -> float:
    cap, pool = EPOCH_CAPS[budget], POOL_SESSIONS[budget]
    per = r.readout_s_per_session * r.train_factor * (r.window_200w_factor if budget == "200w" else 1.0)
    evals = 0 if budget == "200w" else len([e for e in range(1, cap + 1)
                                            if e % -(-cap // MAX_ES_EVALS) == 0 or e == cap])
    return (cap * pool * per + evals * ES_SESSIONS * r.readout_s_per_session) / 3600


def plan_jobs(out: Path, *, wsp_b_archive: Path | None = None, rates: Rates | None = None,
              cpu_threads: int | None = None) -> list[Job]:
    """Every job of a full run, in a dependency-respecting order. Reads no data."""
    r = rates or Rates()
    o = out.resolve()
    val_h, dev_h = _readout_hours(4004, r), _readout_hours(12000, r)
    feats = {s: f"probe/features.{s}.npz" for s in SPLITS}
    jobs = [Job(f"probe-features-{s}", "hypaqwen.probe",
                ["features", "--split", s, "--out", str(o / feats[s]), "--device", "cuda"], "cuda",
                outputs=(feats[s],), hours=dev_h if s == "dev" else val_h, copy="large") for s in SPLITS]
    for b in BUDGETS:
        fit = ["fit", "--budget", b, "--features", str(o / feats["dev"])]
        if b == "200w":  # its 200 training windows are val windows (§7.2)
            fit += ["--val-features", str(o / feats["val"])]
        mult = {"2048": 2, "all": 10}.get(b, 1)
        jobs.append(Job(f"probe-fit-{b}", "hypaqwen.probe", [*fit, "--out", str(o / f"probe/{b}")], "cuda",
                        deps=("probe-features-dev",) + (("probe-features-val",) if b == "200w" else ()),
                        outputs=(f"probe/{b}",), hours=r.probe_fit_hours * mult))
        jobs.append(Job(f"probe-predict-{b}-val", "hypaqwen.probe",
                        ["predict", "--model", str(o / f"probe/{b}"), "--features", str(o / feats["val"]),
                         "--out", str(o / f"columns/probe-{b}.val.npz")], "cuda",
                        deps=(f"probe-fit-{b}", "probe-features-val"), outputs=(f"columns/probe-{b}.val.npz",),
                        hours=0.02))
    for b in BUDGETS:
        for s in SEEDS[b]:
            run = f"lora/{b}/seed{s}"
            jobs.append(Job(f"lora-train-{b}-s{s}", "hypaqwen.lora",
                            ["train", "--budget", b, "--seed", str(s), "--out", str(o / run), "--device", "cuda",
                             "--deterministic"], "cuda", outputs=(run,), hours=_lora_hours(b, r)))
            jobs.append(Job(f"lora-readout-{b}-s{s}-val", "hypaqwen.lora",
                            ["readout", "--run", str(o / run), "--split", "val", "--out",
                             str(o / f"columns/lora-{b}-s{s}.val.npz"), "--device", "cuda"], "cuda",
                            deps=(f"lora-train-{b}-s{s}",), outputs=(f"columns/lora-{b}-s{s}.val.npz",), hours=val_h))
    jobs += replay_jobs(o, wsp_b_archive=wsp_b_archive, rates=r, cpu_threads=cpu_threads)
    return jobs


def replay_jobs(o: Path, *, wsp_b_archive: Path | None, rates: Rates, cpu_threads: int | None) -> list[Job]:
    """§3.3: one CUDA and one CPU readout of the 100 replay sessions per arm, then the agreement report."""
    sub_h = _readout_hours(REPLAY_SESSIONS, rates)
    jobs = [Job("replay-subset", "hypaqwen.cloud", ["replay-subset", "--out", str(o / "replay/root")], "none",
                outputs=("replay/root/SESSIONS.json",), hours=0.01)]
    arms = ["qwen0", "lora-all", "probe-all"] + (["wsp-b"] if wsp_b_archive else [])
    for arm in arms:
        for dev in ("cuda", "cpu"):
            col = f"replay/{arm}.{dev}.npz"
            kw = dict(device=dev, root="replay", hours=sub_h if dev == "cuda" else rates.cpu_replay_hours,
                      threads=cpu_threads if dev == "cpu" else None)
            if arm == "qwen0":
                jobs.append(Job(f"replay-qwen0-{dev}", "hypaqwen.run_readout",
                                ["--split", "val", "--device", dev, "--out", str(o / col)],
                                deps=("replay-subset",), outputs=(col,), **kw))
            elif arm == "wsp-b":
                jobs.append(Job(f"replay-wsp-b-{dev}", "hypaqwen.cloud",
                                ["readout-edit", "--archive", str(Path(wsp_b_archive).resolve()), "--split", "val",
                                 "--device", dev, "--out", str(o / col)],
                                deps=("replay-subset",), outputs=(col,), **kw))
            elif arm == "lora-all":
                jobs.append(Job(f"replay-lora-all-{dev}", "hypaqwen.lora",
                                ["readout", "--run", str(o / "lora/all/seed0"), "--split", "val", "--out", str(o / col),
                                 "--device", dev], deps=("replay-subset", "lora-train-all-s0"), outputs=(col,), **kw))
            else:
                f = f"replay/probe-all.features.{dev}.npz"
                jobs.append(Job(f"replay-probe-all-features-{dev}", "hypaqwen.probe",
                                ["features", "--split", "val", "--out", str(o / f), "--device", dev],
                                deps=("replay-subset",), outputs=(f,), **kw))
                jobs.append(Job(f"replay-probe-all-{dev}", "hypaqwen.probe",
                                ["predict", "--model", str(o / "probe/all"), "--features", str(o / f), "--out",
                                 str(o / col)], device=dev, root="replay", hours=0.01,
                                threads=kw["threads"], deps=(f"replay-probe-all-features-{dev}", "probe-fit-all"),
                                outputs=(col,)))
    report_deps = tuple(j.name for j in jobs if j.name.startswith("replay-") and j.name != "replay-subset"
                        and not j.name.startswith("replay-probe-all-features"))
    full = {"lora-all": "columns/lora-all-s0.val.npz", "probe-all": "columns/probe-all.val.npz"}
    args = ["replay-report", "--out", str(o / "replay/REPLAY.json"), "--replay-dir", str(o / "replay"),
            "--arms", ",".join(arms)]
    for arm, col in full.items():
        args += ["--full-cuda", f"{arm}={o / col}"]
    jobs.append(Job("replay-report", "hypaqwen.cloud", args, "none",
                    deps=report_deps + ("lora-readout-all-s0-val", "probe-predict-all-val"),
                    outputs=("replay/REPLAY.json",), hours=0.01))
    return jobs


def validate_plan(jobs: Sequence[Job]) -> None:
    names = [j.name for j in jobs]
    if len(set(names)) != len(names):
        raise CloudError("duplicate job names")
    seen: set[str] = set()
    for j in jobs:
        if j.module not in JOB_MODULES:
            raise CloudError(f"{j.name}: module {j.module} is not a job module")
        check_args(j.args)
        missing = [d for d in j.deps if d not in seen]
        if missing:
            raise CloudError(f"{j.name}: dependencies {missing} are not earlier jobs")
        seen.add(j.name)


def select(jobs: Sequence[Job], patterns: Sequence[str] | None) -> list[Job]:
    """The jobs matching any pattern (fnmatch), plus everything they depend on, in plan order."""
    if not patterns:
        return list(jobs)
    by = {j.name: j for j in jobs}
    want = {j.name for j in jobs if any(fnmatch.fnmatchcase(j.name, p) for p in patterns)}
    unmatched = [p for p in patterns if not any(fnmatch.fnmatchcase(n, p) for n in by)]
    if unmatched:
        raise CloudError(f"no job matches {unmatched}")
    stack = list(want)
    while stack:
        for d in by[stack.pop()].deps:
            if d not in want:
                want.add(d)
                stack.append(d)
    return [j for j in jobs if j.name in want]


def estimate(jobs: Sequence[Job], r: Rates) -> dict[str, Any]:
    gpu = sum(j.hours for j in jobs if j.device == "cuda")
    cpu = sum(j.hours for j in jobs if j.device == "cpu")
    finish: dict[str, float] = {}
    for j in jobs:
        finish[j.name] = j.hours + max((finish[d] for d in j.deps if d in finish), default=0.0)
    critical = max(finish.values(), default=0.0)
    eff = r.parallel * (r.parallel_efficiency if r.parallel > 1 else 1.0)
    wall = max(critical, gpu / eff, cpu)
    return {"gpu_hours_sequential": round(gpu, 2), "cpu_hours": round(cpu, 2), "critical_path_hours": round(critical, 2),
            "wall_hours": round(wall, 2), "usd": round(wall * r.usd_per_hour, 2),
            "usd_sequential": round(max(gpu, critical) * r.usd_per_hour, 2), "cap_usd": r.cap_usd,
            "over_cap": wall * r.usd_per_hour > r.cap_usd, "rates": asdict(r)}


# ---------------------------------------------------------------------------
# Running jobs
# ---------------------------------------------------------------------------


def outputs_sha256(out: Path, rels: Iterable[str]) -> dict[str, str]:
    """sha256 of every file under each output path (files sorted; a missing output is an error)."""
    got: dict[str, str] = {}
    for rel in rels:
        p = out / rel
        if p.is_file():
            got[rel] = _sha256_file(p)
        elif p.is_dir():
            for f in sorted(x for x in p.rglob("*") if x.is_file()):
                got[f.relative_to(out).as_posix()] = _sha256_file(f)
        else:
            raise CloudError(f"output {rel} was not written")
    return got


def job_record_path(out: Path, name: str) -> Path:
    return out / "jobs" / f"{name}.json"


def is_done(out: Path, job: Job) -> bool:
    p = job_record_path(out, job.name)
    if not p.is_file():
        return False
    rec = json.loads(p.read_text())
    if rec.get("status") != "ok":
        return False
    try:
        return outputs_sha256(out, job.outputs) == rec.get("outputs")
    except CloudError:
        return False


@dataclass
class Scheduler:
    """Runs jobs as processes: at most ``gpu_slots`` CUDA jobs and ``cpu_slots`` other jobs at once.

    ``launch(job) -> handle`` with ``handle.poll() -> int | None``; ``finish(job, returncode)`` records
    the result. A job runs once its dependencies finished ok; a failed or blocked dependency blocks it."""

    jobs: Sequence[Job]
    launch: Callable[[Job], Any]
    finish: Callable[[Job, int], str]
    gpu_slots: int = 1
    cpu_slots: int = 1
    poll_s: float = 5.0
    status: dict[str, str] = field(default_factory=dict)

    def run(self, done: Callable[[Job], bool] = lambda j: False) -> dict[str, str]:
        pending = [j for j in self.jobs if j.name not in self.status]
        for j in pending:
            if done(j):
                self.status[j.name] = "cached"
        pending = [j for j in pending if j.name not in self.status]
        running: dict[str, tuple[Job, Any]] = {}
        while pending or running:
            for name, (job, h) in list(running.items()):
                rc = h.poll()
                if rc is not None:
                    del running[name]
                    self.status[name] = self.finish(job, rc)
            for job in list(pending):
                dep = [self.status.get(d) for d in job.deps]
                if any(s not in (None, "ok", "cached") for s in dep):
                    self.status[job.name] = "blocked"
                    pending.remove(job)
                    continue
                if any(s is None for s in dep):
                    continue  # a dependency has not finished
                gpu = job.device == "cuda"
                busy = sum(1 for j, _ in running.values() if (j.device == "cuda") == gpu)
                if busy >= (self.gpu_slots if gpu else self.cpu_slots):
                    continue
                running[job.name] = (job, self.launch(job))
                pending.remove(job)
            if pending and not running:  # nothing runs and nothing can start: a dependency outside the plan
                for job in pending:
                    self.status[job.name] = "blocked"
                pending = []
            if running:
                time.sleep(self.poll_s)
        return self.status


def _job_cwd(job: Job, out: Path) -> Path:
    return (out / "replay" / "root" / "envs" / "qwen") if job.root == "replay" else QWEN_DIR


def _job_env(job: Job) -> dict[str, str]:
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG, "PYTHONHASHSEED": "0",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    if job.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""  # the replay's CPU side cannot touch the GPU
        if job.threads:
            env["OMP_NUM_THREADS"] = str(job.threads)
    return env


def exec_command(job: Job, out: Path) -> list[str]:
    cmd = [sys.executable, "-m", "hypaqwen.cloud", "exec", "--device", job.device,
           "--record", str(out / "jobs" / f"{job.name}.exec.json")]
    if job.threads:
        cmd += ["--threads", str(job.threads)]
    return [*cmd, "--", *job.argv()]


class _Proc:
    """A job process writing to its log; the log is closed when the process has exited."""

    def __init__(self, cmd: list[str], cwd: Path, env: Mapping[str, str], log: Path):
        self.log = open(log, "ab")
        self.p = subprocess.Popen(cmd, cwd=cwd, env=dict(env), stdout=self.log, stderr=subprocess.STDOUT)

    def poll(self) -> int | None:
        rc = self.p.poll()
        if rc is not None and not self.log.closed:
            self.log.close()
        return rc


def run_jobs(jobs: Sequence[Job], out: Path, *, gpu_slots: int, cpu_slots: int = 1, retry_failed: bool = False,
             poll_s: float = 5.0, on_change: Callable[[], None] | None = None) -> dict[str, str]:
    validate_plan(jobs)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    started: dict[str, float] = {}

    def launch(job: Job) -> _Proc:
        started[job.name] = time.time()
        print(f"[cloud] start {job.name}", flush=True)
        return _Proc(exec_command(job, out), _job_cwd(job, out), _job_env(job), out / "logs" / f"{job.name}.log")

    def finish(job: Job, rc: int) -> str:
        rec: dict[str, Any] = {"name": job.name, "argv": job.argv(), "device": job.device, "root": job.root,
                               "returncode": rc, "seconds": round(time.time() - started[job.name], 1),
                               "estimate_hours": round(job.hours, 3)}
        ex = out / "jobs" / f"{job.name}.exec.json"
        if ex.is_file():
            rec["exec"] = json.loads(ex.read_text())
        status = "ok" if rc == 0 else "failed"
        if status == "ok":
            try:
                rec["outputs"] = outputs_sha256(out, job.outputs)
            except CloudError as e:
                status, rec["error"] = "failed", str(e)
        rec["status"] = status
        _write_json(job_record_path(out, job.name), rec)
        print(f"[cloud] {status} {job.name} ({rec['seconds']} s)", flush=True)
        if on_change:
            on_change()
        return status

    sched = Scheduler(jobs, launch, finish, gpu_slots=gpu_slots, cpu_slots=cpu_slots, poll_s=poll_s)
    if not retry_failed:  # §10: a failed job (e.g. a diverged LoRA run) is reported, not silently re-run
        for j in jobs:
            p = job_record_path(out, j.name)
            if p.is_file() and json.loads(p.read_text()).get("status") == "failed":
                sched.status[j.name] = "failed"
    status = sched.run(lambda j: is_done(out, j))
    return {j.name: status.get(j.name, "blocked") for j in jobs}


def copy_back(out: Path, jobs: Sequence[Job]) -> dict[str, Any]:
    req = ["RUN_MANIFEST.json", "COPY_BACK.json", "jobs", "logs"]
    large: list[str] = []
    for j in jobs:
        for rel in j.outputs:
            if j.copy == "large":
                large.append(rel)
            elif j.copy == "required":
                req.append(rel)
    present = [r for r in req if (out / r).exists()]
    size = {r: sum(f.stat().st_size for f in ([out / r] if (out / r).is_file() else (out / r).rglob("*")) if f.is_file())
            for r in present + [x for x in large if (out / x).exists()]}
    return {"out_dir": str(out), "required": present, "large_optional": [x for x in large if (out / x).exists()],
            "bytes": size, "never": ["replay/root/envs (package copy)", "replay/root/datasets (the subset export)"]}


def write_manifest(out: Path, jobs: Sequence[Job], status: Mapping[str, str], header: Mapping[str, Any]) -> None:
    recs = {}
    for j in jobs:
        p = job_record_path(out, j.name)
        recs[j.name] = json.loads(p.read_text()) if p.is_file() else {"status": status.get(j.name, "pending")}
        recs[j.name]["status"] = status.get(j.name, recs[j.name].get("status"))
    man = {**header, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "jobs": recs}
    if (out / "replay" / "REPLAY.json").is_file():
        man["replay"] = json.loads((out / "replay" / "REPLAY.json").read_text())["summary"]
    _write_json(out / "RUN_MANIFEST.json", man)
    _write_json(out / "COPY_BACK.json", copy_back(out, jobs))


# ---------------------------------------------------------------------------
# §3.3 replay: subset export, readout of an edit, agreement
# ---------------------------------------------------------------------------


def replay_key(session_id: str) -> str:
    return hashlib.sha256(REPLAY_KEY.format(session_id=session_id).encode("utf-8")).hexdigest()


def replay_sessions(session_ids: Iterable[str], n: int = REPLAY_SESSIONS) -> list[str]:
    """§3.3: the ``n`` val ids with the smallest :func:`replay_key`, in key order."""
    ids = list(session_ids)
    if len(set(ids)) != len(ids):
        raise CloudError("duplicate session ids")
    if len(ids) < n:
        raise CloudError(f"only {len(ids)} sessions, need {n}")
    return sorted(ids, key=replay_key)[:n]


def _task_lines(path: Path) -> list[tuple[str, bytes]]:
    with gzip.open(path, "rb") as f:
        return [(json.loads(line)["session_id"], line) for line in f]


def make_replay_root(dest: Path, *, tasks: Path | None = None, package: Path | None = None) -> dict[str, Any]:
    """A copy of the ``hypaqwen`` package whose ``datasets/v0_4/val.tasks.jsonl.gz`` holds only the
    replay sessions' lines, byte for byte and in export order. No labels file."""
    tasks = tasks or V04_DATA / "val.tasks.jsonl.gz"
    package = package or Path(__file__).resolve().parent
    lines = _task_lines(tasks)
    chosen = replay_sessions(sid for sid, _ in lines)
    keep = set(chosen)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(package, dest / "envs" / "qwen" / "hypaqwen", ignore=shutil.ignore_patterns("__pycache__"))
    data = dest / "datasets" / "v0_4"
    data.mkdir(parents=True)
    with open(data / "val.tasks.jsonl.gz", "wb") as fb, gzip.GzipFile(filename="", mode="wb", fileobj=fb,
                                                                       compresslevel=6, mtime=0) as gz:
        for sid, line in lines:
            if sid in keep:
                gz.write(line)
    rec = {"key": REPLAY_KEY, "sessions": chosen, "source_tasks_sha256": _sha256_file(tasks),
           "subset_tasks_sha256": _sha256_file(data / "val.tasks.jsonl.gz"),
           "windows": sum(len(json.loads(line)["window_pos"]) for sid, line in lines if sid in keep)}
    _write_json(dest / "SESSIONS.json", rec)
    return rec


def window_index(tasks: Path) -> dict[str, list[int]]:
    """session id -> window positions, from a task export (label-free)."""
    with gzip.open(tasks, "rt", encoding="utf-8") as f:
        return {d["session_id"]: d["window_pos"] for d in map(json.loads, f)}


def column_argmax(path: Path, windows: Mapping[str, Sequence[int]]) -> dict[tuple[str, int], tuple[int, Any]]:
    """{(session id, row_pos): (first-index argmax, logits)} of a column file, its rows checked against ``windows``."""
    import numpy as np

    with np.load(path) as z:
        logits, sids, pos = z["label_logits"], [str(s) for s in z["session_ids"]], z["row_pos"].tolist()
    keys = [(s, p) for s in sids for p in windows[s]]
    if [p for _, p in keys] != pos or len(keys) != logits.shape[0]:
        raise CloudError(f"{path.name}: rows differ from the task export's windows")
    am = np.argmax(logits, axis=1)
    return {k: (int(a), logits[i]) for i, (k, a) in enumerate(zip(keys, am))}


def agreement(a: Mapping[tuple[str, int], tuple[int, Any]], b: Mapping[tuple[str, int], tuple[int, Any]],
              sessions: Iterable[str] | None = None) -> dict[str, Any]:
    """Window-level argmax agreement of ``a`` and ``b`` on ``sessions`` (default: ``a``'s sessions)."""
    import numpy as np

    want = set(sessions) if sessions is not None else {s for s, _ in a}
    ka = sorted(k for k in a if k[0] in want)
    kb = sorted(k for k in b if k[0] in want)
    if ka != kb:
        raise CloudError("the two columns cover different windows")
    agree = sum(a[k][0] == b[k][0] for k in ka)
    diff = max((float(np.abs(np.asarray(a[k][1], np.float64) - np.asarray(b[k][1], np.float64)).max()) for k in ka),
               default=0.0)
    frac = agree / len(ka) if ka else float("nan")
    return {"windows": len(ka), "agree": agree, "fraction": frac, "max_abs_logit_diff": diff,
            "caveat": bool(frac < AGREEMENT_CAVEAT)}


def replay_report(replay_dir: Path, arms: Sequence[str], full_cuda: Mapping[str, Path], *,
                  mps: Mapping[str, Path] | None = None, tasks: Path | None = None) -> dict[str, Any]:
    tasks = tasks or V04_DATA / "val.tasks.jsonl.gz"
    win = window_index(tasks)
    sess = json.loads((replay_dir / "root" / "SESSIONS.json").read_text())["sessions"]
    out: dict[str, Any] = {"rule": f"§3.3: the {REPLAY_SESSIONS} val sessions with the smallest sha256({REPLAY_KEY!r})",
                           "sessions_sha256": hashlib.sha256("\n".join(sess).encode()).hexdigest(),
                           "caveat_below": AGREEMENT_CAVEAT, "arms": {}}
    for arm in arms:
        cuda = column_argmax(replay_dir / f"{arm}.cuda.npz", win)
        rec = {"cpu_vs_cuda": agreement(column_argmax(replay_dir / f"{arm}.cpu.npz", win), cuda, sess)}
        if arm in full_cuda and Path(full_cuda[arm]).is_file():
            rec["cuda_subset_vs_full_val_column"] = agreement(cuda, column_argmax(Path(full_cuda[arm]), win), sess)
        if mps and arm in mps:
            rec["mps_vs_cuda"] = agreement(column_argmax(Path(mps[arm]), win), cuda, sess)
        out["arms"][arm] = rec
    out["summary"] = {arm: {"cpu_vs_cuda": r["cpu_vs_cuda"]["fraction"], "caveat": r["cpu_vs_cuda"]["caveat"]}
                      for arm, r in out["arms"].items()}
    return out


def readout_edit(archive: Path | None, split: str, device: str, out: Path, *, model_loader: Callable | None = None,
                 tasks: Iterable[Any] | None = None) -> dict[str, Any]:
    """The verbaliser readout of the original weights plus a WSP archive's edit; ``run_readout``'s column format."""
    import numpy as np
    import torch

    from . import interface as iface_mod
    from . import model as qmodel
    from .data import iter_tasks
    from .interface import build_interface
    from .readout import Verbaliser

    check_split(split)
    tok, model = (model_loader or (lambda: qmodel.load(device, "fp32")))()
    edit_sha = None
    if archive is not None:
        from .wsp import runner

        edit = runner.load_edit(archive)
        runner.apply_edit(model, edit)
        edit_sha = json.loads((archive / runner.RECORD_FILE).read_text()).get("edit_state_sha256")
    iface = build_interface(tok)
    verb = Verbaliser(model, iface)
    logits, lse, sids, pos, n_tok = [], [], [], [], 0
    t0 = time.monotonic()
    for task in (tasks if tasks is not None else iter_tasks(split)):
        r = verb.session(iface.block_ids(tok, task.blocks), task.window_pos)
        logits.append(r.label_logits.numpy())
        lse.append(r.logsumexp.numpy())
        sids.append(task.session_id)
        pos.extend(r.pos)
        n_tok += r.tokens
    meta = {"split": split, "limit": None, "device": device, "dtype": "fp32", "torch": torch.__version__,
            "platform": platform.platform(), "attn_implementation": qmodel.ATTN_IMPLEMENTATION,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "file_sha256": qmodel.FILE_SHA256},
            "edit": {"archive": str(archive) if archive else None, "edit_state_sha256": edit_sha},
            "interface": {"name": iface_mod.FROZEN.name, "spec_sha256": iface_mod.spec_sha256()},
            "sessions": len(sids), "tokens": n_tok, "seconds": round(time.monotonic() - t0, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, label_logits=np.concatenate(logits) if logits else np.empty((0, 4), np.float32),
             logsumexp=np.concatenate(lse) if lse else np.empty(0, np.float32), session_ids=np.array(sids),
             row_pos=np.array(pos, dtype=np.int32), meta=json.dumps(meta))
    return meta


# ---------------------------------------------------------------------------
# exec (one job, in its own process)
# ---------------------------------------------------------------------------


def exec_job(module: str, args: Sequence[str], *, device: str, record: Path | None, threads: int | None = None) -> int:
    """Apply §3.1's settings, run ``python -m module args`` in this process, check the settings after."""
    if module not in JOB_MODULES:
        raise CloudError(f"{module} is not a job module {sorted(JOB_MODULES)}")
    if module == "hypaqwen.cloud" and (not args or args[0] not in CLOUD_SUBCOMMANDS):
        raise CloudError(f"exec may run only {sorted(CLOUD_SUBCOMMANDS)} of hypaqwen.cloud")
    check_args(args)
    import torch

    if threads:
        torch.set_num_threads(threads)
    before = apply_determinism()
    if device == "cuda" and not torch.cuda.is_available():
        raise CloudError("a CUDA job without CUDA")
    rec: dict[str, Any] = {"module": module, "args": list(args), "device": device, "settings_before": before,
                           "torch": torch.__version__, "torch_cuda": torch.version.cuda}
    if device == "cuda":
        rec["device_name"] = torch.cuda.get_device_name(0)
    t0 = time.monotonic()
    old_argv = sys.argv
    sys.argv = [module, *args]
    rc = 0
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        sys.argv = old_argv
    rec["settings_after"] = after = settings()
    rec["seconds"] = round(time.monotonic() - t0, 1)
    problems = settings_problems(after)
    if problems:
        rec["settings_changed"] = problems
        rc = rc or 4
    rec["returncode"] = rc
    if record is not None:
        _write_json(record, rec)
    return rc


# ---------------------------------------------------------------------------
# Model fetch and bundle
# ---------------------------------------------------------------------------


def fetch_model() -> Path:
    """Download the pinned snapshot (network) and check every pinned file's sha256."""
    from huggingface_hub import snapshot_download

    from . import model as qmodel

    snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    return qmodel.verify_files(Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)))


def bundle_paths(root: Path = REPO) -> list[str]:
    """Tracked files under ``envs/qwen`` and the registry, plus the present :data:`ALLOWED_DATA` files."""
    out = subprocess.run(["git", "ls-files", "-z", "--", "envs/qwen", "experiments/registry_v0_4.yaml"], cwd=root,
                         capture_output=True, check=True).stdout.decode().split("\0")
    tracked = sorted(p for p in out if p)
    return tracked + [p for p in ALLOWED_DATA if (root / p).is_file() and p not in tracked]


def bundle(dest: Path, root: Path = REPO) -> dict[str, Any]:
    """Write a tar of the repo subset the cloud machine needs, with SOURCE.json (commit, freeze sha)."""
    paths = bundle_paths(root)
    for p in paths:
        if Path(p).parts[0] == "datasets" and p not in ALLOWED_DATA:
            raise CloudError(f"{p}: not an allowed dataset file")
        check_args([p])
    src = source_record(root)
    src["files"] = {p: _sha256_file(root / p) for p in paths}
    body = json.dumps(src, indent=1, sort_keys=True).encode() + b"\n"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w") as tar:
        for p in paths:
            info = tar.gettarinfo(str(root / p), arcname=p)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with open(root / p, "rb") as f:
                tar.addfile(info, f)
        info = tarfile.TarInfo("SOURCE.json")
        info.size, info.mtime = len(body), 0
        tar.addfile(info, io.BytesIO(body))
    return {"files": len(paths), "bytes": dest.stat().st_size, "git_commit": src.get("git_commit")}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _rates(a: argparse.Namespace) -> Rates:
    r = Rates()
    for k in asdict(r):
        v = getattr(a, k, None)
        if v is not None:
            setattr(r, k, v)
    return r


def _add_plan_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", type=Path, default=REPO / "cloud_out")
    p.add_argument("--jobs", nargs="*", default=None, metavar="PATTERN", help="fnmatch patterns (deps are added)")
    p.add_argument("--wsp-b-archive", type=Path, default=None, help="WSP-B archive: adds the WSP-B replay arm")
    p.add_argument("--cpu-threads", type=int, default=None)
    for k, v in asdict(Rates()).items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=None, dest=k)


def _print_plan(jobs: Sequence[Job], est: Mapping[str, Any]) -> None:
    for j in jobs:
        deps = ",".join(j.deps) if j.deps else "-"
        print(f"{j.name:34s} {j.device:5s} {j.hours:6.2f} h  deps={deps}")
        print(f"    python -m {' '.join(j.argv())}")
    print(f"\n{len(jobs)} jobs. Estimate (assumptions in 'rates'):")
    print(json.dumps(est, indent=1))
    if est["over_cap"]:
        print(f"WARNING: estimated US${est['usd']} exceeds the US${est['cap_usd']} cap (§3.1); measure the first "
              "jobs' rates and plan the run in parts before renting for all of it", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hypaqwen.cloud", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="dry run: the job list and a cost estimate (no GPU, no data)")
    _add_plan_args(p)
    p = sub.add_parser("run", help="run the jobs (on the rented H100)")
    _add_plan_args(p)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--allow-other-gpu", action="store_true", help="unofficial run on another GPU")
    p.add_argument("--mps-column", action="append", default=[], metavar="ARM=PATH",
                   help="a local MPS val column (qwen0, wsp-b) for the MPS-vs-CUDA replay comparison")
    p = sub.add_parser("exec")
    p.add_argument("--device", choices=("cuda", "cpu", "none"), required=True)
    p.add_argument("--record", type=Path, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p = sub.add_parser("replay-subset")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", type=Path, default=V04_DATA / "val.tasks.jsonl.gz")
    p = sub.add_parser("replay-report")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--replay-dir", type=Path, required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--full-cuda", action="append", default=[], metavar="ARM=PATH")
    p.add_argument("--mps", action="append", default=[], metavar="ARM=PATH")
    p.add_argument("--tasks", type=Path, default=V04_DATA / "val.tasks.jsonl.gz")
    p = sub.add_parser("readout-edit", help="verbaliser readout of a WSP archive's edit (or the original weights)")
    p.add_argument("--archive", type=Path, default=None)
    p.add_argument("--split", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compare", help="window-level argmax agreement of two column files (label-free)")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--tasks", type=Path, default=V04_DATA / "val.tasks.jsonl.gz")
    p.add_argument("--sessions", type=Path, default=None, help="a SESSIONS.json (default: a's sessions)")
    sub.add_parser("env", help="print the environment record")
    sub.add_parser("fetch-model", help="download the pinned model and check its sha256")
    p = sub.add_parser("bundle", help="(home machine) tar the repo subset for the cloud machine")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        return _dispatch(a)
    except CloudError as e:
        print(f"cloud: {e}", file=sys.stderr)
        return 2


def _pairs(items: Sequence[str]) -> dict[str, Path]:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        if not v:
            raise CloudError(f"{it!r}: want ARM=PATH")
        out[k] = Path(v)
    return out


def _dispatch(a: argparse.Namespace) -> int:
    if a.cmd in ("plan", "run"):
        r = _rates(a)
        jobs = select(plan_jobs(a.out, wsp_b_archive=a.wsp_b_archive, rates=r, cpu_threads=a.cpu_threads), a.jobs)
        validate_plan(jobs)
        if a.cmd == "plan":
            _print_plan(jobs, estimate(jobs, r))
            return 0
        return _run(a, jobs, r)
    if a.cmd == "exec":
        rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
        if not rest:
            raise CloudError("exec: no module")
        return exec_job(rest[0], rest[1:], device=a.device, record=a.record, threads=a.threads)
    if a.cmd == "replay-subset":
        rec = make_replay_root(a.out, tasks=a.tasks)
        print(f"replay: {len(rec['sessions'])} sessions, {rec['windows']} windows -> {a.out}")
        return 0
    if a.cmd == "replay-report":
        rep = replay_report(a.replay_dir, a.arms.split(","), _pairs(a.full_cuda), mps=_pairs(a.mps), tasks=a.tasks)
        _write_json(a.out, rep)
        print(json.dumps(rep["summary"], indent=1))
        return 0
    if a.cmd == "readout-edit":
        print(json.dumps(readout_edit(a.archive, a.split, a.device, a.out), indent=1))
        return 0
    if a.cmd == "compare":
        win = window_index(a.tasks)
        sess = json.loads(a.sessions.read_text())["sessions"] if a.sessions else None
        print(json.dumps(agreement(column_argmax(a.a, win), column_argmax(a.b, win), sess), indent=1))
        return 0
    if a.cmd == "env":
        print(json.dumps({**env_record(), "source": source_record()}, indent=1))
        return 0
    if a.cmd == "fetch-model":
        print(f"verified {MODEL_ID}@{MODEL_REVISION} at {fetch_model()}")
        return 0
    if a.cmd == "bundle":
        print(json.dumps(bundle(a.out), indent=1))
        return 0
    raise CloudError(f"unknown command {a.cmd}")


def _run(a: argparse.Namespace, jobs: list[Job], r: Rates) -> int:
    out = a.out.resolve()
    gpu = preflight(allow_other_gpu=a.allow_other_gpu)  # before anything is read: refuses off the rented H100
    data = check_data_tree(REPO)
    from . import model as qmodel

    qmodel.verify_files()
    s = apply_determinism()
    if a.mps_column:
        mps = _pairs(a.mps_column)
        for j in jobs:
            if j.name == "replay-report":
                for arm, path in mps.items():
                    j.args += ["--mps", f"{arm}={path.resolve()}"]
    header = {"format": FORMAT, "official": not a.allow_other_gpu, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                           time.gmtime()),
              "source": source_record(), "environment": env_record(), "gpu": gpu, "settings": s,
              "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "file_sha256": qmodel.FILE_SHA256, "verified": True},
              "data_sha256": data, "estimate": estimate(jobs, r), "selected": [j.name for j in jobs]}
    out.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}

    def snapshot() -> None:
        write_manifest(out, jobs, status, header)

    status = run_jobs(jobs, out, gpu_slots=r.parallel, retry_failed=a.retry_failed, on_change=snapshot)
    write_manifest(out, jobs, status, header)
    bad = {k: v for k, v in status.items() if v not in ("ok", "cached")}
    print(json.dumps({"status": status, "not_ok": bad}, indent=1))
    print(f"copy back: {out / 'COPY_BACK.json'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
