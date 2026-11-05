#!/usr/bin/env bash
# v0.4 cloud GPU runner (experiments/v0_4/PREREGISTRATION.md §3.1, §3.3): probe features and fits,
# LoRA training, val column readouts and the CPU replay check, on ONE NVIDIA H100 80GB SXM (RunPod,
# Modal or any provider: nothing here is provider-specific). The sealed scoring is not here
# (scripts/v04_sealed_eval.py, after G2); the runner refuses any split but dev and val.
#
# On the home machine (Mac):
#   envs/qwen/scripts/cloud_run.sh --dry-run [plan options]   # the job list and a cost estimate
#   envs/qwen/scripts/cloud_run.sh --bundle OUT.tar           # the repo subset to copy (no sealed file)
# On the rented machine, after unpacking OUT.tar into DIR:
#   DIR/envs/qwen/scripts/cloud_run.sh [--out DIR/cloud_out] [--jobs PATTERN ...] [--parallel N] [--wsp-b-archive A]
#
# Environment. `uv sync --frozen` from envs/qwen/uv.lock, with uv pinned below. No wheel override is
# needed: the lock's torch 2.14.1 for linux x86_64 is PyPI's manylinux_2_28_x86_64 wheel (index
# https://pypi.org/simple, sha256 23011fe29a99b591eabb3a31a25080c62e2c2f0690a3c798744e5489dae50451),
# which is the CUDA 13.0 build (its locked dependencies: cuda-toolkit 13.0.3.0, nvidia-cudnn-cu13
# 9.24.0.43, nvidia-nccl-cu13, triton 3.8.0). It needs an NVIDIA driver for CUDA 13.0 (R580 or
# later), checked in step 1 before anything is downloaded. The runner records torch.version.cuda,
# cuDNN, the device name and the driver, and refuses a torch without CUDA 13.0, anything but one
# H100 80GB SXM ("H100 80GB HBM3"; PCIe and NVL are refused unless --allow-other-gpu, which marks
# the run unofficial), and an importable flash-linear-attention or causal-conv1d.
#
# Cost (§3.1, US$100 cap): `run` refuses when spent + the estimate of the jobs left exceeds the cap
# (--accept-estimate-over-cap starts anyway), and stops launching jobs once spent + its elapsed
# hours x --usd-per-hour reaches it (the rest end as not_run_cap). Spent is the out dir's SPEND.json
# ledger, or --spent-usd (the provider's figure, which also counts setup time).
#
# Settings (§3.1): CUBLAS_WORKSPACE_CONFIG=:4096:8 here; every job then runs through
# `python -m hypaqwen.cloud exec`, which sets torch.use_deterministic_algorithms(True), TF32 off,
# float32 matmul precision "highest", cuDNN benchmark off and SDPA math-only before the job's code
# and checks them after. The model is fetched at the pinned revision and every pinned file's sha256
# is checked before any job.
set -euo pipefail
die() { echo "cloud_run: refusing: $*" >&2; exit 2; }
MIN_DRIVER_MAJOR=580
check_driver() {  # $1: nvidia-smi's driver_version, e.g. 580.65.06
  local major="${1%%.*}"
  [[ "$major" =~ ^[0-9]+$ ]] || die "cannot read the NVIDIA driver version (got '$1')"
  (( major >= MIN_DRIVER_MAJOR )) || die "driver $1 < $MIN_DRIVER_MAJOR: the lock's torch 2.14.1 is the CUDA 13.0 build"
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # envs/qwen
UV_VERSION=0.12.23
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false

case "${1:-}" in
  --dry-run)
    shift
    cd "$HERE"
    # PYTHON lets a test run the plan with an interpreter it already has.
    if [[ -n "${PYTHON:-}" ]]; then exec "$PYTHON" -m hypaqwen.cloud plan "$@"; fi
    exec uv run --frozen python -m hypaqwen.cloud plan "$@"
    ;;
  --bundle)
    shift
    [[ $# -eq 1 ]] || die "usage: $0 --bundle OUT.tar"
    OUT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    cd "$HERE"
    exec uv run --frozen python -m hypaqwen.cloud bundle --out "$OUT"
    ;;
  --check-driver)  # (tests) only the driver rule of step 1
    check_driver "${2:-}"
    echo "driver $2 ok"
    exit 0
    ;;
  -h|--help)
    sed -n '2,32p' "${BASH_SOURCE[0]}"
    exit 0
    ;;
esac

# 1. The machine: Linux x86_64 with an NVIDIA GPU.
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "run on the rented Linux x86_64 GPU machine (or use --dry-run)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
check_driver "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')"

# 2. uv at the pinned version.
if ! command -v uv >/dev/null || [[ "$(uv --version | awk '{print $2}')" != "$UV_VERSION" ]]; then
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[[ "$(uv --version | awk '{print $2}')" == "$UV_VERSION" ]] || die "uv $UV_VERSION not installed"

# 3. The environment, exactly from the lock (no resolution, no upgrade).
cd "$HERE"
uv python install 3.12
uv sync --frozen
uv run --frozen python - <<'EOF'
import torch
print(f"torch {torch.__version__} cuda {torch.version.cuda} cudnn {torch.backends.cudnn.version()} "
      f"available {torch.cuda.is_available()} device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
EOF

# 4. The model at the pinned revision (network), every pinned file's sha256 checked; then offline.
HF_HUB_OFFLINE=0 uv run --frozen python -m hypaqwen.cloud fetch-model
export HF_HUB_OFFLINE=1

# 5. The jobs.
exec uv run --frozen python -m hypaqwen.cloud run "$@"
