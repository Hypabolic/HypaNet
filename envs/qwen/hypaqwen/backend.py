"""Numerical backend settings of the probe and LoRA runs (PREREGISTRATION §3.1).

On CUDA (the one H100 80GB SXM of every probe, LoRA and sealed run): fp32 with TF32 off in matmul
and cuDNN, ``set_float32_matmul_precision("highest")``, ``use_deterministic_algorithms(True)`` with
``CUBLAS_WORKSPACE_CONFIG=:4096:8``, SDPA restricted to the math backend, and the torch Gated
DeltaNet (``flash-linear-attention`` and ``causal-conv1d`` must not be installed). On CPU (the
replay check and the tests): ``use_deterministic_algorithms(True)``. MPS is left as the harness-only
readouts run it, unless ``deterministic`` is asked for.

:func:`configure` must run before the first CUDA call (the cuBLAS workspace is read at init).
:func:`compute` is the context every model call of these recipes runs in (the SDPA backend), and
:func:`autocast` the LoRA training precision: bf16 autocast on CUDA only, fp32 elsewhere (§5).
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import platform
from typing import Any, Iterator

import torch

__all__ = ["CUBLAS_WORKSPACE", "FORBIDDEN_PACKAGES", "configure", "compute", "autocast", "environment"]

CUBLAS_WORKSPACE = ":4096:8"
FORBIDDEN_PACKAGES = ("fla", "causal_conv1d")  # flash-linear-attention, causal-conv1d: torch DeltaNet only


def _kind(device: str) -> str:
    return torch.device(device).type


def configure(device: str, *, deterministic: bool = False) -> dict[str, Any]:
    """Apply the §3.1 settings for ``device``; return them for the run log."""
    kind = _kind(device)
    if kind == "cuda":
        got = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE)
        if got != CUBLAS_WORKSPACE:
            raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG={got!r}, §3.1 needs {CUBLAS_WORKSPACE!r}")
        present = [p for p in FORBIDDEN_PACKAGES if importlib.util.find_spec(p) is not None]
        if present:
            raise RuntimeError(f"{present} installed; §3.1 runs the torch Gated DeltaNet only")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
    elif kind == "cpu" or deterministic:
        torch.use_deterministic_algorithms(True)
    return environment(device)


@contextlib.contextmanager
def compute(device: str) -> Iterator[None]:
    """SDPA restricted to the math backend on CUDA (deterministic forward and backward)."""
    if _kind(device) == "cuda":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.MATH]):
            yield
    else:
        yield


def autocast(device: str):
    """bf16 autocast for the LoRA forward and backward on CUDA; a no-op (fp32) elsewhere."""
    if _kind(device) == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def environment(device: str) -> dict[str, Any]:
    import transformers

    env: dict[str, Any] = {
        "device": device, "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "threads": torch.get_num_threads(),
    }
    try:
        import peft

        env["peft"] = peft.__version__
    except ImportError:  # pragma: no cover - peft is in uv.lock
        pass
    if _kind(device) == "cuda":
        env.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(torch.device(device)),
                   cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                   allow_tf32_cudnn=torch.backends.cudnn.allow_tf32)
    return env
