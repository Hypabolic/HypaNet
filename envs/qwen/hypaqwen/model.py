"""Load Qwen3.5-0.8B (text model only; the vision encoder is not used) at the pinned revision."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import MODEL_ID, MODEL_REVISION

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def load(device: str = "cpu", dtype: str = "fp32"):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=DTYPES[dtype])
    return tok, model.to(device).eval()
