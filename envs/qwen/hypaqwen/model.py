"""Load Qwen3.5-0.8B (text model only; the vision encoder is not used) at the pinned revision.

Loading is offline from the local Hugging Face cache, and every file the readout depends on is
checked against its sha256 first, so a cloud machine or a re-download cannot silently change the
weights, tokeniser or config. Attention runs through SDPA, fixed rather than left to the default.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

from . import MODEL_ID, MODEL_REVISION

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
ATTN_IMPLEMENTATION = "sdpa"

# sha256 of the files at MODEL_REVISION (the LFS oid for the weights).
FILE_SHA256 = {
    "model.safetensors-00001-of-00001.safetensors": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    "config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_dir() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))


def verify_files(root: Path | None = None) -> Path:
    """Check every pinned file's sha256 in the local snapshot; return the snapshot directory."""
    root = root or snapshot_dir()
    for name, want in FILE_SHA256.items():
        got = _sha256(root / name)
        if got != want:
            raise ValueError(f"{MODEL_ID}@{MODEL_REVISION} {name}: sha256 {got} != pinned {want}")
    return root


def load(device: str = "cpu", dtype: str = "fp32", *, verify: bool = True):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = verify_files() if verify else snapshot_dir()
    tok = AutoTokenizer.from_pretrained(root)
    model = AutoModelForCausalLM.from_pretrained(root, dtype=DTYPES[dtype], attn_implementation=ATTN_IMPLEMENTATION)
    return tok, model.to(device).eval()
