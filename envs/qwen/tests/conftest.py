"""Shared fixtures: a tiny random-init Qwen3.5 built from the pinned config, a character tokeniser, synthetic data.

The tiny model keeps the real architecture (Gated DeltaNet layers and a gated full-attention layer,
mrope, tied embeddings, SDPA) with few layers and small dimensions, so recipe tests run on CPU in
seconds. Tests that need the real configuration skip when the pinned snapshot is not in the local
Hugging Face cache. Synthetic labels only: no test here reads a label file of a real split.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from hypaqwen.interface import Interface

TINY = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
            head_dim=64, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
            linear_num_value_heads=4, vocab_size=512, initializer_range=0.2,
            layer_types=["linear_attention"] * 3 + ["full_attention"], eos_token_id=None)
TINY_ROPE_SECTION = [3, 3, 2]  # head_dim 64 * partial_rotary_factor 0.25 = 16 rotary dims -> 8 frequencies


def _real_text_config():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoConfig

    from hypaqwen import model as qmodel

    try:
        root = qmodel.snapshot_dir()
    except Exception as e:  # noqa: BLE001 - no local snapshot
        pytest.skip(f"pinned Qwen snapshot not in the local cache: {type(e).__name__}")
    return AutoConfig.from_pretrained(root)


@pytest.fixture(scope="session")
def real_config():
    """The pinned ``Qwen3_5Config`` (text model inside)."""
    return _real_text_config()


@pytest.fixture(scope="session")
def tiny_config(real_config):
    return tiny_text_config(real_config)


def tiny_text_config(real_config):
    from transformers import AutoConfig

    d = real_config.get_text_config().to_dict()
    d.update(TINY)
    rope = dict(d["rope_parameters"])
    rope["mrope_section"] = TINY_ROPE_SECTION
    d["rope_parameters"] = rope
    d.pop("architectures", None)
    return AutoConfig.for_model(**d)


def make_tiny(cfg, seed: int = 0):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa", dtype=torch.float32)
    return model.eval()


class CharTok:
    """A deterministic stand-in tokeniser: one id per character, inside the tiny vocabulary."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [20 + (ord(c) * 7) % 480 for c in text]


TINY_IFACE = Interface(prefix_ids=tuple(20 + (i * 37) % 480 for i in range(70)),
                       suffix_ids=tuple(20 + (i * 53) % 480 for i in range(9)), label_ids=(5, 6, 7, 8))


@pytest.fixture
def tiny_factory(tiny_config):
    """``loader(device, dtype) -> (tok, model)`` returning the same tiny weights every call."""

    def loader(device: str = "cpu", dtype: str = "fp32"):
        return CharTok(), make_tiny(tiny_config).to(device)

    return loader


def synthetic_sessions(n: int, seed: int = 0, *, prefix: str = "s") -> list[dict]:
    """Task-export-shaped sessions: rendered-looking block text and window endpoints from block 7."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        nb = int(rng.integers(9, 14))
        blocks = [f"00:{k:02d} bash {'x' * int(rng.integers(3, 30))} -> ok {int(rng.integers(1, 99))}/7"
                  for k in range(nb)]
        out.append({"session_id": f"{prefix}{i:04d}", "blocks": blocks, "window_pos": list(range(7, nb))})
    return out


def write_export(data_dir: Path, split: str, sessions: list[dict], seed: int = 0) -> dict[str, np.ndarray]:
    """A synthetic ``{split}.tasks.jsonl.gz`` and ``{split}.labels.npz`` (random labels, some gated or insufficient)."""
    rng = np.random.default_rng(seed + 1)
    data_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(data_dir / f"{split}.tasks.jsonl.gz", "wt", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s, sort_keys=True) + "\n")
    r_sess, r_pos, r_end, r_lab, r_gate = [], [], [], [], []
    for i, s in enumerate(sessions):
        for p in s["window_pos"]:
            r_sess.append(i), r_pos.append(p), r_end.append(p + 1)
            r_lab.append(int(rng.choice(6, p=[0.3, 0.2, 0.2, 0.2, 0.05, 0.05])))
            r_gate.append(bool(rng.random() > 0.1))
    labels = {"session_ids": np.array([s["session_id"] for s in sessions]),
              "families": np.array(["f"] * len(sessions)),
              "row_session": np.array(r_sess, dtype=np.int32), "row_pos": np.array(r_pos, dtype=np.int32),
              "row_endpoint": np.array(r_end, dtype=np.int32), "row_label": np.array(r_lab, dtype=np.int8),
              "row_gate": np.array(r_gate, dtype=bool)}
    np.savez(data_dir / f"{split}.labels.npz", **labels)
    return labels


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the pinned Qwen checkpoint and the dev export (HYPAQWEN_SLOW=1)")


def budgets_doc(dev_ids: list[str], val_ids: list[str], val_labels: dict[str, np.ndarray]) -> dict:
    """A valid ``v04-budgets/1`` document for synthetic splits, built by the rules ``budgets.parse`` checks."""
    from hypaqwen import budgets as B

    full = sorted(dev_ids, key=B.budget_key)
    budgets = {"all": full, **{k: full[:n] for k, n in B.BUDGET_SIZES.items()}}
    disclosed = B.disclosure_rows(val_labels)
    dset = {w.session_id for w in disclosed}
    es = [s for s in sorted(val_ids, key=B.es_key) if s not in dset][: B.ES_SESSIONS]
    in2048 = set(budgets["2048"])
    holdout = sorted([s for s in full if s not in in2048], key=B.holdout_key)[: B.HOLDOUT_SESSIONS]
    windows = [{"session_id": w.session_id, "row_pos": w.row_pos} for w in disclosed]
    lists = {**budgets, "es_val": es, "disclosed_200w": windows, "wspf_holdout": holdout}
    sha = {k: B.canonical_sha256(v) for k, v in lists.items()}  # the writer's convention
    return {"format": B.FORMAT, "budgets": budgets, "es_val": es, "disclosed_200w": windows,
            "wspf_holdout": holdout, "sha256": sha}


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    """Synthetic dev (2,600 sessions) and val (800) exports with random labels, and their BUDGETS.json."""
    root = tmp_path_factory.mktemp("world")
    dev = synthetic_sessions(2600, seed=1, prefix="d")
    val = synthetic_sessions(800, seed=2, prefix="v")
    dev_labels = write_export(root, "dev", dev, seed=1)
    val_labels = write_export(root, "val", val, seed=2)
    doc = budgets_doc([s["session_id"] for s in dev], [s["session_id"] for s in val], val_labels)
    (root / "BUDGETS.json").write_text(json.dumps(doc))
    return {"dir": root, "budgets": root / "BUDGETS.json", "dev": dev, "val": val, "dev_labels": dev_labels,
            "val_labels": val_labels, "doc": doc}
