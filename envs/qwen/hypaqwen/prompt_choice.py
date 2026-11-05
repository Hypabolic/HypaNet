"""Label-free choice of the v0.4 interface text (pre-registration: experiments/v0_4/prompt_choice/README.md).

    uv run python -m hypaqwen.prompt_choice --device mps

Every verbaliser arm (Qwen-0, LoRA, WSP) reads the same frozen prompt, so the prompt must not be
chosen with labels, or the zero-label arms would inherit a labelled choice. The rule here looks
at no label, gate or family. For each candidate in :data:`CANDIDATES`, it runs the readout on
:data:`N_SESSIONS` dev sessions picked by hash key, and computes the **label-word mass** at every
window: the probability the model puts on the four label words together,
``exp(logsumexp(label logits) - logsumexp(all logits))``. The candidate with the highest median
mass wins; a tie goes to the earlier candidate. The mass measures whether the model answers in
the requested format. It says nothing about whether the answer is right.

The report also gives each candidate's predicted-class shares (no labels) for the record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers

from . import MODEL_ID, MODEL_REVISION, REPO
from . import model as qmodel
from .data import iter_tasks
from .interface import CLASS_NAMES, TextSpec, build_interface
from .readout import Verbaliser

SPLIT = "dev"
N_SESSIONS = 100
SUBSET_KEY = "v04-prompt-choice:{session_id}"

WORDS = {
    "w1": (" productive", " recovering", " stalled", " looping"),
    "w2": (" productive", " recovering", " stuck", " looping"),
    "w3": (" progress", " retry", " stuck", " loop"),
}
CUES = ("State:", "Answer:")
STYLES = ("full", "short")
CANDIDATES = tuple(
    TextSpec(f"{w}-{c.rstrip(':').lower()}-{s}", words, c, s)
    for w, words in WORDS.items() for c in CUES for s in STYLES
)
REPORT = REPO / "artifacts" / "reports" / "v04_prompt_choice.json"


def subset_ids(all_ids: list[str], n: int = N_SESSIONS) -> list[str]:
    keyed = sorted((hashlib.sha256(SUBSET_KEY.format(session_id=s).encode()).hexdigest(), s) for s in all_ids)
    return [s for _, s in keyed[:n]]


def label_mass(label_logits: np.ndarray, logsumexp: np.ndarray) -> np.ndarray:
    m = label_logits.max(axis=1, keepdims=True)
    lse4 = (m + np.log(np.exp(label_logits - m).sum(axis=1, keepdims=True)))[:, 0]
    return np.exp(lse4 - logsumexp)


def choose(results: list[dict]) -> str:
    """Highest median mass; ties to the earlier candidate (``results`` is in CANDIDATES order)."""
    best = max(range(len(results)), key=lambda i: (results[i]["mass"]["median"], -i))
    return results[best]["name"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", type=Path, default=REPORT)
    a = ap.parse_args(argv)
    tasks = {t.session_id: t for t in iter_tasks(SPLIT)}
    ids = subset_ids(list(tasks))
    tok, model = qmodel.load(a.device, "fp32")
    results = []
    for spec in CANDIDATES:
        iface = build_interface(tok, spec)
        verb = Verbaliser(model, iface)
        t0 = time.monotonic()
        lg, ls, n_tok = [], [], 0
        for sid in ids:
            r = verb.session(iface.block_ids(tok, tasks[sid].blocks), tasks[sid].window_pos)
            lg.append(r.label_logits.numpy())
            ls.append(r.logsumexp.numpy())
            n_tok += r.tokens
        logits, lse = np.concatenate(lg), np.concatenate(ls)
        mass = label_mass(logits, lse)
        pred = np.bincount(logits.argmax(axis=1), minlength=4)
        results.append({
            "name": spec.name, "label_words": list(spec.label_words), "answer_cue": spec.answer_cue, "style": spec.style,
            "label_token_ids": list(iface.label_ids), "prefix_tokens": len(iface.prefix_ids),
            "suffix_tokens": len(iface.suffix_ids), "windows": int(len(mass)),
            "mass": {"median": float(np.median(mass)), "mean": float(mass.mean()), "p10": float(np.percentile(mass, 10)),
                     "p90": float(np.percentile(mass, 90))},
            "predicted_share": {c: round(float(n) / len(mass), 4) for c, n in zip(CLASS_NAMES, pred)},
            "tokens": n_tok, "seconds": round(time.monotonic() - t0, 1),
        })
        print(json.dumps(results[-1]), flush=True)
    report = {
        "report": "v04_prompt_choice",
        "rule": "experiments/v0_4/prompt_choice/README.md: highest median label-word mass on the dev subset; "
                "ties to the earlier candidate. No labels, gate results or families are read.",
        "split": SPLIT, "sessions": len(ids),
        "session_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "subset_key": SUBSET_KEY,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION}, "device": a.device, "dtype": "fp32",
        "torch": torch.__version__, "transformers": transformers.__version__, "platform": platform.platform(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidates": results,
        "chosen": choose(results),
    }
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    print("chosen:", report["chosen"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
