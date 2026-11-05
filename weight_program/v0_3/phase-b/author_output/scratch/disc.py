"""Disclosure loader: windows, tokenised prefixes (marker removed), block texts and the window range."""
import json
import os

import tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
PATH = os.path.join(ROOT, "disclosure", "DISCLOSURE.json")
CLASSES = ("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING")


def load():
    d = json.load(open(PATH))
    out = []
    for w in d["windows"]:
        lines = w["prefix"].split("\n")
        marker = None
        kept = []
        for ln in lines:
            if ln == d["window_marker"]:
                marker = len(kept)
                continue
            kept.append(ln)
        text = "\n".join(kept)
        # blocks: a header line starts with HH:MM:SS.mmm or "?" (missing timestamp); body lines start with 2 spaces
        blocks = []
        for ln in kept:
            if ln.startswith("  "):
                blocks[-1] += "\n" + ln
            else:
                blocks.append(ln)
        # first block of the window = the block whose header line index is `marker`
        hdr_idx = [i for i, ln in enumerate(kept) if not ln.startswith("  ")]
        start = hdr_idx.index(marker) if marker is not None else 0
        enc = tokenizer.encode_session(blocks)
        out.append({
            "id": w["id"], "label": w["oracle_label"], "net": w["network_class"], "pol": w["reference_policy_class"],
            "blocks": blocks, "start": start, "enc": enc, "text": text,
        })
    return out
