"""F11 (post Figure 16): "Three neurons and a word", the WSP-A / WSP-B readout circuit inside Qwen3.5-0.8B.

Run from the repo root:  uv run python scripts/blog/figures/f11.py

Reads only Fable's committed author notes (EDIT.md, REVISION.md) and the committed edit constants (the finished
vectors the build writes into the model). Every formula string shown in the figure is checked to appear verbatim in
its source note; every wire scale is read from the norm of the committed down-projection column and checked against
the scale the note states. No evaluator, scorer or dataset is opened.
"""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F11.json"
GENERATED_BY = "scripts/blog/figures/f11.py"

WP = ROOT / "weight_program" / "v0_4"
A_EDIT = WP / "wsp-a" / "inputs" / "EDIT.md"
A_CONSTS = WP / "wsp-a" / "inputs" / "edit_constants.safetensors"
A_BUILD = WP / "wsp-a" / "inputs" / "build_edit.py"
B_EDIT = WP / "wsp-b" / "inputs" / "EDIT.md"
B_REV = WP / "wsp-b" / "inputs" / "REVISION.md"
B_CONSTS = WP / "wsp-b" / "inputs" / "edit_constants.safetensors"
B_BUILD = WP / "wsp-b" / "inputs" / "build_edit.py"
A_MECH = WP / "wsp-a" / "MECHANISMS.json"
B_MECH = WP / "wsp-b" / "MECHANISMS.json"
SOURCES = [A_EDIT, A_CONSTS, A_BUILD, A_MECH, B_EDIT, B_REV, B_CONSTS, B_BUILD, B_MECH]
COMMITS = {"wsp-a": "5f58e4a", "wsp-b": "d177712"}

LABELS = ("productive", "recovering", "stalled", "looping")  # LABEL_IDS order in build_edit.py
LABEL_IDS = (25399, 39923, 65552, 60477)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def load_safetensors(p: Path) -> dict[str, np.ndarray]:
    """Minimal reader (header + raw little-endian arrays), so the export needs no extra package."""
    raw = p.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    dtypes = {"F32": np.float32, "F64": np.float64, "F16": np.float16}
    out = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        out[key] = np.frombuffer(raw[8 + n + s:8 + n + e], dtype=dtypes[meta["dtype"]]).reshape(meta["shape"])
    return out


def norm_text(s: str) -> str:
    return " ".join(s.replace("`", "").split())


def quoted(path: Path, needle: str) -> str:
    """Return `needle`, after checking it appears (whitespace and backticks normalised) in `path`."""
    if norm_text(needle) not in norm_text(path.read_text()):
        raise AssertionError(f"{rel(path)} does not contain: {needle!r}")
    return needle


def r(x: float, dp: int = 4) -> float:
    return round(float(x), dp)


# ---------------------------------------------------------------------------
# content
# ---------------------------------------------------------------------------

# Feature directions: mean matched-pair differences of the readout residual, standardised (z-scores).
# "meaning" paraphrases the source bullet; "quote" is the checked source wording.
FEATURES = [
    ("ERR", "errors present and recent", A_EDIT, "errors present and recent (ERR)", ["a"]),
    ("RES", "a success right after errors", A_EDIT, "a success right after errors (RES)", ["a", "b"]),
    ("PROG", "improving validation, then a pass", A_EDIT, "improving validation followed by a pass (PROG)", ["a", "b"]),
    ("REP", "identical failing blocks repeat", A_EDIT,
     "identical failing blocks repeating (REP, saturating after two repeats)", ["a", "b"]),
    ("EDITS", "edits among repeated failures", A_EDIT, "edit blocks present among repeated failures (EDITS)", ["a", "b"]),
    ("EDITREPfree", "edits repeat without errors", A_EDIT, "edit repetition without errors (EDITREPfree)", ["a", "b"]),
    ("CHAIN", "long poll or retry chains", A_EDIT, "long versus short poll/retry chains (CHAIN)", ["a", "b"]),
    ("EDITCOUNT", "several edits, not one", A_EDIT,
     "several edits versus one edit among repeated failures (EDITCOUNT)", ["a", "b"]),
    ("NFR", "reads or searches present", B_EDIT, "reads/searches/list_dir present in the window", ["b"]),
    ("REPF", "a failing test repeats", B_EDIT, "identical failing test / http error / pending poll repeated", ["b"]),
    ("REPH", "an http error repeats", B_EDIT, "identical failing test / http error / pending poll repeated", ["b"]),
    ("REPP", "a pending poll repeats", B_EDIT, "identical failing test / http error / pending poll repeated", ["b"]),
    ("PEND", "pending polls present", B_EDIT, "pending polls present vs plans/http in their place", ["b"]),
    ("RESH", "http retries, then ok (weak)", B_EDIT, "http retries then ok vs retries continuing (weak, consistent)", ["b"]),
]


def term(coef: float, feature: str) -> dict:
    return {"coef": coef, "feature": feature}


def neurons_a(consts: dict) -> list[dict]:
    return [
        {
            "id": "L21", "layer": 21, "unit": 1007, "cls": "RECOVERY", "logit": "recovering", "status": "edited",
            "gate": {"slope": 3, "threshold": 0.0, "terms": [term(1, "ERR")]},
            "up": {"terms": [term(1, "RES")], "const": 0.9},
            "formula": quoted(A_EDIT, "silu(3 z_ERR) · (z_RES + 0.9)"),
            "wire_scale": wire(consts, 21, 0.4),
            "wire_note": quoted(A_EDIT, "writes `0.4 · act · u_recovering`"),
            "effect": "lowers ' recovering' by 2 to 5 logits on stalled windows; raises it by 1 to 3 just after a resolution",
            "effect_split": "Fable's own sessions",
            "effect_quote": quoted(A_EDIT, "lowers `recovering` by 2 to 5 logits on stalled windows"),
        },
        {
            "id": "L22", "layer": 22, "unit": 1613, "cls": "STALLED", "logit": "stalled", "status": "edited",
            "gate": {"slope": 3, "threshold": 2.5, "terms": [term(1, "REP"), term(2, "CHAIN")]},
            "up": {"terms": [term(-1, "RES"), term(-1, "PROG"), term(1, "CHAIN"), term(-2, "EDITS"), term(-2, "EDITREPfree")],
                   "const": 0.0},
            "formula": quoted(A_EDIT, "silu(3 (r − 2.5)) · q"),
            "wire_scale": wire(consts, 22, 0.045),
            "wire_note": quoted(A_EDIT, "writes `0.045 · act · u_stalled_null`"),
            "shielded": True,
            "effect": "+4 to +10 logits on ' stalled' on windows of five or more identical failing blocks",
            "effect_split": "Fable's own sessions",
            "effect_quote": quoted(A_EDIT, "(+4 to +10 logits on `stalled`)"),
        },
        {
            "id": "L23", "layer": 23, "unit": 1046, "cls": "OSCILLATING", "logit": "looping", "status": "edited",
            "gate": {"slope": 3, "threshold": 0.5, "terms": [term(1, "EDITS"), term(1, "EDITREPfree")]},
            "up": {"terms": [term(-1, "PROG"), term(-1, "RES"), term(1, "REP"), term(1, "CHAIN"), term(1.5, "EDITS"),
                             term(1.5, "EDITREPfree"), term(1, "EDITCOUNT")], "const": -2.5},
            "formula": quoted(A_EDIT, "silu(3 (e − 0.5)) · (T + 1.5 e + 1.0 z_EDITCOUNT − 2.5)"),
            "wire_scale": wire(consts, 23, 0.14),
            "wire_note": quoted(A_EDIT, "writes `0.14 · act · u_looping`"),
            "effect": "+5 to +30 logits on ' looping' on edit/revert and A/B edit cycles",
            "effect_split": "Fable's own sessions",
            "effect_quote": quoted(A_EDIT, "(+5 to +30 logits on `looping`)"),
        },
    ]


def neurons_b(consts: dict) -> list[dict]:
    if any(k.endswith("L21") for k in consts):
        raise AssertionError("WSP-B constants still carry a layer-21 vector")
    return [
        {
            "id": "L21", "layer": 21, "unit": 1007, "cls": "RECOVERY", "logit": "recovering", "status": "reverted",
            "note": quoted(B_EDIT, "Layer 21 (edited in WSP-A) is back at the original weights."),
            "reason": "it lowered ' recovering' on recovery windows (median −0.3 logits; disclosed windows, in-sample)",
            "reason_quote": quoted(B_REV, "The WSP-A RECOVERY neuron lowered ` recovering` on recovery windows (median −0.3)"),
        },
        {
            "id": "L22", "layer": 22, "unit": 1613, "cls": "STALLED", "logit": "stalled", "status": "edited",
            "gate": {"slope": 3, "threshold": 3.478,
                     "terms": [term(1, "REP"), term(2, "CHAIN"), term(1, "REPF"), term(-1, "PEND"), term(-0.5, "REPH")]},
            "up": {"terms": [term(-1, "RES"), term(-1, "PROG"), term(1, "CHAIN"), term(-2, "EDITS"), term(-2, "EDITREPfree"),
                             term(-1, "NFR"), term(-1, "PEND"), term(-0.5, "REPH"), term(-0.5, "REPP")], "const": 1.144},
            "formula": quoted(B_EDIT, "silu(3 (r − 3.478)) · (−z_RES − z_PROG + z_CHAIN − 2 z_EDITS − 2 z_EDITREPfree − z_NFR − z_PEND − 0.5 z_REPH − 0.5 z_REPP + 1.144)"),
            "wire_scale": wire(consts, 22, 0.1055),
            "wire_note": quoted(B_EDIT, "writes `0.1055 · act · u_stalled_null`"),
            "shielded": True,
            "effect": "the revision as run added +18.1 logits (median) on disclosed STALLED windows, against a designed +5; rescaled ×0.2764",
            "effect_split": "disclosed windows, in-sample",
            "effect_quote": quoted(B_REV, "median gain of +18.1 on the disclosed STALLED windows (designed: +5)"),
        },
        {
            "id": "L23", "layer": 23, "unit": 1046, "cls": "OSCILLATING", "logit": "looping", "status": "edited",
            "gate": {"slope": 3, "threshold": 1.642, "terms": [term(1, "EDITS"), term(1, "EDITREPfree")]},
            "up": {"terms": [term(1.5, "EDITS"), term(1.5, "EDITREPfree"), term(1, "EDITCOUNT"), term(-1, "NFR")], "const": -2.433},
            "formula": quoted(B_EDIT, "silu(3 (e − 1.642)) · (1.5 e + z_EDITCOUNT − z_NFR − 2.433)"),
            "wire_scale": wire(consts, 23, 0.814),
            "wire_note": quoted(B_EDIT, "writes `0.814 · act · u_looping`"),
            "effect": "+12.8 logits on ' looping' on average on OSCILLATING windows, +2.4 on PRODUCTIVE ones (revision as run)",
            "effect_split": "disclosed windows, in-sample",
            "effect_quote": quoted(B_EDIT, "it adds +12.8 logits on average to `looping` on OSCILLATING windows and +2.4 on PRODUCTIVE ones"),
        },
    ]


def label_rows(consts: dict, state: str) -> list[dict]:
    deltas = consts["label_row_deltas"]
    changed = [LABELS[i] for i in range(4) if np.any(deltas[i] != 0)]
    if state == "a":
        assert changed == ["productive"], changed
        return [{
            "logit": "productive", "token_id": LABEL_IDS[0], "const": 0.0,
            "terms": [term(1.6, "PROG"), term(1.6, "RES"), term(-1.6, "REP")],
            "formula": quoted(A_EDIT, "1.6 (z_PROG + z_RES − z_REP)"),
            "effect": "+4 logits (median) on productive windows, −6 on stalled ones",
            "effect_split": "Fable's own sessions",
            "effect_quote": quoted(A_EDIT, "+4 logits (median) on productive windows, −6 on stalled ones"),
        }]
    assert changed == ["productive", "recovering"], changed
    return [
        {
            "logit": "productive", "token_id": LABEL_IDS[0], "const": 4.2,
            "terms": [term(1.5, "NFR"), term(-1.0, "REPF"), term(-1.0, "REPH"), term(-1.0, "REPP")],
            "terms_note": "REPX = REPF + REPH + REPP",
            "formula": quoted(B_EDIT, "Δ_p = 4.2 + 1.5 z_NFR − 1.0 z_REPX"),
            "effect": "+4.2 is the measured average deficit of ' productive' below the best other logit",
            "effect_split": "disclosed windows, base Qwen, in-sample",
            "effect_quote": quoted(B_REV, "its ` productive` logit is 4.2 below the best other logit on average, on every class"),
        },
        {
            "logit": "recovering", "token_id": LABEL_IDS[1], "const": 0.0,
            "terms": [term(2, "PEND"), term(2, "REPH"), term(2, "REPP"), term(-1.5, "REPF"), term(0.5, "NFR"), term(0.5, "RESH")],
            "formula": quoted(B_EDIT, "Δ_r = 2 z_PEND + 2 z_REPH + 2 z_REPP − 1.5 z_REPF + 0.5 z_NFR + 0.5 z_RESH"),
            "effect": "replaces the layer-21 neuron: a gated RECOVERY neuron could not be separated from the STALLED gate",
            "effect_split": "Fable's offline designs b1-b4",
            "effect_quote": quoted(B_REV, "could not be separated from the STALLED gate, so a linear term is used"),
        },
    ]


def wire(consts: dict, layer: int, stated: float) -> float:
    """Wire scale = norm of the committed down-projection column; must match the scale the note states."""
    v = float(np.linalg.norm(consts[f"down_col_L{layer}"].astype(np.float64)))
    if abs(v - stated) > 5e-4:
        raise AssertionError(f"layer {layer}: down_col norm {v} differs from stated scale {stated}")
    return r(v)


def build() -> dict:
    ca, cb = load_safetensors(A_CONSTS), load_safetensors(B_CONSTS)
    for p in (A_BUILD, B_BUILD):
        quoted(p, "LABEL_IDS = (25399, 39923, 65552, 60477)  # productive, recovering, stalled, looping")

    features = [{"id": f, "meaning": m, "source": rel(p), "quote": quoted(p, q), "states": s} for f, m, p, q, s in FEATURES]

    rescale = {
        "factor": 0.2764,
        "scale_before": 0.382,
        "scale_after": wire(cb, 22, 0.1055),
        "measured_median_gain": 18.1,
        "designed_gain": 5,
        "split": "disclosed windows (200, 50 per class), revision ac483550 as run, in-sample",
        "quote": quoted(B_REV, "STALLED wire rescaled by 0.2764"),
        "quote2": quoted(B_EDIT, "the scale was first set by the set-R rule (0.382)"),
    }
    assert abs(rescale["scale_before"] * rescale["factor"] - rescale["scale_after"]) < 5e-4

    shield = {
        "text": "the ' stalled' readout component, projected into the null space of layer 23's value projection",
        "quote": quoted(A_EDIT, "projected into the null space of `v_proj_23 ⊙ (1 + input_layernorm_23)`"),
        "why": quoted(A_EDIT, "so that if the unit fires at log positions, layer 23's attention cannot carry it to the readout through values"),
    }

    return {
        "_provenance": {
            "source": [rel(p) for p in SOURCES],
            "sha256": {rel(p): sha256(p) for p in SOURCES},
            "commits": COMMITS,
            "git_commit": git_head(),
            "generated_by": GENERATED_BY,
        },
        "figure": "F11",
        "title": "Three neurons and a word",
        "readout": "prediction = highest of the four label-word logits at the 'State:' position",
        "labels": [{"logit": l, "token_id": t} for l, t in zip(LABELS, LABEL_IDS)],
        "features": features,
        "states": {
            "a": {"name": "WSP-A", "edit_hash": "74576466", "labels_seen": 0, "commit": COMMITS["wsp-a"],
                  "neurons": neurons_a(ca), "label_rows": label_rows(ca, "a")},
            "b": {"name": "WSP-B", "edit_hash": "0ea2eb77", "labels_seen": 200, "commit": COMMITS["wsp-b"],
                  "neurons": neurons_b(cb), "label_rows": label_rows(cb, "b")},
        },
        "stalled_rescale": rescale,
        "shield": shield,
        "notes": [
            "Every neuron is SwiGLU: activation = silu(slope · (gate − threshold)) · up, with gate and up weighted sums of "
            "standardised feature directions read at the input of the host MLP.",
            "Each wire writes along the component of its label row orthogonal to the other three rows, so it moves one "
            "label logit only (direct path).",
            "Effects marked 'Fable's own sessions' are from texts Fable wrote and labelled itself; 'disclosed windows' are "
            "the 200 labelled val windows WSP-B saw (in-sample).",
        ],
    }


def main() -> None:
    obj = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{rel(OUT)} {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    sys.exit(main())
