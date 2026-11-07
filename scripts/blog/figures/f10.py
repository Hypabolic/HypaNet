"""F10 (post Figure 15): "10,240 in 752 million", where WSP-A and WSP-B touch Qwen3.5-0.8B.

Run from the repo root:  uv run python scripts/blog/figures/f10.py

Reads only committed files: the two BUILD_RECORD.json files (per-tensor change counts, ranks, shapes and the
language-model parameter count), the two build_edit.py files (which unit each MLP edit rewrites), the two
edit_constants.safetensors files (the finished vectors written into the model; one 1,024-entry row is exported
in full for the barcode) and the handed-over Qwen config.json (layer count and layer types). No dataset,
evaluator or scorer is opened.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F10.json"
GENERATED_BY = "scripts/blog/figures/f10.py"

WP = ROOT / "weight_program" / "v0_4"
PHASES = {
    "wsp-a": {"commit": "5f58e4a", "name": "WSP-A"},
    "wsp-b": {"commit": "d177712", "name": "WSP-B"},
}
CONFIG = ROOT / "experiments" / "v0_4" / "packet_handed_over" / "qwen" / "config.json"
PREFIX = "model.language_model."
LABEL_WORDS = (" productive", " recovering", " stalled", " looping")
LABEL_IDS = (25399, 39923, 65552, 60477)
UNIT_CLASS = {23: "OSCILLATING", 22: "STALLED", 21: "RECOVERY"}  # MECHANISMS.json rows, both phases
BARCODE = ("gate_row_L23", 23, "gate_proj")  # the vector shown in full at the last zoom step
DP = 5


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def load_safetensors(p: Path) -> dict[str, np.ndarray]:
    """Minimal reader (header + raw little-endian arrays), so the export needs no extra package."""
    b = p.read_bytes()
    n = struct.unpack("<Q", b[:8])[0]
    header = json.loads(b[8:8 + n])
    out = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        s, e = v["data_offsets"]
        dt = {"F32": "<f4", "F16": "<f2"}[v["dtype"]]
        out[k] = np.frombuffer(b[8 + n + s:8 + n + e], dtype=dt).reshape(v["shape"])
    return out


def neurons(build_py: Path) -> dict[int, int]:
    m = re.search(r"NEURON = \{([^}]*)\}", build_py.read_text())
    if not m:
        raise SystemExit(f"F10: no NEURON map in {rel(build_py)}")
    return {int(a): int(b) for a, b in re.findall(r"(\d+)\s*:\s*(\d+)", m.group(1))}


def tensor_entry(name: str, t: dict, units: dict[int, int]) -> dict:
    short = name.removeprefix(PREFIX).removesuffix(".weight")
    entry = {
        "name": short,
        "shape": t["shape"],
        "numel": t["numel"],
        "changed_entries": t["changed_entries"],
        "changed_rows": t["changed_rows"],
        "changed_cols": t["changed_cols"],
        "delta_rank": t["delta_rank"],
        "rel_frobenius": round(float(t["rel_frobenius"]), DP),
    }
    m = re.match(r"layers\.(\d+)\.mlp\.(gate|up|down)_proj", short)
    if m:
        layer, kind = int(m.group(1)), m.group(2)
        unit = units[layer]
        entry.update({"kind": "mlp", "layer": layer, "matrix": f"{kind}_proj", "unit": unit,
                      "unit_class": UNIT_CLASS[layer],
                      "along": "column" if kind == "down" else "row"})
        # A row edit changes one row (all 1,024 columns); a down_proj edit changes one column.
        want = (1024, 1) if kind == "down" else (1, 1024)
        if (t["changed_rows"], t["changed_cols"]) != want:
            raise SystemExit(f"F10: {short} changed {t['changed_rows']}x{t['changed_cols']}, expected {want}")
    elif short == "embed_tokens":
        ids = t["changed_label_ids"]
        entry.update({"kind": "embedding", "changed_label_ids": ids,
                      "changed_label_words": [LABEL_WORDS[LABEL_IDS.index(i)] for i in ids]})
    else:
        raise SystemExit(f"F10: unexpected tensor {name}")
    return entry


def build() -> dict:
    cfg = json.loads(CONFIG.read_text())
    text = cfg.get("text_config", cfg)
    sources = [CONFIG]
    edits = []
    for phase, meta in PHASES.items():
        rec_p = WP / phase / "BUILD_RECORD.json"
        build_p = WP / phase / "inputs" / "build_edit.py"
        consts_p = WP / phase / "inputs" / "edit_constants.safetensors"
        sources += [rec_p, build_p, consts_p]
        rec = json.loads(rec_p.read_text())
        units = neurons(build_p)
        tensors = [tensor_entry(k, t, units) for k, t in rec["tensors"].items()]
        overall = rec["overall"]
        if sum(t["changed_entries"] for t in tensors) != overall["changed_entries"]:
            raise SystemExit(f"F10: {phase} per-tensor counts do not sum to overall")
        if len(tensors) != overall["tensors_edited"]:
            raise SystemExit(f"F10: {phase} tensor count mismatch")
        consts = load_safetensors(consts_p)
        key, layer, matrix = BARCODE
        vec = consts[key].astype(np.float64)
        if vec.shape != (1024,):
            raise SystemExit(f"F10: {key} has shape {vec.shape}")
        edits.append({
            "id": phase,
            "name": meta["name"],
            "commit": meta["commit"],
            "edit_state_sha256": rec["edit_state_sha256"],
            "changed_entries": overall["changed_entries"],
            "tensors_edited": overall["tensors_edited"],
            "changed_fraction": overall["changed_fraction"],
            "text_model_params": overall["text_model_params"],
            "max_delta_rank": max(t["delta_rank"] for t in tensors),
            "units": {str(k): v for k, v in sorted(units.items())},
            "tensors": tensors,
            "barcode": {
                "constant": key,
                "tensor": f"layers.{layer}.mlp.{matrix}",
                "layer": layer,
                "unit": units[layer],
                "unit_class": UNIT_CLASS[layer],
                "values": [round(float(v), DP) for v in vec],
                "max_abs": round(float(np.abs(vec).max()), DP),
            },
        })
    params = {e["text_model_params"] for e in edits}
    if len(params) != 1:
        raise SystemExit("F10: parameter counts differ between phases")
    return {
        "_provenance": {
            "source": [rel(p) for p in sources],
            "sha256": {rel(p): sha256(p) for p in sources},
            "source_commits": {"wsp-a": PHASES["wsp-a"]["commit"], "wsp-b": PHASES["wsp-b"]["commit"]},
            "git_commit": git_head(),
            "generated_by": GENERATED_BY,
            "notes": [
                "changed_entries, delta_rank, rel_frobenius and text_model_params are copied from BUILD_RECORD.json",
                "barcode values: the finished layer-23 gate_proj row written by build_edit.py (edit_constants "
                "gate_row_L23), rounded to 5 decimal places",
                "text_model_params counts model.language_model.* (vision encoder and MTP head excluded)",
            ],
        },
        "figure": "F10",
        "model": {
            "id": "Qwen/Qwen3.5-0.8B",
            "layers": text["num_hidden_layers"],
            "layer_types": text["layer_types"],
            "hidden_size": text["hidden_size"],
            "intermediate_size": text["intermediate_size"],
            "vocab_size": text["vocab_size"],
            "tie_word_embeddings": text["tie_word_embeddings"],
            "text_model_params": params.pop(),
        },
        "label_words": list(LABEL_WORDS),
        "label_ids": list(LABEL_IDS),
        "edits": edits,
    }


def main() -> None:
    data = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {rel(OUT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
