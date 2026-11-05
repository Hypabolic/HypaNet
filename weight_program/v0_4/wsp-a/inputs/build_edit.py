"""WSP-A edit: three hand-built SwiGLU neurons (layer 23 MLP: OSCILLATING; layer 22 MLP: STALLED; layer 21 MLP:
RECOVERY) and a PRODUCTIVE readout direction in the label rows. Every vector comes from
edit_constants.safetensors (derivation: output/wsp-a/scratch/derive_v2.py; see EDIT.md)."""
import torch

PREFIX = "model.language_model."
EMBED = PREFIX + "embed_tokens.weight"
LABEL_IDS = (25399, 39923, 65552, 60477)  # productive, recovering, stalled, looping
NEURON = {23: 1046, 22: 1613, 21: 1007}  # neurons that are silent at the readout position in the original model


def build(orig, consts):
    out = {}
    emb = orig[EMBED]
    deltas = consts["label_row_deltas"].to(torch.float32)
    for c, tid in enumerate(LABEL_IDS):
        emb[tid] = emb[tid] + deltas[c]
    out[EMBED] = emb
    for layer, n in NEURON.items():
        gate = orig[f"{PREFIX}layers.{layer}.mlp.gate_proj.weight"]
        up = orig[f"{PREFIX}layers.{layer}.mlp.up_proj.weight"]
        down = orig[f"{PREFIX}layers.{layer}.mlp.down_proj.weight"]
        gate[n] = consts[f"gate_row_L{layer}"].to(torch.float32)
        up[n] = consts[f"up_row_L{layer}"].to(torch.float32)
        down[:, n] = consts[f"down_col_L{layer}"].to(torch.float32)
        out[f"{PREFIX}layers.{layer}.mlp.gate_proj.weight"] = gate
        out[f"{PREFIX}layers.{layer}.mlp.up_proj.weight"] = up
        out[f"{PREFIX}layers.{layer}.mlp.down_proj.weight"] = down
    return out
