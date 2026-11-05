"""Archive copy of output/wsp-a/build_edit.py as used for edit v1c (hash 91863b69...), kept for the audit."""
import torch

PREFIX = "model.language_model."
EMBED = PREFIX + "embed_tokens.weight"
LABEL_IDS = (25399, 39923, 65552, 60477)  # productive, recovering, stalled, looping
NEURON = {23: 1046, 22: 1613}  # neurons that are silent at the readout position in the original model


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
