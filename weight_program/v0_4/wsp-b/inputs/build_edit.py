"""WSP-B edit (one revision of the WSP-A edit): two hand-built SwiGLU neurons kept from WSP-A with revised rows
(layer 23 MLP unit 1046: OSCILLATING; layer 22 MLP unit 1613: STALLED), and revised label rows (PRODUCTIVE default
plus a linear RECOVERY term). The WSP-A layer-21 neuron is dropped (layer 21 returns to the original weights).
Every vector comes from edit_constants.safetensors (derivation: output/wsp-b/scratch/derive_b5.py; see EDIT.md)."""
import torch

PREFIX = "model.language_model."
EMBED = PREFIX + "embed_tokens.weight"
LABEL_IDS = (25399, 39923, 65552, 60477)  # productive, recovering, stalled, looping
NEURON = {23: 1046, 22: 1613}  # units that are silent at the readout position in the original model


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
