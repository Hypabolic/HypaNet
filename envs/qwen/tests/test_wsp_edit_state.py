"""The shared §2 rescaling rule and edit hash (hypaqwen.wsp.edit_state): server and runner use the same one."""

from __future__ import annotations

import torch

from hypaqwen.wsp import edit_state as ES
from hypaqwen.wsp import runner, server

NORM = "model.language_model.layers.3.self_attn.q_norm.weight"
MAT = "model.language_model.layers.3.mlp.down_proj.weight"


def test_server_and_runner_share_the_rule_and_hash():
    assert server.scalar_multiple is ES.scalar_multiple and server.label_rows_rescaled is ES.label_rows_rescaled
    assert server.EMPTY_EDIT_HASH == ES.EMPTY_EDIT_HASH == ES.edit_state_hash({})
    assert server.ModelServer._state_hash({"a": "00" * 32}) == ES.edit_state_hash({"a": "00" * 32})
    assert runner.edit_state is ES


def test_rescaling_rule():
    g = torch.Generator().manual_seed(0)
    w = 0.02 * torch.randn(64, 32, generator=g)
    assert ES.rescaling(MAT, 1.5 * w, w) and ES.rescaling(MAT, torch.zeros_like(w), w)
    assert ES.rescaling(MAT, w + 1e-3 * torch.outer(torch.randn(64, generator=g), torch.randn(32, generator=g)), w) is None
    z = w.clone()
    z[:8] = 0
    assert ES.rescaling(MAT, z, w) is None  # zeroing an eighth of the rows


def test_norm_gain_rescaling_refused():
    g = torch.Generator().manual_seed(1)
    w = 0.1 * torch.randn(128, generator=g)  # zero-centred: the layer multiplies by 1 + w
    assert ES.is_zero_centred_norm(NORM)
    new = 1.5 * (1 + w) - 1
    assert ES.rescaling(NORM, new, w) is not None
    assert ES.rescaling("model.language_model.layers.0.linear_attn.norm.weight", w + 0.05, w) is None  # x * w norm
    local = w.clone()
    local[:16] += 0.05
    assert ES.rescaling(NORM, local, w) is None


def test_label_rows_rescaling_refused():
    g = torch.Generator().manual_seed(2)
    rows = torch.randn(4, 16, generator=g)
    assert ES.rescaling(ES.EMBED, 2.0 * rows + torch.randn(16, generator=g), rows) is not None  # shared offset
    one = rows.clone()
    one[2] *= 1.5
    assert ES.rescaling(ES.EMBED, one, rows) is None
