"""The one-pass branched readout equals a separate prompt per window, and never sees later blocks.

Needs the pinned Qwen checkpoint and the dev task export (``scripts/v04_export_tasks.py``).
Device: HYPAQWEN_TEST_DEVICE (default cpu).
"""

from __future__ import annotations

import itertools
import os

import pytest
import torch

from hypaqwen import model as qmodel
from hypaqwen.data import iter_tasks
from hypaqwen.interface import build_interface
from hypaqwen.readout import Verbaliser

DEVICE = os.environ.get("HYPAQWEN_TEST_DEVICE", "cpu")
N_SESSIONS = 3
LOGIT_ATOL = 2e-3  # fp32; chunked DeltaNet sums in a different order when run block by block


@pytest.fixture(scope="module")
def setup():
    tok, model = qmodel.load(DEVICE, "fp32")
    iface = build_interface(tok)
    return tok, iface, Verbaliser(model, iface, hidden_layers=(12,))


def _tasks(n):
    return list(itertools.islice(iter_tasks("dev"), n))


def test_label_words_are_single_tokens(setup):
    tok, iface, _ = setup
    assert len(iface.label_ids) == 4 and len(set(iface.label_ids)) == 4


def test_branched_equals_separate_prompts(setup):
    tok, iface, verb = setup
    for task in _tasks(N_SESSIONS):
        bids = iface.block_ids(tok, task.blocks)
        got = verb.session(bids, task.window_pos)
        assert got.pos == task.window_pos
        wp = task.window_pos
        for i in sorted({0, len(wp) // 2, len(wp) - 1}):  # first, middle, last window
            t = wp[i]
            ref = verb.reference_window(bids, t)
            diff = (got.label_logits[i] - ref.label_logits[0]).abs().max().item()
            assert diff < LOGIT_ATOL, (task.session_id, t, diff)
            assert torch.argmax(got.label_logits[i]) == torch.argmax(ref.label_logits[0])
            assert abs(got.logsumexp[i] - ref.logsumexp[0]) < LOGIT_ATOL
            assert torch.allclose(got.hidden[12][i], ref.hidden[12][0], atol=5e-3, rtol=1e-3)


def test_label_never_sees_later_blocks(setup):
    tok, iface, verb = setup
    task = _tasks(1)[0]
    bids = iface.block_ids(tok, task.blocks)
    t = task.window_pos[0]
    full = verb.session(bids, task.window_pos)
    cut = verb.session(bids[: t + 1], [t])
    assert torch.equal(full.label_logits[0], cut.label_logits[0])
