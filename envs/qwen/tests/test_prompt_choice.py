"""The label-free prompt-choice rule: candidates are valid, the subset is fixed, the rule is as written."""

from __future__ import annotations

import numpy as np
import pytest
from transformers import AutoTokenizer

from hypaqwen import model as qmodel
from hypaqwen import prompt_choice as P
from hypaqwen.interface import DRAFT, build_interface


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(qmodel.verify_files())


def test_every_candidate_builds(tok):
    assert len(P.CANDIDATES) == 12 and len({c.name for c in P.CANDIDATES}) == 12
    for spec in P.CANDIDATES:
        iface = build_interface(tok, spec)
        assert len(set(iface.label_ids)) == 4


def test_draft_is_a_candidate():
    assert any(c.label_words == DRAFT.label_words and c.answer_cue == DRAFT.answer_cue and c.style == DRAFT.style
               for c in P.CANDIDATES)


def test_subset_is_order_free_and_sized():
    ids = [f"s-{i:016x}" for i in range(500)]
    a = P.subset_ids(ids)
    assert a == P.subset_ids(list(reversed(ids))) and len(a) == P.N_SESSIONS


def test_label_mass():
    logits = np.array([[0.0, 0.0, 0.0, 0.0]])
    lse = np.array([np.log(8.0)])  # the four label words hold half the mass
    assert np.allclose(P.label_mass(logits, lse), [0.5])


def test_choose_highest_median_ties_to_earlier():
    r = [{"name": "a", "mass": {"median": 0.4}}, {"name": "b", "mass": {"median": 0.6}}, {"name": "c", "mass": {"median": 0.6}}]
    assert P.choose(r) == "b"
