"""The label-free prompt-choice rule: candidates are valid, the subset is fixed, the rule is as written."""

from __future__ import annotations

import numpy as np
import pytest
from transformers import AutoTokenizer

from hypaqwen import model as qmodel
from hypaqwen import prompt_choice as P
from hypaqwen.interface import DRAFT, FROZEN, build_interface, spec_sha256


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


def test_revision1_rule():
    def r(name, med, share):
        return {"name": name, "mass": {"median": med}, "predicted_share": dict(zip(P.CLASS_NAMES, share))}
    one = r("one", 0.99, (0.0, 0.0, 1.0, 0.0))
    spread = r("spread", 0.8, (0.1, 0.3, 0.4, 0.2))
    assert P.share_entropy(one["predicted_share"]) == 0.0
    assert P.share_entropy({c: 0.25 for c in P.CLASS_NAMES}) == pytest.approx(1.0)
    assert P.choose_revision1([one, spread]) == "spread"
    assert P.choose_revision1([spread, dict(spread, name="tie")]) == "spread"


def test_frozen_is_the_revision1_choice():
    import json
    rep = json.loads(P.REVISION1_REPORT.read_text())
    assert rep["chosen"] == FROZEN.name
    chosen = next(c for c in P.CANDIDATES if c.name == FROZEN.name)
    assert (chosen.label_words, chosen.answer_cue, chosen.style) == (FROZEN.label_words, FROZEN.answer_cue, FROZEN.style)
    assert spec_sha256(chosen) == spec_sha256(FROZEN) == spec_sha256(DRAFT)  # the M0 draft text, unchanged
