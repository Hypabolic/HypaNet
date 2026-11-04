"""M4 random valid-window sampler (experiments/m4/README.md §3.4; spec v0.2 §4 invariants)."""

from __future__ import annotations

import numpy as np
import pytest

from agent_loop_lab.eval import sampler
from agent_loop_lab.schema import FEATURE_INDEX as FI

N = 20_000


@pytest.fixture(scope="module")
def w():
    x = sampler.sample_windows(N, sampler.SAMPLER_SEED)
    assert x.shape == (N, 120) and x.dtype == np.float32
    return x.reshape(N, 8, 15)


def _cat(w):
    return np.argmax(w[..., FI["cat_INSPECT"] : FI["cat_CONTROL"] + 1], axis=-1)


def test_deterministic():
    a = sampler.sample_windows(500, 7)
    assert np.array_equal(a, sampler.sample_windows(500, 7))
    assert not np.array_equal(a, sampler.sample_windows(500, 8))


def test_one_hot_groups(w):
    assert (w[..., 0:5].sum(-1) == 1).all()
    assert (w[..., 5:8].sum(-1) == 1).all()


def test_spec_invariants(w):
    assert (w[..., FI["repeated_failure"]] <= w[..., FI["out_ERROR"]]).all()
    assert (w[..., FI["state_revisited"]] <= w[..., FI["workspace_changed"]]).all()
    assert (w[..., FI["validation_comparable"]] <= w[..., FI["cat_VALIDATE"]]).all()
    d = w[..., FI["validation_delta"]]
    assert ((d == 0) | (w[..., FI["validation_comparable"]] == 1)).all()
    assert (w[:, 0, FI["repeated_intent"]] == 0).all() and (w[:, 0, FI["repeated_failure"]] == 0).all()
    assert (w[:, 0, FI["state_revisited"]] == 0).all()
    assert ((d >= -1) & (d <= 1)).all()
    other = np.delete(w, FI["validation_delta"], axis=-1)
    assert ((other >= 0) & (other <= 1)).all()


def test_reachability_constraints(w):
    cat = _cat(w)
    err = w[..., FI["out_ERROR"]] == 1
    for i in range(1, 8):
        ri = w[:, i, FI["repeated_intent"]] == 1
        same = (cat[:, :i] == cat[:, i : i + 1]).any(axis=1)
        assert not (ri & ~same).any()
        rf = w[:, i, FI["repeated_failure"]] == 1
        same_err = ((cat[:, :i] == cat[:, i : i + 1]) & err[:, :i]).any(axis=1)
        assert not (rf & ~same_err).any()


def test_coverage_of_feature_values(w):
    # Every flag takes both values, deltas take both signs and zero, costs hit 0, 1 and the interior.
    for f in ("repeated_intent", "repeated_failure", "workspace_changed", "state_revisited", "validation_comparable"):
        vals = set(np.unique(w[..., FI[f]]).tolist())
        assert vals == {0.0, 1.0}, f
    d = w[..., FI["validation_delta"]][w[..., FI["validation_comparable"]] == 1]
    assert (d > 0).any() and (d < 0).any() and (d == 0).any()
    c = w[..., FI["relative_token_cost"]]
    assert (c == 0).any() and (c == 1).any() and ((c > 0) & (c < 1)).any()
    # Uniform categories and outcomes (loose bounds).
    assert np.allclose(np.bincount(_cat(w).ravel(), minlength=5) / (N * 8), 0.2, atol=0.01)


def test_python_floats_exact_in_float32(w):
    x = w.reshape(N, 120)[:50]
    for row in x:
        t = tuple(row.astype(np.float64).tolist())
        assert np.array_equal(np.asarray(t, dtype=np.float32), row)


def test_check_invariants_rejects_bad_windows():
    x = sampler.sample_windows(10, 1)
    bad = x.copy().reshape(10, 8, 15)
    bad[0, 0, FI["repeated_intent"]] = 1.0
    with pytest.raises(AssertionError):
        sampler.check_invariants(bad.reshape(10, 120))
    bad = x.copy().reshape(10, 8, 15)
    bad[0, 3, FI["validation_comparable"]] = 0.0
    bad[0, 3, FI["validation_delta"]] = 0.5
    with pytest.raises(AssertionError):
        sampler.check_invariants(bad.reshape(10, 120))
