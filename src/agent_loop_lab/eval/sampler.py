"""Random valid-window sampler for H2 fidelity part (b) (spec v0.2 §10.1; experiments/m4/README.md §3.4).

Draws 120-vectors that satisfy every §4 invariant plus the encoder-reachability
constraints below, without using the generator. Positions are drawn oldest to
newest; each feature is drawn independently except:

* ``repeated_intent`` may be 1 only at i > 0 and only if an earlier position has
  the same category (the intent fingerprint hashes the operation family);
* ``repeated_failure`` may be 1 only for an ERROR at i > 0 with an earlier ERROR
  of the same category (the failure fingerprint hashes the family);
* ``state_revisited`` may be 1 only if ``workspace_changed`` and i > 0;
* ``validation_comparable`` may be 1 only for VALIDATE; ``validation_delta`` is
  non-zero only when comparable.

Probabilities (pre-registered): category and outcome uniform; every free binary
flag Bernoulli(0.5); delta = 0 w.p. 0.25 else float32(U[-1, 1]); token cost = 0
w.p. 0.1, 1 w.p. 0.05, else float32(U[0, 1]).
"""

from __future__ import annotations

import numpy as np

from ..schema import EVENTS_PER_WINDOW, FEATURE_INDEX, FEATURES_PER_EVENT, INPUT_DIM

__all__ = ["SAMPLER_SEED", "SAMPLER_WINDOWS", "sample_windows", "check_invariants"]

SAMPLER_SEED = 20261105
SAMPLER_WINDOWS = 100_000
P_FLAG = 0.5
P_DELTA_ZERO = 0.25
P_COST_ZERO = 0.10
P_COST_ONE = 0.05

_CAT = [FEATURE_INDEX[f"cat_{c}"] for c in ("INSPECT", "MUTATE", "VALIDATE", "EXTERNAL", "CONTROL")]
_OUT = [FEATURE_INDEX[f"out_{o}"] for o in ("SUCCESS", "ERROR", "NEUTRAL")]
_RI = FEATURE_INDEX["repeated_intent"]
_RF = FEATURE_INDEX["repeated_failure"]
_WC = FEATURE_INDEX["workspace_changed"]
_SR = FEATURE_INDEX["state_revisited"]
_VC = FEATURE_INDEX["validation_comparable"]
_VD = FEATURE_INDEX["validation_delta"]
_TC = FEATURE_INDEX["relative_token_cost"]
_VALIDATE = 2
_ERROR = 1


def sample_windows(n: int = SAMPLER_WINDOWS, seed: int = SAMPLER_SEED) -> np.ndarray:
    """``float32 (n, 120)`` valid windows; deterministic in ``(n, seed)``."""
    if n < 0:
        raise ValueError("n must be >= 0")
    rng = np.random.default_rng(seed)
    P, F = EVENTS_PER_WINDOW, FEATURES_PER_EVENT
    cat = rng.integers(0, 5, size=(n, P))
    out = rng.integers(0, 3, size=(n, P))
    u_ri = rng.random((n, P)) < P_FLAG
    u_rf = rng.random((n, P)) < P_FLAG
    wc = rng.random((n, P)) < P_FLAG
    u_sr = rng.random((n, P)) < P_FLAG
    u_vc = rng.random((n, P)) < P_FLAG
    d_zero = rng.random((n, P)) < P_DELTA_ZERO
    d_val = rng.uniform(-1.0, 1.0, size=(n, P)).astype(np.float32)
    c_u = rng.random((n, P))
    c_val = rng.uniform(0.0, 1.0, size=(n, P)).astype(np.float32)

    w = np.zeros((n, P, F), dtype=np.float32)
    rows = np.arange(n)
    for i in range(P):
        w[rows, i, np.asarray(_CAT)[cat[:, i]]] = 1.0
        w[rows, i, np.asarray(_OUT)[out[:, i]]] = 1.0
        if i > 0:
            same_cat_before = (cat[:, :i] == cat[:, i : i + 1]).any(axis=1)
            err = out[:, i] == _ERROR
            err_same_cat_before = ((cat[:, :i] == cat[:, i : i + 1]) & (out[:, :i] == _ERROR)).any(axis=1)
            w[:, i, _RI] = (u_ri[:, i] & same_cat_before).astype(np.float32)
            w[:, i, _RF] = (u_rf[:, i] & err & err_same_cat_before).astype(np.float32)
            w[:, i, _SR] = (u_sr[:, i] & wc[:, i]).astype(np.float32)
        w[:, i, _WC] = wc[:, i].astype(np.float32)
        comp = u_vc[:, i] & (cat[:, i] == _VALIDATE)
        w[:, i, _VC] = comp.astype(np.float32)
        w[:, i, _VD] = np.where(comp & ~d_zero[:, i], d_val[:, i], np.float32(0.0))
        cost = np.where(c_u[:, i] < P_COST_ZERO, np.float32(0.0),
                        np.where(c_u[:, i] < P_COST_ZERO + P_COST_ONE, np.float32(1.0), c_val[:, i]))
        w[:, i, _TC] = cost.astype(np.float32)
    x = w.reshape(n, INPUT_DIM)
    check_invariants(x)
    return x


def check_invariants(x: np.ndarray) -> None:
    """Raise ``AssertionError`` if any window violates a §4 invariant."""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != INPUT_DIM or x.dtype != np.float32:
        raise AssertionError(f"expected float32 (n, {INPUT_DIM}), got {x.dtype} {x.shape}")
    w = x.reshape(-1, EVENTS_PER_WINDOW, FEATURES_PER_EVENT)
    if not np.isfinite(w).all():
        raise AssertionError("non-finite value")
    assert (w[..., _CAT].sum(-1) == 1).all(), "category one-hot"
    assert (w[..., _OUT].sum(-1) == 1).all(), "outcome one-hot"
    assert np.isin(w[..., _CAT + _OUT + [_RI, _RF, _WC, _SR, _VC]], (0.0, 1.0)).all(), "binary features"
    assert (w[..., _RF] <= w[..., _OUT[_ERROR]]).all(), "repeated_failure <= ERROR"
    assert (w[..., _SR] <= w[..., _WC]).all(), "state_revisited <= workspace_changed"
    assert (w[..., _VC] <= w[..., _CAT[_VALIDATE]]).all(), "validation_comparable <= VALIDATE"
    assert ((w[..., _VD] == 0) | (w[..., _VC] == 1)).all(), "delta != 0 => comparable"
    assert (w[:, 0, _RI] == 0).all() and (w[:, 0, _RF] == 0).all(), "index-0 repeat flags"
    assert ((w[..., _VD] >= -1) & (w[..., _VD] <= 1)).all(), "delta range"
    others = [k for k in range(FEATURES_PER_EVENT) if k != _VD]
    assert ((w[..., others] >= 0) & (w[..., others] <= 1)).all(), "[0, 1] range"
