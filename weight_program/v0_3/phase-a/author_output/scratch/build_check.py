"""Build sanity: time, finiteness, shapes/dtypes against the template order, in-process
determinism, and the count of non-zero entries per tensor (for CIRCUITS.md's accounting)."""
import time

import torch

import model
import build_weights as bw

t0 = time.process_time()
m1 = bw.build()
t1 = time.process_time()
m2 = bw.build()
print(f"build cpu time: {t1 - t0:.2f}s")
sd1, sd2 = m1.state_dict(), m2.state_dict()
assert list(sd1) == list(model.PARAM_SHAPES), "name order"
total_nz = 0
for name, shape in model.PARAM_SHAPES.items():
    a, b = sd1[name], sd2[name]
    assert tuple(a.shape) == shape, name
    assert a.dtype == torch.float32, name
    assert torch.isfinite(a).all(), name
    assert torch.equal(a, b), name
    nz = int((a != 0).sum())
    total_nz += nz
    if nz:
        print(f"{name:28s} nonzero={nz:7d}  max|x|={float(a.abs().max()):.1f}")
print("total nonzero", total_nz, "of", model.PARAM_COUNT)
print("deterministic in-process: OK")
