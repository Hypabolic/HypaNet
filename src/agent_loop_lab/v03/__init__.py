"""Spec v0.3 (raw-log transformer) harness code.

:mod:`.rawop` is the raw operation record both generators emit as a side
channel. It imports nothing from the generators, the datasets or the oracle,
so the renderer and the authoring packet can depend on it.
"""

from .rawop import DROPPED_VOCAB, RawOp, RawOpError

__all__ = ["DROPPED_VOCAB", "RawOp", "RawOpError"]
