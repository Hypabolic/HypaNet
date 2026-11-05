"""Set R: realistic labelled sessions (busy history of 10-22 blocks, then one 8-block class window). The label of
the final window is by construction (the window shapes mirror the disclosed windows' tool sequences)."""
from mk3 import R, VOCAB
from mk1 import hnum


def build(cls, v, vocab_idx, hist_len):
    sid = f"r_{cls}{v}_{vocab_idx}"
    r = R(sid, vocab_idx)
    r.history(hist_len)
    getattr(r, "win_" + cls)(v)
    return r


def all_sessions():
    out = []  # (label, R)
    labels = {"prod": "PRODUCTIVE", "reco": "RECOVERY", "stal": "STALLED", "osc": "OSCILLATING"}
    for cls in ("prod", "reco", "stal", "osc"):
        for v in range(8):
            for vocab_idx in (v % 4, (v + 2) % 4):
                hist = 10 + hnum(cls, v, vocab_idx, hi=13)
                out.append((labels[cls], build(cls, v, vocab_idx, hist)))
    return out
