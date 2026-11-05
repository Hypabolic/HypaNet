"""Matched pairs with identical histories (mk5.R5, seed = pair id). Same pair definitions as scen_hb."""
from mk5 import R5
from mk1 import hnum, variant
import scen_hb

PAIRS = []


def base_pair(pid, vocab_idx, hist):
    out = []
    for tag in ("p", "n"):
        r = R5(pid + tag, vocab_idx, seed=pid); r.history(hist); out.append(r)
    return out


def build_all():
    PAIRS.clear()
    scen_hb.base_pair = base_pair       # the pair builders of scen_hb use this to create members
    scen_hb.PAIRS = PAIRS               # and append to this list
    j = 0
    for vocab_idx in range(4):
        for k in range(2):
            hist = 10 + hnum("hb5", vocab_idx, k, hi=10)
            scen_hb.nf1(f"k{j}", vocab_idx, hist, "read" if k == 0 else "test"); j += 1
            scen_hb.nf2(f"k{j}", vocab_idx, hist + 1, "read" if k == 1 else "test"); j += 1
            scen_hb.repf(f"k{j}", vocab_idx, hist + 2); j += 1
            scen_hb.oscok(f"k{j}", vocab_idx, hist + 3); j += 1
            scen_hb.resh(f"k{j}", vocab_idx, hist + 4); j += 1
            scen_hb.pend(f"k{j}", vocab_idx, hist + 5, 3 + k); j += 1
            scen_hb.reph(f"k{j}", vocab_idx, hist + 6); j += 1
            scen_hb.repp(f"k{j}", vocab_idx, hist + 7); j += 1
            scen_hb.nf3(f"k{j}", vocab_idx, hist + 8); j += 1
            scen_hb.rds(f"k{j}", vocab_idx, hist + 9); j += 1
    return list(PAIRS)
