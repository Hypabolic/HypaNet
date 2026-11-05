"""Candidate v4 = v3 with (a) suite-only intents for run_tests (the tests= selection does not
start a new episode: V125, V142) and (b) an active in-policy episode needs >= 2 retries
(V029, V142), while one retry suffices for a later OK run to be a resolution (V088, V115)."""

import policy_v3 as base

PRODUCTIVE, RECOVERY, STALLED, OSCILLATING = 0, 1, 2, 3


def _block_features(tb):
    f = base._block_features(tb)
    if f["tool"] == "run_tests":
        n = len(tb)
        intent = None
        first_any = None
        for i in range(10, n):
            if tb[i] in base._MARKERS:
                break
            if base._hashed(tb[i]):
                intent = base._token_id(tb[i])
                break
            if first_any is None and base._token_id(tb[i]) is not None:
                first_any = base._token_id(tb[i])
        if intent is None:
            intent = first_any
        f["intent"] = intent
        f["key"] = (f["cls"], intent) if (f["cls"] is not None and intent is not None) else None
    return f


def analyse(blocks):
    saved = (base._block_features, base.INPOL_MIN_R)
    base._block_features = _block_features
    base.INPOL_MIN_R = 2
    try:
        return base.analyse(blocks)
    finally:
        base._block_features, base.INPOL_MIN_R = saved


decide = base.decide


def reference_policy(tokens, event_ord):
    blocks = []
    cur = None
    cur_ord = None
    for s, e in zip(tokens, event_ord):
        if e == 0:
            continue
        if e != cur_ord:
            cur = []
            blocks.append(cur)
            cur_ord = e
        cur.append(s)
    if not blocks:
        return PRODUCTIVE
    feats, rel = analyse(blocks)
    return decide(feats, rel, len(blocks) - 1)
