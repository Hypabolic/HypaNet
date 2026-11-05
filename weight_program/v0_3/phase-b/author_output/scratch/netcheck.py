"""Phase B network vs policy: on the 200 disclosed windows (every prefix ending at the window's
endpoint, plus all earlier prefixes) and on the Phase A scratch sessions.

usage: netcheck.py disc        disclosed windows: endpoint agreement + oracle score of the network
       netcheck.py discall     disclosed windows: agreement at every EOE of every prefix
       netcheck.py phasea      the 27 Phase A scratch sessions
       netcheck.py feats V001  print named features at every EOE of a window
"""
import os
import sys

import torch

import model
import tokenizer
import build_weights as bw
import reference_policy as rp
from disc import CLASSES, load

_MODEL = None


def net():
    global _MODEL
    if _MODEL is None:
        _MODEL = bw.build()
    return _MODEL


def logits_of(enc):
    m = net()
    acts = m.activations(torch.tensor(enc.tokens), torch.tensor(enc.event_ord), torch.tensor(enc.offset),
                         query_positions=[0])
    return acts["logits"], acts["resid_mlp"][-1]


FEATS = ["BSOK", "BSFAIL", "HASHUNK", "PV", "EXISTS", "PSOK", "PSFAIL", "PPV", "NEAREST", "OUTFAIL", "ANCH",
         "MAXPV", "EXISTS7", "NOTIDST", "CNT3", "CNT5", "IMPROVED", "NOOP", "REDINSP", "PEX7", "PPSFAIL", "PNEAR",
         "NVEX", "NVNOTST", "REVBLK", "SI", "INPOL", "RESOLV", "AVGSI", "AVGREV", "AVGIMP", "ANCH8", "ACT",
         "REC4", "V2", "I5", "PROG", "RECB"]


def feats_dump(w, last=10):
    enc = w["enc"]
    logits, resid = logits_of(enc)
    for t, idx in enumerate(enc.eoe_index):
        if t < len(enc.eoe_index) - last:
            continue
        pre = tokenizer.prefix(enc, t)
        pol = rp.reference_policy(pre.surface, pre.event_ord)
        netc = int(torch.argmax(logits[idx]))
        head = w["blocks"][t].split("\n")[0][13:80]
        mark = "" if netc == pol else "  <<< DISAGREE"
        print(f"[{t:2d}] pol={CLASSES[pol][:4]} net={CLASSES[netc][:4]} {head}{mark}")
        vals = " ".join(f"{n}={float(resid[idx, getattr(bw, n)]):.2f}" for n in FEATS)
        print("     " + vals)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "disc"
    if mode in ("disc", "discall"):
        wins = load()
        n_dis = n_tot = 0
        correct = 0
        conf = {}
        for w in wins:
            enc = w["enc"]
            logits, _ = logits_of(enc)
            ts = range(len(enc.eoe_index)) if mode == "discall" else [len(enc.eoe_index) - 1]
            for t in ts:
                pre = tokenizer.prefix(enc, t)
                pol = rp.reference_policy(pre.surface, pre.event_ord)
                netc = int(torch.argmax(logits[enc.eoe_index[t]]))
                n_tot += 1
                if netc != pol:
                    n_dis += 1
                    print(f"DISAGREE {w['id']} t={t} pol={CLASSES[pol]} net={CLASSES[netc]} logits={[round(float(x), 1) for x in logits[enc.eoe_index[t]]]}")
            netc = int(torch.argmax(logits[enc.eoe_index[-1]]))
            correct += CLASSES[netc] == w["label"]
            conf[(w["label"], CLASSES[netc])] = conf.get((w["label"], CLASSES[netc]), 0) + 1
        print(f"network vs policy disagreements: {n_dis} / {n_tot}")
        print(f"network vs oracle (endpoints): {correct} / {len(wins)}")
        for k in sorted(conf):
            print(" ", k, conf[k])
    elif mode == "feats":
        wins = {w["id"]: w for w in load()}
        feats_dump(wins[sys.argv[2]])
    elif mode == "featsA":
        here = os.path.dirname(os.path.abspath(__file__))
        a_scratch = os.path.join(os.path.dirname(os.path.dirname(here)), "v03-phase-a", "scratch")
        sys.path.append(a_scratch)
        import renderer
        import common as ca
        import importlib
        sessions = importlib.import_module(sys.argv[2]).SESSIONS
        ops = sessions[sys.argv[3]]
        blocks = renderer.render_session(ca.stamp(ops), "sid-" + sys.argv[3])
        enc = tokenizer.encode_session(blocks)
        feats_dump({"enc": enc, "blocks": blocks}, last=int(sys.argv[4]) if len(sys.argv) > 4 else 10)
    elif mode == "phasea":
        here = os.path.dirname(os.path.abspath(__file__))
        a_scratch = os.path.join(os.path.dirname(os.path.dirname(here)), "v03-phase-a", "scratch")
        sys.path.append(a_scratch)
        import renderer
        import common as ca  # Phase B copy of the Phase A op builders
        from check1 import SESSIONS as S1
        from check2 import SESSIONS as S2
        from check3 import SESSIONS as S3
        n_dis = n_tot = 0
        for name, sessions in (("check1", S1), ("check2", S2), ("check3", S3)):
            for sname, ops in sessions.items():
                blocks = renderer.render_session(ca.stamp(ops), "sid-" + sname)
                enc = tokenizer.encode_session(blocks)
                logits, _ = logits_of(enc)
                for t, idx in enumerate(enc.eoe_index):
                    pre = tokenizer.prefix(enc, t)
                    pol = rp.reference_policy(pre.surface, pre.event_ord)
                    netc = int(torch.argmax(logits[idx]))
                    n_tot += 1
                    if netc != pol:
                        n_dis += 1
                        print(f"DISAGREE {name}/{sname} t={t} pol={CLASSES[pol]} net={CLASSES[netc]}")
        print(f"phase A sessions: disagreements {n_dis} / {n_tot}")
