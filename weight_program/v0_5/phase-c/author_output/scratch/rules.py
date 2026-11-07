"""Observable-rule evaluator (analysis only; nothing here touches weights).

Mirrors what the network can compute from the rendered log, with switchable rule variants, so that candidate
rules can be checked against the oracle labels of the disclosed windows before they are compiled into weights.

Usage: ./py rules.py [config ...]   (default: all named configs)
"""
import re
import sys
from collections import defaultdict

import model
import v05_load

CN = model.CLASS_NAMES
MUT = ("edit_file", "write_file", "revert")
VAL = ("run_tests", "build", "lint")


def parse(block):
    lines = block.split("\n")
    h = lines[0]
    toks = h.split(" ")
    fam = toks[1]
    m = re.search(r" -> (\S+)(?: (\S+)/(\S+) passed)? in=(\S+) out=(\S+)$", h)
    status = m.group(1)
    count = (m.group(2), m.group(3)) if m.group(2) else None
    fields = h[len(toks[0]) + len(fam) + 2: m.start()]
    hunks = []
    err = None
    for ln in lines[1:]:
        s = ln.strip()
        if s.startswith("@ "):
            parts = s[2:].split(" ")
            if parts[0] != "TRUNC":
                hunks.append((parts[0], parts[1], parts[3]))
        elif s.startswith("! "):
            mm = re.match(r"(\S+) at (\S+):(\S+) \(", s[2:])
            err = (mm.group(1), mm.group(2), mm.group(3)) if mm else ("?", "?", "?")
    return {"fam": fam, "fields": fields, "status": status, "count": count, "hunks": hunks, "err": err}


def target_key(ev, plan_mode):
    """The token the network keys a 'target' on (block 3): read/list path, first search word, plan action,
    build/lint target. Returns a list of keys (plan_mode 'both' yields up to two)."""
    fam, f = ev["fam"], ev["fields"]
    if fam in ("read_file", "list_dir"):
        return [("path", f.split(" ")[0].replace("path=", ""))]
    if fam == "search":
        m = re.search(r'query="([^"]*)"', f)
        q = (m.group(1).split(" ")[0] if m and m.group(1) else "")
        return [("search", q)] if q else []
    if fam in ("build", "lint"):
        return [("target", f.replace("target=", ""))]
    if fam in ("plan", "ask", "yield"):
        act = f.replace("action=", "")
        toks = re.findall(r"[A-Za-z_][A-Za-z0-9_.\-/]*|[0-9]+", act)
        if plan_mode == "off":
            return []
        if plan_mode == "first":
            return [("plan", toks[0])] if toks else []
        if plan_mode == "second":
            # key at offset 12 when a word/digit is there, else at offset 10
            if len(toks) >= 2 and re.match(r"^[A-Za-z_]", toks[1]):
                return [("plan", toks[1])]
            if len(toks) >= 2:
                return []  # a digit at offset 12: keyed with code A only -> never a two-code repeat
            return [("plan", toks[0])] if toks else []
        if plan_mode == "both":
            keys = [("plan", toks[0])] if toks else []
            if len(toks) >= 2:
                keys.append(("plan2", toks[1] if re.match(r"^[A-Za-z_]", toks[1]) else None))
            return keys
    return []


def features(blocks, cfg):
    """Per-event features over the whole prefix (causal)."""
    evs = [parse(b) for b in blocks]
    n = len(evs)
    F = []
    # per-event derived
    for i, ev in enumerate(evs):
        fam, st = ev["fam"], ev["status"]
        f = {}
        f["fam"] = fam
        f["mut"] = fam in MUT
        f["hunk"] = bool(ev["hunks"])
        f["fail"] = st in ("error", "FAILED", "pending")
        f["okish"] = st in ("ok", "ready")
        f["val"] = fam in VAL
        f["outcome"] = f["fail"] or (f["okish"] and (fam in ("http_call", "poll_status") or fam in VAL))
        # identity keys
        f["line"] = ev["err"][2] if ev["err"] else None
        f["cnt"] = ev["count"][0] if ev["count"] else None
        f["poll"] = ev["fields"].split(" ")[0].replace("resource=", "") if fam == "poll_status" else None
        f["tkeys"] = target_key(ev, cfg["plan"])
        f["hunks"] = ev["hunks"]
        F.append(f)
    # cross-event features
    for i, f in enumerate(F):
        ev = evs[i]
        fam = f["fam"]
        # repeated target within 31 events (two codes: exact string match)
        rep = False
        rep_first = rep_second = None
        for kind, key in f["tkeys"]:
            seen = any((kind2, key) in F[j]["tkeys"] or (kind in ("plan", "plan2") and kind2 in ("plan", "plan2", "path") and key2 == key and key is not None)
                       for j in range(max(0, i - 31), i) for kind2, key2 in F[j]["tkeys"])
            if kind == "plan2":
                rep_second = seen
            elif kind == "plan":
                rep_first = seen
            else:
                rep = rep or seen
        if cfg["plan"] == "both" and any(k == "plan" for k, _ in f["tkeys"]):
            has2 = any(k == "plan2" for k, _ in f["tkeys"])
            key2 = [v for k, v in f["tkeys"] if k == "plan2"]
            if rep_first and (not has2 or (key2 and key2[0] is not None and rep_second)):
                rep = True
        elif cfg["plan"] in ("first", "second") and rep_first:
            rep = True
        f["REP"] = rep
        # ok count seen before (ok validations, 31 events)
        f["CNTOK"] = bool(f["okish"] and f["val"] and f["cnt"] is not None and any(
            F[j]["okish"] and F[j]["val"] and F[j]["cnt"] == f["cnt"] for j in range(max(0, i - 31), i)))
        # failure identity matches
        def same_fail(j, mode):
            g = F[j]
            if not (g["fail"] and f["fail"]):
                return False
            if f["poll"] is not None:
                return g["poll"] == f["poll"]
            if f["line"] is None:
                return False
            if mode == "line":
                return g["line"] == f["line"]
            if mode == "lc":
                return g["line"] == f["line"] and g["cnt"] == f["cnt"]
            raise ValueError(mode)
        mode = cfg["ident"]
        f["k15"] = sum(same_fail(j, mode) for j in range(max(0, i - 15), i))
        f["kline15"] = sum(same_fail(j, "line") for j in range(max(0, i - 15), i))
        f["kcnt15"] = sum(1 for j in range(max(0, i - 15), i) if F[j]["fail"] and F[j]["val"] and f["fail"] and f["val"] and f["cnt"] is not None and F[j]["cnt"] == f["cnt"]) if f["val"] else 0
        # progress failure: a FAILED validation right after a mutation whose line was seen but whose count is new
        f["PROG"] = bool(f["fail"] and f["val"] and i >= 1 and F[i - 1]["mut"] and f["kline15"] >= 1 and f["kcnt15"] == 0)
        # network-style count tally: equal passed counts among all validations (ok or FAILED) within 31 events
        f["kcnt31any"] = sum(1 for j in range(max(0, i - 31), i) if F[j]["val"] and F[j]["cnt"] is not None and f["val"] and f["cnt"] is not None and F[j]["cnt"] == f["cnt"]) if f["val"] else 0
        kr = cfg.get("kcnt_range", 15)
        f["kcntR"] = sum(1 for j in range(max(0, i - kr), i) if F[j]["fail"] and F[j]["val"] and f["fail"] and f["val"] and f["cnt"] is not None and F[j]["cnt"] == f["cnt"]
                         and not (cfg.get("excl_prog") and F[j]["PROG"])) if f["val"] else 0
        f["SAME"] = f["k15"] >= 1
        if cfg["kmode"] == "net":
            f["SAME_min"] = (f["kline15"] >= 1 and f["kcnt31any"] >= 1) if (f["val"] and f["fail"]) else f["SAME"]
        else:
            f["SAME_min"] = (min(f["kline15"], f["kcnt15"]) >= 1) if (f["val"] and f["fail"]) else f["SAME"]
        f["within2"] = any(same_fail(j, mode) for j in range(max(0, i - 2), i))
        f["cnt2"] = (f["cnt"] is None) or any(F[j]["fail"] and F[j]["cnt"] == f["cnt"] for j in range(max(0, i - 2), i))
        f["prevmut"] = i >= 1 and F[i - 1]["mut"]
        f["noop"] = f["mut"] and not f["hunk"]
        f["prevfail_same"] = any(same_fail(j, mode) for j in range(max(0, i - 2), i))
        # retries
        if cfg["kmode"] == "k15":
            k = f["k15"]
        elif cfg["kmode"] == "min":
            k = min(f["kline15"], f["kcntR"]) if (f["val"] and f["fail"]) else f["k15"]
        elif cfg["kmode"] == "net":
            k = f["kline15"]  # the line tally; the count tally is applied as a separate penalty condition below
        elif cfg["kmode"] == "chain":
            # chain: link to an identical failure within 2 events with no mutation between (and that event's chain)
            k = 0
            j = i
            while True:
                nxt = None
                for d in (1, 2):
                    jj = j - d
                    if jj < 0:
                        break
                    if same_fail(jj, mode) and not (d == 2 and F[j - 1]["mut"]):
                        nxt = jj
                        break
                if nxt is None:
                    break
                k += 1
                j = nxt
                if k > 30:
                    break
        else:
            raise ValueError
        f["k"] = k
        pollk = sum(1 for j in range(max(0, i - 31), i) if F[j]["poll"] == f["poll"] and F[j]["fail"]) if f["poll"] else 0
        f["pollk"] = pollk
        if cfg["kmode"] == "net" and f["val"] and f["fail"]:
            f["kpen"] = f["kline15"] >= 4 and f["kcnt31any"] >= cfg.get("kpenc_thr", 4)
        else:
            f["kpen"] = (f["poll"] is not None and pollk >= 6) or (f["poll"] is None and k >= 4)
        readfams = ("read_file", "list_dir") if cfg.get("readlist") else ("read_file",)
        prevx = f["prevmut"] or (cfg.get("inpol_prevread") and i >= 1 and F[i - 1]["fam"] in readfams)
        f["INPOL"] = f["fail"] and f["within2"] and f["cnt2"] and not prevx and not f["kpen"]
        if cfg.get("inpol_fresh_break") and f["INPOL"]:
            # a fresh read between the identical failures breaks the chain
            j1 = i - 1
            if j1 >= 0 and not same_fail(j1, mode) and F[j1]["fam"] == "read_file" and not F[j1]["REP"]:
                f["INPOL"] = False
        f["POLLOUT"] = f["poll"] is not None and f["fail"] and pollk >= 6
        # resolution: ok of own kind after a failure of that kind within 3 events
        res = False
        if f["okish"]:
            for j in range(max(0, i - 3), i):
                g = F[j]
                if fam == "http_call" and g["fam"] == "http_call" and g["fail"]:
                    res = True
                if fam == "poll_status" and g["fam"] == "poll_status" and g["fail"]:
                    res = True
                if f["val"] and f["cnt"] is not None and g["val"] and g["fail"]:
                    if cfg["res_trigger"] == "same" and g["SAME"]:
                        res = True
                    if cfg["res_trigger"] == "cnt31any" and g["kcnt31any"] >= 1:
                        res = True
        f["RES"] = res
    # EDR and reversions need the earlier flags
    for i, f in enumerate(F):
        # EDR trigger within 3 events
        trig = False
        for j in range(max(0, i - 3), i):
            g = F[j]
            if cfg["edr"] == "same" and (g["fail"] and g["val"] and g["SAME"]):
                trig = True
            if cfg["edr"] == "cnt15" and (g["fail"] and g["val"] and g["kcnt15"] >= 1):
                trig = True
            if cfg["edr"] == "samemin" and (g["fail"] and g["val"] and g["SAME_min"]):
                trig = True
            if cfg["edr"] == "cnt31any" and (g["fail"] and g["val"] and g["kcnt31any"] >= 1):
                trig = True
            if g["CNTOK"]:
                trig = True
        f["EDR"] = f["mut"] and f["hunk"] and trig
        # h0 append: a write/edit whose hunk has h_old == h0 to a path mutated within N events
        h0 = False
        if cfg.get("h0", 0) and f["mut"]:
            for p, ho, hn in f["hunks"]:
                if ho == "h0" and (cfg.get("h0_any") or any(any(pp == p for pp, _, _ in F[j]["hunks"]) and F[j]["mut"] for j in range(max(0, i - cfg["h0"]), i))):
                    h0 = True
        f["H0"] = h0
        # reversion (lookback 7): h_new equals an earlier hunk id of the same file within 7 events (mutation tools only)
        rev7 = False
        revd = None  # distance to the nearest matching earlier hunk
        if f["mut"]:
            for p, ho, hn in f["hunks"]:
                for j in range(i - 1, max(-1, i - 8), -1):
                    g = F[j]
                    if cfg.get("rev_gate", True) and not g["mut"]:
                        continue
                    if any(pp == p and hn in (ho2, hn2) for pp, ho2, hn2 in g["hunks"]):
                        rev7 = True
                        d = i - j
                        revd = d if revd is None else min(revd, d)
        f["REV7"] = rev7
        f["revd"] = revd
        f["FRESHREAD"] = f["fam"] in (("read_file", "list_dir") if cfg.get("readlist") else ("read_file",)) and not f["REP"]
    for i, f in enumerate(F):
        same = f["SAME_min"] if cfg["ident"] == "line" and cfg["kmode"] in ("min", "net") else f["SAME"]
        parts = {
            "REP": f["REP"], "CNTOK": f["CNTOK"], "SAME": same and f["fail"] and (f["poll"] is None),
            "NOOP": f["noop"], "EDR": f["EDR"], "POLLOUT": f["POLLOUT"], "REV": f["REV7"], "H0": f["H0"],
        }
        if cfg.get("inpol_not_ineff"):
            if f["INPOL"]:
                parts["SAME"] = False
        f["INEFF"] = any(parts.values())
        f["parts"] = parts
    return F


def decide(F, t, cfg):
    lo = max(0, t - 7)
    win = list(range(lo, t + 1))
    # reversions
    if cfg["revmode"] == "lookback7":
        nrev = sum(F[e]["REV7"] for e in win)
    elif cfg["revmode"] == "window":
        nrev = sum(1 for e in win if F[e]["revd"] is not None and e - F[e]["revd"] >= lo)
    elif cfg["revmode"] == "split":
        # exact for the last 4 (lookback 7 within window automatically when delta<=3 -> e-7 >= t-10 ... approximate)
        nrev = 0
        for e in win:
            d = t - e
            if F[e]["revd"] is None:
                continue
            if d <= 3 and F[e]["revd"] <= 7:
                nrev += 1
            elif d >= 4 and F[e]["revd"] <= 3:
                nrev += 1
    elif cfg["revmode"].startswith("sched"):
        # per-distance lookback schedule L[delta]: count e iff revd(e) <= L[t-e]
        L = cfg["sched"]
        nrev = sum(1 for e in win if F[e]["revd"] is not None and F[e]["revd"] <= L[t - e])
    else:
        raise ValueError
    I = sum(F[e]["INEFF"] for e in win)
    outs = [e for e in win if F[e]["outcome"]]
    act = F[outs[-1]]["INPOL"] if outs else False
    resw = any(F[e]["RES"] for e in range(max(0, t - 3), t + 1))
    fresh = any(F[e]["FRESHREAD"] for e in win)
    if nrev >= 2:
        return 3
    if act:
        return 1
    if I >= 5 and F[t]["INEFF"] and not (cfg.get("fresh_blocks_stall") and fresh):
        return 2
    if resw:
        return 1
    return 0


BASE = dict(ident="line", kmode="k15", plan="first", edr="same", res_trigger="same", revmode="lookback7",
            rev_gate=False, h0=0, fresh_blocks_stall=False)
CONFIGS = {
    "phaseB": dict(BASE),
    "ident_lc": dict(BASE, ident="lc"),
    "ident_lc_min": dict(BASE, ident="lc", kmode="min", edr="cnt15"),
    "min_only": dict(BASE, kmode="min", edr="cnt15"),
    "min_plan2": dict(BASE, kmode="min", edr="cnt15", plan="second"),
    "min_planboth": dict(BASE, kmode="min", edr="cnt15", plan="both"),
    "min_planboth_gate": dict(BASE, kmode="min", edr="cnt15", plan="both", rev_gate=True),
    "min_pb_gate_h0": dict(BASE, kmode="min", edr="cnt15", plan="both", rev_gate=True, h0=7),
    "min_pb_gate_h0_fresh": dict(BASE, kmode="min", edr="cnt15", plan="both", rev_gate=True, h0=7, fresh_blocks_stall=True),
    "all_window": dict(BASE, kmode="min", edr="cnt15", plan="both", rev_gate=True, h0=7, fresh_blocks_stall=True, revmode="window"),
    "all_split": dict(BASE, kmode="min", edr="cnt15", plan="both", rev_gate=True, h0=7, fresh_blocks_stall=True, revmode="split"),
    "all_chain": dict(BASE, kmode="chain", edr="cnt15", plan="both", rev_gate=True, h0=7, fresh_blocks_stall=True, revmode="split"),
    "all_split_freshbreak": dict(BASE, kmode="min", edr="cnt15", plan="both", rev_gate=True, h0=7, fresh_blocks_stall=True, revmode="split", inpol_fresh_break=True),
    "C0": dict(BASE, kmode="min", edr="cnt15", plan="both", rev_gate=True, h0=7, fresh_blocks_stall=True, revmode="lookback7", inpol_fresh_break=True),
    "C_kr4": dict(BASE, kmode="min", edr="cnt15", plan="both", rev_gate=True, h0=7, fresh_blocks_stall=True, revmode="lookback7", inpol_fresh_break=True, kcnt_range=4),
    "C_kr5": dict(BASE, kmode="min", edr="cnt15", plan="both", rev_gate=True, h0=7, fresh_blocks_stall=True, revmode="lookback7", inpol_fresh_break=True, kcnt_range=5),
    "C_kr6": dict(BASE, kmode="min", edr="cnt15", plan="both", rev_gate=True, h0=7, fresh_blocks_stall=True, revmode="lookback7", inpol_fresh_break=True, kcnt_range=6),
    "C_excl": dict(BASE, kmode="min", edr="cnt15", plan="both", rev_gate=True, h0=7, fresh_blocks_stall=True, revmode="lookback7", inpol_fresh_break=True, excl_prog=True),
    "C_excl_s7": dict(BASE, kmode="min", edr="cnt15", plan="both", rev_gate=True, h0=7, fresh_blocks_stall=True, revmode="sched", sched=[7, 7, 7, 7, 7, 7, 7, 0], inpol_fresh_break=True, excl_prog=True),
    "C_excl_s71": dict(BASE, kmode="min", edr="cnt15", plan="both", rev_gate=True, h0=7, fresh_blocks_stall=True, revmode="sched", sched=[7, 7, 7, 7, 7, 7, 1, 0], inpol_fresh_break=True, excl_prog=True),
    "C_excl_exact": dict(BASE, kmode="min", edr="cnt15", plan="both", rev_gate=True, h0=7, fresh_blocks_stall=True, revmode="sched", sched=[7, 6, 5, 4, 3, 2, 1, 0], inpol_fresh_break=True, excl_prog=True),
    "C_excl_s4": dict(BASE, kmode="min", edr="cnt15", plan="both", rev_gate=True, h0=7, fresh_blocks_stall=True, revmode="sched", sched=[4, 4, 4, 4, 4, 4, 4, 0], inpol_fresh_break=True, excl_prog=True),
    # network-faithful candidates
    "N4": dict(BASE, kmode="net", kpenc_thr=4, edr="cnt31any", plan="both", rev_gate=True, h0=7, h0_any=True, fresh_blocks_stall=True, revmode="sched", sched=[7, 7, 7, 7, 7, 7, 7, 0], inpol_prevread=True),
    "N5": dict(BASE, kmode="net", kpenc_thr=5, edr="cnt31any", plan="both", rev_gate=True, h0=7, h0_any=True, fresh_blocks_stall=True, revmode="sched", sched=[7, 7, 7, 7, 7, 7, 7, 0], inpol_prevread=True),
    "N5_h0path": dict(BASE, kmode="net", kpenc_thr=5, edr="cnt31any", plan="both", rev_gate=True, h0=31, h0_any=False, fresh_blocks_stall=True, revmode="sched", sched=[7, 7, 7, 7, 7, 7, 7, 0], inpol_prevread=True),
    "N5_nofresh": dict(BASE, kmode="net", kpenc_thr=5, edr="cnt31any", plan="both", rev_gate=True, h0=7, h0_any=True, fresh_blocks_stall=False, revmode="sched", sched=[7, 7, 7, 7, 7, 7, 7, 0], inpol_prevread=True),
    "N5_noprevread": dict(BASE, kmode="net", kpenc_thr=5, edr="cnt31any", plan="both", rev_gate=True, h0=7, h0_any=True, fresh_blocks_stall=True, revmode="sched", sched=[7, 7, 7, 7, 7, 7, 7, 0], inpol_prevread=False),
    "N5_planfirst": dict(BASE, kmode="net", kpenc_thr=5, edr="cnt31any", plan="first", rev_gate=True, h0=7, h0_any=True, fresh_blocks_stall=True, revmode="sched", sched=[7, 7, 7, 7, 7, 7, 7, 0], inpol_prevread=True),
    "FINAL": dict(BASE, kmode="net", kpenc_thr=5, edr="cnt31any", res_trigger="cnt31any", plan="both", rev_gate=True, h0=31, h0_any=False, fresh_blocks_stall=True, revmode="sched", sched=[7, 7, 7, 7, 7, 7, 7, 0], inpol_prevread=True, readlist=True),
    "FINAL_readonly": dict(BASE, kmode="net", kpenc_thr=5, edr="cnt31any", res_trigger="cnt31any", plan="both", rev_gate=True, h0=31, h0_any=False, fresh_blocks_stall=True, revmode="sched", sched=[7, 7, 7, 7, 7, 7, 7, 0], inpol_prevread=True, readlist=False),
}


def evaluate(cfg, which, verbose):
    ws = v05_load.windows(which)
    conf = [[0] * 4 for _ in range(4)]
    wrong = []
    for w in ws:
        F = features(w["blocks"], cfg)
        c = decide(F, w["t"], cfg)
        o = CN.index(w["oracle_label"])
        conf[o][c] += 1
        if c != o:
            wrong.append(f"{w['id']}:{w['oracle_label'][:4]}->{CN[c][:4]}")
    f1s = []
    for c in range(4):
        tp = conf[c][c]
        fp = sum(conf[r][c] for r in range(4)) - tp
        fn = sum(conf[c][k] for k in range(4)) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    mf1 = sum(f1s) / 4
    print(f"  {which}: macro-F1 {mf1:.4f} wrong {len(wrong)}  rows=oracle cols=net {conf}")
    if verbose:
        print("     " + " ".join(wrong))
    return mf1, wrong


def main():
    names = sys.argv[1:] or list(CONFIGS)
    verbose = True
    for name in names:
        cfg = CONFIGS[name]
        print(f"== {name}: {cfg}")
        evaluate(cfg, "V", verbose)
        evaluate(cfg, "W", verbose)


if __name__ == "__main__":
    main()
