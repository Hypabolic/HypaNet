"""Shared scratch helpers: RawOp factories, rendering, encoding, forward passes, feature readout and
reference-policy comparison.  Only ever run through ./py from output/v03-phase-a/scratch/."""

import torch

import model
import renderer
import tokenizer
import build_weights as bw
import reference_policy as rp

_MODEL = None


def net():
    global _MODEL
    if _MODEL is None:
        _MODEL = bw.build()
    return _MODEL


def op(family, target=None, args=None, outcome="SUCCESS", error_class=None, error_location=None,
       validation=None, changed=None, extra=None, input_tokens=1000, output_tokens=50, dropped=None):
    return {
        "family": family, "target": target, "args": args or {}, "outcome": outcome,
        "error_class": error_class, "error_location": error_location, "validation": validation,
        "changed": changed or [], "extra": extra, "input_tokens": input_tokens,
        "output_tokens": output_tokens, "timestamp": None, "dropped": dropped or [],
    }


def stamp(ops):
    out = []
    for i, o in enumerate(ops):
        o = dict(o)
        if not o.get("_nostamp"):
            o["timestamp"] = f"2026-03-01T09:{(i // 60) % 60:02d}:{i % 60:02d}.{(i * 137) % 1000:03d}+00:00"
        o["input_tokens"] = 1000 + 37 * i
        out.append(o)
    return out


# -- convenient op builders ---------------------------------------------------------------------

def read(path, lines=None, ok=True):
    args = {"line_start": lines[0], "line_end": lines[1]} if lines else {}
    if ok:
        return op("read_file", path, args)
    return op("read_file", path, args, outcome="ERROR", error_class="IOError", error_location=path)


def search(scope, query, limit=None):
    args = {"query": query}
    if limit:
        args["limit"] = limit
    return op("search", scope, args)


def list_dir(path):
    return op("list_dir", path)


def change(path, before, after):
    return {"path": path, "before": before, "after": after}


def edit(path, before, after, ok=True):
    if not ok:
        return op("edit_file", path, outcome="ERROR", error_class="PatchError", error_location=path)
    return op("edit_file", path, changed=[change(path, before, after)] if before != after else [])


def write(path, before, after):
    return op("write_file", path, changed=[change(path, before, after)])


def revert(paths, changes):
    return op("revert", list(paths), changed=changes)


def tests(suite, passed, total, tests_list=None, failing_loc="pkg/core.qx"):
    args = {"tests": tests_list} if tests_list else {}
    val = {"suite": suite, "passed": passed, "total": total}
    if passed < total:
        return op("run_tests", suite, args, outcome="ERROR", error_class="AssertionError",
                  error_location=failing_loc, validation=val)
    return op("run_tests", suite, args, validation=val)


def build(target, ok=True):
    if ok:
        return op("build", target)
    return op("build", target, outcome="ERROR", error_class="CompileError", error_location="pkg/core.qx")


def lint(target, ok=True):
    if ok:
        return op("lint", target)
    return op("lint", target, outcome="ERROR", error_class="LintError", error_location="pkg/core.qx")


def http(method, host, path, ok=True, body=None):
    t = {"method": method, "host": host, "path_template": path}
    args = {"body": body} if body else {}
    if ok:
        return op("http_call", t, args)
    return op("http_call", t, args, outcome="ERROR", error_class="TimeoutError", error_location="gateway")


def poll(resource, state="pending", backoff=None):
    args = {"backoff": backoff} if backoff else {}
    outcome = {"pending": "NEUTRAL", "ready": "SUCCESS", "error": "ERROR"}[state]
    if state == "error":
        return op("poll_status", resource, args, outcome=outcome, error_class="PollError", error_location=resource)
    return op("poll_status", resource, args, outcome=outcome)


def plan(action="sketch"):
    return op("plan", None, {"action_type": action})


def ask(action="clarify"):
    return op("ask", None, {"action_type": action})


# -- running ----------------------------------------------------------------------------------

FEATS = {
    "SI": bw.SI, "INPOL": bw.INPOL, "RESOLV": bw.RESOLV, "REVBLK": bw.REVBLK, "OUTALL": bw.OUTALL,
    "OUTFAIL": bw.OUTFAIL, "ANCH2": bw.ANCH2, "PV": bw.PV, "HASP": bw.HASPASSED, "NEAR": bw.NEAREST,
    "REDVAL": bw.REDVAL, "HUNK": bw.HASHUNK, "AVGSI": bw.AVGSI, "AVGREV": bw.AVGREV, "ANCH8": bw.ANCH8,
    "NEAR4": bw.NEAR4, "KILL": bw.KILL, "LPV": bw.LPV, "FPV": bw.FPV, "V2": bw.V2, "I5": bw.I5, "A4": bw.A4,
    "PROGN": bw.PROGN, "PROGF": bw.PROGF, "SIT": bw.SIT,
}


def encode(ops, sid="s1"):
    ops = stamp(ops)
    blocks = renderer.render_session(ops, sid)
    enc = tokenizer.encode_session(blocks)
    return blocks, enc


def run(enc):
    m = net()
    t = torch.tensor(enc.tokens)
    e = torch.tensor(enc.event_ord)
    o = torch.tensor(enc.offset)
    acts = m.activations(t, e, o, query_positions=[0])
    logits = acts["logits"]
    resid = acts["resid_mlp"][-1]
    return logits, resid, acts


def policy_classes(enc):
    out = []
    for t in range(len(enc.eoe_index)):
        pre = tokenizer.prefix(enc, t)
        out.append(rp.reference_policy(pre.surface, pre.event_ord))
    return out


def compare(ops, sid="s1", verbose=True, feats=("SI", "INPOL", "RESOLV", "REVBLK", "OUTALL", "OUTFAIL", "ANCH2",
                                                 "PV", "NEAR", "REDVAL", "AVGSI", "AVGREV", "ANCH8", "NEAR4",
                                                 "KILL", "V2", "I5", "A4", "PROGN", "PROGF")):
    blocks, enc = encode(ops, sid)
    logits, resid, _ = run(enc)
    pol = policy_classes(enc)
    names = model.CLASS_NAMES
    n_dis = 0
    for t, idx in enumerate(enc.eoe_index):
        netc = int(torch.argmax(logits[idx]))
        mark = "" if netc == pol[t] else "   <<< DISAGREE"
        n_dis += netc != pol[t]
        if verbose is True or (verbose == "dis" and netc != pol[t]):
            head = blocks[t].split("\n")[0]
            head = head[13:70]
            fs = " ".join(f"{k}={float(resid[idx, FEATS[k]]):.2f}" for k in feats)
            print(f"[{t:2d}] pol={names[pol[t]][:4]} net={names[netc][:4]} | {head:57s}{mark}")
            print(f"      {fs}")
            print(f"      logits={[round(float(x), 1) for x in logits[idx]]}")
    return n_dis, len(enc.eoe_index)
