"""Scenario library (v1: adds dp to http/poll). Builds RawOp dicts for hand-designed sessions, renders
them with the packet renderer, and labels every window with a small oracle implementing my reading of
SPEC.md v0.2 rules 6.3 (for self-labelling my own texts only).

No random module: all variation comes from hashlib of (session id, index).
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone

import renderer

EFFECTS = ("PROGRESS", "NEW_FACT", "RESOLVES", "RETRY", "POLL", "REVERSION", "REDUNDANT", "NONE", "FAIL0")
R_MAX, P_MAX, K_REC, N_STALL, N_REV, EPS_P, F_PROD = 3, 5, 4, 5, 2, 0.02, 2


def hnum(*parts, lo=0, hi=1000):
    h = hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()
    return lo + int(h[:8], 16) % (hi - lo)


class Session:
    """Accumulates ops with annotations: effect, progress delta, episode role."""

    def __init__(self, sid, start="2026-03-01T09:14:03.250+00:00"):
        self.sid = sid
        self.ops = []
        self.ann = []
        self.t = datetime.fromisoformat(start)
        self.in_tok = hnum(sid, "in0", lo=1200, hi=2600)
        self.files = {}  # path -> current text

    def _stamp(self, i):
        self.t += timedelta(seconds=hnum(self.sid, "dt", i, lo=2, hi=40), milliseconds=hnum(self.sid, "ms", i, lo=0, hi=999))
        return self.t.isoformat(timespec="milliseconds")

    def add(self, family, target, eff, dp=0.0, args=None, outcome="SUCCESS", error_class=None, error_location=None,
            validation=None, changed=None, out_tok=None, extra=None, dropped=None):
        i = len(self.ops)
        self.in_tok += hnum(self.sid, "din", i, lo=40, hi=400)
        if out_tok is None:
            out_tok = hnum(self.sid, "out", i, lo=12, hi=400)
        op = {
            "family": family, "target": target, "args": args or {}, "outcome": outcome,
            "error_class": error_class, "error_location": error_location, "validation": validation,
            "changed": changed or [], "extra": extra, "input_tokens": self.in_tok, "output_tokens": out_tok,
            "timestamp": self._stamp(i), "dropped": sorted(dropped or []),
        }
        assert eff in EFFECTS, eff
        self.ops.append(op)
        self.ann.append({"eff": eff, "dp": dp})
        return op

    # ---- tool helpers -------------------------------------------------------------------------
    def read(self, path, eff="NEW_FACT", lines=None, outcome="SUCCESS", error_class=None):
        args = {}
        if lines:
            args = {"line_start": lines[0], "line_end": lines[1]}
        return self.add("read_file", path, eff, args=args, outcome=outcome, error_class=error_class,
                        error_location=path if outcome == "ERROR" else None, out_tok=hnum(self.sid, "rd", len(self.ops), lo=40, hi=900))

    def search(self, scope, query, eff="NEW_FACT", limit=None):
        args = {"query": query}
        if limit:
            args["limit"] = limit
        return self.add("search", scope, eff, args=args, out_tok=hnum(self.sid, "se", len(self.ops), lo=10, hi=120))

    def list_dir(self, path, eff="NEW_FACT"):
        return self.add("list_dir", path, eff, out_tok=hnum(self.sid, "ls", len(self.ops), lo=10, hi=60))

    def plan(self, action, eff="NONE", family="plan"):
        return self.add(family, None, eff, args={"action_type": action}, out_tok=hnum(self.sid, "pl", len(self.ops), lo=200, hi=900))

    def set_file(self, path, text):
        self.files[path] = text

    def edit(self, path, new_text, eff="PROGRESS", dp=0.0, outcome="SUCCESS"):
        before = self.files.get(path)
        changed = []
        if new_text != before:
            changed = [{"path": path, "before": before, "after": new_text}]
            self.files[path] = new_text
        return self.add("edit_file", path, eff, dp=dp, changed=changed, outcome=outcome,
                        out_tok=hnum(self.sid, "ed", len(self.ops), lo=120, hi=600))

    def write(self, path, new_text, eff="PROGRESS", dp=0.0):
        before = self.files.get(path)
        changed = [] if new_text == before else [{"path": path, "before": before, "after": new_text}]
        self.files[path] = new_text
        return self.add("write_file", path, eff, dp=dp, args={"content": "x"}, changed=changed,
                        out_tok=hnum(self.sid, "wr", len(self.ops), lo=200, hi=900))

    def revert(self, paths_texts, eff="REVERSION"):
        changed = []
        for p, txt in paths_texts.items():
            before = self.files.get(p)
            if txt != before:
                changed.append({"path": p, "before": before, "after": txt})
            self.files[p] = txt
        return self.add("revert", list(paths_texts), eff, changed=changed, out_tok=hnum(self.sid, "rv", len(self.ops), lo=8, hi=40))

    def tests(self, suite, passed, total, eff, dp=0.0, error_class=None, error_location=None, tests=None):
        ok = passed == total
        args = {"tests": tests} if tests else {}
        return self.add("run_tests", suite, eff, dp=dp, args=args, outcome="SUCCESS" if ok else "ERROR",
                        error_class=None if ok else error_class, error_location=None if ok else error_location,
                        validation={"suite": suite, "passed": passed, "total": total},
                        out_tok=hnum(self.sid, "te", len(self.ops), lo=60, hi=400))

    def build(self, target, ok, eff, dp=0.0, error_class=None, error_location=None, family="build"):
        return self.add(family, target, eff, dp=dp, outcome="SUCCESS" if ok else "ERROR",
                        error_class=None if ok else error_class, error_location=None if ok else error_location,
                        out_tok=hnum(self.sid, "bd", len(self.ops), lo=20, hi=200))

    def http(self, method, host, path, ok, eff, dp=0.0, body=None, error_class=None, error_location=None):
        args = {"body": body} if body is not None else {}
        return self.add("http_call", {"method": method, "host": host, "path_template": path}, eff, dp=dp, args=args,
                        outcome="SUCCESS" if ok else "ERROR", error_class=None if ok else error_class,
                        error_location=None if ok else error_location, out_tok=hnum(self.sid, "ht", len(self.ops), lo=10, hi=200))

    def poll(self, resource, state, eff, dp=0.0, backoff=None):
        args = {"backoff": backoff} if backoff is not None else {}
        outcome = {"ready": "SUCCESS", "pending": "NEUTRAL", "error": "ERROR"}[state]
        return self.add("poll_status", resource, eff, dp=dp, args=args, outcome=outcome, out_tok=hnum(self.sid, "po", len(self.ops), lo=8, hi=40))

    # ---- rendering and oracle -----------------------------------------------------------------
    def blocks(self):
        return renderer.render_session(self.ops, self.sid)

    def labels(self):
        """Per-window label by my reading of SPEC rules 6.3 (first match wins)."""
        n = len(self.ops)
        ep_active, ep_in_policy, resolved_at = False, False, -99
        retries = 0
        eff_final = []
        ep_state = []
        for i, a in enumerate(self.ann):
            e = a["eff"]
            if e == "FAIL0":
                ep_active, ep_in_policy, retries = True, True, 0
                eff_final.append("NONE")
            elif e in ("RETRY", "POLL"):
                retries += 1
                limit = R_MAX if e == "RETRY" else P_MAX
                if ep_active and retries <= limit:
                    ep_in_policy = True
                    eff_final.append("IN_POLICY")
                else:
                    ep_in_policy = False
                    eff_final.append("REDUNDANT")
            elif e == "RESOLVES":
                ep_active, ep_in_policy = False, False
                resolved_at = i
                eff_final.append("RESOLVES")
            else:
                eff_final.append(e)
            ep_state.append((ep_active and ep_in_policy, resolved_at))
        out = []
        for t in range(n):
            lo = max(0, t - 7)
            win = range(lo, t + 1)
            dp = sum(self.ann[i]["dp"] for i in win)
            F = sum(1 for i in win if eff_final[i] == "NEW_FACT")
            I = sum(1 for i in win if eff_final[i] in ("REDUNDANT", "REVERSION"))
            V = sum(1 for i in win if eff_final[i] == "REVERSION")
            in_pol, res_at = ep_state[t]
            if V >= N_REV and dp <= EPS_P:
                out.append("OSCILLATING")
            elif dp <= EPS_P and F == 0 and I >= N_STALL and eff_final[t] in ("REDUNDANT", "REVERSION") and not in_pol:
                out.append("STALLED")
            elif in_pol or (t - res_at) < K_REC:
                out.append("RECOVERY")
            elif dp > EPS_P or F >= F_PROD:
                out.append("PRODUCTIVE")
            else:
                out.append("AMBIG")
        return out, eff_final


CLASS_IDX = {"PRODUCTIVE": 0, "RECOVERY": 1, "STALLED": 2, "OSCILLATING": 3}


def write_request(path, sessions, model="original", windows_mode="all", ret=None, meta_path=None):
    """sessions: list of Session. windows_mode: 'all' or 'labelled' (only non-AMBIG windows, always incl. last)."""
    req = {"model": model, "sessions": []}
    meta = []
    for s in sessions:
        blocks = s.blocks()
        labels, effs = s.labels()
        n = len(blocks)
        if windows_mode == "all":
            wins = list(range(n))
        else:
            wins = [i for i in range(n) if labels[i] != "AMBIG"]
            if not wins or wins[-1] != n - 1:
                wins.append(n - 1)
        req["sessions"].append({"id": s.sid, "blocks": blocks, "windows": wins})
        meta.append({"id": s.sid, "labels": labels, "effs": effs, "windows": wins, "n": n})
    if ret:
        req["return"] = ret
    with open(path, "w") as f:
        json.dump(req, f)
    if meta_path:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
    return req, meta


# ---- file text helpers --------------------------------------------------------------------------

def ftext(tag, n=6):
    return "\n".join(f"{tag} line {k}" for k in range(n))


def variant(base, k, note="patched"):
    lines = base.split("\n")
    lines[k % len(lines)] = lines[k % len(lines)] + f" {note}"
    return "\n".join(lines)
