"""Disputes between the two model labellers, the second pass, and the adjudicated answer key.

A node is disputed when the two round-1 labels differ on the binary class (REDUNDANT, KEEP or UNSURE), on the
four-class label, or, for two ECHO or two NARRATE labels, on the pointer target (a target in the other label's equiv
list counts as the same target). Both models then see the guide, the screen and the two labels, anonymised as A and
B in an order fixed per node by sha256("adjudicate:20261116:<screen>:<i>") (the same for both adjudicators; `conf`
is not shown), and each returns a final label with a one-line reason. If the two final labels agree (same tests),
that is the answer; otherwise the node is contested. A node both labellers call UNSURE is contested without a second
pass, and so is any agreed UNSURE.

Output: `labels/<set>.final.jsonl` (labeller "final"): every node record carries `source` (agreed, adjudicated,
contested or unresolved), `candidates` (both round-1 labels) and `binary`, the binary consensus (REDUNDANT or KEEP
when the two final labels share it, else null): a node contested only on its class or target (ECHO against DECOR,
or two ECHO targets) still has a binary label, and the binary headline scores it (SPEC.md §5a, §10). Adjudicated and
contested-after-adjudication records also carry `adjudication` (both second-pass labels with their reasons). A
contested record has class null, conf "contested", target and pattern null; the four-class and pointer metrics leave
it out, and the sensitivity analysis reads its candidates (`resolve_binary`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from modellabel import common as c
from modellabel import prompt as pr
from modellabel.to_records import node_record, normalise_reply, raw_path

lk = c.lk
ADJ_SEED = 20261116
ADJ_NAMESPACE = "adjudicate"
ORDER = (c.LABELLERS["opus"], c.LABELLERS["codex"])


def binary(cls):
    if cls is None:
        return None
    if cls in lk.REDUNDANT:
        return "REDUNDANT"
    return cls  # KEEP or UNSURE


def same_target(a: dict, b: dict) -> bool:
    ta, tb = a.get("target"), b.get("target")
    if ta == tb:
        return True
    return (ta is not None and ta in (b.get("equiv") or [])) or (tb is not None and tb in (a.get("equiv") or []))


def dispute_reasons(a: dict, b: dict) -> list:
    """Why two labels of one node disagree (empty list = they agree)."""
    out = []
    if binary(a["class"]) != binary(b["class"]):
        out.append("binary")
    if a["class"] != b["class"]:
        out.append("class")
    elif a["class"] in ("ECHO", "NARRATE") and not same_target(a, b):
        out.append("target")
    return out


def ab_order(screen_id: str, i: int) -> tuple:
    """(labeller shown as A, labeller shown as B) for one disputed node; the same for both adjudicators."""
    h = hashlib.sha256(f"{ADJ_NAMESPACE}:{ADJ_SEED}:{screen_id}:{i}".encode()).digest()
    return ORDER if h[0] % 2 == 0 else ORDER[::-1]


def tie_pick(screen_id: str, i: int) -> str:
    """Whose pattern code a merged label keeps when the two agree on class and target but name different codes (both
    stay in `candidates`): by hash, so neither family's codes win every tie."""
    h = hashlib.sha256(f"pattern:{ADJ_SEED}:{screen_id}:{i}".encode()).digest()
    return ORDER[h[0] % 2]


def round1(ls: c.LabelSet, labels_dir: Path, guide_sha: str | None = None) -> dict:
    """{screen id: {labeller: labels_by_i}} for screens both labellers finished with valid replies, plus the raw
    records' guide hash. Screens missing either side are left out."""
    out = {}
    for scr in ls.screens:
        sides = {}
        for lab in ORDER:
            raw = c.read_json(raw_path(labels_dir, ls.name, lab, scr.id))
            if not raw or raw.get("status") != "ok" or raw.get("screen_sha256") != scr.sha256:
                break
            if guide_sha and raw.get("guide_sha256") != guide_sha:
                break
            labels, errors, _ = normalise_reply(raw["reply"], scr)
            if errors:
                break
            sides[lab] = {"labels": labels, "reply": raw["reply"], "guide_sha256": raw["guide_sha256"],
                          "time": raw.get("time")}
        if len(sides) == 2 and sides[ORDER[0]]["guide_sha256"] == sides[ORDER[1]]["guide_sha256"]:
            out[scr.id] = sides
    return out


def screen_disputes(sides: dict) -> dict:
    """{"adjudicate": [i...], "both_unsure": [i...], "agreed": [i...], "unresolved": [i...]}"""
    a, b = sides[ORDER[0]]["labels"], sides[ORDER[1]]["labels"]
    res = {"adjudicate": [], "both_unsure": [], "agreed": [], "unresolved": [], "reasons": {}}
    for i in sorted(set(a) | set(b)):
        la, lb = a.get(i), b.get(i)
        if la["class"] is None and lb["class"] is None:
            res["unresolved"].append(i)
        elif la["class"] == "UNSURE" and lb["class"] == "UNSURE":
            res["both_unsure"].append(i)
        else:
            r = dispute_reasons(la, lb)
            if r:
                res["adjudicate"].append(i)
                res["reasons"][i] = r
            else:
                res["agreed"].append(i)
    return res


def adjudication_prompt(scr: c.Screen, guide_text: str, sides: dict, disp: dict) -> tuple:
    """(prompt, order) where order maps each disputed node i (as a string) to {"A": labeller, "B": labeller}."""
    base = sides[ORDER[0]]["labels"]
    agreed = [(i, base[i]) for i in disp["agreed"]]
    disputed, order = [], {}
    for i in disp["adjudicate"]:
        first, second = ab_order(scr.id, i)
        disputed.append((i, sides[first]["labels"][i], sides[second]["labels"][i]))
        order[str(i)] = {"A": first, "B": second}
    return pr.render_adjudication(scr, guide_text, agreed, disputed), order


def validate_adjudication(obj: dict, scr: c.Screen, disputed: list):
    errors = c.check_schema(obj, c.load_schema(c.ADJ_SCHEMA_PATH))
    if errors:
        return errors, []
    if obj.get("screen") != scr.id:
        errors.append(f"screen is {obj.get('screen')!r}, expected {scr.id!r}")
    nodes = scr.by_i()
    seen = set()
    for ent in obj["nodes"]:
        i = ent["i"]
        if i not in disputed:
            errors.append(f"node {i}: not a disputed node (adjudicate only {', '.join(map(str, disputed))})")
            continue
        if i in seen:
            errors.append(f"node {i}: answered more than once")
        seen.add(i)
        if ent["class"] is None:
            errors.append(f"node {i}: class must not be null")
            continue
        for name, ref in [("target", ent.get("target"))] + [("equiv", x) for x in ent.get("equiv") or []]:
            if ref is not None and (ref not in nodes or ref == i):
                errors.append(f"node {i}: {name} {ref} is not another node on this screen")
        if not errors:
            rec = node_record({**ent, "role_ok": True}, nodes[i], nodes, scr.id, "x", "0" * 64,
                              "2026-01-01T00:00:00Z")
            errs, _ = lk.validate_record(rec, "x")
            errors += [f"node {i}: {e}" for e in errs]
    missing = [i for i in disputed if i not in seen]
    if missing:
        errors.append(f"disputed nodes not answered: {', '.join(map(str, missing))}")
    return errors, []


def adj_raw_path(labels_dir: Path, set_name: str, labeller: str, sid: str) -> Path:
    return raw_path(labels_dir, set_name, labeller, sid, phase="adjudicate")


# ---------------------------------------------------------------- final labels


def _merge(la: dict, lb: dict, first: str = ORDER[0]) -> dict:
    """One label from two (la from ORDER[0], lb from ORDER[1]) that agree on class and target: the pattern of `first`
    when both name one and they differ (tie_pick; both are kept in candidates), judgement if either is, ct and partial
    if either is, equiv and asides united."""
    out = dict(la)
    out["conf"] = "judgement" if "judgement" in (la.get("conf"), lb.get("conf")) else "clear"
    out["ct"] = bool(la.get("ct") or lb.get("ct"))
    out["partial"] = bool(la.get("partial") or lb.get("partial")) and la["class"] == "KEEP"
    pa, pb = la.get("pattern"), lb.get("pattern")
    out["pattern"] = (pb if pa is None else pa if pb is None else pa if first == ORDER[0] else pb)
    tgt = out.get("target")
    if tgt is None and lb.get("target") is not None:
        out["target"] = tgt = lb["target"]
    eq = [x for x in dict.fromkeys((la.get("equiv") or []) + (lb.get("equiv") or [])) if x != tgt]
    out["equiv"] = eq if out["class"] == "ECHO" else []
    out["aside"] = list(dict.fromkeys((la.get("aside") or []) + (lb.get("aside") or [])))
    return out


def _cand(lab: dict, labeller: str, nodes: dict, reason_key: str = "why") -> dict:
    loc = lambda i: nodes[i]["loc"] if i is not None and i in nodes else None  # noqa: E731
    return {"labeller": labeller, "class": lab.get("class"), "target": loc(lab.get("target")),
            "pattern": lab.get("pattern"), "conf": lab.get("conf"), "ct": bool(lab.get("ct")),
            "partial": bool(lab.get("partial")), "equiv": [loc(x) for x in lab.get("equiv") or [] if x in nodes],
            "aside": list(lab.get("aside") or []), "reason": lab.get(reason_key)}


def final_screen(scr: c.Screen, sides: dict, adj: dict, time: str) -> list:
    """Final records for one screen. adj: {labeller: {i: label}} from valid second-pass replies (may be empty)."""
    nodes = scr.by_i()
    la, lb = sides[ORDER[0]]["labels"], sides[ORDER[1]]["labels"]
    guide_sha = sides[ORDER[0]]["guide_sha256"]
    disp = screen_disputes(sides)
    recs = []
    lab_id = c.FINAL_LABELLER
    for i in sorted(la):
        a, b = la[i], lb[i]
        cands = [_cand(a, ORDER[0], nodes), _cand(b, ORDER[1], nodes)]
        role_ok = bool(a.get("role_ok", True) and b.get("role_ok", True))
        role_fix = a.get("role_fix") if not a.get("role_ok", True) else b.get("role_fix")
        extra = {}
        first = tie_pick(scr.id, i)
        fin = [a, b]
        if i in disp["unresolved"]:
            lab, source = _merge(a, b, first), "unresolved"
        elif i in disp["agreed"] and a["class"] != "UNSURE":
            lab, source = _merge(a, b, first), "agreed"
        elif i in disp["adjudicate"]:
            xa, xb = (adj.get(ORDER[0]) or {}).get(i), (adj.get(ORDER[1]) or {}).get(i)
            fin = [xa, xb]
            extra["adjudication"] = [_cand(x, who, nodes, "reason") for x, who in ((xa, ORDER[0]), (xb, ORDER[1]))
                                     if x is not None]
            if xa is not None and xb is not None and not dispute_reasons(xa, xb) and xa["class"] != "UNSURE":
                lab, source = _merge(xa, xb, first), "adjudicated"
            else:
                lab, source = None, "contested"
        else:  # both UNSURE, or agreed UNSURE
            lab, source = None, "contested"
        bins = {binary(x["class"]) if x is not None else None for x in fin}
        extra["binary"] = next(iter(bins)) if len(bins) == 1 and bins <= {"REDUNDANT", "KEEP"} else None
        if lab is None:
            lab = {"class": None, "target": None, "pattern": None, "equiv": [], "partial": False,
                   "ct": bool(a.get("ct") or b.get("ct")), "conf": "contested",
                   "aside": list(dict.fromkeys((a.get("aside") or []) + (b.get("aside") or [])))}
        lab = {**lab, "role_ok": role_ok, "role_fix": role_fix}
        rec = node_record(lab, nodes[i], nodes, scr.id, lab_id, guide_sha, time)
        rec.update(source=source, candidates=cands, **extra)
        recs.append(rec)
    meta = {"labeller": lab_id, "guide_sha256": guide_sha, "time": time}
    ra, rb = sides[ORDER[0]]["reply"], sides[ORDER[1]]["reply"]
    asides, missed = screen_level(scr, ra, rb)
    if asides:
        recs.append({"screen": scr.id, "aside_screen": asides, **meta})
    if missed:
        recs.append({"screen": scr.id, "missed": missed, **meta})
    recs.append({"screen": scr.id, "done": True, "nodes": len(la), **meta})
    return recs


def _norm(t: str) -> str:
    return " ".join(str(t).split()).lower()


def match_missed(ma: list, mb: list) -> tuple:
    """Pairs of missed-copy items the two labellers both reported: the same line, and one normalised text contained in
    the other. Returns (pairs, only_a, only_b)."""
    used, pairs = set(), []
    for x in ma:
        for k, y in enumerate(mb):
            if k in used or x["line"] != y["line"]:
                continue
            tx, ty = _norm(x["text"]), _norm(y["text"])
            if tx and ty and (tx in ty or ty in tx):
                used.add(k)
                pairs.append((x, y))
                break
    paired_a = {id(x) for x, _ in pairs}
    return pairs, [x for x in ma if id(x) not in paired_a], [y for k, y in enumerate(mb) if k not in used]


def screen_level(scr: c.Screen, ra: dict, rb: dict) -> tuple:
    """Screen-level records of the key, by one rule for both kinds: an item is in the key only when both labellers
    raised it. Screen asides: codes both raised, their nodes united. Missed copy: items both reported (match_missed),
    with the longer text, and class UNSURE when the two classes differ. One-sided items are counted in the agreement
    report (agree.py), not keyed."""
    nodes = scr.by_i()
    codes_a = {it["code"]: it["nodes"] for it in ra.get("aside_screen") or []}
    codes_b = {it["code"]: it["nodes"] for it in rb.get("aside_screen") or []}
    asides = [{"code": k, "nodes": [nodes[x]["loc"] for x in dict.fromkeys(codes_a[k] + codes_b[k]) if x in nodes]}
              for k in codes_a if k in codes_b]
    pairs, _, _ = match_missed(ra.get("missed") or [], rb.get("missed") or [])
    missed = [{"line": x["line"], "text": max((x["text"], y["text"]), key=len),
               "class": x["class"] if x["class"] == y["class"] else "UNSURE"} for x, y in pairs]
    return asides, missed


def resolve_binary(rec: dict, mode: str):
    """The binary class (REDUNDANT, KEEP or None = not scored) of a final node record under one of the fixed
    resolutions of the sensitivity analysis (SPEC.md §5a), applied to every system alike:

    - "key": the key as it stands (`binary`; contested without a binary consensus -> None);
    - "redundant" / "keep": every node without a binary consensus counted REDUNDANT / KEEP;
    - "opus55" / "codex61": that labeller's candidate, its second-pass label where there is one, else round 1; an
      UNSURE candidate -> None.
    Nodes with no literal words (source unresolved) are None under every resolution."""
    if rec.get("source") == "unresolved" or (rec.get("class") is None and rec.get("source") != "contested"):
        return None
    b = rec.get("binary")
    if mode == "key":
        return b
    if b is not None:
        return b
    if mode == "redundant":
        return "REDUNDANT"
    if mode == "keep":
        return "KEEP"
    if mode in ORDER:
        second = {x.get("labeller"): x for x in rec.get("adjudication") or []}
        first = {x.get("labeller"): x for x in rec.get("candidates") or []}
        lab = second.get(mode) or first.get(mode)
        cls = binary(lab.get("class")) if lab else None
        return cls if cls in ("REDUNDANT", "KEEP") else None
    raise ValueError(mode)


RESOLUTIONS = ("key", "redundant", "keep") + ORDER


def load_adjudications(ls: c.LabelSet, labels_dir: Path, r1: dict) -> dict:
    """{screen: {labeller: {i: label}}} from valid second-pass raw files whose disputes still match round 1."""
    out = {}
    for sid, sides in r1.items():
        scr = ls.get(sid)
        disp = screen_disputes(sides)
        per = {}
        for lab in ORDER:
            raw = c.read_json(adj_raw_path(labels_dir, ls.name, lab, sid))
            if not raw or raw.get("status") != "ok" or raw.get("disputed") != disp["adjudicate"] or not scr:
                continue
            if raw.get("round1_sha256") != round1_sha(sides):
                continue
            errs, _ = validate_adjudication(raw["reply"], scr, disp["adjudicate"])
            if not errs:
                per[lab] = {e["i"]: e for e in raw["reply"]["nodes"]}
        out[sid] = per
    return out


def round1_status(labels_dir: Path, set_name: str, lab: str, scr: c.Screen) -> str:
    """Why a screen has no usable round-1 label from one labeller: none (never run), stale (the screen changed), the
    raw status (refused, invalid, error, ...), or "ok" (usable on this side)."""
    raw = c.read_json(raw_path(labels_dir, set_name, lab, scr.id))
    if not raw:
        return "none"
    if raw.get("screen_sha256") != scr.sha256:
        return "stale"
    if raw.get("status") != "ok":
        return str(raw.get("status"))
    return "ok" if not normalise_reply(raw["reply"], scr)[1] else "invalid"


def round1_sha(sides: dict) -> str:
    return c.sha256_text(json.dumps([sides[x]["reply"] for x in ORDER], sort_keys=True, ensure_ascii=False))


def build_final(ls: c.LabelSet, labels_dir: Path = c.LABELS_DIR, guide: Path = c.GUIDE) -> dict:
    r1 = round1(ls, labels_dir)
    adj = load_adjudications(ls, labels_dir, r1)
    path = lk.label_path(labels_dir, ls.name, c.FINAL_LABELLER)
    lines = []
    counts = {"screens": 0, "screens_skipped": 0, "sources": {}, "pending_adjudication": 0, "missing": {},
              "pending": [], "second_pass_refused": []}
    for scr in ls.screens:
        if scr.id not in r1:
            counts["screens_skipped"] += 1
            counts["missing"][scr.id] = {lab: round1_status(labels_dir, ls.name, lab, scr) for lab in ORDER}
            continue
        disp = screen_disputes(r1[scr.id])
        if disp["adjudicate"] and len(adj.get(scr.id) or {}) < 2:
            # a second pass that was refused is final (its nodes are contested); anything else is still to run
            st = {lab: (c.read_json(adj_raw_path(labels_dir, ls.name, lab, scr.id)) or {}).get("status")
                  for lab in ORDER if lab not in (adj.get(scr.id) or {})}
            if all(v == "refused" for v in st.values()):
                counts["second_pass_refused"].append(scr.id)
            else:
                counts["pending_adjudication"] += 1
                counts["pending"].append(scr.id)
        times = [r1[scr.id][x]["time"] for x in ORDER if r1[scr.id][x]["time"]]
        for rec in final_screen(scr, r1[scr.id], adj.get(scr.id) or {}, max(times) if times else lk.now_iso()):
            if "source" in rec:
                counts["sources"][rec["source"]] = counts["sources"].get(rec["source"], 0) + 1
            lines.append(json.dumps(rec, ensure_ascii=False, separators=(", ", ": ")))
        counts["screens"] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    c.write_text_atomic(path, "".join(x + "\n" for x in lines))
    from modellabel.to_records import write_meta
    write_meta(labels_dir, ls, c.FINAL_LABELLER, guide)
    counts["file"] = str(path)
    return counts
