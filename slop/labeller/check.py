#!/usr/bin/env python3
"""Validate and summarise a slop label file; optionally export the latest-wins view.

    python3 slop/labeller/check.py slop/labels/<set>.<labeller>.jsonl [--screens DIR] [--guide FILE]
        [--export OUT.jsonl] [--json]

Checks: every record is schema-valid (SPEC.md §5); every target, equiv, aside and loc exists in its screen; every
screen of the set is complete (a label on every labelable node, plus a done record); a single guide hash, and
whether it equals the current LABELLING.md. Prints counts per class and per pattern. Exit status 1 on any error.

The screens directory comes from --screens, else from the <set>.<labeller>.meta.json file the server writes beside
the label file. Without screens, targets are checked against the labelled locs only and completeness is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labelkit as lk  # noqa: E402


def check(path: Path, screens_dir=None, guide=None, order=None):
    path = Path(path)
    errors, warnings, info = [], [], []
    set_name, labeller = lk.parse_label_filename(path)
    if set_name is None:
        warnings.append(f"file name {path.name} is not <set>.<labeller>.jsonl; labeller not checked")
    lf = lk.LabelFile.load(path)
    for ln, e in lf.bad_lines:
        errors.append(f"line {ln}: {e}")

    # schema, every record (superseded ones too)
    for ln, rec in lf.records:
        errs, warns = lk.validate_record(rec, labeller)
        errors += [f"line {ln}: {e}" for e in errs]
        warnings += [f"line {ln}: {w}" for w in warns if not _superseded(lf, rec)]

    # screens
    meta = None
    mp = lk.meta_path(path)
    if mp.exists():
        meta = json.loads(mp.read_text())
    if screens_dir is None and meta:
        screens_dir = meta.get("screens_dir")
        order = order or meta.get("order")
        info.append(f"screens from {mp.name}: {screens_dir}")
    screens = None
    if screens_dir:
        if not Path(screens_dir).is_dir():
            errors.append(f"screens directory {screens_dir} not found")
        else:
            loaded, problems = lk.load_screens(Path(screens_dir), order if order and Path(order).exists() else None)
            warnings += [f"screens: {p}" for p in problems]
            screens = {s["screen"]["id"]: s for s in loaded}
            if meta:
                old = {s["id"]: s["sha256"] for s in meta.get("screens", [])}
                changed = [sid for sid, s in screens.items() if sid in old and old[sid] != s["sha256"]]
                if changed:
                    warnings.append(f"{len(changed)} screen files changed since the server last started "
                                    f"(locs may have moved): {', '.join(changed[:5])}")
    else:
        warnings.append("no screens directory: targets checked against labelled locs only; completeness skipped")

    latest = lf.latest
    by_screen = defaultdict(dict)
    for key, rec in latest.items():
        by_screen[key[0]][key[1:] if key[1] == "node" else key[1]] = rec

    # targets and coverage
    for sid, recs in sorted(by_screen.items(), key=lambda x: str(x[0])):
        if screens is not None and sid not in screens:
            errors.append(f"screen {sid}: not in the screens directory")
            continue
        if screens is not None:
            sc = screens[sid]["screen"]
            nodes = {lk.loc_key(n["loc"]): n for n in sc["nodes"]}
        else:
            nodes = {k[1]: None for k in recs if isinstance(k, tuple)}
        for key, rec in recs.items():
            if isinstance(key, tuple):  # node
                lab = nodes.get(key[1], "missing") if screens is not None else None
                if screens is not None and lab == "missing":
                    errors.append(f"screen {sid}: label on loc {rec['loc']} which is not a node")
                    continue
                if lab and lk.node_is_context(lab):
                    errors.append(f"screen {sid}: node {lab['i']} is context only but labelled")
                for name, ref in [("target", rec.get("target"))] + [("equiv", x) for x in rec.get("equiv") or []]:
                    if ref is not None and lk.is_loc(ref) and lk.loc_key(ref) not in nodes:
                        errors.append(f"screen {sid}: {_n(lab, rec)} {name} {ref} does not exist")
                if lab and rec.get("class") == "ECHO" and rec.get("target") is not None:
                    tgt = nodes.get(lk.loc_key(rec["target"]))
                    if tgt and _outranks(lab, tgt):
                        warnings.append(f"screen {sid}: {_n(lab, rec)} ({lab['role']}) is ECHO of node {tgt['i']} "
                                        f"({tgt['role']}), which ranks lower (LABELLING.md §5)")
                if lab and rec.get("role_ok") is False and rec.get("role_fix") == lab.get("role"):
                    warnings.append(f"screen {sid}: {_n(lab, rec)} role_fix equals the extractor role")
                for x in rec.get("equiv") or []:
                    other = recs.get(("node", lk.loc_key(x)))
                    if other is None or other.get("class") != "ECHO":
                        warnings.append(f"screen {sid}: {_n(lab, rec)} equiv {x} is not labelled ECHO")
            elif key == "aside_screen":
                for it in rec.get("aside_screen") or []:
                    for x in it.get("nodes", []):
                        if lk.is_loc(x) and lk.loc_key(x) not in nodes:
                            errors.append(f"screen {sid}: aside {it.get('code')} node {x} does not exist")

    # completeness
    complete = 0
    if screens is not None:
        for sid, s in screens.items():
            lab = lk.labelable_nodes(s["screen"])
            recs = by_screen.get(sid, {})
            missing = [n["i"] for n in lab if not lk.node_complete(recs.get(("node", lk.loc_key(n["loc"]))))]
            done = "done" in recs
            if missing:
                errors.append(f"screen {sid} ({s['file']}): {len(missing)} of {len(lab)} nodes unlabelled "
                              f"(i = {', '.join(map(str, missing[:12]))}{' …' if len(missing) > 12 else ''})")
            elif not done:
                errors.append(f"screen {sid} ({s['file']}): every node labelled but no done record")
            else:
                complete += 1

    # guide hash
    eff_hashes = Counter(r.get("guide_sha256") for r in latest.values())
    all_hashes = Counter(r.get("guide_sha256") for _, r in lf.records)
    if len(eff_hashes) > 1:
        errors.append("more than one guide hash among the effective (latest) records: "
                      + ", ".join(f"{str(h)[:12]} ×{n}" for h, n in eff_hashes.most_common()))
    elif len(all_hashes) > 1:
        warnings.append("superseded records carry other guide hashes: "
                        + ", ".join(f"{str(h)[:12]} ×{n}" for h, n in all_hashes.most_common()))
    cur = None
    if guide and Path(guide).exists():
        cur = lk.sha256_file(Path(guide))
        if eff_hashes and set(eff_hashes) != {cur}:
            warnings.append(f"label hash differs from the current guide {Path(guide).name} ({cur[:12]}); "
                            "freeze.py --check would refuse these labels against that guide")

    return {"file": str(path), "set": set_name, "labeller": labeller, "records": len(lf.records),
            "latest": latest, "screens": screens, "complete": complete, "errors": errors, "warnings": warnings,
            "info": info, "guide_hashes": dict(eff_hashes), "current_guide": cur}


def _superseded(lf, rec):
    return lf.latest.get(lk.record_key(rec)) is not rec


def _n(node, rec):
    return f"node {node['i']}" if node else f"loc {rec.get('loc')}"


def _outranks(a, b):
    return lk.role_rank(a.get("role"), a.get("flags")) < lk.role_rank(b.get("role"), b.get("flags"))


def summary(res):
    nodes = [r for k, r in res["latest"].items() if k[1] == "node"]
    cls = Counter(r.get("class") or ("(contested)" if r.get("conf") == "contested" else "(unresolved)")
                  for r in nodes)
    pat = Counter((r.get("class"), r.get("pattern")) for r in nodes if r.get("class") in lk.REDUNDANT)
    krule = Counter(r.get("pattern") for r in nodes if r.get("class") == "KEEP" and r.get("pattern"))
    asides = Counter(a for r in nodes for a in r.get("aside") or [])
    screen_asides = Counter(it.get("code") for k, r in res["latest"].items() if k[1] == "aside_screen"
                            for it in r.get("aside_screen") or [])
    missed = Counter(it.get("class") for k, r in res["latest"].items() if k[1] == "missed"
                     for it in r.get("missed") or [])
    labelled = [r for r in nodes if r.get("class")]
    n = len(labelled)
    red = sum(1 for r in labelled if r["class"] in lk.REDUNDANT)
    return {
        "nodes": len(nodes), "classes": dict(cls), "redundant": red,
        "patterns": {f"{c} {p or '(none)'}": v for (c, p), v in sorted(pat.items(), key=lambda x: (
            lk.CLASSES.index(x[0][0]), _code_sort(x[0][1])))},
        "k_rules": dict(sorted(krule.items(), key=lambda x: _code_sort(x[0]))),
        "asides": dict(asides), "screen_asides": dict(screen_asides), "missed": dict(missed),
        "conf": dict(Counter(r.get("conf") for r in labelled)),
        "ct": sum(1 for r in labelled if r.get("ct")), "partial": sum(1 for r in labelled if r.get("partial")),
        "role_errors": sum(1 for r in nodes if r.get("role_ok") is False),
        "echo_null_target": sum(1 for r in labelled if r["class"] == "ECHO" and r.get("target") is None),
        "unsure_share": (cls.get("UNSURE", 0) / n) if n else 0.0,
        "screens_done": sum(1 for k in res["latest"] if k[1] == "done"),
    }


def _code_sort(c):
    if not c:
        return (9, 0, "")
    head = "".join(ch for ch in c if ch.isalpha() and ch.isupper())[:1]
    digits = "".join(ch for ch in c[1:] if ch.isdigit())
    return (0 if head == "R" else 1, int(digits or 0), c)


def export(res, out):
    rows = sorted(res["latest"].values(), key=lambda r: (str(r.get("screen")), lk.record_kind(r) != "node",
                                                         r.get("loc") or [0], lk.record_kind(r) or ""))
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n")
    return len(rows)


def print_report(res, s):
    w = sys.stdout.write
    w(f"{res['file']}\n  set {res['set']} · labeller {res['labeller']} · {res['records']} records, "
      f"{len(res['latest'])} effective\n")
    for i in res["info"]:
        w(f"  {i}\n")
    if res["screens"] is not None:
        w(f"  screens complete {res['complete']} / {len(res['screens'])}\n")
    w(f"  guide hash {', '.join(h[:12] if h else str(h) for h in res['guide_hashes']) or '(none)'}"
      + (f" · current guide {res['current_guide'][:12]}" if res["current_guide"] else "") + "\n\n")
    w(f"  nodes {s['nodes']} · REDUNDANT {s['redundant']}\n")
    for c in lk.CLASSES + ["(unresolved)", "(contested)"]:
        if c in s["classes"]:
            w(f"    {c:<12} {s['classes'][c]:>6}\n")
    w(f"  UNSURE share {s['unsure_share']:.1%}" + ("  (above the 5% freeze threshold, SPEC.md §5)"
                                                     if s["unsure_share"] > 0.05 else "") + "\n")
    w(f"  conf {s['conf']} · ct {s['ct']} · partial {s['partial']} · role errors {s['role_errors']} · "
      f"ECHO with null target {s['echo_null_target']}\n")
    w("\n  patterns\n")
    for k, v in s["patterns"].items():
        w(f"    {k:<16} {v:>6}\n")
    if s["k_rules"]:
        w("  KEEP rules\n")
        for k, v in s["k_rules"].items():
            w(f"    {k:<16} {v:>6}\n")
    for title, key in (("asides", "asides"), ("screen asides", "screen_asides"), ("missed nodes", "missed")):
        if s[key]:
            w(f"  {title}  " + " · ".join(f"{k} {v}" for k, v in sorted(s[key].items())) + "\n")
    w("\n")
    for e in res["errors"][:200]:
        w(f"  ERROR    {e}\n")
    if len(res["errors"]) > 200:
        w(f"  … {len(res['errors']) - 200} more errors\n")
    for e in res["warnings"][:100]:
        w(f"  warning  {e}\n")
    if len(res["warnings"]) > 100:
        w(f"  … {len(res['warnings']) - 100} more warnings\n")
    w(f"\n  {'FAIL' if res['errors'] else 'OK'}: {len(res['errors'])} errors, {len(res['warnings'])} warnings\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file")
    ap.add_argument("--screens", help="screens directory (default: from the .meta.json beside the label file)")
    ap.add_argument("--order", help="screen order file")
    ap.add_argument("--guide", default=str(lk.DEFAULT_GUIDE), help="guide to compare hashes with")
    ap.add_argument("--export", help="write the latest-wins records to this JSONL file")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)
    res = check(Path(args.file), args.screens, args.guide, args.order)
    s = summary(res)
    if args.export:
        n = export(res, args.export)
        res["info"].append(f"exported {n} records to {args.export}")
    if args.json:
        print(json.dumps({k: v for k, v in res.items() if k not in ("latest", "screens")} | {"summary": s},
                         indent=1))
    else:
        print_report(res, s)
    return 1 if res["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
