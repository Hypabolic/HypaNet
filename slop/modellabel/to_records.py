"""Model reply -> SPEC.md §5 label records, and the label files built from the raw replies.

    uv run python modellabel/to_records.py --set reference_real --labeller opus|codex [--check]

A reply (schema.json) names nodes by their index i; label records key on loc (SPEC.md §5). `validate_reply` checks a
reply against the schema, the screen and `labelkit.validate_record`; its errors are what a retry is told. The label
file `labels/<set>.<labeller>.jsonl` is rebuilt from `labels/raw/<set>/<labeller>/*.json` every time (the raw files
are the source of truth), so a resumed or repeated run never leaves duplicate records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modellabel import common as c  # noqa: E402

lk = c.lk
_DUMMY_SHA = "0" * 64
_DUMMY_TIME = "2026-01-01T00:00:00Z"


def normalise_reply(obj: dict, scr: c.Screen):
    """Check a parsed reply and return (labels_by_i, errors, warnings).

    labels_by_i maps every labelable node index to its model label (fields of schema.json). Nodes with no literal
    words get class null and aside UNRESOLVED whatever the reply said (LABELLING.md §2), with a warning when the
    reply gave them a class."""
    errors = c.check_schema(obj, c.load_schema())
    warnings = []
    if errors:
        return {}, errors, warnings
    if obj.get("screen") != scr.id:
        errors.append(f"screen is {obj.get('screen')!r}, expected {scr.id!r}")
    nodes = scr.by_i()
    labelable = {n["i"] for n in scr.labelable()}
    out = {}
    for ent in obj["nodes"]:
        i = ent["i"]
        if i not in nodes:
            errors.append(f"node {i}: no such node on this screen")
            continue
        if i not in labelable:
            warnings.append(f"node {i}: context only; its label was dropped")
            continue
        if i in out:
            errors.append(f"node {i}: labelled more than once")
            continue
        lab = dict(ent)
        node = nodes[i]
        if not c.has_literal_words(node):
            if lab["class"] is not None:
                warnings.append(f"node {i}: no literal words, so class {lab['class']} was replaced by null")
            lab.update({"class": None, "target": None, "pattern": None, "equiv": [], "partial": False, "ct": False,
                        "conf": "clear"})
            if "UNRESOLVED" not in lab["aside"]:
                lab["aside"] = list(lab["aside"]) + ["UNRESOLVED"]
        elif lab["class"] is None:
            errors.append(f"node {i}: class is null, but the node has literal words (null is only for nodes marked "
                          "no_literal_words)")
        win = scr.windows()
        for name, ref in [("target", lab.get("target"))] + [("equiv", x) for x in lab.get("equiv") or []]:
            if ref is not None and ref not in nodes:
                errors.append(f"node {i}: {name} {ref} is not a node on this screen")
            elif ref == i:
                errors.append(f"node {i}: {name} is the node itself")
            elif ref is not None and win.get(ref) != win.get(i):
                errors.append(f"node {i}: {name} {ref} is in window {win.get(ref)}, not in this node's window "
                              f"{win.get(i)} (LABELLING.md §2a: a target must be in the same window; copy repeated "
                              "only in another window is labelled as if that window did not exist, aside X-FILE)")
        out[i] = lab
    replied = {e["i"] for e in obj["nodes"]}
    missing = [i for i in sorted(labelable) if i not in out and i not in replied]
    if missing:
        errors.append(f"nodes not labelled: {', '.join(map(str, missing))} (every node that is not context only "
                      "needs an entry)")
    for n, it in enumerate(obj["aside_screen"]):
        bad = [x for x in it["nodes"] if x not in nodes]
        if bad:
            errors.append(f"aside_screen[{n}] {it['code']}: nodes {bad} are not on this screen")
    # record-level rules (labelkit), with placeholder metadata
    if not errors:
        for rec in _records(out, obj, scr, "x", _DUMMY_SHA, _DUMMY_TIME):
            errs, _ = lk.validate_record(rec, "x")
            where = f"node {_i_of(rec, scr)}" if "loc" in rec else lk.record_kind(rec)
            errors += [f"{where}: {e}" for e in errs]
    return out, errors, warnings


def soft_issues(labels: dict, scr: c.Screen) -> list:
    """Valid labels that look inconsistent with the guide; the labeller is asked once to check them (runners
    run_with_retry), and a second reply is accepted as it is. The same checks are warnings in labeller/check.py:
    an ECHO pointing at a node of lower rank (LABELLING.md §5: the lower-ranked copy is the one flagged; ranks use
    role_fix where the labeller corrected the role) and an equiv entry that is not itself labelled ECHO."""
    nodes = scr.by_i()
    out = []

    def rank(i):
        lab, n = labels.get(i) or {}, nodes[i]
        role = lab.get("role_fix") if lab.get("role_ok") is False and lab.get("role_fix") else n.get("role")
        return lk.role_rank(role, n.get("flags"))

    for i, lab in sorted(labels.items()):
        t = lab.get("target")
        if lab.get("class") == "ECHO" and t is not None and t in nodes and t in labels and rank(i) < rank(t):
            out.append(f"node {i} is ECHO of node {t}, but node {i} ranks higher (§5 precedence: the lower-ranked "
                       f"copy is the one labelled ECHO, pointing at the higher)")
        for x in lab.get("equiv") or []:
            if x in labels and labels[x].get("class") != "ECHO":
                out.append(f"node {i}: equiv {x} is labelled {labels[x].get('class')}, not ECHO (§5: equiv lists "
                           "the other flagged copies)")
    return out


def validate_reply(obj: dict, scr: c.Screen):
    """(errors, warnings, soft) for run_with_retry."""
    labels, errors, warnings = normalise_reply(obj, scr)
    return errors, warnings, ([] if errors else soft_issues(labels, scr))


def _i_of(rec, scr):
    n = scr.by_loc().get(lk.loc_key(rec["loc"]))
    return n["i"] if n else rec["loc"]


def node_record(lab: dict, node: dict, nodes: dict, sid: str, labeller: str, guide_sha: str, time: str) -> dict:
    loc_of = lambda i: nodes[i]["loc"] if i is not None and i in nodes else None  # noqa: E731
    rec = {"screen": sid, "loc": node["loc"], "class": lab["class"], "target": loc_of(lab.get("target")),
           "equiv": [loc_of(x) for x in lab.get("equiv") or [] if x in nodes], "pattern": lab.get("pattern"),
           "aside": list(dict.fromkeys(lab.get("aside") or [])), "partial": bool(lab.get("partial")),
           "ct": bool(lab.get("ct")), "conf": lab.get("conf"), "role_ok": bool(lab.get("role_ok", True))}
    if rec["role_ok"] is False:
        rec["role_fix"] = lab.get("role_fix")
    rec.update(labeller=labeller, guide_sha256=guide_sha, time=time)
    return rec


def _records(labels: dict, obj: dict, scr: c.Screen, labeller: str, guide_sha: str, time: str) -> list:
    nodes = scr.by_i()
    meta = {"labeller": labeller, "guide_sha256": guide_sha, "time": time}
    recs = [node_record(labels[i], nodes[i], nodes, scr.id, labeller, guide_sha, time) for i in sorted(labels)]
    asides = [{"code": it["code"], "nodes": [nodes[x]["loc"] for x in it["nodes"] if x in nodes]}
              for it in obj.get("aside_screen") or []]
    if asides:
        recs.append({"screen": scr.id, "aside_screen": asides, **meta})
    missed = [{"line": it["line"], "text": it["text"], "class": it["class"]} for it in obj.get("missed") or []]
    if missed:
        recs.append({"screen": scr.id, "missed": missed, **meta})
    recs.append({"screen": scr.id, "done": True, "nodes": len(labels), **meta})
    return recs


def to_records(obj: dict, scr: c.Screen, labeller: str, guide_sha: str, time: str) -> list:
    """Label records for one valid reply: node records in node order, then aside_screen and missed (when any), then
    the done record. Raises ValueError when the reply does not validate."""
    labels, errors, _ = normalise_reply(obj, scr)
    if errors:
        raise ValueError("; ".join(errors))
    return _records(labels, obj, scr, labeller, guide_sha, time)


# ---------------------------------------------------------------- raw files and label files


def raw_dir(labels_dir: Path, set_name: str, labeller: str, phase: str = "label") -> Path:
    base = Path(labels_dir) / "raw" / set_name
    return base / labeller if phase == "label" else base / phase / labeller


def raw_path(labels_dir: Path, set_name: str, labeller: str, sid: str, phase: str = "label") -> Path:
    return raw_dir(labels_dir, set_name, labeller, phase) / f"{sid}.json"


def write_meta(labels_dir: Path, ls: c.LabelSet, labeller: str, guide: Path) -> Path:
    path = lk.label_path(labels_dir, ls.name, labeller)
    meta = {"set": ls.name, "labeller": labeller, "blind": True, "source": "slop/modellabel",
            "screens_dir": str(ls.screens_dir.resolve()),
            "order": str(ls.order_file.resolve()) if ls.order_file else None,
            "guide": str(Path(guide).resolve()), "written": lk.now_iso(),
            "screens": [{"id": s.id, "file": s.file, "sha256": s.sha256} for s in ls.screens]}
    mp = lk.meta_path(path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(meta, indent=1) + "\n")
    return mp


def build_label_file(ls: c.LabelSet, labeller: str, labels_dir: Path = c.LABELS_DIR, guide: Path = c.GUIDE) -> dict:
    """Rewrite labels/<set>.<labeller>.jsonl from the raw replies whose status is ok, in set order."""
    path = lk.label_path(labels_dir, ls.name, labeller)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "missing": 0, "other": {}, "stale": 0}
    lines = []
    for scr in ls.screens:
        raw = c.read_json(raw_path(labels_dir, ls.name, labeller, scr.id))
        if raw is None:
            counts["missing"] += 1
            continue
        if raw.get("status") != "ok":
            counts["other"][raw.get("status")] = counts["other"].get(raw.get("status"), 0) + 1
            continue
        if raw.get("screen_sha256") != scr.sha256:
            counts["stale"] += 1
            continue
        for rec in to_records(raw["reply"], scr, labeller, raw["guide_sha256"], raw["time"]):
            lines.append(json.dumps(rec, ensure_ascii=False, separators=(", ", ": ")))
        counts["ok"] += 1
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(x + "\n" for x in lines), encoding="utf-8")
    tmp.replace(path)
    write_meta(labels_dir, ls, labeller, guide)
    counts["file"] = str(path)
    return counts


def check_label_file(path: Path, ls: c.LabelSet, guide: Path = c.GUIDE) -> dict:
    """Run labeller/check.py on a label file against the set's screens."""
    sys.path.insert(0, str(c.SLOP_DIR / "labeller"))
    import check as chk  # noqa: E402

    res = chk.check(Path(path), str(ls.screens_dir), str(guide), str(ls.order_file) if ls.order_file else None)
    return {"errors": res["errors"], "warnings": res["warnings"], "complete": res["complete"],
            "screens": len(res["screens"] or {}), "summary": chk.summary(res)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", required=True)
    ap.add_argument("--labeller", required=True, choices=sorted(c.LABELLERS))
    ap.add_argument("--labels-dir", default=str(c.LABELS_DIR))
    ap.add_argument("--datasets-dir", default=str(c.DATASETS_DIR))
    ap.add_argument("--guide", default=str(c.GUIDE))
    ap.add_argument("--check", action="store_true", help="run labeller/check.py on the result")
    a = ap.parse_args(argv)
    ls = c.load_set(a.set, Path(a.datasets_dir))
    lab = c.LABELLERS[a.labeller]
    counts = build_label_file(ls, lab, Path(a.labels_dir), Path(a.guide))
    print(json.dumps(counts))
    if a.check:
        res = check_label_file(Path(counts["file"]), ls, Path(a.guide))
        print(f"check.py: {len(res['errors'])} errors, {len(res['warnings'])} warnings, "
              f"{res['complete']} / {res['screens']} screens complete")
        return 1 if res["errors"] else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
