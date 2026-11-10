#!/usr/bin/env python3
"""Local labelling server for the HypaNet slop experiment (SPEC.md §5, LABELLING.md).

    uv run python slop/labeller/serve.py --set NAME --screens DIR --labeller mg [--port 8765]
        [--source-root DIR] [--kinds FILE] [--order FILE] [--labels-dir DIR] [--guide FILE] [--compare] [--open]

Writes slop/labels/<set>.<labeller>.jsonl, one record per decision (append-only, latest record per key wins), and
resumes at the first screen without a done record. Standard library only; binds 127.0.0.1.

Blind by default: the server never opens another labeller's file for the same set. `--compare` (adjudication only,
never for a blind second labeller) shows other labellers' classes beside each node.

--source-root: a node's source file is looked up at ROOT/<source.repo>/<source.path>, then ROOT/<source.path>.
--kinds: JSON {key: "one-line product kind"}; key is a screen id, "<repo>/<path>", or a repo name (LABELLING.md §2a).
         Defaults to <screens>/kinds.json when present.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labelkit as lk  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
MAX_SOURCE_BYTES = 4 << 20


class State:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.set_name = args.set
        self.labeller = args.labeller
        self.blind = not args.compare
        self.labels_dir = Path(args.labels_dir).resolve()
        self.path = lk.label_path(self.labels_dir, self.set_name, self.labeller)
        self.guide_path = Path(args.guide).resolve()
        self.screens_dir = Path(args.screens).resolve()
        self.source_root = Path(args.source_root).resolve() if args.source_root else None
        self.screens, self.screen_problems = lk.load_screens(self.screens_dir, args.order)
        if not self.screens:
            raise SystemExit(f"no slop.screen/1 screens found in {self.screens_dir}")
        self.by_id = {s["screen"]["id"]: n for n, s in enumerate(self.screens)}
        self.labels = lk.LabelFile.load(self.path)
        self.guide_text = self.guide_path.read_text(encoding="utf-8")
        self.guide_baseline = lk.sha256_file(self.guide_path)
        self.guide_info = lk.parse_guide(self.guide_text)
        self.guide_acks = []
        kinds_file = Path(args.kinds) if args.kinds else self.screens_dir / "kinds.json"
        self.kinds = json.loads(kinds_file.read_text()) if kinds_file.exists() else {}
        self.source_cache = {}
        self.write_meta()

    # ------------------------------------------------------------ helpers

    def write_meta(self):
        meta = {
            "set": self.set_name, "labeller": self.labeller, "blind": self.blind,
            "screens_dir": str(self.screens_dir), "order": str(Path(self.args.order).resolve()) if self.args.order
            else None, "source_root": str(self.source_root) if self.source_root else None,
            "guide": str(self.guide_path), "guide_sha256_at_start": self.guide_baseline, "started": lk.now_iso(),
            "screens": [{"id": s["screen"]["id"], "file": s["file"], "sha256": s["sha256"]} for s in self.screens],
        }
        mp = lk.meta_path(self.path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(meta, indent=1) + "\n")

    def guide_status(self):
        try:
            cur = lk.sha256_file(self.guide_path)
        except OSError:
            cur = None
        hashes = {}
        for _, rec in self.labels.records:
            h = rec.get("guide_sha256")
            hashes[h] = hashes.get(h, 0) + 1
        return {"path": str(self.guide_path), "baseline": self.guide_baseline, "current": cur,
                "changed": cur != self.guide_baseline, "file_hashes": hashes,
                "older_records": sum(n for h, n in hashes.items() if h != cur)}

    def kind_for(self, screen):
        src = screen.get("source") or {}
        for key in (screen.get("id"), f"{src.get('repo')}/{src.get('path')}", src.get("repo")):
            if key in self.kinds:
                return self.kinds[key]
        return None

    def read_source(self, screen):
        src = screen.get("source") or {}
        key = (src.get("repo"), src.get("path"))
        if key in self.source_cache:
            return self.source_cache[key]
        res = {"ok": False, "lines": [], "path": src.get("path"), "error": None}
        if not self.source_root:
            res["error"] = "no --source-root given"
        else:
            cands = []
            if src.get("repo"):
                cands.append(self.source_root / src["repo"] / src.get("path", ""))
            cands.append(self.source_root / src.get("path", ""))
            for c in cands:
                c = c.resolve()
                if self.source_root not in c.parents:
                    continue
                if c.is_file():
                    if c.stat().st_size > MAX_SOURCE_BYTES:
                        res["error"] = f"{c.name} is larger than {MAX_SOURCE_BYTES >> 20} MB"
                        break
                    res.update(ok=True, lines=c.read_text(encoding="utf-8", errors="replace").splitlines(),
                               resolved=str(c))
                    break
            else:
                res["error"] = "source file not found under --source-root"
        self.source_cache[key] = res
        return res

    def screen_summary(self, n):
        s = self.screens[n]["screen"]
        lab = lk.labelable_nodes(s)
        mine = self.labels.for_screen(s["id"])
        labelled = sum(1 for nd in lab if lk.node_complete(mine["nodes"].get(lk.loc_key(nd["loc"]))))
        return {"n": n, "id": s["id"], "file": self.screens[n]["file"], "path": (s.get("source") or {}).get("path"),
                "repo": (s.get("source") or {}).get("repo"), "nodes": len(lab), "labelled": labelled,
                "done": mine["done"] is not None}

    def resume_index(self):
        for n in range(len(self.screens)):
            if not self.labels.for_screen(self.screens[n]["screen"]["id"])["done"]:
                return n
        return 0

    def others(self, screen_id):
        """Other labellers' latest labels for this screen. Never called in blind mode."""
        assert not self.blind, "blind mode must not read other labellers' files"
        out = {}
        for p in sorted(self.labels_dir.glob(f"{self.set_name}.*.jsonl")):
            s, lab = lk.parse_label_filename(p)
            if s != self.set_name or lab == self.labeller:
                continue
            nodes = lk.LabelFile.load(p).for_screen(screen_id)["nodes"]
            out[lab] = {k: {"class": r.get("class"), "target": r.get("target"), "pattern": r.get("pattern")}
                        for k, r in nodes.items()}
        return out

    def stamp(self, rec):
        rec["labeller"] = self.labeller
        rec["guide_sha256"] = lk.sha256_file(self.guide_path)
        rec["time"] = lk.now_iso()
        return rec

    def write(self, rec):
        errs, warns = lk.validate_record(rec, self.labeller)
        if errs:
            raise ValueError("; ".join(errs))
        lk.append_record(self.path, rec)
        self.labels.add(rec)
        return warns


NODE_INPUT = ["loc", "class", "target", "equiv", "pattern", "aside", "partial", "ct", "conf", "role_ok"]


class Handler(BaseHTTPRequestHandler):
    server_version = "slop-labeller/1"
    state: State = None  # set in main

    def log_message(self, fmt, *args):  # quiet
        if self.state.args.verbose:
            super().log_message(fmt, *args)

    # ------------------------------------------------------------ plumbing

    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path):
        try:
            data = path.read_bytes()
        except OSError:
            return self.send_error(HTTPStatus.NOT_FOUND)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def screen_index(self, sid):
        st = self.state
        if sid not in st.by_id:
            raise ValueError(f"unknown screen {sid!r}")
        return st.by_id[sid]

    # ------------------------------------------------------------ routes

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        st = self.state
        try:
            if u.path in ("/", "/index.html"):
                return self.send_file(STATIC / "index.html")
            if u.path.startswith("/static/"):
                p = (STATIC / u.path[len("/static/"):]).resolve()
                if STATIC not in p.parents:
                    return self.send_error(HTTPStatus.FORBIDDEN)
                return self.send_file(p)
            if u.path == "/api/session":
                with st.lock:
                    return self.send_json({
                        "set": st.set_name, "labeller": st.labeller, "blind": st.blind,
                        "labels_file": str(st.path), "screens_dir": str(st.screens_dir),
                        "source_root": str(st.source_root) if st.source_root else None,
                        "screens": [st.screen_summary(n) for n in range(len(st.screens))],
                        "resume": st.resume_index(), "screen_problems": st.screen_problems,
                        "guide": st.guide_status(), "guide_problems": st.guide_info["problems"],
                        "bad_lines": st.labels.bad_lines,
                        "schema": {"classes": lk.CLASSES, "patterns": lk.PATTERNS, "k_rules": lk.K_RULES,
                                   "node_asides": lk.NODE_ASIDES, "screen_asides": lk.SCREEN_ASIDES,
                                   "roles": lk.ROLES, "flags": lk.FLAGS,
                                   "context_roles": sorted(lk.CONTEXT_ROLES),
                                   "context_flags": sorted(lk.CONTEXT_FLAGS)},
                    })
            if u.path == "/api/screen":
                n = int(q.get("n", ["0"])[0])
                with st.lock:
                    if not 0 <= n < len(st.screens):
                        return self.send_json({"error": "no such screen"}, 404)
                    entry = st.screens[n]
                    s = entry["screen"]
                    mine = st.labels.for_screen(s["id"])
                    out = {"n": n, "screen": s, "file": entry["file"], "kind": st.kind_for(s),
                           "source": st.read_source(s), "labels": mine["nodes"],
                           "aside_screen": (mine["aside_screen"] or {}).get("aside_screen", []),
                           "missed": (mine["missed"] or {}).get("missed", []), "done": mine["done"] is not None,
                           "summary": st.screen_summary(n)}
                    if not st.blind:
                        out["others"] = st.others(s["id"])
                    return self.send_json(out)
            if u.path == "/api/guide":
                with st.lock:
                    return self.send_json(st.guide_status())
            if u.path == "/api/reference":
                with st.lock:
                    text = st.guide_path.read_text(encoding="utf-8")
                    info = lk.parse_guide(text)
                    return self.send_json({"sections": lk.guide_sections(text), "patterns": info["patterns"],
                                           "asides": info["asides"], "screen_asides": info["screen_asides"],
                                           "krules": info["krules"]})
            return self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError) as e:
            return self.send_json({"error": str(e)}, 400)

    def do_POST(self):
        u = urlparse(self.path)
        st = self.state
        try:
            body = self.body()
            with st.lock:
                if u.path == "/api/label":
                    return self.send_json(self.post_labels(body))
                if u.path == "/api/screen_record":
                    return self.send_json(self.post_screen_record(body))
                if u.path == "/api/done":
                    n = self.screen_index(body.get("screen"))
                    return self.send_json(self.after_write(n, write_done=True))
                if u.path == "/api/guide/ack":
                    g = st.guide_status()
                    st.guide_acks.append({"from": st.guide_baseline, "to": g["current"], "time": lk.now_iso()})
                    st.guide_baseline = g["current"]
                    st.guide_info = lk.parse_guide(st.guide_path.read_text(encoding="utf-8"))
                    return self.send_json(st.guide_status())
            return self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            return self.send_json({"error": str(e)}, 400)

    def post_labels(self, body):
        st = self.state
        sid = body.get("screen")
        n = self.screen_index(sid)
        screen = st.screens[n]["screen"]
        locs = {lk.loc_key(nd["loc"]): nd for nd in screen["nodes"]}
        items = body.get("records")
        if not isinstance(items, list) or not items:
            raise ValueError("records must be a non-empty list")
        recs, warns = [], []
        for it in items:
            rec = {"screen": sid}
            for f in NODE_INPUT:
                if f not in it:
                    raise ValueError(f"missing {f}")
                rec[f] = it[f]
            if it.get("role_ok") is False:
                rec["role_fix"] = it.get("role_fix")
            node = locs.get(lk.loc_key(rec["loc"])) if lk.is_loc(rec["loc"]) else None
            if node is None:
                raise ValueError(f"loc {rec['loc']} is not a node of this screen")
            if lk.node_is_context(node):
                raise ValueError(f"node {node['i']} is context only (LABELLING.md §2) and is not labelled")
            for name, ref in [("target", rec["target"])] + [("equiv", x) for x in rec.get("equiv") or []]:
                if ref is not None and (not lk.is_loc(ref) or lk.loc_key(ref) not in locs):
                    raise ValueError(f"{name} {ref} is not a node of this screen")
            if rec["role_ok"] is False and rec.get("role_fix") == node.get("role"):
                raise ValueError("role_fix equals the extractor's role")
            st.stamp(rec)
            errs, w = lk.validate_record(rec, st.labeller)
            if errs:
                raise ValueError(f"node {node['i']}: " + "; ".join(errs))
            recs.append(rec)
            warns += [f"node {node['i']}: {x}" for x in w]
        for rec in recs:
            st.write(rec)
        out = self.after_write(n)
        out["records"] = recs
        out["warnings"] = warns
        return out

    def post_screen_record(self, body):
        st = self.state
        sid = body.get("screen")
        n = self.screen_index(sid)
        kind = body.get("kind")
        if kind not in ("aside_screen", "missed"):
            raise ValueError("kind must be aside_screen or missed")
        value = body.get("value")
        if kind == "aside_screen":
            locs = {lk.loc_key(nd["loc"]) for nd in st.screens[n]["screen"]["nodes"]}
            for it in value or []:
                for x in it.get("nodes", []):
                    if not lk.is_loc(x) or lk.loc_key(x) not in locs:
                        raise ValueError(f"aside node {x} is not a node of this screen")
        rec = st.stamp({"screen": sid, kind: value})
        warns = st.write(rec)
        out = self.after_write(n)
        out["record"] = rec
        out["warnings"] = warns
        return out

    def after_write(self, n, write_done=False):
        """Write the screen's done record once every labelable node has a label."""
        st = self.state
        summ = st.screen_summary(n)
        complete = summ["labelled"] == summ["nodes"]
        done_written = False
        if complete and not summ["done"]:
            st.write(st.stamp({"screen": summ["id"], "done": True, "nodes": summ["labelled"]}))
            done_written = True
            summ = st.screen_summary(n)
        elif write_done and not complete:
            raise ValueError(f"screen not complete: {summ['labelled']} of {summ['nodes']} nodes labelled")
        return {"ok": True, "summary": summ, "complete": complete, "done_written": done_written,
                "guide": st.guide_status()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", required=True, help="set name, e.g. reference_real, calibration, sealed_real")
    ap.add_argument("--screens", required=True, help="directory of screen JSON files (slop.screen/1)")
    ap.add_argument("--labeller", required=True, help="labeller id, e.g. mg")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--source-root", help="root for source files (ROOT/<repo>/<path>, then ROOT/<path>)")
    ap.add_argument("--kinds", help="JSON map of product kinds (default <screens>/kinds.json)")
    ap.add_argument("--order", help="file of screen ids, one per line, giving the labelling order")
    ap.add_argument("--labels-dir", default=str(lk.DEFAULT_LABELS_DIR), help="default slop/labels")
    ap.add_argument("--guide", default=str(lk.DEFAULT_GUIDE), help="default slop/LABELLING.md")
    ap.add_argument("--compare", action="store_true",
                    help="not blind: show other labellers' labels for this set (adjudication only)")
    ap.add_argument("--open", action="store_true", help="open the browser")
    ap.add_argument("--verbose", action="store_true", help="log every request")
    args = ap.parse_args(argv)
    for name, v in (("--set", args.set), ("--labeller", args.labeller)):
        if not lk.valid_name(v):
            ap.error(f"{name} must be letters, digits, '_' or '-'")
    st = State(args)
    Handler.state = st
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    done = sum(1 for n in range(len(st.screens)) if st.screen_summary(n)["done"])
    print(f"slop labeller · set {st.set_name} · labeller {st.labeller} · {'blind' if st.blind else 'COMPARE (not blind)'}")
    print(f"  screens  {len(st.screens)} from {st.screens_dir} ({done} done)")
    print(f"  labels   {st.path} ({len(st.labels.records)} records)")
    print(f"  guide    {st.guide_path} sha256 {st.guide_baseline[:12]}")
    for p in st.screen_problems + st.guide_info["problems"]:
        print(f"  warning: {p}")
    for ln, e in st.labels.bad_lines:
        print(f"  warning: label file line {ln}: {e}")
    print(f"  open     {url}")
    sys.stdout.flush()
    if args.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
