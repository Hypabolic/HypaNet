"""Tests for the slop labeller: record schema, latest-wins, check.py, guide parsing, and the server API.

    python3 -m unittest discover -s slop/labeller/tests      (or: uv run pytest slop/labeller/tests)

Uses the fixtures in slop/labeller/fixtures/ and temporary label directories; never writes to slop/labels/.
"""

import io
import json
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELLER = HERE.parent
sys.path.insert(0, str(LABELLER))

import check  # noqa: E402
import labelkit as lk  # noqa: E402
import serve  # noqa: E402

FIX = LABELLER / "fixtures"
H1 = "a" * 64
H2 = "b" * 64


def node_rec(screen, loc, cls="KEEP", **kw):
    r = {"screen": screen, "loc": loc, "class": cls, "target": None, "equiv": [], "pattern": None, "aside": [],
         "partial": False, "ct": False, "conf": "clear", "role_ok": True, "labeller": "t", "guide_sha256": H1,
         "time": "2026-11-10T10:00:00Z"}
    r.update(kw)
    return r


def pause():
    return json.loads((FIX / "01_pause_menu.json").read_text())


class TestRecords(unittest.TestCase):
    def setUp(self):
        self.s = pause()
        self.L = [n["loc"] for n in self.s["nodes"]]

    def ok(self, rec):
        errs, _ = lk.validate_record(rec, "t")
        self.assertEqual(errs, [], rec)

    def bad(self, rec, frag):
        errs, _ = lk.validate_record(rec, "t")
        self.assertTrue(any(frag in e for e in errs), (frag, errs))

    def test_valid_kinds(self):
        sid = self.s["id"]
        self.ok(node_rec(sid, self.L[0], "ECHO", target=self.L[1], pattern="R1"))
        self.ok(node_rec(sid, self.L[1], "KEEP", pattern="K11"))
        self.ok(node_rec(sid, self.L[2], "NARRATE", target=self.L[3], pattern="R6", ct=True, conf="judgement"))
        self.ok(node_rec(sid, self.L[3], "DECOR", pattern="R11"))
        self.ok(node_rec(sid, self.L[4], "UNSURE"))
        self.ok(node_rec(sid, self.L[5], None, aside=["UNRESOLVED"]))
        self.ok(node_rec(sid, self.L[6], "KEEP", role_ok=False, role_fix="menu_item", partial=True))
        base = {"screen": sid, "labeller": "t", "guide_sha256": H1, "time": "2026-11-10T10:00:00Z"}
        self.ok(dict(base, aside_screen=[{"code": "R19", "nodes": self.L[3:6]}]))
        self.ok(dict(base, missed=[{"line": 3, "text": "Press Esc", "class": "NARRATE"}]))
        self.ok(dict(base, done=True, nodes=7))

    def test_invalid(self):
        sid = self.s["id"]
        self.bad(node_rec(sid, self.L[0], "ECHO", target=self.L[1], pattern="R11"), "not a ECHO pattern")
        self.bad(node_rec(sid, self.L[0], "DECOR", target=self.L[1], pattern="R2"), "target set on a DECOR")
        self.bad(node_rec(sid, self.L[0], "ECHO", target=self.L[0], pattern="R1"), "target is the node itself")
        self.bad(node_rec(sid, self.L[0], "ECHO", pattern="R1", partial=True), "partial")
        self.bad(node_rec(sid, self.L[0], "KEEP", pattern="R1"), "KEEP pattern")
        self.bad(node_rec(sid, self.L[0], "KEEP", equiv=[self.L[1]]), "equiv set on a KEEP")
        self.bad(node_rec(sid, self.L[0], None), "class is null")
        self.bad(node_rec(sid, self.L[0], "KEEP", role_ok=False), "role_fix")
        self.bad(node_rec(sid, self.L[0], "KEEP", role_fix="h2"), "role_fix given")
        self.bad(node_rec(sid, self.L[0], "KEEP", aside=["R99"]), "unknown aside")
        self.bad(node_rec(sid, self.L[0], "KEEP", conf="sure"), "conf")
        self.bad(node_rec(sid, self.L[0], "KEEP", guide_sha256="x"), "guide_sha256")
        self.bad(node_rec(sid, self.L[0], "KEEP", labeller="other"), "does not match")
        self.bad(node_rec(sid, [0, 1, 1, 2], "KEEP"), "bad loc")
        r = node_rec(sid, self.L[0])
        del r["ct"]
        self.bad(r, "missing fields")

    def test_echo_null_target_warns(self):
        errs, warns = lk.validate_record(node_rec(self.s["id"], self.L[0], "ECHO", pattern="R1"), "t")
        self.assertEqual(errs, [])
        self.assertTrue(warns)

    def test_latest_wins(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.t.jsonl"
            sid = self.s["id"]
            lk.append_record(p, node_rec(sid, self.L[0], "KEEP"))
            lk.append_record(p, node_rec(sid, self.L[0], "ECHO", target=self.L[1], pattern="R1"))
            with open(p, "a") as f:
                f.write("{not json\n")
            lf = lk.LabelFile.load(p)
            self.assertEqual(len(lf.records), 2)
            self.assertEqual(len(lf.latest), 1)
            self.assertEqual(lf.for_screen(sid)["nodes"][lk.loc_key(self.L[0])]["class"], "ECHO")
            self.assertEqual(lf.bad_lines[0][0], 3)


class TestGuide(unittest.TestCase):
    def test_real_guide_parses(self):
        info = lk.parse_guide(lk.DEFAULT_GUIDE.read_text(encoding="utf-8"))
        self.assertEqual(info["problems"], [])
        self.assertEqual(set(info["krules"]), set(lk.K_RULES))
        self.assertEqual({c: v["class"] for c, v in info["patterns"].items()}, lk.PATTERN_CLASS)
        self.assertIn("R19", info["screen_asides"])
        self.assertTrue(set(lk.NODE_ASIDES) <= set(info["asides"]))
        secs = lk.guide_sections(lk.DEFAULT_GUIDE.read_text(encoding="utf-8"))
        self.assertTrue({"3", "4", "5", "6", "7", "8", "9"} <= set(secs))

    def test_fixtures_valid(self):
        screens, problems = lk.load_screens(FIX)
        self.assertEqual(problems, [])
        self.assertGreaterEqual(len(screens), 5)
        texts = [n["text"] for n in screens[0]["screen"]["nodes"]]
        self.assertEqual(texts, ["OPERATION ON HOLD", "PAUSED", "Resume recaptures the mouse.", "RESUME", "SETTINGS",
                                 "CONTROLS", "RESTART MISSION"])
        for s in screens:  # every loc slices its text out of the source (literal text; nodes with a VAR marker excepted)
            src = (FIX / "src" / s["screen"]["source"]["repo"] / s["screen"]["source"]["path"]).read_text().splitlines()
            for n in s["screen"]["nodes"]:
                l1, c1, l2, c2 = n["loc"]
                if "\ue000" not in n["text"]:
                    self.assertEqual(src[l1 - 1][c1 - 1:c2 - 1], n["text"])


class TestCheck(unittest.TestCase):
    """check.py on synthetic label files for the pause fixture alone."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.screens = self.tmp / "screens"
        self.screens.mkdir()
        shutil.copy(FIX / "01_pause_menu.json", self.screens)
        self.s = pause()
        self.L = [n["loc"] for n in self.s["nodes"]]
        self.path = self.tmp / "ref.t.jsonl"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def write_full(self, hashes=(H1,)):
        sid = self.s["id"]
        recs = [node_rec(sid, self.L[0], "ECHO", target=self.L[1], pattern="R1"),
                node_rec(sid, self.L[2], "NARRATE", target=self.L[3], pattern="R6", ct=True)]
        recs += [node_rec(sid, loc) for loc in (self.L[1],) + tuple(self.L[3:])]
        for n, r in enumerate(recs):
            r["guide_sha256"] = hashes[n % len(hashes)]
            lk.append_record(self.path, r)
        lk.append_record(self.path, {"screen": sid, "done": True, "nodes": 7, "labeller": "t",
                                     "guide_sha256": hashes[0], "time": "2026-11-10T10:00:00Z"})

    def run_check(self):
        return check.check(self.path, str(self.screens), None)

    def test_clean(self):
        self.write_full()
        res = self.run_check()
        self.assertEqual(res["errors"], [])
        s = check.summary(res)
        self.assertEqual(s["classes"], {"ECHO": 1, "NARRATE": 1, "KEEP": 5})
        self.assertEqual(s["patterns"], {"ECHO R1": 1, "NARRATE R6": 1})
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(check.main([str(self.path), "--screens", str(self.screens)]), 0)
        self.assertIn("OK: 0 errors", out.getvalue())

    def test_two_guide_hashes(self):
        self.write_full(hashes=(H1, H2))
        res = self.run_check()
        self.assertTrue(any("more than one guide hash" in e for e in res["errors"]))

    def test_superseded_old_hash_only_warns(self):
        sid = self.s["id"]
        old = node_rec(sid, self.L[1], "UNSURE", guide_sha256=H2)
        lk.append_record(self.path, old)
        self.write_full()
        res = self.run_check()
        self.assertEqual(res["errors"], [])
        self.assertTrue(any("superseded" in w for w in res["warnings"]))

    def test_missing_target_and_incomplete(self):
        sid = self.s["id"]
        lk.append_record(self.path, node_rec(sid, self.L[0], "ECHO", target=[40, 1, 40, 5], pattern="R1"))
        res = self.run_check()
        self.assertTrue(any("does not exist" in e for e in res["errors"]))
        self.assertTrue(any("6 of 7 nodes unlabelled" in e for e in res["errors"]))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(check.main([str(self.path), "--screens", str(self.screens)]), 1)

    def test_complete_without_done(self):
        self.write_full()
        lines = self.path.read_text().splitlines()[:-1]
        self.path.write_text("\n".join(lines) + "\n")
        res = self.run_check()
        self.assertTrue(any("no done record" in e for e in res["errors"]))

    def test_schema_error_reported_with_line(self):
        self.write_full()
        bad = node_rec(self.s["id"], self.L[3], "DECOR", pattern="R1")
        lk.append_record(self.path, bad)
        res = self.run_check()
        self.assertTrue(any(e.startswith("line 9:") and "not a DECOR pattern" in e for e in res["errors"]))

    def test_export_latest(self):
        self.write_full()
        lk.append_record(self.path, node_rec(self.s["id"], self.L[4], "UNSURE"))
        res = self.run_check()
        out = self.tmp / "out.jsonl"
        n = check.export(res, out)
        rows = [json.loads(x) for x in out.read_text().splitlines()]
        self.assertEqual(n, 8)
        self.assertEqual([r["class"] for r in rows if r.get("loc") == self.L[4]], ["UNSURE"])


class TestServer(unittest.TestCase):
    """serve.py in-process on an ephemeral port, labels in a temporary directory."""

    def start(self, compare=False):
        self.tmp = Path(tempfile.mkdtemp())
        args = serve.argparse.Namespace(set="fx", screens=str(FIX), labeller="t", port=0,
                                        source_root=str(FIX / "src"), kinds=None, order=None,
                                        labels_dir=str(self.tmp), guide=str(lk.DEFAULT_GUIDE), compare=compare,
                                        open=False, verbose=False)
        self.st = serve.State(args)
        serve.Handler.state = self.st
        self.srv = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        shutil.rmtree(self.tmp)

    def get(self, path):
        with urllib.request.urlopen(self.base + path) as r:
            return json.loads(r.read())

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())

    def label(self, sid, loc, cls="KEEP", **kw):
        r = {"loc": loc, "class": cls, "target": None, "equiv": [], "pattern": None, "aside": [], "partial": False,
             "ct": False, "conf": "clear", "role_ok": True}
        r.update(kw)
        return self.post("/api/label", {"screen": sid, "records": [r]})

    def test_label_screen_done_and_resume(self):
        self.start()
        sess = self.get("/api/session")
        self.assertEqual(sess["resume"], 0)
        scr = self.get("/api/screen?n=0")
        self.assertTrue(scr["source"]["ok"])
        sid = scr["screen"]["id"]
        L = [n["loc"] for n in scr["screen"]["nodes"]]
        r = self.label(sid, L[0], "ECHO", target=L[1], pattern="R1")
        self.assertFalse(r["done_written"])
        self.assertEqual(r["records"][0]["guide_sha256"], lk.sha256_file(lk.DEFAULT_GUIDE))
        for loc in L[1:-1]:
            self.assertFalse(self.label(sid, loc)["done_written"])
        r = self.label(sid, L[-1])
        self.assertTrue(r["done_written"])
        self.assertTrue(r["summary"]["done"])
        self.assertFalse(self.label(sid, L[-1], conf="judgement")["done_written"])  # once only
        self.assertEqual(self.get("/api/session")["resume"], 1)
        lf = lk.LabelFile.load(self.st.path)
        self.assertEqual(sum(1 for _, x in lf.records if "done" in x), 1)
        self.assertTrue(lk.meta_path(self.st.path).exists())

    def test_rejections(self):
        self.start()
        scr = self.get("/api/screen?n=1")  # profile form: node 11 is sr_only
        sid = scr["screen"]["id"]
        L = [n["loc"] for n in scr["screen"]["nodes"]]
        self.assertIn("context only", self.label(sid, L[11])["error"])
        self.assertIn("not a node", self.label(sid, L[2], "ECHO", target=[99, 1, 99, 2], pattern="R3")["error"])
        self.assertIn("role_fix equals", self.label(sid, L[2], role_ok=False, role_fix="subtitle")["error"])
        self.assertFalse(self.st.path.exists())

    def test_screen_records(self):
        self.start()
        scr = self.get("/api/screen?n=2")
        sid = scr["screen"]["id"]
        L = [n["loc"] for n in scr["screen"]["nodes"]]
        r = self.post("/api/screen_record", {"screen": sid, "kind": "aside_screen",
                                             "value": [{"code": "R19", "nodes": [L[8], L[12], L[16]]}]})
        self.assertTrue(r["ok"])
        r = self.post("/api/screen_record", {"screen": sid, "kind": "missed",
                                             "value": [{"line": 3, "text": "x", "class": "DECOR"}]})
        self.assertTrue(r["ok"])
        r = self.post("/api/screen_record", {"screen": sid, "kind": "missed", "value": [{"line": 0, "text": "x",
                                                                                         "class": "DECOR"}]})
        self.assertIn("line", r["error"])
        scr = self.get("/api/screen?n=2")
        self.assertEqual(scr["aside_screen"][0]["code"], "R19")
        self.assertEqual(len(scr["missed"]), 1)

    def test_blind_never_reads_other_labellers(self):
        self.start()
        other = self.tmp / "fx.other.jsonl"
        sid = self.st.screens[0]["screen"]["id"]
        loc = self.st.screens[0]["screen"]["nodes"][0]["loc"]
        lk.append_record(other, node_rec(sid, loc, "DECOR", labeller="other", pattern="R2"))
        opened = []
        orig = lk.LabelFile.load
        lk.LabelFile.load = classmethod(lambda cls, p: (opened.append(Path(p)), orig.__func__(cls, p))[1])
        try:
            scr = self.get("/api/screen?n=0")
        finally:
            lk.LabelFile.load = orig
        self.assertNotIn("others", scr)
        self.assertNotIn(other, opened)
        self.assertNotIn("DECOR", json.dumps(scr["labels"]))

    def test_compare_mode_shows_others(self):
        self.start(compare=True)
        sid = self.st.screens[0]["screen"]["id"]
        loc = self.st.screens[0]["screen"]["nodes"][0]["loc"]
        lk.append_record(self.tmp / "fx.other.jsonl", node_rec(sid, loc, "DECOR", labeller="other", pattern="R2"))
        scr = self.get("/api/screen?n=0")
        self.assertEqual(scr["others"]["other"][lk.loc_key(loc)]["class"], "DECOR")


if __name__ == "__main__":
    unittest.main()
