"""Tests for the two-model labelling harness. No model is ever called: stub `claude` and `codex` executables
(tests/stubs/) stand in for the CLIs, and PATH holds only the stubs and the system directories.

    uv run pytest modellabel/tests

Screens and sources are the labeller's invented fixtures (slop/labeller/fixtures/); nothing reads slop/labels/ or
the real sets.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SLOP = HERE.parent.parent
STUBS = HERE / "stubs"
FIX = SLOP / "labeller" / "fixtures"
sys.path.insert(0, str(SLOP))
sys.path.insert(0, str(SLOP / "labeller"))

import check as lcheck  # noqa: E402
from modellabel import adjudicate as adj  # noqa: E402
from modellabel import agree  # noqa: E402
from modellabel import common as c  # noqa: E402
from modellabel import estimate as est  # noqa: E402
from modellabel import prompt as pr  # noqa: E402
from modellabel import run  # noqa: E402
from modellabel import runners as rn  # noqa: E402
from modellabel import to_records as tr  # noqa: E402

GUIDE = SLOP / "LABELLING.md"
N_SCREENS = 6


def make_set(datasets: Path, name: str, ids=None) -> Path:
    root = datasets / name
    (root / "screens").mkdir(parents=True)
    shutil.copytree(FIX / "src", root / "sources")
    order = []
    for f in sorted(FIX.glob("0*.json")):
        s = json.loads(f.read_text())
        if ids and s["id"] not in ids:
            continue
        shutil.copy(f, root / "screens" / f.name)
        order.append(s["id"])
    (root / "order.txt").write_text("\n".join(order) + "\n")
    (root / "manifest.json").write_text(json.dumps({"set": name}))
    shutil.copy(FIX / "kinds.json", datasets / "kinds.json")
    return root


@pytest.fixture
def env(tmp_path, monkeypatch):
    datasets = tmp_path / "datasets"
    make_set(datasets, "fx")
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("PATH", f"{STUBS}:/usr/bin:/bin")
    monkeypatch.setenv("STUB_LOG", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("STUB_STATE", str(state))
    monkeypatch.setenv("CLAUDE_CODE_EFFORT_LEVEL", "low")  # must be scrubbed before claude starts
    for k in ("STUB_SEQ_CLAUDE", "STUB_SEQ_CODEX", "STUB_FLAVOUR_CLAUDE", "STUB_FLAVOUR_CODEX", "STUB_ADJ",
              "STUB_COST"):
        monkeypatch.delenv(k, raising=False)
    assert shutil.which("claude") == str(STUBS / "claude"), "the real claude must never be reachable in tests"
    assert shutil.which("codex") == str(STUBS / "codex")

    class E:
        root = tmp_path
        labels = tmp_path / "labels"
        log = tmp_path / "calls.jsonl"

        def args(self, cmd, *extra, labeller="both"):
            a = [cmd, "--set", "fx", "--datasets-dir", str(datasets), "--labels-dir", str(self.labels),
                 "--guide", str(GUIDE)]
            if cmd in ("label", "adjudicate"):
                a += ["--labeller", labeller, "--claude-bin", str(STUBS / "claude"), "--codex-bin",
                      str(STUBS / "codex"), "--cwd-root", str(tmp_path / "cwd"), "--workers", "2"]
            return a + list(extra)

        def calls(self, name=None):
            if not self.log.exists():
                return []
            rows = [json.loads(x) for x in self.log.read_text().splitlines()]
            return [r for r in rows if name is None or r["name"] == name]

        def raw(self, labeller, sid, phase="label"):
            return c.read_json(tr.raw_path(self.labels, "fx", labeller, sid, phase))

        def ls(self):
            return c.load_set("fx", datasets)

    return E()


def first_id():
    return json.loads(sorted(FIX.glob("0*.json"))[0].read_text())["id"]


# ---------------------------------------------------------------- happy path and isolation


def test_happy_path_both_labellers(env):
    rc = run.main(env.args("label", "--max-usd", "5"))
    assert rc == 0
    for lab in ("opus55", "codex61"):
        path = env.labels / f"fx.{lab}.jsonl"
        res = lcheck.check(path, None, str(GUIDE))  # screens dir from the meta file beside it
        assert res["errors"] == [], res["errors"][:3]
        assert res["complete"] == N_SCREENS
        assert set(res["guide_hashes"]) == {c.lk.sha256_file(GUIDE)}
    raw = env.raw("opus55", first_id())
    assert raw["status"] == "ok" and raw["cost_usd"] == pytest.approx(0.10)
    assert raw["runner"]["model"] == "claude-opus-5-5" and raw["runner"]["effort"] == "high"
    assert rn.ledger_total(env.labels / "raw" / "spend.jsonl") == pytest.approx(0.60)
    assert len(env.calls("claude")) == N_SCREENS and len(env.calls("codex")) == N_SCREENS


def test_claude_isolation_flags_and_env(env):
    run.main(env.args("label", "--max-usd", "5", labeller="opus"))
    call = env.calls("claude")[0]
    argv = call["argv"]
    assert argv[argv.index("--model") + 1] == "claude-opus-5-5"
    assert argv[argv.index("--effort") + 1] == "high"
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--output-format") + 1] == "json"
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence", "--safe-mode"):
        assert flag in argv
    assert argv[argv.index("--setting-sources") + 1] == "project"
    assert float(argv[argv.index("--max-budget-usd") + 1]) == 2.0
    system = argv[argv.index("--system-prompt") + 1]
    assert "## 4. Decision procedure" in system  # the whole guide is in the cached prefix
    assert "<screen id=" in call["stdin"] and "## 4. Decision procedure" not in call["stdin"]
    assert call["env"]["CLAUDE_CODE_EFFORT_LEVEL"] is None  # scrubbed: it outranks --effort
    assert call["env"]["CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK"] == "1"
    assert call["env"]["CLAUDECODE"] is None
    assert Path(call["cwd"]).resolve() == (env.root / "cwd" / "opus55").resolve()
    # every screen gets the same prefix (so it caches)
    assert len({c_["argv"][c_["argv"].index("--system-prompt") + 1] for c_ in env.calls("claude")}) == 1


def test_codex_flags(env):
    run.main(env.args("label", labeller="codex"))
    argv = env.calls("codex")[0]["argv"]
    assert argv[0] == "exec" and argv[-1] == "-"
    assert argv[argv.index("-m") + 1] == "gpt-6.1-sol"
    assert 'model_reasoning_effort="high"' in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    for flag in ("--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "--json"):
        assert flag in argv
    assert Path(argv[argv.index("--output-schema") + 1]).name == "schema.json"
    assert "--output-last-message" in argv
    assert Path(argv[argv.index("-C") + 1]) == env.root / "cwd" / "codex61"
    raw = env.raw("codex61", first_id())
    assert raw["status"] == "ok" and raw["attempts"][0]["usage"]["output_tokens"] == 800


def test_working_directory_must_be_empty(env):
    d = env.root / "cwd" / "opus55"
    d.mkdir(parents=True)
    (d / "CLAUDE.md").write_text("stray")
    with pytest.raises(RuntimeError, match="not empty"):
        run.main(env.args("label", "--max-usd", "5", labeller="opus"))
    assert env.calls() == []


def test_opus_needs_a_cap(env):
    assert run.main(env.args("label", labeller="opus")) == 2
    assert env.calls() == []


# ---------------------------------------------------------------- retries, refusals, cap, resume


def test_invalid_json_then_retry(env, monkeypatch):
    monkeypatch.setenv("STUB_SEQ_CLAUDE", "badjson")
    run.main(env.args("label", "--max-usd", "5", "--workers", "1", labeller="opus"))
    raw = env.raw("opus55", first_id())
    assert raw["status"] == "ok" and len(raw["attempts"]) == 2
    assert "not valid JSON" in raw["attempts"][0]["errors"][0]
    second = env.calls("claude")[1]["stdin"]
    assert "failed validation" in second and "not valid JSON" in second
    assert raw["cost_usd"] == pytest.approx(0.20)


def test_fenced_reply_is_accepted(env, monkeypatch):
    monkeypatch.setenv("STUB_SEQ_CLAUDE", "fence")
    run.main(env.args("label", "--max-usd", "5", "--workers", "1", labeller="opus"))
    assert len(env.raw("opus55", first_id())["attempts"]) == 1


def test_schema_failure_twice_is_recorded_then_resumed(env, monkeypatch):
    monkeypatch.setenv("STUB_SEQ_CLAUDE", "schema,schema")
    run.main(env.args("label", "--max-usd", "5", "--workers", "1", labeller="opus"))
    raw = env.raw("opus55", first_id())
    assert raw["status"] == "invalid" and len(raw["attempts"]) == 2
    assert any("not labelled" in e for e in raw["attempts"][1]["errors"])
    lf = c.lk.LabelFile.load(env.labels / "fx.opus55.jsonl")
    assert first_id() not in lf.screens
    n = len(env.calls("claude"))
    run.main(env.args("label", "--max-usd", "5", "--workers", "1", labeller="opus"))
    assert len(env.calls("claude")) == n + 1  # only the failed screen runs again
    assert env.raw("opus55", first_id())["status"] == "ok"


def test_codex_tool_use_is_rejected_and_retried(env, monkeypatch):
    monkeypatch.setenv("STUB_SEQ_CODEX", "tools")
    run.main(env.args("label", "--workers", "1", labeller="codex"))
    raw = env.raw("codex61", first_id())
    assert raw["status"] == "ok" and len(raw["attempts"]) == 2
    assert "tool call" in raw["attempts"][0]["errors"][0]


@pytest.mark.parametrize("who,lab", [("claude", "opus55"), ("codex", "codex61")])
def test_refusal_recorded_not_retried(env, monkeypatch, who, lab):
    monkeypatch.setenv(f"STUB_SEQ_{who.upper()}", "refuse")
    key = "opus" if who == "claude" else "codex"
    run.main(env.args("label", "--max-usd", "5", "--workers", "1", labeller=key))
    raw = env.raw(lab, first_id())
    assert raw["status"] == "refused" and len(raw["attempts"]) == 1
    n = len(env.calls(who))
    run.main(env.args("label", "--max-usd", "5", "--workers", "1", labeller=key))
    assert len(env.calls(who)) == n  # resume skips the refusal
    run.main(env.args("label", "--max-usd", "5", "--workers", "1", "--retry-refused", labeller=key))
    assert len(env.calls(who)) == n + 1


def test_cli_error_is_recorded(env, monkeypatch):
    monkeypatch.setenv("STUB_SEQ_CLAUDE", "error")
    run.main(env.args("label", "--max-usd", "5", "--workers", "1", labeller="opus"))
    raw = env.raw("opus55", first_id())
    assert raw["status"] == "error" and len(raw["attempts"]) == 1


def test_budget_cap_stops_launches(env, monkeypatch):
    monkeypatch.setenv("STUB_COST", "0.40")
    run.main(env.args("label", "--max-usd", "1.0", "--per-call-usd", "0.5", "--workers", "1", labeller="opus"))
    # call 1: 0 + 0.5 <= 1; call 2: 0.4 + 0.4 <= 1; call 3: 0.8 + 0.4 > 1, so no more launches
    assert len(env.calls("claude")) == 2
    assert rn.ledger_total(env.labels / "raw" / "spend.jsonl") == pytest.approx(0.80)
    done = [p for p in (env.labels / "raw" / "fx" / "opus55").glob("*.json")]
    assert len(done) == 2


def test_budget_unit():
    b = rn.Budget(1.0, 0.5)
    assert b.try_reserve() and b.try_reserve()  # 0 + 2 × 0.5 = 1.0
    assert not b.try_reserve(block=False) and b.stopped
    b.release(0.3)
    b.release(0.3)
    assert b.spent == pytest.approx(0.6)


def test_resume_skips_done_screens(env):
    run.main(env.args("label", "--max-usd", "5"))
    n = len(env.calls())
    run.main(env.args("label", "--max-usd", "5"))
    assert len(env.calls()) == n


def test_screen_subset_and_unknown_ids(env):
    run.main(env.args("label", "--max-usd", "5", "--screens", first_id(), labeller="opus"))
    assert len(env.calls("claude")) == 1
    assert run.main(env.args("label", "--max-usd", "5", "--screens", "nope", labeller="opus")) == 2


def test_dry_run_calls_nothing(env):
    out = env.root / "dry"
    assert run.main(env.args("label", "--dry-run", "--out-dir", str(out))) == 0
    assert env.calls() == []
    files = sorted((out / "fx" / "opus55").glob("*.prompt.txt"))
    assert len(files) == N_SCREENS
    assert "## 4. Decision procedure" in files[0].read_text()


def test_sealed_real_refused_until_built(env):
    a = env.args("label", "--max-usd", "5")
    a[a.index("fx")] = "sealed_real"
    assert run.main(a) == 2
    assert env.calls() == []


# ---------------------------------------------------------------- converter


def fixture_screen(n=0):
    ls = c.LabelSet("t", FIX)
    f = sorted(FIX.glob("0*.json"))[n]
    s = json.loads(f.read_text())
    root = FIX
    src = c.find_source(root, s) or (FIX / "src" / s["source"]["repo"] / s["source"]["path"])
    return c.Screen(screen=s, file=f.name, sha256=c.sha256_bytes(f.read_bytes()), source_path=src,
                    source_text=src.read_text(), kind="test")


def reply_for(scr, over=None):
    over = over or {}
    labs = []
    for n in scr.labelable():
        lab = {"i": n["i"], "class": "KEEP", "target": None, "pattern": None, "conf": "clear", "partial": False,
               "ct": False, "equiv": [], "aside": [], "role_ok": True, "role_fix": None, "why": "w"}
        if not c.has_literal_words(n):
            lab.update({"class": None, "aside": ["UNRESOLVED"]})
        lab.update(over.get(n["i"], {}))
        labs.append(lab)
    return {"screen": scr.id, "nodes": labs, "missed": [], "aside_screen": []}


def test_converter_output_passes_check(tmp_path):
    scr = fixture_screen(0)  # pause menu: 0 eyebrow, 1 h1, 2 subtitle, 3.. buttons
    nodes = scr.by_i()
    obj = reply_for(scr, {0: {"class": "ECHO", "target": 1, "pattern": "R1"},
                            2: {"class": "NARRATE", "target": 3, "pattern": "R6", "ct": True, "conf": "judgement"},
                            1: {"pattern": "K11", "role_ok": False, "role_fix": "dialog_title"}})
    obj["missed"] = [{"line": 3, "text": "Invented hint", "class": "DECOR"}]
    obj["aside_screen"] = [{"code": "R19", "nodes": [3, 4]}]
    recs = tr.to_records(obj, scr, "opus55", "a" * 64, "2026-11-10T10:00:00Z")
    by = {tuple(r["loc"]): r for r in recs if "loc" in r}
    assert by[tuple(nodes[0]["loc"])]["target"] == nodes[1]["loc"]
    assert by[tuple(nodes[1]["loc"])]["role_fix"] == "dialog_title"
    assert recs[-1] == {"screen": scr.id, "done": True, "nodes": len(scr.labelable()), "labeller": "opus55",
                        "guide_sha256": "a" * 64, "time": "2026-11-10T10:00:00Z"}
    d = tmp_path / "screens"
    d.mkdir()
    shutil.copy(FIX / scr.file, d / scr.file)
    p = tmp_path / "t.opus55.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    res = lcheck.check(p, str(d))
    assert res["errors"] == [] and res["complete"] == 1


def test_converter_rules():
    scr = fixture_screen(4)  # status dashboard: nodes 14-16 have no literal words; 7 is aria (context only)
    obj = reply_for(scr)
    labels, errors, warnings = tr.normalise_reply(obj, scr)
    assert errors == [] and labels[14]["class"] is None and "UNRESOLVED" in labels[14]["aside"]
    # a class on a no-literal node is replaced by null, with a warning
    obj2 = reply_for(scr, {14: {"class": "KEEP", "aside": []}})
    labels, errors, warnings = tr.normalise_reply(obj2, scr)
    assert errors == [] and labels[14]["class"] is None and any("no literal words" in w for w in warnings)
    # null on a node with literal words, a missing node, a self target, a KEEP with a target, a wrong pattern
    bad = reply_for(scr, {0: {"class": None}, 1: {"class": "ECHO", "target": 1, "pattern": "R1"},
                            2: {"target": 0}, 3: {"class": "DECOR", "pattern": "R1"}})
    bad["nodes"] = [x for x in bad["nodes"] if x["i"] != 5]
    _, errors, _ = tr.normalise_reply(bad, scr)
    text = " | ".join(errors)
    for frag in ("class is null", "not labelled: 5", "target is the node itself"):
        assert frag in text, (frag, errors)
    # labelkit's record rules come after the structural ones
    bad2 = reply_for(scr, {2: {"target": 0}, 3: {"class": "DECOR", "pattern": "R1"}})
    _, errors, _ = tr.normalise_reply(bad2, scr)
    text = " | ".join(errors)
    assert "target set on a KEEP" in text and "not a DECOR pattern" in text
    # a context-only node in the reply is dropped
    obj3 = reply_for(scr)
    obj3["nodes"].append({**obj3["nodes"][0], "i": 7})
    labels, errors, warnings = tr.normalise_reply(obj3, scr)
    assert errors == [] and 7 not in labels and any("context only" in w for w in warnings)


def test_schema_is_codex_strict():
    def walk(s, path="$"):
        if s.get("type") == "object":
            assert s.get("additionalProperties") is False, path
            assert sorted(s["required"]) == sorted(s["properties"]), path
            for k, v in s["properties"].items():
                walk(v, f"{path}.{k}")
        for v in s.get("anyOf", []):
            walk(v, path)
        if "items" in s:
            walk(s["items"], path + "[]")
    for p in (c.SCHEMA_PATH, c.ADJ_SCHEMA_PATH):
        walk(json.loads(p.read_text()))


def test_schema_checker():
    s = c.load_schema()
    assert c.check_schema({"screen": "x", "nodes": [], "missed": [], "aside_screen": []}, s) == []
    errs = c.check_schema({"screen": "x", "nodes": [{"i": "0"}], "missed": [], "extra": 1}, s)
    text = " ".join(errs)
    assert "aside_screen" in text and "unknown fields ['extra']" in text and "expected integer" in text


def test_extract_json():
    assert c.extract_json('{"a": 1}')[0] == {"a": 1}
    assert c.extract_json('```json\n{"a": 1}\n```')[0] == {"a": 1}
    assert c.extract_json("[1]")[0] is None
    assert c.extract_json("nope")[1].startswith("reply is not valid JSON")


def test_literal_words_and_display():
    n = {"text": "Due in  eta hours hours", "vw": [2]}
    assert c.literal_words(n["text"], n["vw"]) == ["Due", "in", "hours"]
    assert c.display_text(n) == "Due in ⟨eta hours⟩ hours"
    assert not c.has_literal_words({"text": " detail", "vw": [1]})
    assert c.has_literal_words({"text": "●"})
    assert c.display_text({"text": "On  Off"}) == "On ‖ Off"


# ---------------------------------------------------------------- prompt


def test_prompt_prefix_stable_and_body_complete():
    a, b = fixture_screen(0), fixture_screen(4)
    guide = GUIDE.read_text()
    pa, pb = pr.render(a, guide), pr.render(b, guide)
    assert pa.prefix == pb.prefix and guide.strip() in pa.prefix
    assert "§4 decision" not in pa.body  # instructions live in the prefix only
    for n in a.screen["nodes"]:
        assert json.dumps(n["loc"]) .replace(" ", "") in pa.body.replace(" ", "")
    assert " 1 | " in pa.body and "Product kind (§2a): test" in pa.body
    assert '"no_literal_words": true' in pb.body and '"context_only": true' in pb.body
    assert "UNSURE may be used when you cannot decide after §4" in pa.prefix and "independently" not in pa.body
    assert "Many screens have no redundant copy" not in pa.prefix  # no prior towards KEEP (review M3)


# ---------------------------------------------------------------- adjudication


def test_dispute_rules():
    k = {"class": "KEEP", "target": None, "equiv": []}
    e1 = {"class": "ECHO", "target": 1, "equiv": []}
    e2 = {"class": "ECHO", "target": 2, "equiv": [1]}
    e3 = {"class": "ECHO", "target": 3, "equiv": []}
    d = {"class": "DECOR", "target": None, "equiv": []}
    assert adj.dispute_reasons(k, e1) == ["binary", "class"]
    assert adj.dispute_reasons(e1, d) == ["class"]
    assert adj.dispute_reasons(e1, e2) == []  # target in the other's equiv
    assert adj.dispute_reasons(e1, e3) == ["target"]
    assert adj.dispute_reasons({"class": "UNSURE"}, k) == ["binary", "class"]
    assert adj.ab_order("x", 0) in (adj.ORDER, adj.ORDER[::-1])
    flips = {adj.ab_order("s", i) for i in range(20)}
    assert len(flips) == 2  # both orders occur within one screen (per node)


def label_both_with_disagreement(env, monkeypatch):
    monkeypatch.setenv("STUB_FLAVOUR_CLAUDE", "a")
    monkeypatch.setenv("STUB_FLAVOUR_CODEX", "b")
    assert run.main(env.args("label", "--max-usd", "5")) == 0


def test_adjudication_agree(env, monkeypatch):
    label_both_with_disagreement(env, monkeypatch)
    monkeypatch.setenv("STUB_ADJ", "agree")
    rc = run.main(env.args("adjudicate", "--max-usd", "5"))
    assert rc == 0
    final = env.labels / "fx.final.jsonl"
    res = lcheck.check(final, None, str(GUIDE))
    assert res["errors"] == [] and res["complete"] == N_SCREENS
    recs = [json.loads(x) for x in final.read_text().splitlines() if '"loc"' in x]
    adjd = [r for r in recs if r["source"] == "adjudicated"]
    assert adjd and all(r["class"] == "NARRATE" and r["ct"] for r in adjd)
    assert all(len(r["adjudication"]) == 2 for r in adjd)
    assert not [r for r in recs if r["source"] == "contested"]
    # the adjudication prompts carry anonymous labels only
    adj_calls = [x for x in env.calls() if "<disputed>" in x["stdin"]]
    assert adj_calls and all("opus55" not in x["stdin"] and "codex61" not in x["stdin"] for x in adj_calls)
    # resume: nothing more to adjudicate
    n = len(env.calls())
    run.main(env.args("adjudicate", "--max-usd", "5"))
    assert len(env.calls()) == n


def test_adjudication_contested(env, monkeypatch):
    label_both_with_disagreement(env, monkeypatch)
    monkeypatch.setenv("STUB_ADJ", "split")
    run.main(env.args("adjudicate", "--max-usd", "5"))
    final = env.labels / "fx.final.jsonl"
    res = lcheck.check(final, None, str(GUIDE))
    assert res["errors"] == []
    recs = [json.loads(x) for x in final.read_text().splitlines() if '"loc"' in x]
    cont = [r for r in recs if r["source"] == "contested"]
    assert cont and all(r["class"] is None and r["conf"] == "contested" for r in cont)
    assert all({x["class"] for x in r["candidates"]} == {"KEEP", "NARRATE"} for r in cont)


def test_both_unsure_is_contested():
    scr = fixture_screen(0)
    ra = reply_for(scr, {2: {"class": "UNSURE"}})
    rb = reply_for(scr, {2: {"class": "UNSURE"}})
    sides = {}
    for lab, r in zip(adj.ORDER, (ra, rb)):
        labels, errs, _ = tr.normalise_reply(r, scr)
        assert errs == []
        sides[lab] = {"labels": labels, "reply": r, "guide_sha256": "a" * 64, "time": "2026-11-10T10:00:00Z"}
    disp = adj.screen_disputes(sides)
    assert disp["both_unsure"] == [2] and disp["adjudicate"] == []
    recs = adj.final_screen(scr, sides, {}, "2026-11-10T10:00:00Z")
    rec = next(r for r in recs if r.get("loc") == scr.by_i()[2]["loc"])
    assert rec["source"] == "contested" and rec["class"] is None
    errs, _ = c.lk.validate_record(rec, "final")
    assert errs == []


def test_final_record_validation():
    base = {"screen": "s", "loc": [1, 1, 1, 2], "class": None, "target": None, "equiv": [], "pattern": None,
            "aside": [], "partial": False, "ct": False, "conf": "contested", "role_ok": True, "labeller": "final",
            "guide_sha256": "a" * 64, "time": "2026-11-10T10:00:00Z", "source": "contested",
            "candidates": [{}, {}]}
    assert c.lk.validate_record(base, "final")[0] == []
    assert c.lk.validate_record({**base, "candidates": [{}]}, "final")[0]
    assert c.lk.validate_record({**base, "source": "agreed"}, "final")[0]
    # outside the final file, contested is not a valid conf
    assert c.lk.validate_record({k: v for k, v in base.items() if k not in ("source", "candidates")}
                                | {"labeller": "x"}, "x")[0]


# ---------------------------------------------------------------- agreement


def test_kappa_known_example():
    # 50 nodes: 20 both REDUNDANT, 15 both KEEP, 5 R/K, 10 K/R -> po 0.7, pe 0.5, κ 0.4
    pairs = ([("REDUNDANT", "REDUNDANT")] * 20 + [("KEEP", "KEEP")] * 15 + [("REDUNDANT", "KEEP")] * 5
             + [("KEEP", "REDUNDANT")] * 10)
    assert agree.cohen_kappa(pairs, ["KEEP", "REDUNDANT"]) == pytest.approx(0.4)
    # AC1: pi_R = (25 + 30) / 100 = 0.55, pe = 2 × 0.55 × 0.45 / 1 = 0.495, AC1 = (0.7 − 0.495) / 0.505
    assert agree.gwet_ac1(pairs, ["KEEP", "REDUNDANT"]) == pytest.approx(0.205 / 0.505)
    pa, na = agree.pos_neg_agreement(pairs)
    assert pa == pytest.approx(40 / 55) and na == pytest.approx(30 / 45)
    assert agree.cohen_kappa([("A", "A"), ("B", "B")]) == pytest.approx(1.0)
    assert agree.cohen_kappa([("A", "A")] * 3) is None  # chance agreement 1


def test_report_before_and_after(env, monkeypatch):
    label_both_with_disagreement(env, monkeypatch)
    monkeypatch.setenv("STUB_ADJ", "agree")
    run.main(env.args("adjudicate", "--max-usd", "5"))
    assert run.main(env.args("report")) == 0
    res = json.loads((env.labels / "fx.agreement.json").read_text())
    assert res["screens_both"] == N_SCREENS and res["disputed_nodes"] > 0
    assert res["before"]["binary"]["kappa"] < 1.0
    assert res["after"]["binary"]["kappa"] == pytest.approx(1.0)
    assert res["final"]["contested_rate"] == 0.0
    assert "subtitle" in res["per_role"] and "game-ui" in res["per_repo"]
    md = (env.labels / "fx.agreement.md").read_text()
    assert "Binary κ" in md and "Per extractor role" in md


# ---------------------------------------------------------------- estimate


def test_estimate_runs_on_rendered_prompts(tmp_path):
    ds = tmp_path / "datasets"
    make_set(ds, "reference_real")
    make_set(ds, "calibration")
    e = est.estimate(["all"], ds, GUIDE)
    rows = {r["row"].split(":")[0]: r for r in e["rows"]}
    assert set(rows) >= {"reference_real", "calibration", "sealed_real"}
    assert rows["sealed_real"]["screens"] == est.SEALED_SCREENS
    assert e["total"]["usd_5m"] > 0 and e["total"]["usd_1h"] > e["total"]["usd_5m"]
    assert "Total" in est.markdown(e)


def test_first_n_screens_for_the_pilot(env):
    out = env.root / "dry"
    assert run.main(env.args("label", "--dry-run", "--first", "3", "--out-dir", str(out), labeller="opus")) == 0
    assert len(list((out / "fx" / "opus55").glob("*.prompt.txt"))) == 3
    run.main(env.args("label", "--max-usd", "5", "--first", "3", labeller="opus"))
    assert len(env.calls("claude")) == 3
    assert run.main(env.args("report")) == 0  # one labeller only: no screens labelled by both, no crash
