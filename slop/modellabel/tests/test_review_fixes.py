"""Tests for the fixes from the harness and protocol reviews of slop/modellabel/ (2026-11-10). No model is called: the
stub `claude` and `codex` (tests/stubs/) stand in, and PATH holds only the stubs and the system directories.
Fixtures are the labeller's invented screens; copy below is invented.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SLOP = HERE.parent.parent
sys.path.insert(0, str(SLOP))

from modellabel import adjudicate as adj  # noqa: E402
from modellabel import agree  # noqa: E402
from modellabel import common as c  # noqa: E402
from modellabel import prompt as pr  # noqa: E402
from modellabel import run  # noqa: E402
from modellabel import runners as rn  # noqa: E402
from modellabel import to_records as tr  # noqa: E402
from modellabel.tests.test_modellabel import (FIX, GUIDE, N_SCREENS, STUBS, env, first_id,  # noqa: E402,F401
                                              fixture_screen, make_set, reply_for)

LEDGER = ("raw", "spend.jsonl")


def ledger(env):
    return env.labels.joinpath(*LEDGER)


# ---------------------------------------------------------------- spend cap (harness review 1, 2, 5, 12, 13)


def test_pilot_both_labellers_run_every_screen(env, monkeypatch):
    """The README pilot shape: a cap that admits fewer calls in flight than workers waits, never stops Codex."""
    rc = run.main(env.args("label", "--first", "3", "--workers", "3", "--max-usd", "3", labeller="both"))
    assert rc == 0
    assert len(env.calls("claude")) == 3 and len(env.calls("codex")) == 3


def test_pilot_flag_picks_stratified_screens(env):
    ls = env.ls()
    picks = run.pilot_screens(ls)
    sizes = sorted(len(s.labelable()) for s in picks)
    assert len(picks) == 3 and sizes[-1] == max(len(s.labelable()) for s in ls.screens)
    assert run.main(env.args("label", "--pilot", "--max-usd", "5", labeller="opus")) == 0
    assert {x["stdin"].split('"')[1] for x in env.calls("claude")} == {s.id for s in picks}
    assert run.main(env.args("label", "--pilot", "--first", "2", "--max-usd", "5", labeller="opus")) == 2


def test_uncosted_call_is_charged_and_stops_the_cap(env, monkeypatch):
    monkeypatch.setenv("STUB_SEQ_CLAUDE", "nocost,nocost,nocost,nocost")
    rc = run.main(env.args("label", "--max-usd", "0.5", "--per-call-usd", "0.5", "--workers", "1", labeller="opus"))
    assert rc == 3
    assert len(env.calls("claude")) == 1
    s = rn.ledger_summary(ledger(env))
    assert s["charged_usd"] == pytest.approx(0.5) and s["estimated_usd"] == pytest.approx(0.5)


def test_garbled_output_is_an_error_and_charged(env, monkeypatch):
    monkeypatch.setenv("STUB_SEQ_CLAUDE", "garbled")
    rc = run.main(env.args("label", "--max-usd", "5", "--per-call-usd", "0.5", "--workers", "1",
                           "--screens", first_id(), labeller="opus"))
    raw = env.raw("opus55", first_id())
    assert rc == 1 and raw["status"] == "error" and raw["cost_usd"] == pytest.approx(0.5)
    assert rn.ledger_total(ledger(env)) == pytest.approx(0.5)


def test_timeout_kills_the_call_and_charges_it(env, monkeypatch):
    monkeypatch.setenv("STUB_SEQ_CLAUDE", "hang")
    rc = run.main(env.args("label", "--max-usd", "5", "--per-call-usd", "0.5", "--timeout", "2", "--workers", "1",
                           "--screens", first_id(), labeller="opus"))
    raw = env.raw("opus55", first_id())
    assert rc == 1 and raw["status"] == "error" and "timed out" in raw["error"]
    assert rn.ledger_total(ledger(env)) == pytest.approx(0.5)
    out = subprocess.run(["pgrep", "-f", str(STUBS / "claude")], capture_output=True, text=True)
    assert out.stdout.strip() == "", "the timed-out stub must be killed with its process group"


@pytest.mark.parametrize("mode", ["weird", "badutf8"])
def test_odd_cli_output_does_not_crash(env, monkeypatch, mode):
    monkeypatch.setenv("STUB_SEQ_CLAUDE", mode)
    rc = run.main(env.args("label", "--max-usd", "5", "--workers", "1", "--screens", first_id(), labeller="opus"))
    raw = env.raw("opus55", first_id())
    assert raw["status"] in ("ok", "error"), raw["status"]
    assert rc in (0, 1)


def test_a_crash_inside_a_call_is_recorded_and_charged():
    class Boom(rn.Runner):
        costs_money = True
        labeller = "opus55"
        model = "m"

        def __init__(self):
            pass

        def call(self, prefix, body, schema_path):
            raise TypeError("boom")

    b = rn.Budget(10.0, 1.5)
    out = rn.run_with_retry(Boom(), "p", "b", None, lambda o: ([], []), b)
    assert out["status"] == "error" and "TypeError" in out["error"]
    assert b.spent == pytest.approx(1.5) and b.inflight == 0


def test_max_tokens_is_not_retried(env, monkeypatch):
    monkeypatch.setenv("STUB_SEQ_CLAUDE", "maxtokens")
    run.main(env.args("label", "--max-usd", "5", "--workers", "1", "--screens", first_id(), labeller="opus"))
    raw = env.raw("opus55", first_id())
    assert raw["status"] == "truncated" and len(raw["attempts"]) == 1


def test_total_cap_counts_earlier_runs(env):
    ledger(env).parent.mkdir(parents=True)
    ledger(env).write_text(json.dumps({"cost_usd": 4.9, "set": "fx", "phase": "label"}) + "\n")
    rc = run.main(env.args("label", "--max-usd", "5", "--max-total-usd", "5", "--per-call-usd", "0.5",
                           labeller="opus"))
    assert rc == 3 and env.calls("claude") == []
    ledger(env).write_text(json.dumps({"cost_usd": 5.0}) + "\n")
    assert run.main(env.args("label", "--max-usd", "5", "--max-total-usd", "5", labeller="opus")) == 2


def test_ledger_counts_orphan_launches_and_survives_torn_lines(tmp_path):
    p = tmp_path / "spend.jsonl"
    b = rn.Budget(None, 2.0, p)
    lid = b.try_reserve()
    b.release(0.25, {"set": "x"}, lid)
    b.try_reserve()  # a launch whose process is killed: no charge line follows
    with open(p, "a") as f:
        f.write('{"time": "t", "event": "charge", "cost_')  # a torn line
    b2 = rn.Budget(None, 2.0, p)
    b2.release(0.5, {}, b2.try_reserve())
    s = rn.ledger_summary(p)
    assert s["charged_usd"] == pytest.approx(0.75) and s["orphan_launches"] == 1
    assert s["total_usd"] == pytest.approx(2.75) and s["bad_lines"] == 1


def test_budget_waits_for_in_flight_calls_instead_of_stopping():
    b = rn.Budget(3.0, 2.0)
    first = b.try_reserve()
    got = []
    t = threading.Thread(target=lambda: got.append(b.try_reserve()))
    t.start()
    t.join(0.2)
    assert t.is_alive() and not b.stopped  # waiting, not refused
    b.release(0.3, None, first)
    t.join(2)
    assert got and got[0] and not b.stopped


# ---------------------------------------------------------------- concurrency, CLI checks (4, 9, 16, 17)


def test_second_run_is_refused_while_one_holds_the_lock(env):
    f = run.take_lock(env.labels)
    try:
        assert run.main(env.args("label", "--max-usd", "5", labeller="opus")) == 2
        assert env.calls() == []
    finally:
        run.release_lock(f)
    assert run.main(env.args("label", "--max-usd", "5", labeller="opus")) == 0


def test_write_json_uses_unique_temporary_files(tmp_path):
    p = tmp_path / "x.json"
    errs = []

    def w(n):
        try:
            for k in range(30):
                c.write_json(p, {"n": n, "k": k})
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=w, args=(n,)) for n in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert errs == [] and json.loads(p.read_text())["k"] == 29
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("n", ["0", "-2"])
def test_first_must_be_positive(env, n):
    assert run.main(env.args("label", "--max-usd", "5", "--first", n, labeller="opus")) == 2


def test_dry_run_defaults_to_the_gitignored_labels_dir(env):
    assert run.main(env.args("label", "--dry-run", labeller="codex")) == 0
    assert len(list((env.labels / "dryrun" / "fx" / "codex61").glob("*.prompt.txt"))) == N_SCREENS


def test_sealed_real_needs_the_freeze_record(env, tmp_path):
    ds = tmp_path / "datasets"
    make_set(ds, "sealed_real")
    a = env.args("label", "--max-usd", "5", "--registry-root", str(tmp_path))
    a[a.index("fx")] = "sealed_real"
    assert run.main(a) == 2
    assert env.calls() == []


def test_sealed_real_is_pinned_to_the_pool(env, tmp_path, monkeypatch):
    ds = tmp_path / "datasets"
    make_set(ds, "sealed_real")
    monkeypatch.setattr(run, "sealed_recorded", lambda root: None)
    a = env.args("label", labeller="codex")
    a[a.index("fx")] = "sealed_real"
    assert run.main(a) == 2  # no pool labels yet
    make_set(ds, "reference_real")
    b = env.args("label", labeller="codex")
    b[b.index("fx")] = "reference_real"
    assert run.main(b) == 0
    assert run.main(a) == 0  # same model, CLI, prompt and prefix as the pool
    assert len(list((env.labels / "raw" / "sealed_real" / "codex61").glob("*.json"))) == N_SCREENS


# ---------------------------------------------------------------- Codex backstop, refusals, soft checks (6, 8)


def test_unknown_or_unfinished_codex_items_count_as_tools(env, monkeypatch):
    monkeypatch.setenv("STUB_SEQ_CODEX", "unknown_item")
    run.main(env.args("label", "--workers", "1", "--screens", first_id(), labeller="codex"))
    raw = env.raw("codex61", first_id())
    assert raw["status"] == "ok" and len(raw["attempts"]) == 2
    assert "tool call" in raw["attempts"][0]["errors"][0]
    r = rn.parse_codex_events(json.dumps({"type": "item.started", "item": {"id": "x", "type": "command_execution"}}),
                              '{"screen": "s"}')
    assert r.tool_calls == 1


def test_codex_disables_every_listed_tool_feature():
    for f in ("sleep_tool", "goals", "code_mode_host", "browser_use_external", "in_app_browser", "remote_plugin"):
        assert f in rn.CODEX_DISABLE


def test_a_broken_reply_starting_i_wont_is_retried_not_refused(env, monkeypatch):
    monkeypatch.setenv("STUB_SEQ_CLAUDE", "wont")
    run.main(env.args("label", "--max-usd", "5", "--workers", "1", "--screens", first_id(), labeller="opus"))
    raw = env.raw("opus55", first_id())
    assert raw["status"] == "ok" and len(raw["attempts"]) == 2
    assert rn.is_refusal_text("I can't help with that request.")
    assert not rn.is_refusal_text("I won't repeat it: {\"screen\"")


def test_precedence_violation_is_sent_back_once(env, monkeypatch):
    monkeypatch.setenv("STUB_SEQ_CLAUDE", "precedence")
    run.main(env.args("label", "--max-usd", "5", "--workers", "1", "--screens", first_id(), labeller="opus"))
    raw = env.raw("opus55", first_id())
    assert raw["status"] == "ok" and len(raw["attempts"]) == 2
    assert raw["attempts"][0]["soft"] and "ranks higher" in raw["attempts"][0]["soft"][0]
    assert "look inconsistent with the guide" in env.calls("claude")[1]["stdin"]


# ---------------------------------------------------------------- tokens, prompt markers, windows (10, 11, M2, M3)


def test_literal_words_match_the_rule_layer():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    cases = [(" नमस्ते", [1]), (" नमस्ते world", [1]), ("Due in eta hours hours", [2]),
             ("Café ́x 2,5 kg", None), ("l’été  naïve", None), ("x² ½ Ⅻ", None), ("a b c d e f", None),
             ("Ünïcödé 3.14 test1", [1]), ("ŉ ǅ ﬁ", None)]
    js = ("import {literalWords} from %s; const cases = JSON.parse(process.argv[1]);"
          "console.log(JSON.stringify(cases.map(([t, v]) => literalWords(t, v ?? undefined).length)));"
          % json.dumps(str(SLOP / "rules" / "src" / "text.js")))
    out = subprocess.run([node, "--input-type=module", "-e", js, json.dumps(cases)], capture_output=True, text=True)
    if out.returncode != 0 and "literalWords" in out.stderr:
        pytest.skip("rules/src/text.js exports no literalWords")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == [len(c.literal_words(t, v)) for t, v in cases]


def test_literal_markers_are_doubled_and_the_fence_outgrows_the_source():
    assert c.display_text({"text": "a ⟨b⟩ ‖ c"}) == "a ⟨⟨b⟩⟩ ‖‖ c"
    assert c.display_text({"text": "x y", "vw": [1]}) == "⟨x⟩ y"
    assert pr.fence_for("x ``` y ```` z") == "`````"
    assert pr.fence_for(None) == "```"


def big_screen(n=80):
    """An invented screen of n nodes, one cluster each, that the tokeniser splits into two windows."""
    nodes = [{"i": i, "k": i, "role": "body", "text": f"Invented line {i}", "flags": [], "loc": [i + 1, 1, i + 1, 10]}
             for i in range(n)]
    s = {"schema": "slop.screen/1", "id": "bigscreen0000001", "framework": "tsx",
         "source": {"repo": "game-ui", "path": "src/Big.tsx"}, "clusters": [{"k": i, "kind": "body", "line": i + 1}
                                                                             for i in range(n)], "nodes": nodes}
    return c.Screen(screen=s, file="big.json", sha256="0" * 64, source_path=None, source_text="x\n" * n, kind="test")


def test_cross_window_targets_are_rejected_and_windows_are_shown():
    scr = big_screen()
    win = scr.windows()
    assert len(set(win.values())) == 2
    far = max(win, key=lambda i: win[i])
    obj = reply_for(scr, {0: {"class": "ECHO", "target": far, "pattern": "R1"}})
    _, errors, _ = tr.normalise_reply(obj, scr)
    assert any("window" in e for e in errors)
    body = pr.render(scr, GUIDE.read_text()).body
    assert '"w": 1' in body and "Windows: 2" in body


def test_prompt_has_no_prior_towards_keep():
    p = pr.render(fixture_screen(0), GUIDE.read_text())
    assert "no redundant copy at all" not in p.prefix and "rather than guess" not in p.prefix
    assert "UNSURE may be used when you cannot decide after §4 (§3)" in p.prefix
    assert "UNSURE may be used when you cannot decide after §4 (§3)" in pr.prefix(GUIDE.read_text(), adjudicate=True)


# ---------------------------------------------------------------- kinds, sibling source (B2, minor)


def test_fixture_note_is_cut_from_the_kind():
    assert c.neutral_kind("first-person game (fixture note: AI-built pause menu, the motivating example)") == \
        "first-person game"
    assert c.kind_for({"game-ui": "first-person game (fixture note: x)"}, {"source": {"repo": "game-ui"}}) == \
        "first-person game"


def test_label_refuses_screens_without_a_kind(env, tmp_path):
    (env.root / "datasets" / "kinds.json").write_text("{}")
    assert run.main(env.args("label", "--dry-run", labeller="codex")) == 2
    assert env.calls() == []


def test_label_refuses_screen_json_as_source(env):
    ls = env.ls()
    scr = ls.screens[0]
    scr.source_path.write_text(json.dumps(scr.screen))
    assert run.main(env.args("label", "--dry-run", labeller="codex")) == 2


# ---------------------------------------------------------------- the key (B1, M1, M4, M5, minor)


def sides_for(scr, ra, rb):
    out = {}
    for lab, r in zip(adj.ORDER, (ra, rb)):
        labels, errs, _ = tr.normalise_reply(r, scr)
        assert errs == [], errs
        out[lab] = {"labels": labels, "reply": r, "guide_sha256": "a" * 64, "time": "2026-11-10T10:00:00Z"}
    return out


def test_class_only_contest_keeps_the_binary_consensus():
    scr = fixture_screen(0)  # 0 eyebrow, 1 h1, 2 subtitle, 3.. buttons
    ra = reply_for(scr, {0: {"class": "ECHO", "target": 1, "pattern": "R1"}})
    rb = reply_for(scr, {0: {"class": "DECOR", "pattern": "R2"}})
    sides = sides_for(scr, ra, rb)
    disp = adj.screen_disputes(sides)
    assert disp["adjudicate"] == [0]
    split = {adj.ORDER[0]: {0: {**ra["nodes"][0], "reason": "r"}}, adj.ORDER[1]: {0: {**rb["nodes"][0], "reason": "r"}}}
    recs = adj.final_screen(scr, sides, split, "2026-11-10T10:00:00Z")
    rec = next(r for r in recs if r.get("loc") == scr.by_i()[0]["loc"])
    assert rec["source"] == "contested" and rec["binary"] == "REDUNDANT"
    assert c.lk.validate_record(rec, "final")[0] == []
    agreed = next(r for r in recs if r.get("loc") == scr.by_i()[3]["loc"])
    assert agreed["binary"] == "KEEP"
    assert adj.resolve_binary(rec, "key") == "REDUNDANT"


def test_fixed_resolutions_for_the_sensitivity_analysis():
    scr = fixture_screen(0)
    ra = reply_for(scr, {2: {"class": "NARRATE", "target": 3, "pattern": "R6", "ct": True}})
    rb = reply_for(scr, {2: {"class": "UNSURE", "pattern": None}})
    sides = sides_for(scr, ra, rb)
    recs = adj.final_screen(scr, sides, {}, "2026-11-10T10:00:00Z")
    rec = next(r for r in recs if r.get("loc") == scr.by_i()[2]["loc"])
    assert rec["source"] == "contested" and rec["binary"] is None
    assert [adj.resolve_binary(rec, m) for m in adj.RESOLUTIONS] == [None, "REDUNDANT", "KEEP", "REDUNDANT", None]


def test_ab_order_is_per_node_and_conf_is_hidden():
    scr = fixture_screen(2)
    over = {n["i"]: {"class": "DECOR", "pattern": "R2"} for n in scr.labelable() if c.has_literal_words(n)}
    sides = sides_for(scr, reply_for(scr, over), reply_for(scr))
    disp = adj.screen_disputes(sides)
    p, order = adj.adjudication_prompt(scr, GUIDE.read_text(), sides, disp)
    assert len({tuple(v.values()) for v in order.values()}) == 2  # both orders on one screen
    assert '"conf"' not in p.body.split("<disputed>")[1]


def test_pattern_ties_do_not_always_go_to_opus():
    picks = {adj.tie_pick("s", i) for i in range(30)}
    assert picks == set(adj.ORDER)


def test_screen_level_items_need_both_labellers():
    scr = fixture_screen(0)
    ra, rb = reply_for(scr), reply_for(scr)
    ra["missed"] = [{"line": 3, "text": "Invented hint text", "class": "DECOR"},
                    {"line": 9, "text": "Only one saw this", "class": "KEEP"}]
    rb["missed"] = [{"line": 3, "text": "invented hint", "class": "ECHO"}]
    ra["aside_screen"] = [{"code": "R19", "nodes": [3, 4]}]
    asides, missed = adj.screen_level(scr, ra, rb)
    assert asides == [] and missed == [{"line": 3, "text": "Invented hint text", "class": "UNSURE"}]


def test_incomplete_key_fails_the_build(env):
    assert run.main(env.args("label", "--max-usd", "5", labeller="opus")) == 0
    assert run.main(env.args("build")) == 1  # Codex has not labelled: no screen has both labels
    assert run.main(env.args("label", labeller="codex")) == 0
    assert run.main(env.args("build")) == 0  # both labelled and the stubs agree: complete


def test_report_has_the_review_statistics(env, monkeypatch):
    monkeypatch.setenv("STUB_FLAVOUR_CLAUDE", "a")
    monkeypatch.setenv("STUB_FLAVOUR_CODEX", "b")
    run.main(env.args("label", "--max-usd", "5"))
    monkeypatch.setenv("STUB_ADJ", "split")
    run.main(env.args("adjudicate", "--max-usd", "5"))
    assert run.main(env.args("report")) == 0
    res = json.loads((env.labels / "fx.agreement.json").read_text())
    for k in ("binary_unsure", "ceiling", "ct_disagree"):
        assert k in res["before"]
    assert res["before_ci"].get("ac1_ci95") is not None or res["before_ci"] == {}
    assert "not a control role" in res["subsets_before"]
    assert set(res["concession"]["by_model"]) == set(adj.ORDER)
    assert res["kappa_gate"]["reads"].startswith("before")
    assert res["coverage"]["screens_missing"] == 0
    md = (env.labels / "fx.agreement.md").read_text()
    assert "Inter-labeller ceiling" in md and "κ gate" in md


def test_guide_example_screen_is_left_out_of_agreement(env, monkeypatch):
    man = env.root / "datasets" / "fx" / "manifest.json"
    man.write_text(json.dumps({"set": "fx", "fixture": {"id": first_id()}}))
    run.main(env.args("label", "--max-usd", "5"))
    run.main(env.args("report"))
    res = json.loads((env.labels / "fx.agreement.json").read_text())
    assert res["excluded_screens"] == [first_id()]
    assert first_id() not in {s.id for s in run.pilot_screens(env.ls())}


# ---------------------------------------------------------------- checks on the key (M6)


def test_nosource_variant_is_written_apart_and_reported(env):
    assert run.main(env.args("label", "--max-usd", "5")) == 0
    assert run.main(env.args("label", "--variant", "nosource", labeller="codex")) == 0
    call = env.calls("codex")[-1]["stdin"]
    assert "<source>(not shown" in call
    assert env.raw("codex61-nosource", first_id())["status"] == "ok"
    run.main(env.args("report"))
    res = json.loads((env.labels / "fx.agreement.json").read_text())
    assert res["variants"]["codex61-nosource"]["screens"] == N_SCREENS
    assert run.main(env.args("build")) == 0
    assert "nosource" not in (env.labels / "fx.final.jsonl").read_text()
    assert (env.labels / "fx.codex61-nosource.jsonl").exists()


def test_retest_variant_samples_a_tenth(env):
    assert run.main(env.args("label", "--variant", "retest", labeller="codex")) == 0
    assert len(env.calls("codex")) == 1  # 10% of 6 screens, at least one


# ---------------------------------------------------------------- freeze gate


def test_freeze_refuses_an_incomplete_key(tmp_path):
    sys.path.insert(0, str(SLOP / "scripts"))
    import freeze

    d = tmp_path / "labels"
    d.mkdir()
    (d / "s.final.meta.json").write_text(json.dumps({"screens": [{"id": "a"}, {"id": "b"}]}))
    (d / "s.final.jsonl").write_text(json.dumps({"screen": "a", "done": True}) + "\n")
    probs = freeze.incomplete_keys(tmp_path)
    assert probs and "1 of 2 screens" in probs[0]
    (d / "s.final.jsonl").write_text("".join(json.dumps({"screen": x, "done": True}) + "\n" for x in "ab"))
    assert freeze.incomplete_keys(tmp_path) == []
