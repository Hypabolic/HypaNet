"""scripts/cost.py, scripts/audit.py and scripts/launch_checks.py on synthetic transcripts (no model is called)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit, cost, launch_checks


def _log(path: Path, events: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events))
    return path


def _result(sid: str, total: float, text: str = "done", subtype: str = "success") -> dict:
    return {"type": "result", "subtype": subtype, "session_id": sid, "total_cost_usd": total, "num_turns": 3,
            "result": text, "modelUsage": {"claude-fable-5-2": {}}}


def _assistant(mid: str, content: list, model: str = "claude-fable-5-2", **extra) -> dict:
    return {"type": "assistant", "message": {"id": mid, "model": model, "content": content,
                                             "usage": {"input_tokens": 100, "output_tokens": 50}, **extra}}


def test_constants_follow_the_spec():
    assert cost.CAP_USD == 200.0
    assert cost.PRICES["claude-fable-5-2"] == (8.0, 40.0, 10.0, 16.0, 0.20)
    assert audit.MODEL == "claude-fable-5-2"
    for ok in ("smoke", "smoke2", "r1", "r1.c1", "r1.c30"):
        assert cost.NAME_RE.match(ok)
    for bad in ("r2", "r1.c0", "r1.x", "clouds"):
        assert not cost.NAME_RE.match(bad)


def test_one_cap_over_smoke_and_r1(tmp_path):
    smoke, r1 = tmp_path / "smoke" / "logs", tmp_path / "r1" / "logs"
    _log(smoke / "smoke.stream.jsonl", [_result("s0", 11.2)])
    _log(r1 / "r1.stream.jsonl", [_result("s1", 40.0)])
    _log(r1 / "r1.c1.stream.jsonl", [_result("s1", 55.5)])
    _log(r1 / "odd.stream.jsonl", [_result("s2", 1.0)])
    s = cost.summary([smoke, r1])
    assert s["launches"]["r1.c1"]["cost_usd"] == pytest.approx(15.5)
    assert s["total_usd"] == pytest.approx(11.2 + 55.5 + 1.0) and s["unexpected"] == ["odd"]
    assert cost.remaining_floor(s) == 132.3
    assert cost.session_total([smoke, r1], "s1") == 55.5
    (r1 / "PRIOR_SPEND.json").write_text(json.dumps([{"usd": 140, "what": "aborted"}]))
    s = cost.summary([smoke, r1])
    assert s["reached"] and cost.check(s)[0] == cost.EXIT_CAP


def test_unresolved_launches(tmp_path):
    logs = tmp_path / "logs"
    _log(logs / "r1.stream.jsonl", [_assistant("m1", [])])
    s = cost.summary([logs])
    assert s["lower_bound_only"] == ["r1"] and cost.check(s)[0] == cost.EXIT_UNRESOLVED
    (logs / "r1.launch.json").write_text(json.dumps({"max_budget_usd": 123.45}))
    s = cost.summary([logs])
    assert s["launches"]["r1"]["cost_usd"] == 123.45 and cost.check(s)[0] == 0
    (logs / "COST_OVERRIDES.json").write_text(json.dumps({"r1": {"usd": 7.0, "reason": "console"}}))
    assert cost.summary([logs])["launches"]["r1"]["cost_usd"] == 7.0


def test_budget_flag_and_version():
    assert launch_checks.budget_flag("smoke", 187.659) == 10.0
    assert launch_checks.budget_flag("smoke", 6.789) == 6.78
    assert launch_checks.budget_flag("start", 187.659) == 187.65
    assert launch_checks.budget_flag("continue", 0.999) == 0.99
    assert launch_checks.version_ok("2.1.322 (Claude Code)", "2.1.321")[0]
    assert launch_checks.version_ok("2.1.321 (Claude Code)", "2.1.321")[0]
    assert launch_checks.version_ok("2.2.0", "2.1.321")[0]
    assert not launch_checks.version_ok("2.1.311 (Claude Code)", "2.1.321")[0]
    assert not launch_checks.version_ok("", "2.1.321")[0]


def test_completion_must_be_a_line_of_its_own(tmp_path):
    for text, done in (("Built.\n\nMODEL COMPLETE", True), ("MODEL COMPLETE  \n", True),
                       ("MODEL COMPLETE.", False), ("The MODEL COMPLETE marker comes later", False),
                       ("NETWORK COMPLETE", False)):
        log = _log(tmp_path / f"{abs(hash(text))}.stream.jsonl", [_result("s", 1.0, text)])
        assert audit.is_complete(log) is done, text


def _audit(tmp_path: Path, blocks: list[dict], **kw) -> dict:
    author = tmp_path / "HypaNet-slop-author"
    (author / "work").mkdir(parents=True, exist_ok=True)
    events = [{"type": "system", "subtype": "init", "session_id": "s", "model": "claude-fable-5-2"},
              _assistant("m1", blocks, **kw), _result("s", 2.0, "MODEL COMPLETE")]
    log = _log(tmp_path / "logs" / "r1.stream.jsonl", events)
    return audit.audit(log, author)


def _tool(name: str, **inp) -> dict:
    return {"type": "tool_use", "id": f"t{abs(hash(json.dumps(inp, sort_keys=True)))}", "name": name, "input": inp}


def test_audit_tools_and_commands(tmp_path):
    rep = _audit(tmp_path, [_tool("Bash", command="./tok --text 'Resume'"),
                            _tool("Bash", command="./score work --set dev"),
                            _tool("Bash", command="./lint examples/dev/x.json"), _tool("Bash", command="ls work"),
                            _tool("Read", file_path="KIT.md")])
    assert rep["violations"] == [] and rep["model_complete"] is True
    rep = _audit(tmp_path, [_tool("Bash", command="python3 -c 'print(1)'"), _tool("Bash", command="./tok x | head"),
                            _tool("Agent", prompt="review"), _tool("Write", file_path="KIT.md", content="x"),
                            _tool("Read", file_path="../HypaNet/slop/datasets/dev/screens.jsonl"),
                            _tool("Grep", pattern="x", path="/Users/someone/Development")])
    v = "\n".join(rep["violations"])
    assert "is not one of ./tok" in v and "compound" in v and "'Agent' is outside the kit" in v
    assert "tool modification: KIT.md" in v and "resolves outside the author dir" in v
    assert "absolute path outside" in v


def test_audit_source_rules(tmp_path):
    fit = "import numpy as np\nW = np.linalg.lstsq(A, b)[0]\n"
    from_examples = ("import json, numpy as np\nrows = json.load(open('examples/dev/x.json'))\n"
                     "np.savez('work/artifact.npz', tok_emb=E)\n")
    lexical = "from nltk.corpus import wordnet\n"
    rep = _audit(tmp_path, [_tool("Write", file_path="work/a.py", content=fit),
                            _tool("Write", file_path="work/b.py", content=from_examples),
                            _tool("Write", file_path="work/c.py", content=lexical),
                            _tool("Write", file_path="work/MODEL.md", content="Scored on dev; sealed_real is unseen."),
                            _tool("Bash", command="./python3 work/a.py")])
    v = "\n".join(rep["violations"])
    assert "fitting" in v and "reads examples/ and writes the artefact" in v and "forbidden library" in v
    assert any("sealed_real" in r for r in rep["review"])
    assert not any("sealed_real" in x for x in rep["violations"])
    rep = _audit(tmp_path, [_tool("Read", file_path="sealed_real/x.json")])
    assert any("sealed_real" in x for x in rep["violations"])


def test_audit_models_refusals_and_thinking(tmp_path):
    rep = _audit(tmp_path, [{"type": "thinking", "thinking": "", "signature": "a"},
                            {"type": "thinking", "thinking": "plan the heads", "signature": "b"},
                            {"type": "text", "text": "ok"}], stop_reason="refusal",
                 stop_details={"category": "frontier_llm"})
    assert rep["refusals"] == {"frontier_llm": 1}
    assert rep["thinking"] == {"blocks": 2, "with_text": 1, "share_with_text": 0.5}
    assert rep["tokens"]["output_tokens"] == 50
    rep = _audit(tmp_path, [{"type": "text", "text": "ok"}], model="claude-opus-5")
    assert any("claude-opus-5" in v for v in rep["violations"])


def test_settings_check(tmp_path, monkeypatch):
    parent = tmp_path / "p"
    repo, author, other = parent / "HypaNet", parent / "HypaNet-slop-author", parent / "HypaNet-slop-smoke"
    for d in (repo, author, other):
        d.mkdir(parents=True)
    home = tmp_path / "home"
    monkeypatch.setattr(launch_checks, "sealed_repo_roots", lambda: [home / "Development" / "Sealed"])
    a = lambda p: "/" + str(p)  # noqa: E731
    import os
    import re

    uid = os.getuid()
    deny = [f"Read({a(repo)}/**)", f"Read(//tmp/claude-{uid}/**)", f"Read(//private/tmp/claude-{uid}/**)",
            f"Read({a(home / 'Development/hypabolic/slop-private')}/**)", f"Read({a(home / 'Development/Sealed')}/**)",
            f"Read({a(home / '.claude/projects' / re.sub(r'[^A-Za-z0-9]', '-', str(repo)))}/**)",
            f"Read({a(parent)}/HypaNet-slop-s*/**)", f"Read({a(home / '.codex')}/**)", f"Read({a(home / '.claude.json')})"]
    allow = [f"Bash(./{t}:*)" for t in launch_checks.TOOLS] + ["Write(./work/**)", "Edit(./work/**)"]
    s = {"autoMemoryEnabled": False, "switchModelsOnFlag": False,
         "permissions": {"defaultMode": "dontAsk", "deny": deny, "allow": allow}}
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(s))
    assert launch_checks.check_settings(p, author, parent, repo, home) == []
    s["permissions"]["allow"].append("Read(//etc/**)")
    s["permissions"]["deny"] = deny[1:]
    p.write_text(json.dumps(s))
    probs = launch_checks.check_settings(p, author, parent, repo, home)
    assert any("the repo" in x for x in probs) and any("absolute path" in x for x in probs)
    s["permissions"]["deny"] = [r for r in deny if ".codex" not in r and ".claude.json" not in r]
    p.write_text(json.dumps(s))
    probs = launch_checks.check_settings(p, author, parent, repo, home)
    assert any("~/.codex" in x for x in probs) and any("~/.claude.json" in x for x in probs)


def test_r2_names_and_its_own_cap(tmp_path):
    for good in ("r2", "r2.c1", "r2.c12"):
        assert cost.R2_NAME_RE.match(good) and not cost.NAME_RE.match(good)
    for bad in ("r1", "r2.c0", "smoke"):
        assert not cost.R2_NAME_RE.match(bad)
    capf = tmp_path / "CAP_USD"
    with pytest.raises(cost.NoCap):
        cost.read_cap(capf)
    capf.write_text("0\n")
    with pytest.raises(cost.NoCap):
        cost.read_cap(capf)
    for bad in ("r2: 250\n", "250 or 300\n", "250 USD max\n", "\n"):
        capf.write_text(bad)
        with pytest.raises(cost.NoCap):
            cost.read_cap(capf)
    for good in ("US$250\n", " $250.00 \n", "250"):
        capf.write_text(good)
        assert cost.read_cap(capf) == 250.0
    capf.write_text("250\n")
    assert cost.read_cap(capf) == 250.0
    logs = tmp_path / "r2" / "logs"
    _log(logs / "r2.stream.jsonl", [_result("t", 30.0)])
    _log(logs / "r2.c1.stream.jsonl", [_result("t", 42.5)])
    dirs, cap, name_re = cost.run_config("r2", [logs], capf)
    s = cost.summary(dirs, cap, name_re)
    assert s["cap_usd"] == 250.0 and s["total_usd"] == pytest.approx(42.5) and s["unexpected"] == []
    assert cost.remaining_floor(s) == 207.5
    assert cost.main(["--run", "r2", "--logs", str(logs), "--cap-file", str(tmp_path / "none"), "--cap"]) == 2
    assert cost.main(["--run", "r2", "--logs", str(logs), "--cap-file", str(capf), "--check"]) == 0


def _audit_named(tmp_path: Path, name: str, blocks: list[dict]) -> dict:
    author = tmp_path / "HypaNet-slop-author-r2"
    (author / "work").mkdir(parents=True, exist_ok=True)
    events = [{"type": "system", "subtype": "init", "session_id": "s", "model": "claude-fable-5-2"},
              _assistant("m1", blocks), _result("s", 2.0, "MODEL COMPLETE")]
    log = _log(tmp_path / name / "logs" / f"{name}.stream.jsonl", events)
    return audit.audit(log, author)


def test_audit_r2_allows_the_thesaurus_but_not_the_examples(tmp_path):
    reads_resources = ("import numpy as np\n# WordNet links from the kit's thesaurus\n"
                       "rows = open('resources/thesaurus/out/stem_links.tsv').read()\n"
                       "np.savez('work/artifact.npz', tok_emb=E)\n")
    from_examples = "rows = open('examples/dev/x.json').read()\nnp.savez('work/artifact.npz', tok_emb=E)\n"
    blocks = [_tool("Write", file_path="work/build.py", content=reads_resources),
              _tool("Read", file_path="resources/thesaurus/README.md")]
    rep = _audit_named(tmp_path / "a", "r2", blocks)
    assert rep["run"] == "r2" and rep["violations"] == [], rep["violations"]
    rep = _audit_named(tmp_path / "b", "r1", blocks)
    assert any("WordNet" in v for v in rep["violations"])
    rep = _audit_named(tmp_path / "c", "r2", [_tool("Write", file_path="work/b.py", content=from_examples),
                                              _tool("Write", file_path="work/c.py", content="import nltk\n"),
                                              _tool("Write", file_path="resources/codex/x.txt", content="x")])
    v = "\n".join(rep["violations"])
    assert "reads examples/ and writes the artefact" in v and "forbidden library" in v
    assert "tool modification: resources/codex/x.txt" in v
    assert audit.default_author(Path("/x/runs/r2/logs/r2.c3.stream.jsonl")).name == "HypaNet-slop-author-r2"
    assert audit.default_author(Path("/x/runs/r1/logs/r1.stream.jsonl")).name == "HypaNet-slop-author"
