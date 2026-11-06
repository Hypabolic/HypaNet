"""v0.5 Phase B author kit: revision applier, tools, prepare, ./py isolation, build, launcher check, audit, cost.

Run ONLY with an explicit --basetemp (r1's runner refuses a turn while pytest-of-$USER exists in a temp root):

    uv run pytest tests/test_v05b_kit.py -q -p no:cacheprovider --basetemp=<scratch>/pytest_v05b

The dry prepare writes into $V05B_DRY_ROOT (default: the session scratchpad's v05b_author_dir/), uses the archived
v0.3 Phase A artefact as a stand-in for r1's weights and a fake replies dir, and never launches anything.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agent_loop_lab.v03.artifact import deserialize_state_dict, serialize_state_dict  # noqa: E402
from agent_loop_lab.v05 import kit_b  # noqa: E402
from agent_loop_lab.v05.edits import build_run, zero_state  # noqa: E402
from agent_loop_lab.v05.revision import apply_revision, build_problems  # noqa: E402

KIT = REPO / "experiments" / "v0_5" / "phase_b" / "author_kit"
STANDIN = REPO / "weight_program" / "v0_3" / "phase-a" / "artifact.dlw"
SCRATCH = Path("/private/tmp/claude-501/-Users-matthew-Development-hypabolic-HypaNet/"
               "84f94084-9d88-46ef-9b1b-8b4fda2dc583/scratchpad")
DRY_ROOT = Path(os.environ.get("V05B_DRY_ROOT", str(SCRATCH / "v05b_author_dry")))


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _no_pytest_of():
    user = os.environ.get("USER", "")
    import tempfile

    for t in {os.path.realpath(tempfile.gettempdir()), "/tmp", "/private/tmp"}:
        assert not os.path.exists(os.path.join(t, f"pytest-of-{user}")), f"pytest-of-{user} appeared in {t}"


# -- revision applier ------------------------------------------------------------------------------


def test_revision_applies_on_top_of_the_base_and_reports_every_problem():
    base = zero_state()
    base["head.bias"][:] = [1, 2, 3, 4]
    base["layers.0.attn.rel_bias"][1, 5] = 7
    text = ("# a comment\nhead.bias[0] = 5\nhead.bias[1:3] *= 2   # scale\n"
            "layers.0.attn.rel_bias[1, 0:3] = [0, -1,\n   -2]\n")
    sd, rep = apply_revision(base, text)
    assert sd["head.bias"].tolist() == [5, 4, 6, 4]
    assert sd["layers.0.attn.rel_bias"][1, :6].tolist() == [0, -1, -2, 0, 0, 7]  # untouched entries keep the base
    assert base["head.bias"].tolist() == [1, 2, 3, 4]  # base not modified
    assert rep["edit_report"]["applied"] == 3 and rep["edit_report"]["skipped"] == 0
    assert rep["changed_entries"] == {"layers.0.attn.rel_bias": 2, "head.bias": 3}
    assert build_problems(rep) == []
    bad = "tok_emb.weight[-1, 0] = 1\n```\nprose that is not a statement\nEND OF WEIGHTS\nhead.bias[3] +="
    sd2, rep2 = apply_revision(base, bad)
    probs = build_problems(rep2)
    assert len(probs) == 5
    assert any("fence line" in p for p in probs) and any("END OF WEIGHTS" in p for p in probs)
    assert any("negative index" in p for p in probs) and any("incomplete statement" in p for p in probs)
    assert torch.equal(sd2["head.bias"], torch.tensor([1.0, 2, 3, 4]))


def test_revision_empty_file_is_the_base_and_many_errors_are_capped():
    base = zero_state()
    sd, rep = apply_revision(base, "# nothing to change\n\n")
    assert rep["edit_report"]["statements"] == 0 and build_problems(rep) == [] and rep["changed_entries_total"] == 0
    _sd, rep = apply_revision(base, "".join(f"nope[{i}] = 1\n" for i in range(300)))
    probs = build_problems(rep)
    assert len(probs) == 51 and probs[-1] == "... and 250 more"


def test_revision_rejects_a_bad_base():
    sd = zero_state()
    sd["head.bias"] = sd["head.bias"].astype("float64")
    with pytest.raises(ValueError):
        apply_revision(sd, "")


# -- kit files ---------------------------------------------------------------------------------------


def test_vendored_files_differ_from_their_sources_only_in_the_named_imports():
    files = kit_b.vendored_files()
    src = (REPO / "src/agent_loop_lab/v05/edits.py").read_text().split("\n")
    ven = files["lib/hn_v05/edits.py"].decode().split("\n")
    diff = [(a, b) for a, b in zip(src, ven) if a != b]
    assert len(src) == len(ven) and diff == [tuple(p) for p in kit_b.EDITS_IMPORTS]
    src = (REPO / "src/agent_loop_lab/v03/artifact.py").read_text().split("\n")
    ven = files["lib/hn_v05/dlw3.py"].decode().split("\n")
    assert [(a, b) for a, b in zip(src, ven) if a != b] == [tuple(p) for p in kit_b.ARTIFACT_IMPORTS]
    assert files["lib/hn_v05/revision.py"] == (REPO / "src/agent_loop_lab/v05/revision.py").read_bytes()
    assert files["lib/v05_load.py"] == (KIT / "v05_load.py").read_bytes()


def _func_src(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    return next(ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_launcher_keeps_the_v03_launcher_and_tightens_it():
    v03 = REPO / "experiments/v0_3/author_kit/launch.py"
    ours = KIT / "launch.py"
    # v0.3's deny roots are kept (as _v03_deny_roots, and they are part of the Phase B list)
    assert _func_src(ours, "_v03_deny_roots") == _func_src(v03, "_deny_roots").replace(
        "def _deny_roots", "def _v03_deny_roots", 1)
    a, b = v03.read_text().split("\n"), ours.read_text().split("\n")
    body_a = a[a.index('"""', 1) + 1:] if '"""' in a[1:] else a
    body_b = b[b.index('"""', 1) + 1:] if '"""' in b[1:] else b
    removed = [x for x in body_a if x not in body_b]
    # every v0.3 line survives except the module path (tools/lib added), the import (more guard names), the deny
    # list call (the Phase B list) and the v0.3 function's def line (renamed _v03_deny_roots)
    assert removed == ["from hn_guard.guards_v03 import TrainingViolation, author_run_guard  # noqa: E402",
                       "def _deny_roots():",
                       "        extra = [PACKET, os.getcwd()]",
                       "        extra = [PACKET, script_dir] + ([parent] if parent.startswith(ROOT + os.sep) else [])",
                       "            io_deny=_deny_roots(),"], removed
    text = ours.read_text()
    for need in ("sys.addaudithook(_phase_b_hook)", "sys.meta_path.insert(0, _RefuseModules())",
                 "torch._C.PyTorchFileWriter = PyTorchFileWriter", "io_deny=deny,", "_tighten()"):
        assert need in text, need


def test_settings_scope_reads_and_writes_to_the_author_dir():
    s = json.loads((KIT / "settings_v05_phase_b.json").read_text())
    assert s["autoMemoryEnabled"] is False and s["permissions"]["defaultMode"] == "dontAsk"
    allow, deny = s["permissions"]["allow"], set(s["permissions"]["deny"])
    assert allow == ["Read(./**)", "Glob(./**)", "Grep(./**)", "Write(./output/v05-phase-b/**)",
                     "Edit(./output/v05-phase-b/**)", "Bash(./py:*)"]
    home = Path.home()
    roots = [REPO, REPO.parent / "HypaNet-m2-author", REPO.parent / "HypaNet-v03-author",
             REPO.parent / "HypaNet-v04-author", Path(f"/private/tmp/claude-{os.getuid()}"),
             Path(f"/tmp/claude-{os.getuid()}"), Path("/private/var/folders"), Path("/var/folders"), home / ".claude"]
    for r in roots:
        for tool in ("Read", "Grep", "Glob", "Edit", "Write"):
            assert f"{tool}(/{r}/**)" in deny, (tool, r)
    for p in ("./packet/**", "./tools/**", "./py", "./.venv/**", "./r1/**", "./disclosure/**"):
        assert f"Edit({p})" in deny and f"Write({p})" in deny
    assert {"WebFetch", "WebSearch"} <= deny
    # every sibling HypaNet* directory that exists now is denied (the prepare script refuses any other)
    for d in REPO.parent.glob("HypaNet*"):
        if d.name != "HypaNet-v05b-author":
            assert f"Read(/{d}/**)" in deny, d


def test_py_wrapper_is_v03s():
    v03 = (REPO / "experiments/v0_3/author_kit/py").read_text()
    assert "-I -B" in v03 and "tools/launch.py" in v03


def test_prompts_stand_alone_and_leak_nothing():
    p = (KIT / "prompt_v05_phase_b.md").read_text()
    for need in ("{R1_SHA256}", "{DISCLOSURE_SHA256}", "REVISION.edits", "CIRCUITS.md", "REVISION.md",
                 "ITERATIONS.md", "V001", "./py -m v05_load check", "r1/PROMPT.txt", "r1/replies/",
                 "no** `reference_policy_class`", "one command per bash call"):
        assert need.lower() in " ".join(p.split()).lower(), need
    for bad in ("v0.3", "v03", "v0.4", "macro-F1 0.", "0.750", "0.330", "Phase A network", "build_weights.py`,",
                "HypaNet", "weight_program", "experiments/"):
        assert bad not in p, bad
    assert p.count("reference_policy_class") == 1
    c = (KIT / "prompt_continue.md").read_text()
    r = (KIT / "prompt_resubmit.md").read_text()
    assert "{PHASE}" in c and all(x in r for x in ("{PHASE}", "{ERROR_TEXT}", "{N}", "{CAP}"))


# -- r1 staging ----------------------------------------------------------------------------------------


def _fake_run(d: Path, replies: dict[str, str], artifact: bytes | None = None) -> Path:
    (d / "replies").mkdir(parents=True)
    units = []
    for i, (name, text) in enumerate(sorted(replies.items()), start=1):
        (d / "replies" / name).write_text(text, encoding="utf-8")
        units.append({"turn": i, "file": name, "sha256": _sha(text.encode()), "cut_off": False})
    (d / "replies" / "INDEX.json").write_text(json.dumps({"units": units}))
    if artifact is None:
        artifact = build_run(d)[0]
    (d / "artifact.dlw").write_bytes(artifact)
    (d / "artifact.sha256").write_text(f"{_sha(artifact)}  artifact.dlw\n")
    return d


R1_REPLIES = {"001.md": "Design: bias towards PRODUCTIVE.\n\n```edits\nhead.bias[0] = 0.5\nhead.bias[2] = -1\n```\n",
              "002.md": "More.\n```edits\nlayers.3.mlp.fc2.bias[0:4] = [1, 2, 3, 4]\n```\nEND OF WEIGHTS\n"}


def test_stage_r1_checks_the_rebuild_and_copies_only_replies(tmp_path):
    run = _fake_run(tmp_path / "r1", R1_REPLIES)
    (run / "run.json").write_text(json.dumps({"status": "finished"}))
    (run / "apply.json").write_text("{}")
    staged = tmp_path / "author" / "r1"
    out = kit_b.stage_r1(run, REPO / "experiments/v0_5/PROMPT.txt", staged)
    assert out["rebuild_identical"] is True and out["status"] == "finished"
    got = sorted(p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file())
    assert got == ["PROMPT.txt", "R1_MANIFEST.json", "artifact.dlw", "artifact.sha256", "replies/001.md",
                   "replies/002.md", "replies/INDEX.json"]
    # the staged copy rebuilds to the same bytes (the run name is r1 in both places)
    assert build_run(staged)[0] == (run / "artifact.dlw").read_bytes()
    bad = _fake_run(tmp_path / "bad" / "r1", R1_REPLIES, artifact=STANDIN.read_bytes())
    (bad / "run.json").write_text(json.dumps({"status": "finished"}))
    with pytest.raises(kit_b.KitError, match="rebuilds r1"):
        kit_b.check_r1_source(bad)
    (bad / "run.json").write_text(json.dumps({"status": "technical_stop"}))
    with pytest.raises(kit_b.KitError, match="not final"):
        kit_b.check_r1_source(bad, rebuild=False)
    (run / "replies" / "notes.txt").write_text("x")
    with pytest.raises(kit_b.KitError, match="unexpected entries"):
        kit_b.stage_r1(run, REPO / "experiments/v0_5/PROMPT.txt", tmp_path / "author_r1b")


# -- disclosure ----------------------------------------------------------------------------------------


def _blocks(n: int = 10) -> list[str]:
    from agent_loop_lab.v03.rawop import RawOp
    from agent_loop_lab.v03.renderer import render_session

    ops = []
    for i in range(n):
        if i % 2 == 0:
            ops.append(RawOp.make(family="read_file", target=f"pkg/m{i}.py", args={"line_start": 1, "line_end": 9},
                                  outcome="SUCCESS", input_tokens=100 + i, output_tokens=5,
                                  timestamp=f"2026-03-14T09:05:{i:02d}.000Z").to_dict())
        else:
            ops.append(RawOp.make(family="run_tests", target="tests", args={"tests": ["t/test_a.py::test_one"]},
                                  outcome="ERROR", error_class="AssertionError", error_location="t/test_a.py",
                                  validation={"suite": "tests", "passed": 3, "total": 5}, input_tokens=50,
                                  output_tokens=9, timestamp=f"2026-03-14T09:06:{i:02d}.000Z").to_dict())
    return render_session(ops, "sess-fake-1")


def _fake_disclosure(d: Path, artifact: Path) -> Path:
    from agent_loop_lab.v03 import scoring, tokenizer
    from agent_loop_lab.v03.artifact import load_artifact
    from agent_loop_lab.v03.infer import eoe_predictions

    blocks = _blocks()
    m, _ = load_artifact(artifact)
    preds = eoe_predictions(m, tokenizer.encode_session(blocks))
    wins = []
    for n, (t, label) in enumerate(((7, "STALLED"), (9, "RECOVERY")), start=1):
        prefix = scoring.render_disclosed_prefix(blocks, t)
        assert kit_b.parse_prefix(prefix) == scoring.parse_disclosed_prefix(prefix)
        wins.append({"id": f"V{n:03d}", "prefix": prefix, "oracle_label": label,
                     "network_class": kit_b.CLASS_NAMES[preds[t]]})
    doc = {"format": "v05-phase-b-disclosure/kit", "class_names": list(kit_b.CLASS_NAMES),
           "window_marker": kit_b.WINDOW_MARKER, "fields": {}, "windows": wins,
           "network": {"run": None, "artifact_sha256": _sha(Path(artifact).read_bytes()), "description": "kit"},
           "selection": "kit", "reference_policy": "none"}
    d.mkdir(parents=True)
    data = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    (d / "DISCLOSURE.json").write_bytes(data)
    (d / "DISCLOSURE.sha256").write_text(f"{_sha(data)}  DISCLOSURE.json\n")
    return d


def test_disclosure_screen(tmp_path):
    d = _fake_disclosure(tmp_path / "disc", STANDIN)
    ok = kit_b.check_disclosure(d / "DISCLOSURE.json", expect=2, artifact=STANDIN)
    assert ok["windows"] == 2
    doc = json.loads((d / "DISCLOSURE.json").read_text())
    doc["windows"][0]["reference_policy_class"] = None
    (tmp_path / "x.json").write_text(json.dumps(doc))
    with pytest.raises(kit_b.KitError, match="keys"):
        kit_b.check_disclosure(tmp_path / "x.json", expect=2)
    doc["windows"][0].pop("reference_policy_class")
    doc["windows"][0]["network_class"] = next(c for c in kit_b.CLASS_NAMES if c != doc["windows"][0]["network_class"])
    (tmp_path / "y.json").write_text(json.dumps(doc))
    with pytest.raises(kit_b.KitError, match="network_class differs"):
        kit_b.check_disclosure(tmp_path / "y.json", expect=2, artifact=STANDIN)
    with pytest.raises(kit_b.KitError, match="expected 200"):
        kit_b.check_disclosure(d / "DISCLOSURE.json")
    doc = json.loads((d / "DISCLOSURE.json").read_text())
    doc["network"]["artifact_sha256"] = "0" * 64
    (tmp_path / "z.json").write_text(json.dumps(doc))
    with pytest.raises(kit_b.KitError, match="header names network artefact"):
        kit_b.check_disclosure(tmp_path / "z.json", expect=2, artifact=STANDIN)
    doc["leak"] = 1
    (tmp_path / "z.json").write_text(json.dumps(doc))
    with pytest.raises(kit_b.KitError, match="unexpected top-level keys"):
        kit_b.check_disclosure(tmp_path / "z.json", expect=2)


# -- audit and cost ------------------------------------------------------------------------------------


def _ev_tool(name, inp, model="claude-fable-5-1", tid="t1"):
    return {"type": "assistant", "message": {"model": model, "content": [
        {"type": "tool_use", "id": tid, "name": name, "input": inp}]}}


def test_audit_phase_b_rules(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("v05b_audit", REPO / "scripts/v05b_audit_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = tmp_path / "author"
    (a / "output/v05-phase-b").mkdir(parents=True)
    events = [
        _ev_tool("Bash", {"command": "./py output/v05-phase-b/scratch/look.py"}, tid="a"),
        _ev_tool("Bash", {"command": "./py -m v05_load check"}, tid="b"),
        _ev_tool("Bash", {"command": "./py -m pip install x"}, tid="c"),
        _ev_tool("Write", {"file_path": "output/v05-phase-b/REVISION.edits"}, tid="d"),
        _ev_tool("Write", {"file_path": "r1/artifact.dlw"}, tid="e"),
        _ev_tool("Read", {"file_path": str(REPO / "README.md")}, tid="f"),
        _ev_tool("Read", {"file_path": "r1/replies/001.md"}, model="claude-opus-5-5", tid="g"),
        {"type": "result", "total_cost_usd": 1.5, "permission_denials": [{"tool_use_id": "f"}]},
    ]
    log = tmp_path / "v05b.stream.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    rep = mod.audit(log, a)
    v = rep["violations"]
    assert any("./py -m pip" in x and x.startswith("[EXECUTED]") for x in v)
    assert any("read-only hand-over" in x for x in v)
    assert any("claude-opus-5-5" in x for x in v)
    assert any(x.startswith("[refused]") and "README.md" in x for x in v)
    assert not any("-m v05_load" in x or "look.py" in x or "REVISION.edits" in x for x in v), v
    assert mod.main(["x", str(log), str(a)]) == 1


def test_cost_by_session_delta_and_cap(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("v05b_cost", REPO / "scripts/v05b_cost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    logs = tmp_path / "logs"
    logs.mkdir()
    res = lambda total: json.dumps({"type": "result", "session_id": "S", "total_cost_usd": total})  # noqa: E731
    (logs / "v05b.stream.jsonl").write_text(res(10.0) + "\n")
    (logs / "v05b.c1.stream.jsonl").write_text(res(25.0) + "\n")
    s = mod.summary(logs)
    assert s["phases"]["v05b"]["total_usd"] == 25.0 and mod.check(s)[0] == 0
    (logs / "v05b.r1.stream.jsonl").write_text(json.dumps({"type": "assistant", "message": {"id": "m", "model": "x"}}))
    (logs / "v05b.r1.launch.json").write_text(json.dumps({"max_budget_usd": 225.0}))
    s = mod.summary(logs)
    assert s["phases"]["v05b"]["total_usd"] == 250.0 and mod.check(s)[0] == 3


# -- the dry hand-over: prepare, ./py, helper, build, launcher check ---------------------------------------


@pytest.fixture(scope="module")
def dry():
    _no_pytest_of()
    if DRY_ROOT.exists():
        for p in DRY_ROOT.rglob("*"):
            if not p.is_symlink():
                p.chmod(0o755 if p.is_dir() else 0o644)
        shutil.rmtree(DRY_ROOT)
    inputs = DRY_ROOT / "inputs"
    run = _fake_run(inputs / "r1", R1_REPLIES, artifact=STANDIN.read_bytes())  # v0.3 A weights as r1 stand-in
    disc = _fake_disclosure(inputs / "disclosure", run / "artifact.dlw")
    author = DRY_ROOT / "HypaNet-v05b-author"
    record = inputs / "HANDOVER.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("V05B_")}
    env.update(V05B_KIT="1", V05B_R1_DIR=str(run), V05B_DISCLOSURE_DIR=str(disc), V05B_RECORD=str(record),
               V05B_NO_REBUILD_CHECK="1", V05B_DISCLOSURE_EXPECT="2", V05B_VENV_FROM=str(REPO / ".venv"))
    p = subprocess.run(["sh", str(REPO / "scripts/v05b_prepare_author_dir.sh"), str(author)], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr
    yield {"author": author, "run": run, "disc": disc, "record": record, "env": env, "inputs": inputs}
    _no_pytest_of()


def _py(author: Path, *args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    return subprocess.run([str(author / "py"), *args], cwd=author, env=env, capture_output=True, text=True,
                          timeout=timeout)


def test_dry_prepare_layout_and_record(dry):
    a = dry["author"]
    rec = json.loads(dry["record"].read_text())
    assert rec["kit"] is True and rec["phase"] == "v05-phase-b" and rec["fallback_model"] is None
    assert rec["packet_sha256"] == _sha((a / "packet/PACKET_MANIFEST.json").read_bytes())
    assert rec["r1_artifact_sha256"] == _sha(STANDIN.read_bytes())
    assert rec["r1_checks"]["rebuild_identical"] is None  # skipped in this dry run (stand-in weights)
    assert sorted(os.listdir(a)) == [".venv", "disclosure", "output", "packet", "py", "r1", "tools"]
    assert os.listdir(a / "output/v05-phase-b") == []
    assert (a / "py").read_bytes() == (REPO / "experiments/v0_3/author_kit/py").read_bytes()
    assert (a / "tools/launch.py").read_bytes() == (KIT / "launch.py").read_bytes()
    for k in ("packet", "tools", "r1", "disclosure", "py"):
        assert kit_b.writable(a / k) == [], k
    assert os.access(a / "output/v05-phase-b", os.W_OK)
    assert sorted(os.listdir(a / "disclosure")) == ["DISCLOSURE.json", "DISCLOSURE.sha256"]
    assert sorted(os.listdir(a / "r1")) == ["PROMPT.txt", "R1_MANIFEST.json", "artifact.dlw", "artifact.sha256",
                                           "replies"]
    assert (a / "r1/PROMPT.txt").read_bytes() == (REPO / "experiments/v0_5/PROMPT.txt").read_bytes()
    assert kit_b.verify(a, rec, r1_dir=dry["run"], disclosure_dir=dry["disc"]) == []


PROBE = r'''
import hashlib, json, sys
import model, v05_load
from hn_v05 import dlw3
r1 = v05_load.load_r1()
rev, report = v05_load.apply_edits_file()
out = {"problems": v05_load.build_problems(report), "r1": [], "rev": [], "labels": []}
for w in v05_load.windows():
    out["r1"].append(model.CLASS_NAMES[v05_load.classify(r1, w["blocks"])])
    out["rev"].append(model.CLASS_NAMES[v05_load.classify(rev, w["blocks"])])
    out["labels"].append(w["network_class"])
act = v05_load.activations(rev, v05_load.windows()[0]["blocks"], query_positions=[0])
out["act_keys"] = sorted(act)
out["rev_sha"] = hashlib.sha256(dlw3.serialize_state_dict(rev.state_dict(), {})).hexdigest()
print(json.dumps(out))
'''

REVISION = "# dry-run revision\nhead.bias[0] = -3\nhead.bias[1] += 0.5\nlayers.3.mlp.fc2.bias[0:4] = [1, 2, 3, 4]\n"


def test_dry_py_helper_runs_the_network_like_the_harness(dry):
    a = dry["author"]
    out = a / "output/v05-phase-b"
    (out / "scratch").mkdir(exist_ok=True)
    (out / "scratch/probe.py").write_text(PROBE)
    (out / "REVISION.edits").write_text(REVISION)
    p = _py(a, "output/v05-phase-b/scratch/probe.py")
    assert p.returncode == 0, p.stderr
    res = json.loads(p.stdout.strip().splitlines()[-1])
    assert res["problems"] == []
    assert res["r1"] == res["labels"]  # the helper reproduces the disclosure's network_class
    assert {"logits", "attn", "eoe_index", "surface"} <= set(res["act_keys"])
    base, _ = deserialize_state_dict(STANDIN.read_bytes())
    sd, _rep = apply_revision(base, REVISION)
    assert res["rev_sha"] == _sha(serialize_state_dict(sd, {}))  # vendored applier == harness applier
    p = _py(a, "-m", "v05_load", "check")
    assert p.returncode == 0 and "would accept" in p.stdout, p.stdout + p.stderr
    (out / "scratch/bad.edits").write_text("head.bias[4] = 1\n")
    p = _py(a, "-m", "v05_load", "check", "output/v05-phase-b/scratch/bad.edits")
    assert p.returncode == 1 and "WOULD REJECT" in p.stdout and "out of range" in p.stdout
    p = _py(a, "-m", "v05_load", "r1")
    assert p.returncode == 1 and "DIFFERENT" in p.stdout  # stand-in weights are not the fake replies' rebuild


REAL_LAYOUT_PROBE = r'''
import json, os, sys
tools, root, repo = sys.argv[1:4]
sys.path.insert(0, tools)
import launch
# the real author venv is private (root/.venv); only the base interpreter lies outside the author dir
path = [p for p in launch._START_PATH
        if not p.startswith(os.path.dirname(tools)) and not os.path.realpath(p).startswith(repo)]
keep = launch._keep_roots(root=root, path=path, prefixes=(sys.base_prefix,))
print(json.dumps({"keep": keep, "deny": launch._deny_roots(root=root, keep=keep), "base": sys.base_prefix}))
'''


def _denied(path, deny):
    p = os.path.realpath(path)
    return any(p == d or p.startswith(d.rstrip("/") + "/") for d in deny)


def test_real_layout_denies_everything_but_the_author_dir_and_the_interpreter(dry):
    # the deny list ./py would use for the real author dir ../HypaNet-v05b-author (computed by the dry launcher,
    # nothing is read): the repo, its siblings, the rest of $HOME and the data-volume firmlink are all denied
    a = dry["author"]
    root = REPO.parent / "HypaNet-v05b-author"
    p = subprocess.run([str(a / ".venv/bin/python"), "-I", "-B", "-c", REAL_LAYOUT_PROBE, str(a / "tools"), str(root),
                        str(REPO)], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    deny, base = out["deny"], out["base"]
    must = [REPO, REPO / "datasets", REPO.parent / "HypaNet-v03-author", REPO.parent.parent / "any-notes.md",
            Path.home() / ".claude", Path("/System/Volumes/Data") / str(REPO).lstrip("/"), Path("/Volumes/x"),
            Path("/usr/local/x"), Path("/private/tmp/x"), Path("/private/var/folders/x")]
    for c in Path.home().iterdir():  # every entry of $HOME except the way to the interpreter
        rc = os.path.realpath(c)
        if not any(k == rc or k.startswith(rc + "/") or rc.startswith(k + "/") for k in out["keep"]):
            must.append(c)
    for m in must:
        assert _denied(m, deny), m
    for ok in (base, os.path.join(base, "lib"), "/usr/lib", "/System/Library/CoreServices/SystemVersion.plist"):
        assert not _denied(ok, deny), ok
    assert str(root) not in deny
    text = (REPO / "scripts/v05b_prepare_author_dir.sh").read_text()
    assert 'AUTHOR_DIR="${1:-$PARENT/HypaNet-v05b-author}"' in text


def _probe(a: Path, name: str, body: str) -> subprocess.CompletedProcess:
    script = a / "output/v05-phase-b/scratch" / f"{name}.py"
    script.parent.mkdir(exist_ok=True)
    script.write_text(body)
    return _py(a, f"output/v05-phase-b/scratch/{name}.py")


def test_dry_py_denies_the_repo_the_firmlink_and_other_dirs(dry):
    a = dry["author"]
    targets = [REPO / "pyproject.toml", Path("/System/Volumes/Data") / str(dry["inputs"] / "disclosure/DISCLOSURE.json").lstrip("/")]
    dev = [c for c in sorted(REPO.parent.parent.iterdir()) if c.name != REPO.parent.name and c.is_file()]
    targets += dev[:1]
    for t in targets:
        if not t.exists():
            continue
        p = _probe(a, "peek_far", f"print(len(open({str(t)!r}, 'rb').read()) > -1)\n")  # prints nothing of t
        assert p.returncode == 3 and "AUTHOR GUARD: refused" in p.stderr, (t, p.returncode, p.stderr[-300:])
    # the kit venv is the repo's .venv, so the repo itself may be listed here, but not what is in it
    p = _probe(a, "peek_far", f"import os\nprint(len(os.listdir({str(REPO / 'experiments')!r})))\n")
    assert p.returncode == 3, p.stderr[-300:]


def test_dry_py_confines_writes_to_output_and_refuses_low_level_file_modules(dry):
    a = dry["author"]
    stray = dry["inputs"] / "x.pt"
    cases = {
        "root_file": (f"open({str(a / 'probe_root.txt')!r}, 'w').write('x')\n", 3),
        "chmod": (f"import os\nos.chmod({str(a / 'disclosure')!r}, 0o755)\n", 3),
        "chmod_fd": (f"import os\nfd = os.open({str(a / 'disclosure/DISCLOSURE.sha256')!r}, os.O_RDONLY)\n"
                     "os.fchmod(fd, 0o644)\n", 3),
        "pth": (f"open({str(a / '.venv/zz_probe.pth')!r}, 'w').write('import os')\n", 3),
        "torch_save": (f"import torch\ntorch.save({{'a': 1}}, {str(stray)!r})\n", 3),
        "sqlite": ("import sqlite3\n", 1),
        "readline": ("import readline\n", 1),
        "dbm": ("import dbm.ndbm\n", 1),
    }
    mode = (a / "disclosure").stat().st_mode
    for name, (body, code) in cases.items():
        p = _probe(a, f"w_{name}", body)
        assert p.returncode == code, (name, p.returncode, p.stderr[-400:])
        if code == 1:
            assert "refused by ./py" in p.stderr, (name, p.stderr[-400:])
    assert not (a / "probe_root.txt").exists() and not stray.exists() and not (a / ".venv/zz_probe.pth").exists()
    assert (a / "disclosure").stat().st_mode == mode
    assert not os.access(a, os.W_OK) and not os.access(a / "output", os.W_OK)
    ok = a / "output/v05-phase-b/scratch/ok.pt"
    sub = ok.parent / "sub"
    p = _probe(a, "w_ok", f"import torch, os, shutil\ntorch.save({{'a': 1}}, {str(ok)!r})\n"
               f"os.makedirs({str(sub / 'deep')!r})\nopen({str(sub / 'deep/f.txt')!r}, 'w').write('x')\n"
               f"os.symlink({str(sub / 'deep')!r}, {str(sub / 'link')!r})\nos.remove({str(sub / 'link')!r})\n"
               f"shutil.copytree({str(sub)!r}, {str(sub) + '2'!r})\nshutil.rmtree({str(sub)!r})\n"
               f"shutil.rmtree({str(sub) + '2'!r})\nos.chmod({str(ok)!r}, 0o600)\n")
    assert p.returncode == 0 and ok.is_file() and not sub.exists(), p.stderr[-400:]


@pytest.mark.parametrize("target", ["sibling", "claude_home", "claude_scratch"])
def test_dry_py_denies_reads_outside_the_author_dir(dry, target):
    a = dry["author"]
    other = DRY_ROOT.parent / "v05b_outside_probe.txt"  # under the Claude scratch deny root, outside the author dir
    other.write_text("outside")
    path = {"sibling": dry["inputs"] / "disclosure/DISCLOSURE.json",
            "claude_home": Path.home() / ".claude" / "settings.json", "claude_scratch": other}[target]
    script = a / "output/v05-phase-b/scratch" / f"peek_{target}.py"
    script.parent.mkdir(exist_ok=True)
    script.write_text(f"print(open({str(path)!r}).read()[:20])\n")
    p = _py(a, f"output/v05-phase-b/scratch/peek_{target}.py")
    assert p.returncode == 3 and "AUTHOR GUARD: refused" in p.stderr, (p.returncode, p.stdout, p.stderr)
    script.write_text("import os\nprint(os.listdir('..'))\n")
    p = _py(a, f"output/v05-phase-b/scratch/peek_{target}.py")
    assert p.returncode == 3 and "AUTHOR GUARD: refused" in p.stderr


def test_dry_py_refuses_writes_outside_output_and_inline_code(dry):
    a = dry["author"]
    script = a / "output/v05-phase-b/scratch/w.py"
    script.write_text(f"open({str(dry['inputs'] / 'evil.txt')!r}, 'w').write('x')\n")
    p = _py(a, "output/v05-phase-b/scratch/w.py")
    assert p.returncode == 3 and not (dry["inputs"] / "evil.txt").exists()
    assert _py(a, "-c", "print(1)").returncode == 2
    script.write_text("import random\nrandom.random()\n")
    assert _py(a, "output/v05-phase-b/scratch/w.py").returncode == 3


def test_dry_launcher_check_mode(dry):
    env = dict(dry["env"], V05B_LOGDIR=str(dry["inputs"] / "logs"))
    for k in ("V05B_NO_REBUILD_CHECK", "V05B_DISCLOSURE_EXPECT", "V05B_VENV_FROM"):
        env.pop(k)
    p = subprocess.run(["sh", str(REPO / "scripts/v05b_launch.sh"), "check", str(dry["author"])], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0 and "check passed" in p.stdout and "fallback=none" in p.stdout, p.stdout + p.stderr
    assert not (dry["inputs"] / "logs").exists()  # check writes nothing
    p = subprocess.run(["sh", str(REPO / "scripts/v05b_launch.sh"), "phase-b", str(dry["author"])], cwd=REPO,
                       env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 2 and "kit mode" in p.stderr
    p = subprocess.run(["sh", str(REPO / "scripts/v05b_launch.sh"), "continue", "12", str(dry["author"])], cwd=REPO,
                       env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 2 and "K must be" in p.stderr


def _build(dry, out_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts/v05b_run_build.py"), "--author-dir", str(dry["author"]),
                           "--r1-dir", str(dry["run"]), "--out-dir", str(out_dir), *extra], cwd=REPO,
                          capture_output=True, text=True, timeout=600)


def test_dry_build_rejects_then_builds(dry):
    a = dry["author"]
    out = a / "output/v05-phase-b"
    arch = dry["inputs"] / "weight_program_v0_5_phase-b"
    (out / "REVISION.edits").write_text(REVISION + "head.bias[9] = 1\n")
    p = _build(dry, arch)
    assert p.returncode == 1 and "missing required file(s)" in p.stderr
    for f, t in (("CIRCUITS.md", "# circuits\n"), ("REVISION.md", "# revision\n"), ("ITERATIONS.md", "# it\n")):
        (out / f).write_text(t)
    p = _build(dry, arch)
    err = (arch / "attempts/0002.error.txt").read_text()
    assert p.returncode == 1 and "line 5: index 9 out of range" in err and len(err) <= 2001
    (out / "REVISION.edits").write_text("".join(f"head.bias[{i % 9}] = 1\n" for i in range(400)))
    assert _build(dry, arch).returncode == 1
    assert len((arch / "attempts/0003.error.txt").read_text()) <= 2001
    (out / "REVISION.edits").write_text(REVISION)
    p = _build(dry, arch)
    assert p.returncode == 0, p.stderr
    data = (arch / "artifact.dlw").read_bytes()
    assert (arch / "artifact.sha256").read_text().split()[0] == _sha(data)
    sd, meta = deserialize_state_dict(data)
    base, _ = deserialize_state_dict(STANDIN.read_bytes())
    want, _ = apply_revision(base, REVISION)
    assert all(torch.equal(sd[k], want[k]) for k in sd)
    assert meta["base_artifact_sha256"] == _sha(STANDIN.read_bytes()) and meta["phase"] == "v05-phase-b"
    rec = json.loads((arch / "BUILD_RECORD.json").read_text())
    _sd, want_rep = apply_revision(base, REVISION)
    assert rec["changed_entries"] == want_rep["changed_entries"] and rec["statements"] == 3
    assert (arch / "author_output/scratch/probe.py").is_file() and (arch / "REVISION.edits").read_text() == REVISION
    apply = json.loads((arch / "apply.json").read_text())
    assert apply["artifact"]["sha256"] == _sha(data) and apply["edit_report"]["skipped"] == 0
    assert json.loads((arch / "attempts/0004.json").read_text())["status"] == "built"
    p = _build(dry, arch)
    assert p.returncode != 0 and "first valid build" in p.stderr


def test_dry_verify_detects_tampering(dry):
    a = dry["author"]
    rec = json.loads(dry["record"].read_text())
    f = a / "tools/lib/v05_load.py"
    f.chmod(0o644)
    try:
        assert any("tools is writable" in x for x in kit_b.verify(a, rec))
        f.write_text(f.read_text() + "\n# tampered\n")
        assert any(x.startswith("tools/ differs") for x in kit_b.verify(a, rec))
    finally:
        f.write_text(f.read_text().replace("\n# tampered\n", ""))
        f.chmod(0o444)
    assert kit_b.verify(a, rec) == []
    # a new entry next to the hand-over (e.g. a script that made the author dir writable again)
    a.chmod(0o755)
    try:
        (a / "stray.txt").write_text("x")
        probs = kit_b.verify(a, rec)
        assert any("author dir entries" in x for x in probs) and any("author dir is writable" in x for x in probs)
    finally:
        (a / "stray.txt").unlink()
        a.chmod(0o555)
    assert kit_b.verify(a, rec) == []
    # the record is never replaced silently
    p = subprocess.run([sys.executable, "-m", "agent_loop_lab.v05.kit_b", "record", str(a), str(dry["record"]), "--kit"],
                       cwd=REPO, capture_output=True, text=True, timeout=300, env=dict(os.environ, PYTHONPATH=str(REPO / "src")))
    assert p.returncode == 2 and "never replaced silently" in p.stderr, p.stderr


def test_venv_state_sees_startup_files(tmp_path):
    a = tmp_path / "author"
    sp = a / ".venv/lib/python3.12/site-packages"
    sp.mkdir(parents=True)
    (sp / "torch.py").write_text("x = 1\n")
    (a / ".venv/pyvenv.cfg").write_text("home = /x\n")
    before = kit_b.venv_state(a)
    assert set(before["startup_files"]) == {"pyvenv.cfg"}
    (sp / "zz.pth").write_text("import os\n")
    after = kit_b.venv_state(a)
    assert after != before and "lib/python3.12/site-packages/zz.pth" in after["startup_files"]
    (sp / "zz.pth").unlink()
    (sp / "torch.py").write_text("x = 2\n")  # same size; mtime differs
    os.utime(sp / "torch.py", ns=(1, 1))
    assert kit_b.venv_state(a)["inventory_sha256"] != before["inventory_sha256"]


def test_audit_phase_b_source_rules(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("v05b_audit", REPO / "scripts/v05b_audit_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = tmp_path / "author"
    s = a / "output/v05-phase-b/scratch"
    s.mkdir(parents=True)
    (s / "ok.py").write_text("import model, v05_load\nprint(v05_load.load_r1())\n")
    (s / "a.py").write_text("import os\nos.chmod('x', 0o755)\n")
    (s / "b.py").write_text("open('../../../.venv/lib/python3.12/site-packages/z.pth', 'w')\n")
    (s / "c.py").write_text("import sqlite3\n")
    (s / "d.py").write_text("import torch\nw = torch._C.PyTorchFileWriter\n")
    got = mod.source_findings(a)
    assert sorted({g.split(":")[0] for g in got}) == [f"output/v05-phase-b/scratch/{n}.py" for n in "abcd"], got


def test_prepare_and_launcher_guards_in_text():
    prep = (REPO / "scripts/v05b_prepare_author_dir.sh").read_text()
    for need in ('[ -e "$RECORD" ] && die', "phase_b/SESSION_ID\" ] && die", "logs/v05b*.stream.jsonl",
                 'chmod -R a-w "$AUTHOR_DIR/.venv"', 'chmod 555 "$AUTHOR_DIR/output" "$AUTHOR_DIR"'):
        assert need in prep, need
    launch = (REPO / "scripts/v05b_launch.sh").read_text()
    assert "wc -c" not in launch and '[ "$ERR_CHARS" -le 2001 ]' in launch
    assert 'mv "$RESUB/$PHASE.r$N.error.txt" "$RESUB/$ATTEMPT.error.txt"' in launch
    assert 'mv "$SID_FILE" "$LOGDIR/$ATTEMPT.SESSION_ID"' in launch
    # a non-ASCII runner error of 2001 characters passes the character count the launcher uses
    from importlib import util

    spec = util.spec_from_file_location("v05b_run_build", REPO / "scripts/v05b_run_build.py")
    rb = util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    err = rb.cap_error("line 1: unexpected character '×': head.bias[0] = 1 × 2 # ≈ wᵢ·xᵢ\n" * 60) + "\n"
    assert len(err) <= 2001 < len(err.encode("utf-8"))
