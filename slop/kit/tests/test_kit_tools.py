"""The prepared author directory and its tools, as the author runs them (SPEC §13).

Builds one TEST hand-over with scripts/prepare_author_dir.sh --test-root (settings and record go to the test root, not
the repo), then runs ./tok, ./lint, ./score, ./inspect and ./python3 through their wrappers from the author dir, on
dev screens, on a probe screen, on the pause-menu source, with a tiny hand-made artefact (built by ./python3 from
kit/lib) and the harness's floor network (validate/build_floor_net.py) as stand-ins. Also the leakage check, on the
kit and on a tampered copy. Nothing is launched.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SLOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SLOP / "kit"))
import kit_build as kb  # noqa: E402
import leakage  # noqa: E402

READY = (all((kb.DEV_DIR / f).is_file() for f in ("screens.jsonl", "dev.gen.jsonl", "dev.meta.jsonl"))
         and (kb.REF_DIR / "manifest.json").is_file() and shutil.which("node") and shutil.which("uv"))
pytestmark = pytest.mark.skipif(not READY, reason="needs the dev and reference sets, node and uv")
PAUSE_ID = "4649cb8941e87ac2"

BUILD_PY = '''import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK.parent / "kit" / "lib"))
import model as mdl
import tokenizer as tk

T = mdl.zero_tensors()
T["role_emb.weight"][tk.ROLE_IDS["eyebrow"], 0] = 1.0   # an eyebrow writes dim 0
T["head.weight"][1, 0] = 2.0                             # dim 0 votes ECHO
T["head.bias"][0] = 1.0                                  # everything else stays KEEP
T["layers.3.attn.rel_bias"][0, 64] = 3.0                 # the pointer head looks one node ahead
print(mdl.save_artifact(WORK / "tiny", T, ["paus", "resum", "hold"], (4, 0)))
'''

# the same network in id layout 2 (SPEC.md §6), with more stems than layout 1 allows and a hashed stem's row in use
BUILD2_PY = '''import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK.parent / "kit" / "lib"))
import model as mdl
import tokenizer as tk

T = mdl.zero_tensors(2)
assert T["tok_emb.weight"].shape == (tk.LAYOUTS[2].vocab_size, 128) == (25088, 128)
T["role_emb.weight"][tk.ROLE_IDS["eyebrow"], 0] = 1.0
T["head.weight"][1, 0] = 2.0
T["head.bias"][0] = 1.0
T["layers.3.attn.rel_bias"][0, 64] = 3.0
T["tok_emb.weight"][tk.hash_bucket("oper", 2), 5] = 1.0     # a hashed stem's row ("OPERATION")
T["tok_emb.weight"][16000 + 512, 1] = 1.0                    # line 16,001 of vocab.txt
stems = ["paus", "resum", "hold"] + [f"x{i}" for i in range(15997)] + ["recaptur"]
print(mdl.save_artifact(WORK / "tiny2", T, stems, (4, 0)))
'''


def chmod_writable(p: Path) -> None:
    for root, dirs, files in os.walk(p):
        os.chmod(root, 0o755)
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                os.chmod(fp, 0o644 | (os.stat(fp).st_mode & 0o111))


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    base = tmp_path_factory.mktemp("slopkit")
    root = base / "root"
    root.mkdir()
    # a smoke-named dir: it gets r1's kit, and prepare_author_dir.sh refuses r1's own name once runs/r1/SESSION_ID
    # exists (the r1 session was launched); runs/smoke has never been launched
    author = base / "HypaNet-slop-smoke-kittest"
    r = subprocess.run(["sh", str(SLOP / "scripts" / "prepare_author_dir.sh"), "--test-root", str(root), str(author)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    from validate import build_floor_net
    build_floor_net.build(root / "floor")
    yield author.resolve(), root.resolve()
    chmod_writable(base)


def tool(author: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    return subprocess.run([f"./{args[0]}", *args[1:]], cwd=author, capture_output=True, text=True, env=env)


def ok(author: Path, *args: str) -> str:
    r = tool(author, *args)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(r.stdout) <= 20_000
    return r.stdout


def test_handover_is_verified_read_only_and_leak_free(kit):
    author, root = kit
    record = root / "HANDOVER.record.json"
    rec = json.loads(record.read_text())
    assert rec["test"] is True and rec["kind"] == "hypanet-slop-author"
    assert rec["examples"]["selection"]["n"] == 400
    r = subprocess.run([sys.executable, str(SLOP / "kit" / "kit_build.py"), "verify", str(author), str(record),
                        "--pristine"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
    assert not os.access(author, os.W_OK) and not os.access(author / "examples" / "INDEX.txt", os.W_OK)
    assert os.access(author / "work", os.W_OK)
    assert (author / "examples" / "reference" / f"{PAUSE_ID}.source.tsx").is_file()
    assert len(list((author / "examples" / "dev").glob("*.screen.json"))) == 400
    index = (author / "examples" / "INDEX.txt").read_text()
    assert "clean" not in index and index.count("\ndev ") == 400
    assert leakage.run(author).problems == []
    assert json.loads((root / "settings.json").read_text())["permissions"]["defaultMode"] == "dontAsk"
    # the author's copies: the guide without its internal preamble, the record without harness and data hashes
    assert (author / "LABELLING.md").read_text(encoding="utf-8") == kb.kit_guide()
    assert rec["examples"]["kit_guide_sha256"] != rec["examples"]["guide_sha256"]
    mine = json.loads((author / "HANDOVER.json").read_text())
    assert set(mine) == set(kb.AUTHOR_RECORD_KEYS) and mine["files"] == rec["files"] and "harness" in rec
    assert rec["node"]["version"].startswith("v")
    readme = (author / "examples" / "README.md").read_text()
    assert "SPEC" not in readme and "test_gen" not in readme and "codex" not in readme.lower()


def test_python3_guard_and_hand_made_artefact(kit):
    author, _ = kit
    (author / "work" / "build.py").write_text(BUILD_PY)
    out = ok(author, "python3", "work/build.py")
    assert len(out.strip()) == 64
    for code, msg in (("import numpy as np; np.linalg.lstsq(np.eye(2), np.ones(2))", "fitting routine"),
                      ("open('examples/INDEX.txt', 'a')", "stay in work/"),
                      ("import torch", "import of torch refused")):
        r = tool(author, "python3", "-c", code)
        assert r.returncode != 0 and msg in (r.stdout + r.stderr), code


def test_id_layout_2_artefact_through_the_tools(kit):
    author, _ = kit
    (author / "work" / "build2.py").write_text(BUILD2_PY)
    assert len(ok(author, "python3", "work/build2.py").strip()) == 64
    meta = json.loads((author / "work" / "tiny2" / "meta.json").read_text())
    assert meta == {"format": "slop.artifact/1", "pointer_head": [4, 0], "ids": 2}
    src = f"examples/reference/{PAUSE_ID}.source.tsx"
    out = ok(author, "lint", src, "--art", "work/tiny2", "--emit", "work/probes/pause2.json")
    assert 'ECHO     eyebrow "OPERATION ON HOLD" restates h1 "PAUSED"' in out
    out = ok(author, "score", "work/tiny2", "--files", "work/probes/pause2.json")
    assert "ECHO" in out
    out = ok(author, "tok", "--text", "Resume recaptures operation", "--vocab-file", "work/tiny2/vocab.txt")
    assert "vocab line 16001" in out and "id 16512" in out and "hash bucket" in out
    out = ok(author, "inspect", "work/tiny2", "work/probes/pause2.json", "--node", "0")
    assert "call ECHO" in out


def test_tok(kit):
    author, _ = kit
    out = ok(author, "tok", "--text", "Resume recaptures the mouse.")
    assert "stem 'resum'" in out and "hash bucket" in out and "EON id 2" in out
    out = ok(author, "tok", f"examples/reference/{PAUSE_ID}.source.tsx")
    assert "nodes 7" in out and "windows 1" in out
    dev = sorted((author / "examples" / "dev").glob("*.screen.json"))[0]
    assert "window 0:" in ok(author, "tok", str(dev.relative_to(author)))
    out = ok(author, "tok", f"examples/reference/{PAUSE_ID}.screen.json", "--node", "1")
    assert "(i=1)" in out and "(i=0)" not in out and "'PAUSED'" in out and "window 0:" in out
    assert tool(author, "tok", "--text", "x", "--node", "1").returncode == 2


def test_lint_score_inspect(kit):
    author, root = kit
    if not (author / "work" / "tiny" / "artifact.npz").is_file():
        (author / "work" / "build.py").write_text(BUILD_PY)
        ok(author, "python3", "work/build.py")
    src = f"examples/reference/{PAUSE_ID}.source.tsx"
    out = ok(author, "lint", src, "--art", "work/tiny", "--emit", "work/probes/pause.json")
    assert 'ECHO     eyebrow "OPERATION ON HOLD" restates h1 "PAUSED"' in out and "slop/redundant-copy" in out
    assert "network skipped" in ok(author, "lint", src)  # no work/artifact.npz
    js = json.loads(ok(author, "lint", src, "--art", str(root / "floor"), "--format", "json"))
    assert len(js["preds"]) == 7 and js["network"]
    # a probe screen, unlabelled then labelled
    out = ok(author, "score", "work/tiny", "--files", "work/probes/pause.json")
    assert "unlabelled" in out and "ECHO" in out and "bos" in out
    (author / "work" / "probes" / "pause.labels.jsonl").write_text(
        "".join(json.dumps({"screen": "p", "loc": n["loc"], "class": "ECHO" if n["i"] == 0 else "KEEP",
                            "target": json.loads((author / "work/probes/pause.json").read_text())["nodes"][1]["loc"]
                            if n["i"] == 0 else None, "pattern": "R1" if n["i"] == 0 else None}) + "\n"
                for n in json.loads((author / "work/probes/pause.json").read_text())["nodes"]))
    out = ok(author, "score", "work/tiny", "--files", "work/probes/pause.json")
    assert "F1 1.000" in out and "pointer accuracy: 1/1" in out
    # the disclosed dev set
    out = ok(author, "score", str(root / "floor"), "--set", "dev")
    assert "400 screens" in out and "per pattern" in out and "per K rule" in out and "pointer accuracy" in out
    assert "wrong nodes:" in out
    out = ok(author, "score", str(root / "floor"), "--set", "dev", "--pattern", "R1", "--offset", "5")
    assert "matching --pattern R1" in out and "showing 6-" in out
    out = ok(author, "score", "work/tiny", "--set", "dev", "--against", str(root / "floor"))
    assert "nodes whose class or pointer differs" in out
    if not kb.REF_LABELS.is_file():
        r = tool(author, "score", "work/tiny", "--set", "reference")
        assert r.returncode == 2 and "unlabelled" in r.stdout
    # inspect
    out = ok(author, "inspect", "work/tiny", f"examples/reference/{PAUSE_ID}.screen.json", "--node", "0", "--layer",
             "4", "--head", "0", "--units", "2")
    assert "call ECHO" in out and 'pointer → 1 h1 "PAUSED"' in out and "L4 h0" in out and "L1 h0" not in out
    (author / "work" / "directions.json").write_text(json.dumps({"eyebrow": [0], "mix": {"1": 0.5}}))
    out = ok(author, "inspect", "work/tiny", "work/probes/pause.json", "--node", "0")
    assert "embed role     eyebrow +1.000  mix +0.000" in out and "logits KEEP +1.0000 ECHO +2.0000" in out
    out = ok(author, "inspect", str(root / "floor"), sorted((author / "examples/dev").glob("*.screen.json"))[0]
             .relative_to(author).as_posix(), "--node", "1", "--tokens", "--units", "40")
    assert out.rstrip().endswith("filters (see --help)]") and "output truncated at 20,000 characters" in out  # no --offset


def test_refusals(kit):
    author, _ = kit
    for args in (("lint", "/etc/hosts"), ("score", "..", "--set", "dev"), ("tok", "--words", "../x.txt"),
                 ("inspect", "work", "/etc/hosts", "--node", "0"),
                 ("lint", f"examples/reference/{PAUSE_ID}.screen.json", "--emit", "examples/x.json")):
        r = tool(author, *args)
        assert r.returncode == 2 and "refusing" in r.stdout, args
    r = tool(author, "tok", "--text", "x", "--cap", "999999")
    assert r.returncode == 2 and "unrecognized arguments" in r.stdout


def test_malformed_probe_gives_one_line_not_a_traceback(kit):
    author, _ = kit
    if not (author / "work" / "tiny" / "artifact.npz").is_file():
        (author / "work" / "build.py").write_text(BUILD_PY)
        ok(author, "python3", "work/build.py")
    (author / "work" / "probes").mkdir(exist_ok=True)
    bad = {"schema": "slop.screen/1", "id": "p", "nodes": [{"i": 0, "k": 0, "role": "h1", "text": "Hi"}]}
    (author / "work" / "probes" / "bad.json").write_text(json.dumps(bad))
    for args in (("score", "work/tiny", "--files", "work/probes/bad.json"), ("lint", "work/probes/bad.json"),
                 ("inspect", "work/tiny", "work/probes/bad.json", "--node", "0"), ("tok", "work/probes/bad.json")):
        r = tool(author, *args)
        assert r.returncode == 2 and '"loc" must be' in r.stdout and "Traceback" not in r.stdout + r.stderr, args


def test_verify_after_work_and_on_tampering(kit):
    author, root = kit
    record = root / "HANDOVER.record.json"
    run = lambda *a: subprocess.run([sys.executable, str(SLOP / "kit" / "kit_build.py"), "verify", str(author),  # noqa
                                     str(record), *a], capture_output=True, text=True)
    (author / "work" / "note.txt").write_text("x")
    assert run().returncode == 0
    r = run("--pristine")
    assert r.returncode == 1 and "work/ is not empty" in r.stdout


def test_leakage_finds_planted_problems(kit, tmp_path):
    author, _ = kit
    copy = tmp_path / "copy"
    shutil.copytree(author, copy, ignore=shutil.ignore_patterns(".venv", "work"))
    chmod_writable(copy)
    templates, _, _, _ = leakage.generator_parts()
    tpl = sorted(templates)[0]
    with open(copy / "examples" / "README.md", "a", encoding="utf-8") as f:
        f.write(f"\n{tpl}\n")
    (copy / "examples" / "extra.txt").write_text("x")
    lab = sorted((copy / "examples" / "dev").glob("*.labels.jsonl"))[0]
    lab.write_text(lab.read_text().replace('"conf"', '"clean": true, "conf"', 1))
    (copy / "KIT.md").chmod(0o644)
    with open(copy / "KIT.md", "a", encoding="utf-8") as f:
        f.write("\nThe ood_codex set is scored too.\n")
    checks = {(p["check"], p["file"]) for p in leakage.run(copy).problems}
    assert ("internal-names", "KIT.md") in checks
    assert ("generator-template", "examples/README.md") in checks
    assert ("files", "examples/extra.txt") in checks
    assert ("files", str(lab.relative_to(copy))) in checks


def test_leakage_fails_closed_without_the_generator(kit, monkeypatch, tmp_path):
    author, _ = kit
    monkeypatch.setattr(leakage, "GENERATOR", tmp_path / "no-generator")
    probs = leakage.run(author).problems
    assert any(p["check"] == "generator-code" and p["match"] == "generator sources not found" for p in probs)


def test_leakage_require_sealed_and_its_waiver(kit, monkeypatch, tmp_path):
    author, _ = kit
    if leakage.SEALED_NGRAMS.is_file():
        pytest.skip("the sealed n-grams exist")
    none = tmp_path / "no-waivers.json"
    none.write_text("[]")
    monkeypatch.setattr(leakage, "WAIVERS", none)  # the repo's own waiver list waives exactly this
    assert any(p["check"] == "sealed-ngrams" for p in leakage.run(author, require_sealed=True).problems)
    w = tmp_path / "waivers.json"
    w.write_text(json.dumps([{"check": "sealed-ngrams", "file": "-", "match": "not-run", "reason": "test"}]))
    monkeypatch.setattr(leakage, "WAIVERS", w)
    rep = leakage.run(author, require_sealed=True)
    assert rep.problems == [] and rep.waived[0]["reason"] == "test"


def test_cap_cuts_at_a_line_end():
    sys.path.insert(0, str(SLOP / "kit" / "tools"))
    import author_tool
    text = "".join(f"line {i:05d} " + "x" * 80 + "\n" for i in range(400))
    out = author_tool.cap(text)
    assert len(out) <= author_tool.CAP and out.endswith("--offset]\n")
    assert out.splitlines()[-2].startswith("line ") and len(out.splitlines()[-2]) == len("line 00000 ") + 80
