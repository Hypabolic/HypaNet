"""The round-2 author kit (kit_build.RUNS["r2"]): one TEST hand-over of an r2 dir made with
scripts/prepare_author_dir.sh --test-root, then its record, seed, resources, tools and leakage check. Nothing is
launched; no held-out file is read by these tests (leakage's held-out sets are replaced by a synthetic one where a
string is planted)."""

from __future__ import annotations

import json
import os
import re
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
         and (kb.REF_DIR / "manifest.json").is_file() and shutil.which("node") and shutil.which("uv")
         and (kb.R1_WORK / "build.py").is_file() and not kb.resource_problems())
pytestmark = pytest.mark.skipif(not READY, reason="needs the dev and reference sets, runs/r1/work, the built "
                                                   "thesaurus, node and uv")


def chmod_writable(p: Path) -> None:
    for root, dirs, files in os.walk(p):
        os.chmod(root, 0o755)
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                os.chmod(fp, 0o644 | (os.stat(fp).st_mode & 0o111))


@pytest.fixture(scope="module")
def r2kit(tmp_path_factory):
    base = tmp_path_factory.mktemp("slopr2")
    root = base / "root"
    root.mkdir()
    author = base / "HypaNet-slop-author-r2-kittest"
    r = subprocess.run(["sh", str(SLOP / "scripts" / "prepare_author_dir.sh"), "--test-root", str(root), str(author)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    yield author.resolve(), root.resolve()
    chmod_writable(base)


def tool(author: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    return subprocess.run([f"./{args[0]}", *args[1:]], cwd=author, capture_output=True, text=True, env=env)


def ok(author: Path, *args: str) -> str:
    r = tool(author, *args)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def verify(author: Path, root: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SLOP / "kit" / "kit_build.py"), "verify", str(author),
                           str(root / "HANDOVER.record.json"), *flags], capture_output=True, text=True)


def test_record_selection_and_settings(r2kit):
    author, root = r2kit
    rec = json.loads((root / "HANDOVER.record.json").read_text())
    assert rec["run"] == "r2" and rec["test"] is True
    sel = rec["examples"]["selection"]
    assert sel["n"] == 1200 and sel["added_to_r1"] == 800
    r1 = SLOP / "runs" / "r1" / "HANDOVER.json"
    if r1.is_file():
        assert sel["r1_ids_sha256"] == json.loads(r1.read_text())["examples"]["dev_ids_sha256"]
    assert len(list((author / "examples" / "dev").glob("*.screen.json"))) == 1200
    assert (author / "PROMPT.md").read_bytes() == (kb.KIT / "prompt_r2.md").read_bytes()
    assert set(rec["work_seed"]) == set(kb.work_seed_files()) and "build.py" in rec["work_seed"]
    assert not any(k.startswith("scratch") for k in rec["work_seed"])
    mine = json.loads((author / "HANDOVER.json").read_text())
    assert set(mine) == set(kb.AUTHOR_RECORD_KEYS)  # the seed hashes and the run stay harness-side
    s = json.loads((root / "settings.json").read_text())
    deny = s["permissions"]["deny"]
    assert "Edit(./resources/**)" in deny and "Write(./resources/**)" in deny
    assert any(r == f"Read(/{kb.SLOP.parent.parent / 'HypaNet-slop-author'}/**)" for r in deny)  # the r1 dir


def test_resources_are_read_only_copies(r2kit):
    author, _ = r2kit
    files = kb.resource_files()
    for rel, src in files.items():
        p = author / rel
        assert p.is_file() and not os.access(p, os.W_OK), rel
        if isinstance(src, Path):
            assert kb.sha256(p) == kb.sha256(src), rel
    readme = (author / "resources/thesaurus/README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Thesaurus\n") and "## Coverage" not in readme and "ui/" not in readme
    assert "datasets" not in readme and "gitignored" not in readme
    attribution = (author / "resources/thesaurus/ATTRIBUTION.md").read_text(encoding="utf-8")
    assert attribution == kb.thesaurus_attribution() and "CC BY 4.0" in attribution and "public domain" in attribution
    for word in ("fetch.py", "build.py", "raw/", "gitignored"):  # the repo's build scripts are not in the kit
        assert word not in attribution, word
    assert not os.access(author / "resources", os.W_OK)


def test_seed_is_writable_and_pristine_means_the_seed(r2kit):
    author, root = r2kit
    assert verify(author, root, "--pristine").returncode == 0
    assert os.access(author / "work" / "build.py", os.W_OK) and os.access(author / "work" / "lexicon", os.W_OK)
    # r1's build.py rebuilds the same artefact from the seed (so the seed stays pristine)
    out = ok(author, "python3", "work/build.py")
    assert "e3e0b1d08f24" in out
    assert verify(author, root, "--pristine").returncode == 0
    note = author / "work" / "probes" / "new.txt"
    note.write_text("x")
    r = verify(author, root, "--pristine")
    assert r.returncode == 1 and "a file the seed does not: probes/new.txt" in r.stdout
    assert verify(author, root).returncode == 0  # a continuation does not need a pristine dir
    note.unlink()
    md = author / "work" / "MODEL.md"
    old = md.read_bytes()
    md.write_bytes(old + b"\n")
    r = verify(author, root, "--pristine")
    assert r.returncode == 1 and "seed file changed: MODEL.md" in r.stdout
    md.write_bytes(old)
    assert verify(author, root, "--pristine").returncode == 0


def test_tools_on_the_r2_kit(r2kit):
    author, _ = r2kit
    out = ok(author, "score", "work", "--set", "dev")
    assert "1200 screens" in out and "artefact e3e0b1d08f24" in out
    out = ok(author, "tok", "--ids", "2", "--text", "Delete album")
    assert "id layout 2" in out and "hash bucket" in out
    out = ok(author, "tok", "--text", "Delete album")  # this kit asks for layout 2, so ./tok defaults to it
    assert out.splitlines()[0].endswith("id layout 2 (this kit's default; --ids 1 for layout 1)")
    out = ok(author, "tok", "--vocab", "work/vocab.txt")  # r1's vocabulary: the count survives, nothing is cut
    assert out.splitlines()[1] == "0 problem(s)" and "output truncated" not in out
    assert re.search(r"^merges \(.*\): \d+; showing 1-50 \(--offset N", out, re.M)
    assert tool(author, "tok", "--text", "x", "--offset", "5").returncode == 2  # --offset goes with --vocab
    code = ("f = open('resources/thesaurus/out/stems.tsv', encoding='utf-8'); f.readline(); "
            "print(sum(1 for _ in f))")
    assert int(ok(author, "python3", "-c", code).strip()) > 60000
    r = tool(author, "python3", "-c", "open('resources/codex/CODEX.md', 'a')")
    assert r.returncode != 0 and "stay in work/" in r.stdout + r.stderr


def test_leakage_passes_and_catches_a_planted_heldout_string(r2kit, tmp_path, monkeypatch):
    author, _ = r2kit
    assert leakage.run(author, run_name="r2").problems == []
    copy = tmp_path / "HypaNet-slop-author-r2-copy"
    shutil.copytree(author, copy, ignore=shutil.ignore_patterns(".venv", "work"))
    chmod_writable(copy)
    planted = "Quietly planted heldout sentence for the test"
    with open(copy / "resources/thesaurus/out/moby.tsv", "a", encoding="utf-8") as f:
        f.write(f"zz\tzz\t1\tsomething|{planted.lower()}|else\n")
    (copy / "resources/codex/lists/extra.txt").write_text("# Provenance: none\nx\n")
    fake = {"val": [{"id": "ffffffffffffffff", "nodes": [{"text": planted}, {"text": "Two words"}]}]}
    monkeypatch.setattr(leakage, "heldout_screens", lambda: fake)
    probs = leakage.run(copy, run_name="r2").problems
    checks = {(p["check"], p["file"]) for p in probs}
    assert ("heldout-text", "resources/thesaurus/out/moby.tsv") in checks
    assert ("files", "resources/codex/lists/extra.txt") in checks
    # cells of a table never join into a match across a tab or "|"
    rep = leakage.Report([])
    idx, firsts, _ = leakage.heldout_index(fake, set(), "")
    cells = leakage.norm(leakage.TSV_SEP.sub(" \x00sep\x00 ", "quietly planted|heldout sentence for the test"))
    assert leakage.word_hits(cells.replace("\x00sep\x00", " zzcellzz "), idx, firsts) == set()
    assert rep.problems == []


def test_internal_names_codex_is_case_sensitive(tmp_path):
    pat = leakage.CASE_SENSITIVE_NAMES["Codex"]
    assert pat.search("labelled by Codex.") and not pat.search("resources/codex/CODEX.md") \
        and not pat.search("the codex lists")


# The invented examples of resources/r1/RESULTS.md (kit/r1_results.md), as screens: each must get the wrong call the
# results say round 1 makes. (id, [(role, text, cluster)], node i, round 1's call)
R1_FALSE_ALARMS = [
    ("fa_consequence", [("dialog_title", "Remove this list?", 0), ("dialog_body", "All items in this list will also be "
                        "removed.", 0), ("button", "Remove", 1), ("button", "Cancel", 1)], 1, "ECHO"),
    ("fa_instruction", [("h2", "Playlists", 0), ("list_item", "Road trip", 0), ("empty_body", "Pick a playlist.", 1)],
     2, "ECHO"),
    ("fa_help", [("label", "Public profile", 0), ("helper", "Your profile is public, so anyone can find it.", 0)], 1,
     "ECHO"),
]
R1_MISSES = [
    ("m_library", [("h1", "Library", 0), ("subtitle", "Everything you've saved, in one spot.", 0)], 1, "KEEP"),
    ("m_eyebrow", [("eyebrow", "THE LOWDOWN", 0), ("h2", "Overview", 0)], 0, "KEEP"),
    ("m_empty", [("empty_title", "No messages", 0), ("empty_body", "Your inbox is clear.", 0)], 1, "KEEP"),
    ("m_helper", [("label", "Display name", 0), ("helper", "The name people see.", 0)], 1, "KEEP"),
    ("m_members", [("h1", "Members", 0), ("subtitle", "The people on your team", 0)], 1, "KEEP"),
    ("m_orders", [("h1", "Orders", 0), ("subtitle", "Things you've bought", 0)], 1, "KEEP"),
    ("m_recipes", [("h1", "Recipes", 0), ("subtitle", "Dishes you've saved", 0)], 1, "KEEP"),
]
FLAGS = {"dialog_title": ["in_dialog"], "dialog_body": ["in_dialog"], "eyebrow": ["caps", "tracked", "small"]}


def r1_probe(sid: str, nodes: list[tuple[str, str, int]]) -> dict:
    kinds = {}
    for role, _, k in nodes:
        kinds.setdefault(k, "header" if role in ("h1", "h2", "eyebrow", "subtitle") else "body")
    return {"schema": "slop.screen/1", "id": sid, "framework": "tsx",
            "source": {"repo": "probe", "path": "probe.tsx", "commit": "0", "licence": "MIT"},
            "clusters": [{"k": k, "kind": v, "line": k + 1} for k, v in sorted(kinds.items())],
            "nodes": [{"i": i, "k": k, "role": role, "text": text,
                       "flags": (["in_dialog"] if role in ("button",) and nodes[0][0] == "dialog_title" else [])
                       + FLAGS.get(role, []), "loc": [i + 1, 1, i + 1, len(text) + 1]}
                      for i, (role, text, k) in enumerate(nodes)]}


def test_r1_results_examples_reproduce_on_the_round1_network(r2kit):
    """Every invented example in r1_results.md gets, from the round-1 network the kit hands over (work/ = runs/r1/work),
    the call the results describe: a flag for the false alarms, KEEP for the misses."""
    author, _ = r2kit
    results = (kb.KIT / "r1_results.md").read_text(encoding="utf-8")
    flat = " ".join(results.split())
    probe = author / "work" / "r1probe"
    probe.mkdir()
    try:
        cases = R1_FALSE_ALARMS + R1_MISSES
        for sid, nodes, x, _ in cases:
            assert f'"{nodes[x][1]}"' in flat, f"{sid}: {nodes[x][1]!r} is not quoted in r1_results.md"
            (probe / f"{sid}.json").write_text(json.dumps(r1_probe(sid, nodes)))
        out = ok(author, "score", "work", "--files", *[f"work/r1probe/{sid}.json" for sid, *_ in cases])
        assert "artefact e3e0b1d08f24" in out
        calls: dict[str, dict[int, str]] = {}
        cur = None
        for ln in out.splitlines():
            m = re.match(r"work/r1probe/(\w+)\.json ", ln)
            if m:
                cur = calls.setdefault(m.group(1), {})
                continue
            m = re.match(r"\s+(\d+) \d+ \S+ \".*\"\s+(KEEP|ECHO|NARRATE|DECOR)\s", ln)
            if m and cur is not None:
                cur[int(m.group(1))] = m.group(2)
        for sid, nodes, x, want in cases:
            assert calls[sid][x] == want, f"{sid}: round 1 calls {nodes[x][1]!r} {calls[sid][x]}, not {want}"
    finally:
        shutil.rmtree(probe)
