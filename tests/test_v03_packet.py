"""v0.3 authoring packet (spec v0.3 §8, §11 "Leakage")."""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent_loop_lab.v03 import tokenizer
from agent_loop_lab.v03.packet import (
    CONTENT_FILES,
    GRAMMAR_EXAMPLES,
    MANIFEST_FILE,
    SPEC_SENTINELS,
    PacketError,
    build_packet,
    extract_packet_spec,
    packet_sha256,
    verify_packet,
)

REPO = Path(__file__).resolve().parents[1]
V03 = REPO / "agent_doom_loop_detector_spec_v0_3.md"
V02 = REPO / "agent_doom_loop_detector_spec_v0_2.md"
SRC = REPO / "src" / "agent_loop_lab" / "v03"


@pytest.fixture(scope="module")
def packet(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("v03pkt") / "packet"
    build_packet(V03, V02, SRC, out)
    return out


def test_packet_is_deterministic(packet, tmp_path):
    again = tmp_path / "again"
    m = build_packet(V03, V02, SRC, again)
    assert m["packet_sha256"] == packet_sha256(packet)
    for name in (*CONTENT_FILES, MANIFEST_FILE):
        assert (again / name).read_bytes() == (packet / name).read_bytes(), name
    assert verify_packet(packet) == []


def test_packet_contents_and_manifest(packet):
    names = sorted(p.name for p in packet.iterdir())
    assert names == sorted((*CONTENT_FILES, MANIFEST_FILE))
    manifest = json.loads((packet / MANIFEST_FILE).read_text())
    assert sorted(manifest["files"]) == sorted(CONTENT_FILES)
    for leaky in ("synth", "datasets", "oracle", "labeller", "engine", "eval", "review", "disclosure", "metrics"):
        assert not any(leaky in n.lower() for n in names), leaky
    for n in ("model.py", "tokenizer.py", "renderer.py"):
        assert (packet / n).read_bytes() == (SRC / n).read_bytes()
        assert not re.search(r"^\s*(from|import)\s+agent_loop_lab", (packet / n).read_text(), re.M)
    # spec §8: rawop.py is withheld (its docstring describes the generators)
    assert "rawop.py" not in names and "rawop.py" not in manifest["files"]
    assert not any("rawop" in k for k in manifest["sources"])


def test_spec_md_keeps_packet_sections_only(packet):
    spec = (packet / "SPEC.md").read_text()
    for s in SPEC_SENTINELS:
        assert s not in spec, s
    # withheld v0.3 sections and v0.2 §6.4 (by heading)
    for withheld in ("## 0.", "### 0.1", "### 2.1", "### 2.3", "## 3.", "## 7.", "## 8.", "## 9.5", "## 10.",
                     "## 11.", "## 12.", "## 13.", "## 14.", "## 15."):
        assert not re.search(rf"^{re.escape(withheld)}", spec, re.M), withheld
    assert "6.4 Feature-collision" not in spec
    for phrase in ("delegated authority", "cleanup pass", "telemetry_gaps", "Codex", "E9", "Sections marked",
                   "Bayes ceiling", "plus a head and dimension budget"):
        assert phrase not in spec
    assert "[packet]" not in spec
    for kept in ("## 1. Objective and hypotheses", "### 2.2 Target model", "## 4. Log rendering (frozen)",
                 "## 5. Tokeniser (frozen)", "Gate rule.", "## 9. Direct-weight construction protocol",
                 "### 9.3 Phase B", "#### 6.1 Latent state", "#### 6.2 Pre-registered oracle parameters",
                 "#### 6.3 Window labelling rules", "#### 7.1 Scenario families", "Under-instrumented sessions",
                 "`N_stall` | 5"):
        assert kept in spec, kept


def test_extract_handles_paragraph_marker():
    v03 = "# T\n\n## 1. A [packet]\n\ntext a\n\n### 1.1 sub\n\nsub a\n\n## 2. B\n\nhidden\n\n**[packet]**\n\n- shown\n\n## 3. C\n\nhidden c\n"
    v02 = "\n".join(f"### {k} x\n\nbody {k}\n" for k in ("6.1", "6.2", "6.3", "6.4", "7.1"))
    out = extract_packet_spec(v03, v02)
    assert "text a" in out and "sub a" in out and "- shown" in out and "## 2. B" in out
    assert "hidden" not in out and "body 6.4" not in out and "body 7.1" in out and "[packet]" not in out


def test_missing_renderer_fails_clearly(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for n in ("model.py", "tokenizer.py"):
        shutil.copy(SRC / n, src / n)
    with pytest.raises(PacketError, match="renderer.py"):
        build_packet(V03, V02, src, tmp_path / "pkt")
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "v03_build_packet.py"), "--out", str(tmp_path / "p2"),
                        "--source-dir", str(src)], capture_output=True, text=True)
    assert r.returncode == 1 and "renderer.py" in r.stderr


def test_verify_detects_tampering_and_strays(packet, tmp_path):
    copy = tmp_path / "pkt"
    shutil.copytree(packet, copy)
    (copy / "SPEC.md").write_text((copy / "SPEC.md").read_text() + "\nE9\n")
    assert verify_packet(copy)
    shutil.rmtree(copy)
    shutil.copytree(packet, copy)
    (copy / "notes.txt").write_text("x")
    assert any("stray" in p for p in verify_packet(copy))


def test_no_packet_file_names_the_generators(packet):
    from agent_loop_lab.v03.packet import GENERATOR_WORDS, leakage_problems

    texts = {n: (packet / n).read_text() for n in CONTENT_FILES}
    assert leakage_problems(texts) == []
    for name, text in texts.items():
        for s in SPEC_SENTINELS:
            assert s not in text, (name, s)
        if name != "SPEC.md":
            for w in ("engine", "OOD", "oracle", "latent", "Session", "fault", "Session.fork", "OracleRecord"):
                assert not re.search(rf"(?<![A-Za-z0-9_]){re.escape(w)}(?![A-Za-z0-9_])", text), (name, w)
    assert "engine" in GENERATOR_WORDS and "OOD" in GENERATOR_WORDS


@pytest.mark.parametrize("name,inject", [
    ("AUTHOR_INSTRUCTIONS.md", "\nthe OOD engine nulls input_tokens only\n"),
    ("renderer.py", "\n# Session.fork copies the latent snapshot\n"),
    ("GRAMMAR_SHEET.md", "\nPhase C\n"),
])
def test_verify_scans_every_file_for_leaks(packet, tmp_path, name, inject):
    import hashlib

    copy = tmp_path / "pkt"
    shutil.copytree(packet, copy)
    f = copy / name
    f.write_text(f.read_text() + inject)
    manifest = json.loads((copy / MANIFEST_FILE).read_text())
    manifest["files"][name] = hashlib.sha256(f.read_bytes()).hexdigest()  # a consistent, re-hashed tamper
    if f"v03/{name}" in manifest["sources"]:
        manifest["sources"][f"v03/{name}"] = manifest["files"][name]
    (copy / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    problems = verify_packet(copy)
    assert any(name in p and ("generators" in p or "withheld" in p) for p in problems), problems


def test_author_instructions_list_the_rawop_fields(packet):
    text = (packet / "AUTHOR_INSTRUCTIONS.md").read_text()
    for field in ("family", "target", "args", "outcome", "error_class", "error_location", "validation", "changed",
                  "extra", "input_tokens", "output_tokens", "timestamp", "dropped"):
        assert f"`{field}`" in text, field
    for word in ("digest_after", "digest_before", "failure", "input_tokens", "output_tokens", "validation_passed",
                 "validation_suite", "validation_total"):
        assert f"`{word}`" in text, word
    assert "rawop.py" not in text and "import rawop" not in text


def test_grammar_sheet_tokens_match_tokenizer(packet):
    sheet = (packet / "GRAMMAR_SHEET.md").read_text()
    tools = {block.split()[1] for _, block in GRAMMAR_EXAMPLES}
    assert set(tokenizer.TOOL_NAMES) <= tools
    for _, block in GRAMMAR_EXAMPLES:
        assert "```text\n" + block + "\n```" in sheet
        enc = tokenizer.encode_session([block])
        assert len(enc.tokens) <= tokenizer.O


def _generator_vocab() -> tuple[set[str], set[str]]:
    words: set[str] = set()
    for pkg in ("synth", "synth_ood"):
        for p in (REPO / "src" / "agent_loop_lab" / pkg).rglob("*.py"):
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    words.update(tokenizer._WORD.findall(node.value))
    comps = {c.lower() for w in words for c in re.split(r"[/.\-_]", w) if c}
    return words, comps


def test_grammar_sheet_words_disjoint_from_generator_vocab(packet):
    from agent_loop_lab.synth import world  # the generator's vocabulary pools

    words, comps = _generator_vocab()
    for name in dir(world):
        val = getattr(world, name)
        if isinstance(val, tuple) and all(isinstance(x, str) for x in val):
            for x in val:
                words.update(tokenizer._WORD.findall(x))
    comps |= {c.lower() for w in words for c in re.split(r"[/.\-_]", w) if c}
    sheet = (packet / "GRAMMAR_SHEET.md").read_text()
    fences = re.findall(r"```text\n(.*?)\n```", sheet, re.S)
    assert len(fences) == len(GRAMMAR_EXAMPLES)
    sheet_words = set()
    for block in fences:
        for _tid, surface in tokenizer.tokenize_block(block):
            if tokenizer._WORD.fullmatch(surface) and surface not in tokenizer.RESERVED_IDS:
                sheet_words.add(surface)
    assert sheet_words
    assert not sheet_words & words, sorted(sheet_words & words)
    sheet_comps = {c.lower() for w in sheet_words for c in re.split(r"[/.\-_]", w) if c}
    assert not sheet_comps & comps, sorted(sheet_comps & comps)


def test_state_dict_template(packet):
    from agent_loop_lab.v03.model import PARAM_COUNT, PARAM_SHAPES

    doc = json.loads((packet / "state_dict_template.json").read_text())
    assert doc["param_count"] == PARAM_COUNT == 2_128_932
    assert [t["name"] for t in doc["tensors"]] == list(PARAM_SHAPES)


def test_author_instructions_cover_interfaces(packet):
    text = (packet / "AUTHOR_INSTRUCTIONS.md").read_text()
    for s in ("def reference_policy(tokens: list[str], event_ord: list[int]) -> int", "def build()", "./py ",
              "One command per Bash call", "torch.outer", "torch.kron", "256 elements", "512 characters",
              "200 KB", "300 s", "cost cap", "CIRCUITS.md", "ITERATIONS.md", "REVISION.md", "hashlib"):
        assert s in text, s
    assert "$" not in text.split("cost cap")[1][:80]  # no number for the cap
