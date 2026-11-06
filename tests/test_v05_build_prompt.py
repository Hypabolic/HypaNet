"""v0.5 prompt builder (scripts/v05_build_prompt.py; experiments/v0_5/PROMPT_NOTES.md)."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "v05_build_prompt.py"
PACKET = REPO / "experiments" / "v0_3" / "packet_handed_over"
OUT = REPO / "experiments" / "v0_5"


def _load_script():
    spec = importlib.util.spec_from_file_location("v05_build_prompt", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v05_build_prompt"] = mod
    spec.loader.exec_module(mod)
    return mod


bp = _load_script()


@pytest.fixture(scope="module")
def prompt() -> str:
    return bp.build()


def test_deterministic_and_matches_committed(prompt):
    assert bp.build() == prompt
    assert (OUT / "PROMPT.txt").read_bytes() == prompt.encode("utf-8")
    assert (OUT / "PROMPT.sha256").read_text() == bp.sha_line(prompt)
    assert bp.main(["--check"]) == 0


def test_check_detects_a_changed_prompt(tmp_path, prompt):
    (tmp_path / "PROMPT.txt").write_bytes(prompt.encode("utf-8") + b"\n")
    (tmp_path / "PROMPT.sha256").write_text(bp.sha_line(prompt))
    assert bp.main(["--check", "--out", str(tmp_path)]) == 1


def test_frozen_sources_verbatim(prompt):
    for name in ("renderer.py", "tokenizer.py", "model.py"):
        src = (PACKET / name).read_text().rstrip("\n")
        assert f"```python\n{src}\n```" in prompt, name
    assert (PACKET / "GRAMMAR_SHEET.md").read_text().strip("\n") in prompt


def test_every_tensor_listed_with_its_shape(prompt):
    template = json.loads((PACKET / "state_dict_template.json").read_text())
    assert len(template["tensors"]) == 56
    for t in template["tensors"]:
        shape = "[" + ", ".join(str(s) for s in t["shape"]) + "]"
        assert re.search(rf"^  {re.escape(t['name'])} +{re.escape(shape)}$", prompt, re.M), t["name"]


def test_token_id_table_matches_the_tokenizer(prompt):
    tok = bp._load_module(PACKET / "tokenizer.py", "_t_v05_tok")
    for word, i in tok.RESERVED_IDS.items():
        assert re.search(rf"(^  |   ){i} {re.escape(word)}(   |$)", prompt, re.M), word
    for ch, i in tok.PUNCT_IDS.items():
        assert re.search(rf"(^  |   ){i} {re.escape(ch)}(   |$)", prompt, re.M), ch
    assert tok.DIGIT_IDS["09"] == 243 and tok.DIGIT_IDS["250"] == 584  # the examples in the table
    assert tok.word_id("run_tests") == 13  # the first edits example


def test_framing_multi_reply_and_end_marker(prompt):
    assert prompt.startswith("YOUR TASK: write, from scratch, the weights of a transformer")
    assert 'you will receive only the message "Continue."' in prompt
    assert "a line that is exactly END OF WEIGHTS (outside any edits block)" in prompt
    assert "Every edits block of every reply is read and applied, in the order written" in prompt
    assert "single response" not in prompt and "Only the LAST edits block" not in prompt
    assert "EVERY WEIGHT STARTS AT 0" in prompt
    for heading in ("THE TASK:", "THE LOGS (", "THE NETWORK (", "HOW YOUR NETWORK IS SCORED:", "HOW TO WRITE THE WEIGHTS ("):
        assert re.search(rf"^{re.escape(heading)}", prompt, re.M), heading
    assert not prompt.endswith("\n") and "\r" not in prompt


def _outside_sources(prompt: str) -> str:
    return re.sub(r"```python\n.*?\n```", "", prompt, flags=re.S)


def test_no_tool_protocol_budget_or_data_text(prompt):
    text = _outside_sources(prompt)
    for bad in ("reference_policy", "build_weights", "ITERATIONS", "CIRCUITS", "./py", "Phase A", "Phase B",
                "cost cap", "US$", "SPEC.md", "AUTHOR_INSTRUCTIONS", "oracle_config", "datasets/",
                "128000", "128,000", "max_tokens", "output token", "budget of", "disclos"):
        assert bad not in text, bad
    assert "macro-F1" in text and "false-alarm rate" in text


def test_edits_examples_name_real_tensors_in_range(prompt):
    template = {t["name"]: t["shape"] for t in json.loads((PACKET / "state_dict_template.json").read_text())["tensors"]}
    block = re.search(r"```edits\n(.*?)\n```", prompt, re.S).group(1)
    stmts = [ln.split("#")[0].strip() for ln in block.split("\n")]
    for s in filter(None, stmts):
        m = re.match(r"^([a-z_.0-9]+)\[([^\]]+)\]\s*(=|\*=|\+=|-=)", s)
        assert m and m.group(1) in template, s
        shape = template[m.group(1)]
        idx = [p.strip() for p in m.group(2).split(",")]
        assert len(idx) == len(shape), s
        for p, n in zip(idx, shape):
            for v in (x for x in p.split(":") if x):
                assert 0 <= int(v) <= n, s


def _copy_packet(tmp_path: Path) -> Path:
    d = tmp_path / "packet"
    shutil.copytree(PACKET, d)
    return d


def test_tampered_packet_file_is_refused(tmp_path):
    d = _copy_packet(tmp_path)
    p = d / "SPEC.md"
    p.write_text(p.read_text().replace("N_stall` | 5", "N_stall` | 4"))
    with pytest.raises(bp.PromptBuildError, match="SPEC.md: sha256"):
        bp.build(d)


def test_tampered_manifest_is_refused(tmp_path):
    d = _copy_packet(tmp_path)
    p = d / "PACKET_MANIFEST.json"
    p.write_text(p.read_text() + " ")
    with pytest.raises(bp.PromptBuildError, match="registered v0.3 packet"):
        bp.build(d)


def test_scorer_source_must_match_packet(tmp_path):
    src = tmp_path / "v03"
    src.mkdir()
    for name in bp.SOURCE_FILES:
        shutil.copy(PACKET / name, src / name)
    (src / "model.py").write_text((src / "model.py").read_text() + "\n")
    with pytest.raises(bp.PromptBuildError, match="run by the scorer"):
        bp.load_packet(PACKET, scorer_sources=src)


def test_split_and_replace_helpers():
    text = "head\n## A\na\n```\n## B\n```\n## B\nb\n"
    pieces = bp.split_at_headings(text, ["## A", "## B"])
    assert pieces == {"": "head\n", "## A": "## A\na\n```\n## B\n```\n", "## B": "## B\nb\n"}
    with pytest.raises(bp.PromptBuildError, match="not found"):
        bp.split_at_headings(text, ["## C"])
    with pytest.raises(bp.PromptBuildError, match="document order"):
        bp.split_at_headings(text, ["## B", "## A"])
    with pytest.raises(bp.PromptBuildError, match="exactly one"):
        bp.replace_once("x x", "x", "y")
    assert bp.drop_table("a\n| h |\n|---|\n| r |\n\nb", "| h |") == "a\nb"


def test_index_rules_stated_and_honoured_by_the_reader(prompt):
    from agent_loop_lab.v05.edits import apply_replies

    assert "no negative indices and no arithmetic (write 48:64, not 3*16:4*16)" in prompt
    assert "a range is not clamped to the axis" in prompt and "a:, :b and ::s are accepted" in prompt
    assert "h*16" not in _outside_sources(prompt)
    assert "if the cut fell inside an edits block, open a new edits block" in prompt
    assert "may instead receive a fixed notice that it was cut off" in prompt
    ok = "layers.0.attn.q.weight[48:64, 0] = 1\nhead.bias[2:] = 1\nhead.bias[:1] = 1\nhead.bias[::2] = 2"
    bad = "head.weight[-1, 0] = 1\nlayers.0.attn.q.weight[3*16:4*16, 0] = 1\nlayers.0.attn.rel_bias[0, 60:70] = 1"
    er = apply_replies([f"```edits\n{ok}\n{bad}\n```"]).report["edit_report"]
    assert er["applied"] == 4 and er["skipped"] == 3
