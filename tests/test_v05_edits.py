"""v0.5 edits reader: every statement form, blocks across replies, skip-and-continue, cut-offs, CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from agent_loop_lab.v03.artifact import artifact_sha256, load_artifact
from agent_loop_lab.v03.model import PARAM_COUNT, PARAM_SHAPES
from agent_loop_lab.v05.edits import (
    END_MARKER,
    apply_replies,
    ends_with_end_marker,
    extract_blocks,
    reply_files,
    RunDirError,
)

REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "scripts" / "v05_apply_reply.py"

FC1 = "layers.0.mlp.fc1.weight"  # [512, 128]
FC1B = "layers.0.mlp.fc1.bias"  # [512]
Q = "layers.0.attn.q.weight"  # [128, 128]
RB = "layers.2.attn.rel_bias"  # [8, 65]


def block(body: str) -> str:
    return f"Design notes.\n\n```edits\n{body}\n```\n"


def run(*replies: str):
    res = apply_replies(list(replies))
    return res.state_dict, res.report


def errs(report) -> list[str]:
    return [e["reason"] for e in report["edit_report"]["errors"]]


# -- every Exp 5 statement form, translated to v0.3 tensors -----------------------------------------


def test_exp5_forms():
    sd, rep = run(block(
        "head.weight[2, 5] = 2.5                       # one entry\n"
        f"{FC1}[0:6, 3] = [1, 0, -1, 0, 2, 0]          # explicit values along a range\n"
        f"{FC1B}[0:4] = -1                             # fill a range\n"
        "tok_emb.weight[3:600:60, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]   # strided\n"
        f"{Q}[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]    # 2-D list for a 2-D block\n"
        "head.weight[:, 5] *= 2                        # scale\n"
        "head.bias[1] += 0.25\n"
        "head.bias[1] -= 1\n"
    ))
    er = rep["edit_report"]
    assert er["errors"] == [] and er["statements"] == 8 and er["applied"] == 8
    assert sd["head.weight"][2, 5].item() == 5.0
    assert sd[FC1][0:6, 3].tolist() == [1, 0, -1, 0, 2, 0]
    assert sd[FC1B][0:4].tolist() == [-1] * 4 and sd[FC1B][4].item() == 0
    col = sd["tok_emb.weight"][:, 0]
    assert col[543].item() == 1 and col[483].item() == 1 and col.abs().sum().item() == 2
    assert sd[Q][0:2, 0:3].tolist() == [[1, -1, 0.5], [0, 2, 0]]
    assert sd["head.bias"][1].item() == -0.75
    assert er["entries_written"] == 1 + 6 + 4 + 10 + 6 + 4 + 1 + 1
    assert er["tensors_touched"] == ["tok_emb.weight", "layers.0.attn.q.weight", "layers.0.mlp.fc1.weight",
                                     "layers.0.mlp.fc1.bias", "head.weight", "head.bias"]


def test_space_after_first_bracket_and_multiline_lists_with_comments():
    sd, rep = run(block(
        f"{Q}[0:2, 0:3] = [ [ 1 , -1 ,0.5 ] ,\n"
        "   # a comment inside the list\n"
        "   [0,2,  0],   # trailing comma allowed\n"
        "]\n"
        f"   {FC1B} [ 0 : 3 ]   =   [\n"
        "  1,\n"
        "\n"
        "  2,  # two\n"
        "  3\n"
        "  ]\n"
        "head.bias[0] =\n"
        "   7\n"
    ))
    assert errs(rep) == []
    assert sd[Q][0:2, 0:3].tolist() == [[1, -1, 0.5], [0, 2, 0]]
    assert sd[FC1B][0:3].tolist() == [1, 2, 3]
    assert sd["head.bias"][0].item() == 7
    assert [e for e in rep["edit_report"]["errors"]] == []


def test_slice_forms_and_strides():
    sd, rep = run(block(
        f"{FC1B}[500:] = 1\n"
        f"{FC1B}[:3] = 2\n"
        f"{FC1B}[10:20:5] = [3, 4]\n"
        f"{FC1B}[100::200] = 5          # 100, 300, 500\n"
        f"{RB}[::4, 64] = [6, 7]        # heads 0 and 4, BOS column\n"
        f"{FC1B}[200:210:3] = [8, 8, 8, 8]\n"
    ))
    assert errs(rep) == []
    b = sd[FC1B]
    assert b[501:512].tolist() == [1] * 11 and b[499].item() == 0
    assert b[0:3].tolist() == [2, 2, 2]
    assert b[10].item() == 3 and b[15].item() == 4
    assert b[100].item() == 5 and b[300].item() == 5 and b[500].item() == 5 and b[501].item() == 1
    assert sd[RB][0, 64].item() == 6 and sd[RB][4, 64].item() == 7 and sd[RB][1, 64].item() == 0
    assert [b[i].item() for i in (200, 203, 206, 209)] == [8] * 4 and b[201].item() == 0


def test_dimension_rules():
    sd, rep = run(block(
        f"{RB}[3] = 1.5                          # missing trailing index = whole axis\n"
        "head.bias = [1, 2, 3, 4]                # bare name = whole tensor\n"
        f"{Q}[5, 0:3] = [1, 2, 3]                # integer index drops its axis\n"
        f"{Q}[0:3, 7] = [[1], [2], [3]]          # same shape once length-1 axes are removed\n"
        f"{Q}[9:10, 0:3] = [4, 5, 6]\n"
        "head.weight[1, 1] = [9]\n"
        f"{Q}[0:2, 0:3] = [1, 2, 3, 4, 5, 6]      # flat list for a 2-D block: rejected\n"
        f"{FC1B}[0:3] = [[1, 2, 3]] \n"
    ))
    assert sd[RB][3].tolist() == [1.5] * 65 and sd[RB][2].abs().sum().item() == 0
    assert sd["head.bias"].tolist() == [1, 2, 3, 4]
    assert sd[Q][5, 0:3].tolist() == [1, 2, 3]
    assert sd[Q][0:3, 7].tolist() == [1, 2, 3]
    assert sd[Q][9, 0:3].tolist() == [4, 5, 6]
    assert sd["head.weight"][1, 1].item() == 9
    reasons = errs(rep)
    assert len(reasons) == 1 and "list shape [6]" in reasons[0]
    assert sd[FC1B][0:3].tolist() == [1, 2, 3]  # accepted: [1, 3] squeezes to [3]


def test_overwrite_order_and_compound_ops():
    sd, rep = run(block(
        "head.bias[:] = 1\n"
        "head.bias[0] = 5\n"
        "head.bias[:] *= 2\n"
        "head.bias[:] += [1, 2, 3, 4]\n"
        "head.bias[0:2] -= [[0.5, 0.5]]\n"
        "head.bias[3] = 0\n"
    ))
    assert errs(rep) == []
    assert sd["head.bias"].tolist() == [10.5, 3.5, 5, 0]


def test_numbers():
    sd, rep = run(block(
        "head.bias[0] = 1e-3\n"
        "head.bias[1] = -2.5E+2\n"
        "head.bias[2] = −0.5\n"
        "head.bias[3] = .5\n"
        "head.weight[0, 0] = +3.\n"
        "head.weight[0, 1] = - 4\n"
        "head.weight[0, 2] = 1000000\n"
        "head.weight[0, 3] = -1e6\n"
    ))
    assert errs(rep) == []
    assert torch.allclose(sd["head.bias"], torch.tensor([1e-3, -250, -0.5, 0.5]))
    assert sd["head.weight"][0, 0:4].tolist() == [3, -4, 1e6, -1e6]
    assert sd["head.weight"].dtype == torch.float32


# -- skip and continue ------------------------------------------------------------------------------


@pytest.mark.parametrize("bad, reason", [
    ("head.bais[0] = 1", "unknown tensor 'head.bais'"),
    ("head.bias[4] = 1", "out of range"),
    ("head.bias[-1] = 1", "negative index"),
    (f"{FC1B}[0:-1] = 1", "negative index"),
    ("head.bias[1.0] = 1", "not a whole number"),
    ("head.bias[0:5] = 1", "exceeds axis"),
    ("head.bias[2:2] = 1", "empty range"),
    ("head.bias[0:4:0] = 1", "step 0"),
    ("head.bias[0:3] = [1, 2]", "list shape [2]"),
    ("head.weight[0:2, 0:2] = [[1, 2], [3]]", "ragged"),
    ("head.weight[0:2, 0:2] = [[1, 2], 3]", "ragged"),
    ("head.bias[0, 1] = 1", "1 axis/axes"),
    ("head.weight[0][1] = 1", "one [ ]"),
    ("head.bias[0] = 2000000", "exceeds |v| <= 1e6"),
    ("head.bias[0] = 1e400", "not finite"),
    ("head.bias[0] = nan", "not a plain number"),
    ("head.bias[0] = inf", "not a plain number"),
    ("head.bias[0] = 1/3", "unexpected character '/'"),
    ("head.bias[0] = 0x10", "after the value"),
    ("head.bias[0] = []", "empty list"),
    ("head.bias[0] /= 2", "unexpected character '/'"),
    ("head.bias[0] 1", "expected '='"),
    ("head.bias[] = 1", "expected an index"),
    ("head.bias[0] = 1; head.bias[1] = 2", "unexpected character ';'"),
    ("Layer 0 sets the bias", "expected '='"),
    ("[1, 2] = head.bias", "starts with a tensor name"),
    ("head.bias[0] = [[[[[[[[[1]]]]]]]]]", "nested too deeply"),
])
def test_bad_statement_is_skipped_and_others_apply(bad, reason):
    sd, rep = run(block(f"head.bias[3] = 7\n{bad}\nhead.weight[3, 127] = 8"))
    er = rep["edit_report"]
    assert sd["head.bias"][3].item() == 7 and sd["head.weight"][3, 127].item() == 8
    assert er["statements"] == 3 and er["applied"] == 2 and er["skipped"] == 1
    (e,) = er["errors"]
    assert reason in e["reason"], e
    assert e["reply"] == 1 and e["block"] == 1 and e["line"] == 5  # reply line numbers, 1-based
    assert sd["head.bias"][0:3].abs().sum().item() == 0


def test_compound_overflow_is_skipped_and_leaves_tensor_unchanged():
    sd, rep = run(block("head.bias[:] = 1000\nhead.bias[:] *= 1001\nhead.bias[0] += 0.5"))
    assert sd["head.bias"].tolist() == [1000.5, 1000, 1000, 1000]
    assert "larger than 1e6" in errs(rep)[0]


def test_unclosed_bracket_in_closed_block_costs_only_its_statement():
    sd, rep = run(block(
        "head.bias[0:2] = [1, 2\n"
        "head.bias[2] = 3\n"
        "head.bias[3] = [4\n"
    ))
    assert sd["head.bias"].tolist() == [0, 0, 3, 0]
    assert errs(rep) == ["unclosed '['", "unclosed '['"]
    assert rep["edit_report"]["errors"][0]["line"] == 4


def test_trailing_operator_in_closed_block_is_an_error():
    sd, rep = run(block("head.bias[0] =\nhead.bias[1] = 2"))
    assert sd["head.bias"].tolist() == [0, 2, 0, 0]
    assert "incomplete statement" in errs(rep)[0]


# -- blocks and replies -----------------------------------------------------------------------------


def test_all_blocks_across_replies_in_order_and_later_overwrites():
    r1 = block("head.bias[:] = 1\nhead.weight[0, 0] = 3") + "\nmore prose\n\n```edits\nhead.bias[1] = 2\n```\n"
    r2 = "Part two.\n```edits\nhead.bias[0] = 9   # overwrites reply 1\n```\n" + END_MARKER + "\n"
    sd, rep = run(r1, r2)
    assert sd["head.bias"].tolist() == [9, 2, 1, 1] and sd["head.weight"][0, 0].item() == 3
    assert rep["edit_report"]["blocks"] == 3 and rep["n_replies"] == 2
    assert [r["blocks"] for r in rep["replies"]] == [2, 1]
    assert [(b["reply"], b["index_in_reply"], b["closed_by"]) for b in rep["blocks"]] == [
        (1, 1, "fence"), (1, 2, "fence"), (2, 1, "fence")]
    assert rep["end_of_weights"] is True and rep["end_of_weights_reply"] == 2
    assert [r["ends_with_end_of_weights"] for r in rep["replies"]] == [False, True]


def test_no_block_and_other_fences_ignored():
    text = ("```python\nhead.bias[0] = 1\n```\n"
            "```\nhead.bias[1] = 1\n```\n"
            "head.bias[2] = 1\n"
            "```edits-v2\nhead.bias[3] = 1\n```\n")
    sd, rep = run(text)
    assert rep["edit_report"]["block"] is False and rep["edit_report"]["statements"] == 0
    assert sd["head.bias"].abs().sum().item() == 0
    assert rep["nonzero_total"] == 0 and rep["end_of_weights"] is False


def test_fence_variants():
    text = ("  ```` Edits   \nhead.bias[0] = 1\n```python\nhead.bias[1] = 1\n```\n"
            "~~~edits\nhead.bias[2] = 1\n~~~\n"
            "```edits\nhead.bias[3] = 1\n```edits\nhead.weight[0, 0] = 1\n```\n")
    sd, rep = run(text)
    assert sd["head.bias"].tolist() == [1, 0, 1, 1] and sd["head.weight"][0, 0].item() == 1
    assert [b["closed_by"] for b in rep["blocks"]] == ["fence_with_info", "fence", "fence_with_info", "fence"]


def test_end_marker_closes_an_open_block():
    sd, rep = run("```edits\nhead.bias[0] = 1\nhead.bias[1] = 2\nEND OF WEIGHTS")
    assert sd["head.bias"].tolist() == [1, 2, 0, 0]
    assert rep["blocks"][0]["closed_by"] == "end_marker" and rep["edit_report"]["cut_off"] is False
    assert rep["end_of_weights"] is True


def test_ends_with_end_marker():
    assert ends_with_end_marker("x\nEND OF WEIGHTS\n\n  ")
    assert ends_with_end_marker("  END OF WEIGHTS  ")
    assert not ends_with_end_marker("END OF WEIGHTS\nmore")
    assert not ends_with_end_marker("**END OF WEIGHTS**")
    assert not ends_with_end_marker("end of weights")
    assert not ends_with_end_marker("")


# -- cut off at the end of a reply ------------------------------------------------------------------


def test_cut_off_mid_list_then_continued_in_next_reply():
    r1 = "```edits\nhead.bias[0] = 1\nhead.weight[0:2, 0:3] = [[1, 2, 3],\n  [4, 5"
    r2 = "```edits\nhead.weight[0:2, 0:3] = [[1, 2, 3], [4, 5, 6]]\n```\nEND OF WEIGHTS\n"
    sd, rep = run(r1, r2)
    er = rep["edit_report"]
    assert sd["head.bias"][0].item() == 1
    assert sd["head.weight"][0:2, 0:3].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert er["cut_off"] is True and er["errors"] == []
    (c,) = er["cut_off_statements"]
    assert (c["reply"], c["line"], c["end_line"]) == (1, 3, 4)
    assert rep["blocks"][0]["cut_off"] is True and rep["blocks"][0]["closed_by"] == "end_of_reply"
    assert rep["replies"][0]["cut_off"] is True and rep["replies"][1]["cut_off"] is False
    assert er["statements"] == er["applied"] + er["skipped"] + len(er["cut_off_statements"])


def test_cut_off_last_number_without_newline_is_dropped():
    sd, rep = run("```edits\nhead.bias[0] = 1\nhead.bias[1] = 0.12")
    assert sd["head.bias"].tolist() == [1, 0, 0, 0]
    assert rep["edit_report"]["cut_off_statements"][0]["line"] == 3


def test_cut_off_after_newline_keeps_the_last_statement():
    sd, rep = run("```edits\nhead.bias[0] = 1\nhead.bias[1] = 0.12\n")
    assert sd["head.bias"].tolist() == pytest.approx([1, 0.12, 0, 0])
    assert rep["edit_report"]["cut_off"] is True and rep["edit_report"]["cut_off_statements"] == []


def test_cut_off_complete_forms_on_last_line():
    sd, rep = run("```edits\nhead.bias[0:2] = [1, 2]")  # closed list: cannot be truncated
    assert sd["head.bias"].tolist() == [1, 2, 0, 0] and rep["edit_report"]["cut_off_statements"] == []
    sd, rep = run("```edits\nhead.bias[1] = 3  # a comm")  # the comment shows the code ended
    assert sd["head.bias"][1].item() == 3 and rep["edit_report"]["cut_off_statements"] == []


@pytest.mark.parametrize("tail", ["head.bias[0] =", "head.bias[0", "head.bi", "head.bias[0]", "head.bias[0:2] = [1,"])
def test_cut_off_truncated_tails_are_recorded_not_errors(tail):
    sd, rep = run(f"```edits\nhead.bias[3] = 4\n{tail}")
    er = rep["edit_report"]
    assert sd["head.bias"].tolist() == [0, 0, 0, 4]
    assert er["errors"] == [] and len(er["cut_off_statements"]) == 1


def test_crlf_replies():
    sd, rep = run("```edits\r\nhead.bias[0] = 1\r\nhead.bias[1] = [2,\r\n 3][0:1]\r\n```\r\n")
    assert sd["head.bias"][0].item() == 1 and len(errs(rep)) == 1


# -- report -----------------------------------------------------------------------------------------


def test_report_counts():
    sd, rep = run(block("head.bias[:] = [0, -3, 0.5, 0]\nhead.weight[:, 0] = 2\nhead.weight[0, 0] = 0"))
    assert rep["parameters"] == PARAM_COUNT == 2_128_932
    assert list(rep["nonzero"]) == list(PARAM_SHAPES) and list(rep["max_abs"]) == list(PARAM_SHAPES)
    assert rep["nonzero"]["head.bias"] == 2 and rep["nonzero"]["head.weight"] == 3
    assert rep["nonzero_total"] == 5
    assert rep["max_abs"]["head.bias"] == 3 and rep["max_abs"]["tok_emb.weight"] == 0
    assert rep["edit_report"]["entries_written"] == 4 + 4 + 1
    assert rep["reader"]["negative_indices"] == "rejected" and rep["reader"]["exponent_numbers"] == "accepted"
    assert len(rep["replies"][0]["sha256"]) == 64
    json.dumps(rep)  # serialisable
    assert list(sd) == list(PARAM_SHAPES)
    assert all(tuple(sd[k].shape) == s and sd[k].dtype == torch.float32 for k, s in PARAM_SHAPES.items())


def test_extract_blocks_line_numbers():
    (b,) = extract_blocks("a\nb\n```edits\nx\ny\n```\n", reply=3)
    assert (b.reply, b.open_line, b.close_line, [ln.number for ln in b.lines]) == (3, 3, 6, [4, 5])


# -- CLI --------------------------------------------------------------------------------------------


def _write_run(root: Path, replies: list[str], start: int = 1) -> Path:
    d = root / "r1"
    (d / "replies").mkdir(parents=True)
    for i, text in enumerate(replies, start=start):
        (d / "replies" / f"{i:03d}.md").write_bytes(text.encode("utf-8"))
    (d / "replies" / "001.json").write_text("{}")  # ignored
    return d


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CLI), *args], capture_output=True, text=True, cwd=REPO)


def test_cli_writes_reads_back_refuses_overwrite_and_checks(tmp_path):
    d = _write_run(tmp_path, [block("head.bias[0] = 1\ntok_emb.weight[5, 0:2] = [0.5, -2]\nbad[0] = 1"),
                              "```edits\nhead.bias[1] = 2\n```\nEND OF WEIGHTS\n"])
    p = _cli(str(d))
    assert p.returncode == 0, p.stderr
    model, meta = load_artifact(d / "artifact.dlw")
    sd = model.state_dict()
    assert sd["head.bias"].tolist() == [1, 2, 0, 0] and sd["tok_emb.weight"][5, 0:2].tolist() == [0.5, -2]
    assert meta["experiment"] == "v0.5" and meta["run"] == "r1" and meta["end_of_weights"] == "true"
    sha = artifact_sha256(d / "artifact.dlw")
    assert (d / "artifact.sha256").read_text() == f"{sha}  artifact.dlw\n"
    rep = json.loads((d / "apply.json").read_text())
    assert rep["artifact"]["sha256"] == sha and rep["edit_report"]["skipped"] == 1
    assert [r["file"] for r in rep["replies"]] == ["001.md", "002.md"]
    assert rep["nonzero_total"] == 4

    before = (d / "artifact.dlw").read_bytes()
    p = _cli(str(d))
    assert p.returncode == 1 and "refusing to overwrite" in p.stderr
    assert (d / "artifact.dlw").read_bytes() == before

    assert _cli(str(d), "--check").returncode == 0
    (d / "replies" / "002.md").write_text("```edits\nhead.bias[1] = 3\n```\n")
    p = _cli(str(d), "--check")
    assert p.returncode == 1 and "differs" in p.stderr


def test_cli_deterministic(tmp_path):
    texts = [block("head.bias[0] = 1"), "nothing\n"]
    a = _write_run(tmp_path / "a", texts)
    b = _write_run(tmp_path / "b", texts)
    assert _cli(str(a)).returncode == 0 and _cli(str(b)).returncode == 0
    assert (a / "artifact.dlw").read_bytes() == (b / "artifact.dlw").read_bytes()
    assert (a / "apply.json").read_bytes() == (b / "apply.json").read_bytes()


def test_reply_numbering(tmp_path):
    d = _write_run(tmp_path, ["a", "b"], start=0)
    assert [p.name for p in reply_files(d)] == ["000.md", "001.md"]
    (d / "replies" / "003.md").write_text("c")
    with pytest.raises(RunDirError, match="contiguous"):
        reply_files(d)
    e = tmp_path / "empty"
    (e / "replies").mkdir(parents=True)
    with pytest.raises(RunDirError, match="no NNN.md"):
        reply_files(e)
    p = _cli(str(tmp_path / "missing"))
    assert p.returncode == 1 and "no replies directory" in p.stderr


# -- review fixes: parts of a turn, stop-reason-aware cut-offs, lines outside blocks --------------


def test_cut_inside_a_turn_is_a_cut_when_parts_are_read_separately():
    part1 = "```edits\nhead.bias[0] = 1\nhead.bias[3] = 0.2"  # cut while writing 0.25
    part2 = "5\nhead.bias[1] = 2\n```\n" + END_MARKER
    joined = apply_replies([part1 + "\n" + part2])  # the old reading: the truncated 0.2 is applied
    assert joined.state_dict["head.bias"].tolist()[3] == pytest.approx(0.2)
    res = apply_replies([part1, part2], cut_off=[True, False])
    assert res.state_dict["head.bias"].tolist() == [1.0, 0.0, 0.0, 0.0]
    er = res.report["edit_report"]
    assert [c["statement"] for c in er["cut_off_statements"]] == ["head.bias[3] = 0.2"] and er["errors"] == []
    assert res.report["replies"][1]["outside_block_statement_lines"] == 1
    assert res.report["replies"][1]["outside_block_examples"] == [{"line": 2, "text": "head.bias[1] = 2"}]


def test_unclosed_block_of_a_reply_that_was_not_cut_is_read_whole():
    replies = ["Layer 0 weights:\n```edits\nhead.bias[0] = 1\nhead.bias[3] = 0.25",
               "```edits\nhead.bias[1] = 2\n```\n" + END_MARKER]
    res = apply_replies(replies, cut_off=[False, False])
    assert res.state_dict["head.bias"].tolist() == [1.0, 2.0, 0.0, 0.25]
    assert res.report["blocks"][0]["closed_by"] == "end_of_reply_not_cut" and not res.report["blocks"][0]["cut_off"]
    assert res.report["edit_report"]["cut_off_statements"] == []
    # an open bracket at a normal end is an unreadable statement, not a cut-off
    res = apply_replies(["```edits\nhead.bias[0:2] = [1,"], cut_off=[False])
    assert res.report["edit_report"]["skipped"] == 1 and res.report["edit_report"]["cut_off_statements"] == []
    # unknown (None) keeps the conservative rule
    res = apply_replies(replies[:1], cut_off=[None])
    assert len(res.report["edit_report"]["cut_off_statements"]) == 1
    with pytest.raises(ValueError):
        apply_replies(replies, cut_off=[True])


def test_lines_outside_blocks_are_counted_not_applied():
    res = apply_replies(["```edits\nhead.bias[0] = 1\nhead.bias[1:4] = [2,",
                         "3, 4]\nhead.bias[0] = 5\nThe head uses layers.0.attn.q.weight for queries.\n"
                         "```python\nx = [1, 2]\n```\n" + END_MARKER])
    assert res.state_dict["head.bias"].tolist() == [1.0, 0.0, 0.0, 0.0]
    r2 = res.report["replies"][1]
    assert r2["statements"] == 0 and r2["outside_block_statement_lines"] == 2
    assert [e["line"] for e in r2["outside_block_examples"]] == [1, 2]
    assert res.report["edit_report"]["outside_block_statement_lines"] == 2


def _index_run(d: Path, units: list[dict], files: dict[str, str]) -> Path:
    import hashlib

    (d / "replies").mkdir(parents=True)
    for name, text in files.items():
        (d / "replies" / name).write_text(text, encoding="utf-8")
    for u in units:
        u.setdefault("sha256", hashlib.sha256(files[u["file"]].encode()).hexdigest())
    (d / "replies" / "INDEX.json").write_text(json.dumps({"units": units}))
    return d


def test_run_directory_is_read_through_its_index(tmp_path):
    from agent_loop_lab.v05.edits import build_run, reading_units

    p1, p2 = "```edits\nhead.bias[0] = 1\nhead.bias[3] = 0.2", "```edits\nhead.bias[3] = 0.25\n```"
    p3 = "```edits\nhead.bias[2] = 3\n```\n" + END_MARKER
    units = [{"turn": 1, "file": "001.md", "cut_off": False, "parts": [
                {"file": "001.p1.md", "cut_off": True}, {"file": "001.p2.md", "cut_off": False}]},
             {"turn": 2, "file": "002.md", "cut_off": False, "parts": []}]
    d = _index_run(tmp_path / "r1", units, {"001.md": p1 + "\n" + p2, "001.p1.md": p1, "001.p2.md": p2,
                                            "002.md": p3})
    files, cuts, info = reading_units(d)
    assert [f.name for f in files] == ["001.p1.md", "001.p2.md", "002.md"] and cuts == [True, False, False]
    _, _, report = build_run(d)
    assert report["reading"]["mode"] == "index" and report["artifact"]["metadata"]["reading"] == "index"
    assert report["nonzero"]["head.bias"] == 3 and report["end_of_weights"]
    # an older index (parts as names): every part but the last counts as cut
    units[0]["parts"] = ["001.p1.md", "001.p2.md"]
    d2 = _index_run(tmp_path / "r2", units, {"001.md": p1 + "\n" + p2, "001.p1.md": p1, "001.p2.md": p2,
                                             "002.md": p3})
    assert reading_units(d2)[1] == [True, False, False]
    # inconsistencies refuse
    (d2 / "replies" / "001.md").write_text(p1 + p2)
    with pytest.raises(RunDirError, match="sha256"):
        reading_units(d2)
    d3 = _index_run(tmp_path / "r3", [{"turn": 1, "file": "001.md", "parts": []}], {"001.md": p3, "002.md": p3})
    with pytest.raises(RunDirError, match="lists"):
        reading_units(d3)
    d4 = _index_run(tmp_path / "r4", [{"turn": 1, "file": "001.md", "parts": [{"file": "001.p1.md"}]}],
                    {"001.md": "x", "001.p1.md": "y"})
    with pytest.raises(RunDirError, match="joined"):
        reading_units(d4)
