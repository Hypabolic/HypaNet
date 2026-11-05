"""EDIT.md's ``## Mechanisms`` table (PREREGISTRATION §8): agent_loop_lab.v04.mechanisms. Pure text; no model."""

from __future__ import annotations

from agent_loop_lab.v04 import mechanisms as M

L = "model.language_model.layers"
A, B, C = f"{L}.3.mlp.down_proj.weight", f"{L}.7.self_attn.o_proj.weight", f"{L}.11.mlp.up_proj.weight"
HEAD = "| id | tensors | target class | claim |\n| --- | --- | --- | --- |\n"


def md(*rows: str, before: str = "# Edit\n\nText.\n\n", after: str = "") -> str:
    return before + M.HEADING + "\n\n" + HEAD + "".join(r + "\n" for r in rows) + after


def statuses(text: str, units) -> list[tuple[str, str]]:
    return [(r["id"], r["status"]) for r in M.preview(text, units)["rows"]]


def test_parse_reads_the_table_and_normalises_label_rows():
    p = M.parse(md(f"| M1 | `{A}`, {B} | stalled | x |", f"| M2 | label rows | **OSCILLATING** | y |",
                   f"| M3 | {M.EMBED} | RECOVERY | z |", after="\n## Notes\n\n| a | b | c | d |\n| --- | --- | --- | --- |\n"))
    assert p["found"] and p["problems"] == []
    assert [r["tensors"] for r in p["rows"]] == [[A, B], ["label rows"], ["label rows"]]
    assert [r["target_class"] for r in p["rows"]] == ["STALLED", "OSCILLATING", "RECOVERY"]
    assert M.edit_units([A, M.EMBED]) == sorted([A, "label rows"])


def test_missing_or_malformed_tables():
    assert M.preview("# Edit\n\nno table\n", [A, B])["q5_possible"] is False
    assert not M.parse("# Edit\n\n## Mechanism\n\n" + HEAD)["found"]  # the heading must be exact
    assert not M.parse(M.HEADING + "\n\n| id | tensor | class | claim |\n| --- | --- | --- | --- |\n")["found"]
    assert not M.parse(M.HEADING + "\n\nprose only\n")["found"]
    p = M.parse(md("| M1 | a | STALLED |", "| M1 | b | STALLED | x |", "| M1 | c | STALLED | y |"))
    assert any("3 cells" in x for x in p["problems"]) and any("duplicate" in x for x in p["problems"])
    assert M.preview(md(), [A, B])["q5_possible"] is False  # an empty table


def test_eligibility_follows_section_8():
    edit = [A, B, C, M.EMBED]
    text = md(f"| M1 | {A} | STALLED | x |", f"| M2 | {B}, {C} | OSCILLATING | y |", f"| M3 | label rows | RECOVERY | z |")
    assert statuses(text, edit) == [("M1", "evaluated"), ("M2", "evaluated"), ("M3", "evaluated")]
    # a row naming the whole edit, an unedited tensor, a non-class, no tensor
    rows = M.preview(md(f"| W | {A}, {B}, {C}, label rows | STALLED | x |", f"| U | {L}.0.mlp.gate_proj.weight | STALLED | x |",
                        f"| K | {A} | LOOPING | x |", "| E |  | STALLED | x |"), edit)["rows"]
    assert [r["status"] for r in rows] == ["ineligible"] * 4
    assert "whole edit" in rows[0]["reason"] and "does not change" in rows[1]["reason"]
    assert "not one of" in rows[2]["reason"] and "no tensor" in rows[3]["reason"]
    # overlapping rows are all ineligible, so table order cannot pick one
    assert statuses(md(f"| M1 | {A} | STALLED | x |", f"| M2 | {A}, {B} | STALLED | y |", f"| M3 | {C} | RECOVERY | z |"),
                    edit) == [("M1", "ineligible"), ("M2", "ineligible"), ("M3", "evaluated")]
    # a single-unit edit cannot support Q5
    assert M.preview(md(f"| M1 | {A} | STALLED | x |"), [A])["q5_possible"] is False


def test_only_the_first_eight_eligible_rows_are_evaluated():
    units = [f"{L}.{k}.mlp.down_proj.weight" for k in range(12)]
    text = md(*[f"| M{k} | {u} | STALLED | x |" for k, u in enumerate(units[:10])], "| BAD | x | STALLED | x |")
    st = [s for _, s in statuses(text, units)]
    assert st == ["evaluated"] * 8 + ["not evaluated"] * 2 + ["ineligible"]
    assert M.MAX_EVALUATED == 8


def test_escaped_pipes_and_extra_cells_keep_the_row_for_the_overlap_rule():
    """Review finding: a claim with a pipe split the row, parse() dropped it, and a row it overlapped was then
    credited. ``\\|`` is a pipe inside a cell; extra cells are read as the claim; a short row is ineligible but
    its tensors still count for the overlap rule."""
    edit = [A, B, C]
    escaped = md(f"| M1 | {A}, {B} | STALLED | A \\| B |", f"| M2 | {B} | STALLED | y |")
    p = M.parse(escaped)
    assert p["problems"] == [] and p["rows"][0]["claim"] == "A | B"
    assert statuses(escaped, edit) == [("M1", "ineligible"), ("M2", "ineligible")]
    extra = md(f"| M1 | {A}, {B} | STALLED | A | B |", f"| M2 | {B} | STALLED | y |")
    p = M.parse(extra)
    assert p["rows"][0]["claim"] == "A | B" and any("5 cells" in x for x in p["problems"])
    assert statuses(extra, edit) == [("M1", "ineligible"), ("M2", "ineligible")]
    short = M.preview(md(f"| M1 | {A}, {B} |", f"| M2 | {B} | STALLED | y |", f"| M3 | {C} | RECOVERY | z |"), edit)
    assert [(r["id"], r["status"]) for r in short["rows"]] == [("M1", "ineligible"), ("M2", "ineligible"),
                                                               ("M3", "evaluated")]
    assert "malformed" in short["rows"][0]["reason"] and "shares tensors with ['M1']" in short["rows"][1]["reason"]


def test_any_gfm_delimiter_row_is_a_table():
    for sep in ("|-|-|-|-|", "| - | :- | -: | :-: |", "|:---|---:|:---:|----|"):
        text = M.HEADING + "\n\n| id | tensors | target class | claim |\n" + sep + f"\n| M1 | {A} | STALLED | x |\n"
        assert M.parse(text)["found"] and M.preview(text, [A, B])["q5_possible"], sep
    assert not M.parse(M.HEADING + "\n\n| id | tensors | target class | claim |\n| x | - | - | - |\n")["found"]
