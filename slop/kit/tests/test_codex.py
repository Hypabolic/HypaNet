"""codex/: the slop codex shipped to the r2 author (CODEX.md, SOURCES.md, lists/)."""

from __future__ import annotations

import re
from pathlib import Path

SLOP = Path(__file__).resolve().parents[2]
CODEX = SLOP / "codex"
LISTS = sorted((CODEX / "lists").glob("*.txt"))


def entries(p: Path) -> list[str]:
    return [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def test_every_list_states_its_provenance_and_licence():
    assert len(LISTS) == 24
    for p in LISTS:
        head = "\n".join(ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.startswith("#"))
        assert re.search(r"^# Provenance:", head, re.M), p.name
        assert re.search(r"licen[cs]e|public domain|CC BY", head, re.I), p.name
        assert entries(p), p.name
    assert (CODEX / "lists" / "ATTRIBUTION-ui.md").is_file()
    excess = (CODEX / "lists" / "llm_excess_vocab.txt").read_text(encoding="utf-8")
    assert "CC BY-SA 4.0" in excess and "MIT" in excess


def test_catalogue_ids_index_and_list_counts():
    text = (CODEX / "CODEX.md").read_text(encoding="utf-8")
    heads = re.findall(r"^#### ([LU]\d\d) ", text, re.M)
    assert heads == [f"L{i:02d}" for i in range(1, 53)] + [f"U{i:02d}" for i in range(1, 53)]
    for i in range(1, 53):
        assert re.search(rf"^\| L{i:02d} \|", text, re.M) and re.search(rf"^\| U{i:02d} \|", text, re.M), i
    total = 0
    for p in LISTS:
        n = len(entries(p))
        total += n
        assert f"| `{p.name}` | {n} |" in text, p.name
    assert f"{total:,} entries in 24 files" in text
    assert "## 2. How the codex maps onto `LABELLING.md`'s classes and the rule layer" in text


def test_codex_names_no_internal_note_person_or_tool():
    # the codex ships to the author: nothing names the people, notes, labelling tools or held-out sets behind the kit
    bad = re.compile(r"Matthew|(?<![\w/])Codex(?!\w)|research/|SPEC|slop-private|test_gen|ood_codex|\bval\b|"
                     r"Fable|\*\*proposal\*\*")
    for p in [CODEX / "CODEX.md", CODEX / "SOURCES.md", *(CODEX / "lists").iterdir()]:
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            assert not bad.search(line.replace("SPECIAL", "")), f"{p.name}:{n}: {line}"
