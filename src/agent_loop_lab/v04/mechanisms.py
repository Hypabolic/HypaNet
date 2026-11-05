"""The ``## Mechanisms`` table of a WSP ``EDIT.md`` (PREREGISTRATION §8; docs/V04_WSP_KERNEL.md §6). Harness only.

The table the prompts and ``packet/AUTHOR_INSTRUCTIONS.md`` ask for: a section headed exactly ``## Mechanisms``
holding one Markdown table with the columns :data:`COLUMNS`, in that order, one row per claimed mechanism:

* ``id``: a short unique id;
* ``tensors``: full checkpoint names, comma-separated (backticks allowed), or ``label rows`` for the four
  label-word rows of the embedding (``model.language_model.embed_tokens.weight`` means the same unit);
* ``target class``: one of the four class names;
* ``claim``: one sentence.

:func:`parse` reads it; :func:`eligibility` applies §8's rules against the units the edit changes (from
``BUILD_RECORD.json``: each edited tensor, the embedding as the one unit ``label rows``): a row is eligible
when its class is a class name, it names at least one unit, every unit is edited, its units are a **strict**
subset of the edit's and share no unit with any other row's (overlapping rows are all ineligible; a malformed
row is ineligible itself but its tensors still count for the overlap rule, so dropping it cannot credit a
row it overlaps). Cells split on unescaped pipes only (``\\|`` is a pipe inside a cell); a row with too many
cells reads the extra cells as part of its claim. The delimiter row is GFM's (``-``, ``---``, ``:-:``...); the
first
:data:`MAX_EVALUATED` eligible rows in table order are evaluated, later ones are not. No table, an empty table
or no eligible row means the edit cannot support Q5. ``scripts/v04_run_build.py`` writes this as a preview
(``ARCHIVE/MECHANISMS.json``, the eligibility report of PREREGISTRATION §8) and never refuses a build over it.
v0.4 gives no Q5 verdict: the statistic needs the sealed counterfactual split (§8).
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from ..schema import CLASS_NAMES

__all__ = ["HEADING", "COLUMNS", "LABEL_ROWS", "EMBED", "MAX_EVALUATED", "edit_units", "parse", "eligibility",
           "preview"]

HEADING = "## Mechanisms"
COLUMNS = ("id", "tensors", "target class", "claim")
LABEL_ROWS = "label rows"
EMBED = "model.language_model.embed_tokens.weight"
MAX_EVALUATED = 8  # PREREGISTRATION §8: each evaluated row costs one counterfactual readout
_SEP_CELL = re.compile(r"^:?-+:?$")  # GFM: one or more dashes, optional alignment colons
_PIPE = re.compile(r"(?<!\\)\|")  # a cell boundary: a pipe not escaped as \|


def _unit(name: str) -> str:
    n = name.strip().strip("`").strip()
    return LABEL_ROWS if n.lower() == LABEL_ROWS or n == EMBED else n


def edit_units(tensor_names: Iterable[str]) -> list[str]:
    """The units of an edit: its tensor names, with the embedding as ``label rows``."""
    return sorted({_unit(n) for n in tensor_names})


def _cells(line: str) -> list[str] | None:
    """The cells of a ``| a | b |`` row, split on unescaped pipes only (``\\|`` is a literal pipe in a cell)."""
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|") and not s.endswith("\\|") and len(s) > 1):
        return None
    return [c.strip().replace("\\|", "|") for c in _PIPE.split(s[1:-1])]


def parse(edit_md: str) -> dict[str, Any]:
    """``{"found", "rows": [{"id", "tensors", "target_class", "claim"}], "problems"}`` of the first table under the
    first ``## Mechanisms`` heading (up to the next heading)."""
    lines = edit_md.splitlines()
    heads = [i for i, x in enumerate(lines) if x.rstrip() == HEADING]
    problems: list[str] = []
    if not heads:
        return {"found": False, "rows": [], "problems": [f"no '{HEADING}' section"]}
    if len(heads) > 1:
        problems.append(f"{len(heads)} '{HEADING}' sections; the first is read")
    body = []
    for x in lines[heads[0] + 1:]:
        if x.startswith("#"):
            break
        body.append(x)
    table = [c for c in (_cells(x) for x in body) if c is not None]
    if len(table) < 2 or not all(_SEP_CELL.match(c) for c in table[1]):
        return {"found": False, "rows": [], "problems": problems + [f"no Markdown table under '{HEADING}'"]}
    header = [c.lower() for c in table[0]]
    if tuple(header) != COLUMNS:
        return {"found": False, "rows": [], "problems": problems + [f"the table's columns are {table[0]}, not "
                                                                    f"{list(COLUMNS)}"]}
    rows = []
    for n, cells in enumerate(table[2:], start=1):
        malformed = None
        if len(cells) > len(COLUMNS):  # an unescaped pipe in the claim: the rest of the row is the claim
            problems.append(f"row {n}: {len(cells)} cells, not {len(COLUMNS)}; cells 4.. read as the claim")
            cells = cells[:3] + [" | ".join(cells[3:])]
        elif len(cells) < len(COLUMNS):  # kept, so its tensors still count for the overlap rule
            problems.append(f"row {n}: {len(cells)} cells, not {len(COLUMNS)}")
            malformed = f"{len(cells)} cells, not {len(COLUMNS)}"
            cells = cells + [""] * (len(COLUMNS) - len(cells))
        rid, tensors, cls, claim = cells
        units = [] if not tensors.strip() else [_unit(t) for t in tensors.split(",") if t.strip()]
        rows.append({"row": n, "id": rid.strip("`"), "tensors": units, "target_class": cls.strip("`*").strip().upper(),
                     "claim": claim, **({"malformed": malformed} if malformed else {})})
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        problems.append("duplicate mechanism ids")
    return {"found": True, "rows": rows, "problems": problems}


def eligibility(rows: Sequence[Mapping[str, Any]], units: Iterable[str]) -> list[dict[str, Any]]:
    """Each row with ``status`` (``evaluated``, ``not evaluated`` or ``ineligible``) and, if ineligible, ``reason``."""
    edit = set(units)
    sets = [set(r["tensors"]) for r in rows]
    out, evaluated = [], 0
    for k, (r, t) in enumerate(zip(rows, sets)):
        overlap = [rows[j]["id"] for j, s in enumerate(sets) if j != k and s & t]
        reason = None
        if r.get("malformed"):
            reason = f"malformed row ({r['malformed']})"
        elif r["target_class"] not in CLASS_NAMES:
            reason = f"target class {r['target_class']!r} is not one of {list(CLASS_NAMES)}"
        elif not t:
            reason = "names no tensor"
        elif t - edit:
            reason = f"names tensors the edit does not change: {sorted(t - edit)}"
        elif t == edit:
            reason = "names the whole edit (no strict subset to ablate)"
        elif overlap:
            reason = f"shares tensors with {overlap}"
        if reason is None and evaluated < MAX_EVALUATED:
            evaluated += 1
            status = "evaluated"
        else:
            status = "ineligible" if reason else "not evaluated"
        out.append({**r, "status": status, **({"reason": reason} if reason else {})})
    return out


def preview(edit_md: str, tensor_names: Iterable[str]) -> dict[str, Any]:
    """:func:`parse` plus :func:`eligibility` against the edit's units, with ``q5_possible``."""
    p = parse(edit_md)
    units = edit_units(tensor_names)
    rows = eligibility(p["rows"], units)
    return {"units": units, "found": p["found"], "problems": p["problems"], "rows": rows,
            "evaluated": sum(r["status"] == "evaluated" for r in rows),
            "q5_possible": any(r["status"] == "evaluated" for r in rows)}
