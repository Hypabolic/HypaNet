"""Regenerate ``ui/unicode_tables.json`` (needs ``node`` on PATH).

The Python and JavaScript tokenisers must give identical ids, but Python 3.12 ships Unicode 15.0 tables and Node
ships whatever its ICU has. This file pins the three things that would otherwise drift:

- ``assigned``: code point ranges assigned in Python's Unicode version (category not ``Cn``, surrogates excluded).
  Anything outside them is a ``SYMBOL`` in both tokenisers and is never normalised.
- ``emoji``: ``Extended_Pictographic`` ranges at or above ``emoji_min_codepoint`` (from ``ids.json``), taken from
  Node's regex engine because Python's ``re`` has no such property.
- ``casefold``: the code points where Python's ``str.casefold`` differs from JavaScript's ``toLowerCase`` applied to
  the single code point. ``lint/tokenizer.js`` folds with ``toLowerCase`` plus this table.

Run ``uv run python ui/gen_unicode_tables.py`` and commit the result; ``ui/tests/test_tokenizer_parity.py`` checks
that the table still matches the local Node.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "unicode_tables.json"

_NODE_SCRIPT = r"""
const fs = require('fs');
const cps = JSON.parse(fs.readFileSync(0, 'utf8'));
const ep = /\p{Extended_Pictographic}/u;
const epRanges = [];
let start = -1;
for (let cp = 0; cp <= 0x110000; cp++) {
  const isEp = cp < 0x110000 && !(cp >= 0xd800 && cp <= 0xdfff) && ep.test(String.fromCodePoint(cp));
  if (isEp && start < 0) start = cp;
  if (!isEp && start >= 0) { epRanges.push([start, cp - 1]); start = -1; }
}
const lower = {};
for (const cp of cps) {
  const ch = String.fromCodePoint(cp);
  const lo = ch.toLowerCase();
  if (lo !== ch) lower[cp] = lo;
}
process.stdout.write(JSON.stringify({ep: epRanges, lower, unicode: process.versions.unicode}));
"""


def _ranges(pred) -> list[list[int]]:
    out: list[list[int]] = []
    start = -1
    for cp in range(0x110001):
        ok = cp <= 0x10FFFF and pred(cp)
        if ok and start < 0:
            start = cp
        elif not ok and start >= 0:
            out.append([start, cp - 1])
            start = -1
    return out


def _assigned(cp: int) -> bool:
    return unicodedata.category(chr(cp)) not in ("Cn", "Cs")


def build() -> dict:
    ids = json.loads((HERE / "ids.json").read_text(encoding="utf-8"))
    assigned = _ranges(_assigned)
    candidates = [cp for cp in range(0x110000) if _assigned(cp)]
    res = subprocess.run(["node", "-e", _NODE_SCRIPT], input=json.dumps(candidates), capture_output=True, text=True, check=True)
    node = json.loads(res.stdout)
    casefold = {}
    for cp in candidates:
        py = chr(cp).casefold()
        js = node["lower"].get(str(cp), chr(cp))
        if py != js:
            casefold[str(cp)] = py
    lo = ids["emoji_min_codepoint"]
    emoji = [[max(a, lo), b] for a, b in node["ep"] if b >= lo]
    return {
        "format": "slop.unicode_tables/1",
        "python_unicode": unicodedata.unidata_version,
        "node_unicode": node["unicode"],
        "assigned": assigned,
        "emoji": emoji,
        "casefold": casefold,
    }


def main() -> int:
    tables = build()
    OUT.write_text(json.dumps(tables, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(tables['assigned'])} assigned ranges, {len(tables['emoji'])} emoji ranges, "
          f"{len(tables['casefold'])} casefold exceptions (Python Unicode {tables['python_unicode']}, Node {tables['node_unicode']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
