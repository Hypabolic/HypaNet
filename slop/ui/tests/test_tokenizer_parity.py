"""Python and JS tokenisers give identical ids, flags and windows (SPEC.md §12.2)."""

from __future__ import annotations

import json
import random
import subprocess

from conftest import SLOP, run_node
from ui import gen_unicode_tables
from ui import tokenizer as tk

TOK_DUMP = SLOP / "lint" / "tests" / "tok_dump.js"
VOCAB_TEXT = "save\nset\nnotif\nproject\naccount\nteam\nsearch\nnew\n"

TRICKY = [
    "", "   ", "OPERATION ON HOLD", "Resume recaptures the mouse.", "createEngine HTTPServer navigationTitle",
    "create_engine iPhone XMLHttpRequest utf8Encode", "\ue000 firstName \ue001 \ue000 HTTPServerError", "VAR ALT VARIANT ALTITUDE Var alt",
    "a\ue000b\ue001c \ue002\uf8ff \ufb01\ue000\uff21 \ue001\u0301x", "Press ALT then F4",
    "● Live — 3 new…", "Loading.... done..", "© 2026 Acme, Inc.™", "👍🏽🇬🇧❤️‍🔥⌚️⭐★✓✔︎", "Straße ẞ café naïve ﬁle ＡＢＣ ½ ²",
    "İstanbul ΣΟΦΊΑ σοφία ǅemal Ꭰ ꭰ ŉ", "Привет мир こんにちは 世界 مرحبا हिन्दी", "Don't they’re O'Neil 'quoted' “curly”",
    "zero​width soft­hyphen a‍b non‑breaking a‐b", "tab\there\nnew\x00line\x7f",
    "1,000 1,2345 12,34 3.14 .5 5. 05.50 007 00 0 2099 2100 1899", "v2.1.0 #4512 09:30–17:00 50% £1,299.99",
    "\U0001fae9 \U00010d50 \U000e0001 \ud800 ￿ \U0010ffff", "a" * 200, " ".join(["word"] * 40),
    "ABCs PDFs McDonald's LinkedIn macOS getHTTPResponseCode",
]


def _fuzz(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    pools = [
        (0x20, 0x7E), (0xA0, 0x24F), (0x300, 0x36F), (0x370, 0x3FF), (0x400, 0x4FF), (0x590, 0x6FF), (0x900, 0x97F),
        (0x1E00, 0x1EFF), (0x2000, 0x206F), (0x2190, 0x21FF), (0x2300, 0x23FF), (0x25A0, 0x27BF), (0x3040, 0x30FF),
        (0x4E00, 0x4E80), (0xFB00, 0xFB4F), (0xFE00, 0xFE0F), (0xFF00, 0xFFEF), (0x1D400, 0x1D4FF), (0x1F1E6, 0x1F1FF),
        (0x1F300, 0x1FAFF), (0x10D40, 0x10D8F), (0xE0020, 0xE007F), (0xE000, 0xE003),
    ]
    out = []
    for _ in range(n):
        chars = []
        for _ in range(rng.randint(1, 40)):
            a, b = rng.choice(pools)
            chars.append(chr(rng.randint(a, b)))
        out.append("".join(chars))
    return out


def _py_text(t: str, vocab: tk.Vocab) -> dict:
    tt = tk.tokenize_text(t, vocab)
    return {"ids": tt.ids, "flags": tt.flags, "dropped": tt.dropped, "stems": [x.stem for x in tt.tokens]}


# (vocab text or None, id layout): layout 2 moves the hash buckets, so both layouts are checked
CASES = [(None, 1), (VOCAB_TEXT, 1), (None, 2), (VOCAB_TEXT, 2)]


def test_text_parity_on_tricky_and_fuzzed_strings():
    texts = TRICKY + _fuzz(3000, 20261110)
    for vtext, ids in CASES:
        vocab = tk.EMPTY_VOCABS[ids] if vtext is None else tk.Vocab.from_text(vtext, ids)
        js = run_node(TOK_DUMP, {"vocab": vtext, "ids": ids, "texts": texts})["texts"]
        diffs = [(t, _py_text(t, vocab), j) for t, j in zip(texts, js) if _py_text(t, vocab) != j]
        print(f"\ntext parity ({'vocab' if vtext else 'no vocab'}, ids {ids}): {len(texts) - len(diffs)}/{len(texts)} strings identical")
        assert diffs == []


def test_screen_parity(test_screens):
    for vtext, ids in CASES:
        vocab = tk.EMPTY_VOCABS[ids] if vtext is None else tk.Vocab.from_text(vtext, ids)
        js = run_node(TOK_DUMP, {"vocab": vtext, "ids": ids, "screens": test_screens})["screens"]
        n_nodes = n_same = 0
        for s, j in zip(test_screens, js):
            enc = tk.encode_screen(s, vocab)
            py_nodes = [{"ids": n.text.ids, "flags": n.flags, "mask": n.flag_mask} for n in enc.nodes]
            n_nodes += len(py_nodes)
            n_same += sum(a == b for a, b in zip(py_nodes, j["nodes"]))
            assert py_nodes == j["nodes"], s["id"]
            assert [w.to_json() for w in enc.windows] == j["windows"], s["id"]
        print(f"\nscreen parity ({'vocab' if vtext else 'no vocab'}, ids {ids}): {len(test_screens)} screens, {n_same}/{n_nodes} nodes identical, windows identical")


def test_js_case_folding_matches_python_on_every_assigned_code_point():
    script = (
        "import {casefold} from './lint/tokenizer.js'; import fs from 'node:fs';"
        "const t=JSON.parse(fs.readFileSync('ui/unicode_tables.json','utf8')); const out=[];"
        "for (const [a,b] of t.assigned) for (let c=a;c<=b;c++) out.push(casefold(String.fromCodePoint(c)));"
        "process.stdout.write(JSON.stringify(out));"
    )
    res = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, cwd=SLOP, check=True)
    js = json.loads(res.stdout)
    py = [chr(c).casefold() for a, b in json.loads((SLOP / "ui" / "unicode_tables.json").read_text())["assigned"] for c in range(a, b + 1)]
    assert len(js) == len(py) > 140_000
    assert [i for i, (a, b) in enumerate(zip(py, js)) if a != b] == []


def test_unicode_tables_are_current():
    stored = json.loads((SLOP / "ui" / "unicode_tables.json").read_text(encoding="utf-8"))
    fresh = gen_unicode_tables.build()
    for key in ("assigned", "emoji", "casefold"):
        assert fresh[key] == stored[key], f"{key} changed: rerun ui/gen_unicode_tables.py"
