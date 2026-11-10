"""Porter stemmer against Martin Porter's published vocabulary and output (ui/tests/data/porter_SOURCE.txt)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ui.porter import stem

DATA = Path(__file__).resolve().parent / "data"
SLOP = Path(__file__).resolve().parents[2]


def _lists() -> tuple[list[str], list[str]]:
    voc = (DATA / "porter_voc.txt").read_text(encoding="ascii").split()
    out = (DATA / "porter_output.txt").read_text(encoding="ascii").split()
    assert len(voc) == len(out) == 23531
    return voc, out


def test_python_porter_matches_every_published_word():
    voc, out = _lists()
    wrong = [(w, e, stem(w)) for w, e in zip(voc, out) if stem(w) != e]
    print(f"\nPython Porter: {len(voc) - len(wrong)}/{len(voc)} words match")
    assert wrong == []


def test_js_porter_matches_every_published_word():
    script = (
        "import {stem} from './lint/porter.js'; import fs from 'node:fs';"
        "const v=fs.readFileSync('ui/tests/data/porter_voc.txt','ascii').split(/\\s+/).filter(Boolean);"
        "const o=fs.readFileSync('ui/tests/data/porter_output.txt','ascii').split(/\\s+/).filter(Boolean);"
        "let bad=[]; v.forEach((w,i)=>{ if (stem(w)!==o[i]) bad.push([w,o[i],stem(w)]); });"
        "console.log(JSON.stringify({n:v.length, bad}));"
    )
    res = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, cwd=SLOP, check=True)
    import json

    r = json.loads(res.stdout)
    print(f"\nJS Porter: {r['n'] - len(r['bad'])}/{r['n']} words match")
    assert r["n"] == 23531 and r["bad"] == []


def test_departures_of_the_reference_version():
    assert stem("possibly") == "possibl"  # bli -> ble, then step 5 drops the e
    assert stem("archaeology") == "archaeolog"  # logi -> log
    assert stem("as") == "as" and stem("is") == "is"  # length 1-2 left alone
    assert stem("news") == stem("new") == "new"
    assert stem("general") == stem("generate") == stem("generic") == "gener"
