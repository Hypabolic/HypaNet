"""Thesaurus resource checks: fetch pins refuse bad bytes, outputs parse, stems equal the tokeniser's, the manifest
matches the files. Output checks skip when ``out/`` has not been built (it is mostly gitignored)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ui import porter, tokenizer

HERE = Path(__file__).resolve().parents[1]
SLOP = HERE.parents[1]
OUT = HERE / "out"
MANIFEST = json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8"))



def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fetch = _load("slop_thesaurus_fetch", HERE / "fetch.py")
build = _load("slop_thesaurus_build", HERE / "build.py")

HEADERS = {
    "synsets.tsv": ["synset", "pos", "lexfile", "lemmas", "gloss"],
    "words.tsv": ["word", "pos", "toks", "n_words", "synsets", "source"],
    "related.tsv": ["synset", "rel", "target", "word", "target_word"],
    "moby.tsv": ["root", "toks", "n", "related"],
    "stems.tsv": ["stem", "wn_words", "wn_synsets", "moby_words", "moby_root"],
    "stem_links.tsv": ["stem", "wn_syn", "wn_similar", "wn_hyper", "wn_deriv", "wn_ant", "moby_mutual"],
}


def _rows(name: str):
    path = OUT / name
    if not path.exists():
        pytest.skip(f"{path} not built (run fetch.py and build.py)")
    with path.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        assert header == HEADERS[name]
        for n, line in enumerate(f, 2):
            assert line.endswith("\n")
            cols = line[:-1].split("\t")
            assert len(cols) == len(header), (name, n, line)
            yield cols


def _stem_of(word: str) -> str:
    toks = [t for t in tokenizer.tokenize_text(word).tokens]
    assert len(toks) == 1 and toks[0].kind == "word", word
    return toks[0].stem


# -- fetch ----------------------------------------------------------------------------------------------------------


def test_sources_are_pinned_https():
    assert {s.key for s in fetch.SOURCES} == {"oewn-2025", "moby-thesaurus-ii"}
    for s in fetch.SOURCES:
        assert len(s.sha256) == 64 and int(s.sha256, 16) >= 0
        assert s.size > 0 and s.urls and all(u.startswith("https://") for u in s.urls)


def test_verify_refuses_wrong_bytes(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    good = fetch.Source("k", "f.bin", hashlib.sha256(b"hello").hexdigest(), 5, ("https://x",), "-")
    fetch.verify(p, good)
    with pytest.raises(fetch.FetchError):
        fetch.verify(p, fetch.Source("k", "f.bin", "0" * 64, 5, ("https://x",), "-"))
    with pytest.raises(fetch.FetchError):
        fetch.verify(p, fetch.Source("k", "f.bin", good.sha256, 6, ("https://x",), "-"))


def test_manifest_sources_match_fetch_pins():
    pins = {s.key: s for s in fetch.SOURCES}
    for src in MANIFEST["sources"]:
        pin = pins[src["key"]]
        assert (src["sha256"], src["bytes"], src["urls"]) == (pin.sha256, pin.size, list(pin.urls))


def test_raw_files_match_pins_when_present():
    for s in fetch.SOURCES:
        path = fetch.RAW / s.key / s.filename
        if path.exists():
            fetch.verify(path, s)


# -- manifest -------------------------------------------------------------------------------------------------------


def test_manifest_lists_every_output_and_hashes_match():
    assert MANIFEST["schema"] == build.SCHEMA
    assert set(MANIFEST["outputs"]) == {f"out/{k}" for k in HEADERS}
    assert MANIFEST["stemmer"]["sha256"] == hashlib.sha256((SLOP / "ui" / "porter.py").read_bytes()).hexdigest()
    for rel, meta in MANIFEST["outputs"].items():
        assert meta["committed"] == (meta["bytes"] <= build.COMMIT_LIMIT)
        path = HERE / rel
        if not path.exists():
            assert not meta["committed"], f"{rel} is a committed output but missing"
            continue
        data = path.read_bytes()
        assert len(data) == meta["bytes"], rel
        assert hashlib.sha256(data).hexdigest() == meta["sha256"], rel
        assert data.count(b"\n") == meta["rows"] + 1, rel


def test_big_outputs_and_raw_are_gitignored():
    if shutil.which("git") is None:
        pytest.skip("git not available")
    paths = ["raw/oewn-2025/x"] + [rel for rel, m in MANIFEST["outputs"].items() if not m["committed"]]
    for rel in paths:
        r = subprocess.run(["git", "check-ignore", "-q", str(HERE / rel)], cwd=HERE, capture_output=True)
        assert r.returncode == 0, f"{rel} should be gitignored"
    for rel, m in MANIFEST["outputs"].items():
        if m["committed"]:
            r = subprocess.run(["git", "check-ignore", "-q", str(HERE / rel)], cwd=HERE, capture_output=True)
            assert r.returncode == 1, f"{rel} is small and should be committed, not ignored"


# -- formats and stems ------------------------------------------------------------------------------------------------


def test_render_matches_tokeniser_on_samples():
    assert build.render("Sign-in") == ("sign - in", 2)
    assert build.render("Settings") == ("set", 1)
    assert build.render("e-mail") == ("e - mail", 2)
    assert build.render("log in") == ("log in", 2)
    assert build.render("WiFi") == ("wi fi", 2)  # camel-case split, as the tokeniser does
    assert build.single_stem("deleted") == porter.stem("deleted") == "delet"
    assert build.single_stem("sign up") is None


def test_stems_match_tokeniser():
    stems = list(_rows("stems.tsv"))
    assert stems == sorted(stems, key=lambda r: r[0])
    rng = random.Random(0)
    for stem, wn_words, syns, moby_words, root in stems:
        assert stem and stem == stem.lower() and " " not in stem
        assert wn_words or moby_words
        assert root in ("0", "1")
        assert bool(syns) == bool(wn_words)
        words = [w for w in (wn_words.split("|") if wn_words else []) + (moby_words.split("|") if moby_words else [])]
        # every word of a short row, a sample of a long one
        for w in words if len(words) <= 8 else rng.sample(words, 8):
            assert _stem_of(w) == stem, (w, stem)


def test_words_toks_match_tokeniser():
    synsets = {r[0] for r in _rows("synsets.tsv")}
    rng = random.Random(1)
    n = 0
    for word, pos, toks, n_words, syns, source in _rows("words.tsv"):
        assert pos in {"n", "v", "a", "r", "s"}
        assert source == "lemma" or source.startswith("form:")
        assert syns and all(s in synsets for s in syns.split(","))
        if rng.random() < 0.05:
            assert build.render(word) == (toks, int(n_words)), word
            n += 1
    assert n > 1000


def test_synsets_and_related_parse():
    ids = set()
    for sid, pos, lexfile, lemmas, gloss in _rows("synsets.tsv"):
        assert sid[-2] == "-" and pos == sid[-1] and lemmas
        ids.add(sid)
    rels = set()
    for syn, rel, target, word, target_word in _rows("related.tsv"):
        assert syn in ids and target in ids
        rels.add(rel)
        assert bool(word) == bool(target_word)
    assert {"hypernym", "similar", "antonym", "derivation"} <= rels


def test_links_parse_and_point_at_stems():
    stems = {r[0] for r in _rows("stems.tsv")}
    for row in _rows("stem_links.tsv"):
        assert row[0] in stems
        for col in row[1:6]:
            for item in col.split() if col else []:
                other, count = item.rsplit(":", 1)
                assert other in stems and other != row[0] and int(count) >= 1
        for other in row[6].split() if row[6] else []:
            assert other in stems and other != row[0]


def test_moby_parses():
    n = 0
    for root, toks, count, related in _rows("moby.tsv"):
        items = related.split("|") if related else []
        assert int(count) == len(items) and root not in items
        n += 1
    assert n > 30000
