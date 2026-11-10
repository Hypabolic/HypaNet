"""Build the author-facing thesaurus tables (README.md) from the files ``fetch.py`` downloaded into ``raw/``.

    uv run python -I resources/thesaurus/fetch.py
    uv run python -I resources/thesaurus/build.py      # writes out/*.tsv and MANIFEST.json

Every word and lemma is keyed by the stems the slop tokeniser itself produces (``ui/tokenizer.py``
``tokenize_text``: NFKC, letter runs, camel-case split, case fold, accent strip, ``ui/porter.py``), so a stem here is
the same string that becomes a word id in the model. A lemma that tokenises to one word token is a *single-token*
word and is indexed by that stem in ``stems.tsv`` and ``stem_links.tsv``; a multi-word or hyphenated lemma keeps its
full token rendering in ``words.tsv`` / ``moby.tsv`` (``toks``) and is not indexed by stem.

The inputs are untrusted data: they are parsed (``xml.etree`` iterparse, refusing internal entity declarations;
plain text split on commas), never executed. Standard library only; run with ``python -I``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLOP = HERE.parents[1]
if str(SLOP) not in sys.path:  # python -I leaves the slop root off sys.path
    sys.path.insert(0, str(SLOP))

from ui import tokenizer  # noqa: E402


def _load_fetch():
    """fetch.py beside this file, under a unique module name (``fetch`` is too generic to put on sys.path)."""
    import importlib.util

    name = "slop_thesaurus_fetch"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "fetch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fetch = _load_fetch()

OUT = HERE / "out"
MANIFEST = HERE / "MANIFEST.json"
COMMIT_LIMIT = 5_000_000  # bytes: larger outputs are gitignored and rebuilt by build.py
SCHEMA = "slop.thesaurus/1"

# Synset relations kept (each direction as OEWN lists it; inverse duplicates such as hyponym are dropped).
SYNSET_RELS = (
    "hypernym", "instance_hypernym", "similar", "also", "attribute", "entails", "causes", "domain_topic",
    "mero_part", "mero_member", "mero_substance",
)
# Sense relations kept; "other" is OEWN's morphosemantic link and is written "morph.<dc:type>".
SENSE_RELS = ("antonym", "derivation", "pertainym", "participle", "also", "similar", "other")

_WS = re.compile(r"\s+")
_SEPARATORS = ("\t", "\n", "\r", "|")


# -- tokeniser rendering ------------------------------------------------------------------------------------------


_TOK_CACHE: dict[str, tuple[str, int]] = {}


def render(text: str) -> tuple[str, int]:
    """(toks, n_words): the tokeniser's view of ``text``.

    ``toks`` is space-separated: a word token is its stem, a punctuation token its character, a number ``#``,
    anything else (glyph, marker) ``<kind>``. ``n_words`` counts word tokens. A single-token word is n_words == 1 and
    a one-element ``toks``.
    """
    hit = _TOK_CACHE.get(text)
    if hit is not None:
        return hit
    parts: list[str] = []
    n_words = 0
    for t in tokenizer.tokenize_text(text).tokens:
        if t.kind == "word":
            parts.append(t.stem)
            n_words += 1
        elif t.kind == "punct":
            parts.append(t.surface if t.surface.isascii() else "-")
        elif t.kind == "number":
            parts.append("#")
        else:
            parts.append(f"<{t.kind}>")
    out = (" ".join(parts), n_words)
    _TOK_CACHE[text] = out
    return out


def single_stem(text: str) -> str | None:
    toks, n = render(text)
    return toks if n == 1 and " " not in toks else None


def clean(s: str) -> str:
    s = _WS.sub(" ", s).strip()
    for ch in _SEPARATORS:
        if ch in s:
            s = s.replace(ch, "/" if ch == "|" else " ")
    return s


# -- Open English WordNet -------------------------------------------------------------------------------------------


def short_id(xml_id: str) -> str:
    return xml_id[5:] if xml_id.startswith("oewn-") else xml_id


def parse_oewn(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        head = f.read(1 << 16)
    if b"<!ENTITY" in head:
        raise SystemExit(f"{path}: refusing a document that declares entities")

    entries: dict[str, dict] = {}  # entry id -> {lemma, pos, forms, senses: [(sense id, synset)]}
    sense_of: dict[str, tuple[str, str]] = {}  # sense id -> (synset, lemma)
    sense_rels: list[tuple[str, str, str]] = []  # (src sense, rel, dst sense)
    synsets: dict[str, dict] = {}
    syn_rels: list[tuple[str, str, str]] = []
    lexicon: dict[str, str] = {}

    with gzip.open(path, "rb") as f:
        for event, el in ET.iterparse(f, events=("start", "end")):
            tag = el.tag
            if event == "start":
                if tag == "Lexicon":
                    lexicon = {k: el.get(k, "") for k in ("id", "label", "version", "license", "url")}
                continue
            if tag == "LexicalEntry":
                lemma_el = el.find("Lemma")
                lemma = lemma_el.get("writtenForm")
                pos = lemma_el.get("partOfSpeech")
                forms = [fe.get("writtenForm") for fe in el.findall("Form")]
                senses = []
                for s in el.findall("Sense"):
                    sid, syn = s.get("id"), short_id(s.get("synset"))
                    senses.append((sid, syn))
                    sense_of[sid] = (syn, lemma)
                    for r in s.findall("SenseRelation"):
                        rel = r.get("relType")
                        if rel not in SENSE_RELS:
                            continue
                        if rel == "other":
                            rel = "morph." + (r.get("{https://globalwordnet.github.io/schemas/dc/}type") or "other")
                        sense_rels.append((sid, rel, r.get("target")))
                entries[el.get("id")] = {"lemma": lemma, "pos": pos, "forms": forms, "senses": senses}
                el.clear()
            elif tag == "Synset":
                sid = short_id(el.get("id"))
                d = el.find("Definition")
                synsets[sid] = {
                    "pos": el.get("partOfSpeech"),
                    "lexfile": el.get("lexfile", ""),
                    "members": el.get("members", "").split(),
                    "gloss": clean("".join(d.itertext())) if d is not None else "",
                }
                for r in el.findall("SynsetRelation"):
                    if r.get("relType") in SYNSET_RELS:
                        syn_rels.append((sid, r.get("relType"), short_id(r.get("target"))))
                el.clear()
    return {"entries": entries, "sense_of": sense_of, "sense_rels": sense_rels, "synsets": synsets,
            "syn_rels": syn_rels, "lexicon": lexicon}


# -- Moby Thesaurus II ----------------------------------------------------------------------------------------------


def parse_moby(path: Path) -> list[tuple[str, list[str]]]:
    raw = path.read_bytes().decode("latin-1")  # documented as ASCII; latin-1 never fails
    records: list[tuple[str, list[str]]] = []
    for line in raw.splitlines():
        items = [clean(x) for x in line.split(",")]
        items = [x for x in items if x]
        if not items:
            continue
        root, rel = items[0], items[1:]
        seen: set[str] = set()
        uniq = []
        for w in rel:
            if w != root and w not in seen:
                seen.add(w)
                uniq.append(w)
        records.append((root, uniq))
    return records


# -- output ---------------------------------------------------------------------------------------------------------


def write_tsv(name: str, header: list[str], rows: list[list[str]]) -> dict:
    path = OUT / name
    lines = ["\t".join(header)]
    for r in rows:
        assert len(r) == len(header), (name, r)
        assert all("\t" not in c and "\n" not in c for c in r), (name, r)
        lines.append("\t".join(r))
    data = ("\n".join(lines) + "\n").encode("utf-8")
    path.write_bytes(data)
    return {"rows": len(rows), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "committed": len(data) <= COMMIT_LIMIT}


def rank_join(d: dict[str, int], limit: int | None = None) -> str:
    items = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
    if limit is not None:
        items = items[:limit]
    return " ".join(f"{k}:{v}" for k, v in items)


def build() -> dict:
    srcs = {s.key: s for s in fetch.SOURCES}
    oewn_path = fetch.fetch(srcs["oewn-2025"], verify_only=True)
    moby_path = fetch.fetch(srcs["moby-thesaurus-ii"], verify_only=True)
    OUT.mkdir(exist_ok=True)
    print("parse OEWN ...", file=sys.stderr)
    wn = parse_oewn(oewn_path)
    print("parse Moby ...", file=sys.stderr)
    moby = parse_moby(moby_path)
    entries, synsets = wn["entries"], wn["synsets"]
    outputs: dict[str, dict] = {}
    stats: dict[str, object] = {}

    # synsets.tsv -----------------------------------------------------------------------------------------------
    entry_lemma = {eid: e["lemma"] for eid, e in entries.items()}
    rows = []
    for sid in sorted(synsets):
        s = synsets[sid]
        lemmas = [clean(entry_lemma[m]) for m in s["members"] if m in entry_lemma]
        rows.append([sid, s["pos"], s["lexfile"], "|".join(lemmas), s["gloss"]])
    outputs["synsets.tsv"] = write_tsv("synsets.tsv", ["synset", "pos", "lexfile", "lemmas", "gloss"], rows)

    # words.tsv -------------------------------------------------------------------------------------------------
    # one row per (written form, pos, source); synsets in the entry's sense order.
    word_rows: dict[tuple[str, str, str], list[str]] = {}
    stem_wn_words: dict[str, set[str]] = defaultdict(set)
    stem_wn_syn: dict[str, dict[str, int]] = defaultdict(dict)  # stem -> synset -> best sense rank
    syn_stems: dict[str, set[str]] = defaultdict(set)  # synset -> single-token member stems
    n_lemmas = n_multi = n_forms = 0
    for e in entries.values():
        syns = [syn for _, syn in e["senses"]]
        for form, source in [(e["lemma"], "lemma")] + [(fm, "form:" + e["lemma"]) for fm in e["forms"]]:
            w = clean(form)
            toks, nw = render(w)
            key = (w, e["pos"], clean(source))
            prev = word_rows.get(key)
            merged = syns if prev is None else list(dict.fromkeys(prev[4].split(",") + syns))
            word_rows[key] = [w, e["pos"], toks, str(nw), ",".join(merged), clean(source)]
            if source == "lemma":
                n_lemmas += 1
                n_multi += nw != 1 or " " in toks
            else:
                n_forms += 1
            st = single_stem(w)
            if st is None:
                continue
            stem_wn_words[st].add(w)
            for rank, syn in enumerate(syns):
                old = stem_wn_syn[st].get(syn)
                if old is None or rank < old:
                    stem_wn_syn[st][syn] = rank
                if source == "lemma":
                    syn_stems[syn].add(st)
    rows = [word_rows[k] for k in sorted(word_rows)]
    outputs["words.tsv"] = write_tsv("words.tsv", ["word", "pos", "toks", "n_words", "synsets", "source"], rows)
    stats["oewn"] = {
        "synsets": len(synsets), "lexical_entries": n_lemmas, "multi_token_lemmas": n_multi,
        "inflected_forms": n_forms, "single_token_stems": len(stem_wn_words),
    }

    # related.tsv -----------------------------------------------------------------------------------------------
    rows = []
    for src, rel, dst in wn["syn_rels"]:
        rows.append([src, rel, dst, "", ""])
    sense_of = wn["sense_of"]
    for src_sense, rel, dst_sense in wn["sense_rels"]:
        if src_sense not in sense_of or dst_sense not in sense_of:
            continue
        (s1, w1), (s2, w2) = sense_of[src_sense], sense_of[dst_sense]
        rows.append([s1, rel, s2, clean(w1), clean(w2)])
    rows = sorted(set(map(tuple, rows)))
    rows = [list(r) for r in rows]
    outputs["related.tsv"] = write_tsv("related.tsv", ["synset", "rel", "target", "word", "target_word"], rows)

    # Moby: moby.tsv and stem edges ----------------------------------------------------------------------------
    rows = []
    moby_words: dict[str, set[str]] = defaultdict(set)
    moby_root_stems: set[str] = set()
    moby_edges: dict[str, set[str]] = defaultdict(set)
    n_items = n_multi_items = 0
    distinct: set[str] = set()
    for root, rel in moby:
        toks, nw = render(root)
        rows.append([root, toks, str(len(rel)), "|".join(rel)])
        rs = single_stem(root)
        if rs is not None:
            moby_words[rs].add(root)
            moby_root_stems.add(rs)
        distinct.add(root)
        for w in rel:
            n_items += 1
            distinct.add(w)
            ws = single_stem(w)
            if ws is None:
                n_multi_items += 1
                continue
            moby_words[ws].add(w)
            if rs is not None and ws != rs:
                moby_edges[rs].add(ws)
    rows.sort(key=lambda r: r[0])
    outputs["moby.tsv"] = write_tsv("moby.tsv", ["root", "toks", "n", "related"], rows)
    multi_distinct = sum(1 for w in distinct if single_stem(w) is None)
    stats["moby"] = {
        "roots": len(moby), "related_items": n_items, "multi_token_items": n_multi_items,
        "distinct_terms": len(distinct), "distinct_multi_token_terms": multi_distinct,
        "single_token_stems": len(moby_words), "root_stems": len(moby_root_stems),
    }

    # stems.tsv -------------------------------------------------------------------------------------------------
    all_stems = sorted(set(stem_wn_words) | set(moby_words))
    rows = []
    for st in all_stems:
        syn_rank = stem_wn_syn.get(st, {})
        syns = sorted(syn_rank, key=lambda s: (syn_rank[s], s))
        rows.append([
            st, "|".join(sorted(stem_wn_words.get(st, ()))), ",".join(syns),
            "|".join(sorted(moby_words.get(st, ()))), "1" if st in moby_root_stems else "0",
        ])
    outputs["stems.tsv"] = write_tsv("stems.tsv", ["stem", "wn_words", "wn_synsets", "moby_words", "moby_root"], rows)
    stats["stems_total"] = len(all_stems)
    stats["stems_both"] = len(set(stem_wn_words) & set(moby_words))

    # stem_links.tsv --------------------------------------------------------------------------------------------
    # stem -> other single-token stems by relation, "stem:count" ranked by count.
    def stems_of(syn: str) -> set[str]:
        return syn_stems.get(syn, set())

    by_rel: dict[str, dict[str, dict[str, int]]] = {
        k: defaultdict(lambda: defaultdict(int)) for k in ("wn_syn", "wn_similar", "wn_hyper", "wn_deriv", "wn_ant")
    }
    for st, syn_rank in stem_wn_syn.items():
        for syn in syn_rank:
            for other in stems_of(syn):
                if other != st:
                    by_rel["wn_syn"][st][other] += 1
    syn_rel_map = {"similar": "wn_similar", "also": "wn_similar", "hypernym": "wn_hyper",
                   "instance_hypernym": "wn_hyper"}
    for src, rel, dst in wn["syn_rels"]:
        col = syn_rel_map.get(rel)
        if col is None:
            continue
        for a in stems_of(src):
            for b in stems_of(dst):
                if a != b:
                    by_rel[col][a][b] += 1
    sense_map = {"antonym": "wn_ant", "derivation": "wn_deriv", "pertainym": "wn_deriv", "participle": "wn_deriv",
                 "similar": "wn_similar", "also": "wn_similar"}
    for src_sense, rel, dst_sense in wn["sense_rels"]:
        col = sense_map.get(rel) or ("wn_deriv" if rel.startswith("morph.") else None)
        if col is None or src_sense not in sense_of or dst_sense not in sense_of:
            continue
        a, b = single_stem(clean(sense_of[src_sense][1])), single_stem(clean(sense_of[dst_sense][1]))
        if a and b and a != b:
            by_rel[col][a][b] += 1
    mutual = {a: {b for b in bs if a in moby_edges.get(b, ())} for a, bs in moby_edges.items()}
    link_cols = ["wn_syn", "wn_similar", "wn_hyper", "wn_deriv", "wn_ant"]
    keys = sorted(set().union(*(by_rel[c].keys() for c in link_cols)) | {a for a, bs in mutual.items() if bs})
    rows = []
    for st in keys:
        rows.append([st] + [rank_join(by_rel[c].get(st, {})) for c in link_cols]
                    + [" ".join(sorted(mutual.get(st, ())))])
    outputs["stem_links.tsv"] = write_tsv("stem_links.tsv", ["stem"] + link_cols + ["moby_mutual"], rows)
    stats["stem_links"] = {c: sum(1 for st in keys if by_rel[c].get(st)) for c in link_cols}
    stats["stem_links"]["moby_mutual"] = sum(1 for bs in mutual.values() if bs)

    porter_sha = hashlib.sha256((SLOP / "ui" / "porter.py").read_bytes()).hexdigest()
    manifest = {
        "schema": SCHEMA,
        "note": "Built by resources/thesaurus/build.py; see README.md. committed=false outputs are gitignored.",
        "stemmer": {"path": "ui/porter.py", "sha256": porter_sha, "via": "ui/tokenizer.py tokenize_text"},
        "sources": [
            {"key": s.key, "file": f"raw/{s.key}/{s.filename}", "sha256": s.sha256, "bytes": s.size,
             "urls": list(s.urls), "licence": s.licence}
            for s in fetch.SOURCES
        ],
        "oewn_lexicon": wn["lexicon"],
        "outputs": {f"out/{k}": v for k, v in sorted(outputs.items())},
        "stats": stats,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    try:
        m = build()
    except fetch.FetchError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    for name, o in m["outputs"].items():
        print(f"{name}: {o['rows']} rows, {o['bytes']:,} bytes, {'commit' if o['committed'] else 'gitignored'}")
    print(json.dumps(m["stats"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
