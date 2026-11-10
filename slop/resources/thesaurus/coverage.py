"""Coverage of the thesaurus over screen copy: what share of word stems has a thesaurus entry.

    uv run python -I resources/thesaurus/coverage.py                       # dev + reference_real
    uv run python -I resources/thesaurus/coverage.py --top-missing 40      # also list dev's commonest missing stems

Reads ``out/stems.tsv`` and ``out/stem_links.tsv`` (build.py) and the screens' node text, tokenised by
``ui/tokenizer.py``. Sets: ``datasets/dev/screens.jsonl`` (generated) and ``datasets/reference_real/screens/*.json``
(real copy from the owner's repositories: only counts are printed for it, never words). It never reads
``datasets/val``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLOP = HERE.parents[1]
if str(SLOP) not in sys.path:
    sys.path.insert(0, str(SLOP))

from ui import tokenizer  # noqa: E402

OUT = HERE / "out"

# Closed-class words (as Porter stems) that no thesaurus lists as headwords; "content" shares leave them out.
FUNCTION_STEMS = frozenset(
    "the a an your you yourself we our ours us my me mine their theirs them they thei it its this thi that these those "
    "which what who whom whose wh of to in on at by for from with without into onto about as and or nor but if "
    "than then so not no is ar be been wa were do doe did ha have had will would can could shall should mai might "
    "must he she him her hi there here when where why how all ani each everi some".split()
)


def load_tables() -> tuple[set[str], set[str], set[str]]:
    wn, moby, linked = set(), set(), set()
    with (OUT / "stems.tsv").open(encoding="utf-8") as f:
        next(f)
        for line in f:
            stem, wn_words, _syns, moby_words, _root = line.rstrip("\n").split("\t")
            if wn_words:
                wn.add(stem)
            if moby_words:
                moby.add(stem)
    with (OUT / "stem_links.tsv").open(encoding="utf-8") as f:
        next(f)
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if cols[1] or cols[-1]:  # a WordNet synonym or a mutual Moby link
                linked.add(cols[0])
    return wn, moby, linked


def stems_of_screen(screen: dict, counts: Counter) -> None:
    for node in screen.get("nodes", []):
        for t in tokenizer.tokenize_text(node.get("text", "")).tokens:
            if t.kind == "word":
                counts[t.stem] += 1


def screens(name: str):
    if name == "dev":
        with (SLOP / "datasets" / "dev" / "screens.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif name == "reference_real":
        for p in sorted((SLOP / "datasets" / "reference_real" / "screens").glob("*.json")):
            yield json.loads(p.read_text(encoding="utf-8"))
    else:
        raise ValueError(name)


def summarise(counts: Counter, wn: set[str], moby: set[str], linked: set[str]) -> dict:
    content = Counter({k: v for k, v in counts.items() if k not in FUNCTION_STEMS})
    res: dict[str, object] = {"distinct_stems": len(counts), "word_tokens": sum(counts.values()),
                              "content_distinct_stems": len(content), "content_word_tokens": sum(content.values())}
    for label, s in (("wordnet", wn), ("moby", moby), ("either", wn | moby), ("has_synonym_link", linked)):
        row = {}
        for tag, c in (("all", counts), ("content", content)):
            d = sum(1 for k in c if k in s)
            t = sum(v for k, v in c.items() if k in s)
            row[tag] = {"distinct": d, "distinct_share": round(d / max(1, len(c)), 4),
                        "token_share": round(t / max(1, sum(c.values())), 4)}
        res[label] = row
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--top-missing", type=int, default=0, help="list dev's N commonest stems in neither source")
    args = ap.parse_args(argv)
    wn, moby, linked = load_tables()
    report: dict[str, object] = {}
    for name in ("dev", "reference_real"):
        counts: Counter = Counter()
        n = 0
        for sc in screens(name):
            stems_of_screen(sc, counts)
            n += 1
        report[name] = {"screens": n, **summarise(counts, wn, moby, linked)}
        if name == "dev" and args.top_missing:
            either = wn | moby
            report["dev_top_missing"] = [
                f"{k}:{v}" for k, v in counts.most_common() if k not in either and k not in FUNCTION_STEMS
            ][: args.top_missing]
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
