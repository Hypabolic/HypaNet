"""``./tok``: show what the slop tokeniser does (SPEC.md §13).

    tok.py FILE                 screen JSON: normalisation, stems, ids and flags per node, and the windows
    tok.py --text "TEXT"        one node's text
    tok.py --words FILE         word list (one or more words per line): word -> folded -> stem -> id
    tok.py --vocab VOCAB        checks a vocab.txt: duplicates, entries that are not clean stems (problems); entries
                                that change when stemmed again (notes: Porter is not idempotent, so a real stem such
                                as 'paus' can; the entry is live if some word stems to it, which --words shows); and
                                (with the lexicon beside it, or --lexicon DIR) every stem that more than one lexicon
                                word maps to. The problem count comes right after the header; each list gives its
                                count and 50 entries (``--offset N`` for the next ones)

``--vocab-file VOCAB`` makes the first three modes look ids up in that vocabulary (otherwise every stem shows its
hash bucket). The id layout (SPEC.md §6) is ``--ids 1|2``, else the one named by a ``meta.json`` beside the vocab
file (``"ids"``, absent = 1), else 1. ``--root DIR`` refuses file arguments that resolve outside DIR (the author kit passes its directory).
Output is capped at 20,000 characters (``--cap``) and says when it is cut.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import porter
    import tokenizer as tk
else:
    from . import porter
    from . import tokenizer as tk

CAP = 20_000
_SPECIAL_NAMES = {v: k for k, v in tk.IDS["special"].items()}
_GLYPH_NAMES = {v: k for k, v in tk.GLYPH_IDS.items()}
_PUNCT_NAMES = {tk.PUNCT_IDS[c]: c for c in tk.PUNCT_IDS}


def id_kind(tid: int, vocab: tk.Vocab) -> str:
    if tid in _SPECIAL_NAMES:
        return _SPECIAL_NAMES[tid]
    if tid in _GLYPH_NAMES:
        return _GLYPH_NAMES[tid]
    if tid in _PUNCT_NAMES:
        return f"punct {_PUNCT_NAMES[tid]!r}"
    if tid == tk.YEAR:
        return "YEAR"
    if tid == tk.NUM:
        return "NUM"
    if tid == tk.DEC:
        return "DEC"
    if tk.NUMBER_BASE <= tid < tk.NUMBER_BASE + 100:
        return f"number {tid - tk.NUMBER_BASE}"
    lay = vocab.layout
    if lay.vocab_base <= tid < lay.hash_base:
        return f"vocab line {tid - lay.vocab_base + 1}"
    if lay.hash_base <= tid < lay.vocab_size:
        return f"hash bucket {tid - lay.hash_base}"
    return "?"


def show_text(text: str, vocab: tk.Vocab, extra_flags: list[str] | None = None) -> list[str]:
    tt = tk.tokenize_text(text, vocab)
    flags = sorted(set(extra_flags or []) | set(tt.flags), key=tk.FLAGS.index)
    out = [f"  text   {text!r}", f"  flags  {' '.join(flags) or '-'}"]
    for t in tt.tokens:
        if t.kind == "word":
            out.append(f"    {t.surface!r:<24} -> {t.word!r} -> stem {t.stem!r:<16} id {t.id:>5}  {id_kind(t.id, vocab)}")
        else:
            out.append(f"    {t.surface!r:<24} -> {t.kind:<6} id {t.id:>5}  {id_kind(t.id, vocab)}")
    if tt.dropped:
        out.append(f"    ... {tt.dropped} more tokens cut (31-token cap; flag truncated)")
    out.append(f"    EON id {tk.EON}")
    return out


def show_screen(screen: dict, vocab: tk.Vocab) -> list[str]:
    enc = tk.encode_screen(screen, vocab)
    out = [f"screen {screen.get('id')}  nodes {len(enc.nodes)}  windows {len(enc.windows)}"]
    for n in enc.nodes:
        out.append(f"node {n.index} (i={n.i}) k={n.k} role={n.role} [{n.role_id}]")
        given = screen["nodes"][n.index].get("flags") or []
        out.extend(show_text(screen["nodes"][n.index]["text"], vocab, given))
    for wi, w in enumerate(enc.windows):
        out.append(f"window {wi}: nodes {w.nodes[0]}-{w.nodes[-1]} ({len(w.nodes)}), positions {len(w.tokens)}")
    return out


def show_words(lines: list[str], vocab: tk.Vocab) -> list[str]:
    out = []
    for line in lines:
        for word in line.split():
            tt = tk.tokenize_text(word, vocab)
            parts = [f"{t.word!r} -> {t.stem!r} -> {t.id} ({id_kind(t.id, vocab)})" if t.kind == "word"
                     else f"{t.surface!r} -> {t.id} ({id_kind(t.id, vocab)})" for t in tt.tokens]
            out.append(f"{word:<24} {'; '.join(parts) or '(no tokens)'}")
    return out


SHOW = 50  # entries of each --vocab list shown from --offset


def _page(name: str, items: list[str], offset: int, show: int) -> list[str]:
    """One --vocab list: its count, then ``show`` of its lines from ``offset``."""
    page = items[offset: offset + show]
    if not items:
        return [f"{name}: 0"]
    if len(items) <= show and offset == 0:
        return [f"{name}: {len(items)}", *page]
    shown = f"{offset + 1}-{offset + len(page)}" if page else "none"
    return [f"{name}: {len(items)}; showing {shown} (--offset N to page, {show} at a time)", *page]


def check_vocab(path: Path, lexicon_dir: Path | None, ids: int | None = None, offset: int = 0,
                show: int = SHOW) -> list[str]:
    """The problem count comes right after the header, so it survives the output cap; each list (problems, merges,
    notes) gives its count and then ``show`` entries from ``offset``."""
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lay = tk.layout(tk.layout_beside(path) if ids is None else ids)
    head = f"{path}: {len(lines)} lines (max {lay.vocab_max}, id layout {lay.version})"
    seen: dict[str, int] = {}
    problems: list[str] = []
    restem: list[tuple[int, str]] = []
    for n, s in enumerate(lines, start=1):
        if s in seen:
            problems.append(f"  duplicate: line {n} {s!r} repeats line {seen[s]}")
        else:
            seen[s] = n
        if not s or s != s.lower() or any(ch.isspace() for ch in s):
            problems.append(f"  not a clean stem (empty, upper case or white space): line {n} {s!r}")
            continue
        toks = tk.tokenize_text(s)
        words = [t for t in toks.tokens if t.kind == "word"]
        if len(toks.tokens) != 1 or len(words) != 1 or words[0].word != s:
            problems.append(f"  does not tokenise to itself as one word: line {n} {s!r} -> "
                            f"{[t.word if t.kind == 'word' else t.surface for t in toks.tokens]}")
        if porter.stem(s) != s:
            restem.append((n, s))
    if len(lines) > lay.vocab_max:
        problems.insert(0, f"  too many lines: {len(lines)} > {lay.vocab_max}")
    plist = _page("problems", problems, offset, show)
    out = [head, f"{len(problems)} problem(s)", *(plist if "showing" in plist[0] else plist[1:])]
    lex = lexicon_dir if lexicon_dir is not None else path.parent / "lexicon"
    by_stem: dict[str, set[str]] = defaultdict(set)
    if lex.is_dir():
        for f in sorted(lex.glob("*.txt")):
            for line in f.read_text(encoding="utf-8").splitlines():
                for word in line.split():
                    for t in tk.tokenize_text(word).tokens:
                        if t.kind == "word":
                            by_stem[t.stem].add(t.word)
        merges = {s: ws for s, ws in by_stem.items() if len(ws) > 1}
        out.append(f"lexicon {lex}: {sum(len(w) for w in by_stem.values())} words, {len(by_stem)} stems")
        out.extend(_page("merges (stems more than one lexicon word maps to)",
                         [f"  merge: {' '.join(sorted(merges[s]))} -> {s!r}"
                          f"{'' if s in seen else '  (stem not in vocab.txt)'}" for s in sorted(merges)], offset, show))
    else:
        out.append(f"no lexicon directory at {lex}")
    notes = []
    for n, s in restem:
        words = sorted(by_stem.get(s, ()))
        why = f"lexicon words stem to it: {' '.join(words)}" if words else "no lexicon word stems to it"
        notes.append(f"  changes when stemmed again: line {n} {s!r} -> {porter.stem(s)!r} ({why})")
    out.extend(_page("notes (entries that change when stemmed again; Porter is not idempotent: fine when some word "
                     "stems to the entry, check with --words)", notes, offset, show))
    return out


def _resolve(p: str, root: Path | None) -> Path:
    path = Path(p).resolve()
    if root is not None and root != path and root not in path.parents:
        raise SystemExit(f"tok: refusing {p}: outside {root}")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Show the slop tokeniser's output.")
    ap.add_argument("file", nargs="?", help="screen JSON")
    ap.add_argument("--text")
    ap.add_argument("--words")
    ap.add_argument("--vocab", help="check this vocab.txt")
    ap.add_argument("--lexicon", help="lexicon directory for --vocab (default: lexicon/ beside the vocab file)")
    ap.add_argument("--vocab-file", help="look ids up in this vocab.txt")
    ap.add_argument("--ids", type=int, choices=sorted(tk.LAYOUTS), help="id layout (default: the meta.json beside the "
                                                                         "vocab file, else 1)")
    ap.add_argument("--offset", type=int, default=0, help=f"--vocab: skip this many entries of each list ({SHOW} "
                                                           "are shown)")
    ap.add_argument("--root", help="refuse file arguments outside this directory")
    ap.add_argument("--cap", type=int, default=CAP)
    args = ap.parse_args(argv)
    root = Path(args.root).resolve() if args.root else None
    modes = [m for m in (args.file, args.text, args.words, args.vocab) if m is not None]
    if len(modes) != 1:
        ap.error("give exactly one of FILE, --text, --words, --vocab")
    if args.vocab_file:
        vocab = tk.Vocab.load(_resolve(args.vocab_file, root), args.ids)
    else:
        vocab = tk.EMPTY_VOCABS[args.ids or 1]
    if args.text is not None:
        lines = show_text(args.text, vocab)
    elif args.words is not None:
        lines = show_words(_resolve(args.words, root).read_text(encoding="utf-8").splitlines(), vocab)
    elif args.vocab is not None:
        lex = _resolve(args.lexicon, root) if args.lexicon else None
        lines = check_vocab(_resolve(args.vocab, root), lex, args.ids, max(args.offset, 0))
    else:
        lines = show_screen(json.loads(_resolve(args.file, root).read_text(encoding="utf-8")), vocab)
    text = "\n".join(lines) + "\n"
    if len(text) > args.cap:
        text = text[: args.cap] + f"\n[output truncated at {args.cap} characters of {len(text)}]\n"
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
