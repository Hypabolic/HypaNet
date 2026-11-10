"""Build a generated set: source files, the real extractor's screens, and span-joined label records.

    uv run python -m generator.build dev                  # 20,000 screens, seed 20261112      -> datasets/dev/
    uv run python -m generator.build val                  #  2,000 screens, seed 20261112 (val) -> datasets/val/
    uv run python -m generator.build test_gen             #  2,000 screens, seed 20261113; refuses unless
                                                          #  registry.yaml holds this generator's frozen hash
    uv run python -m generator.build dev --n 500 --out DIR   # a smaller build elsewhere (tests, smoke runs)

Each set directory gets ``screens.jsonl`` (one slop.screen/1 per line, as the extractor wrote it),
``<set>.gen.jsonl`` (label records, SPEC.md §5, labeller ``gen``), ``sources.jsonl`` (the generated files: path,
framework, idiom, source text), ``<set>.meta.jsonl`` (harness only: each screen's kind, whether it is clean, and its
attempt number; never shipped to a model), ``manifest.json`` (seeds, hashes, counts) and ``SHA256SUMS``. Source
paths are neutral: no screen kind appears in a path or a file name. The data files are gitignored; the manifest and the
sums are not. The same set, seed and generator give the same bytes.

The build stops with an error, writing nothing, if any planted span is not found as a node, if any node did not
come from a slot, or if a slot's span is used twice. A screen whose extracted roles contradict a planted pattern
or a pointer's precedence, or whose pointer crosses a window boundary, is regenerated with the next attempt
number; the manifest counts these.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "generator"

from ui import tokenizer as tk  # noqa: E402

from .compose import CLEAN_SHARE, FRAMEWORK_WEIGHTS, KIND_WEIGHTS, compose  # noqa: E402
from .core import PATTERN_CLASS, PLANTABLE, PLANTABLE_K, GenError, Rng, ident, seams, slug  # noqa: E402
from .join import CONTEXT_ROLES, check_labels, join_screen, node_record, run_extractor  # noqa: E402
from .render_html import HtmlRenderer  # noqa: E402
from .render_swift import SwiftRenderer  # noqa: E402
from .render_tsx import TsxRenderer  # noqa: E402

SLOP = Path(__file__).resolve().parents[1]
GEN_DIR = SLOP / "generator"
GUIDE = SLOP / "LABELLING.md"
REGISTRY = SLOP / "registry.yaml"
REPO = "slop-gen"
LABELLER = "gen"
GEN_TIME = "2026-11-12T00:00:00Z"  # fixed, so that labels are byte-identical across builds
MAX_ATTEMPTS = 8

SETS = {
    "dev": {"seed": 20261112, "namespace": "", "size": 20000},
    "val": {"seed": 20261112, "namespace": "val", "size": 2000},
    "test_gen": {"seed": 20261113, "namespace": "", "size": 2000},
}

IDIOM_WEIGHTS = {
    "tsx": [("tailwind", 40), ("cssmod", 20), ("shadcn", 25), ("inline", 15)],
    "html": [("bem", 45), ("tailwind", 30), ("inline", 25)],
}
SWIFT_IDIOMS = {
    "settings": [("form", 60), ("list", 20), ("stack", 20)],
    "app": [("list", 40), ("stack", 60)],
    "dialog": [("stack", 70), ("list", 30)],
    "empty": [("stack", 70), ("list", 30)],
}
EXT = {"tsx": "tsx", "html": "html", "swift": "swift"}
SRC_DIRS = {"tsx": ["components", "src/components", "src/ui", "ui", "src/views", "web/src", "lib/ui", "src/pages"],
            "html": ["templates", "partials", "site", "public", "views", "www"],
            "swift": ["Sources", "Sources/Views", "Views", "UI", "Screens", "Features"]}
KIND_WORDS = {"marketing", "app", "settings", "dialog", "empty", "dashboard", "game"}
# file-name words that would name a kind, and neutral replacements
NEUTRAL = {r"Settings(?![a-z])": "Prefs", r"settings(?![a-z])": "prefs", r"Dialog(?![a-z])": "Modal",
           r"Dashboard(?![a-z])": "Overview", r"Game(?![a-z])": "Play", r"Empty(?![a-z])": "Blank",
           r"Marketing(?![a-z])": "Promo", r"App(?![a-z])": "Main"}


def path_words(path: str) -> set:
    """Lower-case words of a path, split at separators and case changes (for the kind-name check)."""
    out = set()
    for part in re.split(r"[^A-Za-z0-9]+", path):
        for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+", part):
            out.add(w.lower())
    return out


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def generator_files() -> list[Path]:
    """The files whose hash is the generator's identity (code, tables and the extractor front end; not tests)."""
    return sorted(p for p in GEN_DIR.iterdir() if p.is_file() and p.suffix in (".py", ".ts"))


def generator_sha256() -> str:
    h = hashlib.sha256()
    for p in generator_files():
        h.update(f"{p.name}\0{sha256_file(p)}\n".encode())
    return h.hexdigest()


def extractor_sha256() -> str:
    h = hashlib.sha256()
    for p in sorted((SLOP / "extract" / "src").glob("*.ts")):
        h.update(f"{p.name}\0{sha256_file(p)}\n".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# one screen


def make_screen(seed: int, namespace: str, set_name: str, index: int, attempt: int) -> dict:
    """Compose and render screen ``index``. Deterministic in (seed, namespace, index, attempt)."""
    rng = Rng.for_screen(seed, namespace, index, attempt)
    kind = rng.wpick(KIND_WEIGHTS)
    fw = rng.wpick(FRAMEWORK_WEIGHTS[kind])
    clean = rng.random() < CLEAN_SHARE
    page, c = compose(rng, kind, fw, clean)
    if fw == "tsx":
        idiom = rng.wpick(IDIOM_WEIGHTS["tsx"])
        r = TsxRenderer(rng, idiom)
    elif fw == "html":
        idiom = rng.wpick(IDIOM_WEIGHTS["html"])
        r = HtmlRenderer(rng, idiom)
    else:
        idiom = rng.wpick(SWIFT_IDIOMS.get(kind, [("stack", 1)]))
        r = SwiftRenderer(rng, idiom)
    src, css = r.render(page)
    name = page.name.replace(" ", "")
    if fw == "tsx":
        name = ident(name)
    elif fw == "swift":
        name = ident(name) + ("" if name.endswith("View") else "View")
    else:
        name = slug(name)
    for w, alt in NEUTRAL.items():
        name = re.sub(w, alt, name)
    path = f"{set_name}/{index:05d}/{rng.pick(SRC_DIRS[fw])}/{name}.{EXT[fw]}"
    if path_words(path) & KIND_WORDS:
        raise GenError(f"source path names a screen kind: {path}")
    slots = page.slots() + list(getattr(r, "extra", []))
    return {"index": index, "attempt": attempt, "kind": kind, "framework": fw, "idiom": idiom, "clean": clean,
            "path": path, "src": src, "css": css, "slots": slots, "r19": page.r19}


def orient_duplicates(pairs) -> None:
    """An exact repeat at equal rank flags the later node in reading order (LABELLING.md §5). The composer cannot
    always know reading order (a SwiftUI navigation title is read first; a mapped literal sits above the markup), so
    when a planted exact repeat comes first, its label moves to its target and the two swap places."""
    from .join import rank

    node_of = {id(s): n for n, s in pairs}
    slots = [s for _, s in pairs]
    for n, s in pairs:
        t = s.target
        if s.cls != "ECHO" or t is None or id(t) not in node_of or s.text != t.text:
            continue
        tn = node_of[id(t)]
        if rank(n["role"], n["flags"]) != rank(tn["role"], tn["flags"]) or n["i"] > tn["i"]:
            continue
        if t.cls != "KEEP":
            continue
        keep_fields = (t.pattern, t.conf, t.ct, t.partial, list(t.aside))
        t.cls, t.pattern, t.target, t.conf, t.ct, t.partial = s.cls, s.pattern, s, s.conf, s.ct, s.partial
        t.aside = list(s.aside)
        t.equiv = [x for x in s.equiv]
        s.cls, s.target, s.equiv = "KEEP", None, []
        s.pattern, s.conf, s.ct, s.partial, s.aside = keep_fields
        for o in slots:
            if o is not t and o.target is t:
                o.target = s
        for o in slots:
            if o.equiv:
                o.equiv = [t if x is s else x for x in o.equiv]
        t.equiv = [x for x in t.equiv if x is not t and x is not s]


def window_problems(screen: dict, pairs) -> list:
    """Every pointer must stay inside its node's window (SPEC.md §4)."""
    enc = [tk.encode_node(i, n) for i, n in enumerate(screen["nodes"])]
    win_of = {}
    for w, idx in enumerate(tk.split_windows(enc)):
        for x in idx:
            win_of[x] = w
    index_of = {id(s): n["i"] for n, s in pairs}
    out = []
    for n, s in pairs:
        for ref in ([s.target] if s.target is not None else []) + list(s.equiv):
            j = index_of.get(id(ref))
            if j is not None and win_of[j] != win_of[n["i"]]:
                out.append(("cross_window", f"{s.pattern} {s.text!r}"))
    return out


def label_records(gen: dict, screen: dict, pairs, meta: dict) -> list[dict]:
    sid = screen["id"]
    slot_of = {tuple(n["loc"]): s for n, s in pairs}
    recs = []
    for n in screen["nodes"]:
        rec = node_record(n, slot_of[tuple(n["loc"])], sid, meta)
        if rec is not None:
            recs.append(rec)
    loc_of = {id(s): list(n["loc"]) for n, s in pairs}
    groups = []
    for g in gen["r19"]:
        locs = [loc_of[id(s)] for s in g if id(s) in loc_of]
        if len(locs) >= 2:
            groups.append({"code": "R19", "nodes": locs})
    if groups:
        recs.append({"screen": sid, "aside_screen": groups, **meta})
    labelled = sum(1 for r in recs if "loc" in r)
    recs.append({"screen": sid, "done": True, "nodes": labelled, **meta})
    return recs


# ---------------------------------------------------------------------------------------------------------------
# a set


def text_key(screen: dict) -> tuple[str, frozenset]:
    """Normalised node-text sequence hash and node-text set, for duplicate checks (SPEC.md §12.4)."""
    texts = [" ".join(n["text"].lower().split()) for n in screen["nodes"]]
    return sha256_bytes("\n".join(texts).encode()), frozenset(texts)


class Dupes:
    """Screens already used by other sets: exact normalised-text matches and node sets at Jaccard >= 0.8."""

    def __init__(self, screens: list[dict]):
        self.hashes = set()
        self.sets: list[frozenset] = []
        self.index: dict[str, list[int]] = {}
        for s in screens:
            h, ns = text_key(s)
            self.hashes.add(h)
            k = len(self.sets)
            self.sets.append(ns)
            for t in ns:
                self.index.setdefault(t, []).append(k)

    def hit(self, screen: dict) -> bool:
        h, ns = text_key(screen)
        if h in self.hashes:
            return True
        if not ns:
            return False
        counts: Counter = Counter()
        for t in ns:
            for k in self.index.get(t, ()):
                counts[k] += 1
        for k, inter in counts.items():
            if inter / (len(ns) + len(self.sets[k]) - inter) >= 0.8:
                return True
        return False


def build(set_name: str, n: int | None = None, out: Path | None = None, dupes: Dupes | None = None, log=print) -> dict:
    spec = SETS[set_name]
    seed, ns = spec["seed"], spec["namespace"]
    n = spec["size"] if n is None else n
    guide_sha = sha256_file(GUIDE)
    meta = {"labeller": LABELLER, "guide_sha256": guide_sha, "time": GEN_TIME}
    t0 = time.time()
    attempts = {i: 0 for i in range(n)}
    done: dict[int, tuple] = {}
    regen = Counter()
    pending = list(range(n))
    while pending:
        gens = [make_screen(seed, ns, set_name, i, attempts[i]) for i in pending]
        reqs = [{"repo": REPO, "path": g["path"], "src": g["src"], "css": g["css"]} for g in gens]
        res = []
        for k in range(0, len(reqs), 2000):
            res.extend(run_extractor(reqs[k:k + 2000]))
        retry = []
        for g, x in zip(gens, res):
            screen = x["screen"]
            j = join_screen(screen, g["slots"])
            fatal = j["problems"]
            if fatal:
                raise GenError(f"span join failed for {g['path']} (attempt {g['attempt']}): " +
                               "; ".join(f"{k}: {v}" for k, v in fatal[:10]))
            orient_duplicates(j["pairs"])
            soft = check_labels(j["pairs"], screen) + window_problems(screen, j["pairs"])
            # a template filled into bad grammar ("a order", "branchs", "details details") is regenerated
            soft += [("grammar_seam", f"{n['text']!r}: {x}") for n, _ in j["pairs"] for x in seams(n["text"])][:1]
            if dupes is not None and dupes.hit(screen):
                soft.append(("duplicate_of_dev_or_val", g["path"]))
            if soft:
                for k, _ in soft:
                    regen[k] += 1
                attempts[g["index"]] += 1
                if attempts[g["index"]] >= MAX_ATTEMPTS:
                    raise GenError(f"screen {g['index']} failed {MAX_ATTEMPTS} attempts: {soft[:5]}")
                retry.append(g["index"])
                continue
            done[g["index"]] = (g, x["line"], screen, j["pairs"])
        pending = retry
    gen_secs = time.time() - t0

    # write
    out = Path(out) if out else SLOP / "datasets" / set_name
    out.mkdir(parents=True, exist_ok=True)
    screens_lines, label_lines, source_lines, meta_lines = [], [], [], []
    counts = new_counts()
    for i in range(n):
        g, line, screen, pairs = done[i]
        screens_lines.append(line + "\n")
        source_lines.append(json.dumps({"screen": screen["id"], "path": g["path"], "framework": g["framework"],
                                        "idiom": g["idiom"], "src": g["src"], "css": g["css"]},
                                       ensure_ascii=False) + "\n")
        meta_lines.append(json.dumps({"screen": screen["id"], "index": g["index"], "kind": g["kind"], "clean": g["clean"],
                                      "attempt": g["attempt"]}, ensure_ascii=False) + "\n")
        recs = label_records(g, screen, pairs, meta)
        for r in recs:
            label_lines.append(json.dumps(r, ensure_ascii=False, separators=(", ", ": ")) + "\n")
        tally(counts, g, screen, recs)
    files = {"screens.jsonl": "".join(screens_lines), f"{set_name}.gen.jsonl": "".join(label_lines),
             "sources.jsonl": "".join(source_lines), f"{set_name}.meta.jsonl": "".join(meta_lines)}
    sums = {}
    for name, text in files.items():
        b = text.encode("utf-8")
        (out / name).write_bytes(b)
        sums[name] = sha256_bytes(b)
    manifest = {
        "schema": "slop.genset/1", "set": set_name, "seed": seed, "namespace": ns, "screens": n,
        "generator_sha256": generator_sha256(), "extractor_sha256": extractor_sha256(), "guide_sha256": guide_sha,
        "labeller": LABELLER, "label_time": GEN_TIME, "files": sums, "regenerated": dict(sorted(regen.items())),
        "counts": finish_counts(counts),
    }
    mtext = json.dumps(manifest, indent=1, ensure_ascii=False, sort_keys=False) + "\n"
    (out / "manifest.json").write_text(mtext, encoding="utf-8")
    sums["manifest.json"] = sha256_bytes(mtext.encode())
    (out / "SHA256SUMS").write_text("".join(f"{h}  {k}\n" for k, h in sorted(sums.items())), encoding="utf-8")
    log(f"{set_name}: {n} screens in {time.time() - t0:.1f}s (generate, extract and join {gen_secs:.1f}s) -> {out}")
    return manifest


def new_counts() -> dict:
    return {"kinds": Counter(), "frameworks": Counter(), "idioms": Counter(), "clean_screens": 0, "nodes": 0,
            "labelled": 0, "context": 0, "unresolved": 0, "classes": Counter(), "patterns": Counter(),
            "k_rules": Counter(), "asides": Counter(), "screen_asides": Counter(), "ct": Counter(), "partial": 0,
            "conf": Counter(), "echo_null_target": 0, "equiv_records": 0, "roles": Counter(),
            "screens_with_redundant": 0}


def tally(c: dict, g: dict, screen: dict, recs: list[dict]):
    c["kinds"][g["kind"]] += 1
    c["frameworks"][g["framework"]] += 1
    c["idioms"][f"{g['framework']}/{g['idiom']}"] += 1
    c["clean_screens"] += g["clean"]
    c["nodes"] += len(screen["nodes"])
    for nd in screen["nodes"]:
        c["roles"][nd["role"]] += 1
        if nd["role"] in CONTEXT_ROLES or "sr_only" in nd["flags"]:
            c["context"] += 1
    red = False
    for r in recs:
        if "loc" in r:
            if r["class"] is None:
                c["unresolved"] += 1
                continue
            c["labelled"] += 1
            c["classes"][r["class"]] += 1
            if r["class"] != "KEEP":
                red = True
                c["patterns"][f"{r['class']} {r['pattern']}"] += 1
                if r["class"] == "ECHO" and r["target"] is None:
                    c["echo_null_target"] += 1
            elif r["pattern"]:
                c["k_rules"][r["pattern"]] += 1
            for a in r["aside"]:
                c["asides"][a] += 1
            if r["ct"]:
                c["ct"][r["class"]] += 1
            c["partial"] += r["partial"]
            c["conf"][r["conf"]] += 1
            c["equiv_records"] += bool(r["equiv"])
        elif "aside_screen" in r:
            for it in r["aside_screen"]:
                c["screen_asides"][it["code"]] += 1
    c["screens_with_redundant"] += red


def finish_counts(c: dict) -> dict:
    out = {}
    for k, v in c.items():
        if isinstance(v, Counter):
            out[k] = dict(sorted(v.items(), key=lambda kv: _sort_key(kv[0])))
        else:
            out[k] = v
    out["patterns_missing"] = [f"{PATTERN_CLASS[p]} {p}" for p in PLANTABLE
                               if f"{PATTERN_CLASS[p]} {p}" not in c["patterns"]]
    out["k_rules_missing"] = [k for k in PLANTABLE_K if k not in c["k_rules"]]
    return out


def _sort_key(k):
    s = str(k)
    digits = "".join(ch for ch in s if ch.isdigit())
    return (s.split(" ")[0], int(digits) if digits else 0, s)


# ---------------------------------------------------------------------------------------------------------------


def registry_generator_sha256() -> str | None:
    """The frozen generator hash in ``registry.yaml`` (``generator: {sha256: …}``), or None."""
    if not REGISTRY.exists():
        return None
    import yaml

    reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or {}
    g = reg.get("generator")
    if isinstance(g, dict) and g.get("sha256") is not None:
        return str(g["sha256"])
    return None


def load_screens(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("set", choices=sorted(SETS))
    ap.add_argument("--n", type=int, help="number of screens (default: the set's size)")
    ap.add_argument("--out", help="output directory (default: datasets/<set>)")
    args = ap.parse_args(argv)
    if args.set == "test_gen":
        frozen = registry_generator_sha256()
        if frozen is None:
            print("refusing to build test_gen: slop/registry.yaml with a frozen generator hash (generator.sha256) "
                  "is required (SPEC.md §12.6)", file=sys.stderr)
            return 2
        if frozen != generator_sha256():
            print(f"refusing to build test_gen: the generator has changed since it was frozen "
                  f"(registry {frozen[:12]}, now {generator_sha256()[:12]})", file=sys.stderr)
            return 2
        prior = []
        for s in ("dev", "val"):
            p = SLOP / "datasets" / s / "screens.jsonl"
            if not p.exists():
                print(f"refusing to build test_gen: {p} is needed for the duplicate check (SPEC.md §12.4)",
                      file=sys.stderr)
                return 2
            prior.extend(load_screens(p))
        m = build("test_gen", args.n, Path(args.out) if args.out else None, dupes=Dupes(prior))
    else:
        m = build(args.set, args.n, Path(args.out) if args.out else None)
    print(json.dumps({"set": m["set"], "screens": m["screens"], "regenerated": m["regenerated"],
                      "patterns_missing": m["counts"]["patterns_missing"],
                      "k_rules_missing": m["counts"]["k_rules_missing"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
