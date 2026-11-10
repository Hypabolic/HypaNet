"""The author's ./tok, ./lint, ./score and ./inspect (SPEC §13), run on the harness side.

The author directory holds thin shell wrappers that call this file with the harness's uv environment:

    author_tool.py AUTHOR_DIR EXTRA_ROOT NODE tok|lint|score|inspect [author arguments ...]

AUTHOR_DIR, EXTRA_ROOT and NODE (the node binary) are fixed by the wrapper; EXTRA_ROOT is "-" in a real hand-over (a
TEST hand-over made with scripts/prepare_author_dir.sh --test-root DIR may also read from and write to DIR). The
author's own arguments come after the tool name and cannot change them. Every file argument is resolved (symbolic
links included) and refused unless it lies inside the author directory; writes (./lint --emit) only inside work/.
Output is capped at 20,000 characters and says when it is cut.

What runs: the tokeniser and the numpy model are the kit's own copies (kit/lib/, byte copies of ui/, hashed in
HANDOVER.json); ./tok's report is ui/tok.py's; ./lint is the real extractor (extract/), the real rule layer (rules/) and
the real JS runtime (lint/runtime.js, on the artefact compiled by lint/compile.py), through kit/tools/lint_run.mjs.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SLOP = Path(__file__).resolve().parents[2]
KIT_LIB = SLOP / "kit" / "lib"
LINT_RUN = SLOP / "kit" / "tools" / "lint_run.mjs"
sys.path.insert(0, str(KIT_LIB))
import model as mdl  # noqa: E402  (kit/lib copies; tokenizer and porter come with them)
import porter  # noqa: E402,F401
import tokenizer as tk  # noqa: E402

sys.path.insert(1, str(SLOP))
from lint import compile as net_compile  # noqa: E402


def _load_tok():
    """ui/tok.py (the ./tok report), bound to the kit/lib tokeniser and stemmer already imported above."""
    spec = importlib.util.spec_from_file_location("slop_tok_report", SLOP / "ui" / "tok.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.tk is tk, "ui/tok.py did not bind the kit/lib tokeniser"
    return mod


tokrep = _load_tok()

CAP = 20_000
SOURCE_EXT = (".tsx", ".jsx", ".js", ".mjs", ".html", ".htm", ".swift")
CLASSES = tuple(mdl.CLASSES)
REDUNDANT = ("ECHO", "NARRATE", "DECOR")
SHOW = 40
SHORT = {"KEEP": "KEEP", "ECHO": "ECHO", "NARRATE": "NARR", "DECOR": "DECOR"}


class ToolError(Exception):
    pass


# -- paths ---------------------------------------------------------------------------------------------------------


def _under(path: str, root: str) -> bool:
    root = root.rstrip("/")
    return path == root or path.startswith(root + "/")


class Roots:
    def __init__(self, author_dir: str, extra: str):
        self.author = os.path.realpath(author_dir)
        self.work = os.path.join(self.author, "work")
        self.extra = None if extra in ("", "-") else os.path.realpath(extra)

    def readable(self, p: str) -> bool:
        return _under(p, self.author) or (self.extra is not None and _under(p, self.extra))

    def writable(self, p: str) -> bool:
        return _under(p, self.work) or (self.extra is not None and _under(p, self.extra))

    def where(self, write: bool) -> str:
        base = "work/ in your directory" if write else "your directory"
        return base + (f" (or the test root {self.extra})" if self.extra else "")

    def resolve(self, p: str, *, write: bool = False, what: str = "file") -> Path:
        if not p:
            raise ToolError(f"empty {what} path")
        real = os.path.realpath(os.path.join(os.getcwd(), p))
        ok = self.writable(real) if write else self.readable(real)
        if not ok:
            raise ToolError(f"refusing {what} {p!r}: it resolves outside {self.where(write)}")
        return Path(real)

    def rel(self, p: Path) -> str:
        s = str(p)
        if _under(s, self.author):
            return os.path.relpath(s, self.author)
        return s


# -- shared helpers ------------------------------------------------------------------------------------------------


def run_node(node: str, req: dict) -> dict:
    r = subprocess.run([node, "--no-warnings", str(LINT_RUN)], input=json.dumps(req), capture_output=True, text=True,
                       cwd=str(SLOP))
    if r.returncode != 0:
        msg = (r.stderr.strip().splitlines() or ["(no message)"])
        tail = [ln for ln in msg if ln.strip()][-3:]
        raise ToolError("the extractor / linter failed: " + " | ".join(tail))
    return json.loads(r.stdout)


def load_screen_file(path: Path, roots: Roots, node: str) -> dict:
    """A screen JSON, or a source file run through the real extractor."""
    if path.is_dir():
        raise ToolError(f"{roots.rel(path)} is a directory")
    if not path.is_file():
        raise ToolError(f"no such file: {roots.rel(path)}")
    if path.suffix.lower() == ".json":
        try:
            screen = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ToolError(f"{roots.rel(path)} is not JSON: {e}") from None
        if not isinstance(screen, dict) or screen.get("schema") != "slop.screen/1":
            raise ToolError(f"{roots.rel(path)} is not a slop.screen/1 document")
        check_screen(screen, roots.rel(path))
        return screen
    if path.suffix.lower() == ".py":
        raise ToolError(".py files (comment pairs, track C) are not read in this run")
    if path.suffix.lower() not in SOURCE_EXT:
        raise ToolError(f"{roots.rel(path)}: give a screen JSON or a .tsx, .jsx, .html or .swift file")
    out = run_node(node, {"file": str(path), "root": str(path.parent), "relpath": roots.rel(path), "screen": None,
                          "net": None, "rules": False})
    return out["screen"]


def check_screen(screen: dict, name: str) -> None:
    """The fields the tools read, with their types (KIT.md §4); a ToolError names the first bad node."""
    nodes = screen.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ToolError(f"{name}: \"nodes\" must be a non-empty list")
    if not isinstance(screen.get("clusters", []), list):
        raise ToolError(f"{name}: \"clusters\" must be a list")
    seen = set()
    for x, n in enumerate(nodes):
        where = f"{name}: node {x}"
        if not isinstance(n, dict):
            raise ToolError(f"{where} is not an object")
        for key, typ in (("i", int), ("k", int), ("role", str), ("text", str)):
            if not isinstance(n.get(key), typ) or isinstance(n.get(key), bool):
                raise ToolError(f"{where}: \"{key}\" missing or not {'an integer' if typ is int else 'a string'}")
        loc = n.get("loc")
        if not (isinstance(loc, list) and len(loc) == 4 and all(isinstance(v, int) and not isinstance(v, bool)
                                                                 for v in loc)):
            raise ToolError(f"{where}: \"loc\" must be [line, col, endLine, endCol] (four integers)")
        if n["role"] not in tk.ROLE_IDS:
            raise ToolError(f"{where}: unknown role {n['role']!r} (KIT.md §4 lists the roles)")
        flags = n.get("flags", [])
        if not isinstance(flags, list) or any(f not in tk.FLAG_IDS for f in flags):
            raise ToolError(f"{where}: \"flags\" must be a list of flag names from KIT.md §4")
        if n["i"] in seen:
            raise ToolError(f"{where}: i = {n['i']} is used twice")
        seen.add(n["i"])


def load_art(p: str, roots: Roots) -> tuple[mdl.Artifact, Path]:
    path = roots.resolve(p, what="artefact")
    d = path.parent if path.is_file() else path
    if not d.is_dir():
        raise ToolError(f"no artefact directory at {p!r}")
    try:
        return mdl.load_artifact(d), d
    except (mdl.ArtifactError, tk.TokenizerError, OSError, ValueError) as e:
        raise ToolError(f"artefact {roots.rel(d)}: {e}") from None


def short(text: str, n: int = 48) -> str:
    t = text.replace("", "⟨VAR⟩").replace("", "⟨ALT⟩").replace("\n", " ")
    return json.dumps(t if len(t) <= n else t[: n - 1] + "…", ensure_ascii=False)


def loc_str(loc) -> str:
    return f"{loc[0]}:{loc[1]}"


def margin(logits) -> float:
    return float(max(logits[1:]) - logits[0])


# -- ./tok ---------------------------------------------------------------------------------------------------------


def cmd_tok(argv: list[str], roots: Roots, node: str, out: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="./tok", description="What the slop tokeniser does with a screen or a text (KIT.md §6, §10).")
    ap.add_argument("file", nargs="?", help="screen JSON, or a .tsx/.jsx/.html/.swift file (extracted first)")
    ap.add_argument("--text", help="one node's text")
    ap.add_argument("--words", metavar="FILE", help="word list: word -> folded -> stem -> id")
    ap.add_argument("--vocab", metavar="VOCAB", help="check a vocab.txt (duplicates, unclean entries, entries that "
                                                     "change when stemmed again, lexicon merges)")
    ap.add_argument("--lexicon", metavar="DIR", help="lexicon directory for --vocab (default: lexicon/ beside it)")
    ap.add_argument("--vocab-file", metavar="VOCAB", help="look ids up in this vocab.txt (without it every stem "
                                                          "shows its hash bucket)")
    ap.add_argument("--node", type=int, metavar="K", help="with FILE: only the node whose i is K (and the windows)")
    a = ap.parse_args(argv)
    modes = [m for m in (a.file, a.text, a.words, a.vocab) if m is not None]
    if len(modes) != 1:
        ap.error("give exactly one of FILE, --text, --words, --vocab")
    if a.node is not None and a.file is None:
        ap.error("--node needs FILE")
    if a.vocab is not None:
        lex = roots.resolve(a.lexicon, what="lexicon directory") if a.lexicon else None
        out.extend(tokrep.check_vocab(roots.resolve(a.vocab, what="vocab file"), lex))
        return 0
    vpath = roots.resolve(a.vocab_file, what="vocab file") if a.vocab_file else None
    try:
        vocab = tk.Vocab.load(vpath) if vpath else tk.EMPTY_VOCAB
    except tk.TokenizerError as e:
        raise ToolError(f"{vpath}: {e}") from None
    out.append(f"ids from {roots.rel(vpath) if vpath else 'no vocabulary (hash buckets only)'}")
    if a.text is not None:
        out.extend(tokrep.show_text(a.text, vocab))
    elif a.words is not None:
        p = roots.resolve(a.words, what="word list")
        out.extend(tokrep.show_words(p.read_text(encoding="utf-8").splitlines(), vocab))
    else:
        screen = load_screen_file(roots.resolve(a.file), roots, node)
        lines = tokrep.show_screen(screen, vocab)
        if a.node is not None:
            if not any(n["i"] == a.node for n in screen["nodes"]):
                raise ToolError(f"no node with i = {a.node}")
            keep, on = [], False
            for ln in lines:
                if ln.startswith("node "):
                    on = f"(i={a.node})" in ln.split()[2]
                elif ln.startswith(("window ", "screen ")):
                    on = True
                if on:
                    keep.append(ln)
            lines = keep
        out.extend(lines)
    return 0


# -- ./lint --------------------------------------------------------------------------------------------------------

VERB = {"ECHO": "restates", "NARRATE": "explains"}


def cmd_lint(argv: list[str], roots: Roots, node: str, out: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="./lint", description="The real linter on one file: extractor, network (your artefact, through the JS "
                                   "runtime) and the plain-code rule layer (KIT.md §2, §10).")
    ap.add_argument("file", help="screen JSON, or a .tsx/.jsx/.html/.swift file in your directory")
    ap.add_argument("--art", help="artefact directory (default work/ when work/artifact.npz exists)")
    ap.add_argument("--no-net", action="store_true", help="rule layer only")
    ap.add_argument("--no-rules", action="store_true", help="network only")
    ap.add_argument("--nodes", action="store_true", help="also list every node with its role, flags and call")
    ap.add_argument("--emit", metavar="OUT.json", help="also write the extracted screen JSON here (inside work/)")
    ap.add_argument("--format", choices=("text", "json"), default="text")
    a = ap.parse_args(argv)
    path = roots.resolve(a.file)
    if not path.is_file():
        raise ToolError(f"no such file: {a.file}")
    dst = roots.resolve(a.emit, write=True, what="--emit path") if a.emit else None
    if dst is not None and not str(dst).endswith(".json"):
        raise ToolError("--emit must end in .json")
    screen = load_screen_file(path, roots, node)
    art_dir = None
    if not a.no_net:
        if a.art:
            _, art_dir = load_art(a.art, roots)
        elif (Path(roots.work) / "artifact.npz").is_file():
            _, art_dir = load_art(roots.work, roots)
    net_file = None
    try:
        if art_dir is not None:
            art = mdl.load_artifact(art_dir)
            data = net_compile.compile_artifact(art, (art_dir / "vocab.txt").read_bytes())
            fd, net_file = tempfile.mkstemp(suffix=".slop-net.bin")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        res = run_node(node, {"file": None, "root": None, "relpath": None, "screen": screen, "net": net_file,
                              "rules": not a.no_rules})
    finally:
        if net_file:
            os.unlink(net_file)
    if dst is not None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(screen, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    nodes = screen["nodes"]
    preds = res["preds"]
    net_findings = []
    if preds is not None:
        for p in preds:
            if p["cls"] == "KEEP":
                continue
            n = nodes[p["index"]]
            f = {"file": roots.rel(path), "line": n["loc"][0], "col": n["loc"][1], "class": p["cls"],
                 "role": n["role"], "text": n["text"], "margin": margin(p["logits"]), "rule": "slop/redundant-copy",
                 "network": res["network"], "target": None}
            if p["pointer"] is not None:
                t = nodes[p["pointer"]]
                f["target"] = {"line": t["loc"][0], "col": t["loc"][1], "role": t["role"], "text": t["text"]}
            net_findings.append(f)
    if a.format == "json":
        out.append(json.dumps({"file": roots.rel(path), "screen": screen.get("id"), "network": res["network"],
                               "findings": net_findings, "rules": res["findings"], "preds": preds},
                              ensure_ascii=False))
        return 0
    enc = tk.encode_screen(screen)
    out.append(f"{roots.rel(path)}  (screen {screen.get('id')}, {len(nodes)} nodes, {len(enc.windows)} "
               f"window{'s' if len(enc.windows) != 1 else ''})")
    for f in net_findings:
        at = f"{f['line']}:{f['col']}"
        msg = f"{f['role']} {short(f['text'])}"
        if f["target"] is not None and f["class"] in VERB:
            t = f["target"]
            msg += f" {VERB[f['class']]} {t['role']} {short(t['text'])} ({t['line']}:{t['col']})"
        elif f["class"] == "DECOR":
            msg += " carries no information"
        out.append(f"  {at:<7} {f['class']:<8} {msg}   margin {f['margin']:+.3f}   slop/redundant-copy")
    for x in res["findings"]:
        at = loc_str(x["locs"][0]) if x.get("locs") else "?"
        more = f" (+{len(x['locs']) - 1} more)" if len(x.get("locs") or []) > 1 else ""
        out.append(f"  {at:<7} {x['code']:<8} {x['message']}{more}   {x['rule']}")
    if a.nodes:
        out.append("nodes:")
        by_index = {p["index"]: p for p in preds or []}
        for x, n in enumerate(nodes):
            p = by_index.get(x)
            call = "-"
            if p is not None:
                call = p["cls"] + (f"→{nodes[p['pointer']]['i']}" if p["pointer"] is not None else "")
                call += " [" + " ".join(f"{v:+.3f}" for v in p["logits"]) + "]"
            out.append(f"  {n['i']} {n['k']} {loc_str(n['loc'])} {n['role']} {','.join(n.get('flags') or []) or '-'}"
                       f" | {short(n['text'], 60)} | {call}")
    if preds is None:
        why = "--no-net" if a.no_net else "no artefact (none given and no work/artifact.npz)"
        net = f"network skipped: {why}"
    else:
        net = f"network {res['network'][:8]} · {res['ms']:.0f} ms"
    out.append(f"{len(net_findings)} network finding{'s' if len(net_findings) != 1 else ''}, "
               f"{len(res['findings'])} rule finding{'s' if len(res['findings']) != 1 else ''} · {net}")
    return 0


# -- labels and scoring --------------------------------------------------------------------------------------------


class Item:
    """One screen to score: its JSON, its label records (or None) and a display name."""

    def __init__(self, name: str, screen: dict, labels: list[dict] | None):
        self.name = name
        self.screen = screen
        self.labels = labels
        self.by_loc = {tuple(n["loc"]): x for x, n in enumerate(screen["nodes"])}

    def node_labels(self) -> dict[int, dict]:
        out = {}
        for r in self.labels or []:
            if "class" in r and tuple(r["loc"]) in self.by_loc:
                out[self.by_loc[tuple(r["loc"])]] = r
        return out


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def base_name(path: Path) -> str:
    """X for X.screen.json or X.json: the name its labels file shares (X.labels.jsonl)."""
    n = path.name
    return n[: -len(".screen.json")] if n.endswith(".screen.json") else path.stem


def set_items(name: str, roots: Roots) -> list[Item]:
    d = Path(roots.author) / "examples" / name
    files = sorted(d.glob("*.screen.json"))
    if name == "reference" and not any(d.glob("*.labels.jsonl")):
        raise ToolError("the reference screens are unlabelled in this kit (examples/README.md); run "
                        "./score ART --files examples/reference/<id>.screen.json to see the network's calls")
    items = []
    for f in files:
        lab = f.with_name(base_name(f) + ".labels.jsonl")
        items.append(Item(base_name(f), json.loads(f.read_text(encoding="utf-8")),
                          read_jsonl(lab) if lab.is_file() else []))
    if not items:
        raise ToolError(f"no screens in examples/{name}/")
    return items


def file_items(paths: list[str], roots: Roots, node: str) -> list[Item]:
    items = []
    for p in paths:
        path = roots.resolve(p)
        screen = load_screen_file(path, roots, node)
        labels = None
        lab = path.with_name(base_name(path) + ".labels.jsonl") if path.suffix == ".json" else None
        if lab is not None and lab.is_file():
            labels = read_jsonl(roots.resolve(str(lab)))
        elif isinstance(screen.get("labels"), list):
            labels = screen["labels"]
        items.append(Item(roots.rel(path), screen, labels))
    return items


def predict(art: mdl.Artifact, items: list[Item]) -> list[list[mdl.NodePrediction]]:
    out = []
    for it in items:
        try:
            out.append(mdl.predict_screen(art, it.screen))
        except (tk.TokenizerError, ValueError, KeyError) as e:
            raise ToolError(f"{it.name}: {e}") from None
    return out


def target_ok(item: Item, rec: dict, ptr: int | None) -> bool:
    if ptr is None:
        return False
    want = {item.by_loc.get(tuple(rec["target"]))} | {item.by_loc.get(tuple(e)) for e in rec.get("equiv") or []}
    return ptr in want


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def describe_label(item: Item, rec: dict) -> str:
    nodes = item.screen["nodes"]
    s = rec["class"] if rec["class"] is not None else "UNRESOLVED"
    if rec["class"] in ("ECHO", "NARRATE"):
        t = item.by_loc.get(tuple(rec["target"])) if rec.get("target") else None
        s += "→" + (str(nodes[t]["i"]) if t is not None else "-")
    if rec.get("pattern"):
        s += " " + rec["pattern"]
    if rec.get("ct"):
        s += " ct"
    return s


def describe_pred(item: Item, p: mdl.NodePrediction) -> str:
    s = p.cls
    if p.pointer is not None:
        s += "→" + str(item.screen["nodes"][p.pointer]["i"])
    return s + f" ({margin(p.logits):+.2f})"


def score_report(items: list[Item], preds, a, out: list[str]) -> None:
    conf = {g: np.zeros((4, 4), dtype=np.int64) for g in ("head", "ct")}
    skipped = Counter()
    pat = defaultdict(lambda: [0, 0, 0, 0])  # n, flagged, exact class, of which ct
    krule = defaultdict(lambda: [0, 0])
    ptr = [0, 0]
    wrong = []
    for it, pr in zip(items, preds):
        labs = it.node_labels()
        for x, rec in sorted(labs.items()):
            cls = rec["class"]
            if cls not in CLASSES:
                skipped["UNSURE" if cls == "UNSURE" else "contested" if rec.get("conf") == "contested"
                        else "unresolved"] += 1
                continue
            p = pr[x]
            g = "ct" if rec.get("ct") else "head"
            conf[g][CLASSES.index(cls), CLASSES.index(p.cls)] += 1
            code = rec.get("pattern")
            if cls in REDUNDANT and code:
                v = pat[(code, cls)]
                v[0] += 1
                v[1] += p.cls != "KEEP"
                v[2] += p.cls == cls
                v[3] += bool(rec.get("ct"))
            elif cls == "KEEP" and code:
                krule[code][0] += 1
                krule[code][1] += p.cls != "KEEP"
            bad_ptr = False
            if cls in ("ECHO", "NARRATE") and rec.get("target") and p.cls != "KEEP":
                ok = target_ok(it, rec, p.pointer)
                ptr[0] += ok
                ptr[1] += 1
                bad_ptr = not ok
            if p.cls != cls or bad_ptr:
                wrong.append((it, x, rec, p))
    n_lab = int(sum(c.sum() for c in conf.values()))
    out.append(f"{len(items)} screens, {n_lab} labelled nodes ({int(conf['ct'].sum())} ct)"
               + (f"; not scored: {dict(sorted(skipped.items()))}" if skipped else ""))

    def binary(c: np.ndarray, name: str) -> None:
        tp = int(c[1:, 1:].sum())
        fn = int(c[1:, 0].sum())
        fp = int(c[0, 1:].sum())
        tn = int(c[0, 0])
        p, r, f = prf(tp, fp, fn)
        fa = fp / (fp + tn) if fp + tn else 0.0
        out.append(f"{name}: REDUNDANT vs KEEP  P {p:.3f}  R {r:.3f}  F1 {f:.3f}  false alarms {fp}/{fp + tn} KEEP "
                   f"({100 * fa:.1f}%)   TP {tp} FP {fp} FN {fn} TN {tn}")

    binary(conf["head"], "headline (ct nodes left out)")
    if conf["ct"].sum():
        binary(conf["ct"], "ct column (stretch)")
    c = conf["head"]
    out.append("confusion, headline nodes (rows: label, columns: prediction):")
    out.append("           " + "".join(f"{SHORT[k]:>8}" for k in CLASSES))
    for i, k in enumerate(CLASSES):
        out.append(f"  {k:<9}" + "".join(f"{int(v):>8}" for v in c[i]))
    f1s = []
    for i in range(4):
        f1s.append(prf(int(c[i, i]), int(c[:, i].sum() - c[i, i]), int(c[i, :].sum() - c[i, i]))[2])
    out.append(f"four-class macro-F1 {sum(f1s) / 4:.3f}  (" + "  ".join(f"{k} {f:.3f}" for k, f in zip(CLASSES, f1s))
               + ")")
    if pat:
        out.append("per pattern (labelled REDUNDANT nodes, ct nodes included; flagged = any non-KEEP call; exact = the "
                   "labelled class; ct = how many of n are ct):")
        for (code, cls), (n, fl, ex, ct) in sorted(pat.items(), key=lambda kv: (kv[0][0][0], int(kv[0][0][1:] or 0))):
            out.append(f"  {code:<4} {cls:<8} n {n:>4}  flagged {fl:>4} ({100 * fl / n:5.1f}%)  exact {ex:>4}  ct {ct:>4}")
    if krule:
        out.append("per K rule (KEEP nodes that name it; flagged = false alarms):")
        for code, (n, fl) in sorted(krule.items(), key=lambda kv: int(kv[0][1:])):
            out.append(f"  {code:<4} n {n:>4}  flagged {fl:>4} ({100 * fl / n:5.1f}%)")
    out.append(f"pointer accuracy: {ptr[0]}/{ptr[1]}" + (f" ({100 * ptr[0] / ptr[1]:.1f}%)" if ptr[1] else "")
               + "  (labelled ECHO/NARRATE with a target, called non-KEEP; the target or an equiv copy counts)")
    sel = [w for w in wrong if (a.cls is None or w[2]["class"] == a.cls)
           and (a.pattern is None or w[2].get("pattern") == a.pattern) and (a.screen is None or w[0].name == a.screen
                                                                           or w[0].screen.get("id") == a.screen)]
    filt = " ".join(f"--{k} {v}" for k, v in (("class", a.cls), ("pattern", a.pattern), ("screen", a.screen)) if v)
    page = sel[a.offset: a.offset + SHOW]
    out.append(f"wrong nodes: {len(sel)}{' matching ' + filt if filt else ''}; showing {a.offset + 1 if page else 0}-"
               f"{a.offset + len(page) if page else 0} (--offset to page)")
    for it, x, rec, p in page:
        n = it.screen["nodes"][x]
        out.append(f"  {it.name} {n['i']} {n['role']} {short(n['text'])}  label {describe_label(it, rec)}  "
                   f"called {describe_pred(it, p)}")


def calls_report(items: list[Item], preds, a, out: list[str]) -> None:
    for it, pr in zip(items, preds):
        if a.screen and a.screen not in (it.name, it.screen.get("id")):
            continue
        out.append(f"{it.name} (screen {it.screen.get('id')}, unlabelled): class, logits "
                   f"[{' '.join(CLASSES)}], pointer and BOS weight per node")
        for p in pr:
            n = it.screen["nodes"][p.index]
            tgt = "-"
            if p.pointer is not None:
                tgt = f"→{it.screen['nodes'][p.pointer]['i']}"
            out.append(f"  {n['i']} {n['k']} {n['role']} {short(n['text'])}  {p.cls:<7} [" +
                       " ".join(f"{v:+.3f}" for v in p.logits) + f"] {tgt} bos {p.bos_weight:.2f}")


def against_report(items: list[Item], pa, pb, a, out: list[str]) -> None:
    diff = []
    for it, ra, rb in zip(items, pa, pb):
        if a.screen and a.screen not in (it.name, it.screen.get("id")):
            continue
        labs = it.node_labels()
        for x, (p, q) in enumerate(zip(ra, rb)):
            if p.cls != q.cls or p.pointer != q.pointer:
                rec = labs.get(x)
                if a.cls is not None and (rec is None or rec["class"] != a.cls):
                    continue
                if a.pattern is not None and (rec is None or rec.get("pattern") != a.pattern):
                    continue
                diff.append((it, x, rec, p, q))
    page = diff[a.offset: a.offset + SHOW]
    out.append(f"nodes whose class or pointer differs: {len(diff)}; showing {a.offset + 1 if page else 0}-"
               f"{a.offset + len(page) if page else 0}")
    for it, x, rec, p, q in page:
        n = it.screen["nodes"][x]
        lab = f"  label {describe_label(it, rec)}" if rec is not None else ""
        out.append(f"  {it.name} {n['i']} {n['role']} {short(n['text'])}  A {describe_pred(it, p)}  "
                   f"B {describe_pred(it, q)}{lab}")


def cmd_score(argv: list[str], roots: Roots, node: str, out: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="./score", description="Score an artefact on the disclosed screens (KIT.md §12's measures), or show its "
                                    "calls on screens you wrote.")
    ap.add_argument("art", help="artefact directory (artifact.npz, vocab.txt, meta.json), e.g. work/")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--set", choices=("dev", "reference"))
    src.add_argument("--files", nargs="+", metavar="FILE", help="screen JSON files you wrote (labels optional: "
                                                                "<name>.labels.jsonl beside it, or a \"labels\" list)")
    ap.add_argument("--against", metavar="ART2", help="list the nodes whose class or pointer differs from ART2's")
    ap.add_argument("--class", dest="cls", choices=CLASSES + ("UNSURE",), help="only wrong nodes with this label")
    ap.add_argument("--pattern", help="only wrong nodes whose label names this pattern or K rule (R1, K4, ...)")
    ap.add_argument("--screen", help="only this screen (id, or file name for --files)")
    ap.add_argument("--offset", type=int, default=0, help=f"skip this many listed nodes ({SHOW} are shown)")
    a = ap.parse_args(argv)
    if a.offset < 0:
        ap.error("--offset must be 0 or more")
    for p in a.files or []:
        roots.resolve(p)
    if a.against:
        roots.resolve(a.against, what="artefact")
    art, d = load_art(a.art, roots)
    items = set_items(a.set, roots) if a.set else file_items(a.files, roots, node)
    preds = predict(art, items)
    head = f"artefact {art.sha256[:12]} ({roots.rel(d)}), pointer head {list(art.pointer_head)}"
    if a.against:
        art2, d2 = load_art(a.against, roots)
        out.append(head + f"  against B = {art2.sha256[:12]} ({roots.rel(d2)})  on "
                   + (f"examples/{a.set}" if a.set else f"{len(items)} file(s)"))
        against_report(items, preds, predict(art2, items), a, out)
        return 0
    out.append(head + "  on " + (f"examples/{a.set}" if a.set else f"{len(items)} file(s)"))
    labelled = [i for i in range(len(items)) if items[i].labels is not None]
    if labelled:
        score_report([items[i] for i in labelled], [preds[i] for i in labelled], a, out)
    rest = [i for i in range(len(items)) if items[i].labels is None]
    if rest:
        calls_report([items[i] for i in rest], [preds[i] for i in rest], a, out)
    return 0


# -- ./inspect -----------------------------------------------------------------------------------------------------


def load_directions(p: Path | None, roots: Roots) -> dict[str, np.ndarray] | None:
    if p is None or not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ToolError(f"{roots.rel(p)} is not JSON: {e}") from None
    if not isinstance(raw, dict) or not raw:
        raise ToolError(f"{roots.rel(p)}: expected {{name: [dims] | {{dim: weight}}}}")
    out = {}
    for name, spec in raw.items():
        v = np.zeros(mdl.D)
        items = [(d, 1.0) for d in spec] if isinstance(spec, list) else list(spec.items()) \
            if isinstance(spec, dict) else None
        if items is None:
            raise ToolError(f"direction {name!r}: give a list of dims or {{dim: weight}}")
        for d, w in items:
            try:
                d, w = int(d), float(w)
            except (TypeError, ValueError):
                raise ToolError(f"direction {name!r}: bad entry {d!r}: {w!r}") from None
            if not 0 <= d < mdl.D or not np.isfinite(w):
                raise ToolError(f"direction {name!r}: dim {d} outside 0..{mdl.D - 1} or weight not finite")
            v[d] += w
        out[str(name)] = v
    return out


def proj(x: np.ndarray, dirs: dict[str, np.ndarray] | None) -> str:
    if dirs:
        return "  ".join(f"{k} {float(v @ x):+.3f}" for k, v in dirs.items())
    top = [int(d) for d in np.argsort(-np.abs(x), kind="stable")[:6] if abs(x[d]) >= 5e-4]
    return f"|x| {float(np.linalg.norm(x)):.3f}" + ("; " + " ".join(f"d{d} {float(x[d]):+.3f}" for d in top)
                                                   if top else "")


def cmd_inspect(argv: list[str], roots: Roots, node: str, out: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="./inspect", description="One node's computation, position by position: embedding parts, each head's "
                                      "attention grouped by source node, residual projections, MLP units, logits.")
    ap.add_argument("art", help="artefact directory")
    ap.add_argument("file", help="screen JSON, or a source file (extracted first)")
    ap.add_argument("--node", type=int, required=True, metavar="K", help="the node's i (as in the screen JSON)")
    ap.add_argument("--tokens", action="store_true", help="every token of the node, not only its EON")
    ap.add_argument("--units", type=int, default=0, metavar="TOP_K", help="the TOP_K most active MLP units per layer")
    ap.add_argument("--layer", type=int, choices=range(1, mdl.L + 1), help="attention table for this layer only")
    ap.add_argument("--head", type=int, choices=range(mdl.H), help="attention table for this head only")
    ap.add_argument("--directions", metavar="FILE", help="named directions (default work/directions.json): "
                                                         "{name: [dims] | {dim: weight}}; projection = sum w * h[dim]")
    a = ap.parse_args(argv)
    if not 0 <= a.units <= mdl.M:
        ap.error(f"--units must be 0..{mdl.M}")
    fpath = roots.resolve(a.file)
    dpath0 = roots.resolve(a.directions, what="directions file") if a.directions else None
    art, d = load_art(a.art, roots)
    screen = load_screen_file(fpath, roots, node)
    dpath = dpath0 if dpath0 is not None else Path(roots.work) / "directions.json"
    if a.directions and not dpath.is_file():
        raise ToolError(f"no such file: {a.directions}")
    dirs = load_directions(dpath, roots)
    nodes = screen["nodes"]
    idx = [x for x, n in enumerate(nodes) if n.get("i") == a.node]
    if not idx:
        raise ToolError(f"no node with i = {a.node} (the screen has i = {nodes[0]['i'] if nodes else '-'}.."
                        f"{nodes[-1]['i'] if nodes else '-'})")
    x = idx[0]
    enc = tk.encode_screen(screen, art.vocab)
    wi, w = next((wi, w) for wi, w in enumerate(enc.windows) if x in w.nodes)
    m = w.nodes.index(x) + 1  # window node number
    res = mdl.forward(art, w, collect=True)
    col = res.collect
    p = art.f64
    tok_node = np.asarray(w.node)
    positions = [t for t in range(len(w.tokens)) if tok_node[t] == m]
    eon = w.eon[m - 1]
    rows = positions if a.tokens else [eon]
    preds = mdl.predict_encoded(art, enc)
    pr = preds[x]
    n = nodes[x]

    def label(mm: int) -> str:
        if mm == 0:
            return "BOS"
        nn = nodes[w.nodes[mm - 1]]
        return ("self " if mm == m else "") + f"{nn['i']} {nn['role']} {short(nn['text'], 22)}"

    out.append(f"artefact {art.sha256[:12]} ({roots.rel(d)})  screen {screen.get('id')}  window {wi} "
               f"({len(w.tokens)} positions)")
    out.append(f"node {n['i']} k={n['k']} {n['role']} [{','.join(enc.nodes[x].flags) or '-'}] {short(n['text'], 80)}")
    tgt = "-" if pr.pointer is None else label(w.nodes.index(pr.pointer) + 1)
    out.append(f"call {pr.cls}  logits [" + " ".join(f"{c} {v:+.4f}" for c, v in zip(CLASSES, pr.logits))
               + f"]  pointer → {tgt}  (BOS weight {pr.bos_weight:.3f}; pointer head {list(art.pointer_head)})")
    out.append("directions: " + (", ".join(dirs) + f" (from {roots.rel(dpath)})" if dirs else
                                 "none (no work/directions.json): |x| and the 6 largest dims"))
    flag_ids = [f for f in range(mdl.N_FLAGS) if (w.flags[eon] >> f) & 1]
    for t in rows:
        tid = w.tokens[t]
        kind = tokrep.id_kind(tid, art.vocab)
        out.append(f"== position {t} ({'EON' if t == eon else f'offset {w.offsets[t]}'}, id {tid} {kind})")
        parts = [("tok", p["tok_emb.weight"][tid]), ("off", p["off_emb.weight"][w.offsets[t]]),
                 ("role", p["role_emb.weight"][w.roles[t]]),
                 ("cluster", p["cluster_emb.weight"][w.clusters[t] % mdl.N_CLUSTERS]),
                 ("flags", sum((p["flag_emb.weight"][f] for f in flag_ids), np.zeros(mdl.D)))]
        for name, v in parts:
            out.append(f"  embed {name:<8} {proj(v, dirs)}")
        out.append(f"  embed total    {proj(col['embed'][t], dirs)}")
        for li in range(mdl.L):
            pre = f"layers.{li}."
            h_in = col["embed"] if li == 0 else col["resid_mlp"][li - 1]
            att = col["attn"][li]
            v = h_in @ p[pre + "attn.v.weight"].T + p[pre + "attn.v.bias"]
            wo = p[pre + "attn.o.weight"]
            show_att = a.layer is None or a.layer == li + 1
            total = p[pre + "attn.o.bias"].copy()
            for hd in range(mdl.H):
                sl = slice(hd * mdl.DH, (hd + 1) * mdl.DH)
                row = att[hd, t]
                write = (row @ v[:, sl]) @ wo[:, sl].T
                total += write
                if not show_att or (a.head is not None and a.head != hd):
                    continue
                sums = np.zeros(int(tok_node.max()) + 1)
                np.add.at(sums, tok_node, row)
                order = [mm for mm in np.argsort(-sums, kind="stable") if sums[mm] >= 0.005][:8]
                grp = " | ".join(f"{label(int(mm))} {sums[mm]:.3f}" for mm in order)
                out.append(f"  L{li + 1} h{hd}  {grp}")
                out.append(f"         write {proj(write, dirs)}")
            out.append(f"  L{li + 1} attn out (+bias) {proj(total, dirs)}")
            out.append(f"  L{li + 1} resid after attn {proj(col['resid_attn'][li][t], dirs)}")
            hid = np.maximum(col["resid_attn"][li][t] @ p[pre + "mlp.fc1.weight"].T + p[pre + "mlp.fc1.bias"], 0.0)
            mlp = hid @ p[pre + "mlp.fc2.weight"].T + p[pre + "mlp.fc2.bias"]
            out.append(f"  L{li + 1} MLP: {int((hid > 0).sum())}/{mdl.M} units active; write {proj(mlp, dirs)}")
            if a.units:
                for u in [int(u) for u in np.argsort(-hid, kind="stable")[: a.units] if hid[u] > 0]:
                    out.append(f"         unit {u:>3} act {hid[u]:.4f}  write {proj(hid[u] * p[pre + 'mlp.fc2.weight'][:, u], dirs)}")
            out.append(f"  L{li + 1} resid after MLP  {proj(col['resid_mlp'][li][t], dirs)}")
        if t == eon:
            lg = col["resid_mlp"][-1][t] @ p["head.weight"].T + p["head.bias"]
            out.append("  logits " + " ".join(f"{c} {v:+.4f}" for c, v in zip(CLASSES, lg)))
    return 0


# -- entry ---------------------------------------------------------------------------------------------------------

COMMANDS = {"tok": cmd_tok, "lint": cmd_lint, "score": cmd_score, "inspect": cmd_inspect}


def cap(text: str, limit: int = CAP) -> str:
    """At most ``limit`` characters of output (cut at a line end when one is near), then a notice."""
    if len(text) <= limit:
        return text
    notice = f"[output truncated at {limit:,} characters of {len(text):,}; narrow it with the tool's filters or " \
             "--offset]\n"
    room = limit - len(notice)
    cut = text.rfind("\n", 0, room)
    if cut < room * 0.8:
        cut = room
    return text[:cut].rstrip("\n") + "\n" + notice


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[3] not in COMMANDS:
        print("author_tool.py AUTHOR_DIR EXTRA_ROOT NODE tok|lint|score|inspect [args]", file=sys.stderr)
        return 2
    roots = Roots(argv[0], argv[1])
    node = argv[2]
    tool = argv[3]
    sys.argv[0] = "./" + tool
    out: list[str] = []
    buf = io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = COMMANDS[tool](argv[4:], roots, node, out)
    except SystemExit as e:  # argparse: --help or a usage error
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 2)
    except ToolError as e:
        out.append(f"error: {e}")
        code = 2
    except Exception as e:  # noqa: BLE001  (one line, no harness traceback: the author cannot fix the harness)
        out.append(f"error: the tool failed on this input ({type(e).__name__}: {e}); check the file against KIT.md §4")
        code = 2
    text = buf.getvalue() + ("\n".join(out) + "\n" if out else "")
    sys.stdout.write(cap(text))
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
