"""Leakage check of a built author kit (SPEC §9 "Leakage check", §13). Opens no sealed file.

    uv run python kit/leakage.py AUTHOR_DIR [--require-sealed] [--json OUT]

Exit 0 when nothing unwaived is found, 1 otherwise. scripts/prepare_author_dir.sh runs it on every hand-over.

Checks (each problem names its check, the kit file and the match):

files               Only the kit's own files are present (the allowed list below; .venv/ and work/ are not looked
                    at); kit/lib/ equals ui/, tools/py_guard.py equals clouds', LABELLING.md equals the author's
                    copy of the guide (kit_build.kit_guide()); HANDOVER.json, when present, is the author's view
                    of the record (kit_build.AUTHOR_RECORD_KEYS only);
                    examples/dev holds exactly the disclosed dev screens (screen, labels, text for each, every id a
                    dev id) and examples/reference exactly the reference_real screens; screen files carry only screen
                    keys and label files only label keys (a contested record's candidates only candidate keys);
                    nothing carries the generator's `clean` flag, its attempt number or styling idiom.
internal-names      No document names the people, labellers, generators or notes behind the kit, nor a held-out
                    set (INTERNAL_NAMES: "Matthew", "Codex", "research/", "slop-private", "corpus note", "test_gen",
                    "ood_codex", "SPEC", ...).
sealed-file         No kit file is byte-identical to a sealed file (datasets/sealed_sha256.txt) or to a calibration
                    screen or source (datasets/calibration: manifest hashes and the files themselves).
calibration-copy    No calibration node string of four or more words occurs in a kit document or example screen.
generator-template  No generator template (a string constant with a {slot}) occurs verbatim in any kit file.
generator-table     No run of 4 consecutive entries of a generator table (copy entries only: not identifiers, role
                    names or numbers) occurs, in order, in a kit document or code file (a dumped table). In
                    LABELLING.md such a run is a note: the generator copied the guide's list.
generator-code      No generator identifier (a table, function or class name) or module file name occurs in a kit
                    document or code file. No generator source found at all is itself a problem (the generator
                    checks would otherwise pass on nothing).
generator-strings   No generator string of four or more words occurs in a kit document unless it is also the text
                    of a disclosed screen's node (a document must not show undisclosed generator copy).
rules-lists         No run of 3 consecutive entries of a rule-layer list (rules/src/lists.js) occurs, in order, in a
                    kit document or code file.
heldout             For val, test_gen and ood_codex where they exist: no screen id and no node-text sequence of an
                    example screen is theirs, and no node string of four or more words of theirs occurs in a kit
                    document unless a disclosed screen has it too. (In LABELLING.md, which predates the generator,
                    such a match is a note: the generator quoted the guide.)
sealed-ngrams       SPEC §9: salted sha256 of every normalised whole node string of 2+ words and every 3-stem run in
                    KIT.md, LABELLING.md, PROMPT.md, examples/ and the generator tables, against
                    datasets/sealed_ngrams.json (written by scripts/build_sealed.py: {"format": "slop.sealed_ngrams/1",
                    "hashes": [hex, ...]}, hash = sha256(salt + "\\x00" + stems joined by " "), salt in kit/sealed_salt).
                    Until that file exists the check is reported as not run; --require-sealed makes that a failure
                    (scripts/prepare_author_dir.sh passes it on a real hand-over, scripts/launch.sh before every
                    launch). To go ahead without it, waive {"check": "sealed-ngrams", "file": "-", "match":
                    "not-run"} in kit/leakage_waivers.json, with the reason.

Documents are KIT.md, PROMPT.md, LABELLING.md and examples/README.md; code files are kit/lib/* and tools/*.
Waivers (SPEC §9: listed for Matthew to waive one by one, each with its reason) are kept in kit/leakage_waivers.json:
[{"check", "file", "match", "reason"}]; a waiver matches a problem with the same check, file and match.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

SLOP = Path(__file__).resolve().parents[1]
REPO = SLOP.parent
DATASETS = SLOP / "datasets"
GENERATOR = SLOP / "generator"
RULE_LISTS = SLOP / "rules" / "src" / "lists.js"
WAIVERS = SLOP / "kit" / "leakage_waivers.json"
SEALED_NGRAMS = DATASETS / "sealed_ngrams.json"
SEALED_SALT = SLOP / "kit" / "sealed_salt"
sys.path.insert(0, str(SLOP / "kit" / "lib"))
import tokenizer as tk  # noqa: E402  (the kit's own copy, for the §6 stems)

sys.path.insert(0, str(SLOP / "kit"))
import kit_build as kb  # noqa: E402

DOCS = ("KIT.md", "PROMPT.md", "LABELLING.md", "examples/README.md")
TOP_FILES = {"KIT.md", "PROMPT.md", "LABELLING.md", "HANDOVER.json", "tok", "lint", "score", "inspect", "python3",
             "tools/py_guard.py", "examples/README.md", "examples/INDEX.txt"}
LIB_ALLOWED = {f"kit/lib/{f}" for f in kb.LIB_FILES + ("SHA256SUMS",)}
SKIP_DIRS = (".venv", "work")
SCREEN_KEYS = {"schema", "id", "framework", "source", "clusters", "nodes"}
FORBIDDEN_KEYS = ("clean", "attempt", "idiom")
HELDOUT_SETS = ("val", "test_gen", "ood_codex")
GENERIC_GEN_FILES = {"build.py", "core.py", "join.py", "__init__.py"}  # names too common to be a sentinel
# harness vocabulary the generator shares with the disclosed code and documents (flag names, role lists, ...): an
# identifier that occurs here is public already, so it is no sentinel
SHARED_VOCAB = tuple(sorted((SLOP / "ui").glob("*.py"))) + (SLOP / "ui" / "ids.json", SLOP / "SPEC.md",
                                                            SLOP / "LABELLING.md")
# names that tell the author who or what is behind the kit, or which held-out sets exist (internal-names)
INTERNAL_NAMES = ("Matthew", "Codex", "research/", "slop-private", "corpus note", "corpus-scout", "test_gen",
                  "ood_codex", "SPEC.md", "SPEC §", "modellabel", "labeller \"mg\"", "Fable 5")
MIN_WORDS = 4  # generator-strings and heldout: shorter strings are common English ("is not a", "word of mouth")
# LABELLING.md predates the generator and the held-out sets; when they share a string, the generator quoted the
# guide. Such matches are reported as notes, not problems (the sealed n-gram check still covers the guide).
GUIDE_FILE = "LABELLING.md"
SLOT = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")
WORD = re.compile(r"\w+", re.UNICODE)


norm = kb.norm


def nwords(s: str) -> int:
    return len(WORD.findall(s))


class Report:
    def __init__(self, waivers: list[dict]):
        self.waivers = waivers
        self.used: set[int] = set()
        self.problems: list[dict] = []
        self.waived: list[dict] = []
        self.notes: list[str] = []

    def add(self, check: str, file: str, match: str, detail: str = "") -> None:
        p = {"check": check, "file": file, "match": match, "detail": detail}
        for i, w in enumerate(self.waivers):
            if w.get("check") == check and w.get("file") == file and w.get("match") == match:
                self.used.add(i)
                self.waived.append({**p, "reason": w.get("reason", "")})
                return
        self.problems.append(p)


# -- kit contents --------------------------------------------------------------------------------------------------


def kit_files(author: Path) -> list[str]:
    out = []
    for p in sorted(author.rglob("*")):
        rel = p.relative_to(author)
        if rel.parts[0] in SKIP_DIRS or not p.is_file():
            continue
        out.append(str(rel))
    return out


def read_text(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def example_screens(author: Path) -> dict[str, dict]:
    out = {}
    for sub in ("dev", "reference"):
        for p in sorted((author / "examples" / sub).glob("*.screen.json")):
            try:
                out[f"examples/{sub}/{p.name}"] = json.loads(p.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
    return out


def check_files(author: Path, files: list[str], rep: Report) -> None:
    man = json.loads((kb.REF_DIR / "manifest.json").read_text())
    ref_ids = {e["id"] for e in man["screens"]}
    if man.get("fixture", {}).get("id"):
        ref_ids.add(man["fixture"]["id"])
    ref_sources: dict[str, Path] = {}
    for sid in ref_ids:
        s = json.loads((kb.REF_DIR / "screens" / f"{sid}.json").read_text(encoding="utf-8"))
        path = s["source"]["path"]
        ref_sources[f"examples/reference/{sid}.source{Path(path).suffix}"] = \
            kb.REF_DIR / "sources" / s["source"]["repo"] / path
    dev_ids = {s["id"] for s in kb.iter_jsonl(kb.DEV_DIR / "screens.jsonl")}
    want_dev = set(kb.select_dev()["ids"])
    seen_dev: dict[str, set[str]] = {}
    seen_ref: dict[str, set[str]] = {}
    ex = re.compile(r"^examples/(dev|reference)/([0-9a-f]{16})\.(screen\.json|labels\.jsonl|txt)$")
    for f in files:
        m = ex.match(f)
        if f in ref_sources:
            if kb.sha256(author / f) != kb.sha256(ref_sources[f]):
                rep.add("files", f, f, "not a byte copy of its reference source")
            continue
        if f in TOP_FILES or f in LIB_ALLOWED:
            continue
        if m:
            (seen_dev if m.group(1) == "dev" else seen_ref).setdefault(m.group(2), set()).add(m.group(3))
            continue
        rep.add("files", f, f, "not a kit file")
    for f in sorted(LIB_ALLOWED - {"kit/lib/SHA256SUMS"}):
        if f not in files:
            rep.add("files", f, f, "missing")
        elif kb.sha256(author / f) != kb.sha256(kb.UI / Path(f).name):
            rep.add("files", f, f, "differs from ui/")
    if "tools/py_guard.py" not in files or kb.sha256(author / "tools/py_guard.py") != kb.sha256(kb.CLOUDS_PY_GUARD):
        rep.add("files", "tools/py_guard.py", "tools/py_guard.py", "missing or not a byte copy of "
                f"{kb.CLOUDS_PY_GUARD.relative_to(REPO)}")
    if "LABELLING.md" not in files or (author / "LABELLING.md").read_text(encoding="utf-8") != kb.kit_guide():
        rep.add("files", "LABELLING.md", "LABELLING.md", "missing or not the author's copy of the guide "
                                                         "(kit_build.kit_guide())")
    if "HANDOVER.json" in files:
        try:
            extra = set(json.loads((author / "HANDOVER.json").read_text())) - set(kb.AUTHOR_RECORD_KEYS)
        except (OSError, ValueError):
            extra = {"(not JSON)"}
        if extra:
            rep.add("files", "HANDOVER.json", ",".join(sorted(extra)), "keys beyond the author's view of the record")
    if set(seen_dev) != want_dev:
        rep.add("files", "examples/dev", "selection", f"{len(seen_dev)} screens, not the {len(want_dev)} the rule picks")
    for sid, kinds in seen_dev.items():
        if sid not in dev_ids:
            rep.add("files", f"examples/dev/{sid}.json", sid, "not a dev screen")
        if kinds != {"screen.json", "labels.jsonl", "txt"}:
            rep.add("files", f"examples/dev/{sid}", sid, f"has {sorted(kinds)}")
    if set(seen_ref) != ref_ids:
        rep.add("files", "examples/reference", "screens", "not exactly the reference_real screens")
    for sid, kinds in seen_ref.items():
        if not {"screen.json", "txt"} <= kinds:
            rep.add("files", f"examples/reference/{sid}", sid, f"has {sorted(kinds)}")
    labels_present = kb.REF_LABELS.is_file()
    if any("labels.jsonl" in k for k in seen_ref.values()) != labels_present:
        rep.add("files", "examples/reference", "labels", "reference labels present without the answer key, or the "
                                                         "reverse")
    for f in files:
        if not f.startswith("examples/") or f in ref_sources:
            continue
        text = read_text(author / f)
        if text is None:
            rep.add("files", f, f, "not UTF-8 text")
            continue
        if f.endswith(".labels.jsonl"):
            for ln in text.splitlines():
                r = json.loads(ln)
                allowed = set(kb.NODE_KEYS if "class" in r else kb.SCREEN_KEYS)
                if set(r) - allowed:
                    rep.add("files", f, ",".join(sorted(set(r) - allowed)), "label keys outside the record format")
                if "candidates" in r and not kb.is_contested(r):
                    rep.add("files", f, "candidates", "candidates on a record that is not contested")
                for c in r.get("candidates") or []:
                    if set(c) - set(kb.CAND_KEYS):
                        rep.add("files", f, ",".join(sorted(set(c) - set(kb.CAND_KEYS))),
                                "candidate keys outside the record format")
        elif f.endswith(".json"):
            s = json.loads(text)
            if set(s) - SCREEN_KEYS:
                rep.add("files", f, ",".join(sorted(set(s) - SCREEN_KEYS)), "screen keys outside slop.screen/1")
        for k in FORBIDDEN_KEYS:
            if re.search(rf'"{k}"\s*:', text):
                rep.add("files", f, k, "carries a generator sidecar field")
    idx = read_text(author / "examples/INDEX.txt") or ""
    if re.search(r"\bclean\b", idx):
        rep.add("files", "examples/INDEX.txt", "clean", "names the clean flag")


# -- byte identity -------------------------------------------------------------------------------------------------


def check_sealed_files(author: Path, files: list[str], rep: Report) -> None:
    sealed = set()
    lst = DATASETS / "sealed_sha256.txt"
    if lst.is_file():
        sealed = {ln.strip() for ln in lst.read_text().splitlines() if ln.strip()}
    else:
        rep.notes.append("sealed-file: datasets/sealed_sha256.txt missing; byte check against sealed files not run")
    cal = set()
    cman = DATASETS / "calibration" / "manifest.json"
    if cman.is_file():
        for e in json.loads(cman.read_text())["screens"]:
            cal.update(e.get(k) for k in ("screen_sha256", "source_sha256") if e.get(k))
    for p in (DATASETS / "calibration").rglob("*"):
        if p.is_file() and p.name not in ("manifest.json", "order.txt"):
            cal.add(kb.sha256(p))
    for f in files:
        h = kb.sha256(author / f)
        if h in sealed:
            rep.add("sealed-file", f, h, "byte-identical to a sealed file")
        if h in cal:
            rep.add("sealed-file", f, h, "byte-identical to a calibration screen or source")


# -- strings -------------------------------------------------------------------------------------------------------


def screen_strings(s: dict) -> list[str]:
    return [norm(n.get("text", "")) for n in s.get("nodes", [])]


def check_calibration(docs: dict[str, str], ex_screens: dict[str, dict], rep: Report) -> None:
    strings = set()
    for p in (DATASETS / "calibration" / "screens").glob("*.json"):
        strings.update(t for t in screen_strings(json.loads(p.read_text(encoding="utf-8"))) if nwords(t) >= MIN_WORDS)
    for f, s in ex_screens.items():
        for t in set(screen_strings(s)) & strings:
            rep.add("calibration-copy", f, t)
    for f, text in docs.items():
        for t in strings:
            if t in text:
                rep.add("calibration-copy", f, t)


def generator_sources() -> list[Path]:
    return sorted(p for p in GENERATOR.glob("*.py") if p.name != "__init__.py") + sorted(GENERATOR.glob("*.ts"))


def generator_parts() -> tuple[set[str], list[list[str]], set[str], set[str]]:
    """(templates, tables, identifiers, literal strings) of the generator's Python sources."""
    templates, tables, idents, strings = set(), [], set(), set()
    for p in generator_sources():
        if p.suffix != ".py":
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value
                if SLOT.search(v) and nwords(SLOT.sub(" ", v)) >= 1 and len(v) >= 6:
                    templates.add(v)
                elif nwords(v) >= 3 and "\n" not in v:
                    strings.add(v)
            elif isinstance(node, (ast.List, ast.Tuple)):
                vals = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                if len(vals) == len(node.elts):
                    copy = [v for v in vals if is_copy(v)]
                    if len(copy) >= 4:
                        tables.append(copy)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                idents.add(node.name)
            elif isinstance(node, ast.Assign) and node in tree.body:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        idents.add(t.id)
    idents = {i for i in idents if len(i) >= 6 and ("_" in i.strip("_") or re.search(r"[a-z][A-Z]", i))}
    shared = "\n".join(p.read_text(encoding="utf-8") for p in SHARED_VOCAB if p.is_file())
    idents = {i for i in idents if not re.search(rf"(?<![\w.]){re.escape(i)}(?!\w)", shared)}
    idents |= {p.name for p in generator_sources() if p.name not in GENERIC_GEN_FILES}
    return templates, tables, idents, strings


def is_copy(v: str) -> bool:
    """A table entry that is UI copy, not an identifier, a role name, a number or a one- or two-letter suffix."""
    v = v.strip()
    if re.fullmatch(r"[a-z0-9_]+", v) and " " not in v:
        return False
    return any(len(w) >= 3 and not w.isdigit() for w in WORD.findall(v))


def run_found(entries: list[str], text: str, k: int, span: int = 400) -> str | None:
    """The first k consecutive entries that occur in order, close together, in text (normalised)."""
    ents = [norm(e) for e in entries]
    for i in range(len(ents) - k + 1):
        run = ents[i: i + k]
        if any(not e or nwords(e) == 0 for e in run):
            continue
        pos = text.find(run[0])
        while pos != -1:
            cur, ok = pos + len(run[0]), True
            for e in run[1:]:
                nxt = text.find(e, cur)
                if nxt == -1 or nxt - cur > span:
                    ok = False
                    break
                cur = nxt + len(e)
            if ok:
                return " | ".join(run)
            pos = text.find(run[0], pos + 1)
    return None


def rule_lists() -> dict[str, list[str]]:
    src = RULE_LISTS.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"export const (\w+) = \[(.*?)\];", src, re.S):
        body = m.group(2)
        ents = re.findall(r"phrase:\s*'([^']*)'", body) or re.findall(r"'([^']*)'", body)
        if len(ents) >= 3:
            out[m.group(1)] = ents
    return out


def heldout_screens() -> dict[str, list[dict]]:
    out = {}
    for name in HELDOUT_SETS:
        p = DATASETS / name / "screens.jsonl"
        if p.is_file():
            out[name] = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return out


seq_hash = kb.seq_hash


# -- sealed n-grams ------------------------------------------------------------------------------------------------


def stems(text: str) -> list[str]:
    return [t.stem for t in tk.tokenize_text(text).tokens if t.kind == "word"]


def ngrams_of_text(text: str) -> set[str]:
    out = set()
    for line in text.splitlines():
        st = stems(line)
        out.update(" ".join(st[i: i + 3]) for i in range(len(st) - 2))
    return out


def ngrams_of_node(text: str) -> set[str]:
    st = stems(text)
    out = {" ".join(st[i: i + 3]) for i in range(len(st) - 2)}
    if len(st) >= 2:
        out.add(" ".join(st))
    return out


def check_sealed_ngrams(author: Path, docs: dict[str, str], ex_screens: dict[str, dict], rep: Report,
                        require: bool) -> None:
    if not (SEALED_NGRAMS.is_file() and SEALED_SALT.is_file()):
        msg = "sealed-ngrams: not run (datasets/sealed_ngrams.json and kit/sealed_salt do not exist yet; " \
              "scripts/build_sealed.py writes them at sealing time)"
        if require:
            rep.add("sealed-ngrams", "-", "not-run", msg)
        else:
            rep.notes.append(msg)
        return
    data = json.loads(SEALED_NGRAMS.read_text())
    if data.get("format") != "slop.sealed_ngrams/1":
        rep.add("sealed-ngrams", "datasets/sealed_ngrams.json", "format", "unknown format")
        return
    hashes = set(data["hashes"])
    salt = SEALED_SALT.read_bytes().strip()

    def h(g: str) -> str:
        return hashlib.sha256(salt + b"\x00" + g.encode()).hexdigest()

    units: dict[str, set[str]] = {f: ngrams_of_text(t) for f, t in docs.items()}
    for f, s in ex_screens.items():
        units[f] = set().union(*(ngrams_of_node(n.get("text", "")) for n in s.get("nodes", []))) if s.get("nodes") \
            else set()
    templates, tables, _, strings = generator_parts()
    gen = set()
    for v in list(templates) + list(strings) + [e for t in tables for e in t]:
        gen |= ngrams_of_node(SLOT.sub(" ", v))
    units["generator tables"] = gen
    for f, grams in units.items():
        for g in sorted(grams):
            if h(g) in hashes:
                rep.add("sealed-ngrams", f, g)


# -- main ----------------------------------------------------------------------------------------------------------


def run(author: Path, require_sealed: bool = False) -> Report:
    waivers = json.loads(WAIVERS.read_text()) if WAIVERS.is_file() else []
    rep = Report(waivers)
    files = kit_files(author)
    check_files(author, files, rep)
    check_sealed_files(author, files, rep)
    docs = {f: norm(read_text(author / f) or "") for f in DOCS if (author / f).is_file()}
    raw_all = {f: read_text(author / f) for f in files}
    code = {f: norm(t) for f, t in raw_all.items() if t is not None and f.startswith(("kit/lib/", "tools/"))}
    ex_screens = example_screens(author)
    disclosed = set()
    for s in ex_screens.values():
        disclosed.update(screen_strings(s))
    check_calibration(docs, ex_screens, rep)
    templates, tables, idents, strings = generator_parts()
    if not generator_sources():
        rep.add("generator-code", "-", "generator sources not found",
                f"no *.py or *.ts in {GENERATOR}: the generator checks would pass on nothing")
    for f in docs:
        raw = raw_all.get(f) or ""
        for name in INTERNAL_NAMES:
            if name.lower() in raw.lower():
                rep.add("internal-names", f, name)
    for f, t in raw_all.items():
        if t is None:
            continue
        for tpl in templates:
            if tpl in t:
                rep.add("generator-template", f, tpl)
    for f, t in {**docs, **code}.items():
        for tab in tables:
            hit = run_found(tab, t, 4)
            if hit:
                if f == GUIDE_FILE:
                    rep.notes.append(f"generator-table: the generator uses the guide's list {hit!r}")
                else:
                    rep.add("generator-table", f, hit)
    for f in list(docs) + list(code):
        raw = raw_all.get(f) or ""
        for ident in sorted(idents):
            if re.search(rf"(?<![\w.]){re.escape(ident)}(?!\w)", raw):
                rep.add("generator-code", f, ident)
    gstrings = {norm(s) for s in strings if nwords(s) >= MIN_WORDS} - disclosed
    for f, t in docs.items():
        for s in sorted(gstrings):
            if s in t:
                if f == GUIDE_FILE:
                    rep.notes.append(f"generator-strings: the generator uses the guide's example {s!r}")
                else:
                    rep.add("generator-strings", f, s)
    for name, ents in rule_lists().items():
        for f, t in {**docs, **code}.items():
            hit = run_found(ents, t, 3, span=120)
            if hit:
                rep.add("rules-lists", f, f"{name}: {hit}")
    held = heldout_screens()
    ex_ids = {s.get("id"): f for f, s in ex_screens.items()}
    ex_seq = {seq_hash(s): f for f, s in ex_screens.items()}
    for name, screens in held.items():
        hstrings = set()
        for s in screens:
            if s.get("id") in ex_ids:
                rep.add("heldout", ex_ids[s["id"]], f"{name}:{s['id']}", "screen id of a held-out set")
            if seq_hash(s) in ex_seq:
                rep.add("heldout", ex_seq[seq_hash(s)], f"{name}:{s.get('id')}", "same node texts as a held-out screen")
            hstrings.update(t for t in screen_strings(s) if nwords(t) >= MIN_WORDS)
        for f, t in docs.items():
            for s in sorted(hstrings - disclosed):
                if s in t:
                    if f == GUIDE_FILE:
                        rep.notes.append(f"heldout: {name} holds the guide's example {s!r}")
                    else:
                        rep.add("heldout", f, f"{name}: {s}")
    rep.notes.append("heldout: sets checked: " + (", ".join(held) or "none") + "; absent: "
                     + (", ".join(s for s in HELDOUT_SETS if s not in held) or "none"))
    check_sealed_ngrams(author, docs, ex_screens, rep, require_sealed)
    unused = [w for i, w in enumerate(waivers) if i not in rep.used]
    for w in unused:
        rep.notes.append(f"unused waiver: {w.get('check')} {w.get('file')} {w.get('match')!r}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("author_dir")
    ap.add_argument("--require-sealed", action="store_true", help="fail when the sealed n-gram check cannot run")
    ap.add_argument("--json", help="write the full report here")
    a = ap.parse_args(argv)
    author = Path(a.author_dir).resolve()
    if not (author / "examples").is_dir():
        print(f"{author}: no examples/ (not a built kit)")
        return 2
    rep = run(author, a.require_sealed)
    for n in rep.notes:
        print("note:", n)
    for w in rep.waived:
        print(f"waived: {w['check']} {w['file']} {w['match']!r} ({w['reason']})")
    for p in rep.problems:
        print(f"LEAK: {p['check']} {p['file']} {p['match']!r}" + (f" ({p['detail']})" if p["detail"] else ""))
    if a.json:
        Path(a.json).write_text(json.dumps({"problems": rep.problems, "waived": rep.waived, "notes": rep.notes},
                                           indent=1, ensure_ascii=False) + "\n")
    print(f"leakage check: {len(rep.problems)} problem(s), {len(rep.waived)} waived")
    return 1 if rep.problems else 0


if __name__ == "__main__":
    sys.exit(main())
