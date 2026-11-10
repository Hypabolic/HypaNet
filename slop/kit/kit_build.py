"""Build and check the slop author kit (SPEC §13). Called by scripts/prepare_author_dir.sh (and, later, launch.sh).

    uv run python kit/kit_build.py lib [--check]                 # kit/lib/: hashed copies of ui/ porter, tokeniser, model
    uv run python kit/kit_build.py select [--run r2]             # print the disclosed dev screens and the rule's counts
    uv run python kit/kit_build.py tree AUTHOR_DIR [--run r2] [--allow-missing-docs]   # write the kit into AUTHOR_DIR
    uv run python kit/kit_build.py record AUTHOR_DIR RECORD --node NODE [--run r2] [--test-root DIR]   # HANDOVER.json
    uv run python kit/kit_build.py verify AUTHOR_DIR RECORD [--pristine]        # exit 1 on any difference
    uv run python kit/kit_build.py preview [--run r2] [--check]  # the rendered prompt preview of the run

Runs (RUNS below; `--run` defaults to r1, whose kit is unchanged):

    r1   400 disclosed dev screens, kit/prompt.md, work/ empty (the smoke launch uses the r1 kit too)
    r2   1,200 disclosed dev screens (r1's 400 plus the next 800 by the same rule), kit/prompt_r2.md, work/ a copy of
         runs/r1/work without scratch/, and resources/ (read-only): the codex (codex/CODEX.md, SOURCES.md, lists/),
         the thesaurus (resources/thesaurus: out/*.tsv, README.md and ATTRIBUTION.md as the author's copies) and
         resources/r1/RESULTS.md (kit/r1_results.md). The record keeps the seed's file hashes (`work_seed`), and
         `verify --pristine` then means "work/ is exactly the seed" instead of "work/ is empty".

What `tree` writes (everything else in the author dir comes from prepare_author_dir.sh):

    KIT.md, PROMPT.md          kit/KIT.md and the run's prompt (a TEST hand-over may use placeholders)
    HANDOVER.json              the author's view of the record (kind, time, author dir, file hashes, venv, node);
                               the full record (harness, inputs, selection) stays in slop/runs/<run>/HANDOVER.json
    LABELLING.md               the label guide with its internal preamble and cross-references replaced
                               (kit_guide(): GUIDE_EDITS, each applied exactly once; both hashes are recorded)
    kit/lib/                   porter.py, tokenizer.py, model.py, ids.json, unicode_tables.json, SHA256SUMS
    tools/py_guard.py          ./python3's guard (a byte copy of clouds/kit/tools/py_guard.py)
    examples/README.md         layout, the compact text form, the selection rule, the reference repos' product kinds
    examples/INDEX.txt         one line per screen: set, id, kind, framework, nodes, windows, codes present
    examples/dev/<id>.screen.json, <id>.labels.jsonl, <id>.txt   the 400 disclosed dev screens
    examples/reference/<id>.screen.json, [<id>.labels.jsonl], <id>.txt, <id>.source.<ext>
                                                                every reference_real screen (pause menu included)
                                                                and the whole source file it was extracted from

Disclosure rule (SPEC §9): dev screens are ranked by sha256("disclose:20261112:<screen id>") (hex, ascending). Pass 1
walks the ranking and takes a screen when it adds a node to a quota still short: every scored pattern of LABELLING.md §9
(R1-R6, R8, R9, R11, R14, R16, R17; R18 is track C) needs 10 positive nodes and every K rule (K1-K9, K11-K14; K10 is
track C) 10 KEEP nodes that name it. Pass 2 fills to the run's n (400 in r1, 1,200 in r2) with the best-ranked
screens not yet taken, so a larger n only adds screens (r2 checks that r1's 400 are all in its 1,200). A dev
screen whose normalised node-text sequence equals a val, test_gen or ood_codex screen's is not eligible (24 of 20,000
at the first build matched a val screen). Kind comes from the
generator's harness-only sidecar (<set>.meta.jsonl): SPEC §13 puts it in INDEX.txt. The sidecar's `clean` flag and
attempt number are never written. Reference labels are written only when labels/reference_real.final.jsonl exists;
a contested record (class null, conf "contested") keeps its two candidate readings, without labeller, reason, source
or adjudication; every other record drops `candidates`.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SLOP = Path(__file__).resolve().parents[1]
REPO = SLOP.parent
KIT = SLOP / "kit"
LIB = KIT / "lib"
UI = SLOP / "ui"
DATASETS = SLOP / "datasets"
DEV_DIR = DATASETS / "dev"
REF_DIR = DATASETS / "reference_real"
REF_LABELS = SLOP / "labels" / "reference_real.final.jsonl"
KINDS = DATASETS / "kinds.json"
GUIDE = SLOP / "LABELLING.md"
PY_GUARD = KIT / "tools" / "py_guard.py"
CLOUDS_PY_GUARD = REPO / "clouds" / "kit" / "tools" / "py_guard.py"
CODEX_DIR = SLOP / "codex"
THESAURUS = SLOP / "resources" / "thesaurus"
R1_WORK = SLOP / "runs" / "r1" / "work"
R1_RESULTS = KIT / "r1_results.md"
# the runs this script builds kits for; the smoke launch uses r1's kit
RUNS = {
    "r1": {"n_dev": 400, "prompt": "prompt.md", "preview": "PROMPT_RENDERED_PREVIEW.md", "seed": False,
           "resources": False},
    "r2": {"n_dev": 1200, "prompt": "prompt_r2.md", "preview": "PROMPT_R2_PREVIEW.md", "seed": True,
           "resources": True},
}
SEED_SKIP = ("scratch",)  # runs/r1/work/scratch: dumps of example screens and throwaway scripts, not handed over
SEED_SKIP_NAMES = (".DS_Store", "__pycache__")

LIB_FILES = ("porter.py", "tokenizer.py", "model.py", "ids.json", "unicode_tables.json")
SEED = 20261112
NAMESPACE = "disclose"
N_DEV = 400
QUOTA = 10
SCORED_PATTERNS = ("R1", "R2", "R3", "R4", "R5", "R6", "R8", "R9", "R11", "R14", "R16", "R17")
K_RULES = ("K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K11", "K12", "K13", "K14")
REDUNDANT = ("ECHO", "NARRATE", "DECOR")
CONTEXT_ROLES = ("aria", "alt", "code")
# keys a disclosed label record may carry (the generator's labeller, guide hash and time are dropped)
NODE_KEYS = ("screen", "loc", "class", "target", "equiv", "pattern", "aside", "partial", "ct", "conf", "role_ok",
             "role_fix", "candidates")
# keys of one candidate reading of a contested reference node (modellabel/adjudicate.py writes more: labeller, reason)
CAND_KEYS = ("class", "target", "pattern", "conf", "ct", "partial", "equiv", "aside")
SCREEN_KEYS = ("screen", "aside_screen", "missed")
RECORD_KIND = "hypanet-slop-author"
# harness files the author's tools run or the hand-over depends on; recorded at hand-over, checked before every launch
HARNESS_GLOBS = ("kit/tools/*.py", "kit/tools/*.mjs", "kit/lib/*", "kit/kit_build.py", "kit/make_settings.py",
                 "kit/leakage.py", "kit/leakage_waivers.json", "kit/KIT.md", "kit/prompt.md", "kit/prompt_continue.md",
                 "kit/prompt_r2.md", "kit/r1_results.md",
                 "ui/*.py", "ui/*.json", "lint/*.js", "lint/compile.py", "extract/cli.js", "extract/src/*.ts",
                 "rules/index.js", "rules/src/*.js", "package.json", "package-lock.json", "pyproject.toml", "uv.lock")
AUTHOR_SKIP_DIRS = (".venv", "work")
TOOLS = ("tok", "lint", "score", "inspect", "python3")
# the author's copy of HANDOVER.json: enough to see what was handed over, nothing about the harness or the data
AUTHOR_RECORD_KEYS = ("kind", "created_utc", "test", "author_dir", "files", "venv", "node")

# LABELLING.md as the author receives it: the preamble (labellers, generators, research notes, drafting history) and
# the cross-references to SPEC.md, which the author does not have, are replaced. Each edit must apply exactly once,
# so a revised guide fails the build until these are checked against it.
GUIDE_PREAMBLE_END = "\n## 1. The principle\n"
GUIDE_PREAMBLE = ("# HypaNet Slop: labelling guide\n\n"
                  "The definition of \"redundant copy\" for the slop experiment. Every labelled set and the author kit "
                  "use this\nguide. Pattern codes follow public anti-slop guidance (sources cited inline).\n\n"
                  "Worked examples here are invented, the motivating pause menu, or from the reference repos "
                  "(HypaChef, Mosaic,\nPersonalHomepage).\n")
GUIDE_EDITS = (
    ("for a screen (`SPEC.md` §4), one record", "for a screen (`KIT.md` §4), one record"),
    ("in the same window (`SPEC.md` §4).", "in the same window (`KIT.md` §4)."),
    ("not the headline (`SPEC.md` §10).", "not the headline (`KIT.md` §12)."),
    ("This guide labels it `KEEP`; `SPEC.md` §16 asks Matthew to confirm.", "This guide labels it `KEEP`."),
    ("Note on PAUSED: the user underlined it too. Microsoft's guide", "Note on PAUSED: Microsoft's guide"),
    ("the plain-code rule layer, `SPEC.md` §8, and never", "the plain-code rule layer, `KIT.md` §2, and never"),
    ("(the labeller lists the node ids; `SPEC.md` §5)", "(the labeller lists the node ids)"),
)
# none of these may remain in the author's copy (checked here and by kit/leakage.py)
GUIDE_FORBIDDEN = ("SPEC.md", "research/", "Matthew", "Codex", "corpus note", "sealed", "harness session", "Fable")


def run_for_dir(author: Path | str) -> str:
    """The run an author dir belongs to, by its name (scripts/prepare_author_dir.sh uses the same prefixes)."""
    return "r2" if Path(author).name.startswith("HypaNet-slop-author-r2") else "r1"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


HELDOUT_SETS = ("val", "test_gen", "ood_codex")


def norm(s: str) -> str:
    """Text normalisation of the leakage and duplicate checks: NFKC, one apostrophe, markers as spaces, case folded,
    white space collapsed."""
    s = unicodedata.normalize("NFKC", s).replace("’", "'").replace("‘", "'").replace("\ue000", " ").replace("\ue001",
                                                                                                          " ")
    return " ".join(s.casefold().split())


def seq_hash(screen: dict) -> str:
    """sha256 of a screen's normalised node-text sequence."""
    return hashlib.sha256("\x1f".join(norm(n.get("text", "")) for n in screen.get("nodes", [])).encode()).hexdigest()


def heldout_hashes() -> set[str]:
    out: set[str] = set()
    for name in HELDOUT_SETS:
        p = DATASETS / name / "screens.jsonl"
        if p.is_file():
            out.update(seq_hash(s) for s in iter_jsonl(p))
    return out


def rank_key(screen_id: str) -> str:
    return hashlib.sha256(f"{NAMESPACE}:{SEED}:{screen_id}".encode()).hexdigest()


def _code_order(c: str) -> tuple:
    prefix = c.split(":")[0] if ":" in c else ""
    body = c.split(":")[-1]
    num = "".join(ch for ch in body if ch.isdigit())
    return (prefix, body[0], int(num) if num else 0, body)


def kit_guide() -> str:
    """The author's LABELLING.md (see GUIDE_EDITS). Raises SystemExit if the guide no longer fits the edits."""
    text = GUIDE.read_text(encoding="utf-8")
    cut = text.find(GUIDE_PREAMBLE_END)
    if cut < 0 or not text.startswith("# HypaNet Slop: labelling guide\n"):
        raise SystemExit("LABELLING.md: title or '## 1. The principle' not found; update kit_build.GUIDE_* first")
    out = GUIDE_PREAMBLE + text[cut:]
    for old, new in GUIDE_EDITS:
        if out.count(old) != 1:
            raise SystemExit(f"LABELLING.md: {old!r} occurs {out.count(old)} times, not once; update "
                             "kit_build.GUIDE_EDITS")
        out = out.replace(old, new)
    left = [w for w in GUIDE_FORBIDDEN if w.lower() in out.lower()]
    if left:
        raise SystemExit(f"LABELLING.md: the author's copy still names {left}; extend kit_build.GUIDE_EDITS")
    return out


def author_record(record: dict) -> dict:
    """The author's HANDOVER.json: the record without the harness hashes, data inputs and selection details."""
    return {k: record[k] for k in AUTHOR_RECORD_KEYS if k in record}


# -- kit/lib -------------------------------------------------------------------------------------------------------


def lib_problems() -> list[str]:
    out = []
    for f in LIB_FILES:
        if not (LIB / f).is_file():
            out.append(f"kit/lib/{f} missing")
        elif sha256(LIB / f) != sha256(UI / f):
            out.append(f"kit/lib/{f} differs from ui/{f}")
    sums = LIB / "SHA256SUMS"
    want = "".join(f"{sha256(UI / f)}  {f}\n" for f in LIB_FILES)
    if not sums.is_file() or sums.read_text() != want:
        out.append("kit/lib/SHA256SUMS is stale")
    if not PY_GUARD.is_file() or not CLOUDS_PY_GUARD.is_file() or sha256(PY_GUARD) != sha256(CLOUDS_PY_GUARD):
        out.append("kit/tools/py_guard.py is not a byte copy of clouds/kit/tools/py_guard.py")
    return out


def cmd_lib(a) -> int:
    if a.check:
        probs = lib_problems()
        for p in probs:
            print("PROBLEM:", p)
        if not probs:
            print("kit/lib matches ui/; py_guard.py matches clouds")
        return 1 if probs else 0
    LIB.mkdir(parents=True, exist_ok=True)
    for f in LIB_FILES:
        dst = LIB / f
        if dst.exists():
            os.chmod(dst, 0o644)
        shutil.copyfile(UI / f, dst)
    (LIB / "SHA256SUMS").write_text("".join(f"{sha256(LIB / f)}  {f}\n" for f in LIB_FILES))
    print(f"kit/lib <- ui/ ({len(LIB_FILES)} files)")
    return 0


# -- data ----------------------------------------------------------------------------------------------------------


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dev_counts() -> tuple[dict[str, Counter], set[str]]:
    """Per dev screen: positives per scored pattern and KEEP nodes per K rule."""
    counts: dict[str, Counter] = defaultdict(Counter)
    seen: set[str] = set()
    for r in iter_jsonl(DEV_DIR / "dev.gen.jsonl"):
        sid = r["screen"]
        seen.add(sid)
        if "class" not in r:
            continue
        p = r.get("pattern")
        if r["class"] in REDUNDANT and p in SCORED_PATTERNS:
            counts[sid][p] += 1
        elif r["class"] == "KEEP" and p in K_RULES:
            counts[sid][p] += 1
    return counts, seen


def select_dev(n: int = N_DEV) -> dict:
    held = heldout_hashes()
    ids, excluded = [], 0
    for s in iter_jsonl(DEV_DIR / "screens.jsonl"):
        if seq_hash(s) in held:
            excluded += 1  # same node texts as a val / test_gen / ood_codex screen: never disclosed
        else:
            ids.append(s["id"])
    if len(set(ids)) != len(ids):
        raise SystemExit("dev screens.jsonl has duplicate ids")
    counts, labelled = dev_counts()
    missing = [i for i in ids if i not in labelled]
    if missing:
        raise SystemExit(f"{len(missing)} dev screens have no label records")
    ranked = sorted(ids, key=rank_key)
    quotas = {c: QUOTA for c in SCORED_PATTERNS + K_RULES}
    got: Counter = Counter()
    taken: list[str] = []
    taken_set: set[str] = set()
    for sid in ranked:
        if all(got[c] >= quotas[c] for c in quotas):
            break
        c = counts.get(sid, Counter())
        if any(c[k] > 0 and got[k] < quotas[k] for k in quotas):
            taken.append(sid)
            taken_set.add(sid)
            got.update({k: c[k] for k in quotas if c[k]})
    short = {c: got[c] for c in quotas if got[c] < quotas[c]}
    if short:
        raise SystemExit(f"dev set cannot meet the disclosure quotas: {short}")
    pass1 = len(taken)
    if pass1 > n:
        raise SystemExit(f"the quotas need {pass1} screens, more than {n}")
    for sid in ranked:
        if len(taken) >= n:
            break
        if sid not in taken_set:
            taken.append(sid)
            taken_set.add(sid)
    final: Counter = Counter()
    for sid in taken:
        final.update(counts.get(sid, Counter()))
    return {
        "rule": f'rank = sha256("{NAMESPACE}:{SEED}:<screen id>") hex ascending; pass 1 takes a screen that adds a '
                f"node to an unmet quota ({QUOTA} positive nodes per scored pattern, {QUOTA} KEEP nodes per K rule); "
                f"pass 2 fills to {n} in rank order; a screen whose node texts equal a held-out generated "
                f"screen's is never taken",
        "namespace": NAMESPACE, "seed": SEED, "quota": QUOTA, "n": n, "pass1_screens": pass1,
        "excluded_heldout_duplicates": excluded,
        "ids": sorted(taken, key=rank_key),
        "counts": {c: final[c] for c in SCORED_PATTERNS + K_RULES},
    }


def select_run(run: str) -> dict:
    """The run's disclosed dev screens. A run disclosing more than r1's 400 must disclose all of r1's: the rule's pass 2
    only adds screens as n grows, and this checks it (the held-out sets could have changed since r1)."""
    sel = select_dev(RUNS[run]["n_dev"])
    if RUNS[run]["n_dev"] > N_DEV:
        r1 = select_dev(N_DEV)["ids"]
        lost = set(r1) - set(sel["ids"])
        if lost:
            raise SystemExit(f"{len(lost)} of r1's {N_DEV} disclosed dev screens are not in the {run} selection")
        sel["r1_ids_sha256"] = hashlib.sha256("\n".join(r1).encode()).hexdigest()
        sel["added_to_r1"] = len(sel["ids"]) - len(r1)
    return sel


def cmd_select(a) -> int:
    sel = select_run(a.run)
    print(sel["rule"])
    print(f"pass 1: {sel['pass1_screens']} screens; total {len(sel['ids'])}; not eligible (held-out duplicates): "
          f"{sel['excluded_heldout_duplicates']}")
    print(" ".join(f"{c}={n}" for c, n in sel["counts"].items()))
    return 0


def is_contested(r: dict) -> bool:
    return "class" in r and r["class"] is None and r.get("conf") == "contested"


def strip_record(r: dict) -> dict:
    keys = NODE_KEYS if "class" in r else SCREEN_KEYS
    out = {k: r[k] for k in keys if k in r}
    if "candidates" in out:
        if is_contested(r):
            out["candidates"] = [{k: c[k] for k in CAND_KEYS if k in c} for c in r["candidates"]]
        else:
            del out["candidates"]
    return out


def read_labels(path: Path, ids: set[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in iter_jsonl(path):
        if r.get("screen") in ids and ("class" in r or "aside_screen" in r or "missed" in r):
            out[r["screen"]].append(strip_record(r))
    return out


# -- compact text form ---------------------------------------------------------------------------------------------


def show_text(text: str) -> str:
    return json.dumps(text.replace("", "⟨VAR⟩").replace("", "⟨ALT⟩"), ensure_ascii=False)


def compact(screen: dict, labels: list[dict] | None, header: str) -> str:
    nodes = screen["nodes"]
    by_loc = {tuple(n["loc"]): x for x, n in enumerate(nodes)}
    recs = {tuple(r["loc"]): r for r in (labels or []) if "class" in r}
    lines = [header]
    kinds = {c["k"]: c for c in screen.get("clusters", [])}
    last_k = None
    for x, n in enumerate(nodes):
        if n["k"] != last_k:
            c = kinds.get(n["k"], {})
            lines.append(f"-- cluster {n['k']} {c.get('kind', '?')} (line {c.get('line', '?')})")
            last_k = n["k"]
        flags = ",".join(n.get("flags") or []) or "-"
        if labels is None:
            lab = "(unlabelled)"
        else:
            r = recs.get(tuple(n["loc"]))
            if r is None:
                ctx = n["role"] in CONTEXT_ROLES or "sr_only" in (n.get("flags") or [])
                lab = "(context)" if ctx else "(no label)"
            elif is_contested(r):
                lab = "CONTESTED " + " | ".join(
                    f"{'AB'[j] if j < 2 else j}:{_reading(c, nodes, by_loc)}" for j, c in enumerate(r.get("candidates")
                                                                                         or []))
            elif r["class"] is None:
                lab = "UNRESOLVED"
            else:
                lab = r["class"]
                if r["class"] in ("ECHO", "NARRATE"):
                    t = r.get("target")
                    lab += "→" + ("-" if not t else str(nodes[by_loc[tuple(t)]]["i"]) if tuple(t) in by_loc else "?")
                if r.get("pattern"):
                    lab += " " + r["pattern"]
                lab += " " + str(r.get("conf", "?"))
                if r.get("ct"):
                    lab += " ct"
                if r.get("partial"):
                    lab += " partial"
                if r.get("aside"):
                    lab += " aside:" + ",".join(r["aside"])
                if r.get("equiv"):
                    lab += " equiv:" + ",".join(str(nodes[by_loc[tuple(e)]]["i"]) if tuple(e) in by_loc else "?"
                                                for e in r["equiv"])
        lines.append(f"{n['i']} {n['k']} {n['role']} {flags} | {show_text(n['text'])} | {lab}")
    for r in labels or []:
        for a in r.get("aside_screen", []):
            ids = [str(nodes[by_loc[tuple(l)]]["i"]) if tuple(l) in by_loc else "?" for l in a.get("nodes", [])]
            lines.append(f"screen aside {a.get('code')}: nodes {', '.join(ids)}")
        for m in r.get("missed", []):
            lines.append(f"missed: line {m.get('line')} {show_text(str(m.get('text', '')))} {m.get('class')}"
                         + (f" {m['pattern']}" if m.get("pattern") else ""))
    return "\n".join(lines) + "\n"


def _reading(c: dict, nodes: list[dict], by_loc: dict) -> str:
    """One candidate reading of a contested node, in the compact form."""
    s = str(c.get("class"))
    if c.get("class") in ("ECHO", "NARRATE"):
        t = c.get("target")
        s += "→" + ("-" if not t else str(nodes[by_loc[tuple(t)]]["i"]) if tuple(t) in by_loc else "?")
    if c.get("pattern"):
        s += " " + c["pattern"]
    if c.get("ct"):
        s += " ct"
    return s


def codes_present(labels: list[dict] | None) -> str:
    if labels is None:
        return "(unlabelled)"
    codes = set()
    for r in labels:
        if "class" in r:
            if r.get("pattern"):
                codes.add(r["pattern"])
            for a in r.get("aside") or []:
                codes.add("aside:" + a)
        for a in r.get("aside_screen", []):
            codes.add("screen:" + a["code"])
    return " ".join(sorted(codes, key=_code_order)) or "-"


def _windows(screen: dict) -> int:
    sys.path.insert(0, str(LIB))
    try:
        import tokenizer as tk  # the kit/lib copy
    finally:
        sys.path.pop(0)
    return len(tk.encode_screen(screen).windows)


def _dump(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def screen_json(screen: dict) -> str:
    return json.dumps(screen, ensure_ascii=False, indent=1) + "\n"


def labels_jsonl(labels: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels)


README_TEMPLATE = """# Examples

Two sets of screens in the extractor's `slop.screen/1` form (`KIT.md` §4), with their labels in the record format of
`KIT.md` §5 (one JSON object per line).

## Layout

| Path | What it is |
|---|---|
| `INDEX.txt` | One line per screen: set, screen id, kind, framework, node count, window count, codes present |
| `dev/<id>.screen.json` | One of the {n_dev} disclosed generated screens |
| `dev/<id>.labels.jsonl` | Its labels: one record per labelled node (keyed by `loc`), and any screen-level aside |
| `dev/<id>.txt` | The same screen in the compact text form (below) |
| `reference/<id>.screen.json`, `reference/<id>.txt` | The {n_ref} real reference screens, the pause-menu fixture included |
| `reference/<id>.labels.jsonl` | {ref_labels_row} |
| `reference/<id>.source.<ext>` | The whole source file the screen was extracted from (its repo and path are in the `.txt` header) |

## Compact text form

```
# <set> <id>  <framework>  <kind>  <nodes> nodes  <path>
-- cluster <k> <cluster kind> (line <n>)
<i> <k> <role> <flags> | "<text>" | <CLASS>[→<target i>] [<pattern>] <conf> [ct] [partial] [aside:<codes>] [equiv:<i>,...]
screen aside <code>: nodes <i>, ...
```

`→-` is an `ECHO` or `NARRATE` with no single target; `(context)` marks a context-only node (`aria`, `alt`, `code`,
`sr_only`), which is never labelled; `UNRESOLVED` is a node with no literal words; `CONTESTED A:<reading> | B:<reading>`
is a reference node whose answer key did not settle on one label (both readings shown; not scored); `(unlabelled)` is a
node of an unlabelled screen. In the text, `⟨VAR⟩` and `⟨ALT⟩` stand for the extractor's markers U+E000 and U+E001
(`KIT.md` §4); the JSON files hold the markers themselves.

## The {n_dev} dev screens

Generated screens, built around one product each, with exact labels from the generator. They were chosen by a fixed
hash rule, not by hand: {rule}. Positive nodes per scored pattern and KEEP nodes per K rule in these screens:

{counts}

`kind` in `INDEX.txt` is the kind of screen the generator was asked for: `app`, `dashboard`, `dialog`, `empty`,
`game`, `marketing` or `settings`. It is not in the screen files and the network never sees it.

## The reference screens

Real screens from three repos plus the pause-menu fixture. The product kind of each repo, the one line of
evidence about the product the label guide allows (`LABELLING.md` §2a):

{kinds}

{ref_label_note}

The screen files were extracted with each repo's own stylesheets. The source files here are copied without those
stylesheets, so `./lint` on a source file can give a style flag (and so a role) that differs from the screen file;
the screen file is the one that counts.
"""


def write_examples(author: Path, run: str = "r1") -> dict:
    ex = author / "examples"
    if ex.exists():
        raise SystemExit(f"{ex} exists")
    sel = select_run(run)
    ids = sel["ids"]
    idset = set(ids)
    screens = {s["id"]: s for s in iter_jsonl(DEV_DIR / "screens.jsonl") if s["id"] in idset}
    meta = {m["screen"]: m for m in iter_jsonl(DEV_DIR / "dev.meta.jsonl") if m["screen"] in idset}
    labels = read_labels(DEV_DIR / "dev.gen.jsonl", idset)
    index = ["# set id kind framework nodes windows codes",
             "# codes: pattern and K-rule codes on node labels; aside:<code> node asides; screen:<code> screen asides"]
    for sid in ids:
        s = screens[sid]
        kind = meta[sid]["kind"]
        lab = labels[sid]
        _dump(ex / "dev" / f"{sid}.screen.json", screen_json(s))
        _dump(ex / "dev" / f"{sid}.labels.jsonl", labels_jsonl(lab))
        head = f"# dev {sid}  {s['framework']}  {kind}  {len(s['nodes'])} nodes  {s['source']['path']}"
        _dump(ex / "dev" / f"{sid}.txt", compact(s, lab, head))
        index.append(f"dev {sid} {kind} {s['framework']} {len(s['nodes'])} {_windows(s)} {codes_present(lab)}")

    man = json.loads((REF_DIR / "manifest.json").read_text())
    kinds = json.loads(KINDS.read_text())
    ref_ids = [e["id"] for e in man["screens"]]
    fixture = man.get("fixture", {}).get("id")
    if fixture and fixture not in ref_ids:
        ref_ids.append(fixture)
    have_labels = REF_LABELS.is_file()
    ref_labels = read_labels(REF_LABELS, set(ref_ids)) if have_labels else {}
    repos_used = []
    for sid in ref_ids:
        s = json.loads((REF_DIR / "screens" / f"{sid}.json").read_text(encoding="utf-8"))
        assert s["id"] == sid, sid
        repo, path = s["source"]["repo"], s["source"]["path"]
        if repo not in repos_used:
            repos_used.append(repo)
        src = REF_DIR / "sources" / repo / path
        dst = ex / "reference" / f"{sid}.source{Path(path).suffix}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        _dump(ex / "reference" / f"{sid}.screen.json", screen_json(s))
        lab = ref_labels.get(sid, []) if have_labels else None
        if have_labels:
            _dump(ex / "reference" / f"{sid}.labels.jsonl", labels_jsonl(lab))
        head = f"# reference {sid}  {s['framework']}  {repo}  {len(s['nodes'])} nodes  {path}"
        _dump(ex / "reference" / f"{sid}.txt", compact(s, lab, head))
        index.append(f"reference {sid} real:{repo} {s['framework']} {len(s['nodes'])} {_windows(s)} "
                     f"{codes_present(lab)}")
    _dump(ex / "INDEX.txt", "\n".join(index) + "\n")
    counts = ", ".join(f"{c} {n}" for c, n in sel["counts"].items())
    if have_labels:
        ref_row = "Its labels (the answer key)"
        ref_note = ("The labels are the answer key of the reference screens; `./score ART --set reference` scores "
                    "against them.")
    else:
        ref_row = "Not present: the reference screens are unlabelled in this kit"
        ref_note = ("These are unlabelled real screens: their answer key was not ready when this kit was built, so "
                    "`./score --set reference` is unavailable. Use them to see what real UI copy looks like and run "
                    "your network on them (`./score ART --files examples/reference/<id>.screen.json` prints its calls).")
    readme = README_TEMPLATE.format(
        n_dev=len(ids), n_ref=len(ref_ids), rule=sel["rule"], counts=counts,
        kinds="\n".join(f"- {r}: {kinds.get(r, '(not recorded)')}" for r in repos_used),
        ref_labels_row=ref_row, ref_label_note=ref_note)
    _dump(ex / "README.md", readme)
    return {"run": run,
            "selection": {k: sel[k] for k in ("rule", "namespace", "seed", "quota", "n", "pass1_screens",
                                              "excluded_heldout_duplicates", "counts", "r1_ids_sha256",
                                              "added_to_r1") if k in sel},
            "dev_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "reference_screens": len(ref_ids), "reference_labels": have_labels,
            "guide_sha256": sha256(GUIDE),
            "kit_guide_sha256": hashlib.sha256(kit_guide().encode("utf-8")).hexdigest()}


# -- r2: resources/ and the work/ seed ---------------------------------------------------------------------------------

# the thesaurus README as the author receives it: the repo's build commands, coverage section (it is measured over the
# whole dev set) and repo paths are replaced; each edit must apply exactly once, so a revised README fails the build
THESAURUS_README_HEAD = """# Thesaurus

A full open English thesaurus, reformatted so you can look up synonyms and related words **by the stems the slop
tokeniser produces** (`kit/lib/tokenizer.py`, `KIT.md` §6). Two sources: Open English WordNet 2025 (sense-level
synonym sets with glosses and typed links) and Moby Thesaurus II (broad, loose near-synonym lists). It is read-only
reference data: which entries your lexicon takes from it, and how, is your judgement (`KIT.md` §11, §14). Licences and
attribution: `ATTRIBUTION.md` (CC BY 4.0 and the WordNet licence for OEWN; Moby is public domain).

"""
THESAURUS_README_EDITS = (
    ("`r` adverb. `MANIFEST.json` (committed) records each output's sha256, bytes, rows and whether it is committed,\n"
     "plus the sources' pins, the `ui/porter.py` hash and the build statistics.", "`r` adverb."),
    ("Files over 5 MB are gitignored and rebuilt by `build.py`; `tests/` checks every file against the manifest when it\n"
     "is present. `related.tsv`", "`related.tsv`"),
    ("Every word is run through `ui/tokenizer.py`'s `tokenize_text` (NFKC, letter runs, camel-case split, case fold,\n"
     "accent strip, `ui/porter.py`)", "Every word is run through `kit/lib/tokenizer.py`'s `tokenize_text` (NFKC, letter runs, "
     "camel-case split, case\nfold, accent strip, `kit/lib/porter.py`)"),
    ("| File | Rows | Size | In git | Columns |\n| --- | ---: | ---: | --- | --- |",
     "| File | Rows | Size | Columns |\n| --- | ---: | ---: | --- |"),
)
THESAURUS_OUT = ("stems.tsv", "stem_links.tsv", "synsets.tsv", "words.tsv", "related.tsv", "moby.tsv")


def thesaurus_readme() -> str:
    """The author's copy of resources/thesaurus/README.md (see THESAURUS_README_EDITS)."""
    text = (THESAURUS / "README.md").read_text(encoding="utf-8")
    try:
        body = text[text.index("## Sources and why\n"):text.index("## Coverage\n")]
    except ValueError:
        raise SystemExit("resources/thesaurus/README.md: '## Sources and why' or '## Coverage' not found; update "
                         "kit_build.thesaurus_readme") from None
    for old, new in THESAURUS_README_EDITS:
        if body.count(old) != 1:
            raise SystemExit(f"resources/thesaurus/README.md: {old[:60]!r} occurs {body.count(old)} times, not once; "
                             "update kit_build.THESAURUS_README_EDITS")
        body = body.replace(old, new)
    body = "\n".join(_drop_in_git(ln) for ln in body.splitlines())
    if "ui/" in body or "gitignored" in body or "MANIFEST" in body:
        raise SystemExit("resources/thesaurus/README.md: the author's copy still names repo files; extend "
                         "kit_build.THESAURUS_README_EDITS")
    return THESAURUS_README_HEAD + body.rstrip("\n") + "\n"


# the thesaurus ATTRIBUTION.md as the author receives it: its opening paragraph names the repo's fetch and build
# scripts and the gitignored raw/ directory, none of which is in the kit (and build.py could be mistaken for
# work/build.py); the licences, sources and "Changes made" stay word for word.
THESAURUS_ATTRIBUTION_EDIT = (
    "The tables in `out/` are derived from the two sources below. `fetch.py` downloads them unchanged into `raw/`\n"
    "(gitignored); `build.py` reformats them (see \"Changes made\"). The code in this directory (`fetch.py`, `build.py`,\n"
    "`coverage.py`, `tests/`) is part of HypaNet and under the repository's licence.",
    "The tables in `out/` are reformatted extracts of the two sources below (see \"Changes made\" under each).")


def thesaurus_attribution() -> str:
    """The author's copy of resources/thesaurus/ATTRIBUTION.md (see THESAURUS_ATTRIBUTION_EDIT)."""
    text = (THESAURUS / "ATTRIBUTION.md").read_text(encoding="utf-8")
    old, new = THESAURUS_ATTRIBUTION_EDIT
    if text.count(old) != 1:
        raise SystemExit("resources/thesaurus/ATTRIBUTION.md: its opening paragraph changed; update "
                         "kit_build.THESAURUS_ATTRIBUTION_EDIT")
    text = text.replace(old, new)
    for word in ("fetch.py", "build.py", "raw/", "gitignored", "coverage.py"):
        if word in text:
            raise SystemExit(f"resources/thesaurus/ATTRIBUTION.md: the author's copy still names {word}; update "
                             "kit_build.THESAURUS_ATTRIBUTION_EDIT")
    return text


def _drop_in_git(line: str) -> str:
    """Rows of the README's file table lose their "In git" cell (committed / ignored)."""
    cells = line.split(" | ")
    if line.startswith("| `out/") and len(cells) == 5 and cells[3] in ("committed", "ignored"):
        return " | ".join(cells[:3] + cells[4:])
    return line


def thesaurus_problems() -> list[str]:
    """The built thesaurus outputs must exist and match resources/thesaurus/MANIFEST.json."""
    try:
        man = json.loads((THESAURUS / "MANIFEST.json").read_text())["outputs"]
    except (OSError, ValueError, KeyError):
        return ["resources/thesaurus/MANIFEST.json missing or unreadable (uv run python -I resources/thesaurus/build.py)"]
    out = []
    for f in THESAURUS_OUT:
        p = THESAURUS / "out" / f
        want = (man.get(f"out/{f}") or {}).get("sha256")
        if not p.is_file():
            out.append(f"resources/thesaurus/out/{f} missing (uv run python -I resources/thesaurus/build.py)")
        elif sha256(p) != want:
            out.append(f"resources/thesaurus/out/{f} does not match MANIFEST.json (rebuild the thesaurus)")
    return out


def resource_files() -> dict[str, Path | str]:
    """resources/ of an r2 kit: author path -> source file (a byte copy) or text (the thesaurus README and
    ATTRIBUTION.md, as the author's copies)."""
    out: dict[str, Path | str] = {"resources/codex/CODEX.md": CODEX_DIR / "CODEX.md",
                                  "resources/codex/SOURCES.md": CODEX_DIR / "SOURCES.md"}
    for p in sorted((CODEX_DIR / "lists").glob("*")):
        if p.is_file() and p.suffix in (".txt", ".md"):
            out[f"resources/codex/lists/{p.name}"] = p
    out["resources/thesaurus/README.md"] = thesaurus_readme()
    out["resources/thesaurus/ATTRIBUTION.md"] = thesaurus_attribution()
    for f in THESAURUS_OUT:
        out[f"resources/thesaurus/out/{f}"] = THESAURUS / "out" / f
    out["resources/r1/RESULTS.md"] = R1_RESULTS
    return out


def resource_problems() -> list[str]:
    out = thesaurus_problems()
    for f in ("CODEX.md", "SOURCES.md", "lists/ATTRIBUTION-ui.md"):
        if not (CODEX_DIR / f).is_file():
            out.append(f"codex/{f} missing")
    if not R1_RESULTS.is_file():
        out.append("kit/r1_results.md missing")
    return out


def work_seed_files() -> dict[str, Path]:
    """r2's work/ seed: runs/r1/work without scratch/ (rel path -> source)."""
    if not (R1_WORK / "build.py").is_file():
        raise SystemExit(f"{R1_WORK} has no build.py: the r2 kit starts from r1's work/")
    out = {}
    for root, dirs, files in os.walk(R1_WORK):
        rel_root = os.path.relpath(root, R1_WORK)
        dirs[:] = sorted(d for d in dirs if d not in SEED_SKIP_NAMES and not (rel_root == "." and d in SEED_SKIP))
        for f in sorted(files):
            if f in SEED_SKIP_NAMES or f.endswith(".pyc"):
                continue
            out[os.path.normpath(os.path.join(rel_root, f))] = Path(root) / f
    return dict(sorted(out.items()))


def write_resources(author: Path) -> dict:
    files = resource_files()
    for rel, src in files.items():
        dst = author / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(src, Path):
            shutil.copyfile(src, dst)
        else:
            dst.write_text(src, encoding="utf-8")
    return {"files": len(files), "bytes": sum((author / r).stat().st_size for r in files)}


def write_seed(author: Path) -> dict[str, str]:
    work = author / "work"
    if any(work.iterdir()):
        raise SystemExit(f"{work} is not empty")
    seed = work_seed_files()
    for rel, src in seed.items():
        dst = work / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        os.chmod(dst, 0o644)
    return {rel: sha256(work / rel) for rel in seed}


def cmd_tree(a) -> int:
    author = Path(a.author_dir).resolve()
    if not author.is_dir():
        raise SystemExit(f"{author} is not a directory")
    cfg = RUNS[a.run]
    probs = lib_problems()
    if probs:
        raise SystemExit("kit/lib or py_guard out of date: " + "; ".join(probs) + " (run kit_build.py lib)")
    if cfg["resources"]:
        probs = resource_problems()
        if probs:
            raise SystemExit("resources not ready: " + "; ".join(probs))
    for src, dst, what in ((KIT / "KIT.md", "KIT.md", "the author's manual"),
                           (KIT / cfg["prompt"], "PROMPT.md", "the launch prompt")):
        if src.is_file():
            shutil.copyfile(src, author / dst)
        elif a.allow_missing_docs:
            (author / dst).write_text(f"# {dst}\n\nPlaceholder in a TEST hand-over: kit/{src.name} ({what}) is not "
                                      "written yet.\n")
        else:
            raise SystemExit(f"kit/{src.name} missing ({what}); a real hand-over needs it")
    (author / "LABELLING.md").write_text(kit_guide(), encoding="utf-8")
    (author / "kit" / "lib").mkdir(parents=True, exist_ok=True)
    for f in LIB_FILES + ("SHA256SUMS",):
        shutil.copyfile(LIB / f, author / "kit" / "lib" / f)
    (author / "tools").mkdir(exist_ok=True)
    shutil.copyfile(PY_GUARD, author / "tools" / "py_guard.py")
    info = write_examples(author, a.run)
    if cfg["resources"]:
        info["resources"] = write_resources(author)
    if cfg["seed"]:
        info["work_seed"] = write_seed(author)
    (KIT / "build").mkdir(exist_ok=True)
    (KIT / "build" / "tree.json").write_text(json.dumps(info, indent=2) + "\n")
    extra = ""
    if cfg["resources"]:
        extra += f", {info['resources']['files']} resource files"
    if cfg["seed"]:
        extra += f", work/ seeded with {len(info['work_seed'])} files of runs/r1/work"
    print(f"kit tree ({a.run}) -> {author}: {info['selection']['n']} dev screens (pass 1 "
          f"{info['selection']['pass1_screens']}), {info['reference_screens']} reference screens, reference labels "
          f"{'included' if info['reference_labels'] else 'absent (unlabelled)'}{extra}")
    return 0


# -- hand-over record ----------------------------------------------------------------------------------------------


def author_files(author: Path) -> dict[str, str]:
    out = {}
    for root, dirs, files in os.walk(author):
        rel_root = os.path.relpath(root, author)
        if rel_root == ".":
            dirs[:] = [d for d in dirs if d not in AUTHOR_SKIP_DIRS]
        for f in files:
            rel = os.path.normpath(os.path.join(rel_root, f))
            if rel == "HANDOVER.json":
                continue
            out[rel] = sha256(author / rel)
    return dict(sorted(out.items()))


def harness_files() -> dict[str, str]:
    out = {}
    for g in HARNESS_GLOBS:
        for p in sorted(SLOP.glob(g)):
            if p.is_file():
                out[str(p.relative_to(SLOP))] = sha256(p)
    return dict(sorted(out.items()))


def inputs() -> dict[str, str | None]:
    files = {"datasets/dev/manifest.json": DEV_DIR / "manifest.json", "datasets/dev/screens.jsonl":
             DEV_DIR / "screens.jsonl", "datasets/dev/dev.gen.jsonl": DEV_DIR / "dev.gen.jsonl",
             "datasets/dev/dev.meta.jsonl": DEV_DIR / "dev.meta.jsonl", "datasets/reference_real/manifest.json":
             REF_DIR / "manifest.json", "datasets/kinds.json": KINDS, "LABELLING.md": GUIDE,
             "labels/reference_real.final.jsonl": REF_LABELS}
    return {k: (sha256(p) if p.is_file() else None) for k, p in files.items()}


def venv_info(author: Path) -> dict:
    py = author / ".venv" / "bin" / "python"
    code = ("import sys, numpy, importlib.metadata as m; print(sys.version.split()[0]); print(numpy.__version__); "
            "print(','.join(sorted(d.metadata['Name'] for d in m.distributions())))")
    r = subprocess.run([str(py), "-I", "-B", "-c", code], capture_output=True, text=True, check=True)
    ver, npv, dists = r.stdout.strip().splitlines()
    return {"python": ver, "numpy": npv, "distributions": dists.split(",")}


def node_info(node: str) -> dict:
    """The node binary the author's wrappers run (./lint's extractor and runtime): path and version."""
    r = subprocess.run([node, "--version"], capture_output=True, text=True, check=True)
    return {"path": node, "version": r.stdout.strip()}


def cmd_record(a) -> int:
    author = Path(a.author_dir).resolve()
    tree = KIT / "build" / "tree.json"
    examples = json.loads(tree.read_text()) if tree.is_file() else None
    if examples is not None and examples.get("run", "r1") != a.run:
        raise SystemExit(f"kit/build/tree.json was written for run {examples.get('run', 'r1')}, not {a.run}")
    seed = examples.pop("work_seed", None) if examples else None
    if RUNS[a.run]["seed"]:
        if seed is None:
            raise SystemExit("kit/build/tree.json has no work_seed (run kit_build.py tree --run r2 first)")
        now = {rel: sha256(author / "work" / rel) for rel in seed if (author / "work" / rel).is_file()}
        if now != seed or work_files(author) != set(seed):
            raise SystemExit("work/ is not the seed the tree wrote")
    rec = {
        "kind": RECORD_KIND,
        "run": a.run,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test": bool(a.test_root),
        "test_root": a.test_root,
        "author_dir": str(author),
        "files": author_files(author),
        "venv": venv_info(author),
        "node": node_info(a.node),
        "harness": harness_files(),
        "inputs": inputs(),
        "examples": examples,
    }
    if seed is not None:
        rec["work_seed"] = seed
    text = json.dumps(rec, indent=2) + "\n"
    (author / "HANDOVER.json").write_text(json.dumps(author_record(rec), indent=2) + "\n")
    Path(a.record).parent.mkdir(parents=True, exist_ok=True)
    Path(a.record).write_text(text)
    print(f"hand-over record: {len(rec['files'])} author files, {len(rec['harness'])} harness files -> {a.record}")
    return 0


def work_files(author: Path) -> set[str]:
    work = author / "work"
    out = set()
    for root, _dirs, files in os.walk(work):
        for f in files:
            out.add(os.path.relpath(os.path.join(root, f), work))
    return out


def problems(author: Path, record: dict, pristine: bool) -> list[str]:
    out = []
    if record.get("kind") != RECORD_KIND:
        return ["the record is not a HypaNet slop hand-over record"]
    if os.path.realpath(author) != record["author_dir"]:
        out.append(f"author dir {author} is not the recorded {record['author_dir']}")
    hv = author / "HANDOVER.json"
    if not hv.is_file() or json.loads(hv.read_text()) != author_record(record):
        out.append("AUTHOR_DIR/HANDOVER.json differs from the record")
    now = author_files(author)
    for rel, h in record["files"].items():
        if rel not in now:
            out.append(f"missing: {rel}")
        elif now[rel] != h:
            out.append(f"changed: {rel}")
    for rel in now:
        if rel not in record["files"]:
            out.append(f"added outside work/: {rel}")
    hnow = harness_files()
    for rel, h in record["harness"].items():
        if hnow.get(rel) != h:
            out.append(f"harness file changed since the hand-over: {rel}")
    for rel in hnow:
        if rel not in record["harness"]:
            out.append(f"harness file added since the hand-over: {rel}")
    inow = inputs()
    for rel, h in record.get("inputs", {}).items():
        if inow.get(rel) != h:
            out.append(f"kit input changed since the hand-over: {rel}")
    try:
        if venv_info(author) != record["venv"]:
            out.append("the private venv differs from the hand-over")
    except (subprocess.CalledProcessError, OSError, ValueError) as e:
        out.append(f"the private venv does not run: {e}")
    if "node" in record:
        try:
            if node_info(record["node"]["path"]) != record["node"]:
                out.append(f"node differs from the hand-over ({record['node']['version']} at {record['node']['path']})")
        except (subprocess.CalledProcessError, OSError) as e:
            out.append(f"the recorded node does not run: {e}")
    work = author / "work"
    seed = record.get("work_seed")
    if not work.is_dir():
        out.append("work/ missing")
    elif pristine and seed is None and any(work.iterdir()):
        out.append("work/ is not empty (a fresh session needs a pristine hand-over)")
    elif pristine and seed is not None:
        have = work_files(author)
        for rel in sorted(set(seed) - have):
            out.append(f"work/ seed file missing: {rel} (a fresh session needs the seed exactly)")
        for rel in sorted(have - set(seed)):
            out.append(f"work/ has a file the seed does not: {rel} (a fresh session needs the seed exactly)")
        for rel in sorted(set(seed) & have):
            if sha256(work / rel) != seed[rel]:
                out.append(f"work/ seed file changed: {rel} (a fresh session needs the seed exactly)")
    if os.access(author, os.W_OK):
        out.append("the author dir itself is writable")
    for d in ("examples", "kit", "tools") + (("resources",) if (author / "resources").exists() else ()):
        for p in [author / d, *(author / d).rglob("*")]:
            if os.access(p, os.W_OK):
                out.append(f"writable: {p.relative_to(author)}")
                break
    for f in ("KIT.md", "PROMPT.md", "LABELLING.md", "HANDOVER.json", *TOOLS):
        if os.access(author / f, os.W_OK):
            out.append(f"writable: {f}")
    return out


def cmd_verify(a) -> int:
    author = Path(a.author_dir).resolve()
    record = json.loads(Path(a.record).read_text())
    probs = problems(author, record, a.pristine)
    for p in probs:
        print("PROBLEM:", p)
    if probs:
        return 1
    print(f"hand-over verified ({record.get('run', 'r1')}): {len(record['files'])} author files, "
          f"{len(record['harness'])} harness files" + (" (TEST hand-over)" if record["test"] else ""))
    return 0


PREVIEW_TEMPLATE = """# Prompt preview: what Fable 5.2 receives{title_run}

Rendered by `uv run python kit/kit_build.py preview{run_flag}` from the kit files below. This is a review copy only:
the launcher reads `kit/{prompt_name}` and `kit/prompt_continue.md`, never this file. `preview{run_flag} --check`
fails when any sha256 below no longer matches; re-render then.

| File | sha256 at render time |
|---|---|
| `kit/{prompt_name}` | `{prompt}` |
| `kit/prompt_continue.md` | `{cont}` |
| `kit/KIT.md` | `{kit}` |
| `LABELLING.md` (the frozen guide) | `{guide}` |
| `LABELLING.md` as the author receives it (`kit_build.kit_guide()`) | `{kit_guide}` |
{extra_rows}
## How it is delivered

{delivery}
- The cost cap exists only as the launcher's `--max-budget-usd` flag; no message or kit file Fable receives states a
  spending budget, a cost cap or a price (the word "cost" appears only in `LABELLING.md`'s consequence test{cost_note}).

## Message 1: `start` (verbatim, {n_prompt} characters)

~~~~text
{prompt_text}
~~~~

## Message on `continue K` (verbatim, {n_cont} characters)

~~~~text
{cont_text}
~~~~
"""
DELIVERY = {
    "r1": """- `smoke` and `start`: the whole of `kit/prompt.md` is the one user message of a new headless session
  (`claude -p "$PROMPT" --model claude-fable-5-2 --effort xhigh ...`, `SPEC.md` §13; the shell's `$(cat ...)` drops
  the final newline), run in `../HypaNet-slop-smoke/` or `../HypaNet-slop-author/`. No other user text is sent.
  Claude Code adds its own system prompt, which is not reproduced here; `CLAUDE.md` files and auto-memory are off.
- `continue K`: the whole of `kit/prompt_continue.md` as the next user message of the same session (`--resume`),
  only after a launch that ended without `MODEL COMPLETE` (an API or limit error, or a reply with no tool call).
- In the author directory Fable can read `PROMPT.md` (a copy of `kit/prompt.md`), `KIT.md` (a copy of `kit/KIT.md`),
  `LABELLING.md` (the guide with its internal preamble and its cross-references to `SPEC.md` replaced; the hash
  above), `examples/` and `HANDOVER.json` (file hashes only). `KIT.md` and `LABELLING.md` are not inlined in any
  message.""",
    "r2": """- `start` (`scripts/launch.sh --run r2 start`): the whole of `kit/prompt_r2.md` is the one user message of a
  new headless session (`claude -p "$PROMPT" --model claude-fable-5-2 --effort xhigh ...`, `SPEC.md` §13; the
  shell's `$(cat ...)` drops the final newline), run in `../HypaNet-slop-author-r2/`. No other user text is sent.
  Claude Code adds its own system prompt, which is not reproduced here; `CLAUDE.md` files and auto-memory are off.
  It is a new session: Fable does not see its r1 transcript, only what r1 left in `work/`.
- `continue K` (`scripts/launch.sh --run r2 continue K`): the whole of `kit/prompt_continue.md` as the next user
  message of the same session (`--resume`), only after a launch that ended without `MODEL COMPLETE`.
- In the author directory Fable can read `PROMPT.md` (a copy of `kit/prompt_r2.md`), `KIT.md` (a copy of
  `kit/KIT.md`), `LABELLING.md` (the hash above), `examples/` (1,200 dev screens and the reference screens),
  `resources/` (the codex: `CODEX.md`, `SOURCES.md` and `lists/`; the thesaurus: `out/*.tsv`, `README.md` and
  `ATTRIBUTION.md`; `r1/RESULTS.md`, the r1 results it may see), `HANDOVER.json` (file hashes only) and `work/`,
  which starts as its r1 `work/` without `scratch/`. `KIT.md` and `LABELLING.md` are not inlined in any message.""",
}


def preview_path(run: str) -> Path:
    return KIT / RUNS[run]["preview"]


def render_preview(run: str = "r1") -> str:
    name = RUNS[run]["prompt"]
    prompt = (KIT / name).read_text(encoding="utf-8").rstrip("\n")
    cont = (KIT / "prompt_continue.md").read_text(encoding="utf-8").rstrip("\n")
    extra = ""
    if run == "r2":
        extra = (f"| `kit/r1_results.md` (`resources/r1/RESULTS.md`) | `{sha256(R1_RESULTS)}` |\n"
                 f"| `codex/CODEX.md` (`resources/codex/CODEX.md`) | `{sha256(CODEX_DIR / 'CODEX.md')}` |\n")
    return PREVIEW_TEMPLATE.format(
        title_run="" if run == "r1" else f" (run {run})", run_flag="" if run == "r1" else f" --run {run}",
        prompt_name=name, prompt=sha256(KIT / name), cont=sha256(KIT / "prompt_continue.md"),
        kit=sha256(KIT / "KIT.md"), guide=sha256(GUIDE),
        kit_guide=hashlib.sha256(kit_guide().encode("utf-8")).hexdigest(), extra_rows=extra,
        delivery=DELIVERY[run],
        cost_note="" if run == "r1" else ", and in the codex's and thesaurus's ordinary uses of the word",
        n_prompt=len(prompt), prompt_text=prompt, n_cont=len(cont), cont_text=cont)


def cmd_preview(a) -> int:
    text = render_preview(a.run)
    out = preview_path(a.run)
    if a.check:
        if not out.is_file() or out.read_text(encoding="utf-8") != text:
            print(f"PROBLEM: kit/{out.name} is stale (uv run python kit/kit_build.py preview"
                  + (f" --run {a.run})" if a.run != "r1" else ")"))
            return 1
        print(f"kit/{out.name} is current")
        return 0
    out.write_text(text, encoding="utf-8")
    print(f"preview -> {out.relative_to(SLOP)} (kit/{RUNS[a.run]['prompt']} sha256 {sha256(KIT / RUNS[a.run]['prompt'])})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("lib")
    r.add_argument("--check", action="store_true")
    r = sub.add_parser("select")
    r.add_argument("--run", choices=sorted(RUNS), default="r1")
    r = sub.add_parser("tree")
    r.add_argument("author_dir")
    r.add_argument("--run", choices=sorted(RUNS), default="r1")
    r.add_argument("--allow-missing-docs", action="store_true", help="TEST hand-overs only: placeholder KIT/PROMPT")
    r = sub.add_parser("record")
    r.add_argument("author_dir")
    r.add_argument("record")
    r.add_argument("--test-root")
    r.add_argument("--run", choices=sorted(RUNS), default="r1")
    r.add_argument("--node", required=True, help="the node binary the wrappers run (recorded with its version)")
    r = sub.add_parser("verify")
    r.add_argument("author_dir")
    r.add_argument("record")
    r.add_argument("--pristine", action="store_true")
    r = sub.add_parser("preview")
    r.add_argument("--run", choices=sorted(RUNS), default="r1")
    r.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    return {"lib": cmd_lib, "select": cmd_select, "tree": cmd_tree, "record": cmd_record,
            "verify": cmd_verify, "preview": cmd_preview}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
