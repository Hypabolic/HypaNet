"""Shared constants, record schema, validation and file handling for the slop labeller.

Used by serve.py (the labelling server) and check.py (the validator). Standard library only.

Record kinds in a label file (`slop/labels/<set>.<labeller>.jsonl`), told apart by their fields:

- node:         {"screen", "loc", "class", "target", "equiv", "pattern", "aside", "partial", "ct", "conf",
                 "role_ok", ["role_fix"], "labeller", "guide_sha256", "time"}            (SPEC.md §5)
- aside_screen: {"screen", "aside_screen": [{"code", "nodes": [loc, ...]}], "labeller", "guide_sha256", "time"}
- missed:       {"screen", "missed": [{"line", "text", "class"}], "labeller", "guide_sha256", "time"}
- done:         {"screen", "done": true, "nodes": N, "labeller", "guide_sha256", "time"}  (labeller tool)

The adjudicated answer key (`<set>.final.jsonl`, labeller "final", written by slop/modellabel/adjudicate.py) adds
four fields to node records: `source` (agreed, adjudicated, contested or unresolved), `candidates` (the two model
labels), `binary` (the binary consensus, REDUNDANT, KEEP or null) and, after a second pass, `adjudication`. A
contested node has class null and conf "contested".

The file is append-only; the latest record per key wins. Keys: (screen, "node", loc) for node records and
(screen, kind) for the three screen-level kinds.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLOP_DIR = HERE.parent
DEFAULT_GUIDE = SLOP_DIR / "LABELLING.md"
DEFAULT_LABELS_DIR = SLOP_DIR / "labels"

SCREEN_SCHEMA = "slop.screen/1"

CLASSES = ["KEEP", "ECHO", "NARRATE", "DECOR", "UNSURE"]
REDUNDANT = {"ECHO", "NARRATE", "DECOR"}
CONF = ["clear", "judgement"]
FINAL_LABELLER = "final"
FINAL_SOURCES = ["agreed", "adjudicated", "contested", "unresolved"]
FINAL_FIELDS = {"source", "candidates", "adjudication", "binary"}

# LABELLING.md §9 (scored pattern codes, by class)
PATTERNS = {
    "ECHO": ["R1", "R3", "R4", "R5", "R8", "R9", "R14", "R17", "R18"],
    "NARRATE": ["R6", "R16"],
    "DECOR": ["R2", "R11"],
}
PATTERN_CLASS = {code: cls for cls, codes in PATTERNS.items() for code in codes}
# LABELLING.md §6. A KEEP record may name the K rule that decided it in `pattern` (optional; see spec gaps).
K_RULES = [f"K{i}" for i in range(1, 15)]
# LABELLING.md §9, `aside` rows (node level)
NODE_ASIDES = ["R7", "R8b", "R10", "R11g", "R12", "R13", "R15", "X-FILE", "UNRESOLVED"]
# Screen-level asides: R19 plus the other rule-layer codes, for devices that span several nodes
SCREEN_ASIDES = ["R19", "R10", "R12", "R11g", "R7", "R8b", "R13", "R15", "X-FILE"]

# SPEC.md §4
ROLES = (
    "h1 h2 h3 h4 h5 h6 eyebrow subtitle body list_item caption label helper placeholder input_value button link "
    "nav tab breadcrumb menu_item badge tooltip empty_title empty_body dialog_title dialog_body toast error stat "
    "cell aria alt comment code other"
).split()
FLAGS = (
    "caps title_case tracked small mono pill dot icon emoji_lead arrow_tail dynamic stateful conditional in_nav "
    "in_table in_form in_list in_dialog in_header in_footer in_card in_state in_control section_first section_last "
    "repeated sr_only truncated"
).split()
CLUSTER_KINDS = ["header", "actions", "control", "state", "card", "list", "body"]

# LABELLING.md §2: context only, never labelled
CONTEXT_ROLES = {"aria", "alt", "code"}
CONTEXT_FLAGS = {"sr_only"}

# LABELLING.md §5 precedence (1 = highest). A `label` counts as rank 1 when it sits in a form or control.
_RANK = {}
for _r in ("button", "link", "nav", "tab", "menu_item", "breadcrumb", "h1", "dialog_title", "empty_title"):
    _RANK[_r] = 1.0
for _n, _r in enumerate(("h2", "h3", "h4", "h5", "h6")):
    _RANK[_r] = 2.0 + _n / 10
for _r in ("body", "list_item", "dialog_body", "empty_body", "toast", "error", "stat", "cell", "input_value",
           "other"):
    _RANK[_r] = 3.0
for _r in ("subtitle", "helper", "caption"):
    _RANK[_r] = 4.0
for _r in ("eyebrow", "badge", "tooltip", "placeholder"):
    _RANK[_r] = 5.0


def role_rank(role: str, flags=()) -> float:
    """Precedence rank of a role (LABELLING.md §5); lower is kept. Unknown roles rank 3."""
    if role == "label":
        return 1.0 if ({"in_form", "in_control"} & set(flags or ())) else 3.0
    return _RANK.get(role, 3.0)


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def valid_name(s: str) -> bool:
    return bool(_NAME.match(s or ""))


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def loc_key(loc) -> str:
    return ":".join(str(int(x)) for x in loc)


def is_loc(v) -> bool:
    if not isinstance(v, list) or len(v) != 4:
        return False
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        return False
    l1, c1, l2, c2 = v
    return l1 >= 1 and c1 >= 1 and l2 >= l1 and c2 >= 1 and (l2 > l1 or c2 >= c1)


def label_path(labels_dir: Path, set_name: str, labeller: str) -> Path:
    return Path(labels_dir) / f"{set_name}.{labeller}.jsonl"


def meta_path(labels_path: Path) -> Path:
    p = Path(labels_path)
    return p.with_name(p.name[: -len(".jsonl")] + ".meta.json") if p.name.endswith(".jsonl") else p.with_suffix(
        ".meta.json")


def parse_label_filename(path: Path):
    """`<set>.<labeller>.jsonl` -> (set, labeller), or (None, None)."""
    name = Path(path).name
    if not name.endswith(".jsonl"):
        return None, None
    stem = name[: -len(".jsonl")]
    if "." not in stem:
        return None, None
    s, lab = stem.rsplit(".", 1)
    return s, lab


# ---------------------------------------------------------------- screens


def node_is_context(node: dict) -> bool:
    return node.get("role") in CONTEXT_ROLES or bool(CONTEXT_FLAGS & set(node.get("flags") or ()))


def labelable_nodes(screen: dict) -> list:
    return [n for n in screen.get("nodes", []) if not node_is_context(n)]


def validate_screen(screen) -> list:
    """Problems with a screen JSON (slop.screen/1). Empty list = fine."""
    errs = []
    if not isinstance(screen, dict):
        return ["screen is not an object"]
    if screen.get("schema") != SCREEN_SCHEMA:
        errs.append(f"schema is {screen.get('schema')!r}, expected {SCREEN_SCHEMA!r}")
    if not isinstance(screen.get("id"), str) or not screen.get("id"):
        errs.append("missing id")
    src = screen.get("source")
    if not isinstance(src, dict) or not isinstance(src.get("path"), str):
        errs.append("source.path missing")
    ks = set()
    for c in screen.get("clusters") or []:
        ks.add(c.get("k"))
        if c.get("kind") not in CLUSTER_KINDS:
            errs.append(f"cluster {c.get('k')}: unknown kind {c.get('kind')!r}")
    seen = {}
    nodes = screen.get("nodes")
    if not isinstance(nodes, list):
        return errs + ["nodes missing"]
    for n in nodes:
        i = n.get("i")
        if n.get("role") not in ROLES:
            errs.append(f"node {i}: unknown role {n.get('role')!r}")
        for f in n.get("flags") or []:
            if f not in FLAGS:
                errs.append(f"node {i}: unknown flag {f!r}")
        if n.get("k") not in ks:
            errs.append(f"node {i}: cluster {n.get('k')} not declared")
        if not is_loc(n.get("loc")):
            errs.append(f"node {i}: bad loc {n.get('loc')!r}")
            continue
        key = loc_key(n["loc"])
        if key in seen:
            errs.append(f"node {i}: loc {n['loc']} duplicates node {seen[key]} (labels key on loc)")
        seen[key] = i
    return errs


def load_screens(screens_dir: Path, order_file: Path | None = None):
    """Load screen JSON files from a directory (`*.json`, and `*.jsonl` with one screen per line).

    Returns (screens, problems) where screens is a list of dicts {"screen", "file", "sha256"} in labelling order:
    the order file's ids first (one id per line), otherwise sorted by file name.
    """
    screens_dir = Path(screens_dir)
    out, problems = [], []
    files = sorted(p for p in screens_dir.iterdir() if p.is_file() and p.suffix in (".json", ".jsonl")
                   and not p.name.startswith("."))
    for p in files:
        raw = p.read_bytes()
        if p.suffix == ".json":
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                problems.append(f"{p.name}: {e}")
                continue
            if not (isinstance(obj, dict) and obj.get("schema") == SCREEN_SCHEMA):
                continue  # not a screen (e.g. kinds.json)
            out.append({"screen": obj, "file": p.name, "sha256": hashlib.sha256(raw).hexdigest()})
        else:
            for ln, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    problems.append(f"{p.name}:{ln}: {e}")
                    continue
                if isinstance(obj, dict) and obj.get("schema") == SCREEN_SCHEMA:
                    out.append({"screen": obj, "file": f"{p.name}:{ln}",
                                "sha256": hashlib.sha256(line.encode("utf-8")).hexdigest()})
    ids = {}
    for s in out:
        for e in validate_screen(s["screen"]):
            problems.append(f"{s['file']}: {e}")
        sid = s["screen"].get("id")
        if sid in ids:
            problems.append(f"{s['file']}: duplicate screen id {sid} (also {ids[sid]})")
        ids[sid] = s["file"]
    if order_file:
        order = [ln.strip() for ln in Path(order_file).read_text().splitlines() if ln.strip()]
        rank = {sid: n for n, sid in enumerate(order)}
        missing = [s["screen"]["id"] for s in out if s["screen"]["id"] not in rank]
        if missing:
            problems.append(f"{len(missing)} screens are not in the order file; they go last")
        out.sort(key=lambda s: (rank.get(s["screen"]["id"], len(rank)), s["file"]))
    return out, problems


# ---------------------------------------------------------------- records


def record_kind(rec: dict) -> str | None:
    if not isinstance(rec, dict):
        return None
    if "loc" in rec:
        return "node"
    for k in ("aside_screen", "missed", "done"):
        if k in rec:
            return k
    return None


def record_key(rec: dict):
    kind = record_kind(rec)
    if kind == "node":
        return (rec.get("screen"), "node", loc_key(rec["loc"]) if is_loc(rec.get("loc")) else repr(rec.get("loc")))
    return (rec.get("screen"), kind)


NODE_FIELDS = ["screen", "loc", "class", "target", "equiv", "pattern", "aside", "partial", "ct", "conf", "role_ok",
               "labeller", "guide_sha256", "time"]
COMMON_FIELDS = ["screen", "labeller", "guide_sha256", "time"]


def _common(rec, errs, labeller=None):
    if not isinstance(rec.get("screen"), str) or not rec.get("screen"):
        errs.append("screen must be a non-empty string")
    if not isinstance(rec.get("labeller"), str) or not valid_name(rec.get("labeller", "")):
        errs.append("labeller must be a short id")
    elif labeller and rec["labeller"] != labeller:
        errs.append(f"labeller {rec['labeller']!r} does not match file labeller {labeller!r}")
    if not isinstance(rec.get("guide_sha256"), str) or not _HEX64.match(rec.get("guide_sha256", "")):
        errs.append("guide_sha256 must be 64 lowercase hex characters")
    if not isinstance(rec.get("time"), str) or not _TIME.match(rec.get("time", "")):
        errs.append("time must be ISO 8601 UTC ending in Z")


def validate_record(rec, labeller=None):
    """Returns (errors, warnings) for one record. Structural checks only; targets are checked against screens
    elsewhere."""
    errs, warns = [], []
    kind = record_kind(rec)
    if kind is None:
        return ["unknown record kind (no loc, aside_screen, missed or done)"], warns
    _common(rec, errs, labeller)
    final = FINAL_LABELLER in (labeller, rec.get("labeller"))
    contested = final and rec.get("conf") == "contested"
    if kind == "node":
        allowed = set(NODE_FIELDS) | {"role_fix"} | (FINAL_FIELDS if final else set())
        extra = set(rec) - allowed
        if extra:
            errs.append(f"unknown fields {sorted(extra)}")
        missing = [f for f in NODE_FIELDS if f not in rec]
        if missing:
            errs.append(f"missing fields {missing}")
            return errs, warns
        loc = rec["loc"]
        if not is_loc(loc):
            errs.append(f"bad loc {loc!r}")
        cls = rec["class"]
        aside = rec["aside"]
        if not isinstance(aside, list) or not all(isinstance(a, str) for a in aside):
            errs.append("aside must be a list of codes")
            aside = []
        else:
            bad = [a for a in aside if a not in NODE_ASIDES]
            if bad:
                errs.append(f"unknown aside codes {bad}")
            if len(set(aside)) != len(aside):
                errs.append("aside codes repeat")
        if final:
            if rec.get("source") not in FINAL_SOURCES:
                errs.append(f"final record needs source in {FINAL_SOURCES}")
            cands = rec.get("candidates")
            if not isinstance(cands, list) or len(cands) != 2 or not all(isinstance(x, dict) for x in cands):
                errs.append("final record needs candidates: the two labellers' labels")
            if rec.get("binary") not in ("REDUNDANT", "KEEP", None):
                errs.append("binary must be REDUNDANT, KEEP or null")
            elif cls is not None and rec.get("binary") is not None and rec["binary"] != (
                    "REDUNDANT" if cls in REDUNDANT else cls):
                errs.append(f"binary {rec['binary']} does not match class {cls}")
            if (rec.get("source") == "contested") != contested:
                errs.append("conf contested goes with source contested")
            if contested and (cls is not None or tgt_set(rec) or rec.get("pattern") is not None):
                errs.append("a contested record has class, target and pattern null (the candidates hold the labels)")
        if cls is None:
            if "UNRESOLVED" not in aside and not contested:
                errs.append("class is null without aside UNRESOLVED")
        elif cls not in CLASSES:
            errs.append(f"unknown class {cls!r}")
        tgt = rec["target"]
        if tgt is not None:
            if not is_loc(tgt):
                errs.append(f"bad target {tgt!r}")
            elif is_loc(loc) and tgt == loc:
                errs.append("target is the node itself")
            if cls not in ("ECHO", "NARRATE"):
                errs.append(f"target set on a {cls} node (only ECHO and NARRATE point)")
        elif cls == "ECHO":
            warns.append("ECHO with null target (allowed only when the information is spread over several nodes)")
        eq = rec["equiv"]
        if not isinstance(eq, list) or not all(is_loc(x) for x in eq):
            errs.append("equiv must be a list of locs")
        elif eq:
            if cls != "ECHO":
                errs.append(f"equiv set on a {cls} node (only ECHO stacks list equivalents)")
            if is_loc(loc) and loc in eq:
                errs.append("equiv lists the node itself")
            if tgt is not None and tgt in eq:
                errs.append("equiv lists the target (equiv holds the other flagged copies)")
            if len({loc_key(x) for x in eq}) != len(eq):
                errs.append("equiv repeats a loc")
        pat = rec["pattern"]
        if pat is not None and not isinstance(pat, str):
            errs.append("pattern must be a string or null")
        elif cls in PATTERNS:
            if pat is None:
                warns.append(f"{cls} with no pattern code")
            elif PATTERN_CLASS.get(pat) != cls:
                errs.append(f"pattern {pat} is not a {cls} pattern (allowed: {', '.join(PATTERNS[cls])})")
        elif cls == "KEEP":
            if pat is not None and pat not in K_RULES:
                errs.append(f"KEEP pattern must be null or a K rule, got {pat!r}")
        elif pat is not None:
            errs.append(f"{cls} must have pattern null")
        for b in ("partial", "ct", "role_ok"):
            if not isinstance(rec[b], bool):
                errs.append(f"{b} must be true or false")
        if rec["partial"] is True and cls != "KEEP":
            errs.append("partial: true is only valid on KEEP (LABELLING.md §4 step 5)")
        if rec["conf"] not in CONF and not contested:
            errs.append(f"conf must be one of {CONF}")
        if rec["role_ok"] is False:
            if rec.get("role_fix") not in ROLES:
                errs.append(f"role_ok false needs role_fix from the role list, got {rec.get('role_fix')!r}")
        elif "role_fix" in rec and rec["role_fix"] is not None:
            errs.append("role_fix given while role_ok is true")
        if "R11g" in aside and cls not in (None, "KEEP", "UNSURE"):
            warns.append("aside R11g marks a glyph on a KEEP node, but the node is " + str(cls))
    elif kind == "aside_screen":
        extra = set(rec) - set(COMMON_FIELDS) - {"aside_screen"}
        if extra:
            errs.append(f"unknown fields {sorted(extra)}")
        items = rec["aside_screen"]
        if not isinstance(items, list):
            errs.append("aside_screen must be a list")
        else:
            for n, it in enumerate(items):
                if not isinstance(it, dict) or set(it) != {"code", "nodes"}:
                    errs.append(f"aside_screen[{n}] must be {{code, nodes}}")
                    continue
                if it["code"] not in SCREEN_ASIDES:
                    errs.append(f"aside_screen[{n}]: unknown code {it['code']!r}")
                if not isinstance(it["nodes"], list) or not all(is_loc(x) for x in it["nodes"]):
                    errs.append(f"aside_screen[{n}]: nodes must be a list of locs")
                elif len(it["nodes"]) < 2:
                    warns.append(f"aside_screen[{n}] {it['code']}: fewer than two nodes")
    elif kind == "missed":
        extra = set(rec) - set(COMMON_FIELDS) - {"missed"}
        if extra:
            errs.append(f"unknown fields {sorted(extra)}")
        items = rec["missed"]
        if not isinstance(items, list):
            errs.append("missed must be a list")
        else:
            for n, it in enumerate(items):
                if not isinstance(it, dict) or set(it) != {"line", "text", "class"}:
                    errs.append(f"missed[{n}] must be {{line, text, class}}")
                    continue
                if not isinstance(it["line"], int) or isinstance(it["line"], bool) or it["line"] < 1:
                    errs.append(f"missed[{n}]: line must be a positive integer")
                if not isinstance(it["text"], str) or not it["text"].strip():
                    errs.append(f"missed[{n}]: text must be non-empty")
                if it["class"] not in CLASSES:
                    errs.append(f"missed[{n}]: unknown class {it['class']!r}")
    elif kind == "done":
        extra = set(rec) - set(COMMON_FIELDS) - {"done", "nodes"}
        if extra:
            errs.append(f"unknown fields {sorted(extra)}")
        if rec.get("done") is not True:
            errs.append("done must be true")
        if not isinstance(rec.get("nodes"), int) or isinstance(rec.get("nodes"), bool) or rec["nodes"] < 0:
            errs.append("done record needs nodes (count of labelled nodes)")
    return errs, warns


def tgt_set(rec) -> bool:
    return rec.get("target") is not None


def node_complete(rec) -> bool:
    return rec is not None and (rec.get("class") is not None or "UNRESOLVED" in (rec.get("aside") or [])
                                or rec.get("conf") == "contested")


@dataclass
class LabelFile:
    """Latest-wins view of an append-only label file."""

    records: list = field(default_factory=list)  # every parsed record, in file order
    latest: dict = field(default_factory=dict)  # key -> record
    bad_lines: list = field(default_factory=list)  # (line number, error)
    screens: dict = field(default_factory=dict)  # screen id -> {key[1:]: record}

    @classmethod
    def load(cls, path: Path) -> "LabelFile":
        lf = cls()
        path = Path(path)
        if not path.exists():
            return lf
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    lf.bad_lines.append((ln, f"not JSON: {e}"))
                    continue
                if record_kind(rec) is None:
                    lf.bad_lines.append((ln, "unknown record kind"))
                    continue
                lf.add(rec, ln)
        return lf

    def add(self, rec: dict, line_no: int | None = None):
        self.records.append((line_no, rec))
        key = record_key(rec)
        self.latest[key] = rec
        self.screens.setdefault(key[0], {})[key[1:]] = rec

    def for_screen(self, screen_id: str) -> dict:
        nodes = {}
        out = {"nodes": nodes, "aside_screen": None, "missed": None, "done": None}
        for key, rec in self.screens.get(screen_id, {}).items():
            if key[0] == "node":
                nodes[key[1]] = rec
            else:
                out[key[0]] = rec
        return out


def append_record(path: Path, rec: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False, separators=(", ", ": ")) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------- guide


def guide_sections(text: str) -> dict:
    """Split LABELLING.md into {"3": markdown, ...} by '## N.' headings."""
    out, cur, buf = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^## (\d+)\.\s", line)
        if m:
            if cur:
                out[cur] = "\n".join(buf).strip() + "\n"
            cur, buf = m.group(1), [line]
        elif cur:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf).strip() + "\n"
    return out


def parse_guide(text: str) -> dict:
    """Pattern descriptions and K rules from the guide, for the quick reference and pickers."""
    patterns, asides, screen_asides, krules = {}, {}, {}, {}
    for line in text.splitlines():
        m = re.match(r"^\|\s*(screen aside |aside )?(R\d+[a-z]?|X-FILE|UNRESOLVED)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$",
                     line)
        if m:
            kind, code, desc, cls = m.groups()
            if kind is None and cls in PATTERNS:
                patterns[code] = {"desc": desc, "class": cls}
            elif kind and kind.startswith("screen"):
                screen_asides[code] = {"desc": desc, "use": cls}
            elif kind:
                asides[code] = {"desc": desc, "use": cls}
    cur = None
    for line in text.splitlines():
        m = re.match(r"^- \*\*(K\d+)\*\*\s*(.*)$", line)
        if m:
            cur = m.group(1)
            krules[cur] = m.group(2)
        elif cur and line.startswith("  ") and line.strip():
            krules[cur] += " " + line.strip()
        else:
            cur = None
    problems = []
    for cls, codes in PATTERNS.items():
        for c in codes:
            if c not in patterns:
                problems.append(f"pattern {c} ({cls}) not found in the guide's §9 table")
            elif patterns[c]["class"] != cls:
                problems.append(f"guide gives {c} class {patterns[c]['class']}, the tool expects {cls}")
    for c in patterns:
        if c not in PATTERN_CLASS:
            problems.append(f"guide pattern {c} unknown to the tool")
    for k in K_RULES:
        if k not in krules:
            problems.append(f"K rule {k} not found in the guide's §6")
    for k in krules:
        if k not in K_RULES:
            problems.append(f"guide K rule {k} unknown to the tool")
    return {"patterns": patterns, "asides": asides, "screen_asides": screen_asides, "krules": krules,
            "problems": problems}
