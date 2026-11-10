"""Shared paths, set loading, literal-word detection and a small JSON Schema checker for the model labellers.

Standard library only (like `slop/labeller/`). The label schema itself lives in `slop/labeller/labelkit.py`; this
module imports it rather than restating it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLOP_DIR = HERE.parent
sys.path.insert(0, str(SLOP_DIR / "labeller"))
import labelkit as lk  # noqa: E402

DATASETS_DIR = SLOP_DIR / "datasets"
LABELS_DIR = SLOP_DIR / "labels"
GUIDE = SLOP_DIR / "LABELLING.md"
SCHEMA_PATH = HERE / "schema.json"
ADJ_SCHEMA_PATH = HERE / "adjudicate_schema.json"

# Who labels (SPEC.md §5, "Answer key"). Short names on the command line -> labeller ids in the label files.
LABELLERS = {"opus": "opus55", "codex": "codex61"}
LABELLER_NAMES = {v: k for k, v in LABELLERS.items()}
FINAL_LABELLER = "final"

# Sets the harness may label. `sealed_real` is built only after the extractor freeze (SPEC.md §12.1); `label`
# refuses it unless its directory exists.
KNOWN_SETS = ["reference_real", "calibration", "sealed_real"]

VAR_MARK = ""
ALT_MARK = ""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


# ---------------------------------------------------------------- literal words (port of rules/src/text.js tokens())
# A differential test (tests/test_modellabel.py) runs rules/src/text.js under node on the same strings.

def _class_of(prefix: str) -> str:
    """A regex character class body of every code point whose general category starts with `prefix` (L or M)."""
    out, start, prev = [], None, None
    for cp in range(sys.maxunicode + 1):
        if unicodedata.category(chr(cp)).startswith(prefix):
            if start is None:
                start = cp
            prev = cp
        elif start is not None:
            out.append(re.escape(chr(start)) + ("-" + re.escape(chr(prev)) if prev > start else ""))
            start = None
    if start is not None:
        out.append(re.escape(chr(start)) + ("-" + re.escape(chr(prev)) if prev > start else ""))
    return "".join(out)


# JS: /(\uE000)|(\uE001)|([\p{L}\p{M}]+(?:['’][\p{L}\p{M}]+)*)|(\p{Nd}+(?:[.,]\p{Nd}+)*)|(\s+)|(.)/gsu (Python's \d is \p{Nd})
_LM = _class_of("L") + _class_of("M")
_TOKEN = re.compile(
    "(\ue000)|(\ue001)|([" + _LM + "]+(?:['’][" + _LM + "]+)*)|(\\d+(?:[.,]\\d+)*)|(\\s+)|(.)",
    re.S,
)
_FALLBACK_IDENT_WORDS = 4


def tokens(text: str, vw=None) -> list:
    """Tokens of a node's text: dicts {k: var|alt|word|number|space|other, s: surface, ident: bool}.

    Words (and numbers) that follow a VAR marker are the interpolated identifier's words (SPEC.md §4), not copy. `vw`
    gives the exact count per marker; without a usable `vw`, up to four words after a marker are taken (the same
    fallback as the rule layer)."""
    out = []
    for m in _TOKEN.finditer(unicodedata.normalize("NFKC", text or "")):
        if m.group(1):
            out.append({"k": "var", "s": m.group(1)})
        elif m.group(2):
            out.append({"k": "alt", "s": m.group(2)})
        elif m.group(3):
            out.append({"k": "word", "s": m.group(3)})
        elif m.group(4):
            out.append({"k": "number", "s": m.group(4)})
        elif m.group(5):
            out.append({"k": "space", "s": m.group(5)})
        else:
            out.append({"k": "other", "s": m.group(6)})
    nvars = sum(1 for t in out if t["k"] == "var")
    exact = isinstance(vw, list) and len(vw) == nvars and all(isinstance(c, int) and c >= 0 for c in vw)
    v = 0
    for x, t in enumerate(out):
        if t["k"] != "var":
            continue
        cap = vw[v] if exact else _FALLBACK_IDENT_WORDS
        v += 1
        kinds = ("word", "number") if exact else ("word",)
        n = 0
        y = x + 1
        while y < len(out) and n < cap:
            if out[y]["k"] == "space":
                y += 1
                continue
            if out[y]["k"] not in kinds:
                break
            out[y]["ident"] = True
            n += 1
            y += 1
    return out


def literal_words(text: str, vw=None) -> list:
    return [t["s"] for t in tokens(text, vw) if t["k"] in ("word", "number") and not t.get("ident")]


def has_literal_words(node: dict) -> bool:
    """False for a node whose text is only VAR markers and identifier words (LABELLING.md §2: such nodes get
    `class: null` with `aside: ["UNRESOLVED"]`). A node with no VAR marker always counts as having literal copy, so
    a glyph-only node ("●") is labelled like any other."""
    text = node.get("text") or ""
    if VAR_MARK not in text:
        return True
    return bool(literal_words(text, node.get("vw")))


def display_text(node: dict) -> str:
    """The node's text for a prompt: each interpolation as ⟨identifier words⟩, conditional branches split by ' ‖ ', and
    a literal ⟨, ⟩ or ‖ doubled."""
    toks = tokens(node.get("text") or "", node.get("vw"))
    out, x = [], 0
    while x < len(toks):
        t = toks[x]
        if t["k"] == "var":
            words, y = [], x + 1
            while y < len(toks) and (toks[y].get("ident") or (toks[y]["k"] == "space" and any(
                    tt.get("ident") for tt in toks[y + 1:y + 2]))):
                if toks[y].get("ident"):
                    words.append(toks[y]["s"])
                y += 1
            out.append("⟨" + " ".join(words) + "⟩")
            x = y
            continue
        out.append(" ‖ " if t["k"] == "alt" else _escape_marks(t["s"]))
        x += 1
    return re.sub(r"[ \t]{2,}", " ", "".join(out)).strip()


def _escape_marks(s: str) -> str:
    """A literal ⟨, ⟩ or ‖ in copy is doubled, so it never reads as an interpolation or a branch marker."""
    return s.replace("⟨", "⟨⟨").replace("⟩", "⟩⟩").replace("‖", "‖‖")


# ---------------------------------------------------------------- sets


@dataclass
class Screen:
    screen: dict
    file: str
    sha256: str
    source_path: Path | None
    source_text: str | None
    kind: str | None
    _windows: dict | None = field(default=None, repr=False, compare=False)

    @property
    def id(self) -> str:
        return self.screen["id"]

    @property
    def repo(self) -> str:
        return (self.screen.get("source") or {}).get("repo") or ""

    def labelable(self) -> list:
        return lk.labelable_nodes(self.screen)

    def by_i(self) -> dict:
        return {n["i"]: n for n in self.screen["nodes"]}

    def by_loc(self) -> dict:
        return {lk.loc_key(n["loc"]): n for n in self.screen["nodes"]}

    def windows(self) -> dict:
        """{node i: window index} (cached)."""
        if self._windows is None:
            self._windows = node_windows(self.screen)
        return self._windows


@dataclass
class LabelSet:
    name: str
    root: Path
    screens: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    manifest: dict = field(default_factory=dict)

    @property
    def guide_example_ids(self) -> set:
        """Screens whose answer the guide itself works through (the pause-menu fixture in reference_real, LABELLING.md
        §8): labelled like any other, but left out of the agreement statistics."""
        fx = self.manifest.get("fixture") if isinstance(self.manifest, dict) else None
        return {fx["id"]} if isinstance(fx, dict) and fx.get("id") else set()

    @property
    def screens_dir(self) -> Path:
        return self.root / "screens"

    @property
    def order_file(self) -> Path | None:
        p = self.root / "order.txt"
        return p if p.exists() else None

    def get(self, sid: str) -> Screen:
        for s in self.screens:
            if s.id == sid:
                return s
        raise KeyError(sid)


class SetMissing(Exception):
    pass


def set_root(name: str, datasets_dir: Path = DATASETS_DIR) -> Path:
    return Path(datasets_dir) / name


def load_kinds(datasets_dir: Path, set_dir: Path) -> dict:
    kinds = {}
    for p in (Path(datasets_dir) / "kinds.json", set_dir / "kinds.json", set_dir / "screens" / "kinds.json"):
        if p.exists():
            kinds.update(json.loads(p.read_text(encoding="utf-8")))
    return kinds


_FIXTURE_NOTE = re.compile(r"\s*\(fixture note:[^)]*\)", re.I)


def neutral_kind(kind):
    """The product kind as the labellers see it: a fixture note ("AI-built pause menu, the motivating example") is a
    hint about the answer, not about the product, so it is cut ("first-person game", as LABELLING.md §8 says)."""
    if not isinstance(kind, str):
        return None
    return _FIXTURE_NOTE.sub("", kind).strip() or None


def kind_for(kinds: dict, screen: dict):
    src = screen.get("source") or {}
    for key in (screen.get("id"), f"{src.get('repo')}/{src.get('path')}", src.get("repo")):
        if key in kinds:
            return neutral_kind(kinds[key])
    return None


def source_is_screen_json(text) -> bool:
    """True when a screen's "source file" is itself a slop.screen/1 document (build_sealed.py --sibling copies the
    sibling fixture's screen JSON as its source): the labellers must see real source, so `label` refuses."""
    if not isinstance(text, str) or not text.lstrip().startswith("{"):
        return False
    try:
        return json.loads(text).get("schema") == "slop.screen/1"
    except (ValueError, AttributeError):
        return False


def node_windows(screen: dict) -> dict:
    """{node i: window index} from the frozen tokeniser's split (SPEC.md §4 Windows). A target must be in the same
    window as its node (LABELLING.md §2a)."""
    if str(SLOP_DIR) not in sys.path:
        sys.path.insert(0, str(SLOP_DIR))
    from ui import tokenizer as tk

    try:
        enc = tk.encode_screen(screen)
    except tk.TokenizerError:
        return {n.get("i"): 0 for n in screen.get("nodes") or []}  # not encodable: one window (the real sets encode)
    return {enc.nodes[x].i: w for w, win in enumerate(enc.windows) for x in win.nodes}


MAX_SOURCE_BYTES = 400_000


def find_source(set_dir: Path, screen: dict):
    src = screen.get("source") or {}
    roots = [set_dir / "sources", set_dir / "src"]
    for root in roots:
        cands = []
        if src.get("repo"):
            cands.append(root / src["repo"] / (src.get("path") or ""))
        cands.append(root / (src.get("path") or ""))
        for c in cands:
            c = c.resolve()
            if root.resolve() not in c.parents:
                continue
            if c.is_file():
                return c
    return None


def load_set(name: str, datasets_dir: Path = DATASETS_DIR) -> LabelSet:
    """A set directory holds `screens/` (slop.screen/1 JSON), `sources/<repo>/<path>`, `order.txt` and a manifest."""
    root = set_root(name, datasets_dir)
    if not (root / "screens").is_dir():
        raise SetMissing(f"set {name!r} has no screens directory at {root / 'screens'}")
    man = read_json(root / "manifest.json")
    ls = LabelSet(name=name, root=root, manifest=man if isinstance(man, dict) else {})
    loaded, problems = lk.load_screens(root / "screens", ls.order_file)
    ls.problems = problems
    kinds = load_kinds(datasets_dir, root)
    for s in loaded:
        sp = find_source(root, s["screen"])
        text = None
        if sp is not None:
            raw = sp.read_bytes()
            text = raw[:MAX_SOURCE_BYTES].decode("utf-8", errors="replace")
        ls.screens.append(Screen(screen=s["screen"], file=s["file"], sha256=s["sha256"], source_path=sp,
                                 source_text=text, kind=kind_for(kinds, s["screen"])))
    return ls


# ---------------------------------------------------------------- JSON Schema subset


def check_schema(value, schema: dict, path: str = "$") -> list:
    """Validate against the subset of JSON Schema used by schema.json (type, enum, properties, required,
    additionalProperties false, items, anyOf, minimum). Returns a list of error strings."""
    errs = []
    if "anyOf" in schema:
        subs = [check_schema(value, s, path) for s in schema["anyOf"]]
        if all(subs):
            best = min(subs, key=len)
            errs.append(f"{path}: matches none of the allowed forms ({'; '.join(best[:2])})")
        return errs
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_is_type(value, tt) for tt in types):
            return [f"{path}: expected {'/'.join(types)}, got {_type_name(value)}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if "minimum" in schema and _is_type(value, "number") and value < schema["minimum"]:
        errs.append(f"{path}: {value} is below {schema['minimum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}: missing field {k!r}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(props))
            if extra:
                errs.append(f"{path}: unknown fields {extra}")
        for k, v in value.items():
            if k in props:
                errs += check_schema(v, props[k], f"{path}.{k}")
    if isinstance(value, list) and "items" in schema:
        for n, v in enumerate(value):
            errs += check_schema(v, schema["items"], f"{path}[{n}]")
    return errs


def _is_type(v, t):
    if t == "null":
        return v is None
    if t == "boolean":
        return isinstance(v, bool)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "string":
        return isinstance(v, str)
    if t == "array":
        return isinstance(v, list)
    if t == "object":
        return isinstance(v, dict)
    return False


def _type_name(v):
    for t in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _is_type(v, t):
            return t
    return type(v).__name__


def load_schema(path: Path = SCHEMA_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_json(text: str):
    """The single JSON object in a model reply. Accepts a bare object or one fenced block; returns (obj, error)."""
    if text is None:
        return None, "empty reply"
    s = text.strip()
    m = re.match(r"^```(?:json)?\s*\n(.*)\n```\s*$", s, re.S)
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        a, b = s.find("{"), s.rfind("}")
        if a >= 0 and b > a:
            try:
                obj = json.loads(s[a:b + 1])
            except json.JSONDecodeError:
                return None, f"reply is not valid JSON: {e}"
        else:
            return None, f"reply is not valid JSON: {e}"
    if not isinstance(obj, dict):
        return None, "reply is JSON but not an object"
    return obj, None


def write_json(path: Path, obj) -> None:
    """Atomic write through a unique temporary file (two writers never share a .tmp name)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
