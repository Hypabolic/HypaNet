"""v0.4 freeze registry (plan v0.4 G1; experiments/v0_4/PREREGISTRATION.md §12-13): ``experiments/registry_v0_4.yaml``.

Modelled on :mod:`agent_loop_lab.v03.registry`, which stays frozen. Differences:

* **The file list is computed, and its parts are required.** Every path in :data:`REQUIRED_FILES`
  must exist and every component of :data:`REQUIRED_COMPONENTS` (the recipes, the harness, the
  author kit; named by path patterns where their final file names belong to the components) must
  match at least one file, so a component that is not merged yet makes the freeze refuse instead of
  being frozen later. :data:`FROZEN_GLOBS` then adds everything under the v0.4 trees (all of
  ``envs/qwen`` except its venv and caches, the author kit, ``scripts/v04_*``, the ``v04`` package
  at any depth). Each glob's base is walked without following links: a symlink anywhere a glob
  reaches, file or directory, is refused. ``check`` names any file added later under a glob.
* **The root-env import closure.** Every frozen root-env module (``src/agent_loop_lab/**``,
  ``scripts/*.py``) is parsed with ``ast`` and every ``agent_loop_lab`` module it imports (absolute
  or relative, at module level or inside a function), and every package ``__init__`` on the way,
  is frozen too, to a fixed point (``synth/families``, which loads its modules by name, adds them
  all). An import that resolves to no file is refused. This is what
  freezes ``eval/m4.py``, ``eval/metrics.py``, ``v03/scoring.py``, ``v03/renderer.py`` and the rest
  of the scoring path without a hand-kept list.
* **Interface hash.** ``envs/qwen/hypaqwen/interface.py`` is standard-library only, so it is executed
  here from its source (no import of ``hypaqwen``, no bytecode written) and its ``spec_sha256()`` is
  recorded. A freeze refuses any value other than :data:`INTERFACE_SPEC_SHA256`, the one G1
  pre-registers.
* **Model pins.** ``MODEL_ID``, ``MODEL_REVISION`` (``hypaqwen/__init__.py``) and ``FILE_SHA256``
  (``hypaqwen/model.py``, which imports torch 2.14) are read as literals from the source by ``ast``.
* **Packet hash.** ``scripts/v04_build_packet.py --out DIR`` is run in a fresh interpreter into a
  temporary directory; it prints ``packet sha256: <hex>``, which must equal the sha256 of the
  ``PACKET_MANIFEST.json`` it wrote. A missing builder is a :class:`RegistryError`.
* **Additive revisions** (``revise``; PREREGISTRATION.md §13). A revision loads the registry,
  refuses unless it is intact, refuses any change to an existing file except the paths it is given
  (recorded), any removed file and any change of interface, model or packet, and appends
  ``{date, reason, previous_freeze_sha256, globs_before, added, changed}`` to ``revisions``.
  ``check`` undoes the revisions one by one and verifies that each reconstructed record hashes to
  the next one's ``previous_freeze_sha256``, so the G1 record and every step are verifiable.
* ``check`` also runs the v0.3 registry check (:func:`agent_loop_lab.v03.registry.check_registry`):
  the v0.3 artefacts re-scored here, and the v0.3 renderer the packet carries, stay verified.

Record keys: ``spec_version`` ("0.4"), ``globs``, ``files`` {path: sha256},
``interface_spec_sha256``, ``model`` {``id``, ``revision``, ``file_sha256``}, ``packet_sha256``,
``revisions`` (``[]`` at G1) and ``freeze_sha256`` (sha256 of the canonical JSON of the record
without that key).

This module only reads bytes for hashing (and parses root-env Python source); it never decodes a
dataset file and walks no directory under ``datasets/``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import types
from datetime import date as _date
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

__all__ = [
    "REGISTRY_RELPATH",
    "REQUIRED_FILES",
    "REQUIRED_COMPONENTS",
    "FROZEN_GLOBS",
    "INTERFACE_SPEC_SHA256",
    "RegistryError",
    "frozen_files",
    "import_closure",
    "interface_spec_sha256",
    "model_pins",
    "packet_hash",
    "compute_freeze",
    "render",
    "write_registry",
    "revise_registry",
    "check_registry",
]

SPEC_VERSION = "0.4"
REGISTRY_RELPATH = "experiments/registry_v0_4.yaml"
QWEN = "envs/qwen"
PKG = f"{QWEN}/hypaqwen"
KIT = "experiments/v0_4/author_kit"
SRC = "src/agent_loop_lab"
INTERFACE_FILE = f"{PKG}/interface.py"
MODEL_FILE = f"{PKG}/model.py"
PKG_INIT = f"{PKG}/__init__.py"
PACKET_BUILDER = "scripts/v04_build_packet.py"
PACKET_MANIFEST = "PACKET_MANIFEST.json"
# experiments/v0_4/prompt_choice/README.md Revision 1; interface.FROZEN (w1-state-full).
INTERFACE_SPEC_SHA256 = "5922eb4500122921ac7d07e3d35599b95cb10b462c4f50b89e5d6bb989437879"

REQUIRED_FILES: tuple[str, ...] = (
    # Plan, spec, pre-registration, contract.
    "agent_doom_loop_detector_spec_v0_4.md",
    "agent_doom_loop_detector_plan_v0_4.md",
    "experiments/v0_4/PREREGISTRATION.md",
    "docs/V04_WSP_KERNEL.md",
    # The Qwen environment: interface, readout, model pins, data, the kernel's ./py launcher.
    PKG_INIT,
    INTERFACE_FILE,
    f"{PKG}/readout.py",
    MODEL_FILE,
    f"{PKG}/data.py",
    f"{PKG}/run_readout.py",
    f"{PKG}/wsp/author/run_py.py",
    f"{QWEN}/scripts/make_author_venv.sh",
    f"{QWEN}/pyproject.toml",
    f"{QWEN}/uv.lock",
    # The root (frozen v0.3) environment and the v0.4 code on its side.
    "pyproject.toml",
    "uv.lock",
    "oracle_config.v0_2.yaml",  # the oracle labels every split was built with
    f"{SRC}/v04/scoring.py",
    f"{SRC}/v04/export.py",
    f"{SRC}/schema.py",
    f"{SRC}/eval/m4.py",
    f"{SRC}/eval/metrics.py",
    f"{SRC}/eval/collisions.py",
    f"{SRC}/v03/scoring.py",  # disclosure_selection (WSP-B, probe-200w, LoRA-200w)
    f"{SRC}/v03/sealed.py",
    f"{SRC}/v03/renderer.py",
    f"{SRC}/baselines/phase_c_v03.py",  # budget_subset, es_session_ids (probe and LoRA subsets)
    f"{SRC}/baselines/phase_c.py",  # macro_f1
    f"{SRC}/synth/counterfactual.py",  # INTENDED (the Q5 pair kinds)
    # Harness scripts whose names the contract and this file fix.
    "scripts/v04_export_tasks.py",
    "scripts/v04_score_validation.py",
    PACKET_BUILDER,
    "scripts/v04_prepare_author_dir.sh",
    "scripts/v04_launch.sh",
    "scripts/v04_cost.py",
    # Author kit files the session harness names.
    f"{KIT}/prompt_wsp_a.md",
    f"{KIT}/prompt_continue.md",
    f"{KIT}/prompt_resubmit.md",
    f"{KIT}/settings_wsp_a.json",
    f"{KIT}/settings_wsp_b.json",
    f"{KIT}/py",
    f"{KIT}/fwd",
    # Committed data manifests and reports every number depends on (hashed as bytes, never decoded).
    "datasets/v0_2/splits.json",
    "datasets/v0_2/MANIFEST.json",
    "datasets/v0_3/MANIFEST_v0_3.json",
    "datasets/v0_4/EXPORT_MANIFEST.json",
    "datasets/v0_4/BUDGETS.json",  # probe/LoRA budgets, ES set, disclosed windows, WSP-F holdout (§4, §5, §7)
    "datasets/v0_4/fresh_test/MANIFEST_fresh_test.json",
    "experiments/v0_4/fresh_test/REGISTRATION.json",
    "experiments/v0_4/prompt_choice/README.md",
    "artifacts/reports/v04_prompt_choice.json",
    "artifacts/reports/v04_prompt_choice_revision1.json",
    "artifacts/reports/v03_sealed.json",  # the cited v0.3 numbers and rows_sha256
    "experiments/v0_3/phase_c/README.md",
)
# Components whose file names are the components' own: each must match at least one file. A
# component with several patterns is present when any pattern matches.
REQUIRED_COMPONENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("probe recipe (PREREGISTRATION §4)", (f"{PKG}/**/probe*.py",)),
    ("LoRA recipe (PREREGISTRATION §5)", (f"{PKG}/**/lora*.py",)),
    ("cloud GPU runner for the probe and LoRA (§3.1)",
     ("scripts/v04_*cloud*", f"{QWEN}/scripts/*cloud*", f"{PKG}/**/cloud*.py")),
    ("WSP-B disclosure builder (§7.1)", ("scripts/v04_*disclos*.py",)),
    ("session audit (§10)", ("scripts/v04_audit*.py",)),
    ("WSP-B prompt (§6)", (f"{KIT}/**/*wsp_b*.md",)),
    ("author-kit command hook (kernel contract §7)", (f"{KIT}/**/*hook*",)),
)
# Every match is frozen ("**" spans any number of directories, "*" stays within one).
FROZEN_GLOBS: tuple[str, ...] = (
    f"{QWEN}/**/*",  # hypaqwen (interface .. wsp, recipes, any data file), scripts, tests, lock
    f"{KIT}/**/*",
    "scripts/v04_*",
    f"{SRC}/v04/**/*",
)
_IGNORED_DIRS = frozenset({"__pycache__", ".pytest_cache", ".venv", ".mypy_cache", ".ruff_cache", ".git"})
_IGNORED_FILES = frozenset({".DS_Store"})
_PACKET_RE = re.compile(r"^packet sha256: ([0-9a-f]{64})\s*$", re.MULTILINE)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ROOT_PACKAGE = "agent_loop_lab"
# Imports the closure cannot follow statically: a module that loads others by name, and the frozen
# v0.3 guard's fallback for its standalone copy (tools/), which never runs from src/.
_DYNAMIC_IMPORTS: dict[str, str] = {f"{SRC}/synth/families/__init__.py": f"{SRC}/synth/families/*.py"}
_UNRESOLVED_OK = frozenset({(f"{SRC}/v03/guards.py", f"{ROOT_PACKAGE}.v03.guards_v02")})
KEYS = frozenset({"spec_version", "globs", "files", "interface_spec_sha256", "model", "packet_sha256",
                  "revisions", "freeze_sha256"})
REVISION_KEYS = frozenset({"date", "reason", "previous_freeze_sha256", "globs_before", "added", "changed"})
PACKET_TIMEOUT_S = 1800
_HEADER = (
    "# Freeze registry (plan v0.4 G1; experiments/v0_4/PREREGISTRATION.md). Generated by scripts/v04_freeze.py.\n"
    "# Do not edit by hand. A change is an additive revision (scripts/v04_freeze.py --revise; PREREGISTRATION.md §13).\n"
)


class RegistryError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def freeze_digest(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "freeze_sha256"}
    return _sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


# ---------------------------------------------------------------------------
# Globs
# ---------------------------------------------------------------------------


def _glob_regex(pattern: str) -> re.Pattern[str]:
    segs = pattern.split("/")
    out = []
    for i, seg in enumerate(segs):
        if seg == "**":
            out.append("(?:[^/]+/)*")
            continue
        piece = "".join("[^/]*" if c == "*" else "[^/]" if c == "?" else re.escape(c) for c in seg)
        out.append(piece + ("/" if i < len(segs) - 1 else ""))
    return re.compile("".join(out))


def _glob_base(pattern: str) -> str:
    base = []
    for seg in pattern.split("/")[:-1]:
        if any(c in seg for c in "*?["):
            break
        base.append(seg)
    return "/".join(base)


def _ignored(rel: str) -> bool:
    parts = rel.split("/")
    return (any(p in _IGNORED_DIRS or p.endswith(".egg-info") for p in parts[:-1]) or parts[-1] in _IGNORED_FILES
            or rel.endswith(".pyc"))


def _expand(base: Path, pattern: str) -> tuple[list[str], list[str]]:
    """``(matches, symlinks)`` of one glob: its base walked without following links. For a ``**``
    glob every symlink under the base is reported; otherwise a symlink that matches."""
    rx, recursive = _glob_regex(pattern), "**" in pattern
    top = _glob_base(pattern)
    start = base / top if top else base
    matches: list[str] = []
    links: list[str] = []
    if start.is_symlink():
        return [], [top]
    if not start.is_dir():
        return [], []
    for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
        here = Path(dirpath).relative_to(base).as_posix()
        here = "" if here == "." else here
        keep = []
        for d in sorted(dirnames):
            rel = f"{here}/{d}" if here else d
            if d in _IGNORED_DIRS or d.endswith(".egg-info"):
                continue
            if (Path(dirpath) / d).is_symlink():
                if recursive or rx.fullmatch(rel):
                    links.append(rel)
                continue
            keep.append(d)
        dirnames[:] = keep
        for f in sorted(filenames):
            rel = f"{here}/{f}" if here else f
            if _ignored(rel) or not rx.fullmatch(rel):
                continue
            if (Path(dirpath) / f).is_symlink():
                links.append(rel)
            else:
                matches.append(rel)
    return sorted(matches), sorted(links)


# ---------------------------------------------------------------------------
# Root-env import closure
# ---------------------------------------------------------------------------


def _module_file(base: Path, module: str) -> str | None:
    parts = module.split(".")
    if parts[0] != ROOT_PACKAGE:
        return None
    stem = "src/" + "/".join(parts)
    for rel in (f"{stem}.py", f"{stem}/__init__.py"):
        if (base / rel).is_file():
            return rel
    return None


def _module_of(rel: str) -> tuple[str, bool] | None:
    """``(dotted name, is_package)`` of a root-env source file, or None for a script."""
    if not rel.startswith("src/") or not rel.endswith(".py"):
        return None
    parts = rel[len("src/"):-len(".py")].split("/")
    if parts[-1] == "__init__":
        return ".".join(parts[:-1]), True
    return ".".join(parts), False


def _imports(base: Path, rel: str) -> tuple[set[str], list[str]]:
    """Root-env files ``rel`` imports (with their parent packages' ``__init__``), and problems."""
    try:
        tree = ast.parse((base / rel).read_text(encoding="utf-8"), filename=rel)
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        return set(), [f"frozen module does not parse: {rel}: {type(exc).__name__}"]
    me = _module_of(rel)
    wanted: list[tuple[str, list[str]]] = []  # (module, [names that may be submodules])
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            wanted.extend((a.name, []) for a in node.names if a.name.split(".")[0] == ROOT_PACKAGE)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if me is None:
                    problems.append(f"relative import outside the package: {rel}")
                    continue
                pkg = me[0].split(".") if me[1] else me[0].split(".")[:-1]
                if node.level - 1 >= len(pkg):
                    problems.append(f"relative import beyond {ROOT_PACKAGE}: {rel}")
                    continue
                up = pkg[: len(pkg) - (node.level - 1)]
                target = ".".join(up + ([node.module] if node.module else []))
            elif node.module and node.module.split(".")[0] == ROOT_PACKAGE:
                target = node.module
            else:
                continue
            wanted.append((target, [a.name for a in node.names if a.name != "*"]))
    out: set[str] = set()
    for module, names in wanted:
        f = _module_file(base, module)
        if f is None and (rel, module) in _UNRESOLVED_OK:
            continue
        if f is None and not (base / "src" / module.replace(".", "/")).is_dir():  # a namespace package resolves
            problems.append(f"{rel} imports {module}, which is not a file under src/")
            continue
        parts = module.split(".")
        for k in range(1, len(parts) + 1):
            init = _module_file(base, ".".join(parts[:k]))
            if init:
                out.add(init)
        for n in names:
            sub = _module_file(base, f"{module}.{n}")
            if sub:
                out.add(sub)
    return out, problems


def import_closure(root: str | Path, seeds: Iterable[str]) -> tuple[set[str], list[str]]:
    """Every root-env file reachable by import from the ``.py`` files among ``seeds`` that live under
    ``src/agent_loop_lab/`` or ``scripts/``, with the problems found on the way."""
    base = Path(root)
    todo = [s for s in seeds if s.endswith(".py") and (s.startswith(f"{SRC}/") or s.startswith("scripts/"))
            and (base / s).is_file()]
    seen: set[str] = set()
    problems: list[str] = []
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        found, probs = _imports(base, rel)
        problems.extend(probs)
        if rel in _DYNAMIC_IMPORTS:
            matches, links = _expand(base, _DYNAMIC_IMPORTS[rel])
            found.update(matches)
            problems.extend(f"symlink under a frozen tree: {link}" for link in links)
        todo.extend(sorted(found - seen))
    return seen, problems


# ---------------------------------------------------------------------------
# The file list
# ---------------------------------------------------------------------------


def frozen_files(root: str | Path = ".") -> tuple[list[str], list[str]]:
    """``(files, problems)``: the sorted frozen paths that exist, and what is wrong with the list
    (a missing required file or component, a glob that matches nothing, a symlink, an import that
    resolves to no file)."""
    base = Path(root)
    problems: list[str] = []
    found: set[str] = set()
    for rel in REQUIRED_FILES:
        if (base / rel).is_file():
            found.add(rel)
        else:
            problems.append(f"frozen file missing: {rel}")
    for pattern in FROZEN_GLOBS:
        matches, links = _expand(base, pattern)
        if not matches:
            problems.append(f"glob matches no file: {pattern}")
        problems.extend(f"symlink under a frozen tree: {link}" for link in links)
        found.update(matches)
    for name, patterns in REQUIRED_COMPONENTS:
        rxs = [_glob_regex(p) for p in patterns]
        if not any(rx.fullmatch(rel) for rel in found for rx in rxs):
            problems.append(f"required component missing: {name} (no file matches {' or '.join(patterns)})")
    closure, probs = import_closure(base, found)
    problems.extend(probs)
    found |= closure
    for rel in sorted(found):
        p = base / rel
        if p.is_symlink() or any(par.is_symlink() for par in p.parents if base in par.parents):
            problems.append(f"frozen file is a symlink or under one: {rel}")
    return sorted(found), list(dict.fromkeys(problems))


# ---------------------------------------------------------------------------
# Interface, model pins, packet
# ---------------------------------------------------------------------------


def interface_spec_sha256(root: str | Path = ".") -> str:
    """``spec_sha256()`` of ``interface.FROZEN``, executed from the source (stdlib only; nothing
    imported from ``hypaqwen``, no bytecode written)."""
    path = Path(root) / INTERFACE_FILE
    try:
        src = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryError(f"{INTERFACE_FILE} unreadable: {exc}") from exc
    name = "_hn_v04_frozen_interface"
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod  # dataclasses resolve annotations through sys.modules
    try:
        exec(compile(src, str(path), "exec"), mod.__dict__)
        value = mod.spec_sha256()
    except Exception as exc:  # noqa: BLE001 - any failure of the frozen file is a freeze failure
        raise RegistryError(f"{INTERFACE_FILE}: spec_sha256() failed: {type(exc).__name__}: {exc}") from exc
    finally:
        sys.modules.pop(name, None)
    if not isinstance(value, str) or not _HEX64.match(value):
        raise RegistryError(f"{INTERFACE_FILE}: spec_sha256() returned {value!r}")
    return value


def _literal(path: Path, name: str) -> Any:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise RegistryError(f"{path.name} unreadable: {exc}") from exc
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.value is not None:
            try:
                return ast.literal_eval(node.value)
            except ValueError as exc:
                raise RegistryError(f"{path.name}: {name} is not a literal") from exc
    raise RegistryError(f"{path.name}: no module-level {name}")


def model_pins(root: str | Path = ".") -> dict[str, Any]:
    base = Path(root)
    pins = {
        "id": _literal(base / PKG_INIT, "MODEL_ID"),
        "revision": _literal(base / PKG_INIT, "MODEL_REVISION"),
        "file_sha256": _literal(base / MODEL_FILE, "FILE_SHA256"),
    }
    fs = pins["file_sha256"]
    if not (isinstance(fs, dict) and fs and all(isinstance(k, str) and isinstance(v, str) and _HEX64.match(v)
                                                for k, v in fs.items())):
        raise RegistryError(f"{MODEL_FILE}: FILE_SHA256 is not a mapping of file name to sha256")
    if not (isinstance(pins["id"], str) and isinstance(pins["revision"], str)):
        raise RegistryError(f"{PKG_INIT}: MODEL_ID and MODEL_REVISION must be strings")
    return pins


def packet_hash(root: str | Path = ".", *, python: str | None = None) -> str:
    """Build the packet into a temporary directory and return its sha256 (of PACKET_MANIFEST.json)."""
    base = Path(root).resolve()
    script = base / PACKET_BUILDER
    if not script.is_file():
        raise RegistryError(f"{PACKET_BUILDER} not found: the packet builder must be merged before the freeze")
    with tempfile.TemporaryDirectory(prefix="v04_packet_") as tmp:
        out = Path(tmp) / "packet"
        try:
            proc = subprocess.run([python or sys.executable, str(script), "--out", str(out)], cwd=base,
                                  capture_output=True, text=True, timeout=PACKET_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RegistryError(f"{PACKET_BUILDER} could not run: {exc}") from exc
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout)[-2000:]
            raise RegistryError(f"{PACKET_BUILDER} failed with status {proc.returncode}: {tail}")
        found = _PACKET_RE.findall(proc.stdout)
        if len(found) != 1:
            raise RegistryError(f"{PACKET_BUILDER} printed {len(found)} 'packet sha256: <hex>' lines, expected 1")
        manifest = out / PACKET_MANIFEST
        if not manifest.is_file():
            raise RegistryError(f"{PACKET_BUILDER} wrote no {PACKET_MANIFEST}")
        actual = _sha256(manifest.read_bytes())
    if actual != found[0]:
        raise RegistryError(f"{PACKET_BUILDER} printed {found[0]} but {PACKET_MANIFEST} hashes to {actual}")
    return actual


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


def compute_freeze(root: str | Path = ".", *, python: str | None = None) -> dict[str, Any]:
    """The current tree's record, with ``revisions`` empty (``revise`` fills it)."""
    base = Path(root)
    files, problems = frozen_files(base)
    if problems:
        raise RegistryError("the frozen file list is incomplete:\n  " + "\n  ".join(problems))
    data = {rel: (base / rel).read_bytes() for rel in files}
    crlf = [rel for rel, b in data.items() if b"\r" in b]
    if crlf:
        raise RegistryError(f"frozen files have CR line endings: {crlf}")
    spec = interface_spec_sha256(base)
    if spec != INTERFACE_SPEC_SHA256:
        raise RegistryError(f"interface spec_sha256 {spec} != the pre-registered {INTERFACE_SPEC_SHA256}")
    record: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "globs": list(FROZEN_GLOBS),
        "files": {rel: _sha256(b) for rel, b in data.items()},
        "interface_spec_sha256": spec,
        "model": model_pins(base),
        "packet_sha256": packet_hash(base, python=python),
        "revisions": [],
    }
    record["freeze_sha256"] = freeze_digest(record)
    return record


def render(record: dict[str, Any]) -> str:
    return _HEADER + yaml.safe_dump(record, sort_keys=True, default_flow_style=False, width=1000)


def _load(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as exc:
        raise RegistryError(f"registry unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistryError("registry is not a mapping")
    return data


def _undo(record: dict[str, Any]) -> dict[str, Any]:
    """The record before its last revision (raises :class:`RegistryError` if that revision is malformed)."""
    rev = record["revisions"][-1]
    if not isinstance(rev, dict) or set(rev) != REVISION_KEYS:
        raise RegistryError(f"revision {len(record['revisions'])} has keys {sorted(rev) if isinstance(rev, dict) else rev!r}")
    files = dict(record.get("files") or {})
    added, changed = rev.get("added") or {}, rev.get("changed") or {}
    if not (isinstance(added, dict) and isinstance(changed, dict)):
        raise RegistryError(f"revision {len(record['revisions'])}: added and changed must be mappings")
    for rel, sha in added.items():
        if files.pop(rel, None) != sha:
            raise RegistryError(f"revision {len(record['revisions'])}: added {rel} is not in the record with that hash")
    for rel, ba in changed.items():
        if not (isinstance(ba, dict) and set(ba) == {"before", "after"} and files.get(rel) == ba["after"]):
            raise RegistryError(f"revision {len(record['revisions'])}: changed {rel} does not match the record")
        files[rel] = ba["before"]
    prev = {**record, "files": files, "globs": rev.get("globs_before"), "revisions": record["revisions"][:-1]}
    prev["freeze_sha256"] = freeze_digest(prev)
    return prev


def chain_problems(record: dict[str, Any]) -> list[str]:
    """Undo every revision and check each step's ``previous_freeze_sha256``."""
    revs = record.get("revisions")
    if not isinstance(revs, list):
        return ["registry revisions is not a list"]
    problems: list[str] = []
    cur = record
    while cur["revisions"]:
        n = len(cur["revisions"])
        try:
            prev = _undo(cur)
        except RegistryError as exc:
            return problems + [str(exc)]
        if prev["freeze_sha256"] != cur["revisions"][-1].get("previous_freeze_sha256"):
            problems.append(f"revision {n}: the record it revised does not hash to its previous_freeze_sha256")
        cur = prev
    return problems


def write_registry(root: str | Path = ".", *, overwrite: bool = False, python: str | None = None) -> tuple[Path, dict[str, Any]]:
    """Write the G1 registry. Refuses if one exists, unless ``overwrite`` (``--force``: only before
    any launch or scoring run has used the registry; afterwards a change is a :func:`revise_registry`)."""
    path = Path(root) / REGISTRY_RELPATH
    if path.exists() and not overwrite:
        raise RegistryError(f"{path} exists; check it with --check, revise it with --revise, or pass --force "
                            "to replace it (only before any launch or scoring run)")
    record = compute_freeze(root, python=python)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render(record).encode("utf-8"))
    return path, record


def revise_registry(root: str | Path = ".", *, reason: str, allow_change: Sequence[str] = (),
                    date: str | None = None, python: str | None = None) -> tuple[Path, dict[str, Any]]:
    """Append an additive revision (PREREGISTRATION.md §13): new files are added; an existing file may
    change only if it is in ``allow_change`` (each such path must actually change); nothing may be
    removed; interface, model and packet must be unchanged."""
    base = Path(root)
    path = base / REGISTRY_RELPATH
    if not path.is_file():
        raise RegistryError(f"no registry to revise at {path}; write the G1 registry first")
    old = _load(path)
    bad = []
    if set(old) != KEYS:
        bad.append(f"registry keys {sorted(map(str, old))} != {sorted(KEYS)}")
    elif old.get("freeze_sha256") != freeze_digest(old):
        bad.append("registry freeze_sha256 does not match its record")
    else:
        bad.extend(chain_problems(old))
    if bad:
        raise RegistryError("the registry is not intact; nothing was revised:\n  " + "\n  ".join(bad))
    reason = (reason or "").strip()
    if not reason:
        raise RegistryError("a revision needs a reason")
    day = date or _date.today().isoformat()
    if not _DATE.match(day):
        raise RegistryError(f"revision date {day!r} is not YYYY-MM-DD")
    new = compute_freeze(base, python=python)
    errors = [f"{k} differs from the registry; a revision may not change it"
              for k in ("spec_version", "interface_spec_sha256", "model", "packet_sha256") if new[k] != old[k]]
    old_files, new_files = old["files"], new["files"]
    allowed = set(allow_change)
    errors += [f"frozen file removed: {rel}" for rel in sorted(set(old_files) - set(new_files))]
    changed = {rel: {"before": old_files[rel], "after": new_files[rel]}
               for rel in sorted(set(old_files) & set(new_files)) if old_files[rel] != new_files[rel]}
    errors += [f"frozen file modified without --allow-change: {rel}" for rel in changed if rel not in allowed]
    errors += [f"--allow-change {rel} names a file that did not change" for rel in sorted(allowed - set(changed))]
    added = {rel: new_files[rel] for rel in sorted(set(new_files) - set(old_files))}
    if not (added or changed or new["globs"] != old["globs"]):
        errors.append("nothing to revise: the tree matches the registry")
    if errors:
        raise RegistryError("revision refused:\n  " + "\n  ".join(errors))
    record = {**new, "revisions": [*old["revisions"], {
        "date": day, "reason": reason, "previous_freeze_sha256": old["freeze_sha256"],
        "globs_before": old["globs"], "added": added, "changed": changed}]}
    record["freeze_sha256"] = freeze_digest(record)
    path.write_bytes(render(record).encode("utf-8"))
    return path, record


def _v03_problems(base: Path) -> list[str]:
    from ..v03 import registry as v03_registry

    return [f"v0.3 registry: {p}" for p in v03_registry.check_registry(base)]


def check_registry(root: str | Path = ".", *, python: str | None = None, check_v03: bool = True) -> list[str]:
    """Every mismatch between the registry and the tree, one line each (empty when it holds). With
    ``check_v03`` (the default, and always from the CLI) the v0.3 registry is checked as well."""
    base = Path(root)
    path = base / REGISTRY_RELPATH
    if not path.is_file():
        return [f"registry not found: {path}"]
    try:
        data = _load(path)
    except RegistryError as exc:
        return [str(exc)]
    problems: list[str] = []
    if set(data) != KEYS:
        problems.append(f"registry keys {sorted(map(str, data))} != {sorted(KEYS)}")
    if data.get("spec_version") != SPEC_VERSION:
        problems.append(f"spec_version {data.get('spec_version')!r} != {SPEC_VERSION!r}")
    if data.get("globs") != list(FROZEN_GLOBS):
        problems.append("registry globs differ from the freeze script's FROZEN_GLOBS")
    files = data.get("files") if isinstance(data.get("files"), dict) else {}
    current, list_problems = frozen_files(base)
    problems.extend(list_problems)
    for rel in sorted(set(current) | set(REQUIRED_FILES)):
        if rel not in files:
            problems.append(f"not in registry: {rel}")
    for rel in sorted(files):
        p = base / rel
        if not p.is_file():
            problems.append(f"frozen file missing: {rel}")
        elif rel not in current:
            problems.append(f"registry lists a file that is not frozen: {rel}")
        elif _sha256(p.read_bytes()) != files[rel]:
            problems.append(f"frozen file modified: {rel}")
    for key, fn in (("interface_spec_sha256", interface_spec_sha256), ("model", model_pins),
                    ("packet_sha256", lambda r: packet_hash(r, python=python))):
        try:
            actual = fn(base)
        except RegistryError as exc:
            problems.append(f"{key}: {exc}")
            continue
        if actual != data.get(key):
            problems.append(f"{key} differs (registry {data.get(key)!r}, recomputed {actual!r})")
    if data.get("interface_spec_sha256") != INTERFACE_SPEC_SHA256:
        problems.append(f"registry interface_spec_sha256 is not the pre-registered {INTERFACE_SPEC_SHA256}")
    if data.get("freeze_sha256") != freeze_digest(data):
        problems.append("registry freeze_sha256 does not match its record")
    if "revisions" in data:
        problems.extend(chain_problems(data))
    if check_v03:
        problems.extend(_v03_problems(base))
    return list(dict.fromkeys(problems))


def g1_freeze_sha256(record: dict[str, Any]) -> str:
    """The ``freeze_sha256`` of the G1 record, reconstructed by undoing every revision."""
    cur = record
    while cur.get("revisions"):
        cur = _undo(cur)
    return cur["freeze_sha256"]


def problems_text(problems: Sequence[str]) -> str:
    return "\n".join(f"MISMATCH: {p}" for p in problems)
