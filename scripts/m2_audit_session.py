"""Audit a weight-author session transcript (spec v0.2 §8, §9.2-9.3).

    uv run python scripts/m2_audit_session.py LOG.stream.jsonl AUTHOR_DIR PHASE

Scans every tool call in the stream-json transcript and reports:
- any file path or command that refers outside AUTHOR_DIR (repo, harness scratch, Claude project data);
- writes inside packet/ (or, in A-selfcheck, inside output/phase-a-strict/; in phase-b-revision, inside
  PHASE_B_READ_ONLY: output/phase-a-strict/, output/phase-a-selfcheck/, output/selfcheck_scripts/,
  disclosure/ and tools/);
- Bash commands other than the phase's allowed ./py invocations (A-strict: py_compile and shape_check only);
- the code the author writes and runs: Write ``content``, Edit ``new_string`` (applied to the replayed
  file when its content is known) and inline ``./py -c`` code are scanned for the forbidden markers,
  references to the repo directory (``../<repo name>``), string paths that resolve outside AUTHOR_DIR,
  and file-changing calls (open(..., 'w'), shutil.copy/move, Path.write_*, os.remove, ...) whose target
  lies in a directory the phase protects (``protected_dirs``). ``./py`` is an unsandboxed interpreter,
  so this is where repo or archive access through a script shows up. Static only: targets that cannot
  be resolved are listed under ``unresolved_writes`` and scripts run without a scanned Write under
  ``unscanned_scripts`` for the adjudicator; neither is a violation;
- refused tool calls (permission denials) and the model actually used.
Writes LOG.audit.json next to the log and exits 1 if any violation was found.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FORBIDDEN_MARKERS = (str(REPO) + "/", str(REPO) + '"', "/private/tmp/claude-501", "/.claude/projects", "HypaNet/src", "HypaNet/datasets")
PATH_KEYS = ("file_path", "path", "notebook_path", "pattern", "command", "glob")
PHASE_B = "phase-b-revision"
# Author-dir subdirectories the Phase B author may read but never write (experiments/m3/README.md §4);
# packet/ is covered for every phase by the generic check.
PHASE_B_READ_ONLY = ("output/phase-a-strict/", "output/phase-a-selfcheck/", "output/selfcheck_scripts/", "disclosure/", "tools/")


def phase_b_read_only_hit(target: str, author_dir: Path) -> str | None:
    """The read-only Phase B directory ``target`` lies in, or None.

    Relative paths are taken from AUTHOR_DIR (the session's working directory). After
    normalisation, a path under AUTHOR_DIR is matched from the author-dir root; any other
    absolute path is matched as a substring.
    """
    if not target:
        return None
    norm = os.path.normpath(target if os.path.isabs(target) else os.path.join(str(author_dir), target))
    root = str(author_dir).rstrip("/") + "/"
    rel = norm[len(root) :] + "/" if norm.startswith(root) else None
    for d in PHASE_B_READ_ONLY:
        if (rel is not None and rel.startswith(d)) or (rel is None and "/" + d in norm + "/"):
            return d
    return None


def protected_dirs(phase: str) -> tuple[str, ...]:
    """Author-dir directories the phase may not change (packet/ always)."""
    if phase == PHASE_B:
        return PHASE_B_READ_ONLY + ("packet/",)
    if phase == "phase-a-selfcheck":
        return ("packet/", "output/phase-a-strict/")
    return ("packet/",)


def protected_hit(target: str, author_dir: Path, dirs: tuple[str, ...]) -> str | None:
    """Like :func:`phase_b_read_only_hit` for an arbitrary set of author-dir directories."""
    if not target:
        return None
    norm = os.path.normpath(target if os.path.isabs(target) else os.path.join(str(author_dir), target))
    root = str(author_dir).rstrip("/") + "/"
    rel = norm[len(root) :] + "/" if norm.startswith(root) else None
    for d in dirs:
        if (rel is not None and rel.startswith(d)) or (rel is None and "/" + d in norm + "/"):
            return d
    return None


# -- static scan of code the author writes or runs ------------------------------------------------

# A reference to the repo directory itself as a path component ("../HypaNet", "../../HypaNet/x"),
# but not to a sibling such as the author dir "HypaNet-m2-author".
REPO_REF = re.compile(r"(?:\.\./|/)" + re.escape(REPO.name) + r"(?![\w-])")
_WILD = "\0"  # stands for any part of a path the static folder cannot resolve
_ABS_ALLOWED = ("/dev/null", "/usr/bin", "/bin")
_PATH_CTORS = {"Path", "PurePath", "PosixPath", "PurePosixPath"}
_OPEN_WRITE_MODES = set("wax+")
# (qualified call name) -> indices of the positional args that name a changed path
_FUNC_SINKS: dict[str, tuple[int, ...]] = {
    **{f"shutil.{n}": (1,) for n in ("copy", "copy2", "copyfile", "copytree", "copymode", "copystat")},
    "shutil.move": (0, 1),
    "shutil.rmtree": (0,),
    **{f"os.{n}": (0, 1) for n in ("rename", "replace", "renames")},
    **{f"os.{n}": (0,) for n in ("remove", "unlink", "rmdir", "removedirs", "mkdir", "makedirs", "truncate", "chmod", "utime")},
    **{f"os.{n}": (1,) for n in ("symlink", "link")},
    **{f"numpy.{n}": (0,) for n in ("save", "savez", "savez_compressed", "savetxt")},
    **{f"np.{n}": (0,) for n in ("save", "savez", "savez_compressed", "savetxt")},
    "torch.save": (1,),
}
# Path methods that change the receiver (and, for rename/replace, the first argument).
_METHOD_SINKS = {"write_text", "write_bytes", "unlink", "rmdir", "mkdir", "touch", "chmod", "symlink_to", "hardlink_to"}
_METHOD_MOVE_SINKS = {"rename", "replace"}
_LINE_SINK = re.compile(
    r"open\s*\(|shutil\.|os\.(?:rename|replace|remove|unlink|rmdir|mkdir|makedirs|truncate|chmod|symlink|link)\b"
    r"|\.(?:write_text|write_bytes|unlink|rmdir|mkdir|touch|rename|replace|chmod)\s*\(|\b(?:np|numpy|torch)\.save"
)
_STR_LIT = re.compile(r"""(?:[rbuRBU]{0,2})("([^"\\\n]*)"|'([^'\\\n]*)')""")


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _join(parts: list[str]) -> str:
    out = ""
    for p in parts:
        if p.startswith("/") or not out:
            out = p
        else:
            out = out.rstrip("/") + "/" + p
    return out


def _fold(node: ast.AST | None, env: dict[str, str]) -> str:
    """Best-effort static value of a path expression; unknown parts become ``_WILD``."""
    if node is None:
        return _WILD
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id, _WILD)
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else _WILD for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _fold(node.left, env) + _fold(node.right, env)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _join([_fold(node.left, env), _fold(node.right, env)])
    if isinstance(node, ast.Call):
        name = _dotted(node.func) or ""
        last = name.rsplit(".", 1)[-1]
        if last in _PATH_CTORS or name in ("os.path.join", "posixpath.join"):
            return _join([_fold(a, env) for a in node.args]) if node.args else "."
        if isinstance(node.func, ast.Attribute) and node.func.attr == "joinpath":
            return _join([_fold(node.func.value, env)] + [_fold(a, env) for a in node.args])
        if name in ("str", "os.fspath", "os.path.normpath", "os.path.abspath") and node.args:
            return _fold(node.args[0], env)
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("resolve", "absolute", "expanduser"):
            return _fold(node.func.value, env)
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        inner = _fold(node.value, env)
        return _WILD if _WILD in inner else os.path.dirname(inner.rstrip("/"))
    return _WILD


def _write_targets(tree: ast.AST) -> list[tuple[int, str, str]]:
    """``(line, call, folded target)`` of every file-changing call in ``tree``."""
    env: dict[str, str] = {}
    out: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):  # breadth-first; good enough for straight-line scripts
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            env[node.targets[0].id] = _fold(node.value, env)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func) or ""
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        if name in ("open", "io.open", "builtins.open", "os.open", "codecs.open"):
            mode = kw.get("mode", node.args[1] if len(node.args) > 1 else None)
            mode_s = mode.value if isinstance(mode, ast.Constant) and isinstance(mode.value, str) else None
            if name == "os.open" or mode_s is None and mode is not None or (mode_s and _OPEN_WRITE_MODES & set(mode_s)):
                out.append((node.lineno, name, _fold(kw.get("file", node.args[0] if node.args else None), env)))
            continue
        if name in _FUNC_SINKS:
            for i in _FUNC_SINKS[name]:
                arg = node.args[i] if len(node.args) > i else kw.get(("src", "dst")[min(i, 1)])
                if arg is not None:
                    out.append((node.lineno, name, _fold(arg, env)))
            continue
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr == "open" and name not in ("os.open",):
                mode = kw.get("mode", node.args[0] if node.args else None)
                mode_s = mode.value if isinstance(mode, ast.Constant) and isinstance(mode.value, str) else None
                if mode_s and _OPEN_WRITE_MODES & set(mode_s):
                    out.append((node.lineno, f".{attr}", _fold(node.func.value, env)))
            elif attr in _METHOD_SINKS:
                out.append((node.lineno, f".{attr}", _fold(node.func.value, env)))
            elif attr in _METHOD_MOVE_SINKS and node.args:
                out.append((node.lineno, f".{attr}", _fold(node.func.value, env)))
                out.append((node.lineno, f".{attr}", _fold(node.args[0], env)))
    return out


def _outside(path: str, author_dir: Path) -> bool:
    if path.startswith(_ABS_ALLOWED):
        return False
    norm = os.path.normpath(path if os.path.isabs(path) else os.path.join(str(author_dir), path))
    root = str(author_dir).rstrip("/")
    return norm != root and not norm.startswith(root + "/")


def _looks_like_escape(s: str) -> bool:
    """A string constant that names a path outside the working directory."""
    if s.startswith("~"):
        return True
    if re.match(r"^/[A-Za-z0-9_.~-]", s):
        return True
    parts = s.split("/")
    # "../x" or "a/../../x": a dot-dot segment followed by a named segment (bare "..", "../.." are
    # usually relative to __file__ and are not flagged on their own).
    return ".." in parts and any(p not in ("", ".", "..") for p in parts[parts.index("..") :])


def scan_code(text: str, *, python: bool, author_dir: Path, dirs: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Static findings for code or text the author wrote: ``(violations, unresolved write targets)``."""
    found: list[str] = []
    unresolved: list[str] = []
    for marker in FORBIDDEN_MARKERS:
        if marker in text:
            found.append(f"refers to {marker!r}")
    if REPO_REF.search(text):
        found.append(f"refers to the repo directory {REPO.name!r}: {REPO_REF.search(text).group(0)!r}")
    if not python:
        return found, unresolved
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "\n" not in node.value:
                s = node.value
                if _looks_like_escape(s) and _outside(os.path.expanduser(s) if s.startswith("~") else s, author_dir):
                    found.append(f"line {node.lineno}: path outside the author dir: {s[:120]!r}")
        for line, call, target in _write_targets(tree):
            prefix = target.split(_WILD, 1)[0]
            hit = protected_hit(prefix, author_dir, dirs) if prefix else None
            if hit:  # the resolved prefix already lies in a protected directory
                found.append(f"line {line}: {call} changes {target.replace(_WILD, '*')!r} inside protected {hit}")
            elif _WILD in target and not prefix.strip("./"):
                unresolved.append(f"line {line}: {call} target {target.replace(_WILD, '*')!r}")
            elif _WILD not in target and _outside(target, author_dir):
                found.append(f"line {line}: {call} changes {target!r} outside the author dir")
        return found, unresolved
    # Not parseable (e.g. an Edit fragment): line-level fallback over string literals.
    for n, line in enumerate(text.splitlines(), start=1):
        lits = [m.group(2) if m.group(2) is not None else m.group(3) for m in _STR_LIT.finditer(line)]
        for s in lits:
            if _looks_like_escape(s) and _outside(s, author_dir):
                found.append(f"line {n}: path outside the author dir: {s[:120]!r}")
        if _LINE_SINK.search(line):
            for s in lits:
                hit = protected_hit(s, author_dir, dirs)
                if hit:
                    found.append(f"line {n}: file-changing call names {s!r} inside protected {hit}")
    return found, unresolved


def _inline_code(cmd: str) -> str | None:
    """The code of ``./py -c CODE`` (None for other commands)."""
    try:
        argv = shlex.split(cmd)
    except ValueError:
        return cmd if " -c " in f" {cmd} " else None
    if len(argv) >= 3 and argv[0] == "./py" and argv[1] == "-c":
        return argv[2]
    return None


def _script_arg(cmd: str) -> str | None:
    try:
        argv = shlex.split(cmd)
    except ValueError:
        return None
    if len(argv) >= 2 and argv[0] == "./py" and argv[1].endswith(".py"):
        return argv[1]
    return None


def tool_calls(log: Path):
    for line in log.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = ev.get("message") or {}
        for block in msg.get("content") or [] if isinstance(msg, dict) else []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                yield ev, block
        if ev.get("type") == "result":
            yield ev, None


def main(argv: list[str]) -> int:
    log, author_dir, phase = Path(argv[1]), Path(argv[2]).resolve(), argv[3]
    violations, calls, models, denials = [], [], set(), []
    dirs = protected_dirs(phase)
    files: dict[str, str | None] = {}  # replayed author-dir files: normpath -> content (None = unknown)
    unresolved_writes: list[str] = []
    unscanned_scripts: list[str] = []
    scanned = 0

    def norm(p: str) -> str:
        return os.path.normpath(p if os.path.isabs(p) else os.path.join(str(author_dir), p))

    def scan(tag: str, what: str, text: str, python: bool) -> None:
        nonlocal scanned
        scanned += 1
        found, unresolved = scan_code(text, python=python, author_dir=author_dir, dirs=dirs)
        violations.extend(tag + f"{what}: {f}" for f in found)
        unresolved_writes.extend(f"{tag}{what}: {u}" for u in unresolved)

    events = list(tool_calls(log))
    denied_ids = {d.get("tool_use_id") for ev, b in events if b is None for d in ev.get("permission_denials") or []}
    for ev, block in events:
        if block is None:
            for d in ev.get("permission_denials") or []:
                denials.append(d)
            continue
        tag = "[refused] " if block.get("id") in denied_ids else "[EXECUTED] "
        model = (ev.get("message") or {}).get("model")
        if model:
            models.add(model)
        name, inp = block.get("name"), block.get("input") or {}
        text = " ".join(str(inp.get(k, "")) for k in PATH_KEYS)
        calls.append({"tool": name, "input": {k: inp[k] for k in PATH_KEYS if k in inp}})
        for marker in FORBIDDEN_MARKERS:
            if marker in text:
                violations.append(tag + f"{name}: refers to {marker!r}: {text[:200]}")
        for m in re.findall(r"(?:^|(?<=[\s=\"'(]))(/[^\s'\";|&)]+)", text):
            p = Path(m)
            if p.is_absolute() and not str(p).startswith(str(author_dir)) and not m.startswith(("/dev/null", "/usr/bin", "/bin")):
                violations.append(tag + f"{name}: absolute path outside author dir: {m}")
        target = str(inp.get("file_path", ""))
        if name in ("Write", "Edit", "NotebookEdit") and "/packet/" in target:
            violations.append(tag + f"{name}: write inside packet: {target}")
        if phase == "phase-a-selfcheck" and name in ("Write", "Edit") and "/output/phase-a-strict/" in target:
            violations.append(tag + f"{name}: edit of archived A-strict deliverable: {target}")
        if phase == PHASE_B and name in ("Write", "Edit", "NotebookEdit"):
            hit = phase_b_read_only_hit(target, author_dir)
            if hit:
                violations.append(tag + f"{name}: write inside read-only {hit} in Phase B: {target}")
        executed = tag == "[EXECUTED] "
        if name in ("Write", "Edit", "MultiEdit", "NotebookEdit") and target:
            key, python = norm(target), target.endswith(".py")
            if name == "Write":
                text = str(inp.get("content", ""))
                if executed:
                    files[key] = text
                scan(tag, f"{name} {target} content", text, python)
            else:
                edits = inp.get("edits") if name == "MultiEdit" else [inp]
                current = files.get(key)
                for e in edits or []:
                    old, new = str(e.get("old_string", "")), str(e.get("new_string", e.get("new_source", "")))
                    if current is not None and old and old in current:
                        current = current.replace(old, new) if e.get("replace_all") else current.replace(old, new, 1)
                    else:
                        current = None
                        scan(tag, f"{name} {target} new_string", new, python)
                if current is not None:
                    scan(tag, f"{name} {target} (replayed file)", current, python)
                if executed:
                    files[key] = current
        if name == "Bash":
            cmd = str(inp.get("command", "")).strip()
            code = _inline_code(cmd)
            if code is not None:
                scan(tag, "Bash inline code", code, True)
            script = _script_arg(cmd)
            if executed and script and norm(script) not in files:
                unscanned_scripts.append(script)
            if phase == "phase-a-strict":
                ok = cmd.startswith("./py -m py_compile ") or cmd == "./py tools/shape_check.py output/phase-a-strict"
            else:
                ok = cmd.startswith("./py ")
            if not ok:
                violations.append(tag + f"Bash outside phase allowance: {cmd[:200]}")
    report = {
        "phase": phase,
        "log": str(log),
        "models": sorted(models),
        "tool_calls": len(calls),
        "permission_denials": denials,
        "violations": violations,
        "calls": calls,
        "content_scans": scanned,
        "protected_dirs": list(dirs),
        "unresolved_writes": unresolved_writes,
        "unscanned_scripts": sorted(set(unscanned_scripts)),
    }
    out = log.with_suffix(".audit.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"audit: {len(calls)} tool calls, models {sorted(models)}, {len(denials)} denials, {len(violations)} violations -> {out}")
    for v in violations:
        print("  VIOLATION:", v)
    executed = [v for v in violations if v.startswith("[EXECUTED]")]
    report["executed_violations"] = len(executed)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"  executed violations: {len(executed)} (refused attempts are listed but were blocked)")
    return 1 if executed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
