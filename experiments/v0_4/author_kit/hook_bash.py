"""PreToolUse hook of the v0.4 weight-author session (docs/V04_WSP_KERNEL.md §7). Standard library only.

Installed read-only as ``AUTHOR_DIR/tools/hn_hook/hook_bash.py`` by the prepare script and registered in
the session settings (``settings_wsp_a.json`` / ``settings_wsp_b.json``) as::

    "$CLAUDE_PROJECT_DIR"/.venv/bin/python -I -B "$CLAUDE_PROJECT_DIR"/tools/hn_hook/hook_bash.py PHASE

Claude Code shows the session the hook's command line verbatim when the hook blocks a call
("PreToolUse:Bash hook error: [<command>]: <reason>"; checked with a real ``claude -p`` session), so the
command names no harness path: only ``$CLAUDE_PROJECT_DIR``, unexpanded. The author dir is this file's
grandparent's parent. The input is Claude Code's PreToolUse JSON on stdin (``tool_name``, ``tool_input``,
``cwd``, ``session_id``, ``transcript_path``, ``hook_event_name``, ...).

Decision: exit 0 with no output lets the call go on to the permission rules (it never approves anything
by itself); exit 2 blocks it with the reason on stderr. A hook that fails any other way fails *open*
(Claude Code runs the call), so every error inside the hook is turned into a block, and the launcher
runs this hook on sample calls before each session.

Rules (every rule is also checked by the session audit, which loads this file by path):

* ``Bash``: the command must match ``^\\./(py|fwd) [^;&|<>`$\\n]+$`` as a whole string (``re.fullmatch``,
  so a trailing newline does not slip past ``$``), and also, stricter: every space-separated word is
  made of ``[A-Za-z0-9_./=:,+@%-]`` (no quotes, globs, ``~``, braces, backslashes, ``#``, ``!``, ``(``,
  tabs or carriage returns), no word has a ``..`` component or is absolute, and
  - ``./fwd REQUEST.json``: exactly one argument, a ``.json`` file inside ``output/<phase>/``;
  - ``./py SCRIPT.py [ARGS]``: the script is a ``.py`` file inside ``output/<phase>/``;
    ``./py -m py_compile FILE.py`` likewise; ``./py -c`` and any other ``-m`` are refused.
* ``Write`` / ``Edit`` / ``MultiEdit`` / ``NotebookEdit``: the target, after resolving symlinks, lies inside
  ``output/<phase>/`` and not inside ``output/<phase>/fwd/`` (the forward tool's records).
* ``Read`` / ``Glob`` / ``Grep``: every path argument, after resolving symlinks, lies inside the author
  dir and not inside ``.kernel/``; a ``Glob`` pattern or ``Grep`` glob is relative and has no ``..``.
* The session's ``cwd`` must be the author dir. Any other tool is refused.

CLI: ``hook_bash.py PHASE [AUTHOR_DIR]`` (the second argument is for tests).
"""

import json
import os
import re
import sys

COMMAND_RE = re.compile(r"\./(py|fwd) [^;&|<>`$\n]+")  # the contract's regex, applied with fullmatch
WORD_RE = re.compile(r"[A-Za-z0-9_./=:,+@%-]+")
PHASE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
READ_TOOLS = ("Read", "Glob", "Grep")
WRITE_TOOLS = ("Write", "Edit", "MultiEdit", "NotebookEdit")
KIT_TOOLS = READ_TOOLS + WRITE_TOOLS + ("Bash",)


def _inside(path, root):
    root = root.rstrip(os.sep)
    return path == root or path.startswith(root + os.sep)


def _real(path, author):
    """``path`` (absolute, or relative to the author dir) with symlinks resolved."""
    return os.path.realpath(path if os.path.isabs(path) else os.path.join(author, path))


def _relative_word(word, what):
    """A command argument that names a file: relative, no ``..`` component, not an option."""
    if word.startswith("-"):
        return f"{what} may not start with '-'"
    if word.startswith("/"):
        return f"{what} must be a relative path"
    if ".." in word.split("/"):
        return f"{what} may not contain '..'"
    return None


def _in_phase(word, author, phase, suffix, what):
    bad = _relative_word(word, what)
    if bad:
        return bad
    phase_dir = os.path.realpath(os.path.join(author, "output", phase))
    path = _real(word, author)
    if not _inside(path, phase_dir) or path == phase_dir:
        return f"{what} must lie inside output/{phase}/"
    if not path.endswith(suffix):
        return f"{what} must be a {suffix} file"
    return None


def check_bash(command, author, phase):
    """None if ``command`` is an allowed ``./py`` / ``./fwd`` call, else the reason it is refused."""
    if not isinstance(command, str) or not COMMAND_RE.fullmatch(command):
        return ("only a single './py output/{p}/SCRIPT.py [ARGS]' or './fwd output/{p}/REQUEST.json' command "
                "is allowed per Bash call: no ';', '&&', '||', pipes, redirection, backticks, '$', newlines "
                "or other programs").format(p=phase)
    words = command.split(" ")
    if any(not WORD_RE.fullmatch(w) for w in words[1:]):
        return ("arguments may contain only letters, digits and _ . / = : , + @ % - separated by single spaces "
                "(no quotes, globs, ~, braces, backslashes, tabs or other shell syntax)")
    tool, args = words[0], words[1:]
    if tool == "./fwd":
        if len(args) != 1:
            return "./fwd takes exactly one argument: the request file"
        return _in_phase(args[0], author, phase, ".json", "the request file")
    if args[0] == "-c":
        return "./py -c is not supported: write the code to a file under output/{p}/ and run it".format(p=phase)
    if args[0] == "-m":
        if len(args) != 3 or args[1] != "py_compile":
            return "the only -m form is: ./py -m py_compile output/{p}/FILE.py".format(p=phase)
        return _in_phase(args[2], author, phase, ".py", "the file")
    return _in_phase(args[0], author, phase, ".py", "the script")


def check_write(path, author, phase):
    if not isinstance(path, str) or not path:
        return "missing file path"
    phase_dir = os.path.realpath(os.path.join(author, "output", phase))
    target = _real(path, author)
    if not _inside(target, phase_dir) or target == phase_dir:
        return f"writes are allowed only inside output/{phase}/"
    if _inside(target, os.path.join(phase_dir, "fwd")):
        return f"output/{phase}/fwd/ holds the forward tool's records and is written only by ./fwd"
    return None


def check_read_path(path, author):
    if not isinstance(path, str):
        return "bad path argument"
    if not path:
        return None
    target = _real(path, os.path.realpath(author))
    root = os.path.realpath(author)
    if not _inside(target, root):
        return "reads are allowed only inside this directory"
    if _inside(target, os.path.join(root, ".kernel")):
        return ".kernel/ holds the kernel's sockets and cannot be read"
    return None


def _pattern(p, what):
    if not isinstance(p, str):
        return f"bad {what}"
    if p.startswith(("/", "~")) or ".." in re.split(r"[/\\]", p):
        return f"the {what} must be relative to this directory and may not contain '..'"
    return None


def decide(event, author, phase):
    """None to let the call proceed to the permission rules, else the reason it is blocked."""
    if not PHASE_RE.fullmatch(phase or ""):
        return "hook misconfigured: bad phase"
    author = os.path.realpath(author)
    if not isinstance(event, dict):
        return "hook input is not an object"
    cwd = event.get("cwd")
    if not isinstance(cwd, str) or os.path.realpath(cwd) != author:
        return "the session must run in the author directory"
    tool, inp = event.get("tool_name"), event.get("tool_input")
    if not isinstance(inp, dict):
        return "hook input has no tool_input object"
    if tool == "Bash":
        return check_bash(inp.get("command"), author, phase)
    if tool in WRITE_TOOLS:
        return check_write(inp.get("file_path") or inp.get("notebook_path"), author, phase)
    if tool == "Read":
        return check_read_path(inp.get("file_path") or "", author) if inp.get("file_path") else "missing file path"
    if tool == "Glob":
        return check_read_path(inp.get("path") or "", author) or _pattern(inp.get("pattern"), "pattern")
    if tool == "Grep":
        bad = check_read_path(inp.get("path") or "", author)
        if bad is None and inp.get("glob") is not None:
            bad = _pattern(inp.get("glob"), "glob")
        return bad
    return f"tool {tool!r} is not part of the author kit (Read, Glob, Grep, Write, Edit, Bash)"


def default_author_dir():
    """``AUTHOR_DIR`` for the installed copy ``AUTHOR_DIR/tools/hn_hook/hook_bash.py``."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def main(argv):
    try:
        if len(argv) not in (1, 2):
            raise ValueError("usage: hook_bash.py PHASE [AUTHOR_DIR]")
        phase = argv[0]
        author = argv[1] if len(argv) == 2 else default_author_dir()
        event = json.loads(sys.stdin.read())
        reason = decide(event, author, phase)
    except Exception as exc:  # noqa: BLE001 - fail closed: any error blocks the call
        reason = f"hook error ({type(exc).__name__}); the call is blocked"
    if reason is None:
        return 0
    print(f"refused: {reason}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        code = main(sys.argv[1:])
    except BaseException:  # noqa: BLE001
        code = 2
    sys.exit(code)
