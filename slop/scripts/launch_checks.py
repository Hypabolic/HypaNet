"""Pre-launch checks for scripts/launch.sh that are easier in Python than in sh (SPEC.md §3, §12.7, §13).

    uv run python scripts/launch_checks.py version "VERSION TEXT" MIN        # exit 1 if claude --version < MIN
    uv run python scripts/launch_checks.py settings SETTINGS AUTHOR_DIR      # exit 1 on any PROBLEM line
    uv run python scripts/launch_checks.py budget MODE REMAINING             # print the --max-budget-usd flag

`settings` checks kit/settings.json beyond `kit/make_settings.py --check` (which compares it with what that script
would write now): it must be valid JSON with `permissions.defaultMode` `dontAsk`, `autoMemoryEnabled: false` and
`switchModelsOnFlag: false`; deny Read of the repo, of /tmp/claude-<uid> and /private/tmp/claude-<uid>, of the
private note directory (~/Development/hypabolic/slop-private), of ~/.codex and ~/.claude.json (Codex and Claude
Code session state) and of every sealed repo (scripts/build_sets.py SEALED_REPOS; paths only, nothing is opened); deny Read of every existing HypaNet* sibling of the repo except the
author dir itself (an exact rule or a glob that matches it); allow the five author tools and Write/Edit in work/;
and never allow anything outside the author dir except the session's own Claude Code project directory.
"""

from __future__ import annotations

import fnmatch
import json
import math
import os
import re
import sys
from pathlib import Path

SLOP = Path(__file__).resolve().parents[1]
REPO = SLOP.parent
PARENT = REPO.parent
HOME = Path.home()
PRIVATE_NOTES = HOME / "Development" / "hypabolic" / "slop-private"
TOOLS = ("tok", "lint", "score", "inspect", "python3")
SMOKE_BUDGET_USD = 10.0
MIN_BUDGET_USD = 1.0


def parse_version(text: str) -> tuple[int, ...] | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(x) for x in m.groups()) if m else None


def version_ok(text: str, minimum: str) -> tuple[bool, str]:
    have, want = parse_version(text), parse_version(minimum)
    if have is None:
        return False, f"could not read a version from {text!r}"
    if have < want:
        have_s = ".".join(map(str, have))
        return False, f"Claude Code {have_s} is older than {minimum} (the first with claude-fable-5-2)"
    return True, f"Claude Code {'.'.join(map(str, have))} >= {minimum}"


def project_name(path: Path | str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", str(path))


def sealed_repo_roots() -> list[Path]:
    sys.path.insert(0, str(SLOP / "scripts"))
    import build_sets  # noqa: PLC0415  (paths only; build_sets opens nothing at import)

    return [root for _, root in build_sets.SEALED_REPOS]


def _read_globs(rules: list[str]) -> list[str]:
    """The absolute path globs of the Read(...) rules: Claude Code writes an absolute path as //abs/path."""
    out = []
    for r in rules:
        m = re.fullmatch(r"Read\((//.+)\)", r)
        if m:
            out.append(m.group(1)[1:])
    return out


def denied(path: Path, read_globs: list[str]) -> bool:
    """True if some Read deny glob covers everything under `path` (`<glob>/**` matching the path itself)."""
    p = str(path)
    for g in read_globs:
        if g.endswith("/**") and (fnmatch.fnmatchcase(p, g[:-3]) or fnmatch.fnmatchcase(p + "/x", g)):
            return True
    return False


def check_settings(settings_path: Path, author: Path, parent: Path = PARENT, repo: Path = REPO,
                   home: Path = HOME) -> list[str]:
    problems = []
    try:
        s = json.loads(settings_path.read_text())
    except (OSError, ValueError) as e:
        return [f"{settings_path} is missing or not valid JSON ({e})"]
    perms = s.get("permissions") if isinstance(s.get("permissions"), dict) else {}
    if perms.get("defaultMode") != "dontAsk":
        problems.append("permissions.defaultMode is not dontAsk")
    if s.get("autoMemoryEnabled") is not False:
        problems.append("autoMemoryEnabled is not false")
    if s.get("switchModelsOnFlag") is not False:
        problems.append("switchModelsOnFlag is not false (refusal fallback to another model must be off)")
    deny = [r for r in perms.get("deny") or [] if isinstance(r, str)]
    allow = [r for r in perms.get("allow") or [] if isinstance(r, str)]
    globs = _read_globs(deny)
    uid = os.getuid()
    must = [(repo, "the repo"), (Path(f"/tmp/claude-{uid}"), "/tmp/claude-<uid>"),
            (Path(f"/private/tmp/claude-{uid}"), "/private/tmp/claude-<uid>"),
            (home / "Development" / "hypabolic" / "slop-private", "the private note directory"),
            (home / ".codex", "~/.codex")]
    must += [(root, f"sealed repo {root.name}") for root in sealed_repo_roots()]
    for path, what in must:
        if not denied(path, globs):
            problems.append(f"settings do not deny Read of {what} ({path})")
    if f"Read(/{home / '.claude.json'})" not in deny:
        problems.append(f"settings do not deny Read of ~/.claude.json ({home / '.claude.json'})")
    author = author.resolve() if author.exists() else Path(os.path.abspath(author))
    if denied(author, globs):
        problems.append(f"settings deny Read of the author dir itself ({author}); regenerate them for this dir")
    for d in sorted(parent.glob("HypaNet*")):
        if d.resolve() in (repo.resolve(), author):
            continue
        if not denied(d, globs):
            problems.append(f"settings do not deny Read of the sibling {d} (extend kit/make_settings.py first)")
    if not denied(home / ".claude" / "projects" / project_name(repo), globs):
        problems.append("settings do not deny Read of the repo's Claude Code project directory")
    for t in TOOLS:
        if f"Bash(./{t}:*)" not in allow:
            problems.append(f"settings do not allow Bash(./{t}:*)")
    for r in ("Write(./work/**)", "Edit(./work/**)"):
        if r not in allow:
            problems.append(f"settings do not allow {r}")
    own = f"Read(/{home / '.claude' / 'projects' / project_name(author)}/**)"
    for r in allow:
        m = re.fullmatch(r"(\w+)\((.*)\)", r)
        arg = m.group(2) if m else r
        if arg.startswith("//") and r != own:
            problems.append(f"settings allow an absolute path other than the session's own project dir: {r}")
        if arg.startswith("../") or "/../" in arg:
            problems.append(f"settings allow a path outside the author dir: {r}")
    return problems


def budget_flag(mode: str, remaining: float) -> float:
    """The --max-budget-usd flag: the remaining cap (start, continue), at most US$10 for the smoke launch; rounded
    down to the cent."""
    flag = min(SMOKE_BUDGET_USD, remaining) if mode == "smoke" else remaining
    return math.floor(round(flag * 100, 6)) / 100


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    cmd = argv[1]
    if cmd == "version" and len(argv) == 4:
        ok, msg = version_ok(argv[2], argv[3])
        print(msg, file=sys.stderr if not ok else sys.stdout)
        return 0 if ok else 1
    if cmd == "settings" and len(argv) == 4:
        problems = check_settings(Path(argv[2]), Path(argv[3]))
        for p in problems:
            print(f"PROBLEM: {p}", file=sys.stderr)
        return 1 if problems else 0
    if cmd == "budget" and len(argv) == 4:
        try:
            remaining = float(argv[3])
        except ValueError:
            print(f"not a number: {argv[3]!r}", file=sys.stderr)
            return 1
        flag = budget_flag(argv[2], remaining)
        if flag < MIN_BUDGET_USD:
            print(f"less than US${MIN_BUDGET_USD:.0f} of the cap remains (flag would be {flag:.2f})", file=sys.stderr)
            return 1
        print(f"{flag:.2f}")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
