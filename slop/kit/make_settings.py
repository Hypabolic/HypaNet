"""Write kit/settings.json, the Claude Code settings of the slop author session (SPEC §3, §13).

    uv run python kit/make_settings.py [AUTHOR_DIR] [--out FILE]            # write the run's settings (or FILE)
    uv run python kit/make_settings.py --check [AUTHOR_DIR] [--out FILE]    # exit 1 if it misses a rule it would write now

The run's settings file is kit/settings.json for an r1 author dir (HypaNet-slop-author*), kit/settings.r2.json for an
r2 one (HypaNet-slop-author-r2*) and kit/settings.smoke.json for the smoke dir (HypaNet-slop-smoke*), so preparing one
run never rewrites another's. An r2 session is denied the r1 author dir like every other HypaNet* sibling, and r2's
resources/ is read-only like examples/.

From clouds/kit/make_settings.py. Allowed: Read / Glob / Grep in the author dir; Write / Edit in work/; Bash only
./tok, ./lint, ./score, ./inspect, ./python3; Read of the session's own Claude Code project directory
(~/.claude/projects/<author dir as a name>/), where Claude Code saves a large tool output and tells the model to Read
it. Denied (Read, Glob, Grep, Edit, Write): the HypaNet repo; every other HypaNet* sibling of the author dir, the smoke
dir ../HypaNet-slop-smoke included; the Claude scratch and temp roots (/tmp/claude-<uid>, /private/var/folders);
~/Development/hypabolic/slop-private (the unredacted corpus note, SPEC §9); every repo that supplies sealed_real or
comments_real (scripts/build_sets.py SEALED_REPOS), the reference and calibration repos (REFERENCE_REPOS,
CALIBRATION_REPOS: their copy reaches the author only through examples/reference/) and any public_clean source tree
(PUBLIC_CLEAN_ROOTS); every entry of ~/.claude other than projects/, and every entry of ~/.claude/projects other than
the session's own (enumerated now: deny rules beat allow rules, so a blanket ~/.claude deny cannot be used);
~/.claude.json and ~/.codex (Codex labeller sessions keep their transcripts there); writes
to the kit's read-only files; web tools; the usual shell commands by name (BASH_DENY). Also: autoMemoryEnabled false,
switchModelsOnFlag false (no content-based model switch, research/fable-5-2.md §5), showThinkingSummaries true
(SPEC §12.7: tested in the smoke launch), bypass-permissions mode disabled. Rerun it (scripts/prepare_author_dir.sh
does, and the launcher runs --check) when ~/.claude gains entries.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
from pathlib import Path

SLOP = Path(__file__).resolve().parents[1]
REPO = SLOP.parent
PARENT = REPO.parent
HOME = Path.home()
sys.path.insert(0, str(SLOP))
from scripts.build_sets import CALIBRATION_REPOS, REFERENCE_REPOS, SEALED_REPOS  # noqa: E402  (lists only)

DEFAULT_AUTHOR = PARENT / "HypaNet-slop-author"
R2_AUTHOR = PARENT / "HypaNet-slop-author-r2"
SMOKE = PARENT / "HypaNet-slop-smoke"
CODEX_HOME = HOME / ".codex"
CLAUDE_JSON = HOME / ".claude.json"
PRIVATE_NOTES = HOME / "Development" / "hypabolic" / "slop-private"
# public_clean (SPEC §9: AOSP strings.xml, Firefox Fluent) has no local source tree yet; add its roots here when
# scripts/public_to_screen.py gets one.
PUBLIC_CLEAN_ROOTS: list[Path] = []
TOOLS = ("tok", "lint", "score", "inspect", "python3")
READ_ONLY = ("./KIT.md", "./PROMPT.md", "./LABELLING.md", "./HANDOVER.json", *(f"./{t}" for t in TOOLS))
READ_ONLY_DIRS = ("./examples/**", "./kit/**", "./tools/**", "./.venv/**", "./.claude/**", "./resources/**")
FS_TOOLS = ("Read", "Grep", "Glob", "Edit", "Write")
# Never add a deny whose name is tok, lint, score, inspect or python3: in clouds r1's first launch Bash(python3:*)
# also denied every ./python3 command (a deny can match a command's base name).
BASH_DENY = ("ls", "cat", "head", "tail", "less", "more", "find", "grep", "rg", "sed", "awk", "wc", "file", "stat",
             "du", "tree", "xxd", "od", "strings", "cd", "echo", "printf", "open", "uv", "git",
             "npx", "node", "curl", "env", "pwd", "sh", "bash", "zsh", "cp", "mv", "rm", "ln", "chmod", "touch",
             "mkdir", "tee", "xargs", "perl", "ruby", "osascript", "wget", "readlink", "realpath", "diff", "sort",
             "uniq", "cut", "tr", "basename", "dirname", "which", "type", "command", "export", "source", "test",
             "true", "false", "sleep", "date", "id", "whoami", "hostname", "uname", "ps", "kill", "lsof", "mdfind",
             "mdls", "plutil", "defaults", "security", "sqlite3", "jq", "base64", "shasum", "md5", "xattr", "npm",
             "claude", "codex")
SCALARS = {"autoMemoryEnabled": False, "switchModelsOnFlag": False, "showThinkingSummaries": True,
           "disableBypassPermissionsMode": "disable"}
assert not set(BASH_DENY) & set(TOOLS)


def project_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", str(path))


def repo_roots() -> list[Path]:
    """Sealed, reference and calibration repos: each recorded root and the top-level repo directory above it."""
    dev = HOME / "Development"
    out: list[Path] = []
    roots = [r for _, r in SEALED_REPOS] + [r.root for r in REFERENCE_REPOS + CALIBRATION_REPOS] + PUBLIC_CLEAN_ROOTS
    for r in roots:
        out.append(r)
        try:
            rel = r.relative_to(dev)
        except ValueError:
            continue
        if len(rel.parts) > 2:  # ~/Development/<group>/<repo>/<subdir>: deny the whole repo too
            out.append(dev / rel.parts[0] / rel.parts[1])
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def build(author: Path) -> dict:
    own = project_name(author)
    claude = HOME / ".claude"
    a = lambda p: "/" + str(p)  # noqa: E731  (Claude Code: //abs/path is an absolute path)
    uid = os.getuid()
    siblings = sorted(PARENT / e for e in os.listdir(PARENT) if e.startswith("HypaNet") and PARENT / e != author)
    roots = [REPO, *siblings, SMOKE, DEFAULT_AUTHOR, R2_AUTHOR, PARENT / "HypaNet-slop-author-r2*",
             PARENT / "HypaNet-slop-author*", PARENT / "HypaNet-clouds*", PARENT / "HypaNet-v0*", PARENT / "HypaNet-v1*",
             PARENT / "HypaNet-m*", PARENT / "HypaNet-slop-smoke*", Path(f"/private/tmp/claude-{uid}"),
             Path(f"/tmp/claude-{uid}"), Path("/private/var/folders"), Path("/var/folders"), PRIVATE_NOTES,
             CODEX_HOME, *repo_roots()]
    deny = ["WebFetch", "WebSearch"]
    seen = set()
    for r in roots:
        # the smoke dir is denied to r1, but a smoke session's own dir must stay readable (deny beats allow)
        if r in seen or r == author or fnmatch.fnmatchcase(str(author), str(r)):
            continue
        seen.add(r)
        for t in FS_TOOLS:
            deny.append(f"{t}({a(r)}/**)")
    for t in FS_TOOLS:
        deny.append(f"{t}({a(CLAUDE_JSON)})")
    for e in sorted(os.listdir(claude)):
        if e == "projects":
            continue
        for t in FS_TOOLS:
            deny += [f"{t}({a(claude / e)})", f"{t}({a(claude / e)}/**)"]
    # project dirs that may appear later (sessions in the repo, other author dirs, the smoke dir, scratch dirs)
    hyp = project_name(PARENT)
    future = [f"{hyp}-HypaNet", f"{hyp}-HypaNet-clouds*", f"{hyp}-HypaNet-v*", f"{hyp}-HypaNet-m*",
              f"{hyp}-HypaNet-slop-smoke*", f"{hyp}-HypaNet-slop-author", f"{hyp}-HypaNet-slop-author-r2*",
              "-private-tmp-*", "-tmp-*", "-private-var-*"]
    for e in sorted(set(os.listdir(claude / "projects")) | set(future)):
        if e == own or fnmatch.fnmatchcase(own, e):
            continue
        for t in FS_TOOLS:
            deny.append(f"{t}({a(claude / 'projects' / e)}/**)")
    for t in ("Edit", "Write"):
        deny.append(f"{t}({a(claude / 'projects' / own)}/**)")
    deny += [f"Bash({c}:*)" for c in BASH_DENY]
    for p in READ_ONLY + READ_ONLY_DIRS:
        deny += [f"Edit({p})", f"Write({p})"]
    allow = ["Read(./**)", "Glob(./**)", "Grep(./**)", "Write(./work/**)", "Edit(./work/**)",
             *(f"Bash(./{t}:*)" for t in TOOLS), f"Read({a(claude / 'projects' / own)}/**)"]
    return {**SCALARS, "permissions": {"defaultMode": "dontAsk", "allow": allow, "deny": deny}}


def default_out(author: Path) -> Path:
    """kit/settings.smoke.json for the smoke dir, kit/settings.r2.json for an r2 dir, kit/settings.json otherwise
    (r1)."""
    if author.name.startswith("HypaNet-slop-smoke"):
        return SLOP / "kit" / "settings.smoke.json"
    if author.name.startswith("HypaNet-slop-author-r2"):
        return SLOP / "kit" / "settings.r2.json"
    return SLOP / "kit" / "settings.json"


def main(argv: list[str]) -> int:
    check = "--check" in argv
    argv = [x for x in argv[1:] if x != "--check"]
    out = None
    if "--out" in argv:
        i = argv.index("--out")
        if i + 1 >= len(argv):
            print("--out needs a file")
            return 2
        out = Path(argv[i + 1])
        del argv[i: i + 2]
    if len(argv) > 1 or any(x.startswith("-") for x in argv):
        print(__doc__)
        return 2
    author = Path(os.path.abspath(argv[0])) if argv else DEFAULT_AUTHOR
    out = out if out is not None else default_out(author)
    settings = build(author)
    if check:
        try:
            cur = json.loads(out.read_text())
            perms = cur["permissions"]
        except (OSError, ValueError, KeyError):
            print(f"{out} missing or unreadable")
            return 1
        want = settings["permissions"]
        missing = [r for r in want["deny"] if r not in perms.get("deny", [])] + \
                  [r for r in want["allow"] if r not in perms.get("allow", [])]
        extra = [r for r in perms.get("allow", []) if r not in want["allow"]]
        bad = [f"{k} is {cur.get(k)!r}, want {v!r}" for k, v in SCALARS.items() if cur.get(k) != v]
        if perms.get("defaultMode") != "dontAsk":
            bad.append(f"defaultMode is {perms.get('defaultMode')!r}, want 'dontAsk'")
        for r in missing:
            print(f"MISSING: {r}")
        for r in extra:
            print(f"UNEXPECTED ALLOW: {r}")
        for r in bad:
            print(f"WRONG: {r}")
        if missing or extra or bad:
            print("re-run: uv run python kit/make_settings.py" + (f" {author}" if author != DEFAULT_AUTHOR else ""))
            return 1
        print(f"settings cover every current entry of {HOME / '.claude'} ({len(perms['deny'])} deny rules)")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"settings -> {out} ({len(settings['permissions']['deny'])} deny rules; own project dir "
          f"{project_name(author)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
