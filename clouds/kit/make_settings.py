"""Write kit/settings.json, the Claude Code settings of the author session (SPEC section 7).

    uv run python kit/make_settings.py [AUTHOR_DIR]           # write kit/settings.json
    uv run python kit/make_settings.py --check [AUTHOR_DIR]   # exit 1 if kit/settings.json misses a rule it would write now

Allowed: Read / Glob / Grep in the author dir; Write / Edit in work/; Bash only ./render, ./inspect, ./python3; Read
of the session's own Claude Code project directory (~/.claude/projects/<author dir as a name>/), because Claude Code
saves a large tool output there and tells the model to Read it (v0.5 Phase C lost such a read to a blanket ~/.claude
deny). Denied: web tools; every access to the repo, every HypaNet* sibling author dir and the Claude scratch / temp
roots; every other entry of ~/.claude and of ~/.claude/projects (enumerated now: deny rules beat allow rules, so the
blanket ~/.claude deny cannot be used); writes to the kit's read-only files; the usual shell commands by name
(Bash(ls:*), Bash(cat:*), ... BASH_DENY), because Claude Code auto-approves some read-only ones. Rerun it (scripts/prepare_author_dir.sh
does) when ~/.claude gains entries.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

CLOUDS = Path(__file__).resolve().parents[1]
REPO = CLOUDS.parent
PARENT = REPO.parent
HOME = Path.home()


BASH_DENY = ("ls", "cat", "head", "tail", "less", "more", "find", "grep", "rg", "sed", "awk", "wc", "file", "stat",
             "du", "tree", "xxd", "od", "strings", "cd", "echo", "printf", "open", "uv", "git",
             "npx", "node", "curl", "env", "pwd", "sh", "bash", "zsh", "cp", "mv", "rm", "ln", "chmod", "touch",
             "mkdir", "tee", "xargs", "perl", "ruby", "osascript", "wget", "readlink", "realpath", "diff", "sort",
             "uniq", "cut", "tr", "basename", "dirname", "which", "type", "command", "export", "source", "test",
             "true", "false", "sleep", "date", "id", "whoami", "hostname", "uname", "ps", "kill", "lsof", "mdfind",
             "mdls", "plutil", "defaults", "security", "sqlite3", "jq", "base64", "shasum", "md5", "xattr")


def project_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", str(path))


def main(argv: list[str]) -> int:
    check = "--check" in argv
    argv = [x for x in argv if x != "--check"]
    author = Path(argv[1]) if len(argv) > 1 else PARENT / "HypaNet-clouds-author"
    own = project_name(author)
    claude = HOME / ".claude"
    a = lambda p: "/" + str(p)  # noqa: E731  (Claude Code: //abs/path is an absolute path)
    roots = [REPO, PARENT / "HypaNet-v0*", PARENT / "HypaNet-v1*", PARENT / "HypaNet-m*",
             Path("/private/tmp/claude-%d" % os.getuid()), Path("/tmp/claude-%d" % os.getuid()),
             Path("/private/var/folders"), Path("/var/folders")]
    deny = ["WebFetch", "WebSearch"]
    for r in roots:
        for t in ("Read", "Grep", "Glob", "Edit", "Write"):
            deny.append(f"{t}({a(r)}/**)")
    for e in sorted(os.listdir(claude)):
        if e == "projects":
            continue
        for t in ("Read", "Grep", "Glob", "Edit", "Write"):
            deny += [f"{t}({a(claude / e)})", f"{t}({a(claude / e)}/**)"]
    # project dirs that may appear later (sessions started in the repo, clouds/, other author dirs, scratch dirs)
    hyp = project_name(PARENT)
    future = [f"{hyp}-HypaNet", f"{hyp}-HypaNet-clouds", f"{hyp}-HypaNet-v*", f"{hyp}-HypaNet-m*", "-private-tmp-*",
              "-tmp-*", "-private-var-*"]
    for e in sorted(set(os.listdir(claude / "projects")) | set(future)):
        if e == own:
            continue
        for t in ("Read", "Grep", "Glob", "Edit", "Write"):
            deny.append(f"{t}({a(claude / 'projects' / e)}/**)")
    for t in ("Edit", "Write"):
        deny.append(f"{t}({a(claude / 'projects' / own)}/**)")
    # Claude Code approves some read-only shell commands without an allow rule (v0.5 Phase C ran `ls -la` and
    # `ls -laR` outside its dir that way); explicit denies take precedence. No python/python3 deny: in r1's first
    # launch (2026-11-09) Bash(python3:*) also denied every ./python3 command, so a deny can match a command's base
    # name, not only its prefix. Plain python3 is not auto-approved, so dontAsk denies it without a rule. Never add
    # a deny whose name is render, inspect or python3.
    deny += [f"Bash({c}:*)" for c in BASH_DENY]
    for p in ("./KIT.md", "./PROMPT.md", "./HANDOVER.json", "./render", "./inspect", "./python3"):
        deny += [f"Edit({p})", f"Write({p})"]
    for p in ("./references/**", "./tools/**", "./.venv/**", "./.claude/**"):
        deny += [f"Edit({p})", f"Write({p})"]
    settings = {
        "autoMemoryEnabled": False,
        "disableBypassPermissionsMode": "disable",
        "permissions": {
            "defaultMode": "dontAsk",
            "allow": ["Read(./**)", "Glob(./**)", "Grep(./**)", "Write(./work/**)", "Edit(./work/**)",
                      "Bash(./render:*)", "Bash(./inspect:*)", "Bash(./python3:*)",
                      f"Read({a(claude / 'projects' / own)}/**)"],
            "deny": deny,
        },
    }
    out = CLOUDS / "kit" / "settings.json"
    if check:
        try:
            cur = json.loads(out.read_text())["permissions"]
        except (OSError, ValueError, KeyError):
            print(f"{out} missing or unreadable")
            return 1
        missing = [r for r in deny if r not in cur["deny"]] + [r for r in settings["permissions"]["allow"]
                                                                if r not in cur["allow"]]
        extra = [r for r in cur["allow"] if r not in settings["permissions"]["allow"]]
        for r in missing:
            print(f"MISSING: {r}")
        for r in extra:
            print(f"UNEXPECTED ALLOW: {r}")
        if missing or extra:
            print("re-run: uv run python kit/make_settings.py")
            return 1
        print(f"settings cover every current entry of {claude} ({len(cur['deny'])} deny rules)")
        return 0
    out.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"settings -> {out} ({len(deny)} deny rules; own project dir {own})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
