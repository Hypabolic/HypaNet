# v0.4 author-kit tests (run on the prepared author dir before the WSP-A launch)

These are kit tests, not WSP-A attempts. They run in the prepared `../HypaNet-v04-author`, after `scripts/v04_prepare_author_dir.sh` and before `scripts/v04_launch.sh wsp-a`, with the frozen kit. One command runs them all:

```sh
scripts/v04_kit_tests.sh            # [AUTHOR_DIR]; V04_DEVICE=mps|cpu; V04_KIT_LOGDIR (default experiments/v0_4/kit_tests/logs)
```

- The kit logs go to their own directory: transcripts, the model server's ledger, kernel log and journal, the rendered settings and the build archive. The real session's meter, journal and ledger in `experiments/v0_4/logs/` are never touched.
- Everything the tests write under `output/wsp-a/` is deleted at the end; `output/wsp-a/scratch/` is left empty.
- The script exits 0 only if every step passes, and writes `KIT_RESULT.json`.

## 1. Before any session

Run by the launcher on every launch, kit or real:

- **Tools.** `hypaqwen.wsp.install verify` checks that `tools/` holds exactly `hn_kernel/` (client, fwd, protocol, run_py, wsp), `hn_guard/` and `hn_hook/hook_bash.py`. Every file and directory must be read-only and match the harness sources by sha256.
  - `./py` and `./fwd` must equal the kit copies.
  - The author dir, `packet/`, `.venv/` and `output/` must not be writable; `output/wsp-a/` must be.
- **Settings.** The phase template (`author_kit/settings_wsp_a.json`) is rendered for this launch into `LOGDIR/NAME.settings.json`. Its `@DENY_ROOTS@` entry expands to Read/Glob/Grep/Edit/Write deny rules for:
  - the repo;
  - every sibling of every ancestor of the author dir, up to `~`;
  - `~/.claude`, `~/.cache`, `~/.config`, `~/.local` and `~/Library`;
  - `/tmp`, `/private/tmp`, `/var/folders`, `/private/var/folders` and the system temp dir.

  The launcher also checks that the rendered file holds the required allow and deny rules (`agent_loop_lab.v04.kit.required_rules`).
- **Hook self-test.** The exact hook command from the rendered settings runs under `sh -c` with `CLAUDE_PROJECT_DIR` set, through the author venv's Python and the installed `tools/hn_hook/hook_bash.py`, on 30 cases:
  - It allows `./py` scripts, `./fwd` requests, `-m py_compile`, Write/Edit of the deliverables and reads inside the dir.
  - It refuses `;`, `&&`, a pipe, redirection, a newline, a trailing newline, `$(...)`, backticks, `./py -c`, other programs, scripts outside `output/wsp-a/`, `..` and globs.
  - It refuses writes to `packet/`, `tools/`, `./py` and `output/wsp-a/fwd/`, and reads of siblings and of `~/.claude`.
  - It refuses other tools, the wrong cwd, garbage stdin and empty stdin.
- **Harness scratch.** No `pytest-of-$USER` may exist in the temp roots, including the real per-user temp dir (`getconf DARWIN_USER_TEMP_DIR`). A kit launch only warns.

## 2. One throwaway session through the real launcher

`scripts/v04_launch.sh wsp-a` runs with `V04_KIT=1`. It uses the real code path: the model server in the background, its ready line, `caffeinate`, the scrubbed environment and the same claude flags. The kit-specific settings are:

- `claude-haiku-5-5` at low effort, at most US$1;
- a 2,000-token phase budget and a 1,500-token request cap;
- the kit prompt from `agent_loop_lab.v04.kit.kit_prompt`.

`agent_loop_lab.v04.kit.check_kit_session` then checks the transcript mechanically.

**Init event:**
- The tool list is exactly `Bash, Edit, Glob, Grep, Read, Write`, with no `Agent`, `ListAgents`, `SendMessage`, `WebFetch`, `Skill` or MCP tool.
- There is no MCP server.
- The permission mode is `dontAsk`, and the cwd is the author dir.

**Allowed, with the expected output:**
- reading `packet/SPEC.md`;
- writing `output/wsp-a/scratch/kt_note.txt`;
- `./py output/wsp-a/scratch/kt_wsp.py`: `from wsp import ...` against the real server, giving `NAMES`, `LABEL_IDS`, the edit hash, the budget, `tok` and a `load_original`;
- `./py .../kt_guard.py`: every action inside it is refused by the KERNEL_CLIENT guard (it exits 3). The actions are:
  - reading `/etc/hosts`;
  - listing `..`;
  - writing at the author-dir root;
  - connecting to `.kernel/fwd.sock`;
  - running a subprocess;
  - importing `torch.optim`;
  - a backward pass;
  - calling `random`;
- `./fwd .../kt_req.json`: one metered forward pass, call 1;
- `./fwd .../kt_req_big.json`: refused by the meter as `BudgetRefused`, because it is over the request cap and nothing is charged;
- `./py .../kt_edit.py`: one `save_edit`, journalled.

**Refused:**
- by the permissions or the hook:
  - reading the repo's `pyproject.toml` and `~/.claude/settings.json`;
  - `Glob ../*` and `Grep ..`;
  - writing `packet/SPEC.md`, `tools/hn_kernel/kt.py`, `./py` and `output/wsp-a/fwd/kt.json`;
  - `./py ... ; ls ..`, `./py ... && cat ...`, `./py ... | cat` and `./py ... > file`;
  - `ls ..`, `./py -c print(1)`, `./py $(...)` and `./fwd ../kt.json`;
- by being absent: no tool outside the kit is called.

The session audit (`scripts/v04_audit_session.py`, run by the launcher) must report:
- no executed violation;
- the static scan's findings on `kt_guard.py` (its `subprocess` and `socket` imports), as a positive control;
- ledger statuses of one `ok` and one `refused`;
- one journalled `save_edit` in the kernel log;
- the permission denials recorded.

## 3. A fake build

The kit's `build_edit.py` adds 1/64 to one entry of the layer-0 input norm, the same edit `kt_edit.py` saved. `EDIT.md` and `ITERATIONS.md` are present. `scripts/v04_run_build.py wsp-a --logs <kit logs> --archive <kit logs>/archive` must:

- build it under BUILD_EDIT;
- rebuild it byte-identically in a second fresh interpreter;
- match the kit journal's edit state tensor for tensor.

The record must show `journal_checked: true`.

## 4. After

- `output/wsp-a/` holds only an empty `scratch/` again.
- The server's sockets are gone.
- `tools/` still verifies.

The kit sessions' transcripts also land in `~/.claude/projects/<author-dir slug>/`. The author cannot read them, because `~/.claude` is denied by the settings, by the hook and by `--restricted`, and auto-memory is off.

## Facts checked with real `claude -p` sessions (Claude Code 2.1.311, 5 Nov 2026)

These were measured with throwaway `claude-haiku-5-5` sessions in the session scratchpad, at about US$0.005 in total. They are the reasons for the kit's design.

- **Hook input.** It is JSON on stdin with `session_id`, `transcript_path`, `cwd`, `prompt_id`, `permission_mode`, `effort`, `hook_event_name` (`PreToolUse`), `tool_name`, `tool_input` (Bash: `command`, `description`; Write: `file_path`, `content`; Read: `file_path`) and `tool_use_id`.
  - `CLAUDE_PROJECT_DIR` is set to the session cwd, and the hook runs in that cwd.
- **A blocking hook shows the session its command line.** The session sees `PreToolUse:Bash hook error: [<the hook command>]: <stderr>`. So the hook command names no harness path: it is `"$CLAUDE_PROJECT_DIR"/.venv/bin/python -I -B "$CLAUDE_PROJECT_DIR"/tools/hn_hook/hook_bash.py wsp-a`, shown unexpanded. The hook is installed read-only in the author dir rather than run from the repo.
- **A hook that exits with any status other than 0 or 2 fails open.** For example, an exit of 1 let the command run. So `hook_bash.py` turns every internal error into exit 2, and the launcher self-tests it before each session.
- **`--tools Read,Glob,Grep,Write,Edit,Bash`** gives exactly those six tools in the init event.
- **`--restricted` keeps `--settings` hooks working.** A Read outside the cwd is refused under `dontAsk`.
- **`total_cost_usd` is cumulative per session across resumed launches.** Three launches of one session reported 0.00119, 0.00134 and 0.00144 for one-word replies, which is why `scripts/v04_cost.py` uses deltas. `--max-budget-usd 0.0013` did not stop a resumed launch whose session total was already 0.00134, so the flag appears to bound the launch's own spend. The launcher passes the phase's remaining cap.
- **Delta accounting on v0.3's own transcripts** gives Phase A US$42.73 and Phase B US$21.89, the corrected figures of plan revision 11.
- **In `-p` mode Claude Code silently ignores a settings file that fails validation.** So the kit session checks that the hook and the rules really apply, not only that the file exists.
