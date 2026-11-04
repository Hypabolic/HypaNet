# M2 launch kit: Phase A weight authoring

Weight author: a fresh **Claude Fable 5.1** session (`claude-fable-5-1`), with **Opus 5.5** (`claude-opus-5-5`) as the automatic fallback. It runs headless at effort `max` in an isolated directory **outside** the repo (default `../HypaNet-m2-author`). It sees only the authoring packet, its own output directory, `tools/shape_check.py` and the `./py` wrapper (spec §8, `experiments/roles.yaml`).

## Precondition

M1 data hashes (`datasets/v0_2/MANIFEST.json`, splits, and the OOD set) are committed **before** step 1 (spec §7.3).

## Run

```sh
scripts/m2_prepare_author_dir.sh                  # fresh packet, verified against the registry; private venv
git add experiments/m2/packet_handed_over experiments/m2/HANDOVER.json && git commit   # pin what the author received

scripts/m2_launch.sh phase-a-strict               # Fable writes reference_policy.py, build_weights.py, CIRCUITS.md
uv run python scripts/run_build.py phase-a-strict \
  --author-dir ../HypaNet-m2-author/output/phase-a-strict \
  --packet-dir ../HypaNet-m2-author/packet --out-dir weight_program/phase-a-strict
git add weight_program/phase-a-strict experiments/m2/logs experiments/m2/SESSION_ID && git commit   # archive BEFORE any behavioural test

scripts/m2_launch.sh phase-a-selfcheck            # resumes the same session (spec §9.3)
uv run python scripts/run_build.py phase-a-selfcheck \
  --author-dir ../HypaNet-m2-author/output/phase-a-selfcheck \
  --packet-dir ../HypaNet-m2-author/packet --out-dir weight_program/phase-a-selfcheck
git add weight_program/phase-a-selfcheck experiments/m2/logs && git commit
```

Override the defaults with `M2_MODEL`, `M2_FALLBACK_MODEL` and `M2_EFFORT`. Each launch refuses to overwrite an existing transcript or session.

## Rejected submissions (pre-registered, 2026-11-04)

Spec §9.2 archives the first **valid** artefact, and spec §8 says compile and shape checks are not behavioural feedback. If `run_build.py` rejects a submission with a `BuildError`:

1. Save the runner's error text, and nothing else, to a file.
2. Run `scripts/m2_launch.sh resubmit <phase> <N> <error-file>`. This resumes the same session with only that text.
3. Rebuild.

The rules:
- At most **3** resubmissions per phase.
- Every attempt stays in the run directory's `attempts/` log, and every resubmission transcript stays in `logs/<phase>.rN.stream.jsonl`.
- The archive commit message records the number of resubmissions.
- A kit failure is not a Phase A attempt. That covers `claude` exiting on a flag or auth error before the model does any work. In that case, delete `SESSION_ID` and the log by hand and relaunch.
- Read the audit's `models` field after every launch. `--fallback-model` can switch to Opus mid-session, and the archive must record which model actually wrote the weights.

## Technical continuations

If a response ends on an API or limit error before anything is submitted, the session is resumed with the fixed, feedback-free `prompt_continue.md`, via `scripts/m2_launch.sh continue <phase> <K>`. These continuations are logged as `<phase>.cK` and are not Phase A attempts. `CLAUDE_CODE_MAX_OUTPUT_TOKENS` defaults to 128000; set `M2_MAX_OUTPUT_TOKENS` to change it.

**Incident log**
- 2026-11-04, phase-a-strict: the first response hit the 64,000 output-token cap after 56.8 min (12 turns, $13.95, claude-fable-5-1), so no files were written. The audit flagged one executed `ls -la` of the author's own directory. Claude Code auto-allows read-only commands inside the working directory; this is harmless, but it is recorded. Resumed as phase-a-strict.c1.

## Isolation controls

| Control | Mechanism |
|---|---|
| No repo, harness-scratch or Claude-project reads | `deny` rules in `settings_phase_*.json` for Read, Grep, Glob, Edit and Write |
| No web, subagents, workflows, skills or MCP | `--disallowedTools …`, `--strict-mcp-config` |
| No user-level settings or allow-lists | `--setting-sources project` (the author dir has none) plus `--settings <phase file>` |
| Anything not explicitly allowed is refused | `--permission-mode dontAsk` |
| A-strict: zero execution | Bash allowed only for `./py -m py_compile …` and `./py tools/shape_check.py output/phase-a-strict` (no forward pass) |
| A-selfcheck: own inputs only | Bash allowed only for `./py …`; the archived A-strict directory is write-denied |
| Audit trail | stream-json transcript in `logs/`, checked by `scripts/m2_audit_session.py`, which separates refused attempts from executed violations |

These controls were tested with throwaway Fable sessions on 2026-11-04:
- reading the repo was refused;
- `ls ..` was refused;
- compound shell commands were refused;
- the two allowed A-strict commands ran;
- nothing was written into the packet.

The prompts tell the author to run each command on its own, because Fable otherwise appends `; echo` and the command is refused.
