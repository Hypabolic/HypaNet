#!/bin/sh
# Launch (or resume) one v0.5 author run: a tool-free conversation with claude-fable-5-1 (Exp-5 premise).
#
#   scripts/v05_launch.sh r1                  # the one run (Matthew, 2026-11-06: one run, as in Exp 5) (refuses if the run has a transcript or a run record)
#   scripts/v05_launch.sh r1 --resume         # continue an interrupted run at its next turn
#
# Turn 001 sends experiments/v0_5/PROMPT.txt (checked against PROMPT.sha256) in a new session; while the last
# reply does not end with the line "END OF WEIGHTS", the same session is resumed with exactly "Continue."
# (40 continuation turns at most; a technical failure of a turn is retried at most 3 times with the same
# message after a wait of 2, 10 and 30 minutes, recorded; a rate or usage limit stops the run resumably; a
# conversation that no longer fits the context window stops it for good as context_limit). Each turn: claude -p --model claude-fable-5-1 --effort max (no --fallback-model;
# CLAUDE_CODE_NO_MODEL_FALLBACK=1) --tools "" --system-prompt "Answer the user's request." --strict-mcp-config
# --safe-mode --disable-slash-commands --setting-sources project --settings <auto-memory and hooks off>
# --permission-mode dontAsk --permission-prompts none --max-budget-usd <min(60, remaining cap)>
# --output-format stream-json --verbose, from a fresh empty directory in the system temp root, with the
# environment scrubbed of any parent Claude Code session (agent_loop_lab.v05.session has the exact command and
# environment, and what Claude Code still adds on its own). Transcripts: experiments/v0_5/logs/RUN.tNNN.*
# (never overwritten); replies: experiments/v0_5/runs/RUN/replies/NNN.md; record: runs/RUN/run.json.
#
# Refuses (before every turn): the US$500 cap summed over all v0.5 launches (scripts/v05_cost.py, session-id
# delta), harness scratch pytest-of-$USER in the system temp roots, PROMPT.txt not matching PROMPT.sha256, a
# non-empty working directory or a CLAUDE.md on its path, a run locked by a live process.
# When the run stops: scripts/v05_audit_session.py RUN, and scripts/v05_apply_reply.py on the run dir only for a
# final status (finished, turn_cap, context_limit, cost_cap); a resumable stop writes no artefact.
# caffeinate keeps the Mac awake while it runs.
set -eu
RUN="${1:?usage: v05_launch.sh r1 [--resume]}"
REPO="$(cd "$(dirname "$0")/.." && pwd -P)"
die() { echo "refusing: $*" >&2; exit 2; }
case "$RUN" in r1) ;; *) die "RUN must be r1 (v0.5 has one run)";; esac
RESUME=""
case "${2:-}" in
  "") ;;
  --resume) RESUME="--resume";;
  *) die "unknown option ${2}";;
esac
LOGDIR="$REPO/experiments/v0_5/logs"
if [ -z "$RESUME" ] && ls "$LOGDIR/$RUN".t*.stream.* > /dev/null 2>&1; then
  die "$RUN already has transcripts in $LOGDIR (use --resume to continue it)"
fi
[ -f "$REPO/experiments/v0_5/PROMPT.txt" ] || die "experiments/v0_5/PROMPT.txt missing"
[ -f "$REPO/experiments/v0_5/PROMPT.sha256" ] || die "experiments/v0_5/PROMPT.sha256 missing"
WANT="$(head -n 1 "$REPO/experiments/v0_5/PROMPT.sha256" | cut -d' ' -f1)"
GOT="$(shasum -a 256 "$REPO/experiments/v0_5/PROMPT.txt" | cut -d' ' -f1)"
[ "$WANT" = "$GOT" ] || die "PROMPT.txt sha256 $GOT does not match PROMPT.sha256 $WANT"
(cd "$REPO" && uv run python scripts/v05_cost.py --check > /dev/null) || die "v0.5 cost cap reached or unresolved (scripts/v05_cost.py)"
# Harness scratch (pytest's pytest-of-$USER) must not exist while a session runs (checked again before each turn).
USER_NAME="${USER:-$(id -un)}"
SYS_TMP="$(cd "$REPO" && uv run python -c 'import os, tempfile; print(os.path.realpath(tempfile.gettempdir()))')"
USER_TMP="$(getconf DARWIN_USER_TEMP_DIR 2>/dev/null || echo /tmp)"
for t in "$SYS_TMP" "${TMPDIR:-/tmp}" "$USER_TMP" /tmp /private/tmp; do
  [ -e "$t/pytest-of-$USER_NAME" ] && die "harness scratch $t/pytest-of-$USER_NAME exists; remove it (rm -rf \"$t/pytest-of-$USER_NAME\") and do not run the test suite while the author session runs"
done
command -v claude > /dev/null || die "claude not on PATH"
cd "$REPO"
CAFF=""
command -v caffeinate > /dev/null && CAFF="caffeinate -i"
# shellcheck disable=SC2086
exec $CAFF uv run python scripts/v05_run.py "$RUN" $RESUME
